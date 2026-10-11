"""Audit raw AMASS files by verified format before safe parameter loading.

Only *_poses.npz files are read. This produces provenance and a deterministic
directory-subject split; it does not produce SMPL targets, images, or features.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import re
from typing import Any
from zipfile import BadZipFile, ZipFile, is_zipfile
from zlib import error as ZlibError

import numpy as np


SEED = 2026
TRAIN_FRACTION = 0.8
NUMERIC_KINDS = frozenset("fiu")
MAX_MEMBERS = 32
MAX_MEMBER_BYTES = 1024 * 1024 * 1024
MAX_UNCOMPRESSED_BYTES = 2 * 1024 * 1024 * 1024


class ArchiveRejected(ValueError):
    """An archive cannot be accepted under the safe raw-parameter contract."""


def source_fingerprint(path: Path) -> dict[str, Any]:
    """Hash and count every byte; samples alone never prove an all-zero file."""
    before = path.stat()
    digest = hashlib.sha256()
    scanned = nonzero = 0
    head = b""
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            if not head:
                head = chunk[:512]
            digest.update(chunk)
            scanned += len(chunk)
            nonzero += len(chunk) - chunk.count(b"\x00")
    after = path.stat()
    result = dict(source_sha256=digest.hexdigest(), source_bytes=before.st_size,
                  bytes_scanned=scanned, nonzero_bytes_count=nonzero,
                  full_scan_complete=scanned == before.st_size,
                  full_file_confirmed_all_zero=scanned > 0 and nonzero == 0,
                  first_64_bytes_hex=head[:64].hex(),
                  first_up_to_512_bytes_all_zero=bool(head) and not any(head))
    if (before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns
            or scanned != before.st_size):
        raise ArchiveRejected("source_changed_during_full_scan; repeat in a new output directory")
    return result


def classify_source(path: Path, fingerprint: dict[str, Any]) -> dict[str, Any]:
    """Recognize containers by bytes and structure, never by .npz extension."""
    if not fingerprint["source_bytes"]:
        return dict(format="empty_file", accepted_container=False)
    if fingerprint["full_file_confirmed_all_zero"]:
        return dict(format="full_file_all_zero", accepted_container=False,
                    evidence="full file SHA256 scan counted zero nonzero bytes; not a sampled inference")
    with path.open("rb") as stream:
        head = stream.read(512)
    # ZIP central directories can identify a valid archive with an arbitrary
    # leading prefix. np.load(path) rejects such a file even when every member
    # is readable; inspect the central directory without changing source bytes.
    if is_zipfile(path):
        with ZipFile(path) as archive:
            names = archive.namelist()
            required = {"poses.npy", "trans.npy", "betas.npy"}
            if len(names) != len(set(names)):
                return dict(format="zip_with_duplicate_members", accepted_container=False)
            if (not names or len(names) > MAX_MEMBERS
                    or any(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*\.npy", name) is None for name in names)):
                return dict(format="zip_unsafe_or_unsupported_member_names", accepted_container=False)
            if not required.issubset(names):
                return dict(format="zip_missing_raw_numpy_parameters", accepted_container=False,
                            missing_members=sorted(required - set(names)))
            if not {"mocap_framerate.npy", "fps.npy"}.intersection(names):
                return dict(format="zip_missing_numpy_fps", accepted_container=False)
            infos = archive.infolist()
            total = sum(item.file_size for item in infos)
            if total > MAX_UNCOMPRESSED_BYTES or any(not 6 <= item.file_size <= MAX_MEMBER_BYTES for item in infos):
                return dict(format="zip_uncompressed_size_limit_exceeded", accepted_container=False)
            if any(item.flag_bits & 1 or item.compress_type not in (0, 8) for item in infos):
                return dict(format="zip_encrypted_or_unsupported_compression", accepted_container=False)
            offsets = {item.filename: item.header_offset for item in infos}
            prefix = min(offsets.values())
            if prefix < 0 or any(offset >= fingerprint["source_bytes"] for offset in offsets.values()):
                return dict(format="zip_invalid_local_header_offsets", accepted_container=False)
            # Metadata is provisional until all member CRCs and NPY header
            # lengths pass in safe_parameter_archive below.
            return dict(format="numpy_npz_zip_with_prefix" if prefix else "numpy_npz_zip",
                        accepted_container=True, leading_prefix_bytes=prefix,
                        local_header_offsets=offsets, member_names=names,
                        declared_uncompressed_bytes=total,
                        first_up_to_512_bytes_all_zero=fingerprint["first_up_to_512_bytes_all_zero"],
                        all_members_crc_verified=False, all_npy_headers_verified=False,
                        source_repaired=False,
                        prefix_all_zero_not_inferred_from_header_sample=True,
                        evidence="valid ZIP central directory; raw parameter/fps names present; full member verification pending")
    if head.startswith(b"version https://git-lfs.github.com/spec/v1"):
        return dict(format="git_lfs_pointer_prefix", accepted_container=False,
                    evidence="pointer text prefix; actual array content is unavailable")
    if head.startswith(b"\x93NUMPY"):
        return dict(format="numpy_npy_not_npz_parameter_archive", accepted_container=False,
                    evidence="NPY magic; standalone/object parameter dictionaries are not supported")
    if head.startswith(b"PK\x03\x04") or head.startswith(b"PK\x05\x06") or head.startswith(b"PK\x07\x08"):
        return dict(format="invalid_zip_prefix", accepted_container=False)
    if fingerprint["first_up_to_512_bytes_all_zero"]:
        return dict(format="zero_header_with_nonzero_content", accepted_container=False,
                    evidence="head is zero, full scan found nonzero bytes; cannot infer a usable format")
    if head.startswith(b"\x80"):
        return dict(format="pickle_protocol_prefix_refused", accepted_container=False,
                    evidence="pickle-like prefix is not accepted or loaded")
    return dict(format="unrecognized_non_numpy_file", accepted_container=False,
                evidence="unsupported header; np.load error text alone does not establish pickle content")


class SafeParameterFields(dict):
    @property
    def files(self) -> list[str]:
        return list(self)


@contextmanager
def safe_parameter_archive(path: Path, format_report: dict[str, Any]):
    """Verify CRC for every ZIP member, then safely decode exact NPY arrays."""
    raw_members = {}
    details = {}
    with ZipFile(path) as archive:
        if archive.namelist() != format_report["member_names"]:
            raise ArchiveRejected("zip_directory_changed_between_inspection_and_read")
        infos = archive.infolist()
        if (len(infos) > MAX_MEMBERS or sum(item.file_size for item in infos) > MAX_UNCOMPRESSED_BYTES
                or any(not 6 <= item.file_size <= MAX_MEMBER_BYTES or item.flag_bits & 1
                       or item.compress_type not in (0, 8) for item in infos)
                or {item.filename: item.header_offset for item in infos} != format_report["local_header_offsets"]):
            raise ArchiveRejected("zip_metadata_changed_or_member_limits_exceeded")
        for item in infos:
            # A complete read reaches ZipExtFile's CRC check. Checking only the
            # NPY header or selected fields would miss corruption elsewhere.
            raw = archive.read(item.filename)
            if len(raw) != item.file_size:
                raise ArchiveRejected("zip_member_size_mismatch:" + item.filename)
            raw_members[item.filename] = raw
            details[item.filename] = dict(crc32=f"{item.CRC:08x}", crc_verified=True,
                                          uncompressed_bytes=len(raw),
                                          sha256=hashlib.sha256(raw).hexdigest(),
                                          local_header_offset=item.header_offset)
    format_report.update(all_members_crc_verified=True, members=details)
    arrays = SafeParameterFields()
    for name, raw in raw_members.items():
        stream = BytesIO(raw)
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream, max_header_size=10000)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream, max_header_size=10000)
        else:
            raise ArchiveRejected("unsupported_npy_header_version:" + name)
        key = name[:-4]
        if dtype.hasobject:
            raise ArchiveRejected(f"unsafe_object_dtype_refused:{key}; allow_pickle=False; cannot load safely")
        if (dtype.kind not in NUMERIC_KINDS | {"U", "S", "b"}
                or any(not isinstance(size, int) or size < 0 for size in shape)):
            raise ArchiveRejected("unsupported_npy_dtype_or_shape:" + name)
        payload_bytes = math.prod(shape) * dtype.itemsize
        if payload_bytes > MAX_MEMBER_BYTES or stream.tell() + payload_bytes != len(raw):
            raise ArchiveRejected("npy_header_shape_byte_length_mismatch:" + name)
        value = np.load(BytesIO(raw), allow_pickle=False)
        if not isinstance(value, np.ndarray) or value.shape != shape or value.dtype != dtype:
            raise ArchiveRejected("npy_decode_differs_from_validated_header:" + name)
        arrays[key] = value
        details[name].update(dtype=str(dtype), shape=list(shape), fortran_order=fortran,
                             header_and_payload_bytes_verified=True)
    format_report.update(all_npy_headers_verified=True,
                         evidence="all ZIP members passed complete-read CRC and exact NPY header/payload verification; decoded with allow_pickle=False")
    yield arrays


def directory_identity(path: Path) -> tuple[str, str]:
    """Use the standard AMASS dataset/subject/sequence file hierarchy.

    Identity is based on the closest two enclosing directories, making the
    split independent of the absolute mount point or supplied root depth.
    Cross-dataset human identity and nonstandard nested layouts are not proven.
    """
    dataset, subject = path.parent.parent.name, path.parent.name
    if not dataset or not subject:
        raise ArchiveRejected("missing_dataset_or_subject_directory")
    return dataset, subject


def group_split(dataset: str, subject: str) -> tuple[str, str]:
    key = json.dumps(["amass-directory-subject", SEED, dataset, subject],
                     ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(key).hexdigest()
    value = int(digest[:16], 16)
    split = "train" if value < int(TRAIN_FRACTION * (1 << 64)) else "validation"
    return split, digest


def safe_field(archive: Any, key: str) -> np.ndarray:
    try:
        value = archive[key]
    except ValueError as error:
        if "Object arrays cannot be loaded" in str(error):
            raise ArchiveRejected(
                f"unsafe_object_dtype_refused:{key}; allow_pickle=False; "
                "cannot audit this field safely and will not retry with unsafe loading"
            ) from error
        raise ArchiveRejected(f"field_read_failed:{key}:{error}") from error
    if not isinstance(value, np.ndarray) or value.dtype.hasobject:
        raise ArchiveRejected(f"unsafe_object_dtype_refused:{key}")
    return value


def finite_numeric(value: np.ndarray, key: str) -> None:
    if value.dtype.kind not in NUMERIC_KINDS:
        raise ArchiveRejected(f"non_numeric_dtype:{key}:{value.dtype}")
    if not bool(np.isfinite(value).all()):
        raise ArchiveRejected(f"nonfinite_values:{key}")


def audit_archive(path: Path, format_report: dict[str, Any]) -> dict[str, Any]:
    with safe_parameter_archive(path, format_report) as archive:
        keys = set(archive.files)
        missing = sorted({"poses", "trans", "betas"} - keys)
        if missing:
            raise ArchiveRejected("missing_required_fields:" + ",".join(missing))
        # Read gender first, so an object gender is explicitly refused before
        # processing large numeric fields. Never retry with allow_pickle=True.
        warnings: list[str] = []
        gender = None
        gender_dtype = None
        if "gender" in keys:
            raw_gender = safe_field(archive, "gender")
            gender_dtype = str(raw_gender.dtype)
            if raw_gender.size != 1 or raw_gender.dtype.kind not in ("U", "S"):
                raise ArchiveRejected("gender_must_be_one_nonobject_string")
            value = raw_gender.reshape(-1)[0]
            if isinstance(value, (bytes, np.bytes_)):
                value = bytes(value).decode("utf-8", errors="strict")
            gender = str(value).strip().lower()
            if gender not in ("m", "male", "f", "female", "neutral"):
                raise ArchiveRejected("unsupported_gender_value:" + gender)
        else:
            warnings.append("gender_missing; body-model gender mapping remains unresolved")

        poses, trans, betas = (safe_field(archive, name)
                               for name in ("poses", "trans", "betas"))
        for name, value in (("poses", poses), ("trans", trans), ("betas", betas)):
            finite_numeric(value, name)
        if poses.ndim != 2 or poses.shape[0] < 1 or poses.shape[1] < 3 or poses.shape[1] % 3:
            raise ArchiveRejected("poses_must_be_nonempty_frames_by_axis_angle_dimension")
        frames, pose_dim = map(int, poses.shape)
        if trans.shape != (frames, 3):
            raise ArchiveRejected("trans_must_match_pose_frames_by_three")
        if not (betas.ndim == 1 or (betas.ndim == 2 and betas.shape[0] == 1)) or betas.size < 1:
            raise ArchiveRejected("betas_must_be_one_nonempty_sequence_shape_vector")

        fps_fields = [key for key in ("mocap_framerate", "fps") if key in keys]
        if not fps_fields:
            raise ArchiveRejected("missing_fps_or_mocap_framerate")
        fps_values = []
        for key in fps_fields:
            value = safe_field(archive, key)
            finite_numeric(value, key)
            if value.size != 1:
                raise ArchiveRejected("fps_must_be_one_scalar:" + key)
            fps = float(value.reshape(-1)[0])
            if not math.isfinite(fps) or fps <= 0:
                raise ArchiveRejected("fps_must_be_finite_and_positive:" + key)
            fps_values.append(fps)
        if any(not math.isclose(value, fps_values[0], rel_tol=1e-9, abs_tol=1e-9)
               for value in fps_values[1:]):
            raise ArchiveRejected("fps_fields_disagree")

        hypotheses = {72: "SMPL 24-joint axis-angle dimensions",
                      156: "SMPL-H 52-joint axis-angle dimensions",
                      165: "SMPL-X 55-joint axis-angle dimensions"}
        hypothesis = hypotheses.get(pose_dim, "unknown axis-angle joint layout")
        warnings.append("body-model representation is a dimensional hypothesis, not a verified mapping")
        return dict(frames=frames, fps=fps_values[0], fps_fields=fps_fields,
                    pose_dim=pose_dim, pose_dtype=str(poses.dtype),
                    translation_dtype=str(trans.dtype), betas_dim=int(betas.size),
                    betas_dtype=str(betas.dtype), gender=gender, gender_dtype=gender_dtype,
                    field_names=sorted(keys), representation_hypothesis=hypothesis,
                    body_model_mapping_verified=False, warnings=warnings)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data_root = args.data_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not data_root.is_dir():
        parser.error("--data-root must be an existing raw AMASS directory")
    # Reject any existing output, including an empty directory. All earlier
    # audit evidence is preserved; subsequent audits need a new output path.
    output.mkdir(parents=True, exist_ok=False)
    paths = sorted((path for path in data_root.rglob("*_poses.npz") if path.is_file()),
                   key=lambda path: path.relative_to(data_root).as_posix())
    records = []
    groups: dict[str, dict[str, Any]] = {}
    accepted = []
    rejected = []
    for index, path in enumerate(paths):
        relative = path.relative_to(data_root).as_posix()
        record: dict[str, Any] = dict(source_relative_path=relative, source_path=str(path))
        try:
            record.update(source_fingerprint(path))
            dataset, subject = directory_identity(path)
            split, group_sha = group_split(dataset, subject)
            group_key = json.dumps([dataset, subject], ensure_ascii=False, separators=(",", ":"))
            record.update(dataset=dataset, subject_directory=subject, group_key=group_key,
                          split=split, group_sha256=group_sha)
            record["source_format"] = classify_source(path, record)
            if not record["source_format"]["accepted_container"]:
                raise ArchiveRejected("unsupported_source_format:" + record["source_format"]["format"]
                                      + "; source remains unchanged; no unsafe loading or repair attempted")
            record.update(audit_archive(path, record["source_format"]), status="accepted_raw_parameters")
            group = groups.setdefault(group_key, dict(dataset=dataset, subject_directory=subject,
                                                      split=split, group_sha256=group_sha,
                                                      sequences=0, frames=0))
            group["sequences"] += 1
            group["frames"] += record["frames"]
            accepted.append(record)
        except (ArchiveRejected, ValueError, OSError, KeyError, UnicodeError, EOFError, BadZipFile, ZlibError) as error:
            record.update(status="rejected", rejection=str(error), unsafe_loading_attempted=False)
            rejected.append(record)
        records.append(record)
        if (index + 1) % 100 == 0:
            print(json.dumps(dict(event="amass_audit_progress", files=index + 1,
                                  accepted=len(accepted), rejected=len(rejected))), flush=True)

    split_stats = {}
    for split in ("train", "validation"):
        sequences = [record for record in accepted if record["split"] == split]
        split_stats[split] = dict(sequences=len(sequences),
                                 directory_subject_groups=sum(group["split"] == split for group in groups.values()),
                                 frames=sum(record["frames"] for record in sequences))
    fps_sequences = Counter(format(record["fps"], ".12g") for record in accepted)
    fps_frames: Counter[str] = Counter()
    for record in accepted:
        fps_frames[format(record["fps"], ".12g")] += record["frames"]
    stats = dict(discovered_raw_npz_files=len(paths), accepted_sequences=len(accepted),
                 rejected_sequences=len(rejected), directory_subject_groups=len(groups),
                 datasets=sorted({record["dataset"] for record in accepted}),
                 frames=sum(record["frames"] for record in accepted),
                 pose_dimensions=dict(sorted(Counter(str(record["pose_dim"]) for record in accepted).items())),
                 betas_dimensions=dict(sorted(Counter(str(record["betas_dim"]) for record in accepted).items())),
                 fps_sequences=dict(sorted(fps_sequences.items())), fps_frames=dict(sorted(fps_frames.items())),
                 split_counts=split_stats,
                 unhashed_sources=sum("source_sha256" not in record for record in records),
                 full_scanned_sources=sum(record.get("full_scan_complete", False) for record in records),
                 bytes_scanned=sum(record.get("bytes_scanned", 0) for record in records),
                 nonzero_bytes_count=sum(record.get("nonzero_bytes_count", 0) for record in records),
                 confirmed_all_zero_sources=sum(record.get("full_file_confirmed_all_zero", False) for record in records),
                 formats=dict(sorted(Counter(record.get("source_format", {}).get("format", "inspection_failed")
                                             for record in records).items())))
    status = ("complete" if accepted and not rejected else
              "complete_with_rejections" if paths else "no_raw_pose_archives_found")
    manifest = dict(format="v29-amass-raw-parameter-audit-v2", status=status,
                    data_root=str(data_root), statistics=stats, files=records,
                    subject_groups=[groups[key] for key in sorted(groups)],
                    safety=dict(numpy_allow_pickle=False, processed_pickle_files_read=False,
                                only_source_pattern="*_poses.npz", object_arrays_refused=True,
                                unsafe_loading_attempted=False, source_files_modified=False,
                                source_files_repaired=False, formats_verified_before_numpy_loading=True,
                                zip_member_load_method="complete ZIP member CRC read; BytesIO NPY np.load allow_pickle=False",
                                limits=dict(max_members=MAX_MEMBERS, max_member_uncompressed_bytes=MAX_MEMBER_BYTES,
                                            max_archive_uncompressed_bytes=MAX_UNCOMPRESSED_BYTES),
                                all_zero_inference="full streamed nonzero byte count; never inferred only from samples"),
                    split_policy=dict(seed=SEED, train_fraction=TRAIN_FRACTION,
                                      grouping="dataset directory + subject directory; whole group in one split",
                                      hash_algorithm="SHA256 of [policy, seed, dataset, subject]; first 64 bits",
                                      identity_inference="closest two parent directories of each raw pose archive",
                                      limitations="nonstandard nested layouts require manual identity review; human identity across datasets is not verified",
                                      validation_used_for_training=False, test_split_created=False),
                    representation=dict(content="raw motion/body parameters and translation; no RGB images",
                                        units_verified=False, joint_layout_verified=False,
                                        pose_dim_is_not_body_model_verification=True,
                                        converted_to_smpl=False, converted_to_smpl_h=False,
                                        contains_rgb=False, rendered_rgb=False, encoded_features=False,
                                        ready_for_model_training=False),
                    prerequisites_before_use=[
                        "确认数据使用许可、原始文件来源及目录中的 dataset/subject 身份分组。",
                        "确认原始 SMPL-H 或其他 body model、参数顺序、单位及匹配的模型资产。",
                        "完成并验证 SMPL-H 到目标 SMPL 的姿态、关节、形状映射；本审计没有转换。",
                        "需要图像任务时渲染 RGB、相机、遮挡和对应标签，并保留完整来源。",
                        "使用项目真实 RGB/VACE 编码器生成特征，禁止把动作参数宣称为 RGB 数据。",
                        "逐组检查 train/validation 无交叉；任何 object gender 被拒绝的序列不得投入训练。",
                    ])
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False,
                                                     indent=2, allow_nan=False), encoding="utf-8")
    summary = ["# AMASS 原始参数审计", "", f"状态：`{status}`。当前不能直接用于模型训练。", "",
               f"发现 {len(paths)} 个原始 pose npz；接受 {len(accepted)} 个，拒绝 {len(rejected)} 个。",
               f"接受部分共 {stats['frames']} 帧，{len(groups)} 个目录 subject 分组。", "",
               "这些是动作和人体参数，没有 RGB。尚未完成 body model 验证、SMPL-H/SMPL 映射、渲染或真实特征编码。", "",
               "按 dataset/subject 目录整体分组，seed 2026 的 SHA256 固定划分约 80% train、20% validation。",
               "目录分组不等于已验证的跨数据集真实人物身份，非标准嵌套目录须人工复核。", "",
               "所有 npz 使用 allow_pickle=False。object 类型的 gender 会拒绝整条序列，不尝试不安全加载；没有读取 processed pickle。", "",
               f"完整流式扫描 {stats['bytes_scanned']} 字节；确认整文件全零 {stats['confirmed_all_zero_sources']} 个。",
               "全零结论来自完整文件的非零字节计数，不根据头部或中尾样本推断；零头但非零正文另行报告。", "",
               "支持中央目录有效的带前缀 NumPy ZIP：完整读取所有 member 校验 CRC，再用 BytesIO 和 allow_pickle=False 解码 NPY。",
               "前缀长度、local-header offsets、成员摘要和 header/shape 字节长度校验均保留；不修复或覆盖源数据。", "",
               "完整逐文件 SHA256、格式、非零字节计数、拒绝原因、维度、帧率和分组见 manifest.json。", ""]
    (output / "SUMMARY_ZH.md").write_text("\n".join(summary), encoding="utf-8")
    print(json.dumps(dict(status=status, statistics=stats, ready_for_model_training=False,
                          manifest=str(output / "manifest.json")), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
