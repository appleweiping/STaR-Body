"""Isolated official source snapshot; derived from preserved fetch v1.

The API-free fallback records archive/file SHA256, without claiming a Git
commit. It extracts selected ordinary files only and does not execute code.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    url = 'https://codeload.github.com/facebookresearch/MHR/zip/refs/heads/main'
    rows = []
    report = dict(status='failed', repository='https://github.com/facebookresearch/MHR',
                  source_url=url, commit_not_established=True, files=rows,
                  dependencies_installed=False, conversion_not_yet_run=True)
    try:
        request = urllib.request.Request(url, headers={'User-Agent': 'v29-source-audit'})
        with urllib.request.urlopen(request, timeout=45) as response:
            archive = response.read(64 * 1024**2 + 1)
        if len(archive) > 64 * 1024**2:
            raise ValueError('Official archive exceeds bounded 64 MiB download')
        report['archive_sha256'] = hashlib.sha256(archive).hexdigest()
        report['archive_bytes'] = len(archive)
        with zipfile.ZipFile(io.BytesIO(archive)) as source:
            selected = []
            for info in source.infolist():
                if info.is_dir():
                    continue
                parts = Path(info.filename).parts
                if len(parts) < 2 or parts[0] != 'MHR-main':
                    raise ValueError('Unexpected official archive layout')
                relative = Path(*parts[1:])
                name = relative.as_posix()
                if name.startswith('tools/mhr_smpl_conversion/') or name in (
                    'LICENSE', 'mhr/mhr.py', 'mhr/__init__.py'):
                    if (info.external_attr >> 16) & 0o170000 == 0o120000:
                        raise ValueError('Refusing symbolic links')
                    selected.append((info, relative))
            if not selected or sum(info.file_size for info, _ in selected) > 64 * 1024**2:
                raise ValueError('Missing or excessive selected conversion sources')
            for info, relative in selected:
                target = (args.output / relative).resolve()
                if not target.is_relative_to(args.output.resolve()):
                    raise ValueError('Unsafe archive path')
                data = source.read(info)
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open('xb') as stream:
                    stream.write(data)
                rows.append(dict(path=relative.as_posix(), bytes=len(data),
                    sha256=hashlib.sha256(data).hexdigest(),
                    lfs_pointer=data.startswith(b'version https://git-lfs.github.com/spec/v1')))
                print('Fetched', relative.as_posix(), len(data), flush=True)
        report['status'] = 'complete'
    except Exception as error:
        report['error'] = repr(error)
        raise
    finally:
        with (args.output / 'source_manifest.json').open('x') as stream:
            json.dump(report, stream, indent=2)


if __name__ == '__main__':
    main()
