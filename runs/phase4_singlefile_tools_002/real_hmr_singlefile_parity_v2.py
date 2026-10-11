"""Real cached 16-frame I/O only, Linux Torch 2.5.1, independent reopen."""
import argparse
import gc
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import zipfile

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('singlefile_parity_impl', HERE / 'recover_hmr_singlefile_v2.py')
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)
spec2=importlib.util.spec_from_file_location('real_crc_finalizer', HERE/'finalize_hmr_partial_v1.py')
f=importlib.util.module_from_spec(spec2);spec2.loader.exec_module(f)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prior-parity-result', type=Path)
    p.add_argument('--expected-prior-receipt-sha256')
    p.add_argument('--run-dir', type=Path)
    p.add_argument('--copy-source', type=Path, default=HERE / r.COPY_NAME)
    p.add_argument('--child-source', type=Path)
    p.add_argument('--child-output', type=Path)
    p.add_argument('--child-report', type=Path)
    args = p.parse_args()
    import torch
    r.check_torch(torch)
    copy = r.copied_module(args.copy_source)
    if args.child_source:
        a = torch.load(args.child_source, map_location='cpu', weights_only=False, mmap=True)
        b = torch.load(args.child_output, map_location='cpu', weights_only=False, mmap=True)
        base = {key: a[key] for key in ('signature', 'frame_uids', 'window_uids', 'window_frame_indices')}
        r.exact_payload(torch, b, base, [(0, 16, a)], copy)
        with zipfile.ZipFile(args.child_output) as z: r.require(z.testzip() is None, 'ZIP CRC failed')
        r.write_new_json(args.child_report, {'status': 'passed', 'passed': True, 'tensor_fields': 5,
            'crc_valid': True, 'torch_version': torch.__version__, 'independent_process': True})
        return
    r.require(args.prior_parity_result and args.expected_prior_receipt_sha256 and args.run_dir,
              'Pinned successful prior parity receipt and new root run directory required')
    r.require(r.digest(args.prior_parity_result) == args.expected_prior_receipt_sha256, 'Prior receipt SHA differs')
    receipt = r.read_json(args.prior_parity_result)
    r.require(receipt['status'] == 'passed' and receipt['passed'] is True and
        receipt['real_tensor_parity'] is True and receipt['original_input_unchanged'] is True
        and receipt['frames'] == 16 and receipt['model_forward_performed'] is False, 'Prior real source evidence failed')
    original = receipt['outputs']['v31_original']['file']
    source = Path(original['path'])
    before = r.record(source)
    r.require(before['sha256'] == original['sha256'] and before['bytes'] == original['bytes'], 'Real probe bytes differ')
    parent = r.checked_directory(args.run_dir.parent)
    r.require(parent.stat().st_dev != source.stat().st_dev, 'Real test must write root filesystem from data2 source')
    run = parent / args.run_dir.name
    r.require(not run.exists() and not run.is_symlink(), 'Parity run must be new')
    run.mkdir(mode=0o700)
    sources = {'helper': r.digest(r.__file__), 'copy': r.digest(args.copy_source), 'test': r.digest(__file__), 'finalizer': r.digest(f.__file__)}
    source_payload = torch.load(source, map_location='cpu', weights_only=False, mmap=True)
    r.require(len(source_payload['frame_uids']) == 16 and
        source_payload['signature']['source_input_sha256'] == receipt['input_before']['sha256'], 'Full-source provenance differs')
    base = {key: source_payload[key] for key in ('signature', 'frame_uids', 'window_uids', 'window_frame_indices')}
    destination = run / 'recovered_real16.pt'
    storage = r.create_and_fill(torch, copy, destination, base, [(0, 7,
        {key: source_payload[key][:7] for key in r.FIELDS}), (7, 16,
        {key: source_payload[key][7:] for key in r.FIELDS})])
    # Deliberately recreate skip_data CRCs in this new test output only.
    import os, struct
    with destination.open('r+b',buffering=0) as stream:
        for row in r.crc_patch_plan(destination):
            for position in row['positions']:
                stream.seek(position);stream.write(struct.pack('<I',0))
        os.fsync(stream.fileno())
    original_crc_image=r.digest(destination)
    repair=f.journal_crc(destination,run,original_crc_image)
    r.require(repair['exact_original_reconstructible'] and repair['tensor_bytes_rewritten']==0,'CRC undo proof failed')
    del source_payload, base
    gc.collect()
    cmd = [sys.executable, str(Path(__file__).resolve()), '--child-source', str(source),
        '--child-output', str(destination), '--child-report', str(run / 'independent_reopen.json'),
        '--copy-source', str(args.copy_source.resolve())]
    with (run / 'independent_reopen.log').open('xb') as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
    r.require(r.record(source) == before, 'Original real input changed')
    r.require(r.digest(args.prior_parity_result) == args.expected_prior_receipt_sha256, 'Prior receipt changed')
    r.require(sources == {'helper': r.digest(r.__file__), 'copy': r.digest(args.copy_source),
                         'test': r.digest(__file__), 'finalizer': r.digest(f.__file__)}, 'Test sources changed')
    report = {'status': 'passed', 'passed': True, 'torch_version': torch.__version__,
        'real_tensor_parity': True, 'crc_valid': True, 'original_input_unchanged': True,
        'frames': 16, 'tensor_fields': 5, 'model_forward_performed': False,
        'source_sha256': {
            'runs/phase4_singlefile_tools_002/recover_hmr_singlefile_v2.py': sources['helper'],
            'runs/phase4_singlefile_tools_001/' + r.COPY_NAME: sources['copy'],
            'runs/phase4_singlefile_tools_002/real_hmr_singlefile_parity_v2.py': sources['test'],
            'runs/phase4_singlefile_tools_002/finalize_hmr_partial_v1.py': sources['finalizer']},
        'checks': 12, 'different_filesystems': True,
        'input': before, 'source_full_cache': receipt['input_before'],
        'prior_evidence': r.record(args.prior_parity_result), 'output': r.record(destination),
        'independent_reopen': r.record(run / 'independent_reopen.json'), 'storage': storage,'crc_repair':repair}
    r.write_new_json(run / 'result.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__': main()
