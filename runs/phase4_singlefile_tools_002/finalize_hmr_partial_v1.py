"""Recover a completely filled HMR partial using reversible CRC-only updates.

No tensor byte is rewritten. The old pathname is retained, and a durable journal
plus a virtual-undo SHA proves exact recoverability of the failed byte image.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import zipfile

sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location('crc_recovery_v2', Path(__file__).with_name('recover_hmr_singlefile_v2.py'))
h = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = h
spec.loader.exec_module(h)
require = h.require
GIB = 1024**3


def virtual_sha(path, patches):
    """Hash an original byte image without allocating a second large file."""
    digest, cursor = hashlib.sha256(), 0
    with Path(path).open('rb') as stream:
        for patch in sorted(patches, key=lambda item: item['offset']):
            offset = patch['offset']
            require(offset >= cursor, 'Overlapping journal entries')
            remaining = offset - cursor
            while remaining:
                data = stream.read(min(8*1024**2, remaining))
                require(data, 'Truncated original image')
                digest.update(data); remaining -= len(data)
            current = stream.read(4)
            require(current in (bytes.fromhex(patch['before_hex']), bytes.fromhex(patch['after_hex'])),
                    'CRC field has an unrecorded value')
            digest.update(bytes.fromhex(patch['before_hex'])); cursor = offset+4
        for data in iter(lambda: stream.read(8*1024**2), b''):
            digest.update(data)
    return digest.hexdigest()


def journal_crc(path, directory, expected_sha):
    path, directory = Path(path), Path(directory)
    before = h.record(path)
    require(before['sha256'] == expected_sha, 'Existing partial SHA differs')
    st = path.stat()
    require(path.is_file() and not path.is_symlink(), 'Partial must be a regular file')
    planned = h.crc_patch_plan(path)
    patches = []
    with path.open('rb') as stream:
        for row in planned:
            for offset in row['positions']:
                stream.seek(offset); old=stream.read(4)
                require(len(old)==4, 'Truncated CRC field')
                patches.append(dict(offset=offset, before_hex=old.hex(), after_hex=struct.pack('<I',row['crc32']).hex(), entry=row['name']))
    journal = dict(format='star-body-crc-undo-journal-v1', original=before,
        device=st.st_dev, inode=st.st_ino, mtime_ns=st.st_mtime_ns,
        patches=patches, crc_records=planned, tensor_bytes_rewritten=0,
        old_partial_path_retained=True, old_partial_crc_fields_mutated=True)
    h.write_new_json(directory/'crc_undo_journal.json', journal)
    h.fsync_dir(directory)
    # Hash immediately before mutation; keep a complete durable inverse first.
    require(h.record(path)==before, 'Partial changed while preparing journal')
    with path.open('r+b',buffering=0) as stream:
        for patch in patches:
            stream.seek(patch['offset'])
            require(stream.read(4).hex()==patch['before_hex'], 'CRC changed before patch')
            stream.seek(patch['offset']); stream.write(bytes.fromhex(patch['after_hex']))
        os.fsync(stream.fileno())
    now=path.stat()
    require((now.st_dev,now.st_ino,now.st_size)==(st.st_dev,st.st_ino,st.st_size),'Partial identity/size changed')
    restored_sha=virtual_sha(path, patches)
    require(restored_sha==expected_sha,'Virtual undo does not reproduce exact failed byte image')
    with zipfile.ZipFile(path) as archive:
        require(archive.testzip() is None,'Repaired ZIP CRC failed')
    result=dict(status='passed',original_sha256=expected_sha,virtual_undo_sha256=restored_sha,
        exact_original_reconstructible=True,after=h.record(path),journal=h.record(directory/'crc_undo_journal.json'),
        tensor_bytes_rewritten=0,crc_fields=len(patches),old_partial_crc_fields_mutated=True)
    h.write_new_json(directory/'crc_repair_receipt.json',result)
    return result


def verify_payload(args):
    import torch
    h.check_torch(torch)
    copy=h.copied_module(args.copy_source)
    parts=h.checked_directory(args.parts_dir)
    contract,state,base,before=h.load_plan(torch,copy,parts,args.expected_contract_sha256)
    actual=torch.load(args.verify_existing,map_location='cpu',weights_only=False,mmap=True)
    h.exact_payload(torch,actual,base,h.shard_chunks(torch,copy,parts,state,args.expected_contract_sha256,base['frame_uids']),copy)
    del actual;gc.collect()
    if not args.before_crc:
        with zipfile.ZipFile(args.verify_existing) as archive:
            require(archive.testzip() is None,'Independent ZIP CRC invalid')
    require(before=={name:h.record(parts/name) for name in before},'Original build metadata changed')
    h.write_new_json(args.verification_report,dict(status='passed',passed=True,torch_version=torch.__version__,
        frames=len(base['frame_uids']),shards=len(state['shards']),tensor_fields=5,
        crc_checked=not args.before_crc,crc_valid=not args.before_crc,bitwise_equal=True,
        output=h.record(args.verify_existing),original_metadata=before))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parts-dir',type=Path,required=True)
    p.add_argument('--run-dir',type=Path)
    p.add_argument('--expected-contract-sha256',required=True)
    p.add_argument('--copy-source',type=Path,required=True)
    p.add_argument('--existing-partial',type=Path)
    p.add_argument('--expected-partial-sha256')
    p.add_argument('--reserve-bytes',type=int,default=2*GIB)
    p.add_argument('--publish-alias',action='store_true')
    p.add_argument('--verify-existing',type=Path)
    p.add_argument('--verification-report',type=Path)
    p.add_argument('--before-crc',action='store_true')
    args=p.parse_args(argv)
    fd,lock_identity=h.inherited_controller_lock()
    if args.verify_existing:
        return verify_payload(args)
    import torch
    h.check_torch(torch)
    require(args.run_dir is not None and args.existing_partial is not None and args.expected_partial_sha256,
            'Dedicated new run, old partial and expected SHA required')
    require(args.reserve_bytes>=2*GIB,'Root reserve must be at least 2GiB')
    parent=h.checked_directory(args.run_dir.parent);parts=h.checked_directory(args.parts_dir)
    partial=args.existing_partial.absolute();run=parent/args.run_dir.name
    require(partial.is_file() and not partial.is_symlink(),'Existing regular partial required')
    require(not run.exists() and not run.is_symlink(),'Never replay an existing repair directory')
    require(partial.stat().st_dev==parent.stat().st_dev!=parts.stat().st_dev,'Storage layout differs')
    require(os.statvfs(parent).f_bavail*os.statvfs(parent).f_frsize>=args.reserve_bytes+256*1024**2,'Root repair reserve unavailable')
    copy=h.copied_module(args.copy_source)
    with h.readonly_build_lock(parts):
        contract,state,base,before=h.load_plan(torch,copy,parts,args.expected_contract_sha256)
        logical=Path(contract['parameters']['output']).absolute();h.checked_directory(logical.parent)
        require(not logical.exists() and not logical.is_symlink(),'Logical output exists')
        for row in contract['source_files']:
            require(h.digest(row['path'])==row['sha256'],'Original source changed')
        require(h.digest(contract['vace']['path'])==contract['vace']['sha256'],'Original VACE changed')
        run.mkdir(mode=0o700);final=run/logical.name
        sources={name:h.record(path) for name,path in dict(finalizer=__file__,crc_helper=h.__file__,copy=args.copy_source).items()}
        h.write_new_json(run/'recovery_inputs.json',dict(format='star-body-hmr-crc-finalization-v1',
            expected_partial_sha256=args.expected_partial_sha256,existing_partial=str(partial),
            original_metadata=before,original_sources=contract['source_files'],recovery_sources=sources,
            logical_output=str(logical),physical_output=str(final),inherited_controller_lock=lock_identity,
            old_partial_crc_fields_will_change=True,tensor_bytes_rewritten=0,original_state_mutated=False))
        try:
            for phase,pre in (('before_crc',True),('after_crc',False)):
                if not pre:
                    repair=journal_crc(partial,run,args.expected_partial_sha256)
                report=run/(phase+'.verification.json')
                cmd=[sys.executable,str(Path(__file__).resolve()),'--parts-dir',str(parts),
                    '--expected-contract-sha256',args.expected_contract_sha256,'--copy-source',str(args.copy_source.resolve()),
                    '--verify-existing',str(partial),'--verification-report',str(report)]
                if pre:cmd.append('--before-crc')
                with (run/(phase+'.verification.log')).open('xb') as log:
                    subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT,pass_fds=(fd,))
                verified=h.read_json(report)
                require(verified['passed'] and verified['bitwise_equal'] and verified['frames']==22735 and verified['shards']==178,
                        'Independent full tensor validation incomplete')
                if pre:require(verified['output']['sha256']==args.expected_partial_sha256,'Pre-repair SHA differs')
            require(before=={name:h.record(parts/name) for name in before},'Original metadata changed')
            require(sources=={name:h.record(path) for name,path in dict(finalizer=__file__,crc_helper=h.__file__,copy=args.copy_source).items()},'Recovery source changed')
            os.link(partial,final);h.fsync_dir(run)
            alias=dict(requested=args.publish_alias,published=False,logical=str(logical),target=str(final))
            if args.publish_alias:
                os.symlink(str(final),logical);h.fsync_dir(logical.parent)
                require(logical.is_symlink() and logical.resolve(strict=True)==final,'Alias differs')
                alias.update(published=True,symlink_inode=logical.lstat().st_ino)
            h.write_new_json(run/'recovery_receipt.json',dict(format='star-body-hmr-crc-finalization-publication-v1',
                status='recovered_pending_original_verifier',recovery_passed=True,contract_sha256=args.expected_contract_sha256,
                original_metadata=before,original_sources=contract['source_files'],recovery_sources=sources,alias=alias,
                output=dict(verified['output'],path=str(final)),verification=h.record(run/'after_crc.verification.json'),
                crc_repair=repair,frames=22735,windows=2909,shards=178,torch_version=torch.__version__,
                original_state_mutated=False,old_partial_crc_fields_mutated=True,old_partial_path_retained=str(partial),
                model_forward_performed=False,original_full_verifier_required=True,tensor_bytes_rewritten=0,
                inherited_controller_lock=lock_identity,all_original_assets_revalidated_by_recovery=False))
        except BaseException as error:
            h.write_new_json(run/'failure.json',dict(status='failed',error_type=type(error).__name__,error=str(error),
                original_state_mutated=False,automatic_cleanup_performed=False,partial_retained=partial.exists(),
                crc_journal_exists=(run/'crc_undo_journal.json').exists(),final_exists=final.exists()))
            raise


if __name__=='__main__':main()
