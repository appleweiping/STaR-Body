"""Build four independently encoded training phases without overwriting artifacts."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
import torch

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-vace',type=Path,required=True)
    p.add_argument('--source-hmr',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--stage',choices=('smoke','full'),required=True)
    p.add_argument('--crop-mode',choices=('legacy','square_pad'),default='legacy')
    p.add_argument('--device',default='cuda:0')
    a=p.parse_args()
    a.output_dir.mkdir(parents=True,exist_ok=False)
    source=torch.load(a.source_vace,map_location='cpu',weights_only=False,mmap=True)
    sig=source['signature']
    if sig['split']!='train':
        raise ValueError('Training source required')
    protocol=sig['encoder_protocol']
    entries=[]
    for phase in (0,2,4,6):
        output=a.output_dir/f'phase_{phase}.pt'
        cmd=[sys.executable,'-u','-m','divibody_training.cache_vace_phase_v14',
             '--data-root',sig['dataset_root'],'--split','train',
             '--vace-checkpoint-dir',sig['checkpoint']['directory'],
             '--output',str(output),'--window-phase',str(phase),
             '--crop-mode',a.crop_mode,'--num-workers','0',
             '--device',a.device,'--log-interval','1',
             '--vace-timestep',str(protocol['vace_timestep']),
             '--diffusion-input-protocol',protocol['diffusion_input_protocol'],
             '--feature-layers',*map(str,sig['feature_layers'])]
        if protocol.get('text_embedding_path'):
            cmd += ['--text-embedding-path',protocol['text_embedding_path']]
        if a.stage=='smoke':
            cmd += ['--development-max-windows','2']
        (a.output_dir/f'phase_{phase}_command.json').write_text(json.dumps(cmd,indent=2))
        with (a.output_dir/f'phase_{phase}.log').open('x') as log:
            result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
        (a.output_dir/f'phase_{phase}.exit').write_text(str(result.returncode))
        if result.returncode:
            raise RuntimeError(f'Phase {phase} failed; inspect its log')
        entries.append(dict(phase=phase,vace=output.name))
        print(json.dumps(dict(completed_phase=phase,stage=a.stage)),flush=True)
    manifest=dict(format='v9-training-phases-v14',source_vace=str(a.source_vace.resolve()),
                  source_hmr=str(a.source_hmr.resolve()),phases=entries)
    (a.output_dir/'phases.json').write_text(json.dumps(manifest,indent=2))
    control=dict(manifest,phases=[entry for entry in entries if entry['phase']==0])
    (a.output_dir/'phase0_control.json').write_text(json.dumps(control,indent=2))
    print('PHASE_CACHES_COMPLETE',flush=True)

if __name__=='__main__':main()
