"""Exercise actual engine checkpoints with a tiny CPU network, not benchmark inputs."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from torch.utils.data import DataLoader,Dataset,IterableDataset
from divibody_training import v28d_training_engine as engine
from divibody_training.parts_training_utils_v28 import ModelEMA


class Samples(Dataset):
    files=[Path(f'{index}_clean.pt') for index in range(7)]
    def __len__(self): return 7
    def __getitem__(self,index):
        return dict(x=torch.tensor([index/7.,1.]),y=torch.tensor(index/14.))


class PoseSamples(IterableDataset):
    def __init__(self): self.cycle=0
    def __len__(self): return 7
    def set_cycle(self,cycle): self.cycle=cycle
    def __iter__(self):
        order=torch.randperm(7,generator=torch.Generator().manual_seed(19+self.cycle)).tolist()
        for index in order:
            yield Samples()[index] | dict(window_uid=f'pose/{index}')


def same(left,right):
    if torch.is_tensor(left): return torch.is_tensor(right) and torch.equal(left,right)
    if isinstance(left,dict): return left.keys()==right.keys() and all(same(left[key],right[key]) for key in left)
    if isinstance(left,(tuple,list)): return type(left)==type(right) and len(left)==len(right) and all(same(a,b) for a,b in zip(left,right))
    return left==right


def forward(model,batch,*unused): return model(batch['x']).squeeze(-1)
def geometry(prediction,batch,*unused):
    loss=(prediction-batch['y']).square().mean()
    return dict(total=loss,update_regularization=loss*0)
def pose(prediction,batch,*unused): return dict(pose_2d=(prediction-batch['y']).square().mean())
def evaluate(network):
    network.eval()
    return dict(pve_mm=50.,mpjpe_mm=40.),dict(**{'projected_PCKn@0.05':.6,'projected_PCKn@0.10':.8,
        'projected_mean_normalized_joint_error':.1})


class EngineResumeTest(unittest.TestCase):
    def test_optimizer_ema_samples_and_interrupted_logs_replay_exactly(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); manifest=root/'manifest.json'; manifest.write_text('{}')
            def run(directory,stop=0,resume=False):
                directory.mkdir(exist_ok=True); torch.manual_seed(2026)
                model=torch.nn.Sequential(torch.nn.Linear(2,4),torch.nn.Dropout(.2),torch.nn.Linear(4,1))
                optimizer=torch.optim.AdamW([dict(params=list(model.parameters()),group_name='shared',base_lr=.001)],lr=.001)
                ema=ModelEMA(model)
                args=SimpleNamespace(output_dir=directory,batch_size=3,seed=2026,coco_train_manifest=manifest,
                    resume_checkpoint=directory/'latest.pt' if resume else None,eval_only=False,diagnostic_every=1,
                    max_steps=4,eval_every=2,stop_after=stop,warmup_epochs=1.,amp_dtype='fp32',update_regularization=.001,
                    pose_2d_weight=.02,coco_weight=1.,coco_warmup_steps=0,grad_clip=1.,pve_tolerance_mm=.5,
                    pose_pck_tolerance=.005,num_workers=0)
                metadata=dict(args={key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()},
                    source_sha256='tiny-unit-fixture',coverage=dict(manifest_sha256=None),development_only=True)
                with patch.object(engine,'forward_batch',forward),patch.object(engine,'geometric_losses',geometry),patch.object(engine,'pose_losses',pose):
                    engine.execute_run(model,optimizer,ema,DataLoader(Samples(),batch_size=3),
                        DataLoader(PoseSamples(),batch_size=3),DataLoader(Samples(),batch_size=3),
                        None,None,None,evaluate,args,torch.device('cpu'),metadata)
            control=root/'control'; interrupted=root/'interrupted'
            run(control); run(interrupted,stop=2)
            with (interrupted/'samples.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(step=3,geometry=['unsaved']))+'\npartial')
            run(interrupted,resume=True)
            a,b=[torch.load(path/'latest.pt',map_location='cpu',weights_only=False) for path in (control,interrupted)]
            for key in ('model','optimizer','ema_model','ema_updates','stream_states','step','best_step','history','torch_rng'):
                self.assertTrue(same(a[key],b[key]),key)
            self.assertEqual((control/'samples.jsonl').read_text(),(interrupted/'samples.jsonl').read_text())
            self.assertTrue(list(interrupted.glob('samples.jsonl.before_resume_*')))
            self.assertFalse((interrupted/'paused.json').exists())
            self.assertEqual(json.loads((interrupted/'completion.json').read_text())['completed_steps'],4)


if __name__=='__main__': unittest.main()
