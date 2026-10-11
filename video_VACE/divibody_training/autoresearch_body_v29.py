"""Autoresearch body adapter: fixed validation, isolated runs, keep/discard ledger."""
import json,os,subprocess,sys,time,hashlib
from pathlib import Path
ROOT=Path('/mnt/32a56f7d-8270-4d4e-b9cb-fb8857bd7e93/testpro/xmj/sam-3d-body/divibody_vace')
PROJECT=ROOT/'video_VACE'
SESSION=ROOT/'training_runs/autoresearch_v29_20260928'
def main():
 SESSION.mkdir(exist_ok=False)
 env=os.environ.copy()
 env.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1')
 env['PYTHONPATH']=':'.join([str(ROOT.parent.parent/'Backup/4D-Humans'),str(PROJECT),str(PROJECT/'divibody_training'),str(ROOT/'dependencies/Wan2.1'),str(PROJECT/'vace/models')])
 template=json.loads((ROOT/'training_runs/v28_iterative_train_20260928/launch.json').read_text())['command']
 def run(name,options,protect=False,smoke=False):
  out=SESSION/name;out.mkdir(exist_ok=False)
  command=list(template);command[command.index('divibody_training.train_parts_v28')]='divibody_training.train_parts_v29_research'
  options={'output-dir':out/'run',**options}
  for key,value in options.items():
   flag='--'+key
   if flag in command:command[command.index(flag)+1]=str(value)
   else:command.extend([flag,str(value)])
  if protect:command.append('--protect-geometry-gradient')
  (out/'launch.json').write_text(json.dumps({'command':command,'gpu':0,'test_used':False},indent=2))
  with (out/'train.log').open('x') as log:
   p=subprocess.Popen(command,cwd=PROJECT,env=env,stdout=log,stderr=subprocess.STDOUT)
   (out/'pid').write_text(str(p.pid));code=p.wait()
  (out/'exit_code.txt').write_text(str(code))
  if code: return {'name':name,'status':'crash','exit_code':code}
  rows=[json.loads(x) for x in (out/'run/metrics.jsonl').read_text().splitlines()]
  first=rows[0]; b=first['validation'];bp=first['pose_validation']
  def score(x):
   v=x['validation'];q=x['pose_validation']
   if v['pve_mm']>b['pve_mm']+.5:return -1e9
   return (q['projected_PCKn@0.05']-bp['projected_PCKn@0.05'])+(q['projected_PCKn@0.10']-bp['projected_PCKn@0.10'])+(b['mpjpe_mm']-v['mpjpe_mm'])/100+(b['pa_mpjpe_mm']-v['pa_mpjpe_mm'])/100
  best=max(rows,key=score)
  return {'name':name,'status':'keep' if score(best)>0 else 'discard','score':score(best),'epoch':best['epoch'],'validation':best['validation'],'pose_validation':best['pose_validation']}
 def record(r):
  with (SESSION/'results.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
  print(json.dumps(r),flush=True)
 smoke=run('smoke',{'epochs':1,'min-epochs':1,'max-train-windows':32,'max-val-windows':32,'max-pose-train-windows':64,'max-pose-val-windows':32,'num-workers':0,'diagnostic-every':1},True,True)
 record(smoke)
 if smoke['status']=='crash':raise RuntimeError('Smoke failed; full trials not started')
 trials=[('control',1.,False),('low_2d',.25,False),('protected',1.,True)]
 results=[]
 for name,weight,protect in trials:
  opts={'epochs':4,'min-epochs':4,'pose-2d-weight':weight,'max-train-windows':1024,'max-pose-train-windows':4096,'max-val-windows':0,'max-pose-val-windows':0}
  r=run(name,opts,protect);record(r);results.append((r,weight,protect))
 viable=[x for x in results if x[0]['status']!='crash']
 if not viable:raise RuntimeError('All trials failed')
 winner,weight,protect=max(viable,key=lambda x:x[0]['score'])
 # Full confirmation always restarts from the same initialization, never from test feedback.
 r=run('full_confirmation',{'epochs':12,'min-epochs':8,'pose-2d-weight':weight,'max-train-windows':0,'max-pose-train-windows':0},protect);record(r)
 (SESSION/'completion.json').write_text(json.dumps({'screening_winner':winner,'full':r,'ram_superiority_verified':False},indent=2))
if __name__=='__main__':main()
