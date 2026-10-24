import json
from pathlib import Path
import numpy as np
from ur5e_kinematics import UR5e
from collision_world import CollisionWorld
from toilet_geometry import ToiletGeometry

root=Path(__file__).resolve().parents[1];out=root/'output/observation'
arm=UR5e(tool_length=.26);world=CollisionWorld(arm,ToiletGeometry())
selection=json.loads((out/'pose_selection.json').read_text());park=np.array(selection['selected_q_rad'])
records=[];paths=[];offsets=[0]
def checked_edge(a,b):
    n=max(1,int(np.ceil(np.abs(b-a).max()/np.deg2rad(.25))))
    path=np.linspace(a,b,n+1)
    for q in path:
        hits=world.check(q,'transfer')
        if hits:return [],hits
    return list(path),[]

def via_lift(target):
    tip=arm.fk(target);xyz=tip[:3,3];high=xyz.copy();high[2]=.65
    q=target.copy();outpath=[q]
    for alpha in np.linspace(0,1,max(2,int(np.ceil(np.linalg.norm(high-xyz)/.005))+1))[1:]:
        qnext,pe,ae=arm.ik((1-alpha)*xyz+alpha*high,tip[:3,:3],q)
        if pe>.0001 or ae>.001:return [],['lift IK failed']
        edge,hits=checked_edge(q,qnext)
        if hits:return [],hits
        outpath.extend(edge[1:]);q=qnext
    edge,hits=checked_edge(q,park)
    if hits:return [],hits
    outpath.extend(edge[1:])
    return np.array(outpath)[::-1],[]  # saved park -> standoff, reversible
for seed in [42,123,999,2026,53772]:
    d=np.load(root/f'output/geometry_trials/tool_260mm/motion_{seed}.npz');phases=d['phases'];q=d['joints'];actions=d['actions']
    endpoints=[]
    for i,p in enumerate(phases):
        if p=='approach' and (i==0 or phases[i-1]!='approach'):endpoints.append((max(0,i-1),'before_approach'))
        if p=='retract' and (i==len(phases)-1 or phases[i+1]!='retract'):endpoints.append((i,'after_retract'))
    for i,kind in endpoints:
        target=q[i];path,hits=via_lift(target)
        record=dict(seed=seed,action=int(actions[i]),endpoint=kind,samples=len(path),valid=not hits,collisions=hits)
        records.append(record)
        if not hits:paths.extend(path);offsets.append(len(paths))
    print(f'Seed {seed}: route checks completed',flush=True)
summary=dict(selected_q_rad=park.tolist(),transitions=len(records),valid=sum(r['valid'] for r in records),
    checked_samples=world.calls,records=records,scope='lift tip to z=0.65m then rotate; max 0.25 deg; reversible geometry only; no dynamics')
(out/'routes.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
np.savez_compressed(out/'routes.npz',joints=np.array(paths),offsets=np.array(offsets))
print(json.dumps({k:v for k,v in summary.items() if k!='records'},indent=2))
if summary['valid']!=summary['transitions']:raise RuntimeError('Some observation routes failed; do not claim complete integration')
