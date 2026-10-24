"""Retract-transfer-approach planning with sampled FCL collision validation."""
import json
from pathlib import Path
import numpy as np
from ur5e_kinematics import UR5e, tool_rotation
from collision_world import CollisionWorld
from cleaning_model import scene

root=Path(__file__).resolve().parents[1]
out=root/'output/continuous';out.mkdir(exist_ok=True)
arm=UR5e();world=CollisionWorld(arm)
poses=np.loadtxt(root/'output/view3d/tool_poses_xyz_normal.csv',delimiter=',',skiprows=1).reshape(-1,105,6)
old=np.load(root/'output/ur5e/motion.npz')['joints']
MAX_STEP=np.deg2rad(.25)


class Rejected(Exception): pass


def edge(q0,q1,phase):
    n=max(1,int(np.ceil(np.max(np.abs(q1-q0))/MAX_STEP)))
    samples=[]
    for t in np.linspace(0,1,n+1)[1:]:
        q=(1-t)*q0+t*q1
        hits=world.check(q,phase)
        if hits:raise Rejected(f'{phase}: {hits}')
        samples.append(q)
    return samples


def linear_tip(q,xyz0,xyz1,rotation,phase):
    samples=[]
    n=max(1,int(np.ceil(np.linalg.norm(xyz1-xyz0)/.001)))
    for t in np.linspace(0,1,n+1)[1:]:
        xyz=(1-t)*xyz0+t*xyz1
        target,pe,ae=arm.ik(xyz,rotation,q)
        if pe>.0001 or ae>.001:raise Rejected(f'{phase}: IK error {pe}, {ae}')
        samples.extend(edge(q,target,phase));q=target
    return q,samples


path=[];phases=[];actions=[];attempts=[]
def append(samples,phase,action):
    path.extend(samples);phases.extend([phase]*len(samples));actions.extend([action]*len(samples))

q=old[0]
first=poses[0,0];R=tool_rotation(first[:3],first[3:])
q,pe,ae=arm.ik(first[:3]+.08*first[3:],R,q)
if pe>.0001 or ae>.001 or world.check(q,'transfer'):raise RuntimeError('Initial standoff unreachable/colliding')
append([q],'start',1)
q,samples=linear_tip(q,arm.fk(q)[:3,3],first[:3],R,'approach');append(samples,'approach',1)
for action,stroke in enumerate(poses,1):
    for pose in stroke:
        target,pe,ae=arm.ik(pose[:3],tool_rotation(pose[:3],pose[3:]),q)
        if pe>.0001 or ae>.001:raise RuntimeError(f'Cleaning IK failed {action}')
        append(edge(q,target,'clean'),'clean',action);q=target
    end=stroke[-1];end_R=tool_rotation(end[:3],end[3:])
    if action==len(poses):
        q,samples=linear_tip(q,end[:3],end[:3]+.08*end[3:],end_R,'retract')
        append(samples,'retract',action);break
    next_pose=poses[action,0];next_R=tool_rotation(next_pose[:3],next_pose[3:])
    accepted=False
    for clearance in [.08,.12,.16]:
        try:
            q1,retract=linear_tip(q,end[:3],end[:3]+clearance*end[3:],end_R,'retract')
            destination=next_pose[:3]+clearance*next_pose[3:]
            q2,pe,ae=arm.ik(destination,next_R,q1)
            if pe>.0001 or ae>.001:raise Rejected('transfer IK unreachable')
            transfer=edge(q1,q2,'transfer')
            q3,approach=linear_tip(q2,destination,next_pose[:3],next_R,'approach')
            append(retract,'retract',action);append(transfer,'transfer',action+1);append(approach,'approach',action+1)
            q=q3;accepted=True
            attempts.append(dict(after_action=action,clearance_m=clearance,result='accepted'))
            break
        except Rejected as exc:
            attempts.append(dict(after_action=action,clearance_m=clearance,result='rejected',reason=str(exc)))
    if not accepted:
        (out/'failed_attempts.json').write_text(json.dumps(attempts,indent=2),encoding='utf-8')
        raise RuntimeError(f'No valid transfer after action {action}; no successful output published')
    print(f'Planned {action}/{len(poses)}',flush=True)

qs=np.array(path);cleaner,_=scene(42);frames=[];time=0.;previous=qs[0]
max_step=float(np.max(np.abs(np.diff(qs,axis=0))))
# Playback timestamps use a slow 15 deg/s per-joint bound. Acceleration is not checked.
times=[]
for i,q in enumerate(qs):
    time+=max(.02,float(np.max(np.abs(q-previous)))/np.deg2rad(15)) if i else 0.
    times.append(time);previous=q
    actual=arm.fk(q)
    if phases[i]=='clean':cleaner.update_pose(actual[:3,3],-actual[:3,2])
    else:
        cleaner.inside[:]=False;cleaner.travel[:]=0.;cleaner.previous=None
    # Save phase boundaries and 1/8 samples. Browser interpolates nothing across gaps.
    boundary=i==0 or i==len(qs)-1 or phases[i]!=phases[i-1] or (i+1<len(qs) and phases[i]!=phases[i+1])
    if i%8==0 or boundary:
        tf=arm.transforms(q)
        frames.append(dict(action=int(actions[i]),phase=phases[i],time_s=time,
            xyz=actual[:3,3].tolist(),rotation=[actual[:3,0].tolist(),(-actual[:3,1]).tolist(),(-actual[:3,2]).tolist()],
            remaining=np.flatnonzero(cleaner.remaining).tolist(),joints_deg=np.rad2deg(q).tolist(),
            arm={m['link']:tf[m['link']].tolist() for m in world.meshes}))
summary=dict(status='all_discrete_samples_collision_checked',actions=len(poses),samples=len(qs),frames=len(frames),
    max_joint_step_deg=float(np.rad2deg(max_step)),duration_s=time,planned_joint_speed_limit_deg_s=15,
    remaining_cells=int(cleaner.remaining.sum()),collision_queries=world.calls,transfers=len(poses)-1,
    rejected_candidates=sum(a['result']=='rejected' for a in attempts),
    phase_counts={p:phases.count(p) for p in sorted(set(phases))},
    allowed_self_pairs=[sorted(p) for p in world.exclusions],
    limitations=['discrete sampling not mathematical continuous collision proof',
                 'mesh containment and discretization limitations; simplified toilet/cabinet/camera geometry',
                 'adjacent mechanical links excluded; base/cabinet mounting excluded',
                 'brush/bowl contact allowed only near target with aligned pose during approach/clean/retract',
                 'no acceleration torque force or real robot validation', 'seed42 only; starts at a verified standoff pose'])
(out/'verification.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
(out/'planning_attempts.json').write_text(json.dumps(attempts,indent=2),encoding='utf-8')
np.savez_compressed(out/'motion.npz',joints=qs,phases=np.array(phases),actions=np.array(actions),time_s=np.array(times))
np.savetxt(out/'joint_trajectory.csv',np.column_stack([times,qs]),delimiter=',',header='time_s,'+','.join(arm.names),comments='')
source=(root/'output/ur5e/ur5e_cleaning_3d.html').read_text(encoding='utf-8')
start=source.index('const data=')+len('const data=');end=source.index(';\nconst canvas',start)
data=json.loads(source[start:end]);data['frames']=frames;data['continuous']=summary
template=(root/'scripts/viewer3d.html').read_text(encoding='utf-8')
html=template.replace('__SCENE_DATA__',json.dumps(data,separators=(',',':')))
notice=(root/'output/ur5e/THIRD_PARTY_LICENSE.txt').read_text(encoding='utf-8')
(out/'continuous_cleaning_3d.html').write_text(html.replace('</html>','<!--\n'+notice+'\n--></html>'),encoding='utf-8')
print(json.dumps(summary,indent=2))
