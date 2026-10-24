"""Replay seed42 with checked imaging detours; camera frames include the robot."""
import base64,io,json
from pathlib import Path
import numpy as np
from PIL import Image
from ur5e_kinematics import UR5e
from toilet_geometry import ToiletGeometry,SurfaceCamera
from camera_loop import Environment,detect
from robot_camera import robot_foreground

root=Path(__file__).resolve().parents[1];out=root/'output/observation';seed=42
arm=UR5e(tool_length=.26);g=ToiletGeometry();camera=SurfaceCamera(g)
base=np.load(root/'output/geometry_trials/tool_260mm/motion_42.npz')
routes=np.load(out/'routes.npz');records=json.loads((out/'routes.json').read_text())['records']
lookup={}
for j,r in enumerate(records):
    if r['seed']==seed:lookup[(r['action'],r['endpoint'])]=routes['joints'][routes['offsets'][j]:routes['offsets'][j+1]]
obs=np.load(out/'observation_camera.npz');park=obs['q_rad'];robot=robot_foreground(camera,obs['robot_depth'])
qpaths=[];phases=[];actions=[]
def add(path,phase,action):qpaths.extend(path);phases.extend([phase]*len(path));actions.extend([action]*len(path))
add([park],'observe',1)
for action in range(1,int(base['actions'].max())+1):
    add(lookup[(action,'before_approach')],'observation_move',action)
    ids=np.flatnonzero((base['actions']==action)&np.isin(base['phases'],['approach','clean','retract']))
    for i in ids:add([base['joints'][i]],str(base['phases'][i]),action)
    add(lookup[(action,'after_retract')][::-1],'observation_move',action)
    add([park],'observe',action)
qs=np.array(qpaths)
assert np.abs(np.diff(qs,axis=0)).max()<=np.deg2rad(.25)+1e-10
model,_=g.make_scene(seed);env=Environment(model,camera);frames=[];captures=[];time=0.;last=qs[0];observation=-1
source=(root/'output/geometry_trials/tool_260mm/toilet_with_rim_3d.html').read_text(encoding='utf-8')
start=source.index('const data=')+len('const data=');end=source.index(';\nconst canvas',start);data=json.loads(source[start:end])
meshes=arm.collision_meshes()
for i,(q,phase,action) in enumerate(zip(qs,phases,actions)):
    time+=1.0 if phase=='observe' else max(.02,float(np.abs(q-last).max())/np.deg2rad(15));last=q
    tip=arm.fk(q)
    if phase=='clean':model.update_pose(tip[:3,3],-tip[:3,2])
    else:model.inside[:]=False;model.travel[:]=0.;model.previous=None
    if phase=='observe':
        rgb=env.capture();rgb[robot]=obs['robot_rgb'][robot]
        image=Image.fromarray(rgb);image.save(out/f'capture_{len(captures):02d}.png')
        buf=io.BytesIO();image.resize((320,240)).save(buf,format='PNG')
        captures.append(dict(time_s=time,action=action,detected_pixels=int(detect(rgb).sum()),
            image='data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode()))
        observation=len(captures)-1
    boundary=i==0 or i==len(qs)-1 or phases[i-1]!=phase or (i+1<len(qs) and phases[i+1]!=phase)
    if i%16==0 or boundary or phase=='observe':
        tf=arm.transforms(q)
        frames.append(dict(action=int(action),phase=phase,time_s=time,xyz=tip[:3,3].tolist(),
            rotation=[tip[:3,0].tolist(),(-tip[:3,1]).tolist(),(-tip[:3,2]).tolist()],
            remaining=np.flatnonzero(model.remaining).tolist(),joints_deg=np.rad2deg(q).tolist(),observation=observation,
            arm={m['link']:tf[m['link']].tolist() for m in meshes}))
summary=dict(seed=seed,samples=len(qs),observations=len(captures),remaining_cells=int(model.remaining.sum()),
    planned_duration_s=time,max_joint_step_deg=float(np.rad2deg(np.abs(np.diff(qs,axis=0))).max()),
    description='Replay of existing targets with validated observation routes; not newly trained perception')
data['frames']=frames;data['observation_run']=summary;data['camera_captures']=captures
template=(root/'scripts/viewer3d.html').read_text(encoding='utf-8')
html=template.replace('__SCENE_DATA__',json.dumps(data,separators=(',',':')))
notice=(root/'output/ur5e/THIRD_PARTY_LICENSE.txt').read_text(encoding='utf-8')
(out/'observation_cleaning_3d.html').write_text(html.replace('</html>','<!--\n'+notice+'\n--></html>'),encoding='utf-8')
(out/'replay_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
np.savez_compressed(out/'replay.npz',joints=qs,phases=np.array(phases),actions=np.array(actions))
print(json.dumps(summary,indent=2))
