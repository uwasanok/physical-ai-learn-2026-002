import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from ur5e_kinematics import UR5e
from collision_world import CollisionWorld
from toilet_geometry import ToiletGeometry,SurfaceCamera
from camera_loop import Environment
from robot_camera import rasterize,robot_triangles,robot_foreground,roi_mask

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/observation';OUT.mkdir(exist_ok=True)
arm=UR5e(tool_length=.26);g=ToiletGeometry();cam=SurfaceCamera(g);world=CollisionWorld(arm,g)
data=np.load(ROOT/'output/geometry_trials/tool_260mm/motion_42.npz')
base=data['joints'][0];roi=roi_mask(cam);reports=[];best=None
tip=arm.fk(base);raised,pe,ae=arm.ik(np.array([*tip[:2,3],.65]),tip[:3,:3],base)
if pe>.0001 or ae>.001:raise RuntimeError('Raised pose IK failed')
for family,angle in [('low',a) for a in [0,-45,45,-90,90,-120,120]]+[('raised',a) for a in [0,45,90,-45]]:
    q=(base if family=='low' else raised).copy();q[0]+=np.deg2rad(angle)
    hits=world.check(q,'transfer')
    record=dict(family=family,shoulder_delta_deg=angle,q_rad=q.tolist(),collisions=hits)
    if not hits:
        depth,rgb=rasterize(cam,robot_triangles(arm,q));foreground=robot_foreground(cam,depth)
        count=int((foreground&roi).sum());record.update(roi_pixels=int(roi.sum()),occluded_roi_pixels=count)
        if family=='raised' and (best is None or count<best[0]):best=(count,q.copy(),depth,rgb)
    reports.append(record);print(json.dumps(record),flush=True)
if best is None or best[0]!=0:raise RuntimeError('No fully clear observation pose among candidates')
_,q,depth,rgb=best
np.savez_compressed(OUT/'observation_camera.npz',q_rad=q,robot_depth=depth,robot_rgb=rgb)
summary=dict(selected_q_rad=q.tolist(),roi_pixels=int(roi.sum()),occluded_roi_pixels=best[0],candidates=reports,
    geometry='quartic_bowl_with_rim_v1',tool_length_m=.26,scope='candidate pose checked; transition paths checked separately')
(OUT/'pose_selection.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
model,_=g.make_scene(42);background=Environment(model,cam).capture();images=[]
for title,pose in [('During cleaning',data['joints'][np.flatnonzero(data['phases']=='clean')[0]]),('Observation pose',q)]:
    d,c=rasterize(cam,robot_triangles(arm,pose));mask=robot_foreground(cam,d);frame=background.copy();frame[mask]=c[mask]
    canvas=Image.new('RGB',(640,520),'white');canvas.paste(Image.fromarray(frame),(0,40));draw=ImageDraw.Draw(canvas)
    draw.text((12,12),f'{title}: occluded ROI {int((mask&roi).sum())}/{int(roi.sum())} pixels',fill='black');images.append(canvas)
sheet=Image.new('RGB',(1280,520),'white');sheet.paste(images[0],(0,0));sheet.paste(images[1],(640,0));sheet.save(OUT/'occlusion_comparison.png')
