"""Online RGB inference -> checked UR5e motion -> actual FK contact -> reobserve.
Simulator truth is private to rendering/contact and post-hoc evaluation.
Frozen perception weights are never updated here.
"""
import base64,io,json,time,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from camera_loop import Environment,plan_mask
from cleaning_model import Cleaner,residual_area_fraction
from toilet_geometry import ToiletGeometry,SurfaceCamera
from ur5e_kinematics import UR5e,tool_rotation
from collision_world import CollisionWorld
from geometry_trials import Planner,PathRejected
from learned_perception import LearnedDirtDetector
from robot_camera import rasterize,robot_triangles,robot_foreground

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/closed_loop_v1'
CASES=[dict(name=f'random_{s}',seed=s,max_actions=100) for s in [410001,410002,410003]]+[
    dict(name='empty',seed=410004,empty=True,max_actions=10),
    dict(name='hard_budget',seed=410001,required=999,max_actions=3)]

class Scene(Environment):
    def __init__(self,model,camera,seed,robot_mask,robot_rgb):
        super().__init__(model,camera)
        self.robot_mask,self.robot_rgb=robot_mask,robot_rgb
        rng=np.random.default_rng(seed+700000)
        self.exposure=float(rng.uniform(.8,1.1));self.temperature=rng.uniform(.94,1.06,3)
        palette=np.array([[105,57,26],[158,135,69],[95,102,108],[82,104,61]])
        # Color varies spatially across the three stains; unrelated to required n.
        self.colors=palette[np.clip(((model.uv[:,0]+.11)/.055).astype(int),0,3)]
        self.noise=rng.normal(0,1,(camera.height,camera.width,3))
    def capture(self):
        rgb=super().capture().astype(float);ids=self.cell_id;mask=ids>=0
        if len(self.model.remaining):
            mask &= self.model.remaining[np.maximum(ids,0)]
            rgb[mask]=self.colors[ids[mask]]
        rgb[self.robot_mask]=self.robot_rgb[self.robot_mask]
        return np.clip(rgb*self.exposure*self.temperature+self.noise,0,255).astype(np.uint8)

class OnlinePlanner(Planner):
    def to_park(self,target,park):
        tip=self.arm.fk(target);xyz=tip[:3,3];high=xyz.copy();high[2]=.65
        q=target.copy();path=[q]
        for alpha in np.linspace(0,1,max(2,int(np.ceil(np.linalg.norm(high-xyz)/.005))+1))[1:]:
            nxt=self.solve((1-alpha)*xyz+alpha*high,tip[:3,:3],q)
            path.extend(self.edge(q,nxt,'transfer'));q=nxt
        path.extend(self.edge(q,park,'transfer'))
        return path
    def action(self,target,park):
        g=self.world.geometry if hasattr(self.world,'geometry') else ToiletGeometry()
        u,v=target[:2];xs=np.linspace(u-.026,u+.026,105)
        xyz=g.surface(xs,np.full_like(xs,v));normal=g.normal(xs,np.full_like(xs,v))
        R=tool_rotation(xyz[0],normal[0]);standoff=xyz[0]+.08*normal[0]
        seed=np.load(ROOT/'output/ur5e/motion.npz')['joints'][0]
        q=self.solve(standoff,R,seed)
        # Plan/check the ENTIRE action before applying it to simulated dirt.
        path=list(reversed(self.to_park(q,park)));phases=['observation_move']*len(path)
        q,part=self.cartesian(q,standoff,xyz[0],R,'approach');path.extend(part);phases.extend(['approach']*len(part))
        for p,n in zip(xyz,normal):
            nxt=self.solve(p,tool_rotation(p,n),q);part=self.edge(q,nxt,'clean')
            path.extend(part);phases.extend(['clean']*len(part));q=nxt
        q,part=self.cartesian(q,xyz[-1],xyz[-1]+.08*normal[-1],tool_rotation(xyz[-1],normal[-1]),'retract')
        path.extend(part);phases.extend(['retract']*len(part))
        part=self.to_park(q,park);path.extend(part);phases.extend(['observation_move']*len(part))
        return path,phases

def run_case(case,detector,camera,park,robot_mask,robot_rgb,export=False):
    folder=OUT/case['name'];folder.mkdir(parents=True,exist_ok=True)
    g=camera.geometry;model,patches=g.make_scene(case['seed'])
    if case.get('empty'):model=Cleaner([],[],g.surface,g.normal)
    if 'required' in case:model.required[:]=case['required']
    env=Scene(model,camera,case['seed'],robot_mask,robot_rgb)
    arm=UR5e(tool_length=.26);world=CollisionWorld(arm,g);planner=OnlinePlanner(arm,world)
    q=park.copy();qs=[q];phases=['observe'];actions=[0];times=[0.];frames=[];captures=[];records=[]
    elapsed=0.;done=0;status='budget_exhausted';started=time.perf_counter();reason=None
    assert not world.check(q,'transfer')
    def frame(phase,action,observation):
        tip=arm.fk(q);tf=arm.transforms(q)
        frames.append(dict(action=action,phase=phase,time_s=elapsed,xyz=tip[:3,3].tolist(),
            rotation=[tip[:3,0].tolist(),(-tip[:3,1]).tolist(),(-tip[:3,2]).tolist()],
            remaining=np.flatnonzero(model.remaining).tolist(),joints_deg=np.rad2deg(q).tolist(),observation=observation,
            arm={m['link']:tf[m['link']].tolist() for m in world.meshes}))
    for step in range(case['max_actions']+1):
        assert np.allclose(q,park,atol=1e-8)
        rgb=env.capture();mask=detector.predict(rgb);target,pixels=plan_mask(mask,camera)
        Image.fromarray(rgb).save(folder/f'capture_{step:03d}.png')
        Image.fromarray(mask.astype(np.uint8)*255).save(folder/f'mask_{step:03d}.png')
        overlay=rgb.copy();overlay[mask]=(overlay[mask]*.4+np.array([255,30,110])*.6).astype(np.uint8)
        buf=io.BytesIO();Image.fromarray(overlay).save(buf,format='PNG')
        elapsed+=1.;captures.append(dict(time_s=elapsed,action=done,detected_pixels=pixels,image='data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode()))
        records.append(dict(step=step,pixels=pixels,target=None if target is None else target.tolist(),
            evaluation_remaining_cells=int(model.remaining.sum()),evaluation_residual_area=residual_area_fraction(model)))
        if export:frame('observe',done,step)
        if target is None:
            status='observed_roi_clear' if pixels==0 else 'outside_roi';break
        if step==case['max_actions']:break
        try:path,parts=planner.action(target,park)
        except PathRejected as exc:status='path_rejected';reason=str(exc);break
        done+=1
        for i,(nxt,phase) in enumerate(zip(path,parts)):
            elapsed+=max(.02,float(np.abs(nxt-q).max())/np.deg2rad(15));q=nxt;tip=arm.fk(q)
            if phase=='clean':model.update_pose(tip[:3,3],-tip[:3,2])
            else:model.inside[:]=False;model.travel[:]=0.;model.previous=None
            qs.append(q);phases.append(phase);actions.append(done);times.append(elapsed)
            if export and (i%20==0 or i==len(path)-1 or i==0 or parts[i-1]!=phase):frame(phase,done,step)
        print(f'{case["name"]} action={done} remaining={int(model.remaining.sum())}',flush=True)
    joint_array=np.array(qs)
    result=dict(**case,status=status,reason=reason,actions=done,observations=len(records),records=records,
        remaining_cells=int(model.remaining.sum()),residual_area=residual_area_fraction(model),
        false_clear=status=='observed_roi_clear' and bool(model.remaining.any()),
        checked_configurations=world.calls,executed_samples=len(qs),max_joint_step_deg=float(np.rad2deg(np.abs(np.diff(joint_array,axis=0))).max()) if len(qs)>1 else 0.,
        planned_duration_s=elapsed,compute_duration_s=time.perf_counter()-started,patches=patches,
        required_passes_values=np.unique(model.required).tolist(),
        scope='RGB-only online perception; sampled kinematics/collision and geometric contact surrogate; no dynamics or torque control')
    np.savez_compressed(folder/'motion.npz',joints=joint_array,phases=np.array(phases),actions=actions,time_s=times,
                        dirt_xyz=model.xyz,required=model.required,passes=model.passes)
    (folder/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    if export:
        source=(ROOT/'output/observation/observation_cleaning_3d.html').read_text(encoding='utf-8')
        a=source.index('const data=')+len('const data=');b=source.index(';\nconst canvas',a);data=json.loads(source[a:b])
        data.update(frames=frames,dirt=model.xyz.tolist(),observation_run=result,camera_captures=captures)
        data['metrics']['cleaning_actions']=done;data['continuous']=result
        template=(ROOT/'scripts/viewer3d.html').read_text(encoding='utf-8')
        html=template.replace('__SCENE_DATA__',json.dumps(data,separators=(',',':')))
        html=html.replace('<title>','<title>学習AIの閉ループ清掃 | ')
        notice=(ROOT/'output/ur5e/THIRD_PARTY_LICENSE.txt').read_text(encoding='utf-8')
        (OUT/'learned_cleaning_3d.html').write_text(html.replace('</html>','<!--\n'+notice+'\n--></html>'),encoding='utf-8')
    return result

def main():
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    frozen=json.loads((ROOT/'output/learning_v1/frozen_config.json').read_text())
    assert hashlib.sha256((ROOT/'output/learning_v1/best_model.pt').read_bytes()).hexdigest()==frozen['model_sha256']
    (OUT/'protocol.json').write_text(json.dumps(dict(cases=CASES,model_sha256=frozen['model_sha256'],threshold=frozen['threshold'],
        policy='Frozen model; no retuning on closed-loop scenes; 100-action cap; stop on rejected path',
        pass_criterion='random scenes: observed clear AND evaluation remaining=0; empty: no actions; hard: budget stop without false clear'),indent=2))
    detector=LearnedDirtDetector('cuda');g=ToiletGeometry();camera=SurfaceCamera(g,320,240)
    park=np.array(json.loads((ROOT/'output/observation/pose_selection.json').read_text())['selected_q_rad'])
    depth,robot_rgb=rasterize(camera,robot_triangles(UR5e(tool_length=.26),park));robot_mask=robot_foreground(camera,depth)
    results=[]
    for i,case in enumerate(CASES):
        results.append(run_case(case,detector,camera,park,robot_mask,robot_rgb,export=i==0))
        (OUT/'summary.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print(json.dumps([{k:r[k] for k in ['name','status','actions','remaining_cells','false_clear','checked_configurations']} for r in results],indent=2))

if __name__=='__main__':main()
