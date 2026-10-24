"""Fixed geometry/ROI, multiple development scenes, honest rejected-path reporting."""
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image
from ur5e_kinematics import UR5e,tool_rotation
from collision_world import CollisionWorld
from toilet_geometry import ToiletGeometry,SurfaceCamera
from camera_loop import Environment,run_loop

ROOT=Path(__file__).resolve().parents[1]
TOOL_LENGTH=float(os.environ.get('TOILET_TOOL_LENGTH','.18'))
OUT=ROOT/'output/geometry_trials'/f'tool_{round(TOOL_LENGTH*1000)}mm';OUT.mkdir(parents=True,exist_ok=True)
SEEDS=[42,123,999,2026,53772]


class PathRejected(Exception):pass


class Planner:
    def __init__(self,arm,world):self.arm,self.world=arm,world;self.path=[];self.phases=[];self.actions=[];self.attempts=[]

    def solve(self,xyz,R,q):
        target,pe,ae=self.arm.ik(xyz,R,q)
        if pe>.0001 or ae>.001:raise PathRejected(f'IK position={pe:.6f}, angle={ae:.6f}')
        return target

    def edge(self,start,end,phase):
        n=max(1,int(np.ceil(np.max(np.abs(end-start))/np.deg2rad(.25))))
        result=[]
        for t in np.linspace(0,1,n+1)[1:]:
            q=(1-t)*start+t*end;hits=self.world.check(q,phase)
            if hits:raise PathRejected(f'{phase}: '+', '.join(hits))
            result.append(q)
        return result

    def cartesian(self,q,p0,p1,R,phase):
        result=[];n=max(1,int(np.ceil(np.linalg.norm(p1-p0)/.001)))
        for t in np.linspace(0,1,n+1)[1:]:
            nextq=self.solve((1-t)*p0+t*p1,R,q)
            result.extend(self.edge(q,nextq,phase));q=nextq
        return q,result

    def append(self,qs,phase,action):
        self.path.extend(qs);self.phases.extend([phase]*len(qs));self.actions.extend([action]*len(qs))

    def run(self,strokes):
        first=strokes[0,0];R=tool_rotation(first[:3],first[3:])
        seed=np.load(ROOT/'output/ur5e/motion.npz')['joints'][0]
        q=self.solve(first[:3]+.08*first[3:],R,seed)
        hits=self.world.check(q,'transfer')
        if hits:raise PathRejected('initial pose: '+', '.join(hits))
        self.append([q],'start',1)
        q,part=self.cartesian(q,self.arm.fk(q)[:3,3],first[:3],R,'approach');self.append(part,'approach',1)
        for action,stroke in enumerate(strokes,1):
            for pose in stroke:
                nextq=self.solve(pose[:3],tool_rotation(pose[:3],pose[3:]),q)
                self.append(self.edge(q,nextq,'clean'),'clean',action);q=nextq
            last=stroke[-1];R=tool_rotation(last[:3],last[3:])
            if action==len(strokes):
                q,part=self.cartesian(q,last[:3],last[:3]+.08*last[3:],R,'retract');self.append(part,'retract',action);break
            nxt=strokes[action,0];Rn=tool_rotation(nxt[:3],nxt[3:]);accepted=False
            for clearance in [.08,.12,.16]:
                try:
                    q1,p1=self.cartesian(q,last[:3],last[:3]+clearance*last[3:],R,'retract')
                    destination=nxt[:3]+clearance*nxt[3:]
                    q2=self.solve(destination,Rn,q1);p2=self.edge(q1,q2,'transfer')
                    q3,p3=self.cartesian(q2,destination,nxt[:3],Rn,'approach')
                    self.append(p1,'retract',action);self.append(p2,'transfer',action+1);self.append(p3,'approach',action+1)
                    self.attempts.append(dict(after_action=action,clearance_m=clearance,result='accepted'))
                    q=q3;accepted=True;break
                except PathRejected as exc:self.attempts.append(dict(after_action=action,clearance_m=clearance,result='rejected',reason=str(exc)))
            if not accepted:raise PathRejected(f'No transfer after action {action}: '+self.attempts[-1]['reason'])
        return np.array(self.path)


def export_view(g,arm,planner,model,seed,summary):
    qs=np.array(planner.path);frames=[];time=0.;previous=qs[0];times=[]
    for i,q in enumerate(qs):
        time+=max(.02,float(np.max(np.abs(q-previous)))/np.deg2rad(15)) if i else 0.
        times.append(time);previous=q;t=arm.fk(q);phase=planner.phases[i]
        if phase=='clean':model.update_pose(t[:3,3],-t[:3,2])
        else:model.inside[:]=False;model.travel[:]=0.;model.previous=None
        boundary=i==0 or i==len(qs)-1 or phase!=planner.phases[i-1] or (i+1<len(qs) and phase!=planner.phases[i+1])
        if seed==42 and (i%8==0 or boundary):
            tf=arm.transforms(q)
            frames.append(dict(action=int(planner.actions[i]),phase=phase,time_s=time,xyz=t[:3,3].tolist(),
                rotation=[t[:3,0].tolist(),(-t[:3,1]).tolist(),(-t[:3,2]).tolist()],
                remaining=np.flatnonzero(model.remaining).tolist(),joints_deg=np.rad2deg(q).tolist(),
                arm={m['link']:tf[m['link']].tolist() for m in planner.world.meshes}))
    summary.update(samples=len(qs),fk_remaining_cells=int(model.remaining.sum()),duration_s=time,
        max_joint_step_deg=float(np.rad2deg(np.abs(np.diff(qs,axis=0))).max()))
    np.savez_compressed(OUT/f'motion_{seed}.npz',joints=qs,phases=np.array(planner.phases),actions=np.array(planner.actions),time_s=np.array(times))
    if seed==42:
        source=(ROOT/'output/continuous/continuous_cleaning_3d.html').read_text(encoding='utf-8')
        start=source.index('const data=')+len('const data=');end=source.index(';\nconst canvas',start)
        data=json.loads(source[start:end]);data['frames']=frames;data['dirt']=model.xyz.tolist()
        data['continuous']=summary;data['metrics']['cleaning_actions']=summary['cleaning_actions']
        data['geometry_version']='quartic_bowl_with_rim_v1'
        data['arm_verification']['tool_offset_tool0_m']=[0,0,arm.tool_length]
        data['bowl_mesh']=np.round(g.bowl_mesh(),6).tolist();data['rim_mesh']=np.round(g.rim_mesh(),6).tolist()
        template=(ROOT/'scripts/viewer3d.html').read_text(encoding='utf-8')
        html=template.replace('__SCENE_DATA__',json.dumps(data,separators=(',',':')))
        notice=(ROOT/'output/ur5e/THIRD_PARTY_LICENSE.txt').read_text(encoding='utf-8')
        (OUT/'toilet_with_rim_3d.html').write_text(html.replace('</html>','<!--\n'+notice+'\n--></html>'),encoding='utf-8')


def main():
    g=ToiletGeometry();camera=SurfaceCamera(g);results=[]
    # Freeze the ROI before generating/evaluating any scenes.
    u,v=np.meshgrid(np.linspace(-.11,.11,56),np.linspace(.025,.155,34))
    points=g.surface(u,v);visible=camera.visible(points)
    full_u,full_v=np.meshgrid(np.linspace(-.18,.18,91),np.linspace(-.24,.24,121))
    domain=(full_u/.18)**2+(full_v/.24)**2<.99**2
    full_points=g.surface(full_u[domain],full_v[domain]);full_visible=camera.visible(full_points)
    np.savez_compressed(OUT/'visibility.npz',xyz=full_points,visible=full_visible,roi_xyz=points,roi_visible=visible)
    vis=dict(roi_uv_m=[[-.11,.11],[.025,.155]],grid_points=int(visible.size),visible_points=int(visible.sum()),
             full_inner_points=len(full_points),full_inner_visible_points=int(full_visible.sum()),
             scope='fixed camera versus bowl/rim; arm occlusion not included')
    (OUT/'roi_visibility.json').write_text(json.dumps(vis,indent=2),encoding='utf-8')
    for seed in SEEDS:
        print(f'Seed {seed}: camera loop',flush=True)
        model,patches=g.make_scene(seed);env=Environment(model,camera)
        Image.fromarray(env.capture()).save(OUT/f'camera_before_{seed}.png')
        observation,_,strokes=run_loop(env,max_actions=100)
        Image.fromarray(env.capture()).save(OUT/f'camera_after_{seed}.png')
        fresh,_=g.make_scene(seed)
        report=dict(seed=seed,role='development',camera_status=observation['status'],
            camera_residual=observation['evaluation_residual_area'],cleaning_actions=len(strokes),
            dirty_cells=len(fresh.xyz),visible_dirty_cells=int(camera.visible(fresh.xyz).sum()),patches=patches)
        arm=UR5e(tool_length=TOOL_LENGTH);world=CollisionWorld(arm,g);planner=Planner(arm,world)
        report['tool_length_m']=TOOL_LENGTH
        try:
            if observation['evaluation_residual_area']>0:raise PathRejected('Camera loop left uncleaned dirt')
            planner.run(np.array(strokes));report['status']='planned'
            export_view(g,arm,planner,fresh,seed,report)
            if report['fk_remaining_cells']>0:report['status']='incomplete_cleaning'
        except PathRejected as exc:report.update(status='rejected',reason=str(exc),partial_samples=len(planner.path))
        report['collision_queries']=world.calls;report['attempts']=planner.attempts
        (OUT/f'result_{seed}.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        results.append(report)
        (OUT/'summary.json').write_text(json.dumps(dict(geometry='quartic_bowl_with_rim_v1',roi=vis,
            conditions_frozen_before_scenes=True,seeds=SEEDS,runs=results),indent=2),encoding='utf-8')
        print(json.dumps({k:v for k,v in report.items() if k not in ['patches','attempts']}),flush=True)


if __name__=='__main__':main()
