"""Synthetic segmentation corpus v1. Dataset creation only, no training."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from toilet_geometry import ToiletGeometry,SurfaceCamera
from ur5e_kinematics import UR5e
from robot_camera import rasterize,robot_triangles,robot_foreground,roi_mask

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/segmentation_v1'
SPLITS={'train':(100000,600),'validation':(200000,120),'test':(300000,120)}
PALETTE={'brown':[105,57,26],'ochre':[158,135,69],'gray':[95,102,108],'olive':[82,104,61]}


class Generator:
    def __init__(self):
        self.geometry=ToiletGeometry();self.camera=SurfaceCamera(self.geometry,320,240)
        selected=json.loads((ROOT/'output/observation/pose_selection.json').read_text())
        arm=UR5e(tool_length=.26)
        depth,self.robot_rgb=rasterize(self.camera,robot_triangles(arm,np.array(selected['selected_q_rad'])))
        self.robot_mask=robot_foreground(self.camera,depth)
        self.roi=roi_mask(self.camera);self.observable=self.roi&~self.robot_mask
        if np.any(self.roi&self.robot_mask):raise RuntimeError('Observation pose hides training ROI')
        self.x,self.y=self.camera.points[...,0],self.camera.points[...,1]
        self.normals=self.geometry.normal(self.x,self.y)
        yy,xx=np.indices(self.roi.shape);self.sx=xx/self.camera.width;self.sy=yy/self.camera.height

    def generate(self,seed):
        rng=np.random.default_rng(seed)
        light=np.array([rng.uniform(-.6,.6),rng.uniform(-.6,.3),rng.uniform(.6,1.)]);light/=np.linalg.norm(light)
        shade=.55+.45*np.clip(self.normals@light,0,1)
        rgb=np.full((*self.roi.shape,3),[42.,53.,65.]);ceramic=shade[...,None]*np.array([229.,234.,234.])
        rgb[self.camera.valid]=ceramic[self.camera.valid]
        label=np.zeros(self.roi.shape,dtype=np.uint8);patches=[]
        count=0 if rng.random()<.15 else int(rng.integers(1,6))
        for _ in range(count):
            cx=rng.uniform(-.085,.085);cy=rng.uniform(.05,.13)
            rx,ry=rng.uniform(.009,.024,2);angle=rng.uniform(0,2*np.pi)
            dx=self.x-cx;dy=self.y-cy
            u=(np.cos(angle)*dx+np.sin(angle)*dy)/rx;v=(-np.sin(angle)*dx+np.cos(angle)*dy)/ry
            theta=np.arctan2(v,u);phase=rng.uniform(0,2*np.pi)
            radius=1+.18*np.sin(3*theta+phase)+.10*np.cos(5*theta-phase)
            mask=(u*u+v*v<=radius*radius)&self.roi
            color_name=list(PALETTE)[int(rng.integers(0,len(PALETTE)))];color=np.array(PALETTE[color_name],dtype=float)
            alpha=rng.uniform(.4,.95);texture=np.clip(rng.normal(1,.07,self.roi.shape),.75,1.2)
            rgb[mask]=rgb[mask]*(1-alpha)+(color*shade[mask,None]*texture[mask,None])*alpha
            label[mask]=1
            patches.append(dict(center_uv_m=[cx,cy],radius_uv_m=[rx,ry],angle_rad=angle,color=color_name,
                                opacity=alpha,required_passes=int(rng.integers(1,6))))
        # Deliberate synthetic shadow/highlight distractors; not physical path tracing.
        shadow=np.zeros(self.roi.shape)
        for _ in range(int(rng.integers(0,3))):
            px,py=rng.uniform(0,1,2);wx,wy=rng.uniform(.06,.3,2)
            shadow+=rng.uniform(.1,.45)*np.exp(-.5*(((self.sx-px)/wx)**2+((self.sy-py)/wy)**2))
        rgb*=np.clip(1-shadow,.45,1)[...,None]
        if rng.random()<.6:
            px,py=rng.uniform(.15,.85,2);width=rng.uniform(.02,.12)
            highlight=rng.uniform(.05,.45)*np.exp(-((self.sx-px)**2+(self.sy-py)**2)/(2*width*width))
            highlight*=self.camera.valid
            rgb=rgb*(1-highlight[...,None])+255*highlight[...,None]
        rgb[self.robot_mask]=self.robot_rgb[self.robot_mask]
        exposure=rng.uniform(.65,1.2);temperature=rng.uniform(.9,1.1,3);noise=rng.uniform(0,3)
        rgb=rgb*exposure*temperature+rng.normal(0,noise,rgb.shape)
        label[self.roi&self.robot_mask]=255
        return np.clip(rgb,0,255).astype(np.uint8),label,dict(seed=seed,patches=patches,exposure=exposure,temperature=temperature.tolist(),noise_sigma=noise)


def main():
    DATA.mkdir(parents=True,exist_ok=True);gen=Generator();rows=[];stats={};preview=[]
    Image.fromarray(gen.observable.astype(np.uint8)*255).save(DATA/'observable_roi.png')
    for split,(start,count) in SPLITS.items():
        folder=DATA/split;folder.mkdir(exist_ok=True);metadata=[];empty=0;dirty_pixels=0
        for seed in range(start,start+count):
            rgb,mask,meta=gen.generate(seed);image_path=folder/f'{seed}_rgb.png';mask_path=folder/f'{seed}_mask.png'
            Image.fromarray(rgb).save(image_path);Image.fromarray(mask).save(mask_path)
            pixels=int((mask==1).sum());empty+=pixels==0;dirty_pixels+=pixels
            meta['dirty_pixels']=pixels;metadata.append(meta)
            rows.append(dict(split=split,seed=seed,image=str(image_path.relative_to(DATA)),mask=str(mask_path.relative_to(DATA)),
                dirty_pixels=pixels,image_sha256=hashlib.sha256(image_path.read_bytes()).hexdigest(),mask_sha256=hashlib.sha256(mask_path.read_bytes()).hexdigest()))
            if split=='train' and seed<start+8:
                overlay=rgb.copy();overlay[mask==1]=(np.array([255,30,110])*.6+overlay[mask==1]*.4).astype(np.uint8)
                tile=Image.new('RGB',(640,270),'white');tile.paste(Image.fromarray(rgb),(0,30));tile.paste(Image.fromarray(overlay),(320,30))
                ImageDraw.Draw(tile).text((10,10),f'TRAIN {seed} | RGB / true dirt overlay | patches {len(meta["patches"])}',fill='black');preview.append(tile)
        (folder/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
        stats[split]=dict(images=count,empty_images=int(empty),dirty_pixels=dirty_pixels,seed_range=[start,start+count-1])
        print(f'{split}: {count} images, {empty} empty',flush=True)
    with (DATA/'manifest.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary=dict(version='segmentation_v1',resolution=[320,240],labels={'0':'clean or background','1':'dirt','255':'unobserved ROI; ignore loss'},
        splits=stats,split_unit='scene seed; one image per scene; disjoint seed ranges',
        metadata_not_model_inputs=['seed','patches','required_passes','true mask'],
        geometry='quartic_bowl_with_rim_v1',observation='raised and rotated UR5e pose; full declared ROI visible',
        training_status='not trained',limitations=['synthetic illustrative geometry only','procedural shadows/highlights not physical lighting',
            'fixed camera and robot observation pose','only four stain color families','no real-image validation'])
    (DATA/'dataset.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (DATA/'THIRD_PARTY_LICENSE.txt').write_text((ROOT/'output/ur5e/THIRD_PARTY_LICENSE.txt').read_text(encoding='utf-8'),encoding='utf-8')
    sheet=Image.new('RGB',(1280,1080),'white')
    for i,tile in enumerate(preview):sheet.paste(tile,((i%2)*640,(i//2)*270))
    sheet.save(ROOT/'output/observation/training_examples.png')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
