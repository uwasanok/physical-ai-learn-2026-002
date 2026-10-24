"""Geometry-only feasibility sketch. No robot, contact, cleaning, or learned AI."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output'; OUT.mkdir(exist_ok=True)
# Illustrative dimensions, not measured from a manufacturer's toilet.
a,b,depth,bottom=0.18,0.24,0.18,0.22
camera=np.array([0.0,-0.65,1.0]); look_at=np.array([0,0,0.32])
x=np.linspace(-a,a,101); y=np.linspace(-b,b,121)
X,Y=np.meshgrid(x,y); radius2=(X/a)**2+(Y/b)**2
valid=radius2<=1
Z=bottom+depth*radius2
gx=2*depth*X/a**2; gy=2*depth*Y/b**2
normal=np.stack([-gx,-gy,np.ones_like(X)],axis=-1)
normal/=np.linalg.norm(normal,axis=-1,keepdims=True)
tilt=np.degrees(np.arccos(normal[:,:,2]))
assert np.allclose(np.linalg.norm(normal[valid],axis=-1),1)
# Validate the analytic surface gradient with a central difference.
eps=1e-6
f=lambda xx,yy:bottom+depth*((xx/a)**2+(yy/b)**2)
assert np.allclose((f(X+eps,Y)-f(X-eps,Y))/(2*eps),gx,atol=1e-8)
assert np.allclose((f(X,Y+eps)-f(X,Y-eps))/(2*eps),gy,atol=1e-8)

summaries=[]
for seed in [42,123,999]:
    rng=np.random.default_rng(seed); dirt=np.zeros_like(X); patches=[]
    for _ in range(3):
        # Restricted example patch on far inner wall; visibility is NOT established.
        cx=float(rng.uniform(-0.075,0.075)); cy=float(rng.uniform(0.04,0.14))
        sx=float(rng.uniform(0.012,0.028)); sy=float(rng.uniform(0.012,0.025))
        blob=np.exp(-0.5*(((X-cx)/sx)**2+((Y-cy)/sy)**2))
        dirt=np.maximum(dirt,blob); patches.append({'center_xy_m':[cx,cy],'sigma_xy_m':[sx,sy]})
    dirt[~valid]=0
    # For future perception experiments these are labels/evaluator data, not policy observations.
    np.savez_compressed(OUT/f'geometry_seed_{seed}.npz',x=X,y=Y,z=Z,valid=valid,normals=normal,dirt=dirt)
    summaries.append({'seed':seed,'patches':patches})
    if seed!=42: continue
    fig=plt.figure(figsize=(12,5.5),facecolor='white')
    ax=fig.add_subplot(121,projection='3d')
    colors=plt.cm.YlOrBr(np.clip(dirt,0,1)); colors[dirt<0.12]=[0.80,0.88,0.92,1]
    ax.plot_surface(X,Y,np.where(valid,Z,np.nan),facecolors=colors,rcount=100,ccount=100,shade=False)
    for xx,yy in [(-.10,.06),(0,.09),(.10,.06),(0,.18)]:
        p=np.array([xx,yy,f(xx,yy)]); n=np.array([-2*depth*xx/a**2,-2*depth*yy/b**2,1.0]); n/=np.linalg.norm(n)
        ax.quiver(*p,*n,length=.08,color='#137d66',arrow_length_ratio=.2)
    ax.scatter(*camera,s=55,color='#2465b3',label='Fixed overhead camera (proposal)')
    ax.plot([camera[0],look_at[0]],[camera[1],look_at[1]],[camera[2],look_at[2]],'--',color='#2465b3')
    # Front chassis footprint, symbolic box outline.
    for zz in [0,.65]:
        ax.plot([-.17,.17,.17,-.17,-.17],[-.82,-.82,-.53,-.53,-.82],[zz]*5,color='#75828c')
    for xx in [-.17,.17]:
        for yy in [-.82,-.53]: ax.plot([xx,xx],[yy,yy],[0,.65],color='#75828c')
    ax.set(xlabel='x [m]',ylabel='y [m]',zlabel='z [m]',title='Fixed chassis / curved bowl / random dirt')
    ax.view_init(elev=28,azim=-48); ax.legend(loc='upper left',fontsize=8)
    ax2=fig.add_subplot(122)
    im=ax2.pcolormesh(X,Y,np.where(valid,tilt,np.nan),shading='auto',cmap='viridis',vmin=0,vmax=65)
    ax2.contour(X,Y,np.where(valid,dirt,np.nan),levels=[.35,.70],colors=['#ffb14a','#d16010'],linewidths=1.8)
    fig.colorbar(im,ax=ax2,label='Surface-normal tilt from vertical [degrees]')
    ax2.set(xlabel='x [m]',ylabel='y [m]',title='Different locations need different tool angles',aspect='equal')
    fig.suptitle('GEOMETRY STUDY ONLY | illustrative bowl, not a real CAD model',fontsize=13)
    fig.text(.5,.02,'Orange contours: random dirt. No visibility / reach / collision / cleaning performance is evaluated.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.05,1,.94]); fig.savefig(OUT/'geometry_study.png',dpi=160); plt.close(fig)

summary={'status':'geometry_only_not_robot_simulation','dimensions_are':'illustrative_not_manufacturer_measurements',
         'bowl_parameters_m':{'half_width':a,'half_length':b,'depth':depth,'bottom_height':bottom},
         'camera_position_m':camera.tolist(),'normal_tilt_range_degrees':[float(tilt[valid].min()),float(tilt[valid].max())],
         'checks':['unit normals','analytic gradient versus finite difference'],
         'not_evaluated':['visibility','camera calibration','arm reach','collisions','physical contact','cleaning','AI'],
         'random_scenes':summaries}
(OUT/'geometry_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
