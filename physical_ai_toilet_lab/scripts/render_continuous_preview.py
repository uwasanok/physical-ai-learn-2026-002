from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from ur5e_kinematics import UR5e
from cleaning_model import surface

root=Path(__file__).resolve().parents[1];out=root/'output/continuous'
d=np.load(out/'motion.npz');arm=UR5e();meshes=arm.collision_meshes()
phase_colors={'clean':'#e43b69','retract':'#f0a332','transfer':'#497ad3','approach':'#19a487'}
# Show first switch between different stains (action 6); earlier actions repeat locally.
indices=np.flatnonzero((d['actions']>=5)&(d['actions']<=7))
fig=plt.figure(figsize=(13,6));ax=fig.add_subplot(121,projection='3d')
sample=int(indices[len(indices)//2]);tf=arm.transforms(d['joints'][sample])
for mesh in meshes:
    t=tf[mesh['link']];points=mesh['triangles']@t[:3,:3].T+t[:3,3]
    n=np.cross(points[:,1]-points[:,0],points[:,2]-points[:,0]);n/=np.maximum(np.linalg.norm(n,axis=1,keepdims=True),1e-12)
    c=(.6+.4*np.abs(n[:,2]))[:,None]*np.array(to_rgb('#6c9eb9'))
    ax.add_collection3d(Poly3DCollection(points,facecolor=c,edgecolor='none'))
r,t=np.meshgrid(np.linspace(0,1,20),np.linspace(0,2*np.pi,65));x=.18*r*np.cos(t);y=.24*r*np.sin(t)
ax.plot_surface(x,y,surface(x,y)[...,2],color='#dde8ec',alpha=.3)
for phase,color in phase_colors.items():
    ids=indices[d['phases'][indices]==phase];p=np.array([arm.fk(q)[:3,3] for q in d['joints'][ids]])
    ax.scatter(*p.T,color=color,s=2,label=phase)
ax.set(xlim=(-.35,.35),ylim=(-.75,.30),zlim=(.15,1.1),xlabel='x [m]',ylabel='y [m]',zlabel='z [m]',title='Tool retreat / transfer / approach (actions 5-7)')
ax.set_box_aspect((.7,1.05,.95));ax.view_init(elev=25,azim=-55);ax.legend(fontsize=8)
ax2=fig.add_subplot(222)
for j in range(6):ax2.plot(d['time_s'],np.rad2deg(d['joints'][:,j]),label=f'J{j+1}',linewidth=.8)
ax2.set(xlabel='Planned time [s]',ylabel='Joint angle [deg]',title='No pose jumps at cleaning transitions');ax2.legend(ncol=3,fontsize=7)
ax3=fig.add_subplot(224);steps=np.rad2deg(np.abs(np.diff(d['joints'],axis=0))).max(1)
ax3.plot(d['time_s'][1:],steps,color='#278b83',linewidth=.8);ax3.axhline(.25,color='#c54863',linestyle='--')
ax3.set(xlabel='Planned time [s]',ylabel='Max joint step [deg]',ylim=(0,.3),title='FCL checked every saved configuration')
fig.suptitle('32 cleaning strokes + 31 transfers | sampled mesh collision validation',fontsize=14)
fig.text(.5,.02,'Simplified geometry and contact exceptions. Not continuous-time collision proof or real robot validation.',ha='center',fontsize=10)
fig.tight_layout(rect=[0,.05,1,.95]);fig.savefig(out/'continuous_preview.png',dpi=150)
