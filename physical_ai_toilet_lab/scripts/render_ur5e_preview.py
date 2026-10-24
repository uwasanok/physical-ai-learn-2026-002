from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from ur5e_kinematics import UR5e
from cleaning_model import surface, scene

root = Path(__file__).resolve().parents[1]
out = root/'output/ur5e'
arm = UR5e(); meshes = arm.collision_meshes()
qs = np.load(out/'motion.npz')['joints']
dirt, _ = scene(42)
fig = plt.figure(figsize=(13, 6.5))
for panel, sample in enumerate([0, 2100]):
    ax = fig.add_subplot(1,2,panel+1,projection='3d')
    transforms = arm.transforms(qs[sample])
    for mesh in meshes:
        t = transforms[mesh['link']]
        pts = mesh['triangles'] @ t[:3,:3].T + t[:3,3]
        color = '#aabdc9' if 'arm' in mesh['link'] else '#5288b5'
        normals = np.cross(pts[:,1]-pts[:,0],pts[:,2]-pts[:,0])
        normals /= np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-12)
        shade = .55+.45*np.abs(normals @ np.array([.3,.4,.866]))
        colors = shade[:,None]*np.array(to_rgb(color))
        ax.add_collection3d(Poly3DCollection(pts, facecolor=colors, edgecolor='none', zsort='average'))
    r, theta = np.meshgrid(np.linspace(0,1,20),np.linspace(0,2*np.pi,65))
    x,y = .18*r*np.cos(theta),.24*r*np.sin(theta)
    ax.plot_surface(x,y,surface(x,y)[...,2],color='#d8e4e8',alpha=.35,edgecolor='#bbccd2',linewidth=.15)
    ax.scatter(*dirt.xyz.T,color='#97531f',s=6)
    tip = arm.fk(qs[sample])[:3,3]; flange = transforms['tool0'][:3,3]
    ax.plot(*np.array([tip,flange]).T,color='#16967a',linewidth=5)
    ax.scatter(*tip,color='#ed3265',s=20)
    for z in [0,.55]:
        ax.plot([-.22,.22,.22,-.22,-.22],[-.75,-.75,-.30,-.30,-.75],[z]*5,color='#6e8291')
    for x in [-.22,.22]:
        for y in [-.75,-.30]:ax.plot([x,x],[y,y],[0,.55],color='#6e8291')
    ax.scatter(0,-.65,1.,color='#263a52',marker='s',s=35)
    ax.set(xlim=(-.5,.5),ylim=(-.85,.3),zlim=(0,1.2),xlabel='x [m]',ylabel='y [m]',zlabel='z [m]',title=f'UR5e official collision meshes | stroke {sample//105+1}')
    ax.set_box_aspect((1,1.15,1.2)); ax.view_init(elev=24,azim=35 if panel==0 else -55)
fig.suptitle('UR5e + 180 mm brush | bounded IK, fixed pedestal | geometry feasibility only',fontsize=14)
fig.text(.5,.025,'Dirt shown at initial positions for context. No self-collision / transfer path / dynamics validation.',ha='center')
fig.tight_layout(rect=[0,.04,1,.95]); fig.savefig(out/'ur5e_preview.png',dpi=150)
