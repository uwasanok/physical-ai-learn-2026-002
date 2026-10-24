import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from toilet_geometry import ToiletGeometry

root=Path(__file__).resolve().parents[1];out=root/'output/geometry_trials';g=ToiletGeometry()
old=json.loads((out/'tool_180mm/summary.json').read_text())
new=json.loads((out/'tool_260mm/summary.json').read_text())
vis=np.load(out/'tool_260mm/visibility.npz')
fig=plt.figure(figsize=(13,9))
ax=fig.add_subplot(221,projection='3d')
ax.add_collection3d(Poly3DCollection(g.bowl_mesh(),facecolor='#d4e1e7',edgecolor='#b0c3cd',linewidth=.1,alpha=.30))
ax.add_collection3d(Poly3DCollection(g.rim_mesh(),facecolor='#f0e8d3',edgecolor='#b4a684',linewidth=.1,alpha=.80))
model,_=g.make_scene(42);ax.scatter(*model.xyz.T,c='#915025',s=8)
ax.set(xlim=(-.22,.22),ylim=(-.28,.28),zlim=(.15,.45),xlabel='x [m]',ylabel='y [m]',zlabel='z [m]',title='Flatter bottom / rising inner wall / overhanging rim')
ax.set_box_aspect((.44,.56,.30));ax.view_init(elev=30,azim=-55)
ax2=fig.add_subplot(222);x=np.linspace(-.18,.18,300)
ax2.plot(x,.22+.18*(x/.18)**2,label='Previous paraboloid',linestyle='--',color='#8194a1')
ax2.plot(x,g.surface(x,0)[:,2],label='New inner profile',color='#188e7b',linewidth=2)
for sign in [-1,1]:ax2.fill_between(sign*np.array([.18*.94,.18*1.14]),.395,.425,color='#d7caa2')
ax2.set(title='Cross-section at y = 0',xlabel='x [m]',ylabel='z [m]');ax2.legend()
ax3=fig.add_subplot(223);xyz=vis['xyz'];visible=vis['visible']
ax3.scatter(xyz[~visible,0],xyz[~visible,1],s=3,color='#b87f7c',label='Occluded')
ax3.scatter(xyz[visible,0],xyz[visible,1],s=3,color='#b8d7e4',label='Visible')
ax3.plot([-.11,.11,.11,-.11,-.11],[.025,.025,.155,.155,.025],color='#087962',linewidth=2,label='Fixed cleaning ROI')
ax3.set(aspect='equal',xlabel='u [m]',ylabel='v [m]',title='Camera vs bowl/rim visibility (arm not included)');ax3.legend(fontsize=8)
ax4=fig.add_subplot(224);ax4.axis('off')
rows=[]
for a,b in zip(old['runs'],new['runs']):rows.append([a['seed'],'Rim interference' if a['status']=='rejected' else a['status'],b['status'],b.get('fk_remaining_cells','-')])
table=ax4.table(cellText=rows,colLabels=['Seed','180 mm tool','260 mm tool','Dirt cells left'],loc='center',cellLoc='center',colWidths=[.13,.34,.25,.23]);table.auto_set_font_size(False);table.set_fontsize(9);table.scale(1,2)
ax4.set_title('Same geometry / ROI / dirt; tool length comparison')
fig.suptitle('DEVELOPMENT STUDY | illustrative toilet geometry, not manufacturer CAD',fontsize=14)
fig.text(.5,.015,'5 development seeds only. Sampled collision checking; no force simulation or learned perception yet.',ha='center')
fig.tight_layout(rect=[0,.04,1,.95]);fig.savefig(out/'geometry_and_trials.png',dpi=150)
print(json.dumps({'baseline':[r['status'] for r in old['runs']],'candidate':[r['status'] for r in new['runs']]},indent=2))
