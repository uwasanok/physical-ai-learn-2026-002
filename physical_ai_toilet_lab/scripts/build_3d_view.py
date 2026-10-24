"""Replay actual xyz tool poses and dirt states in a standalone orbit viewer."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cleaning_model import scene, surface, normal
from camera_loop import Camera, Environment, plan

root = Path(__file__).resolve().parents[1]
out = root/'output'/'view3d'; out.mkdir(exist_ok=True)
model, _ = scene(42)
env = Environment(model, Camera())
frames = []
all_poses = []
for action in range(100):
    target, pixels = plan(env.capture(), env.camera)
    if target is None: break
    u, v = target[:2]
    model.update_pose(surface(u-.026, v)+[0, 0, .03], normal(u-.026, v))
    for index, x in enumerate(np.linspace(u-.026, u+.026, 105)):
        xyz, axis = surface(x, v), normal(x, v)
        # Tool frame: X tangent in increasing u, Z away from surface, Y = Z x X.
        tangent = np.array([1., 0., 2*.18*x/.18**2]); tangent /= np.linalg.norm(tangent)
        rotation = np.column_stack([tangent, np.cross(axis, tangent), axis])
        assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-10)
        assert np.isclose(np.linalg.det(rotation), 1.)
        model.update_pose(xyz, axis)
        all_poses.append([*xyz, *axis])
        if index % 5 == 0 or index == 104:
            frames.append(dict(action=action+1, xyz=xyz.tolist(), rotation=rotation.T.tolist(),
                               remaining=np.flatnonzero(model.remaining).tolist()))
    model.update_pose(surface(u+.026, v)+[0, 0, .03], normal(u+.026, v))

poses = np.asarray(all_poses)
metrics = dict(coordinates='world xyz metres; tool axes are rotation columns', seed=42,
               cleaning_actions=action, pose_samples=len(poses),
               xyz_min_m=poses[:, :3].min(0).tolist(), xyz_max_m=poses[:, :3].max(0).tolist(),
               max_surface_error_m=float(np.abs(poses[:, 2]-surface(poses[:, 0], poses[:, 1])[:, 2]).max()),
               tilt_min_max_deg=[float(np.rad2deg(np.arccos(poses[:, 5])).min()), float(np.rad2deg(np.arccos(poses[:, 5])).max())],
               remaining_cells=int(model.remaining.sum()),
               limitations=['height-field bowl, not manufacturer mesh', 'ideal tool motion, no robot IK',
                            'simplified geometric contact', 'transfers between strokes omitted from animation'])
np.savetxt(out/'tool_poses_xyz_normal.csv', poses, delimiter=',', header='x_m,y_m,z_m,nx,ny,nz', comments='')
(out/'verification.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
data = dict(dirt=model.xyz.tolist(), frames=frames, metrics=metrics)
template = (root/'scripts'/'viewer3d.html').read_text(encoding='utf-8')
(out/'toilet_cleaning_3d.html').write_text(template.replace('__SCENE_DATA__', json.dumps(data, separators=(',', ':'))), encoding='utf-8')

# Standalone scientific preview with equal coordinate scale and a true 3D path.
fig = plt.figure(figsize=(12, 6)); ax = fig.add_subplot(121, projection='3d')
r, t = np.meshgrid(np.linspace(0, 1, 22), np.linspace(0, 2*np.pi, 65))
x, y = .18*r*np.cos(t), .24*r*np.sin(t); z = surface(x, y)[..., 2]
ax.plot_surface(x, y, z, color='#d3e1ea', alpha=.35, edgecolor='#b5c4cc', linewidth=.2)
ax.scatter(*model.xyz.T, c='#98531f', s=8, label='Initial dirt on curved surface')
ax.plot(*poses[:105, :3].T, color='#e0366d', linewidth=3, label='First 3D wiping stroke')
for i in [0, 52, 104]: ax.quiver(*poses[i, :3], *poses[i, 3:], length=.06, color='#07836b')
ax.set(xlabel='World x [m]', ylabel='World y [m]', zlabel='World z [m]', title='3D geometry / trajectory / tool orientation')
ax.set_box_aspect((.36, .48, .25)); ax.view_init(elev=28, azim=-55); ax.legend(fontsize=8)
ax2 = fig.add_subplot(222)
ax2.plot(np.arange(len(poses)), poses[:, 2]*1000, color='#bb2c60')
ax2.set(title='Actual tool height changes with the curved surface', ylabel='World z [mm]', xlabel='Contact pose sample')
ax3 = fig.add_subplot(224)
ax3.plot(np.arange(len(poses)), np.rad2deg(np.arccos(poses[:, 5])), color='#07836b')
ax3.set(ylabel='Tool tilt from vertical [deg]', xlabel='Contact pose sample')
fig.suptitle('3D TOOL-POSE BASELINE | synthetic bowl, no articulated robot or force simulation')
fig.tight_layout(); fig.savefig(out/'3d_verification.png', dpi=150)
print(json.dumps(metrics, indent=2))
