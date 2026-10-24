"""Map saved camera-driven brush poses onto the official UR5e URDF."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from ur5e_kinematics import UR5e, tool_rotation
from cleaning_model import scene

root = Path(__file__).resolve().parents[1]
out = root/'output'/'ur5e'; out.mkdir(exist_ok=True)
arm = UR5e()
meshes = arm.collision_meshes()
poses = np.loadtxt(root/'output/view3d/tool_poses_xyz_normal.csv', delimiter=',', skiprows=1)
replay, _ = scene(42)
replay_states, actual_tool_frames = [], []


def penetration_screen(transforms):
    """Vertex-only coarse screen. NOT triangle collision or continuous checking."""
    bowl, cabinet = 0, 0
    for mesh in meshes:
        p = mesh['triangles'].reshape(-1, 3)
        t = transforms[mesh['link']]
        p = p @ t[:3, :3].T+t[:3, 3]
        radius = (p[:, 0]/.18)**2+(p[:, 1]/.24)**2
        bowl += int(np.count_nonzero((radius <= 1) & (p[:, 2] >= .16) & (p[:, 2] < .22+.18*radius-.002)))
        cabinet += int(np.count_nonzero((np.abs(p[:, 0]) < .22) & (p[:, 1] > -.75) & (p[:, 1] < -.30) & (p[:, 2] > 0) & (p[:, 2] < .548)))
    return bowl, cabinet


rng = np.random.default_rng(53772)
seeds = [np.array([0., -1.5, 1.5, -1.5, -1.5, 0.])]
seeds += [rng.uniform(-np.pi, np.pi, 6) for _ in range(15)]
candidates = []
for seed in seeds:
    q, pe, ae = arm.ik(poses[0, :3], tool_rotation(poses[0, :3], poses[0, 3:]), seed)
    if pe < .0001 and ae < .001:
        b, c = penetration_screen(arm.transforms(q))
        candidates.append((b+c, np.linalg.norm(q), q))
if not candidates: raise RuntimeError('No valid initial IK solution')
q = min(candidates, key=lambda item: item[:2])[2]
joint_path, errors, screens, frame_transforms = [], [], [], []
for i, pose in enumerate(poses):
    q, pe, ae = arm.ik(pose[:3], tool_rotation(pose[:3], pose[3:]), q)
    if pe > .0001 or ae > .001: raise RuntimeError(f'IK failure at {i}: {pe}, {ae}')
    transforms = arm.transforms(q)
    actual = arm.fk(q)
    if i % 105 == 0:
        replay.update_pose(actual[:3, 3]+[0, 0, .03], -actual[:3, 2])
    replay.update_pose(actual[:3, 3], -actual[:3, 2])
    joint_path.append(q.copy()); errors.append([pe, ae])
    screens.append(penetration_screen(transforms))
    j = i % 105
    if j % 5 == 0 or j == 104:
        frame_transforms.append({mesh['link']:transforms[mesh['link']].tolist() for mesh in meshes})
        replay_states.append(np.flatnonzero(replay.remaining).tolist())
        actual_tool_frames.append(actual)
    if i % 500 == 0: print(f'IK {i}/{len(poses)}', flush=True)

qs = np.asarray(joint_path); errors = np.asarray(errors); screens = np.asarray(screens)
steps = np.abs(np.diff(qs, axis=0))
within = np.array([i % 105 != 0 for i in range(1, len(qs))])
summary = dict(status='kinematics_verified_collision_screen_only', robot='UR5e',
    source_commit='7ce98d8c3512b9d0239da93b75de68ea64f61bd6', base_xyz_m=arm.base[:3, 3].tolist(),
    tool_offset_tool0_m=[0, 0, arm.tool_length], samples=len(qs), actions=len(qs)//105,
    max_position_error_m=float(errors[:, 0].max()), max_orientation_error_deg=float(np.rad2deg(errors[:, 1]).max()),
    all_joint_limits_satisfied=bool(np.all(qs >= arm.lower) and np.all(qs <= arm.upper)),
    max_within_stroke_joint_step_deg=float(np.rad2deg(steps[within]).max()),
    max_between_stroke_joint_jump_deg=float(np.rad2deg(steps[~within]).max()),
    bowl_vertex_penetration_frames=int(np.count_nonzero(screens[:, 0])),
    cabinet_vertex_penetration_frames=int(np.count_nonzero(screens[:, 1])),
    initial_ik_solutions=len(candidates), joint_names=arm.names,
    fk_driven_remaining_cells=int(replay.remaining.sum()),
    limitations=['no self-collision or exact triangle collision validation', 'vertex screening can miss intersections',
                 'between-stroke transfer paths not planned', 'no velocity acceleration torque or physical calibration validation',
                 'default manufacturer kinematics, not a calibrated physical serial number'])
(out/'verification.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
np.savetxt(out/'joint_trajectory_rad.csv', qs, delimiter=',', header=','.join(arm.names), comments='')
np.savez_compressed(out/'motion.npz', joints=qs, errors=errors, vertex_screens=screens)

source = (root/'output/view3d/toilet_cleaning_3d.html').read_text(encoding='utf-8')
start = source.index('const data=')+len('const data='); end = source.index(';\nconst canvas', start)
data = json.loads(source[start:end])
assert len(frame_transforms) == len(data['frames'])
for i, frame in enumerate(data['frames']):
    frame['arm'] = frame_transforms[i]
    actual = actual_tool_frames[i]
    frame['xyz'] = actual[:3, 3].tolist()
    frame['rotation'] = [actual[:3, 0].tolist(), (-actual[:3, 1]).tolist(), (-actual[:3, 2]).tolist()]
    frame['remaining'] = replay_states[i]
    sample = (frame['action']-1)*105 + min((i % 22)*5, 104)
    frame['joints_deg'] = np.rad2deg(qs[sample]).tolist()
data['arm_meshes'] = [dict(link=m['link'], triangles=np.round(m['triangles'], 6).tolist()) for m in meshes]
data['arm_verification'] = summary
template = (root/'scripts/viewer3d.html').read_text(encoding='utf-8')
license_text = (root/'vendor/ur_description/LICENSE').read_text(encoding='utf-8')
notice = 'Universal Robots ROS2 Description, commit '+summary['source_commit']+'\nhttps://github.com/UniversalRobots/Universal_Robots_ROS2_Description\n\n'+license_text
(out/'THIRD_PARTY_LICENSE.txt').write_text(notice, encoding='utf-8')
html = template.replace('__SCENE_DATA__', json.dumps(data, separators=(',', ':')))
html = html.replace('</body>', '')  # template uses implicit body close
html = html.replace('</html>', '<!--\n'+notice+'\n--></html>')
(out/'ur5e_cleaning_3d.html').write_text(html, encoding='utf-8')
print(json.dumps(summary, indent=2))
