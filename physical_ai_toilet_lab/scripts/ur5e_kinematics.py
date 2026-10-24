"""URDF-derived transforms and bounded numerical IK. No physical robot driver."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]


def origin(element):
    t = np.eye(4)
    if element is not None:
        t[:3, 3] = np.fromstring(element.get('xyz', '0 0 0'), sep=' ')
        t[:3, :3] = Rotation.from_euler('xyz', np.fromstring(element.get('rpy', '0 0 0'), sep=' ')).as_matrix()
    return t


class UR5e:
    def __init__(self, base=(0., -.52, .55), tool_length=.18):
        self.root = ET.parse(ROOT/'output/ur5e/ur5e.urdf').getroot()
        self.base = np.eye(4); self.base[:3, 3] = base
        self.tool_length = tool_length
        self.joints = []
        self.names, lower, upper = [], [], []
        for j in self.root.findall('joint'):
            active = j.get('type') == 'revolute'
            idx = len(self.names) if active else None
            if active:
                self.names.append(j.get('name'))
                lower.append(float(j.find('limit').get('lower')))
                upper.append(float(j.find('limit').get('upper')))
            axis = np.fromstring(j.find('axis').get('xyz'), sep=' ') if active else None
            self.joints.append((j.find('parent').get('link'), j.find('child').get('link'), origin(j.find('origin')), idx, axis))
        self.lower, self.upper = np.array(lower), np.array(upper)

    def transforms(self, q):
        frames = {'world': self.base}
        pending = list(self.joints)
        while pending:
            progress = False
            for joint in pending[:]:
                parent, child, fixed, idx, axis = joint
                if parent not in frames: continue
                local = fixed.copy()
                if idx is not None:
                    rot = np.eye(4); rot[:3, :3] = Rotation.from_rotvec(axis*q[idx]).as_matrix()
                    local = local @ rot
                frames[child] = frames[parent] @ local
                pending.remove(joint); progress = True
            if not progress: raise ValueError('Disconnected URDF tree')
        return frames

    def fk(self, q):
        t = self.transforms(q)['tool0'].copy()
        t[:3, 3] += t[:3, 2]*self.tool_length
        return t

    def ik(self, xyz, rotation, seed):
        def residual(q):
            actual = self.fk(q)
            return np.r_[actual[:3, 3]-xyz, .2*Rotation.from_matrix(rotation @ actual[:3, :3].T).as_rotvec()]
        result = least_squares(residual, np.clip(seed, self.lower+1e-6, self.upper-1e-6),
                               bounds=(self.lower, self.upper), max_nfev=150,
                               ftol=1e-10, xtol=1e-10, gtol=1e-10)
        actual = self.fk(result.x)
        position_error = float(np.linalg.norm(actual[:3, 3]-xyz))
        angle_error = float(np.linalg.norm(Rotation.from_matrix(rotation @ actual[:3, :3].T).as_rotvec()))
        return result.x, position_error, angle_error

    def collision_meshes(self):
        meshes = []
        for link in self.root.findall('link'):
            for collision in link.findall('collision'):
                mesh = collision.find('geometry/mesh')
                if mesh is None: continue
                path = ROOT/'vendor/ur_description'/mesh.get('filename').split('package://ur_description/')[1]
                binary = path.read_bytes()
                count = int.from_bytes(binary[80:84], 'little')
                dtype = np.dtype([('normal', '<f4', 3), ('vertices', '<f4', (3, 3)), ('attribute', '<u2')])
                tris = np.frombuffer(binary, dtype=dtype, offset=84, count=count)['vertices'].astype(float)
                scale = np.fromstring(mesh.get('scale', '1 1 1'), sep=' ')
                t = origin(collision.find('origin'))
                tris = (tris*scale) @ t[:3, :3].T + t[:3, 3]
                meshes.append(dict(link=link.get('name'), triangles=tris))
        return meshes


def tool_rotation(xyz, outward_normal):
    z = -np.asarray(outward_normal)  # tool0 +Z extends from flange into the brush tip
    x = np.array([1., 0., 0.]); x -= np.dot(x, z)*z; x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])
