"""Synthetic RGB camera -> known-surface intersection -> repeat cleaning.

Only Environment sees true dirt and required pass counts. Planner takes RGB
and a calibrated camera/known bowl model. No learned perception or robot IK.
"""
import numpy as np
from cleaning_model import normal, surface, residual_area_fraction


class Camera:
    def __init__(self, width=640, height=480):
        self.width, self.height = width, height
        self.origin = np.array([0., -.65, 1.])
        forward = np.array([0., .035, .30])-self.origin
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, [0., 0., 1.]); right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        self.rotation = np.column_stack([right, down, forward])
        self.focal = 840. * width/640
        yy, xx = np.indices((height, width))
        self.points, self.valid = self.unproject(xx, yy)

    def project(self, xyz):
        local = (np.asarray(xyz)-self.origin) @ self.rotation
        return local[..., :2]/local[..., 2, None]*self.focal + [self.width/2, self.height/2]

    def unproject(self, px, py):
        """Pinhole ray intersected with calibrated known paraboloid, nearest hit."""
        px, py = np.broadcast_arrays(px, py)
        rays = np.stack([(px-self.width/2)/self.focal, (py-self.height/2)/self.focal, np.ones_like(px)], -1) @ self.rotation.T
        ox, oy, oz = self.origin
        dx, dy, dz = np.moveaxis(rays, -1, 0)
        ku, kv = .18/.18**2, .18/.24**2
        a = ku*dx*dx+kv*dy*dy
        b = 2*ku*ox*dx+2*kv*oy*dy-dz
        c = .22+ku*ox*ox+kv*oy*oy-oz
        disc = b*b-4*a*c
        roots = [(-b-np.sqrt(np.maximum(disc, 0)))/(2*a), (-b+np.sqrt(np.maximum(disc, 0)))/(2*a)]
        best = np.full(px.shape, np.inf)
        for t in roots:
            hit = self.origin + t[..., None]*rays
            inside = (hit[..., 0]/.18)**2+(hit[..., 1]/.24)**2 <= 1
            ok = (disc >= 0) & (t > 0) & inside
            best = np.minimum(best, np.where(ok, t, np.inf))
        valid = np.isfinite(best)
        points = self.origin + np.where(valid, best, 0)[..., None]*rays
        return points, valid


class Environment:
    def __init__(self, model, camera):
        self.model, self.camera = model, camera
        # Surface texture lookup is private simulator state, never planner input.
        self.lookup = np.full((34, 56), -1, dtype=int)
        ii = np.rint((model.uv[:, 0]+.11)/(.22/55)).astype(int)
        jj = np.rint((model.uv[:, 1]-.025)/(.13/33)).astype(int)
        if len(ii): self.lookup[jj, ii] = np.arange(len(ii))
        p = camera.points
        i = np.rint((p[..., 0]+.11)/(.22/55)).astype(int)
        j = np.rint((p[..., 1]-.025)/(.13/33)).astype(int)
        ok = camera.valid & (i >= 0) & (i < 56) & (j >= 0) & (j < 34)
        self.cell_id = np.full(i.shape, -1, dtype=int)
        self.cell_id[ok] = self.lookup[j[ok], i[ok]]

    def capture(self):
        rgb = np.full((self.camera.height, self.camera.width, 3), [40, 52, 65], dtype=np.uint8)
        n = self.model.normal_fn(self.camera.points[..., 0], self.camera.points[..., 1])
        shade = .72+.28*n[..., 2]
        porcelain = (shade[..., None]*np.array([230, 237, 240])).astype(np.uint8)
        rgb[self.camera.valid] = porcelain[self.camera.valid]
        ids = self.cell_id
        mask = ids >= 0
        if len(self.model.remaining):
            mask &= self.model.remaining[np.maximum(ids, 0)]
        else:
            mask[:] = False
        # All n values look the same: required passes cannot be inferred by color.
        rgb[mask] = [114, 57, 24]
        return rgb

    def execute(self, target):
        u, v = target[:2]
        self.model.update(u-.026, v, gap=.03)
        trace = []
        for x in np.linspace(u-.026, u+.026, 105):
            self.model.update_pose(self.model.surface_fn(x, v), self.model.normal_fn(x, v))
            trace.append([*self.model.surface_fn(x, v).tolist(), *self.model.normal_fn(x, v).tolist()])
        self.model.update(u+.026, v, gap=.03)
        return trace


def detect(rgb):
    """Deliberately simple, untrained brown threshold baseline."""
    if rgb is None or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError('Missing or invalid RGB frame')
    r, g, b = np.moveaxis(rgb.astype(float), -1, 0)
    return (r > 65) & (r < 170) & (r > 1.35*g) & (g > 1.25*b)


def plan(rgb, camera):
    mask = detect(rgb)
    return plan_mask(mask, camera)


def plan_mask(mask, camera):
    """Only an observed mask and calibrated geometry; no simulator truth."""
    if mask.shape != (camera.height,camera.width):raise ValueError('Mask/camera resolution mismatch')
    yy, xx = np.where(mask)
    if not len(xx): return None, int(mask.sum())
    xyz, valid = camera.unproject(xx, yy)
    # Declared cleaning ROI, not selected using true dirt positions.
    allowed = valid & (np.abs(xyz[:, 0]) <= .11) & (xyz[:, 1] >= .025) & (xyz[:, 1] <= .155)
    xyz = xyz[allowed]
    if not len(xyz): return None, int(mask.sum())
    # Pick a densely observed small region; never average separate stains together.
    bins = np.floor(xyz[:, :2]/.008).astype(int)
    keys, inverse, counts = np.unique(bins, axis=0, return_inverse=True, return_counts=True)
    local = xyz[inverse == np.argmax(counts)]
    uv = np.median(local[:, :2], axis=0)
    return getattr(camera, 'surface_fn', surface)(*uv), int(mask.sum())


def run_loop(env, max_actions=100, keep_frames=False):
    records, frames, trajectories = [], [], []
    status = 'budget_exhausted'
    for step in range(max_actions+1):
        rgb = env.capture()
        try:
            target, pixels = plan(rgb, env.camera)
        except ValueError:
            status = 'invalid_observation'; break
        if keep_frames: frames.append(rgb)
        record = dict(step=step, detected_pixels=pixels, target_xyz_m=None if target is None else target.tolist())
        # Evaluation-only truth: never read by plan(), or used to stop/choose actions.
        record['evaluation_residual_area'] = residual_area_fraction(env.model)
        records.append(record)
        if target is None:
            status = 'observed_roi_clear' if pixels == 0 else 'detected_outside_cleaning_roi'
            break
        if step == max_actions: break
        trajectories.append(env.execute(target))
    return dict(status=status, actions=len(trajectories), observations=records,
                evaluation_residual_area=residual_area_fraction(env.model),
                evaluation_false_clear=status == 'observed_roi_clear' and bool(env.model.remaining.any())), frames, trajectories
