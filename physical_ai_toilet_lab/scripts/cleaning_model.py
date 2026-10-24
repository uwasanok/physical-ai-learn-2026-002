"""Surface-coordinate cleaning surrogate; no physics, camera, robot or learned AI."""
import numpy as np


def surface(u, v):
    return np.stack(np.broadcast_arrays(u, v, .22 + .18*((u/.18)**2+(v/.24)**2)), axis=-1)


def normal(u, v):
    n = np.stack(np.broadcast_arrays(-2*.18*u/.18**2, -2*.18*v/.24**2, np.ones_like(u)), axis=-1)
    return n / np.linalg.norm(n, axis=-1, keepdims=True)


class Cleaner:
    # Illustrative geometric tolerances; not calibrated physical contact parameters.
    radius = .012
    gap_limit = .004
    max_step = .004
    min_travel = .006

    def __init__(self, uv, required, surface_fn=surface, normal_fn=normal):
        self.surface_fn, self.normal_fn = surface_fn, normal_fn
        self.uv = np.asarray(uv, dtype=float).reshape(-1, 2)
        self.xyz = surface_fn(self.uv[:, 0], self.uv[:, 1])
        self.normals = normal_fn(self.uv[:, 0], self.uv[:, 1])
        self.required = np.asarray(required, dtype=int).reshape(-1)
        if len(self.required) != len(self.uv) or np.any(self.required < 1):
            raise ValueError('Each surface cell must have a positive required pass count')
        self.passes = np.zeros(len(self.uv), dtype=int)
        self.inside = np.zeros(len(self.uv), dtype=bool)
        self.travel = np.zeros(len(self.uv))
        self.previous = None

    @property
    def remaining(self):
        # Binary threshold: no visible fading before pass n. Forces re-observation.
        return self.passes < self.required

    def update(self, u, v, gap=0., aligned=True):
        """Compatibility adapter. New callers should submit a real xyz pose."""
        point = self.surface_fn(np.asarray(u), np.asarray(v)) + np.array([0., 0., gap])
        axis = self.normal_fn(np.asarray(u), np.asarray(v)) if aligned else -self.normal_fn(np.asarray(u), np.asarray(v))
        self.update_pose(point, axis)

    def update_pose(self, point, tool_axis):
        """Brush-tip xyz in metres and unit axis pointing away from the surface.

        Actual xyz and orientation determine contact; no caller-supplied contact
        flag. Vertical projection is specific to this height-field surface.
        """
        point = np.asarray(point, dtype=float)
        axis = np.asarray(tool_axis, dtype=float)
        if point.shape != (3,) or axis.shape != (3,) or not np.isfinite(point).all() or not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-12:
            raise ValueError('Finite xyz and nonzero tool axis are required')
        axis = axis / np.linalg.norm(axis)
        u, v = point[:2]
        on_surface = self.surface_fn(np.asarray(u), np.asarray(v))
        gap = float(np.linalg.norm(point-on_surface))
        delta = point - self.xyz
        separation = np.linalg.norm(delta, axis=1)
        alignment = self.normals @ axis
        local_alignment = float(self.normal_fn(np.asarray(u), np.asarray(v)) @ axis)
        eligible = gap <= self.gap_limit and local_alignment >= np.cos(np.deg2rad(25)) and (u/.18)**2+(v/.24)**2 <= 1
        step = 0. if self.previous is None else float(np.linalg.norm(point-self.previous))
        if not eligible or step > self.max_step:
            self.inside[:] = False
            self.travel[:] = 0.
            self.previous = point if eligible else None
            return
        covered = (separation <= self.radius) & (alignment >= np.cos(np.deg2rad(25)))
        # Only motion with both consecutive samples inside the footprint contributes.
        self.travel[covered & self.inside] += step
        leaving = self.inside & ~covered
        completed = leaving & (self.travel >= self.min_travel)
        self.passes[completed] += 1
        self.travel[~covered] = 0.
        self.inside = covered
        self.previous = point


def scene(seed):
    """Three random ellipses in a restricted illustrative bowl ROI, n=1,3,5."""
    rng = np.random.default_rng(seed)
    u, v = np.meshgrid(np.linspace(-.11, .11, 56), np.linspace(.025, .155, 34))
    required = np.zeros_like(u, dtype=int)
    patches = []
    # Distinct horizontal bands prevent overlapping stains from hiding n labels.
    for cx, n in zip([-.065, 0., .065], rng.permutation([1, 3, 5]).tolist()):
        cx += rng.uniform(-.008, .008)
        cy = rng.uniform(.055, .125)
        rx, ry = rng.uniform(.012, .020, 2)
        mask = ((u-cx)/rx)**2+((v-cy)/ry)**2 <= 1
        required[mask] = n
        patches.append(dict(center_uv_m=[float(cx), float(cy)], radius_uv_m=[float(rx), float(ry)], required_passes=n))
    dirty = required > 0
    return Cleaner(np.column_stack([u[dirty], v[dirty]]), required[dirty]), patches


def sweep(model):
    """One raster round; oracle/geometry baseline, no camera-based planner."""
    # Each cell is close to at least one row; some can receive more than one pass
    # per round. n refers to local passes, NOT global raster rounds.
    for v in np.arange(.02, .161, .008):
        model.update(-.12, v, gap=.03)  # lifted reposition, never counted
        for u in np.linspace(-.12, .12, 241):
            model.update(float(u), float(v))


def residual_area_fraction(model):
    # Uniform uv cells have curved area proportional to sqrt(1+du_z²+dv_z²).
    weights = 1. / model.normals[:, 2]
    return float(weights[model.remaining].sum()/weights.sum()) if len(weights) else 0.
