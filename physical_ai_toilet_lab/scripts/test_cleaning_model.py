"""Deterministic behavioral checks for the simplified cleaning model."""
import unittest
import numpy as np
from cleaning_model import Cleaner, scene, sweep


def traverse(model, reverse=False, **kwargs):
    xs = np.linspace(-.035, .035, 141)
    for u in xs[::-1] if reverse else xs:
        model.update(float(u), .08, **kwargs)


class CleaningTests(unittest.TestCase):
    def test_exact_n_and_reverse(self):
        for n in [1, 2, 3, 5]:
            m = Cleaner([[0., .08]], [n])
            for i in range(n):
                traverse(m, reverse=bool(i % 2))
                self.assertEqual(m.passes[0], i+1)
                self.assertEqual(bool(m.remaining[0]), i+1 < n)

    def test_stationary_hover_wrong_angle_and_penetration(self):
        for kwargs in [dict(gap=.03), dict(gap=-.03), dict(aligned=False)]:
            m = Cleaner([[0., .08]], [1])
            traverse(m, **kwargs)
            self.assertEqual(m.passes[0], 0)
        m = Cleaner([[0., .08]], [1])
        for _ in range(2000):
            m.update(0., .08)
        m.update(.03, .08)
        self.assertEqual(m.passes[0], 0)

    def test_teleport_and_lift_do_not_complete_pass(self):
        m = Cleaner([[0., .08]], [1])
        m.update(-.03, .08)
        m.update(.03, .08)
        self.assertEqual(m.passes[0], 0)
        for u in np.linspace(-.02, 0, 41):
            m.update(float(u), .08)
        m.update(0., .08, gap=.03)
        self.assertEqual(m.passes[0], 0)

    def test_different_location_is_not_cleaned(self):
        m = Cleaner([[0., .08], [.07, .12]], [1, 1])
        traverse(m)
        self.assertEqual(m.passes.tolist(), [1, 0])

    def test_seed_reproducibility_and_full_raster(self):
        m, p = scene(42)
        other, q = scene(42)
        self.assertEqual(p, q)
        np.testing.assert_equal(m.uv, other.uv)
        self.assertNotEqual(p, scene(123)[1])
        for _ in range(5):
            sweep(m)
        self.assertFalse(m.remaining.any())

    def test_empty_scene(self):
        m = Cleaner([], [])
        traverse(m)
        self.assertFalse(m.remaining.any())


if __name__ == '__main__':
    unittest.main()
