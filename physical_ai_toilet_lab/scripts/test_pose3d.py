import unittest
import numpy as np
from cleaning_model import Cleaner, surface, normal


class Pose3DTests(unittest.TestCase):
    def test_actual_z_and_orientation_control_contact(self):
        for offset, flipped, expected in [(0, False, 1), (.025, False, 0), (-.025, False, 0), (0, True, 0)]:
            m = Cleaner([[.06, .10]], [1])
            for x in np.linspace(.025, .095, 141):
                p = surface(x, .10) + [0, 0, offset]
                axis = normal(x, .10) * (-1 if flipped else 1)
                m.update_pose(p, axis)
            self.assertEqual(m.passes[0], expected)

    def test_flat_xy_path_fails_curved_surface(self):
        m = Cleaner([[.08, .10]], [1])
        fixed_z = float(surface(0., .10)[2])
        for x in np.linspace(.045, .115, 141):
            m.update_pose([x, .10, fixed_z], normal(x, .10))
        self.assertEqual(m.passes[0], 0)

    def test_invalid_pose_rejected(self):
        m = Cleaner([[0, .08]], [1])
        with self.assertRaises(ValueError): m.update_pose([0, 0, np.nan], [0, 0, 1])
        with self.assertRaises(ValueError): m.update_pose([0, 0, .22], [0, 0, 0])


if __name__ == '__main__': unittest.main()
