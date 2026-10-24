import unittest
import numpy as np
from ur5e_kinematics import UR5e


class UR5eTests(unittest.TestCase):
    def setUp(self): self.arm = UR5e()

    def test_zero_configuration_matches_published_link_offsets(self):
        # Independent arithmetic from official default_kinematics offsets.
        arm = UR5e(base=(0,0,0), tool_length=0)
        np.testing.assert_allclose(arm.fk(np.zeros(6))[:3,3], [.425+.3922, .1333+.0996, .1625-.0997], atol=1e-8)

    def test_ik_fk_round_trip(self):
        q = np.array([.3,-1.3,1.7,-1.2,.8,-.4])
        target = self.arm.fk(q)
        solved, pe, ae = self.arm.ik(target[:3,3], target[:3,:3], q+.05)
        self.assertLess(pe, 1e-6); self.assertLess(ae, 1e-5)
        self.assertTrue(np.all(solved >= self.arm.lower) and np.all(solved <= self.arm.upper))

    def test_unreachable_target_is_not_success(self):
        _, pe, _ = self.arm.ik(np.array([10.,10.,10.]), np.eye(3), np.zeros(6))
        self.assertGreater(pe, 1.)

    def test_saved_motion_limits_and_discontinuities(self):
        from ur5e_kinematics import ROOT
        data = np.load(ROOT/'output/ur5e/motion.npz')
        self.assertEqual(len(data['joints']), 3360)
        self.assertTrue(np.all(data['joints'] >= self.arm.lower))
        self.assertTrue(np.all(data['joints'] <= self.arm.upper))
        self.assertLess(data['errors'][:,0].max(), .0001)
        self.assertLess(data['errors'][:,1].max(), .001)


if __name__ == '__main__': unittest.main()
