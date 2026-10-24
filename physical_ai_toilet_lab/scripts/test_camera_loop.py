import unittest
import numpy as np
from cleaning_model import scene, surface
from camera_loop import Camera, Environment, detect, plan, run_loop


class CameraTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.camera = Camera(320, 240)

    def test_projection_roundtrip(self):
        for u in [-.09, 0., .09]:
            for v in [.04, .09, .14]:
                xyz = surface(np.array(u), np.array(v))
                pixel = self.camera.project(xyz)
                restored, valid = self.camera.unproject(*pixel)
                self.assertTrue(valid)
                np.testing.assert_allclose(xyz, restored, atol=1e-10)

    def test_different_hidden_n_same_observation_same_action(self):
        first, _ = scene(42); second, _ = scene(42)
        second.required[:] = 100
        rgb1 = Environment(first, self.camera).capture()
        rgb2 = Environment(second, self.camera).capture()
        np.testing.assert_equal(rgb1, rgb2)
        np.testing.assert_equal(plan(rgb1, self.camera)[0], plan(rgb2, self.camera)[0])

    def test_repeat_cleaning_and_different_positions(self):
        for seed in [42, 123, 999]:
            model, _ = scene(seed)
            result, _, _ = run_loop(Environment(model, self.camera))
            self.assertEqual(result['status'], 'observed_roi_clear')
            self.assertEqual(result['evaluation_residual_area'], 0.)
            self.assertGreater(result['actions'], 1)
            self.assertFalse(result['evaluation_false_clear'])

    def test_hard_stain_budget_does_not_claim_success(self):
        model, _ = scene(42); model.required[:] = 100
        result, _, _ = run_loop(Environment(model, self.camera), max_actions=2)
        self.assertEqual(result['status'], 'budget_exhausted')
        self.assertEqual(result['evaluation_residual_area'], 1.)

    def test_no_dirt_and_missing_frame(self):
        model, _ = scene(42); model.passes[:] = model.required
        result, _, _ = run_loop(Environment(model, self.camera))
        self.assertEqual(result['actions'], 0)
        self.assertEqual(result['status'], 'observed_roi_clear')
        with self.assertRaises(ValueError): detect(None)
        env = Environment(model, self.camera)
        env.capture = lambda: None
        result, _, _ = run_loop(env)
        self.assertEqual(result['status'], 'invalid_observation')


if __name__ == '__main__': unittest.main()
