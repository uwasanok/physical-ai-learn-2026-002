import json,unittest
import numpy as np
from pathlib import Path
from ur5e_kinematics import UR5e
from collision_world import CollisionWorld
from toilet_geometry import ToiletGeometry

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/closed_loop_v1'
class ClosedLoopTests(unittest.TestCase):
    def test_three_unseen_scenes_and_negative_cases(self):
        runs=json.loads((OUT/'summary.json').read_text())
        self.assertEqual(len(runs),5)
        for r in runs[:3]:
            self.assertEqual(r['status'],'observed_roi_clear');self.assertEqual(r['remaining_cells'],0)
            self.assertFalse(r['false_clear']);self.assertGreater(r['actions'],1)
            self.assertEqual(r['required_passes_values'],[1,3,5])
        self.assertEqual(runs[3]['actions'],0);self.assertEqual(runs[3]['status'],'observed_roi_clear')
        self.assertEqual(runs[4]['status'],'budget_exhausted');self.assertEqual(runs[4]['actions'],3)
        self.assertGreater(runs[4]['remaining_cells'],0);self.assertFalse(runs[4]['false_clear'])
    def test_hidden_n_does_not_change_initial_rgb_or_action(self):
        from PIL import Image
        a=OUT/'random_410001';b=OUT/'hard_budget'
        np.testing.assert_array_equal(np.array(Image.open(a/'capture_000.png')),np.array(Image.open(b/'capture_000.png')))
        x=json.loads((a/'result.json').read_text());y=json.loads((b/'result.json').read_text())
        self.assertEqual(x['records'][0]['target'],y['records'][0]['target'])
    def test_saved_motion_against_collision_world(self):
        arm=UR5e(tool_length=.26);world=CollisionWorld(arm,ToiletGeometry())
        for run in json.loads((OUT/'summary.json').read_text()):
            d=np.load(OUT/run['name']/'motion.npz');q=d['joints']
            self.assertTrue(np.all((q>=arm.lower)&(q<=arm.upper)))
            if len(q)>1:self.assertLessEqual(np.abs(np.diff(q,axis=0)).max(),np.deg2rad(.25)+1e-9)
            for joints,phase in zip(q,d['phases']):
                self.assertFalse(world.check(joints,str(phase)))
    def test_model_remains_frozen(self):
        import hashlib
        protocol=json.loads((OUT/'protocol.json').read_text())
        digest=hashlib.sha256((ROOT/'output/learning_v1/best_model.pt').read_bytes()).hexdigest()
        self.assertEqual(protocol['model_sha256'],digest)

if __name__=='__main__':unittest.main()
