import unittest
import numpy as np
from ur5e_kinematics import UR5e,ROOT
from collision_world import CollisionWorld
from toilet_geometry import ToiletGeometry


class GeometryTrialTests(unittest.TestCase):
    def test_five_scene_paths_against_new_geometry(self):
        arm=UR5e(tool_length=.26);g=ToiletGeometry();world=CollisionWorld(arm,g)
        for seed in [42,123,999,2026,53772]:
            data=np.load(ROOT/f'output/geometry_trials/tool_260mm/motion_{seed}.npz')
            q=data['joints'];phases=data['phases'];model,_=g.make_scene(seed)
            self.assertLessEqual(np.max(np.abs(np.diff(q,axis=0))),np.deg2rad(.25)+1e-10)
            for i,(joints,phase) in enumerate(zip(q,phases)):
                self.assertEqual(world.check(joints,str(phase)),[],f'seed {seed} sample {i}')
                tip=arm.fk(joints)
                if phase=='clean':model.update_pose(tip[:3,3],-tip[:3,2])
                else:model.inside[:]=False;model.travel[:]=0.;model.previous=None
            self.assertFalse(model.remaining.any(),f'seed {seed} leaves dirt')


if __name__=='__main__':unittest.main()
