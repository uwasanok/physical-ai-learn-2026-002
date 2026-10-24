import unittest
from pathlib import Path
import numpy as np
import fcl
from collision_world import CollisionWorld, mesh_object
from ur5e_kinematics import UR5e, ROOT


class CollisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.arm=UR5e();cls.world=CollisionWorld(cls.arm)
        cls.data=np.load(ROOT/'output/continuous/motion.npz')

    def test_triangle_crossing_without_matching_vertices(self):
        a=mesh_object([[[-1,-1,0],[1,-1,0],[0,1,0]]])
        b=mesh_object([[[0,0,-1],[0,0,1],[2,0,0]]])
        self.assertGreater(fcl.collide(a,b,fcl.CollisionRequest(),fcl.CollisionResult()),0)

    def test_new_obstacle_is_detected(self):
        world=CollisionWorld(self.arm)
        q=self.data['joints'][0]
        tip=self.arm.fk(q)[:3,3]
        world.obstacles['injected_obstacle']=fcl.CollisionObject(fcl.Box(.08,.08,.08),fcl.Transform(tip))
        self.assertTrue(any('injected_obstacle' in pair for pair in world.check(q,'transfer')))

    def test_self_collision_is_detected(self):
        rng=np.random.default_rng(2026)
        found=False
        for _ in range(80):
            q=rng.uniform(-np.pi,np.pi,6)
            hits=self.world.check(q,'transfer')
            if any(all(name in self.world.links for name in hit.split(' / ')) for hit in hits):
                found=True;break
        self.assertTrue(found)

    def test_all_saved_samples_and_no_joint_jumps(self):
        q=self.data['joints'];phases=self.data['phases'];times=self.data['time_s']
        self.assertLessEqual(np.max(np.abs(np.diff(q,axis=0))),np.deg2rad(.25)+1e-10)
        self.assertTrue(np.all(np.diff(times)>0))
        speed=np.abs(np.diff(q,axis=0))/np.diff(times)[:,None]
        self.assertLessEqual(speed.max(),np.deg2rad(15)+1e-10)
        self.assertEqual(set(phases),{'start','clean','approach','retract','transfer'})
        for i,(joints,phase) in enumerate(zip(q,phases)):
            self.assertEqual(self.world.check(joints,str(phase)),[],f'Collision at sample {i}')

    def test_brush_contact_is_not_globally_exempt(self):
        idx=np.flatnonzero(self.data['phases']=='clean')[20]
        q=self.data['joints'][idx]
        self.assertEqual(self.world.check(q,'clean'),[])
        self.assertIn('brush / bowl',self.world.check(q,'transfer'))


if __name__=='__main__':unittest.main()
