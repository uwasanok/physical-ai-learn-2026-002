import unittest
import numpy as np
from toilet_geometry import ToiletGeometry,SurfaceCamera
from ur5e_kinematics import UR5e,ROOT
from collision_world import CollisionWorld


class ToiletGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.g=ToiletGeometry();cls.camera=SurfaceCamera(cls.g,320,240)

    def test_normals_match_geometry(self):
        for u,v in [(.02,.04),(.10,.14),(-.10,-.10)]:
            e=1e-6;g=self.g
            dx=(g.surface(u+e,v)[2]-g.surface(u-e,v)[2])/(2*e)
            dy=(g.surface(u,v+e)[2]-g.surface(u,v-e)[2])/(2*e)
            expected=np.array([-dx,-dy,1.]);expected/=np.linalg.norm(expected)
            np.testing.assert_allclose(expected,g.normal(u,v),atol=1e-8)

    def test_declared_roi_visible_but_near_wall_occluded(self):
        u,v=np.meshgrid(np.linspace(-.11,.11,56),np.linspace(.025,.155,34))
        self.assertTrue(self.camera.visible(self.g.surface(u,v)).all())
        self.assertFalse(bool(self.camera.visible(self.g.surface(0.,-.18))))

    def test_surface_roundtrip(self):
        p=self.g.surface(np.array([-.08,0.,.08]),np.array([.05,.1,.14]))
        uv=self.camera.project(p);hit,valid=self.camera.unproject(uv[:,0],uv[:,1])
        self.assertTrue(valid.all());np.testing.assert_allclose(hit,p,atol=1e-8)

    def test_rim_is_not_contact_exempt(self):
        arm=UR5e();world=CollisionWorld(arm,self.g)
        self.assertIn('rim',world.obstacles)
        # Reproduce a known unsafe wrist placement using first new camera target.
        from camera_loop import Environment,plan
        from ur5e_kinematics import tool_rotation
        camera=SurfaceCamera(self.g)
        model,_=self.g.make_scene(42);target,_=plan(Environment(model,camera).capture(),camera)
        tip=self.g.surface(target[0]-.026,target[1]);R=tool_rotation(tip,self.g.normal(*tip[:2]))
        seed=np.load(ROOT/'output/ur5e/motion.npz')['joints'][0]
        n=self.g.normal(*tip[:2]);q=seed;rim_hits=[]
        for offset in np.linspace(.08,0,81):
            q,pe,ae=arm.ik(tip+offset*n,R,q)
            self.assertLess(pe,.0001)
            rim_hits.extend(hit for hit in world.check(q,'approach') if '/ rim' in hit)
        self.assertTrue(rim_hits)


if __name__=='__main__':unittest.main()
