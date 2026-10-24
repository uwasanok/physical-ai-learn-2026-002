import json
import unittest
import numpy as np
from ur5e_kinematics import UR5e,ROOT
from toilet_geometry import ToiletGeometry,SurfaceCamera
from robot_camera import robot_triangles,occluded_points,rasterize,robot_foreground,roi_mask


class ObservationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.g=ToiletGeometry();cls.cam=SurfaceCamera(cls.g,320,240);cls.arm=UR5e(tool_length=.26)
        cls.q=np.array(json.loads((ROOT/'output/observation/pose_selection.json').read_text())['selected_q_rad'])

    def test_roi_ray_intersections_independent_of_renderer(self):
        u,v=np.meshgrid(np.linspace(-.11,.11,56),np.linspace(.025,.155,34));points=self.g.surface(u,v).reshape(-1,3)
        groups=robot_triangles(self.arm,self.q)
        self.assertFalse(occluded_points(self.cam,groups,points).any())
        # No unsupported triangle crossing the rasterizer's near plane.
        for triangles,_ in groups:
            z=((triangles-self.cam.origin)@self.cam.rotation)[...,2]
            self.assertFalse(((z.min(1)<=.01)&(z.max(1)>.01)).any())

    def test_cleaning_pose_hides_roi_but_observation_pose_does_not(self):
        data=np.load(ROOT/'output/geometry_trials/tool_260mm/motion_42.npz')
        clean=data['joints'][np.flatnonzero(data['phases']=='clean')[0]]
        roi=roi_mask(self.cam)
        before,_=rasterize(self.cam,robot_triangles(self.arm,clean))
        after,_=rasterize(self.cam,robot_triangles(self.arm,self.q))
        self.assertGreater((robot_foreground(self.cam,before)&roi).sum(),roi.sum()*.5)
        self.assertEqual((robot_foreground(self.cam,after)&roi).sum(),0)

    def test_saved_route_endpoints_and_sampling(self):
        summary=json.loads((ROOT/'output/observation/routes.json').read_text());d=np.load(ROOT/'output/observation/routes.npz')
        self.assertEqual(summary['valid'],318);self.assertEqual(len(d['offsets']),319)
        for start,end in zip(d['offsets'][:-1],d['offsets'][1:]):
            path=d['joints'][start:end]
            np.testing.assert_allclose(path[0],self.q,atol=1e-10)
            self.assertLessEqual(np.abs(np.diff(path,axis=0)).max(),np.deg2rad(.25)+1e-10)


if __name__=='__main__':unittest.main()
