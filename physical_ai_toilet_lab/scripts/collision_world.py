"""FCL mesh/primitive discrete collision checking, with explicit contact policy."""
import numpy as np
import fcl
from cleaning_model import normal, surface


def mesh_object(triangles):
    vertices = np.asarray(triangles, dtype=float).reshape(-1, 3)
    indices = np.arange(len(vertices), dtype=np.int32).reshape(-1, 3)
    mesh = fcl.BVHModel(); mesh.beginModel(len(vertices), len(indices))
    mesh.addSubModel(vertices, indices); mesh.endModel()
    return fcl.CollisionObject(mesh)


def bowl_mesh():
    # Closed solid: inner paraboloid, elliptical outer wall and underside.
    triangles = []
    segments, rings = 96, 32
    def p(r, i, z=None):
        t = i*2*np.pi/segments
        return [.18*r*np.cos(t), .24*r*np.sin(t), .22+.18*r*r if z is None else z]
    for r in range(rings):
        for i in range(segments):
            a,b,c,d = p(r/rings,i),p((r+1)/rings,i),p((r+1)/rings,i+1),p(r/rings,i+1)
            if r: triangles.append([a,b,d])
            triangles.append([b,c,d])
    for i in range(segments):
        a,b,c,d=p(1,i),p(1,i+1),p(1,i+1,.16),p(1,i,.16)
        triangles += [[a,b,c],[a,c,d],[[0,0,.16],d,c]]
    return np.array(triangles)


class CollisionWorld:
    def __init__(self, arm, geometry=None):
        self.arm = arm
        self.surface_fn=geometry.surface if geometry else surface
        self.normal_fn=geometry.normal if geometry else normal
        self.meshes = arm.collision_meshes()
        self.links = {m['link']:mesh_object(m['triangles']) for m in self.meshes}
        self.links['shaft'] = fcl.CollisionObject(fcl.Cylinder(.006, arm.tool_length-.05))
        self.links['brush'] = fcl.CollisionObject(fcl.Cylinder(.012, .02))
        self.bowl_triangles = geometry.bowl_mesh() if geometry else bowl_mesh()
        self.obstacles = {'bowl':mesh_object(self.bowl_triangles),
            'cabinet':fcl.CollisionObject(fcl.Box(.44,.45,.55), fcl.Transform(np.array([0.,-.525,.275]))),
            'floor':fcl.CollisionObject(fcl.Box(3.,3.,.05),fcl.Transform(np.array([0.,0.,-.025]))),
            'camera':fcl.CollisionObject(fcl.Box(.06,.04,.04),fcl.Transform(np.array([0.,-.65,1.]))),
            'camera_post':fcl.CollisionObject(fcl.Cylinder(.012,.43),fcl.Transform(np.array([0.,-.65,.765])))}
        if geometry:self.obstacles['rim']=mesh_object(geometry.rim_mesh())
        names=list(self.links)
        # Only mechanical neighbors whose meshes meet at the joint are excluded.
        self.exclusions={frozenset(pair) for pair in zip(names[:-1],names[1:])}
        self.pairs=[(a,b) for i,a in enumerate(names) for b in names[i+1:] if frozenset((a,b)) not in self.exclusions]
        self.calls=0

    def check(self, q, phase='transfer'):
        self.calls+=1
        if np.any(q<self.arm.lower) or np.any(q>self.arm.upper):return ['joint_limits']
        tf=self.arm.transforms(q)
        for name,obj in self.links.items():
            t=tf['tool0'].copy() if name in ('shaft','brush') else tf[name]
            if name=='shaft':t[:3,3]+=t[:3,2]*(.025+(self.arm.tool_length-.05)/2)
            if name=='brush':t[:3,3]+=t[:3,2]*(self.arm.tool_length-.01)
            obj.setTransform(fcl.Transform(t[:3,:3],t[:3,3]))
        tip=self.arm.fk(q)
        uv=tip[:2,3]
        gap=np.linalg.norm(tip[:3,3]-self.surface_fn(*uv))
        contact_allowed=(phase in ('clean','approach','retract') and gap<=.004
                         and abs(uv[0])<=.12 and .015<=uv[1]<=.17
                         and np.dot(-tip[:3,2],self.normal_fn(*uv))>=np.cos(np.deg2rad(25)))
        hits=[]
        request=fcl.CollisionRequest()
        def collides(a,b):return bool(fcl.collide(a,b,request,fcl.CollisionResult()))
        for a,b in self.pairs:
            if collides(self.links[a],self.links[b]):hits.append(a+' / '+b)
        for a,obj in self.links.items():
            for b,obstacle in self.obstacles.items():
                if a=='base_link_inertia' and b=='cabinet':continue
                if a=='brush' and b=='bowl' and contact_allowed:continue
                if collides(obj,obstacle):hits.append(a+' / '+b)
        return hits
