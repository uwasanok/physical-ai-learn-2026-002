"""Illustrative bowl with flatter bottom, rising walls and a separate overhanging rim.
Not manufacturer CAD or measured sanitary ware dimensions.
"""
import numpy as np
from camera_loop import Camera
from cleaning_model import Cleaner, scene


class ToiletGeometry:
    a,b=.18,.24
    rim_inner,rim_outer=.94,1.14
    rim_bottom,rim_top=.395,.425

    def surface(self,u,v):
        u,v=np.broadcast_arrays(u,v);s=(u/self.a)**2+(v/self.b)**2
        return np.stack([u,v,.22+.04*s+.14*s*s],-1)

    def normal(self,u,v):
        u,v=np.broadcast_arrays(u,v);s=(u/self.a)**2+(v/self.b)**2
        n=np.stack([-(.04+.28*s)*2*u/self.a**2,-(.04+.28*s)*2*v/self.b**2,np.ones_like(u)],-1)
        return n/np.linalg.norm(n,axis=-1,keepdims=True)

    def bowl_mesh(self):
        from collision_world import bowl_mesh
        triangles=bowl_mesh().copy()
        # Original z=.22+.18*r² inner surface; keep lower outer wall/bottom.
        for tri in triangles:
            s=(tri[:,0]/self.a)**2+(tri[:,1]/self.b)**2
            mask=np.isclose(tri[:,2],.22+.18*s,atol=1e-8)
            tri[mask,2]=.22+.04*s[mask]+.14*s[mask]**2
        return triangles

    def rim_mesh(self):
        triangles=[]
        def p(r,i,z):
            t=i*2*np.pi/96
            return [self.a*r*np.cos(t),self.b*r*np.sin(t),z]
        for i in range(96):
            for r0,z0,r1,z1 in [(self.rim_inner,self.rim_top,self.rim_outer,self.rim_top),
                (self.rim_outer,self.rim_top,self.rim_outer,self.rim_bottom),
                (self.rim_outer,self.rim_bottom,self.rim_inner,self.rim_bottom),
                (self.rim_inner,self.rim_bottom,self.rim_inner,self.rim_top)]:
                a,b,c,d=p(r0,i,z0),p(r1,i,z1),p(r1,i+1,z1),p(r0,i+1,z0)
                triangles.extend([[a,b,c],[a,c,d]])
        return np.array(triangles)

    def make_scene(self,seed):
        original,patches=scene(seed)
        return Cleaner(original.uv,original.required,self.surface,self.normal),patches


class SurfaceCamera(Camera):
    def __init__(self,geometry,width=640,height=480):
        self.geometry=geometry;self.surface_fn=geometry.surface
        super().__init__(width,height)

    def unproject(self,px,py):
        px,py=np.broadcast_arrays(px,py)
        rays=np.stack([(px-self.width/2)/self.focal,(py-self.height/2)/self.focal,np.ones_like(px)],-1)@self.rotation.T
        dz=rays[...,2];safe=np.where(np.abs(dz)>1e-12,dz,-1e-12)
        lo=(.43-self.origin[2])/safe;hi=(.21-self.origin[2])/safe
        # Convex quartic graph along a downward camera ray: bracket the FIRST crossing.
        best=np.full(px.shape,np.inf);previous=lo
        def value(t):
            p=self.origin+t[...,None]*rays
            return p[...,2]-self.surface_fn(p[...,0],p[...,1])[...,2]
        fprev=value(previous);found=np.zeros(px.shape,dtype=bool);left=lo.copy();right=hi.copy()
        for alpha in np.linspace(0,1,49)[1:]:
            t=lo+(hi-lo)*alpha;f=value(t)
            bracket=(fprev>=0)&(f<=0)&~found&(dz<0)
            left=np.where(bracket,previous,left);right=np.where(bracket,t,right);found|=bracket
            previous=t;fprev=f
        for _ in range(25):
            mid=(left+right)/2;fm=value(mid)
            left=np.where(fm>0,mid,left);right=np.where(fm>0,right,mid)
        root=(left+right)/2;p=self.origin+root[...,None]*rays
        inside=(p[...,0]/.18)**2+(p[...,1]/.24)**2<=1
        best=np.where(found&inside,root,best)
        g=self.geometry
        # Top/bottom of the rim plus its vertical inner/outer elliptical walls.
        for z in [g.rim_top,g.rim_bottom]:
            t=(z-self.origin[2])/safe;p=self.origin+t[...,None]*rays
            r2=(p[...,0]/g.a)**2+(p[...,1]/g.b)**2
            ok=(t>0)&(r2>=g.rim_inner**2)&(r2<=g.rim_outer**2)
            best=np.minimum(best,np.where(ok,t,np.inf))
        ox,oy=self.origin[:2];dx,dy=rays[...,0],rays[...,1]
        a=(dx/g.a)**2+(dy/g.b)**2;b=2*(ox*dx/g.a**2+oy*dy/g.b**2)
        for radius,zmin,zmax in [(g.rim_inner,g.rim_bottom,g.rim_top),(g.rim_outer,g.rim_bottom,g.rim_top),(1.,.16,.4)]:
            c=(ox/g.a)**2+(oy/g.b)**2-radius**2;disc=b*b-4*a*c
            for sign in [-1,1]:
                t=(-b+sign*np.sqrt(np.maximum(disc,0)))/(2*a)
                z=self.origin[2]+t*dz
                ok=(disc>=0)&(t>0)&(z>=zmin)&(z<=zmax)
                best=np.minimum(best,np.where(ok,t,np.inf))
        valid=np.isfinite(best)
        return self.origin+np.where(valid,best,0)[...,None]*rays,valid

    def visible(self,xyz):
        pixels=self.project(xyz);hit,valid=self.unproject(pixels[...,0],pixels[...,1])
        inside=(pixels[...,0]>=0)&(pixels[...,0]<self.width)&(pixels[...,1]>=0)&(pixels[...,1]<self.height)
        return valid&inside&(np.linalg.norm(hit-xyz,axis=-1)<.001)
