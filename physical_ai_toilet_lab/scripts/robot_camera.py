"""Software triangle z-buffer for official robot meshes in the fixed RGB camera."""
import numpy as np


def cylinder_triangles(radius,length,sides=24):
    tris=[]
    for i in range(sides):
        t0,t1=2*np.pi*i/sides,2*np.pi*(i+1)/sides
        a=np.array([radius*np.cos(t0),radius*np.sin(t0),-length/2])
        b=np.array([radius*np.cos(t1),radius*np.sin(t1),-length/2])
        c=b+[0,0,length];d=a+[0,0,length]
        tris.extend([[a,b,c],[a,c,d],[[0,0,-length/2],b,a],[[0,0,length/2],d,c]])
    return np.array(tris)


def robot_triangles(arm,q):
    frames=arm.transforms(q);groups=[]
    for m in arm.collision_meshes():
        t=frames[m['link']];tri=m['triangles']@t[:3,:3].T+t[:3,3]
        color=[155,173,185] if 'arm' in m['link'] else [77,128,157]
        groups.append((tri,color))
    t=frames['tool0']
    for radius,length,center in [(.006,arm.tool_length-.05,(arm.tool_length)/2),(.012,.02,arm.tool_length-.01)]:
        tris=cylinder_triangles(radius,length)+[0,0,center]
        groups.append((tris@t[:3,:3].T+t[:3,3],[37,131,108]))
    return groups


def rasterize(camera,groups):
    """Integer pixel centers, matching Camera.unproject. Perspective-correct z."""
    depth=np.full((camera.height,camera.width),np.inf)
    rgb=np.zeros((*depth.shape,3),dtype=np.uint8)
    for triangles,color in groups:
        for world in triangles:
            local=(world-camera.origin)@camera.rotation
            if np.any(local[:,2]<=.01):continue  # near-plane clipping not needed in verified poses
            screen=local[:,:2]/local[:,2,None]*camera.focal+[camera.width/2,camera.height/2]
            xmin=max(0,int(np.ceil(screen[:,0].min())));xmax=min(camera.width-1,int(np.floor(screen[:,0].max())))
            ymin=max(0,int(np.ceil(screen[:,1].min())));ymax=min(camera.height-1,int(np.floor(screen[:,1].max())))
            if xmin>xmax or ymin>ymax:continue
            x,y=np.meshgrid(np.arange(xmin,xmax+1),np.arange(ymin,ymax+1))
            a,b,c=screen;den=(b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1])
            if abs(den)<1e-10:continue
            w0=((b[1]-c[1])*(x-c[0])+(c[0]-b[0])*(y-c[1]))/den
            w1=((c[1]-a[1])*(x-c[0])+(a[0]-c[0])*(y-c[1]))/den;w2=1-w0-w1
            inside=(w0>=-1e-8)&(w1>=-1e-8)&(w2>=-1e-8)
            inverse=w0/local[0,2]+w1/local[1,2]+w2/local[2,2]
            z=1/np.maximum(inverse,1e-12)
            view=depth[ymin:ymax+1,xmin:xmax+1];take=inside&(z<view)
            view[take]=z[take]
            n=np.cross(world[1]-world[0],world[2]-world[0]);n/=max(np.linalg.norm(n),1e-12)
            shade=.65+.35*abs(n@np.array([.3,-.4,.866]))
            rgb[ymin:ymax+1,xmin:xmax+1][take]=np.array(color)*shade
    return depth,rgb


def surface_depth(camera):return ((camera.points-camera.origin)@camera.rotation)[...,2]


def robot_foreground(camera,robot_depth):
    return np.isfinite(robot_depth)&(~camera.valid|(robot_depth<surface_depth(camera)-1e-5))


def roi_mask(camera):
    p=camera.points
    return camera.valid&(abs(p[...,0])<=.11)&(p[...,1]>=.025)&(p[...,1]<=.155)


def occluded_points(camera,groups,points):
    """Independent Moller-Trumbore segment intersections (no raster quantization)."""
    triangles=np.concatenate([tri for tri,_ in groups]);v0=triangles[:,0]
    e1=triangles[:,1]-v0;e2=triangles[:,2]-v0
    tvec=camera.origin-v0;qvec=np.cross(tvec,e1);numerator=np.einsum('ij,ij->i',e2,qvec)
    results=[]
    for start in range(0,len(points),64):
        rays=np.asarray(points[start:start+64])-camera.origin
        pvec=np.cross(rays[:,None,:],e2[None,:,:]);det=np.einsum('ij,kij->ki',e1,pvec)
        inv=np.divide(1.,det,out=np.zeros_like(det),where=abs(det)>1e-10)
        u=np.einsum('ij,kij->ki',tvec,pvec)*inv
        v=(rays@qvec.T)*inv;t=numerator[None,:]*inv
        hit=(abs(det)>1e-10)&(u>=-1e-9)&(v>=-1e-9)&(u+v<=1+1e-9)&(t>1e-6)&(t<1-1e-6)
        results.extend(hit.any(axis=1).tolist())
    return np.array(results,dtype=bool)
