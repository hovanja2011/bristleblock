"""Rectangular closed-box geometry and its boundary-aware operators."""
from dataclasses import dataclass
import numpy as np
from toner_tu.fields import Grid

@dataclass(frozen=True)
class BoxMaze:
    """Empty rectangular container with four impermeable outer walls."""
    width: float
    height: float
    def __post_init__(self):
        if not np.isfinite([self.width,self.height]).all() or self.width<=0 or self.height<=0:
            raise ValueError("Box dimensions must be finite and positive.")
    def grid(self,nx:int,ny:int)->Grid:
        """Construct a cell-centred grid covering the physical box."""
        return Grid(nx=nx,ny=ny,lx=self.width,ly=self.height)
    def operators(self):
        """Return the numerical operators that impose closed-box walls."""
        return BoxDomainOperators()

class BoxDomainOperators:
    """Spatial operators for a closed box.

    All four walls impose zero normal mass flux. Density and tangential
    velocity use even reflection; the velocity component normal to a wall
    uses odd reflection. This is a simple cell-centred free-slip wall model.
    """
    @staticmethod
    def _ghost(field,component):
        ny,nx=field.shape
        g=np.empty((ny+2,nx+2),dtype=float)
        g[1:-1,1:-1]=field
        # Left/right walls: x velocity is the wall-normal component.
        g[1:-1,0]= -field[:,0] if component=="vx" else field[:,0]
        g[1:-1,-1]= -field[:,-1] if component=="vx" else field[:,-1]
        # Bottom/top walls: y velocity is the wall-normal component.
        g[0,1:-1]= -field[0,:] if component=="vy" else field[0,:]
        g[-1,1:-1]= -field[-1,:] if component=="vy" else field[-1,:]
        g[0,0],g[0,-1]=g[0,1],g[0,-2]
        g[-1,0],g[-1,-1]=g[-1,1],g[-1,-2]
        return g
    def gradient(self,field,grid):
        """Central gradient with zero-normal-gradient scalar wall values."""
        g=self._ghost(field,"rho")
        return ((g[1:-1,2:]-g[1:-1,:-2])/(2*grid.dx),
                (g[2:,1:-1]-g[:-2,1:-1])/(2*grid.dy))
    def laplacian(self,field,grid,component):
        """Five-point Laplacian with component-aware reflected wall values."""
        g=self._ghost(field,component); c=g[1:-1,1:-1]
        return ((g[1:-1,2:]-2*c+g[1:-1,:-2])/grid.dx**2
                +(g[2:,1:-1]-2*c+g[:-2,1:-1])/grid.dy**2)
    def advective_derivative(self,vx,vy,grid):
        """First-order upwind (v·grad)v using wall-consistent ghost cells."""
        def upwind(field,component):
            g=self._ghost(field,component)
            dx=np.where(vx>=0,(field-g[1:-1,:-2])/grid.dx,(g[1:-1,2:]-field)/grid.dx)
            dy=np.where(vy>=0,(field-g[:-2,1:-1])/grid.dy,(g[2:,1:-1]-field)/grid.dy)
            return dx,dy
        ax,ay=upwind(vx,"vx"); bx,by=upwind(vy,"vy")
        return vx*ax+vy*ay, vx*bx+vy*by
    def density_rhs(self,rho,vx,vy,grid):
        """Conservative upwind finite-volume continuity equation.

        Boundary face fluxes are exactly zero, so total mass is conserved.
        """
        ny,nx=rho.shape
        u=np.zeros((ny,nx+1)); v=np.zeros((ny+1,nx))
        u[:,1:-1]=0.5*(vx[:,:-1]+vx[:,1:])
        v[1:-1,:]=0.5*(vy[:-1,:]+vy[1:,:])
        fx=np.zeros_like(u); fy=np.zeros_like(v)
        fx[:,1:-1]=np.maximum(u[:,1:-1],0)*rho[:,:-1]+np.minimum(u[:,1:-1],0)*rho[:,1:]
        fy[1:-1,:]=np.maximum(v[1:-1,:],0)*rho[:-1,:]+np.minimum(v[1:-1,:],0)*rho[1:,:]
        return -((fx[:,1:]-fx[:,:-1])/grid.dx+(fy[1:,:]-fy[:-1,:])/grid.dy)