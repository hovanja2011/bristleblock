import numpy as np
from toner_tu.fields import Fields,Grid
from toner_tu.parameters import TonerTuParameters
from toner_tu.solver import integrate

def _damped_exact(t,amplitude):
    k=2*np.pi; matrix=np.array([[0.,-k],[k,-.04*k*k]])
    values,vectors=np.linalg.eig(matrix)
    initial=np.array([amplitude,0.])
    return np.real_if_close(vectors@(np.exp(values*t)*np.linalg.solve(vectors,initial))).real

def test_damped_linear_acoustic_mode_converges_to_continuum():
    amplitude=1e-5; final=.1; errors=[]
    for nx in (24,48,96):
        grid=Grid(nx=nx,ny=4,lx=1.,ly=1.); x,_=grid.coordinates()
        fields=Fields(1+amplitude*np.cos(2*np.pi*x),np.zeros((4,nx)),np.zeros((4,nx)))
        params=TonerTuParameters(alpha=0.,beta=1.,sigma=1.,diffusion=.04,lambda_=0.)
        dt=.015*grid.dx; n=round(final/dt); dt=final/n
        result=integrate(fields,grid,params,dt=dt,n_steps=n)
        ar,au=_damped_exact(final,amplitude)
        rho_exact=1+ar*np.cos(2*np.pi*x); vx_exact=au*np.sin(2*np.pi*x)
        errors.append(float(np.sqrt(np.mean(((result.rho-rho_exact)/amplitude)**2+
                                            ((result.vx-vx_exact)/amplitude)**2))))
    assert errors[0]>errors[1]>errors[2]
    assert errors[0]/errors[1]>2.5 and errors[1]/errors[2]>2.5

def test_nonlinear_homogeneous_ordering_matches_exact_solution():
    alpha=beta=1.; speed0=.2; final=1.; errors=[]
    for requested_dt in (.08,.04,.02):
        grid=Grid(nx=8,ny=8,lx=1.,ly=1.); rho=np.ones((8,8))
        initial=Fields(rho,np.full_like(rho,speed0),np.zeros_like(rho))
        params=TonerTuParameters(alpha=alpha,beta=beta,sigma=0.,diffusion=0.,lambda_=0.)
        n=round(final/requested_dt); dt=final/n; state=initial.copy()
        numerical=np.empty(n+1); numerical[0]=speed0
        for i in range(1,n+1):
            state=integrate(state,grid,params,dt=dt,n_steps=1)
            numerical[i]=np.mean(state.vx)
        times=np.arange(n+1)*dt; q0=speed0**2
        exact=np.sqrt(alpha/(beta+(alpha/q0-beta)*np.exp(-2*alpha*times)))
        errors.append(float(np.max(np.abs(numerical-exact))))
        assert np.allclose(state.rho,1.)
    assert errors[0]>errors[1]>errors[2]
    assert errors[0]/errors[1]>5 and errors[1]/errors[2]>5
