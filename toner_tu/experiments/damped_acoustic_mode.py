"""Damped linear Toner–Tu Fourier mode vs continuum matrix-exponential solution."""
import argparse
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from toner_tu.fields import Fields,Grid
from toner_tu.parameters import TonerTuParameters
from toner_tu.solver import integrate
from toner_tu.visualization import animate_fields

AMPLITUDE,RHO0,SIGMA,DIFFUSION,FINAL_TIME=1e-5,1.0,1.0,.04,.1

def exact_amplitudes(t):
    k=2*np.pi
    matrix=np.array([[0.,-RHO0*k],[SIGMA*k,-DIFFUSION*k*k]])
    values,vectors=np.linalg.eig(matrix)
    initial=np.array([AMPLITUDE,0.])
    state=vectors@(np.exp(values*t)*np.linalg.solve(vectors,initial))
    return np.real_if_close(state).real

def run(nx,history=False):
    grid=Grid(nx=nx,ny=4,lx=1.,ly=1.); x,_=grid.coordinates()
    fields=Fields(RHO0+AMPLITUDE*np.cos(2*np.pi*x),np.zeros((4,nx)),np.zeros((4,nx)))
    params=TonerTuParameters(alpha=0.,beta=1.,sigma=SIGMA,diffusion=DIFFUSION,lambda_=0.)
    dt=.015*grid.dx; n=round(FINAL_TIME/dt); dt=FINAL_TIME/n
    rh=[fields.rho.copy()]; vxh=[fields.vx.copy()]; vyh=[fields.vy.copy()]
    for _ in range(n):
        fields=integrate(fields,grid,params,dt=dt,n_steps=1)
        if history: rh.append(fields.rho.copy()); vxh.append(fields.vx.copy()); vyh.append(fields.vy.copy())
    ar,au=exact_amplitudes(FINAL_TIME)
    rho_exact=RHO0+ar*np.cos(2*np.pi*x); vx_exact=au*np.sin(2*np.pi*x)
    error=np.sqrt(np.mean(((fields.rho-rho_exact)/AMPLITUDE)**2+((fields.vx-vx_exact)/AMPLITUDE)**2))
    result={"error":float(error),"grid":grid,"fields":fields,"dt":dt}
    if history: result.update(times=np.linspace(0,FINAL_TIME,n+1),rho_history=np.asarray(rh),
      vx_history=np.asarray(vxh),vy_history=np.asarray(vyh))
    return result

def main():
    p=argparse.ArgumentParser(); p.add_argument("--output-dir",default="results/validation")
    p.add_argument("--no-show",action="store_true"); args=p.parse_args()
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    resolutions=(24,48,96); rows=[]; previous=None
    for nx in resolutions:
        r=run(nx); order="" if previous is None else np.log2(previous/r["error"])
        rows.append((nx,r["dt"],r["error"],order)); previous=r["error"]
    with (out/"damped_acoustic_mode.csv").open("w",newline="") as f:
        w=csv.writer(f); w.writerow(("nx","dt","normalized_l2_error","observed_order")); w.writerows(rows)
    fine=run(resolutions[-1],True); x,_=fine["grid"].coordinates(); ar,au=exact_amplitudes(FINAL_TIME)
    fig,axes=plt.subplots(2,1,figsize=(9,7),sharex=True)
    axes[0].plot(x[0],fine["fields"].rho[0],label="Numerical")
    axes[0].plot(x[0],RHO0+ar*np.cos(2*np.pi*x[0]),"--",label="Continuum")
    axes[0].set_ylabel("Density"); axes[0].legend()
    axes[1].plot(x[0],fine["fields"].vx[0],label="Numerical")
    axes[1].plot(x[0],au*np.sin(2*np.pi*x[0]),"--",label="Continuum")
    axes[1].set_ylabel("vx"); axes[1].set_xlabel("x")
    for ax in axes: ax.grid(alpha=.3)
    fig.suptitle("Damped acoustic mode at final time"); fig.tight_layout()
    fig.savefig(out/"damped_acoustic_mode.png",dpi=160)
    fig2,ax=plt.subplots(figsize=(6,4)); ax.loglog(resolutions,[r[2] for r in rows],"o-")
    ax.set_xlabel("Nx"); ax.set_ylabel("Normalized L2 error"); ax.set_title("Damped acoustic convergence")
    ax.grid(True,which="both",alpha=.3); fig2.tight_layout()
    fig2.savefig(out/"damped_acoustic_convergence.png",dpi=160)
    if not args.no_show:
        animate_fields(fine["grid"],fine["times"],fine["rho_history"],fine["vx_history"],fine["vy_history"])
        plt.show()
    print("Nx, dt, normalized L2 error, order")
    for r in rows: print(f"{r[0]}, {r[1]:.6e}, {r[2]:.6e}, {r[3] or '-'}")
    print(f"Saved diagnostics in {out}")
if __name__=="__main__": main()
