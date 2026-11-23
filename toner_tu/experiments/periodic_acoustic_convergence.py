"""Periodic undamped acoustic Fourier-mode benchmark with plots and animation."""
import argparse
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from toner_tu.fields import Fields,Grid
from toner_tu.parameters import TonerTuParameters
from toner_tu.solver import integrate
from toner_tu.visualization import animate_fields

AMPLITUDE=1e-5
FINAL_TIME=.1

def run(nx,history=False):
    grid=Grid(nx=nx,ny=4,lx=1.,ly=1.); x,_=grid.coordinates()
    fields=Fields(1+AMPLITUDE*np.cos(2*np.pi*x),np.zeros((4,nx)),np.zeros((4,nx)))
    params=TonerTuParameters(alpha=0.,beta=1.,sigma=1.,diffusion=0.,lambda_=0.)
    dt=FINAL_TIME/(nx*nx); n=nx*nx
    rh=[fields.rho.copy()]; vxh=[fields.vx.copy()]; vyh=[fields.vy.copy()]
    for i in range(n):
        fields=integrate(fields,grid,params,dt=dt,n_steps=1)
        if history and (i+1)%16==0 or history and i==n-1:
            rh.append(fields.rho.copy()); vxh.append(fields.vx.copy()); vyh.append(fields.vy.copy())
    phase=2*np.pi*FINAL_TIME
    rho_exact=1+AMPLITUDE*np.cos(2*np.pi*x)*np.cos(phase)
    vx_exact=AMPLITUDE*np.sin(2*np.pi*x)*np.sin(phase)
    error=float(np.sqrt(np.mean(((fields.rho-rho_exact)/AMPLITUDE)**2+
                                ((fields.vx-vx_exact)/AMPLITUDE)**2)))
    result={"grid":grid,"fields":fields,"error":error,"dt":dt}
    if history:
        indices=[0]+list(range(16,n+1,16))
        if indices[-1]!=n: indices.append(n)
        result.update(times=np.array(indices)*dt,rho_history=np.asarray(rh),
                      vx_history=np.asarray(vxh),vy_history=np.asarray(vyh))
    return result

def main():
    p=argparse.ArgumentParser(); p.add_argument("--output-dir",default="results/validation")
    p.add_argument("--no-show",action="store_true"); args=p.parse_args()
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    resolutions=(16,32,64); rows=[]; previous=None
    for nx in resolutions:
        r=run(nx); order="" if previous is None else np.log2(previous/r["error"])
        rows.append((nx,r["dt"],r["error"],order)); previous=r["error"]
    with (out/"periodic_acoustic_convergence.csv").open("w",newline="") as f:
        w=csv.writer(f); w.writerow(("nx","dt","normalized_l2_error","observed_order")); w.writerows(rows)
    fine=run(64,True); x,_=fine["grid"].coordinates(); phase=2*np.pi*FINAL_TIME
    fig,axes=plt.subplots(2,1,figsize=(9,7),sharex=True)
    axes[0].plot(x[0],fine["fields"].rho[0],label="Numerical")
    axes[0].plot(x[0],1+AMPLITUDE*np.cos(2*np.pi*x[0])*np.cos(phase),"--",label="Continuum")
    axes[0].set_ylabel("Density"); axes[0].legend()
    axes[1].plot(x[0],fine["fields"].vx[0],label="Numerical")
    axes[1].plot(x[0],AMPLITUDE*np.sin(2*np.pi*x[0])*np.sin(phase),"--",label="Continuum")
    axes[1].set_ylabel("vx"); axes[1].set_xlabel("x")
    for ax in axes: ax.grid(alpha=.3)
    fig.suptitle("Undamped acoustic mode at final time"); fig.tight_layout()
    fig.savefig(out/"periodic_acoustic_convergence.png",dpi=160)
    if not args.no_show:
        animate_fields(fine["grid"],fine["times"],fine["rho_history"],fine["vx_history"],fine["vy_history"])
        plt.show()
    print("Nx, dt, normalized L2 error, order")
    for r in rows: print(f"{r[0]}, {r[1]:.6e}, {r[2]:.6e}, {r[3] or '-'}")
    print(f"Saved diagnostics in {out}")
if __name__=="__main__": main()
