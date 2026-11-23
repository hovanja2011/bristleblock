"""Nonlinear homogeneous Toner–Tu ordering vs exact logistic solution."""
import argparse
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from toner_tu.fields import Fields,Grid
from toner_tu.parameters import TonerTuParameters
from toner_tu.solver import integrate

ALPHA,BETA,SPEED0,FINAL_TIME=1.,1.,.2,1.

def exact_speed(t):
    q0=SPEED0**2
    return np.sqrt(ALPHA/(BETA+(ALPHA/q0-BETA)*np.exp(-2*ALPHA*t)))

def run(dt):
    grid=Grid(nx=8,ny=8,lx=1.,ly=1.); rho=np.ones((8,8))
    state=Fields(rho,np.full_like(rho,SPEED0),np.zeros_like(rho))
    params=TonerTuParameters(alpha=ALPHA,beta=BETA,sigma=0.,diffusion=0.,lambda_=0.)
    n=round(FINAL_TIME/dt); dt=FINAL_TIME/n; times=np.arange(n+1)*dt
    speed=np.empty(n+1); speed[0]=SPEED0
    for i in range(1,n+1):
        state=integrate(state,grid,params,dt=dt,n_steps=1)
        speed[i]=np.mean(state.vx)
    exact=exact_speed(times)
    return {"dt":dt,"times":times,"speed":speed,"exact":exact,
            "error":float(np.max(np.abs(speed-exact)))}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--output-dir",default="results/validation")
    p.add_argument("--no-show",action="store_true"); args=p.parse_args()
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    dts=(.08,.04,.02,.01); rows=[]; previous=None
    for dt in dts:
        r=run(dt); order="" if previous is None else np.log2(previous/r["error"])
        rows.append((r["dt"],r["error"],order)); previous=r["error"]
    with (out/"nonlinear_ordering.csv").open("w",newline="") as f:
        w=csv.writer(f); w.writerow(("dt","max_abs_speed_error","observed_order")); w.writerows(rows)
    fine=run(dts[-1]); fig,axes=plt.subplots(2,1,figsize=(9,7),sharex=True)
    axes[0].plot(fine["times"],fine["speed"],label="Numerical")
    axes[0].plot(fine["times"],fine["exact"],"--",label="Analytical")
    axes[0].set_ylabel("Mean speed"); axes[0].legend()
    axes[1].plot(fine["times"],fine["speed"]-fine["exact"])
    axes[1].set_ylabel("Numerical - exact"); axes[1].set_xlabel("Time")
    for ax in axes: ax.grid(alpha=.3)
    fig.suptitle("Nonlinear homogeneous ordering"); fig.tight_layout()
    fig.savefig(out/"nonlinear_ordering.png",dpi=160)
    fig2,ax=plt.subplots(figsize=(6,4)); ax.loglog([r[0] for r in rows],[r[1] for r in rows],"o-")
    ax.set_xlabel("dt"); ax.set_ylabel("Maximum absolute speed error")
    ax.set_title("Nonlinear ordering convergence"); ax.grid(True,which="both",alpha=.3)
    fig2.tight_layout(); fig2.savefig(out/"nonlinear_ordering_convergence.png",dpi=160)
    if not args.no_show: plt.show()
    print("dt, max speed error, order")
    for r in rows: print(f"{r[0]:.6g}, {r[1]:.6e}, {r[2] or '-'}")
    print(f"Saved diagnostics in {out}")
if __name__=="__main__": main()
