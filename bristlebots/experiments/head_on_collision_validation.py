"""Head-on damped collision vs the exact spring-dashpot solution."""
import argparse
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from bristlebots.robot import Robot
from bristlebots.contacts import ContactParameters
from bristlebots.simulation import step_robots
from bristlebots.visualization import animate_robots

MASS,RADIUS,STIFFNESS,DAMPING=1.0,0.5,2000.0,1.5
INITIAL_SPEED_EACH=0.5

def exact_overlap(t):
    meff=MASS/2
    decay=DAMPING/(2*meff)
    w0=np.sqrt(STIFFNESS/meff); wd=np.sqrt(w0*w0-decay*decay)
    delta0=0.0; rate0=2*INITIAL_SPEED_EACH
    a=delta0; b=(rate0+decay*delta0)/wd
    e=np.exp(-decay*t); c=a*np.cos(wd*t); s=b*np.sin(wd*t)
    delta=e*(c+s)
    rate=e*(-decay*(c+s)-a*wd*np.sin(wd*t)+b*wd*np.cos(wd*t))
    return delta,rate

def run(dt,history=False):
    params=ContactParameters(normal_stiffness=STIFFNESS,normal_damping=DAMPING,
      mu_static=0,mu_kinetic=0,tangential_stiffness=1000,tangential_damping=0)
    a=Robot(position=(-RADIUS,0),velocity=(INITIAL_SPEED_EACH,0),radius=RADIUS,mass=MASS)
    b=Robot(position=(RADIUS,0),velocity=(-INITIAL_SPEED_EACH,0),radius=RADIUS,mass=MASS)
    robots=[a,b]
    wd=np.sqrt(STIFFNESS/(MASS/2)-(DAMPING/MASS)**2)
    end=0.8*np.pi/wd; n=round(end/dt); times=np.arange(n+1)*dt
    overlap=np.zeros(n+1); rate=np.zeros(n+1); rate[0]=2*INITIAL_SPEED_EACH
    positions=[[a.position.copy(),b.position.copy()]]
    for i in range(1,n+1):
        step_robots(robots,dt,contact_parameters=params)
        overlap[i]=max(0,2*RADIUS-np.linalg.norm(b.position-a.position))
        rate[i]=a.velocity[0]-b.velocity[0]
        positions.append([a.position.copy(),b.position.copy()])
    exact,exact_rate=exact_overlap(times)
    exact_x=np.column_stack((-RADIUS+0.5*exact,RADIUS-0.5*exact))
    result={"times":times,"overlap":overlap,"rate":rate,"exact":exact,
      "exact_rate":exact_rate,"overlap_error":float(np.max(np.abs(overlap-exact))),
      "rate_error":float(np.max(np.abs(rate-exact_rate))),
      "position_error":float(np.max(np.abs(np.asarray(positions)[:,:,0]-exact_x)))}
    if history: result.update(robots=robots,positions=np.asarray(positions)+np.array([4.,3.]))
    return result

def main():
    p=argparse.ArgumentParser(); p.add_argument("--output-dir",default="results/validation")
    p.add_argument("--no-show",action="store_true"); args=p.parse_args()
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    dts=(.001,.0005,.00025,.000125); rows=[]; previous=None
    for dt in dts:
        r=run(dt)
        for metric in ("overlap_error","rate_error","position_error"):
            err=r[metric]; order="" if previous is None else np.log2(previous[metric]/err)
            rows.append((dt,metric,err,order))
        previous={m:r[m] for m in ("overlap_error","rate_error","position_error")}
    with (out/"head_on_collision.csv").open("w",newline="") as f:
        w=csv.writer(f); w.writerow(("dt","metric","max_abs_error","observed_order")); w.writerows(rows)
    fine=run(dts[-1],True)
    fig,axes=plt.subplots(2,1,figsize=(9,7),sharex=True)
    axes[0].plot(fine["times"],fine["overlap"],label="Numerical")
    axes[0].plot(fine["times"],fine["exact"],"--",label="Analytical")
    axes[0].set_ylabel("Overlap"); axes[0].legend()
    axes[1].plot(fine["times"],fine["rate"],label="Numerical")
    axes[1].plot(fine["times"],fine["exact_rate"],"--",label="Analytical")
    axes[1].set_ylabel("Overlap rate"); axes[1].set_xlabel("Time")
    for ax in axes: ax.grid(alpha=.3)
    fig.suptitle("Head-on damped collision"); fig.tight_layout()
    fig.savefig(out/"head_on_collision.png",dpi=160)
    fig2,ax=plt.subplots(figsize=(6,4))
    for metric in ("overlap_error","rate_error","position_error"):
        sub=[r for r in rows if r[1]==metric]
        ax.loglog([r[0] for r in sub],[r[2] for r in sub],"o-",label=metric)
    ax.set_xlabel("dt"); ax.set_ylabel("Maximum absolute error")
    ax.set_title("Head-on collision convergence"); ax.grid(True,which="both",alpha=.3); ax.legend()
    fig2.tight_layout(); fig2.savefig(out/"head_on_collision_convergence.png",dpi=160)
    if not args.no_show:
        animate_robots(fine["robots"],fine["positions"],box_size=(10,6),interval=35,
                       trail_length=100,show_angular_velocity=False)
        plt.show()
    print("dt, overlap error/order, rate error/order, position error/order")
    for i,dt in enumerate(dts):
        print(f"{dt:.6g}, "+", ".join(f"{r[2]:.5e}/{r[3] or '-'}" for r in rows[3*i:3*i+3]))
    print(f"Saved diagnostics in {out}")
if __name__=="__main__": main()
