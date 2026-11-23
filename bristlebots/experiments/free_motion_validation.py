"""Quantitative free translation benchmark against the linear-drag solution."""
import argparse
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from bristlebots.robot import Robot
from bristlebots.simulation import step

MASS,DRAG,FORCE=1.4,.6,.8
X0=np.array([-.3,.2]); V0=np.array([.25,-.15]); FINAL_TIME=2.

def exact_state(t,x0,v0,force):
    terminal=force/DRAG; tau=MASS/DRAG; decay=np.exp(-t/tau)
    return x0+terminal*t+(v0-terminal)*tau*(1-decay), terminal+(v0-terminal)*decay

def run(dt):
    robot=Robot(position=X0,velocity=V0,theta=0.,radius=.2,mass=MASS,
                propulsion_force=FORCE,translational_drag=DRAG)
    n=round(FINAL_TIME/dt); pe=[]; ve=[]
    for i in range(1,n+1):
        step(robot,dt); t=i*dt
        ex,vx=exact_state(t,X0[0],V0[0],FORCE)
        ey,vy=exact_state(t,X0[1],V0[1],0.)
        pe.append(np.linalg.norm(robot.position-[ex,ey]))
        ve.append(np.linalg.norm(robot.velocity-[vx,vy]))
    return max(pe),max(ve)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output-dir",default="results/validation")
    parser.add_argument("--no-show",action="store_true")
    args=parser.parse_args(); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    dts=(.04,.02,.01,.005); results=[run(dt) for dt in dts]
    rows=[]; previous=None
    for dt,(ep,ev) in zip(dts,results):
        for metric,error in (("position",ep),("velocity",ev)):
            order="" if previous is None else np.log2(previous[metric]/error)
            rows.append((dt,metric,error,order))
        previous={"position":ep,"velocity":ev}
    with (out/"free_motion_validation.csv").open("w",newline="") as f:
        w=csv.writer(f); w.writerow(("dt","metric","max_abs_error","observed_order")); w.writerows(rows)
    fig,ax=plt.subplots(figsize=(6,4))
    ax.loglog(dts,[r[0] for r in results],"o-",label="Position")
    ax.loglog(dts,[r[1] for r in results],"o-",label="Velocity")
    ax.set_xlabel("dt"); ax.set_ylabel("Maximum absolute error")
    ax.set_title("Free translation convergence"); ax.grid(True,which="both",alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(out/"free_motion_validation.png",dpi=160)
    if not args.no_show: plt.show()
    print("dt, position error, velocity error")
    for dt,(ep,ev) in zip(dts,results): print(f"{dt:.5g}, {ep:.6e}, {ev:.6e}")
    print(f"Saved diagnostics in {out}")
if __name__=="__main__": main()
