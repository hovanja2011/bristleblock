"""Rotational benchmark: active torque plus rotational drag."""
import argparse
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from bristlebots.robot import Robot
from bristlebots.simulation import step
from bristlebots.visualization import animate_robots

MASS, RADIUS = 1.2, 0.4
TORQUE, ROTATIONAL_DRAG = 0.3, 0.08
OMEGA0, THETA0, FINAL_TIME = -0.4, 0.2, 4.0

def exact_state(time):
    inertia = 0.5*MASS*RADIUS**2
    relaxation = inertia/ROTATIONAL_DRAG
    omega_inf = TORQUE/ROTATIONAL_DRAG
    decay = np.exp(-time/relaxation)
    return (THETA0 + omega_inf*time + (OMEGA0-omega_inf)*relaxation*(1-decay),
            omega_inf + (OMEGA0-omega_inf)*decay)

def simulate(dt, history=False):
    robot = Robot(position=(2,2), velocity=(0,0), theta=THETA0, omega=OMEGA0,
                  radius=RADIUS, mass=MASS, active_torque=TORQUE,
                  rotational_drag=ROTATIONAL_DRAG)
    n = round(FINAL_TIME/dt)
    times = np.arange(n+1)*dt
    theta = np.empty(n+1); omega = np.empty(n+1)
    theta[0], omega[0] = robot.theta, robot.omega
    for i in range(1,n+1):
        step(robot,dt); theta[i],omega[i]=robot.theta,robot.omega
    exact_theta,exact_omega=exact_state(times)
    result={"times":times,"theta":theta,"omega":omega,
            "theta_exact":exact_theta,"omega_exact":exact_omega,
            "theta_error":float(np.max(np.abs(theta-exact_theta))),
            "omega_error":float(np.max(np.abs(omega-exact_omega)))}
    if history:
        result.update(robot=robot,
          positions=np.repeat(robot.position[None,None,:],n+1,axis=0),
          theta_history=theta[:,None],omega_history=omega[:,None])
    return result

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output-dir",default="results/validation")
    parser.add_argument("--no-show",action="store_true")
    args=parser.parse_args(); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    dts=(0.08,0.04,0.02,0.01); rows=[]; previous=None
    for dt in dts:
        r=simulate(dt)
        for metric in ("theta","omega"):
            error=r[f"{metric}_error"]
            order="" if previous is None else np.log2(previous[metric]/error)
            rows.append((dt,metric,error,order))
        previous={m:r[f"{m}_error"] for m in ("theta","omega")}
    with (out/"rotational_convergence.csv").open("w",newline="") as f:
        w=csv.writer(f); w.writerow(("dt","metric","max_abs_error","observed_order")); w.writerows(rows)
    fine=simulate(dts[-1],True)
    fig,axes=plt.subplots(2,1,figsize=(9,7),sharex=True)
    axes[0].plot(fine["times"],fine["theta"],label="Numerical")
    axes[0].plot(fine["times"],fine["theta_exact"],"--",label="Analytical")
    axes[0].set_ylabel("theta"); axes[0].legend()
    axes[1].plot(fine["times"],fine["omega"],label="Numerical")
    axes[1].plot(fine["times"],fine["omega_exact"],"--",label="Analytical")
    axes[1].set_ylabel("omega"); axes[1].set_xlabel("Time")
    for ax in axes: ax.grid(alpha=.3)
    fig.suptitle("Active torque and rotational drag"); fig.tight_layout()
    fig.savefig(out/"rotational_dynamics.png",dpi=160)
    fig2,ax=plt.subplots(figsize=(6,4))
    for metric in ("theta","omega"):
        sub=[r for r in rows if r[1]==metric]
        ax.loglog([r[0] for r in sub],[r[2] for r in sub],"o-",label=metric)
    ax.set_xlabel("dt"); ax.set_ylabel("Maximum absolute error")
    ax.set_title("Rotational convergence"); ax.grid(True,which="both",alpha=.3); ax.legend()
    fig2.tight_layout(); fig2.savefig(out/"rotational_convergence.png",dpi=160)
    if not args.no_show:
        animate_robots([fine["robot"]],fine["positions"],theta_history=fine["theta_history"],
                       omega_history=fine["omega_history"],box_size=(4,4),show_trails=False)
        plt.show()
    print("dt, theta error/order, omega error/order")
    for i,dt in enumerate(dts):
        a,b=rows[2*i:2*i+2]
        print(f"{dt:.5g}, {a[2]:.6e}/{a[3] or '-'}, {b[2]:.6e}/{b[3] or '-'}")
    print(f"Saved diagnostics in {out}")
if __name__=="__main__": main()
