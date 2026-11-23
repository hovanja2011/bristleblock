"""Oblique frictionless collision: hard-disk impulse limit and invariants."""
import argparse
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from bristlebots.robot import Robot
from bristlebots.contacts import ContactParameters
from bristlebots.simulation import step_robots
from bristlebots.visualization import animate_robots

RADIUS,MASS,STIFFNESS,GAP=0.5,1.0,2000.0,0.04
ANGLE=np.deg2rad(35.0); VCOM=np.array([.2,-.1]); REL_SPEED=1.0; FINAL_TIME=.15

def setup():
    n=np.array([np.cos(ANGLE),np.sin(ANGLE)]); distance=2*RADIUS+GAP
    a=Robot(position=np.array([4.,4.])-.5*distance*n,
      velocity=VCOM+.5*REL_SPEED*n,theta=.3,omega=.7,radius=RADIUS,mass=MASS)
    b=Robot(position=np.array([4.,4.])+.5*distance*n,
      velocity=VCOM-.5*REL_SPEED*n,theta=-.2,omega=-.4,radius=RADIUS,mass=MASS)
    p=ContactParameters(normal_stiffness=STIFFNESS,normal_damping=0,
      mu_static=0,mu_kinetic=0,tangential_stiffness=500,tangential_damping=0)
    return [a,b],p,n

def invariants(robots):
    momentum=sum((r.mass*r.velocity for r in robots),start=np.zeros(2))
    energy=sum(.5*r.mass*np.dot(r.velocity,r.velocity)+.5*r.moment_of_inertia*r.omega**2 for r in robots)
    angular=sum(r.mass*(r.position[0]*r.velocity[1]-r.position[1]*r.velocity[0])
                +r.moment_of_inertia*r.omega for r in robots)
    return momentum,energy,angular

def simulate(dt,history=False):
    robots,p,n=setup(); p0,e0,l0=invariants(robots)
    positions=[np.array([r.position.copy() for r in robots])]
    max_overlap=0.; steps=round(FINAL_TIME/dt)
    for _ in range(steps):
        step_robots(robots,dt,contact_parameters=p)
        d=np.linalg.norm(robots[1].position-robots[0].position)
        max_overlap=max(max_overlap,max(0.,2*RADIUS-d))
        if history: positions.append(np.array([r.position.copy() for r in robots]))
    momentum,energy,angular=invariants(robots)
    expected_a=VCOM-.5*REL_SPEED*n; expected_b=VCOM+.5*REL_SPEED*n
    velocity_error=np.sqrt(np.sum((robots[0].velocity-expected_a)**2)
                           +np.sum((robots[1].velocity-expected_b)**2))
    result={"velocity_error":float(velocity_error),
      "momentum_drift":float(np.linalg.norm(momentum-p0)),
      "energy_drift":float(abs(energy-e0)),
      "angular_momentum_drift":float(abs(angular-l0)),
      "max_overlap":float(max_overlap)}
    if history: result.update(robots=robots,positions=np.asarray(positions))
    return result

def main():
    p=argparse.ArgumentParser(); p.add_argument("--output-dir",default="results/validation")
    p.add_argument("--no-show",action="store_true"); args=p.parse_args()
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    dts=(5e-4,2.5e-4,1.25e-4,6.25e-5); rows=[(dt,simulate(dt)) for dt in dts]
    with (out/"oblique_collision.csv").open("w",newline="") as f:
        w=csv.writer(f); w.writerow(("dt","velocity_error","momentum_drift","energy_drift","angular_momentum_drift","maximum_overlap"))
        for dt,r in rows: w.writerow((dt,r["velocity_error"],r["momentum_drift"],r["energy_drift"],r["angular_momentum_drift"],r["max_overlap"]))
    fig,axes=plt.subplots(2,1,figsize=(8,7),sharex=True)
    for metric in ("velocity_error","energy_drift","angular_momentum_drift"):
        axes[0].loglog(dts,[r[metric] for _,r in rows],"o-",label=metric)
    axes[0].set_ylabel("Error / drift"); axes[0].legend()
    axes[1].plot(dts,[r["max_overlap"] for _,r in rows],"o-")
    axes[1].set_xlabel("dt"); axes[1].set_ylabel("Maximum overlap")
    for ax in axes: ax.grid(True,which="both",alpha=.3)
    fig.suptitle("Oblique frictionless collision"); fig.tight_layout()
    fig.savefig(out/"oblique_collision.png",dpi=160)
    fine=simulate(dts[-1],True)
    if not args.no_show:
        animate_robots(fine["robots"],fine["positions"],box_size=(8,8),interval=30,trail_length=120)
        plt.show()
    print("dt, velocity error, momentum drift, energy drift, angular drift, max overlap")
    for dt,r in rows:
        print(f"{dt:.6g}, "+", ".join(f"{r[k]:.6e}" for k in
          ("velocity_error","momentum_drift","energy_drift","angular_momentum_drift","max_overlap")))
    print(f"Saved diagnostics in {out}")
if __name__=="__main__": main()
