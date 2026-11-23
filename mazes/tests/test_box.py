"""Tests for maze geometry and boundary conditions (not Toner–Tu bulk physics)."""
import numpy as np
import pytest
from mazes import BoxMaze
from toner_tu.initial_conditions import uniform_state
from toner_tu.parameters import TonerTuParameters
from toner_tu.solver import step

def test_box_domain_conserves_mass_in_one_step():
    maze=BoxMaze(4.0,2.0); grid=maze.grid(40,20); ops=maze.operators()
    state=uniform_state(grid,1.0,0.2,-0.1)
    state.rho[:]+=0.02*np.sin(2*np.pi*grid.x/grid.lx)
    mass0=state.rho.sum()
    result=step(state,grid,TonerTuParameters(0.0,1.0,0.0,0.01),1e-4,ops)
    assert result.rho.sum()==pytest.approx(mass0,abs=1e-10)
    assert np.all(result.rho>=0)

def test_box_domain_density_rhs_has_zero_net_flux():
    maze=BoxMaze(3.0,2.0); grid=maze.grid(32,24)
    state=uniform_state(grid,1.0,0.4,-0.3)
    rhs=maze.operators().density_rhs(state.rho,state.vx,state.vy,grid)
    assert rhs.sum()==pytest.approx(0.0,abs=1e-10)

def test_box_dimensions_must_be_positive_and_finite():
    for dimensions in [(0,2),(-1,2),(np.inf,2),(2,np.nan)]:
        with pytest.raises(ValueError): BoxMaze(*dimensions)