"""Regression checks for the contact stiffness/friction parameter study."""
from bristlebots.experiments.contact_parameter_sweep import run_case


def test_penalty_overlap_decreases_when_normal_stiffness_increases():
    soft = run_case(500.0, 0.0)
    stiff = run_case(2000.0, 0.0)
    assert stiff["max_overlap"] < soft["max_overlap"]


def test_friction_changes_rotation_and_preserves_pair_invariants():
    frictionless = run_case(1000.0, 0.0)
    frictional = run_case(1000.0, 0.3)
    assert frictional["spin_change"] > frictionless["spin_change"] + 1e-5
    assert frictional["momentum_drift"] < 1e-10
    assert frictional["angular_momentum_drift"] < 1e-9
    assert frictional["tangential_impulse"] > 0.0


def test_time_refinement_diagnostics_are_finite():
    result = run_case(2000.0, 0.3)
    assert result["dt"] > 0.0
    assert result["max_overlap"] > 0.0
    assert result["kinetic_energy_loss"] >= -1e-9
