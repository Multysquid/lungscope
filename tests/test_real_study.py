"""The real 12-sample study, skipped when its output is not there.

The study is looked for at ``output/random_field/study.json`` in the repository (where the
supplied run_uq.py writes it), or wherever the environment variable LUNGSCOPE_STUDY points.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from lungscope import analysis, data

ROOT = Path(__file__).resolve().parents[1]
STUDY = Path(os.environ.get("LUNGSCOPE_STUDY", ROOT / "output" / "random_field" / "study.json")).expanduser()
pytestmark = pytest.mark.skipif(not STUDY.is_file(), reason="study output not available (see LUNGSCOPE_STUDY)")


@pytest.fixture(scope="module")
def study():
    return data.load_study(STUDY)


def test_dimensions(study):
    assert (study.sizes["realization"], study.sizes["time"], study.sizes["entity"]) == (12, 50, 23)


def test_reproduces_the_queens_response(study):
    volume = study["volume"].sel(time=0.5, method="nearest").sum("entity")
    reference = study["v0"].where(analysis.terminal_units(study)).sum()
    np.testing.assert_allclose(volume / reference, study["total_volume_ratio_at_0_5_s"], rtol=1e-12)


def test_quantities_for_the_viewer(study):
    pressure, differential, spread = analysis.time_quantities(study)
    units = analysis.terminal_units(study)
    assert pressure.start.shape == (50, 23)
    assert pressure.members.shape == (12, 50, 23)
    assert np.isnan(differential.start[:, units]).all()
    assert np.isfinite(differential.start[:, ~units]).all()
    assert spread.uncertain
    assert np.isfinite(spread.start).all() and (spread.start >= 0).all()
    # the inlet pressure is prescribed, the same in every run: no spread at the trachea's inlet
    np.testing.assert_allclose(spread.start[:, 0], 0.0, atol=1e-9)
    assert spread.end[:, ~units].max() > 0
