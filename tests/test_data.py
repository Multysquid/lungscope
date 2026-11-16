"""Result cells are matched to input cells by both endpoints (no files needed)."""

import numpy as np
import pytest

from lungscope.data.study import match_cells

# three line cells: a trachea and two branches
ENDS = np.array(
    [[[0, 0, 0], [0, 0, 1]], [[0, 0, 1], [1, 0, 2]], [[0, 0, 1], [-1, 0, 2]]],
    dtype=float,
)


def test_same_order():
    assert match_cells(ENDS, ENDS.copy(), scale=2.0).tolist() == [0, 1, 2]


def test_reordered_cells():
    result = ENDS[[2, 0, 1]]  # 4C may write the cells in another order
    assert match_cells(ENDS, result, scale=2.0).tolist() == [1, 2, 0]


def test_swapped_ends_are_refused():
    result = ENDS.copy()
    result[1] = result[1][::-1]
    with pytest.raises(ValueError, match="swapped"):
        match_cells(ENDS, result, scale=2.0)


def test_missing_and_unknown_cells_are_refused():
    with pytest.raises(ValueError):
        match_cells(ENDS, ENDS[:2], scale=2.0)
    moved = ENDS.copy()
    moved[0, 1, 2] = 5.0
    with pytest.raises(ValueError, match="no result cell"):
        match_cells(ENDS, moved, scale=2.0)
