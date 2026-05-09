from __future__ import annotations

import numpy as np
import pandas as pd

from gaia_bav_ring_temp.features import average_columns_in_windows, compute_window_bounds


def test_compute_window_bounds_matches_half_step_logic() -> None:
    t = np.array([10.0, 12.0, 15.0])
    left, right = compute_window_bounds(t)
    assert np.allclose(left, [9.0, 10.5, 13.5])
    assert np.allclose(right, [11.0, 13.5, 16.5])


def test_average_columns_in_windows() -> None:
    source = pd.DataFrame(
        {
            "OBMT_rev": [9.5, 10.5, 11.5, 12.5, 14.0, 15.2, 16.0],
            "NEI00054": [1, 3, 5, 7, 9, 11, 13],
        }
    )
    target = np.array([10.0, 12.0, 15.0])
    out = average_columns_in_windows(
        source_df=source,
        source_time_col="OBMT_rev",
        target_times=target,
        value_cols=["NEI00054"],
    )
    assert np.allclose(out["NEI00054_avg"], [2.0, 5.0, 11.0])


def test_compute_window_bounds_matches_notebook_rule():
    import numpy as np
    from gaia_bav_ring_temp.features import compute_window_bounds

    t = np.array([10.0, 12.0, 15.0])
    left, right = compute_window_bounds(t)

    assert np.allclose(left, [9.0, 10.5, 13.5])
    assert np.allclose(right, [11.0, 13.5, 16.5])
