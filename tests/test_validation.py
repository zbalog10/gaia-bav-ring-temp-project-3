from __future__ import annotations

import pandas as pd

from gaia_bav_ring_temp.validation import make_random_split, make_time_blocked_split



def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "obmtRev": [5.0, 1.0, 4.0, 3.0, 2.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "sensor_a_avg": [50, 10, 40, 30, 20, 60, 70, 80, 90, 100],
            "sensor_b_avg": [500, 100, 400, 300, 200, 600, 700, 800, 900, 1000],
            "y_mas": [0.5, 0.1, 0.4, 0.3, 0.2, 0.6, 0.7, 0.8, 0.9, 1.0],
        }
    )



def test_random_split_keeps_requested_test_fraction() -> None:
    df = _sample_df()
    run = make_random_split(
        df=df,
        feature_cols=["sensor_a_avg", "sensor_b_avg"],
        target_col="y_mas",
        obmt_col="obmtRev",
        test_size=0.3,
        random_state=42,
    )
    assert run.name == "random_split"
    assert run.split.n_train == 7
    assert run.split.n_test == 3



def test_time_blocked_split_uses_final_contiguous_block() -> None:
    df = _sample_df()
    run = make_time_blocked_split(
        df=df,
        feature_cols=["sensor_a_avg", "sensor_b_avg"],
        target_col="y_mas",
        obmt_col="obmtRev",
        test_size=0.3,
        gap_size=2,
    )

    assert run.name == "time_blocked"
    assert list(run.split.obmt_test) == [8.0, 9.0, 10.0]
    assert list(run.split.obmt_train) == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert run.summary.gap_size == 2
    assert run.summary.train_obmt_max == 5.0
    assert run.summary.test_obmt_min == 8.0



def test_time_blocked_split_rejects_impossible_gap() -> None:
    df = _sample_df()
    try:
        make_time_blocked_split(
            df=df,
            feature_cols=["sensor_a_avg", "sensor_b_avg"],
            target_col="y_mas",
            obmt_col="obmtRev",
            test_size=0.9,
            gap_size=2,
        )
    except ValueError as exc:
        assert "no training data" in str(exc)
    else:
        raise AssertionError("Expected ValueError for impossible time-blocked split")
