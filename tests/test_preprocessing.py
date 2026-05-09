import numpy as np
import pandas as pd

from gaia_bav_ring_temp.preprocessing import detrend_temperature_features


def test_linear_detrending_removes_simple_slope():
    df = pd.DataFrame(
        {
            "obmtRev": [0.0, 1.0, 2.0, 3.0, 4.0],
            "A_avg": [10.0, 11.0, 12.0, 13.0, 14.0],
        }
    )

    out = detrend_temperature_features(
        df,
        feature_cols=["A_avg"],
        obmt_col="obmtRev",
        method="linear",
        keep_raw_columns=True,
        preserve_mean=True,
    )

    # After removing a perfect linear trend and preserving the mean,
    # the feature should be constant.
    assert np.allclose(out["A_avg"].to_numpy(), np.full(5, 12.0))
    assert "A_avg_raw" in out.columns
    assert "A_avg_trend" in out.columns