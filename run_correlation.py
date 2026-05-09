from pathlib import Path
import pandas as pd

from gaia_bav_ring_temp.correlation import infer_feature_columns, run_correlation_analysis

# Change this if you want FoV2 or a different experiment
input_csv = Path("outputs/fov1_zoom/enriched_dataset.csv")
out_dir = Path("outputs/fov1_zoom/correlation")

df = pd.read_csv(input_csv)

feature_cols = infer_feature_columns(df)

result = run_correlation_analysis(
    df=df,
    feature_cols=feature_cols,
    obmt_col="obmtRev",
    out_dir=out_dir,
    top_n_pairs=15,
)

print("Correlation analysis finished.")
print(f"Output directory: {out_dir}")
print()

print("Top correlated raw feature pairs:")
print(result.top_raw_pairs)
print()

print("PCA explained variance:")
print(result.pca.explained_variance_ratio)
print()

print("VIF:")
print(result.vif)