# %% cell 0
import pandas as pd
import matplotlib.pyplot as plt

file = '/Users/zbalog/Work/GAIA/BAMOnly/ring_temp.csv'
df1 = pd.read_csv(file)

scale_factor = 2.0626480624709636E8
df2 = pd.read_csv('/Users/zbalog/Work/GAIA/BAMOnly/sfunc7000_fov2_detrended_newtime.csv')
df2['y'] = df2['DN_LINE_OF_SIGHT_VARIATIONS'] * scale_factor
df2['trend_scaled'] = df2['trend'] * scale_factor

layers = [
    ((16406.73, 16414.53), 'yellow', 'slow slew'),
    ((16414.53, 16490.09), 'orange', r'SAA$_{0\_SM}$'),
    ((16490.09, 16514.16), 'red', r'SAA$_{0}$'),
    ((16515.15, 16527.14), 'blue', r'SAA$_{5}$'),
    ((16528.19, 16540.15), 'green', r'SAA$_{15}$'),
    ((16541.17, 16553.16), 'grey', r'SAA$_{28}$'),
    ((16554.17, 16566.17), 'magenta', r'SAA$_{43}$'),
    ((16567.18, 16571.17), 'cyan', r'SAA$_{45}$'),
    ((16571.38, 16579.20), 'yellow', None),  # No label
]

# ❌ ERROR 1: Incorrect function name (should be `plt.subplots`)
fig, ax1 = plt.subplots(figsize=(12, 6))

# ➕ Plot ring_temp data on left y-axis
for (start, end), color, label in layers:
    mask = (df1["OBMT_rev"] > start) & (df1["OBMT_rev"] < end)
    ax1.plot(df1.loc[mask, "OBMT_rev"], df1.loc[mask, "NEI00054"],
             linestyle='none', marker='o', markersize=2, color=color, label=label)

# ➕ Plot scaled BAM data on right y-axis
ax2 = ax1.twinx()
for (start, end), color, label in layers:
    mask = (df2["obmtRev"] > start) & (df2["obmtRev"] < end)
    ax2.plot(df2.loc[mask, "obmtRev"], df2.loc[mask, "y"],
             linestyle='none', marker='o', markersize=2, color=color)
trend_range = df2[(df2['obmtRev'] > 16406.73) & (df2['obmtRev'] < 16579.20)]
ax2.plot(trend_range['obmtRev'], trend_range['trend_scaled'], color='black', linewidth=2, label='Trend')

# ❌ ERROR 2: Wrong method names for labels and ticks — should be `.set_xlabel`, etc.
ax1.set_xlabel(r"OBMT [rev]", fontsize=20)
ax2.set_ylabel(r"$^{\rm FoV1}\delta\eta$ [mas]", fontsize=20)
ax1.set_ylabel(r"T[$^{\circ}{\rm C}$]", fontsize=20)
# ❌ ERROR 3: `xlim`, `xticks`, `yticks` are not valid as `ax1` methods — should be `set_xlim`, etc.
ax1.set_xlim(16489, 16575)
# ax1.set_ylim(-0.3, 0.3)  # Uncomment and adjust if needed
ax1.tick_params(axis='both', labelsize=16)
ax2.set_ylim(-30,60)
# ❌ ERROR 4: `ax1.legend()` will skip `None` labels automatically, but multiple identical labels can repeat
handles, labels = ax1.get_legend_handles_labels()
unique = dict(zip(labels, handles))
ax1.legend(unique.values(), unique.keys(), fontsize=10, loc='lower left')

plt.grid(True)
plt.tight_layout()
plt.show()

# %% cell 1
import numpy as np
import pandas as pd

# Ensure df2 is sorted by MEASUREMENT_TIME
df2 = df2.sort_values("obmtRev").reset_index(drop=True)
sensor_cols = ["NEI00054", "NEI00230", "NEI00321"]

# Prepare new columns in df2
for col in sensor_cols:
    df2[f"{col}_avg"] = np.nan

# Loop over df2 rows
for i in range(len(df2)):
    t = df2.loc[i, "obmtRev"]
    
    # Estimate time delta to next point (or previous if last point)
    if i < len(df2) - 1:
        d = df2.loc[i + 1, "obmtRev"] - t
    elif i > 0:
        d = t - df2.loc[i - 1, "obmtRev"]
    else:
        continue  # Can't compute for single-row df2

    t_min = t - d / 2
    t_max = t + d / 2

    # Select rows in df1 that fall in the time window
    mask = (df1["OBMT_rev"] >= t_min) & (df1["OBMT_rev"] <= t_max)
    window_data = df1.loc[mask]

    for col in sensor_cols:
        if not window_data.empty:
            df2.at[i, f"{col}_avg"] = window_data[col].mean()

# %% cell 2
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

scale_factor = 2.0626480624709636e8  # converts from rad to mas
clean_df = df2.dropna(subset=["NEI00054_avg", "NEI00230_avg", "NEI00321_avg"])
clean_df["y_mas"] = clean_df["DN_LINE_OF_SIGHT_VARIATIONS"] * scale_factor
obmt = clean_df["obmtRev"]
# 1. Features and target (in mas)
features = ["NEI00054_avg", "NEI00230_avg", "NEI00321_avg"]
target = "y_mas"  # DN_LINE_OF_SIGHT_VARIATIONS scaled to mas


# Drop NaNs
data = clean_df[features + [target]].dropna()

# 2. Split data
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test, obmt_train, obmt_test = train_test_split(X, y, obmt, test_size=0.2, random_state=42)

# 3. Train model
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Evaluate
from sklearn.metrics import mean_squared_error, r2_score
y_pred = model.predict(X_test)
print(f"MSE (mas²): {mean_squared_error(y_test, y_pred):.3f}")
print(f"R² Score: {r2_score(y_test, y_pred):.3f}")

# %% cell 3
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, s=10, alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted DN_LINE_OF_SIGHT_VARIATIONS")
plt.grid(True)
plt.tight_layout()
plt.show()

# %% cell 4
import matplotlib.pyplot as plt
importances = model.feature_importances_
plt.bar(features, importances)
plt.ylabel("Feature importance")
plt.title("Sensor contribution to BAM signal")
plt.show()

# %% cell 5
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("True $\delta\eta$ [mas]")
plt.ylabel("Predicted $\delta\eta$ [mas]")
plt.title("Model Predictions vs True Values")
plt.grid(True)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.show()

# %% cell 6
residuals = y_test - y_pred
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted $\delta\eta$ [mas]")
plt.ylabel("Residual [mas]")
plt.title("Residuals vs. Predicted Values")
plt.grid(True)
plt.show()

# %% cell 7
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
features = ["NEI00054_avg", "NEI00230_avg", "NEI00321_avg"]

for i, sensor in enumerate(features):
    axs[i].scatter(X_test[sensor], residuals, alpha=0.5)
    axs[i].axhline(0, color='red', linestyle='--')
    axs[i].set_xlabel(f"{sensor}")
    axs[i].set_ylabel("Residual [mas]")
    axs[i].grid(True)

plt.suptitle("Residuals vs. Sensor Averages")
plt.tight_layout()
plt.show()

# %% cell 8
plt.figure(figsize=(6, 4))
plt.hist(residuals, bins=400, edgecolor='black', alpha=0.5)
plt.xlabel("Residual [mas]")
plt.ylabel("Frequency")
plt.xlim(-5,5)
plt.title("Distribution of Residuals")
plt.grid(True)
plt.show()

# %% cell 9
plt.figure(figsize=(10, 4))
plt.scatter(obmt_test, residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("OBMT [rev]")
plt.ylabel("Residual [mas]")
plt.title("Residuals over Time")
plt.grid(True)
plt.show()

# %% cell 10
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

scale_factor = 2.0626480624709636e8  # converts from rad to mas
short_df = df2[(df2["obmtRev"] > 16490.09) & (df2["obmtRev"] < 16571.17)]
clean_df = short_df.dropna(subset=["NEI00054_avg", "NEI00230_avg", "NEI00321_avg"])
clean_df["y_mas"] = clean_df["DN_LINE_OF_SIGHT_VARIATIONS"] * scale_factor
obmt = clean_df["obmtRev"]
# 1. Features and target (in mas)
features = ["NEI00054_avg", "NEI00230_avg", "NEI00321_avg"]
target = "y_mas"  # DN_LINE_OF_SIGHT_VARIATIONS scaled to mas


# Drop NaNs
data = clean_df[features + [target]].dropna()

# 2. Split data
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test, obmt_train, obmt_test = train_test_split(X, y, obmt, test_size=0.2, random_state=42)

# 3. Train model
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Evaluate
from sklearn.metrics import mean_squared_error, r2_score
y_pred = model.predict(X_test)
print(f"MSE (mas²): {mean_squared_error(y_test, y_pred):.3f}")
print(f"R² Score: {r2_score(y_test, y_pred):.3f}")

# %% cell 11
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted DN_LINE_OF_SIGHT_VARIATIONS")
plt.grid(True)
plt.tight_layout()
plt.show()

# %% cell 12
residuals = y_test - y_pred
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted $\delta\eta$ [mas]")
plt.ylabel("Residual [mas]")
plt.title("Residuals vs. Predicted Values")
plt.grid(True)
plt.show()

# %% cell 13
import matplotlib.pyplot as plt
importances = model.feature_importances_
plt.bar(features, importances)
plt.ylabel("Feature importance")
plt.title("Sensor contribution to BAM signal")
plt.show()

# %% cell 14
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
features = ["NEI00054_avg", "NEI00230_avg", "NEI00321_avg"]

for i, sensor in enumerate(features):
    axs[i].scatter(X_test[sensor], residuals, alpha=0.5)
    axs[i].axhline(0, color='red', linestyle='--')
    axs[i].set_xlabel(f"{sensor}")
    axs[i].set_ylabel("Residual [mas]")
    axs[i].grid(True)

plt.suptitle("Residuals vs. Sensor Averages")
plt.tight_layout()
plt.show()

# %% cell 15
plt.figure(figsize=(6, 4))
plt.hist(residuals, bins=100, edgecolor='black', alpha=0.5)
plt.xlabel("Residual [mas]")
plt.ylabel("Frequency")
plt.xlim(-5,5)
plt.title("Distribution of Residuals")
plt.grid(True)
plt.show()

# %% cell 16
plt.figure(figsize=(10, 4))
plt.scatter(obmt_test, residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("OBMT [rev]")
plt.ylabel("Residual [mas]")
plt.title("Residuals over Time")
plt.grid(True)
plt.show()

# %% cell 17


