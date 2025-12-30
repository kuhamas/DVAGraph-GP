# density_plot.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import odr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from matplotlib.colors import LinearSegmentedColormap
from train_eval import all_predictions, all_actuals  # Import results from training script

# ======== OPTIMIZED DENSITY PLOT VISUALIZATION ========
# Convert collected predictions to arrays
all_actuals = np.array(all_actuals)
all_predictions = np.array(all_predictions)

# Calculate overall metrics
overall_mse = mean_squared_error(all_actuals, all_predictions)
overall_mae = mean_absolute_error(all_actuals, all_predictions)
overall_r2 = r2_score(all_actuals, all_predictions)
overall_corr, _ = pearsonr(all_actuals, all_predictions)

# Calculate robust axis limits (ignore outliers using 1st and 99th percentiles)
def get_robust_range(data):
    q1 = np.percentile(data, 1)
    q99 = np.percentile(data, 99)
    padding = (q99 - q1) * 0.05  # 5% padding
    return q1 - padding, q99 + padding

# Get robust ranges for both axes
x_min, x_max = get_robust_range(all_actuals)
y_min, y_max = get_robust_range(all_predictions)

# Filter data points within robust ranges for regression
mask = (all_actuals >= x_min) & (all_actuals <= x_max) & \
       (all_predictions >= y_min) & (all_predictions <= y_max)
filtered_actuals = all_actuals[mask]
filtered_predictions = all_predictions[mask]

# Orthogonal Distance Regression (ODR) using filtered data
def linear_model(B, x):
    return B[0] * x + B[1]

data = odr.RealData(filtered_actuals, filtered_predictions)
model_odr = odr.Model(linear_model)
odr_inst = odr.ODR(data, model_odr, beta0=[1.0, 0.0])
out = odr_inst.run()
slope_odr, intercept_odr = out.beta
x_line = np.linspace(x_min, x_max, 200)
y_line = slope_odr * x_line + intercept_odr

# Create plot
plt.figure(figsize=(6, 5))

# Create custom colormap (gray → orange → red → dark red)
orange_lightblue_cmap = LinearSegmentedColormap.from_list(
    "orange_lightblue",
    ["#808080", "#FF8C00", "#FF6347", "#8B0000"]
)

# Plot density with custom colormap
sns.kdeplot(
    x=all_actuals,
    y=all_predictions,
    cmap=orange_lightblue_cmap,
    fill=True,
    alpha=0.8,
    levels=10,
    thresh=0.1
)

# Add white contour lines
sns.kdeplot(
    x=all_actuals,
    y=all_predictions,
    color='white',
    levels=10,
    linewidths=1.0
)

# Reference lines
plt.plot(x_line, x_line, 'k-', linewidth=1.8)  # Identity line
plt.plot(x_line, y_line, 'r--', linewidth=1.8)  # Regression line

# Annotate metrics with semi-transparent background
text_str = (
    f"y = {slope_odr:.2f}x + {intercept_odr:.2f}\n"
    f"Corr = {overall_corr:.3f}\n"
    f"R² = {overall_r2:.3f}\n"
    f"MSE = {overall_mse:.3f}\n"
    f"MAE = {overall_mae:.3f}\n"
)

plt.text(
    x_min + 0.05 * (x_max - x_min),
    y_max - 0.05 * (y_max - y_min),
    text_str,
    fontsize=16,
    fontweight='bold',
    ha='left',
    va='top',
    bbox=dict(facecolor='none', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.5')
)

# Final styling
plt.xlabel('Observed', fontweight='bold')
plt.ylabel('Predicted', fontweight='bold')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# Save and display
plt.savefig('density_plot.png', dpi=300, bbox_inches='tight')
plt.show()
