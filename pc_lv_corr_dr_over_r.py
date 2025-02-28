import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit

# -------------------
# Original data
# -------------------
r = np.array([0, 0.17, 0.41, 0.63, 0.83, 1])
r_pc1 = np.array([0.978, 0.846, 0.879, 0.917, 0.955, 0.993])
r_pc2 = np.array([0.973, 0.743, 0.615, 0.471, 0.302, 0.02])
er = np.vstack((r_pc1, r_pc2))  # shape (2, 6)

# -------------------
# Define functions
# -------------------
def linear_func(x, m, b):
    return m * x + b

def calculate_r_squared(x, y, popt):
    residuals = y - linear_func(x, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    return 1 - (ss_res / ss_tot)

# -------------------
# 1) Combined fit using ALL data
# -------------------
# Flatten the 'er' array
er_flat = er.flatten()  # shape (12,)
# Create extended 'r' array to match flattened 'er'
r_extended = np.tile(r, 2)  # shape (12,)
# Fit
popt_combined, _ = curve_fit(linear_func, r_extended, er_flat)
m_comb, b_comb = popt_combined
# R2 for combined (on flattened data)
r2_comb = calculate_r_squared(r_extended, er_flat, popt_combined)

# Prepare points for plotting combined fit
r_fitted_comb = np.linspace(r.min(), r.max(), 100)
er_fitted_comb = linear_func(r_fitted_comb, m_comb, b_comb)

# -------------------
# 2) PC1 fit excluding r=0
# -------------------
r_pc1_excl = r[1:]       # exclude first point
pc1_excl   = r_pc1[1:]   # exclude first PC1 point

popt_pc1, _ = curve_fit(linear_func, r_pc1_excl, pc1_excl)
m_pc1, b_pc1 = popt_pc1
r2_pc1 = calculate_r_squared(r_pc1_excl, pc1_excl, popt_pc1)

# Prepare points for plotting PC1 fit
r_fitted_pc1 = np.linspace(r_pc1_excl.min(), r_pc1_excl.max(), 100)
pc1_fitted = linear_func(r_fitted_pc1, m_pc1, b_pc1)

# -------------------
# 3) PC2 fit excluding r=0
# -------------------
r_pc2_excl = r[1:]      # exclude first point
pc2_excl   = r_pc2[1:]  # exclude first PC2 point

popt_pc2, _ = curve_fit(linear_func, r_pc2_excl, pc2_excl)
m_pc2, b_pc2 = popt_pc2
r2_pc2 = calculate_r_squared(r_pc2_excl, pc2_excl, popt_pc2)

# Prepare points for plotting PC2 fit
r_fitted_pc2 = np.linspace(r_pc2_excl.min(), r_pc2_excl.max(), 100)
pc2_fitted = linear_func(r_fitted_pc2, m_pc2, b_pc2)

# -------------------
# Plot
# -------------------
sns.set_theme(style="whitegrid", palette="pastel")
colors = [
    (0.9, 0.8, 1.0),  # r=0.0
    (0.7, 0.8, 1.0),  # r=0.17
    (0.7, 1.0, 0.7),  # r=0.41
    (1.0, 1.0, 0.7),  # r=0.63
    (1.0, 0.7, 0.7),  # r=0.83
    (0.9, 0.7, 0.7)   # r=1.0
]

plt.figure(figsize=(12, 6))

# --- Combined fit ---
plt.plot(r_fitted_comb, er_fitted_comb, color="gray", linewidth=3, alpha=0.8,
         label=f'Combined fit: y={m_comb:.2f}x+{b_comb:.2f}, R²={r2_comb:.2f}', zorder=1)

# --- PC1 fit (excluding r=0) ---
plt.plot(r_fitted_pc1, pc1_fitted, color="magenta", linewidth=3, alpha=0.5,
         label=f'PC1 fit: y={m_pc1:.2f}x+{b_pc1:.2f}, R²={r2_pc1:.2f}', zorder=1)

# --- PC2 fit (excluding r=0) ---
plt.plot(r_fitted_pc2, pc2_fitted, color="blue", linewidth=3, alpha=0.5,
         label=f'PC2 fit: y={m_pc2:.2f}x+{b_pc2:.2f}, R²={r2_pc2:.2f}', zorder=1)

# --- Scatter points ---
plt.scatter(r, r_pc1, alpha=1, lw=2, edgecolor='black', s=250,
            c=colors, label='PC1', zorder=2)
plt.scatter(r, r_pc2, alpha=1, lw=3, edgecolor='gray', s=250,
            c=colors, label='PC2', zorder=2)

# Labels, ticks, etc.
plt.xlabel("r(LV-LV)", fontsize=20)
plt.ylabel("r(PC-LV)", fontsize=20)
plt.title("", fontsize=30, fontweight='bold')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid(True, linestyle='--')
plt.legend(fontsize=14)
plt.tight_layout()
plt.show()
