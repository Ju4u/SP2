import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.stats import norm, ttest_ind, levene, ks_2samp  # Added ks_2samp here
import seaborn as sns

phase_shift = np.load("phase_shift_difamp.npy")

# Define the transformation function
def transform_column_1(value):
    return ((value - 40) * 180 / 40) % 360

# Apply the transformation to both columns
transformed_array = np.column_stack((
    np.vectorize(transform_column_1)(phase_shift[:, 0]),
    np.vectorize(transform_column_1)(phase_shift[:, 1])
))

# Define bin edges and compute histograms
num_bins = 90
bin_edges = np.linspace(0, 360, num_bins + 1)
hist_1, _ = np.histogram(transformed_array[:, 0], bins=bin_edges)
hist_2, _ = np.histogram(transformed_array[:, 1], bins=bin_edges)

# Calculate bin centers (in radians)
bin_centers = np.deg2rad((bin_edges[:-1] + bin_edges[1:]) / 2)

# Determine maximum count for scaling
max_count = max(max(hist_1), max(hist_2), 1)

# Create polar plot
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(14, 14))

# --- ADD BACKGROUND SHADING ---
ax.fill_between(np.linspace(np.deg2rad(45), np.deg2rad(135), 100),
                0, max_count, color='purple', alpha=0.2, label='PC1')
ax.fill_between(np.linspace(np.deg2rad(315), np.deg2rad(360), 100),
                0, max_count, color='blue', alpha=0.2)
ax.fill_between(np.linspace(np.deg2rad(0), np.deg2rad(45), 100),
                0, max_count, color='blue', alpha=0.2, label='PC2')

# --- HISTOGRAM BARS ---
bar_width = np.deg2rad(360 / num_bins)
ax.bar(bin_centers, hist_1, width=bar_width, bottom=0,
       color='purple', alpha=0.3, edgecolor='black')
ax.bar(bin_centers, hist_2, width=bar_width, bottom=0,
       color='blue', alpha=0.3, edgecolor='black')

# Fit GMM to both columns (PC1 and PC2)
gmm_1 = GaussianMixture(n_components=2, covariance_type='full')
gmm_1.fit(transformed_array[:, 0].reshape(-1, 1))

gmm_2 = GaussianMixture(n_components=2, covariance_type='full')
gmm_2.fit(transformed_array[:, 1].reshape(-1, 1))

# Get the probability density from GMM for plotting
x_vals = np.linspace(0, 360, 1000)
pdf_1 = np.exp(gmm_1.score_samples(x_vals.reshape(-1, 1)))
pdf_2 = np.exp(gmm_2.score_samples(x_vals.reshape(-1, 1)))

# Plot GMM components (each component's Gaussian)
for mean, covar in zip(gmm_1.means_, gmm_1.covariances_):
    ax.plot(
        np.deg2rad(x_vals),
        np.exp(norm.logpdf(x_vals, mean[0], np.sqrt(covar[0]))) * len(hist_1) * (360 / num_bins),
        color='purple', linestyle='dashed', lw=2
    )

for mean, covar in zip(gmm_2.means_, gmm_2.covariances_):
    ax.plot(
        np.deg2rad(x_vals),
        np.exp(norm.logpdf(x_vals, mean[0], np.sqrt(covar[0]))) * len(hist_2) * (360 / num_bins),
        color='blue', linestyle='dashed', lw=2
    )

# --- Plot scatter points and add text individually ---
# Loop over each component for PC1
text_pc1 = [20, 24]
for i in range(len(gmm_1.means_)):
    angle_deg = gmm_1.means_[i, 0]  # GMM mean (in degrees)
    radius_val = gmm_1.weights_[i] * len(hist_1)  # Scaled by histogram size

    # Plot the scatter point
    ax.scatter(np.deg2rad(angle_deg), radius_val,
               color='purple', s=200, edgecolor='black', zorder=3,
               label=f"PC1 Max" if i == 0 else '')

    # Place the text near the scatter point:
    ax.text(
        np.deg2rad(angle_deg),
        radius_val + (max_count * 0.1),
        f"{text_pc1[i]}°",  # Or round however you like
        color='purple',
        fontsize=20,
        ha='center', bbox=dict(facecolor='white', edgecolor='purple', boxstyle='round,pad=0.1')
    )

# Loop over each component for PC2
text_pc2 = [19, 25]
for i in range(len(gmm_2.means_)):
    angle_deg = gmm_2.means_[i, 0]
    radius_val = gmm_2.weights_[i] * len(hist_2)

    ax.scatter(np.deg2rad(angle_deg), radius_val,
               color='blue', s=200, edgecolor='black', zorder=3,
               label=f"PC2 Max" if i == 0 else '')

    ax.text(
        np.deg2rad(angle_deg),
        radius_val + (max_count * 0.1),
        f"{text_pc2[i]}°",
        color='blue',
        fontsize=20,
        ha='center', bbox=dict(facecolor='white', edgecolor='blue', boxstyle='round,pad=0.1')
    )

# --- CUSTOM AXIS LABELS ---
custom_angles_deg = [0, 45, 90, 135, 180, 225, 270, 315]  # Original positions
custom_labels = ["-90", "-45°", "0°", "45°", "90°", "135°", "(-)180°", "-135°"]
custom_angles_rad = np.deg2rad(custom_angles_deg)

ax.set_xticks(custom_angles_rad)
ax.set_xticklabels(custom_labels, fontsize=20)

# Add radial scale circles with labels
radial_ticks = np.linspace(0, max_count, num=5, dtype=int)
ax.set_rticks(radial_ticks)
ax.set_yticklabels([str(tick) for tick in radial_ticks], fontsize=20)
ax.tick_params(axis='x', labelsize=20)

# Formatting and finishing touches
ax.set_theta_zero_location('E')
ax.set_theta_direction(-1)
ax.set_title("PCA Phase Shifts", va='bottom', pad=30, fontsize=30)
ax.legend(fontsize=20)

plt.show()

# Extract GMM parameters for PC1
means_pc1 = gmm_1.means_.flatten()  # Means
variances_pc1 = gmm_1.covariances_.flatten()  # Variances
weights_pc1 = gmm_1.weights_.flatten()  # Mixing Coefficients

# Extract GMM parameters for PC2
means_pc2 = gmm_2.means_.flatten()  # Means
variances_pc2 = gmm_2.covariances_.flatten()  # Variances
weights_pc2 = gmm_2.weights_.flatten()  # Mixing Coefficients

# Print the extracted parameters
print("\n--- GMM Parameters for PC1 ---")
print(f"Means (Degrees): {means_pc1}")
print(f"Variances: {variances_pc1}")
print(f"Mixing Coefficients: {weights_pc1}")

print("\n--- GMM Parameters for PC2 ---")
print(f"Means (Degrees): {means_pc2}")
print(f"Variances: {variances_pc2}")
print(f"Mixing Coefficients: {weights_pc2}")

# Define samples for statistical comparison
pc1 = phase_shift[:, 0]
pc2 = phase_shift[:, 1] + 20

# --- TWO-SAMPLE KOLMOGOROV-SMIRNOV TEST ---
# The two-sample KS test compares the empirical distribution functions of pc1 and pc2.
ks_stat, ks_p_value = ks_2samp(pc1, pc2)
print("\n--- Two-sample KS test result ---")
print(f"KS Statistic: {ks_stat}")
print(f"P-value: {ks_p_value}")

# Plot histograms and KDE for visual comparison
plt.hist(pc1, bins=30, density=True, alpha=0.6, color='b', edgecolor='black', label='PC1 Histogram')
plt.hist(pc2, bins=30, density=True, alpha=0.6, color='r', edgecolor='black', label='PC2 Histogram')

# Add KDE
sns.kdeplot(pc1, color='b', linewidth=2, label="PC1 KDE")
sns.kdeplot(pc2, color='r', linewidth=2, label="PC2 KDE")

plt.legend()
plt.show()
