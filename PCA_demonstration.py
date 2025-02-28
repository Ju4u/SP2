import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Increase global font sizes for clarity
plt.rcParams.update({'font.size': 14})

##############################################################################
# Scenario 1: Two crossing lines (non-orthogonal slopes)
##############################################################################
np.random.seed(0)
N = 200

# Line 1: steeper slope
x1 = np.random.uniform(-3, 3, N)
y1 = 2.0 * x1 + np.random.normal(0, 0.2, N)
data1 = np.column_stack((x1, y1))

# Line 2: shallower slope
x2 = np.random.uniform(-3, 3, N)
y2 = 0.5 * x2 + np.random.normal(0, 0.2, N)
data2 = np.column_stack((x2, y2))

# Combine into one dataset
data_xshape = np.vstack((data1, data2))

# Fit PCA for Scenario 1
pca1 = PCA(n_components=2)
pca1.fit(data_xshape)
transformed_xshape = pca1.transform(data_xshape)

# Retrieve PCA details for Scenario 1
mean_1 = pca1.mean_
components_1 = pca1.components_
variances_1 = pca1.explained_variance_

##############################################################################
# Scenario 2: Two orthogonal slopes with uneven variance
##############################################################################
# Line 1: 45° slope with higher variance (wider spread)
x3 = np.random.uniform(-3, 3, N)
y3 = x3 + np.random.normal(0, 0.2, N)
data3 = np.column_stack((x3, y3))

# Line 2: -45° slope with lower variance (narrower spread)
x4 = np.random.uniform(-1.5, 1.5, N)  # smaller range for lower variance
y4 = -x4 + np.random.normal(0, 0.2, N)
data4 = np.column_stack((x4, y4))

# Combine into one dataset
data_orthogonal = np.vstack((data3, data4))

# Fit PCA for Scenario 2
pca2 = PCA(n_components=2)
pca2.fit(data_orthogonal)
transformed_orthogonal = pca2.transform(data_orthogonal)

# Retrieve PCA details for Scenario 2
mean_2 = pca2.mean_
components_2 = pca2.components_
variances_2 = pca2.explained_variance_

##############################################################################
# Plotting
##############################################################################
fig, axs = plt.subplots(2, 2, figsize=(12, 12))

# Define consistent axis limits
axis_limits = [-4, 4, -4, 4]

# Colors and labels for principal components
colors = ['orange', 'green']
pc_labels = ['PC1', 'PC2']

# ---------------- Scenario 1: Data in XY space ----------------
axs[0, 0].scatter(data_xshape[:, 0], data_xshape[:, 1], alpha=0.5, label='Data')
for i, (var, vec) in enumerate(zip(variances_1, components_1)):
    scale = 2.0 * np.sqrt(var)
    start = mean_1
    end = mean_1 + scale * vec
    axs[0, 0].plot([start[0], end[0]], [start[1], end[1]],
                   color=colors[i], linewidth=3, label=pc_labels[i])
axs[0, 0].set_title('Scenario 1: Data in XY Space', fontsize=16)
axs[0, 0].set_aspect('equal')
axs[0, 0].axis(axis_limits)
axs[0, 0].legend()

# ---------------- Scenario 1: Data in Principal Component Space ----------------
axs[0, 1].scatter(transformed_xshape[:, 0], transformed_xshape[:, 1], alpha=0.5, label='Data')
axs[0, 1].set_title('Scenario 1: Data in PC Space', fontsize=16)
axs[0, 1].set_aspect('equal')
axs[0, 1].axis(axis_limits)
axs[0, 1].legend()

# ---------------- Scenario 2: Data in XY space (Orthogonal Lines with Uneven Variance) ----------------
axs[1, 0].scatter(data_orthogonal[:, 0], data_orthogonal[:, 1], alpha=0.5, label='Data')
for i, (var, vec) in enumerate(zip(variances_2, components_2)):
    scale = 2.0 * np.sqrt(var)
    start = mean_2
    end = mean_2 + scale * vec
    axs[1, 0].plot([start[0], end[0]], [start[1], end[1]],
                   color=colors[i], linewidth=3, label=pc_labels[i])
axs[1, 0].set_title('Scenario 2: Data in XY Space', fontsize=16)
axs[1, 0].set_aspect('equal')
axs[1, 0].axis(axis_limits)
axs[1, 0].legend()

# ---------------- Scenario 2: Data in Principal Component Space ----------------
axs[1, 1].scatter(transformed_orthogonal[:, 0], transformed_orthogonal[:, 1], alpha=0.5, label='Data')
axs[1, 1].set_title('Scenario 2: Data in PC Space', fontsize=16)
axs[1, 1].set_aspect('equal')
axs[1, 1].axis(axis_limits)
axs[1, 1].legend()

plt.tight_layout()
plt.show()
