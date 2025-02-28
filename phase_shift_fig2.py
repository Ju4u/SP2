import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Load phase shift data
phase_shift = np.load('pc12_acc_difamp.npy')  # Shape (2000, 80)

# Get x-axis range from phase_shift
x = np.arange(phase_shift.shape[1])  # If phase_shift has 80 columns, x = [0,1,2,...,79]

# Compute y-range of phase_shift to match the sine waves
y_min, y_max = np.min(phase_shift), np.max(phase_shift)

# Scale sine waves to fit the y-range of phase_shift
y1 = np.sin(np.linspace(0, 2 * np.pi, phase_shift.shape[1]))  # LV1
y2 = np.sin(np.linspace(0, 2 * np.pi, phase_shift.shape[1]) + np.pi/2)  # LV2

# Normalize and scale to match phase_shift range
y1 = y1 * (y_max - y_min) / 2 + (y_max + y_min) / 2
y2 = y2 * (y_max - y_min) / 2 + (y_max + y_min) / 2

# Create the plot
plt.figure(figsize=(16, 8))

# Plot first half of the data in purple (PC1)
for i in range(phase_shift.shape[0] // 2):
    plt.plot(x, phase_shift[i], color='purple', alpha=0.02, linewidth=1.5, label='PC1' if i == 0 else '')

# Plot second half of the data in blue (PC2)
for i in range(phase_shift.shape[0] // 2, phase_shift.shape[0]):
    plt.plot(x, phase_shift[i], color='blue', alpha=0.05, linewidth=1.5, label='PC2' if i == phase_shift.shape[0] // 2 else '')

# Plot theoretical sine waves (LV1 and LV2) with matched x and y ranges
plt.plot(x, y1, color='purple', linewidth=6, label='LV1')
plt.plot(x, y2, color='blue', linewidth=6, label='LV2')

# Custom legend elements
legend_elements = [
    Line2D([0], [0], color='purple', linewidth=2, label='PC1', alpha=1),
    Line2D([0], [0], color='blue', linewidth=2, label='PC2', alpha=1),
    Line2D([0], [0], color='purple', linewidth=2, linestyle='-', label='LV1'),
    Line2D([0], [0], color='blue', linewidth=2, linestyle='-', label='LV2')
]

# Formatting
plt.xlabel("")
plt.ylabel("")
plt.title("PC1 and PC2 Phase Shift Distribution", fontsize=30)
plt.tick_params(axis='x', labelsize=20)
plt.tick_params(axis='y', labelsize=20)
plt.grid(False)

# Move the legend outside to avoid overlap
plt.legend(handles=legend_elements, fontsize=15, loc='lower left')

# Show the plot
plt.tight_layout()
plt.show()

