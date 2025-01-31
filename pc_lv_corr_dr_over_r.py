import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit

# Data
r = np.array([0, 0.17, 0.41, 0.63, 0.83, 1])
dr = np.array([0.005, 0.103, 0.264, 0.446, 0.653, 0.973])

# Define the exponential function
def exponential_func(x, a, b, c):
    return a * np.exp(b * x) + c

# Fit the exponential function to the data
popt, pcov = curve_fit(exponential_func, r, dr, p0=(1, 1, 0))

# Extract fitted parameters
a, b, c = popt

# Generate fitted curve
r_fitted = np.linspace(min(r), max(r), 100)
dr_fitted = exponential_func(r_fitted, a, b, c)

# Set Seaborn theme
sns.set_theme(style="whitegrid", palette="pastel")

# Define colors for each pair (6 color pairs)
colors = ["purple", "blue", "green", "orange", "red", "brown"]
corrs = ['0.0', '0.17', '0.41', '0.63', '0.83', '1.0']

# Create figure
plt.figure(figsize=(12, 6))

# Scatter plot for data points
for i in range(6):
    plt.scatter(r[i], dr[i],
                alpha=0.7,
                edgecolor='k', s=100, c='black')

# Plot the fitted exponential curve
plt.plot(r_fitted, dr_fitted,
         color="black", linewidth=2, alpha=0.5)

# Adjust labels and title
plt.xlabel("r", fontsize=14)
plt.ylabel("Δr", fontsize=14)
plt.title("Exponential Fit: Difference of PC --> LV correlations over r",
          fontsize=16, fontweight='bold')


# Improve grid visibility
plt.grid(True, linestyle='--')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Show the plot
plt.tight_layout()
plt.show()
