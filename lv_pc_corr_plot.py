import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# List of .npy files (replace with your actual filenames)
file_paths = ["lv_pc_corr_00.npy", "lv_pc_corr_02.npy",
              "lv_pc_corr_04.npy", "lv_pc_corr_06.npy",
              "lv_pc_corr_08.npy", "lv_pc_corr_10.npy"]

# Load each file and store them in a list
arrays = [np.load(file) for file in file_paths]

# Horizontally stack them to create a single array with 12 columns
merged_array = np.hstack(arrays)


# Set Seaborn theme
sns.set_theme(style="whitegrid", palette="pastel")

# Define colors for each pair (6 color pairs)
colors = ["purple", "blue", "green", "orange", "red", "brown"]
corrs = ['0.0', '0.0', '0.17', '0.17', '0.41', '0.41', '0.63', '0.63', '0.83', '0.83', '1.0', '1.0']

# Create a single figure
plt.figure(figsize=(16, 8))

# Loop through each **pair** of columns (1&2, 3&4, etc.)
for i in range(0, merged_array.shape[1], 2):
    color = colors[i // 2]  # Select a color from the list

    # Plot the histogram for the first column in the pair
    sns.histplot(merged_array[:, i], kde=True, binwidth=0.01, color=color,
                 label=f"r =  {corrs[i]}", alpha=0.3)

    # Plot the histogram for the second column in the pair
    if i + 1 < merged_array.shape[1]:  # Ensure there's a second column
        sns.histplot(merged_array[:, i + 1], kde=True, binwidth=0.01, color=color,
                     alpha=0.3)  # Slightly higher alpha for distinction

# Adjusting labels and title
plt.xlabel("r(PC-LV)", fontsize=20)
plt.ylabel("Frequency", fontsize=20)
plt.title(f"(N={merged_array.shape[0]})",
          fontsize=20)
# Customizing legend
plt.legend(loc="upper left", fontsize=20)

# Make the grid less intrusive and improve ticks
plt.grid(True, linestyle='--', alpha=0.5)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlim(0, 1)
plt.ylim(0, 100)

# Show the plot
plt.tight_layout()
plt.show()