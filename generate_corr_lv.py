import numpy as np
import matplotlib.pyplot as plt


def generate_latent_variables_with_interpolation(correlation: float, num_points: int = 16,
                                                 total_points: int = 8000) -> np.ndarray:
    """
    Generates a pair of latent variables with adjustable correlation using interpolation.

    Parameters:
        correlation (float): Desired correlation between the two variables (-1 to 1).
        num_points (int): Number of data points to generate (default: 16).
        total_points (int): Total number of points after interpolation (default: 8000).

    Returns:
        np.ndarray: A (total_points, 2) array containing the generated latent variables.
    """
    if not -1 <= correlation <= 1:
        raise ValueError("Correlation must be between -1 and 1.")

    # Generate 16 uncorrelated standard normal variables
    mean = [0, 0]
    cov = [[1, correlation], [correlation, 1]]
    latent_vars = np.random.multivariate_normal(mean, cov, size=num_points)

    # Interpolate to create 8000 points
    x = np.linspace(0, num_points - 1, num_points)  # Original 16 points
    x_interp = np.linspace(0, num_points - 1, total_points)  # 8000 points

    latent_var1_interp = np.interp(x_interp, x, latent_vars[:, 0])
    latent_var2_interp = np.interp(x_interp, x, latent_vars[:, 1])

    # Stack them together into a (8000, 2) array
    interpolated_latent_vars = np.stack((latent_var1_interp, latent_var2_interp), axis=1)

    # Scale to [10, 15]
    interpolated_latent_vars = (interpolated_latent_vars - interpolated_latent_vars.min(axis=0)) / (
                interpolated_latent_vars.max(axis=0) - interpolated_latent_vars.min(axis=0))
    interpolated_latent_vars = interpolated_latent_vars * 5 + 20  # Scale to [20, 22]

    return interpolated_latent_vars


# Generate latent data with interpolation
latent_data = generate_latent_variables_with_interpolation(0.0)
print(latent_data.shape)  # Should be (8000, 2)
r_real = np.corrcoef(latent_data[:, 0], latent_data[:, 1])
print(r_real[0, 1])
latent_data00 = latent_data
np.save('q1ab_latent_data00.npy', latent_data00)

# Time series plot for both latent variables
plt.figure(figsize=(10, 6))
plt.plot(latent_data[:, 0], label='Latent Variable 1', color='blue', alpha=0.7)
plt.plot(latent_data[:, 1], label='Latent Variable 2', color='red', alpha=0.7)
plt.title(f'Interpolated Time Series of Latent Variables, r={r_real[0, 1]:.3f}')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()
