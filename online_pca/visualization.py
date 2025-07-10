from typing import Sequence, List, Optional, Dict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy.typing import NDArray
import os

import online_pca
from online_pca.utilities import get_unique_plot_filename
import matplotlib.figure as MPL_Fig


def plot_samples_components(samples: NDArray[np.float64],
							components: NDArray[np.float64], rule: str, color: str = "tab:blue",
							filename: Optional[str] = None, extension: str = 'png', dpi: int = 300,
							directory: str = "plots",
							leading_eigvecs: Optional[NDArray[np.float64]] = None) -> None:
	colors = ["tab:orange", "tab:red", "tab:brown", "tab:pink",
			  "tab:gray", "tab:olive", "tab:cyan", "tab:purple", "tab:green"]
	fig = plt.figure(figsize=(9, 9))
	ax = fig.add_subplot(111, projection='3d')

	mean = np.mean(samples, axis=0, keepdims=True)
	centered_samples = samples - mean
	# Scatter all the samples
	ax.scatter(centered_samples[:, 0], centered_samples[:, 1], centered_samples[:, 2], color=color, alpha=0.3,
			   label="Samples", s=10)

	origin = np.zeros(components.shape[0])
	if online_pca.DEBUG_MODE:
		avg_sample = np.mean(samples, axis=0)
		print(f"Average sample is: {avg_sample}")
	for i in range(components.shape[1]):
		vec = components[:, i]
		ax.quiver(origin[0], origin[1], origin[2], vec[0], vec[1], vec[2], length=10,
				  color=colors[i % len(colors)], label=f"Component {i + 1}")
		if leading_eigvecs is not None:  # meaning leading_eigvecs is provided and debug mode is enabled
			eigvec = leading_eigvecs[:, i]
			ax.quiver(*origin, *eigvec, length=10, color=colors[(-i - 1) % len(colors)],
					  label=f"Leading Eigenvector {i + 1}")

	ax.scatter([0], [0], [0], color='black', s=60,
			   label='Origin')  # Center the plot view on origin with equal scaling
	max_range = np.max([
		np.max(np.abs(centered_samples[:, 0])),
		np.max(np.abs(centered_samples[:, 1])),
		np.max(np.abs(centered_samples[:, 2]))
		]) * 1.1  # Add 10% margin

	# Set equal and symmetric limits around origin
	ax.set_xlim(-max_range, max_range)
	ax.set_ylim(-max_range, max_range)
	ax.set_zlim(-max_range, max_range)

	# Set equal aspect ratio for 3D plot
	ax.view_init(elev=20, azim=30)  # Or any combo you like
	ax.set_box_aspect([1, 1, 1])

	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')
	ax.set_title(f'3D Samples and Learned Components ({rule})', fontsize=12)
	ax.legend()
	plot_and_save_figure(fig, filename, extension, dpi, directory, show_fig=(not online_pca.BATCH_MODE))


def plot_generalization_errors_trajectory(errors: Sequence[float], rule: str, color: str = "tab:blue",
										  filename: Optional[str] = None, extension: str = 'png',
										  dpi: int = 300, directory: str = "plots") -> None:
	"""
	Plots the trajectory of a scalar metric over time (e.g., generalization error or angle).

	Args:
		errors (Sequence[float]): List of generalization errors at each step.
		color (str): Line color for the plot.
		filename (Optional[str]): If provided, the plot will be saved to this file.
		extension (str): File extension for saving the plot (default is 'png').
		dpi (int): Dots per inch for the saved plot (default is 300).
		directory (str): Directory where the plot will be saved (default is "plots").
	"""
	steps = np.arange(1, len(errors) + 1)
	fig, ax = plt.subplots(figsize=(9, 4.5))
	ax.plot(steps, errors, color=color, marker='o', markersize=3, linewidth=1.2)
	ax.set_title(f"Generalization Error Trajectory ({rule})", fontsize=12)
	ax.set_xlabel("Learning Step")
	ax.set_ylabel("Generalization Error")
	ax.grid(True)
	plot_and_save_figure(fig, filename, extension, dpi, directory, show_fig=(not online_pca.BATCH_MODE))


def plot_angle_trajectories(angle_trajectories: List[List[float]], eigvals: List[float],
							eigenspace_dims: Optional[List[int]], rule: str, color: str = "tab:blue",
							filename: Optional[str] = None, extension: str = 'png', dpi: int = 300,
							directory: str = "plots") -> None:
	num_subplots = len(angle_trajectories)
	steps = np.arange(1, len(angle_trajectories[0]) + 1)

	fig, axes = plt.subplots(nrows=num_subplots, ncols=1, figsize=(9, max(3 * num_subplots, 4)),
							 sharex=True)

	if num_subplots == 1:
		axes = [axes]  # make it iterable

	for i, ax in enumerate(axes):
		ax.plot(steps, angle_trajectories[i], color=color, marker='o', markersize=3, linewidth=1.2)
		if eigenspace_dims is None:
			ax.set_title(f"Angle Trajectory for Component {i + 1} (Eigenvalue: {eigvals[i]:.3f})",
						 fontsize=10)
		else:
			ax.set_title(
					f"Angle Trajectory for Components Space {i + 1} (Eigenvalue: {eigvals[i]:.3f}, Dimension: {eigenspace_dims[i]})",
					fontsize=10)
		ax.set_ylabel('Angle (°)')

	# Common labels
	axes[-1].set_xlabel("Learning Step")
	if eigenspace_dims is None:
		fig.suptitle(f"Angle Trajectories Between Learned Components and Leading Eigenvectors ({rule})",
					 fontsize=12)
	else:
		fig.suptitle(f"Angle Trajectories Between Learned Components Spaces and Leading Eigenspaces ({rule})",
					 fontsize=12)
	plot_and_save_figure(fig, filename, extension, dpi, directory, {'rect': (0, 0, 1, 0.97)},
						 show_fig=(not online_pca.BATCH_MODE))


def plot_and_save_figure(fig: MPL_Fig, filename: Optional[str] = None, extension: str = 'png', dpi: int = 300,
						 directory: str = "plots", tight_layout_args: Optional[Dict] = None,
						 show_fig: Optional[bool] = True) -> None:
	# Save before showing
	if tight_layout_args is None:
		tight_layout_args = {}
	fig.tight_layout(**tight_layout_args)  # Adjust layout to prevent overlap
	if filename is not None:
		os.makedirs(directory, exist_ok=True)  # Ensure the plots directory exists
		filename = os.path.join(directory, filename)
		# Save the figure with a unique filename
		fig.savefig(get_unique_plot_filename(filename, extension), dpi=dpi)
	if show_fig:
		plt.show()
	else:
		plt.close(fig)
