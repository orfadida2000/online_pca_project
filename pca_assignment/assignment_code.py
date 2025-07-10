import online_pca
import argparse
from online_pca.rules.sanger_rule import run_sanger_online_pca
from online_pca.rules.simultaneous_oja_rule import run_oja_online_pca
from typing import Optional, List, Callable
import numpy as np
from numpy.typing import NDArray
from online_pca.utilities import generate_samples, stopping_rule
from online_pca.learning.learning_rate_schemas import lr_inverse_decay_no_floor, schedule_inverse_smooth
from online_pca.visualization import (plot_samples_components, plot_generalization_errors_trajectory,
									  plot_angle_trajectories)


def assignment_plotting(components: NDArray[np.float64], collected_samples: NDArray[np.float64],
						generalization_errors: List[float],
						angle_trajectories: List[List[float]], eigvals: List[float],
						eigenspace_dims: Optional[List[int]], rule: str,
						filename_prefix: Optional[str] = None,
						extension: str = 'png', dpi: int = 300,
						directory: str = "plots",
						leading_eigvecs: Optional[NDArray[np.float64]] = None) -> None:
	# Plot results
	# Part A: Plot samples and components
	if online_pca.DEBUG_MODE:
		plot_samples_components(collected_samples, components, rule, filename=(filename_prefix + "_results"),
								extension=extension, dpi=dpi, directory=directory,
								leading_eigvecs=leading_eigvecs)
	else:
		plot_samples_components(collected_samples, components, rule, filename=(filename_prefix + "_results"),
								extension=extension, dpi=dpi, directory=directory)
	# Part B: Plot generalization error trajectory
	plot_generalization_errors_trajectory(generalization_errors, rule,
										  filename=(filename_prefix + "_errors_trajectory"),
										  extension=extension, dpi=dpi, directory=directory)
	# Part C: Plot angle trajectories
	plot_angle_trajectories(angle_trajectories, eigvals, eigenspace_dims, rule,
							filename=(filename_prefix + "_angle_trajectories"),
							extension=extension, dpi=dpi, directory=directory)


def run_assignment_sanger(
		input_dim: int,
		learning_rate_schema: Callable[[int], float],
		stopping_rule: Callable[[NDArray[np.float64], float, int], bool],
		max_iterations: int = 40000,
		components_num: int = 1,
		svd_threshold: float = 1e-5,
		svd_interval: int = 200,
		stop_threshold: float = 1e-4,
		valid_set_size: int = 5000,
		angle_set_size: int = 5000,
		min_iterations: int = 5000,
		calc_spaces_angles: bool = False) -> None:
	components, collected_samples, generalization_errors, angle_trajectories, eigvals, eigenspace_dims, leading_eigvecs = run_sanger_online_pca(
			input_dim,
			generate_samples,
			learning_rate_schema,
			stopping_rule,
			max_iterations,
			components_num,
			svd_threshold,
			svd_interval,
			stop_threshold,
			valid_set_size,
			angle_set_size,
			min_iterations,
			calc_spaces_angles)

	assignment_plotting(components, collected_samples, generalization_errors, angle_trajectories, eigvals,
						eigenspace_dims, "Sanger's rule", filename_prefix="sanger_pca",
						directory="sanger_plots", leading_eigvecs=leading_eigvecs)


def run_assignment_oja(
		input_dim: int,
		learning_rate_schema: Callable[[int], float],
		stopping_rule: Callable[[NDArray[np.float64], float, int], bool],
		max_iterations: int = 40000,
		components_num: int = 1,
		svd_threshold: float = 1e-5,
		svd_interval: int = 200,
		stop_threshold: float = 7.5e-4,
		valid_set_size: int = 1000,
		angle_set_size: int = 500,
		min_iterations: int = 5000,
		calc_spaces_angles: bool = False) -> None:
	components, collected_samples, generalization_errors, angle_trajectories, eigvals, eigenspace_dims, leading_eigvecs = run_oja_online_pca(
			input_dim,
			generate_samples,
			learning_rate_schema,
			stopping_rule,
			max_iterations,
			components_num,
			svd_threshold,
			svd_interval,
			stop_threshold,
			valid_set_size,
			angle_set_size,
			min_iterations,
			calc_spaces_angles)

	assignment_plotting(components, collected_samples, generalization_errors, angle_trajectories, eigvals,
						eigenspace_dims, "Oja's rule", filename_prefix="oja_pca", directory="oja_plots",
						leading_eigvecs=leading_eigvecs)


def parse_and_set_run_modes():
	# Parse command line arguments
	parser = argparse.ArgumentParser(description="Run PCA assignments with Sanger's and Oja's rules.")
	parser.add_argument('--debug', action='store_true', help="Enable debug mode for detailed output.")
	parser.add_argument('--batch', action='store_true',
						help="Enable batch mode for faster hyperparameter tuning.")
	args = parser.parse_args()
	# Set the global debug mode dynamically
	online_pca.DEBUG_MODE = args.debug
	online_pca.BATCH_MODE = args.batch
	if online_pca.DEBUG_MODE:
		print("Debug mode is enabled. Detailed output will be printed.")
	if online_pca.BATCH_MODE:
		print(
				"Batch mode is enabled, Hyperparameter tuning will be faster.\n Plots will not be shown, only saved to files.")


def run_assignment():
	parse_and_set_run_modes()  # Set debug mode based on command line argument

	print("Running Sanger's rule with learning rate schema (1 component): Inverse decay without floor")
	run_assignment_sanger(input_dim=3, learning_rate_schema=lr_inverse_decay_no_floor,
						  stopping_rule=stopping_rule,
						  components_num=1, min_iterations=0)
	print("Running Sanger's rule with learning rate schema (2 components): Inverse decay without floor")
	run_assignment_sanger(input_dim=3, learning_rate_schema=lr_inverse_decay_no_floor,
						  stopping_rule=stopping_rule,
						  components_num=2, min_iterations=0)


if __name__ == "__main__":
	run_assignment()
