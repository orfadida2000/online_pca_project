import numpy as np
from numpy.typing import NDArray
from typing import Callable, Tuple, List, Optional
from online_pca.learning.training import run_online_pca, general_components_recalibration


def oja_delta_calculation(components, learning_rate, sample):
	# Project the sample onto the components
	y = components.T @ sample  # shape (k,)
	reconstructed_sample = components @ y  # shape (d,)
	# Update the components using the sample
	delta = learning_rate * (sample - reconstructed_sample).reshape(-1, 1) @ y.reshape(1, -1)  # shape (d, k)
	return delta


def oja_components_recalibration(components: NDArray[np.float64], step: int,
								 svd_threshold: float, svd_interval: int) -> None:
	general_components_recalibration(components, step, svd_threshold, svd_interval, True)


def run_oja_online_pca(
		input_dim: int,
		generate_fn: Callable[..., NDArray[np.float64]],
		learning_rate_schema: Callable[[int], float],
		stopping_rule: Callable[[NDArray[np.float64], float, int], bool],
		max_iterations: int = 10000,
		components_num: int = 1,
		svd_threshold: float = 1e-4,
		svd_interval: int = 100,
		stop_threshold: float = 1e-4,
		valid_set_size: int = 300,
		angle_set_size: int = 100,
		min_iterations: int = 5000,
		calc_spaces_angles: bool = False) -> Tuple[
	NDArray[np.float64], NDArray[np.float64], List[float], List[List[float]], List[float], Optional[
		List[int]], NDArray[np.float64]]:
	# input_dim = d, components_num = k
	components, collected_samples, generalization_errors, angle_trajectories, eigvals, eigenspace_dims, leading_eigvecs = run_online_pca(
			input_dim,
			generate_fn,
			learning_rate_schema,
			stopping_rule,
			oja_delta_calculation,
			oja_components_recalibration,
			max_iterations,
			components_num,
			svd_threshold,
			svd_interval,
			stop_threshold,
			valid_set_size,
			angle_set_size,
			min_iterations,
			calc_spaces_angles)
	return components, collected_samples, generalization_errors, angle_trajectories, eigvals, eigenspace_dims, leading_eigvecs
