import numpy as np
from numpy.typing import NDArray
from typing import Callable, Tuple, List, Optional
from online_pca.learning.training import run_online_pca, general_components_recalibration


def sanger_delta_calculation(components, learning_rate, sample):
	# Step 1: project the sample onto each component → y_i = w_iᵗ x
	y = components.T @ sample  # shape: (k,)
	# Step 2: build the (k, k) shaped, upper-triangular matrix U where U[i, j] = y[i] if j >= i, else 0
	k = y.shape[0]
	U = np.triu(np.ones((k, k), dtype=np.float64)) * y[:, np.newaxis]  # shape: (k, k)
	# Step 3: compute reconstruction terms to subtract from sample
	# Component @ U gives shape (d, k), where each column is ∑_{j=1}^i y_j * w_j
	residuals = sample[:, np.newaxis] - components @ U  # shape: (d, k)
	# Step 4: compute the full delta matrix Δ = etta · residuals · yᵢ
	delta = learning_rate * residuals * y[np.newaxis, :]  # shape: (d, k)
	return delta


def sanger_components_recalibration(components: NDArray[np.float64], step: int,
									svd_threshold: float, svd_interval: int) -> None:
	general_components_recalibration(components, step, svd_threshold, svd_interval, False)


def run_sanger_online_pca(
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
			sanger_delta_calculation,
			sanger_components_recalibration,
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
