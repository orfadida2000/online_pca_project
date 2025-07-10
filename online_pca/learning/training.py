import numpy as np
from numpy.typing import NDArray
from typing import Callable, Tuple, List, Optional

import online_pca
from online_pca.utilities import (normalize_vectors, orthonormalize_vectors_using_qr,
								  orthonormalize_vectors_using_svd, reset_stopping_rule_counter)
from online_pca.validation.initialization import (validation_init, general_angles_and_eigvecs_init,
												  compute_leading_eigvals_eigvecs)
from online_pca.validation.update import (errors_trajectory_update, general_angles_trajectory_update,
										  error_calculation)


def general_components_recalibration(components: NDArray[np.float64], step: int, svd_threshold: float,
									 svd_interval: int, use_qr_decomp: bool = False) -> None:
	"""
	Ensures the components matrix is orthonormal.

	Args:
		components (np.ndarray): The (d x k) matrix of PCA components.
		step (int): The current iteration number.
		svd_threshold (float): Max allowed deviation from orthogonality.
		svd_interval (int): Frequency (in steps) to force SVD recalibration.
		use_qr_decomp (bool): If True, do QR decomposition instead of doing renormalization.
	"""
	# Calculate the orthogonality error
	ortho_error = ortho_error_calc(components)

	if step % svd_interval == 0 or ortho_error > svd_threshold:
		# Use SVD for re-orthogonalization, also normalizes the components.
		orthonormalize_vectors_using_svd(components)
	elif use_qr_decomp:
		# Use QR decomposition for re-orthogonalization, also normalizes the components.
		orthonormalize_vectors_using_qr(components)
	else:
		# Normalize each component to unit length
		normalize_vectors(components)


def ortho_error_calc(components: NDArray[np.float64]):
	# Compute orthogonality error via off-diagonal max
	gram = components.T @ components
	np.fill_diagonal(gram, 0.0)
	ortho_error = np.max(np.abs(gram))
	return ortho_error


def single_step_pca(components: NDArray[np.float64], sample: NDArray[np.float64], learning_rate: float,
					step: int, svd_threshold: float, svd_interval: int,
					delta_calc_fn: Callable[..., NDArray[np.float64]],
					components_recalibration_fn: Callable[..., None]) -> NDArray[np.float64]:
	delta = delta_calc_fn(components=components, learning_rate=learning_rate, sample=sample)
	# delta is a (d, k) matrix: each column is the update vector Δw_i
	components += delta

	# Recalibrate the components
	components_recalibration_fn(components=components, step=step, svd_threshold=svd_threshold,
								svd_interval=svd_interval)

	return delta


def pca_initialization(input_dim: int, generate_fn: Callable[..., NDArray[np.float64]], components_num: int,
					   valid_set_size: int, angle_set_size: int, calc_spaces_angles: bool) -> Tuple:
	# Reset the stopping rule counter at the start of each PCA run
	reset_stopping_rule_counter()
	# Initialize the random number generator
	rng = np.random.default_rng()
	collected_samples = []  # Store the samples collected during the PCA run
	generalization_errors, validation_set = validation_init(generate_fn, rng, valid_set_size)
	angle_trajectories, leading_eigvecs, eigvals, eigenspace_dims = general_angles_and_eigvecs_init(
			angle_set_size, components_num, generate_fn, rng, calc_spaces_angles)

	# Initialize the PCA components
	initial_components = rng.normal(size=(input_dim, components_num))  # shape (d, k)
	# Orthonormalize the initial components
	orthonormalize_vectors_using_qr(initial_components)
	ret = (
		rng, initial_components, collected_samples, validation_set, generalization_errors, angle_trajectories,
		leading_eigvecs, eigvals, eigenspace_dims)
	return ret


def run_online_pca(
		input_dim: int,
		generate_fn: Callable[..., NDArray[np.float64]],
		learning_rate_schema: Callable[[int], float],
		stopping_rule: Callable[[NDArray[np.float64], float, int], bool],
		delta_calc_fn: Callable[..., NDArray[np.float64]],
		components_recalibration_fn: Callable[..., None],
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

	rng, components, collected_samples, validation_set, generalization_errors, angle_trajectories, leading_eigvecs, eigvals, eigenspace_dims = pca_initialization(
			input_dim, generate_fn, components_num, valid_set_size, angle_set_size, calc_spaces_angles)

	for t in range(max_iterations):
		# Generate a sample
		sample = generate_fn(num_samples=1, rng=rng)  # shape (d,)
		collected_samples.append(sample)
		delta = single_step_pca(components=components,
								sample=sample,
								learning_rate=learning_rate_schema(t), step=t, svd_threshold=svd_threshold,
								svd_interval=svd_interval, delta_calc_fn=delta_calc_fn,
								components_recalibration_fn=components_recalibration_fn)

		errors_trajectory_update(components, generalization_errors, validation_set)
		general_angles_trajectory_update(angle_trajectories, components, leading_eigvecs, eigenspace_dims,
										 calc_spaces_angles)

		# Check the stopping rule
		if stopping_rule(delta, stop_threshold, min_iterations):
			print(f"Stopping at iteration {t} with delta norm: {np.linalg.norm(delta)}")
			break
	# Final orthonormalization of components using SVD decomposition
	orthonormalize_vectors_using_svd(components)
	if online_pca.DEBUG_MODE:
		log_final_and_best_results(components, validation_set)

	return components, np.array(
			collected_samples), generalization_errors, angle_trajectories, eigvals, eigenspace_dims, leading_eigvecs


def log_final_and_best_results(components, validation_set):
	# Print final components
	d, k = components.shape
	print(f"Final PCA components:")
	for i in range(k):
		print(f"Component {i + 1}: {components[:, i]}")
	# Calculate and print the final generalization error of the components
	final_error = error_calculation(components, validation_set)
	print(f"Final generalization error of computed components is: {final_error}")

	# Calculate leading eigenvalues and eigenvectors of validation set
	validation_set_fn = lambda x, y: validation_set

	leading_eigvals, leading_eigvecs = compute_leading_eigvals_eigvecs(angle_set_size=validation_set.shape[1],
																	   components_num=d,
																	   generate_fn=validation_set_fn,
																	   rng=None)
	# Print leading eigenvectors
	print(f"Computed eigenvectors:")
	for i in range(leading_eigvecs.shape[1]):
		print(f"Leading eigenvector {i + 1}: {leading_eigvecs[:, i]}, eigenvalue: {leading_eigvals[i]}")
	# Calculate and print the best generalization error we can get (using leading eigenvectors)
	best_error = error_calculation(leading_eigvecs[:, :k], validation_set)
	print(
			f"Best generalization error we can get is (using {k} leading eigenvectors of validation set): {best_error}")
