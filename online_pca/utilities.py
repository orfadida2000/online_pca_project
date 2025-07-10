import os
import numpy as np
from numpy.typing import NDArray
from typing import Optional, Callable


def generate_samples(num_samples: int = 1, rng: Optional[np.random.Generator] = None) -> NDArray[np.float64]:
	# Define the static cluster means (only once)
	if not hasattr(generate_samples, "_cluster_means"):
		generate_samples._cluster_means = np.array([
			[-10, 10, -10],
			[-10, -10, -10],
			[20, 0, 20]], dtype=np.float64)

	# Now check and create RNG if needed
	if rng is None:
		rng = np.random.default_rng()

	# Retrieve the cluster means
	cluster_means = generate_samples._cluster_means

	# Choose cluster indices uniformly
	cluster_indices = rng.integers(low=0, high=3, size=num_samples)
	means: NDArray[np.float64] = cluster_means[cluster_indices].reshape(3, -1)

	# Add Gaussian noise with std=1 to each coordinate
	noise = rng.normal(loc=0.0, scale=1.0, size=(3, num_samples))
	samples = means + noise

	if num_samples == 1:
		return samples[:, 0]  # shape (3,)
	else:
		return samples  # shape (3, num_samples)


def stopping_rule(delta: NDArray[np.float64], epsilon: float, min_iterations: int = 5000) -> bool:
	"""
	More robust stopping rule with minimum iteration requirement.
	"""
	# Initialize step counter as attribute if needed
	if not hasattr(stopping_rule, "_step_counter"):
		stopping_rule._step_counter = 0
	stopping_rule._step_counter += 1

	# Don't allow stopping before minimum iterations
	k = delta.shape[1]
	scaled_min_iterations = min_iterations * k  # Scale by components

	if stopping_rule._step_counter < scaled_min_iterations:
		return False

	# Original criterion but with stricter threshold for multi-component
	return np.linalg.norm(delta) / np.sqrt(k) < epsilon


def reset_stopping_rule_counter():
	"""Reset the step counter for the stopping rule."""
	if hasattr(stopping_rule, "_step_counter"):
		stopping_rule._step_counter = 0


def get_unique_plot_filename(base_name: str, extension: str = "png") -> str:
	filename = f"{base_name}.{extension}"
	if not os.path.exists(filename):
		return filename
	i = 1
	while True:
		filename = f"{base_name}({i}).{extension}"
		if not os.path.exists(filename):
			return filename
		i += 1


def normalize_vectors(vectors: NDArray[np.float64]) -> None:
	"""
	Normalizes the vectors to unit length.

	Args:
		vectors (np.ndarray): The (d x k) matrix of vectors as columns.
	"""

	for i in range(vectors.shape[1]):
		vectors[:, i] /= np.linalg.norm(vectors[:, i])


def orthonormalize_vectors(vectors: NDArray[np.float64],
						   orthonormalize_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]]) -> None:
	"""
	Ensures the vectors are orthonormal and orthonormalizes them using the orthonormalize_fn parameter.

	Args:
		vectors (np.ndarray): The (d x k) matrix of vectors as columns.
		orthonormalize_fn (Callable): Function to orthonormalize the vectors.
	"""
	if vectors.shape[1] == 1:
		# If there's only one vector, normalize it to unit length
		normalize_vectors(vectors)
		return
	# Use orthonormalize_fn to orthonormalize the vectors
	ortho_vectors = orthonormalize_fn(vectors)
	vectors[:] = ortho_vectors


def orthonormalize_vectors_using_qr(vectors: NDArray[np.float64]) -> None:
	"""
	Ensures the vectors are orthonormal and orthonormalizes them using OR decomposition.

	Args:
		vectors (np.ndarray): The (d x k) matrix of vectors as columns.
	"""
	# OR decomposition orthonormalization function
	qr_orthonormalization = lambda V: np.linalg.qr(V)[0]
	orthonormalize_vectors(vectors, qr_orthonormalization)


def orthonormalize_vectors_using_svd(vectors: NDArray[np.float64]) -> None:
	"""
	Ensures the vectors are orthonormal and orthonormalizes them using SVD decomposition.

	Args:
		vectors (np.ndarray): The (d x k) matrix of vectors as columns.
	"""
	# SVD decomposition orthonormalization function
	svd_orthonormalization = lambda V: np.linalg.svd(V, full_matrices=False)[0]
	orthonormalize_vectors(vectors, svd_orthonormalization)
