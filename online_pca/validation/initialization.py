import numpy as np


def validation_init(generate_fn, rng, valid_set_size):
	# Draw validation set for generalization error estimation
	validation_set = generate_fn(num_samples=valid_set_size, rng=rng)  # shape (d, n)
	generalization_errors = []  # Store generalization error per step
	return generalization_errors, validation_set


def compute_leading_eigvals_eigvecs(angle_set_size, components_num, generate_fn, rng, validation_set=None):
	if validation_set is None:
		eigen_samples = generate_fn(angle_set_size, rng)  # shape (d, N)
	else:
		eigen_samples = validation_set  # Use validation set if provided, shape (d, N)
	eigen_mean = np.mean(eigen_samples, axis=1, keepdims=True)  # shape (d, 1)
	eigen_centered = eigen_samples - eigen_mean
	cov = (eigen_centered @ eigen_centered.T) / eigen_samples.shape[1]
	eigvals, eigvecs = np.linalg.eigh(cov)
	eigvecs = eigvecs[:, ::-1]
	eigvals = eigvals[::-1]
	leading_eigvecs = eigvecs[:, :components_num]  # Take the first k eigenvectors, shape (d, k)
	leading_eigvals = eigvals[:components_num]  # Take the first k eigenvalues
	return leading_eigvals, leading_eigvecs


def angles_and_eigvecs_init(angle_set_size, components_num, generate_fn, rng, validation_set):
	# --- Part C: Prepare leading eigenvector set ---
	leading_eigvals, leading_eigvecs = compute_leading_eigvals_eigvecs(angle_set_size, components_num,
																	   generate_fn, rng, validation_set)
	angle_trajectories = [[] for _ in range(components_num)]  # Initialize the angle trajectories list
	return angle_trajectories, leading_eigvecs, leading_eigvals


def spaces_angles_and_eigvecs_init(angle_set_size, components_num, generate_fn, rng, validation_set):
	leading_eigvals, leading_eigvecs = compute_leading_eigvals_eigvecs(angle_set_size, components_num,
																	   generate_fn, rng, validation_set)

	# Compute multiplicities (rounded to handle numerical issues)
	eigenspace_dims = []
	eigenspace_vals = []
	current_val = None
	tol = 1e-8
	for val in leading_eigvals:
		if current_val is None or abs(val - current_val) > tol:
			eigenspace_dims.append(1)
			current_val = val
			eigenspace_vals.append(val)
		else:
			eigenspace_dims[-1] += 1

	angle_trajectories = [[] for _ in eigenspace_dims]
	return angle_trajectories, leading_eigvecs, eigenspace_vals, eigenspace_dims


def general_angles_and_eigvecs_init(angle_set_size, components_num, generate_fn, rng, calc_spaces_angles,
									validation_set=None):
	if calc_spaces_angles:
		angle_trajectories, leading_eigvecs, eigvals, eigenspace_dims = spaces_angles_and_eigvecs_init(
				angle_set_size,
				components_num,
				generate_fn, rng, validation_set)
	else:
		angle_trajectories, leading_eigvecs, eigvals = angles_and_eigvecs_init(angle_set_size,
																			   components_num,
																			   generate_fn, rng,
																			   validation_set)
		eigenspace_dims = None  # Not needed if not calculating eigenspaces

	return angle_trajectories, leading_eigvecs, eigvals, eigenspace_dims
