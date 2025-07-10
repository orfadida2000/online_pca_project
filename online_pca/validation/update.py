import numpy as np
from scipy.linalg import subspace_angles


def angles_trajectory_update(angle_trajectories, components, leading_eigvecs):
	# Compute angles with leading eigenvectors
	for i in range(components.shape[1]):
		w = components[:, i]  # shape (d,)
		v = leading_eigvecs[:, i]  # shape (d,)
		cosine = np.dot(w, v)  # both w and v are normalized, so this is the cosine of the angle
		unsigned_angle = np.arccos(np.clip(np.abs(cosine), 0.0, 1.0))  # radians
		angle_trajectories[i].append(np.degrees(unsigned_angle))


def spaces_angles_trajectory_update(angle_trajectories, components, leading_eigvecs, eigenspace_dims):
	assert components.shape[1] == sum(
			eigenspace_dims), "Components must match the total dimension of eigenspaces"
	start = 0
	for i, dim in enumerate(eigenspace_dims):
		W_i = components[:, start:start + dim]
		V_i = leading_eigvecs[:, start:start + dim]
		# Ensure W_i has orthonormal columns
		Q_i, _ = np.linalg.qr(W_i)
		angles_rad = subspace_angles(Q_i, V_i)
		# Take the largest angle as the representative for the subspace
		max_angle_deg = np.degrees(np.max(angles_rad))  # Convert to degrees
		angle_trajectories[i].append(max_angle_deg)
		start += dim


def general_angles_trajectory_update(angle_trajectories, components, leading_eigvecs, eigenspace_dims,
									 calc_spaces_angles):
	# Update angles based on whether we are calculating eigenspaces or not
	if calc_spaces_angles:
		spaces_angles_trajectory_update(angle_trajectories, components, leading_eigvecs, eigenspace_dims)
	else:
		angles_trajectory_update(angle_trajectories, components, leading_eigvecs)


def error_calculation(components, validation_set):
	projections = components.T @ validation_set
	reconstructed_samples = components @ projections
	errors = np.linalg.norm(validation_set - reconstructed_samples, axis=0)
	gen_error = 0.5 * np.mean(errors ** 2)  # Mean squared error
	return gen_error


def errors_trajectory_update(components, generalization_errors, validation_set):
	# Compute generalization error on the fixed validation set
	gen_error = error_calculation(components, validation_set)
	generalization_errors.append(gen_error)
