#  The learning rate schemas for online PCA I eventually used in my assignments.
def lr_inverse_decay_no_floor(t, initial=0.01025, k=0.085):
	"""Inverse decay without floor"""
	return initial / (1 + k * t)


def lr_inverse_decay_no_floor2(t, initial=0.1, k=0.5):
	"""Inverse decay without floor"""
	if k <= 250:
		return initial
	return initial / (1 + k * t)


def schedule_inverse_smooth(step, eta0=0.005, k=0.001, eta_min=0.00001):
	eta = eta0 / (1 + k * step)
	return max(eta, eta_min)
