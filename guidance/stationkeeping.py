import numpy as np

def stationkeeping_reference(x_ref: float, y_ref: float, psi_ref: float):
    """
    Returns:
      eta_d   (3,1)
      eta_ds  (3,1)  = 0
      eta_ds2 (3,1)  = 0
    """
    eta_d = np.array([x_ref, y_ref, psi_ref], dtype=float).reshape(3, 1)
    eta_ds = np.zeros((3, 1), dtype=float)
    eta_ds2 = np.zeros((3, 1), dtype=float)
    return eta_d, eta_ds, eta_ds2