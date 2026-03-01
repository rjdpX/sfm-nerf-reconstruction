import numpy as np
from scipy.optimize import least_squares
from LinearTriangulation import camera_projection_matrix


def project(P: np.ndarray, X: np.ndarray) -> np.ndarray:
    """_summary_

    Args:
        P (np.ndarray): _description_
        X (np.ndarray): _description_

    Returns:
        np.ndarray: _description_
    """
    X_h = np.append(X, 1.0)
    x_h = P @ X_h

    return x_h[:2] / x_h[2]


def residuals(X, P1, P2, x1, x2):
    """_summary_

    Args:
        X (_type_): _description_
        P1 (_type_): _description_
        P2 (_type_): _description_
        x1 (_type_): _description_
        x2 (_type_): _description_
    """
    x1_hat = project(P1, X)
    x2_hat = project(P2, X)

    return np.array(
        [
            x1[0] - x1_hat[0],  # u vector in camera 1
            x1[1] - x1_hat[1],  # v vector in camera 1
            x2[0] - x2_hat[0],  # u vector in camera 2
            x2[1] - x2_hat[1],  # v vector in camera 2
        ]
    )


def NonLinearTriangulation(
    K: np.ndarray,
    C1: np.ndarray,
    R1: np.ndarray,
    C2: np.ndarray,
    R2: np.ndarray,
    x1: np.ndarray,
    x2: np.ndarray,
    X_init: np.ndarray,
) -> np.ndarray:
    """_summary_

    Args:
        K (np.ndarray): _description_
        C1 (np.ndarray): _description_
        R1 (np.ndarray): _description_
        C2 (np.ndarray): _description_
        R2 (np.ndarray): _description_
        x1 (np.ndarray): _description_
        x2 (np.ndarray): _description_
        X_init (np.ndarray): _description_

    Returns:
        np.ndarray: _description_
    """
    assert x1.shape == x2.shape
    assert X_init.shape[0] == x1.shape[0]

    P1 = camera_projection_matrix(K, C1, R1)
    P2 = camera_projection_matrix(K, C2, R2)

    N = x1.shape[0]
    X_refined = np.zeros((N, 3))

    for i in range(N):
        result = least_squares(
            residuals, X_init[i], args=(P1, P2, x1[i], x2[i]), method="trf", max_nfev=50
        )
        X_refined[i] = result.x

    return X_refined
