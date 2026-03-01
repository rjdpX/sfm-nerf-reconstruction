import numpy as np
from scipy.optimize import least_squares
from LinearTriangulation import camera_projection_matrix


def project(P: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Project a 3D point onto the image plane using a projection matrix.

    Args:
        P (np.ndarray): Projection matrix (shape: (3, 4)).
        X (np.ndarray): 3D point in world coordinates (shape: (3,)).

    Returns:
        np.ndarray: Projected 2D point in image plane (shape: (2,)).
    """
    X_h = np.append(X, 1.0)
    x_h = P @ X_h

    return x_h[:2] / x_h[2]


def residuals(X, P1, P2, x1, x2):
    """Calculate the reprojection residuals for nonlinear triangulation.

    Computes the difference between observed image points and their projections
    using the estimated 3D point. This is used for optimization in nonlinear
    triangulation to minimize the reprojection error across both cameras.

        X (np.ndarray): 3D point in world coordinates (shape: (3,) or (4,)).
        P1 (np.ndarray): Projection matrix for camera 1 (shape: (3, 4)).
        P2 (np.ndarray): Projection matrix for camera 2 (shape: (3, 4)).
        x1 (np.ndarray): Observed 2D point in camera 1 image plane (shape: (2,)).
        x2 (np.ndarray): Observed 2D point in camera 2 image plane (shape: (2,)).

    Returns:
        np.ndarray: Vector of residuals with shape (4,) containing the differences
                   [u1_error, v1_error, u2_error, v2_error] where u and v are
                   image coordinates in cameras 1 and 2 respectively.
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
    """Refine 3D points using nonlinear least squares optimization.

    Args:
        K (np.ndarray): Camera intrinsic matrix (3x3)
        C1 (np.ndarray): Camera 1 center (3,)
        R1 (np.ndarray): Camera 1 rotation matrix (3x3)
        C2 (np.ndarray): Camera 2 center (3,)
        R2 (np.ndarray): Camera 2 rotation matrix (3x3)
        x1 (np.ndarray): Image points in camera 1 (Nx2)
        x2 (np.ndarray): Image points in camera 2 (Nx2)
        X_init (np.ndarray): Initial 3D point estimates (Nx3)

    Returns:
        np.ndarray: Refined 3D points (Nx3)
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
