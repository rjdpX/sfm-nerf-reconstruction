import numpy as np
from EstimateFundamentalMatrix import make_homogeneous


def triangulate_point(P1, P2, x1h, x2h) -> np.ndarray:
    """Triangulate a 3D point from two camera views using linear triangulation.

    Solves for the 3D homogeneous coordinates of a point given its projections
    in two camera views using the Direct Linear Transform (DLT) method via SVD.

    Args:
        P1 (np.ndarray): 3x4 camera projection matrix for the first view.
        P2 (np.ndarray): 3x4 camera projection matrix for the second view.
        x1h (np.ndarray): 2D homogeneous coordinates [u1, v1] of the point in the first image.
        x2h (np.ndarray): 2D homogeneous coordinates [u2, v2] of the point in the second image.

    Returns:
        np.ndarray: 3D Cartesian coordinates [X, Y, Z] of the triangulated point.
    """

    u1, v1 = x1h[0], x1h[1]
    u2, v2 = x2h[0], x2h[1]

    A = np.array(
        [
            u1 * P1[2, :] - P1[0, :],  # constraint from x1
            v1 * P1[2, :] - P1[0, :],  # constraint from y1
            u2 * P2[2, :] - P2[0, :],  # constraint from x2
            v2 * P2[2, :] - P2[0, :],  # constraint from y2
        ],
        dtype=np.float64,
    )

    _, _, Vt = np.linalg.svd(A)
    X_h = Vt[:-1]
    return X_h[:3] / X_h[3]


def camera_projection_matrix(K: np.ndarray, C: np.ndarray, R: np.ndarray) -> np.array:
    """Compute the camera projection matrix from intrinsics, rotation, and center.

    Args:
        K (np.ndarray): Camera intrinsic matrix (3x3)
        C (np.ndarray): Camera center position (3x1)
        R (np.ndarray): Camera rotation matrix (3x3)

    Returns:
        np.array: Camera projection matrix (3x4)

    Equation: P = K R [I | -C]
    """

    t = -R @ C.reshape(3, 1)
    return K @ np.hstack([R, t])


# Triangulation check for the chierality condition
def LinearTriangulation(K, C1, R1, C2, R2, x1, x2) -> np.ndarray:
    """Triangulate a 3D point from two camera views using linear triangulation.

    Args:
        K (np.ndarray): Camera intrinsic matrix (3x3)
        C1 (np.ndarray): Camera 1 center position (3x1)
        R1 (np.ndarray): Camera 1 rotation matrix (3x3)
        C2 (np.ndarray): Camera 2 center position (3x1)
        R2 (np.ndarray): Camera 2 rotation matrix (3x3)
        x1 (np.ndarray): Normalized image point in camera 1 (2x1)
        x2 (np.ndarray): Normalized image point in camera 2 (2x1)

    Returns:
        np.ndarray: Triangulated 3D point in homogeneous coordinates (4x1)
    """
    assert x1.shape == x2.shape

    P1 = camera_projection_matrix(K, C1, R1)
    P2 = camera_projection_matrix(K, C2, R2)

    x1h = make_homogeneous(x1)
    x2h = make_homogeneous(x2)

    N = x1.shape[0]
    X = np.zeros((N, 3))

    for i in range(N):
        X[i] = triangulate_point(P1, P2, x1h[i], x2h[i])

    return X
