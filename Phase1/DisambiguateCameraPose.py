import numpy as np

# Given four camera pose configurations and their triangulated points, find the unique
# camera pose by checking the cheirality condition - the reconstructed points must be
# infront of the camera

from LinearTriangulation import LinearTriangulation


def is_in_front(C: np.ndarray, R: np.ndarray, X: np.ndarray) -> np.ndarray:

    r3 = R[2, :]  # third row = camera forward axis (3,)
    return (X - C) @ r3 > 0  # (N,) dot product per point


def DisambiguateCameraPose(
    K: np.ndarray, C_list: list, R_list: list, x1: np.ndarray, x2: np.ndarray
):
    C_1 = np.zeros(3)
    R_1 = np.eye(3)

    best_i = -1
    best_count = -1
    best_X = None

    for i in range(4):
        # Triangulate using this candidate pose
        X = LinearTriangulation(K, C_1, R_1, C_list[i], R_list[i], x1, x2)  # (N, 3)

        # Count points in front of BOTH cameras
        in_front = is_in_front(C_1, R_1, X) & is_in_front(C_list[i], R_list[i], X)
        count = int(np.sum(in_front))

        if count > best_count:
            best_count = count
            best_i = i
            best_x = X

        return C_list[best_i], R_list[best_i], best_X
