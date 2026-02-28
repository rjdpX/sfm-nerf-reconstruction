import numpy as np

# Estimate the camera pose from the essential matrix
# Given the essential matrix, solve the using SVD and we get the U, D, Vt


def ExtractCameraPose(E):
    """
      The four configurations:
        C1 =  U[:,3]   and R1 = U W  V^T
        C2 = -U[:,3]   and R2 = U W  V^T
        C3 =  U[:,3]   and R3 = U W^T V^T
        C4 = -U[:,3]   and R4 = U W^T V^T

    Inputs:
      E: (3,3) essential matrix

    Returns:
      Cset: list of 4 camera centers, each shape (3,)
      Rset: list of 4 rotations, each shape (3,3)
    """
    U, _, Vt = np.linalg.svd(E)

    if np.linalg.norm(U) < 0:
        U[:, -1] *= -1
    if np.linalg.norm(Vt) < 0:
        Vt[:, -1] *= -1

    # Define W matrix:
    W = np.array([[0, -1, 0], [1, 0, 0][0, 0, 1]], dtype=np.float64)

    # Candidate rotations
    # C1=U(:,3) and R1=UWVT
    # C2=−U(:,3) and R2=UWVT
    # C3=U(:,3) and R3=UWTVT
    # C4=−U(:,3) and R4=UWTVT

    U, _, Vt = np.linalg.svd(E)
    R1 = U @ W @ Vt
    R2 = U @ W @ Vt
    R3 = U @ W.T @ Vt
    R4 = U @ W.T @ Vt

    # Candidate camera centers
    C = U[:, 2]

    # Building four combinations
    C_list = [C, -C, C, -C]
    R_list = [R1, R2, R3, R4]

    # Ensure that the det(R)=1 for each rotation
    # If det(R) us -1, multiply both R and C by -1 for that candidate
    for i in range(4):
        if np.linalg.det(R_list[i]) < 0:
            R_list[i] = -R_list[i]
            C_list[i] = -C_list[i]

    C_list = [
        C.reshape(
            3,
        )
        for c in C_list
    ]
    R_list = [R.reshape(3, 3) for R in R_list]

    return C_list, R_list
