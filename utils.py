import numpy as np


def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    if v1 is None or v2 is None:
        return 0
    dot_prod = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    cos_theta = dot_prod / (magnitude_v1 * magnitude_v2)
    return np.arccos(cos_theta)


def angle_between_vectors_signed(v1: np.ndarray, v2: np.ndarray) -> float:
    if v1 is None or v2 is None:
        return 0
    # Dot product of two vectors
    dot_prod = np.dot(v1, v2)
    # Determinant (pseudo cross-product) in 2D
    det = v1[0] * v2[1] - v1[1] * v2[0]
    # Angle in radians
    angle_radians = np.arctan2(det, dot_prod)
    return angle_radians


def coordinates_to_vector(point1: float, point2: float) -> np.ndarray[int, int]:
    point1_int = np.array([int(point1[0]), int(point1[1])])
    point2_int = np.array([int(point2[0]), int(point2[1])])
    return point2_int - point1_int


def is_coordinates_close(point1: np.ndarray, point2: np.ndarray, threshold: float) -> bool:
    return np.linalg.norm(point1 - point2) < threshold
