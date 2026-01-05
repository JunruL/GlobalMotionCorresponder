import torch


def normalize_quaternion(q):
    norm = torch.sqrt(torch.sum(q * q, dim=1, keepdim=True))
    return q / norm


def quaternions_to_rotation_matrices(quaternions):
    """
    Convert a batch of quaternions to a batch of 3x3 rotation matrices.

    Parameters:
    - quaternions: A torch tensor of shape (N, 4).

    Returns:
    - A torch tensor of shape (N, 3, 3).
    """
    # Normalize quaternions
    quaternions = normalize_quaternion(quaternions)

    w, x, y, z = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]

    # Precompute products
    ww, wx, wy, wz = w*w, w*x, w*y, w*z
    xx, xy, xz = x*x, x*y, x*z
    yy, yz = y*y, y*z
    zz = z*z

    # Construct rotation matrices
    rotation_matrices = torch.empty((quaternions.size(0), 3, 3), device=quaternions.device)

    rotation_matrices[:, 0, 0] = 1 - 2 * (yy + zz)
    rotation_matrices[:, 0, 1] = 2 * (xy - wz)
    rotation_matrices[:, 0, 2] = 2 * (xz + wy)
    
    rotation_matrices[:, 1, 0] = 2 * (xy + wz)
    rotation_matrices[:, 1, 1] = 1 - 2 * (xx + zz)
    rotation_matrices[:, 1, 2] = 2 * (yz - wx)
    
    rotation_matrices[:, 2, 0] = 2 * (xz - wy)
    rotation_matrices[:, 2, 1] = 2 * (yz + wx)
    rotation_matrices[:, 2, 2] = 1 - 2 * (xx + yy)

    return rotation_matrices


def rotation_matrices_to_quaternions(rotation_matrices):
    """
    Convert a batch of 3x3 rotation matrices to quaternions.

    Parameters:
    - rotation_matrices: A torch tensor of shape (N, 3, 3).

    Returns:
    - A torch tensor of shape (N, 4) representing the quaternions [w, x, y, z].
    """
    R = rotation_matrices
    N = R.shape[0]

    quaternions = torch.empty((N, 4), device=R.device)

    # Compute the trace of each rotation matrix
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    # Define masks for different cases based on the trace and diagonal elements
    mask_positive_trace = trace > 0
    mask_diag_1 = (R[:, 1, 1] > R[:, 0, 0]) & (R[:, 1, 1] > R[:, 2, 2])
    mask_diag_2 = (R[:, 2, 2] > R[:, 0, 0]) & (R[:, 2, 2] > R[:, 1, 1])
    mask_diag_3 = (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])

    # Case 1: Positive trace
    s_positive = torch.sqrt(trace[mask_positive_trace] + 1.0) * 2.0
    quaternions[mask_positive_trace, 0] = 0.25 * s_positive  # w
    quaternions[mask_positive_trace, 1] = (R[mask_positive_trace, 2, 1] - R[mask_positive_trace, 1, 2]) / s_positive  # x
    quaternions[mask_positive_trace, 2] = (R[mask_positive_trace, 0, 2] - R[mask_positive_trace, 2, 0]) / s_positive  # y
    quaternions[mask_positive_trace, 3] = (R[mask_positive_trace, 1, 0] - R[mask_positive_trace, 0, 1]) / s_positive  # z

    # Case 2: R[1,1] is the largest diagonal term
    s_diag_1 = torch.sqrt(1.0 + R[mask_diag_1, 1, 1] - R[mask_diag_1, 0, 0] - R[mask_diag_1, 2, 2] + 1e-6) * 2.0
    quaternions[mask_diag_1, 0] = (R[mask_diag_1, 0, 2] - R[mask_diag_1, 2, 0]) / s_diag_1
    quaternions[mask_diag_1, 1] = (R[mask_diag_1, 0, 1] + R[mask_diag_1, 1, 0]) / s_diag_1
    quaternions[mask_diag_1, 2] = 0.25 * s_diag_1
    quaternions[mask_diag_1, 3] = (R[mask_diag_1, 1, 2] + R[mask_diag_1, 2, 1]) / s_diag_1

    # Case 3: R[2,2] is the largest diagonal term
    s_diag_2 = torch.sqrt(1.0 + R[mask_diag_2, 2, 2] - R[mask_diag_2, 0, 0] - R[mask_diag_2, 1, 1] + 1e-6) * 2.0
    quaternions[mask_diag_2, 0] = (R[mask_diag_2, 1, 0] - R[mask_diag_2, 0, 1]) / s_diag_2
    quaternions[mask_diag_2, 1] = (R[mask_diag_2, 0, 2] + R[mask_diag_2, 2, 0]) / s_diag_2
    quaternions[mask_diag_2, 2] = (R[mask_diag_2, 1, 2] + R[mask_diag_2, 2, 1]) / s_diag_2
    quaternions[mask_diag_2, 3] = 0.25 * s_diag_2

    # Case 4: R[0,0] is the largest diagonal term
    s_diag_3 = torch.sqrt(1.0 + R[mask_diag_3, 0, 0] - R[mask_diag_3, 1, 1] - R[mask_diag_3, 2, 2] + 1e-6) * 2.0
    quaternions[mask_diag_3, 0] = (R[mask_diag_3, 2, 1] - R[mask_diag_3, 1, 2]) / s_diag_3
    quaternions[mask_diag_3, 1] = 0.25 * s_diag_3
    quaternions[mask_diag_3, 2] = (R[mask_diag_3, 0, 1] + R[mask_diag_3, 1, 0]) / s_diag_3
    quaternions[mask_diag_3, 3] = (R[mask_diag_3, 0, 2] + R[mask_diag_3, 2, 0]) / s_diag_3

    return quaternions


def slerp(q0, q1, t):
    """
    Perform Spherical Linear Interpolation (SLERP) between two quaternions q0 and q1.

    Parameters:
    - q0: A torch tensor of shape (N, 4) representing the initial quaternions.
    - q1: A torch tensor of shape (N, 4) representing the target quaternions.
    - t: A float or a torch tensor of shape (N,) representing the interpolation factor [0, 1].

    Returns:
    - A torch tensor of shape (N, 4) representing the interpolated quaternions.
    """
    dot_product = torch.sum(q0 * q1, dim=1, keepdim=True)
    
    # Ensure dot_product is within range [-1.0, 1.0]
    dot_product = torch.clamp(dot_product, -1.0, 1.0)

    # Use the short path for interpolation by negating q1 if necessary
    q1 = torch.where(dot_product < 0.0, -q1, q1)
    dot_product = torch.where(dot_product < 0.0, -dot_product, dot_product)

    # Compute angles
    theta_0 = torch.acos(dot_product)
    sin_theta_0 = torch.sin(theta_0)

    # Handle edge case of very small sin_theta_0
    very_small_threshold = 1e-6
    scale_q0 = torch.where(sin_theta_0 < very_small_threshold, 1.0 - t, torch.sin((1.0 - t) * theta_0) / sin_theta_0)
    scale_q1 = torch.where(sin_theta_0 < very_small_threshold, t, torch.sin(t * theta_0) / sin_theta_0)

    # Compute interpolated quaternion
    interpolated_quaternion = scale_q0 * q0 + scale_q1 * q1
    return normalize_quaternion(interpolated_quaternion)


def interpolate_rotations(rotation_matrices, t):
    """
    Interpolate a batch of rotation matrices between identity and given rotations.

    Parameters:
    - rotation_matrices: A torch tensor of shape (N, 3, 3).
    - t: A float or a torch tensor of shape (N,) representing the interpolation factor [0, 1].

    Returns:
    - A torch tensor of shape (N, 3, 3) representing the interpolated rotation matrices.
    """
    N = rotation_matrices.size(0)

    # Identity quaternion [1, 0, 0, 0]
    identity_quaternion = torch.tensor([1, 0, 0, 0], device=rotation_matrices.device).expand(N, 4)

    # Convert rotation matrices to quaternions
    target_quaternions = rotation_matrices_to_quaternions(rotation_matrices)

    # Perform SLERP between identity and target quaternions
    interpolated_quaternions = slerp(identity_quaternion, target_quaternions, t)

    # Convert interpolated quaternions back to rotation matrices
    interpolated_rotation_matrices = quaternions_to_rotation_matrices(interpolated_quaternions)

    return interpolated_rotation_matrices
