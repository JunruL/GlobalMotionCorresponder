import torch
import faiss

def _handle_pointcloud_input(x, x_lengths=None, x_normals=None):
    """
    Ensure that x is a tensor of shape (N, P, D) and x_lengths is properly set.

    Args:
        x: Tensor of shape (N, P, D)
        x_lengths: Optional tensor of shape (N,)
    Returns:
        x: Tensor of shape (N, P, D)
        x_lengths: Tensor of shape (N,)
        x_normals: None (not used in this context)
    """
    if isinstance(x, torch.Tensor):
        N, P, D = x.shape
        if x_lengths is None:
            x_lengths = torch.full((N,), P, dtype=torch.long, device=x.device)
        return x, x_lengths, x_normals
    else:
        raise ValueError("Input x must be a tensor of shape (N, P, D).")

def knn_faiss(p1, p2, lengths1=None, lengths2=None, K=1, use_gpu=False):
    """
    Compute the k-nearest neighbors for each point in p1 from p2 using Faiss.

    Args:
        p1: Tensor of shape (N, P1, D)
        p2: Tensor of shape (N, P2, D)
        K: Number of nearest neighbors to find
        use_gpu: Boolean indicating whether to use GPU acceleration.
    Returns:
        A dictionary with 'dists' and 'idx' tensors.
    """
    N, P1, D = p1.shape
    _, P2, _ = p2.shape

    dists = []
    indices = []

    for i in range(N):
        x = p1[i].detach().cpu().numpy().astype('float32')
        y = p2[i].detach().cpu().numpy().astype('float32')

        if use_gpu:
            res = faiss.StandardGpuResources()
            index = faiss.GpuIndexFlatL2(res, D)
        else:
            index = faiss.IndexFlatL2(D)

        index.add(y)  # Add database vectors
        distances, idx = index.search(x, K)  # Search x in y

        dists.append(torch.from_numpy(distances).to(p1.device))
        indices.append(torch.from_numpy(idx).to(p1.device))

        index.reset()
        if use_gpu:
            del res

    dists = torch.stack(dists)  # Shape: (N, P1, K)
    idx = torch.stack(indices)   # Shape: (N, P1, K)

    return {'dists': dists, 'idx': idx}


def knn_gather(p2, idx):
    """
    Gather the k-nearest neighbor points from p2 using indices idx.

    Args:
        p2: Tensor of shape (N, P2, D)
        idx: Tensor of indices with shape (N, P1, K)

    Returns:
        gathered: Tensor of shape (N, P1, K, D)
    """
    N, P2, D = p2.shape
    N_idx, P1, K = idx.shape
    assert N == N_idx, "Batch sizes of p2 and idx must match."

    gathered = []

    for i in range(N):
        p2_i = p2[i]  # (P2, D)
        idx_i = idx[i]  # (P1, K)
        gathered_i = p2_i[idx_i.view(-1), :].view(P1, K, D)
        gathered.append(gathered_i)

    gathered = torch.stack(gathered)  # (N, P1, K, D)
    return gathered


def SPED_dist(x_coor_near, y_coor_near, neighbors):
    # Compute distances between central point and neighbors
    x_vectors = x_coor_near[:, :, :neighbors] - x_coor_near[:, :, 0:1, :]
    x_distances = torch.norm(x_vectors, dim=3)
    y_vectors = y_coor_near[:, :, :neighbors] - y_coor_near[:, :, 0:1, :]
    y_distances = torch.norm(y_vectors, dim=3)

    x_mean = x_distances.mean(dim=2)
    y_mean = y_distances.mean(dim=2)

    mean_diff = (x_mean - y_mean).abs()
    dis_energy = mean_diff.sum(dim=1)  # Sum over points

    return dis_energy


def feeature_pooling_1(x, x_lengths, y, y_lengths, neighbors, y_at_x, use_gpu=False):
    N, P1, D = x.shape

    if P1 < 15 or y.shape[1] < 15:
        raise ValueError("x or y does not have enough points (at least 15 points).")

    # k-NN search within x and from x to y_at_x
    x_nn = knn_faiss(x, x, K=neighbors, use_gpu=use_gpu)
    y_correspondence_nn = knn_faiss(x, y_at_x, K=1, use_gpu=use_gpu)

    y_correspondence = knn_gather(y, y_correspondence_nn['idx']).squeeze(-2)  # Shape: (N, P1, D)
    y_nn = knn_faiss(y_correspondence, y, K=neighbors, use_gpu=use_gpu)

    # Gather neighbor coordinates
    x_coor_near = knn_gather(x, x_nn['idx'])  # Shape: (N, P1, K, D)
    y_coor_near = knn_gather(y, y_nn['idx'])  # Shape: (N, P1, K, D)

    # Use the modified SPED_invariant function
    SPED1 = SPED_dist(x_coor_near, y_coor_near, neighbors)
    SPED2 = SPED_dist(x_coor_near, y_coor_near, neighbors // 2)
    SPED3 = SPED_dist(x_coor_near, y_coor_near, max(1, neighbors // 10))

    # Aggregate the SPED scores
    MPED_SCORE = (SPED1.sum() + SPED2.sum() + SPED3.sum()) / (P1 * N)

    return MPED_SCORE

def feeature_pooling_2(x, x_lengths, y, y_lengths, neighbors, x_at_y, use_gpu=False):
    N, P1, D = x.shape

    if P1 < 15 or y.shape[1] < 15:
        raise ValueError("x or y does not have enough points (at least 15 points).")

    # k-NN search within x and from x_at_y to y
    x_nn = knn_faiss(x, x, K=neighbors, use_gpu=use_gpu)
    y_correspondence_nn = knn_faiss(x_at_y, y, K=1, use_gpu=use_gpu)

    y_correspondence = knn_gather(y, y_correspondence_nn['idx']).squeeze(-2)  # Shape: (N, P1, D)
    y_nn = knn_faiss(y_correspondence, y, K=neighbors, use_gpu=use_gpu)

    # Gather neighbor coordinates
    x_coor_near = knn_gather(x, x_nn['idx'])  # Shape: (N, P1, K, D)
    y_coor_near = knn_gather(y, y_nn['idx'])  # Shape: (N, P1, K, D)

    # Use the modified SPED_invariant function
    SPED1 = SPED_dist(x_coor_near, y_coor_near, neighbors)
    SPED2 = SPED_dist(x_coor_near, y_coor_near, neighbors // 2)
    SPED3 = SPED_dist(x_coor_near, y_coor_near, max(1, neighbors // 10))

    # Aggregate the SPED scores
    MPED_SCORE = (SPED1.sum() + SPED2.sum() + SPED3.sum()) / (P1 * N)

    return MPED_SCORE

def MPED_VALUE(x, y, y_at_x, neighbors=10, use_gpu=True):
    x, x_lengths, _ = _handle_pointcloud_input(x)
    y, y_lengths, _ = _handle_pointcloud_input(y)

    MPED1 = feeature_pooling_1(x, x_lengths, y, y_lengths, neighbors, y_at_x, use_gpu=use_gpu)
    MPED2 = feeature_pooling_2(y, y_lengths, x, x_lengths, neighbors, y_at_x, use_gpu=use_gpu)

    MPED_SCORE = MPED1 + MPED2

    return MPED_SCORE

