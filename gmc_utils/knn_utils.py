import faiss


def update_index_with_new_data(index_gpu, new_data):
    new_data_cpu = new_data.detach().cpu().numpy()
    index_gpu.reset()  # Clear the index
    index_gpu.add(new_data_cpu)  # Add new data to the index


def setup_faiss(dim_base, dim_noise):
    # Create the indices on CPU
    index_base = faiss.IndexFlatL2(dim_base)
    index_noise = faiss.IndexFlatL2(dim_noise)
    index_for_gt_motion = faiss.IndexFlatL2(3)

    # Move the indices to GPU using the same StandardGpuResources instance
    res = faiss.StandardGpuResources()

    index_gpu = faiss.index_cpu_to_gpu(res, 0, index_base)
    index_gpu_noise = faiss.index_cpu_to_gpu(res, 0, index_noise)
    index_gpu_for_gt_motion = faiss.index_cpu_to_gpu(res, 0, index_for_gt_motion)

    return index_gpu, index_gpu_noise, index_gpu_for_gt_motion