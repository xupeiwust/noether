#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import torch_geometric  # type: ignore[import-untyped]
import torch_scatter  # type: ignore[import-untyped]


def radius(x, y, r, max_num_neighbors, batch_x, batch_y):
    # Move tensors to CPU if on MPS device
    device = x.device
    if device.type == "mps":
        x = x.cpu()
        y = y.cpu()
        batch_x = batch_x.cpu() if batch_x is not None else None
        batch_y = batch_y.cpu() if batch_y is not None else None

    result = torch_geometric.nn.pool.radius(
        x,
        y,
        r,
        batch_x,
        batch_y,
        max_num_neighbors=max_num_neighbors,
    )

    # Move result back to MPS if original tensors were on MPS
    if device.type == "mps":
        result = result.to(device)

    return result


def knn(x, y, k, batch_x=None, batch_y=None):
    # Move tensors to CPU if on MPS device
    device = x.device
    if device.type == "mps":
        x = x.cpu()
        y = y.cpu()
        batch_x = batch_x.cpu() if batch_x is not None else None
        batch_y = batch_y.cpu() if batch_y is not None else None

    result = torch_geometric.nn.pool.knn(
        x=x,
        y=y,
        k=k,
        batch_x=batch_x,
        batch_y=batch_y,
    )

    # Move result back to MPS if original tensors were on MPS
    if device.type == "mps":
        result = result.to(device)

    return result


def segment_csr(src, indptr, reduce):
    # Move tensors to CPU if on MPS device
    device = src.device
    if device.type == "mps":
        src = src.cpu()
        indptr = indptr.cpu()

    result = torch_scatter.segment_csr(
        src=src,
        indptr=indptr,
        reduce=reduce,
    )

    # Move result back to MPS if original tensors were on MPS
    if device.type == "mps":
        result = result.to(device)

    return result
