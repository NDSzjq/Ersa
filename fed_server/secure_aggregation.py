from __future__ import annotations

import secrets
import time
from collections import OrderedDict

import crypten
import torch

from defense_detect.clustering import get_benign_set


MASK_BLOCK_SIZE = 16
FULD_SKETCH_STRIDE = 512
FULD_EXACT_MAX_ELEMENTS = 4096
# One-round LoRA updates can be below the 24-bit fixed-point quantum after
# BF-SVD.  Both factors use this public scale before sharing; the final SVD
# factors are divided by it, so the represented update is unchanged.
LORA_NUMERICAL_SCALE = 16.0

def parameter_pairs(parameters):
    pairs = OrderedDict()
    for b_name in sorted(name for name in parameters if name.endswith("lora_B")):
        a_name = b_name[:-len("lora_B")] + "lora_A"
        if a_name not in parameters:
            raise ValueError(f"Missing LoRA A factor for {b_name}")
        B, A = parameters[b_name], parameters[a_name]
        if B.ndim != 2 or A.ndim != 2 or B.shape[1] != A.shape[0]:
            raise ValueError(f"Invalid LoRA shapes for {b_name}: {B.shape}, {A.shape}")
        pairs[b_name] = a_name
    if not pairs:
        raise ValueError("No LoRA parameter pairs found")
    return pairs


def _share(value, source, party):
    return crypten.cryptensor(value if party == source else torch.zeros_like(value), src=source)


def _stats(communicator):
    raw = communicator.get_communication_stats() or {}
    return {"rounds": int(raw.get("rounds", 0)), "bytes": float(raw.get("bytes", 0)), "time": float(raw.get("time", 0.0))}



def _private_generator():
    generator = torch.Generator(device="cpu")
    generator.manual_seed(secrets.randbits(63))
    return generator


def _orthogonal_blocks(size, party, block_size=MASK_BLOCK_SIZE):
    """CS0 secret-shares a private block-diagonal orthogonal matrix."""
    blocks = []
    generator = _private_generator() if party == 0 else None
    for start in range(0, size, block_size):
        width = min(block_size, size - start)
        if party == 0:
            block, _ = torch.linalg.qr(torch.randn(width, width, generator=generator))
        else:
            block = torch.zeros(width, width)
        blocks.append(crypten.cryptensor(block, src=0))
    return blocks


def _left_mask(matrix, blocks, transpose=False):
    pieces = []
    offset = 0
    for block in blocks:
        width = block.size(0)
        mask = block.transpose(0, 1) if transpose else block
        pieces.append(mask.matmul(matrix[offset:offset + width, :]))
        offset += width
    return crypten.cat(pieces, dim=0)


def _right_mask(matrix, blocks, transpose=False):
    return _left_mask(matrix.transpose(0, 1), blocks, transpose=transpose).transpose(0, 1)


def _secure_masked_svd(secret_matrix, target_rank, party):
    """Block-orthogonally mask, open only the masked input to CS1, then unmask."""
    rows, cols = secret_matrix.size()
    reduced = min(rows, cols)
    dtype = torch.float32
    left = _orthogonal_blocks(rows, party)
    right = _orthogonal_blocks(cols, party)
    masked = _right_mask(_left_mask(secret_matrix, left), right)
    masked_plain = masked.get_plain_text(dst=1)

    if party == 1:
        masked_plain = masked_plain.cpu()
        u_masked, singular_values, vh_masked = torch.linalg.svd(masked_plain, full_matrices=False)
        rank = min(target_rank, singular_values.numel())
        root = torch.diag(torch.sqrt(torch.clamp(singular_values[:rank], min=0)))
        b_masked = u_masked[:, :rank] @ root
        a_masked = root @ vh_masked[:rank, :]
        reconstruction_error = float((b_masked @ a_masked - masked_plain).norm()) / max(float(masked_plain.norm()), 1e-12)
    else:
        rank = min(target_rank, reduced)
        b_masked = torch.zeros(rows, rank, dtype=dtype)
        a_masked = torch.zeros(rank, cols, dtype=dtype)
        reconstruction_error = torch.zeros(1, dtype=dtype)

    b_share = crypten.cryptensor(b_masked, src=1)
    a_share = crypten.cryptensor(a_masked, src=1)
    error_share = crypten.cryptensor(torch.as_tensor(reconstruction_error).reshape(1), src=1)
    b_factor = _left_mask(b_share, left, transpose=True)
    a_factor = _right_mask(a_share, right, transpose=True)
    error = error_share.get_plain_text(dst=0)
    return b_factor, a_factor, float(error.item()) if party == 0 else 0.0

def _bfsvd_module(local_params, local_client_id, num_clients, b_name, a_name, module_index, seed, party,
                  numerical_scale=1.0):
    """Algorithm 2: masked federated SVD of concatenated LoRA factors."""
    B, A = local_params[b_name], local_params[a_name]
    rank = B.shape[1]
    rows, cols = B.shape[0], A.shape[1]
    compact = num_clients * rank
    p_b = _orthogonal_blocks(rows, party)
    q_b = _orthogonal_blocks(compact, party)
    p_a = _orthogonal_blocks(compact, party)
    q_a = _orthogonal_blocks(cols, party)
    B_bar = None
    A_bar = None
    for source in range(num_clients):
        B_pad = torch.zeros(rows, num_clients * rank)
        A_pad = torch.zeros(num_clients * rank, cols)
        if source == local_client_id:
            B_pad[:, source * rank:(source + 1) * rank] = B * numerical_scale
            A_pad[source * rank:(source + 1) * rank, :] = A * numerical_scale
        B_part = crypten.cryptensor(B_pad, src=source)
        A_part = crypten.cryptensor(A_pad, src=source)
        B_bar = B_part if B_bar is None else B_bar + B_part
        A_bar = A_part if A_bar is None else A_bar + A_part

    B_bar = _right_mask(_left_mask(B_bar, p_b), q_b)
    A_bar = _right_mask(_left_mask(A_bar, p_a), q_a)

    masked_b = B_bar.get_plain_text(dst=1)
    masked_a = A_bar.get_plain_text(dst=1)
    if party == 1:
        U_b, S_b, Vh_b = torch.linalg.svd(masked_b, full_matrices=False)
        U_a, S_a, Vh_a = torch.linalg.svd(masked_a, full_matrices=False)
    else:
        U_b = torch.zeros(rows, min(rows, num_clients * rank))
        S_b = torch.zeros(min(rows, num_clients * rank))
        Vh_b = torch.zeros(min(rows, num_clients * rank), num_clients * rank)
        U_a = torch.zeros(num_clients * rank, min(num_clients * rank, cols))
        S_a = torch.zeros(min(num_clients * rank, cols))
        Vh_a = torch.zeros(min(num_clients * rank, cols), cols)
    U_b = crypten.cryptensor(U_b, src=1)
    S_b = crypten.cryptensor(S_b, src=1)
    Vh_b = crypten.cryptensor(Vh_b, src=1)
    U_a = crypten.cryptensor(U_a, src=1)
    S_a = crypten.cryptensor(S_a, src=1)
    Vh_a = crypten.cryptensor(Vh_a, src=1)
    U_b = _left_mask(U_b, p_b, transpose=True)
    V_b = _right_mask(Vh_b, q_b, transpose=True)
    U_a = _left_mask(U_a, p_a, transpose=True)
    V_a = _right_mask(Vh_a, q_a, transpose=True)
    return {"U_B": U_b, "S_B": S_b, "V_B_blocks": [V_b[:, i * rank:(i + 1) * rank] for i in range(num_clients)], "U_A_blocks": [U_a[i * rank:(i + 1) * rank, :] for i in range(num_clients)], "S_A": S_a, "V_A": V_a, "rank": rank}

def _reveal_public_scalar(secret, dst=0):
    value = secret.get_plain_text(dst=dst)

    if value is None:
        public_value = torch.zeros(1, dtype=torch.float64)
    else:
        public_value = torch.as_tensor(
            value, dtype=torch.float64
        ).detach().cpu().reshape(1)

    crypten.communicator.get().broadcast(public_value, dst)
    return float(public_value.item())


def _secure_mean(values, block_size=1024):
    """Mean with bounded encrypted reductions to limit fixed-point noise."""
    count = values.numel()
    if count == 0:
        raise ValueError("Cannot take the mean of an empty encrypted vector")
    complete_blocks = count // block_size
    means = [values[offset:offset + block_size].sum() / block_size
             for offset in range(0, complete_blocks * block_size, block_size)]
    if complete_blocks:
        result = sum(means) / complete_blocks
        complete_weight = (complete_blocks * block_size) / count
        result = result * complete_weight
    else:
        result = values[:0].sum()
        complete_weight = 0.0
    remainder = count - complete_blocks * block_size
    if remainder:
        tail = values[complete_blocks * block_size:].sum() / remainder
        result = result + tail * (remainder / count)
    return result


def _local_coordinate(decomposition, client_id):
    return (decomposition["S_B"][:, None] * decomposition["V_B_blocks"][client_id]).matmul(decomposition["U_A_blocks"][client_id]) * decomposition["S_A"][None, :]

def _secure_fuld(encrypted_coordinates, num_clients, sample_counts=None, dbscan_quantile=None):
    start = time.perf_counter()
    similarities = {}
    modules = list(encrypted_coordinates.values())
    raw_dimension = sum(values[0].numel() for values in modules)
    similarity_mode = ("encrypted_exact" if raw_dimension <= FULD_EXACT_MAX_ELEMENTS
                       else f"encrypted_strided_sketch_{FULD_SKETCH_STRIDE}")
    # Secret normalization keeps raw inner products and norms out of the transcript.
    # Bound squared norms by one before invoking CrypTen's inverse square root.
    from crypten.config import cfg
    normalized = []
    with cfg.temp_override({"functions.reciprocal_nr_iters": 32,
                            "functions.sqrt_nr_iters": 32,
                            "functions.reciprocal_all_pos": True}):
        for client_id in range(num_clients):
            vector = crypten.cat([values[client_id].flatten() for values in modules])
            # CrypTen 0.4 TFP gives incorrect results for a very large
            # elementwise multiplication batch.  This deterministic public
            # sketch bounds the batch while preserving a reproducible sample
            # of every flattened coordinate stream.
            if vector.numel() > FULD_EXACT_MAX_ELEMENTS:
                vector = vector[::FULD_SKETCH_STRIDE]
            maximum = vector.abs().max() + 1e-6
            # Rescale before reciprocal: squaring a very large reciprocal can
            # overflow the 64-bit fixed-point ring even for tiny LoRA updates.
            thresholds = torch.tensor([4.0 ** exponent for exponent in range(-6, 7)])
            below = maximum <= thresholds
            bins = crypten.cat([below[:1], below[1:] - below[:-1]])
            # Comparisons have integer precision; promote before fractional weights.
            reciprocal_thresholds = crypten.cryptensor(thresholds.reciprocal(), src=0)
            scale = (reciprocal_thresholds * bins).sum()
            bounded = (vector * scale) / (maximum * scale + 1e-6)
            mean_square = _secure_mean(bounded.square())
            inv_norm = (mean_square + 1e-6).inv_sqrt()
            normalized.append(bounded * inv_norm)
    for i in range(num_clients):
        for j in range(i + 1, num_clients):
            similarity = _secure_mean(normalized[i] * normalized[j])
            value = _reveal_public_scalar(similarity, dst=1)
            if not torch.isfinite(torch.tensor(value)):
                raise FloatingPointError("Nonfinite secure cosine")
            similarities[(i, j)] = max(-1.0, min(1.0, value))
    counts = torch.tensor(sample_counts if sample_counts is not None else [1] * num_clients,
                          dtype=torch.float32)
    if counts.numel() != num_clients or not torch.isfinite(counts).all() or (counts <= 0).any():
        raise ValueError("Positive public sample counts required")
    comm = crypten.communicator.get()
    membership = torch.zeros(num_clients, dtype=torch.float32)
    dbscan_eps = torch.zeros(1, dtype=torch.float32)
    detection_failed = torch.zeros(1, dtype=torch.float32)
    if comm.get_rank() == 1:
        matrix = torch.eye(num_clients)
        for (i, j), value in similarities.items():
            matrix[i, j] = matrix[j, i] = value
        if dbscan_quantile is None:
            benign_set, _ = get_benign_set(matrix, eps=0.5, min_samples=2)
        else:
            benign_set, _ = get_benign_set(
                matrix, eps=None, min_samples=2, quantile=float(dbscan_quantile)
            )
        from defense_detect import clustering as clustering_module
        diagnostics = clustering_module.LAST_DIAGNOSTICS or {}
        dbscan_eps[0] = float(diagnostics.get("eps") or 0.0)
        detection_failed[0] = float(bool(diagnostics.get("detection_failed", False)))
        membership[benign_set] = 1.0
    comm.broadcast(membership, 1)
    comm.broadcast(dbscan_eps, 1)
    comm.broadcast(detection_failed, 1)
    weights = counts * membership
    weights /= weights.sum().clamp_min(1)
    return {"weights": weights, "similarities": similarities,
            "benign_set": torch.nonzero(membership).flatten().tolist(),
            "similarity_mode": similarity_mode,
            "dbscan_quantile": dbscan_quantile,
            "dbscan_eps": float(dbscan_eps.item()),
            "detection_failed": bool(detection_failed.item()),
            "seconds": time.perf_counter() - start}



def full_svd(local_params, local_client_id, num_clients, target_rank, communicator, sample_counts=None,
             numerical_scale=LORA_NUMERICAL_SCALE):
    """Exact-Full baseline: shared full-update aggregation plus masked SVD."""
    start = time.perf_counter()
    svd_seconds = 0.0
    max_error = 0.0
    aggregated_params = {}
    updates = OrderedDict()
    pairs = parameter_pairs(local_params)
    for b_name, a_name in pairs.items():
        updates[b_name] = []
        for source in range(num_clients):
            B = _share(local_params[b_name] * numerical_scale, source, local_client_id)
            A = _share(local_params[a_name] * numerical_scale, source, local_client_id)
            updates[b_name].append(B.matmul(A))
    fuld = _secure_fuld(updates, num_clients, sample_counts)
    for b_name, a_name in pairs.items():
        aggregate = sum(value * float(weight) for value, weight
                        in zip(updates[b_name], fuld["weights"]))
        t = time.perf_counter()
        B_global, A_global, error = _secure_masked_svd(aggregate, target_rank, local_client_id)
        svd_seconds += time.perf_counter() - t
        max_error = max(max_error, error)
        aggregated_params[b_name] = B_global / numerical_scale
        aggregated_params[a_name] = A_global / numerical_scale
    return {"server_compute_seconds": time.perf_counter() - start, "bfsvd_seconds": 0.0, "fuld_seconds": fuld["seconds"], "csa_seconds": 0.0, "svd_seconds": svd_seconds, "reveal_boundary": "masked_full_update_to_party1_svd", "low_rank_reconstruction_relative_error": max_error, "aggregated_params": aggregated_params, "communication": _stats(communicator), "weights": fuld["weights"].tolist(), "similarities": fuld["similarities"], "similarity_mode": fuld["similarity_mode"]}

def ersa(local_params, local_client_id, num_clients, seed, communicator, sample_counts=None,
         numerical_scale=LORA_NUMERICAL_SCALE):
    pairs = parameter_pairs(local_params)
    start = time.perf_counter()
    t = time.perf_counter()
    decompositions = OrderedDict((
        b_name,
        _bfsvd_module(local_params, local_client_id, num_clients, b_name, a_name,
                      index, seed, local_client_id, numerical_scale=numerical_scale),
    ) for index, (b_name, a_name) in enumerate(pairs.items()))
    bfsvd_seconds = time.perf_counter() - t
    encrypted_coordinates = OrderedDict()
    for b_name in pairs:
        coordinate = _local_coordinate(decompositions[b_name], local_client_id)
        encrypted_coordinates[b_name] = [_local_coordinate(decompositions[b_name], source) for source in range(num_clients)]
    fuld = _secure_fuld(encrypted_coordinates, num_clients, sample_counts)
    csa_start = time.perf_counter()
    svd_seconds = 0.0
    aggregated_params = {}
    max_error = 0.0
    for index, b_name in enumerate(pairs):
        aggregate = None
        for source, coordinate in enumerate(encrypted_coordinates[b_name]):
            term = coordinate * float(fuld["weights"][source])
            aggregate = term if aggregate is None else aggregate + term
        # Only the privately masked compact aggregate is opened to the SVD party.
        rank = decompositions[b_name]["rank"]
        t = time.perf_counter()
        compact_B, compact_A, error = _secure_masked_svd(aggregate, rank, local_client_id)
        svd_seconds += time.perf_counter() - t
        max_error = max(max_error, error)
        aggregated_params[b_name] = decompositions[b_name]["U_B"].matmul(compact_B) / numerical_scale
        aggregated_params[pairs[b_name]] = compact_A.matmul(decompositions[b_name]["V_A"]) / numerical_scale
    return {"server_compute_seconds": time.perf_counter() - start, "bfsvd_seconds": bfsvd_seconds, "fuld_seconds": fuld["seconds"], "csa_seconds": time.perf_counter() - csa_start, "svd_seconds": svd_seconds, "reveal_boundary": "masked_compact_csa_to_party1_svd", "low_rank_reconstruction_relative_error": max_error, "communication": _stats(communicator), "similarities": fuld["similarities"], "weights": fuld["weights"].tolist(), "similarity_mode": fuld["similarity_mode"], "aggregated_params": aggregated_params}
