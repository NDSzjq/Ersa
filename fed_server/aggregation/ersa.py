from collections import OrderedDict

import torch

import base_llm.config as config
from defense_detect.clustering import get_benign_set
from defense_detect.spectral_similarity import compute_pairwise_similarity
from defense_detect.weights import compute_weights


METHOD_ALIASES = {
    "bfsvd": "ersa",
    "svd": "full_update_svd",
    "full": "factor_avg",
}
SUPPORTED_METHODS = {
    "ersa",
    "full_update_svd",
    "factor_avg",
    "freeze_a",
    "freeze_b",
}


def normalize_method(method):
    normalized = METHOD_ALIASES.get(method.lower(), method.lower())
    if normalized not in SUPPORTED_METHODS:
        raise ValueError(
            f"Unsupported aggregation method {method!r}; "
            f"choose one of {sorted(SUPPORTED_METHODS)}"
        )
    return normalized


def lora_parameter_pairs(lora_params):
    """Discover every LoRA A/B pair without assuming a model depth."""
    pairs = OrderedDict()
    names = set(lora_params)

    for b_name in sorted(name for name in names if name.endswith("lora_B")):
        stem = b_name[: -len("lora_B")]
        a_name = f"{stem}lora_A"
        if a_name not in lora_params:
            raise ValueError(f"Missing LoRA A factor for {b_name}: expected {a_name}")

        B = lora_params[b_name]
        A = lora_params[a_name]
        if B.ndim != 2 or A.ndim != 2 or B.shape[1] != A.shape[0]:
            raise ValueError(
                f"Invalid LoRA shapes for {stem.rstrip('.')}: "
                f"B={tuple(B.shape)}, A={tuple(A.shape)}"
            )
        pairs[stem.rstrip(".") or "lora"] = {
            "b_name": b_name,
            "a_name": a_name,
        }

    paired_names = {
        name
        for pair in pairs.values()
        for name in (pair["b_name"], pair["a_name"])
    }
    unpaired = sorted(names - paired_names)
    if unpaired:
        raise ValueError(f"Unpaired LoRA parameters: {unpaired}")
    if not pairs:
        raise ValueError("No LoRA A/B pairs were found")
    return pairs


def _client_lora(client_params):
    if "lora_params" in client_params:
        return client_params["lora_params"]
    if "lora_B" in client_params and "lora_A" in client_params:
        return {
            "lora_B": client_params["lora_B"],
            "lora_A": client_params["lora_A"],
        }
    raise ValueError(f"Unsupported client parameter format: {client_params.keys()}")


def _validate_clients(params_list, pairs):
    reference = _client_lora(params_list[0])
    expected_names = set(reference)
    for client_index, params in enumerate(params_list[1:], start=1):
        current = _client_lora(params)
        if set(current) != expected_names:
            raise ValueError(f"Client {client_index} has a different LoRA structure")
        for pair in pairs.values():
            for name in (pair["b_name"], pair["a_name"]):
                if current[name].shape != reference[name].shape:
                    raise ValueError(
                        f"Client {client_index} has a different shape for {name}"
                    )


def sample_count_weights(sample_counts):
    counts = torch.as_tensor(sample_counts, dtype=torch.float32)
    if counts.ndim != 1 or len(counts) == 0 or counts.sum() <= 0:
        raise ValueError(f"Invalid client sample counts: {sample_counts}")
    return counts / counts.sum()


def _decompose_module(B_list, A_list, device):
    if not B_list:
        raise ValueError("BF-SVD requires at least one client")

    B_list = [B.to(device) for B in B_list]
    A_list = [A.to(device) for A in A_list]
    rank = B_list[0].shape[1]
    in_dim = B_list[0].shape[0]
    out_dim = A_list[0].shape[1]
    nr = len(B_list) * rank
    if nr > min(in_dim, out_dim):
        raise ValueError(
            "Ersa requires N*R <= min(D1, D2) for the paper's compact "
            f"factorization, but N*R={nr}, D1={in_dim}, D2={out_dim}"
        )

    B_bar = torch.cat(B_list, dim=1)
    A_bar = torch.cat(A_list, dim=0)
    U_B, S_B, Vh_B = torch.linalg.svd(B_bar, full_matrices=False)
    U_A, S_A, Vh_A = torch.linalg.svd(A_bar, full_matrices=False)

    # The paper uses B_n = U_B Sigma_B V_n^B. PyTorch's Vh_B is
    # already the transposed right-singular matrix, so do not transpose it.
    V_B_blocks = list(torch.split(Vh_B, rank, dim=1))
    U_A_blocks = list(torch.split(U_A, rank, dim=0))
    if len(V_B_blocks) != len(B_list) or len(U_A_blocks) != len(A_list):
        raise RuntimeError("BF-SVD produced an invalid number of client blocks")

    return {
        "U_B": U_B,
        "S_B": S_B,
        "V_B": Vh_B,
        "U_A": U_A,
        "S_A": S_A,
        "V_A": Vh_A,
        "V_B_blocks": V_B_blocks,
        "U_A_blocks": U_A_blocks,
        "num_clients": len(B_list),
        "lora_rank": rank,
        "in_dim": in_dim,
        "out_dim": out_dim,
    }


def bfsvd_decompose(params_list, device="cpu"):
    """Run plaintext BF-SVD independently for every LoRA target module."""
    if not params_list:
        raise ValueError("BF-SVD requires at least one client")

    first = _client_lora(params_list[0])
    pairs = lora_parameter_pairs(first)
    _validate_clients(params_list, pairs)
    results = OrderedDict()

    for module_name, pair in pairs.items():
        b_name = pair["b_name"]
        a_name = pair["a_name"]
        module_result = _decompose_module(
            [_client_lora(params)[b_name] for params in params_list],
            [_client_lora(params)[a_name] for params in params_list],
            device=torch.device(device),
        )
        module_result.update(
            {"module_name": module_name, "b_name": b_name, "a_name": a_name}
        )
        results[module_name] = module_result
    return results


def spectral_coordinates(module_result):
    """Compute K_n = Sigma_B V_n^B U_n^A Sigma_A for all clients."""
    S_B = module_result["S_B"]
    S_A = module_result["S_A"]
    return [
        (S_B[:, None] * V_B_n) @ U_A_n * S_A[None, :]
        for V_B_n, U_A_n in zip(
            module_result["V_B_blocks"], module_result["U_A_blocks"]
        )
    ]


def all_spectral_coordinates(bfsvd_results):
    return OrderedDict(
        (module_name, spectral_coordinates(result))
        for module_name, result in bfsvd_results.items()
    )


def _csa_module(weights, K_list, module_result):
    C = torch.zeros_like(K_list[0])
    weights = torch.as_tensor(weights, device=C.device, dtype=C.dtype)
    for weight, K_n in zip(weights, K_list):
        C.add_(K_n.to(C.device), alpha=float(weight))

    U_C, S_C, Vh_C = torch.linalg.svd(C, full_matrices=False)
    rank = module_result["lora_rank"]
    sqrt_S = torch.diag(torch.sqrt(torch.clamp(S_C[:rank], min=0)))
    B_global = module_result["U_B"] @ U_C[:, :rank] @ sqrt_S
    A_global = sqrt_S @ Vh_C[:rank, :] @ module_result["V_A"]
    return B_global, A_global


def csa_aggregate(bfsvd_results, coordinates_by_module, weights):
    """Run plaintext CSA for all LoRA target modules."""
    result = {}
    for module_name, module_result in bfsvd_results.items():
        B_global, A_global = _csa_module(
            weights,
            coordinates_by_module[module_name],
            module_result,
        )
        result[module_result["b_name"]] = B_global.cpu()
        result[module_result["a_name"]] = A_global.cpu()
    return result


def full_update_svd(params_list, weights, device="cpu"):
    """Materialize sum_n omega_n B_n A_n and apply rank-R truncated SVD."""
    first = _client_lora(params_list[0])
    pairs = lora_parameter_pairs(first)
    _validate_clients(params_list, pairs)
    target_device = torch.device(device)
    weights = torch.as_tensor(weights, device=target_device)
    result = {}

    for pair in pairs.values():
        b_name = pair["b_name"]
        a_name = pair["a_name"]
        B0 = first[b_name].to(target_device)
        A0 = first[a_name].to(target_device)
        update = torch.zeros(
            B0.shape[0],
            A0.shape[1],
            device=target_device,
            dtype=B0.dtype,
        )
        for weight, params in zip(weights, params_list):
            lora = _client_lora(params)
            update.add_(
                lora[b_name].to(target_device)
                @ lora[a_name].to(target_device),
                alpha=float(weight),
            )

        U, S, Vh = torch.linalg.svd(update, full_matrices=False)
        rank = B0.shape[1]
        sqrt_S = torch.diag(torch.sqrt(torch.clamp(S[:rank], min=0)))
        result[b_name] = (U[:, :rank] @ sqrt_S).cpu()
        result[a_name] = (sqrt_S @ Vh[:rank, :]).cpu()
    return result


def factor_average(params_list, weights):
    """FedIT-style baseline that averages A and B independently."""
    first = _client_lora(params_list[0])
    result = {name: torch.zeros_like(parameter) for name, parameter in first.items()}
    for weight, params in zip(weights, params_list):
        for name, parameter in _client_lora(params).items():
            result[name].add_(parameter.to(result[name].device), alpha=float(weight))
    return result


def frozen_factor_average(params_list, weights, previous_lora, frozen_factor):
    """Average the trainable factor while retaining the shared frozen factor."""
    if frozen_factor not in {"a", "b"}:
        raise ValueError("frozen_factor must be 'a' or 'b'")
    result = factor_average(params_list, weights)
    suffix = "lora_A" if frozen_factor == "a" else "lora_B"
    for name in result:
        if name.endswith(suffix):
            result[name] = previous_lora[name].clone()
    return result


def fuld_detect_multi_module(bfsvd_results, sample_counts, eps=None, min_samples=None, quantile=20.0):
    """Plaintext FULD using the joint spectral representation of all modules."""
    coordinates_by_module = all_spectral_coordinates(bfsvd_results)
    module_lists = list(coordinates_by_module.values())
    client_count = len(module_lists[0])
    features = [
        torch.cat([coordinates[index].reshape(-1) for coordinates in module_lists])
        for index in range(client_count)
    ]
    similarity = compute_pairwise_similarity(features, method="cosine")
    benign_set, labels = get_benign_set(
        similarity,
        eps=eps,
        min_samples=min_samples,
        quantile=quantile,
    )
    from defense_detect import clustering as clustering_module
    detection_failed = bool(clustering_module.LAST_DIAGNOSTICS.get("detection_failed", False))
    weights = sample_count_weights(sample_counts) if detection_failed else compute_weights(benign_set, sample_counts)
    return {
        "weights": weights,
        "benign_set": benign_set,
        "labels": labels,
        "similarity": similarity,
        "coordinates_by_module": coordinates_by_module,
        "detection_failed": detection_failed,
    }


def aggregate(
    method,
    params_list,
    weights,
    previous_lora,
    device="cpu",
    use_fuld=False,
    sample_counts=None,
    fuld_weight_selector=None,
    dbscan_quantile=20.0,
):
    """Aggregate all LoRA modules with one experiment-one method."""
    method = normalize_method(method)
    bfsvd_results = None
    coordinates_by_module = None

    if method == "ersa" or use_fuld:
        bfsvd_results = bfsvd_decompose(params_list, device=device)
        coordinates_by_module = all_spectral_coordinates(bfsvd_results)

    if use_fuld:
        if sample_counts is None:
            raise ValueError("FULD requires sample counts")
        fuld = fuld_detect_multi_module(bfsvd_results, sample_counts, quantile=dbscan_quantile)
        weights = fuld["weights"]
        coordinates_by_module = fuld["coordinates_by_module"]
        if fuld_weight_selector is not None:
            weights, selection = fuld_weight_selector(fuld)
            fuld.update(selection)
    else:
        fuld = None

    if torch.as_tensor(weights).sum() <= 0:
        if fuld is not None and fuld.get("no_effective_clients", False):
            return {name: value.clone() for name, value in previous_lora.items()}, torch.as_tensor(weights, dtype=torch.float32), fuld
        raise RuntimeError("Aggregation weights sum to zero")

    if method == "ersa":
        lora = csa_aggregate(bfsvd_results, coordinates_by_module, weights)
    elif method == "full_update_svd":
        lora = full_update_svd(params_list, weights, device=device)
    elif method == "factor_avg":
        lora = factor_average(params_list, weights)
    elif method == "freeze_a":
        lora = frozen_factor_average(
            params_list, weights, previous_lora, frozen_factor="a"
        )
    else:
        lora = frozen_factor_average(
            params_list, weights, previous_lora, frozen_factor="b"
        )

    return lora, torch.as_tensor(weights, dtype=torch.float32), fuld
