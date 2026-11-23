import numpy as np
from sklearn.cluster import DBSCAN

LAST_DIAGNOSTICS = None

def get_benign_set(similarity_matrix, eps=None, min_samples=None, quantile=20.0):
    global LAST_DIAGNOSTICS
    similarity = similarity_matrix.detach().cpu().numpy().astype(np.float64)
    dist = (1.0 - similarity) / 2.0
    dist = np.maximum((dist + dist.T) / 2.0, 0.0)
    np.fill_diagonal(dist, 0.0)
    n = dist.shape[0]
    requested_eps = eps
    min_samples = 2 if min_samples is None else int(min_samples)
    failed = False; failure_reason = None; labels = np.full(n, -1, dtype=int); clusters = {}; noise = list(range(n)); benign_set = []; actual_eps = None
    try:
        if n < 2: raise ValueError("at least two clients are required")
        if eps is None:
            upper = dist[np.triu_indices(n, k=1)]; nonzero = upper[upper > 0]
            actual_eps = float(np.percentile(nonzero, float(quantile))) if len(nonzero) else 0.3
        else: actual_eps = float(eps)
        labels = DBSCAN(eps=actual_eps, min_samples=min_samples, metric="precomputed").fit_predict(dist)
        unique = sorted(int(x) for x in np.unique(labels) if x != -1)
        clusters = {str(x): np.where(labels == x)[0].astype(int).tolist() for x in unique}
        noise = np.where(labels == -1)[0].astype(int).tolist()
        if not unique: raise ValueError("all clients are noise")
        sizes = {x: len(clusters[str(x)]) for x in unique}
        largest = max(sizes.values())
        if len(unique) == 1 and not noise: raise ValueError("all clients form one cluster")
        if list(sizes.values()).count(largest) > 1: raise ValueError("largest cluster tie")
        benign_set = clusters[str(max(unique, key=lambda x: sizes[x]))]
        if not benign_set: raise ValueError("empty benign set")
    except Exception as exc:
        failed = True; failure_reason = str(exc); benign_set = []; print(f"[DBSCAN] detection failed: {exc}")
    predicted = sorted(set(range(n)) - set(benign_set)) if not failed else []
    LAST_DIAGNOSTICS = {"similarity": similarity.tolist(), "distance": dist.tolist(), "eps": actual_eps, "dbscan_quantile": float(quantile), "requested_eps": requested_eps, "min_samples": min_samples, "labels": labels.tolist(), "clusters": clusters, "noise": noise, "benign_set": benign_set, "predicted_malicious": predicted, "detection_failed": failed, "failure_reason": failure_reason}
    print(f"[DBSCAN] eps={actual_eps} min_samples={min_samples} clusters={clusters} noise={noise} benign_set={benign_set} detection_failed={failed}")
    return benign_set, labels
