"""Cross-round state and metrics for optional Temporal-FULD."""


def update_temporal_state(consecutive_hits, quarantine_set, raw_predicted_malicious,
                          detection_failed, threshold):
    """Update consecutive raw DBSCAN hits; failed rounds leave state untouched."""
    newly_quarantined = []
    if detection_failed:
        return newly_quarantined
    predicted = set(raw_predicted_malicious)
    for client_id in range(len(consecutive_hits)):
        if client_id in predicted:
            consecutive_hits[client_id] += 1
            if consecutive_hits[client_id] >= threshold and client_id not in quarantine_set:
                quarantine_set.add(client_id)
                newly_quarantined.append(client_id)
        else:
            consecutive_hits[client_id] = 0
    return newly_quarantined


def temporal_detection_metrics(effective_predicted_malicious, malicious_ids, num_clients):
    """Compute metrics only; ground truth never affects the detector/state."""
    predicted = set(effective_predicted_malicious)
    actual = set(malicious_ids)
    tp = len(predicted & actual)
    fp = len(predicted - actual)
    fn = len(actual - predicted)
    tn = num_clients - tp - fp - fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {
        "temporal_precision": precision,
        "temporal_recall": recall,
        "temporal_f1": f1,
        "temporal_false_positive_rate": fpr,
    }
