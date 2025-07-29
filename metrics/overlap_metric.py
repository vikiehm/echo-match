import numpy as np
from utils.registry import METRIC_REGISTRY
from metrics.geodist_metric import plot_pcks


@METRIC_REGISTRY.register()
def calculate_overlap_iou(pred_overlap, gt_overlap, threshold=0.5):
    """
    Calculate IoU between predicted overlap scores and ground truth overlap mask

    Args:
        pred_overlap (np.ndarray): Predicted overlap scores
        gt_overlap (np.ndarray): Ground truth overlap mask
        threshold (float, optional): Threshold for binarizing predicted scores. Default 0.5
    
    Returns:
        iou (float): IoU score between prediction and ground truth
    """
    # Binarize predicted overlap scores
    pred_overlap_mask = pred_overlap > threshold
    
    # Calculate intersection and union
    intersection = np.logical_and(gt_overlap, pred_overlap_mask)
    union = np.logical_or(gt_overlap, pred_overlap_mask)
    
    # Calculate IoU
    iou = np.sum(intersection) / np.sum(union)
    
    return iou
@METRIC_REGISTRY.register()
def plot_pck_p2p(geo_err, threshold=0.20, steps=40):
    # threshold is always hardcoded to 1, and steps is now meaningless but kept for compatibility
    geo_err = np.ravel(geo_err)
    def calculate_pck_values(threshold_steps, error_distances):
        pck_values = np.zeros((len(threshold_steps), 1))
        for idx, step in enumerate(threshold_steps):
            threshold = step / 100  # Convert step to absolute threshold
            is_correct = error_distances <= threshold
            pck_values[idx] = np.mean(is_correct)
        return pck_values, np.array(threshold_steps) / 100
    threshold_steps = range(0, 100)  # Original hardcoded range preserved
    pcks, thresholds = calculate_pck_values(threshold_steps, geo_err)
    auc = np.trapz(pcks.flatten()[:100], thresholds[:100])
    pcks = pcks.flatten()[:100]
    fig = plot_pcks([pcks], ["err"], threshold=1.0, steps=100)
    return auc, fig, pcks

def count_mismatches(pred_score, gt_mask):
    """Count false positives and false negatives between predicted and ground truth overlap masks."""
    pred_binary = (pred_score >= 0.5).astype(bool)
    false_pos = np.sum((gt_mask == 0) & (pred_binary == 1))
    false_neg = np.sum((gt_mask == 1) & (pred_binary == 0))
    return false_pos + false_neg

def calculate_overlap_auc(geo_err, overlap_score21, gt_partiality_mask21, corr_y):
    """
    Calculate overlap AUC metric by filtering geodesic errors based on overlap predictions.
    Adds inf values for vertices that are in ground truth but not predicted.
    
    Args:
        geo_err (np.ndarray): Geodesic error array
        overlap_score21 (np.ndarray): Predicted overlap scores from shape 2 to 1 
        gt_partiality_mask21 (np.ndarray): Ground truth partiality mask from shape 2 to 1
        corr_y (np.ndarray): Target vertex indices for correspondences
        
    Returns:
        np.ndarray: Filtered geodesic errors where both predicted and ground truth overlap agree,
                   padded with inf values for missed ground truth vertices
    """
    # Input validation
    assert gt_partiality_mask21.shape == overlap_score21.shape, "Overlap masks must have same shape"
    assert geo_err.shape == corr_y.shape, "Geodesic error must match correspondence shape"
    assert gt_partiality_mask21[corr_y].all(), "Ground truth mask must be 1 at all correspondence targets"

    # Create boolean masks
    overlap_mask = (overlap_score21 > 0.5).astype(bool)
    gt_mask = gt_partiality_mask21.astype(bool)
    
    # Filter correspondences where both predicted and ground truth overlap agree
    valid_target_vertices = (overlap_mask & gt_mask)
    valid_correspondences = valid_target_vertices[corr_y]
    filtered_geo_err = geo_err[valid_correspondences]

    # Count mismatches
    missed_vertices = count_mismatches(overlap_score21, gt_partiality_mask21)
    if missed_vertices > 0:
        inf_geo_err = np.inf * np.ones(missed_vertices)
        filtered_geo_err = np.concatenate([filtered_geo_err, inf_geo_err])
    
    return np.ravel(filtered_geo_err)

def write_geo_error_to_file(geo_err, filename, max_steps=150):
    """
    Process geodesic errors and write to file in a format compatible with tikz plotting.
    
    Args:
        geo_err (np.ndarray): Array of geodesic errors
        filename (str): Output filename to write results
        max_steps (int, optional): Maximum number of steps. Defaults to 150.
    """
    steps = range(0, max_steps)
    steps_used = np.array(steps) / 100  # Convert to threshold values
    
    # Calculate percentage of points below each threshold
    geo_values = np.zeros(len(steps))
    for i, threshold in enumerate(steps_used):
        geo_values[i] = np.mean(geo_err <= threshold) * 100  # Convert to percentage
    
    # Write to file in the required format
    steps_list_write = []
    for i in range(len(steps_used)):
        steps_list_write.append(
            str(steps_used[i]) + "  " + str(geo_values[i]) + r"\\" + "\n"
        )
        
    with open(filename, "w") as f:
        f.writelines(steps_list_write)
    
    # Calculate and print AUC for first 100 points
    auc = np.trapz(geo_values[:100] / 100, steps_used[:100])
    print(f"AUC for {filename}: {auc}")
    
    return auc





