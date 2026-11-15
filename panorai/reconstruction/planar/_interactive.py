"""Small, session-local segmentation algorithms; no pretrained model or weights.

The classifier learns color, not planarity. ROI/support are explicit domains,
never inferred from RGB. All fitting is deterministic and discarded after use.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def marker_watershed(image, positive, negative, roi, support):
    """Flood a smoothed Lab-gradient relief from foreground/background markers.

    Called on a gnomonic chart, not on a distorted/periodically cut ERP. Outside
    ROI and the chart border compete as background; unsupported data never flood.
    """
    from skimage.segmentation import watershed

    lab = cv2.cvtColor(image.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)
    smooth = cv2.GaussianBlur(lab, (5, 5), 1)
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    relief = np.sqrt(np.sum(gx * gx + gy * gy, axis=-1))
    markers = np.zeros(roi.shape, np.int32)
    markers[negative | ~roi] = 2
    markers[[0, -1], :] = 2
    markers[:, [0, -1]] = 2
    markers[positive & roi & support & ~negative] = 1
    labels = watershed(relief, markers, mask=support, connectivity=1)
    return (labels == 1) & roi & support & ~negative


def _features(rgb):
    # OpenCV float Lab has L in [0,100], a/b approximately [-128,128].
    colors = np.asarray(rgb, dtype=np.float32).reshape(-1, 1, 3) / 255
    lab = cv2.cvtColor(colors, cv2.COLOR_RGB2LAB).reshape(-1, 3)
    lab /= np.array([100, 128, 128], np.float32)
    return np.column_stack(
        (
            lab,
            lab * lab,
            lab[:, 0] * lab[:, 1],
            lab[:, 0] * lab[:, 2],
            lab[:, 1] * lab[:, 2],
        )
    ).astype(np.float64)


def color_classifier(image, positive, negative, roi, support):
    """Balanced L2 logistic regression on nine Lab/quadratic color features.

    Fit at most 4096 seed pixels/class, selected deterministically in raster
    order. Predict only the explicit supported ROI, in native-grid row blocks.
    No coordinates, target domain labels, external images or depth are features.
    Scores are decision scores, not calibrated correctness probabilities.
    """
    domain = roi & support
    masks = (positive & domain, negative & domain)
    counts = [int(np.count_nonzero(mask)) for mask in masks]
    if min(counts) < 2:
        raise ValueError(
            "classifier requires at least two positive and two negative seed "
            "pixels inside the supported ROI"
        )
    samples = []
    flat_image = image.reshape(-1, 3)
    for mask, count in zip(masks, counts):
        indices = np.flatnonzero(mask)
        selected = indices[np.linspace(0, count - 1, min(count, 4096), dtype=int)]
        samples.append(_features(flat_image[selected]))
    x = np.concatenate(samples)
    y = np.concatenate((np.ones(len(samples[0])), np.zeros(len(samples[1]))))
    weights = np.concatenate(
        [np.full(len(sample), 0.5 / len(sample)) for sample in samples]
    )
    mean = weights @ x
    scale = np.sqrt(weights @ ((x - mean) ** 2))
    scale = np.maximum(scale, 0.05)
    x = np.column_stack((np.ones(len(x)), (x - mean) / scale))
    penalty = 0.01

    def objective(beta):
        score = x @ beta
        loss = weights @ (np.logaddexp(0, score) - y * score)
        loss += 0.5 * penalty * (beta[1:] @ beta[1:])
        gradient = x.T @ (weights * (expit(score) - y))
        gradient[1:] += penalty * beta[1:]
        return float(loss), gradient

    fit = minimize(
        objective,
        np.zeros(x.shape[1]),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 150, "ftol": 1e-12, "gtol": 1e-7},
    )
    if not fit.success or not np.isfinite(fit.x).all():
        raise ValueError("local classifier did not converge; revise the seeds")
    beta = fit.x
    scores = x @ beta
    # Equal colors with contradictory class labels contain no usable evidence.
    separation = float(
        np.mean(scores[: len(samples[0])]) - np.mean(scores[len(samples[0]) :])
    )
    if separation < 1e-4:
        raise ValueError(
            "positive and negative colors are indistinguishable; revise seeds "
            "or choose watershed/SAM"
        )
    mask = np.zeros(domain.shape, bool)
    for row in range(0, domain.shape[0], 128):
        block = domain[row : row + 128]
        if not block.any():
            continue
        features = (_features(image[row : row + 128][block]) - mean) / scale
        mask[row : row + 128][block] = features @ beta[1:] + beta[0] > 0
    mask = (mask | masks[0]) & ~negative & domain
    diagnostics = {
        "algorithm": "balanced-l2-logistic-lab-quadratic/v1",
        "positive_seed_pixels": counts[0],
        "negative_seed_pixels": counts[1],
        "training_samples": [len(sample) for sample in samples],
        "maximum_samples_per_class": 4096,
        "l2_penalty": penalty,
        "iterations": int(fit.nit),
        "threshold_logit": 0.0,
        "prediction_domain": "explicit-supported-roi-native-grid",
        "warning": "Color classification is not evidence of planarity or accuracy.",
    }
    return mask, diagnostics
