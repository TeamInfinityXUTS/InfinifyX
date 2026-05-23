import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def _ensure_plot_dir():
    output_dir = os.path.join(os.getcwd(), "clearml_plots")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def _save_confusion_matrix(cm, labels, path):
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True label",
        xlabel="Predicted label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fmt = "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    ax.set_title("GNN Evaluation Confusion Matrix", fontsize=14)
    ax.tick_params(axis="x", labelsize=10, rotation=45)
    ax.tick_params(axis="y", labelsize=10)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def _save_class_metrics_bar(precision_vals, recall_vals, f1_vals, labels, path):
    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.bar(x - width, precision_vals, width, label="Precision", color="#4C72B0")
    ax.bar(x, recall_vals, width, label="Recall", color="#55A868")
    ax.bar(x + width, f1_vals, width, label="F1", color="#C44E52")
    ax.set_title("Per-class Precision / Recall / F1")
    ax.set_xlabel("Risk group")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.legend()
    for i, (p, r, f) in enumerate(zip(precision_vals, recall_vals, f1_vals)):
        ax.text(i - width, p + 0.02, f"{p:.3f}", ha="center", va="bottom")
        ax.text(i, r + 0.02, f"{r:.3f}", ha="center", va="bottom")
        ax.text(i + width, f + 0.02, f"{f:.3f}", ha="center", va="bottom")
    ax.set_title("Per-class Precision / Recall / F1", fontsize=14)
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=11)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def _save_risk_score_histogram(y_true_vals, y_pred_vals, path):
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.hist(y_true_vals, bins=20, alpha=0.5, label="True risk", color="#1f77b4")
    ax.hist(y_pred_vals, bins=20, alpha=0.5, label="Predicted risk", color="#55a868")
    ax.set_title("Risk Score Distribution")
    ax.set_xlabel("Risk score")
    ax.set_ylabel("Count")
    ax.legend()
    ax.set_title("Risk Score Distribution", fontsize=14)
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=11)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def _save_risk_calibration_curve(y_true_vals, y_pred_vals, path, n_bins=10):
    bins = np.linspace(
        min(y_pred_vals.min(), y_true_vals.min()),
        max(y_pred_vals.max(), y_true_vals.max()),
        n_bins + 1,
    )
    digitized = np.digitize(y_pred_vals, bins) - 1
    mean_pred = []
    mean_true = []
    for i in range(n_bins):
        mask = digitized == i
        if not np.any(mask):
            continue
        mean_pred.append(y_pred_vals[mask].mean())
        mean_true.append(y_true_vals[mask].mean())

    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    ax.plot(mean_pred, mean_true, marker="o", label="Calibration")
    ax.plot([bins[0], bins[-1]], [bins[0], bins[-1]], linestyle="--", color="gray", label="Ideal")
    ax.set_title("Risk Calibration Curve")
    ax.set_xlabel("Mean predicted risk")
    ax.set_ylabel("Mean true risk")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_title("Risk Calibration Curve", fontsize=14)
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=11)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def _save_class_error_bars(error_dict, path):
    labels = list(error_dict.keys())
    mae_vals = [error_dict[k]["mae"] for k in labels]
    rmse_vals = [error_dict[k]["rmse"] for k in labels]
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.bar(x - width / 2, mae_vals, width, label="MAE", color="#4C72B0")
    ax.bar(x + width / 2, rmse_vals, width, label="RMSE", color="#55A868")
    ax.set_title("Risk Group MAE / RMSE")
    ax.set_xlabel("Risk group")
    ax.set_ylabel("Error")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    for i, (mae_val, rmse_val) in enumerate(zip(mae_vals, rmse_vals)):
        ax.text(i - width / 2, mae_val + 0.01, f"{mae_val:.3f}", ha="center", va="bottom")
        ax.text(i + width / 2, rmse_val + 0.01, f"{rmse_val:.3f}", ha="center", va="bottom")
    ax.set_title("Risk Group MAE / RMSE", fontsize=13)
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=11)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def _save_precision_recall_heatmap(precision_vals, recall_vals, labels, path):
    """Save precision/recall heatmap visualization."""
    values = np.vstack([precision_vals, recall_vals])
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.6), 4), constrained_layout=True)
    im = ax.imshow(values, cmap="viridis", vmin=0, vmax=1)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Precision", "Recall"])
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i, j]:.3f}", ha="center", va="center",
                    color="white" if values[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    ax.set_title("Per-class Precision / Recall Heatmap", fontsize=13)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def _save_one_vs_rest_confusion_matrices(y_true_vals, y_pred_vals, labels, path):
    fig, axes = plt.subplots(1, len(labels), figsize=(len(labels) * 5, 5), constrained_layout=True)
    if len(labels) == 1:
        axes = [axes]
    for i, label_name in enumerate(labels):
        y_true_bin = [1 if y == i else 0 for y in y_true_vals]
        y_pred_bin = [1 if y == i else 0 for y in y_pred_vals]
        cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
        ax = axes[i]
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.set_title(f"{label_name} One-vs-Rest")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels([f"Not {label_name}", label_name], rotation=45, ha="right")
        ax.set_yticklabels([f"Not {label_name}", label_name])
        for j in range(cm.shape[0]):
            for k in range(cm.shape[1]):
                ax.text(k, j, cm[j, k], ha="center", va="center",
                        color="white" if cm[j, k] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.02, pad=0.04)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def _plot_evaluation_results(
    y_true,
    y_pred,
    y_true_raw,
    y_pred_raw,
    class_names,
    low_t,
    high_t,
    task,
):
    # Classification report
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        target_names=class_names,
        digits=3,
        zero_division=0,
    )
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    class_precisions, class_recalls, class_f1s, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        average=None,
        zero_division=0,
    )

    # Raw score metrics
    y_true_raw = np.array(y_true_raw, dtype=float)
    y_pred_raw = np.array(y_pred_raw, dtype=float)
    mae = np.mean(np.abs(y_true_raw - y_pred_raw))
    rmse = np.sqrt(np.mean((y_true_raw - y_pred_raw) ** 2))

    # Per-class error metrics
    class_errors = {}
    for idx, cls_name in enumerate(class_names):
        mask = np.where(np.array(y_true) == idx)[0]
        if len(mask) > 0:
            class_mae = np.mean(np.abs(y_true_raw[mask] - y_pred_raw[mask]))
            class_rmse = np.sqrt(np.mean((y_true_raw[mask] - y_pred_raw[mask]) ** 2))
        else:
            class_mae = float("nan")
            class_rmse = float("nan")
        class_errors[cls_name] = {
            "mae": class_mae,
            "rmse": class_rmse,
        }

    # Generate plots
    output_dir = _ensure_plot_dir()
    cm_path = os.path.join(output_dir, "gnn_confusion_matrix.png")
    metrics_path = os.path.join(output_dir, "gnn_metrics_bar.png")
    risk_hist_path = os.path.join(output_dir, "gnn_risk_score_histogram.png")
    calib_path = os.path.join(output_dir, "gnn_risk_calibration_curve.png")
    class_error_path = os.path.join(output_dir, "gnn_risk_class_error.png")
    heatmap_path = os.path.join(output_dir, "gnn_precision_recall_heatmap.png")
    ovr_cm_path = os.path.join(output_dir, "gnn_one_vs_rest_confusion_matrices.png")

    _save_confusion_matrix(confusion_matrix(y_true, y_pred, labels=[0, 1, 2]), class_names, cm_path)
    _save_class_metrics_bar(class_precisions, class_recalls, class_f1s, class_names, metrics_path)
    _save_risk_score_histogram(y_true_raw, y_pred_raw, risk_hist_path)
    _save_risk_calibration_curve(y_true_raw, y_pred_raw, calib_path)
    _save_class_error_bars(class_errors, class_error_path)
    _save_precision_recall_heatmap(class_precisions, class_recalls, class_names, heatmap_path)
    _save_one_vs_rest_confusion_matrices(np.array(y_true), np.array(y_pred), class_names, ovr_cm_path)

    # Log metrics and images to ClearML
    logger = task.get_logger()
    logger.report_text(report)
    logger.report_scalar("metrics", "accuracy", float(accuracy), 0)
    logger.report_scalar("metrics", "macro_precision", float(precision), 0)
    logger.report_scalar("metrics", "macro_recall", float(recall), 0)
    logger.report_scalar("metrics", "macro_f1", float(f1), 0)
    logger.report_scalar("risk", "mae", float(mae), 0)
    logger.report_scalar("risk", "rmse", float(rmse), 0)
    for name, error in class_errors.items():
        logger.report_scalar("risk_levels", f"{name}_mae", float(error["mae"]), 0)
        logger.report_scalar("risk_levels", f"{name}_rmse", float(error["rmse"]), 0)
    
    logger.report_image("evaluation", "confusion_matrix", local_path=cm_path, iteration=0)
    logger.report_image("evaluation", "summary_metrics", local_path=metrics_path, iteration=0)
    logger.report_image("evaluation", "risk_score_histogram", local_path=risk_hist_path, iteration=0)
    logger.report_image("evaluation", "risk_calibration_curve", local_path=calib_path, iteration=0)
    logger.report_image("evaluation", "risk_class_error", local_path=class_error_path, iteration=0)
    logger.report_image("evaluation", "precision_recall_heatmap", local_path=heatmap_path, iteration=0)
    logger.report_image("evaluation", "one_vs_rest_confusion", local_path=ovr_cm_path, iteration=0)

    return {
        "accuracy": float(accuracy),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
    }
