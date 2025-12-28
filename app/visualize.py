def plot_prediction(img, probs, labels, thresholds):
    import numpy as np
    import matplotlib.pyplot as plt
    from io import BytesIO

    # --- normalize inputs ---
    if isinstance(probs, dict):
        probs = np.array([probs[label] for label in labels])

    if isinstance(thresholds, dict):
        thresholds = np.array([thresholds[label] for label in labels])

    # --- plotting code ---
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4))

    ax[0].imshow(img)
    ax[0].axis("off")
    ax[0].set_title("Patient X-ray")


    sorted_idx = np.argsort(probs)[::-1]
    sorted_probs = []
    sorted_labels = []
    sorted_thresholds = []
    thresholds = np.asarray(thresholds)
    if thresholds.ndim == 0:
        thresholds = np.full(len(probs), thresholds)

    for i in sorted_idx:
        label = labels[i]
        if label == "No Finding":
            continue
        sorted_probs.append(probs[i])
        sorted_labels.append(label)
        sorted_thresholds.append(thresholds[i])
    
    sorted_probs = np.array(sorted_probs)
    sorted_thresholds = np.array(sorted_thresholds)


# --- color bars based on threshold ---
    colors = [
        "red" if p >= t else "green"
        for p, t in zip(sorted_probs, sorted_thresholds)
    ]

    ax[1].barh(sorted_labels, sorted_probs, color=colors)

    # draw threshold lines
    for y, t in enumerate(sorted_thresholds):
        ax[1].plot(
            [t, t],
            [y - 0.4, y + 0.4],
            linestyle="--",
            color="black",
            linewidth=1
        )

    from matplotlib.patches import Patch

    legend_items = [
        Patch(color="red", label="Predicted Positive"),
        Patch(color="green", label="Predicted Negative"),
        Patch(edgecolor="black", facecolor="none", linestyle="--", label="Threshold")
    ]

    ax[1].legend(
        handles=legend_items,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=9,
        frameon=False
    )



    ax[1].invert_yaxis()
    ax[1].set_xlim(0, 1)
    ax[1].set_title("Model predictions")

    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close("all")
    buf.seek(0)

    return buf
