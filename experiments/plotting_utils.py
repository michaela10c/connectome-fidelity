"""
Plotting utilities shared between test_item1_all_null_schemes.py (which
generates plots live during evaluation) and replot_from_saved.py (which
regenerates them later from saved data, with no network evaluation needed).
Deliberately dependency-light -- only matplotlib and numpy -- so replotting
never requires flyvis, torch, or a GPU.
"""
import matplotlib.pyplot as plt


def plot_comparison(cc_rdm, null_rdm, scheme, polarity, angles, null_r, obs_r,
                     out_prefix, checkpoint_label="untrained"):
    """Same visual structure as moving_edge_on_off.ipynb's own RDM heatmap
    (section 7h) and permutation null-distribution plot (7f2), adapted for
    a real-vs-null-scheme comparison instead of real-vs-weight-shuffled.
    checkpoint_label: 'untrained' (checkpoint 0) or 'trained' (final
    checkpoint) -- purely descriptive, for accurate titles/filenames."""
    n_stim = cc_rdm.shape[0]
    if polarity == "on_off":
        stim_labels = [f"{'OFF' if i % 2 == 0 else 'ON'} {angles[i // 2]}°" for i in range(n_stim)]
    else:
        stim_labels = [f"{a}°" for a in angles]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.suptitle(f"{checkpoint_label.capitalize()} comparison: CC vs {scheme} [{polarity}]", fontsize=10)
    for ax, rdm, title in zip(axes, [cc_rdm, null_rdm], [f"CC ({checkpoint_label}) — Cosine RDM",
                                                           f"{scheme} ({checkpoint_label}) — Cosine RDM"]):
        im = ax.imshow(rdm, cmap="viridis", vmin=0)
        ax.set_title(title, fontsize=8)
        ax.set_xticks(range(n_stim))
        ax.set_xticklabels(stim_labels, fontsize=5, rotation=90)
        ax.set_yticks(range(n_stim))
        ax.set_yticklabels(stim_labels, fontsize=5)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fname_rdm = f"{out_prefix}_{scheme}_{polarity}_{checkpoint_label}_rdms.png"
    fig.savefig(fname_rdm, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname_rdm}")

    fig2, ax2 = plt.subplots(figsize=(5, 3.5))
    ax2.hist(null_r, bins=50, color="steelblue", alpha=0.7, label="Null distribution")
    ax2.axvline(obs_r, color="red", linewidth=2, label=f"Observed = {obs_r:.3f}")
    ax2.set_xlabel("Spearman r (cosine RDM)")
    ax2.set_ylabel("Count")
    ax2.set_title(f"Permutation null: CC vs {scheme} [{polarity}]", fontsize=9)
    ax2.legend(fontsize=8)
    plt.tight_layout()
    fname_perm = f"{out_prefix}_{scheme}_{polarity}_{checkpoint_label}_permtest.png"
    fig2.savefig(fname_perm, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved: {fname_perm}")
