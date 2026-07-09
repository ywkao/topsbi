from numpy import linspace, trapz


def netEval(netOut, bWeights, sWeights, threshold=0.5, nPoints=200):
    """
    Compute a weighted ROC curve and threshold accuracy from a histogram of
    network output scores.

    Args:
        netOut: network output scores, shape (N,)
        bWeights: event weights under the background (c0) hypothesis, shape (N,)
        sWeights: event weights under the signal (c1) hypothesis, shape (N,)
        threshold: score cut used for the reported accuracy value
        nPoints: number of score bins used to build the ROC curve
    Returns:
        fpr, tpr: weighted false/true positive rate at each bin edge
        auc: trapezoidal area under the (fpr, tpr) curve
        a: weighted accuracy at `threshold`
    """
    eps = 1e-6

    n_neg_b = int((bWeights < 0).sum())
    n_neg_s = int((sWeights < 0).sum())
    print(f'[DEBUG] netEval: bWeights min={bWeights.min():.3e} max={bWeights.max():.3e} '
          f'negative={n_neg_b}/{bWeights.size}')
    print(f'[DEBUG] netEval: sWeights min={sWeights.min():.3e} max={sWeights.max():.3e} '
          f'negative={n_neg_s}/{sWeights.size}')
    if n_neg_b > 0 or n_neg_s > 0:
        # tpr/fpr are built as *cumulative* weighted sums while lowering the
        # score threshold; that's only guaranteed non-decreasing (and trapz
        # over it a valid AUC) if event weights are non-negative. A negative
        # weight can make the cumulative sum drop as more events are let in,
        # producing a non-monotonic curve and a meaningless AUC/accuracy with
        # no error raised.
        print(f'[WARNING] netEval: negative event weights break the monotonicity assumption '
              f'the ROC/AUC calculation relies on; clamping to 0 for this evaluation only')
    bWeights = bWeights.clip(min=0)
    sWeights = sWeights.clip(min=0)

    if netOut.max() == netOut.min():
        print(f'[WARNING] netEval: netOut is constant ({netOut.min():.3e}); ROC/AUC is degenerate '
              f'(every threshold gives the same rate, trapz will report auc=0)')

    bins = linspace(netOut.min(), netOut.max(), nPoints + 1)

    bTotal = bWeights.sum()
    sTotal = sWeights.sum()
    print(f'[DEBUG] netEval: bTotal={bTotal:.3e} sTotal={sTotal:.3e}')
    if bTotal < eps or sTotal < eps:
        print(f'[WARNING] netEval: bTotal or sTotal ~0 after clamping; clamping denominator to {eps:g}')
    bTotal = max(bTotal, eps)
    sTotal = max(sTotal, eps)

    tpr = []; fpr = []
    for i in range(len(bins)):
        tpr += [(sWeights[(netOut >= bins[-(i+1)]).ravel()].sum()/sTotal).item()]
        fpr += [(bWeights[(netOut >= bins[-(i+1)]).ravel()].sum()/bTotal).item()]

    a = ((sWeights[netOut >= threshold].sum() + bWeights[netOut <= threshold].sum())/(bTotal + sTotal)).item()
    auc = trapz(tpr, x=fpr).item()

    return fpr, tpr, auc, a
