
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, rbf_kernel
from scipy.special import rel_entr
from scipy.stats import wasserstein_distance

def compute_mmd(X_real, X_gen):
    K_rr = rbf_kernel(X_real, X_real)
    K_rg = rbf_kernel(X_real, X_gen)
    K_gg = rbf_kernel(X_gen, X_gen)
    return K_rr.mean() - 2 * K_rg.mean() + K_gg.mean()

def compute_kld(P, Q, bins=10):
    hist_p, _ = np.histogramdd(P, bins=bins, density=True)
    hist_q, _ = np.histogramdd(Q, bins=bins, density=True)
    hist_p += 1e-8
    hist_q += 1e-8
    hist_p /= hist_p.sum()
    hist_q /= hist_q.sum()
    return np.sum(rel_entr(hist_p, hist_q))

def compute_emd(P, Q):
    return np.mean([wasserstein_distance(P[:, i], Q[:, i]) for i in range(P.shape[1])])

def compute_cosine_sim(P, Q):
    sim = cosine_similarity(P.mean(axis=0).reshape(1, -1), Q.mean(axis=0).reshape(1, -1))
    return sim[0, 0]
