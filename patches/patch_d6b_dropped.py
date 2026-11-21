# ── D6b 補丁：被丟掉的 30% pixel 單獨跑 P(D)，跟保留的比較 ──
# 在已經跑完新版 notebook 的同一個 session 執行（需要 CACHE、_pd_groups、guard）
@guard('D6b')
def _d6b():
    G, G1, G2 = CACHE['G'], CACHE['G1'], CACHE['G2']
    _lg = lambda a: np.log10(np.clip(a, 1e-12, None))
    Y, Y1, Y2 = _lg(CACHE['dens']), _lg(CACHE.get('g1d', CACHE['dens'])), _lg(CACHE.get('g2d', CACHE['dens']))
    X, X1, X2 = (_lg(1.0 / np.clip(CACHE[k], 1e-6, None)) for k in ('gamma', 'g1g', 'g2g'))
    nuc = (CACHE['nuc'] & np.isfinite(G).all(-1) & np.isfinite(G1).all(-1) & np.isfinite(G2).all(-1)
           & np.isfinite(Y) & np.isfinite(Y1) & np.isfinite(Y2) & np.isfinite(X) & np.isfinite(X1) & np.isfinite(X2))
    feats = {
        'G':     (G1[nuc], G2[nuc], G[nuc]),
        'G+CV2': tuple(np.concatenate([g[nuc], y[nuc][:, None]], 1) for g, y in ((G1, Y1), (G2, Y2), (G, Y))),
        'XY':    tuple(np.stack([x[nuc], y[nuc]], 1) for x, y in ((X1, Y1), (X2, Y2), (X, Y))),
    }
    def _dis(F1, F2, Ff):
        F1, F2, Ff = (np.asarray(a, float) for a in (F1, F2, Ff))
        mu = (F1.mean(0) + F2.mean(0)) / 2; sd = (F1.std(0) + F2.std(0)) / 2 + 1e-9
        a1, a2 = (F1 - mu) / sd, (F2 - mu) / sd
        Cg = (a1.T @ a2) / len(a1); Cg = (Cg + Cg.T) / 2
        Ct = np.cov(np.vstack([a1, a2]).T)
        ev, U = np.linalg.eigh(Ct); Wh = U / np.sqrt(np.clip(ev, 1e-12, None))
        lam, V = np.linalg.eigh(Wh.T @ Cg @ Wh); o = np.argsort(lam)[::-1]
        lam, Wg = lam[o], (Wh @ V)[:, o]
        keep = np.where(lam >= D6_REL_MIN)[0]
        if len(keep) == 0: keep = np.array([0])
        P1, P2 = a1 @ Wg[:, keep], a2 @ Wg[:, keep]
        return ((P1 - P2) ** 2).sum(1) / (((P1 ** 2).sum(1) + (P2 ** 2).sum(1)) + 1e-12)
    idx = np.flatnonzero(nuc); H, W = nuc.shape
    groups = []
    for k, v in feats.items():
        d = _dis(*v); good = d <= np.quantile(d, 1 - D6_DROP_FRAC)
        for tag, sel in (('kept', good), ('DROPPED', ~good)):
            mk = np.zeros((H, W), bool); mk.flat[idx[sel]] = True
            groups.append((f'{k} {tag}', mk))
    out = _pd_groups(groups, 'D6b: kept (70%) vs dropped (30%) pixels, per noise filter')
    print('\n--- summary (best-lambda run) ---')
    for t, _ in groups:
        if t not in out: continue
        r = min(out[t]['runs'], key=lambda r: r['resid'])
        print(f'{t:16s} white {r["white"]:.2f} | slow-edge {r["slow_edge"]:.2f} | peaks {np.round(r["peaks"], 3)} '
              f'mass {np.round(r["peak_mass"], 2)} | noise {out[t]["noise"]:.4f}')
    # map of where the dropped pixels are (G+CV2 filter)
    d = _dis(*feats['G+CV2']); dm = np.full((H, W), np.nan); dm.flat[idx] = (d > np.quantile(d, 1 - D6_DROP_FRAC))
    plt.figure(figsize=(4.5, 4.5)); plt.imshow(dm, cmap='gray_r', interpolation='nearest')
    plt.title('dropped pixels (G+CV2), black', fontsize=9); plt.axis('off'); plt.show()
    RESULT['D6b'] = dict(verdict='ok', **{t: v for t, v in out.items()})
save_results()
