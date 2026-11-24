"""
Phase 4 - Noise, jitter, bathtub curves and BER for the equalised PCB3 links
--------------------------------------------------------------------------
Uses the best equaliser settings from Phase 3 (FFE + CTLE + DFE) and adds:
  * RX noise      : Gaussian, SIGMA_N (V rms, input-referred)
  * Random jitter : Gaussian, RJ_PS (ps rms)
  * Determ. jitter: dual-Dirac, DJ_PS (ps peak-to-peak)
BER is computed analytically from every simulated ISI sample (Q-function), so it
reaches 1e-15 and below without simulating billions of bits.

Outputs: phase4_bathtub.png, phase4_noise_sweep.png + printed table
Usage  : python phase4_jitter_ber.py   (needs phase3_equalization.py and PCB3.s4p)
"""
import math
import numpy as np
import matplotlib.pyplot as plt
import phase3_equalization as p3

SIGMA_N = 2e-3      # RX input-referred noise, V rms
RJ_PS   = 0.3       # random jitter, ps rms
DJ_PS   = 2.0       # deterministic jitter, ps pk-pk (dual-Dirac)
TARGET  = {2: 1e-12, 4: 2.4e-4}   # NRZ raw BER target; PAM4 pre-FEC target (KP4 RS-FEC)

_erfc = np.vectorize(math.erfc)
Q = lambda x: 0.5 * _erfc(x / math.sqrt(2))

def q_inv(p):
    lo, hi = 0.0, 40.0
    for _ in range(80):
        m = (lo + hi) / 2
        lo, hi = (m, hi) if Q(m) > p else (lo, m)
    return (lo + hi) / 2

def ber_vs_phase(rows, sym, sigma):
    """BER at every sampling phase of the centre UI, from ISI samples + Gaussian noise."""
    sps = rows.shape[1] // 2
    cols = np.arange(sps // 2, sps + sps // 2 + 1)            # centre UI, -0.5..+0.5 UI
    lv = np.unique(sym); M = len(lv)
    centre = rows[:, sps]
    means = np.array([centre[sym == v].mean() for v in lv])
    thr = (means[:-1] + means[1:]) / 2                        # decision thresholds
    ber = []
    for c in cols:
        x = rows[:, c]; p = np.zeros_like(x)
        for j, v in enumerate(lv):
            m = sym == v
            if j > 0:     p[m] += Q((x[m] - thr[j - 1]) / sigma)   # falls below lower threshold
            if j < M - 1: p[m] += Q((thr[j] - x[m]) / sigma)       # rises above upper threshold
        ber.append(p.mean() / math.log2(M))                    # Gray coding: 1 bit error / symbol error
    phase = (cols - sps) / sps
    return phase, np.clip(np.array(ber), 1e-40, 0.5), thr, means

def add_jitter(phase, ber, rj_ui, dj_ui):
    """Convolve the BER-vs-phase curve with an RJ (Gaussian) + DJ (dual-Dirac) jitter pdf."""
    fine = np.arange(-0.5, 0.5 + 1e-9, 0.002)
    lb = np.interp(fine, phase, np.log10(ber))
    ber_f = 10 ** lb
    span = dj_ui / 2 + 8 * rj_ui
    tau = np.arange(-span, span + 1e-12, 0.002)
    pdf = 0.5 * (np.exp(-0.5 * ((tau - dj_ui / 2) / rj_ui) ** 2) + np.exp(-0.5 * ((tau + dj_ui / 2) / rj_ui) ** 2))
    pdf /= pdf.sum()
    out = np.zeros_like(fine)
    for t, w in zip(tau, pdf):
        out += w * np.interp(fine + t, fine, ber_f, left=0.5, right=0.5)
    return fine, np.clip(out, 1e-40, 0.5)

def opening_at(phase, ber, target):
    ok = phase[ber <= target]
    return (ok.max() - ok.min()) if len(ok) else 0.0

if __name__ == "__main__":
    rows_out, sweep = [], {}
    fig, ax = plt.subplots(1, len(p3.CASES), figsize=(16, 4.8))
    for a, (label, baud, lv) in zip(ax, p3.CASES):
        ui_ps = 1e12 / baud
        st = p3.optimise_case(label, baud, lv)
        p, taps, pk, dfe = st["FFE + CTLE + DFE"]
        rows, eh, ew = p3.simulate_eye(p, lv, dfe)
        sym = p3.SYM_CACHE[lv][200:200 + rows.shape[0]]

        ph, ber_n = ber_vs_phase(rows, sym, SIGMA_N)[:2]
        fine, ber_j = add_jitter(ph, ber_n, RJ_PS / ui_ps, DJ_PS / ui_ps)
        tgt = TARGET[lv]
        heo = opening_at(fine, ber_j, tgt)
        veo = eh - 2 * q_inv(tgt * math.log2(lv)) * SIGMA_N        # vertical opening left at target
        rows_out.append((label, eh * 1e3, ew, ber_n.min(), ber_j.min(), tgt, heo, veo * 1e3))

        a.semilogy(ph, ber_n, "C0", lw=1.5, label=f"noise only (σn = {SIGMA_N*1e3:.0f} mV)")
        a.semilogy(fine, ber_j, "C3", lw=2, label=f"+ jitter (RJ {RJ_PS} ps rms, DJ {DJ_PS} ps pp)")
        a.axhline(tgt, c="k", ls="--", lw=1, label=f"target {tgt:.1e}")
        a.set(ylim=(1e-18, 1), xlim=(-0.5, 0.5), xlabel="sampling phase (UI)", ylabel="BER",
              title=f"Bathtub – {label}  (eq: FFE+CTLE+DFE)\nopening at target = {heo:.2f} UI")
        a.grid(alpha=.3, which="both"); a.legend(fontsize=7, loc="lower center")

        sig = np.linspace(0.5e-3, 12e-3, 24)
        sweep[label] = (sig, [ber_vs_phase(rows, sym, s_)[1].min() for s_ in sig], tgt)
    plt.tight_layout(); plt.savefig("results/python/phase4_bathtub.png", dpi=130)

    fig, a = plt.subplots(figsize=(7, 4.5))
    for i, (label, (sig, b, tgt)) in enumerate(sweep.items()):
        a.semilogy(sig * 1e3, np.clip(b, 1e-40, 1), f"C{i}", lw=2, label=label)
        a.axhline(tgt, c=f"C{i}", ls=":", lw=1)
    a.set(xlabel="RX noise σn (mV rms)", ylabel="BER at best phase", ylim=(1e-30, 1),
          title="Noise tolerance (dotted = each link's BER target)")
    a.grid(alpha=.3, which="both"); a.legend()
    plt.tight_layout(); plt.savefig("results/python/phase4_noise_sweep.png", dpi=130)

    print(f"Assumptions: RX noise {SIGMA_N*1e3:.1f} mV rms, RJ {RJ_PS} ps rms, DJ {DJ_PS} ps pk-pk\n")
    print(f"{'Case':10s} {'eyeH mV':>8s} {'eyeW UI':>8s} {'BER noise':>10s} {'BER +jit':>10s} {'target':>8s} "
          f"{'H-open UI':>10s} {'V-open mV':>10s}  verdict")
    for lab, eh, ew, bn, bj, tgt, heo, veo in rows_out:
        verdict = "PASS" if bj <= tgt else "FAIL"
        print(f"{lab:10s} {eh:8.0f} {ew:8.2f} {bn:10.1e} {bj:10.1e} {tgt:8.1e} {heo:10.2f} {veo:10.0f}  {verdict}")
    print("Saved phase4_bathtub.png and phase4_noise_sweep.png")
