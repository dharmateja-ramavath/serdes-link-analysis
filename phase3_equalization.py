"""
Phase 3 - Equalization: TX FFE, RX CTLE and RX DFE on the PCB3 channel
---------------------------------------------------------------------
Stages compared for each case (same channel, step by step):
  1. No EQ
  2. TX FFE (3 taps: pre, main, post)          - optimised by grid search
  3. TX FFE + RX CTLE (1 zero, 2 poles)        - peaking swept 0..15 dB
  4. TX FFE + RX CTLE + RX DFE (3 taps, ideal) - post-cursors h1..h3 cancelled
Outputs: phase3_eyes.png, phase3_pulses.png, phase3_ctle.png + printed table

Usage   : python phase3_equalization.py      (run from the repository root)
Requires: numpy, matplotlib
"""
import numpy as np
import matplotlib.pyplot as plt

FILE = "data/PCB3.s4p"
P_IN, N_IN, P_OUT, N_OUT = 2, 0, 3, 1          # 1=C_N_J1 2=C_N_J2 3=C_P_J1 4=C_P_J2
SPS, N_SYM, TX_SWING = 32, 4000, 1.0
N_DFE = 3                                      # DFE taps
CASES = [("25G NRZ", 25e9, 2), ("56G PAM4", 28e9, 4), ("112G PAM4", 56e9, 4)]
STAGES = ["No EQ", "FFE", "FFE + CTLE", "FFE + CTLE + DFE"]

# ---------------------------------------------------------------- channel
def load_touchstone(fn):
    vals = []
    for line in open(fn, encoding="latin1"):
        line = line.split("!")[0].strip()
        if line and not line.startswith("#"):
            vals += line.split()
    N = int(fn.lower().split(".s")[-1].rstrip("p"))
    v = np.array(vals, float).reshape(-1, 1 + 2 * N * N)
    s = (v[:, 1::2] * np.exp(1j * np.deg2rad(v[:, 2::2]))).reshape(-1, N, N)
    return v[:, 0], s

f, s = load_touchstone(FILE)
H_CH = 0.5 * (s[:, P_OUT, P_IN] - s[:, P_OUT, N_IN] - s[:, N_OUT, P_IN] + s[:, N_OUT, N_IN])

def ctle(fr, f_nyq, peak_db):
    """1-zero / 2-pole CTLE, normalised to 0 dB peak gain (DC gain = -peaking)."""
    if peak_db <= 0:
        return np.ones_like(fr, dtype=complex)
    wp1, wp2 = 2 * np.pi * f_nyq, 2 * np.pi * 2 * f_nyq
    wz = wp1 / 10 ** (peak_db / 20)
    sj = 1j * 2 * np.pi * fr
    Hc = (wp1 * wp2 / wz) * (sj + wz) / ((sj + wp1) * (sj + wp2))
    return Hc / np.abs(Hc).max()

def impulse_response(H, dt, n_fft):
    fg = np.fft.rfftfreq(n_fft, dt)
    fin = fg <= f[-1]
    mag = np.interp(fg, f, np.abs(H), right=0.0)
    ph = np.interp(fg, f, np.unwrap(np.angle(H)))
    fl = fg[fin]; slope = (ph[fin][-1] - ph[fin][-50]) / (fl[-1] - fl[-50])
    ph[~fin] = ph[fin][-1] + slope * (fg[~fin] - fl[-1])
    t0, fmax = 0.8 * f[-1], f[-1]
    w = np.ones_like(fg); b = (fg > t0) & fin
    w[b] = 0.5 * (1 + np.cos(np.pi * (fg[b] - t0) / (fmax - t0))); w[~fin] = 0
    return np.fft.irfft(mag * w * np.exp(1j * ph), n_fft)

def pulse_response(H, ui):
    dt = ui / SPS
    n_fft = int(2 ** np.ceil(np.log2(20e-9 / dt)))
    p = np.convolve(impulse_response(H, dt, n_fft), np.ones(SPS))[:n_fft] * TX_SWING / 2
    return p

def main_cursor(p):
    a = np.where(p > 0.5 * p.max())[0]
    return int(round((a[0] + a[-1]) / 2))

def apply_ffe(p, taps):
    """TX FFE taps (c-1, c0, c1) applied to the pulse response."""
    out = taps[1] * p
    out += taps[0] * np.roll(p, -SPS)            # pre-cursor tap: one UI earlier
    out += taps[2] * np.roll(p, SPS)             # post-cursor tap: one UI later
    return out

def cursors(p, k0, n_pre=3, n_post=12):
    return p[k0 + SPS * np.arange(-n_pre, n_post + 1)], n_pre

def pda_eye(p, levels, dfe_taps):
    """Peak-distortion worst-case eye height at the best phase (fast objective)."""
    k0 = main_cursor(p); best = -1e9
    for d in range(-SPS // 4, SPS // 4 + 1, 2):
        c, npre = cursors(p, k0 + d)
        h0 = c[npre]
        isi = np.concatenate([c[:npre], c[npre + 1 + dfe_taps:]])
        eye = (2 * h0 - 2 * np.abs(isi).sum()) if levels == 2 else (2 * h0 / 3 - 2 * np.abs(isi).sum())
        best = max(best, eye)
    return best

# ---------------------------------------------------------------- data + eye
def prbs(n, order=15, seed=0x5A5A):
    st, out = seed, np.empty(n, int)
    for i in range(n):
        b = ((st >> 14) ^ (st >> 13)) & 1
        out[i] = st & 1
        st = ((st << 1) | b) & 0x7FFF
    return out

def symbols(n, levels):
    bits = prbs(n * (1 if levels == 2 else 2))
    if levels == 2:
        return bits * 2.0 - 1.0
    gray = {(0, 0): -3, (0, 1): -1, (1, 1): 1, (1, 0): 3}
    return np.array([gray[(a, b)] for a, b in bits.reshape(-1, 2)]) / 3.0

SYM_CACHE = {}

def simulate_eye(p, levels, dfe_taps):
    """Time-domain eye. Ideal DFE subtracts h1..hN (sampled at the main cursor) using past symbols."""
    k0_full = main_cursor(p)
    lo = max(k0_full - 8 * SPS, 0)
    p_win = p[lo: k0_full + 40 * SPS]             # keep only the part of the pulse that matters
    k0 = k0_full - lo
    sym = SYM_CACHE.setdefault(levels, symbols(N_SYM, levels))
    x = np.zeros(N_SYM * SPS); x[::SPS] = sym
    n = len(x) + len(p_win)
    nf = 1 << int(np.ceil(np.log2(n)))
    wave = np.fft.irfft(np.fft.rfft(x, nf) * np.fft.rfft(p_win, nf), nf)[:n]   # fast FFT convolution
    first, n_ui = 200, N_SYM - 400
    idx = k0 - SPS + (first + np.arange(n_ui))[:, None] * SPS + np.arange(2 * SPS)[None, :]
    rows = wave[idx]
    if dfe_taps:
        h = p_win[k0 + SPS * np.arange(1, dfe_taps + 1)]
        k = first + np.arange(n_ui)
        fb = sum(h[i] * sym[k - (i + 1)] for i in range(dfe_taps))
        rows = rows - fb[:, None]                 # feedback held for the whole UI (centre UI)
    s_rows = sym[first:first + n_ui]
    lv = np.unique(s_rows)
    open_ph = np.array([min(rows[s_rows == lv[i + 1], ph].min() - rows[s_rows == lv[i], ph].max()
                            for i in range(len(lv) - 1)) for ph in range(2 * SPS)])
    c = open_ph[SPS // 2: SPS + SPS // 2 + 1]; mid = SPS // 2
    l = r = mid
    while l > 0 and c[l - 1] > 0: l -= 1
    while r < len(c) - 1 and c[r + 1] > 0: r += 1
    width = (r - l + 1) / SPS if c[mid] > 0 else 0.0
    return rows, c.max(), width

# ---------------------------------------------------------------- optimise
FFE_GRID = [(cm1, c1) for cm1 in np.arange(0, -0.301, -0.025) for c1 in np.arange(0, -0.401, -0.025)]

def best_ffe(p, levels, dfe):
    best = (-1e9, (0, 1, 0))
    for cm1, c1 in FFE_GRID:
        taps = (cm1, 1 - abs(cm1) - abs(c1), c1)   # |c-1|+|c0|+|c1| = 1 (TX peak-swing limit)
        e = pda_eye(apply_ffe(p, taps), levels, dfe)
        if e > best[0]:
            best = (e, taps)
    return best

def ranked_ffe(p, levels, dfe, top=4):
    """Fast FFE search on sampled cursors (no full-waveform convolutions)."""
    k0 = main_cursor(p)
    offs = np.arange(-SPS // 4, SPS // 4 + 1, 2)
    npre, npost = 4, 13
    # C[d, k]: pulse sampled at phase offset d, cursor k = -npre..npost
    C = np.array([p[k0 + d + SPS * np.arange(-npre, npost + 1)] for d in offs])
    sc = []
    for cm1, c1 in FFE_GRID:
        c0 = 1 - abs(cm1) - abs(c1)               # |c-1|+|c0|+|c1| = 1 (TX peak-swing limit)
        E = c0 * C[:, 1:-1] + cm1 * C[:, 2:] + c1 * C[:, :-2]   # FFE on the cursor vectors
        h0 = E[:, npre - 1]
        isi = np.abs(np.concatenate([E[:, :npre - 1], E[:, npre + dfe:]], axis=1)).sum(axis=1)
        eye = 2 * h0 - 2 * isi if levels == 2 else 2 * h0 / 3 - 2 * isi
        sc.append((eye.max(), (cm1, c0, c1)))
    sc.sort(key=lambda x: -x[0])
    return sc[:top]

def pick(cands, levels, dfe):
    """Final choice by time-domain eye height among the best peak-distortion candidates."""
    best = None
    for pk, taps, p in cands:
        _, eh, ew = simulate_eye(p, levels, dfe)
        if best is None or eh > best[0]:
            best = (eh, p, taps, pk)
    return best[1], best[2], best[3]

def optimise_case(label, baud, lv):
    """Return {stage: (equalised pulse, FFE taps, CTLE peaking dB, DFE taps)}."""
    ui, fn = 1 / baud, baud / 2
    p0 = pulse_response(H_CH, ui)
    pc_cache = {pk: pulse_response(H_CH * ctle(f, fn, pk), ui) for pk in np.arange(0, 15.1, 1.0)}
    settings = {"No EQ": (p0, (0, 1, 0), 0, 0)}
    # FFE only
    c = [(0, t, apply_ffe(p0, t)) for _, t in ranked_ffe(p0, lv, 0)] + [(0, (0, 1, 0), p0)]
    p, t, pk = pick(c, lv, 0); settings["FFE"] = (p, t, 0, 0)
    prev = [(0, t, p)]
    for stage, dfe in [("FFE + CTLE", 0), ("FFE + CTLE + DFE", N_DFE)]:
        allc = []
        for pk, pc in pc_cache.items():
            allc += [(e, pk, t, apply_ffe(pc, t)) for e, t in ranked_ffe(pc, lv, dfe, top=2)]
        allc.sort(key=lambda x: -x[0])
        c = [(pk, t, p) for _, pk, t, p in allc[:6]] + prev    # previous stage is always a candidate
        p, t, pk = pick(c, lv, dfe)
        settings[stage] = (p, t, pk, dfe)
        prev = [(pk, t, p)]
    return settings


if __name__ == "__main__":
  results, eyes, pulses, ctle_used = [], {}, {}, {}
  for label, baud, lv in CASES:
    settings = optimise_case(label, baud, lv)
    for stage in STAGES:
      p, taps, pk, dfe = settings[stage]
      rows, eh, ew = simulate_eye(p, lv, dfe)
      eyes[(label, stage)] = (rows, eh, ew)
      results.append((label, stage, taps, pk, dfe, eh * 1e3, ew))
    pulses[label] = {st: settings[st][0] for st in STAGES}
    ctle_used[label] = (baud / 2, settings["FFE + CTLE + DFE"][2])

  # ---------------------------------------------------------------- report
  print(f"{'Case':10s} {'Stage':18s} {'FFE taps (c-1, c0, c1)':>24s} {'CTLE dB':>8s} {'DFE':>4s} {'eye H mV':>9s} {'eye W UI':>9s}")
  for lab, st, t, pk, dfe, eh, ew in results:
      print(f"{lab:10s} {st:18s} ({t[0]:6.3f},{t[1]:6.3f},{t[2]:6.3f}) {pk:8.0f} {dfe:4d} {eh:9.0f} {ew:9.2f}")

  # eyes grid: rows = stages, cols = cases
  fig, ax = plt.subplots(len(STAGES), len(CASES), figsize=(14, 13))
  for j, (label, _, _) in enumerate(CASES):
      for i, st in enumerate(STAGES):
          rows, eh, ew = eyes[(label, st)]
          a = ax[i, j]
          a.hist2d(np.tile(np.linspace(0, 2, 2 * SPS), rows.shape[0]), rows.ravel() * 1e3,
                   bins=[2 * SPS, 150], cmap="inferno", cmin=1)
          a.set_title(f"{label} – {st}\nH = {max(eh*1e3, 0):.0f} mV,  W = {ew:.2f} UI", fontsize=9)
          a.tick_params(labelsize=7)
          if j == 0: a.set_ylabel("mV")
          if i == len(STAGES) - 1: a.set_xlabel("UI")
  plt.tight_layout(); plt.savefig("results/python/phase3_eyes.png", dpi=130)

  # pulse responses per stage
  fig, ax = plt.subplots(1, len(CASES), figsize=(15, 4.2))
  for a, (label, baud, _) in zip(ax, CASES):
      for st in STAGES:
          p = pulses[label][st]; k0 = main_cursor(p)
          tt = (np.arange(len(p)) - k0) / SPS; m = (tt > -3) & (tt < 8)
          a.plot(tt[m], p[m] * 1e3, label=st, lw=1.3)
      a.axhline(0, c="gray", lw=.6)
      for k in range(-3, 9): a.axvline(k, c="gray", lw=.3, ls=":")
      a.set(title=f"Pulse response – {label}", xlabel="UI from main cursor", ylabel="mV")
      a.grid(alpha=.2); a.legend(fontsize=7)
  plt.tight_layout(); plt.savefig("results/python/phase3_pulses.png", dpi=130)

  # CTLE frequency response for each case
  fig, ax = plt.subplots(figsize=(8, 4.5))
  fg = f / 1e9
  ax.plot(fg, 20 * np.log10(np.abs(H_CH)), "k", lw=2, label="Channel SDD21")
  for label, (fn, pk) in ctle_used.items():
      Hc = ctle(f, fn, pk)
      ax.plot(fg, 20 * np.log10(np.abs(Hc)), "--", label=f"CTLE {label} ({pk:.0f} dB)")
      ax.plot(fg, 20 * np.log10(np.abs(H_CH * Hc)), label=f"Channel × CTLE {label}")
  ax.set(xlabel="GHz", ylabel="dB", ylim=(-30, 3), title="CTLE shaping")
  ax.grid(alpha=.3); ax.legend(fontsize=7)
  plt.tight_layout(); plt.savefig("results/python/phase3_ctle.png", dpi=130)
  print("Saved phase3_eyes.png, phase3_pulses.png, phase3_ctle.png")
