"""
Phase 2 - Pulse response and eye diagrams for the PCB3 differential channel
-------------------------------------------------------------------------
Channel : PCB3.s4p (SIwave, 0-40 GHz, 150 mm 100-ohm diff microstrip)
Cases   : 10G NRZ, 25G NRZ, 56G PAM4 (28 GBd), 112G PAM4 (56 GBd)
Outputs : phase2_pulse_response.png, phase2_eyes.png, printed cursor/eye table

Usage   : python phase2_pulse_eye.py   (run from the repository root)
Requires: numpy, matplotlib   (scikit-rf optional)
"""
import numpy as np
import matplotlib.pyplot as plt

FILE = "data/PCB3.s4p"
# Touchstone port order from SIwave: 1=C_N_J1  2=C_N_J2  3=C_P_J1  4=C_P_J2  (0-based below)
P_IN, N_IN, P_OUT, N_OUT = 2, 0, 3, 1
SPS = 32                      # samples per UI
N_SYM = 4000                  # symbols per simulation
TX_SWING = 1.0                # differential peak-to-peak swing at TX (V)

CASES = [  # (label, baud rate, levels)
    ("10G NRZ",    10e9, 2),
    ("25G NRZ",    25e9, 2),
    ("56G PAM4",   28e9, 4),
    ("112G PAM4",  56e9, 4),
]

# ------------------------------------------------------------------ channel
def load_touchstone(fn):
    try:
        import skrf as rf
        n = rf.Network(fn); return n.f, n.s
    except ImportError:
        vals = []
        for line in open(fn, encoding="latin1"):
            line = line.split("!")[0].strip()
            if line and not line.startswith("#"):
                vals += line.split()
        N = int(fn.lower().split(".s")[-1].rstrip("p"))
        v = np.array(vals, float).reshape(-1, 1 + 2 * N * N)
        s = (v[:, 1::2] * np.exp(1j * np.deg2rad(v[:, 2::2]))).reshape(-1, N, N)
        return v[:, 0], s

def sdd21(s):
    """Differential insertion loss from single-ended 4-port."""
    return 0.5 * (s[:, P_OUT, P_IN] - s[:, P_OUT, N_IN] - s[:, N_OUT, P_IN] + s[:, N_OUT, N_IN])

f, s = load_touchstone(FILE)
H = sdd21(s)

def impulse_response(f, H, dt, n_fft):
    """Resample H onto an FFT grid with step 1/(n_fft*dt); taper above the data range."""
    fg = np.fft.rfftfreq(n_fft, dt)
    mag = np.interp(fg, f, np.abs(H), right=0.0)
    ph = np.interp(fg, f, np.unwrap(np.angle(H)))
    # extrapolate phase linearly (pure delay) beyond the measured band
    slope = (ph[fg <= f[-1]][-1] - ph[fg <= f[-1]][-50]) / (fg[fg <= f[-1]][-1] - fg[fg <= f[-1]][-50])
    hi = fg > f[-1]
    ph[hi] = ph[~hi][-1] + slope * (fg[hi] - fg[~hi][-1])
    # raised-cosine taper 0.8*fmax -> fmax so the band edge does not ring
    fmax = f[-1]; t0 = 0.8 * fmax
    w = np.ones_like(fg)
    band = (fg > t0) & (fg <= fmax)
    w[band] = 0.5 * (1 + np.cos(np.pi * (fg[band] - t0) / (fmax - t0)))
    w[fg > fmax] = 0
    return np.fft.irfft(mag * w * np.exp(1j * ph), n_fft)      # impulse samples (sum = H(0))

# ------------------------------------------------------------------ helpers
def prbs(n, order=15, seed=0x5A5A):
    """PRBS-15 bit sequence (x^15 + x^14 + 1)."""
    state, out = seed & ((1 << order) - 1), np.empty(n, int)
    for i in range(n):
        b = ((state >> 14) ^ (state >> 13)) & 1
        out[i] = state & 1
        state = ((state << 1) | b) & ((1 << order) - 1)
    return out

def symbols(n, levels):
    bits = prbs(n * (1 if levels == 2 else 2))
    if levels == 2:
        return bits * 2.0 - 1.0                                   # -1, +1
    gray = {(0, 0): -3, (0, 1): -1, (1, 1): 1, (1, 0): 3}         # Gray-coded PAM4
    return np.array([gray[(a, b)] for a, b in bits.reshape(-1, 2)]) / 3.0

def eye_rows(wave, k0, sps, n_ui, first):
    """Rows of 2 UI centred on each symbol's main cursor (row i <-> symbol first+i)."""
    idx = k0 - sps + (first + np.arange(n_ui))[:, None] * sps + np.arange(2 * sps)[None, :]
    return wave[idx]

def eye_metrics(rows, sym, levels):
    """Worst-eye height at best phase and width (UI), using the known transmitted symbols."""
    lv = np.unique(sym)
    open_ph = []
    for ph in range(rows.shape[1]):
        col = rows[:, ph]
        open_ph.append(min(col[sym == lv[i + 1]].min() - col[sym == lv[i]].max()
                           for i in range(len(lv) - 1)))
    open_ph = np.array(open_ph)
    sps = rows.shape[1] // 2
    c = open_ph[sps // 2: sps + sps // 2 + 1]                    # search around the eye centre
    # contiguous open region around the centre
    mid = sps // 2
    l = mid
    while l > 0 and c[l - 1] > 0: l -= 1
    r = mid
    while r < len(c) - 1 and c[r + 1] > 0: r += 1
    width = (r - l + 1) / sps if c[mid] > 0 else 0.0
    return c.max(), width

# ------------------------------------------------------------------ simulate
results, pulses, waves = [], {}, {}
for label, baud, lv in CASES:
    ui = 1 / baud
    dt = ui / SPS
    n_fft = int(2 ** np.ceil(np.log2(20e-9 / dt)))               # 20 ns window
    h = impulse_response(f, H, dt, n_fft)
    pulse = np.convolve(h, np.ones(SPS))[:n_fft] * (TX_SWING / 2)   # response to one +1 symbol
    above = np.where(pulse > 0.5 * pulse.max())[0]              # main cursor = centre of the
    k0 = int(round((above[0] + above[-1]) / 2))                  # pulse's half-max region
    cursors = pulse[k0 + SPS * np.arange(-2, 6)]                 # h(-2) ... h(5)
    pulses[label] = (np.arange(n_fft) * dt, pulse, k0, ui)

    sym = symbols(N_SYM, lv)
    x = np.zeros(N_SYM * SPS + n_fft); x[: N_SYM * SPS: SPS] = sym
    wave = np.convolve(x, pulse)[: N_SYM * SPS + n_fft]
    first, n_ui = 200, N_SYM - 400                               # skip start-up transient
    rows = eye_rows(wave, k0, SPS, n_ui, first)
    eh, ew = eye_metrics(rows, sym[first:first + n_ui], lv)
    waves[label] = (wave, lv, k0)
    loss_nyq = 20 * np.log10(abs(np.interp(baud / 2, f, np.abs(H))))
    results.append((label, baud / 2e9, loss_nyq, cursors / cursors[2], eh * 1e3, ew))

# ------------------------------------------------------------------ report
print(f"{'Case':10s} {'Nyq GHz':>7s} {'IL@Nyq':>7s}   cursors h-1  h0   h1    h2    h3   | eye H (mV)  eye W (UI)")
for lab, fn, il, c, eh, ew in results:
    print(f"{lab:10s} {fn:7.1f} {il:7.1f}   {c[1]:6.2f} {c[2]:4.2f} {c[3]:5.2f} {c[4]:5.2f} {c[5]:5.2f} | "
          f"{eh:9.0f}  {ew:9.2f}")

# pulse responses
fig, ax = plt.subplots(2, 2, figsize=(12, 7))
for a, (label, (t, p, k0, ui)) in zip(ax.flat, pulses.items()):
    tt = (np.arange(len(p)) - k0) * (t[1] - t[0]) / ui
    m = (tt > -3) & (tt < 8)
    a.plot(tt[m], p[m] * 1e3, lw=1.5)
    cur = np.arange(-2, 8)
    a.plot(cur, p[k0 + (cur * SPS).astype(int)] * 1e3, "o", c="C3", ms=5, label="cursor samples")
    a.axhline(0, c="gray", lw=.6)
    a.set(title=f"Pulse response – {label}", xlabel="time (UI from main cursor)", ylabel="mV")
    a.grid(alpha=.3); a.legend(fontsize=8)
plt.tight_layout(); plt.savefig("results/python/phase2_pulse_response.png", dpi=140)

# eye diagrams (2-UI density plots)
fig, ax = plt.subplots(2, 2, figsize=(12, 8))
for a, (label, (wave, lv, k0)), r in zip(ax.flat, waves.items(), results):
    n_ui = N_SYM - 400
    idx = k0 - SPS + (200 + np.arange(n_ui))[:, None] * SPS + np.arange(2 * SPS)[None, :]
    traces = wave[idx]                                           # 2-UI windows, eye centred at 1 UI
    tx = np.tile(np.linspace(0, 2, 2 * SPS), n_ui)
    a.hist2d(tx, traces.ravel() * 1e3, bins=[2 * SPS, 160], cmap="inferno", cmin=1)
    a.set(title=f"{label}  |  eye H = {max(r[4],0):.0f} mV, W = {r[5]:.2f} UI",
          xlabel="UI", ylabel="mV")
plt.tight_layout(); plt.savefig("results/python/phase2_eyes.png", dpi=140)
print("Saved phase2_pulse_response.png and phase2_eyes.png")
