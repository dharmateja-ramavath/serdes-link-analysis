"""
Phase 6 - Edge-rate study (supports the Ansys Nexxim comparison)

Question : why do Python and Nexxim agree without equalization, but drift
           apart once TX FFE is added, especially at 112G?
Idea     : if Nexxim's channel model carries less high-frequency content,
           removing high frequencies from the Python model should reproduce
           Nexxim's behaviour. This is done by giving the transmitter a
           Gaussian edge with a given 10-90 % rise time.

For each edge rate the script reports the PAM4 worst-eye height with
  (a) the original ideal-edge FFE taps, and
  (b) FFE taps re-optimized for that edge.

Usage    : python phase6_edge_rate_study.py   (needs phase3_equalization.py and data/PCB3.s4p)
Output   : printed table + results/python/phase6_edge_rate.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import phase3_equalization as eq

EDGES_PS = [0, 5, 10, 15, 20]          # 10-90 % rise time of the TX edge
CASES = [                              # label, baud, original (ideal-edge) FFE taps
    ("56G PAM4", 28e9, (0.0, 0.825, -0.175)),
    ("112G PAM4", 56e9, (-0.05, 0.675, -0.275)),
]
NEXXIM_MV = {"56G PAM4": 117.0, "112G PAM4": 19.1}   # Nexxim Quick Eye, TX FFE only


def edge_filter(fr, tr):
    """Gaussian edge: H(f) = exp(-(2*pi*f*sigma)^2 / 2), sigma = tr / 2.563 (10-90 %)."""
    if tr == 0:
        return np.ones_like(fr)
    sigma = tr / 2.563
    return np.exp(-0.5 * (2 * np.pi * fr * sigma) ** 2)


def worst_eye(p, taps):
    _, h, w = eq.simulate_eye(eq.apply_ffe(p, taps), 4, 0)
    return h, w


results = {}
for label, baud, taps0 in CASES:
    print(f"\n{label}  (Nexxim FFE-only result: {NEXXIM_MV[label]} mV)")
    print(f"{'edge':>6} | {'original taps':>14} | re-optimized taps (c-1, c0, c1) -> eye")
    rows = []
    for tr in EDGES_PS:
        p = eq.pulse_response(eq.H_CH * edge_filter(eq.f, tr * 1e-12), 1 / baud)
        h0, _ = worst_eye(p, taps0)
        best = None
        for _, t in eq.ranked_ffe(p, 4, 0, top=4):
            h, w = worst_eye(p, t)
            if best is None or h > best[0]:
                best = (h, w, t)
        h1, w1, t1 = best
        rows.append((tr, h0 * 1e3, h1 * 1e3))
        print(f"{tr:4d}ps | {h0*1e3:9.1f} mV   | ({t1[0]:+.3f}, {t1[1]:.3f}, {t1[2]:+.3f}) -> "
              f"{h1*1e3:6.1f} mV, {w1:.2f} UI")
    results[label] = rows

# ------------------------------------------------------------------ plot
colors = {"56G PAM4": "#2563eb", "112G PAM4": "#ea580c"}
fig, ax = plt.subplots(figsize=(9, 4.8))
for label, rows in results.items():
    tr = [r[0] for r in rows]
    ax.plot(tr, [r[1] for r in rows], "--o", color=colors[label], lw=2, ms=5, label=f"{label}, original taps")
    ax.plot(tr, [r[2] for r in rows], "-o", color=colors[label], lw=2, ms=5, label=f"{label}, re-optimized taps")
    ax.axhline(NEXXIM_MV[label], color=colors[label], lw=0.9, ls=":")
    ax.text(0.2, NEXXIM_MV[label] + 4, f"Nexxim {label} ({NEXXIM_MV[label]:g} mV)", color=colors[label], fontsize=9)
ax.axhline(0, color="#6b7280", lw=1)
ax.set_xlabel("Transmitter edge, 10-90 % rise time (ps)")
ax.set_ylabel("Worst-eye height (mV)")
ax.set_title("Python PAM4 eye vs transmitter edge rate (TX FFE only)")
ax.grid(alpha=0.3)
ax.legend(loc="lower left", fontsize=9, handlelength=3.5)
plt.tight_layout()
plt.savefig("results/python/phase6_edge_rate.png", dpi=140)
print("\nSaved results/python/phase6_edge_rate.png")
