"""
Phase 5 - Margin sweeps (shmoo) for the PCB3 channel
---------------------------------------------------
Part A: data-rate sweep  - eye height/width vs data rate, NRZ and PAM4, No EQ vs best EQ
Part B: shmoo heatmaps   - 56G and 112G PAM4: CTLE peaking x FFE post-tap (DFE on, 3 taps)
         colour = eye height; black contour = PASS region (enough margin for noise + jitter)
Outputs: phase5_rate_sweep.png, phase5_shmoo.png + printed summary

Usage  : python phase5_margin_sweep.py  (needs phase3_equalization.py, phase4_jitter_ber.py, PCB3.s4p)
"""
import math
import numpy as np
import matplotlib.pyplot as plt
import phase3_equalization as p3
from phase4_jitter_ber import q_inv, SIGMA_N, RJ_PS, DJ_PS, TARGET

def required_margin(levels, baud):
    """Minimum eye height (V) and width (UI) needed for the BER target with the Phase 4 noise/jitter."""
    q = q_inv(TARGET[levels] * math.log2(levels))
    ui_ps = 1e12 / baud
    return 2 * q * SIGMA_N, (DJ_PS + 2 * q * RJ_PS) / ui_ps

# ------------------------------------------------------------- Part A: data-rate sweep
sweeps = {"NRZ": (2, np.arange(10, 61, 5)), "PAM4": (4, np.arange(20, 129, 12))}   # Gb/s
rate_res = {}
for mod, (lv, rates) in sweeps.items():
    res = []
    for r in rates:
        baud = r * 1e9 / math.log2(lv)
        st = p3.optimise_case(f"{r}G {mod}", baud, lv)
        p0 = st["No EQ"][0]
        pb, taps, pk, dfe = st["FFE + CTLE + DFE"]
        _, h0, w0 = p3.simulate_eye(p0, lv, 0)
        _, hb, wb = p3.simulate_eye(pb, lv, dfe)
        hreq, wreq = required_margin(lv, baud)
        res.append((r, h0, w0, hb, wb, hreq, wreq, taps, pk))
        print(f"{r:4d}G {mod:4s}  no-EQ H={max(h0,0)*1e3:5.0f} mV W={w0:.2f} | best EQ H={max(hb,0)*1e3:5.0f} mV "
              f"W={wb:.2f}  (FFE {taps[0]:.2f}/{taps[1]:.2f}/{taps[2]:.2f}, CTLE {pk:.0f} dB)  "
              f"{'PASS' if hb > hreq and wb > wreq else 'FAIL'}")
    rate_res[mod] = res

fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
for i, (mod, res) in enumerate(rate_res.items()):
    r = np.array([x[0] for x in res])
    for j, (key, lab, ls) in enumerate([(1, "No EQ", "--"), (3, "Best EQ (FFE+CTLE+DFE)", "-")]):
        ax[0].plot(r, np.maximum([x[key] * 1e3 for x in res], 0), ls=ls, c=f"C{i}", marker="o", ms=4, label=f"{mod} – {lab}")
        ax[1].plot(r, [x[key + 1] for x in res], ls=ls, c=f"C{i}", marker="o", ms=4, label=f"{mod} – {lab}")
    ax[0].plot(r, [x[5] * 1e3 for x in res], ":", c=f"C{i}", label=f"{mod} – required height")
    ax[1].plot(r, [x[6] for x in res], ":", c=f"C{i}", label=f"{mod} – required width")
ax[0].set(xlabel="Data rate (Gb/s)", ylabel="Eye height (mV)", title="Eye height vs data rate")
ax[1].set(xlabel="Data rate (Gb/s)", ylabel="Eye width (UI)", title="Eye width vs data rate")
for a in ax: a.grid(alpha=.3); a.legend(fontsize=7)
plt.tight_layout(); plt.savefig("results/python/phase5_rate_sweep.png", dpi=130)

# ------------------------------------------------------------- Part B: shmoo heatmaps
pk_axis = np.arange(0, 12.1, 1.0)
c1_axis = np.arange(0, -0.401, -0.025)
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
for a, (label, baud) in zip(ax, [("56G PAM4", 28e9), ("112G PAM4", 56e9)]):
    lv, ui, fn = 4, 1 / baud, baud / 2
    cm1 = p3.optimise_case(label, baud, lv)["FFE + CTLE + DFE"][1][0]   # keep best pre-tap fixed
    hreq, wreq = required_margin(lv, baud)
    Hm = np.zeros((len(c1_axis), len(pk_axis))); Pm = np.zeros_like(Hm)
    for j, pk in enumerate(pk_axis):
        pc = p3.pulse_response(p3.H_CH * p3.ctle(p3.f, fn, pk), ui)
        for i, c1 in enumerate(c1_axis):
            taps = (cm1, 1 - abs(cm1) - abs(c1), c1)
            _, h, w = p3.simulate_eye(p3.apply_ffe(pc, taps), lv, p3.N_DFE)
            Hm[i, j] = max(h, 0) * 1e3
            Pm[i, j] = 1.0 if (h > hreq and w > wreq) else 0.0
    im = a.imshow(Hm, origin="upper", aspect="auto", cmap="viridis",
                  extent=[pk_axis[0] - .5, pk_axis[-1] + .5, c1_axis[-1] - .0125, c1_axis[0] + .0125])
    a.contour(pk_axis, c1_axis, Pm, levels=[0.5], colors="k", linewidths=2)
    bi = np.unravel_index(np.argmax(Hm), Hm.shape)
    a.plot(pk_axis[bi[1]], c1_axis[bi[0]], "r*", ms=14)
    a.set(xlabel="CTLE peaking (dB)", ylabel="FFE post-tap c1",
          title=f"Shmoo – {label} (DFE 3 taps, pre-tap {cm1:.3f})\n"
                f"★ best {Hm[bi]:.0f} mV  |  PASS on {Pm.mean()*100:.0f}% of grid (black line = boundary)")
    plt.colorbar(im, ax=a, label="eye height (mV)")
    print(f"{label}: pass region covers {Pm.mean()*100:.0f}% of the settings grid; best eye {Hm[bi]:.0f} mV "
          f"at CTLE {pk_axis[bi[1]]:.0f} dB, c1 {c1_axis[bi[0]]:.3f}")
plt.tight_layout(); plt.savefig("results/python/phase5_shmoo.png", dpi=130)
print("Saved phase5_rate_sweep.png and phase5_shmoo.png")
