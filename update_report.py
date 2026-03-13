"""
Parse results JSON files and update macros.tex (shared by report.tex and slides.tex).
Run after eval.py and ablate.py have completed.
"""

import json, os, re

RESULT_DIR = "/data/koe/ECE285-Final/results"
MACROS     = "/data/koe/ECE285-Final/macros.tex"


def load(path):
    if not os.path.exists(path):
        print(f"  [SKIP] {path} not found")
        return None
    with open(path) as f:
        return json.load(f)


def subst(tex, macro, value):
    pattern = rf"(\\newcommand{{\\{macro}}}{{)[^}}]*(}})"
    repl    = rf"\g<1>{value}\g<2>"
    new, n  = re.subn(pattern, repl, tex)
    if n == 0:
        print(f"  [WARN] macro \\{macro} not found in macros.tex")
    return new


def main():
    with open(MACROS) as f:
        tex = f.read()

    # ── Default (full model) ──────────────────────────────────────────────
    d = load(f"{RESULT_DIR}/eval_default.json")
    if d:
        o = d["one_step"]
        r = d["rollout"]
        tex = subst(tex, "MAINPSNR",   f"{o['psnr']:.2f}")
        tex = subst(tex, "MAINSSIM",   f"{o['ssim']:.4f}")
        tex = subst(tex, "MAINMSE",    f"{o['mse']:.4f}")
        tex = subst(tex, "MAINEMSE",   f"{o['embed_mse']:.4f}")
        tex = subst(tex, "MAINDRIFTV", f"{r['drift_5']:.5f}")
        tex = subst(tex, "MAINDRIFTX", f"{r['drift_10']:.5f}")
        tex = subst(tex, "MAINDRIFTXX",f"{r['drift_20']:.5f}")
        stage0_psnr = 31.30
        diff = o["psnr"] - stage0_psnr
        tex = subst(tex, "MAINPSNRDIFF", f"{diff:.2f}")
        print(f"  Default PSNR={o['psnr']:.2f} SSIM={o['ssim']:.4f} "
              f"D5={r['drift_5']:.5f} D10={r['drift_10']:.5f}")

    # ── Ablation results ──────────────────────────────────────────────────
    ab = load(f"{RESULT_DIR}/ablation_results.json")
    if ab:
        def get_m(tag):
            if tag not in ab: return None
            return ab[tag].get("metrics") or ab[tag]

        for tag, psnr_k, ssim_k, dv_k, dx_k in [
            ("no_stage0", "NOSTPSNR", "NOSTSSIM", "NOSTDRIFTV", "NOSTDRIFTX"),
            ("lam0p0",    "LAMZPSNR", "LAMZSSIM", "LAMZDRIFTV", "LAMZDRIFTX"),
        ]:
            m = get_m(tag)
            if m:
                o, r = m["one_step"], m["rollout"]
                tex = subst(tex, psnr_k, f"{o['psnr']:.2f}")
                tex = subst(tex, ssim_k, f"{o['ssim']:.4f}")
                tex = subst(tex, dv_k,   f"{r['drift_5']:.5f}")
                tex = subst(tex, dx_k,   f"{r['drift_10']:.5f}")
                print(f"  {tag} PSNR={o['psnr']:.2f}")

        for tag, psnr_k, ssim_k in [
            ("lam0p01",    "LAMINEPSNR",    "LAMINESSIM"),
            ("lam1p0",     "LAMONEPSNR",    "LAMONESSIM"),
            ("K2",         "KTWOPSNR",      "KTWOSSIM"),
            ("K4",         "KFOURPSNR",     "KFOURSSIM"),
            ("K16",        "KSIXTEENPSNR",  "KSIXTEENSSIM"),
            ("dec_partial","DECPRTPSNR",    "DECPRTSSIM"),
        ]:
            m = get_m(tag)
            if m:
                o = m["one_step"]
                tex = subst(tex, psnr_k, f"{o['psnr']:.2f}")
                tex = subst(tex, ssim_k, f"{o['ssim']:.4f}")
                print(f"  {tag} PSNR={o['psnr']:.2f}")

    # ── Planning evaluation ───────────────────────────────────────────────────
    pl = load(f"{RESULT_DIR}/planning_eval.json")
    if pl:
        def fmt_sr(r): return f"{r['success_rate']*100:.0f}\\%"
        def fmt_st(r):
            s = r.get("avg_steps_success")
            return f"{s:.1f}" if s else "--"
        tex = subst(tex, "PLANVLMSR",     fmt_sr(pl["vlm_only"]))
        tex = subst(tex, "PLANVLMWMSR",   fmt_sr(pl["vlm_wm"]))
        tex = subst(tex, "PLANBFSSR",     fmt_sr(pl["oracle_bfs"]))
        tex = subst(tex, "PLANVLMSTEPS",  fmt_st(pl["vlm_only"]))
        tex = subst(tex, "PLANVLMWMSTEPS",fmt_st(pl["vlm_wm"]))
        tex = subst(tex, "PLANBFSSTEPS",  fmt_st(pl["oracle_bfs"]))
        print(f"  VLM-only SR={pl['vlm_only']['success_rate']*100:.1f}%  "
              f"VLM+WM SR={pl['vlm_wm']['success_rate']*100:.1f}%  "
              f"BFS SR={pl['oracle_bfs']['success_rate']*100:.1f}%")

    with open(MACROS, "w") as f:
        f.write(tex)
    print(f"\nUpdated {MACROS}")


if __name__ == "__main__":
    main()
