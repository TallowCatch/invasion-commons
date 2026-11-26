"""Render the deck's formulas as PNGs with LaTeX (pdflatex + sips), all at the same type size.

  python3 notes/slides/2026-11-16_weekly_update/make_formulas.py
Writes formulas/<name>.png and formulas/sizes.json (natural width/height in points, for placing at one scale).
"""
import json
import subprocess
import tempfile
from pathlib import Path

OUT = Path(__file__).parent / "formulas"
F = {
    "regrow": r"p(n)=0.025,\ 0.005,\ 0.001,\ 0 \quad\text{for}\quad n\ge 3,\ 2,\ 1,\ 0",
    "rule": r"v_{i,t}=\mathbf{1}\big[\,i\text{ takes an apple at }t\,\big]\cdot\mathbf{1}\big[\,n_{i,t}<3\,\big]",
    "detect": r"\Pr(\text{caught}\mid v{=}1)=1-\alpha,\qquad \Pr(\text{caught}\mid v{=}0)=\beta",
    "reward": r"\tilde r_{i,t}=r_{i,t}-7P\cdot\mathrm{caught}_{i,t},\qquad |C_t|\le k",
    "agent": r"\max_{\pi_i}\ \mathbb{E}\Big[\sum_{t}\gamma^{t}\,\tilde r_{i,t}\Big]",
    "design": r"\max_{\theta_O}\ W(\theta_O,\pi^*)-\lambda\,\mathrm{Cost}(\theta_O)\quad\text{s.t.}\quad"
              r"\pi^*\in\arg\max_{\pi}\,U(\pi;\theta_O),\quad \mathrm{Cost}(\theta_O)\le B",
    "cost": r"\mathrm{Cost}(\theta_O)=\sum_{t}\sum_{i\in C_t}\big(c_{\mathrm{check}}+c_{\mathrm{hist}}\,w\big)",
    "eta": r"\eta(\theta_O)=\frac{W(\theta_O)-W_{\mathrm{none}}}{W_{\mathrm{all}}-W_{\mathrm{none}}}",
    "worst": r"W_{\mathrm{worst}}(\theta_O)=\min_{\pi\in\Pi_{\mathrm{adapt}}}W(\theta_O,\pi)",
    "deter": r"\frac{k}{N}\,(1-\alpha)\,P\ \ \ge\ \ g",
    "honest": r"\text{honest loss per step}=\frac{k}{N}\,\beta\,P",
    "catch": r"\text{catch rate}=\mathbb{E}\big[\,q_i(t)\mid v_{i,t}=1\,\big]",
    "mix": r"\text{random: }\ q_i=\tfrac{k}{N}\qquad\quad \text{mixed targeting: }\ \max_{q}\ \min_{i}\ \big(q_i(1-\alpha)P-g_i\big)"
           r"\ \ \text{s.t.}\ \textstyle\sum_i q_i=k",
}
TEX = r"""\documentclass[border=3pt]{standalone}
\usepackage{amsmath,amssymb,xcolor}
\begin{document}
\color[HTML]{062746}$\displaystyle %s$
\end{document}
"""


def main():
    OUT.mkdir(exist_ok=True)
    sizes = {}
    for name, body in F.items():
        with tempfile.TemporaryDirectory() as d:
            tex = Path(d) / "f.tex"
            tex.write_text(TEX % body)
            subprocess.run(["pdflatex", "-interaction=nonstopmode", "f.tex"], cwd=d, check=True, capture_output=True)
            pdf = Path(d) / "f.pdf"
            box = subprocess.run(["sips", "-g", "pixelWidth", "-g", "pixelHeight", str(pdf)], capture_output=True,
                                 text=True).stdout.split()
            w, h = float(box[box.index("pixelWidth:") + 1]), float(box[box.index("pixelHeight:") + 1])  # points
            png = OUT / f"{name}.png"
            subprocess.run(["sips", "-s", "format", "png", "-Z", str(int(max(w, h) * 300 / 72)), str(pdf), "--out",
                            str(png)], check=True, capture_output=True)
            sizes[name] = dict(w_pt=w, h_pt=h)
    (OUT / "sizes.json").write_text(json.dumps(sizes, indent=1))
    print(json.dumps(sizes))


if __name__ == "__main__":
    main()
