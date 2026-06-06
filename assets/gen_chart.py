#!/usr/bin/env python3
"""Generate performance progression chart for README."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

kernels = [
    ("01 Naive",            5.67),
    ("02 Coalescing",       5.67),
    ("03 Shared Mem",       8.15),
    ("04 1D Tiling",       21.68),
    ("05 2D Tiling",       33.92),
    ("06 Vectorized",      28.22),
    ("07 WMMA TC",         48.92),
    ("08 PTX mma",         45.29),
    ("09 WMMA Dbuf",       68.09),
    ("10 Large Tile",     100.47),
    ("cuBLAS SGEMM",       58.55),
    ("cuBLAS HGEMM",      295.65),
]

labels = [k[0] for k in kernels]
values = [k[1] for k in kernels]

cublas_hgemm = 295.65

colors = []
for label in labels:
    if "cuBLAS" in label:
        colors.append("#2d2d2d")
    elif "WMMA" in label or "PTX" in label or "Large" in label:
        colors.append("#76b900")
    else:
        colors.append("#4a90d9")

fig, ax = plt.subplots(figsize=(12, 6.5))
bars = ax.barh(labels, values, color=colors, height=0.7, edgecolor='white', linewidth=0.5)

for bar, val in zip(bars, values):
    w = bar.get_width()
    pct_hgemm = val / cublas_hgemm * 100
    ax.text(w + 2, bar.get_y() + bar.get_height()/2,
            f'{val:.1f} TFLOPS ({pct_hgemm:.0f}% of HGEMM)',
            va='center', ha='left', fontsize=8.5, fontweight='bold',
            color='#333333')

ax.set_xlabel('TFLOPS', fontsize=11, fontweight='bold')
ax.set_title('Performance Progression: Naive Matmul to Tensor Cores\n'
             'NVIDIA RTX PRO 6000 Blackwell (sm_120)  |  4096x4096x4096',
             fontsize=12, fontweight='bold', pad=12)
ax.set_xlim(0, 380)
ax.invert_yaxis()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='y', labelsize=9.5)
ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
ax.grid(axis='x', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('assets/performance.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Saved assets/performance.png")
