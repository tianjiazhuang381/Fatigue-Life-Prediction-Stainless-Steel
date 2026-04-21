import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data_preprocessing import load_and_preprocess

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Liberation Serif', 'DejaVu Serif'],
    'mathtext.fontset': 'dejavuserif',
    'font.size': 10,
})

df, _ = load_and_preprocess('fatigue_dataset.xlsx')
features = ['σys', 'σuts', 'El', 'σa', 'σfl', 'σa/σuts', 'σa/σys', 'G1', 'logN']
labels = [r'σys', r'σuts', r'El', r'σa', r'σfl', r'σa/σuts', r'σa/σys', r'G1', r'logN']
corr = df[features].corr()
fig, ax = plt.subplots(figsize=(5, 4.2))
im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
ax.set_yticklabels(labels, fontsize=9)

for i in range(len(features)):
    for j in range(len(features)):
        val = corr.values[i, j]
        color = 'white' if abs(val) > 0.6 else 'black'
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=6.5, color=color)

fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
fig.tight_layout()
fig.savefig('Fig2_correlation.tiff', dpi=1200, bbox_inches='tight', facecolor='white')
plt.close(fig)
print("Saved: Fig2_correlation.tiff")