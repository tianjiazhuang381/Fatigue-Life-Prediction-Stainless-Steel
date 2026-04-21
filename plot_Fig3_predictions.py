import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
import pickle

RANDOM_STATE = 42
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Liberation Serif', 'DejaVu Serif'],
    'mathtext.fontset': 'dejavuserif',
    'font.size': 10,
})
data = np.load('predictions.npz')
y_test = data['y_test']

with open('trained_models.pkl', 'rb') as f:
    saved = pickle.load(f)
from data_preprocessing import load_and_preprocess
df, _ = load_and_preprocess('fatigue_dataset.xlsx')
features_basic = ['σys', 'σuts', 'El', 'σa', 'Type_enc']
X_bas = df[features_basic].values
y_all = df['logN'].values
X_tr_b, X_te_b, y_tr, y_te = train_test_split(
    X_bas, y_all, test_size=0.2, random_state=RANDOM_STATE)
sc_b = MinMaxScaler()
X_tr_b = sc_b.fit_transform(X_tr_b)
X_te_b = sc_b.transform(X_te_b)
gbr_params = saved['models']['GBR'].get_params()
gbr_basic = GradientBoostingRegressor(**gbr_params)
gbr_basic.fit(X_tr_b, y_tr)
pred_gbr_basic = gbr_basic.predict(X_te_b)
fig = plt.figure(figsize=(7, 9.5))
gs = gridspec.GridSpec(3, 2, hspace=0.35, wspace=0.35)
lim = [2.5, 7.5]
x_line = np.linspace(lim[0], lim[1], 100)

def plot_panel(ax, y_exp, y_pred, title, color='#2171b5'):
    r2 = r2_score(y_exp, y_pred)
    mae = mean_absolute_error(y_exp, y_pred)
    ax.fill_between(x_line, x_line - np.log10(3), x_line + np.log10(3),
                    alpha=0.08, color='orange', label=r'pm3 band')
    ax.fill_between(x_line, x_line - np.log10(2), x_line + np.log10(2),
                    alpha=0.12, color='green', label=r'pm2 band')
    ax.plot(x_line, x_line, 'k--', lw=0.8, alpha=0.7)
    ax.scatter(y_exp, y_pred, s=18, c=color, alpha=0.7,
              edgecolors='white', linewidths=0.3, zorder=5)
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect('equal')
    ax.set_xlabel(r'Experimental logNf', fontsize=9)
    ax.set_ylabel(r'Predicted logNf', fontsize=9)
    ax.text(0.05, 0.95, f'R2 = {r2:.3f}\nMAE = {mae:.3f}',
            transform=ax.transAxes, fontsize=8, va='top',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85, ec='#ccc'))
    ax.set_title(title, fontsize=10, loc='left')
    ax.legend(loc='lower right', fontsize=6, framealpha=0.8)

panels = [('RF','(a) RF'), ('XGBoost','(b) XGBoost'), ('LightGBM','(c) LightGBM'), ('ANN','(d) ANN')]
for idx, (key, title) in enumerate(panels):
    ax = fig.add_subplot(gs[idx // 2, idx % 2])
    plot_panel(ax, y_test, data[key], title)
ax_e = fig.add_subplot(gs[2, 0])
plot_panel(ax_e, y_te, pred_gbr_basic, '(e) GBR without FE', color='#e6550d')
ax_f = fig.add_subplot(gs[2, 1])
plot_panel(ax_f, y_test, data['GBR'], '(f) GBR with FE')
fig.savefig('Fig3_predictions.tiff', dpi=1200, bbox_inches='tight', facecolor='white')
plt.close(fig)
print("Saved: Fig3_predictions.tiff")