import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb
import lightgbm as lgb
import warnings
import pickle

warnings.filterwarnings('ignore')

RANDOM_STATE = 42

FEATURES_BASIC = ['σys', 'σuts', 'El', 'σa', 'Type_enc']
FEATURES_ENG = ['σys', 'σuts', 'El', 'σa', 'σfl', 'σa/σuts', 'σa/σys', 'G1', 'Type_enc']
TARGET = 'logN'


def get_param_grids():
    return {
       'RF': {
            'model': RandomForestRegressor(random_state=RANDOM_STATE),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [5, 8, 10, 15],
                'min_samples_leaf': [3, 5, 10],
            }
        },
        'XGBoost': {
            'model': xgb.XGBRegressor(random_state=RANDOM_STATE, verbosity=0),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 4, 5, 6],
                'learning_rate': [0.05, 0.1, 0.2],
            }
        },
        'LightGBM': {
            'model': lgb.LGBMRegressor(random_state=RANDOM_STATE, verbose=-1),
            'params': {
                'n_estimators': [200, 300, 500],
                'max_depth': [4, 5, 7],
                'learning_rate': [0.1, 0.15, 0.2],
                'num_leaves': [20, 28, 35],
            }
        },
        'GBR': {
            'model': GradientBoostingRegressor(random_state=RANDOM_STATE),
            'params': {
                'n_estimators': [300, 500, 800],
                'max_depth': [3, 4, 5],
                'learning_rate': [0.05, 0.1, 0.15],
                'min_samples_leaf': [3, 5, 8],
            }
        },
    }


def tune_tree_models(X_train, y_train):
    param_grids = get_param_grids()
    best_models = {}

    for name, config in param_grids.items():
        print(f"\nTuning {name}...")
        gs = GridSearchCV(
            config['model'], config['params'],
            cv=10, scoring='r2', n_jobs=-1, refit=True
        )
        gs.fit(X_train, y_train)
        best_models[name] = gs.best_estimator_
        print(f"  Best params: {gs.best_params_}")
        print(f"  Best CV R²:  {gs.best_score_:.4f}")

    return best_models


def tune_ann(X_train, y_train):
    configs = [
        {'hidden_layer_sizes': (64, 32),      'learning_rate_init': 0.001},
        {'hidden_layer_sizes': (100, 50),      'learning_rate_init': 0.001},
        {'hidden_layer_sizes': (128, 64, 32),  'learning_rate_init': 0.0005},
        {'hidden_layer_sizes': (128, 64, 32),  'learning_rate_init': 0.001},
        {'hidden_layer_sizes': (256, 128),     'learning_rate_init': 0.0005},
    ]
    best_r2 = -np.inf
    best_model = None
    print("\nTuning ANN...")
    for cfg in configs:
        ann = MLPRegressor(
            **cfg, max_iter=2000, random_state=RANDOM_STATE,
            early_stopping=True, validation_fraction=0.15, alpha=0.001
        )
        scores = cross_val_score(ann, X_train, y_train, cv=10, scoring='r2')
        mean_r2 = scores.mean()
        print(f"  {cfg['hidden_layer_sizes']}, lr={cfg['learning_rate_init']} "
              f"-> CV R²={mean_r2:.4f} ± {scores.std():.4f}")
        if mean_r2 > best_r2:
            best_r2 = mean_r2
            best_cfg = cfg
    best_model = MLPRegressor(
        **best_cfg, max_iter=2000, random_state=RANDOM_STATE,
        early_stopping=True, validation_fraction=0.15, alpha=0.001
    )
    best_model.fit(X_train, y_train)
    print(f"  Selected: {best_cfg}, CV R²={best_r2:.4f}")
    return best_model

def evaluate_model(model, X_train, X_test, y_train, y_test):
    yp_train = model.predict(X_train)
    yp_test = model.predict(X_test)
    return {
        'R2_train': round(r2_score(y_train, yp_train), 3),
        'R2_test':  round(r2_score(y_test, yp_test), 3),
        'MAE_test': round(mean_absolute_error(y_test, yp_test), 3),
        'MSE_test': round(mean_squared_error(y_test, yp_test), 3),
    }


def run_training(df):
    X_eng = df[FEATURES_ENG].values
    X_bas = df[FEATURES_BASIC].values
    y = df[TARGET].values

    X_tr_e, X_te_e, y_tr, y_te = train_test_split(
        X_eng, y, test_size=0.2, random_state=RANDOM_STATE)
    X_tr_b, X_te_b, _, _ = train_test_split(
        X_bas, y, test_size=0.2, random_state=RANDOM_STATE)

    sc_e = MinMaxScaler()
    X_tr_e = sc_e.fit_transform(X_tr_e)
    X_te_e = sc_e.transform(X_te_e)

    sc_b = MinMaxScaler()
    X_tr_b = sc_b.fit_transform(X_tr_b)
    X_te_b = sc_b.transform(X_te_b)

    print("=" * 60)
    print("Hyperparameter Optimization (10-fold CV, engineered features)")
    print("=" * 60)

    best_models = tune_tree_models(X_tr_e, y_tr)
    best_models['ANN'] = tune_ann(X_tr_e, y_tr)

    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)

    results = []
    for name, model in best_models.items():

        metrics_eng = evaluate_model(model, X_tr_e, X_te_e, y_tr, y_te)
        metrics_eng.update({'Model': name, 'Features': 'Engineered'})
        results.append(metrics_eng)

        model_bas = type(model)(**model.get_params())
        model_bas.fit(X_tr_b, y_tr)
        metrics_bas = evaluate_model(model_bas, X_tr_b, X_te_b, y_tr, y_te)
        metrics_bas.update({'Model': name, 'Features': 'Basic'})
        results.append(metrics_bas)

    results_df = pd.DataFrame(results)
    results_df = results_df[['Model','Features','R2_train','R2_test','MAE_test','MSE_test']]
    print(results_df.to_string(index=False))

    results_df.to_csv('model_results.csv', index=False)
    print("\nSaved: model_results.csv")

    predictions = {}
    for name, model in best_models.items():
        predictions[name] = model.predict(X_te_e)
    np.savez('predictions.npz', y_test=y_te, y_train=y_tr, **predictions)
    print("Saved: predictions.npz")

    with open('trained_models.pkl', 'wb') as f:
        pickle.dump({
            'models': best_models,
            'scaler_eng': sc_e,
            'scaler_bas': sc_b,
        }, f)
    print("Saved: trained_models.pkl")

    return best_models, results_df

if __name__ == '__main__':
    from data_preprocessing import load_and_preprocess
    df, _ = load_and_preprocess('fatigue_dataset.xlsx')
    best_models, results_df = run_training(df)
