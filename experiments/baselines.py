import os
"""
Baseline models: XGBoost, Random Forest, SVR, kNN, ElasticNet, MLP.
"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


def _train_sklearn_baseline(X_train, y_train, X_test, y_test, task_names, make_model, model_name='model'):
    """Generic per-task sklearn baseline trainer."""
    results = {}
    all_preds = np.full_like(y_test, np.nan)
    for i, name in enumerate(task_names):
        train_valid = ~np.isnan(y_train[:, i])
        test_valid = ~np.isnan(y_test[:, i])
        if train_valid.sum() < 10:
            results[name] = {'R2': 0, 'RMSE': 0, 'MAE': 0}
            continue
        model = make_model()
        model.fit(X_train[train_valid], y_train[train_valid, i])
        y_pred = model.predict(X_test[test_valid])
        all_preds[test_valid, i] = y_pred
        r2 = r2_score(y_test[test_valid, i], y_pred)
        rmse = np.sqrt(mean_squared_error(y_test[test_valid, i], y_pred))
        mae = mean_absolute_error(y_test[test_valid, i], y_pred)
        results[name] = {'R2': float(r2), 'RMSE': float(rmse), 'MAE': float(mae), 'model': model}
    return results, all_preds


def train_xgboost_baseline(X_train, y_train, X_test, y_test, 
                           task_names, n_estimators=100, max_depth=3):
    """Train XGBoost for each task independently (default params)."""
    try:
        import xgboost as xgb
    except ImportError:
        print('  XGBoost not installed, skipping')
        return {}, np.full_like(y_test, np.nan)
    return _train_sklearn_baseline(
        X_train, y_train, X_test, y_test, task_names,
        lambda: xgb.XGBRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                 learning_rate=0.1, random_state=42, n_jobs=-1, verbosity=0),
        'XGBoost')


def train_random_forest_baseline(X_train, y_train, X_test, y_test,
                                  task_names, n_estimators=300, max_depth=12):
    """Train Random Forest for each task independently."""
    return _train_sklearn_baseline(
        X_train, y_train, X_test, y_test, task_names,
        lambda: RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                      min_samples_leaf=3, random_state=42, n_jobs=-1),
        'Random Forest')


def train_svr_baseline(X_train, y_train, X_test, y_test, task_names):
    """Train SVR for each task independently."""
    return _train_sklearn_baseline(
        X_train, y_train, X_test, y_test, task_names,
        lambda: SVR(kernel='rbf', C=1.0, gamma='scale'),
        'SVR')


def train_knn_baseline(X_train, y_train, X_test, y_test, task_names, n_neighbors=7):
    """Train kNN for each task independently."""
    return _train_sklearn_baseline(
        X_train, y_train, X_test, y_test, task_names,
        lambda: KNeighborsRegressor(n_neighbors=n_neighbors, weights='distance'),
        'kNN')


def train_elasticnet_baseline(X_train, y_train, X_test, y_test, task_names):
    """Train ElasticNet for each task independently."""
    return _train_sklearn_baseline(
        X_train, y_train, X_test, y_test, task_names,
        lambda: ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000, random_state=42),
        'ElasticNet')


if __name__ == '__main__':
    from data_loader import ThinFilmDataset
    from sklearn.model_selection import train_test_split

    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'unified_dataset', 'training_dataset.csv')
    ds = ThinFilmDataset(data_path)
    raw = ds.df[ds.PROP_NAMES].values.astype(np.float32)
    X = ds.features
    X_train, X_test = train_test_split(np.arange(len(ds)), test_size=0.2, random_state=42)
    # rebuild proper raw split
    idx = np.arange(len(ds))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42)
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = raw[train_idx], raw[test_idx]

    for name, fn in [('XGBoost', train_xgboost_baseline),
                     ('RF', train_random_forest_baseline),
                     ('SVR', train_svr_baseline),
                     ('kNN', train_knn_baseline),
                     ('ElasticNet', train_elasticnet_baseline)]:
        print(f'\nTraining {name}...')
        results, _ = fn(X_tr, y_tr, X_te, y_te, ds.PROP_NAMES)
        for t, m in results.items():
            print(f'  {name} {t}: R²={m["R2"]:.4f}')
