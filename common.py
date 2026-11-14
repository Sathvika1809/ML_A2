import numpy as np

def _split_and_scale(X, y, test_frac=0.2, seed=42):
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(y))
    X, y = X[perm], y[perm]
    n_test = int(round(test_frac * len(y)))
    X_train, y_train = X[n_test:], y[n_test:]
    X_test, y_test = X[:n_test], y[:n_test]
    mu = X_train.mean(axis=0, keepdims=True)
    sd = X_train.std(axis=0, keepdims=True) + 1e-8
    return (X_train - mu) / sd, y_train, (X_test - mu) / sd, y_test

def generate_moons(n_samples=500, noise=0.20, test_frac=0.2, seed=42):
    rng = np.random.RandomState(seed)
    n_out = n_samples // 2
    n_in = n_samples - n_out
    theta_out = np.linspace(0.0, np.pi, n_out)
    theta_in = np.linspace(0.0, np.pi, n_in)
    X_out = np.column_stack([np.cos(theta_out), np.sin(theta_out)])
    X_in = np.column_stack([1.0 - np.cos(theta_in), 0.5 - np.sin(theta_in)])
    X = np.vstack([X_out, X_in]) + rng.normal(0.0, noise, size=(n_samples, 2))
    y = np.hstack([np.zeros(n_out, dtype=int), np.ones(n_in, dtype=int)])
    return _split_and_scale(X, y, test_frac=test_frac, seed=seed)

def generate_spirals(n_samples=600, noise=0.05, test_frac=0.2, seed=42):
    rng = np.random.RandomState(seed)
    n_arm = n_samples // 2
    t = np.linspace(0.5, 3.0 * np.pi, n_arm)
    r = t / (3.0 * np.pi)
    X0 = np.column_stack([r * np.cos(t), r * np.sin(t)])
    X1 = np.column_stack([-r * np.cos(t), -r * np.sin(t)])
    X = np.vstack([X0, X1]) + rng.normal(0.0, noise, size=(n_samples, 2))
    y = np.hstack([np.zeros(n_arm, dtype=int), np.ones(n_arm, dtype=int)])
    return _split_and_scale(X, y, test_frac=test_frac, seed=seed)