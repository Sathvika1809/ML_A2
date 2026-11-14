import numpy as np
import matplotlib.pyplot as plt
from common import generate_moons

class ScratchMLP:

    def __init__(self,n_hidden,activation='leaky_relu',lr=0.1,reg_lambda=1e-4,n_epochs=2000,seed=42):
        self.n_hidden = n_hidden
        self.activation = activation
        self.lr = lr
        self.reg_lambda = reg_lambda
        self.n_epochs = n_epochs
        self.seed = seed

        rng = np.random.RandomState(seed)

        # Given draw W(1) first then W(2)
        # rng.normal(0, sigma, size)
        W1 = rng.normal(0.0, 1.0/np.sqrt(2.0) , size=(2,n_hidden))
        W2 = rng.normal(0.0, 1.0/np.sqrt(n_hidden), size=(n_hidden, 1))

        #Set biases to zero
        b1 = np.zeros((1, n_hidden))
        b2 = np.zeros((1,1))

        self.params = {"W1": W1,"b1": b1,"W2": W2, "b2": b2}
        self.loss_history = []
        self.snapshots = {}

    def _actfn(self, z):
        # leaky relu
        # fn(z) = {
        #      z   if z>0
        #     0.05z if z<=0
        # }
        if self.activation == "leaky_relu":
            return np.where(z>0, z, 0.05*z)
        elif self.activation == "tanh":
            return np.tanh(z)
        raise ValueError(f"Unsupported activation: {self.activation}")

    def _actfn_grad(self,z):
        if self.activation == "leaky_relu":
            return np.where(z>0, 1.0, 0.05)
        elif self.activation == "tanh":
            t = np.tanh(z)
            return 1.0-t**2
        raise ValueError(f"Unsupported activation: {self.activation}")

    def _sigmoid(self, z):
        # np.clip forces every value in the array x to stay inside range [-500,500]
        # anything above 500 becomes 500 and below -500 becomes -500
        # Because of Runtime Overflow
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def loss_and_grads(self, X, y):
        N = X.shape[0]
        y_col = y.reshape(-1,1)

        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]

        # Forward Propogation
        Z1 = X @ W1 + b1
        A1 = self._actfn(Z1)
        Z2 = A1 @ W2 + b2
        y_hat = self._sigmoid(Z2)

        #Regularised binary cross-entropy loss
        y_hat_clip = np.clip(y_hat, 1e-15, 1.0-1e-15)
        bce = -np.mean(y_col * np.log(y_hat_clip)+ (1.0 - y_col)* np.log(1.0 - y_hat_clip))
        # Regularization
        reg = (self.reg_lambda / (2.0*N)) * (np.sum(W1 ** 2) + np.sum(W2 ** 2))
        loss = float(bce + reg)

        # Backpropogation
        dZ2 = (y_hat - y_col)/N
        dW2 = A1.T @ dZ2 + (self.reg_lambda/N)*W2
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        dA1 = dZ2 @ W2.T
        dZ1 = dA1 * self._actfn_grad(Z1)
        dW1 = X.T @ dZ1 + (self.reg_lambda/N)*W1
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
        return loss,grads

    def fit(self,X,y,snapshot_epochs=None):
        self.loss_history = []
        self.snapshots = {}
        snap_set = set(snapshot_epochs) if snapshot_epochs is not None else set()

        for epoch in range(1, self.n_epochs + 1):
            loss,grads = self.loss_and_grads(X, y)
            self.loss_history.append(loss)

            for key in self.params:
                self.params[key] -= self.lr * grads[key]

            if epoch in snap_set:
                self.snapshots[epoch] = {k: v.copy() for k,v in self.params.items()}

        return self
    def predict_proba(self, X):

        Z1 = X @ self.params["W1"]+self.params["b1"]
        A1 = self._actfn(Z1)
        Z2 = A1 @ self.params["W2"]+self.params["b2"]
        return self._sigmoid(Z2).ravel()
    
    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)
    
def accuracy_score(y_true, y_pred):
    return float(np.mean(y_true == y_pred)*100.0)

def run_gradient_check(X_small,y_small,activation,n_hidden = 3,eps = 1e-5):
    model = ScratchMLP(n_hidden=n_hidden,activation=activation, reg_lambda=1e-4, seed = 42)
    _, analytical_grads = model.loss_and_grads(X_small,y_small)

    max_rel_err = 0.0
    for key in ["W1","W2"]:
        param = model.params[key]
        g_ana = analytical_grads[key]
        it = np.nditer(param, flags = ["multi_index"], op_flags=["readwrite"])

        while not it.finished:
            idx = it.multi_index
            org_val = param[idx]

            param[idx] = org_val + eps
            loss_plus, _ = model.loss_and_grads(X_small, y_small)

            param[idx] = org_val - eps
            loss_minus, _ = model.loss_and_grads(X_small, y_small)

            param[idx] = org_val  # restore
            g_num = (loss_plus - loss_minus) / (2.0 * eps)

            denom = max(abs(g_ana[idx]) + abs(g_num), 1e-12)
            rel_err = abs(g_ana[idx] - g_num) / denom
            if rel_err > max_rel_err:
                max_rel_err = rel_err

            it.iternext()

    return max_rel_err
def plot_boundary(ax, model, X, y, title):

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250), np.linspace(y_min, y_max, 250))
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = model.predict_proba(grid).reshape(xx.shape)

    ax.contourf(xx, yy, probs, levels=[0.0, 0.5, 1.0], colors=["#ffcccc", "#cce5ff"], alpha=0.75)
    ax.contour(xx, yy, probs, levels=[0.5], colors="black", linewidths=1.2)
    ax.scatter(X[y == 0, 0], X[y == 0, 1], c="#d62728", s=18, edgecolors="k", linewidth=0.4, label="Class 0")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c="#1f77b4", s=18, edgecolors="k", linewidth=0.4, label="Class 1")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")

def main():
    X_train, y_train, X_test, y_test = generate_moons()

    # 1. Gradient check on first 5 rows, H=3
    print("Finite-Difference Gradient Check (N=5, H=3)")
    for act in ["leaky_relu", "tanh"]:
        err = run_gradient_check(X_train[:5], y_train[:5], activation=act,n_hidden = 3)
        print(f"Activation: {act:10s} | Max Relative Error: {err:.4e}")

    # 2. Train models for Table 2
    print("\n Table 2: ScratchMLP on Two-Moons ")
    print(f"{'Activation':<12} | {'H':<4} | {'Final Loss':<12} | {'Train Acc (%)':<14} | {'Test Acc (%)':<12}")
    print("-" * 65)

    hidden_sizes = [1, 2, 8, 32]
    activations = ["leaky_relu", "tanh"]
    snap_epochs = [10, 200, 2000]
    h8_models = {}

    for act in activations:
        for h in hidden_sizes:
            mlp = ScratchMLP(n_hidden=h, activation=act, lr=0.1, reg_lambda=1e-4, n_epochs=2000, seed=42)
            snaps = snap_epochs if h == 8 else None
            mlp.fit(X_train, y_train, snapshot_epochs=snaps)

            final_loss = mlp.loss_history[-1]
            train_acc = accuracy_score(y_train, mlp.predict(X_train))
            test_acc = accuracy_score(y_test, mlp.predict(X_test))

            print(f"{act:<12} | {h:<4} | {final_loss:<12.4f} | {train_acc:<14.2f} | {test_acc:<12.2f}")

            if h == 8:
                h8_models[act] = mlp

    # 3. Decision boundary figure for H=8 (2x3)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for row, act in enumerate(activations):
        model = h8_models[act]
        for col, ep in enumerate(snap_epochs):
            # Temporarily swap in snapshot weights to plot
            model.params = model.snapshots[ep]
            plot_boundary(axes[row, col], model, X_train, y_train, f"{act} (Epoch {ep})")

    plt.tight_layout()
    plt.savefig("part1_boundaries.png", dpi=200)
    plt.close()
    print("\nSaved part1_boundaries.png successfully.")


if __name__ == "__main__":
    main()
    