import numpy as np
import matplotlib.pyplot as plt
from common import generate_spirals
from sklearn.neural_network import MLPClassifier
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
#Load dataset
X_train,y_train,X_test,y_test = generate_spirals()
activations = ["logistic","relu"]

#Accuracy
def acc(model,x,y):
    return model.score(x,y)*100

# Function to draw decision boundary(Non linear)
def plot_boundary(ax,model,title):
    x = np.linspace(-3,3,150)
    y = np.linspace(-3,3,150)

    xx,yy = np.meshgrid(x,y)
    grid = np.c_[xx.ravel(),yy.ravel()]

    pred = model.predict(grid)
    pred = pred.reshape(xx.shape)
    ax.contourf(
        xx,yy,pred,
        alpha = 0.3,
        cmap="coolwarm"
    )

    ax.scatter(
        X_train[:,0],X_train[:,1],
        c=y_train,cmap="coolwarm",edgecolor="black",
        s=12
    )
    ax.set_title(title)
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")

# 2.1 Depth vs Width
architectures = [
    (36,),
    (18,18),
    (12,12,12),
    (6,6,6,6,6,6)
]

fig, axes = plt.subplots(
    4, 2,
    figsize=(14,22),
    constrained_layout=True
)

# to plot A4 -> save the models
deep_logistic = None
deep_relu = None


print("\nPART 2.1 RESULTS")

for i in range(4):
    for j in range(2):
        arc = architectures[i]
        act = activations[j]

        model = MLPClassifier(
            hidden_layer_sizes=arc,
            activation = act,
            solver = 'adam',
            learning_rate_init = 0.01,
            alpha = 0.0001,
            max_iter = 1500,
            n_iter_no_change=1500,
            random_state = 42
        )

        model.fit(X_train,y_train)

        train_acc = acc(model,X_train,y_train)
        test_acc = acc(model,X_test,y_test)

        loss = model.loss_
        norm = np.linalg.norm(model.coefs_[0], "fro")

        print("\nArchitecture:", arc)
        print("Activation:", act)
        print("Train Accuracy:", round(train_acc, 2))
        print("Test Accuracy:", round(test_acc, 2))
        print("Final Loss:", round(loss, 4))
        print("Layer 1 Norm:", round(norm, 4))

        title = str(arc) + " "+act
        plot_boundary(axes[i,j], model, title)

        if i==3 and j==0:
            # Save model here to use for loss curve
            deep_logistic = model

        if i==3 and j==1:
            deep_relu = model

plt.savefig("part2_depth_vs_width.png")
plt.show()

#Loss curves

plt.figure(figsize=(8,5))
plt.plot(deep_logistic.loss_curve_,label="Logistic")
plt.plot(deep_relu.loss_curve_,label="ReLU")

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Curves for Deepest Architecture A4")

plt.legend()
plt.grid()

plt.savefig("part2_loss_curves.png")
plt.show()

# 2.2 - Width sweep
print("\nPART 2.2 WIDTH SWEEP")
widths = [2,4,16,64]
for act in activations:
    print("\nActivation:", act)

    for M in widths:
        model = MLPClassifier(
            hidden_layer_sizes=(M,M),
            activation=act,
            solver = 'adam',
            learning_rate_init = 0.01,
            alpha = 0.0001,
            max_iter = 1500,
            n_iter_no_change=1500,
            random_state = 42
        )
        model.fit(X_train,y_train)
        train_acc = acc(model,X_train,y_train)
        test_acc = acc(model,X_test,y_test)

        P = M*M+5*M+1
        print("\nWidth:", M)
        print("Parameters:", P)
        print("Train Accuracy:", round(train_acc, 2))
        print("Test Accuracy:", round(test_acc, 2))

#Decision Boundary Evaluation
epochs = [1,25,100,300,1000]
fig, axes = plt.subplots(2,5,figsize=(18,9),constrained_layout=True)

for i in range(2):
    act = activations[i]

    model = MLPClassifier(
        hidden_layer_sizes=(16,16),
        activation = act,
        solver = 'adam',
        learning_rate_init = 0.01,
        alpha = 0.0001,
        random_state = 42
    )
    for epoch in range(1,1001):
        model.partial_fit(
            X_train,
            y_train,
            classes = [0,1]
        )
        if epoch in epochs:
            j = epochs.index(epoch)
            title = act+" Epoch "+str(epoch)
            plot_boundary(axes[i,j], model,title)

plt.savefig("part2_boundary_evolution.png")
plt.show()

print("\nPart 2 tasks completed")



