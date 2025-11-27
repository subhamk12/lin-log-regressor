import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)


np.random.seed(42)  # for reproducibility of the sampling & SGD shuffles

N = 1000000

# true parameters (theta_0, theta_1, theta_2) - will be needed later
theta_true = np.array([3.0, 1.0, 2.0]).reshape(-1, 1)

sample_x1 = np.random.normal(3, 2, N)
sample_x2 = np.random.normal(-1, 2, N)

# extra column for intercept term x0 = 1
X = np.column_stack([np.ones(N), sample_x1, sample_x2])

# y = theta_0 + theta_1 *x1 + theta_2 *x2 + e; e is N(0,sqrt(2))

y_noiseless = X @ theta_true
error = np.random.normal(0, np.sqrt(2), N).reshape(-1, 1)
y = y_noiseless + error

# Train/Test split = 80 - 20
train_ratio = 0.8
train_size = int(N * train_ratio)

X_train = X[:train_size]
y_train = y[:train_size]
X_test  = X[train_size:]
y_test  = y[train_size:]



def batch_loss(theta, Xb, yb):
    residue = Xb @ theta - yb
    r = Xb.shape[0]
    return (residue**2).sum() / (2.0 * r)

def batch_grad(theta, Xb, yb):
    r = Xb.shape[0]
    return (Xb.T @ (Xb @ theta - yb)) / r


def sgd_round_robin(X_train, y_train,learning_rate, batch_size,max_epochs, tolerance, history_step):
    #  we will shuffle after one epoch and visit every batch in order (round robin) 
    # stop when abs(theta_new - theta_old) < tol over an epoch (or hit maximum epochs)
    
    # we will also store the list of parameter vectors (for plot of trajectory) after a certain number of steps (history step)
    # num_updates is the ttotal mini batch updates performed
    # epochs_ran will store the number of epochs run
    
    m, d = X_train.shape
    theta = np.zeros((d, 1))  
    num_batches = m // batch_size
    m_eff = num_batches * batch_size  

    theta_history = [theta.copy()]
    num_updates = 0

    last_theta_epoch = theta.copy()

    for epoch in range(1, max_epochs + 1):
        # shuffle 
        perm = np.random.permutation(m)
        Xs = X_train[perm]
        ys = y_train[perm]

        Xs = Xs[:m_eff]
        ys = ys[:m_eff]

        for b in range(num_batches):
            start = b * batch_size
            end = start + batch_size
            Xb= Xs[start:end]
            yb = ys[start:end]

            gradient = batch_grad(theta, Xb, yb)
            theta= theta - learning_rate * gradient
            num_updates += 1
             # Keep a subsampled theta trajectory for plotting (memory aware)
            if (num_updates % history_step) == 0:
                theta_history.append(theta.copy())

        epoch_delta = np.linalg.norm(theta - last_theta_epoch)
        # we chose tthe tolerance in change in paramateres asa stopping criteria
        #ie if over a full epoch the change in theta is less than a aprticular value we will assume we have converged
        if epoch_delta < tolerance:
            break
        last_theta_epoch = theta.copy()

    if theta_history[-1] is not theta:
        theta_history.append(theta.copy())

    return theta, np.array(theta_history).reshape(-1, d), num_updates, epoch


batch_sizes = [1, 80, 8000, 800000]
step_size = 1e-3

results = {}
for r in batch_sizes:
    theta_sgd, theta_hist, updates, epochs = sgd_round_robin(
        X_train, y_train, learning_rate=step_size, batch_size=r,
        max_epochs=80 if r == 1 else 60 if r == 80 else 40 if r == 8000 else 20,
        tolerance=1e-6 if r >= 8000 else 5e-6 if r == 80 else 1e-5 if r == 1 else 1e-6,
        history_step=10 if r == 1 else 5 if r == 80 else 2 if r == 8000 else 1
    )
    results[r] = {
        "theta": theta_sgd,
        "history": theta_hist,
        "updates": updates,
        "epochs": epochs
    }
    print(f"[SGD] batch_size={r:>7}  epochs={epochs:<3}  updates={updates:<8}  theta≈ {theta_sgd.ravel()}")

#using pinv for numerical stability of inverse
theta_closed_form = np.linalg.pinv(X_train) @ y_train
print("\n")
print("[Closed-form] θ ≈ ", theta_closed_form.ravel())


#util fucntion for summarizing the paramter values obtained
def summarize(name, th):
    return f"{name:<14} θ0={th[0,0]: .6f}, θ1={th[1,0]: .6f}, θ2={th[2,0]: .6f}"

#lets print the values for compact comparison 
print("\n=== Parameter comparison ===")
print(summarize("True", theta_true))
print(summarize("Closed-form", theta_closed_form))


for r in batch_sizes:
    print(summarize(f"SGD r={r}", results[r]["theta"]))


#util function to calculate the mean squared error
def mse(X, y, theta):
    residue = np.matmul(X, theta) - y
    return float((residue**2).mean())

mse_train_cf = mse(X_train, y_train, theta_closed_form)
mse_test_cf  = mse(X_test,  y_test,  theta_closed_form)

print("\n === MSE (Closed-form) ===")
print(f"Train MSE: {mse_train_cf:.6f}")
print(f"Test  MSE: {mse_test_cf:.6f}")


r_ref = 8000
mse_train_sgd = mse(X_train, y_train, results[r_ref]["theta"])
mse_test_sgd  = mse(X_test,  y_test,  results[r_ref]["theta"])
print(f"\n=== MSE (SGD, r={r_ref}) ===")
print(f"Train MSE: {mse_train_sgd:.6f}")
print(f"Test  MSE: {mse_test_sgd:.6f}")

# we will now plot the trajectory
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection='3d')
ax.set_title("θ Trajectory during SGD (until convergence)")
ax.set_xlabel("θ0")
ax.set_ylabel("θ1")
ax.set_zlabel("θ2")

for r in batch_sizes:
    hist = results[r]["history"]
    ax.plot(hist[:,0], hist[:,1], hist[:,2], label=f"r={r}") 
    # maark start and end points
    ax.scatter(hist[0,0],  hist[0,1],  hist[0,2],  marker='x')
    ax.scatter(hist[-1,0], hist[-1,1], hist[-1,2], marker='o')

ax.legend()
plt.tight_layout()
plt.show()

