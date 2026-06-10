import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

tf.keras.backend.set_floatx('float32')
np.random.seed(1234)
tf.random.set_seed(1234)

# ================================================================
# Material parameters (mm, MPa)
# ================================================================
L   = 20.0
E   = 450000.0
nu  = 0.19

lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
mu  = E / (2.0 * (1.0 + nu))

# Asymmetric loading
sigma_right = 5.0    # MPa on x = L
sigma_top   = 10.0   # MPa on y = L

# Analytical solution (plane strain)
slope_ux = ((1 - nu**2)*sigma_right - nu*(1+nu)*sigma_top)  / E
slope_uy = ((1 - nu**2)*sigma_top   - nu*(1+nu)*sigma_right) / E


# ================================================================
# RJF class
# ================================================================
class RJF:
    def __init__(self, X, X_0, X_1, Y_0, Y_1, layers):
        self.x = tf.convert_to_tensor(X[:, 0:1], dtype=tf.float32)
        self.y = tf.convert_to_tensor(X[:, 1:2], dtype=tf.float32)

        self.xx_0 = tf.convert_to_tensor(X_0[:, 0:1], dtype=tf.float32)
        self.yx_0 = tf.convert_to_tensor(X_0[:, 1:2], dtype=tf.float32)
        self.xx_1 = tf.convert_to_tensor(X_1[:, 0:1], dtype=tf.float32)
        self.yx_1 = tf.convert_to_tensor(X_1[:, 1:2], dtype=tf.float32)
        self.xy_0 = tf.convert_to_tensor(Y_0[:, 0:1], dtype=tf.float32)
        self.yy_0 = tf.convert_to_tensor(Y_0[:, 1:2], dtype=tf.float32)
        self.xy_1 = tf.convert_to_tensor(Y_1[:, 0:1], dtype=tf.float32)
        self.yy_1 = tf.convert_to_tensor(Y_1[:, 1:2], dtype=tf.float32)

        # Dirichlet boundaries
        self.xx_D = tf.convert_to_tensor(X_0[:, 0:1], dtype=tf.float32)  # x = 0
        self.yx_D = tf.convert_to_tensor(X_0[:, 1:2], dtype=tf.float32)
        self.xy_D = tf.convert_to_tensor(Y_0[:, 0:1], dtype=tf.float32)  # y = 0
        self.yy_D = tf.convert_to_tensor(Y_0[:, 1:2], dtype=tf.float32)

        self.layers = layers
        self.weights, self.biases = self.initialize_NN(layers)
        self.iteration = 0

    def initialize_NN(self, layers):
        weights, biases = [], []
        for l in range(len(layers)-1):
            W = tf.Variable(tf.random.normal([layers[l], layers[l+1]], dtype=tf.float32)
                            * tf.sqrt(2./(layers[l]+layers[l+1])))
            b = tf.Variable(tf.zeros([1, layers[l+1]], dtype=tf.float32))
            weights.append(W); biases.append(b)
        return weights, biases

    def neural_net(self, X):
        H = X
        for l in range(len(self.layers)-2):
            H = tf.tanh(tf.add(tf.matmul(H, self.weights[l]), self.biases[l]))
        return tf.add(tf.matmul(H, self.weights[-1]), self.biases[-1])

    def model(self, x, y):
        return self.neural_net(tf.concat([x, y], axis=1))

    def get_strain(self, x, y):
        with tf.GradientTape(persistent=True) as g:
            g.watch(x); g.watch(y)
            u_pred = self.model(x, y)
            u_x = u_pred[:, 0:1]
            u_y = u_pred[:, 1:2]
        dux_dx = g.gradient(u_x, x)
        duy_dy = g.gradient(u_y, y)
        dux_dy = g.gradient(u_x, y)
        duy_dx = g.gradient(u_y, x)
        del g
        return dux_dx, duy_dy, 0.5*(dux_dy + duy_dx)

    def R_physic(self, x, y):
        with tf.GradientTape(persistent=True) as g1:
            g1.watch(x); g1.watch(y)
            eps_xx, eps_yy, eps_xy = self.get_strain(x, y)
            sig_xx = lam*(eps_xx+eps_yy) + 2*mu*eps_xx
            sig_yy = lam*(eps_xx+eps_yy) + 2*mu*eps_yy
            sig_xy = 2*mu*eps_xy
        eq1 = g1.gradient(sig_xx, x) + g1.gradient(sig_xy, y)
        eq2 = g1.gradient(sig_xy, x) + g1.gradient(sig_yy, y)
        del g1
        return tf.concat([eq1, eq2], axis=0)

    def r_dirichlet(self):
        """Dirichlet residuals: u_x = 0 on x=0, u_y = 0 on y=0"""
        u_left   = self.model(self.xx_D, self.yx_D)
        u_bottom = self.model(self.xy_D, self.yy_D)
        return tf.concat([u_left[:, 0:1], u_bottom[:, 1:2]], axis=0)

    def r_neumann(self):
        """Neumann residuals: traction on x=L and y=L"""
        # x = L : sigma_xx = sigma_right, sigma_xy = 0
        eps_xx, eps_yy, eps_xy = self.get_strain(self.xx_1, self.yx_1)
        r_right = tf.concat([
            lam*(eps_xx+eps_yy) + 2*mu*eps_xx - sigma_right,
            2*mu*eps_xy
        ], axis=0)
        # y = L : sigma_yy = sigma_top, sigma_xy = 0
        eps_xx, eps_yy, eps_xy = self.get_strain(self.xy_1, self.yy_1)
        r_top = tf.concat([
            lam*(eps_xx+eps_yy) + 2*mu*eps_yy - sigma_top,
            2*mu*eps_xy
        ], axis=0)
        return r_right, r_top

    def get_weights(self):
        return tf.concat([tf.reshape(v, [-1]) for v in [*self.weights, *self.biases]], axis=0)

    def set_weights(self, flat):
        idx = 0
        for var in [*self.weights, *self.biases]:
            size = tf.reduce_prod(tf.shape(var))
            var.assign(tf.cast(tf.reshape(flat[idx:idx+size], tf.shape(var)), tf.float32))
            idx += size

    def R(self, p, print_loss=True):
        self.set_weights(p)
        r_ph       = self.R_physic(self.x, self.y)
        r_di       = self.r_dirichlet()
        r_ri, r_to = self.r_neumann()
        r = tf.concat([r_ph, r_di, r_ri, r_to], axis=0)
        if print_loss and self.iteration % 10 == 0:
            loss = tf.reduce_mean(tf.square(r))
            print(f"Iteration {self.iteration:4d}  |  Loss = {loss.numpy():.4e}")
        self.iteration += 1
        return tf.reshape(r, [-1])

    def J(self, p):
        self.set_weights(p)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([*self.weights, *self.biases])
            r = self.R(p, print_loss=False)
        jac = tf.concat([
            tf.reshape(tape.jacobian(r, v), [r.shape[0], -1])
            for v in [*self.weights, *self.biases]
        ], axis=1)
        del tape
        return jac.numpy()

    def train_trf(self, max_iter=500):
        p0  = self.get_weights().numpy()
        res = least_squares(
            fun=lambda p: self.R(p).numpy(),
            jac=lambda p: self.J(p),
            x0=p0,
            method='trf'
        )
        self.set_weights(res.x)
        print(f"Optimization completed: {res.success}")
        return res

    def predict(self, X_star):
        X_tf = tf.convert_to_tensor(X_star, dtype=tf.float32)
        return self.model(X_tf[:, 0:1], X_tf[:, 1:2]).numpy()


# ================================================================
# Load FEniCS reference data
# ================================================================
data     = np.load("elastic.npz")
coor     = data["coor"]
u_x_fem  = data["u_x"]
u_y_fem  = data["u_y"]

x_vals = np.unique(coor[:, 0])
y_vals = np.unique(coor[:, 1])
X_mm, Y_mm = np.meshgrid(x_vals, y_vals)
ux_fem_grid = u_x_fem.reshape(X_mm.shape)
uy_fem_grid = u_y_fem.reshape(X_mm.shape)

X_star = np.hstack((X_mm.flatten()[:,None], Y_mm.flatten()[:,None]))

# ================================================================
# Training points
# ================================================================
N_u  = 500
n_bc = 31

idx  = np.random.choice(X_star.shape[0], N_u, replace=False)
X_u_train = X_star[idx, :]

Y_0 = np.hstack((X_mm[0,:].flatten()[:,None],  Y_mm[0,:].flatten()[:,None]))
Y_1 = np.hstack((X_mm[-1,:].flatten()[:,None], Y_mm[-1,:].flatten()[:,None]))
X_0 = np.hstack((X_mm[:,0].flatten()[:,None],  Y_mm[:,0].flatten()[:,None]))
X_1 = np.hstack((X_mm[:,-1].flatten()[:,None], Y_mm[:,-1].flatten()[:,None]))

Y_train_0 = Y_0[np.random.choice(Y_0.shape[0], n_bc, replace=False)]
Y_train_1 = Y_1[np.random.choice(Y_1.shape[0], n_bc, replace=False)]
X_train_0 = X_0[np.random.choice(X_0.shape[0], n_bc, replace=False)]
X_train_1 = X_1[np.random.choice(X_1.shape[0], n_bc, replace=False)]

# ================================================================
# Training
# ================================================================
layers = [2, 20, 20,20,20, 2]
model  = RJF(X_u_train, X_train_0, X_train_1, Y_train_0, Y_train_1, layers)
model.train_trf(max_iter=500)

# ================================================================
# Prediction and analytical solution
# ================================================================
u_pred   = model.predict(X_star)
u_x_rjf  = u_pred[:, 0].reshape(X_mm.shape)
u_y_rjf  = u_pred[:, 1].reshape(X_mm.shape)

u_x_ana  = slope_ux * X_mm
u_y_ana  = slope_uy * Y_mm

# ================================================================
# Error metrics
# ================================================================
print("\n=== Max/Min values ===")
print(f"u_x  RJF        : {np.max(u_x_rjf):.6f} / {np.min(u_x_rjf):.6f} mm")
print(f"u_x  FEM        : {np.max(ux_fem_grid):.6f} / {np.min(ux_fem_grid):.6f} mm")
print(f"u_x  Analytical : {slope_ux*L:.6f} / 0.000000 mm")
print(f"u_y  RJF        : {np.max(u_y_rjf):.6f} / {np.min(u_y_rjf):.6f} mm")
print(f"u_y  FEM        : {np.max(uy_fem_grid):.6f} / {np.min(uy_fem_grid):.6f} mm")
print(f"u_y  Analytical : {slope_uy*L:.6f} / 0.000000 mm")

ex_rjf_fem = np.linalg.norm(ux_fem_grid - u_x_rjf) / np.linalg.norm(ux_fem_grid) * 100
ey_rjf_fem = np.linalg.norm(uy_fem_grid - u_y_rjf) / np.linalg.norm(uy_fem_grid) * 100
ex_rjf_ana = np.linalg.norm(u_x_ana - u_x_rjf) / np.linalg.norm(u_x_ana) * 100
ey_rjf_ana = np.linalg.norm(u_y_ana - u_y_rjf) / np.linalg.norm(u_y_ana) * 100
ex_fem_ana = np.linalg.norm(u_x_ana - ux_fem_grid) / np.linalg.norm(u_x_ana) * 100
ey_fem_ana = np.linalg.norm(u_y_ana - uy_fem_grid) / np.linalg.norm(u_y_ana) * 100

print(f"\n=== Relative L2 errors ===")
print(f"RJF vs FEM        :  u_x = {ex_rjf_fem:.4f} %   u_y = {ey_rjf_fem:.4f} %")
print(f"RJF vs Analytical :  u_x = {ex_rjf_ana:.4f} %   u_y = {ey_rjf_ana:.4f} %")
print(f"FEM vs Analytical :  u_x = {ex_fem_ana:.4f} %   u_y = {ey_fem_ana:.4f} %")

# ================================================================
# Figure u_x : RJF | FEM | Analytical
# ================================================================
vmin_x = min(np.min(u_x_rjf), np.min(ux_fem_grid), np.min(u_x_ana))
vmax_x = max(np.max(u_x_rjf), np.max(ux_fem_grid), np.max(u_x_ana))

plt.figure(figsize=(18, 5))
plt.subplot(1, 3, 1)
plt.contourf(X_mm, Y_mm, u_x_rjf, levels=50, cmap='plasma', vmin=vmin_x, vmax=vmax_x)
plt.colorbar(label='$u_x$ (mm)')
plt.title("$u_x$ — RJF")
plt.xlabel("x (mm)"); plt.ylabel("y (mm)")

plt.subplot(1, 3, 2)
plt.contourf(X_mm, Y_mm, ux_fem_grid, levels=50, cmap='plasma', vmin=vmin_x, vmax=vmax_x)
plt.colorbar(label='$u_x$ (mm)')
plt.title("$u_x$ — FEM")
plt.xlabel("x (mm)"); plt.ylabel("y (mm)")

plt.subplot(1, 3, 3)
plt.contourf(X_mm, Y_mm, u_x_ana, levels=50, cmap='plasma', vmin=vmin_x, vmax=vmax_x)
plt.colorbar(label='$u_x$ (mm)')
plt.title("$u_x$ — Analytical")
plt.xlabel("x (mm)"); plt.ylabel("y (mm)")

plt.tight_layout()
plt.savefig("comparison_ux.png", dpi=300, bbox_inches='tight')
plt.show()

# ================================================================
# Figure u_y : RJF | FEM | Analytical
# ================================================================
vmin_y = min(np.min(u_y_rjf), np.min(uy_fem_grid), np.min(u_y_ana))
vmax_y = max(np.max(u_y_rjf), np.max(uy_fem_grid), np.max(u_y_ana))

plt.figure(figsize=(18, 5))
plt.subplot(1, 3, 1)
plt.contourf(X_mm, Y_mm, u_y_rjf, levels=50, cmap='plasma', vmin=vmin_y, vmax=vmax_y)
plt.colorbar(label='$u_y$ (mm)')
plt.title("$u_y$ — RJF")
plt.xlabel("x (mm)"); plt.ylabel("y (mm)")

plt.subplot(1, 3, 2)
plt.contourf(X_mm, Y_mm, uy_fem_grid, levels=50, cmap='plasma', vmin=vmin_y, vmax=vmax_y)
plt.colorbar(label='$u_y$ (mm)')
plt.title("$u_y$ — FEM")
plt.xlabel("x (mm)"); plt.ylabel("y (mm)")

plt.subplot(1, 3, 3)
plt.contourf(X_mm, Y_mm, u_y_ana, levels=50, cmap='plasma', vmin=vmin_y, vmax=vmax_y)
plt.colorbar(label='$u_y$ (mm)')
plt.title("$u_y$ — Analytical")
plt.xlabel("x (mm)"); plt.ylabel("y (mm)")

plt.tight_layout()
plt.savefig("comparison_uy.png", dpi=300, bbox_inches='tight')
plt.show()

# ================================================================
# Profile comparison at y = L/2
# ================================================================
def compare_profile(X, Y, u_fem, u_rjf, u_ana, y_target, label):
    idx = np.argmin(np.abs(Y[:, 0] - y_target))
    plt.figure(figsize=(8, 5))
    plt.plot(X[idx,:], u_fem[idx,:], 'b-',  lw=2,   label='FEM')
    plt.plot(X[idx,:], u_rjf[idx,:], 'r--', lw=2,   label='RJF')
    plt.plot(X[idx,:], u_ana[idx,:], 'g:',  lw=2.5, label='Analytical')
    plt.title(f"Comparison of {label} at y = {Y[idx,0]:.1f} mm")
    plt.xlabel("x (mm)"); plt.ylabel(f"{label} (mm)")
    plt.legend(); plt.grid()
    plt.tight_layout()
    plt.savefig(f"profile_{label}_y{int(y_target)}.png", dpi=300, bbox_inches='tight')
    plt.show()

compare_profile(X_mm, Y_mm, ux_fem_grid, u_x_rjf, u_x_ana, L/2, 'u_x')
compare_profile(X_mm, Y_mm, uy_fem_grid, u_y_rjf, u_y_ana, L/2, 'u_y')
