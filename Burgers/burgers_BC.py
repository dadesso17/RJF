
import sys

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import time
from scipy.optimize import least_squares

np.random.seed(1234)
tf.random.set_seed(1234)

class RJF:
    def __init__(self, X, u, X_bc0, X_bc1, X_ic, layers):

        self.x = X[:, 0:1]
        self.t = X[:, 1:2]
        self.u = u
        self.layers = layers

        # A is KNOWN and fixed
        self.A = tf.constant(1.0, dtype=tf.float32)

        # Only B and C are unknown/trainable
        self.B = tf.Variable(1.0, dtype=tf.float32, trainable=True)
        self.C = tf.Variable(1.0, dtype=tf.float32, trainable=True)

        # Initialize neural network
        self.weights, self.biases = self.initialize_NN(layers)

        # Tensors
        self.x_tf  = tf.convert_to_tensor(self.x, dtype=tf.float32)
        self.t_tf  = tf.convert_to_tensor(self.t, dtype=tf.float32)
        self.u_tf  = tf.convert_to_tensor(self.u, dtype=tf.float32)

        self.x_bc0 = tf.convert_to_tensor(X_bc0[:, 0:1], dtype=tf.float32)
        self.t_bc0 = tf.convert_to_tensor(X_bc0[:, 1:2], dtype=tf.float32)

        self.x_bc1 = tf.convert_to_tensor(X_bc1[:, 0:1], dtype=tf.float32)
        self.t_bc1 = tf.convert_to_tensor(X_bc1[:, 1:2], dtype=tf.float32)

        self.x_ic  = tf.convert_to_tensor(X_ic[:, 0:1],  dtype=tf.float32)
        self.t_ic  = tf.convert_to_tensor(X_ic[:, 1:2],  dtype=tf.float32)

        # History — only B and C
        self.param_hist_B = []
        self.param_hist_C = []
        self.loss_hist     = []
        self.iteration_hist = []
        self.err_B = []
        self.err_C = []

        self.iteration = 0

        # Prior (regularisation) for B and C
        self.mean_B = 1.0
        self.mean_C = 0.0
        self.std_B  = 0.1
        self.std_C  = 0.1

    # ------------------------------------------------------------------
    def initialize_NN(self, layers):
        weights, biases = [], []
        for l in range(len(layers) - 1):
            W = self.xavier_init([layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1, layers[l+1]], dtype=tf.float32))
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        stddev = np.sqrt(2 / (size[0] + size[1]))
        return tf.Variable(
            tf.random.truncated_normal(size, stddev=stddev, dtype=tf.float32))

    def neural_net(self, X, weights, biases):
        H = X
        for W, b in zip(weights[:-1], biases[:-1]):
            H = tf.tanh(tf.matmul(H, W) + b)
        return tf.matmul(H, weights[-1]) + biases[-1]

    def net_u(self, x, t):
        return self.neural_net(tf.concat([x, t], axis=1),
                               self.weights, self.biases)

    def net_f(self, x, t):
        """PDE residual:  A*u_t + B*u*u_x - C*u_xx = 0  (A=1 fixed)"""
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([x, t])
            u   = self.net_u(x, t)
            u_x = tape.gradient(u, x)
        u_t  = tape.gradient(u, t)
        u_xx = tape.gradient(u_x, x)
        del tape
        # A is constant = 1
        return self.A * u_t + self.B * u * u_x - self.C * u_xx

    # ------------------------------------------------------------------
    def get_weights(self):
        """Flatten all trainable variables: NN weights + B + C."""
        variables = [*self.weights, *self.biases, self.B, self.C]
        return tf.concat([tf.reshape(v, [-1]) for v in variables], axis=0)

    def set_weights(self, flat):
        idx = 0
        for var in [*self.weights, *self.biases, self.B, self.C]:
            size = tf.reduce_prod(tf.shape(var))
            var.assign(tf.cast(
                tf.reshape(flat[idx:idx+size], tf.shape(var)), tf.float32))
            idx += size

    # ------------------------------------------------------------------
    def residu(self, p, print_loss=True):
        self.set_weights(p)

        u_pred  = self.net_u(self.x_tf, self.t_tf)
        f_pred  = self.net_f(self.x_tf, self.t_tf)

        r_data  = u_pred - self.u_tf
        r_bc0   = self.net_u(self.x_bc0, self.t_bc0)          # = 0
        r_bc1   = self.net_u(self.x_bc1, self.t_bc1)          # = 0
        r_ic    = self.net_u(self.x_ic,  self.t_ic) \
                  - tf.cast(-tf.sin(np.pi * self.x_ic), tf.float32)

        # Prior residuals for B and C only
        #rl_B = tf.reshape((self.B - self.mean_B) / self.std_B, [-1, 1])
        #rl_C = tf.reshape((self.C - self.mean_C) / self.std_C, [-1, 1])

        r = tf.concat([r_data, f_pred, r_bc0, r_bc1, r_ic], axis=0)

        loss = tf.reduce_sum(tf.square(r)).numpy()

        # True values
        B_true = 1.0
        C_true = 0.003183

        p_B = float(p[-2])
        p_C = float(p[-1])

        self.err_B.append((B_true - p_B) * 100 / B_true)
        self.err_C.append((C_true - p_C) * 100 / C_true)
        self.loss_hist.append(loss)
        self.param_hist_B.append(p_B)
        self.param_hist_C.append(p_C)
        self.iteration_hist.append(self.iteration)

        if print_loss and self.iteration % 10 == 0:
            print(f"Iter {self.iteration:4d} | Loss: {loss:.4e} "
                  f"| B: {p_B:.6f}  C: {p_C:.6f}")

        self.iteration += 1
        return tf.reshape(r, [-1])

    def J(self, p):
        self.set_weights(p)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([*self.weights, *self.biases, self.B, self.C])
            r = self.residu(p, print_loss=False)
        jac_list = tape.jacobian(
            r, [*self.weights, *self.biases, self.B, self.C])
        del tape
        return np.hstack(
            [tf.reshape(j, [r.shape[0], -1]).numpy() for j in jac_list])

    def residus(self, p):
        return self.residu(p, print_loss=True).numpy()

    # ------------------------------------------------------------------
    def train(self):
        p0 = self.get_weights()
        result = least_squares(
            self.residus, p0,
            jac=self.J,
            method='trf',
        )
        self.set_weights(result.x)
        np.savez('burger_BC_0.1.npz',
                 p_B=self.param_hist_B,
                 p_C=self.param_hist_C,
                 err_B=self.err_B,
                 err_C=self.err_C,
                 loss_hist=self.loss_hist,
                 iteration_hist=self.iteration_hist)
        print("Done. Results saved to burger_BC.npz")

    def predict(self, X_star):
        Xt = tf.convert_to_tensor(X_star, dtype=tf.float32)
        u  = self.net_u(Xt[:, 0:1], Xt[:, 1:2])
        f  = self.net_f(Xt[:, 0:1], Xt[:, 1:2])
        return u.numpy(), f.numpy()


# ======================================================================
if __name__ == "__main__":

    N_u    = 500
    layers = [2, 20, 20,1]

    data  = np.load('./Burgers.npz')
    x     = data['x']
    t     = data['t']
    Exact = data['usol'].T

    X, T     = np.meshgrid(x, t)
    X_star   = np.hstack([X.flatten()[:, None], T.flatten()[:, None]])
    u_star   = Exact.flatten()[:, None]

    idx        = np.random.choice(X_star.shape[0], N_u, replace=False)
    X_u_train  = X_star[idx]
    u_train    = u_star[idx]

    # Initial condition  t=0
    X_ic   = np.hstack([X[0].flatten()[:, None], T[0].flatten()[:, None]])
    idx_ic = np.random.choice(X_ic.shape[0], 200, replace=False)
    X_train_ic = X_ic[idx_ic]

    # Boundary x=-1
    X_bc0   = np.hstack([X[:, 0].flatten()[:, None], T[:, 0].flatten()[:, None]])
    idx_bc0 = np.random.choice(X_bc0.shape[0], 100, replace=False)
    X_train_bc0 = X_bc0[idx_bc0]

    # Boundary x=+1
    X_bc1   = np.hstack([X[:, -1].flatten()[:, None], T[:, -1].flatten()[:, None]])
    idx_bc1 = np.random.choice(X_bc1.shape[0], 100, replace=False)
    X_train_bc1 = X_bc1[idx_bc1]
    #noise = 0.1 #   


    #Add Noise
    #u_train = u_train + noise*np.std(u_train)*np.random.randn(u_train.shape[0], u_train.shape[1])


    model = RJF(X_u_train, u_train,
                     X_train_bc0, X_train_bc1, X_train_ic, layers)
    model.train()

    u_pred, _ = model.predict(X_star)
    print(f"\nFinal estimated B = {float(model.B):.6f}  (true: 1.0)")
    print(f"Final estimated C = {float(model.C):.6f}  (true: 0.003183)")
