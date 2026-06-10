
import sys


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import time
from scipy.optimize import least_squares, root, minimize

np.random.seed(1234)
tf.random.set_seed(1234)

class RJ_PINNs:
    def __init__(self, X, u,X_bc0,X_bc1,X_ic, layers):
        
        self.x = X[:, 0:1]
        self.t = X[:, 1:2]
        self.u = u
        self.layers = layers

        # Initialize neural network
        self.weights, self.biases = self.initialize_NN(layers)

        # Variables for optimization
        self.A = tf.Variable(1.0, dtype=tf.float32, trainable=True)
        self.B = tf.Variable(1.0, dtype=tf.float32, trainable=True)
        self.C = tf.Variable(1.0, dtype=tf.float32, trainable=True)


        # Define placeholders for inputs (compatible with TensorFlow v2)
        self.x_tf = tf.convert_to_tensor(self.x, dtype=tf.float32)
        self.t_tf = tf.convert_to_tensor(self.t, dtype=tf.float32)
        self.u_tf = tf.convert_to_tensor(self.u, dtype=tf.float32)
        self.x_bc0 = tf.convert_to_tensor(X_bc0[:, 0:1], dtype=tf.float32)
        self.t_bc0 = tf.convert_to_tensor(X_bc0[:, 1:2], dtype=tf.float32)




        self.x_bc1 = tf.convert_to_tensor(X_bc1[:, 0:1], dtype=tf.float32)
        self.t_bc1 = tf.convert_to_tensor(X_bc1[:, 1:2], dtype=tf.float32)





        self.x_ic = tf.convert_to_tensor(X_ic[:, 0:1], dtype=tf.float32)
        self.t_ic = tf.convert_to_tensor(X_ic[:, 1:2], dtype=tf.float32)

        
        # Historique
        self.param_hist1 = []
        self.param_hist2 = []
        self.param_hist3 = []


        self.loss_hist = []
        self.iteration_hist = []
        self.err_l1=[]
        self.err_l2=[]
        self.err_l3=[]


        self.iteration = 0
        self.mean_A = 1.0
        self.mean_B = 1.0
        self.mean_C = 0
        self.std_A = 1
        self.std_B = 1
        self.std_C=1
        

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.random.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        #H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        H=X
        for l in range(num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_u(self, x, t):
        X = tf.concat([x, t], axis=1)
        u = self.neural_net(X, self.weights, self.biases)
        return u
    def R_bc(self, x, t):
        X = tf.concat([x, t], axis=1)
        u = self.neural_net(X, self.weights, self.biases)
        return u

    def R_ic(self, x, t):
        X = tf.concat([x, t], axis=1)
        u = self.neural_net(X, self.weights, self.biases)
        return u

    def net_f(self, x, t):
        """
        Calculate the residual of the PDE.

        Uses tf.GradientTape to compute gradients, which is required for eager execution.
        """
        lambda_1 = self.A
        lambda_2 = self.B
        lambda_3 = self.C

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            tape.watch(t)
            u = self.net_u(x, t)
            u_x = tape.gradient(u, x)

        u_t = tape.gradient(u, t)
        u_xx = tape.gradient(u_x, x)

        del tape # Drop the tape
        #f = lambda_1*u_t - lambda_2* u_xx + tf.exp(-t)* (tf.sin(np.pi *x) - np.pi**2 * tf.sin(np.pi * x))

        f = lambda_1*u_t + lambda_2*u* u_x - lambda_3 * u_xx
        return f
    
    def get_weights(self):
        variables = [*self.weights, *self.biases, self.A,self.B,self.C]
        return tf.concat([tf.reshape(var, [-1]) for var in variables], axis=0)

    def set_weights(self, flat_weights):
        idx = 0
        for var in [*self.weights, *self.biases, self.A,self.B,self.C]:
            shape = tf.shape(var)
            size = tf.reduce_prod(shape)
            # Convert new_values to float32 before assigning
            new_values = tf.reshape(flat_weights[idx:idx + size], shape)
            # Use tf.cast to convert the tensor to float32
            new_values = tf.cast(new_values, dtype=tf.float32) # Changed this line
            var.assign(new_values)
            idx += size

    def residu(self,p, print_loss=True):
     self.set_weights(p)
     u_pred = self.net_u(self.x_tf, self.t_tf)
     f_pred = self.net_f(self.x_tf, self.t_tf)
     r_data=(u_pred-self.u_tf)
     u_bc0d=tf.convert_to_tensor(0, dtype=tf.float32)
     u_bc1d= tf.convert_to_tensor(0, dtype=tf.float32)
     u_icd=tf.convert_to_tensor(-np.sin(np.pi* self.x_ic), dtype=tf.float32)
     u_bc0p= self.net_u(self.x_bc0, self.t_bc0)
     u_bc1p=self.R_bc(self.x_bc1, self.t_bc1)
     u_icp=self.R_ic(self.x_ic, self.t_ic)

     
     r_bc0= u_bc0p- u_bc0d
     r_bc1=u_bc1p-u_bc1d
     r_ic=u_icp-u_icd
     
       
     rl1=(self.A - self.mean_A) / self.std_A
     rl2=(self.B- self.mean_B) / self.std_B
     rl3=(self.C - self.mean_C) / self.std_C


     
     rl1 = tf.reshape(rl1, [-1,1])
     rl2 = tf.reshape(rl2, [-1,1])
     rl3 = tf.reshape(rl3, [-1,1])
       
     r=tf.concat([r_data,f_pred,rl1,rl2,rl3],axis=0)
     loss = tf.reduce_sum(tf.square(r)).numpy()  # Norme quadratique
     #p1,p2 = p[-3:]  # Derniers paramètres optimisés
     #compute error at each iteration
     A=1
     B=1
     C=0.003183
     
     p_predA=p[-3]
     p_predB=p[-2]
     p_predC=p[-1]
     params=p[-3:]

     err1=(A-p_predA)*100/A
     err2=(B-p_predB)*100/B
     err3=(C-p_predC)*100/C

     self.err_l1.append(err1)
     self.err_l2.append(err2)
     self.err_l3.append(err3)

     self.loss_hist.append(loss)
     self.param_hist1.append(p_predA)
     self.param_hist2.append(p_predB)
     self.param_hist3.append(p_predC)
     self.iteration_hist.append(self.iteration)

       
     if print_loss and self.iteration % 10 == 0:
            print(f"Iteration: {self.iteration}, Loss: {loss:.6f}, Parameters: {params}")

     self.iteration += 1  # Incrémenter le compteur d'itérations
     return tf.reshape(r, [-1]) # Flatten using tf.reshape

    def J(self,p):
        self.set_weights(p)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([*self.weights, *self.biases, self.A,self.B,self.C])
            r = self.residu(p,print_loss=False)
        jac_list = tape.jacobian(r, [*self.weights, *self.biases, self.A,self.B,self.C])
        del tape
        return np.hstack([tf.reshape(j, [r.shape[0], -1]).numpy() for j in jac_list])
    def residus(self,p,):
      re=self.residu(p,print_loss=True)
      return re.numpy()
    def train_lm(self):
      p=self.get_weights()
      result = least_squares(
               self.residus, 
               p, 
               jac=self.J, 
               method='trf',
               #max_nfev=1000,  # Increase max function evaluations
               #ftol=1e-6,      # Relax function tolerance
               #xtol=1e-6,      # Relax parameter tolerance
               #gtol=1e-6,      # Relax gradient tolerance
              # x_scale='jac',  # Scale parameters based on Jacobian
               #verbose=2       # Detailed output
       )            
      d=result.x
      
      
      self.set_weights(result.x)
      np.savez('burger_ABC_0.1.npz', 
                 p1=self.param_hist1,
                 p2=self.param_hist2,
                 p3=self.param_hist3,
                
                 err1=self.err_l1,
                 err2=self.err_l2,
                 err3=self.err_l3,
                 loss_hist=self.loss_hist, 
                 iteration_hist=self.iteration_hist)
      print("Optimization complete. Results saved to result.npz.")

    def predict(self, X_star):
        X_star_tf = tf.convert_to_tensor(X_star, dtype=tf.float32)
        x_star = X_star_tf[:, 0:1]
        t_star = X_star_tf[:, 1:2]
        u_star = self.net_u(x_star, t_star)
        f_star = self.net_f(x_star, t_star)
        return u_star.numpy(), f_star.numpy()

# Example usage of the model would go here

if __name__ == "__main__":
    nu = 0.01 / np.pi
    N_u = 500
    
    layers = [2, 20,20,1]
    
    data = np.load('./Burgers.npz')
    x = data['x']
    t = data['t']
    usol = data['usol']
    Exact = usol.T
    
    
    print(Exact.shape)
    X, T = np.meshgrid(x, t)
    #print(X.shape)
    #print(T.shape)
    #print("iteration \t loss \t ")
    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    u_star = Exact.flatten()[:, None]
    #Exact = np.exp(-(10 * np.pi * beta)**2 * T) * np.sin(10 * np.pi * X)

    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    u_star = Exact.flatten()[:, None]

    
    idx = np.random.choice(X_star.shape[0], N_u, replace=False)
    X_u_train = X_star[idx, :]
    u_train = u_star[idx, :]



    #t=0
    X_ic=np.hstack((X[0,:].flatten()[:, None], T[0,:].flatten()[:, None]))
    #print(X_ic.shape[0])
    n_ic=200

    idx_ic = np.random.choice(X_ic.shape[0], n_ic, replace=False)
    X_train_ic = X_ic[idx_ic, :]
    #u_train = u_star[idx, :]

    n_bc=100
    #x=0
    X_bc0=np.hstack((X[:,0].flatten()[:, None], T[:,0].flatten()[:, None]))
    #print(X_bc0.shape)
    idx_bc0 = np.random.choice(X_bc0.shape[0], n_bc, replace=False)
    X_train_bc0 = X_bc0[idx_bc0, :]

    n_bc
    #x=1
    X_bc1=np.hstack((X[:,-1].flatten()[:, None], T[:,-1].flatten()[:, None]))

    idx_bc1 = np.random.choice(X_bc1.shape[0], n_bc, replace=False)
    X_train_bc1 = X_bc1[idx_bc1, :]
    def plot3D(x,t,y):
     x_plot =x.squeeze(1)
     t_plot =t.squeeze(1)
     X,T= np.meshgrid(x_plot,t_plot)
     F_xt = y
     fig,ax=plt.subplots(1,1)
  # Convert X and T to NumPy arrays
     cp = ax.contourf(T,X, F_xt,20,cmap="rainbow")
     fig.colorbar(cp) # Add a colorbar to a plot
     ax.set_title('F(x,t)')
     ax.set_xlabel('t')
     ax.set_ylabel('x')
     plt.show()
     ax = plt.axes(projection='3d')
     ax.plot_surface(T, X, F_xt,cmap="rainbow")
     ax.set_xlabel('t')
     ax.set_ylabel('x')
     ax.set_zlabel('f(x,t)')
     plt.show()
    
    idx = np.random.choice(X_star.shape[0], N_u, replace=False)
    X_u_train = X_star[idx, :]
    u_train = u_star[idx, :]
    #noise = 0.1   
    #u_train = u_train + noise*np.std(u_train)*np.random.randn(u_train.shape[0], u_train.shape[1])
      
    #model = PhysicsInformedNN(X_u_train, u_train, layers, lb, ub)
    model = RJ_PINNs(X_u_train, u_train, X_train_bc0,X_train_bc1,X_train_ic,layers)
    model.train_lm()
    u_pred, f_pred = model.predict(X_star)


    plot3D(x,t,Exact)
    u_pred=u_pred.reshape(Exact.shape)
    plot3D(x,t,u_pred)
