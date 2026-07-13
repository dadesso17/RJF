# RJF: A Residual--Jacobian Framework for Multi-Objective Optimization



RJF (Residual–Jacobian Framework) is a general framework for addressing multi-objective problems in Artificial Intelligence and Scientific Machine Learning. Instead of combining multiple objectives into a single weighted loss function, RJF reformulates each objective as a residual component and directly exploits the Jacobian of the residual vector with respect to trainable and physical parameters. Automatic differentiation can be used to compute Jacobians with respect to neural network parameters, while analytical expressions, automatic differentiation, or finite-difference approximations can be employed for external physical parameters.

RJF is designed for multi-objective learning problems in Artificial Intelligence, including Scientific Machine Learning applications such as Physics-Informed Neural Networks (PINNs), inverse problems, parameter identification, and other learning tasks involving multiple residual objectives.








## Relation with RJ-PINNs

RJF (Residual–Jacobian Framework) is an evolution of RJ-PINNs (Residual Jacobian Physics-Informed Neural Networks). While RJ-PINNs introduced the residual–Jacobian formulation for Physics-Informed Neural Networks, RJF extends this concept into a more general framework for multi-objective learning problems in Artificial Intelligence and Scientific Machine Learning.

The original RJ-PINNs work can be cited as:

Dadesso, Dadoyi. "Residual Jacobian Physics-Informed Neural Networks (RJ-PINNs) for improved convergence and stability." Available at SSRN: https://ssrn.com/abstract=5506728.

