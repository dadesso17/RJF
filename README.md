# RJF: A Residual--Jacobian Framework for Multi-Objective Optimization with applications to Physics-Informed Learning



RJF (Residual–Jacobian Framework) is a general framework for addressing multi-objective problems in Artificial Intelligence and Scientific Machine Learning. Instead of combining multiple objectives into a single weighted loss function, RJF reformulates each objective as a residual component and directly exploits the Jacobian of the residual vector with respect to trainable and physical parameters. Automatic differentiation can be used to compute Jacobians with respect to neural network parameters, while analytical expressions, automatic differentiation, or finite-difference approximations can be employed for external physical parameters.

RJF is designed for multi-objective learning problems in Artificial Intelligence, including Scientific Machine Learning applications such as Physics-Informed Neural Networks (PINNs), inverse problems, parameter identification, and other learning tasks involving multiple residual objectives.








## Evolution from RJ-PINNs

RJF (Residual–Jacobian Framework) is an evolution and generalization of RJ-PINNs (Residual Jacobian Physics-Informed Neural Networks).

RJ-PINNs introduced the residual–Jacobian formulation for Physics-Informed Neural Networks by reformulating PINN optimization as a residual least-squares problem and exploiting the Jacobian of the residuals. RJF extends this idea beyond PINNs by providing a more general framework for multi-objective learning problems in Artificial Intelligence and Scientific Machine Learning.

# Thereby, this reformulation eliminates the need for manually tuned loss weights, which is a common challenge in conventional PINN formulations.


-----------------------------------------------------------------------------------------------------------------------------------------------------------------------


## Please cite RJF if you use it:


 Dadesso, Dadoyi, RJF: A Residual--Jacobian Framework for Multi-Objective Optimization with Applications to Physics-Informed Learning (June 10, 2026). Available at SSRN: https://ssrn.com/abstract=5506728 or http://dx.doi.org/10.2139/ssrn.5506728 
