import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# Load results
# ============================================================================
result = np.load('burger_BC_without_noise.npz')

param1 = result["p_B"]
param2 = result["p_C"]
err1 = result["err_B"]
err2 = result["err_C"]
loss_hist = result["loss_hist"]
iteration_hist = result["iteration_hist"]

# ============================================================================
# Error evolution - two side-by-side subplots
# ============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left subplot: Error on B
ax1.plot(
    iteration_hist,
    err1,
    color='black',
    linestyle='-',
    marker='o',
    markevery=max(1, len(iteration_hist)//20),
    linewidth=2,
    label='Error on B'
)
ax1.set_xlabel('Iterations', fontsize=12)
ax1.set_ylabel('Error on B', fontsize=12)
ax1.legend(frameon=True)
ax1.grid(True, linestyle=':', alpha=0.6)

# Right subplot: Error on C
ax2.plot(
    iteration_hist,
    err2,
    color='black',
    linestyle='--',
    marker='s',
    markevery=max(1, len(iteration_hist)//20),
    linewidth=2,
    label='Error on C'
)
ax2.set_xlabel('Iterations', fontsize=12)
ax2.set_ylabel('Error on C', fontsize=12)
ax2.legend(frameon=True)
ax2.grid(True, linestyle=':', alpha=0.6)

plt.suptitle('Evolution of Relative Errors', fontsize=14)
plt.tight_layout()
plt.savefig("Error_Evolution_SideBySide.pdf", format="pdf", bbox_inches="tight")
plt.show()

# ============================================================================
# Parameter convergence - two side-by-side subplots
# ============================================================================
param1_real = 1.0
param2_real = 0.01 / np.pi

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left subplot: Parameter B
ax1.plot(
    iteration_hist,
    param1,
    color='black',
    linestyle='-',
    marker='o',
    markevery=max(1, len(iteration_hist)//20),
    linewidth=2,
    label='Estimated B'
)
ax1.axhline(
    y=param1_real,
    color='black',
    linestyle=':',
    linewidth=2,
    label='True B'
)
ax1.set_xlabel('Iterations', fontsize=12)
ax1.set_ylabel('Parameter B', fontsize=12)
ax1.legend(frameon=True)
ax1.grid(True, linestyle=':', alpha=0.6)

# Right subplot: Parameter C
ax2.plot(
    iteration_hist,
    param2,
    color='black',
    linestyle='--',
    marker='s',
    markevery=max(1, len(iteration_hist)//20),
    linewidth=2,
    label='Estimated C'
)
ax2.axhline(
    y=param2_real,
    color='black',
    linestyle='-.',
    linewidth=2,
    label='True C'
)
ax2.set_xlabel('Iterations', fontsize=12)
ax2.set_ylabel('Parameter C', fontsize=12)
ax2.legend(frameon=True)
ax2.grid(True, linestyle=':', alpha=0.6)

plt.suptitle('Convergence of Estimated Parameters', fontsize=14)
plt.tight_layout()
plt.savefig("Parameter_Convergence_SideBySide.pdf", format="pdf", bbox_inches="tight")
plt.show()

# ============================================================================
# Cost function (unchanged)
# ============================================================================
plt.figure(figsize=(8, 6))

plt.plot(
    iteration_hist,
    loss_hist,
    color='black',
    marker='d',
    markevery=max(1, len(iteration_hist)//20),
    linewidth=2,
    label='Cost Function'
)

plt.xlabel('Iterations', fontsize=12)
plt.ylabel('Cost Function', fontsize=12)
plt.title('Evolution of Cost Function')
plt.legend(frameon=True)
plt.tight_layout()
plt.savefig("Cost_Function.pdf", format="pdf", bbox_inches="tight")
plt.show()
