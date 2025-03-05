# import numpy as np
# from transformers import pipeline
# from qiskit import QuantumCircuit, transpile
# from qiskit_aer import AerSimulator
# from qiskit.visualization import plot_histogram
# from qiskit.circuit import ParameterVector
# import matplotlib.pyplot as plt
# from qiskit.algorithms.optimizers import COBYLA

# #############################
# # 1. Classical Text Embedding
# #############################

# # Create a BERT feature extraction pipeline.
# feature_extractor = pipeline("feature-extraction", model="bert-base-uncased", tokenizer="bert-base-uncased")

# def get_text_embedding(text):
#     """
#     Extracts the [CLS] token embedding from text using BERT.
#     Returns a 768-dimensional vector.
#     """
#     outputs = feature_extractor(text)
#     cls_embedding = np.array(outputs[0][0])
#     return cls_embedding

# # Example text
# text = "Quantum computing and machine learning converge in hybrid systems."
# embedding = get_text_embedding(text)
# print("Original embedding shape:", embedding.shape)

# ############################################
# # 2. Dimensionality Reduction & Normalization
# ############################################

# # For a 3-qubit circuit we need 2^3 = 8 amplitudes.
# # (In practice, use PCA or another method. Here we simply take the first 8 features.)
# embedding_reduced = embedding[:8]
# print("Reduced embedding (8 dims):", embedding_reduced)

# # Normalize to create a valid quantum state (state_vector)
# norm = np.linalg.norm(embedding_reduced)
# state_vector = embedding_reduced / norm
# print("Normalized state vector:", state_vector)

# # Target probability distribution: squared amplitudes
# target_probs = np.abs(state_vector)**2
# print("Target probabilities:", target_probs)

# ############################################
# # 3. Build a Variational Quantum Circuit (VQC)
# ############################################

# num_qubits = 3  # since 2^3 = 8
# qc = QuantumCircuit(num_qubits, num_qubits)

# # Create parameter vectors for rotations.
# theta = ParameterVector('theta', num_qubits)
# phi   = ParameterVector('phi', num_qubits)

# # First layer: parameterized Ry rotations
# for i in range(num_qubits):
#     qc.ry(theta[i], i)

# # Add an entangling layer.
# qc.cx(0, 1)
# qc.cx(1, 2)

# # Second layer: parameterized Rz rotations
# for i in range(num_qubits):
#     qc.rz(phi[i], i)

# # Add measurement (we use the measurement probabilities for the cost function)
# qc.measure(range(num_qubits), range(num_qubits))

# ############################################
# # 4. Define the Cost Function for Optimization
# ############################################

# simulator = AerSimulator()

# def cost_function(params):
#     """
#     Runs the variational circuit with given parameters and computes
#     the squared error between the measured probability distribution
#     and the target distribution (derived from the classical embedding).
#     """
#     # Bind parameters: first num_qubits for theta, next for phi.
#     param_dict = {}
#     for i in range(num_qubits):
#         param_dict[theta[i]] = params[i]
#         param_dict[phi[i]] = params[i + num_qubits]
#     qc_bound = qc.bind_parameters(param_dict)
    
#     compiled = transpile(qc_bound, simulator)
#     job = simulator.run(compiled, shots=1024)
#     result = job.result()
#     counts = result.get_counts()
    
#     # Convert counts to probability vector.
#     probs = np.zeros(2**num_qubits)
#     for state, count in counts.items():
#         index = int(state, 2)
#         probs[index] = count / 1024.0
    
#     # Compute squared error cost.
#     cost = np.sum((probs - target_probs)**2)
#     return cost

# ############################################
# # 5. Optimize the VQC Parameters
# ############################################

# optimizer = COBYLA(maxiter=200)
# initial_params = np.random.rand(2 * num_qubits)
# opt_result = optimizer.optimize(num_vars=2 * num_qubits, 
#                                 objective_function=cost_function, 
#                                 initial_point=initial_params)
# opt_params = opt_result[0]
# print("Optimized parameters:", opt_params)

# # Evaluate final cost.
# final_cost = cost_function(opt_params)
# print("Final cost:", final_cost)

# # Run circuit with optimized parameters to get measurement counts.
# param_dict = {}
# for i in range(num_qubits):
#     param_dict[theta[i]] = opt_params[i]
#     param_dict[phi[i]] = opt_params[i + num_qubits]
# qc_opt = qc.bind_parameters(param_dict)
# compiled_opt = transpile(qc_opt, simulator)
# job = simulator.run(compiled_opt, shots=1024)
# result = job.result()
# counts = result.get_counts()

# print("Optimized circuit measurement counts:", counts)
# plot_histogram(counts)
# plt.show()



import numpy as np
from transformers import pipeline
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_bloch_multivector
import matplotlib.pyplot as plt
from qiskit.circuit import ParameterVector
from scipy.optimize import minimize

##############################
# 1. Get Text Embedding (BERT)
##############################

# Create a feature extraction pipeline using BERT.
feature_extractor = pipeline("feature-extraction", model="bert-base-uncased", tokenizer="bert-base-uncased")

def get_text_embedding(text):
    """Extracts the [CLS] token embedding from text using BERT."""
    outputs = feature_extractor(text)
    # Use the [CLS] token embedding (first token)
    cls_embedding = np.array(outputs[0][0])
    return cls_embedding

# Example text
text = "Quantum computing and machine learning converge in hybrid systems."
embedding = get_text_embedding(text)
print("Original embedding shape:", embedding.shape)  # Typically (768,)

###############################################
# 2. Reduce Dimensions & Normalize to 8 Numbers
###############################################

# For our toy example, we slice the first 8 components.
# In practice, you'd use PCA or an autoencoder.
embedding_reduced = embedding[:8]
print("Reduced embedding (8 dims):", embedding_reduced)

# Normalize to create a valid quantum state (|ψ⟩ with ∑|aᵢ|² = 1)
norm = np.linalg.norm(embedding_reduced)
target_state = embedding_reduced / norm
print("Target state vector:", target_state)

###############################################
# 3. Define a Variational Quantum Circuit Ansatz
###############################################

# We use 3 qubits to encode 8 amplitudes.
num_qubits = 3
# We'll use 2 parameters per qubit: one for Ry and one for Rz.
param_count = num_qubits * 2

def create_variational_circuit(params):
    """Creates a parameterized quantum circuit (ansatz) using Ry and Rz gates."""
    qc = QuantumCircuit(num_qubits)
    # For each qubit, apply an Ry and an Rz with the corresponding parameters.
    for i in range(num_qubits):
        qc.ry(params[i], i)
        qc.rz(params[i + num_qubits], i)
    return qc

###############################################
# 4. Define the Cost Function (1 - Fidelity)
###############################################

def cost_function(params):
    """
    Cost is defined as 1 - fidelity between the circuit state and target_state.
    We compute the statevector using the statevector simulator.
    """
    qc = create_variational_circuit(params)
    simulator = AerSimulator(method="statevector")
    compiled_qc = transpile(qc, simulator)
    result = simulator.run(compiled_qc).result()
    statevector = result.get_statevector(compiled_qc)
    # Fidelity: |⟨target|ψ⟩|^2
    fidelity = np.abs(np.vdot(target_state, statevector))**2
    return 1 - fidelity

###############################################
# 5. Optimize Circuit Parameters
###############################################

# Random initial guess for parameters.
initial_params = np.random.rand(param_count)
# Optimize using COBYLA (a derivative-free method)
res = minimize(cost_function, initial_params, method="COBYLA")
optimized_params = res.x
print("Optimized parameters:", optimized_params)
print("Final cost (1 - fidelity):", res.fun)

###############################################
# 6. Create and Simulate the Optimized Circuit
###############################################

optimized_circuit = create_variational_circuit(optimized_params)
optimized_circuit.draw(output='mpl')
plt.show()

# Simulate the optimized circuit and get the final statevector.
simulator = AerSimulator(method="statevector")
compiled_qc = transpile(optimized_circuit, simulator)
result = simulator.run(compiled_qc).result()
final_state = result.get_statevector(compiled_qc)
print("Final statevector:", final_state)

# Visualize the final state on the Bloch sphere for each qubit.
plot_bloch_multivector(final_state)
plt.show()

