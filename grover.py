# from qiskit import QuantumCircuit, transpile
# from qiskit_aer import AerSimulator
# from qiskit.visualization import plot_histogram
# import matplotlib.pyplot as plt

# # Number of qubits (Grover’s search space is 2^n)
# num_qubits = 3  # Searching in 2^3 = 8 elements

# # Step 1: Create a Quantum Circuit
# qc = QuantumCircuit(num_qubits, num_qubits)

# # Apply Hadamard to all qubits (Superposition)
# qc.h(range(num_qubits))

# # Step 2: Oracle - Flip the marked state (Example: |101⟩)
# qc.cz(0, 2)  # Example of marking |101>
# qc.cz(1, 2)  
# # Step 3: Grover Diffusion Operator
# qc.h(range(num_qubits))
# qc.x(range(num_qubits))
# qc.h(num_qubits - 1)
# qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)  # Multi-controlled X
# qc.h(num_qubits - 1)
# qc.x(range(num_qubits))
# qc.h(range(num_qubits))

# # Step 4: Measure all qubits
# qc.measure(range(num_qubits), range(num_qubits))

# # Use Qiskit Aer Simulator
# simulator = AerSimulator()  # Using Qiskit Aer for high-performance simulation
# compiled_circuit = transpile(qc, simulator)

# # Run the circuit on the simulator
# job = simulator.run(compiled_circuit, shots=1024)
# result = job.result()

# # Get and plot the results
# counts = result.get_counts()
# print("Grover's Algorithm Results:", counts)
# plot_histogram(counts)
# plt.show()



# from qiskit import QuantumCircuit, transpile
# from qiskit_aer import AerSimulator
# from qiskit.visualization import plot_histogram
# import matplotlib.pyplot as plt

# # Number of qubits (3 qubits represent an 8-element vector space)
# num_qubits = 3

# # Create a Quantum Circuit with 3 qubits and 3 classical bits
# qc = QuantumCircuit(num_qubits, num_qubits)

# # --- Step 1: Create a uniform superposition ---
# # Apply Hadamard gates to all qubits so that each basis state (each vector) 
# # has equal amplitude.
# qc.h(range(num_qubits))

# # --- Step 2: Oracle to mark the target state |101⟩ ---
# # For a target |101⟩, note:
# #   - Qubit 0 should be |1⟩ (no change needed)
# #   - Qubit 1 should be |0⟩ (so we apply an X gate to flip it, marking 0)
# #   - Qubit 2 should be |1⟩ (no change needed)
# # Then, we use a multi-controlled Z gate (implemented with H, multi-controlled-X, and H)
# # to flip the phase of the target state.

# # Invert qubits where target bit is 0 (here, qubit 1)
# qc.x(1)

# # Apply the multi-controlled-Z gate
# qc.h(num_qubits - 1)                          # Put last qubit into basis for MCX
# qc.mcx([0, 1], num_qubits - 1)                 # Multi-controlled-X (target: qubit 2)
# qc.h(num_qubits - 1)                          # Bring back to Z basis

# # Revert the earlier inversion on qubit 1
# qc.x(1)

# # --- Step 3: Grover Diffusion Operator ---
# # This operator "inverts about the mean" to amplify the amplitude of the marked state.
# qc.h(range(num_qubits))
# qc.x(range(num_qubits))
# qc.h(num_qubits - 1)
# qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
# qc.h(num_qubits - 1)
# qc.x(range(num_qubits))
# qc.h(range(num_qubits))

# # --- Step 4: Measurement ---
# qc.measure(range(num_qubits), range(num_qubits))

# # --- Simulation using Qiskit Aer ---
# simulator = AerSimulator()  # Use Qiskit Aer simulator for high-performance simulation
# compiled_circuit = transpile(qc, simulator)
# job = simulator.run(compiled_circuit, shots=1024)
# result = job.result()
# counts = result.get_counts()

# # Print and plot the results
# print("Grover's Algorithm Results:", counts)
# plot_histogram(counts)
# plt.show()



from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

# Number of qubits: 3 qubits represent an 8-dimensional embedding space.
num_qubits = 3

# Define the target embedding as a basis state (e.g., target vector = |101⟩)
target_state = '101'

def build_oracle(qc, target):
    """
    This oracle flips the phase of the target state.
    
    For each qubit, if the corresponding target bit is 0, an X gate is applied.
    Then a multi-controlled Z (implemented via H, MCX, H) is applied.
    Finally, the X gates are reverted.
    """
    # Apply X gates to qubits where target has a '0'
    for i, bit in enumerate(target):
        if bit == '0':
            qc.x(i)
    # Apply a multi-controlled-Z gate:
    # Convert the Z to an X by sandwiching it with H on the target qubit.
    qc.h(num_qubits - 1)
    qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)  # Multi-controlled X (target: last qubit)
    qc.h(num_qubits - 1)
    # Revert the X gates
    for i, bit in enumerate(target):
        if bit == '0':
            qc.x(i)
    return qc

# Create a quantum circuit with 3 qubits and 3 classical bits.
qc = QuantumCircuit(num_qubits, num_qubits)

# Step 1: Initialize the state in uniform superposition (all embeddings equally likely)
qc.h(range(num_qubits))

# Step 2: Oracle that marks the target embedding (|101⟩ in this example)
build_oracle(qc, target_state)

# Step 3: Grover Diffusion Operator (inverts amplitudes about the mean)
qc.h(range(num_qubits))
qc.x(range(num_qubits))
qc.h(num_qubits - 1)
qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
qc.h(num_qubits - 1)
qc.x(range(num_qubits))
qc.h(range(num_qubits))

# Step 4: Measure all qubits
qc.measure(range(num_qubits), range(num_qubits))

# --- Simulation using Qiskit Aer ---
simulator = AerSimulator()  # Use high-performance simulator from qiskit_aer
compiled_circuit = transpile(qc, simulator)
job = simulator.run(compiled_circuit, shots=1024)
result = job.result()
counts = result.get_counts()

print("Quantum search results for target embedding (|101>):", counts)
plot_histogram(counts)
plt.show()
