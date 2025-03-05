from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

def build_oracle(qc, target):
    """
    Marks the target state by flipping its phase.
    
    For each qubit, if the target bit is '0', an X gate is applied.
    Then, a multi-controlled-Z is implemented (via an H, MCX, H sandwich)
    to flip the phase of the state.
    Finally, the X gates are undone.
    """
    num_qubits = len(target)
    # Flip bits where target is 0
    for i, bit in enumerate(target):
        if bit == '0':
            qc.x(i)
    # Multi-controlled-Z on the last qubit
    qc.h(num_qubits - 1)
    qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    qc.h(num_qubits - 1)
    # Undo the flips
    for i, bit in enumerate(target):
        if bit == '0':
            qc.x(i)
    return qc

def quantum_search(query, shots=1024):
    """
    Uses Grover's algorithm to search for a target query in an embedding database.
    
    For a toy example, the query is given as a binary string (e.g. "101").
    The circuit is built on len(query) qubits, placing the system into equal superposition,
    applying an oracle to mark the target, and then a Grover diffusion operator.
    """
    num_qubits = len(query)
    qc = QuantumCircuit(num_qubits, num_qubits)
    
    # Step 1: Initialize uniform superposition (all embeddings equally likely)
    qc.h(range(num_qubits))
    
    # Step 2: Oracle that marks the target state (query)
    build_oracle(qc, query)
    
    # Step 3: Grover Diffusion Operator
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    qc.h(num_qubits - 1)
    qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    qc.h(num_qubits - 1)
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))
    
    # Step 4: Measure all qubits
    qc.measure(range(num_qubits), range(num_qubits))
    
    # Simulation using Qiskit Aer
    simulator = AerSimulator()
    compiled_circuit = transpile(qc, simulator)
    job = simulator.run(compiled_circuit, shots=shots)
    result = job.result()
    counts = result.get_counts()
    return counts

# A toy embedding database mapping binary strings to document names.
dataset = {
    "000": "Document A",
    "001": "Document B",
    "010": "Document C",
    "011": "Document D",
    "100": "Document E",
    "101": "Document F",  # Suppose this is our target document.
    "110": "Document G",
    "111": "Document H",
}

# Example query: a binary string representing the target embedding.
query = "101"
counts = quantum_search(query, shots=1024)

print("Quantum search results:", counts)
plot_histogram(counts)
plt.show()

# Determine the most frequently measured state.
most_freq_state = max(counts, key=counts.get)
print("Most frequent state:", most_freq_state)
print("Retrieved document:", dataset.get(most_freq_state, "Not found"))
