from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit.primitives import Sampler
from qiskit.visualization import plot_histogram
import numpy as np
import matplotlib.pyplot as plt
import random

# Function to create an N-qubit GHZ state
def create_ghz_state(n):
    qc = QuantumCircuit(n, n)

    # Step 1: Create GHZ entanglement (generalized Bell state)
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)

    return qc

# Function to create an advanced multi-qubit Bell test circuit
def create_advanced_bell_test(n):
    qc = create_ghz_state(n)  # Start with an N-qubit GHZ state

    # Step 2: Apply randomized measurement bases
    for i in range(n):
        theta = random.choice([0, np.pi / 4, np.pi / 8, -np.pi / 8])  # Random basis
        qc.ry(2 * theta, i)

    # Step 3: Measure all qubits
    qc.measure(range(n), range(n))
    return qc

# Define number of qubits
num_qubits = 4  # Modify this for larger entangled states

# Create multiple randomized Bell test circuits
num_tests = 4
circuits = {f"Test_{i}": create_advanced_bell_test(num_qubits) for i in range(num_tests)}

# Draw one example circuit
circuits["Test_0"].draw(output='mpl')
plt.show()

# Use Qiskit Sampler to run circuits
sampler = Sampler()

# Run the circuits
results = {}
for key, circuit in circuits.items():
    job = sampler.run(circuit)
    result = job.result()
    counts = result.quasi_dists[0] 
    results[key] = counts

# Print and visualize results
for key, counts in results.items():
    print(f"Results for {key}: {counts}")

# Show one example histogram
plot_histogram(results["Test_0"])
plt.show()
