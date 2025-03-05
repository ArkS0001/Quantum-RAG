# from qiskit import QuantumCircuit
# from qiskit.visualization import plot_circuit_layout
# import matplotlib.pyplot as plt

# q = QuantumCircuit(5)  # Initialize with 5 qubits

# q.h(0)
# q.cx(0,1)
# q.h(2)
# q.cx(1,2)
# # q.h(4)
# q.cx(2,3)
# q.cx(3,4)

# q.draw(output='mpl')  # This creates the plot
# plt.show()  # Display it in a command prompt




from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit.primitives import Sampler
from qiskit.visualization import plot_histogram
import numpy as np
import matplotlib.pyplot as plt

# Function to create a CHSH Bell Test circuit
def create_chsh_circuit(theta_a, theta_b):
    qc = QuantumCircuit(2, 2)  # 2 qubits, 2 classical bits

    # Step 1: Create an entangled Bell pair
    qc.h(0)
    qc.cx(0, 1)

    # Step 2: Apply measurement settings (rotations) based on angles
    qc.ry(2 * theta_a, 0)  # Alice's measurement basis
    qc.ry(2 * theta_b, 1)  # Bob's measurement basis

    # Step 3: Measure the qubits
    qc.measure([0, 1], [0, 1])

    return qc

# Define CHSH measurement settings
angle_settings = {
    "A0": 0,                # Alice's first setting (Z basis)
    "A1": np.pi / 4,        # Alice's second setting (X+Z basis)
    "B0": np.pi / 8,        # Bob's first setting
    "B1": -np.pi / 8        # Bob's second setting
}

# Create four circuits for different measurement settings
circuits = {
    "A0B0": create_chsh_circuit(angle_settings["A0"], angle_settings["B0"]),
    "A0B1": create_chsh_circuit(angle_settings["A0"], angle_settings["B1"]),
    "A1B0": create_chsh_circuit(angle_settings["A1"], angle_settings["B0"]),
    "A1B1": create_chsh_circuit(angle_settings["A1"], angle_settings["B1"])
}

# Draw one example circuit
circuits["A0B0"].draw(output='mpl')
plt.show()

# Use Qiskit Sampler to run circuits (New Qiskit 1.0+ method)
sampler = Sampler()

# Run the circuits
results = {}
for key, circuit in circuits.items():
    job = sampler.run(circuit)
    result = job.result()
    counts = result.quasi_dists[0]  # Qiskit 1.0+ returns quasi-probabilities
    results[key] = counts

# Print and visualize results
for key, counts in results.items():
    print(f"Results for {key}: {counts}")

# Show one example histogram
plot_histogram(results["A0B0"])
plt.show()
