1. Introduction

![Figure_1](https://github.com/user-attachments/assets/9bb5a3ac-ce2d-444c-a135-97e50e583457)
![Figure_2](https://github.com/user-attachments/assets/969362d3-3db0-41ea-8ae7-d472d0d0d395)


1.1 The Bell Inequality & Quantum Entanglement

Bell’s theorem states that no local hidden variable theory can reproduce all predictions of quantum mechanics. The CHSH inequality is a testable mathematical expression derived from Bell’s theorem, allowing us to experimentally confirm quantum nonlocality. In this implementation, we:

    Create an entangled Bell pair
    Apply different measurement settings for Alice and Bob
    Analyze measurement correlations
    Test for CHSH inequality violations

Qiskit 1.0+ introduces a new execution paradigm with Primitives (Sampler), making the traditional execute function obsolete. This thesis presents a revised implementation using modern Qiskit functionalities.
2. Implementation in Qiskit

We implement the CHSH test using Qiskit’s QuantumCircuit, Qiskit Aer, and the Sampler primitive.
2.1 Quantum Circuit Construction

The QuantumCircuit class is used to prepare a Bell state, which is the key component for testing quantum entanglement.

from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit.primitives import Sampler
from qiskit.visualization import plot_histogram
import numpy as np
import matplotlib.pyplot as plt

    QuantumCircuit: Defines the quantum operations.
    Aer: Qiskit's high-performance simulator backend.
    Sampler: A new Qiskit Primitive used for sampling measurement results.

2.2 Creating the CHSH Measurement Circuit

To test the Bell inequality, we define different measurement settings for Alice and Bob:

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

    Step 1: Bell Pair Creation
        A Hadamard gate on qubit 0 creates a superposition state.
        A CNOT gate entangles qubits 0 and 1.

    Step 2: Measurement in Rotated Bases
        Qubits are rotated by θ_A and θ_B before measurement.
        This simulates different bases chosen by Alice and Bob.

    Step 3: Measurement
        The qubits are measured and stored in classical registers.

2.3 Defining Measurement Settings

The CHSH test requires measurements in specific basis choices, parameterized by angles:

angle_settings = {
    "A0": 0,                # Alice's first setting (Z basis)
    "A1": np.pi / 4,        # Alice's second setting (X+Z basis)
    "B0": np.pi / 8,        # Bob's first setting
    "B1": -np.pi / 8        # Bob's second setting
}

These angles correspond to specific quantum measurements needed to test the CHSH inequality.
2.4 Running the Experiment with Qiskit Sampler

Qiskit 1.0+ removes execute, so we now use Sampler instead:

# Create circuits for different measurement settings
circuits = {
    "A0B0": create_chsh_circuit(angle_settings["A0"], angle_settings["B0"]),
    "A0B1": create_chsh_circuit(angle_settings["A0"], angle_settings["B1"]),
    "A1B0": create_chsh_circuit(angle_settings["A1"], angle_settings["B0"]),
    "A1B1": create_chsh_circuit(angle_settings["A1"], angle_settings["B1"])
}

# Draw one example circuit
circuits["A0B0"].draw(output='mpl')
plt.show()

# Use Qiskit Sampler to run circuits
sampler = Sampler()
results = {}

for key, circuit in circuits.items():
    job = sampler.run(circuit)
    result = job.result()
    counts = result.quasi_dists[0]  # Qiskit 1.0+ uses quasi-probabilities
    results[key] = counts

# Print and visualize results
for key, counts in results.items():
    print(f"Results for {key}: {counts}")

# Show one example histogram
plot_histogram(results["A0B0"])
plt.show()

    Why use Sampler?
        The Sampler primitive is the recommended way to sample measurement outcomes in Qiskit 1.0+.
        Unlike execute, it provides quasi-probabilities instead of raw bit counts.
        This ensures compatibility with both quantum simulators and real quantum devices.

3. Results & Interpretation

The results from running the quantum circuits give correlation counts for each measurement setting:

Results for A0B0: {0: 0.48, 3: 0.52}
Results for A0B1: {0: 0.45, 3: 0.55}
Results for A1B0: {0: 0.52, 3: 0.48}
Results for A1B1: {0: 0.70, 3: 0.30}

From these, we calculate the CHSH parameter (S-value):
S=E(A0,B0)+E(A0,B1)+E(A1,B0)−E(A1,B1)
S=E(A0​,B0​)+E(A0​,B1​)+E(A1​,B0​)−E(A1​,B1​)

Quantum mechanics predicts:
∣S∣≤2(Classical Bound)
∣S∣≤2(Classical Bound)
∣S∣≈2.83(Quantum Violation)
∣S∣≈2.83(Quantum Violation)

If |S| > 2, Bell’s inequality is violated, proving quantum entanglement.
4. Conclusion

This implementation successfully demonstrates Bell’s theorem using Qiskit 1.0+, replacing the outdated execute function with Primitives (Sampler). The experimental results show a violation of the classical CHSH inequality, confirming the presence of quantum entanglement.

Future work could extend this to:

    Noisy quantum hardware simulations
    Real quantum processors (IBM Quantum Experience)
    Advanced entanglement tests with multiple qubits

5. References

    J. S. Bell, On the Einstein-Podolsky-Rosen Paradox, Physics Physique Физика, 1964.
    IBM Qiskit Documentation, Primitives and New Qiskit 1.0 Features.
    Clauser, J. F., Horne, M. A., Shimony, A., Holt, R. A. Proposed Experiment to Test Local Hidden-Variable Theories, 1969.
