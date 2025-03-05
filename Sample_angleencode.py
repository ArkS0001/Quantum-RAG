import numpy as np
from qiskit import QuantumCircuit
import matplotlib.pyplot as plt

# Step 1: Classical Text Embedding
# For demonstration, assume we've obtained the following 4-dimensional embedding
# from some text embedding model (e.g., using a pre-trained BERT model or TF-IDF).
text_embedding = np.array([0.5, 1.0, 0.2, 0.8])

# Normalize the embedding vector to ensure proper scaling
norm = np.linalg.norm(text_embedding)
embedding_normalized = text_embedding / norm

# Step 2: Quantum Encoding using Angle Encoding
# Create a quantum circuit with as many qubits as the embedding dimension.
num_qubits = len(embedding_normalized)
qc = QuantumCircuit(num_qubits)

# Map each component of the embedding to a rotation around the Y-axis.
for i, angle_value in enumerate(embedding_normalized):
    # Multiply by pi (or any scaling factor) to map the normalized value to an angle
    qc.ry(angle_value * np.pi, i)

# Visualize the circuit
qc.draw(output='mpl')
plt.show()
