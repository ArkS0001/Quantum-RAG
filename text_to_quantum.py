import numpy as np
from transformers import pipeline
from qiskit import QuantumCircuit
import matplotlib.pyplot as plt

# -----------------------------
# 1. Convert Text to Classical Embedding Using BERT
# -----------------------------

# Create a feature-extraction pipeline with a pre-trained BERT model.
feature_extractor = pipeline("feature-extraction", model="bert-base-uncased", tokenizer="bert-base-uncased")

def get_text_embedding(text):
    """
    Extracts a text embedding from the [CLS] token using BERT.
    The output is a high-dimensional vector (typically 768 dimensions).
    """
    outputs = feature_extractor(text)
    # outputs is a nested list of shape [1, sequence_length, hidden_size]
    # We use the [CLS] token embedding (first token)
    cls_embedding = np.array(outputs[0][0])
    return cls_embedding

# Sample text to encode
text = "Hello, quantum world! This is a test of converting text to a quantum state."

# Get the BERT embedding (768-dimensional vector)
embedding = get_text_embedding(text)
print("Original embedding shape:", embedding.shape)

# -----------------------------
# 2. Dimensionality Reduction and Normalization
# -----------------------------
# For our toy example, we reduce the dimensionality to the number of qubits we can use.
# Here, we simply take the first N components. In practice, use PCA or other methods.
num_qubits = 4  # For example, encode into a 4-qubit state
embedding_reduced = embedding[:num_qubits]

# Normalize the reduced embedding to [0, 1] range.
min_val = np.min(embedding_reduced)
max_val = np.max(embedding_reduced)
embedding_norm = (embedding_reduced - min_val) / (max_val - min_val)
print("Reduced & normalized embedding:", embedding_norm)

# Map the normalized values to angles in the range [0, π].
angles = embedding_norm * np.pi
print("Rotation angles (radians):", angles)

# -----------------------------
# 3. Quantum Encoding: Map the Angles to a Quantum Circuit
# -----------------------------
# Create a quantum circuit with 'num_qubits' qubits.
qc = QuantumCircuit(num_qubits)

# Encode the embedding into the quantum state by applying Ry rotations.
for i, angle in enumerate(angles):
    qc.ry(angle, i)

# Visualize the quantum circuit.
qc.draw(output='mpl')
plt.show()
