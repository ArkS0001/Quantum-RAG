# import numpy as np
# from transformers import pipeline
# from qiskit import QuantumCircuit
# import matplotlib.pyplot as plt

# # -----------------------------
# # 1. Convert Text to Classical Embedding Using BERT
# # -----------------------------

# # Create a feature-extraction pipeline with a pre-trained BERT model.
# feature_extractor = pipeline("feature-extraction", model="bert-base-uncased", tokenizer="bert-base-uncased")

# def get_text_embedding(text):
#     """
#     Extracts a text embedding from the [CLS] token using BERT.
#     The output is a high-dimensional vector (typically 768 dimensions).
#     """
#     outputs = feature_extractor(text)
#     # outputs is a nested list of shape [1, sequence_length, hidden_size]
#     # We use the [CLS] token embedding (first token)
#     cls_embedding = np.array(outputs[0][0])
#     return cls_embedding

# # Sample text to encode
# text = "Hello, quantum world! This is a test of converting text to a quantum state."

# # Get the BERT embedding (768-dimensional vector)
# embedding = get_text_embedding(text)
# print("Original embedding shape:", embedding.shape)

# # -----------------------------
# # 2. Dimensionality Reduction and Normalization
# # -----------------------------
# # For our toy example, we reduce the dimensionality to the number of qubits we can use.
# # Here, we simply take the first N components. In practice, use PCA or other methods.
# num_qubits = 4  # For example, encode into a 4-qubit state
# embedding_reduced = embedding[:num_qubits]

# # Normalize the reduced embedding to [0, 1] range.
# min_val = np.min(embedding_reduced)
# max_val = np.max(embedding_reduced)
# embedding_norm = (embedding_reduced - min_val) / (max_val - min_val)
# print("Reduced & normalized embedding:", embedding_norm)

# # Map the normalized values to angles in the range [0, π].
# angles = embedding_norm * np.pi
# print("Rotation angles (radians):", angles)

# # -----------------------------
# # 3. Quantum Encoding: Map the Angles to a Quantum Circuit
# # -----------------------------
# # Create a quantum circuit with 'num_qubits' qubits.
# qc = QuantumCircuit(num_qubits)

# # Encode the embedding into the quantum state by applying Ry rotations.
# for i, angle in enumerate(angles):
#     qc.ry(angle, i)

# # Visualize the quantum circuit.
# qc.draw(output='mpl')
# plt.show()



import numpy as np
from transformers import pipeline
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

###########################
# 1. Get Text Embedding   #
###########################

# Create a feature-extraction pipeline with a pre-trained BERT model.
feature_extractor = pipeline("feature-extraction", model="bert-base-uncased", tokenizer="bert-base-uncased")

def get_text_embedding(text):
    """
    Extracts the [CLS] token embedding from the text using BERT.
    Returns a 768-dimensional vector.
    """
    outputs = feature_extractor(text)
    # Use the [CLS] token embedding (first token)
    cls_embedding = np.array(outputs[0][0])
    return cls_embedding

# Example text
text = "Quantum computing and machine learning converge in hybrid systems."
embedding = get_text_embedding(text)
print("Original embedding shape:", embedding.shape)  # Typically (768,)

###########################
# 2. Dimensionality Reduction to 8 Dimensions
###########################

# Since we have only one sample, we can't use PCA.
# Instead, we slice the embedding to take the first 8 features.
embedding_reduced = embedding[:8]
print("Reduced embedding vector (8 dimensions):", embedding_reduced)

###########################
# 3. Normalize the Vector for Amplitude Encoding
###########################

norm = np.linalg.norm(embedding_reduced)
state_vector_normalized = embedding_reduced / norm
print("Normalized state vector:", state_vector_normalized)
# Now, ∑|a_i|² = 1

###########################
# 4. Quantum Encoding via Amplitude Encoding
###########################

# Create a quantum circuit with 3 qubits (since 2^3 = 8)
qc = QuantumCircuit(3, 3)

# Initialize the circuit with the normalized state vector.
qc.initialize(state_vector_normalized, [0, 1, 2])

# Optionally, add measurements to verify the distribution.
qc.measure([0, 1, 2], [0, 1, 2])

# Draw the circuit.
qc.draw(output='mpl')
plt.show()

###########################
# 5. Simulate the Circuit
###########################

simulator = AerSimulator()
compiled_circuit = transpile(qc, simulator)
job = simulator.run(compiled_circuit, shots=1024)
result = job.result()
counts = result.get_counts()
print("Measurement counts:", counts)
plot_histogram(counts)
plt.show()


