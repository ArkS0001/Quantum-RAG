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



# import numpy as np
# from transformers import pipeline
# from qiskit import QuantumCircuit, transpile
# from qiskit_aer import AerSimulator
# from qiskit.visualization import plot_histogram
# import matplotlib.pyplot as plt

# ###########################
# # 1. Get Text Embedding   #
# ###########################

# # Create a feature-extraction pipeline with a pre-trained BERT model.
# feature_extractor = pipeline("feature-extraction", model="bert-base-uncased", tokenizer="bert-base-uncased")

# def get_text_embedding(text):
#     """
#     Extracts the [CLS] token embedding from the text using BERT.
#     Returns a 768-dimensional vector.
#     """
#     outputs = feature_extractor(text)
#     # Use the [CLS] token embedding (first token)
#     cls_embedding = np.array(outputs[0][0])
#     return cls_embedding

# # Example text
# text = "Quantum computing and machine learning converge in hybrid systems."
# embedding = get_text_embedding(text)
# print("Original embedding shape:", embedding.shape)  # Typically (768,)

# ###########################
# # 2. Dimensionality Reduction to 8 Dimensions
# ###########################

# # Since we have only one sample, we can't use PCA.
# # Instead, we slice the embedding to take the first 8 features.
# embedding_reduced = embedding[:8]
# print("Reduced embedding vector (8 dimensions):", embedding_reduced)

# ###########################
# # 3. Normalize the Vector for Amplitude Encoding
# ###########################

# norm = np.linalg.norm(embedding_reduced)
# state_vector_normalized = embedding_reduced / norm
# print("Normalized state vector:", state_vector_normalized)
# # Now, ∑|a_i|² = 1

# ###########################
# # 4. Quantum Encoding via Amplitude Encoding
# ###########################

# # Create a quantum circuit with 3 qubits (since 2^3 = 8)
# qc = QuantumCircuit(3, 3)

# # Initialize the circuit with the normalized state vector.
# qc.initialize(state_vector_normalized, [0, 1, 2])

# # Optionally, add measurements to verify the distribution.
# qc.measure([0, 1, 2], [0, 1, 2])

# # Draw the circuit.
# qc.draw(output='mpl')
# plt.show()

# ###########################
# # 5. Simulate the Circuit
# ###########################

# simulator = AerSimulator()
# compiled_circuit = transpile(qc, simulator)
# job = simulator.run(compiled_circuit, shots=1024)
# result = job.result()
# counts = result.get_counts()
# print("Measurement counts:", counts)
# plot_histogram(counts)
# plt.show()


# import numpy as np
# from transformers import pipeline
# from qiskit import QuantumCircuit, transpile
# from qiskit_aer import AerSimulator
# from qiskit.visualization import plot_histogram
# import matplotlib.pyplot as plt

# ##############################
# # 1. Classical Text Embedding
# ##############################

# # Create a feature-extraction pipeline with a pre-trained BERT model.
# feature_extractor = pipeline("feature-extraction", 
#                              model="bert-base-uncased", 
#                              tokenizer="bert-base-uncased")

# def get_text_embedding(text):
#     """
#     Extracts the [CLS] token embedding from the text using BERT.
#     Returns a 768-dimensional vector.
#     """
#     outputs = feature_extractor(text)
#     # Use the [CLS] token embedding (first token)
#     cls_embedding = np.array(outputs[0][0])
#     return cls_embedding

# # Define two texts for retrieval comparison.
# text1 = "Quantum computing and machine learning converge in hybrid systems."
# text2 = "Hybrid systems combining quantum computing with machine learning show promising results."

# # Get the 768-dimensional embeddings for both texts.
# embedding1 = get_text_embedding(text1)
# embedding2 = get_text_embedding(text2)
# print("Original embedding shapes:", embedding1.shape, embedding2.shape)

# ##############################################
# # 2. Dimensionality Reduction to 8 Dimensions
# ##############################################

# # For simplicity, slice the first 8 components.
# embedding1_reduced = embedding1[:8]
# embedding2_reduced = embedding2[:8]
# print("Reduced embeddings (8 dimensions):")
# print("Embedding1:", embedding1_reduced)
# print("Embedding2:", embedding2_reduced)

# #################################################
# # 3. Normalization for Quantum Amplitude Encoding
# #################################################

# # Normalize each reduced vector so that the sum of squares equals 1.
# norm1 = np.linalg.norm(embedding1_reduced)
# norm2 = np.linalg.norm(embedding2_reduced)
# state_vector1 = embedding1_reduced / norm1
# state_vector2 = embedding2_reduced / norm2

# print("Normalized state vectors:")
# print("State vector 1:", state_vector1)
# print("State vector 2:", state_vector2)

# ###########################################################
# # 4. Quantum Encoding & Similarity Estimation via Swap Test
# ###########################################################

# # The swap test will compare two quantum states.
# # For 8-dimensional states, we need 3 qubits per state (since 2^3 = 8).
# # Total qubits = 1 ancilla + 3 for state1 + 3 for state2 = 7 qubits.
# # We will measure the ancilla to estimate the overlap.

# # Create a quantum circuit with 7 qubits and 1 classical bit.
# qc = QuantumCircuit(7, 1)

# # Qubit assignment:
# # Qubit 0: Ancilla for swap test.
# # Qubits 1-3: Register for state_vector1.
# # Qubits 4-6: Register for state_vector2.

# # Initialize registers with the normalized state vectors.
# qc.initialize(state_vector1, [1, 2, 3])
# qc.initialize(state_vector2, [4, 5, 6])

# # Begin swap test.
# qc.h(0)  # Apply Hadamard to ancilla.

# # Apply controlled-swap (Fredkin) gates for each pair of corresponding qubits.
# for i in range(3):
#     qc.cswap(0, 1+i, 4+i)

# qc.h(0)  # Apply another Hadamard to ancilla.

# # Measure the ancilla qubit.
# qc.measure(0, 0)

# # Visualize the quantum circuit.
# qc.draw(output='mpl')
# plt.show()

# ###########################################
# # 5. Simulation and Similarity Calculation
# ###########################################

# # Use Qiskit's AerSimulator to simulate the circuit.
# simulator = AerSimulator()
# qc_transpiled = transpile(qc, simulator)
# job = simulator.run(qc_transpiled, shots=2048)
# result = job.result()
# counts = result.get_counts(qc)
# print("Swap test measurement counts:", counts)

# # Plot a histogram of the measurement results.
# plot_histogram(counts)
# plt.show()

# # Calculate the probability of measuring '0' on the ancilla.
# p0 = counts.get('0', 0) / 2048
# # The swap test yields: P(0) = 1/2 + 1/2 * |<psi|phi>|^2,
# # so we estimate the squared inner product as:
# similarity = 2 * p0 - 1
# print(f"Estimated squared inner product (similarity): {similarity:.4f}")



import numpy as np
from transformers import pipeline
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

##############################
# 1. Classical Text Embedding
##############################

# Create a feature-extraction pipeline with a pre-trained BERT model.
feature_extractor = pipeline("feature-extraction", 
                             model="bert-base-uncased", 
                             tokenizer="bert-base-uncased")

def get_text_embedding(text):
    """
    Extracts the [CLS] token embedding from the text using BERT.
    Returns a 768-dimensional vector.
    """
    outputs = feature_extractor(text)
    # Use the [CLS] token embedding (first token)
    cls_embedding = np.array(outputs[0][0])
    return cls_embedding

###############################################
# 2. Dimensionality Reduction and Normalization
###############################################

def reduce_and_normalize(embedding, target_dim=8):
    """
    Reduces the embedding to target_dim dimensions by slicing
    and normalizes it so that the sum of squares is 1.
    """
    reduced = embedding[:target_dim]
    norm = np.linalg.norm(reduced)
    state_vector = reduced / norm
    return state_vector

###############################################
# 3. Quantum Similarity via Swap Test Function
###############################################

def compute_swap_similarity(state_vector_query, state_vector_doc, shots=2048):
    """
    Builds a swap test circuit to compare two amplitude-encoded states.
    Both state vectors are assumed to be 8-dimensional, so they are encoded on 3 qubits each.
    The circuit uses one ancilla qubit (total 7 qubits).
    Returns the estimated squared inner product (similarity).
    """
    # Total qubits: 1 ancilla + 3 for query + 3 for document = 7 qubits.
    qc = QuantumCircuit(7, 1)
    
    # Qubit assignment:
    # Qubit 0: Ancilla for swap test.
    # Qubits 1-3: Register for the query state.
    # Qubits 4-6: Register for the document state.
    qc.initialize(state_vector_query, [1, 2, 3])
    qc.initialize(state_vector_doc, [4, 5, 6])
    
    # Swap test procedure:
    qc.h(0)  # Put the ancilla into superposition.
    for i in range(3):
        qc.cswap(0, 1 + i, 4 + i)  # Controlled-swap between corresponding qubits.
    qc.h(0)  # Interfere the ancilla.
    
    qc.measure(0, 0)  # Measure the ancilla.
    
    # Uncomment to visualize the circuit.
    # qc.draw(output='mpl')
    # plt.show()
    
    # Simulation using AerSimulator.
    simulator = AerSimulator()
    qc_transpiled = transpile(qc, simulator)
    job = simulator.run(qc_transpiled, shots=shots)
    result = job.result()
    counts = result.get_counts(qc)
    
    # Compute the probability of measuring '0' on the ancilla.
    p0 = counts.get('0', 0) / shots
    # From the swap test: P(0) = 1/2 + 1/2 * |<psi|phi>|^2,
    # so the similarity (squared inner product) is:
    similarity = 2 * p0 - 1
    return similarity

###############################################
# 4. Prepare the Corpus and Query
###############################################

# Define a corpus of text documents.
documents = [
    "Quantum computing and machine learning converge in hybrid systems.",
    "Classical computers are the foundation of modern computation.",
    "Machine learning algorithms can predict outcomes from data.",
    "Quantum entanglement is a fundamental resource in quantum information.",
    "Hybrid systems integrating quantum circuits and neural networks are emerging."
]

# Compute and store the normalized quantum state for each document.
doc_states = []
for doc in documents:
    emb = get_text_embedding(doc)
    state = reduce_and_normalize(emb, target_dim=8)
    doc_states.append(state)

# Define a query text.
query_text = "Find me a text about integrating quantum circuits with neural networks."
query_embedding = get_text_embedding(query_text)
query_state = reduce_and_normalize(query_embedding, target_dim=8)

###############################################
# 5. Compute Similarity and Retrieve the Best Match
###############################################

similarities = []
for idx, doc_state in enumerate(doc_states):
    sim = compute_swap_similarity(query_state, doc_state)
    similarities.append(sim)
    print(f"Document {idx} similarity: {sim:.4f}")

# Retrieve the document with the highest similarity.
best_match_index = np.argmax(similarities)
print("\nBest matching document:")
print(documents[best_match_index])
