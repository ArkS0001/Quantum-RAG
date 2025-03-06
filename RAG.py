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
#     Returns a 768-dimensional numpy vector.
#     """
#     outputs = feature_extractor(text)
#     # Use the [CLS] token embedding (first token)
#     cls_embedding = np.array(outputs[0][0])
#     return cls_embedding

# ###############################################
# # 2. Dimensionality Reduction and Normalization
# ###############################################

# def reduce_and_normalize(embedding, target_dim=8):
#     """
#     Reduces the embedding to target_dim dimensions by slicing
#     and normalizes it so that the sum of squares equals 1.
#     """
#     reduced = embedding[:target_dim]
#     norm = np.linalg.norm(reduced)
#     state_vector = reduced / norm
#     return state_vector

# ###############################################
# # 3. Quantum Similarity via Swap Test Function
# ###############################################

# def compute_swap_similarity(state_vector_query, state_vector_doc, shots=2048):
#     """
#     Builds a swap test circuit to compare two amplitude-encoded states.
#     Both state vectors must be 8-dimensional (encoded on 3 qubits each).
#     The circuit uses one ancilla qubit (total 7 qubits).
#     Returns the estimated squared inner product (similarity) between the states.
#     """
#     # Create a quantum circuit with 7 qubits (1 ancilla + 3 for query + 3 for document)
#     qc = QuantumCircuit(7, 1)
    
#     # Qubit assignment:
#     # Qubit 0: Ancilla.
#     # Qubits 1-3: Query state.
#     # Qubits 4-6: Document state.
#     qc.initialize(state_vector_query, [1, 2, 3])
#     qc.initialize(state_vector_doc, [4, 5, 6])
    
#     # Swap test:
#     qc.h(0)  # Hadamard on ancilla.
#     for i in range(3):
#         qc.cswap(0, 1+i, 4+i)  # Controlled-swap on each corresponding pair.
#     qc.h(0)  # Final Hadamard on ancilla.
    
#     qc.measure(0, 0)  # Measure the ancilla.
    
#     # Uncomment the next two lines to visualize the circuit.
#     # qc.draw(output='mpl')
#     # plt.show()
    
#     # Simulation:
#     simulator = AerSimulator()
#     qc_transpiled = transpile(qc, simulator)
#     job = simulator.run(qc_transpiled, shots=shots)
#     result = job.result()
#     counts = result.get_counts(qc)
    
#     # Compute probability of measuring '0' on the ancilla.
#     p0 = counts.get('0', 0) / shots
#     # In a swap test, P(0) = 1/2 + 1/2 * |<psi|phi>|^2.
#     similarity = 2 * p0 - 1
#     return similarity

# ###############################################
# # 4. Build a Vast Knowledge Base
# ###############################################

# # For demonstration, we simulate a vast knowledge base as a list of many text documents.
# knowledge_base = [
#     "Quantum computing is the future of processing and secure communications.",
#     "Machine learning algorithms improve data analysis and predictions.",
#     "Classical computing remains the backbone of traditional technologies.",
#     "Neural networks have advanced the field of artificial intelligence.",
#     "Quantum entanglement is a core phenomenon in quantum physics.",
#     "Superconductors have applications in quantum devices and medical imaging.",
#     "Cryptography is evolving with the advent of quantum algorithms.",
#     "Artificial intelligence is transforming industries and research.",
#     "Hybrid quantum-classical systems may solve complex optimization problems.",
#     "Big data analytics is essential for modern business intelligence.",
#     "Quantum sensors provide enhanced precision in measurement.",
#     "Distributed computing leverages networked resources for efficiency.",
#     "Reinforcement learning enables autonomous systems to learn from experience.",
#     "Quantum machine learning merges quantum computing with deep learning.",
#     "The Internet of Things connects devices and enables smart systems.",
#     "Cloud computing offers scalable resources on demand.",
#     "Advanced robotics integrates AI with mechanical engineering.",
#     "Augmented reality blends digital content with the physical world.",
#     "Edge computing processes data near the source for speed.",
#     "Nanotechnology creates materials and devices at the atomic scale.",
#     # ... Add more documents as needed to simulate a vast knowledge base.
# ]

# # Precompute the quantum state (normalized 8D vector) for each document.
# doc_states = []
# for doc in knowledge_base:
#     emb = get_text_embedding(doc)
#     state = reduce_and_normalize(emb, target_dim=8)
#     doc_states.append(state)

# ###############################################
# # 5. User Input Query and Retrieval
# ###############################################

# # Let the user input a query.
# query_text = input("Enter your query: ")

# # Get and process the query embedding.
# query_embedding = get_text_embedding(query_text)
# query_state = reduce_and_normalize(query_embedding, target_dim=8)

# # Compute similarity between the query and each document using the swap test.
# similarities = []
# for idx, doc_state in enumerate(doc_states):
#     sim = compute_swap_similarity(query_state, doc_state)
#     similarities.append(sim)
#     print(f"Document {idx} similarity: {sim:.4f}")

# # Retrieve and display the document with the highest similarity.
# best_match_index = np.argmax(similarities)
# print("\nBest matching document:")
# print(knowledge_base[best_match_index])



import numpy as np
from transformers import pipeline
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

##############################################
# 1. Classical Embedding with BERT & PCA
##############################################

# Initialize the BERT feature extraction pipeline.
feature_extractor = pipeline("feature-extraction", 
                             model="bert-base-uncased", 
                             tokenizer="bert-base-uncased")

def get_text_embedding(text):
    """
    Uses BERT to extract the [CLS] token embedding for a given text.
    Returns a 768-dimensional numpy array.
    """
    outputs = feature_extractor(text)
    cls_embedding = np.array(outputs[0][0])
    return cls_embedding

# Define a large knowledge base (for demonstration, 20 sample documents).
knowledge_base = [
    "Quantum computing harnesses quantum mechanics to perform computations.",
    "Machine learning algorithms have transformed data analysis.",
    "Classical computers are based on binary logic and circuits.",
    "Neural networks are powerful tools in artificial intelligence.",
    "Quantum entanglement plays a key role in quantum communication.",
    "Hybrid quantum-classical systems combine strengths from both worlds.",
    "Deep learning is a subset of machine learning that uses neural networks.",
    "The future of computing may involve quantum processors.",
    "Artificial intelligence is transforming industries worldwide.",
    "Quantum supremacy promises exponential speedups over classical methods.",
    "Data science involves statistics, machine learning, and computer science.",
    "High-performance computing is essential for large-scale simulations.",
    "Cloud computing enables scalable resources on demand.",
    "Quantum sensors offer high-precision measurements.",
    "Cryptography is evolving with the advent of quantum algorithms.",
    "Reinforcement learning is used for training autonomous systems.",
    "Natural language processing enables machines to understand human language.",
    "Quantum error correction is vital for reliable quantum computation.",
    "Superconducting circuits are one of the platforms for quantum computers.",
    "Quantum machine learning may accelerate future data processing."
]

# Compute 768-dimensional embeddings for all documents.
embeddings = np.array([get_text_embedding(doc) for doc in knowledge_base])
print("Computed embeddings shape:", embeddings.shape)  # (n_docs, 768)

# Use PCA to reduce dimensions from 768 to 8.
pca = PCA(n_components=8)
embeddings_reduced = pca.fit_transform(embeddings)
print("Reduced embeddings shape:", embeddings_reduced.shape)

def normalize_vector(vec):
    """Normalizes a vector so that the sum of squares equals 1."""
    norm = np.linalg.norm(vec)
    return vec / norm if norm != 0 else vec

# Normalize each reduced embedding.
doc_states = np.array([normalize_vector(vec) for vec in embeddings_reduced])

##############################################
# 2. Quantum Swap Test Function
##############################################

def compute_swap_similarity(state_vector_query, state_vector_doc, shots=2048):
    """
    Constructs a quantum circuit to perform the swap test between two
    8-dimensional amplitude-encoded states (using 3 qubits each).
    Returns the estimated squared inner product between the states.
    """
    # Total qubits: 1 ancilla + 3 for query + 3 for document = 7 qubits.
    qc = QuantumCircuit(7, 1)
    
    # Qubit mapping:
    # Qubit 0: Ancilla.
    # Qubits 1-3: Query state.
    # Qubits 4-6: Document state.
    qc.initialize(state_vector_query, [1, 2, 3])
    qc.initialize(state_vector_doc, [4, 5, 6])
    
    # Swap test:
    qc.h(0)  # Hadamard on ancilla.
    for i in range(3):
        qc.cswap(0, 1 + i, 4 + i)  # Controlled-swap on each corresponding pair.
    qc.h(0)  # Final Hadamard on ancilla.
    
    qc.measure(0, 0)  # Measure ancilla.
    
    # Uncomment to visualize the circuit:
    # qc.draw(output='mpl')
    # plt.show()
    
    # Run simulation.
    simulator = AerSimulator()
    qc_transpiled = transpile(qc, simulator)
    job = simulator.run(qc_transpiled, shots=shots)
    result = job.result()
    counts = result.get_counts(qc)
    
    # Compute probability of measuring '0' on the ancilla.
    p0 = counts.get('0', 0) / shots
    # From the swap test: P(0) = 1/2 + 1/2 * |<psi|phi>|^2,
    # so the similarity (squared inner product) is:
    similarity = 2 * p0 - 1
    return similarity

##############################################
# 3. User Query & Retrieval
##############################################

# Prompt the user for a query.
query_text = input("Enter your query: ")

# Compute and reduce the query embedding using the same PCA.
query_embedding = get_text_embedding(query_text)
query_embedding_reduced = pca.transform(query_embedding.reshape(1, -1))[0]
query_state = normalize_vector(query_embedding_reduced)

# Compute similarities between the query and each document via swap test.
similarities = []
for idx, doc_state in enumerate(doc_states):
    sim = compute_swap_similarity(query_state, doc_state)
    similarities.append(sim)
    print(f"Document {idx} similarity: {sim:.4f}")

# Retrieve and display the best matching document.
best_match_index = np.argmax(similarities)
print("\nBest matching document:")
print(knowledge_base[best_match_index])

# Optionally, plot the similarity histogram for the best match swap test:
# (Uncomment the following lines to see the histogram for the best match.)
# best_qc = QuantumCircuit(7, 1)
# best_qc.initialize(query_state, [1, 2, 3])
# best_qc.initialize(doc_states[best_match_index], [4, 5, 6])
# best_qc.h(0)
# for i in range(3):
#     best_qc.cswap(0, 1 + i, 4 + i)
# best_qc.h(0)
# best_qc.measure(0, 0)
# counts = AerSimulator().run(transpile(best_qc, AerSimulator()), shots=2048).result().get_counts()
# plot_histogram(counts)
# plt.show()
