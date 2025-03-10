import numpy as np
from transformers import pipeline
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import math

# Try importing plot_histogram from Qiskit; if not available, define our own.
try:
    from qiskit.visualization import plot_histogram
except ImportError:
    def plot_histogram(counts):
        labels = list(counts.keys())
        values = list(counts.values())
        plt.bar(labels, values)
        plt.xlabel("Measurement Outcome")
        plt.ylabel("Counts")
        plt.show()

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

# Define a knowledge base (20 sample documents).
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
print("Computed embeddings shape:", embeddings.shape)  # Expected: (20, 768)

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
# 2. User Query & Classical Retrieval
##############################################

# Prompt the user for a query.
query_text = input("Enter your query: ")

# Compute and reduce the query embedding using the same PCA.
query_embedding = get_text_embedding(query_text)
query_embedding_reduced = pca.transform(query_embedding.reshape(1, -1))[0]
query_state = normalize_vector(query_embedding_reduced)

# Compute classical similarity using squared dot product.
similarities = []
for idx, doc_state in enumerate(doc_states):
    sim = np.dot(query_state, doc_state)**2  # squared inner product
    similarities.append(sim)
    print(f"Document {idx} similarity: {sim:.4f}")

# Identify the best matching document index.
best_match_index = np.argmax(similarities)
print("\nClassically determined best matching document index:", best_match_index)
print("Best matching document:")
print(knowledge_base[best_match_index])

##############################################
# 3. Grover Search for the Target Document
##############################################

# For a knowledge base of 20 documents, we need:
num_docs = len(knowledge_base)
num_qubits = math.ceil(math.log2(num_docs))  # 5 qubits for 20 docs.
print("\nUsing Grover search on", num_qubits, "qubits.")

# Convert the best match index to a binary string of length num_qubits.
target_binary = format(best_match_index, f'0{num_qubits}b')
print("Target binary representation:", target_binary)

# Build the oracle that marks the target state.
def build_oracle(target_bin, num_qubits):
    oracle = QuantumCircuit(num_qubits)
    # For each bit that is '0', apply an X gate.
    for i, bit in enumerate(target_bin):
        if bit == '0':
            oracle.x(i)
    # Apply a multi-controlled Z gate.
    oracle.h(num_qubits - 1)
    oracle.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    oracle.h(num_qubits - 1)
    # Undo the X gates.
    for i, bit in enumerate(target_bin):
        if bit == '0':
            oracle.x(i)
    oracle_gate = oracle.to_gate()
    oracle_gate.label = "Oracle"
    return oracle_gate

oracle_gate = build_oracle(target_binary, num_qubits)

# Build the diffuser (inversion about the mean).
def build_diffuser(num_qubits):
    diffuser = QuantumCircuit(num_qubits)
    diffuser.h(range(num_qubits))
    diffuser.x(range(num_qubits))
    diffuser.h(num_qubits - 1)
    diffuser.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    diffuser.h(num_qubits - 1)
    diffuser.x(range(num_qubits))
    diffuser.h(range(num_qubits))
    diffuser_gate = diffuser.to_gate()
    diffuser_gate.label = "Diffuser"
    return diffuser_gate

diffuser_gate = build_diffuser(num_qubits)

# Create the Grover search circuit.
grover_circuit = QuantumCircuit(num_qubits, num_qubits)
grover_circuit.h(range(num_qubits))  # Initialize in superposition

# Determine number of Grover iterations (for one marked element).
iterations = int(round((math.pi / 4) * math.sqrt(2 ** num_qubits)))
print("Number of Grover iterations:", iterations)

for _ in range(iterations):
    grover_circuit.append(oracle_gate, range(num_qubits))
    grover_circuit.append(diffuser_gate, range(num_qubits))

grover_circuit.measure(range(num_qubits), range(num_qubits))

# Visualize the Grover circuit.
grover_circuit.draw(output='mpl')
plt.show()

# Run the Grover search simulation using backend.run()
backend = AerSimulator()
qc_transpiled = transpile(grover_circuit, backend)
job = backend.run(qc_transpiled, shots=1024)
result = job.result()
counts = result.get_counts()
print("Grover search result counts:")
print(counts)
plot_histogram(counts)
plt.show()

# Interpret the result.
most_common_state = max(counts, key=counts.get)
retrieved_index = int(most_common_state, 2)
print("Retrieved document index from Grover search:", retrieved_index)
print("Document from Grover search:")
print(knowledge_base[retrieved_index])

##############################################
# 4. Lambeq DisCoCat Circuit & Qiskit Visualization
##############################################
from lambeq import BobcatParser, IQPAnsatz, AtomicType

# Initialize the BobcatParser.
parser = BobcatParser()
sentence = "Alice loves Bob."
diagram = parser.sentence2diagram(sentence)
print("Atomic types in the diagram:", {box.cod for box in diagram.boxes if box.cod.is_atomic})

# Define atomic types.
N = AtomicType.NOUN
S = AtomicType.SENTENCE
ob_map = {N: 1, S: 1}
n_layers = 1

# Initialize the IQPAnsatz.
ansatz = IQPAnsatz(ob_map=ob_map, n_layers=n_layers)

# Convert the diagram into a quantum circuit (lambeq's circuit).
quantum_circuit = ansatz(diagram)

# Draw using lambeq's built-in method.
quantum_circuit.draw()
plt.show()

# Convert lambeq's quantum circuit to a standard Qiskit QuantumCircuit via its QASM.
qiskit_circuit = QuantumCircuit.from_qasm_str(quantum_circuit.qasm())
print(qiskit_circuit.draw(output='text'))

# Visualize using Qiskit's circuit_drawer.
from qiskit.visualization import circuit_drawer
circuit_drawer(qiskit_circuit, output='mpl')
plt.show()
