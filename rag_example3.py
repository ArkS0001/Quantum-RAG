import numpy as np
from transformers import pipeline
from sklearn.decomposition import PCA
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

###########################
# 1. Classical Embedding  #
###########################

# Build a small knowledge base of documents.
knowledge_base = {
    "Doc A": "Quantum computing is a new paradigm that leverages quantum mechanics.",
    "Doc B": "Classical computers use bits to perform computations.",
    "Doc C": "Machine learning algorithms can be very effective for data analysis.",
    "Doc D": "Quantum algorithms such as Grover's search offer speedups over classical methods."
}

# Create a Hugging Face feature extraction pipeline using BERT.
# (This pipeline returns a list of token embeddings; we use the first token [CLS].)
feature_extractor = pipeline("feature-extraction", model="bert-base-uncased", tokenizer="bert-base-uncased")

def get_bert_embedding(text):
    # The pipeline returns a nested list of shape [1, seq_length, hidden_size].
    outputs = feature_extractor(text)
    # Use the [CLS] token embedding (index 0 of the sequence)
    cls_embedding = np.array(outputs[0][0])
    return cls_embedding

# Compute embeddings for each document.
embeddings = {}
for doc, text in knowledge_base.items():
    embeddings[doc] = get_bert_embedding(text)

# Stack all embeddings into an array.
embedding_matrix = np.stack(list(embeddings.values()))

# For our toy example, reduce embeddings to 3 dimensions using PCA.
pca = PCA(n_components=3)
embedding_reduced = pca.fit_transform(embedding_matrix)

# Binarize each 3-dimensional embedding into a 3-bit string.
# We use a simple threshold: if a component is >= 0 then bit '1', else '0'.
def binarize_embedding(vec):
    bits = ['1' if x >= 0 else '0' for x in vec]
    return "".join(bits)

binary_embeddings = {}
for i, doc in enumerate(knowledge_base):
    binary_embeddings[doc] = binarize_embedding(embedding_reduced[i])

print("Knowledge Base Binary Embeddings:")
for doc, b_emb in binary_embeddings.items():
    print(f"{doc}: {b_emb}")

#############################
# 2. Encode a Query in BERT #
#############################

# Example query
query_text = "Which algorithms provide quantum speedup?"
query_embedding = get_bert_embedding(query_text)
query_reduced = pca.transform(query_embedding.reshape(1, -1))[0]
query_binary = binarize_embedding(query_reduced)

print("\nQuery Text:", query_text)
print("Query Binary Embedding:", query_binary)

#########################################
# 3. Quantum Grover Search Preparation  #
#########################################

# For demonstration, we assume our search space is 3 qubits.
# Each binary embedding is a 3-bit string (e.g., "101").

num_qubits = 3

def build_oracle(qc, target):
    """
    Build an oracle that flips the phase of the target state.
    For each qubit where the target bit is '0', apply an X gate.
    Then apply a multi-controlled Z (via H, MCX, H) on the last qubit.
    Finally, undo the X gates.
    """
    for i, bit in enumerate(target):
        if bit == '0':
            qc.x(i)
    qc.h(num_qubits - 1)
    qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    qc.h(num_qubits - 1)
    for i, bit in enumerate(target):
        if bit == '0':
            qc.x(i)
    return qc

def grover_search(target, shots=1024):
    """
    Build and run a Grover search circuit to find the target binary state.
    """
    qc = QuantumCircuit(num_qubits, num_qubits)
    
    # Step 1: Create an equal superposition over all states.
    qc.h(range(num_qubits))
    
    # Step 2: Apply the oracle that marks the target embedding.
    build_oracle(qc, target)
    
    # Step 3: Grover Diffusion Operator (inversion about the mean).
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    qc.h(num_qubits - 1)
    qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    qc.h(num_qubits - 1)
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))
    
    # Step 4: Measure all qubits.
    qc.measure(range(num_qubits), range(num_qubits))
    
    # Run on the Aer simulator.
    simulator = AerSimulator()
    compiled_circuit = transpile(qc, simulator)
    job = simulator.run(compiled_circuit, shots=shots)
    result = job.result()
    counts = result.get_counts()
    
    return qc, counts

#########################################
# 4. Run Grover Search with the Query    #
#########################################

print("\nRunning Grover Search for the Query Binary Embedding...")
qc, counts = grover_search(query_binary, shots=1024)
print("Grover Search Results:", counts)
plot_histogram(counts)
plt.show()

# Determine the most frequent state measured.
retrieved_state = max(counts, key=counts.get)
print("\nMost Frequent Quantum State:", retrieved_state)

#########################################
# 5. Retrieve the Document from the KB   #
#########################################

retrieved_doc = None
for doc, b_emb in binary_embeddings.items():
    if b_emb == retrieved_state:
        retrieved_doc = doc
        break

if retrieved_doc:
    print("Retrieved Document:", retrieved_doc)
else:
    print("No matching document found in the knowledge base.")
