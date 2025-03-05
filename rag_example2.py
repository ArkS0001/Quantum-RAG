import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

# ----------------------------
# 1. Sample Text and Dummy Embeddings
# ----------------------------

def dummy_text_embedding(text):
    """
    A toy embedding function that maps a text string to a 3-bit binary string.
    (In practice, use a proper text embedding model like BERT or TF-IDF + PCA.)
    """
    # Use Python's built-in hash function (absolute value and modulo 8 to get a number between 0 and 7)
    num = abs(hash(text)) % 8  
    # Format as a 3-bit binary string
    return format(num, '03b')

# Sample dataset of documents
dataset = {
    "Doc A": "This is a sample document about cats.",
    "Doc B": "This is another sample about dogs.",
    "Doc C": "This text covers quantum computing and algorithms.",
    "Doc D": "This document discusses classical search algorithms."
}

# Compute dummy embeddings for each document
embeddings = {doc: dummy_text_embedding(text) for doc, text in dataset.items()}

print("Document Embeddings:")
for doc, emb in embeddings.items():
    print(f"{doc}: {emb}")

# ----------------------------
# 2. Choose a Query and Encode It
# ----------------------------

# For this example, we choose a query text.
query_text = "This document discusses classical search algorithms."
query_embedding = dummy_text_embedding(query_text)
print("\nQuery Embedding:", query_embedding)

# Determine the number of qubits from the length of the binary embedding.
num_qubits = len(query_embedding)

# ----------------------------
# 3. Grover's Algorithm: Oracle and Diffusion Operator
# ----------------------------

def build_oracle(qc, target):
    """
    Build an oracle that flips the phase of the target state.
    For each qubit where the target bit is '0', apply an X gate.
    Then, implement a multi-controlled-Z (via H, MCX, H) on the last qubit.
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

# Run Grover search for the query embedding.
qc, counts = grover_search(query_embedding, shots=1024)
print("\nGrover's Search Results:", counts)
plot_histogram(counts)
plt.show()

# ----------------------------
# 4. Interpret the Results
# ----------------------------

# Find the most frequent measured state.
most_frequent_state = max(counts, key=counts.get)
print("\nMost Frequent State:", most_frequent_state)

# Retrieve the document that matches the state.
retrieved_doc = None
for doc, emb in embeddings.items():
    if emb == most_frequent_state:
        retrieved_doc = doc
        break

if retrieved_doc:
    print("Retrieved Document:", retrieved_doc)
else:
    print("No matching document found.")
