# import hashlib
# from lambeq import BobcatParser, RemoveCupsRewriter
# from qiskit import QuantumCircuit, transpile
# from qiskit_aer import AerSimulator
# from qiskit.visualization import plot_histogram
# import matplotlib.pyplot as plt

# ##############################################
# # Step 1: Build the Knowledge Base from Text #
# ##############################################

# # Example knowledge base (imagine these come from PDFs)
# knowledge_base = {
#     "Doc A": "Quantum computing leverages quantum phenomena such as superposition and entanglement.",
#     "Doc B": "Classical computing relies on bits and Boolean logic to perform operations.",
#     "Doc C": "Machine learning algorithms can process large amounts of data to find patterns."
# }

# # Initialize the lambeq parser and a rewriter to simplify diagrams.
# parser = BobcatParser()
# rewriter = RemoveCupsRewriter()

# def process_text_to_binary(text, n_bits=3):
#     """
#     Parse the text using lambeq, rewrite the diagram,
#     and convert its string representation to a binary string.
#     We use a hash of the diagram string mod 2^n_bits.
#     """
#     # Parse the sentence into a DisCoCat diagram.
#     diagram = parser.sentence2diagram(text)
#     # Rewrite (simplify) the diagram to remove cups/adjoints.
#     rewritten = rewriter.rewrite(diagram)
#     # Convert the diagram to its string representation.
#     diag_str = str(rewritten)
#     # Hash the string and convert to an integer.
#     hash_val = int(hashlib.sha256(diag_str.encode('utf-8')).hexdigest(), 16)
#     # Take modulo 2^n_bits to get a number in the desired range.
#     mod_val = hash_val % (2**n_bits)
#     # Format as a binary string of fixed length.
#     binary_str = format(mod_val, f'0{n_bits}b')
#     return binary_str, rewritten

# # Process each document in the knowledge base.
# doc_binaries = {}  # maps binary string -> document title
# doc_diagrams = {}  # store diagrams if needed
# for doc, text in knowledge_base.items():
#     binary, diag = process_text_to_binary(text, n_bits=3)
#     doc_binaries[binary] = doc
#     doc_diagrams[doc] = diag
#     print(f"{doc} -> Binary Embedding: {binary}")

# ##############################################
# # Step 2: Process the Query Text             #
# ##############################################

# query_text = "Quantum computing uses principles of superposition."
# query_binary, query_diag = process_text_to_binary(query_text, n_bits=3)
# print(f"\nQuery Text: {query_text}")
# print(f"Query Binary Embedding: {query_binary}")

# ##############################################
# # Step 3: Grover Search to Retrieve Document   #
# ##############################################

# def build_oracle(qc, target):
#     """
#     Build an oracle that flips the phase of the target state.
#     This implementation assumes that the quantum circuit has n qubits.
#     """
#     n = qc.num_qubits
#     # For each qubit, if the target bit is '0', apply an X gate.
#     for i, bit in enumerate(target):
#         if bit == '0':
#             qc.x(i)
#     # Apply a multi-controlled Z gate on the last qubit.
#     qc.h(n-1)
#     qc.mcx(list(range(n-1)), n-1)
#     qc.h(n-1)
#     # Undo the X gates.
#     for i, bit in enumerate(target):
#         if bit == '0':
#             qc.x(i)
#     return qc

# def grover_search(target, n_qubits=3, shots=1024):
#     """
#     Build and run a Grover search circuit to find the target binary string.
#     Returns the circuit and the measurement counts.
#     """
#     qc = QuantumCircuit(n_qubits, n_qubits)
#     # Create an equal superposition over all states.
#     qc.h(range(n_qubits))
#     # Apply the oracle to mark the target state.
#     build_oracle(qc, target)
#     # Apply the Grover diffusion operator.
#     qc.h(range(n_qubits))
#     qc.x(range(n_qubits))
#     qc.h(n_qubits-1)
#     qc.mcx(list(range(n_qubits-1)), n_qubits-1)
#     qc.h(n_qubits-1)
#     qc.x(range(n_qubits))
#     qc.h(range(n_qubits))
#     # Measure the qubits.
#     qc.measure(range(n_qubits), range(n_qubits))
    
#     # Run the circuit on the simulator.
#     simulator = AerSimulator()
#     compiled_circuit = transpile(qc, simulator)
#     job = simulator.run(compiled_circuit, shots=shots)
#     result = job.result()
#     counts = result.get_counts()
#     return qc, counts

# # Run Grover search with the query's binary embedding as the target.
# qc_grover, counts = grover_search(query_binary, n_qubits=3, shots=1024)
# print("\nGrover Search Results:")
# print(counts)
# plot_histogram(counts)
# plt.title("Grover Search Outcome")
# plt.show()

# # Determine the most frequent measurement outcome.
# retrieved_state = max(counts, key=counts.get)
# print(f"Retrieved Binary State: {retrieved_state}")

# # Look up the corresponding document from the knowledge base.
# retrieved_doc = doc_binaries.get(retrieved_state, "No matching document found")
# print(f"\nRetrieved Document: {retrieved_doc}")


# import hashlib
# from lambeq import BobcatParser, RemoveCupsRewriter
# from qiskit import QuantumCircuit, transpile
# from qiskit_aer import AerSimulator
# from qiskit.visualization import plot_histogram
# import matplotlib.pyplot as plt

# ##############################################
# # Step 1: Build the Knowledge Base from Text #
# ##############################################

# knowledge_base = {
#     "Doc A": "Quantum computing leverages quantum phenomena such as superposition and entanglement.",
#     "Doc B": "Classical computing relies on bits and Boolean logic to perform operations.",
#     "Doc C": "Machine learning algorithms can process large amounts of data to find patterns."
# }

# # Initialize the lambeq parser and a rewriter to simplify diagrams.
# parser = BobcatParser()
# rewriter = RemoveCupsRewriter()

# def process_text_to_binary(text, n_bits=3):
#     """
#     Convert text to a DisCoCat diagram using lambeq, rewrite (simplify) it,
#     and produce a binary embedding by hashing its string representation.
#     The output binary string has a fixed length (n_bits).
#     """
#     # Parse the text into a DisCoCat diagram.
#     diagram = parser.sentence2diagram(text)
#     # Rewrite the diagram to remove cups/adjoints and simplify types.
#     simplified = rewriter.rewrite(diagram)
#     # Convert the diagram to a string.
#     diag_str = str(simplified)
#     # Use SHA-256 to hash the string, then convert to an integer.
#     hash_val = int(hashlib.sha256(diag_str.encode('utf-8')).hexdigest(), 16)
#     # Reduce the integer to an n_bits number (range 0 to 2^n_bits - 1).
#     mod_val = hash_val % (2**n_bits)
#     # Format as a binary string with leading zeros.
#     binary_str = format(mod_val, f'0{n_bits}b')
#     return binary_str, simplified

# # Process each document in the knowledge base.
# doc_binaries = {}  # maps binary embedding -> document title
# for doc, text in knowledge_base.items():
#     binary, diag = process_text_to_binary(text, n_bits=3)
#     doc_binaries[binary] = doc
#     print(f"{doc} -> Binary Embedding: {binary}")

# ##############################################
# # Step 2: Process the Query Text             #
# ##############################################

# query_text = "Quantum computing uses principles of superposition."
# query_binary, query_diag = process_text_to_binary(query_text, n_bits=3)
# print(f"\nQuery: {query_text}")
# print(f"Query Binary Embedding: {query_binary}")

# ##############################################
# # Step 3: Grover Search for Retrieval         #
# ##############################################

# def build_oracle(qc, target):
#     """
#     Build an oracle on a circuit qc that flips the phase of the target state.
#     Assumes qc has n qubits; for each qubit where target bit is '0', apply an X gate,
#     then a multi-controlled Z is applied on the last qubit.
#     """
#     n = qc.num_qubits
#     for i, bit in enumerate(target):
#         if bit == '0':
#             qc.x(i)
#     qc.h(n-1)
#     qc.mcx(list(range(n-1)), n-1)
#     qc.h(n-1)
#     for i, bit in enumerate(target):
#         if bit == '0':
#             qc.x(i)
#     return qc

# def grover_search(target, n_qubits=3, shots=1024):
#     """
#     Build and run a Grover search circuit to amplify the amplitude of the target state.
#     Returns the circuit and the measurement counts.
#     """
#     qc = QuantumCircuit(n_qubits, n_qubits)
#     qc.h(range(n_qubits))  # Create equal superposition
#     build_oracle(qc, target)  # Mark the target state
#     # Grover diffusion operator
#     qc.h(range(n_qubits))
#     qc.x(range(n_qubits))
#     qc.h(n_qubits-1)
#     qc.mcx(list(range(n_qubits-1)), n_qubits-1)
#     qc.h(n_qubits-1)
#     qc.x(range(n_qubits))
#     qc.h(range(n_qubits))
#     qc.measure(range(n_qubits), range(n_qubits))
    
#     simulator = AerSimulator()
#     compiled = transpile(qc, simulator)
#     job = simulator.run(compiled, shots=shots)
#     result = job.result()
#     counts = result.get_counts()
#     return qc, counts

# # Run Grover search with the query's binary embedding as target.
# grover_circuit, counts = grover_search(query_binary, n_qubits=3, shots=1024)
# print("\nGrover Search Results:")
# print(counts)
# plot_histogram(counts)
# plt.title("Grover Search Outcome")
# plt.show()

# # Determine the most frequent measurement outcome.
# retrieved_state = max(counts, key=counts.get)
# print(f"Retrieved Binary State: {retrieved_state}")

# # Lookup the corresponding document.
# retrieved_doc = doc_binaries.get(retrieved_state, "No matching document found")
# print(f"\nRetrieved Document: {retrieved_doc}")



# import hashlib
# from lambeq import BobcatParser, RemoveCupsRewriter
# from qiskit import QuantumCircuit, transpile
# from qiskit_aer import AerSimulator
# from qiskit.visualization import plot_histogram
# import matplotlib.pyplot as plt

# ##############################################
# # Step 1: Build the Knowledge Base from Text #
# ##############################################

# knowledge_base = {
#     "Doc A": "Quantum computing leverages quantum phenomena such as superposition and entanglement.",
#     "Doc B": "Classical computing relies on bits and Boolean logic to perform operations.",
#     "Doc C": "Machine learning algorithms can process large amounts of data to find patterns."
# }

# # Initialize the lambeq parser and rewriter.
# parser = BobcatParser()
# rewriter = RemoveCupsRewriter()

# def process_text_to_binary(text, n_bits=4):
#     """
#     Convert text to a DisCoCat diagram using lambeq, rewrite (simplify) it,
#     and produce a binary embedding by hashing its string representation.
#     The output binary string has fixed length (n_bits).
#     """
#     # Parse the text into a DisCoCat diagram.
#     diagram = parser.sentence2diagram(text)
#     # Rewrite the diagram to remove cups/adjoints and simplify types.
#     simplified = rewriter.rewrite(diagram)
#     # Convert the diagram to a string.
#     diag_str = str(simplified)
#     # Hash the string using SHA-256 and convert to an integer.
#     hash_val = int(hashlib.sha256(diag_str.encode('utf-8')).hexdigest(), 16)
#     # Reduce the integer modulo 2^n_bits to obtain a number in [0, 2^n_bits - 1].
#     mod_val = hash_val % (2**n_bits)
#     # Format as a binary string with leading zeros.
#     binary_str = format(mod_val, f'0{n_bits}b')
#     return binary_str, simplified

# # Process each document in the knowledge base.
# doc_binaries = {}  # maps binary embedding -> document title
# for doc, text in knowledge_base.items():
#     binary, diag = process_text_to_binary(text, n_bits=4)
#     doc_binaries[binary] = doc
#     print(f"{doc} -> Binary Embedding (4 bits): {binary}")

# ##############################################
# # Step 2: Process the Query Text             #
# ##############################################

# query_text = "Quantum computing uses principles of superposition."
# query_binary, query_diag = process_text_to_binary(query_text, n_bits=4)
# print(f"\nQuery Text: {query_text}")
# print(f"Query Binary Embedding (4 bits): {query_binary}")

# ##############################################
# # Step 3: Grover Search for Retrieval         #
# ##############################################

# def build_oracle(qc, target):
#     """
#     Build an oracle on the circuit qc that flips the phase of the target state.
#     For each qubit where the target bit is '0', apply an X gate,
#     then apply a multi-controlled Z on the last qubit.
#     """
#     n = qc.num_qubits
#     for i, bit in enumerate(target):
#         if bit == '0':
#             qc.x(i)
#     qc.h(n-1)
#     qc.mcx(list(range(n-1)), n-1)
#     qc.h(n-1)
#     for i, bit in enumerate(target):
#         if bit == '0':
#             qc.x(i)
#     return qc

# def grover_search(target, n_qubits=4, shots=1024):
#     """
#     Build and run a Grover search circuit to amplify the target state.
#     Returns the circuit and the measurement counts.
#     """
#     qc = QuantumCircuit(n_qubits, n_qubits)
#     qc.h(range(n_qubits))  # Create equal superposition
#     build_oracle(qc, target)  # Mark the target state
#     # Grover diffusion operator:
#     qc.h(range(n_qubits))
#     qc.x(range(n_qubits))
#     qc.h(n_qubits-1)
#     qc.mcx(list(range(n_qubits-1)), n_qubits-1)
#     qc.h(n_qubits-1)
#     qc.x(range(n_qubits))
#     qc.h(range(n_qubits))
#     qc.measure(range(n_qubits), range(n_qubits))
    
#     simulator = AerSimulator()
#     compiled = transpile(qc, simulator)
#     job = simulator.run(compiled, shots=shots)
#     result = job.result()
#     counts = result.get_counts()
#     return qc, counts

# # Run Grover search using the query's binary embedding as target.
# grover_circuit, counts = grover_search(query_binary, n_qubits=4, shots=1024)
# print("\nGrover Search Results:")
# print(counts)
# plot_histogram(counts)
# plt.title("Grover Search Outcome (4 Qubits)")
# plt.show()

# # Determine the most frequent measurement outcome.
# retrieved_state = max(counts, key=counts.get)
# print(f"Retrieved Binary State: {retrieved_state}")

# # Lookup the corresponding document from the knowledge base.
# retrieved_doc = doc_binaries.get(retrieved_state, "No matching document found")
# print(f"\nRetrieved Document: {retrieved_doc}")



import hashlib
from lambeq import BobcatParser, RemoveCupsRewriter
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

##############################################
# Step 1: Build the Knowledge Base from Text #
##############################################

knowledge_base = {
    "Doc A": "Quantum computing leverages quantum phenomena such as superposition and entanglement.",
    "Doc B": "Classical computing relies on bits and Boolean logic to perform operations.",
    "Doc C": "Machine learning algorithms can process large amounts of data to find patterns."
}

# Initialize the lambeq parser and rewriter.
parser = BobcatParser()
rewriter = RemoveCupsRewriter()

def process_text_to_binary(text, n_bits=8):
    """
    Convert text to a DisCoCat diagram using lambeq, rewrite it,
    and produce a binary embedding by hashing its string representation.
    The output binary string has fixed length (n_bits).
    """
    # Parse the text into a DisCoCat diagram.
    diagram = parser.sentence2diagram(text)
    # Rewrite the diagram to simplify types.
    simplified = rewriter.rewrite(diagram)
    # Convert the diagram to its string representation.
    diag_str = str(simplified)
    # Hash the string using SHA-256 and convert to an integer.
    hash_val = int(hashlib.sha256(diag_str.encode('utf-8')).hexdigest(), 16)
    # Reduce the integer modulo 2^n_bits.
    mod_val = hash_val % (2**n_bits)
    # Format as a binary string with leading zeros.
    binary_str = format(mod_val, f'0{n_bits}b')
    return binary_str, simplified

# Process each document.
doc_binaries = {}  # maps binary embedding -> document title
for doc, text in knowledge_base.items():
    binary, diag = process_text_to_binary(text, n_bits=8)
    doc_binaries[binary] = doc
    print(f"{doc} -> Binary Embedding (8 bits): {binary}")

##############################################
# Step 2: Process the Query Text             #
##############################################

query_text = "Quantum computing uses principles of superposition."
query_binary, query_diag = process_text_to_binary(query_text, n_bits=8)
print(f"\nQuery Text: {query_text}")
print(f"Query Binary Embedding (8 bits): {query_binary}")

##############################################
# Step 3: Grover Search for Retrieval         #
##############################################

def build_oracle(qc, target):
    """
    Build an oracle on circuit qc that flips the phase of the target state.
    For each qubit where the target bit is '0', apply an X gate,
    then apply a multi-controlled Z gate on the last qubit.
    """
    n = qc.num_qubits
    for i, bit in enumerate(target):
        if bit == '0':
            qc.x(i)
    qc.h(n-1)
    qc.mcx(list(range(n-1)), n-1)
    qc.h(n-1)
    for i, bit in enumerate(target):
        if bit == '0':
            qc.x(i)
    return qc

def grover_search(target, n_qubits=8, shots=1024):
    """
    Build and run a Grover search circuit on n_qubits to amplify the target state.
    Returns the circuit and measurement counts.
    """
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))  # Equal superposition.
    build_oracle(qc, target)  # Mark target.
    # Diffusion operator.
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits-1)
    qc.mcx(list(range(n_qubits-1)), n_qubits-1)
    qc.h(n_qubits-1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    qc.measure(range(n_qubits), range(n_qubits))
    
    simulator = AerSimulator()
    compiled = transpile(qc, simulator)
    job = simulator.run(compiled, shots=shots)
    result = job.result()
    counts = result.get_counts()
    return qc, counts

# Run Grover search using the query's 8-bit binary embedding.
grover_circuit, counts = grover_search(query_binary, n_qubits=8, shots=1024)
print("\nGrover Search Results:")
print(counts)
plot_histogram(counts)
plt.title("Grover Search Outcome (8 Qubits)")
plt.show()

# Determine the most frequent measurement outcome.
retrieved_state = max(counts, key=counts.get)
print(f"Retrieved Binary State: {retrieved_state}")

# Lookup the corresponding document.
retrieved_doc = doc_binaries.get(retrieved_state, "No matching document found")
print(f"\nRetrieved Document: {retrieved_doc}")
