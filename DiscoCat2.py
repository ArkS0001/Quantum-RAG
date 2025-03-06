# from lambeq import BobcatParser, IQPAnsatz, AtomicType
# import matplotlib.pyplot as plt

# # Initialize the parser
# parser = BobcatParser()

# # Define your sentences
# sentences = [
#     "Alice loves Bob.",
#     "The cat sat on the mat.",
#     "Quantum computing is fascinating.",
#     "Hello Atomic types in the diagram for 'Quantum computing is fascinating "
#     # Add more sentences as needed
# ]

# # Define atomic types
# N = AtomicType.NOUN
# S = AtomicType.SENTENCE

# # Create the object map, ensuring all atomic types are included
# ob_map = {N: 1, S: 1}

# # Set the number of layers
# n_layers = 2

# # Initialize the IQPAnsatz
# ansatz = IQPAnsatz(ob_map=ob_map, n_layers=n_layers)

# # Process each sentence
# for sentence in sentences:
#     # Parse the sentence into a DisCoCat diagram
#     diagram = parser.sentence2diagram(sentence)

#     # Identify atomic types in the diagram
#     atomic_types = set(box.cod for box in diagram.boxes if box.cod.is_atomic)
#     print(f"Atomic types in the diagram for '{sentence}':", atomic_types)

#     # Convert the diagram into a quantum circuit
#     quantum_circuit = ansatz(diagram)

#     # Visualize the quantum circuit
#     quantum_circuit.draw()
#     plt.title(f"Quantum Circuit for: '{sentence}'")
#     plt.show()


from lambeq import BobcatParser, IQPAnsatz, AtomicType, RemoveCupsRewriter
import matplotlib.pyplot as plt

# Initialize the parser
parser = BobcatParser()

# Define your sentences
sentences = [
    "Alice loves Bob.",
    "The cat sat on the mat.",
    "Quantum computing is fascinating.",
    # Add more sentences as needed
]

# Parse each sentence into a DisCoCat diagram
diagrams = [parser.sentence2diagram(sentence) for sentence in sentences]

# Combine diagrams into a single diagram via tensor product ("@" operator)
combined_diagram = diagrams[0]
for diagram in diagrams[1:]:
    combined_diagram = combined_diagram @ diagram

print("Combined diagram before rewriting:")
print(combined_diagram)

# Apply a rewriter to simplify the diagram and remove cups (adjoints)
rewriter = RemoveCupsRewriter()
rewritten_diagram = rewriter.rewrite(combined_diagram)

print("Combined diagram after rewriting:")
print(rewritten_diagram)

# Define atomic types
N = AtomicType.NOUN
S = AtomicType.SENTENCE

# Create the object map, assuming each atomic type is encoded on 1 qubit.
ob_map = {N: 1, S: 1}

# Set the number of layers for the ansatz
n_layers = 2

# Initialize the IQPAnsatz with the object map and number of layers
ansatz = IQPAnsatz(ob_map=ob_map, n_layers=n_layers)

# Convert the rewritten diagram into a quantum circuit
quantum_circuit = ansatz(rewritten_diagram)

# Visualize the quantum circuit
quantum_circuit.draw()
plt.title("Quantum Circuit for Combined Sentences (Rewritten)")
plt.show()
