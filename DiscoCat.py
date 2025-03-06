from lambeq import BobcatParser, IQPAnsatz, AtomicType
import matplotlib.pyplot as plt

# Initialize the parser
parser = BobcatParser()

# Define your sentence
sentence = "Alice loves Bob."

# Parse the sentence into a DisCoCat diagram
diagram = parser.sentence2diagram(sentence)

# Identify atomic types in the diagram
atomic_types = set(box.cod for box in diagram.boxes if box.cod.is_atomic)
print("Atomic types in the diagram:", atomic_types)

# Define atomic types
N = AtomicType.NOUN
S = AtomicType.SENTENCE

# Create the object map, ensuring all atomic types are included
ob_map = {N: 1, S: 1}

# Set the number of layers
n_layers = 2

# Initialize the IQPAnsatz
ansatz = IQPAnsatz(ob_map=ob_map, n_layers=n_layers)

# Convert the diagram into a quantum circuit
quantum_circuit = ansatz(diagram)

# Visualize the quantum circuit
quantum_circuit.draw()
plt.show()
