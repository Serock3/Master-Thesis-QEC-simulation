# %%
from qiskit.providers.aer.noise import NoiseModel,pauli_error
from qiskit import execute,ClassicalRegister,QuantumRegister,QuantumCircuit,Aer
from IPython.display import display
# Define a 50% error to introduce non-determinance 
p_error = 0.5
error_gate1 = pauli_error([('X',p_error), ('I', 1 - p_error)])
noise_bit_flip = NoiseModel()
noise_bit_flip.add_all_qubit_quantum_error( error_gate1, ["z","CZ"] )

# Define the circuit
cl = ClassicalRegister(2, 'classical_reg')
qr = QuantumRegister(1, 'qubit')

circuit = QuantumCircuit( qr, cl )
circuit.z(0) # As cl is initialized in 0, this never triggers

circuit.measure(qr, cl[0])
circuit.z(0)
# circuit.measure(qr, cl[1])
circuit.snapshot('asd', 'density_matrix')#statevector
display(circuit.draw())
results = execute(
    circuit,  
    Aer.get_backend('qasm_simulator'),
    noise_model=noise_bit_flip,
    shots=5
    # memory=True
).result()

print(results.get_counts())
print(results.data()['snapshots'])
# %% See if measurements on qubits not included in the snapshot are possible to condition on
# This will let condition on ancilla-stabilizer measurements for our code qubits
# If it doesn't work then we can probably just take the reduced density matrix (trace out qb 5&6)
from qiskit.providers.aer.extensions import Snapshot, snapshot_density_matrix
# Define the circuit
cl = ClassicalRegister(2, 'classical_reg')
qr = QuantumRegister(2, 'qubit')

circuit = QuantumCircuit( qr, cl )
# circuit.z(1) 
circuit.z(0)
circuit.cnot(0,1)
circuit.measure(qr[1], cl[0])
# Without this barrier, you can't condition on the measurement above
circuit.barrier() 

# To manually define the register for a snapshot, we need to use the .append method instead
# If snapshot_type='statevector' is set then it ignores the register input and goes on all qubits
# (but will still show it in the drawing)
circuit.append(Snapshot('State of qb 0','density_matrix',num_qubits=1),[qr[0]])
# circuit.snapshot('asd2', 'density_matrix')

results = execute(
    circuit,  
    Aer.get_backend('qasm_simulator'),
    noise_model=noise_bit_flip,
    shots=10
    # memory=True
).result()

display(circuit.draw())
print(results.get_counts())
print(results.data()['snapshots']['density_matrix']['State of qb 0'])

# %%
