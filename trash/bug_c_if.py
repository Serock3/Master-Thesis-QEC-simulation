# %%
from qiskit.providers.aer.noise import NoiseModel,pauli_error
from qiskit import execute,ClassicalRegister,QuantumRegister,QuantumCircuit,Aer

# Define the error
p_error = 0.5
error_gate1 = pauli_error([('X',p_error), ('I', 1 - p_error)])
noise_bit_flip = NoiseModel()
noise_bit_flip.add_all_qubit_quantum_error( error_gate1, ["z"] )

# Define the circuit
cl = ClassicalRegister(1, 'classical_reg')
qr = QuantumRegister(1, 'qubit')

circuit = QuantumCircuit( qr, cl )
circuit.z(0).c_if(cl,1) # As cl is initialized in 0, this never triggers
circuit.measure(qr, cl)

results = execute(
    circuit,  
    Aer.get_backend('qasm_simulator'),
    noise_model=noise_bit_flip,
).result()

print(results.get_counts())
# %%
