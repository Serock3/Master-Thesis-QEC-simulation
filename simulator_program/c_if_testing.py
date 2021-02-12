# %%
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise import pauli_error
from qiskit import *
from qiskit.visualization import plot_histogram

# Define the error
p_error = 0.5
error_gate1 = pauli_error([('X',p_error), ('I', 1 - p_error)])
noise_bit_flip = NoiseModel()
noise_bit_flip.add_all_qubit_quantum_error( error_gate1, ["x"] )

# Define the circuit
cl = ClassicalRegister(1, 'classical_reg')
qr = QuantumRegister(1, 'qubit')

circuit = QuantumCircuit( qr, cl )
circuit.x(0).c_if(cl,1) # As cl is initialized in 0, this never goes off
circuit.measure(qr, cl)

# Run simulation
n_shots = 1024

results = execute(
    circuit,  
    Aer.get_backend('qasm_simulator'),
    noise_model=noise_bit_flip,
    shots=n_shots
).result()

circuit.draw(output='mpl')
plot_histogram(results.get_counts())