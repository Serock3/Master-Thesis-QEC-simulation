
#%%

from qiskit.providers.aer.noise import ReadoutError, NoiseModel
from qiskit import QuantumCircuit, QuantumRegister, execute, Aer

qb = QuantumRegister(2)

circ = QuantumCircuit(qb)
circ.measure_all()

circ.draw()

readout_error = NoiseModel()
readout_error.add_readout_error(ReadoutError([[0.25]*4]*4),circ.qubits)

results = execute(
    circ,
    Aer.get_backend('qasm_simulator'),
    noise_model=readout_error,
    shots=10,
).result()
# %%
