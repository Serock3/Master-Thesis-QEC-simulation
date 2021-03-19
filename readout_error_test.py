
#%%

from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors import ReadoutError
from qiskit import QuantumCircuit, QuantumRegister, execute, Aer
from IPython.core.display import display


circ = QuantumCircuit(2)
circ.measure_all()
circ.measure_all()
circ.snapshot('asd','statevector')

display(circ.draw())

noise_model = NoiseModel()
probabilities = [[0.5]*2]*2
#  0.013, 0.019, 0.003, 0.013 och 0.016 
p1given0 = 1
p0given1 = p1given0
read_error = ReadoutError([[1 - p1given0, p1given0], [p0given1, 1 - p0given1]])
noise_model.add_all_qubit_readout_error(read_error, ['measure'])
# readout_error = ReadoutError(probabilities)

# noise_model.add_readout_error(readout_error, [0], warnings=True)

# noise_model.add_all_qubit_readout_error(readout_error)
results = execute(
    circ,
    Aer.get_backend('qasm_simulator'),
    noise_model=noise_model,
    shots=1,
).result()
print(results.get_counts())
print(results.data()['snapshots']['statevector']['asd'])
# %%