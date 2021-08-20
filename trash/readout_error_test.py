
#%%

from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors import ReadoutError
from qiskit import QuantumCircuit, QuantumRegister, execute, Aer, ClassicalRegister
from IPython.core.display import display
import numpy as np

qb = QuantumRegister(2)
cl = ClassicalRegister(2)
circ = QuantumCircuit(qb,cl)
# circ.measure_all()
# circ.measure_all()
circ.measure(qb,cl)
circ.snapshot('asd','statevector')

display(circ.draw())

noise_model = NoiseModel()


# Some typical values for readout errors that Alexander found in some paper I think
#  0.013, 0.019, 0.003, 0.013 och 0.016 

# Define a single qb redout error (no correlation)
p1given0 = 1
p0given1 = p1given0
read_error = ReadoutError([[1 - p1given0, p1given0], [p0given1, 1 - p0given1]])

# NOTE: 'All-qubit readout errors must defined as single-qubit errors.' which we do here
noise_model.add_all_qubit_readout_error(read_error, ['measure']) # ['measure'] seems to do nothing

# We can also add a correlated error (or not, I guess, doesn't work at all)
probabilities = [[0, 0, 0, 1], [0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]]
probabilities = np.identity(4)
probabilities[0][0] = 0
probabilities[0][1] = 1
readout_error = ReadoutError(probabilities)
noise_model = NoiseModel()
noise_model.add_readout_error(readout_error, [0, 1], warnings=True)


results = execute(
    circ,
    Aer.get_backend('qasm_simulator'),
    noise_model=noise_model,
    shots=100,
).result()
print(results.get_counts())
# print(results.data()['snapshots']['statevector']['asd'])
# %%