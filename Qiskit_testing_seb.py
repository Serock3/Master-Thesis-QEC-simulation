#%%
import numpy as np
from qiskit import *
# Useful additional packages
import matplotlib.pyplot as plt
%matplotlib inline
from math import pi

#%%

q = QuantumRegister(1)
c = ClassicalRegister(1)
qc = QuantumCircuit(q,c)
qc.x(q[1])
qc.cnot(q[0],q[1])
qc.measure(q[0], c)
qc.draw()
# %%
backend = BasicAer.get_backend('qasm_simulator')
qc = QuantumCircuit(q, c)
qc.x(q[0]).c_if(c, 0)
qc.measure(q,c)
qc.draw()
job = execute(qc, backend, shots=1024)
job.result().get_counts(qc)
# %%
import math

desired_vector = 1.5*np.array([
    1 / math.sqrt(16) * complex(0, 1),
    1 / math.sqrt(8) * complex(1, 0),
    1 / math.sqrt(16) * complex(1, 1),
    0,
    0,
    1 / math.sqrt(8) * complex(1, 2),
    1 / math.sqrt(16) * complex(1, 0),
    0])

desired_vector /= np.linalg.norm(desired_vector)
q = QuantumRegister(3)

qc = QuantumCircuit(q)

qc.initialize(desired_vector, [q[0],q[1],q[2]])
qc.draw()

# %%
backend = BasicAer.get_backend('statevector_simulator')
job = execute(qc, backend)
qc_state = job.result().get_statevector(qc)
qc_state
# %%
from qiskit.quantum_info import state_fidelity
state_fidelity(desired_vector,qc_state)
# %%

from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors import pauli_error, depolarizing_error

def get_noise(p_meas,p_gate):

    error_meas = pauli_error([('X',p_meas), ('I', 1 - p_meas)])
    error_gate1 = depolarizing_error(p_gate, 1)
    error_gate2 = error_gate1.tensor(error_gate1)

    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(error_meas, "measure") # measurement error is applied to measurements
    noise_model.add_all_qubit_quantum_error(error_gate1, ["x"]) # single qubit gate error is applied to x gates
    noise_model.add_all_qubit_quantum_error(error_gate2, ["cx"]) # two qubit gate error is applied to cx gates
        
    return noise_model

#%% Kolla på detta
from qiskit.ignis.verification.topological_codes import fitters
from qiskit.quantum_info.operators.symplectic import stabilizer_table
from qiskit.ignis.verification.topological_codes import circuits


# %%
