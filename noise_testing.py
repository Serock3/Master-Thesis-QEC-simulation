# %% Import modules
import pickle
from qiskit.quantum_info import purity
from numpy.core.fromnumeric import repeat
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from qiskit import *
from qiskit.visualization import plot_histogram
from IPython.display import display

from qiskit.quantum_info import state_fidelity
from qiskit.providers.aer.extensions.snapshot_statevector import *

# Import from Qiskit Aer noise module
from qiskit.providers.aer.noise import pauli_error,depolarizing_error
from qiskit.providers.aer.noise import (thermal_relaxation_error, 
                                        amplitude_damping_error, 
                                        phase_damping_error, 
                                        phase_amplitude_damping_error
                                        )
from qiskit.providers.aer.noise import NoiseModel,QuantumError, ReadoutError
from qiskit.providers.aer import noise
import numpy as np

# %%
def thermal_relaxation_model():
    # T1 and T2 values for qubits 0-3
    # Sampled from normal distribution mean 50 microsec
    T1s = np.random.normal(50e3, 10e3, 4)
    # Sampled from normal distribution mean 50 microsec
    T2s = np.random.normal(70e3, 10e3, 4)

    # Truncate random T2s <= T1s
    T2s = np.array([min(T2s[j], 2 * T1s[j]) for j in range(4)])

    # Instruction times (in nanoseconds)
    time_u1 = 0   # virtual gate
    time_u2 = 50  # (single X90 pulse)
    time_u3 = 100  # (two X90 pulses)
    time_cx = 300
    time_reset = 1000  # 1 microsecond
    time_measure = 1000  # 1 microsecond

    # QuantumError objects
    errors_reset = [thermal_relaxation_error(t1, t2, time_reset)
                    for t1, t2 in zip(T1s, T2s)]
    errors_measure = [thermal_relaxation_error(t1, t2, time_measure)
                      for t1, t2 in zip(T1s, T2s)]
    errors_u1 = [thermal_relaxation_error(t1, t2, time_u1)
                 for t1, t2 in zip(T1s, T2s)]
    errors_u2 = [thermal_relaxation_error(t1, t2, time_u2)
                 for t1, t2 in zip(T1s, T2s)]
    errors_u3 = [thermal_relaxation_error(t1, t2, time_u3)
                 for t1, t2 in zip(T1s, T2s)]
    errors_cx = [[thermal_relaxation_error(t1a, t2a, time_cx).expand(
        thermal_relaxation_error(t1b, t2b, time_cx))
        for t1a, t2a in zip(T1s, T2s)]
        for t1b, t2b in zip(T1s, T2s)]

    # Add errors to noise model
    noise_thermal = NoiseModel()

    for j in range(4):
        noise_thermal.add_quantum_error(errors_reset[j], "reset", [j])
        noise_thermal.add_quantum_error(errors_measure[j], "measure", [j])
        noise_thermal.add_quantum_error(errors_u1[j], "u1", [j])
        noise_thermal.add_quantum_error(errors_u2[j], "u2", [j])
        noise_thermal.add_quantum_error(errors_u3[j], "u3", [j])
        for k in range(4):
            noise_thermal.add_quantum_error(errors_cx[j][k], "cx", [j, k])
    return noise_thermal

def thermal_relaxation_testing(T1=40e3, T2=60e3):

    # Instruction times (in nanoseconds)
    time_single = 15
    time_cz = 300

    # QuantumError objects
    error_single = thermal_relaxation_error(T1, T2, time_single)
    error_cz = thermal_relaxation_error(T1, T2, time_cz).expand(
        thermal_relaxation_error(T1, T2, time_cz))

    # Add errors to noise model
    noise_thermal = NoiseModel()

    noise_thermal.add_all_qubit_quantum_error(error_single, ["x", "z", "h"])
    noise_thermal.add_all_qubit_quantum_error(error_cz, ["cx", "cz"])

    return noise_thermal

def phase_amplitude_testing(T1=40e3, T2=60e3):

    # Instruction times (in nanoseconds)
    time_single = 15
    time_cz = 300

    pAD_single = 1-np.exp(-time_single/T1)
    pPD_single = 1 - (np.exp(-2*time_single/T2))/(np.exp(-time_single/T1))

    pAD_cz = 1-np.exp(-time_cz/T1)
    pPD_cz = 1 - (np.exp(-2*time_cz/T2))/(np.exp(-time_cz/T1))

    # QuantumError objects
    error_single = phase_amplitude_damping_error(pAD_single, pPD_single)
    error_cz = phase_amplitude_damping_error(pAD_cz, pPD_cz).expand(
        phase_amplitude_damping_error(pAD_cz, pPD_cz))

    # Add errors to noise model
    noise_damping = NoiseModel()

    noise_damping.add_all_qubit_quantum_error(error_single, ["x", "z", "h","id"])
    noise_damping.add_all_qubit_quantum_error(error_cz, ["cx", "cz"])

    return noise_damping



def encode_input(qbReg):
    """Encode the input into logical 0 and 1 for the [[5,1,3]] code. This
    assumes that the 0:th qubit is the original state |psi> = a|0> + b|1>
    """
    circ = QuantumCircuit(qbReg)

    circ.h(qbReg[3])
    circ.cz(qbReg[3], qbReg[1])
    circ.cz(qbReg[3], qbReg[2])
    circ.cx(qbReg[3], qbReg[0])

    circ.h(qbReg[2])
    circ.cx(qbReg[2], qbReg[0])
    circ.cz(qbReg[2], qbReg[3])
    circ.cz(qbReg[2], qbReg[4])

    circ.h(qbReg[1])
    circ.cz(qbReg[1], qbReg[0])
    circ.cx(qbReg[1], qbReg[3])
    circ.cz(qbReg[1], qbReg[4])

    circ.h(qbReg[4])
    circ.cz(qbReg[4], qbReg[2])
    circ.cz(qbReg[4], qbReg[3])
    circ.cx(qbReg[4], qbReg[1])

    return circ

# %%

# Define registers
n_qubits = 2
qb = QuantumRegister(n_qubits, 'code_qubit')
an = AncillaRegister(2, 'ancilla_qubit')
cr = ClassicalRegister(5, 'syndrome_bit')
readout = ClassicalRegister(n_qubits, 'readout')

circ = QuantumCircuit(qb, readout)
#circ += encode_input(qb)
#circ += encode_input(qb).inverse()
circ.h(qb[0])
circ.cx(qb[0], qb[1])
circ.h(qb[0])
circ.cx(qb[0], qb[1])
circ.h(qb[0])
circ.cx(qb[0], qb[1])
#circ.cx(qb[0], qb[1])
#circ.h(qb[0])

#circ.barrier(qb)
circ.snapshot('measure', snapshot_type="density_matrix")
circ.measure(qb, readout)

#circ.draw(output='mpl')
# %

# Run both models
n_shots = 1
results_thermal = execute(
    circ,  
    Aer.get_backend('qasm_simulator'),
    noise_model=thermal_relaxation_testing(),
    shots=n_shots
).result()
results_damping = execute(
    circ,  
    Aer.get_backend('qasm_simulator'),
    noise_model=phase_amplitude_testing(),
    shots=n_shots
).result()

# If fidelity is not (close to) 1, next cell can print/analyze further
sv_thermal = results_thermal.data()['snapshots']['density_matrix']['measure'][0]['value']
sv_damping = results_damping.data()['snapshots']['density_matrix']['measure'][0]['value']
print(state_fidelity(sv_damping, sv_thermal))
#counts = results.get_counts()
#plot_histogram( counts )
# %%

print(sv_damping)
print(' ')
print(sv_thermal)
print(state_fidelity(sv_damping, sv_thermal))

#%%
from qiskit.quantum_info import DensityMatrix

test_dm = DensityMatrix(sv_damping)
test_sv = testy.to_statevector()
print(test_sv)