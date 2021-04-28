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

# Our own files
from simulator_program.custom_noise_models import phase_amplitude_model
from simulator_program.stabilizers import *



# %% ===== Comparing Thermal Relaxation with Phase+Amplitude Damping =====

# Define registers
n_qubits = 2
qb = QuantumRegister(n_qubits, 'code_qubit')
an = AncillaRegister(2, 'ancilla_qubit')
cr = ClassicalRegister(5, 'syndrome_bit')
readout = ClassicalRegister(n_qubits, 'readout')

circ = QuantumCircuit(qb, readout)
circ += get_full_stabilizer_circuit(registers)
circ += encode_input_v2(qb)
circ += encode_input_v2(qb).inverse()
circ.h(qb[0])
circ.cx(qb[0], qb[1])
circ.h(qb[0])
circ.cx(qb[0], qb[1])
circ.h(qb[0])
circ.cx(qb[0], qb[1])
# This circuit creates equal superposition of |01> and |11>
circ.snapshot('measure', snapshot_type="density_matrix")
circ.measure(qb, readout)

#circ.draw(output='mpl')

# Run both models
n_shots = 1000
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
# Note: Might have to rerun several times to get fidelity != 1
sv_thermal = results_thermal.data()['snapshots']['density_matrix']['measure'][0]['value']
sv_damping = results_damping.data()['snapshots']['density_matrix']['measure'][0]['value']
print(state_fidelity(sv_damping, sv_thermal))

# %% Further comparison between the two models

print(sv_damping)
print(' ')
print(sv_thermal)
print(state_fidelity(sv_damping, sv_thermal))

#%%
from qiskit.quantum_info import DensityMatrix

test_dm = DensityMatrix(sv_damping)
test_sv = testy.to_statevector()
print(test_sv)


# %% =================  Testing noise model + stabilizer
#from simulator_program.stabilizers import _flagged_stabilizer_XZZXI
from qiskit.quantum_info import partial_trace

# Define our registers (Maybe to be written as function?)
qb = QuantumRegister(5, 'code_qubit')
an = AncillaRegister(2, 'ancilla_qubit')
cr = ClassicalRegister(5, 'syndrome_bit') # The typical register
#cr = get_classical_register(n_cycles, flag) # Advanced list of registers
readout = ClassicalRegister(5, 'readout')

#registers = [qb, an, cr, readout] # Pack them together
registers = StabilizerRegisters(qb, an, cr, readout)
#circ.x(qb[0])
circ = get_empty_stabilizer_circuit(registers)

# Settings for circuit
n_cycles = 1
reset=False
flag=False
recovery=False

# Get the circuit
circ += get_full_stabilizer_circuit(registers,
    n_cycles=n_cycles,
    reset=reset,
    recovery=recovery,
    flag=flag,
)

# Run the circuit
n_shots = 2048
results = execute(
    circ,  
    Aer.get_backend('qasm_simulator'),
    noise_model=None,#phase_amplitude_model(),
    shots=n_shots
).result()

counts = results.get_counts()
#plot_histogram(results.get_counts())
#circ.draw(output='mpl')

# %
# Analyze results
logical = logical_states()
#sv_post_encoding = results.data()['snapshots']['statevector']['stabilizer_0'][0]
#print(sv_post_encoding)
log0 = partial_trace(logical[0],[5,6])
fid = 0
for i in range(n_shots):
    sv_post_encoding = results.data()['snapshots']['statevector']['stabilizer_0'][i]
    partial_sv = partial_trace(sv_post_encoding, [5,6])
    fid += state_fidelity(log0, partial_sv)
print('Average fidelity: ', round(fid/n_shots,5))
# %%
fid = 0
# For using state_fidelity, it is necessary to reset due to ancilla
for i in range(n_shots):
    sv_post_encoding = results.data()['snapshots']['statevector']['stabilizer_0'][i]
    #log0 = logical[0][np.arange(128,step=4)]
    sv_test = sv_post_encoding[0:32]
#    fid += state_fidelity(log0, sv_test)
    fid += state_fidelity(logical[0], sv_post_encoding)
print(fid)

# %% CODE FOR USING PARTIAL TRACE
testy = partial_trace(sv_post_encoding, [5,6])
testx = partial_trace(logical[0], [5,6])
print(state_fidelity(sv_post_encoding, logical[0]))
print(state_fidelity(testy, log0))
# %%
print(np.linalg.norm(sv_test))
# %% Group the counts into bigger sets

output_groups = {'no syndrome, log0 state': 0, 'no syndrome, log1 state': 0,
    'syndrome, log0 state': 0, 'syndrome, log1 state': 0}
for output in counts:
    output_state = int(output[0:5])
    output_msmnt = int(output[6:11])


    log0_list = [0,11,101,110,1001,1010,1100,1111,10001,10010,10100,10111,
        11000,11011,11101,11110]
    log1_list = [1,10,100,111,1000,1011,1101,1110,10000,10011,10101,10110,
        11001,11010,11100,11111]

    # Check if we're in logical zero or 1
    if output_state in log0_list and output_msmnt == 0:
        output_groups['no syndrome, log0 state'] += counts[output]
    elif output_state in log1_list and output_msmnt == 0:
        output_groups['no syndrome, log1 state'] += counts[output]
    elif output_state in log0_list and output_msmnt > 0:
        output_groups['syndrome, log0 state'] += counts[output]
    elif output_state in log1_list and output_msmnt > 0:
        output_groups['syndrome, log1 state'] += counts[output]

plot_histogram(output_groups)

# %%
def logical_states():
    logical_0 = np.zeros(2**5)
    logical_0[0b00000] = 1/4
    logical_0[0b10010] = 1/4
    logical_0[0b01001] = 1/4
    logical_0[0b10100] = 1/4
    logical_0[0b01010] = 1/4
    logical_0[0b11011] = -1/4
    logical_0[0b00110] = -1/4
    logical_0[0b11000] = -1/4
    logical_0[0b11101] = -1/4
    logical_0[0b00011] = -1/4
    logical_0[0b11110] = -1/4
    logical_0[0b01111] = -1/4
    logical_0[0b10001] = -1/4
    logical_0[0b01100] = -1/4
    logical_0[0b10111] = -1/4
    logical_0[0b00101] = 1/4

    logical_1 = np.zeros(2**5)
    logical_1[0b11111] = 1/4
    logical_1[0b01101] = 1/4
    logical_1[0b10110] = 1/4
    logical_1[0b01011] = 1/4
    logical_1[0b10101] = 1/4
    logical_1[0b00100] = -1/4
    logical_1[0b11001] = -1/4
    logical_1[0b00111] = -1/4
    logical_1[0b00010] = -1/4
    logical_1[0b11100] = -1/4
    logical_1[0b00001] = -1/4
    logical_1[0b10000] = -1/4
    logical_1[0b01110] = -1/4
    logical_1[0b10011] = -1/4
    logical_1[0b01000] = -1/4
    logical_1[0b11010] = 1/4

    # Add two ancillas in |0>
    an0 = np.zeros(2**2)
    an0[0] = 1.0

    #logical_1 = np.kron(logical_1, an0)
    #logical_0 = np.kron(logical_0, an0)
    logical_0 = np.kron(an0, logical_0)
    logical_1 = np.kron(an0, logical_1)

    return [logical_0, logical_1]
# %%
