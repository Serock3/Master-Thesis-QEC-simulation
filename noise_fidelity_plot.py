# %% Import modules
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from qiskit import *
from qiskit.visualization import plot_histogram

# Import from Qiskit Aer noise module
from qiskit.providers.aer.noise import pauli_error,depolarizing_error
from qiskit.providers.aer.noise import (thermal_relaxation_error, 
                                        amplitude_damping_error, 
                                        phase_damping_error, 
                                        phase_amplitude_damping_error
                                        )

from qiskit.quantum_info import partial_trace
from qiskit.quantum_info import DensityMatrix
from qiskit.quantum_info import state_fidelity

# Our own files
from simulator_program.custom_noise_models import phase_amplitude_model
from simulator_program.custom_noise_models import thermal_relaxation_model
from simulator_program.custom_transpiler import *
from simulator_program.stabilizers import *

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

    return [logical_0, logical_1]

# %% =================  Testing noise model + stabilizer ======================
# DEFINE AND TRANSPILE THE CIRCUIT

# Define our registers
qb = QuantumRegister(5, 'code_qubit')
an = AncillaRegister(2, 'ancilla_qubit')
cr = ClassicalRegister(5, 'syndrome_bit') # The typical register
#cr = get_classical_register(n_cycles, flag) # Advanced list of registers
readout = ClassicalRegister(5, 'readout')

registers = StabilizerRegisters(qb, an, cr, readout)
circ = get_empty_stabilizer_circuit(registers)

# Settings for circuit
n_cycles = 1
reset=False
flag=False
recovery=False

# Get the circuit
circ.x(qb[0])
circ += get_full_stabilizer_circuit(registers,
    n_cycles=n_cycles,
    reset=reset,
    recovery=recovery,
    flag=flag,
)

# Transpilation
routing_method = 'sabre'  # basic lookahead stochastic sabre
initial_layout = None  # Overwriting the above layout
layout_method = 'sabre'  # trivial 'dense', 'noise_adaptive' sabre
translation_method = None  # 'unroller',  translator , synthesis
repeats = 10
optimization_level = 1
circ_t = shortest_transpile_from_distribution(
    circ,
    print_depths=False,
    repeats=repeats,
    routing_method=routing_method,
    initial_layout=initial_layout,
    layout_method=layout_method,
    translation_method=translation_method,
    optimization_level=optimization_level,
    **WAQCT_device_properties
)

# Error free run for fidelity measurements
results_t = execute(
    circ_t,
    Aer.get_backend('qasm_simulator'),
    noise_model=None,
    shots=1,
).result()
logical_t = results_t.data()['snapshots']['statevector']['stabilizer_0'][0]

# %% Analyze results (NOTE: Might take a long time due to n_shots)
T2_list = np.arange(40, 81, 2)*1e3 # 40-80 mus
t_cz_list = np.arange(100,301, 10) # 100-300 ns
logical = logical_states()
log0 = DensityMatrix(logical[1])
n_shots = 2048*8

# No transpilation, vary T2, fixed t_cz
fid_T2 = np.zeros(len(T2_list))
for i in range(len(fid_T2)):
    T2 = T2_list[i]

    # Run the circuit
    results = execute(
        circ,  
        Aer.get_backend('qasm_simulator'),
        noise_model=thermal_relaxation_model(T2=T2, t_cz=200),
        shots=n_shots
    ).result()
    counts = results.get_counts()

    # Analyze results
    for j in range(n_shots):
        statevector = results.data()['snapshots']['statevector']['stabilizer_0'][j]
        partial_sv = partial_trace(statevector, [5,6])
        fid_T2[i] += state_fidelity(log0, partial_sv)
fid_T2 /= n_shots
print('Finished varying T2 without transpilation')

# No transpilation, vary t_cz, fixed T2
fid_cz = np.zeros(len(t_cz_list))
for i in range(len(fid_cz)):
    t_cz = t_cz_list[i]

    # Run the circuit
    results = execute(
        circ,  
        Aer.get_backend('qasm_simulator'),
        noise_model=thermal_relaxation_model(T2=60e3, t_cz=t_cz),
        shots=n_shots
    ).result()
    counts = results.get_counts()

    # Analyze results
    for j in range(n_shots):
        statevector = results.data()['snapshots']['statevector']['stabilizer_0'][j]
        partial_sv = partial_trace(statevector, [5,6])
        fid_cz[i] += state_fidelity(log0, partial_sv)
fid_cz /= n_shots
print('Finished varying t_cz without transpilation')

# With transpilation, vary T2, fixed t_cz
fid_T2_t = np.zeros(len(T2_list))
for i in range(len(fid_T2)):
    T2 = T2_list[i]

    # Run the circuit
    results = execute(
        circ_t,  
        Aer.get_backend('qasm_simulator'),
        noise_model=thermal_relaxation_model(T2=T2, t_cz=200),
        shots=n_shots
    ).result()
    counts = results.get_counts()

    # Analyze results
    for j in range(n_shots):
        statevector = results.data()['snapshots']['statevector']['stabilizer_0'][j]
        fid_T2_t[i] += state_fidelity(logical_t, statevector)
fid_T2_t /= n_shots
print('Finished varying T2 with transpilation')

# With transpilation, vary t_cz, fixed T2
fid_cz_t = np.zeros(len(t_cz_list))
for i in range(len(fid_cz)):
    t_cz = t_cz_list[i]

    # Run the circuit
    results = execute(
        circ_t,  
        Aer.get_backend('qasm_simulator'),
        noise_model=thermal_relaxation_model(T2=60e3, t_cz=t_cz),
        shots=n_shots
    ).result()
    counts = results.get_counts()

    # Analyze results
    for j in range(n_shots):
        statevector = results.data()['snapshots']['statevector']['stabilizer_0'][j]
        fid_cz_t[i] += state_fidelity(logical_t, statevector)
fid_cz_t /= n_shots
print('Finished varying t_cz with transpilation')
# %%
fig, axs = plt.subplots(2, figsize=(14, 10))
ax1 = axs[0]
ax2 = axs[1]
ax1.plot(T2_list*1e-3, fid_T2,'o-')
ax1.plot(T2_list*1e-3, fid_T2_t,'o-')
ax2.plot(t_cz_list, fid_cz,'o-')
ax2.plot(t_cz_list, fid_cz_t,'o-')


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

    return [logical_0, logical_1]

