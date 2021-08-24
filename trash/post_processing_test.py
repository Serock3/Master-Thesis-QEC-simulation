# %% Import modules
from qiskit.aqua.utils import get_subsystems_counts
from qiskit.circuit.library import XGate, ZGate
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from qiskit import *
from qiskit.visualization import plot_histogram

# Import from Qiskit Aer noise module
from qiskit.providers.aer.noise import pauli_error, depolarizing_error
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
from simulator_program.custom_noise_models import thermal_relaxation_model_V2, pauli_noise_model
from simulator_program.custom_transpiler import *
from simulator_program.stabilizers import *

from simulator_program.stabilizers import _get_stabilizer_XZZXI
# %% =================  Testing noise model + stabilizer ======================
# DEFINE AND TRANSPILE THE CIRCUIT

# Define our registers
qb = QuantumRegister(5, 'code_qubit')
an = AncillaRegister(2, 'ancilla_qubit')
cr = ClassicalRegister(5, 'syndrome_bit')  # The typical register
readout = ClassicalRegister(5, 'readout')

registers = StabilizerRegisters(qb, an, cr, readout)

# Settings for circuit
n_cycles = 1
reset = True
flag = False
recovery = False

circ = encode_input_v2(registers)
circ.z(qb[3])  # Added manual error to correct, should induce syndrome 1
circ += get_stabilizer_cycle(registers,
                                   reset=reset,
                                   recovery=recovery
                                   )
circ.barrier()
# circ.snapshot('stabilizer_0', 'statevector') # Gives the same result
circ.append(Snapshot('stabilizer_0', 'statevector', num_qubits=5), qb)
circ.draw()
# %% Basic post processing (always correct x(1))

# List containing the recovery for every possible syndrome. The entry at an index equalt a syndrome is a
# list containing lists of touples, in the touples are a gate and qargs it applies to, which corrects the error
syndrome_table = [[],
                  [(XGate, 1)],
                  [(ZGate, 4)],
                  [(XGate, 2)],
                  [(ZGate, 2)],
                  [(ZGate, 0)],
                  [(XGate, 3)],
                  [(XGate, 2), (ZGate, 2)],
                  [(XGate, 0)],
                  [(ZGate, 3)],
                  [(ZGate, 1)],
                  [(XGate, 1), (ZGate, 1)],
                  [(XGate, 4)],
                  [(XGate, 0), (ZGate, 0)],
                  [(XGate, 4), (ZGate, 4)],
                  [(XGate, 3), (ZGate, 3)]]


def post_process_v1(state, syndrome) -> List[float]:
    """Version one of post processing the state. Creates a QuantumCircuit and initializes it as state_vec, 
    then applies correction to according to syndrome.

    Args:
        state (list): [description]

    Returns:
        list[float]: The corrected statevector
    """
    circ = QuantumCircuit(int(np.log2(state.shape[0])))
    circ.initialize(state, [circ.qubits[i]
                            for i in range(7)])  # [2,3,4,5,6,0,1]
    for correction_stratagy in syndrome_table[int(syndrome, 2)]:
        circ.append(correction_stratagy[0](), [correction_stratagy[1]])
    results = execute(
        circ,
        Aer.get_backend('statevector_simulator'),
    ).result()

    return results.get_statevector()


def state_print(statevec):
    print(np.real(np.round(statevec, 3)))


results = execute(
    circ,
    Aer.get_backend('qasm_simulator'),
    noise_model=None,
    shots=2,
    memory=True
).result()
syndromes = [str.split()[1] for str in results.get_memory()]
state = results.data()['snapshots']['statevector']['stabilizer_0'][0]
correct_state = logical_states(include_ancillas='back')[0]
state_print(post_process_v1(state, syndromes[0]))
print(correct_state)
print(state_fidelity(correct_state, post_process_v1(state, syndromes[0])))

# %% Test fidelity after correction of each syndrome

for errors in syndrome_table:
    circ = encode_input_v2(registers)
    for ops in errors:
        # Added manual error to correct, should induce syndrome 1
        circ.append(ops[0](), [ops[1]])
    circ += get_stabilizer_cycle(registers,
                                       reset=reset,
                                       recovery=recovery
                                       )
    circ.barrier()
    # circ.snapshot('stabilizer_0', 'statevector') # Gives the same result

    circ.append(Snapshot('stabilizer_0', 'statevector', num_qubits=5), qb)
    results = execute(
        circ,
        Aer.get_backend('qasm_simulator'),
        noise_model=None,
        shots=1,
        memory=True
    ).result()
    print(state_fidelity(correct_state, post_process_v1(state, syndromes[0])))

# %% post process for multiple syndromes


def post_process_v2(state, syndromes) -> List[float]:
    """Version two of post-processing. Takes one statevector and it corresponding syndrome,
    applies post-processing and returns a corrected statevector.

    Processes the syndromes using XOR to find where new errors are introduced.

    Args:
        state (list): comples statevector
        syndromes (list(int)): list of syndromes as integers ('0'th element corresponds to 'stabilizer_' etc)

    Returns:
        List[float]: the processed statevector
    """

    # Convert to only new syndromes
    syndromes = [syndromes[0]] + [syndromes[i-1] ^ syndromes[i]
                                  for i in range(1, len(syndromes))]

    circ = QuantumCircuit(int(np.log2(state.shape[0])))
    circ.initialize(state, [circ.qubits[i]
                            for i in range(7)])  # [2,3,4,5,6,0,1]

    for syndrome in syndromes:
        for correction_stratagy in syndrome_table[syndrome]:
            circ.append(correction_stratagy[0](), [correction_stratagy[1]])
    results = execute(
        circ,
        Aer.get_backend('statevector_simulator'),
    ).result()

    return results.get_statevector()


def post_process_v2_all_shots(results, n_cycles):
    """Wrapper for post_process_v2 that processes every statevector for a repeated stabilizer cycle
    simulation with memory=True.

    Args:
        results (Results object): results from completed simulation
        n_cycles (int): number of stabilizer cycles in simulated circuit

    Returns:
        List[List[float]]: List of corrected statvectors where first index indicates cycle, second shots
    """

    # TODO: If this crashes a lot, make it yield instead of return? (Or just return fidelities instead)
    mem = results.get_memory()
    shots = len(mem)
    statevector_dim = results.data(
    )['snapshots']['statevector']['stabilizer_0'][0].shape[0]
    # TODO: Make this format same as with in other places?
    post_processed_states = np.empty(
        (n_cycles, shots, statevector_dim), dtype=np.complex_)
    for current_cycle in range(n_cycles):
        statevectors = results.data(
        )['snapshots']['statevector']['stabilizer_' + str(current_cycle)]
        assert shots == len(statevectors)
        for shot in range(shots):
            # Convert the text format to integers. The order is from right to left, and the last entry is for final measurements ans should be removed
            syndromes = [int(syn, 2) for syn in reversed(
                mem[shot].split()[-(1+current_cycle):])]
            post_processed_states[current_cycle][shot] = post_process_v2(
                statevectors[shot], syndromes)

    return post_processed_states


n_cycles = 3
cr = get_classical_register(n_cycles, reset=reset,
                            recovery=recovery, flag=False)
registers = StabilizerRegisters(qb, an, cr, readout)

circ = encode_input_v2(registers)
circ.z(qb[3])

circ += get_stabilizer_cycle(registers,
                                   reset=reset,
                                   recovery=recovery,
                                   current_cycle=0
                                   )
circ.barrier()
# circ.snapshot('stabilizer_0', 'statevector') # Gives the same result
circ.append(Snapshot('stabilizer_0', 'statevector', num_qubits=5), qb)
circ.barrier()
circ.x(qb[4])
circ += get_stabilizer_cycle(registers,
                                   reset=reset,
                                   recovery=recovery,
                                   current_cycle=1
                                   )
circ.barrier()
# circ.snapshot('stabilizer_0', 'statevector') # Gives the same result
circ.append(Snapshot('stabilizer_1', 'statevector', num_qubits=5), qb)
circ.barrier()
circ.x(qb[2])
circ += get_stabilizer_cycle(registers,
                                   reset=reset,
                                   recovery=recovery,
                                   current_cycle=2
                                   )
circ.barrier()
# circ.snapshot('stabilizer_0', 'statevector') # Gives the same result
circ.append(Snapshot('stabilizer_2', 'statevector', num_qubits=5), qb)

results = execute(
    circ,
    Aer.get_backend('qasm_simulator'),
    noise_model=None,
    shots=10,
    memory=True
).result()


for cycle in post_process_v2_all_shots(results, n_cycles):
    print("\nNew cycle")
    for state_of_shot in cycle:
        print(state_fidelity(correct_state, state_of_shot))

# %% Same as above but with noise model intead of artifical errors, and a fidelity plot
n_cycles = 15
shots = 1024
reset = True
recovery = False
flag = False

qb = QuantumRegister(5, 'code_qubit')
an = AncillaRegister(2, 'ancilla_qubit')
readout = ClassicalRegister(5, 'readout')
cr = get_classical_register(n_cycles, reset=reset,
                            recovery=recovery, flag=False)
registers = StabilizerRegisters(qb, an, cr, readout)

circ = encode_input_v2(registers)
circ += get_repeated_stabilization(registers, n_cycles=n_cycles,
                                   reset=reset, recovery=recovery, flag=flag, snapshot_type='statevector')

results = execute(
    circ,
    Aer.get_backend('qasm_simulator'),
    noise_model=thermal_relaxation_model_V2(),
    shots=shots,
    memory=True
).result()
correct_state = logical_states(include_ancillas='back')[0]
# Using the code in post_process.py now
# from simulator_program.post_process import post_process_statevec_all_shots
fidelities = np.empty((n_cycles, shots))
post_states = post_process_v2_all_shots(results, n_cycles)
for cycle in range(post_states.shape[0]):
    for shot in range(post_states.shape[1]):
        fidelities[cycle][shot] = state_fidelity(
            correct_state, post_states[cycle][shot])
del post_states  # I imagine this could take up quite some memory, so delete it

# %%
plt.errorbar(range(n_cycles), np.mean(fidelities, axis=1),
             yerr=np.std(fidelities, axis=1))
# mean_fid = np.mean(fidelities,axis=1)
mode_fid = np.quantile(fidelities, 0.5, axis=1)
plt.errorbar(range(n_cycles), mode_fid, yerr=(
    np.quantile(fidelities, [0.75, 0.25], axis=1)-mode_fid))
# del fidelities
# del mean_fid
# plt.boxplot(fidelities.T)
# for cycle in range(n_cycles):
#     plt.scatter([cycle]*fidelities.shape[1],fidelities[cycle])
plt.xlabel(r'Error detection cycle $n$')
plt.ylabel('Post-processed fidelity')
# %% TODO: Try it with density matrices?
