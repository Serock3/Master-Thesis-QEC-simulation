# Script to transpile and manage circuits
# Includes functionality to e.g. save circuits and very transpilation

# %% Imports
from qiskit.quantum_info import purity
from qiskit.quantum_info.states.measures import state_fidelity
from stabilizers import*
from stabilizers import _unflagged_stabilizer_XZZXI
import numpy as np
from IPython.display import display

from matplotlib import pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, AncillaRegister, ClassicalRegister, Aer, execute
from custom_transpiler import *
from custom_transpiler import WACQT_device_properties, diamond_device_properties
from qiskit.quantum_info.states.densitymatrix import DensityMatrix
from qiskit.quantum_info.states.statevector import Statevector
from idle_noise import get_circuit_time
from qiskit.visualization import array_to_latex, latex
# from qiskit.visualization.latex import math
# %%


def print_vec_diffs(state_vec1, state_vec2):
    state1 = DensityMatrix(state_vec1).to_statevector()
    state2 = DensityMatrix(state_vec2).to_statevector()

    if state1.dim != state2.dim:
        raise Exception("Error, dims not matching")

    dim = int(np.log2(state1.dim))

    diff = np.round(state1.data-state2.data, 3)
    for i in np.where(diff != 0)[0]:
        print(f"Diff in |{format(i, 'b').zfill(dim)}>:", np.round(
            state1.data, 3)[i], '->', np.round(state2.data, 3)[i])


def comp_states_mat(results1, results2, n_cycles):
    """Compares two versions of circuits supposed to be identical. 
    Looks at density matrix snapshots and measurement counts.
    Works even if register sizes are different and permuted.

    TODO: Make it work for non-trivial measurement outcomes

    Args:
        results1 (result): result() from a qasm execution
        results2 (result): result() from a qasm execution
    """
    for current_cycle in range(n_cycles):
        label = get_snapshot_label("dm", False, current_cycle)
        fidelity = np.round(state_fidelity(
            results1.data()[label], results2.data()[label]), 3)
        print('Fidelity', label, fidelity)
        if fidelity != 1:
            print_vec_diffs(
                results1.data()[label], results2.data()[label])
    # print('Purity of encoded state = ', purity(snapshot_list2[0][1][0]['value']))
    # TODO: Do something useful with this information
    try:
        counts = results1.get_counts()
        print('Counts 1\n', counts)
    except:
        pass
    try:
        counts = results2.get_counts()
        print('Counts 2\n', counts)
    except:
        pass


def verify_transpilation(circ, transpiled_circuit, n_cycles):
    results1 = execute(
        transpiled_circuit,
        Aer.get_backend('aer_simulator'),
        shots=1
    ).result()
    results2 = execute(
        circ,
        Aer.get_backend('aer_simulator'),
        shots=1
    ).result()

    comp_states_mat(results1, results2, n_cycles)


# display(WACQT_device_properties['coupling_map'].draw())
# %% Transpile the (v2) encoding circuit for [[5,1,3]]
n_cycles = 3
reset = True
flag = False
recovery = True

qb = QuantumRegister(5, 'code_qubit')
an = AncillaRegister(2, 'ancilla_qubit')
cr = get_classical_register(n_cycles, flag=flag) # Advanced list of registers
registers = StabilizerRegisters(qb, an, cr)
# circ = QuantumCircuit()
# circ += encode_input_v2(registers)
# circ.snapshot('post_encoding', 'density_matrix')

circ = get_repeated_stabilization(registers,
                                   n_cycles=n_cycles,
                                   reset=reset,
                                   recovery=recovery,
                                   num_ancillas=1,
                                   snapshot_type=['exp','dm'])
# circ = get_full_stabilizer_circuit(registers, n_cycles, reset=reset,
#                                    recovery=recovery,
#                                    num_ancillas=1,
#                                    snapshot_type=['exp','dm'],
#                                    conditional=False)

routing_method = 'sabre'  # basic lookahead stochastic sabre
# initial_layout = {qb[0]: 0,
#                   qb[1]: 1,
#                   qb[2]: 2,
#                   qb[3]: 3,
#                   qb[4]: 4,
#                   an[0]: 5,
#                   an[1]: 6}
# initial_layout = {an[0]: 0,
#                     an[1]: 1,
#                     qb[0]: 2,
#                     qb[1]: 3,
#                     qb[2]: 4,
#                     qb[3]: 5,
#                     qb[4]: 6}
initial_layout = None  # Overwriting the above layout
layout_method = 'sabre'  # trivial 'dense', 'noise_adaptive' sabre
translation_method = None  # 'unroller',  translator , synthesis
repeats = 10
optimization_level = 1
circ_t = shortest_transpile_from_distribution(circ,
                                              #   cost_func=depth_cost_func,
                                              repeats=repeats,
                                              print_cost=True,
                                              routing_method=routing_method,
                                              initial_layout=initial_layout,
                                              layout_method=layout_method,
                                              translation_method=translation_method,
                                              optimization_level=optimization_level,
                                              coupling_map=diamond_device_properties['coupling_map'], 
                                              **{'basis_gates': ['id', 'u1', 'x', 'y', 'z', 'sx', 'sy', 'iswap', 'cz', 'barrier', 'set_density_matrix',
                                                                                                                       'save_density_matrix', 'save_expval']})
#   **WACQT_device_properties)

print(f'Final circuit time = {get_circuit_time(circ_t)["end"]} ns')
print('Final depth = ', circ_t.depth())
print('Final #2qb gates = ', circ_t.num_nonlocal_gates())
print('Final gates = ', circ_t.count_ops())
# verify_transpilation(circ, circ_t,n_cycles)
# %% print to qasm to make circuit exportable to IBM quantum experience
display(circ_t.draw(output='mpl'))  # output='mpl'
# print(circ_t)

# %%
def print_qasm(circ):
    print(circ_t.qasm())

def save_circ(circ, name):
    import pickle
    with open(name, 'wb') as transpiled_circuit_file:
        pickle.dump(circ, transpiled_circuit_file)

def load_circ(name):
    import pickle
    if name not in locals():
        with open(name, 'rb') as transpiled_circuit_file:
            circ = pickle.load(transpiled_circuit_file)
    return circ