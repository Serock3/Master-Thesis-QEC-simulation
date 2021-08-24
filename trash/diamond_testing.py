# %% Import modules
import numpy as np
from qiskit import *

from qiskit.quantum_info import DensityMatrix
from qiskit.quantum_info import state_fidelity

# Our own files
from simulator_program.custom_noise_models import phase_amplitude_model
from simulator_program.custom_noise_models import thermal_relaxation_model_V2
from simulator_program.custom_transpiler import *
from simulator_program.stabilizers import *


# %% Density Matrix testing

# Define our registers
qb = QuantumRegister(5, 'code_qubit')
an = AncillaRegister(2, 'ancilla_qubit')
cr = ClassicalRegister(4, 'syndrome_bit') # The typical register
readout = ClassicalRegister(5, 'readout')

registers = StabilizerRegisters(qb, an, cr, readout)
circ = get_empty_stabilizer_circuit(registers)

# Settings for circuit
n_cycles = 1
reset=False
flag=False
recovery=False

#circ.x(qb[0])
circ += encode_input_v2(registers)

circ += get_stabilizer_cycle(registers,
    reset=reset,
    recovery=recovery
)

#circ.snapshot('second_snap', 'density_matrix')
#circ.append(Snapshot('booper','density_matrix',num_qubits=5),qb)
#circ.measure(qb, readout)

# Transpilation
routing_method = 'sabre'  # basic lookahead stochastic sabre
initial_layout = None  # Overwriting the above layout
layout_method = 'sabre'  # trivial 'dense', 'noise_adaptive' sabre
translation_method = None  # 'unroller',  translator , synthesis
repeats = 200
optimization_level = 1
circ_diamond = shortest_transpile_from_distribution(
    circ,
    print_cost=False,
    repeats=repeats,
    routing_method=routing_method,
    initial_layout=initial_layout,
    layout_method=layout_method,
    translation_method=translation_method,
    optimization_level=optimization_level,
    **diamond_device_properties
)
circ_WACQT = shortest_transpile_from_distribution(
    circ,
    print_cost=False,
    repeats=repeats,
    routing_method=routing_method,
    initial_layout=initial_layout,
    layout_method=layout_method,
    translation_method=translation_method,
    optimization_level=optimization_level,
    **WACQT_device_properties
)
#circ_WACQT.draw(output='mpl')
#circ_diamond.draw(output='mpl')

#verify_transpilation(circ, circ_diamond)
#verify_transpilation(circ, circ_WACQT)

# %%
print('Final depth diamond = ', circ_diamond.depth())
print('Final 2qb-gates diamond = ', circ_diamond.num_nonlocal_gates())
print('Final depth WACQT = ', circ_WACQT.depth())
print('Final 2qb- gates WACQT = ', circ_WACQT.num_nonlocal_gates())


# %%
results = execute(
    circ,
    Aer.get_backend('qasm_simulator'),
    noise_model=thermal_relaxation_model_V2(),
    memory=True,
    shots=10,
).result()
print(results.get_counts())
# %%
#print(results.data()['snapshots']['density_matrix'].items())
a = [(name, state) for (name, state) in results.data()[
        'snapshots']['density_matrix'].items()]
# %%
def print_vec_diffs(state_vec1, state_vec2):
    state1 = DensityMatrix(state_vec1).to_statevector()
    state2 = DensityMatrix(state_vec2).to_statevector()

    if state1.dim != state2.dim:
        raise Exception("Error, dims not matching")

    dim = int(np.log2(state1.dim))

    diff = np.round(state1.data-state2.data, 3)
    for i in np.where(diff != 0)[0]:
        print("Diff in", format(i, 'b').zfill(dim), np.round(
            state1.data, 3)[i], ' -> ', np.round(state2.data, 3)[i])


def comp_states_mat(results1, results2):
    """Compares two versions of circuits supposed to be identical. 
    Looks at desity matrix snapshots and measurement counts.
    Works even if register sizer are different and permuted.

    TODO: Currently does not care about shots > 1 for result2.

    Args:
        results1 (result): result() from a qasm execution
        results2 (result): result() from a qasm execution
    """
    snapshot_type = 'density_matrix'  # 'density_matrix'# TODO: Make this automatic
    snapshot_list1 = [(name, state) for (name, state) in results1.data()[
        'snapshots'][snapshot_type].items()]
    snapshot_list2 = [(name, state) for (name, state) in results2.data()[
        'snapshots'][snapshot_type].items()]

    # print('Purity of encoded state = ', purity(snapshot_list2[0][1][0]['value']))

    if len(snapshot_list2[0][1]) > 1:
        print('SEVERAL MATRICES IN comp_states_mat FOR results2, SOMETHING NOT RIGHT')
    if len(snapshot_list1[0][1]) > 1:
        print('SEVERAL MATRICES IN comp_states_mat FOR results1, SOMETHING NOT RIGHT')

    for i in range(len(snapshot_list1)):
        for shot in range(1):
            fidelity = np.round(state_fidelity(
                snapshot_list2[i][1][0]['value'], snapshot_list1[i][1][0]['value']), 3)

        print('Fidelity', snapshot_list2[i][0], fidelity)
        if fidelity != 1:
            print_vec_diffs(
                snapshot_list1[i][1][0]['value'], snapshot_list2[i][1][0]['value'])
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


def verify_transpilation(circ, transpiled_circuit):
    results1 = execute(
        transpiled_circuit,
        Aer.get_backend('qasm_simulator'),
        shots=1
    ).result()
    results2 = execute(
        circ,
        Aer.get_backend('qasm_simulator'),
        shots=1
    ).result()

    comp_states_mat(results1, results2)
