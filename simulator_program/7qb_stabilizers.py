# Functions for running the [[7,1,3]] steane code, mainly on the 10qb chip
# Stabi
e#s  %% Import modules
import numpy as np
from qiskit import (QuantumCircuit,
                    QuantumRegister, 
                    ClassicalRegister, 
                    AncillaRegister, 
                    execute, 
                    Aer
                    )
from qiskit.providers.aer.extensions.snapshot_statevector import *
from qiskit.providers.aer.extensions.snapshot_density_matrix import *
# %% General functions
class StabilizerRegistersSteane:

    def __init__(self,
            qbReg=QuantumRegister(7, 'code_qubit'),
            anReg=AncillaRegister(3, 'ancilla_qubit'),
            clReg=ClassicalRegister(6, 'syndrome_bit'),
            readout=ClassicalRegister(7, 'readout')
    ):
        self.QubitRegister = qbReg
        self.AncillaRegister = anReg
        self.SyndromeRegister = clReg
        self.ReadoutRegister = readout

def get_full_stabilizer_circuit(registers, n_cycles=1,
        reset=True, recovery=False, flag=True):
    """Returns the circuit for a full repeating stabilizer circuit, including encoding, 
    n_cycles of repeated stabilizers (with optional flags and recovery) and final measurement.
    """


    # Unpack registers
    #qbReg, anReg, clReg, readout = registers
    qbReg = registers.QubitRegister
    anReg = registers.AncillaRegister
    clReg = registers.SyndromeRegister
    readout = registers.ReadoutRegister
    if not anReg.size == 2 and not anReg.size == 5:
        raise Exception('Ancilla register must be of size 2 or 5')

    # Define the circuit
    # circ = get_empty_stabilizer_circuit(registers)

    # Encode the state
    circ = encode_input_v2(registers)
    circ.snapshot('post_encoding', 'statevector')

    # Stabilizer
    circ = get_repeated_stabilization(registers, n_cycles=1,
        reset=reset, recovery=recovery, flag=flag)

    # Final readout
    circ.measure(qbReg, readout)
    circ.snapshot_statevector('post_measure')

    return circ
    
def get_repeated_stabilization(registers, n_cycles=1,
        reset=True, recovery=False, flag=True, snapshot_type='statevector'):

    circ = get_empty_stabilizer_circuit(registers)

    for current_cycle in range(n_cycles):
        if flag is True:
            circ += flagged_stabilizer_cycle(registers,
                reset=reset,
                recovery=recovery,
                current_cycle=current_cycle,
            )
        else:
            circ += unflagged_stabilizer_cycle(registers,
                reset=reset,
                recovery=recovery,
                current_cycle=current_cycle
            )
        if snapshot_type:
            circ.snapshot('stabilizer_' + str(current_cycle), snapshot_type)
    return circ

def get_empty_stabilizer_circuit(registers):
    """Create an empty qiskit circuit adapted for stabilizer circuits"""

    # Unpack registers
    #qbReg, anReg, clReg, readout = registers
    qbReg = registers.QubitRegister
    anReg = registers.AncillaRegister
    clReg = registers.SyndromeRegister
    readout = registers.ReadoutRegister

    circ = QuantumCircuit(qbReg, anReg)
    if isinstance(clReg, list):
        circ = QuantumCircuit(qbReg, anReg)
        for reg_type in clReg:
            for reg_index in reg_type:
                for reg in reg_index:
                    circ.add_register(reg)
    else:
        circ.add_register(clReg)
    circ.add_register(readout)

    return circ

def encode_input_steane(registers):
    """Encode the input into logical 0 and 1 for the [[7,1,3]] code. This
    assumes that the 0:th qubit is the original state |psi> = a|0> + b|1>
    """

    # Unpack registers
    qbReg = registers.QubitRegister
    circ = QuantumCircuit(qbReg)

    circ.h(qbReg[1])
    circ.h(qbReg[2])
    circ.h(qbReg[3])

    circ.cx(qbReg[3], qbReg[5])
    circ.cx(qbReg[2], qbReg[4])
    circ.cx(qbReg[1], qbReg[6])

    circ.cx(qbReg[0], qbReg[3])
    circ.cx(qbReg[0], qbReg[2])
    circ.cx(qbReg[0], qbReg[1])

    circ.cx(qbReg[4], qbReg[0])
    circ.cx(qbReg[5], qbReg[0])
    circ.cx(qbReg[6], qbReg[0])

    circ.cx(qbReg[1], qbReg[5])
    circ.cx(qbReg[2], qbReg[6])
    circ.cx(qbReg[3], qbReg[4])

    return circ

def get_classical_register(n_cycles, flag=True):
    """Generate lists of classical registers for storing all measurement data.
    
    The function accepts the flag bool to determine whether to create 
    registers for additional flag measurements. The names are as follows:
    
    syndrome_register: The standard four unflagged stabilizer measurements.
        Without flags, its dimensions are [n_cycles][1][4] and with flags they
        are [n_cycles][current_step][4] where current step represents where
        in the full flagged stabilizer cycle measurements were made (0-3).

    flag_register: All flag measurements, only appears if flag is set to True.
        It has the dimensions [n_cycles][current_step][1].

    ancilla_msmnt_register: Contains the extra single ancilla measurement
        performed along with a flag measurement in the flagged cycle. Only
        appears if flag is set to True. Dimensions are
        [n_cycles][current_step][1]
    """

    if flag:
        # List of registers for each iteration of the conditional
        # step of 'all four unflagged stabilizers'
        syndrome_register = [
            [ ClassicalRegister( 4, 'syndrome_cycle_' +str(i) +'_step_' +str(j)) 
            for j in range(4) ] for i in range(n_cycles) ]

        # List of registers for each step in the flagged stabilizer cycle
        flag_register = [ 
            [ ClassicalRegister( 1, 'flag_cycle_' +str(i) +'_step_' +str(j))
            for j in range(4) ] for i in range(n_cycles) ]

        # List of registers for the single stabilizer run with flag
        ancilla_msmnt_register = [
            [ ClassicalRegister( 1, 'ancilla_cycle_' +str(i) +'_step_' +str(j))
            for j in range(4) ] for i in range(n_cycles) ]

        return [syndrome_register, flag_register, ancilla_msmnt_register]
    
    else:
        # A register of four bits per cycle
        syndrome_register = [
            [ ClassicalRegister( 4, 'syndrome_cycle_' +str(i) +'_step_' +str(j)) 
            for j in range(1) ] for i in range(n_cycles) ]
        return [syndrome_register]


# %% All unflagged stabilizers
def stabilizer_cycle_7qb(registers, reset=True, recovery=False,
        current_cycle=0):
    """Run all four stabilizers without flags, as well as an optional
    recovery. The input current_step is only relevant for flagged cycles, and
    should be set to 0 otherwise.
    """
    # Create circuit and run stabilizers
    circ = get_empty_stabilizer_circuit(registers)
    circ += _stabilizer_X0456(registers, reset, current_cycle)
    circ += _stabilizer_X1356(registers, reset, current_cycle)
    circ += _stabilizer_X2345(registers, reset, current_cycle)
    circ += _stabilizer_Z0456(registers, reset, current_cycle)
    circ += _stabilizer_Z1356(registers, reset, current_cycle)
    circ += _stabilizer_Z2345(registers, reset, current_cycle)

    # Recovery
    if recovery is True:
        circ += unflagged_recovery(registers, reset, current_cycle)
    
    return circ

def _stabilizer_X0456(registers, reset=True, current_cycle=0):
    """Gives the circuit for running the regular XZZXI stabilizer without flag.
    The current_step input should be set to zero unless running flagged cycles.
    """

    # Create a circuit
    qbReg = registers.QubitRegister
    anReg = registers.AncillaRegister
    clReg = registers.SyndromeRegister
    circ = get_empty_stabilizer_circuit(registers)
    anQb = anReg[0]

    circ.h(anQb)
    circ.cx(anQb, qbReg[0])
    circ.cx(anQb, qbReg[1])
    circ.cx(anQb, qbReg[5])
    circ.cx(anQb, qbReg[6])
    circ.h(anQb)

    # Measure
    if isinstance(clReg, list):
        syndrome_reg = clReg[0]
        circ.measure(anQb, syndrome_reg[current_cycle][current_step][0])
    else:
        circ.measure(anQb, clReg[0])
    
    # Reset
    if reset:
        circ.reset(anQb)

    return circ

def _stabilizer_X1356(registers, reset=True, current_cycle=0):
    """Gives the circuit for running the regular XZZXI stabilizer without flag.
    The current_step input should be set to zero unless running flagged cycles.
    """

    # Create a circuit
    qbReg = registers.QubitRegister
    anReg = registers.AncillaRegister
    clReg = registers.SyndromeRegister
    circ = get_empty_stabilizer_circuit(registers)
    anQb = anReg[1]
    
    circ.h(anQb)
    circ.cx(anQb, qbReg[0])
    circ.cx(anQb, qbReg[2])
    circ.cx(anQb, qbReg[4])
    circ.cx(anQb, qbReg[6])
    circ.h(anQb)

    # Measure
    if isinstance(clReg, list):
        syndrome_reg = clReg[0]
        circ.measure(anQb, syndrome_reg[current_cycle][current_step][1])
    else:
        circ.measure(anQb, clReg[1])
    
    # Reset
    if reset:
        circ.reset(anQb)

    return circ

def _stabilizer_X2345(registers, reset=True, current_cycle=0):
    """Gives the circuit for running the regular XZZXI stabilizer without flag.
    The current_step input should be set to zero unless running flagged cycles.
    """

    # Create a circuit
    qbReg = registers.QubitRegister
    anReg = registers.AncillaRegister
    clReg = registers.SyndromeRegister
    circ = get_empty_stabilizer_circuit(registers)
    anQb = anReg[2]
    
    circ.h(anQb)
    circ.cx(anQb, qbReg[0])
    circ.cx(anQb, qbReg[3])
    circ.cx(anQb, qbReg[4])
    circ.cx(anQb, qbReg[5])
    circ.h(anQb)

    # Measure
    if isinstance(clReg, list):
        syndrome_reg = clReg[0]
        circ.measure(anQb, syndrome_reg[current_cycle][current_step][2])
    else:
        circ.measure(anQb, clReg[2])
    
    # Reset
    if reset:
        circ.reset(anQb)

    return circ

def _stabilizer_Z0456(registers, reset=True, current_cycle=0):
    """Gives the circuit for running the regular XZZXI stabilizer without flag.
    The current_step input should be set to zero unless running flagged cycles.
    """

    # Create a circuit
    qbReg = registers.QubitRegister
    anReg = registers.AncillaRegister
    clReg = registers.SyndromeRegister
    circ = get_empty_stabilizer_circuit(registers)
    anQb = anReg[0]
    
    circ.cx(qbReg[0], anQb)
    circ.cx(qbReg[1], anQb)
    circ.cx(qbReg[5], anQb)
    circ.cx(qbReg[6], anQb)

    # Measure
    if isinstance(clReg, list):
        syndrome_reg = clReg[0]
        circ.measure(anQb, syndrome_reg[current_cycle][current_step][3])
    else:
        circ.measure(anQb, clReg[3])
    
    # Reset
    if reset:
        circ.reset(anQb)

    return circ

def _stabilizer_Z1356(registers, reset=True, current_cycle=0):
    """Gives the circuit for running the regular XZZXI stabilizer without flag.
    The current_step input should be set to zero unless running flagged cycles.
    """

    # Create a circuit
    qbReg = registers.QubitRegister
    anReg = registers.AncillaRegister
    clReg = registers.SyndromeRegister
    circ = get_empty_stabilizer_circuit(registers)
    anQb = anReg[1]
    
    circ.cx(qbReg[0], anQb)
    circ.cx(qbReg[2], anQb)
    circ.cx(qbReg[4], anQb)
    circ.cx(qbReg[6], anQb)

    # Measure
    if isinstance(clReg, list):
        syndrome_reg = clReg[0]
        circ.measure(anQb, syndrome_reg[current_cycle][current_step][4])
    else:
        circ.measure(anQb, clReg[4])
    
    # Reset
    if reset:
        circ.reset(anQb)

    return circ

def _stabilizer_Z2345(registers, reset=True, current_cycle=0):
    """Gives the circuit for running the regular XZZXI stabilizer without flag.
    The current_step input should be set to zero unless running flagged cycles.
    """

    # Create a circuit
    qbReg = registers.QubitRegister
    anReg = registers.AncillaRegister
    clReg = registers.SyndromeRegister
    circ = get_empty_stabilizer_circuit(registers)
    anQb = anReg[2]
    
    circ.cx(qbReg[0], anQb)
    circ.cx(qbReg[3], anQb)
    circ.cx(qbReg[4], anQb)
    circ.cx(qbReg[5], anQb)

    # Measure
    if isinstance(clReg, list):
        syndrome_reg = clReg[0]
        circ.measure(anQb, syndrome_reg[current_cycle][current_step][5])
    else:
        circ.measure(anQb, clReg[5])
    
    # Reset
    if reset:
        circ.reset(anQb)

    return circ



# TODO: Add recovery


# %% 10 qb TRANSPILER testing
from qiskit.compiler import transpile
from qiskit.transpiler import PassManager, CouplingMap, Layout
from qiskit.visualization import plot_circuit_layout
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary
from qiskit.circuit.library.standard_gates import iSwapGate, SwapGate, SGate, CZGate
from qiskit import QuantumCircuit, QuantumRegister
import warnings

def shortest_transpile_from_distribution(circuit, repeats=40, print_depths=True, **kwargs):
    depth = 10000
    for i in range(repeats):
        with warnings.catch_warnings():  # sabre causes deprication warning, this will ignore them
            warnings.simplefilter("ignore")
            transpiled_circuit_tmp = transpile(circuit, **kwargs)
        if print_depths:
            print('depth: ', transpiled_circuit_tmp.depth())
        if transpiled_circuit_tmp.depth() < depth:
            depth = transpiled_circuit_tmp.depth()
            transpiled_circuit = transpiled_circuit_tmp
    return transpiled_circuit


basis_gates = ['id', 'u1', 'h', 'u3', 'swap', 'cz']
couplinglist = [[0,4], [0,1], [1,4], [1,5], [1,2], [2,5], [2,6], [2,3],
    [3,6], [4,7], [4,5], [5,7], [5,8], [5,6], [6,8], [7,8], [7,9], [8,9]]
reverse_couplinglist = [[y, x] for [x, y] in couplinglist]
coupling_map = CouplingMap(
    couplinglist=couplinglist+reverse_couplinglist, description='A hexagoal 7qb code with two ancillas')

# Dict with device properties of the WAQCT QC to be used for transpilation.
WAQCT_device_properties = {
    "basis_gates": basis_gates, "coupling_map": coupling_map}

def _add_custom_device_equivalences():
    """ Ads custom gate equivalences to the SessionEquivalenceLibrary for transpilation
    NOTE: One needs to be run once!
    """
    print('Adding custom device equivalences')
    q = QuantumRegister(2, 'q')
    def_swap = QuantumCircuit(q)
    for inst, qargs, cargs in [
            (iSwapGate(), [q[0], q[1]], []),
            (CZGate(), [q[0], q[1]], []),
            (SGate().inverse(), [q[1]], []),
            (SGate().inverse(), [q[0]], [])
    ]:
        def_swap.append(inst, qargs, cargs)
    SessionEquivalenceLibrary.add_equivalence(SwapGate(), def_swap)

_add_custom_device_equivalences()

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

    TODO: Make it work for non-trivial measurement outcomes

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


# %% Internal testing/running
if __name__ == '__main__'
    # Define quantum registers and circuit
    qb = QuantumRegister(7, 'qubit')
    an = AncillaRegister(3, 'ancilla_qubit')
    readout= ClassicalRegister(7, 'readout')
    cr = ClassicalRegister(6, 'syndrome bits')

    registers = StabilizerRegistersSteane(qb, an, cr, readout)
    #circ = get_empty_stabilizer_circuit(registers)
    circ = QuantumCircuit(qb,readout)
    circ += encode_input_steane(registers)
    circ.barrier()
    #circ.x(qb[0])
    #circ += stabilizer_cycle_7qb(registers, reset=True, recovery=False)
    circ.barrier()
    circ.snapshot('test','density_matrix')
    circ.measure(qb, readout)

    print('Starting transpilation')
    # Transpilation
    routing_method = 'sabre'  # basic lookahead stochastic sabre
    initial_layout = None  # Overwriting the above layout
    layout_method = 'sabre'  # trivial 'dense', 'noise_adaptive' sabre
    translation_method = None  # 'unroller',  translator , synthesis
    repeats = 200
    optimization_level = 3
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
    verify_transpilation(circ, circ_t)

    # Run it
    n_shots = 2000
    results = execute(
        circ_t,  
        Aer.get_backend('qasm_simulator'),
        noise_model=None,
        shots=n_shots
    ).result()

    # Plot results
    counts = results.get_counts()
    plot_histogram(counts)
# %%
