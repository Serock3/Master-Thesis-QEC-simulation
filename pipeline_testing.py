#%% Import modules
import numpy as np
from qiskit import *
from qiskit.visualization import plot_histogram

from simulator_program.stabilizers import *
from simulator_program.custom_transpiler import *
from simulator_program.idle_noise import *
from simulator_program.custom_noise_models import (thermal_relaxation_model,
    thermal_relaxation_model_V2,
    WACQT_target_times,
    WACQT_demonstrated_times)
# %%
def mod_unflagged_stabilizer_cycle(registers, reset=True, recovery=False,
        current_cycle=0, current_step=0, num_ancillas=None, 
        include_barriers=True, pipeline=False):
    """Run all four stabilizers without flags, as well as an optional
    recovery. The input current_step is only relevant for flagged cycles, and
    should be set to 0 otherwise.

    NOTE: Maybe use a list if ancilla indices instead? E.g. ancillas = [1,2,1,2]
    Args:
        num_ancillas: Specifies how many ancillas to spread the measurements over
    """

    # Use the previous behavior if num_ancillas=None
    if not num_ancillas:
        if registers.AncillaRegister.size == 2:
            anQb_list = [registers.AncillaRegister[1]]*4
        elif registers.AncillaRegister.size >= 4:
            # I don't like this really, we don't use the flagged circuit anymore so it shouldn't get the 0 spot by default
            anQb_list = [registers.AncillaRegister[n] for n in [1, 2, 3, 4]]
        else:
            Warning("Ancilla reg too small (this should never happen)")
    else:
        anQb_list = [registers.AncillaRegister[n % num_ancillas]
                     for n in range(4)]

    if pipeline:
        stabilizer_list = [_pipeline_unflagged_stabilizer_XZZXI,
                           _pipeline_unflagged_stabilizer_IXZZX,
                           _pipeline_unflagged_stabilizer_XIXZZ,
                           _pipeline_unflagged_stabilizer_ZXIXZ]
    else:
        stabilizer_list = [_unflagged_stabilizer_XZZXI,
                           _unflagged_stabilizer_IXZZX,
                           _unflagged_stabilizer_XIXZZ,
                           _unflagged_stabilizer_ZXIXZ]

    # Create list of syndrome bits
    if isinstance(registers.SyndromeRegister, list):
        syn_reg = registers.SyndromeRegister[0][current_cycle][current_step]
        syn_bit_list = [syn_reg[n] for n in range(4)]
    else:
        syn_bit_list = [registers.SyndromeRegister[n] for n in range(4)]

    # Create circuit and run stabilizers
    circ = get_empty_stabilizer_circuit(registers)
    for i in range(4):
        circ += stabilizer_list[i](registers, anQb=anQb_list[i],
            syn_bit=syn_bit_list[i], reset=reset, pipeline=pipeline)
        if include_barriers:
            circ.barrier()
    # Add an extra measurement to the next syndrome register
    # TODO: Make this compatible with using more than 1 ancilla
    if recovery and not reset:
        if current_cycle < len(registers.SyndromeRegister[0])-1:
            circ.measure(anQb_list[-1],
                         registers.SyndromeRegister[0][current_cycle+1][current_step][4])
            if include_barriers:
                circ.barrier()

    # Recovery
    if recovery is True:
        circ.barrier()
        circ += unflagged_recovery(registers, reset, current_cycle)
        #if include_barriers:
        circ.barrier()
    return circ


def _pipeline_stabilizer_XZZXI(registers, anQb=None, syn_bit=None, reset=True):
    """Gives the circuit for running the regular XZZXI stabilizer in a pipelined
    scheme. Note that this assumes ancilla reset, as there currently is no
    lookup table to handle no-reset for this purpose.

    Args:
        registers (StabilizerRegister): Register object
        anQb (AncillaQubit, optional): Specifies the ancilla to use for the measurement. Defaults to None.
        reset (bool, optional): Whether to reset ancillas between measurements. Defaults to True.
        current_cycle (int, optional): [description]. Defaults to 0.
        current_step (int, optional): [description]. Defaults to 0.

    Returns:
        [type]: [description]
    """

    # Create a circuit
    qbReg = registers.QubitRegister
    anReg = registers.AncillaRegister
    clReg = registers.SyndromeRegister
    circ = get_empty_stabilizer_circuit(registers)
    # NOTE: Use the empty circuit below instead? It doesn't add the readout register
    # circ = QuantumCircuit(qbReg,anReg,clReg)
    if not anQb:
        if anReg.size == 2:
            anQb = anReg[1]
        else:
            anQb = anReg[1]

    # X
    circ.h(anQb)
    circ.h(qbReg[0])
    circ.cz(anQb, qbReg[0])
    circ.h(qbReg[0])

    # Z
    circ.cz(anQb, qbReg[1])

    # Z
    circ.cz(anQb, qbReg[2])

    # X
    circ.h(qbReg[3])
    circ.cz(anQb, qbReg[3])
    circ.h(qbReg[3])
    circ.h(anQb)

    # Measure and reset
    circ.swap(anReg[0], anReg[1])
    circ.measure(anReg[0], syn_bit)
    circ.reset(anReg[0])
    return circ


def _pipeline_stabilizer_IXZZX(registers, anQb=None, syn_bit=None, reset=True,
        pipeline=False):
    """Gives the circuit for running the regular IXZZX stabilizer in a pipelined
    scheme. Note that this assumes ancilla reset, as there currently is no
    lookup table to handle no-reset for this purpose.
    """

    # Create a circuit
    qbReg = registers.QubitRegister
    anReg = registers.AncillaRegister
    clReg = registers.SyndromeRegister
    circ = get_empty_stabilizer_circuit(registers)
    if not anQb:
        if anReg.size == 2:
            anQb = anReg[1]
        else:
            anQb = anReg[2]

    # X
    circ.h(anQb)
    circ.h(qbReg[1])
    circ.cz(anQb, qbReg[1])
    circ.h(qbReg[1])

    # Z
    circ.cz(anQb, qbReg[2])

    # Z
    circ.cz(anQb, qbReg[3])

    # X
    circ.h(qbReg[4])
    circ.cz(anQb, qbReg[4])
    circ.h(qbReg[4])
    circ.h(anQb)

    # Measure and reset
    circ.swap(anReg[0], anReg[1])
    circ.measure(anReg[0], syn_bit)
    circ.reset(anReg[0])
    return circ


def _pipeline_stabilizer_XIXZZ(registers, anQb=None, syn_bit=None, reset=True,
        pipeline=False):
    """Gives the circuit for running the regular XIXZZ stabilizer in a pipelined
    scheme. Note that this assumes ancilla reset, as there currently is no
    lookup table to handle no-reset for this purpose.
    """
    # Create a circuit
    qbReg = registers.QubitRegister
    anReg = registers.AncillaRegister
    clReg = registers.SyndromeRegister
    circ = get_empty_stabilizer_circuit(registers)

    if not anQb:
        if anReg.size == 2:
            anQb = anReg[1]
        else:
            anQb = anReg[3]

    # X
    circ.h(anQb)
    circ.h(qbReg[0])
    circ.cz(anQb, qbReg[0])
    circ.h(qbReg[0])

    # X
    circ.h(qbReg[2])
    circ.cz(anQb, qbReg[2])
    circ.h(qbReg[2])

    # Z
    circ.cz(anQb, qbReg[3])

    # Z
    circ.cz(anQb, qbReg[4])
    circ.h(anQb)

    # Measure and reset
    circ.swap(anReg[0], anReg[1])
    circ.measure(anReg[0], syn_bit)
    circ.reset(anReg[0])
    return circ


def _pipeline_stabilizer_ZXIXZ(registers, anQb=None, syn_bit=None, reset=True,
        pipeline=False):
     """Gives the circuit for running the regular ZXIXZ stabilizer in a pipelined
    scheme. Note that this assumes ancilla reset, as there currently is no
    lookup table to handle no-reset for this purpose. 

    As this is the final stabilizer in a cycle, this one does not include a
    swap before measurements, as it is essentially useless.
    """
    # Create a circuit
    qbReg = registers.QubitRegister
    anReg = registers.AncillaRegister
    clReg = registers.SyndromeRegister
    circ = get_empty_stabilizer_circuit(registers)

    if not anQb:
        if anReg.size == 2:
            anQb = anReg[1]
        else:
            anQb = anReg[4]

    # Z
    circ.h(anQb)
    circ.cz(anQb, qbReg[0])

    # X
    circ.h(qbReg[1])
    circ.cz(anQb, qbReg[1])
    circ.h(qbReg[1])

    # X
    circ.h(qbReg[3])
    circ.cz(anQb, qbReg[3])
    circ.h(qbReg[3])

    # Z
    circ.cz(anQb, qbReg[4])
    circ.h(anQb)

    # Measure and reset
    circ.measure(anReg[1], syn_bit)
    circ.reset(anReg[1])
    return circ



#%% Compare cycle times: Pipeline vs normal
n_cycles=3
recovery=True
flag=False
reset=True
# Registers
qb = QuantumRegister(5, 'code_qubit')
an = AncillaRegister(2, 'ancilla_qubit')
cr = get_classical_register(n_cycles, reset=False, recovery=False, flag=False)
readout = ClassicalRegister(5, 'readout')
registers = StabilizerRegisters(qb, an, cr, readout)

# Circuits
circ = encode_input_v2(registers)
circ.barrier()
circ.append(Snapshot('post_encoding', 'density_matrix', num_qubits=5), registers.QubitRegister)
#for current_cycle in range(n_cycles):
#    circ += mod_unflagged_stabilizer_cycle(registers, reset=reset, recovery=recovery,
#        current_cycle=current_cycle, pipeline=False, include_barriers=False)
#    circ.append(Snapshot('stabilizer_' + str(current_cycle),
#        snapshot_type='density_matrix', num_qubits=5), registers.QubitRegister)
circ += get_repeated_stabilization(registers, n_cycles=n_cycles,
    reset=reset, recovery=recovery, flag=flag, include_barriers=False, pipeline=True)
circ = shortest_transpile_from_distribution(circ, print_cost=False)
circ, time = add_idle_noise_to_circuit(circ, gate_times=WACQT_demonstrated_times,
    return_time=True, rename=False)

n_shots = 1024
noise_model = thermal_relaxation_model_V2(gate_times=WACQT_demonstrated_times)
results = execute(circ, Aer.get_backend('qasm_simulator'),
        noise_model=noise_model, shots=n_shots).result()

time_diff = 0
a = True
for key in time:
    current_time = time[key]
    if a:
        a = False
        previous_time = time[key]
        continue
    time_diff = current_time - previous_time
    print(time_diff)
    previous_time = time[key]
