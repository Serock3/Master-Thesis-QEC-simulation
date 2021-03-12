

# %% Import modules
from IPython.display import display
from qiskit.visualization import plot_histogram
from qiskit.quantum_info import state_fidelity
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


class StabilizerRegisters:

    def __init__(self,
                 qbReg=QuantumRegister(5, 'code_qubit'),
                 anReg=AncillaRegister(2, 'ancilla_qubit'),
                 clReg=ClassicalRegister(4, 'syndrome_bit'),
                 readout=ClassicalRegister(5, 'readout')
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
    circ += get_repeated_stabilization(registers, n_cycles=n_cycles,
                                      reset=reset, recovery=recovery, flag=flag)

    # Final readout
    circ.measure(qbReg, readout)
    circ.snapshot_statevector('post_measure')

    return circ


def get_repeated_stabilization(registers, n_cycles=1,
        reset=True, recovery=False, flag=True, snapshot_type='statevector'):
    """Generates a circuit for repeated stabilizers. Including recovery and 
    fault tolerant flagged circuits of selected.

    Args:
        registers (Register): Register object containing all registers
        n_cycles (int, optional): Number of stabilizer circuits. Defaults to 1.
        reset (bool, optional): Whether or not to reset ancillas. Defaults to True.
        recovery (bool, optional): Whether or not to apply recovery operations. Defaults to False.
        flag (bool, optional): Whether or not to use the fault-tolerant flagged circuit. Defaults to True.
        snapshot_type (str, optional): Type of snapshot (None,'statevector' or 'density_matrix'). Defaults to 'statevector'.

    Returns:
        QuantumCircuit: The resulting circuit
    """

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


def unpack_stabilizer_registers(registers):
    """Unpacks all registers packed as a list according to customs used
    in all stabilizer functions (stabilizer.py). Kind of obsolete
    """
    qbReg, anReg, clReg, readout = registers
    return qbReg, anReg, clReg, readout


def encode_input(registers):
    """Encode the input into logical 0 and 1 for the [[5,1,3]] code. This
    assumes that the 0:th qubit is the original state |psi> = a|0> + b|1>
    """

    # Unpack registers
    qbReg = registers.QubitRegister
    circ = get_empty_stabilizer_circuit(registers)

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


def encode_input_v2(registers):
    """Encode the input into logical 0 and 1 for the [[5,1,3]] code. This
    assumes that the 0:th qubit is the original state |psi> = a|0> + b|1>.

    Alternate version Basudha found on stackoverflow.
    """

    qbReg = registers.QubitRegister
    #circ = get_empty_stabilizer_circuit(registers)
    circ = QuantumCircuit(qbReg)

    circ.z(qbReg[0])
    circ.h(qbReg[1])
    circ.h(qbReg[2])
    circ.h(qbReg[3])
    circ.h(qbReg[4])

    circ.h(qbReg[0])
    circ.cz(qbReg[0], qbReg[1])
    circ.cz(qbReg[0], qbReg[2])
    circ.cz(qbReg[0], qbReg[3])
    circ.cz(qbReg[0], qbReg[4])
    circ.h(qbReg[0])

    circ.cz(qbReg[0], qbReg[1])
    circ.cz(qbReg[2], qbReg[3])
    circ.cz(qbReg[1], qbReg[2])
    circ.cz(qbReg[3], qbReg[4])
    circ.cz(qbReg[0], qbReg[4])

    return circ


def get_classical_register(n_cycles, reset=True, recovery=False, flag=True):
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
            [ClassicalRegister(4, 'syndrome_cycle_' + str(i) + '_step_' + str(j))
             for j in range(4)] for i in range(n_cycles)]

        # List of registers for each step in the flagged stabilizer cycle
        flag_register = [
            [ClassicalRegister(1, 'flag_cycle_' + str(i) + '_step_' + str(j))
             for j in range(4)] for i in range(n_cycles)]

        # List of registers for the single stabilizer run with flag
        ancilla_msmnt_register = [
            [ClassicalRegister(1, 'ancilla_cycle_' + str(i) + '_step_' + str(j))
             for j in range(4)] for i in range(n_cycles)]

        return [syndrome_register, flag_register, ancilla_msmnt_register]

    # TODO: Make this functional with flags? If necessary
    elif recovery and not reset:
        # A register of four bits per cycle
        syndrome_register = [
            [ClassicalRegister(5, 'syndrome_cycle_' + str(i) + '_step_' + str(j))
             for j in range(1)] for i in range(n_cycles)]
        return [syndrome_register]

    else:
        # A register of four bits per cycle
        syndrome_register = [
            [ClassicalRegister(4, 'syndrome_cycle_' + str(i) + '_step_' + str(j))
             for j in range(1)] for i in range(n_cycles)]
        return [syndrome_register]


# %% All unflagged stabilizers
def unflagged_stabilizer_cycle(registers, reset=True, recovery=False,
                               current_cycle=0, current_step=0, num_ancillas=None):
    """Run all four stabilizers without flags, as well as an optional
    recovery. The input current_step is only relevant for flagged cycles, and
    should be set to 0 otherwise.

    NOTE: Maybe use a list if ancilla idices instead? E.g. ancillas = [1,2,1,2]
    Args:
        num_ancillas: Specifies how many acillas to spread the measurements over
    """

    if not num_ancillas:
        if registers.AncillaRegister.size == 2:
            anQb_list = [registers.AncillaRegister[1]]*4
        elif registers.AncillaRegister.size >= 4:
            # I don't like this really, we don't use the flagged ciruit anymore so it shouldn't get the 0 stop by default
            anQb_list = [registers.AncillaRegister[n] for n in [1,2,3,4]]
        else:
            Warning("Ancilla reg too small (this should never happen)")
    else:
        anQb_list = [registers.AncillaRegister[n%num_ancillas] for n in range(4)]

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

    # Run all four stabilizers
    for i in range(4):
        circ += stabilizer_list[i](registers, anQb=anQb_list[i], 
            syn_bit=syn_bit_list[i], reset=reset)

    # Add an extra measurement to next syndrome register
    # TODO: Make this compatible with using more than 1 ancilla
    if recovery and not reset:
        if current_cycle < len(registers.SyndromeRegister[0])-1:
            circ.measure(anQb_list[-1], 
                registers.SyndromeRegister[0][current_cycle+1][current_step][4])

    # Recovery
    if recovery is True:
        circ += unflagged_recovery(registers, reset, current_cycle)

    return circ


def _unflagged_stabilizer_XZZXI(registers, anQb=None, syn_bit=None, reset=True):
    """Gives the circuit for running the regular XZZXI stabilizer without flag.
    The current_step input should be set to zero unless running flagged cycles.

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

    # Measure
    circ.measure(anQb, syn_bit)

    # Reset
    if reset:
        circ.reset(anQb)

    return circ


def _unflagged_stabilizer_IXZZX(registers, anQb=None, syn_bit=None, reset=True):
    """Gives the circuit for running the regular IXZZX stabilizer without flag.
    The current_step input should be set to zero unless running flagged cycles.
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

    # Measure
    circ.measure(anQb, syn_bit)

    # Reset
    if reset:
        circ.reset(anQb)

    return circ


def _unflagged_stabilizer_XIXZZ(registers, anQb=None, syn_bit=None, reset=True):
    """Gives the circuit for running the regular XIXZZ stabilizer without flag.
    The current_step input should be set to zero unless running flagged cycles.
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

    # Measure
    circ.measure(anQb, syn_bit)

    # Reset
    if reset:
        circ.reset(anQb)

    return circ


def _unflagged_stabilizer_ZXIXZ(registers, anQb=None, syn_bit=None, reset=True):
    """Gives the circuit for running the regular ZXIXZ stabilizer without flag.
    The current_step input should be set to zero unless running flagged cycles.
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

    # Measure
    circ.measure(anQb, syn_bit)

    # Reset
    if reset:
        circ.reset(anQb)

    return circ


# %% All recoveries
def unflagged_recovery(registers, reset=True, current_cycle=0, current_step=0):
    """Lookup table for recovery from a
    single qubit error on code qubits"""

    # Create a circuit
    qbReg = registers.QubitRegister
    clReg = registers.SyndromeRegister
    circ = get_empty_stabilizer_circuit(registers)

    # Unpack registers
    if isinstance(clReg, list):
        syndrome_reg = clReg[0][current_cycle][current_step]
    else:
        syndrome_reg = clReg

    # If the ancilla is reset to |0> between measurements
    if reset:
        circ.x(qbReg[1]).c_if(syndrome_reg, 1)
        circ.z(qbReg[4]).c_if(syndrome_reg, 2)
        circ.x(qbReg[2]).c_if(syndrome_reg, 3)
        circ.z(qbReg[2]).c_if(syndrome_reg, 4)
        circ.z(qbReg[0]).c_if(syndrome_reg, 5)
        circ.x(qbReg[3]).c_if(syndrome_reg, 6)
        circ.x(qbReg[2]).c_if(syndrome_reg, 7)
        circ.z(qbReg[2]).c_if(syndrome_reg, 7)
        circ.x(qbReg[0]).c_if(syndrome_reg, 8)
        circ.z(qbReg[3]).c_if(syndrome_reg, 9)
        circ.z(qbReg[1]).c_if(syndrome_reg, 10)
        circ.x(qbReg[1]).c_if(syndrome_reg, 11)
        circ.z(qbReg[1]).c_if(syndrome_reg, 11)
        circ.x(qbReg[4]).c_if(syndrome_reg, 12)
        circ.x(qbReg[0]).c_if(syndrome_reg, 13)
        circ.z(qbReg[0]).c_if(syndrome_reg, 13)
        circ.x(qbReg[4]).c_if(syndrome_reg, 14)
        circ.z(qbReg[4]).c_if(syndrome_reg, 14)
        circ.x(qbReg[3]).c_if(syndrome_reg, 15)
        circ.z(qbReg[3]).c_if(syndrome_reg, 15)

    # If the ancilla is NOT reset between measurements
    else:
        circ.x(qbReg[2]).c_if(syndrome_reg, 1)
        circ.x(qbReg[3]).c_if(syndrome_reg, 2)
        circ.z(qbReg[0]).c_if(syndrome_reg, 3)
        circ.x(qbReg[4]).c_if(syndrome_reg, 4)
        circ.z(qbReg[3]).c_if(syndrome_reg, 5)
        circ.x(qbReg[3]).c_if(syndrome_reg, 5)
        circ.z(qbReg[1]).c_if(syndrome_reg, 6)
        circ.z(qbReg[3]).c_if(syndrome_reg, 7)
        circ.x(qbReg[0]).c_if(syndrome_reg, 8)
        circ.z(qbReg[1]).c_if(syndrome_reg, 9)
        circ.x(qbReg[1]).c_if(syndrome_reg, 9)
        circ.z(qbReg[4]).c_if(syndrome_reg, 10)
        circ.x(qbReg[4]).c_if(syndrome_reg, 10)
        circ.z(qbReg[0]).c_if(syndrome_reg, 11)
        circ.x(qbReg[0]).c_if(syndrome_reg, 11)
        circ.z(qbReg[2]).c_if(syndrome_reg, 12)
        circ.z(qbReg[2]).c_if(syndrome_reg, 13)
        circ.x(qbReg[2]).c_if(syndrome_reg, 13)
        circ.z(qbReg[4]).c_if(syndrome_reg, 14)
        circ.x(qbReg[1]).c_if(syndrome_reg, 15)

        circ.x(qbReg[2]).c_if(syndrome_reg, 16+15-1)
        circ.x(qbReg[3]).c_if(syndrome_reg, 16+15-2)
        circ.z(qbReg[0]).c_if(syndrome_reg, 16+15-3)
        circ.x(qbReg[4]).c_if(syndrome_reg, 16+15-4)
        circ.z(qbReg[3]).c_if(syndrome_reg, 16+15-5)
        circ.x(qbReg[3]).c_if(syndrome_reg, 16+15-5)
        circ.z(qbReg[1]).c_if(syndrome_reg, 16+15-6)
        circ.z(qbReg[3]).c_if(syndrome_reg, 16+15-7)
        circ.x(qbReg[0]).c_if(syndrome_reg, 16+15-8)
        circ.z(qbReg[1]).c_if(syndrome_reg, 16+15-9)
        circ.x(qbReg[1]).c_if(syndrome_reg, 16+15-9)
        circ.z(qbReg[4]).c_if(syndrome_reg, 16+15-10)
        circ.x(qbReg[4]).c_if(syndrome_reg, 16+15-10)
        circ.z(qbReg[0]).c_if(syndrome_reg, 16+15-11)
        circ.x(qbReg[0]).c_if(syndrome_reg, 16+15-11)
        circ.z(qbReg[2]).c_if(syndrome_reg, 16+15-12)
        circ.z(qbReg[2]).c_if(syndrome_reg, 16+15-13)
        circ.x(qbReg[2]).c_if(syndrome_reg, 16+15-13)
        circ.z(qbReg[4]).c_if(syndrome_reg, 16+15-14)
        circ.x(qbReg[1]).c_if(syndrome_reg, 16+15-15)

    return circ



# %% Internal testing of functions above

if __name__ == "__main__":
    # The settings for our circuit
    n_cycles = 2
    reset = False
    recovery = True
    flag = False

    # Define our registers (Maybe to be written as function?)
    qb = QuantumRegister(5, 'code_qubit')
    an = AncillaRegister(2, 'ancilla_qubit')
    cr = get_classical_register(n_cycles, reset, recovery, flag) # Advanced list of registers
    #cr = ClassicalRegister(4, 'syndrome_bit')
    readout = ClassicalRegister(5, 'readout')

    registers = StabilizerRegisters(qb, an, cr, readout)
    circ = get_empty_stabilizer_circuit(registers)

    # Get the complete circuit
    circ += get_full_stabilizer_circuit(registers,
                                        n_cycles=n_cycles,
                                        reset=reset,
                                        recovery=recovery,
                                        flag=flag,
                                        )

    #display(circ.draw())

    # Run it
    n_shots = 1024
    results = execute(
        circ,
        Aer.get_backend('qasm_simulator'),
        noise_model=None,
        shots=n_shots
    ).result()

    display(plot_histogram(results.get_counts()))

    # Analyze results
    logical = logical_states()
    fid = 0
    for i in range(10):
        sv_post_encoding = results.data(
        )['snapshots']['statevector']['testing'][i]

        log0 = logical[0][np.arange(128, step=4)]
        sv_test = sv_post_encoding[0:32]
        fid += state_fidelity(log0, sv_test)

    print('Average fidelity across 10 shots:')
    print(fid/10)

    # Plot results
    counts = results.get_counts()
    plot_histogram(counts)

# %% Function used for internal testing
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

    logical_1 = np.kron(logical_1, an0)
    logical_0 = np.kron(logical_0, an0)

    return [logical_0, logical_1]
