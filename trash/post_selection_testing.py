
# %% Import modules
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
from simulator_program.custom_noise_models import thermal_relaxation_model_V2
from simulator_program.custom_transpiler import *
from simulator_program.stabilizers import *


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
    circ = get_empty_stabilizer_circuit(registers)

    # Encode the state
    circ += encode_input_v2(registers)
    circ.snapshot('post_encoding', 'statevector')

    # Stabilizer
    for current_cycle in range(n_cycles):
        if flag is True:
            circ += flagged_stabilizer_cycle(registers,
                                             reset=reset,
                                             recovery=recovery,
                                             current_cycle=current_cycle,
                                             )
        else:
            circ += get_stabilizer_cycle(registers,
                                               reset=reset,
                                               recovery=recovery,
                                               current_cycle=current_cycle
                                               )
        circ.snapshot('stabilizer_' + str(current_cycle), 'statevector')

    # Final readout
    circ.measure(qbReg, readout)
    circ.snapshot_statevector('post_measure')

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
    circ = get_empty_stabilizer_circuit(registers)

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

    else:
        # A register of four bits per cycle
        syndrome_register = [
            [ClassicalRegister(4, 'syndrome_cycle_' + str(i) + '_step_' + str(j))
             for j in range(1)] for i in range(n_cycles)]
        return [syndrome_register]


# %% All flagged stabilizers
def flagged_stabilizer_cycle(registers, reset=True, recovery=True,
                             current_cycle=0):
    """Runs one cycle of the [[5,1,3]] code with two ancillas as described in
    the article by Chao & Reichardt (2017).
    This includes the (optional) recovery from any detected errors.

    Currently, it requires reset=True to appropriately
    correct errors, and cannot perform recovery with the advanced registers
    """

    # Create a circuit
    circ = get_empty_stabilizer_circuit(registers)

    # === Step 1: XZZXI ===
    circ += _flagged_stabilizer_XZZXI(registers, reset, current_cycle)
    circ += get_stabilizer_cycle(registers, reset, recovery=False,
                                       current_cycle=current_cycle, current_step=0)
    if recovery:
        circ += full_recovery_XZZXI(registers, reset, current_cycle, 0)

    # === Step 2: IXZZX ===
    circ += _flagged_stabilizer_IXZZX(registers, reset, current_cycle)
    circ += get_stabilizer_cycle(registers, reset, recovery=False,
                                       current_cycle=current_cycle, current_step=1)
    if recovery:
        circ += full_recovery_IXZZX(registers, reset, current_cycle, 1)

    # === Step 3: XIXZZ ===
    circ += _flagged_stabilizer_XIXZZ(registers, reset, current_cycle)
    circ += get_stabilizer_cycle(registers, reset, recovery=False,
                                       current_cycle=current_cycle, current_step=2)
    if recovery:
        circ += full_recovery_XIXZZ(registers, reset, current_cycle, 2)

    # === Step 4: ZXIXZ ===
    circ += _flagged_stabilizer_ZXIXZ(registers, reset, current_cycle)
    circ += get_stabilizer_cycle(registers, reset, recovery=False,
                                       current_cycle=current_cycle, current_step=3)
    if recovery:
        circ += full_recovery_ZXIXZ(registers, reset, current_cycle, 3)

    return circ


def _flagged_stabilizer_XZZXI(registers, reset=True, current_cycle=0):
    """Gives the circuit for running the XZZXI stabilizer with a flag ancilla,
    connected by a cz gate after the first and before the last operation.
    """

    # Create a circuit
    qbReg = registers.QubitRegister
    anReg = registers.AncillaRegister
    clReg = registers.SyndromeRegister
    circ = get_empty_stabilizer_circuit(registers)

    # X
    circ.h(qbReg[0])
    circ.h(anReg[1])
    circ.cz(anReg[1], qbReg[0])
    circ.h(qbReg[0])

    # Flag
    circ.h(anReg[0])  # Initialize in +
    circ.cz(anReg[0], anReg[1])  # Flag CX

    # Z
    circ.cz(anReg[1], qbReg[1])

    # Z
    circ.cz(anReg[1], qbReg[2])

    # Flag
    circ.cz(anReg[0], anReg[1])

    # X
    circ.h(qbReg[3])
    circ.cz(anReg[1], qbReg[3])
    circ.h(anReg[1])
    circ.h(qbReg[3])

    # Measure
    if isinstance(clReg, list):
        flag_register = clReg[1]
        ancilla_msmnt_register = clReg[2]

        circ.measure(anReg[1], ancilla_msmnt_register[current_cycle][0])
        circ.h(anReg[0])
        circ.measure(anReg[0], flag_register[current_cycle][0])

    else:
        circ.measure(anReg[1], clReg[0])
        circ.h(anReg[0])
        circ.measure(anReg[0], clReg[4])

    # Reset
    if reset:
        circ.reset(anReg[1])
        circ.reset(anReg[0])

    return circ


def _flagged_stabilizer_IXZZX(registers, reset=True, current_cycle=0):
    """Gives the circuit for running the IXZZX stabilizer with a flag ancilla,
    connected by a cz gate after the first and before the last operation.
    """

    # Create a circuit
    qbReg = registers.QubitRegister
    anReg = registers.AncillaRegister
    clReg = registers.SyndromeRegister
    circ = get_empty_stabilizer_circuit(registers)

    # X
    circ.h(qbReg[1])
    circ.h(anReg[1])
    circ.cz(anReg[1], qbReg[1])
    circ.h(qbReg[1])

    # Flag
    circ.h(anReg[0])  # Initialize in +
    circ.cz(anReg[0], anReg[1])  # Flag CX

    # Z
    circ.cz(anReg[1], qbReg[2])

    # Z
    circ.cz(anReg[1], qbReg[3])

    # Flag
    circ.cz(anReg[0], anReg[1])

    # X
    circ.h(qbReg[4])
    circ.cz(anReg[1], qbReg[4])
    circ.h(anReg[1])
    circ.h(qbReg[4])

    # Measure
    if isinstance(clReg, list):
        flag_register = clReg[1]
        ancilla_msmnt_register = clReg[2]

        circ.measure(anReg[1], ancilla_msmnt_register[current_cycle][1])
        circ.h(anReg[0])
        circ.measure(anReg[0], flag_register[current_cycle][1])
    else:
        circ.measure(anReg[1], clReg[1])
        circ.h(anReg[0])
        circ.measure(anReg[0], clReg[4])

    # Reset
    if reset:
        circ.reset(anReg[1])
        circ.reset(anReg[0])

    return circ


def _flagged_stabilizer_XIXZZ(registers, reset=True, current_cycle=0):
    """Gives the circuit for running the XIXZZ stabilizer with a flag ancilla,
    connected by a cz gate after the first and before the last operation.
    """

    # Create a circuit
    qbReg = registers.QubitRegister
    anReg = registers.AncillaRegister
    clReg = registers.SyndromeRegister
    circ = get_empty_stabilizer_circuit(registers)

    # X
    circ.h(qbReg[0])
    circ.h(anReg[1])
    circ.cz(anReg[1], qbReg[0])
    circ.h(qbReg[0])

    # Flag
    circ.h(anReg[0])  # Initialize in +
    circ.cz(anReg[0], anReg[1])  # Flag CX

    # X
    circ.h(qbReg[2])
    circ.cz(anReg[1], qbReg[2])
    circ.h(qbReg[2])

    # Z
    circ.cz(anReg[1], qbReg[3])

    # Flag
    circ.cz(anReg[0], anReg[1])

    # Z
    circ.cz(anReg[1], qbReg[4])
    circ.h(anReg[1])

    # Measure
    if isinstance(clReg, list):
        flag_register = clReg[1]
        ancilla_msmnt_register = clReg[2]

        circ.measure(anReg[1], ancilla_msmnt_register[current_cycle][2])
        circ.h(anReg[0])
        circ.measure(anReg[0], flag_register[current_cycle][2])
    else:
        circ.measure(anReg[1], clReg[2])
        circ.h(anReg[0])
        circ.measure(anReg[0], clReg[4])

    # Reset
    if reset:
        circ.reset(anReg[1])
        circ.reset(anReg[0])

    return circ


def _flagged_stabilizer_ZXIXZ(registers, reset=True, current_cycle=0):
    """Gives the circuit for running the ZXIXZ stabilizer with a flag ancilla,
    connected by a cz gate after the first and before the last operation.
    """

    # Create a circuit
    qbReg = registers.QubitRegister
    anReg = registers.AncillaRegister
    clReg = registers.SyndromeRegister
    circ = get_empty_stabilizer_circuit(registers)

    # Z
    circ.h(anReg[1])
    circ.cz(anReg[1], qbReg[0])

    # Flag
    circ.h(anReg[0])  # Initialize in +
    circ.cz(anReg[0], anReg[1])  # Flag CX

    # X
    circ.h(qbReg[1])
    circ.cz(anReg[1], qbReg[1])
    circ.h(qbReg[1])

    # X
    circ.h(qbReg[3])
    circ.cz(anReg[1], qbReg[3])
    circ.h(qbReg[3])

    # Flag
    circ.cz(anReg[0], anReg[1])

    # Z
    circ.cz(anReg[1], qbReg[4])
    circ.h(anReg[1])

    # Measure
    if isinstance(clReg, list):
        flag_register = clReg[1]
        ancilla_msmnt_register = clReg[2]

        circ.measure(anReg[1], ancilla_msmnt_register[current_cycle][3])
        circ.h(anReg[0])
        circ.measure(anReg[0], flag_register[current_cycle][3])
    else:
        circ.measure(anReg[1], clReg[3])
        circ.h(anReg[0])
        circ.measure(anReg[0], clReg[4])

    # Reset
    if reset:
        circ.reset(anReg[1])
        circ.reset(anReg[0])

    return circ


# %% All unflagged stabilizers
def get_stabilizer_cycle(registers, reset=True, recovery=False,
                               current_cycle=0, current_step=0):
    """Run all four stabilizers without flags, as well as an optional
    recovery. The input current_step is only relevant for flagged cycles, and
    should be set to 0 otherwise.
    """
    # Create circuit and run stabilizers
    circ = get_empty_stabilizer_circuit(registers)
    # circ.x(qb[3])
    circ += _get_stabilizer_XZZXI(registers, reset=reset,
                                        current_cycle=current_cycle, current_step=current_step)
    circ += _get_stabilizer_IXZZX(registers, reset=reset,
                                        current_cycle=current_cycle, current_step=current_step)
    circ += _get_stabilizer_XIXZZ(registers, reset=reset,
                                        current_cycle=current_cycle, current_step=current_step)
    circ += _get_stabilizer_ZXIXZ(registers, reset=reset,
                                        current_cycle=current_cycle, current_step=current_step)

    # Recovery
    if recovery is True:
        circ += get_recovery(registers, reset, current_cycle)

    return circ


def _get_stabilizer_XZZXI(registers, reset=True,
                                current_cycle=0, current_step=0):
    """Gives the circuit for running the regular XZZXI stabilizer without flag.
    The current_step input should be set to zero unless running flagged cycles.
    """

    # Create a circuit
    qbReg = registers.QubitRegister
    anReg = registers.AncillaRegister
    clReg = registers.SyndromeRegister
    circ = get_empty_stabilizer_circuit(registers)
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
    if isinstance(clReg, list):
        syndrome_reg = clReg[0]
        circ.measure(anQb, syndrome_reg[current_cycle][current_step][0])
    else:
        circ.measure(anQb, clReg[0])

    # Reset
    if reset:
        circ.reset(anQb)

    return circ


def _get_stabilizer_IXZZX(registers, reset=True,
                                current_cycle=0, current_step=0):
    """Gives the circuit for running the regular IXZZX stabilizer without flag.
    The current_step input should be set to zero unless running flagged cycles.
    """

    # Create a circuit
    qbReg = registers.QubitRegister
    anReg = registers.AncillaRegister
    clReg = registers.SyndromeRegister
    circ = get_empty_stabilizer_circuit(registers)
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
    if isinstance(clReg, list):
        syndrome_reg = clReg[0]
        circ.measure(anQb, syndrome_reg[current_cycle][current_step][1])
    else:
        circ.measure(anQb, clReg[1])

    # Reset
    if reset:
        circ.reset(anQb)

    return circ


def _get_stabilizer_XIXZZ(registers, reset=True,
                                current_cycle=0, current_step=0):
    """Gives the circuit for running the regular XIXZZ stabilizer without flag.
    The current_step input should be set to zero unless running flagged cycles.
    """
    # Create a circuit
    qbReg = registers.QubitRegister
    anReg = registers.AncillaRegister
    clReg = registers.SyndromeRegister
    circ = get_empty_stabilizer_circuit(registers)
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
    if isinstance(clReg, list):
        syndrome_reg = clReg[0]
        circ.measure(anQb, syndrome_reg[current_cycle][current_step][2])
    else:
        circ.measure(anQb, clReg[2])

    # Reset
    if reset:
        circ.reset(anQb)
    return circ


def _get_stabilizer_ZXIXZ(registers, reset=True,
                                current_cycle=0, current_step=0):
    """Gives the circuit for running the regular ZXIXZ stabilizer without flag.
    The current_step input should be set to zero unless running flagged cycles.
    """
    # Create a circuit
    qbReg = registers.QubitRegister
    anReg = registers.AncillaRegister
    clReg = registers.SyndromeRegister
    circ = get_empty_stabilizer_circuit(registers)
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
    if isinstance(clReg, list):
        syndrome_reg = clReg[0]
        circ.measure(anQb, syndrome_reg[current_cycle][current_step][3])
    else:
        circ.measure(anQb, clReg[3])

    # Reset
    if reset:
        circ.reset(anQb)

    return circ

# %% =================  Testing noise model + stabilizer ======================
# DEFINE AND TRANSPILE THE CIRCUIT


# Define our registers
qb = QuantumRegister(5, 'code_qubit')
an = AncillaRegister(2, 'ancilla_qubit')
cr = ClassicalRegister(4, 'syndrome_bit')  # The typical register
# cr = get_classical_register(n_cycles, flag) # Advanced list of registers
readout = ClassicalRegister(5, 'readout')

registers = StabilizerRegisters(qb, an, cr, readout)
circ = get_empty_stabilizer_circuit(registers)

# Settings for circuit
n_cycles = 1
reset = False
flag = False
recovery = False

# Get the circuit
circ.x(qb[0])
circ += get_full_stabilizer_circuit(registers,
                                    n_cycles=n_cycles,
                                    reset=reset,
                                    recovery=recovery,
                                    flag=flag,
                                    )

results = execute(
    circ,
    Aer.get_backend('qasm_simulator'),
    noise_model=thermal_relaxation_model_V2(),
    memory=True,
    shots=10,
).result()

mem = results.get_memory()
# print(mem)

# %%

#mem[ int(mem[0][6:10]) == 0 ]
#test = np.array( [int(item[6:10]) for item in mem] ) == 0
# print(test)


def select_no_errors(mem):
    return np.array([int(item[6:10]) for item in mem]) == 0


selection = select_no_errors(mem)
#np.array( results.data()['snapshots']['statevector']['post_measure'])[selection].shape


# %% Testing with plot_fidelity

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
cr = ClassicalRegister(5, 'syndrome_bit')  # The typical register
# cr = get_classical_register(n_cycles, flag) # Advanced list of registers
readout = ClassicalRegister(5, 'readout')

registers = StabilizerRegisters(qb, an, cr, readout)
circ = get_empty_stabilizer_circuit(registers)

# Settings for circuit
n_cycles = 1
reset = False
flag = False
recovery = False

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
#optimization_level = 1
circ_t0 = shortest_transpile_from_distribution(
    circ,
    print_cost=False,
    repeats=repeats,
    routing_method=routing_method,
    initial_layout=initial_layout,
    layout_method=layout_method,
    translation_method=translation_method,
    optimization_level=0,
    **WACQT_device_properties
)
circ_t1 = shortest_transpile_from_distribution(
    circ,
    print_cost=False,
    repeats=repeats,
    routing_method=routing_method,
    initial_layout=initial_layout,
    layout_method=layout_method,
    translation_method=translation_method,
    optimization_level=1,
    **WACQT_device_properties
)
circ_t2 = shortest_transpile_from_distribution(
    circ,
    print_cost=False,
    repeats=repeats,
    routing_method=routing_method,
    initial_layout=initial_layout,
    layout_method=layout_method,
    translation_method=translation_method,
    optimization_level=2,
    **WACQT_device_properties
)

# Error free run for fidelity measurements
results_t0 = execute(
    circ_t0,
    Aer.get_backend('qasm_simulator'),
    noise_model=None,
    shots=1,
).result()
logical_t0 = results_t0.data()['snapshots']['statevector']['stabilizer_0'][0]
results_t1 = execute(
    circ_t1,
    Aer.get_backend('qasm_simulator'),
    noise_model=None,
    shots=1,
).result()
logical_t1 = results_t1.data()['snapshots']['statevector']['stabilizer_0'][0]
results_t2 = execute(
    circ_t2,
    Aer.get_backend('qasm_simulator'),
    noise_model=None,
    shots=1,
).result()
logical_t2 = results_t2.data()['snapshots']['statevector']['stabilizer_0'][0]

# %% Analyze results (NOTE: Might take a long time due to n_shots)
T2_list = np.arange(40, 81, 10)*1e3  # 40-80 mus
# t_cz_list = np.arange(100,301, 10) # 100-300 ns
logical = logical_states()
log0 = DensityMatrix(logical[1])
n_shots = 1024

# Arrays for storing all the stuff
T2_data = np.zeros([len(T2_list), n_shots])
T2_data_t0 = np.zeros([len(T2_list), n_shots])
T2_data_t1 = np.zeros([len(T2_list), n_shots])
T2_data_t2 = np.zeros([len(T2_list), n_shots])
#t_cz_data = np.zeros([len(t_cz_list), n_shots])
#t_cz_data_t0 = np.zeros([len(t_cz_list), n_shots])
#t_cz_data_t1 = np.zeros([len(t_cz_list), n_shots])
#t_cz_data_t2 = np.zeros([len(t_cz_list), n_shots])

# No transpilation, vary T2, fixed t_cz
fid_T2 = np.zeros(len(T2_list))
for i in range(len(fid_T2)):
    T2 = T2_list[i]

    # Run the circuit
    results = execute(
        circ,
        Aer.get_backend('qasm_simulator'),
        noise_model=thermal_relaxation_model_V2(T2=T2, t_cz=200),
        memory=True,
        shots=n_shots
    ).result()
    counts = results.get_counts()

    # Post-selection
    select_indices = select_no_errors(results.get_memory())
    select_count = np.sum(select_indices)
    post_selection = np.array(
        results.data()['snapshots']['statevector']['stabilizer_0'])[select_indices]
    print(select_count)
    # Analyze results
    for j in range(select_count):
        statevector = post_selection[j]
        partial_sv = partial_trace(statevector, [5, 6])
        T2_data[i][j] = state_fidelity(log0, partial_sv)
fid_T2 = np.sum(T2_data, axis=1)/select_count
print('Finished varying T2 without transpilation')


# %%
fig, axs = plt.subplots(2, figsize=(14, 10))
ax1 = axs[0]
ax2 = axs[1]

# Vary T2
ax1.plot(T2_list*1e-3, fid_T2, 'o-', label='No transpilation')

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

# %% Density Matrix testing


# Define our registers
qb = QuantumRegister(5, 'code_qubit')
an = AncillaRegister(2, 'ancilla_qubit')
cr = ClassicalRegister(4, 'syndrome_bit')  # The typical register
readout = ClassicalRegister(5, 'readout')

registers = StabilizerRegisters(qb, an, cr, readout)
circ = get_empty_stabilizer_circuit(registers)

# Settings for circuit
n_cycles = 1
reset = False
flag = False
recovery = False

circ.x(qb[0])
circ += encode_input_v2(registers)
circ += get_stabilizer_cycle(registers,
                                   reset=reset,
                                   recovery=recovery
                                   )
circ.barrier()
circ.append(Snapshot('booper', 'density_matrix', num_qubits=5), qb)
circ.measure(qb, readout)

results = execute(
    circ,
    Aer.get_backend('qasm_simulator'),
    noise_model=thermal_relaxation_model_V2(),
    memory=True,
    shots=100,
).result()

correct_state = results.data()[
    'snapshots']['density_matrix']['booper'][0]['value']
print(correct_state.shape)
logical = logical_states()
log1 = DensityMatrix(logical[1])
print(state_fidelity(correct_state, log1))
print(make_result_dict(results))
# %% Density matrix post select testing


def reformat_density_snapshot(results) -> dict:
    """Reformats the snapshot data of the results object to be a 
    dictionary with the measurement results as keys
    """
    snap_dict = {}
    for snapshot_name in results.data()['snapshots']['density_matrix']:
        res_dict = {}
        for item in results.data()[
                'snapshots']['density_matrix'][snapshot_name]:
            res_dict[item['memory']] = item['value']
        snap_dict[snapshot_name] = res_dict
    return snap_dict

from qiskit.aqua.utils import get_subsystems_counts


def get_post_select_fraction_for_density_matrix(results,shots):
    syndrome_reg_counts = get_subsystems_counts(results.get_counts())[1]
    for key in syndrome_reg_counts:
        if int(key) == 0:
            count_trivial_syndrome = syndrome_reg_counts[key]
    return count_trivial_syndrome/shots

# %%


def select_no_errors(mem):
    return np.array([int(item[6:10]) for item in mem]) == 0


def get_fidelity_data(circ, correct_state, param_list, n_shots=2048):
    '''Inputs:
    circ: The circuit to be tested
    correct_state: The correct state for comparison
    param_list: The error model parameters, currently only [T2, t_cz]
    n_shots: Number of shots to average over
    '''
    T2, t_cz = param_list

    # Run the circuit
    results = execute(
        circ,
        Aer.get_backend('qasm_simulator'),
        noise_model=thermal_relaxation_model_V2(T2=T2, t_cz=t_cz),
        memory=True,
        shots=n_shots
    ).result()

    # Post-selection
    select_indices = select_no_errors(results.get_memory())
    select_count = np.sum(select_indices)
    post_selection = np.array(
        results.data()['snapshots']['density_matrix']['booper']
    )[select_indices]

    # Analyze results
    data = np.zeros(n_shots)
    for j in range(select_count):
        statevector = results.data()[
            'snapshots']['statevector']['stabilizer_0'][j]
        data[j] = state_fidelity(statevector, correct_state)
    fid = np.sum(data)/select_count
    print('Boop')

# %%

# %%
