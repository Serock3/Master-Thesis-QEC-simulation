""" Functions for running the [[7,1,3]] steane code, mainly on a triangular
10qb chip. Most functions here have an equivalent one in stabilizers.py, for
the [[5,1,3]] code, which has better documentation.
NOTE: The code in this file is not fully maintained. It should be functional but
lacks many capabilities that stabilizers.py or stabililzers_422.py may have.
"""

# TODO: Add recovery
# TODO: Remove dependence of having 3 ancillas
# TODO: Implement an adapted get_classical_register, tailored for [[7,1,3]]
# %% Import modules
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
from custom_transpiler import *
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

def get_full_stabilizer_circuit_steane(registers, n_cycles=1,
        reset=True, recovery=False, snapshot_type='statevector'):
    """Returns the circuit for a full repeating stabilizer circuit of the
    [[7,1,3]] (Steane) code. This includes encoding, n_cycles of repeated
    stabilizers (with optional recovery) and final measurement.
    """

    # Unpack registers
    qbReg = registers.QubitRegister
    anReg = registers.AncillaRegister
    clReg = registers.SyndromeRegister
    readout = registers.ReadoutRegister

    # Encode the state
    circ = encode_input_steane(registers)

    # Stabilizer
    circ += get_repeated_stabilization_steane(registers, n_cycles=n_cycles,
        reset=reset, recovery=recovery, snapshot_type=snapshot_type)

    # Final readout
    circ.measure(qbReg, readout)

    return circ
    
def get_repeated_stabilization_steane(registers, n_cycles=1,
        reset=True, recovery=False, snapshot_type='statevector'):

    circ = get_empty_stabilizer_circuit_steane(registers)

    for current_cycle in range(n_cycles):
        circ += stabilizer_cycle_steane(registers,
            reset=reset,
            recovery=recovery
        )
        if snapshot_type:
            circ.snapshot('stabilizer_' + str(current_cycle), snapshot_type)
    return circ

def get_empty_stabilizer_circuit_steane(registers):
    """Create an empty qiskit circuit adapted for stabilizer circuits"""

    # Unpack registers
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

# %% All unflagged stabilizers
def stabilizer_cycle_steane(registers, reset=True, recovery=False):
    """Run all stabilizers for the [[7,1,3]] code. This assumes (at least) 3
    ancilla qubits"""

    # Create circuit and run stabilizers
    circ = get_empty_stabilizer_circuit_steane(registers)
    circ += _stabilizer_X0156(registers, reset)
    circ += _stabilizer_X0246(registers, reset)
    circ += _stabilizer_X0345(registers, reset)
    circ += _stabilizer_Z0156(registers, reset)
    circ += _stabilizer_Z0246(registers, reset)
    circ += _stabilizer_Z0345(registers, reset)

    # Recovery
    if recovery is True:
        print('Recovery not yet implemented, skipping this section')
    
    return circ

def _stabilizer_X0156(registers, reset=True):
    """Gives the circuit for running the X stabilizer on data qubits
    0, 1, 5 and 6.
    """

    # Create a circuit
    qbReg = registers.QubitRegister
    anReg = registers.AncillaRegister
    clReg = registers.SyndromeRegister
    circ = get_empty_stabilizer_circuit_steane(registers)
    anQb = anReg[0]

    circ.h(anQb)
    circ.cx(anQb, qbReg[0])
    circ.cx(anQb, qbReg[1])
    circ.cx(anQb, qbReg[5])
    circ.cx(anQb, qbReg[6])
    circ.h(anQb)

    # Measure
    circ.measure(anQb, clReg[0])
    
    # Reset
    if reset:
        circ.reset(anQb)

    return circ

def _stabilizer_X0246(registers, reset=True):
    """Gives the circuit for running the X stabilizer on data qubits
    0, 2, 4 and 6.
    """

    # Create a circuit
    qbReg = registers.QubitRegister
    anReg = registers.AncillaRegister
    clReg = registers.SyndromeRegister
    circ = get_empty_stabilizer_circuit_steane(registers)
    anQb = anReg[1]
    
    circ.h(anQb)
    circ.cx(anQb, qbReg[0])
    circ.cx(anQb, qbReg[2])
    circ.cx(anQb, qbReg[4])
    circ.cx(anQb, qbReg[6])
    circ.h(anQb)

    # Measure
    circ.measure(anQb, clReg[1])
    
    # Reset
    if reset:
        circ.reset(anQb)

    return circ

def _stabilizer_X0345(registers, reset=True):
    """Gives the circuit for running the X stabilizer on data qubits
    0, 3, 4 and 5.
    """

    # Create a circuit
    qbReg = registers.QubitRegister
    anReg = registers.AncillaRegister
    clReg = registers.SyndromeRegister
    circ = get_empty_stabilizer_circuit_steane(registers)
    anQb = anReg[2]
    
    circ.h(anQb)
    circ.cx(anQb, qbReg[0])
    circ.cx(anQb, qbReg[3])
    circ.cx(anQb, qbReg[4])
    circ.cx(anQb, qbReg[5])
    circ.h(anQb)

    # Measure
    circ.measure(anQb, clReg[2])
    
    # Reset
    if reset:
        circ.reset(anQb)

    return circ

def _stabilizer_Z0156(registers, reset=True):
    """Gives the circuit for running the Z stabilizer on data qubits
    0, 1, 5 and 6.
    """

    # Create a circuit
    qbReg = registers.QubitRegister
    anReg = registers.AncillaRegister
    clReg = registers.SyndromeRegister
    circ = get_empty_stabilizer_circuit_steane(registers)
    anQb = anReg[0]
    
    circ.cx(qbReg[0], anQb)
    circ.cx(qbReg[1], anQb)
    circ.cx(qbReg[5], anQb)
    circ.cx(qbReg[6], anQb)

    # Measure
    circ.measure(anQb, clReg[3])
    
    # Reset
    if reset:
        circ.reset(anQb)

    return circ

def _stabilizer_Z0246(registers, reset=True):
    """Gives the circuit for running the Z stabilizer on data qubits
    0, 2, 4 and 6.
    """

    # Create a circuit
    qbReg = registers.QubitRegister
    anReg = registers.AncillaRegister
    clReg = registers.SyndromeRegister
    circ = get_empty_stabilizer_circuit_steane(registers)
    anQb = anReg[1]
    
    circ.cx(qbReg[0], anQb)
    circ.cx(qbReg[2], anQb)
    circ.cx(qbReg[4], anQb)
    circ.cx(qbReg[6], anQb)

    # Measure
    circ.measure(anQb, clReg[4])
    
    # Reset
    if reset:
        circ.reset(anQb)

    return circ

def _stabilizer_Z0345(registers, reset=True):
    """Gives the circuit for running the X stabilizer on data qubits
    0, 3, 4 and 5.
    """

    # Create a circuit
    qbReg = registers.QubitRegister
    anReg = registers.AncillaRegister
    clReg = registers.SyndromeRegister
    circ = get_empty_stabilizer_circuit_steane(registers)
    anQb = anReg[2]
    
    circ.cx(qbReg[0], anQb)
    circ.cx(qbReg[3], anQb)
    circ.cx(qbReg[4], anQb)
    circ.cx(qbReg[5], anQb)

    # Measure
    circ.measure(anQb, clReg[5])
    
    # Reset
    if reset:
        circ.reset(anQb)

    return circ


# %% Internal testing/running
from qiskit.visualization import plot_histogram
if __name__ == '__main__':
    # Circuit settings
    n_cycles = 1
    reset = True
    recovery = False
    snapshot_type = 'statevector'

    # Define quantum registers and circuit
    qb = QuantumRegister(7, 'qubit')
    an = AncillaRegister(3, 'ancilla_qubit')
    readout= ClassicalRegister(7, 'readout')
    cr = ClassicalRegister(6, 'syndrome bits')
    registers = StabilizerRegistersSteane(qb, an, cr, readout)

    # Full circuit
    circ = get_full_stabilizer_circuit_steane(registers, n_cycles=n_cycles,
        reset=reset, recovery=recovery, snapshot_type=snapshot_type)


    # Transpilation
    circ_t = shortest_transpile_from_distribution(
        circ,
        print_cost=False,
        repeats=20,
        routing_method='sabre',  # basic lookahead stochastic sabre
        initial_layout=None,
        layout_method='sabre',  # trivial 'dense', 'noise_adaptive' sabre
        translation_method=None,  # 'unroller',  translator , synthesis
        optimization_level=1,
        **triangle_device_properties
    )

    # Run it
    n_shots = 2000
    results = execute(
        circ_t,  
        Aer.get_backend('qasm_simulator'),
        noise_model=None,
        shots=n_shots
    ).result()

    # Plot results
    display(plot_histogram(results.get_counts()))

