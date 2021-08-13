# Disable pylinter giving false positives of error.
# pylint: disable=E1101
#%% Import modules
# Standard libraries
import warnings

# Qiskit
from qiskit import QuantumCircuit
from qiskit.providers.aer.library import set_density_matrix, set_statevector
from qiskit.circuit.instruction import Instruction
from qiskit.converters import circuit_to_dag

#%%

# Calling add_snapshot_to_circuit here causes circular imports, so for now this
# works as a completely separate function, albeit with similar functionality.
# TODO: Better solution?
def add_stop_to_circuit(circ, snapshot_type='density_matrix',
                        snap_label='end', qubits=None, conditional=True,
                        **kwargs):
    """TODO: Maybe this is too similar to add_snapshot_to_circuit() to warrant
    a separate function. Its mostly here for clarity in function names and to
    change some of the default arguments (i.e. defaulting conditional to True).
    
    Adds a 'stop' to a circuit in the form of a snapshot."""
    if qubits is None:
        qubits = circ.qubits
    
    circ.barrier()
    if snapshot_type == 'dm' or snapshot_type == 'density_matrix':
        circ.save_density_matrix(qubits, label=snap_label, conditional=conditional)
    return circ

def add_start_to_circuit(circ, state, simulator_type='density_matrix'):
    """Adds a start to the circuit in the form of a set_density_matrix
    (or set_statevector) instruction.
    
    Args:
        circ: QuantumCircuit object to add the initialization to.
        state: Density matrix or statevector to initialize.
        simulator_type: Type of simulation to be carried out for the circuit.
                        Can be either 'density_matrix' ('dm') or 
                        'statevector' ('sv'), defaults to 'density_matrix'.
                        
    Returns:
        new_circ: QuantumCircuit object identical to the circ input, with the
                  exception of an set_density_matrix (or set_statevector) at
                  the start.
    """

    # Create a new circuit with the same registers
    new_circ = QuantumCircuit()
    for reg in circ.qregs + circ.cregs:
        new_circ.add_register(reg)
    
    # Initialize the circuit
    if simulator_type == 'statevector' or simulator_type == 'sv':
        new_circ.set_statevector(state)
    elif simulator_type == 'density_matrix' or simulator_type == 'dm':
        new_circ.set_density_matrix(state)
    else:
        warnings.warn('Invalid simulator type, assuming density matrix.')
        new_circ.set_density_matrix(state)

    # Append everything already in the circuit
    dag = circuit_to_dag(circ)
    for node in dag.op_nodes():
        new_circ.append(node.op, node.qargs, node.cargs)
    
    return new_circ

def split_circuit(circ, add_stop=True, **kwargs):
    """Splits up a single circuit into several. The splitting points must be
    defined by a custom gate, defined in 'add_split_marker()'.
    """

    # List of all subcircuits
    circ_list = []

    # Create a new circuit with the same registers
    new_circ = QuantumCircuit()
    for reg in circ.qregs + circ.cregs:
        new_circ.add_register(reg)

    # Iterate through all nodes 
    dag = circuit_to_dag(circ)
    for node in dag.op_nodes():

        # If we found a split, append the old circuit and start a new
        if node.name == 'split':
            if add_stop:
                add_stop_to_circuit(new_circ, **kwargs)
            circ_list.append(new_circ)
            new_circ = QuantumCircuit()
            for reg in circ.qregs + circ.cregs:
                new_circ.add_register(reg)

        # Otherwise, append the gate
        else:
            new_circ.append(node.op, node.qargs, node.cargs)

    # Append the final circuit
    if add_stop:
        add_stop_to_circuit(new_circ, **kwargs)
    circ_list.append(new_circ)

    return circ_list

def add_split_marker(circ):
    """Adds a so-called 'split marker' custom gate to a circuit. This marker 
    signals to the function split_circuit() to cut off the circuit here. Any 
    subsequent gates are then made into a separate circuit with the same 
    registers.
    """

    # Create a custom instruction that covers all qubits+clbits
    split_gate = Instruction(name='split', 
                            num_qubits=len(circ.qubits), 
                            num_clbits=0, # This does not seem to matter
                            params=[0],) # No clue what this is, but is mandatory
                            #label='split') # In documentation, but could not be found here. TODO: check version?
    split_gate._directive = True # By making it a directive (like barrier) we can
                                 # bypass the transpiler easier
    # Append the gate
    circ.append(split_gate, circ.qubits, circ.clbits)
    return circ

# TODO: Unfinished, not fully sure how I want it to work
#def combine_syndromes(syndromes, syn_length=4):
#    """Combines a list of syndromes into one full syndrome. Assumes syndromes
#    are given in hexadecimal, beginning with '0x'. It also assumes that the 
#    latest measurement (and latest entry in the syndrome list) is the most 
#    significant number, following Qiskits convention."""
#    
#    if isinstance(syn_length, int):
#        formatted_syndromes = [bin(int(syndrome,16))[2:].zfill(syn_length) 
#                               for syndrome in syndromes]
#        formatted_syndromes.reverse()
#        full_syndrome = hex(int(''.join(formatted_syndromes),2))
#    return full_syndrome
#
