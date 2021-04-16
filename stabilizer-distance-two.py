# Code intended to reproduce the experiment/simulation done by Delft (Logical-qubit operations in an error-detecting surface code)
# and/or (Repeated quantum error detection in a surface code) on the distance 2 stabilizer code with post selection.

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
from simulator_program.custom_transpiler import *
from typing import List
from qiskit.quantum_info import Statevector, state_fidelity
from simulator_program.custom_noise_models import *
from simulator_program.idle_noise import *
from simulator_program.post_select import get_trivial_post_select_counts, get_trivial_post_select_den_mat, get_trivial_exp_value
from simulator_program.stabilizers import add_snapshot_to_circuit
from matplotlib import pyplot as plt
from qiskit.quantum_info.operators.symplectic.pauli import Pauli
# %% Logical states (for initialization)


def logical_states(include_ancillas='front') -> List[List[float]]:
    """Returns the logical states for the distance two code.

    Args:
        include_ancillas (str/None, optional): Whether to append the ancillas by tensor product to the end. Defaults to True.

    Returns:
        List[List[float]]: List of both logical states
    """
    logical_0 = np.zeros(2**4)
    logical_0[0b0000] = 1/np.sqrt(2)
    logical_0[0b1111] = 1/np.sqrt(2)

    logical_1 = np.zeros(2**4)
    logical_1[0b0101] = 1/np.sqrt(2)
    logical_1[0b1010] = 1/np.sqrt(2)

    if include_ancillas:
        # Add three ancillas in |0>
        an0 = np.zeros(2**3)
        an0[0] = 1.0
        if include_ancillas == 'front':
            logical_1 = np.kron(logical_1, an0)
            logical_0 = np.kron(logical_0, an0)
        elif include_ancillas == 'back':
            logical_1 = np.kron(an0, logical_1)
            logical_0 = np.kron(an0, logical_0)
    # TODO: Make equivalent funcs also return Stabilizer object?
    return [Statevector(logical_0), Statevector(logical_1)]

# %% Custom circuits


def pipelined_delft(n_cycles=1, reset=True, **kwargs):
    qbReg = QuantumRegister(4, 'code_qubit')
    anReg = AncillaRegister(3, 'ancilla_qubit')

    clRegs = [ClassicalRegister(3, 'syndrome_cycle_' + str(i))
              for i in range(n_cycles)]
    # clReg = ClassicalRegister(3, 'syndrome_bit')
    # readout = ClassicalRegister(4, 'readout')

    circ = QuantumCircuit(qbReg, anReg, *clRegs)
    circ.set_density_matrix(logical_states('back')[0])
    add_snapshot_to_circuit(circ, ['exp', 'dm'], 0, conditional=[
                            True, False], qubits=qbReg, pauliop='ZZII')

    for cycle in range(n_cycles):
        # Blue half, (XXXX) on the D register
        # TODO: add 'parking'?
        for D in qbReg:
            circ.ry(-1*np.pi/2, D)
        circ.ry(-1*np.pi/2, anReg[1])

        circ.cz(qbReg[1], anReg[1])
        circ.cz(qbReg[0], anReg[1])
        circ.cz(qbReg[3], anReg[1])
        circ.cz(qbReg[2], anReg[1])

        for D in qbReg:
            circ.ry(np.pi/2, D)
        circ.ry(np.pi/2, anReg[1])
        # Order of stabs are {Z1Z3,XXXX ,Z2Z4}
        circ.measure(anReg[1], clRegs[cycle][1])
        if reset:
            circ.reset(anReg[1])
        circ.barrier()
        # Green half, (Z1Z3,Z2Z4) on the D register
        circ.ry(np.pi/2, anReg[0])
        circ.ry(np.pi/2, anReg[2])
        circ.cz(qbReg[0], anReg[0])
        circ.cz(qbReg[1], anReg[2])
        circ.cz(qbReg[2], anReg[0])
        circ.cz(qbReg[3], anReg[2])
        circ.ry(np.pi/2, anReg[0])
        circ.ry(np.pi/2, anReg[2])
        circ.measure(anReg[0], clRegs[cycle][0])
        circ.measure(anReg[2], clRegs[cycle][2])
        if reset:
            circ.reset(anReg[0])
            circ.reset(anReg[2])
        circ.barrier()
        add_snapshot_to_circuit(circ, ['exp', 'dm'], cycle+1, conditional=[
                            True, False], qubits=qbReg, pauliop='ZZII')
        # circ.save_density_matrix(qubits=list(
        #     qbReg), label='stabilizer_' + str(cycle), conditional=True)
        # circ.save_expectation_value(
        #     Pauli('ZZII'), qbReg, label='exp_value_'+str(cycle), conditional=True)
    return circ
# %% Custom noise models

# Info of T1/T2 and gate times is the Mendeley paper


# %% Demo
if __name__ == '__main__':
    n_cycles = 10
    circ = pipelined_delft(n_cycles)
    # display(circ.draw(output='mpl'))

    n_shots = 100
    simulator = Aer.get_backend('aer_simulator')  # qasm_simulator
    simulator.set_option("method", 'density_matrix')
    # circ = transpile(circ, simulator)

    # Run and get saved data
    results = simulator.run(add_idle_noise_to_circuit(circ),
                            shots=n_shots,
                            noise_model=thermal_relaxation_model_V2()).result()

    trivial_key = '101'  # A trivial syndrome is given by 101 and not 000 here
    correct_state = logical_states(None)[0]
    fidelities_select = [state_fidelity(post_selected_state, correct_state) for post_selected_state
                         in get_trivial_post_select_den_mat(results, n_cycles, trivial_key)]
    select_counts = get_trivial_post_select_counts(
        results.get_counts(), n_cycles, trivial_key)

    trivial_key_list = [hex(int(trivial_key*(current_cycle+1), 2))
                        for current_cycle in range(n_cycles)]
    exp_values = get_trivial_exp_value(results, n_cycles, trivial_key)
    fig, axs = plt.subplots(2, figsize=(14, 10))
    ax1 = axs[0]
    ax2 = axs[1]

    # ax1.plot(range(n_cycles), fidelities_normal, 'o-', label='No processing')
    ax1.plot(range(n_cycles+1), fidelities_select, 'o-', label='Post select')
    ax1.plot(range(n_cycles+1), exp_values, 'o-', label='Post select exp')
    # ax1.plot(range(n_cycles), fidelities_post_process, 'o-', label='Post process')
    ax1.set_xlabel(r'Error detection cycle $n$')
    ax1.set_ylabel('Post selected count')
    ax1.set_ylim(0, 1)
    ax1.legend()
    ax1.grid(linewidth=1)

    ax2.plot(range(n_cycles+1), select_counts, 'o-', label='No transpilation')
    ax2.set_xlabel(r'Error detection cycle $n$')
    ax2.set_ylabel(r'Post select fraction')
    ax2.legend()
    ax2.grid(linewidth=1)

# %%
