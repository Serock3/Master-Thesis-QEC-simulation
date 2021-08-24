# Script for testing the effects of resetting ancilla in the context of QEC

#%% Import modules
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from qiskit import *
from qiskit.visualization import plot_histogram

from qiskit.quantum_info import partial_trace
from qiskit.quantum_info import DensityMatrix
from qiskit.quantum_info import state_fidelity

# Our own files
from simulator_program.custom_noise_models import thermal_relaxation_model_V2
from simulator_program.custom_transpiler import *
from simulator_program.stabilizers import *
from simulator_program.post_select import *
from simulator_program.stabilizers import(
    _get_stabilizer_IXZZX,
    _get_stabilizer_XIXZZ,
    _get_stabilizer_XZZXI,
    _get_stabilizer_ZXIXZ
    )
#%% Modified stabilizer functions
# These reset ancillas to 1 instead of 0 by adding a bit-flip directly after.
# This is not meant to be a functionality and is thus only used here internally.

def mod_unflagged_stabilizer_cycle(registers, reset=True, recovery=False,
        current_cycle=0, current_step=0, num_ancillas=None):
    """Run all four stabilizers without flags, as well as an optional
    recovery. The input current_step is only relevant for flagged cycles, and
    should be set to 0 otherwise.

    NOTE: Maybe use a list if ancilla idices instead? E.g. ancillas = [1,2,1,2]
    Args:
        num_ancillas: Specifies how many acillas to spread the measurements over
    """

    # Use the previous behaviour if num_ancillas=None
    if not num_ancillas:
        if registers.AncillaRegister.size == 2:
            anQb_list = [registers.AncillaRegister[1]]*4
        elif registers.AncillaRegister.size >= 4:
            # I don't like this really, we don't use the flagged ciruit anymore so it shouldn't get the 0 stop by default
            anQb_list = [registers.AncillaRegister[n] for n in [1, 2, 3, 4]]
        else:
            Warning("Ancilla reg too small (this should never happen)")
    else:
        anQb_list = [registers.AncillaRegister[n % num_ancillas]
                     for n in range(4)]

    stabilizer_list = [mod_unflagged_stabilizer_XZZXI,
                       mod_unflagged_stabilizer_IXZZX,
                       mod_unflagged_stabilizer_XIXZZ,
                       mod_unflagged_stabilizer_ZXIXZ]

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
                                   syn_bit=syn_bit_list[i], reset=reset)
        circ.barrier()
    # Add an extra measurement to the next syndrome register
    # TODO: Make this compatible with using more than 1 ancilla
    #if recovery and not reset:
    #    if current_cycle < len(registers.SyndromeRegister[0])-1:
    #        circ.measure(anQb_list[-1],
    #                     registers.SyndromeRegister[0][current_cycle+1][current_step][4])

    # Recovery
    if recovery is True:
        circ += mod_unflagged_recovery(registers, reset, current_cycle)
        circ.barrier()
    return circ

def barrier_unflagged_stabilizer_cycle(registers, reset=True, recovery=False,
        current_cycle=0, current_step=0, num_ancillas=None):
    """Run all four stabilizers without flags, as well as an optional
    recovery. The input current_step is only relevant for flagged cycles, and
    should be set to 0 otherwise.

    NOTE: Maybe use a list if ancilla idices instead? E.g. ancillas = [1,2,1,2]
    Args:
        num_ancillas: Specifies how many acillas to spread the measurements over
    """

    # Use the previous behaviour if num_ancillas=None
    if not num_ancillas:
        if registers.AncillaRegister.size == 2:
            anQb_list = [registers.AncillaRegister[1]]*4
        elif registers.AncillaRegister.size >= 4:
            # I don't like this really, we don't use the flagged ciruit anymore so it shouldn't get the 0 stop by default
            anQb_list = [registers.AncillaRegister[n] for n in [1, 2, 3, 4]]
        else:
            Warning("Ancilla reg too small (this should never happen)")
    else:
        anQb_list = [registers.AncillaRegister[n % num_ancillas]
                     for n in range(4)]

    stabilizer_list = [_get_stabilizer_XZZXI,
                       _get_stabilizer_IXZZX,
                       _get_stabilizer_XIXZZ,
                       _get_stabilizer_ZXIXZ]

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
                                   syn_bit=syn_bit_list[i], reset=reset)
        circ.barrier()
    # Add an extra measurement to the next syndrome register
    # TODO: Make this compatible with using more than 1 ancilla
    if recovery and not reset:
        if current_cycle < len(registers.SyndromeRegister[0])-1:
            circ.measure(anQb_list[-1],
                         registers.SyndromeRegister[0][current_cycle+1][current_step][4])
            circ.barrier()

    # Recovery
    if recovery is True:
        circ += get_recovery(registers, reset, current_cycle)
        circ.barrier
    return circ

def mod_unflagged_stabilizer_XZZXI(registers, anQb=None, syn_bit=None, reset=True):
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
    # NOTE: Use the empty circuit below instead? It doesn't add the readout register
    # circ = QuantumCircuit(qbReg,anReg,clReg)
    if not anQb:
        if anReg.size == 2:
            anQb = anReg[1]
        else:
            anQb = anReg[1]
    circ.x(anQb)
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


def mod_unflagged_stabilizer_IXZZX(registers, anQb=None, syn_bit=None, reset=True):
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
    circ.x(anQb)
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


def mod_unflagged_stabilizer_XIXZZ(registers, anQb=None, syn_bit=None, reset=True):
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
    circ.x(anQb)
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


def mod_unflagged_stabilizer_ZXIXZ(registers, anQb=None, syn_bit=None, reset=True):
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
    circ.x(anQb)
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


def mod_unflagged_recovery(registers, reset=True, current_cycle=0, current_step=0):
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
        circ.x(qbReg[1]).c_if(syndrome_reg, 15-1)
        circ.z(qbReg[4]).c_if(syndrome_reg, 15-2)
        circ.x(qbReg[2]).c_if(syndrome_reg, 15-3)
        circ.z(qbReg[2]).c_if(syndrome_reg, 15-4)
        circ.z(qbReg[0]).c_if(syndrome_reg, 15-5)
        circ.x(qbReg[3]).c_if(syndrome_reg, 15-6)
        circ.x(qbReg[2]).c_if(syndrome_reg, 15-7)
        circ.z(qbReg[2]).c_if(syndrome_reg, 15-7)
        circ.x(qbReg[0]).c_if(syndrome_reg, 15-8)
        circ.z(qbReg[3]).c_if(syndrome_reg, 15-9)
        circ.z(qbReg[1]).c_if(syndrome_reg, 15-10)
        circ.x(qbReg[1]).c_if(syndrome_reg, 15-11)
        circ.z(qbReg[1]).c_if(syndrome_reg, 15-11)
        circ.x(qbReg[4]).c_if(syndrome_reg, 15-12)
        circ.x(qbReg[0]).c_if(syndrome_reg, 15-13)
        circ.z(qbReg[0]).c_if(syndrome_reg, 15-13)
        circ.x(qbReg[4]).c_if(syndrome_reg, 15-14)
        circ.z(qbReg[4]).c_if(syndrome_reg, 15-14)
        circ.x(qbReg[3]).c_if(syndrome_reg, 15-15)
        circ.z(qbReg[3]).c_if(syndrome_reg, 15-15)

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

def old_get_running_fidelity_data_den_mat(circ, n_cycles=15,n_shots=2048,
        noise_model=thermal_relaxation_model_V2(), post_select=True):
    '''Inputs:
    circ: The circuit to be tested
    correct_state: The correct state for comparison
    param_list: The error model parameters, currently only [T2, t_cz]
    n_shots: Number of shots to average over
    '''

    # Get correct state
    results = execute(
        circ,
        Aer.get_backend('qasm_simulator'),
        noise_model=None,
        shots=1,
    ).result()

    # TODO: Make this work if circuit it permuted for later stabilizers
    # TODO: More sophisticated key than '0x0'?
    #correct_state = snapshots['stabilizer_0'][[key for key in snapshots['stabilizer_0'] if int(key,16) == 0][0]]
    correct_state=results.data()['snapshots']['density_matrix']['stabilizer_0'][0]['value']

    # Run the circuit
    results = execute(
        circ,
        Aer.get_backend('qasm_simulator'),
        noise_model=noise_model,
        shots=n_shots
    ).result()

    # Post-selection
    fidelities = []
    snapshots = reformat_density_snapshot(results)
    select_fractions = []
    if post_select:
        for current_cycle in range(n_cycles):
            try:
                post_selection = snapshots['stabilizer_' + str(current_cycle)][[key for key in snapshots['stabilizer_0'] if int(key,16) == 0][0]]
                select_fraction = get_running_post_select_fraction_for_density_matrix_v2(results,n_shots,current_cycle)
                select_fractions.append(select_fraction)
                fidelities.append(state_fidelity(post_selection, correct_state))
            except:
                print("No selectable states")
                fidelities.append(-1)
                select_fractions.append(0)
        return fidelities, select_fractions

    else:
        cl_reg_size = len(list(results.get_counts().keys())[0].split()[1])
        counts = results.get_counts()
        snapshots = reformat_density_snapshot(results)  # TODO: remove this
        for current_cycle in range(n_cycles):
            fid = 0
            for key in snapshots['stabilizer_'+str(current_cycle)]:
                bin_string = bin(int(key, 16))[2:].zfill(
                    cl_reg_size*(current_cycle+1))[-cl_reg_size*(current_cycle+1):]
                current_state = snapshots['stabilizer_' +
                                          str(current_cycle)][key]
                for outcome in results.get_counts():
                    formated_outcome = outcome.replace(
                        ' ', '')[-cl_reg_size*(current_cycle+1):]
                    if formated_outcome == bin_string:
                        fid += state_fidelity(current_state,
                                              correct_state)*counts[outcome]
            fidelities.append(fid/n_shots)
        return fidelities
#%% Actual stuff
n_cycles=15

def get_testing_circuit(registers, reset, n_cycles):
    circ = encode_input_v2(registers)
    qb = registers.QubitRegister
    for current_cycle in range(n_cycles):
        if reset == 0:
            circ += get_stabilizer_cycle(registers,
                reset=True,
                recovery=True,
                current_cycle=current_cycle
            )
        elif reset == 1:
            circ += mod_unflagged_stabilizer_cycle(registers,
                reset=True,
                recovery=True,
                current_cycle=current_cycle
            )
        else:
            circ += get_stabilizer_cycle(registers,
                reset=False,
                recovery=True,
                current_cycle=current_cycle
            )
        circ.barrier()
        circ.append(Snapshot('stabilizer_'+str(current_cycle),
            'density_matrix', num_qubits=5), qb)
        circ.barrier()
    return circ

def get_ancilla_snapshots(registers, n_cycles):
    "Assumes that we do not reset"
    circ = encode_input_v2(registers)
    an = registers.AncillaRegister
    for current_cycle in range(n_cycles):
        circ += get_stabilizer_cycle(registers,
            reset=False,
            recovery=True,
            current_cycle=current_cycle
        )
        circ.barrier()
        circ.append(Snapshot('ancilla_'+str(current_cycle),
            'density_matrix', num_qubits=2), an)
        circ.barrier()
    return circ
#%%
# Registers
qb = QuantumRegister(5, 'code_qubit')
an = AncillaRegister(2, 'ancilla_qubit')
cr = get_classical_register(n_cycles, reset=False, recovery=True, flag=False) # Advanced list of registers
readout = ClassicalRegister(5, 'readout')
registers = StabilizerRegisters(qb, an, cr, readout)

# Without barrier, without reset
circ_rec = get_testing_circuit(registers, reset=False, n_cycles=n_cycles)

# Without barrier, with reset
circ_res_rec = get_testing_circuit(registers, reset=0, n_cycles=n_cycles)

# With barrier, reset to 1
circ_mod_rec = get_testing_circuit(registers, reset=1, n_cycles=n_cycles)

# With barrier, reset to 0
circ_nob_rec = get_testing_circuit(registers, reset=0, n_cycles=n_cycles)

# With barrier, without reset
circ_bar_rec = get_testing_circuit(registers, reset=False, n_cycles=n_cycles)


def get_standard_transpilation(circ):
    return shortest_transpile_from_distribution(circ, print_cost=False,
        repeats=10, routing_method='sabre', initial_layout=None,
        translation_method=None, layout_method='sabre',
        optimization_level=1, **WACQT_device_properties)

circ_rec_WACQT = get_standard_transpilation(circ_rec)
circ_res_rec_WACQT = get_standard_transpilation(circ_res_rec)
circ_mod_rec_WACQT = get_standard_transpilation(circ_mod_rec)
circ_nob_rec_WACQT = get_standard_transpilation(circ_nob_rec)
circ_bar_rec_WACQT = get_standard_transpilation(circ_bar_rec)

#%% Running simulations
n_shots=512
# Without reset, with recovery
fid_rec_t = get_running_fidelity_data_den_mat(circ_rec_WACQT, 
    n_cycles=n_cycles,
    n_shots=n_shots,
    noise_model=thermal_relaxation_model_V2(),
    post_select=False,
)
print('Check!')
# With reset, with recovery
fid_res_rec_t = get_running_fidelity_data_den_mat(circ_res_rec_WACQT, 
    n_cycles=n_cycles,
    n_shots=n_shots,
    noise_model=thermal_relaxation_model_V2(),
    post_select=False,
)
print('Check!')
#% With modified reset, with recovery
fid_mod_rec_t = old_get_running_fidelity_data_den_mat(circ_mod_rec_WACQT, 
    n_cycles=n_cycles,
    n_shots=n_shots,
    noise_model=thermal_relaxation_model_V2(),
    post_select=False,
)
print('Check!')

#% Without barrier, with recovery
fid_nob_rec_t = old_get_running_fidelity_data_den_mat(circ_nob_rec_WACQT, 
    n_cycles=n_cycles,
    n_shots=n_shots,
    noise_model=thermal_relaxation_model_V2(),
    post_select=False,
)
print('Check!')
#% With barrier, without reset
fid_bar_rec_t = old_get_running_fidelity_data_den_mat(circ_bar_rec_WACQT, 
    n_cycles=n_cycles,
    n_shots=n_shots,
    noise_model=thermal_relaxation_model_V2(),
    post_select=False,
)
print('Check!')


#%% Curve fitting with OLS
fid_list = [fid_rec_t, fid_res_rec_t, fid_mod_rec_t, fid_nob_rec_t, fid_bar_rec_t]
theta_list = []
for fid in fid_list:
    x_D = np.ones((n_cycles-1,2))
    for i in range(n_cycles-1):
        x_D[i][1] += i
    y = np.log( np.reshape(np.asarray(fid), (n_cycles,1)) )[0:14]
    theta = np.dot(np.dot(np.linalg.inv(np.dot(x_D.T, x_D)), x_D.T), y)
    theta_list.append(theta)
    print(' ')
#
x = np.linspace(0,15,100)
y_pred_list = []
for theta in theta_list:
    y_pred = np.exp(theta[0]) * np.exp(x*theta[1])
    y_pred_list.append(y_pred)

# %% Plotting
fig, axs = plt.subplots(1, figsize=(10, 6))
x_dis = np.arange(1,n_cycles+1)
# Plot 3: Recovery, post-selection and nothing
#axs.plot(x_dis, fid_rec_t, 'o', color='red', label='No reset')
#axs.plot(x_dis, fid_res_rec_t, 'o', color='blue', label='Reset to 0 state')
axs.plot(x_dis, fid_mod_rec_t, 'o', color='orange', label='Reset to 1 state')
axs.plot(x_dis, fid_nob_rec_t, 'o', color='green', label='Reset to 0 state')
#axs.plot(x_dis, fid_bar_rec_t, 'o', color='black', label='Added barriers, no reset')
#axs.plot(x_dis, fid_bar_rec, 'o', color='cyan', label='Added barriers, no reset, no transpilation')
#%
#axs.plot(x, y_pred_list[0], color='red')
#axs.plot(x, y_pred_list[1], color='blue')
axs.plot(x, y_pred_list[2], color='orange')
axs.plot(x, y_pred_list[3], color='green')
#axs.plot(x, y_pred_list[4], color='black')

axs.set_xlabel('Number of cycles')
axs.set_ylabel('Average fidelity')
axs.set_title('Average fidelites after error correction, with fitted curves (OLS)')
axs.legend()
axs.grid(linewidth=1)

#%% Testing recovery
from qiskit.quantum_info import state_fidelity

n_cycles=4
# Registers
qb = QuantumRegister(5, 'code_qubit')
an = AncillaRegister(2, 'ancilla_qubit')
cr = get_classical_register(n_cycles, reset=True, recovery=True, flag=False) # Advanced list of registers
readout = ClassicalRegister(5, 'readout')
registers = StabilizerRegisters(qb, an, cr, readout)


circ_mod_rec = encode_input_v2(registers)
#circ_mod_rec.x(qb[4])
for current_cycle in range(n_cycles):
    circ_mod_rec += mod_unflagged_stabilizer_cycle(registers,
                                   reset=True,
                                   recovery=True,
                                   current_cycle=current_cycle
                                   )
    circ_mod_rec.barrier()
    circ_mod_rec.append(Snapshot('stabilizer_'+str(current_cycle),
        'density_matrix', num_qubits=5), qb)
    circ_mod_rec.barrier()

results = execute(
        circ_mod_rec,  
        Aer.get_backend('qasm_simulator'),
        noise_model=thermal_relaxation_model_V2(),
        shots=n_shots
).result()

correct_state = logical_states(include_ancillas=None)
snap = results.data()['snapshots']['density_matrix']['stabilizer_0'][0]['value']
fid = state_fidelity(correct_state[0], snap)
print(fid)
print()
# %% Analyze ancilla
n_cycles = 15
n_shots = 4096
state_00 = np.array([1., 0., 0., 0.])
state_01 = np.array([0., 1., 0., 0.])
state_10 = np.array([0., 0., 1., 0.])
state_11 = np.array([0., 0., 0., 1.])

circ = get_ancilla_snapshots(registers, n_cycles=n_cycles)
circ_t = get_standard_transpilation(circ)
noise_model=thermal_relaxation_model_V2()


# Get correct state
#results = execute(
#    circ,
#    Aer.get_backend('qasm_simulator'),
#    noise_model=None,
#    shots=1,
#).result()
#correct_state=results.data()['snapshots']['density_matrix']['stabilizer_0'][0]['value']
# Run the circuit
results = execute(
    circ_t,
    Aer.get_backend('qasm_simulator'),
    noise_model=noise_model,
    shots=n_shots
).result()
#%
def reformat_ancilla_snapshot(results) -> dict:
    """
    DEPRICATED
    Reformats the snapshot data of the results object to be a
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
#%
fidelities = []
count_00 = 0
count_01 = 0
count_10 = 0
count_11 = 0
count_none = 0
cl_reg_size = len(list(results.get_counts().keys())[0].split()[1])
counts = results.get_counts()
snapshots = reformat_ancilla_snapshot(results)
for current_cycle in range(n_cycles):
#    fid = 0
    for key in snapshots['ancilla_'+str(current_cycle)]:
        bin_string = bin(int(key, 16))[2:].zfill(
            cl_reg_size*(current_cycle+1))[-cl_reg_size*(current_cycle+1):]
        current_state = snapshots['ancilla_' +
                                  str(current_cycle)][key]
        #print(bin_string)
        for outcome in results.get_counts():
            tester=False
            formated_outcome = outcome.replace(
                ' ', '')[-cl_reg_size*(current_cycle+1):]
            if formated_outcome == bin_string:
                fid_00 = state_fidelity(current_state, state_00)
                fid_01 = state_fidelity(current_state, state_01)
                fid_10 = state_fidelity(current_state, state_10)
                fid_11 = state_fidelity(current_state, state_11)
                if fid_00 > 0.9:
                    count_00 += counts[outcome]
                    tester=True
                if fid_01 > 0.9:
                    count_01 += counts[outcome]
                    if tester:
                        print('Warning: both 00 and 01 show high fidelity')
                    tester=True
                if fid_10 > 0.9:
                    count_10 += counts[outcome]
                    if tester:
                        print('Warning: both 10 and 01 or 00 show high fidelity')
                    tester=True
                if fid_11 > 0.9:
                    count_11 += counts[outcome]
                    if tester:
                        print('Warning: both 11 and another state show high fidelity')
                    tester=True
                if not tester:
                    count_none += counts[outcome]
                
#    fidelities.append(fid/n_shots)
print('Count 00: ', count_00)
print('Count 01: ', count_01)
print('Count 10: ', count_10)
print('Count 11: ', count_11)
print('Count None: ', count_none)
print('Expected count:', n_shots*n_cycles)
#print(counts)
# %%
