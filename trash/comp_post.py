# %%
if __package__:
    from .post_process import *
    from .post_select import *
    from .stabilizers import logical_states, get_classical_register, StabilizerRegisters, encode_input_v2, get_repeated_stabilization, get_full_stabilizer_circuit
    from .custom_noise_models import standard_times
    from .idle_noise import add_idle_noise_to_circuit
    from .custom_transpiler import WACQT_device_properties, shortest_transpile_from_distribution
    from .custom_noise_models import thermal_relaxation_model_V2
    from .data_analysis_tools import fidelity_from_scratch, default_execute
else:
    from post_process import *
    from post_select import *
    from stabilizers import logical_states, get_classical_register, StabilizerRegisters, encode_input_v2, get_repeated_stabilization, get_full_stabilizer_circuit
    from custom_noise_models import standard_times
    from idle_noise import add_idle_noise_to_circuit
    from custom_transpiler import WACQT_device_properties, shortest_transpile_from_distribution
    from custom_noise_models import thermal_relaxation_model_V2
    from data_analysis_tools import fidelity_from_scratch, default_execute

from qiskit.quantum_info import state_fidelity
import numpy as np
from matplotlib import pyplot as plt
from qiskit.providers.aer.noise.errors import ReadoutError
from qiskit.providers.aer.noise.noise_model import NoiseModel

# %%
reset = True
recovery = False
flag = False
n_cycles = 4
n_shots = 1024/2

qb = QuantumRegister(5, 'code_qubit')
an = AncillaRegister(2, 'ancilla_qubit')
cr = get_classical_register(n_cycles, reset=reset,
                            recovery=recovery)
readout = ClassicalRegister(5, 'readout')
registers = StabilizerRegisters(qb, an, cr, readout)

circ = get_full_stabilizer_circuit(registers, n_cycles=n_cycles, reset=reset,
                                   recovery=recovery,
                                   snapshot_type='dm',
                                   conditional=True,
                                   encoding=False, theta=0, phi=0)


# circ = shortest_transpile_from_distribution(circ, print_cost=False,
#                                             **WACQT_device_properties)

circ, time = add_idle_noise_to_circuit(circ, return_time=True)


# %%
p1given0 = 0.1
p0given1 = p1given0
noise_model = thermal_relaxation_model_V2(
    gate_times=standard_times)  # NoiseModel()#
read_error = ReadoutError([[1 - p1given0, p1given0], [p0given1, 1 - p0given1]])
# noise_model.add_all_qubit_readout_error(read_error, ['measure'])
results = default_execute(
    circ, n_shots, noise_model)


def get_av_fidelities(states_and_counts, correct_state, n_shots):
    av_fidelities = []
    for cycle in states_and_counts:
        fid = 0
        for state, counts in cycle:
            fid += state_fidelity(state, correct_state)*counts
        av_fidelities.append(fid/n_shots)
    return av_fidelities


correct_state = logical_states(include_ancillas=None)[0]
fidelities_normal = get_av_fidelities(get_states_and_counts(
    results, n_cycles, post_process=False), correct_state, n_shots)
fidelities_post_process = get_av_fidelities(get_states_and_counts(
    results, n_cycles, post_process=True), correct_state, n_shots)

fidelities_select = [state_fidelity(post_selected_state, correct_state) for post_selected_state
                     in get_trivial_post_select_den_mat(results, n_cycles)]
select_counts = get_trivial_post_select_counts(
    results.get_counts(), n_cycles)
# %%

fig, axs = plt.subplots(2, figsize=(14, 10))
ax1 = axs[0]
ax2 = axs[1]

ax1.plot(range(n_cycles+1), fidelities_normal, 'o-', label='No processing')
ax1.plot(range(n_cycles+1), fidelities_select, 'o-', label='Post select')
ax1.plot(range(n_cycles+1), fidelities_post_process,
         'o-', label='Post process')
ax1.set_xlabel(r'Error detection cycle $n$')
ax1.set_ylabel('Post selected count')
ax1.legend()
ax1.grid(linewidth=1)


ax2.plot(range(n_cycles+1), select_counts, 'o-', label='No transpilation')
ax2.set_xlabel(r'Error detection cycle $n$')
ax2.set_ylabel(r'Post select fraction')
ax2.legend()
ax2.grid(linewidth=1)
plt.show()
# %% With correction
recovery = True

fidelities_QEC, times = fidelity_from_scratch(
    n_cycles, n_shots, gate_times={'feedback': 0}, encoding=False, transpile=False)
fig, axs = plt.subplots(2, figsize=(14, 10))
ax1 = axs[0]
ax2 = axs[1]

ax1.plot(range(n_cycles+1), fidelities_normal, 'o-', label='No processing')
ax1.plot(range(n_cycles+1), fidelities_select, 'o-', label='Post select')
ax1.plot(range(n_cycles+1), fidelities_post_process,
         'o-', label='Post process')
ax1.plot(range(n_cycles+1), fidelities_QEC, 'o-', label='QEC')
ax1.set_xlabel(r'Error detection cycle $n$')
ax1.set_ylabel('Fidelity')
ax1.legend()
ax1.grid(linewidth=1)


ax2.plot(range(n_cycles+1), select_counts, 'o-', label='No transpilation')
ax2.set_xlabel(r'Error detection cycle $n$')
ax2.set_ylabel(r'Post select count')
ax2.legend()
ax2.grid(linewidth=1)
plt.show()
# %%
