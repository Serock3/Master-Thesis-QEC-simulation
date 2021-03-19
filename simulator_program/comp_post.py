#%%
from post_process import *
from post_select import *
import numpy as np
from stabilizers import logical_states
from matplotlib import pyplot as plt
from qiskit.providers.aer.noise.errors import ReadoutError
from qiskit.providers.aer.noise.noise_model import NoiseModel

reset = True
recovery = False
flag = False
n_cycles = 5
qb = QuantumRegister(5, 'code_qubit')
an = AncillaRegister(2, 'ancilla_qubit')
cr = get_classical_register(n_cycles, reset=reset,
                            recovery=recovery, flag=False)
readout = ClassicalRegister(5, 'readout')
registers = StabilizerRegisters(qb, an, cr, readout)



circ = encode_input_v2(registers)
circ.snapshot('post_encoding', 'density_matrix')
# Stabilizer
circ += get_repeated_stabilization(registers, n_cycles=n_cycles,
                                reset=reset, recovery=recovery, flag=flag, snapshot_type='density_matrix')

n_shots = 100
p1given0 = 0.1
p0given1 = p1given0
noise_model = NoiseModel() #thermal_relaxation_model()
read_error = ReadoutError([[1 - p1given0, p1given0], [p0given1, 1 - p0given1]])
noise_model.add_all_qubit_readout_error(read_error, ['measure'])
results = execute(
    circ,
    Aer.get_backend('qasm_simulator'),
    noise_model=noise_model,
    shots=n_shots
).result()

def get_av_fidelities(states_and_counts, correct_state, n_shots):
    av_fidelities = []
    for cycle in states_and_counts:
        fid = 0
        for state, counts in cycle:
            fid += state_fidelity(state, correct_state)*counts
        av_fidelities.append(fid/n_shots)
    return av_fidelities

correct_state = logical_states(include_ancillas=None)[0]
fidelities_normal = get_av_fidelities(get_states_and_counts(results, n_cycles, post_process = False), correct_state, n_shots)
fidelities_post_process = get_av_fidelities(get_states_and_counts(results, n_cycles, post_process = True), correct_state, n_shots)

correct_statevec = get_trivial_state(circ)
fidelities_select = [state_fidelity(post_selected_state, correct_statevec) for post_selected_state
            in get_trivial_post_select_den_mat(results, n_cycles)]
select_counts = get_trivial_post_select_counts(
    results.get_counts(), n_cycles)
#%%

fig, axs = plt.subplots(2, figsize=(14, 10))
ax1 = axs[0]
ax2 = axs[1]

ax1.plot(range(n_cycles), fidelities_normal, 'o-', label='No processing')
ax1.plot(range(n_cycles), fidelities_select, 'o-', label='Post select')
ax1.plot(range(n_cycles), fidelities_post_process, 'o-', label='Post process')
ax1.set_xlabel(r'Error detection cycle $n$')
ax1.set_ylabel('Post selected count')
ax1.legend()
ax1.grid(linewidth=1)


ax2.plot(range(n_cycles), select_counts, 'o-', label='No transpilation')
ax2.set_xlabel(r'Error detection cycle $n$')
ax2.set_ylabel(r'Post select fraction')
ax2.legend()
ax2.grid(linewidth=1)

#%% With correction
recovery = True
cr = get_classical_register(n_cycles, reset=reset,
                            recovery=recovery, flag=False)

registers = StabilizerRegisters(qb, an, cr, readout)


circ = encode_input_v2(registers)
circ.snapshot('post_encoding', 'density_matrix')
# Stabilizer
circ += get_repeated_stabilization(registers, n_cycles=n_cycles,
                                reset=reset, recovery=recovery, flag=flag, snapshot_type='density_matrix')
results = execute(
    circ,
    Aer.get_backend('qasm_simulator'),
    noise_model=noise_model,
    shots=n_shots
).result()

fidelities_QEC = get_av_fidelities(get_states_and_counts(results, n_cycles, post_process = False), correct_state, n_shots)
fig, axs = plt.subplots(2, figsize=(14, 10))
ax1 = axs[0]
ax2 = axs[1]

ax1.plot(range(n_cycles), fidelities_normal, 'o-', label='No processing')
ax1.plot(range(n_cycles), fidelities_select, 'o-', label='Post select')
ax1.plot(range(n_cycles), fidelities_post_process, 'o-', label='Post process')
ax1.plot(range(n_cycles), fidelities_QEC, 'o-', label='QEC')
ax1.set_xlabel(r'Error detection cycle $n$')
ax1.set_ylabel('Fidelity')
ax1.legend()
ax1.grid(linewidth=1)


ax2.plot(range(n_cycles), select_counts, 'o-', label='No transpilation')
ax2.set_xlabel(r'Error detection cycle $n$')
ax2.set_ylabel(r'Post select count')
ax2.legend()
ax2.grid(linewidth=1)
# %%
