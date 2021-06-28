


#%% Test Int counter object
class Int(object):
    def __init__(self, value):
        self.value = value

    def __add__(self, other):
        self.value += other
        return self

    def __str__(self):
        return str(self.value)

    def inc(self):
        self.value += 1

my_counter = Int(0)
# def int_counter():
#     global my_counter
#     my_counter += 1

def test(hej = my_counter):
    if hej != my_counter:
        my_counter.value = hej
    # hej = hej + 1
    print(hej)
    print(my_counter)
    my_counter.inc()

def test_2(hej = Int(0)):
    hej = hej + 1

    print(hej)  
# %% Test pre-recovery snapshot
# Exp fit func
def monoExp(t, T, c, A):
    return (A-c) * np.exp(-t/T) + c

def idealExp(t,T):
    return 0.5 * np.exp(-t/T) + 0.5
import scipy
from simulator_program.data_analysis_tools import fidelity_from_scratch, perfect_stab_circuit, project_dm_to_logical_subspace_V1
from matplotlib import pyplot as plt
from simulator_program.custom_noise_models import standard_times
import numpy as np
import matplotlib.transforms as mtransforms
from simulator_program.stabilizers import label_counter, get_encoded_state
from simulator_program import decay
from qiskit.quantum_info.states.measures import state_fidelity
n_shots = 1024*4

n_cycles = 5
pauliop = 'ZZZZZ'
gate_times = standard_times

T1 = 40e3
T2 = 60e3

#%%
F_L, P_L ,time = fidelity_from_scratch(
    n_cycles, n_shots, T1=T1, T2=T2, encoding=False, project = True)
times = np.array([time['dm_' + str(n)] for n in range(label_counter.value)])
print("Base QEC (hexagonal) done")

# Decay encoded qubit
n_datapoints = 100
timespan = np.linspace(0, times[-1], n_datapoints)
res_0 = decay.get_idle_encoded_513(timespan, snapshot_type=['dm'], T1=T1, T2=T2)
fid_0_encoded = [state_fidelity(get_encoded_state(0, 0, None), res_0.data()['dm_'+str(index)])
                 for index in range(n_datapoints)]
fid_0_encoded_L = [state_fidelity([1, 0], project_dm_to_logical_subspace_V1(res_0.data()['dm_'+str(index)]))
           for index in range(n_datapoints)]
print("Decay encoded qubit done")

# # Perfect decoding (no noise in stabilizer cycle)
# fidelities_perf, time = perfect_stab_circuit(
#     n_cycles, n_shots, T1=T1, T2=T2)
# times_perf = np.array([time['dm_' + str(n)] for n in range(n_cycles)]+[time['dm_' + str(n_cycles)]])
# print("Perfect decoding (no noise in stabilizer cycle) done")
# %%
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
colors = plt.get_cmap("tab10")
color_count = 0
# Position of fractions
trans_offset = mtransforms.offset_copy(ax.transData, fig=fig,
                                       x=-0.05, y=-0.3, units='inches')
trans_offset_perf = mtransforms.offset_copy(ax.transData, fig=fig,
                                       x=0, y=0.1, units='inches')

trans_offset_lifetime = mtransforms.offset_copy(ax.transData, fig=fig,
                                       x=-0.05, y=0.15, units='inches')
                                       
# Plot normal QEC
fidelities = [F_L_i*P_L_i for F_L_i, P_L_i in zip(F_L,P_L)]
ax.plot(times/1000, fidelities,'.-', color=colors(color_count), label=r'$F$, (fid inital state)')
# Exp fit it
# p0 = (T1, 0, 0.9)  # start with values near those we expect
# pars, cov = scipy.optimize.curve_fit(monoExp, times[1:], fidelities[1:], p0)
# T, c, A = pars
# ax.text(times[-3]/1000,monoExp(times[-3], *pars), rf'$T_L ={T/1000:.0f}$ μs',color=colors(color_count), transform=trans_offset)
# ax.text(times[-3]/1000,monoExp(times[-3], *pars), rf'$P_L ={np.mean(P_L[1:]):.2f}$',color='k', transform=trans_offset_lifetime)
# ax.plot(times/1000, monoExp(times, *pars), ':', color=colors(color_count))
color_count += 1

# Plot normal QEC projected
ax.plot(times/1000, F_L,'.-', color=colors(color_count), label=r'$F_L$, (projected fid)')
# Exp fit it
# p0 = T1  # start with values near those we expect
# pars_L, cov = scipy.optimize.curve_fit(idealExp, times[1:]-times[1], F_L[1:], p0)
# T= pars_L[0]
# ax.plot(times/1000, idealExp(times-times[1], *pars_L), ':', color=colors(color_count))
color_count += 1

# Plot P_L
ax.plot(times/1000, P_L,'.-', color=colors(color_count), label=r'$P_L=F/F_L$')
color_count += 1

# Plot encoded qubit decay
ax.plot(timespan/1000, fid_0_encoded, '--',
        color=colors(color_count), label='Encoded qubit decay')
# p0 = (T1, 0, 0.9) # start with values near those we expect
# pars, cov = scipy.optimize.curve_fit(monoExp, timespan[:-5], fid_0_encoded[:-5], p0)
# T, c, A = pars
# trans_offset_lifetime = mtransforms.offset_copy(ax.transData, fig=fig,
#                                        x=-0.05, y=0.20, units='inches')
# ax.text(times[-3]/1000,fid_0_encoded[-3],rf'$T_L ={T/1000:.0f}$ μs',color=colors(color_count), transform=trans_offset_lifetime)
# print('Encoded qubit',r'$(A-c)e^{-t/T}+c$,'+f' T={T/1000:.1f} ns, A={A:.2f}, c={c:.3f}')
color_count += 1

# # Plot encoded qubit decay projected
# ax.plot(timespan/1000, fid_0_encoded_L, '--',
#         color=colors(color_count), label='Encoded qubit decay projected')
# color_count += 1

ax.set_xlabel('Time [μs]')
ax.set_ylabel(r'Probability')
ax.set_ylim((0.4,1.05))
ax.legend()
# %%
