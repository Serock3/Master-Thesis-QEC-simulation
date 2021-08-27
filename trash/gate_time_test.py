# %%
import time
import matplotlib.pyplot as plt
import numpy as np
import scipy

# Our own files
from simulator_program.custom_noise_models import (standard_times)
from simulator_program.data_analysis_tools import (fidelity_from_scratch,
                                                   perfect_stab_circuit,
                                                   scale_gate_times)
default_colors = plt.get_cmap("tab10")



T1 = 40e3
def monoExp(t, T, c, A):
    return (A-c) * np.exp(-t/T) + c
# For the projected qubit we set A=1 and c=0.5
def idealExp(t,T):
    return 0.5 * np.exp(-t/T) + 0.5

p0 = (T1, 0.4, 0.9)  # start with values near those we expect*
def_colors = plt.get_cmap("tab10")

def get_circuit_time_formula(gate_times = {}):
    tot_gate_times= standard_times.get_gate_times(gate_times)
    return (8*tot_gate_times['h']+16*tot_gate_times['cz']+ 4*tot_gate_times['measure']+ \
            tot_gate_times['feedback'] + tot_gate_times['delay'])/1000

#%% Initialize settings
# Comment away the parts not used for the specific run
n_cycles = 15
n_shots = 1024*8*2

# Single qubit gate
#resolution = 7
#single_qubit_min = 0
#single_qubit_max = 60
#single_qubit_span = np.linspace(single_qubit_min, single_qubit_max, resolution)

# Two qubit gate
#resolution = 17
#two_qubit_min = 0
#two_qubit_max = 400
#two_qubit_span = np.linspace(two_qubit_min, two_qubit_max, resolution)

# Measurement
#resolution = 16
#measure_min = 0
#measure_max = 1500
#measure_span = np.linspace(measure_min, measure_max, resolution)

# Feedback
#resolution = 17
#feedback_min = 0
#feedback_max = 8000
#feedback_span = np.linspace(feedback_min, feedback_max, resolution)

# Delay
#resolution = 17
#delay_min = 0
#delay_max = 8000
#delay_span = np.linspace(delay_min, delay_max, resolution)

# All times (except delay)
scalings = np.array([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0])
resolution = len(scalings)
all_times = scale_gate_times(gate_times=standard_times, scalings=scalings)
single_qubit_span = np.zeros(resolution)
two_qubit_span = np.zeros(resolution)
measure_span = np.zeros(resolution)
feedback_span = np.zeros(resolution)
for i in range(resolution):
    single_qubit_span[i] = all_times['single_qubit_gate'][i]
    two_qubit_span[i] = all_times['two_qubit_gate'][i]
    measure_span[i] = all_times['measure'][i]
    feedback_span[i] = all_times['feedback'][i]

fid_span = np.zeros((resolution, n_cycles+1))
P_L_span = np.zeros((resolution, n_cycles+1))
time_span = np.zeros((resolution, n_cycles+1))
par_span = np.zeros((resolution, 3))
par_L_span = np.zeros((resolution, 3))
cov_span = np.zeros((resolution, 3,3))
cov_L_span = np.zeros((resolution, 3,3))

# Define variables to save/load
# Scaling single qubit
#function_data = [n_cycles,
#                 n_shots,
#                 resolution,
#                 single_qubit_min,
#                 single_qubit_max,
#                 single_qubit_span,
#                 fid_span,
#                 P_L_span,
#                 time_span,
#                 par_span,
#                 par_L_span,
#                 cov_span,
#                 cov_L_span]

# Scaling two-qubit gates
#function_data = [n_cycles,
#                 n_shots,
#                 resolution,
#                 two_qubit_min,
#                 two_qubit_max,
#                 two_qubit_span,
#                 fid_span,
#                 P_L_span,
#                 time_span,
#                 par_span,
#                 par_L_span,
#                 cov_span,
#                 cov_L_span]

# Scaling measure time
#function_data = [n_cycles,
#                 n_shots,
#                 resolution,
#                 measure_min,
#                 measure_max,
#                 measure_span,
#                 fid_span,
#                 P_L_span,
#                 time_span,
#                 par_span,
#                 par_L_span,
#                 cov_span,
#                 cov_L_span]

# Scaling feedback time
#function_data = [n_cycles,
#                 n_shots,
#                 resolution,
#                 feedback_min,
#                 feedback_max,
#                 feedback_span,
#                 fid_span,
#                 P_L_span,
#                 time_span,
#                 par_span,
#                 par_L_span,
#                 cov_span,
#                 cov_L_span]

# Scaling delay time
#function_data = [n_cycles,
#                 n_shots,
#                 resolution,
#                 delay_min,
#                 delay_max,
#                 delay_span,
#                 fid_span,
#                 P_L_span,
#                 time_span,
#                 par_span,
#                 par_L_span,
#                 cov_span,
#                 cov_L_span]

# Scaling all gate times:
function_data = [n_cycles,
                 n_shots,
                 resolution,
                 single_qubit_span,
                 two_qubit_span,
                 measure_span,
                 feedback_span,
                 scalings,
                 fid_span,
                 P_L_span,
                 time_span,
                 par_span,
                 par_L_span,
                 cov_span,
                 cov_L_span]

# %% Test run
num_data_points = resolution
print('Will take roughly ',num_data_points*10, ' minutes to run')

for i in range(resolution):

    seconds = time.time()

    # Define the gatetime we're scaling
    gate_times = {
        'h': single_qubit_span[i],
        'cz': two_qubit_span[i],
        'measure': measure_span[i],
        'feedback': feedback_span[i],
        #'delay': delay_span[i],
    }

    # Calculate F_L and P_L
    fid_span[i], P_L_span[i], times = fidelity_from_scratch(
        n_cycles, n_shots, gate_times,
        T1=40e3, T2=60e3, encoding=False, transpile=False, project=True)
    
    # Convert time to list
    time_span[i] = np.array([times['dm_'+str(i)]
                                for i in range(n_cycles+1)])
    
    # Optimize curve
    pars, cov = scipy.optimize.curve_fit(
        monoExp, time_span[i][1:], (fid_span[i]*P_L_span[i])[1:], p0)
    pars_L, cov_L = scipy.optimize.curve_fit(
        idealExp, time_span[i][1:]-time_span[i][1], fid_span[i][1:], T1)
        
    # Store curve parameters
    par_span[i] = pars
    par_L_span[i] = pars_L
    cov_span[i] = cov
    cov_L_span[i] = cov_L

    # Plot results of current iteration
    plt.plot(time_span[i]/1000, fid_span[i] *
             P_L_span[i], 'o', label='F', color=def_colors(0))
    plt.plot(time_span[i]/1000, fid_span[i],
             '^', label='F_L', color=def_colors(2))
    plt.plot(time_span[i]/1000, P_L_span[i],
             's', label='P_L', color=def_colors(3))

    plt.plot(time_span[i]/1000, monoExp(time_span[i],
                                        *pars), '--', color=def_colors(0))
    plt.plot(time_span[i]/1000, idealExp(time_span[i]-time_span[i][1], *pars_L), '--', color=def_colors(2))
    plt.plot(time_span[i]/1000, monoExp(time_span[i], *pars) /
             idealExp(time_span[i]-time_span[i][1], *pars_L), '--', color=def_colors(3))
    plt.title(
        #r'$t_{1qb}$ = ' + str(int(single_qubit_span[i]/1000)) + 
        #r'$t_{2qb}$ = ' + str(int(two_qubit_span[i]/1000)) + 
        r'$t_{measure}$ = ' + str(int(measure_span[i]/1000)) + 
        #r'$t_{feedback}$ = ' + str(int(feedback_span[i]/1000)) + 
        #r'$t_{delay}$ = ' + str(int(delay_span[i]/1000)) + 
        ' \n-> F: T='+str(np.round(pars[0]/1000, 1)) + 
        ' c='+str(np.round(pars[1], 2)) + ' A='+str(np.round(pars[2], 2)) +
        '\n F_L: '+'T='+str(np.round(pars_L[0]/1000, 1)))
    plt.legend()
    plt.show()

    print('This took ', int(time.time()-seconds), 'seconds')
    print('current_iteration: ', i)


#%% Save data: Uncomment the relevant block of code to save it
#with open('data/single_qubit_test_data.npy', 'wb') as f:
#    for data in function_data:
#        np.save(f, data)
#%%
#with open('data/two_qubit_test_data.npy', 'wb') as f:
#    for data in function_data:
#        np.save(f, data)
#%%
#with open('data/measure_test_data.npy', 'wb') as f:
#    for data in function_data:
#        np.save(f, data)
#%%
#with open('data/feedback_test_data.npy', 'wb') as f:
#    for data in function_data:
#        np.save(f, data)
#%%
#with open('data/delay_test_data.npy', 'wb') as f:
#    for data in function_data:
#        np.save(f, data)


# %% Load
with open('data/two_qubit_test_data.npy', 'rb') as f:
        n_cycles = np.load(f)
        n_shots = np.load(f)
        resolution = np.load(f)
        two_qubit_min= np.load(f)
        two_qubit_max= np.load(f)
        two_qubit_span= np.load(f)
        fid_span = np.load(f)
        P_L_span_two_qubit = np.load(f)
        time_span = np.load(f)
        par_span = np.load(f)
        par_L_span_two_qubit = np.load(f)
        cov_span = np.load(f)
        cov_L_span_two_qubit = np.load(f)
with open('data/measure_test_data.npy', 'rb') as f:
        n_cycles = np.load(f)
        n_shots = np.load(f)
        resolution = np.load(f)
        measure_min= np.load(f)
        measure_max= np.load(f)
        measure_span= np.load(f)
        fid_span = np.load(f)
        P_L_span_measure = np.load(f)
        time_span = np.load(f)
        par_span = np.load(f)
        par_L_span_measure = np.load(f)
        cov_span = np.load(f)
        cov_L_span_measure = np.load(f)
with open('data/feedback_test_data.npy', 'rb') as f:
        n_cycles = np.load(f)
        n_shots = np.load(f)
        resolution = np.load(f)
        feedback_min= np.load(f)
        feedback_max= np.load(f)
        feedback_span= np.load(f)
        fid_span = np.load(f)
        P_L_span_feedback = np.load(f)
        time_span = np.load(f)
        par_span = np.load(f)
        par_L_span_feedback = np.load(f)
        cov_span = np.load(f)
        cov_L_span_feedback = np.load(f)
with open('data/delay_test_data.npy', 'rb') as f:
        n_cycles = np.load(f)
        n_shots = np.load(f)
        resolution = np.load(f)
        delay_min= np.load(f)
        delay_max= np.load(f)
        delay_span= np.load(f)
        fid_span = np.load(f)
        P_L_span_delay = np.load(f)
        time_span = np.load(f)
        par_span = np.load(f)
        par_L_span_delay = np.load(f)
        cov_span = np.load(f)
        cov_L_span_delay = np.load(f)

with open('data/all_gates_test_data.npy', 'rb') as f:
        n_cycles = np.load(f)
        n_shots = np.load(f)
        resolution = np.load(f)
        single_qubit_span_all = np.load(f)
        two_qubit_span_all = np.load(f)
        measure_span_all = np.load(f)
        feedback_span_all = np.load(f)
        scalings = np.load(f)
        fid_span = np.load(f)
        P_L_span_all = np.load(f)
        time_span = np.load(f)
        par_span = np.load(f)
        par_L_span_all = np.load(f)
        cov_span = np.load(f)
        cov_L_span_all = np.load(f)
#%% Plots

import matplotlib.transforms as mtransforms
fig, ax = plt.subplots(1,2, figsize=(10, 4))
def_colors = plt.get_cmap("tab10")

# Calculate cycle times
times_all = []
for i in range(len(scalings)):
    single_qubit = all_times['single_qubit_gate'][i]
    two_qubit = all_times['two_qubit_gate'][i]
    measure = all_times['measure'][i]
    feedback = all_times['feedback'][i]
    times_all.append(get_circuit_time_formula({'h': single_qubit,
                                               'cz': two_qubit,
                                               'measure': measure,
                                               'feedback': feedback}))
times_two_qubit = []
for time in two_qubit_span:
    times_two_qubit.append(get_circuit_time_formula({'cz': time}))
times_measure = []
for time in measure_span:
    times_measure.append(get_circuit_time_formula({'measure': time}))
times_feedback = []
for time in feedback_span:
    times_feedback.append(get_circuit_time_formula({'feedback': time}))
times_delay = []
for time in delay_span:
    times_delay.append(get_circuit_time_formula({'delay': time}))

# Subplot 1: T_L
ax[0].plot(times_all, par_L_span_all[:,0]*1e-3,
    linestyle='None', marker='o', label='All gates')
ax[0].plot(times_two_qubit, par_L_span_two_qubit[:,0]*1e-3,
    linestyle='None', marker='s', label='Two qb gates')
ax[0].plot(times_measure, par_L_span_measure[:,0]*1e-3,
    linestyle='None', marker='D', label='Measure')
ax[0].plot(times_feedback, par_L_span_feedback[:,0]*1e-3, 
    linestyle='None', marker='^', label='Feedback')
ax[0].plot(times_delay, par_L_span_delay[:,0]*1e-3,
    linestyle='None', marker='v', label='Delay')
#ax.errorbar(times_all, par_L_span_all[:,0]*1e-3, 
#               np.sqrt(cov_L_span_all[:,0,0])*1e-3,
#               linestyle='None',marker='o', label='All gates')
#ax.errorbar(times_two_qubit, par_L_span_two_qubit[:,0]*1e-3, 
#               np.sqrt(cov_L_span_two_qubit[:,0,0])*1e-3,
#               linestyle='None',marker='s', label='Two qb gates')
#ax.errorbar(times_measure, par_L_span_measure[:,0]*1e-3, 
#               np.sqrt(cov_L_span_measure[:,0,0])*1e-3,
#               linestyle='None',marker='D', label='Measure')
#ax.errorbar(times_feedback, par_L_span_feedback[:,0]*1e-3, 
#               np.sqrt(cov_L_span_feedback[:,0,0])*1e-3,
#               linestyle='None',marker='^', label='Feedback')
#ax.errorbar(times_delay[:16], par_L_span_delay[:16,0]*1e-3, 
#               np.sqrt(cov_L_span_delay[:16,0,0])*1e-3,
#               linestyle='None',marker='v', label='Delay')


trans_offset_up = mtransforms.offset_copy(ax[0].transData, fig=fig,
                                       x=.05, y=-0.15, units='inches')
trans_offset_down = mtransforms.offset_copy(ax[0].transData, fig=fig,
                                       x=.05, y=0.15, units='inches')
trans_offset_left = mtransforms.offset_copy(ax[0].transData, fig=fig,
                                       x=-.65, y=-0.1, units='inches')
trans_offset_delay = mtransforms.offset_copy(ax[0].transData, fig=fig,
                                       x=-.65, y=0.3, units='inches')
#ax[0].text(times_all[-1], par_L_span_all[-1,0]*1e-3, 
#           rf'$t_2 ={two_qubit_span[-1]*1e-3:.1f}$ μs',
#           color=colors_def(1), transform=trans_offset_up)
ax[0].text(times_two_qubit[-1], par_L_span_two_qubit[-1,0]*1e-3, 
           rf'$t_2 ={two_qubit_span[-1]*1e-3:.1f}$ μs',
           color='C1', transform=trans_offset_up)
ax[0].text(times_measure[-1], par_L_span_measure[-1,0]*1e-3,
           rf'$t_M ={measure_span[-1]*1e-3:.1f}$ μs',
           color='C2', transform=trans_offset_down)
ax[0].text(times_feedback[-1], par_L_span_feedback[-1,0]*1e-3,
           rf'$t_f ={feedback_span[-1]*1e-3:.0f}$ μs',
           color='C3', transform=trans_offset_left)
ax[0].text(times_delay[-1], par_L_span_delay[-1,0]*1e-3,
           rf'$t_d ={delay_span[-1]*1e-3:.0f}$ μs',
           color='C4', transform=trans_offset_delay)

#ax.set_title(r'Logical Lifetime over scaling gate times')
ax[0].set_xlabel(r'Total stabilizer cycle duration $[\mu s]$')
ax[0].set_ylabel(r'Logical lifetime $T_L$ $[\mu s]$')
ax[0].set(ylim=(20,42))

ax[0].legend(loc='upper left')
#%

# Subplot 2: P_L
P_L_all = np.sum(P_L_span_all[:,1:],1)/n_cycles
ax[1].plot(times_all, P_L_all, 'o', label='All gates')
P_L_two_qubit = np.sum(P_L_span_two_qubit[:,1:],1)/n_cycles
ax[1].plot(times_two_qubit, P_L_two_qubit, 's', label='Two qb gate')
P_L_measure = np.sum(P_L_span_measure[:,1:],1)/n_cycles
ax[1].plot(times_measure, P_L_measure, 'D', label='Measure')
P_L_feedback = np.sum(P_L_span_feedback[:,1:],1)/n_cycles
ax[1].plot(times_feedback, P_L_feedback, '^', label='Feedback')
P_L_delay = np.sum(P_L_span_delay[:,1:],1)/n_cycles
ax[1].plot(times_delay[:16], P_L_delay[:16], 'v', label='Delay')


colors_def = plt.get_cmap("tab10")




ax[1].set_xlabel(r'Total stabilizer cycle duration $[\mu s]$')
ax[1].set_ylabel(r'Prob. of remaining in code space $P_L$')

ax[1].set(ylim=(0.45,1.05))
ax[1].legend()


