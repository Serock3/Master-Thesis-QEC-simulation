#%%
import matplotlib.pyplot as plt
import numpy as np                                  

from qiskit.quantum_info import partial_trace
from qiskit.quantum_info import DensityMatrix
from qiskit.quantum_info import state_fidelity

# Our own files
from simulator_program.custom_noise_models import (thermal_relaxation_model,
    thermal_relaxation_model_V2,
    WACQT_target_times,
    WACQT_demonstrated_times,
    standard_times)
from simulator_program.data_analysis_tools import *


#% TEST encoding fidelities
#fid, circ = encoding_fidelity(1, gate_times=WACQT_target_times, T1=40e3, T2=60e3,
#        reset=True, idle_noise=True, theta=np.pi, phi=0,
#        snapshot_type='dm', device='Nah', pauliop='ZZZZZ')
#print(fid)

T1 = 40e3
def monoExp(t, T, c, A):
    return (A-c)* np.exp(-t/T) + c
p0 = (T1, 0.4, 0.9) # start with values near those we expect*
colors = plt.get_cmap("tab10")
#%% Test run
n_curves = 4
n_cycles = 8
n_shots=1024*2

fids = []
times = []
t_fs = np.linspace(0,1000,n_curves)
for t_f in t_fs:
    fid, time = fidelity_from_scratch(n_cycles,n_shots,gate_times={'feedback':t_f},encoding=False,transpile=False)
    time = np.array([time['dm_'+str(i)] for i in range(n_cycles+1)])
    fids.append(fid)
    times.append(time)
    plt.plot(time/1000,fid, 'o',label = str(t_f))

    pars, cov = scipy.optimize.curve_fit(monoExp, time[1:], fid[1:], p0)
    print('t_f= ', t_fs[i])
    print('T=',np.round(pars[0]/1000,1))
    print('c=',np.round(pars[1],2))
    print('A=',np.round(pars[2],2),'\n')
    plt.plot(time/1000, monoExp(time,*pars), '--')
    plt.legend()
    plt.show()

#%%
for i,(fid,time) in enumerate(zip(fids,times)):
    plt.plot(time/1000,fid, 'o',color = colors(i),label = str(t_fs[i]))

    pars, cov = scipy.optimize.curve_fit(monoExp, time[1:], fid[1:], p0)
    print('t_f= ', t_fs[i])
    print('T=',np.round(pars[0]/1000,1))
    print('c=',np.round(pars[1],2))
    print('A=',np.round(pars[2],2),'\n')
    plt.plot(time/1000, monoExp(time,*pars),'--', color = colors(i))

plt.legend()
plt.show()

# %%
