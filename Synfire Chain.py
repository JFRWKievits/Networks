#!/usr/bin/env python
# coding: utf-8
import numpy as np
from datetime import datetime
from brian2 import *
import matplotlib
from random import randrange
from random import gauss
import pandas as pd
import os

def gaussian(mu, sigma, l):
    dist = [gauss(mu, sigma) for i in range(l)]
    return dist # returns a Gaussian distribution of length l

if __name__ == "__main__":
    start_scope()

    # Simulation parameters
    duration = 100*ms
    noise_rate = 100*Hz

    # Neuron parameters
    Vr_ideal = 0*mV # Ideal reset voltage
    Vt_ideal = 70*mV # Ideal threshold voltage
    t_ref = 20*ms # Ideal refractoriness
    taum_ideal = 10*ms # Ideal membrane time constant
    taupsp_ideal = 1.7*ms # Ideal PSP time constant

    f = 0.3 # Non_ideality factor for analog neurons

    # Synapse parameters
    weight_ideal = 70*mV # Ideal synaptic weight
    s_delay_ideal = 5*msecond # Ideal synaptic delay

    # Model parameters
    n_groups = 5 # Number of groups
    group_size = 20 # Number of neurons per group

    # Initial spike volley parameters
    t0 = 10*ms # Mean spike timing
    a = 1 # Activity of spikes in first pulse packet (has to be 1 for full activation)
    t_disp = 2*ms # Temporal dispersion of spikes in first pulse packet
    p_connect = 0.9 # Fraction of activated neurons in first group

    # Neuron model
    eqs = Equations('''
    taum : second
    taupsp : second
    Vt : volt
    Vr : volt
    dV/dt = (-(V-Vr)+x)*(1./(taum)) : volt (unless refractory)
    dx/dt = (-x+y)*(1./taupsp) : volt
    dy/dt = -y*(1./taupsp) : volt
    ''')

    # Neuron Groups (Input (I), Neurons (N), Noise (G))
    I = SpikeGeneratorGroup(int(a*group_size), np.arange(int(a*group_size)), gaussian(t0, t_disp, int(a*group_size)))

    N = NeuronGroup(N=n_groups*group_size, model=eqs,
                    threshold='V>Vt', reset='V=Vr', refractory=t_ref,
                    method='rk4')
    N.Vr = gaussian(Vr_ideal, Vr_ideal*f, len(N))
    N.Vt = gaussian(Vt_ideal, Vt_ideal*f, len(N))
    N.taum = gaussian(taum_ideal, taum_ideal*f, len(N))
    N.taupsp = gaussian(taupsp_ideal, taupsp_ideal*f, len(N))

    G = PoissonGroup(n_groups*group_size, rates=noise_rate)

    # Synapses
    S = Synapses(N, N,'weight:volt', on_pre='y+=weight')
    S.connect(j='k for k in range((int(i/group_size)+1)*group_size, (int(i/group_size)+2)*group_size)'
                'if i<N_pre-group_size')
    S.weight = gaussian(weight_ideal, weight_ideal*f, len(S))
    S.delay = gaussian(s_delay_ideal, s_delay_ideal*f, len(S))

    Sinput = Synapses(I, N[:group_size], 'weight:volt', on_pre='y += weight')
    Sinput.connect(p=p_connect)
    Sinput.weight = gaussian(weight_ideal, weight_ideal*f, len(Sinput))

    S_G = Synapses(G, N, 'weight:volt', on_pre='V += weight')
    S_G.connect('i==j')
    S_G.weight = gaussian(0*mV, 50*mV, len(S_G))

    # Record the spikes
    SpikeMon_N = SpikeMonitor(N)
    SpikeMon_I = SpikeMonitor(I)
    StateMon = StateMonitor(N, 'V', record=True)

    random_neuron = randrange(group_size) # Select random neuron for plotting purposes

    # Run the network
    run(duration, profile=True)

    figure(1)
    plot(StateMon.t/ms, 1000*StateMon.V[random_neuron])
    xlabel('Time (ms)')
    ylabel('Membrane voltage (mV)')
    xlim(0,100)
    title('Membrane voltage for neuron '+str(random_neuron))

    figure(2)
    plot(SpikeMon_N.t/ms, 1.0*SpikeMon_N.i/group_size+1, '.')
    plot(SpikeMon_I.t/ms, 1.0*SpikeMon_I.i/int(a*group_size), '.')
    plot([0, duration/ms], np.arange(n_groups+1).repeat(2).reshape(-1, 2).T, 'k:')
    ylabel('group number')
    yticks(np.arange(n_groups+1))
    xlabel('time (ms)')
    show()

    print(profiling_summary(show=5))  # show the 5 objects that took the longest

    # Write data frame to disk as a CSV file:
    data = SpikeMon_N.get_states(['i', 't'], units=False, format='pandas')
    date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print(os.getcwd())
    save_path = "./Synfire_chain_trials"
    file_name = f"spikes_{date}.csv"
    completeName = os.path.join(save_path, file_name)
    data.to_csv(completeName, index=False)
