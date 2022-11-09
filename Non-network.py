#!/usr/bin/env python
# coding: utf-8

# In this file, a non-network is proposed that can be used for investigating the effects of Total Ionizing Dose on a
# neuromorphic circuit (representative of an Innatera chip).
import numpy as np
from random import randrange
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from brian2 import *
import pickle
import os

def visualize_connectivity(S):
    Ns = len(S.source)
    Nt = len(S.target)
    figure(0, figsize=(10, 4))
    subplot(121)
    plot(zeros(Ns), arange(Ns), 'ok', ms=10)
    plot(ones(Nt), arange(Nt), 'ok', ms=10)
    for i, j in zip(S.i, S.j):
        plot([0, 1], [i, j], '-k')
    xticks([0, 1], ['Source', 'Target'])
    ylabel('Neuron index')
    xlim(-0.1, 1.1)
    ylim(-1, max(Ns, Nt))
    subplot(122)
    scatter(S.i, S.j, S.w * 10)
    xlim(-1, Ns)
    ylim(-1, Nt)
    xlabel('Source neuron index')
    ylabel('Target neuron index')


if __name__ == "__main__":

    start_scope()

    time = 1  # Time of circuit run in seconds
    duration = time * second

    n = 64  # Number of neurons on chip (representative of Innatera T0 chip)
    Poissonrate = 100 * Hz  # The poisson rate of input spikes

    I = 5 * mamp  # Input current to each individual neuron
    R = 5 * ohm  # The resistance of the neuron (unknown value in this setup)

    # The equations governing the leaky-integrate-and-fire neurons
    eqs = """
    tau : second
    mismatch : second
    dv/dt = (R*I-v)/(tau+mismatch): volt (unless refractory)
    """

    # Input spikes following Poisson distribution for each individual neuron to simulate (radiation) noise
    P = PoissonGroup(n, rates=Poissonrate)

    # The neurons with threshold voltage, reset voltage and refractoriness
    N = NeuronGroup(n, eqs, threshold='v > 20*mvolt', reset='v = 0*mvolt', refractory=5 * ms, method='rk4')
    N.tau = '10*msecond'  # Time constant (equal for each neuron)
    N.mismatch = 'rand()*msecond'  # Mismatch factor to simulate device mismatch of analog devices

    # The synapses connecting one input Poisson neuron to an actual physical neuron (all other connections are set to zero)
    S = Synapses(P, N, 'w:volt', on_pre='v_post += w')
    S.connect(condition='i==j')
    S.w = '2*mvolt'  # Weight of the connection (equal for each neuron)
    S.delay = '1*msecond'  # Synaptic delay (equal for each neuron)

    # # Can be used to visualize the connectivity of the neurons (not useful for large neuron groups)
    # visualize_connectivity(S)

    SpikeMon_N = SpikeMonitor(N)
    StateMon_N = StateMonitor(N, 'v', record=True)

    random_neuron = randrange(n)  # Select random neuron for plotting purposes

    run(duration, profile=True)

    # plt.figure(1)
    # plt.plot(StateMon_N.t / ms, 1000 * StateMon_N.v[random_neuron])
    # plt.xlabel('Time (ms)')
    # plt.ylabel('Membrane voltage (mV)')
    # plt.xlim(0, 100)
    # plt.title('Membrane voltage for neuron ' + str(random_neuron));
    #
    # plt.figure(2)
    # plt.plot(SpikeMon_N.t / ms, SpikeMon_N.i, '.k')
    # plt.xlabel('Time (ms)')
    # plt.ylabel('Neuron index (-)')
    # plt.ylim([random_neuron - 5, random_neuron + 5])
    # plt.title('Output spikes per neuron over time')
    #
    # spike_freq = []
    # for k in range(n):
    #     spike_freq.append(np.mean(np.diff(SpikeMon_N.spike_trains()[k])) / ms)
    #
    # y = spike_freq
    #
    # plt.fig = plt.figure(3, figsize=(10, 6))
    # gs = gridspec.GridSpec(nrows=1, ncols=4)
    # ax_main = plt.subplot(gs[0, 0:3])
    # plt.title('Average inter-spike-interval per neuron')
    # ax_yDist = plt.subplot(gs[0, 3], sharey=ax_main)
    #
    # ax_main.plot(N.i, spike_freq)
    # ax_main.set(xlabel='Neuron index', ylabel='Inter-spike-interval (ms)')
    #
    # ax_yDist.hist(y, bins=100, orientation='horizontal', align='mid')
    # ax_yDist.set(xlabel='count')
    # ax_yDist.tick_params(direction='in', labelleft=False)
    #
    # plt.show()

    print(profiling_summary(show=5))  # show the 5 objects that took the longest

    # Write data frame to disk as a CSV file:
    data = SpikeMon_N.get_states(['i', 't'], units=False, format='pandas')
    date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print(os.getcwd())
    save_path = "./Non_network_trials"
    file_name = f"spikes_{date}.csv"
    completeName = os.path.join(save_path, file_name)
    data.to_csv(completeName, index=False)
