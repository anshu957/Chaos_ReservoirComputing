'''
File: rossler.py
File Created: Sunday, 7th July 2019 10:25:02
Author: well-well-well
-----------------------------------------------------------
Description: This script trains the reservoir for chaotic
             Rossler oscillator
------------------------------------------------------------
'''

import numpy as np
from jitcode import jitcode, y
import argparse
import os
import sys

# adding current directory path to the system's path
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(THIS_DIR)

import Reservoir as res
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--res_size', default=500, type=int,
                        help='Size of the reservoir')
    parser.add_argument('--leaky_rate', default=0.7, type=float,
                        help='leaky rate for the reservoir')
    parser.add_argument('--spectral_radius', default=0.8, type=float,
                        help='spectral radius of the reservoir')
    parser.add_argument('--res_density', default=0.4, type=float,
                        help='reservoir density')
    parser.add_argument('--input_density', default=0.4, type=float,
                        help='input-to-reservoir connection density')
    parser.add_argument('--inputScaling_radius', default=0.02, type=float,
                        help='standard variation of input weights')
    parser.add_argument('--random_seed', default=None, type=int,
                        help='random seed initialization for repeatability')
    parser.add_argument('--save_dir', default=THIS_DIR, type=str,
                        help='where to save the trained data or figures')
    parser.add_argument('--train_data_ratio', default=0.6, type=float,
                        help='what percentage of training time is needed')
    parser.add_argument('--trans_ratio', default=0.1, type=float,
                        help='transience time fraction of total time')
    parser.add_argument('--beta', default=1e-07, type=float,
                        help='regularization coefficient')

    parser.set_defaults(feature=True)

    return parser.parse_args()


def getData(train_data_ratio, trans_ratio):
    '''
        Integrates the rossler system, records the data into
        train_data, test_data based on train_data_ratio (train_data_length/total length of data)
        trans_ratio = (transience length/ total length of data)

        Returns: (train_data, test_data)
    '''
    # Rossler equation for motion
    f = [
        -y(1) - y(2),
        y(0) + 0.2 * y(1),
        0.2 + (y(0) - 7.0) * y(2)
    ]

    inital_state = 2.0 * (1 - 2 * np.random.random(3))
    ODE = jitcode(f)
    ODE.set_integrator("dopri5")
    ODE.set_initial_value(inital_state, 0.0)

    data = []
    times = np.arange(1000, 10000, 0.1)
    for time in times:
        data.append(ODE.integrate(time))

    data_arr = np.array(data)

    train_len = int(train_data_ratio * len(data_arr))
    trans_len = int(trans_ratio * len(data_arr))

    train_data = data_arr[0:train_len, :]
    test_data = data_arr[train_len + 1:, :]

    return train_data, test_data, trans_len


if __name__ == "__main__":

    args = get_args()  # getting command line arguments if default is not suitable

    input_size = 3

    rc_rossler = res.Reservoir(args.res_size, input_size,
                               args.spectral_radius, args.leaky_rate,
                               args.res_density, args.inputScaling_radius,
                               args.input_density, args.random_seed, args.beta)

    # getting data
    train_data, test_data, trans_len = getData(
        args.train_data_ratio, args.trans_ratio)

    # Train the reservoir
    rc_rossler.train(train_data, trans_len)

    print("Generating future trajectories and plotting them ....")
    # letting the reservoir run on "self generative" mode
    res_states, outputs = rc_rossler.generateFutureTrajectories(test_data)

    # saving the data
    os.makedirs(args.save_dir) if not os.path.exists(args.save_dir) else None
    label1 = 'trajectories'
    path1 = '{}/{}.dat'.format(args.save_dir, label1)

    label2 = 'nodes_states'
    path2 = '{}/{}.dat'.format(args.save_dir, label2)

    np.savetxt(path1, outputs, delimiter='\t')
    np.savetxt(path2, res_states, delimiter='\t')

    # plotting the data
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.set_title(' Comparing trajectories')
    ax.plot(test_data[:1000, 1], '-b', label=' Original Trajectory')
    ax.plot(outputs[:1000, 1], '-r', label='Reservoir generated')
    ax.set_xlabel('Time')
    ax.set_ylabel('Y')
    plt.show()

    fig1 = plt.figure(2)
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.set_title('Learned Attarctor')
    ax1.plot(outputs[:, 0], outputs[:, 1], outputs[:, 2])
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    plt.show()
