'''
discrete time reachability


dynamics are:
x' = x + T*y - T*z  + 0.5*T*T*u1 - 0.5*T*T*u2
y' = y + T*u1
z' = z + T*u2
starting from [0.53, 10], [0.2, 0.6], [0.2, 0.6]
with u1 in [-0.46, 0.26]
u2 in [-0.47, 0.26]
'''

import math

import numpy as np
from scipy.sparse import csr_matrix

from hylaa.hybrid_automaton import HybridAutomaton
from hylaa.settings import HylaaSettings, PlotSettings
from hylaa.core import Core
from hylaa.stateset import StateSet
from hylaa import lputil

# Moved gurobi810 to bk in this directory '/usr/local/lib/python3.6/dist-packages
from gurobipy import *

T_const = 0.1


def define_ha():
    '''make the hybrid automaton'''

    ha = HybridAutomaton(discrete=True)

    a_matrix = [[1, T_const, -T_const], [0, 1, 0], [0, 0, 1]]
    a_matrix_inv = np.linalg.inv(a_matrix)

    # b_mat = [[0.5 * T_const * T_const, -0.5 * T_const * T_const], [T_const, 0], [0, T_const]]
    b_mat = [[0.5*T_const*T_const, -0.5*T_const*T_const, 1], [T_const, 0, 1], [0, T_const, 1]]
    a_inv_b_mat = np.matmul(a_matrix_inv, b_mat)

    # b_constraints = [[1, 0], [-1, 0], [0, 1], [0, -1]]
    # b_rhs = [0.26, 0.46, 0.26, 0.47]
    b_constraints = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
    b_rhs = [0.26, 0.46, 0.26, 0.47, 0.005, 0.005]
    mode = ha.new_mode('mode')
    mode.set_dynamics(a_matrix_inv)
    mode.set_inputs(a_inv_b_mat, b_constraints, b_rhs)

    # error = ha.new_mode('error')
    #
    # trans1 = ha.new_transition(mode, error)
    # trans1.set_guard([[-0, -1, -0], ], [-0.8, ])

    return ha

def make_init(ha):
    '''returns list of initial states'''

    mode = ha.modes['mode']
    init_lpi = lputil.from_box([[0.53, 10], [0.2, 0.6], [0.2, 0.6]], mode)

    init_list = [StateSet(init_lpi, mode)]

    return init_list


def define_settings():
    'get the hylaa settings object'

    step = 1
    max_time = 21
    settings = HylaaSettings(step, max_time)

    settings.stdout = stdout = HylaaSettings.STDOUT_VERBOSE

    plot_settings = settings.plot
    plot_settings.plot_mode = PlotSettings.PLOT_IMAGE
    plot_settings.xdim_dir = 0
    plot_settings.ydim_dir = 2

    # plot_settings.plot_mode = PlotSettings.PLOT_VIDEO
    # plot_settings.filename = 'tmpc.mp4'
    plot_settings.video_fps = 1
    plot_settings.video_extra_frames = 2  # extra frames at the end of a video so it doesn't end so abruptly
    plot_settings.video_pause_frames = 2  # frames to render in video whenever a 'pause' occurs
    
    plot_settings.label.y_label = '$z$'
    plot_settings.label.x_label = '$x$'
    plot_settings.label.title = 'TMPC mode 1 X vs z'

    return settings


def run_hylaa():
    'Runs hylaa with the given settings'

    ha = define_ha()
    settings = define_settings()

    tuples = []
    # tuples.append((HylaaSettings.APPROX_NONE, "tmpc1_x_z.mp4"))
    tuples.append((HylaaSettings.APPROX_NONE, "tmpc1_x_z.png"))
    # tuples.append((HylaaSettings.APPROX_CHULL, "tmpc_chull1.png"))
    # tuples.append((HylaaSettings.APPROX_LGG, "approx_lgg.png"))

    for model, filename in tuples: 
        settings.approx_model, settings.plot.filename = model, filename

        init_states = make_init(ha)
        print(f"\nMaking {filename}...")
        Core(ha, settings).run(init_states)


if __name__ == '__main__':
    run_hylaa()
