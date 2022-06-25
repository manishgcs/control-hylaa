import math

import numpy as np
from scipy.sparse import csr_matrix

from hylaa.hybrid_automaton import HybridAutomaton
from hylaa.settings import HylaaSettings, PlotSettings
from hylaa.core import Core
from hylaa.stateset import StateSet
from hylaa import lputil
from farkas_central.process_stars import process_stars
from farkas_central.bdd4Ce import BDD4CE
from hylaa.timerutil import Timers


def define_ha():
    '''make the hybrid automaton'''

    ha = HybridAutomaton(discrete=False)

    a_matrix = np.array([[-0.05, -1, 0], [1.5, -0.1, 0], [0, 0, -0.12]], dtype=float)
    a_csr = csr_matrix(a_matrix, dtype=float)

    mode = ha.new_mode('mode')
    mode.set_dynamics(a_csr)
    # mode.set_inputs(b_mat, b_constraints, b_rhs, allow_constants=True)

    error = ha.new_mode('error')

    trans = ha.new_transition(mode, error)
    # y >= 0.4
    trans.set_guard([[0, -1, 0], ], [-0.4, ])

    return ha


def make_init(ha):
    '''returns list of initial states'''

    mode = ha.modes['mode']

    init_lpi = lputil.from_box([[-0.1, 0.1], [-0.8, -0.4], [-1.07, -1]], mode)

    # init_lpi = lputil.from_box([[-0.1, 0.1], [-1.1, -0.5], [1, 1.1]], mode)

    init_list = [StateSet(init_lpi, mode)]

    return init_list


def define_settings():
    'get the hylaa settings object'

    step = 0.6
    max_time = 9.0
    settings = HylaaSettings(step, max_time)

    plot_settings = settings.plot
    plot_settings.plot_mode = PlotSettings.PLOT_IMAGE
    plot_settings.xdim_dir = 2
    plot_settings.ydim_dir = 1

    # plot_settings.plot_mode = PlotSettings.PLOT_VIDEO
    # plot_settings.filename = 'ha.mp4'
    # plot_settings.video_fps = 2
    # plot_settings.video_extra_frames = 10 # extra frames at the end of a video so it doesn't end so abruptly
    # plot_settings.video_pause_frames = 5 # frames to render in video whenever a 'pause' occurs

    plot_settings.label.y_label = '$y$'
    plot_settings.label.x_label = '$z$'
    plot_settings.label.title = 'Oscillating Particle'

    return settings


def run_hylaa():
    'Runs hylaa with the given settings'

    ha = define_ha()
    settings = define_settings()
    init_states = make_init(ha)
    core = Core(ha, settings)
    core.run(init_states)
    error_states = core.get_error_stars()  # error states are of type StarData
    print(" no of error states: " + str(len(error_states)))

    # Just to get the simulations for each characterization
    # non_error_states = core.get_non_error_stars()
    # to_process_stars = []
    # for state in non_error_states:
    #     to_process_stars.append(state)
    # for state in error_states:
    #     to_process_stars.append(state)
    # # print(len(to_process_stars))
    # for idx in range(len(to_process_stars)-1):
    #     for idy in range(idx+1, len(to_process_stars)):
    #         if to_process_stars[idx].step[0] > to_process_stars[idy].step[0]:
    #             s1 = to_process_stars[idx]
    #             to_process_stars[idx] = to_process_stars[idy]
    #             to_process_stars[idy] = s1
    # print(len(to_process_stars))
    #
    # solution_1 = np.array([0.029476484958649962, -0.7999999999992724, -1.0])
    # solution_2 = np.array([0.1000000000003638, -0.7406622401649656, -1.0])
    # solution_3 = np.array([0.09943471519000013, -0.7411358029839903, -1.01])
    # solution_4 = np.array([0.0698596917, -0.6392583483, -1.0])   # SMT
    # solution_5 = np.array([0.1000000000003638, -0.5903276515346079, -1.0])
    # solution_6 = np.array([-0.1000000000003638, -0.7100485767150531, -1.0])
    # solution_7 = np.array([-0.1000000000003638, -0.5153278693142056, -1.0])
    # solution_8 = np.array([0.10, -0.4617778016, -1.0])  # SMT
    # solution_9 = np.array([-0.1000000000003638, -0.3999999999996362, -1.0])
    #
    # traj = []
    # traj.append(np.dot(np.eye(3), solution_3))  # For the point in initial star
    # for state in to_process_stars:
    #     traj.append(np.dot(state.basis_matrix, solution_3))
    # for state in traj:
    #     print(str(state[1]) + ", " + str(state[2]) + ";")

    usafeset_preds = core.get_errorset_preds()

    Timers.tic("BDD Construction")
    process_stars(error_states)

    bdd_ce_object = BDD4CE(error_states, usafeset_preds, equ_run=True, smt_mip='mip')
    bdd_graphs = bdd_ce_object.create_bdd_w_level_merge(level_merge=0, order='default')
    valid_exps, invalid_exps = bdd_graphs[0].generate_expressions()
    print(len(valid_exps), len(invalid_exps))
    # print(valid_exps)
    Timers.toc("BDD Construction")
    Timers.print_stats()


if __name__ == '__main__':
    run_hylaa()
