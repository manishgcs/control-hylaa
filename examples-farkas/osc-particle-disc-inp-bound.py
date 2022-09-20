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
import time


def define_ha(inp=0):
    '''make the hybrid automaton'''

    ha = HybridAutomaton(discrete=True)

    a_matrix = np.array([[0.722468865032875, -0.523371053120237, 0], [0.785056579680355, 0.696300312376864, 0],
                         [0, 0, 0.930530895811206]], dtype=float)
    a_csr = csr_matrix(a_matrix, dtype=float)
    mode = ha.new_mode('mode')
    mode.set_dynamics(a_csr)

    if inp > 0:
        if inp == 1:
            b_mat = [[0], [0.1], [0.1]]
            b_constraints = [[1], [-1]]
            b_rhs = [0.04, 0.04]
        else:
            b_mat = [[0.1, 0.1], [0.1, 0.1], [0.01, 0.1]]
            b_constraints = [[1, 0], [-1, 0], [0, 1], [0, -1]]
            b_rhs = [0.04, 0.04, 0.04, 0.04]

        mode.set_inputs(b_mat, b_constraints, b_rhs)

    error = ha.new_mode('error')

    trans = ha.new_transition(mode, error)
    # y >= 0.4
    trans.set_guard([[0, -1, 0], ], [-0.4, ])

    return ha


def make_init(ha):
    '''returns list of initial states'''

    mode = ha.modes['mode']

    init_lpi = lputil.from_box([[-0.1, 0.1], [-0.8, -0.4], [-1.07, -1]], mode)

    init_list = [StateSet(init_lpi, mode)]

    return init_list


def define_settings():
    'get the hylaa settings object'

    step = 0.6
    max_time = 9
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
    plot_settings.label.x_label = '$x$'
    plot_settings.label.title = 'Oscillating Particle'

    return settings


def run_hylaa(order=None, level_merge=None, bdd_f=None, inp=None):
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
    # print(" no of error states: " + str(len(error_states)))
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
    # process_stars(to_process_stars)
    # # Note. solution is a 16 element vector because our last error star is at 13th step. We had to add two (some random
    # # values in the set) to the end to accommodate for stars at 14 and 15 steps to get a full simulation.
    # solution_1 = np.array([0.1, -0.6592294225, -1, 0.04, 0.04, 0, 0.04, 0.04, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    # traj = []
    # id_matrix = np.eye(3, dtype=float)
    # placeholder_matrix = np.zeros((3, 15), dtype=float)
    # id_matrix = np.concatenate((id_matrix, placeholder_matrix), axis=1)
    # traj.append(np.dot(id_matrix, solution_1))  # For the point in initial star
    # for state in to_process_stars:
    #     traj.append(np.dot(state.basis_matrix, solution_1))
    # for state in traj:
    #     print(str(state[1]) + ", " + str(state[2]) + ";")
    # soln_f = open("/home/manishg/Research/Conferences/IEEE-TAC/solutions-w-inp.txt", "r")
    ##

    usafeset_preds = core.get_errorset_preds()

    Timers.tic("BDD Construction")
    start_time = time.time()
    process_stars(error_states)

    bdd_ce_object = BDD4CE(error_states, usafeset_preds, smt_mip='mip')
    # #
    bdd_graphs = bdd_ce_object.create_bdd_w_level_merge(level_merge=level_merge, order=order, bdd_f=bdd_f)
    valid_exps, invalid_exps = bdd_graphs[0].generate_expressions()
    print(len(valid_exps), len(invalid_exps))
    Timers.toc("BDD Construction")
    Timers.print_stats()
    bdd_f.write("\nUnique states:" + str(len(valid_exps)))
    bdd_f.write("\nTime taken: " + str(time.time() - start_time))


if __name__ == '__main__':
    level_merges = [-1, 0]
    orders = ['default', 'mid-order', 'random']
    inputs = [0, 1, 2]

    for inp in inputs:
        bdd_f = open("realsyn-b2-bdd-" + str(inp) + ".txt", "a+")
        for order in orders:
            for level in level_merges:
                bdd_f.write("\ninput: " + str(inp))
                bdd_f.write("\norder: " + str(order))
                bdd_f.write("\nlevel: " + str(level))
                run_hylaa(order, level, bdd_f, inp)
        bdd_f.close()
