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

    ha = HybridAutomaton(discrete=True)

    a_matrix = np.array([[0.722468865032875, -0.523371053120237, 0], [0.785056579680355, 0.696300312376864, 0],
                         [0, 0, 0.930530895811206]], dtype=float)
    a_csr = csr_matrix(a_matrix, dtype=float)
    mode = ha.new_mode('mode')
    mode.set_dynamics(a_csr)

    # b_mat = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    # b_constraints = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
    # b_rhs = [0.0, -0.0, 0.1, -0.1, 0.03, -0.03]

    b_mat = [[0, 0], [1, 0], [0, 1]]
    b_constraints = [[1, 0], [-1, 0], [0, 1], [0, -1]]
    b_rhs = [0.1, -0.1, 0.1, -0.1]

    mode.set_inputs(b_mat, b_constraints, b_rhs, allow_constants=True)

    error = ha.new_mode('error')

    trans = ha.new_transition(mode, error)
    # y >= 0.4
    trans.set_guard([[0, -1, 0], ], [-0.5, ])

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


def run_hylaa():
    'Runs hylaa with the given settings'

    ha = define_ha()
    settings = define_settings()
    init_states = make_init(ha)
    core = Core(ha, settings)
    core.run(init_states)
    error_states = core.get_error_stars()  # error states are of type StarData
    print(" no of error states: " + str(len(error_states)))

    usafeset_preds = core.get_errorset_preds()

    Timers.tic("BDD Construction")
    process_stars(error_states)

    bdd_ce_object = BDD4CE(error_states, usafeset_preds, smt_mip='mip')
    bdd_graphs = bdd_ce_object.create_bdd_w_level_merge(level_merge=0, order='mid-order')
    valid_exps, invalid_exps = bdd_graphs[0].generate_expressions()
    print(len(valid_exps), len(invalid_exps))
    # print(valid_exps)
    Timers.toc("BDD Construction")
    Timers.print_stats()


if __name__ == '__main__':
    run_hylaa()
