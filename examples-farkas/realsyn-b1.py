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
from farkas_central.control_utils import get_input


def define_ha():
    '''make the hybrid automaton'''

    a_matrix = np.array([[0.0, 2.0], [-1.5, 0.0]], dtype=float)

    b_matrix = np.array([[1], [-1]], dtype=float)

    print(" ****** define ha start ****** ")
    print(a_matrix, b_matrix)
    R_mult_factor = 0.2

    Q_matrix = np.eye(len(a_matrix[0]), dtype=float)

    u_dim = len(b_matrix[0])
    R_matrix = R_mult_factor * np.eye(u_dim)

    print(a_matrix, b_matrix, Q_matrix, R_matrix)
    k_matrix = get_input(a_matrix, b_matrix, Q_matrix, R_matrix)

    print(k_matrix)
    a_bk_matrix = np.array(a_matrix - np.matmul(b_matrix, k_matrix), dtype=float)
    print(a_bk_matrix)
    ha = HybridAutomaton(discrete=False)

    a_matrix = a_bk_matrix
    a_csr = csr_matrix(a_matrix, dtype=float)

    mode = ha.new_mode('mode')
    mode.set_dynamics(a_csr)

    # b_mat = [[1, 0], [0, 1]]
    # b_constraints = [[1, 0], [-1, 0], [0, 1], [0, -1]]
    # b_rhs = [0.1, 0.1, 0.1, 0.1]

    b_mat = [[1], [1]]
    b_constraints = [[1], [-1]]
    b_rhs = [0.1, 0.1]

    mode.set_inputs(b_mat, b_constraints, b_rhs, allow_constants=False)

    error = ha.new_mode('error')

    trans = ha.new_transition(mode, error)
    # x >= 2.0
    trans.set_guard([[-1, 0], ], [-2.1, ])

    print(" ****** define ha end ****** ")

    return ha


def make_init(ha):
    '''returns list of initial states'''

    mode = ha.modes['mode']
    init_lpi = lputil.from_box([[1.0, 1.5], [1.0, 1.5]], mode)

    init_list = [StateSet(init_lpi, mode)]

    return init_list


def define_settings():
    'get the hylaa settings object'

    step = 0.02
    max_time = 2.0
    settings = HylaaSettings(step, max_time)

    plot_settings = settings.plot
    plot_settings.plot_mode = PlotSettings.PLOT_IMAGE
    plot_settings.xdim_dir = 0
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

    bdd_ce_object = BDD4CE(error_states, usafeset_preds, equ_run=True, smt_mip='mip')
    # #
    bdd_graphs = bdd_ce_object.create_bdd_w_level_merge(level_merge=0, order='random')
    valid_exps, invalid_exps = bdd_graphs[0].generate_expressions()
    print(len(valid_exps), len(invalid_exps))
    Timers.toc("BDD Construction")
    Timers.print_stats()


if __name__ == '__main__':
    run_hylaa()
