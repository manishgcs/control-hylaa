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

    ha = HybridAutomaton(discrete=False)

    a_matrix = np.array([
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1.702423734, 3.929356551, -4.3607983776, -1.01374489, -1.6167727749, 0.2653009364, -0.2375199245,
         -0.4793543458, 0.06412815, -0.1079326841, -0.2463610381, 0.0276872161, -0.0605561959, -0.1501445039,
         0.0151944922, -0.0374830081, -0.0986391305, 0.009628751, -0.0242136837, -0.0665592904, 0.0067836913,
         -0.015601062, -0.0442510048, 0.0052325207, 0.0093924696, 0.0272127915, 0.0043984935, -0.0044278796,
         -0.0129879863, 0.0040303349],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0.688678844, 2.3125837761, 0.2653009364, 1.4649038095, 3.4500022052, -4.2966702275, -1.1216775741,
         -1.863133813, 0.2929881525, -0.2980761204, -0.6294988497, 0.0793226422, -0.1454156921, -0.3450001686,
         0.0373159671, -0.0847698796, -0.2167037943, 0.0219781835, -0.0530840701, -0.1428901352, 0.0148612718,
         -0.0336061533, -0.0937720819, 0.0111821848, -0.0200289416, -0.057238991, 0.0092628557, -0.0093924696,
         -0.0272127915, 0.0084288284],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0.4511589195, 1.8332294303, 0.06412815, 0.5807461599, 2.066222738, 0.2929881525, 1.4043476136, 3.2998577013,
         -4.2814757354, -1.1591605822, -1.9617729435, 0.3026169036, -0.3222898041, -0.6960581401, 0.0861063336,
         -0.1610167541, -0.3892511733, 0.0425484878, -0.0941623492, -0.2439165858, 0.026376677, -0.0575119497,
         -0.1558781215, 0.0188916067, -0.0336061533, -0.0937720819, 0.0152125197, -0.015601062, -0.0442510048,
         0.0136613491],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0.3432262354, 1.5868683922, 0.0276872161, 0.3906027236, 1.6830849264, 0.0793226422, 0.5432631518, 1.9675836075,
         0.3026169036, 1.3801339299, 3.2332984109, -4.274692044, -1.1747616442, -2.0060239482, 0.3078494243,
         -0.3316822737, -0.7232709316, 0.090504827, -0.1654446337, -0.4022391596, 0.0465788228, -0.0941623492,
         -0.2439165858, 0.0304070119, -0.0530840701, -0.1428901352, 0.0232901001, -0.0242136837, -0.0665592904,
         0.0204450405],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0.2826700395, 1.4367238883, 0.0151944922, 0.3057432273, 1.4882292617, 0.0373159671, 0.3663890398, 1.616525636,
         0.0861063336, 0.5276620899, 1.9233326028, 0.3078494243, 1.3707414603, 3.2060856194, -4.2702935506,
         -1.1791895238, -2.0190119345, 0.3118797592, -0.3316822737, -0.7232709316, 0.094535162, -0.1610167541,
         -0.3892511733, 0.0509773162, -0.0847698796, -0.2167037943, 0.0356395326, -0.0374830081, -0.0986391305,
         0.0300737915],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0.2451870315, 1.3380847578, 0.009628751, 0.2584563558, 1.3701645979, 0.0219781835, 0.2901421653, 1.443978257,
         0.0425484878, 0.3569965702, 1.5893128445, 0.090504827, 0.5232342102, 1.9103446165, 0.3118797592, 1.3707414603,
         3.2060856194, -4.2662632156, -1.1747616442, -2.0060239482, 0.3162782527, -0.3222898041, -0.6960581401,
         0.0997676827, -0.1454156921, -0.3450001686, 0.0577610076, -0.0605561959, -0.1501445039, 0.0452682837],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0.2209733477, 1.2715254674, 0.0067836913, 0.2295859695, 1.2938337531, 0.0148612718, 0.2490638862, 1.3429518064,
         0.026376677, 0.2857142857, 1.4309902707, 0.0465788228, 0.3569965702, 1.5893128445, 0.094535162, 0.5276620899,
         1.9233326028, 0.3162782527, 1.3801339299, 3.2332984109, -4.2610306949, -1.1591605822, -1.9617729435,
         0.323061944, -0.2980761204, -0.6294988497, 0.1093964337, 0.1079326841, -0.2463610381, 0.0729554998],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0],
        [0.2053722857, 1.2272744627, 0.0052325207, 0.2115808781, 1.2443126759, 0.0111821848, 0.2251580898, 1.2808457668,
         0.0188916067, 0.2490638862, 1.3429518064, 0.0304070119, 0.2901421653, 1.443978257, 0.0509773162, 0.3663890398,
         1.616525636, 0.0997676827, 0.5432631518, 1.9675836075, 0.323061944, 1.4043476136, 3.2998577013, -4.2514019439,
         -1.1216775741, -1.863133813, 0.3382564362, -0.2375199245, -0.4793543458, 0.1370836498],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0],
        [0.1959798161, 1.2000616712, 0.0043984935, 0.2009444061, 1.2142864764, 0.0092628557, 0.2115808781, 1.2443126759,
         0.0152125197, 0.2295859695, 1.2938337531, 0.0232901001, 0.2584563558, 1.3701645979, 0.0356395326, 0.3057432273,
         1.4882292617, 0.0577610076, 0.3906027236, 1.6830849264, 0.1093964337, 0.5807461599, 2.066222738, 0.3382564362,
         1.4649038095, 3.4500022052, -4.2237147278, -1.01374489, -1.6167727749, 0.4023845862],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, -1],
        [0.1915519365, 1.1870736849, 0.0040303349, 0.1959798161, 1.2000616712, 0.0084288284, 0.2053722857, 1.2272744627,
         0.0136613491, 0.2209733477, 1.2715254674, 0.0204450405, 0.2451870315, 1.3380847578, 0.0300737915, 0.2826700395,
         1.4367238883, 0.0452682837, 0.3432262354, 1.5868683922, 0.0729554998, 0.4511589195, 1.8332294303, 0.1370836498,
         0.688678844, 2.3125837761, 0.4023845862, 1.702423734, 3.929356551, -3.9584137913]
    ], dtype=float)

    a_csr = csr_matrix(a_matrix, dtype=float)

    mode = ha.new_mode('mode')
    mode.set_dynamics(a_csr)

    # b_mat = [[1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1]]
    # b_constraints = [[1], [-1]]
    # b_rhs = [0.01, 0.01]

    b_mat = [[1, 1], [1, 1], [0, 1], [0, 1], [1, 0], [0, 1], [1, 0], [1, 0], [1, 1], [1, 1], [1, 1], [0, 1], [0, 1], [1, 0], [0, 1], [1, 0], [1, 0], [1, 1],
             [1, 1], [1, 1], [0, 1], [0, 1], [1, 0], [0, 1], [1, 0], [1, 0], [1, 1], [1, 1], [0, 1], [1, 0]]
    b_constraints = [[1, 0], [-1, 0], [0, 1], [0, -1]]
    b_rhs = [0.01, 0.01, 0.01, 0.01]

    mode.set_inputs(b_mat, b_constraints, b_rhs, allow_constants=False)

    error = ha.new_mode('error')

    trans = ha.new_transition(mode, error)

    trans.set_guard([[-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], ], [-2.5, ])

    print(" ****** define ha end ****** ")

    return ha


def make_init(ha):
    '''returns list of initial states'''

    mode = ha.modes['mode']
    init_lpi = lputil.from_box([[0.9, 1.1], [0.9, 1.1], [0.9, 1.1], [0.9, 1.1], [0.9, 1.1], [0.9, 1.1], [0.9, 1.1], [0.9, 1.1], [0.9, 1.1],
                                [0.9, 1.1], [0.9, 1.1], [0.9, 1.1], [0.9, 1.1], [0.9, 1.1], [0.9, 1.1], [0.9, 1.1], [0.9, 1.1], [0.9, 1.1],
                                [0.9, 1.1], [0.9, 1.1], [0.9, 1.1], [0.9, 1.1], [0.9, 1.1], [0.9, 1.1], [0.9, 1.1], [0.9, 1.1], [0.9, 1.1],
                                [0.9, 1.1], [0.9, 1.1], [0.9, 1.1]], mode)

    init_list = [StateSet(init_lpi, mode)]

    return init_list


def define_settings():
    'get the hylaa settings object'

    step = 0.2
    max_time = 20.0
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

    bdd_ce_object = BDD4CE(error_states, usafeset_preds, smt_mip='mip')
    # #
    bdd_graphs = bdd_ce_object.create_bdd_w_level_merge(level_merge=-1, order='random')
    valid_exps, invalid_exps = bdd_graphs[0].generate_expressions()
    print(len(valid_exps), len(invalid_exps))
    Timers.toc("BDD Construction")
    Timers.print_stats()


if __name__ == '__main__':
    run_hylaa()
