'''
Main Hylaa Reachability Implementation
Stanley Bak, 2018
'''

import numpy as np
from termcolor import cprint

from hylaa.settings import HylaaSettings, PlotSettings

from hylaa.plotutil import PlotManager
from hylaa.aggdag import AggDag
from hylaa.result import HylaaResult, make_counterexample
from hylaa.stateset import StateSet
from hylaa.hybrid_automaton import HybridAutomaton, was_tt_taken
from hylaa.timerutil import Timers
from hylaa.util import Freezable
from hylaa.lpinstance import LpInstance
from hylaa import lputil
from hylaa.result import PlotData
from hylaa.h_ah_polytope import HPolytope, minkowski_sum
from farkas_central.star_data import LinearPredicate, StarData


class Core(Freezable):
    'main computation object. initialize and call run()'

    def __init__(self, ha, hylaa_settings):
        assert isinstance(hylaa_settings, HylaaSettings)
        assert isinstance(ha, HybridAutomaton)

        self.hybrid_automaton = ha
        
        self.settings = hylaa_settings

        self.plotman = PlotManager(self)

        # computation
        self.aggdag = AggDag(hylaa_settings, self) # manages the waiting list and aggregation dag computation state
        
        self.max_steps_remaining = None # bound on num steps left in current mode ; assigned on pop

        self.took_tt_transition = False # flag for if a tt transition was taken (cur_state should be cleared)

        self.result = None # a HylaaResult... assigned on run() to store verification result

        self.continuous_steps = 0

        self.discrete = ha.discrete

        # simulation
        self.doing_simulation = False
        self.sim_waiting_list = None # list of (mode, pt, num_step)
        self.sim_states = None # list of (mode, pt, num_step)
        self.sim_basis_matrix = None # one-step basis matrix in current mode
        self.sim_should_try_guards = None # False for the first step unless urgent_guards is True
        self.sim_took_transition = None  # list of booleans for each sim
        self.error_stars = []
        self.non_error_stars = []
        self.errorset_preds = None

        # make random number generation (for example, to find orthogonal directions) deterministic
        np.random.seed(hylaa_settings.random_seed)

        LpInstance.print_normal = self.print_normal
        LpInstance.print_verbose = self.print_verbose
        LpInstance.print_debug = self.print_debug

        self.freeze_attrs()

    def get_error_stars(self):
        return self.error_stars

    def get_non_error_stars(self):
        return self.non_error_stars

    def set_errorset_preds(self, error_transition):
        pred_vector = error_transition.guard_csr.todense().tolist()[0]
        pred_rhs = error_transition.guard_rhs[0]
        # print(pred_vector, pred_rhs)
        self.errorset_preds = [LinearPredicate(pred_vector, pred_rhs)]

    def get_errorset_preds(self):
        return self.errorset_preds

    def print_normal(self, msg):
        'print function for STDOUT_NORMAL and above'

        if self.settings.stdout >= HylaaSettings.STDOUT_NORMAL:
            cprint(msg, self.settings.stdout_colors[HylaaSettings.STDOUT_NORMAL])

    def print_verbose(self, msg):
        'print function for STDOUT_VERBOSE and above'

        if self.settings.stdout >= HylaaSettings.STDOUT_VERBOSE:
            cprint(msg, self.settings.stdout_colors[HylaaSettings.STDOUT_VERBOSE])

    def print_debug(self, msg):
        'print function for STDOUT_DEBUG and above'

        if self.settings.stdout >= HylaaSettings.STDOUT_DEBUG:
            cprint(msg, self.settings.stdout_colors[HylaaSettings.STDOUT_DEBUG])

    def is_finished(self):
        'is the computation finished'

        finished = False

        if self.doing_simulation:
            finished = not self.sim_states and not self.sim_waiting_list
        else:
            if self.settings.stop_on_aggregated_error and self.result.has_aggregated_error:
                finished = True
            elif self.settings.stop_on_concrete_error and self.result.has_concrete_error:
                finished = True
            else:    
                finished = self.aggdag.get_cur_state() is None and not self.aggdag.waiting_list and \
                           not self.aggdag.deagg_man.doing_replay()
            
        return finished

    def error_reached(self, state, t, t_lpi):
        'an error mode was reached after taking transition t, report and create counterexample'

        step_num = state.cur_steps_since_start
        times = [round(self.settings.step_size * step_num[0], 12), round(self.settings.step_size * step_num[1], 12)]

        if step_num[0] == step_num[1]:
            step_num = step_num[0]
            times = times[0]

        if not self.result.has_aggregated_error and not state.is_concrete:
            self.result.has_aggregated_error = True
            
            self.print_normal(f"Unsafe Mode (aggregated) Reached at Step: {step_num} / {self.settings.num_steps}, " +
                              f"time {times}")

        # if this is a concrete state (not aggregated) and we don't yet have a counter-example
        if not self.result.has_concrete_error and state.is_concrete:
            self.result.has_concrete_error = True
            self.print_normal(f"Unsafe Mode (concrete) Reached at Step: {step_num} / " + \
                              f"{self.settings.num_steps}, time {times}")

            if self.settings.make_counterexample and not self.result.counterexample:
                self.print_verbose("Reached concrete error state; making concrete counter-example")
                self.result.counterexample = make_counterexample(self.hybrid_automaton, state, t, t_lpi)

                # todo: implement this with inputs
                #self.plotman.draw_counterexample(self.result.counterexample)

    def check_guards(self):
        '''check for discrete successors with the guards'''

        Timers.tic("check_guards")

        cur_state = self.aggdag.get_cur_state()
        # print(cur_state.lpi)
        cur_step = cur_state.cur_steps_since_start.copy()
        for t in cur_state.mode.transitions:
            t_lpi = t.get_guard_intersection(cur_state.lpi)

            if t_lpi:
                if t.to_mode.is_error():
                    self.error_reached(cur_state, t, t_lpi)

                    # step_in_mode = cur_state.cur_step_in_mode
                    # basis_matrix, input_effects_matrix = cur_state.mode.time_elapse.get_basis_matrix(step_in_mode)
                    # print(basis_matrix, input_effects_matrix, cur_state.lpi)
                    # print(cur_state.basis_matrix, cur_state.input_effects_list)

                    if cur_state.input_effects_list is None:
                        self.error_stars.append(StarData(cur_step, cur_state.lpi.clone(), np.copy(cur_state.basis_matrix)))
                    else:
                        self.error_stars.append(StarData(cur_step, cur_state.lpi.clone(), np.copy(cur_state.basis_matrix), np.copy(cur_state.input_effects_list)))
                    # print(np.copy(cur_state.basis_matrix))
                    if self.errorset_preds is None:
                        self.set_errorset_preds(t)

                    if self.settings.stop_on_aggregated_error:
                        break

                    if cur_state.is_concrete and self.settings.stop_on_concrete_error:
                        break
                self.aggdag.add_transition_successor(t, t_lpi)

                self.print_verbose(f"Took transition {t} at steps {cur_state.cur_steps_since_start}")

                # if it's a time-triggered transition, we may remove cur_state immediately
                if self.settings.optimize_tt_transitions and t.time_triggered:
                    if was_tt_taken(cur_state.lpi, t, self.settings.step_size, self.settings.num_steps):
                        self.print_verbose("Transition was time-triggered, finished with current state analysis")
                        self.took_tt_transition = True
                    else:
                        self.print_verbose("Transition was NOT taken as time-triggered, due to runtime checks")

            else:
                self.non_error_stars.append(StarData(cur_step, cur_state.lpi.clone(), np.copy(cur_state.basis_matrix),
                                                     np.copy(cur_state.input_effects_list)))

        Timers.toc("check_guards")

    def intersect_invariant(self, state, add_ops_to_aggdag=True):
        '''intersect the (current or passed-in) state with the mode invariant'''

        Timers.tic("intersect_invariant")

        is_feasible = True

        for invariant_index, lc in enumerate(state.mode.inv_list):
            if lputil.check_intersection(state.lpi, lc.negate()):
                old_row = state.invariant_constraint_rows[invariant_index]
                vec = lc.csr.toarray()[0]
                rhs = lc.rhs

                if old_row is None:
                    # new constraint
                    row = lputil.add_init_constraint(state.lpi, vec, rhs, state.basis_matrix,
                                                     state.input_effects_list)
                    state.invariant_constraint_rows[invariant_index] = row
                    is_stronger = False
                else:
                    # strengthen existing constraint possibly
                    row, is_stronger = lputil.try_replace_init_constraint(state.lpi, old_row, vec, rhs, \
                        state.basis_matrix, state.input_effects_list)
                    state.invariant_constraint_rows[invariant_index] = row

                if add_ops_to_aggdag:
                    self.aggdag.add_invariant_op(state.cur_step_in_mode, invariant_index, is_stronger)

                # adding the invariant condition may make the lp infeasible
                if not state.lpi.is_feasible():
                    is_feasible = False
                    break

        Timers.toc("intersect_invariant")

        return is_feasible

    def print_current_step_time(self):
        'print the current step and time'

        step_num = self.aggdag.get_cur_state().cur_steps_since_start
        times = [round(self.settings.step_size * step_num[0], 12), round(self.settings.step_size * step_num[1], 12)]

        if step_num[0] == step_num[1]:
            step_num = step_num[0]
            times = times[0]

        self.print_verbose("Step From {} / {} ({})".format(step_num, self.settings.num_steps, times))

    # The most efficient as we don't need to compute the inverse of V. We use the basis matrix to
    # directly compute x \in R^n, where  x = x_bar + XP_x. Here P_x is H * x \leq rhs
    def minkowski_sum_direct_at_new_step(self, state):
        new_basis_matrix = state.basis_matrix
        input_effects_list = state.input_effects_list
        cur_lpi = state.lpi
        con_matrix = cur_lpi.get_full_constraints().toarray()
        lpi_rhs = cur_lpi.get_rhs().tolist()
        lpi_types = cur_lpi.get_types()
        X_matrix = new_basis_matrix
        input_steps = len(input_effects_list)
        n_inputs = 0
        for idx in range(input_steps):
            input_effect = input_effects_list[idx]
            n_inputs = input_effect.shape[1]
            # print(input_effect, new_basis_matrix)
            X_matrix = np.concatenate((X_matrix, input_effect), axis=1)

        X_matrix_flattened = []
        for idx in range(len(X_matrix)):
            X_matrix_row = X_matrix[idx]
            X_matrix_row = X_matrix_row.tolist()[0]
            X_matrix_flattened.append(X_matrix_row)
        X_matrix_flattened = np.array(X_matrix_flattened)
        # print(np.array(X_matrix_flattened).shape)
        X_matrix = X_matrix_flattened
        dim1, dim2 = X_matrix.shape
        # print(dim2-dim1, input_steps,n_inputs)
        assert dim2-dim1 == input_steps * n_inputs
        H_matrix = np.zeros((dim2*2, dim2), dtype=float)
        h_rhs = np.zeros(dim2*2, dtype=float)
        con_types = np.empty(dim2*2)
        con_types.fill(3)
        H_matrix[0:dim1*2, 0:dim1] = con_matrix[dim1:dim1*3, 0:dim1]
        h_rhs[0:dim1*2] = lpi_rhs[dim1:dim1*3]
        con_types[0:dim1*2] = lpi_types[dim1:dim1*3]
        x1 = dim1*2
        y1 = dim1
        x2 = dim1*3 + dim1
        y2 = dim1*dim1

        for idx in range(input_steps):
            H_matrix[x1:x1+n_inputs*2, y1:y1+n_inputs] = con_matrix[x2:x2+n_inputs*2, y2:y2+n_inputs]
            h_rhs[x1:x1+n_inputs*2] = lpi_rhs[x2:x2+n_inputs*2]
            con_types[x1:x1+n_inputs*2] = lpi_types[x2:x2+n_inputs*2]
            x1 = x1+n_inputs*2
            y1 = y1+n_inputs
            x2 = x2+n_inputs*2
            y2 = y2+n_inputs

        cur_h_polytope = HPolytope(H_matrix=np.array(H_matrix), h_rhs=np.array(h_rhs), con_types=np.array(con_types))
        cur_h_polytope.convert_to_std_h_polytope()
        cur_ah_polytope = cur_h_polytope.convert_to_AH_polytope(T_matrix=X_matrix)
        # cur_ah_polytope.print()
        return cur_ah_polytope

    # Working with input matrix of same dimension as system's dimensions. Breaking the basis matrix into two parts,
    # and for both parts, converting star predicates into std basis. Requires inversion of V that's why for now
    # I only tried it on square matrices i.e., with n inputs with n dimensional systems
    def minkowski_sum_at_new_step_std_basis(self, state):
        new_basis_matrix = state.basis_matrix
        input_effects_list = state.input_effects_list
        cur_lpi = state.lpi
        dims = len(new_basis_matrix[0])
        con_matrix = cur_lpi.get_full_constraints().toarray()
        std_basis_init_pred_h_matrix = con_matrix[dims:dims*dims, 0:dims]
        auto_h_matrix = np.matmul(std_basis_init_pred_h_matrix, np.linalg.inv(new_basis_matrix))
        auto_rhs = cur_lpi.get_rhs().tolist()[dims:dims * dims]
        auto_types = cur_lpi.get_types()[dims:dims * dims]
        auto_h_polytope = HPolytope(H_matrix=np.array(auto_h_matrix), h_rhs=np.array(auto_rhs),
                                    con_types=np.array(auto_types))
        auto_h_polytope.convert_to_std_h_polytope()
        auto_ah_polytope = auto_h_polytope.convert_to_AH_polytope()
        # auto_h_polytope.print()

        # cur_inp_h_matrix = auto_h_matrix.copy()
        cur_new_input_effect = np.array(input_effects_list[len(input_effects_list) - 1])
        # print("Here2 " + str(np.linalg.inv(cur_new_input_effect)))
        dim1, dim2 = cur_new_input_effect.shape
        cur_inp_h_matrix = np.matmul(std_basis_init_pred_h_matrix.copy(), np.linalg.inv(cur_new_input_effect))
        # cur_inp_rhs = auto_rhs.copy()
        cur_inp_rhs = cur_lpi.get_rhs().tolist()[-dim2 * 2:]
        cur_inp_types = auto_types.copy()
        cur_inp_h_polytope = HPolytope(H_matrix=np.array(cur_inp_h_matrix), h_rhs=np.array(cur_inp_rhs),
                                       con_types=np.array(cur_inp_types))
        cur_inp_h_polytope.convert_to_std_h_polytope()
        cur_inp_ah_polytope = cur_inp_h_polytope.convert_to_AH_polytope()

        input_ah_polytopes_list = state.input_ah_polytopes_list
        if len(input_ah_polytopes_list) > 0:
            print("Computing subsequent step ****")
            prev_inp_ah_polytopes_effect = input_ah_polytopes_list[len(input_ah_polytopes_list) - 1]
            inp_ah_polytope = minkowski_sum(cur_inp_ah_polytope, prev_inp_ah_polytopes_effect)
        else:
            print("Computing first step ****")
            inp_ah_polytope = cur_inp_ah_polytope

        assert inp_ah_polytope is not None
        state.input_ah_polytopes_list.append(inp_ah_polytope)

        new_ah_polytope = minkowski_sum(auto_ah_polytope, inp_ah_polytope)
        # new_ah_polytope.print()
        return new_ah_polytope

    # old - not working. Taking entire basis matrix as is (by only considering alpha's for both auto and inputs)
    def minkowski_sum_at_new_step_std_star_basis(self, state):
        new_basis_matrix = state.basis_matrix
        print("Here1 " + str(np.linalg.inv(new_basis_matrix)))
        input_effects_list = state.input_effects_list
        cur_lpi = state.lpi
        dims = len(new_basis_matrix[0])
        con_matrix = cur_lpi.get_full_constraints().toarray()
        auto_h_matrix = con_matrix[0:dims*dims, 0:dims*2]
        auto_rhs = cur_lpi.get_rhs().tolist()[0:dims*dims]
        auto_types = cur_lpi.get_types()[0:dims*dims]
        auto_h_polytope = HPolytope(H_matrix=np.array(auto_h_matrix), h_rhs=np.array(auto_rhs),
                                    con_types=np.array(auto_types))
        auto_h_polytope.convert_to_std_h_polytope()
        auto_ah_polytope = auto_h_polytope.convert_to_AH_polytope()
        # auto_h_polytope.print()

        cur_inp_h_matrix = auto_h_matrix.copy()
        cur_new_input_effect = np.array(input_effects_list[len(input_effects_list)-1])
        print("Here2 " + str(np.linalg.inv(cur_new_input_effect)))
        dim1, dim2 = cur_new_input_effect.shape
        cur_inp_h_matrix[0:dim1, 0:dim2] = cur_new_input_effect
        cur_inp_rhs = auto_rhs.copy()
        cur_inp_rhs[-dim2*2:] = cur_lpi.get_rhs().tolist()[-dim2*2:]
        cur_inp_types = auto_types.copy()
        cur_inp_h_polytope = HPolytope(H_matrix=np.array(cur_inp_h_matrix), h_rhs=np.array(cur_inp_rhs),
                                       con_types=np.array(cur_inp_types))
        cur_inp_h_polytope.convert_to_std_h_polytope()
        cur_inp_ah_polytope = cur_inp_h_polytope.convert_to_AH_polytope()

        input_ah_polytopes_list = state.input_ah_polytopes_list
        if len(input_ah_polytopes_list) > 0:
            print("Computing subsequent step ****")
            prev_inp_ah_polytopes_effect = input_ah_polytopes_list[len(input_ah_polytopes_list)-1]
            inp_ah_polytope = minkowski_sum(cur_inp_ah_polytope, prev_inp_ah_polytopes_effect)
        else:
            print("Computing first step ****")
            inp_ah_polytope = cur_inp_ah_polytope

        assert inp_ah_polytope is not None
        state.input_ah_polytopes_list.append(inp_ah_polytope)

        new_ah_polytope = minkowski_sum(auto_ah_polytope, inp_ah_polytope)
        # new_ah_polytope.print()
        return new_ah_polytope

    def do_step_continuous_post(self, p1_ah_polytope=None):
        '''do a step where it's part of a continuous post'''

        Timers.tic('do_step_continuous_post')

        cur_state = self.aggdag.get_cur_state()
        # print(cur_state.basis_matrix, cur_state.input_effects_list, cur_state.lpi)
        # self.print_current_step_time()

        if not self.is_finished():
            if cur_state.cur_steps_since_start[0] >= self.settings.num_steps:
                self.print_normal("State reached computation time bound")
                self.aggdag.cur_state_left_invariant(reached_time_bound=True)

            elif self.took_tt_transition:
                self.print_normal("State reached a time-triggered transition")
                self.took_tt_transition = False # reset the flag
                self.aggdag.cur_state_left_invariant()
            else:
                still_feasible = self.intersect_invariant(cur_state)
                
                if not still_feasible:
                    self.print_normal("State left the invariant after {} steps".format(cur_state.cur_step_in_mode))
                    self.aggdag.cur_state_left_invariant()
                else:
                    # print("********* doing the step *******")
                    cur_state.step()
                    self.check_guards()  # check guards here, before doing an invariant intersection
                    Timers.tic("PContain")
                    # new_ah_polytope = self.minkowski_sum_at_new_step_std_basis(cur_state)
                    # P1_ah_polytope = convert_lpi_to_ah_polytope_std_basis(P1_lpi=P1_lpi)
                    # ret_status = new_ah_polytope.check_inclusion_std_basis(P1_ah_polytope=P1_ah_polytope)

                    # Uncomment 1st and 3rd lines for PContain
                    # new_ah_polytope = self.minkowski_sum_direct_at_new_step(cur_state)
                    # P1_ah_polytope = convert_lpi_to_ah_polytope(P1_lpi=P1_lpi, dims=cur_state.basis_matrix.shape[0])
                    # ret_status = new_ah_polytope.check_inclusion(p1_ah_polytope=p1_ah_polytope)
                    Timers.toc("PContain")
                    # print("********* finished the step *******")

                    # if the current mode has zero dynamics, remove it here
                    if cur_state.mode.a_csr.nnz == 0 and self.settings.skip_zero_dynamics_modes:
                        self.print_normal("State in mode '{}' with zero dynamics, skipping remaining steps".format( \
                            cur_state.mode.name))
                        self.aggdag.cur_state_left_invariant()

        if self.is_finished():
            self.print_normal("Computation finished after {} continuous-post steps.".format(self.continuous_steps))

        Timers.toc('do_step_continuous_post')

    def do_step_pop(self):
        'do a step where we pop from the waiting list'

        Timers.tic('do_step_pop')

        self.plotman.state_popped() # reset certain per-mode plot variables
        self.aggdag.print_waiting_list()

        self.result.last_cur_state = cur_state = self.aggdag.pop_waiting_list()

        self.print_normal("Removed state in mode '{}' at step {} ({} in mode) (Waiting list has {} left)".format( \
                cur_state.mode.name, cur_state.cur_steps_since_start, cur_state.cur_step_in_mode, \
                len(self.aggdag.waiting_list)))

        # if a_matrix is None, it's an error mode
        if cur_state.mode.a_csr is None:
            self.print_normal("Mode '{}' was an error mode; skipping.".format(cur_state.mode.name))

            self.aggdag.cur_state_left_invariant()
        else:
            self.max_steps_remaining = self.settings.num_steps - cur_state.cur_steps_since_start[0]

            still_feasible = self.intersect_invariant(cur_state)

            if not still_feasible:
                self.print_normal("Continuous state was outside of the mode's invariant; skipping.")
                self.aggdag.cur_state_left_invariant()
            else:
                cur_state.apply_approx_model(self.settings.approx_model)

        # pause plot
        self.print_verbose("Pausing due to step_pop()")
        self.plotman.pause()

        Timers.toc('do_step_pop')

    def do_step(self, p1_ah_polytope=None):
        'do a single step of the computation'

        if self.doing_simulation:
            self.do_step_sim()
        else:
            self.do_step_reach(p1_ah_polytope=p1_ah_polytope)
            
    def do_step_reach(self, p1_ah_polytope=None):
        'do a single reach step of the computation'

        Timers.tic('do_step')

        if not self.is_finished():
            if self.aggdag.get_cur_state():
                self.do_step_continuous_post(p1_ah_polytope=p1_ah_polytope)
                self.continuous_steps += 1
            elif self.aggdag.deagg_man.doing_replay():
                # in the middle of a deaggregation replay
                self.aggdag.deagg_man.do_step_replay()
            else:
                # begin a deaggregation replay or pop a state off the waiting list
                deagg_node = self.settings.aggstrat.get_deagg_node(self.aggdag)

                if deagg_node:
                    self.aggdag.deagg_man.begin_replay(deagg_node)
                    self.aggdag.deagg_man.do_step_replay()
                else:
                    # print(".core popping, calling aggdag.save_viz()")
                    # self.aggdag.save_viz()
            
                    # pop state off waiting list
                    self.do_step_pop()

                    if self.settings.process_urgent_guards and self.aggdag.get_cur_state() is not None:
                        self.check_guards()

        Timers.toc('do_step')

    def setup_ha(self, ha):
        'setup hybrid automata for computation / simulation; a substep of setup()'

        # initialize time elapse in each mode of the hybrid automaton
        for mode in ha.modes.values():
            mode.init_time_elapse(self.settings.step_size)

        if self.settings.optimize_tt_transitions:
            ha.detect_tt_transitions(self.settings.step_size, self.settings.num_steps, self.print_debug)

        if self.settings.do_guard_strengthening:
            ha.do_guard_strengthening()

        ha.check_transitions()

    def setup(self, init_state_list):
        'setup the computation (called by run())'

        Timers.tic('setup')

        assert init_state_list and isinstance(init_state_list, list), "expected list of initial states"

        for state in init_state_list:
            assert isinstance(state, StateSet), "initial states should be a list of StateSet objects"

        self.result = HylaaResult()

        self.setup_ha(init_state_list[0].mode.ha)
        
        self.plotman.create_plot()

        # populate waiting list
        assert not self.aggdag.waiting_list, "waiting list was not empty"

        for state in init_state_list:
            if not state.lpi.is_feasible():
                self.print_normal("Removed an infeasible initial set in mode {}".format(state.mode.name))
                continue
            
            still_feasible = self.intersect_invariant(state, add_ops_to_aggdag=False)

            if still_feasible:
                # reset the cached info about invariant intersections, since the intersection ops are not in the aggdag
                state.invariant_constraint_rows = [None] * len(state.mode.inv_list)
                
                self.aggdag.add_init_state(state)
            else:
                self.print_normal("Removed an infeasible initial set after invariant intersection in mode {}".format( \
                        state.mode.name))

        if not self.aggdag.waiting_list:
            raise RuntimeError("Error: No feasible initial states were defined.")

        Timers.toc('setup')

    def run_to_completion(self, p1_ah_polytope=None):
        'run the model to completion (called by run() if not plot is desired)'

        if self.settings.plot.store_plot_result and self.result.plot_data is None:
            self.result.plot_data = PlotData(self.plotman.num_subplots)

        self.plotman.run_to_completion(compute_plot=self.settings.plot.store_plot_result, p1_ah_polytope=p1_ah_polytope)

    def run(self, init_state_list, p1_ah_polytope=None):
        '''
        Run the computation (main entry point)

        init_star is the initial state

        fixed_dim_list, if used, is a list of dimensions with fixed initial values
        '''

        Timers.reset()
        Timers.tic("total")

        self.setup(init_state_list)

        if self.settings.plot.plot_mode == PlotSettings.PLOT_INTERACTIVE:
            # make sure to store plot result for on_click listener to report on
            self.print_verbose(f"Setting store_plot_result to true since PLOT_INTERACTIVE has click listener")
            self.settings.plot.store_plot_result = True

        if self.settings.plot.store_plot_result:
            self.result.plot_data = PlotData(self.plotman.num_subplots)

        if self.settings.plot.plot_mode == PlotSettings.PLOT_NONE:
            self.run_to_completion(p1_ah_polytope=p1_ah_polytope)
        else:
            self.plotman.compute_and_animate(p1_ah_polytope=p1_ah_polytope)

        Timers.toc("total")

        if self.settings.stdout >= HylaaSettings.STDOUT_VERBOSE:
            Timers.print_stats()

        if self.result.has_concrete_error:
            self.print_normal("Result: Error modes are reachable (found counter-example).\n")
        elif self.result.has_aggregated_error:
            self.print_normal("Result: System is safe, although error modes were reachable when aggregation " + \
                              "(overapproximation) was used.\n")
        else:
            self.print_normal("Result: System is safe. Error modes are NOT reachable.\n")

        self.print_normal("Total Runtime: {:.2f} sec".format(Timers.top_level_timer.total_secs))

        # assign results
        self.result.top_level_timer = Timers.top_level_timer
        Timers.reset()

        return self.result

    def simulate(self, init_mode, box, num_sims):
        '''
        run a number of discrete time simulations (and plot them according to the settings)

        if num_sims is a tuple, it will skip a certain number of simulations, 
            for example num_sims=[5, 10] will plot simulations 5-10
        '''

        if isinstance(num_sims, int):
            num_sims = [0, num_sims]

        # initialize time elapse in each mode of the hybrid automaton
        ha = init_mode.ha

        # initialize result object (many fields unused during simulation)
        self.result = HylaaResult()

        self.setup_ha(ha)

        # each simulation is a tuple (mode, pt, num_steps)
        self.plotman.create_plot()

        assert self.settings.plot.plot_mode != PlotSettings.PLOT_NONE, "simulate called with PLOT_NONE"
        dims = len(box)
        assert dims == init_mode.a_csr.shape[0]

        self.doing_simulation = True
        self.sim_waiting_list = []

        for sim_index in range(num_sims[1]):
            rand_array = np.random.rand(dims)

            if sim_index < num_sims[0]:
                continue

            pt = [box[i][0] + rand_array[i] * (box[i][1] - box[i][0]) for i in range(dims)]

            self.sim_waiting_list.append([init_mode, np.array(pt, dtype=float), 0])
        
        self.plotman.compute_and_animate()

        return self.result

    def sim_pop_waiting_list(self):
        'pop a state off the simulation waiting list'

        min_time_mode = self.settings.aggstrat.get_simulation_pop_mode(self.sim_waiting_list)

        # pop all states in the same mode
        new_waiting_list = []
        self.sim_states = []

        for item in self.sim_waiting_list:
            mode, _, _ = item
            
            if mode is min_time_mode:
                self.sim_states.append(item)
            else:
                new_waiting_list.append(item)

        self.sim_waiting_list = new_waiting_list

        assert min_time_mode is not None
        assert min_time_mode.time_elapse is not None, f"time elapse was None for mode {min_time_mode.name}"
        self.sim_basis_matrix, ie_mat = min_time_mode.time_elapse.get_basis_matrix(1)
        assert ie_mat is None, "simulation with inputs unimplemented"

        self.print_verbose(f"Popped {len(self.sim_states)} off waiting list ({len(self.sim_waiting_list)} remaining)")
                
    def do_step_sim(self):
        'do a simulation step'

        if not self.sim_states:
            if self.sim_waiting_list: # if simulation is not done
                # pop minimum time state
                self.sim_pop_waiting_list()

                self.sim_should_try_guards = self.settings.process_urgent_guards
                self.sim_took_transition = [False] * len(self.sim_states)
                self.plotman.pause()
                self.print_verbose("Pausing due to sim pop()")
        else:
            # simulate one step for all states in sim_states
            finished = True
            for i, obj in enumerate(self.sim_states):
                if obj is None:
                    continue

                mode, pt, steps = obj

                if steps == self.settings.num_steps:
                    self.print_verbose(f'Sim #{i} reached the time bound')
                    self.sim_states[i] = None
                    continue

                if self.sim_took_transition[i]:
                    self.sim_states[i] = None
                    continue

                # try guards (before advancing)
                if self.sim_should_try_guards:
                    for transition in mode.transitions:
                        if transition.is_guard_true_for_point(pt):

                            # print(f"guard was true for point {pt}, MAT:")
                            # print(f"{transition.guard_csr.toarray()}\nRHS:\n{transition.guard_rhs}")
                            
                            post_pt, post_mode = transition.apply_reset_for_point(pt)

                            if not post_mode.is_error():
                                self.sim_waiting_list.append([post_mode, post_pt, steps])
                                
                            self.sim_took_transition[i] = True
                            obj[1] = post_pt
                            self.print_verbose(f'Sim #{i} took transition {transition}')
                            break

                if self.sim_took_transition[i]:
                    finished = False
                    continue
                        
                if not mode.point_in_invariant(pt):
                    self.print_verbose(f'Invariant became False for Sim #{i}')
                    self.sim_states[i] = None
                    continue

                self.sim_should_try_guards = True
                finished = False
                
                # advance time
                pt = np.dot(self.sim_basis_matrix, pt)

                obj[1] = pt
                obj[2] = steps + 1

            if finished:
                self.plotman.commit_cur_sims()
                self.sim_states = None
