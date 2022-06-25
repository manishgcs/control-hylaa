import numpy as np
from farkas_central.star_data import LinearPredicate


def process_stars(error_states):
    if error_states[0].input_effects_list is None:
        process_stars_wo_inputs(error_states)
    else:
        process_stars_w_inputs(error_states)


def process_stars_wo_inputs(error_states):
    n_error_states = len(error_states)
    n_dims = len(error_states[0].basis_matrix)

    n_final_vars = n_dims
    n_final_constraints = 0  # constraint for the unsafe set is not considered (do it while converting U in star basis)
    for lpi_type in error_states[n_error_states-1].lpi.get_types():
        if lpi_type == 3.0:  # of type <=
            n_final_constraints = n_final_constraints + 1
    print(n_final_constraints, n_final_vars)

    # basis_matrices = []
    for error_state in error_states:
        error_state_lpi = error_state.lpi
        lpi_lhs = error_state_lpi.get_full_constraints().toarray()
        lpi_rhs = error_state_lpi.get_rhs().tolist()
        lpi_types = error_state_lpi.get_types()
        # print(error_state_lpi, lpi_lhs)

        preds = []
        idy = 0
        idx = 0
        while idx < len(lpi_types):
            if lpi_types[idx] == 3.0:
                pred_vector = np.zeros(n_final_vars, dtype=float)
                pred_rhs = lpi_rhs[idx]
                non_zeros_val = lpi_lhs[idx][np.nonzero(lpi_lhs[idx])]
                pred_vector[idy] = non_zeros_val
                pred = LinearPredicate(pred_vector, pred_rhs)
                preds.append(pred)
                idx = idx + 1
                pred_vector = np.zeros(n_final_vars, dtype=float)
                pred_rhs = lpi_rhs[idx]
                non_zeros_val = lpi_lhs[idx][np.nonzero(lpi_lhs[idx])]
                pred_vector[idy] = non_zeros_val
                pred = LinearPredicate(pred_vector, pred_rhs)
                preds.append(pred)
                idy = idy + 1
            idx = idx + 1
        error_state.set_constraints(preds)
    print("process_stars end ******")


def process_stars_w_inputs(error_states):
    n_error_states = len(error_states)
    n_dims = len(error_states[0].basis_matrix)
    # print(error_states[0].input_effects_list)
    n_inputs = error_states[0].input_effects_list.shape[2]  # shape is a tuple (array_elements, elem dim1, elem dim 2)
    # print(error_states[0].input_effects_list)
    n_input_steps = len(error_states[n_error_states-1].input_effects_list)
    print(n_dims, n_inputs, n_input_steps)

    n_final_vars = n_dims + n_input_steps*n_inputs
    n_final_constraints = 0  # constraint for the unsafe set is not considered (do it while converting U in star basis)
    for lpi_type in error_states[n_error_states-1].lpi.get_types():
        if lpi_type == 3.0:  # of type <=
            n_final_constraints = n_final_constraints + 1
    print(n_final_constraints, n_final_vars)

    for error_state in error_states:
        error_state.input_effects_list = error_state.input_effects_list[::-1]

    # basis_matrices = []
    for error_state in error_states:
        basis_matrix = error_state.basis_matrix

        # print(basis_matrix.shape, error_state.input_effects_list.shape)
        step = len(error_state.input_effects_list)
        input_effect_shape = error_state.input_effects_list[0].shape
        for input_effects in error_state.input_effects_list:
            basis_matrix = np.concatenate((basis_matrix, input_effects), axis=1)
        for idx in range(n_input_steps - step):
            placeholder_input_effect = np.zeros((input_effect_shape[0], input_effect_shape[1]), dtype=float)
            basis_matrix = np.concatenate((basis_matrix, placeholder_input_effect), axis=1)
        # print(basis_matrix, basis_matrix.shape)
        # basis_matrices.append(basis_matrix)
        error_state.basis_matrix = basis_matrix
        # print(error_state.basis_matrix)

        error_state_lpi = error_state.lpi
        lpi_lhs = error_state_lpi.get_full_constraints().toarray()
        lpi_rhs = error_state_lpi.get_rhs().tolist()
        lpi_types = error_state_lpi.get_types()

        preds = []
        idy = 0
        idx = 0
        while idx < len(lpi_types):
            if lpi_types[idx] == 3.0:
                pred_vector = np.zeros(n_final_vars, dtype=float)
                pred_rhs = lpi_rhs[idx]
                non_zeros_val = lpi_lhs[idx][np.nonzero(lpi_lhs[idx])]
                pred_vector[idy] = non_zeros_val
                pred = LinearPredicate(pred_vector, pred_rhs)
                # pred.print()
                preds.append(pred)
                idx = idx + 1
                pred_vector = np.zeros(n_final_vars, dtype=float)
                pred_rhs = lpi_rhs[idx]
                non_zeros_val = lpi_lhs[idx][np.nonzero(lpi_lhs[idx])]
                pred_vector[idy] = non_zeros_val
                pred = LinearPredicate(pred_vector, pred_rhs)
                preds.append(pred)
                # pred.print()
                idy = idy + 1
            idx = idx + 1
        error_state.set_constraints(preds)

    # X_matrix = np.concatenate((X_matrix, input_effect), axis=1)

    # print(cur_state.basis_matrix, cur_state.input_effects_list)
    # print(error_states[1].lpi.get_full_constraints().toarray())
    # print(cur_state.lpi.get_rhs().tolist())
    # print(cur_state.lpi.get_types())
    print("process_stars end ******")
