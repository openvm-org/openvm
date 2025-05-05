use core::mem::MaybeUninit;

use openvm_stark_backend::{p3_field, p3_matrix};
use p3_field::PrimeField;
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixViewMut};
use p3_poseidon2::GenericPoseidon2LinearLayers;

// #[instrument(name = "generate vectorized Poseidon2 trace", skip_all)]
// pub fn generate_vectorized_trace_rows<
//     F: PrimeField,
//     LinearLayers: GenericPoseidon2LinearLayers<F, WIDTH>,
//     const WIDTH: usize,
//     const SBOX_DEGREE: u64,
//     const SBOX_REGISTERS: usize,
//     const HALF_FULL_ROUNDS: usize,
//     const PARTIAL_ROUNDS: usize,
//     const VECTOR_LEN: usize,
// >(
//     inputs: Vec<[F; WIDTH]>,
//     round_constants: &RoundConstants<F, WIDTH, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>,
//     extra_capacity_bits: usize,
// ) -> RowMajorMatrix<F> {
//     let n = inputs.len();
//     assert!(
//         n % VECTOR_LEN == 0 && (n / VECTOR_LEN).is_power_of_two(),
//         "Callers expected to pad inputs to VECTOR_LEN times a power of two"
//     );

//     let nrows = n.div_ceil(VECTOR_LEN);
//     let ncols = num_cols::<WIDTH, SBOX_DEGREE, SBOX_REGISTERS, HALF_FULL_ROUNDS,
// PARTIAL_ROUNDS>()
//         * VECTOR_LEN;
//     let mut vec = Vec::with_capacity((nrows * ncols) << extra_capacity_bits);
//     let trace: &mut [MaybeUninit<F>] = &mut vec.spare_capacity_mut()[..nrows * ncols];
//     let trace: RowMajorMatrixViewMut<MaybeUninit<F>> = RowMajorMatrixViewMut::new(trace,
// ncols);

//     let (prefix, perms, suffix) = unsafe {
//         trace.values.align_to_mut::<MultiRowPoseidon2Cols<
//             MaybeUninit<F>,
//             WIDTH,
//             SBOX_DEGREE,
//             SBOX_REGISTERS,
//             HALF_FULL_ROUNDS,
//             PARTIAL_ROUNDS,
//         >>()
//     };
//     assert!(prefix.is_empty(), "Alignment should match");
//     assert!(suffix.is_empty(), "Alignment should match");
//     assert_eq!(perms.len(), n);

//     perms.par_iter_mut().zip(inputs).for_each(|(perm, input)| {
//         generate_trace_rows_for_perm::<
//             F,
//             LinearLayers,
//             WIDTH,
//             SBOX_DEGREE,
//             SBOX_REGISTERS,
//             HALF_FULL_ROUNDS,
//             PARTIAL_ROUNDS,
//         >(perm, input, round_constants);
//     });

//     unsafe {
//         vec.set_len(nrows * ncols);
//     }

//     RowMajorMatrix::new(vec, ncols)
// }
// use tracing::instrument;
use crate::{
    columns::{num_cols, MultiRowPoseidon2Cols, SBox},
    constants::RoundConstants,
};

// TODO: Take generic iterable
// #[instrument(name = "generate Poseidon2 trace", skip_all)]
#[allow(clippy::needless_range_loop)]
pub fn generate_trace_rows<
    F: PrimeField,
    LinearLayers: GenericPoseidon2LinearLayers<F, WIDTH>,
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
>(
    inputs: Vec<[F; WIDTH]>,
    constants: &RoundConstants<F, WIDTH, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>,
) -> RowMajorMatrix<F> {
    let n = inputs.len();
    assert!(
        n.is_power_of_two(),
        "Callers expected to pad inputs to a power of two"
    );
    let rows_per_perm = 2 * HALF_FULL_ROUNDS + PARTIAL_ROUNDS + 1;
    // I'm lazy
    let round_up = rows_per_perm.next_power_of_two();
    let ncols = num_cols::<WIDTH, SBOX_DEGREE, SBOX_REGISTERS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>();
    let mut vec = Vec::with_capacity(round_up * ncols * n * 2);
    let trace: &mut [MaybeUninit<F>] = &mut vec.spare_capacity_mut()[..n * round_up * ncols];
    let trace: RowMajorMatrixViewMut<MaybeUninit<F>> = RowMajorMatrixViewMut::new(trace, ncols);

    let (prefix, perms, suffix) = unsafe {
        trace.values.align_to_mut::<MultiRowPoseidon2Cols<
            MaybeUninit<F>,
            WIDTH,
            SBOX_DEGREE,
            SBOX_REGISTERS,
            HALF_FULL_ROUNDS,
            PARTIAL_ROUNDS,
        >>()
    };
    assert!(prefix.is_empty(), "Alignment should match");
    assert!(suffix.is_empty(), "Alignment should match");
    assert_eq!(perms.len(), n * round_up);
    let mut state: [F; WIDTH] = inputs[0];
    let mut input = inputs[0];
    for i in 0..round_up * n {
        let idx = i % rows_per_perm;
        let input_idx = i / rows_per_perm;
        if idx == 0 {
            state = if input_idx >= n {
                input = inputs[0];
                inputs[0]
            } else {
                input = inputs[input_idx];
                inputs[input_idx]
            }
        }
        let mut pidx = 0;
        let mut is_full = false;
        let mut is_partial = false;
        if idx == 0 {
        } else if idx < 1 + HALF_FULL_ROUNDS {
            is_full = true;
            pidx = idx - 1;
        } else if idx < 1 + HALF_FULL_ROUNDS + PARTIAL_ROUNDS {
            is_partial = true;
            pidx = idx - 1 - HALF_FULL_ROUNDS;
        } else {
            is_full = true;
            pidx = idx - 1 - PARTIAL_ROUNDS;
        }

        state = generate_trace_rows_for_perm::<
            F,
            LinearLayers,
            WIDTH,
            SBOX_DEGREE,
            SBOX_REGISTERS,
            HALF_FULL_ROUNDS,
            PARTIAL_ROUNDS,
        >(
            &mut perms[i],
            input,
            state,
            constants,
            idx == rows_per_perm - 1,
            idx == 0,
            is_full,
            is_partial,
            pidx,
        );
    }

    unsafe {
        vec.set_len(n * ncols * round_up);
    }

    RowMajorMatrix::new(vec, ncols)
}

/// `rows` will normally consist of 24 rows, with an exception for the final row.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop)]
fn generate_trace_rows_for_perm<
    F: PrimeField,
    LinearLayers: GenericPoseidon2LinearLayers<F, WIDTH>,
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
>(
    perm: &mut MultiRowPoseidon2Cols<
        MaybeUninit<F>,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
    >,
    input: [F; WIDTH],
    mut state: [F; WIDTH],
    constants: &RoundConstants<F, WIDTH, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>,
    is_export: bool,
    is_beginning: bool,
    is_full: bool,
    is_partial: bool,
    // for full/partial rounds
    idx: usize,
) -> [F; WIDTH] {
    for (i, c) in input.iter().zip(perm.inputs.iter_mut()) {
        c.write(*i);
    }
    for (i, c) in state.iter().zip(perm.cur_state.iter_mut()) {
        c.write(*i);
    }
    if is_export {
        perm.export.write(F::ONE);
    } else {
        perm.export.write(F::ZERO);
    }
    perm.is_beginning
        .write(if is_beginning { F::ONE } else { F::ZERO });
    perm.is_full.write(if is_full { F::ONE } else { F::ZERO });
    perm.is_partial
        .write(if is_partial { F::ONE } else { F::ZERO });
    if is_beginning {
        for c in perm.constants.iter_mut() {
            c.write(F::ZERO);
        }
    } else if is_full {
        if idx >= HALF_FULL_ROUNDS {
            for (c, v) in perm
                .constants
                .iter_mut()
                .zip(constants.ending_full_round_constants[idx - HALF_FULL_ROUNDS])
            {
                c.write(v);
            }
            for (s, c) in state
                .iter_mut()
                .zip(constants.ending_full_round_constants[idx - HALF_FULL_ROUNDS])
            {
                *s += c;
            }
        } else {
            for (c, v) in perm
                .constants
                .iter_mut()
                .zip(constants.beginning_full_round_constants[idx])
            {
                c.write(v);
            }
            for (s, c) in state
                .iter_mut()
                .zip(constants.beginning_full_round_constants[idx])
            {
                *s += c;
            }
        }
    } else if is_partial {
        for (i, c) in perm.constants.iter_mut().enumerate() {
            if i == 0 {
                c.write(constants.partial_round_constants[idx]);
            } else {
                c.write(F::ZERO);
            }
        }
        state[0] += constants.partial_round_constants[idx];
    }
    let mut nstate = state;
    for (state_i, sbox_i) in nstate.iter_mut().zip(perm.sbox.iter_mut()) {
        generate_sbox(&mut sbox_i.sbox, state_i);
        sbox_i.post_sbox.write(*state_i);
    }

    if is_beginning {
        LinearLayers::external_linear_layer(&mut state);
        for i in 0..WIDTH {
            perm.end_state[i].write(state[i]);
        }
    } else if is_full {
        LinearLayers::external_linear_layer(&mut nstate);
        for i in 0..WIDTH {
            state[i] = nstate[i];
            perm.end_state[i].write(state[i]);
        }
    } else if is_partial {
        state[0] = nstate[0];
        LinearLayers::internal_linear_layer(&mut state);
        for i in 0..WIDTH {
            perm.end_state[i].write(state[i]);
        }
    }
    state
}

#[inline]
fn generate_sbox<F: PrimeField, const DEGREE: u64, const REGISTERS: usize>(
    sbox: &mut SBox<MaybeUninit<F>, DEGREE, REGISTERS>,
    x: &mut F,
) {
    *x = match (DEGREE, REGISTERS) {
        (3, 0) => x.cube(),
        (5, 0) => x.exp_const_u64::<5>(),
        (7, 0) => x.exp_const_u64::<7>(),
        (5, 1) => {
            let x2 = x.square();
            let x3 = x2 * *x;
            sbox.0[0].write(x3);
            x3 * x2
        }
        (7, 1) => {
            let x3 = x.cube();
            sbox.0[0].write(x3);
            x3 * x3 * *x
        }
        (11, 2) => {
            let x2 = x.square();
            let x3 = x2 * *x;
            let x9 = x3.cube();
            sbox.0[0].write(x3);
            sbox.0[1].write(x9);
            x9 * x2
        }
        _ => panic!(
            "Unexpected (DEGREE, REGISTERS) of ({}, {})",
            DEGREE, REGISTERS
        ),
    }
}
