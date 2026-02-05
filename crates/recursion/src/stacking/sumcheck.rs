use std::{
    borrow::{Borrow, BorrowMut},
    collections::HashSet,
};

use itertools::{Itertools, izip};
use openvm_circuit_primitives::{
    SubAir,
    utils::{and, assert_array_eq, not},
};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{
    BasedVectorSpace, Field, PrimeCharacteristicRing, PrimeField32, TwoAdicField,
    extension::BinomiallyExtendable,
};
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use p3_maybe_rayon::prelude::*;
use stark_backend_v2::{
    D_EF, EF, F,
    keygen::types::MultiStarkVerifyingKeyV2,
    poly_common::{eval_eq_uni, eval_eq_uni_at_one, interpolate_quadratic_at_012},
    proof::Proof,
};
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::{
    bus::{
        ConstraintSumcheckRandomness, ConstraintSumcheckRandomnessBus, TranscriptBus,
        TranscriptBusMessage, WhirOpeningPointBus, WhirOpeningPointMessage,
    },
    stacking::{
        bus::{
            EqBaseBus, EqBaseMessage, EqKernelLookupBus, EqKernelLookupMessage,
            EqRandValuesLookupBus, EqRandValuesLookupMessage, StackingModuleTidxBus,
            StackingModuleTidxMessage, SumcheckClaimsBus, SumcheckClaimsMessage,
        },
        utils::get_stacked_slice_data,
    },
    subairs::nested_for_loop::{NestedForLoopAuxCols, NestedForLoopIoCols, NestedForLoopSubAir},
    system::Preflight,
    utils::{
        assert_zeros, ext_field_add, ext_field_multiply, ext_field_multiply_scalar,
        ext_field_one_minus, ext_field_subtract,
    },
};

///////////////////////////////////////////////////////////////////////////
/// COLUMNS
///////////////////////////////////////////////////////////////////////////
#[repr(C)]
#[derive(AlignedBorrow)]
pub struct SumcheckRoundsCols<F> {
    // Proof index columns for continuations
    pub proof_idx: F,
    pub is_valid: F,
    pub is_first: F,
    pub is_last: F,

    // Sumcheck round this row represents
    pub round: F,

    // Starting tidx for this sumcheck round
    pub tidx: F,

    // Evaluations of polynomial s_round
    pub s_eval_at_0: [F; D_EF],
    pub s_eval_at_1: [F; D_EF],
    pub s_eval_at_2: [F; D_EF],
    pub s_eval_at_u: [F; D_EF],

    // Values of sampled u and r for this round
    pub u_round: [F; D_EF],
    pub r_round: [F; D_EF],
    pub has_r: F,
    pub u_mult: F,

    // Values of eq(u_0, r_0), eq(u_0, r_0 * omega), and eq(u_0, 1) * eq(r_0 * omega, 1)
    pub eq_prism_base: [F; D_EF],
    pub eq_cube_base: [F; D_EF],
    pub rot_cube_base: [F; D_EF],

    // Value of eq_cube
    pub eq_cube: [F; D_EF],

    // Intermediate values to compute rot_cube recursively
    pub r_not_u_prod: [F; D_EF],
    pub rot_cube_minus_prod: [F; D_EF],

    // Multiplicity of eq_round(u, r) lookup
    pub eq_rot_mult: F,
}

///////////////////////////////////////////////////////////////////////////
/// AIR
///////////////////////////////////////////////////////////////////////////
pub struct SumcheckRoundsAir {
    // External buses
    pub constraint_randomness_bus: ConstraintSumcheckRandomnessBus,
    pub whir_opening_point_bus: WhirOpeningPointBus,
    pub transcript_bus: TranscriptBus,

    // Internal buses
    pub stacking_tidx_bus: StackingModuleTidxBus,
    pub sumcheck_claims_bus: SumcheckClaimsBus,
    pub eq_base_bus: EqBaseBus,
    pub eq_rand_values_bus: EqRandValuesLookupBus,
    pub eq_kernel_lookup_bus: EqKernelLookupBus,

    pub l_skip: usize,
}

impl BaseAirWithPublicValues<F> for SumcheckRoundsAir {}
impl PartitionedBaseAir<F> for SumcheckRoundsAir {}

impl<F> BaseAir<F> for SumcheckRoundsAir {
    fn width(&self) -> usize {
        SumcheckRoundsCols::<F>::width()
    }
}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for SumcheckRoundsAir
where
    AB::F: PrimeField32,
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<D_EF>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (
            main.row_slice(0).expect("window should have two elements"),
            main.row_slice(1).expect("window should have two elements"),
        );

        let local: &SumcheckRoundsCols<AB::Var> = (*local).borrow();
        let next: &SumcheckRoundsCols<AB::Var> = (*next).borrow();

        NestedForLoopSubAir::<1, 0> {}.eval(
            builder,
            (
                (
                    NestedForLoopIoCols {
                        is_enabled: local.is_valid,
                        counter: [local.proof_idx],
                        is_first: [local.is_first],
                    }
                    .map_into(),
                    NestedForLoopIoCols {
                        is_enabled: next.is_valid,
                        counter: [next.proof_idx],
                        is_first: [next.is_first],
                    }
                    .map_into(),
                ),
                NestedForLoopAuxCols { is_transition: [] },
            ),
        );

        builder.assert_bool(local.is_valid);
        builder.assert_bool(local.is_first);
        builder.assert_bool(local.is_last);
        builder
            .when(and(local.is_valid, local.is_last))
            .assert_zero((local.proof_idx + AB::F::ONE - next.proof_idx) * next.proof_idx);
        builder
            .when(and(not(local.is_valid), local.is_last))
            .assert_zero(next.proof_idx);
        builder.when(local.is_first).assert_one(local.is_valid);

        /*
         * Constrain that round increments correctly.
         */
        builder.when(local.is_first).assert_one(local.round);
        builder
            .when(and(not(local.is_last), local.is_valid))
            .assert_eq(local.round + AB::Expr::ONE, next.round);

        /*
         * Constrain that s_round(u_round) is the quadratic interpolation using values
         * s_round(0), s_round(1), and s_round(2). Additionally, constrain that we have
         * s_round(u_round) = s_{round + 1}(0) + s_{round + 1}(1), and send the value of
         * s_{n_stack}(u_{n_stack}) to StackingClaimsAir.
         */
        self.sumcheck_claims_bus.receive(
            builder,
            local.proof_idx,
            SumcheckClaimsMessage {
                module_idx: AB::Expr::ONE,
                value: ext_field_add(local.s_eval_at_0, local.s_eval_at_1),
            },
            local.is_first,
        );

        let s1 = ext_field_subtract(local.s_eval_at_1, local.s_eval_at_0);
        let s2 = ext_field_subtract(local.s_eval_at_2, local.s_eval_at_1);
        let p = ext_field_multiply_scalar::<AB::Expr>(
            ext_field_subtract::<AB::Expr>(s2, s1.clone()),
            AB::F::TWO.inverse(),
        );
        let q = ext_field_subtract::<AB::Expr>(s1, p.clone());

        assert_array_eq(
            builder,
            ext_field_add(
                ext_field_multiply(
                    ext_field_add::<AB::Expr>(ext_field_multiply(p, local.u_round), q),
                    local.u_round,
                ),
                local.s_eval_at_0,
            ),
            local.s_eval_at_u,
        );

        assert_array_eq(
            &mut builder.when(not(local.is_last)),
            local.s_eval_at_u,
            ext_field_add(next.s_eval_at_0, next.s_eval_at_1),
        );

        self.sumcheck_claims_bus.send(
            builder,
            local.proof_idx,
            SumcheckClaimsMessage {
                module_idx: AB::Expr::TWO,
                value: local.s_eval_at_u.map(Into::into),
            },
            and(local.is_last, local.is_valid),
        );

        /*
         * Constrain the correctness of eq_cube and rot_cube at each round and provide the
         * lookups for eq_round(u, r) and k_rot_round(u, r). Computing rot_cube recursively
         * requires us to store the prefix product of r_round * (1 - u_round), which we
         * denote r_not_u_prod, and rot_cube - r_not_u_prod.
         */
        self.eq_base_bus.receive(
            builder,
            local.proof_idx,
            EqBaseMessage {
                eq_u_r: local.eq_prism_base,
                eq_u_r_omega: local.eq_cube_base,
                eq_u_r_prod: local.rot_cube_base,
            },
            local.is_first,
        );

        assert_array_eq(
            &mut builder.when(not(local.is_last)),
            local.eq_prism_base,
            next.eq_prism_base,
        );

        assert_array_eq(
            &mut builder.when(not(local.is_last)),
            local.eq_cube_base,
            next.eq_cube_base,
        );

        assert_array_eq(
            &mut builder.when(not(local.is_last)),
            local.rot_cube_base,
            next.rot_cube_base,
        );

        let local_u_not_r = ext_field_multiply(local.u_round, ext_field_one_minus(local.r_round));
        let local_r_not_u = ext_field_multiply(local.r_round, ext_field_one_minus(local.u_round));
        let next_u_not_r = ext_field_multiply(next.u_round, ext_field_one_minus(next.r_round));
        let next_r_not_u = ext_field_multiply(next.r_round, ext_field_one_minus(next.u_round));

        assert_array_eq(
            &mut builder.when(local.is_first),
            ext_field_one_minus::<AB::Expr>(ext_field_add::<AB::Expr>(
                local_u_not_r.clone(),
                local_r_not_u.clone(),
            )),
            local.eq_cube,
        );

        assert_array_eq(
            &mut builder.when(not(local.is_last)),
            ext_field_multiply(
                local.eq_cube,
                ext_field_one_minus::<AB::Expr>(ext_field_add::<AB::Expr>(
                    next_u_not_r.clone(),
                    next_r_not_u.clone(),
                )),
            ),
            next.eq_cube,
        );

        assert_array_eq(
            &mut builder.when(local.is_first),
            local.r_not_u_prod,
            local_r_not_u,
        );

        assert_array_eq(
            &mut builder.when(not(local.is_last)),
            ext_field_multiply(local.r_not_u_prod, next_r_not_u.clone()),
            next.r_not_u_prod,
        );

        assert_array_eq(
            &mut builder.when(local.is_first),
            local.rot_cube_minus_prod,
            local_u_not_r,
        );

        assert_array_eq(
            &mut builder.when(not(local.is_last)),
            ext_field_add::<AB::Expr>(
                ext_field_multiply(
                    local.rot_cube_minus_prod,
                    ext_field_one_minus::<AB::Expr>(ext_field_add::<AB::Expr>(
                        next_u_not_r.clone(),
                        next_r_not_u,
                    )),
                ),
                ext_field_multiply(next_u_not_r, local.r_not_u_prod),
            ),
            next.rot_cube_minus_prod,
        );

        self.eq_kernel_lookup_bus.send(
            builder,
            local.proof_idx,
            EqKernelLookupMessage {
                n: local.round.into(),
                eq_in: ext_field_multiply(local.eq_prism_base, local.eq_cube),
                k_rot_in: ext_field_add(
                    ext_field_multiply(local.eq_cube_base, local.eq_cube),
                    ext_field_multiply(
                        local.rot_cube_base,
                        ext_field_subtract(
                            ext_field_add(local.r_not_u_prod, local.rot_cube_minus_prod),
                            local.eq_cube,
                        ),
                    ),
                ),
            },
            local.is_valid * local.eq_rot_mult,
        );

        builder.assert_bool(local.has_r);
        builder
            .when(not(local.has_r))
            .assert_zero(local.eq_rot_mult);

        assert_zeros(&mut builder.when(not(local.has_r)), local.r_round);

        /*
         * Because we sample u_round and r_round from the transcript here, we send
         * them to other AIRs that need to use it.
         */
        self.constraint_randomness_bus.receive(
            builder,
            local.proof_idx,
            ConstraintSumcheckRandomness {
                idx: local.round,
                challenge: local.r_round,
            },
            and(local.is_valid, local.has_r),
        );

        self.whir_opening_point_bus.send(
            builder,
            local.proof_idx,
            WhirOpeningPointMessage {
                idx: local.round + AB::Expr::from_usize(self.l_skip - 1),
                value: local.u_round.map(Into::into),
            },
            local.is_valid,
        );

        self.eq_rand_values_bus.send(
            builder,
            local.proof_idx,
            EqRandValuesLookupMessage {
                idx: local.round,
                u: local.u_round,
            },
            local.u_mult,
        );
        builder.when(not(local.is_valid)).assert_zero(local.u_mult);

        /*
         * Constrain transcript operations and send the final tidx to StackingClaimsAir.
         */
        self.stacking_tidx_bus.receive(
            builder,
            local.proof_idx,
            StackingModuleTidxMessage {
                module_idx: AB::Expr::ONE,
                tidx: local.tidx.into(),
            },
            local.is_first,
        );

        builder
            .when(not(local.is_last) * local.is_valid)
            .assert_eq(local.tidx + AB::F::from_usize(3 * D_EF), next.tidx);

        for i in 0..D_EF {
            self.transcript_bus.receive(
                builder,
                local.proof_idx,
                TranscriptBusMessage {
                    tidx: AB::Expr::from_usize(i) + local.tidx,
                    value: local.s_eval_at_1[i].into(),
                    is_sample: AB::Expr::ZERO,
                },
                local.is_valid,
            );

            self.transcript_bus.receive(
                builder,
                local.proof_idx,
                TranscriptBusMessage {
                    tidx: AB::Expr::from_usize(i + D_EF) + local.tidx,
                    value: local.s_eval_at_2[i].into(),
                    is_sample: AB::Expr::ZERO,
                },
                local.is_valid,
            );

            self.transcript_bus.receive(
                builder,
                local.proof_idx,
                TranscriptBusMessage {
                    tidx: AB::Expr::from_usize(i + 2 * D_EF) + local.tidx,
                    value: local.u_round[i].into(),
                    is_sample: AB::Expr::ONE,
                },
                local.is_valid,
            );
        }

        self.stacking_tidx_bus.send(
            builder,
            local.proof_idx,
            StackingModuleTidxMessage {
                module_idx: AB::Expr::TWO,
                tidx: AB::Expr::from_usize(3 * D_EF) + local.tidx,
            },
            and(local.is_last, local.is_valid),
        );
    }
}

///////////////////////////////////////////////////////////////////////////
/// TRACE GENERATOR
///////////////////////////////////////////////////////////////////////////
pub struct SumcheckRoundsTraceGenerator;

impl SumcheckRoundsTraceGenerator {
    #[tracing::instrument(level = "trace", skip_all)]
    pub fn generate_trace(
        vk: &MultiStarkVerifyingKeyV2,
        proofs: &[&Proof],
        preflights: &[&Preflight],
    ) -> RowMajorMatrix<F> {
        debug_assert_eq!(proofs.len(), preflights.len());

        let width = SumcheckRoundsCols::<usize>::width();

        if proofs.is_empty() {
            return RowMajorMatrix::new(vec![F::ZERO; width], width);
        }

        let traces = proofs
            .par_iter()
            .zip(preflights.par_iter())
            .enumerate()
            .map(|(proof_idx, (proof, preflight))| {
                let sumcheck_rounds = &proof.stacking_proof.sumcheck_round_polys;

                let eq_mults = {
                    let mut eq_mults = vec![0usize; vk.inner.params.n_stack];
                    for (sort_idx, (_, vdata)) in
                        preflight.proof_shape.sorted_trace_vdata.iter().enumerate()
                    {
                        if vdata.log_height > vk.inner.params.l_skip {
                            let n = vdata.log_height - vk.inner.params.l_skip;
                            eq_mults[n - 1] += proof.batch_constraint_proof.column_openings
                                [sort_idx]
                                .iter()
                                .flatten()
                                .collect_vec()
                                .len();
                        }
                    }
                    eq_mults
                };

                let u_mults = {
                    let mut u_mults = vec![0usize; vk.inner.params.n_stack];
                    let stacked_slices =
                        get_stacked_slice_data(vk, &preflight.proof_shape.sorted_trace_vdata);

                    let mut b_value_set = HashSet::<(usize, usize)>::new();
                    for slice in stacked_slices {
                        let n_lift = slice.n.max(0) as usize;
                        let b_value = slice.row_idx >> (n_lift + vk.inner.params.l_skip);
                        let total_num_bits = vk.inner.params.n_stack - n_lift;

                        for num_bits in (1..=total_num_bits).rev() {
                            let shifted_b_value = b_value >> (total_num_bits - num_bits);
                            if b_value_set.insert((shifted_b_value, num_bits)) {
                                u_mults[vk.inner.params.n_stack - num_bits] += 1;
                            } else {
                                break;
                            }
                        }
                    }
                    u_mults
                };

                let (eq_prism_base, eq_cube_base, rot_cube_base) = {
                    let l_skip = vk.inner.params.l_skip;
                    let omega = F::two_adic_generator(l_skip);
                    let u = preflight.stacking.sumcheck_rnd[0];
                    let r = preflight.batch_constraint.sumcheck_rnd[0];

                    let eq_prism_base = eval_eq_uni(l_skip, u, r);
                    let eq_cube_base = eval_eq_uni(l_skip, u, r * omega);
                    let rot_cube_base =
                        eval_eq_uni_at_one(l_skip, u) * eval_eq_uni_at_one(l_skip, r * omega);
                    (eq_prism_base, eq_cube_base, rot_cube_base)
                };

                let num_rows = sumcheck_rounds.len();
                let proof_idx_value = F::from_usize(proof_idx);

                let mut trace = vec![F::ZERO; num_rows * width];

                for chunk in trace.chunks_mut(width) {
                    let cols: &mut SumcheckRoundsCols<F> = chunk.borrow_mut();
                    cols.proof_idx = proof_idx_value;
                }

                let u = &preflight.stacking.sumcheck_rnd[1..];
                let batch_sumcheck_randomness = preflight.batch_constraint_sumcheck_randomness();
                let r = &batch_sumcheck_randomness[1..];

                let initial_tidx = preflight.stacking.intermediate_tidx[1];

                let mut s_eval_at_u = preflight.stacking.univariate_poly_rand_eval;

                let mut eq_cube = EF::ONE;
                let mut r_not_u_prod = EF::ONE;
                let mut rot_cube_minus_prod = EF::ZERO;

                for (round, (sumcheck_round, chunk, &u_round)) in
                    izip!(sumcheck_rounds.iter(), trace.chunks_mut(width), u.iter()).enumerate()
                {
                    let cols: &mut SumcheckRoundsCols<F> = chunk.borrow_mut();

                    let s_eval_at_0 = s_eval_at_u - sumcheck_round[0];
                    s_eval_at_u = interpolate_quadratic_at_012(
                        &[s_eval_at_0, sumcheck_round[0], sumcheck_round[1]],
                        u_round,
                    );

                    cols.proof_idx = proof_idx_value;
                    cols.is_valid = F::ONE;
                    cols.is_first = F::from_bool(round == 0);
                    cols.is_last = F::from_bool(round + 1 == num_rows);

                    cols.round = F::from_usize(round + 1);
                    cols.tidx = F::from_usize(initial_tidx + (3 * D_EF * round));

                    cols.s_eval_at_0
                        .copy_from_slice(s_eval_at_0.as_basis_coefficients_slice());
                    cols.s_eval_at_1
                        .copy_from_slice(sumcheck_round[0].as_basis_coefficients_slice());
                    cols.s_eval_at_2
                        .copy_from_slice(sumcheck_round[1].as_basis_coefficients_slice());
                    cols.s_eval_at_u
                        .copy_from_slice(s_eval_at_u.as_basis_coefficients_slice());

                    cols.u_round
                        .copy_from_slice(u_round.as_basis_coefficients_slice());
                    let r_round = if round < r.len() {
                        cols.r_round = r[round].challenge;
                        cols.has_r = F::ONE;
                        EF::from_basis_coefficients_iter(r[round].challenge.into_iter()).unwrap()
                    } else {
                        EF::ZERO
                    };
                    cols.u_mult = F::from_usize(u_mults[round]);

                    cols.eq_prism_base
                        .copy_from_slice(eq_prism_base.as_basis_coefficients_slice());
                    cols.eq_cube_base
                        .copy_from_slice(eq_cube_base.as_basis_coefficients_slice());
                    cols.rot_cube_base
                        .copy_from_slice(rot_cube_base.as_basis_coefficients_slice());

                    let u_not_r = u_round * (EF::ONE - r_round);
                    let r_not_u = r_round * (EF::ONE - u_round);
                    let next_eq_term = EF::ONE - (u_not_r + r_not_u);
                    eq_cube *= next_eq_term;
                    cols.eq_cube
                        .copy_from_slice(eq_cube.as_basis_coefficients_slice());

                    rot_cube_minus_prod =
                        (rot_cube_minus_prod * next_eq_term) + u_not_r * r_not_u_prod;
                    r_not_u_prod *= r_not_u;
                    cols.r_not_u_prod
                        .copy_from_slice(r_not_u_prod.as_basis_coefficients_slice());
                    cols.rot_cube_minus_prod
                        .copy_from_slice(rot_cube_minus_prod.as_basis_coefficients_slice());

                    cols.eq_rot_mult = F::from_usize(eq_mults[round]);
                }

                (trace, num_rows)
            })
            .collect::<Vec<_>>();

        let total_rows: usize = traces.iter().map(|(_trace, rows)| *rows).sum();
        let padded_rows = total_rows.next_power_of_two();
        let mut combined_trace = Vec::with_capacity(padded_rows * width);
        for (trace, _num_rows) in traces {
            combined_trace.extend(trace);
        }

        if padded_rows > total_rows {
            let padding_start = combined_trace.len();
            combined_trace.resize(padded_rows * width, F::ZERO);

            let padding_proof_idx = F::from_usize(proofs.len());
            let mut chunks = combined_trace[padding_start..].chunks_mut(width);
            let num_padded_rows = padded_rows - total_rows;
            for i in 0..num_padded_rows {
                let chunk = chunks.next().unwrap();
                let cols: &mut SumcheckRoundsCols<F> = chunk.borrow_mut();
                cols.proof_idx = padding_proof_idx;
                if i + 1 == num_padded_rows {
                    cols.is_last = F::ONE;
                }
            }
        }

        RowMajorMatrix::new(combined_trace, width)
    }
}
