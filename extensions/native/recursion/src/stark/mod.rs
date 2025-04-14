use std::{iter::zip, marker::PhantomData};

use itertools::Itertools;
use openvm_circuit::arch::instructions::program::Program;
use openvm_native_compiler::{
    conversion::CompilerOptions,
    ir::{Array, ArrayLike, Builder, Config, DslIr, Ext, ExtConst, Felt, SymbolicExt, Usize, Var},
    prelude::RVar,
};
use openvm_native_compiler_derive::iter_zip;
use openvm_stark_backend::{
    air_builders::symbolic::SymbolicExpressionDag,
    p3_commit::LagrangeSelectors,
    p3_field::{FieldAlgebra, FieldExtensionAlgebra, TwoAdicField},
    p3_matrix::{dense::RowMajorMatrixView, stack::VerticalPair},
    proof::{AdjacentOpenedValues, Proof},
    verifier::GenericVerifierConstraintFolder,
};
use openvm_stark_sdk::{
    config::{baby_bear_poseidon2::BabyBearPoseidon2Config, FriParameters},
    p3_baby_bear::BabyBear,
};

use crate::{
    challenger::{duplex::DuplexChallengerVariable, ChallengerVariable},
    commit::{PcsVariable, PolynomialSpaceVariable},
    folder::RecursiveVerifierConstraintFolder,
    fri::{
        types::{TwoAdicPcsMatsVariable, TwoAdicPcsRoundVariable},
        TwoAdicFriPcsVariable, TwoAdicMultiplicativeCosetVariable, MAX_TWO_ADICITY,
    },
    hints::Hintable,
    types::{InnerConfig, MultiStarkVerificationAdvice, StarkVerificationAdvice},
    utils::const_fri_config,
    vars::{
        AdjacentOpenedValuesVariable, AirProofDataVariable, CommitmentsVariable,
        StarkProofVariable, TraceHeightConstraintSystem,
    },
    view::get_advice_per_air,
};

#[cfg(feature = "static-verifier")]
pub mod outer;

#[derive(Debug, Clone, Copy)]
pub struct VerifierProgram<C: Config> {
    _phantom: PhantomData<C>,
}

impl VerifierProgram<InnerConfig> {
    /// Create a new instance of the program for the
    /// [`openvm_stark_sdk::config::baby_bear_poseidon2`]
    pub fn build(
        constants: MultiStarkVerificationAdvice<InnerConfig>,
        fri_params: &FriParameters,
    ) -> Program<BabyBear> {
        let options = CompilerOptions {
            enable_cycle_tracker: true,
            ..Default::default()
        };
        Self::build_with_options(constants, fri_params, options)
    }

    /// Create a new instance of the program for the
    /// [`openvm_stark_sdk::config::baby_bear_poseidon2`]
    pub fn build_with_options(
        constants: MultiStarkVerificationAdvice<InnerConfig>,
        fri_params: &FriParameters,
        options: CompilerOptions,
    ) -> Program<BabyBear> {
        let mut builder = Builder::<InnerConfig>::default();

        builder.cycle_tracker_start("VerifierProgram");
        builder.cycle_tracker_start("ReadingProofFromInput");
        let input: StarkProofVariable<_> = builder.uninit();
        Proof::<BabyBearPoseidon2Config>::witness(&input, &mut builder);
        builder.cycle_tracker_end("ReadingProofFromInput");

        builder.cycle_tracker_start("InitializePcsConst");
        let pcs = TwoAdicFriPcsVariable {
            config: const_fri_config(&mut builder, fri_params),
        };
        builder.cycle_tracker_end("InitializePcsConst");
        StarkVerifier::verify::<DuplexChallengerVariable<_>>(
            &mut builder,
            &pcs,
            &constants,
            &input,
        );

        builder.cycle_tracker_end("VerifierProgram");
        builder.halt();

        builder.compile_isa_with_options(options)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct StarkVerifier<C: Config> {
    _phantom: PhantomData<C>,
}

impl<C: Config> StarkVerifier<C>
where
    C::F: TwoAdicField,
{
    /// Reference: [openvm_stark_backend::verifier::MultiTraceStarkVerifier::verify].
    pub fn verify<CH: ChallengerVariable<C>>(
        builder: &mut Builder<C>,
        pcs: &TwoAdicFriPcsVariable<C>,
        m_advice: &MultiStarkVerificationAdvice<C>,
        proof: &StarkProofVariable<C>,
    ) {
        if builder.flags.static_only {
            let mut challenger = CH::new(builder);
            Self::verify_raps(builder, pcs, m_advice, &mut challenger, proof);
        } else {
            // Recycle stack space after verifying
            let mut tmp_builder = builder.create_sub_builder();
            // Recycle heap space after verifying by resetting the heap pointer.
            let old_heap_ptr = tmp_builder.load_heap_ptr();
            let mut challenger = CH::new(&mut tmp_builder);
            Self::verify_raps(&mut tmp_builder, pcs, m_advice, &mut challenger, proof);
            tmp_builder.store_heap_ptr(old_heap_ptr);
            builder.operations.extend(tmp_builder.operations);
        }
    }

    /// Reference: [openvm_stark_backend::verifier::MultiTraceStarkVerifier::verify_raps].
    pub fn verify_raps(
        builder: &mut Builder<C>,
        pcs: &TwoAdicFriPcsVariable<C>,
        m_advice: &MultiStarkVerificationAdvice<C>,
        challenger: &mut impl ChallengerVariable<C>,
        proof: &StarkProofVariable<C>,
    ) where
        C::F: TwoAdicField,
        C::EF: TwoAdicField,
    {
        let pre_hash = builder.constant(m_advice.pre_hash.clone());
        challenger.observe_digest(builder, pre_hash);
        let air_ids = proof.get_air_ids(builder);
        let num_airs = cast_usize_to_felt(builder, air_ids.len());
        challenger.observe(builder, num_airs);
        iter_zip!(builder, air_ids).for_each(|ptr_vec, builder| {
            let air_id = builder.iter_ptr_get(&air_ids, ptr_vec[0]);
            let air_id = cast_usize_to_felt(builder, air_id);
            challenger.observe(builder, air_id);
        });
        let m_advice_var = get_advice_per_air(builder, m_advice, &air_ids);
        // (T01a): `air_ids` is a subsequence of `0..(m_advice.per_air.len())`.

        let StarkProofVariable::<C> {
            commitments,
            opening,
            per_air: air_proofs,
            air_perm_by_height,
            log_up_pow_witness,
        } = proof;

        if m_advice.num_challenges_to_sample.len() > 1 {
            panic!("Only support 0 or 1 phase is supported");
        }

        let num_airs = RVar::from(air_proofs.len());
        let num_challenges_to_sample = m_advice_var.num_challenges_to_sample(builder);
        // Currently only support 0 or 1 phase is supported.
        // (T01b): `num_challenges_to_sample.len() < 2`.

        let num_phases = RVar::from(num_challenges_to_sample.len());
        // Here the shape of `exposed_values_after_challenge` is not verified. But it's verified
        // later (T01c).
        assert_cumulative_sums(builder, air_proofs, &num_challenges_to_sample);

        let air_perm_by_height = if builder.flags.static_only {
            builder.assert_usize_eq(num_airs, RVar::from(m_advice.per_air.len()));
            let num_airs = num_airs.value();
            let perm = (0..num_airs).map(|i| builder.eval(RVar::from(i))).collect();
            &builder.vec(perm)
        } else {
            builder.assert_usize_eq(air_perm_by_height.len(), num_airs);
            // Assert that each index in `air_perm_by_height` is unique and in range [0, num_airs).
            let mask: Array<_, Usize<_>> = builder.dyn_array(num_airs);
            let one: Usize<_> = builder.eval(C::N::ONE);
            iter_zip!(builder, air_perm_by_height).for_each(|ptr_vec, builder| {
                let perm_i = builder.iter_ptr_get(air_perm_by_height, ptr_vec[0]);
                builder.assert_less_than_slow_small_rhs(perm_i.clone(), num_airs);
                builder.set_value(&mask, perm_i.clone(), one.clone());
            });
            // Check that each index of mask was set, i.e., that `air_perm_by_height` is a
            // permutation. Also check that permutation is decreasing by height.
            let prev_log_height_plus_one: Usize<_> =
                builder.eval(RVar::from(MAX_TWO_ADICITY - pcs.config.log_blowup + 1));
            iter_zip!(builder, air_perm_by_height).for_each(|ptr_vec, builder| {
                let perm_i = builder.iter_ptr_get(air_perm_by_height, ptr_vec[0]);
                let mask_i = builder.get(&mask, perm_i.clone());
                builder.assert_usize_eq(mask_i, one.clone());

                let air_proof = builder.get(air_proofs, perm_i.clone());
                builder.assert_less_than_slow_small_rhs(
                    air_proof.log_degree.clone(),
                    prev_log_height_plus_one.clone(),
                );
                builder.assign(
                    &prev_log_height_plus_one,
                    air_proof.log_degree.clone() + RVar::one(),
                );
            });
            air_perm_by_height
        };
        // (T02a): `air_perm_by_height` is a valid permutation of `0..num_airs`.
        // (T02b): For all `i`, `air_proofs[i].log_degree <= MAX_TWO_ADICITY - log_blowup`.
        // (T02c): For all `0<=i<num_air-1`, `air_proofs[air_perm_by_height[i]].log_degree >=
        // air_proofs[air_perm_by_height[i+1]].log_degree`.
        let log_max_height = {
            let index = builder.get(air_perm_by_height, RVar::zero());
            let air_proof = builder.get(air_proofs, index);
            RVar::from(air_proof.log_degree.clone())
        };

        // OK: trace_height_constraint_system comes from vkey so requirements of
        // `check_trace_height_constraints` are met.
        builder.cycle_tracker_start("CheckTraceHeightConstraints");
        Self::check_trace_height_constraints(
            builder,
            &m_advice_var.trace_height_constraint_system,
            air_proofs,
        );
        builder.cycle_tracker_end("CheckTraceHeightConstraints");

        builder.range(0, num_airs).for_each(|i_vec, builder| {
            let i = i_vec[0];
            let air_proof_data = builder.get(air_proofs, i);
            let pvs = air_proof_data.public_values;
            let air_advice = builder.get(&m_advice_var.per_air, i);
            builder.assert_usize_eq(air_advice.num_public_values, pvs.len());
            challenger.observe_slice(builder, pvs);
        });
        // (T03a): shapes of public values in `air_proofs` have been validated.

        builder.cycle_tracker_start("stage-c-build-rounds");

        // Count the number of main trace commitments together to save a loop.
        let num_cached_mains: Usize<_> = builder.eval(RVar::zero());
        let num_common_main_traces: Usize<_> = builder.eval(RVar::zero());
        let num_after_challenge_traces: Usize<_> = builder.eval(RVar::zero());
        iter_zip!(builder, m_advice_var.per_air).for_each(|ptr_vec, builder| {
            let air_advice = builder.iter_ptr_get(&m_advice_var.per_air, ptr_vec[0]);
            builder
                .if_eq(air_advice.preprocessed_data.len(), RVar::one())
                .then(|builder| {
                    let commit = builder.get(&air_advice.preprocessed_data, RVar::zero());
                    challenger.observe_digest(builder, commit);
                });

            builder.assign(
                &num_cached_mains,
                num_cached_mains.clone() + air_advice.width.cached_mains.len(),
            );
            builder
                .if_ne(air_advice.width.common_main, RVar::zero())
                .then(|builder| {
                    builder.assign(
                        &num_common_main_traces,
                        num_common_main_traces.clone() + RVar::one(),
                    );
                });
            builder
                .if_ne(air_advice.width.after_challenge.len(), RVar::zero())
                .then(|builder| {
                    builder.assign(
                        &num_after_challenge_traces,
                        num_after_challenge_traces.clone() + RVar::one(),
                    );
                });
        });
        let num_cached_mains = RVar::from(num_cached_mains);
        let num_common_main_traces = RVar::from(num_common_main_traces);
        let num_after_challenge_traces = RVar::from(num_after_challenge_traces);

        let num_main_commits: Usize<_> = builder.eval(num_cached_mains + RVar::one());
        let num_main_commits = RVar::from(num_main_commits);

        let CommitmentsVariable {
            main_trace: main_trace_commits,
            after_challenge: after_challenge_commits,
            quotient: quotient_commit,
        } = commitments;

        // Observe main trace commitments
        builder.assert_usize_eq(main_trace_commits.len(), num_main_commits);
        // (T04a): shapes of `main_trace_commits` have been validated.
        iter_zip!(builder, main_trace_commits).for_each(|ptr_vec, builder| {
            let main_commit = builder.iter_ptr_get(main_trace_commits, ptr_vec[0]);
            challenger.observe_digest(builder, main_commit);
        });

        iter_zip!(builder, air_proofs).for_each(|ptr_vec, builder| {
            let air_proof = builder.iter_ptr_get(air_proofs, ptr_vec[0]);
            let log_degree = cast_usize_to_felt(builder, air_proof.log_degree);
            challenger.observe(builder, log_degree);
        });

        let challenges_per_phase = builder.array(num_phases);

        // Assumption: (T01b) `num_phases` is 0 or 1.
        builder.if_eq(num_phases, RVar::one()).then(|builder| {
            // Proof of work phase.
            challenger.check_witness(builder, m_advice.log_up_pow_bits, *log_up_pow_witness);

            let phase_idx = RVar::zero();
            let num_to_sample = RVar::from(2);
            let provided_num_to_sample = builder.get(&num_challenges_to_sample, phase_idx);
            builder.assert_usize_eq(provided_num_to_sample, num_to_sample);

            let challenges: Array<C, Ext<C::F, C::EF>> = builder.array(num_to_sample);
            // Sample challenges needed in this phase.
            builder.range(0, num_to_sample).for_each(|i_vec, builder| {
                let challenge = challenger.sample_ext(builder);
                builder.set_value(&challenges, i_vec[0], challenge);
            });
            builder.set_value(&challenges_per_phase, phase_idx, challenges);

            builder.range(0, num_airs).for_each(|j_vec, builder| {
                let j = j_vec[0];
                let air_advice = builder.get(&m_advice_var.per_air, j);
                builder
                    .if_ne(
                        air_advice.num_exposed_values_after_challenge.len(),
                        RVar::zero(),
                    )
                    .then(|builder| {
                        // Only support 1 challenge phase
                        builder.assert_usize_eq(
                            air_advice.num_exposed_values_after_challenge.len(),
                            RVar::one(),
                        );
                        let air_proof_data = builder.get(&proof.per_air, j);
                        let exposed_values = air_proof_data.exposed_values_after_challenge;
                        let values = builder.get(&exposed_values, phase_idx);
                        let values_len =
                            builder.get(&air_advice.num_exposed_values_after_challenge, phase_idx);
                        // (T01c): shape of `exposed_values_after_challenge` is verified
                        builder.assert_usize_eq(values_len, values.len());

                        iter_zip!(builder, values).for_each(|ptr_vec, builder| {
                            let value = builder.iter_ptr_get(&values, ptr_vec[0]);
                            let felts = builder.ext2felt(value);
                            challenger.observe_slice(builder, felts);
                        });
                    });
            });

            // Observe single commitment to all trace matrices in this phase.
            builder.assert_usize_eq(after_challenge_commits.len(), RVar::one());
            let commit = builder.get(after_challenge_commits, phase_idx);
            challenger.observe_digest(builder, commit);
        });

        let alpha = challenger.sample_ext(builder);

        challenger.observe_digest(builder, quotient_commit.clone());

        let zeta = challenger.sample_ext(builder);

        let num_prep_rounds: Usize<_> = builder.eval(RVar::zero());

        // Build domains
        let domains = builder.array(num_airs);
        let quotient_domains = builder.array(num_airs);
        let trace_points_per_domain = builder.array(num_airs);
        let quotient_chunk_domains = builder.array(num_airs);
        let num_quotient_mats: Usize<_> = builder.eval(RVar::zero());
        builder.range(0, num_airs).for_each(|i_vec, builder| {
            let i = i_vec[0];
            let air_proof = builder.get(air_proofs, i);
            let log_degree: RVar<_> = air_proof.log_degree.clone().into();
            let advice = builder.get(&m_advice_var.per_air, i);

            // Assumption: (T02b) `log_degree < MAX_TWO_ADICITY`
            let domain = pcs.natural_domain_for_log_degree(builder, log_degree);

            let trace_points = builder.array::<Ext<_, _>>(2);
            let zeta_next = domain.next_point(builder, zeta);
            builder.set_value(&trace_points, RVar::zero(), zeta);
            builder.set_value(&trace_points, RVar::one(), zeta_next);

            let log_quotient_degree = RVar::from(advice.log_quotient_degree);
            let quotient_degree =
                RVar::from(builder.sll::<Usize<_>>(RVar::one(), log_quotient_degree));
            let log_quotient_size = builder.eval_expr(log_degree + log_quotient_degree);
            // Assumption: (T02b) `log_degree <= MAX_TWO_ADICITY - low_blowup`
            // Because the VK ensures `log_quotient_degree <= log_blowup`, this won't access an out
            // of bound index.
            let quotient_domain =
                domain.create_disjoint_domain(builder, log_quotient_size, Some(pcs.config.clone()));
            builder.set_value(&quotient_domains, i, quotient_domain.clone());

            let qc_domains =
                quotient_domain.split_domains(builder, log_quotient_degree, quotient_degree);
            builder.assign(
                &num_quotient_mats,
                num_quotient_mats.clone() + quotient_degree,
            );

            builder.set_value(&domains, i, domain);
            builder.set_value(&trace_points_per_domain, i, trace_points);
            builder.set_value(&quotient_chunk_domains, i, qc_domains);

            builder
                .if_eq(advice.preprocessed_data.len(), RVar::one())
                .then(|builder| {
                    builder.assign(&num_prep_rounds, num_prep_rounds.clone() + RVar::one());
                });
        });
        let num_quotient_mats = RVar::from(num_quotient_mats);

        // Build the opening rounds

        // <Number of main trace commitments> = <number of cached main traces> + 1
        // All common main traces are committed together.
        let num_main_rounds = builder.eval_expr(num_cached_mains + RVar::one());
        let num_challenge_rounds: RVar<_> = num_challenges_to_sample.len().into();
        let num_quotient_rounds = RVar::one();

        builder.assert_usize_eq(opening.values.preprocessed.len(), num_prep_rounds.clone());
        // T05a: The shape of `opening.values.preprocessed` has been validated.

        let total_rounds = builder.eval_expr(
            num_prep_rounds + num_main_rounds + num_challenge_rounds + num_quotient_rounds,
        );

        let rounds = builder.array::<TwoAdicPcsRoundVariable<_>>(total_rounds);
        // For rounds which don't need permutation
        let null_perm = builder.array(0);

        // 1. First the preprocessed trace openings: one round per AIR with preprocessing.
        let round_idx: Usize<_> = builder.eval(RVar::zero());
        builder.range(0, num_airs).for_each(|i_vec, builder| {
            let i = i_vec[0];
            let advice = builder.get(&m_advice_var.per_air, i);
            builder
                .if_eq(advice.preprocessed_data.len(), RVar::one())
                .then(|builder| {
                    // Assumption: (T05a) `opening.values.preprocessed` has been validated.
                    let prep = builder.get(&opening.values.preprocessed, round_idx.clone());
                    let batch_commit = builder.get(&advice.preprocessed_data, RVar::zero());
                    let prep_width =
                        RVar::from(builder.get(&advice.width.preprocessed, RVar::zero()));

                    let domain = builder.get(&domains, i);
                    let trace_points = builder.get(&trace_points_per_domain, i);

                    builder.assert_usize_eq(prep_width, prep.local.len());
                    builder.assert_usize_eq(prep_width, prep.next.len());
                    // Assumption: each AIR with preprocessed trace has its own commitment and
                    // opening values
                    let values = builder.array::<Array<C, _>>(2);
                    builder.set_value(&values, 0, prep.local);
                    builder.set_value(&values, 1, prep.next);
                    let prep_mat = TwoAdicPcsMatsVariable::<C> {
                        domain,
                        values,
                        points: trace_points.clone(),
                    };

                    let mats: Array<_, TwoAdicPcsMatsVariable<_>> = builder.array(1);
                    builder.set_value(&mats, 0, prep_mat);

                    builder.set_value(
                        &rounds,
                        round_idx.clone(),
                        TwoAdicPcsRoundVariable {
                            batch_commit,
                            mats,
                            permutation: null_perm.clone(),
                        },
                    );
                    builder.assign(&round_idx, round_idx.clone() + RVar::one());
                });
        });

        // 2. Then the main trace openings.
        let main_commit_idx: Usize<_> = builder.eval(RVar::zero());
        builder.assert_usize_eq(opening.values.main.len(), num_main_commits);
        let common_main_values_per_mat = builder.get(&opening.values.main, num_cached_mains);
        let common_main_mats = builder.array(num_common_main_traces);
        let common_main_matrix_idx: Usize<_> = builder.eval(RVar::zero());
        builder.range(0, num_airs).for_each(|i_vec, builder| {
            let i = i_vec[0];
            let advice = builder.get(&m_advice_var.per_air, i);
            let cached_main_widths = &advice.width.cached_mains;

            let domain = builder.get(&domains, i);
            let trace_points = builder.get(&trace_points_per_domain, i);

            iter_zip!(builder, cached_main_widths).for_each(|ptr_vec, builder| {
                let cached_main_width = builder.iter_ptr_get(cached_main_widths, ptr_vec[0]);
                let values_per_mat = builder.get(&opening.values.main, main_commit_idx.clone());
                let batch_commit = builder.get(main_trace_commits, main_commit_idx.clone());
                builder.assign(&main_commit_idx, main_commit_idx.clone() + RVar::one());

                builder.assert_usize_eq(values_per_mat.len(), RVar::one());
                let main = builder.get(&values_per_mat, RVar::zero());
                let values = builder.array::<Array<C, _>>(2);
                builder.assert_usize_eq(main.local.len(), cached_main_width.clone());
                builder.assert_usize_eq(main.next.len(), cached_main_width);
                builder.set_value(&values, 0, main.local);
                builder.set_value(&values, 1, main.next);
                let main_mat = TwoAdicPcsMatsVariable::<C> {
                    domain: domain.clone(),
                    values,
                    points: trace_points.clone(),
                };
                let mats = builder.array(1);
                builder.set_value(&mats, 0, main_mat);

                builder.set_value(
                    &rounds,
                    round_idx.clone(),
                    TwoAdicPcsRoundVariable {
                        batch_commit,
                        mats,
                        permutation: null_perm.clone(),
                    },
                );
                builder.assign(&round_idx, round_idx.clone() + RVar::one());
            });

            let common_main_width = RVar::from(advice.width.common_main);
            builder
                .if_ne(common_main_width, RVar::zero())
                .then(|builder| {
                    // common_main_mats
                    let main =
                        builder.get(&common_main_values_per_mat, common_main_matrix_idx.clone());

                    let values = builder.array::<Array<C, _>>(2);
                    builder.assert_usize_eq(main.local.len(), common_main_width);
                    builder.assert_usize_eq(main.next.len(), common_main_width);
                    builder.set_value(&values, 0, main.local);
                    builder.set_value(&values, 1, main.next);
                    let main_mat = TwoAdicPcsMatsVariable::<C> {
                        domain: domain.clone(),
                        values,
                        points: trace_points.clone(),
                    };
                    builder.set_value(&common_main_mats, common_main_matrix_idx.clone(), main_mat);
                    builder.assign(
                        &common_main_matrix_idx,
                        common_main_matrix_idx.clone() + RVar::one(),
                    );
                });
        });
        // T05b: the shape of `opening.values.main` has been validated
        {
            let batch_commit = builder.get(main_trace_commits, main_commit_idx.clone());
            builder.set_value(
                &rounds,
                round_idx.clone(),
                TwoAdicPcsRoundVariable {
                    batch_commit,
                    mats: common_main_mats,
                    permutation: air_perm_by_height.clone(),
                },
            );
            builder.assign(&round_idx, round_idx.clone() + RVar::one());
        }

        // 3. After challenge: one per phase
        builder.assert_usize_eq(opening.values.after_challenge.len(), num_phases);
        builder
            .range(0, num_phases)
            .for_each(|phase_idx_vec, builder| {
                let phase_idx = phase_idx_vec[0];
                let values_per_mat = builder.get(&opening.values.after_challenge, phase_idx);
                builder.assert_usize_eq(values_per_mat.len(), num_after_challenge_traces);
                let batch_commit = builder.get(after_challenge_commits, phase_idx);

                let mat_idx: Usize<_> = builder.eval(RVar::zero());
                let mats: Array<_, TwoAdicPcsMatsVariable<_>> =
                    builder.array(num_after_challenge_traces);
                builder.range(0, num_airs).for_each(|i_vec, builder| {
                    let i = i_vec[0];
                    let advice = builder.get(&m_advice_var.per_air, i);
                    builder
                        .if_ne(advice.width.after_challenge.len(), RVar::zero())
                        .then(|builder| {
                            // Only 1 phase is supported.
                            builder
                                .assert_usize_eq(advice.width.after_challenge.len(), RVar::one());
                            let after_challenge_width = RVar::from(
                                builder.get(&advice.width.after_challenge, RVar::zero()),
                            );
                            let domain = builder.get(&domains, i);
                            let trace_points = builder.get(&trace_points_per_domain, i);

                            let after_challenge = builder.get(&values_per_mat, mat_idx.clone());

                            let values = builder.array::<Array<C, _>>(2);
                            builder.assert_usize_eq(
                                after_challenge.local.len(),
                                after_challenge_width * RVar::from(C::EF::D),
                            );
                            builder.assert_usize_eq(
                                after_challenge.next.len(),
                                after_challenge.local.len(),
                            );
                            builder.set_value(&values, 0, after_challenge.local);
                            builder.set_value(&values, 1, after_challenge.next);
                            let after_challenge_mat = TwoAdicPcsMatsVariable::<C> {
                                domain,
                                values,
                                points: trace_points,
                            };
                            builder.set_value(&mats, mat_idx.clone(), after_challenge_mat);
                            builder.inc(&mat_idx);
                        });
                });
                builder.assert_usize_eq(mat_idx, num_after_challenge_traces);

                builder.set_value(
                    &rounds,
                    round_idx.clone(),
                    TwoAdicPcsRoundVariable {
                        batch_commit,
                        mats,
                        permutation: air_perm_by_height.clone(),
                    },
                );
                builder.assign(&round_idx, round_idx.clone() + RVar::one());
            });
        // T05c: the shape of `opening.values.main` has been validated

        // 4. Quotient domains and openings

        // The permutation array for the quotient chunks.
        // For example:
        // There are 2 AIRs, X and Y. X has 2 quotient chunks, X_1 and X_2. Y has 3
        // quotient chunks, Y_1, Y_2, and Y_3.
        // `air_perm_by_height` is [1, 0].
        // Because quotient chunks have the same height as the trace of its AIR. So the permutation
        // array is [Y_1, Y_2, Y_3, X_1, X_2] = [2, 3, 4, 0, 1].
        // AIR index -> its offset in the permutation array.
        let quotient_perm = builder.array(num_quotient_mats);
        let perm_offset_per_air = builder.array::<Usize<_>>(num_airs);
        let offset: Usize<_> = builder.eval(RVar::zero());
        iter_zip!(builder, air_perm_by_height).for_each(|ptr_vec, builder| {
            let air_index = builder.iter_ptr_get(air_perm_by_height, ptr_vec[0]);
            builder.set(&perm_offset_per_air, air_index.clone(), offset.clone());
            let qc_domains = builder.get(&quotient_chunk_domains, air_index);
            builder.assign(&offset, offset.clone() + qc_domains.len());
        });

        let quotient_mats: Array<_, TwoAdicPcsMatsVariable<_>> = builder.array(num_quotient_mats);
        let qc_points = builder.array::<Ext<_, _>>(1);
        builder.set_value(&qc_points, 0, zeta);

        let qc_index: Usize<_> = builder.eval(RVar::zero());
        builder.assert_usize_eq(opening.values.quotient.len(), num_airs);
        builder.range(0, num_airs).for_each(|i_vec, builder| {
            let i = i_vec[0];
            let opened_quotient = builder.get(&opening.values.quotient, i);
            let qc_domains = builder.get(&quotient_chunk_domains, i);
            let air_offset = builder.get(&perm_offset_per_air, i);

            builder.assert_usize_eq(opened_quotient.len(), qc_domains.len());
            let quotient_degree = qc_domains.len();
            builder
                .range(0, quotient_degree)
                .for_each(|j_vec, builder| {
                    let j = j_vec[0];
                    let qc_dom = builder.get(&qc_domains, j);
                    let qc_vals_array = builder.get(&opened_quotient, j);
                    builder.assert_usize_eq(qc_vals_array.len(), RVar::from(4));
                    let qc_values = builder.array::<Array<C, _>>(1);
                    builder.set_value(&qc_values, 0, qc_vals_array);
                    let qc_mat = TwoAdicPcsMatsVariable::<C> {
                        domain: qc_dom,
                        values: qc_values,
                        points: qc_points.clone(),
                    };
                    let qc_offset = builder.eval_expr(air_offset.clone() + j);
                    builder.set_value(&quotient_mats, qc_index.clone(), qc_mat);
                    builder.set(&quotient_perm, qc_offset, RVar::from(qc_index.clone()));
                    builder.assign(&qc_index, qc_index.clone() + RVar::one());
                });
        });
        let quotient_round = TwoAdicPcsRoundVariable {
            batch_commit: quotient_commit.clone(),
            mats: quotient_mats,
            permutation: quotient_perm,
        };
        builder.set_value(&rounds, round_idx.clone(), quotient_round);
        builder.assign(&round_idx, round_idx.clone() + RVar::one());

        // Sanity check: the number of rounds matches.
        builder.assert_usize_eq(round_idx, total_rounds);

        builder.cycle_tracker_end("stage-c-build-rounds");

        // Verify the pcs proof
        builder.cycle_tracker_start("stage-d-verify-pcs");
        pcs.verify(
            builder,
            rounds,
            opening.proof.clone(),
            log_max_height,
            challenger,
        );
        builder.cycle_tracker_end("stage-d-verify-pcs");

        builder.cycle_tracker_start("stage-e-verify-constraints");
        let after_challenge_idx: Usize<C::N> = builder.eval(C::N::ZERO);
        let preprocessed_idx: Usize<_> = builder.eval(C::N::ZERO);
        let cached_main_commit_idx: Usize<_> = builder.eval(C::N::ZERO);
        let common_main_matrix_idx: Usize<_> = builder.eval(C::N::ZERO);
        let air_idx: Usize<_> = builder.eval(C::N::ZERO);
        let common_main_openings = builder.get(&opening.values.main, num_cached_mains);

        // Convert challenges into a fixed-shape array.
        let challenges = m_advice
            .num_challenges_to_sample
            .iter()
            .enumerate()
            .map(|(phase, &num_challenges_to_sample)| {
                (0..num_challenges_to_sample)
                    .map(|i| {
                        let challenge: Ext<_, _> = builder.constant(C::EF::ZERO);
                        builder
                            .if_eq(
                                m_advice_var.num_challenges_to_sample_mask[phase][i].clone(),
                                RVar::one(),
                            )
                            .then(|builder| {
                                let chs = builder.get(&challenges_per_phase, phase);
                                let v = builder.get(&chs, i);
                                builder.assign(&challenge, v);
                            });
                        challenge
                    })
                    .collect_vec()
            })
            .collect_vec();

        for (i, air_const) in m_advice.per_air.iter().enumerate() {
            let abs_air_idx = builder.get(&air_ids, air_idx.clone());
            builder.if_eq(abs_air_idx, RVar::from(i)).then(|builder| {
                let preprocessed_values = if air_const.preprocessed_data.is_some() {
                    let ret =
                        Some(builder.get(&opening.values.preprocessed, preprocessed_idx.clone()));
                    builder.inc(&preprocessed_idx);
                    ret
                } else {
                    None
                };
                let mut partitioned_main_values = (0..air_const.width.cached_mains.len())
                    .map(|_| {
                        let ret = builder.get(&opening.values.main, cached_main_commit_idx.clone());
                        builder.inc(&cached_main_commit_idx);
                        builder.get(&ret, 0)
                    })
                    .collect_vec();
                if air_const.width.common_main > 0 {
                    let common_main =
                        builder.get(&common_main_openings, common_main_matrix_idx.clone());
                    builder.inc(&common_main_matrix_idx);
                    partitioned_main_values.push(common_main);
                }

                let after_challenge_values = if air_const.width.after_challenge.is_empty() {
                    AdjacentOpenedValuesVariable {
                        local: builder.vec(vec![]),
                        next: builder.vec(vec![]),
                    }
                } else {
                    // One phase for now
                    let after_challenge_values = builder.get(&opening.values.after_challenge, 0);
                    let after_challenge_values =
                        builder.get(&after_challenge_values, after_challenge_idx.clone());
                    builder.inc(&after_challenge_idx);
                    after_challenge_values
                };
                let trace_domain = builder.get(&domains, air_idx.clone());
                let quotient_domain: TwoAdicMultiplicativeCosetVariable<_> =
                    builder.get(&quotient_domains, air_idx.clone());
                // Check that the quotient data matches the chip's data.
                let log_quotient_degree = air_const.log_quotient_degree();
                let quotient_chunks = builder.get(&opening.values.quotient, air_idx.clone());

                // Get the domains from the chip itself.
                let qc_domains = quotient_domain.split_domains_const(builder, log_quotient_degree);
                let air_proof = builder.get(air_proofs, air_idx.clone());
                let pvs = (0..air_const.num_public_values)
                    .map(|x| builder.get(&air_proof.public_values, x))
                    .collect_vec();

                let exposed_values = air_const
                    .num_exposed_values_after_challenge
                    .iter()
                    .enumerate()
                    .map(|(phase, &num_exposed)| {
                        let exposed_values =
                            builder.get(&air_proof.exposed_values_after_challenge, phase);
                        (0..num_exposed)
                            .map(|j| builder.get(&exposed_values, j))
                            .collect_vec()
                    })
                    .collect_vec();

                Self::verify_single_rap_constraints(
                    builder,
                    air_const,
                    preprocessed_values,
                    &partitioned_main_values,
                    quotient_chunks,
                    &pvs,
                    trace_domain,
                    qc_domains,
                    zeta,
                    alpha,
                    after_challenge_values,
                    &challenges,
                    &exposed_values,
                );

                builder.inc(&air_idx);
            });
        }
        // Assert that all provided AIRs were verified.
        builder.assert_usize_eq(air_idx, air_ids.len());

        builder.cycle_tracker_end("stage-e-verify-constraints");
    }

    /// For constraint `i` in `trace_height_constraints` with coefficients a_i1, ..., a_ik and
    /// threshold b_i, checks that
    ///    a_i1 * x_1 + ... + a_ik * x_k < b_i,
    /// where `x_i` is the height of the trace given in `air_proofs[j]` with `j` such that
    /// `air_proofs[j].air_id = i`.
    ///
    /// The caller must ensure the following is met:
    /// * `constraint_system.constraints.len() == num_airs`
    /// * `constraint_system.constraints[air_proofs[i].air_id]` is valid for all `i`
    /// * For all `i`, `air_proofs[i].log_degree < MAX_TWO_ADICITY`
    fn check_trace_height_constraints(
        builder: &mut Builder<C>,
        constraint_system: &TraceHeightConstraintSystem<C>,
        air_proofs: &Array<C, AirProofDataVariable<C>>,
    ) {
        // We compute and store `a_i1 * x_1 + ... + a_ik * x_k` in `values[i]`.
        let values: Vec<Var<C::N>> = (0..constraint_system.height_constraints.len())
            .map(|_| builder.eval(C::N::ZERO))
            .collect();

        let assert_less_than = |builder: &mut Builder<C>, a, b| {
            if builder.flags.static_only {
                builder.push(DslIr::CircuitLessThan(a, b));
            } else {
                builder.assert_less_than_slow_bit_decomp(a, b);
            }
        };

        // Will index by log_air_height. Max value is assumed to be MAX_TWO_ADICITY - 1 because
        // `log_blowup >= 1`.
        let pows_of_two: Array<C, Var<C::N>> = builder.array(MAX_TWO_ADICITY);
        let cur: Var<C::N> = builder.constant(C::N::ONE);
        iter_zip!(builder, pows_of_two).for_each(|ptr_vec, builder| {
            builder.iter_ptr_set(&pows_of_two, ptr_vec[0], cur);
            builder.assign(&cur, cur + cur);
        });

        iter_zip!(builder, air_proofs).for_each(|ptr_vec, builder| {
            let air_proof_data = builder.iter_ptr_get(air_proofs, ptr_vec[0]);

            let height = builder.get(&pows_of_two, air_proof_data.log_degree);
            let height_max = builder.get(
                &constraint_system.height_maxes,
                air_proof_data.air_id.clone(),
            );
            builder
                .if_eq(height_max.is_some, C::N::ONE)
                .then(|builder| {
                    assert_less_than(builder, height, height_max.value);
                });

            for (i, constraint) in constraint_system.height_constraints.iter().enumerate() {
                let coeff = builder.get(&constraint.coefficients, air_proof_data.air_id.clone());

                let new_value: Var<C::N> = builder.eval(values[i] + coeff * height);
                let new_value_plus_one: Var<C::N> = builder.eval(new_value + C::N::ONE);
                assert_less_than(builder, values[i], new_value_plus_one);
                builder.assign(&values[i], new_value);
            }
        });
        for (value, constraint) in zip(values, &constraint_system.height_constraints) {
            if builder.flags.static_only || !constraint.is_threshold_at_p {
                assert_less_than(builder, value, constraint.threshold);
            }
        }
    }

    /// Reference: [openvm_stark_backend::verifier::constraints::verify_single_rap_constraints]
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::type_complexity)]
    pub fn verify_single_rap_constraints(
        builder: &mut Builder<C>,
        constants: &StarkVerificationAdvice<C>,
        preprocessed_values: Option<AdjacentOpenedValuesVariable<C>>,
        partitioned_main_values: &[AdjacentOpenedValuesVariable<C>],
        quotient_chunks: Array<C, Array<C, Ext<C::F, C::EF>>>,
        public_values: &[Felt<C::F>],
        trace_domain: TwoAdicMultiplicativeCosetVariable<C>,
        qc_domains: Vec<TwoAdicMultiplicativeCosetVariable<C>>,
        zeta: Ext<C::F, C::EF>,
        alpha: Ext<C::F, C::EF>,
        after_challenge_values: AdjacentOpenedValuesVariable<C>,
        challenges: &[Vec<Ext<C::F, C::EF>>],
        exposed_values_after_challenge: &[Vec<Ext<C::F, C::EF>>],
    ) {
        let sels = trace_domain.selectors_at_point(builder, zeta);

        let mut preprocessed = AdjacentOpenedValues {
            local: vec![],
            next: vec![],
        };
        if let Some(preprocessed_values) = preprocessed_values {
            for i in 0..constants.width.preprocessed.unwrap() {
                preprocessed
                    .local
                    .push(builder.get(&preprocessed_values.local, i));
                preprocessed
                    .next
                    .push(builder.get(&preprocessed_values.next, i));
            }
        }

        let main_widths = constants.width.main_widths();
        assert_eq!(partitioned_main_values.len(), main_widths.len());
        let partitioned_main_values = partitioned_main_values
            .iter()
            .zip_eq(main_widths.iter())
            .map(|(main_values, &width)| {
                builder.assert_usize_eq(main_values.local.len(), RVar::from(width));
                builder.assert_usize_eq(main_values.next.len(), RVar::from(width));
                let mut main = AdjacentOpenedValues {
                    local: vec![],
                    next: vec![],
                };
                for i in 0..width {
                    main.local.push(builder.get(&main_values.local, i));
                    main.next.push(builder.get(&main_values.next, i));
                }
                main
            })
            .collect_vec();

        let mut after_challenge = AdjacentOpenedValues {
            local: vec![],
            next: vec![],
        };

        let after_challenge_width = if constants.width.after_challenge.is_empty() {
            0
        } else {
            C::EF::D * constants.width.after_challenge[0]
        };
        builder.assert_usize_eq(
            after_challenge_values.local.len(),
            RVar::from(after_challenge_width),
        );
        builder.assert_usize_eq(
            after_challenge_values.next.len(),
            RVar::from(after_challenge_width),
        );
        for i in 0..after_challenge_width {
            after_challenge
                .local
                .push(builder.get(&after_challenge_values.local, i));
            after_challenge
                .next
                .push(builder.get(&after_challenge_values.next, i));
        }

        let folded_constraints = Self::eval_constraints(
            builder,
            &constants.symbolic_constraints,
            preprocessed,
            &partitioned_main_values,
            public_values,
            &sels,
            alpha,
            after_challenge,
            challenges,
            exposed_values_after_challenge,
        );

        let num_quotient_chunks = 1 << constants.log_quotient_degree();
        let mut quotient = vec![];
        // Assert that the length of the quotient chunk arrays match the expected length.
        builder.assert_usize_eq(quotient_chunks.len(), RVar::from(num_quotient_chunks));
        // Collect the quotient values into vectors.
        for i in 0..num_quotient_chunks {
            let chunk = builder.get(&quotient_chunks, i);
            // Assert that the chunk length matches the expected length.
            builder.assert_usize_eq(RVar::from(C::EF::D), RVar::from(chunk.len()));
            // Collect the quotient values into vectors.
            let mut quotient_vals = vec![];
            for j in 0..C::EF::D {
                let value = builder.get(&chunk, j);
                quotient_vals.push(value);
            }
            quotient.push(quotient_vals);
        }

        let quotient: Ext<_, _> = Self::recompute_quotient(builder, &quotient, qc_domains, zeta);

        // Assert that the quotient times the zerofier is equal to the folded constraints.
        builder.assert_ext_eq(folded_constraints * sels.inv_zeroifier, quotient);
    }

    #[allow(clippy::too_many_arguments)]
    fn eval_constraints(
        builder: &mut Builder<C>,
        constraints: &SymbolicExpressionDag<C::F>,
        preprocessed_values: AdjacentOpenedValues<Ext<C::F, C::EF>>,
        partitioned_main_values: &[AdjacentOpenedValues<Ext<C::F, C::EF>>],
        public_values: &[Felt<C::F>],
        selectors: &LagrangeSelectors<Ext<C::F, C::EF>>,
        alpha: Ext<C::F, C::EF>,
        after_challenge: AdjacentOpenedValues<Ext<C::F, C::EF>>,
        challenges: &[Vec<Ext<C::F, C::EF>>],
        exposed_values_after_challenge: &[Vec<Ext<C::F, C::EF>>],
    ) -> Ext<C::F, C::EF> {
        let mut unflatten = |v: &[Ext<C::F, C::EF>]| {
            v.chunks_exact(C::EF::D)
                .map(|chunk| {
                    builder.eval(
                        chunk
                            .iter()
                            .enumerate()
                            .map(|(e_i, &x)| x * C::EF::monomial(e_i).cons())
                            .sum::<SymbolicExt<_, _>>(),
                    )
                })
                .collect::<Vec<Ext<_, _>>>()
        };

        let after_challenge_values = AdjacentOpenedValues {
            local: unflatten(&after_challenge.local),
            next: unflatten(&after_challenge.next),
        };

        let mut folder: RecursiveVerifierConstraintFolder<C> = GenericVerifierConstraintFolder {
            preprocessed: VerticalPair::new(
                RowMajorMatrixView::new_row(&preprocessed_values.local),
                RowMajorMatrixView::new_row(&preprocessed_values.next),
            ),
            partitioned_main: partitioned_main_values
                .iter()
                .map(|main_values| {
                    VerticalPair::new(
                        RowMajorMatrixView::new_row(&main_values.local),
                        RowMajorMatrixView::new_row(&main_values.next),
                    )
                })
                .collect(),
            after_challenge: vec![VerticalPair::new(
                RowMajorMatrixView::new_row(&after_challenge_values.local),
                RowMajorMatrixView::new_row(&after_challenge_values.next),
            )],
            challenges,
            is_first_row: selectors.is_first_row,
            is_last_row: selectors.is_last_row,
            is_transition: selectors.is_transition,
            alpha,
            accumulator: SymbolicExt::ZERO,
            public_values,
            exposed_values_after_challenge,
            _marker: PhantomData,
        };
        folder.eval_constraints(constraints);

        builder.eval(folder.accumulator)
    }

    fn recompute_quotient(
        builder: &mut Builder<C>,
        quotient_chunks: &[Vec<Ext<C::F, C::EF>>],
        qc_domains: Vec<TwoAdicMultiplicativeCosetVariable<C>>,
        zeta: Ext<C::F, C::EF>,
    ) -> Ext<C::F, C::EF> {
        let zps = qc_domains
            .iter()
            .enumerate()
            .map(|(i, domain)| {
                qc_domains
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| *j != i)
                    .map(|(_, other_domain)| {
                        let first_point: Ext<_, _> = builder.eval(domain.first_point());
                        other_domain.zp_at_point(builder, zeta)
                            * other_domain.zp_at_point(builder, first_point).inverse()
                    })
                    .product::<SymbolicExt<_, _>>()
            })
            .collect::<Vec<SymbolicExt<_, _>>>()
            .into_iter()
            .map(|x| builder.eval(x))
            .collect::<Vec<Ext<_, _>>>();

        builder.eval(
            quotient_chunks
                .iter()
                .enumerate()
                .map(|(ch_i, ch)| {
                    assert_eq!(ch.len(), C::EF::D);
                    ch.iter()
                        .enumerate()
                        .map(|(e_i, &c)| zps[ch_i] * C::EF::monomial(e_i) * c)
                        .sum::<SymbolicExt<_, _>>()
                })
                .sum::<SymbolicExt<_, _>>(),
        )
    }
}

fn assert_cumulative_sums<C: Config>(
    builder: &mut Builder<C>,
    air_proofs: &Array<C, AirProofDataVariable<C>>,
    num_challenges_to_sample: &Array<C, Usize<C::N>>,
) {
    let num_phase = num_challenges_to_sample.len();
    // Currently only support 0 or 1 phase is supported.
    builder.if_eq(num_phase, RVar::one()).then(|builder| {
        let cumulative_sum: Ext<C::F, C::EF> = builder.eval(C::F::ZERO);
        builder
            .range(0, air_proofs.len())
            .for_each(|i_vec, builder| {
                let i = i_vec[0];
                let air_proof_input = builder.get(air_proofs, i);
                let exposed_values = air_proof_input.exposed_values_after_challenge;

                builder
                    .if_ne(exposed_values.len(), RVar::zero())
                    .then(|builder| {
                        // Verifier does not support more than 1 challenge phase
                        builder.assert_usize_eq(exposed_values.len(), RVar::one());
                        let values = builder.get(&exposed_values, RVar::zero());
                        // Only exposed value should be cumulative sum
                        builder.assert_usize_eq(values.len(), RVar::one());

                        let summand = builder.get(&values, RVar::zero());
                        builder.assign(&cumulative_sum, cumulative_sum + summand);
                    });
            });
        builder.assert_ext_eq(cumulative_sum, C::EF::ZERO.cons());
    });
}

/// Conversion used for challenger to observe usize. The `val` must come
/// from compile time constant in static mode.
fn cast_usize_to_felt<C: Config>(builder: &mut Builder<C>, val: Usize<C::N>) -> Felt<C::F>
where
    C::F: TwoAdicField,
{
    if builder.flags.static_only {
        builder.eval(C::F::from_canonical_usize(val.value()))
    } else {
        builder.unsafe_cast_var_to_felt(val.get_var())
    }
}
