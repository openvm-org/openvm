pub mod outer;

use std::{cmp::Reverse, marker::PhantomData};

use afs_compiler::{
    conversion::CompilerOptions,
    ir::{Array, Builder, Config, Ext, ExtConst, Felt, SymbolicExt, Usize, Var},
    prelude::RVar,
};
use afs_stark_backend::{
    air_builders::{
        symbolic::symbolic_expression::SymbolicExpression,
        verifier::GenericVerifierConstraintFolder,
    },
    prover::opener::AdjacentOpenedValues,
    rap::AnyRap,
};
use afs_test_utils::config::{baby_bear_poseidon2::BabyBearPoseidon2Config, FriParameters};
use itertools::{izip, Itertools};
use p3_baby_bear::BabyBear;
use p3_commit::LagrangeSelectors;
use p3_field::{AbstractExtensionField, AbstractField, TwoAdicField};
use p3_matrix::{
    dense::{RowMajorMatrix, RowMajorMatrixView},
    stack::VerticalPair,
    Matrix,
};
use stark_vm::program::Program;

use crate::{
    challenger::{duplex::DuplexChallengerVariable, ChallengerVariable},
    commit::{PcsVariable, PolynomialSpaceVariable},
    folder::RecursiveVerifierConstraintFolder,
    fri::{
        types::{TwoAdicPcsMatsVariable, TwoAdicPcsRoundVariable},
        TwoAdicFriPcsVariable, TwoAdicMultiplicativeCosetVariable,
    },
    hints::Hintable,
    types::{
        AdjacentOpenedValuesVariable, CommitmentsVariable, InnerConfig,
        MultiStarkVerificationAdvice, StarkVerificationAdvice, VerifierInput,
        VerifierInputVariable, PROOF_MAX_NUM_PVS,
    },
    utils::const_fri_config,
};

#[derive(Debug, Clone, Copy)]
pub struct VerifierProgram<C: Config> {
    _phantom: std::marker::PhantomData<C>,
}

impl VerifierProgram<InnerConfig> {
    /// Create a new instance of the program for the [BabyBearPoseidon2] config.
    pub fn build(
        constants: MultiStarkVerificationAdvice<InnerConfig>,
        fri_params: &FriParameters,
    ) -> Program<BabyBear> {
        let mut builder = Builder::<InnerConfig>::default();

        builder.cycle_tracker_start("VerifierProgram");
        let input: VerifierInputVariable<_> = builder.uninit();
        VerifierInput::<BabyBearPoseidon2Config>::witness(&input, &mut builder);

        let pcs = TwoAdicFriPcsVariable {
            config: const_fri_config(&mut builder, fri_params),
        };
        StarkVerifier::verify::<DuplexChallengerVariable<_>>(&mut builder, &pcs, constants, &input);

        builder.cycle_tracker_end("VerifierProgram");
        builder.halt();

        const WORD_SIZE: usize = 1;
        builder.compile_isa_with_options::<WORD_SIZE>(CompilerOptions {
            enable_cycle_tracker: true,
            ..Default::default()
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct StarkVerifier<C: Config> {
    _phantom: std::marker::PhantomData<C>,
}

impl<C: Config> StarkVerifier<C>
where
    C::F: TwoAdicField,
{
    /// Reference: [afs_stark_backend::verifier::MultiTraceStarkVerifier::verify].
    pub fn verify<CH: ChallengerVariable<C>>(
        builder: &mut Builder<C>,
        pcs: &TwoAdicFriPcsVariable<C>,
        constants: MultiStarkVerificationAdvice<C>,
        input: &VerifierInputVariable<C>,
    ) {
        let proof = &input.proof;

        let cumulative_sum: Ext<C::F, C::EF> = builder.eval(C::F::zero());
        let num_phases = constants.num_challenges_to_sample.len();
        // Currently only support 0 or 1 phase is supported.
        assert!(num_phases <= 1);
        // Tmp solution to support 0 or 1 phase.
        if num_phases > 0 {
            builder
                .range(0, proof.exposed_values_after_challenge.len())
                .for_each(|i, builder| {
                    let exposed_values = builder.get(&proof.exposed_values_after_challenge, i);

                    // Verifier does not support more than 1 challenge phase
                    builder.assert_eq::<Usize<_>>(exposed_values.len(), RVar::one());

                    let values = builder.get(&exposed_values, RVar::zero());

                    // Only exposed value should be cumulative sum
                    builder.assert_eq::<Usize<_>>(values.len(), RVar::one());

                    let summand = builder.get(&values, RVar::zero());
                    builder.assign(&cumulative_sum, cumulative_sum + summand);
                });
        }
        builder.assert_ext_eq(cumulative_sum, C::EF::zero().cons());

        let mut challenger = CH::new(builder);

        Self::verify_raps(builder, pcs, constants, &mut challenger, input);
    }

    /// Reference: [afs_stark_backend::verifier::MultiTraceStarkVerifier::verify_raps].
    pub fn verify_raps(
        builder: &mut Builder<C>,
        pcs: &TwoAdicFriPcsVariable<C>,
        vk: MultiStarkVerificationAdvice<C>,
        challenger: &mut impl ChallengerVariable<C>,
        input: &VerifierInputVariable<C>,
    ) where
        C::F: TwoAdicField,
        C::EF: TwoAdicField,
    {
        Self::validate_inputs(builder, &vk, input);

        let VerifierInputVariable::<C> {
            proof,
            log_degree_per_air,
            public_values,
        } = input;

        let num_airs = vk.per_air.len();
        let r_num_airs = num_airs;
        let num_phases = vk.num_challenges_to_sample.len();
        // Currently only support 0 or 1 phase is supported.
        assert!(num_phases <= 1);

        for k in 0..num_airs {
            let pvs = builder.get(public_values, k);
            for j in 0..vk.per_air[k].num_public_values {
                let element = builder.get(&pvs, j);
                challenger.observe(builder, element);
            }
        }

        builder.cycle_tracker_start("stage-c-build-rounds");

        for i in 0..num_airs {
            if let Some(preprocessed_data) = vk.per_air[i].preprocessed_data.as_ref() {
                let commit = builder.constant(preprocessed_data.commit.clone());
                challenger.observe_digest(builder, commit);
            }
        }

        let CommitmentsVariable {
            main_trace: main_trace_commits,
            after_challenge: after_challenge_commits,
            quotient: quotient_commit,
        } = &proof.commitments;

        // Observe main trace commitments
        for i in 0..vk.num_main_trace_commitments {
            let main_commit = builder.get(main_trace_commits, i);
            challenger.observe_digest(builder, main_commit.clone());
        }

        let mut challenges = Vec::new();
        let mut exposed_values_by_air = vec![vec![vec![]; num_phases]; num_airs];
        for phase_idx in 0..num_phases {
            let num_to_sample: usize = 2;

            let provided_num_to_sample = vk.num_challenges_to_sample[phase_idx];
            assert_eq!(provided_num_to_sample, num_to_sample);

            // Sample challenges needed in this phase.
            challenges.push(
                (0..num_to_sample)
                    .map(|_| challenger.sample_ext(builder))
                    .collect_vec(),
            );

            // For each RAP, the exposed values in the current phase
            for (j, exposed_values_after_challenge) in exposed_values_by_air.iter_mut().enumerate()
            {
                let exposed_values = builder.get(&proof.exposed_values_after_challenge, j);
                let values = builder.get(&exposed_values, phase_idx);
                let values_len = vk.per_air[j].num_exposed_values_after_challenge[phase_idx];
                for k in 0..values_len {
                    let value = builder.get(&values, k);
                    exposed_values_after_challenge[phase_idx].push(value);
                    let felts = builder.ext2felt(value);
                    challenger.observe_slice(builder, felts);
                }
            }

            // Observe single commitment to all trace matrices in this phase.
            let commit = builder.get(after_challenge_commits, phase_idx);
            challenger.observe_digest(builder, commit);
        }

        let alpha = challenger.sample_ext(builder);

        challenger.observe_digest(builder, quotient_commit.clone());

        let zeta = challenger.sample_ext(builder);

        let mut trace_domains = builder.array::<TwoAdicMultiplicativeCosetVariable<_>>(r_num_airs);

        let mut num_prep_rounds = 0;

        // Build domains
        let mut domains = builder.array(r_num_airs);
        let mut quotient_domains = builder.array(r_num_airs);
        let mut trace_points_per_domain = builder.array(r_num_airs);
        let mut quotient_chunk_domains = builder.array(r_num_airs);
        for i in 0..num_airs {
            let log_degree: RVar<_> = builder.get(log_degree_per_air, i).into();

            let domain = pcs.natural_domain_for_log_degree(builder, log_degree);
            builder.set_value(&mut trace_domains, i, domain.clone());

            let mut trace_points = builder.array::<Ext<_, _>>(2);
            let zeta_next = domain.next_point(builder, zeta);
            builder.set_value(&mut trace_points, RVar::zero(), zeta);
            builder.set_value(&mut trace_points, RVar::one(), zeta_next);

            let log_quotient_degree = RVar::from(vk.per_air[i].log_quotient_degree());
            let quotient_degree = vk.per_air[i].quotient_degree;
            let log_quotient_size = builder.eval_expr(log_degree + log_quotient_degree);
            let quotient_domain =
                domain.create_disjoint_domain(builder, log_quotient_size, Some(pcs.config.clone()));
            builder.set_value(&mut quotient_domains, i, quotient_domain.clone());

            let qc_domains =
                quotient_domain.split_domains(builder, log_quotient_degree, quotient_degree);

            builder.set_value(&mut domains, i, domain);
            builder.set_value(&mut trace_points_per_domain, i, trace_points);
            builder.set_value(&mut quotient_chunk_domains, i, qc_domains);

            if vk.per_air[i].preprocessed_data.is_some() {
                num_prep_rounds += 1;
            }
        }

        // Build the opening rounds

        let num_main_rounds = vk.num_main_trace_commitments;
        let num_challenge_rounds = vk.num_challenges_to_sample.len();
        let num_quotient_rounds = 1;

        let total_rounds =
            num_prep_rounds + num_main_rounds + num_challenge_rounds + num_quotient_rounds;

        let mut rounds = builder.array::<TwoAdicPcsRoundVariable<_>>(total_rounds);
        let mut round_idx = 0;

        // 1. First the preprocessed trace openings: one round per AIR with preprocessing.
        let prep_idx: Var<_> = builder.constant(C::N::zero());
        for i in 0..num_airs {
            if let Some(preprocessed_data) = vk.per_air[i].preprocessed_data.as_ref() {
                let prep = builder.get(&proof.opening.values.preprocessed, prep_idx);
                builder.assign(&prep_idx, prep_idx + C::N::one());
                let batch_commit = builder.constant(preprocessed_data.commit.clone());

                let domain = builder.get(&domains, i);
                let trace_points = builder.get(&trace_points_per_domain, i);

                // Assumption: each AIR with preprocessed trace has its own commitment and opening values
                let mut values = builder.array::<Array<C, _>>(2);
                builder.set_value(&mut values, 0, prep.local);
                builder.set_value(&mut values, 1, prep.next);
                let prep_mat = TwoAdicPcsMatsVariable::<C> {
                    domain,
                    values,
                    points: trace_points.clone(),
                };

                let mut mats: Array<_, TwoAdicPcsMatsVariable<_>> = builder.array(1);
                builder.set_value(&mut mats, 0, prep_mat);

                builder.set_value(
                    &mut rounds,
                    round_idx,
                    TwoAdicPcsRoundVariable { batch_commit, mats },
                );
                round_idx += 1;
            }
        }

        // 2. Then the main trace openings.
        vk.main_commit_to_air_graph
            .commit_to_air_index
            .iter()
            .enumerate()
            .for_each(|(commit_idx, matrix_to_air_index)| {
                let values_per_mat = builder.get(&proof.opening.values.main, commit_idx);
                let batch_commit = builder.get(main_trace_commits, commit_idx);

                builder.assert_eq::<Usize<_>>(
                    values_per_mat.len(),
                    RVar::from(matrix_to_air_index.len()),
                );
                let mut mats: Array<_, TwoAdicPcsMatsVariable<_>> =
                    builder.array(matrix_to_air_index.len());

                matrix_to_air_index
                    .iter()
                    .enumerate()
                    .for_each(|(matrix_idx, &air_idx)| {
                        let main = builder.get(&values_per_mat, matrix_idx);
                        let domain = builder.get(&domains, air_idx);
                        let trace_points = builder.get(&trace_points_per_domain, air_idx);
                        let mut values = builder.array::<Array<C, _>>(2);
                        builder.set_value(&mut values, 0, main.local);
                        builder.set_value(&mut values, 1, main.next);
                        let main_mat = TwoAdicPcsMatsVariable::<C> {
                            domain,
                            values,
                            points: trace_points,
                        };
                        builder.set_value(&mut mats, air_idx, main_mat);
                    });
                builder.set_value(
                    &mut rounds,
                    round_idx,
                    TwoAdicPcsRoundVariable { batch_commit, mats },
                );
                round_idx += 1;
            });

        // 3. After challenge: one per phase
        for phase_idx in 0..vk.num_challenges_to_sample.len() {
            let values_per_mat = builder.get(&proof.opening.values.after_challenge, phase_idx);
            let batch_commit = builder.get(after_challenge_commits, phase_idx);

            builder.assert_eq::<Usize<_>>(values_per_mat.len(), RVar::from(num_airs));

            let mut mats: Array<_, TwoAdicPcsMatsVariable<_>> = builder.array(num_airs);
            for i in 0..num_airs {
                let domain = builder.get(&domains, i);
                let trace_points = builder.get(&trace_points_per_domain, i);

                let after_challenge = builder.get(&values_per_mat, i);

                let mut values = builder.array::<Array<C, _>>(2);
                builder.set_value(&mut values, 0, after_challenge.local);
                builder.set_value(&mut values, 1, after_challenge.next);
                let after_challenge_mat = TwoAdicPcsMatsVariable::<C> {
                    domain,
                    values,
                    points: trace_points,
                };
                builder.set_value(&mut mats, i, after_challenge_mat);
            }

            builder.set_value(
                &mut rounds,
                round_idx,
                TwoAdicPcsRoundVariable { batch_commit, mats },
            );
            round_idx += 1;
        }

        // 4. Quotient domains and openings
        let num_quotient_mats = vk
            .per_air
            .iter()
            .map(|air| air.quotient_degree)
            .sum::<usize>();

        let mut quotient_mats: Array<_, TwoAdicPcsMatsVariable<_>> =
            builder.array(num_quotient_mats);
        let qc_index: Usize<_> = builder.eval(C::N::zero());

        let mut qc_points = builder.array::<Ext<_, _>>(1);
        builder.set_value(&mut qc_points, 0, zeta);

        for i in 0..num_airs {
            let opened_quotient = builder.get(&proof.opening.values.quotient, i);
            let qc_domains = builder.get(&quotient_chunk_domains, i);

            builder.range(0, qc_domains.len()).for_each(|j, builder| {
                let qc_dom = builder.get(&qc_domains, j);
                let qc_vals_array = builder.get(&opened_quotient, j);
                let mut qc_values = builder.array::<Array<C, _>>(1);
                builder.set_value(&mut qc_values, 0, qc_vals_array);
                let qc_mat = TwoAdicPcsMatsVariable::<C> {
                    domain: qc_dom,
                    values: qc_values,
                    points: qc_points.clone(),
                };
                builder.set_value(&mut quotient_mats, qc_index.clone(), qc_mat);
                builder.assign(&qc_index, qc_index.clone() + C::N::one());
            });
        }
        let quotient_round = TwoAdicPcsRoundVariable {
            batch_commit: quotient_commit.clone(),
            mats: quotient_mats,
        };
        builder.set_value(&mut rounds, round_idx, quotient_round);
        round_idx += 1;
        // Sanity check: the number of rounds matches.
        assert_eq!(round_idx, total_rounds);

        builder.cycle_tracker_end("stage-c-build-rounds");

        // Verify the pcs proof
        builder.cycle_tracker_start("stage-d-verify-pcs");
        pcs.verify(builder, rounds, proof.opening.proof.clone(), challenger);
        builder.cycle_tracker_end("stage-d-verify-pcs");

        // TODO[sp1] CONSTRAIN: that the preprocessed chips get called with verify_constraints.
        builder.cycle_tracker_start("stage-e-verify-constraints");

        // TODO[zach]: make per phase; for now just 1 phase so OK
        let after_challenge_idx: Usize<C::N> = builder.eval(C::N::zero());

        let mut preprocessed_idx = 0;

        for (index, air_const) in vk.per_air.iter().enumerate() {
            let preprocessed_values = if air_const.preprocessed_data.is_some() {
                let ret = Some(builder.get(&proof.opening.values.preprocessed, preprocessed_idx));
                preprocessed_idx += 1;
                ret
            } else {
                None
            };

            let partitioned_main_values = air_const
                .main_graph
                .matrix_ptrs
                .iter()
                .map(|ptr| {
                    let main_traces = builder.get(&proof.opening.values.main, ptr.commit_index);
                    builder.get(&main_traces, ptr.matrix_index)
                })
                .collect_vec();

            let after_challenge_values = if air_const.width.after_challenge.is_empty() {
                AdjacentOpenedValuesVariable {
                    local: builder.vec(vec![]),
                    next: builder.vec(vec![]),
                }
            } else {
                // One phase for now
                let after_challenge_values = builder.get(&proof.opening.values.after_challenge, 0);
                let after_challenge_values =
                    builder.get(&after_challenge_values, after_challenge_idx.clone());
                builder.assign(
                    &after_challenge_idx,
                    after_challenge_idx.clone() + C::N::one(),
                );
                after_challenge_values
            };

            let trace_domain = builder.get(&trace_domains, index);
            let quotient_domain: TwoAdicMultiplicativeCosetVariable<_> =
                builder.get(&quotient_domains, index);

            // Check that the quotient data matches the chip's data.
            let log_quotient_degree = air_const.log_quotient_degree();
            let quotient_chunks = builder.get(&proof.opening.values.quotient, index);

            // Get the domains from the chip itself.
            let qc_domains = quotient_domain.split_domains_const(builder, log_quotient_degree);

            let pvs = builder.get(public_values, index);
            Self::verify_single_rap_constraints(
                builder,
                air_const,
                preprocessed_values,
                &partitioned_main_values,
                quotient_chunks,
                pvs,
                trace_domain,
                qc_domains,
                zeta,
                alpha,
                after_challenge_values,
                &challenges,
                &exposed_values_by_air[index],
            );
        }

        builder.cycle_tracker_end("stage-e-verify-constraints");
    }

    /// Reference: [afs_stark_backend::verifier::constraints::verify_single_rap_constraints]
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::type_complexity)]
    pub fn verify_single_rap_constraints(
        builder: &mut Builder<C>,
        constants: &StarkVerificationAdvice<C>,
        preprocessed_values: Option<AdjacentOpenedValuesVariable<C>>,
        partitioned_main_values: &[AdjacentOpenedValuesVariable<C>],
        quotient_chunks: Array<C, Array<C, Ext<C::F, C::EF>>>,
        public_values: Array<C, Felt<C::F>>,
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

        assert_eq!(
            partitioned_main_values.len(),
            constants.width.partitioned_main.len()
        );
        let partitioned_main_values = partitioned_main_values
            .iter()
            .zip_eq(constants.width.partitioned_main.iter())
            .map(|(main_values, &width)| {
                builder.assert_eq::<Usize<_>>(main_values.local.len(), RVar::from(width));
                builder.assert_eq::<Usize<_>>(main_values.next.len(), RVar::from(width));
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
        builder.assert_eq::<Usize<_>>(
            after_challenge_values.local.len(),
            RVar::from(after_challenge_width),
        );
        builder.assert_eq::<Usize<_>>(
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
        builder.assert_eq::<Usize<_>>(quotient_chunks.len(), RVar::from(num_quotient_chunks));
        // Collect the quotient values into vectors.
        for i in 0..num_quotient_chunks {
            let chunk = builder.get(&quotient_chunks, i);
            // Assert that the chunk length matches the expected length.
            builder.assert_eq::<Usize<_>>(RVar::from(C::EF::D), RVar::from(chunk.len()));
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
        constraints: &[SymbolicExpression<C::F>],
        preprocessed_values: AdjacentOpenedValues<Ext<C::F, C::EF>>,
        partitioned_main_values: &[AdjacentOpenedValues<Ext<C::F, C::EF>>],
        public_values: Array<C, Felt<C::F>>,
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

        let mut folder_pv = Vec::new();
        let num_pvs = if builder.flags.static_only {
            public_values.len().value()
        } else {
            PROOF_MAX_NUM_PVS
        };
        for i in 0..num_pvs {
            folder_pv.push(builder.get(&public_values, i));
        }

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
            accumulator: SymbolicExt::zero(),
            public_values: &folder_pv,
            exposed_values_after_challenge, // FIXME
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

    fn validate_inputs(
        builder: &mut Builder<C>,
        vk: &MultiStarkVerificationAdvice<C>,
        input: &VerifierInputVariable<C>,
    ) {
        let num_airs = vk.per_air.len();
        let num_phases = vk.num_challenges_to_sample.len();
        // Currently only support 0 or 1 phase is supported.
        assert!(num_phases <= 1);

        let VerifierInputVariable::<C> {
            proof,
            log_degree_per_air,
            public_values,
        } = input;

        builder.assert_eq::<Usize<_>>(log_degree_per_air.len(), RVar::from(num_airs));
        // Challenger must observe public values
        builder.assert_eq::<Usize<_>>(public_values.len(), RVar::from(num_airs));

        builder.assert_eq::<Usize<_>>(
            proof.commitments.main_trace.len(),
            RVar::from(vk.num_main_trace_commitments),
        );
        for commit_idx in 0..vk.num_main_trace_commitments {
            let values_per_mat = builder.get(&proof.opening.values.main, commit_idx);
            builder.assert_eq::<Usize<_>>(values_per_mat.len(), RVar::from(num_airs));
        }

        builder.assert_eq::<Usize<_>>(
            proof.opening.values.after_challenge.len(),
            RVar::from(num_phases),
        );
        builder.assert_eq::<Usize<_>>(
            proof.commitments.after_challenge.len(),
            RVar::from(num_phases),
        );

        builder.assert_eq::<Usize<_>>(
            proof.exposed_values_after_challenge.len(),
            RVar::from(num_airs),
        );
        builder.assert_eq::<Usize<_>>(proof.opening.values.quotient.len(), RVar::from(num_airs));
        let mut num_preprocessed = 0;
        vk.per_air.iter().enumerate().for_each(|(i, air_const)| {
            let pvs = builder.get(public_values, i);
            builder.assert_eq::<Usize<_>>(pvs.len(), RVar::from(air_const.num_public_values));

            if air_const.preprocessed_data.is_some() {
                let preprocessed_width = air_const.width.preprocessed.unwrap();
                let preprocessed_value =
                    builder.get(&proof.opening.values.preprocessed, num_preprocessed);
                builder.assert_eq::<Usize<_>>(
                    preprocessed_value.local.len(),
                    RVar::from(preprocessed_width),
                );
                builder.assert_eq::<Usize<_>>(
                    preprocessed_value.next.len(),
                    RVar::from(preprocessed_width),
                );
                num_preprocessed += 1;
            }

            let exposed_values = builder.get(&proof.exposed_values_after_challenge, i);
            builder.assert_eq::<Usize<_>>(
                exposed_values.len(),
                RVar::from(air_const.num_exposed_values_after_challenge.len()),
            );
            air_const
                .num_exposed_values_after_challenge
                .iter()
                .enumerate()
                .for_each(|(phase_idx, &value_len)| {
                    let values = builder.get(&exposed_values, phase_idx);
                    builder.assert_eq::<Usize<_>>(values.len(), RVar::from(value_len));
                });

            for i in 0..(air_const.num_exposed_values_after_challenge.len()) {
                let num_exposed_values = air_const.num_exposed_values_after_challenge[i];
                let values = builder.get(&exposed_values, i);
                builder.assert_eq::<Usize<_>>(values.len(), RVar::from(num_exposed_values));
            }
        });

        builder.assert_eq::<Usize<_>>(
            proof.opening.values.preprocessed.len(),
            RVar::from(num_preprocessed),
        );
        // FIXME: check if all necessary validation is covered.
    }
}

#[allow(clippy::type_complexity)]
pub fn sort_chips(
    chips: Vec<&dyn AnyRap<BabyBearPoseidon2Config>>,
    traces: Vec<RowMajorMatrix<BabyBear>>,
    pvs: Vec<Vec<BabyBear>>,
) -> (
    Vec<&dyn AnyRap<BabyBearPoseidon2Config>>,
    Vec<RowMajorMatrix<BabyBear>>,
    Vec<Vec<BabyBear>>,
) {
    let mut groups = izip!(chips, traces, pvs).collect_vec();
    groups.sort_by_key(|(_, trace, _)| Reverse(trace.height()));

    let chips = groups.iter().map(|(x, _, _)| *x).collect_vec();
    let pvs = groups.iter().map(|(_, _, x)| x.clone()).collect_vec();
    let traces = groups.into_iter().map(|(_, x, _)| x).collect_vec();

    (chips, traces, pvs)
}
