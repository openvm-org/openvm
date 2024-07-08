use itertools::Itertools;
use p3_air::{Air, BaseAir};
use p3_baby_bear::BabyBear;
use p3_commit::LagrangeSelectors;
use p3_field::{AbstractExtensionField, AbstractField, TwoAdicField};
use p3_matrix::dense::RowMajorMatrixView;
use p3_matrix::stack::VerticalPair;

use afs_compiler::ir::{
    Array, Builder, Config, Ext, ExtConst, Felt, SymbolicExt, Usize, Var,
};
use afs_stark_backend::interaction::{AirBridge, InteractiveAir};
use afs_stark_backend::prover::opener::AdjacentOpenedValues;
use afs_stark_backend::rap::Rap;
use afs_test_utils::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
use afs_test_utils::config::FriParameters;
use stark_vm::cpu::trace::Instruction;

use crate::challenger::{CanObserveVariable, DuplexChallengerVariable, FeltChallenger};
use crate::commit::{PcsVariable, PolynomialSpaceVariable};
use crate::folder::AxiomRecursiveVerifierConstraintFolder;
use crate::fri::{TwoAdicFriPcsVariable, TwoAdicMultiplicativeCosetVariable};
use crate::fri::types::{TwoAdicPcsMatsVariable, TwoAdicPcsRoundVariable};
use crate::hints::Hintable;
use crate::types::{AdjacentOpenedValuesVariable, AxiomCommitmentsVariable, AxiomMemoryLayout, AxiomMemoryLayoutVariable, AxiomProofVariable, ChipDimensions, InnerConfig, MultiStarkPartialVerifyingKeyVariable, PROOF_MAX_NUM_PVS};
use crate::utils::const_fri_config;

pub trait DynRapForRecursion<C: Config>:
    for<'a> InteractiveAir<AxiomRecursiveVerifierConstraintFolder<'a, C>>
+ for<'a> Rap<AxiomRecursiveVerifierConstraintFolder<'a, C>>
+ BaseAir<C::F>
+ AirBridge<C::F>
{}

impl<C, T> DynRapForRecursion<C> for T
where
    C: Config,
    T: for<'a> InteractiveAir<AxiomRecursiveVerifierConstraintFolder<'a, C>>
    + for<'a> Air<AxiomRecursiveVerifierConstraintFolder<'a, C>>
    + BaseAir<C::F>
    + AirBridge<C::F>,
{}

#[derive(Debug, Clone, Copy)]
pub struct AxiomVerifier<C: Config> {
    _phantom: std::marker::PhantomData<C>,
}

impl AxiomVerifier<InnerConfig> {
    /// Create a new instance of the program for the [BabyBearPoseidon2] config.
    pub fn build(
        raps: Vec<&dyn DynRapForRecursion<InnerConfig>>,
        chip_dims: Vec<ChipDimensions>,
        config: &BabyBearPoseidon2Config,
        fri_params: &FriParameters,
    ) -> Vec<Instruction<BabyBear>> {
        let mut builder = Builder::<InnerConfig>::default();

        let input: AxiomMemoryLayoutVariable<_> = builder.uninit();
        AxiomMemoryLayout::<BabyBearPoseidon2Config>::witness(&input, &mut builder);

        let pcs = TwoAdicFriPcsVariable {
            config: const_fri_config(&mut builder, fri_params),
        };
        Self::verify(&mut builder, &pcs, raps, chip_dims, &input);

        builder.compile_isa()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct AxiomStarkVerifier<C: Config> {
    _phantom: std::marker::PhantomData<C>,
}

impl<C: Config> AxiomStarkVerifier<C>
where
    C::F: TwoAdicField,
{
    /// Reference: [afs_stark_backend::verifier::MultiTraceStarkVerifier::verify_raps].
    pub fn verify_raps(
        builder: &mut Builder<C>,
        pcs: &TwoAdicFriPcsVariable<C>,
        raps: Vec<&dyn DynRapForRecursion<C>>,
        chip_dims: Vec<ChipDimensions>,
        challenger: &mut DuplexChallengerVariable<C>,
        proof: &AxiomProofVariable<C>,
        partial_vk: &MultiStarkPartialVerifyingKeyVariable<C>,
        public_values: Array<C, Array<C, Felt<C::F>>>,
    ) where
        C::F: TwoAdicField,
        C::EF: TwoAdicField,
    {
        // Challenger must observe public values
        builder.assert_usize_eq(public_values.len(), raps.len());
        for k in 0..raps.len() {
            let pvs = builder.get(&public_values, k);
            builder.range(0, pvs.len()).for_each(|j, builder| {
                let element = builder.get(&pvs, j);
                challenger.observe(builder, element);
            });
        }

        builder.cycle_tracker("stage-c-build-rounds");

        let AxiomCommitmentsVariable {
            main_trace: main_trace_commits,
            after_challenge: after_challenge_commits,
            quotient: quotient_commit,
        } = &proof.commitments;

        // Observe main trace commitments
        builder
            .range(0, main_trace_commits.len())
            .for_each(|i, builder| {
                let main_commit = builder.get(&main_trace_commits, i);
                challenger.observe(builder, main_commit.clone());
            });

        let num_phases = 1;
        builder.assert_usize_eq(partial_vk.num_challenges_to_sample.len(), num_phases);
        builder.assert_usize_eq(
            partial_vk.num_challenges_to_sample.len(),
            after_challenge_commits.len(),
        );

        let mut challenges = Vec::new();
        for phase_idx in 0..num_phases {
            let num_to_sample: usize = 2;

            let provided_num_to_sample =
                builder.get(&partial_vk.num_challenges_to_sample, phase_idx);
            builder.assert_usize_eq(provided_num_to_sample, num_to_sample);

            // Sample challenges needed in this phase.
            challenges.push(
                (0..num_to_sample)
                    .map(|_| challenger.sample_ext(builder))
                    .collect_vec(),
            );

            // For each RAP, the exposed values in the current phase
            builder
                .range(0, proof.exposed_values_after_challenge.len())
                .for_each(|j, builder| {
                    let exposed_values = builder.get(&proof.exposed_values_after_challenge, j);
                    let values = builder.get(&exposed_values, phase_idx);
                    builder.range(0, values.len()).for_each(|k, builder| {
                        let value = builder.get(&values, k);
                        let felts = builder.ext2felt(value);
                        challenger.observe_slice(builder, felts);
                    });
                });

            // Observe single commitment to all trace matrices in this phase.
            let commit = builder.get(&after_challenge_commits, phase_idx);
            challenger.observe(builder, commit);
        }

        let alpha = challenger.sample_ext(builder);
        builder.print_e(alpha);

        challenger.observe(builder, quotient_commit.clone());

        let zeta = challenger.sample_ext(builder);
        builder.print_e(zeta);

        let num_airs = partial_vk.per_air.len();
        let mut trace_domains =
            builder.dyn_array::<TwoAdicMultiplicativeCosetVariable<_>>(num_airs);

        // Build domains
        let mut domains = builder.dyn_array(num_airs);
        let mut quotient_domains = builder.dyn_array(num_airs);
        let mut trace_points_per_domain = builder.dyn_array(num_airs);
        let mut quotient_chunk_domains = builder.dyn_array(num_airs);
        builder.range(0, num_airs).for_each(|i, builder| {
            let vk = builder.get(&partial_vk.per_air, i);

            let domain = pcs.natural_domain_for_log_degree(builder, Usize::Var(vk.log_degree));
            builder.set_value(&mut trace_domains, i, domain.clone());

            let mut trace_points = builder.dyn_array::<Ext<_, _>>(2);
            let zeta_next = domain.next_point(builder, zeta);
            builder.set_value(&mut trace_points, 0, zeta);
            builder.set_value(&mut trace_points, 1, zeta_next);

            let log_quotient_size: Usize<_> = builder.eval(vk.log_degree + vk.log_quotient_degree);
            let quotient_domain =
                domain.create_disjoint_domain(builder, log_quotient_size, Some(pcs.config.clone()));
            builder.set_value(&mut quotient_domains, i, quotient_domain.clone());

            let qc_domains =
                quotient_domain.split_domains(builder, vk.log_quotient_degree, vk.quotient_degree);

            builder.set_value(&mut domains, i, domain);
            builder.set_value(&mut trace_points_per_domain, i, trace_points);
            builder.set_value(&mut quotient_chunk_domains, i, qc_domains);
        });

        // Build the opening rounds

        let num_prep_rounds: Var<C::N> = builder.eval(C::N::zero()); // FIXME
        let num_main_rounds = partial_vk.num_main_trace_commitments;
        let num_challenge_rounds = partial_vk.num_challenges_to_sample.len().materialize(builder);
        let num_quotient_rounds: Var<C::N> = builder.eval(C::N::one());

        let total_rounds: Var<C::N> = builder.eval(num_prep_rounds + num_main_rounds + num_challenge_rounds + num_quotient_rounds);

        let mut rounds = builder.dyn_array::<TwoAdicPcsRoundVariable<_>>(total_rounds);
        let round_idx: Var<_> = builder.eval(C::N::zero());

        // 1. First the preprocessed trace openings: one round per AIR with preprocessing.
        // TODO[zach]: No preprocessed data yet.

        // 2. Then the main trace openings.
        builder.assert_usize_eq(proof.opening.values.main.len(), main_trace_commits.len());
        builder.range(0, main_trace_commits.len()).for_each(|commit_idx, builder| {
            let values_per_mat = builder.get(&proof.opening.values.main, commit_idx);
            let batch_commit = builder.get(&main_trace_commits, commit_idx);

            builder.assert_usize_eq(values_per_mat.len(), raps.len());

            let mut mats: Array<_, TwoAdicPcsMatsVariable<_>> = builder.dyn_array(raps.len());
            for i in 0..raps.len() {
                let main = builder.get(&values_per_mat, i);

                let domain = builder.get(&domains, i);
                let trace_points = builder.get(&trace_points_per_domain, i);

                let mut values = builder.dyn_array::<Array<C, _>>(2);
                builder.set_value(&mut values, 0, main.local);
                builder.set_value(&mut values, 1, main.next);
                let main_mat = TwoAdicPcsMatsVariable::<C> {
                    domain,
                    values,
                    points: trace_points.clone(),
                };
                builder.set_value(&mut mats, i, main_mat);
            }
            builder.set_value(
                &mut rounds,
                round_idx,
                TwoAdicPcsRoundVariable { batch_commit, mats },
            );
            builder.assign(round_idx, round_idx + C::N::one());
        });

        builder.assert_usize_eq(proof.opening.values.after_challenge.len(), partial_vk.num_challenges_to_sample.len());
        builder.assert_usize_eq(after_challenge_commits.len(), partial_vk.num_challenges_to_sample.len());

        // 3. After challenge: one per phase
        builder.range(0, partial_vk.num_challenges_to_sample.len()).for_each(|phase_idx, builder| {
            let values_per_mat = builder.get(&proof.opening.values.after_challenge, phase_idx);
            let batch_commit = builder.get(&after_challenge_commits, phase_idx);

            builder.assert_usize_eq(values_per_mat.len(), num_airs);

            let mut mats: Array<_, TwoAdicPcsMatsVariable<_>> = builder.dyn_array(num_airs);
            for i in 0..raps.len() {
                let domain = builder.get(&domains, i);
                let trace_points = builder.get(&trace_points_per_domain, i);

                let after_challenge = builder.get(&values_per_mat, i);

                let mut values = builder.dyn_array::<Array<C, _>>(2);
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

            builder.assign(round_idx, round_idx + C::N::one());
        });

        // 4. Quotient domains and openings
        let num_quotient_mats: Var<_> = builder.eval(C::N::zero());
        builder.range(0, num_airs).for_each(|i, builder| {
            let num_quotient_chunks = builder.get(&partial_vk.per_air, i).quotient_degree;
            builder.assign(num_quotient_mats, num_quotient_mats + num_quotient_chunks);
        });

        let mut quotient_mats: Array<_, TwoAdicPcsMatsVariable<_>> =
            builder.dyn_array(num_quotient_mats);
        let qc_index: Var<_> = builder.eval(C::N::zero());

        let mut qc_points = builder.dyn_array::<Ext<_, _>>(1);
        builder.set_value(&mut qc_points, 0, zeta);

        builder.assert_usize_eq(proof.opening.values.quotient.len(), num_airs);

        for i in 0..raps.len() {
            let opened_quotient = builder.get(&proof.opening.values.quotient, i);
            let qc_domains = builder.get(&mut quotient_chunk_domains, i);

            builder.range(0, qc_domains.len()).for_each(|j, builder| {
                let qc_dom = builder.get(&qc_domains, j);
                let qc_vals_array = builder.get(&opened_quotient, j);
                let mut qc_values = builder.dyn_array::<Array<C, _>>(1);
                builder.set_value(&mut qc_values, 0, qc_vals_array);
                let qc_mat = TwoAdicPcsMatsVariable::<C> {
                    domain: qc_dom,
                    values: qc_values,
                    points: qc_points.clone(),
                };
                builder.set_value(&mut quotient_mats, qc_index, qc_mat);
                builder.assign(qc_index, qc_index + C::N::one());
            });
        }
        let quotient_round = TwoAdicPcsRoundVariable {
            batch_commit: quotient_commit.clone(),
            mats: quotient_mats,
        };
        builder.set_value(&mut rounds, round_idx, quotient_round);
        builder.assign(round_idx, round_idx + C::N::one());

        builder.cycle_tracker("stage-c-build-rounds");

        // Verify the pcs proof
        builder.cycle_tracker("stage-d-verify-pcs");
        pcs.verify(builder, rounds, proof.opening.proof.clone(), challenger);
        builder.cycle_tracker("stage-d-verify-pcs");

        // TODO[sp1] CONSTRAIN: that the preprocessed chips get called with verify_constraints.
        builder.cycle_tracker("stage-e-verify-constraints");

        // TODO[zach]: make per phase; for now just 1 phase so OK
        let after_challenge_idx: Var<C::N> = builder.constant(C::N::zero());

        for (index, (&rap, chip_dim)) in raps.iter().zip_eq(chip_dims).enumerate() {
            let vk = builder.get(&partial_vk.per_air, index);

            // if rap.preprocessed_trace().is_some() {
            //     builder.assert_var_ne(index, C::N::from_canonical_usize(EMPTY));
            // }

            // FIXME: one matrix per commitment for now
            let main_values = builder.get(&proof.opening.values.main, 0);
            let main_values = builder.get(&main_values, index);

            // FIXME: one phase for now
            let after_challenge_values = builder.get(&proof.opening.values.after_challenge, 0);
            let after_challenge_values = builder.get(&after_challenge_values, after_challenge_idx);
            builder.assign(after_challenge_idx, after_challenge_idx + C::N::one());

            let trace_domain = builder.get(&trace_domains, index);
            let quotient_domain: TwoAdicMultiplicativeCosetVariable<_> =
                builder.get(&quotient_domains, index);

            // Check that the quotient data matches the chip's data.
            let log_quotient_degree = chip_dim.log_quotient_degree;

            let quotient_size = 1 << log_quotient_degree;
            builder.assert_usize_eq(vk.log_quotient_degree, log_quotient_degree);
            builder.assert_usize_eq(vk.quotient_degree, quotient_size);
            let quotient_chunks = builder.get(&proof.opening.values.quotient, index);

            // Get the domains from the chip itself.
            let qc_domains = quotient_domain.split_domains_const(builder, log_quotient_degree);

            // Get the exposed values after challenge.
            let mut exposed_values_after_challenge = Vec::new();

            let exposed_values = builder.get(&proof.exposed_values_after_challenge, index);
            for j in 0..1 {
                // FIXME
                let values = builder.get(&exposed_values, j);
                let mut values_vec = Vec::new();
                for k in 0..1 {
                    // FIXME
                    let value = builder.get(&values, k);
                    values_vec.push(value);
                }
                exposed_values_after_challenge.push(values_vec);
            }

            let pvs = builder.get(&public_values, index);
            Self::verify_single_rap_constraints(
                builder,
                rap,
                chip_dim,
                main_values,
                quotient_chunks,
                pvs,
                trace_domain,
                qc_domains,
                zeta,
                alpha,
                after_challenge_values,
                &challenges,
                &exposed_values_after_challenge,
            );
        }

        builder.cycle_tracker("stage-e-verify-constraints");
    }

    /// Reference: [afs_stark_backend::verifier::constraints::verify_single_rap_constraints]
    pub fn verify_single_rap_constraints<R>(
        builder: &mut Builder<C>,
        rap: &R,
        chip_dims: ChipDimensions,
        main_values: AdjacentOpenedValuesVariable<C>,
        quotient_chunks: Array<C, Array<C, Ext<C::F, C::EF>>>,
        public_values: Array<C, Felt<C::F>>,
        trace_domain: TwoAdicMultiplicativeCosetVariable<C>,
        qc_domains: Vec<TwoAdicMultiplicativeCosetVariable<C>>,
        zeta: Ext<C::F, C::EF>,
        alpha: Ext<C::F, C::EF>,
        after_challenge_values: AdjacentOpenedValuesVariable<C>,
        challenges: &[Vec<Ext<C::F, C::EF>>],
        exposed_values_after_challenge: &[Vec<Ext<C::F, C::EF>>],
    ) where
        R: for<'b> Rap<AxiomRecursiveVerifierConstraintFolder<'b, C>> + Sync + ?Sized,
    {
        let sels = trace_domain.selectors_at_point(builder, zeta);

        let mut main = AdjacentOpenedValues {
            local: vec![],
            next: vec![],
        };
        let main_width = chip_dims.main_width;
        // Assert that the length of the dynamic arrays match the expected length of the vectors.
        builder.assert_usize_eq(main_width, main_values.local.len());
        builder.assert_usize_eq(main_width, main_values.next.len());
        // Collect the main values into vectors.
        for i in 0..main_width {
            main.local.push(builder.get(&main_values.local, i));
            main.next.push(builder.get(&main_values.next, i));
        }

        let mut after_challenge = AdjacentOpenedValues {
            local: vec![],
            next: vec![],
        };

        let after_challenge_width = C::EF::D * chip_dims.permutation_width;
        builder.assert_usize_eq(after_challenge_width, after_challenge_values.local.len());
        builder.assert_usize_eq(after_challenge_width, after_challenge_values.next.len());
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
            rap,
            main,
            public_values,
            &sels,
            alpha,
            after_challenge,
            &challenges,
            exposed_values_after_challenge,
        );

        let num_quotient_chunks = 1 << chip_dims.log_quotient_degree;
        let mut quotient = vec![];
        // Assert that the length of the quotient chunk arrays match the expected length.
        builder.assert_usize_eq(num_quotient_chunks, quotient_chunks.len());
        // Collect the quotient values into vectors.
        for i in 0..num_quotient_chunks {
            let chunk = builder.get(&quotient_chunks, i);
            // Assert that the chunk length matches the expected length.
            builder.assert_usize_eq(C::EF::D, chunk.len());
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

    fn eval_constraints<R>(
        builder: &mut Builder<C>,
        rap: &R,
        main_values: AdjacentOpenedValues<Ext<C::F, C::EF>>,
        public_values: Array<C, Felt<C::F>>,
        selectors: &LagrangeSelectors<Ext<C::F, C::EF>>,
        alpha: Ext<C::F, C::EF>,
        after_challenge: AdjacentOpenedValues<Ext<C::F, C::EF>>,
        challenges: &[Vec<Ext<C::F, C::EF>>],
        exposed_values_after_challenge: &[Vec<Ext<C::F, C::EF>>],
    ) -> Ext<C::F, C::EF>
    where
        R: for<'b> Rap<AxiomRecursiveVerifierConstraintFolder<'b, C>> + Sync + ?Sized,
    {
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
        for i in 0..PROOF_MAX_NUM_PVS {
            folder_pv.push(builder.get(&public_values, i));
        }

        let mut folder = AxiomRecursiveVerifierConstraintFolder::<C> {
            preprocessed: VerticalPair::new(
                RowMajorMatrixView::new_row(&[]),
                RowMajorMatrixView::new_row(&[]),
            ), // TODO[zach]: support preprocessed trace
            partitioned_main: vec![VerticalPair::new(
                RowMajorMatrixView::new_row(&main_values.local),
                RowMajorMatrixView::new_row(&main_values.next),
            )],
            after_challenge: vec![VerticalPair::new(
                RowMajorMatrixView::new_row(&after_challenge_values.local),
                RowMajorMatrixView::new_row(&after_challenge_values.next),
            )],
            challenges: &challenges,
            is_first_row: selectors.is_first_row,
            is_last_row: selectors.is_last_row,
            is_transition: selectors.is_transition,
            alpha,
            accumulator: SymbolicExt::zero(),
            public_values: &folder_pv,
            exposed_values_after_challenge, // FIXME
        };

        rap.eval(&mut folder);
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
