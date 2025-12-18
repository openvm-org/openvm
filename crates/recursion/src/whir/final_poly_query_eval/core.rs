use core::borrow::{Borrow, BorrowMut};

use openvm_circuit_primitives::{SubAir, utils::assert_array_eq};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{FieldAlgebra, FieldExtensionAlgebra, extension::BinomiallyExtendable};
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use p3_maybe_rayon::prelude::*;
use stark_backend_v2::{
    D_EF, EF, F, SystemParams, keygen::types::MultiStarkVerifyingKeyV2, proof::Proof,
};
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::{
    subairs::nested_for_loop::{NestedForLoopAuxCols, NestedForLoopIoCols, NestedForLoopSubAir},
    system::Preflight,
    utils::{eq_1, ext_field_add, ext_field_multiply},
    whir::bus::{
        FinalPolyQueryEvalBus, FinalPolyQueryEvalMessage, WhirAlphaBus, WhirAlphaMessage,
        WhirFinalPolyBus, WhirFinalPolyBusMessage, WhirGammaBus, WhirGammaMessage, WhirQueryBus,
        WhirQueryBusMessage,
    },
};

#[repr(C)]
#[derive(AlignedBorrow)]
pub(in crate::whir::final_poly_query_eval) struct FinalyPolyQueryEvalCols<T> {
    is_enabled: T,
    // loop indices
    proof_idx: T,
    whir_round: T,
    query_idx: T,
    phase_idx: T,
    eval_idx: T,
    // is_first flags
    is_first_in_proof: T,
    is_first_in_round: T,
    is_first_in_query: T,
    is_first_in_phase: T,
    is_last_round: T,
    is_query_zero: T,
    query_pow: [T; D_EF],
    alpha: [T; D_EF],
    gamma: [T; D_EF],
    gamma_pow: [T; D_EF],
    final_poly_coeff: [T; D_EF],
    final_value_acc: [T; D_EF],
    gamma_eq_acc: [T; D_EF],
    horner_acc: [T; D_EF],
    do_carry: T,
}

#[derive(Debug)]
pub struct FinalPolyQueryEvalAir {
    pub query_bus: WhirQueryBus,
    pub alpha_bus: WhirAlphaBus,
    pub gamma_bus: WhirGammaBus,
    pub final_poly_bus: WhirFinalPolyBus,
    pub final_poly_query_eval_bus: FinalPolyQueryEvalBus,
    pub num_whir_rounds: usize,
    pub k_whir: usize,
    pub log_final_poly_len: usize,
}

impl BaseAirWithPublicValues<F> for FinalPolyQueryEvalAir {}
impl PartitionedBaseAir<F> for FinalPolyQueryEvalAir {}

impl<F> BaseAir<F> for FinalPolyQueryEvalAir {
    fn width(&self) -> usize {
        FinalyPolyQueryEvalCols::<F>::width()
    }
}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for FinalPolyQueryEvalAir
where
    <AB::Expr as FieldAlgebra>::F: BinomiallyExtendable<D_EF>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (main.row_slice(0), main.row_slice(1));
        let local: &FinalyPolyQueryEvalCols<AB::Var> = (*local).borrow();
        let next: &FinalyPolyQueryEvalCols<AB::Var> = (*next).borrow();

        let k_whir_f = AB::Expr::from_canonical_usize(self.k_whir);
        let eq_phase_len = k_whir_f.clone()
            * (AB::Expr::from_canonical_usize(self.num_whir_rounds - 1) - local.whir_round);
        let final_poly_phase_len =
            AB::Expr::from_canonical_usize((1 << self.log_final_poly_len) - 1);

        let round_base = (local.whir_round + AB::Expr::ONE) * k_whir_f;

        let proof_idx = local.proof_idx;
        let local_is_eq_phase = local.is_enabled - local.phase_idx;

        builder.assert_bool(local.is_enabled);
        builder.assert_bool(local.phase_idx);

        builder
            .when(local.is_first_in_proof)
            .assert_one(local.is_enabled);
        builder
            .when(local.is_first_in_round)
            .assert_one(local.is_enabled);
        builder
            .when(local.is_first_in_query)
            .assert_one(local.is_enabled);
        builder
            .when(local.is_first_in_phase)
            .assert_one(local.is_enabled);

        let is_same_proof = next.is_enabled - next.is_first_in_proof;
        let is_same_round = next.is_enabled - next.is_first_in_round;
        let is_same_query = next.is_enabled - next.is_first_in_query;
        let is_same_phase = next.is_enabled - next.is_first_in_phase;

        builder
            .when(local.phase_idx)
            .when(local.is_enabled - is_same_phase.clone())
            .assert_eq(local.eval_idx, final_poly_phase_len.clone());

        builder
            .when(AB::Expr::ONE - local.phase_idx)
            .when(local.is_enabled - is_same_phase.clone())
            .assert_eq(local.eval_idx, eq_phase_len - AB::Expr::ONE);

        NestedForLoopSubAir.eval(
            builder,
            (
                (
                    NestedForLoopIoCols {
                        is_enabled: local.is_enabled,
                        counter: [
                            local.proof_idx,
                            local.whir_round,
                            local.query_idx,
                            local.phase_idx,
                            local.eval_idx,
                        ],
                        is_first: [
                            local.is_first_in_proof,
                            local.is_first_in_round,
                            local.is_first_in_query,
                            local.is_first_in_phase,
                            local.is_enabled,
                        ],
                    }
                    .map_into(),
                    NestedForLoopIoCols {
                        is_enabled: next.is_enabled,
                        counter: [
                            next.proof_idx,
                            next.whir_round,
                            next.query_idx,
                            next.phase_idx,
                            next.eval_idx,
                        ],
                        is_first: [
                            next.is_first_in_proof,
                            next.is_first_in_round,
                            next.is_first_in_query,
                            next.is_first_in_phase,
                            next.is_enabled,
                        ],
                    }
                    .map_into(),
                ),
                NestedForLoopAuxCols {
                    is_transition: [
                        is_same_proof.clone(),
                        is_same_round.clone(),
                        is_same_query.clone(),
                        is_same_phase.clone(),
                    ],
                },
            ),
        );

        assert_array_eq(
            &mut builder.when(local_is_eq_phase.clone()),
            next.query_pow,
            ext_field_multiply::<AB::Expr>(local.query_pow, local.query_pow),
        );

        assert_array_eq(
            &mut builder.when(local.phase_idx * is_same_query.clone()),
            next.query_pow,
            local.query_pow,
        );

        // The value of post_horner_acc *after* this inner-most loop iteration, if we are not in
        // eq_phase. Degree 2.
        let post_horner_acc = ext_field_add(
            ext_field_multiply(local.horner_acc, local.query_pow),
            local.final_poly_coeff,
        );

        assert_array_eq(
            &mut builder.when(local.is_first_in_query),
            local.gamma_eq_acc,
            local.gamma_pow,
        );
        assert_array_eq(
            &mut builder.when(local_is_eq_phase.clone()),
            next.gamma_eq_acc,
            ext_field_multiply(local.gamma_eq_acc, eq_1(local.alpha, local.query_pow)),
        );
        assert_array_eq(
            &mut builder.when((AB::Expr::ONE - local_is_eq_phase.clone()) * is_same_query.clone()),
            next.gamma_eq_acc,
            local.gamma_eq_acc,
        );
        assert_array_eq(
            &mut builder.when(local.is_first_in_round),
            local.gamma_pow,
            local.gamma,
        );
        assert_array_eq(
            &mut builder.when(is_same_query.clone()),
            next.gamma_pow,
            local.gamma_pow,
        );
        assert_array_eq(
            &mut builder.when(is_same_round.clone() * next.is_first_in_query),
            next.gamma_pow,
            ext_field_multiply(local.gamma_pow, local.gamma),
        );

        assert_array_eq(
            &mut builder.when(is_same_query.clone()),
            next.final_value_acc,
            local.final_value_acc,
        );
        builder
            .when(local.is_first_in_round)
            .assert_one(local.is_query_zero);
        builder
            .when(is_same_query.clone())
            .assert_eq(local.is_query_zero, next.is_query_zero);
        builder
            .when(local.query_idx)
            .assert_zero(local.is_query_zero);
        builder
            .when(local.is_enabled - is_same_proof.clone())
            .assert_one(local.is_last_round);
        builder
            .when(is_same_round.clone())
            .assert_eq(local.is_last_round, next.is_last_round);
        builder
            .when(local.whir_round - AB::Expr::from_canonical_usize(self.num_whir_rounds - 1))
            .assert_zero(local.is_last_round);

        builder.assert_bool(local.do_carry);
        builder.assert_eq(
            local.is_enabled * local.do_carry,
            (is_same_proof.clone() - is_same_query.clone())
                * (AB::Expr::ONE - local.is_query_zero * local.is_last_round),
        );

        let gamma_eq_post = ext_field_multiply(local.gamma_eq_acc, post_horner_acc.clone());

        assert_array_eq(
            &mut builder.when(local_is_eq_phase.clone()),
            local.horner_acc,
            [AB::F::ZERO; 4],
        );
        assert_array_eq(
            &mut builder.when(is_same_query.clone()),
            next.horner_acc,
            post_horner_acc.clone(),
        );
        let contrib = ext_field_multiply::<AB::Expr>(local.gamma_eq_acc, post_horner_acc.clone());
        let post_final_value_acc = ext_field_add(local.final_value_acc, contrib);
        assert_array_eq(
            &mut builder.when(local.do_carry),
            next.final_value_acc,
            post_final_value_acc,
        );
        assert_array_eq(
            &mut builder.when(
                (is_same_proof.clone() - is_same_query.clone())
                    * (local.is_query_zero * local.is_last_round),
            ),
            next.final_value_acc,
            local.final_value_acc,
        );

        self.query_bus.receive(
            builder,
            proof_idx,
            WhirQueryBusMessage {
                whir_round: local.whir_round,
                query_idx: local.query_idx,
                value: local.query_pow,
            },
            local.is_first_in_query,
        );
        let alpha_idx = round_base + local.eval_idx;
        self.alpha_bus.lookup_key(
            builder,
            proof_idx,
            WhirAlphaMessage {
                idx: alpha_idx,
                challenge: local.alpha.map(Into::into),
            },
            local_is_eq_phase.clone(),
        );
        self.gamma_bus.receive(
            builder,
            proof_idx,
            WhirGammaMessage {
                idx: local.whir_round,
                challenge: local.gamma,
            },
            local.is_first_in_round,
        );
        assert_array_eq(
            &mut builder.when(is_same_round.clone()),
            local.gamma,
            next.gamma,
        );
        let is_last = local.is_enabled - is_same_proof.clone();
        let final_value = ext_field_add(local.final_value_acc, gamma_eq_post);
        self.final_poly_query_eval_bus.receive(
            builder,
            proof_idx,
            FinalPolyQueryEvalMessage {
                last_whir_round: AB::Expr::from_canonical_usize(self.num_whir_rounds),
                value: final_value,
            },
            is_last,
        );

        let coeff_idx = final_poly_phase_len - local.eval_idx;
        self.final_poly_bus.receive(
            builder,
            proof_idx,
            WhirFinalPolyBusMessage {
                idx: coeff_idx,
                coeff: local.final_poly_coeff.map(Into::into),
            },
            local.is_enabled * local.phase_idx,
        );
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(in crate::whir) struct FinalPolyQueryEvalRecord {
    pub alpha: EF,
    pub query_pow: EF,
    pub gamma_eq_acc: EF,
    pub horner_acc: EF,
    pub final_poly_coeff: EF,
    pub final_value_acc: EF,
    pub gamma: EF,
    pub gamma_pow: EF,
}

pub(in crate::whir) fn build_final_poly_query_eval_records(
    params: &SystemParams,
    proofs: &[&Proof],
    preflights: &[&Preflight],
) -> Vec<FinalPolyQueryEvalRecord> {
    debug_assert_eq!(proofs.len(), preflights.len());
    let k_whir = params.k_whir();
    let num_queries_per_round: Vec<usize> =
        params.whir.rounds.iter().map(|r| r.num_queries).collect();
    let final_poly_len = 1usize << params.log_final_poly_len();
    let num_whir_rounds = params.num_whir_rounds();

    let mut rows_per_proof = 0usize;
    for (whir_round, &num_queries) in num_queries_per_round.iter().enumerate() {
        let eq_phase_len = k_whir * (num_whir_rounds - (whir_round + 1));
        let query_count = num_queries + 1;
        rows_per_proof += query_count * (eq_phase_len + final_poly_len);
    }
    let mut records = Vec::with_capacity(rows_per_proof * proofs.len());

    for (proof_idx, proof) in proofs.iter().enumerate() {
        let preflight = preflights[proof_idx];

        let final_poly_coeffs = &proof.whir_proof.final_poly;
        debug_assert_eq!(final_poly_coeffs.len(), final_poly_len);

        let gammas = &preflight.whir.gammas;
        let z0s = &preflight.whir.z0s;
        let zis = &preflight.whir.zjs;
        let alphas = &preflight.whir.alphas;

        let mut final_value_acc = EF::ZERO;
        for whir_round in 0..num_whir_rounds {
            let eq_phase_len = k_whir * (num_whir_rounds - (whir_round + 1));
            let gamma = gammas[whir_round];
            let mut gamma_pow = gamma;

            let query_count = num_queries_per_round[whir_round] + 1;
            for query_idx in 0..query_count {
                let mut gamma_eq_acc = gamma_pow;
                let mut horner_acc = EF::ZERO;
                let mut query_pow = if query_idx == 0 {
                    if whir_round < num_whir_rounds - 1 {
                        z0s[whir_round]
                    } else {
                        EF::ZERO
                    }
                } else {
                    EF::from_base(zis[whir_round][query_idx - 1])
                };

                for eval_idx in 0..eq_phase_len {
                    let alpha = alphas[(whir_round + 1) * k_whir + eval_idx];
                    records.push(FinalPolyQueryEvalRecord {
                        alpha,
                        query_pow,
                        gamma_eq_acc,
                        horner_acc,
                        final_poly_coeff: EF::ZERO,
                        final_value_acc,
                        gamma,
                        gamma_pow,
                    });
                    gamma_eq_acc *= EF::ONE - query_pow - alpha + query_pow * alpha.double();
                    query_pow *= query_pow;
                }

                for &final_poly_coeff in final_poly_coeffs.iter().rev() {
                    records.push(FinalPolyQueryEvalRecord {
                        alpha: EF::ZERO,
                        query_pow,
                        gamma_eq_acc,
                        horner_acc,
                        final_poly_coeff,
                        final_value_acc,
                        gamma,
                        gamma_pow,
                    });
                    horner_acc = horner_acc * query_pow + final_poly_coeff;
                }

                if query_idx != 0 || whir_round != num_whir_rounds - 1 {
                    final_value_acc += gamma_eq_acc * horner_acc;
                }

                gamma_pow *= gamma;
            }
        }
    }

    records
}

#[inline]
pub(in crate::whir) fn compute_round_offsets(
    num_whir_rounds: usize,
    k_whir: usize,
    final_poly_len: usize,
    num_queries_per_round: &[usize],
) -> Vec<usize> {
    let mut offsets = Vec::with_capacity(num_whir_rounds + 1);
    offsets.push(0);
    let mut rows_acc = 0;
    for (whir_round, &num_queries) in num_queries_per_round.iter().enumerate() {
        let eq_phase_len = k_whir * (num_whir_rounds - (whir_round + 1));
        let query_count = num_queries + 1;
        let rows_per_round = query_count * (eq_phase_len + final_poly_len);
        rows_acc += rows_per_round;
        offsets.push(rows_acc);
    }
    offsets
}

#[inline]
fn compute_indices_from_row_idx(
    row_idx: usize,
    rows_per_proof: usize,
    round_offsets: &[usize],
    num_whir_rounds: usize,
    k_whir: usize,
    final_poly_len: usize,
) -> (usize, usize, usize, usize, usize, usize) {
    debug_assert!(rows_per_proof > 0);
    debug_assert!(!round_offsets.is_empty());
    debug_assert_eq!(*round_offsets.last().unwrap(), rows_per_proof);

    let proof_idx = row_idx / rows_per_proof;
    let row_in_proof = row_idx % rows_per_proof;

    let round_idx = round_offsets
        .partition_point(|&start| start <= row_in_proof)
        .saturating_sub(1);
    debug_assert!(round_idx + 1 < round_offsets.len());

    let round_start = round_offsets[round_idx];
    let row_in_round = row_in_proof - round_start;

    let eq_phase_len_round = k_whir * (num_whir_rounds - (round_idx + 1));
    let rows_per_query = eq_phase_len_round + final_poly_len;

    let query_idx = row_in_round / rows_per_query;
    let row_in_query = row_in_round % rows_per_query;

    let (phase_idx, eval_idx) = if row_in_query < eq_phase_len_round {
        (0, row_in_query)
    } else {
        (1, row_in_query - eq_phase_len_round)
    };

    (
        proof_idx,
        round_idx,
        query_idx,
        phase_idx,
        eval_idx,
        eq_phase_len_round,
    )
}

#[tracing::instrument(level = "trace", skip_all)]
pub(crate) fn generate_trace(
    mvk: &MultiStarkVerifyingKeyV2,
    _proofs: &[&Proof],
    records: &[FinalPolyQueryEvalRecord],
) -> RowMajorMatrix<F> {
    let params = &mvk.inner.params;
    let k_whir = params.k_whir();
    let num_queries_per_round: Vec<usize> =
        params.whir.rounds.iter().map(|r| r.num_queries).collect();
    let final_poly_len = 1usize << params.log_final_poly_len();
    let num_whir_rounds = params.num_whir_rounds();

    let round_offsets = compute_round_offsets(
        num_whir_rounds,
        k_whir,
        final_poly_len,
        &num_queries_per_round,
    );
    let rows_per_proof = *round_offsets
        .last()
        .expect("round offsets vector must include sentinel");
    debug_assert!(rows_per_proof > 0);
    let total_valid_rows = records.len();
    let height = total_valid_rows.next_power_of_two();
    let width = FinalyPolyQueryEvalCols::<F>::width();
    let mut trace = F::zero_vec(width * height);
    trace
        .par_chunks_exact_mut(width)
        .zip(records.par_iter())
        .enumerate()
        .for_each(|(row_idx, (row, record))| {
            let (proof_idx, whir_round, query_idx, phase_idx, eval_idx, num_alphas) =
                compute_indices_from_row_idx(
                    row_idx,
                    rows_per_proof,
                    &round_offsets,
                    num_whir_rounds,
                    k_whir,
                    final_poly_len,
                );

            let is_first_in_phase = eval_idx == 0;
            let is_first_in_query = is_first_in_phase && (phase_idx == 0 || num_alphas == 0);
            let is_first_in_round = is_first_in_query && query_idx == 0;
            let is_first_in_proof = is_first_in_round && whir_round == 0;

            let num_in_domain_queries = num_queries_per_round[whir_round];
            let is_same_phase = if phase_idx == 0 {
                eval_idx + 1 < num_alphas
            } else {
                eval_idx + 1 < final_poly_len
            };
            let is_same_query = is_same_phase || phase_idx == 0;
            let is_same_round = is_same_query || query_idx < num_in_domain_queries;
            let is_same_proof = is_same_round || whir_round + 1 < num_whir_rounds;

            let cols: &mut FinalyPolyQueryEvalCols<F> = row.borrow_mut();
            cols.is_enabled = F::ONE;
            cols.proof_idx = F::from_canonical_usize(proof_idx);
            cols.whir_round = F::from_canonical_usize(whir_round);
            cols.query_idx = F::from_canonical_usize(query_idx);
            cols.phase_idx = F::from_canonical_usize(phase_idx);
            cols.eval_idx = F::from_canonical_usize(eval_idx);
            cols.is_first_in_proof = F::from_bool(is_first_in_proof);
            cols.is_first_in_round = F::from_bool(is_first_in_round);
            cols.is_first_in_query = F::from_bool(is_first_in_query);
            cols.is_first_in_phase = F::from_bool(is_first_in_phase);
            cols.is_query_zero = F::from_bool(query_idx == 0);
            cols.is_last_round = F::from_bool(whir_round + 1 == num_whir_rounds);

            let is_q0_last = query_idx == 0 && (whir_round + 1 == num_whir_rounds);
            let do_carry = (!is_same_query) && is_same_proof && !is_q0_last;
            cols.do_carry = F::from_bool(do_carry);

            cols.gamma_eq_acc
                .copy_from_slice(record.gamma_eq_acc.as_base_slice());

            cols.alpha.copy_from_slice(record.alpha.as_base_slice());
            cols.gamma.copy_from_slice(record.gamma.as_base_slice());
            cols.gamma_pow
                .copy_from_slice(record.gamma_pow.as_base_slice());

            cols.query_pow
                .copy_from_slice(record.query_pow.as_base_slice());
            cols.final_poly_coeff
                .copy_from_slice(record.final_poly_coeff.as_base_slice());
            cols.final_value_acc
                .copy_from_slice(record.final_value_acc.as_base_slice());
            cols.horner_acc
                .copy_from_slice(record.horner_acc.as_base_slice());
        });

    RowMajorMatrix::new(trace, width)
}
