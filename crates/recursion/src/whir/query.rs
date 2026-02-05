use core::borrow::{Borrow, BorrowMut};

use openvm_circuit_primitives::{SubAir, utils::assert_array_eq};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{
    BasedVectorSpace, PrimeCharacteristicRing, TwoAdicField, extension::BinomiallyExtendable,
};
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use p3_maybe_rayon::prelude::*;
use stark_backend_v2::{D_EF, F, keygen::types::MultiStarkVerifyingKeyV2, proof::Proof};
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::{
    bus::TranscriptBus,
    primitives::bus::{ExpBitsLenBus, ExpBitsLenMessage},
    subairs::nested_for_loop::{NestedForLoopAuxCols, NestedForLoopIoCols, NestedForLoopSubAir},
    system::Preflight,
    utils::{ext_field_add, ext_field_multiply},
    whir::bus::{
        VerifyQueriesBus, VerifyQueriesBusMessage, VerifyQueryBus, VerifyQueryBusMessage,
        WhirQueryBus, WhirQueryBusMessage,
    },
};

#[repr(C)]
#[derive(AlignedBorrow)]
struct WhirQueryCols<T> {
    is_enabled: T,
    // loop counters
    proof_idx: T,
    whir_round: T,
    query_idx: T,
    // first flags
    is_first_in_proof: T,
    is_first_in_round: T,
    tidx: T,
    num_queries: T,
    omega: T,
    sample: T,
    zi_root: T,
    zi: T,
    yi: [T; D_EF],
    gamma: [T; D_EF],
    gamma_pow: [T; D_EF],
    pre_claim: [T; D_EF],
    post_claim: [T; D_EF],
}

pub struct WhirQueryAir {
    pub transcript_bus: TranscriptBus,
    pub verify_queries_bus: VerifyQueriesBus,
    pub query_bus: WhirQueryBus,
    pub verify_query_bus: VerifyQueryBus,
    pub exp_bits_len_bus: ExpBitsLenBus,
    pub k: usize,
    pub initial_log_domain_size: usize,
}

impl BaseAirWithPublicValues<F> for WhirQueryAir {}
impl PartitionedBaseAir<F> for WhirQueryAir {}

impl<F> BaseAir<F> for WhirQueryAir {
    fn width(&self) -> usize {
        WhirQueryCols::<F>::width()
    }
}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for WhirQueryAir
where
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<D_EF>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let (local, next) = (
            main.row_slice(0).expect("window should have two elements"),
            main.row_slice(1).expect("window should have two elements"),
        );
        let local: &WhirQueryCols<AB::Var> = (*local).borrow();
        let next: &WhirQueryCols<AB::Var> = (*next).borrow();

        let proof_idx = local.proof_idx;
        let is_enabled = local.is_enabled;
        builder.assert_bool(is_enabled);
        builder.when(local.is_first_in_proof).assert_one(is_enabled);
        builder.when(local.is_first_in_round).assert_one(is_enabled);

        let is_same_proof = next.is_enabled - next.is_first_in_proof;
        let is_same_round = next.is_enabled - next.is_first_in_round;

        NestedForLoopSubAir.eval(
            builder,
            (
                (
                    NestedForLoopIoCols {
                        is_enabled,
                        counter: [local.proof_idx, local.whir_round, local.query_idx],
                        is_first: [local.is_first_in_proof, local.is_first_in_round, is_enabled],
                    }
                    .map_into(),
                    NestedForLoopIoCols {
                        is_enabled: next.is_enabled,
                        counter: [next.proof_idx, next.whir_round, next.query_idx],
                        is_first: [
                            next.is_first_in_proof,
                            next.is_first_in_round,
                            next.is_enabled,
                        ],
                    }
                    .map_into(),
                ),
                NestedForLoopAuxCols {
                    is_transition: [is_same_proof.clone(), is_same_round.clone()],
                },
            ),
        );

        assert_array_eq(
            &mut builder.when(local.is_first_in_round),
            local.gamma_pow,
            ext_field_multiply::<AB::Expr>(local.gamma, local.gamma),
        );

        builder
            .when(local.is_enabled - is_same_round.clone())
            .assert_eq(local.query_idx + AB::Expr::ONE, local.num_queries);

        // num_queries is constrained via the bus interaction with WhirRoundAir
        self.verify_queries_bus.receive(
            builder,
            proof_idx,
            VerifyQueriesBusMessage {
                tidx: local.tidx,
                whir_round: local.whir_round,
                num_queries: local.num_queries,
                gamma: local.gamma,
                pre_claim: local.pre_claim,
                post_claim: local.post_claim,
            },
            local.is_first_in_round,
        );
        assert_array_eq(
            &mut builder.when(is_same_round.clone()),
            next.gamma,
            local.gamma,
        );
        assert_array_eq(
            &mut builder.when(is_same_round.clone()),
            next.gamma_pow,
            ext_field_multiply::<AB::Expr>(local.gamma, local.gamma_pow),
        );
        assert_array_eq(
            &mut builder.when(is_same_round.clone()),
            next.pre_claim,
            ext_field_add::<AB::Expr>(
                local.pre_claim,
                ext_field_multiply::<AB::Expr>(local.gamma_pow, local.yi),
            ),
        );
        assert_array_eq(
            &mut builder.when(is_same_round.clone()),
            next.post_claim,
            local.post_claim,
        );
        assert_array_eq(
            &mut builder.when(local.is_enabled - is_same_round.clone()),
            ext_field_add::<AB::Expr>(
                local.pre_claim,
                ext_field_multiply::<AB::Expr>(local.yi, local.gamma_pow),
            ),
            local.post_claim,
        );

        self.transcript_bus
            .sample(builder, proof_idx, local.tidx, local.sample, is_enabled);
        self.query_bus.send(
            builder,
            proof_idx,
            WhirQueryBusMessage {
                whir_round: local.whir_round.into(),
                query_idx: local.query_idx.into() + AB::Expr::ONE,
                value: [
                    local.zi.into(),
                    AB::Expr::ZERO,
                    AB::Expr::ZERO,
                    AB::Expr::ZERO,
                ],
            },
            is_enabled,
        );
        self.verify_query_bus.send(
            builder,
            proof_idx,
            VerifyQueryBusMessage {
                whir_round: local.whir_round,
                query_idx: local.query_idx,
                merkle_idx_bit_src: local.sample,
                zi_root: local.zi_root,
                zi: local.zi,
                yi: local.yi,
            },
            is_enabled,
        );
        self.exp_bits_len_bus.lookup_key(
            builder,
            ExpBitsLenMessage {
                base: local.omega.into(),
                bit_src: local.sample.into(),
                num_bits: AB::Expr::from_usize(self.initial_log_domain_size - self.k)
                    - local.whir_round,
                result: local.zi_root.into(),
            },
            is_enabled,
        );
    }
}

#[tracing::instrument(level = "trace", skip_all)]
pub(crate) fn generate_trace(
    mvk: &MultiStarkVerifyingKeyV2,
    proofs: &[&Proof],
    preflights: &[&Preflight],
) -> RowMajorMatrix<F> {
    debug_assert_eq!(proofs.len(), preflights.len());

    let params = &mvk.inner.params;
    let m = params.n_stack + params.l_skip + params.log_blowup;

    let num_queries_per_round: Vec<usize> =
        params.whir.rounds.iter().map(|r| r.num_queries).collect();
    let num_whir_rounds = params.num_whir_rounds();

    let mut round_row_offsets = Vec::with_capacity(num_whir_rounds + 1);
    round_row_offsets.push(0usize);
    for &num_queries in &num_queries_per_round {
        round_row_offsets.push(round_row_offsets.last().unwrap() + num_queries);
    }
    let num_rows_per_proof = *round_row_offsets.last().unwrap();

    let num_valid_rows: usize = num_rows_per_proof * preflights.len();

    let height = num_valid_rows.next_power_of_two();
    let width = WhirQueryCols::<usize>::width();
    let mut trace = F::zero_vec(width * height);

    trace
        .par_chunks_exact_mut(width)
        .take(num_valid_rows)
        .enumerate()
        .for_each(|(row_idx, row)| {
            let proof_idx = row_idx / num_rows_per_proof;
            let i = row_idx % num_rows_per_proof;

            let whir_round = round_row_offsets[1..].partition_point(|&offset| offset <= i);
            let query_idx = i - round_row_offsets[whir_round];
            let num_queries = num_queries_per_round[whir_round];

            let preflight = &preflights[proof_idx];
            let query_offset = preflight.whir.query_offsets[whir_round];

            let cols: &mut WhirQueryCols<F> = row.borrow_mut();
            cols.is_enabled = F::ONE;
            cols.proof_idx = F::from_usize(proof_idx);
            cols.is_first_in_proof = F::from_bool(whir_round == 0 && query_idx == 0);
            cols.is_first_in_round = F::from_bool(query_idx == 0);
            cols.tidx = F::from_usize(preflight.whir.query_tidx_per_round[whir_round] + query_idx);
            cols.sample = preflight.whir.queries[query_offset + query_idx];
            cols.whir_round = F::from_usize(whir_round);
            cols.query_idx = F::from_usize(query_idx);
            cols.num_queries = F::from_usize(num_queries);
            cols.omega = F::two_adic_generator(m - whir_round);
            cols.zi = preflight.whir.zjs[whir_round][query_idx];
            cols.zi_root = preflight.whir.zj_roots[whir_round][query_idx];
            cols.yi.copy_from_slice(
                preflight.whir.yjs[whir_round][query_idx].as_basis_coefficients_slice(),
            );
            let gamma = preflight.whir.gammas[whir_round];
            cols.gamma
                .copy_from_slice(gamma.as_basis_coefficients_slice());
            let gamma_pow = gamma.exp_u64(query_idx as u64 + 2);
            cols.gamma_pow
                .copy_from_slice(gamma_pow.as_basis_coefficients_slice());
            let mut pre_claim = preflight.whir.pre_query_claims[whir_round];
            for (q, gamma_pow) in gamma.powers().skip(2).take(query_idx).enumerate() {
                pre_claim += gamma_pow * preflight.whir.yjs[whir_round][q];
            }
            if query_idx == num_queries - 1 {
                debug_assert_eq!(
                    pre_claim + gamma_pow * preflight.whir.yjs[whir_round][query_idx],
                    preflight.whir.initial_claim_per_round[whir_round + 1]
                );
            }
            cols.pre_claim
                .copy_from_slice(pre_claim.as_basis_coefficients_slice());
            cols.post_claim.copy_from_slice(
                preflight.whir.initial_claim_per_round[whir_round + 1]
                    .as_basis_coefficients_slice(),
            );
        });

    RowMajorMatrix::new(trace, width)
}
