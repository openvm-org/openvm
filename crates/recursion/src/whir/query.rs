use core::borrow::{Borrow, BorrowMut};

use openvm_stark_backend::{
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{FieldAlgebra, FieldExtensionAlgebra, TwoAdicField};
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use stark_backend_v2::{D_EF, F, keygen::types::MultiStarkVerifyingKeyV2, proof::Proof};
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::{
    bus::{ExpBitsLenBus, ExpBitsLenMessage, TranscriptBus},
    system::Preflight,
    whir::bus::{
        VerifyQueriesBus, VerifyQueriesBusMessage, VerifyQueryBus, VerifyQueryBusMessage,
        WhirQueryBus, WhirQueryBusMessage,
    },
};

#[repr(C)]
#[derive(AlignedBorrow)]
struct WhirQueryCols<T> {
    is_valid: T,
    is_first_in_group: T,
    proof_idx: T,
    tidx: T,
    whir_round: T,
    is_initial_round: T,
    query_idx: T,
    omega: T,
    num_bits: T,
    sample: T,
    zi_root: T,
    zi: T,
    yi: [T; D_EF],
    gamma: [T; D_EF],
    pre_claim: [T; D_EF],
    post_claim: [T; D_EF],
}

// Temporary dummy AIR to represent this module.
pub struct WhirQueryAir {
    pub transcript_bus: TranscriptBus,
    pub verify_queries_bus: VerifyQueriesBus,
    pub query_bus: WhirQueryBus,
    pub verify_query_bus: VerifyQueryBus,
    pub exp_bits_len_bus: ExpBitsLenBus,
}

impl BaseAirWithPublicValues<F> for WhirQueryAir {}
impl PartitionedBaseAir<F> for WhirQueryAir {}

impl<F> BaseAir<F> for WhirQueryAir {
    fn width(&self) -> usize {
        WhirQueryCols::<F>::width()
    }
}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for WhirQueryAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let local = main.row_slice(0);
        let local: &WhirQueryCols<AB::Var> = (*local).borrow();

        self.verify_queries_bus.receive(
            builder,
            local.proof_idx,
            VerifyQueriesBusMessage {
                tidx: local.tidx,
                whir_round: local.whir_round,
                gamma: local.gamma,
                pre_claim: local.pre_claim,
                post_claim: local.post_claim,
            },
            local.is_first_in_group,
        );

        self.transcript_bus.sample(
            builder,
            local.proof_idx,
            local.tidx,
            local.sample,
            local.is_valid,
        );
        self.query_bus.send(
            builder,
            local.proof_idx,
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
            local.is_valid,
        );
        self.verify_query_bus.send(
            builder,
            local.proof_idx,
            VerifyQueryBusMessage {
                whir_round: local.whir_round,
                query_idx: local.query_idx,
                zi_root: local.zi_root,
                zi: local.zi,
                yi: local.yi,
            },
            local.is_valid,
        );
        self.exp_bits_len_bus.lookup_key(
            builder,
            ExpBitsLenMessage {
                base: local.omega,
                bit_src: local.sample,
                num_bits: local.num_bits,
                result: local.zi_root,
            },
            local.is_valid,
        );
    }
}

pub(crate) fn generate_trace(
    mvk: &MultiStarkVerifyingKeyV2,
    _proofs: &[Proof],
    preflights: &[Preflight],
) -> RowMajorMatrix<F> {
    let params = mvk.inner.params;

    let num_rows_per_proof = params.num_whir_rounds() * params.num_whir_queries;
    let num_valid_rows: usize = num_rows_per_proof * preflights.len();
    let height = num_valid_rows.next_power_of_two();
    let width = WhirQueryCols::<usize>::width();
    let mut trace = vec![F::ZERO; height * width];

    let m = params.n_stack + params.l_skip + params.log_blowup;

    for (row_idx, row) in trace.chunks_mut(width).take(num_valid_rows).enumerate() {
        let proof_idx = row_idx / num_rows_per_proof;
        let i = row_idx % num_rows_per_proof;

        let preflight = &preflights[proof_idx];

        let cols: &mut WhirQueryCols<F> = row.borrow_mut();
        cols.is_valid = F::ONE;
        cols.proof_idx = F::from_canonical_usize(proof_idx);
        let whir_round = i / params.num_whir_queries;
        let query_idx = i % params.num_whir_queries;

        cols.tidx =
            F::from_canonical_usize(preflight.whir.query_tidx_per_round[whir_round] + query_idx);
        cols.is_first_in_group = F::from_bool(query_idx == 0);
        cols.sample = preflight.whir.queries[i];
        cols.whir_round = F::from_canonical_usize(whir_round);
        cols.query_idx = F::from_canonical_usize(query_idx);
        cols.omega = F::two_adic_generator(m - whir_round - params.k_whir);
        cols.num_bits = F::from_canonical_usize(m - whir_round - params.k_whir);
        cols.zi = preflight.whir.zjs[whir_round][query_idx];
        cols.zi_root = preflight.whir.zj_roots[whir_round][query_idx];
        cols.yi = preflight.whir.yjs[whir_round][query_idx]
            .as_base_slice()
            .try_into()
            .unwrap();
        cols.gamma = preflight.whir.gammas[whir_round]
            .as_base_slice()
            .try_into()
            .unwrap();
        cols.pre_claim = preflight.whir.pre_query_claims[whir_round]
            .as_base_slice()
            .try_into()
            .unwrap();
        cols.post_claim = preflight.whir.initial_claim_per_round[whir_round + 1]
            .as_base_slice()
            .try_into()
            .unwrap();
    }

    RowMajorMatrix::new(trace, width)
}
