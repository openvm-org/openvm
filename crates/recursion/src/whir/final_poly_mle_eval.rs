use core::borrow::{Borrow, BorrowMut};

use openvm_stark_backend::{
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{FieldAlgebra, FieldExtensionAlgebra, extension::BinomiallyExtendable};
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use stark_backend_v2::{D_EF, F, keygen::types::MultiStarkVerifyingKeyV2, proof::Proof};
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::{
    bus::{TranscriptBus, WhirOpeningPointBus, WhirOpeningPointMessage},
    system::Preflight,
    utils::ext_field_multiply,
    whir::bus::{
        FinalPolyMleEvalBus, FinalPolyMleEvalMessage, WhirEqAlphaUBus, WhirEqAlphaUMessage,
        WhirFinalPolyBus, WhirFinalPolyBusMessage,
    },
};

#[repr(C)]
#[derive(AlignedBorrow)]
struct FinalyPolyMleEvalCols<T> {
    is_valid: T,
    is_first: T,
    proof_idx: T,
    whir_opening_point_msg: WhirOpeningPointMessage<T>,
    has_whir_opening_point_msg: T,
    tidx: T,
    idx: T,
    coeff: [T; D_EF],
    result: [T; D_EF],
    eq_alpha_u: [T; D_EF],
    last_whir_round: T,
}

pub struct FinalPoleMleEvalAir {
    pub whir_opening_point_bus: WhirOpeningPointBus,
    pub final_poly_mle_eval_bus: FinalPolyMleEvalBus,
    pub transcript_bus: TranscriptBus,
    pub eq_alpha_u_bus: WhirEqAlphaUBus,
    pub final_poly_bus: WhirFinalPolyBus,
}

impl BaseAirWithPublicValues<F> for FinalPoleMleEvalAir {}
impl PartitionedBaseAir<F> for FinalPoleMleEvalAir {}

impl<F> BaseAir<F> for FinalPoleMleEvalAir {
    fn width(&self) -> usize {
        FinalyPolyMleEvalCols::<F>::width()
    }
}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for FinalPoleMleEvalAir
where
    <AB::Expr as FieldAlgebra>::F: BinomiallyExtendable<D_EF>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let local = main.row_slice(0);
        let local: &FinalyPolyMleEvalCols<AB::Var> = (*local).borrow();

        self.whir_opening_point_bus.receive(
            builder,
            local.proof_idx,
            local.whir_opening_point_msg.clone(),
            local.has_whir_opening_point_msg,
        );
        self.final_poly_mle_eval_bus.receive(
            builder,
            local.proof_idx,
            FinalPolyMleEvalMessage {
                tidx: local.tidx.into(),
                last_whir_round: local.last_whir_round.into(),
                value: ext_field_multiply(local.result, local.eq_alpha_u),
            },
            local.is_first,
        );
        self.eq_alpha_u_bus.receive(
            builder,
            local.proof_idx,
            WhirEqAlphaUMessage {
                value: local.eq_alpha_u,
            },
            local.is_first,
        );
        self.transcript_bus.observe_ext(
            builder,
            local.proof_idx,
            local.tidx,
            local.coeff,
            local.is_valid,
        );
        self.final_poly_bus.send(
            builder,
            local.proof_idx,
            WhirFinalPolyBusMessage {
                idx: local.idx,
                coeff: local.coeff,
            },
            local.is_valid,
        );
    }
}

pub(crate) fn generate_trace(
    mvk: &MultiStarkVerifyingKeyV2,
    proofs: &[Proof],
    preflights: &[Preflight],
) -> RowMajorMatrix<F> {
    let params = mvk.inner.params;

    let num_rows_per_proof = 1 << params.log_final_poly_len;
    let num_valid_rows = num_rows_per_proof * proofs.len();
    let num_rows = num_valid_rows.next_power_of_two();
    let width = FinalyPolyMleEvalCols::<F>::width();

    let mut trace = vec![F::ZERO; num_rows * width];

    for (row_idx, row) in trace.chunks_mut(width).take(num_valid_rows).enumerate() {
        let proof_idx = row_idx / num_rows_per_proof;
        let i = row_idx % num_rows_per_proof;

        let proof = &proofs[proof_idx];
        let preflight = &preflights[proof_idx];

        let cols: &mut FinalyPolyMleEvalCols<F> = row.borrow_mut();
        cols.is_valid = F::ONE;
        cols.is_first = F::from_bool(i == 0);

        let tidx = preflight.whir.tidx_per_round.last().unwrap() + 3 * params.k_whir * D_EF; // skip sumcheck
        cols.tidx = F::from_canonical_usize(tidx + D_EF * i);
        cols.idx = F::from_canonical_usize(i);
        cols.coeff = proof.whir_proof.final_poly[i]
            .as_base_slice()
            .try_into()
            .unwrap();
        cols.last_whir_round = F::from_canonical_usize(preflight.whir.gammas.len());
        cols.eq_alpha_u = preflight
            .whir
            .eq_partials
            .last()
            .unwrap()
            .as_base_slice()
            .try_into()
            .unwrap();
        cols.result = preflight
            .whir
            .final_poly_at_u
            .as_base_slice()
            .try_into()
            .unwrap();

        if let Some(msg) = preflight
            .whir_opening_point_messages(params.l_skip)
            .get(params.num_whir_sumcheck_rounds() + i)
        {
            cols.whir_opening_point_msg = msg.clone();
            cols.has_whir_opening_point_msg = F::ONE;
        }
    }

    RowMajorMatrix::new(trace, width)
}
