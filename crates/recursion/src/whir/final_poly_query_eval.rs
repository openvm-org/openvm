use core::borrow::{Borrow, BorrowMut};

use openvm_stark_backend::{
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{FieldAlgebra, FieldExtensionAlgebra};
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use stark_backend_v2::{D_EF, F, keygen::types::MultiStarkVerifyingKeyV2, proof::Proof};
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::{
    system::Preflight,
    whir::bus::{
        FinalPolyQueryEvalBus, FinalPolyQueryEvalMessage, WhirAlphaBus, WhirAlphaMessage,
        WhirFinalPolyBus, WhirFinalPolyBusMessage, WhirGammaBus, WhirGammaMessage, WhirQueryBus,
        WhirQueryBusMessage,
    },
};

#[repr(C)]
#[derive(AlignedBorrow)]
struct FinalyPolyQueryEvalCols<T> {
    is_valid: T,
    proof_idx: T,
    is_first: T,
    whir_round: T,
    last_whir_round: T,
    query_idx: T,
    query: [T; D_EF],
    recv_query: T,
    alpha_idx: T,
    alpha: [T; D_EF],
    has_alpha: T,
    gamma_idx: T,
    gamma: [T; D_EF],
    has_gamma: T,
    eq_alpha_u: [T; D_EF],
    final_poly_idx: T,
    final_poly_coeff: [T; D_EF],
    has_final_poly: T,
    value: [T; D_EF],
}

pub struct FinalPolyQueryEvalAir {
    pub query_bus: WhirQueryBus,
    pub alpha_bus: WhirAlphaBus,
    pub gamma_bus: WhirGammaBus,
    pub final_poly_bus: WhirFinalPolyBus,
    pub final_poly_query_eval_bus: FinalPolyQueryEvalBus,
}

impl BaseAirWithPublicValues<F> for FinalPolyQueryEvalAir {}
impl PartitionedBaseAir<F> for FinalPolyQueryEvalAir {}

impl<F> BaseAir<F> for FinalPolyQueryEvalAir {
    fn width(&self) -> usize {
        FinalyPolyQueryEvalCols::<F>::width()
    }
}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for FinalPolyQueryEvalAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let local = main.row_slice(0);
        let local: &FinalyPolyQueryEvalCols<AB::Var> = (*local).borrow();

        self.query_bus.receive(
            builder,
            local.proof_idx,
            WhirQueryBusMessage {
                whir_round: local.whir_round,
                query_idx: local.query_idx,
                value: local.query,
            },
            local.recv_query,
        );
        self.alpha_bus.lookup_key(
            builder,
            local.proof_idx,
            WhirAlphaMessage {
                idx: local.alpha_idx,
                challenge: local.alpha,
            },
            local.has_alpha,
        );
        self.gamma_bus.receive(
            builder,
            local.proof_idx,
            WhirGammaMessage {
                idx: local.gamma_idx,
                challenge: local.gamma,
            },
            local.has_gamma,
        );
        self.final_poly_query_eval_bus.receive(
            builder,
            local.proof_idx,
            FinalPolyQueryEvalMessage {
                last_whir_round: local.last_whir_round,
                value: local.value,
            },
            local.is_first,
        );

        self.final_poly_bus.receive(
            builder,
            local.proof_idx,
            WhirFinalPolyBusMessage {
                idx: local.final_poly_idx,
                coeff: local.final_poly_coeff,
            },
            local.has_final_poly,
        );
    }
}

pub(crate) fn generate_trace(
    mvk: &MultiStarkVerifyingKeyV2,
    proofs: &[Proof],
    preflights: &[Preflight],
) -> RowMajorMatrix<F> {
    let params = mvk.inner.params;

    let k_whir = params.k_whir;
    let num_in_domain_queries = params.num_whir_queries;
    let num_rounds = params.num_whir_rounds();
    let num_alphas = params.num_whir_sumcheck_rounds();

    let num_rows_per_proof = num_rounds * (num_in_domain_queries + 1) * num_alphas;
    let num_valid_rows = num_rows_per_proof * proofs.len();
    let height = num_valid_rows.next_power_of_two();
    let width = FinalyPolyQueryEvalCols::<F>::width();
    let mut trace = vec![F::ZERO; height * width];

    for (row_idx, row) in trace.chunks_mut(width).take(num_valid_rows).enumerate() {
        let proof_idx = row_idx / num_rows_per_proof;
        let i = row_idx % num_rows_per_proof;

        let proof = &proofs[proof_idx];
        let preflight = &preflights[proof_idx];

        let cols: &mut FinalyPolyQueryEvalCols<F> = row.borrow_mut();
        cols.is_valid = F::ONE;
        cols.proof_idx = F::from_canonical_usize(proof_idx);
        let alpha_idx = i % num_alphas + k_whir;
        let global_query = i / num_alphas;
        let query_idx = global_query % (num_in_domain_queries + 1);
        let whir_round = global_query / (num_in_domain_queries + 1);

        if i == 0 {
            cols.is_first = F::ONE;
        }
        cols.whir_round = F::from_canonical_usize(whir_round);
        cols.query_idx = F::from_canonical_usize(query_idx);
        if query_idx == 0 && whir_round != num_rounds - 1 {
            cols.query = preflight.whir.z0s[whir_round]
                .as_base_slice()
                .try_into()
                .unwrap();
        } else if query_idx > 0 {
            cols.query[0] = preflight.whir.zjs[whir_round][query_idx - 1];
        };
        if alpha_idx == k_whir {
            cols.recv_query = F::ONE;
        }
        if i < preflight.whir.alphas.len() {
            cols.alpha_idx = F::from_canonical_usize(i);
            cols.alpha = preflight.whir.alphas[i].as_base_slice().try_into().unwrap();
            cols.has_alpha = F::ONE;
        }
        if i < preflight.whir.gammas.len() {
            cols.gamma_idx = F::from_canonical_usize(i);
            cols.gamma = preflight.whir.gammas[i].as_base_slice().try_into().unwrap();
            cols.has_gamma = F::ONE;
        }
        if i < proof.whir_proof.final_poly.len() {
            cols.final_poly_idx = F::from_canonical_usize(i);
            cols.final_poly_coeff = proof.whir_proof.final_poly[i]
                .as_base_slice()
                .try_into()
                .unwrap();
            cols.has_final_poly = F::ONE;
        }
        cols.value = (*preflight.whir.initial_claim_per_round.last().unwrap()
            - *preflight.whir.eq_partials.last().unwrap() * preflight.whir.final_poly_at_u)
            .as_base_slice()
            .try_into()
            .unwrap();
        cols.last_whir_round = F::from_canonical_usize(preflight.whir.gammas.len());
    }

    RowMajorMatrix::new(trace, width)
}
