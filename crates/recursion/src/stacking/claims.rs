use std::{
    array::from_fn,
    borrow::{Borrow, BorrowMut},
};

use itertools::Itertools;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{FieldAlgebra, FieldExtensionAlgebra, PrimeField32};
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use stark_backend_v2::{
    D_EF, F, keygen::types::MultiStarkVerifyingKeyV2, poseidon2::sponge::FiatShamirTranscript,
    proof::Proof,
};
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::{
    bus::{
        ClaimCoefficientsBus, ClaimCoefficientsMessage, StackingIndexMessage, StackingIndicesBus,
        StackingModuleTidxBus, StackingModuleTidxMessage, SumcheckClaimsBus, SumcheckClaimsMessage,
        TranscriptBus, TranscriptBusMessage, WhirModuleBus, WhirModuleMessage,
    },
    system::{BusInventory, Preflight},
};

///////////////////////////////////////////////////////////////////////////
/// COLUMNS
///////////////////////////////////////////////////////////////////////////
#[repr(C)]
#[derive(AlignedBorrow)]
pub struct StackingClaimsCols<F> {
    pub proof_idx: F,
    pub is_valid: F,
    pub is_last: F,
    pub is_first: F,

    pub commit_idx: F,
    pub stacked_col_idx: F,

    pub tidx: F,
    pub mu: [F; D_EF],

    pub stacking_claim: [F; D_EF],
    pub claim_coefficient: [F; D_EF],

    pub s_n_u_n: [F; D_EF],
}

///////////////////////////////////////////////////////////////////////////
/// TRACE GENERATOR
///////////////////////////////////////////////////////////////////////////
pub struct StackingClaimsTraceGenerator;

impl StackingClaimsTraceGenerator {
    pub fn generate_trace<TS: FiatShamirTranscript>(
        _vk: &MultiStarkVerifyingKeyV2,
        proof: &Proof,
        preflight: &Preflight<TS>,
    ) -> RowMajorMatrix<F> {
        let claims = proof
            .stacking_proof
            .stacking_openings
            .iter()
            .enumerate()
            .flat_map(|(commit_idx, openings)| {
                openings
                    .iter()
                    .enumerate()
                    .map(move |(stacked_col_idx, opening)| (commit_idx, stacked_col_idx, opening))
            })
            .collect_vec();

        let width = StackingClaimsCols::<usize>::width();
        let num_rows = claims.len().next_power_of_two();

        let mut trace = vec![F::ZERO; num_rows * width];

        let initial_tidx = preflight.stacking.intermediate_tidx[2];

        for (i, (&(commit_idx, stacked_col_idx, claim), chunk)) in
            claims.iter().zip(trace.chunks_mut(width)).enumerate()
        {
            let cols: &mut StackingClaimsCols<F> = chunk.borrow_mut();
            cols.is_valid = F::ONE;

            cols.commit_idx = F::from_canonical_usize(commit_idx);
            cols.stacked_col_idx = F::from_canonical_usize(stacked_col_idx);
            cols.stacking_claim = from_fn(|i| claim.as_base_slice()[i]);

            let mu = preflight.stacking.stacking_batching_challenge;
            cols.mu = from_fn(|i| mu.as_base_slice()[i]);

            cols.tidx = F::from_canonical_usize(initial_tidx + (D_EF * i));
            cols.is_last = F::from_bool(i + 1 == claims.len());
            cols.is_first = F::from_bool(i == 0);
        }
        RowMajorMatrix::new(trace, width)
    }
}

///////////////////////////////////////////////////////////////////////////
/// AIR
///////////////////////////////////////////////////////////////////////////
pub struct StackingClaimsAir {
    // External buses
    pub stacking_indices_bus: StackingIndicesBus,
    pub whir_module_bus: WhirModuleBus,
    pub transcript_bus: TranscriptBus,

    // Internal buses
    pub stacking_tidx_bus: StackingModuleTidxBus,
    pub claim_coefficients_bus: ClaimCoefficientsBus,
    pub sumcheck_claims_bus: SumcheckClaimsBus,
}

impl StackingClaimsAir {
    pub fn new(buses: &BusInventory) -> Self {
        Self {
            stacking_indices_bus: buses.stacking_widths_bus,
            whir_module_bus: buses.whir_module_bus,
            transcript_bus: buses.transcript_bus,
            stacking_tidx_bus: buses.stacking_tidx_bus,
            claim_coefficients_bus: buses.claim_coefficients_bus,
            sumcheck_claims_bus: buses.sumcheck_claims_bus,
        }
    }
}

impl BaseAirWithPublicValues<F> for StackingClaimsAir {}
impl PartitionedBaseAir<F> for StackingClaimsAir {}

impl<F> BaseAir<F> for StackingClaimsAir {
    fn width(&self) -> usize {
        StackingClaimsCols::<F>::width()
    }
}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for StackingClaimsAir
where
    AB::F: PrimeField32,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (main.row_slice(0), main.row_slice(1));

        let local: &StackingClaimsCols<AB::Var> = (*local).borrow();
        let _next: &StackingClaimsCols<AB::Var> = (*next).borrow();

        self.stacking_indices_bus.receive(
            builder,
            local.proof_idx,
            StackingIndexMessage {
                commit_idx: local.commit_idx,
                col_idx: local.stacked_col_idx,
            },
            local.is_valid,
        );

        for i in 0..D_EF {
            self.transcript_bus.receive(
                builder,
                local.proof_idx,
                TranscriptBusMessage {
                    tidx: AB::Expr::from_canonical_usize(i) + local.tidx,
                    value: local.stacking_claim[i].into(),
                    is_sample: AB::Expr::ZERO,
                },
                local.is_valid,
            );

            self.transcript_bus.receive(
                builder,
                local.proof_idx,
                TranscriptBusMessage {
                    tidx: AB::Expr::from_canonical_usize(i + D_EF) + local.tidx,
                    value: local.mu[i].into(),
                    is_sample: AB::Expr::ONE,
                },
                local.is_last,
            );
        }

        // TODO[stephenh]: claim should be the RLC
        self.whir_module_bus.send(
            builder,
            local.proof_idx,
            WhirModuleMessage {
                tidx: AB::Expr::from_canonical_usize(2 * D_EF) + local.tidx,
                mu: local.mu.map(Into::into),
                claim: local.stacking_claim.map(Into::into),
            },
            local.is_last,
        );

        self.stacking_tidx_bus.receive(
            builder,
            local.proof_idx,
            StackingModuleTidxMessage {
                module_idx: AB::Expr::TWO,
                tidx: local.tidx.into(),
            },
            local.is_first,
        );

        self.claim_coefficients_bus.receive(
            builder,
            local.proof_idx,
            ClaimCoefficientsMessage {
                commit_idx: local.commit_idx,
                stacked_col_idx: local.stacked_col_idx,
                coefficient: local.claim_coefficient,
            },
            local.is_valid,
        );

        self.sumcheck_claims_bus.receive(
            builder,
            local.proof_idx,
            SumcheckClaimsMessage {
                module_idx: AB::Expr::TWO,
                value: local.s_n_u_n.map(Into::into),
            },
            local.is_last,
        );
    }
}
