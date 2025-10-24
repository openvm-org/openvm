use std::{
    array::from_fn,
    borrow::{Borrow, BorrowMut},
};

use openvm_stark_backend::{
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{FieldAlgebra, FieldExtensionAlgebra, PrimeField32};
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use stark_backend_v2::{D_EF, F, keygen::types::MultiStarkVerifyingKeyV2, proof::Proof};
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::{
    bus::{
        ColumnClaimsBus, ColumnClaimsMessage, StackingModuleBus, StackingModuleMessage,
        TranscriptBus, TranscriptBusMessage,
    },
    stacking::bus::{
        ClaimCoefficientsBus, ClaimCoefficientsMessage, EqBitsLookupBus, EqKernelLookupBus,
        StackingModuleTidxBus, StackingModuleTidxMessage, SumcheckClaimsBus, SumcheckClaimsMessage,
    },
    system::Preflight,
};

///////////////////////////////////////////////////////////////////////////
/// COLUMNS
///////////////////////////////////////////////////////////////////////////
#[repr(C)]
#[derive(AlignedBorrow)]
pub struct OpeningClaimsCols<F> {
    pub proof_idx: F,
    pub is_valid: F,
    pub is_last: F,

    // Recevied from batch constraints module
    pub idx: F,
    pub sort_idx: F,
    pub part_idx: F,
    pub col_idx: F,
    pub col_claim: [F; D_EF],
    pub rot_claim: [F; D_EF],

    // Sampled transcript values
    pub tidx: F,
    pub lambda: [F; D_EF],
    pub lambda_pow: [F; D_EF],

    // Location in stacked matrices
    pub commit_idx: F,
    pub stacked_col_idx: F,
    pub row_idx: F,
    pub is_last_for_claim: F,
    pub claim_coefficient: [F; D_EF],

    pub s_0: [F; D_EF],
}

///////////////////////////////////////////////////////////////////////////
/// TRACE GENERATOR
///////////////////////////////////////////////////////////////////////////
pub struct OpeningClaimsTraceGenerator;

impl OpeningClaimsTraceGenerator {
    pub fn generate_trace(
        vk: &MultiStarkVerifyingKeyV2,
        proof: &Proof,
        preflight: &Preflight,
    ) -> RowMajorMatrix<F> {
        let claims = preflight.column_claims_messages(vk, proof);

        let width = OpeningClaimsCols::<usize>::width();
        let num_rows = claims.len().next_power_of_two();

        let mut trace = vec![F::ZERO; num_rows * width];

        let stacked_height = 1usize << vk.inner.params.n_stack;

        let mut current_commit_idx = 0usize;
        let mut current_col_idx = 0usize;
        let mut current_row_idx = 0usize;

        for (i, (claim, chunk)) in claims.iter().zip(trace.chunks_mut(width)).enumerate() {
            let ColumnClaimsMessage {
                idx,
                sort_idx,
                part_idx,
                col_idx,
                col_claim,
                rot_claim,
                lambda_pow,
            } = claim;
            let cols: &mut OpeningClaimsCols<F> = chunk.borrow_mut();
            cols.is_valid = F::ONE;
            cols.is_last = F::from_bool(i + 1 == claims.len());

            cols.idx = *idx;
            cols.sort_idx = *sort_idx;
            cols.part_idx = *part_idx;
            cols.col_idx = *col_idx;
            cols.col_claim = *col_claim;
            cols.rot_claim = *rot_claim;

            cols.tidx = F::from_canonical_usize(preflight.batch_constraint.post_tidx);
            cols.lambda = from_fn(|i| preflight.stacking.lambda.as_base_slice()[i]);
            cols.lambda_pow = *lambda_pow;

            cols.commit_idx = F::from_canonical_usize(current_commit_idx);
            cols.stacked_col_idx = F::from_canonical_usize(current_col_idx);
            cols.row_idx = F::from_canonical_usize(current_row_idx);

            if i + 1 == claims.len()
                || *part_idx != claims[i + 1].part_idx
                || (*sort_idx != claims[i + 1].sort_idx && *part_idx != F::ZERO)
            {
                cols.is_last_for_claim = F::ONE;
                current_commit_idx += 1;
                current_col_idx = 0;
                current_row_idx = 0;
            } else {
                let col_height = 1
                    << (preflight.proof_shape.sorted_trace_vdata
                        [sort_idx.as_canonical_u32() as usize]
                        .1
                        .hypercube_dim
                        + vk.inner.params.l_skip);
                debug_assert!(current_row_idx + col_height <= stacked_height);
                current_row_idx = (current_row_idx + col_height) % stacked_height;
                if current_row_idx == 0 {
                    current_col_idx += 1;
                }
            }
        }
        RowMajorMatrix::new(trace, width)
    }
}

///////////////////////////////////////////////////////////////////////////
/// AIR
///////////////////////////////////////////////////////////////////////////
pub struct OpeningClaimsAir {
    // External buses
    pub stacking_module_bus: StackingModuleBus,
    pub column_claims_bus: ColumnClaimsBus,
    pub transcript_bus: TranscriptBus,

    // Internal buses
    pub stacking_tidx_bus: StackingModuleTidxBus,
    pub claim_coefficients_bus: ClaimCoefficientsBus,
    pub sumcheck_claims_bus: SumcheckClaimsBus,
    pub eq_kernel_lookup_bus: EqKernelLookupBus,
    pub eq_bits_lookup_bus: EqBitsLookupBus,
}

impl BaseAirWithPublicValues<F> for OpeningClaimsAir {}
impl PartitionedBaseAir<F> for OpeningClaimsAir {}

impl<F> BaseAir<F> for OpeningClaimsAir {
    fn width(&self) -> usize {
        OpeningClaimsCols::<F>::width()
    }
}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for OpeningClaimsAir
where
    AB::F: PrimeField32,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (main.row_slice(0), main.row_slice(1));

        let local: &OpeningClaimsCols<AB::Var> = (*local).borrow();
        let _next: &OpeningClaimsCols<AB::Var> = (*next).borrow();

        self.stacking_module_bus.receive(
            builder,
            local.proof_idx,
            StackingModuleMessage {
                tidx: local.tidx.into(),
            },
            local.is_last,
        );

        self.column_claims_bus.receive(
            builder,
            local.proof_idx,
            ColumnClaimsMessage {
                idx: local.idx,
                sort_idx: local.sort_idx,
                part_idx: local.part_idx,
                col_idx: local.col_idx,
                col_claim: local.col_claim,
                rot_claim: local.rot_claim,
                lambda_pow: local.lambda_pow,
            },
            local.is_valid,
        );

        for i in 0..D_EF {
            self.transcript_bus.receive(
                builder,
                local.proof_idx,
                TranscriptBusMessage {
                    tidx: AB::Expr::from_canonical_usize(i) + local.tidx,
                    value: local.lambda[i].into(),
                    is_sample: AB::Expr::ONE,
                },
                local.is_last,
            );
        }

        self.stacking_tidx_bus.send(
            builder,
            local.proof_idx,
            StackingModuleTidxMessage {
                module_idx: AB::Expr::ZERO,
                tidx: local.tidx + AB::Expr::from_canonical_usize(D_EF),
            },
            local.is_last,
        );

        self.claim_coefficients_bus.send(
            builder,
            local.proof_idx,
            ClaimCoefficientsMessage {
                commit_idx: local.commit_idx,
                stacked_col_idx: local.stacked_col_idx,
                coefficient: local.claim_coefficient,
            },
            local.is_last_for_claim,
        );

        self.sumcheck_claims_bus.send(
            builder,
            local.proof_idx,
            SumcheckClaimsMessage {
                module_idx: AB::Expr::ZERO,
                value: local.s_0.map(Into::into),
            },
            local.is_last,
        );
    }
}
