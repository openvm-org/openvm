mod air;
mod trace;

pub use air::{ExpressionClaimAir, ExpressionClaimCols};
pub(in crate::batch_constraint) use trace::{
    ExpressionClaimBlob, ExpressionClaimCtx, ExpressionClaimTraceGenerator,
    generate_expression_claim_blob,
};
