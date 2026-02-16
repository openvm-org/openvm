mod air;
mod trace;

pub use air::{ExpressionClaimAir, ExpressionClaimCols};
pub(in crate::batch_constraint) use trace::{
    generate_expression_claim_blob, ExpressionClaimBlob, ExpressionClaimCtx,
    ExpressionClaimTraceGenerator,
};
