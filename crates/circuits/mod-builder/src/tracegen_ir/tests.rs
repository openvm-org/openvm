use num_bigint::BigUint;

use super::*;
use crate::{ExprBuilder, ExprBuilderConfig, FieldExpr, FieldExpressionProgram};

#[test]
fn compiler_rejects_non_byte_limbs() {
    let range_bus = openvm_circuit_primitives::var_range::VariableRangeCheckerBus::new(1, 16);
    let builder = ExprBuilder::new(
        ExprBuilderConfig {
            modulus: BigUint::from(17u32),
            num_limbs: 2,
            limb_bits: 4,
        },
        range_bus.range_max_bits,
    );
    let expr = FieldExpr::new(FieldExpressionProgram::new(builder, false), range_bus);
    let error = compile_tracegen_ir(&expr, vec![0], vec![], 1).unwrap_err();
    assert_eq!(
        error.to_string(),
        "CUDA tracegen requires 8-bit limbs, got 4"
    );
}
