use openvm_native_circuit::execute_program;
use openvm_native_compiler::{asm::AsmBuilder, prelude::*};
use openvm_stark_backend::p3_field::{extension::BinomialExtensionField, FieldAlgebra};
use openvm_stark_sdk::p3_baby_bear::BabyBear;

const D: usize = 4;
type F = BabyBear;
type EF = BinomialExtensionField<BabyBear, D>;
#[test]
fn test_range_check_v() {
    let mut builder = AsmBuilder::<F, EF>::default();
    {
        let v: Var<_> = builder.eval(F::ONE);
        builder.range_check_var(v, 1);
    }
    {
        let v: Var<_> = builder.eval(F::from_canonical_u32(1 << 16));
        builder.range_check_var(v, 17);
    }
    {
        let v: Var<_> = builder.eval(F::from_canonical_u32((1 << 29) - 1));
        builder.range_check_var(v, 29);
    }
    builder.halt();

    let program = builder.compile_isa();
    execute_program(program, vec![]);
}

#[test]
#[should_panic]
fn test_range_check_v_neg() {
    let mut builder = AsmBuilder::<F, EF>::default();
    {
        let v: Var<_> = builder.eval(F::from_canonical_u32(1 << 16));
        builder.range_check_var(v, 15);
    }
    builder.halt();

    let program = builder.compile_isa();
    execute_program(program, vec![]);
}
