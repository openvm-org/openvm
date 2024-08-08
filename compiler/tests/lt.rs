use afs_compiler::{
    asm::AsmBuilder, conversion::CompilerOptions, prelude::*, util::execute_program_with_config,
};
use p3_baby_bear::BabyBear;
use p3_field::{extension::BinomialExtensionField, AbstractField};
use stark_vm::vm::config::VmConfig;

type F = BabyBear;
type EF = BinomialExtensionField<BabyBear, 4>;

const WORD_SIZE: usize = 1;

#[test]
fn test_compiler_less_than() {
    let mut builder = AsmBuilder::<F, EF>::default();

    let a: Var<_> = builder.constant(F::from_canonical_u32(10));
    let b: Var<_> = builder.constant(F::from_canonical_u32(20));
    let c = builder.lt(a, b);
    builder.assert_var_eq(c, F::one());

    let a: Var<_> = builder.constant(F::from_canonical_u32(20));
    let b: Var<_> = builder.constant(F::from_canonical_u32(10));
    let c = builder.lt(a, b);
    builder.assert_var_eq(c, F::zero());

    builder.halt();

    let program = builder.compile_isa_with_options::<WORD_SIZE>(CompilerOptions {
        compile_prints: false,
        enable_cycle_tracker: true,
        field_arithmetic_enabled: true,
        field_extension_enabled: true,
        field_less_than_enabled: true,
    });
    let config = VmConfig {
        num_public_values: 4,
        is_less_than_enabled: true,
        ..Default::default()
    };
    execute_program_with_config::<WORD_SIZE>(config, program, vec![]);
}
