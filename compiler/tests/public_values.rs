use p3_baby_bear::BabyBear;
use p3_field::extension::BinomialExtensionField;
use p3_field::AbstractField;

use afs_compiler::asm::AsmBuilder;
use afs_compiler::prelude::*;
use afs_compiler::util::{execute_program, execute_program_with_public_values};

type F = BabyBear;
type EF = BinomialExtensionField<BabyBear, 4>;

const WORD_SIZE: usize = 1;

#[test]
fn test_compiler_public_values() {
    let mut builder = AsmBuilder::<F, EF>::default();

    let public_value_0 = F::from_canonical_u32(10);
    let public_value_1 = F::from_canonical_u32(20);

    let a: Felt<_> = builder.constant(public_value_0);
    let b: Felt<_> = builder.constant(public_value_1);

    let dyn_len: Var<_> = builder.eval(F::from_canonical_usize(2));
    let mut var_array = builder.dyn_array::<Felt<_>>(dyn_len);
    builder.set(&mut var_array, 0, a);
    builder.set(&mut var_array, 1, b);

    builder.commit_public_values(&var_array);

    builder.halt();

    let program = builder.compile_isa::<WORD_SIZE>();
    execute_program_with_public_values::<WORD_SIZE>(
        program,
        vec![],
        &[(0, public_value_0), (1, public_value_1)],
    );
}

#[test]
fn test_compiler_public_values_no_initial() {
    let mut builder = AsmBuilder::<F, EF>::default();

    let public_value_0 = F::from_canonical_u32(10);
    let public_value_1 = F::from_canonical_u32(20);

    let a: Felt<_> = builder.constant(public_value_0);
    let b: Felt<_> = builder.constant(public_value_1);

    let dyn_len: Var<_> = builder.eval(F::from_canonical_usize(2));
    let mut var_array = builder.dyn_array::<Felt<_>>(dyn_len);
    builder.set(&mut var_array, 0, a);
    builder.set(&mut var_array, 1, b);

    builder.commit_public_values(&var_array);

    builder.halt();

    let program = builder.compile_isa::<WORD_SIZE>();
    execute_program::<WORD_SIZE>(program, vec![]);
}

#[test]
#[should_panic]
fn test_compiler_public_values_negative() {
    let mut builder = AsmBuilder::<F, EF>::default();

    let public_value_0 = F::from_canonical_u32(10);
    let public_value_1 = F::from_canonical_u32(20);
    let not_public_value = F::from_canonical_u32(30);

    let a: Felt<_> = builder.constant(not_public_value);
    let b: Felt<_> = builder.constant(public_value_1);

    let dyn_len: Var<_> = builder.eval(F::from_canonical_usize(2));
    let mut var_array = builder.dyn_array::<Felt<_>>(dyn_len);
    builder.set(&mut var_array, 0, a);
    builder.set(&mut var_array, 1, b);

    builder.commit_public_values(&var_array);

    builder.halt();

    let program = builder.compile_isa::<WORD_SIZE>();
    execute_program_with_public_values::<WORD_SIZE>(
        program,
        vec![],
        &[(0, public_value_0), (1, public_value_1)],
    );
}
