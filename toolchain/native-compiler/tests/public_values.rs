use axvm_circuit::{
    arch::{ExecutorName, SingleSegmentVmExecutor, VmConfig},
    system::program::util::execute_program,
};
use axvm_native_compiler::{asm::AsmBuilder, prelude::*};
use p3_baby_bear::BabyBear;
use p3_field::{extension::BinomialExtensionField, AbstractField};

type F = BabyBear;
type EF = BinomialExtensionField<BabyBear, 4>;

#[test]
fn test_compiler_public_values() {
    let public_value_0 = F::from_canonical_u32(10);
    let public_value_1 = F::from_canonical_u32(20);
    let mut builder = AsmBuilder::<F, EF>::default();

    {
        let a: Felt<_> = builder.constant(public_value_0);
        let b: Felt<_> = builder.constant(public_value_1);

        let dyn_len: Var<_> = builder.eval(F::from_canonical_usize(2));
        let var_array = builder.dyn_array::<Felt<_>>(dyn_len);
        builder.set(&var_array, RVar::zero(), a);
        builder.set(&var_array, RVar::one(), b);

        builder.commit_public_values(&var_array);

        builder.halt();
    }

    let program = builder.compile_isa();
    let executor = SingleSegmentVmExecutor::new(
        VmConfig {
            num_public_values: 2,
            ..Default::default()
        }
        .add_executor(ExecutorName::LoadStore)
        .add_executor(ExecutorName::Jal)
        .add_executor(ExecutorName::FieldArithmetic)
        .add_executor(ExecutorName::BranchEqual),
    );
    let exe_result = executor.execute(program, vec![]).unwrap();
    assert_eq!(
        exe_result
            .public_values
            .into_iter()
            .flatten()
            .collect::<Vec<_>>(),
        vec![public_value_0, public_value_1]
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
    let var_array = builder.dyn_array::<Felt<_>>(dyn_len);
    builder.set(&var_array, RVar::zero(), a);
    builder.set(&var_array, RVar::one(), b);

    builder.commit_public_values(&var_array);

    builder.halt();

    let program = builder.compile_isa();
    execute_program(program, vec![]);
}
