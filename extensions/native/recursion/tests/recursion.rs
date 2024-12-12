use ax_stark_backend::{
    config::{Domain, StarkGenericConfig},
    p3_commit::PolynomialSpace,
    p3_field::{extension::BinomialExtensionField, AbstractField},
};
use ax_stark_sdk::{
    config::fri_params::standard_fri_params_with_100_bits_conjectured_security,
    engine::ProofInputForTest, p3_baby_bear::BabyBear,
};
use openvm_circuit::arch::{instructions::program::Program, SystemConfig, VmExecutor};
use openvm_native_circuit::{Native, NativeConfig};
use openvm_native_compiler::{asm::AsmBuilder, ir::Felt};
use openvm_native_recursion::testing_utils::inner::run_recursive_test;

fn fibonacci_program(a: u32, b: u32, n: u32) -> Program<BabyBear> {
    type F = BabyBear;
    type EF = BinomialExtensionField<BabyBear, 4>;

    let mut builder = AsmBuilder::<F, EF>::default();

    let prev: Felt<_> = builder.constant(F::from_canonical_u32(a));
    let next: Felt<_> = builder.constant(F::from_canonical_u32(b));

    builder.commit_public_value(prev);
    builder.commit_public_value(next);

    for _ in 2..n {
        let tmp: Felt<_> = builder.uninit();
        builder.assign(&tmp, next);
        builder.assign(&next, prev + next);
        builder.assign(&prev, tmp);
    }

    builder.commit_public_value(next);

    builder.halt();

    builder.compile_isa()
}

pub(crate) fn fibonacci_program_test_proof_input<SC: StarkGenericConfig>(
    a: u32,
    b: u32,
    n: u32,
) -> ProofInputForTest<SC>
where
    Domain<SC>: PolynomialSpace<Val = BabyBear>,
{
    let fib_program = fibonacci_program(a, b, n);
    let vm_config = NativeConfig::new(SystemConfig::default().with_public_values(3), Native);

    let executor = VmExecutor::<BabyBear, NativeConfig>::new(vm_config);

    let mut result = executor.execute_and_generate(fib_program, vec![]).unwrap();
    assert_eq!(result.per_segment.len(), 1, "unexpected continuation");
    let proof_input = result.per_segment.remove(0);
    ProofInputForTest {
        per_air: proof_input.into_air_proof_input_vec(),
    }
}

#[test]
fn test_fibonacci_program_verify() {
    let fib_program_stark = fibonacci_program_test_proof_input(0, 1, 32);
    run_recursive_test(
        fib_program_stark,
        standard_fri_params_with_100_bits_conjectured_security(3),
    );
}

#[cfg(feature = "static-verifier")]
#[test]
#[ignore = "slow"]
fn test_fibonacci_program_halo2_verify() {
    use openvm_native_recursion::halo2::testing_utils::run_static_verifier_test;

    let fib_program_stark = fibonacci_program_test_proof_input(0, 1, 32);
    run_static_verifier_test(
        fib_program_stark,
        standard_fri_params_with_100_bits_conjectured_security(3),
    );
}
