use openvm_circuit::arch::{
    instructions::program::Program, SystemConfig, VirtualMachine, VmConfig,
};
use openvm_native_circuit::{Native, NativeConfig};
use openvm_native_compiler::{asm::AsmBuilder, ir::Felt};
use openvm_native_recursion::testing_utils::inner::run_recursive_test;
use openvm_stark_backend::{
    config::{Domain, StarkGenericConfig},
    p3_commit::PolynomialSpace,
    p3_field::{extension::BinomialExtensionField, FieldAlgebra},
};
use openvm_stark_sdk::{
    config::{baby_bear_poseidon2::default_engine, FriParameters},
    p3_baby_bear::BabyBear,
    utils::ProofInputForTest,
};

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
    let airs = vm_config.create_chip_complex().unwrap().airs();

    let vm = VirtualMachine::new(default_engine(), vm_config);
    let pk = vm.keygen();
    let vk = pk.get_vk();
    let segments = vm
        .executor
        .execute_metered(
            fib_program.clone(),
            vec![],
            &vk.total_widths(),
            &vk.num_interactions(),
        )
        .unwrap();

    let mut result = vm
        .executor
        .execute_and_generate(fib_program, vec![], &segments)
        .unwrap();
    assert_eq!(result.per_segment.len(), 1, "unexpected continuation");
    let proof_input = result.per_segment.remove(0);
    // Filter out unused AIRS (where trace is empty)
    let (used_airs, per_air) = proof_input
        .per_air
        .into_iter()
        .map(|(air_id, x)| (airs[air_id].clone(), x))
        .unzip();
    ProofInputForTest {
        airs: used_airs,
        per_air,
    }
}

#[test]
fn test_fibonacci_program_verify() {
    let fib_program_stark = fibonacci_program_test_proof_input(0, 1, 32);
    run_recursive_test(fib_program_stark, FriParameters::new_for_testing(3));
}

#[cfg(feature = "static-verifier")]
#[test]
#[ignore = "slow"]
fn test_fibonacci_program_halo2_verify() {
    use openvm_native_recursion::halo2::testing_utils::run_static_verifier_test;

    let fib_program_stark = fibonacci_program_test_proof_input(0, 1, 32);
    run_static_verifier_test(fib_program_stark, FriParameters::new_for_testing(3));
}
