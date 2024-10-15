/// A E2E benchmark to aggregate a program with minimal number of VM chips.
/// Proofs:
/// 1. Prove a program to compute fibonacci numbers.
/// 2. Verify the proof of 1. in the outer config.
/// 3. Verify the proof of 2. using a Halo2 static verifier. (Outer Recursion)
/// 4. Wrapper Halo2 circuit to reduce the size of 3.
use afs_compiler::{asm::AsmBuilder, conversion::CompilerOptions, ir::Felt};
use afs_recursion::testing_utils::inner::build_verification_program;
use ax_sdk::{
    bench::run_with_metric_collection,
    config::{
        baby_bear_poseidon2::BabyBearPoseidon2Engine,
        fri_params::standard_fri_params_with_100_bits_conjectured_security,
    },
    engine::{ProofInputForTest, StarkFriEngine},
};
use p3_baby_bear::BabyBear;
use p3_commit::PolynomialSpace;
use p3_field::{extension::BinomialExtensionField, AbstractField};
use p3_uni_stark::{Domain, StarkGenericConfig};
use stark_vm::{
    arch::ExecutorName,
    sdk::gen_vm_program_test_proof_input,
    system::{program::Program, vm::config::VmConfig},
};
use tracing::info_span;

fn fibonacci_program(a: u32, b: u32, n: u32) -> Program<BabyBear> {
    type F = BabyBear;
    type EF = BinomialExtensionField<BabyBear, 4>;

    let mut builder = AsmBuilder::<F, EF>::default();

    let prev: Felt<_> = builder.constant(F::from_canonical_u32(a));
    let next: Felt<_> = builder.constant(F::from_canonical_u32(b));

    for _ in 2..n {
        let tmp: Felt<_> = builder.uninit();
        builder.assign(&tmp, next);
        builder.assign(&next, prev + next);
        builder.assign(&prev, tmp);
    }

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

    let vm_config = VmConfig::core().add_default_executor(ExecutorName::FieldArithmetic);
    gen_vm_program_test_proof_input(fib_program, vec![], vm_config)
}

fn main() {
    run_with_metric_collection("OUTPUT_PATH", || {
        let span =
            info_span!("Fibonacci Program Inner", group = "fibonacci_program_inner").entered();
        let fib_program_stark = fibonacci_program_test_proof_input(0, 1, 32);
        let engine =
            BabyBearPoseidon2Engine::new(standard_fri_params_with_100_bits_conjectured_security(3));
        let vdata = fib_program_stark.run_test(&engine).unwrap();
        span.exit();

        let compiler_options = CompilerOptions {
            enable_cycle_tracker: true,
            ..Default::default()
        };
        #[cfg(feature = "static-verifier")]
        info_span!("Recursive Verify e2e", group = "recursive_verify_e2e").in_scope(|| {
            let (program, witness_stream) = build_verification_program(vdata, compiler_options);
            let inner_verifier_sft = gen_vm_program_test_proof_input(
                program,
                witness_stream,
                VmConfig {
                    num_public_values: 4,
                    ..Default::default()
                },
            );
            afs_recursion::halo2::testing_utils::run_evm_verifier_e2e_test(
                inner_verifier_sft,
                // log_blowup = 3 because of poseidon2 chip.
                Some(standard_fri_params_with_100_bits_conjectured_security(3)),
            );
        });
    });
}
