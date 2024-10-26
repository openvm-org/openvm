/// A E2E benchmark to aggregate a small program with common VM chips.
/// Proofs:
/// 1. Prove a program with some common operations.
/// 2. Verify the proof of 1. in the inner config.
/// 2. Verify the proof of 2. in the outer config.
/// 3. Verify the proof of 3. using a Halo2 static verifier.
/// 4. Wrapper Halo2 circuit to reduce the size of 4.
use afs_compiler::{
    asm::AsmBuilder,
    conversion::CompilerOptions,
    ir::{Ext, Felt, RVar, Var},
};
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
    arch::{instructions::program::Program, ExecutorName},
    sdk::gen_vm_program_test_proof_input,
    system::vm::config::VmConfig,
};
use tracing::info_span;

/// A simple benchmark program to run most operations: keccak256, field arithmetic, field extension,
/// for loop, if-then statement
fn bench_program() -> Program<BabyBear> {
    type F = BabyBear;
    type EF = BinomialExtensionField<BabyBear, 4>;

    let mut builder = AsmBuilder::<F, EF>::default();

    let n: Var<_> = builder.eval(F::from_canonical_u32(2));
    let arr = builder.dyn_array(n);
    let v: Var<_> = builder.eval(F::from_canonical_u32(0));
    builder.range(RVar::zero(), n).for_each(|i, builder| {
        builder.if_eq(i, F::from_canonical_u32(0)).then(|builder| {
            builder.assign(&v, v + F::from_canonical_u32(2));
        });
        builder.assign(&v, v + F::from_canonical_u32(3));
        builder.set_value(&arr, i, v);
    });
    builder.keccak256(&arr);
    let f1: Felt<_> = builder.eval(F::from_canonical_u32(1));
    let f2: Felt<_> = builder.eval(F::from_canonical_u32(2));
    let _: Felt<_> = builder.eval(f1 + f2);
    let ext1: Ext<_, _> = builder.eval(F::from_canonical_u32(1));
    let ext2: Ext<_, _> = builder.eval(F::from_canonical_u32(2));
    let _: Ext<_, _> = builder.eval(ext1 + ext2);

    builder.halt();

    builder.compile_isa()
}

fn bench_program_test_proof_input<SC: StarkGenericConfig>() -> ProofInputForTest<SC>
where
    Domain<SC>: PolynomialSpace<Val = BabyBear>,
{
    let fib_program = bench_program();

    let vm_config = VmConfig::default_with_no_executors()
        .add_executor(ExecutorName::BranchEqual)
        .add_executor(ExecutorName::Jal)
        .add_executor(ExecutorName::LoadStore)
        .add_executor(ExecutorName::Keccak256)
        .add_executor(ExecutorName::FieldArithmetic)
        .add_executor(ExecutorName::FieldExtension);
    gen_vm_program_test_proof_input(fib_program, vec![], vm_config)
}

fn main() {
    run_with_metric_collection("OUTPUT_PATH", || {
        let vdata =
            info_span!("Bench Program Inner", group = "bench_program_inner").in_scope(|| {
                let program_stark = bench_program_test_proof_input();
                program_stark
                    .run_test(&BabyBearPoseidon2Engine::new(
                        standard_fri_params_with_100_bits_conjectured_security(4),
                    ))
                    .unwrap()
            });

        let compiler_options = CompilerOptions {
            enable_cycle_tracker: true,
            ..Default::default()
        };
        let vdata = info_span!("Inner Verifier", group = "inner_verifier").in_scope(|| {
            let (program, witness_stream) =
                build_verification_program(vdata, compiler_options.clone());
            let inner_verifier_stf = gen_vm_program_test_proof_input(
                program,
                witness_stream,
                VmConfig {
                    num_public_values: 4,
                    ..Default::default()
                },
            );
            inner_verifier_stf
                .run_test(&BabyBearPoseidon2Engine::new(
                    // log_blowup = 3 because of poseidon2 chip.
                    standard_fri_params_with_100_bits_conjectured_security(3),
                ))
                .unwrap()
        });

        #[cfg(feature = "static-verifier")]
        info_span!("Recursive Verify e2e", group = "recursive_verify_e2e").in_scope(|| {
            let (program, witness_stream) =
                build_verification_program(vdata, compiler_options.clone());
            let outer_verifier_sft = gen_vm_program_test_proof_input(
                program,
                witness_stream,
                VmConfig {
                    num_public_values: 4,
                    ..Default::default()
                },
            );
            afs_recursion::halo2::testing_utils::run_evm_verifier_e2e_test(
                outer_verifier_sft,
                // log_blowup = 3 because of poseidon2 chip.
                Some(standard_fri_params_with_100_bits_conjectured_security(3)),
            );
        });
    });
}
