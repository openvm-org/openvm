/// E2E benchmark to aggregate small program with ALU chips.
/// Proofs:
/// 1. Prove a program with some ALU operations.
/// 2. Verify the proof of 1. in the inner config.
/// 2. Verify the proof of 2. in the outer config.
/// 3. Verify the proof of 3. using a Halo2 static verifier.
/// 4. Wrapper Halo2 circuit to reduce the size of 4.
use std::iter;

use afs_compiler::{
    asm::AsmBuilder,
    conversion::CompilerOptions,
    ir::{RVar, Var},
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
use num_bigint_dig::BigUint;
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

const NUM_DIGITS: usize = 8;

fn bench_program() -> Program<BabyBear> {
    type F = BabyBear;
    type EF = BinomialExtensionField<BabyBear, 4>;
    let mut builder = AsmBuilder::<F, EF>::default();

    let sum_digits = iter::repeat(0u32).take(NUM_DIGITS).collect::<Vec<_>>();
    let min_digits = iter::repeat(u32::MAX).take(NUM_DIGITS).collect::<Vec<_>>();
    let val_digits = iter::once(246)
        .chain(iter::repeat(0u32))
        .take(NUM_DIGITS)
        .collect::<Vec<_>>();
    let one_digits = iter::once(1)
        .chain(iter::repeat(0u32))
        .take(NUM_DIGITS)
        .collect::<Vec<_>>();

    let n: Var<_> = builder.eval(F::from_canonical_u32(32));
    let sum = builder.eval_biguint(BigUint::new(sum_digits));
    let min = builder.eval_biguint(BigUint::new(min_digits));
    let val = builder.eval_biguint(BigUint::new(val_digits));
    let one = builder.eval_biguint(BigUint::new(one_digits));

    builder.range(RVar::zero(), n).for_each(|_, builder| {
        let add = builder.add_256(&sum, &val);
        let sub = builder.sub_256(&min, &val);

        let and = builder.and_256(&add, &sub);
        let xor = builder.xor_256(&add, &sub);
        let or = builder.or_256(&and, &xor);

        let sltu = builder.sltu_256(&add, &sub);
        let slt = builder.slt_256(&add, &sub);

        let shift_val = or.clone();
        builder
            .if_eq(sltu, F::from_canonical_u32(1))
            .then(|builder| {
                let srl = builder.srl_256(&shift_val, &one);
                builder.assign(&shift_val, srl);
            });
        builder
            .if_eq(slt, F::from_canonical_u32(0))
            .then(|builder| {
                let sra = builder.sra_256(&shift_val, &one);
                builder.assign(&shift_val, sra);
            });

        let sll = builder.sll_256(&shift_val, &one);
        let eq = builder.eq_256(&sll, &or);
        builder.if_eq(eq, F::from_canonical_u32(0)).then(|builder| {
            let temp = builder.add_256(&add, &one);
            builder.assign(&add, temp);
        });
        builder.if_eq(eq, F::from_canonical_u32(1)).then(|builder| {
            let temp = builder.sub_256(&sub, &one);
            builder.assign(&sub, temp);
        });

        builder.assign(&sum, add);
        builder.assign(&min, sub);
    });

    builder.halt();
    builder.compile_isa_with_options(CompilerOptions {
        word_size: 32,
        ..Default::default()
    })
}

fn bench_program_test_proof_input<SC: StarkGenericConfig>() -> ProofInputForTest<SC>
where
    Domain<SC>: PolynomialSpace<Val = BabyBear>,
{
    let program = bench_program();

    let vm_config = VmConfig {
        ..Default::default()
    }
    .add_executor(ExecutorName::BranchEqual)
    .add_executor(ExecutorName::Jal)
    .add_executor(ExecutorName::LoadStore)
    .add_executor(ExecutorName::FieldArithmetic)
    .add_executor(ExecutorName::ArithmeticLogicUnit256)
    .add_executor(ExecutorName::Shift256);
    gen_vm_program_test_proof_input(program, vec![], vm_config)
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
        #[allow(unused_variables)]
        let vdata = info_span!("Inner Verifier", group = "inner_verifier").in_scope(|| {
            let (program, witness_stream) =
                build_verification_program(vdata, compiler_options.clone());
            let inner_verifier_stf = gen_vm_program_test_proof_input(
                program,
                witness_stream,
                VmConfig::aggregation(4, 7),
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
                VmConfig::aggregation(4, 7),
            );
            afs_recursion::halo2::testing_utils::run_evm_verifier_e2e_test(
                outer_verifier_sft,
                // log_blowup = 3 because of poseidon2 chip.
                Some(standard_fri_params_with_100_bits_conjectured_security(3)),
            );
        });
    });
}
