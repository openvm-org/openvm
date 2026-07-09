use std::str::FromStr;
#[cfg(feature = "cuda")]
use std::sync::Arc;

use derive_new::new;
use num_bigint::BigUint;
use num_traits::Zero;
use openvm_algebra_transpiler::Fp2Opcode;
use openvm_circuit::arch::{
    testing::{memory::gen_pointer, TestBuilder, TestChipHarness, VmChipTestBuilder},
    Arena, PreflightExecutor, MEMORY_BLOCK_BYTES,
};
use openvm_circuit_primitives::bigint::utils::secp256k1_coord_prime;
use openvm_instructions::{
    instruction::Instruction,
    riscv::{RV64_MEMORY_AS, RV64_REGISTER_AS, RV64_REGISTER_NUM_LIMBS},
    LocalOpcode, VmOpcode,
};
use openvm_mod_circuit_builder::{
    test_utils::generate_random_biguint, utils::biguint_to_limbs_vec, ExprBuilderConfig,
};
use openvm_pairing_guest::{bls12_381::BLS12_381_MODULUS, bn254::BN254_MODULUS};
use openvm_stark_backend::p3_field::PrimeCharacteristicRing;
use openvm_stark_sdk::{p3_baby_bear::BabyBear, utils::create_seeded_rng};
use rand::{rngs::StdRng, Rng};
use test_case::test_case;

use crate::{
    fp2_chip::{
        get_fp2_addsub_air, get_fp2_addsub_chip, get_fp2_addsub_executor, get_fp2_muldiv_air,
        get_fp2_muldiv_chip, get_fp2_muldiv_executor, Fp2Air, Fp2Chip, Fp2Executor,
    },
    FP2_BLOCKS_32, FP2_BLOCKS_48, NUM_LIMBS_32, NUM_LIMBS_48,
};

const LIMB_BITS: usize = 8;
const MAX_INS_CAPACITY: usize = 128;
type F = BabyBear;
type Harness<const BLOCKS: usize> =
    TestChipHarness<F, Fp2Executor<BLOCKS>, Fp2Air<BLOCKS>, Fp2Chip<F, BLOCKS>>;

fn create_addsub_test_chips<const BLOCKS: usize>(
    tester: &mut VmChipTestBuilder<F>,
    config: ExprBuilderConfig,
    offset: usize,
) -> Harness<BLOCKS> {
    let air = get_fp2_addsub_air(
        tester.execution_bridge(),
        tester.memory_bridge(),
        config.clone(),
        tester.range_checker().bus(),
        tester.address_bits(),
        offset,
    );
    let executor = get_fp2_addsub_executor(
        config.clone(),
        tester.range_checker().bus().range_max_bits,
        tester.address_bits(),
        offset,
    );
    let chip = get_fp2_addsub_chip(
        config,
        tester.memory_helper(),
        tester.range_checker(),
        tester.address_bits(),
    );
    Harness::with_capacity(executor, air, chip, MAX_INS_CAPACITY)
}

fn create_muldiv_test_chips<const BLOCKS: usize>(
    tester: &mut VmChipTestBuilder<F>,
    config: ExprBuilderConfig,
    offset: usize,
) -> Harness<BLOCKS> {
    let air = get_fp2_muldiv_air(
        tester.execution_bridge(),
        tester.memory_bridge(),
        config.clone(),
        tester.range_checker().bus(),
        tester.address_bits(),
        offset,
    );
    let executor = get_fp2_muldiv_executor(
        config.clone(),
        tester.range_checker().bus().range_max_bits,
        tester.address_bits(),
        offset,
    );
    let chip = get_fp2_muldiv_chip(
        config,
        tester.memory_helper(),
        tester.range_checker(),
        tester.address_bits(),
    );
    Harness::with_capacity(executor, air, chip, MAX_INS_CAPACITY)
}

#[allow(clippy::too_many_arguments)]
fn set_and_execute_fp2<const BLOCKS: usize, const NUM_LIMBS: usize, RA>(
    tester: &mut impl TestBuilder<F>,
    executor: &mut Fp2Executor<BLOCKS>,
    arena: &mut RA,
    rng: &mut StdRng,
    modulus: &BigUint,
    is_setup: bool,
    is_addsub: bool,
    offset: usize,
) where
    RA: Arena,
    Fp2Executor<BLOCKS>: PreflightExecutor<F, RA>,
{
    let (a_c0, a_c1, b_c0, b_c1, op_local) = if is_setup {
        (
            modulus.clone(),
            BigUint::zero(),
            BigUint::zero(),
            BigUint::zero(),
            if is_addsub {
                Fp2Opcode::SETUP_ADDSUB as usize
            } else {
                Fp2Opcode::SETUP_MULDIV as usize
            },
        )
    } else {
        let a_c0 = generate_random_biguint(modulus);
        let a_c1 = generate_random_biguint(modulus);

        let b_c0 = generate_random_biguint(modulus);
        let b_c1 = generate_random_biguint(modulus);

        let op = rng.random_range(0..2);
        let op = if is_addsub {
            match op {
                0 => Fp2Opcode::ADD as usize,
                1 => Fp2Opcode::SUB as usize,
                _ => panic!(),
            }
        } else {
            match op {
                0 => Fp2Opcode::MUL as usize,
                1 => Fp2Opcode::DIV as usize,
                _ => panic!(),
            }
        };
        (a_c0, a_c1, b_c0, b_c1, op)
    };

    let ptr_as = RV64_REGISTER_AS as usize;
    let data_as = RV64_MEMORY_AS as usize;

    let rs1_ptr = gen_pointer(rng, RV64_REGISTER_NUM_LIMBS);
    let rs2_ptr = gen_pointer(rng, RV64_REGISTER_NUM_LIMBS);
    let rd_ptr = gen_pointer(rng, RV64_REGISTER_NUM_LIMBS);

    let a_base_addr = gen_pointer(rng, RV64_REGISTER_NUM_LIMBS) as u32;
    let b_base_addr = gen_pointer(rng, RV64_REGISTER_NUM_LIMBS) as u32;
    let result_base_addr = gen_pointer(rng, RV64_REGISTER_NUM_LIMBS) as u32;

    tester.write_bytes::<RV64_REGISTER_NUM_LIMBS>(
        ptr_as,
        rs1_ptr,
        (a_base_addr as u64).to_le_bytes().map(F::from_u8),
    );
    tester.write_bytes::<RV64_REGISTER_NUM_LIMBS>(
        ptr_as,
        rs2_ptr,
        (b_base_addr as u64).to_le_bytes().map(F::from_u8),
    );
    tester.write_bytes::<RV64_REGISTER_NUM_LIMBS>(
        ptr_as,
        rd_ptr,
        (result_base_addr as u64).to_le_bytes().map(F::from_u8),
    );

    let a_c0_limbs: Vec<F> = biguint_to_limbs_vec(&a_c0, NUM_LIMBS)
        .into_iter()
        .map(F::from_u8)
        .collect();
    let a_c1_limbs: Vec<F> = biguint_to_limbs_vec(&a_c1, NUM_LIMBS)
        .into_iter()
        .map(F::from_u8)
        .collect();
    let b_c0_limbs: Vec<F> = biguint_to_limbs_vec(&b_c0, NUM_LIMBS)
        .into_iter()
        .map(F::from_u8)
        .collect();
    let b_c1_limbs: Vec<F> = biguint_to_limbs_vec(&b_c1, NUM_LIMBS)
        .into_iter()
        .map(F::from_u8)
        .collect();

    for i in (0..NUM_LIMBS).step_by(MEMORY_BLOCK_BYTES) {
        tester.write_bytes::<{ MEMORY_BLOCK_BYTES }>(
            data_as,
            a_base_addr as usize + i,
            a_c0_limbs[i..i + MEMORY_BLOCK_BYTES].try_into().unwrap(),
        );

        tester.write_bytes::<{ MEMORY_BLOCK_BYTES }>(
            data_as,
            (a_base_addr + NUM_LIMBS as u32) as usize + i,
            a_c1_limbs[i..i + MEMORY_BLOCK_BYTES].try_into().unwrap(),
        );

        tester.write_bytes::<{ MEMORY_BLOCK_BYTES }>(
            data_as,
            b_base_addr as usize + i,
            b_c0_limbs[i..i + MEMORY_BLOCK_BYTES].try_into().unwrap(),
        );

        tester.write_bytes::<{ MEMORY_BLOCK_BYTES }>(
            data_as,
            (b_base_addr + NUM_LIMBS as u32) as usize + i,
            b_c1_limbs[i..i + MEMORY_BLOCK_BYTES].try_into().unwrap(),
        );
    }

    let instruction = Instruction::from_isize(
        VmOpcode::from_usize(offset + op_local),
        rd_ptr as isize,
        rs1_ptr as isize,
        rs2_ptr as isize,
        ptr_as as isize,
        data_as as isize,
    );
    tester.execute(executor, arena, &instruction);
}

#[derive(new)]
struct TestConfig<const BLOCKS: usize, const NUM_LIMBS: usize> {
    pub modulus: BigUint,
    pub is_addsub: bool,
    pub num_ops: usize,
}

#[test_case(TestConfig::<{FP2_BLOCKS_32}, {NUM_LIMBS_32}>::new(
    BigUint::from_str("357686312646216567629137").unwrap(),
    true,
    50,
))]
#[test_case(TestConfig::<{FP2_BLOCKS_32}, {NUM_LIMBS_32}>::new(
    secp256k1_coord_prime(),
    true,
    50,
))]
#[test_case(TestConfig::<{FP2_BLOCKS_32}, {NUM_LIMBS_32}>::new(
    BN254_MODULUS.clone(),
    true,
    50,
))]
#[test_case(TestConfig::<{FP2_BLOCKS_48}, {NUM_LIMBS_48}>::new(
    BLS12_381_MODULUS.clone(),
    true,
    50,
))]
#[test_case(TestConfig::<{FP2_BLOCKS_32}, {NUM_LIMBS_32}>::new(
    BigUint::from_str("357686312646216567629137").unwrap(),
    false,
    50,
))]
#[test_case(TestConfig::<{FP2_BLOCKS_32}, {NUM_LIMBS_32}>::new(
    secp256k1_coord_prime(),
    false,
    50,
))]
#[test_case(TestConfig::<{FP2_BLOCKS_32}, {NUM_LIMBS_32}>::new(
    BN254_MODULUS.clone(),
    false,
    50,
))]
#[test_case(TestConfig::<{FP2_BLOCKS_48}, {NUM_LIMBS_48}>::new(
    BLS12_381_MODULUS.clone(),
    false,
    50,
))]
fn run_test_with_config<const BLOCKS: usize, const NUM_LIMBS: usize>(
    test_config: TestConfig<BLOCKS, NUM_LIMBS>,
) {
    let mut rng = create_seeded_rng();
    let mut tester: VmChipTestBuilder<F> = VmChipTestBuilder::default();
    let config = ExprBuilderConfig {
        modulus: test_config.modulus.clone(),
        num_limbs: NUM_LIMBS,
        limb_bits: LIMB_BITS,
    };

    let offset = Fp2Opcode::CLASS_OFFSET;

    let mut harness = if test_config.is_addsub {
        create_addsub_test_chips::<BLOCKS>(&mut tester, config, offset)
    } else {
        create_muldiv_test_chips::<BLOCKS>(&mut tester, config, offset)
    };

    for i in 0..test_config.num_ops {
        set_and_execute_fp2::<BLOCKS, NUM_LIMBS, _>(
            &mut tester,
            &mut harness.executor,
            &mut harness.arena,
            &mut rng,
            &test_config.modulus,
            i == 0,
            test_config.is_addsub,
            offset,
        );
    }

    tester
        .build()
        .load(harness)
        .finalize()
        .simple_test()
        .unwrap();
}

#[cfg(feature = "cuda")]
mod cuda_tests {
    use openvm_circuit::arch::{
        testing::{default_var_range_checker_bus, GpuChipTestBuilder, GpuTestChipHarness},
        DenseRecordArena,
    };
    use openvm_circuit_primitives::{var_range::VariableRangeCheckerChip, Chip};
    use openvm_cuda_backend::GpuBackend;
    use test_case::test_case;

    use super::*;
    use crate::extension::HybridFp2Chip;

    pub type GpuHarness<const BLOCKS: usize, T> =
        GpuTestChipHarness<F, Fp2Executor<BLOCKS>, Fp2Air<BLOCKS>, T, Fp2Chip<F, BLOCKS>>;

    fn create_addsub_cuda_test_harness<const BLOCKS: usize>(
        tester: &GpuChipTestBuilder,
        config: ExprBuilderConfig,
        offset: usize,
    ) -> GpuHarness<BLOCKS, HybridFp2Chip<F, BLOCKS>> {
        // getting bus from tester since `gpu_chip` and `air` must use the same bus
        let range_bus = default_var_range_checker_bus();
        // creating a dummy chip for Cpu so we only count `add_count`s from GPU
        let dummy_range_checker_chip = Arc::new(VariableRangeCheckerChip::new(range_bus));

        let air = get_fp2_addsub_air(
            tester.execution_bridge(),
            tester.memory_bridge(),
            config.clone(),
            range_bus,
            tester.address_bits(),
            offset,
        );
        let executor = get_fp2_addsub_executor(
            config.clone(),
            range_bus.range_max_bits,
            tester.address_bits(),
            offset,
        );

        let cpu_chip = get_fp2_addsub_chip(
            config.clone(),
            tester.dummy_memory_helper(),
            dummy_range_checker_chip,
            tester.address_bits(),
        );
        let hybrid_chip = HybridFp2Chip::new(
            get_fp2_addsub_chip(
                config,
                tester.cpu_memory_helper(),
                tester.cpu_range_checker(),
                tester.address_bits(),
            ),
            tester.address_bits(),
            tester.timestamp_max_bits(),
            tester.range_checker(),
        );

        GpuTestChipHarness::with_capacity(executor, air, hybrid_chip, cpu_chip, MAX_INS_CAPACITY)
    }

    fn create_muldiv_cuda_test_harness<const BLOCKS: usize>(
        tester: &GpuChipTestBuilder,
        config: ExprBuilderConfig,
        offset: usize,
    ) -> GpuHarness<BLOCKS, HybridFp2Chip<F, BLOCKS>> {
        // getting bus from tester since `gpu_chip` and `air` must use the same bus
        let range_bus = default_var_range_checker_bus();
        // creating a dummy chip for Cpu so we only count `add_count`s from GPU
        let dummy_range_checker_chip = Arc::new(VariableRangeCheckerChip::new(range_bus));

        let air = get_fp2_muldiv_air(
            tester.execution_bridge(),
            tester.memory_bridge(),
            config.clone(),
            range_bus,
            tester.address_bits(),
            offset,
        );
        let executor = get_fp2_muldiv_executor(
            config.clone(),
            range_bus.range_max_bits,
            tester.address_bits(),
            offset,
        );

        let cpu_chip = get_fp2_muldiv_chip(
            config.clone(),
            tester.dummy_memory_helper(),
            dummy_range_checker_chip,
            tester.address_bits(),
        );
        let hybrid_chip = HybridFp2Chip::new(
            get_fp2_muldiv_chip(
                config,
                tester.cpu_memory_helper(),
                tester.cpu_range_checker(),
                tester.address_bits(),
            ),
            tester.address_bits(),
            tester.timestamp_max_bits(),
            tester.range_checker(),
        );

        GpuTestChipHarness::with_capacity(executor, air, hybrid_chip, cpu_chip, MAX_INS_CAPACITY)
    }

    #[test_case(TestConfig::<FP2_BLOCKS_32, NUM_LIMBS_32>::new(
    BigUint::from_str("357686312646216567629137").unwrap(),
    true,
    50),
    create_addsub_cuda_test_harness::<FP2_BLOCKS_32>
)]
    #[test_case(TestConfig::<FP2_BLOCKS_32, NUM_LIMBS_32>::new(
    secp256k1_coord_prime(),
    true,
    50),
    create_addsub_cuda_test_harness::<FP2_BLOCKS_32>
)]
    #[test_case(TestConfig::<FP2_BLOCKS_32, NUM_LIMBS_32>::new(
    BN254_MODULUS.clone(),
    true,
    50),
    create_addsub_cuda_test_harness::<FP2_BLOCKS_32>
)]
    #[test_case(TestConfig::<FP2_BLOCKS_48, NUM_LIMBS_48>::new(
    BLS12_381_MODULUS.clone(),
    true,
    50),
    create_addsub_cuda_test_harness::<FP2_BLOCKS_48>
)]
    #[test_case(TestConfig::<FP2_BLOCKS_32, NUM_LIMBS_32>::new(
    BigUint::from_str("357686312646216567629137").unwrap(),
    false,
    50),
    create_muldiv_cuda_test_harness::<FP2_BLOCKS_32>
)]
    #[test_case(TestConfig::<FP2_BLOCKS_32, NUM_LIMBS_32>::new(
    secp256k1_coord_prime(),
    false,
    50),
    create_muldiv_cuda_test_harness::<FP2_BLOCKS_32>
)]
    #[test_case(TestConfig::<FP2_BLOCKS_32, NUM_LIMBS_32>::new(
    BN254_MODULUS.clone(),
    false,
    50),
    create_muldiv_cuda_test_harness::<FP2_BLOCKS_32>
)]
    #[test_case(TestConfig::<FP2_BLOCKS_48, NUM_LIMBS_48>::new(
    BLS12_381_MODULUS.clone(),
    false,
    50),
    create_muldiv_cuda_test_harness::<FP2_BLOCKS_48>
)]
    fn run_cuda_test_with_config<
        const BLOCKS: usize,
        const NUM_LIMBS: usize,
        C: Chip<DenseRecordArena, GpuBackend>,
    >(
        test_config: TestConfig<BLOCKS, NUM_LIMBS>,
        create_cuda_test_harness: impl Fn(
            &GpuChipTestBuilder,
            ExprBuilderConfig,
            usize,
        ) -> GpuHarness<BLOCKS, C>,
    ) {
        use crate::AlgebraRecord;

        let mut rng = create_seeded_rng();

        let mut tester = GpuChipTestBuilder::default();

        let offset = Fp2Opcode::CLASS_OFFSET;
        let config = ExprBuilderConfig {
            modulus: test_config.modulus.clone(),
            num_limbs: NUM_LIMBS,
            limb_bits: LIMB_BITS,
        };

        let mut harness = create_cuda_test_harness(&tester, config, offset);
        for i in 0..test_config.num_ops {
            set_and_execute_fp2::<BLOCKS, NUM_LIMBS, DenseRecordArena>(
                &mut tester,
                &mut harness.executor,
                &mut harness.dense_arena,
                &mut rng,
                &test_config.modulus,
                i == 0,
                test_config.is_addsub,
                offset,
            );
        }

        harness
            .dense_arena
            .get_record_seeker::<AlgebraRecord<2, BLOCKS>, _>()
            .transfer_to_matrix_arena(
                &mut harness.matrix_arena,
                harness.executor.get_record_layout::<F>(),
            );

        tester
            .build()
            .load_gpu_harness(harness)
            .finalize()
            .simple_test()
            .unwrap();
    }
}
