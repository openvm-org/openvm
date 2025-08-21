use std::{str::FromStr, sync::Arc};

use derive_new::new;
use num_bigint::BigUint;
use num_traits::Zero;
use openvm_algebra_transpiler::Fp2Opcode;
use openvm_circuit::arch::{
    testing::{
        memory::gen_pointer, TestBuilder, TestChipHarness, VmChipTestBuilder, BITWISE_OP_LOOKUP_BUS,
    },
    Arena, PreflightExecutor,
};
use openvm_circuit_primitives::{
    bigint::utils::secp256k1_coord_prime,
    bitwise_op_lookup::{
        BitwiseOperationLookupAir, BitwiseOperationLookupBus, BitwiseOperationLookupChip,
        SharedBitwiseOperationLookupChip,
    },
};
use openvm_instructions::{
    instruction::Instruction,
    riscv::{RV32_CELL_BITS, RV32_MEMORY_AS, RV32_REGISTER_AS, RV32_REGISTER_NUM_LIMBS},
    LocalOpcode, VmOpcode,
};
use openvm_mod_circuit_builder::{
    test_utils::generate_random_biguint, utils::biguint_to_limbs_vec, ExprBuilderConfig,
};
use openvm_pairing_guest::{bls12_381::BLS12_381_MODULUS, bn254::BN254_MODULUS};
use openvm_stark_backend::p3_field::FieldAlgebra;
use openvm_stark_sdk::{p3_baby_bear::BabyBear, utils::create_seeded_rng};
use rand::{rngs::StdRng, Rng};
use test_case::test_case;

use crate::fp2_chip::{
    get_fp2_addsub_air, get_fp2_addsub_chip, get_fp2_addsub_step, get_fp2_muldiv_air,
    get_fp2_muldiv_chip, get_fp2_muldiv_step, Fp2Air, Fp2Chip, Fp2Executor,
};

const LIMB_BITS: usize = 8;
const MAX_INS_CAPACITY: usize = 128;
type F = BabyBear;
type Harness<const BLOCKS: usize, const BLOCK_SIZE: usize> = TestChipHarness<
    F,
    Fp2Executor<BLOCKS, BLOCK_SIZE>,
    Fp2Air<BLOCKS, BLOCK_SIZE>,
    Fp2Chip<F, BLOCKS, BLOCK_SIZE>,
>;

fn create_addsub_test_chips<const BLOCKS: usize, const BLOCK_SIZE: usize>(
    tester: &mut VmChipTestBuilder<F>,
    config: ExprBuilderConfig,
    offset: usize,
) -> (
    Harness<BLOCKS, BLOCK_SIZE>,
    (
        BitwiseOperationLookupAir<RV32_CELL_BITS>,
        SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
    ),
) {
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV32_CELL_BITS>::new(
        bitwise_bus,
    ));

    let air = get_fp2_addsub_air(
        tester.execution_bridge(),
        tester.memory_bridge(),
        config.clone(),
        tester.range_checker().bus(),
        bitwise_bus,
        tester.address_bits(),
        offset,
    );
    let executor = get_fp2_addsub_step(
        config.clone(),
        tester.range_checker().bus(),
        tester.address_bits(),
        offset,
    );
    let chip = get_fp2_addsub_chip(
        config,
        tester.memory_helper(),
        tester.range_checker(),
        bitwise_chip.clone(),
        tester.address_bits(),
    );
    let harness = Harness::with_capacity(executor, air, chip, MAX_INS_CAPACITY);

    (harness, (bitwise_chip.air, bitwise_chip))
}

fn create_muldiv_test_chips<const BLOCKS: usize, const BLOCK_SIZE: usize>(
    tester: &mut VmChipTestBuilder<F>,
    config: ExprBuilderConfig,
    offset: usize,
) -> (
    Harness<BLOCKS, BLOCK_SIZE>,
    (
        BitwiseOperationLookupAir<RV32_CELL_BITS>,
        SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
    ),
) {
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV32_CELL_BITS>::new(
        bitwise_bus,
    ));

    let air = get_fp2_muldiv_air(
        tester.execution_bridge(),
        tester.memory_bridge(),
        config.clone(),
        tester.range_checker().bus(),
        bitwise_bus,
        tester.address_bits(),
        offset,
    );
    let executor = get_fp2_muldiv_step(
        config.clone(),
        tester.range_checker().bus(),
        tester.address_bits(),
        offset,
    );
    let chip = get_fp2_muldiv_chip(
        config,
        tester.memory_helper(),
        tester.range_checker(),
        bitwise_chip.clone(),
        tester.address_bits(),
    );
    let harness = Harness::with_capacity(executor, air, chip, MAX_INS_CAPACITY);

    (harness, (bitwise_chip.air, bitwise_chip))
}

#[allow(clippy::too_many_arguments)]
fn set_and_execute_fp2<const BLOCKS: usize, const BLOCK_SIZE: usize, const NUM_LIMBS: usize, RA>(
    tester: &mut impl TestBuilder<F>,
    executor: &mut Fp2Executor<BLOCKS, BLOCK_SIZE>,
    arena: &mut RA,
    rng: &mut StdRng,
    modulus: &BigUint,
    is_setup: bool,
    is_addsub: bool,
    offset: usize,
) where
    RA: Arena,
    Fp2Executor<BLOCKS, BLOCK_SIZE>: PreflightExecutor<F, RA>,
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

        let op = rng.gen_range(0..2);
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

    let ptr_as = RV32_REGISTER_AS as usize;
    let data_as = RV32_MEMORY_AS as usize;

    let rs1_ptr = gen_pointer(rng, RV32_REGISTER_NUM_LIMBS);
    let rs2_ptr = gen_pointer(rng, RV32_REGISTER_NUM_LIMBS);
    let rd_ptr = gen_pointer(rng, RV32_REGISTER_NUM_LIMBS);

    let a_base_addr = gen_pointer(rng, RV32_REGISTER_NUM_LIMBS) as u32;
    let b_base_addr = gen_pointer(rng, RV32_REGISTER_NUM_LIMBS) as u32;
    let result_base_addr = gen_pointer(rng, RV32_REGISTER_NUM_LIMBS) as u32;

    tester.write::<RV32_REGISTER_NUM_LIMBS>(
        ptr_as,
        rs1_ptr,
        a_base_addr.to_le_bytes().map(F::from_canonical_u8),
    );
    tester.write::<RV32_REGISTER_NUM_LIMBS>(
        ptr_as,
        rs2_ptr,
        b_base_addr.to_le_bytes().map(F::from_canonical_u8),
    );
    tester.write::<RV32_REGISTER_NUM_LIMBS>(
        ptr_as,
        rd_ptr,
        result_base_addr.to_le_bytes().map(F::from_canonical_u8),
    );

    let a_c0_limbs: Vec<F> = biguint_to_limbs_vec(&a_c0, NUM_LIMBS)
        .into_iter()
        .map(F::from_canonical_u8)
        .collect();
    let a_c1_limbs: Vec<F> = biguint_to_limbs_vec(&a_c1, NUM_LIMBS)
        .into_iter()
        .map(F::from_canonical_u8)
        .collect();
    let b_c0_limbs: Vec<F> = biguint_to_limbs_vec(&b_c0, NUM_LIMBS)
        .into_iter()
        .map(F::from_canonical_u8)
        .collect();
    let b_c1_limbs: Vec<F> = biguint_to_limbs_vec(&b_c1, NUM_LIMBS)
        .into_iter()
        .map(F::from_canonical_u8)
        .collect();

    for i in (0..NUM_LIMBS).step_by(BLOCK_SIZE) {
        tester.write::<BLOCK_SIZE>(
            data_as,
            a_base_addr as usize + i,
            a_c0_limbs[i..i + BLOCK_SIZE].try_into().unwrap(),
        );

        tester.write::<BLOCK_SIZE>(
            data_as,
            (a_base_addr + NUM_LIMBS as u32) as usize + i,
            a_c1_limbs[i..i + BLOCK_SIZE].try_into().unwrap(),
        );

        tester.write::<BLOCK_SIZE>(
            data_as,
            b_base_addr as usize + i,
            b_c0_limbs[i..i + BLOCK_SIZE].try_into().unwrap(),
        );

        tester.write::<BLOCK_SIZE>(
            data_as,
            (b_base_addr + NUM_LIMBS as u32) as usize + i,
            b_c1_limbs[i..i + BLOCK_SIZE].try_into().unwrap(),
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
struct TestConfig<const BLOCKS: usize, const BLOCK_SIZE: usize, const NUM_LIMBS: usize> {
    pub modulus: BigUint,
    pub is_addsub: bool,
    pub num_ops: usize,
}

#[test_case(TestConfig::<2, 32, 32>::new(
    BigUint::from_str("357686312646216567629137").unwrap(),
    true,
    50,
))]
#[test_case(TestConfig::<2, 32, 32>::new(
    secp256k1_coord_prime(),
    true,
    50,
))]
#[test_case(TestConfig::<2, 32, 32>::new(
    BN254_MODULUS.clone(),
    true,
    50,
))]
#[test_case(TestConfig::<6, 16, 48>::new(
    BLS12_381_MODULUS.clone(),
    true,
    50,
))]
#[test_case(TestConfig::<2, 32, 32>::new(
    BigUint::from_str("357686312646216567629137").unwrap(),
    false,
    50,
))]
#[test_case(TestConfig::<2, 32, 32>::new(
    secp256k1_coord_prime(),
    false,
    50,
))]
#[test_case(TestConfig::<2, 32, 32>::new(
    BN254_MODULUS.clone(),
    false,
    50,
))]
#[test_case(TestConfig::<6, 16, 48>::new(
    BLS12_381_MODULUS.clone(),
    false,
    50,
))]
fn run_test_with_config<const BLOCKS: usize, const BLOCK_SIZE: usize, const NUM_LIMBS: usize>(
    test_config: TestConfig<BLOCKS, BLOCK_SIZE, NUM_LIMBS>,
) {
    let mut rng = create_seeded_rng();
    let mut tester: VmChipTestBuilder<F> = VmChipTestBuilder::default();
    let config = ExprBuilderConfig {
        modulus: test_config.modulus.clone(),
        num_limbs: NUM_LIMBS,
        limb_bits: LIMB_BITS,
    };

    let offset = Fp2Opcode::CLASS_OFFSET;

    let (mut harness, bitwise) = if test_config.is_addsub {
        create_addsub_test_chips::<BLOCKS, BLOCK_SIZE>(&mut tester, config, offset)
    } else {
        create_muldiv_test_chips::<BLOCKS, BLOCK_SIZE>(&mut tester, config, offset)
    };

    for i in 0..test_config.num_ops {
        set_and_execute_fp2::<BLOCKS, BLOCK_SIZE, NUM_LIMBS, _>(
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
        .load_periphery(bitwise)
        .finalize()
        .simple_test()
        .unwrap();
}

#[cfg(feature = "cuda")]
mod cuda_tests {
    use openvm_circuit::arch::{
        testing::{
            default_bitwise_lookup_bus, default_var_range_checker_bus, GpuChipTestBuilder,
            GpuTestChipHarness,
        },
        DenseRecordArena,
    };
    use openvm_circuit_primitives::var_range::VariableRangeCheckerChip;
    use openvm_cuda_backend::prover_backend::GpuBackend;
    use openvm_stark_backend::Chip;
    use test_case::test_case;

    use super::*;
    use crate::fp2_chip::{Fp2AddSubChipGpu, Fp2MulDivChipGpu};

    pub type GpuHarness<const BLOCKS: usize, const BLOCK_SIZE: usize, T> = GpuTestChipHarness<
        F,
        Fp2Executor<BLOCKS, BLOCK_SIZE>,
        Fp2Air<BLOCKS, BLOCK_SIZE>,
        T,
        Fp2Chip<F, BLOCKS, BLOCK_SIZE>,
    >;

    fn create_addsub_cuda_test_harness<const BLOCKS: usize, const BLOCK_SIZE: usize>(
        tester: &GpuChipTestBuilder,
        config: ExprBuilderConfig,
        offset: usize,
    ) -> GpuHarness<BLOCKS, BLOCK_SIZE, Fp2AddSubChipGpu<BLOCKS, BLOCK_SIZE>> {
        // getting bus from tester since `gpu_chip` and `air` must use the same bus
        let range_bus = default_var_range_checker_bus();
        let bitwise_bus = default_bitwise_lookup_bus();
        // creating a dummy chip for Cpu so we only count `add_count`s from GPU
        let dummy_range_checker_chip = Arc::new(VariableRangeCheckerChip::new(range_bus));
        let dummy_bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV32_CELL_BITS>::new(
            bitwise_bus,
        ));

        let air = get_fp2_addsub_air(
            tester.execution_bridge(),
            tester.memory_bridge(),
            config.clone(),
            range_bus,
            bitwise_bus,
            tester.address_bits(),
            offset,
        );
        let executor =
            get_fp2_addsub_step(config.clone(), range_bus, tester.address_bits(), offset);

        let cpu_chip = get_fp2_addsub_chip(
            config.clone(),
            tester.dummy_memory_helper(),
            dummy_range_checker_chip,
            dummy_bitwise_chip,
            tester.address_bits(),
        );
        let gpu_chip = Fp2AddSubChipGpu::new(
            tester.range_checker(),
            tester.bitwise_op_lookup(),
            config,
            offset,
            tester.address_bits() as u32,
            tester.timestamp_max_bits() as u32,
        );

        GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
    }

    fn create_muldiv_cuda_test_harness<const BLOCKS: usize, const BLOCK_SIZE: usize>(
        tester: &GpuChipTestBuilder,
        config: ExprBuilderConfig,
        offset: usize,
    ) -> GpuHarness<BLOCKS, BLOCK_SIZE, Fp2MulDivChipGpu<BLOCKS, BLOCK_SIZE>> {
        // getting bus from tester since `gpu_chip` and `air` must use the same bus
        let range_bus = default_var_range_checker_bus();
        let bitwise_bus = default_bitwise_lookup_bus();
        // creating a dummy chip for Cpu so we only count `add_count`s from GPU
        let dummy_range_checker_chip = Arc::new(VariableRangeCheckerChip::new(range_bus));
        let dummy_bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV32_CELL_BITS>::new(
            bitwise_bus,
        ));

        let air = get_fp2_muldiv_air(
            tester.execution_bridge(),
            tester.memory_bridge(),
            config.clone(),
            range_bus,
            bitwise_bus,
            tester.address_bits(),
            offset,
        );
        let executor =
            get_fp2_muldiv_step(config.clone(), range_bus, tester.address_bits(), offset);

        let cpu_chip = get_fp2_muldiv_chip(
            config.clone(),
            tester.dummy_memory_helper(),
            dummy_range_checker_chip,
            dummy_bitwise_chip,
            tester.address_bits(),
        );
        let gpu_chip = Fp2MulDivChipGpu::new(
            tester.range_checker(),
            tester.bitwise_op_lookup(),
            config,
            offset,
            tester.address_bits() as u32,
            tester.timestamp_max_bits() as u32,
        );

        GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
    }

    #[test_case(TestConfig::<2, 32, 32>::new(
    BigUint::from_str("357686312646216567629137").unwrap(),
    true,
    50),
    create_addsub_cuda_test_harness::<2, 32>
)]
    #[test_case(TestConfig::<2, 32, 32>::new(
    secp256k1_coord_prime(),
    true,
    50),
    create_addsub_cuda_test_harness::<2, 32>
)]
    #[test_case(TestConfig::<2, 32, 32>::new(
    BN254_MODULUS.clone(),
    true,
    50),
    create_addsub_cuda_test_harness::<2, 32>
)]
    #[test_case(TestConfig::<6, 16, 48>::new(
    BLS12_381_MODULUS.clone(),
    true,
    50),
    create_addsub_cuda_test_harness::<6, 16>
)]
    #[test_case(TestConfig::<2, 32, 32>::new(
    BigUint::from_str("357686312646216567629137").unwrap(),
    false,
    50),
    create_muldiv_cuda_test_harness::<2, 32>
)]
    #[test_case(TestConfig::<2, 32, 32>::new(
    secp256k1_coord_prime(),
    false,
    50),
    create_muldiv_cuda_test_harness::<2, 32>
)]
    #[test_case(TestConfig::<2, 32, 32>::new(
    BN254_MODULUS.clone(),
    false,
    50),
    create_muldiv_cuda_test_harness::<2, 32>
)]
    #[test_case(TestConfig::<6, 16, 48>::new(
    BLS12_381_MODULUS.clone(),
    false,
    50),
    create_muldiv_cuda_test_harness::<6, 16>
)]
    fn run_cuda_test_with_config<
        const BLOCKS: usize,
        const BLOCK_SIZE: usize,
        const NUM_LIMBS: usize,
        C: Chip<DenseRecordArena, GpuBackend>,
    >(
        test_config: TestConfig<BLOCKS, BLOCK_SIZE, NUM_LIMBS>,
        create_cuda_test_harness: impl Fn(
            &GpuChipTestBuilder,
            ExprBuilderConfig,
            usize,
        ) -> GpuHarness<BLOCKS, BLOCK_SIZE, C>,
    ) {
        use crate::AlgebraRecord;

        let mut rng = create_seeded_rng();

        let mut tester =
            GpuChipTestBuilder::default().with_bitwise_op_lookup(default_bitwise_lookup_bus());

        let offset = Fp2Opcode::CLASS_OFFSET;
        let config = ExprBuilderConfig {
            modulus: test_config.modulus.clone(),
            num_limbs: NUM_LIMBS,
            limb_bits: LIMB_BITS,
        };

        let mut harness = create_cuda_test_harness(&tester, config, offset);
        for i in 0..test_config.num_ops {
            set_and_execute_fp2::<BLOCKS, BLOCK_SIZE, NUM_LIMBS, DenseRecordArena>(
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
            .get_record_seeker::<AlgebraRecord<2, BLOCKS, BLOCK_SIZE>, _>()
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
