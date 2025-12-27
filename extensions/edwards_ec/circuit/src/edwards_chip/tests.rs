use std::{str::FromStr, sync::Arc};

use num_bigint::BigUint;
use num_traits::{FromPrimitive, One};
use openvm_circuit::arch::{
    testing::{
        memory::gen_pointer, TestBuilder, TestChipHarness, VmChipTestBuilder, BITWISE_OP_LOOKUP_BUS,
    },
    Arena, MatrixRecordArena, PreflightExecutor,
};
use openvm_circuit_primitives::{
    bigint::utils::big_uint_to_limbs,
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
use openvm_mod_circuit_builder::{utils::biguint_to_limbs_vec, ExprBuilderConfig};
use openvm_stark_backend::p3_field::FieldAlgebra;
use openvm_stark_sdk::{p3_baby_bear::BabyBear, utils::create_seeded_rng};
use openvm_te_transpiler::Rv32EdwardsOpcode;
use rand::{rngs::StdRng, Rng};
#[cfg(feature = "cuda")]
use {
    crate::{edwards_chip::TeAddChipGpu, EdwardsRecord},
    openvm_circuit::arch::testing::{
        default_bitwise_lookup_bus, default_var_range_checker_bus, GpuChipTestBuilder,
        GpuTestChipHarness,
    },
    openvm_circuit_primitives::var_range::VariableRangeCheckerChip,
};

use crate::edwards_chip::{
    get_te_add_air, get_te_add_chip, get_te_add_step, EdwardsAir, EdwardsChip, TeAddExecutor,
};

const NUM_LIMBS: usize = 32;
const BLOCK_SIZE: usize = 32;
const LIMB_BITS: usize = 8;
const MAX_INS_CAPACITY: usize = 128;
type F = BabyBear;

lazy_static::lazy_static! {
    pub static ref SampleEcPoints: Vec<(BigUint, BigUint)> = {
        // Base point of edwards25519
        let x1 = BigUint::from_str(
            "15112221349535400772501151409588531511454012693041857206046113283949847762202",
        )
        .unwrap();
        let y1 = BigUint::from_str(
            "46316835694926478169428394003475163141307993866256225615783033603165251855960",
        )
        .unwrap();

        // random point on edwards25519
        let x2 = BigUint::from_u32(2).unwrap();
        let y2 = BigUint::from_str(
            "11879831548380997166425477238087913000047176376829905612296558668626594440753",
        )
        .unwrap();

        // This is the sum of (x1, y1) and (x2, y2).
        let x3 = BigUint::from_str(
            "44969869612046584870714054830543834361257841801051546235130567688769346152934",
        )
        .unwrap();
        let y3 = BigUint::from_str(
            "50796027728050908782231253190819121962159170739537197094456293084373503699602",
        )
        .unwrap();

        // This is 2 * (x1, y1)
        let x4 = BigUint::from_str(
            "24727413235106541002554574571675588834622768167397638456726423682521233608206",
        )
        .unwrap();
        let y4 = BigUint::from_str(
            "15549675580280190176352668710449542251549572066445060580507079593062643049417",
        )
        .unwrap();

        vec![(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    };

    pub static ref Edwards25519_Prime: BigUint = BigUint::from_str(
        "57896044618658097711785492504343953926634992332820282019728792003956564819949",
    )
    .unwrap();

    pub static ref Edwards25519_A: BigUint = BigUint::from_str(
        "57896044618658097711785492504343953926634992332820282019728792003956564819948",
    )
    .unwrap();

    pub static ref Edwards25519_D: BigUint = BigUint::from_str(
        "37095705934669439343138083508754565189542113879843219016388785533085940283555",
    )
    .unwrap();

    pub static ref Edwards25519_A_LIMBS: [BabyBear; NUM_LIMBS] =
        big_uint_to_limbs(&Edwards25519_A, LIMB_BITS)
            .into_iter()
            .map(BabyBear::from_canonical_usize)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
    pub static ref Edwards25519_D_LIMBS: [BabyBear; NUM_LIMBS] =
        big_uint_to_limbs(&Edwards25519_D, LIMB_BITS)
            .into_iter()
            .map(BabyBear::from_canonical_usize)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
}

type TeAddHarness<const BLOCKS: usize, const BLOCK_SIZE: usize> = TestChipHarness<
    F,
    TeAddExecutor<BLOCKS, BLOCK_SIZE>,
    EdwardsAir<2, BLOCKS, BLOCK_SIZE>,
    EdwardsChip<F, 2, BLOCKS, BLOCK_SIZE>,
    MatrixRecordArena<F>,
>;

#[cfg(feature = "cuda")]
type GpuHarness<const BLOCKS: usize, const BLOCK_SIZE: usize> = GpuTestChipHarness<
    F,
    TeAddExecutor<BLOCKS, BLOCK_SIZE>,
    EdwardsAir<2, BLOCKS, BLOCK_SIZE>,
    TeAddChipGpu<BLOCKS, BLOCK_SIZE>,
    EdwardsChip<F, 2, BLOCKS, BLOCK_SIZE>,
>;

fn create_harness<const BLOCKS: usize, const BLOCK_SIZE: usize>(
    tester: &VmChipTestBuilder<F>,
    config: ExprBuilderConfig,
    offset: usize,
    a_biguint: BigUint,
    d_biguint: BigUint,
) -> (
    TeAddHarness<BLOCKS, BLOCK_SIZE>,
    (
        BitwiseOperationLookupAir<RV32_CELL_BITS>,
        SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
    ),
) {
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV32_CELL_BITS>::new(
        bitwise_bus,
    ));
    let air = get_te_add_air(
        tester.execution_bridge(),
        tester.memory_bridge(),
        config.clone(),
        tester.range_checker().bus(),
        bitwise_bus,
        tester.address_bits(),
        offset,
        a_biguint.clone(),
        d_biguint.clone(),
    );
    let executor = get_te_add_step(
        config.clone(),
        tester.range_checker().bus(),
        tester.address_bits(),
        offset,
        a_biguint.clone(),
        d_biguint.clone(),
    );
    let chip = get_te_add_chip(
        config.clone(),
        tester.memory_helper(),
        tester.range_checker(),
        bitwise_chip.clone(),
        tester.address_bits(),
        a_biguint,
        d_biguint,
    );
    let harness = TeAddHarness::with_capacity(executor, air, chip, MAX_INS_CAPACITY);

    (harness, (bitwise_chip.air, bitwise_chip))
}

#[cfg(feature = "cuda")]
fn create_cuda_harness<const BLOCKS: usize, const BLOCK_SIZE: usize>(
    tester: &GpuChipTestBuilder,
    config: ExprBuilderConfig,
    offset: usize,
    a_biguint: BigUint,
    d_biguint: BigUint,
) -> GpuHarness<BLOCKS, BLOCK_SIZE> {
    use openvm_circuit::arch::testing::{
        default_bitwise_lookup_bus, default_var_range_checker_bus,
    };

    // getting bus from tester since `gpu_chip` and `air` must use the same bus
    let range_bus = default_var_range_checker_bus();
    let bitwise_bus = default_bitwise_lookup_bus();
    // creating a dummy chip for Cpu so we only count `add_count`s from GPU
    let dummy_range_checker_chip = Arc::new(VariableRangeCheckerChip::new(range_bus));
    let dummy_bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV32_CELL_BITS>::new(
        bitwise_bus,
    ));

    let air = get_te_add_air(
        tester.execution_bridge(),
        tester.memory_bridge(),
        config.clone(),
        range_bus,
        bitwise_bus,
        tester.address_bits(),
        offset,
        a_biguint.clone(),
        d_biguint.clone(),
    );
    let executor = get_te_add_step(
        config.clone(),
        range_bus,
        tester.address_bits(),
        offset,
        a_biguint.clone(),
        d_biguint.clone(),
    );

    let cpu_chip = get_te_add_chip(
        config.clone(),
        tester.dummy_memory_helper(),
        dummy_range_checker_chip,
        dummy_bitwise_chip,
        tester.address_bits(),
        a_biguint,
        d_biguint,
    );
    let gpu_chip = TeAddChipGpu::new(
        tester.range_checker(),
        tester.bitwise_op_lookup(),
        config,
        offset,
        tester.address_bits() as u32,
        tester.timestamp_max_bits() as u32,
    );

    GpuHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
}

#[allow(clippy::too_many_arguments)]
fn set_and_execute_ec_add<
    const BLOCKS: usize,
    const BLOCK_SIZE: usize,
    const NUM_LIMBS: usize,
    RA: Arena,
>(
    tester: &mut impl TestBuilder<F>,
    executor: &mut TeAddExecutor<BLOCKS, BLOCK_SIZE>,
    arena: &mut RA,
    rng: &mut StdRng,
    modulus: &BigUint,
    a_biguint: BigUint,
    d_biguint: BigUint,
    is_setup: bool,
    offset: usize,
    p1: Option<(BigUint, BigUint)>,
    p2: Option<(BigUint, BigUint)>,
) where
    TeAddExecutor<BLOCKS, BLOCK_SIZE>: PreflightExecutor<F, RA>,
{
    let (x1, y1, x2, y2, op_local) = if is_setup {
        (
            modulus.clone(),
            a_biguint,
            d_biguint,
            BigUint::one(),
            Rv32EdwardsOpcode::SETUP_TE_ADD as usize,
        )
    } else if let Some((x1, y1)) = p1 {
        let (x2, y2) = p2.unwrap();
        let x1 = x1 % modulus;
        let y1 = y1 % modulus;
        let x2 = x2 % modulus;
        let y2 = y2 % modulus;
        if rng.gen_bool(0.5) {
            (x1, y1, x2, y2, Rv32EdwardsOpcode::TE_ADD as usize)
        } else {
            (x2, y2, x1, y1, Rv32EdwardsOpcode::TE_ADD as usize)
        }
    } else {
        panic!("Generating random inputs generically is harder because the input points need to be on the curve.");
    };

    let ptr_as = RV32_REGISTER_AS as usize;
    let data_as = RV32_MEMORY_AS as usize;

    let rs1_ptr = gen_pointer(rng, RV32_REGISTER_NUM_LIMBS);
    let rs2_ptr = gen_pointer(rng, RV32_REGISTER_NUM_LIMBS);
    let rd_ptr = gen_pointer(rng, RV32_REGISTER_NUM_LIMBS);

    let p1_base_addr = gen_pointer(rng, BLOCK_SIZE) as u32;
    let p2_base_addr = gen_pointer(rng, BLOCK_SIZE) as u32;
    let result_base_addr = gen_pointer(rng, BLOCK_SIZE) as u32;

    tester.write::<RV32_REGISTER_NUM_LIMBS>(
        ptr_as,
        rs1_ptr,
        p1_base_addr.to_le_bytes().map(F::from_canonical_u8),
    );
    tester.write::<RV32_REGISTER_NUM_LIMBS>(
        ptr_as,
        rs2_ptr,
        p2_base_addr.to_le_bytes().map(F::from_canonical_u8),
    );
    tester.write::<RV32_REGISTER_NUM_LIMBS>(
        ptr_as,
        rd_ptr,
        result_base_addr.to_le_bytes().map(F::from_canonical_u8),
    );

    let x1_limbs: Vec<F> = biguint_to_limbs_vec(&x1, NUM_LIMBS)
        .into_iter()
        .map(F::from_canonical_u8)
        .collect();
    let x2_limbs: Vec<F> = biguint_to_limbs_vec(&x2, NUM_LIMBS)
        .into_iter()
        .map(F::from_canonical_u8)
        .collect();
    let y1_limbs: Vec<F> = biguint_to_limbs_vec(&y1, NUM_LIMBS)
        .into_iter()
        .map(F::from_canonical_u8)
        .collect();
    let y2_limbs: Vec<F> = biguint_to_limbs_vec(&y2, NUM_LIMBS)
        .into_iter()
        .map(F::from_canonical_u8)
        .collect();

    for i in (0..NUM_LIMBS).step_by(BLOCK_SIZE) {
        tester.write::<BLOCK_SIZE>(
            data_as,
            p1_base_addr as usize + i,
            x1_limbs[i..i + BLOCK_SIZE].try_into().unwrap(),
        );

        tester.write::<BLOCK_SIZE>(
            data_as,
            (p1_base_addr + NUM_LIMBS as u32) as usize + i,
            y1_limbs[i..i + BLOCK_SIZE].try_into().unwrap(),
        );

        tester.write::<BLOCK_SIZE>(
            data_as,
            p2_base_addr as usize + i,
            x2_limbs[i..i + BLOCK_SIZE].try_into().unwrap(),
        );

        tester.write::<BLOCK_SIZE>(
            data_as,
            (p2_base_addr + NUM_LIMBS as u32) as usize + i,
            y2_limbs[i..i + BLOCK_SIZE].try_into().unwrap(),
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

fn run_ec_add_test<const BLOCKS: usize, const BLOCK_SIZE: usize, const NUM_LIMBS: usize>(
    offset: usize,
    modulus: BigUint,
    a_biguint: BigUint,
    d_biguint: BigUint,
) {
    let mut rng = create_seeded_rng();
    let mut tester: VmChipTestBuilder<F> = VmChipTestBuilder::default();
    let config = ExprBuilderConfig {
        modulus: modulus.clone(),
        num_limbs: NUM_LIMBS,
        limb_bits: LIMB_BITS,
    };

    let (mut harness, bitwise) = create_harness::<BLOCKS, BLOCK_SIZE>(
        &tester,
        config,
        offset,
        a_biguint.clone(),
        d_biguint.clone(),
    );

    set_and_execute_ec_add::<BLOCKS, BLOCK_SIZE, NUM_LIMBS, _>(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        &modulus,
        a_biguint.clone(),
        d_biguint.clone(),
        true,
        offset,
        None,
        None,
    );

    set_and_execute_ec_add::<BLOCKS, BLOCK_SIZE, NUM_LIMBS, _>(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        &modulus,
        a_biguint.clone(),
        d_biguint.clone(),
        false,
        offset,
        Some(SampleEcPoints[0].clone()),
        Some(SampleEcPoints[1].clone()),
    );

    set_and_execute_ec_add::<BLOCKS, BLOCK_SIZE, NUM_LIMBS, _>(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        &modulus,
        a_biguint,
        d_biguint,
        false,
        offset,
        Some(SampleEcPoints[0].clone()),
        Some(SampleEcPoints[0].clone()),
    );

    let tester = tester
        .build()
        .load(harness)
        .load_periphery(bitwise)
        .finalize();

    tester.simple_test().expect("Verification failed");
}

#[test]
fn test_ec_add_2x32() {
    run_ec_add_test::<2, BLOCK_SIZE, NUM_LIMBS>(
        Rv32EdwardsOpcode::CLASS_OFFSET,
        Edwards25519_Prime.clone(),
        Edwards25519_A.clone(),
        Edwards25519_D.clone(),
    );
}

#[cfg(feature = "cuda")]
fn run_cuda_ec_add<const BLOCKS: usize, const BLOCK_SIZE: usize, const NUM_LIMBS: usize>(
    offset: usize,
    modulus: BigUint,
    a_biguint: BigUint,
    d_biguint: BigUint,
) {
    let mut rng = create_seeded_rng();

    let mut tester =
        GpuChipTestBuilder::default().with_bitwise_op_lookup(default_bitwise_lookup_bus());

    let config = ExprBuilderConfig {
        modulus: modulus.clone(),
        num_limbs: NUM_LIMBS,
        limb_bits: LIMB_BITS,
    };

    let mut harness =
        create_cuda_harness::<BLOCKS, BLOCK_SIZE>(&tester, config, offset, a_biguint, d_biguint);

    set_and_execute_ec_add::<BLOCKS, BLOCK_SIZE, NUM_LIMBS, _>(
        &mut tester,
        &mut harness.executor,
        &mut harness.dense_arena,
        &mut rng,
        &modulus,
        a_biguint,
        d_biguint,
        true,
        offset,
        None,
        None,
    );

    set_and_execute_ec_add::<BLOCKS, BLOCK_SIZE, NUM_LIMBS, _>(
        &mut tester,
        &mut harness.executor,
        &mut harness.dense_arena,
        &mut rng,
        &modulus,
        a_biguint,
        d_biguint,
        false,
        offset,
        Some(SampleEcPoints[0].clone()),
        Some(SampleEcPoints[1].clone()),
    );

    set_and_execute_ec_add::<BLOCKS, BLOCK_SIZE, NUM_LIMBS, _>(
        &mut tester,
        &mut harness.executor,
        &mut harness.dense_arena,
        &mut rng,
        &modulus,
        a_biguint,
        d_biguint,
        false,
        offset,
        Some(SampleEcPoints[0].clone()),
        Some(SampleEcPoints[0].clone()),
    );

    harness
        .dense_arena
        .get_record_seeker::<EccRecord<2, BLOCKS, BLOCK_SIZE>, _>()
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

#[cfg(feature = "cuda")]
#[test]
fn test_ec_add_cuda_2x32() {
    run_cuda_ec_add::<2, 32, 32>(
        Rv32EdwardsOpcode::CLASS_OFFSET,
        Edwards25519_Prime.clone(),
        Edwards25519_A.clone(),
        Edwards25519_D.clone(),
    );
}

///////////////////////////////////////////////////////////////////////////////////////
/// SANITY TESTS
///
/// Ensure that execute functions produce the correct results.
///////////////////////////////////////////////////////////////////////////////////////
#[test]
fn ec_add_sanity_test() {
    let tester: VmChipTestBuilder<F> = VmChipTestBuilder::default();
    let config = ExprBuilderConfig {
        modulus: Edwards25519_Prime.clone(),
        num_limbs: 32,
        limb_bits: LIMB_BITS,
    };

    let executor = get_te_add_step::<2, 32>(
        config.clone(),
        tester.range_checker().bus(),
        tester.address_bits(),
        Rv32EdwardsOpcode::CLASS_OFFSET,
        Edwards25519_A.clone(),
        Edwards25519_D.clone(),
    );

    let (p1_x, p1_y) = SampleEcPoints[0].clone();
    let (p2_x, p2_y) = SampleEcPoints[1].clone();
    assert_eq!(executor.expr.builder.num_variables, 12);
    let r = executor
        .expr
        .execute(vec![p1_x, p1_y, p2_x, p2_y], vec![true]);

    assert_eq!(r.len(), 12);
    let outputs = executor
        .expr
        .output_indices()
        .iter()
        .map(|i| &r[*i])
        .collect::<Vec<_>>();
    assert_eq!(outputs.len(), 2);
    assert_eq!(outputs[0], &SampleEcPoints[2].0);
    assert_eq!(outputs[1], &SampleEcPoints[2].1);

    let (p1_x, p1_y) = SampleEcPoints[0].clone();
    let (p2_x, p2_y) = SampleEcPoints[0].clone();

    assert_eq!(executor.expr.builder.num_variables, 12);
    let r = executor
        .expr
        .execute(vec![p1_x, p1_y, p2_x, p2_y], vec![true]);

    assert_eq!(r.len(), 12);
    let outputs = executor
        .expr
        .output_indices()
        .iter()
        .map(|i| &r[*i])
        .collect::<Vec<_>>();

    assert_eq!(outputs.len(), 2);
    assert_eq!(outputs[0], &SampleEcPoints[3].0);
    assert_eq!(outputs[1], &SampleEcPoints[3].1);
}
