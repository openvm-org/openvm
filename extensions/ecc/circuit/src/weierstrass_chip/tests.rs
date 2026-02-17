use std::{str::FromStr, sync::Arc};

use halo2curves_axiom::secp256r1;
use num_bigint::BigUint;
use num_traits::{FromPrimitive, Num, Zero};
use openvm_circuit::arch::{
    testing::{
        memory::gen_pointer, TestBuilder, TestChipHarness, VmChipTestBuilder, BITWISE_OP_LOOKUP_BUS,
    },
    Arena, MatrixRecordArena, PreflightExecutor,
};
use openvm_circuit_primitives::{
    bigint::utils::{secp256k1_coord_prime, secp256r1_coord_prime},
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
use openvm_pairing_guest::bls12_381::BLS12_381_MODULUS;
use openvm_stark_backend::p3_field::FieldAlgebra;
use openvm_stark_sdk::{p3_baby_bear::BabyBear, utils::create_seeded_rng};
use openvm_ecc_transpiler::Rv32WeierstrassOpcode;
use rand::{rngs::StdRng, Rng};
#[cfg(feature = "cuda")]
use {
    crate::{EccRecord, WeierstrassAddChipGpu},
    openvm_circuit::arch::testing::{
        default_bitwise_lookup_bus, default_var_range_checker_bus, GpuChipTestBuilder,
        GpuTestChipHarness,
    },
    openvm_circuit_primitives::var_range::VariableRangeCheckerChip,
};

use crate::{
    get_ec_add_air, get_ec_add_chip, get_ec_add_step, get_ec_double_air, get_ec_double_chip,
    get_ec_double_step, EcDoubleExecutor, WeierstrassAir, WeierstrassChip,
};

const LIMB_BITS: usize = 8;
const MAX_INS_CAPACITY: usize = 128;
type F = BabyBear;

lazy_static::lazy_static! {
    // Sample points got from https://asecuritysite.com/ecc/ecc_points2 and
    // https://learnmeabitcoin.com/technical/cryptography/elliptic-curve/#add
    pub static ref SampleEcPoints: Vec<(BigUint, BigUint)> = {
        let x1 = BigUint::from_u32(1).unwrap();
        let y1 = BigUint::from_str(
            "29896722852569046015560700294576055776214335159245303116488692907525646231534",
        )
        .unwrap();
        let x2 = BigUint::from_u32(2).unwrap();
        let y2 = BigUint::from_str(
            "69211104694897500952317515077652022726490027694212560352756646854116994689233",
        )
        .unwrap();

        // This is the sum of (x1, y1) and (x2, y2).
        let x3 = BigUint::from_str(
            "109562500687829935604265064386702914290271628241900466384583316550888437213118",
        )
        .unwrap();
        let y3 = BigUint::from_str(
            "54782835737747434227939451500021052510566980337100013600092875738315717035444",
        )
        .unwrap();

        // This is the double of (x2, y2).
        let x4 = BigUint::from_str(
            "23158417847463239084714197001737581570653996933128112807891516801581766934331",
        )
        .unwrap();
        let y4 = BigUint::from_str(
            "25821202496262252602076867233819373685524812798827903993634621255495124276396",
        )
        .unwrap();

        // This is the sum of (x3, y3) and (x4, y4).
        let x5 = BigUint::from_str(
            "88733411122275068320336854419305339160905807011607464784153110222112026831518",
        )
        .unwrap();
        let y5 = BigUint::from_str(
            "69295025707265750480609159026651746584753914962418372690287755773539799515030",
        )
        .unwrap();

        vec![(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x5, y5)]
    };
}

mod ec_add_tests {
    use num_traits::One;

    use super::*;
    use crate::EcAddExecutor;

    type EcAddHarness<const BLOCKS: usize, const BLOCK_SIZE: usize> = TestChipHarness<
        F,
        EcAddExecutor<BLOCKS, BLOCK_SIZE>,
        WeierstrassAir<2, BLOCKS, BLOCK_SIZE>,
        WeierstrassChip<F, 2, BLOCKS, BLOCK_SIZE>,
    >;

    #[cfg(feature = "cuda")]
    type GpuHarness<const BLOCKS: usize, const BLOCK_SIZE: usize> = GpuTestChipHarness<
        F,
        EcAddExecutor<BLOCKS, BLOCK_SIZE>,
        WeierstrassAir<2, BLOCKS, BLOCK_SIZE>,
        WeierstrassAddChipGpu<BLOCKS, BLOCK_SIZE>,
        WeierstrassChip<F, 2, BLOCKS, BLOCK_SIZE>,
    >;

    fn create_harness<const BLOCKS: usize, const BLOCK_SIZE: usize>(
        tester: &VmChipTestBuilder<F>,
        config: ExprBuilderConfig,
        offset: usize,
        a: BigUint,
        b: BigUint,
    ) -> (
        EcAddHarness<BLOCKS, BLOCK_SIZE>,
        (
            BitwiseOperationLookupAir<RV32_CELL_BITS>,
            SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
        ),
    ) {
        let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
        let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV32_CELL_BITS>::new(
            bitwise_bus,
        ));

        let air = get_ec_add_air::<BLOCKS, BLOCK_SIZE>(
            tester.execution_bridge(),
            tester.memory_bridge(),
            config.clone(),
            tester.range_checker().bus(),
            bitwise_bus,
            tester.address_bits(),
            offset,
            a.clone(),
            b.clone(),
        );
        let executor = get_ec_add_step::<BLOCKS, BLOCK_SIZE>(
            config.clone(),
            tester.range_checker().bus(),
            tester.address_bits(),
            offset,
            a.clone(),
            b.clone(),
        );
        let chip = get_ec_add_chip::<F, BLOCKS, BLOCK_SIZE>(
            config.clone(),
            tester.memory_helper(),
            tester.range_checker(),
            bitwise_chip.clone(),
            tester.address_bits(),
            a,
            b,
        );

        let harness = EcAddHarness::with_capacity(executor, air, chip, MAX_INS_CAPACITY);

        (harness, (bitwise_chip.air, bitwise_chip))
    }

    #[cfg(feature = "cuda")]
    fn create_cuda_harness<const BLOCKS: usize, const BLOCK_SIZE: usize>(
        tester: &GpuChipTestBuilder,
        config: ExprBuilderConfig,
        offset: usize,
        a: BigUint,
        b: BigUint,
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

        let air = get_ec_add_air(
            tester.execution_bridge(),
            tester.memory_bridge(),
            config.clone(),
            range_bus,
            bitwise_bus,
            tester.address_bits(),
            offset,
            a.clone(),
            b.clone(),
        );
        let executor = get_ec_add_step(
            config.clone(),
            range_bus,
            tester.address_bits(),
            offset,
            a.clone(),
            b.clone(),
        );

        let cpu_chip = get_ec_add_chip(
            config.clone(),
            tester.dummy_memory_helper(),
            dummy_range_checker_chip,
            dummy_bitwise_chip,
            tester.address_bits(),
            a.clone(),
            b.clone(),
        );
        let gpu_chip = WeierstrassAddChipGpu::new(
            tester.range_checker(),
            tester.bitwise_op_lookup(),
            config,
            offset,
            a,
            b,
            tester.address_bits() as u32,
            tester.timestamp_max_bits() as u32,
        );

        GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
    }

    #[allow(clippy::too_many_arguments)]
    fn set_and_execute_ec_add<
        const BLOCKS: usize,
        const BLOCK_SIZE: usize,
        const NUM_LIMBS: usize,
        RA: Arena,
    >(
        tester: &mut impl TestBuilder<F>,
        executor: &mut EcAddExecutor<BLOCKS, BLOCK_SIZE>,
        arena: &mut RA,
        rng: &mut StdRng,
        modulus: &BigUint,
        a: &BigUint,
        b: &BigUint,
        is_setup: bool,
        offset: usize,
        p1: Option<(BigUint, BigUint)>,
        p2: Option<(BigUint, BigUint)>,
    ) where
        EcAddExecutor<BLOCKS, BLOCK_SIZE>: PreflightExecutor<F, RA>,
    {
        // For projective coordinates, each point has 3 coordinates (X, Y, Z)
        // For setup: P1 = (modulus, a, b), P2 = (1, 1, 1) (dummy)
        // For normal: P1 = (x1, y1, 1), P2 = (x2, y2, 1) (affine to projective)
        let (x1, y1, z1, x2, y2, z2, op_local) = if is_setup {
            (
                modulus.clone(),
                a.clone(),
                b.clone(),
                BigUint::one(),
                BigUint::one(),
                BigUint::one(),
                Rv32WeierstrassOpcode::SETUP_SW_EC_ADD_PROJ as usize,
            )
        } else if let Some((px1, py1)) = p1 {
            let (px2, py2) = p2.unwrap();
            let px1 = px1 % modulus;
            let py1 = py1 % modulus;
            let px2 = px2 % modulus;
            let py2 = py2 % modulus;
            let one = BigUint::one();
            if rng.gen_bool(0.5) {
                (
                    px1,
                    py1,
                    one.clone(),
                    px2,
                    py2,
                    one,
                    Rv32WeierstrassOpcode::SW_EC_ADD_PROJ as usize,
                )
            } else {
                (
                    px2,
                    py2,
                    one.clone(),
                    px1,
                    py1,
                    one,
                    Rv32WeierstrassOpcode::SW_EC_ADD_PROJ as usize,
                )
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
        let y1_limbs: Vec<F> = biguint_to_limbs_vec(&y1, NUM_LIMBS)
            .into_iter()
            .map(F::from_canonical_u8)
            .collect();
        let z1_limbs: Vec<F> = biguint_to_limbs_vec(&z1, NUM_LIMBS)
            .into_iter()
            .map(F::from_canonical_u8)
            .collect();
        let x2_limbs: Vec<F> = biguint_to_limbs_vec(&x2, NUM_LIMBS)
            .into_iter()
            .map(F::from_canonical_u8)
            .collect();
        let y2_limbs: Vec<F> = biguint_to_limbs_vec(&y2, NUM_LIMBS)
            .into_iter()
            .map(F::from_canonical_u8)
            .collect();
        let z2_limbs: Vec<F> = biguint_to_limbs_vec(&z2, NUM_LIMBS)
            .into_iter()
            .map(F::from_canonical_u8)
            .collect();

        // Write projective point P1 = (X1, Y1, Z1)
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
                (p1_base_addr + 2 * NUM_LIMBS as u32) as usize + i,
                z1_limbs[i..i + BLOCK_SIZE].try_into().unwrap(),
            );

            // Write projective point P2 = (X2, Y2, Z2)
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

            tester.write::<BLOCK_SIZE>(
                data_as,
                (p2_base_addr + 2 * NUM_LIMBS as u32) as usize + i,
                z2_limbs[i..i + BLOCK_SIZE].try_into().unwrap(),
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
        a: BigUint,
        b: BigUint,
    ) {
        let mut rng = create_seeded_rng();
        let mut tester: VmChipTestBuilder<F> = VmChipTestBuilder::default();
        let config = ExprBuilderConfig {
            modulus: modulus.clone(),
            num_limbs: NUM_LIMBS,
            limb_bits: LIMB_BITS,
        };

        let (mut harness, bitwise) =
            create_harness::<BLOCKS, BLOCK_SIZE>(&tester, config, offset, a.clone(), b.clone());

        set_and_execute_ec_add::<BLOCKS, BLOCK_SIZE, NUM_LIMBS, _>(
            &mut tester,
            &mut harness.executor,
            &mut harness.arena,
            &mut rng,
            &modulus,
            &a,
            &b,
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
            &a,
            &b,
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
            &a,
            &b,
            false,
            offset,
            Some(SampleEcPoints[2].clone()),
            Some(SampleEcPoints[3].clone()),
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
        // secp256k1: a=0, b=7, b3=21
        run_ec_add_test::<3, 32, 32>(
            Rv32WeierstrassOpcode::CLASS_OFFSET,
            secp256k1_coord_prime(),
            BigUint::zero(),
            BigUint::from(7u32), // secp256k1 b coefficient,
        );
    }

    #[test]
    fn test_ec_add_6x16() {
        // BLS12-381: a=0, b=4, b3=12
        run_ec_add_test::<9, 16, 48>(
            Rv32WeierstrassOpcode::CLASS_OFFSET,
            BLS12_381_MODULUS.clone(),
            BigUint::zero(),
            BigUint::from(4u32), // BLS12-381 b coefficient,
        );
    }

    #[cfg(feature = "cuda")]
    fn run_cuda_ec_add<const BLOCKS: usize, const BLOCK_SIZE: usize, const NUM_LIMBS: usize>(
        offset: usize,
        modulus: BigUint,
        a: BigUint,
        b: BigUint,
    ) {
        let mut rng = create_seeded_rng();

        let mut tester =
            GpuChipTestBuilder::default().with_bitwise_op_lookup(default_bitwise_lookup_bus());

        let config = ExprBuilderConfig {
            modulus: modulus.clone(),
            num_limbs: NUM_LIMBS,
            limb_bits: LIMB_BITS,
        };

        let mut harness = create_cuda_harness::<BLOCKS, BLOCK_SIZE>(
            &tester,
            config,
            offset,
            a.clone(),
            b.clone(),
        );

        set_and_execute_ec_add::<BLOCKS, BLOCK_SIZE, NUM_LIMBS, _>(
            &mut tester,
            &mut harness.executor,
            &mut harness.dense_arena,
            &mut rng,
            &modulus,
            &a,
            &b,
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
            &a,
            &b,
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
            &a,
            &b,
            false,
            offset,
            Some(SampleEcPoints[2].clone()),
            Some(SampleEcPoints[3].clone()),
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
    fn test_weierstrass_add_cuda_2x32() {
        run_cuda_ec_add::<3, 32, 32>(
            Rv32WeierstrassOpcode::CLASS_OFFSET,
            secp256k1_coord_prime(),
            BigUint::zero(),
            BigUint::from(7u32), // secp256k1 b coefficient,
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_weierstrass_add_cuda_6x16() {
        run_cuda_ec_add::<9, 16, 48>(
            Rv32WeierstrassOpcode::CLASS_OFFSET,
            BLS12_381_MODULUS.clone(),
            BigUint::zero(),
            BigUint::from(4u32), // BLS12-381 b coefficient,
        );
    }

    ///////////////////////////////////////////////////////////////////////////////////////
    /// SANITY TESTS
    ///
    /// Ensure that execute functions produce the correct results.
    ///////////////////////////////////////////////////////////////////////////////////////

    /// Helper to convert projective (X, Y, Z) to affine (x, y) via x = X/Z, y = Y/Z
    fn proj_to_affine(
        x_proj: &BigUint,
        y_proj: &BigUint,
        z_proj: &BigUint,
        p: &BigUint,
    ) -> (BigUint, BigUint) {
        let z_inv = z_proj.modpow(&(p - BigUint::from(2u32)), p);
        let x_affine = (x_proj * &z_inv) % p;
        let y_affine = (y_proj * &z_inv) % p;
        (x_affine, y_affine)
    }

    #[test]
    fn ec_add_sanity_test() {
        let tester: VmChipTestBuilder<F> = VmChipTestBuilder::default();
        let p = secp256k1_coord_prime();
        let config = ExprBuilderConfig {
            modulus: p.clone(),
            num_limbs: 32,
            limb_bits: LIMB_BITS,
        };

        // secp256k1: a=0, b=7
        let executor = get_ec_add_step::<3, 32>(
            config,
            tester.range_checker().bus(),
            tester.address_bits(),
            Rv32WeierstrassOpcode::CLASS_OFFSET,
            BigUint::zero(),
            BigUint::from(7u32),
        );

        let (p1_x, p1_y) = SampleEcPoints[0].clone();
        let (p2_x, p2_y) = SampleEcPoints[1].clone();

        // Projective input: (X1, Y1, Z1, X2, Y2, Z2) where Z=1 for affine points
        let z = BigUint::one();
        let r = executor.expr.execute_with_output(
            vec![p1_x, p1_y, z.clone(), p2_x, p2_y, z.clone()],
            vec![true],
        );

        assert_eq!(r.len(), 3); // X3, Y3, Z3
        let (x3_affine, y3_affine) = proj_to_affine(&r[0], &r[1], &r[2], &p);
        assert_eq!(x3_affine, SampleEcPoints[2].0);
        assert_eq!(y3_affine, SampleEcPoints[2].1);

        let (p1_x, p1_y) = SampleEcPoints[2].clone();
        let (p2_x, p2_y) = SampleEcPoints[3].clone();
        let r = executor.expr.execute_with_output(
            vec![p1_x, p1_y, z.clone(), p2_x, p2_y, z],
            vec![true],
        );

        assert_eq!(r.len(), 3); // X3, Y3, Z3
        let (x3_affine, y3_affine) = proj_to_affine(&r[0], &r[1], &r[2], &p);
        assert_eq!(x3_affine, SampleEcPoints[4].0);
        assert_eq!(y3_affine, SampleEcPoints[4].1);
    }
}

mod ec_double_tests {
    use num_traits::One;

    use super::*;

    type EcDoubleHarness<const BLOCKS: usize, const BLOCK_SIZE: usize> = TestChipHarness<
        F,
        EcDoubleExecutor<BLOCKS, BLOCK_SIZE>,
        WeierstrassAir<1, BLOCKS, BLOCK_SIZE>,
        WeierstrassChip<F, 1, BLOCKS, BLOCK_SIZE>,
        MatrixRecordArena<F>,
    >;

    #[cfg(feature = "cuda")]
    use crate::WeierstrassDoubleChipGpu;

    #[cfg(feature = "cuda")]
    type GpuHarness<const BLOCKS: usize, const BLOCK_SIZE: usize> = GpuTestChipHarness<
        F,
        EcDoubleExecutor<BLOCKS, BLOCK_SIZE>,
        WeierstrassAir<1, BLOCKS, BLOCK_SIZE>,
        WeierstrassDoubleChipGpu<BLOCKS, BLOCK_SIZE>,
        WeierstrassChip<F, 1, BLOCKS, BLOCK_SIZE>,
    >;

    fn create_harness<const BLOCKS: usize, const BLOCK_SIZE: usize>(
        tester: &VmChipTestBuilder<F>,
        config: ExprBuilderConfig,
        offset: usize,
        a_biguint: BigUint,
        b3: BigUint,
    ) -> (
        EcDoubleHarness<BLOCKS, BLOCK_SIZE>,
        (
            BitwiseOperationLookupAir<RV32_CELL_BITS>,
            SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
        ),
    ) {
        let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
        let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV32_CELL_BITS>::new(
            bitwise_bus,
        ));
        let air = get_ec_double_air(
            tester.execution_bridge(),
            tester.memory_bridge(),
            config.clone(),
            tester.range_checker().bus(),
            bitwise_bus,
            tester.address_bits(),
            offset,
            a_biguint.clone(),
            b3.clone(),
        );
        let executor = get_ec_double_step(
            config.clone(),
            tester.range_checker().bus(),
            tester.address_bits(),
            offset,
            a_biguint.clone(),
            b3.clone(),
        );
        let chip = get_ec_double_chip(
            config.clone(),
            tester.memory_helper(),
            tester.range_checker(),
            bitwise_chip.clone(),
            tester.address_bits(),
            a_biguint,
            b3,
        );
        let harness = EcDoubleHarness::with_capacity(executor, air, chip, MAX_INS_CAPACITY);

        (harness, (bitwise_chip.air, bitwise_chip))
    }

    #[cfg(feature = "cuda")]
    fn create_cuda_harness<const BLOCKS: usize, const BLOCK_SIZE: usize>(
        tester: &GpuChipTestBuilder,
        config: ExprBuilderConfig,
        offset: usize,
        a_biguint: BigUint,
        b: BigUint,
    ) -> GpuHarness<BLOCKS, BLOCK_SIZE> {
        // getting bus from tester since `gpu_chip` and `air` must use the same bus
        let range_bus = default_var_range_checker_bus();
        let bitwise_bus = default_bitwise_lookup_bus();
        // creating a dummy chip for Cpu so we only count `add_count`s from GPU
        let dummy_range_checker_chip = Arc::new(VariableRangeCheckerChip::new(range_bus));
        let dummy_bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV32_CELL_BITS>::new(
            bitwise_bus,
        ));

        let air = get_ec_double_air(
            tester.execution_bridge(),
            tester.memory_bridge(),
            config.clone(),
            range_bus,
            bitwise_bus,
            tester.address_bits(),
            offset,
            a_biguint.clone(),
            b.clone(),
        );
        let executor = get_ec_double_step(
            config.clone(),
            range_bus,
            tester.address_bits(),
            offset,
            a_biguint.clone(),
            b.clone(),
        );

        let cpu_chip = get_ec_double_chip(
            config.clone(),
            tester.dummy_memory_helper(),
            dummy_range_checker_chip,
            dummy_bitwise_chip,
            tester.address_bits(),
            a_biguint.clone(),
            b.clone(),
        );
        let gpu_chip = WeierstrassDoubleChipGpu::new(
            tester.range_checker(),
            tester.bitwise_op_lookup(),
            config,
            offset,
            a_biguint,
            b,
            tester.address_bits() as u32,
            tester.timestamp_max_bits() as u32,
        );

        GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
    }

    #[allow(clippy::too_many_arguments)]
    fn set_and_execute_ec_double<
        const BLOCKS: usize,
        const BLOCK_SIZE: usize,
        const NUM_LIMBS: usize,
        RA: Arena,
    >(
        tester: &mut impl TestBuilder<F>,
        executor: &mut EcDoubleExecutor<BLOCKS, BLOCK_SIZE>,
        arena: &mut RA,
        rng: &mut StdRng,
        modulus: &BigUint,
        a_biguint: &BigUint,
        b_biguint: &BigUint,
        is_setup: bool,
        offset: usize,
        x: Option<BigUint>,
        y: Option<BigUint>,
    ) where
        EcDoubleExecutor<BLOCKS, BLOCK_SIZE>: PreflightExecutor<F, RA>,
    {
        // For projective coordinates, each point has 3 coordinates (X, Y, Z)
        // For setup: P = (modulus, a, b)
        // For normal: P = (x, y, 1) (affine to projective)
        let (x1, y1, z1, op_local) = if is_setup {
            (
                modulus.clone(),
                a_biguint.clone(),
                b_biguint.clone(),
                Rv32WeierstrassOpcode::SETUP_SW_EC_DOUBLE_PROJ as usize,
            )
        } else if let Some(x) = x {
            let y = y.unwrap();
            let x = x % modulus;
            let y = y % modulus;
            (
                x,
                y,
                BigUint::one(),
                Rv32WeierstrassOpcode::SW_EC_DOUBLE_PROJ as usize,
            )
        } else {
            let x = generate_random_biguint(modulus);
            let y = generate_random_biguint(modulus);

            (
                x,
                y,
                BigUint::one(),
                Rv32WeierstrassOpcode::SW_EC_DOUBLE_PROJ as usize,
            )
        };

        let ptr_as = RV32_REGISTER_AS as usize;
        let data_as = RV32_MEMORY_AS as usize;

        let rs1_ptr = gen_pointer(rng, RV32_REGISTER_NUM_LIMBS);
        let rd_ptr = gen_pointer(rng, RV32_REGISTER_NUM_LIMBS);

        let p1_base_addr = gen_pointer(rng, BLOCK_SIZE) as u32;
        let result_base_addr = gen_pointer(rng, BLOCK_SIZE) as u32;

        tester.write::<RV32_REGISTER_NUM_LIMBS>(
            ptr_as,
            rs1_ptr,
            p1_base_addr.to_le_bytes().map(F::from_canonical_u8),
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
        let y1_limbs: Vec<F> = biguint_to_limbs_vec(&y1, NUM_LIMBS)
            .into_iter()
            .map(F::from_canonical_u8)
            .collect();
        let z1_limbs: Vec<F> = biguint_to_limbs_vec(&z1, NUM_LIMBS)
            .into_iter()
            .map(F::from_canonical_u8)
            .collect();

        // Write projective point P = (X, Y, Z)
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
                (p1_base_addr + 2 * NUM_LIMBS as u32) as usize + i,
                z1_limbs[i..i + BLOCK_SIZE].try_into().unwrap(),
            );
        }

        let instruction = Instruction::from_isize(
            VmOpcode::from_usize(offset + op_local),
            rd_ptr as isize,
            rs1_ptr as isize,
            0,
            ptr_as as isize,
            data_as as isize,
        );

        tester.execute(executor, arena, &instruction);
    }

    fn run_ec_double_test<const BLOCKS: usize, const BLOCK_SIZE: usize, const NUM_LIMBS: usize>(
        offset: usize,
        modulus: BigUint,
        num_ops: usize,
        a: BigUint,
        b: BigUint,
    ) {
        let mut rng = create_seeded_rng();
        let mut tester: VmChipTestBuilder<F> = VmChipTestBuilder::default();
        let config = ExprBuilderConfig {
            modulus: modulus.clone(),
            num_limbs: NUM_LIMBS,
            limb_bits: LIMB_BITS,
        };

        let (mut harness, bitwise) =
            create_harness::<BLOCKS, BLOCK_SIZE>(&tester, config, offset, a.clone(), b.clone());

        for i in 0..num_ops {
            set_and_execute_ec_double::<BLOCKS, BLOCK_SIZE, NUM_LIMBS, _>(
                &mut tester,
                &mut harness.executor,
                &mut harness.arena,
                &mut rng,
                &modulus,
                &a,
                &b,
                i == 0,
                offset,
                None,
                None,
            );
        }

        set_and_execute_ec_double::<BLOCKS, BLOCK_SIZE, NUM_LIMBS, _>(
            &mut tester,
            &mut harness.executor,
            &mut harness.arena,
            &mut rng,
            &modulus,
            &a,
            &b,
            false,
            offset,
            Some(SampleEcPoints[0].0.clone()),
            Some(SampleEcPoints[0].1.clone()),
        );

        set_and_execute_ec_double::<BLOCKS, BLOCK_SIZE, NUM_LIMBS, _>(
            &mut tester,
            &mut harness.executor,
            &mut harness.arena,
            &mut rng,
            &modulus,
            &a,
            &b,
            false,
            offset,
            Some(SampleEcPoints[1].0.clone()),
            Some(SampleEcPoints[1].1.clone()),
        );

        // Testing data from: http://point-at-infinity.org/ecc/nisttv
        let p1_x = BigUint::from_str_radix(
            "6B17D1F2E12C4247F8BCE6E563A440F277037D812DEB33A0F4A13945D898C296",
            16,
        )
        .unwrap();
        let p1_y = BigUint::from_str_radix(
            "4FE342E2FE1A7F9B8EE7EB4A7C0F9E162BCE33576B315ECECBB6406837BF51F5",
            16,
        )
        .unwrap();

        set_and_execute_ec_double::<BLOCKS, BLOCK_SIZE, NUM_LIMBS, _>(
            &mut tester,
            &mut harness.executor,
            &mut harness.arena,
            &mut rng,
            &modulus,
            &a,
            &b,
            false,
            offset,
            Some(p1_x),
            Some(p1_y),
        );

        let tester = tester
            .build()
            .load(harness)
            .load_periphery(bitwise)
            .finalize();

        tester.simple_test().expect("Verification failed");
    }

    #[test]
    fn test_ec_double_2x32() {
        // secp256k1: a=0, b=7, b3=21
        run_ec_double_test::<3, 32, 32>(
            Rv32WeierstrassOpcode::CLASS_OFFSET,
            secp256k1_coord_prime(),
            50,
            BigUint::zero(),
            BigUint::from(7u32), // secp256k1 b coefficient,
        );
    }

    #[test]
    fn test_ec_double_2x32_nonzero_a_1() {
        // secp256r1: a=-3 (p-3),
        // b=0x5ac635d8aa3a93e7b3ebbd55769886bc651d06b0cc53b0f63bce3c3e27d2604b
        let coeff_a = (-secp256r1::Fp::from(3)).to_bytes();
        let a = BigUint::from_bytes_le(&coeff_a);
        // b coefficient (functions compute b3 = 3*b internally)
        let b = BigUint::from_str_radix(
            "5ac635d8aa3a93e7b3ebbd55769886bc651d06b0cc53b0f63bce3c3e27d2604b",
            16,
        )
        .unwrap();

        run_ec_double_test::<3, 32, 32>(
            Rv32WeierstrassOpcode::CLASS_OFFSET,
            secp256r1_coord_prime(),
            50,
            a,
            b,
        );
    }

    #[test]
    fn test_ec_double_6x16() {
        // BLS12-381: a=0, b=4, b3=12
        run_ec_double_test::<9, 16, 48>(
            Rv32WeierstrassOpcode::CLASS_OFFSET,
            BLS12_381_MODULUS.clone(),
            50,
            BigUint::zero(),
            BigUint::from(4u32), // BLS12-381 b coefficient,
        );
    }

    #[cfg(feature = "cuda")]
    fn run_ec_double_cuda_test<
        const BLOCKS: usize,
        const BLOCK_SIZE: usize,
        const NUM_LIMBS: usize,
    >(
        offset: usize,
        modulus: BigUint,
        num_ops: usize,
        a: BigUint,
        b: BigUint,
    ) {
        let mut rng = create_seeded_rng();

        let mut tester =
            GpuChipTestBuilder::default().with_bitwise_op_lookup(default_bitwise_lookup_bus());

        let config = ExprBuilderConfig {
            modulus: modulus.clone(),
            num_limbs: NUM_LIMBS,
            limb_bits: LIMB_BITS,
        };

        let mut harness = create_cuda_harness::<BLOCKS, BLOCK_SIZE>(
            &tester,
            config,
            offset,
            a.clone(),
            b.clone(),
        );

        for i in 0..num_ops {
            set_and_execute_ec_double::<BLOCKS, BLOCK_SIZE, NUM_LIMBS, _>(
                &mut tester,
                &mut harness.executor,
                &mut harness.dense_arena,
                &mut rng,
                &modulus,
                &a,
                &b,
                i == 0,
                offset,
                None,
                None,
            );
        }

        set_and_execute_ec_double::<BLOCKS, BLOCK_SIZE, NUM_LIMBS, _>(
            &mut tester,
            &mut harness.executor,
            &mut harness.dense_arena,
            &mut rng,
            &modulus,
            &a,
            &b,
            false,
            offset,
            Some(SampleEcPoints[0].0.clone()),
            Some(SampleEcPoints[0].1.clone()),
        );

        set_and_execute_ec_double::<BLOCKS, BLOCK_SIZE, NUM_LIMBS, _>(
            &mut tester,
            &mut harness.executor,
            &mut harness.dense_arena,
            &mut rng,
            &modulus,
            &a,
            &b,
            false,
            offset,
            Some(SampleEcPoints[1].0.clone()),
            Some(SampleEcPoints[1].1.clone()),
        );

        // Testing data from: http://point-at-infinity.org/ecc/nisttv
        let p1_x = BigUint::from_str_radix(
            "6B17D1F2E12C4247F8BCE6E563A440F277037D812DEB33A0F4A13945D898C296",
            16,
        )
        .unwrap();
        let p1_y = BigUint::from_str_radix(
            "4FE342E2FE1A7F9B8EE7EB4A7C0F9E162BCE33576B315ECECBB6406837BF51F5",
            16,
        )
        .unwrap();

        set_and_execute_ec_double::<BLOCKS, BLOCK_SIZE, NUM_LIMBS, _>(
            &mut tester,
            &mut harness.executor,
            &mut harness.dense_arena,
            &mut rng,
            &modulus,
            &a,
            &b,
            false,
            offset,
            Some(p1_x),
            Some(p1_y),
        );

        harness
            .dense_arena
            .get_record_seeker::<EccRecord<1, BLOCKS, BLOCK_SIZE>, _>()
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
    fn test_ec_double_cuda_2x32() {
        // secp256k1: a=0, b=7, b3=21
        run_ec_double_cuda_test::<3, 32, 32>(
            Rv32WeierstrassOpcode::CLASS_OFFSET,
            secp256k1_coord_prime(),
            50,
            BigUint::zero(),
            BigUint::from(7u32), // secp256k1 b coefficient,
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_ec_double_cuda_2x32_nonzero_a_1() {
        // secp256r1: a=-3 (p-3),
        // b=0x5ac635d8aa3a93e7b3ebbd55769886bc651d06b0cc53b0f63bce3c3e27d2604b
        let coeff_a = (-secp256r1::Fp::from(3)).to_bytes();
        let a = BigUint::from_bytes_le(&coeff_a);
        // b coefficient (functions compute b3 = 3*b internally)
        let b = BigUint::from_str_radix(
            "5ac635d8aa3a93e7b3ebbd55769886bc651d06b0cc53b0f63bce3c3e27d2604b",
            16,
        )
        .unwrap();

        run_ec_double_cuda_test::<3, 32, 32>(
            Rv32WeierstrassOpcode::CLASS_OFFSET,
            secp256r1_coord_prime(),
            50,
            a,
            b,
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_ec_double_cuda_6x16() {
        // BLS12-381: a=0, b=4, b3=12
        run_ec_double_cuda_test::<9, 16, 48>(
            Rv32WeierstrassOpcode::CLASS_OFFSET,
            BLS12_381_MODULUS.clone(),
            50,
            BigUint::zero(),
            BigUint::from(4u32), // BLS12-381 b coefficient,
        );
    }

    ///////////////////////////////////////////////////////////////////////////////////////
    /// SANITY TESTS
    ///
    /// Ensure that execute functions produce the correct results.
    ///////////////////////////////////////////////////////////////////////////////////////

    /// Helper to convert projective (X, Y, Z) to affine (x, y) via x = X/Z, y = Y/Z
    fn proj_to_affine(
        x_proj: &BigUint,
        y_proj: &BigUint,
        z_proj: &BigUint,
        p: &BigUint,
    ) -> (BigUint, BigUint) {
        // Compute z^{-1} mod p using Fermat's little theorem: z^{-1} = z^{p-2} mod p
        let z_inv = z_proj.modpow(&(p - BigUint::from(2u32)), p);
        let x_affine = (x_proj * &z_inv) % p;
        let y_affine = (y_proj * &z_inv) % p;
        (x_affine, y_affine)
    }

    #[test]
    fn ec_double_sanity_test_sample_ec_points() {
        let tester: VmChipTestBuilder<F> = VmChipTestBuilder::default();
        let p = secp256k1_coord_prime();
        let config = ExprBuilderConfig {
            modulus: p.clone(),
            num_limbs: 32,
            limb_bits: LIMB_BITS,
        };

        // secp256k1: a=0, b=7, b3=21
        let executor = get_ec_double_step::<3, 32>(
            config,
            tester.range_checker().bus(),
            tester.address_bits(),
            Rv32WeierstrassOpcode::CLASS_OFFSET,
            BigUint::zero(),
            BigUint::from(7u32), // secp256k1 b coefficient,
        );

        let (p1_x, p1_y) = SampleEcPoints[1].clone();

        // Projective input: (X, Y, Z) where Z=1 for affine point
        let z1 = BigUint::from(1u32);
        let r = executor
            .expr
            .execute_with_output(vec![p1_x, p1_y, z1], vec![true]);

        // Output is projective coordinates in (X3, Y3, Z3) order.
        assert_eq!(r.len(), 3);

        // Convert projective output to affine and compare
        let (x3_affine, y3_affine) = proj_to_affine(&r[0], &r[1], &r[2], &p);
        assert_eq!(x3_affine, SampleEcPoints[3].0);
        assert_eq!(y3_affine, SampleEcPoints[3].1);
    }

    #[test]
    fn ec_double_sanity_test() {
        let tester: VmChipTestBuilder<F> = VmChipTestBuilder::default();
        let p = secp256r1_coord_prime();
        let config = ExprBuilderConfig {
            modulus: p.clone(),
            num_limbs: 32,
            limb_bits: LIMB_BITS,
        };
        // secp256r1: a=-3 (p-3),
        // b=0x5ac635d8aa3a93e7b3ebbd55769886bc651d06b0cc53b0f63bce3c3e27d2604b
        let a = BigUint::from_str_radix(
            "ffffffff00000001000000000000000000000000fffffffffffffffffffffffc",
            16,
        )
        .unwrap();
        // b coefficient (functions compute b3 = 3*b internally)
        let b = BigUint::from_str_radix(
            "5ac635d8aa3a93e7b3ebbd55769886bc651d06b0cc53b0f63bce3c3e27d2604b",
            16,
        )
        .unwrap();

        let executor = get_ec_double_step::<3, 32>(
            config.clone(),
            tester.range_checker().bus(),
            tester.address_bits(),
            Rv32WeierstrassOpcode::CLASS_OFFSET,
            a.clone(),
            b,
        );

        // Testing data from: http://point-at-infinity.org/ecc/nisttv
        let p1_x = BigUint::from_str_radix(
            "6B17D1F2E12C4247F8BCE6E563A440F277037D812DEB33A0F4A13945D898C296",
            16,
        )
        .unwrap();
        let p1_y = BigUint::from_str_radix(
            "4FE342E2FE1A7F9B8EE7EB4A7C0F9E162BCE33576B315ECECBB6406837BF51F5",
            16,
        )
        .unwrap();

        // Projective input: (X, Y, Z) where Z=1 for affine point
        let z1 = BigUint::from(1u32);
        let r = executor
            .expr
            .execute_with_output(vec![p1_x, p1_y, z1], vec![true]);

        // Output is projective coordinates in (X3, Y3, Z3) order.
        assert_eq!(r.len(), 3);

        let expected_double_x = BigUint::from_str_radix(
            "7CF27B188D034F7E8A52380304B51AC3C08969E277F21B35A60B48FC47669978",
            16,
        )
        .unwrap();
        let expected_double_y = BigUint::from_str_radix(
            "07775510DB8ED040293D9AC69F7430DBBA7DADE63CE982299E04B79D227873D1",
            16,
        )
        .unwrap();

        // Convert projective output to affine and compare
        let (x3_affine, y3_affine) = proj_to_affine(&r[0], &r[1], &r[2], &p);
        assert_eq!(x3_affine, expected_double_x);
        assert_eq!(y3_affine, expected_double_y);
    }
}
