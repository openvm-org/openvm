use std::{array::from_fn, borrow::BorrowMut};

use num_bigint::BigUint;
use num_traits::Zero;
use openvm_algebra_transpiler::Rv32ModularArithmeticOpcode;
use openvm_circuit::arch::{
    instructions::LocalOpcode,
    testing::{VmChipTestBuilder, BITWISE_OP_LOOKUP_BUS},
};
use openvm_circuit_primitives::{
    bigint::utils::{big_uint_to_limbs, secp256k1_coord_prime, secp256k1_scalar_prime},
    bitwise_op_lookup::{BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip},
};
use openvm_instructions::{instruction::Instruction, riscv::RV32_CELL_BITS, VmOpcode};
use openvm_mod_circuit_builder::{
    test_utils::{biguint_to_limbs, generate_field_element},
    ExprBuilderConfig,
};
use openvm_pairing_guest::bls12_381::BLS12_381_MODULUS;
use openvm_rv32_adapters::{rv32_write_heap_default, write_ptr_reg};
use openvm_rv32im_circuit::adapters::RV32_REGISTER_NUM_LIMBS;
use openvm_stark_backend::p3_field::FieldAlgebra;
use openvm_stark_sdk::{p3_baby_bear::BabyBear, utils::create_seeded_rng};
use rand::Rng;

use super::{
    ModularAddSubChip, ModularIsEqualChip, ModularIsEqualCoreAir, ModularIsEqualCoreCols,
    ModularMulDivChip,
};
use rand::rngs::StdRng;

const NUM_LIMBS: usize = 32;
const LIMB_BITS: usize = 8;
const BLOCK_SIZE: usize = 32;
const MAX_INS_CAPACITY: usize = 128;
type F = BabyBear;

#[cfg(test)]
mod addsubtests {
    use super::*;
    use test_case::test_case;

    const ADD_LOCAL: usize = Rv32ModularArithmeticOpcode::ADD as usize;

    fn set_and_execute_addsub(
        tester: &mut VmChipTestBuilder<F>,
        chip: &mut ModularAddSubChip<F, 2, NUM_LIMBS>,
        modulus: &BigUint,
        is_setup: bool,
    ) {
        let mut rng = create_seeded_rng();

        let (a, b, op) = if is_setup {
            (modulus.clone(), BigUint::zero(), ADD_LOCAL + 2)
        } else {
            let a_digits: Vec<_> = (0..NUM_LIMBS)
                .map(|_| rng.gen_range(0..(1 << LIMB_BITS)))
                .collect();
            let mut a = BigUint::new(a_digits.clone());
            let b_digits: Vec<_> = (0..NUM_LIMBS)
                .map(|_| rng.gen_range(0..(1 << LIMB_BITS)))
                .collect();
            let mut b = BigUint::new(b_digits.clone());

            let op = rng.gen_range(0..2) + ADD_LOCAL; // 0 for add, 1 for sub
            a %= modulus;
            b %= modulus;
            (a, b, op)
        };

        let expected_answer = match op - ADD_LOCAL {
            0 => (&a + &b) % modulus,
            1 => (&a + modulus - &b) % modulus,
            2 => a.clone() % modulus,
            _ => panic!(),
        };

        // Write to memories
        // For each biguint (a, b, r), there are 2 writes:
        // 1. address_ptr which stores the actual address
        // 2. actual address which stores the biguint limbs
        // The write of result r is done in the chip.
        let ptr_as = 1;
        let addr_ptr1 = 0;
        let addr_ptr2 = 3 * RV32_REGISTER_NUM_LIMBS;
        let addr_ptr3 = 6 * RV32_REGISTER_NUM_LIMBS;

        let data_as = 2;
        let address1 = 0u32;
        let address2 = 128u32;
        let address3 = (1 << 28) + 1234; // a large memory address to test heap adapter

        write_ptr_reg(tester, ptr_as, addr_ptr1, address1);
        write_ptr_reg(tester, ptr_as, addr_ptr2, address2);
        write_ptr_reg(tester, ptr_as, addr_ptr3, address3);

        let a_limbs: [BabyBear; NUM_LIMBS] =
            biguint_to_limbs(a.clone(), LIMB_BITS).map(BabyBear::from_canonical_u32);
        tester.write(data_as, address1 as usize, a_limbs);
        let b_limbs: [BabyBear; NUM_LIMBS] =
            biguint_to_limbs(b.clone(), LIMB_BITS).map(BabyBear::from_canonical_u32);
        tester.write(data_as, address2 as usize, b_limbs);

        let instruction = Instruction::from_isize(
            VmOpcode::from_usize(chip.0.step.offset + op),
            addr_ptr3 as isize,
            addr_ptr1 as isize,
            addr_ptr2 as isize,
            ptr_as as isize,
            data_as as isize,
        );
        tester.execute(chip, &instruction);

        let expected_limbs = biguint_to_limbs::<NUM_LIMBS>(expected_answer, LIMB_BITS);
        for i in (0..NUM_LIMBS).step_by(4) {
            let address = address3 as usize + i;
            let read_vals = tester.read::<4>(data_as, address);
            for (j, expected) in expected_limbs[i..i + 4].iter().enumerate() {
                assert_eq!(F::from_canonical_u32(*expected), read_vals[j]);
            }
        }
    }

    #[test_case(0, secp256k1_coord_prime(), 50)]
    #[test_case(4, secp256k1_scalar_prime(), 50)]
    fn test_addsub(opcode_offset: usize, modulus: BigUint, num_ops: usize) {
        let mut tester: VmChipTestBuilder<F> = VmChipTestBuilder::default();
        let config = ExprBuilderConfig {
            modulus: modulus.clone(),
            num_limbs: NUM_LIMBS,
            limb_bits: LIMB_BITS,
        };
        let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
        let bitwise_chip = SharedBitwiseOperationLookupChip::<RV32_CELL_BITS>::new(bitwise_bus);

        // doing 1xNUM_LIMBS reads and writes
        let mut chip = ModularAddSubChip::new(
            tester.execution_bridge(),
            tester.memory_bridge(),
            tester.memory_helper(),
            tester.address_bits(),
            config,
            Rv32ModularArithmeticOpcode::CLASS_OFFSET + opcode_offset,
            bitwise_chip.clone(),
            tester.range_checker(),
            MAX_INS_CAPACITY,
        );

        for i in 0..num_ops {
            set_and_execute_addsub(&mut tester, &mut chip, &modulus, i == 0);
        }

        let tester = tester.build().load(chip).load(bitwise_chip).finalize();
        tester.simple_test().expect("Verification failed");
    }
}

#[cfg(test)]
mod muldivtests {
    use super::*;
    use test_case::test_case;

    const MUL_LOCAL: usize = Rv32ModularArithmeticOpcode::MUL as usize;

    fn set_and_execute_muldiv(
        tester: &mut VmChipTestBuilder<F>,
        chip: &mut ModularMulDivChip<F, 2, NUM_LIMBS>,
        modulus: &BigUint,
        is_setup: bool,
    ) {
        let mut rng = create_seeded_rng();

        let (a, b, op) = if is_setup {
            (modulus.clone(), BigUint::zero(), MUL_LOCAL + 2)
        } else {
            let a_digits: Vec<_> = (0..NUM_LIMBS)
                .map(|_| rng.gen_range(0..(1 << LIMB_BITS)))
                .collect();
            let mut a = BigUint::new(a_digits.clone());
            let b_digits: Vec<_> = (0..NUM_LIMBS)
                .map(|_| rng.gen_range(0..(1 << LIMB_BITS)))
                .collect();
            let mut b = BigUint::new(b_digits.clone());

            let op = rng.gen_range(0..2) + MUL_LOCAL; // 0 for add, 1 for sub
            a %= modulus;
            b %= modulus;
            (a, b, op)
        };

        let expected_answer = match op - MUL_LOCAL {
            0 => (&a + &b) % modulus,
            1 => (&a + modulus - &b) % modulus,
            2 => a.clone() % modulus,
            _ => panic!(),
        };

        // Write to memories
        // For each biguint (a, b, r), there are 2 writes:
        // 1. address_ptr which stores the actual address
        // 2. actual address which stores the biguint limbs
        // The write of result r is done in the chip.
        let ptr_as = 1;
        let addr_ptr1 = 0;
        let addr_ptr2 = 12;
        let addr_ptr3 = 24;

        let data_as = 2;
        let address1 = 0;
        let address2 = 128;
        let address3 = 256;

        write_ptr_reg(tester, ptr_as, addr_ptr1, address1);
        write_ptr_reg(tester, ptr_as, addr_ptr2, address2);
        write_ptr_reg(tester, ptr_as, addr_ptr3, address3);

        let a_limbs: [F; NUM_LIMBS] =
            biguint_to_limbs(a.clone(), LIMB_BITS).map(F::from_canonical_u32);
        tester.write(data_as, address1 as usize, a_limbs);
        let b_limbs: [F; NUM_LIMBS] =
            biguint_to_limbs(b.clone(), LIMB_BITS).map(F::from_canonical_u32);
        tester.write(data_as, address2 as usize, b_limbs);

        let instruction = Instruction::from_isize(
            VmOpcode::from_usize(chip.0.step.offset + op),
            addr_ptr3 as isize,
            addr_ptr1 as isize,
            addr_ptr2 as isize,
            ptr_as as isize,
            data_as as isize,
        );
        tester.execute(chip, &instruction);

        let expected_limbs = biguint_to_limbs::<NUM_LIMBS>(expected_answer, LIMB_BITS);
        for i in (0..NUM_LIMBS).step_by(4) {
            let address = address3 as usize + i;
            let read_vals = tester.read::<4>(data_as, address);
            for (j, expected) in expected_limbs[i..i + 4].iter().enumerate() {
                assert_eq!(F::from_canonical_u32(*expected), read_vals[j]);
            }
        }
    }

    #[test_case(0, secp256k1_coord_prime(), 50)]
    #[test_case(4, secp256k1_scalar_prime(), 50)]
    fn test_muldiv(opcode_offset: usize, modulus: BigUint, num_ops: usize) {
        let mut tester: VmChipTestBuilder<F> = VmChipTestBuilder::default();
        let config = ExprBuilderConfig {
            modulus: modulus.clone(),
            num_limbs: NUM_LIMBS,
            limb_bits: LIMB_BITS,
        };
        let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
        let bitwise_chip = SharedBitwiseOperationLookupChip::<RV32_CELL_BITS>::new(bitwise_bus);
        // doing 1xNUM_LIMBS reads and writes
        let mut chip = ModularMulDivChip::new(
            tester.execution_bridge(),
            tester.memory_bridge(),
            tester.memory_helper(),
            tester.address_bits(),
            config,
            Rv32ModularArithmeticOpcode::CLASS_OFFSET + opcode_offset,
            bitwise_chip.clone(),
            tester.range_checker(),
            MAX_INS_CAPACITY,
        );

        for i in 0..num_ops {
            set_and_execute_muldiv(&mut tester, &mut chip, &modulus, i == 0);
        }
        let tester = tester.build().load(chip).load(bitwise_chip).finalize();

        tester.simple_test().expect("Verification failed");
    }
}

#[cfg(test)]
mod is_equal_tests {
    use crate::modular_chip::{ModularIsEqualAir, ModularIsEqualStep};

    use super::*;
    use openvm_rv32_adapters::{Rv32IsEqualModAdapterAir, Rv32IsEqualModeAdapterStep};
    use openvm_stark_backend::{
        p3_air::BaseAir,
        p3_matrix::{dense::DenseMatrix, Matrix},
        utils::disable_debug_builder,
        verifier::VerificationError,
    };

    fn create_test_chips<
        const NUM_LANES: usize,
        const LANE_SIZE: usize,
        const TOTAL_LIMBS: usize,
    >(
        tester: &mut VmChipTestBuilder<F>,
        modulus: &BigUint,
        modulus_limbs: [u8; TOTAL_LIMBS],
        offset: usize,
    ) -> (
        ModularIsEqualChip<F, NUM_LANES, LANE_SIZE, TOTAL_LIMBS>,
        SharedBitwiseOperationLookupChip<LIMB_BITS>,
    ) {
        let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
        let bitwise_chip = SharedBitwiseOperationLookupChip::<LIMB_BITS>::new(bitwise_bus);

        let chip = ModularIsEqualChip::<F, NUM_LANES, LANE_SIZE, TOTAL_LIMBS>::new(
            ModularIsEqualAir::new(
                Rv32IsEqualModAdapterAir::new(
                    tester.execution_bridge(),
                    tester.memory_bridge(),
                    bitwise_bus,
                    tester.address_bits(),
                ),
                ModularIsEqualCoreAir::new(modulus.clone(), bitwise_bus, offset),
            ),
            ModularIsEqualStep::new(
                Rv32IsEqualModeAdapterStep::new(tester.address_bits(), bitwise_chip.clone()),
                modulus_limbs,
                offset,
                bitwise_chip.clone(),
            ),
            MAX_INS_CAPACITY,
            tester.memory_helper(),
        );

        (chip, bitwise_chip)
    }

    fn set_and_execute_is_equal<
        const NUM_LANES: usize,
        const LANE_SIZE: usize,
        const TOTAL_LIMBS: usize,
    >(
        tester: &mut VmChipTestBuilder<F>,
        chip: &mut ModularIsEqualChip<F, NUM_LANES, LANE_SIZE, TOTAL_LIMBS>,
        rng: &mut StdRng,
        modulus: &BigUint,
        offset: usize,
        modulus_limbs: [F; TOTAL_LIMBS],
        is_setup: bool,
        b: Option<[F; TOTAL_LIMBS]>,
        c: Option<[F; TOTAL_LIMBS]>,
    ) {
        let instruction = if is_setup {
            rv32_write_heap_default::<TOTAL_LIMBS>(
                tester,
                vec![modulus_limbs],
                vec![[F::ZERO; TOTAL_LIMBS]],
                offset + Rv32ModularArithmeticOpcode::SETUP_ISEQ as usize,
            )
        } else {
            let b = b.unwrap_or(
                generate_field_element::<TOTAL_LIMBS, LIMB_BITS>(modulus, rng)
                    .map(F::from_canonical_u32),
            );
            let c = c.unwrap_or(if rng.gen_bool(0.5) {
                b
            } else {
                generate_field_element::<TOTAL_LIMBS, LIMB_BITS>(modulus, rng)
                    .map(F::from_canonical_u32)
            });

            rv32_write_heap_default::<TOTAL_LIMBS>(
                tester,
                vec![b],
                vec![c],
                offset + Rv32ModularArithmeticOpcode::IS_EQ as usize,
            )
        };
        tester.execute(chip, &instruction);
    }

    //////////////////////////////////////////////////////////////////////////////////////
    // POSITIVE TESTS
    //
    // Randomly generate computations and execute, ensuring that the generated trace
    // passes all constraints.
    //////////////////////////////////////////////////////////////////////////////////////

    #[test]
    fn test_modular_is_equal_1x32() {
        test_is_equal::<1, 32, 32>(17, secp256k1_coord_prime(), 100);
    }

    #[test]
    fn test_modular_is_equal_3x16() {
        test_is_equal::<3, 16, 48>(17, BLS12_381_MODULUS.clone(), 100);
    }

    fn test_is_equal<const NUM_LANES: usize, const LANE_SIZE: usize, const TOTAL_LIMBS: usize>(
        opcode_offset: usize,
        modulus: BigUint,
        num_tests: usize,
    ) {
        let mut rng = create_seeded_rng();
        let mut tester: VmChipTestBuilder<F> = VmChipTestBuilder::default();

        let vec = big_uint_to_limbs(&modulus, LIMB_BITS);
        let modulus_limbs: [u8; TOTAL_LIMBS] =
            from_fn(|i| if i < vec.len() { vec[i] as u8 } else { 0 });

        let (mut chip, bitwise_chip) = create_test_chips::<NUM_LANES, LANE_SIZE, TOTAL_LIMBS>(
            &mut tester,
            &modulus,
            modulus_limbs,
            opcode_offset,
        );

        let modulus_limbs = modulus_limbs.map(F::from_canonical_u8);

        for _ in 0..num_tests {
            set_and_execute_is_equal(
                &mut tester,
                &mut chip,
                &mut rng,
                &modulus,
                opcode_offset,
                modulus_limbs,
                false,
                None,
                None,
            );
        }

        // Special case where b == c are close to the prime
        let mut b = modulus_limbs;
        b[0] -= F::ONE;
        set_and_execute_is_equal(
            &mut tester,
            &mut chip,
            &mut rng,
            &modulus,
            opcode_offset,
            modulus_limbs,
            false,
            Some(b),
            None,
        );

        let tester = tester.build().load(chip).load(bitwise_chip).finalize();
        tester.simple_test().expect("Verification failed");
    }

    //////////////////////////////////////////////////////////////////////////////////////
    // NEGATIVE TESTS
    //
    // Given a fake trace of a single operation, setup a chip and run the test. We replace
    // part of the trace and check that the chip throws the expected error.
    //////////////////////////////////////////////////////////////////////////////////////

    /// Negative tests test for 3 "type" of errors determined by the value of b[0]:
    fn run_negative_is_equal_test<
        const NUM_LANES: usize,
        const LANE_SIZE: usize,
        const READ_LIMBS: usize,
    >(
        modulus: BigUint,
        opcode_offset: usize,
        b: [F; READ_LIMBS],
        c: [F; READ_LIMBS],
        expected_error: VerificationError,
    ) {
        let mut rng = create_seeded_rng();
        let mut tester: VmChipTestBuilder<F> = VmChipTestBuilder::default();

        let vec = big_uint_to_limbs(&modulus, LIMB_BITS);
        let modulus_limbs: [u8; READ_LIMBS] =
            from_fn(|i| if i < vec.len() { vec[i] as u8 } else { 0 });

        let (mut chip, bitwise_chip) = create_test_chips::<NUM_LANES, LANE_SIZE, READ_LIMBS>(
            &mut tester,
            &modulus,
            modulus_limbs,
            opcode_offset,
        );

        let modulus_limbs = modulus_limbs.map(F::from_canonical_u8);

        set_and_execute_is_equal(
            &mut tester,
            &mut chip,
            &mut rng,
            &modulus,
            opcode_offset,
            modulus_limbs,
            true,
            Some(b),
            Some(c),
        );

        let adapter_width = BaseAir::<F>::width(&chip.air.adapter);
        let modify_trace = |trace: &mut DenseMatrix<F>| {
            let mut values = trace.row_slice(0).to_vec();
            let row_slice: &mut ModularIsEqualCoreCols<_, READ_LIMBS> =
                values.split_at_mut(adapter_width).1.borrow_mut();
            if row_slice.b[0] == F::ONE {
                // test the constraint that c_lt_mark = 2 when is_setup = 1
                row_slice.c_lt_mark = F::ONE;
                row_slice.lt_marker = [F::ZERO; READ_LIMBS];
                row_slice.lt_marker[READ_LIMBS - 1] = F::ONE;
                row_slice.c_lt_diff = modulus_limbs[READ_LIMBS - 1] - row_slice.c[READ_LIMBS - 1];
                row_slice.b_lt_diff = modulus_limbs[READ_LIMBS - 1] - row_slice.b[READ_LIMBS - 1];
            } else if row_slice.b[0] == F::from_canonical_u32(2) {
                // test the constraint that b[i] = N[i] for all i when prefix_sum is not 1 or
                // lt_marker_sum - is_setup
                row_slice.c_lt_mark = F::from_canonical_u8(2);
                row_slice.lt_marker = [F::ZERO; READ_LIMBS];
                row_slice.lt_marker[READ_LIMBS - 1] = F::from_canonical_u8(2);
                row_slice.c_lt_diff = modulus_limbs[READ_LIMBS - 1] - row_slice.c[READ_LIMBS - 1];
            } else if row_slice.b[0] == F::from_canonical_u32(3) {
                // test the constraint that sum_i lt_marker[i] = 2 when is_setup = 1
                row_slice.c_lt_mark = F::from_canonical_u8(2);
                row_slice.lt_marker = [F::ZERO; READ_LIMBS];
                row_slice.lt_marker[READ_LIMBS - 1] = F::from_canonical_u8(2);
                row_slice.lt_marker[0] = F::ONE;
                row_slice.b_lt_diff = modulus_limbs[0] - row_slice.b[0];
                row_slice.c_lt_diff = modulus_limbs[READ_LIMBS - 1] - row_slice.c[READ_LIMBS - 1];
            }
        };

        disable_debug_builder();
        let tester = tester
            .build()
            .load_and_prank_trace(chip, modify_trace)
            .load(bitwise_chip)
            .finalize();
        tester.simple_test_with_expected_error(expected_error);
    }

    #[test]
    fn negative_test_modular_is_equal_1x32() {
        let c_limbs = [F::ZERO; 32];
        let mut b_limbs = [F::ZERO; 32];
        b_limbs[0] = F::from_canonical_u32(1);
        run_negative_is_equal_test::<1, 32, 32>(
            secp256k1_coord_prime(),
            17,
            b_limbs,
            c_limbs,
            VerificationError::OodEvaluationMismatch,
        );

        b_limbs[0] = F::from_canonical_u32(2);
        run_negative_is_equal_test::<1, 32, 32>(
            secp256k1_coord_prime(),
            17,
            b_limbs,
            c_limbs,
            VerificationError::OodEvaluationMismatch,
        );

        b_limbs[0] = F::from_canonical_u32(3);
        run_negative_is_equal_test::<1, 32, 32>(
            secp256k1_coord_prime(),
            17,
            b_limbs,
            c_limbs,
            VerificationError::OodEvaluationMismatch,
        );
    }

    #[test]
    fn negative_test_modular_is_equal_3x16() {
        let c_limbs = [F::ZERO; 48];
        let mut b_limbs = [F::ZERO; 48];
        b_limbs[0] = F::from_canonical_u32(1);
        run_negative_is_equal_test::<3, 16, 48>(
            BLS12_381_MODULUS.clone(),
            17,
            b_limbs,
            c_limbs,
            VerificationError::OodEvaluationMismatch,
        );

        b_limbs[0] = F::from_canonical_u32(2);
        run_negative_is_equal_test::<3, 16, 48>(
            BLS12_381_MODULUS.clone(),
            17,
            b_limbs,
            c_limbs,
            VerificationError::OodEvaluationMismatch,
        );

        b_limbs[0] = F::from_canonical_u32(3);
        run_negative_is_equal_test::<3, 16, 48>(
            BLS12_381_MODULUS.clone(),
            17,
            b_limbs,
            c_limbs,
            VerificationError::OodEvaluationMismatch,
        );
    }
}
