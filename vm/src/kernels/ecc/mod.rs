use afs_derive::{Chip, ChipUsageGetter, InstructionExecutor};
use afs_primitives::bigint::utils::secp256k1_coord_prime;
use p3_field::PrimeField32;

use super::adapters::native_vec_heap_adapter::NativeVecHeapAdapterChip;
use crate::{
    arch::{instructions::EccOpcode, VmChipWrapper},
    intrinsics::{
        ecc::sw::{ec_add_ne_expr, ec_double_expr},
        field_expression::FieldExpressionCoreChip,
    },
    system::memory::MemoryControllerRef,
};

#[derive(Chip, ChipUsageGetter, InstructionExecutor)]
pub struct KernelEcAddNeChip<F: PrimeField32, const NUM_LIMBS: usize>(
    VmChipWrapper<
        F,
        NativeVecHeapAdapterChip<F, 2, 2, 2, NUM_LIMBS, NUM_LIMBS>,
        FieldExpressionCoreChip,
    >,
);

impl<F: PrimeField32, const NUM_LIMBS: usize> KernelEcAddNeChip<F, NUM_LIMBS> {
    pub fn new(
        adapter: NativeVecHeapAdapterChip<F, 2, 2, 2, NUM_LIMBS, NUM_LIMBS>,
        memory_controller: MemoryControllerRef<F>,
        limb_bits: usize,
        offset: usize,
    ) -> Self {
        let expr = ec_add_ne_expr(
            secp256k1_coord_prime(),
            NUM_LIMBS,
            limb_bits,
            memory_controller.borrow().range_checker.bus(),
        );
        let core = FieldExpressionCoreChip::new(
            expr,
            offset,
            vec![EccOpcode::EC_ADD_NE as usize],
            memory_controller.borrow().range_checker.clone(),
            "EcAddNe",
        );
        Self(VmChipWrapper::new(adapter, core, memory_controller))
    }
}

#[derive(Chip, ChipUsageGetter, InstructionExecutor)]
pub struct KernelEcDoubleChip<F: PrimeField32, const NUM_LIMBS: usize>(
    VmChipWrapper<
        F,
        NativeVecHeapAdapterChip<F, 1, 2, 2, NUM_LIMBS, NUM_LIMBS>,
        FieldExpressionCoreChip,
    >,
);

impl<F: PrimeField32, const NUM_LIMBS: usize> KernelEcDoubleChip<F, NUM_LIMBS> {
    pub fn new(
        adapter: NativeVecHeapAdapterChip<F, 1, 2, 2, NUM_LIMBS, NUM_LIMBS>,
        memory_controller: MemoryControllerRef<F>,
        limb_bits: usize,
        offset: usize,
    ) -> Self {
        let expr = ec_double_expr(
            secp256k1_coord_prime(),
            NUM_LIMBS,
            limb_bits,
            memory_controller.borrow().range_checker.bus(),
        );
        let core = FieldExpressionCoreChip::new(
            expr,
            offset,
            vec![EccOpcode::EC_DOUBLE as usize],
            memory_controller.borrow().range_checker.clone(),
            "EcDouble",
        );
        Self(VmChipWrapper::new(adapter, core, memory_controller))
    }
}
