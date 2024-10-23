use super::adapters::native_vec_heap_adapter::{NativeVecHeapAdapterAir, NativeVecHeapAdapterChip};
use crate::{
    arch::{VmAirWrapper, VmChipWrapper},
    intrinsics::{
        ecc::sw::{SwEcDoubleCoreAir, SwEcDoubleCoreChip},
        field_expression::{FieldExpressionCoreAir, FieldExpressionCoreChip},
    },
};

pub type KernelEcAddNeAir<const NUM_LIMBS: usize> =
    VmAirWrapper<NativeVecHeapAdapterAir<2, 2, 2, NUM_LIMBS, NUM_LIMBS>, FieldExpressionCoreAir>;
pub type KernelEcAddNeChip<F, const NUM_LIMBS: usize> = VmChipWrapper<
    F,
    NativeVecHeapAdapterChip<F, 2, 2, 2, NUM_LIMBS, NUM_LIMBS>,
    FieldExpressionCoreChip,
>;

pub type KernelEcDoubleAir<const NUM_LIMBS: usize> =
    VmAirWrapper<NativeVecHeapAdapterAir<1, 2, 2, NUM_LIMBS, NUM_LIMBS>, SwEcDoubleCoreAir>;
pub type KernelEcDoubleChip<F, const NUM_LIMBS: usize> = VmChipWrapper<
    F,
    NativeVecHeapAdapterChip<F, 1, 2, 2, NUM_LIMBS, NUM_LIMBS>,
    SwEcDoubleCoreChip,
>;
