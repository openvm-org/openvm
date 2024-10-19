use crate::{
    arch::{VmAirWrapper, VmChipWrapper},
    intrinsics::modular::{
        ModularAddSubCoreAir, ModularAddSubCoreChip, ModularMulDivCoreAir, ModularMulDivCoreChip,
    },
    kernels::adapters::native_vec_heap_adapter::{
        NativeVecHeapAdapterAir, NativeVecHeapAdapterChip,
    },
};

pub type KernelModularAddSubAir<const NUM_LIMBS: usize> =
    VmAirWrapper<NativeVecHeapAdapterAir<1, 1, NUM_LIMBS, NUM_LIMBS>, ModularAddSubCoreAir>;
pub type KernelModularAddSubChip<F, const NUM_LIMBS: usize> = VmChipWrapper<
    F,
    NativeVecHeapAdapterChip<F, 1, 1, NUM_LIMBS, NUM_LIMBS>,
    ModularAddSubCoreChip,
>;
pub type KernelModularMulDivAir<const NUM_LIMBS: usize> =
    VmAirWrapper<NativeVecHeapAdapterAir<1, 1, NUM_LIMBS, NUM_LIMBS>, ModularMulDivCoreAir>;
pub type KernelModularMulDivChip<F, const NUM_LIMBS: usize> = VmChipWrapper<
    F,
    NativeVecHeapAdapterChip<F, 1, 1, NUM_LIMBS, NUM_LIMBS>,
    ModularMulDivCoreChip,
>;
