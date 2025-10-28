#![cfg_attr(feature = "tco", allow(incomplete_features))]
#![cfg_attr(feature = "tco", feature(explicit_tail_calls))]
#![cfg_attr(feature = "tco", feature(core_intrinsics))]
use openvm_circuit::{
    self,
    arch::{InitFileGenerator, SystemConfig, VmAirWrapper, VmChipWrapper},
    system::SystemExecutor,
};
use openvm_circuit_derive::{PreflightExecutor, VmConfig};
use openvm_rv32_adapters::{
    Rv32HeapAdapterAir, Rv32HeapAdapterExecutor, Rv32HeapAdapterFiller, Rv32HeapBranchAdapterAir,
    Rv32HeapBranchAdapterExecutor, Rv32HeapBranchAdapterFiller,
};
use openvm_rv32im_circuit::{
    adapters::{INT256_NUM_LIMBS, RV32_CELL_BITS},
    BaseAluCoreAir, BaseAluExecutor, BaseAluFiller, BranchEqualCoreAir, BranchEqualExecutor,
    BranchEqualFiller, BranchLessThanCoreAir, BranchLessThanExecutor, BranchLessThanFiller,
    LessThanCoreAir, LessThanExecutor, LessThanFiller, MultiplicationCoreAir,
    MultiplicationExecutor, MultiplicationFiller, Rv32I, Rv32IExecutor, Rv32Io, Rv32IoExecutor,
    Rv32M, Rv32MExecutor, ShiftCoreAir, ShiftExecutor, ShiftFiller,
};
use serde::{Deserialize, Serialize};

mod extension;
pub use extension::*;

mod base_alu;
mod branch_eq;
mod branch_lt;
pub(crate) mod common;
mod less_than;
mod mult;
mod shift;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;

/// BaseAlu256
pub type Rv32BaseAlu256Air = VmAirWrapper<
    Rv32HeapAdapterAir<2, INT256_NUM_LIMBS, INT256_NUM_LIMBS>,
    BaseAluCoreAir<INT256_NUM_LIMBS, RV32_CELL_BITS>,
>;
#[derive(Clone, PreflightExecutor)]
pub struct Rv32BaseAlu256Executor(
    BaseAluExecutor<
        Rv32HeapAdapterExecutor<2, INT256_NUM_LIMBS, INT256_NUM_LIMBS>,
        INT256_NUM_LIMBS,
        RV32_CELL_BITS,
    >,
);
pub type Rv32BaseAlu256Chip<F> = VmChipWrapper<
    F,
    BaseAluFiller<
        Rv32HeapAdapterFiller<2, INT256_NUM_LIMBS, INT256_NUM_LIMBS>,
        INT256_NUM_LIMBS,
        RV32_CELL_BITS,
    >,
>;

/// LessThan256
pub type Rv32LessThan256Air = VmAirWrapper<
    Rv32HeapAdapterAir<2, INT256_NUM_LIMBS, INT256_NUM_LIMBS>,
    LessThanCoreAir<INT256_NUM_LIMBS, RV32_CELL_BITS>,
>;
#[derive(Clone, PreflightExecutor)]
pub struct Rv32LessThan256Executor(
    LessThanExecutor<
        Rv32HeapAdapterExecutor<2, INT256_NUM_LIMBS, INT256_NUM_LIMBS>,
        INT256_NUM_LIMBS,
        RV32_CELL_BITS,
    >,
);
pub type Rv32LessThan256Chip<F> = VmChipWrapper<
    F,
    LessThanFiller<
        Rv32HeapAdapterFiller<2, INT256_NUM_LIMBS, INT256_NUM_LIMBS>,
        INT256_NUM_LIMBS,
        RV32_CELL_BITS,
    >,
>;

/// Multiplication256
pub type Rv32Multiplication256Air = VmAirWrapper<
    Rv32HeapAdapterAir<2, INT256_NUM_LIMBS, INT256_NUM_LIMBS>,
    MultiplicationCoreAir<INT256_NUM_LIMBS, RV32_CELL_BITS>,
>;
#[derive(Clone, PreflightExecutor)]
pub struct Rv32Multiplication256Executor(
    MultiplicationExecutor<
        Rv32HeapAdapterExecutor<2, INT256_NUM_LIMBS, INT256_NUM_LIMBS>,
        INT256_NUM_LIMBS,
        RV32_CELL_BITS,
    >,
);
pub type Rv32Multiplication256Chip<F> = VmChipWrapper<
    F,
    MultiplicationFiller<
        Rv32HeapAdapterFiller<2, INT256_NUM_LIMBS, INT256_NUM_LIMBS>,
        INT256_NUM_LIMBS,
        RV32_CELL_BITS,
    >,
>;

/// Shift256
pub type Rv32Shift256Air = VmAirWrapper<
    Rv32HeapAdapterAir<2, INT256_NUM_LIMBS, INT256_NUM_LIMBS>,
    ShiftCoreAir<INT256_NUM_LIMBS, RV32_CELL_BITS>,
>;
#[derive(Clone, PreflightExecutor)]
pub struct Rv32Shift256Executor(
    ShiftExecutor<
        Rv32HeapAdapterExecutor<2, INT256_NUM_LIMBS, INT256_NUM_LIMBS>,
        INT256_NUM_LIMBS,
        RV32_CELL_BITS,
    >,
);
pub type Rv32Shift256Chip<F> = VmChipWrapper<
    F,
    ShiftFiller<
        Rv32HeapAdapterFiller<2, INT256_NUM_LIMBS, INT256_NUM_LIMBS>,
        INT256_NUM_LIMBS,
        RV32_CELL_BITS,
    >,
>;

/// BranchEqual256
pub type Rv32BranchEqual256Air = VmAirWrapper<
    Rv32HeapBranchAdapterAir<2, INT256_NUM_LIMBS>,
    BranchEqualCoreAir<INT256_NUM_LIMBS>,
>;
#[derive(Clone, PreflightExecutor)]
pub struct Rv32BranchEqual256Executor(
    BranchEqualExecutor<Rv32HeapBranchAdapterExecutor<2, INT256_NUM_LIMBS>, INT256_NUM_LIMBS>,
);
pub type Rv32BranchEqual256Chip<F> = VmChipWrapper<
    F,
    BranchEqualFiller<Rv32HeapBranchAdapterFiller<2, INT256_NUM_LIMBS>, INT256_NUM_LIMBS>,
>;

/// BranchLessThan256
pub type Rv32BranchLessThan256Air = VmAirWrapper<
    Rv32HeapBranchAdapterAir<2, INT256_NUM_LIMBS>,
    BranchLessThanCoreAir<INT256_NUM_LIMBS, RV32_CELL_BITS>,
>;
#[derive(Clone, PreflightExecutor)]
pub struct Rv32BranchLessThan256Executor(
    BranchLessThanExecutor<
        Rv32HeapBranchAdapterExecutor<2, INT256_NUM_LIMBS>,
        INT256_NUM_LIMBS,
        RV32_CELL_BITS,
    >,
);
pub type Rv32BranchLessThan256Chip<F> = VmChipWrapper<
    F,
    BranchLessThanFiller<
        Rv32HeapBranchAdapterFiller<2, INT256_NUM_LIMBS>,
        INT256_NUM_LIMBS,
        RV32_CELL_BITS,
    >,
>;

#[derive(Clone, Debug, VmConfig, derive_new::new, Serialize, Deserialize)]
pub struct Int256Rv32Config {
    #[config(executor = "SystemExecutor<F>")]
    pub system: SystemConfig,
    #[extension]
    pub rv32i: Rv32I,
    #[extension]
    pub rv32m: Rv32M,
    #[extension]
    pub io: Rv32Io,
    #[extension]
    pub bigint: Int256,
}

// Default implementation uses no init file
impl InitFileGenerator for Int256Rv32Config {}

impl Default for Int256Rv32Config {
    fn default() -> Self {
        Self {
            system: SystemConfig::default(),
            rv32i: Rv32I,
            rv32m: Rv32M::default(),
            io: Rv32Io,
            bigint: Int256::default(),
        }
    }
}
