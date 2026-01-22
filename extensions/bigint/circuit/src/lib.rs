#![cfg_attr(feature = "tco", allow(incomplete_features))]
#![cfg_attr(feature = "tco", feature(explicit_tail_calls))]
#![cfg_attr(feature = "tco", allow(internal_features))]
#![cfg_attr(feature = "tco", feature(core_intrinsics))]
use openvm_circuit::{
    self,
    arch::{InitFileGenerator, SystemConfig, VmAirWrapper, VmChipWrapper},
    system::SystemExecutor,
};
use openvm_circuit_derive::{PreflightExecutor, VmConfig};
use openvm_rv32_adapters::{
    Rv32VecHeapAdapterAir, Rv32VecHeapAdapterExecutor, Rv32VecHeapAdapterFiller,
    Rv32VecHeapBranchAdapterAir, Rv32VecHeapBranchAdapterExecutor, Rv32VecHeapBranchAdapterFiller,
    VecToFlatAluAdapterAir, VecToFlatAluAdapterExecutor, VecToFlatBranchAdapterAir,
    VecToFlatBranchAdapterExecutor,
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

pub const CONST_BLOCK_SIZE: usize = 4;

// Constants for the VecToFlat adapter wrappers
const BLOCKS_PER_OP: usize = INT256_NUM_LIMBS / CONST_BLOCK_SIZE;

/// Type alias for the ALU adapter AIR wrapper
type AluAdapterAir = VecToFlatAluAdapterAir<
    Rv32VecHeapAdapterAir<2, BLOCKS_PER_OP, BLOCKS_PER_OP, CONST_BLOCK_SIZE, CONST_BLOCK_SIZE>,
    2,
    BLOCKS_PER_OP,
    BLOCKS_PER_OP,
    CONST_BLOCK_SIZE,
    INT256_NUM_LIMBS,
    INT256_NUM_LIMBS,
>;

/// Type alias for the ALU adapter executor wrapper
type AluAdapterExecutor = VecToFlatAluAdapterExecutor<
    Rv32VecHeapAdapterExecutor<2, BLOCKS_PER_OP, BLOCKS_PER_OP, CONST_BLOCK_SIZE, CONST_BLOCK_SIZE>,
    2,
    BLOCKS_PER_OP,
    BLOCKS_PER_OP,
    CONST_BLOCK_SIZE,
    INT256_NUM_LIMBS,
    INT256_NUM_LIMBS,
>;

/// Type alias for the Branch adapter AIR wrapper
type BranchAdapterAir = VecToFlatBranchAdapterAir<
    Rv32VecHeapBranchAdapterAir<2, BLOCKS_PER_OP, CONST_BLOCK_SIZE>,
    2,
    BLOCKS_PER_OP,
    CONST_BLOCK_SIZE,
    INT256_NUM_LIMBS,
>;

/// Type alias for the Branch adapter executor wrapper
type BranchAdapterExecutor = VecToFlatBranchAdapterExecutor<
    Rv32VecHeapBranchAdapterExecutor<2, BLOCKS_PER_OP, CONST_BLOCK_SIZE>,
    2,
    BLOCKS_PER_OP,
    CONST_BLOCK_SIZE,
    INT256_NUM_LIMBS,
>;

/// BaseAlu256
pub type Rv32BaseAlu256Air =
    VmAirWrapper<AluAdapterAir, BaseAluCoreAir<INT256_NUM_LIMBS, RV32_CELL_BITS>>;
#[derive(Clone, PreflightExecutor)]
pub struct Rv32BaseAlu256Executor(BaseAluExecutor<AluAdapterExecutor, INT256_NUM_LIMBS, RV32_CELL_BITS>);
pub type Rv32BaseAlu256Chip<F> = VmChipWrapper<
    F,
    BaseAluFiller<
        Rv32VecHeapAdapterFiller<
            2,
            BLOCKS_PER_OP,
            BLOCKS_PER_OP,
            CONST_BLOCK_SIZE,
            CONST_BLOCK_SIZE,
        >,
        INT256_NUM_LIMBS,
        RV32_CELL_BITS,
    >,
>;

/// LessThan256
pub type Rv32LessThan256Air =
    VmAirWrapper<AluAdapterAir, LessThanCoreAir<INT256_NUM_LIMBS, RV32_CELL_BITS>>;
#[derive(Clone, PreflightExecutor)]
pub struct Rv32LessThan256Executor(
    LessThanExecutor<AluAdapterExecutor, INT256_NUM_LIMBS, RV32_CELL_BITS>,
);
pub type Rv32LessThan256Chip<F> = VmChipWrapper<
    F,
    LessThanFiller<
        Rv32VecHeapAdapterFiller<
            2,
            BLOCKS_PER_OP,
            BLOCKS_PER_OP,
            CONST_BLOCK_SIZE,
            CONST_BLOCK_SIZE,
        >,
        INT256_NUM_LIMBS,
        RV32_CELL_BITS,
    >,
>;

/// Multiplication256
pub type Rv32Multiplication256Air =
    VmAirWrapper<AluAdapterAir, MultiplicationCoreAir<INT256_NUM_LIMBS, RV32_CELL_BITS>>;
#[derive(Clone, PreflightExecutor)]
pub struct Rv32Multiplication256Executor(
    MultiplicationExecutor<AluAdapterExecutor, INT256_NUM_LIMBS, RV32_CELL_BITS>,
);
pub type Rv32Multiplication256Chip<F> = VmChipWrapper<
    F,
    MultiplicationFiller<
        Rv32VecHeapAdapterFiller<
            2,
            BLOCKS_PER_OP,
            BLOCKS_PER_OP,
            CONST_BLOCK_SIZE,
            CONST_BLOCK_SIZE,
        >,
        INT256_NUM_LIMBS,
        RV32_CELL_BITS,
    >,
>;

/// Shift256
pub type Rv32Shift256Air =
    VmAirWrapper<AluAdapterAir, ShiftCoreAir<INT256_NUM_LIMBS, RV32_CELL_BITS>>;
#[derive(Clone, PreflightExecutor)]
pub struct Rv32Shift256Executor(ShiftExecutor<AluAdapterExecutor, INT256_NUM_LIMBS, RV32_CELL_BITS>);
pub type Rv32Shift256Chip<F> = VmChipWrapper<
    F,
    ShiftFiller<
        Rv32VecHeapAdapterFiller<
            2,
            BLOCKS_PER_OP,
            BLOCKS_PER_OP,
            CONST_BLOCK_SIZE,
            CONST_BLOCK_SIZE,
        >,
        INT256_NUM_LIMBS,
        RV32_CELL_BITS,
    >,
>;

/// BranchEqual256
pub type Rv32BranchEqual256Air = VmAirWrapper<BranchAdapterAir, BranchEqualCoreAir<INT256_NUM_LIMBS>>;
#[derive(Clone, PreflightExecutor)]
pub struct Rv32BranchEqual256Executor(
    BranchEqualExecutor<BranchAdapterExecutor, INT256_NUM_LIMBS>,
);
pub type Rv32BranchEqual256Chip<F> = VmChipWrapper<
    F,
    BranchEqualFiller<
        Rv32VecHeapBranchAdapterFiller<2, BLOCKS_PER_OP, CONST_BLOCK_SIZE>,
        INT256_NUM_LIMBS,
    >,
>;

/// BranchLessThan256
pub type Rv32BranchLessThan256Air =
    VmAirWrapper<BranchAdapterAir, BranchLessThanCoreAir<INT256_NUM_LIMBS, RV32_CELL_BITS>>;
#[derive(Clone, PreflightExecutor)]
pub struct Rv32BranchLessThan256Executor(
    BranchLessThanExecutor<BranchAdapterExecutor, INT256_NUM_LIMBS, RV32_CELL_BITS>,
);
pub type Rv32BranchLessThan256Chip<F> = VmChipWrapper<
    F,
    BranchLessThanFiller<
        Rv32VecHeapBranchAdapterFiller<2, BLOCKS_PER_OP, CONST_BLOCK_SIZE>,
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
