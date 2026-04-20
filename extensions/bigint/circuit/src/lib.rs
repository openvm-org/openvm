#![cfg_attr(feature = "tco", allow(incomplete_features))]
#![cfg_attr(feature = "tco", feature(explicit_tail_calls))]
#![cfg_attr(feature = "tco", allow(internal_features))]
#![cfg_attr(feature = "tco", feature(core_intrinsics))]
use openvm_circuit::{
    self,
    arch::{InitFileGenerator, SystemConfig, VmAirWrapper, VmChipWrapper, DEFAULT_BLOCK_SIZE},
    system::SystemExecutor,
};
use openvm_circuit_derive::{PreflightExecutor, VmConfig};
use openvm_riscv_adapters::{
    Rv64VecHeapAdapterAir, Rv64VecHeapAdapterExecutor, Rv64VecHeapAdapterFiller,
    Rv64VecHeapBranchAdapterAir, Rv64VecHeapBranchAdapterExecutor, Rv64VecHeapBranchAdapterFiller,
    VecToFlatAluAdapterAir, VecToFlatAluAdapterExecutor, VecToFlatBranchAdapterAir,
    VecToFlatBranchAdapterExecutor,
};
use openvm_riscv_circuit::{
    adapters::{INT256_NUM_LIMBS, RV64_CELL_BITS},
    BaseAluCoreAir, BaseAluExecutor, BaseAluFiller, BranchEqualCoreAir, BranchEqualExecutor,
    BranchEqualFiller, BranchLessThanCoreAir, BranchLessThanExecutor, BranchLessThanFiller,
    LessThanCoreAir, LessThanExecutor, LessThanFiller, MultiplicationCoreAir,
    MultiplicationExecutor, MultiplicationFiller, Rv64I, Rv64IExecutor, Rv64Io, Rv64IoExecutor,
    Rv64M, Rv64MExecutor, ShiftCoreAir, ShiftExecutor, ShiftFiller,
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

/// Number of blocks for INT256 operations (INT256_NUM_LIMBS / DEFAULT_BLOCK_SIZE)
pub const INT256_NUM_BLOCKS: usize = INT256_NUM_LIMBS / DEFAULT_BLOCK_SIZE;

/// Type alias for the ALU adapter AIR wrapper
type AluAdapterAir = VecToFlatAluAdapterAir<
    Rv64VecHeapAdapterAir<
        2,
        INT256_NUM_BLOCKS,
        INT256_NUM_BLOCKS,
        DEFAULT_BLOCK_SIZE,
        DEFAULT_BLOCK_SIZE,
    >,
    2,
    INT256_NUM_BLOCKS,
    INT256_NUM_BLOCKS,
    DEFAULT_BLOCK_SIZE,
    INT256_NUM_LIMBS,
    INT256_NUM_LIMBS,
>;

/// Type alias for the ALU adapter executor wrapper
type AluAdapterExecutor = VecToFlatAluAdapterExecutor<
    Rv64VecHeapAdapterExecutor<
        2,
        INT256_NUM_BLOCKS,
        INT256_NUM_BLOCKS,
        DEFAULT_BLOCK_SIZE,
        DEFAULT_BLOCK_SIZE,
    >,
    2,
    INT256_NUM_BLOCKS,
    INT256_NUM_BLOCKS,
    DEFAULT_BLOCK_SIZE,
    INT256_NUM_LIMBS,
    INT256_NUM_LIMBS,
>;

/// Type alias for the Branch adapter AIR wrapper
type BranchAdapterAir = VecToFlatBranchAdapterAir<
    Rv64VecHeapBranchAdapterAir<2, INT256_NUM_BLOCKS, DEFAULT_BLOCK_SIZE>,
    2,
    INT256_NUM_BLOCKS,
    DEFAULT_BLOCK_SIZE,
    INT256_NUM_LIMBS,
>;

/// Type alias for the Branch adapter executor wrapper
type BranchAdapterExecutor = VecToFlatBranchAdapterExecutor<
    Rv64VecHeapBranchAdapterExecutor<2, INT256_NUM_BLOCKS, DEFAULT_BLOCK_SIZE>,
    2,
    INT256_NUM_BLOCKS,
    DEFAULT_BLOCK_SIZE,
    INT256_NUM_LIMBS,
>;

/// BaseAlu256
pub type Rv64BaseAlu256Air =
    VmAirWrapper<AluAdapterAir, BaseAluCoreAir<INT256_NUM_LIMBS, RV64_CELL_BITS>>;
#[derive(Clone, PreflightExecutor)]
pub struct Rv64BaseAlu256Executor(
    BaseAluExecutor<AluAdapterExecutor, INT256_NUM_LIMBS, RV64_CELL_BITS>,
);
pub type Rv64BaseAlu256Chip<F> = VmChipWrapper<
    F,
    BaseAluFiller<
        Rv64VecHeapAdapterFiller<
            2,
            INT256_NUM_BLOCKS,
            INT256_NUM_BLOCKS,
            DEFAULT_BLOCK_SIZE,
            DEFAULT_BLOCK_SIZE,
        >,
        INT256_NUM_LIMBS,
        RV64_CELL_BITS,
    >,
>;

/// LessThan256
pub type Rv64LessThan256Air =
    VmAirWrapper<AluAdapterAir, LessThanCoreAir<INT256_NUM_LIMBS, RV64_CELL_BITS>>;
#[derive(Clone, PreflightExecutor)]
pub struct Rv64LessThan256Executor(
    LessThanExecutor<AluAdapterExecutor, INT256_NUM_LIMBS, RV64_CELL_BITS>,
);
pub type Rv64LessThan256Chip<F> = VmChipWrapper<
    F,
    LessThanFiller<
        Rv64VecHeapAdapterFiller<
            2,
            INT256_NUM_BLOCKS,
            INT256_NUM_BLOCKS,
            DEFAULT_BLOCK_SIZE,
            DEFAULT_BLOCK_SIZE,
        >,
        INT256_NUM_LIMBS,
        RV64_CELL_BITS,
    >,
>;

/// Multiplication256
pub type Rv64Multiplication256Air =
    VmAirWrapper<AluAdapterAir, MultiplicationCoreAir<INT256_NUM_LIMBS, RV64_CELL_BITS>>;
#[derive(Clone, PreflightExecutor)]
pub struct Rv64Multiplication256Executor(
    MultiplicationExecutor<AluAdapterExecutor, INT256_NUM_LIMBS, RV64_CELL_BITS>,
);
pub type Rv64Multiplication256Chip<F> = VmChipWrapper<
    F,
    MultiplicationFiller<
        Rv64VecHeapAdapterFiller<
            2,
            INT256_NUM_BLOCKS,
            INT256_NUM_BLOCKS,
            DEFAULT_BLOCK_SIZE,
            DEFAULT_BLOCK_SIZE,
        >,
        INT256_NUM_LIMBS,
        RV64_CELL_BITS,
    >,
>;

/// Shift256
pub type Rv64Shift256Air =
    VmAirWrapper<AluAdapterAir, ShiftCoreAir<INT256_NUM_LIMBS, RV64_CELL_BITS>>;
#[derive(Clone, PreflightExecutor)]
pub struct Rv64Shift256Executor(
    ShiftExecutor<AluAdapterExecutor, INT256_NUM_LIMBS, RV64_CELL_BITS>,
);
pub type Rv64Shift256Chip<F> = VmChipWrapper<
    F,
    ShiftFiller<
        Rv64VecHeapAdapterFiller<
            2,
            INT256_NUM_BLOCKS,
            INT256_NUM_BLOCKS,
            DEFAULT_BLOCK_SIZE,
            DEFAULT_BLOCK_SIZE,
        >,
        INT256_NUM_LIMBS,
        RV64_CELL_BITS,
    >,
>;

/// BranchEqual256
pub type Rv64BranchEqual256Air =
    VmAirWrapper<BranchAdapterAir, BranchEqualCoreAir<INT256_NUM_LIMBS>>;
#[derive(Clone, PreflightExecutor)]
pub struct Rv64BranchEqual256Executor(BranchEqualExecutor<BranchAdapterExecutor, INT256_NUM_LIMBS>);
pub type Rv64BranchEqual256Chip<F> = VmChipWrapper<
    F,
    BranchEqualFiller<
        Rv64VecHeapBranchAdapterFiller<2, INT256_NUM_BLOCKS, DEFAULT_BLOCK_SIZE>,
        INT256_NUM_LIMBS,
    >,
>;

/// BranchLessThan256
pub type Rv64BranchLessThan256Air =
    VmAirWrapper<BranchAdapterAir, BranchLessThanCoreAir<INT256_NUM_LIMBS, RV64_CELL_BITS>>;
#[derive(Clone, PreflightExecutor)]
pub struct Rv64BranchLessThan256Executor(
    BranchLessThanExecutor<BranchAdapterExecutor, INT256_NUM_LIMBS, RV64_CELL_BITS>,
);
pub type Rv64BranchLessThan256Chip<F> = VmChipWrapper<
    F,
    BranchLessThanFiller<
        Rv64VecHeapBranchAdapterFiller<2, INT256_NUM_BLOCKS, DEFAULT_BLOCK_SIZE>,
        INT256_NUM_LIMBS,
        RV64_CELL_BITS,
    >,
>;

#[derive(Clone, Debug, VmConfig, derive_new::new, Serialize, Deserialize)]
pub struct Int256Rv64Config {
    #[config(executor = "SystemExecutor<F>")]
    pub system: SystemConfig,
    #[extension]
    pub rv64i: Rv64I,
    #[extension]
    pub rv64m: Rv64M,
    #[extension]
    pub io: Rv64Io,
    #[extension]
    pub bigint: Int256,
}

// Default implementation uses no init file
impl InitFileGenerator for Int256Rv64Config {}

impl Default for Int256Rv64Config {
    fn default() -> Self {
        Self {
            system: SystemConfig::default(),
            rv64i: Rv64I,
            rv64m: Rv64M::default(),
            io: Rv64Io,
            bigint: Int256::default(),
        }
    }
}
