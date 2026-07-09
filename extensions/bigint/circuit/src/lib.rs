#![cfg_attr(feature = "tco", allow(incomplete_features))]
#![cfg_attr(feature = "tco", feature(explicit_tail_calls))]
#![cfg_attr(feature = "tco", allow(internal_features))]
#![cfg_attr(feature = "tco", feature(core_intrinsics))]
use openvm_circuit::{
    self,
    arch::{
        InitFileGenerator, SystemConfig, VmAirWrapper, VmChipWrapper, BLOCK_FE_WIDTH,
        MEMORY_BLOCK_BYTES,
    },
    system::SystemExecutor,
};
use openvm_circuit_derive::{PreflightExecutor, VmConfig};
use openvm_riscv_adapters::{
    Rv64VecHeapAdapterAir, Rv64VecHeapAdapterExecutor, Rv64VecHeapAdapterFiller,
    Rv64VecHeapBranchU16AdapterAir, Rv64VecHeapBranchU16AdapterExecutor,
    Rv64VecHeapBranchU16AdapterFiller, Rv64VecHeapU16AdapterAir, Rv64VecHeapU16AdapterExecutor,
    Rv64VecHeapU16AdapterFiller, VecToFlatAluAdapterAir, VecToFlatAluAdapterExecutor,
    VecToFlatAluU16AdapterExecutor, VecToFlatBranchAdapterAir, VecToFlatBranchAdapterExecutor,
};
use openvm_riscv_circuit::{
    adapters::{RV64_BYTE_BITS, U16_BITS},
    AddSubCoreAir, AddSubExecutor, AddSubFiller, BitwiseLogicCoreAir, BitwiseLogicExecutor,
    BitwiseLogicFiller, BranchEqualCoreAir, BranchEqualExecutor, BranchEqualFiller,
    BranchLessThanCoreAir, BranchLessThanExecutor, BranchLessThanFiller, LessThanCoreAir,
    LessThanExecutor, LessThanFiller, MultiplicationCoreAir, MultiplicationExecutor,
    MultiplicationFiller, Rv64I, Rv64IExecutor, Rv64Io, Rv64IoExecutor, Rv64M, Rv64MExecutor,
    ShiftLogicalCoreAir, ShiftLogicalExecutor, ShiftLogicalFiller, ShiftRightArithmeticCoreAir,
    ShiftRightArithmeticExecutor, ShiftRightArithmeticFiller,
};
use serde::{Deserialize, Serialize};

mod extension;
pub use extension::*;

mod add_sub;
mod bitwise_logic;
mod branch_eq;
mod branch_lt;
pub(crate) mod common;
mod less_than;
#[cfg(feature = "rvr")]
pub mod log_native;
mod mult;
mod shift;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(all(test, feature = "rvr"))]
mod rvr_preflight_tests;
#[cfg(test)]
mod tests;

/// 256-bit heap integer stored as 32 bytes.
pub const INT256_NUM_U8_LIMBS: usize = 32;
/// Number of blocks for INT256 operations (INT256_NUM_U8_LIMBS / MEMORY_BLOCK_BYTES).
pub const INT256_NUM_MEMORY_BLOCKS: usize = INT256_NUM_U8_LIMBS / MEMORY_BLOCK_BYTES;
/// Number of u64 limbs in a 256-bit integer.
pub const INT256_NUM_U64_LIMBS: usize = INT256_NUM_U8_LIMBS / size_of::<u64>();
/// Number of u32 limbs in a 256-bit integer.
pub const INT256_NUM_U32_LIMBS: usize = INT256_NUM_U8_LIMBS / size_of::<u32>();
/// Number of u16 limbs in a 256-bit integer.
pub const INT256_NUM_U16_LIMBS: usize = INT256_NUM_U8_LIMBS / size_of::<u16>();
/// Number of source operand reads (rs1, rs2) for binary 256-bit instructions.
pub(crate) const NUM_READS: usize = 2;

/// Type alias for the ALU adapter AIR wrapper
type AluAdapterAir = VecToFlatAluAdapterAir<
    Rv64VecHeapAdapterAir<NUM_READS, INT256_NUM_MEMORY_BLOCKS, INT256_NUM_MEMORY_BLOCKS>,
    NUM_READS,
    INT256_NUM_MEMORY_BLOCKS,
    INT256_NUM_MEMORY_BLOCKS,
    MEMORY_BLOCK_BYTES,
    INT256_NUM_U8_LIMBS,
    INT256_NUM_U8_LIMBS,
>;

/// Type alias for the ALU adapter executor wrapper
pub type AluAdapterExecutor = VecToFlatAluAdapterExecutor<
    Rv64VecHeapAdapterExecutor<NUM_READS, INT256_NUM_MEMORY_BLOCKS, INT256_NUM_MEMORY_BLOCKS>,
    NUM_READS,
    INT256_NUM_MEMORY_BLOCKS,
    INT256_NUM_MEMORY_BLOCKS,
    MEMORY_BLOCK_BYTES,
    INT256_NUM_U8_LIMBS,
    INT256_NUM_U8_LIMBS,
>;

type AluU16AdapterAir = VecToFlatAluAdapterAir<
    Rv64VecHeapU16AdapterAir<NUM_READS, INT256_NUM_MEMORY_BLOCKS, INT256_NUM_MEMORY_BLOCKS>,
    NUM_READS,
    INT256_NUM_MEMORY_BLOCKS,
    INT256_NUM_MEMORY_BLOCKS,
    BLOCK_FE_WIDTH,
    INT256_NUM_U16_LIMBS,
    INT256_NUM_U16_LIMBS,
>;

pub type AluU16AdapterExecutor = VecToFlatAluU16AdapterExecutor<
    Rv64VecHeapU16AdapterExecutor<NUM_READS, INT256_NUM_MEMORY_BLOCKS, INT256_NUM_MEMORY_BLOCKS>,
    NUM_READS,
    INT256_NUM_MEMORY_BLOCKS,
    INT256_NUM_MEMORY_BLOCKS,
    BLOCK_FE_WIDTH,
    INT256_NUM_U16_LIMBS,
    INT256_NUM_U16_LIMBS,
>;

/// Type alias for the Branch adapter AIR wrapper
type BranchAdapterAir = VecToFlatBranchAdapterAir<
    Rv64VecHeapBranchU16AdapterAir<NUM_READS, INT256_NUM_MEMORY_BLOCKS>,
    NUM_READS,
    INT256_NUM_MEMORY_BLOCKS,
    BLOCK_FE_WIDTH,
    INT256_NUM_U16_LIMBS,
>;

/// Type alias for the Branch adapter executor wrapper
type BranchAdapterExecutor = VecToFlatBranchAdapterExecutor<
    Rv64VecHeapBranchU16AdapterExecutor<NUM_READS, INT256_NUM_MEMORY_BLOCKS>,
    NUM_READS,
    INT256_NUM_MEMORY_BLOCKS,
    BLOCK_FE_WIDTH,
    INT256_NUM_U16_LIMBS,
>;

/// AddSub256 — u16 limbs, range checker (shares the AluU16 adapter with LessThan256)
pub type Rv64AddSub256Air =
    VmAirWrapper<AluU16AdapterAir, AddSubCoreAir<INT256_NUM_U16_LIMBS, U16_BITS, true>>;
#[derive(Clone, PreflightExecutor)]
pub struct Rv64AddSub256Executor(
    AddSubExecutor<AluU16AdapterExecutor, INT256_NUM_U16_LIMBS, U16_BITS>,
);
pub type Rv64AddSub256Chip<F> = VmChipWrapper<
    F,
    AddSubFiller<
        Rv64VecHeapU16AdapterFiller<NUM_READS, INT256_NUM_MEMORY_BLOCKS, INT256_NUM_MEMORY_BLOCKS>,
        INT256_NUM_U16_LIMBS,
        U16_BITS,
        true,
    >,
>;

/// BitwiseLogic256 — byte limbs, bitwise lookup for XOR/OR/AND.
pub type Rv64BitwiseLogic256Air =
    VmAirWrapper<AluAdapterAir, BitwiseLogicCoreAir<INT256_NUM_U8_LIMBS, RV64_BYTE_BITS>>;
#[derive(Clone, PreflightExecutor)]
pub struct Rv64BitwiseLogic256Executor(
    BitwiseLogicExecutor<AluAdapterExecutor, INT256_NUM_U8_LIMBS, RV64_BYTE_BITS>,
);
pub type Rv64BitwiseLogic256Chip<F> = VmChipWrapper<
    F,
    BitwiseLogicFiller<
        Rv64VecHeapAdapterFiller<NUM_READS, INT256_NUM_MEMORY_BLOCKS, INT256_NUM_MEMORY_BLOCKS>,
        INT256_NUM_U8_LIMBS,
        RV64_BYTE_BITS,
    >,
>;

/// LessThan256
pub type Rv64LessThan256Air =
    VmAirWrapper<AluU16AdapterAir, LessThanCoreAir<INT256_NUM_U16_LIMBS, U16_BITS>>;
#[derive(Clone, PreflightExecutor)]
pub struct Rv64LessThan256Executor(
    LessThanExecutor<AluU16AdapterExecutor, INT256_NUM_U16_LIMBS, U16_BITS>,
);
pub type Rv64LessThan256Chip<F> = VmChipWrapper<
    F,
    LessThanFiller<
        Rv64VecHeapU16AdapterFiller<NUM_READS, INT256_NUM_MEMORY_BLOCKS, INT256_NUM_MEMORY_BLOCKS>,
        INT256_NUM_U16_LIMBS,
        U16_BITS,
    >,
>;

/// Multiplication256
pub type Rv64Multiplication256Air =
    VmAirWrapper<AluAdapterAir, MultiplicationCoreAir<INT256_NUM_U8_LIMBS, RV64_BYTE_BITS>>;
#[derive(Clone, PreflightExecutor)]
pub struct Rv64Multiplication256Executor(
    MultiplicationExecutor<AluAdapterExecutor, INT256_NUM_U8_LIMBS, RV64_BYTE_BITS>,
);
pub type Rv64Multiplication256Chip<F> = VmChipWrapper<
    F,
    MultiplicationFiller<
        Rv64VecHeapAdapterFiller<NUM_READS, INT256_NUM_MEMORY_BLOCKS, INT256_NUM_MEMORY_BLOCKS>,
        INT256_NUM_U8_LIMBS,
        RV64_BYTE_BITS,
    >,
>;

/// Shift256 — SLL/SRL/SRA all use u16 limbs (AluU16 adapter).
pub type Rv64ShiftLogical256Air =
    VmAirWrapper<AluU16AdapterAir, ShiftLogicalCoreAir<INT256_NUM_U16_LIMBS, U16_BITS>>;
pub type Rv64ShiftRightArithmetic256Air =
    VmAirWrapper<AluU16AdapterAir, ShiftRightArithmeticCoreAir<INT256_NUM_U16_LIMBS, U16_BITS>>;
#[derive(Clone, PreflightExecutor)]
pub struct Rv64ShiftLogical256Executor(
    ShiftLogicalExecutor<AluU16AdapterExecutor, INT256_NUM_U16_LIMBS, U16_BITS>,
);
#[derive(Clone, PreflightExecutor)]
pub struct Rv64ShiftRightArithmetic256Executor(
    ShiftRightArithmeticExecutor<AluU16AdapterExecutor, INT256_NUM_U16_LIMBS, U16_BITS>,
);
pub type Rv64ShiftLogical256Chip<F> = VmChipWrapper<
    F,
    ShiftLogicalFiller<
        Rv64VecHeapU16AdapterFiller<NUM_READS, INT256_NUM_MEMORY_BLOCKS, INT256_NUM_MEMORY_BLOCKS>,
        INT256_NUM_U16_LIMBS,
        U16_BITS,
    >,
>;
pub type Rv64ShiftRightArithmetic256Chip<F> = VmChipWrapper<
    F,
    ShiftRightArithmeticFiller<
        Rv64VecHeapU16AdapterFiller<NUM_READS, INT256_NUM_MEMORY_BLOCKS, INT256_NUM_MEMORY_BLOCKS>,
        INT256_NUM_U16_LIMBS,
        U16_BITS,
    >,
>;

/// BranchEqual256
pub type Rv64BranchEqual256Air =
    VmAirWrapper<BranchAdapterAir, BranchEqualCoreAir<INT256_NUM_U16_LIMBS>>;
#[derive(Clone, PreflightExecutor)]
pub struct Rv64BranchEqual256Executor(
    BranchEqualExecutor<BranchAdapterExecutor, INT256_NUM_U16_LIMBS>,
);
pub type Rv64BranchEqual256Chip<F> = VmChipWrapper<
    F,
    BranchEqualFiller<
        Rv64VecHeapBranchU16AdapterFiller<NUM_READS, INT256_NUM_MEMORY_BLOCKS>,
        INT256_NUM_U16_LIMBS,
    >,
>;

/// BranchLessThan256
pub type Rv64BranchLessThan256Air =
    VmAirWrapper<BranchAdapterAir, BranchLessThanCoreAir<INT256_NUM_U16_LIMBS, U16_BITS>>;
#[derive(Clone, PreflightExecutor)]
pub struct Rv64BranchLessThan256Executor(
    BranchLessThanExecutor<BranchAdapterExecutor, INT256_NUM_U16_LIMBS, U16_BITS>,
);
pub type Rv64BranchLessThan256Chip<F> = VmChipWrapper<
    F,
    BranchLessThanFiller<
        Rv64VecHeapBranchU16AdapterFiller<NUM_READS, INT256_NUM_MEMORY_BLOCKS>,
        INT256_NUM_U16_LIMBS,
        U16_BITS,
    >,
>;

#[derive(Clone, Debug, VmConfig, derive_new::new, Serialize, Deserialize)]
pub struct Int256Rv64Config {
    #[config(executor = "SystemExecutor")]
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
