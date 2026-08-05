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
use openvm_circuit_derive::VmConfig;
use openvm_riscv_adapters::{
    VecHeapAdapterAir, VecHeapBranchU16AdapterAir, VecHeapU16AdapterAir, VecToFlatAluAdapterAir,
    VecToFlatBranchAdapterAir,
};
use openvm_riscv_circuit::{
    adapters::{BYTE_BITS, U16_BITS},
    AddSubCoreAir, AddSubFiller, BitwiseLogicCoreAir, BitwiseLogicFiller, BranchEqualCoreAir,
    BranchEqualFiller, BranchLessThanCoreAir, BranchLessThanFiller, LessThanCoreAir,
    LessThanFiller, MultiplicationCoreAir, MultiplicationFiller, RiscvI, RiscvIExecutor, RiscvIo,
    RiscvIoExecutor, RiscvM, RiscvMExecutor, ShiftLogicalCoreAir, ShiftLogicalFiller,
    ShiftRightArithmeticCoreAir, ShiftRightArithmeticFiller,
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
mod mult;
mod shift;
mod trace;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

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
    VecHeapAdapterAir<NUM_READS, INT256_NUM_MEMORY_BLOCKS, INT256_NUM_MEMORY_BLOCKS>,
    NUM_READS,
    INT256_NUM_MEMORY_BLOCKS,
    INT256_NUM_MEMORY_BLOCKS,
    MEMORY_BLOCK_BYTES,
    INT256_NUM_U8_LIMBS,
    INT256_NUM_U8_LIMBS,
>;

type AluU16AdapterAir = VecToFlatAluAdapterAir<
    VecHeapU16AdapterAir<NUM_READS, INT256_NUM_MEMORY_BLOCKS, INT256_NUM_MEMORY_BLOCKS>,
    NUM_READS,
    INT256_NUM_MEMORY_BLOCKS,
    INT256_NUM_MEMORY_BLOCKS,
    BLOCK_FE_WIDTH,
    INT256_NUM_U16_LIMBS,
    INT256_NUM_U16_LIMBS,
>;

/// Type alias for the Branch adapter AIR wrapper
type BranchAdapterAir = VecToFlatBranchAdapterAir<
    VecHeapBranchU16AdapterAir<NUM_READS, INT256_NUM_MEMORY_BLOCKS>,
    NUM_READS,
    INT256_NUM_MEMORY_BLOCKS,
    BLOCK_FE_WIDTH,
    INT256_NUM_U16_LIMBS,
>;

/// AddSub256 — u16 limbs, range checker (shares the AluU16 adapter with LessThan256)
pub type AddSub256Air =
    VmAirWrapper<AluU16AdapterAir, AddSubCoreAir<INT256_NUM_U16_LIMBS, U16_BITS, true>>;
#[derive(Clone, Default)]
pub struct AddSub256Executor;
pub type AddSub256Chip<F> = VmChipWrapper<F, AddSubFiller>;

/// BitwiseLogic256 — byte limbs, bitwise lookup for XOR/OR/AND.
pub type BitwiseLogic256Air =
    VmAirWrapper<AluAdapterAir, BitwiseLogicCoreAir<INT256_NUM_U8_LIMBS, BYTE_BITS>>;
#[derive(Clone, Default)]
pub struct BitwiseLogic256Executor;
pub type BitwiseLogic256Chip<F> = VmChipWrapper<F, BitwiseLogicFiller<BYTE_BITS>>;

/// LessThan256
pub type LessThan256Air =
    VmAirWrapper<AluU16AdapterAir, LessThanCoreAir<INT256_NUM_U16_LIMBS, U16_BITS>>;
#[derive(Clone, Default)]
pub struct LessThan256Executor;
pub type LessThan256Chip<F> = VmChipWrapper<F, LessThanFiller>;

/// Multiplication256
pub type Multiplication256Air =
    VmAirWrapper<AluAdapterAir, MultiplicationCoreAir<INT256_NUM_U8_LIMBS, BYTE_BITS>>;
#[derive(Clone, Default)]
pub struct Multiplication256Executor;
pub type Multiplication256Chip<F> =
    VmChipWrapper<F, MultiplicationFiller<INT256_NUM_U8_LIMBS, BYTE_BITS>>;

/// Shift256 — SLL/SRL/SRA all use u16 limbs (AluU16 adapter).
pub type ShiftLogical256Air =
    VmAirWrapper<AluU16AdapterAir, ShiftLogicalCoreAir<INT256_NUM_U16_LIMBS, U16_BITS>>;
pub type ShiftRightArithmetic256Air =
    VmAirWrapper<AluU16AdapterAir, ShiftRightArithmeticCoreAir<INT256_NUM_U16_LIMBS, U16_BITS>>;
#[derive(Clone, Default)]
pub struct ShiftLogical256Executor;
#[derive(Clone, Default)]
pub struct ShiftRightArithmetic256Executor;
pub type ShiftLogical256Chip<F> = VmChipWrapper<F, ShiftLogicalFiller>;
pub type ShiftRightArithmetic256Chip<F> = VmChipWrapper<F, ShiftRightArithmeticFiller>;

/// BranchEqual256
pub type BranchEqual256Air =
    VmAirWrapper<BranchAdapterAir, BranchEqualCoreAir<INT256_NUM_U16_LIMBS>>;
#[derive(Clone, Default)]
pub struct BranchEqual256Executor;
pub type BranchEqual256Chip<F> = VmChipWrapper<F, BranchEqualFiller>;

/// BranchLessThan256
pub type BranchLessThan256Air =
    VmAirWrapper<BranchAdapterAir, BranchLessThanCoreAir<INT256_NUM_U16_LIMBS, U16_BITS>>;
#[derive(Clone, Default)]
pub struct BranchLessThan256Executor;
pub type BranchLessThan256Chip<F> = VmChipWrapper<F, BranchLessThanFiller>;

#[derive(Clone, Debug, VmConfig, derive_new::new, Serialize, Deserialize)]
pub struct Int256Config {
    #[config(executor = "SystemExecutor")]
    pub system: SystemConfig,
    #[extension]
    pub riscv_i: RiscvI,
    #[extension]
    pub riscv_m: RiscvM,
    #[extension]
    pub io: RiscvIo,
    #[extension]
    pub bigint: Int256,
}

// Default implementation uses no init file
impl InitFileGenerator for Int256Config {}

impl Default for Int256Config {
    fn default() -> Self {
        Self {
            system: SystemConfig::default(),
            riscv_i: RiscvI,
            riscv_m: RiscvM::default(),
            io: RiscvIo,
            bigint: Int256::default(),
        }
    }
}
