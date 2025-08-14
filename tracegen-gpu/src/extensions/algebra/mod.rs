pub mod cuda;
mod extension;
mod fp2_chip;
mod modular_chip;

pub use cuda::*;
pub use extension::*;
pub use fp2_chip::*;
pub use modular_chip::*;
use openvm_mod_circuit_builder::FieldExpressionCoreRecordMut;
use openvm_rv32_adapters::Rv32VecHeapAdapterRecord;

type AlgebraRecord<'a, const NUM_READS: usize, const BLOCKS: usize, const BLOCK_SIZE: usize> = (
    &'a mut Rv32VecHeapAdapterRecord<NUM_READS, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>,
    FieldExpressionCoreRecordMut<'a>,
);
