mod extension;
pub mod weierstrass_chip;

pub use extension::*;
use openvm_mod_circuit_builder::FieldExpressionCoreRecordMut;
use openvm_rv32_adapters::Rv32VecHeapAdapterRecord;
pub use weierstrass_chip::*;

type EccRecord<'a, const NUM_READS: usize, const BLOCKS: usize, const BLOCK_SIZE: usize> = (
    &'a mut Rv32VecHeapAdapterRecord<NUM_READS, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>,
    FieldExpressionCoreRecordMut<'a>,
);
