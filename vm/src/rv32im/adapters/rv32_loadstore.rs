use std::{marker::PhantomData, sync::Arc};

use afs_derive::AlignedBorrow;
use afs_primitives::var_range::VariableRangeCheckerChip;
use afs_stark_backend::interaction::InteractionBuilder;
use p3_air::BaseAir;
use p3_field::{Field, PrimeField32};

use super::{compose, RV32_REGISTER_NUM_LANES, RV_IS_TYPE_IMM_BITS};
use crate::{
    arch::{
        instructions::{
            Rv32LoadStoreOpcode::{self, *},
            UsizeOpcode,
        },
        AdapterAirContext, AdapterRuntimeContext, ExecutionState, Result, VmAdapterAir,
        VmAdapterChip, VmAdapterInterface,
    },
    system::{
        memory::{
            offline_checker::{MemoryReadAuxCols, MemoryWriteAuxCols},
            MemoryChip, MemoryReadRecord, MemoryWriteRecord,
        },
        program::Instruction,
    },
};

#[repr(C)]
#[derive(AlignedBorrow, Clone, Debug)]
pub struct Rv32LoadStoreAdapterCols<T, const NUM_CELLS: usize> {
    pub a: T,
    pub b: T,
    pub c: T,
    pub d: T, // will fix to 1 to save a column
    pub e: T,
    pub ptr: [T; RV32_REGISTER_NUM_LANES],
    // pub read: [T; NUM_CELLS],
    // pub write: [T; NUM_CELLS],
    pub read_ptr_aux: MemoryReadAuxCols<T, RV32_REGISTER_NUM_LANES>,
    pub read_data_aux: MemoryReadAuxCols<T, NUM_CELLS>,
    pub write_aux: MemoryWriteAuxCols<T, NUM_CELLS>,
}

#[derive(Debug, Clone, Copy)]
pub struct Rv32LoadStoreAdapterAir<F: Field, const NUM_CELLS: usize> {
    marker: PhantomData<F>,
}

impl<F: Field, const NUM_CELLS: usize> BaseAir<F> for Rv32LoadStoreAdapterAir<F, NUM_CELLS> {
    fn width(&self) -> usize {
        todo!()
    }
}

impl<AB: InteractionBuilder, const NUM_CELLS: usize> VmAdapterAir<AB>
    for Rv32LoadStoreAdapterAir<AB::F, NUM_CELLS>
{
    type Interface = Rv32LoadStoreAdapterInterface<AB::Expr, NUM_CELLS>;

    fn eval(
        &self,
        _builder: &mut AB,
        _local: &[AB::Var],
        _ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        todo!()
    }
}

#[derive(Debug, Clone)]
pub struct Rv32LoadStoreAdapterReadRecord<F: Field, const NUM_CELLS: usize> {
    pub rs1: MemoryReadRecord<F, RV32_REGISTER_NUM_LANES>,

    // This will be a read from a register in case of Stores and a read from RISC-V memory in case of Loads
    pub read: MemoryReadRecord<F, NUM_CELLS>,
}

#[derive(Debug, Clone)]
pub struct Rv32LoadStoreAdapterWriteRecord<F: Field, const NUM_CELLS: usize> {
    // This will be a write to a register in case of Load and a write to RISC-V memory in case of Stores
    pub write: MemoryWriteRecord<F, NUM_CELLS>,
}

#[derive(Debug, Clone)]
pub struct Rv32LoadStoreAdapterInterface<T, const NUM_CELLS: usize> {
    _marker: PhantomData<T>,
}

impl<T, const NUM_CELLS: usize> VmAdapterInterface<T>
    for Rv32LoadStoreAdapterInterface<T, NUM_CELLS>
{
    /// `[read_data, prev_data]` where `prev_data` is currenlty only used when this is a STORE instruction.
    type Reads = [[T; NUM_CELLS]; 2];
    type Writes = [T; NUM_CELLS];
    type ProcessedInstruction = Instruction<T>;
}

#[derive(Debug, Clone)]
pub struct Rv32LoadStoreAdapter<F: Field, const NUM_CELLS: usize> {
    pub air: Rv32LoadStoreAdapterAir<F, NUM_CELLS>,
    pub offset: usize,
    pub range_checker_chip: Arc<VariableRangeCheckerChip>,
}

impl<F: Field, const NUM_CELLS: usize> Rv32LoadStoreAdapter<F, NUM_CELLS> {
    pub fn new(range_checker_chip: Arc<VariableRangeCheckerChip>, offset: usize) -> Self {
        Self {
            air: Rv32LoadStoreAdapterAir::<F, NUM_CELLS> {
                marker: PhantomData,
            },
            offset,
            range_checker_chip,
        }
    }
}

impl<F: PrimeField32, const NUM_CELLS: usize> VmAdapterChip<F>
    for Rv32LoadStoreAdapter<F, NUM_CELLS>
{
    type ReadRecord = Rv32LoadStoreAdapterReadRecord<F, NUM_CELLS>;
    type WriteRecord = Rv32LoadStoreAdapterWriteRecord<F, NUM_CELLS>;
    type Air = Rv32LoadStoreAdapterAir<F, NUM_CELLS>;
    type Interface = Rv32LoadStoreAdapterInterface<F, NUM_CELLS>;

    #[allow(clippy::type_complexity)]
    fn preprocess(
        &mut self,
        memory: &mut MemoryChip<F>,
        instruction: &Instruction<F>,
    ) -> Result<(
        <Self::Interface as VmAdapterInterface<F>>::Reads,
        Self::ReadRecord,
    )> {
        let Instruction {
            opcode,
            op_a: a,
            op_b: b,
            op_c: c,
            d,
            e,
            ..
        } = *instruction;

        debug_assert_eq!(d.as_canonical_u32(), 1);
        debug_assert_eq!(e.as_canonical_u32(), 2);

        // We constrain that the pointer to the memory has ar most addr_bits
        let addr_bits = memory.mem_config.pointer_max_bits;
        debug_assert!(addr_bits >= (RV32_REGISTER_NUM_LANES - 1) * 8);

        let rs1_record = memory.read::<RV32_REGISTER_NUM_LANES>(d, b);
        let rs1_val = compose(rs1_record.data);

        // Note: c is a field element and immediate is a signed integer
        let imm = (c + F::from_canonical_u32(1 << (RV_IS_TYPE_IMM_BITS - 1))).as_canonical_u32();
        let ptr_val = rs1_val + imm - (1 << (RV_IS_TYPE_IMM_BITS - 1));

        assert!(imm < (1 << RV_IS_TYPE_IMM_BITS));
        assert!(ptr_val < (1 << addr_bits));

        let local_opcode_index = Rv32LoadStoreOpcode::from_usize(opcode - self.offset);

        let read_record = match local_opcode_index {
            LOADW | LOADB | LOADH | LOADBU | LOADHU => {
                memory.read::<NUM_CELLS>(e, F::from_canonical_u32(ptr_val))
            }
            STOREW | STOREH | STOREB => memory.read::<NUM_CELLS>(d, a),
        };

        // We need to keep values of some cells to keep them unchanged when writing to those cells
        let mut prev_data = [F::zero(); NUM_CELLS];
        match local_opcode_index {
            STOREH => {
                for (i, cell) in prev_data
                    .iter_mut()
                    .enumerate()
                    .take(NUM_CELLS)
                    .skip(NUM_CELLS / 2)
                {
                    *cell =
                        memory.unsafe_read_cell(e, F::from_canonical_usize(ptr_val as usize + i));
                }
            }
            STOREB => {
                for (i, cell) in prev_data.iter_mut().enumerate().take(NUM_CELLS).skip(1) {
                    *cell =
                        memory.unsafe_read_cell(e, F::from_canonical_usize(ptr_val as usize + i));
                }
            }
            _ => (),
        }

        // TODO[arayi]: send VariableRangeChecker requests

        let read_data = read_record.data;
        Ok((
            [read_data, prev_data],
            Self::ReadRecord {
                rs1: rs1_record,
                read: read_record,
            },
        ))
    }

    fn postprocess(
        &mut self,
        memory: &mut MemoryChip<F>,
        instruction: &Instruction<F>,
        from_state: ExecutionState<usize>,
        output: AdapterRuntimeContext<F, Self::Interface>,
        read_record: &Self::ReadRecord,
    ) -> Result<(ExecutionState<usize>, Self::WriteRecord)> {
        let Instruction {
            opcode,
            op_a: a,
            op_c: c,
            d,
            e,
            ..
        } = *instruction;

        let local_opcode_index = Rv32LoadStoreOpcode::from_usize(opcode - self.offset);

        let write_record = match local_opcode_index {
            STOREW | STOREH | STOREB => {
                let ptr = compose(read_record.rs1.data);
                let imm =
                    (c + F::from_canonical_u32(1 << (RV_IS_TYPE_IMM_BITS - 1))).as_canonical_u32();
                let ptr = ptr + imm - (1 << (RV_IS_TYPE_IMM_BITS - 1));
                memory.write(e, F::from_canonical_u32(ptr), output.writes)
            }
            LOADW | LOADB | LOADH | LOADBU | LOADHU => {
                if a.as_canonical_u32() != 0 {
                    memory.write(d, a, output.writes)
                } else {
                    memory.write(d, a, [F::zero(); NUM_CELLS])
                }
            }
        };

        Ok((
            ExecutionState {
                pc: output
                    .to_pc
                    .unwrap_or(F::from_canonical_usize(from_state.pc + 4))
                    .as_canonical_u32() as usize,
                timestamp: memory.timestamp().as_canonical_u32() as usize,
            },
            Self::WriteRecord {
                write: write_record,
            },
        ))
    }

    fn generate_trace_row(
        &self,
        _row_slice: &mut [F],
        _read_record: Self::ReadRecord,
        _write_record: Self::WriteRecord,
    ) {
        todo!()
    }

    fn air(&self) -> &Self::Air {
        &self.air
    }
}
