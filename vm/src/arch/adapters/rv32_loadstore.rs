use std::sync::Arc;

use afs_derive::AlignedBorrow;
use afs_primitives::var_range::VariableRangeCheckerChip;
use afs_stark_backend::interaction::InteractionBuilder;
use p3_air::{AirBuilderWithPublicValues, BaseAir, PairBuilder};
use p3_field::{AbstractField, Field, PrimeField32};

use crate::{
    arch::{
        instructions::{
            LoadStoreOpcode::{self, *},
            UsizeOpcode,
        },
        ExecutionState, InstructionOutput, IntegrationInterface, MachineAdapter,
        MachineAdapterInterface, Result, RV32_IMM_BITS, RV32_REGISTER_NUM_LANES,
    },
    memory::{
        offline_checker::{MemoryReadAuxCols, MemoryWriteAuxCols},
        MemoryChip, MemoryReadRecord, MemoryWriteRecord,
    },
    program::Instruction,
};

#[repr(C)]
#[derive(AlignedBorrow, Clone, Debug)]
pub struct Rv32LoadStoreAdapterCols<T, const NUM_CELLS: usize> {
    pub a: usize,
    pub b: usize,
    pub c: usize,
    pub d: usize, // will fix to 1 to save a column
    pub e: usize,
    pub ptr: [T; RV32_REGISTER_NUM_LANES],
    // pub read: [T; NUM_CELLS],
    // pub write: [T; NUM_CELLS],
    pub read_ptr_aux: MemoryReadAuxCols<T, RV32_REGISTER_NUM_LANES>,
    pub read_data_aux: MemoryReadAuxCols<T, NUM_CELLS>,
    pub write_aux: MemoryWriteAuxCols<T, NUM_CELLS>,
}

#[derive(Debug, Clone, Copy)]
pub struct Rv32LoadStoreAdapterAir<F: Field> {
    marker: std::marker::PhantomData<F>,
}

impl<F: Field> BaseAir<F> for Rv32LoadStoreAdapterAir<F> {
    fn width(&self) -> usize {
        todo!()
    }
}

#[derive(Debug, Clone)]
pub struct Rv32LoadStoreAdapterReadRecord<F: Field, const NUM_CELLS: usize> {
    pub ptr: MemoryReadRecord<F, RV32_REGISTER_NUM_LANES>,
    pub read: MemoryReadRecord<F, NUM_CELLS>,
}

#[derive(Debug, Clone)]
pub struct Rv32LoadStoreAdapterWriteRecord<F: Field, const NUM_CELLS: usize> {
    pub write: MemoryWriteRecord<F, NUM_CELLS>,
}

#[derive(Debug, Clone)]
pub struct Rv32LoadStoreAdapterInterface<T, const NUM_CELLS: usize> {
    _marker: std::marker::PhantomData<T>,
}

impl<T, const NUM_CELLS: usize> MachineAdapterInterface<T>
    for Rv32LoadStoreAdapterInterface<T, NUM_CELLS>
{
    type Reads = [[T; NUM_CELLS]; 2];
    type Writes = [T; NUM_CELLS];
    type ProcessedInstruction = Instruction<T>;
}

#[derive(Debug, Clone)]
pub struct Rv32LoadStoreAdapter<F: Field, const NUM_CELLS: usize> {
    pub air: Rv32LoadStoreAdapterAir<F>,
    pub offset: usize,
    pub range_checker_chip: Arc<VariableRangeCheckerChip>,
}

impl<F: Field, const NUM_CELLS: usize> Rv32LoadStoreAdapter<F, NUM_CELLS> {
    pub fn new(range_checker_chip: Arc<VariableRangeCheckerChip>, offset: usize) -> Self {
        Self {
            air: Rv32LoadStoreAdapterAir::<F> {
                marker: std::marker::PhantomData,
            },
            offset,
            range_checker_chip,
        }
    }
}

impl<F: PrimeField32, const NUM_CELLS: usize> MachineAdapter<F>
    for Rv32LoadStoreAdapter<F, NUM_CELLS>
{
    type ReadRecord = Rv32LoadStoreAdapterReadRecord<F, NUM_CELLS>;
    type WriteRecord = Rv32LoadStoreAdapterWriteRecord<F, NUM_CELLS>;
    type Air = Rv32LoadStoreAdapterAir<F>;
    type Cols<T> = Rv32LoadStoreAdapterCols<T, NUM_CELLS>;

    type Interface<T: AbstractField> = Rv32LoadStoreAdapterInterface<T, NUM_CELLS>;

    #[allow(clippy::type_complexity)]
    fn preprocess(
        &mut self,
        memory: &mut MemoryChip<F>,
        instruction: &Instruction<F>,
    ) -> Result<(
        <Self::Interface<F> as MachineAdapterInterface<F>>::Reads,
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

        // TODO: add comment here
        let addr_bits = memory.mem_config.pointer_max_bits;
        debug_assert_eq!(d.as_canonical_u32(), 1);
        debug_assert_eq!(e.as_canonical_u32(), 2); // not sure if this is needed
        debug_assert!(addr_bits >= (RV32_REGISTER_NUM_LANES - 1) * 8);

        let ptr_record = memory.read::<RV32_REGISTER_NUM_LANES>(d, b);
        let ptr_data = ptr_record.data.map(|x| x.as_canonical_u32());

        for limb in ptr_data {
            debug_assert!(limb < (1 << 8));
        }
        debug_assert!(
            ptr_data[RV32_REGISTER_NUM_LANES - 1]
                < (1 << (addr_bits - (RV32_REGISTER_NUM_LANES - 1) * 8))
        );

        // TODO: add comment here
        let ptr_val = compose(ptr_data);
        let imm = (c + F::from_canonical_u32(1 << (RV32_IMM_BITS - 1))).as_canonical_u32();
        let ptr_val = ptr_val + imm - (1 << (RV32_IMM_BITS - 1));

        assert!(imm < (1 << RV32_IMM_BITS));
        assert!(ptr_val < (1 << addr_bits));

        let opcode = LoadStoreOpcode::from_usize(opcode - self.offset);

        let read_record = match opcode {
            LOADW => memory.read::<NUM_CELLS>(e, F::from_canonical_u32(ptr_val)),
            STOREW | STOREH | STOREB => memory.read::<NUM_CELLS>(d, a),
        };
        let read_data = read_record.data;

        let mut prev_data = [F::zero(); NUM_CELLS];

        // TODO: add comment here
        match opcode {
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

        // TODO: send VariableRangeChecker requests
        Ok((
            [read_data, prev_data],
            Self::ReadRecord {
                ptr: ptr_record,
                read: read_record,
            },
        ))
    }

    fn postprocess(
        &mut self,
        memory: &mut MemoryChip<F>,
        instruction: &Instruction<F>,
        from_state: ExecutionState<usize>,
        output: InstructionOutput<F, Self::Interface<F>>,
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

        let opcode = LoadStoreOpcode::from_usize(opcode - self.offset);

        let write_record = match opcode {
            STOREW | STOREH | STOREB => {
                let ptr = compose(read_record.ptr.data.map(|x| x.as_canonical_u32()));
                let imm = (c + F::from_canonical_u32(1 << (RV32_IMM_BITS - 1))).as_canonical_u32();
                let ptr = ptr + imm - (1 << (RV32_IMM_BITS - 1));
                memory.write(e, F::from_canonical_u32(ptr), output.writes)
            }
            LOADW => {
                if a.as_canonical_u32() != 0 {
                    memory.write(d, a, output.writes)
                } else {
                    memory.write(d, a, [F::zero(); NUM_CELLS])
                }
            }
        };

        Ok((
            ExecutionState {
                pc: from_state.pc + 4,
                timestamp: memory.timestamp().as_canonical_u32() as usize,
            },
            Self::WriteRecord {
                write: write_record,
            },
        ))
    }

    fn generate_trace_row(
        &self,
        _row_slice: &mut Self::Cols<F>,
        _read_record: Self::ReadRecord,
        _write_record: Self::WriteRecord,
    ) {
        todo!()
    }

    fn eval_adapter_constraints<
        AB: InteractionBuilder<F = F> + PairBuilder + AirBuilderWithPublicValues,
    >(
        _air: &Self::Air,
        _builder: &mut AB,
        _local: &Self::Cols<AB::Var>,
        _interface: IntegrationInterface<AB::Expr, Self::Interface<AB::Expr>>,
    ) -> AB::Expr {
        todo!()
    }

    fn air(&self) -> Self::Air {
        todo!()
    }
}

// TODO[arayi]: make it more general and usable for other structs as well
pub fn compose<const N: usize>(ptr_data: [u32; N]) -> u32 {
    let mut val = 0;
    for (i, limb) in ptr_data.iter().enumerate() {
        val += limb << (i * 8);
    }
    val
}
