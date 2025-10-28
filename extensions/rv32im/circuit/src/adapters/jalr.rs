use std::borrow::{Borrow, BorrowMut};

use openvm_circuit::{
    arch::{
        get_record_from_slice, AdapterAirContext, AdapterTraceExecutor, AdapterTraceFiller,
        BasicAdapterInterface, ExecutionBridge, ExecutionState, SignedImmInstruction, VmAdapterAir,
    },
    system::memory::{
        offline_checker::{
            MemoryBridge, MemoryReadAuxCols, MemoryReadAuxRecord, MemoryWriteAuxCols,
            MemoryWriteBytesAuxRecord,
        },
        online::TracingMemory,
        MemoryAddress, MemoryAuxColsFactory,
    },
};
use openvm_circuit_primitives::{utils::not, AlignedBytesBorrow};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{
    instruction::Instruction, program::DEFAULT_PC_STEP, riscv::RV32_REGISTER_AS,
};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{AirBuilder, BaseAir},
    p3_field::{Field, FieldAlgebra, PrimeField32},
};

use super::RV32_REGISTER_NUM_LIMBS;
use crate::adapters::{tracing_read, tracing_write};

#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow)]
pub struct Rv32JalrAdapterCols<T> {
    pub from_state: ExecutionState<T>,
    pub rs1_ptr: T,
    pub rs1_aux_cols: MemoryReadAuxCols<T>,
    pub rd_ptr: T,
    pub rd_aux_cols: MemoryWriteAuxCols<T, RV32_REGISTER_NUM_LIMBS>,
    /// Only writes if `needs_write`.
    /// Sets `needs_write` to 0 iff `rd == x0`
    pub needs_write: T,
}

#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct Rv32JalrAdapterAir {
    pub(super) memory_bridge: MemoryBridge,
    pub(super) execution_bridge: ExecutionBridge,
}

impl<F: Field> BaseAir<F> for Rv32JalrAdapterAir {
    fn width(&self) -> usize {
        Rv32JalrAdapterCols::<F>::width()
    }
}

impl<AB: InteractionBuilder> VmAdapterAir<AB> for Rv32JalrAdapterAir {
    type Interface = BasicAdapterInterface<
        AB::Expr,
        SignedImmInstruction<AB::Expr>,
        1,
        1,
        RV32_REGISTER_NUM_LIMBS,
        RV32_REGISTER_NUM_LIMBS,
    >;

    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        let local_cols: &Rv32JalrAdapterCols<AB::Var> = local.borrow();

        let timestamp: AB::Var = local_cols.from_state.timestamp;
        let mut timestamp_delta: usize = 0;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            timestamp + AB::Expr::from_canonical_usize(timestamp_delta - 1)
        };

        let write_count = local_cols.needs_write;

        builder.assert_bool(write_count);
        builder
            .when::<AB::Expr>(not(ctx.instruction.is_valid.clone()))
            .assert_zero(write_count);

        self.memory_bridge
            .read(
                MemoryAddress::new(
                    AB::F::from_canonical_u32(RV32_REGISTER_AS),
                    local_cols.rs1_ptr,
                ),
                ctx.reads[0].clone(),
                timestamp_pp(),
                &local_cols.rs1_aux_cols,
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        self.memory_bridge
            .write(
                MemoryAddress::new(
                    AB::F::from_canonical_u32(RV32_REGISTER_AS),
                    local_cols.rd_ptr,
                ),
                ctx.writes[0].clone(),
                timestamp_pp(),
                &local_cols.rd_aux_cols,
            )
            .eval(builder, write_count);

        let to_pc = ctx
            .to_pc
            .unwrap_or(local_cols.from_state.pc + AB::F::from_canonical_u32(DEFAULT_PC_STEP));

        // regardless of `needs_write`, must always execute instruction when `is_valid`.
        self.execution_bridge
            .execute(
                ctx.instruction.opcode,
                [
                    local_cols.rd_ptr.into(),
                    local_cols.rs1_ptr.into(),
                    ctx.instruction.immediate,
                    AB::Expr::from_canonical_u32(RV32_REGISTER_AS),
                    AB::Expr::ZERO,
                    write_count.into(),
                    ctx.instruction.imm_sign,
                ],
                local_cols.from_state,
                ExecutionState {
                    pc: to_pc,
                    timestamp: timestamp + AB::F::from_canonical_usize(timestamp_delta),
                },
            )
            .eval(builder, ctx.instruction.is_valid);
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        let cols: &Rv32JalrAdapterCols<_> = local.borrow();
        cols.from_state.pc
    }
}

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug)]
pub struct Rv32JalrAdapterRecord {
    pub from_pc: u32,
    pub from_timestamp: u32,

    pub rs1_ptr: u32,
    // Will use u32::MAX to indicate no write
    pub rd_ptr: u32,

    pub reads_aux: MemoryReadAuxRecord,
    pub writes_aux: MemoryWriteBytesAuxRecord<RV32_REGISTER_NUM_LIMBS>,
}

// This adapter reads from [b:4]_d (rs1) and writes to [a:4]_d (rd)
#[derive(Clone, Copy, derive_new::new)]
pub struct Rv32JalrAdapterExecutor;

#[derive(Clone, Copy, derive_new::new)]
pub struct Rv32JalrAdapterFiller;

impl<F> AdapterTraceExecutor<F> for Rv32JalrAdapterExecutor
where
    F: PrimeField32,
{
    const WIDTH: usize = size_of::<Rv32JalrAdapterCols<u8>>();
    type ReadData = [u8; RV32_REGISTER_NUM_LIMBS];
    type WriteData = [u8; RV32_REGISTER_NUM_LIMBS];
    type RecordMut<'a> = &'a mut Rv32JalrAdapterRecord;

    #[inline(always)]
    fn start(pc: u32, memory: &TracingMemory, record: &mut Self::RecordMut<'_>) {
        record.from_pc = pc;
        record.from_timestamp = memory.timestamp;
    }

    #[inline(always)]
    fn read(
        &self,
        memory: &mut TracingMemory,
        instruction: &Instruction<F>,
        record: &mut Self::RecordMut<'_>,
    ) -> Self::ReadData {
        let &Instruction { b, d, .. } = instruction;

        debug_assert_eq!(d.as_canonical_u32(), RV32_REGISTER_AS);

        record.rs1_ptr = b.as_canonical_u32();
        tracing_read(
            memory,
            RV32_REGISTER_AS,
            b.as_canonical_u32(),
            &mut record.reads_aux.prev_timestamp,
        )
    }

    #[inline(always)]
    fn write(
        &self,
        memory: &mut TracingMemory,
        instruction: &Instruction<F>,
        data: Self::WriteData,
        record: &mut Self::RecordMut<'_>,
    ) {
        let &Instruction {
            a, d, f: enabled, ..
        } = instruction;

        debug_assert_eq!(d.as_canonical_u32(), RV32_REGISTER_AS);

        if enabled.is_one() {
            record.rd_ptr = a.as_canonical_u32();

            tracing_write(
                memory,
                RV32_REGISTER_AS,
                a.as_canonical_u32(),
                data,
                &mut record.writes_aux.prev_timestamp,
                &mut record.writes_aux.prev_data,
            );
        } else {
            record.rd_ptr = u32::MAX;
            memory.increment_timestamp();
        }
    }
}

impl<F: PrimeField32> AdapterTraceFiller<F> for Rv32JalrAdapterFiller {
    const WIDTH: usize = size_of::<Rv32JalrAdapterCols<u8>>();

    #[inline(always)]
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, mut adapter_row: &mut [F]) {
        // SAFETY:
        // - caller ensures `adapter_row` contains a valid record representation that was previously
        //   written by the executor
        // - get_record_from_slice correctly interprets the bytes as Rv32JalrAdapterRecord
        let record: &Rv32JalrAdapterRecord = unsafe { get_record_from_slice(&mut adapter_row, ()) };
        let adapter_row: &mut Rv32JalrAdapterCols<F> = adapter_row.borrow_mut();

        // We must assign in reverse
        adapter_row.needs_write = F::from_bool(record.rd_ptr != u32::MAX);

        if record.rd_ptr != u32::MAX {
            adapter_row
                .rd_aux_cols
                .set_prev_data(record.writes_aux.prev_data.map(F::from_canonical_u8));
            mem_helper.fill(
                record.writes_aux.prev_timestamp,
                record.from_timestamp + 1,
                adapter_row.rd_aux_cols.as_mut(),
            );
            adapter_row.rd_ptr = F::from_canonical_u32(record.rd_ptr);
        } else {
            adapter_row.rd_ptr = F::ZERO;
        }

        mem_helper.fill(
            record.reads_aux.prev_timestamp,
            record.from_timestamp,
            adapter_row.rs1_aux_cols.as_mut(),
        );
        adapter_row.rs1_ptr = F::from_canonical_u32(record.rs1_ptr);
        adapter_row.from_state.timestamp = F::from_canonical_u32(record.from_timestamp);
        adapter_row.from_state.pc = F::from_canonical_u32(record.from_pc);
    }
}
