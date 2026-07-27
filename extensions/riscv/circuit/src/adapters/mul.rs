use std::borrow::{Borrow, BorrowMut};

#[cfg(test)]
use openvm_circuit::arch::{Postflight, PostflightError, PostflightStep};
use openvm_circuit::{
    arch::{
        get_record_from_slice, AdapterAirContext, AdapterTraceExecutor, AdapterTraceFiller,
        BasicAdapterInterface, ExecutionBridge, ExecutionState, MinimalInstruction, VmAdapterAir,
        BLOCK_FE_WIDTH,
    },
    system::memory::{
        offline_checker::{
            pack_u8_block, pack_u8_block_bytes, MemoryBridge, MemoryReadAuxCols,
            MemoryReadAuxRecord, MemoryWriteAuxCols, MemoryWriteBytesAuxRecord,
        },
        online::TracingMemory,
        MemoryAddress, MemoryAuxColsFactory,
    },
};
use openvm_circuit_primitives::{
    AlignedBytesBorrow, ColumnsAir, StructReflection, StructReflectionHelper,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{
    instruction::Instruction, program::DEFAULT_PC_STEP, riscv::RV64_REGISTER_AS,
};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, PrimeCharacteristicRing, PrimeField32},
};

use super::{tracing_write, RV64_REGISTER_NUM_LIMBS};
use crate::adapters::{byte_ptr_to_u16_ptr, tracing_read};
#[cfg(test)]
use crate::adapters::{
    checked_byte_ptr_to_u16_ptr_value, rv64_bytes_to_u16_block, rv64_u16_block_to_bytes,
};

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct Rv64MultAdapterCols<T> {
    pub from_state: ExecutionState<T>,
    pub rd_ptr: T,
    pub rs1_ptr: T,
    pub rs2_ptr: T,
    pub reads_aux: [MemoryReadAuxCols<T>; 2],
    pub writes_aux: MemoryWriteAuxCols<T, BLOCK_FE_WIDTH>,
}

/// Reads instructions of the form OP a, b, c, d where \[a:8\]_d = \[b:8\]_d op \[c:8\]_d.
/// Operand d can only be 1, and there is no immediate support.
#[derive(Clone, Copy, Debug, derive_new::new, ColumnsAir)]
#[columns_via(Rv64MultAdapterCols<u8>)]
pub struct Rv64MultAdapterAir {
    pub(super) execution_bridge: ExecutionBridge,
    pub(super) memory_bridge: MemoryBridge,
}

impl<F: Field> BaseAir<F> for Rv64MultAdapterAir {
    fn width(&self) -> usize {
        Rv64MultAdapterCols::<F>::width()
    }
}

impl<AB: InteractionBuilder> VmAdapterAir<AB> for Rv64MultAdapterAir {
    type Interface = BasicAdapterInterface<
        AB::Expr,
        MinimalInstruction<AB::Expr>,
        2,
        1,
        RV64_REGISTER_NUM_LIMBS,
        RV64_REGISTER_NUM_LIMBS,
    >;

    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        let local: &Rv64MultAdapterCols<_> = local.borrow();
        let timestamp = local.from_state.timestamp;
        let mut timestamp_delta: usize = 0;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            timestamp + AB::F::from_usize(timestamp_delta - 1)
        };

        self.memory_bridge
            .read(
                MemoryAddress::new(
                    AB::F::from_u32(RV64_REGISTER_AS),
                    byte_ptr_to_u16_ptr::<AB>(local.rs1_ptr),
                ),
                pack_u8_block::<AB>(&ctx.reads[0].clone()),
                timestamp_pp(),
                &local.reads_aux[0],
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        self.memory_bridge
            .read(
                MemoryAddress::new(
                    AB::F::from_u32(RV64_REGISTER_AS),
                    byte_ptr_to_u16_ptr::<AB>(local.rs2_ptr),
                ),
                pack_u8_block::<AB>(&ctx.reads[1].clone()),
                timestamp_pp(),
                &local.reads_aux[1],
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        self.memory_bridge
            .write(
                MemoryAddress::new(
                    AB::F::from_u32(RV64_REGISTER_AS),
                    byte_ptr_to_u16_ptr::<AB>(local.rd_ptr),
                ),
                pack_u8_block::<AB>(&ctx.writes[0].clone()),
                timestamp_pp(),
                &local.writes_aux,
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        self.execution_bridge
            .execute_and_increment_or_set_pc(
                ctx.instruction.opcode,
                [
                    local.rd_ptr.into(),
                    local.rs1_ptr.into(),
                    local.rs2_ptr.into(),
                    AB::Expr::from_u32(RV64_REGISTER_AS),
                    AB::Expr::ZERO,
                ],
                local.from_state,
                AB::F::from_usize(timestamp_delta),
                (DEFAULT_PC_STEP, ctx.to_pc),
            )
            .eval(builder, ctx.instruction.is_valid);
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        let cols: &Rv64MultAdapterCols<_> = local.borrow();
        cols.from_state.pc
    }
}

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug)]
pub struct Rv64MultAdapterRecord {
    pub from_pc: u32,
    pub from_timestamp: u32,

    pub rd_ptr: u32,
    pub rs1_ptr: u32,
    pub rs2_ptr: u32,

    pub reads_aux: [MemoryReadAuxRecord; 2],
    pub writes_aux: MemoryWriteBytesAuxRecord<RV64_REGISTER_NUM_LIMBS>,
}

#[derive(Clone, Copy, derive_new::new)]
pub struct Rv64MultAdapterExecutor;

#[derive(Clone, Copy, derive_new::new)]
pub struct Rv64MultAdapterFiller;

impl<F> AdapterTraceExecutor<F> for Rv64MultAdapterExecutor
where
    F: PrimeField32,
{
    const WIDTH: usize = size_of::<Rv64MultAdapterCols<u8>>();
    type ReadData = [[u8; RV64_REGISTER_NUM_LIMBS]; 2];
    type WriteData = [[u8; RV64_REGISTER_NUM_LIMBS]; 1];
    type RecordMut<'a> = &'a mut Rv64MultAdapterRecord;

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
        let &Instruction { b, c, d, .. } = instruction;

        debug_assert_eq!(d.as_canonical_u32(), RV64_REGISTER_AS);

        record.rs1_ptr = b.as_canonical_u32();
        let rs1 = tracing_read(
            memory,
            RV64_REGISTER_AS,
            b.as_canonical_u32(),
            &mut record.reads_aux[0].prev_timestamp,
        );
        record.rs2_ptr = c.as_canonical_u32();
        let rs2 = tracing_read(
            memory,
            RV64_REGISTER_AS,
            c.as_canonical_u32(),
            &mut record.reads_aux[1].prev_timestamp,
        );

        [rs1, rs2]
    }

    #[inline(always)]
    fn write(
        &self,
        memory: &mut TracingMemory,
        instruction: &Instruction<F>,
        data: Self::WriteData,
        record: &mut Self::RecordMut<'_>,
    ) {
        let &Instruction { a, d, .. } = instruction;

        debug_assert_eq!(d.as_canonical_u32(), RV64_REGISTER_AS);

        record.rd_ptr = a.as_canonical_u32();
        tracing_write(
            memory,
            RV64_REGISTER_AS,
            a.as_canonical_u32(),
            data[0],
            &mut record.writes_aux.prev_timestamp,
            &mut record.writes_aux.prev_data,
        )
    }
}

impl<F: PrimeField32> AdapterTraceFiller<F> for Rv64MultAdapterFiller {
    const WIDTH: usize = size_of::<Rv64MultAdapterCols<u8>>();

    #[inline(always)]
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, mut adapter_row: &mut [F]) {
        // SAFETY:
        // - caller ensures `adapter_row` contains a valid record representation that was previously
        //   written by the executor
        // - get_record_from_slice correctly interprets the bytes as Rv64MultAdapterRecord
        let record: &Rv64MultAdapterRecord = unsafe { get_record_from_slice(&mut adapter_row, ()) };
        let adapter_row: &mut Rv64MultAdapterCols<F> = adapter_row.borrow_mut();

        let timestamp = record.from_timestamp;

        adapter_row
            .writes_aux
            .set_prev_data(pack_u8_block_bytes(&record.writes_aux.prev_data));
        mem_helper.fill(
            record.writes_aux.prev_timestamp,
            timestamp + 2,
            adapter_row.writes_aux.as_mut(),
        );

        mem_helper.fill(
            record.reads_aux[1].prev_timestamp,
            timestamp + 1,
            adapter_row.reads_aux[1].as_mut(),
        );

        mem_helper.fill(
            record.reads_aux[0].prev_timestamp,
            timestamp,
            adapter_row.reads_aux[0].as_mut(),
        );

        adapter_row.rs2_ptr = F::from_u32(record.rs2_ptr);
        adapter_row.rs1_ptr = F::from_u32(record.rs1_ptr);
        adapter_row.rd_ptr = F::from_u32(record.rd_ptr);

        adapter_row.from_state.timestamp = F::from_u32(record.from_timestamp);
        adapter_row.from_state.pc = F::from_u32(record.from_pc);
    }
}

#[cfg(test)]
impl Rv64MultAdapterFiller {
    pub(crate) fn replay<F: PrimeField32>(
        postflight: &Postflight<'_, F>,
        step: PostflightStep,
        mem_helper: &MemoryAuxColsFactory<F>,
        adapter_row: &mut Rv64MultAdapterCols<F>,
        compute: impl FnOnce([[u8; RV64_REGISTER_NUM_LIMBS]; 2]) -> [u8; RV64_REGISTER_NUM_LIMBS],
    ) -> Result<
        (
            [[u8; RV64_REGISTER_NUM_LIMBS]; 2],
            [u8; RV64_REGISTER_NUM_LIMBS],
        ),
        PostflightError,
    > {
        let instruction = postflight.instruction(step);
        if instruction.d.as_canonical_u32() != RV64_REGISTER_AS
            || instruction.e.as_canonical_u32() != 0
        {
            return Err(PostflightError::new(
                "multiplication instruction has invalid address spaces",
            ));
        }
        let from_pc = postflight.pc(step);
        let from_timestamp = postflight.timestamp(step);
        let rs1_ptr = instruction.b.as_canonical_u32();
        let rs2_ptr = instruction.c.as_canonical_u32();
        let rd_ptr = instruction.a.as_canonical_u32();
        let rs1_u16_ptr = checked_byte_ptr_to_u16_ptr_value(rs1_ptr)?;
        let rs2_u16_ptr = checked_byte_ptr_to_u16_ptr_value(rs2_ptr)?;
        let rd_u16_ptr = checked_byte_ptr_to_u16_ptr_value(rd_ptr)?;
        let mut replay = postflight.replay(step);
        let rs1 = replay.read_u16(RV64_REGISTER_AS, rs1_u16_ptr)?;
        let rs2 = replay.read_u16(RV64_REGISTER_AS, rs2_u16_ptr)?;
        let inputs = [
            rv64_u16_block_to_bytes(rs1.value),
            rv64_u16_block_to_bytes(rs2.value),
        ];
        let output = compute(inputs);
        let write = replay.write_u16(
            RV64_REGISTER_AS,
            rd_u16_ptr,
            rv64_bytes_to_u16_block(output),
        )?;
        replay.finish(from_pc.wrapping_add(DEFAULT_PC_STEP))?;

        adapter_row
            .writes_aux
            .set_prev_data(write.previous_value.map(F::from_u16));
        mem_helper.fill(
            write.previous_timestamp,
            write.timestamp,
            adapter_row.writes_aux.as_mut(),
        );
        mem_helper.fill(
            rs2.previous_timestamp,
            rs2.timestamp,
            adapter_row.reads_aux[1].as_mut(),
        );
        mem_helper.fill(
            rs1.previous_timestamp,
            rs1.timestamp,
            adapter_row.reads_aux[0].as_mut(),
        );
        adapter_row.rs2_ptr = F::from_u32(rs2_ptr);
        adapter_row.rs1_ptr = F::from_u32(rs1_ptr);
        adapter_row.rd_ptr = F::from_u32(rd_ptr);
        adapter_row.from_state.timestamp = F::from_u32(from_timestamp);
        adapter_row.from_state.pc = F::from_u32(from_pc);

        Ok((inputs, output))
    }
}
