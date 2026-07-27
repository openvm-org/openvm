use std::borrow::{Borrow, BorrowMut};

#[cfg(test)]
use openvm_circuit::arch::{Postflight, PostflightError, PostflightStep};
use openvm_circuit::{
    arch::{
        get_record_from_slice, AdapterAirContext, AdapterTraceExecutor, AdapterTraceFiller,
        BasicAdapterInterface, ExecutionBridge, ExecutionState, ImmInstruction, VmAdapterAir,
        BLOCK_FE_WIDTH,
    },
    system::memory::{
        offline_checker::{MemoryBridge, MemoryReadAuxCols, MemoryReadAuxRecord},
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

#[cfg(test)]
use crate::adapters::checked_byte_ptr_to_u16_ptr_value;
use crate::adapters::{byte_ptr_to_u16_ptr, byte_ptr_to_u16_ptr_value, tracing_read_u16};

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct Rv64BranchAdapterCols<T> {
    pub from_state: ExecutionState<T>,
    pub rs1_ptr: T,
    pub rs2_ptr: T,
    pub reads_aux: [MemoryReadAuxCols<T>; 2],
}

#[derive(Clone, Copy, Debug, derive_new::new, ColumnsAir)]
#[columns_via(Rv64BranchAdapterCols<u8>)]
pub struct Rv64BranchAdapterAir {
    pub(super) execution_bridge: ExecutionBridge,
    pub(super) memory_bridge: MemoryBridge,
}

impl<F: Field> BaseAir<F> for Rv64BranchAdapterAir {
    fn width(&self) -> usize {
        Rv64BranchAdapterCols::<F>::width()
    }
}

impl<AB: InteractionBuilder> VmAdapterAir<AB> for Rv64BranchAdapterAir {
    type Interface =
        BasicAdapterInterface<AB::Expr, ImmInstruction<AB::Expr>, 2, 0, BLOCK_FE_WIDTH, 0>;

    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        let local: &Rv64BranchAdapterCols<_> = local.borrow();
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
                ctx.reads[0].clone(),
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
                ctx.reads[1].clone(),
                timestamp_pp(),
                &local.reads_aux[1],
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        self.execution_bridge
            .execute_and_increment_or_set_pc(
                ctx.instruction.opcode,
                [
                    local.rs1_ptr.into(),
                    local.rs2_ptr.into(),
                    ctx.instruction.immediate,
                    AB::Expr::from_u32(RV64_REGISTER_AS),
                    AB::Expr::from_u32(RV64_REGISTER_AS),
                ],
                local.from_state,
                AB::F::from_usize(timestamp_delta),
                (DEFAULT_PC_STEP, ctx.to_pc),
            )
            .eval(builder, ctx.instruction.is_valid);
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        let cols: &Rv64BranchAdapterCols<_> = local.borrow();
        cols.from_state.pc
    }
}

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug)]
pub struct Rv64BranchAdapterRecord {
    pub from_pc: u32,
    pub from_timestamp: u32,
    pub rs1_ptr: u32,
    pub rs2_ptr: u32,
    pub reads_aux: [MemoryReadAuxRecord; 2],
}

/// Reads instructions of the form OP a, b, c, d, e where if(\[a:8\]_d op \[b:8\]_e) pc += c.
/// Operands d and e can only be 1.
#[derive(Clone, Copy, derive_new::new)]
pub struct Rv64BranchAdapterExecutor;

#[derive(derive_new::new)]
pub struct Rv64BranchAdapterFiller;

impl<F> AdapterTraceExecutor<F> for Rv64BranchAdapterExecutor
where
    F: PrimeField32,
{
    const WIDTH: usize = size_of::<Rv64BranchAdapterCols<u8>>();
    type ReadData = [[u16; BLOCK_FE_WIDTH]; 2];
    type WriteData = ();
    type RecordMut<'a> = &'a mut Rv64BranchAdapterRecord;

    #[inline(always)]
    fn start(pc: u32, memory: &TracingMemory, record: &mut &mut Rv64BranchAdapterRecord) {
        record.from_pc = pc;
        record.from_timestamp = memory.timestamp;
    }

    #[inline(always)]
    fn read(
        &self,
        memory: &mut TracingMemory,
        instruction: &Instruction<F>,
        record: &mut &mut Rv64BranchAdapterRecord,
    ) -> Self::ReadData {
        let &Instruction { a, b, d, e, .. } = instruction;

        debug_assert_eq!(d.as_canonical_u32(), RV64_REGISTER_AS);
        debug_assert_eq!(e.as_canonical_u32(), RV64_REGISTER_AS);

        record.rs1_ptr = a.as_canonical_u32();
        let rs1 = tracing_read_u16::<BLOCK_FE_WIDTH>(
            memory,
            RV64_REGISTER_AS,
            byte_ptr_to_u16_ptr_value(record.rs1_ptr),
            &mut record.reads_aux[0].prev_timestamp,
        );
        record.rs2_ptr = b.as_canonical_u32();
        let rs2 = tracing_read_u16::<BLOCK_FE_WIDTH>(
            memory,
            RV64_REGISTER_AS,
            byte_ptr_to_u16_ptr_value(record.rs2_ptr),
            &mut record.reads_aux[1].prev_timestamp,
        );

        [rs1, rs2]
    }

    #[inline(always)]
    fn write(
        &self,
        _memory: &mut TracingMemory,
        _instruction: &Instruction<F>,
        _data: Self::WriteData,
        _record: &mut Self::RecordMut<'_>,
    ) {
        // This function is intentionally left empty
    }
}

impl<F: PrimeField32> AdapterTraceFiller<F> for Rv64BranchAdapterFiller {
    const WIDTH: usize = size_of::<Rv64BranchAdapterCols<u8>>();

    #[inline(always)]
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, mut adapter_row: &mut [F]) {
        // SAFETY:
        // - caller ensures `adapter_row` contains a valid record representation that was previously
        //   written by the executor
        // - get_record_from_slice correctly interprets the bytes as Rv64BranchAdapterRecord
        let record: &Rv64BranchAdapterRecord =
            unsafe { get_record_from_slice(&mut adapter_row, ()) };
        let adapter_row: &mut Rv64BranchAdapterCols<F> = adapter_row.borrow_mut();

        // We must assign in reverse
        let timestamp = record.from_timestamp;

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

        adapter_row.from_state.pc = F::from_u32(record.from_pc);
        adapter_row.from_state.timestamp = F::from_u32(record.from_timestamp);
        adapter_row.rs1_ptr = F::from_u32(record.rs1_ptr);
        adapter_row.rs2_ptr = F::from_u32(record.rs2_ptr);
    }
}

#[cfg(test)]
impl Rv64BranchAdapterFiller {
    pub(crate) fn replay<F: PrimeField32>(
        postflight: &Postflight<'_, F>,
        step: PostflightStep,
        mem_helper: &MemoryAuxColsFactory<F>,
        adapter_row: &mut Rv64BranchAdapterCols<F>,
        next_pc: impl FnOnce(u32, [[u16; BLOCK_FE_WIDTH]; 2], u32) -> u32,
    ) -> Result<([[u16; BLOCK_FE_WIDTH]; 2], u32), PostflightError> {
        let instruction = postflight.instruction(step);
        if instruction.d.as_canonical_u32() != RV64_REGISTER_AS
            || instruction.e.as_canonical_u32() != RV64_REGISTER_AS
        {
            return Err(PostflightError::new(
                "branch instruction has invalid address spaces",
            ));
        }
        let from_pc = postflight.pc(step);
        let from_timestamp = postflight.timestamp(step);
        let rs1_ptr = instruction.a.as_canonical_u32();
        let rs2_ptr = instruction.b.as_canonical_u32();
        let immediate = instruction.c.as_canonical_u32();
        let rs1_u16_ptr = checked_byte_ptr_to_u16_ptr_value(rs1_ptr)?;
        let rs2_u16_ptr = checked_byte_ptr_to_u16_ptr_value(rs2_ptr)?;
        let mut replay = postflight.replay(step);
        let rs1 = replay.read_u16(RV64_REGISTER_AS, rs1_u16_ptr)?;
        let rs2 = replay.read_u16(RV64_REGISTER_AS, rs2_u16_ptr)?;
        let inputs = [rs1.value, rs2.value];
        let next_pc = next_pc(from_pc, inputs, immediate);
        replay.finish(next_pc)?;

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
        adapter_row.from_state.pc = F::from_u32(from_pc);
        adapter_row.from_state.timestamp = F::from_u32(from_timestamp);
        adapter_row.rs1_ptr = F::from_u32(rs1_ptr);
        adapter_row.rs2_ptr = F::from_u32(rs2_ptr);

        Ok((inputs, next_pc))
    }
}
