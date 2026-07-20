use std::borrow::{Borrow, BorrowMut};

use openvm_circuit::{
    arch::{
        get_record_from_slice, AdapterAirContext, AdapterTraceExecutor, AdapterTraceFiller,
        BasicAdapterInterface, ExecutionBridge, ExecutionState, ImmInstruction, VmAdapterAir,
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
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV64_IMM_AS, RV64_REGISTER_AS, RV64_REGISTER_NUM_LIMBS},
};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, PrimeCharacteristicRing, PrimeField32},
};

use super::{byte_ptr_to_u16_ptr, tracing_read, tracing_write};

/// Immediate-only byte-limb adapter (single register read + register write). The immediate
/// itself lives in the core, which passes it back as the `ImmInstruction` immediate expression.
#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct Rv64BaseAluImmAdapterCols<T> {
    pub from_state: ExecutionState<T>,
    pub rd_ptr: T,
    pub rs1_ptr: T,
    pub reads_aux: MemoryReadAuxCols<T>,
    pub writes_aux: MemoryWriteAuxCols<T, BLOCK_FE_WIDTH>,
}

/// Reads instructions of the form OP a, b, c, d, e where \[a:4\]_d = \[b:4\]_d op c, with c
/// always an immediate (d = 1, e = 0).
#[derive(Clone, Copy, Debug, derive_new::new, ColumnsAir)]
#[columns_via(Rv64BaseAluImmAdapterCols<u8>)]
pub struct Rv64BaseAluImmAdapterAir {
    pub(super) execution_bridge: ExecutionBridge,
    pub(super) memory_bridge: MemoryBridge,
}

impl<F: Field> BaseAir<F> for Rv64BaseAluImmAdapterAir {
    fn width(&self) -> usize {
        Rv64BaseAluImmAdapterCols::<F>::width()
    }
}

impl<AB: InteractionBuilder> VmAdapterAir<AB> for Rv64BaseAluImmAdapterAir {
    type Interface = BasicAdapterInterface<
        AB::Expr,
        ImmInstruction<AB::Expr>,
        1,
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
        let local: &Rv64BaseAluImmAdapterCols<_> = local.borrow();
        let timestamp = local.from_state.timestamp;
        let mut timestamp_delta: usize = 0;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            timestamp + AB::F::from_usize(timestamp_delta - 1)
        };

        // rs1 register read (timestamp slot 0).
        self.memory_bridge
            .read(
                MemoryAddress::new(
                    AB::F::from_u32(RV64_REGISTER_AS),
                    byte_ptr_to_u16_ptr::<AB>(local.rs1_ptr),
                ),
                pack_u8_block::<AB>(&ctx.reads[0].clone()),
                timestamp_pp(),
                &local.reads_aux,
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        // rd write (timestamp slot 1).
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
                    ctx.instruction.immediate,
                    AB::Expr::from_u32(RV64_REGISTER_AS),
                    AB::Expr::from_u32(RV64_IMM_AS),
                ],
                local.from_state,
                AB::F::from_usize(timestamp_delta),
                (DEFAULT_PC_STEP, ctx.to_pc),
            )
            .eval(builder, ctx.instruction.is_valid);
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        let cols: &Rv64BaseAluImmAdapterCols<_> = local.borrow();
        cols.from_state.pc
    }
}

#[derive(Clone, derive_new::new)]
pub struct Rv64BaseAluImmAdapterExecutor;

#[derive(Clone, derive_new::new)]
pub struct Rv64BaseAluImmAdapterFiller;

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug)]
pub struct Rv64BaseAluImmAdapterRecord {
    pub from_pc: u32,
    pub from_timestamp: u32,
    pub rd_ptr: u32,
    pub rs1_ptr: u32,
    pub reads_aux: MemoryReadAuxRecord,
    pub writes_aux: MemoryWriteBytesAuxRecord<RV64_REGISTER_NUM_LIMBS>,
}

impl<F: PrimeField32> AdapterTraceExecutor<F> for Rv64BaseAluImmAdapterExecutor {
    const WIDTH: usize = size_of::<Rv64BaseAluImmAdapterCols<u8>>();
    type ReadData = [[u8; RV64_REGISTER_NUM_LIMBS]; 1];
    type WriteData = [[u8; RV64_REGISTER_NUM_LIMBS]; 1];
    type RecordMut<'a> = &'a mut Rv64BaseAluImmAdapterRecord;

    #[inline(always)]
    fn start(pc: u32, memory: &TracingMemory, record: &mut &mut Rv64BaseAluImmAdapterRecord) {
        record.from_pc = pc;
        record.from_timestamp = memory.timestamp;
    }

    #[inline(always)]
    fn read(
        &self,
        memory: &mut TracingMemory,
        instruction: &Instruction<F>,
        record: &mut &mut Rv64BaseAluImmAdapterRecord,
    ) -> Self::ReadData {
        let &Instruction { b, d, e, .. } = instruction;

        debug_assert_eq!(d.as_canonical_u32(), RV64_REGISTER_AS);
        debug_assert_eq!(e.as_canonical_u32(), RV64_IMM_AS);

        record.rs1_ptr = b.as_canonical_u32();
        let rs1 = tracing_read(
            memory,
            RV64_REGISTER_AS,
            record.rs1_ptr,
            &mut record.reads_aux.prev_timestamp,
        );

        [rs1]
    }

    #[inline(always)]
    fn write(
        &self,
        memory: &mut TracingMemory,
        instruction: &Instruction<F>,
        data: Self::WriteData,
        record: &mut &mut Rv64BaseAluImmAdapterRecord,
    ) {
        let &Instruction { a, d, .. } = instruction;

        debug_assert_eq!(d.as_canonical_u32(), RV64_REGISTER_AS);

        record.rd_ptr = a.as_canonical_u32();
        tracing_write(
            memory,
            RV64_REGISTER_AS,
            record.rd_ptr,
            data[0],
            &mut record.writes_aux.prev_timestamp,
            &mut record.writes_aux.prev_data,
        );
    }
}

impl<F: PrimeField32> AdapterTraceFiller<F> for Rv64BaseAluImmAdapterFiller {
    const WIDTH: usize = size_of::<Rv64BaseAluImmAdapterCols<u8>>();

    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, mut adapter_row: &mut [F]) {
        // SAFETY:
        // - caller ensures `adapter_row` contains a valid record representation
        // - get_record_from_slice correctly interprets the bytes as Rv64BaseAluImmAdapterRecord
        let record: &Rv64BaseAluImmAdapterRecord =
            unsafe { get_record_from_slice(&mut adapter_row, ()) };
        let adapter_row: &mut Rv64BaseAluImmAdapterCols<F> = adapter_row.borrow_mut();

        adapter_row
            .writes_aux
            .set_prev_data(pack_u8_block_bytes(&record.writes_aux.prev_data));
        // Write is at timestamp slot 1 (after the rs1 read at slot 0).
        mem_helper.fill(
            record.writes_aux.prev_timestamp,
            record.from_timestamp + 1,
            adapter_row.writes_aux.as_mut(),
        );

        mem_helper.fill(
            record.reads_aux.prev_timestamp,
            record.from_timestamp,
            adapter_row.reads_aux.as_mut(),
        );

        adapter_row.rs1_ptr = F::from_u32(record.rs1_ptr);
        adapter_row.rd_ptr = F::from_u32(record.rd_ptr);
        adapter_row.from_state.timestamp = F::from_u32(record.from_timestamp);
        adapter_row.from_state.pc = F::from_u32(record.from_pc);
    }
}
