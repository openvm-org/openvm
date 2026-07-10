use std::borrow::{Borrow, BorrowMut};

use openvm_circuit::{
    arch::{
        get_record_from_slice, AdapterAirContext, AdapterTraceExecutor, AdapterTraceFiller,
        BasicAdapterInterface, ExecutionBridge, ExecutionState, MinimalInstruction, VmAdapterAir,
        BLOCK_FE_WIDTH,
    },
    system::memory::{
        offline_checker::{
            MemoryBridge, MemoryReadAuxCols, MemoryReadAuxRecord, MemoryWriteAuxCols,
            MemoryWriteU16AuxRecord,
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

use super::{byte_ptr_to_u16_ptr, byte_ptr_to_u16_ptr_value, tracing_read_u16, tracing_write_u16};

/// Adapter columns for register-only base ALU ops (e.g. ADD/SUB).
/// Both rs1 and rs2 are always register reads; there is no immediate path.
/// Compared to Rv64BaseAluU16AdapterCols, the `rs2_as` and `rs2_imm_sign` columns are removed.
#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct Rv64RegBaseAluU16AdapterCols<T> {
    pub from_state: ExecutionState<T>,
    pub rd_ptr: T,
    pub rs1_ptr: T,
    pub rs2_ptr: T,
    pub reads_aux: [MemoryReadAuxCols<T>; 2],
    pub writes_aux: MemoryWriteAuxCols<T, BLOCK_FE_WIDTH>,
}

/// Adapter AIR for register-only base ALU ops — no range_bus needed because there is no
/// immediate to range-check.
#[derive(Clone, Copy, Debug, derive_new::new, ColumnsAir)]
#[columns_via(Rv64RegBaseAluU16AdapterCols<u8>)]
pub struct Rv64RegBaseAluU16AdapterAir {
    pub(super) execution_bridge: ExecutionBridge,
    pub(super) memory_bridge: MemoryBridge,
}

impl<F: Field> BaseAir<F> for Rv64RegBaseAluU16AdapterAir {
    fn width(&self) -> usize {
        Rv64RegBaseAluU16AdapterCols::<F>::width()
    }
}

impl<AB: InteractionBuilder> VmAdapterAir<AB> for Rv64RegBaseAluU16AdapterAir {
    type Interface = BasicAdapterInterface<
        AB::Expr,
        MinimalInstruction<AB::Expr>,
        2,
        1,
        BLOCK_FE_WIDTH,
        BLOCK_FE_WIDTH,
    >;

    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        let local: &Rv64RegBaseAluU16AdapterCols<_> = local.borrow();
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

        self.memory_bridge
            .write(
                MemoryAddress::new(
                    AB::F::from_u32(RV64_REGISTER_AS),
                    byte_ptr_to_u16_ptr::<AB>(local.rd_ptr),
                ),
                ctx.writes[0].clone(),
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
                    AB::Expr::from_u32(RV64_REGISTER_AS),
                ],
                local.from_state,
                AB::F::from_usize(timestamp_delta),
                (DEFAULT_PC_STEP, ctx.to_pc),
            )
            .eval(builder, ctx.instruction.is_valid);
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        let cols: &Rv64RegBaseAluU16AdapterCols<_> = local.borrow();
        cols.from_state.pc
    }
}

#[derive(Clone, derive_new::new)]
pub struct Rv64RegBaseAluU16AdapterExecutor;

/// No range_checker_chip — no immediate limb to range-check.
#[derive(Clone, Copy, Default, derive_new::new)]
pub struct Rv64RegBaseAluU16AdapterFiller;

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug)]
pub struct Rv64RegBaseAluU16AdapterRecord {
    pub from_pc: u32,
    pub from_timestamp: u32,
    pub rd_ptr: u32,
    pub rs1_ptr: u32,
    pub rs2_ptr: u32,
    pub reads_aux: [MemoryReadAuxRecord; 2],
    pub writes_aux: MemoryWriteU16AuxRecord<BLOCK_FE_WIDTH>,
}

impl<F: PrimeField32> AdapterTraceExecutor<F> for Rv64RegBaseAluU16AdapterExecutor {
    const WIDTH: usize = size_of::<Rv64RegBaseAluU16AdapterCols<u8>>();
    type ReadData = [[u16; BLOCK_FE_WIDTH]; 2];
    type WriteData = [[u16; BLOCK_FE_WIDTH]; 1];
    type RecordMut<'a> = &'a mut Rv64RegBaseAluU16AdapterRecord;

    #[inline(always)]
    fn start(pc: u32, memory: &TracingMemory, record: &mut &mut Rv64RegBaseAluU16AdapterRecord) {
        record.from_pc = pc;
        record.from_timestamp = memory.timestamp;
    }

    #[inline(always)]
    fn read(
        &self,
        memory: &mut TracingMemory,
        instruction: &Instruction<F>,
        record: &mut &mut Rv64RegBaseAluU16AdapterRecord,
    ) -> Self::ReadData {
        let &Instruction { b, c, d, e, .. } = instruction;

        debug_assert_eq!(d.as_canonical_u32(), RV64_REGISTER_AS);
        debug_assert_eq!(e.as_canonical_u32(), RV64_REGISTER_AS);

        record.rs1_ptr = b.as_canonical_u32();
        let rs1 = tracing_read_u16::<BLOCK_FE_WIDTH>(
            memory,
            RV64_REGISTER_AS,
            byte_ptr_to_u16_ptr_value(record.rs1_ptr),
            &mut record.reads_aux[0].prev_timestamp,
        );

        record.rs2_ptr = c.as_canonical_u32();
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
        memory: &mut TracingMemory,
        instruction: &Instruction<F>,
        data: Self::WriteData,
        record: &mut &mut Rv64RegBaseAluU16AdapterRecord,
    ) {
        let &Instruction { a, d, .. } = instruction;

        debug_assert_eq!(d.as_canonical_u32(), RV64_REGISTER_AS);

        record.rd_ptr = a.as_canonical_u32();
        tracing_write_u16(
            memory,
            RV64_REGISTER_AS,
            byte_ptr_to_u16_ptr_value(record.rd_ptr),
            data[0],
            &mut record.writes_aux.prev_timestamp,
            &mut record.writes_aux.prev_data,
        );
    }
}

impl<F: PrimeField32> AdapterTraceFiller<F> for Rv64RegBaseAluU16AdapterFiller {
    const WIDTH: usize = size_of::<Rv64RegBaseAluU16AdapterCols<u8>>();

    #[inline(always)]
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, mut adapter_row: &mut [F]) {
        // SAFETY: caller ensures `adapter_row` contains a valid record representation
        let record: &Rv64RegBaseAluU16AdapterRecord =
            unsafe { get_record_from_slice(&mut adapter_row, ()) };
        let adapter_row: &mut Rv64RegBaseAluU16AdapterCols<F> = adapter_row.borrow_mut();

        adapter_row
            .writes_aux
            .set_prev_data(record.writes_aux.prev_data.map(F::from_u16));
        mem_helper.fill(
            record.writes_aux.prev_timestamp,
            record.from_timestamp + 2,
            adapter_row.writes_aux.as_mut(),
        );

        mem_helper.fill(
            record.reads_aux[1].prev_timestamp,
            record.from_timestamp + 1,
            adapter_row.reads_aux[1].as_mut(),
        );

        mem_helper.fill(
            record.reads_aux[0].prev_timestamp,
            record.from_timestamp,
            adapter_row.reads_aux[0].as_mut(),
        );

        adapter_row.rs2_ptr = F::from_u32(record.rs2_ptr);
        adapter_row.rs1_ptr = F::from_u32(record.rs1_ptr);
        adapter_row.rd_ptr = F::from_u32(record.rd_ptr);

        adapter_row.from_state.timestamp = F::from_u32(record.from_timestamp);
        adapter_row.from_state.pc = F::from_u32(record.from_pc);
    }
}
