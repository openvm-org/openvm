use std::{
    array,
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use openvm_circuit::{
    arch::{
        get_record_from_slice, AdapterAirContext, AdapterTraceExecutor, AdapterTraceFiller,
        BasicAdapterInterface, ExecutionBridge, ExecutionState, ImmInstruction, VmAdapterAir,
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
    var_range::{SharedVariableRangeCheckerChip, VariableRangeCheckerBus},
    AlignedBytesBorrow, ColumnsAir, StructReflection, StructReflectionHelper,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV64_IMM_AS, RV64_REGISTER_AS},
};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, PrimeCharacteristicRing, PrimeField32},
};

use super::{
    byte_ptr_to_u16_ptr, byte_ptr_to_u16_ptr_value, concat_rv64_u16_block, tracing_read_u16,
    tracing_write_u16, RV64_WORD_U16_LIMBS, U16_BITS,
};

/// Adapter columns for RV64 word instructions with an immediate operand.
///
/// The core sees only the low 32-bit word as two u16 limbs. The upper half of the source register
/// is retained solely to authenticate the full-width register read, while the full-width write is
/// rebuilt by sign-extending the core result.
#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct Rv64BaseAluWImmU16AdapterCols<T> {
    pub from_state: ExecutionState<T>,
    pub rd_ptr: T,
    pub rs1_ptr: T,
    pub rs1_high: [T; RV64_WORD_U16_LIMBS],
    pub result_sign: T,
    pub reads_aux: MemoryReadAuxCols<T>,
    pub writes_aux: MemoryWriteAuxCols<T, BLOCK_FE_WIDTH>,
}

const _: () = assert!(size_of::<Rv64BaseAluWImmU16AdapterCols<u8>>() == 17);

#[derive(Clone, Copy, Debug, derive_new::new, ColumnsAir)]
#[columns_via(Rv64BaseAluWImmU16AdapterCols<u8>)]
pub struct Rv64BaseAluWImmU16AdapterAir {
    pub(super) execution_bridge: ExecutionBridge,
    pub(super) memory_bridge: MemoryBridge,
    pub range_bus: VariableRangeCheckerBus,
}

impl<F: Field> BaseAir<F> for Rv64BaseAluWImmU16AdapterAir {
    fn width(&self) -> usize {
        Rv64BaseAluWImmU16AdapterCols::<F>::width()
    }
}

impl<AB: InteractionBuilder> VmAdapterAir<AB> for Rv64BaseAluWImmU16AdapterAir {
    type Interface = BasicAdapterInterface<
        AB::Expr,
        ImmInstruction<AB::Expr>,
        1,
        1,
        RV64_WORD_U16_LIMBS,
        RV64_WORD_U16_LIMBS,
    >;

    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        let local: &Rv64BaseAluWImmU16AdapterCols<_> = local.borrow();
        let timestamp = local.from_state.timestamp;
        let mut timestamp_delta = 0usize;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            timestamp + AB::F::from_usize(timestamp_delta - 1)
        };

        let rs1_data: [AB::Expr; BLOCK_FE_WIDTH] =
            concat_rv64_u16_block(&ctx.reads[0], &local.rs1_high);
        self.memory_bridge
            .read(
                MemoryAddress::new(
                    AB::F::from_u32(RV64_REGISTER_AS),
                    byte_ptr_to_u16_ptr::<AB>(local.rs1_ptr),
                ),
                rs1_data,
                timestamp_pp(),
                &local.reads_aux,
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        // Recover the sign of the 32-bit result from its top u16 limb. This simultaneously proves
        // that the top result limb is a canonical u16 value.
        builder.assert_bool(local.result_sign);
        let result_high = ctx.writes[0][RV64_WORD_U16_LIMBS - 1].clone();
        self.range_bus
            .range_check(
                result_high - local.result_sign * AB::Expr::from_u32(1 << (U16_BITS - 1)),
                U16_BITS - 1,
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        let sign_extend = local.result_sign * AB::Expr::from_u32(u16::MAX as u32);
        let sign_extend_limbs: [AB::Expr; RV64_WORD_U16_LIMBS] =
            array::from_fn(|_| sign_extend.clone());
        let write_data: [AB::Expr; BLOCK_FE_WIDTH] =
            concat_rv64_u16_block(&ctx.writes[0], &sign_extend_limbs);
        self.memory_bridge
            .write(
                MemoryAddress::new(
                    AB::F::from_u32(RV64_REGISTER_AS),
                    byte_ptr_to_u16_ptr::<AB>(local.rd_ptr),
                ),
                write_data,
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
        let local: &Rv64BaseAluWImmU16AdapterCols<_> = local.borrow();
        local.from_state.pc
    }
}

#[derive(Clone, derive_new::new)]
pub struct Rv64BaseAluWImmU16AdapterExecutor;

#[derive(derive_new::new)]
pub struct Rv64BaseAluWImmU16AdapterFiller {
    pub range_checker_chip: SharedVariableRangeCheckerChip,
}

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug)]
pub struct Rv64BaseAluWImmU16AdapterRecord {
    pub from_pc: u32,
    pub from_timestamp: u32,
    pub rd_ptr: u32,
    pub rs1_ptr: u32,
    pub rs1_high: [u16; RV64_WORD_U16_LIMBS],
    pub result_high: u16,
    pub result_sign: u8,
    pub reads_aux: MemoryReadAuxRecord,
    pub writes_aux: MemoryWriteU16AuxRecord<BLOCK_FE_WIDTH>,
}

const _: () = assert!(size_of::<Rv64BaseAluWImmU16AdapterRecord>() == 40);

impl<F: PrimeField32> AdapterTraceExecutor<F> for Rv64BaseAluWImmU16AdapterExecutor {
    const WIDTH: usize = size_of::<Rv64BaseAluWImmU16AdapterCols<u8>>();
    type ReadData = [[u16; RV64_WORD_U16_LIMBS]; 1];
    type WriteData = [[u16; RV64_WORD_U16_LIMBS]; 1];
    type RecordMut<'a> = &'a mut Rv64BaseAluWImmU16AdapterRecord;

    #[inline(always)]
    fn start(pc: u32, memory: &TracingMemory, record: &mut &mut Rv64BaseAluWImmU16AdapterRecord) {
        record.from_pc = pc;
        record.from_timestamp = memory.timestamp;
    }

    #[inline(always)]
    fn read(
        &self,
        memory: &mut TracingMemory,
        instruction: &Instruction<F>,
        record: &mut &mut Rv64BaseAluWImmU16AdapterRecord,
    ) -> Self::ReadData {
        let &Instruction { b, d, e, .. } = instruction;
        debug_assert_eq!(d.as_canonical_u32(), RV64_REGISTER_AS);
        debug_assert_eq!(e.as_canonical_u32(), RV64_IMM_AS);

        record.rs1_ptr = b.as_canonical_u32();
        let rs1_full = tracing_read_u16::<BLOCK_FE_WIDTH>(
            memory,
            RV64_REGISTER_AS,
            byte_ptr_to_u16_ptr_value(record.rs1_ptr),
            &mut record.reads_aux.prev_timestamp,
        );
        record.rs1_high = array::from_fn(|i| rs1_full[RV64_WORD_U16_LIMBS + i]);
        [array::from_fn(|i| rs1_full[i])]
    }

    #[inline(always)]
    fn write(
        &self,
        memory: &mut TracingMemory,
        instruction: &Instruction<F>,
        data: Self::WriteData,
        record: &mut &mut Rv64BaseAluWImmU16AdapterRecord,
    ) {
        let &Instruction { a, d, .. } = instruction;
        debug_assert_eq!(d.as_canonical_u32(), RV64_REGISTER_AS);

        record.rd_ptr = a.as_canonical_u32();
        let write_low = data[0];
        record.result_high = write_low[RV64_WORD_U16_LIMBS - 1];
        record.result_sign = (record.result_high >> (U16_BITS - 1)) as u8;
        let sign_extend_limb = if record.result_sign != 0 { u16::MAX } else { 0 };
        let write_data: [u16; BLOCK_FE_WIDTH] = array::from_fn(|i| {
            if i < RV64_WORD_U16_LIMBS {
                write_low[i]
            } else {
                sign_extend_limb
            }
        });
        tracing_write_u16(
            memory,
            RV64_REGISTER_AS,
            byte_ptr_to_u16_ptr_value(record.rd_ptr),
            write_data,
            &mut record.writes_aux.prev_timestamp,
            &mut record.writes_aux.prev_data,
        );
    }
}

impl<F: PrimeField32> AdapterTraceFiller<F> for Rv64BaseAluWImmU16AdapterFiller {
    const WIDTH: usize = size_of::<Rv64BaseAluWImmU16AdapterCols<u8>>();

    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, mut adapter_row: &mut [F]) {
        let record: &Rv64BaseAluWImmU16AdapterRecord =
            unsafe { get_record_from_slice(&mut adapter_row, ()) };

        // Copy the overlaid record before writing the wider field-element column representation.
        let from_pc = record.from_pc;
        let from_timestamp = record.from_timestamp;
        let rd_ptr = record.rd_ptr;
        let rs1_ptr = record.rs1_ptr;
        let rs1_high = record.rs1_high;
        let result_high = record.result_high;
        let result_sign = record.result_sign;
        let reads_aux_prev_timestamp = record.reads_aux.prev_timestamp;
        let writes_aux_prev_timestamp = record.writes_aux.prev_timestamp;
        let writes_aux_prev_data = record.writes_aux.prev_data;

        let result_low15 = (result_high & ((1 << (U16_BITS - 1)) - 1)) as u32;
        self.range_checker_chip
            .add_count(result_low15, U16_BITS - 1);

        let adapter_row: &mut Rv64BaseAluWImmU16AdapterCols<F> = adapter_row.borrow_mut();
        adapter_row
            .writes_aux
            .set_prev_data(writes_aux_prev_data.map(F::from_u16));
        mem_helper.fill(
            writes_aux_prev_timestamp,
            from_timestamp + 1,
            adapter_row.writes_aux.as_mut(),
        );
        mem_helper.fill(
            reads_aux_prev_timestamp,
            from_timestamp,
            adapter_row.reads_aux.as_mut(),
        );

        adapter_row.result_sign = F::from_u8(result_sign);
        adapter_row.rs1_high = rs1_high.map(F::from_u16);
        adapter_row.rs1_ptr = F::from_u32(rs1_ptr);
        adapter_row.rd_ptr = F::from_u32(rd_ptr);
        adapter_row.from_state.timestamp = F::from_u32(from_timestamp);
        adapter_row.from_state.pc = F::from_u32(from_pc);
    }
}
