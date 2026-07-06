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
    byte_ptr_to_u16_ptr, byte_ptr_to_u16_ptr_value, imm_to_rv64_u64, tracing_read_u16,
    tracing_write_u16, U16_BITS,
};

/// Immediate-only variant of [`Rv64BaseAluU16AdapterCols`](super::Rv64BaseAluU16AdapterCols):
/// the second operand always comes from the instruction, so there is no `rs2_as` selector and
/// no second register read.
#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct Rv64BaseAluU16ImmAdapterCols<T> {
    pub from_state: ExecutionState<T>,
    pub rd_ptr: T,
    pub rs1_ptr: T,
    /// The 24-bit immediate operand (sign-extended 12-bit encoding from the transpiler).
    pub rs2_imm: T,
    /// Sign bit of the immediate: 0 if positive, 1 if negative.
    pub rs2_imm_sign: T,
    pub reads_aux: MemoryReadAuxCols<T>,
    pub writes_aux: MemoryWriteAuxCols<T, BLOCK_FE_WIDTH>,
}

/// Reads instructions of the form OP a, b, c, d, e where \[a\]_d = \[b\]_d op c, with c always
/// an immediate (d = 1, e = 0).
#[derive(Clone, Copy, Debug, derive_new::new, ColumnsAir)]
#[columns_via(Rv64BaseAluU16ImmAdapterCols<u8>)]
pub struct Rv64BaseAluU16ImmAdapterAir {
    pub(super) execution_bridge: ExecutionBridge,
    pub(super) memory_bridge: MemoryBridge,
    pub range_bus: VariableRangeCheckerBus,
}

impl<F: Field> BaseAir<F> for Rv64BaseAluU16ImmAdapterAir {
    fn width(&self) -> usize {
        Rv64BaseAluU16ImmAdapterCols::<F>::width()
    }
}

impl<AB: InteractionBuilder> VmAdapterAir<AB> for Rv64BaseAluU16ImmAdapterAir {
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
        let local: &Rv64BaseAluU16ImmAdapterCols<_> = local.borrow();
        let timestamp = local.from_state.timestamp;
        let mut timestamp_delta: usize = 0;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            timestamp + AB::F::from_usize(timestamp_delta - 1)
        };

        // The second read handed to the core is the sign-extended u16 decomposition of the
        // immediate operand:
        // 1. rs2_limbs[0] is the low u16 cell
        // 2. rs2_limbs[1..] are sign extension cells
        // 3. The 24-bit signed encoding reconstructs to local.rs2_imm
        // All rows in this chip use an immediate, so the constraints are unconditional (they are
        // trivially satisfied by all-zero padding rows).
        let rs2_limbs = ctx.reads[1].clone();
        let rs2_sign_u16 = local.rs2_imm_sign * AB::Expr::from_u32(u16::MAX as u32);
        builder.assert_bool(local.rs2_imm_sign);
        builder.assert_eq(
            local.rs2_imm,
            rs2_limbs[0].clone() + local.rs2_imm_sign * AB::Expr::from_u32(0xff_0000),
        );
        for limb in rs2_limbs.iter().skip(1) {
            builder.assert_eq(limb.clone(), rs2_sign_u16.clone());
        }
        // Range check the low u16 immediate limb.
        self.range_bus
            .range_check(rs2_limbs[0].clone(), U16_BITS)
            .eval(builder, ctx.instruction.is_valid.clone());

        // rs1 register read (timestamp slot 0).
        self.memory_bridge
            .read(
                MemoryAddress::new(
                    AB::F::from_u32(RV64_REGISTER_AS),
                    byte_ptr_to_u16_ptr::<AB>(local.rs1_ptr),
                ),
                ctx.reads[0].clone(),
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
                    local.rs2_imm.into(),
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
        let cols: &Rv64BaseAluU16ImmAdapterCols<_> = local.borrow();
        cols.from_state.pc
    }
}

#[derive(Clone, derive_new::new)]
pub struct Rv64BaseAluU16ImmAdapterExecutor;

#[derive(derive_new::new)]
pub struct Rv64BaseAluU16ImmAdapterFiller {
    pub range_checker_chip: SharedVariableRangeCheckerChip,
}

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug)]
pub struct Rv64BaseAluU16ImmAdapterRecord {
    pub from_pc: u32,
    pub from_timestamp: u32,
    pub rd_ptr: u32,
    pub rs1_ptr: u32,
    /// The 24-bit immediate operand.
    pub rs2_imm: u32,
    /// Sign bit of the immediate.
    pub rs2_imm_sign: bool,
    pub reads_aux: MemoryReadAuxRecord,
    pub writes_aux: MemoryWriteU16AuxRecord<BLOCK_FE_WIDTH>,
}

impl<F: PrimeField32> AdapterTraceExecutor<F> for Rv64BaseAluU16ImmAdapterExecutor {
    const WIDTH: usize = size_of::<Rv64BaseAluU16ImmAdapterCols<u8>>();
    type ReadData = [[u16; BLOCK_FE_WIDTH]; 2];
    type WriteData = [[u16; BLOCK_FE_WIDTH]; 1];
    type RecordMut<'a> = &'a mut Rv64BaseAluU16ImmAdapterRecord;

    #[inline(always)]
    fn start(pc: u32, memory: &TracingMemory, record: &mut &mut Rv64BaseAluU16ImmAdapterRecord) {
        record.from_pc = pc;
        record.from_timestamp = memory.timestamp;
    }

    #[inline(always)]
    fn read(
        &self,
        memory: &mut TracingMemory,
        instruction: &Instruction<F>,
        record: &mut &mut Rv64BaseAluU16ImmAdapterRecord,
    ) -> Self::ReadData {
        let &Instruction { b, c, d, e, .. } = instruction;

        debug_assert_eq!(d.as_canonical_u32(), RV64_REGISTER_AS);
        debug_assert_eq!(e.as_canonical_u32(), RV64_IMM_AS);

        record.rs1_ptr = b.as_canonical_u32();
        let rs1 = tracing_read_u16::<BLOCK_FE_WIDTH>(
            memory,
            RV64_REGISTER_AS,
            byte_ptr_to_u16_ptr_value(record.rs1_ptr),
            &mut record.reads_aux.prev_timestamp,
        );

        let imm = c.as_canonical_u32();
        record.rs2_imm = imm;
        let imm64 = imm_to_rv64_u64(imm);
        let sign_u16 = (imm64 >> U16_BITS) as u16;
        record.rs2_imm_sign = sign_u16 != 0;

        [rs1, [imm64 as u16, sign_u16, sign_u16, sign_u16]]
    }

    #[inline(always)]
    fn write(
        &self,
        memory: &mut TracingMemory,
        instruction: &Instruction<F>,
        data: Self::WriteData,
        record: &mut &mut Rv64BaseAluU16ImmAdapterRecord,
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

impl<F: PrimeField32> AdapterTraceFiller<F> for Rv64BaseAluU16ImmAdapterFiller {
    const WIDTH: usize = size_of::<Rv64BaseAluU16ImmAdapterCols<u8>>();

    #[inline(always)]
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, mut adapter_row: &mut [F]) {
        // SAFETY:
        // - caller ensures `adapter_row` contains a valid record representation
        // - get_record_from_slice correctly interprets the bytes as Rv64BaseAluU16ImmAdapterRecord
        let record: &Rv64BaseAluU16ImmAdapterRecord =
            unsafe { get_record_from_slice(&mut adapter_row, ()) };
        let adapter_row: &mut Rv64BaseAluU16ImmAdapterCols<F> = adapter_row.borrow_mut();

        self.range_checker_chip
            .add_count(record.rs2_imm & (u16::MAX as u32), U16_BITS);

        adapter_row
            .writes_aux
            .set_prev_data(record.writes_aux.prev_data.map(F::from_u16));
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

        adapter_row.rs2_imm_sign = F::from_bool(record.rs2_imm_sign);
        adapter_row.rs2_imm = F::from_u32(record.rs2_imm);
        adapter_row.rs1_ptr = F::from_u32(record.rs1_ptr);
        adapter_row.rd_ptr = F::from_u32(record.rd_ptr);

        adapter_row.from_state.timestamp = F::from_u32(record.from_timestamp);
        adapter_row.from_state.pc = F::from_u32(record.from_pc);
    }
}
