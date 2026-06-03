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
    utils::not,
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
    p3_air::{AirBuilder, BaseAir},
    p3_field::{Field, PrimeCharacteristicRing, PrimeField32},
};

use super::{byte_ptr_to_u16_ptr, byte_ptr_to_u16_ptr_value, tracing_read_u16, tracing_write_u16};
use crate::adapters::{imm_to_rv64_u64, U16_BITS};

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct Rv64BaseAluU16AdapterCols<T> {
    pub from_state: ExecutionState<T>,
    pub rd_ptr: T,
    pub rs1_ptr: T,
    /// Pointer if rs2 was a read, immediate value otherwise
    pub rs2: T,
    /// 1 if rs2 was a read, 0 if an immediate
    pub rs2_as: T,
    /// Sign bit of `rs2` when it is an immediate: 0 if positive, 1 if negative. Unused (= 0)
    /// when `rs2_as == RV64_REGISTER_AS`.
    pub rs2_imm_sign: T,
    pub reads_aux: [MemoryReadAuxCols<T>; 2],
    pub writes_aux: MemoryWriteAuxCols<T, BLOCK_FE_WIDTH>,
}

/// Reads instructions of the form OP a, b, c, d, e where \[a\]_d = \[b\]_d op \[c\]_e.
/// Operand d can only be 1, and e can be either 1 (for register reads) or 0 (when c
/// is an immediate).
#[derive(Clone, Copy, Debug, derive_new::new, ColumnsAir)]
#[columns_via(Rv64BaseAluU16AdapterCols<u8>)]
pub struct Rv64BaseAluU16AdapterAir {
    pub(super) execution_bridge: ExecutionBridge,
    pub(super) memory_bridge: MemoryBridge,
    pub range_bus: VariableRangeCheckerBus,
}

impl<F: Field> BaseAir<F> for Rv64BaseAluU16AdapterAir {
    fn width(&self) -> usize {
        Rv64BaseAluU16AdapterCols::<F>::width()
    }
}

impl<AB: InteractionBuilder> VmAdapterAir<AB> for Rv64BaseAluU16AdapterAir {
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
        let local: &Rv64BaseAluU16AdapterCols<_> = local.borrow();
        let timestamp = local.from_state.timestamp;
        let mut timestamp_delta: usize = 0;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            timestamp + AB::F::from_usize(timestamp_delta - 1)
        };

        // If rs2 is an immediate value, constrain that:
        // 1. rs2_limbs[0] is the low u16 cell
        // 2. rs2_limbs[1..] are sign extension cells
        // 3. The 24-bit signed encoding reconstructs to local.rs2
        let rs2_limbs = ctx.reads[1].clone();
        let rs2_sign_u16 = local.rs2_imm_sign * AB::Expr::from_u32(u16::MAX as u32);
        let rs2_low_u16 = rs2_limbs[0].clone();
        let rs2_high_u16 = rs2_limbs[1].clone();

        builder.assert_bool(local.rs2_as);
        builder.assert_bool(local.rs2_imm_sign);

        let mut rs2_imm_when = builder.when(not(local.rs2_as));
        rs2_imm_when.assert_eq(
            local.rs2,
            rs2_low_u16.clone() + local.rs2_imm_sign * AB::Expr::from_u32(0xff_0000),
        );
        rs2_imm_when.assert_eq(rs2_high_u16.clone(), rs2_sign_u16.clone());
        for limb in rs2_limbs.iter().skip(2) {
            rs2_imm_when.assert_eq(limb.clone(), rs2_sign_u16.clone());
        }
        // Range check the low u16 immediate limb.
        self.range_bus
            .range_check(rs2_low_u16, U16_BITS)
            .eval(builder, ctx.instruction.is_valid.clone() - local.rs2_as);

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

        builder
            .when(local.rs2_as)
            .assert_one(ctx.instruction.is_valid.clone());
        self.memory_bridge
            .read(
                MemoryAddress::new(local.rs2_as, byte_ptr_to_u16_ptr::<AB>(local.rs2)),
                ctx.reads[1].clone(),
                timestamp_pp(),
                &local.reads_aux[1],
            )
            .eval(builder, local.rs2_as);

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
                    local.rs2.into(),
                    AB::Expr::from_u32(RV64_REGISTER_AS),
                    local.rs2_as.into(),
                ],
                local.from_state,
                AB::F::from_usize(timestamp_delta),
                (DEFAULT_PC_STEP, ctx.to_pc),
            )
            .eval(builder, ctx.instruction.is_valid);
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        let cols: &Rv64BaseAluU16AdapterCols<_> = local.borrow();
        cols.from_state.pc
    }
}

#[derive(Clone, derive_new::new)]
pub struct Rv64BaseAluU16AdapterExecutor;

#[derive(derive_new::new)]
pub struct Rv64BaseAluU16AdapterFiller {
    pub range_checker_chip: SharedVariableRangeCheckerChip,
}

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug)]
pub struct Rv64BaseAluU16AdapterRecord {
    pub from_pc: u32,
    pub from_timestamp: u32,
    pub rd_ptr: u32,
    pub rs1_ptr: u32,
    /// Pointer if rs2 was a read, immediate value otherwise.
    pub rs2: u32,
    /// 1 if rs2 was a read, 0 if an immediate.
    pub rs2_as: u8,
    /// Sign bit of the immediate; unused when `rs2_as == RV64_REGISTER_AS`.
    pub rs2_imm_sign: bool,
    pub reads_aux: [MemoryReadAuxRecord; 2],
    pub writes_aux: MemoryWriteU16AuxRecord<BLOCK_FE_WIDTH>,
}

impl<F: PrimeField32> AdapterTraceExecutor<F> for Rv64BaseAluU16AdapterExecutor {
    const WIDTH: usize = size_of::<Rv64BaseAluU16AdapterCols<u8>>();
    type ReadData = [[u16; BLOCK_FE_WIDTH]; 2];
    type WriteData = [[u16; BLOCK_FE_WIDTH]; 1];
    type RecordMut<'a> = &'a mut Rv64BaseAluU16AdapterRecord;

    #[inline(always)]
    fn start(pc: u32, memory: &TracingMemory, record: &mut &mut Rv64BaseAluU16AdapterRecord) {
        record.from_pc = pc;
        record.from_timestamp = memory.timestamp;
    }

    #[inline(always)]
    fn read(
        &self,
        memory: &mut TracingMemory,
        instruction: &Instruction<F>,
        record: &mut &mut Rv64BaseAluU16AdapterRecord,
    ) -> Self::ReadData {
        let &Instruction { b, c, d, e, .. } = instruction;

        debug_assert_eq!(d.as_canonical_u32(), RV64_REGISTER_AS);
        debug_assert!(
            e.as_canonical_u32() == RV64_REGISTER_AS || e.as_canonical_u32() == RV64_IMM_AS
        );

        record.rs1_ptr = b.as_canonical_u32();
        let rs1 = tracing_read_u16::<BLOCK_FE_WIDTH>(
            memory,
            RV64_REGISTER_AS,
            byte_ptr_to_u16_ptr_value(record.rs1_ptr),
            &mut record.reads_aux[0].prev_timestamp,
        );

        let rs2 = if e.as_canonical_u32() == RV64_REGISTER_AS {
            record.rs2_as = RV64_REGISTER_AS as u8;
            record.rs2_imm_sign = false;
            record.rs2 = c.as_canonical_u32();

            tracing_read_u16::<BLOCK_FE_WIDTH>(
                memory,
                RV64_REGISTER_AS,
                byte_ptr_to_u16_ptr_value(record.rs2),
                &mut record.reads_aux[1].prev_timestamp,
            )
        } else {
            record.rs2_as = RV64_IMM_AS as u8;
            let imm = c.as_canonical_u32();
            record.rs2 = imm;
            let imm64 = imm_to_rv64_u64(imm);
            let sign_u16 = (imm64 >> U16_BITS) as u16;
            record.rs2_imm_sign = sign_u16 != 0;
            memory.increment_timestamp();
            [imm64 as u16, sign_u16, sign_u16, sign_u16]
        };

        [rs1, rs2]
    }

    #[inline(always)]
    fn write(
        &self,
        memory: &mut TracingMemory,
        instruction: &Instruction<F>,
        data: Self::WriteData,
        record: &mut &mut Rv64BaseAluU16AdapterRecord,
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

impl<F: PrimeField32> AdapterTraceFiller<F> for Rv64BaseAluU16AdapterFiller {
    const WIDTH: usize = size_of::<Rv64BaseAluU16AdapterCols<u8>>();

    #[inline(always)]
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, mut adapter_row: &mut [F]) {
        // SAFETY:
        // - caller ensures `adapter_row` contains a valid record representation
        // - get_record_from_slice correctly interprets the bytes as Rv64BaseAluU16AdapterRecord
        let record: &Rv64BaseAluU16AdapterRecord =
            unsafe { get_record_from_slice(&mut adapter_row, ()) };
        let adapter_row: &mut Rv64BaseAluU16AdapterCols<F> = adapter_row.borrow_mut();

        if record.rs2_as as u32 == RV64_IMM_AS {
            let imm_low_u16 = record.rs2 & (u16::MAX as u32);
            self.range_checker_chip.add_count(imm_low_u16, U16_BITS);
        }

        adapter_row
            .writes_aux
            .set_prev_data(record.writes_aux.prev_data.map(F::from_u16));
        mem_helper.fill(
            record.writes_aux.prev_timestamp,
            record.from_timestamp + 2,
            adapter_row.writes_aux.as_mut(),
        );

        if record.rs2_as as u32 == RV64_IMM_AS {
            mem_helper.fill_zero(adapter_row.reads_aux[1].as_mut());
        } else {
            mem_helper.fill(
                record.reads_aux[1].prev_timestamp,
                record.from_timestamp + 1,
                adapter_row.reads_aux[1].as_mut(),
            );
        }

        mem_helper.fill(
            record.reads_aux[0].prev_timestamp,
            record.from_timestamp,
            adapter_row.reads_aux[0].as_mut(),
        );

        adapter_row.rs2_imm_sign = F::from_bool(record.rs2_imm_sign);
        adapter_row.rs2_as = F::from_u8(record.rs2_as);
        adapter_row.rs2 = F::from_u32(record.rs2);
        adapter_row.rs1_ptr = F::from_u32(record.rs1_ptr);
        adapter_row.rd_ptr = F::from_u32(record.rd_ptr);

        adapter_row.from_state.timestamp = F::from_u32(record.from_timestamp);
        adapter_row.from_state.pc = F::from_u32(record.from_pc);
    }
}
