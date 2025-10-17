use std::borrow::{Borrow, BorrowMut};

use openvm_circuit::{
    arch::{
        get_record_from_slice, AdapterAirContext, AdapterTraceExecutor, AdapterTraceFiller,
        BasicAdapterInterface, ExecutionBridge, ExecutionState, MinimalInstruction, VmAdapterAir,
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
use openvm_circuit_primitives::{
    bitwise_op_lookup::{BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip},
    utils::not,
    AlignedBytesBorrow,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_IMM_AS, RV32_REGISTER_AS},
};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{AirBuilder, BaseAir},
    p3_field::{Field, FieldAlgebra, PrimeField32},
};

use super::{
    tracing_read, tracing_read_imm, tracing_write, RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS,
};

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct Rv32BaseAluAdapterCols<T> {
    pub from_state: ExecutionState<T>,
    pub rd_ptr: T,
    pub rs1_ptr: T,
    /// Pointer if rs2 was a read, immediate value otherwise
    pub rs2: T,
    /// 1 if rs2 was a read, 0 if an immediate
    pub rs2_as: T,
    pub reads_aux: [MemoryReadAuxCols<T>; 2],
    pub writes_aux: MemoryWriteAuxCols<T, RV32_REGISTER_NUM_LIMBS>,
}

/// Reads instructions of the form OP a, b, c, d, e where \[a:4\]_d = \[b:4\]_d op \[c:4\]_e.
/// Operand d can only be 1, and e can be either 1 (for register reads) or 0 (when c
/// is an immediate).
#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct Rv32BaseAluAdapterAir {
    pub(super) execution_bridge: ExecutionBridge,
    pub(super) memory_bridge: MemoryBridge,
    bitwise_lookup_bus: BitwiseOperationLookupBus,
}

impl<F: Field> BaseAir<F> for Rv32BaseAluAdapterAir {
    fn width(&self) -> usize {
        Rv32BaseAluAdapterCols::<F>::width()
    }
}

impl<AB: InteractionBuilder> VmAdapterAir<AB> for Rv32BaseAluAdapterAir {
    type Interface = BasicAdapterInterface<
        AB::Expr,
        MinimalInstruction<AB::Expr>,
        2,
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
        let local: &Rv32BaseAluAdapterCols<_> = local.borrow();
        let timestamp = local.from_state.timestamp;
        let mut timestamp_delta: usize = 0;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            timestamp + AB::F::from_canonical_usize(timestamp_delta - 1)
        };

        // If rs2 is an immediate value, constrain that:
        // 1. It's a 16-bit two's complement integer (stored in rs2_limbs[0] and rs2_limbs[1])
        // 2. It's properly sign-extended to 32-bits (the upper limbs must match the sign bit)
        let rs2_limbs = ctx.reads[1].clone();
        let rs2_sign = rs2_limbs[2].clone();
        let rs2_imm = rs2_limbs[0].clone()
            + rs2_limbs[1].clone() * AB::Expr::from_canonical_usize(1 << RV32_CELL_BITS)
            + rs2_sign.clone() * AB::Expr::from_canonical_usize(1 << (2 * RV32_CELL_BITS));
        builder.assert_bool(local.rs2_as);
        let mut rs2_imm_when = builder.when(not(local.rs2_as));
        rs2_imm_when.assert_eq(local.rs2, rs2_imm);
        rs2_imm_when.assert_eq(rs2_sign.clone(), rs2_limbs[3].clone());
        rs2_imm_when.assert_zero(
            rs2_sign.clone()
                * (AB::Expr::from_canonical_usize((1 << RV32_CELL_BITS) - 1) - rs2_sign),
        );
        self.bitwise_lookup_bus
            .send_range(rs2_limbs[0].clone(), rs2_limbs[1].clone())
            .eval(builder, ctx.instruction.is_valid.clone() - local.rs2_as);

        self.memory_bridge
            .read(
                MemoryAddress::new(AB::F::from_canonical_u32(RV32_REGISTER_AS), local.rs1_ptr),
                ctx.reads[0].clone(),
                timestamp_pp(),
                &local.reads_aux[0],
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        // This constraint ensures that the following memory read only occurs when `is_valid == 1`.
        builder
            .when(local.rs2_as)
            .assert_one(ctx.instruction.is_valid.clone());
        self.memory_bridge
            .read(
                MemoryAddress::new(local.rs2_as, local.rs2),
                ctx.reads[1].clone(),
                timestamp_pp(),
                &local.reads_aux[1],
            )
            .eval(builder, local.rs2_as);

        self.memory_bridge
            .write(
                MemoryAddress::new(AB::F::from_canonical_u32(RV32_REGISTER_AS), local.rd_ptr),
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
                    AB::Expr::from_canonical_u32(RV32_REGISTER_AS),
                    local.rs2_as.into(),
                ],
                local.from_state,
                AB::F::from_canonical_usize(timestamp_delta),
                (DEFAULT_PC_STEP, ctx.to_pc),
            )
            .eval(builder, ctx.instruction.is_valid);
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        let cols: &Rv32BaseAluAdapterCols<_> = local.borrow();
        cols.from_state.pc
    }
}

#[derive(Clone, derive_new::new)]
pub struct Rv32BaseAluAdapterExecutor<const LIMB_BITS: usize>;

#[derive(derive_new::new)]
pub struct Rv32BaseAluAdapterFiller<const LIMB_BITS: usize> {
    bitwise_lookup_chip: SharedBitwiseOperationLookupChip<LIMB_BITS>,
}

// Intermediate type that should not be copied or cloned and should be directly written to
#[repr(C)]
#[derive(AlignedBytesBorrow, Debug)]
pub struct Rv32BaseAluAdapterRecord {
    pub from_pc: u32,
    pub from_timestamp: u32,

    pub rd_ptr: u32,
    pub rs1_ptr: u32,
    /// Pointer if rs2 was a read, immediate value otherwise
    pub rs2: u32,
    /// 1 if rs2 was a read, 0 if an immediate
    pub rs2_as: u8,

    pub reads_aux: [MemoryReadAuxRecord; 2],
    pub writes_aux: MemoryWriteBytesAuxRecord<RV32_REGISTER_NUM_LIMBS>,
}

impl<F: PrimeField32, const LIMB_BITS: usize> AdapterTraceExecutor<F>
    for Rv32BaseAluAdapterExecutor<LIMB_BITS>
{
    const WIDTH: usize = size_of::<Rv32BaseAluAdapterCols<u8>>();
    type ReadData = [[u8; RV32_REGISTER_NUM_LIMBS]; 2];
    type WriteData = [[u8; RV32_REGISTER_NUM_LIMBS]; 1];
    type RecordMut<'a> = &'a mut Rv32BaseAluAdapterRecord;

    #[inline(always)]
    fn start(pc: u32, memory: &TracingMemory, record: &mut &mut Rv32BaseAluAdapterRecord) {
        record.from_pc = pc;
        record.from_timestamp = memory.timestamp;
    }

    // @dev cannot get rid of double &mut due to trait
    #[inline(always)]
    fn read(
        &self,
        memory: &mut TracingMemory,
        instruction: &Instruction<F>,
        record: &mut &mut Rv32BaseAluAdapterRecord,
    ) -> Self::ReadData {
        let &Instruction { b, c, d, e, .. } = instruction;

        debug_assert_eq!(d.as_canonical_u32(), RV32_REGISTER_AS);
        debug_assert!(
            e.as_canonical_u32() == RV32_REGISTER_AS || e.as_canonical_u32() == RV32_IMM_AS
        );

        record.rs1_ptr = b.as_canonical_u32();
        let rs1 = tracing_read(
            memory,
            RV32_REGISTER_AS,
            record.rs1_ptr,
            &mut record.reads_aux[0].prev_timestamp,
        );

        let rs2 = if e.as_canonical_u32() == RV32_REGISTER_AS {
            record.rs2_as = RV32_REGISTER_AS as u8;
            record.rs2 = c.as_canonical_u32();

            tracing_read(
                memory,
                RV32_REGISTER_AS,
                record.rs2,
                &mut record.reads_aux[1].prev_timestamp,
            )
        } else {
            record.rs2_as = RV32_IMM_AS as u8;

            tracing_read_imm(memory, c.as_canonical_u32(), &mut record.rs2)
        };

        [rs1, rs2]
    }

    #[inline(always)]
    fn write(
        &self,
        memory: &mut TracingMemory,
        instruction: &Instruction<F>,
        data: Self::WriteData,
        record: &mut &mut Rv32BaseAluAdapterRecord,
    ) {
        let &Instruction { a, d, .. } = instruction;

        debug_assert_eq!(d.as_canonical_u32(), RV32_REGISTER_AS);

        record.rd_ptr = a.as_canonical_u32();
        tracing_write(
            memory,
            RV32_REGISTER_AS,
            record.rd_ptr,
            data[0],
            &mut record.writes_aux.prev_timestamp,
            &mut record.writes_aux.prev_data,
        );
    }
}

impl<F: PrimeField32, const LIMB_BITS: usize> AdapterTraceFiller<F>
    for Rv32BaseAluAdapterFiller<LIMB_BITS>
{
    const WIDTH: usize = size_of::<Rv32BaseAluAdapterCols<u8>>();

    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, mut adapter_row: &mut [F]) {
        // SAFETY: the following is highly unsafe. We are going to cast `adapter_row` to a record
        // buffer, and then do an _overlapping_ write to the `adapter_row` as a row of field
        // elements. This requires:
        // - Cols struct should be repr(C) and we write in reverse order (to ensure non-overlapping)
        // - Do not overwrite any reference in `record` before it has already been used or moved
        // - alignment of `F` must be >= alignment of Record (AlignedBytesBorrow will panic
        //   otherwise)
        // - adapter_row contains a valid Rv32BaseAluAdapterRecord representation
        // - get_record_from_slice correctly interprets the bytes as Rv32BaseAluAdapterRecord
        let record: &Rv32BaseAluAdapterRecord =
            unsafe { get_record_from_slice(&mut adapter_row, ()) };
        let adapter_row: &mut Rv32BaseAluAdapterCols<F> = adapter_row.borrow_mut();

        // We must assign in reverse
        const TIMESTAMP_DELTA: u32 = 2;
        let mut timestamp = record.from_timestamp + TIMESTAMP_DELTA;

        adapter_row
            .writes_aux
            .set_prev_data(record.writes_aux.prev_data.map(F::from_canonical_u8));
        mem_helper.fill(
            record.writes_aux.prev_timestamp,
            timestamp,
            adapter_row.writes_aux.as_mut(),
        );
        timestamp -= 1;

        if record.rs2_as != 0 {
            mem_helper.fill(
                record.reads_aux[1].prev_timestamp,
                timestamp,
                adapter_row.reads_aux[1].as_mut(),
            );
        } else {
            mem_helper.fill_zero(adapter_row.reads_aux[1].as_mut());
            let rs2_imm = record.rs2;
            let mask = (1 << RV32_CELL_BITS) - 1;
            self.bitwise_lookup_chip
                .request_range(rs2_imm & mask, (rs2_imm >> 8) & mask);
        }
        timestamp -= 1;

        mem_helper.fill(
            record.reads_aux[0].prev_timestamp,
            timestamp,
            adapter_row.reads_aux[0].as_mut(),
        );

        adapter_row.rs2_as = F::from_canonical_u8(record.rs2_as);
        adapter_row.rs2 = F::from_canonical_u32(record.rs2);
        adapter_row.rs1_ptr = F::from_canonical_u32(record.rs1_ptr);
        adapter_row.rd_ptr = F::from_canonical_u32(record.rd_ptr);
        adapter_row.from_state.timestamp = F::from_canonical_u32(timestamp);
        adapter_row.from_state.pc = F::from_canonical_u32(record.from_pc);
    }
}
