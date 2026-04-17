use std::{
    array,
    borrow::{Borrow, BorrowMut},
};

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
    riscv::{
        RV64_CELL_BITS, RV64_IMM_AS, RV64_REGISTER_AS, RV64_REGISTER_NUM_LIMBS, RV64_WORD_NUM_LIMBS,
    },
};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{AirBuilder, BaseAir},
    p3_field::{Field, FieldAlgebra, PrimeField32},
};

use super::{tracing_read, tracing_read_imm, tracing_write};

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct Rv64BaseAluWAdapterCols<T> {
    pub from_state: ExecutionState<T>,
    pub rd_ptr: T,
    pub rs1_ptr: T,
    /// Upper 4 bytes of rs1 register read (kept in adapter to satisfy full-width memory read).
    pub rs1_high: [T; RV64_REGISTER_NUM_LIMBS - RV64_WORD_NUM_LIMBS],
    /// Pointer if rs2 was a read, immediate value otherwise
    pub rs2: T,
    /// 1 if rs2 was a read, 0 if an immediate
    pub rs2_as: T,
    /// Upper 4 bytes of rs2 register read (unused when rs2 is immediate).
    pub rs2_high: [T; RV64_REGISTER_NUM_LIMBS - RV64_WORD_NUM_LIMBS],
    /// Sign bit of the low-word core result used to build full-width sign-extended writes.
    pub result_sign: T,
    pub reads_aux: [MemoryReadAuxCols<T>; 2],
    pub writes_aux: MemoryWriteAuxCols<T, RV64_REGISTER_NUM_LIMBS>,
}

/// Same instruction format as `Rv64BaseAluAdapterAir`, but only exposes the low 32-bit limbs
/// (`RV64_WORD_NUM_LIMBS`) for reads and writes. Full-width RV64 writes are rebuilt in-adapter by
/// sign-extending the low-word result.
#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct Rv64BaseAluWAdapterAir {
    pub(super) execution_bridge: ExecutionBridge,
    pub(super) memory_bridge: MemoryBridge,
    bitwise_lookup_bus: BitwiseOperationLookupBus,
}

impl<F: Field> BaseAir<F> for Rv64BaseAluWAdapterAir {
    fn width(&self) -> usize {
        Rv64BaseAluWAdapterCols::<F>::width()
    }
}

impl<AB: InteractionBuilder> VmAdapterAir<AB> for Rv64BaseAluWAdapterAir {
    type Interface = BasicAdapterInterface<
        AB::Expr,
        MinimalInstruction<AB::Expr>,
        2,
        1,
        RV64_WORD_NUM_LIMBS,
        RV64_WORD_NUM_LIMBS,
    >;

    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        let local: &Rv64BaseAluWAdapterCols<_> = local.borrow();
        let timestamp = local.from_state.timestamp;
        let mut timestamp_delta: usize = 0;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            timestamp + AB::F::from_canonical_usize(timestamp_delta - 1)
        };

        // If rs2 is an immediate value, constrain that:
        // 1. rs2_limbs[0] and rs2_limbs[1] are valid bytes encoding the low 16 bits
        // 2. rs2_limbs[2] is the sign byte (0x00 or 0xFF)
        // 3. The 24-bit value limbs[0..3] reconstructs to local.rs2
        // 4. Limbs[3..4] are sign-extended (equal to the sign byte)
        let rs2_limbs = ctx.reads[1].clone();
        let rs2_sign = rs2_limbs[2].clone();
        let rs2_imm = rs2_limbs[0].clone()
            + rs2_limbs[1].clone() * AB::Expr::from_canonical_usize(1 << RV64_CELL_BITS)
            + rs2_sign.clone() * AB::Expr::from_canonical_usize(1 << (2 * RV64_CELL_BITS));
        builder.assert_bool(local.rs2_as);
        let mut rs2_imm_when = builder.when(not(local.rs2_as));
        rs2_imm_when.assert_eq(local.rs2, rs2_imm);
        for i in 3..RV64_WORD_NUM_LIMBS {
            rs2_imm_when.assert_eq(rs2_sign.clone(), rs2_limbs[i].clone());
        }
        rs2_imm_when.assert_zero(
            rs2_sign.clone()
                * (AB::Expr::from_canonical_usize((1 << RV64_CELL_BITS) - 1) - rs2_sign),
        );
        self.bitwise_lookup_bus
            .send_range(rs2_limbs[0].clone(), rs2_limbs[1].clone())
            .eval(builder, ctx.instruction.is_valid.clone() - local.rs2_as);

        let rs1_data: [AB::Expr; RV64_REGISTER_NUM_LIMBS] = array::from_fn(|i| {
            if i < RV64_WORD_NUM_LIMBS {
                ctx.reads[0][i].clone()
            } else {
                local.rs1_high[i - RV64_WORD_NUM_LIMBS].into()
            }
        });
        self.memory_bridge
            .read(
                MemoryAddress::new(AB::F::from_canonical_u32(RV64_REGISTER_AS), local.rs1_ptr),
                rs1_data,
                timestamp_pp(),
                &local.reads_aux[0],
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        // This constraint ensures that the following memory read only occurs when `is_valid == 1`.
        builder
            .when(local.rs2_as)
            .assert_one(ctx.instruction.is_valid.clone());
        let rs2_data: [AB::Expr; RV64_REGISTER_NUM_LIMBS] = array::from_fn(|i| {
            if i < RV64_WORD_NUM_LIMBS {
                ctx.reads[1][i].clone()
            } else {
                local.rs2_high[i - RV64_WORD_NUM_LIMBS].into()
            }
        });
        self.memory_bridge
            .read(
                MemoryAddress::new(local.rs2_as, local.rs2),
                rs2_data,
                timestamp_pp(),
                &local.reads_aux[1],
            )
            .eval(builder, local.rs2_as);

        // Sign-extend the 32-bit result to 64 bits: extract the sign bit of the
        // most-significant limb of the 32-bit word, then fill the upper limbs with
        // 0x00 (positive) or 0xFF (negative).
        builder.assert_bool(local.result_sign);
        let sign_mask = AB::Expr::from_canonical_u32(1 << (RV64_CELL_BITS - 1));
        let result_word_msl = ctx.writes[0][RV64_WORD_NUM_LIMBS - 1].clone();
        self.bitwise_lookup_bus
            .send_xor(
                result_word_msl.clone(),
                sign_mask.clone(),
                result_word_msl + sign_mask.clone()
                    - AB::Expr::from_canonical_u32(2) * local.result_sign * sign_mask,
            )
            .eval(builder, ctx.instruction.is_valid.clone());
        let sign_extend_limb =
            AB::Expr::from_canonical_u32((1 << RV64_CELL_BITS) - 1) * local.result_sign;
        let write_data: [AB::Expr; RV64_REGISTER_NUM_LIMBS] = array::from_fn(|i| {
            if i < RV64_WORD_NUM_LIMBS {
                ctx.writes[0][i].clone()
            } else {
                sign_extend_limb.clone()
            }
        });
        self.memory_bridge
            .write(
                MemoryAddress::new(AB::F::from_canonical_u32(RV64_REGISTER_AS), local.rd_ptr),
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
                    local.rs2.into(),
                    AB::Expr::from_canonical_u32(RV64_REGISTER_AS),
                    local.rs2_as.into(),
                ],
                local.from_state,
                AB::F::from_canonical_usize(timestamp_delta),
                (DEFAULT_PC_STEP, ctx.to_pc),
            )
            .eval(builder, ctx.instruction.is_valid);
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        let cols: &Rv64BaseAluWAdapterCols<_> = local.borrow();
        cols.from_state.pc
    }
}

#[derive(Clone, derive_new::new)]
pub struct Rv64BaseAluWAdapterExecutor;

#[derive(derive_new::new)]
pub struct Rv64BaseAluWAdapterFiller {
    bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV64_CELL_BITS>,
}

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug)]
pub struct Rv64BaseAluWAdapterRecord {
    pub from_pc: u32,
    pub from_timestamp: u32,

    pub rd_ptr: u32,
    pub rs1_ptr: u32,
    pub rs1_high: [u8; RV64_REGISTER_NUM_LIMBS - RV64_WORD_NUM_LIMBS],
    /// Pointer if rs2 was a read, immediate value otherwise
    pub rs2: u32,
    /// 1 if rs2 was a read, 0 if an immediate
    pub rs2_as: u8,
    pub rs2_high: [u8; RV64_REGISTER_NUM_LIMBS - RV64_WORD_NUM_LIMBS],
    pub result_sign: u8,
    pub result_word_msl: u8,

    pub reads_aux: [MemoryReadAuxRecord; 2],
    pub writes_aux: MemoryWriteBytesAuxRecord<RV64_REGISTER_NUM_LIMBS>,
}

impl<F: PrimeField32> AdapterTraceExecutor<F> for Rv64BaseAluWAdapterExecutor {
    const WIDTH: usize = size_of::<Rv64BaseAluWAdapterCols<u8>>();
    type ReadData = [[u8; RV64_WORD_NUM_LIMBS]; 2];
    type WriteData = [[u8; RV64_WORD_NUM_LIMBS]; 1];
    type RecordMut<'a> = &'a mut Rv64BaseAluWAdapterRecord;

    #[inline(always)]
    fn start(pc: u32, memory: &TracingMemory, record: &mut &mut Rv64BaseAluWAdapterRecord) {
        record.from_pc = pc;
        record.from_timestamp = memory.timestamp;
    }

    #[inline(always)]
    fn read(
        &self,
        memory: &mut TracingMemory,
        instruction: &Instruction<F>,
        record: &mut &mut Rv64BaseAluWAdapterRecord,
    ) -> Self::ReadData {
        let &Instruction { b, c, d, e, .. } = instruction;

        debug_assert_eq!(d.as_canonical_u32(), RV64_REGISTER_AS);
        debug_assert!(
            e.as_canonical_u32() == RV64_REGISTER_AS || e.as_canonical_u32() == RV64_IMM_AS
        );

        record.rs1_ptr = b.as_canonical_u32();
        let rs1_full: [u8; RV64_REGISTER_NUM_LIMBS] = tracing_read(
            memory,
            RV64_REGISTER_AS,
            record.rs1_ptr,
            &mut record.reads_aux[0].prev_timestamp,
        );
        record
            .rs1_high
            .copy_from_slice(&rs1_full[RV64_WORD_NUM_LIMBS..]);
        let rs1 = rs1_full[..RV64_WORD_NUM_LIMBS].try_into().unwrap();

        let rs2 = if e.as_canonical_u32() == RV64_REGISTER_AS {
            record.rs2_as = RV64_REGISTER_AS as u8;
            record.rs2 = c.as_canonical_u32();

            let rs2_full: [u8; RV64_REGISTER_NUM_LIMBS] = tracing_read(
                memory,
                RV64_REGISTER_AS,
                record.rs2,
                &mut record.reads_aux[1].prev_timestamp,
            );
            record
                .rs2_high
                .copy_from_slice(&rs2_full[RV64_WORD_NUM_LIMBS..]);
            rs2_full[..RV64_WORD_NUM_LIMBS].try_into().unwrap()
        } else {
            record.rs2_as = RV64_IMM_AS as u8;
            let imm_full = tracing_read_imm(memory, c.as_canonical_u32(), &mut record.rs2);
            record
                .rs2_high
                .copy_from_slice(&imm_full[RV64_WORD_NUM_LIMBS..]);
            array::from_fn(|i| imm_full[i])
        };

        [rs1, rs2]
    }

    #[inline(always)]
    fn write(
        &self,
        memory: &mut TracingMemory,
        instruction: &Instruction<F>,
        data: Self::WriteData,
        record: &mut &mut Rv64BaseAluWAdapterRecord,
    ) {
        let &Instruction { a, d, .. } = instruction;

        debug_assert_eq!(d.as_canonical_u32(), RV64_REGISTER_AS);

        record.rd_ptr = a.as_canonical_u32();
        let write_low = data[0];
        record.result_word_msl = write_low[RV64_WORD_NUM_LIMBS - 1];
        record.result_sign = record.result_word_msl >> (RV64_CELL_BITS as u8 - 1);
        let sign_extend_limb = ((1u16 << RV64_CELL_BITS) - 1) as u8 * record.result_sign;
        let mut write_data = [sign_extend_limb; RV64_REGISTER_NUM_LIMBS];
        write_data[..RV64_WORD_NUM_LIMBS].copy_from_slice(&write_low);
        tracing_write(
            memory,
            RV64_REGISTER_AS,
            record.rd_ptr,
            write_data,
            &mut record.writes_aux.prev_timestamp,
            &mut record.writes_aux.prev_data,
        );
    }
}

impl<F: PrimeField32> AdapterTraceFiller<F> for Rv64BaseAluWAdapterFiller {
    const WIDTH: usize = size_of::<Rv64BaseAluWAdapterCols<u8>>();

    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, mut adapter_row: &mut [F]) {
        // SAFETY: the following is highly unsafe. We are going to cast `adapter_row` to a record
        // buffer, and then do an _overlapping_ write to the `adapter_row` as a row of field
        // elements. This requires:
        // - Cols struct should be repr(C) and we write in reverse order (to ensure non-overlapping)
        // - Do not overwrite any reference in `record` before it has already been used or moved
        // - alignment of `F` must be >= alignment of Record (AlignedBytesBorrow will panic
        //   otherwise)
        // - adapter_row contains a valid Rv64BaseAluWAdapterRecord representation
        // - get_record_from_slice correctly interprets the bytes as Rv64BaseAluWAdapterRecord
        let record: &Rv64BaseAluWAdapterRecord =
            unsafe { get_record_from_slice(&mut adapter_row, ()) };
        let adapter_row: &mut Rv64BaseAluWAdapterCols<F> = adapter_row.borrow_mut();

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
            let mask = (1 << RV64_CELL_BITS) - 1;
            self.bitwise_lookup_chip
                .request_range(rs2_imm & mask, (rs2_imm >> RV64_CELL_BITS) & mask);
        }
        timestamp -= 1;

        mem_helper.fill(
            record.reads_aux[0].prev_timestamp,
            timestamp,
            adapter_row.reads_aux[0].as_mut(),
        );
        self.bitwise_lookup_chip
            .request_xor(record.result_word_msl as u32, 1u32 << (RV64_CELL_BITS - 1));

        adapter_row.result_sign = F::from_canonical_u8(record.result_sign);
        adapter_row.rs2_as = F::from_canonical_u8(record.rs2_as);
        adapter_row.rs2 = F::from_canonical_u32(record.rs2);
        adapter_row.rs2_high = record.rs2_high.map(F::from_canonical_u8);
        adapter_row.rs1_high = record.rs1_high.map(F::from_canonical_u8);
        adapter_row.rs1_ptr = F::from_canonical_u32(record.rs1_ptr);
        adapter_row.rd_ptr = F::from_canonical_u32(record.rd_ptr);
        adapter_row.from_state.timestamp = F::from_canonical_u32(timestamp);
        adapter_row.from_state.pc = F::from_canonical_u32(record.from_pc);
    }
}
