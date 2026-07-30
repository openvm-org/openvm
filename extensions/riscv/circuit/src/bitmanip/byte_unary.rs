use std::{
    array,
    borrow::{Borrow, BorrowMut},
};

use openvm_circuit::{
    arch::*,
    system::memory::{online::TracingMemory, MemoryAuxColsFactory},
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::{BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip},
    AlignedBytesBorrow, ColumnsAir, StructReflection, StructReflectionHelper,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{
    instruction::Instruction, program::DEFAULT_PC_STEP, riscv::RV64_REGISTER_NUM_LIMBS,
};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, PrimeCharacteristicRing, PrimeField32},
    BaseAirWithPublicValues,
};

use super::core::{
    is_byte_unary_opcode, run_bitmanip_imm, BITMANIP_OFFSET, ORC_B, REV8, SEXT_B, SEXT_H, ZEXT_H,
};
use crate::adapters::RV64_BYTE_BITS;

/// Unary byte ops (SEXT.B / SEXT.H / ZEXT.H / ORC.B / REV8) over byte limbs.
///
/// The written value is an expression over the operand bytes and a handful of
/// witnesses, so there are no output columns:
///   REV8:   out[i] = b[7 - i]                       (pure wiring)
///   ORC.B:  out[i] = 255 * nz[i], with nz[i] = (b[i] != 0) proven via the
///           standard inverse trick (b * inv = nz, b * (1 - nz) = 0)
///   SEXT.B: b[0] = lo + 128 * t, out = [b[0], 255*t, ...]
///   SEXT.H: b[1] = lo + 128 * t, out = [b[0], b[1], 255*t, ...]
///   ZEXT.H: out = [b[0], b[1], 0, ...]
/// The byte-limb adapter's u16 packing does not bound individual bytes, so the
/// `b` bytes are range checked pairwise through the bitwise lookup; `lo < 128`
/// is checked as `2 * lo < 256` on the same lookup.
#[repr(C)]
#[derive(AlignedBorrow, StructReflection, Debug)]
pub struct BitManipByteUnaryCoreCols<T> {
    pub b: [T; RV64_REGISTER_NUM_LIMBS],
    /// ORC.B only: nz[i] = 1 iff b[i] != 0.
    pub nz: [T; RV64_REGISTER_NUM_LIMBS],
    /// ORC.B only: inverse of b[i] when b[i] != 0, else 0.
    pub inv: [T; RV64_REGISTER_NUM_LIMBS],
    /// SEXT.B/SEXT.H only: sign bit and low 7 bits of the extended byte.
    pub sext_bit: T,
    pub sext_low: T,
    pub opcode_sext_b_flag: T,
    pub opcode_sext_h_flag: T,
    pub opcode_zext_h_flag: T,
    pub opcode_orc_b_flag: T,
    pub opcode_rev8_flag: T,
}

#[derive(Copy, Clone, Debug, derive_new::new, ColumnsAir)]
#[columns_via(BitManipByteUnaryCoreCols<u8>)]
pub struct BitManipByteUnaryCoreAir {
    pub bus: BitwiseOperationLookupBus,
}

impl<F: Field> BaseAir<F> for BitManipByteUnaryCoreAir {
    fn width(&self) -> usize {
        BitManipByteUnaryCoreCols::<F>::width()
    }
}
impl<F: Field> BaseAirWithPublicValues<F> for BitManipByteUnaryCoreAir {}

impl<AB, I> VmCoreAir<AB, I> for BitManipByteUnaryCoreAir
where
    AB: InteractionBuilder,
    I: VmAdapterInterface<AB::Expr>,
    I::Reads: From<[[AB::Expr; RV64_REGISTER_NUM_LIMBS]; 1]>,
    I::Writes: From<[[AB::Expr; RV64_REGISTER_NUM_LIMBS]; 1]>,
    I::ProcessedInstruction: From<ImmInstruction<AB::Expr>>,
{
    fn eval(
        &self,
        builder: &mut AB,
        local_core: &[AB::Var],
        _from_pc: AB::Var,
    ) -> AdapterAirContext<AB::Expr, I> {
        let cols: &BitManipByteUnaryCoreCols<_> = local_core.borrow();
        let flags = [
            (cols.opcode_sext_b_flag, SEXT_B),
            (cols.opcode_sext_h_flag, SEXT_H),
            (cols.opcode_zext_h_flag, ZEXT_H),
            (cols.opcode_orc_b_flag, ORC_B),
            (cols.opcode_rev8_flag, REV8),
        ];

        let is_valid = flags.iter().fold(AB::Expr::ZERO, |acc, &(flag, _)| {
            builder.assert_bool(flag);
            acc + flag.into()
        });
        builder.assert_bool(is_valid.clone());

        let max_byte = AB::Expr::from_u8(u8::MAX);

        // The adapter packs byte pairs into u16 memory cells, which bounds the
        // pairs but not the individual bytes; range check them here.
        for i in 0..RV64_REGISTER_NUM_LIMBS / 2 {
            self.bus
                .send_range(cols.b[2 * i], cols.b[2 * i + 1])
                .eval(builder, is_valid.clone());
        }

        // ORC.B witnesses: nz[i] boolean, and (gated on the flag) nz[i] = 1
        // exactly when b[i] != 0.
        for i in 0..RV64_REGISTER_NUM_LIMBS {
            builder.assert_bool(cols.nz[i]);
            builder.assert_zero(cols.opcode_orc_b_flag * (cols.b[i] * cols.inv[i] - cols.nz[i]));
            builder.assert_zero(cols.opcode_orc_b_flag * cols.b[i] * (AB::Expr::ONE - cols.nz[i]));
        }

        // SEXT.B/SEXT.H witness: the extended byte decomposes as
        // lo + 128 * t with t boolean and lo < 128 (checked as 2 * lo < 256).
        builder.assert_bool(cols.sext_bit);
        let sext_flags = cols.opcode_sext_b_flag + cols.opcode_sext_h_flag;
        let selected_byte =
            cols.opcode_sext_b_flag * cols.b[0] + cols.opcode_sext_h_flag * cols.b[1];
        builder.assert_eq(
            selected_byte,
            sext_flags.clone() * (cols.sext_low + AB::Expr::from_u8(128) * cols.sext_bit),
        );
        self.bus
            .send_range(cols.sext_low + cols.sext_low, AB::Expr::ZERO)
            .eval(builder, sext_flags.clone());

        let sign_fill = max_byte.clone() * cols.sext_bit;
        let writes: [AB::Expr; RV64_REGISTER_NUM_LIMBS] = array::from_fn(|i| {
            let mut out = cols.opcode_orc_b_flag * (max_byte.clone() * cols.nz[i])
                + cols.opcode_rev8_flag * cols.b[RV64_REGISTER_NUM_LIMBS - 1 - i];
            out += match i {
                0 => (sext_flags.clone() + cols.opcode_zext_h_flag) * cols.b[0],
                1 => {
                    (cols.opcode_sext_h_flag + cols.opcode_zext_h_flag) * cols.b[1]
                        + cols.opcode_sext_b_flag * sign_fill.clone()
                }
                _ => sext_flags.clone() * sign_fill.clone(),
            };
            out
        });

        let expected_local_opcode = flags
            .iter()
            .fold(AB::Expr::ZERO, |acc, &(flag, local_opcode)| {
                acc + flag * AB::Expr::from_usize(local_opcode)
            });
        let expected_opcode = expected_local_opcode + AB::Expr::from_usize(BITMANIP_OFFSET);

        AdapterAirContext {
            to_pc: None,
            reads: [cols.b.map(Into::into)].into(),
            writes: [writes].into(),
            instruction: ImmInstruction {
                is_valid,
                opcode: expected_opcode,
                immediate: AB::Expr::ZERO,
            }
            .into(),
        }
    }

    fn start_offset(&self) -> usize {
        BITMANIP_OFFSET
    }
}

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug)]
pub struct BitManipByteUnaryCoreRecord {
    pub b: [u8; RV64_REGISTER_NUM_LIMBS],
    pub local_opcode: u8,
}

#[derive(Clone, Copy, derive_new::new)]
pub struct BitManipByteUnaryExecutor<A> {
    adapter: A,
}

#[derive(Clone, derive_new::new)]
pub struct BitManipByteUnaryFiller<A> {
    adapter: A,
    pub bitwise_lookup_chip: SharedBitwiseOperationLookupChip<{ RV64_BYTE_BITS }>,
}

impl<F, A, RA> PreflightExecutor<F, RA> for BitManipByteUnaryExecutor<A>
where
    F: PrimeField32,
    A: 'static
        + AdapterTraceExecutor<
            F,
            ReadData: Into<[[u8; RV64_REGISTER_NUM_LIMBS]; 1]>,
            WriteData: From<[[u8; RV64_REGISTER_NUM_LIMBS]; 1]>,
        >,
    for<'buf> RA: RecordArena<
        'buf,
        EmptyAdapterCoreLayout<F, A>,
        (A::RecordMut<'buf>, &'buf mut BitManipByteUnaryCoreRecord),
    >,
{
    fn get_opcode_name(&self, opcode: usize) -> String {
        format!("Rv64BByteUnary({})", opcode - BITMANIP_OFFSET)
    }

    fn execute(
        &self,
        state: VmStateMut<TracingMemory, RA>,
        instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError> {
        let local_opcode = instruction.opcode.local_opcode_idx(BITMANIP_OFFSET);
        debug_assert!(is_byte_unary_opcode(local_opcode));

        let (mut adapter_record, core_record) = state.ctx.alloc(EmptyAdapterCoreLayout::new());
        A::start(*state.pc, state.memory, &mut adapter_record);
        [core_record.b] = self
            .adapter
            .read(state.memory, instruction, &mut adapter_record)
            .into();
        core_record.local_opcode = local_opcode as u8;

        let output = run_bitmanip_imm(local_opcode, u64::from_le_bytes(core_record.b), 0);
        self.adapter.write(
            state.memory,
            instruction,
            [output.to_le_bytes()].into(),
            &mut adapter_record,
        );
        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);
        Ok(())
    }
}

impl<F, A> TraceFiller<F> for BitManipByteUnaryFiller<A>
where
    F: PrimeField32,
    A: 'static + AdapterTraceFiller<F>,
{
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        let (adapter_row, mut core_row) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };
        self.adapter.fill_trace_row(mem_helper, adapter_row);
        let record: &BitManipByteUnaryCoreRecord =
            unsafe { get_record_from_slice(&mut core_row, ()) };
        let b = record.b;
        let local_opcode = record.local_opcode as usize;

        for i in 0..RV64_REGISTER_NUM_LIMBS / 2 {
            self.bitwise_lookup_chip
                .request_range(b[2 * i] as u32, b[2 * i + 1] as u32);
        }

        let core_row: &mut BitManipByteUnaryCoreCols<F> = core_row.borrow_mut();
        let is_orc_b = local_opcode == ORC_B;
        let (sext_bit, sext_low) = match local_opcode {
            SEXT_B => (b[0] >> 7, b[0] & 0x7f),
            SEXT_H => (b[1] >> 7, b[1] & 0x7f),
            _ => (0, 0),
        };
        if matches!(local_opcode, SEXT_B | SEXT_H) {
            self.bitwise_lookup_chip
                .request_range(2 * sext_low as u32, 0);
        }

        core_row.opcode_rev8_flag = F::from_bool(local_opcode == REV8);
        core_row.opcode_orc_b_flag = F::from_bool(is_orc_b);
        core_row.opcode_zext_h_flag = F::from_bool(local_opcode == ZEXT_H);
        core_row.opcode_sext_h_flag = F::from_bool(local_opcode == SEXT_H);
        core_row.opcode_sext_b_flag = F::from_bool(local_opcode == SEXT_B);
        core_row.sext_low = F::from_u8(sext_low);
        core_row.sext_bit = F::from_u8(sext_bit);
        for i in (0..RV64_REGISTER_NUM_LIMBS).rev() {
            let (nz, inv) = if is_orc_b && b[i] != 0 {
                (F::ONE, F::from_u8(b[i]).inverse())
            } else {
                (F::ZERO, F::ZERO)
            };
            core_row.inv[i] = inv;
            core_row.nz[i] = nz;
        }
        core_row.b = b.map(F::from_u8);
    }
}
