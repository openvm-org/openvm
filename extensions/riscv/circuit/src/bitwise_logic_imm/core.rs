use std::{
    array,
    borrow::{Borrow, BorrowMut},
    iter::zip,
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
use openvm_instructions::{instruction::Instruction, program::DEFAULT_PC_STEP, LocalOpcode};
use openvm_riscv_transpiler::{BaseAluOpcode, BitwiseImmOpcode};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, PrimeCharacteristicRing, PrimeField32},
    BaseAirWithPublicValues,
};

use crate::{adapters::imm_to_rv64_bytes, bitwise_logic::run_bitwise_logic};

/// Immediate-only fork of `BitwiseLogicCoreCols`: the eight `c` byte limbs are replaced by the
/// two low immediate bytes plus a sign bit; the sign-extension bytes are expressions.
#[repr(C)]
#[derive(AlignedBorrow, StructReflection, Debug)]
pub struct BitwiseLogicImmCoreCols<T, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub a: [T; NUM_LIMBS],
    pub b: [T; NUM_LIMBS],
    /// The two low bytes of the 24-bit immediate operand.
    pub c_low: [T; 2],
    /// Sign bit of the immediate: byte 2 of the operand is `0xFF * imm_sign`.
    pub imm_sign: T,

    pub opcode_xor_flag: T,
    pub opcode_or_flag: T,
    pub opcode_and_flag: T,
}

#[derive(Copy, Clone, Debug, derive_new::new, ColumnsAir)]
#[columns_via(BitwiseLogicImmCoreCols<u8, NUM_LIMBS, LIMB_BITS>)]
pub struct BitwiseLogicImmCoreAir<const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub bus: BitwiseOperationLookupBus,
    pub offset: usize,
}

impl<F: Field, const NUM_LIMBS: usize, const LIMB_BITS: usize> BaseAir<F>
    for BitwiseLogicImmCoreAir<NUM_LIMBS, LIMB_BITS>
{
    fn width(&self) -> usize {
        BitwiseLogicImmCoreCols::<F, NUM_LIMBS, LIMB_BITS>::width()
    }
}
impl<F: Field, const NUM_LIMBS: usize, const LIMB_BITS: usize> BaseAirWithPublicValues<F>
    for BitwiseLogicImmCoreAir<NUM_LIMBS, LIMB_BITS>
{
}

impl<AB, I, const NUM_LIMBS: usize, const LIMB_BITS: usize> VmCoreAir<AB, I>
    for BitwiseLogicImmCoreAir<NUM_LIMBS, LIMB_BITS>
where
    AB: InteractionBuilder,
    I: VmAdapterInterface<AB::Expr>,
    I::Reads: From<[[AB::Expr; NUM_LIMBS]; 1]>,
    I::Writes: From<[[AB::Expr; NUM_LIMBS]; 1]>,
    I::ProcessedInstruction: From<ImmInstruction<AB::Expr>>,
{
    fn eval(
        &self,
        builder: &mut AB,
        local_core: &[AB::Var],
        _from_pc: AB::Var,
    ) -> AdapterAirContext<AB::Expr, I> {
        let cols: &BitwiseLogicImmCoreCols<_, NUM_LIMBS, LIMB_BITS> = local_core.borrow();
        let flags = [
            cols.opcode_xor_flag,
            cols.opcode_or_flag,
            cols.opcode_and_flag,
        ];

        let is_valid = flags.iter().fold(AB::Expr::ZERO, |acc, &flag| {
            builder.assert_bool(flag);
            acc + flag.into()
        });
        builder.assert_bool(is_valid.clone());
        builder.assert_bool(cols.imm_sign);

        // Range check the two low immediate bytes; together with the boolean sign this makes
        // the decomposition of the 24-bit operand unique.
        self.bus
            .send_range(cols.c_low[0], cols.c_low[1])
            .eval(builder, is_valid.clone());

        // Sign-extended byte limbs of the immediate, as expressions.
        let sign_byte = cols.imm_sign * AB::Expr::from_u32((1 << LIMB_BITS) - 1);
        let c: [AB::Expr; NUM_LIMBS] = array::from_fn(|i| match i {
            0 => cols.c_low[0].into(),
            1 => cols.c_low[1].into(),
            _ => sign_byte.clone(),
        });

        let a = &cols.a;
        let b = &cols.b;

        for i in 0..NUM_LIMBS {
            let x_xor_y = cols.opcode_xor_flag * a[i]
                + cols.opcode_or_flag * ((AB::Expr::from_u32(2) * a[i]) - b[i] - c[i].clone())
                + cols.opcode_and_flag * (b[i] + c[i].clone() - (AB::Expr::from_u32(2) * a[i]));
            self.bus
                .send_xor(b[i], c[i].clone(), x_xor_y)
                .eval(builder, is_valid.clone());
        }

        let expected_opcode = VmCoreAir::<AB, I>::expr_to_global_expr(
            self,
            cols.opcode_xor_flag * AB::Expr::from_u8(BitwiseImmOpcode::XORI as u8)
                + cols.opcode_or_flag * AB::Expr::from_u8(BitwiseImmOpcode::ORI as u8)
                + cols.opcode_and_flag * AB::Expr::from_u8(BitwiseImmOpcode::ANDI as u8),
        );

        // 24-bit encoding matching i12_to_u24 in the transpiler (byte 2 is the sign byte).
        let imm = cols.c_low[0]
            + cols.c_low[1] * AB::Expr::from_u32(1 << LIMB_BITS)
            + sign_byte * AB::Expr::from_u32(1 << (2 * LIMB_BITS));

        AdapterAirContext {
            to_pc: None,
            reads: [cols.b.map(Into::into)].into(),
            writes: [cols.a.map(Into::into)].into(),
            instruction: ImmInstruction {
                is_valid,
                opcode: expected_opcode,
                immediate: imm,
            }
            .into(),
        }
    }

    fn start_offset(&self) -> usize {
        self.offset
    }
}

#[repr(C, align(4))]
#[derive(AlignedBytesBorrow, Debug)]
pub struct BitwiseLogicImmCoreRecord<const NUM_LIMBS: usize> {
    pub b: [u8; NUM_LIMBS],
    pub c_low: [u8; 2],
    pub imm_sign: u8,
    pub local_opcode: u8,
}

#[derive(Clone, Copy, derive_new::new)]
pub struct BitwiseLogicImmExecutor<A, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    adapter: A,
    pub offset: usize,
}

#[derive(derive_new::new)]
pub struct BitwiseLogicImmFiller<A, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    adapter: A,
    pub bitwise_lookup_chip: SharedBitwiseOperationLookupChip<LIMB_BITS>,
    pub offset: usize,
}

impl<F, A, RA, const NUM_LIMBS: usize, const LIMB_BITS: usize> PreflightExecutor<F, RA>
    for BitwiseLogicImmExecutor<A, NUM_LIMBS, LIMB_BITS>
where
    F: PrimeField32,
    A: 'static
        + AdapterTraceExecutor<
            F,
            ReadData: Into<[[u8; NUM_LIMBS]; 1]>,
            WriteData: From<[[u8; NUM_LIMBS]; 1]>,
        >,
    for<'buf> RA: RecordArena<
        'buf,
        EmptyAdapterCoreLayout<F, A>,
        (
            A::RecordMut<'buf>,
            &'buf mut BitwiseLogicImmCoreRecord<NUM_LIMBS>,
        ),
    >,
{
    fn get_opcode_name(&self, opcode: usize) -> String {
        format!("{:?}", BitwiseImmOpcode::from_usize(opcode - self.offset))
    }

    fn execute(
        &self,
        state: VmStateMut<F, TracingMemory, RA>,
        instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError> {
        let Instruction { opcode, c, .. } = instruction;

        let local_opcode = BitwiseImmOpcode::from_usize(opcode.local_opcode_idx(self.offset));

        let (mut adapter_record, core_record) = state.ctx.alloc(EmptyAdapterCoreLayout::new());
        A::start(*state.pc, state.memory, &mut adapter_record);

        [core_record.b] = self
            .adapter
            .read(state.memory, instruction, &mut adapter_record)
            .into();

        let c_bytes_full = imm_to_rv64_bytes(c.as_canonical_u32());
        let mut c_bytes = [0u8; NUM_LIMBS];
        c_bytes.copy_from_slice(&c_bytes_full[..NUM_LIMBS]);
        core_record.c_low = [c_bytes[0], c_bytes[1]];
        core_record.imm_sign = (c_bytes[2] != 0) as u8;
        core_record.local_opcode = local_opcode as u8;

        // BitwiseImmOpcode mirrors the XOR/OR/AND slots of BaseAluOpcode.
        let reg_opcode = BaseAluOpcode::from_usize(local_opcode as usize);
        let rd = run_bitwise_logic::<NUM_LIMBS, LIMB_BITS>(reg_opcode, &core_record.b, &c_bytes);

        self.adapter
            .write(state.memory, instruction, [rd].into(), &mut adapter_record);

        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);

        Ok(())
    }
}

impl<F, A, const NUM_LIMBS: usize, const LIMB_BITS: usize> TraceFiller<F>
    for BitwiseLogicImmFiller<A, NUM_LIMBS, LIMB_BITS>
where
    F: PrimeField32,
    A: 'static + AdapterTraceFiller<F>,
{
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        let (adapter_row, mut core_row) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };
        self.adapter.fill_trace_row(mem_helper, adapter_row);
        let record: &BitwiseLogicImmCoreRecord<NUM_LIMBS> =
            unsafe { get_record_from_slice(&mut core_row, ()) };
        let core_row: &mut BitwiseLogicImmCoreCols<F, NUM_LIMBS, LIMB_BITS> = core_row.borrow_mut();

        let b = record.b;
        let c_low = record.c_low;
        let imm_sign = record.imm_sign;
        let local_opcode = record.local_opcode;

        let sign_byte = imm_sign * (((1u32 << LIMB_BITS) - 1) as u8);
        let c: [u8; NUM_LIMBS] = array::from_fn(|i| match i {
            0 => c_low[0],
            1 => c_low[1],
            _ => sign_byte,
        });

        let reg_opcode = BaseAluOpcode::from_usize(local_opcode as usize);
        let a = run_bitwise_logic::<NUM_LIMBS, LIMB_BITS>(reg_opcode, &b, &c);

        self.bitwise_lookup_chip
            .request_range(c_low[0] as u32, c_low[1] as u32);
        for (b_val, c_val) in zip(b, c) {
            self.bitwise_lookup_chip
                .request_xor(b_val as u32, c_val as u32);
        }

        core_row.opcode_and_flag = F::from_bool(local_opcode == BitwiseImmOpcode::ANDI as u8);
        core_row.opcode_or_flag = F::from_bool(local_opcode == BitwiseImmOpcode::ORI as u8);
        core_row.opcode_xor_flag = F::from_bool(local_opcode == BitwiseImmOpcode::XORI as u8);
        core_row.imm_sign = F::from_u8(imm_sign);
        core_row.c_low = c_low.map(F::from_u8);
        core_row.b = b.map(F::from_u8);
        core_row.a = a.map(F::from_u8);
    }
}
