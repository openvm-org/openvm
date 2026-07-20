use std::{
    array,
    borrow::{Borrow, BorrowMut},
};

use openvm_circuit::{
    arch::*,
    system::memory::{online::TracingMemory, MemoryAuxColsFactory},
};
use openvm_circuit_primitives::{
    var_range::{SharedVariableRangeCheckerChip, VariableRangeCheckerBus},
    AlignedBytesBorrow, ColumnsAir, StructReflection, StructReflectionHelper,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{instruction::Instruction, program::DEFAULT_PC_STEP, LocalOpcode};
use openvm_riscv_transpiler::{BaseAluImmOpcode, BaseAluWImmOpcode};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{AirBuilder, BaseAir},
    p3_field::{Field, PrimeCharacteristicRing, PrimeField32},
    BaseAirWithPublicValues,
};

use crate::adapters::{is_canonical_i12, U16_BITS};

#[repr(C)]
#[derive(AlignedBorrow, StructReflection, Debug)]
pub struct AddICoreCols<T, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    /// Result limbs.
    pub rd: [T; NUM_LIMBS],
    /// Source operand limbs.
    pub rs1: [T; NUM_LIMBS],
    /// Low 11 bits (`imm[10:0]`) of the signed 12-bit immediate.
    pub imm_low11: T,
    /// Sign bit (`imm[11]`), used to sign-extend the immediate across all limbs.
    pub imm_sign: T,
    /// Whether this row contains an instruction.
    pub is_valid: T,
}

#[derive(Copy, Clone, Debug, derive_new::new, ColumnsAir)]
#[columns_via(AddICoreCols<u8, NUM_LIMBS, LIMB_BITS>)]
pub struct AddICoreAir<
    const NUM_LIMBS: usize,
    const LIMB_BITS: usize,
    const RANGE_CHECK_TOP_LIMB: bool,
> {
    pub range_bus: VariableRangeCheckerBus,
    pub offset: usize,
    pub local_opcode: usize,
}

impl<
        F: Field,
        const NUM_LIMBS: usize,
        const LIMB_BITS: usize,
        const RANGE_CHECK_TOP_LIMB: bool,
    > BaseAir<F> for AddICoreAir<NUM_LIMBS, LIMB_BITS, RANGE_CHECK_TOP_LIMB>
{
    fn width(&self) -> usize {
        AddICoreCols::<F, NUM_LIMBS, LIMB_BITS>::width()
    }
}
impl<
        F: Field,
        const NUM_LIMBS: usize,
        const LIMB_BITS: usize,
        const RANGE_CHECK_TOP_LIMB: bool,
    > BaseAirWithPublicValues<F> for AddICoreAir<NUM_LIMBS, LIMB_BITS, RANGE_CHECK_TOP_LIMB>
{
}

impl<AB, I, const NUM_LIMBS: usize, const LIMB_BITS: usize, const RANGE_CHECK_TOP_LIMB: bool>
    VmCoreAir<AB, I> for AddICoreAir<NUM_LIMBS, LIMB_BITS, RANGE_CHECK_TOP_LIMB>
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
        assert!(NUM_LIMBS > 0 && (12..=U16_BITS).contains(&LIMB_BITS));

        let cols: &AddICoreCols<_, NUM_LIMBS, LIMB_BITS> = local_core.borrow();

        builder.assert_bool(cols.is_valid);
        builder.assert_bool(cols.imm_sign);

        self.range_bus
            .range_check(cols.imm_low11, 11)
            .eval(builder, cols.is_valid.into());

        let limb_base = 1usize << LIMB_BITS;
        let limb_mask = limb_base - 1;
        let imm_sign: AB::Expr = cols.imm_sign.into();
        let sign_limb: AB::Expr = imm_sign.clone() * AB::Expr::from_usize(limb_mask);
        let imm0: AB::Expr =
            cols.imm_low11 + imm_sign.clone() * AB::Expr::from_usize(limb_base - (1 << 11));

        let carry_divide = AB::F::from_usize(limb_base).inverse();
        let mut carry: [AB::Expr; NUM_LIMBS] = array::from_fn(|_| AB::Expr::ZERO);

        carry[0] = AB::Expr::from(carry_divide) * (cols.rs1[0] + imm0 - cols.rd[0]);
        builder.when(cols.is_valid).assert_bool(carry[0].clone());

        for i in 1..NUM_LIMBS {
            carry[i] = AB::Expr::from(carry_divide)
                * (cols.rs1[i] + sign_limb.clone() - cols.rd[i] + carry[i - 1].clone());
            builder.when(cols.is_valid).assert_bool(carry[i].clone());
        }

        let range_limb_count = NUM_LIMBS - usize::from(!RANGE_CHECK_TOP_LIMB);
        for &rd_limb in &cols.rd[..range_limb_count] {
            self.range_bus
                .range_check(rd_limb, LIMB_BITS)
                .eval(builder, cols.is_valid.into());
        }

        // 24-bit encoding matching i12_to_u24 in the transpiler.
        let instr_c: AB::Expr = cols.imm_low11 + imm_sign * AB::Expr::from_u32(0xFFF800);

        let expected_opcode =
            VmCoreAir::<AB, I>::expr_to_global_expr(self, AB::Expr::from_usize(self.local_opcode));

        AdapterAirContext {
            to_pc: None,
            reads: [cols.rs1.map(Into::into)].into(),
            writes: [cols.rd.map(Into::into)].into(),
            instruction: ImmInstruction {
                is_valid: cols.is_valid.into(),
                opcode: expected_opcode,
                immediate: instr_c,
            }
            .into(),
        }
    }

    fn start_offset(&self) -> usize {
        self.offset
    }
}

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug)]
pub struct AddICoreRecord<const NUM_LIMBS: usize> {
    pub rs1: [u16; NUM_LIMBS],
    pub imm_low11: u16,
    pub imm_sign: u16,
}

#[derive(Clone, Copy, derive_new::new)]
pub struct AddIExecutor<A, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    adapter: A,
    pub offset: usize,
    pub local_opcode: usize,
}

#[derive(derive_new::new)]
pub struct AddIFiller<
    A,
    const NUM_LIMBS: usize,
    const LIMB_BITS: usize,
    const RANGE_CHECK_TOP_LIMB: bool,
> {
    adapter: A,
    pub range_checker_chip: SharedVariableRangeCheckerChip,
}

impl<F, A, RA, const NUM_LIMBS: usize, const LIMB_BITS: usize> PreflightExecutor<F, RA>
    for AddIExecutor<A, NUM_LIMBS, LIMB_BITS>
where
    F: PrimeField32,
    A: 'static
        + AdapterTraceExecutor<
            F,
            ReadData: Into<[[u16; NUM_LIMBS]; 1]>,
            WriteData: From<[[u16; NUM_LIMBS]; 1]>,
        >,
    for<'buf> RA: RecordArena<
        'buf,
        EmptyAdapterCoreLayout<F, A>,
        (A::RecordMut<'buf>, &'buf mut AddICoreRecord<NUM_LIMBS>),
    >,
{
    fn get_opcode_name(&self, opcode: usize) -> String {
        if NUM_LIMBS * LIMB_BITS == 32 {
            format!("{:?}", BaseAluWImmOpcode::from_usize(opcode - self.offset))
        } else {
            format!("{:?}", BaseAluImmOpcode::from_usize(opcode - self.offset))
        }
    }

    fn execute(
        &self,
        state: VmStateMut<TracingMemory, RA>,
        instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError> {
        debug_assert_eq!(
            instruction.opcode.local_opcode_idx(self.offset),
            self.local_opcode
        );
        let c_u32 = instruction.c.as_canonical_u32();
        debug_assert!(is_canonical_i12(c_u32));

        let (mut adapter_record, core_record) = state.ctx.alloc(EmptyAdapterCoreLayout::new());

        A::start(*state.pc, state.memory, &mut adapter_record);

        [core_record.rs1] = self
            .adapter
            .read(state.memory, instruction, &mut adapter_record)
            .into();

        core_record.imm_low11 = (c_u32 & 0x7FF) as u16;
        core_record.imm_sign = ((c_u32 >> 11) & 1) as u16;

        let rd = run_addi::<NUM_LIMBS, LIMB_BITS>(
            &core_record.rs1,
            core_record.imm_low11,
            core_record.imm_sign,
        );

        self.adapter
            .write(state.memory, instruction, [rd].into(), &mut adapter_record);

        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);

        Ok(())
    }
}

impl<F, A, const NUM_LIMBS: usize, const LIMB_BITS: usize, const RANGE_CHECK_TOP_LIMB: bool>
    TraceFiller<F> for AddIFiller<A, NUM_LIMBS, LIMB_BITS, RANGE_CHECK_TOP_LIMB>
where
    F: PrimeField32,
    A: 'static + AdapterTraceFiller<F>,
{
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        let (adapter_row, mut core_row) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };
        self.adapter.fill_trace_row(mem_helper, adapter_row);
        let record: &AddICoreRecord<NUM_LIMBS> =
            unsafe { get_record_from_slice(&mut core_row, ()) };
        let core_row: &mut AddICoreCols<F, NUM_LIMBS, LIMB_BITS> = core_row.borrow_mut();

        let rd = run_addi::<NUM_LIMBS, LIMB_BITS>(&record.rs1, record.imm_low11, record.imm_sign);

        core_row.is_valid = F::ONE;
        core_row.imm_sign = F::from_u16(record.imm_sign);
        core_row.imm_low11 = F::from_u16(record.imm_low11);
        self.range_checker_chip
            .add_count(record.imm_low11 as u32, 11);
        core_row.rs1 = record.rs1.map(F::from_u16);
        core_row.rd = rd.map(F::from_u16);
        let range_limb_count = NUM_LIMBS - usize::from(!RANGE_CHECK_TOP_LIMB);
        for &rd_val in &rd[..range_limb_count] {
            self.range_checker_chip.add_count(rd_val as u32, LIMB_BITS);
        }
    }
}

#[inline(always)]
pub(crate) fn run_addi<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    rs1: &[u16; NUM_LIMBS],
    imm_low11: u16,
    imm_sign: u16,
) -> [u16; NUM_LIMBS] {
    debug_assert!(NUM_LIMBS > 0 && (12..=U16_BITS).contains(&LIMB_BITS));

    let mut z = [0u16; NUM_LIMBS];
    let limb_base = 1u32 << LIMB_BITS;
    let limb_mask = limb_base - 1;

    let mut overflow = rs1[0] as u32 + imm_low11 as u32 + imm_sign as u32 * (limb_base - (1 << 11));
    let mut carry = overflow >> LIMB_BITS;
    z[0] = (overflow & limb_mask) as u16;

    let sign_limb = imm_sign as u32 * limb_mask;
    for i in 1..NUM_LIMBS {
        overflow = rs1[i] as u32 + sign_limb + carry;
        carry = overflow >> LIMB_BITS;
        z[i] = (overflow & limb_mask) as u16;
    }
    z
}
