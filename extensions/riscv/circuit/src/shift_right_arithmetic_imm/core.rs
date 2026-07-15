use std::borrow::{Borrow, BorrowMut};

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
use openvm_riscv_transpiler::ShiftImmOpcode;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{AirBuilder, BaseAir},
    p3_field::{Field, PrimeCharacteristicRing, PrimeField32},
    BaseAirWithPublicValues,
};

use crate::shift_right_arithmetic::run_shift_right_arithmetic;

#[repr(C)]
#[derive(AlignedBorrow, StructReflection, Clone, Copy, Debug)]
pub struct ShiftRightArithmeticImmCoreCols<T, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub a: [T; NUM_LIMBS],
    pub b: [T; NUM_LIMBS],

    pub is_valid: T,
    pub bit_multiplier: T,
    pub carry_multiplier: T,
    pub b_sign: T,

    pub bit_shift_marker: [T; LIMB_BITS],
    pub limb_shift_marker: [T; NUM_LIMBS],
    pub bit_shift_carry: [T; NUM_LIMBS],
    pub bit_shift_aux: [T; NUM_LIMBS],
}

/// Arithmetic shift-right-by-immediate AIR over u16 limbs.
///
/// The marker columns uniquely encode a shift in `0..NUM_LIMBS * LIMB_BITS`; the execution
/// bridge binds that encoding directly to the instruction immediate. Consequently this core
/// needs neither immediate limbs nor the quotient range check used by the register SRA core.
#[derive(Copy, Clone, Debug, derive_new::new, ColumnsAir)]
#[columns_via(ShiftRightArithmeticImmCoreCols<u8, NUM_LIMBS, LIMB_BITS>)]
pub struct ShiftRightArithmeticImmCoreAir<const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub range_bus: VariableRangeCheckerBus,
    pub offset: usize,
}

impl<F: Field, const NUM_LIMBS: usize, const LIMB_BITS: usize> BaseAir<F>
    for ShiftRightArithmeticImmCoreAir<NUM_LIMBS, LIMB_BITS>
{
    fn width(&self) -> usize {
        ShiftRightArithmeticImmCoreCols::<F, NUM_LIMBS, LIMB_BITS>::width()
    }
}

impl<F: Field, const NUM_LIMBS: usize, const LIMB_BITS: usize> BaseAirWithPublicValues<F>
    for ShiftRightArithmeticImmCoreAir<NUM_LIMBS, LIMB_BITS>
{
}

impl<AB, I, const NUM_LIMBS: usize, const LIMB_BITS: usize> VmCoreAir<AB, I>
    for ShiftRightArithmeticImmCoreAir<NUM_LIMBS, LIMB_BITS>
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
        let cols: &ShiftRightArithmeticImmCoreCols<_, NUM_LIMBS, LIMB_BITS> = local_core.borrow();
        builder.assert_bool(cols.is_valid);
        let is_valid: AB::Expr = cols.is_valid.into();

        let mut bit_marker_sum = AB::Expr::ZERO;
        let mut bit_shift = AB::Expr::ZERO;
        for i in 0..LIMB_BITS {
            builder.assert_bool(cols.bit_shift_marker[i]);
            bit_marker_sum += cols.bit_shift_marker[i].into();
            bit_shift += AB::Expr::from_usize(i) * cols.bit_shift_marker[i];

            let mut when_bit_shift = builder.when(cols.bit_shift_marker[i]);
            when_bit_shift.assert_eq(
                cols.bit_multiplier,
                AB::Expr::from_usize(1 << i) * is_valid.clone(),
            );
            when_bit_shift.assert_eq(
                cols.carry_multiplier,
                AB::Expr::from_usize(1 << (LIMB_BITS - i)) * is_valid.clone(),
            );
        }
        builder.assert_eq(bit_marker_sum, is_valid.clone());

        for (k, &b_limb) in cols.b.iter().enumerate() {
            builder.assert_eq(
                b_limb,
                cols.bit_shift_carry[k] + cols.bit_shift_aux[k] * cols.bit_multiplier,
            );
        }

        let mut limb_marker_sum = AB::Expr::ZERO;
        let mut limb_shift = AB::Expr::ZERO;
        for i in 0..NUM_LIMBS {
            builder.assert_bool(cols.limb_shift_marker[i]);
            limb_marker_sum += cols.limb_shift_marker[i].into();
            limb_shift += AB::Expr::from_usize(i) * cols.limb_shift_marker[i];

            let mut when_limb_shift = builder.when(cols.limb_shift_marker[i]);
            let carry_multiplier: AB::Expr = cols.carry_multiplier.into();
            for (j, &a_limb) in cols.a.iter().enumerate() {
                if j + i > NUM_LIMBS - 1 {
                    when_limb_shift.assert_eq(
                        a_limb,
                        cols.b_sign * AB::F::from_usize((1 << LIMB_BITS) - 1),
                    );
                } else {
                    let carry_in = if j + i == NUM_LIMBS - 1 {
                        (AB::Expr::from_usize(1 << LIMB_BITS) - carry_multiplier.clone())
                            * cols.b_sign
                    } else {
                        carry_multiplier.clone() * cols.bit_shift_carry[j + i + 1]
                    };
                    when_limb_shift.assert_eq(a_limb, carry_in + cols.bit_shift_aux[j + i]);
                }
            }
        }
        builder.assert_eq(limb_marker_sum, is_valid.clone());

        builder.assert_bool(cols.b_sign);
        self.range_bus
            .range_check(
                cols.b[NUM_LIMBS - 1] - cols.b_sign * AB::F::from_u32(1 << (LIMB_BITS - 1)),
                LIMB_BITS - 1,
            )
            .eval(builder, is_valid.clone());

        let aux_bits = AB::Expr::from_usize(LIMB_BITS) - bit_shift.clone();
        for k in 0..NUM_LIMBS {
            self.range_bus
                .send(cols.bit_shift_carry[k], bit_shift.clone())
                .eval(builder, is_valid.clone());
            self.range_bus
                .send(cols.bit_shift_aux[k], aux_bits.clone())
                .eval(builder, is_valid.clone());
        }

        let immediate = limb_shift * AB::Expr::from_usize(LIMB_BITS) + bit_shift;
        let expected_opcode = VmCoreAir::<AB, I>::expr_to_global_expr(
            self,
            AB::Expr::from_u8(ShiftImmOpcode::SRAI as u8),
        );

        AdapterAirContext {
            to_pc: None,
            reads: [cols.b.map(Into::into)].into(),
            writes: [cols.a.map(Into::into)].into(),
            instruction: ImmInstruction {
                is_valid,
                opcode: expected_opcode,
                immediate,
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
pub struct ShiftRightArithmeticImmCoreRecord<const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub b: [u16; NUM_LIMBS],
    pub shamt: u8,
}

#[derive(Clone, Copy, derive_new::new)]
pub struct ShiftRightArithmeticImmExecutor<A, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    adapter: A,
    pub offset: usize,
}

#[derive(Clone, derive_new::new)]
pub struct ShiftRightArithmeticImmFiller<A, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    adapter: A,
    pub range_checker_chip: SharedVariableRangeCheckerChip,
}

impl<F, A, RA, const NUM_LIMBS: usize, const LIMB_BITS: usize> PreflightExecutor<F, RA>
    for ShiftRightArithmeticImmExecutor<A, NUM_LIMBS, LIMB_BITS>
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
        (
            A::RecordMut<'buf>,
            &'buf mut ShiftRightArithmeticImmCoreRecord<NUM_LIMBS, LIMB_BITS>,
        ),
    >,
{
    fn get_opcode_name(&self, opcode: usize) -> String {
        format!("{:?}", ShiftImmOpcode::from_usize(opcode - self.offset))
    }

    fn execute(
        &self,
        state: VmStateMut<F, TracingMemory, RA>,
        instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError> {
        let Instruction { opcode, c, .. } = instruction;
        debug_assert_eq!(
            ShiftImmOpcode::from_usize(opcode.local_opcode_idx(self.offset)),
            ShiftImmOpcode::SRAI
        );

        let shamt = c.as_canonical_u32();
        if shamt >= (NUM_LIMBS * LIMB_BITS) as u32 {
            return Err(ExecutionError::Fail {
                pc: *state.pc,
                msg: "SRAI shift amount out of range",
            });
        }

        let (mut adapter_record, core_record) = state.ctx.alloc(EmptyAdapterCoreLayout::new());
        A::start(*state.pc, state.memory, &mut adapter_record);
        [core_record.b] = self
            .adapter
            .read(state.memory, instruction, &mut adapter_record)
            .into();
        core_record.shamt = shamt as u8;

        let mut shamt_limbs = [0u16; NUM_LIMBS];
        shamt_limbs[0] = shamt as u16;
        let (output, _, _) =
            run_shift_right_arithmetic::<NUM_LIMBS, LIMB_BITS>(&core_record.b, &shamt_limbs);
        self.adapter.write(
            state.memory,
            instruction,
            [output].into(),
            &mut adapter_record,
        );
        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);
        Ok(())
    }
}

impl<F, A, const NUM_LIMBS: usize, const LIMB_BITS: usize> TraceFiller<F>
    for ShiftRightArithmeticImmFiller<A, NUM_LIMBS, LIMB_BITS>
where
    F: PrimeField32,
    A: 'static + AdapterTraceFiller<F>,
{
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        let (adapter_row, mut core_row) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };
        self.adapter.fill_trace_row(mem_helper, adapter_row);
        let record: &ShiftRightArithmeticImmCoreRecord<NUM_LIMBS, LIMB_BITS> =
            unsafe { get_record_from_slice(&mut core_row, ()) };

        let mut shamt_limbs = [0u16; NUM_LIMBS];
        shamt_limbs[0] = record.shamt as u16;
        let (a, limb_shift, bit_shift) =
            run_shift_right_arithmetic::<NUM_LIMBS, LIMB_BITS>(&record.b, &shamt_limbs);

        let aux_bits = LIMB_BITS - bit_shift;
        let mut bit_shift_carry = [F::ZERO; NUM_LIMBS];
        let mut bit_shift_aux = [F::ZERO; NUM_LIMBS];
        for k in 0..NUM_LIMBS {
            let limb = record.b[k] as u32;
            let carry = limb & ((1u32 << bit_shift) - 1);
            let aux = limb >> bit_shift;
            self.range_checker_chip.add_count(carry, bit_shift);
            self.range_checker_chip.add_count(aux, aux_bits);
            bit_shift_carry[k] = F::from_u32(carry);
            bit_shift_aux[k] = F::from_u32(aux);
        }

        let mut limb_shift_marker = [F::ZERO; NUM_LIMBS];
        limb_shift_marker[limb_shift] = F::ONE;
        let mut bit_shift_marker = [F::ZERO; LIMB_BITS];
        bit_shift_marker[bit_shift] = F::ONE;

        let b_sign = record.b[NUM_LIMBS - 1] >> (LIMB_BITS - 1);
        self.range_checker_chip.add_count(
            (record.b[NUM_LIMBS - 1] as u32) - ((b_sign as u32) << (LIMB_BITS - 1)),
            LIMB_BITS - 1,
        );

        let core_row: &mut ShiftRightArithmeticImmCoreCols<F, NUM_LIMBS, LIMB_BITS> =
            core_row.borrow_mut();
        core_row.is_valid = F::ONE;
        core_row.bit_multiplier = F::from_u32(1 << bit_shift);
        core_row.carry_multiplier = F::from_u32(1 << aux_bits);
        core_row.b_sign = F::from_u16(b_sign);
        core_row.bit_shift_marker = bit_shift_marker;
        core_row.limb_shift_marker = limb_shift_marker;
        core_row.bit_shift_aux = bit_shift_aux;
        core_row.bit_shift_carry = bit_shift_carry;
        core_row.b = record.b.map(F::from_u16);
        core_row.a = a.map(F::from_u16);
    }
}
