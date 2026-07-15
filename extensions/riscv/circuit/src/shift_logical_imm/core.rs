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
use openvm_riscv_transpiler::{ShiftImmOpcode, ShiftOpcode};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{AirBuilder, BaseAir},
    p3_field::{Field, PrimeCharacteristicRing, PrimeField32},
    BaseAirWithPublicValues,
};

use crate::shift_logical::run_shift_logical;

/// Core columns for logical shifts with an immediate shift amount.
#[repr(C)]
#[derive(AlignedBorrow, StructReflection, Clone, Copy, Debug)]
pub struct ShiftLogicalImmCoreCols<T, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub a: [T; NUM_LIMBS],
    pub b: [T; NUM_LIMBS],

    pub opcode_sll_flag: T,
    pub opcode_srl_flag: T,

    pub bit_multiplier_left: T,
    pub carry_multiplier_left: T,

    pub bit_shift_marker: [T; LIMB_BITS],
    pub limb_shift_marker: [T; NUM_LIMBS],

    pub bit_shift_carry: [T; NUM_LIMBS],
    pub bit_shift_aux: [T; NUM_LIMBS],
}

/// Logical shift-by-immediate AIR (SLLI/SRLI) over u16 limbs.
///
/// The marker columns uniquely encode `shamt`, and the execution bridge binds
/// `limb_shift * LIMB_BITS + bit_shift` to the immediate operand.
#[derive(Copy, Clone, Debug, derive_new::new, ColumnsAir)]
#[columns_via(ShiftLogicalImmCoreCols<u8, NUM_LIMBS, LIMB_BITS>)]
pub struct ShiftLogicalImmCoreAir<const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub range_bus: VariableRangeCheckerBus,
    pub offset: usize,
}

impl<F: Field, const NUM_LIMBS: usize, const LIMB_BITS: usize> BaseAir<F>
    for ShiftLogicalImmCoreAir<NUM_LIMBS, LIMB_BITS>
{
    fn width(&self) -> usize {
        ShiftLogicalImmCoreCols::<F, NUM_LIMBS, LIMB_BITS>::width()
    }
}
impl<F: Field, const NUM_LIMBS: usize, const LIMB_BITS: usize> BaseAirWithPublicValues<F>
    for ShiftLogicalImmCoreAir<NUM_LIMBS, LIMB_BITS>
{
}

impl<AB, I, const NUM_LIMBS: usize, const LIMB_BITS: usize> VmCoreAir<AB, I>
    for ShiftLogicalImmCoreAir<NUM_LIMBS, LIMB_BITS>
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
        let cols: &ShiftLogicalImmCoreCols<_, NUM_LIMBS, LIMB_BITS> = local_core.borrow();
        let flags = [cols.opcode_sll_flag, cols.opcode_srl_flag];

        let is_valid = flags.iter().fold(AB::Expr::ZERO, |acc, &flag| {
            builder.assert_bool(flag);
            acc + flag.into()
        });
        builder.assert_bool(is_valid.clone());

        let a = &cols.a;
        let b = &cols.b;

        // Constrain that bit_shift and the (bit/carry) multipliers are correct.
        let mut bit_marker_sum = AB::Expr::ZERO;
        let mut bit_shift = AB::Expr::ZERO;
        let mut bit_multiplier = AB::Expr::ZERO;
        let mut carry_multiplier = AB::Expr::ZERO;

        for i in 0..LIMB_BITS {
            builder.assert_bool(cols.bit_shift_marker[i]);
            bit_marker_sum += cols.bit_shift_marker[i].into();
            bit_shift += AB::Expr::from_usize(i) * cols.bit_shift_marker[i];
            bit_multiplier += AB::Expr::from_usize(1 << i) * cols.bit_shift_marker[i];
            carry_multiplier +=
                AB::Expr::from_usize(1 << (LIMB_BITS - i)) * cols.bit_shift_marker[i];

            let mut when_bit_shift = builder.when(cols.bit_shift_marker[i]);
            when_bit_shift.assert_eq(
                cols.bit_multiplier_left,
                AB::Expr::from_usize(1 << i) * cols.opcode_sll_flag,
            );
            when_bit_shift.assert_eq(
                cols.carry_multiplier_left,
                AB::Expr::from_usize(1 << (LIMB_BITS - i)) * cols.opcode_sll_flag,
            );
        }
        builder.when(is_valid.clone()).assert_one(bit_marker_sum);

        // Decompose each b[k] into carry/aux parts (see ShiftLogicalCoreAir).
        for (k, &b_limb) in b.iter().enumerate() {
            builder.assert_eq(
                b_limb * cols.opcode_sll_flag,
                cols.bit_shift_aux[k] * cols.opcode_sll_flag
                    + cols.bit_shift_carry[k] * cols.carry_multiplier_left,
            );
            builder.assert_eq(
                b_limb * cols.opcode_srl_flag,
                cols.bit_shift_carry[k] * cols.opcode_srl_flag
                    + cols.bit_shift_aux[k] * (bit_multiplier.clone() - cols.bit_multiplier_left),
            );
        }

        // Check that a[i] = b[i] <</>> shamt both on the bit and limb shift level.
        let mut limb_marker_sum = AB::Expr::ZERO;
        let mut limb_shift = AB::Expr::ZERO;
        for i in 0..NUM_LIMBS {
            builder.assert_bool(cols.limb_shift_marker[i]);
            limb_marker_sum += cols.limb_shift_marker[i].into();
            limb_shift += AB::Expr::from_usize(i) * cols.limb_shift_marker[i];

            let mut when_limb_shift = builder.when(cols.limb_shift_marker[i]);

            for (j, &a_limb) in a.iter().enumerate() {
                // SLL: a[j] = aux[j-i] * 2^bit_shift + carry[j-i-1]
                if j < i {
                    when_limb_shift.assert_zero(a_limb * cols.opcode_sll_flag);
                } else {
                    let carry_in = if j - i == 0 {
                        AB::Expr::ZERO
                    } else {
                        cols.bit_shift_carry[j - i - 1].into() * cols.opcode_sll_flag
                    };
                    when_limb_shift.assert_eq(
                        a_limb * cols.opcode_sll_flag,
                        cols.bit_shift_aux[j - i] * cols.bit_multiplier_left + carry_in,
                    );
                }

                // SRL: a[j] = aux[j+i] + carry[j+i+1] * 2^(LIMB_BITS - bit_shift)
                if j + i > NUM_LIMBS - 1 {
                    when_limb_shift.assert_zero(a_limb * cols.opcode_srl_flag);
                } else {
                    let carry_in = if j + i == NUM_LIMBS - 1 {
                        AB::Expr::ZERO
                    } else {
                        cols.bit_shift_carry[j + i + 1].into()
                            * (carry_multiplier.clone() - cols.carry_multiplier_left)
                    };
                    when_limb_shift.assert_eq(
                        a_limb * cols.opcode_srl_flag,
                        cols.bit_shift_aux[j + i] * cols.opcode_srl_flag + carry_in,
                    );
                }
            }
        }
        builder.when(is_valid.clone()).assert_one(limb_marker_sum);

        // The immediate operand is exactly limb_shift * LIMB_BITS + bit_shift; both parts are
        // bounded by the marker-sum constraints, so no range check is needed.
        let imm = limb_shift * AB::Expr::from_usize(LIMB_BITS) + bit_shift.clone();

        // Range check the carry/aux decomposition of each b limb.
        let aux_bits = AB::Expr::from_usize(LIMB_BITS) - bit_shift.clone();
        for k in 0..NUM_LIMBS {
            self.range_bus
                .send(cols.bit_shift_carry[k], bit_shift.clone())
                .eval(builder, is_valid.clone());
            self.range_bus
                .send(cols.bit_shift_aux[k], aux_bits.clone())
                .eval(builder, is_valid.clone());
        }

        let expected_opcode = VmCoreAir::<AB, I>::expr_to_global_expr(
            self,
            [
                (cols.opcode_sll_flag, ShiftImmOpcode::SLLI),
                (cols.opcode_srl_flag, ShiftImmOpcode::SRLI),
            ]
            .iter()
            .fold(AB::Expr::ZERO, |acc, (flag, opcode)| {
                acc + (*flag).into() * AB::Expr::from_u8(*opcode as u8)
            }),
        );

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

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug)]
pub struct ShiftLogicalImmCoreRecord<const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub b: [u16; NUM_LIMBS],
    pub shamt: u8,
    pub local_opcode: u8,
}

#[derive(Clone, Copy, derive_new::new)]
pub struct ShiftLogicalImmExecutor<A, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    adapter: A,
    pub offset: usize,
}

#[derive(Clone, derive_new::new)]
pub struct ShiftLogicalImmFiller<A, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    adapter: A,
    pub range_checker_chip: SharedVariableRangeCheckerChip,
}

impl<F, A, RA, const NUM_LIMBS: usize, const LIMB_BITS: usize> PreflightExecutor<F, RA>
    for ShiftLogicalImmExecutor<A, NUM_LIMBS, LIMB_BITS>
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
            &'buf mut ShiftLogicalImmCoreRecord<NUM_LIMBS, LIMB_BITS>,
        ),
    >,
{
    fn get_opcode_name(&self, opcode: usize) -> String {
        format!("{:?}", ShiftImmOpcode::from_usize(opcode - self.offset))
    }

    fn execute(
        &self,
        state: VmStateMut<TracingMemory, RA>,
        instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError> {
        let Instruction { opcode, c, .. } = instruction;

        let local_opcode = ShiftImmOpcode::from_usize(opcode.local_opcode_idx(self.offset));
        debug_assert_ne!(local_opcode, ShiftImmOpcode::SRAI);

        let (mut adapter_record, core_record) = state.ctx.alloc(EmptyAdapterCoreLayout::new());

        A::start(*state.pc, state.memory, &mut adapter_record);

        [core_record.b] = self
            .adapter
            .read(state.memory, instruction, &mut adapter_record)
            .into();

        let shamt = c.as_canonical_u32();
        debug_assert!(shamt < (NUM_LIMBS * LIMB_BITS) as u32);
        core_record.shamt = shamt as u8;
        core_record.local_opcode = local_opcode as u8;

        let reg_opcode = if local_opcode == ShiftImmOpcode::SLLI {
            ShiftOpcode::SLL
        } else {
            ShiftOpcode::SRL
        };
        let mut shamt_limbs = [0u16; NUM_LIMBS];
        shamt_limbs[0] = shamt as u16;
        let (output, _, _) =
            run_shift_logical::<NUM_LIMBS, LIMB_BITS>(reg_opcode, &core_record.b, &shamt_limbs);

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
    for ShiftLogicalImmFiller<A, NUM_LIMBS, LIMB_BITS>
where
    F: PrimeField32,
    A: 'static + AdapterTraceFiller<F>,
{
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        let (adapter_row, mut core_row) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };
        self.adapter.fill_trace_row(mem_helper, adapter_row);
        let record: &ShiftLogicalImmCoreRecord<NUM_LIMBS, LIMB_BITS> =
            unsafe { get_record_from_slice(&mut core_row, ()) };

        let is_sll = record.local_opcode == ShiftImmOpcode::SLLI as u8;
        let reg_opcode = if is_sll {
            ShiftOpcode::SLL
        } else {
            ShiftOpcode::SRL
        };
        let mut shamt_limbs = [0u16; NUM_LIMBS];
        shamt_limbs[0] = record.shamt as u16;
        let (a, limb_shift, bit_shift) =
            run_shift_logical::<NUM_LIMBS, LIMB_BITS>(reg_opcode, &record.b, &shamt_limbs);

        let aux_bits = LIMB_BITS - bit_shift;
        let mut bit_shift_carry = [F::ZERO; NUM_LIMBS];
        let mut bit_shift_aux = [F::ZERO; NUM_LIMBS];
        for k in 0..NUM_LIMBS {
            let limb = record.b[k] as u32;
            let (carry, aux) = if is_sll {
                (limb >> aux_bits, limb & ((1u32 << aux_bits) - 1))
            } else {
                (limb & ((1u32 << bit_shift) - 1), limb >> bit_shift)
            };
            self.range_checker_chip.add_count(carry, bit_shift);
            self.range_checker_chip.add_count(aux, aux_bits);
            bit_shift_carry[k] = F::from_u32(carry);
            bit_shift_aux[k] = F::from_u32(aux);
        }

        let mut limb_shift_marker = [F::ZERO; NUM_LIMBS];
        limb_shift_marker[limb_shift] = F::ONE;
        let mut bit_shift_marker = [F::ZERO; LIMB_BITS];
        bit_shift_marker[bit_shift] = F::ONE;

        let b = record.b;
        let core_row: &mut ShiftLogicalImmCoreCols<F, NUM_LIMBS, LIMB_BITS> = core_row.borrow_mut();
        let bit_mult = F::from_u32(1 << bit_shift);
        let carry_mult = F::from_u32(1 << aux_bits);
        core_row.bit_shift_aux = bit_shift_aux;
        core_row.bit_shift_carry = bit_shift_carry;
        core_row.limb_shift_marker = limb_shift_marker;
        core_row.bit_shift_marker = bit_shift_marker;
        core_row.carry_multiplier_left = if is_sll { carry_mult } else { F::ZERO };
        core_row.bit_multiplier_left = if is_sll { bit_mult } else { F::ZERO };
        core_row.opcode_srl_flag = F::from_bool(!is_sll);
        core_row.opcode_sll_flag = F::from_bool(is_sll);
        core_row.b = b.map(F::from_u16);
        core_row.a = a.map(F::from_u16);
    }
}
