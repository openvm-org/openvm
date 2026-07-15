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
use openvm_riscv_transpiler::ShiftOpcode;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{AirBuilder, BaseAir},
    p3_field::{Field, PrimeCharacteristicRing, PrimeField32},
    BaseAirWithPublicValues,
};

#[repr(C)]
#[derive(AlignedBorrow, StructReflection, Clone, Copy, Debug)]
pub struct ShiftRightArithmeticCoreCols<T, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub a: [T; NUM_LIMBS],
    pub b: [T; NUM_LIMBS],
    pub c: [T; NUM_LIMBS],

    // Sign of b for SRA
    pub b_sign: T,

    // Boolean columns that are 1 exactly at the index of the bit/limb shift amount
    pub bit_shift_marker: [T; LIMB_BITS],
    pub limb_shift_marker: [T; NUM_LIMBS],

    // Right-shift decomposition of each b[k] into the low `bit_shift` bits that cross into the
    // previous limb (`carry`) and the high bits that stay (`aux`):
    //   b[k] = carry[k] + aux[k] * 2^bit_shift
    // `carry` is range checked to bit_shift bits, `aux` to LIMB_BITS - bit_shift bits.
    pub bit_shift_carry: [T; NUM_LIMBS],
    pub bit_shift_aux: [T; NUM_LIMBS],
}

/// Arithmetic shift-right AIR (SRA) over u16 limbs.
///
/// To stay sound at `LIMB_BITS = 16`, each `b` limb is split into `carry`/`aux` parts and each
/// output limb is recombined additively so every constraint term stays below BabyBear's modulus.
/// The vacated high bits are filled with the sign bit of the top limb.
///
/// Note: when the shift amount from operand is greater than the number of bits, only shift
/// `shift_amount % num_bits` bits. This matches the RISC-V specs for SRA.
#[derive(Copy, Clone, Debug, derive_new::new, ColumnsAir)]
#[columns_via(ShiftRightArithmeticCoreCols<u8, NUM_LIMBS, LIMB_BITS>)]
pub struct ShiftRightArithmeticCoreAir<const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub range_bus: VariableRangeCheckerBus,
    pub offset: usize,
}

impl<F: Field, const NUM_LIMBS: usize, const LIMB_BITS: usize> BaseAir<F>
    for ShiftRightArithmeticCoreAir<NUM_LIMBS, LIMB_BITS>
{
    fn width(&self) -> usize {
        ShiftRightArithmeticCoreCols::<F, NUM_LIMBS, LIMB_BITS>::width()
    }
}
impl<F: Field, const NUM_LIMBS: usize, const LIMB_BITS: usize> BaseAirWithPublicValues<F>
    for ShiftRightArithmeticCoreAir<NUM_LIMBS, LIMB_BITS>
{
}

impl<AB, I, const NUM_LIMBS: usize, const LIMB_BITS: usize> VmCoreAir<AB, I>
    for ShiftRightArithmeticCoreAir<NUM_LIMBS, LIMB_BITS>
where
    AB: InteractionBuilder,
    I: VmAdapterInterface<AB::Expr>,
    I::Reads: From<[[AB::Expr; NUM_LIMBS]; 2]>,
    I::Writes: From<[[AB::Expr; NUM_LIMBS]; 1]>,
    I::ProcessedInstruction: From<MinimalInstruction<AB::Expr>>,
{
    fn eval(
        &self,
        builder: &mut AB,
        local_core: &[AB::Var],
        _from_pc: AB::Var,
    ) -> AdapterAirContext<AB::Expr, I> {
        let cols: &ShiftRightArithmeticCoreCols<_, NUM_LIMBS, LIMB_BITS> = local_core.borrow();
        let a = &cols.a;
        let b = &cols.b;
        let c = &cols.c;

        let mut bit_marker_sum = AB::Expr::ZERO;
        let mut bit_shift = AB::Expr::ZERO;
        let mut bit_multiplier = AB::Expr::ZERO;
        let mut carry_multiplier = AB::Expr::ZERO;

        for i in 0..LIMB_BITS {
            builder.assert_bool(cols.bit_shift_marker[i]);
            let marker: AB::Expr = cols.bit_shift_marker[i].into();
            bit_marker_sum += marker.clone();
            bit_shift += AB::Expr::from_usize(i) * marker.clone();
            bit_multiplier += AB::Expr::from_usize(1 << i) * marker.clone();
            carry_multiplier += AB::Expr::from_usize(1 << (LIMB_BITS - i)) * marker;
        }
        builder.assert_bool(bit_marker_sum.clone());
        let is_valid = bit_marker_sum;

        // Decompose each b[k] into carry/aux parts: b[k] = carry[k] + aux[k] * 2^bit_shift.
        for (k, &b_limb) in b.iter().enumerate() {
            builder.assert_eq(
                b_limb,
                cols.bit_shift_carry[k] + cols.bit_shift_aux[k] * bit_multiplier.clone(),
            );
        }

        // Check that a is the result of SRA b c. Each output limb is recombined additively from the
        // (already range-checked) carry/aux parts, so the result is automatically in
        // [0, 2^LIMB_BITS). The vacated high bits are filled with the sign bit.
        let mut limb_marker_sum = AB::Expr::ZERO;
        let mut limb_shift = AB::Expr::ZERO;
        for i in 0..NUM_LIMBS {
            builder.assert_bool(cols.limb_shift_marker[i]);
            limb_marker_sum += cols.limb_shift_marker[i].into();
            limb_shift += AB::Expr::from_usize(i) * cols.limb_shift_marker[i];

            let mut when_limb_shift = builder.when(cols.limb_shift_marker[i]);

            for (j, &a_limb) in a.iter().enumerate() {
                // a[j] = aux[j+i] + carry_in * 2^(LIMB_BITS - bit_shift)
                if j + i > NUM_LIMBS - 1 {
                    // Fully sign-extended limb.
                    when_limb_shift.assert_eq(
                        a_limb,
                        cols.b_sign * AB::F::from_usize((1 << LIMB_BITS) - 1),
                    );
                } else {
                    // For the top limb the bits shifted in from above are the sign bit, i.e.
                    // carry_in = b_sign * (2^bit_shift - 1). Since
                    // (2^bit_shift - 1) * carry_multiplier = 2^LIMB_BITS - carry_multiplier,
                    // we can write the fill without forming a triple product.
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

        // Check that bit_shift and limb_shift are correct.
        let num_bits = AB::F::from_usize(NUM_LIMBS * LIMB_BITS);
        self.range_bus
            .range_check(
                (c[0] - limb_shift * AB::F::from_usize(LIMB_BITS) - bit_shift.clone())
                    * num_bits.inverse(),
                LIMB_BITS - ((NUM_LIMBS * LIMB_BITS) as u32).ilog2() as usize,
            )
            .eval(builder, is_valid.clone());

        // Constrain b_sign to be the sign bit of b[NUM_LIMBS - 1]. b arrives range checked to
        // [0, 2^LIMB_BITS) from the u16 memory bus, so range checking the low LIMB_BITS-1 bits of
        // b[NUM_LIMBS-1] (i.e. b[NUM_LIMBS-1] minus the contribution of the sign bit) forces b_sign
        // to equal its top bit.
        builder.assert_bool(cols.b_sign);
        self.range_bus
            .range_check(
                b[NUM_LIMBS - 1] - cols.b_sign * AB::F::from_u32(1 << (LIMB_BITS - 1)),
                LIMB_BITS - 1,
            )
            .eval(builder, is_valid.clone());

        // Range check the carry/aux decomposition of each b limb. b and c arrive range checked to
        // [0, 2^LIMB_BITS) from the u16 memory bus, so no further bounds on b/c are needed; a is
        // bounded implicitly by the additive recombination above.
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
            AB::Expr::from_u8(ShiftOpcode::SRA as u8),
        );

        AdapterAirContext {
            to_pc: None,
            reads: [cols.b.map(Into::into), cols.c.map(Into::into)].into(),
            writes: [cols.a.map(Into::into)].into(),
            instruction: MinimalInstruction {
                is_valid,
                opcode: expected_opcode,
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
pub struct ShiftRightArithmeticCoreRecord<const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub b: [u16; NUM_LIMBS],
    pub c: [u16; NUM_LIMBS],
}

#[derive(Clone, Copy, derive_new::new)]
pub struct ShiftRightArithmeticExecutor<A, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    adapter: A,
    pub offset: usize,
}

#[derive(Clone, derive_new::new)]
pub struct ShiftRightArithmeticFiller<A, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    adapter: A,
    pub range_checker_chip: SharedVariableRangeCheckerChip,
    pub offset: usize,
}

impl<F, A, RA, const NUM_LIMBS: usize, const LIMB_BITS: usize> PreflightExecutor<F, RA>
    for ShiftRightArithmeticExecutor<A, NUM_LIMBS, LIMB_BITS>
where
    F: PrimeField32,
    A: 'static
        + AdapterTraceExecutor<
            F,
            ReadData: Into<[[u16; NUM_LIMBS]; 2]>,
            WriteData: From<[[u16; NUM_LIMBS]; 1]>,
        >,
    for<'buf> RA: RecordArena<
        'buf,
        EmptyAdapterCoreLayout<F, A>,
        (
            A::RecordMut<'buf>,
            &'buf mut ShiftRightArithmeticCoreRecord<NUM_LIMBS, LIMB_BITS>,
        ),
    >,
{
    fn get_opcode_name(&self, opcode: usize) -> String {
        format!("{:?}", ShiftOpcode::from_usize(opcode - self.offset))
    }

    fn execute(
        &self,
        state: VmStateMut<TracingMemory, RA>,
        instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError> {
        let Instruction { opcode, .. } = instruction;

        let local_opcode = ShiftOpcode::from_usize(opcode.local_opcode_idx(self.offset));
        debug_assert_eq!(local_opcode, ShiftOpcode::SRA);

        let (mut adapter_record, core_record) = state.ctx.alloc(EmptyAdapterCoreLayout::new());

        A::start(*state.pc, state.memory, &mut adapter_record);

        let [rs1, rs2] = self
            .adapter
            .read(state.memory, instruction, &mut adapter_record)
            .into();

        let (output, _, _) = run_shift_right_arithmetic::<NUM_LIMBS, LIMB_BITS>(&rs1, &rs2);

        core_record.b = rs1;
        core_record.c = rs2;

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
    for ShiftRightArithmeticFiller<A, NUM_LIMBS, LIMB_BITS>
where
    F: PrimeField32,
    A: 'static + AdapterTraceFiller<F>,
{
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        // SAFETY: row_slice is guaranteed by the caller to have at least A::WIDTH +
        // ShiftRightArithmeticCoreCols::width() elements
        let (adapter_row, mut core_row) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };
        self.adapter.fill_trace_row(mem_helper, adapter_row);
        // SAFETY: core_row contains a valid ShiftRightArithmeticCoreRecord written by the
        // executor during trace generation
        let record: &ShiftRightArithmeticCoreRecord<NUM_LIMBS, LIMB_BITS> =
            unsafe { get_record_from_slice(&mut core_row, ()) };

        let (a, limb_shift, bit_shift) =
            run_shift_right_arithmetic::<NUM_LIMBS, LIMB_BITS>(&record.b, &record.c);

        let num_bits_log = (NUM_LIMBS * LIMB_BITS).ilog2();
        self.range_checker_chip.add_count(
            ((record.c[0] as usize - bit_shift - limb_shift * LIMB_BITS) >> num_bits_log) as u32,
            LIMB_BITS - num_bits_log as usize,
        );

        // carry = the low bit_shift bits of b[k] that cross into the previous limb;
        // aux = the remaining (LIMB_BITS - bit_shift) bits. Both are computed via u32 intermediates
        // so a zero shift amount does not produce a shift-by-LIMB_BITS overflow.
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
        // b_sign correctness: range check the low LIMB_BITS-1 bits of b[NUM_LIMBS - 1].
        self.range_checker_chip.add_count(
            (record.b[NUM_LIMBS - 1] as u32) - ((b_sign as u32) << (LIMB_BITS - 1)),
            LIMB_BITS - 1,
        );

        let core_row: &mut ShiftRightArithmeticCoreCols<F, NUM_LIMBS, LIMB_BITS> =
            core_row.borrow_mut();
        core_row.b_sign = F::from_u16(b_sign);
        core_row.bit_shift_marker = bit_shift_marker;
        core_row.limb_shift_marker = limb_shift_marker;
        core_row.bit_shift_aux = bit_shift_aux;
        core_row.bit_shift_carry = bit_shift_carry;
        core_row.c = record.c.map(F::from_u16);
        core_row.b = record.b.map(F::from_u16);
        core_row.a = a.map(F::from_u16);
    }
}

// Returns (result, limb_shift, bit_shift)
#[inline(always)]
pub(crate) fn run_shift_right_arithmetic<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    x: &[u16; NUM_LIMBS],
    y: &[u16; NUM_LIMBS],
) -> ([u16; NUM_LIMBS], usize, usize) {
    let fill = (((1u32 << LIMB_BITS) - 1) as u16) * (x[NUM_LIMBS - 1] >> (LIMB_BITS - 1));
    let mut result = [fill; NUM_LIMBS];

    let (limb_shift, bit_shift) = get_shift::<NUM_LIMBS, LIMB_BITS>(y);

    for i in 0..(NUM_LIMBS - limb_shift) {
        let res = if i + limb_shift + 1 < NUM_LIMBS {
            (((x[i + limb_shift] as u32) >> bit_shift)
                | ((x[i + limb_shift + 1] as u32) << (LIMB_BITS - bit_shift)))
                % (1u32 << LIMB_BITS)
        } else {
            (((x[i + limb_shift] as u32) >> bit_shift) | ((fill as u32) << (LIMB_BITS - bit_shift)))
                % (1u32 << LIMB_BITS)
        };
        result[i] = res as u16;
    }
    (result, limb_shift, bit_shift)
}

#[inline(always)]
fn get_shift<const NUM_LIMBS: usize, const LIMB_BITS: usize>(y: &[u16]) -> (usize, usize) {
    debug_assert!(NUM_LIMBS * LIMB_BITS <= (1 << LIMB_BITS));
    // We assume `NUM_LIMBS * LIMB_BITS <= 2^LIMB_BITS` so the shift is defined
    // entirely in y[0].
    let shift = (y[0] as usize) % (NUM_LIMBS * LIMB_BITS);
    (shift / LIMB_BITS, shift % LIMB_BITS)
}
