use std::{
    array,
    borrow::{Borrow, BorrowMut},
    sync::Arc,
};

use ax_circuit_derive::AlignedBorrow;
use ax_circuit_primitives::{
    utils::not,
    var_range::{VariableRangeCheckerBus, VariableRangeCheckerChip},
    xor::{XorBus, XorLookupChip},
};
use ax_stark_backend::{interaction::InteractionBuilder, rap::BaseAirWithPublicValues};
use axvm_instructions::instruction::Instruction;
use p3_air::{AirBuilder, BaseAir};
use p3_field::{AbstractField, Field, PrimeField32};
use strum::IntoEnumIterator;

use crate::arch::{
    instructions::{ShiftOpcode, UsizeOpcode},
    AdapterAirContext, AdapterRuntimeContext, MinimalInstruction, Result, VmAdapterInterface,
    VmCoreAir, VmCoreChip,
};

#[repr(C)]
#[derive(AlignedBorrow, Clone, Copy, Debug)]
pub struct ShiftCoreCols<T, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub a: [T; NUM_LIMBS],
    pub b: [T; NUM_LIMBS],
    pub c: [T; NUM_LIMBS],

    pub opcode_sll_flag: T,
    pub opcode_srl_flag: T,
    pub opcode_sra_flag: T,

    // bit_multiplier = 2^bit_shift
    pub bit_shift: T,
    pub bit_multiplier_left: T,
    pub bit_multiplier_right: T,

    // Sign of x for SRA
    pub b_sign: T,

    // Boolean columns that are 1 exactly at the index of the bit/limb shift amount
    pub bit_shift_marker: [T; LIMB_BITS],
    pub limb_shift_marker: [T; NUM_LIMBS],

    // Part of each x[i] that gets bit shifted to the next limb
    pub bit_shift_carry: [T; NUM_LIMBS],
}

#[derive(Copy, Clone, Debug)]
pub struct ShiftCoreAir<const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub xor_bus: XorBus,
    pub range_bus: VariableRangeCheckerBus,
    offset: usize,
}

impl<F: Field, const NUM_LIMBS: usize, const LIMB_BITS: usize> BaseAir<F>
    for ShiftCoreAir<NUM_LIMBS, LIMB_BITS>
{
    fn width(&self) -> usize {
        ShiftCoreCols::<F, NUM_LIMBS, LIMB_BITS>::width()
    }
}
impl<F: Field, const NUM_LIMBS: usize, const LIMB_BITS: usize> BaseAirWithPublicValues<F>
    for ShiftCoreAir<NUM_LIMBS, LIMB_BITS>
{
}

impl<AB, I, const NUM_LIMBS: usize, const LIMB_BITS: usize> VmCoreAir<AB, I>
    for ShiftCoreAir<NUM_LIMBS, LIMB_BITS>
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
        let cols: &ShiftCoreCols<_, NUM_LIMBS, LIMB_BITS> = local_core.borrow();
        let flags = [
            cols.opcode_sll_flag,
            cols.opcode_srl_flag,
            cols.opcode_sra_flag,
        ];

        let is_valid = flags.iter().fold(AB::Expr::zero(), |acc, &flag| {
            builder.assert_bool(flag);
            acc + flag.into()
        });
        builder.assert_bool(is_valid.clone());

        let a = &cols.a;
        let b = &cols.b;
        let c = &cols.c;
        let right_shift = cols.opcode_srl_flag + cols.opcode_sra_flag;

        // Constrain that bit_shift, bit_multiplier are correct, i.e. that bit_multiplier =
        // 1 << bit_shift. We check that bit_shift is correct below if c < NUM_LIMBS * LIMB_BITS,
        // otherwise we don't really care what its value is. Note that bit_shift < LIMB_BITS is
        // constrained in bridge.rs via the range checker.
        builder
            .when(cols.opcode_sll_flag)
            .assert_zero(cols.bit_multiplier_right);
        builder
            .when(right_shift.clone())
            .assert_zero(cols.bit_multiplier_left);

        for i in 0..LIMB_BITS {
            let mut when_bit_shift = builder.when(cols.bit_shift_marker[i]);
            when_bit_shift.assert_eq(cols.bit_shift, AB::F::from_canonical_usize(i));
            when_bit_shift.when(cols.opcode_sll_flag).assert_eq(
                cols.bit_multiplier_left,
                AB::F::from_canonical_usize(1 << i),
            );
            when_bit_shift.when(right_shift.clone()).assert_eq(
                cols.bit_multiplier_right,
                AB::F::from_canonical_usize(1 << i),
            );
        }

        builder.assert_bool(cols.b_sign);
        builder
            .when(not(cols.opcode_sra_flag))
            .assert_zero(cols.b_sign);

        // Check that a[i] = b[i] <</>> c[i] both on the bit and limb shift level if c <
        // NUM_LIMBS * LIMB_BITS.
        let mut marker_sum = AB::Expr::zero();
        for i in 0..NUM_LIMBS {
            marker_sum += cols.limb_shift_marker[i].into();
            builder.assert_bool(cols.limb_shift_marker[i]);

            let mut when_limb_shift = builder.when(cols.limb_shift_marker[i]);
            when_limb_shift.assert_eq(
                c[1] * AB::F::from_canonical_usize(1 << LIMB_BITS) + c[0] - cols.bit_shift,
                AB::F::from_canonical_usize(i * LIMB_BITS),
            );

            for j in 0..NUM_LIMBS {
                // SLL constraints
                if j < i {
                    when_limb_shift.assert_zero(a[j] * cols.opcode_sll_flag);
                } else {
                    let expected_a_left = if j - i == 0 {
                        AB::Expr::zero()
                    } else {
                        cols.bit_shift_carry[j - i - 1].into() * cols.opcode_sll_flag
                    } + b[j - i] * cols.bit_multiplier_left
                        - AB::Expr::from_canonical_usize(1 << LIMB_BITS)
                            * cols.bit_shift_carry[j - i]
                            * cols.opcode_sll_flag;
                    when_limb_shift.assert_eq(a[j] * cols.opcode_sll_flag, expected_a_left);
                }

                // SRL and SRA constraints. Combining with above would require an additional column.
                if j + i > NUM_LIMBS - 1 {
                    when_limb_shift.assert_eq(
                        a[j] * right_shift.clone(),
                        cols.b_sign * AB::F::from_canonical_usize((1 << LIMB_BITS) - 1),
                    );
                } else {
                    let expected_a_right = if j + i == NUM_LIMBS - 1 {
                        cols.b_sign * (cols.bit_multiplier_right - AB::F::one())
                    } else {
                        cols.bit_shift_carry[j + i + 1].into() * right_shift.clone()
                    } * AB::F::from_canonical_usize(1 << LIMB_BITS)
                        + right_shift.clone() * (b[j + i] - cols.bit_shift_carry[j + i]);
                    when_limb_shift.assert_eq(a[j] * cols.bit_multiplier_right, expected_a_right);
                }

                // Ensure c is defined entirely within c[0] and c[1] if limb shifting
                if j > 1 {
                    when_limb_shift.assert_zero(c[j]);
                }
            }
        }

        for a_val in a {
            builder.when(not::<AB::Expr>(marker_sum.clone())).assert_eq(
                *a_val,
                cols.b_sign * AB::F::from_canonical_usize((1 << LIMB_BITS) - 1),
            );
        }

        // Check that bit_shift < LIMB_BITS
        self.range_bus
            .range_check(cols.bit_shift, LIMB_BITS.ilog2() as usize)
            .eval(builder, is_valid.clone());

        // Check x_sign & x[NUM_LIMBS - 1] == x_sign using XOR
        let mask = AB::F::from_canonical_u32(1 << (LIMB_BITS - 1));
        let b_sign_shifted = cols.b_sign * mask;
        self.xor_bus
            .send(
                b[NUM_LIMBS - 1],
                mask,
                b[NUM_LIMBS - 1] + mask - (AB::Expr::from_canonical_u32(2) * b_sign_shifted),
            )
            .eval(builder, cols.opcode_sra_flag);

        for (a_val, carry) in a.iter().zip(cols.bit_shift_carry.iter()) {
            self.range_bus
                .range_check(*a_val, LIMB_BITS)
                .eval(builder, is_valid.clone());
            self.range_bus
                .send(*carry, cols.bit_shift)
                .eval(builder, is_valid.clone());
        }

        let expected_opcode = flags
            .iter()
            .zip(ShiftOpcode::iter())
            .fold(AB::Expr::zero(), |acc, (flag, opcode)| {
                acc + (*flag).into() * AB::Expr::from_canonical_u8(opcode as u8)
            })
            + AB::Expr::from_canonical_usize(self.offset);

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
}

#[derive(Clone, Debug)]
pub struct ShiftCoreRecord<T, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub opcode: ShiftOpcode,
    pub a: [T; NUM_LIMBS],
    pub b: [T; NUM_LIMBS],
    pub c: [T; NUM_LIMBS],
    pub bit_shift_carry: [T; NUM_LIMBS],
    pub bit_shift: usize,
    pub limb_shift: usize,
    pub b_sign: T,
}

#[derive(Debug)]
pub struct ShiftCoreChip<const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub air: ShiftCoreAir<NUM_LIMBS, LIMB_BITS>,
    pub xor_lookup_chip: Arc<XorLookupChip<LIMB_BITS>>,
    pub range_checker_chip: Arc<VariableRangeCheckerChip>,
}

impl<const NUM_LIMBS: usize, const LIMB_BITS: usize> ShiftCoreChip<NUM_LIMBS, LIMB_BITS> {
    pub fn new(
        xor_lookup_chip: Arc<XorLookupChip<LIMB_BITS>>,
        range_checker_chip: Arc<VariableRangeCheckerChip>,
        offset: usize,
    ) -> Self {
        Self {
            air: ShiftCoreAir {
                xor_bus: xor_lookup_chip.bus(),
                range_bus: range_checker_chip.bus(),
                offset,
            },
            xor_lookup_chip,
            range_checker_chip,
        }
    }
}

impl<F: PrimeField32, I: VmAdapterInterface<F>, const NUM_LIMBS: usize, const LIMB_BITS: usize>
    VmCoreChip<F, I> for ShiftCoreChip<NUM_LIMBS, LIMB_BITS>
where
    I::Reads: Into<[[F; NUM_LIMBS]; 2]>,
    I::Writes: From<[[F; NUM_LIMBS]; 1]>,
{
    type Record = ShiftCoreRecord<F, NUM_LIMBS, LIMB_BITS>;
    type Air = ShiftCoreAir<NUM_LIMBS, LIMB_BITS>;

    #[allow(clippy::type_complexity)]
    fn execute_instruction(
        &self,
        instruction: &Instruction<F>,
        _from_pc: u32,
        reads: I::Reads,
    ) -> Result<(AdapterRuntimeContext<F, I>, Self::Record)> {
        let Instruction { opcode, .. } = instruction;
        let shift_opcode = ShiftOpcode::from_usize(opcode - self.air.offset);

        let data: [[F; NUM_LIMBS]; 2] = reads.into();
        let b = data[0].map(|x| x.as_canonical_u32());
        let c = data[1].map(|y| y.as_canonical_u32());
        let (a, limb_shift, bit_shift) = run_shift::<NUM_LIMBS, LIMB_BITS>(shift_opcode, &b, &c);

        let bit_shift_carry = array::from_fn(|i| match shift_opcode {
            ShiftOpcode::SLL => b[i] >> (LIMB_BITS - bit_shift),
            _ => b[i] % (1 << bit_shift),
        });

        let mut b_sign = 0;
        if shift_opcode == ShiftOpcode::SRA {
            b_sign = b[NUM_LIMBS - 1] >> (LIMB_BITS - 1);
            self.xor_lookup_chip
                .request(b[NUM_LIMBS - 1], 1 << (LIMB_BITS - 1));
        }

        self.range_checker_chip
            .add_count(bit_shift as u32, LIMB_BITS.ilog2() as usize);
        for (a_val, carry_val) in a.iter().zip(bit_shift_carry.iter()) {
            self.range_checker_chip.add_count(*a_val, LIMB_BITS);
            self.range_checker_chip.add_count(*carry_val, bit_shift);
        }

        let output = AdapterRuntimeContext::without_pc([a.map(F::from_canonical_u32)]);
        let record = ShiftCoreRecord {
            opcode: shift_opcode,
            a: a.map(F::from_canonical_u32),
            b: data[0],
            c: data[1],
            bit_shift_carry: bit_shift_carry.map(F::from_canonical_u32),
            bit_shift,
            limb_shift,
            b_sign: F::from_canonical_u32(b_sign),
        };

        Ok((output, record))
    }

    fn get_opcode_name(&self, opcode: usize) -> String {
        format!("{:?}", ShiftOpcode::from_usize(opcode - self.air.offset))
    }

    fn generate_trace_row(&self, row_slice: &mut [F], record: Self::Record) {
        let row_slice: &mut ShiftCoreCols<_, NUM_LIMBS, LIMB_BITS> = row_slice.borrow_mut();
        row_slice.a = record.a;
        row_slice.b = record.b;
        row_slice.c = record.c;
        row_slice.bit_shift = F::from_canonical_usize(record.bit_shift);
        row_slice.bit_multiplier_left = match record.opcode {
            ShiftOpcode::SLL => F::from_canonical_usize(1 << record.bit_shift),
            _ => F::zero(),
        };
        row_slice.bit_multiplier_right = match record.opcode {
            ShiftOpcode::SLL => F::zero(),
            _ => F::from_canonical_usize(1 << record.bit_shift),
        };
        row_slice.b_sign = record.b_sign;
        row_slice.bit_shift_marker = array::from_fn(|i| F::from_bool(i == record.bit_shift));
        row_slice.limb_shift_marker = array::from_fn(|i| F::from_bool(i == record.limb_shift));
        row_slice.bit_shift_carry = record.bit_shift_carry;
        row_slice.opcode_sll_flag = F::from_bool(record.opcode == ShiftOpcode::SLL);
        row_slice.opcode_srl_flag = F::from_bool(record.opcode == ShiftOpcode::SRL);
        row_slice.opcode_sra_flag = F::from_bool(record.opcode == ShiftOpcode::SRA);
    }

    fn air(&self) -> &Self::Air {
        &self.air
    }
}

pub(super) fn run_shift<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    opcode: ShiftOpcode,
    x: &[u32; NUM_LIMBS],
    y: &[u32; NUM_LIMBS],
) -> ([u32; NUM_LIMBS], usize, usize) {
    match opcode {
        ShiftOpcode::SLL => run_shift_left::<NUM_LIMBS, LIMB_BITS>(x, y),
        ShiftOpcode::SRL => run_shift_right::<NUM_LIMBS, LIMB_BITS>(x, y, true),
        ShiftOpcode::SRA => run_shift_right::<NUM_LIMBS, LIMB_BITS>(x, y, false),
    }
}

fn run_shift_left<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    x: &[u32; NUM_LIMBS],
    y: &[u32; NUM_LIMBS],
) -> ([u32; NUM_LIMBS], usize, usize) {
    let mut result = [0u32; NUM_LIMBS];

    let (is_zero, limb_shift, bit_shift) = get_shift::<NUM_LIMBS, LIMB_BITS>(y);
    if is_zero {
        return (result, limb_shift, bit_shift);
    }

    for i in limb_shift..NUM_LIMBS {
        result[i] = if i > limb_shift {
            ((x[i - limb_shift] << bit_shift) + (x[i - limb_shift - 1] >> (LIMB_BITS - bit_shift)))
                % (1 << LIMB_BITS)
        } else {
            (x[i - limb_shift] << bit_shift) % (1 << LIMB_BITS)
        };
    }
    (result, limb_shift, bit_shift)
}

fn run_shift_right<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    x: &[u32; NUM_LIMBS],
    y: &[u32; NUM_LIMBS],
    logical: bool,
) -> ([u32; NUM_LIMBS], usize, usize) {
    let fill = if logical {
        0
    } else {
        ((1 << LIMB_BITS) - 1) * (x[NUM_LIMBS - 1] >> (LIMB_BITS - 1))
    };
    let mut result = [fill; NUM_LIMBS];

    let (is_zero, limb_shift, bit_shift) = get_shift::<NUM_LIMBS, LIMB_BITS>(y);
    if is_zero {
        return (result, limb_shift, bit_shift);
    }

    for i in 0..(NUM_LIMBS - limb_shift) {
        result[i] = if i + limb_shift + 1 < NUM_LIMBS {
            ((x[i + limb_shift] >> bit_shift) + (x[i + limb_shift + 1] << (LIMB_BITS - bit_shift)))
                % (1 << LIMB_BITS)
        } else {
            ((x[i + limb_shift] >> bit_shift) + (fill << (LIMB_BITS - bit_shift)))
                % (1 << LIMB_BITS)
        }
    }
    (result, limb_shift, bit_shift)
}

fn get_shift<const NUM_LIMBS: usize, const LIMB_BITS: usize>(y: &[u32]) -> (bool, usize, usize) {
    // We assume `NUM_LIMBS * LIMB_BITS < 2^(2*LIMB_BITS)` so if there are any higher limbs,
    // the shifted value is zero.
    // TODO: revisit this, may be able to get away with defining the shift only in y[0]
    let shift = (y[0] + (y[1] * (1 << LIMB_BITS))) as usize;
    if shift < NUM_LIMBS * LIMB_BITS && y[2..].iter().all(|&val| val == 0) {
        (false, shift / LIMB_BITS, shift % LIMB_BITS)
    } else {
        (true, NUM_LIMBS, shift % LIMB_BITS)
    }
}
