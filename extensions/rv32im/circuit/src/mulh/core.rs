use std::{
    array,
    borrow::{Borrow, BorrowMut},
    sync::Arc,
};

use ax_circuit_derive::AlignedBorrow;
use ax_circuit_primitives::{
    bitwise_op_lookup::{BitwiseOperationLookupBus, BitwiseOperationLookupChip},
    range_tuple::{RangeTupleCheckerBus, RangeTupleCheckerChip},
};
use ax_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{AirBuilder, BaseAir},
    p3_field::{AbstractField, Field, PrimeField32},
    rap::BaseAirWithPublicValues,
};
use openvm_circuit::arch::{
    AdapterAirContext, AdapterRuntimeContext, MinimalInstruction, Result, VmAdapterInterface,
    VmCoreAir, VmCoreChip,
};
use openvm_instructions::{instruction::Instruction, UsizeOpcode};
use openvm_rv32im_transpiler::MulHOpcode;
use strum::IntoEnumIterator;

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct MulHCoreCols<T, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub a: [T; NUM_LIMBS],
    pub b: [T; NUM_LIMBS],
    pub c: [T; NUM_LIMBS],

    pub a_mul: [T; NUM_LIMBS],
    pub b_ext: T,
    pub c_ext: T,

    pub opcode_mulh_flag: T,
    pub opcode_mulhsu_flag: T,
    pub opcode_mulhu_flag: T,
}

#[derive(Copy, Clone, Debug)]
pub struct MulHCoreAir<const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub bitwise_lookup_bus: BitwiseOperationLookupBus,
    pub range_tuple_bus: RangeTupleCheckerBus<2>,
    offset: usize,
}

impl<F: Field, const NUM_LIMBS: usize, const LIMB_BITS: usize> BaseAir<F>
    for MulHCoreAir<NUM_LIMBS, LIMB_BITS>
{
    fn width(&self) -> usize {
        MulHCoreCols::<F, NUM_LIMBS, LIMB_BITS>::width()
    }
}
impl<F: Field, const NUM_LIMBS: usize, const LIMB_BITS: usize> BaseAirWithPublicValues<F>
    for MulHCoreAir<NUM_LIMBS, LIMB_BITS>
{
}

impl<AB, I, const NUM_LIMBS: usize, const LIMB_BITS: usize> VmCoreAir<AB, I>
    for MulHCoreAir<NUM_LIMBS, LIMB_BITS>
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
        let cols: &MulHCoreCols<_, NUM_LIMBS, LIMB_BITS> = local_core.borrow();
        let flags = [
            cols.opcode_mulh_flag,
            cols.opcode_mulhsu_flag,
            cols.opcode_mulhu_flag,
        ];

        let is_valid = flags.iter().fold(AB::Expr::ZERO, |acc, &flag| {
            builder.assert_bool(flag);
            acc + flag.into()
        });
        builder.assert_bool(is_valid.clone());

        let b = &cols.b;
        let c = &cols.c;
        let carry_divide = AB::F::from_canonical_u32(1 << LIMB_BITS).inverse();

        // Note b * c = a << LIMB_BITS + a_mul, in order to constrain that a is correct we
        // need to compute the carries generated by a_mul.
        let a_mul = &cols.a_mul;
        let mut carry_mul: [AB::Expr; NUM_LIMBS] = array::from_fn(|_| AB::Expr::ZERO);

        for i in 0..NUM_LIMBS {
            let expected_limb = if i == 0 {
                AB::Expr::ZERO
            } else {
                carry_mul[i - 1].clone()
            } + (0..=i).fold(AB::Expr::ZERO, |ac, k| ac + (b[k] * c[i - k]));
            carry_mul[i] = AB::Expr::from(carry_divide) * (expected_limb - a_mul[i]);
        }

        for (a_mul, carry_mul) in a_mul.iter().zip(carry_mul.iter()) {
            self.range_tuple_bus
                .send(vec![(*a_mul).into(), carry_mul.clone()])
                .eval(builder, is_valid.clone());
        }

        // We can now constrain that a is correct using carry_mul[NUM_LIMBS - 1]
        let a = &cols.a;
        let mut carry: [AB::Expr; NUM_LIMBS] = array::from_fn(|_| AB::Expr::ZERO);

        for j in 0..NUM_LIMBS {
            let expected_limb = if j == 0 {
                carry_mul[NUM_LIMBS - 1].clone()
            } else {
                carry[j - 1].clone()
            } + ((j + 1)..NUM_LIMBS)
                .fold(AB::Expr::ZERO, |acc, k| acc + (b[k] * c[NUM_LIMBS + j - k]))
                + (0..(j + 1)).fold(AB::Expr::ZERO, |acc, k| {
                    acc + (b[k] * cols.c_ext) + (c[k] * cols.b_ext)
                });
            carry[j] = AB::Expr::from(carry_divide) * (expected_limb - a[j]);
        }

        for (a, carry) in a.iter().zip(carry.iter()) {
            self.range_tuple_bus
                .send(vec![(*a).into(), carry.clone()])
                .eval(builder, is_valid.clone());
        }

        // Check that b_ext and c_ext are correct using bitwise lookup. We check
        // both b and c when the opcode is MULH, and only b when MULHSU.
        let sign_mask = AB::F::from_canonical_u32(1 << (LIMB_BITS - 1));
        let ext_inv = AB::F::from_canonical_u32((1 << LIMB_BITS) - 1).inverse();
        let b_sign = cols.b_ext * ext_inv;
        let c_sign = cols.c_ext * ext_inv;

        builder.assert_bool(b_sign.clone());
        builder.assert_bool(c_sign.clone());
        builder
            .when(cols.opcode_mulhu_flag)
            .assert_zero(b_sign.clone());
        builder
            .when(cols.opcode_mulhu_flag + cols.opcode_mulhsu_flag)
            .assert_zero(c_sign.clone());

        self.bitwise_lookup_bus
            .send_range(
                AB::Expr::from_canonical_u32(2) * (b[NUM_LIMBS - 1] - b_sign * sign_mask),
                (cols.opcode_mulh_flag + AB::Expr::ONE) * (c[NUM_LIMBS - 1] - c_sign * sign_mask),
            )
            .eval(builder, cols.opcode_mulh_flag + cols.opcode_mulhsu_flag);

        let expected_opcode = flags.iter().zip(MulHOpcode::iter()).fold(
            AB::Expr::ZERO,
            |acc, (flag, local_opcode)| {
                acc + (*flag).into() * AB::Expr::from_canonical_u8(local_opcode as u8)
            },
        ) + AB::Expr::from_canonical_usize(self.offset);

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

#[derive(Debug)]
pub struct MulHCoreChip<const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub air: MulHCoreAir<NUM_LIMBS, LIMB_BITS>,
    pub bitwise_lookup_chip: Arc<BitwiseOperationLookupChip<LIMB_BITS>>,
    pub range_tuple_chip: Arc<RangeTupleCheckerChip<2>>,
}

impl<const NUM_LIMBS: usize, const LIMB_BITS: usize> MulHCoreChip<NUM_LIMBS, LIMB_BITS> {
    pub fn new(
        bitwise_lookup_chip: Arc<BitwiseOperationLookupChip<LIMB_BITS>>,
        range_tuple_chip: Arc<RangeTupleCheckerChip<2>>,
        offset: usize,
    ) -> Self {
        // The RangeTupleChecker is used to range check (a[i], carry[i]) pairs where 0 <= i
        // < 2 * NUM_LIMBS. a[i] must have LIMB_BITS bits and carry[i] is the sum of i + 1
        // bytes (with LIMB_BITS bits). BitwiseOperationLookup is used to sign check bytes.
        debug_assert!(
            range_tuple_chip.sizes()[0] == 1 << LIMB_BITS,
            "First element of RangeTupleChecker must have size {}",
            1 << LIMB_BITS
        );
        debug_assert!(
            range_tuple_chip.sizes()[1] >= (1 << LIMB_BITS) * 2 * NUM_LIMBS as u32,
            "Second element of RangeTupleChecker must have size of at least {}",
            (1 << LIMB_BITS) * 2 * NUM_LIMBS as u32
        );

        Self {
            air: MulHCoreAir {
                bitwise_lookup_bus: bitwise_lookup_chip.bus(),
                range_tuple_bus: *range_tuple_chip.bus(),
                offset,
            },
            bitwise_lookup_chip,
            range_tuple_chip,
        }
    }
}

#[derive(Clone, Debug)]
pub struct MulHCoreRecord<T, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub opcode: MulHOpcode,
    pub a: [T; NUM_LIMBS],
    pub b: [T; NUM_LIMBS],
    pub c: [T; NUM_LIMBS],
    pub a_mul: [T; NUM_LIMBS],
    pub b_ext: T,
    pub c_ext: T,
}

impl<F: PrimeField32, I: VmAdapterInterface<F>, const NUM_LIMBS: usize, const LIMB_BITS: usize>
    VmCoreChip<F, I> for MulHCoreChip<NUM_LIMBS, LIMB_BITS>
where
    I::Reads: Into<[[F; NUM_LIMBS]; 2]>,
    I::Writes: From<[[F; NUM_LIMBS]; 1]>,
{
    type Record = MulHCoreRecord<F, NUM_LIMBS, LIMB_BITS>;
    type Air = MulHCoreAir<NUM_LIMBS, LIMB_BITS>;

    #[allow(clippy::type_complexity)]
    fn execute_instruction(
        &self,
        instruction: &Instruction<F>,
        _from_pc: u32,
        reads: I::Reads,
    ) -> Result<(AdapterRuntimeContext<F, I>, Self::Record)> {
        let Instruction { opcode, .. } = instruction;
        let mulh_opcode = MulHOpcode::from_usize(opcode.local_opcode_idx(self.air.offset));

        let data: [[F; NUM_LIMBS]; 2] = reads.into();
        let b = data[0].map(|x| x.as_canonical_u32());
        let c = data[1].map(|y| y.as_canonical_u32());
        let (a, a_mul, carry, b_ext, c_ext) = run_mulh::<NUM_LIMBS, LIMB_BITS>(mulh_opcode, &b, &c);

        for i in 0..NUM_LIMBS {
            self.range_tuple_chip.add_count(&[a_mul[i], carry[i]]);
            self.range_tuple_chip
                .add_count(&[a[i], carry[NUM_LIMBS + i]]);
        }

        if mulh_opcode != MulHOpcode::MULHU {
            let b_sign_mask = if b_ext == 0 { 0 } else { 1 << (LIMB_BITS - 1) };
            let c_sign_mask = if c_ext == 0 { 0 } else { 1 << (LIMB_BITS - 1) };
            self.bitwise_lookup_chip.request_range(
                (b[NUM_LIMBS - 1] - b_sign_mask) << 1,
                (c[NUM_LIMBS - 1] - c_sign_mask) << ((mulh_opcode == MulHOpcode::MULH) as u32),
            );
        }

        let output = AdapterRuntimeContext::without_pc([a.map(F::from_canonical_u32)]);
        let record = MulHCoreRecord {
            opcode: mulh_opcode,
            a: a.map(F::from_canonical_u32),
            b: data[0],
            c: data[1],
            a_mul: a_mul.map(F::from_canonical_u32),
            b_ext: F::from_canonical_u32(b_ext),
            c_ext: F::from_canonical_u32(c_ext),
        };

        Ok((output, record))
    }

    fn get_opcode_name(&self, opcode: usize) -> String {
        format!("{:?}", MulHOpcode::from_usize(opcode - self.air.offset))
    }

    fn generate_trace_row(&self, row_slice: &mut [F], record: Self::Record) {
        let row_slice: &mut MulHCoreCols<_, NUM_LIMBS, LIMB_BITS> = row_slice.borrow_mut();
        row_slice.a = record.a;
        row_slice.b = record.b;
        row_slice.c = record.c;
        row_slice.a_mul = record.a_mul;
        row_slice.b_ext = record.b_ext;
        row_slice.c_ext = record.c_ext;
        row_slice.opcode_mulh_flag = F::from_bool(record.opcode == MulHOpcode::MULH);
        row_slice.opcode_mulhsu_flag = F::from_bool(record.opcode == MulHOpcode::MULHSU);
        row_slice.opcode_mulhu_flag = F::from_bool(record.opcode == MulHOpcode::MULHU);
    }

    fn air(&self) -> &Self::Air {
        &self.air
    }
}

// returns mulh[[s]u], mul, carry, x_ext, y_ext
pub(super) fn run_mulh<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    opcode: MulHOpcode,
    x: &[u32; NUM_LIMBS],
    y: &[u32; NUM_LIMBS],
) -> ([u32; NUM_LIMBS], [u32; NUM_LIMBS], Vec<u32>, u32, u32) {
    let mut mul = [0; NUM_LIMBS];
    let mut carry = vec![0; 2 * NUM_LIMBS];
    for i in 0..NUM_LIMBS {
        if i > 0 {
            mul[i] = carry[i - 1];
        }
        for j in 0..=i {
            mul[i] += x[j] * y[i - j];
        }
        carry[i] = mul[i] >> LIMB_BITS;
        mul[i] %= 1 << LIMB_BITS;
    }

    let x_ext = (x[NUM_LIMBS - 1] >> (LIMB_BITS - 1))
        * if opcode == MulHOpcode::MULHU {
            0
        } else {
            (1 << LIMB_BITS) - 1
        };
    let y_ext = (y[NUM_LIMBS - 1] >> (LIMB_BITS - 1))
        * if opcode == MulHOpcode::MULH {
            (1 << LIMB_BITS) - 1
        } else {
            0
        };

    let mut mulh = [0; NUM_LIMBS];
    let mut x_prefix = 0;
    let mut y_prefix = 0;

    for i in 0..NUM_LIMBS {
        x_prefix += x[i];
        y_prefix += y[i];
        mulh[i] = carry[NUM_LIMBS + i - 1] + x_prefix * y_ext + y_prefix * x_ext;
        for j in (i + 1)..NUM_LIMBS {
            mulh[i] += x[j] * y[NUM_LIMBS + i - j];
        }
        carry[NUM_LIMBS + i] = mulh[i] >> LIMB_BITS;
        mulh[i] %= 1 << LIMB_BITS;
    }

    (mulh, mul, carry, x_ext, y_ext)
}
