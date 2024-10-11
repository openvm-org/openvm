use std::{array, borrow::Borrow, mem::size_of, sync::Arc};

use afs_derive::AlignedBorrow;
use afs_primitives::xor::{bus::XorBus, lookup::XorLookupChip};
use afs_stark_backend::interaction::InteractionBuilder;
use p3_air::{AirBuilder, AirBuilderWithPublicValues, BaseAir, PairBuilder};
use p3_field::{AbstractField, Field, PrimeField32};
use strum::IntoEnumIterator;

use crate::{
    arch::{
        instructions::{AluOpcode, UsizeOpcode},
        InstructionOutput, IntegrationInterface, MachineAdapter, MachineAdapterInterface,
        MachineIntegration, MachineIntegrationAir, ProcessedInstruction, Reads, Result, Writes,
    },
    program::Instruction,
};

// TODO: Replace current ALU module upon completion

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct ArithmeticLogicCols<T, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub a: [T; NUM_LIMBS],
    pub b: [T; NUM_LIMBS],
    pub c: [T; NUM_LIMBS],

    pub opcode_add_flag: T,
    pub opcode_sub_flag: T,
    pub opcode_xor_flag: T,
    pub opcode_and_flag: T,
    pub opcode_or_flag: T,
}

impl<T, const NUM_LIMBS: usize, const LIMB_BITS: usize>
    ArithmeticLogicCols<T, NUM_LIMBS, LIMB_BITS>
{
    pub fn width() -> usize {
        size_of::<ArithmeticLogicCols<u8, NUM_LIMBS, LIMB_BITS>>()
    }
}

#[derive(Copy, Clone, Debug)]
pub struct ArithmeticLogicAir<const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub bus: XorBus,
    offset: usize,
}

impl<F: Field, const NUM_LIMBS: usize, const LIMB_BITS: usize> BaseAir<F>
    for ArithmeticLogicAir<NUM_LIMBS, LIMB_BITS>
{
    fn width(&self) -> usize {
        ArithmeticLogicCols::<F, NUM_LIMBS, LIMB_BITS>::width()
    }
}

impl<
        F: PrimeField32,
        A: MachineAdapter<F>,
        AB: InteractionBuilder + PairBuilder + AirBuilderWithPublicValues,
        const NUM_LIMBS: usize,
        const LIMB_BITS: usize,
    > MachineIntegrationAir<F, A, ArithmeticLogicIntegration<NUM_LIMBS, LIMB_BITS>, AB>
    for ArithmeticLogicAir<NUM_LIMBS, LIMB_BITS>
where
    Reads<F, A::Interface<F>>: Into<[[F; NUM_LIMBS]; 2]>,
    Reads<AB::Expr, A::Interface<AB::Expr>>: From<[[AB::Expr; NUM_LIMBS]; 2]>,
    Writes<F, A::Interface<F>>: From<[F; NUM_LIMBS]>,
    Writes<AB::Expr, A::Interface<AB::Expr>>: From<[AB::Expr; NUM_LIMBS]>,
    ProcessedInstruction<AB::Expr, A::Interface<AB::Expr>>: From<(AB::Expr, AB::Expr)>,
{
    /// Returns `(to_pc, interface)`
    fn eval_primitive(
        &self,
        builder: &mut AB,
        local: &ArithmeticLogicCols<AB::Var, NUM_LIMBS, LIMB_BITS>,
        _local_adapter: &A::Cols<AB::Var>,
    ) -> IntegrationInterface<AB::Expr, A::Interface<AB::Expr>> {
        let cols: &ArithmeticLogicCols<_, NUM_LIMBS, LIMB_BITS> = (*local).borrow();
        let flags = [
            cols.opcode_add_flag,
            cols.opcode_sub_flag,
            cols.opcode_xor_flag,
            cols.opcode_and_flag,
            cols.opcode_or_flag,
        ];

        for flag in flags {
            builder.assert_bool(flag);
        }

        let is_valid = flags
            .iter()
            .fold(AB::Expr::zero(), |acc, &flag| acc + flag.into());
        builder.assert_bool(is_valid.clone());

        let expected_opcode = flags
            .iter()
            .zip(AluOpcode::iter())
            .fold(AB::Expr::zero(), |acc, (flag, opcode)| {
                acc + (*flag).into() * AB::Expr::from_canonical_u8(opcode as u8)
            })
            - AB::Expr::from_canonical_usize(self.offset);

        let a = &cols.a;
        let b = &cols.b;
        let c = &cols.c;

        // For ADD, define carry[i] = (b[i] + c[i] + carry[i - 1] - a[i]) / 2^LIMB_BITS. If
        // each carry[i] is boolean and 0 <= a[i] < 2^LIMB_BITS, it can be proven that
        // a[i] = (b[i] + c[i]) % 2^LIMB_BITS as necessary. The same holds for SUB when
        // carry[i] is (a[i] + b[i] - c[i] + carry[i - 1]) / 2^LIMB_BITS.
        let mut carry_add: [AB::Expr; NUM_LIMBS] = array::from_fn(|_| AB::Expr::zero());
        let mut carry_sub: [AB::Expr; NUM_LIMBS] = array::from_fn(|_| AB::Expr::zero());
        let carry_divide = AB::F::from_canonical_usize(1 << LIMB_BITS).inverse();

        for i in 0..NUM_LIMBS {
            // We explicitly separate the constraints for ADD and SUB in order to keep degree
            // cubic. Because we constrain that the carry (which is arbitrary) is bool, if
            // carry has degree larger than 1 the max-degree constrain could be at least 4.
            carry_add[i] = AB::Expr::from(carry_divide)
                * (b[i] + c[i] - a[i]
                    + if i > 0 {
                        carry_add[i - 1].clone()
                    } else {
                        AB::Expr::zero()
                    });
            builder
                .when(cols.opcode_add_flag)
                .assert_bool(carry_add[i].clone());
            carry_sub[i] = AB::Expr::from(carry_divide)
                * (a[i] + c[i] - c[i]
                    + if i > 0 {
                        carry_sub[i - 1].clone()
                    } else {
                        AB::Expr::zero()
                    });
            builder
                .when(cols.opcode_sub_flag)
                .assert_bool(carry_sub[i].clone());
        }

        // Interaction with XorLookup to range check a for ADD and SUB, and constrain a's
        // correctness for XOR, AND, and OR. XorLookup expects interaction [x, y, x ^ y].
        let bitwise = cols.opcode_xor_flag + cols.opcode_and_flag + cols.opcode_or_flag;
        for i in 0..NUM_LIMBS {
            let x = (AB::Expr::one() - bitwise.clone()) * a[i] + bitwise.clone() * b[i];
            let y = (AB::Expr::one() - bitwise.clone()) * a[i] + bitwise.clone() * c[i];
            let x_xor_y = cols.opcode_xor_flag * a[i]
                + cols.opcode_and_flag * (b[i] + c[i] - (AB::Expr::from_canonical_u32(2) * a[i]))
                + cols.opcode_or_flag * ((AB::Expr::from_canonical_u32(2) * a[i]) - b[i] - c[i]);
            self.bus.send(x, y, x_xor_y).eval(builder, AB::Expr::one());
        }

        IntegrationInterface {
            to_pc: None,
            reads: [cols.a.map(Into::into), cols.b.map(Into::into)].into(),
            writes: cols.c.map(Into::into).into(),
            instruction: (is_valid, expected_opcode).into(),
        }
    }
}

#[derive(Debug)]
pub struct ArithmeticLogicRecord<T, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub opcode: AluOpcode,
    pub a: [T; NUM_LIMBS],
    pub b: [T; NUM_LIMBS],
    pub c: [T; NUM_LIMBS],
}

#[derive(Debug)]
pub struct ArithmeticLogicIntegration<const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub air: ArithmeticLogicAir<NUM_LIMBS, LIMB_BITS>,
    pub xor_lookup_chip: Arc<XorLookupChip<LIMB_BITS>>,
}

impl<const NUM_LIMBS: usize, const LIMB_BITS: usize>
    ArithmeticLogicIntegration<NUM_LIMBS, LIMB_BITS>
{
    pub fn new(xor_lookup_chip: Arc<XorLookupChip<LIMB_BITS>>, offset: usize) -> Self {
        Self {
            air: ArithmeticLogicAir {
                bus: xor_lookup_chip.bus(),
                offset,
            },
            xor_lookup_chip,
        }
    }
}

impl<F: PrimeField32, A: MachineAdapter<F>, const NUM_LIMBS: usize, const LIMB_BITS: usize>
    MachineIntegration<F, A> for ArithmeticLogicIntegration<NUM_LIMBS, LIMB_BITS>
where
    Reads<F, A::Interface<F>>: Into<[[F; NUM_LIMBS]; 2]>,
    Writes<F, A::Interface<F>>: From<[F; NUM_LIMBS]>,
{
    // TODO: update for trace generation
    type Record = ArithmeticLogicRecord<F, NUM_LIMBS, LIMB_BITS>;
    type Cols<T> = ArithmeticLogicCols<T, NUM_LIMBS, LIMB_BITS>;
    type Air = ArithmeticLogicAir<NUM_LIMBS, LIMB_BITS>;

    #[allow(clippy::type_complexity)]
    fn execute_instruction(
        &self,
        instruction: &Instruction<F>,
        _from_pc: F,
        reads: <A::Interface<F> as MachineAdapterInterface<F>>::Reads,
    ) -> Result<(InstructionOutput<F, A::Interface<F>>, Self::Record)> {
        let Instruction { opcode, .. } = instruction;
        let opcode = AluOpcode::from_usize(opcode - self.air.offset);

        let data: [[F; NUM_LIMBS]; 2] = reads.into();
        let b = data[0].map(|x| x.as_canonical_u32());
        let c = data[1].map(|y| y.as_canonical_u32());
        let a = solve_alu::<NUM_LIMBS, LIMB_BITS>(opcode, &b, &c);

        // Integration doesn't modify PC directly, so we let Adapter handle the increment
        let output: InstructionOutput<F, A::Interface<F>> = InstructionOutput {
            to_pc: None,
            writes: a.map(F::from_canonical_u32).into(),
        };

        if opcode == AluOpcode::ADD || opcode == AluOpcode::SUB {
            for a_val in a {
                self.xor_lookup_chip.request(a_val, a_val);
            }
        } else {
            for (b_val, c_val) in b.iter().zip(c.iter()) {
                self.xor_lookup_chip.request(*b_val, *c_val);
            }
        }

        let record = Self::Record {
            opcode,
            a: a.map(F::from_canonical_u32),
            b: data[0],
            c: data[1],
        };

        Ok((output, record))
    }

    fn get_opcode_name(&self, opcode: usize) -> String {
        format!("{:?}", AluOpcode::from_usize(opcode - self.air.offset))
    }

    fn generate_trace_row(&self, row_slice: &mut Self::Cols<F>, record: Self::Record) {
        row_slice.a = record.a;
        row_slice.b = record.b;
        row_slice.c = record.c;
        row_slice.opcode_add_flag = F::from_bool(record.opcode == AluOpcode::ADD);
        row_slice.opcode_sub_flag = F::from_bool(record.opcode == AluOpcode::SUB);
        row_slice.opcode_xor_flag = F::from_bool(record.opcode == AluOpcode::XOR);
        row_slice.opcode_and_flag = F::from_bool(record.opcode == AluOpcode::AND);
        row_slice.opcode_or_flag = F::from_bool(record.opcode == AluOpcode::OR);
    }

    fn air(&self) -> Self::Air {
        self.air
    }
}

pub(super) fn solve_alu<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    opcode: AluOpcode,
    x: &[u32; NUM_LIMBS],
    y: &[u32; NUM_LIMBS],
) -> [u32; NUM_LIMBS] {
    match opcode {
        AluOpcode::ADD => solve_add::<NUM_LIMBS, LIMB_BITS>(x, y),
        AluOpcode::SUB => solve_subtract::<NUM_LIMBS, LIMB_BITS>(x, y),
        AluOpcode::XOR => solve_xor::<NUM_LIMBS, LIMB_BITS>(x, y),
        AluOpcode::OR => solve_or::<NUM_LIMBS, LIMB_BITS>(x, y),
        AluOpcode::AND => solve_and::<NUM_LIMBS, LIMB_BITS>(x, y),
    }
}

fn solve_add<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    x: &[u32; NUM_LIMBS],
    y: &[u32; NUM_LIMBS],
) -> [u32; NUM_LIMBS] {
    let mut z = [0u32; NUM_LIMBS];
    let mut carry = [0u32; NUM_LIMBS];
    for i in 0..NUM_LIMBS {
        z[i] = x[i] + y[i] + if i > 0 { carry[i - 1] } else { 0 };
        carry[i] = z[i] >> LIMB_BITS;
        z[i] &= (1 << LIMB_BITS) - 1;
    }
    z
}

fn solve_subtract<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    x: &[u32; NUM_LIMBS],
    y: &[u32; NUM_LIMBS],
) -> [u32; NUM_LIMBS] {
    let mut z = [0u32; NUM_LIMBS];
    let mut carry = [0u32; NUM_LIMBS];
    for i in 0..NUM_LIMBS {
        let rhs = y[i] + if i > 0 { carry[i - 1] } else { 0 };
        if x[i] >= rhs {
            z[i] = x[i] - rhs;
            carry[i] = 0;
        } else {
            z[i] = x[i] + (1 << LIMB_BITS) - rhs;
            carry[i] = 1;
        }
    }
    z
}

fn solve_xor<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    x: &[u32; NUM_LIMBS],
    y: &[u32; NUM_LIMBS],
) -> [u32; NUM_LIMBS] {
    array::from_fn(|i| x[i] ^ y[i])
}

fn solve_or<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    x: &[u32; NUM_LIMBS],
    y: &[u32; NUM_LIMBS],
) -> [u32; NUM_LIMBS] {
    array::from_fn(|i| x[i] | y[i])
}

fn solve_and<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    x: &[u32; NUM_LIMBS],
    y: &[u32; NUM_LIMBS],
) -> [u32; NUM_LIMBS] {
    array::from_fn(|i| x[i] & y[i])
}
