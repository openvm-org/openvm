use std::{array, sync::Arc};

use afs_derive::AlignedBorrow;
use afs_primitives::xor::{bus::XorBus, lookup::XorLookupChip};
use afs_stark_backend::{interaction::InteractionBuilder, rap::BaseAirWithPublicValues};
use p3_air::BaseAir;
use p3_field::{Field, PrimeField32};

use crate::{
    arch::{
        instructions::{AluOpcode, UsizeOpcode},
        AdapterAirContext, AdapterRuntimeContext, MinimalInstruction, Reads, Result, VmAdapter,
        VmAdapterInterface, VmCore, VmCoreAir, Writes,
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
impl<F: Field, const NUM_LIMBS: usize, const LIMB_BITS: usize> BaseAirWithPublicValues<F>
    for ArithmeticLogicAir<NUM_LIMBS, LIMB_BITS>
{
}

impl<AB, I, const NUM_LIMBS: usize, const LIMB_BITS: usize> VmCoreAir<AB, I>
    for ArithmeticLogicAir<NUM_LIMBS, LIMB_BITS>
where
    AB: InteractionBuilder,
    I: VmAdapterInterface<AB::Expr>,
    I::Reads: From<[[AB::Expr; NUM_LIMBS]; 2]>,
    I::Writes: From<[[AB::Expr; NUM_LIMBS]; 1]>,
    I::ProcessedInstruction: From<MinimalInstruction<AB::Expr>>,
{
    fn eval(
        &self,
        _builder: &mut AB,
        _local: &[AB::Var],
        _local_adapter: &[AB::Var],
    ) -> AdapterAirContext<AB::Expr, I> {
        todo!()
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
pub struct ArithmeticLogicCore<const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub air: ArithmeticLogicAir<NUM_LIMBS, LIMB_BITS>,
    pub xor_lookup_chip: Arc<XorLookupChip<LIMB_BITS>>,
}

impl<const NUM_LIMBS: usize, const LIMB_BITS: usize> ArithmeticLogicCore<NUM_LIMBS, LIMB_BITS> {
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

impl<F, A, const NUM_LIMBS: usize, const LIMB_BITS: usize> VmCoreChip<F, A>
    for ArithmeticLogicCore<NUM_LIMBS, LIMB_BITS>
where
    F: PrimeField32,
    A: VmAdapterChip<F>,
    Reads<F, A::Interface<F>>: Into<[[F; NUM_LIMBS]; 2]>,
    Writes<F, A::Interface<F>>: From<[[F; NUM_LIMBS]; 1]>,
{
    type Record = ArithmeticLogicRecord<F, NUM_LIMBS, LIMB_BITS>;
    type Air = ArithmeticLogicAir<NUM_LIMBS, LIMB_BITS>;

    #[allow(clippy::type_complexity)]
    fn execute_instruction(
        &self,
        instruction: &Instruction<F>,
        _from_pc: F,
        reads: <A::Interface<F> as VmAdapterInterface<F>>::Reads,
    ) -> Result<(AdapterRuntimeContext<F, A::Interface<F>>, Self::Record)> {
        let Instruction { opcode, .. } = instruction;
        let opcode = AluOpcode::from_usize(opcode - self.air.offset);

        let data: [[F; NUM_LIMBS]; 2] = reads.into();
        let b = data[0].map(|x| x.as_canonical_u32());
        let c = data[1].map(|y| y.as_canonical_u32());
        let a = solve_alu::<NUM_LIMBS, LIMB_BITS>(opcode, &b, &c);

        // Core doesn't modify PC directly, so we let Adapter handle the increment
        let output: AdapterRuntimeContext<F, A::Interface<F>> = AdapterRuntimeContext {
            to_pc: None,
            writes: [a.map(F::from_canonical_u32)].into(),
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

    fn get_opcode_name(&self, _opcode: usize) -> String {
        todo!()
    }

    fn generate_trace_row(&self, _row_slice: &mut [F], _record: Self::Record) {
        todo!()
    }

    fn air(&self) -> &Self::Air {
        &self.air
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
