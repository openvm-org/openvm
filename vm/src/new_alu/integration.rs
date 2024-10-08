use std::{array, mem::size_of, sync::Arc};

use afs_derive::AlignedBorrow;
use afs_primitives::xor::{bus::XorBus, lookup::XorLookupChip};
use afs_stark_backend::interaction::InteractionBuilder;
use p3_air::{Air, AirBuilderWithPublicValues, BaseAir, PairBuilder};
use p3_field::{Field, PrimeField32};

use crate::{
    arch::{
        instructions::{AluOpcode, UsizeOpcode},
        InstructionOutput, IntegrationInterface, MachineAdapter, MachineAdapterInterface,
        MachineIntegration, Result,
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
}

impl<F: Field, const NUM_LIMBS: usize, const LIMB_BITS: usize> BaseAir<F>
    for ArithmeticLogicAir<NUM_LIMBS, LIMB_BITS>
{
    fn width(&self) -> usize {
        ArithmeticLogicCols::<F, NUM_LIMBS, LIMB_BITS>::width()
    }
}

impl<AB: InteractionBuilder, const NUM_LIMBS: usize, const LIMB_BITS: usize> Air<AB>
    for ArithmeticLogicAir<NUM_LIMBS, LIMB_BITS>
{
    fn eval(&self, _builder: &mut AB) {
        todo!();
    }
}

#[derive(Debug)]
pub struct ArithmeticLogicIntegration<const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub air: ArithmeticLogicAir<NUM_LIMBS, LIMB_BITS>,
    pub xor_lookup_chip: Arc<XorLookupChip<LIMB_BITS>>,
    offset: usize,
}

impl<const NUM_LIMBS: usize, const LIMB_BITS: usize>
    ArithmeticLogicIntegration<NUM_LIMBS, LIMB_BITS>
{
    pub fn new(xor_lookup_chip: Arc<XorLookupChip<LIMB_BITS>>, offset: usize) -> Self {
        Self {
            air: ArithmeticLogicAir {
                bus: xor_lookup_chip.bus(),
            },
            xor_lookup_chip,
            offset,
        }
    }
}

impl<F: PrimeField32, A: MachineAdapter<F>, const NUM_LIMBS: usize, const LIMB_BITS: usize>
    MachineIntegration<F, A> for ArithmeticLogicIntegration<NUM_LIMBS, LIMB_BITS>
where
    A::Interface<F>: MachineAdapterInterface<F>,
    <A::Interface<F> as MachineAdapterInterface<F>>::Reads: Into<[[F; NUM_LIMBS]; 2]>,
    <A::Interface<F> as MachineAdapterInterface<F>>::Writes: From<[F; NUM_LIMBS]>,
{
    // TODO: update for trace generation
    type Record = u32;
    type Cols<T> = ArithmeticLogicCols<T, NUM_LIMBS, LIMB_BITS>;
    type Air = ArithmeticLogicAir<NUM_LIMBS, LIMB_BITS>;

    #[allow(clippy::type_complexity)]
    fn execute_instruction(
        &self,
        instruction: &Instruction<F>,
        from_pc: F,
        reads: <A::Interface<F> as MachineAdapterInterface<F>>::Reads,
    ) -> Result<(InstructionOutput<F, A::Interface<F>>, Self::Record)> {
        let Instruction { opcode, .. } = instruction;
        let opcode = AluOpcode::from_usize(opcode - self.offset);

        let data: [[F; NUM_LIMBS]; 2] = reads.into();
        let x = data[0].map(|x| x.as_canonical_u32());
        let y = data[1].map(|y| y.as_canonical_u32());
        let z = solve_alu::<NUM_LIMBS, LIMB_BITS>(opcode, &x, &y);

        // Integration doesn't modify PC directly, so we let Adapter handle the increment
        let output: InstructionOutput<F, A::Interface<F>> = InstructionOutput {
            to_pc: from_pc,
            writes: z.map(F::from_canonical_u32).into(),
        };

        // TODO: send XorLookupChip requests
        // TODO: create Record and return

        Ok((output, 0))
    }

    fn get_opcode_name(&self, _opcode: usize) -> String {
        todo!()
    }

    fn generate_trace_row(&self, _row_slice: &mut Self::Cols<F>, _record: Self::Record) {
        todo!()
    }

    /// Returns `(to_pc, interface)`.
    fn eval_primitive<AB: InteractionBuilder<F = F> + PairBuilder + AirBuilderWithPublicValues>(
        _air: &Self::Air,
        _builder: &mut AB,
        _local: &Self::Cols<AB::Var>,
        _local_adapter: &A::Cols<AB::Var>,
    ) -> IntegrationInterface<AB::Expr, A::Interface<AB::Expr>> {
        todo!()
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
