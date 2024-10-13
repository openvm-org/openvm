use std::sync::Arc;

use afs_derive::AlignedBorrow;
use afs_primitives::range_tuple::{RangeTupleCheckerBus, RangeTupleCheckerChip};
use afs_stark_backend::{interaction::InteractionBuilder, rap::BaseAirWithPublicValues};
use p3_air::{Air, BaseAir};
use p3_field::{Field, PrimeField32};

use crate::{
    arch::{
        instructions::{MulOpcode, UsizeOpcode},
        CoreIO, InstructionOutput, Reads, Result, VmAdapter, VmAdapterInterface, VmCore,
        VmCoreAir, Writes,
    },
    program::Instruction,
};

// TODO: Replace current multiplication module upon completion

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct MultiplicationCols<T, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub a: [T; NUM_LIMBS],
    pub b: [T; NUM_LIMBS],
    pub c: [T; NUM_LIMBS],

    pub is_valid: T,
}

#[derive(Copy, Clone, Debug)]
pub struct MultiplicationAir<const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub bus: RangeTupleCheckerBus<2>,
}

impl<F: Field, const NUM_LIMBS: usize, const LIMB_BITS: usize> BaseAir<F>
    for MultiplicationAir<NUM_LIMBS, LIMB_BITS>
{
    fn width(&self) -> usize {
        MultiplicationCols::<F, NUM_LIMBS, LIMB_BITS>::width()
    }
}

impl<F: Field, const NUM_LIMBS: usize, const LIMB_BITS: usize> BaseAirWithPublicValues<F>
    for MultiplicationAir<NUM_LIMBS, LIMB_BITS>
{
}

impl<AB: InteractionBuilder, const NUM_LIMBS: usize, const LIMB_BITS: usize> Air<AB>
    for MultiplicationAir<NUM_LIMBS, LIMB_BITS>
{
    fn eval(&self, _builder: &mut AB) {
        todo!();
    }
}

impl<AB, I, const NUM_LIMBS: usize, const LIMB_BITS: usize> VmCoreAir<AB, I>
    for MultiplicationAir<NUM_LIMBS, LIMB_BITS>
where
    AB: InteractionBuilder,
    I: VmAdapterInterface<AB::Expr>,
{
    fn eval(
        &self,
        _builder: &mut AB,
        _local: &[AB::Var],
        _local_adapter: &[AB::Var],
    ) -> CoreIO<AB::Expr, I> {
        todo!()
    }
}

#[derive(Debug)]
pub struct MultiplicationCore<const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub air: MultiplicationAir<NUM_LIMBS, LIMB_BITS>,
    pub range_tuple_chip: Arc<RangeTupleCheckerChip<2>>,
    offset: usize,
}

impl<const NUM_LIMBS: usize, const LIMB_BITS: usize> MultiplicationCore<NUM_LIMBS, LIMB_BITS> {
    pub fn new(range_tuple_chip: Arc<RangeTupleCheckerChip<2>>, offset: usize) -> Self {
        Self {
            air: MultiplicationAir {
                bus: *range_tuple_chip.bus(),
            },
            range_tuple_chip,
            offset,
        }
    }
}

impl<F: PrimeField32, A: VmAdapter<F>, const NUM_LIMBS: usize, const LIMB_BITS: usize> VmCore<F, A>
    for MultiplicationCore<NUM_LIMBS, LIMB_BITS>
where
    Reads<F, A::Interface<F>>: Into<[[F; NUM_LIMBS]; 2]>,
    Writes<F, A::Interface<F>>: From<[F; NUM_LIMBS]>,
{
    // TODO: update for trace generation
    type Record = u32;
    type Air = MultiplicationAir<NUM_LIMBS, LIMB_BITS>;

    #[allow(clippy::type_complexity)]
    fn execute_instruction(
        &self,
        instruction: &Instruction<F>,
        _from_pc: F,
        reads: <A::Interface<F> as VmAdapterInterface<F>>::Reads,
    ) -> Result<(InstructionOutput<F, A::Interface<F>>, Self::Record)> {
        let Instruction { opcode, .. } = instruction;
        assert_eq!(MulOpcode::from_usize(opcode - self.offset), MulOpcode::MUL);

        let data: [[F; NUM_LIMBS]; 2] = reads.into();
        let x = data[0].map(|x| x.as_canonical_u32());
        let y = data[1].map(|y| y.as_canonical_u32());
        let z = solve_mul::<NUM_LIMBS, LIMB_BITS>(&x, &y);

        // Core doesn't modify PC directly, so we let Adapter handle the increment
        let output: InstructionOutput<F, A::Interface<F>> = InstructionOutput {
            to_pc: None,
            writes: z.map(F::from_canonical_u32).into(),
        };

        // TODO: send RangeTupleChecker requests
        // TODO: create Record and return

        Ok((output, 0))
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

pub(super) fn solve_mul<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    x: &[u32; NUM_LIMBS],
    y: &[u32; NUM_LIMBS],
) -> [u32; NUM_LIMBS] {
    let mut result = [0; NUM_LIMBS];
    let mut carry = [0; NUM_LIMBS];
    for i in 0..NUM_LIMBS {
        if i > 0 {
            result[i] = carry[i - 1];
        }
        for j in 0..=i {
            result[i] += x[j] * y[i - j];
        }
        carry[i] = result[i] >> LIMB_BITS;
        result[i] %= 1 << LIMB_BITS;
    }
    result
}
