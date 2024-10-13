use std::sync::Arc;

use afs_derive::AlignedBorrow;
use afs_primitives::range_tuple::{RangeTupleCheckerBus, RangeTupleCheckerChip};
use afs_stark_backend::{interaction::InteractionBuilder, rap::BaseAirWithPublicValues};
use p3_air::{Air, AirBuilderWithPublicValues, BaseAir, PairBuilder};
use p3_field::{Field, PrimeField32};

use crate::{
    arch::{
        instructions::{MulHOpcode, UsizeOpcode},
        InstructionOutput, IntegrationInterface, MachineAdapter, MachineAdapterInterface,
        MachineIntegration, Reads, Result, Writes,
    },
    program::Instruction,
};

// TODO: Replace current ALU module upon completion

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct MulHCols<T, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
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
pub struct MulHAir<const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub bus: RangeTupleCheckerBus<2>,
}

impl<F: Field, const NUM_LIMBS: usize, const LIMB_BITS: usize> BaseAir<F>
    for MulHAir<NUM_LIMBS, LIMB_BITS>
{
    fn width(&self) -> usize {
        MulHCols::<F, NUM_LIMBS, LIMB_BITS>::width()
    }
}

impl<AB: InteractionBuilder, const NUM_LIMBS: usize, const LIMB_BITS: usize> Air<AB>
    for MulHAir<NUM_LIMBS, LIMB_BITS>
{
    fn eval(&self, _builder: &mut AB) {
        todo!();
    }
}

impl<F: Field, const NUM_LIMBS: usize, const LIMB_BITS: usize> BaseAirWithPublicValues<F>
    for MulHAir<NUM_LIMBS, LIMB_BITS>
{
}

#[derive(Debug)]
pub struct MulHIntegration<const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub air: MulHAir<NUM_LIMBS, LIMB_BITS>,
    pub range_tuple_chip: Arc<RangeTupleCheckerChip<2>>,
    offset: usize,
}

impl<const NUM_LIMBS: usize, const LIMB_BITS: usize> MulHIntegration<NUM_LIMBS, LIMB_BITS> {
    pub fn new(range_tuple_chip: Arc<RangeTupleCheckerChip<2>>, offset: usize) -> Self {
        Self {
            air: MulHAir {
                bus: *range_tuple_chip.bus(),
            },
            range_tuple_chip,
            offset,
        }
    }
}

impl<F: PrimeField32, A: MachineAdapter<F>, const NUM_LIMBS: usize, const LIMB_BITS: usize>
    MachineIntegration<F, A> for MulHIntegration<NUM_LIMBS, LIMB_BITS>
where
    Reads<F, A::Interface<F>>: Into<[[F; NUM_LIMBS]; 2]>,
    Writes<F, A::Interface<F>>: From<[F; NUM_LIMBS]>,
{
    // TODO: update for trace generation
    type Record = u32;
    type Air = MulHAir<NUM_LIMBS, LIMB_BITS>;

    #[allow(clippy::type_complexity)]
    fn execute_instruction(
        &self,
        instruction: &Instruction<F>,
        _from_pc: F,
        reads: <A::Interface<F> as MachineAdapterInterface<F>>::Reads,
    ) -> Result<(InstructionOutput<F, A::Interface<F>>, Self::Record)> {
        let Instruction { opcode, .. } = instruction;
        let opcode = MulHOpcode::from_usize(opcode - self.offset);

        let data: [[F; NUM_LIMBS]; 2] = reads.into();
        let x = data[0].map(|x| x.as_canonical_u32());
        let y = data[1].map(|y| y.as_canonical_u32());
        let (z, _z_mul, _x_ext, _y_ext) = solve_mulh::<NUM_LIMBS, LIMB_BITS>(opcode, &x, &y);

        // Integration doesn't modify PC directly, so we let Adapter handle the increment
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

// returns mulh[[s]u], mul, x_ext, y_ext
pub(super) fn solve_mulh<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    opcode: MulHOpcode,
    x: &[u32; NUM_LIMBS],
    y: &[u32; NUM_LIMBS],
) -> ([u32; NUM_LIMBS], [u32; NUM_LIMBS], u32, u32) {
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

    (mulh, mul, x_ext, y_ext)
}
