use std::{mem::size_of, sync::Arc};

use afs_derive::AlignedBorrow;
use afs_primitives::{
    var_range::{bus::VariableRangeCheckerBus, VariableRangeCheckerChip},
    xor::{bus::XorBus, lookup::XorLookupChip},
};
use afs_stark_backend::interaction::InteractionBuilder;
use p3_air::{Air, AirBuilderWithPublicValues, BaseAir, PairBuilder};
use p3_field::{Field, PrimeField32};

use crate::{
    arch::{
        instructions::{ShiftOpcode, UsizeOpcode},
        InstructionOutput, IntegrationInterface, MachineAdapter, MachineAdapterInterface,
        MachineIntegration, Result,
    },
    program::Instruction,
};

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct ShiftCols<T, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
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
    pub x_sign: T,

    // Boolean columns that are 1 exactly at the index of the bit/limb shift amount
    pub bit_shift_marker: [T; LIMB_BITS],
    pub limb_shift_marker: [T; NUM_LIMBS],

    // Part of each x[i] that gets bit shifted to the next limb
    pub bit_shift_carry: [T; NUM_LIMBS],
}

impl<T, const NUM_LIMBS: usize, const LIMB_BITS: usize> ShiftCols<T, NUM_LIMBS, LIMB_BITS> {
    pub fn width() -> usize {
        size_of::<ShiftCols<u8, NUM_LIMBS, LIMB_BITS>>()
    }
}

#[derive(Copy, Clone, Debug)]
pub struct ShiftAir<const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub xor_bus: XorBus,
    pub range_bus: VariableRangeCheckerBus,
}

impl<F: Field, const NUM_LIMBS: usize, const LIMB_BITS: usize> BaseAir<F>
    for ShiftAir<NUM_LIMBS, LIMB_BITS>
{
    fn width(&self) -> usize {
        ShiftCols::<F, NUM_LIMBS, LIMB_BITS>::width()
    }
}

impl<AB: InteractionBuilder, const NUM_LIMBS: usize, const LIMB_BITS: usize> Air<AB>
    for ShiftAir<NUM_LIMBS, LIMB_BITS>
{
    fn eval(&self, _builder: &mut AB) {
        todo!();
    }
}

#[derive(Debug)]
pub struct ShiftIntegration<const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub air: ShiftAir<NUM_LIMBS, LIMB_BITS>,
    pub xor_lookup_chip: Arc<XorLookupChip<LIMB_BITS>>,
    pub range_checker_chip: Arc<VariableRangeCheckerChip>,
    offset: usize,
}

impl<const NUM_LIMBS: usize, const LIMB_BITS: usize> ShiftIntegration<NUM_LIMBS, LIMB_BITS> {
    pub fn new(
        xor_lookup_chip: Arc<XorLookupChip<LIMB_BITS>>,
        range_checker_chip: Arc<VariableRangeCheckerChip>,
        offset: usize,
    ) -> Self {
        Self {
            air: ShiftAir {
                xor_bus: xor_lookup_chip.bus(),
                range_bus: range_checker_chip.bus(),
            },
            xor_lookup_chip,
            range_checker_chip,
            offset,
        }
    }
}

impl<F: PrimeField32, A: MachineAdapter<F>, const NUM_LIMBS: usize, const LIMB_BITS: usize>
    MachineIntegration<F, A> for ShiftIntegration<NUM_LIMBS, LIMB_BITS>
where
    A::Interface<F>: MachineAdapterInterface<F>,
    <A::Interface<F> as MachineAdapterInterface<F>>::Reads: Into<[[F; NUM_LIMBS]; 2]>,
    <A::Interface<F> as MachineAdapterInterface<F>>::Writes: From<[F; NUM_LIMBS]>,
{
    // TODO: update for trace generation
    type Record = u32;
    type Cols<T> = ShiftCols<T, NUM_LIMBS, LIMB_BITS>;
    type Air = ShiftAir<NUM_LIMBS, LIMB_BITS>;

    #[allow(clippy::type_complexity)]
    fn execute_instruction(
        &self,
        instruction: &Instruction<F>,
        _from_pc: F,
        reads: <A::Interface<F> as MachineAdapterInterface<F>>::Reads,
    ) -> Result<(InstructionOutput<F, A::Interface<F>>, Self::Record)> {
        let Instruction { opcode, .. } = instruction;
        let opcode = ShiftOpcode::from_usize(opcode - self.offset);

        let data: [[F; NUM_LIMBS]; 2] = reads.into();
        let x = data[0].map(|x| x.as_canonical_u32());
        let y = data[1].map(|y| y.as_canonical_u32());
        let (z, _limb_shift, _bit_shift) = solve_shift::<NUM_LIMBS, LIMB_BITS>(opcode, &x, &y);

        // Integration doesn't modify PC directly, so we let Adapter handle the increment
        let output: InstructionOutput<F, A::Interface<F>> = InstructionOutput {
            to_pc: None,
            writes: z.map(F::from_canonical_u32).into(),
        };

        // TODO: send XorLookupChip and VariableRangeCheckerChip requests
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

pub(super) fn solve_shift<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    opcode: ShiftOpcode,
    x: &[u32; NUM_LIMBS],
    y: &[u32; NUM_LIMBS],
) -> ([u32; NUM_LIMBS], usize, usize) {
    match opcode {
        ShiftOpcode::SLL => solve_shift_left::<NUM_LIMBS, LIMB_BITS>(x, y),
        ShiftOpcode::SRL => solve_shift_right::<NUM_LIMBS, LIMB_BITS>(x, y, true),
        ShiftOpcode::SRA => solve_shift_right::<NUM_LIMBS, LIMB_BITS>(x, y, false),
    }
}

fn solve_shift_left<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
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

fn solve_shift_right<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
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
    let shift = (y[0] + (y[1] * (1 << LIMB_BITS))) as usize;
    if shift < NUM_LIMBS * LIMB_BITS && y[2..].iter().all(|&val| val == 0) {
        (false, shift / LIMB_BITS, shift % LIMB_BITS)
    } else {
        (true, NUM_LIMBS, shift % LIMB_BITS)
    }
}
