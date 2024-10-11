use std::sync::Arc;

use afs_derive::AlignedBorrow;
use afs_primitives::xor::{bus::XorBus, lookup::XorLookupChip};
use afs_stark_backend::interaction::InteractionBuilder;
use p3_air::{Air, AirBuilderWithPublicValues, BaseAir, PairBuilder};
use p3_field::{Field, PrimeField32};

use crate::{
    arch::{
        instructions::{BranchLessThanOpcode, UsizeOpcode},
        InstructionOutput, IntegrationInterface, MachineAdapter, MachineAdapterInterface,
        MachineIntegration, Reads, Result, Writes,
    },
    program::Instruction,
};

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct BranchLessThanCols<T, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub a: [T; NUM_LIMBS],
    pub b: [T; NUM_LIMBS],
    pub cmp_result: T,
    pub next_pc: T,

    pub opcode_blt_flag: T,
    pub opcode_bltu_flag: T,
    pub opcode_bge_flag: T,
    pub opcode_bgeu_flag: T,

    pub x_sign: T,
    pub y_sign: T,

    // 1 at the most significant index i such that b[i] != c[i], otherwise 0. If such
    // an i exists, diff_val = c[i] - b[i]
    pub diff_marker: [T; NUM_LIMBS],
    pub diff_val: T,
}

#[derive(Copy, Clone, Debug)]
pub struct BranchLessThanAir<const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub bus: XorBus,
}

impl<F: Field, const NUM_LIMBS: usize, const LIMB_BITS: usize> BaseAir<F>
    for BranchLessThanAir<NUM_LIMBS, LIMB_BITS>
{
    fn width(&self) -> usize {
        BranchLessThanCols::<F, NUM_LIMBS, LIMB_BITS>::width()
    }
}

impl<AB: InteractionBuilder, const NUM_LIMBS: usize, const LIMB_BITS: usize> Air<AB>
    for BranchLessThanAir<NUM_LIMBS, LIMB_BITS>
{
    fn eval(&self, _builder: &mut AB) {
        todo!();
    }
}

#[derive(Debug)]
pub struct BranchLessThanIntegration<const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub air: BranchLessThanAir<NUM_LIMBS, LIMB_BITS>,
    pub xor_lookup_chip: Arc<XorLookupChip<LIMB_BITS>>,
    offset: usize,
}

impl<const NUM_LIMBS: usize, const LIMB_BITS: usize>
    BranchLessThanIntegration<NUM_LIMBS, LIMB_BITS>
{
    pub fn new(xor_lookup_chip: Arc<XorLookupChip<LIMB_BITS>>, offset: usize) -> Self {
        Self {
            air: BranchLessThanAir {
                bus: xor_lookup_chip.bus(),
            },
            xor_lookup_chip,
            offset,
        }
    }
}

impl<F: PrimeField32, A: MachineAdapter<F>, const NUM_LIMBS: usize, const LIMB_BITS: usize>
    MachineIntegration<F, A> for BranchLessThanIntegration<NUM_LIMBS, LIMB_BITS>
where
    Reads<F, A::Interface<F>>: Into<[[F; NUM_LIMBS]; 2]>,
    Writes<F, A::Interface<F>>: Default,
{
    // TODO: update for trace generation
    type Record = u32;
    type Cols<T> = BranchLessThanCols<T, NUM_LIMBS, LIMB_BITS>;
    type Air = BranchLessThanAir<NUM_LIMBS, LIMB_BITS>;

    #[allow(clippy::type_complexity)]
    fn execute_instruction(
        &self,
        instruction: &Instruction<F>,
        from_pc: F,
        reads: <A::Interface<F> as MachineAdapterInterface<F>>::Reads,
    ) -> Result<(InstructionOutput<F, A::Interface<F>>, Self::Record)> {
        let Instruction {
            opcode, op_c: imm, ..
        } = *instruction;
        let opcode = BranchLessThanOpcode::from_usize(opcode - self.offset);

        let data: [[F; NUM_LIMBS]; 2] = reads.into();
        let x = data[0].map(|x| x.as_canonical_u32());
        let y = data[1].map(|y| y.as_canonical_u32());
        let (cmp_result, _diff_idx, _x_sign, _y_sign) =
            solve_cmp::<NUM_LIMBS, LIMB_BITS>(opcode, &x, &y);

        let output: InstructionOutput<F, A::Interface<F>> = InstructionOutput {
            to_pc: if cmp_result {
                Some(from_pc + imm)
            } else {
                None
            },
            writes: Default::default(),
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

// Returns (cmp_result, diff_idx, x_sign, y_sign)
pub(super) fn solve_cmp<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    opcode: BranchLessThanOpcode,
    x: &[u32; NUM_LIMBS],
    y: &[u32; NUM_LIMBS],
) -> (bool, usize, bool, bool) {
    let signed = opcode == BranchLessThanOpcode::BLT || opcode == BranchLessThanOpcode::BGE;
    let ge_op = opcode == BranchLessThanOpcode::BGE || opcode == BranchLessThanOpcode::BGEU;
    let x_sign = (x[NUM_LIMBS - 1] >> (LIMB_BITS - 1) == 1) && signed;
    let y_sign = (y[NUM_LIMBS - 1] >> (LIMB_BITS - 1) == 1) && signed;
    for i in (0..NUM_LIMBS).rev() {
        if x[i] != y[i] {
            return ((x[i] < y[i]) ^ x_sign ^ y_sign ^ ge_op, i, x_sign, y_sign);
        }
    }
    (ge_op, 0, x_sign, y_sign)
}
