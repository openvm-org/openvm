use std::mem::size_of;

use afs_derive::AlignedBorrow;
use afs_stark_backend::interaction::InteractionBuilder;
use p3_air::{AirBuilderWithPublicValues, BaseAir, PairBuilder};
use p3_field::{Field, PrimeField32};

use crate::{
    arch::{
        instructions::{BranchEqualOpcode, UsizeOpcode},
        InstructionOutput, IntegrationInterface, MachineAdapter, MachineAdapterInterface,
        MachineIntegration, MachineIntegrationAir, Reads, Result, Writes,
    },
    program::Instruction,
};

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct BranchEqualCols<T, const NUM_LIMBS: usize> {
    pub a: [T; NUM_LIMBS],
    pub b: [T; NUM_LIMBS],
    pub cmp_result: T,
    pub next_pc: T,

    pub opcode_beq_flag: T,
    pub opcode_bne_flag: T,

    pub diff_marker: [T; NUM_LIMBS],
}

impl<T, const NUM_LIMBS: usize> BranchEqualCols<T, NUM_LIMBS> {
    pub fn width() -> usize {
        size_of::<BranchEqualCols<u8, NUM_LIMBS>>()
    }
}

#[derive(Copy, Clone, Debug)]
pub struct BranchEqualAir<const NUM_LIMBS: usize> {}

impl<F: Field, const NUM_LIMBS: usize> BaseAir<F> for BranchEqualAir<NUM_LIMBS> {
    fn width(&self) -> usize {
        BranchEqualCols::<F, NUM_LIMBS>::width()
    }
}

impl<
        F: PrimeField32,
        A: MachineAdapter<F>,
        AB: InteractionBuilder + PairBuilder + AirBuilderWithPublicValues,
        const NUM_LIMBS: usize,
    > MachineIntegrationAir<F, A, BranchEqualIntegration<NUM_LIMBS>, AB>
    for BranchEqualAir<NUM_LIMBS>
where
    Reads<F, A::Interface<F>>: Into<[[F; NUM_LIMBS]; 2]>,
    Writes<F, A::Interface<F>>: Default,
{
    /// Returns `(to_pc, interface)`
    fn eval_primitive(
        &self,
        _builder: &mut AB,
        _local: &BranchEqualCols<AB::Var, NUM_LIMBS>,
        _local_adapter: &A::Cols<AB::Var>,
    ) -> IntegrationInterface<AB::Expr, A::Interface<AB::Expr>> {
        todo!()
    }
}

#[derive(Debug)]
pub struct BranchEqualIntegration<const NUM_LIMBS: usize> {
    pub air: BranchEqualAir<NUM_LIMBS>,
    offset: usize,
}

impl<const NUM_LIMBS: usize> BranchEqualIntegration<NUM_LIMBS> {
    pub fn new(offset: usize) -> Self {
        Self {
            air: BranchEqualAir {},
            offset,
        }
    }
}

impl<F: PrimeField32, A: MachineAdapter<F>, const NUM_LIMBS: usize> MachineIntegration<F, A>
    for BranchEqualIntegration<NUM_LIMBS>
where
    Reads<F, A::Interface<F>>: Into<[[F; NUM_LIMBS]; 2]>,
    Writes<F, A::Interface<F>>: Default,
{
    // TODO: update for trace generation
    type Record = u32;
    type Cols<T> = BranchEqualCols<T, NUM_LIMBS>;
    type Air = BranchEqualAir<NUM_LIMBS>;

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
        let opcode = BranchEqualOpcode::from_usize(opcode - self.offset);

        let data: [[F; NUM_LIMBS]; 2] = reads.into();
        let x = data[0].map(|x| x.as_canonical_u32());
        let y = data[1].map(|y| y.as_canonical_u32());
        let (cmp_result, _diff_idx, _diff_val) = solve_eq::<F, NUM_LIMBS>(opcode, &x, &y);

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

    fn air(&self) -> Self::Air {
        self.air
    }
}

// Returns (cmp_result, diff_idx, x[diff_idx] - y[diff_idx])
pub(super) fn solve_eq<F: PrimeField32, const NUM_LIMBS: usize>(
    opcode: BranchEqualOpcode,
    x: &[u32; NUM_LIMBS],
    y: &[u32; NUM_LIMBS],
) -> (bool, usize, F) {
    for i in 0..NUM_LIMBS {
        if x[i] != y[i] {
            return (
                opcode == BranchEqualOpcode::BNE,
                i,
                (F::from_canonical_u32(x[i]) - F::from_canonical_u32(y[i])).inverse(),
            );
        }
    }
    (opcode == BranchEqualOpcode::BEQ, 0, F::zero())
}
