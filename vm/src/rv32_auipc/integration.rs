use std::{array, marker::PhantomData, mem::size_of};

use afs_stark_backend::interaction::InteractionBuilder;
use p3_air::{AirBuilderWithPublicValues, BaseAir, PairBuilder};
use p3_field::{Field, PrimeField32};

use crate::{
    arch::{
        instructions::{Rv32AuipcOpcode, UsizeOpcode},
        InstructionOutput, IntegrationInterface, MachineAdapter, MachineAdapterInterface,
        MachineIntegration, Result, Writes, RV32_REGISTER_NUM_LANES,
    },
    program::Instruction,
};

#[derive(Debug, Clone)]
pub struct Rv32AuipcCols<T> {
    pub _marker: PhantomData<T>,
}

impl<T> Rv32AuipcCols<T> {
    pub fn width() -> usize {
        size_of::<Rv32AuipcCols<T>>()
    }
}

#[derive(Debug, Clone)]
pub struct Rv32AuipcAir<F: Field> {
    pub _marker: PhantomData<F>,
    pub offset: usize,
}

impl<F: Field> BaseAir<F> for Rv32AuipcAir<F> {
    fn width(&self) -> usize {
        Rv32AuipcCols::<F>::width()
    }
}

#[derive(Debug, Clone)]
pub struct Rv32AuipcIntegration<F: Field> {
    pub air: Rv32AuipcAir<F>,
}

impl<F: Field> Rv32AuipcIntegration<F> {
    pub fn new(offset: usize) -> Self {
        Self {
            air: Rv32AuipcAir::<F> {
                _marker: PhantomData,
                offset,
            },
        }
    }
}

impl<F: PrimeField32, A: MachineAdapter<F>> MachineIntegration<F, A> for Rv32AuipcIntegration<F>
where
    Writes<F, A::Interface<F>>: From<[F; RV32_REGISTER_NUM_LANES]>,
{
    type Record = ();
    type Air = Rv32AuipcAir<F>;
    type Cols<T> = Rv32AuipcCols<T>;

    #[allow(clippy::type_complexity)]
    fn execute_instruction(
        &self,
        instruction: &Instruction<F>,
        from_pc: F,
        _reads: <A::Interface<F> as MachineAdapterInterface<F>>::Reads,
    ) -> Result<(InstructionOutput<F, A::Interface<F>>, Self::Record)> {
        let opcode = Rv32AuipcOpcode::from_usize(instruction.opcode - self.air.offset);
        let c = instruction.op_c.as_canonical_u32();
        let rd_data = solve_auipc(opcode, from_pc.as_canonical_u32(), c);
        let rd_data = rd_data.map(F::from_canonical_u32);

        let output: InstructionOutput<F, A::Interface<F>> = InstructionOutput {
            to_pc: None,
            writes: rd_data.into(),
        };

        // TODO: send XorLookUpChip requests
        // TODO: create Record and return

        Ok((output, ()))
    }

    fn get_opcode_name(&self, opcode: usize) -> String {
        format!(
            "{:?}",
            Rv32AuipcOpcode::from_usize(opcode - self.air.offset)
        )
    }

    fn generate_trace_row(&self, _row_slice: &mut Self::Cols<F>, _record: Self::Record) {
        todo!()
    }

    fn eval_primitive<AB: InteractionBuilder<F = F> + PairBuilder + AirBuilderWithPublicValues>(
        _air: &Self::Air,
        _builder: &mut AB,
        _local: &Self::Cols<AB::Var>,
        _local_adapter: &A::Cols<AB::Var>,
    ) -> IntegrationInterface<AB::Expr, A::Interface<AB::Expr>> {
        todo!()
    }

    fn air(&self) -> Self::Air {
        todo!()
    }
}

// returns rd_data
pub(super) fn solve_auipc(
    _opcode: Rv32AuipcOpcode,
    pc: u32,
    imm: u32,
) -> [u32; RV32_REGISTER_NUM_LANES] {
    let rd = pc.wrapping_add(imm << 8);
    array::from_fn(|i| (rd >> (8 * i)) & 255)
}
