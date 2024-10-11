use std::{array, marker::PhantomData, mem::size_of};

use afs_stark_backend::interaction::InteractionBuilder;
use p3_air::{AirBuilderWithPublicValues, BaseAir, PairBuilder};
use p3_field::{Field, PrimeField32};

use crate::{
    arch::{
        instructions::{
            Rv32JalLuiOpcode::{self, *},
            UsizeOpcode,
        },
        InstructionOutput, IntegrationInterface, MachineAdapter, MachineAdapterInterface,
        MachineIntegration, Result, Writes, RV32_REGISTER_NUM_LANES, RV_J_TYPE_IMM_BITS,
    },
    program::Instruction,
};

#[derive(Debug, Clone)]
pub struct Rv32JalLuiCols<T> {
    pub _marker: PhantomData<T>,
}

impl<T> Rv32JalLuiCols<T> {
    pub fn width() -> usize {
        size_of::<Rv32JalLuiCols<T>>()
    }
}

#[derive(Debug, Clone)]
pub struct Rv32JalLuiAir<F: Field> {
    pub _marker: PhantomData<F>,
    pub offset: usize,
}

impl<F: Field> BaseAir<F> for Rv32JalLuiAir<F> {
    fn width(&self) -> usize {
        Rv32JalLuiCols::<F>::width()
    }
}

#[derive(Debug, Clone)]
pub struct Rv32JalLuiIntegration<F: Field> {
    pub air: Rv32JalLuiAir<F>,
}

impl<F: Field> Rv32JalLuiIntegration<F> {
    pub fn new(offset: usize) -> Self {
        Self {
            air: Rv32JalLuiAir::<F> {
                _marker: PhantomData,
                offset,
            },
        }
    }
}

impl<F: PrimeField32, A: MachineAdapter<F>> MachineIntegration<F, A> for Rv32JalLuiIntegration<F>
where
    Writes<F, A::Interface<F>>: From<[F; RV32_REGISTER_NUM_LANES]>,
{
    type Record = ();
    type Air = Rv32JalLuiAir<F>;
    type Cols<T> = Rv32JalLuiCols<T>;

    #[allow(clippy::type_complexity)]
    fn execute_instruction(
        &self,
        instruction: &Instruction<F>,
        from_pc: F,
        _reads: <A::Interface<F> as MachineAdapterInterface<F>>::Reads,
    ) -> Result<(InstructionOutput<F, A::Interface<F>>, Self::Record)> {
        let opcode = Rv32JalLuiOpcode::from_usize(instruction.opcode - self.air.offset);
        let c = instruction.op_c;

        let imm = match opcode {
            JAL => {
                // Note: immediate is a signed integer and c is a field element
                (c + F::from_canonical_u32(1 << (RV_J_TYPE_IMM_BITS - 1))).as_canonical_u32() as i32
                    - (1 << (RV_J_TYPE_IMM_BITS - 1))
            }
            LUI => c.as_canonical_u32() as i32,
        };
        let (to_pc, rd_data) = solve_jal_lui(opcode, from_pc.as_canonical_u32() as usize, imm);
        let rd_data = rd_data.map(F::from_canonical_u32);

        let output: InstructionOutput<F, A::Interface<F>> = InstructionOutput {
            to_pc: Some(F::from_canonical_usize(to_pc)),
            writes: rd_data.into(),
        };

        Ok((output, ()))
    }

    fn get_opcode_name(&self, opcode: usize) -> String {
        format!(
            "{:?}",
            Rv32JalLuiOpcode::from_usize(opcode - self.air.offset)
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

// returns (to_pc, rd_data)
pub(super) fn solve_jal_lui(
    opcode: Rv32JalLuiOpcode,
    pc: usize,
    imm: i32,
) -> (usize, [u32; RV32_REGISTER_NUM_LANES]) {
    match opcode {
        JAL => {
            let rd_data = array::from_fn(|i| ((pc as u32 + 4) >> (8 * i)) & 255);
            let next_pc = pc as i32 + imm;
            assert!(next_pc >= 0);
            (next_pc as usize, rd_data)
        }
        LUI => {
            let imm = imm as u32;
            let rd = imm << 12;
            let rd_data = array::from_fn(|i| (rd >> (8 * i)) & 255);
            (pc + 4, rd_data)
        }
    }
}
