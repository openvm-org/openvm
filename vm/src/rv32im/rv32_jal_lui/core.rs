use std::{array, marker::PhantomData, mem::size_of};

use afs_stark_backend::{interaction::InteractionBuilder, rap::BaseAirWithPublicValues};
use p3_air::BaseAir;
use p3_field::{Field, PrimeField32};

use crate::{
    arch::{
        instructions::{
            Rv32JalLuiOpcode::{self, *},
            UsizeOpcode,
        },
        AdapterAirContext, AdapterRuntimeContext, Result, VmAdapterInterface, VmCoreAir,
        VmCoreChip,
    },
    rv32im::adapters::{RV32_REGISTER_NUM_LANES, RV_J_TYPE_IMM_BITS},
    system::program::Instruction,
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
pub struct Rv32JalLuiCoreAir<F: Field> {
    pub _marker: PhantomData<F>,
    pub offset: usize,
}

impl<F: Field> BaseAir<F> for Rv32JalLuiCoreAir<F> {
    fn width(&self) -> usize {
        Rv32JalLuiCols::<F>::width()
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for Rv32JalLuiCoreAir<F> {}

impl<AB, I> VmCoreAir<AB, I> for Rv32JalLuiCoreAir<AB::F>
where
    AB: InteractionBuilder,
    I: VmAdapterInterface<AB::Expr>,
{
    fn eval(
        &self,
        _builder: &mut AB,
        _local_core: &[AB::Var],
        _local_adapter: &[AB::Var],
    ) -> AdapterAirContext<AB::Expr, I> {
        todo!()
    }
}

#[derive(Debug, Clone)]
pub struct Rv32JalLuiCoreChip<F: Field> {
    pub air: Rv32JalLuiCoreAir<F>,
}

impl<F: Field> Rv32JalLuiCoreChip<F> {
    pub fn new(offset: usize) -> Self {
        Self {
            air: Rv32JalLuiCoreAir::<F> {
                _marker: PhantomData,
                offset,
            },
        }
    }
}

impl<F: PrimeField32, I: VmAdapterInterface<F>> VmCoreChip<F, I> for Rv32JalLuiCoreChip<F>
where
    I::Writes: From<[F; RV32_REGISTER_NUM_LANES]>,
{
    type Record = ();
    type Air = Rv32JalLuiCoreAir<F>;

    #[allow(clippy::type_complexity)]
    fn execute_instruction(
        &self,
        instruction: &Instruction<F>,
        from_pc: u32,
        _reads: I::Reads,
    ) -> Result<(AdapterRuntimeContext<F, I>, Self::Record)> {
        let local_opcode_index = Rv32JalLuiOpcode::from_usize(instruction.opcode - self.air.offset);
        let c = instruction.op_c;

        let imm = match local_opcode_index {
            JAL => {
                // Note: immediate is a signed integer and c is a field element
                (c + F::from_canonical_u32(1 << (RV_J_TYPE_IMM_BITS - 1))).as_canonical_u32() as i32
                    - (1 << (RV_J_TYPE_IMM_BITS - 1))
            }
            LUI => c.as_canonical_u32() as i32,
        };
        let (to_pc, rd_data) = solve_jal_lui(local_opcode_index, from_pc, imm);
        let rd_data = rd_data.map(F::from_canonical_u32);

        let output = AdapterRuntimeContext {
            to_pc: Some(to_pc),
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

    fn generate_trace_row(&self, _row_slice: &mut [F], _record: Self::Record) {
        todo!()
    }

    fn air(&self) -> &Self::Air {
        &self.air
    }
}

// returns (to_pc, rd_data)
pub(super) fn solve_jal_lui(
    opcode: Rv32JalLuiOpcode,
    pc: u32,
    imm: i32,
) -> (u32, [u32; RV32_REGISTER_NUM_LANES]) {
    match opcode {
        JAL => {
            let rd_data = array::from_fn(|i| ((pc + 4) >> (8 * i)) & 255);
            let next_pc = pc as i32 + imm;
            assert!(next_pc >= 0);
            (next_pc as u32, rd_data)
        }
        LUI => {
            let imm = imm as u32;
            let rd = imm << 12;
            let rd_data = array::from_fn(|i| (rd >> (8 * i)) & 255);
            (pc + 4, rd_data)
        }
    }
}
