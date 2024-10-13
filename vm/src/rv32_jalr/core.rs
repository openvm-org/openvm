use std::{array, marker::PhantomData, mem::size_of};

use afs_stark_backend::{interaction::InteractionBuilder, rap::BaseAirWithPublicValues};
use p3_air::BaseAir;
use p3_field::{Field, PrimeField32};

use crate::{
    arch::{
        compose,
        instructions::{Rv32JalrOpcode, UsizeOpcode},
        AdapterAirContext, AdapterRuntimeContext, Reads, Result, VmAdapterChip, VmAdapterInterface,
        VmCoreAir, VmCoreChip, Writes, PC_BITS, RV32_REGISTER_NUM_LANES, RV_IS_TYPE_IMM_BITS,
    },
    program::Instruction,
};

#[derive(Debug, Clone)]
pub struct Rv32JalrCols<T> {
    pub _marker: PhantomData<T>,
}

impl<T> Rv32JalrCols<T> {
    pub fn width() -> usize {
        size_of::<Rv32JalrCols<T>>()
    }
}

#[derive(Debug, Clone)]
pub struct Rv32JalrCoreAir<F: Field> {
    pub _marker: PhantomData<F>,
    pub offset: usize,
}

impl<F: Field> BaseAir<F> for Rv32JalrCoreAir<F> {
    fn width(&self) -> usize {
        Rv32JalrCols::<F>::width()
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for Rv32JalrCoreAir<F> {}

impl<AB: InteractionBuilder, I> VmCoreAir<AB, I> for Rv32JalrCoreAir<AB::F>
where
    I: VmAdapterInterface<AB::Expr>,
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

#[derive(Debug, Clone)]
pub struct Rv32JalrCoreChip<F: Field> {
    pub air: Rv32JalrCoreAir<F>,
}

impl<F: Field> Rv32JalrCoreChip<F> {
    pub fn new(offset: usize) -> Self {
        Self {
            air: Rv32JalrCoreAir::<F> {
                _marker: PhantomData,
                offset,
            },
        }
    }
}

impl<F: PrimeField32, A: VmAdapterChip<F>> VmCoreChip<F, A> for Rv32JalrCoreChip<F>
where
    Reads<F, A::Interface<F>>: Into<[F; RV32_REGISTER_NUM_LANES]>,
    Writes<F, A::Interface<F>>: From<[F; RV32_REGISTER_NUM_LANES]>,
{
    type Record = ();
    type Air = Rv32JalrCoreAir<F>;

    #[allow(clippy::type_complexity)]
    fn execute_instruction(
        &self,
        instruction: &Instruction<F>,
        from_pc: F,
        reads: <A::Interface<F> as VmAdapterInterface<F>>::Reads,
    ) -> Result<(AdapterRuntimeContext<F, A::Interface<F>>, Self::Record)> {
        let Instruction {
            opcode, op_c: c, ..
        } = *instruction;
        let local_opcode_index = Rv32JalrOpcode::from_usize(opcode - self.air.offset);

        // Note: immediate is a signed integer and c is a field element
        let imm = (c + F::from_canonical_u32(1 << (RV_IS_TYPE_IMM_BITS - 1))).as_canonical_u32()
            as i32
            - (1 << (RV_IS_TYPE_IMM_BITS - 1));

        let rs1 = compose(reads.into());
        let (to_pc, rd_data) = solve_jalr(local_opcode_index, from_pc.as_canonical_u32(), imm, rs1);
        let rd_data = rd_data.map(F::from_canonical_u32);

        let output: AdapterRuntimeContext<F, A::Interface<F>> = AdapterRuntimeContext {
            to_pc: Some(F::from_canonical_u32(to_pc)),
            writes: rd_data.into(),
        };

        Ok((output, ()))
    }

    fn get_opcode_name(&self, opcode: usize) -> String {
        format!("{:?}", Rv32JalrOpcode::from_usize(opcode - self.air.offset))
    }

    fn generate_trace_row(&self, _row_slice: &mut [F], _record: Self::Record) {
        todo!()
    }

    fn air(&self) -> &Self::Air {
        &self.air
    }
}

// returns (to_pc, rd_data)
pub(super) fn solve_jalr(
    _opcode: Rv32JalrOpcode,
    pc: u32,
    imm: i32,
    rs1: u32,
) -> (u32, [u32; RV32_REGISTER_NUM_LANES]) {
    let next_pc: i32 = rs1 as i32 + imm;
    assert!(next_pc >= 0);
    let next_pc = ((next_pc as u32) >> 1) << 1;
    assert!(next_pc < (1 << PC_BITS));
    (
        next_pc,
        array::from_fn(|i: usize| ((pc + 4) >> (8 * i)) & 255),
    )
}
