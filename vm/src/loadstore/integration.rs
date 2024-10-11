use std::{marker::PhantomData, mem::size_of};

use afs_stark_backend::interaction::InteractionBuilder;
use p3_air::{AirBuilderWithPublicValues, BaseAir, PairBuilder};
use p3_field::{Field, PrimeField32};

use crate::{
    arch::{
        instructions::{
            Rv32LoadStoreOpcode::{self, *},
            UsizeOpcode,
        },
        InstructionOutput, IntegrationInterface, MachineAdapter, MachineAdapterInterface,
        MachineIntegration, MachineIntegrationAir, Reads, Result, Writes,
    },
    program::Instruction,
};

#[derive(Debug, Clone)]
pub struct LoadStoreCols<T, const NUM_CELLS: usize> {
    pub _marker: PhantomData<T>,
}

impl<T, const NUM_CELLS: usize> LoadStoreCols<T, NUM_CELLS> {
    pub fn width() -> usize {
        size_of::<LoadStoreCols<T, NUM_CELLS>>()
    }
}

#[derive(Debug, Clone)]
pub struct LoadStoreAir<const NUM_CELLS: usize> {
    pub offset: usize,
}

impl<F: Field, const NUM_CELLS: usize> BaseAir<F> for LoadStoreAir<NUM_CELLS> {
    fn width(&self) -> usize {
        LoadStoreCols::<F, NUM_CELLS>::width()
    }
}

impl<
        F: PrimeField32,
        A: MachineAdapter<F>,
        AB: InteractionBuilder + PairBuilder + AirBuilderWithPublicValues,
        const NUM_CELLS: usize,
    > MachineIntegrationAir<F, A, LoadStoreIntegration<NUM_CELLS>, AB> for LoadStoreAir<NUM_CELLS>
where
    Reads<F, A::Interface<F>>: Into<[[F; NUM_CELLS]; 2]>,
    Writes<F, A::Interface<F>>: From<[F; NUM_CELLS]>,
{
    /// Returns `(to_pc, interface)`
    fn eval_primitive(
        &self,
        _builder: &mut AB,
        _local: &LoadStoreCols<AB::Var, NUM_CELLS>,
        _local_adapter: &A::Cols<AB::Var>,
    ) -> IntegrationInterface<AB::Expr, A::Interface<AB::Expr>> {
        todo!()
    }
}

#[derive(Debug, Clone)]
pub struct LoadStoreIntegration<const NUM_CELLS: usize> {
    pub air: LoadStoreAir<NUM_CELLS>,
}

impl<const NUM_CELLS: usize> LoadStoreIntegration<NUM_CELLS> {
    pub fn new(offset: usize) -> Self {
        Self {
            air: LoadStoreAir::<NUM_CELLS> { offset },
        }
    }
}

impl<F: PrimeField32, A: MachineAdapter<F>, const NUM_CELLS: usize> MachineIntegration<F, A>
    for LoadStoreIntegration<NUM_CELLS>
where
    Reads<F, A::Interface<F>>: Into<[[F; NUM_CELLS]; 2]>,
    Writes<F, A::Interface<F>>: From<[F; NUM_CELLS]>,
{
    type Record = ();
    type Air = LoadStoreAir<NUM_CELLS>;
    type Cols<T> = LoadStoreCols<T, NUM_CELLS>;

    #[allow(clippy::type_complexity)]
    fn execute_instruction(
        &self,
        instruction: &Instruction<F>,
        _from_pc: F,
        reads: <A::Interface<F> as MachineAdapterInterface<F>>::Reads,
    ) -> Result<(InstructionOutput<F, A::Interface<F>>, Self::Record)> {
        let opcode = Rv32LoadStoreOpcode::from_usize(instruction.opcode - self.air.offset);
        let data: [[F; NUM_CELLS]; 2] = reads.into();
        let write_data = solve_write_data(opcode, data[0], data[1]);

        let output: InstructionOutput<F, A::Interface<F>> = InstructionOutput {
            to_pc: None,
            writes: write_data.into(),
        };

        Ok((output, ()))
    }

    fn get_opcode_name(&self, opcode: usize) -> String {
        format!(
            "{:?}",
            Rv32LoadStoreOpcode::from_usize(opcode - self.air.offset)
        )
    }

    fn generate_trace_row(&self, _row_slice: &mut Self::Cols<F>, _record: Self::Record) {
        todo!()
    }

    fn air(&self) -> Self::Air {
        todo!()
    }
}

pub(super) fn solve_write_data<F: PrimeField32, const NUM_CELLS: usize>(
    opcode: Rv32LoadStoreOpcode,
    read_data: [F; NUM_CELLS],
    prev_data: [F; NUM_CELLS],
) -> [F; NUM_CELLS] {
    let mut write_data = read_data;
    match opcode {
        LOADW => (),
        LOADH => {
            let ext = read_data[NUM_CELLS / 2 - 1].as_canonical_u32();
            let ext = (ext >> 7) * 255;
            for cell in write_data.iter_mut().take(NUM_CELLS).skip(NUM_CELLS / 2) {
                *cell = F::from_canonical_u32(ext);
            }
        }
        LOADB => {
            let ext = read_data[0].as_canonical_u32();
            let ext = (ext >> 7) * 255;
            for cell in write_data.iter_mut().take(NUM_CELLS).skip(1) {
                *cell = F::from_canonical_u32(ext);
            }
        }
        LOADHU => {
            for cell in write_data.iter_mut().take(NUM_CELLS).skip(NUM_CELLS / 2) {
                *cell = F::zero();
            }
        }
        LOADBU => {
            for cell in write_data.iter_mut().take(NUM_CELLS).skip(1) {
                *cell = F::zero();
            }
        }
        STOREW => (),
        STOREH => {
            for (i, cell) in write_data
                .iter_mut()
                .enumerate()
                .take(NUM_CELLS)
                .skip(NUM_CELLS / 2)
            {
                *cell = prev_data[i];
            }
        }
        STOREB => {
            for (i, cell) in write_data.iter_mut().enumerate().take(NUM_CELLS).skip(1) {
                *cell = prev_data[i];
            }
        }
    };
    write_data
}
