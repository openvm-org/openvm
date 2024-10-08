use std::mem::size_of;

use afs_stark_backend::interaction::InteractionBuilder;
use p3_air::{AirBuilderWithPublicValues, BaseAir, PairBuilder};
use p3_field::{Field, PrimeField32};

use crate::{
    arch::{
        instructions::{LoadStoreOpcode, UsizeOpcode},
        InstructionOutput, IntegrationInterface, MachineAdapter, MachineAdapterInterface,
        MachineIntegration, Result,
    },
    program::Instruction,
};

#[derive(Debug, Clone)]
pub struct LoadStoreCols<T, const NUM_CELLS: usize> {
    pub _marker: std::marker::PhantomData<T>,
}

impl<T, const NUM_CELLS: usize> LoadStoreCols<T, NUM_CELLS> {
    pub fn width() -> usize {
        size_of::<LoadStoreCols<T, NUM_CELLS>>()
    }
}

#[derive(Debug, Clone)]
pub struct LoadStoreAir<F: Field, const NUM_CELLS: usize> {
    pub _marker: std::marker::PhantomData<F>,
    pub offset: usize,
}

impl<F: Field, const NUM_CELLS: usize> BaseAir<F> for LoadStoreAir<F, NUM_CELLS> {
    fn width(&self) -> usize {
        LoadStoreCols::<F, NUM_CELLS>::width()
    }
}

#[derive(Debug, Clone)]
pub struct LoadStoreIntegration<F: Field, const NUM_CELLS: usize> {
    pub air: LoadStoreAir<F, NUM_CELLS>,
    pub offset: usize,
}

impl<F: Field, const NUM_CELLS: usize> LoadStoreIntegration<F, NUM_CELLS> {
    pub fn new(offset: usize) -> Self {
        Self {
            air: LoadStoreAir::<F, NUM_CELLS> {
                _marker: std::marker::PhantomData,
                offset,
            },
            offset,
        }
    }
}

impl<F: PrimeField32, A: MachineAdapter<F>, const NUM_CELLS: usize> MachineIntegration<F, A>
    for LoadStoreIntegration<F, NUM_CELLS>
where
    <A::Interface<F> as MachineAdapterInterface<F>>::Reads: Into<[[F; NUM_CELLS]; 2]>,
    <A::Interface<F> as MachineAdapterInterface<F>>::Writes: From<[F; NUM_CELLS]>,
{
    type Record = std::marker::PhantomData<F>;
    type Air = LoadStoreAir<F, NUM_CELLS>;
    type Cols<T> = LoadStoreCols<T, NUM_CELLS>;

    #[allow(clippy::type_complexity)]
    fn execute_instruction(
        &self,
        instruction: &Instruction<F>,
        from_pc: F,
        reads: <A::Interface<F> as MachineAdapterInterface<F>>::Reads,
    ) -> Result<(InstructionOutput<F, A::Interface<F>>, Self::Record)> {
        let opcode = LoadStoreOpcode::from_usize(instruction.opcode - self.offset);
        let data: [[F; NUM_CELLS]; 2] = reads.into();
        let write_data = solve_write_data(opcode, data[0], data[1]);

        let output: InstructionOutput<F, A::Interface<F>> = InstructionOutput {
            to_pc: from_pc,
            writes: write_data.into(),
        };

        Ok((output, std::marker::PhantomData))
    }

    fn get_opcode_name(&self, _opcode: usize) -> String {
        todo!()
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

pub(super) fn solve_write_data<F: PrimeField32, const NUM_CELLS: usize>(
    opcode: LoadStoreOpcode,
    read_data: [F; NUM_CELLS],
    prev_data: [F; NUM_CELLS],
) -> [F; NUM_CELLS] {
    let mut write_data = read_data;
    match opcode {
        LoadStoreOpcode::LOADW => (),
        LoadStoreOpcode::STOREW => (),
        LoadStoreOpcode::STOREH => {
            for (i, cell) in write_data
                .iter_mut()
                .enumerate()
                .take(NUM_CELLS)
                .skip(NUM_CELLS / 2)
            {
                *cell = prev_data[i];
            }
        }
        LoadStoreOpcode::STOREB => {
            for (i, cell) in write_data.iter_mut().enumerate().take(NUM_CELLS).skip(1) {
                *cell = prev_data[i];
            }
        }
    };
    write_data
}
