use std::{marker::PhantomData, mem::size_of};

use afs_stark_backend::{interaction::InteractionBuilder, rap::BaseAirWithPublicValues};
use p3_air::BaseAir;
use p3_field::{Field, PrimeField32};

use crate::{
    arch::{
        instructions::{
            Rv32LoadStoreOpcode::{self, *},
            UsizeOpcode,
        },
        AdapterContext, AdapterAirContext, Reads, Result, VmAdapter, VmAdapterInterface, VmCore, VmCoreAir,
        Writes,
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
pub struct LoadStoreAir<F: Field, const NUM_CELLS: usize> {
    pub _marker: PhantomData<F>,
    pub offset: usize,
}

impl<F: Field, const NUM_CELLS: usize> BaseAir<F> for LoadStoreAir<F, NUM_CELLS> {
    fn width(&self) -> usize {
        LoadStoreCols::<F, NUM_CELLS>::width()
    }
}

impl<F: Field, const NUM_CELLS: usize> BaseAirWithPublicValues<F> for LoadStoreAir<F, NUM_CELLS> {}

impl<AB, I, const NUM_CELLS: usize> VmCoreAir<AB, I> for LoadStoreAir<AB::F, NUM_CELLS>
where
    AB: InteractionBuilder,
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
pub struct LoadStoreCore<F: Field, const NUM_CELLS: usize> {
    pub air: LoadStoreAir<F, NUM_CELLS>,
}

impl<F: Field, const NUM_CELLS: usize> LoadStoreCore<F, NUM_CELLS> {
    pub fn new(offset: usize) -> Self {
        Self {
            air: LoadStoreAir::<F, NUM_CELLS> {
                _marker: PhantomData,
                offset,
            },
        }
    }
}

impl<F: PrimeField32, A: VmAdapter<F>, const NUM_CELLS: usize> VmCore<F, A>
    for LoadStoreCore<F, NUM_CELLS>
where
    Reads<F, A::Interface<F>>: Into<[[F; NUM_CELLS]; 2]>,
    Writes<F, A::Interface<F>>: From<[F; NUM_CELLS]>,
{
    type Record = ();
    type Air = LoadStoreAir<F, NUM_CELLS>;

    #[allow(clippy::type_complexity)]
    fn execute_instruction(
        &self,
        instruction: &Instruction<F>,
        _from_pc: F,
        reads: <A::Interface<F> as VmAdapterInterface<F>>::Reads,
    ) -> Result<(AdapterContext<F, A::Interface<F>>, Self::Record)> {
        let opcode = Rv32LoadStoreOpcode::from_usize(instruction.opcode - self.air.offset);
        let data: [[F; NUM_CELLS]; 2] = reads.into();
        let write_data = solve_write_data(opcode, data[0], data[1]);

        let output: AdapterContext<F, A::Interface<F>> = AdapterContext {
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

    fn generate_trace_row(&self, _row_slice: &mut [F], _record: Self::Record) {
        todo!()
    }

    fn air(&self) -> &Self::Air {
        &self.air
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
