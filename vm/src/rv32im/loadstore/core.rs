use std::{
    array,
    borrow::{Borrow, BorrowMut},
    sync::Arc,
};

use afs_derive::AlignedBorrow;
use afs_stark_backend::{interaction::InteractionBuilder, rap::BaseAirWithPublicValues};
use p3_air::BaseAir;
use p3_field::{AbstractField, Field, PrimeField32};
use parking_lot::Mutex;
use strum::IntoEnumIterator;

use crate::{
    arch::{
        instructions::{
            Rv32LoadStoreOpcode::{self, *},
            UsizeOpcode,
        },
        AdapterAirContext, AdapterRuntimeContext, Result, VmAdapterInterface, VmCoreAir,
        VmCoreChip,
    },
    rv32im::adapters::LoadStoreProcessedInstruction,
    system::{
        program::{ExecutionError, Instruction},
        vm::Streams,
    },
};

/// LoadStore Core Chip handles byte/halfword into word conversions and unsigned extends
/// This chip uses read_data and prev_data to get the write_data
#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow)]
pub struct LoadStoreCoreCols<T, const NUM_CELLS: usize> {
    pub opcode_loadw_flag: T,
    pub opcode_loadhu_flag: T,
    pub opcode_loadbu_flag: T,
    pub opcode_storew_flag: T,
    pub opcode_storeh_flag: T,
    pub opcode_storeb_flag: T,
    pub opcode_hint_storew_flag: T,

    pub read_data: [T; NUM_CELLS],
    pub prev_data: [T; NUM_CELLS],
}

#[derive(Debug, Clone)]
pub struct LoadStoreCoreRecord<F, const NUM_CELLS: usize> {
    pub opcode: Rv32LoadStoreOpcode,

    pub read_data: [F; NUM_CELLS],
    pub prev_data: [F; NUM_CELLS],
}

#[derive(Debug, Clone)]
pub struct LoadStoreCoreAir<const NUM_CELLS: usize> {
    pub offset: usize,
}

impl<F: Field, const NUM_CELLS: usize> BaseAir<F> for LoadStoreCoreAir<NUM_CELLS> {
    fn width(&self) -> usize {
        LoadStoreCoreCols::<F, NUM_CELLS>::width()
    }
}

impl<F: Field, const NUM_CELLS: usize> BaseAirWithPublicValues<F> for LoadStoreCoreAir<NUM_CELLS> {}

impl<AB, I, const NUM_CELLS: usize> VmCoreAir<AB, I> for LoadStoreCoreAir<NUM_CELLS>
where
    AB: InteractionBuilder,
    I: VmAdapterInterface<AB::Expr>,
    I::Reads: From<[[AB::Var; NUM_CELLS]; 2]>,
    I::Writes: From<[[AB::Expr; NUM_CELLS]; 1]>,
    I::ProcessedInstruction: From<LoadStoreProcessedInstruction<AB::Expr>>,
{
    fn eval(
        &self,
        builder: &mut AB,
        local_core: &[AB::Var],
        _from_pc: AB::Var,
    ) -> AdapterAirContext<AB::Expr, I> {
        let cols: &LoadStoreCoreCols<AB::Var, NUM_CELLS> = (*local_core).borrow();
        let LoadStoreCoreCols::<AB::Var, NUM_CELLS> {
            read_data,
            prev_data,
            opcode_loadw_flag: is_loadw,
            opcode_loadbu_flag: is_loadbu,
            opcode_loadhu_flag: is_loadhu,
            opcode_storew_flag: is_storew,
            opcode_storeb_flag: is_storeb,
            opcode_storeh_flag: is_storeh,
            opcode_hint_storew_flag: is_hint_storew,
        } = *cols;
        let flags = [
            is_loadw,
            is_loadbu,
            is_loadhu,
            is_storew,
            is_storeh,
            is_storeb,
            is_hint_storew,
        ];

        let is_valid = flags.iter().fold(AB::Expr::zero(), |acc, &flag| {
            builder.assert_bool(flag);
            acc + flag.into()
        });
        builder.assert_bool(is_valid.clone());

        let expected_opcode = flags.iter().zip(Rv32LoadStoreOpcode::iter()).fold(
            AB::Expr::zero(),
            |acc, (flag, local_opcode)| {
                acc + (*flag).into() * AB::Expr::from_canonical_u8(local_opcode as u8)
            },
        ) + AB::Expr::from_canonical_usize(self.offset);

        // there are three parts to write_data:
        // 1st limb is always read_data
        // 2nd to (NUM_CELLS/2)th limbs are read_data if loadw/loadhu/storew/storeh/hint_storew
        //                                  prev_data if storeb
        //                                  zero if loadbu
        // (NUM_CELLS/2 + 1)th to last limbs are read_data if loadw/storew/hint_storew
        //                                  prev_data if storeb/storeh
        //                                  zero if loadbu/loadhu
        let write_data: [AB::Expr; NUM_CELLS] = array::from_fn(|i| {
            if i == 0 {
                read_data[i].into()
            } else if i < NUM_CELLS / 2 {
                read_data[i] * (is_loadw + is_loadhu + is_storew + is_storeh + is_hint_storew)
                    + prev_data[i] * is_storeb
            } else {
                read_data[i] * (is_loadw + is_storew + is_hint_storew)
                    + prev_data[i] * (is_storeb + is_storeh)
            }
        });

        let is_load = is_loadw + is_loadhu + is_loadbu;

        AdapterAirContext {
            to_pc: None,
            reads: [prev_data, read_data].into(),
            writes: [write_data].into(),
            instruction: LoadStoreProcessedInstruction {
                is_valid,
                opcode: expected_opcode,
                is_load,
                is_hint: is_hint_storew.into(),
            }
            .into(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct LoadStoreCoreChip<F: Field, const NUM_CELLS: usize> {
    pub air: LoadStoreCoreAir<NUM_CELLS>,
    pub streams: Arc<Mutex<Streams<F>>>,
}

impl<F: PrimeField32, const NUM_CELLS: usize> LoadStoreCoreChip<F, NUM_CELLS> {
    pub fn new(streams: Arc<Mutex<Streams<F>>>, offset: usize) -> Self {
        Self {
            air: LoadStoreCoreAir::<NUM_CELLS> { offset },
            streams,
        }
    }
}

impl<F: PrimeField32, I: VmAdapterInterface<F>, const NUM_CELLS: usize> VmCoreChip<F, I>
    for LoadStoreCoreChip<F, NUM_CELLS>
where
    I::Reads: Into<[[F; NUM_CELLS]; 2]>,
    I::Writes: From<[[F; NUM_CELLS]; 1]>,
{
    type Record = LoadStoreCoreRecord<F, NUM_CELLS>;
    type Air = LoadStoreCoreAir<NUM_CELLS>;

    #[allow(clippy::type_complexity)]
    fn execute_instruction(
        &self,
        instruction: &Instruction<F>,
        from_pc: u32,
        reads: I::Reads,
    ) -> Result<(AdapterRuntimeContext<F, I>, Self::Record)> {
        let local_opcode = Rv32LoadStoreOpcode::from_usize(instruction.opcode - self.air.offset);

        let reads = reads.into();
        let prev_data = reads[0];
        let read_data: [F; NUM_CELLS] = if local_opcode == HINT_STOREW {
            // In case of HINT_STOREW will get the read data from the hint_stream
            let mut streams = self.streams.lock();
            if streams.hint_stream.len() < NUM_CELLS {
                return Err(ExecutionError::HintOutOfBounds(from_pc));
            }
            array::from_fn(|_| streams.hint_stream.pop_front().unwrap())
        } else {
            reads[1]
        };

        let write_data = run_write_data(local_opcode, read_data, prev_data);

        let output = AdapterRuntimeContext::without_pc([write_data]);

        Ok((
            output,
            LoadStoreCoreRecord {
                opcode: local_opcode,
                prev_data,
                read_data,
            },
        ))
    }

    fn get_opcode_name(&self, opcode: usize) -> String {
        format!(
            "{:?}",
            Rv32LoadStoreOpcode::from_usize(opcode - self.air.offset)
        )
    }

    fn generate_trace_row(&self, row_slice: &mut [F], record: Self::Record) {
        let core_cols: &mut LoadStoreCoreCols<F, NUM_CELLS> = row_slice.borrow_mut();
        let opcode = record.opcode;
        core_cols.opcode_loadw_flag = F::from_bool(opcode == LOADW);
        core_cols.opcode_loadhu_flag = F::from_bool(opcode == LOADHU);
        core_cols.opcode_loadbu_flag = F::from_bool(opcode == LOADBU);
        core_cols.opcode_storew_flag = F::from_bool(opcode == STOREW);
        core_cols.opcode_storeh_flag = F::from_bool(opcode == STOREH);
        core_cols.opcode_storeb_flag = F::from_bool(opcode == STOREB);
        core_cols.opcode_hint_storew_flag = F::from_bool(opcode == HINT_STOREW);
        core_cols.prev_data = record.prev_data;
        core_cols.read_data = record.read_data;
    }

    fn air(&self) -> &Self::Air {
        &self.air
    }
}

pub(super) fn run_write_data<F: PrimeField32, const NUM_CELLS: usize>(
    opcode: Rv32LoadStoreOpcode,
    read_data: [F; NUM_CELLS],
    prev_data: [F; NUM_CELLS],
) -> [F; NUM_CELLS] {
    let mut write_data = read_data;
    match opcode {
        LOADW => (),
        LOADBU => {
            for cell in write_data.iter_mut().take(NUM_CELLS).skip(1) {
                *cell = F::zero();
            }
        }
        LOADHU => {
            for cell in write_data.iter_mut().take(NUM_CELLS).skip(NUM_CELLS / 2) {
                *cell = F::zero();
            }
        }
        STOREW => (),
        STOREB => {
            for (i, cell) in write_data.iter_mut().enumerate().take(NUM_CELLS).skip(1) {
                *cell = prev_data[i];
            }
        }
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
        HINT_STOREW => (),
        _ => unreachable!(),
    };
    write_data
}
