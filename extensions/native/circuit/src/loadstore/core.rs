use std::{
    array,
    borrow::{Borrow, BorrowMut},
    sync::{Arc, Mutex, OnceLock},
};

use openvm_circuit::arch::{
    instructions::LocalOpcode, AdapterAirContext, AdapterRuntimeContext, ExecutionError, Result,
    Streams, VmAdapterInterface, VmCoreAir, VmCoreChip,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::instruction::Instruction;
use openvm_native_compiler::NativeLoadStoreOpcode;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, FieldAlgebra, PrimeField32},
    rap::BaseAirWithPublicValues,
};
use serde::{Deserialize, Serialize};
use serde_big_array::BigArray;
use strum::IntoEnumIterator;

use super::super::adapters::loadstore_native_adapter::NativeLoadStoreInstruction;

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct NativeLoadStoreCoreCols<T, const NUM_CELLS: usize> {
    pub is_loadw: T,
    pub is_storew: T,
    pub is_hint_storew: T,

    pub pointer_read: T,
    pub data: [T; NUM_CELLS],
}

#[repr(C)]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NativeLoadStoreCoreRecord<F, const NUM_CELLS: usize> {
    pub opcode: NativeLoadStoreOpcode,

    pub pointer_read: F,
    #[serde(with = "BigArray")]
    pub data: [F; NUM_CELLS],
}

#[derive(Clone, Debug)]
pub struct NativeLoadStoreCoreAir<const NUM_CELLS: usize> {
    pub offset: usize,
}

impl<F: Field, const NUM_CELLS: usize> BaseAir<F> for NativeLoadStoreCoreAir<NUM_CELLS> {
    fn width(&self) -> usize {
        NativeLoadStoreCoreCols::<F, NUM_CELLS>::width()
    }
}

impl<F: Field, const NUM_CELLS: usize> BaseAirWithPublicValues<F>
    for NativeLoadStoreCoreAir<NUM_CELLS>
{
}

impl<AB, I, const NUM_CELLS: usize> VmCoreAir<AB, I> for NativeLoadStoreCoreAir<NUM_CELLS>
where
    AB: InteractionBuilder,
    I: VmAdapterInterface<AB::Expr>,
    I::Reads: From<(AB::Expr, [AB::Expr; NUM_CELLS])>,
    I::Writes: From<[AB::Expr; NUM_CELLS]>,
    I::ProcessedInstruction: From<NativeLoadStoreInstruction<AB::Expr>>,
{
    fn eval(
        &self,
        builder: &mut AB,
        local_core: &[AB::Var],
        _from_pc: AB::Var,
    ) -> AdapterAirContext<AB::Expr, I> {
        let cols: &NativeLoadStoreCoreCols<_, NUM_CELLS> = (*local_core).borrow();
        let flags = [cols.is_loadw, cols.is_storew, cols.is_hint_storew];
        let is_valid = flags.iter().fold(AB::Expr::ZERO, |acc, &flag| {
            builder.assert_bool(flag);
            acc + flag.into()
        });
        builder.assert_bool(is_valid.clone());

        let expected_opcode = VmCoreAir::<AB, I>::expr_to_global_expr(
            self,
            flags.iter().zip(NativeLoadStoreOpcode::iter()).fold(
                AB::Expr::ZERO,
                |acc, (flag, local_opcode)| {
                    acc + (*flag).into()
                        * AB::Expr::from_canonical_usize(local_opcode.local_usize())
                },
            ),
        );

        AdapterAirContext {
            to_pc: None,
            reads: (cols.pointer_read.into(), cols.data.map(Into::into)).into(),
            writes: cols.data.map(Into::into).into(),
            instruction: NativeLoadStoreInstruction {
                is_valid,
                opcode: expected_opcode,
                is_loadw: cols.is_loadw.into(),
                is_storew: cols.is_storew.into(),
                is_hint_storew: cols.is_hint_storew.into(),
            }
            .into(),
        }
    }

    fn start_offset(&self) -> usize {
        self.offset
    }
}

pub struct NativeLoadStoreCoreChip<F: Field, const NUM_CELLS: usize> {
    pub air: NativeLoadStoreCoreAir<NUM_CELLS>,
    pub streams: OnceLock<Arc<Mutex<Streams<F>>>>,
}

impl<F: Field, const NUM_CELLS: usize> NativeLoadStoreCoreChip<F, NUM_CELLS> {
    pub fn new(offset: usize) -> Self {
        Self {
            air: NativeLoadStoreCoreAir::<NUM_CELLS> { offset },
            streams: OnceLock::new(),
        }
    }
    pub fn set_streams(&mut self, streams: Arc<Mutex<Streams<F>>>) {
        self.streams
            .set(streams)
            .map_err(|_| "streams have already been set.")
            .unwrap();
    }
}

impl<F: Field, const NUM_CELLS: usize> Default for NativeLoadStoreCoreChip<F, NUM_CELLS> {
    fn default() -> Self {
        Self::new(NativeLoadStoreOpcode::CLASS_OFFSET)
    }
}

impl<F: PrimeField32, I: VmAdapterInterface<F>, const NUM_CELLS: usize> VmCoreChip<F, I>
    for NativeLoadStoreCoreChip<F, NUM_CELLS>
where
    I::Reads: Into<(F, [F; NUM_CELLS])>,
    I::Writes: From<[F; NUM_CELLS]>,
{
    type Record = NativeLoadStoreCoreRecord<F, NUM_CELLS>;
    type Air = NativeLoadStoreCoreAir<NUM_CELLS>;

    fn execute_instruction(
        &self,
        instruction: &Instruction<F>,
        from_pc: u32,
        reads: I::Reads,
    ) -> Result<(AdapterRuntimeContext<F, I>, Self::Record)> {
        let Instruction { opcode, .. } = *instruction;
        let local_opcode =
            NativeLoadStoreOpcode::from_usize(opcode.local_opcode_idx(self.air.offset));
        let (pointer_read, data_read) = reads.into();

        let data = if local_opcode == NativeLoadStoreOpcode::HINT_STOREW {
            let mut streams = self.streams.get().unwrap().lock().unwrap();
            if streams.hint_stream.len() < NUM_CELLS {
                return Err(ExecutionError::HintOutOfBounds { pc: from_pc });
            }
            array::from_fn(|_| streams.hint_stream.pop_front().unwrap())
        } else {
            data_read
        };

        let output = AdapterRuntimeContext::without_pc(data);
        let record = NativeLoadStoreCoreRecord {
            opcode: NativeLoadStoreOpcode::from_usize(opcode.local_opcode_idx(self.air.offset)),
            pointer_read,
            data,
        };
        Ok((output, record))
    }

    fn get_opcode_name(&self, opcode: usize) -> String {
        format!(
            "{:?}",
            NativeLoadStoreOpcode::from_usize(opcode - self.air.offset)
        )
    }

    fn generate_trace_row(&self, row_slice: &mut [F], record: Self::Record) {
        let cols: &mut NativeLoadStoreCoreCols<_, NUM_CELLS> = row_slice.borrow_mut();
        cols.is_loadw = F::from_bool(record.opcode == NativeLoadStoreOpcode::LOADW);
        cols.is_storew = F::from_bool(record.opcode == NativeLoadStoreOpcode::STOREW);
        cols.is_hint_storew = F::from_bool(record.opcode == NativeLoadStoreOpcode::HINT_STOREW);

        cols.pointer_read = record.pointer_read;
        cols.data = record.data;
    }

    fn air(&self) -> &Self::Air {
        &self.air
    }
}
