use std::{
    array,
    borrow::{Borrow, BorrowMut},
    sync::Arc,
};

use ax_circuit_derive::AlignedBorrow;
use ax_circuit_primitives::xor::{XorBus, XorLookupChip};
use ax_stark_backend::{interaction::InteractionBuilder, rap::BaseAirWithPublicValues};
use axvm_instructions::instruction::Instruction;
use p3_air::BaseAir;
use p3_field::{AbstractField, Field, PrimeField32};
use parking_lot::Mutex;

use crate::{
    arch::{
        instructions::{
            Rv32HintStoreOpcode::{self, *},
            UsizeOpcode,
        },
        AdapterAirContext, AdapterRuntimeContext, MinimalInstruction, Result, Streams,
        VmAdapterInterface, VmCoreAir, VmCoreChip,
    },
    rv32im::adapters::{RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS},
    system::program::ExecutionError,
};

/// HintStore Core Chip handles the range checking of the data to be written to memory
#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow)]
pub struct Rv32HintStoreCoreCols<T> {
    pub is_valid: T,
    pub data: [T; RV32_REGISTER_NUM_LIMBS],
    pub xor_range_check: [T; RV32_REGISTER_NUM_LIMBS / 2],
}

#[derive(Debug, Clone)]
pub struct Rv32HintStoreCoreRecord<F> {
    pub data: [F; RV32_REGISTER_NUM_LIMBS],
    pub xor_range_check: [F; RV32_REGISTER_NUM_LIMBS / 2],
}

#[derive(Debug, Clone)]
pub struct Rv32HintStoreCoreAir {
    pub range_bus: XorBus,
    pub offset: usize,
}

impl<F: Field> BaseAir<F> for Rv32HintStoreCoreAir {
    fn width(&self) -> usize {
        Rv32HintStoreCoreCols::<F>::width()
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for Rv32HintStoreCoreAir {}

impl<AB, I> VmCoreAir<AB, I> for Rv32HintStoreCoreAir
where
    AB: InteractionBuilder,
    I: VmAdapterInterface<AB::Expr>,
    I::Reads: From<[[AB::Expr; RV32_REGISTER_NUM_LIMBS]; 0]>,
    I::Writes: From<[[AB::Expr; RV32_REGISTER_NUM_LIMBS]; 1]>,
    I::ProcessedInstruction: From<MinimalInstruction<AB::Expr>>,
{
    fn eval(
        &self,
        builder: &mut AB,
        local_core: &[AB::Var],
        _from_pc: AB::Var,
    ) -> AdapterAirContext<AB::Expr, I> {
        let cols: &Rv32HintStoreCoreCols<AB::Var> = (*local_core).borrow();

        builder.assert_bool(cols.is_valid);

        let expected_opcode = AB::Expr::from_canonical_usize(HINT_STOREW as usize)
            + AB::Expr::from_canonical_usize(self.offset);

        for i in 0..RV32_REGISTER_NUM_LIMBS / 2 {
            self.range_bus
                .send(
                    cols.data[i * 2],
                    cols.data[i * 2 + 1],
                    cols.xor_range_check[i],
                )
                .eval(builder, cols.is_valid);
        }

        AdapterAirContext {
            to_pc: None,
            reads: [].into(),
            writes: [cols.data.map(|x| x.into())].into(),
            instruction: MinimalInstruction {
                is_valid: cols.is_valid.into(),
                opcode: expected_opcode,
            }
            .into(),
        }
    }
}

#[derive(Debug)]
pub struct Rv32HintStoreCoreChip<F: Field> {
    pub air: Rv32HintStoreCoreAir,
    pub streams: Arc<Mutex<Streams<F>>>,
    pub xor_lookup_chip: Arc<XorLookupChip<RV32_CELL_BITS>>,
}

impl<F: PrimeField32> Rv32HintStoreCoreChip<F> {
    pub fn new(
        streams: Arc<Mutex<Streams<F>>>,
        xor_lookup_chip: Arc<XorLookupChip<RV32_CELL_BITS>>,
        offset: usize,
    ) -> Self {
        Self {
            air: Rv32HintStoreCoreAir {
                range_bus: xor_lookup_chip.bus(),
                offset,
            },
            streams,
            xor_lookup_chip,
        }
    }
}

impl<F: PrimeField32, I: VmAdapterInterface<F>> VmCoreChip<F, I> for Rv32HintStoreCoreChip<F>
where
    I::Reads: Into<[[F; RV32_REGISTER_NUM_LIMBS]; 0]>,
    I::Writes: From<[[F; RV32_REGISTER_NUM_LIMBS]; 1]>,
{
    type Record = Rv32HintStoreCoreRecord<F>;
    type Air = Rv32HintStoreCoreAir;

    #[allow(clippy::type_complexity)]
    fn execute_instruction(
        &self,
        _instruction: &Instruction<F>,
        from_pc: u32,
        _reads: I::Reads,
    ) -> Result<(AdapterRuntimeContext<F, I>, Self::Record)> {
        let mut streams = self.streams.lock();
        if streams.hint_stream.len() < RV32_REGISTER_NUM_LIMBS {
            return Err(ExecutionError::HintOutOfBounds(from_pc));
        }
        let data: [F; RV32_REGISTER_NUM_LIMBS] =
            array::from_fn(|_| streams.hint_stream.pop_front().unwrap());
        let write_data = data;

        let output = AdapterRuntimeContext::without_pc([write_data]);
        let xor_range_check = array::from_fn(|i| {
            F::from_canonical_u32(self.xor_lookup_chip.request(
                write_data[2 * i].as_canonical_u32(),
                write_data[2 * i + 1].as_canonical_u32(),
            ))
        });
        Ok((
            output,
            Rv32HintStoreCoreRecord {
                data: write_data,
                xor_range_check,
            },
        ))
    }

    fn get_opcode_name(&self, opcode: usize) -> String {
        format!(
            "{:?}",
            Rv32HintStoreOpcode::from_usize(opcode - self.air.offset)
        )
    }

    fn generate_trace_row(&self, row_slice: &mut [F], record: Self::Record) {
        let core_cols: &mut Rv32HintStoreCoreCols<F> = row_slice.borrow_mut();
        core_cols.is_valid = F::one();
        core_cols.data = record.data;
        core_cols.xor_range_check = record.xor_range_check;
    }

    fn air(&self) -> &Self::Air {
        &self.air
    }
}
