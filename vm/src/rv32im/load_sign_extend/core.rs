use std::{
    array,
    borrow::{Borrow, BorrowMut},
    sync::Arc,
};

use afs_derive::AlignedBorrow;
use afs_primitives::var_range::{bus::VariableRangeCheckerBus, VariableRangeCheckerChip};
use afs_stark_backend::{interaction::InteractionBuilder, rap::BaseAirWithPublicValues};
use p3_air::BaseAir;
use p3_field::{AbstractField, Field, PrimeField32};
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
    system::program::Instruction,
};

/// LoadSignExtend Core Chip handles byte/halfword into word conversions through sign extend
/// This chip uses read_data to construct write_data
/// This chip doesn't require prev_data, but it is added for compatibility with the LoadStore Adapter chip
#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow)]
pub struct LoadSignExtendCoreCols<T, const NUM_CELLS: usize> {
    pub opcode_loadb_flag: T,
    pub opcode_loadh_flag: T,

    // The bit that is extended to the remaining bits
    pub most_sig_bit: T,

    pub read_data: [T; NUM_CELLS],
    pub prev_data: [T; NUM_CELLS],
}

#[derive(Debug, Clone)]
pub struct LoadSignExtendCoreRecord<F, const NUM_CELLS: usize> {
    pub opcode: Rv32LoadStoreOpcode,
    pub most_sig_bit: bool,
    pub read_data: [F; NUM_CELLS],
    pub prev_data: [F; NUM_CELLS],
}

#[derive(Debug, Clone)]
pub struct LoadSignExtendCoreAir<const NUM_CELLS: usize, const LIMB_BITS: usize> {
    pub range_bus: VariableRangeCheckerBus,
    pub offset: usize,
}

impl<F: Field, const NUM_CELLS: usize, const LIMB_BITS: usize> BaseAir<F>
    for LoadSignExtendCoreAir<NUM_CELLS, LIMB_BITS>
{
    fn width(&self) -> usize {
        LoadSignExtendCoreCols::<F, NUM_CELLS>::width()
    }
}

impl<F: Field, const NUM_CELLS: usize, const LIMB_BITS: usize> BaseAirWithPublicValues<F>
    for LoadSignExtendCoreAir<NUM_CELLS, LIMB_BITS>
{
}

impl<AB, I, const NUM_CELLS: usize, const LIMB_BITS: usize> VmCoreAir<AB, I>
    for LoadSignExtendCoreAir<NUM_CELLS, LIMB_BITS>
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
        let cols: &LoadSignExtendCoreCols<AB::Var, NUM_CELLS> = (*local_core).borrow();
        let LoadSignExtendCoreCols::<AB::Var, NUM_CELLS> {
            read_data,
            prev_data,
            opcode_loadb_flag,
            opcode_loadh_flag,
            most_sig_bit,
        } = *cols;
        let flags = [opcode_loadb_flag, opcode_loadh_flag];

        let is_valid = flags.iter().fold(AB::Expr::zero(), |acc, &flag| {
            builder.assert_bool(flag);
            acc + flag.into()
        });
        builder.assert_bool(is_valid.clone());
        builder.assert_bool(most_sig_bit);

        let expected_opcode = flags
            .iter()
            .zip(Rv32LoadStoreOpcode::iter().skip(7))
            .fold(AB::Expr::zero(), |acc, (flag, opcode)| {
                acc + (*flag).into() * AB::Expr::from_canonical_u8(opcode as u8)
            })
            + AB::Expr::from_canonical_usize(self.offset);

        let write_data: [AB::Expr; NUM_CELLS] = array::from_fn(|i| {
            if i == 0 {
                read_data[i].into()
            } else if i < NUM_CELLS / 2 {
                read_data[i] * opcode_loadh_flag
                    + opcode_loadb_flag
                        * most_sig_bit
                        * AB::Expr::from_canonical_u32((1 << LIMB_BITS) - 1)
            } else {
                most_sig_bit * AB::Expr::from_canonical_u32((1 << LIMB_BITS) - 1)
            }
        });

        // Constrain that most_sig_bit is correct
        let most_sig_limb =
            read_data[0] * opcode_loadb_flag + read_data[NUM_CELLS / 2 - 1] * opcode_loadh_flag;
        self.range_bus
            .range_check(
                most_sig_limb - most_sig_bit * AB::Expr::from_canonical_u32(1 << (LIMB_BITS - 1)),
                LIMB_BITS - 1,
            )
            .eval(builder, is_valid.clone());

        AdapterAirContext {
            to_pc: None,
            reads: [prev_data, read_data].into(),
            writes: [write_data].into(),
            instruction: LoadStoreProcessedInstruction {
                is_valid,
                opcode: expected_opcode,
                is_load: AB::Expr::one(),
                is_hint: AB::Expr::zero(),
            }
            .into(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct LoadSignExtendCoreChip<const NUM_CELLS: usize, const LIMB_BITS: usize> {
    pub air: LoadSignExtendCoreAir<NUM_CELLS, LIMB_BITS>,
    pub range_checker_chip: Arc<VariableRangeCheckerChip>,
}

impl<const NUM_CELLS: usize, const LIMB_BITS: usize> LoadSignExtendCoreChip<NUM_CELLS, LIMB_BITS> {
    pub fn new(range_checker_chip: Arc<VariableRangeCheckerChip>, offset: usize) -> Self {
        Self {
            air: LoadSignExtendCoreAir::<NUM_CELLS, LIMB_BITS> {
                range_bus: range_checker_chip.bus(),
                offset,
            },
            range_checker_chip,
        }
    }
}

impl<F: PrimeField32, I: VmAdapterInterface<F>, const NUM_CELLS: usize, const LIMB_BITS: usize>
    VmCoreChip<F, I> for LoadSignExtendCoreChip<NUM_CELLS, LIMB_BITS>
where
    I::Reads: Into<[[F; NUM_CELLS]; 2]>,
    I::Writes: From<[[F; NUM_CELLS]; 1]>,
{
    type Record = LoadSignExtendCoreRecord<F, NUM_CELLS>;
    type Air = LoadSignExtendCoreAir<NUM_CELLS, LIMB_BITS>;

    #[allow(clippy::type_complexity)]
    fn execute_instruction(
        &self,
        instruction: &Instruction<F>,
        _from_pc: u32,
        reads: I::Reads,
    ) -> Result<(AdapterRuntimeContext<F, I>, Self::Record)> {
        let local_opcode_index =
            Rv32LoadStoreOpcode::from_usize(instruction.opcode - self.air.offset);

        let data: [[F; NUM_CELLS]; 2] = reads.into();
        let write_data: [F; NUM_CELLS] = run_write_data_sign_extend::<_, NUM_CELLS, LIMB_BITS>(
            local_opcode_index,
            data[1],
            data[0],
        );
        let output = AdapterRuntimeContext::without_pc([write_data]);

        let most_sig_limb = match local_opcode_index {
            LOADB => data[1][0],
            LOADH => data[1][NUM_CELLS / 2 - 1],
            _ => unreachable!(),
        }
        .as_canonical_u32();

        let most_sig_bit = most_sig_limb & (1 << (LIMB_BITS - 1));
        self.range_checker_chip
            .add_count(most_sig_limb - most_sig_bit, LIMB_BITS - 1);
        Ok((
            output,
            LoadSignExtendCoreRecord {
                opcode: local_opcode_index,
                most_sig_bit: most_sig_bit != 0,
                prev_data: data[0],
                read_data: data[1],
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
        let core_cols: &mut LoadSignExtendCoreCols<F, NUM_CELLS> = row_slice.borrow_mut();
        let opcode = record.opcode;
        core_cols.opcode_loadb_flag = F::from_bool(opcode == LOADB);
        core_cols.opcode_loadh_flag = F::from_bool(opcode == LOADH);
        core_cols.most_sig_bit = F::from_bool(record.most_sig_bit);
        core_cols.prev_data = record.prev_data;
        core_cols.read_data = record.read_data;
    }

    fn air(&self) -> &Self::Air {
        &self.air
    }
}

pub(super) fn run_write_data_sign_extend<
    F: PrimeField32,
    const NUM_CELLS: usize,
    const LIMB_BITS: usize,
>(
    opcode: Rv32LoadStoreOpcode,
    read_data: [F; NUM_CELLS],
    _prev_data: [F; NUM_CELLS],
) -> [F; NUM_CELLS] {
    let mut write_data = read_data;
    match opcode {
        LOADH => {
            let ext = read_data[NUM_CELLS / 2 - 1].as_canonical_u32();
            let ext = (ext >> (LIMB_BITS - 1)) * ((1 << LIMB_BITS) - 1);
            for cell in write_data.iter_mut().take(NUM_CELLS).skip(NUM_CELLS / 2) {
                *cell = F::from_canonical_u32(ext);
            }
        }
        LOADB => {
            let ext = read_data[0].as_canonical_u32();
            let ext = (ext >> (LIMB_BITS - 1)) * ((1 << LIMB_BITS) - 1);
            for cell in write_data.iter_mut().take(NUM_CELLS).skip(1) {
                *cell = F::from_canonical_u32(ext);
            }
        }
        _ => unreachable!(),
    };
    write_data
}
