use std::{
    array,
    borrow::{Borrow, BorrowMut},
};

use openvm_circuit::{
    arch::{
        AdapterAirContext, AdapterExecutorE1, AdapterTraceStep, Result, StepExecutorE1, TraceStep,
        VmAdapterInterface, VmCoreAir, VmStateMut,
    },
    system::memory::{
        online::{GuestMemory, TracingMemory},
        MemoryAuxColsFactory,
    },
};
use openvm_circuit_primitives::{
    utils::select,
    var_range::{SharedVariableRangeCheckerChip, VariableRangeCheckerBus},
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{instruction::Instruction, program::DEFAULT_PC_STEP, LocalOpcode};
use openvm_rv32im_transpiler::Rv32LoadStoreOpcode::{self, *};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, FieldAlgebra, PrimeField32},
    rap::BaseAirWithPublicValues,
};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_big_array::BigArray;

use crate::adapters::LoadStoreInstruction;

/// LoadSignExtend Core Chip handles byte/halfword into word conversions through sign extend
/// This chip uses read_data to construct write_data
/// prev_data columns are not used in constraints defined in the CoreAir, but are used in
/// constraints by the Adapter shifted_read_data is the read_data shifted by (shift_amount & 2),
/// this reduces the number of opcode flags needed using this shifted data we can generate the
/// write_data as if the shift_amount was 0 for loadh and 0 or 1 for loadb
#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow)]
pub struct LoadSignExtendCoreCols<T, const NUM_CELLS: usize> {
    /// This chip treats loadb with 0 shift and loadb with 1 shift as different instructions
    pub opcode_loadb_flag0: T,
    pub opcode_loadb_flag1: T,
    pub opcode_loadh_flag: T,

    pub shift_most_sig_bit: T,
    // The bit that is extended to the remaining bits
    pub data_most_sig_bit: T,

    pub shifted_read_data: [T; NUM_CELLS],
    pub prev_data: [T; NUM_CELLS],
}

#[repr(C)]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "F: Serialize + DeserializeOwned")]
pub struct LoadSignExtendCoreRecord<F, const NUM_CELLS: usize> {
    #[serde(with = "BigArray")]
    pub shifted_read_data: [F; NUM_CELLS],
    #[serde(with = "BigArray")]
    pub prev_data: [F; NUM_CELLS],
    pub opcode: Rv32LoadStoreOpcode,
    pub shift_amount: u32,
    pub most_sig_bit: bool,
}

#[derive(Debug, Clone, derive_new::new)]
pub struct LoadSignExtendCoreAir<const NUM_CELLS: usize, const LIMB_BITS: usize> {
    pub range_bus: VariableRangeCheckerBus,
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
    I::Reads: From<([AB::Var; NUM_CELLS], [AB::Expr; NUM_CELLS])>,
    I::Writes: From<[[AB::Expr; NUM_CELLS]; 1]>,
    I::ProcessedInstruction: From<LoadStoreInstruction<AB::Expr>>,
{
    fn eval(
        &self,
        builder: &mut AB,
        local_core: &[AB::Var],
        _from_pc: AB::Var,
    ) -> AdapterAirContext<AB::Expr, I> {
        let cols: &LoadSignExtendCoreCols<AB::Var, NUM_CELLS> = (*local_core).borrow();
        let LoadSignExtendCoreCols::<AB::Var, NUM_CELLS> {
            shifted_read_data,
            prev_data,
            opcode_loadb_flag0: is_loadb0,
            opcode_loadb_flag1: is_loadb1,
            opcode_loadh_flag: is_loadh,
            data_most_sig_bit,
            shift_most_sig_bit,
        } = *cols;

        let flags = [is_loadb0, is_loadb1, is_loadh];

        let is_valid = flags.iter().fold(AB::Expr::ZERO, |acc, &flag| {
            builder.assert_bool(flag);
            acc + flag
        });

        builder.assert_bool(is_valid.clone());
        builder.assert_bool(data_most_sig_bit);
        builder.assert_bool(shift_most_sig_bit);

        let expected_opcode = (is_loadb0 + is_loadb1) * AB::F::from_canonical_u8(LOADB as u8)
            + is_loadh * AB::F::from_canonical_u8(LOADH as u8)
            + AB::Expr::from_canonical_usize(Rv32LoadStoreOpcode::CLASS_OFFSET);

        let limb_mask = data_most_sig_bit * AB::Expr::from_canonical_u32((1 << LIMB_BITS) - 1);

        // there are three parts to write_data:
        // - 1st limb is always shifted_read_data
        // - 2nd to (NUM_CELLS/2)th limbs are read_data if loadh and sign extended if loadb
        // - (NUM_CELLS/2 + 1)th to last limbs are always sign extended limbs
        let write_data: [AB::Expr; NUM_CELLS] = array::from_fn(|i| {
            if i == 0 {
                (is_loadh + is_loadb0) * shifted_read_data[i].into()
                    + is_loadb1 * shifted_read_data[i + 1].into()
            } else if i < NUM_CELLS / 2 {
                shifted_read_data[i] * is_loadh + (is_loadb0 + is_loadb1) * limb_mask.clone()
            } else {
                limb_mask.clone()
            }
        });

        // Constrain that most_sig_bit is correct
        let most_sig_limb = shifted_read_data[0] * is_loadb0
            + shifted_read_data[1] * is_loadb1
            + shifted_read_data[NUM_CELLS / 2 - 1] * is_loadh;

        self.range_bus
            .range_check(
                most_sig_limb
                    - data_most_sig_bit * AB::Expr::from_canonical_u32(1 << (LIMB_BITS - 1)),
                LIMB_BITS - 1,
            )
            .eval(builder, is_valid.clone());

        // Unshift the shifted_read_data to get the original read_data
        let read_data = array::from_fn(|i| {
            select(
                shift_most_sig_bit,
                shifted_read_data[(i + NUM_CELLS - 2) % NUM_CELLS],
                shifted_read_data[i],
            )
        });
        let load_shift_amount = shift_most_sig_bit * AB::Expr::TWO + is_loadb1;

        AdapterAirContext {
            to_pc: None,
            reads: (prev_data, read_data).into(),
            writes: [write_data].into(),
            instruction: LoadStoreInstruction {
                is_valid: is_valid.clone(),
                opcode: expected_opcode,
                is_load: is_valid,
                load_shift_amount,
                store_shift_amount: AB::Expr::ZERO,
            }
            .into(),
        }
    }

    fn start_offset(&self) -> usize {
        Rv32LoadStoreOpcode::CLASS_OFFSET
    }
}

pub struct LoadSignExtendStep<A, const NUM_CELLS: usize, const LIMB_BITS: usize> {
    adapter: A,
    pub range_checker_chip: SharedVariableRangeCheckerChip,
}

impl<A, const NUM_CELLS: usize, const LIMB_BITS: usize>
    LoadSignExtendStep<A, NUM_CELLS, LIMB_BITS>
{
    pub fn new(adapter: A, range_checker_chip: SharedVariableRangeCheckerChip) -> Self {
        Self {
            adapter,
            range_checker_chip,
        }
    }
}

impl<F, CTX, A, const NUM_CELLS: usize, const LIMB_BITS: usize> TraceStep<F, CTX>
    for LoadSignExtendStep<A, NUM_CELLS, LIMB_BITS>
where
    F: PrimeField32,
    A: 'static
        + for<'a> AdapterTraceStep<
            F,
            CTX,
            ReadData = (([u8; NUM_CELLS], [u8; NUM_CELLS]), u32),
            WriteData = [u8; NUM_CELLS],
            TraceContext<'a> = &'a SharedVariableRangeCheckerChip,
        >,
{
    fn get_opcode_name(&self, opcode: usize) -> String {
        format!(
            "{:?}",
            Rv32LoadStoreOpcode::from_usize(opcode - Rv32LoadStoreOpcode::CLASS_OFFSET)
        )
    }

    fn execute(
        &mut self,
        state: VmStateMut<TracingMemory<F>, CTX>,
        instruction: &Instruction<F>,
        trace: &mut [F],
        trace_offset: &mut usize,
        width: usize,
    ) -> Result<()> {
        let Instruction { opcode, .. } = instruction;

        let local_opcode = Rv32LoadStoreOpcode::from_usize(
            opcode.local_opcode_idx(Rv32LoadStoreOpcode::CLASS_OFFSET),
        );

        let mut row_slice = &mut trace[*trace_offset..*trace_offset + width];
        let (adapter_row, core_row) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };

        A::start(*state.pc, state.memory, adapter_row);

        let ((prev_data, read_data), shift_amount) =
            self.adapter.read(state.memory, instruction, adapter_row);
        let prev_data = prev_data.map(F::from_canonical_u8);
        let read_data = read_data.map(F::from_canonical_u8);

        // TODO(ayush): should functions operate on u8 limbs instead of F?
        let write_data: [F; NUM_CELLS] = run_write_data_sign_extend::<_, NUM_CELLS, LIMB_BITS>(
            local_opcode,
            read_data,
            prev_data,
            shift_amount,
        );

        let most_sig_limb = match local_opcode {
            LOADB => write_data[0],
            LOADH => write_data[NUM_CELLS / 2 - 1],
            _ => unreachable!(),
        }
        .as_canonical_u32();

        let most_sig_bit = most_sig_limb & (1 << (LIMB_BITS - 1));

        let read_shift = shift_amount & 2;

        let core_row: &mut LoadSignExtendCoreCols<F, NUM_CELLS> = core_row.borrow_mut();
        core_row.opcode_loadb_flag0 =
            F::from_bool(local_opcode == LOADB && (shift_amount & 1) == 0);
        core_row.opcode_loadb_flag1 =
            F::from_bool(local_opcode == LOADB && (shift_amount & 1) == 1);
        core_row.opcode_loadh_flag = F::from_bool(local_opcode == LOADH);
        core_row.shift_most_sig_bit = F::from_canonical_u32((shift_amount & 2) >> 1);
        core_row.data_most_sig_bit = F::from_bool(most_sig_bit != 0);
        core_row.prev_data = prev_data;
        core_row.shifted_read_data =
            array::from_fn(|i| read_data[(i + read_shift as usize) % NUM_CELLS]);

        self.adapter.write(
            state.memory,
            instruction,
            adapter_row,
            &write_data.map(|x| x.as_canonical_u32() as u8),
        );

        // TODO(ayush): move to fill_trace_row
        self.range_checker_chip
            .add_count(most_sig_limb - most_sig_bit, LIMB_BITS - 1);

        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);

        *trace_offset += width;

        Ok(())
    }

    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        let (adapter_row, core_row) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };
        let _core_row: &mut LoadSignExtendCoreCols<F, NUM_CELLS> = core_row.borrow_mut();

        self.adapter
            .fill_trace_row(mem_helper, &self.range_checker_chip, adapter_row);
    }
}

impl<F, A, const NUM_CELLS: usize, const LIMB_BITS: usize> StepExecutorE1<F>
    for LoadSignExtendStep<A, NUM_CELLS, LIMB_BITS>
where
    F: PrimeField32,
    A: 'static
        + for<'a> AdapterExecutorE1<
            F,
            ReadData = (([u8; NUM_CELLS], [u8; NUM_CELLS]), u32),
            WriteData = [u8; NUM_CELLS],
        >,
{
    fn execute_e1<Ctx>(
        &mut self,
        state: VmStateMut<GuestMemory, Ctx>,
        instruction: &Instruction<F>,
    ) -> Result<()> {
        let Instruction { opcode, .. } = instruction;

        let local_opcode = Rv32LoadStoreOpcode::from_usize(
            opcode.local_opcode_idx(Rv32LoadStoreOpcode::CLASS_OFFSET),
        );

        let ((_, read_data), shift_amount) = self.adapter.read(state.memory, instruction);
        let read_data = read_data.map(F::from_canonical_u8);

        // TODO(ayush): clean this up for e1
        let write_data = run_write_data_sign_extend::<_, NUM_CELLS, LIMB_BITS>(
            local_opcode,
            read_data,
            [F::ZERO; NUM_CELLS],
            shift_amount,
        );
        let write_data = write_data.map(|x| x.as_canonical_u32() as u8);

        self.adapter.write(state.memory, instruction, &write_data);

        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);

        Ok(())
    }
}

// TODO(ayush): remove _prev_data
#[inline(always)]
pub(super) fn run_write_data_sign_extend<
    F: PrimeField32,
    const NUM_CELLS: usize,
    const LIMB_BITS: usize,
>(
    opcode: Rv32LoadStoreOpcode,
    read_data: [F; NUM_CELLS],
    _prev_data: [F; NUM_CELLS],
    shift: u32,
) -> [F; NUM_CELLS] {
    let shift = shift as usize;
    let mut write_data = read_data;
    match (opcode, shift) {
        (LOADH, 0) | (LOADH, 2) => {
            let ext = read_data[NUM_CELLS / 2 - 1 + shift].as_canonical_u32();
            let ext = (ext >> (LIMB_BITS - 1)) * ((1 << LIMB_BITS) - 1);
            for cell in write_data.iter_mut().take(NUM_CELLS).skip(NUM_CELLS / 2) {
                *cell = F::from_canonical_u32(ext);
            }
            write_data[0..NUM_CELLS / 2]
                .copy_from_slice(&read_data[shift..(NUM_CELLS / 2 + shift)]);
        }
        (LOADB, 0) | (LOADB, 1) | (LOADB, 2) | (LOADB, 3) => {
            let ext = read_data[shift].as_canonical_u32();
            let ext = (ext >> (LIMB_BITS - 1)) * ((1 << LIMB_BITS) - 1);
            for cell in write_data.iter_mut().take(NUM_CELLS).skip(1) {
                *cell = F::from_canonical_u32(ext);
            }
            write_data[0] = read_data[shift];
        }
        // Currently the adapter AIR requires `ptr_val` to be aligned to the data size in bytes.
        // The circuit requires that `shift = ptr_val % 4` so that `ptr_val - shift` is a multiple of 4.
        // This requirement is non-trivial to remove, because we use it to ensure that `ptr_val - shift + 4 <= 2^pointer_max_bits`.
        _ => unreachable!(
            "unaligned memory access not supported by this execution environment: {opcode:?}, shift: {shift}"
        ),
    };
    write_data
}
