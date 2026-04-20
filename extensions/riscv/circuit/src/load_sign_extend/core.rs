use std::{
    array,
    borrow::{Borrow, BorrowMut},
};

use openvm_circuit::{
    arch::*,
    system::memory::{online::TracingMemory, MemoryAuxColsFactory},
};
use openvm_circuit_primitives::{
    utils::select,
    var_range::{SharedVariableRangeCheckerChip, VariableRangeCheckerBus},
    AlignedBytesBorrow,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV64_CELL_BITS, RV64_REGISTER_NUM_LIMBS},
    LocalOpcode,
};
use openvm_riscv_transpiler::Rv64LoadStoreOpcode::{self, *};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, PrimeCharacteristicRing, PrimeField32},
    BaseAirWithPublicValues,
};

use crate::adapters::{LoadStoreInstruction, Rv64LoadStoreAdapterFiller};

/// LoadSignExtend Core Chip handles byte/halfword/word into doubleword conversions through sign
/// extend. This chip uses read_data to construct write_data.
/// prev_data columns are not used in constraints defined in the CoreAir, but are used in
/// constraints by the Adapter. shifted_read_data is the read_data shifted by (shift_amount & 4),
/// this reduces the number of opcode flags needed. Using this shifted data we can generate the
/// write_data as if the shift_amount was 0..3 for loadb, 0 or 2 for loadh, and 0 for loadw.
#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow)]
pub struct LoadSignExtendCoreCols<T, const NUM_CELLS: usize> {
    /// This chip treats each (opcode, inner_shift) pair as a different instruction
    pub opcode_loadb_flag0: T,
    pub opcode_loadb_flag1: T,
    pub opcode_loadb_flag2: T,
    pub opcode_loadb_flag3: T,
    pub opcode_loadh_flag0: T,
    pub opcode_loadh_flag2: T,
    pub opcode_loadw_flag: T,

    pub shift_most_sig_bit: T,
    // The bit that is extended to the remaining bits
    pub data_most_sig_bit: T,

    pub shifted_read_data: [T; NUM_CELLS],
    pub prev_data: [T; NUM_CELLS],
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
            opcode_loadb_flag2: is_loadb2,
            opcode_loadb_flag3: is_loadb3,
            opcode_loadh_flag0: is_loadh0,
            opcode_loadh_flag2: is_loadh2,
            opcode_loadw_flag: is_loadw,
            data_most_sig_bit,
            shift_most_sig_bit,
        } = *cols;

        let flags = [
            is_loadb0, is_loadb1, is_loadb2, is_loadb3, is_loadh0, is_loadh2, is_loadw,
        ];

        let is_valid = flags.iter().fold(AB::Expr::ZERO, |acc, &flag| {
            builder.assert_bool(flag);
            acc + flag
        });

        builder.assert_bool(is_valid.clone());
        builder.assert_bool(data_most_sig_bit);
        builder.assert_bool(shift_most_sig_bit);

        let expected_opcode = (is_loadb0 + is_loadb1 + is_loadb2 + is_loadb3)
            * AB::F::from_u8(LOADB as u8)
            + (is_loadh0 + is_loadh2) * AB::F::from_u8(LOADH as u8)
            + is_loadw * AB::F::from_u8(LOADW as u8)
            + AB::Expr::from_usize(Rv64LoadStoreOpcode::CLASS_OFFSET);

        let limb_mask = data_most_sig_bit * AB::Expr::from_u32((1 << LIMB_BITS) - 1);

        let sd = shifted_read_data;

        // there are four parts to write_data:
        // - 1st limb is the sign-extended byte (selected by opcode and inner_shift)
        // - 2nd limb is read_data if loadh/loadw and sign extended if loadb
        // - 3rd to 4th limbs are read_data if loadw and sign extended otherwise
        // - 5th to last limbs are always sign extended limbs
        let write_data: [AB::Expr; NUM_CELLS] = array::from_fn(|i| {
            if i == 0 {
                (is_loadb0 + is_loadh0 + is_loadw) * sd[0]
                    + is_loadb1 * sd[1]
                    + (is_loadb2 + is_loadh2) * sd[2]
                    + is_loadb3 * sd[3]
            } else if i == 1 {
                (is_loadh0 + is_loadw) * sd[1]
                    + is_loadh2 * sd[3]
                    + (is_loadb0 + is_loadb1 + is_loadb2 + is_loadb3) * limb_mask.clone()
            } else if i < 4 {
                is_loadw * sd[i] + (is_valid.clone() - is_loadw) * limb_mask.clone()
            } else {
                limb_mask.clone()
            }
        });

        // Constrain that most_sig_bit is correct
        let most_sig_limb = is_loadb0 * sd[0]
            + (is_loadb1 + is_loadh0) * sd[1]
            + is_loadb2 * sd[2]
            + (is_loadb3 + is_loadh2 + is_loadw) * sd[3];

        self.range_bus
            .range_check(
                most_sig_limb
                    - data_most_sig_bit * AB::Expr::from_u32(1 << (LIMB_BITS - 1)),
                LIMB_BITS - 1,
            )
            .eval(builder, is_valid.clone());

        // Unshift the shifted_read_data to get the original read_data
        let read_data: [AB::Expr; NUM_CELLS] = array::from_fn(|i| {
            select(
                shift_most_sig_bit,
                sd[(i + NUM_CELLS - 4) % NUM_CELLS],
                sd[i],
            )
        });

        let load_shift_amount = shift_most_sig_bit * AB::Expr::from_u32(4)
            + is_loadb1
            + (is_loadb2 + is_loadh2) * AB::Expr::TWO
            + is_loadb3 * AB::Expr::from_u32(3);

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
        Rv64LoadStoreOpcode::CLASS_OFFSET
    }
}

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug)]
pub struct LoadSignExtendCoreRecord<const NUM_CELLS: usize> {
    pub is_byte: bool,
    pub is_word: bool,
    pub shift_amount: u8,
    pub read_data: [u8; NUM_CELLS],
    pub prev_data: [u8; NUM_CELLS],
}

#[derive(Clone, Copy, derive_new::new)]
pub struct LoadSignExtendExecutor<A, const NUM_CELLS: usize, const LIMB_BITS: usize> {
    adapter: A,
}

#[derive(Clone, derive_new::new)]
pub struct LoadSignExtendFiller<
    A = Rv64LoadStoreAdapterFiller,
    const NUM_CELLS: usize = RV64_REGISTER_NUM_LIMBS,
    const LIMB_BITS: usize = RV64_CELL_BITS,
> {
    adapter: A,
    pub range_checker_chip: SharedVariableRangeCheckerChip,
}

impl<F, A, RA, const NUM_CELLS: usize, const LIMB_BITS: usize> PreflightExecutor<F, RA>
    for LoadSignExtendExecutor<A, NUM_CELLS, LIMB_BITS>
where
    F: PrimeField32,
    A: 'static
        + AdapterTraceExecutor<
            F,
            ReadData = (([u32; NUM_CELLS], [u8; NUM_CELLS]), u8),
            WriteData = [u32; NUM_CELLS],
        >,
    for<'buf> RA: RecordArena<
        'buf,
        EmptyAdapterCoreLayout<F, A>,
        (
            A::RecordMut<'buf>,
            &'buf mut LoadSignExtendCoreRecord<NUM_CELLS>,
        ),
    >,
{
    fn get_opcode_name(&self, opcode: usize) -> String {
        format!(
            "{:?}",
            Rv64LoadStoreOpcode::from_usize(opcode - Rv64LoadStoreOpcode::CLASS_OFFSET)
        )
    }

    fn execute(
        &self,
        state: VmStateMut<F, TracingMemory, RA>,
        instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError> {
        let Instruction { opcode, .. } = instruction;

        let local_opcode = Rv64LoadStoreOpcode::from_usize(
            opcode.local_opcode_idx(Rv64LoadStoreOpcode::CLASS_OFFSET),
        );

        let (mut adapter_record, core_record) = state.ctx.alloc(EmptyAdapterCoreLayout::new());

        A::start(*state.pc, state.memory, &mut adapter_record);

        let tmp = self
            .adapter
            .read(state.memory, instruction, &mut adapter_record);

        core_record.is_byte = local_opcode == LOADB;
        core_record.is_word = local_opcode == LOADW;
        core_record.prev_data = tmp.0 .0.map(|x| x as u8);
        core_record.read_data = tmp.0 .1;
        core_record.shift_amount = tmp.1;

        let write_data = run_write_data_sign_extend(
            local_opcode,
            core_record.read_data,
            core_record.shift_amount as usize,
        );

        self.adapter.write(
            state.memory,
            instruction,
            write_data.map(u32::from),
            &mut adapter_record,
        );

        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);

        Ok(())
    }
}

impl<F, A, const NUM_CELLS: usize, const LIMB_BITS: usize> TraceFiller<F>
    for LoadSignExtendFiller<A, NUM_CELLS, LIMB_BITS>
where
    F: PrimeField32,
    A: 'static + AdapterTraceFiller<F>,
{
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        // SAFETY: row_slice is guaranteed by the caller to have at least A::WIDTH +
        // LoadSignExtendCoreCols::width() elements
        let (adapter_row, mut core_row) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };
        self.adapter.fill_trace_row(mem_helper, adapter_row);
        // SAFETY: core_row contains a valid LoadSignExtendCoreRecord written by the executor
        // during trace generation
        let record: &LoadSignExtendCoreRecord<NUM_CELLS> =
            unsafe { get_record_from_slice(&mut core_row, ()) };

        let core_row: &mut LoadSignExtendCoreCols<F, NUM_CELLS> = core_row.borrow_mut();

        let shift = record.shift_amount as usize;
        let shift_most_sig_bit = (shift >> 2) & 1;
        let inner_shift = shift & 3;

        let mut shifted = record.read_data;
        shifted.rotate_left(shift_most_sig_bit * 4);

        let most_sig_limb = if record.is_byte {
            shifted[inner_shift]
        } else if record.is_word {
            shifted[3]
        } else {
            shifted[inner_shift + 1]
        };

        let most_sig_bit = most_sig_limb & (1 << 7);
        self.range_checker_chip
            .add_count((most_sig_limb - most_sig_bit) as u32, 7);

        core_row.prev_data = record.prev_data.map(F::from_u8);
        core_row.shifted_read_data = shifted.map(F::from_u8);

        core_row.data_most_sig_bit = F::from_bool(most_sig_bit != 0);
        core_row.shift_most_sig_bit = F::from_bool(shift_most_sig_bit == 1);

        let is_byte = record.is_byte;
        let is_word = record.is_word;
        let is_half = !is_byte && !is_word;

        core_row.opcode_loadb_flag0 = F::from_bool(is_byte && inner_shift == 0);
        core_row.opcode_loadb_flag1 = F::from_bool(is_byte && inner_shift == 1);
        core_row.opcode_loadb_flag2 = F::from_bool(is_byte && inner_shift == 2);
        core_row.opcode_loadb_flag3 = F::from_bool(is_byte && inner_shift == 3);
        core_row.opcode_loadh_flag0 = F::from_bool(is_half && inner_shift == 0);
        core_row.opcode_loadh_flag2 = F::from_bool(is_half && inner_shift == 2);
        core_row.opcode_loadw_flag = F::from_bool(is_word);
    }
}

// Returns write_data
#[inline(always)]
pub(super) fn run_write_data_sign_extend<const NUM_CELLS: usize>(
    opcode: Rv64LoadStoreOpcode,
    read_data: [u8; NUM_CELLS],
    shift: usize,
) -> [u8; NUM_CELLS] {
    assert!(
        NUM_CELLS > 4,
        "sign extension must extend at least one byte"
    );
    match opcode {
        LOADW => {
            assert!(
                shift == 0 || shift == 4,
                "LOADW requires 4-byte aligned shift, got {shift}"
            );
            assert!(shift + 4 <= NUM_CELLS);
            let ext = (read_data[shift + 3] >> 7) * u8::MAX;
            array::from_fn(|i| {
                if i < 4 {
                    read_data[i + shift]
                } else {
                    ext
                }
            })
        }
        LOADH => {
            assert!(
                shift % 2 == 0,
                "LOADH requires 2-byte aligned shift, got {shift}"
            );
            debug_assert!(shift + 2 <= NUM_CELLS);
            let ext = (read_data[shift + 1] >> 7) * u8::MAX;
            array::from_fn(|i| {
                if i < 2 {
                    read_data[i + shift]
                } else {
                    ext
                }
            })
        }
        LOADB => {
            debug_assert!(shift < NUM_CELLS);
            let ext = (read_data[shift] >> 7) * u8::MAX;
            array::from_fn(|i| {
                if i == 0 {
                    read_data[i + shift]
                } else {
                    ext
                }
            })
        }
        // Currently the adapter AIR requires `ptr_val` to be aligned to the data size in bytes.
        // The circuit requires that `shift = ptr_val % 8` so that `ptr_val - shift` is a multiple of 8.
        // This requirement is non-trivial to remove, because we use it to ensure that `ptr_val - shift + 8 <= 2^pointer_max_bits`.
        _ => unreachable!(
            "unaligned memory access not supported by this execution environment: {opcode:?}, shift: {shift}"
        ),
    }
}
