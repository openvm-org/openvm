use std::{
    array,
    borrow::{Borrow, BorrowMut},
};

use openvm_circuit::{
    arch::*,
    system::memory::{online::TracingMemory, MemoryAuxColsFactory},
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::{BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip},
    var_range::{SharedVariableRangeCheckerChip, VariableRangeCheckerBus},
    AlignedBytesBorrow, ColumnsAir, StructReflection, StructReflectionHelper,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV64_BYTE_BITS, RV64_MEMORY_AS, RV64_REGISTER_AS, RV64_REGISTER_NUM_LIMBS},
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

/// One case per `(opcode, shift)` pair, with op order [LOADB, LOADH, LOADW] and case index
/// `op_idx * 8 + shift`. Access width is `1 << op_idx`.
pub(crate) const LOAD_SIGN_EXTEND_CASES: usize = 24;

#[inline(always)]
fn sign_extend_op_idx(opcode: Rv64LoadStoreOpcode) -> usize {
    match opcode {
        LOADB => 0,
        LOADH => 1,
        LOADW => 2,
        _ => unreachable!("load sign extend core only handles LOADB/LOADH/LOADW"),
    }
}

#[inline(always)]
fn case_index(opcode: Rv64LoadStoreOpcode, shift: usize) -> usize {
    debug_assert!(shift < RV64_REGISTER_NUM_LIMBS);
    sign_extend_op_idx(opcode) * RV64_REGISTER_NUM_LIMBS + shift
}

#[inline(always)]
fn case_opcode(case: usize) -> Rv64LoadStoreOpcode {
    match case / RV64_REGISTER_NUM_LIMBS {
        0 => LOADB,
        1 => LOADH,
        2 => LOADW,
        _ => unreachable!("invalid load sign extend case"),
    }
}

#[inline(always)]
fn case_shift(case: usize) -> usize {
    case % RV64_REGISTER_NUM_LIMBS
}

#[inline(always)]
fn case_width(case: usize) -> usize {
    1 << (case / RV64_REGISTER_NUM_LIMBS)
}

/// Whether the access at this shift spans two adjacent 8-byte blocks.
#[inline(always)]
fn case_crosses(case: usize) -> bool {
    case_shift(case) + case_width(case) > RV64_REGISTER_NUM_LIMBS
}

/// LoadSignExtend Core Chip handles byte/halfword/word into doubleword conversions through
/// sign extend, at any byte shift inside the containing 8-byte block, including accesses
/// spanning two adjacent blocks (read_data ++ read_data1). Each `(opcode, shift)` pair is a
/// separate boolean flag. prev_data columns are not used in constraints defined in the
/// CoreAir, but are used in constraints by the Adapter.
#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow, StructReflection)]
pub struct LoadSignExtendCoreCols<T, const NUM_CELLS: usize> {
    /// One boolean flag per `(opcode, shift)` case.
    pub flags: [T; LOAD_SIGN_EXTEND_CASES],
    /// The bit that is extended to the remaining bits
    pub data_most_sig_bit: T,

    pub read_data: [T; NUM_CELLS],
    /// Second block for block-spanning loads; zero otherwise.
    pub read_data1: [T; NUM_CELLS],
    pub prev_data: [T; NUM_CELLS],
}

#[derive(Debug, Clone, ColumnsAir)]
#[columns_via(LoadSignExtendCoreCols<u8, NUM_CELLS>)]
pub struct LoadSignExtendCoreAir<const NUM_CELLS: usize, const LIMB_BITS: usize> {
    pub range_bus: VariableRangeCheckerBus,
    pub bitwise_lookup_bus: BitwiseOperationLookupBus,
}

impl<const NUM_CELLS: usize, const LIMB_BITS: usize> LoadSignExtendCoreAir<NUM_CELLS, LIMB_BITS> {
    pub fn new(
        range_bus: VariableRangeCheckerBus,
        bitwise_lookup_bus: BitwiseOperationLookupBus,
    ) -> Self {
        assert!(NUM_CELLS.is_multiple_of(2));
        Self {
            range_bus,
            bitwise_lookup_bus,
        }
    }
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
    I::Reads: From<(
        ([AB::Var; NUM_CELLS], [AB::Expr; NUM_CELLS]),
        ([AB::Expr; NUM_CELLS], [AB::Expr; NUM_CELLS]),
    )>,
    I::Writes: From<[[AB::Expr; NUM_CELLS]; 2]>,
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
            flags,
            data_most_sig_bit,
            read_data,
            read_data1,
            prev_data,
        } = *cols;

        let is_valid = flags.iter().fold(AB::Expr::ZERO, |acc, &flag| {
            builder.assert_bool(flag);
            acc + flag
        });
        builder.assert_bool(is_valid.clone());
        builder.assert_bool(data_most_sig_bit);

        // Selects the byte at global position `k` in read_data ++ read_data1.
        let src = |k: usize| -> AB::Expr {
            if k < NUM_CELLS {
                read_data[k].into()
            } else {
                read_data1[k - NUM_CELLS].into()
            }
        };

        let load_cross = (0..LOAD_SIGN_EXTEND_CASES).fold(AB::Expr::ZERO, |acc, case| {
            if case_crosses(case) {
                acc + flags[case]
            } else {
                acc
            }
        });

        let expected_opcode = (0..LOAD_SIGN_EXTEND_CASES).fold(AB::Expr::ZERO, |acc, case| {
            acc + flags[case] * AB::Expr::from_u8(case_opcode(case) as u8)
        }) + AB::Expr::from_usize(Rv64LoadStoreOpcode::CLASS_OFFSET);

        let load_shift_amount = (0..LOAD_SIGN_EXTEND_CASES).fold(AB::Expr::ZERO, |acc, case| {
            acc + flags[case] * AB::Expr::from_usize(case_shift(case))
        });

        // Constrain that data_most_sig_bit is the most significant bit of the most
        // significant loaded byte.
        let most_sig_limb = (0..LOAD_SIGN_EXTEND_CASES).fold(AB::Expr::ZERO, |acc, case| {
            acc + flags[case] * src(case_shift(case) + case_width(case) - 1)
        });
        self.range_bus
            .range_check(
                most_sig_limb - data_most_sig_bit * AB::Expr::from_u32(1 << (LIMB_BITS - 1)),
                LIMB_BITS - 1,
            )
            .eval(builder, is_valid.clone());

        for pair in read_data.chunks_exact(2) {
            self.bitwise_lookup_bus
                .send_range(pair[0], pair[1])
                .eval(builder, is_valid.clone());
        }
        // The second block is only live on block-spanning rows; range check it exactly there.
        for pair in read_data1.chunks_exact(2) {
            self.bitwise_lookup_bus
                .send_range(pair[0], pair[1])
                .eval(builder, load_cross.clone());
        }

        // write_data: the `width` loaded bytes followed by the extension byte.
        let limb_mask = data_most_sig_bit * AB::Expr::from_u32((1 << LIMB_BITS) - 1);
        let write_data: [AB::Expr; NUM_CELLS] = array::from_fn(|i| {
            (0..LOAD_SIGN_EXTEND_CASES).fold(AB::Expr::ZERO, |acc, case| {
                let term = if i < case_width(case) {
                    src(case_shift(case) + i)
                } else {
                    limb_mask.clone()
                };
                acc + flags[case] * term
            })
        });

        AdapterAirContext {
            to_pc: None,
            // Sign-extension loads never write a second block; its slots are unused zeros.
            reads: (
                (prev_data, array::from_fn(|_| AB::Expr::ZERO)),
                (read_data.map(|x| x.into()), read_data1.map(|x| x.into())),
            )
                .into(),
            writes: [write_data, array::from_fn(|_| AB::Expr::ZERO)].into(),
            instruction: LoadStoreInstruction {
                is_valid: is_valid.clone(),
                opcode: expected_opcode,
                is_load: is_valid,
                load_shift_amount,
                store_shift_amount: AB::Expr::ZERO,
                load_cross,
                store_cross: AB::Expr::ZERO,
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
    /// Second block contents for block-spanning loads; zero otherwise.
    pub read_data1: [u8; NUM_CELLS],
    pub prev_data: [u8; NUM_CELLS],
}

#[derive(Clone, Copy, derive_new::new)]
pub struct LoadSignExtendExecutor<A, const NUM_CELLS: usize, const LIMB_BITS: usize> {
    adapter: A,
}

#[derive(Clone)]
pub struct LoadSignExtendFiller<
    A = Rv64LoadStoreAdapterFiller,
    const NUM_CELLS: usize = RV64_REGISTER_NUM_LIMBS,
    const LIMB_BITS: usize = RV64_BYTE_BITS,
> {
    adapter: A,
    pub range_checker_chip: SharedVariableRangeCheckerChip,
    pub bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV64_BYTE_BITS>,
}

impl<A, const NUM_CELLS: usize, const LIMB_BITS: usize>
    LoadSignExtendFiller<A, NUM_CELLS, LIMB_BITS>
{
    pub fn new(
        adapter: A,
        range_checker_chip: SharedVariableRangeCheckerChip,
        bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV64_BYTE_BITS>,
    ) -> Self {
        assert!(NUM_CELLS.is_multiple_of(2));
        Self {
            adapter,
            range_checker_chip,
            bitwise_lookup_chip,
        }
    }
}

impl<F, A, RA, const NUM_CELLS: usize, const LIMB_BITS: usize> PreflightExecutor<F, RA>
    for LoadSignExtendExecutor<A, NUM_CELLS, LIMB_BITS>
where
    F: PrimeField32,
    A: 'static
        + AdapterTraceExecutor<
            F,
            ReadData = (([[u8; NUM_CELLS]; 2], [[u8; NUM_CELLS]; 2]), u8),
            WriteData = [[u8; NUM_CELLS]; 2],
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
        let Instruction { opcode, d, e, .. } = instruction;

        let local_opcode = Rv64LoadStoreOpcode::from_usize(
            opcode.local_opcode_idx(Rv64LoadStoreOpcode::CLASS_OFFSET),
        );
        debug_assert_eq!(d.as_canonical_u32(), RV64_REGISTER_AS);
        debug_assert_eq!(e.as_canonical_u32(), RV64_MEMORY_AS);

        let (mut adapter_record, core_record) = state.ctx.alloc(EmptyAdapterCoreLayout::new());

        A::start(*state.pc, state.memory, &mut adapter_record);

        let (([prev_data, _prev_data1], [read_data, read_data1]), shift_amount) = self
            .adapter
            .read(state.memory, instruction, &mut adapter_record);

        core_record.is_byte = local_opcode == LOADB;
        core_record.is_word = local_opcode == LOADW;
        core_record.prev_data = prev_data;
        core_record.read_data = read_data;
        core_record.read_data1 = read_data1;
        core_record.shift_amount = shift_amount;

        let write_data = run_write_data_sign_extend::<NUM_CELLS, LIMB_BITS>(
            local_opcode,
            core_record.read_data,
            core_record.read_data1,
            core_record.shift_amount as usize,
        );

        self.adapter.write(
            state.memory,
            instruction,
            [write_data, [0; NUM_CELLS]],
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

        // Copy the record fields before writing columns to the shared row buffer.
        let opcode = if record.is_byte {
            LOADB
        } else if record.is_word {
            LOADW
        } else {
            LOADH
        };
        let shift = record.shift_amount as usize;
        let read_data = record.read_data;
        let read_data1 = record.read_data1;
        let prev_data = record.prev_data;

        let core_row: &mut LoadSignExtendCoreCols<F, NUM_CELLS> = core_row.borrow_mut();

        let case = case_index(opcode, shift);
        let width = case_width(case);
        let crosses = shift + width > NUM_CELLS;

        let src = |k: usize| -> u8 {
            if k < NUM_CELLS {
                read_data[k]
            } else {
                read_data1[k - NUM_CELLS]
            }
        };
        let most_sig_limb = src(shift + width - 1);
        let most_sig_bit = most_sig_limb & (1 << (LIMB_BITS - 1));
        self.range_checker_chip
            .add_count((most_sig_limb - most_sig_bit) as u32, LIMB_BITS - 1);

        for pair in read_data.chunks_exact(2) {
            self.bitwise_lookup_chip
                .request_range(pair[0] as u32, pair[1] as u32);
        }
        if crosses {
            for pair in read_data1.chunks_exact(2) {
                self.bitwise_lookup_chip
                    .request_range(pair[0] as u32, pair[1] as u32);
            }
        }

        core_row.prev_data = prev_data.map(F::from_u8);
        core_row.read_data1 = read_data1.map(F::from_u8);
        core_row.read_data = read_data.map(F::from_u8);
        core_row.data_most_sig_bit = F::from_bool(most_sig_bit != 0);
        core_row.flags = array::from_fn(|i| F::from_bool(i == case));
    }
}

// Returns write_data
#[inline(always)]
pub(super) fn run_write_data_sign_extend<const NUM_CELLS: usize, const LIMB_BITS: usize>(
    opcode: Rv64LoadStoreOpcode,
    read_data: [u8; NUM_CELLS],
    read_data1: [u8; NUM_CELLS],
    shift: usize,
) -> [u8; NUM_CELLS] {
    const { assert!(NUM_CELLS == RV64_REGISTER_NUM_LIMBS) };
    debug_assert!(shift < NUM_CELLS);
    let width = 1 << sign_extend_op_idx(opcode);

    let src = |k: usize| -> u8 {
        if k < NUM_CELLS {
            read_data[k]
        } else {
            read_data1[k - NUM_CELLS]
        }
    };
    let ext = (src(shift + width - 1) >> (LIMB_BITS - 1)) * u8::MAX;
    array::from_fn(|i| if i < width { src(shift + i) } else { ext })
}
