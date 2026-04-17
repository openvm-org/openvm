use std::{
    array,
    borrow::{Borrow, BorrowMut},
    fmt::Debug,
};

use openvm_circuit::{
    arch::*,
    system::memory::{online::TracingMemory, MemoryAuxColsFactory},
};
use openvm_circuit_primitives::{encoder::Encoder, AlignedBorrow, AlignedBytesBorrow, SubAir};
use openvm_instructions::{
    instruction::Instruction, program::DEFAULT_PC_STEP, riscv::RV64_REGISTER_NUM_LIMBS, LocalOpcode,
};
use openvm_riscv_transpiler::Rv64LoadStoreOpcode::{self, *};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, FieldAlgebra, PrimeField32},
    rap::BaseAirWithPublicValues,
};

use crate::adapters::{LoadStoreInstruction, Rv64LoadStoreAdapterFiller};

const LOADSTORE_SELECTOR_CASES: usize = 30;
const LOADSTORE_SELECTOR_MAX_DEGREE: u32 = 2;
pub(crate) const LOADSTORE_SELECTOR_WIDTH: usize = 7;

#[derive(Debug, Clone, Copy)]
enum InstructionCase {
    LoadD0,
    LoadWu0,
    LoadWu4,
    LoadHu0,
    LoadHu2,
    LoadHu4,
    LoadHu6,
    LoadBu0,
    LoadBu1,
    LoadBu2,
    LoadBu3,
    LoadBu4,
    LoadBu5,
    LoadBu6,
    LoadBu7,
    StoreD0,
    StoreW0,
    StoreW4,
    StoreH0,
    StoreH2,
    StoreH4,
    StoreH6,
    StoreB0,
    StoreB1,
    StoreB2,
    StoreB3,
    StoreB4,
    StoreB5,
    StoreB6,
    StoreB7,
}

use InstructionCase::*;

impl InstructionCase {
    const ALL: [Self; LOADSTORE_SELECTOR_CASES] = [
        LoadD0, LoadWu0, LoadWu4, LoadHu0, LoadHu2, LoadHu4, LoadHu6, LoadBu0, LoadBu1, LoadBu2,
        LoadBu3, LoadBu4, LoadBu5, LoadBu6, LoadBu7, StoreD0, StoreW0, StoreW4, StoreH0, StoreH2,
        StoreH4, StoreH6, StoreB0, StoreB1, StoreB2, StoreB3, StoreB4, StoreB5, StoreB6, StoreB7,
    ];

    fn opcode(self) -> Rv64LoadStoreOpcode {
        match self {
            LoadD0 => LOADD,
            LoadWu0 | LoadWu4 => LOADWU,
            LoadHu0 | LoadHu2 | LoadHu4 | LoadHu6 => LOADHU,
            LoadBu0 | LoadBu1 | LoadBu2 | LoadBu3 | LoadBu4 | LoadBu5 | LoadBu6 | LoadBu7 => LOADBU,
            StoreD0 => STORED,
            StoreW0 | StoreW4 => STOREW,
            StoreH0 | StoreH2 | StoreH4 | StoreH6 => STOREH,
            StoreB0 | StoreB1 | StoreB2 | StoreB3 | StoreB4 | StoreB5 | StoreB6 | StoreB7 => STOREB,
        }
    }

    fn shift(self) -> usize {
        match self {
            LoadD0 | StoreD0 => 0,
            LoadWu0 | StoreW0 | LoadHu0 | StoreH0 | LoadBu0 | StoreB0 => 0,
            LoadBu1 | StoreB1 => 1,
            LoadHu2 | StoreH2 | LoadBu2 | StoreB2 => 2,
            LoadBu3 | StoreB3 => 3,
            LoadWu4 | StoreW4 | LoadHu4 | StoreH4 | LoadBu4 | StoreB4 => 4,
            LoadBu5 | StoreB5 => 5,
            LoadHu6 | StoreH6 | LoadBu6 | StoreB6 => 6,
            LoadBu7 | StoreB7 => 7,
        }
    }

    fn is_load(self) -> bool {
        matches!(
            self,
            LoadD0
                | LoadWu0
                | LoadWu4
                | LoadHu0
                | LoadHu2
                | LoadHu4
                | LoadHu6
                | LoadBu0
                | LoadBu1
                | LoadBu2
                | LoadBu3
                | LoadBu4
                | LoadBu5
                | LoadBu6
                | LoadBu7
        )
    }

    fn width(self) -> usize {
        match self.opcode() {
            LOADD | STORED => 8,
            LOADWU | STOREW => 4,
            LOADHU | STOREH => 2,
            LOADBU | STOREB => 1,
            _ => unreachable!("loadstore core should not handle sign-extension loads"),
        }
    }

    fn from_opcode_shift(opcode: Rv64LoadStoreOpcode, shift: usize) -> Self {
        match (opcode, shift) {
            (LOADD, 0) => LoadD0,
            (LOADWU, 0) => LoadWu0,
            (LOADWU, 4) => LoadWu4,
            (LOADHU, 0) => LoadHu0,
            (LOADHU, 2) => LoadHu2,
            (LOADHU, 4) => LoadHu4,
            (LOADHU, 6) => LoadHu6,
            (LOADBU, 0) => LoadBu0,
            (LOADBU, 1) => LoadBu1,
            (LOADBU, 2) => LoadBu2,
            (LOADBU, 3) => LoadBu3,
            (LOADBU, 4) => LoadBu4,
            (LOADBU, 5) => LoadBu5,
            (LOADBU, 6) => LoadBu6,
            (LOADBU, 7) => LoadBu7,
            (STORED, 0) => StoreD0,
            (STOREW, 0) => StoreW0,
            (STOREW, 4) => StoreW4,
            (STOREH, 0) => StoreH0,
            (STOREH, 2) => StoreH2,
            (STOREH, 4) => StoreH4,
            (STOREH, 6) => StoreH6,
            (STOREB, 0) => StoreB0,
            (STOREB, 1) => StoreB1,
            (STOREB, 2) => StoreB2,
            (STOREB, 3) => StoreB3,
            (STOREB, 4) => StoreB4,
            (STOREB, 5) => StoreB5,
            (STOREB, 6) => StoreB6,
            (STOREB, 7) => StoreB7,
            _ => unreachable!(
                "unaligned memory access not supported by this execution environment: {opcode:?}, shift: {shift}"
            ),
        }
    }
}

fn loadstore_encoder() -> Encoder {
    let encoder = Encoder::new(
        LOADSTORE_SELECTOR_CASES,
        LOADSTORE_SELECTOR_MAX_DEGREE,
        true,
    );
    debug_assert_eq!(encoder.width(), LOADSTORE_SELECTOR_WIDTH);
    encoder
}

pub(crate) fn selector_point_for_opcode_shift(
    opcode: Rv64LoadStoreOpcode,
    shift: usize,
) -> [u32; LOADSTORE_SELECTOR_WIDTH] {
    loadstore_encoder()
        .get_flag_pt(InstructionCase::from_opcode_shift(opcode, shift) as usize)
        .try_into()
        .unwrap()
}

/// LoadStore Core Chip handles byte/halfword/word into doubleword conversions and unsigned
/// extends. This chip uses read_data and prev_data to constrain the write_data. It also handles
/// the shifting in case of not 8 byte aligned instructions. This chip treats each `(opcode,
/// shift)` pair as a separate instruction.
#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow)]
pub struct LoadStoreCoreCols<T, const NUM_CELLS: usize> {
    pub selector: [T; LOADSTORE_SELECTOR_WIDTH],
    /// we need to keep the degree of is_valid and is_load to 1
    pub is_valid: T,
    pub is_load: T,

    pub read_data: [T; NUM_CELLS],
    pub prev_data: [T; NUM_CELLS],
    /// write_data will be constrained against read_data and prev_data
    /// depending on the opcode and the shift amount
    pub write_data: [T; NUM_CELLS],
}

#[derive(Debug, Clone)]
pub struct LoadStoreCoreAir<const NUM_CELLS: usize> {
    pub offset: usize,
    encoder: Encoder,
}

impl<const NUM_CELLS: usize> LoadStoreCoreAir<NUM_CELLS> {
    pub fn new(offset: usize) -> Self {
        Self {
            offset,
            encoder: loadstore_encoder(),
        }
    }
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
        let cols: &LoadStoreCoreCols<AB::Var, NUM_CELLS> = (*local_core).borrow();
        let LoadStoreCoreCols::<AB::Var, NUM_CELLS> {
            selector,
            is_valid,
            is_load,
            read_data,
            prev_data,
            write_data,
        } = *cols;

        self.encoder.eval(builder, &selector);

        let selector_flags = self.encoder.flags::<AB>(&selector);
        let expected_is_valid = self.encoder.is_valid::<AB>(&selector);
        let expected_is_load = InstructionCase::ALL
            .iter()
            .fold(AB::Expr::ZERO, |acc, &case| {
                if case.is_load() {
                    acc + selector_flags[case as usize].clone()
                } else {
                    acc
                }
            });

        builder.assert_eq(is_valid, expected_is_valid.clone());
        builder.assert_eq(is_load, expected_is_load.clone());

        let expected_opcode = InstructionCase::ALL
            .iter()
            .fold(AB::Expr::ZERO, |acc, &case| {
                acc + selector_flags[case as usize].clone()
                    * AB::Expr::from_canonical_u8(case.opcode() as u8)
            });
        let expected_opcode = VmCoreAir::<AB, I>::expr_to_global_expr(self, expected_opcode);

        let load_shift_amount = InstructionCase::ALL
            .iter()
            .fold(AB::Expr::ZERO, |acc, &case| {
                if case.is_load() {
                    acc + selector_flags[case as usize].clone()
                        * AB::Expr::from_canonical_usize(case.shift())
                } else {
                    acc
                }
            });
        let store_shift_amount = InstructionCase::ALL
            .iter()
            .fold(AB::Expr::ZERO, |acc, &case| {
                if case.is_load() {
                    acc
                } else {
                    acc + selector_flags[case as usize].clone()
                        * AB::Expr::from_canonical_usize(case.shift())
                }
            });

        for (i, cell) in write_data.iter().enumerate() {
            let expected = InstructionCase::ALL
                .iter()
                .fold(AB::Expr::ZERO, |acc, &case| {
                    let width = case.width();
                    let shift = case.shift();
                    debug_assert!(shift + width <= NUM_CELLS);
                    let term = if case.is_load() {
                        if i < width {
                            read_data[i + shift].into()
                        } else {
                            AB::Expr::ZERO
                        }
                    } else if i >= shift && i < shift + width {
                        read_data[i - shift].into()
                    } else {
                        prev_data[i].into()
                    };
                    acc + selector_flags[case as usize].clone() * term
                });
            builder.assert_eq(*cell, expected);
        }

        AdapterAirContext {
            to_pc: None,
            reads: (prev_data, read_data.map(|x| x.into())).into(),
            writes: [write_data.map(|x| x.into())].into(),
            instruction: LoadStoreInstruction {
                is_valid: is_valid.into(),
                opcode: expected_opcode,
                is_load: is_load.into(),
                load_shift_amount,
                store_shift_amount,
            }
            .into(),
        }
    }

    fn start_offset(&self) -> usize {
        self.offset
    }
}

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug)]
pub struct LoadStoreCoreRecord<const NUM_CELLS: usize> {
    pub local_opcode: u8,
    pub shift_amount: u8,
    pub read_data: [u8; NUM_CELLS],
    // Note: `prev_data` can be from native address space, so we need to use u32.
    pub prev_data: [u32; NUM_CELLS],
}

#[derive(Clone, Copy, derive_new::new)]
pub struct LoadStoreExecutor<A, const NUM_CELLS: usize> {
    adapter: A,
    pub offset: usize,
}

#[derive(Clone)]
pub struct LoadStoreFiller<
    A = Rv64LoadStoreAdapterFiller,
    const NUM_CELLS: usize = RV64_REGISTER_NUM_LIMBS,
> {
    adapter: A,
    pub offset: usize,
    encoder: Encoder,
}

impl<A, const NUM_CELLS: usize> LoadStoreFiller<A, NUM_CELLS> {
    pub fn new(adapter: A, offset: usize) -> Self {
        Self {
            adapter,
            offset,
            encoder: loadstore_encoder(),
        }
    }
}

impl<F, A, RA, const NUM_CELLS: usize> PreflightExecutor<F, RA> for LoadStoreExecutor<A, NUM_CELLS>
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
        (A::RecordMut<'buf>, &'buf mut LoadStoreCoreRecord<NUM_CELLS>),
    >,
{
    fn get_opcode_name(&self, opcode: usize) -> String {
        format!(
            "{:?}",
            Rv64LoadStoreOpcode::from_usize(opcode - self.offset)
        )
    }

    fn execute(
        &self,
        state: VmStateMut<F, TracingMemory, RA>,
        instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError> {
        let Instruction { opcode, .. } = instruction;

        let (mut adapter_record, core_record) = state.ctx.alloc(EmptyAdapterCoreLayout::new());

        A::start(*state.pc, state.memory, &mut adapter_record);

        (
            (core_record.prev_data, core_record.read_data),
            core_record.shift_amount,
        ) = self
            .adapter
            .read(state.memory, instruction, &mut adapter_record);

        let local_opcode = Rv64LoadStoreOpcode::from_usize(opcode.local_opcode_idx(self.offset));
        core_record.local_opcode = local_opcode as u8;

        let write_data = run_write_data(
            local_opcode,
            core_record.read_data,
            core_record.prev_data,
            core_record.shift_amount as usize,
        );
        self.adapter
            .write(state.memory, instruction, write_data, &mut adapter_record);

        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);

        Ok(())
    }
}

impl<F, A, const NUM_CELLS: usize> TraceFiller<F> for LoadStoreFiller<A, NUM_CELLS>
where
    F: PrimeField32,
    A: 'static + AdapterTraceFiller<F>,
{
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        // SAFETY: row_slice is guaranteed by the caller to have at least A::WIDTH +
        // LoadStoreCoreCols::width() elements
        let (adapter_row, mut core_row) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };
        self.adapter.fill_trace_row(mem_helper, adapter_row);
        // SAFETY: core_row contains a valid LoadStoreCoreRecord written by the executor
        // during trace generation
        let record: &LoadStoreCoreRecord<NUM_CELLS> =
            unsafe { get_record_from_slice(&mut core_row, ()) };
        let core_row: &mut LoadStoreCoreCols<F, NUM_CELLS> = core_row.borrow_mut();

        let opcode = Rv64LoadStoreOpcode::from_usize(record.local_opcode as usize);
        let shift = record.shift_amount as usize;
        let write_data = run_write_data(opcode, record.read_data, record.prev_data, shift);

        core_row.write_data = write_data.map(F::from_canonical_u32);
        core_row.prev_data = record.prev_data.map(F::from_canonical_u32);
        core_row.read_data = record.read_data.map(F::from_canonical_u8);
        core_row.is_load = F::from_bool(matches!(opcode, LOADD | LOADWU | LOADHU | LOADBU));
        core_row.is_valid = F::ONE;
        let pt: [u32; LOADSTORE_SELECTOR_WIDTH] = self
            .encoder
            .get_flag_pt(InstructionCase::from_opcode_shift(opcode, shift) as usize)
            .try_into()
            .unwrap();
        core_row.selector = pt.map(F::from_canonical_u32);
    }
}

// Returns the write data
#[inline(always)]
pub(super) fn run_write_data<const NUM_CELLS: usize>(
    opcode: Rv64LoadStoreOpcode,
    read_data: [u8; NUM_CELLS],
    prev_data: [u32; NUM_CELLS],
    shift: usize,
) -> [u32; NUM_CELLS] {
    debug_assert_eq!(NUM_CELLS, RV64_REGISTER_NUM_LIMBS);
    let word_width = NUM_CELLS / 2;
    let half_width = NUM_CELLS / 4;

    match opcode {
        LOADD if shift == 0 => read_data.map(u32::from),
        LOADWU if shift == 0 || shift == word_width => array::from_fn(|i| {
            if i < word_width {
                read_data[i + shift] as u32
            } else {
                0
            }
        }),
        LOADHU if [0, half_width, word_width, word_width + half_width].contains(&shift) => {
            array::from_fn(|i| {
                if i < half_width {
                    read_data[i + shift] as u32
                } else {
                    0
                }
            })
        }
        LOADBU if shift < NUM_CELLS => array::from_fn(|i| {
            if i == 0 {
                read_data[shift] as u32
            } else {
                0
            }
        }),
        STORED if shift == 0 => read_data.map(u32::from),
        STOREW if shift == 0 || shift == word_width => array::from_fn(|i| {
            if i >= shift && i < shift + word_width {
                read_data[i - shift] as u32
            } else {
                prev_data[i]
            }
        }),
        STOREH if [0, half_width, word_width, word_width + half_width].contains(&shift) => {
            array::from_fn(|i| {
                if i >= shift && i < shift + half_width {
                    read_data[i - shift] as u32
                } else {
                    prev_data[i]
                }
            })
        }
        STOREB if shift < NUM_CELLS => {
            let mut write_data = prev_data;
            write_data[shift] = read_data[0] as u32;
            write_data
        }
        _ => unreachable!(
            "unaligned memory access not supported by this execution environment: {opcode:?}, shift: {shift}"
        ),
    }
}
