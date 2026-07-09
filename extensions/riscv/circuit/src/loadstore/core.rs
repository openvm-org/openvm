use std::{
    array,
    borrow::{Borrow, BorrowMut},
    fmt::Debug,
};

use openvm_circuit::{
    arch::*,
    system::memory::{online::TracingMemory, MemoryAuxColsFactory},
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::{BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip},
    encoder::Encoder,
    AlignedBorrow, AlignedBytesBorrow, ColumnsAir, StructReflection, StructReflectionHelper,
    SubAir,
};
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV64_MEMORY_AS, RV64_REGISTER_AS, RV64_REGISTER_NUM_LIMBS},
    LocalOpcode, PUBLIC_VALUES_AS,
};
use openvm_riscv_transpiler::Rv64LoadStoreOpcode::{self, *};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, PrimeCharacteristicRing, PrimeField32},
    BaseAirWithPublicValues,
};

use crate::adapters::{LoadStoreInstruction, Rv64LoadStoreAdapterFiller};

/// Every `(opcode, shift)` pair is a selector case: 8 opcodes (LOADD..STOREB, transpiler
/// order 0..=7) times 8 byte shifts. Case index = `opcode as usize * 8 + shift`.
const LOADSTORE_SELECTOR_CASES: usize = 64;
const LOADSTORE_SELECTOR_MAX_DEGREE: u32 = 2;
pub(crate) const LOADSTORE_SELECTOR_WIDTH: usize = 10;

const NUM_LOADSTORE_OPCODES: usize = 8;

#[inline(always)]
fn case_index(opcode: Rv64LoadStoreOpcode, shift: usize) -> usize {
    debug_assert!((opcode as usize) < NUM_LOADSTORE_OPCODES);
    debug_assert!(shift < RV64_REGISTER_NUM_LIMBS);
    opcode as usize * RV64_REGISTER_NUM_LIMBS + shift
}

#[inline(always)]
fn case_opcode(case: usize) -> Rv64LoadStoreOpcode {
    Rv64LoadStoreOpcode::from_usize(case / RV64_REGISTER_NUM_LIMBS)
}

#[inline(always)]
fn case_shift(case: usize) -> usize {
    case % RV64_REGISTER_NUM_LIMBS
}

#[inline(always)]
fn opcode_width(opcode: Rv64LoadStoreOpcode) -> usize {
    match opcode {
        LOADD | STORED => 8,
        LOADWU | STOREW => 4,
        LOADHU | STOREH => 2,
        LOADBU | STOREB => 1,
        _ => unreachable!("loadstore core should not handle sign-extension loads"),
    }
}

#[inline(always)]
fn opcode_is_load(opcode: Rv64LoadStoreOpcode) -> bool {
    matches!(opcode, LOADD | LOADWU | LOADHU | LOADBU)
}

/// Whether the access at this shift spans two adjacent 8-byte blocks.
#[inline(always)]
fn case_crosses(case: usize) -> bool {
    case_shift(case) + opcode_width(case_opcode(case)) > RV64_REGISTER_NUM_LIMBS
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

#[cfg(test)]
pub(crate) fn selector_point_for_opcode_shift(
    opcode: Rv64LoadStoreOpcode,
    shift: usize,
) -> [u32; LOADSTORE_SELECTOR_WIDTH] {
    loadstore_encoder()
        .get_flag_pt(case_index(opcode, shift))
        .try_into()
        .unwrap()
}

/// LoadStore Core Chip handles byte/halfword/word/doubleword loads and stores at any byte
/// shift inside the containing 8-byte block, including accesses that span two adjacent
/// blocks. Loads select their bytes from the concatenation `read_data ++ read_data1` and
/// zero-extend into `write_data`. Stores merge the source register bytes (`read_data`) into
/// the previous contents of the touched block(s) (`prev_data`, `prev_data1`), producing
/// `write_data` and (for block-spanning stores) `write_data1`. This chip treats each
/// `(opcode, shift)` pair as a separate instruction.
#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow, StructReflection)]
pub struct LoadStoreCoreCols<T, const NUM_CELLS: usize> {
    pub selector: [T; LOADSTORE_SELECTOR_WIDTH],
    /// we need to keep the degree of is_valid, is_load and the cross flags to 1
    pub is_valid: T,
    pub is_load: T,
    /// 1 iff this is a load spanning two blocks; multiplicity of the second block read.
    pub load_cross: T,
    /// 1 iff this is a store spanning two blocks; multiplicity of the second block write.
    pub store_cross: T,

    /// First (always accessed) block for loads; source register value for stores.
    pub read_data: [T; NUM_CELLS],
    /// Second block for block-spanning loads; zero otherwise.
    pub read_data1: [T; NUM_CELLS],
    /// Previous rd value for loads; previous contents of the first block for stores.
    pub prev_data: [T; NUM_CELLS],
    /// Previous contents of the second block for block-spanning stores; zero otherwise.
    pub prev_data1: [T; NUM_CELLS],
    /// write_data (and write_data1 for block-spanning stores) will be constrained against
    /// read_data, read_data1, prev_data and prev_data1 depending on the opcode and shift
    pub write_data: [T; NUM_CELLS],
    pub write_data1: [T; NUM_CELLS],
}

#[derive(Debug, Clone, ColumnsAir)]
#[columns_via(LoadStoreCoreCols<u8, NUM_CELLS>)]
pub struct LoadStoreCoreAir<const NUM_CELLS: usize> {
    pub offset: usize,
    encoder: Encoder,
    bitwise_lookup_bus: BitwiseOperationLookupBus,
}

impl<const NUM_CELLS: usize> LoadStoreCoreAir<NUM_CELLS> {
    pub fn new(offset: usize, bitwise_lookup_bus: BitwiseOperationLookupBus) -> Self {
        assert!(NUM_CELLS.is_multiple_of(2));
        Self {
            offset,
            encoder: loadstore_encoder(),
            bitwise_lookup_bus,
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
        let cols: &LoadStoreCoreCols<AB::Var, NUM_CELLS> = (*local_core).borrow();
        let LoadStoreCoreCols::<AB::Var, NUM_CELLS> {
            selector,
            is_valid,
            is_load,
            load_cross,
            store_cross,
            read_data,
            read_data1,
            prev_data,
            prev_data1,
            write_data,
            write_data1,
        } = *cols;

        self.encoder.eval(builder, &selector);

        let selector_flags = self.encoder.flags::<AB>(&selector);
        let expected_is_valid = self.encoder.is_valid::<AB>(&selector);

        let sum_flags_where = |pred: &dyn Fn(usize) -> bool| {
            (0..LOADSTORE_SELECTOR_CASES).fold(AB::Expr::ZERO, |acc, case| {
                if pred(case) {
                    acc + selector_flags[case].clone()
                } else {
                    acc
                }
            })
        };

        let expected_is_load = sum_flags_where(&|case| opcode_is_load(case_opcode(case)));
        let expected_load_cross =
            sum_flags_where(&|case| opcode_is_load(case_opcode(case)) && case_crosses(case));
        let expected_store_cross =
            sum_flags_where(&|case| !opcode_is_load(case_opcode(case)) && case_crosses(case));

        builder.assert_eq(is_valid, expected_is_valid.clone());
        builder.assert_eq(is_load, expected_is_load.clone());
        builder.assert_eq(load_cross, expected_load_cross);
        builder.assert_eq(store_cross, expected_store_cross);

        for pair in read_data.chunks_exact(2) {
            self.bitwise_lookup_bus
                .send_range(pair[0], pair[1])
                .eval(builder, is_valid.into());
        }
        for pair in prev_data.chunks_exact(2) {
            self.bitwise_lookup_bus
                .send_range(pair[0], pair[1])
                .eval(builder, is_valid.into());
        }
        // The second-block witnesses are only live (and only reach a bus with nonzero
        // multiplicity) on block-spanning rows; range check them exactly there.
        for pair in read_data1.chunks_exact(2) {
            self.bitwise_lookup_bus
                .send_range(pair[0], pair[1])
                .eval(builder, load_cross.into());
        }
        for pair in prev_data1.chunks_exact(2) {
            self.bitwise_lookup_bus
                .send_range(pair[0], pair[1])
                .eval(builder, store_cross.into());
        }

        let expected_opcode = (0..LOADSTORE_SELECTOR_CASES).fold(AB::Expr::ZERO, |acc, case| {
            acc + selector_flags[case].clone() * AB::Expr::from_u8(case_opcode(case) as u8)
        });
        let expected_opcode = VmCoreAir::<AB, I>::expr_to_global_expr(self, expected_opcode);

        let load_shift_amount = (0..LOADSTORE_SELECTOR_CASES).fold(AB::Expr::ZERO, |acc, case| {
            if opcode_is_load(case_opcode(case)) {
                acc + selector_flags[case].clone() * AB::Expr::from_usize(case_shift(case))
            } else {
                acc
            }
        });
        let store_shift_amount = (0..LOADSTORE_SELECTOR_CASES).fold(AB::Expr::ZERO, |acc, case| {
            if opcode_is_load(case_opcode(case)) {
                acc
            } else {
                acc + selector_flags[case].clone() * AB::Expr::from_usize(case_shift(case))
            }
        });

        // Loads select `width` bytes starting at `shift` from read_data ++ read_data1 and
        // zero-extend. Stores merge the low `width` source bytes into the previous block
        // contents at byte positions [shift, shift + width), spilling into the second block
        // when shift + width > NUM_CELLS.
        for (i, cell) in write_data.iter().enumerate() {
            let expected = (0..LOADSTORE_SELECTOR_CASES).fold(AB::Expr::ZERO, |acc, case| {
                let opcode = case_opcode(case);
                let width = opcode_width(opcode);
                let shift = case_shift(case);
                let term = if opcode_is_load(opcode) {
                    if i < width {
                        let src = i + shift;
                        if src < NUM_CELLS {
                            read_data[src].into()
                        } else {
                            read_data1[src - NUM_CELLS].into()
                        }
                    } else {
                        AB::Expr::ZERO
                    }
                } else if i >= shift && i < shift + width {
                    read_data[i - shift].into()
                } else {
                    prev_data[i].into()
                };
                acc + selector_flags[case].clone() * term
            });
            builder.assert_eq(*cell, expected);
        }
        for (i, cell) in write_data1.iter().enumerate() {
            let expected = (0..LOADSTORE_SELECTOR_CASES).fold(AB::Expr::ZERO, |acc, case| {
                let opcode = case_opcode(case);
                let width = opcode_width(opcode);
                let shift = case_shift(case);
                // Only block-spanning stores write a second block; every other case forces
                // write_data1 to zero.
                let term = if !opcode_is_load(opcode) && case_crosses(case) {
                    let global = NUM_CELLS + i;
                    if global < shift + width {
                        read_data[global - shift].into()
                    } else {
                        prev_data1[i].into()
                    }
                } else {
                    AB::Expr::ZERO
                };
                acc + selector_flags[case].clone() * term
            });
            builder.assert_eq(*cell, expected);
        }

        AdapterAirContext {
            to_pc: None,
            reads: (
                (prev_data, prev_data1.map(|x| x.into())),
                (read_data.map(|x| x.into()), read_data1.map(|x| x.into())),
            )
                .into(),
            writes: [write_data.map(|x| x.into()), write_data1.map(|x| x.into())].into(),
            instruction: LoadStoreInstruction {
                is_valid: is_valid.into(),
                opcode: expected_opcode,
                is_load: is_load.into(),
                load_shift_amount,
                store_shift_amount,
                load_cross: load_cross.into(),
                store_cross: store_cross.into(),
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
    /// Second block contents for block-spanning loads; zero otherwise.
    pub read_data1: [u8; NUM_CELLS],
    pub prev_data: [u8; NUM_CELLS],
    /// Second block previous contents for block-spanning stores; zero otherwise.
    pub prev_data1: [u8; NUM_CELLS],
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
    bitwise_lookup_chip: SharedBitwiseOperationLookupChip<8>,
}

impl<A, const NUM_CELLS: usize> LoadStoreFiller<A, NUM_CELLS> {
    pub fn new(
        adapter: A,
        offset: usize,
        bitwise_lookup_chip: SharedBitwiseOperationLookupChip<8>,
    ) -> Self {
        assert!(NUM_CELLS.is_multiple_of(2));
        Self {
            adapter,
            offset,
            encoder: loadstore_encoder(),
            bitwise_lookup_chip,
        }
    }
}

impl<F, A, RA, const NUM_CELLS: usize> PreflightExecutor<F, RA> for LoadStoreExecutor<A, NUM_CELLS>
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
        let Instruction { opcode, d, e, .. } = instruction;
        let local_opcode = Rv64LoadStoreOpcode::from_usize(opcode.local_opcode_idx(self.offset));
        debug_assert_eq!(d.as_canonical_u32(), RV64_REGISTER_AS);
        debug_assert!(match local_opcode {
            LOADD | LOADWU | LOADHU | LOADBU => e.as_canonical_u32() == RV64_MEMORY_AS,
            STORED | STOREW | STOREH | STOREB =>
                e.as_canonical_u32() == RV64_MEMORY_AS || e.as_canonical_u32() == PUBLIC_VALUES_AS,
            _ => false,
        });

        let (mut adapter_record, core_record) = state.ctx.alloc(EmptyAdapterCoreLayout::new());

        A::start(*state.pc, state.memory, &mut adapter_record);

        let (([prev_data, prev_data1], [read_data, read_data1]), shift_amount) =
            self.adapter
                .read(state.memory, instruction, &mut adapter_record);
        core_record.prev_data = prev_data;
        core_record.prev_data1 = prev_data1;
        core_record.read_data = read_data;
        core_record.read_data1 = read_data1;
        core_record.shift_amount = shift_amount;

        core_record.local_opcode = local_opcode as u8;

        let write_data = run_write_data(
            local_opcode,
            core_record.read_data,
            core_record.read_data1,
            core_record.prev_data,
            core_record.prev_data1,
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
        let is_load = opcode_is_load(opcode);
        let crosses = shift + opcode_width(opcode) > NUM_CELLS;
        let [write_data, write_data1] = run_write_data(
            opcode,
            record.read_data,
            record.read_data1,
            record.prev_data,
            record.prev_data1,
            shift,
        );

        for pair in record.read_data.chunks_exact(2) {
            self.bitwise_lookup_chip
                .request_range(pair[0] as u32, pair[1] as u32);
        }
        for pair in record.prev_data.chunks_exact(2) {
            self.bitwise_lookup_chip
                .request_range(pair[0] as u32, pair[1] as u32);
        }
        if crosses {
            let checked_block1 = if is_load {
                &record.read_data1
            } else {
                &record.prev_data1
            };
            for pair in checked_block1.chunks_exact(2) {
                self.bitwise_lookup_chip
                    .request_range(pair[0] as u32, pair[1] as u32);
            }
        }

        core_row.write_data1 = write_data1.map(F::from_u8);
        core_row.write_data = write_data.map(F::from_u8);
        core_row.prev_data1 = record.prev_data1.map(F::from_u8);
        core_row.prev_data = record.prev_data.map(F::from_u8);
        core_row.read_data1 = record.read_data1.map(F::from_u8);
        core_row.read_data = record.read_data.map(F::from_u8);
        core_row.store_cross = F::from_bool(!is_load && crosses);
        core_row.load_cross = F::from_bool(is_load && crosses);
        core_row.is_load = F::from_bool(is_load);
        core_row.is_valid = F::ONE;
        let pt: [u32; LOADSTORE_SELECTOR_WIDTH] = self
            .encoder
            .get_flag_pt(case_index(opcode, shift))
            .try_into()
            .unwrap();
        core_row.selector = pt.map(F::from_u32);
    }
}

// Returns [write_data, write_data1]: the new rd value (loads, write_data1 unused) or the new
// contents of the touched block(s) (stores).
#[inline(always)]
pub(super) fn run_write_data<const NUM_CELLS: usize>(
    opcode: Rv64LoadStoreOpcode,
    read_data: [u8; NUM_CELLS],
    read_data1: [u8; NUM_CELLS],
    prev_data: [u8; NUM_CELLS],
    prev_data1: [u8; NUM_CELLS],
    shift: usize,
) -> [[u8; NUM_CELLS]; 2] {
    const { assert!(NUM_CELLS == RV64_REGISTER_NUM_LIMBS) };
    debug_assert!(shift < NUM_CELLS);
    let width = opcode_width(opcode);

    if opcode_is_load(opcode) {
        let write_data = array::from_fn(|i| {
            if i < width {
                let src = i + shift;
                if src < NUM_CELLS {
                    read_data[src]
                } else {
                    read_data1[src - NUM_CELLS]
                }
            } else {
                0
            }
        });
        [write_data, [0; NUM_CELLS]]
    } else {
        let write_data = array::from_fn(|i| {
            if i >= shift && i < shift + width {
                read_data[i - shift]
            } else {
                prev_data[i]
            }
        });
        let write_data1 = if shift + width > NUM_CELLS {
            array::from_fn(|i| {
                let global = NUM_CELLS + i;
                if global < shift + width {
                    read_data[global - shift]
                } else {
                    prev_data1[i]
                }
            })
        } else {
            [0; NUM_CELLS]
        };
        [write_data, write_data1]
    }
}
