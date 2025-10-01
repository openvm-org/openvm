use std::{
    array,
    borrow::{Borrow, BorrowMut},
    mem::size_of,
    sync::Arc,
};

use openvm_circuit::{
    arch::{
        get_record_from_slice, CustomBorrow, E2PreCompute, ExecuteFunc, ExecutionCtxTrait,
        ExecutionError, Executor, MeteredExecutionCtxTrait, MeteredExecutor, MultiRowLayout,
        MultiRowMetadata, PreflightExecutor, RecordArena, SizedRecord, StaticProgramError,
        TraceFiller, VmChipWrapper, VmExecState, VmStateMut,
    },
    system::memory::{
        offline_checker::{
            MemoryBaseAuxRecord, MemoryBridge, MemoryReadAuxCols, MemoryReadAuxRecord,
            MemoryWriteAuxCols, MemoryWriteBytesAuxRecord,
        },
        online::{GuestMemory, TracingMemory},
        MemoryAddress, MemoryAuxColsFactory, POINTER_MAX_BITS,
    },
};
use openvm_circuit_primitives::{
    utils::{and, not, or, select},
    var_range::{SharedVariableRangeCheckerChip, VariableRangeCheckerBus},
    AlignedBytesBorrow,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_MEMORY_AS, RV32_REGISTER_AS},
    LocalOpcode,
};
use openvm_memcpy_transpiler::Rv32MemcpyOpcode;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{Air, AirBuilder, BaseAir},
    p3_field::{Field, FieldAlgebra, PrimeField32},
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    p3_maybe_rayon::prelude::*,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};

use crate::{
    bus::MemcpyBus, read_rv32_register, tracing_read, tracing_write, MemcpyLoopChip,
    A1_REGISTER_PTR, A2_REGISTER_PTR, A3_REGISTER_PTR, A4_REGISTER_PTR,
};
// Import constants from lib.rs
use crate::{MEMCPY_LOOP_LIMB_BITS, MEMCPY_LOOP_NUM_LIMBS};

#[repr(C)]
#[derive(AlignedBorrow, Debug)]
pub struct MemcpyIterCols<T> {
    pub timestamp: T,
    pub dest: T,
    pub source: T,
    pub len: [T; 2],
    // 0: [0, 0, 0], 1: [1, 0, 0], 2: [0, 1, 0], 3: [0, 0, 1]
    pub shift: [T; 3],
    pub is_valid: T,
    pub is_valid_not_start: T,
    // This should be 0 if is_valid = 0. We use this to determine whether we need ro read data_4.
    pub is_shift_non_zero_or_not_start: T,
    // -1 for the first iteration, 1 for the last iteration, 0 for the middle iterations
    pub is_boundary: T,
    pub data_1: [T; MEMCPY_LOOP_NUM_LIMBS],
    pub data_2: [T; MEMCPY_LOOP_NUM_LIMBS],
    pub data_3: [T; MEMCPY_LOOP_NUM_LIMBS],
    pub data_4: [T; MEMCPY_LOOP_NUM_LIMBS],
    pub read_aux: [MemoryReadAuxCols<T>; 4],
    pub write_aux: [MemoryWriteAuxCols<T, MEMCPY_LOOP_NUM_LIMBS>; 4],
    // 1-hot encoding for source = 0, 4, 8
    pub is_source_0_4_8: [T; 3],
}

pub const NUM_MEMCPY_ITER_COLS: usize = size_of::<MemcpyIterCols<u8>>();

#[derive(Copy, Clone, Debug, derive_new::new)]
pub struct MemcpyIterAir {
    pub memory_bridge: MemoryBridge,
    pub range_bus: VariableRangeCheckerBus,
    pub memcpy_bus: MemcpyBus,
    pub pointer_max_bits: usize,
}

impl<F: Field> BaseAir<F> for MemcpyIterAir {
    fn width(&self) -> usize {
        MemcpyIterCols::<F>::width()
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for MemcpyIterAir {}
impl<F: Field> PartitionedBaseAir<F> for MemcpyIterAir {}

impl<AB: InteractionBuilder> Air<AB> for MemcpyIterAir {
    // assertions for AIR constraints
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (prev, local) = (main.row_slice(0), main.row_slice(1));
        let prev: &MemcpyIterCols<AB::Var> = (*prev).borrow();
        let local: &MemcpyIterCols<AB::Var> = (*local).borrow();

        let timestamp: AB::Var = local.timestamp;
        let mut timestamp_delta: AB::Expr = AB::Expr::ZERO;
        let mut timestamp_pp = |timestamp_increase_value: AB::Expr| {
            let timestamp_increase_clone = timestamp_increase_value.clone();
            timestamp_delta += timestamp_increase_value.into();
            timestamp + timestamp_delta.clone() - timestamp_increase_clone
        };

        let shift = local
            .shift
            .iter()
            .enumerate()
            .fold(AB::Expr::ZERO, |acc, (i, x)| {
                acc + (*x) * AB::Expr::from_canonical_u32(i as u32 + 1)
            });
        let is_shift_non_zero = local
            .shift
            .iter()
            .fold(AB::Expr::ZERO, |acc: <AB as AirBuilder>::Expr, x| {
                acc + (*x)
            });
        let is_shift_zero = not::<AB::Expr>(is_shift_non_zero.clone());
        let is_shift_one = local.shift[0];
        let is_shift_two = local.shift[1];
        let is_shift_three = local.shift[2];

        let is_end =
            (local.is_boundary + AB::Expr::ONE) * local.is_boundary * (AB::F::TWO).inverse();
        let is_not_start = (local.is_boundary + AB::Expr::ONE)
            * (AB::Expr::TWO - local.is_boundary)
            * (AB::F::TWO).inverse();

        let prev_is_not_end = not::<AB::Expr>(
            // returns 0 if prev.isBoundary == 1, 1 otherwise, since we take the not
            (prev.is_boundary + AB::Expr::ONE) * prev.is_boundary * (AB::F::TWO).inverse(),
        );

        let len = local.len[0]
            + local.len[1] * AB::Expr::from_canonical_u32(1 << (2 * MEMCPY_LOOP_LIMB_BITS));
        let prev_len = prev.len[0]
            + prev.len[1] * AB::Expr::from_canonical_u32(1 << (2 * MEMCPY_LOOP_LIMB_BITS));

        let write_data_pairs = [
            (prev.data_4, local.data_1),
            (local.data_1, local.data_2),
            (local.data_2, local.data_3),
            (local.data_3, local.data_4),
        ];

        let write_data = write_data_pairs
            .iter()
            .map(|(prev_data, next_data)| {
                // iterate i from 0 ... 3
                array::from_fn::<_, MEMCPY_LOOP_NUM_LIMBS, _>(|i| {
                    is_shift_zero.clone() * (next_data[i])
                        + is_shift_one.clone()
                            * (if i < 1 {
                                prev_data[i + 3]
                            } else {
                                next_data[i - 1]
                            })
                        + is_shift_two.clone()
                            * (if i < 2 {
                                prev_data[i + 2]
                            } else {
                                next_data[i - 2]
                            })
                        + is_shift_three.clone()
                            * (if i < 3 {
                                prev_data[i + 1]
                            } else {
                                next_data[i - 3]
                            })
                })
            })
            .collect::<Vec<_>>();

        builder.assert_bool(local.is_valid);
        local.shift.iter().for_each(|x| builder.assert_bool(*x));
        builder.assert_bool(is_shift_non_zero.clone());
        builder.assert_bool(local.is_valid_not_start);
        builder.assert_bool(local.is_shift_non_zero_or_not_start);
        // is_boundary is either -1, 0 or 1
        builder.assert_tern(local.is_boundary + AB::Expr::ONE);
        // is_valid_not_start = is_valid and is_not_start:
        builder.assert_eq(
            local.is_valid_not_start,
            and::<AB::Expr>(local.is_valid, is_not_start.clone()),
        );

        // is_shift_non_zero_or_not_start is correct
        builder.assert_eq(
            local.is_shift_non_zero_or_not_start,
            or::<AB::Expr>(is_shift_non_zero.clone(), local.is_valid_not_start),
        );

        // builder.assert_eq(
        //     is_valid_is_start.clone() * (is_shift_zero.clone() * local.source),
        //     is_valid_is_start.clone()
        //         * (is_shift_zero.clone() * (prev.source + AB::Expr::from_canonical_u32(16))),
        // );

        // builder.assert_eq(
        //     is_not_start.clone() * local.source,
        //     is_not_start.clone() * (prev.source + AB::Expr::from_canonical_u32(16)),
        // );

        // if !is_valid, then is_boundary = 0, shift = 0 (we will use this assumption later)
        // let mut is_not_valid_when = builder.when(not::<AB::Expr>(local.is_valid));
        // is_not_valid_when.assert_zero(local.is_boundary);
        // is_not_valid_when.assert_zero(shift.clone());

        // if is_valid_not_start, then len = prev_len - 16, source = prev_source + 16,
        // and dest = prev_dest + 16, shift = prev_shift

        // is_valid_not_start is degree 1, since it uses the variable as a precondition
        let mut is_valid_not_start_when = builder.when(local.is_valid_not_start);
        is_valid_not_start_when.assert_eq(len.clone(), prev_len - AB::Expr::from_canonical_u32(16));

        // TODO: fix this constraint? or why is this constraint failing

        /*

        error is if the initial source value is < 12, then -16, we do a saturating sub so its bounded below by 0
        then, it results in a mismatch of values
         */
        // degree 1 * deg 2 * deg 1 = deg 4

        // is_valid_not_start_when.assert_eq(
        //     prev_is_not_start.clone() * local.source,
        //     prev_is_not_start.clone() * (prev.source + AB::Expr::from_canonical_u32(16)),
        // );

        is_valid_not_start_when.assert_eq(local.dest, prev.dest + AB::Expr::from_canonical_u32(16));

        local
            .shift
            .iter()
            .zip(prev.shift.iter())
            .for_each(|(local_shift, prev_shift)| {
                is_valid_not_start_when.assert_eq(*local_shift, *prev_shift);
            });

        // make sure if previous row is valid and not end, then local.is_valid = 1
        builder
            .when(prev_is_not_end.clone() - not::<AB::Expr>(prev.is_valid))
            .assert_one(local.is_valid);

        // if prev.is_valid_start, then timestamp = prev_timestamp + is_shift_non_zero
        // since is_shift_non_zero degree is 2, we need to keep the degree of the condition to 1
        builder
            .when(not::<AB::Expr>(prev.is_valid_not_start) - not::<AB::Expr>(prev.is_valid))
            .assert_eq(local.timestamp, prev.timestamp + AB::Expr::ONE);

        // if prev.is_valid_not_start and local.is_valid_not_start, then timestamp=prev_timestamp+8
        // prev.is_valid_not_start is the opposite of previous condition
        builder
            .when(
                local.is_valid_not_start
                    - (not::<AB::Expr>(prev.is_valid_not_start) - not::<AB::Expr>(prev.is_valid)),
            )
            .assert_eq(
                local.timestamp,
                prev.timestamp + AB::Expr::from_canonical_usize(8),
            );

        // Receive message from memcpy bus or send message to it
        // The last data is shift if is_boundary = -1, and 4 if is_boundary = 1
        // This actually receives when is_boundary = -1

        /*
            data mismatched along memcpy bus, local.source value?
            we only send if is_boundary = -1 or 1
            start or end
        */

        // local_source_0_4_8
        // eprintln!("is_source_small: {:?}", is_source_small);
        // eprintln!("is_shift_non_zero: {:?}", is_shift_non_zero);
        // eprintln!(
        //     "current timestamp: {:?}",
        //     local.timestamp * AB::Expr::from_canonical_u32(1)
        // );

        self.memcpy_bus
            .send(
                local.timestamp
                    + (local.is_boundary + AB::Expr::ONE) * AB::Expr::from_canonical_usize(4),
                // - (is_shift_non_zero.clone() * is_source_small.clone()),
                local.dest,
                local.source,
                len.clone(),
                (AB::Expr::ONE - local.is_boundary) * shift.clone() * (AB::F::TWO).inverse()
                    + (local.is_boundary + AB::Expr::ONE) * AB::Expr::TWO,
            )
            .eval(builder, local.is_boundary);

        // Read data from memory
        let read_data = [local.data_1, local.data_2, local.data_3, local.data_4];

        eprintln!(
            "starting timestamp: {:?}",
            timestamp * AB::Expr::from_canonical_u32(1)
        );
        read_data.iter().enumerate().for_each(|(idx, data)| {
            // is valid read of entire 16 block chunk?
            let is_valid_read = if idx == 3 {
                // will always be a valid read
                AB::Expr::ONE * (local.is_valid)
            } else {
                // if idx < 3, its not an entire block read, if its the first block
                AB::Expr::ONE * (local.is_valid_not_start)
            };
            self.memory_bridge
                .read(
                    MemoryAddress::new(
                        AB::Expr::from_canonical_u32(RV32_MEMORY_AS),
                        local.source - AB::Expr::from_canonical_usize(16 - idx * 4),
                    ),
                    *data,
                    timestamp_pp(is_valid_read.clone()),
                    &local.read_aux[idx],
                )
                .eval(builder, is_valid_read.clone());
            eprintln!(
                "local.source: {:?}, data: {:?}, local.source - AB::Expr::from_canonical_usize(16 - idx * 4): {:?}, timestamp_pp: {:?}, is_valid_read: {:?}",
                local.source * AB::Expr::from_canonical_u32(1),
                data.clone().map(|x| x * (AB::Expr::from_canonical_u32(1))),
                local.source - AB::Expr::from_canonical_usize(16 - idx * 4) * AB::Expr::from_canonical_u32(1),
                timestamp_pp(AB::Expr::ZERO),
                is_valid_read.clone()
            );
        });
        /*
        7 values go:
            address sspace
            pointer to address
            data (4)
            timestamp

        rn timestamp is wrong...
        */
        // memory bridge data is tracing_read
        // based off of the unshifted data values

        // tracing_write writes the corectly shifted values

        // eprintln!(
        //     "memory bridge read_data: {:?}",
        //     data.clone().map(|x| x * (AB::Expr::from_canonical_u32(1)))
        // );

        //read timestamp is off?
        // seems fine? since we use the read_aux variable to store the timestamp
        // error seems to be off by one for timestamp again, or off by 17 or 21??
        // timestamps are inconsistent everywhere...

        // is timestamping off, is it pointer is off, for small source values
        write_data.iter().enumerate().for_each(|(idx, data)| {
            self.memory_bridge
                .write(
                    MemoryAddress::new(
                        AB::Expr::from_canonical_u32(RV32_MEMORY_AS),
                        local.dest - AB::Expr::from_canonical_usize(16 - idx * 4),
                    ),
                    data.clone(),
                    timestamp_pp(AB::Expr::ONE * (local.is_valid_not_start)),
                    &local.write_aux[idx],
                )
                .eval(builder, local.is_valid_not_start);
            // eprintln!(
            //     "Eval: write_data: {:?}, timestamp_pp: {:?}",
            //     data.clone().map(|x| x * (AB::Expr::from_canonical_u32(1))),
            //     timestamp_pp(AB::Expr::ZERO * (local.is_valid_not_start))
            // );
        });

        // Range check len
        let len_bits_limit = [
            select::<AB::Expr>(
                is_end.clone(),
                AB::Expr::from_canonical_usize(4),
                AB::Expr::from_canonical_usize(MEMCPY_LOOP_LIMB_BITS * 2),
            ),
            select::<AB::Expr>(
                is_end.clone(),
                AB::Expr::ZERO,
                AB::Expr::from_canonical_usize(self.pointer_max_bits - MEMCPY_LOOP_LIMB_BITS * 2),
            ),
        ];
        self.range_bus
            .push(local.len[0], len_bits_limit[0].clone(), true)
            .eval(builder, local.is_valid);
        self.range_bus
            .push(local.len[1], len_bits_limit[1].clone(), true)
            .eval(builder, local.is_valid);
    }
}

#[derive(derive_new::new, Clone, Copy)]
pub struct MemcpyIterExecutor {
    pub offset: usize,
}

#[derive(Copy, Clone, Debug)]
pub struct MemcpyIterMetadata {
    num_rows: usize,
}

impl MultiRowMetadata for MemcpyIterMetadata {
    #[inline(always)]
    fn get_num_rows(&self) -> usize {
        self.num_rows
    }
}

pub type MemcpyIterLayout = MultiRowLayout<MemcpyIterMetadata>;

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug)]
pub struct MemcpyIterRecordHeader {
    pub shift: u8,
    pub dest: u32,
    pub source: u32,
    pub len: u32,
    pub from_pc: u32,
    pub from_timestamp: u32,
    pub register_aux: [MemoryBaseAuxRecord; 3],
}

// This is the part of the record that we keep `(len & !15) + (shift != 0)` times per instruction
#[repr(C)]
#[derive(AlignedBytesBorrow, Debug)]
pub struct MemcpyIterRecordVar {
    pub data: [[u8; MEMCPY_LOOP_NUM_LIMBS]; 4],
    pub read_aux: [MemoryReadAuxRecord; 4],
    pub write_aux: [MemoryWriteBytesAuxRecord<4>; 4],
}

/// **SAFETY**: the order of the fields in `MemcpyLoopRecordMut` and `MemcpyLoopRecordVar`
/// is important.
#[derive(Debug)]
pub struct MemcpyIterRecordMut<'a> {
    pub inner: &'a mut MemcpyIterRecordHeader,
    pub var: &'a mut [MemcpyIterRecordVar],
}

/// Custom borrowing that splits the buffer into a fixed `MemcpyLoopRecordHeader` header
/// followed by a slice of `MemcpyLoopRecordVar`'s of length `num_words` provided at runtime.
/// Uses `align_to_mut()` to make sure the slice is properly aligned to `MemcpyLoopRecordVar`.
/// Has debug assertions to make sure the above works as expected.
impl<'a> CustomBorrow<'a, MemcpyIterRecordMut<'a>, MemcpyIterLayout> for [u8] {
    fn custom_borrow(&'a mut self, layout: MemcpyIterLayout) -> MemcpyIterRecordMut<'a> {
        let (header_buf, rest) =
            unsafe { self.split_at_mut_unchecked(size_of::<MemcpyIterRecordHeader>()) };

        let (_, vars, _) = unsafe { rest.align_to_mut::<MemcpyIterRecordVar>() };
        MemcpyIterRecordMut {
            inner: header_buf.borrow_mut(),
            var: &mut vars[..layout.metadata.num_rows],
        }
    }

    unsafe fn extract_layout(&self) -> MemcpyIterLayout {
        let header: &MemcpyIterRecordHeader = self.borrow();
        let num_rows = ((header.len - header.shift as u32) >> 4) as usize + 1;
        MultiRowLayout::new(MemcpyIterMetadata { num_rows })
    }
}

impl SizedRecord<MemcpyIterLayout> for MemcpyIterRecordMut<'_> {
    fn size(layout: &MemcpyIterLayout) -> usize {
        let mut total_len = size_of::<MemcpyIterRecordHeader>();
        // Align the pointer to the alignment of `Rv32HintStoreVar`
        total_len = total_len.next_multiple_of(align_of::<MemcpyIterRecordVar>());
        total_len += size_of::<MemcpyIterRecordVar>() * layout.metadata.num_rows;
        total_len
    }

    fn alignment(_layout: &MemcpyIterLayout) -> usize {
        align_of::<MemcpyIterRecordHeader>()
    }
}

#[derive(derive_new::new)]
pub struct MemcpyIterFiller {
    pub pointer_max_bits: usize,
    pub range_checker_chip: SharedVariableRangeCheckerChip,
    pub memcpy_loop_chip: Arc<MemcpyLoopChip>,
}

pub type MemcpyIterChip<F> = VmChipWrapper<F, MemcpyIterFiller>;

impl<F, RA> PreflightExecutor<F, RA> for MemcpyIterExecutor
where
    F: PrimeField32,
    for<'buf> RA: RecordArena<'buf, MultiRowLayout<MemcpyIterMetadata>, MemcpyIterRecordMut<'buf>>,
{
    fn get_opcode_name(&self, _: usize) -> String {
        format!("{:?}", Rv32MemcpyOpcode::MEMCPY_LOOP)
    }
    /*

        preflight executor, execute_e12 are for actual execution
        e1: pure execution
        e2: metered execution
    */
    fn execute(
        &self,
        state: VmStateMut<F, TracingMemory, RA>,
        instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError> {
        eprintln!("extensions/memcpy/circuit/src/iteration.rs::execute: PREFLIGHT: MemcpyIterExecutor executing MEMCPY_LOOP opcode");
        let Instruction { opcode, c, .. } = instruction;
        debug_assert_eq!(*opcode, Rv32MemcpyOpcode::MEMCPY_LOOP.global_opcode());
        let shift = c.as_canonical_u32() as u8; // written into c slot
        debug_assert!([0, 1, 2, 3].contains(&shift));

        let mut dest = read_rv32_register(
            state.memory.data(),
            if shift == 0 {
                A3_REGISTER_PTR
            } else {
                A1_REGISTER_PTR
            } as u32,
        );
        let mut source = read_rv32_register(
            state.memory.data(),
            if shift == 0 {
                A4_REGISTER_PTR
            } else {
                A3_REGISTER_PTR
            } as u32,
        );

        let mut len = read_rv32_register(state.memory.data(), A2_REGISTER_PTR as u32);

        // source = source.saturating_sub(12 * (shift != 0) as u32);
        debug_assert!(
            shift == 0 || (dest % 4 == 0),
            "dest must be 4-byte aligned in MEMCPY_LOOP"
        );
        debug_assert!(len >= shift as u32);

        // Create a record sized to the exact number of 16-byte iterations (header + iterations)
        // This calculation must match extract_layout and fill_trace

        let head = if shift == 0 { 0 } else { 4 - shift as u32 };
        let effective_len = len.saturating_sub(head);
        let num_iters = (effective_len / 16);

        // eprintln!(
        //     "PREFLIGHT: len={}, shift={}, effective_len={}, num_iters={}, allocated_rows={}",
        //     len,
        //     shift,
        //     effective_len,
        //     num_iters,
        //     num_iters + 1
        // );
        let record: MemcpyIterRecordMut<'_> =
            state.ctx.alloc(MultiRowLayout::new(MemcpyIterMetadata {
                //allocating based on number of rows needed
                num_rows: num_iters as usize + 1,
            })); // is this too big then??

        // Store the original values in the record
        record.inner.shift = shift;
        record.inner.from_pc = *state.pc;
        record.inner.from_timestamp = state.memory.timestamp;
        eprintln!("state.memory.timestamp: {:?}", state.memory.timestamp);
        record.inner.dest = dest;
        record.inner.source = source;
        record.inner.len = len;
        // eprintln!(
        //     "shift = {:?}, len = {:?}, source = {:?}, source%16 = {:?}, dest = {:?}, dest%16 = {:?}",
        //     shift, len, source, source % 16, dest, dest % 16
        // );

        // Fill record.var for the first row of iteration trace
        // FIX 2: read source-4 (the word ending at s[-1]); zero if out-of-bounds.

        // this causes timestamp errors, if shift == 0
        // let first_word: [u8; MEMCPY_LOOP_NUM_LIMBS] = tracing_read(
        //     state.memory,
        //     RV32_MEMORY_AS,
        //     source,
        //     &mut record.var[0].read_aux[2].prev_timestamp,
        // );
        source = source.saturating_sub(12 * (shift != 0) as u32);
        // we have saturating sub, which isnt a perfect sub,
        // 0, 4, 20
        // source is tOO SMALL, SO READING SAME DATA TWICE??

        record.var[0].data[3] = tracing_read(
            state.memory,
            RV32_MEMORY_AS,
            source - 4 * (source >= 4) as u32,
            &mut record.var[0].read_aux[3].prev_timestamp,
        );

        eprintln!(
            "record.var[0].read_aux[3].prev_timestamp: {:?}, record.var[0].data[3]: {:?}",
            record.var[0].read_aux[3].prev_timestamp, record.var[0].data[3]
        );

        // Fill record.var for the rest of the rows of iteration trace
        let mut idx = 1;
        for _ in 0..num_iters {
            let writes_data: [[u8; MEMCPY_LOOP_NUM_LIMBS]; 4] = array::from_fn(|i| {
                record.var[idx].data[i] = tracing_read(
                    state.memory,
                    RV32_MEMORY_AS,
                    source + 4 * i as u32,
                    &mut record.var[idx].read_aux[i].prev_timestamp,
                );
                //use shifted data, to construct the write data for each given word
                let write_data: [u8; MEMCPY_LOOP_NUM_LIMBS] = array::from_fn(|j| {
                    if j < shift as usize {
                        if i > 0 {
                            // First s bytes come from previous 4-byte word tail, take from previous word, in our 16 byte chunk
                            record.var[idx].data[i - 1][j + (4 - shift as usize)]
                        } else {
                            // For i == 0, take from previous chunk's last word tail; otherwise, take last word of previous chunk
                            record.var[idx - 1].data[3][j + (4 - shift as usize)]
                        }
                    } else {
                        // Remaining 4 - s bytes come from current word head
                        record.var[idx].data[i][j - shift as usize]
                    }
                });
                eprintln!(
                    "src {:?}, dest {:?}, idx {:?}",
                    source + 4 * i as u32,
                    dest + 4 * i as u32,
                    idx
                );
                eprintln!("execute read_data: {:?}", record.var[idx].data[i]);
                write_data
            });
            eprintln!("record.var[idx].data: {:?}", record.var[idx].data);
            writes_data.iter().enumerate().for_each(|(i, write_data)| {
                tracing_write(
                    state.memory,
                    RV32_MEMORY_AS,
                    dest + 4 * i as u32,
                    *write_data,
                    &mut record.var[idx].write_aux[i].prev_timestamp,
                    &mut record.var[idx].write_aux[i].prev_data,
                );
            });
            len -= 16;
            source += 16;
            dest += 16;
            idx += 1;
        }

        let mut dest_data = [0; 4];
        let mut source_data = [0; 4];
        let mut len_data = [0; 4];
        source = source.saturating_add(12 * (shift != 0) as u32);
        tracing_write(
            state.memory,
            RV32_REGISTER_AS,
            if shift == 0 {
                A3_REGISTER_PTR
            } else {
                A1_REGISTER_PTR
            } as u32,
            dest.to_le_bytes(),
            &mut record.inner.register_aux[0].prev_timestamp,
            &mut dest_data,
        );

        tracing_write(
            state.memory,
            RV32_REGISTER_AS,
            if shift == 0 {
                A4_REGISTER_PTR
            } else {
                A3_REGISTER_PTR
            } as u32,
            source.to_le_bytes(),
            &mut record.inner.register_aux[1].prev_timestamp,
            &mut source_data,
        );

        tracing_write(
            state.memory,
            RV32_REGISTER_AS,
            A2_REGISTER_PTR as u32,
            len.to_le_bytes(),
            &mut record.inner.register_aux[2].prev_timestamp,
            &mut len_data,
        );

        debug_assert_eq!(record.inner.dest, u32::from_le_bytes(dest_data));
        debug_assert_eq!(record.inner.source, u32::from_le_bytes(source_data));
        debug_assert_eq!(record.inner.len, u32::from_le_bytes(len_data));

        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);
        eprintln!("extensions/memcpy/circuit/src/iteration.rs::execute: PREFLIGHT: preflight height: {:?}", num_iters + 1);
        eprintln!("extensions/memcpy/circuit/src/iteration.rs::execute: PREFLIGHT: Preflight MemcpyIterExecutor finished");
        Ok(())
    }
}

// dummy air is wrong?
// initial timestamp is 18, but the tracing read is filing it in as 1?
// this is bc in the testing framework, it will write things into memory to "set up" the initial state

/*
- generate_proving_ctx is what creates trace fill
- row major matrix, so stored in a vector, row by row
- look at common_main, is where trace is filled

- print into excel, look in trace


- LAST STEP:
 */
impl<F: PrimeField32> TraceFiller<F> for MemcpyIterFiller {
    fn fill_trace(
        &self,
        mem_helper: &MemoryAuxColsFactory<F>,
        trace: &mut RowMajorMatrix<F>,
        rows_used: usize,
    ) {
        if rows_used == 0 {
            return;
        }

        let width = trace.width;
        debug_assert_eq!(width, NUM_MEMCPY_ITER_COLS);
        let mut trace = &mut trace.values[..width * rows_used];
        let mut sizes = Vec::with_capacity(rows_used >> 1);
        let mut chunks = Vec::with_capacity(rows_used >> 1);

        let mut num_loops: usize = 0;
        let mut num_iters: usize = 0;

        while !trace.is_empty() {
            let record: &MemcpyIterRecordHeader = unsafe { get_record_from_slice(&mut trace, ()) };
            let num_rows = ((record.len - record.shift as u32) >> 4) as usize + 1;
            let (chunk, rest) = trace.split_at_mut(width * num_rows as usize);
            sizes.push(num_rows);
            chunks.push(chunk);
            trace = rest;
            num_loops += 1;
            num_iters += num_rows;
        }
        tracing::info!(
            "num_loops: {:?}, num_iters: {:?}, sizes: {:?}",
            num_loops,
            num_iters,
            sizes
        );

        chunks.iter_mut().zip(sizes.iter()).enumerate().for_each(
            |(_row_idx, (chunk, &num_rows))| {
                let record: MemcpyIterRecordMut = unsafe {
                    get_record_from_slice(
                        chunk,
                        MultiRowLayout::new(MemcpyIterMetadata { num_rows }),
                    )
                };

                tracing::info!("shift: {:?}", record.inner.shift);

                self.memcpy_loop_chip.add_new_loop(
                    mem_helper,
                    record.inner.from_pc,
                    record.inner.from_timestamp,
                    record.inner.dest,
                    record.inner.source,
                    record.inner.len,
                    record.inner.shift,
                    record.inner.register_aux.clone(),
                );

                // Calculate the timestamp for the last memory access
                // 4 reads + 4 writes per iteration + (shift != 0) read for the loop header
                let timestamp = record.inner.from_timestamp + ((num_rows - 1) << 3) as u32 + 1;
                let mut timestamp_delta: u32 = 0;
                let mut get_timestamp = |is_access: bool| {
                    if is_access {
                        timestamp_delta += 1;
                    }
                    timestamp - timestamp_delta
                };

                // eprintln!("record.inner.source: {:?}", record.inner.source);
                // eprintln!(
                //     "num_rows: {:?}, record.inner.source + ((num_rows - 1) << 4): {:?}",
                //     num_rows,
                //     record.inner.source + ((num_rows - 1) << 4) as u32
                // );

                let mut dest = record.inner.dest + ((num_rows - 1) << 4) as u32;
                let mut source = (record.inner.source + ((num_rows - 1) << 4) as u32)
                    .saturating_sub(12 * (record.inner.shift != 0) as u32);
                eprintln!(
                    "record.inner.source: {:?}, num_rows: {:?}, source: {:?}",
                    record.inner.source, num_rows, source
                );
                let mut len =
                    record.inner.len - ((num_rows - 1) << 4) as u32 - record.inner.shift as u32;

                // We are going to fill row in the reverse order
                chunk
                    .rchunks_exact_mut(width)
                    .zip(record.var.iter().enumerate().rev())
                    .for_each(|(row, (idx, var))| {
                        let cols: &mut MemcpyIterCols<F> = row.borrow_mut();

                        let is_start = idx == 0;
                        let is_end = idx == num_rows - 1;

                        // Range check len
                        let len_u16_limbs = [len & 0xffff, len >> 16];
                        if is_end {
                            self.range_checker_chip.add_count(len_u16_limbs[0], 4);
                            self.range_checker_chip.add_count(len_u16_limbs[1], 0);
                        } else {
                            self.range_checker_chip
                                .add_count(len_u16_limbs[0], 2 * MEMCPY_LOOP_LIMB_BITS);
                            self.range_checker_chip.add_count(
                                len_u16_limbs[1],
                                self.pointer_max_bits - 2 * MEMCPY_LOOP_LIMB_BITS,
                            );
                        }

                        // Fill memory read/write auxiliary columns
                        if is_start {
                            cols.write_aux.iter_mut().rev().for_each(|aux_col| {
                                mem_helper.fill_zero(aux_col.as_mut());
                            });
                            mem_helper.fill(
                                var.read_aux[3].prev_timestamp,
                                get_timestamp(true),
                                cols.read_aux[3].as_mut(),
                            );
                            cols.read_aux[..3].iter_mut().rev().for_each(|aux_col| {
                                mem_helper.fill_zero(aux_col.as_mut());
                            });

                            debug_assert_eq!(get_timestamp(false), record.inner.from_timestamp);
                        } else {
                            var.write_aux
                                .iter()
                                .zip(cols.write_aux.iter_mut())
                                .rev()
                                .for_each(|(aux_record, aux_col)| {
                                    mem_helper.fill(
                                        aux_record.prev_timestamp,
                                        get_timestamp(true),
                                        aux_col.as_mut(),
                                    );
                                    aux_col.set_prev_data(
                                        aux_record.prev_data.map(F::from_canonical_u8),
                                    );
                                });

                            var.read_aux
                                .iter()
                                .zip(cols.read_aux.iter_mut())
                                .rev()
                                .for_each(|(aux_record, aux_col)| {
                                    mem_helper.fill(
                                        aux_record.prev_timestamp,
                                        get_timestamp(true), // BUG was HERE. given current timestamp, need to read from memory at an earlier timestamp, cant read form the current one
                                        aux_col.as_mut(),
                                    );
                                });
                        }

                        cols.data_4 = var.data[3].map(F::from_canonical_u8);
                        cols.data_3 = var.data[2].map(F::from_canonical_u8);
                        cols.data_2 = var.data[1].map(F::from_canonical_u8);
                        cols.data_1 = var.data[0].map(F::from_canonical_u8);
                        cols.is_boundary = if is_end {
                            F::ONE
                        } else if is_start {
                            F::NEG_ONE
                        } else {
                            F::ZERO
                        };
                        cols.is_shift_non_zero_or_not_start =
                            F::from_bool(record.inner.shift != 0 || !is_start);
                        cols.is_valid_not_start = F::from_bool(!is_start);
                        cols.is_valid = F::ONE;
                        cols.shift = [
                            record.inner.shift == 1,
                            record.inner.shift == 2,
                            record.inner.shift == 3,
                        ]
                        .map(F::from_bool);
                        cols.len = [len & 0xffff, len >> 16].map(F::from_canonical_u32);
                        cols.source = F::from_canonical_u32(source);
                        cols.dest = F::from_canonical_u32(dest);
                        cols.timestamp = F::from_canonical_u32(get_timestamp(false));
                        cols.is_source_0_4_8 =
                            [source == 0, source == 4, source == 8].map(F::from_bool);
                        dest = dest.saturating_sub(16);
                        source = source.saturating_sub(16);
                        len += 16;

                        // if row_idx == 0 && is_start {
                        //     tracing::info!("first_roooooow, timestamp: {:?}, dest: {:?}, source: {:?}, len_0: {:?}, len_1: {:?}, shift: {:?}, is_valid: {:?}, is_valid_not_start: {:?}, is_shift_non_zero: {:?}, is_boundary: {:?}, data_1: {:?}, data_2: {:?}, data_3: {:?}, data_4: {:?}, read_aux: {:?}, read_aux_lt: {:?}",
                        //     cols.timestamp.as_canonical_u32(),
                        //     cols.dest.as_canonical_u32(),
                        //     cols.source.as_canonical_u32(),
                        //     cols.len[0].as_canonical_u32(),
                        //     cols.len[1].as_canonical_u32(),
                        //     cols.shift[1].as_canonical_u32() * 2 + cols.shift[0].as_canonical_u32(),
                        //     cols.is_valid.as_canonical_u32(),
                        //     cols.is_valid_not_start.as_canonical_u32(),
                        //     cols.is_shift_non_zero.as_canonical_u32(),
                        //     cols.is_boundary.as_canonical_u32(),
                        //     cols.data_1.map(|x| x.as_canonical_u32()).to_vec(),
                        //     cols.data_2.map(|x| x.as_canonical_u32()).to_vec(),
                        //     cols.data_3.map(|x| x.as_canonical_u32()).to_vec(),
                        //     cols.data_4.map(|x| x.as_canonical_u32()).to_vec(),
                        //     cols.read_aux.map(|x| x.get_base().prev_timestamp.as_canonical_u32()).to_vec(),
                        //     cols.read_aux.map(|x| x.get_base().timestamp_lt_aux.lower_decomp.iter().map(|x| x.as_canonical_u32()).collect::<Vec<_>>()).to_vec());
                        //     }
                    });
            },
        );

        // chunks.iter().enumerate().for_each(|(row_idx, chunk)| {
        // let mut prv_data = [0; 4];
        // tracing::info!("row_idx: {:?}", row_idx);

        // chunk.chunks_exact(width)
        // .enumerate()
        // .for_each(|(idx, row)| {
        //     let cols: &MemcpyIterCols<F> = row.borrow();
        //     let is_valid_not_start = cols.is_valid_not_start.as_canonical_u32() != 0;
        //     let is_shift_non_zero = cols.is_shift_non_zero.as_canonical_u32() != 0;
        //     let source = cols.source.as_canonical_u32();
        //     let dest = cols.dest.as_canonical_u32();
        //     let mut bad_col = false;
        //     tracing::info!("source: {:?}, dest: {:?}", source, dest);
        //     cols.read_aux.iter().enumerate().for_each(|(idx, aux)| {
        // if is_valid_not_start || (is_shift_non_zero && idx == 3) {
        //     let prev_t = aux.get_base().prev_timestamp.as_canonical_u32();
        //     let curr_t = cols.timestamp.as_canonical_u32();
        //     let ts_lt = aux.get_base().timestamp_lt_aux.lower_decomp.iter()
        //     .enumerate()
        //     .fold(F::ZERO, |acc, (i, &val)| {
        //         acc + val * F::from_canonical_usize(1 << (i * 17))
        //     }).as_canonical_u32();
        //     if curr_t + idx as u32 != ts_lt + prev_t + 1 {
        //         bad_col = true;
        //     }
        // }
        //         if dest + 4 * idx as u32 == 2097216 || dest - 4 * (idx + 1) as u32 == 2097216 || dest + 4 * idx as u32 == 2097280 || dest - 4 * (idx + 1) as u32 == 2097280 {
        //             bad_col = true;
        //         }
        //     });
        //     if bad_col {
        //         let write_data_pairs = [
        //             (prv_data, cols.data_1.map(|x| x.as_canonical_u32())),
        //             (cols.data_1.map(|x| x.as_canonical_u32()), cols.data_2.map(|x| x.as_canonical_u32())),
        //             (cols.data_2.map(|x| x.as_canonical_u32()), cols.data_3.map(|x| x.as_canonical_u32())),
        //             (cols.data_3.map(|x| x.as_canonical_u32()), cols.data_4.map(|x| x.as_canonical_u32())),
        //         ];

        //         let shift = cols.shift[1].as_canonical_u32() * 2 + cols.shift[0].as_canonical_u32();
        //         let write_data = write_data_pairs
        //             .iter()
        //             .map(|(prev_data, next_data)| {
        //                 array::from_fn::<_, MEMCPY_LOOP_NUM_LIMBS, _>(|i| {
        //                 (shift == 0) as u32 * (next_data[i])
        //                         + (shift == 1) as u32
        //                             * (if i < 3 {
        //                                 next_data[i + 1]
        //                             } else {
        //                                 prev_data[i - 3]
        //                             })
        //                         + (shift == 2) as u32
        //                             * (if i < 2 {
        //                                 next_data[i + 2]
        //                             } else {
        //                                 prev_data[i - 2]
        //                             })
        //                         + (shift == 3) as u32
        //                             * (if i < 1 {
        //                                 next_data[i + 3]
        //                             } else {
        //                                 prev_data[i - 1]
        //                             })
        //                 })
        //             })
        //             .collect::<Vec<_>>();

        //         tracing::info!("row_idx: {:?}, idx: {:?}, timestamp: {:?}, dest: {:?}, source: {:?}, len_0: {:?}, len_1: {:?}, shift_0: {:?}, shift_1: {:?}, is_valid: {:?}, is_valid_not_start: {:?}, is_shift_non_zero: {:?}, is_boundary: {:?}, write_data: {:?}, prv_data: {:?}, data_1: {:?}, data_2: {:?}, data_3: {:?}, data_4: {:?}, read_aux: {:?}, read_aux_lt: {:?}, write_aux: {:?}, write_aux_lt: {:?}, write_aux_prev_data: {:?}",
        //         row_idx,
        //         idx,
        //         cols.timestamp.as_canonical_u32(),
        //         cols.dest.as_canonical_u32(),
        //         cols.source.as_canonical_u32(),
        //         cols.len[0].as_canonical_u32(),
        //         cols.len[1].as_canonical_u32(),
        //         cols.shift[0].as_canonical_u32(),
        //         cols.shift[1].as_canonical_u32(),
        //         cols.is_valid.as_canonical_u32(),
        //         cols.is_valid_not_start.as_canonical_u32(),
        //         cols.is_shift_non_zero.as_canonical_u32(),
        //         cols.is_boundary.as_canonical_u32(),
        //         write_data,
        //         prv_data,
        //         cols.data_1.map(|x| x.as_canonical_u32()).to_vec(),
        //         cols.data_2.map(|x| x.as_canonical_u32()).to_vec(),
        //         cols.data_3.map(|x| x.as_canonical_u32()).to_vec(),
        //         cols.data_4.map(|x| x.as_canonical_u32()).to_vec(),
        //         cols.read_aux.map(|x| x.get_base().prev_timestamp.as_canonical_u32()).to_vec(),
        //         cols.read_aux.map(|x| x.get_base().timestamp_lt_aux.lower_decomp.iter().map(|x| x.as_canonical_u32()).collect::<Vec<_>>()).to_vec(),
        //         cols.write_aux.map(|x| x.get_base().prev_timestamp.as_canonical_u32()).to_vec(),
        //         cols.write_aux.map(|x| x.get_base().timestamp_lt_aux.lower_decomp.iter().map(|x| x.as_canonical_u32()).collect::<Vec<_>>()).to_vec(),
        //         cols.write_aux.map(|x| x.prev_data.map(|x| x.as_canonical_u32()).to_vec()).to_vec());
        //     }
        //     prv_data = cols.data_4.map(|x| x.as_canonical_u32());
        // });
        // });
        // assert!(false);
    }
}

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct MemcpyIterPreCompute {
    c: u8,
}

impl<F: PrimeField32> Executor<F> for MemcpyIterExecutor {
    fn pre_compute_size(&self) -> usize {
        size_of::<MemcpyIterPreCompute>()
    }

    fn pre_compute<Ctx>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError>
    where
        Ctx: ExecutionCtxTrait,
    {
        let data: &mut MemcpyIterPreCompute = data.borrow_mut();
        self.pre_compute_impl(pc, inst, data)?;
        Ok(execute_e1_impl::<_, _>)
    }
}

impl<F: PrimeField32> MeteredExecutor<F> for MemcpyIterExecutor {
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<MemcpyIterPreCompute>>()
    }

    fn metered_pre_compute<Ctx>(
        &self,
        chip_idx: usize,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError>
    where
        Ctx: MeteredExecutionCtxTrait,
    {
        let data: &mut E2PreCompute<MemcpyIterPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        self.pre_compute_impl(pc, inst, &mut data.data)?;
        Ok(execute_e2_impl::<_, _>)
    }
}

#[inline(always)]
unsafe fn execute_e12_impl<F: PrimeField32, CTX: ExecutionCtxTrait>(
    pre_compute: &MemcpyIterPreCompute,
    instret: &mut u64,
    pc: &mut u32,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) -> u32 {
    eprintln!("E12 MemcpyIterExecutor started");
    let shift = pre_compute.c;
    let mut height = 1;
    // Read dest and source from registers
    let (dest, source) = if shift == 0 {
        (
            exec_state.vm_read::<u8, 4>(RV32_REGISTER_AS, A3_REGISTER_PTR as u32),
            exec_state.vm_read::<u8, 4>(RV32_REGISTER_AS, A4_REGISTER_PTR as u32),
        )
    } else {
        (
            exec_state.vm_read::<u8, 4>(RV32_REGISTER_AS, A1_REGISTER_PTR as u32),
            exec_state.vm_read::<u8, 4>(RV32_REGISTER_AS, A3_REGISTER_PTR as u32),
        )
    };
    // Read length from a2 register
    let len = exec_state.vm_read::<u8, 4>(RV32_REGISTER_AS, A2_REGISTER_PTR as u32);
    let mut dest = u32::from_le_bytes(dest);
    let mut source = u32::from_le_bytes(source).saturating_sub(12 * (shift != 0) as u32);
    let mut len = u32::from_le_bytes(len);

    let head = if shift == 0 { 0 } else { 4 - shift as u32 };
    let effective_len = len.saturating_sub(head);
    let num_iters = (effective_len / 16) as u32; // floor((len - head)/16)

    // Check address ranges are valid

    /*
    difference in code is with modifiyng source, shift !=0, * 12 etc.
    executing same PC instruction over and over?
        invalid instruction probably??
     */

    debug_assert!(dest < (1 << POINTER_MAX_BITS));
    debug_assert!((source - 4 * (shift != 0) as u32) < (1 << POINTER_MAX_BITS));

    let to_dest = dest + ((len - shift as u32) & !15);
    let to_source = source + ((len - shift as u32) & !15);
    debug_assert!(to_dest <= (1 << POINTER_MAX_BITS));
    debug_assert!(to_source <= (1 << POINTER_MAX_BITS));
    // Make sure the destination and source are not overlapping
    debug_assert!(to_dest <= source || to_source <= dest);

    // Read the previous data from memory if shift != 0
    // - 12 * (shift != 0) as u32 is affecting the write_data lol
    let mut prev_data: [u8; 4] = if shift != 0 {
        exec_state.vm_read::<u8, 4>(RV32_MEMORY_AS, source - 4 * (source >= 4) as u32)
    } else {
        exec_state.vm_read::<u8, 4>(RV32_MEMORY_AS, source as u32)
    };

    eprintln!("num_iters: {:?}", num_iters);
    eprintln!("source: {:?}, dest: {:?}, pc: {:?}", source, dest, *pc);
    for _ in 0..num_iters {
        for i in 0..4u32 {
            let cur_word = exec_state.vm_read::<u8, 4>(RV32_MEMORY_AS, source + 4 * i);
            let write_data: [u8; 4] = array::from_fn(|j| {
                if (j as u8) < shift {
                    prev_data[j + (4 - shift as usize)]
                } else {
                    cur_word[j - shift as usize]
                }
            });
            eprintln!(
                "source: {:?}, dest: {:?}, write_data: {:?}",
                source + 4 * i,
                dest + 4 * i,
                write_data
            );
            exec_state.vm_write(RV32_MEMORY_AS, dest + 4 * i, &write_data);
            prev_data = cur_word;
        }
        len -= 16;
        source += 16;
        dest += 16;
        height += 1;
    }

    // Write the result back to memory
    if shift == 0 {
        exec_state.vm_write(
            RV32_REGISTER_AS,
            A3_REGISTER_PTR as u32,
            &dest.to_le_bytes(),
        );
        exec_state.vm_write(
            RV32_REGISTER_AS,
            A4_REGISTER_PTR as u32,
            &source.to_le_bytes(),
        );
    } else {
        source += 12;
        exec_state.vm_write(
            RV32_REGISTER_AS,
            A1_REGISTER_PTR as u32,
            &dest.to_le_bytes(),
        );
        exec_state.vm_write(
            RV32_REGISTER_AS,
            A3_REGISTER_PTR as u32,
            &source.to_le_bytes(),
        );
    };
    exec_state.vm_write(RV32_REGISTER_AS, A2_REGISTER_PTR as u32, &len.to_le_bytes());
    *pc = pc.wrapping_add(DEFAULT_PC_STEP);
    *instret += 1;
    assert!(height == num_iters + 1);
    eprintln!("height: {:?}", height);
    eprintln!("E12 MemcpyIterExecutor finished");
    height
}

unsafe fn execute_e1_impl<F: PrimeField32, CTX: ExecutionCtxTrait>(
    pre_compute: &[u8],
    instret: &mut u64,
    pc: &mut u32,
    _instret_end: u64,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &MemcpyIterPreCompute = pre_compute.borrow();
    execute_e12_impl::<F, CTX>(pre_compute, instret, pc, exec_state);
}

unsafe fn execute_e2_impl<F: PrimeField32, CTX: MeteredExecutionCtxTrait>(
    pre_compute: &[u8],
    instret: &mut u64,
    pc: &mut u32,
    _arg: u64,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<MemcpyIterPreCompute> = pre_compute.borrow();
    let height = execute_e12_impl::<F, CTX>(&pre_compute.data, instret, pc, exec_state);
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, height);
}

impl MemcpyIterExecutor {
    fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut MemcpyIterPreCompute,
    ) -> Result<(), StaticProgramError> {
        let Instruction { opcode, c, .. } = inst;
        let c_u32 = c.as_canonical_u32();
        if ![0, 1, 2, 3].contains(&c_u32) {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }
        *data = MemcpyIterPreCompute { c: c_u32 as u8 };
        assert_eq!(*opcode, Rv32MemcpyOpcode::MEMCPY_LOOP.global_opcode());
        Ok(())
    }
}
