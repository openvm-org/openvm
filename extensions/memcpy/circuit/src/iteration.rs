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
        MemoryAddress, MemoryAuxColsFactory,
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
use openvm_rv32im_circuit::adapters::{read_rv32_register, tracing_read, tracing_write};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{Air, AirBuilder, BaseAir},
    p3_field::{Field, FieldAlgebra, PrimeField32},
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    p3_maybe_rayon::prelude::*,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};

use crate::{
    bus::MemcpyBus, MemcpyLoopChip, A1_REGISTER_PTR, A2_REGISTER_PTR, A3_REGISTER_PTR,
    A4_REGISTER_PTR,
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
    pub shift: [T; 2],
    pub is_valid: T,
    pub is_valid_not_start: T,
    pub is_shift_zero: T,
    // -1 for the first iteration, 1 for the last iteration, 0 for the middle iterations
    pub is_boundary: T,
    pub data_1: [T; MEMCPY_LOOP_NUM_LIMBS],
    pub data_2: [T; MEMCPY_LOOP_NUM_LIMBS],
    pub data_3: [T; MEMCPY_LOOP_NUM_LIMBS],
    pub data_4: [T; MEMCPY_LOOP_NUM_LIMBS],
    pub read_aux: [MemoryReadAuxCols<T>; 4],
    pub write_aux: [MemoryWriteAuxCols<T, MEMCPY_LOOP_NUM_LIMBS>; 4],
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
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (prev, local) = (main.row_slice(0), main.row_slice(1));
        let prev: &MemcpyIterCols<AB::Var> = (*prev).borrow();
        let local: &MemcpyIterCols<AB::Var> = (*local).borrow();

        let timestamp: AB::Var = local.timestamp;
        let mut timestamp_delta: usize = 0;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            timestamp + AB::Expr::from_canonical_usize(timestamp_delta - 1)
        };

        let shift = local.shift[0] * AB::Expr::TWO + local.shift[1];
        let is_shift_non_zero = not::<AB::Expr>(local.is_shift_zero);
        let is_shift_one = and::<AB::Expr>(local.shift[0], not::<AB::Expr>(local.shift[1]));
        let is_shift_two = and::<AB::Expr>(not::<AB::Expr>(local.shift[0]), local.shift[1]);
        let is_shift_three = and::<AB::Expr>(local.shift[0], local.shift[1]);

        let is_end =
            (local.is_boundary + AB::Expr::ONE) * local.is_boundary * (AB::F::TWO).inverse();
        let is_not_start = (local.is_boundary + AB::Expr::ONE)
            * (AB::Expr::TWO - local.is_boundary)
            * (AB::F::TWO).inverse();
        let prev_is_not_end = not::<AB::Expr>(
            (prev.is_boundary + AB::Expr::ONE) * prev.is_boundary * (AB::F::TWO).inverse(),
        );

        let len = local.len[0]
            + local.len[1] * AB::Expr::from_canonical_u32(1 << (2 * MEMCPY_LOOP_LIMB_BITS));
        let prev_len = prev.len[0]
            + prev.len[1] * AB::Expr::from_canonical_u32(1 << (2 * MEMCPY_LOOP_LIMB_BITS));

        // write_data =
        //  (local.data_1[shift..4], prev.data_4[0..shift]),
        //  (local.data_2[shift..4], local.data_1[0..shift]),
        //  (local.data_3[shift..4], local.data_2[0..shift]),
        //  (local.data_4[shift..4], local.data_3[0..shift])
        let write_data_pairs = [
            (prev.data_4, local.data_1),
            (local.data_1, local.data_2),
            (local.data_2, local.data_3),
            (local.data_3, local.data_4),
        ];

        let write_data = write_data_pairs
            .iter()
            .map(|(prev_data, next_data)| {
                array::from_fn::<_, MEMCPY_LOOP_NUM_LIMBS, _>(|i| {
                    local.is_shift_zero.clone() * (next_data[i])
                        + is_shift_one.clone()
                            * (if i < 3 {
                                next_data[i + 1]
                            } else {
                                prev_data[i - 3]
                            })
                        + is_shift_two.clone()
                            * (if i < 2 {
                                next_data[i + 2]
                            } else {
                                prev_data[i - 2]
                            })
                        + is_shift_three.clone()
                            * (if i < 1 {
                                next_data[i + 3]
                            } else {
                                prev_data[i - 1]
                            })
                })
            })
            .collect::<Vec<_>>();

        builder.assert_bool(local.is_valid);
        local.shift.iter().for_each(|x| builder.assert_bool(*x));
        builder.assert_bool(local.is_valid_not_start);
        builder.assert_bool(local.is_shift_zero);
        // is_boundary is either -1, 0 or 1
        builder.assert_tern(local.is_boundary + AB::Expr::ONE);

        // is_valid_not_start = is_valid and is_not_start:
        builder.assert_eq(
            local.is_valid_not_start,
            and::<AB::Expr>(local.is_valid, is_not_start),
        );

        // is_shift_non_zero is correct
        builder.assert_eq(local.is_shift_zero, not::<AB::Expr>(or::<AB::Expr>(local.shift[0], local.shift[1])));

        // if !is_valid, then is_boundary = 0, shift = 0 (we will use this assumption later)
        let mut is_not_valid_when = builder.when(not::<AB::Expr>(local.is_valid));
        is_not_valid_when.assert_zero(local.is_boundary);
        is_not_valid_when.assert_zero(shift.clone());

        // if is_valid_not_start, then len = prev_len - 16, source = prev_source + 16,
        // and dest = prev_dest + 16, shift = prev_shift
        let mut is_valid_not_start_when = builder.when(local.is_valid_not_start);
        is_valid_not_start_when
            .assert_eq(len.clone(), prev_len - AB::Expr::from_canonical_u32(16));
        is_valid_not_start_when
            .assert_eq(local.source, prev.source + AB::Expr::from_canonical_u32(16));
        is_valid_not_start_when.assert_eq(local.dest, prev.dest + AB::Expr::from_canonical_u32(16));
        is_valid_not_start_when.assert_eq(local.shift[0], prev.shift[0]);
        is_valid_not_start_when.assert_eq(local.shift[1], prev.shift[1]);

        // make sure if previous row is valid and not end, then local.is_valid = 1
        builder
            .when(prev_is_not_end - not::<AB::Expr>(prev.is_valid))
            .assert_one(local.is_valid);

        // if prev.is_valid_start, then timestamp = prev_timestamp + is_shift_non_zero
        // since is_shift_non_zero degree is 2, we need to keep the degree of the condition to 1
        builder
            .when(not::<AB::Expr>(prev.is_valid_not_start) - not::<AB::Expr>(prev.is_valid))
            .assert_eq(local.timestamp, prev.timestamp + is_shift_non_zero.clone());

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
        self.memcpy_bus
            .send(
                local.timestamp
                    + (local.is_boundary + AB::Expr::ONE) * AB::Expr::from_canonical_usize(4),
                local.dest,
                local.source,
                len.clone(),
                (AB::Expr::ONE - local.is_boundary) * shift.clone() * (AB::F::TWO).inverse()
                    + (local.is_boundary + AB::Expr::ONE) * AB::Expr::TWO,
            )
            .eval(builder, local.is_boundary);

        // Read data from memory
        let read_data = [
            (local.data_1, local.read_aux[0]),
            (local.data_2, local.read_aux[1]),
            (local.data_3, local.read_aux[2]),
            (local.data_4, local.read_aux[3]),
        ];

        read_data
            .iter()
            .enumerate()
            .for_each(|(idx, (data, read_aux))| {
                let is_valid_read = if idx == 3 {
                    or::<AB::Expr>(is_shift_non_zero.clone(), local.is_valid_not_start)
                } else {
                    local.is_valid_not_start.into()
                };

                self.memory_bridge
                    .read(
                        MemoryAddress::new(
                            AB::Expr::from_canonical_u32(RV32_MEMORY_AS),
                            local.source - AB::Expr::from_canonical_usize(16 - idx * 4),
                        ),
                        *data,
                        timestamp_pp(),
                        read_aux,
                    )
                    .eval(builder, is_valid_read);
            });

        // Write final data to registers
        write_data.iter().enumerate().for_each(|(idx, data)| {
            self.memory_bridge
                .write(
                    MemoryAddress::new(
                        AB::Expr::from_canonical_u32(RV32_MEMORY_AS),
                        local.dest - AB::Expr::from_canonical_usize(16 - idx * 4),
                    ),
                    data.clone(),
                    timestamp_pp(),
                    &local.write_aux[idx],
                )
                .eval(builder, local.is_valid_not_start);
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
            .push(local.len[0].clone(), len_bits_limit[0].clone(), true)
            .eval(builder, local.is_valid);
        self.range_bus
            .push(local.len[1].clone(), len_bits_limit[1].clone(), true)
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
        MultiRowLayout::new(MemcpyIterMetadata {
            num_rows: ((header.len - header.shift as u32) >> 4) as usize + 1,
        })
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

    fn execute(
        &self,
        state: VmStateMut<F, TracingMemory, RA>,
        instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError> {
        let Instruction { opcode, c, .. } = instruction;
        debug_assert_eq!(*opcode, Rv32MemcpyOpcode::MEMCPY_LOOP.global_opcode());
        let shift = c.as_canonical_u32() as u8;
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

        let record = state.ctx.alloc(MultiRowLayout::new(MemcpyIterMetadata {
            num_rows: ((len - shift as u32) >> 4) as usize + 1,
        }));

        // Store the original values in the record
        record.inner.shift = shift;
        record.inner.from_pc = *state.pc;
        record.inner.from_timestamp = state.memory.timestamp;

        if shift != 0 {
            source -= 12;
            record.var[0].data[3] = tracing_read(
                state.memory,
                RV32_MEMORY_AS,
                source - 4,
                &mut record.var[0].read_aux[3].prev_timestamp,
            );
        };

        let mut idx = 1;
        while len - shift as u32 > 15 {
            let writes_data: [[u8; MEMCPY_LOOP_NUM_LIMBS]; 4] = array::from_fn(|i| {
                record.var[idx].data[i] = tracing_read(
                    state.memory,
                    RV32_MEMORY_AS,
                    source + 4 * i as u32,
                    &mut record.var[idx].read_aux[i].prev_timestamp,
                );
                let write_data: [u8; MEMCPY_LOOP_NUM_LIMBS] = array::from_fn(|j| {
                    if j < 4 - shift as usize {
                        record.var[idx].data[i][j + shift as usize]
                    } else if i > 0 {
                        record.var[idx].data[i - 1][j - (4 - shift as usize)]
                    } else {
                        record.var[idx - 1].data[i][j - (4 - shift as usize)]
                    }
                });
                write_data
            });
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

        // Handle the core loop
        if shift != 0 {
            source += 12;
        }

        let mut dest_data = [0; 4];
        let mut source_data = [0; 4];
        let mut len_data = [0; 4];

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

        record.inner.dest = u32::from_le_bytes(dest_data);
        record.inner.source = u32::from_le_bytes(source_data);
        record.inner.len = u32::from_le_bytes(len_data);

        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);

        Ok(())
    }
}

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

        while !trace.is_empty() {
            let record: &MemcpyIterRecordHeader = unsafe { get_record_from_slice(&mut trace, ()) };
            let num_rows = ((record.len - record.shift as u32) >> 4) as usize + 1;
            let (chunk, rest) = trace.split_at_mut(width * num_rows as usize);
            sizes.push(num_rows);
            chunks.push(chunk);
            trace = rest;
        }

        chunks
            .par_iter_mut()
            .zip(sizes.par_iter())
            .for_each(|(chunk, &num_rows)| {
                let record: MemcpyIterRecordMut = unsafe {
                    get_record_from_slice(
                        chunk,
                        MultiRowLayout::new(MemcpyIterMetadata { num_rows }),
                    )
                };

                // Fill memcpy loop record
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

                // 4 reads + 4 writes per iteration + (shift != 0) read for the loop header
                let timestamp = record.inner.from_timestamp
                    + ((num_rows - 1) << 3) as u32
                    + (record.inner.shift != 0) as u32;
                let mut timestamp_delta: u32 = 0;
                let mut get_timestamp = |is_access: bool| {
                    if is_access {
                        timestamp_delta += 1;
                    }
                    timestamp - timestamp_delta
                };

                let mut dest = record.inner.dest + ((num_rows - 1) << 4) as u32;
                let mut source = record.inner.source + ((num_rows - 1) << 4) as u32
                    - 12 * (record.inner.shift != 0) as u32;
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

                            if record.inner.shift == 0 {
                                mem_helper.fill_zero(cols.read_aux[3].as_mut());
                            } else {
                                mem_helper.fill(
                                    var.read_aux[3].prev_timestamp,
                                    get_timestamp(true),
                                    cols.read_aux[3].as_mut(),
                                );
                            }
                            cols.read_aux[..2].iter_mut().rev().for_each(|aux_col| {
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
                                        get_timestamp(true),
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
                        cols.is_shift_zero = F::from_canonical_u8((record.inner.shift == 0) as u8);
                        cols.is_valid_not_start = F::from_canonical_u8(1 - is_start as u8);
                        cols.is_valid = F::ONE;
                        cols.shift = [record.inner.shift & 1, record.inner.shift >> 1]
                            .map(F::from_canonical_u8);
                        cols.len = [len & 0xffff, len >> 16].map(F::from_canonical_u32);
                        cols.source = F::from_canonical_u32(source);
                        cols.dest = F::from_canonical_u32(dest);
                        cols.timestamp = F::from_canonical_u32(get_timestamp(false));

                        dest -= 16;
                        source -= 16;
                        len += 16;
                    });
            });
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
    vm_state: &mut VmExecState<F, GuestMemory, CTX>,
) -> u32 {
    let shift = pre_compute.c;
    let mut height = 1;
    let (dest, source) = if shift == 0 {
        (
            vm_state.vm_read::<u8, 4>(RV32_REGISTER_AS, A3_REGISTER_PTR as u32),
            vm_state.vm_read::<u8, 4>(RV32_REGISTER_AS, A4_REGISTER_PTR as u32),
        )
    } else {
        (
            vm_state.vm_read::<u8, 4>(RV32_REGISTER_AS, A1_REGISTER_PTR as u32),
            vm_state.vm_read::<u8, 4>(RV32_REGISTER_AS, A3_REGISTER_PTR as u32),
        )
    };
    let len = vm_state.vm_read::<u8, 4>(RV32_REGISTER_AS, A2_REGISTER_PTR as u32);

    let mut dest = u32::from_le_bytes(dest);
    let mut source = u32::from_le_bytes(source);
    let mut len = u32::from_le_bytes(len);

    let mut prev_data = if shift == 0 {
        [0; 4]
    } else {
        source -= 12;
        vm_state.vm_read::<u8, 4>(RV32_MEMORY_AS, source - 4)
    };

    while len - shift as u32 > 15 {
        for i in 0..4 {
            let data = vm_state.vm_read::<u8, 4>(RV32_MEMORY_AS, source + 4 * i);
            let write_data: [u8; 4] = array::from_fn(|i| {
                if i < 4 - shift as usize {
                    data[i + shift as usize]
                } else {
                    prev_data[i - (4 - shift as usize)]
                }
            });
            vm_state.vm_write(RV32_MEMORY_AS, dest + 4 * i, &write_data);
            prev_data = data;
        }
        len -= 16;
        source += 16;
        dest += 16;
        height += 1;
    }

    // Write the result back to memory
    if shift == 0 {
        vm_state.vm_write(
            RV32_REGISTER_AS,
            A3_REGISTER_PTR as u32,
            &dest.to_le_bytes(),
        );
        vm_state.vm_write(
            RV32_REGISTER_AS,
            A4_REGISTER_PTR as u32,
            &source.to_le_bytes(),
        );
    } else {
        source += 12;
        vm_state.vm_write(
            RV32_REGISTER_AS,
            A1_REGISTER_PTR as u32,
            &dest.to_le_bytes(),
        );
        vm_state.vm_write(
            RV32_REGISTER_AS,
            A3_REGISTER_PTR as u32,
            &source.to_le_bytes(),
        );
    };
    vm_state.vm_write(RV32_REGISTER_AS, A2_REGISTER_PTR as u32, &len.to_le_bytes());

    vm_state.pc = vm_state.pc.wrapping_add(DEFAULT_PC_STEP);
    vm_state.instret += 1;
    height
}

unsafe fn execute_e1_impl<F: PrimeField32, CTX: ExecutionCtxTrait>(
    pre_compute: &[u8],
    vm_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &MemcpyIterPreCompute = pre_compute.borrow();
    execute_e12_impl::<F, CTX>(pre_compute, vm_state);
}

unsafe fn execute_e2_impl<F: PrimeField32, CTX: MeteredExecutionCtxTrait>(
    pre_compute: &[u8],
    vm_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<MemcpyIterPreCompute> = pre_compute.borrow();
    let height = execute_e12_impl::<F, CTX>(&pre_compute.data, vm_state);
    vm_state
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
