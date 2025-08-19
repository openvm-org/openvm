use std::{
    array,
    borrow::{Borrow, BorrowMut},
    mem::size_of,
    sync::{atomic::AtomicU32, Arc, Mutex},
};

use openvm_circuit::system::memory::{
    offline_checker::{
        MemoryBaseAuxCols, MemoryBridge, MemoryExtendedAuxRecord, MemoryReadAuxCols,
        MemoryReadAuxRecord, MemoryWriteAuxCols, MemoryWriteBytesAuxRecord,
    },
    MemoryAddress, MemoryAuxColsFactory,
};
use openvm_circuit_primitives::{
    utils::{and, not, or, select},
    var_range::{SharedVariableRangeCheckerChip, VariableRangeCheckerBus},
    AlignedBytesBorrow,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::riscv::RV32_MEMORY_AS;
use openvm_stark_backend::{
    config::{StarkGenericConfig, Val},
    interaction::InteractionBuilder,
    p3_air::{Air, AirBuilder, BaseAir},
    p3_field::{Field, FieldAlgebra, PrimeField32},
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    prover::{cpu::CpuBackend, types::AirProvingContext},
    rap::{get_air_name, BaseAirWithPublicValues, PartitionedBaseAir},
    Chip, ChipUsageGetter,
};

use crate::bus::MemcpyBus;

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
        let is_shift_non_zero = or::<AB::Expr>(local.shift[0], local.shift[1]);
        let is_shift_zero = not::<AB::Expr>(is_shift_non_zero.clone());
        let is_shift_one = and::<AB::Expr>(local.shift[0], not::<AB::Expr>(local.shift[1]));
        let is_shift_two = and::<AB::Expr>(not::<AB::Expr>(local.shift[0]), local.shift[1]);
        let is_shift_three = and::<AB::Expr>(local.shift[0], local.shift[1]);

        // TODO:since if is_valid = 0, then is_boundary = 0, we can reduce the degree of the following expressions by removing the is_valid term
        let is_end =
            (local.is_boundary + AB::Expr::ONE) * local.is_boundary * (AB::F::TWO).inverse();
        let is_not_start = (local.is_boundary + AB::Expr::ONE)
            * (AB::Expr::TWO - local.is_boundary)
            * (AB::F::TWO).inverse();

        let len = local.len[0]
            + local.len[1] * AB::F::from_canonical_u32(1 << (2 * MEMCPY_LOOP_LIMB_BITS));

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
                array::from_fn(|i| {
                    is_shift_zero.clone() * (next_data[i])
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
        for i in 0..2 {
            builder.assert_bool(local.shift[i]);
        }
        builder.assert_bool(local.is_valid_not_start);
        // is_boundary is either -1, 0 or 1
        builder.assert_tern(local.is_boundary + AB::Expr::ONE);

        // is_valid_not_start = is_valid and is_not_start:
        builder.assert_eq(local.is_valid_not_start, local.is_valid * is_not_start);

        // if is_valid = 0, then is_boundary = 0, shift = 0
        let mut is_not_valid_when = builder.when(not::<AB::Expr>(local.is_valid));
        is_not_valid_when.assert_zero(local.is_boundary);
        is_not_valid_when.assert_zero(shift.clone());

        // if is_valid_not_start = 1, then len = prev_len - 16, source = prev_source + 16, dest = prev_dest + 16
        let mut is_valid_not_start_when = builder.when(local.is_valid_not_start);
        is_valid_not_start_when
            .assert_eq(local.len[0], prev.len[0] - AB::Expr::from_canonical_u32(16));
        is_valid_not_start_when
            .assert_eq(local.source, prev.source + AB::Expr::from_canonical_u32(16));
        is_valid_not_start_when.assert_eq(local.dest, prev.dest + AB::Expr::from_canonical_u32(16));

        // if prev.is_valid_start, then timestamp = prev_timestamp + is_shift_non_zero
        // since is_shift_non_zero degree is 2, we need to keep the degree of the condition to 1
        builder
            .when(not::<AB::Expr>(prev.is_valid_not_start) - not::<AB::Expr>(prev.is_valid))
            .assert_eq(local.timestamp, prev.timestamp + is_shift_non_zero.clone());
        // if prev.is_valid_not_start and local.is_valid_not_start, then timestamp = prev_timestamp + 8
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
                local.timestamp,
                local.dest,
                local.source,
                len,
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
                            local.source + AB::Expr::from_canonical_usize(idx * 4),
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
                        local.dest + AB::Expr::from_canonical_usize(idx * 4),
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
            .push(local.len[0], len_bits_limit[0].clone(), true)
            .eval(builder, local.is_valid);
        self.range_bus
            .push(local.len[1], len_bits_limit[1].clone(), true)
            .eval(builder, local.is_valid);
    }
}

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug)]
pub struct MemcpyIterRecord {
    pub timestamp: u32,
    pub dest: u32,
    pub source: u32,
    pub len: u32,
    pub shift: u8,
    pub memory_read_data: Vec<[u8; MEMCPY_LOOP_NUM_LIMBS]>,
    pub read_aux: Vec<MemoryExtendedAuxRecord>,
    pub write_aux: Vec<MemoryExtendedAuxRecord>,
}

pub struct MemcpyIterChip {
    pub air: MemcpyIterAir,
    pub records: Arc<Mutex<Vec<MemcpyIterRecord>>>,
    pub num_rows: AtomicU32,
    pub pointer_max_bits: usize,
    pub range_checker_chip: SharedVariableRangeCheckerChip,
}

impl MemcpyIterChip {
    pub fn new(
        memory_bridge: MemoryBridge,
        range_bus: VariableRangeCheckerBus,
        memcpy_bus: MemcpyBus,
        pointer_max_bits: usize,
        range_checker_chip: SharedVariableRangeCheckerChip,
    ) -> Self {
        Self {
            air: MemcpyIterAir::new(memory_bridge, range_bus, memcpy_bus, pointer_max_bits),
            records: Arc::new(Mutex::new(Vec::new())),
            num_rows: AtomicU32::new(0),
            pointer_max_bits,
            range_checker_chip,
        }
    }

    pub fn bus(&self) -> MemcpyBus {
        self.air.memcpy_bus
    }

    pub fn clear(&self) {
        self.records.lock().unwrap().clear();
        self.num_rows.store(0, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn add_new_loop<F: PrimeField32>(
        &self,
        mem_helper: &MemoryAuxColsFactory<F>,
        timestamp: u32,
        dest: u32,
        source: u32,
        len: u32,
        shift: u8,
        memory_read_data: Vec<[u8; MEMCPY_LOOP_NUM_LIMBS]>,
        read_aux: Vec<MemoryReadAuxRecord>,
        write_aux: Vec<MemoryWriteBytesAuxRecord<MEMCPY_LOOP_NUM_LIMBS>>,
    ) {
        let mut len = len;
        // Update number of rows
        self.num_rows
            .fetch_add(len / 16 + 1, std::sync::atomic::Ordering::Relaxed);

        let word_to_u16 = |data: u32| [data & 0xffff, data >> 16];
        let has_shift = (shift != 0) as usize;

        // Range check len
        loop {
            let len_u16_limbs = word_to_u16(len);
            if len > 15 {
                self.range_checker_chip
                    .add_count(len_u16_limbs[0], 2 * MEMCPY_LOOP_LIMB_BITS);
                self.range_checker_chip.add_count(
                    len_u16_limbs[1],
                    self.pointer_max_bits - 2 * MEMCPY_LOOP_LIMB_BITS,
                );
            } else {
                self.range_checker_chip.add_count(len_u16_limbs[0], 4);
                self.range_checker_chip.add_count(len_u16_limbs[1], 0);
            }
            if len < 16 {
                break;
            }
            len -= 16;
        }

        // Read data from memory
        let mut row_read_aux = Vec::new();
        read_aux.iter().enumerate().for_each(|(i, aux)| {
            let mut aux_cols = MemoryBaseAuxCols::<F>::default();
            let read_timestamp = timestamp
                + if i == 0 {
                    0
                } else {
                    (i + (i - has_shift) / 4 * 4) as u32
                };
            mem_helper.fill(aux.prev_timestamp, read_timestamp, &mut aux_cols);
            row_read_aux.push(MemoryExtendedAuxRecord::from_aux_cols(aux_cols));
        });

        // Write data to memory
        let mut row_write_aux = Vec::new();
        write_aux.iter().enumerate().for_each(|(i, aux)| {
            let mut aux_cols = MemoryBaseAuxCols::<F>::default();
            mem_helper.fill(
                aux.prev_timestamp,
                (timestamp as usize + i + has_shift + (i / 4 + 1) * 4) as u32,
                &mut aux_cols,
            );
            row_write_aux.push(MemoryExtendedAuxRecord::from_aux_cols(aux_cols));
        });

        // Create record
        let row = MemcpyIterRecord {
            timestamp,
            dest,
            source,
            len,
            shift,
            memory_read_data,
            read_aux: row_read_aux,
            write_aux: row_write_aux,
        };

        // Thread-safe push to rows vector
        if let Ok(mut rows_guard) = self.records.lock() {
            rows_guard.push(row);
        }
    }

    /// Generates trace
    pub fn generate_trace<F: PrimeField32>(&self) -> RowMajorMatrix<F> {
        let mut rows = F::zero_vec(
            (self.num_rows.load(std::sync::atomic::Ordering::Relaxed) as usize)
                * NUM_MEMCPY_ITER_COLS,
        );
        let mut current_row = 0;
        let word_to_u16 = |data: u32| [data & 0xffff, data >> 16].map(F::from_canonical_u32);

        for record in self.records.lock().unwrap().iter() {
            let mut timestamp = record.timestamp;
            let shift = [record.shift % 2, record.shift / 2].map(F::from_canonical_u8);
            let has_shift = (record.shift != 0) as usize;
            let mut prev_data = [F::ZERO; MEMCPY_LOOP_NUM_LIMBS];

            for n in 0..(record.len / 16 + 1) as usize {
                let row_start = current_row + n * NUM_MEMCPY_ITER_COLS;
                let row = &mut rows[row_start..row_start + NUM_MEMCPY_ITER_COLS];
                let cols: &mut MemcpyIterCols<F> = row.borrow_mut();
                cols.timestamp = F::from_canonical_u32(timestamp);
                cols.dest = F::from_canonical_u32(record.dest + (n << 2) as u32);
                cols.source = F::from_canonical_u32(record.source + (n << 2) as u32);
                cols.len = word_to_u16(record.len - (n << 2) as u32);
                cols.shift = shift;
                cols.is_valid = F::ONE;
                cols.is_valid_not_start = F::ONE;
                if n == 0 {
                    cols.is_boundary = F::NEG_ONE;
                    if has_shift != 0 {
                        cols.data_4 = record.memory_read_data[0].map(F::from_canonical_u8);
                        prev_data = cols.data_4;
                        cols.read_aux[3].set_base(record.read_aux[0].to_aux_cols());
                    }
                } else {
                    cols.is_boundary = if n as u32 == record.len / 16 {
                        F::ONE
                    } else {
                        F::ZERO
                    };
                    let mut data = [[F::ZERO; MEMCPY_LOOP_NUM_LIMBS]; 4];
                    for i in 0..4 {
                        data[i] = record.memory_read_data[(n - 1) * 4 + i + has_shift]
                            .map(F::from_canonical_u8);
                        cols.read_aux[i]
                            .set_base(record.read_aux[(n - 1) * 4 + i + has_shift].to_aux_cols());
                        cols.write_aux[i].set_base(record.write_aux[(n - 1) * 4 + i].to_aux_cols());
                        let write_data: [F; MEMCPY_LOOP_NUM_LIMBS] = std::array::from_fn(|j| {
                            if j < 4 - record.shift as usize {
                                data[i][record.shift as usize + j]
                            } else {
                                prev_data[j - (4 - record.shift as usize)]
                            }
                        });
                        cols.write_aux[i].set_prev_data(write_data);
                        prev_data = data[i];
                    }
                    cols.data_1 = data[0];
                    cols.data_2 = data[1];
                    cols.data_3 = data[2];
                    cols.data_4 = data[3];
                }
                if n == 0 {
                    timestamp += (record.shift != 0) as u32;
                } else {
                    timestamp += 8;
                }
            }
            current_row += (record.len / 16 + 1) as usize * NUM_MEMCPY_ITER_COLS;
        }
        RowMajorMatrix::new(rows, NUM_MEMCPY_ITER_COLS)
    }
}

// We allow any `R` type so this can work with arbitrary record arenas.
impl<R, SC: StarkGenericConfig> Chip<R, CpuBackend<SC>> for MemcpyIterChip
where
    Val<SC>: PrimeField32,
{
    /// Generates trace and resets the internal counters all to 0.
    fn generate_proving_ctx(&self, _: R) -> AirProvingContext<CpuBackend<SC>> {
        let trace = self.generate_trace::<Val<SC>>();
        AirProvingContext::simple_no_pis(Arc::new(trace))
    }
}

impl ChipUsageGetter for MemcpyIterChip {
    fn air_name(&self) -> String {
        get_air_name(&self.air)
    }
    fn constant_trace_height(&self) -> Option<usize> {
        Some(self.num_rows.load(std::sync::atomic::Ordering::Relaxed) as usize)
    }
    fn current_trace_height(&self) -> usize {
        self.num_rows.load(std::sync::atomic::Ordering::Relaxed) as usize
    }
    fn trace_width(&self) -> usize {
        NUM_MEMCPY_ITER_COLS
    }
}
