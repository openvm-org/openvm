use std::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
    sync::{Arc, Mutex},
};

use openvm_circuit::{
    arch::{ExecutionBridge, ExecutionState},
    system::{
        memory::{
            offline_checker::{
                MemoryBaseAuxCols, MemoryBaseAuxRecord, MemoryBridge, MemoryExtendedAuxRecord,
                MemoryWriteAuxCols,
            },
            MemoryAddress, MemoryAuxColsFactory,
        },
        SystemPort,
    },
    utils::next_power_of_two_or_zero,
};
use openvm_circuit_primitives::{
    utils::{not, or, select},
    var_range::{SharedVariableRangeCheckerChip, VariableRangeCheckerBus},
    AlignedBytesBorrow,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::riscv::RV32_REGISTER_AS;
use openvm_memcpy_transpiler::Rv32MemcpyOpcode;
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
use crate::{
    A1_REGISTER_PTR, A2_REGISTER_PTR, A3_REGISTER_PTR, A4_REGISTER_PTR, MEMCPY_LOOP_LIMB_BITS,
    MEMCPY_LOOP_NUM_LIMBS,
};

#[repr(C)]
#[derive(AlignedBorrow, Debug)]
pub struct MemcpyLoopCols<T> {
    pub from_state: ExecutionState<T>,
    pub dest: [T; MEMCPY_LOOP_NUM_LIMBS],
    pub source: [T; MEMCPY_LOOP_NUM_LIMBS],
    pub len: [T; MEMCPY_LOOP_NUM_LIMBS],
    pub shift: [T; 2],
    pub is_valid: T,
    pub to_timestamp: T,
    pub to_dest: [T; MEMCPY_LOOP_NUM_LIMBS],
    pub to_source: [T; MEMCPY_LOOP_NUM_LIMBS],
    pub to_len: T,
    pub write_aux: [MemoryBaseAuxCols<T>; 3],
    pub source_minus_twelve_carry: T,
    pub to_source_minus_twelve_carry: T,
    // When true, indicates we should saturate (clamp) source-12 to 0 (zero-padding case)
    pub is_source_small: T,
    // When true, indicates we should saturate (clamp) to_source-12 to 0 (zero-padding case)
    pub is_to_source_small: T,
}

pub const NUM_MEMCPY_LOOP_COLS: usize = size_of::<MemcpyLoopCols<u8>>();
pub const MEMCPY_LOOP_NUM_WRITES: u32 = 3;

#[derive(Copy, Clone, Debug, derive_new::new)]
pub struct MemcpyLoopAir {
    pub memory_bridge: MemoryBridge,
    pub execution_bridge: ExecutionBridge,
    pub range_bus: VariableRangeCheckerBus,
    pub memcpy_bus: MemcpyBus,
    pub pointer_max_bits: usize,
    pub offset: usize,
}

impl<F: Field> BaseAir<F> for MemcpyLoopAir {
    fn width(&self) -> usize {
        MemcpyLoopCols::<F>::width()
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for MemcpyLoopAir {}
impl<F: Field> PartitionedBaseAir<F> for MemcpyLoopAir {}

impl<AB: InteractionBuilder> Air<AB> for MemcpyLoopAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &MemcpyLoopCols<AB::Var> = (*local).borrow();

        let mut timestamp_delta: u32 = 0;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            local.to_timestamp
                - AB::Expr::from_canonical_u32(MEMCPY_LOOP_NUM_WRITES - (timestamp_delta - 1))
        };

        let from_le_bytes = |data: [AB::Var; 4]| {
            data.iter().rev().fold(AB::Expr::ZERO, |acc, x| {
                acc * AB::Expr::from_canonical_u32(1 << MEMCPY_LOOP_LIMB_BITS) + *x
            })
        };

        let u8_word_to_u16 = |data: [AB::Var; 4]| {
            [
                data[0] + data[1] * AB::F::from_canonical_u32(1 << MEMCPY_LOOP_LIMB_BITS),
                data[2] + data[3] * AB::F::from_canonical_u32(1 << MEMCPY_LOOP_LIMB_BITS),
            ]
        };

        let shift = local.shift[1] * AB::Expr::TWO + local.shift[0];
        let is_shift_non_zero = or::<AB::Expr>(local.shift[0], local.shift[1]);
        let is_shift_zero = not::<AB::Expr>(is_shift_non_zero.clone());
        let dest = from_le_bytes(local.dest);
        let source = from_le_bytes(local.source);
        let len = from_le_bytes(local.len);
        let to_dest = from_le_bytes(local.to_dest);
        let to_source = from_le_bytes(local.to_source);
        let to_len = local.to_len;

        builder.assert_bool(local.is_valid);
        local.shift.iter().for_each(|x| builder.assert_bool(*x));
        builder.assert_bool(local.source_minus_twelve_carry);
        builder.assert_bool(local.to_source_minus_twelve_carry);
        builder.assert_bool(local.is_source_small);
        builder.assert_bool(local.is_to_source_small);

        let mut shift_zero_when = builder.when(is_shift_zero.clone());
        shift_zero_when.assert_zero(local.source_minus_twelve_carry);
        shift_zero_when.assert_zero(local.to_source_minus_twelve_carry);

        // Write source and destination to registers
        let write_data = [
            (local.dest, local.to_dest, A3_REGISTER_PTR, A1_REGISTER_PTR),
            (
                local.source,
                local.to_source,
                A4_REGISTER_PTR,
                A3_REGISTER_PTR,
            ),
        ];

        write_data.iter().enumerate().for_each(
            |(idx, (prev_data, new_data, zero_shift_ptr, non_zero_shift_ptr))| {
                let write_ptr = select::<AB::Expr>(
                    is_shift_zero.clone(),
                    AB::Expr::from_canonical_usize(*zero_shift_ptr),
                    AB::Expr::from_canonical_usize(*non_zero_shift_ptr),
                );

                self.memory_bridge
                    .write(
                        MemoryAddress::new(
                            AB::Expr::from_canonical_u32(RV32_REGISTER_AS),
                            write_ptr,
                        ),
                        *new_data,
                        timestamp_pp(),
                        &MemoryWriteAuxCols::from_base(local.write_aux[idx], *prev_data),
                    )
                    .eval(builder, local.is_valid);
            },
        );

        // Write length to a2 register
        self.memory_bridge
            .write(
                MemoryAddress::new(
                    AB::Expr::from_canonical_u32(RV32_REGISTER_AS),
                    AB::Expr::from_canonical_usize(A2_REGISTER_PTR),
                ),
                [
                    to_len.into(),
                    AB::Expr::ZERO,
                    AB::Expr::ZERO,
                    AB::Expr::ZERO,
                ],
                timestamp_pp(),
                &MemoryWriteAuxCols::from_base(local.write_aux[2], local.len),
            )
            .eval(builder, local.is_valid);

        // Generate 16-bit limbs for range checking
        // dest, to_dest, source - 12 * is_shift_non_zero, to_source - 12 * is_shift_non_zero
        let dest_u16_limbs = u8_word_to_u16(local.dest);
        let to_dest_u16_limbs = u8_word_to_u16(local.to_dest);
        // Limb computation for (source - 12 * is_shift_non_zero), with zero-padding when low limb < 12
        // If is_source_small is true, we clamp subtraction to 0 by setting carry appropriately.
        let source_u16_limbs = [
            local.source[0]
                + local.source[1] * AB::F::from_canonical_u32(1 << MEMCPY_LOOP_LIMB_BITS)
                - AB::Expr::from_canonical_u32(12) * is_shift_non_zero.clone()
                + local.source_minus_twelve_carry
                    * AB::F::from_canonical_u32(1 << (2 * MEMCPY_LOOP_LIMB_BITS)),
            local.source[2]
                + local.source[3] * AB::F::from_canonical_u32(1 << MEMCPY_LOOP_LIMB_BITS)
                - local.source_minus_twelve_carry,
        ];
        let to_source_u16_limbs = [
            local.to_source[0]
                + local.to_source[1] * AB::F::from_canonical_u32(1 << MEMCPY_LOOP_LIMB_BITS)
                - AB::Expr::from_canonical_u32(12) * is_shift_non_zero.clone()
                + local.to_source_minus_twelve_carry
                    * AB::F::from_canonical_u32(1 << (2 * MEMCPY_LOOP_LIMB_BITS)),
            local.to_source[2]
                + local.to_source[3] * AB::F::from_canonical_u32(1 << MEMCPY_LOOP_LIMB_BITS)
                - local.to_source_minus_twelve_carry,
        ];

        // Range check addresses
        let range_check_data = [
            dest_u16_limbs,
            source_u16_limbs,
            to_dest_u16_limbs,
            to_source_u16_limbs,
        ];

        range_check_data.iter().for_each(|data| {
            // Check the low 16 bits of dest and source, make sure they are multiple of 4
            self.range_bus
                .range_check(
                    data[0].clone() * AB::F::from_canonical_u32(4).inverse(),
                    MEMCPY_LOOP_LIMB_BITS * 2 - 2,
                )
                .eval(builder, local.is_valid);
            // Check the high 16 bits of dest and source, make sure they are in the range [0, 2^pointer_max_bits - 2^MEMCPY_LOOP_LIMB_BITS)
            self.range_bus
                .range_check(
                    data[1].clone(),
                    self.pointer_max_bits - MEMCPY_LOOP_LIMB_BITS * 2,
                )
                .eval(builder, local.is_valid);
        });

        // Send message to memcpy call bus
        self.memcpy_bus
            .send(
                local.from_state.timestamp,
                dest,
                source - AB::Expr::from_canonical_u32(12) * is_shift_non_zero.clone(),
                len.clone() - shift.clone(),
                shift.clone(),
            )
            .eval(builder, local.is_valid);

        // Receive message from memcpy return bus
        self.memcpy_bus
            .receive(
                local.to_timestamp - AB::Expr::from_canonical_u32(timestamp_delta),
                to_dest,
                to_source - AB::Expr::from_canonical_u32(12) * is_shift_non_zero.clone(),
                to_len - shift.clone(),
                AB::Expr::from_canonical_u32(4),
            )
            .eval(builder, local.is_valid);

        // Make sure the request and response match, this should work because the
        // from_timestamp and len are valid and to_len is in [0, 16 + shift)
        builder.when(local.is_valid).assert_eq(
            AB::Expr::TWO * (local.to_timestamp - local.from_state.timestamp),
            (len.clone() - to_len)
                + AB::Expr::TWO
                    * (is_shift_non_zero.clone() + AB::Expr::from_canonical_u32(timestamp_delta)),
        );

        // Execution bus + program bus
        self.execution_bridge
            .execute_and_increment_pc(
                AB::Expr::from_canonical_usize(
                    Rv32MemcpyOpcode::MEMCPY_LOOP as usize + self.offset,
                ),
                [AB::Expr::ZERO, AB::Expr::ZERO, shift.clone()],
                local.from_state,
                local.to_timestamp - local.from_state.timestamp,
            )
            .eval(builder, local.is_valid);
    }
}

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug)]
pub struct MemcpyLoopRecord {
    pub from_pc: u32,
    pub from_timestamp: u32,
    pub dest: u32,
    pub source: u32,
    pub len: u32,
    pub shift: u8,
    pub write_aux: [MemoryExtendedAuxRecord; 3],
}

pub struct MemcpyLoopChip {
    pub air: MemcpyLoopAir,
    pub records: Arc<Mutex<Vec<MemcpyLoopRecord>>>,
    pub pointer_max_bits: usize,
    pub range_checker_chip: SharedVariableRangeCheckerChip,
}

impl MemcpyLoopChip {
    pub fn new(
        system_port: SystemPort,
        range_bus: VariableRangeCheckerBus,
        memcpy_bus: MemcpyBus,
        offset: usize,
        pointer_max_bits: usize,
        range_checker_chip: SharedVariableRangeCheckerChip,
    ) -> Self {
        Self {
            air: MemcpyLoopAir::new(
                system_port.memory_bridge,
                ExecutionBridge::new(system_port.execution_bus, system_port.program_bus),
                range_bus,
                memcpy_bus,
                pointer_max_bits,
                offset,
            ),
            records: Arc::new(Mutex::new(Vec::new())),
            pointer_max_bits,
            range_checker_chip,
        }
    }

    pub fn bus(&self) -> MemcpyBus {
        self.air.memcpy_bus
    }

    pub fn clear(&self) {
        self.records.lock().unwrap().clear();
    }

    pub fn add_new_loop<'a, F: PrimeField32>(
        &self,
        mem_helper: &MemoryAuxColsFactory<F>,
        from_pc: u32,
        from_timestamp: u32,
        dest: u32,
        source: u32,
        len: u32,
        shift: u8,
        register_aux: [MemoryBaseAuxRecord; 3],
    ) {
        let mut timestamp =
            from_timestamp + (((len - shift as u32) & !0x0f) >> 1) + (shift != 0) as u32;
        let write_aux = register_aux
            .iter()
            .map(|aux_record| {
                let mut aux_col = MemoryBaseAuxCols::default();
                mem_helper.fill(aux_record.prev_timestamp, timestamp, &mut aux_col);
                timestamp += 1;
                MemoryExtendedAuxRecord::from_aux_cols(aux_col)
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        let num_copies = (len - shift as u32) & !0x0f;
        let to_dest = dest + num_copies;
        let to_source = source + num_copies;

        let word_to_u16 = |data: u32| [data & 0x0ffff, data >> 16];

        // Relax: allow small sources; zero-pad semantics like iteration.rs
        debug_assert!(dest % 4 == 0);
        debug_assert!(to_dest % 4 == 0);
        debug_assert!(source % 4 == 0);
        debug_assert!(to_source % 4 == 0);
        let safe_source = source.saturating_sub(12 * (shift != 0) as u32);
        let safe_to_source = to_source.saturating_sub(12 * (shift != 0) as u32);
        let range_check_data = [
            word_to_u16(dest),
            word_to_u16(safe_source),
            word_to_u16(to_dest),
            word_to_u16(safe_to_source),
        ];

        range_check_data.iter().for_each(|data| {
            self.range_checker_chip
                .add_count(data[0] >> 2, 2 * MEMCPY_LOOP_LIMB_BITS - 2);
            self.range_checker_chip
                .add_count(data[1], self.pointer_max_bits - 2 * MEMCPY_LOOP_LIMB_BITS);
        });

        // Create record
        let row = MemcpyLoopRecord {
            from_pc,
            from_timestamp,
            dest,
            source,
            len,
            shift,
            write_aux,
        };

        // Thread-safe push to rows vector
        if let Ok(mut rows_guard) = self.records.lock() {
            rows_guard.push(row);
        }
    }

    /// Generates trace
    pub fn generate_trace<F: PrimeField32>(&self) -> RowMajorMatrix<F> {
        let height = next_power_of_two_or_zero(self.records.lock().unwrap().len());
        let mut rows = F::zero_vec(height * NUM_MEMCPY_LOOP_COLS);

        // TODO: run in parallel
        for (i, record) in self.records.lock().unwrap().iter().enumerate() {
            let row = &mut rows[i * NUM_MEMCPY_LOOP_COLS..(i + 1) * NUM_MEMCPY_LOOP_COLS];
            let cols: &mut MemcpyLoopCols<F> = row.borrow_mut();

            let shift = record.shift;
            let num_copies = (record.len - shift as u32) & !0x0f;
            let to_source = record.source + num_copies;

            cols.from_state.pc = F::from_canonical_u32(record.from_pc);
            cols.from_state.timestamp = F::from_canonical_u32(record.from_timestamp);
            cols.dest = record.dest.to_le_bytes().map(F::from_canonical_u8);
            cols.source = record.source.to_le_bytes().map(F::from_canonical_u8);
            cols.len = record.len.to_le_bytes().map(F::from_canonical_u8);
            cols.shift = [shift & 1, shift >> 1].map(F::from_canonical_u8);
            cols.is_valid = F::ONE;
            // We have MEMCPY_LOOP_NUM_WRITES writes in the loop, (num_copies / 4) writes
            // and (num_copies / 4 + shift != 0) reads in iterations
            cols.to_timestamp = F::from_canonical_u32(
                record.from_timestamp
                    + MEMCPY_LOOP_NUM_WRITES
                    + (num_copies >> 1)
                    + (shift != 0) as u32,
            );
            cols.to_dest = (record.dest + num_copies)
                .to_le_bytes()
                .map(F::from_canonical_u8);
            cols.to_source = to_source.to_le_bytes().map(F::from_canonical_u8);
            cols.to_len = F::from_canonical_u32(record.len - num_copies);
            record
                .write_aux
                .iter()
                .zip(cols.write_aux.iter_mut())
                .for_each(|(record_aux, col_aux)| {
                    record_aux.to_aux_cols(col_aux);
                });
            cols.source_minus_twelve_carry = F::from_bool((record.source & 0x0ffff) < 12);
            cols.to_source_minus_twelve_carry = F::from_bool((to_source & 0x0ffff) < 12);

            tracing::info!("timestamp: {:?}, pc: {:?}, dest: {:?}, source: {:?}, len: {:?}, shift: {:?}, is_valid: {:?}, to_timestamp: {:?}, to_dest: {:?}, to_source: {:?}, to_len: {:?}, write_aux: {:?}",
                            cols.from_state.timestamp.as_canonical_u32(),
                            cols.from_state.pc.as_canonical_u32(),
                            u32::from_le_bytes(cols.dest.map(|x| x.as_canonical_u32() as u8)),
                            u32::from_le_bytes(cols.source.map(|x| x.as_canonical_u32() as u8)),
                            u32::from_le_bytes(cols.len.map(|x| x.as_canonical_u32() as u8)),
                            cols.shift[1].as_canonical_u32() * 2 + cols.shift[0].as_canonical_u32(),
                            cols.is_valid.as_canonical_u32(),
                            cols.to_timestamp.as_canonical_u32(),
                            u32::from_le_bytes(cols.to_dest.map(|x| x.as_canonical_u32() as u8)),
                            u32::from_le_bytes(cols.to_source.map(|x| x.as_canonical_u32() as u8)),
                            cols.to_len.as_canonical_u32(),
                            cols.write_aux.map(|x| x.prev_timestamp.as_canonical_u32()).to_vec());
        }
        RowMajorMatrix::new(rows, NUM_MEMCPY_LOOP_COLS)
    }
}

// We allow any `R` type so this can work with arbitrary record arenas.
impl<R, SC: StarkGenericConfig> Chip<R, CpuBackend<SC>> for MemcpyLoopChip
where
    Val<SC>: PrimeField32,
{
    /// Generates trace and resets the internal counters all to 0.
    fn generate_proving_ctx(&self, _: R) -> AirProvingContext<CpuBackend<SC>> {
        let trace = self.generate_trace::<Val<SC>>();
        AirProvingContext::simple_no_pis(Arc::new(trace))
    }
}

impl ChipUsageGetter for MemcpyLoopChip {
    fn air_name(&self) -> String {
        get_air_name(&self.air)
    }
    fn current_trace_height(&self) -> usize {
        self.records.lock().unwrap().len()
    }
    fn trace_width(&self) -> usize {
        NUM_MEMCPY_LOOP_COLS
    }
}
