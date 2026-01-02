//! Trace generation for multirow EC_MUL chip
//!
//! ## Components
//!
//! - `EcMulExecutor`: Executes the EC_MUL instruction, reads/writes memory, records data
//! - `EcMulFiller`: Fills the trace matrix with 257 rows per instruction
//!
//! ## Trace Generation Flow
//!
//! 1. **Execution Phase** (`EcMulExecutor::execute`):
//!    - Read 3 register pointers (rd, rs1, rs2)
//!    - Read scalar (32 bytes) and base point (2 coordinates)
//!    - Call native EC multiplication (k256 library)
//!    - Write result to memory
//!    - Store all data in the record
//!
//! 2. **Filling Phase** (`EcMulFiller::fill_trace`):
//!    - Parse record from trace buffer
//!    - **IMPORTANT**: Copy record data BEFORE zeroing (timestamp fix)
//!    - Zero the trace
//!    - Fill 256 compute rows with FieldExpr witnesses
//!    - Fill digest row with memory I/O data
//!
//! ## Record Layout
//!
//! The executor stores data at the start of the trace chunk:
//! ```text
//! [EcMulRecordHeader | scalar_data | basepoint_data | result_data]
//! ```
//!
//! ## Known Issues
//!
//! ### Timestamp Ordering (FIXED)
//! Previously, the filler zeroed the trace before reading the record, causing
//! `prev_timestamp >= timestamp` errors. Fixed by copying record data first.
//!
//! ### FieldExpr Columns on Digest Row (OPEN)
//! The filler sets FieldExpr columns only on compute rows. For digest rows,
//! the columns at offset `base_width` contain digest-specific data, not FieldExpr.
//! This causes AIR constraint failures. See `columns.rs` and `air.rs` for details.

use std::{borrow::BorrowMut, mem::size_of, sync::Arc};

use num_bigint::BigUint;
use openvm_circuit::{
    arch::*,
    system::memory::{
        offline_checker::{MemoryReadAuxRecord, MemoryWriteBytesAuxRecord},
        online::TracingMemory,
        MemoryAuxColsFactory,
    },
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::SharedBitwiseOperationLookupChip, encoder::Encoder,
    var_range::VariableRangeCheckerChip, AlignedBytesBorrow, TraceSubRowGenerator,
};
use openvm_ecc_transpiler::Rv32WeierstrassOpcode;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_CELL_BITS, RV32_MEMORY_AS, RV32_REGISTER_AS, RV32_REGISTER_NUM_LIMBS},
};
use openvm_mod_circuit_builder::FieldExpr;
use openvm_rv32im_circuit::adapters::{tracing_read, tracing_write};
use openvm_stark_backend::{
    p3_air::BaseAir, p3_field::PrimeField32, p3_matrix::dense::RowMajorMatrix,
    p3_maybe_rayon::prelude::*,
};

use super::{
    ec_mul_compute_base_width, ec_mul_digest_width, EcMulComputeCols, EcMulDigestCols,
    EcMulExecutor, EC_MUL_REGISTER_READS, EC_MUL_SCALAR_BITS, EC_MUL_SCALAR_BYTES,
    EC_MUL_TOTAL_ROWS,
};
use crate::weierstrass_chip::curves::{ec_mul, get_curve_type, CurveType};

// ===== Metadata and Layout =====

/// Metadata for EC_MUL instruction (257 rows: 256 compute + 1 digest)
#[derive(Clone, Copy)]
pub struct EcMulMetadata;

impl MultiRowMetadata for EcMulMetadata {
    fn get_num_rows(&self) -> usize {
        EC_MUL_TOTAL_ROWS
    }
}

pub type EcMulRecordLayout = MultiRowLayout<EcMulMetadata>;

// ===== Record Structures =====

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug, Clone)]
pub struct EcMulRecordHeader<const NUM_BLOCKS: usize, const BLOCK_SIZE: usize> {
    pub from_pc: u32,
    pub timestamp: u32,
    /// True if this is SETUP_EC_MUL opcode, false for EC_MUL
    pub is_setup: u32,

    pub rd_ptr: u32,
    pub rs1_ptr: u32,
    pub rs2_ptr: u32,

    pub dst_ptr: u32,
    pub basepoint_ptr: u32,
    pub scalar_ptr: u32,

    pub rs_read_aux: [MemoryReadAuxRecord; EC_MUL_REGISTER_READS],
    pub reads_point_aux: [MemoryReadAuxRecord; NUM_BLOCKS],
    pub reads_scalar_aux: MemoryReadAuxRecord,
    pub writes_aux: [MemoryWriteBytesAuxRecord<BLOCK_SIZE>; NUM_BLOCKS],
}

pub struct EcMulRecordMut<'a, const NUM_BLOCKS: usize, const BLOCK_SIZE: usize> {
    pub inner: &'a mut EcMulRecordHeader<NUM_BLOCKS, BLOCK_SIZE>,
    pub scalar_data: &'a mut [u8],
    pub basepoint_data: &'a mut [u8],
    pub result_data: &'a mut [u8],
}

impl<'a, const NUM_BLOCKS: usize, const BLOCK_SIZE: usize>
    CustomBorrow<'a, EcMulRecordMut<'a, NUM_BLOCKS, BLOCK_SIZE>, EcMulRecordLayout> for [u8]
{
    fn custom_borrow(
        &'a mut self,
        _layout: EcMulRecordLayout,
    ) -> EcMulRecordMut<'a, NUM_BLOCKS, BLOCK_SIZE> {
        let (header_buf, rest) = unsafe {
            self.split_at_mut_unchecked(size_of::<EcMulRecordHeader<NUM_BLOCKS, BLOCK_SIZE>>())
        };

        let (scalar_buf, rest) = unsafe { rest.split_at_mut_unchecked(EC_MUL_SCALAR_BYTES) };
        let (basepoint_buf, rest) = unsafe { rest.split_at_mut_unchecked(NUM_BLOCKS * BLOCK_SIZE) };
        let (result_buf, _rest) = unsafe { rest.split_at_mut_unchecked(NUM_BLOCKS * BLOCK_SIZE) };

        EcMulRecordMut {
            inner: header_buf.borrow_mut(),
            scalar_data: scalar_buf,
            basepoint_data: basepoint_buf,
            result_data: result_buf,
        }
    }

    unsafe fn extract_layout(&self) -> EcMulRecordLayout {
        EcMulRecordLayout {
            metadata: EcMulMetadata,
        }
    }
}

impl<const NUM_BLOCKS: usize, const BLOCK_SIZE: usize> SizedRecord<EcMulRecordLayout>
    for EcMulRecordMut<'_, NUM_BLOCKS, BLOCK_SIZE>
{
    fn size(_layout: &EcMulRecordLayout) -> usize {
        ec_mul_record_size::<NUM_BLOCKS, BLOCK_SIZE>()
    }

    fn alignment(_layout: &EcMulRecordLayout) -> usize {
        ec_mul_record_alignment::<NUM_BLOCKS, BLOCK_SIZE>()
    }
}

/// Helper functions for record size/alignment (shared between EC_MUL and SETUP layouts)
const fn ec_mul_record_size<const NUM_BLOCKS: usize, const BLOCK_SIZE: usize>() -> usize {
    size_of::<EcMulRecordHeader<NUM_BLOCKS, BLOCK_SIZE>>()
        + EC_MUL_SCALAR_BYTES
        + NUM_BLOCKS * BLOCK_SIZE * 2 // basepoint + result
}

const fn ec_mul_record_alignment<const NUM_BLOCKS: usize, const BLOCK_SIZE: usize>() -> usize {
    align_of::<EcMulRecordHeader<NUM_BLOCKS, BLOCK_SIZE>>()
}

// ===== Executor Implementation =====

impl<F, RA, const NUM_BLOCKS: usize, const BLOCK_SIZE: usize> PreflightExecutor<F, RA>
    for EcMulExecutor<NUM_BLOCKS, BLOCK_SIZE>
where
    F: PrimeField32,
    for<'buf> RA:
        RecordArena<'buf, EcMulRecordLayout, EcMulRecordMut<'buf, NUM_BLOCKS, BLOCK_SIZE>>,
{
    fn get_opcode_name(&self, _: usize) -> String {
        format!("{:?}", Rv32WeierstrassOpcode::EC_MUL)
    }

    fn execute(
        &self,
        state: VmStateMut<F, TracingMemory, RA>,
        instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError> {
        let Instruction {
            opcode,
            a,
            b,
            c,
            d,
            e,
            ..
        } = instruction;

        debug_assert_eq!(d.as_canonical_u32(), RV32_REGISTER_AS);
        debug_assert_eq!(e.as_canonical_u32(), RV32_MEMORY_AS);

        // Determine if this is a setup operation
        let local_opcode = opcode.local_opcode_idx(self.offset);
        let is_setup = local_opcode == Rv32WeierstrassOpcode::SETUP_EC_MUL as usize;

        let record = state.ctx.alloc(EcMulRecordLayout {
            metadata: EcMulMetadata,
        });

        record.inner.from_pc = *state.pc;
        record.inner.timestamp = state.memory.timestamp();
        record.inner.is_setup = if is_setup { 1 } else { 0 };
        record.inner.rd_ptr = a.as_canonical_u32();
        record.inner.rs1_ptr = b.as_canonical_u32();
        record.inner.rs2_ptr = c.as_canonical_u32();

        // Read register pointers
        record.inner.dst_ptr = u32::from_le_bytes(tracing_read(
            state.memory,
            RV32_REGISTER_AS,
            record.inner.rd_ptr,
            &mut record.inner.rs_read_aux[0].prev_timestamp,
        ));

        record.inner.basepoint_ptr = u32::from_le_bytes(tracing_read(
            state.memory,
            RV32_REGISTER_AS,
            record.inner.rs1_ptr,
            &mut record.inner.rs_read_aux[1].prev_timestamp,
        ));

        record.inner.scalar_ptr = u32::from_le_bytes(tracing_read(
            state.memory,
            RV32_REGISTER_AS,
            record.inner.rs2_ptr,
            &mut record.inner.rs_read_aux[2].prev_timestamp,
        ));

        // Read base point (NUM_BLOCKS blocks of BLOCK_SIZE bytes)
        for i in 0..NUM_BLOCKS {
            let block: [u8; BLOCK_SIZE] = tracing_read(
                state.memory,
                RV32_MEMORY_AS,
                record.inner.basepoint_ptr + (i * BLOCK_SIZE) as u32,
                &mut record.inner.reads_point_aux[i].prev_timestamp,
            );
            record.basepoint_data[i * BLOCK_SIZE..(i + 1) * BLOCK_SIZE].copy_from_slice(&block);
        }

        // Read scalar (32 bytes)
        let scalar: [u8; EC_MUL_SCALAR_BYTES] = tracing_read(
            state.memory,
            RV32_MEMORY_AS,
            record.inner.scalar_ptr,
            &mut record.inner.reads_scalar_aux.prev_timestamp,
        );
        record.scalar_data.copy_from_slice(&scalar);

        // Compute EC multiplication
        if let Some(curve_type) = {
            let modulus = &self.expr.builder.prime;
            let a_coeff = &self.expr.setup_values[0];
            get_curve_type(modulus, a_coeff)
        } {
            match curve_type {
                CurveType::K256 => {
                    let result = ec_mul_solve::<{ CurveType::K256 as u8 }, NUM_BLOCKS, BLOCK_SIZE>(
                        record.scalar_data,
                        record.basepoint_data,
                    );
                    record.result_data.copy_from_slice(&result);
                }
                CurveType::P256 => {
                    let result = ec_mul_solve::<{ CurveType::P256 as u8 }, NUM_BLOCKS, BLOCK_SIZE>(
                        record.scalar_data,
                        record.basepoint_data,
                    );
                    record.result_data.copy_from_slice(&result);
                }
                CurveType::BN254 => {
                    let result = ec_mul_solve::<{ CurveType::BN254 as u8 }, NUM_BLOCKS, BLOCK_SIZE>(
                        record.scalar_data,
                        record.basepoint_data,
                    );
                    record.result_data.copy_from_slice(&result);
                }
                CurveType::BLS12_381 => {
                    let result = ec_mul_solve::<
                        { CurveType::BLS12_381 as u8 },
                        NUM_BLOCKS,
                        BLOCK_SIZE,
                    >(record.scalar_data, record.basepoint_data);
                    record.result_data.copy_from_slice(&result);
                }
            }
        } else {
            panic!("Unsupported curve type");
        }

        // Write result (NUM_BLOCKS blocks of BLOCK_SIZE bytes)
        for i in 0..NUM_BLOCKS {
            let block: [u8; BLOCK_SIZE] = record.result_data[i * BLOCK_SIZE..(i + 1) * BLOCK_SIZE]
                .try_into()
                .unwrap();
            tracing_write(
                state.memory,
                RV32_MEMORY_AS,
                record.inner.dst_ptr + (i * BLOCK_SIZE) as u32,
                block,
                &mut record.inner.writes_aux[i].prev_timestamp,
                &mut record.inner.writes_aux[i].prev_data,
            );
        }

        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);

        Ok(())
    }
}

// ===== Native EC Solver =====

/// Performs EC scalar multiplication using the native curve implementation.
pub fn ec_mul_solve<const CURVE_TYPE: u8, const NUM_BLOCKS: usize, const BLOCK_SIZE: usize>(
    scalar: &[u8],
    basepoint: &[u8],
) -> Vec<u8> {
    // Convert flat slices to arrays of arrays for ec_mul
    let mut point_data = [[0u8; BLOCK_SIZE]; NUM_BLOCKS];
    for i in 0..NUM_BLOCKS {
        point_data[i].copy_from_slice(&basepoint[i * BLOCK_SIZE..(i + 1) * BLOCK_SIZE]);
    }

    // Scalar is always 32 bytes
    let mut scalar_data = [0u8; 32];
    scalar_data.copy_from_slice(scalar);

    // Call the native EC multiplication
    let output = ec_mul::<CURVE_TYPE, NUM_BLOCKS, BLOCK_SIZE>(point_data, scalar_data);

    // Convert back to flat Vec<u8>
    output.as_flattened().to_vec()
}

// ===== Trace Filler =====

pub struct EcMulFiller<const NUM_LIMBS: usize, const NUM_BLOCKS: usize, const BLOCK_SIZE: usize> {
    pub ec_mul_step_expr: FieldExpr,
    pub local_opcode_idx: Vec<usize>,
    pub opcode_flag_idx: Vec<usize>,
    pub range_checker: Arc<VariableRangeCheckerChip>,
    pub bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
    pub row_idx_encoder: Encoder,
    pub pointer_max_bits: usize,
    pub should_finalize: bool,
    /// Scalar field modulus (curve order) for setup validation
    pub scalar_biguint: BigUint,
}

impl<F: PrimeField32, const NUM_LIMBS: usize, const NUM_BLOCKS: usize, const BLOCK_SIZE: usize>
    TraceFiller<F> for EcMulFiller<NUM_LIMBS, NUM_BLOCKS, BLOCK_SIZE>
{
    fn fill_trace(
        &self,
        mem_helper: &MemoryAuxColsFactory<F>,
        trace_matrix: &mut RowMajorMatrix<F>,
        rows_used: usize,
    ) {
        if rows_used == 0 {
            return;
        }

        let width = trace_matrix.width;
        let compute_base_width = ec_mul_compute_base_width::<NUM_LIMBS>();
        let expr_width = BaseAir::<F>::width(&self.ec_mul_step_expr);
        let digest_width = ec_mul_digest_width::<NUM_LIMBS, NUM_BLOCKS, BLOCK_SIZE>();

        trace_matrix.values[..rows_used * width]
            .par_chunks_mut(EC_MUL_TOTAL_ROWS * width)
            .for_each(|slice| {
                // SAFETY:
                // - caller ensures `slice` contains a valid record representation that was
                //   previously written by the executor
                // - get_record_from_slice will correctly split the buffer into header and data
                //   components based on the layout
                // IMPORTANT: Read record FIRST before zeroing!
                let mut slice_ptr = slice as &mut [F];
                let record: EcMulRecordMut<NUM_BLOCKS, BLOCK_SIZE> = unsafe {
                    get_record_from_slice(
                        &mut slice_ptr,
                        EcMulRecordLayout {
                            metadata: EcMulMetadata,
                        },
                    )
                };

                // ================================================================
                // TIMESTAMP FIX: Copy record data BEFORE zeroing the trace
                // ================================================================
                //
                // The record is stored at the beginning of the trace chunk, overlapping
                // with the first few rows. If we zero the trace first, we destroy the
                // record data including `timestamp`, causing `prev_timestamp >= timestamp`
                // errors in the memory controller.
                //
                // Solution (following SHA256's pattern):
                // 1. Copy all record data to local variables
                // 2. Zero the trace
                // 3. Fill the trace using the copied data
                //
                // This ensures the record's timestamp is preserved for trace filling.
                // ================================================================
                let scalar_bytes: Vec<u8> = record.scalar_data.to_vec();
                let basepoint_bytes: Vec<u8> = record.basepoint_data.to_vec();
                let vm_record = record.inner.clone();

                // NOW zero the trace (safe because we have copies)
                unsafe {
                    std::ptr::write_bytes(
                        slice.as_mut_ptr() as *mut u8,
                        0,
                        EC_MUL_TOTAL_ROWS * width * size_of::<F>(),
                    );
                }

                self.fill_instruction_trace::<F>(
                    slice,
                    width,
                    compute_base_width,
                    expr_width,
                    digest_width,
                    &vm_record,
                    &scalar_bytes,
                    &basepoint_bytes,
                    mem_helper,
                );
            });
    }
}

impl<const NUM_LIMBS: usize, const NUM_BLOCKS: usize, const BLOCK_SIZE: usize>
    EcMulFiller<NUM_LIMBS, NUM_BLOCKS, BLOCK_SIZE>
{
    #[allow(clippy::too_many_arguments)]
    fn fill_instruction_trace<F: PrimeField32>(
        &self,
        chunk: &mut [F],
        width: usize,
        compute_base_width: usize,
        expr_width: usize,
        digest_width: usize,
        record: &EcMulRecordHeader<NUM_BLOCKS, BLOCK_SIZE>,
        scalar_bytes: &[u8],
        basepoint_bytes: &[u8],
        mem_helper: &MemoryAuxColsFactory<F>,
    ) {
        let scalar_bits = bytes_to_bits(scalar_bytes);
        let (base_px_limbs, base_py_limbs) =
            blocks_to_coordinate_limbs::<F, NUM_LIMBS>(basepoint_bytes, NUM_BLOCKS, BLOCK_SIZE);

        let mut current_rx = vec![F::ZERO; NUM_LIMBS];
        let mut current_ry = vec![F::ZERO; NUM_LIMBS];

        // Track whether the accumulator R is at infinity
        // is_inf = 1 on the first row (R starts at infinity)
        // is_inf_next = is_inf * (1 - bit) (stays at infinity only if bit = 0)
        let mut is_inf = true;

        // Fill compute rows (0-255)
        for row_idx in 0..256 {
            let row = &mut chunk[row_idx * width..(row_idx + 1) * width];
            let compute_cols: &mut EcMulComputeCols<F, NUM_LIMBS> =
                row[..compute_base_width].borrow_mut();

            // Extract is_setup from record
            let is_setup = record.is_setup != 0;

            compute_cols.flags.is_compute_row = F::ONE;
            compute_cols.flags.is_digest_row = F::ZERO;
            compute_cols.flags.is_first_compute_row = if row_idx == 0 { F::ONE } else { F::ZERO };
            compute_cols.flags.is_setup = if is_setup { F::ONE } else { F::ZERO };
            compute_cols.flags.is_inf = if is_inf { F::ONE } else { F::ZERO };
            set_row_idx(
                &self.row_idx_encoder,
                &mut compute_cols.flags.row_idx,
                row_idx,
            );

            for (i, &bit) in scalar_bits.iter().enumerate() {
                compute_cols.control.scalar_bits[i] = if bit { F::ONE } else { F::ZERO };
            }
            compute_cols
                .control
                .base_px_limbs
                .copy_from_slice(&base_px_limbs);
            compute_cols
                .control
                .base_py_limbs
                .copy_from_slice(&base_py_limbs);
            compute_cols.control.rx_limbs.copy_from_slice(&current_rx);
            compute_cols.control.ry_limbs.copy_from_slice(&current_ry);

            let bit_idx = 255 - row_idx;
            let scalar_bit = scalar_bits[bit_idx];

            let rx_biguint = limbs_to_biguint(&current_rx);
            let ry_biguint = limbs_to_biguint(&current_ry);
            let px_biguint = limbs_to_biguint(&base_px_limbs);
            let py_biguint = limbs_to_biguint(&base_py_limbs);

            let inputs = vec![rx_biguint, ry_biguint, px_biguint, py_biguint];
            // flags[0] = scalar bit, flags[1] = use_safe_denom
            // Use safe denominators when is_inf = true OR is_setup = true
            // This prevents division by zero when R is at infinity
            let use_safe_denom = is_setup || is_inf;
            let flags = vec![scalar_bit, use_safe_denom];

            self.ec_mul_step_expr.generate_subrow(
                (self.range_checker.as_ref(), inputs, flags),
                &mut row[compute_base_width..compute_base_width + expr_width],
            );

            let field_cols = self
                .ec_mul_step_expr
                .load_vars(&row[compute_base_width..compute_base_width + expr_width]);
            let output_indices = self.ec_mul_step_expr.output_indices();

            current_rx[..NUM_LIMBS]
                .copy_from_slice(&field_cols.vars[output_indices[0]][..NUM_LIMBS]);
            current_ry[..NUM_LIMBS]
                .copy_from_slice(&field_cols.vars[output_indices[1]][..NUM_LIMBS]);

            // Update is_inf for the next row
            // is_inf_next = is_inf && !scalar_bit
            // R stays at infinity only if R was at infinity AND we didn't add P (bit = 0)
            is_inf = is_inf && !scalar_bit;
        }

        // Fill digest row (256)
        let digest_row = &mut chunk[256 * width..257 * width];
        let digest_cols: &mut EcMulDigestCols<F, NUM_LIMBS, NUM_BLOCKS, BLOCK_SIZE> =
            digest_row[..digest_width].borrow_mut();

        // Extract is_setup from record
        let is_setup = record.is_setup != 0;

        digest_cols.flags.is_compute_row = F::ZERO;
        digest_cols.flags.is_digest_row = F::ONE;
        digest_cols.flags.is_first_compute_row = F::ZERO;
        digest_cols.flags.is_setup = if is_setup { F::ONE } else { F::ZERO };
        // is_inf at the digest row is the final state after all 256 compute rows
        // This would be true only if all 256 bits were 0 (scalar = 0)
        digest_cols.flags.is_inf = if is_inf { F::ONE } else { F::ZERO };
        set_row_idx(&self.row_idx_encoder, &mut digest_cols.flags.row_idx, 256);

        for (i, &bit) in scalar_bits.iter().enumerate() {
            digest_cols.control.scalar_bits[i] = if bit { F::ONE } else { F::ZERO };
        }
        digest_cols
            .control
            .base_px_limbs
            .copy_from_slice(&base_px_limbs);
        digest_cols
            .control
            .base_py_limbs
            .copy_from_slice(&base_py_limbs);
        digest_cols.control.rx_limbs.copy_from_slice(&current_rx);
        digest_cols.control.ry_limbs.copy_from_slice(&current_ry);

        digest_cols.from_state.pc = F::from_canonical_u32(record.from_pc);
        digest_cols.from_state.timestamp = F::from_canonical_u32(record.timestamp);

        digest_cols.rd_ptr = F::from_canonical_u32(record.rd_ptr);
        digest_cols.rs1_ptr = F::from_canonical_u32(record.rs1_ptr);
        digest_cols.rs2_ptr = F::from_canonical_u32(record.rs2_ptr);

        let dst_ptr_limbs = u32_to_limbs(record.dst_ptr);
        let basepoint_ptr_limbs = u32_to_limbs(record.basepoint_ptr);
        let scalar_ptr_limbs = u32_to_limbs(record.scalar_ptr);

        for i in 0..RV32_REGISTER_NUM_LIMBS {
            digest_cols.dst_ptr[i] = F::from_canonical_u8(dst_ptr_limbs[i]);
            digest_cols.basepoint_ptr[i] = F::from_canonical_u8(basepoint_ptr_limbs[i]);
            digest_cols.scalar_ptr[i] = F::from_canonical_u8(scalar_ptr_limbs[i]);
        }

        // Request range checks for pointer MSLs (matching AIR send_range calls)
        let limb_shift_bits = RV32_CELL_BITS * RV32_REGISTER_NUM_LIMBS - self.pointer_max_bits;
        const MSL_SHIFT: u32 = (RV32_CELL_BITS * (RV32_REGISTER_NUM_LIMBS - 1)) as u32;
        // Pair 1: (basepoint_ptr MSL, scalar_ptr MSL)
        self.bitwise_lookup_chip.request_range(
            (record.basepoint_ptr >> MSL_SHIFT) << limb_shift_bits as u32,
            (record.scalar_ptr >> MSL_SHIFT) << limb_shift_bits as u32,
        );
        // Pair 2: (dst_ptr MSL, 0)
        self.bitwise_lookup_chip
            .request_range((record.dst_ptr >> MSL_SHIFT) << limb_shift_bits as u32, 0);

        let mut timestamp_delta: u32 = 0;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            record.timestamp + timestamp_delta - 1
        };

        for i in 0..EC_MUL_REGISTER_READS {
            mem_helper.fill(
                record.rs_read_aux[i].prev_timestamp,
                timestamp_pp(),
                digest_cols.rs_read_aux[i].as_mut(),
            );
        }

        for i in 0..NUM_BLOCKS {
            mem_helper.fill(
                record.reads_point_aux[i].prev_timestamp,
                timestamp_pp(),
                digest_cols.reads_point_aux[i].as_mut(),
            );
        }

        mem_helper.fill(
            record.reads_scalar_aux.prev_timestamp,
            timestamp_pp(),
            digest_cols.reads_scalar_aux.as_mut(),
        );

        for i in 0..NUM_BLOCKS {
            digest_cols.writes_aux[i]
                .set_prev_data(record.writes_aux[i].prev_data.map(F::from_canonical_u8));
            mem_helper.fill(
                record.writes_aux[i].prev_timestamp,
                timestamp_pp(),
                digest_cols.writes_aux[i].as_mut(),
            );
        }

        for (dst, &src) in digest_cols.scalar_data.iter_mut().zip(scalar_bytes.iter()) {
            *dst = F::from_canonical_u8(src);
        }

        for block_idx in 0..NUM_BLOCKS {
            for byte_idx in 0..BLOCK_SIZE {
                digest_cols.basepoint_data[block_idx][byte_idx] =
                    F::from_canonical_u8(basepoint_bytes[block_idx * BLOCK_SIZE + byte_idx]);
            }
        }

        let result_blocks =
            coordinate_limbs_to_blocks::<F>(&current_rx, &current_ry, NUM_BLOCKS, BLOCK_SIZE);

        for block_idx in 0..NUM_BLOCKS {
            for byte_idx in 0..BLOCK_SIZE {
                digest_cols.result_data[block_idx][byte_idx] =
                    F::from_canonical_u8(result_blocks[block_idx * BLOCK_SIZE + byte_idx]);
            }
        }
    }
}

// ===== Helper Functions =====

fn bytes_to_bits(bytes: &[u8]) -> [bool; EC_MUL_SCALAR_BITS] {
    let mut bits = [false; EC_MUL_SCALAR_BITS];
    for (i, &byte) in bytes[..EC_MUL_SCALAR_BYTES].iter().enumerate() {
        for bit_idx in 0..8 {
            bits[i * 8 + bit_idx] = (byte >> bit_idx) & 1 == 1;
        }
    }
    bits
}

fn set_row_idx<F: PrimeField32>(encoder: &Encoder, row_idx_cols: &mut [F; 22], value: usize) {
    let pt = encoder.get_flag_pt(value);
    for (col, &coord) in row_idx_cols.iter_mut().zip(pt.iter()) {
        *col = F::from_canonical_u32(coord);
    }
}

fn u32_to_limbs(value: u32) -> [u8; RV32_REGISTER_NUM_LIMBS] {
    value.to_le_bytes()
}

fn limbs_to_biguint<F: PrimeField32>(limbs: &[F]) -> BigUint {
    limbs
        .iter()
        .enumerate()
        .map(|(i, &l)| BigUint::from(l.as_canonical_u64()) << (i * 8))
        .fold(BigUint::from(0u32), |acc, x| acc + x)
}

fn blocks_to_coordinate_limbs<F: PrimeField32, const NUM_LIMBS: usize>(
    blocks: &[u8],
    num_blocks: usize,
    block_size: usize,
) -> (Vec<F>, Vec<F>) {
    let block_per_coord = num_blocks / 2;
    let mut px_limbs = Vec::with_capacity(NUM_LIMBS);
    let mut py_limbs = Vec::with_capacity(NUM_LIMBS);

    for block_idx in 0..block_per_coord {
        for byte_idx in 0..block_size {
            px_limbs.push(F::from_canonical_u8(
                blocks[block_idx * block_size + byte_idx],
            ));
        }
    }

    for block_idx in block_per_coord..num_blocks {
        for byte_idx in 0..block_size {
            py_limbs.push(F::from_canonical_u8(
                blocks[block_idx * block_size + byte_idx],
            ));
        }
    }

    (px_limbs, py_limbs)
}

fn coordinate_limbs_to_blocks<F: PrimeField32>(
    x_limbs: &[F],
    y_limbs: &[F],
    num_blocks: usize,
    block_size: usize,
) -> Vec<u8> {
    let block_per_coord = num_blocks / 2;
    let mut blocks = vec![0u8; num_blocks * block_size];

    for block_idx in 0..block_per_coord {
        for byte_idx in 0..block_size {
            let limb_idx = block_idx * block_size + byte_idx;
            blocks[block_idx * block_size + byte_idx] = x_limbs[limb_idx].as_canonical_u64() as u8;
        }
    }

    for block_idx in 0..block_per_coord {
        for byte_idx in 0..block_size {
            let limb_idx = block_idx * block_size + byte_idx;
            blocks[(block_idx + block_per_coord) * block_size + byte_idx] =
                y_limbs[limb_idx].as_canonical_u64() as u8;
        }
    }

    blocks
}
