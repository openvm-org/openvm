use std::mem::{align_of, size_of};

use openvm_circuit::{
    arch::{
        rvr::{
            ArenaNativeGeometry, ArenaNativeLayout, LogNativeAccessView,
            LogNativeAssemblerRegistry, PreflightMemoryAccessAux, VmRvrLogNativeExtension,
            PREFLIGHT_MEMORY_KIND_READ, PREFLIGHT_MEMORY_KIND_WRITE,
        },
        Arena, ExecutionError, RecordArena, MEMORY_BLOCK_BYTES,
    },
    system::memory::offline_checker::MemoryWriteBytesAuxRecord,
};
use openvm_instructions::{
    instruction::Instruction,
    riscv::{RV64_MEMORY_AS, RV64_REGISTER_AS},
    LocalOpcode,
};
use openvm_keccak256_transpiler::{KeccakfOpcode, XorinOpcode};
use openvm_riscv_circuit::{
    adapters::rv64_u16_block_to_bytes, log_native::Rv64StandardRecordArena,
};
use openvm_stark_backend::p3_field::PrimeField32;
use rvr_openvm_ext_keccak::{KECCAKF_DIRECT_RECORD_SIZE, XORIN_DIRECT_RECORD_SIZE};

use crate::{
    keccakf_op::trace::{KeccakfMetadata, KeccakfRecord, KeccakfRecordLayout},
    xorin::trace::{XorinVmMetadata, XorinVmRecordHeader, XorinVmRecordLayout, XorinVmRecordMut},
    Keccak256, Keccak256Rv64Config, KECCAK_RATE_BYTES, KECCAK_RATE_MEM_OPS, KECCAK_WIDTH_MEM_OPS,
};

const _: () = assert!(size_of::<KeccakfRecord>() == KECCAKF_DIRECT_RECORD_SIZE);
const _: () = assert!(size_of::<XorinVmRecordHeader>() == XORIN_DIRECT_RECORD_SIZE);

pub trait KeccakRecordArena<F>:
    Arena
    + for<'a> RecordArena<'a, KeccakfRecordLayout, &'a mut KeccakfRecord>
    + for<'a> RecordArena<'a, XorinVmRecordLayout, XorinVmRecordMut<'a>>
{
}

impl<F, RA> KeccakRecordArena<F> for RA where
    RA: Arena
        + for<'a> RecordArena<'a, KeccakfRecordLayout, &'a mut KeccakfRecord>
        + for<'a> RecordArena<'a, XorinVmRecordLayout, XorinVmRecordMut<'a>>
{
}

pub trait Keccak256Rv64RecordArena<F>: Rv64StandardRecordArena<F> + KeccakRecordArena<F> {}

impl<F, RA> Keccak256Rv64RecordArena<F> for RA where
    RA: Rv64StandardRecordArena<F> + KeccakRecordArena<F>
{
}

impl<F, RA> VmRvrLogNativeExtension<F, RA> for Keccak256
where
    F: PrimeField32,
    RA: KeccakRecordArena<F>,
{
    fn extend_rvr_log_native(&self, registry: &mut LogNativeAssemblerRegistry<F, RA>) {
        registry.register(
            [KeccakfOpcode::KECCAKF.global_opcode()],
            assemble_keccakf_op::<F, RA>,
        );
        registry.register_inline_arena_native(
            [KeccakfOpcode::KECCAKF.global_opcode()],
            size_of::<KeccakfRecord>(),
            assemble_keccakf_op_inline::<F, RA>,
            ArenaNativeGeometry {
                adapter_size: size_of::<KeccakfRecord>(),
                adapter_align: align_of::<KeccakfRecord>(),
                core_size: 0,
                core_align: 1,
                core_off_matrix: 0,
                layout: ArenaNativeLayout::Custom {
                    residual_memory_chronology: true,
                    max_residual_events_per_record: 26,
                    layout_id: "openvm.rvr.keccakf-final.v1",
                },
            },
        );
        registry.register(
            [XorinOpcode::XORIN.global_opcode()],
            assemble_xorin::<F, RA>,
        );
        registry.register_inline_arena_native(
            [XorinOpcode::XORIN.global_opcode()],
            size_of::<XorinVmRecordHeader>(),
            assemble_xorin_inline::<F, RA>,
            ArenaNativeGeometry {
                adapter_size: size_of::<XorinVmRecordHeader>(),
                adapter_align: align_of::<XorinVmRecordHeader>(),
                core_size: 0,
                core_align: 1,
                core_off_matrix: 0,
                layout: ArenaNativeLayout::Custom {
                    residual_memory_chronology: true,
                    max_residual_events_per_record: 54,
                    layout_id: "openvm.rvr.xorin-final.v1",
                },
            },
        );
    }
}

fn assemble_xorin_inline<F: PrimeField32, RA: KeccakRecordArena<F>>(
    arena: &mut RA,
    _instruction: &Instruction<F>,
    compact: &[u8],
    pc: u32,
) -> Result<(), ExecutionError> {
    if compact.len() != size_of::<XorinVmRecordHeader>() {
        return Err(rvr_error(format!(
            "invalid XORIN inline record size {} at pc {pc:#x}; expected {}",
            compact.len(),
            size_of::<XorinVmRecordHeader>()
        )));
    }
    let record = arena.alloc(XorinVmRecordLayout::new(XorinVmMetadata {}));
    unsafe {
        std::ptr::copy_nonoverlapping(
            compact.as_ptr(),
            (record.inner as *mut XorinVmRecordHeader).cast::<u8>(),
            compact.len(),
        );
    }
    if record.inner.from_pc != pc {
        return Err(rvr_error(format!(
            "XORIN inline record pc mismatch: record={:#x}, program={pc:#x}",
            record.inner.from_pc
        )));
    }
    Ok(())
}

fn assemble_keccakf_op_inline<F: PrimeField32, RA: KeccakRecordArena<F>>(
    arena: &mut RA,
    _instruction: &Instruction<F>,
    compact: &[u8],
    pc: u32,
) -> Result<(), ExecutionError> {
    if compact.len() != size_of::<KeccakfRecord>() {
        return Err(rvr_error(format!(
            "invalid Keccakf inline record size {} at pc {pc:#x}; expected {}",
            compact.len(),
            size_of::<KeccakfRecord>()
        )));
    }
    let record = arena.alloc(KeccakfRecordLayout::new(KeccakfMetadata));
    // The generated C layout is statically asserted against this repr(C)
    // record. Copying through bytes also avoids imposing its alignment on the
    // compact wire slice.
    unsafe {
        std::ptr::copy_nonoverlapping(
            compact.as_ptr(),
            (record as *mut KeccakfRecord).cast::<u8>(),
            compact.len(),
        );
    }
    if record.pc != pc {
        return Err(rvr_error(format!(
            "Keccakf inline record pc mismatch: record={:#x}, program={pc:#x}",
            record.pc
        )));
    }
    Ok(())
}

impl<F, RA> VmRvrLogNativeExtension<F, RA> for Keccak256Rv64Config
where
    F: PrimeField32,
    RA: Keccak256Rv64RecordArena<F>,
{
    fn extend_rvr_log_native(&self, registry: &mut LogNativeAssemblerRegistry<F, RA>) {
        self.rv64i.extend_rvr_log_native(registry);
        self.rv64m.extend_rvr_log_native(registry);
        self.io.extend_rvr_log_native(registry);
        self.keccak.extend_rvr_log_native(registry);
    }
}

pub(crate) fn assemble_keccakf_op<F: PrimeField32, RA: KeccakRecordArena<F>>(
    arena: &mut RA,
    access: &LogNativeAccessView<'_, F>,
    instruction: &Instruction<F>,
    pc: u32,
    timestamp: u32,
) -> Result<(), ExecutionError> {
    let record = arena.alloc(KeccakfRecordLayout::new(KeccakfMetadata));
    record.pc = pc;
    record.timestamp = timestamp;
    record.rd_ptr = instruction.a.as_canonical_u32();

    let rd_aux = expect_reg_read(access, timestamp, record.rd_ptr, pc)?;
    record.rd_aux.prev_timestamp = rd_aux.prev_timestamp;
    record.buffer_ptr = rd_aux.entry.value as u32;

    for (word_idx, aux_record) in record.buffer_word_aux.iter_mut().enumerate() {
        let address = record.buffer_ptr + (word_idx * MEMORY_BLOCK_BYTES) as u32;
        let write_aux = expect_memory_write(access, timestamp + 1 + word_idx as u32, address, pc)?;
        aux_record.prev_timestamp = write_aux.prev_timestamp;
        let start = word_idx * MEMORY_BLOCK_BYTES;
        record.preimage_buffer_bytes[start..start + MEMORY_BLOCK_BYTES]
            .copy_from_slice(&prev_bytes(write_aux));
    }

    Ok(())
}

pub(crate) fn assemble_xorin<F: PrimeField32, RA: KeccakRecordArena<F>>(
    arena: &mut RA,
    access: &LogNativeAccessView<'_, F>,
    instruction: &Instruction<F>,
    pc: u32,
    timestamp: u32,
) -> Result<(), ExecutionError> {
    let record = arena.alloc(XorinVmRecordLayout::new(XorinVmMetadata {}));
    let record = record.inner;
    record.from_pc = pc;
    record.timestamp = timestamp;
    record.rd_ptr = instruction.a.as_canonical_u32();
    record.rs1_ptr = instruction.b.as_canonical_u32();
    record.rs2_ptr = instruction.c.as_canonical_u32();

    let register_ptrs = [record.rd_ptr, record.rs1_ptr, record.rs2_ptr];
    let mut register_values = [0u32; 3];
    for (idx, (reg_ptr, aux_record)) in register_ptrs
        .into_iter()
        .zip(record.register_aux_cols.iter_mut())
        .enumerate()
    {
        let read_aux = expect_reg_read(access, timestamp + idx as u32, reg_ptr, pc)?;
        aux_record.prev_timestamp = read_aux.prev_timestamp;
        register_values[idx] = read_aux.entry.value as u32;
    }
    [record.buffer, record.input, record.len] = register_values;

    let len = record.len as usize;
    if len > KECCAK_RATE_BYTES || !len.is_multiple_of(MEMORY_BLOCK_BYTES) {
        return Err(rvr_error(format!(
            "invalid xorin length {len} at pc {pc:#x}; expected an 8-byte multiple at most {KECCAK_RATE_BYTES}"
        )));
    }
    let num_words = len / MEMORY_BLOCK_BYTES;
    debug_assert!(num_words <= KECCAK_RATE_MEM_OPS);

    let mut next_timestamp = timestamp + 3;
    for word_idx in 0..num_words {
        let address = record.buffer + (word_idx * MEMORY_BLOCK_BYTES) as u32;
        let read_aux = expect_memory_read(access, next_timestamp, address, pc)?;
        record.buffer_read_aux_cols[word_idx].prev_timestamp = read_aux.prev_timestamp;
        let start = word_idx * MEMORY_BLOCK_BYTES;
        record.buffer_limbs[start..start + MEMORY_BLOCK_BYTES]
            .copy_from_slice(&read_aux.entry.value.to_le_bytes());
        next_timestamp += 1;
    }

    for word_idx in 0..num_words {
        let address = record.input + (word_idx * MEMORY_BLOCK_BYTES) as u32;
        let read_aux = expect_memory_read(access, next_timestamp, address, pc)?;
        record.input_read_aux_cols[word_idx].prev_timestamp = read_aux.prev_timestamp;
        let start = word_idx * MEMORY_BLOCK_BYTES;
        record.input_limbs[start..start + MEMORY_BLOCK_BYTES]
            .copy_from_slice(&read_aux.entry.value.to_le_bytes());
        next_timestamp += 1;
    }

    for word_idx in 0..num_words {
        let address = record.buffer + (word_idx * MEMORY_BLOCK_BYTES) as u32;
        let write_aux = expect_memory_write(access, next_timestamp, address, pc)?;
        record.buffer_write_aux_cols[word_idx] = MemoryWriteBytesAuxRecord {
            prev_timestamp: write_aux.prev_timestamp,
            prev_data: prev_bytes(write_aux),
        };
        next_timestamp += 1;
    }

    Ok(())
}

fn expect_reg_read<'a, F: PrimeField32>(
    access: &LogNativeAccessView<'a, F>,
    timestamp: u32,
    reg_ptr: u32,
    pc: u32,
) -> Result<&'a PreflightMemoryAccessAux<F>, ExecutionError> {
    access.expect(
        timestamp,
        PREFLIGHT_MEMORY_KIND_READ,
        RV64_REGISTER_AS,
        u64::from(reg_ptr),
        MEMORY_BLOCK_BYTES,
        pc,
    )
}

fn expect_memory_read<'a, F: PrimeField32>(
    access: &LogNativeAccessView<'a, F>,
    timestamp: u32,
    address: u32,
    pc: u32,
) -> Result<&'a PreflightMemoryAccessAux<F>, ExecutionError> {
    access.expect(
        timestamp,
        PREFLIGHT_MEMORY_KIND_READ,
        RV64_MEMORY_AS,
        u64::from(address),
        MEMORY_BLOCK_BYTES,
        pc,
    )
}

fn expect_memory_write<'a, F: PrimeField32>(
    access: &LogNativeAccessView<'a, F>,
    timestamp: u32,
    address: u32,
    pc: u32,
) -> Result<&'a PreflightMemoryAccessAux<F>, ExecutionError> {
    access.expect(
        timestamp,
        PREFLIGHT_MEMORY_KIND_WRITE,
        RV64_MEMORY_AS,
        u64::from(address),
        MEMORY_BLOCK_BYTES,
        pc,
    )
}

fn prev_bytes<F: PrimeField32>(aux: &PreflightMemoryAccessAux<F>) -> [u8; MEMORY_BLOCK_BYTES] {
    let prev_u16 = aux
        .prev_data
        .map(|cell| cell.as_canonical_u32().try_into().expect("u16 memory cell"));
    rv64_u16_block_to_bytes(prev_u16)
}

fn rvr_error(message: String) -> ExecutionError {
    ExecutionError::RvrExecution(message)
}

const _: () = assert!(KECCAK_WIDTH_MEM_OPS * MEMORY_BLOCK_BYTES == 200);
