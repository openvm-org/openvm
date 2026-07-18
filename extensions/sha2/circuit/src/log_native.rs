use std::mem::align_of;

use openvm_circuit::{
    arch::{
        rvr::{
            ArenaNativeGeometry, ArenaNativeLayout, LogNativeAccessView,
            LogNativeAssemblerRegistry, PreflightMemoryAccessAux, VmRvrLogNativeExtension,
            PREFLIGHT_MEMORY_KIND_READ, PREFLIGHT_MEMORY_KIND_WRITE,
        },
        Arena, ExecutionError, RecordArena, SizedRecord,
    },
    system::memory::offline_checker::MemoryWriteBytesAuxRecord,
};
use openvm_instructions::{
    instruction::Instruction,
    riscv::{RV64_MEMORY_AS, RV64_REGISTER_AS, RV64_REGISTER_NUM_LIMBS},
    LocalOpcode,
};
use openvm_riscv_circuit::{
    adapters::rv64_u16_block_to_bytes, log_native::Rv64StandardRecordArena,
};
use openvm_sha2_air::{Sha256Config, Sha2BlockHasherSubairConfig, Sha2Variant};
use openvm_sha2_transpiler::Rv64Sha2Opcode;
use openvm_stark_backend::p3_field::PrimeField32;
use rvr_openvm_ext_sha2::SHA256_DIRECT_RECORD_SIZE;

use crate::{
    Sha2, Sha2MainChipConfig, Sha2Metadata, Sha2RecordLayout, Sha2RecordMut, Sha2Rv64Config,
    SHA2_READ_SIZE, SHA2_REGISTER_READS, SHA2_WRITE_SIZE,
};

/// Record-arena capability required by SHA-256 log-native preflight assembly.
pub trait Sha256RecordArena<F>:
    Arena + for<'a> RecordArena<'a, Sha2RecordLayout, Sha2RecordMut<'a>>
{
}

impl<F, RA> Sha256RecordArena<F> for RA where
    RA: Arena + for<'a> RecordArena<'a, Sha2RecordLayout, Sha2RecordMut<'a>>
{
}

impl<F, RA> VmRvrLogNativeExtension<F, RA> for Sha2
where
    F: PrimeField32,
    RA: Sha256RecordArena<F>,
{
    fn extend_rvr_log_native(&self, registry: &mut LogNativeAssemblerRegistry<F, RA>) {
        let opcodes = [Rv64Sha2Opcode::SHA256.global_opcode()];
        registry.register_if(opcodes, is_sha256_instruction, assemble_sha256::<F, RA>);
        let layout = Sha2RecordLayout::new(Sha2Metadata {
            variant: Sha2Variant::Sha256,
        });
        assert_eq!(Sha2RecordMut::size(&layout), SHA256_DIRECT_RECORD_SIZE);
        assert_eq!(
            Sha2RecordMut::alignment(&layout),
            align_of::<crate::Sha2RecordHeader>()
        );
        registry.register_inline_arena_native(
            opcodes,
            SHA256_DIRECT_RECORD_SIZE,
            assemble_sha256_inline::<F, RA>,
            ArenaNativeGeometry {
                adapter_size: SHA256_DIRECT_RECORD_SIZE,
                adapter_align: Sha2RecordMut::alignment(&layout),
                core_size: 0,
                core_align: 1,
                core_off_matrix: 0,
                layout: ArenaNativeLayout::Custom {
                    residual_memory_chronology: true,
                    max_residual_events_per_record: 19,
                    layout_id: "openvm.rvr.sha256-final.v1",
                },
            },
        );
    }
}

impl<F, RA> VmRvrLogNativeExtension<F, RA> for Sha2Rv64Config
where
    F: PrimeField32,
    RA: Rv64StandardRecordArena<F> + Sha256RecordArena<F>,
{
    fn extend_rvr_log_native(&self, registry: &mut LogNativeAssemblerRegistry<F, RA>) {
        self.rv64i.extend_rvr_log_native(registry);
        self.rv64m.extend_rvr_log_native(registry);
        self.io.extend_rvr_log_native(registry);
        self.sha2.extend_rvr_log_native(registry);
    }
}

fn is_sha256_instruction<F: PrimeField32>(instruction: &Instruction<F>) -> bool {
    instruction.d.as_canonical_u32() == RV64_REGISTER_AS
        && instruction.e.as_canonical_u32() == RV64_MEMORY_AS
}

fn assemble_sha256_inline<F, RA>(
    arena: &mut RA,
    _instruction: &Instruction<F>,
    compact: &[u8],
    pc: u32,
) -> Result<(), ExecutionError>
where
    F: PrimeField32,
    RA: Sha256RecordArena<F>,
{
    if compact.len() != SHA256_DIRECT_RECORD_SIZE {
        return Err(rvr_error(format!(
            "invalid SHA-256 inline record size {} at pc {pc:#x}; expected {}",
            compact.len(),
            SHA256_DIRECT_RECORD_SIZE
        )));
    }
    let record = arena.alloc(Sha2RecordLayout::new(Sha2Metadata {
        variant: Sha2Variant::Sha256,
    }));
    unsafe {
        std::ptr::copy_nonoverlapping(
            compact.as_ptr(),
            (record.inner as *mut crate::Sha2RecordHeader).cast::<u8>(),
            compact.len(),
        );
    }
    if u32::from(record.inner.variant) != u32::from(Sha2Variant::Sha256) {
        return Err(rvr_error(format!(
            "SHA-256 inline record variant mismatch at pc {pc:#x}"
        )));
    }
    if record.inner.from_pc != pc {
        return Err(rvr_error(format!(
            "SHA-256 inline record pc mismatch: record={:#x}, program={pc:#x}",
            record.inner.from_pc
        )));
    }
    Ok(())
}

/// Reconstruct one incremental SHA-256 compression record from normalized rvr logs.
///
/// The current SHA-2 ISA executes exactly one 64-byte block per instruction. A full message with
/// multiple padded blocks therefore contributes one record per SHA-256 program-log entry, and the
/// block-hasher chip expands those records to `Sha256Config::ROWS_PER_BLOCK` rows each.
pub fn assemble_sha256<F, RA>(
    arena: &mut RA,
    access: &LogNativeAccessView<'_, F>,
    instruction: &Instruction<F>,
    pc: u32,
    timestamp: u32,
) -> Result<(), ExecutionError>
where
    F: PrimeField32,
    RA: Sha256RecordArena<F>,
{
    let record = arena.alloc(Sha2RecordLayout::new(Sha2Metadata {
        variant: Sha2Variant::Sha256,
    }));

    record.inner.variant = Sha2Variant::Sha256;
    record.inner.from_pc = pc;
    record.inner.timestamp = timestamp;
    record.inner.dst_reg_ptr = instruction.a.as_canonical_u32();
    record.inner.state_reg_ptr = instruction.b.as_canonical_u32();
    record.inner.input_reg_ptr = instruction.c.as_canonical_u32();

    let mut next_timestamp = timestamp;
    let register_ptrs = [
        record.inner.dst_reg_ptr,
        record.inner.state_reg_ptr,
        record.inner.input_reg_ptr,
    ];
    let mut resolved_ptrs = [0u32; SHA2_REGISTER_READS];
    for (idx, (&register_ptr, resolved_ptr)) in register_ptrs
        .iter()
        .zip(resolved_ptrs.iter_mut())
        .enumerate()
    {
        let aux = access.expect(
            next_timestamp,
            PREFLIGHT_MEMORY_KIND_READ,
            RV64_REGISTER_AS,
            u64::from(register_ptr),
            RV64_REGISTER_NUM_LIMBS,
            pc,
        )?;
        record.inner.register_reads_aux[idx].prev_timestamp = aux.prev_timestamp;
        *resolved_ptr = u32::try_from(aux.entry.value).map_err(|_| {
            rvr_error(format!(
                "SHA-256 pointer register at pc {pc:#x} contains non-zero upper bytes: {:#x}",
                aux.entry.value
            ))
        })?;
        next_timestamp += 1;
    }
    [
        record.inner.dst_ptr,
        record.inner.state_ptr,
        record.inner.input_ptr,
    ] = resolved_ptrs;

    for (idx, message_word) in record
        .message_bytes
        .chunks_exact_mut(SHA2_READ_SIZE)
        .enumerate()
    {
        let address = record.inner.input_ptr + (idx * SHA2_READ_SIZE) as u32;
        let aux = expect_memory_access(
            access,
            next_timestamp,
            PREFLIGHT_MEMORY_KIND_READ,
            address,
            pc,
        )?;
        message_word.copy_from_slice(&aux.entry.value.to_le_bytes());
        record.input_reads_aux[idx].prev_timestamp = aux.prev_timestamp;
        next_timestamp += 1;
    }

    for (idx, state_word) in record
        .prev_state
        .chunks_exact_mut(SHA2_READ_SIZE)
        .enumerate()
    {
        let address = record.inner.state_ptr + (idx * SHA2_READ_SIZE) as u32;
        let aux = expect_memory_access(
            access,
            next_timestamp,
            PREFLIGHT_MEMORY_KIND_READ,
            address,
            pc,
        )?;
        state_word.copy_from_slice(&aux.entry.value.to_le_bytes());
        record.state_reads_aux[idx].prev_timestamp = aux.prev_timestamp;
        next_timestamp += 1;
    }

    for (idx, state_word) in record
        .new_state
        .chunks_exact_mut(SHA2_WRITE_SIZE)
        .enumerate()
    {
        let address = record.inner.dst_ptr + (idx * SHA2_WRITE_SIZE) as u32;
        let aux = expect_memory_access(
            access,
            next_timestamp,
            PREFLIGHT_MEMORY_KIND_WRITE,
            address,
            pc,
        )?;
        state_word.copy_from_slice(&aux.entry.value.to_le_bytes());
        record.write_aux[idx] = MemoryWriteBytesAuxRecord {
            prev_timestamp: aux.prev_timestamp,
            prev_data: prev_bytes(aux),
        };
        next_timestamp += 1;
    }

    debug_assert_eq!(
        next_timestamp - timestamp,
        Sha256Config::TIMESTAMP_DELTA as u32
    );
    debug_assert_eq!(Sha256Config::ROWS_PER_BLOCK, 17);
    Ok(())
}

fn expect_memory_access<'a, F: PrimeField32>(
    access: &LogNativeAccessView<'a, F>,
    timestamp: u32,
    kind: u8,
    address: u32,
    pc: u32,
) -> Result<&'a PreflightMemoryAccessAux<F>, ExecutionError> {
    access.expect(
        timestamp,
        kind,
        RV64_MEMORY_AS,
        u64::from(address),
        SHA2_READ_SIZE,
        pc,
    )
}

fn prev_bytes<F: PrimeField32>(aux: &PreflightMemoryAccessAux<F>) -> [u8; SHA2_WRITE_SIZE] {
    rv64_u16_block_to_bytes(
        aux.prev_data
            .map(|cell| cell.as_canonical_u32().try_into().expect("u16 memory cell")),
    )
}

fn rvr_error(message: impl Into<String>) -> ExecutionError {
    ExecutionError::RvrExecution(message.into())
}
