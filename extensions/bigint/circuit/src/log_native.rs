//! Log-native record assembly for the Int256 extension.

use std::array::from_fn;

use openvm_bigint_transpiler::{
    Rv64BaseAlu256Opcode, Rv64BranchEqual256Opcode, Rv64BranchLessThan256Opcode,
    Rv64LessThan256Opcode, Rv64Mul256Opcode, Rv64Shift256Opcode,
};
use openvm_circuit::{
    arch::{
        rvr::{
            LogNativeAccessView, LogNativeAssemblerRegistry, PreflightMemoryAccessAux,
            VmRvrLogNativeExtension, PREFLIGHT_MEMORY_KIND_READ, PREFLIGHT_MEMORY_KIND_WRITE,
        },
        Arena, EmptyAdapterCoreLayout, ExecutionError, RecordArena, BLOCK_FE_WIDTH,
        MEMORY_BLOCK_BYTES,
    },
    system::memory::offline_checker::{
        MemoryReadAuxRecord, MemoryWriteBytesAuxRecord, MemoryWriteU16AuxRecord,
    },
};
use openvm_instructions::{
    instruction::Instruction,
    riscv::{RV64_MEMORY_AS, RV64_REGISTER_AS, RV64_REGISTER_NUM_LIMBS},
    LocalOpcode,
};
use openvm_riscv_adapters::{
    Rv64VecHeapAdapterRecord, Rv64VecHeapBranchU16AdapterExecutor,
    Rv64VecHeapBranchU16AdapterRecord, Rv64VecHeapU16AdapterRecord,
};
use openvm_riscv_circuit::{
    adapters::{rv64_u16_block_to_bytes, RV64_BYTE_BITS, U16_BITS},
    log_native::{Rv64IRecordArena, Rv64IoRecordArena, Rv64MRecordArena},
    AddSubCoreRecord, BitwiseLogicCoreRecord, BranchEqualCoreRecord, BranchLessThanCoreRecord,
    LessThanCoreRecord, MultiplicationCoreRecord, ShiftLogicalCoreRecord,
    ShiftRightArithmeticCoreRecord,
};
use openvm_riscv_transpiler::{BaseAluOpcode, MulOpcode, ShiftOpcode};
use openvm_stark_backend::p3_field::PrimeField32;

use crate::{
    AluAdapterExecutor, AluU16AdapterExecutor, Int256, Int256Rv64Config, INT256_NUM_MEMORY_BLOCKS,
    INT256_NUM_U16_LIMBS, INT256_NUM_U8_LIMBS, NUM_READS,
};

const NUM_BLOCKS: usize = INT256_NUM_MEMORY_BLOCKS;

pub type Int256AluBytesLayout<F> = EmptyAdapterCoreLayout<F, AluAdapterExecutor>;
pub type Int256AluU16Layout<F> = EmptyAdapterCoreLayout<F, AluU16AdapterExecutor>;
pub type Int256BranchLayout<F> = EmptyAdapterCoreLayout<
    F,
    Rv64VecHeapBranchU16AdapterExecutor<NUM_READS, INT256_NUM_MEMORY_BLOCKS>,
>;

pub type Int256AluBytesAdapterRecord = Rv64VecHeapAdapterRecord<NUM_READS, NUM_BLOCKS, NUM_BLOCKS>;
pub type Int256AluU16AdapterRecord = Rv64VecHeapU16AdapterRecord<NUM_READS, NUM_BLOCKS, NUM_BLOCKS>;
pub type Int256BranchAdapterRecord = Rv64VecHeapBranchU16AdapterRecord<NUM_READS, NUM_BLOCKS>;

pub type Int256AddSubRecordMut<'a> = (
    &'a mut Int256AluU16AdapterRecord,
    &'a mut AddSubCoreRecord<INT256_NUM_U16_LIMBS>,
);
pub type Int256BitwiseRecordMut<'a> = (
    &'a mut Int256AluBytesAdapterRecord,
    &'a mut BitwiseLogicCoreRecord<INT256_NUM_U8_LIMBS>,
);
pub type Int256LessThanRecordMut<'a> = (
    &'a mut Int256AluU16AdapterRecord,
    &'a mut LessThanCoreRecord<INT256_NUM_U16_LIMBS, U16_BITS>,
);
pub type Int256MultiplicationRecordMut<'a> = (
    &'a mut Int256AluBytesAdapterRecord,
    &'a mut MultiplicationCoreRecord<INT256_NUM_U8_LIMBS, RV64_BYTE_BITS>,
);
pub type Int256ShiftLogicalRecordMut<'a> = (
    &'a mut Int256AluU16AdapterRecord,
    &'a mut ShiftLogicalCoreRecord<INT256_NUM_U16_LIMBS, U16_BITS>,
);
pub type Int256ShiftRightArithmeticRecordMut<'a> = (
    &'a mut Int256AluU16AdapterRecord,
    &'a mut ShiftRightArithmeticCoreRecord<INT256_NUM_U16_LIMBS, U16_BITS>,
);
pub type Int256BranchEqualRecordMut<'a> = (
    &'a mut Int256BranchAdapterRecord,
    &'a mut BranchEqualCoreRecord<INT256_NUM_U16_LIMBS>,
);
pub type Int256BranchLessThanRecordMut<'a> = (
    &'a mut Int256BranchAdapterRecord,
    &'a mut BranchLessThanCoreRecord<INT256_NUM_U16_LIMBS, U16_BITS>,
);

/// Arena union for every Int256 instruction record layout.
pub trait Int256RecordArena<F>:
    Arena
    + for<'a> RecordArena<'a, Int256AluU16Layout<F>, Int256AddSubRecordMut<'a>>
    + for<'a> RecordArena<'a, Int256AluBytesLayout<F>, Int256BitwiseRecordMut<'a>>
    + for<'a> RecordArena<'a, Int256AluU16Layout<F>, Int256LessThanRecordMut<'a>>
    + for<'a> RecordArena<'a, Int256AluBytesLayout<F>, Int256MultiplicationRecordMut<'a>>
    + for<'a> RecordArena<'a, Int256AluU16Layout<F>, Int256ShiftLogicalRecordMut<'a>>
    + for<'a> RecordArena<'a, Int256AluU16Layout<F>, Int256ShiftRightArithmeticRecordMut<'a>>
    + for<'a> RecordArena<'a, Int256BranchLayout<F>, Int256BranchEqualRecordMut<'a>>
    + for<'a> RecordArena<'a, Int256BranchLayout<F>, Int256BranchLessThanRecordMut<'a>>
{
}

impl<F, RA> Int256RecordArena<F> for RA where
    RA: Arena
        + for<'a> RecordArena<'a, Int256AluU16Layout<F>, Int256AddSubRecordMut<'a>>
        + for<'a> RecordArena<'a, Int256AluBytesLayout<F>, Int256BitwiseRecordMut<'a>>
        + for<'a> RecordArena<'a, Int256AluU16Layout<F>, Int256LessThanRecordMut<'a>>
        + for<'a> RecordArena<'a, Int256AluBytesLayout<F>, Int256MultiplicationRecordMut<'a>>
        + for<'a> RecordArena<'a, Int256AluU16Layout<F>, Int256ShiftLogicalRecordMut<'a>>
        + for<'a> RecordArena<'a, Int256AluU16Layout<F>, Int256ShiftRightArithmeticRecordMut<'a>>
        + for<'a> RecordArena<'a, Int256BranchLayout<F>, Int256BranchEqualRecordMut<'a>>
        + for<'a> RecordArena<'a, Int256BranchLayout<F>, Int256BranchLessThanRecordMut<'a>>
{
}

type AccessView<'a, F> = LogNativeAccessView<'a, F>;

impl<F, RA> VmRvrLogNativeExtension<F, RA> for Int256
where
    F: PrimeField32,
    RA: Int256RecordArena<F>,
{
    fn extend_rvr_log_native(&self, registry: &mut LogNativeAssemblerRegistry<F, RA>) {
        registry.register_if(
            [BaseAluOpcode::ADD, BaseAluOpcode::SUB]
                .map(|opcode| Rv64BaseAlu256Opcode(opcode).global_opcode()),
            is_int256_vec_heap_instruction,
            assemble_add_sub::<F, RA>,
        );
        registry.register_if(
            [BaseAluOpcode::XOR, BaseAluOpcode::OR, BaseAluOpcode::AND]
                .map(|opcode| Rv64BaseAlu256Opcode(opcode).global_opcode()),
            is_int256_vec_heap_instruction,
            assemble_bitwise::<F, RA>,
        );
        registry.register_if(
            Rv64LessThan256Opcode::iter().map(|opcode| opcode.global_opcode()),
            is_int256_vec_heap_instruction,
            assemble_less_than::<F, RA>,
        );
        registry.register_if(
            Rv64Mul256Opcode::iter().map(|opcode| opcode.global_opcode()),
            is_int256_vec_heap_instruction,
            assemble_multiplication::<F, RA>,
        );
        registry.register_if(
            [
                Rv64Shift256Opcode(ShiftOpcode::SLL),
                Rv64Shift256Opcode(ShiftOpcode::SRL),
                Rv64Shift256Opcode(ShiftOpcode::SRA),
            ]
            .map(|opcode| opcode.global_opcode()),
            is_int256_vec_heap_instruction,
            assemble_shift::<F, RA>,
        );
        registry.register_if(
            Rv64BranchEqual256Opcode::iter().map(|opcode| opcode.global_opcode()),
            is_int256_vec_heap_instruction,
            assemble_branch_equal::<F, RA>,
        );
        registry.register_if(
            Rv64BranchLessThan256Opcode::iter().map(|opcode| opcode.global_opcode()),
            is_int256_vec_heap_instruction,
            assemble_branch_less_than::<F, RA>,
        );
    }
}

impl<F, RA> VmRvrLogNativeExtension<F, RA> for Int256Rv64Config
where
    F: PrimeField32,
    RA: Rv64IRecordArena<F> + Rv64MRecordArena<F> + Rv64IoRecordArena<F> + Int256RecordArena<F>,
{
    fn extend_rvr_log_native(&self, registry: &mut LogNativeAssemblerRegistry<F, RA>) {
        // Inner RV64 registrations own all base opcodes, including the shared PHANTOM opcode.
        self.rv64i.extend_rvr_log_native(registry);
        self.rv64m.extend_rvr_log_native(registry);
        self.io.extend_rvr_log_native(registry);
        self.bigint.extend_rvr_log_native(registry);
    }
}

fn is_int256_vec_heap_instruction<F: PrimeField32>(instruction: &Instruction<F>) -> bool {
    instruction.d.as_canonical_u32() == RV64_REGISTER_AS
        && instruction.e.as_canonical_u32() == RV64_MEMORY_AS
}

struct VecHeapAccesses {
    rs_ptrs: [u32; NUM_READS],
    rd_ptr: Option<u32>,
    rs_vals: [u32; NUM_READS],
    rd_val: Option<u32>,
    rs_read_prev_timestamps: [u32; NUM_READS],
    rd_read_prev_timestamp: Option<u32>,
    read_prev_timestamps: [[u32; NUM_BLOCKS]; NUM_READS],
    read_words: [[u64; NUM_BLOCKS]; NUM_READS],
    write_prev_timestamps: [u32; NUM_BLOCKS],
    write_prev_u16: [[u16; BLOCK_FE_WIDTH]; NUM_BLOCKS],
}

fn collect_vec_heap_accesses<F: PrimeField32>(
    access: &AccessView<'_, F>,
    instruction: &Instruction<F>,
    pc: u32,
    timestamp: u32,
    has_destination: bool,
) -> Result<VecHeapAccesses, ExecutionError> {
    let rs_ptrs = if has_destination {
        [
            instruction.b.as_canonical_u32(),
            instruction.c.as_canonical_u32(),
        ]
    } else {
        [
            instruction.a.as_canonical_u32(),
            instruction.b.as_canonical_u32(),
        ]
    };
    let rd_ptr = has_destination.then(|| instruction.a.as_canonical_u32());

    let mut next_timestamp = timestamp;
    let mut rs_vals = [0; NUM_READS];
    let mut rs_read_prev_timestamps = [0; NUM_READS];
    for i in 0..NUM_READS {
        let aux = expect_access(
            access,
            next_timestamp,
            PREFLIGHT_MEMORY_KIND_READ,
            RV64_REGISTER_AS,
            u64::from(rs_ptrs[i]),
            pc,
        )?;
        rs_vals[i] = aux.entry.value as u32;
        rs_read_prev_timestamps[i] = aux.prev_timestamp;
        next_timestamp += 1;
    }

    let (rd_val, rd_read_prev_timestamp) = if let Some(rd_ptr) = rd_ptr {
        let aux = expect_access(
            access,
            next_timestamp,
            PREFLIGHT_MEMORY_KIND_READ,
            RV64_REGISTER_AS,
            u64::from(rd_ptr),
            pc,
        )?;
        next_timestamp += 1;
        (Some(aux.entry.value as u32), Some(aux.prev_timestamp))
    } else {
        (None, None)
    };

    let mut read_prev_timestamps = [[0; NUM_BLOCKS]; NUM_READS];
    let mut read_words = [[0; NUM_BLOCKS]; NUM_READS];
    for i in 0..NUM_READS {
        for block in 0..NUM_BLOCKS {
            let address = u64::from(rs_vals[i]) + (block * MEMORY_BLOCK_BYTES) as u64;
            let aux = expect_access(
                access,
                next_timestamp,
                PREFLIGHT_MEMORY_KIND_READ,
                RV64_MEMORY_AS,
                address,
                pc,
            )?;
            read_prev_timestamps[i][block] = aux.prev_timestamp;
            read_words[i][block] = aux.entry.value;
            next_timestamp += 1;
        }
    }

    let mut write_prev_timestamps = [0; NUM_BLOCKS];
    let mut write_prev_u16 = [[0; BLOCK_FE_WIDTH]; NUM_BLOCKS];
    if let Some(rd_val) = rd_val {
        for block in 0..NUM_BLOCKS {
            let address = u64::from(rd_val) + (block * MEMORY_BLOCK_BYTES) as u64;
            let aux = expect_access(
                access,
                next_timestamp,
                PREFLIGHT_MEMORY_KIND_WRITE,
                RV64_MEMORY_AS,
                address,
                pc,
            )?;
            write_prev_timestamps[block] = aux.prev_timestamp;
            write_prev_u16[block] = prev_u16(aux);
            next_timestamp += 1;
        }
    }

    Ok(VecHeapAccesses {
        rs_ptrs,
        rd_ptr,
        rs_vals,
        rd_val,
        rs_read_prev_timestamps,
        rd_read_prev_timestamp,
        read_prev_timestamps,
        read_words,
        write_prev_timestamps,
        write_prev_u16,
    })
}

fn fill_alu_u16_adapter<F: PrimeField32>(
    access: &AccessView<'_, F>,
    instruction: &Instruction<F>,
    pc: u32,
    timestamp: u32,
    record: &mut Int256AluU16AdapterRecord,
) -> Result<[[u16; INT256_NUM_U16_LIMBS]; NUM_READS], ExecutionError> {
    let values = collect_vec_heap_accesses(access, instruction, pc, timestamp, true)?;
    record.from_pc = pc;
    record.from_timestamp = timestamp;
    record.rs_ptrs = values.rs_ptrs;
    record.rd_ptr = values.rd_ptr.expect("destination requested");
    record.rs_vals = values.rs_vals;
    record.rd_val = values.rd_val.expect("destination requested");
    record.rs_read_aux = values
        .rs_read_prev_timestamps
        .map(|prev_timestamp| MemoryReadAuxRecord { prev_timestamp });
    record.rd_read_aux = MemoryReadAuxRecord {
        prev_timestamp: values
            .rd_read_prev_timestamp
            .expect("destination requested"),
    };
    record.reads_aux = values
        .read_prev_timestamps
        .map(|reads| reads.map(|prev_timestamp| MemoryReadAuxRecord { prev_timestamp }));
    record.writes_aux = from_fn(|block| MemoryWriteU16AuxRecord {
        prev_timestamp: values.write_prev_timestamps[block],
        prev_data: values.write_prev_u16[block],
    });
    Ok(values.read_words.map(words_to_u16_limbs))
}

fn fill_alu_bytes_adapter<F: PrimeField32>(
    access: &AccessView<'_, F>,
    instruction: &Instruction<F>,
    pc: u32,
    timestamp: u32,
    record: &mut Int256AluBytesAdapterRecord,
) -> Result<[[u8; INT256_NUM_U8_LIMBS]; NUM_READS], ExecutionError> {
    let values = collect_vec_heap_accesses(access, instruction, pc, timestamp, true)?;
    record.from_pc = pc;
    record.from_timestamp = timestamp;
    record.rs_ptrs = values.rs_ptrs;
    record.rd_ptr = values.rd_ptr.expect("destination requested");
    record.rs_vals = values.rs_vals;
    record.rd_val = values.rd_val.expect("destination requested");
    record.rs_read_aux = values
        .rs_read_prev_timestamps
        .map(|prev_timestamp| MemoryReadAuxRecord { prev_timestamp });
    record.rd_read_aux = MemoryReadAuxRecord {
        prev_timestamp: values
            .rd_read_prev_timestamp
            .expect("destination requested"),
    };
    record.reads_aux = values
        .read_prev_timestamps
        .map(|reads| reads.map(|prev_timestamp| MemoryReadAuxRecord { prev_timestamp }));
    record.writes_aux = from_fn(|block| MemoryWriteBytesAuxRecord {
        prev_timestamp: values.write_prev_timestamps[block],
        prev_data: rv64_u16_block_to_bytes(values.write_prev_u16[block]),
    });
    Ok(values.read_words.map(words_to_bytes))
}

fn fill_branch_adapter<F: PrimeField32>(
    access: &AccessView<'_, F>,
    instruction: &Instruction<F>,
    pc: u32,
    timestamp: u32,
    record: &mut Int256BranchAdapterRecord,
) -> Result<[[u16; INT256_NUM_U16_LIMBS]; NUM_READS], ExecutionError> {
    let values = collect_vec_heap_accesses(access, instruction, pc, timestamp, false)?;
    record.from_pc = pc;
    record.from_timestamp = timestamp;
    record.rs_ptrs = values.rs_ptrs;
    record.rs_vals = values.rs_vals;
    record.rs_read_aux = values
        .rs_read_prev_timestamps
        .map(|prev_timestamp| MemoryReadAuxRecord { prev_timestamp });
    record.reads_aux = values
        .read_prev_timestamps
        .map(|reads| reads.map(|prev_timestamp| MemoryReadAuxRecord { prev_timestamp }));
    Ok(values.read_words.map(words_to_u16_limbs))
}

fn assemble_add_sub<F: PrimeField32, RA: Int256RecordArena<F>>(
    arena: &mut RA,
    access: &AccessView<'_, F>,
    instruction: &Instruction<F>,
    pc: u32,
    timestamp: u32,
) -> Result<(), ExecutionError> {
    let (adapter_record, core_record): Int256AddSubRecordMut<'_> =
        arena.alloc(Int256AluU16Layout::<F>::new());
    let [rs1, rs2] = fill_alu_u16_adapter(access, instruction, pc, timestamp, adapter_record)?;
    core_record.b = rs1;
    core_record.c = rs2;
    core_record.local_opcode = instruction
        .opcode
        .local_opcode_idx(Rv64BaseAlu256Opcode::CLASS_OFFSET) as u8;
    Ok(())
}

fn assemble_bitwise<F: PrimeField32, RA: Int256RecordArena<F>>(
    arena: &mut RA,
    access: &AccessView<'_, F>,
    instruction: &Instruction<F>,
    pc: u32,
    timestamp: u32,
) -> Result<(), ExecutionError> {
    let (adapter_record, core_record): Int256BitwiseRecordMut<'_> =
        arena.alloc(Int256AluBytesLayout::<F>::new());
    let [rs1, rs2] = fill_alu_bytes_adapter(access, instruction, pc, timestamp, adapter_record)?;
    core_record.b = rs1;
    core_record.c = rs2;
    core_record.local_opcode = instruction
        .opcode
        .local_opcode_idx(Rv64BaseAlu256Opcode::CLASS_OFFSET) as u8;
    Ok(())
}

fn assemble_less_than<F: PrimeField32, RA: Int256RecordArena<F>>(
    arena: &mut RA,
    access: &AccessView<'_, F>,
    instruction: &Instruction<F>,
    pc: u32,
    timestamp: u32,
) -> Result<(), ExecutionError> {
    let (adapter_record, core_record): Int256LessThanRecordMut<'_> =
        arena.alloc(Int256AluU16Layout::<F>::new());
    let [rs1, rs2] = fill_alu_u16_adapter(access, instruction, pc, timestamp, adapter_record)?;
    core_record.b = rs1;
    core_record.c = rs2;
    core_record.local_opcode = instruction
        .opcode
        .local_opcode_idx(Rv64LessThan256Opcode::CLASS_OFFSET) as u8;
    Ok(())
}

fn assemble_multiplication<F: PrimeField32, RA: Int256RecordArena<F>>(
    arena: &mut RA,
    access: &AccessView<'_, F>,
    instruction: &Instruction<F>,
    pc: u32,
    timestamp: u32,
) -> Result<(), ExecutionError> {
    let (adapter_record, core_record): Int256MultiplicationRecordMut<'_> =
        arena.alloc(Int256AluBytesLayout::<F>::new());
    let [rs1, rs2] = fill_alu_bytes_adapter(access, instruction, pc, timestamp, adapter_record)?;
    core_record.b = rs1;
    core_record.c = rs2;
    debug_assert_eq!(
        instruction
            .opcode
            .local_opcode_idx(Rv64Mul256Opcode::CLASS_OFFSET),
        MulOpcode::MUL as usize
    );
    Ok(())
}

fn assemble_shift<F: PrimeField32, RA: Int256RecordArena<F>>(
    arena: &mut RA,
    access: &AccessView<'_, F>,
    instruction: &Instruction<F>,
    pc: u32,
    timestamp: u32,
) -> Result<(), ExecutionError> {
    let local_opcode = ShiftOpcode::from_usize(
        instruction
            .opcode
            .local_opcode_idx(Rv64Shift256Opcode::CLASS_OFFSET),
    );
    if local_opcode == ShiftOpcode::SRA {
        let (adapter_record, core_record): Int256ShiftRightArithmeticRecordMut<'_> =
            arena.alloc(Int256AluU16Layout::<F>::new());
        let [rs1, rs2] = fill_alu_u16_adapter(access, instruction, pc, timestamp, adapter_record)?;
        core_record.b = rs1;
        core_record.c = rs2;
    } else {
        let (adapter_record, core_record): Int256ShiftLogicalRecordMut<'_> =
            arena.alloc(Int256AluU16Layout::<F>::new());
        let [rs1, rs2] = fill_alu_u16_adapter(access, instruction, pc, timestamp, adapter_record)?;
        core_record.b = rs1;
        core_record.c = rs2;
        core_record.local_opcode = local_opcode as u8;
    }
    Ok(())
}

fn assemble_branch_equal<F: PrimeField32, RA: Int256RecordArena<F>>(
    arena: &mut RA,
    access: &AccessView<'_, F>,
    instruction: &Instruction<F>,
    pc: u32,
    timestamp: u32,
) -> Result<(), ExecutionError> {
    let (adapter_record, core_record): Int256BranchEqualRecordMut<'_> =
        arena.alloc(Int256BranchLayout::<F>::new());
    let [rs1, rs2] = fill_branch_adapter(access, instruction, pc, timestamp, adapter_record)?;
    core_record.a = rs1;
    core_record.b = rs2;
    core_record.imm = instruction.c.as_canonical_u32();
    core_record.local_opcode = instruction
        .opcode
        .local_opcode_idx(Rv64BranchEqual256Opcode::CLASS_OFFSET)
        as u8;
    Ok(())
}

fn assemble_branch_less_than<F: PrimeField32, RA: Int256RecordArena<F>>(
    arena: &mut RA,
    access: &AccessView<'_, F>,
    instruction: &Instruction<F>,
    pc: u32,
    timestamp: u32,
) -> Result<(), ExecutionError> {
    let (adapter_record, core_record): Int256BranchLessThanRecordMut<'_> =
        arena.alloc(Int256BranchLayout::<F>::new());
    let [rs1, rs2] = fill_branch_adapter(access, instruction, pc, timestamp, adapter_record)?;
    core_record.a = rs1;
    core_record.b = rs2;
    core_record.imm = instruction.c.as_canonical_u32();
    core_record.local_opcode = instruction
        .opcode
        .local_opcode_idx(Rv64BranchLessThan256Opcode::CLASS_OFFSET)
        as u8;
    Ok(())
}

fn expect_access<'a, F: PrimeField32>(
    access: &'a AccessView<'_, F>,
    timestamp: u32,
    kind: u8,
    addr_space: u32,
    address: u64,
    pc: u32,
) -> Result<&'a PreflightMemoryAccessAux<F>, ExecutionError> {
    access.expect(
        timestamp,
        kind,
        addr_space,
        address,
        RV64_REGISTER_NUM_LIMBS,
        pc,
    )
}

fn words_to_bytes(words: [u64; NUM_BLOCKS]) -> [u8; INT256_NUM_U8_LIMBS] {
    let mut bytes = [0; INT256_NUM_U8_LIMBS];
    for (block, word) in words.into_iter().enumerate() {
        let start = block * MEMORY_BLOCK_BYTES;
        bytes[start..start + MEMORY_BLOCK_BYTES].copy_from_slice(&word.to_le_bytes());
    }
    bytes
}

fn words_to_u16_limbs(words: [u64; NUM_BLOCKS]) -> [u16; INT256_NUM_U16_LIMBS] {
    let bytes = words_to_bytes(words);
    from_fn(|idx| u16::from_le_bytes([bytes[idx * 2], bytes[idx * 2 + 1]]))
}

fn prev_u16<F: PrimeField32>(aux: &PreflightMemoryAccessAux<F>) -> [u16; BLOCK_FE_WIDTH] {
    aux.prev_data
        .map(|cell| cell.as_canonical_u32().try_into().expect("u16 memory cell"))
}
