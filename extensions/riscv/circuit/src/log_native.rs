use openvm_circuit::{
    arch::{
        rvr::{
            generate_record_arenas_from_logs, Alu3ArenaFieldOffsets, Alu3WArenaFieldOffsets,
            ArenaNativeGeometry, ArenaNativeLayout, Branch2ArenaFieldOffsets, DeltaAccessPattern,
            LoadStoreArenaFieldOffsets, LogNativeAccessView, LogNativeAssemblerRegistry,
            PreflightMemoryAccessAux, RvrPreflightOutput, Rw1ArenaFieldOffsets,
            VmRvrLogNativeExtension, Wr1ArenaFieldOffsets, PREFLIGHT_ADDSUB_RECORD_SIZE,
            PREFLIGHT_BRANCH2_RECORD_SIZE, PREFLIGHT_MEMORY_KIND_READ, PREFLIGHT_MEMORY_KIND_WRITE,
            PREFLIGHT_RW1_RECORD_SIZE, PREFLIGHT_WR1_RECORD_SIZE,
        },
        AdapterTraceExecutor, Arena, EmptyAdapterCoreLayout, EmptyMultiRowLayout, ExecutionError,
        MultiRowLayout, RecordArena, BLOCK_FE_WIDTH,
    },
    system::{
        memory::offline_checker::{
            MemoryReadAuxRecord, MemoryWriteAuxRecord, MemoryWriteBytesAuxRecord,
        },
        phantom::PhantomRecord,
    },
};
use openvm_instructions::{
    exe::VmExe,
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV64_IMM_AS, RV64_MEMORY_AS, RV64_REGISTER_AS},
    LocalOpcode, SystemOpcode, PUBLIC_VALUES_AS,
};
use openvm_riscv_transpiler::{
    BaseAluOpcode, BaseAluWOpcode, BranchEqualOpcode, BranchLessThanOpcode, DivRemOpcode,
    DivRemWOpcode, LessThanOpcode, MulHOpcode, MulOpcode, MulWOpcode, Rv64AuipcOpcode,
    Rv64HintStoreOpcode, Rv64JalLuiOpcode, Rv64JalrOpcode, Rv64LoadStoreOpcode, ShiftOpcode,
    ShiftWOpcode,
};
use openvm_stark_backend::p3_field::PrimeField32;
use rvr_openvm_ext_riscv::HintStoreRecordDescriptor;
use strum::IntoEnumIterator;

use crate::{
    adapters::{
        imm_to_rv64_u64, rv64_bytes_to_u16_block, rv64_u16_block_to_bytes, rv64_u32_to_u16_block,
        Rv64BaseAluAdapterExecutor, Rv64BaseAluAdapterRecord, Rv64BaseAluU16AdapterExecutor,
        Rv64BaseAluU16AdapterRecord, Rv64BaseAluWU16AdapterExecutor, Rv64BaseAluWU16AdapterRecord,
        Rv64BranchAdapterExecutor, Rv64BranchAdapterRecord, Rv64CondRdWriteAdapterExecutor,
        Rv64JalrAdapterExecutor, Rv64JalrAdapterRecord, Rv64LoadStoreAdapterExecutor,
        Rv64LoadStoreAdapterRecord, Rv64MultAdapterExecutor, Rv64MultAdapterRecord,
        Rv64MultWAdapterExecutor, Rv64MultWAdapterRecord, Rv64RdWriteAdapterExecutor,
        Rv64RdWriteAdapterRecord, RV64_BYTE_BITS, RV64_REGISTER_NUM_LIMBS, RV64_WORD_NUM_LIMBS,
        RV64_WORD_U16_LIMBS, U16_BITS,
    },
    AddSubCoreRecord, BitwiseLogicCoreRecord, BranchEqualCoreRecord, BranchLessThanCoreRecord,
    DivRemCoreRecord, LessThanCoreRecord, LoadSignExtendCoreRecord, LoadStoreCoreRecord,
    MulHCoreRecord, MultiplicationCoreRecord, Rv64AuipcCoreRecord, Rv64HintStoreLayout,
    Rv64HintStoreMetadata, Rv64HintStoreRecordHeader, Rv64HintStoreRecordMut, Rv64HintStoreVar,
    Rv64JalLuiCoreRecord, Rv64JalrCoreRecord, ShiftLogicalCoreRecord,
    ShiftRightArithmeticCoreRecord,
};

const JAL: usize = Rv64JalLuiOpcode::JAL as usize;

pub(crate) type BaseAluU16Layout<F> = EmptyAdapterCoreLayout<F, Rv64BaseAluU16AdapterExecutor>;
pub(crate) type BaseAluBytesLayout<F> =
    EmptyAdapterCoreLayout<F, Rv64BaseAluAdapterExecutor<RV64_BYTE_BITS>>;
pub(crate) type BaseAluWU16Layout<F> = EmptyAdapterCoreLayout<F, Rv64BaseAluWU16AdapterExecutor>;
pub(crate) type BranchLayout<F> = EmptyAdapterCoreLayout<F, Rv64BranchAdapterExecutor>;
pub(crate) type CondRdWriteLayout<F> = EmptyAdapterCoreLayout<F, Rv64CondRdWriteAdapterExecutor>;
pub(crate) type JalrLayout<F> = EmptyAdapterCoreLayout<F, Rv64JalrAdapterExecutor>;
pub(crate) type RdWriteLayout<F> = EmptyAdapterCoreLayout<F, Rv64RdWriteAdapterExecutor>;
pub(crate) type LoadStoreLayout<F> = EmptyAdapterCoreLayout<F, Rv64LoadStoreAdapterExecutor>;
pub(crate) type MultLayout<F> = EmptyAdapterCoreLayout<F, Rv64MultAdapterExecutor>;
pub(crate) type MultWLayout<F> = EmptyAdapterCoreLayout<F, Rv64MultWAdapterExecutor>;

pub(crate) type AddSubRecordMut<'a> = (
    &'a mut Rv64BaseAluU16AdapterRecord,
    &'a mut AddSubCoreRecord<BLOCK_FE_WIDTH>,
);
pub(crate) type LessThanRecordMut<'a> = (
    &'a mut Rv64BaseAluU16AdapterRecord,
    &'a mut LessThanCoreRecord<BLOCK_FE_WIDTH, U16_BITS>,
);
pub(crate) type BitwiseRecordMut<'a> = (
    &'a mut Rv64BaseAluAdapterRecord,
    &'a mut BitwiseLogicCoreRecord<RV64_REGISTER_NUM_LIMBS>,
);
pub(crate) type ShiftLogicalRecordMut<'a> = (
    &'a mut Rv64BaseAluU16AdapterRecord,
    &'a mut ShiftLogicalCoreRecord<BLOCK_FE_WIDTH, U16_BITS>,
);
pub(crate) type ShiftRightArithmeticRecordMut<'a> = (
    &'a mut Rv64BaseAluU16AdapterRecord,
    &'a mut ShiftRightArithmeticCoreRecord<BLOCK_FE_WIDTH, U16_BITS>,
);
pub(crate) type AddSubWRecordMut<'a> = (
    &'a mut Rv64BaseAluWU16AdapterRecord,
    &'a mut AddSubCoreRecord<RV64_WORD_U16_LIMBS>,
);
pub(crate) type ShiftWLogicalRecordMut<'a> = (
    &'a mut Rv64BaseAluWU16AdapterRecord,
    &'a mut ShiftLogicalCoreRecord<RV64_WORD_U16_LIMBS, U16_BITS>,
);
pub(crate) type ShiftWRightArithmeticRecordMut<'a> = (
    &'a mut Rv64BaseAluWU16AdapterRecord,
    &'a mut ShiftRightArithmeticCoreRecord<RV64_WORD_U16_LIMBS, U16_BITS>,
);
pub(crate) type BranchEqRecordMut<'a> = (
    &'a mut Rv64BranchAdapterRecord,
    &'a mut BranchEqualCoreRecord<BLOCK_FE_WIDTH>,
);
pub(crate) type BranchLtRecordMut<'a> = (
    &'a mut Rv64BranchAdapterRecord,
    &'a mut BranchLessThanCoreRecord<BLOCK_FE_WIDTH, U16_BITS>,
);
pub(crate) type JalLuiRecordMut<'a> = (
    &'a mut Rv64RdWriteAdapterRecord,
    &'a mut Rv64JalLuiCoreRecord,
);
pub(crate) type JalrRecordMut<'a> = (&'a mut Rv64JalrAdapterRecord, &'a mut Rv64JalrCoreRecord);
pub(crate) type AuipcRecordMut<'a> = (
    &'a mut Rv64RdWriteAdapterRecord,
    &'a mut Rv64AuipcCoreRecord,
);
pub(crate) type MulRecordMut<'a> = (
    &'a mut Rv64MultAdapterRecord,
    &'a mut MultiplicationCoreRecord<RV64_REGISTER_NUM_LIMBS, RV64_BYTE_BITS>,
);
pub(crate) type MulHRecordMut<'a> = (
    &'a mut Rv64MultAdapterRecord,
    &'a mut MulHCoreRecord<RV64_REGISTER_NUM_LIMBS, RV64_BYTE_BITS>,
);
pub(crate) type MulWRecordMut<'a> = (
    &'a mut Rv64MultWAdapterRecord,
    &'a mut MultiplicationCoreRecord<RV64_WORD_NUM_LIMBS, RV64_BYTE_BITS>,
);
pub(crate) type LoadStoreRecordMut<'a> = (
    &'a mut Rv64LoadStoreAdapterRecord,
    &'a mut LoadStoreCoreRecord<RV64_REGISTER_NUM_LIMBS>,
);
pub(crate) type LoadSignExtendRecordMut<'a> = (
    &'a mut Rv64LoadStoreAdapterRecord,
    &'a mut LoadSignExtendCoreRecord<RV64_REGISTER_NUM_LIMBS>,
);
pub(crate) type DivRemRecordMut<'a> = (
    &'a mut Rv64MultAdapterRecord,
    &'a mut DivRemCoreRecord<RV64_REGISTER_NUM_LIMBS>,
);
pub(crate) type DivRemWRecordMut<'a> = (
    &'a mut Rv64MultWAdapterRecord,
    &'a mut DivRemCoreRecord<RV64_WORD_NUM_LIMBS>,
);
pub(crate) type HintStoreRecordMut<'a> = Rv64HintStoreRecordMut<'a>;
pub(crate) type PhantomRecordMut<'a> = &'a mut PhantomRecord;

pub trait Rv64IRecordArena<F>:
    Arena
    + for<'a> RecordArena<'a, BaseAluU16Layout<F>, AddSubRecordMut<'a>>
    + for<'a> RecordArena<'a, BaseAluU16Layout<F>, LessThanRecordMut<'a>>
    + for<'a> RecordArena<'a, BaseAluBytesLayout<F>, BitwiseRecordMut<'a>>
    + for<'a> RecordArena<'a, BaseAluU16Layout<F>, ShiftLogicalRecordMut<'a>>
    + for<'a> RecordArena<'a, BaseAluU16Layout<F>, ShiftRightArithmeticRecordMut<'a>>
    + for<'a> RecordArena<'a, BaseAluWU16Layout<F>, AddSubWRecordMut<'a>>
    + for<'a> RecordArena<'a, BaseAluWU16Layout<F>, ShiftWLogicalRecordMut<'a>>
    + for<'a> RecordArena<'a, BaseAluWU16Layout<F>, ShiftWRightArithmeticRecordMut<'a>>
    + for<'a> RecordArena<'a, BranchLayout<F>, BranchEqRecordMut<'a>>
    + for<'a> RecordArena<'a, BranchLayout<F>, BranchLtRecordMut<'a>>
    + for<'a> RecordArena<'a, CondRdWriteLayout<F>, JalLuiRecordMut<'a>>
    + for<'a> RecordArena<'a, JalrLayout<F>, JalrRecordMut<'a>>
    + for<'a> RecordArena<'a, RdWriteLayout<F>, AuipcRecordMut<'a>>
    + for<'a> RecordArena<'a, LoadStoreLayout<F>, LoadStoreRecordMut<'a>>
    + for<'a> RecordArena<'a, LoadStoreLayout<F>, LoadSignExtendRecordMut<'a>>
    + for<'a> RecordArena<'a, EmptyMultiRowLayout, PhantomRecordMut<'a>>
{
}

impl<F, RA> Rv64IRecordArena<F> for RA where
    RA: Arena
        + for<'a> RecordArena<'a, BaseAluU16Layout<F>, AddSubRecordMut<'a>>
        + for<'a> RecordArena<'a, BaseAluU16Layout<F>, LessThanRecordMut<'a>>
        + for<'a> RecordArena<'a, BaseAluBytesLayout<F>, BitwiseRecordMut<'a>>
        + for<'a> RecordArena<'a, BaseAluU16Layout<F>, ShiftLogicalRecordMut<'a>>
        + for<'a> RecordArena<'a, BaseAluU16Layout<F>, ShiftRightArithmeticRecordMut<'a>>
        + for<'a> RecordArena<'a, BaseAluWU16Layout<F>, AddSubWRecordMut<'a>>
        + for<'a> RecordArena<'a, BaseAluWU16Layout<F>, ShiftWLogicalRecordMut<'a>>
        + for<'a> RecordArena<'a, BaseAluWU16Layout<F>, ShiftWRightArithmeticRecordMut<'a>>
        + for<'a> RecordArena<'a, BranchLayout<F>, BranchEqRecordMut<'a>>
        + for<'a> RecordArena<'a, BranchLayout<F>, BranchLtRecordMut<'a>>
        + for<'a> RecordArena<'a, CondRdWriteLayout<F>, JalLuiRecordMut<'a>>
        + for<'a> RecordArena<'a, JalrLayout<F>, JalrRecordMut<'a>>
        + for<'a> RecordArena<'a, RdWriteLayout<F>, AuipcRecordMut<'a>>
        + for<'a> RecordArena<'a, LoadStoreLayout<F>, LoadStoreRecordMut<'a>>
        + for<'a> RecordArena<'a, LoadStoreLayout<F>, LoadSignExtendRecordMut<'a>>
        + for<'a> RecordArena<'a, EmptyMultiRowLayout, PhantomRecordMut<'a>>
{
}

pub trait Rv64MRecordArena<F>:
    Arena
    + for<'a> RecordArena<'a, MultLayout<F>, MulRecordMut<'a>>
    + for<'a> RecordArena<'a, MultLayout<F>, MulHRecordMut<'a>>
    + for<'a> RecordArena<'a, MultWLayout<F>, MulWRecordMut<'a>>
    + for<'a> RecordArena<'a, MultLayout<F>, DivRemRecordMut<'a>>
    + for<'a> RecordArena<'a, MultWLayout<F>, DivRemWRecordMut<'a>>
{
}

impl<F, RA> Rv64MRecordArena<F> for RA where
    RA: Arena
        + for<'a> RecordArena<'a, MultLayout<F>, MulRecordMut<'a>>
        + for<'a> RecordArena<'a, MultLayout<F>, MulHRecordMut<'a>>
        + for<'a> RecordArena<'a, MultWLayout<F>, MulWRecordMut<'a>>
        + for<'a> RecordArena<'a, MultLayout<F>, DivRemRecordMut<'a>>
        + for<'a> RecordArena<'a, MultWLayout<F>, DivRemWRecordMut<'a>>
{
}

pub trait Rv64IoRecordArena<F>:
    Arena + for<'a> RecordArena<'a, Rv64HintStoreLayout, HintStoreRecordMut<'a>>
{
}

impl<F, RA> Rv64IoRecordArena<F> for RA where
    RA: Arena + for<'a> RecordArena<'a, Rv64HintStoreLayout, HintStoreRecordMut<'a>>
{
}

/// Arena union for the composed RV64IM configuration.
pub trait Rv64StandardRecordArena<F>:
    Rv64IRecordArena<F> + Rv64MRecordArena<F> + Rv64IoRecordArena<F>
{
}

impl<F, RA> Rv64StandardRecordArena<F> for RA where
    RA: Rv64IRecordArena<F> + Rv64MRecordArena<F> + Rv64IoRecordArena<F>
{
}

type AccessView<'a, F> = LogNativeAccessView<'a, F>;

trait Rv64AccessView<F> {
    fn expect_reg_read(
        &self,
        timestamp: u32,
        reg_ptr: u32,
        pc: u32,
    ) -> Result<&PreflightMemoryAccessAux<F>, ExecutionError>;

    fn expect_reg_write(
        &self,
        timestamp: u32,
        reg_ptr: u32,
        pc: u32,
    ) -> Result<&PreflightMemoryAccessAux<F>, ExecutionError>;

    fn expect_memory_read(
        &self,
        timestamp: u32,
        address: u32,
        pc: u32,
    ) -> Result<&PreflightMemoryAccessAux<F>, ExecutionError>;

    fn expect_memory_write(
        &self,
        timestamp: u32,
        addr_space: u32,
        address: u32,
        pc: u32,
    ) -> Result<&PreflightMemoryAccessAux<F>, ExecutionError>;
}

impl<F: PrimeField32> Rv64AccessView<F> for AccessView<'_, F> {
    fn expect_reg_read(
        &self,
        timestamp: u32,
        reg_ptr: u32,
        pc: u32,
    ) -> Result<&PreflightMemoryAccessAux<F>, ExecutionError> {
        self.expect(
            timestamp,
            PREFLIGHT_MEMORY_KIND_READ,
            RV64_REGISTER_AS,
            u64::from(reg_ptr),
            RV64_REGISTER_NUM_LIMBS,
            pc,
        )
    }

    fn expect_reg_write(
        &self,
        timestamp: u32,
        reg_ptr: u32,
        pc: u32,
    ) -> Result<&PreflightMemoryAccessAux<F>, ExecutionError> {
        self.expect(
            timestamp,
            PREFLIGHT_MEMORY_KIND_WRITE,
            RV64_REGISTER_AS,
            u64::from(reg_ptr),
            RV64_REGISTER_NUM_LIMBS,
            pc,
        )
    }

    fn expect_memory_read(
        &self,
        timestamp: u32,
        address: u32,
        pc: u32,
    ) -> Result<&PreflightMemoryAccessAux<F>, ExecutionError> {
        self.expect(
            timestamp,
            PREFLIGHT_MEMORY_KIND_READ,
            RV64_MEMORY_AS,
            u64::from(address),
            RV64_REGISTER_NUM_LIMBS,
            pc,
        )
    }

    fn expect_memory_write(
        &self,
        timestamp: u32,
        addr_space: u32,
        address: u32,
        pc: u32,
    ) -> Result<&PreflightMemoryAccessAux<F>, ExecutionError> {
        self.expect(
            timestamp,
            PREFLIGHT_MEMORY_KIND_WRITE,
            addr_space,
            u64::from(address),
            RV64_REGISTER_NUM_LIMBS,
            pc,
        )
    }
}

pub fn generate_rv64im_record_arenas_from_logs<F: PrimeField32, RA: Rv64StandardRecordArena<F>>(
    exe: &VmExe<F>,
    output: &mut RvrPreflightOutput<F>,
    capacities: &[(usize, usize)],
    pc_to_air_idx: &[Option<usize>],
) -> Result<Vec<RA>, ExecutionError> {
    let mut registry = LogNativeAssemblerRegistry::new();
    crate::Rv64ImConfig::default().extend_rvr_log_native(&mut registry);
    generate_record_arenas_from_logs(&registry, exe, output, capacities, pc_to_air_idx)
}

/// Host mirror of the C `PreflightAddSubRecord` (R3 L1+L5 compact record),
/// shared by every migrated 2-read-1-write single-row shape ("alu3"): the
/// dynamic witness only — program-redundant operands are re-derived from the
/// instruction at `from_pc` by the per-opcode inline assemblers. The 8-byte
/// read/write values are kept as u64s; shape-specific limb views come from
/// [`u16x4`] / [`bytes8`]. Parsed field-by-field from little-endian bytes, so
/// no alignment requirements.
struct PreflightAlu3Compact {
    from_pc: u32,
    from_timestamp: u32,
    reads_prev_timestamp: [u32; 2],
    write_prev_timestamp: u32,
    write_prev_data: u64,
    b: u64,
    c: u64,
}

// Drift guard: the compact stride must match the C record the preflight
// tracer writes (guarded by `_Static_assert`s on the C side).
const _: () = assert!(PREFLIGHT_ADDSUB_RECORD_SIZE == 44);

impl PreflightAlu3Compact {
    /// Parse + order guard: the C emits one record per migrated retired
    /// instruction in program-log order, so the record's `from_pc` must match
    /// the program-log pc being assembled.
    fn read_for_pc(bytes: &[u8], pc: u32) -> Result<Self, ExecutionError> {
        debug_assert_eq!(bytes.len(), PREFLIGHT_ADDSUB_RECORD_SIZE);
        let u32_at =
            |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().expect("4-byte field"));
        let u64_at =
            |at: usize| u64::from_le_bytes(bytes[at..at + 8].try_into().expect("8-byte field"));
        let compact = Self {
            from_pc: u32_at(0),
            from_timestamp: u32_at(4),
            reads_prev_timestamp: [u32_at(8), u32_at(12)],
            write_prev_timestamp: u32_at(16),
            write_prev_data: u64_at(20),
            b: u64_at(28),
            c: u64_at(36),
        };
        if compact.from_pc != pc {
            return Err(ExecutionError::RvrExecution(format!(
                "inline record order mismatch: record from_pc {:#x} vs program-log pc {pc:#x}",
                compact.from_pc
            )));
        }
        Ok(compact)
    }
}

/// u16 limb view of an 8-byte register value (U16-cell chips).
fn u16x4(value: u64) -> [u16; BLOCK_FE_WIDTH] {
    core::array::from_fn(|i| (value >> (16 * i)) as u16)
}

/// Byte limb view of an 8-byte register value (byte-cell chips).
fn bytes8(value: u64) -> [u8; RV64_REGISTER_NUM_LIMBS] {
    value.to_le_bytes()
}

/// Expand one C-written compact record into the full AddSub adapter+core
/// record, re-deriving the program-redundant operands
/// (rd_ptr/rs1_ptr/rs2/rs2_as/rs2_imm_sign/local_opcode) from the instruction
/// exactly as [`fill_base_alu_u16_adapter`] does from the log path.
fn assemble_add_sub_inline<F: PrimeField32, RA: Rv64IRecordArena<F>>(
    arena: &mut RA,
    instruction: &Instruction<F>,
    compact: &[u8],
    pc: u32,
) -> Result<(), ExecutionError> {
    let compact = PreflightAlu3Compact::read_for_pc(compact, pc)?;
    let (adapter_record, core_record): (
        &mut Rv64BaseAluU16AdapterRecord,
        &mut AddSubCoreRecord<BLOCK_FE_WIDTH>,
    ) = arena.alloc(EmptyAdapterCoreLayout::<F, Rv64BaseAluU16AdapterExecutor>::new());
    fill_base_alu_u16_from_compact(&compact, instruction, pc, adapter_record);
    core_record.b = u16x4(compact.b);
    core_record.c = u16x4(compact.c);
    core_record.local_opcode = instruction
        .opcode
        .local_opcode_idx(BaseAluOpcode::CLASS_OFFSET) as u8;
    Ok(())
}

/// Fill the W u16 adapter from an alu3 compact record. The W adapter keeps
/// the source high halves and result-word sign witness in addition to the
/// common alu3 fields.
fn fill_base_alu_w_u16_from_compact<F: PrimeField32>(
    compact: &PreflightAlu3Compact,
    instruction: &Instruction<F>,
    pc: u32,
    result_word: u32,
    record: &mut Rv64BaseAluWU16AdapterRecord,
) {
    let operands = derive_base_alu_u16_operands(instruction);
    let rs1 = u16x4(compact.b);
    let rs2 = u16x4(compact.c);
    record.from_pc = pc;
    record.from_timestamp = compact.from_timestamp;
    record.rd_ptr = operands.rd_ptr;
    record.rs1_ptr = operands.rs1_ptr;
    record.rs1_high = [rs1[2], rs1[3]];
    record.rs2 = operands.rs2;
    record.rs2_as = operands.rs2_as;
    record.rs2_imm_sign = operands.rs2_imm_sign;
    record.rs2_high = [rs2[2], rs2[3]];
    record.result_high = (result_word >> U16_BITS) as u16;
    record.result_sign = (result_word >> (u32::BITS - 1)) as u8;
    record.reads_aux[0].prev_timestamp = compact.reads_prev_timestamp[0];
    record.reads_aux[1].prev_timestamp = compact.reads_prev_timestamp[1];
    record.writes_aux = MemoryWriteAuxRecord {
        prev_timestamp: compact.write_prev_timestamp,
        prev_data: u16x4(compact.write_prev_data),
    };
}

fn assemble_add_sub_w_inline<F: PrimeField32, RA: Rv64IRecordArena<F>>(
    arena: &mut RA,
    instruction: &Instruction<F>,
    compact: &[u8],
    pc: u32,
) -> Result<(), ExecutionError> {
    let compact = PreflightAlu3Compact::read_for_pc(compact, pc)?;
    let local_opcode = BaseAluWOpcode::from_repr(
        instruction
            .opcode
            .local_opcode_idx(BaseAluWOpcode::CLASS_OFFSET),
    )
    .expect("opcode already classified");
    let result = match local_opcode {
        BaseAluWOpcode::ADDW => (compact.b as u32).wrapping_add(compact.c as u32),
        BaseAluWOpcode::SUBW => (compact.b as u32).wrapping_sub(compact.c as u32),
    };
    let (adapter_record, core_record): (
        &mut Rv64BaseAluWU16AdapterRecord,
        &mut AddSubCoreRecord<RV64_WORD_U16_LIMBS>,
    ) = arena.alloc(EmptyAdapterCoreLayout::<F, Rv64BaseAluWU16AdapterExecutor>::new());
    fill_base_alu_w_u16_from_compact(&compact, instruction, pc, result, adapter_record);
    core_record.b = u16x4(compact.b)[..RV64_WORD_U16_LIMBS].try_into().unwrap();
    core_record.c = u16x4(compact.c)[..RV64_WORD_U16_LIMBS].try_into().unwrap();
    core_record.local_opcode = match local_opcode {
        BaseAluWOpcode::ADDW => BaseAluOpcode::ADD as u8,
        BaseAluWOpcode::SUBW => BaseAluOpcode::SUB as u8,
    };
    Ok(())
}

/// Fill a Mult adapter record (Mul/MulH/DivRem) from a compact alu3 record,
/// mirroring [`fill_mult_adapter`]'s derivation from the log path.
fn fill_mult_adapter_from_compact<F: PrimeField32>(
    compact: &PreflightAlu3Compact,
    instruction: &Instruction<F>,
    pc: u32,
    record: &mut Rv64MultAdapterRecord,
) {
    record.from_pc = pc;
    record.from_timestamp = compact.from_timestamp;
    record.rs1_ptr = instruction.b.as_canonical_u32();
    record.rs2_ptr = instruction.c.as_canonical_u32();
    record.rd_ptr = instruction.a.as_canonical_u32();
    record.reads_aux[0].prev_timestamp = compact.reads_prev_timestamp[0];
    record.reads_aux[1].prev_timestamp = compact.reads_prev_timestamp[1];
    record.writes_aux = MemoryWriteAuxRecord {
        prev_timestamp: compact.write_prev_timestamp,
        prev_data: bytes8(compact.write_prev_data),
    };
}

fn assemble_mul_inline<F: PrimeField32, RA: Rv64MRecordArena<F>>(
    arena: &mut RA,
    instruction: &Instruction<F>,
    compact: &[u8],
    pc: u32,
) -> Result<(), ExecutionError> {
    let compact = PreflightAlu3Compact::read_for_pc(compact, pc)?;
    let (adapter_record, core_record): (
        &mut Rv64MultAdapterRecord,
        &mut MultiplicationCoreRecord<RV64_REGISTER_NUM_LIMBS, RV64_BYTE_BITS>,
    ) = arena.alloc(EmptyAdapterCoreLayout::<F, Rv64MultAdapterExecutor>::new());
    fill_mult_adapter_from_compact(&compact, instruction, pc, adapter_record);
    core_record.b = bytes8(compact.b);
    core_record.c = bytes8(compact.c);
    Ok(())
}

fn assemble_mulh_inline<F: PrimeField32, RA: Rv64MRecordArena<F>>(
    arena: &mut RA,
    instruction: &Instruction<F>,
    compact: &[u8],
    pc: u32,
) -> Result<(), ExecutionError> {
    let compact = PreflightAlu3Compact::read_for_pc(compact, pc)?;
    let (adapter_record, core_record): (
        &mut Rv64MultAdapterRecord,
        &mut MulHCoreRecord<RV64_REGISTER_NUM_LIMBS, RV64_BYTE_BITS>,
    ) = arena.alloc(EmptyAdapterCoreLayout::<F, Rv64MultAdapterExecutor>::new());
    fill_mult_adapter_from_compact(&compact, instruction, pc, adapter_record);
    core_record.b = bytes8(compact.b);
    core_record.c = bytes8(compact.c);
    core_record.local_opcode = instruction
        .opcode
        .local_opcode_idx(MulHOpcode::CLASS_OFFSET) as u8;
    Ok(())
}

fn assemble_divrem_inline<F: PrimeField32, RA: Rv64MRecordArena<F>>(
    arena: &mut RA,
    instruction: &Instruction<F>,
    compact: &[u8],
    pc: u32,
) -> Result<(), ExecutionError> {
    let compact = PreflightAlu3Compact::read_for_pc(compact, pc)?;
    let (adapter_record, core_record): (
        &mut Rv64MultAdapterRecord,
        &mut DivRemCoreRecord<RV64_REGISTER_NUM_LIMBS>,
    ) = arena.alloc(EmptyAdapterCoreLayout::<F, Rv64MultAdapterExecutor>::new());
    fill_mult_adapter_from_compact(&compact, instruction, pc, adapter_record);
    core_record.b = bytes8(compact.b);
    core_record.c = bytes8(compact.c);
    core_record.local_opcode = instruction
        .opcode
        .local_opcode_idx(DivRemOpcode::CLASS_OFFSET) as u8;
    Ok(())
}

/// RV64 W-op result kinds; see [`run_mul_div_w_result`].
#[derive(Clone, Copy)]
enum MulDivWKind {
    Mul,
    Div,
    Divu,
    Rem,
    Remu,
}

/// RV64 W-op result word, matching the generated C's `rv_divw`-family
/// semantics (div-by-zero -> all ones / dividend, MIN/-1 overflow -> MIN).
/// Needed because the compact record does not carry the written value, from
/// which [`fill_mult_w_adapter`] derives `result_word_msl`/`result_sign`.
fn run_mul_div_w_result(op: MulDivWKind, b: u32, c: u32) -> u32 {
    match op {
        MulDivWKind::Mul => b.wrapping_mul(c),
        MulDivWKind::Div => {
            let (b, c) = (b as i32, c as i32);
            if c == 0 {
                u32::MAX
            } else if b == i32::MIN && c == -1 {
                b as u32
            } else {
                (b / c) as u32
            }
        }
        MulDivWKind::Divu => {
            if c == 0 {
                u32::MAX
            } else {
                b / c
            }
        }
        MulDivWKind::Rem => {
            let (b, c) = (b as i32, c as i32);
            if c == 0 {
                b as u32
            } else if b == i32::MIN && c == -1 {
                0
            } else {
                (b % c) as u32
            }
        }
        MulDivWKind::Remu => {
            if c == 0 {
                b
            } else {
                b % c
            }
        }
    }
}

/// Fill a MultW adapter record from a compact alu3 record, mirroring
/// [`fill_mult_w_adapter`]: the register-read high bytes come from the full
/// 8-byte values the compact record carries, and the result word (for
/// `result_word_msl`/`result_sign`) is recomputed from the operands.
fn fill_mult_w_adapter_from_compact<F: PrimeField32>(
    compact: &PreflightAlu3Compact,
    instruction: &Instruction<F>,
    pc: u32,
    result_word: u32,
    record: &mut Rv64MultWAdapterRecord,
) {
    let rs1_full = bytes8(compact.b);
    let rs2_full = bytes8(compact.c);
    record.from_pc = pc;
    record.from_timestamp = compact.from_timestamp;
    record.rs1_ptr = instruction.b.as_canonical_u32();
    record.rs2_ptr = instruction.c.as_canonical_u32();
    record.rd_ptr = instruction.a.as_canonical_u32();
    record
        .rs1_high
        .copy_from_slice(&rs1_full[RV64_WORD_NUM_LIMBS..]);
    record
        .rs2_high
        .copy_from_slice(&rs2_full[RV64_WORD_NUM_LIMBS..]);
    record.reads_aux[0].prev_timestamp = compact.reads_prev_timestamp[0];
    record.reads_aux[1].prev_timestamp = compact.reads_prev_timestamp[1];
    record.result_word_msl = result_word.to_le_bytes()[RV64_WORD_NUM_LIMBS - 1];
    record.result_sign = record.result_word_msl >> (RV64_BYTE_BITS as u8 - 1);
    record.writes_aux = MemoryWriteAuxRecord {
        prev_timestamp: compact.write_prev_timestamp,
        prev_data: bytes8(compact.write_prev_data),
    };
}

fn assemble_mul_w_inline<F: PrimeField32, RA: Rv64MRecordArena<F>>(
    arena: &mut RA,
    instruction: &Instruction<F>,
    compact: &[u8],
    pc: u32,
) -> Result<(), ExecutionError> {
    let compact = PreflightAlu3Compact::read_for_pc(compact, pc)?;
    let (adapter_record, core_record): (
        &mut Rv64MultWAdapterRecord,
        &mut MultiplicationCoreRecord<RV64_WORD_NUM_LIMBS, RV64_BYTE_BITS>,
    ) = arena.alloc(EmptyAdapterCoreLayout::<F, Rv64MultWAdapterExecutor>::new());
    let result = run_mul_div_w_result(MulDivWKind::Mul, compact.b as u32, compact.c as u32);
    fill_mult_w_adapter_from_compact(&compact, instruction, pc, result, adapter_record);
    core_record.b = bytes8(compact.b)[..RV64_WORD_NUM_LIMBS].try_into().unwrap();
    core_record.c = bytes8(compact.c)[..RV64_WORD_NUM_LIMBS].try_into().unwrap();
    Ok(())
}

fn assemble_divrem_w_inline<F: PrimeField32, RA: Rv64MRecordArena<F>>(
    arena: &mut RA,
    instruction: &Instruction<F>,
    compact: &[u8],
    pc: u32,
) -> Result<(), ExecutionError> {
    let compact = PreflightAlu3Compact::read_for_pc(compact, pc)?;
    let local_opcode = DivRemWOpcode::from_repr(
        instruction
            .opcode
            .local_opcode_idx(DivRemWOpcode::CLASS_OFFSET),
    )
    .expect("opcode already classified");
    let kind = match local_opcode {
        DivRemWOpcode::DIVW => MulDivWKind::Div,
        DivRemWOpcode::DIVUW => MulDivWKind::Divu,
        DivRemWOpcode::REMW => MulDivWKind::Rem,
        DivRemWOpcode::REMUW => MulDivWKind::Remu,
    };
    let (adapter_record, core_record): (
        &mut Rv64MultWAdapterRecord,
        &mut DivRemCoreRecord<RV64_WORD_NUM_LIMBS>,
    ) = arena.alloc(EmptyAdapterCoreLayout::<F, Rv64MultWAdapterExecutor>::new());
    let result = run_mul_div_w_result(kind, compact.b as u32, compact.c as u32);
    fill_mult_w_adapter_from_compact(&compact, instruction, pc, result, adapter_record);
    core_record.b = bytes8(compact.b)[..RV64_WORD_NUM_LIMBS].try_into().unwrap();
    core_record.c = bytes8(compact.c)[..RV64_WORD_NUM_LIMBS].try_into().unwrap();
    core_record.local_opcode = local_opcode as u8;
    Ok(())
}

fn base_alu_w_arena_geometry<F: PrimeField32>(
    core_size: usize,
    core_align: usize,
    core_b: usize,
    core_c: usize,
    core_local_opcode: usize,
) -> ArenaNativeGeometry {
    ArenaNativeGeometry {
        adapter_size: size_of::<Rv64BaseAluWU16AdapterRecord>(),
        adapter_align: align_of::<Rv64BaseAluWU16AdapterRecord>(),
        core_size,
        core_align,
        core_off_matrix: <Rv64BaseAluWU16AdapterExecutor as AdapterTraceExecutor<F>>::WIDTH
            * size_of::<F>(),
        layout: ArenaNativeLayout::Alu3(Alu3ArenaFieldOffsets {
            from_pc: core::mem::offset_of!(Rv64BaseAluWU16AdapterRecord, from_pc),
            from_timestamp: core::mem::offset_of!(Rv64BaseAluWU16AdapterRecord, from_timestamp),
            rd_ptr: core::mem::offset_of!(Rv64BaseAluWU16AdapterRecord, rd_ptr),
            rs1_ptr: core::mem::offset_of!(Rv64BaseAluWU16AdapterRecord, rs1_ptr),
            rs2: core::mem::offset_of!(Rv64BaseAluWU16AdapterRecord, rs2),
            rs2_as: core::mem::offset_of!(Rv64BaseAluWU16AdapterRecord, rs2_as),
            rs2_imm_sign: core::mem::offset_of!(Rv64BaseAluWU16AdapterRecord, rs2_imm_sign),
            reads_aux0_prev_ts: core::mem::offset_of!(Rv64BaseAluWU16AdapterRecord, reads_aux)
                + core::mem::offset_of!(MemoryReadAuxRecord, prev_timestamp),
            reads_aux1_prev_ts: core::mem::offset_of!(Rv64BaseAluWU16AdapterRecord, reads_aux)
                + size_of::<MemoryReadAuxRecord>()
                + core::mem::offset_of!(MemoryReadAuxRecord, prev_timestamp),
            write_prev_ts: core::mem::offset_of!(Rv64BaseAluWU16AdapterRecord, writes_aux)
                + core::mem::offset_of!(
                    MemoryWriteAuxRecord<u16, BLOCK_FE_WIDTH>,
                    prev_timestamp
                ),
            write_prev_data: core::mem::offset_of!(Rv64BaseAluWU16AdapterRecord, writes_aux)
                + core::mem::offset_of!(MemoryWriteAuxRecord<u16, BLOCK_FE_WIDTH>, prev_data),
            core_b,
            core_c,
            core_local_opcode,
            w: Some(Alu3WArenaFieldOffsets {
                rs1_high: core::mem::offset_of!(Rv64BaseAluWU16AdapterRecord, rs1_high),
                rs2_high: core::mem::offset_of!(Rv64BaseAluWU16AdapterRecord, rs2_high),
                result_word_msl: core::mem::offset_of!(Rv64BaseAluWU16AdapterRecord, result_high),
                result_sign: core::mem::offset_of!(Rv64BaseAluWU16AdapterRecord, result_sign),
                result_word_msl_shift: U16_BITS as u8,
                result_word_msl_bytes: size_of::<u16>() as u8,
            }),
        }),
    }
}

fn mult_w_arena_geometry<F: PrimeField32>(
    core_size: usize,
    core_align: usize,
    core_b: usize,
    core_c: usize,
    core_local_opcode: usize,
) -> ArenaNativeGeometry {
    ArenaNativeGeometry {
        adapter_size: size_of::<Rv64MultWAdapterRecord>(),
        adapter_align: align_of::<Rv64MultWAdapterRecord>(),
        core_size,
        core_align,
        core_off_matrix: <Rv64MultWAdapterExecutor as AdapterTraceExecutor<F>>::WIDTH
            * size_of::<F>(),
        layout: ArenaNativeLayout::Alu3(Alu3ArenaFieldOffsets {
            from_pc: core::mem::offset_of!(Rv64MultWAdapterRecord, from_pc),
            from_timestamp: core::mem::offset_of!(Rv64MultWAdapterRecord, from_timestamp),
            rd_ptr: core::mem::offset_of!(Rv64MultWAdapterRecord, rd_ptr),
            rs1_ptr: core::mem::offset_of!(Rv64MultWAdapterRecord, rs1_ptr),
            rs2: core::mem::offset_of!(Rv64MultWAdapterRecord, rs2_ptr),
            rs2_as: usize::MAX,
            rs2_imm_sign: usize::MAX,
            reads_aux0_prev_ts: core::mem::offset_of!(Rv64MultWAdapterRecord, reads_aux)
                + core::mem::offset_of!(MemoryReadAuxRecord, prev_timestamp),
            reads_aux1_prev_ts: core::mem::offset_of!(Rv64MultWAdapterRecord, reads_aux)
                + size_of::<MemoryReadAuxRecord>()
                + core::mem::offset_of!(MemoryReadAuxRecord, prev_timestamp),
            write_prev_ts: core::mem::offset_of!(Rv64MultWAdapterRecord, writes_aux)
                + core::mem::offset_of!(
                    MemoryWriteAuxRecord<u8, RV64_REGISTER_NUM_LIMBS>,
                    prev_timestamp
                ),
            write_prev_data: core::mem::offset_of!(Rv64MultWAdapterRecord, writes_aux)
                + core::mem::offset_of!(
                    MemoryWriteAuxRecord<u8, RV64_REGISTER_NUM_LIMBS>,
                    prev_data
                ),
            core_b,
            core_c,
            core_local_opcode,
            w: Some(Alu3WArenaFieldOffsets {
                rs1_high: core::mem::offset_of!(Rv64MultWAdapterRecord, rs1_high),
                rs2_high: core::mem::offset_of!(Rv64MultWAdapterRecord, rs2_high),
                result_word_msl: core::mem::offset_of!(Rv64MultWAdapterRecord, result_word_msl),
                result_sign: core::mem::offset_of!(Rv64MultWAdapterRecord, result_sign),
                result_word_msl_shift: (RV64_BYTE_BITS * (RV64_WORD_NUM_LIMBS - 1)) as u8,
                result_word_msl_bytes: size_of::<u8>() as u8,
            }),
        }),
    }
}

impl<F, RA> VmRvrLogNativeExtension<F, RA> for crate::Rv64I
where
    F: PrimeField32,
    RA: Rv64IRecordArena<F>,
{
    fn extend_rvr_log_native(&self, registry: &mut LogNativeAssemblerRegistry<F, RA>) {
        registry.register(
            [BaseAluOpcode::ADD, BaseAluOpcode::SUB].map(|opcode| opcode.global_opcode()),
            assemble_add_sub::<F, RA>,
        );
        registry.register_inline_arena_native(
            [BaseAluOpcode::ADD, BaseAluOpcode::SUB].map(|opcode| opcode.global_opcode()),
            PREFLIGHT_ADDSUB_RECORD_SIZE,
            assemble_add_sub_inline::<F, RA>,
            ArenaNativeGeometry {
                adapter_size: size_of::<Rv64BaseAluU16AdapterRecord>(),
                adapter_align: align_of::<Rv64BaseAluU16AdapterRecord>(),
                core_size: size_of::<AddSubCoreRecord<BLOCK_FE_WIDTH>>(),
                core_align: align_of::<AddSubCoreRecord<BLOCK_FE_WIDTH>>(),
                core_off_matrix: <Rv64BaseAluU16AdapterExecutor as AdapterTraceExecutor<F>>::WIDTH
                    * size_of::<F>(),
                layout: ArenaNativeLayout::Alu3(Alu3ArenaFieldOffsets {
                    from_pc: core::mem::offset_of!(Rv64BaseAluU16AdapterRecord, from_pc),
                    from_timestamp: core::mem::offset_of!(
                        Rv64BaseAluU16AdapterRecord,
                        from_timestamp
                    ),
                    rd_ptr: core::mem::offset_of!(Rv64BaseAluU16AdapterRecord, rd_ptr),
                    rs1_ptr: core::mem::offset_of!(Rv64BaseAluU16AdapterRecord, rs1_ptr),
                    rs2: core::mem::offset_of!(Rv64BaseAluU16AdapterRecord, rs2),
                    rs2_as: core::mem::offset_of!(Rv64BaseAluU16AdapterRecord, rs2_as),
                    rs2_imm_sign: core::mem::offset_of!(Rv64BaseAluU16AdapterRecord, rs2_imm_sign),
                    reads_aux0_prev_ts: core::mem::offset_of!(
                        Rv64BaseAluU16AdapterRecord,
                        reads_aux
                    ) + core::mem::offset_of!(
                        MemoryReadAuxRecord,
                        prev_timestamp
                    ),
                    reads_aux1_prev_ts: core::mem::offset_of!(
                        Rv64BaseAluU16AdapterRecord,
                        reads_aux
                    ) + size_of::<MemoryReadAuxRecord>()
                        + core::mem::offset_of!(MemoryReadAuxRecord, prev_timestamp),
                    write_prev_ts: core::mem::offset_of!(Rv64BaseAluU16AdapterRecord, writes_aux)
                        + core::mem::offset_of!(
                            MemoryWriteAuxRecord<u16, BLOCK_FE_WIDTH>,
                            prev_timestamp
                        ),
                    write_prev_data: core::mem::offset_of!(Rv64BaseAluU16AdapterRecord, writes_aux)
                        + core::mem::offset_of!(
                            MemoryWriteAuxRecord<u16, BLOCK_FE_WIDTH>,
                            prev_data
                        ),
                    core_b: core::mem::offset_of!(AddSubCoreRecord<BLOCK_FE_WIDTH>, b),
                    core_c: core::mem::offset_of!(AddSubCoreRecord<BLOCK_FE_WIDTH>, c),
                    core_local_opcode: core::mem::offset_of!(
                        AddSubCoreRecord<BLOCK_FE_WIDTH>,
                        local_opcode
                    ),
                    w: None,
                }),
            },
        );
        registry.register_inline_arena_native(
            [BaseAluOpcode::XOR, BaseAluOpcode::OR, BaseAluOpcode::AND]
                .map(|opcode| opcode.global_opcode()),
            PREFLIGHT_ADDSUB_RECORD_SIZE,
            assemble_bitwise_inline::<F, RA>,
            ArenaNativeGeometry {
                adapter_size: size_of::<Rv64BaseAluAdapterRecord>(),
                adapter_align: align_of::<Rv64BaseAluAdapterRecord>(),
                core_size: size_of::<BitwiseLogicCoreRecord<RV64_REGISTER_NUM_LIMBS>>(),
                core_align: align_of::<BitwiseLogicCoreRecord<RV64_REGISTER_NUM_LIMBS>>(),
                core_off_matrix:
                    <Rv64BaseAluAdapterExecutor<RV64_BYTE_BITS> as AdapterTraceExecutor<F>>::WIDTH
                        * size_of::<F>(),
                layout: ArenaNativeLayout::Alu3(Alu3ArenaFieldOffsets {
                    from_pc: core::mem::offset_of!(Rv64BaseAluAdapterRecord, from_pc),
                    from_timestamp: core::mem::offset_of!(Rv64BaseAluAdapterRecord, from_timestamp),
                    rd_ptr: core::mem::offset_of!(Rv64BaseAluAdapterRecord, rd_ptr),
                    rs1_ptr: core::mem::offset_of!(Rv64BaseAluAdapterRecord, rs1_ptr),
                    rs2: core::mem::offset_of!(Rv64BaseAluAdapterRecord, rs2),
                    rs2_as: core::mem::offset_of!(Rv64BaseAluAdapterRecord, rs2_as),
                    // The byte adapter has no imm-sign field.
                    rs2_imm_sign: usize::MAX,
                    reads_aux0_prev_ts: core::mem::offset_of!(Rv64BaseAluAdapterRecord, reads_aux)
                        + core::mem::offset_of!(MemoryReadAuxRecord, prev_timestamp),
                    reads_aux1_prev_ts: core::mem::offset_of!(Rv64BaseAluAdapterRecord, reads_aux)
                        + size_of::<MemoryReadAuxRecord>()
                        + core::mem::offset_of!(MemoryReadAuxRecord, prev_timestamp),
                    write_prev_ts: core::mem::offset_of!(Rv64BaseAluAdapterRecord, writes_aux)
                        + core::mem::offset_of!(
                            MemoryWriteAuxRecord<u8, RV64_REGISTER_NUM_LIMBS>,
                            prev_timestamp
                        ),
                    write_prev_data: core::mem::offset_of!(Rv64BaseAluAdapterRecord, writes_aux)
                        + core::mem::offset_of!(
                            MemoryWriteAuxRecord<u8, RV64_REGISTER_NUM_LIMBS>,
                            prev_data
                        ),
                    core_b: core::mem::offset_of!(
                        BitwiseLogicCoreRecord<RV64_REGISTER_NUM_LIMBS>,
                        b
                    ),
                    core_c: core::mem::offset_of!(
                        BitwiseLogicCoreRecord<RV64_REGISTER_NUM_LIMBS>,
                        c
                    ),
                    core_local_opcode: core::mem::offset_of!(
                        BitwiseLogicCoreRecord<RV64_REGISTER_NUM_LIMBS>,
                        local_opcode
                    ),
                    w: None,
                }),
            },
        );
        registry.register_inline_arena_native(
            LessThanOpcode::iter().map(|opcode| opcode.global_opcode()),
            PREFLIGHT_ADDSUB_RECORD_SIZE,
            assemble_less_than_inline::<F, RA>,
            ArenaNativeGeometry {
                adapter_size: size_of::<Rv64BaseAluU16AdapterRecord>(),
                adapter_align: align_of::<Rv64BaseAluU16AdapterRecord>(),
                core_size: size_of::<LessThanCoreRecord<BLOCK_FE_WIDTH, U16_BITS>>(),
                core_align: align_of::<LessThanCoreRecord<BLOCK_FE_WIDTH, U16_BITS>>(),
                core_off_matrix: <Rv64BaseAluU16AdapterExecutor as AdapterTraceExecutor<F>>::WIDTH
                    * size_of::<F>(),
                layout: ArenaNativeLayout::Alu3(Alu3ArenaFieldOffsets {
                    from_pc: core::mem::offset_of!(Rv64BaseAluU16AdapterRecord, from_pc),
                    from_timestamp: core::mem::offset_of!(
                        Rv64BaseAluU16AdapterRecord,
                        from_timestamp
                    ),
                    rd_ptr: core::mem::offset_of!(Rv64BaseAluU16AdapterRecord, rd_ptr),
                    rs1_ptr: core::mem::offset_of!(Rv64BaseAluU16AdapterRecord, rs1_ptr),
                    rs2: core::mem::offset_of!(Rv64BaseAluU16AdapterRecord, rs2),
                    rs2_as: core::mem::offset_of!(Rv64BaseAluU16AdapterRecord, rs2_as),
                    rs2_imm_sign: core::mem::offset_of!(Rv64BaseAluU16AdapterRecord, rs2_imm_sign),
                    reads_aux0_prev_ts: core::mem::offset_of!(
                        Rv64BaseAluU16AdapterRecord,
                        reads_aux
                    ) + core::mem::offset_of!(MemoryReadAuxRecord, prev_timestamp),
                    reads_aux1_prev_ts: core::mem::offset_of!(
                        Rv64BaseAluU16AdapterRecord,
                        reads_aux
                    ) + size_of::<MemoryReadAuxRecord>()
                        + core::mem::offset_of!(MemoryReadAuxRecord, prev_timestamp),
                    write_prev_ts: core::mem::offset_of!(Rv64BaseAluU16AdapterRecord, writes_aux)
                        + core::mem::offset_of!(
                            MemoryWriteAuxRecord<u16, BLOCK_FE_WIDTH>,
                            prev_timestamp
                        ),
                    write_prev_data: core::mem::offset_of!(Rv64BaseAluU16AdapterRecord, writes_aux)
                        + core::mem::offset_of!(
                            MemoryWriteAuxRecord<u16, BLOCK_FE_WIDTH>,
                            prev_data
                        ),
                    core_b: core::mem::offset_of!(LessThanCoreRecord<BLOCK_FE_WIDTH, U16_BITS>, b),
                    core_c: core::mem::offset_of!(LessThanCoreRecord<BLOCK_FE_WIDTH, U16_BITS>, c),
                    core_local_opcode: core::mem::offset_of!(LessThanCoreRecord<BLOCK_FE_WIDTH, U16_BITS>, local_opcode),
                    w: None,
                }),
            },
        );
        registry.register_inline_arena_native(
            [ShiftOpcode::SLL, ShiftOpcode::SRL].map(|opcode| opcode.global_opcode()),
            PREFLIGHT_ADDSUB_RECORD_SIZE,
            assemble_shift_inline::<F, RA>,
            ArenaNativeGeometry {
                adapter_size: size_of::<Rv64BaseAluU16AdapterRecord>(),
                adapter_align: align_of::<Rv64BaseAluU16AdapterRecord>(),
                core_size: size_of::<ShiftLogicalCoreRecord<BLOCK_FE_WIDTH, U16_BITS>>(),
                core_align: align_of::<ShiftLogicalCoreRecord<BLOCK_FE_WIDTH, U16_BITS>>(),
                core_off_matrix: <Rv64BaseAluU16AdapterExecutor as AdapterTraceExecutor<F>>::WIDTH
                    * size_of::<F>(),
                layout: ArenaNativeLayout::Alu3(Alu3ArenaFieldOffsets {
                    from_pc: core::mem::offset_of!(Rv64BaseAluU16AdapterRecord, from_pc),
                    from_timestamp: core::mem::offset_of!(
                        Rv64BaseAluU16AdapterRecord,
                        from_timestamp
                    ),
                    rd_ptr: core::mem::offset_of!(Rv64BaseAluU16AdapterRecord, rd_ptr),
                    rs1_ptr: core::mem::offset_of!(Rv64BaseAluU16AdapterRecord, rs1_ptr),
                    rs2: core::mem::offset_of!(Rv64BaseAluU16AdapterRecord, rs2),
                    rs2_as: core::mem::offset_of!(Rv64BaseAluU16AdapterRecord, rs2_as),
                    rs2_imm_sign: core::mem::offset_of!(Rv64BaseAluU16AdapterRecord, rs2_imm_sign),
                    reads_aux0_prev_ts: core::mem::offset_of!(
                        Rv64BaseAluU16AdapterRecord,
                        reads_aux
                    ) + core::mem::offset_of!(MemoryReadAuxRecord, prev_timestamp),
                    reads_aux1_prev_ts: core::mem::offset_of!(
                        Rv64BaseAluU16AdapterRecord,
                        reads_aux
                    ) + size_of::<MemoryReadAuxRecord>()
                        + core::mem::offset_of!(MemoryReadAuxRecord, prev_timestamp),
                    write_prev_ts: core::mem::offset_of!(Rv64BaseAluU16AdapterRecord, writes_aux)
                        + core::mem::offset_of!(
                            MemoryWriteAuxRecord<u16, BLOCK_FE_WIDTH>,
                            prev_timestamp
                        ),
                    write_prev_data: core::mem::offset_of!(Rv64BaseAluU16AdapterRecord, writes_aux)
                        + core::mem::offset_of!(
                            MemoryWriteAuxRecord<u16, BLOCK_FE_WIDTH>,
                            prev_data
                        ),
                    core_b: core::mem::offset_of!(ShiftLogicalCoreRecord<BLOCK_FE_WIDTH, U16_BITS>, b),
                    core_c: core::mem::offset_of!(ShiftLogicalCoreRecord<BLOCK_FE_WIDTH, U16_BITS>, c),
                    core_local_opcode: core::mem::offset_of!(ShiftLogicalCoreRecord<BLOCK_FE_WIDTH, U16_BITS>, local_opcode),
                    w: None,
                }),
            },
        );
        registry.register_inline_arena_native(
            [ShiftOpcode::SRA].map(|opcode| opcode.global_opcode()),
            PREFLIGHT_ADDSUB_RECORD_SIZE,
            assemble_shift_inline::<F, RA>,
            // SRA's core record has no local_opcode field: the sentinel
            // tells the emitter to skip that store.
            ArenaNativeGeometry {
                adapter_size: size_of::<Rv64BaseAluU16AdapterRecord>(),
                adapter_align: align_of::<Rv64BaseAluU16AdapterRecord>(),
                core_size: size_of::<ShiftRightArithmeticCoreRecord<BLOCK_FE_WIDTH, U16_BITS>>(),
                core_align: align_of::<ShiftRightArithmeticCoreRecord<BLOCK_FE_WIDTH, U16_BITS>>(),
                core_off_matrix: <Rv64BaseAluU16AdapterExecutor as AdapterTraceExecutor<F>>::WIDTH
                    * size_of::<F>(),
                layout: ArenaNativeLayout::Alu3(Alu3ArenaFieldOffsets {
                    from_pc: core::mem::offset_of!(Rv64BaseAluU16AdapterRecord, from_pc),
                    from_timestamp: core::mem::offset_of!(
                        Rv64BaseAluU16AdapterRecord,
                        from_timestamp
                    ),
                    rd_ptr: core::mem::offset_of!(Rv64BaseAluU16AdapterRecord, rd_ptr),
                    rs1_ptr: core::mem::offset_of!(Rv64BaseAluU16AdapterRecord, rs1_ptr),
                    rs2: core::mem::offset_of!(Rv64BaseAluU16AdapterRecord, rs2),
                    rs2_as: core::mem::offset_of!(Rv64BaseAluU16AdapterRecord, rs2_as),
                    rs2_imm_sign: core::mem::offset_of!(Rv64BaseAluU16AdapterRecord, rs2_imm_sign),
                    reads_aux0_prev_ts: core::mem::offset_of!(
                        Rv64BaseAluU16AdapterRecord,
                        reads_aux
                    ) + core::mem::offset_of!(MemoryReadAuxRecord, prev_timestamp),
                    reads_aux1_prev_ts: core::mem::offset_of!(
                        Rv64BaseAluU16AdapterRecord,
                        reads_aux
                    ) + size_of::<MemoryReadAuxRecord>()
                        + core::mem::offset_of!(MemoryReadAuxRecord, prev_timestamp),
                    write_prev_ts: core::mem::offset_of!(Rv64BaseAluU16AdapterRecord, writes_aux)
                        + core::mem::offset_of!(
                            MemoryWriteAuxRecord<u16, BLOCK_FE_WIDTH>,
                            prev_timestamp
                        ),
                    write_prev_data: core::mem::offset_of!(Rv64BaseAluU16AdapterRecord, writes_aux)
                        + core::mem::offset_of!(
                            MemoryWriteAuxRecord<u16, BLOCK_FE_WIDTH>,
                            prev_data
                        ),
                    core_b: core::mem::offset_of!(ShiftRightArithmeticCoreRecord<BLOCK_FE_WIDTH, U16_BITS>, b),
                    core_c: core::mem::offset_of!(ShiftRightArithmeticCoreRecord<BLOCK_FE_WIDTH, U16_BITS>, c),
                    core_local_opcode: usize::MAX,
                    w: None,
                }),
            },
        );
        registry.register_inline_arena_native(
            BranchEqualOpcode::iter().map(|opcode| opcode.global_opcode()),
            PREFLIGHT_BRANCH2_RECORD_SIZE,
            assemble_branch_eq_inline::<F, RA>,
            ArenaNativeGeometry {
                adapter_size: size_of::<Rv64BranchAdapterRecord>(),
                adapter_align: align_of::<Rv64BranchAdapterRecord>(),
                core_size: size_of::<BranchEqualCoreRecord<BLOCK_FE_WIDTH>>(),
                core_align: align_of::<BranchEqualCoreRecord<BLOCK_FE_WIDTH>>(),
                core_off_matrix: <Rv64BranchAdapterExecutor as AdapterTraceExecutor<F>>::WIDTH
                    * size_of::<F>(),
                layout: ArenaNativeLayout::Branch2(Branch2ArenaFieldOffsets {
                    from_pc: core::mem::offset_of!(Rv64BranchAdapterRecord, from_pc),
                    from_timestamp: core::mem::offset_of!(Rv64BranchAdapterRecord, from_timestamp),
                    rs1_ptr: core::mem::offset_of!(Rv64BranchAdapterRecord, rs1_ptr),
                    rs2_ptr: core::mem::offset_of!(Rv64BranchAdapterRecord, rs2_ptr),
                    reads_aux0_prev_ts: core::mem::offset_of!(Rv64BranchAdapterRecord, reads_aux)
                        + core::mem::offset_of!(MemoryReadAuxRecord, prev_timestamp),
                    reads_aux1_prev_ts: core::mem::offset_of!(Rv64BranchAdapterRecord, reads_aux)
                        + size_of::<MemoryReadAuxRecord>()
                        + core::mem::offset_of!(MemoryReadAuxRecord, prev_timestamp),
                    core_a: core::mem::offset_of!(BranchEqualCoreRecord<BLOCK_FE_WIDTH>, a),
                    core_b: core::mem::offset_of!(BranchEqualCoreRecord<BLOCK_FE_WIDTH>, b),
                    core_imm: core::mem::offset_of!(BranchEqualCoreRecord<BLOCK_FE_WIDTH>, imm),
                    core_local_opcode: core::mem::offset_of!(
                        BranchEqualCoreRecord<BLOCK_FE_WIDTH>,
                        local_opcode
                    ),
                }),
            },
        );
        registry.register_inline_arena_native(
            BranchLessThanOpcode::iter().map(|opcode| opcode.global_opcode()),
            PREFLIGHT_BRANCH2_RECORD_SIZE,
            assemble_branch_lt_inline::<F, RA>,
            ArenaNativeGeometry {
                adapter_size: size_of::<Rv64BranchAdapterRecord>(),
                adapter_align: align_of::<Rv64BranchAdapterRecord>(),
                core_size: size_of::<BranchLessThanCoreRecord<BLOCK_FE_WIDTH, U16_BITS>>(),
                core_align: align_of::<BranchLessThanCoreRecord<BLOCK_FE_WIDTH, U16_BITS>>(),
                core_off_matrix: <Rv64BranchAdapterExecutor as AdapterTraceExecutor<F>>::WIDTH
                    * size_of::<F>(),
                layout: ArenaNativeLayout::Branch2(Branch2ArenaFieldOffsets {
                    from_pc: core::mem::offset_of!(Rv64BranchAdapterRecord, from_pc),
                    from_timestamp: core::mem::offset_of!(Rv64BranchAdapterRecord, from_timestamp),
                    rs1_ptr: core::mem::offset_of!(Rv64BranchAdapterRecord, rs1_ptr),
                    rs2_ptr: core::mem::offset_of!(Rv64BranchAdapterRecord, rs2_ptr),
                    reads_aux0_prev_ts: core::mem::offset_of!(Rv64BranchAdapterRecord, reads_aux)
                        + core::mem::offset_of!(MemoryReadAuxRecord, prev_timestamp),
                    reads_aux1_prev_ts: core::mem::offset_of!(Rv64BranchAdapterRecord, reads_aux)
                        + size_of::<MemoryReadAuxRecord>()
                        + core::mem::offset_of!(MemoryReadAuxRecord, prev_timestamp),
                    core_a: core::mem::offset_of!(BranchLessThanCoreRecord<BLOCK_FE_WIDTH, U16_BITS>, a),
                    core_b: core::mem::offset_of!(BranchLessThanCoreRecord<BLOCK_FE_WIDTH, U16_BITS>, b),
                    core_imm: core::mem::offset_of!(BranchLessThanCoreRecord<BLOCK_FE_WIDTH, U16_BITS>, imm),
                    core_local_opcode: core::mem::offset_of!(BranchLessThanCoreRecord<BLOCK_FE_WIDTH, U16_BITS>, local_opcode),
                }),
            },
        );
        registry.register_inline_arena_native(
            Rv64JalLuiOpcode::iter().map(|opcode| opcode.global_opcode()),
            PREFLIGHT_WR1_RECORD_SIZE,
            assemble_jal_lui_inline::<F, RA>,
            ArenaNativeGeometry {
                adapter_size: size_of::<Rv64RdWriteAdapterRecord>(),
                adapter_align: align_of::<Rv64RdWriteAdapterRecord>(),
                core_size: size_of::<Rv64JalLuiCoreRecord>(),
                core_align: align_of::<Rv64JalLuiCoreRecord>(),
                core_off_matrix: <Rv64CondRdWriteAdapterExecutor as AdapterTraceExecutor<F>>::WIDTH
                    * size_of::<F>(),
                layout: ArenaNativeLayout::Wr1(Wr1ArenaFieldOffsets {
                    from_pc: core::mem::offset_of!(Rv64RdWriteAdapterRecord, from_pc),
                    from_timestamp: core::mem::offset_of!(Rv64RdWriteAdapterRecord, from_timestamp),
                    rd_ptr: core::mem::offset_of!(Rv64RdWriteAdapterRecord, rd_ptr),
                    rd_prev_ts: core::mem::offset_of!(Rv64RdWriteAdapterRecord, rd_aux_record)
                        + core::mem::offset_of!(
                            MemoryWriteAuxRecord<u16, BLOCK_FE_WIDTH>,
                            prev_timestamp
                        ),
                    rd_prev_data: core::mem::offset_of!(Rv64RdWriteAdapterRecord, rd_aux_record)
                        + core::mem::offset_of!(
                            MemoryWriteAuxRecord<u16, BLOCK_FE_WIDTH>,
                            prev_data
                        ),
                    core_imm: core::mem::offset_of!(Rv64JalLuiCoreRecord, imm),
                    core_rd_data: core::mem::offset_of!(Rv64JalLuiCoreRecord, rd_data),
                    core_is_jal: core::mem::offset_of!(Rv64JalLuiCoreRecord, is_jal),
                    core_from_pc: usize::MAX,
                }),
            },
        );
        registry.register_inline_arena_native(
            Rv64AuipcOpcode::iter().map(|opcode| opcode.global_opcode()),
            PREFLIGHT_WR1_RECORD_SIZE,
            assemble_auipc_inline::<F, RA>,
            ArenaNativeGeometry {
                adapter_size: size_of::<Rv64RdWriteAdapterRecord>(),
                adapter_align: align_of::<Rv64RdWriteAdapterRecord>(),
                core_size: size_of::<Rv64AuipcCoreRecord>(),
                core_align: align_of::<Rv64AuipcCoreRecord>(),
                core_off_matrix: <Rv64RdWriteAdapterExecutor as AdapterTraceExecutor<F>>::WIDTH
                    * size_of::<F>(),
                layout: ArenaNativeLayout::Wr1(Wr1ArenaFieldOffsets {
                    from_pc: core::mem::offset_of!(Rv64RdWriteAdapterRecord, from_pc),
                    from_timestamp: core::mem::offset_of!(Rv64RdWriteAdapterRecord, from_timestamp),
                    rd_ptr: core::mem::offset_of!(Rv64RdWriteAdapterRecord, rd_ptr),
                    rd_prev_ts: core::mem::offset_of!(Rv64RdWriteAdapterRecord, rd_aux_record)
                        + core::mem::offset_of!(
                            MemoryWriteAuxRecord<u16, BLOCK_FE_WIDTH>,
                            prev_timestamp
                        ),
                    rd_prev_data: core::mem::offset_of!(Rv64RdWriteAdapterRecord, rd_aux_record)
                        + core::mem::offset_of!(
                            MemoryWriteAuxRecord<u16, BLOCK_FE_WIDTH>,
                            prev_data
                        ),
                    core_imm: core::mem::offset_of!(Rv64AuipcCoreRecord, imm),
                    core_rd_data: usize::MAX,
                    core_is_jal: usize::MAX,
                    core_from_pc: core::mem::offset_of!(Rv64AuipcCoreRecord, from_pc),
                }),
            },
        );
        registry.register_inline_arena_native(
            Rv64JalrOpcode::iter().map(|opcode| opcode.global_opcode()),
            PREFLIGHT_RW1_RECORD_SIZE,
            assemble_jalr_inline::<F, RA>,
            ArenaNativeGeometry {
                adapter_size: size_of::<Rv64JalrAdapterRecord>(),
                adapter_align: align_of::<Rv64JalrAdapterRecord>(),
                core_size: size_of::<Rv64JalrCoreRecord>(),
                core_align: align_of::<Rv64JalrCoreRecord>(),
                core_off_matrix: <Rv64JalrAdapterExecutor as AdapterTraceExecutor<F>>::WIDTH
                    * size_of::<F>(),
                layout: ArenaNativeLayout::Rw1(Rw1ArenaFieldOffsets {
                    from_pc: core::mem::offset_of!(Rv64JalrAdapterRecord, from_pc),
                    from_timestamp: core::mem::offset_of!(Rv64JalrAdapterRecord, from_timestamp),
                    rs1_ptr: core::mem::offset_of!(Rv64JalrAdapterRecord, rs1_ptr),
                    rd_ptr: core::mem::offset_of!(Rv64JalrAdapterRecord, rd_ptr),
                    read_prev_ts: core::mem::offset_of!(Rv64JalrAdapterRecord, reads_aux)
                        + core::mem::offset_of!(MemoryReadAuxRecord, prev_timestamp),
                    write_prev_ts: core::mem::offset_of!(Rv64JalrAdapterRecord, writes_aux)
                        + core::mem::offset_of!(
                            MemoryWriteAuxRecord<u16, BLOCK_FE_WIDTH>,
                            prev_timestamp
                        ),
                    write_prev_data: core::mem::offset_of!(Rv64JalrAdapterRecord, writes_aux)
                        + core::mem::offset_of!(
                            MemoryWriteAuxRecord<u16, BLOCK_FE_WIDTH>,
                            prev_data
                        ),
                    core_imm: core::mem::offset_of!(Rv64JalrCoreRecord, imm),
                    core_from_pc: core::mem::offset_of!(Rv64JalrCoreRecord, from_pc),
                    core_rs1_val: core::mem::offset_of!(Rv64JalrCoreRecord, rs1_val),
                    core_imm_sign: core::mem::offset_of!(Rv64JalrCoreRecord, imm_sign),
                }),
            },
        );
        registry.register_inline_arena_native(
            Rv64LoadStoreOpcode::iter()
                .take(Rv64LoadStoreOpcode::STOREB as usize + 1)
                .map(|opcode| opcode.global_opcode()),
            PREFLIGHT_ADDSUB_RECORD_SIZE,
            assemble_loadstore_inline::<F, RA>,
            ArenaNativeGeometry {
                adapter_size: size_of::<Rv64LoadStoreAdapterRecord>(),
                adapter_align: align_of::<Rv64LoadStoreAdapterRecord>(),
                core_size: size_of::<LoadStoreCoreRecord<RV64_REGISTER_NUM_LIMBS>>(),
                core_align: align_of::<LoadStoreCoreRecord<RV64_REGISTER_NUM_LIMBS>>(),
                core_off_matrix: <Rv64LoadStoreAdapterExecutor as AdapterTraceExecutor<F>>::WIDTH
                    * size_of::<F>(),
                layout: ArenaNativeLayout::LoadStore(LoadStoreArenaFieldOffsets {
                    from_pc: core::mem::offset_of!(Rv64LoadStoreAdapterRecord, from_pc),
                    from_timestamp: core::mem::offset_of!(
                        Rv64LoadStoreAdapterRecord,
                        from_timestamp
                    ),
                    rs1_ptr: core::mem::offset_of!(Rv64LoadStoreAdapterRecord, rs1_ptr),
                    rs1_val: core::mem::offset_of!(Rv64LoadStoreAdapterRecord, rs1_val),
                    rs1_aux_prev_ts: core::mem::offset_of!(
                        Rv64LoadStoreAdapterRecord,
                        rs1_aux_record
                    ) + core::mem::offset_of!(MemoryReadAuxRecord, prev_timestamp),
                    rd_rs2_ptr: core::mem::offset_of!(Rv64LoadStoreAdapterRecord, rd_rs2_ptr),
                    read_data_aux_prev_ts: core::mem::offset_of!(
                        Rv64LoadStoreAdapterRecord,
                        read_data_aux
                    ) + core::mem::offset_of!(
                        MemoryReadAuxRecord,
                        prev_timestamp
                    ),
                    imm: core::mem::offset_of!(Rv64LoadStoreAdapterRecord, imm),
                    imm_sign: core::mem::offset_of!(Rv64LoadStoreAdapterRecord, imm_sign),
                    mem_as: core::mem::offset_of!(Rv64LoadStoreAdapterRecord, mem_as),
                    write_prev_ts: core::mem::offset_of!(
                        Rv64LoadStoreAdapterRecord,
                        write_prev_timestamp
                    ),
                    core_local_opcode: core::mem::offset_of!(
                        LoadStoreCoreRecord<RV64_REGISTER_NUM_LIMBS>,
                        local_opcode
                    ),
                    core_is_byte: usize::MAX,
                    core_is_word: usize::MAX,
                    core_shift_amount: core::mem::offset_of!(
                        LoadStoreCoreRecord<RV64_REGISTER_NUM_LIMBS>,
                        shift_amount
                    ),
                    core_read_data: core::mem::offset_of!(
                        LoadStoreCoreRecord<RV64_REGISTER_NUM_LIMBS>,
                        read_data
                    ),
                    core_prev_data: core::mem::offset_of!(
                        LoadStoreCoreRecord<RV64_REGISTER_NUM_LIMBS>,
                        prev_data
                    ),
                }),
            },
        );
        registry.register_inline_arena_native(
            Rv64LoadStoreOpcode::iter()
                .skip(Rv64LoadStoreOpcode::STOREB as usize + 1)
                .map(|opcode| opcode.global_opcode()),
            PREFLIGHT_ADDSUB_RECORD_SIZE,
            assemble_load_sign_extend_inline::<F, RA>,
            ArenaNativeGeometry {
                adapter_size: size_of::<Rv64LoadStoreAdapterRecord>(),
                adapter_align: align_of::<Rv64LoadStoreAdapterRecord>(),
                core_size: size_of::<LoadSignExtendCoreRecord<RV64_REGISTER_NUM_LIMBS>>(),
                core_align: align_of::<LoadSignExtendCoreRecord<RV64_REGISTER_NUM_LIMBS>>(),
                core_off_matrix: <Rv64LoadStoreAdapterExecutor as AdapterTraceExecutor<F>>::WIDTH
                    * size_of::<F>(),
                layout: ArenaNativeLayout::LoadStore(LoadStoreArenaFieldOffsets {
                    from_pc: core::mem::offset_of!(Rv64LoadStoreAdapterRecord, from_pc),
                    from_timestamp: core::mem::offset_of!(
                        Rv64LoadStoreAdapterRecord,
                        from_timestamp
                    ),
                    rs1_ptr: core::mem::offset_of!(Rv64LoadStoreAdapterRecord, rs1_ptr),
                    rs1_val: core::mem::offset_of!(Rv64LoadStoreAdapterRecord, rs1_val),
                    rs1_aux_prev_ts: core::mem::offset_of!(
                        Rv64LoadStoreAdapterRecord,
                        rs1_aux_record
                    ) + core::mem::offset_of!(MemoryReadAuxRecord, prev_timestamp),
                    rd_rs2_ptr: core::mem::offset_of!(Rv64LoadStoreAdapterRecord, rd_rs2_ptr),
                    read_data_aux_prev_ts: core::mem::offset_of!(
                        Rv64LoadStoreAdapterRecord,
                        read_data_aux
                    ) + core::mem::offset_of!(
                        MemoryReadAuxRecord,
                        prev_timestamp
                    ),
                    imm: core::mem::offset_of!(Rv64LoadStoreAdapterRecord, imm),
                    imm_sign: core::mem::offset_of!(Rv64LoadStoreAdapterRecord, imm_sign),
                    mem_as: core::mem::offset_of!(Rv64LoadStoreAdapterRecord, mem_as),
                    write_prev_ts: core::mem::offset_of!(
                        Rv64LoadStoreAdapterRecord,
                        write_prev_timestamp
                    ),
                    core_local_opcode: usize::MAX,
                    core_is_byte: core::mem::offset_of!(
                        LoadSignExtendCoreRecord<RV64_REGISTER_NUM_LIMBS>,
                        is_byte
                    ),
                    core_is_word: core::mem::offset_of!(
                        LoadSignExtendCoreRecord<RV64_REGISTER_NUM_LIMBS>,
                        is_word
                    ),
                    core_shift_amount: core::mem::offset_of!(
                        LoadSignExtendCoreRecord<RV64_REGISTER_NUM_LIMBS>,
                        shift_amount
                    ),
                    core_read_data: core::mem::offset_of!(
                        LoadSignExtendCoreRecord<RV64_REGISTER_NUM_LIMBS>,
                        read_data
                    ),
                    core_prev_data: core::mem::offset_of!(
                        LoadSignExtendCoreRecord<RV64_REGISTER_NUM_LIMBS>,
                        prev_data
                    ),
                }),
            },
        );
        registry.register_delta_pattern(
            [
                BaseAluOpcode::ADD,
                BaseAluOpcode::SUB,
                BaseAluOpcode::XOR,
                BaseAluOpcode::OR,
                BaseAluOpcode::AND,
            ]
            .map(|opcode| opcode.global_opcode())
            .into_iter()
            .chain(LessThanOpcode::iter().map(|opcode| opcode.global_opcode()))
            .chain(ShiftOpcode::iter().map(|opcode| opcode.global_opcode())),
            DeltaAccessPattern::Alu3,
        );
        registry.register_delta_pattern(
            BranchEqualOpcode::iter()
                .map(|opcode| opcode.global_opcode())
                .chain(BranchLessThanOpcode::iter().map(|opcode| opcode.global_opcode())),
            DeltaAccessPattern::Branch2,
        );
        registry.register_delta_pattern(
            Rv64JalLuiOpcode::iter().map(|opcode| opcode.global_opcode()),
            DeltaAccessPattern::Wr1,
        );
        registry.register_delta_pattern(
            Rv64AuipcOpcode::iter().map(|opcode| opcode.global_opcode()),
            DeltaAccessPattern::Wr1Always,
        );
        registry.register_delta_pattern(
            Rv64JalrOpcode::iter().map(|opcode| opcode.global_opcode()),
            DeltaAccessPattern::Rw1,
        );
        registry.register_delta_pattern(
            Rv64LoadStoreOpcode::iter()
                .take(Rv64LoadStoreOpcode::STORED as usize)
                .chain(Rv64LoadStoreOpcode::iter().skip(Rv64LoadStoreOpcode::STOREB as usize + 1))
                .map(|opcode| opcode.global_opcode()),
            DeltaAccessPattern::Load,
        );
        registry.register_delta_pattern(
            [
                Rv64LoadStoreOpcode::STORED,
                Rv64LoadStoreOpcode::STOREW,
                Rv64LoadStoreOpcode::STOREH,
                Rv64LoadStoreOpcode::STOREB,
            ]
            .map(|opcode| opcode.global_opcode()),
            DeltaAccessPattern::Store,
        );
        registry.register(
            [BaseAluOpcode::XOR, BaseAluOpcode::OR, BaseAluOpcode::AND]
                .map(|opcode| opcode.global_opcode()),
            assemble_bitwise::<F, RA>,
        );
        registry.register(
            BaseAluWOpcode::iter().map(|opcode| opcode.global_opcode()),
            assemble_add_sub_w::<F, RA>,
        );
        registry.register_inline_arena_native(
            BaseAluWOpcode::iter().map(|opcode| opcode.global_opcode()),
            PREFLIGHT_ADDSUB_RECORD_SIZE,
            assemble_add_sub_w_inline::<F, RA>,
            base_alu_w_arena_geometry::<F>(
                size_of::<AddSubCoreRecord<RV64_WORD_U16_LIMBS>>(),
                align_of::<AddSubCoreRecord<RV64_WORD_U16_LIMBS>>(),
                core::mem::offset_of!(AddSubCoreRecord<RV64_WORD_U16_LIMBS>, b),
                core::mem::offset_of!(AddSubCoreRecord<RV64_WORD_U16_LIMBS>, c),
                core::mem::offset_of!(AddSubCoreRecord<RV64_WORD_U16_LIMBS>, local_opcode),
            ),
        );
        registry.register_delta_pattern(
            BaseAluWOpcode::iter().map(|opcode| opcode.global_opcode()),
            DeltaAccessPattern::Alu3,
        );
        registry.register(
            LessThanOpcode::iter().map(|opcode| opcode.global_opcode()),
            assemble_less_than::<F, RA>,
        );
        registry.register(
            ShiftOpcode::iter().map(|opcode| opcode.global_opcode()),
            assemble_shift::<F, RA>,
        );
        registry.register(
            ShiftWOpcode::iter().map(|opcode| opcode.global_opcode()),
            assemble_shift_w::<F, RA>,
        );
        registry.register_inline_arena_native(
            [ShiftWOpcode::SLLW, ShiftWOpcode::SRLW].map(|opcode| opcode.global_opcode()),
            PREFLIGHT_ADDSUB_RECORD_SIZE,
            assemble_shift_w_inline::<F, RA>,
            base_alu_w_arena_geometry::<F>(
                size_of::<ShiftLogicalCoreRecord<RV64_WORD_U16_LIMBS, U16_BITS>>(),
                align_of::<ShiftLogicalCoreRecord<RV64_WORD_U16_LIMBS, U16_BITS>>(),
                core::mem::offset_of!(
                    ShiftLogicalCoreRecord<RV64_WORD_U16_LIMBS, U16_BITS>,
                    b
                ),
                core::mem::offset_of!(
                    ShiftLogicalCoreRecord<RV64_WORD_U16_LIMBS, U16_BITS>,
                    c
                ),
                core::mem::offset_of!(
                    ShiftLogicalCoreRecord<RV64_WORD_U16_LIMBS, U16_BITS>,
                    local_opcode
                ),
            ),
        );
        registry.register_inline_arena_native(
            [ShiftWOpcode::SRAW].map(|opcode| opcode.global_opcode()),
            PREFLIGHT_ADDSUB_RECORD_SIZE,
            assemble_shift_w_inline::<F, RA>,
            base_alu_w_arena_geometry::<F>(
                size_of::<ShiftRightArithmeticCoreRecord<RV64_WORD_U16_LIMBS, U16_BITS>>(),
                align_of::<ShiftRightArithmeticCoreRecord<RV64_WORD_U16_LIMBS, U16_BITS>>(),
                core::mem::offset_of!(
                    ShiftRightArithmeticCoreRecord<RV64_WORD_U16_LIMBS, U16_BITS>,
                    b
                ),
                core::mem::offset_of!(
                    ShiftRightArithmeticCoreRecord<RV64_WORD_U16_LIMBS, U16_BITS>,
                    c
                ),
                usize::MAX,
            ),
        );
        registry.register_delta_pattern(
            ShiftWOpcode::iter().map(|opcode| opcode.global_opcode()),
            DeltaAccessPattern::Alu3,
        );
        registry.register(
            BranchEqualOpcode::iter().map(|opcode| opcode.global_opcode()),
            assemble_branch_eq::<F, RA>,
        );
        registry.register(
            BranchLessThanOpcode::iter().map(|opcode| opcode.global_opcode()),
            assemble_branch_lt::<F, RA>,
        );
        registry.register(
            Rv64JalLuiOpcode::iter().map(|opcode| opcode.global_opcode()),
            assemble_jal_lui::<F, RA>,
        );
        registry.register(
            Rv64JalrOpcode::iter().map(|opcode| opcode.global_opcode()),
            assemble_jalr::<F, RA>,
        );
        registry.register(
            Rv64AuipcOpcode::iter().map(|opcode| opcode.global_opcode()),
            assemble_auipc::<F, RA>,
        );
        // Zero-extension loads (LOADD..=LOADWU) read main memory only.
        registry.register_if(
            Rv64LoadStoreOpcode::iter()
                .take(Rv64LoadStoreOpcode::STORED as usize)
                .map(|opcode| opcode.global_opcode()),
            is_rv64_memory_instruction,
            assemble_loadstore::<F, RA>,
        );
        // Stores (STORED..=STOREB) additionally write the public-values
        // address space (REVEAL). One stored predicate gates both routing and
        // assembly, mirroring the interpreter LoadStoreExecutor domain.
        registry.register_if(
            [
                Rv64LoadStoreOpcode::STORED,
                Rv64LoadStoreOpcode::STOREW,
                Rv64LoadStoreOpcode::STOREH,
                Rv64LoadStoreOpcode::STOREB,
            ]
            .map(|opcode| opcode.global_opcode()),
            is_rv64_store_instruction,
            assemble_loadstore::<F, RA>,
        );
        registry.register_if(
            Rv64LoadStoreOpcode::iter()
                .skip(Rv64LoadStoreOpcode::STOREB as usize + 1)
                .map(|opcode| opcode.global_opcode()),
            is_rv64_memory_instruction,
            assemble_load_sign_extend::<F, RA>,
        );
        registry.register(
            [SystemOpcode::PHANTOM.global_opcode()],
            assemble_phantom::<F, RA>,
        );
        registry.register_inline_arena_native(
            [SystemOpcode::PHANTOM.global_opcode()],
            size_of::<PhantomRecord>(),
            assemble_phantom_inline::<F, RA>,
            ArenaNativeGeometry {
                adapter_size: size_of::<PhantomRecord>(),
                adapter_align: align_of::<PhantomRecord>(),
                core_size: 0,
                core_align: 1,
                core_off_matrix: 0,
                layout: ArenaNativeLayout::Custom {
                    residual_memory_chronology: true,
                },
            },
        );
    }
}

impl<F, RA> VmRvrLogNativeExtension<F, RA> for crate::Rv64M
where
    F: PrimeField32,
    RA: Rv64MRecordArena<F>,
{
    fn extend_rvr_log_native(&self, registry: &mut LogNativeAssemblerRegistry<F, RA>) {
        registry.register(
            MulOpcode::iter().map(|opcode| opcode.global_opcode()),
            assemble_mul::<F, RA>,
        );
        registry.register_inline_arena_native(
            MulOpcode::iter().map(|opcode| opcode.global_opcode()),
            PREFLIGHT_ADDSUB_RECORD_SIZE,
            assemble_mul_inline::<F, RA>,
            ArenaNativeGeometry {
                adapter_size: size_of::<Rv64MultAdapterRecord>(),
                adapter_align: align_of::<Rv64MultAdapterRecord>(),
                core_size: size_of::<MultiplicationCoreRecord<RV64_REGISTER_NUM_LIMBS, RV64_BYTE_BITS>>(),
                core_align: align_of::<MultiplicationCoreRecord<RV64_REGISTER_NUM_LIMBS, RV64_BYTE_BITS>>(),
                core_off_matrix: <Rv64MultAdapterExecutor as AdapterTraceExecutor<F>>::WIDTH
                    * size_of::<F>(),
                layout: ArenaNativeLayout::Alu3(Alu3ArenaFieldOffsets {
                    from_pc: core::mem::offset_of!(Rv64MultAdapterRecord, from_pc),
                    from_timestamp: core::mem::offset_of!(Rv64MultAdapterRecord, from_timestamp),
                    rd_ptr: core::mem::offset_of!(Rv64MultAdapterRecord, rd_ptr),
                    rs1_ptr: core::mem::offset_of!(Rv64MultAdapterRecord, rs1_ptr),
                    rs2: core::mem::offset_of!(Rv64MultAdapterRecord, rs2_ptr),
                    // The Mult adapter has neither rs2_as nor imm-sign.
                    rs2_as: usize::MAX,
                    rs2_imm_sign: usize::MAX,
                    reads_aux0_prev_ts: core::mem::offset_of!(Rv64MultAdapterRecord, reads_aux)
                        + core::mem::offset_of!(MemoryReadAuxRecord, prev_timestamp),
                    reads_aux1_prev_ts: core::mem::offset_of!(Rv64MultAdapterRecord, reads_aux)
                        + size_of::<MemoryReadAuxRecord>()
                        + core::mem::offset_of!(MemoryReadAuxRecord, prev_timestamp),
                    write_prev_ts: core::mem::offset_of!(Rv64MultAdapterRecord, writes_aux)
                        + core::mem::offset_of!(
                            MemoryWriteAuxRecord<u8, RV64_REGISTER_NUM_LIMBS>,
                            prev_timestamp
                        ),
                    write_prev_data: core::mem::offset_of!(Rv64MultAdapterRecord, writes_aux)
                        + core::mem::offset_of!(
                            MemoryWriteAuxRecord<u8, RV64_REGISTER_NUM_LIMBS>,
                            prev_data
                        ),
                    core_b: core::mem::offset_of!(MultiplicationCoreRecord<RV64_REGISTER_NUM_LIMBS, RV64_BYTE_BITS>, b),
                    core_c: core::mem::offset_of!(MultiplicationCoreRecord<RV64_REGISTER_NUM_LIMBS, RV64_BYTE_BITS>, c),
                    core_local_opcode: usize::MAX,
                    w: None,
                }),
            },
        );
        registry.register_inline_arena_native(
            MulHOpcode::iter().map(|opcode| opcode.global_opcode()),
            PREFLIGHT_ADDSUB_RECORD_SIZE,
            assemble_mulh_inline::<F, RA>,
            ArenaNativeGeometry {
                adapter_size: size_of::<Rv64MultAdapterRecord>(),
                adapter_align: align_of::<Rv64MultAdapterRecord>(),
                core_size: size_of::<MulHCoreRecord<RV64_REGISTER_NUM_LIMBS, RV64_BYTE_BITS>>(),
                core_align: align_of::<MulHCoreRecord<RV64_REGISTER_NUM_LIMBS, RV64_BYTE_BITS>>(),
                core_off_matrix: <Rv64MultAdapterExecutor as AdapterTraceExecutor<F>>::WIDTH
                    * size_of::<F>(),
                layout: ArenaNativeLayout::Alu3(Alu3ArenaFieldOffsets {
                    from_pc: core::mem::offset_of!(Rv64MultAdapterRecord, from_pc),
                    from_timestamp: core::mem::offset_of!(Rv64MultAdapterRecord, from_timestamp),
                    rd_ptr: core::mem::offset_of!(Rv64MultAdapterRecord, rd_ptr),
                    rs1_ptr: core::mem::offset_of!(Rv64MultAdapterRecord, rs1_ptr),
                    rs2: core::mem::offset_of!(Rv64MultAdapterRecord, rs2_ptr),
                    // The Mult adapter has neither rs2_as nor imm-sign.
                    rs2_as: usize::MAX,
                    rs2_imm_sign: usize::MAX,
                    reads_aux0_prev_ts: core::mem::offset_of!(Rv64MultAdapterRecord, reads_aux)
                        + core::mem::offset_of!(MemoryReadAuxRecord, prev_timestamp),
                    reads_aux1_prev_ts: core::mem::offset_of!(Rv64MultAdapterRecord, reads_aux)
                        + size_of::<MemoryReadAuxRecord>()
                        + core::mem::offset_of!(MemoryReadAuxRecord, prev_timestamp),
                    write_prev_ts: core::mem::offset_of!(Rv64MultAdapterRecord, writes_aux)
                        + core::mem::offset_of!(
                            MemoryWriteAuxRecord<u8, RV64_REGISTER_NUM_LIMBS>,
                            prev_timestamp
                        ),
                    write_prev_data: core::mem::offset_of!(Rv64MultAdapterRecord, writes_aux)
                        + core::mem::offset_of!(
                            MemoryWriteAuxRecord<u8, RV64_REGISTER_NUM_LIMBS>,
                            prev_data
                        ),
                    core_b: core::mem::offset_of!(MulHCoreRecord<RV64_REGISTER_NUM_LIMBS, RV64_BYTE_BITS>, b),
                    core_c: core::mem::offset_of!(MulHCoreRecord<RV64_REGISTER_NUM_LIMBS, RV64_BYTE_BITS>, c),
                    core_local_opcode: core::mem::offset_of!(MulHCoreRecord<RV64_REGISTER_NUM_LIMBS, RV64_BYTE_BITS>, local_opcode),
                    w: None,
                }),
            },
        );
        registry.register_inline_arena_native(
            MulWOpcode::iter().map(|opcode| opcode.global_opcode()),
            PREFLIGHT_ADDSUB_RECORD_SIZE,
            assemble_mul_w_inline::<F, RA>,
            mult_w_arena_geometry::<F>(
                size_of::<MultiplicationCoreRecord<RV64_WORD_NUM_LIMBS, RV64_BYTE_BITS>>(),
                align_of::<MultiplicationCoreRecord<RV64_WORD_NUM_LIMBS, RV64_BYTE_BITS>>(),
                core::mem::offset_of!(
                    MultiplicationCoreRecord<RV64_WORD_NUM_LIMBS, RV64_BYTE_BITS>,
                    b
                ),
                core::mem::offset_of!(
                    MultiplicationCoreRecord<RV64_WORD_NUM_LIMBS, RV64_BYTE_BITS>,
                    c
                ),
                usize::MAX,
            ),
        );
        registry.register_inline_arena_native(
            DivRemOpcode::iter().map(|opcode| opcode.global_opcode()),
            PREFLIGHT_ADDSUB_RECORD_SIZE,
            assemble_divrem_inline::<F, RA>,
            ArenaNativeGeometry {
                adapter_size: size_of::<Rv64MultAdapterRecord>(),
                adapter_align: align_of::<Rv64MultAdapterRecord>(),
                core_size: size_of::<DivRemCoreRecord<RV64_REGISTER_NUM_LIMBS>>(),
                core_align: align_of::<DivRemCoreRecord<RV64_REGISTER_NUM_LIMBS>>(),
                core_off_matrix: <Rv64MultAdapterExecutor as AdapterTraceExecutor<F>>::WIDTH
                    * size_of::<F>(),
                layout: ArenaNativeLayout::Alu3(Alu3ArenaFieldOffsets {
                    from_pc: core::mem::offset_of!(Rv64MultAdapterRecord, from_pc),
                    from_timestamp: core::mem::offset_of!(Rv64MultAdapterRecord, from_timestamp),
                    rd_ptr: core::mem::offset_of!(Rv64MultAdapterRecord, rd_ptr),
                    rs1_ptr: core::mem::offset_of!(Rv64MultAdapterRecord, rs1_ptr),
                    rs2: core::mem::offset_of!(Rv64MultAdapterRecord, rs2_ptr),
                    // The Mult adapter has neither rs2_as nor imm-sign.
                    rs2_as: usize::MAX,
                    rs2_imm_sign: usize::MAX,
                    reads_aux0_prev_ts: core::mem::offset_of!(Rv64MultAdapterRecord, reads_aux)
                        + core::mem::offset_of!(MemoryReadAuxRecord, prev_timestamp),
                    reads_aux1_prev_ts: core::mem::offset_of!(Rv64MultAdapterRecord, reads_aux)
                        + size_of::<MemoryReadAuxRecord>()
                        + core::mem::offset_of!(MemoryReadAuxRecord, prev_timestamp),
                    write_prev_ts: core::mem::offset_of!(Rv64MultAdapterRecord, writes_aux)
                        + core::mem::offset_of!(
                            MemoryWriteAuxRecord<u8, RV64_REGISTER_NUM_LIMBS>,
                            prev_timestamp
                        ),
                    write_prev_data: core::mem::offset_of!(Rv64MultAdapterRecord, writes_aux)
                        + core::mem::offset_of!(
                            MemoryWriteAuxRecord<u8, RV64_REGISTER_NUM_LIMBS>,
                            prev_data
                        ),
                    core_b: core::mem::offset_of!(DivRemCoreRecord<RV64_REGISTER_NUM_LIMBS>, b),
                    core_c: core::mem::offset_of!(DivRemCoreRecord<RV64_REGISTER_NUM_LIMBS>, c),
                    core_local_opcode: core::mem::offset_of!(
                        DivRemCoreRecord<RV64_REGISTER_NUM_LIMBS>,
                        local_opcode
                    ),
                    w: None,
                }),
            },
        );
        registry.register_inline_arena_native(
            DivRemWOpcode::iter().map(|opcode| opcode.global_opcode()),
            PREFLIGHT_ADDSUB_RECORD_SIZE,
            assemble_divrem_w_inline::<F, RA>,
            mult_w_arena_geometry::<F>(
                size_of::<DivRemCoreRecord<RV64_WORD_NUM_LIMBS>>(),
                align_of::<DivRemCoreRecord<RV64_WORD_NUM_LIMBS>>(),
                core::mem::offset_of!(DivRemCoreRecord<RV64_WORD_NUM_LIMBS>, b),
                core::mem::offset_of!(DivRemCoreRecord<RV64_WORD_NUM_LIMBS>, c),
                core::mem::offset_of!(DivRemCoreRecord<RV64_WORD_NUM_LIMBS>, local_opcode),
            ),
        );
        registry.register_delta_pattern(
            MulOpcode::iter()
                .map(|opcode| opcode.global_opcode())
                .chain(MulHOpcode::iter().map(|opcode| opcode.global_opcode()))
                .chain(MulWOpcode::iter().map(|opcode| opcode.global_opcode()))
                .chain(DivRemOpcode::iter().map(|opcode| opcode.global_opcode()))
                .chain(DivRemWOpcode::iter().map(|opcode| opcode.global_opcode())),
            DeltaAccessPattern::Alu3Reg,
        );
        registry.register(
            MulHOpcode::iter().map(|opcode| opcode.global_opcode()),
            assemble_mulh::<F, RA>,
        );
        registry.register(
            MulWOpcode::iter().map(|opcode| opcode.global_opcode()),
            assemble_mul_w::<F, RA>,
        );
        registry.register(
            DivRemOpcode::iter().map(|opcode| opcode.global_opcode()),
            assemble_divrem::<F, RA>,
        );
        registry.register(
            DivRemWOpcode::iter().map(|opcode| opcode.global_opcode()),
            assemble_divrem_w::<F, RA>,
        );
    }
}

impl<F, RA> VmRvrLogNativeExtension<F, RA> for crate::Rv64Io
where
    F: PrimeField32,
    RA: Rv64IoRecordArena<F>,
{
    fn extend_rvr_log_native(&self, registry: &mut LogNativeAssemblerRegistry<F, RA>) {
        registry.register(
            Rv64HintStoreOpcode::iter().map(|opcode| opcode.global_opcode()),
            assemble_hintstore::<F, RA>,
        );
        let descriptor = HintStoreRecordDescriptor::new();
        assert_eq!(
            size_of::<Rv64HintStoreRecordHeader>(),
            descriptor.header_size
        );
        assert_eq!(
            align_of::<Rv64HintStoreRecordHeader>(),
            descriptor.header_align
        );
        assert_eq!(size_of::<Rv64HintStoreVar>(), descriptor.var_size);
        assert_eq!(align_of::<Rv64HintStoreVar>(), descriptor.var_align);
        assert_eq!(descriptor.record_size(1), descriptor.capacity_per_row);
        registry.register_inline_arena_native(
            Rv64HintStoreOpcode::iter().map(|opcode| opcode.global_opcode()),
            descriptor.capacity_per_row,
            reject_hintstore_compact::<F, RA>,
            ArenaNativeGeometry {
                adapter_size: descriptor.header_size,
                adapter_align: descriptor.header_align,
                core_size: descriptor.var_size,
                core_align: descriptor.var_align,
                core_off_matrix: 0,
                layout: ArenaNativeLayout::CustomVariableRows {
                    residual_memory_chronology: true,
                },
            },
        );
    }
}

impl<F, RA> VmRvrLogNativeExtension<F, RA> for crate::Rv64IConfig
where
    F: PrimeField32,
    RA: Rv64IRecordArena<F> + Rv64IoRecordArena<F>,
{
    fn extend_rvr_log_native(&self, registry: &mut LogNativeAssemblerRegistry<F, RA>) {
        self.base.extend_rvr_log_native(registry);
        self.io.extend_rvr_log_native(registry);
    }
}

impl<F, RA> VmRvrLogNativeExtension<F, RA> for crate::Rv64ImConfig
where
    F: PrimeField32,
    RA: Rv64StandardRecordArena<F>,
{
    fn extend_rvr_log_native(&self, registry: &mut LogNativeAssemblerRegistry<F, RA>) {
        self.rv64i.extend_rvr_log_native(registry);
        self.mul.extend_rvr_log_native(registry);
    }
}

fn is_rv64_memory_instruction<F: PrimeField32>(instruction: &Instruction<F>) -> bool {
    instruction.e.as_canonical_u32() == RV64_MEMORY_AS
}

fn is_rv64_store_instruction<F: PrimeField32>(instruction: &Instruction<F>) -> bool {
    let mem_as = instruction.e.as_canonical_u32();
    mem_as == RV64_MEMORY_AS || mem_as == PUBLIC_VALUES_AS
}

fn assemble_add_sub<F: PrimeField32, RA: Rv64IRecordArena<F>>(
    arena: &mut RA,
    access: &AccessView<'_, F>,
    instruction: &Instruction<F>,
    pc: u32,
    timestamp: u32,
) -> Result<(), ExecutionError> {
    let local = instruction
        .opcode
        .local_opcode_idx(BaseAluOpcode::CLASS_OFFSET) as u8;
    let (adapter_record, core_record): (
        &mut Rv64BaseAluU16AdapterRecord,
        &mut AddSubCoreRecord<BLOCK_FE_WIDTH>,
    ) = arena.alloc(EmptyAdapterCoreLayout::<F, Rv64BaseAluU16AdapterExecutor>::new());
    let [rs1, rs2] = fill_base_alu_u16_adapter(access, instruction, pc, timestamp, adapter_record)?;
    core_record.b = rs1;
    core_record.c = rs2;
    core_record.local_opcode = local;
    Ok(())
}

/// The program-redundant BaseAluU16 adapter operands, derived from an
/// instruction. Single source of truth for host expansion AND the GPU
/// device operand table (M-GPUDEC): both consumers call this; the CUDA
/// decoder consumes the table entries it produces.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct BaseAluU16Operands {
    pub rd_ptr: u32,
    pub rs1_ptr: u32,
    pub rs2: u32,
    pub rs2_as: u8,
    pub rs2_imm_sign: bool,
}

pub(crate) fn derive_base_alu_u16_operands<F: PrimeField32>(
    instruction: &Instruction<F>,
) -> BaseAluU16Operands {
    let (rs2_as, rs2, rs2_imm_sign) = if instruction.e.as_canonical_u32() == RV64_REGISTER_AS {
        (
            RV64_REGISTER_AS as u8,
            instruction.c.as_canonical_u32(),
            false,
        )
    } else {
        let imm = instruction.c.as_canonical_u32();
        let imm64 = imm_to_rv64_u64(imm);
        (RV64_IMM_AS as u8, imm, ((imm64 >> U16_BITS) as u16) != 0)
    };
    BaseAluU16Operands {
        rd_ptr: instruction.a.as_canonical_u32(),
        rs1_ptr: instruction.b.as_canonical_u32(),
        rs2,
        rs2_as,
        rs2_imm_sign,
    }
}

/// Fill a BaseAluU16 adapter record from a compact alu3 record, mirroring
/// [`fill_base_alu_u16_adapter`]'s derivation from the log path (shared by
/// the LessThan and Shift inline assemblers; AddSub keeps its own inline fill
/// with identical semantics).
fn fill_base_alu_u16_from_compact<F: PrimeField32>(
    compact: &PreflightAlu3Compact,
    instruction: &Instruction<F>,
    pc: u32,
    record: &mut Rv64BaseAluU16AdapterRecord,
) {
    let operands = derive_base_alu_u16_operands(instruction);
    record.from_pc = pc;
    record.from_timestamp = compact.from_timestamp;
    record.rs1_ptr = operands.rs1_ptr;
    record.reads_aux[0].prev_timestamp = compact.reads_prev_timestamp[0];
    record.reads_aux[1].prev_timestamp = compact.reads_prev_timestamp[1];
    record.rs2_as = operands.rs2_as;
    record.rs2 = operands.rs2;
    record.rs2_imm_sign = operands.rs2_imm_sign;
    record.rd_ptr = operands.rd_ptr;
    record.writes_aux = MemoryWriteAuxRecord {
        prev_timestamp: compact.write_prev_timestamp,
        prev_data: u16x4(compact.write_prev_data),
    };
}

fn assemble_less_than_inline<F: PrimeField32, RA: Rv64IRecordArena<F>>(
    arena: &mut RA,
    instruction: &Instruction<F>,
    compact: &[u8],
    pc: u32,
) -> Result<(), ExecutionError> {
    let compact = PreflightAlu3Compact::read_for_pc(compact, pc)?;
    let (adapter_record, core_record): (
        &mut Rv64BaseAluU16AdapterRecord,
        &mut LessThanCoreRecord<BLOCK_FE_WIDTH, U16_BITS>,
    ) = arena.alloc(EmptyAdapterCoreLayout::<F, Rv64BaseAluU16AdapterExecutor>::new());
    fill_base_alu_u16_from_compact(&compact, instruction, pc, adapter_record);
    core_record.b = u16x4(compact.b);
    core_record.c = u16x4(compact.c);
    core_record.local_opcode = instruction
        .opcode
        .local_opcode_idx(LessThanOpcode::CLASS_OFFSET) as u8;
    Ok(())
}

/// Mirrors [`assemble_shift`]: SRA fills the arithmetic core (no
/// local_opcode); SLL/SRL fill the logical core.
fn assemble_shift_inline<F: PrimeField32, RA: Rv64IRecordArena<F>>(
    arena: &mut RA,
    instruction: &Instruction<F>,
    compact: &[u8],
    pc: u32,
) -> Result<(), ExecutionError> {
    let compact = PreflightAlu3Compact::read_for_pc(compact, pc)?;
    let local_opcode = ShiftOpcode::from_repr(
        instruction
            .opcode
            .local_opcode_idx(ShiftOpcode::CLASS_OFFSET),
    )
    .expect("opcode already classified");
    if local_opcode == ShiftOpcode::SRA {
        let (adapter_record, core_record): (
            &mut Rv64BaseAluU16AdapterRecord,
            &mut ShiftRightArithmeticCoreRecord<BLOCK_FE_WIDTH, U16_BITS>,
        ) = arena.alloc(EmptyAdapterCoreLayout::<F, Rv64BaseAluU16AdapterExecutor>::new());
        fill_base_alu_u16_from_compact(&compact, instruction, pc, adapter_record);
        core_record.b = u16x4(compact.b);
        core_record.c = u16x4(compact.c);
    } else {
        let (adapter_record, core_record): (
            &mut Rv64BaseAluU16AdapterRecord,
            &mut ShiftLogicalCoreRecord<BLOCK_FE_WIDTH, U16_BITS>,
        ) = arena.alloc(EmptyAdapterCoreLayout::<F, Rv64BaseAluU16AdapterExecutor>::new());
        fill_base_alu_u16_from_compact(&compact, instruction, pc, adapter_record);
        core_record.b = u16x4(compact.b);
        core_record.c = u16x4(compact.c);
        core_record.local_opcode = local_opcode as u8;
    }
    Ok(())
}

fn assemble_shift_w_inline<F: PrimeField32, RA: Rv64IRecordArena<F>>(
    arena: &mut RA,
    instruction: &Instruction<F>,
    compact: &[u8],
    pc: u32,
) -> Result<(), ExecutionError> {
    let compact = PreflightAlu3Compact::read_for_pc(compact, pc)?;
    let local_opcode = ShiftWOpcode::from_repr(
        instruction
            .opcode
            .local_opcode_idx(ShiftWOpcode::CLASS_OFFSET),
    )
    .expect("opcode already classified");
    let b = compact.b as u32;
    let shamt = compact.c as u32 & 31;
    let result = match local_opcode {
        ShiftWOpcode::SLLW => b.wrapping_shl(shamt),
        ShiftWOpcode::SRLW => b.wrapping_shr(shamt),
        ShiftWOpcode::SRAW => ((b as i32) >> shamt) as u32,
    };
    if local_opcode == ShiftWOpcode::SRAW {
        let (adapter_record, core_record): (
            &mut Rv64BaseAluWU16AdapterRecord,
            &mut ShiftRightArithmeticCoreRecord<RV64_WORD_U16_LIMBS, U16_BITS>,
        ) = arena.alloc(EmptyAdapterCoreLayout::<F, Rv64BaseAluWU16AdapterExecutor>::new());
        fill_base_alu_w_u16_from_compact(&compact, instruction, pc, result, adapter_record);
        core_record.b = u16x4(compact.b)[..RV64_WORD_U16_LIMBS].try_into().unwrap();
        core_record.c = u16x4(compact.c)[..RV64_WORD_U16_LIMBS].try_into().unwrap();
    } else {
        let (adapter_record, core_record): (
            &mut Rv64BaseAluWU16AdapterRecord,
            &mut ShiftLogicalCoreRecord<RV64_WORD_U16_LIMBS, U16_BITS>,
        ) = arena.alloc(EmptyAdapterCoreLayout::<F, Rv64BaseAluWU16AdapterExecutor>::new());
        fill_base_alu_w_u16_from_compact(&compact, instruction, pc, result, adapter_record);
        core_record.b = u16x4(compact.b)[..RV64_WORD_U16_LIMBS].try_into().unwrap();
        core_record.c = u16x4(compact.c)[..RV64_WORD_U16_LIMBS].try_into().unwrap();
        core_record.local_opcode = match local_opcode {
            ShiftWOpcode::SLLW => ShiftOpcode::SLL as u8,
            ShiftWOpcode::SRLW => ShiftOpcode::SRL as u8,
            ShiftWOpcode::SRAW => unreachable!("handled above"),
        };
    }
    Ok(())
}

/// Mirrors [`assemble_bitwise`] over the byte adapter (no `rs2_imm_sign`
/// field; byte-limb operand views).
fn assemble_bitwise_inline<F: PrimeField32, RA: Rv64IRecordArena<F>>(
    arena: &mut RA,
    instruction: &Instruction<F>,
    compact: &[u8],
    pc: u32,
) -> Result<(), ExecutionError> {
    let compact = PreflightAlu3Compact::read_for_pc(compact, pc)?;
    let (adapter_record, core_record): (
        &mut Rv64BaseAluAdapterRecord,
        &mut BitwiseLogicCoreRecord<RV64_REGISTER_NUM_LIMBS>,
    ) = arena.alloc(EmptyAdapterCoreLayout::<
        F,
        Rv64BaseAluAdapterExecutor<RV64_BYTE_BITS>,
    >::new());
    adapter_record.from_pc = pc;
    adapter_record.from_timestamp = compact.from_timestamp;
    adapter_record.rs1_ptr = instruction.b.as_canonical_u32();
    adapter_record.reads_aux[0].prev_timestamp = compact.reads_prev_timestamp[0];
    adapter_record.reads_aux[1].prev_timestamp = compact.reads_prev_timestamp[1];
    if instruction.e.as_canonical_u32() == RV64_REGISTER_AS {
        adapter_record.rs2_as = RV64_REGISTER_AS as u8;
        adapter_record.rs2 = instruction.c.as_canonical_u32();
    } else {
        adapter_record.rs2_as = RV64_IMM_AS as u8;
        adapter_record.rs2 = instruction.c.as_canonical_u32();
    }
    adapter_record.rd_ptr = instruction.a.as_canonical_u32();
    adapter_record.writes_aux = MemoryWriteAuxRecord {
        prev_timestamp: compact.write_prev_timestamp,
        prev_data: bytes8(compact.write_prev_data),
    };
    core_record.b = bytes8(compact.b);
    core_record.c = bytes8(compact.c);
    core_record.local_opcode = instruction
        .opcode
        .local_opcode_idx(BaseAluOpcode::CLASS_OFFSET) as u8;
    Ok(())
}

/// Little-endian field readers for the compact wire shapes (no alignment
/// requirements; each `read_for_pc` also order-guards against the program
/// log like [`PreflightAlu3Compact::read_for_pc`]).
fn compact_u32_at(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().expect("4-byte field"))
}

fn compact_u64_at(bytes: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(bytes[at..at + 8].try_into().expect("8-byte field"))
}

fn compact_order_guard(from_pc: u32, pc: u32) -> Result<(), ExecutionError> {
    if from_pc != pc {
        return Err(ExecutionError::RvrExecution(format!(
            "inline record order mismatch: record from_pc {from_pc:#x} vs program-log pc {pc:#x}"
        )));
    }
    Ok(())
}

/// Host mirror of the C branch2 record (2 reads, no write).
struct PreflightBranch2Compact {
    from_timestamp: u32,
    reads_prev_timestamp: [u32; 2],
    b: u64,
    c: u64,
}

impl PreflightBranch2Compact {
    fn read_for_pc(bytes: &[u8], pc: u32) -> Result<Self, ExecutionError> {
        debug_assert_eq!(bytes.len(), PREFLIGHT_BRANCH2_RECORD_SIZE);
        compact_order_guard(compact_u32_at(bytes, 0), pc)?;
        Ok(Self {
            from_timestamp: compact_u32_at(bytes, 4),
            reads_prev_timestamp: [compact_u32_at(bytes, 8), compact_u32_at(bytes, 12)],
            b: compact_u64_at(bytes, 16),
            c: compact_u64_at(bytes, 24),
        })
    }
}

/// Host mirror of the C wr1 record (one conditional write).
struct PreflightWr1Compact {
    from_timestamp: u32,
    write_prev_timestamp: u32,
    write_prev_data: u64,
}

impl PreflightWr1Compact {
    fn read_for_pc(bytes: &[u8], pc: u32) -> Result<Self, ExecutionError> {
        debug_assert_eq!(bytes.len(), PREFLIGHT_WR1_RECORD_SIZE);
        compact_order_guard(compact_u32_at(bytes, 0), pc)?;
        Ok(Self {
            from_timestamp: compact_u32_at(bytes, 4),
            write_prev_timestamp: compact_u32_at(bytes, 8),
            write_prev_data: compact_u64_at(bytes, 12),
        })
    }
}

/// Host mirror of the C rw1 record (one read + one conditional write).
struct PreflightRw1Compact {
    from_timestamp: u32,
    read_prev_timestamp: u32,
    write_prev_timestamp: u32,
    b: u64,
    write_prev_data: u64,
}

impl PreflightRw1Compact {
    fn read_for_pc(bytes: &[u8], pc: u32) -> Result<Self, ExecutionError> {
        debug_assert_eq!(bytes.len(), PREFLIGHT_RW1_RECORD_SIZE);
        compact_order_guard(compact_u32_at(bytes, 0), pc)?;
        Ok(Self {
            from_timestamp: compact_u32_at(bytes, 4),
            read_prev_timestamp: compact_u32_at(bytes, 8),
            write_prev_timestamp: compact_u32_at(bytes, 12),
            b: compact_u64_at(bytes, 16),
            write_prev_data: compact_u64_at(bytes, 24),
        })
    }
}

/// Fill a branch adapter record from a compact branch2 record, mirroring
/// [`fill_branch_adapter`].
fn fill_branch_adapter_from_compact<F: PrimeField32>(
    compact: &PreflightBranch2Compact,
    instruction: &Instruction<F>,
    pc: u32,
    record: &mut Rv64BranchAdapterRecord,
) {
    record.from_pc = pc;
    record.from_timestamp = compact.from_timestamp;
    record.rs1_ptr = instruction.a.as_canonical_u32();
    record.rs2_ptr = instruction.b.as_canonical_u32();
    record.reads_aux[0].prev_timestamp = compact.reads_prev_timestamp[0];
    record.reads_aux[1].prev_timestamp = compact.reads_prev_timestamp[1];
}

fn assemble_branch_eq_inline<F: PrimeField32, RA: Rv64IRecordArena<F>>(
    arena: &mut RA,
    instruction: &Instruction<F>,
    compact: &[u8],
    pc: u32,
) -> Result<(), ExecutionError> {
    let compact = PreflightBranch2Compact::read_for_pc(compact, pc)?;
    let (adapter_record, core_record): (
        &mut Rv64BranchAdapterRecord,
        &mut BranchEqualCoreRecord<BLOCK_FE_WIDTH>,
    ) = arena.alloc(EmptyAdapterCoreLayout::<F, Rv64BranchAdapterExecutor>::new());
    fill_branch_adapter_from_compact(&compact, instruction, pc, adapter_record);
    core_record.a = u16x4(compact.b);
    core_record.b = u16x4(compact.c);
    core_record.imm = instruction.c.as_canonical_u32();
    core_record.local_opcode = instruction
        .opcode
        .local_opcode_idx(BranchEqualOpcode::CLASS_OFFSET) as u8;
    Ok(())
}

fn assemble_branch_lt_inline<F: PrimeField32, RA: Rv64IRecordArena<F>>(
    arena: &mut RA,
    instruction: &Instruction<F>,
    compact: &[u8],
    pc: u32,
) -> Result<(), ExecutionError> {
    let compact = PreflightBranch2Compact::read_for_pc(compact, pc)?;
    let (adapter_record, core_record): (
        &mut Rv64BranchAdapterRecord,
        &mut BranchLessThanCoreRecord<BLOCK_FE_WIDTH, U16_BITS>,
    ) = arena.alloc(EmptyAdapterCoreLayout::<F, Rv64BranchAdapterExecutor>::new());
    fill_branch_adapter_from_compact(&compact, instruction, pc, adapter_record);
    core_record.a = u16x4(compact.b);
    core_record.b = u16x4(compact.c);
    core_record.imm = instruction.c.as_canonical_u32();
    core_record.local_opcode = instruction
        .opcode
        .local_opcode_idx(BranchLessThanOpcode::CLASS_OFFSET) as u8;
    Ok(())
}

/// Fill a (conditional) rd-write adapter record from a compact wr1 record,
/// mirroring [`fill_cond_rdwrite_adapter`] / [`fill_rdwrite_adapter`].
fn fill_rdwrite_adapter_from_compact<F: PrimeField32>(
    compact: &PreflightWr1Compact,
    instruction: &Instruction<F>,
    pc: u32,
    write_enabled: bool,
    record: &mut Rv64RdWriteAdapterRecord,
) {
    record.from_pc = pc;
    record.from_timestamp = compact.from_timestamp;
    if write_enabled {
        record.rd_ptr = instruction.a.as_canonical_u32();
        record.rd_aux_record = MemoryWriteAuxRecord {
            prev_timestamp: compact.write_prev_timestamp,
            prev_data: u16x4(compact.write_prev_data),
        };
    } else {
        record.rd_ptr = u32::MAX;
    }
}

fn assemble_jal_lui_inline<F: PrimeField32, RA: Rv64IRecordArena<F>>(
    arena: &mut RA,
    instruction: &Instruction<F>,
    compact: &[u8],
    pc: u32,
) -> Result<(), ExecutionError> {
    let compact = PreflightWr1Compact::read_for_pc(compact, pc)?;
    let (adapter_record, core_record): (&mut Rv64RdWriteAdapterRecord, &mut Rv64JalLuiCoreRecord) =
        arena.alloc(EmptyAdapterCoreLayout::<F, Rv64CondRdWriteAdapterExecutor>::new());
    fill_rdwrite_adapter_from_compact(
        &compact,
        instruction,
        pc,
        instruction.f.is_one(),
        adapter_record,
    );
    let local = instruction
        .opcode
        .local_opcode_idx(Rv64JalLuiOpcode::CLASS_OFFSET);
    let is_jal = local == JAL;
    let imm = instruction.c.as_canonical_u32();
    let signed_imm = if is_jal {
        if imm < (1 << 20) {
            imm as i32
        } else {
            let neg_imm = F::ORDER_U32 - imm;
            -(neg_imm as i32)
        }
    } else {
        imm as i32
    };
    core_record.imm = imm;
    core_record.rd_data = run_jal_lui_value(is_jal, pc, signed_imm);
    core_record.is_jal = is_jal;
    Ok(())
}

fn assemble_auipc_inline<F: PrimeField32, RA: Rv64IRecordArena<F>>(
    arena: &mut RA,
    instruction: &Instruction<F>,
    compact: &[u8],
    pc: u32,
) -> Result<(), ExecutionError> {
    let compact = PreflightWr1Compact::read_for_pc(compact, pc)?;
    let (adapter_record, core_record): (&mut Rv64RdWriteAdapterRecord, &mut Rv64AuipcCoreRecord) =
        arena.alloc(EmptyAdapterCoreLayout::<F, Rv64RdWriteAdapterExecutor>::new());
    fill_rdwrite_adapter_from_compact(&compact, instruction, pc, true, adapter_record);
    core_record.from_pc = pc;
    core_record.imm = instruction.c.as_canonical_u32();
    Ok(())
}

fn assemble_jalr_inline<F: PrimeField32, RA: Rv64IRecordArena<F>>(
    arena: &mut RA,
    instruction: &Instruction<F>,
    compact: &[u8],
    pc: u32,
) -> Result<(), ExecutionError> {
    let compact = PreflightRw1Compact::read_for_pc(compact, pc)?;
    let (adapter_record, core_record): (&mut Rv64JalrAdapterRecord, &mut Rv64JalrCoreRecord) =
        arena.alloc(EmptyAdapterCoreLayout::<F, Rv64JalrAdapterExecutor>::new());
    adapter_record.from_pc = pc;
    adapter_record.from_timestamp = compact.from_timestamp;
    adapter_record.rs1_ptr = instruction.b.as_canonical_u32();
    adapter_record.reads_aux = MemoryReadAuxRecord {
        prev_timestamp: compact.read_prev_timestamp,
    };
    if instruction.f.is_one() {
        adapter_record.rd_ptr = instruction.a.as_canonical_u32();
        adapter_record.writes_aux = MemoryWriteAuxRecord {
            prev_timestamp: compact.write_prev_timestamp,
            prev_data: u16x4(compact.write_prev_data),
        };
    } else {
        adapter_record.rd_ptr = u32::MAX;
    }
    core_record.imm = instruction.c.as_canonical_u32() as u16;
    core_record.from_pc = pc;
    core_record.rs1_val = compact.b as u32;
    core_record.imm_sign = instruction.g.is_one();
    Ok(())
}

/// Shared derivation for the load/store inline assemblers, mirroring
/// [`fill_loadstore_start`]: rs1_val comes from the compact record; the
/// pointer, alignment shift, and immediate fields come from the instruction.
struct LoadStoreCompactStart {
    shift_amount: u8,
}

fn fill_loadstore_adapter_from_compact<F: PrimeField32>(
    compact: &PreflightAlu3Compact,
    instruction: &Instruction<F>,
    pc: u32,
    record: &mut Rv64LoadStoreAdapterRecord,
) -> LoadStoreCompactStart {
    record.from_pc = pc;
    record.from_timestamp = compact.from_timestamp;
    record.rs1_ptr = instruction.b.as_canonical_u32();
    record.rs1_aux_record.prev_timestamp = compact.reads_prev_timestamp[0];
    let rs1_val = compact.b as u32;
    record.rs1_val = rs1_val;
    record.imm = instruction.c.as_canonical_u32() as u16;
    record.imm_sign = instruction.g.is_one();
    let ptr = rs1_val.wrapping_add(sign_extend_imm16(record.imm as u32, record.imm_sign));
    let shift_amount = (ptr & (RV64_REGISTER_NUM_LIMBS as u32 - 1)) as u8;
    record.read_data_aux.prev_timestamp = compact.reads_prev_timestamp[1];
    LoadStoreCompactStart { shift_amount }
}

/// Inline assembler for the zero-extension loads and the main-memory stores
/// (the pcs the compile migrates: main-memory `Instr::Load`/`Instr::Store`
/// plus extension-lifted public-values REVEAL). Mirrors
/// [`assemble_loadstore`].
fn assemble_loadstore_inline<F: PrimeField32, RA: Rv64IRecordArena<F>>(
    arena: &mut RA,
    instruction: &Instruction<F>,
    compact: &[u8],
    pc: u32,
) -> Result<(), ExecutionError> {
    let compact = PreflightAlu3Compact::read_for_pc(compact, pc)?;
    let local_opcode = Rv64LoadStoreOpcode::from_repr(
        instruction
            .opcode
            .local_opcode_idx(Rv64LoadStoreOpcode::CLASS_OFFSET),
    )
    .expect("assembler is registered only for RV64 load/store opcodes");
    let mem_as = instruction.e.as_canonical_u32();
    if mem_as != RV64_MEMORY_AS && mem_as != PUBLIC_VALUES_AS {
        return Err(ExecutionError::RvrExecution(format!(
            "inline load/store at pc {pc:#x} must target main memory or public values, got AS \
             {mem_as}"
        )));
    }
    let (adapter_record, core_record): (
        &mut Rv64LoadStoreAdapterRecord,
        &mut LoadStoreCoreRecord<RV64_REGISTER_NUM_LIMBS>,
    ) = arena.alloc(EmptyAdapterCoreLayout::<F, Rv64LoadStoreAdapterExecutor>::new());
    let start = fill_loadstore_adapter_from_compact(&compact, instruction, pc, adapter_record);
    adapter_record.mem_as = mem_as as u8;
    let enabled = instruction.f.is_one();
    let is_load = matches!(
        local_opcode,
        Rv64LoadStoreOpcode::LOADD
            | Rv64LoadStoreOpcode::LOADWU
            | Rv64LoadStoreOpcode::LOADHU
            | Rv64LoadStoreOpcode::LOADBU
    );
    let (read_data, prev_data) = if is_load {
        let prev_data = if enabled {
            adapter_record.rd_rs2_ptr = instruction.a.as_canonical_u32();
            adapter_record.write_prev_timestamp = compact.write_prev_timestamp;
            bytes8(compact.write_prev_data)
        } else {
            adapter_record.rd_rs2_ptr = u32::MAX;
            [0; RV64_REGISTER_NUM_LIMBS]
        };
        (bytes8(compact.c), prev_data)
    } else {
        adapter_record.rd_rs2_ptr = instruction.a.as_canonical_u32();
        adapter_record.write_prev_timestamp = compact.write_prev_timestamp;
        (bytes8(compact.c), bytes8(compact.write_prev_data))
    };
    core_record.local_opcode = local_opcode as u8;
    core_record.shift_amount = start.shift_amount;
    core_record.read_data = read_data;
    core_record.prev_data = prev_data;
    Ok(())
}

/// Inline assembler for the sign-extending loads (LOADB/LOADH/LOADW),
/// mirroring [`assemble_load_sign_extend`].
fn assemble_load_sign_extend_inline<F: PrimeField32, RA: Rv64IRecordArena<F>>(
    arena: &mut RA,
    instruction: &Instruction<F>,
    compact: &[u8],
    pc: u32,
) -> Result<(), ExecutionError> {
    let compact = PreflightAlu3Compact::read_for_pc(compact, pc)?;
    let local_opcode = Rv64LoadStoreOpcode::from_repr(
        instruction
            .opcode
            .local_opcode_idx(Rv64LoadStoreOpcode::CLASS_OFFSET),
    )
    .expect("assembler is registered only for RV64 load/store opcodes");
    let (adapter_record, core_record): (
        &mut Rv64LoadStoreAdapterRecord,
        &mut LoadSignExtendCoreRecord<RV64_REGISTER_NUM_LIMBS>,
    ) = arena.alloc(EmptyAdapterCoreLayout::<F, Rv64LoadStoreAdapterExecutor>::new());
    let start = fill_loadstore_adapter_from_compact(&compact, instruction, pc, adapter_record);
    adapter_record.mem_as = RV64_MEMORY_AS as u8;
    let prev_data = if instruction.f.is_one() {
        adapter_record.rd_rs2_ptr = instruction.a.as_canonical_u32();
        adapter_record.write_prev_timestamp = compact.write_prev_timestamp;
        bytes8(compact.write_prev_data)
    } else {
        adapter_record.rd_rs2_ptr = u32::MAX;
        [0; RV64_REGISTER_NUM_LIMBS]
    };
    core_record.is_byte = local_opcode == Rv64LoadStoreOpcode::LOADB;
    core_record.is_word = local_opcode == Rv64LoadStoreOpcode::LOADW;
    core_record.shift_amount = start.shift_amount;
    core_record.read_data = bytes8(compact.c);
    core_record.prev_data = prev_data;
    Ok(())
}

fn assemble_less_than<F: PrimeField32, RA: Rv64IRecordArena<F>>(
    arena: &mut RA,
    access: &AccessView<'_, F>,
    instruction: &Instruction<F>,
    pc: u32,
    timestamp: u32,
) -> Result<(), ExecutionError> {
    let (adapter_record, core_record): (
        &mut Rv64BaseAluU16AdapterRecord,
        &mut LessThanCoreRecord<BLOCK_FE_WIDTH, U16_BITS>,
    ) = arena.alloc(EmptyAdapterCoreLayout::<F, Rv64BaseAluU16AdapterExecutor>::new());
    let [rs1, rs2] = fill_base_alu_u16_adapter(access, instruction, pc, timestamp, adapter_record)?;
    core_record.b = rs1;
    core_record.c = rs2;
    core_record.local_opcode = instruction
        .opcode
        .local_opcode_idx(LessThanOpcode::CLASS_OFFSET) as u8;
    Ok(())
}

fn assemble_bitwise<F: PrimeField32, RA: Rv64IRecordArena<F>>(
    arena: &mut RA,
    access: &AccessView<'_, F>,
    instruction: &Instruction<F>,
    pc: u32,
    timestamp: u32,
) -> Result<(), ExecutionError> {
    let (adapter_record, core_record): (
        &mut Rv64BaseAluAdapterRecord,
        &mut BitwiseLogicCoreRecord<RV64_REGISTER_NUM_LIMBS>,
    ) = arena.alloc(EmptyAdapterCoreLayout::<
        F,
        Rv64BaseAluAdapterExecutor<RV64_BYTE_BITS>,
    >::new());
    let [rs1, rs2] =
        fill_base_alu_bytes_adapter(access, instruction, pc, timestamp, adapter_record)?;
    core_record.b = rs1;
    core_record.c = rs2;
    core_record.local_opcode = instruction
        .opcode
        .local_opcode_idx(BaseAluOpcode::CLASS_OFFSET) as u8;
    Ok(())
}

fn assemble_shift<F: PrimeField32, RA: Rv64IRecordArena<F>>(
    arena: &mut RA,
    access: &AccessView<'_, F>,
    instruction: &Instruction<F>,
    pc: u32,
    timestamp: u32,
) -> Result<(), ExecutionError> {
    let local_opcode = ShiftOpcode::from_repr(
        instruction
            .opcode
            .local_opcode_idx(ShiftOpcode::CLASS_OFFSET),
    )
    .expect("opcode already classified");

    if local_opcode == ShiftOpcode::SRA {
        let (adapter_record, core_record): (
            &mut Rv64BaseAluU16AdapterRecord,
            &mut ShiftRightArithmeticCoreRecord<BLOCK_FE_WIDTH, U16_BITS>,
        ) = arena.alloc(EmptyAdapterCoreLayout::<F, Rv64BaseAluU16AdapterExecutor>::new());
        let [rs1, rs2] =
            fill_base_alu_u16_adapter(access, instruction, pc, timestamp, adapter_record)?;
        core_record.b = rs1;
        core_record.c = rs2;
    } else {
        let (adapter_record, core_record): (
            &mut Rv64BaseAluU16AdapterRecord,
            &mut ShiftLogicalCoreRecord<BLOCK_FE_WIDTH, U16_BITS>,
        ) = arena.alloc(EmptyAdapterCoreLayout::<F, Rv64BaseAluU16AdapterExecutor>::new());
        let [rs1, rs2] =
            fill_base_alu_u16_adapter(access, instruction, pc, timestamp, adapter_record)?;
        core_record.b = rs1;
        core_record.c = rs2;
        core_record.local_opcode = local_opcode as u8;
    }
    Ok(())
}

fn assemble_add_sub_w<F: PrimeField32, RA: Rv64IRecordArena<F>>(
    arena: &mut RA,
    access: &AccessView<'_, F>,
    instruction: &Instruction<F>,
    pc: u32,
    timestamp: u32,
) -> Result<(), ExecutionError> {
    let (adapter_record, core_record): (
        &mut Rv64BaseAluWU16AdapterRecord,
        &mut AddSubCoreRecord<RV64_WORD_U16_LIMBS>,
    ) = arena.alloc(EmptyAdapterCoreLayout::<F, Rv64BaseAluWU16AdapterExecutor>::new());
    let [rs1, rs2] =
        fill_base_alu_w_u16_adapter(access, instruction, pc, timestamp, adapter_record)?;
    core_record.b = rs1;
    core_record.c = rs2;
    core_record.local_opcode = match BaseAluWOpcode::from_repr(
        instruction
            .opcode
            .local_opcode_idx(BaseAluWOpcode::CLASS_OFFSET),
    )
    .expect("opcode already classified")
    {
        BaseAluWOpcode::ADDW => BaseAluOpcode::ADD as u8,
        BaseAluWOpcode::SUBW => BaseAluOpcode::SUB as u8,
    };
    Ok(())
}

fn assemble_shift_w<F: PrimeField32, RA: Rv64IRecordArena<F>>(
    arena: &mut RA,
    access: &AccessView<'_, F>,
    instruction: &Instruction<F>,
    pc: u32,
    timestamp: u32,
) -> Result<(), ExecutionError> {
    let local_opcode = ShiftWOpcode::from_repr(
        instruction
            .opcode
            .local_opcode_idx(ShiftWOpcode::CLASS_OFFSET),
    )
    .expect("opcode already classified");

    if local_opcode == ShiftWOpcode::SRAW {
        let (adapter_record, core_record): (
            &mut Rv64BaseAluWU16AdapterRecord,
            &mut ShiftRightArithmeticCoreRecord<RV64_WORD_U16_LIMBS, U16_BITS>,
        ) = arena.alloc(EmptyAdapterCoreLayout::<F, Rv64BaseAluWU16AdapterExecutor>::new());
        let [rs1, rs2] =
            fill_base_alu_w_u16_adapter(access, instruction, pc, timestamp, adapter_record)?;
        core_record.b = rs1;
        core_record.c = rs2;
    } else {
        let (adapter_record, core_record): (
            &mut Rv64BaseAluWU16AdapterRecord,
            &mut ShiftLogicalCoreRecord<RV64_WORD_U16_LIMBS, U16_BITS>,
        ) = arena.alloc(EmptyAdapterCoreLayout::<F, Rv64BaseAluWU16AdapterExecutor>::new());
        let [rs1, rs2] =
            fill_base_alu_w_u16_adapter(access, instruction, pc, timestamp, adapter_record)?;
        core_record.b = rs1;
        core_record.c = rs2;
        core_record.local_opcode = match local_opcode {
            ShiftWOpcode::SLLW => ShiftOpcode::SLL as u8,
            ShiftWOpcode::SRLW => ShiftOpcode::SRL as u8,
            ShiftWOpcode::SRAW => unreachable!("handled above"),
        };
    }
    Ok(())
}

fn assemble_branch_eq<F: PrimeField32, RA: Rv64IRecordArena<F>>(
    arena: &mut RA,
    access: &AccessView<'_, F>,
    instruction: &Instruction<F>,
    pc: u32,
    timestamp: u32,
) -> Result<(), ExecutionError> {
    let (adapter_record, core_record): (
        &mut Rv64BranchAdapterRecord,
        &mut BranchEqualCoreRecord<BLOCK_FE_WIDTH>,
    ) = arena.alloc(EmptyAdapterCoreLayout::<F, Rv64BranchAdapterExecutor>::new());
    let [rs1, rs2] = fill_branch_adapter(access, instruction, pc, timestamp, adapter_record)?;
    core_record.a = rs1;
    core_record.b = rs2;
    core_record.imm = instruction.c.as_canonical_u32();
    core_record.local_opcode = instruction
        .opcode
        .local_opcode_idx(BranchEqualOpcode::CLASS_OFFSET) as u8;
    Ok(())
}

fn assemble_branch_lt<F: PrimeField32, RA: Rv64IRecordArena<F>>(
    arena: &mut RA,
    access: &AccessView<'_, F>,
    instruction: &Instruction<F>,
    pc: u32,
    timestamp: u32,
) -> Result<(), ExecutionError> {
    let (adapter_record, core_record): (
        &mut Rv64BranchAdapterRecord,
        &mut BranchLessThanCoreRecord<BLOCK_FE_WIDTH, U16_BITS>,
    ) = arena.alloc(EmptyAdapterCoreLayout::<F, Rv64BranchAdapterExecutor>::new());
    let [rs1, rs2] = fill_branch_adapter(access, instruction, pc, timestamp, adapter_record)?;
    core_record.a = rs1;
    core_record.b = rs2;
    core_record.imm = instruction.c.as_canonical_u32();
    core_record.local_opcode = instruction
        .opcode
        .local_opcode_idx(BranchLessThanOpcode::CLASS_OFFSET) as u8;
    Ok(())
}

fn assemble_jal_lui<F: PrimeField32, RA: Rv64IRecordArena<F>>(
    arena: &mut RA,
    access: &AccessView<'_, F>,
    instruction: &Instruction<F>,
    pc: u32,
    timestamp: u32,
) -> Result<(), ExecutionError> {
    let (adapter_record, core_record): (&mut Rv64RdWriteAdapterRecord, &mut Rv64JalLuiCoreRecord) =
        arena.alloc(EmptyAdapterCoreLayout::<F, Rv64CondRdWriteAdapterExecutor>::new());
    fill_cond_rdwrite_adapter(access, instruction, pc, timestamp, adapter_record)?;
    let local = instruction
        .opcode
        .local_opcode_idx(Rv64JalLuiOpcode::CLASS_OFFSET);
    let is_jal = local == JAL;
    let imm = instruction.c.as_canonical_u32();
    let signed_imm = if is_jal {
        if imm < (1 << 20) {
            imm as i32
        } else {
            let neg_imm = F::ORDER_U32 - imm;
            -(neg_imm as i32)
        }
    } else {
        imm as i32
    };
    core_record.imm = imm;
    core_record.rd_data = run_jal_lui_value(is_jal, pc, signed_imm);
    core_record.is_jal = is_jal;
    Ok(())
}

fn assemble_jalr<F: PrimeField32, RA: Rv64IRecordArena<F>>(
    arena: &mut RA,
    access: &AccessView<'_, F>,
    instruction: &Instruction<F>,
    pc: u32,
    timestamp: u32,
) -> Result<(), ExecutionError> {
    let (adapter_record, core_record): (&mut Rv64JalrAdapterRecord, &mut Rv64JalrCoreRecord) =
        arena.alloc(EmptyAdapterCoreLayout::<F, Rv64JalrAdapterExecutor>::new());
    adapter_record.from_pc = pc;
    adapter_record.from_timestamp = timestamp;
    adapter_record.rs1_ptr = instruction.b.as_canonical_u32();
    let rs1_aux = access.expect_reg_read(timestamp, adapter_record.rs1_ptr, pc)?;
    adapter_record.reads_aux = MemoryReadAuxRecord {
        prev_timestamp: rs1_aux.prev_timestamp,
    };
    let rs1 = u32::from_le_bytes(read_bytes(rs1_aux.entry.value)[..4].try_into().unwrap());
    if instruction.f.is_one() {
        adapter_record.rd_ptr = instruction.a.as_canonical_u32();
        let write_aux = access.expect_reg_write(timestamp + 1, adapter_record.rd_ptr, pc)?;
        adapter_record.writes_aux = MemoryWriteAuxRecord {
            prev_timestamp: write_aux.prev_timestamp,
            prev_data: prev_u16(write_aux),
        };
    } else {
        adapter_record.rd_ptr = u32::MAX;
    }

    core_record.imm = instruction.c.as_canonical_u32() as u16;
    core_record.from_pc = pc;
    core_record.rs1_val = rs1;
    core_record.imm_sign = instruction.g.is_one();
    Ok(())
}

fn assemble_auipc<F: PrimeField32, RA: Rv64IRecordArena<F>>(
    arena: &mut RA,
    access: &AccessView<'_, F>,
    instruction: &Instruction<F>,
    pc: u32,
    timestamp: u32,
) -> Result<(), ExecutionError> {
    let (adapter_record, core_record): (&mut Rv64RdWriteAdapterRecord, &mut Rv64AuipcCoreRecord) =
        arena.alloc(EmptyAdapterCoreLayout::<F, Rv64RdWriteAdapterExecutor>::new());
    fill_rdwrite_adapter(access, instruction, pc, timestamp, adapter_record)?;
    core_record.from_pc = pc;
    core_record.imm = instruction.c.as_canonical_u32();
    Ok(())
}

fn assemble_mul<F: PrimeField32, RA: Rv64MRecordArena<F>>(
    arena: &mut RA,
    access: &AccessView<'_, F>,
    instruction: &Instruction<F>,
    pc: u32,
    timestamp: u32,
) -> Result<(), ExecutionError> {
    let (adapter_record, core_record): (
        &mut Rv64MultAdapterRecord,
        &mut MultiplicationCoreRecord<RV64_REGISTER_NUM_LIMBS, RV64_BYTE_BITS>,
    ) = arena.alloc(EmptyAdapterCoreLayout::<F, Rv64MultAdapterExecutor>::new());
    let [rs1, rs2] = fill_mult_adapter(access, instruction, pc, timestamp, adapter_record)?;
    core_record.b = rs1;
    core_record.c = rs2;
    Ok(())
}

fn assemble_mulh<F: PrimeField32, RA: Rv64MRecordArena<F>>(
    arena: &mut RA,
    access: &AccessView<'_, F>,
    instruction: &Instruction<F>,
    pc: u32,
    timestamp: u32,
) -> Result<(), ExecutionError> {
    let (adapter_record, core_record): (
        &mut Rv64MultAdapterRecord,
        &mut MulHCoreRecord<RV64_REGISTER_NUM_LIMBS, RV64_BYTE_BITS>,
    ) = arena.alloc(EmptyAdapterCoreLayout::<F, Rv64MultAdapterExecutor>::new());
    let [rs1, rs2] = fill_mult_adapter(access, instruction, pc, timestamp, adapter_record)?;
    core_record.b = rs1;
    core_record.c = rs2;
    core_record.local_opcode = instruction
        .opcode
        .local_opcode_idx(MulHOpcode::CLASS_OFFSET) as u8;
    Ok(())
}

fn assemble_mul_w<F: PrimeField32, RA: Rv64MRecordArena<F>>(
    arena: &mut RA,
    access: &AccessView<'_, F>,
    instruction: &Instruction<F>,
    pc: u32,
    timestamp: u32,
) -> Result<(), ExecutionError> {
    let (adapter_record, core_record): (
        &mut Rv64MultWAdapterRecord,
        &mut MultiplicationCoreRecord<RV64_WORD_NUM_LIMBS, RV64_BYTE_BITS>,
    ) = arena.alloc(EmptyAdapterCoreLayout::<F, Rv64MultWAdapterExecutor>::new());
    let [rs1, rs2] = fill_mult_w_adapter(access, instruction, pc, timestamp, adapter_record)?;
    core_record.b = rs1;
    core_record.c = rs2;
    Ok(())
}

fn assemble_loadstore<F: PrimeField32, RA: Rv64IRecordArena<F>>(
    arena: &mut RA,
    access: &AccessView<'_, F>,
    instruction: &Instruction<F>,
    pc: u32,
    timestamp: u32,
) -> Result<(), ExecutionError> {
    let local_opcode = Rv64LoadStoreOpcode::from_repr(
        instruction
            .opcode
            .local_opcode_idx(Rv64LoadStoreOpcode::CLASS_OFFSET),
    )
    .expect("assembler is registered only for RV64 load/store opcodes");
    let (adapter_record, core_record): (
        &mut Rv64LoadStoreAdapterRecord,
        &mut LoadStoreCoreRecord<RV64_REGISTER_NUM_LIMBS>,
    ) = arena.alloc(EmptyAdapterCoreLayout::<F, Rv64LoadStoreAdapterExecutor>::new());

    let LoadStoreInputs {
        rs1_val,
        aligned_ptr,
        shift_amount,
    } = fill_loadstore_start(access, instruction, pc, timestamp, adapter_record)?;
    let enabled = instruction.f.is_one();
    let is_load = matches!(
        local_opcode,
        Rv64LoadStoreOpcode::LOADD
            | Rv64LoadStoreOpcode::LOADWU
            | Rv64LoadStoreOpcode::LOADHU
            | Rv64LoadStoreOpcode::LOADBU
    );

    let (read_data, prev_data) = if is_load {
        adapter_record.mem_as = RV64_MEMORY_AS as u8;
        let read_aux = access.expect_memory_read(timestamp + 1, aligned_ptr, pc)?;
        adapter_record.read_data_aux.prev_timestamp = read_aux.prev_timestamp;
        let read_data = read_bytes(read_aux.entry.value);
        let prev_data = if enabled {
            adapter_record.rd_rs2_ptr = instruction.a.as_canonical_u32();
            let write_aux =
                access.expect_reg_write(timestamp + 2, adapter_record.rd_rs2_ptr, pc)?;
            adapter_record.write_prev_timestamp = write_aux.prev_timestamp;
            prev_bytes(write_aux)
        } else {
            adapter_record.rd_rs2_ptr = u32::MAX;
            [0; RV64_REGISTER_NUM_LIMBS]
        };
        (read_data, prev_data)
    } else {
        // Stores write the instruction's actual address space: main memory or
        // the public-values address space for REVEAL, exactly as the
        // interpreter LoadStoreAdapterExecutor does.
        let mem_as = instruction.e.as_canonical_u32();
        adapter_record.mem_as = mem_as as u8;
        adapter_record.rd_rs2_ptr = instruction.a.as_canonical_u32();
        let read_aux = access.expect_reg_read(timestamp + 1, adapter_record.rd_rs2_ptr, pc)?;
        adapter_record.read_data_aux.prev_timestamp = read_aux.prev_timestamp;
        let write_aux = access.expect_memory_write(timestamp + 2, mem_as, aligned_ptr, pc)?;
        adapter_record.write_prev_timestamp = write_aux.prev_timestamp;
        (read_bytes(read_aux.entry.value), prev_bytes(write_aux))
    };

    debug_assert_eq!(adapter_record.rs1_val, rs1_val);
    core_record.local_opcode = local_opcode as u8;
    core_record.shift_amount = shift_amount;
    core_record.read_data = read_data;
    core_record.prev_data = prev_data;
    Ok(())
}

fn assemble_load_sign_extend<F: PrimeField32, RA: Rv64IRecordArena<F>>(
    arena: &mut RA,
    access: &AccessView<'_, F>,
    instruction: &Instruction<F>,
    pc: u32,
    timestamp: u32,
) -> Result<(), ExecutionError> {
    let local_opcode = Rv64LoadStoreOpcode::from_repr(
        instruction
            .opcode
            .local_opcode_idx(Rv64LoadStoreOpcode::CLASS_OFFSET),
    )
    .expect("assembler is registered only for RV64 load/store opcodes");
    let (adapter_record, core_record): (
        &mut Rv64LoadStoreAdapterRecord,
        &mut LoadSignExtendCoreRecord<RV64_REGISTER_NUM_LIMBS>,
    ) = arena.alloc(EmptyAdapterCoreLayout::<F, Rv64LoadStoreAdapterExecutor>::new());

    let LoadStoreInputs {
        aligned_ptr,
        shift_amount,
        ..
    } = fill_loadstore_start(access, instruction, pc, timestamp, adapter_record)?;
    adapter_record.mem_as = RV64_MEMORY_AS as u8;
    let read_aux = access.expect_memory_read(timestamp + 1, aligned_ptr, pc)?;
    adapter_record.read_data_aux.prev_timestamp = read_aux.prev_timestamp;
    let read_data = read_bytes(read_aux.entry.value);

    let prev_data = if instruction.f.is_one() {
        adapter_record.rd_rs2_ptr = instruction.a.as_canonical_u32();
        let write_aux = access.expect_reg_write(timestamp + 2, adapter_record.rd_rs2_ptr, pc)?;
        adapter_record.write_prev_timestamp = write_aux.prev_timestamp;
        prev_bytes(write_aux)
    } else {
        adapter_record.rd_rs2_ptr = u32::MAX;
        [0; RV64_REGISTER_NUM_LIMBS]
    };

    core_record.is_byte = local_opcode == Rv64LoadStoreOpcode::LOADB;
    core_record.is_word = local_opcode == Rv64LoadStoreOpcode::LOADW;
    core_record.shift_amount = shift_amount;
    core_record.read_data = read_data;
    core_record.prev_data = prev_data;
    Ok(())
}

fn assemble_divrem<F: PrimeField32, RA: Rv64MRecordArena<F>>(
    arena: &mut RA,
    access: &AccessView<'_, F>,
    instruction: &Instruction<F>,
    pc: u32,
    timestamp: u32,
) -> Result<(), ExecutionError> {
    let (adapter_record, core_record): (
        &mut Rv64MultAdapterRecord,
        &mut DivRemCoreRecord<RV64_REGISTER_NUM_LIMBS>,
    ) = arena.alloc(EmptyAdapterCoreLayout::<F, Rv64MultAdapterExecutor>::new());
    let [rs1, rs2] = fill_mult_adapter(access, instruction, pc, timestamp, adapter_record)?;
    core_record.b = rs1;
    core_record.c = rs2;
    core_record.local_opcode = instruction
        .opcode
        .local_opcode_idx(DivRemOpcode::CLASS_OFFSET) as u8;
    Ok(())
}

fn assemble_divrem_w<F: PrimeField32, RA: Rv64MRecordArena<F>>(
    arena: &mut RA,
    access: &AccessView<'_, F>,
    instruction: &Instruction<F>,
    pc: u32,
    timestamp: u32,
) -> Result<(), ExecutionError> {
    let (adapter_record, core_record): (
        &mut Rv64MultWAdapterRecord,
        &mut DivRemCoreRecord<RV64_WORD_NUM_LIMBS>,
    ) = arena.alloc(EmptyAdapterCoreLayout::<F, Rv64MultWAdapterExecutor>::new());
    let [rs1, rs2] = fill_mult_w_adapter(access, instruction, pc, timestamp, adapter_record)?;
    core_record.b = rs1;
    core_record.c = rs2;
    core_record.local_opcode = instruction
        .opcode
        .local_opcode_idx(DivRemWOpcode::CLASS_OFFSET) as u8;
    Ok(())
}

fn assemble_hintstore<F: PrimeField32, RA: Rv64IoRecordArena<F>>(
    arena: &mut RA,
    access: &AccessView<'_, F>,
    instruction: &Instruction<F>,
    pc: u32,
    timestamp: u32,
) -> Result<(), ExecutionError> {
    let local_opcode = Rv64HintStoreOpcode::from_repr(
        instruction
            .opcode
            .local_opcode_idx(Rv64HintStoreOpcode::CLASS_OFFSET),
    )
    .expect("opcode already classified");
    let mem_ptr_ptr = instruction.b.as_canonical_u32();
    let mem_ptr_aux = access.expect_reg_read(timestamp, mem_ptr_ptr, pc)?;
    let mem_ptr = read_low_u32(mem_ptr_aux.entry.value);

    let (num_words, num_words_aux) = if local_opcode == Rv64HintStoreOpcode::HINT_STORED {
        (1, None)
    } else {
        let num_words_ptr = instruction.a.as_canonical_u32();
        let aux = access.expect_reg_read(timestamp + 1, num_words_ptr, pc)?;
        (read_low_u32(aux.entry.value), Some(aux))
    };
    if num_words == 0 {
        return Err(rvr_error(format!(
            "hintstore at pc {pc:#x} requested zero rows"
        )));
    }

    let record: Rv64HintStoreRecordMut<'_> = arena.alloc(MultiRowLayout::new(
        Rv64HintStoreMetadata::new(num_words as usize),
    ));
    record.inner.num_words = num_words;
    record.inner.from_pc = pc;
    record.inner.timestamp = timestamp;
    record.inner.mem_ptr_ptr = mem_ptr_ptr;
    record.inner.mem_ptr = mem_ptr;
    record.inner.mem_ptr_aux_record.prev_timestamp = mem_ptr_aux.prev_timestamp;
    if let Some(aux) = num_words_aux {
        record.inner.num_words_ptr = instruction.a.as_canonical_u32();
        record.inner.num_words_read.prev_timestamp = aux.prev_timestamp;
    } else {
        record.inner.num_words_ptr = u32::MAX;
        record.inner.num_words_read.prev_timestamp = 0;
    }

    for (idx, var) in record.var.iter_mut().enumerate() {
        let write_timestamp = timestamp + 2 + idx as u32 * 3;
        let write_addr = mem_ptr + idx as u32 * RV64_REGISTER_NUM_LIMBS as u32;
        let write_aux =
            access.expect_memory_write(write_timestamp, RV64_MEMORY_AS, write_addr, pc)?;
        var.data = read_bytes(write_aux.entry.value);
        var.data_write_aux = MemoryWriteBytesAuxRecord {
            prev_timestamp: write_aux.prev_timestamp,
            prev_data: prev_bytes(write_aux),
        };
    }
    Ok(())
}

fn reject_hintstore_compact<F: PrimeField32, RA: Rv64IoRecordArena<F>>(
    _arena: &mut RA,
    _instruction: &Instruction<F>,
    _compact: &[u8],
    pc: u32,
) -> Result<(), ExecutionError> {
    Err(rvr_error(format!(
        "HintStore at pc {pc:#x} requires its packed variable-row arena-native target"
    )))
}

fn assemble_phantom<F: PrimeField32, RA: Rv64IRecordArena<F>>(
    arena: &mut RA,
    _access: &AccessView<'_, F>,
    instruction: &Instruction<F>,
    pc: u32,
    timestamp: u32,
) -> Result<(), ExecutionError> {
    let record: &mut PhantomRecord = arena.alloc(EmptyMultiRowLayout::default());
    record.pc = pc;
    record.timestamp = timestamp;
    record.operands = [instruction.a, instruction.b, instruction.c].map(|x| x.as_canonical_u32());
    Ok(())
}

fn assemble_phantom_inline<F: PrimeField32, RA: Rv64IRecordArena<F>>(
    arena: &mut RA,
    _instruction: &Instruction<F>,
    compact: &[u8],
    pc: u32,
) -> Result<(), ExecutionError> {
    if compact.len() != size_of::<PhantomRecord>() {
        return Err(rvr_error(format!(
            "invalid Phantom inline record size {} at pc {pc:#x}; expected {}",
            compact.len(),
            size_of::<PhantomRecord>()
        )));
    }
    let mut words = compact
        .chunks_exact(size_of::<u32>())
        .map(|word| u32::from_ne_bytes(word.try_into().unwrap()));
    let record: &mut PhantomRecord = arena.alloc(EmptyMultiRowLayout::default());
    record.pc = words.next().unwrap();
    record.operands = [
        words.next().unwrap(),
        words.next().unwrap(),
        words.next().unwrap(),
    ];
    record.timestamp = words.next().unwrap();
    debug_assert!(words.next().is_none());
    Ok(())
}

struct LoadStoreInputs {
    rs1_val: u32,
    aligned_ptr: u32,
    shift_amount: u8,
}

fn fill_loadstore_start<F: PrimeField32>(
    access: &AccessView<'_, F>,
    instruction: &Instruction<F>,
    pc: u32,
    timestamp: u32,
    record: &mut Rv64LoadStoreAdapterRecord,
) -> Result<LoadStoreInputs, ExecutionError> {
    record.from_pc = pc;
    record.from_timestamp = timestamp;
    record.rs1_ptr = instruction.b.as_canonical_u32();
    let rs1_aux = access.expect_reg_read(timestamp, record.rs1_ptr, pc)?;
    record.rs1_aux_record.prev_timestamp = rs1_aux.prev_timestamp;
    let rs1_val = read_low_u32(rs1_aux.entry.value);
    record.rs1_val = rs1_val;
    record.imm = instruction.c.as_canonical_u32() as u16;
    record.imm_sign = instruction.g.is_one();
    let ptr = rs1_val.wrapping_add(sign_extend_imm16(record.imm as u32, record.imm_sign));
    let shift_amount = (ptr & (RV64_REGISTER_NUM_LIMBS as u32 - 1)) as u8;
    Ok(LoadStoreInputs {
        rs1_val,
        aligned_ptr: ptr - shift_amount as u32,
        shift_amount,
    })
}

fn fill_base_alu_u16_adapter<F: PrimeField32>(
    access: &AccessView<'_, F>,
    instruction: &Instruction<F>,
    pc: u32,
    timestamp: u32,
    record: &mut Rv64BaseAluU16AdapterRecord,
) -> Result<[[u16; BLOCK_FE_WIDTH]; 2], ExecutionError> {
    record.from_pc = pc;
    record.from_timestamp = timestamp;
    record.rs1_ptr = instruction.b.as_canonical_u32();
    let rs1_aux = access.expect_reg_read(timestamp, record.rs1_ptr, pc)?;
    record.reads_aux[0].prev_timestamp = rs1_aux.prev_timestamp;
    let rs1 = read_u16_block(rs1_aux.entry.value);

    let rs2 = if instruction.e.as_canonical_u32() == RV64_REGISTER_AS {
        record.rs2_as = RV64_REGISTER_AS as u8;
        record.rs2_imm_sign = false;
        record.rs2 = instruction.c.as_canonical_u32();
        let rs2_aux = access.expect_reg_read(timestamp + 1, record.rs2, pc)?;
        record.reads_aux[1].prev_timestamp = rs2_aux.prev_timestamp;
        read_u16_block(rs2_aux.entry.value)
    } else {
        record.rs2_as = RV64_IMM_AS as u8;
        let imm = instruction.c.as_canonical_u32();
        record.rs2 = imm;
        let imm64 = imm_to_rv64_u64(imm);
        let sign_u16 = (imm64 >> U16_BITS) as u16;
        record.rs2_imm_sign = sign_u16 != 0;
        [imm64 as u16, sign_u16, sign_u16, sign_u16]
    };

    record.rd_ptr = instruction.a.as_canonical_u32();
    let write_aux = access.expect_reg_write(timestamp + 2, record.rd_ptr, pc)?;
    record.writes_aux = MemoryWriteAuxRecord {
        prev_timestamp: write_aux.prev_timestamp,
        prev_data: prev_u16(write_aux),
    };
    Ok([rs1, rs2])
}

fn fill_base_alu_bytes_adapter<F: PrimeField32>(
    access: &AccessView<'_, F>,
    instruction: &Instruction<F>,
    pc: u32,
    timestamp: u32,
    record: &mut Rv64BaseAluAdapterRecord,
) -> Result<[[u8; RV64_REGISTER_NUM_LIMBS]; 2], ExecutionError> {
    record.from_pc = pc;
    record.from_timestamp = timestamp;
    record.rs1_ptr = instruction.b.as_canonical_u32();
    let rs1_aux = access.expect_reg_read(timestamp, record.rs1_ptr, pc)?;
    record.reads_aux[0].prev_timestamp = rs1_aux.prev_timestamp;
    let rs1 = read_bytes(rs1_aux.entry.value);

    let rs2 = if instruction.e.as_canonical_u32() == RV64_REGISTER_AS {
        record.rs2_as = RV64_REGISTER_AS as u8;
        record.rs2 = instruction.c.as_canonical_u32();
        let rs2_aux = access.expect_reg_read(timestamp + 1, record.rs2, pc)?;
        record.reads_aux[1].prev_timestamp = rs2_aux.prev_timestamp;
        read_bytes(rs2_aux.entry.value)
    } else {
        record.rs2_as = RV64_IMM_AS as u8;
        let imm = instruction.c.as_canonical_u32();
        record.rs2 = imm;
        imm_to_rv64_u64(imm).to_le_bytes()
    };

    record.rd_ptr = instruction.a.as_canonical_u32();
    let write_aux = access.expect_reg_write(timestamp + 2, record.rd_ptr, pc)?;
    record.writes_aux = MemoryWriteAuxRecord {
        prev_timestamp: write_aux.prev_timestamp,
        prev_data: prev_bytes(write_aux),
    };
    Ok([rs1, rs2])
}

fn fill_base_alu_w_u16_adapter<F: PrimeField32>(
    access: &AccessView<'_, F>,
    instruction: &Instruction<F>,
    pc: u32,
    timestamp: u32,
    record: &mut Rv64BaseAluWU16AdapterRecord,
) -> Result<[[u16; RV64_WORD_U16_LIMBS]; 2], ExecutionError> {
    record.from_pc = pc;
    record.from_timestamp = timestamp;
    record.rs1_ptr = instruction.b.as_canonical_u32();
    let rs1_aux = access.expect_reg_read(timestamp, record.rs1_ptr, pc)?;
    record.reads_aux[0].prev_timestamp = rs1_aux.prev_timestamp;
    let rs1_full = read_u16_block(rs1_aux.entry.value);
    record.rs1_high = [rs1_full[2], rs1_full[3]];
    let rs1 = [rs1_full[0], rs1_full[1]];

    let rs2 = if instruction.e.as_canonical_u32() == RV64_REGISTER_AS {
        record.rs2_as = RV64_REGISTER_AS as u8;
        record.rs2_imm_sign = false;
        record.rs2 = instruction.c.as_canonical_u32();
        let rs2_aux = access.expect_reg_read(timestamp + 1, record.rs2, pc)?;
        record.reads_aux[1].prev_timestamp = rs2_aux.prev_timestamp;
        let rs2_full = read_u16_block(rs2_aux.entry.value);
        record.rs2_high = [rs2_full[2], rs2_full[3]];
        [rs2_full[0], rs2_full[1]]
    } else {
        record.rs2_as = RV64_IMM_AS as u8;
        let imm = instruction.c.as_canonical_u32();
        record.rs2 = imm;
        let imm64 = imm_to_rv64_u64(imm);
        let sign_u16 = (imm64 >> U16_BITS) as u16;
        record.rs2_imm_sign = sign_u16 != 0;
        record.rs2_high = [sign_u16; RV64_WORD_U16_LIMBS];
        [(imm64 & u16::MAX as u64) as u16, (imm64 >> U16_BITS) as u16]
    };

    record.rd_ptr = instruction.a.as_canonical_u32();
    let write_aux = access.expect_reg_write(timestamp + 2, record.rd_ptr, pc)?;
    let write_full = read_u16_block(write_aux.entry.value);
    record.result_high = write_full[RV64_WORD_U16_LIMBS - 1];
    record.result_sign = (record.result_high >> (U16_BITS - 1)) as u8;
    record.writes_aux = MemoryWriteAuxRecord {
        prev_timestamp: write_aux.prev_timestamp,
        prev_data: prev_u16(write_aux),
    };
    Ok([rs1, rs2])
}

fn fill_branch_adapter<F: PrimeField32>(
    access: &AccessView<'_, F>,
    instruction: &Instruction<F>,
    pc: u32,
    timestamp: u32,
    record: &mut Rv64BranchAdapterRecord,
) -> Result<[[u16; BLOCK_FE_WIDTH]; 2], ExecutionError> {
    record.from_pc = pc;
    record.from_timestamp = timestamp;
    record.rs1_ptr = instruction.a.as_canonical_u32();
    let rs1_aux = access.expect_reg_read(timestamp, record.rs1_ptr, pc)?;
    record.reads_aux[0].prev_timestamp = rs1_aux.prev_timestamp;
    record.rs2_ptr = instruction.b.as_canonical_u32();
    let rs2_aux = access.expect_reg_read(timestamp + 1, record.rs2_ptr, pc)?;
    record.reads_aux[1].prev_timestamp = rs2_aux.prev_timestamp;
    Ok([
        read_u16_block(rs1_aux.entry.value),
        read_u16_block(rs2_aux.entry.value),
    ])
}

fn fill_rdwrite_adapter<F: PrimeField32>(
    access: &AccessView<'_, F>,
    instruction: &Instruction<F>,
    pc: u32,
    timestamp: u32,
    record: &mut Rv64RdWriteAdapterRecord,
) -> Result<(), ExecutionError> {
    record.from_pc = pc;
    record.from_timestamp = timestamp;
    record.rd_ptr = instruction.a.as_canonical_u32();
    let write_aux = access.expect_reg_write(timestamp, record.rd_ptr, pc)?;
    record.rd_aux_record = MemoryWriteAuxRecord {
        prev_timestamp: write_aux.prev_timestamp,
        prev_data: prev_u16(write_aux),
    };
    Ok(())
}

fn fill_cond_rdwrite_adapter<F: PrimeField32>(
    access: &AccessView<'_, F>,
    instruction: &Instruction<F>,
    pc: u32,
    timestamp: u32,
    record: &mut Rv64RdWriteAdapterRecord,
) -> Result<(), ExecutionError> {
    record.from_pc = pc;
    record.from_timestamp = timestamp;
    if instruction.f.is_one() {
        record.rd_ptr = instruction.a.as_canonical_u32();
        let write_aux = access.expect_reg_write(timestamp, record.rd_ptr, pc)?;
        record.rd_aux_record = MemoryWriteAuxRecord {
            prev_timestamp: write_aux.prev_timestamp,
            prev_data: prev_u16(write_aux),
        };
    } else {
        record.rd_ptr = u32::MAX;
    }
    Ok(())
}

fn fill_mult_adapter<F: PrimeField32>(
    access: &AccessView<'_, F>,
    instruction: &Instruction<F>,
    pc: u32,
    timestamp: u32,
    record: &mut Rv64MultAdapterRecord,
) -> Result<[[u8; RV64_REGISTER_NUM_LIMBS]; 2], ExecutionError> {
    record.from_pc = pc;
    record.from_timestamp = timestamp;
    record.rs1_ptr = instruction.b.as_canonical_u32();
    let rs1_aux = access.expect_reg_read(timestamp, record.rs1_ptr, pc)?;
    record.reads_aux[0].prev_timestamp = rs1_aux.prev_timestamp;
    record.rs2_ptr = instruction.c.as_canonical_u32();
    let rs2_aux = access.expect_reg_read(timestamp + 1, record.rs2_ptr, pc)?;
    record.reads_aux[1].prev_timestamp = rs2_aux.prev_timestamp;
    record.rd_ptr = instruction.a.as_canonical_u32();
    let write_aux = access.expect_reg_write(timestamp + 2, record.rd_ptr, pc)?;
    record.writes_aux = MemoryWriteAuxRecord {
        prev_timestamp: write_aux.prev_timestamp,
        prev_data: prev_bytes(write_aux),
    };
    Ok([
        read_bytes(rs1_aux.entry.value),
        read_bytes(rs2_aux.entry.value),
    ])
}

fn fill_mult_w_adapter<F: PrimeField32>(
    access: &AccessView<'_, F>,
    instruction: &Instruction<F>,
    pc: u32,
    timestamp: u32,
    record: &mut Rv64MultWAdapterRecord,
) -> Result<[[u8; RV64_WORD_NUM_LIMBS]; 2], ExecutionError> {
    record.from_pc = pc;
    record.from_timestamp = timestamp;
    record.rs1_ptr = instruction.b.as_canonical_u32();
    let rs1_aux = access.expect_reg_read(timestamp, record.rs1_ptr, pc)?;
    record.reads_aux[0].prev_timestamp = rs1_aux.prev_timestamp;
    let rs1_full = read_bytes(rs1_aux.entry.value);
    record
        .rs1_high
        .copy_from_slice(&rs1_full[RV64_WORD_NUM_LIMBS..]);

    record.rs2_ptr = instruction.c.as_canonical_u32();
    let rs2_aux = access.expect_reg_read(timestamp + 1, record.rs2_ptr, pc)?;
    record.reads_aux[1].prev_timestamp = rs2_aux.prev_timestamp;
    let rs2_full = read_bytes(rs2_aux.entry.value);
    record
        .rs2_high
        .copy_from_slice(&rs2_full[RV64_WORD_NUM_LIMBS..]);

    record.rd_ptr = instruction.a.as_canonical_u32();
    let write_aux = access.expect_reg_write(timestamp + 2, record.rd_ptr, pc)?;
    let write_full = read_bytes(write_aux.entry.value);
    record.result_word_msl = write_full[RV64_WORD_NUM_LIMBS - 1];
    record.result_sign = record.result_word_msl >> (RV64_BYTE_BITS as u8 - 1);
    record.writes_aux = MemoryWriteAuxRecord {
        prev_timestamp: write_aux.prev_timestamp,
        prev_data: prev_bytes(write_aux),
    };
    Ok([
        rs1_full[..RV64_WORD_NUM_LIMBS].try_into().unwrap(),
        rs2_full[..RV64_WORD_NUM_LIMBS].try_into().unwrap(),
    ])
}

fn read_bytes(value: u64) -> [u8; RV64_REGISTER_NUM_LIMBS] {
    value.to_le_bytes()
}

fn read_low_u32(value: u64) -> u32 {
    u32::from_le_bytes(read_bytes(value)[..RV64_WORD_NUM_LIMBS].try_into().unwrap())
}

fn sign_extend_imm16(imm: u32, sign: bool) -> u32 {
    imm + (sign as u32) * (u32::MAX << U16_BITS)
}

fn read_u16_block(value: u64) -> [u16; BLOCK_FE_WIDTH] {
    rv64_bytes_to_u16_block(read_bytes(value))
}

fn prev_u16<F: PrimeField32>(aux: &PreflightMemoryAccessAux<F>) -> [u16; BLOCK_FE_WIDTH] {
    aux.prev_data
        .map(|cell| cell.as_canonical_u32().try_into().expect("u16 memory cell"))
}

fn prev_bytes<F: PrimeField32>(aux: &PreflightMemoryAccessAux<F>) -> [u8; RV64_REGISTER_NUM_LIMBS] {
    rv64_u16_block_to_bytes(prev_u16(aux))
}

fn run_jal_lui_value(is_jal: bool, pc: u32, imm: i32) -> [u16; BLOCK_FE_WIDTH] {
    if is_jal {
        let rd_low = pc.wrapping_add(DEFAULT_PC_STEP);
        rv64_u32_to_u16_block(rd_low)
    } else {
        let rd_low = (imm as u32) << 12;
        let lo = (rd_low & u16::MAX as u32) as u16;
        let hi = (rd_low >> U16_BITS) as u16;
        let sign = if (hi >> (U16_BITS - 1)) & 1 == 1 {
            u16::MAX
        } else {
            0
        };
        [lo, hi, sign, sign]
    }
}

fn rvr_error(message: impl Into<String>) -> ExecutionError {
    ExecutionError::RvrExecution(message.into())
}
