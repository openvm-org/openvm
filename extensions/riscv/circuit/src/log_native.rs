use openvm_circuit::{
    arch::{
        rvr::{
            generate_record_arenas_from_logs, LogNativeAccessView, LogNativeAssemblerRegistry,
            PreflightMemoryAccessAux, RvrPreflightOutput, VmRvrLogNativeExtension,
            PREFLIGHT_MEMORY_KIND_READ, PREFLIGHT_MEMORY_KIND_WRITE,
        },
        Arena, EmptyAdapterCoreLayout, EmptyMultiRowLayout, ExecutionError, MultiRowLayout,
        RecordArena, BLOCK_FE_WIDTH,
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
    Rv64HintStoreMetadata, Rv64HintStoreRecordMut, Rv64JalLuiCoreRecord, Rv64JalrCoreRecord,
    ShiftLogicalCoreRecord, ShiftRightArithmeticCoreRecord,
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
    output: &RvrPreflightOutput<F>,
    capacities: &[(usize, usize)],
    pc_to_air_idx: &[Option<usize>],
) -> Result<Vec<RA>, ExecutionError> {
    let mut registry = LogNativeAssemblerRegistry::new();
    crate::Rv64ImConfig::default().extend_rvr_log_native(&mut registry);
    generate_record_arenas_from_logs(&registry, exe, output, capacities, pc_to_air_idx)
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
        registry.register(
            [BaseAluOpcode::XOR, BaseAluOpcode::OR, BaseAluOpcode::AND]
                .map(|opcode| opcode.global_opcode()),
            assemble_bitwise::<F, RA>,
        );
        registry.register(
            BaseAluWOpcode::iter().map(|opcode| opcode.global_opcode()),
            assemble_add_sub_w::<F, RA>,
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
