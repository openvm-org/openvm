//! Log-native record assembly for modular and quadratic-extension arithmetic.

use std::mem::{align_of, offset_of, size_of};

use openvm_algebra_transpiler::{Fp2Opcode, Rv64ModularArithmeticOpcode};
use openvm_circuit::{
    arch::{
        rvr::{
            ArenaNativeGeometry, ArenaNativeLayout, LogNativeAccessView,
            LogNativeAssemblerRegistry, PreflightMemoryAccessAux, VmRvrLogNativeExtension,
            PREFLIGHT_MEMORY_KIND_READ, PREFLIGHT_MEMORY_KIND_TOUCH, PREFLIGHT_MEMORY_KIND_WRITE,
        },
        AdapterCoreLayout, AdapterTraceExecutor, Arena, EmptyAdapterCoreLayout, ExecutionError,
        RecordArena, BLOCK_FE_WIDTH, MEMORY_BLOCK_BYTES,
    },
    system::memory::offline_checker::{
        MemoryReadAuxRecord, MemoryWriteBytesAuxRecord, MemoryWriteU16AuxRecord,
    },
};
use openvm_instructions::{
    instruction::Instruction,
    riscv::{RV64_MEMORY_AS, RV64_REGISTER_AS, RV64_REGISTER_NUM_LIMBS},
    LocalOpcode, VmOpcode,
};
use openvm_mod_circuit_builder::{
    FieldExpressionCoreRecordMut, FieldExpressionMetadata, FieldExpressionRecordLayout,
};
use openvm_riscv_adapters::{
    Rv64IsEqualModU16AdapterExecutor, Rv64IsEqualModU16AdapterRecord, Rv64VecHeapAdapterExecutor,
    Rv64VecHeapAdapterRecord,
};
use openvm_riscv_circuit::log_native::{Rv64IRecordArena, Rv64IoRecordArena, Rv64MRecordArena};
use openvm_stark_backend::p3_field::PrimeField32;
use rvr_openvm_ext_algebra::{ModIsEqRecordDescriptor, VecHeapRecordDescriptor};
use strum::EnumCount;

use crate::{
    modular_chip::ModularIsEqualRecord, Fp2Extension, ModularExtension, Rv64ModularConfig,
    Rv64ModularWithFp2Config, FP2_BLOCKS_32, FP2_BLOCKS_48, MODULAR_BLOCKS_32, MODULAR_BLOCKS_48,
    NUM_LIMBS_32, NUM_LIMBS_32_U16, NUM_LIMBS_48, NUM_LIMBS_48_U16,
};

/// Field-expression record layout backed by `Rv64VecHeapAdapter`.
pub type VecHeapLayout<
    F,
    const NUM_READS: usize,
    const READ_BLOCKS: usize,
    const WRITE_BLOCKS: usize,
> = FieldExpressionRecordLayout<
    F,
    Rv64VecHeapAdapterExecutor<NUM_READS, READ_BLOCKS, WRITE_BLOCKS>,
>;

/// Mutable record pair for a field expression backed by `Rv64VecHeapAdapter`.
pub type VecHeapRecordMut<
    'a,
    const NUM_READS: usize,
    const READ_BLOCKS: usize,
    const WRITE_BLOCKS: usize,
> = (
    &'a mut Rv64VecHeapAdapterRecord<NUM_READS, READ_BLOCKS, WRITE_BLOCKS>,
    FieldExpressionCoreRecordMut<'a>,
);

type IsEq32Adapter = Rv64IsEqualModU16AdapterExecutor<2, MODULAR_BLOCKS_32, NUM_LIMBS_32_U16>;
type IsEq48Adapter = Rv64IsEqualModU16AdapterExecutor<2, MODULAR_BLOCKS_48, NUM_LIMBS_48_U16>;
type IsEq32RecordMut<'a> = (
    &'a mut Rv64IsEqualModU16AdapterRecord<2, MODULAR_BLOCKS_32>,
    &'a mut ModularIsEqualRecord<NUM_LIMBS_32_U16>,
);
type IsEq48RecordMut<'a> = (
    &'a mut Rv64IsEqualModU16AdapterRecord<2, MODULAR_BLOCKS_48>,
    &'a mut ModularIsEqualRecord<NUM_LIMBS_48_U16>,
);

/// Record-arena requirements contributed by the modular extension.
///
/// The bounds are deliberately arena-agnostic: both `MatrixRecordArena` and
/// `DenseRecordArena` can satisfy them.
pub trait ModularRecordArena<F>:
    Arena
    + for<'a> RecordArena<
        'a,
        VecHeapLayout<F, 2, MODULAR_BLOCKS_32, MODULAR_BLOCKS_32>,
        VecHeapRecordMut<'a, 2, MODULAR_BLOCKS_32, MODULAR_BLOCKS_32>,
    > + for<'a> RecordArena<
        'a,
        VecHeapLayout<F, 2, MODULAR_BLOCKS_48, MODULAR_BLOCKS_48>,
        VecHeapRecordMut<'a, 2, MODULAR_BLOCKS_48, MODULAR_BLOCKS_48>,
    > + for<'a> RecordArena<'a, EmptyAdapterCoreLayout<F, IsEq32Adapter>, IsEq32RecordMut<'a>>
    + for<'a> RecordArena<'a, EmptyAdapterCoreLayout<F, IsEq48Adapter>, IsEq48RecordMut<'a>>
{
}

impl<F, RA> ModularRecordArena<F> for RA where
    RA: Arena
        + for<'a> RecordArena<
            'a,
            VecHeapLayout<F, 2, MODULAR_BLOCKS_32, MODULAR_BLOCKS_32>,
            VecHeapRecordMut<'a, 2, MODULAR_BLOCKS_32, MODULAR_BLOCKS_32>,
        > + for<'a> RecordArena<
            'a,
            VecHeapLayout<F, 2, MODULAR_BLOCKS_48, MODULAR_BLOCKS_48>,
            VecHeapRecordMut<'a, 2, MODULAR_BLOCKS_48, MODULAR_BLOCKS_48>,
        > + for<'a> RecordArena<'a, EmptyAdapterCoreLayout<F, IsEq32Adapter>, IsEq32RecordMut<'a>>
        + for<'a> RecordArena<'a, EmptyAdapterCoreLayout<F, IsEq48Adapter>, IsEq48RecordMut<'a>>
{
}

/// Record-arena requirements contributed by the Fp2 extension.
///
/// Each record holds two base-field coordinates, so the 32-byte and 48-byte
/// base-field configurations use 8 and 12 VecHeap blocks respectively.
pub trait Fp2RecordArena<F>:
    Arena
    + for<'a> RecordArena<
        'a,
        VecHeapLayout<F, 2, FP2_BLOCKS_32, FP2_BLOCKS_32>,
        VecHeapRecordMut<'a, 2, FP2_BLOCKS_32, FP2_BLOCKS_32>,
    > + for<'a> RecordArena<
        'a,
        VecHeapLayout<F, 2, FP2_BLOCKS_48, FP2_BLOCKS_48>,
        VecHeapRecordMut<'a, 2, FP2_BLOCKS_48, FP2_BLOCKS_48>,
    >
{
}

impl<F, RA> Fp2RecordArena<F> for RA where
    RA: Arena
        + for<'a> RecordArena<
            'a,
            VecHeapLayout<F, 2, FP2_BLOCKS_32, FP2_BLOCKS_32>,
            VecHeapRecordMut<'a, 2, FP2_BLOCKS_32, FP2_BLOCKS_32>,
        > + for<'a> RecordArena<
            'a,
            VecHeapLayout<F, 2, FP2_BLOCKS_48, FP2_BLOCKS_48>,
            VecHeapRecordMut<'a, 2, FP2_BLOCKS_48, FP2_BLOCKS_48>,
        >
{
}

impl<F, RA> VmRvrLogNativeExtension<F, RA> for ModularExtension
where
    F: PrimeField32,
    RA: ModularRecordArena<F>,
{
    fn extend_rvr_log_native(&self, registry: &mut LogNativeAssemblerRegistry<F, RA>) {
        for (mod_idx, modulus) in self.supported_moduli.iter().enumerate() {
            let offset = Rv64ModularArithmeticOpcode::CLASS_OFFSET
                + mod_idx * Rv64ModularArithmeticOpcode::COUNT;
            let bytes = modulus.bits().div_ceil(8) as usize;
            if bytes <= NUM_LIMBS_32 {
                register_modulus::<F, RA, MODULAR_BLOCKS_32, NUM_LIMBS_32_U16>(registry, offset);
            } else if bytes <= NUM_LIMBS_48 {
                register_modulus::<F, RA, MODULAR_BLOCKS_48, NUM_LIMBS_48_U16>(registry, offset);
            } else {
                panic!("modulus exceeds maximum supported size of 384 bits");
            }
        }
    }
}

impl<F, RA> VmRvrLogNativeExtension<F, RA> for Fp2Extension
where
    F: PrimeField32,
    RA: Fp2RecordArena<F>,
{
    fn extend_rvr_log_native(&self, registry: &mut LogNativeAssemblerRegistry<F, RA>) {
        for (mod_idx, (_, modulus)) in self.supported_moduli.iter().enumerate() {
            let offset = Fp2Opcode::CLASS_OFFSET + mod_idx * Fp2Opcode::COUNT;
            let bytes = modulus.bits().div_ceil(8) as usize;
            if bytes <= NUM_LIMBS_32 {
                register_fp2::<F, RA, FP2_BLOCKS_32>(registry, offset);
            } else if bytes <= NUM_LIMBS_48 {
                register_fp2::<F, RA, FP2_BLOCKS_48>(registry, offset);
            } else {
                panic!("Fp2 modulus exceeds maximum supported size of 384 bits");
            }
        }
    }
}

impl<F, RA> VmRvrLogNativeExtension<F, RA> for Rv64ModularConfig
where
    F: PrimeField32,
    RA: Rv64IRecordArena<F> + Rv64MRecordArena<F> + Rv64IoRecordArena<F> + ModularRecordArena<F>,
{
    fn extend_rvr_log_native(&self, registry: &mut LogNativeAssemblerRegistry<F, RA>) {
        self.base.extend_rvr_log_native(registry);
        self.mul.extend_rvr_log_native(registry);
        self.io.extend_rvr_log_native(registry);
        self.modular.extend_rvr_log_native(registry);
    }
}

impl<F, RA> VmRvrLogNativeExtension<F, RA> for Rv64ModularWithFp2Config
where
    F: PrimeField32,
    RA: Rv64IRecordArena<F>
        + Rv64MRecordArena<F>
        + Rv64IoRecordArena<F>
        + ModularRecordArena<F>
        + Fp2RecordArena<F>,
{
    fn extend_rvr_log_native(&self, registry: &mut LogNativeAssemblerRegistry<F, RA>) {
        self.modular.extend_rvr_log_native(registry);
        self.fp2.extend_rvr_log_native(registry);
    }
}

fn register_modulus<F: PrimeField32, RA, const BLOCKS: usize, const U16_LIMBS: usize>(
    registry: &mut LogNativeAssemblerRegistry<F, RA>,
    offset: usize,
) where
    RA: ModularRecordArena<F>,
    for<'a> RA: RecordArena<
            'a,
            VecHeapLayout<F, 2, BLOCKS, BLOCKS>,
            VecHeapRecordMut<'a, 2, BLOCKS, BLOCKS>,
        > + RecordArena<
            'a,
            EmptyAdapterCoreLayout<F, Rv64IsEqualModU16AdapterExecutor<2, BLOCKS, U16_LIMBS>>,
            (
                &'a mut Rv64IsEqualModU16AdapterRecord<2, BLOCKS>,
                &'a mut ModularIsEqualRecord<U16_LIMBS>,
            ),
        >,
{
    let addsub_opcodes = [
        Rv64ModularArithmeticOpcode::ADD,
        Rv64ModularArithmeticOpcode::SUB,
        Rv64ModularArithmeticOpcode::SETUP_ADDSUB,
    ]
    .map(|opcode| modular_opcode(offset, opcode));
    registry.register_if(
        addsub_opcodes,
        is_modular_instruction,
        assemble_addsub::<F, RA, BLOCKS>,
    );
    let vec_heap = vec_heap_geometry::<F, 2, BLOCKS>();
    registry.register_inline_arena_native(
        addsub_opcodes,
        VecHeapRecordDescriptor::new(BLOCKS * MEMORY_BLOCK_BYTES).record_size,
        assemble_vec_heap_inline::<F, RA, 2, BLOCKS>,
        vec_heap,
    );
    let muldiv_opcodes = [
        Rv64ModularArithmeticOpcode::MUL,
        Rv64ModularArithmeticOpcode::DIV,
        Rv64ModularArithmeticOpcode::SETUP_MULDIV,
    ]
    .map(|opcode| modular_opcode(offset, opcode));
    registry.register_if(
        muldiv_opcodes,
        is_modular_instruction,
        assemble_muldiv::<F, RA, BLOCKS>,
    );
    registry.register_inline_arena_native(
        muldiv_opcodes,
        VecHeapRecordDescriptor::new(BLOCKS * MEMORY_BLOCK_BYTES).record_size,
        assemble_vec_heap_inline::<F, RA, 2, BLOCKS>,
        vec_heap,
    );
    let iseq_opcodes = [
        Rv64ModularArithmeticOpcode::IS_EQ,
        Rv64ModularArithmeticOpcode::SETUP_ISEQ,
    ]
    .map(|opcode| modular_opcode(offset, opcode));
    registry.register_if(
        iseq_opcodes,
        is_modular_is_eq_instruction,
        assemble_is_eq::<F, RA, BLOCKS, U16_LIMBS>,
    );
    let iseq = mod_iseq_geometry::<F, BLOCKS, U16_LIMBS>();
    registry.register_inline_arena_native(
        iseq_opcodes,
        ModIsEqRecordDescriptor::new(BLOCKS * MEMORY_BLOCK_BYTES).record_size,
        assemble_mod_iseq_inline::<F, RA, BLOCKS, U16_LIMBS>,
        iseq,
    );
}

pub fn vec_heap_geometry<F: PrimeField32, const NUM_READS: usize, const BLOCKS: usize>(
) -> ArenaNativeGeometry {
    type Adapter<const N: usize, const B: usize> = Rv64VecHeapAdapterRecord<N, B, B>;
    let descriptor =
        VecHeapRecordDescriptor::new_with_reads(BLOCKS * MEMORY_BLOCK_BYTES, NUM_READS);
    assert_eq!(
        size_of::<Adapter<NUM_READS, BLOCKS>>(),
        descriptor.adapter_size
    );
    assert_eq!(
        align_of::<Adapter<NUM_READS, BLOCKS>>(),
        descriptor.adapter_align
    );
    assert_eq!(
        offset_of!(Adapter<NUM_READS, BLOCKS>, reads_aux),
        descriptor.reads_aux
    );
    assert_eq!(
        offset_of!(Adapter<NUM_READS, BLOCKS>, writes_aux),
        descriptor.writes_aux
    );
    ArenaNativeGeometry {
        adapter_size: descriptor.adapter_size,
        adapter_align: descriptor.adapter_align,
        core_size: descriptor.core_size,
        core_align: descriptor.core_align,
        core_off_matrix: <Rv64VecHeapAdapterExecutor<NUM_READS, BLOCKS, BLOCKS> as AdapterTraceExecutor<F>>::WIDTH
            * size_of::<F>(),
        layout: ArenaNativeLayout::Custom {
            residual_memory_chronology: true,
        },
    }
}

fn mod_iseq_geometry<F: PrimeField32, const BLOCKS: usize, const U16_LIMBS: usize>(
) -> ArenaNativeGeometry {
    type Adapter<const B: usize> = Rv64IsEqualModU16AdapterRecord<2, B>;
    type Core<const L: usize> = ModularIsEqualRecord<L>;
    let descriptor = ModIsEqRecordDescriptor::new(BLOCKS * MEMORY_BLOCK_BYTES);
    assert_eq!(U16_LIMBS, descriptor.u16_limbs);
    assert_eq!(size_of::<Adapter<BLOCKS>>(), descriptor.adapter_size);
    assert_eq!(align_of::<Adapter<BLOCKS>>(), descriptor.adapter_align);
    assert_eq!(size_of::<Core<U16_LIMBS>>(), descriptor.core_size);
    assert_eq!(align_of::<Core<U16_LIMBS>>(), descriptor.core_align);
    assert_eq!(
        offset_of!(Adapter<BLOCKS>, heap_read_aux),
        descriptor.heap_read_aux
    );
    assert_eq!(offset_of!(Adapter<BLOCKS>, rd_ptr), descriptor.rd_ptr);
    assert_eq!(
        offset_of!(Adapter<BLOCKS>, writes_aux),
        descriptor.writes_aux
    );
    ArenaNativeGeometry {
        adapter_size: descriptor.adapter_size,
        adapter_align: descriptor.adapter_align,
        core_size: descriptor.core_size,
        core_align: descriptor.core_align,
        core_off_matrix: <Rv64IsEqualModU16AdapterExecutor<2, BLOCKS, U16_LIMBS> as AdapterTraceExecutor<F>>::WIDTH
            * size_of::<F>(),
        layout: ArenaNativeLayout::Custom {
            residual_memory_chronology: true,
        },
    }
}

fn register_fp2<F: PrimeField32, RA, const BLOCKS: usize>(
    registry: &mut LogNativeAssemblerRegistry<F, RA>,
    offset: usize,
) where
    RA: Fp2RecordArena<F>,
    for<'a> RA: RecordArena<
        'a,
        VecHeapLayout<F, 2, BLOCKS, BLOCKS>,
        VecHeapRecordMut<'a, 2, BLOCKS, BLOCKS>,
    >,
{
    let addsub_opcodes = [Fp2Opcode::ADD, Fp2Opcode::SUB, Fp2Opcode::SETUP_ADDSUB]
        .map(|opcode| fp2_opcode(offset, opcode));
    registry.register_if(
        addsub_opcodes,
        is_vec_heap_instruction,
        assemble_fp2_addsub::<F, RA, BLOCKS>,
    );
    let descriptor = VecHeapRecordDescriptor::new(BLOCKS * MEMORY_BLOCK_BYTES);
    let geometry = vec_heap_geometry::<F, 2, BLOCKS>();
    registry.register_inline_arena_native(
        addsub_opcodes,
        descriptor.record_size,
        assemble_vec_heap_inline::<F, RA, 2, BLOCKS>,
        geometry,
    );
    let muldiv_opcodes = [Fp2Opcode::MUL, Fp2Opcode::DIV, Fp2Opcode::SETUP_MULDIV]
        .map(|opcode| fp2_opcode(offset, opcode));
    registry.register_if(
        muldiv_opcodes,
        is_vec_heap_instruction,
        assemble_fp2_muldiv::<F, RA, BLOCKS>,
    );
    registry.register_inline_arena_native(
        muldiv_opcodes,
        descriptor.record_size,
        assemble_vec_heap_inline::<F, RA, 2, BLOCKS>,
        geometry,
    );
}

fn modular_opcode(offset: usize, opcode: Rv64ModularArithmeticOpcode) -> VmOpcode {
    VmOpcode::from_usize(offset + opcode as usize)
}

fn fp2_opcode(offset: usize, opcode: Fp2Opcode) -> VmOpcode {
    VmOpcode::from_usize(offset + opcode as usize)
}

fn is_modular_instruction<F: PrimeField32>(instruction: &Instruction<F>) -> bool {
    is_vec_heap_instruction(instruction)
}

fn is_vec_heap_instruction<F: PrimeField32>(instruction: &Instruction<F>) -> bool {
    instruction.d.as_canonical_u32() == RV64_REGISTER_AS
        && instruction.e.as_canonical_u32() == RV64_MEMORY_AS
}

fn is_modular_is_eq_instruction<F: PrimeField32>(instruction: &Instruction<F>) -> bool {
    is_modular_instruction(instruction) && !instruction.a.is_zero()
}

pub fn assemble_vec_heap_inline<F: PrimeField32, RA, const NUM_READS: usize, const BLOCKS: usize>(
    arena: &mut RA,
    _instruction: &Instruction<F>,
    compact: &[u8],
    pc: u32,
) -> Result<(), ExecutionError>
where
    RA: Arena
        + for<'a> RecordArena<
            'a,
            VecHeapLayout<F, NUM_READS, BLOCKS, BLOCKS>,
            VecHeapRecordMut<'a, NUM_READS, BLOCKS, BLOCKS>,
        >,
{
    let descriptor =
        VecHeapRecordDescriptor::new_with_reads(BLOCKS * MEMORY_BLOCK_BYTES, NUM_READS);
    if compact.len() != descriptor.record_size {
        return Err(rvr_error(format!(
            "invalid VecHeap inline record size {} at pc {pc:#x}; expected {}",
            compact.len(),
            descriptor.record_size
        )));
    }
    let layout =
        AdapterCoreLayout::with_metadata(FieldExpressionMetadata::<
            F,
            Rv64VecHeapAdapterExecutor<NUM_READS, BLOCKS, BLOCKS>,
        >::new(NUM_READS * BLOCKS * MEMORY_BLOCK_BYTES));
    let (adapter, core): (
        &mut Rv64VecHeapAdapterRecord<NUM_READS, BLOCKS, BLOCKS>,
        FieldExpressionCoreRecordMut<'_>,
    ) = arena.alloc(layout);
    unsafe {
        std::ptr::copy_nonoverlapping(
            compact.as_ptr(),
            (adapter as *mut Rv64VecHeapAdapterRecord<NUM_READS, BLOCKS, BLOCKS>).cast::<u8>(),
            descriptor.adapter_size,
        );
        std::ptr::copy_nonoverlapping(
            compact.as_ptr().add(descriptor.core_off_dense),
            (core.opcode as *mut u8).cast::<u8>(),
            descriptor.core_size,
        );
    }
    if adapter.from_pc != pc {
        return Err(rvr_error(format!(
            "VecHeap inline record pc mismatch: record={:#x}, program={pc:#x}",
            adapter.from_pc
        )));
    }
    Ok(())
}

fn assemble_mod_iseq_inline<F: PrimeField32, RA, const BLOCKS: usize, const U16_LIMBS: usize>(
    arena: &mut RA,
    _instruction: &Instruction<F>,
    compact: &[u8],
    pc: u32,
) -> Result<(), ExecutionError>
where
    RA: Arena
        + for<'a> RecordArena<
            'a,
            EmptyAdapterCoreLayout<F, Rv64IsEqualModU16AdapterExecutor<2, BLOCKS, U16_LIMBS>>,
            (
                &'a mut Rv64IsEqualModU16AdapterRecord<2, BLOCKS>,
                &'a mut ModularIsEqualRecord<U16_LIMBS>,
            ),
        >,
{
    let descriptor = ModIsEqRecordDescriptor::new(BLOCKS * MEMORY_BLOCK_BYTES);
    if compact.len() != descriptor.record_size {
        return Err(rvr_error(format!(
            "invalid modular IS_EQ inline record size {} at pc {pc:#x}; expected {}",
            compact.len(),
            descriptor.record_size
        )));
    }
    let (adapter, core): (
        &mut Rv64IsEqualModU16AdapterRecord<2, BLOCKS>,
        &mut ModularIsEqualRecord<U16_LIMBS>,
    ) = arena.alloc(EmptyAdapterCoreLayout::<
        F,
        Rv64IsEqualModU16AdapterExecutor<2, BLOCKS, U16_LIMBS>,
    >::new());
    unsafe {
        std::ptr::copy_nonoverlapping(
            compact.as_ptr(),
            (adapter as *mut Rv64IsEqualModU16AdapterRecord<2, BLOCKS>).cast::<u8>(),
            descriptor.adapter_size,
        );
        std::ptr::copy_nonoverlapping(
            compact.as_ptr().add(descriptor.core_off_dense),
            (core as *mut ModularIsEqualRecord<U16_LIMBS>).cast::<u8>(),
            descriptor.core_size,
        );
    }
    if adapter.from_pc != pc {
        return Err(rvr_error(format!(
            "modular IS_EQ inline record pc mismatch: record={:#x}, program={pc:#x}",
            adapter.from_pc
        )));
    }
    Ok(())
}

fn assemble_addsub<F: PrimeField32, RA, const BLOCKS: usize>(
    arena: &mut RA,
    access: &LogNativeAccessView<'_, F>,
    instruction: &Instruction<F>,
    pc: u32,
    timestamp: u32,
) -> Result<(), ExecutionError>
where
    RA: Arena
        + for<'a> RecordArena<
            'a,
            VecHeapLayout<F, 2, BLOCKS, BLOCKS>,
            VecHeapRecordMut<'a, 2, BLOCKS, BLOCKS>,
        >,
{
    assemble_rv64_vec_heap_field_expression::<F, RA, 2, BLOCKS, BLOCKS>(
        arena,
        access,
        instruction,
        pc,
        timestamp,
        modular_local_opcode(instruction),
        modular_heap_read_kinds(instruction),
    )
}

fn assemble_muldiv<F: PrimeField32, RA, const BLOCKS: usize>(
    arena: &mut RA,
    access: &LogNativeAccessView<'_, F>,
    instruction: &Instruction<F>,
    pc: u32,
    timestamp: u32,
) -> Result<(), ExecutionError>
where
    RA: Arena
        + for<'a> RecordArena<
            'a,
            VecHeapLayout<F, 2, BLOCKS, BLOCKS>,
            VecHeapRecordMut<'a, 2, BLOCKS, BLOCKS>,
        >,
{
    assemble_rv64_vec_heap_field_expression::<F, RA, 2, BLOCKS, BLOCKS>(
        arena,
        access,
        instruction,
        pc,
        timestamp,
        modular_local_opcode(instruction),
        modular_heap_read_kinds(instruction),
    )
}

fn assemble_fp2_addsub<F: PrimeField32, RA, const BLOCKS: usize>(
    arena: &mut RA,
    access: &LogNativeAccessView<'_, F>,
    instruction: &Instruction<F>,
    pc: u32,
    timestamp: u32,
) -> Result<(), ExecutionError>
where
    RA: Arena
        + for<'a> RecordArena<
            'a,
            VecHeapLayout<F, 2, BLOCKS, BLOCKS>,
            VecHeapRecordMut<'a, 2, BLOCKS, BLOCKS>,
        >,
{
    assemble_rv64_vec_heap_field_expression::<F, RA, 2, BLOCKS, BLOCKS>(
        arena,
        access,
        instruction,
        pc,
        timestamp,
        fp2_local_opcode(instruction),
        fp2_heap_read_kinds(instruction),
    )
}

fn assemble_fp2_muldiv<F: PrimeField32, RA, const BLOCKS: usize>(
    arena: &mut RA,
    access: &LogNativeAccessView<'_, F>,
    instruction: &Instruction<F>,
    pc: u32,
    timestamp: u32,
) -> Result<(), ExecutionError>
where
    RA: Arena
        + for<'a> RecordArena<
            'a,
            VecHeapLayout<F, 2, BLOCKS, BLOCKS>,
            VecHeapRecordMut<'a, 2, BLOCKS, BLOCKS>,
        >,
{
    assemble_rv64_vec_heap_field_expression::<F, RA, 2, BLOCKS, BLOCKS>(
        arena,
        access,
        instruction,
        pc,
        timestamp,
        fp2_local_opcode(instruction),
        fp2_heap_read_kinds(instruction),
    )
}

fn assemble_is_eq<F: PrimeField32, RA, const BLOCKS: usize, const U16_LIMBS: usize>(
    arena: &mut RA,
    access: &LogNativeAccessView<'_, F>,
    instruction: &Instruction<F>,
    pc: u32,
    timestamp: u32,
) -> Result<(), ExecutionError>
where
    RA: Arena
        + for<'a> RecordArena<
            'a,
            EmptyAdapterCoreLayout<F, Rv64IsEqualModU16AdapterExecutor<2, BLOCKS, U16_LIMBS>>,
            (
                &'a mut Rv64IsEqualModU16AdapterRecord<2, BLOCKS>,
                &'a mut ModularIsEqualRecord<U16_LIMBS>,
            ),
        >,
{
    let (adapter_record, core_record): (
        &mut Rv64IsEqualModU16AdapterRecord<2, BLOCKS>,
        &mut ModularIsEqualRecord<U16_LIMBS>,
    ) = arena.alloc(EmptyAdapterCoreLayout::<
        F,
        Rv64IsEqualModU16AdapterExecutor<2, BLOCKS, U16_LIMBS>,
    >::new());

    adapter_record.from_pc = pc;
    adapter_record.timestamp = timestamp;
    let mut next_timestamp = timestamp;
    for i in 0..2 {
        let ptr = if i == 0 {
            instruction.b.as_canonical_u32()
        } else {
            instruction.c.as_canonical_u32()
        };
        adapter_record.rs_ptr[i] = ptr;
        let aux = expect_access(
            access,
            next_timestamp,
            PREFLIGHT_MEMORY_KIND_READ,
            RV64_REGISTER_AS,
            ptr,
            RV64_REGISTER_NUM_LIMBS,
            pc,
        )?;
        adapter_record.rs_val[i] = read_low_u32(aux.entry.value);
        adapter_record.rs_read_aux[i] = MemoryReadAuxRecord {
            prev_timestamp: aux.prev_timestamp,
        };
        next_timestamp += 1;
    }

    let mut input = [[0u16; U16_LIMBS]; 2];
    for (read_idx, limbs) in input.iter_mut().enumerate() {
        let kind = if read_idx == 1
            && modular_local_opcode(instruction) == Rv64ModularArithmeticOpcode::SETUP_ISEQ as u8
        {
            PREFLIGHT_MEMORY_KIND_TOUCH
        } else {
            PREFLIGHT_MEMORY_KIND_READ
        };
        for block_idx in 0..BLOCKS {
            let address = adapter_record.rs_val[read_idx] + (block_idx * MEMORY_BLOCK_BYTES) as u32;
            let aux = expect_access(
                access,
                next_timestamp,
                kind,
                RV64_MEMORY_AS,
                address,
                MEMORY_BLOCK_BYTES,
                pc,
            )?;
            adapter_record.heap_read_aux[read_idx][block_idx] = MemoryReadAuxRecord {
                prev_timestamp: aux.prev_timestamp,
            };
            limbs[block_idx * BLOCK_FE_WIDTH..(block_idx + 1) * BLOCK_FE_WIDTH]
                .copy_from_slice(&prev_u16(aux));
            next_timestamp += 1;
        }
    }

    adapter_record.rd_ptr = instruction.a.as_canonical_u32();
    let write_aux = expect_access(
        access,
        next_timestamp,
        PREFLIGHT_MEMORY_KIND_WRITE,
        RV64_REGISTER_AS,
        adapter_record.rd_ptr,
        RV64_REGISTER_NUM_LIMBS,
        pc,
    )?;
    adapter_record.writes_aux = MemoryWriteU16AuxRecord {
        prev_timestamp: write_aux.prev_timestamp,
        prev_data: prev_u16(write_aux),
    };

    core_record.is_setup =
        modular_local_opcode(instruction) == Rv64ModularArithmeticOpcode::SETUP_ISEQ as u8;
    [core_record.b, core_record.c] = input;
    Ok(())
}

/// Allocate and fill a FieldExpr record backed by an `Rv64VecHeapAdapter`.
///
/// This is public so ECC log-native assemblers can reuse the exact same
/// register/heap timestamp model. `heap_read_kinds` permits setup operations
/// to use read-equivalent preflight `TOUCH` entries without filtering logs.
#[allow(clippy::too_many_arguments)]
pub fn assemble_rv64_vec_heap_field_expression<
    F: PrimeField32,
    RA,
    const NUM_READS: usize,
    const READ_BLOCKS: usize,
    const WRITE_BLOCKS: usize,
>(
    arena: &mut RA,
    access: &LogNativeAccessView<'_, F>,
    instruction: &Instruction<F>,
    pc: u32,
    timestamp: u32,
    local_opcode: u8,
    heap_read_kinds: [u8; NUM_READS],
) -> Result<(), ExecutionError>
where
    RA: Arena
        + for<'a> RecordArena<
            'a,
            VecHeapLayout<F, NUM_READS, READ_BLOCKS, WRITE_BLOCKS>,
            VecHeapRecordMut<'a, NUM_READS, READ_BLOCKS, WRITE_BLOCKS>,
        >,
{
    let total_input_limbs = NUM_READS * READ_BLOCKS * MEMORY_BLOCK_BYTES;
    let layout = AdapterCoreLayout::with_metadata(FieldExpressionMetadata::<
        F,
        Rv64VecHeapAdapterExecutor<NUM_READS, READ_BLOCKS, WRITE_BLOCKS>,
    >::new(total_input_limbs));
    let (adapter_record, core_record): (
        &mut Rv64VecHeapAdapterRecord<NUM_READS, READ_BLOCKS, WRITE_BLOCKS>,
        FieldExpressionCoreRecordMut<'_>,
    ) = arena.alloc(layout);
    let read_data = fill_rv64_vec_heap_adapter(
        access,
        instruction,
        pc,
        timestamp,
        heap_read_kinds,
        adapter_record,
    )?;

    *core_record.opcode = local_opcode;
    let mut cursor = 0;
    for read in read_data {
        for block in read {
            core_record.input_limbs[cursor..cursor + MEMORY_BLOCK_BYTES].copy_from_slice(&block);
            cursor += MEMORY_BLOCK_BYTES;
        }
    }
    Ok(())
}

/// Fill the reusable `Rv64VecHeapAdapter` record from normalized preflight logs.
#[allow(clippy::too_many_arguments)]
pub fn fill_rv64_vec_heap_adapter<
    F: PrimeField32,
    const NUM_READS: usize,
    const READ_BLOCKS: usize,
    const WRITE_BLOCKS: usize,
>(
    access: &LogNativeAccessView<'_, F>,
    instruction: &Instruction<F>,
    pc: u32,
    timestamp: u32,
    heap_read_kinds: [u8; NUM_READS],
    record: &mut Rv64VecHeapAdapterRecord<NUM_READS, READ_BLOCKS, WRITE_BLOCKS>,
) -> Result<[[[u8; MEMORY_BLOCK_BYTES]; READ_BLOCKS]; NUM_READS], ExecutionError> {
    record.from_pc = pc;
    record.from_timestamp = timestamp;
    let mut next_timestamp = timestamp;

    for i in 0..NUM_READS {
        let ptr = if i == 0 {
            instruction.b.as_canonical_u32()
        } else {
            instruction.c.as_canonical_u32()
        };
        record.rs_ptrs[i] = ptr;
        let aux = expect_access(
            access,
            next_timestamp,
            PREFLIGHT_MEMORY_KIND_READ,
            RV64_REGISTER_AS,
            ptr,
            RV64_REGISTER_NUM_LIMBS,
            pc,
        )?;
        record.rs_vals[i] = read_low_u32(aux.entry.value);
        record.rs_read_aux[i] = MemoryReadAuxRecord {
            prev_timestamp: aux.prev_timestamp,
        };
        next_timestamp += 1;
    }

    record.rd_ptr = instruction.a.as_canonical_u32();
    let rd_aux = expect_access(
        access,
        next_timestamp,
        PREFLIGHT_MEMORY_KIND_READ,
        RV64_REGISTER_AS,
        record.rd_ptr,
        RV64_REGISTER_NUM_LIMBS,
        pc,
    )?;
    record.rd_val = read_low_u32(rd_aux.entry.value);
    record.rd_read_aux = MemoryReadAuxRecord {
        prev_timestamp: rd_aux.prev_timestamp,
    };
    next_timestamp += 1;

    let mut read_data = [[[0u8; MEMORY_BLOCK_BYTES]; READ_BLOCKS]; NUM_READS];
    for (read_idx, blocks) in read_data.iter_mut().enumerate() {
        for (block_idx, block) in blocks.iter_mut().enumerate() {
            let address = record.rs_vals[read_idx] + (block_idx * MEMORY_BLOCK_BYTES) as u32;
            let aux = expect_access(
                access,
                next_timestamp,
                heap_read_kinds[read_idx],
                RV64_MEMORY_AS,
                address,
                MEMORY_BLOCK_BYTES,
                pc,
            )?;
            record.reads_aux[read_idx][block_idx] = MemoryReadAuxRecord {
                prev_timestamp: aux.prev_timestamp,
            };
            *block = prev_bytes(aux);
            next_timestamp += 1;
        }
    }

    for block_idx in 0..WRITE_BLOCKS {
        let address = record.rd_val + (block_idx * MEMORY_BLOCK_BYTES) as u32;
        let aux = expect_access(
            access,
            next_timestamp,
            PREFLIGHT_MEMORY_KIND_WRITE,
            RV64_MEMORY_AS,
            address,
            MEMORY_BLOCK_BYTES,
            pc,
        )?;
        record.writes_aux[block_idx] = MemoryWriteBytesAuxRecord {
            prev_timestamp: aux.prev_timestamp,
            prev_data: prev_bytes(aux),
        };
        next_timestamp += 1;
    }

    Ok(read_data)
}

fn modular_local_opcode<F: PrimeField32>(instruction: &Instruction<F>) -> u8 {
    (instruction
        .opcode
        .as_usize()
        .wrapping_sub(Rv64ModularArithmeticOpcode::CLASS_OFFSET)
        % Rv64ModularArithmeticOpcode::COUNT) as u8
}

fn modular_heap_read_kinds<F: PrimeField32>(instruction: &Instruction<F>) -> [u8; 2] {
    let local = modular_local_opcode(instruction) as usize;
    let second = if matches!(
        local,
        x if x == Rv64ModularArithmeticOpcode::SETUP_ADDSUB as usize
            || x == Rv64ModularArithmeticOpcode::SETUP_MULDIV as usize
    ) {
        PREFLIGHT_MEMORY_KIND_TOUCH
    } else {
        PREFLIGHT_MEMORY_KIND_READ
    };
    [PREFLIGHT_MEMORY_KIND_READ, second]
}

fn fp2_local_opcode<F: PrimeField32>(instruction: &Instruction<F>) -> u8 {
    (instruction
        .opcode
        .as_usize()
        .wrapping_sub(Fp2Opcode::CLASS_OFFSET)
        % Fp2Opcode::COUNT) as u8
}

fn fp2_heap_read_kinds<F: PrimeField32>(instruction: &Instruction<F>) -> [u8; 2] {
    let local = fp2_local_opcode(instruction) as usize;
    let second = if matches!(
        local,
        x if x == Fp2Opcode::SETUP_ADDSUB as usize
            || x == Fp2Opcode::SETUP_MULDIV as usize
    ) {
        PREFLIGHT_MEMORY_KIND_TOUCH
    } else {
        PREFLIGHT_MEMORY_KIND_READ
    };
    [PREFLIGHT_MEMORY_KIND_READ, second]
}

#[allow(clippy::too_many_arguments)]
fn expect_access<'a, F: PrimeField32>(
    access: &LogNativeAccessView<'a, F>,
    timestamp: u32,
    kind: u8,
    addr_space: u32,
    address: u32,
    width: usize,
    pc: u32,
) -> Result<&'a PreflightMemoryAccessAux<F>, ExecutionError> {
    access.expect(timestamp, kind, addr_space, u64::from(address), width, pc)
}

fn read_low_u32(value: u64) -> u32 {
    value as u32
}

fn prev_u16<F: PrimeField32>(aux: &PreflightMemoryAccessAux<F>) -> [u16; BLOCK_FE_WIDTH] {
    aux.prev_data
        .map(|cell| cell.as_canonical_u32().try_into().expect("u16 memory cell"))
}

fn prev_bytes<F: PrimeField32>(aux: &PreflightMemoryAccessAux<F>) -> [u8; MEMORY_BLOCK_BYTES] {
    let limbs = prev_u16(aux);
    let mut bytes = [0; MEMORY_BLOCK_BYTES];
    for (chunk, limb) in bytes.chunks_exact_mut(2).zip(limbs) {
        chunk.copy_from_slice(&limb.to_le_bytes());
    }
    bytes
}

fn rvr_error(message: String) -> ExecutionError {
    ExecutionError::RvrExecution(message)
}
