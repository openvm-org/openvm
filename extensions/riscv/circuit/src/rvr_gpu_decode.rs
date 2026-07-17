//! M-GPUDEC (G2): per-exe device operand table + per-segment emission-mode
//! state, shared between the GPU builder (producer side: binds the exe, builds
//! the table, selects per-air modes each segment) and the migrated GPU chips
//! (consumer side: read the mode, upload the table once, launch the compact
//! decode kernels).
//!
//! One-derivation-two-consumers rule: table entries are produced by the same
//! `derive_*_operands` helpers the host inline assemblers use, so the CUDA
//! decoder is the only new derivation surface and the three-way differential
//! pins it.

#[cfg(feature = "rvr")]
use std::collections::{HashMap, HashSet};
use std::{
    mem::{align_of, size_of},
    sync::{Arc, Mutex},
};

#[cfg(all(feature = "cuda", feature = "rvr"))]
use openvm_circuit::arch::rvr::gpu_profile::CudaStageTimer;
#[cfg(feature = "rvr")]
use openvm_circuit::arch::rvr::{
    RvrDeltaDecodeEntry, RvrDeltaDecodeInfo, RvrDeltaDecodePrecompute,
};
#[cfg(all(feature = "cuda", feature = "rvr"))]
use openvm_circuit::system::cuda::memory::{
    DeviceInitialMemory, DeviceTouchedMemory, DeviceTouchedMemoryProvider,
    DEVICE_TOUCHED_RECORD_WORDS,
};
#[cfg(all(feature = "cuda", feature = "rvr"))]
use openvm_circuit::system::cuda::program::{
    DeviceProgramFrequencies, DeviceProgramFrequenciesProvider,
};
#[cfg(feature = "cuda")]
use openvm_cuda_common::{
    copy::{cuda_memcpy_on, MemCopyD2H, MemCopyH2D},
    d_buffer::DeviceBuffer,
    stream::GpuDeviceCtx,
};
#[cfg(feature = "rvr")]
use openvm_instructions::{
    exe::VmExe, instruction::Instruction, program::DEFAULT_PC_STEP, LocalOpcode,
};
#[cfg(feature = "rvr")]
use openvm_riscv_transpiler::{
    BaseAluImmOpcode, BaseAluOpcode, BaseAluWOpcode, BranchEqualOpcode, BranchLessThanOpcode,
    DivRemOpcode, DivRemWOpcode, LessThanOpcode, MulHOpcode, MulOpcode, MulWOpcode,
    Rv64AuipcOpcode, Rv64JalLuiOpcode, Rv64JalrOpcode, Rv64LoadStoreOpcode, ShiftOpcode,
    ShiftWOpcode,
};
#[cfg(feature = "rvr")]
use openvm_stark_backend::p3_field::PrimeField32;

#[cfg(feature = "rvr")]
use crate::log_native::{derive_addi_operands, derive_base_alu_u16_operands};

/// One 20-byte operand-table entry per program slot, indexed by
/// `(from_pc - pc_base) / DEFAULT_PC_STEP`. Field meanings are per wire
/// format; for alu3 over the BaseAluU16 adapter: `a` = rd_ptr, `b` = rs1_ptr,
/// `c` = rs2 (register ptr or immediate), flags carry rs2_as/imm-sign, and
/// `local_opcode` is the class-local opcode index for the owning AIR.
/// Mirrored by the CUDA `RvrOperandEntry` (static-asserted there).
pub type DeviceOperandEntry = RvrDeltaDecodeEntry;

const _: () = assert!(size_of::<DeviceOperandEntry>() == 20);
const _: () = assert!(align_of::<DeviceOperandEntry>() == 4);
const RVR_MULTI_BLOCK_RECORD_SIZE: usize = 60;
const _: () = assert!(
    size_of::<(
        crate::adapters::Rv64LoadMultiByteAdapterRecord,
        crate::load::LoadRecord,
    )>() == RVR_MULTI_BLOCK_RECORD_SIZE
);
const _: () = assert!(
    size_of::<(
        crate::adapters::Rv64StoreMultiByteAdapterRecord,
        crate::store::StoreRecord,
    )>() == RVR_MULTI_BLOCK_RECORD_SIZE
);

/// `rs2` is an immediate (else a register pointer).
pub const OPERAND_FLAG_RS2_IMM: u8 = 1 << 0;
/// The sign-extension bit for a 16-bit-signed immediate (`rs2_imm_sign`).
pub const OPERAND_FLAG_RS2_IMM_SIGN: u8 = 1 << 1;
/// wr1/rw1: the conditional rd write is enabled (`f != 0`; x0 suppression).
pub const OPERAND_FLAG_WRITE_ENABLED: u8 = 1 << 2;
/// wr1 (JalLui): the opcode is JAL (else LUI).
pub const OPERAND_FLAG_IS_JAL: u8 = 1 << 3;
/// rw1 (Jalr): the immediate sign bit (`g != 0`).
pub const OPERAND_FLAG_JALR_IMM_SIGN: u8 = 1 << 4;
/// loadstore: target public values (REVEAL) instead of main memory. Bit 4 is
/// format-local and does not overlap another loadstore flag.
pub const OPERAND_FLAG_LS_PUBLIC_VALUES: u8 = 1 << 4;
/// loadstore: the 16-bit immediate sign bit (`g != 0`).
pub const OPERAND_FLAG_LS_IMM_SIGN: u8 = 1 << 5;
/// loadstore: zero-extension load (vs store); per-format reuse of bit 3.
pub const OPERAND_FLAG_LS_IS_LOAD: u8 = 1 << 3;
/// loadstore (sign-extending): LOADB / LOADW selectors.
pub const OPERAND_FLAG_LS_IS_BYTE: u8 = 1 << 6;
pub const OPERAND_FLAG_LS_IS_WORD: u8 = 1 << 7;

/// Device-visible access pattern. Values are mirrored in `rvr_delta.cu` and
/// deliberately match the CPU oracle's `DeltaAccessPattern` semantics.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceDeltaAccessPattern {
    Alu3 = 0,
    Alu3Reg = 1,
    Load = 2,
    Store = 3,
    Branch2 = 4,
    Wr1 = 5,
    Wr1Always = 6,
    Rw1 = 7,
    AddI = 8,
    /// G2-only one-row HintStore replay record. This never participates in
    /// the established 24-byte delta producer route.
    HintStore = 9,
    /// G2-only direct-final custom instruction. Its timestamp-bearing
    /// accesses are carried exclusively by the residual lanes.
    OpaqueFinal = 10,
    /// G2-only timestamp-only custom instruction (system PHANTOM).
    OpaqueTimestamp = 11,
}

/// One unique compact-decoder consumer. This is independent of the global AIR
/// index so the GPU chips can request their device buffer without changing AIR
/// or verifying-key construction.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DeltaAirKind {
    AddSub = 0,
    Bitwise = 1,
    LessThan = 2,
    ShiftLogical = 3,
    ShiftRightArithmetic = 4,
    AddSubW = 5,
    ShiftWLogical = 6,
    ShiftWRightArithmetic = 7,
    LoadByte = 8,
    LoadSignExtendByte = 9,
    BranchEqual = 10,
    BranchLessThan = 11,
    JalLui = 12,
    Jalr = 13,
    Auipc = 14,
    Mul = 15,
    MulH = 16,
    MulW = 17,
    DivRem = 18,
    DivRemW = 19,
    LoadHalfword = 20,
    LoadWord = 21,
    LoadDoubleword = 22,
    StoreByte = 23,
    StoreHalfword = 24,
    StoreWord = 25,
    StoreDoubleword = 26,
    LoadSignExtendHalfword = 27,
    LoadSignExtendWord = 28,
    AddI = 29,
    /// Private G2 HintStore event replay; there is deliberately no payload
    /// lane for this kind.
    HintStore = 30,
}

impl DeltaAirKind {
    pub const COUNT: usize = 31;

    fn from_repr(value: u8) -> Option<Self> {
        Some(match value {
            0 => Self::AddSub,
            1 => Self::Bitwise,
            2 => Self::LessThan,
            3 => Self::ShiftLogical,
            4 => Self::ShiftRightArithmetic,
            5 => Self::AddSubW,
            6 => Self::ShiftWLogical,
            7 => Self::ShiftWRightArithmetic,
            8 => Self::LoadByte,
            9 => Self::LoadSignExtendByte,
            10 => Self::BranchEqual,
            11 => Self::BranchLessThan,
            12 => Self::JalLui,
            13 => Self::Jalr,
            14 => Self::Auipc,
            15 => Self::Mul,
            16 => Self::MulH,
            17 => Self::MulW,
            18 => Self::DivRem,
            19 => Self::DivRemW,
            20 => Self::LoadHalfword,
            21 => Self::LoadWord,
            22 => Self::LoadDoubleword,
            23 => Self::StoreByte,
            24 => Self::StoreHalfword,
            25 => Self::StoreWord,
            26 => Self::StoreDoubleword,
            27 => Self::LoadSignExtendHalfword,
            28 => Self::LoadSignExtendWord,
            29 => Self::AddI,
            30 => Self::HintStore,
            _ => return None,
        })
    }

    const fn crossing_residual_capable(self) -> bool {
        matches!(
            self,
            Self::LoadHalfword
                | Self::LoadWord
                | Self::LoadDoubleword
                | Self::StoreHalfword
                | Self::StoreWord
                | Self::StoreDoubleword
                | Self::LoadSignExtendHalfword
                | Self::LoadSignExtendWord
        )
    }

    pub const fn wire_size(self) -> usize {
        use openvm_circuit::arch::rvr::{
            PREFLIGHT_ADDSUB_RECORD_SIZE, PREFLIGHT_BRANCH2_RECORD_SIZE, PREFLIGHT_RW1_RECORD_SIZE,
            PREFLIGHT_WR1_RECORD_SIZE,
        };
        match self {
            kind if kind.crossing_residual_capable() => RVR_MULTI_BLOCK_RECORD_SIZE,
            Self::BranchEqual | Self::BranchLessThan => PREFLIGHT_BRANCH2_RECORD_SIZE,
            Self::JalLui | Self::Auipc => PREFLIGHT_WR1_RECORD_SIZE,
            Self::Jalr => PREFLIGHT_RW1_RECORD_SIZE,
            Self::HintStore => 64,
            _ => PREFLIGHT_ADDSUB_RECORD_SIZE,
        }
    }
}

/// How a migrated AIR's records are fed to the GPU this segment (naming
/// shared with the R4-fuse work).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InlineEmissionMode {
    /// Compact wire records + on-device operand-table decode (G2).
    CompactWire,
    /// Full in-arena records written by C, fed zero-copy (G1; requires the
    /// R4 batch-2 arena-native emitters).
    ArenaNative,
    /// Global chronological 24-byte records. Previous timestamps and stable
    /// per-AIR compact buffers are reconstructed by the shared CUDA predecode.
    Delta,
    /// Program-run + opcode-arity private wire v1.
    G2,
}

/// Env toggle for GPU record transport. An explicit setting always wins;
/// CUDA builds default to G2, while non-CUDA builds retain the expanded host
/// record path.
pub fn configured_emission_mode() -> Option<InlineEmissionMode> {
    match std::env::var("OPENVM_RVR_GPU_RECORDS") {
        Ok(mode) => match mode.as_str() {
            "compact" => Some(InlineEmissionMode::CompactWire),
            "arena-native" => Some(InlineEmissionMode::ArenaNative),
            "delta" => Some(InlineEmissionMode::Delta),
            "g2" => Some(InlineEmissionMode::G2),
            _ => None,
        },
        Err(std::env::VarError::NotPresent) if cfg!(feature = "cuda") => {
            Some(InlineEmissionMode::G2)
        }
        Err(_) => None,
    }
}

struct HostOperandTable {
    /// The compiled preflight metadata is unique to one VmExe + route and is
    /// retained here, so its allocation cannot be recycled into a false cache
    /// hit. Pointer comparison stays constant-time on every segment.
    #[cfg_attr(not(feature = "rvr"), allow(dead_code))]
    compiled_identity: Arc<Vec<bool>>,
    pc_base: u32,
    entries: Arc<Vec<DeviceOperandEntry>>,
    /// Compiler-scope mixed-AIR classification. Building this walks the
    /// entire program, so cache it with the operand table instead of repeating
    /// the same immutable scan at both bind sites on every segment.
    #[cfg_attr(not(feature = "rvr"), allow(dead_code))]
    kind_to_air: HashMap<DeltaAirKind, usize>,
}

#[cfg(all(feature = "cuda", feature = "rvr"))]
struct HostDeltaSegment {
    delta: openvm_circuit::arch::rvr::RvrDeltaRecords,
    memory_log: Vec<openvm_circuit::arch::rvr::MemoryLogEntry>,
    delta_memory_log: Vec<openvm_circuit::arch::rvr::DeltaMemoryLogEntry>,
    program_log: Vec<openvm_circuit::arch::rvr::ProgramLogEntry>,
    program_runs: Vec<openvm_circuit::arch::rvr::ProgramRunEntry>,
    device_program_references: Vec<openvm_circuit::arch::rvr::DeviceProgramEntry>,
    program_instruction_count: usize,
    program_frequency_count: usize,
    program_frequency_reference: Vec<u32>,
    touched: Vec<openvm_circuit::arch::rvr::TouchedBlock>,
    device_aux_patches: Vec<openvm_circuit::arch::rvr::DeviceAuxPatch>,
    device_aux_references: Vec<openvm_circuit::arch::rvr::DeviceAuxReference>,
    oracle_expected: HashMap<usize, Vec<u8>>,
    oracle_arena_expected: Vec<openvm_circuit::arch::rvr::DeviceAuxArenaReference>,
    arena_native_flags: Vec<u8>,
    specs: Vec<DeltaAirSpec>,
}

#[cfg(all(feature = "cuda", feature = "rvr"))]
struct HostG2Segment {
    segment: openvm_circuit::arch::rvr::RvrG2SegmentV1,
    meta: Arc<openvm_circuit::arch::rvr::RvrG2MetaV1>,
    initial_timestamp: u32,
    specs: Vec<DeltaAirSpec>,
    opaque: Vec<G2ExpectedOpaqueV1>,
    total_record_count: usize,
    program_frequency_count: usize,
    oracle_expected: HashMap<usize, Vec<u8>>,
    program_frequency_reference: Vec<u32>,
}

#[cfg(all(feature = "cuda", feature = "rvr"))]
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct G2ExpectedKindV1 {
    kind: u32,
    air_idx: u32,
    count: u32,
}

#[cfg(all(feature = "cuda", feature = "rvr"))]
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct G2ExpectedOpaqueV1 {
    lane_kind: u32,
    air_idx: u32,
    count: u32,
    payload_bytes: u32,
    stride: u32,
}

#[cfg(all(feature = "cuda", feature = "rvr"))]
const _: () = assert!(size_of::<G2ExpectedOpaqueV1>() == 20);

#[cfg(all(feature = "cuda", feature = "rvr"))]
const _: () = assert!(size_of::<G2ExpectedKindV1>() == 12);

#[cfg(all(feature = "cuda", feature = "rvr"))]
#[derive(Clone, Copy)]
struct DeltaAirSpec {
    kind: DeltaAirKind,
    air_idx: usize,
    count: usize,
}

#[cfg(all(feature = "cuda", feature = "rvr"))]
struct DeviceDeltaSegment {
    outputs: HashMap<DeltaAirKind, Arc<DeviceBuffer<u8>>>,
    touched_memory: Option<DeviceTouchedMemory>,
    program_frequencies: Option<DeviceProgramFrequencies>,
    g2_segment_id: Option<u32>,
}

#[cfg(all(feature = "cuda", feature = "rvr"))]
struct DeviceCompactResidualSegment {
    memory: Arc<DeviceBuffer<u8>>,
    count: usize,
}

/// Device descriptor indexed by global AIR. CUDA writes each stable partition
/// directly into the owning compact buffer.
#[cfg(all(feature = "cuda", feature = "rvr"))]
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct DeltaAirOutputDesc {
    base: u64,
    count: u32,
    stride: u32,
    sorted_start: u32,
    kind: u32,
}

#[cfg(all(feature = "cuda", feature = "rvr"))]
const _: () = {
    assert!(size_of::<DeltaAirOutputDesc>() == 24);
    assert!(align_of::<DeltaAirOutputDesc>() == 8);
    assert!(core::mem::offset_of!(DeltaAirOutputDesc, kind) == 20);
};

/// Shared producer/consumer state. The builder holds one `Arc` per VM and
/// clones it into every migrated GPU chip at construction.
#[derive(Default)]
pub struct RvrGpuDecodeState {
    table: Mutex<Option<HostOperandTable>>,
    /// Device copy of the operand table, keyed by the host table's identity.
    #[cfg(feature = "cuda")]
    device_table: Mutex<Option<(usize, Arc<DeviceBuffer<u8>>)>>,
    #[cfg(all(feature = "cuda", feature = "rvr"))]
    delta_host: Mutex<Option<HostDeltaSegment>>,
    #[cfg(all(feature = "cuda", feature = "rvr"))]
    delta_device: Mutex<Option<DeviceDeltaSegment>>,
    /// Full residual-memory events retained by compact mode. Multi-block
    /// compact consumers upload this once and reconstruct only crossing rows
    /// into the upstream-native 60-byte record geometry on the device.
    #[cfg(all(feature = "cuda", feature = "rvr"))]
    compact_residual_host: Mutex<Option<Vec<openvm_circuit::arch::rvr::MemoryLogEntry>>>,
    #[cfg(all(feature = "cuda", feature = "rvr"))]
    compact_residual_device: Mutex<Option<DeviceCompactResidualSegment>>,
    #[cfg(all(feature = "cuda", feature = "rvr"))]
    g2_host: Mutex<Option<HostG2Segment>>,
    #[cfg(all(feature = "cuda", feature = "rvr"))]
    g2_device: Mutex<Option<DeviceDeltaSegment>>,
}

impl RvrGpuDecodeState {
    #[cfg(feature = "rvr")]
    fn bind_program<F: PrimeField32>(
        &self,
        exe: &VmExe<F>,
        pc_to_air_idx: &[Option<usize>],
        compiled_identity: &Arc<Vec<bool>>,
        precomputed: Option<&RvrDeltaDecodePrecompute>,
    ) -> HashMap<DeltaAirKind, usize> {
        let mut table = self.table.lock().unwrap();
        let rebuild = table
            .as_ref()
            .map(|t| !Arc::ptr_eq(&t.compiled_identity, compiled_identity))
            .unwrap_or(true);
        if rebuild {
            let bound = if let Some(precomputed) = precomputed {
                let precomputed =
                    host_table_from_precomputed(Arc::clone(compiled_identity), precomputed);
                #[cfg(debug_assertions)]
                {
                    let lazy =
                        build_operand_table(exe, Arc::clone(compiled_identity), pc_to_air_idx);
                    assert_host_tables_equal(&precomputed, &lazy);
                }
                precomputed
            } else {
                build_operand_table(exe, Arc::clone(compiled_identity), pc_to_air_idx)
            };
            *table = Some(bound);
            #[cfg(feature = "cuda")]
            {
                *self.device_table.lock().unwrap() = None;
            }
        }
        table
            .as_ref()
            .expect("operand table was just bound")
            .kind_to_air
            .clone()
    }

    /// The device operand table (uploaded once per bound exe) + its pc base.
    #[cfg(feature = "cuda")]
    pub fn device_operand_table(
        &self,
        device_ctx: &GpuDeviceCtx,
    ) -> Option<(Arc<DeviceBuffer<u8>>, u32)> {
        let (entries, pc_base) = self.operand_table()?;
        let key = Arc::as_ptr(&entries) as usize;
        let mut cache = self.device_table.lock().unwrap();
        if let Some((cached_key, buf)) = cache.as_ref() {
            if *cached_key == key {
                return Some((Arc::clone(buf), pc_base));
            }
        }
        // SAFETY: DeviceOperandEntry is repr(C), size 20, no padding invariants
        // beyond the static asserts; reinterpreting as bytes for H2D is sound.
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                entries.as_ptr() as *const u8,
                entries.len() * size_of::<DeviceOperandEntry>(),
            )
        };
        let buf = Arc::new(bytes.to_device_on(device_ctx).expect("operand table H2D"));
        *cache = Some((key, Arc::clone(&buf)));
        Some((buf, pc_base))
    }

    /// The bound exe's operand table + its pc base (consumer side).
    pub fn operand_table(&self) -> Option<(Arc<Vec<DeviceOperandEntry>>, u32)> {
        let table = self.table.lock().unwrap();
        table.as_ref().map(|t| (Arc::clone(&t.entries), t.pc_base))
    }

    /// Producer side: bind `exe` (building the alu_u16-family operand table if
    /// this exe isn't already bound) and mark the family's AIRs as
    /// `CompactWire` for the CURRENT segment. Returns the compact AIR set.
    #[cfg(feature = "rvr")]
    pub fn bind_compact_segment<F: PrimeField32>(
        &self,
        exe: &VmExe<F>,
        pc_to_air_idx: &[Option<usize>],
        compiled_identity: &Arc<Vec<bool>>,
        precomputed: Option<&RvrDeltaDecodePrecompute>,
    ) -> HashSet<usize> {
        let airs = self
            .bind_program(exe, pc_to_air_idx, compiled_identity, precomputed)
            .into_values()
            .collect::<HashSet<_>>();
        // Each supported format maps to at most one decode-kernel
        // AIRs; more means opcode->air routing changed under us.
        assert!(
            airs.len() <= DeltaAirKind::COUNT,
            "gpu-decode formats mapped to {} AIRs (expected <= {})",
            airs.len(),
            DeltaAirKind::COUNT,
        );
        airs
    }

    /// Pure route classification used before native execution selects the
    /// compact staging targets. The actual operand table is bound later from
    /// the compiled metadata identity carried by the preflight output.
    #[cfg(feature = "rvr")]
    pub fn compact_record_airs<F: PrimeField32>(
        exe: &VmExe<F>,
        pc_to_air_idx: &[Option<usize>],
        inline_pc_slots: &[bool],
        precomputed: Option<&RvrDeltaDecodePrecompute>,
    ) -> HashSet<usize> {
        if let Some(precomputed) = precomputed {
            let airs = precomputed
                .kind_to_air
                .iter()
                .filter_map(|&(kind, air)| DeltaAirKind::from_repr(kind).map(|_| air))
                .collect();
            #[cfg(debug_assertions)]
            assert_eq!(
                airs,
                classify_kind_to_air(exe, pc_to_air_idx, inline_pc_slots)
                    .into_values()
                    .collect()
            );
            airs
        } else {
            classify_kind_to_air(exe, pc_to_air_idx, inline_pc_slots)
                .into_values()
                .collect()
        }
    }

    /// Bind the compact segment's residual events after host assembly has
    /// consumed all non-compact AIRs. The vector is normally empty; crossing
    /// multi-byte rows contribute exactly two complete block events apiece.
    #[cfg(all(feature = "cuda", feature = "rvr"))]
    pub(crate) fn bind_compact_residual_segment(
        &self,
        memory_log: Vec<openvm_circuit::arch::rvr::MemoryLogEntry>,
    ) {
        *self.compact_residual_host.lock().unwrap() = Some(memory_log);
        *self.compact_residual_device.lock().unwrap() = None;
    }

    #[cfg(all(feature = "cuda", feature = "rvr"))]
    pub(crate) fn clear_compact_residual_segment(&self) {
        *self.compact_residual_host.lock().unwrap() = None;
        *self.compact_residual_device.lock().unwrap() = None;
    }

    #[cfg(all(feature = "cuda", feature = "rvr"))]
    fn compact_residual_device(&self, device_ctx: &GpuDeviceCtx) -> (Arc<DeviceBuffer<u8>>, usize) {
        let mut device = self.compact_residual_device.lock().unwrap();
        if let Some(bound) = device.as_ref() {
            return (Arc::clone(&bound.memory), bound.count);
        }
        let entries = self
            .compact_residual_host
            .lock()
            .unwrap()
            .take()
            .expect("compact multi-block decode without a bound residual segment");
        let bytes = unsafe {
            std::slice::from_raw_parts(
                entries.as_ptr().cast::<u8>(),
                std::mem::size_of_val(entries.as_slice()),
            )
        };
        let memory = Arc::new(if bytes.is_empty() {
            DeviceBuffer::new()
        } else {
            bytes
                .to_device_on(device_ctx)
                .expect("compact residual-memory H2D")
        });
        let count = entries.len();
        *device = Some(DeviceCompactResidualSegment {
            memory: Arc::clone(&memory),
            count,
        });
        (memory, count)
    }

    /// Expand one multi-byte compact AIR on the device. Non-crossing records
    /// synthesize the absent second block; crossing records consume the two
    /// full residual events at their fixed instruction-local timestamps.
    #[cfg(all(feature = "cuda", feature = "rvr"))]
    pub fn expand_compact_multiblock(
        &self,
        kind: DeltaAirKind,
        records: &DeviceBuffer<u8>,
        device_ctx: &GpuDeviceCtx,
    ) -> Arc<DeviceBuffer<u8>> {
        assert!(kind.crossing_residual_capable());
        use openvm_circuit::arch::rvr::PREFLIGHT_ADDSUB_RECORD_SIZE;
        assert_eq!(records.len() % PREFLIGHT_ADDSUB_RECORD_SIZE, 0);
        let count = records.len() / PREFLIGHT_ADDSUB_RECORD_SIZE;
        let output = Arc::new(DeviceBuffer::<u8>::with_capacity_on(
            count * RVR_MULTI_BLOCK_RECORD_SIZE,
            device_ctx,
        ));
        let (memory, memory_count) = self.compact_residual_device(device_ctx);
        let (table, pc_base) = self
            .device_operand_table(device_ctx)
            .expect("compact multi-block segment without an operand table");
        let error = DeviceBuffer::<u32>::with_capacity_on(1, device_ctx);
        error
            .fill_zero_on(device_ctx)
            .expect("compact multi-block error clear");
        unsafe {
            crate::cuda_abi::rvr_delta_cuda::expand_compact_multiblock(
                records,
                count,
                &memory,
                memory_count,
                &table,
                pc_base,
                kind,
                &output,
                &error,
                device_ctx.stream.as_raw(),
            )
            .expect("compact multi-block expansion launch");
        }
        let error = error
            .to_host_on(device_ctx)
            .expect("compact multi-block error D2H")[0];
        assert_eq!(error, 0, "compact multi-block decode error {error}");
        output
    }

    #[cfg(feature = "rvr")]
    pub fn bind_delta_airs<F: PrimeField32>(
        &self,
        exe: &VmExe<F>,
        pc_to_air_idx: &[Option<usize>],
        compiled_identity: &Arc<Vec<bool>>,
        precomputed: Option<&RvrDeltaDecodePrecompute>,
    ) -> HashSet<usize> {
        self.bind_program(exe, pc_to_air_idx, compiled_identity, precomputed)
            .into_values()
            .collect()
    }

    /// Bind one delta segment without invoking the CPU reference decoder.
    /// The escaping raw logs and pinned delta backing remain owned here until
    /// the first GPU chip launches the shared predecode.
    #[cfg(all(feature = "cuda", feature = "rvr"))]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn bind_delta_segment<F: PrimeField32>(
        &self,
        exe: &VmExe<F>,
        pc_to_air_idx: &[Option<usize>],
        compiled_identity: &Arc<Vec<bool>>,
        precomputed: Option<&RvrDeltaDecodePrecompute>,
        delta: openvm_circuit::arch::rvr::RvrDeltaRecords,
        memory_log: Vec<openvm_circuit::arch::rvr::MemoryLogEntry>,
        delta_memory_log: Vec<openvm_circuit::arch::rvr::DeltaMemoryLogEntry>,
        program_log: Vec<openvm_circuit::arch::rvr::ProgramLogEntry>,
        program_runs: Vec<openvm_circuit::arch::rvr::ProgramRunEntry>,
        device_program_references: Vec<openvm_circuit::arch::rvr::DeviceProgramEntry>,
        program_frequency_count: usize,
        program_frequency_reference: Vec<u32>,
        touched: Vec<openvm_circuit::arch::rvr::TouchedBlock>,
        device_aux_patches: Vec<openvm_circuit::arch::rvr::DeviceAuxPatch>,
        device_aux_references: Vec<openvm_circuit::arch::rvr::DeviceAuxReference>,
        oracle_expected: HashMap<usize, Vec<u8>>,
        oracle_arena_expected: Vec<openvm_circuit::arch::rvr::DeviceAuxArenaReference>,
        chip_counts: &[u32],
        arena_native_written: &[(usize, u32)],
    ) -> Result<HashSet<usize>, openvm_circuit::arch::ExecutionError> {
        use openvm_circuit::arch::rvr::PREFLIGHT_DELTA_RECORD_SIZE;

        let kind_to_air = self.bind_program(exe, pc_to_air_idx, compiled_identity, precomputed);
        let arena_native = arena_native_written
            .iter()
            .map(|&(air, _)| air)
            .collect::<HashSet<_>>();
        let mut specs = Vec::new();
        let mut delta_airs = HashSet::new();
        for (kind, air_idx) in kind_to_air {
            if arena_native.contains(&air_idx) {
                continue;
            }
            let count = chip_counts.get(air_idx).copied().ok_or_else(|| {
                openvm_circuit::arch::ExecutionError::RvrExecution(format!(
                    "delta AIR {air_idx} has no chip-count slot"
                ))
            })? as usize;
            delta_airs.insert(air_idx);
            specs.push(DeltaAirSpec {
                kind,
                air_idx,
                count,
            });
        }
        let records = delta.bytes().len() / PREFLIGHT_DELTA_RECORD_SIZE;
        let expected = specs.iter().map(|spec| spec.count).sum::<usize>();
        if expected != records {
            return Err(openvm_circuit::arch::ExecutionError::RvrExecution(format!(
                "device delta count mismatch: {records} chronological records but per-AIR counts sum to {expected}"
            )));
        }
        let mut program_instruction_count = 0usize;
        for (run_index, run) in program_runs.iter().enumerate() {
            if run.complete != 1
                || run.instruction_count == 0
                || run.chronology_offset as usize != program_instruction_count
            {
                return Err(openvm_circuit::arch::ExecutionError::RvrExecution(format!(
                    "device chronology run {run_index} is not a complete contiguous block"
                )));
            }
            program_instruction_count = program_instruction_count
                .checked_add(run.instruction_count as usize)
                .ok_or_else(|| {
                    openvm_circuit::arch::ExecutionError::RvrExecution(
                        "device chronology instruction count overflow".to_string(),
                    )
                })?;
        }
        if program_instruction_count == 0
            || program_frequency_count == 0
            || (!device_program_references.is_empty()
                && device_program_references.len() != program_instruction_count)
            || (!program_frequency_reference.is_empty()
                && program_frequency_reference.len() != program_frequency_count)
        {
            return Err(openvm_circuit::arch::ExecutionError::RvrExecution(format!(
                "invalid device chronology dimensions: instructions={program_instruction_count}, frequencies={program_frequency_count}, chronology_reference={}, frequency_reference={}",
                device_program_references.len(),
                program_frequency_reference.len(),
            )));
        }
        let mut arena_native_flags = vec![0u8; chip_counts.len()];
        for &air in &arena_native {
            let Some(flag) = arena_native_flags.get_mut(air) else {
                return Err(openvm_circuit::arch::ExecutionError::RvrExecution(format!(
                    "arena-native AIR {air} exceeds chip-count table"
                )));
            };
            *flag = 1;
        }
        let host = HostDeltaSegment {
            delta,
            memory_log,
            delta_memory_log,
            program_log,
            program_runs,
            device_program_references,
            program_instruction_count,
            program_frequency_count,
            program_frequency_reference,
            touched,
            device_aux_patches,
            device_aux_references,
            oracle_expected,
            oracle_arena_expected,
            arena_native_flags,
            specs,
        };
        *self.delta_host.lock().unwrap() = Some(host);
        *self.delta_device.lock().unwrap() = None;
        Ok(delta_airs)
    }

    /// Bind one G2 segment after its O(lanes) host finalizer. The complete
    /// wire backing stays owned here until the system inventory initiates the
    /// one-shot CUDA expansion.
    #[cfg(all(feature = "cuda", feature = "rvr"))]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn bind_g2_segment<F: PrimeField32>(
        &self,
        exe: &VmExe<F>,
        pc_to_air_idx: &[Option<usize>],
        compiled_identity: &Arc<Vec<bool>>,
        precomputed: &RvrDeltaDecodePrecompute,
        segment: openvm_circuit::arch::rvr::RvrG2SegmentV1,
        meta: Arc<openvm_circuit::arch::rvr::RvrG2MetaV1>,
        initial_timestamp: u32,
        chip_counts: &[u32],
        program_frequency_count: usize,
        oracle_expected: HashMap<usize, Vec<u8>>,
        program_frequency_reference: Vec<u32>,
    ) -> Result<HashSet<usize>, openvm_circuit::arch::ExecutionError> {
        let kind_to_air =
            self.bind_program(exe, pc_to_air_idx, compiled_identity, Some(precomputed));
        if kind_to_air.len() != meta.air_bindings.len()
            || meta.air_bindings.iter().any(|binding| {
                DeltaAirKind::from_repr(binding.kind)
                    .and_then(|kind| kind_to_air.get(&kind).copied())
                    != Some(binding.air_idx)
            })
        {
            return Err(openvm_circuit::arch::ExecutionError::RvrExecution(
                "G2 Phase-2a AIR binding differs from the compiled manifest".to_string(),
            ));
        }
        let descs = segment.validate(&meta.fingerprint)?;
        let specs = meta
            .air_bindings
            .iter()
            .map(|binding| {
                let kind = DeltaAirKind::from_repr(binding.kind).ok_or_else(|| {
                    openvm_circuit::arch::ExecutionError::RvrExecution(format!(
                        "G2 manifest contains unknown decoder kind {}",
                        binding.kind
                    ))
                })?;
                let count = chip_counts.get(binding.air_idx).copied().ok_or_else(|| {
                    openvm_circuit::arch::ExecutionError::RvrExecution(format!(
                        "G2 AIR {} has no chip-count slot",
                        binding.air_idx
                    ))
                })? as usize;
                Ok::<_, openvm_circuit::arch::ExecutionError>(DeltaAirSpec {
                    kind,
                    air_idx: binding.air_idx,
                    count,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let total_record_count = specs.iter().try_fold(0usize, |sum, spec| {
            sum.checked_add(spec.count).ok_or_else(|| {
                openvm_circuit::arch::ExecutionError::RvrExecution(
                    "G2 record count overflow".to_string(),
                )
            })
        })?;
        let opaque = meta
            .opaque_bindings
            .iter()
            .map(|binding| {
                let count = chip_counts.get(binding.air_idx).copied().ok_or_else(|| {
                    openvm_circuit::arch::ExecutionError::RvrExecution(format!(
                        "G2 opaque AIR {} has no chip-count slot",
                        binding.air_idx
                    ))
                })?;
                let stride = u32::try_from(binding.geometry.stride_dense()).map_err(|_| {
                    openvm_circuit::arch::ExecutionError::RvrExecution(
                        "G2 opaque stride exceeds the frozen u32 descriptor".to_string(),
                    )
                })?;
                let payload_bytes = count.checked_mul(stride).ok_or_else(|| {
                    openvm_circuit::arch::ExecutionError::RvrExecution(
                        "G2 opaque payload byte count overflow".to_string(),
                    )
                })?;
                let lane_kind = u32::from(binding.lane_kind());
                let desc = descs
                    .iter()
                    .find(|desc| u32::from(desc.kind) == lane_kind)
                    .ok_or_else(|| {
                        openvm_circuit::arch::ExecutionError::RvrExecution(format!(
                            "G2 opaque AIR {} has no committed descriptor",
                            binding.air_idx
                        ))
                    })?;
                if desc.count != count || desc.payload_bytes != payload_bytes {
                    return Err(openvm_circuit::arch::ExecutionError::RvrExecution(format!(
                        "G2 opaque AIR {} descriptor count drifted",
                        binding.air_idx
                    )));
                }
                Ok(G2ExpectedOpaqueV1 {
                    lane_kind,
                    air_idx: binding.air_idx as u32,
                    count,
                    payload_bytes,
                    stride,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        *self.delta_host.lock().unwrap() = None;
        *self.delta_device.lock().unwrap() = None;
        *self.g2_host.lock().unwrap() = Some(HostG2Segment {
            segment,
            meta,
            initial_timestamp,
            specs,
            opaque,
            total_record_count,
            program_frequency_count,
            oracle_expected,
            program_frequency_reference,
        });
        *self.g2_device.lock().unwrap() = None;
        Ok(kind_to_air.into_values().collect())
    }

    /// Drop any previous segment's delta state before entering a non-delta
    /// route. This prevents a builder reused across emission modes from
    /// exposing stale device buffers to a later segment.
    #[cfg(all(feature = "cuda", feature = "rvr"))]
    pub fn clear_delta_segment(&self) {
        *self.delta_host.lock().unwrap() = None;
        *self.delta_device.lock().unwrap() = None;
        *self.g2_host.lock().unwrap() = None;
        *self.g2_device.lock().unwrap() = None;
    }

    /// Perform the whole segment predecode exactly once. The system inventory
    /// calls this first for device-replayed touched memory; compact-record
    /// consumers then clone their already-populated per-AIR outputs.
    #[cfg(all(feature = "cuda", feature = "rvr"))]
    fn ensure_device_delta_segment(
        &self,
        device_ctx: &GpuDeviceCtx,
        initial_memory: Option<&[DeviceInitialMemory]>,
    ) -> bool {
        if self.delta_device.lock().unwrap().is_some() {
            return true;
        }

        let mut host_guard = self.delta_host.lock().unwrap();
        let Some(mut host) = host_guard.take() else {
            return false;
        };
        let initial_memory = initial_memory.expect(
            "delta predecode must be initiated by the system memory inventory before chip tracegen",
        );
        let (d_table, pc_base) = self
            .device_operand_table(device_ctx)
            .expect("delta segment without a bound operand table");

        let mut outputs = HashMap::new();
        let mut descs = vec![DeltaAirOutputDesc::default(); host.arena_native_flags.len()];
        let mut sorted_specs = host.specs.clone();
        sorted_specs.sort_unstable_by_key(|spec| spec.air_idx);
        let mut sorted_start = 0usize;
        for spec in &sorted_specs {
            let stride = spec.kind.wire_size();
            if spec.count != 0 {
                let buffer = Arc::new(DeviceBuffer::<u8>::with_capacity_on(
                    spec.count * stride,
                    device_ctx,
                ));
                descs[spec.air_idx] = DeltaAirOutputDesc {
                    base: buffer.as_ptr() as u64,
                    count: u32::try_from(spec.count).expect("delta AIR count exceeds u32"),
                    stride: stride as u32,
                    sorted_start: u32::try_from(sorted_start)
                        .expect("delta sorted offset exceeds u32"),
                    kind: spec.kind as u32,
                };
                outputs.insert(spec.kind, buffer);
            }
            sorted_start += spec.count;
        }

        let d_delta = host
            .delta
            .bytes()
            .to_device_on(device_ctx)
            .expect("delta H2D");
        if !host.memory_log.is_empty() && !host.delta_memory_log.is_empty() {
            panic!("delta segment populated both full and compact residual-memory schemas");
        }
        let (memory_bytes, memory_count, memory_stride) = if !host.delta_memory_log.is_empty() {
            let entries = host.delta_memory_log.as_slice();
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    entries.as_ptr().cast::<u8>(),
                    std::mem::size_of_val(entries),
                )
            };
            (
                bytes,
                entries.len(),
                std::mem::size_of::<openvm_circuit::arch::rvr::DeltaMemoryLogEntry>(),
            )
        } else {
            let entries = host.memory_log.as_slice();
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    entries.as_ptr().cast::<u8>(),
                    std::mem::size_of_val(entries),
                )
            };
            (
                bytes,
                entries.len(),
                std::mem::size_of::<openvm_circuit::arch::rvr::MemoryLogEntry>(),
            )
        };
        let delta_count =
            host.delta.bytes().len() / openvm_circuit::arch::rvr::PREFLIGHT_DELTA_RECORD_SIZE;
        let event_capacity = delta_count
            .checked_mul(4)
            .and_then(|count| count.checked_add(memory_count))
            .and_then(|count| count.checked_add(host.program_log.len().saturating_mul(3)))
            .expect("delta event capacity overflow");
        let d_memory = if memory_bytes.is_empty() {
            DeviceBuffer::new()
        } else {
            memory_bytes
                .to_device_on(device_ctx)
                .expect("delta residual-memory H2D")
        };
        let d_program = if host.program_log.is_empty() {
            DeviceBuffer::new()
        } else {
            host.program_log
                .as_slice()
                .to_device_on(device_ctx)
                .expect("delta W-chronology H2D")
        };
        let d_program_runs = host
            .program_runs
            .as_slice()
            .to_device_on(device_ctx)
            .expect("device program runs H2D");
        let d_program_frequencies =
            DeviceBuffer::<u32>::with_capacity_on(host.program_frequency_count, device_ctx);
        let d_program_chronology =
            DeviceBuffer::<openvm_circuit::arch::rvr::DeviceProgramEntry>::with_capacity_on(
                host.program_instruction_count,
                device_ctx,
            );
        let d_initial_memory = initial_memory
            .to_device_on(device_ctx)
            .expect("delta initial-memory descriptors H2D");
        let d_touched_output = if event_capacity == 0 {
            DeviceBuffer::new()
        } else {
            DeviceBuffer::<u32>::with_capacity_on(
                event_capacity * DEVICE_TOUCHED_RECORD_WORDS,
                device_ctx,
            )
        };
        let d_touched_count = DeviceBuffer::<u32>::with_capacity_on(1, device_ctx);
        let d_memory_prev_timestamps = if memory_count == 0 {
            DeviceBuffer::new()
        } else {
            DeviceBuffer::<u32>::with_capacity_on(memory_count, device_ctx)
        };
        let d_memory_prev_values = if memory_count == 0 {
            DeviceBuffer::new()
        } else {
            DeviceBuffer::<u64>::with_capacity_on(memory_count, device_ctx)
        };
        let d_flags = host
            .arena_native_flags
            .as_slice()
            .to_device_on(device_ctx)
            .expect("delta arena flags H2D");
        let d_descs = descs
            .as_slice()
            .to_device_on(device_ctx)
            .expect("delta output descriptors H2D");
        let d_error = DeviceBuffer::<u32>::with_capacity_on(1, device_ctx);
        let d_expected_blocks = DeviceBuffer::<u64>::new();
        let d_expected_modes = DeviceBuffer::<u8>::new();
        d_error.fill_zero_on(device_ctx).expect("delta error clear");
        unsafe {
            crate::cuda_abi::rvr_delta_cuda::predecode(
                &d_delta,
                delta_count,
                &d_memory,
                memory_count,
                memory_stride,
                &d_program,
                host.program_log.len(),
                &d_program_runs,
                host.program_instruction_count,
                &d_program_frequencies,
                &d_program_chronology,
                &d_initial_memory,
                &d_touched_output,
                &d_touched_count,
                &d_memory_prev_timestamps,
                &d_memory_prev_values,
                &d_table,
                pc_base,
                &d_flags,
                &d_descs,
                &d_expected_blocks,
                &d_expected_modes,
                u32::MAX,
                &d_error,
                device_ctx.stream.as_raw(),
            )
            .expect("CUDA delta predecode launch");
        }
        let error = d_error.to_host_on(device_ctx).expect("delta error D2H")[0];
        assert_eq!(error, 0, "CUDA delta predecode fail-closed error {error}");
        if !host.device_program_references.is_empty()
            || !host.program_frequency_reference.is_empty()
        {
            assert!(
                !host.device_program_references.is_empty()
                    && !host.program_frequency_reference.is_empty()
                    && host
                        .program_frequency_reference
                        .iter()
                        .any(|&count| count != 0),
                "device chronology oracle references must be non-vacuous"
            );
            let actual_chronology = d_program_chronology
                .to_host_on(device_ctx)
                .expect("device chronology oracle D2H");
            let actual_frequencies = d_program_frequencies
                .to_host_on(device_ctx)
                .expect("device program-frequency oracle D2H");
            let chronology_bytes = |entries: &[openvm_circuit::arch::rvr::DeviceProgramEntry]| {
                // SAFETY: DeviceProgramEntry is a fully initialized repr(C)
                // pair of u32s with no padding.
                unsafe {
                    std::slice::from_raw_parts(
                        entries.as_ptr().cast::<u8>(),
                        std::mem::size_of_val(entries),
                    )
                }
            };
            assert_eq!(
                chronology_bytes(&actual_chronology),
                chronology_bytes(&host.device_program_references),
                "device program chronology differs byte-for-byte from host chronology"
            );
            let frequency_bytes = |entries: &[u32]| {
                // SAFETY: u32 has no padding and both slices are initialized.
                unsafe {
                    std::slice::from_raw_parts(
                        entries.as_ptr().cast::<u8>(),
                        std::mem::size_of_val(entries),
                    )
                }
            };
            assert_eq!(
                frequency_bytes(&actual_frequencies),
                frequency_bytes(&host.program_frequency_reference),
                "device program frequencies differ byte-for-byte from host frequencies"
            );
        }
        let touched_count = d_touched_count
            .to_host_on(device_ctx)
            .expect("delta touched count D2H")[0] as usize;
        assert!(
            touched_count <= event_capacity,
            "CUDA delta replay returned {touched_count} touched groups for {event_capacity} events"
        );

        let memory_prev_timestamps = d_memory_prev_timestamps
            .to_host_on(device_ctx)
            .expect("delta residual prev-timestamp D2H");
        let memory_prev_values = d_memory_prev_values
            .to_host_on(device_ctx)
            .expect("delta residual prev-value D2H");
        assert_eq!(memory_prev_timestamps.len(), memory_count);
        assert_eq!(memory_prev_values.len(), memory_count);
        if !host.device_aux_references.is_empty() {
            assert_eq!(
                host.device_aux_references.len(),
                memory_count,
                "device aux oracle reference count differs from residual chronology"
            );
            for (index, reference) in host.device_aux_references.iter().enumerate() {
                assert_eq!(
                    memory_prev_timestamps[index], reference.prev_timestamp,
                    "device residual prev_timestamp differs from host at event {index}"
                );
                assert_eq!(
                    memory_prev_values[index], reference.prev_value,
                    "device residual prev_value differs from host at event {index}"
                );
            }
        }
        for (patch_index, patch) in host.device_aux_patches.iter().enumerate() {
            let event_index = patch.event_index as usize;
            assert!(
                event_index < memory_count,
                "device aux patch {patch_index} references event {event_index}/{memory_count}"
            );
            let value = match patch.kind {
                openvm_circuit::arch::rvr::DEVICE_AUX_PATCH_U32 => {
                    u64::from(memory_prev_timestamps[event_index])
                }
                openvm_circuit::arch::rvr::DEVICE_AUX_PATCH_U64 => memory_prev_values[event_index],
                kind => panic!("device aux patch {patch_index} has invalid kind {kind}"),
            };
            if !host.device_aux_references.is_empty() {
                assert_eq!(
                    value, patch.expected,
                    "device aux patch {patch_index} differs from its host field"
                );
            }
            // SAFETY: native preflight validated every target and width against
            // a live staged-arena backing. Those Vec allocations have moved by
            // ownership only, so their data pointers remain stable until chip
            // trace generation consumes them after this system-first replay.
            unsafe {
                match patch.kind {
                    openvm_circuit::arch::rvr::DEVICE_AUX_PATCH_U32 => {
                        (patch.target as *mut u32).write_unaligned(value as u32);
                    }
                    openvm_circuit::arch::rvr::DEVICE_AUX_PATCH_U64 => {
                        (patch.target as *mut u64).write_unaligned(value);
                    }
                    _ => unreachable!(),
                }
            }
        }

        for arena in &host.oracle_arena_expected {
            // SAFETY: this points into the live DenseRecordArena allocation
            // whose aux fields were just patched above. Moving the arena Vec
            // does not move its byte allocation.
            let actual = unsafe {
                std::slice::from_raw_parts(arena.base as *const u8, arena.expected.len())
            };
            assert_eq!(
                actual,
                arena.expected.as_slice(),
                "device aux full-arena bytes differ from host for AIR {}",
                arena.air_idx
            );
        }

        if !host.oracle_expected.is_empty() {
            for spec in &host.specs {
                if spec.count == 0 || spec.kind.crossing_residual_capable() {
                    continue;
                }
                let expected = host.oracle_expected.get(&spec.air_idx).unwrap_or_else(|| {
                    panic!("device delta oracle omitted expected AIR {}", spec.air_idx)
                });
                let actual = outputs
                    .get(&spec.kind)
                    .unwrap_or_else(|| panic!("device delta oracle omitted {:?}", spec.kind))
                    .to_host_on(device_ctx)
                    .expect("device delta compact oracle D2H");
                assert_eq!(
                    actual.as_slice(),
                    expected.as_slice(),
                    "device delta compact bytes differ from host for AIR {} ({:?})",
                    spec.air_idx,
                    spec.kind
                );
            }
        }

        let program_log = std::mem::take(&mut host.program_log);
        let program_runs = std::mem::take(&mut host.program_runs);
        let device_program_references = std::mem::take(&mut host.device_program_references);
        let memory_log = std::mem::take(&mut host.memory_log);
        let delta_memory_log = std::mem::take(&mut host.delta_memory_log);
        let touched = std::mem::take(&mut host.touched);
        host.delta.recycle_device_inputs(
            program_log,
            program_runs,
            device_program_references,
            memory_log,
            delta_memory_log,
            touched,
        );
        drop(host);
        let device = DeviceDeltaSegment {
            outputs,
            touched_memory: Some(DeviceTouchedMemory {
                records: d_touched_output,
                num_records: touched_count,
            }),
            program_frequencies: Some(DeviceProgramFrequencies {
                frequencies: d_program_frequencies,
            }),
            g2_segment_id: None,
        };
        *self.delta_device.lock().unwrap() = Some(device);
        true
    }

    #[cfg(all(feature = "cuda", feature = "rvr"))]
    fn ensure_device_g2_segment(
        &self,
        device_ctx: &GpuDeviceCtx,
        initial_memory: Option<&[DeviceInitialMemory]>,
    ) -> bool {
        if self.g2_device.lock().unwrap().is_some() {
            return true;
        }
        let mut host_guard = self.g2_host.lock().unwrap();
        let Some(host) = host_guard.take() else {
            return false;
        };
        let header = host
            .segment
            .header_acquire()
            .expect("G2 committed header before device decode");
        let segment_id = header.segment_id;
        let initial_memory = initial_memory.expect(
            "G2 predecode must be initiated by the system memory inventory before chip tracegen",
        );
        let (d_table, pc_base) = self
            .device_operand_table(device_ctx)
            .expect("G2 segment without a bound operand table");
        let h2d_timer = CudaStageTimer::start(device_ctx);
        let d_wire =
            DeviceBuffer::<u8>::with_capacity_on(host.segment.transfer_byte_len(), device_ctx);
        d_wire
            .fill_zero_on(device_ctx)
            .expect("G2 wire padding clear");
        for (offset, bytes) in host.segment.wire_parts() {
            if bytes.is_empty() {
                continue;
            }
            unsafe {
                // SAFETY: finalization validated every compact wire range;
                // the compact destination covers every transferred range
                // and remains alive through the queued decode. Opaque-final
                // payloads stay in their separately owned custom arenas.
                cuda_memcpy_on::<false, true>(
                    d_wire.as_mut_ptr().add(offset).cast(),
                    bytes.as_ptr().cast(),
                    bytes.len(),
                    device_ctx,
                )
            }
            .expect("G2 scatter/gather wire H2D");
        }
        let d_fingerprint = host
            .meta
            .fingerprint
            .as_slice()
            .to_device_on(device_ctx)
            .expect("G2 fingerprint H2D");
        let d_blocks = host
            .meta
            .blocks
            .as_slice()
            .to_device_on(device_ctx)
            .expect("G2 static block table H2D");
        let d_initial_memory = initial_memory
            .to_device_on(device_ctx)
            .expect("G2 initial-memory descriptors H2D");
        let expected_kinds = host
            .specs
            .iter()
            .map(|spec| G2ExpectedKindV1 {
                kind: spec.kind as u32,
                air_idx: u32::try_from(spec.air_idx).expect("G2 AIR index exceeds u32"),
                count: u32::try_from(spec.count).expect("G2 AIR count exceeds u32"),
            })
            .collect::<Vec<_>>();
        let d_expected_kinds = expected_kinds
            .as_slice()
            .to_device_on(device_ctx)
            .expect("G2 expected-kind table H2D");
        let d_expected_opaque = host
            .opaque
            .as_slice()
            .to_device_on(device_ctx)
            .expect("G2 expected opaque table H2D");
        if let Some(timer) = h2d_timer {
            timer.finish("wire_h2d", segment_id, host.segment.transfer_byte_len());
        }
        let residual_capacity = host
            .segment
            .header_acquire()
            .expect("G2 committed header before device decode")
            .residual_event_count as usize;
        let d_opaque_residual = DeviceBuffer::<u8>::with_capacity_on(
            residual_capacity
                * std::mem::size_of::<openvm_circuit::arch::rvr::DeltaMemoryLogEntry>(),
            device_ctx,
        );
        let d_opaque_residual_count = DeviceBuffer::<u32>::with_capacity_on(1, device_ctx);
        d_opaque_residual_count
            .fill_zero_on(device_ctx)
            .expect("G2 opaque-residual count clear");
        let d_program_frequencies =
            DeviceBuffer::<u32>::with_capacity_on(host.program_frequency_count, device_ctx);
        d_program_frequencies
            .fill_zero_on(device_ctx)
            .expect("G2 program-frequency clear");
        let num_airs = host
            .specs
            .iter()
            .map(|spec| spec.air_idx + 1)
            .max()
            .unwrap_or(0);
        let mut outputs = HashMap::new();
        let mut descs = vec![DeltaAirOutputDesc::default(); num_airs];
        let mut sorted_specs = host.specs.clone();
        sorted_specs.sort_unstable_by_key(|spec| spec.air_idx);
        let mut sorted_start = 0usize;
        for spec in &sorted_specs {
            let stride = spec.kind.wire_size();
            if spec.count != 0 {
                let buffer = Arc::new(DeviceBuffer::<u8>::with_capacity_on(
                    spec.count * stride,
                    device_ctx,
                ));
                descs[spec.air_idx] = DeltaAirOutputDesc {
                    base: buffer.as_ptr() as u64,
                    count: spec.count as u32,
                    stride: stride as u32,
                    sorted_start: sorted_start as u32,
                    kind: spec.kind as u32,
                };
                outputs.insert(spec.kind, buffer);
            }
            sorted_start += spec.count;
        }
        assert_eq!(sorted_start, host.total_record_count);
        let d_descs = descs
            .as_slice()
            .to_device_on(device_ctx)
            .expect("G2 output descriptors H2D");
        let touched_capacity = host
            .total_record_count
            .saturating_mul(4)
            .saturating_add(residual_capacity)
            .saturating_add(32);
        let d_touched_output = if touched_capacity == 0 {
            DeviceBuffer::new()
        } else {
            DeviceBuffer::<u32>::with_capacity_on(
                touched_capacity * DEVICE_TOUCHED_RECORD_WORDS,
                device_ctx,
            )
        };
        let d_touched_count = DeviceBuffer::<u32>::with_capacity_on(1, device_ctx);
        let d_opaque_prev_timestamps = if residual_capacity == 0 {
            DeviceBuffer::<u32>::new()
        } else {
            DeviceBuffer::<u32>::with_capacity_on(residual_capacity, device_ctx)
        };
        let d_opaque_prev_values = if residual_capacity == 0 {
            DeviceBuffer::<u64>::new()
        } else {
            DeviceBuffer::<u64>::with_capacity_on(residual_capacity, device_ctx)
        };
        let d_error = DeviceBuffer::<u32>::with_capacity_on(1, device_ctx);
        d_error.fill_zero_on(device_ctx).expect("G2 error clear");
        let g2_predecode_timer = CudaStageTimer::start(device_ctx);
        unsafe {
            crate::cuda_abi::rvr_g2_cuda::predecode(
                &d_wire,
                host.segment.byte_len(),
                header.run_count as usize,
                header.instruction_count as usize,
                &d_fingerprint,
                &d_blocks,
                &d_table,
                pc_base,
                &d_initial_memory,
                host.initial_timestamp,
                &d_expected_kinds,
                &d_expected_opaque,
                &d_program_frequencies,
                host.total_record_count,
                &d_opaque_residual,
                &d_opaque_residual_count,
                &d_descs,
                &d_touched_output,
                &d_touched_count,
                &d_opaque_prev_timestamps,
                &d_opaque_prev_values,
                &d_error,
                device_ctx.stream.as_raw(),
            )
            .expect("CUDA G2 wire expansion launch");
        }
        if let Some(timer) = g2_predecode_timer {
            timer.finish("g2_predecode", segment_id, host.total_record_count);
        }
        let error = d_error.to_host_on(device_ctx).expect("G2 error D2H")[0];
        assert_eq!(error, 0, "CUDA G2 wire validation error {error}");
        if !host.program_frequency_reference.is_empty() {
            let actual = d_program_frequencies
                .to_host_on(device_ctx)
                .expect("G2 program-frequency oracle D2H");
            assert_eq!(
                actual.as_slice(),
                host.program_frequency_reference.as_slice(),
                "G2 device program frequencies differ byte-for-byte from host execution"
            );
        }
        let opaque_residual_count = d_opaque_residual_count
            .to_host_on(device_ctx)
            .expect("G2 opaque-residual count D2H")[0] as usize;
        assert!(
            opaque_residual_count <= residual_capacity,
            "G2 opaque-residual count overflow"
        );
        let touched_count = d_touched_count
            .to_host_on(device_ctx)
            .expect("G2 touched count D2H")[0] as usize;
        assert!(
            touched_count <= touched_capacity,
            "G2 touched-memory count overflow"
        );
        if !host.oracle_expected.is_empty() {
            for spec in &host.specs {
                if spec.count == 0 {
                    continue;
                }
                let expected = host.oracle_expected.get(&spec.air_idx).unwrap_or_else(|| {
                    panic!("G2 device oracle omitted expected AIR {}", spec.air_idx)
                });
                let actual = outputs
                    .get(&spec.kind)
                    .unwrap_or_else(|| panic!("G2 device oracle omitted {:?}", spec.kind))
                    .to_host_on(device_ctx)
                    .expect("G2 consumer-byte oracle D2H");
                assert_eq!(
                    actual.as_slice(),
                    expected.as_slice(),
                    "G2 device consumer bytes differ from CPU reference for AIR {} ({:?})",
                    spec.air_idx,
                    spec.kind
                );
            }
            eprintln!(
                "OPENVM_RVR_G2_DEVICE_ORACLE_PASS=1 airs={} records={}",
                host.oracle_expected.len(),
                host.total_record_count
            );
        }
        drop(host);

        let device = DeviceDeltaSegment {
            outputs,
            touched_memory: Some(DeviceTouchedMemory {
                records: d_touched_output,
                num_records: touched_count,
            }),
            program_frequencies: Some(DeviceProgramFrequencies {
                frequencies: d_program_frequencies,
            }),
            g2_segment_id: Some(segment_id),
        };
        *self.g2_device.lock().unwrap() = Some(device);
        true
    }

    /// Return the device-resident compact buffer for one chip.
    #[cfg(all(feature = "cuda", feature = "rvr"))]
    pub fn device_delta_records(
        &self,
        kind: DeltaAirKind,
        device_ctx: &GpuDeviceCtx,
    ) -> Option<Arc<DeviceBuffer<u8>>> {
        self.ensure_device_delta_segment(device_ctx, None);
        if self.delta_device.lock().unwrap().is_none() {
            self.ensure_device_g2_segment(device_ctx, None);
        }
        self.delta_device
            .lock()
            .unwrap()
            .as_ref()
            .and_then(|device| device.outputs.get(&kind).cloned())
            .or_else(|| {
                self.g2_device
                    .lock()
                    .unwrap()
                    .as_ref()
                    .and_then(|device| device.outputs.get(&kind).cloned())
            })
    }

    #[cfg(all(feature = "cuda", feature = "rvr"))]
    pub(crate) fn g2_segment_id(&self) -> Option<u32> {
        self.g2_device
            .lock()
            .unwrap()
            .as_ref()
            .and_then(|device| device.g2_segment_id)
    }
}

#[cfg(all(feature = "cuda", feature = "rvr"))]
impl DeviceTouchedMemoryProvider for RvrGpuDecodeState {
    fn take_device_touched_memory(
        &self,
        device_ctx: &GpuDeviceCtx,
        initial_memory: &[DeviceInitialMemory],
    ) -> Option<DeviceTouchedMemory> {
        self.ensure_device_delta_segment(device_ctx, Some(initial_memory));
        if self.delta_device.lock().unwrap().is_none() {
            self.ensure_device_g2_segment(device_ctx, Some(initial_memory));
        }
        self.delta_device
            .lock()
            .unwrap()
            .as_mut()
            .and_then(|device| device.touched_memory.take())
            .or_else(|| {
                self.g2_device
                    .lock()
                    .unwrap()
                    .as_mut()
                    .and_then(|device| device.touched_memory.take())
            })
    }
}

#[cfg(all(feature = "cuda", feature = "rvr"))]
impl DeviceProgramFrequenciesProvider for RvrGpuDecodeState {
    fn take_device_program_frequencies(
        &self,
        device_ctx: &GpuDeviceCtx,
        initial_memory: &[DeviceInitialMemory],
    ) -> Option<DeviceProgramFrequencies> {
        self.ensure_device_delta_segment(device_ctx, Some(initial_memory));
        if self.delta_device.lock().unwrap().is_none() {
            self.ensure_device_g2_segment(device_ctx, Some(initial_memory));
        }
        self.delta_device
            .lock()
            .unwrap()
            .as_mut()
            .and_then(|device| device.program_frequencies.take())
            .or_else(|| {
                self.g2_device
                    .lock()
                    .unwrap()
                    .as_mut()
                    .and_then(|device| device.program_frequencies.take())
            })
    }
}

/// The operand-table entry for one instruction, if its opcode belongs to a
/// wire format with a device decode kernel. THE shared list: the table
/// builder and the per-segment bind both consult this, so the compact air
/// set and the table can never drift apart. Derivations mirror the host
/// inline assemblers exactly (alu3 via `derive_base_alu_u16_operands`).
#[cfg(feature = "rvr")]
fn gpu_decode_entry<F: PrimeField32>(
    instruction: &Instruction<F>,
) -> Option<(DeviceOperandEntry, DeltaAirKind)> {
    use openvm_instructions::riscv::RV64_REGISTER_AS;
    let opcode = instruction.opcode.as_usize();

    if opcode == BaseAluImmOpcode::ADDI.global_opcode_usize() {
        let operands = derive_addi_operands(instruction);
        let mut flags = OPERAND_FLAG_RS2_IMM;
        if ((operands.immediate >> 11) & 1) != 0 {
            flags |= OPERAND_FLAG_RS2_IMM_SIGN;
        }
        return Some((
            DeviceOperandEntry {
                a: operands.rd_ptr,
                b: operands.rs1_ptr,
                c: operands.immediate,
                flags,
                local_opcode: BaseAluImmOpcode::ADDI as u8,
                air_idx: u8::MAX,
                access_pattern: DeviceDeltaAccessPattern::AddI as u8,
                filtered_index: u32::MAX,
            },
            DeltaAirKind::AddI,
        ));
    }

    // alu3 over the BaseAluU16 adapter: AddSub, LessThan, ShiftLogical, SRA.
    let alu_u16_local = if opcode == BaseAluOpcode::ADD.global_opcode_usize()
        || opcode == BaseAluOpcode::SUB.global_opcode_usize()
    {
        Some((opcode - BaseAluOpcode::CLASS_OFFSET) as u8)
    } else if opcode == LessThanOpcode::SLT.global_opcode_usize()
        || opcode == LessThanOpcode::SLTU.global_opcode_usize()
    {
        Some((opcode - LessThanOpcode::CLASS_OFFSET) as u8)
    } else if opcode == ShiftOpcode::SLL.global_opcode_usize()
        || opcode == ShiftOpcode::SRL.global_opcode_usize()
        || opcode == ShiftOpcode::SRA.global_opcode_usize()
    {
        Some((opcode - ShiftOpcode::CLASS_OFFSET) as u8)
    } else {
        None
    };
    if let Some(local_opcode) = alu_u16_local {
        let operands = derive_base_alu_u16_operands(instruction);
        let mut flags = 0u8;
        if operands.rs2_as != RV64_REGISTER_AS as u8 {
            flags |= OPERAND_FLAG_RS2_IMM;
        }
        if operands.rs2_imm_sign {
            flags |= OPERAND_FLAG_RS2_IMM_SIGN;
        }
        let kind = if opcode == BaseAluOpcode::ADD.global_opcode_usize()
            || opcode == BaseAluOpcode::SUB.global_opcode_usize()
        {
            DeltaAirKind::AddSub
        } else if opcode == LessThanOpcode::SLT.global_opcode_usize()
            || opcode == LessThanOpcode::SLTU.global_opcode_usize()
        {
            DeltaAirKind::LessThan
        } else if opcode == ShiftOpcode::SRA.global_opcode_usize() {
            DeltaAirKind::ShiftRightArithmetic
        } else {
            DeltaAirKind::ShiftLogical
        };
        let access_pattern = if kind == DeltaAirKind::AddSub {
            DeviceDeltaAccessPattern::Alu3Reg
        } else {
            DeviceDeltaAccessPattern::Alu3
        };
        return Some((
            DeviceOperandEntry {
                a: operands.rd_ptr,
                b: operands.rs1_ptr,
                c: operands.rs2,
                flags,
                local_opcode,
                air_idx: u8::MAX,
                access_pattern: access_pattern as u8,
                filtered_index: u32::MAX,
            },
            kind,
        ));
    }

    // alu3 over the W u16 adapter: BaseAluW and ShiftW. The operand
    // derivation is identical to the full-word u16 adapter; the W kernels
    // additionally recover the high halves and result-sign witness.
    let alu_w_local = if opcode == BaseAluWOpcode::ADDW.global_opcode_usize() {
        Some(0)
    } else if opcode == BaseAluWOpcode::SUBW.global_opcode_usize() {
        Some(1)
    } else if opcode == ShiftWOpcode::SLLW.global_opcode_usize() {
        Some(0)
    } else if opcode == ShiftWOpcode::SRLW.global_opcode_usize() {
        Some(1)
    } else if opcode == ShiftWOpcode::SRAW.global_opcode_usize() {
        Some(0)
    } else {
        None
    };
    if let Some(local_opcode) = alu_w_local {
        let operands = derive_base_alu_u16_operands(instruction);
        let mut flags = 0u8;
        if operands.rs2_as != RV64_REGISTER_AS as u8 {
            flags |= OPERAND_FLAG_RS2_IMM;
        }
        if operands.rs2_imm_sign {
            flags |= OPERAND_FLAG_RS2_IMM_SIGN;
        }
        let kind = if opcode == BaseAluWOpcode::ADDW.global_opcode_usize()
            || opcode == BaseAluWOpcode::SUBW.global_opcode_usize()
        {
            DeltaAirKind::AddSubW
        } else if opcode == ShiftWOpcode::SRAW.global_opcode_usize() {
            DeltaAirKind::ShiftWRightArithmetic
        } else {
            DeltaAirKind::ShiftWLogical
        };
        return Some((
            DeviceOperandEntry {
                a: operands.rd_ptr,
                b: operands.rs1_ptr,
                c: operands.rs2,
                flags,
                local_opcode,
                air_idx: u8::MAX,
                access_pattern: DeviceDeltaAccessPattern::Alu3 as u8,
                filtered_index: u32::MAX,
            },
            kind,
        ));
    }

    // branch2: BranchEqual (BEQ/BNE), BranchLessThan (BLT/BLTU/BGE/BGEU).
    let branch_local = if opcode == BranchEqualOpcode::BEQ.global_opcode_usize()
        || opcode == BranchEqualOpcode::BNE.global_opcode_usize()
    {
        Some((opcode - BranchEqualOpcode::CLASS_OFFSET) as u8)
    } else if opcode == BranchLessThanOpcode::BLT.global_opcode_usize()
        || opcode == BranchLessThanOpcode::BLTU.global_opcode_usize()
        || opcode == BranchLessThanOpcode::BGE.global_opcode_usize()
        || opcode == BranchLessThanOpcode::BGEU.global_opcode_usize()
    {
        Some((opcode - BranchLessThanOpcode::CLASS_OFFSET) as u8)
    } else {
        None
    };
    if let Some(local_opcode) = branch_local {
        let kind = if opcode == BranchEqualOpcode::BEQ.global_opcode_usize()
            || opcode == BranchEqualOpcode::BNE.global_opcode_usize()
        {
            DeltaAirKind::BranchEqual
        } else {
            DeltaAirKind::BranchLessThan
        };
        return Some((
            DeviceOperandEntry {
                a: instruction.a.as_canonical_u32(),
                b: instruction.b.as_canonical_u32(),
                c: instruction.c.as_canonical_u32(),
                flags: 0,
                local_opcode,
                air_idx: u8::MAX,
                access_pattern: DeviceDeltaAccessPattern::Branch2 as u8,
                filtered_index: u32::MAX,
            },
            kind,
        ));
    }

    // wr1: JalLui (conditional write via f) and Auipc (always written).
    if opcode == Rv64JalLuiOpcode::JAL.global_opcode_usize()
        || opcode == Rv64JalLuiOpcode::LUI.global_opcode_usize()
    {
        let mut flags = 0u8;
        if !instruction.f.is_zero() {
            flags |= OPERAND_FLAG_WRITE_ENABLED;
        }
        if opcode == Rv64JalLuiOpcode::JAL.global_opcode_usize() {
            flags |= OPERAND_FLAG_IS_JAL;
        }
        return Some((
            DeviceOperandEntry {
                a: instruction.a.as_canonical_u32(),
                b: 0,
                c: instruction.c.as_canonical_u32(),
                flags,
                local_opcode: 0,
                air_idx: u8::MAX,
                access_pattern: DeviceDeltaAccessPattern::Wr1 as u8,
                filtered_index: u32::MAX,
            },
            DeltaAirKind::JalLui,
        ));
    }
    if opcode == Rv64AuipcOpcode::AUIPC.global_opcode_usize() {
        return Some((
            DeviceOperandEntry {
                a: instruction.a.as_canonical_u32(),
                b: 0,
                c: instruction.c.as_canonical_u32(),
                flags: OPERAND_FLAG_WRITE_ENABLED,
                local_opcode: 0,
                air_idx: u8::MAX,
                access_pattern: DeviceDeltaAccessPattern::Wr1Always as u8,
                filtered_index: u32::MAX,
            },
            DeltaAirKind::Auipc,
        ));
    }

    // alu3 over the Mult adapter: Mul, MulH class, DivRem class.
    let mult_local = if opcode == MulOpcode::MUL.global_opcode_usize() {
        Some(0u8)
    } else if opcode == MulHOpcode::MULH.global_opcode_usize()
        || opcode == MulHOpcode::MULHSU.global_opcode_usize()
        || opcode == MulHOpcode::MULHU.global_opcode_usize()
    {
        Some((opcode - MulHOpcode::CLASS_OFFSET) as u8)
    } else if opcode == DivRemOpcode::DIV.global_opcode_usize()
        || opcode == DivRemOpcode::DIVU.global_opcode_usize()
        || opcode == DivRemOpcode::REM.global_opcode_usize()
        || opcode == DivRemOpcode::REMU.global_opcode_usize()
    {
        Some((opcode - DivRemOpcode::CLASS_OFFSET) as u8)
    } else {
        None
    };
    // alu3 over the MultW adapter: MulW, DivRemW class. The class-local index
    // doubles as the device W-result kind for DivRemW (the kernel passes 0xFF
    // for MULW itself).
    let mult_w_local = if opcode == MulWOpcode::MULW.global_opcode_usize() {
        Some(0u8)
    } else if opcode == DivRemWOpcode::DIVW.global_opcode_usize()
        || opcode == DivRemWOpcode::DIVUW.global_opcode_usize()
        || opcode == DivRemWOpcode::REMW.global_opcode_usize()
        || opcode == DivRemWOpcode::REMUW.global_opcode_usize()
    {
        Some((opcode - DivRemWOpcode::CLASS_OFFSET) as u8)
    } else {
        None
    };
    if let Some(local_opcode) = mult_local.or(mult_w_local) {
        let kind = if opcode == MulOpcode::MUL.global_opcode_usize() {
            DeltaAirKind::Mul
        } else if (MulHOpcode::CLASS_OFFSET..MulHOpcode::CLASS_OFFSET + 3).contains(&opcode) {
            DeltaAirKind::MulH
        } else if (DivRemOpcode::CLASS_OFFSET..DivRemOpcode::CLASS_OFFSET + 4).contains(&opcode) {
            DeltaAirKind::DivRem
        } else if opcode == MulWOpcode::MULW.global_opcode_usize() {
            DeltaAirKind::MulW
        } else {
            DeltaAirKind::DivRemW
        };
        return Some((
            DeviceOperandEntry {
                a: instruction.a.as_canonical_u32(),
                b: instruction.b.as_canonical_u32(),
                c: instruction.c.as_canonical_u32(),
                flags: 0,
                local_opcode,
                air_idx: u8::MAX,
                access_pattern: DeviceDeltaAccessPattern::Alu3Reg as u8,
                filtered_index: u32::MAX,
            },
            kind,
        ));
    }

    // alu3 over the byte adapter: Bitwise (XOR/OR/AND).
    if opcode == BaseAluOpcode::XOR.global_opcode_usize()
        || opcode == BaseAluOpcode::OR.global_opcode_usize()
        || opcode == BaseAluOpcode::AND.global_opcode_usize()
    {
        let mut flags = 0u8;
        if instruction.e.as_canonical_u32() != RV64_REGISTER_AS {
            flags |= OPERAND_FLAG_RS2_IMM;
            if instruction.c.as_canonical_u32() & (1 << 23) != 0 {
                flags |= OPERAND_FLAG_RS2_IMM_SIGN;
            }
        }
        return Some((
            DeviceOperandEntry {
                a: instruction.a.as_canonical_u32(),
                b: instruction.b.as_canonical_u32(),
                c: instruction.c.as_canonical_u32(),
                flags,
                local_opcode: (opcode - BaseAluOpcode::CLASS_OFFSET) as u8,
                air_idx: u8::MAX,
                access_pattern: DeviceDeltaAccessPattern::Alu3 as u8,
                filtered_index: u32::MAX,
            },
            DeltaAirKind::Bitwise,
        ));
    }

    // alu3 over the LoadStore adapter (zero-ext loads, stores, sign-ext loads,
    // and AS=3 REVEAL stores sharing the same AIR).
    let ls_local = opcode
        .checked_sub(Rv64LoadStoreOpcode::CLASS_OFFSET)
        .and_then(|local| Rv64LoadStoreOpcode::from_repr(local).map(|_| local as u8));
    if let Some(local_opcode) = ls_local {
        let mem_as = instruction.e.as_canonical_u32();
        if mem_as != openvm_instructions::riscv::RV64_MEMORY_AS
            && mem_as != openvm_instructions::PUBLIC_VALUES_AS
        {
            return None;
        }
        let op = Rv64LoadStoreOpcode::from_repr(local_opcode as usize)
            .expect("local index from the enum itself");
        let mut flags = 0u8;
        if !instruction.f.is_zero() {
            flags |= OPERAND_FLAG_WRITE_ENABLED;
        }
        if !instruction.g.is_zero() {
            flags |= OPERAND_FLAG_LS_IMM_SIGN;
        }
        if mem_as == openvm_instructions::PUBLIC_VALUES_AS {
            flags |= OPERAND_FLAG_LS_PUBLIC_VALUES;
        }
        match op {
            Rv64LoadStoreOpcode::LOADD
            | Rv64LoadStoreOpcode::LOADWU
            | Rv64LoadStoreOpcode::LOADHU
            | Rv64LoadStoreOpcode::LOADBU => flags |= OPERAND_FLAG_LS_IS_LOAD,
            Rv64LoadStoreOpcode::LOADB => flags |= OPERAND_FLAG_LS_IS_BYTE,
            Rv64LoadStoreOpcode::LOADW => flags |= OPERAND_FLAG_LS_IS_WORD,
            _ => {}
        }
        let kind = match op {
            Rv64LoadStoreOpcode::LOADBU => DeltaAirKind::LoadByte,
            Rv64LoadStoreOpcode::LOADHU => DeltaAirKind::LoadHalfword,
            Rv64LoadStoreOpcode::LOADWU => DeltaAirKind::LoadWord,
            Rv64LoadStoreOpcode::LOADD => DeltaAirKind::LoadDoubleword,
            Rv64LoadStoreOpcode::STOREB => DeltaAirKind::StoreByte,
            Rv64LoadStoreOpcode::STOREH => DeltaAirKind::StoreHalfword,
            Rv64LoadStoreOpcode::STOREW => DeltaAirKind::StoreWord,
            Rv64LoadStoreOpcode::STORED => DeltaAirKind::StoreDoubleword,
            Rv64LoadStoreOpcode::LOADB => DeltaAirKind::LoadSignExtendByte,
            Rv64LoadStoreOpcode::LOADH => DeltaAirKind::LoadSignExtendHalfword,
            Rv64LoadStoreOpcode::LOADW => DeltaAirKind::LoadSignExtendWord,
        };
        let pattern = if matches!(
            op,
            Rv64LoadStoreOpcode::STORED
                | Rv64LoadStoreOpcode::STOREW
                | Rv64LoadStoreOpcode::STOREH
                | Rv64LoadStoreOpcode::STOREB
        ) {
            DeviceDeltaAccessPattern::Store
        } else {
            DeviceDeltaAccessPattern::Load
        };
        return Some((
            DeviceOperandEntry {
                a: instruction.a.as_canonical_u32(),
                b: instruction.b.as_canonical_u32(),
                c: instruction.c.as_canonical_u32(),
                flags,
                local_opcode,
                air_idx: u8::MAX,
                access_pattern: pattern as u8,
                filtered_index: u32::MAX,
            },
            kind,
        ));
    }

    // rw1: Jalr (conditional write via f, imm sign via g).
    if opcode == Rv64JalrOpcode::JALR.global_opcode_usize() {
        let mut flags = 0u8;
        if !instruction.f.is_zero() {
            flags |= OPERAND_FLAG_WRITE_ENABLED;
        }
        if !instruction.g.is_zero() {
            flags |= OPERAND_FLAG_JALR_IMM_SIGN;
        }
        return Some((
            DeviceOperandEntry {
                a: instruction.a.as_canonical_u32(),
                b: instruction.b.as_canonical_u32(),
                c: instruction.c.as_canonical_u32(),
                flags,
                local_opcode: 0,
                air_idx: u8::MAX,
                access_pattern: DeviceDeltaAccessPattern::Rw1 as u8,
                filtered_index: u32::MAX,
            },
            DeltaAirKind::Jalr,
        ));
    }

    None
}

/// Registry callback used by `compile_preflight` to persist the exact table
/// input and decoder kind before the timed proving path begins.
#[cfg(feature = "rvr")]
pub(crate) fn gpu_decode_precompute<F: PrimeField32>(
    instruction: &Instruction<F>,
) -> Option<RvrDeltaDecodeInfo> {
    gpu_decode_entry(instruction).map(|(entry, kind)| RvrDeltaDecodeInfo {
        entry,
        kind: kind as u8,
    })
}

#[cfg(feature = "rvr")]
fn sign_extend_word(value: u32) -> u64 {
    value as i32 as i64 as u64
}

#[cfg(feature = "rvr")]
fn divrem_result(local_opcode: u8, b: u64, c: u64) -> Option<u64> {
    Some(match local_opcode {
        0 => {
            let (b, c) = (b as i64, c as i64);
            if c == 0 {
                u64::MAX
            } else if b == i64::MIN && c == -1 {
                b as u64
            } else {
                (b / c) as u64
            }
        }
        1 => {
            if c == 0 {
                u64::MAX
            } else {
                b / c
            }
        }
        2 => {
            let (b, c) = (b as i64, c as i64);
            if c == 0 {
                b as u64
            } else if b == i64::MIN && c == -1 {
                0
            } else {
                (b % c) as u64
            }
        }
        3 => {
            if c == 0 {
                b
            } else {
                b % c
            }
        }
        _ => return None,
    })
}

#[cfg(feature = "rvr")]
fn divrem_w_result(local_opcode: u8, b: u64, c: u64) -> Option<u64> {
    let (b, c) = (b as u32, c as u32);
    let result = match local_opcode {
        0 => {
            let (b, c) = (b as i32, c as i32);
            if c == 0 {
                u32::MAX
            } else if b == i32::MIN && c == -1 {
                b as u32
            } else {
                (b / c) as u32
            }
        }
        1 => {
            if c == 0 {
                u32::MAX
            } else {
                b / c
            }
        }
        2 => {
            let (b, c) = (b as i32, c as i32);
            if c == 0 {
                b as u32
            } else if b == i32::MIN && c == -1 {
                0
            } else {
                (b % c) as u32
            }
        }
        3 => {
            if c == 0 {
                b
            } else {
                b % c
            }
        }
        _ => return None,
    };
    Some(sign_extend_word(result))
}

/// Extension-owned execution semantics for the 24-byte delta record. The CPU
/// oracle and CUDA predecoder independently apply this same exhaustive kind
/// table; unknown kinds/opcodes return `None` and fail the delta route closed.
#[cfg(feature = "rvr")]
pub(crate) fn delta_post_write_value<F: PrimeField32>(
    instruction: &Instruction<F>,
    pc: u32,
    v1: u64,
    v2: u64,
) -> Option<u64> {
    let (entry, kind) = gpu_decode_entry(instruction)?;
    Some(match kind {
        DeltaAirKind::AddI => {
            if entry.local_opcode != BaseAluImmOpcode::ADDI as u8 {
                return None;
            }
            v1.wrapping_add(v2)
        }
        DeltaAirKind::AddSub => match entry.local_opcode {
            0 => v1.wrapping_add(v2),
            1 => v1.wrapping_sub(v2),
            _ => return None,
        },
        DeltaAirKind::Bitwise => match entry.local_opcode {
            2 => v1 ^ v2,
            3 => v1 | v2,
            4 => v1 & v2,
            _ => return None,
        },
        DeltaAirKind::LessThan => match entry.local_opcode {
            0 => u64::from((v1 as i64) < (v2 as i64)),
            1 => u64::from(v1 < v2),
            _ => return None,
        },
        DeltaAirKind::ShiftLogical => match entry.local_opcode {
            0 => v1.wrapping_shl((v2 & 63) as u32),
            1 => v1.wrapping_shr((v2 & 63) as u32),
            _ => return None,
        },
        DeltaAirKind::ShiftRightArithmetic => {
            if entry.local_opcode != 2 {
                return None;
            }
            ((v1 as i64) >> (v2 & 63)) as u64
        }
        DeltaAirKind::AddSubW => {
            let result = match entry.local_opcode {
                0 => (v1 as u32).wrapping_add(v2 as u32),
                1 => (v1 as u32).wrapping_sub(v2 as u32),
                _ => return None,
            };
            sign_extend_word(result)
        }
        DeltaAirKind::ShiftWLogical => {
            let result = match entry.local_opcode {
                0 => (v1 as u32).wrapping_shl((v2 & 31) as u32),
                1 => (v1 as u32).wrapping_shr((v2 & 31) as u32),
                _ => return None,
            };
            sign_extend_word(result)
        }
        DeltaAirKind::ShiftWRightArithmetic => {
            if entry.local_opcode != 0 {
                return None;
            }
            sign_extend_word(((v1 as u32 as i32) >> (v2 & 31)) as u32)
        }
        DeltaAirKind::LoadByte
        | DeltaAirKind::LoadHalfword
        | DeltaAirKind::LoadWord
        | DeltaAirKind::LoadDoubleword
        | DeltaAirKind::StoreByte
        | DeltaAirKind::StoreHalfword
        | DeltaAirKind::StoreWord
        | DeltaAirKind::StoreDoubleword
        | DeltaAirKind::LoadSignExtendByte
        | DeltaAirKind::LoadSignExtendHalfword
        | DeltaAirKind::LoadSignExtendWord => {
            if entry.flags & OPERAND_FLAG_WRITE_ENABLED == 0 {
                return None;
            }
            let imm = entry.c as u16;
            let offset = if entry.flags & OPERAND_FLAG_LS_IMM_SIGN != 0 {
                i32::from(imm as i16) as u32
            } else {
                u32::from(imm)
            };
            let byte_offset = (v1 as u32).wrapping_add(offset) & 7;
            let load_width = match entry.local_opcode {
                0 => 8,
                1 | 8 => 1,
                2 | 9 => 2,
                3 | 10 => 4,
                _ => return None,
            };
            if load_width > 1 && byte_offset + load_width > 8 {
                return Some(v2);
            }
            let shift = byte_offset * 8;
            let shifted = v2 >> shift;
            match entry.local_opcode {
                0 => shifted,
                1 => shifted as u8 as u64,
                2 => shifted as u16 as u64,
                3 => shifted as u32 as u64,
                8 => shifted as u8 as i8 as i64 as u64,
                9 => shifted as u16 as i16 as i64 as u64,
                10 => shifted as u32 as i32 as i64 as u64,
                _ => return None,
            }
        }
        DeltaAirKind::BranchEqual | DeltaAirKind::BranchLessThan => return None,
        DeltaAirKind::JalLui => {
            if entry.flags & OPERAND_FLAG_WRITE_ENABLED == 0 {
                return None;
            }
            if entry.flags & OPERAND_FLAG_IS_JAL != 0 {
                u64::from(pc.wrapping_add(DEFAULT_PC_STEP))
            } else {
                (entry.c << 12) as i32 as i64 as u64
            }
        }
        DeltaAirKind::Jalr => {
            if entry.flags & OPERAND_FLAG_WRITE_ENABLED == 0 {
                return None;
            }
            u64::from(pc.wrapping_add(DEFAULT_PC_STEP))
        }
        DeltaAirKind::Auipc => {
            let offset = (entry.c << 8) as i32 as i64 as u64;
            u64::from(pc).wrapping_add(offset)
        }
        DeltaAirKind::Mul => v1.wrapping_mul(v2),
        DeltaAirKind::MulH => match entry.local_opcode {
            0 => (((v1 as i64 as i128) * (v2 as i64 as i128)) >> 64) as u64,
            1 => (((v1 as i64 as i128) * (v2 as u128 as i128)) >> 64) as u64,
            2 => (((v1 as u128) * (v2 as u128)) >> 64) as u64,
            _ => return None,
        },
        DeltaAirKind::MulW => sign_extend_word((v1 as u32).wrapping_mul(v2 as u32)),
        DeltaAirKind::DivRem => divrem_result(entry.local_opcode, v1, v2)?,
        DeltaAirKind::DivRemW => divrem_w_result(entry.local_opcode, v1, v2)?,
        DeltaAirKind::HintStore => return None,
    })
}

#[cfg(feature = "rvr")]
fn host_table_from_precomputed(
    compiled_identity: Arc<Vec<bool>>,
    precomputed: &RvrDeltaDecodePrecompute,
) -> HostOperandTable {
    let kind_to_air = precomputed
        .kind_to_air
        .iter()
        .map(|&(kind, air)| {
            (
                DeltaAirKind::from_repr(kind)
                    .unwrap_or_else(|| panic!("unknown persisted delta decoder kind {kind}")),
                air,
            )
        })
        .collect();
    HostOperandTable {
        compiled_identity,
        pc_base: precomputed.pc_base,
        entries: Arc::clone(&precomputed.entries),
        kind_to_air,
    }
}

#[cfg(all(feature = "rvr", debug_assertions))]
fn assert_host_tables_equal(precomputed: &HostOperandTable, lazy: &HostOperandTable) {
    assert_eq!(precomputed.pc_base, lazy.pc_base);
    assert_eq!(precomputed.entries, lazy.entries);
    assert_eq!(precomputed.kind_to_air, lazy.kind_to_air);
}

#[cfg(feature = "rvr")]
fn build_operand_table<F: PrimeField32>(
    exe: &VmExe<F>,
    compiled_identity: Arc<Vec<bool>>,
    pc_to_air_idx: &[Option<usize>],
) -> HostOperandTable {
    let program = &exe.program;
    let mut entries = vec![
        DeviceOperandEntry {
            air_idx: u8::MAX,
            access_pattern: u8::MAX,
            filtered_index: u32::MAX,
            ..DeviceOperandEntry::default()
        };
        program.instructions_and_debug_infos.len()
    ];
    let mut filtered_index = 0u32;
    for (slot_idx, slot) in program.instructions_and_debug_infos.iter().enumerate() {
        let Some((instruction, _)) = slot else {
            continue;
        };
        entries[slot_idx].filtered_index = filtered_index;
        filtered_index = filtered_index
            .checked_add(1)
            .expect("filtered program index exceeds u32 ABI");
        if compiled_identity.get(slot_idx).copied().unwrap_or(false) {
            if let Some((mut entry, _)) = gpu_decode_entry(instruction) {
                let air_idx = pc_to_air_idx
                    .get(slot_idx)
                    .copied()
                    .flatten()
                    .expect("device-decodable instruction must map to an AIR");
                entry.air_idx = u8::try_from(air_idx).expect("delta device AIR index exceeds u8");
                entry.filtered_index = entries[slot_idx].filtered_index;
                entries[slot_idx] = entry;
            }
        }
    }
    let _ = DEFAULT_PC_STEP; // index = (from_pc - pc_base) / DEFAULT_PC_STEP
    let kind_to_air = classify_kind_to_air(exe, pc_to_air_idx, &compiled_identity);
    HostOperandTable {
        compiled_identity,
        pc_base: program.pc_base,
        entries: Arc::new(entries),
        kind_to_air,
    }
}

#[cfg(feature = "rvr")]
fn classify_kind_to_air<F: PrimeField32>(
    exe: &VmExe<F>,
    pc_to_air_idx: &[Option<usize>],
    inline_pc_slots: &[bool],
) -> HashMap<DeltaAirKind, usize> {
    let mut kind_to_air = HashMap::new();
    let mut air_to_kind = HashMap::new();
    let mut tainted = HashSet::new();
    for (slot_idx, slot) in exe.program.instructions_and_debug_infos.iter().enumerate() {
        let Some((instruction, _)) = slot else {
            continue;
        };
        let Some(air_idx) = pc_to_air_idx.get(slot_idx).copied().flatten() else {
            continue;
        };
        if !inline_pc_slots.get(slot_idx).copied().unwrap_or(false) {
            tainted.insert(air_idx);
            continue;
        }
        let Some((_, kind)) = gpu_decode_entry(instruction) else {
            tainted.insert(air_idx);
            continue;
        };
        if let Some(previous) = kind_to_air.insert(kind, air_idx) {
            assert_eq!(
                previous, air_idx,
                "delta kind {kind:?} maps to multiple AIRs"
            );
        }
        if let Some(previous) = air_to_kind.insert(air_idx, kind) {
            assert_eq!(previous, kind, "AIR {air_idx} maps to multiple delta kinds");
        }
    }
    // An AIR is decoded only if EVERY pc routed to it is device-decodable.
    // LoadStore explicitly supports both AS=2 ordinary and AS=3 REVEAL rows.
    kind_to_air.retain(|_, air| !tainted.contains(air));
    kind_to_air
}
