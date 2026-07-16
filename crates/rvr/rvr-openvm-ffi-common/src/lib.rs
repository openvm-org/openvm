//! Memory and host-I/O helpers shared by Rust RVR extensions.
//!
//! Extension staticlibs call the C functions defined in `rvr_ext_wrappers.c`
//! for execution-mode-specific memory access.
//! Register access stays in generated C; resolved values are passed to extension FFI entry points.
//!
//! The `state` parameter is always an opaque `*mut c_void` pointing
//! to the C `RvState` struct.

use std::ffi::c_void;

use openvm_platform::{memory::MEM_SIZE, WORD_SIZE};

// TODO(dedup): make a common crate that is imported by rvr and openvm-circuit
// ── OpenVM preflight tracer ABI ─────────────────────────────────────

/// Preflight tracer kind ID. Existing IDs: MeteredCost=10, Metered=11, Pure=12.
pub const PREFLIGHT_TRACER_KIND: u32 = 13;

/// Initial timestamp for preflight memory logs. Matches `TracingMemory`:
/// `INITIAL_TIMESTAMP + 1`.
pub const PREFLIGHT_INITIAL_TIMESTAMP: u32 = 1;

pub const PREFLIGHT_MEMORY_KIND_READ: u8 = 0;
pub const PREFLIGHT_MEMORY_KIND_WRITE: u8 = 1;
pub const PREFLIGHT_MEMORY_KIND_TOUCH: u8 = 2;

/// Program chronology packs its writer-complete guard into bit 0 of the
/// naturally four-byte-aligned OpenVM pc. The remaining fields are the
/// timestamp and post-write value, giving a 16-byte side-log entry.
pub const PREFLIGHT_PROGRAM_LOG_ENTRY_SIZE: usize = 16;
pub const PREFLIGHT_PROGRAM_LOG_ENTRY_ALIGN: usize = 8;
// R1: MemoryLogEntry is self-contained — it carries `prev_timestamp` (the
// block's previous-access timestamp, from the C timestamp shadow) and
// `prev_value` (the block's value before this access), so the host side no
// longer replays the log to recover them.
pub const PREFLIGHT_MEMORY_LOG_ENTRY_SIZE: usize = 40;
pub const PREFLIGHT_MEMORY_LOG_ENTRY_ALIGN: usize = 8;
/// Delta-only residual-memory wire. The full host log remains the ABI for
/// CPU, compact, partial-direct, and non-delta routes.
pub const PREFLIGHT_DELTA_MEMORY_LOG_ENTRY_SIZE: usize = 24;
pub const PREFLIGHT_DELTA_MEMORY_LOG_ENTRY_ALIGN: usize = 8;
// R1: the tracer gained per-address-space timestamp-shadow pointers, a
// public-values base pointer (for `prev_value` reads on reveal writes), a
// touched-block buffer, and its length/capacity counters.
// R3: the tracer gained a `chip_records` pointer for inline compact per-chip
// record emission. ZG2 adds a direct execution-frequency buffer after it so
// final-form records need no duplicate program-log row or host frequency scan.
// Device replay adds bounded custom-memory scratch metadata, and M-CPOOL
// appends first-touch cursors for lazy counter-table reset. M-NATIVE then
// appends an optional profiling-detail pointer. It is null unless the
// profiling-only OPENVM_RVR_NATIVE_DETAIL route is selected. L1 appends
// device-aux patch/reference side logs and a dirty-page bitmap for the
// all-direct delta route; their host counterparts remain absent elsewhere.
// L2 appends block-run/device-chronology buffers used to reconstruct the
// filtered program frequencies without per-instruction host accounting.
pub const PREFLIGHT_TRACER_DATA_SIZE: usize = 272;
pub const PREFLIGHT_TRACER_DATA_ALIGN: usize = 8;
/// One entry in the preflight touched-block buffer: the address space and the
/// block-aligned byte address of a block touched (for the first time) this
/// segment. The host finalizes `touched_memory` from this list.
pub const PREFLIGHT_TOUCHED_BLOCK_SIZE: usize = 16;
pub const PREFLIGHT_TOUCHED_BLOCK_ALIGN: usize = 8;

// R3/R4: inline records. `ChipRecordBuf` is one per-chip record-buffer
// descriptor (base pointer + byte cursor + byte capacity + record stride);
// the tracer holds an array of `chip_counts_len` of them. The cursor advances
// by `stride` per record, so record i sits at `base + i*stride`; compact-wire
// buffers use stride == packed record size, arena-native buffers use the
// arena row/record pitch. A null `base` means the chip is not migrated to
// inline records (it uses the verbose memory log instead).
pub const PREFLIGHT_CHIP_RECORD_BUF_SIZE: usize = 32;
pub const PREFLIGHT_CHIP_RECORD_BUF_ALIGN: usize = 8;
/// Generated C writes this target in final consumer form. Its duplicate
/// program-log row and all host adoption/assembly are suppressed.
pub const PREFLIGHT_CHIP_RECORD_FLAG_DIRECT_FINAL: u32 = 1;
/// A generated record reservation exceeded or mismatched its target. The
/// host rejects the segment before any decode/proof work.
pub const PREFLIGHT_CHIP_RECORD_FLAG_OVERFLOW: u32 = 2;
/// Every timestamp-bearing access for this direct-final record remains in the
/// residual memory log. Delta chronology therefore needs no duplicate program
/// log entry for the instruction.
pub const PREFLIGHT_CHIP_RECORD_FLAG_RESIDUAL_MEMORY_CHRONOLOGY: u32 = 4;
/// `ChipRecordBuf::len` is a packed byte cursor and `core_off` is repurposed
/// as the emitted row count. Only custom variable-row direct-final targets
/// set this flag.
pub const PREFLIGHT_CHIP_RECORD_FLAG_VARIABLE_ROWS: u32 = 8;
/// Variable-row records reserve `rows * stride` bytes (Matrix flavor) rather
/// than their packed byte size (Dense flavor).
pub const PREFLIGHT_CHIP_RECORD_FLAG_VARIABLE_ROW_STRIDE: u32 = 16;
/// The shared `memory_log` pointer targets the delta-only 24-byte residual
/// schema instead of the full 40-byte host `MemoryLogEntry` schema.
pub const PREFLIGHT_CHIP_RECORD_FLAG_COMPACT_RESIDUAL_MEMORY: u32 = 32;
/// The all-direct CUDA delta route reconstructs memory predecessor data and
/// first-touch state on device. The host tracer emits chronology only.
pub const PREFLIGHT_CHIP_RECORD_FLAG_DEVICE_AUX: u32 = 64;
/// Test/oracle mode: retain the legacy host predecessor computation solely for
/// a fail-hard full-vector comparison with the device reconstruction.
pub const PREFLIGHT_CHIP_RECORD_FLAG_DEVICE_AUX_ORACLE: u32 = 128;
/// The all-direct CUDA delta route emits one compact descriptor per executed
/// basic block. Device predecode expands those runs into exact program order
/// and filtered execution frequencies.
pub const PREFLIGHT_CHIP_RECORD_FLAG_DEVICE_CHRONOLOGY: u32 = 256;
/// One device-chronology block-run descriptor.
pub const PREFLIGHT_PROGRAM_RUN_ENTRY_SIZE: usize = 16;
pub const PREFLIGHT_PROGRAM_RUN_ENTRY_ALIGN: usize = 4;
/// One oracle-only expanded program-chronology entry `(pc, filtered_index)`.
pub const PREFLIGHT_DEVICE_PROGRAM_ENTRY_SIZE: usize = 8;
pub const PREFLIGHT_DEVICE_PROGRAM_ENTRY_ALIGN: usize = 4;
/// Byte size of one compact base-ALU AddSub record as stored by the preflight
/// tracer (R3 L1+L5): the dynamic witness only — from_pc, from_timestamp, the
/// three access prev_timestamps, the old rd block, and the b/c operand limbs.
/// Program-redundant operands (rd_ptr/rs1_ptr/rs2/rs2_as/rs2_imm_sign/
/// local_opcode) are re-derived from the instruction at `from_pc` during host
/// record assembly. The riscv circuit asserts this against its compact-record
/// mirror (see its rvr record-ABI guard).
pub const PREFLIGHT_ADDSUB_RECORD_SIZE: usize = 44;

/// Byte size of one compact branch record (R3): the 2-read-no-write dynamic
/// witness — from_pc, from_timestamp, two read prev_timestamps, and the two
/// operand values. Used by the BranchEq/BranchLt shapes.
pub const PREFLIGHT_BRANCH2_RECORD_SIZE: usize = 32;

/// Byte size of one compact write-only record (R3): from_pc, from_timestamp,
/// the (conditional) rd write prev_timestamp, and the old rd block. Used by
/// the JalLui and Auipc shapes; for a suppressed write (rd = x0) the write
/// fields are zero and the host uses the instruction's enable flag.
pub const PREFLIGHT_WR1_RECORD_SIZE: usize = 20;

/// Byte size of one compact read+conditional-write record (R3): from_pc,
/// from_timestamp, the rs1 read prev_timestamp and value, and the
/// (conditional) rd write prev_timestamp and old rd block. Used by Jalr.
pub const PREFLIGHT_RW1_RECORD_SIZE: usize = 32;

/// Stage-2 chronological delta record. One record covers every inline-routed
/// instruction, independent of AIR: pc + from-timestamp + three dynamic u64
/// values. The three access previous-timestamps are implicit in chronological
/// stream order and are reconstructed by the decoder.
pub const PREFLIGHT_DELTA_RECORD_SIZE: usize = 24;

// ── G2 private compact wire v1 ─────────────────────────────────────────

/// Private transport version. This is a prover-side wire contract only; it is
/// deliberately independent of AIR, VK, and public record schemas.
pub const G2_WIRE_VERSION_V1: u16 = 1;
pub const G2_SEGMENT_MAGIC_V1: [u8; 8] = *b"OVMG2W1\0";
pub const G2_SEGMENT_HEADER_V1_SIZE: usize = 64;
pub const G2_LANE_DESC_V1_SIZE: usize = 32;
pub const G2_WIRE_ALIGNMENT: usize = 128;

pub const G2_FLAG_COMMITTED: u16 = 1 << 0;
pub const G2_FLAG_U32_POINTERS: u16 = 1 << 1;
pub const G2_FLAG_SPLIT_RESIDUAL: u16 = 1 << 2;
pub const G2_FLAG_STATIC_BLOCK_IDS: u16 = 1 << 3;
pub const G2_FLAGS_V1: u16 =
    G2_FLAG_U32_POINTERS | G2_FLAG_SPLIT_RESIDUAL | G2_FLAG_STATIC_BLOCK_IDS;

pub const G2_LANE_FLAG_REQUIRED: u32 = 1 << 0;
pub const G2_LANE_FLAG_ATOMIC_GROUP: u32 = 1 << 1;
pub const G2_LANE_FLAG_OPAQUE_FINAL: u32 = 1 << 2;

pub const G2_ENCODING_FIXED_LE: u8 = 0;
pub const G2_ENCODING_OPAQUE_FINAL: u8 = 1;
pub const G2_LANE_RUN_BLOCK_ID: u16 = 0x0001;
pub const G2_LANE_RESIDUAL_CTRL: u16 = 0x0080;
pub const G2_LANE_RESIDUAL_TAG: u16 = 0x0081;
pub const G2_LANE_RESIDUAL_VALUE: u16 = 0x0082;
/// Exact residual-event span for each opaque custom instruction, in program
/// order. This removes any need to infer custom boundaries from event values.
pub const G2_LANE_OPAQUE_EVENT_COUNT: u16 = 0x0083;
pub const G2_GROUP_LOAD_STORE: u32 = 1;
pub const G2_GROUP_RESIDUAL: u32 = 2;
pub const G2_LOAD_STORE_KINDS: [u8; 11] = [8, 9, 20, 21, 22, 23, 24, 25, 26, 27, 28];
pub const G2_PHASE2B_TWO_LANE_KINDS: [u8; 15] =
    [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 15, 16, 17, 18, 19];
pub const G2_ZERO_ARITY_KINDS: [u8; 2] = [12, 14];
pub const G2_PRODUCER_RUN_SLOT: usize = 0;
pub const G2_PRODUCER_RESIDUAL_CTRL_SLOT: usize = 1;
pub const G2_PRODUCER_RESIDUAL_TAG_SLOT: usize = 2;
pub const G2_PRODUCER_RESIDUAL_VALUE_SLOT: usize = 3;
pub const G2_PRODUCER_ADDI_SLOT: usize = 4;
pub const G2_PRODUCER_LOAD_STORE_SLOT_BASE: usize = 5;
pub const G2_PRODUCER_PHASE2B_SLOT_BASE: usize =
    G2_PRODUCER_LOAD_STORE_SLOT_BASE + G2_LOAD_STORE_KINDS.len() * 2;
pub const G2_PRODUCER_JALR_SLOT: usize =
    G2_PRODUCER_PHASE2B_SLOT_BASE + G2_PHASE2B_TWO_LANE_KINDS.len() * 2;
pub const G2_PRODUCER_OPAQUE_EVENT_COUNT_SLOT: usize = G2_PRODUCER_JALR_SLOT + 1;
pub const G2_PRODUCER_LANE_COUNT: usize = G2_PRODUCER_OPAQUE_EVENT_COUNT_SLOT + 1;
/// `0x0100 + 2 * DeltaAirKind::AddI`.
pub const G2_LANE_ADDI_V0: u16 = 0x013a;

pub const fn g2_lane_v0(kind: u8) -> u16 {
    0x0100 + 2 * kind as u16
}

pub const fn g2_lane_v1(kind: u8) -> u16 {
    g2_lane_v0(kind) + 1
}

pub const fn g2_load_store_producer_slot(kind: u8, value_lane: bool) -> Option<usize> {
    let mut index = 0;
    while index < G2_LOAD_STORE_KINDS.len() {
        if G2_LOAD_STORE_KINDS[index] == kind {
            return Some(G2_PRODUCER_LOAD_STORE_SLOT_BASE + index * 2 + value_lane as usize);
        }
        index += 1;
    }
    None
}

pub const fn g2_standard_producer_slot(kind: u8, value_lane: bool) -> Option<usize> {
    if kind == 29 {
        return if value_lane {
            None
        } else {
            Some(G2_PRODUCER_ADDI_SLOT)
        };
    }
    if let Some(slot) = g2_load_store_producer_slot(kind, value_lane) {
        return Some(slot);
    }
    let mut index = 0;
    while index < G2_PHASE2B_TWO_LANE_KINDS.len() {
        if G2_PHASE2B_TWO_LANE_KINDS[index] == kind {
            return Some(G2_PRODUCER_PHASE2B_SLOT_BASE + index * 2 + value_lane as usize);
        }
        index += 1;
    }
    if kind == 13 && !value_lane {
        return Some(G2_PRODUCER_JALR_SLOT);
    }
    None
}

pub const fn g2_standard_lane_width(kind: u8, value_lane: bool) -> Option<u8> {
    let _slot = match g2_standard_producer_slot(kind, value_lane) {
        Some(slot) => slot,
        None => return None,
    };
    if !value_lane && (kind == 13 || g2_load_store_producer_slot(kind, false).is_some()) {
        Some(4)
    } else {
        Some(8)
    }
}

/// Frozen 64-byte segment header. Producers write this with `flags` lacking
/// [`G2_FLAG_COMMITTED`], then publish that bit with a release store.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct G2SegmentHeaderV1 {
    pub magic: [u8; 8],
    pub version: u16,
    pub header_bytes: u16,
    pub lane_count: u16,
    pub flags: u16,
    pub segment_id: u32,
    pub instruction_count: u32,
    pub run_count: u32,
    pub residual_event_count: u32,
    pub schema_fingerprint: [u8; 32],
}

/// Frozen 32-byte lane descriptor. Descriptors are unique and sorted by
/// `kind`; every payload offset and the total segment size are 128-aligned.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct G2LaneDescV1 {
    pub kind: u16,
    pub elem_width: u8,
    pub encoding: u8,
    pub flags: u32,
    pub count: u32,
    pub payload_bytes: u32,
    pub offset: u64,
    pub group_id: u32,
    pub reserved: u32,
}

/// One mutable lane cursor passed to generated C. The slot order is fixed by
/// the `G2_PRODUCER_*` constants; the committed descriptor table remains
/// sorted by public lane kind and omits zero-count standard lanes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct G2ProducerLaneV1 {
    pub offset: u64,
    pub len: u32,
    pub cap: u32,
}

/// Mutable producer ABI passed to generated C. Payloads are written directly
/// into their final backing positions; Rust only validates lane cursors and
/// publishes the header/descriptors after native execution.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct G2ProducerV1 {
    pub base: *mut u8,
    pub capacity: u64,
    pub lanes: *mut G2ProducerLaneV1,
    pub lane_count: u32,
    pub instruction_count: u32,
    pub overflow: u32,
    pub reserved: u32,
}

pub const G2_PRODUCER_LANE_V1_SIZE: usize = 16;
pub const G2_PRODUCER_V1_SIZE: usize = 40;
pub const G2_PRODUCER_V1_ALIGN: usize = 8;

const _: () = {
    assert!(std::mem::size_of::<G2SegmentHeaderV1>() == G2_SEGMENT_HEADER_V1_SIZE);
    assert!(std::mem::align_of::<G2SegmentHeaderV1>() == 4);
    assert!(std::mem::offset_of!(G2SegmentHeaderV1, flags) == 14);
    assert!(std::mem::offset_of!(G2SegmentHeaderV1, schema_fingerprint) == 32);
    assert!(std::mem::size_of::<G2LaneDescV1>() == G2_LANE_DESC_V1_SIZE);
    assert!(std::mem::align_of::<G2LaneDescV1>() == 8);
    assert!(std::mem::offset_of!(G2LaneDescV1, offset) == 16);
    assert!(std::mem::size_of::<G2ProducerLaneV1>() == G2_PRODUCER_LANE_V1_SIZE);
    assert!(std::mem::align_of::<G2ProducerLaneV1>() == 8);
    assert!(std::mem::size_of::<G2ProducerV1>() == G2_PRODUCER_V1_SIZE);
    assert!(std::mem::align_of::<G2ProducerV1>() == G2_PRODUCER_V1_ALIGN);
};

const _: () = assert!(MEM_SIZE / WORD_SIZE <= u32::MAX as usize);

extern "C" {
    fn peek_mem_u64_wrapper(state: *mut c_void, addr: u64) -> u64;
    fn peek_mem_u64_range_wrapper(
        state: *mut c_void,
        base_addr: u64,
        out: *mut u64,
        num_words: u32,
    );
    fn read_mem_u64_range_wrapper(
        state: *mut c_void,
        base_addr: u64,
        out: *mut u64,
        num_words: u32,
    );
    fn write_mem_u64_range_wrapper(
        state: *mut c_void,
        base_addr: u64,
        vals: *const u64,
        num_words: u32,
    );
    fn record_page_access_u64_range_wrapper(
        state: *mut c_void,
        base_addr: u64,
        num_words: u32,
        addr_space: u32,
    );

    // ── Memory access (u64-word ranges, trace-only) ───────────────────
    pub fn trace_rd_mem_u64_range_wrapper(
        state: *mut c_void,
        base_addr: u64,
        vals: *const u64,
        num_words: u32,
    );
    pub fn trace_wr_mem_u64_range_wrapper(
        state: *mut c_void,
        base_addr: u64,
        vals: *const u64,
        num_words: u32,
    );

    // ── Chip cost ─────────────────────────────────────────────────────
    pub fn trace_chip_wrapper(state: *mut c_void, chip_idx: u32, count: u32);
    pub fn trace_timestamp_wrapper(state: *mut c_void);
    // ── Hint stream (for extension phantom instructions) ──────────────
    /// Replace the hint stream contents. Forwarded through `openvm_io.c`
    /// to `OpenVmIoState.hint_stream` via the host callback mechanism.
    pub fn ext_hint_stream_set(data: *const u8, len: u64);
}

// ── Zero-copy lane reinterpretation ─────────────────────────────────────

// We assume a little-endian host (x86_64, aarch64). The u64→u32 reinterpret
// helper below depends on that.
const _: () = assert!(
    cfg!(target_endian = "little"),
    "rvr-openvm-ext-ffi-common assumes a little-endian host"
);

/// Reinterpret a `[u64]` slice as a `[u32]` slice (twice as many elements),
/// mutably. Zero-copy on LE: the low u32 of each u64 comes first, then the
/// high u32. Mutating through the returned slice mutates the underlying u64
/// lanes.
#[inline(always)]
pub fn u64s_as_u32s_mut(lanes: &mut [u64]) -> &mut [u32] {
    let len = lanes.len() * 2;
    // SAFETY: u64 alignment (8) >= u32 alignment (4); total bytes match;
    // POD types; LE host (asserted above) makes the layout (lo, hi).
    unsafe { std::slice::from_raw_parts_mut(lanes.as_mut_ptr().cast::<u32>(), len) }
}

/// View u64 words as their little-endian bytes without copying.
#[inline(always)]
pub fn u64s_as_bytes(words: &[u64]) -> &[u8] {
    let len = core::mem::size_of_val(words);
    // SAFETY: u8 has alignment 1, the byte length matches the source slice,
    // and the supported little-endian host layout matches guest word order.
    unsafe { std::slice::from_raw_parts(words.as_ptr().cast::<u8>(), len) }
}

/// View mutable u64 words as their little-endian bytes without copying.
#[inline(always)]
pub fn u64s_as_bytes_mut(words: &mut [u64]) -> &mut [u8] {
    let len = core::mem::size_of_val(words);
    // SAFETY: the same layout argument as [`u64s_as_bytes`] applies, and the
    // mutable borrow keeps the returned byte slice exclusive.
    unsafe { std::slice::from_raw_parts_mut(words.as_mut_ptr().cast::<u8>(), len) }
}

// ── Ergonomic batched memory helpers ─────────────────────────────────────

#[inline(always)]
fn memory_word_count(len: usize) -> u32 {
    debug_assert!(len > 0, "memory word range must be non-empty");
    debug_assert!(
        len <= MEM_SIZE / WORD_SIZE,
        "memory word range must fit in guest memory"
    );
    len as u32
}

/// Read `out.len()` consecutive u64 words as VM memory accesses.
///
/// `out` must be non-empty; the mode-specific functions assume `num_words >= 1`
/// to avoid an underflow check on the hot path. Callers handling
/// guest-supplied dynamic sizes (e.g. xorin with `len = 0`) must guard
/// the empty case themselves.
///
/// # Safety
///
/// `state` must point to a valid `RvState`, and
/// `[base_addr, base_addr + out.len() * WORD_SIZE)` must fit in guest memory.
pub unsafe fn read_mem_words(state: *mut c_void, base_addr: u64, out: &mut [u64]) {
    let num_words = memory_word_count(out.len());
    read_mem_u64_range_wrapper(state, base_addr, out.as_mut_ptr(), num_words);
}

/// Write `vals` as VM memory accesses.
///
/// `vals` must be non-empty; see [`read_mem_words`] for rationale.
///
/// # Safety
///
/// `state` must point to a valid `RvState`, and
/// `[base_addr, base_addr + vals.len() * WORD_SIZE)` must fit in guest memory.
pub unsafe fn write_mem_words(state: *mut c_void, base_addr: u64, vals: &[u64]) {
    let num_words = memory_word_count(vals.len());
    write_mem_u64_range_wrapper(state, base_addr, vals.as_ptr(), num_words);
}

/// Read one u64 whose value affects execution but is not a VM memory access.
///
/// # Safety
/// `state` must point to a valid `RvState`, and `addr` must address one u64 in
/// guest memory.
pub unsafe fn peek_mem_u64(state: *mut c_void, addr: u64) -> u64 {
    peek_mem_u64_wrapper(state, addr)
}

/// Read guest words whose values affect execution but are not VM memory
/// accesses.
///
/// # Safety
/// `state` must point to a valid `RvState`. `out` must be non-empty, and its
/// guest-memory range must be valid.
pub unsafe fn peek_mem_words(state: *mut c_void, base_addr: u64, out: &mut [u64]) {
    let num_words = memory_word_count(out.len());
    peek_mem_u64_range_wrapper(state, base_addr, out.as_mut_ptr(), num_words);
}

/// Record the pages touched by `num_words` consecutive words in `addr_space`.
/// This is metering data only; it does not record the accessed values.
///
/// `num_words` must be `>= 1`; see [`read_mem_words`] for rationale.
///
/// # Safety
/// `state` must be a valid `RvState` pointer.
pub unsafe fn record_page_access_range(
    state: *mut c_void,
    base_addr: u64,
    num_words: u32,
    addr_space: u32,
) {
    debug_assert!(
        num_words >= 1,
        "record_page_access_range requires num_words >= 1"
    );
    record_page_access_u64_range_wrapper(state, base_addr, num_words, addr_space);
}

/// Advance the active tracer timestamp without recording a memory event.
///
/// # Safety
/// `state` must be a valid `RvState` pointer.
pub unsafe fn trace_timestamp_tick(state: *mut c_void) {
    trace_timestamp_wrapper(state);
}
