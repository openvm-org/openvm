//! G2 private compact-wire transport.
//!
//! The generated native producer writes lane payloads directly into this
//! backing. Finalization is deliberately O(lanes): it validates cursors,
//! writes the active descriptors and header, and release-publishes COMMITTED.

use std::{
    collections::BTreeMap,
    mem::{align_of, size_of},
    sync::atomic::{AtomicU16, AtomicU32, Ordering},
};

use rvr_openvm_ext_ffi_common::{
    g2_lane_v0, g2_lane_v1, g2_load_store_producer_slot, g2_standard_lane_width,
    g2_standard_producer_slot, G2_ENCODING_FIXED_LE, G2_ENCODING_OPAQUE_FINAL, G2_FLAGS_V1,
    G2_FLAG_COMMITTED, G2_GROUP_LOAD_STORE, G2_GROUP_RESIDUAL, G2_LANE_ADDI_V0,
    G2_LANE_DESC_V1_SIZE, G2_LANE_FLAG_ATOMIC_GROUP, G2_LANE_FLAG_OPAQUE_FINAL,
    G2_LANE_FLAG_REQUIRED, G2_LANE_OPAQUE_EVENT_COUNT, G2_LANE_RESIDUAL_CTRL, G2_LANE_RESIDUAL_TAG,
    G2_LANE_RESIDUAL_VALUE, G2_LANE_RUN_BLOCK_ID, G2_LOAD_STORE_KINDS, G2_PRODUCER_ADDI_SLOT,
    G2_PRODUCER_LANE_COUNT, G2_PRODUCER_OPAQUE_EVENT_COUNT_SLOT, G2_PRODUCER_RESIDUAL_CTRL_SLOT,
    G2_PRODUCER_RESIDUAL_TAG_SLOT, G2_PRODUCER_RESIDUAL_VALUE_SLOT, G2_PRODUCER_RUN_SLOT,
    G2_SEGMENT_HEADER_V1_SIZE, G2_SEGMENT_MAGIC_V1, G2_WIRE_ALIGNMENT, G2_WIRE_VERSION_V1,
};
pub use rvr_openvm_ext_ffi_common::{
    G2LaneDescV1, G2ProducerLaneV1, G2ProducerV1, G2SegmentHeaderV1,
};

use super::{RvrDeltaDecodeEntry, RvrDeltaDecodePrecompute, PREFLIGHT_ADDSUB_RECORD_SIZE};
use crate::arch::ExecutionError;

static NEXT_G2_SEGMENT_ID: AtomicU32 = AtomicU32::new(0);
const G2_MAX_OPAQUE_LANES: usize = 128;

pub(crate) fn next_segment_id() -> Result<u32, ExecutionError> {
    NEXT_G2_SEGMENT_ID
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
        .map_err(|_| g2_error("segment id exhausted its frozen u32 range"))
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RvrG2BlockEntryV1 {
    pub program_slot: u32,
    pub instruction_count: u32,
}

const _: () = assert!(size_of::<RvrG2BlockEntryV1>() == 8);

/// Host-only counts for standard kinds without a current-value lane. These
/// are summed once per entered run after native execution; they never expand
/// the generated block or the device block-table ABI.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RvrG2BlockHostCountsV1 {
    pub kind12: u32,
    pub kind14: u32,
    pub kind30: u32,
}

#[derive(Clone, Debug)]
pub struct RvrG2MetaV1 {
    /// Frozen wire schema/program fingerprint published in every segment.
    /// Deliberately independent of the generated producer's check policy.
    pub fingerprint: [u8; 32],
    /// Generated-C producer schema. This binds the block-span algorithm and
    /// checked/production policy without changing the G2 wire fingerprint.
    pub producer_schema_fingerprint: [u8; 32],
    pub emission_mode: u8,
    pub program_fingerprint: [u8; 32],
    pub block_fingerprint: [u8; 32],
    pub air_manifest_fingerprint: [u8; 32],
    pub blocks: std::sync::Arc<Vec<RvrG2BlockEntryV1>>,
    /// Aligned one-for-one with `blocks` after its program-slot sort.
    pub block_host_counts: std::sync::Arc<Vec<RvrG2BlockHostCountsV1>>,
    /// Sorted stable decoder-kind to global AIR bindings admitted by this
    /// executable. Phase 2b admits every fixed standard RV64 kind.
    pub air_bindings: std::sync::Arc<Vec<RvrG2AirBindingV1>>,
    /// Custom AIRs whose generated-C emitters write the established final
    /// dense record layout. Their payload is transported as an opaque lane
    /// and consumed unchanged; only the fingerprinted geometry is generic.
    pub opaque_bindings: std::sync::Arc<Vec<RvrG2OpaqueBindingV1>>,
}

impl RvrG2MetaV1 {
    pub fn checked_emission(&self) -> bool {
        // Frozen producer manifest value; see rvr_openvm::G2EmissionMode.
        self.emission_mode == 1
    }

    pub fn air_idx(&self, kind: u8) -> Option<usize> {
        self.air_bindings
            .binary_search_by_key(&kind, |binding| binding.kind)
            .ok()
            .map(|index| self.air_bindings[index].air_idx)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RvrG2AirBindingV1 {
    pub kind: u8,
    pub air_idx: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RvrG2OpaqueBindingV1 {
    pub air_idx: usize,
    pub geometry: super::ArenaNativeGeometry,
    pub max_residual_events_per_record: u32,
    pub air_identity_digest: [u8; 32],
    pub layout_digest: [u8; 32],
}

impl RvrG2OpaqueBindingV1 {
    pub fn lane_kind(self) -> u16 {
        0x8000
            | u16::try_from(self.air_idx)
                .expect("G2 opaque AIR index must fit the frozen 15-bit lane namespace")
    }
}

#[derive(Clone, Debug, Default)]
pub struct RvrG2CapacitiesV1 {
    pub run: u32,
    pub residual: u32,
    pub opaque_events: u32,
    /// Per `DeltaAirKind`; unsupported entries must remain zero.
    pub kinds: [u32; 31],
}

impl RvrG2CapacitiesV1 {
    /// Build the fail-closed producer capacities for one real metered segment.
    ///
    /// Standard producer lanes are mutually exclusive instruction families,
    /// so their capacities must come from the same segment shape rather than
    /// a componentwise maximum assembled from different segments. Residual
    /// chronology is tighter still: a standard instruction contributes at
    /// most two crossing-memory events, a narrow public-values store can add
    /// one more event, and HintStore and opaque extensions retain their
    /// floor-defined per-row fail-closed bounds.
    pub(crate) fn for_metered_segment(
        g2: &RvrG2MetaV1,
        trace_heights: &[u32],
        num_insns: u64,
    ) -> Result<Self, ExecutionError> {
        let program_capacity = usize::try_from(num_insns)
            .map_err(|_| g2_error("G2 instruction count exceeds usize"))?
            .checked_add(16)
            .ok_or_else(|| g2_error("G2 run capacity overflow"))?
            .max(64);
        let run =
            u32::try_from(program_capacity).map_err(|_| g2_error("G2 run capacity exceeds u32"))?;

        let mut capacities = Self {
            run,
            residual: 0,
            opaque_events: if g2.opaque_bindings.is_empty() {
                0
            } else {
                run
            },
            ..Default::default()
        };
        for binding in g2.air_bindings.iter() {
            capacities.kinds[binding.kind as usize] =
                trace_heights
                    .get(binding.air_idx)
                    .copied()
                    .ok_or_else(|| g2_error("G2 AIR binding exceeds metered trace heights"))?;
        }

        let hintstore_rows = capacities.kinds[30] as usize;
        // DeltaAirKind::{StoreByte, StoreHalfword, StoreWord}. StoreDoubleword
        // uses the ordinary two-event public-values path.
        const NARROW_STORE_KINDS: [usize; 3] = [23, 24, 25];
        let narrow_reveal_rows = NARROW_STORE_KINDS.iter().try_fold(0usize, |sum, &kind| {
            sum.checked_add(capacities.kinds[kind] as usize)
                .ok_or_else(|| g2_error("G2 narrow-store row count overflow"))
        })?;
        let mut residual_capacity = program_capacity
            .checked_mul(2)
            .and_then(|capacity| capacity.checked_add(narrow_reveal_rows))
            .and_then(|capacity| capacity.checked_add(hintstore_rows.checked_mul(3)?))
            .and_then(|capacity| capacity.checked_add(64))
            .ok_or_else(|| g2_error("G2 residual capacity overflow"))?;
        for binding in g2.opaque_bindings.iter() {
            let height = trace_heights
                .get(binding.air_idx)
                .copied()
                .ok_or_else(|| g2_error("G2 opaque AIR exceeds metered trace heights"))?;
            residual_capacity = (height as usize)
                .checked_mul(binding.max_residual_events_per_record as usize)
                .and_then(|capacity| residual_capacity.checked_add(capacity))
                .ok_or_else(|| g2_error("G2 opaque residual capacity overflow"))?;
        }
        capacities.residual = u32::try_from(residual_capacity)
            .map_err(|_| g2_error("G2 residual capacity exceeds u32"))?;
        Ok(capacities)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RvrG2AddIReferenceV1 {
    /// Established 44-byte `RvrAlu3Compact` consumer records.
    pub compact_records: Vec<u8>,
    pub final_registers: [u64; 32],
    pub final_timestamps: [u32; 32],
    pub final_timestamp: u32,
    pub expanded_program_slots: Vec<u32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RvrG2ReferenceV1 {
    /// Established compact consumer records, keyed by stable decoder kind.
    pub compact_records: BTreeMap<u8, Vec<u8>>,
    pub final_registers: [u64; 32],
    pub final_timestamps: BTreeMap<(u8, u32), u32>,
    pub final_blocks: BTreeMap<(u8, u32), u64>,
    pub final_timestamp: u32,
    pub expanded_program_slots: Vec<u32>,
}

/// Committed compact segment. The pooled backing is aligned before native
/// u64 lane stores; all public offsets remain segment-relative.
pub struct RvrG2SegmentV1 {
    backing: Option<Vec<u8>>,
    backing_offset: usize,
    backing_capacity: usize,
    pool: super::RvrPreflightBufferPool,
    backing_route: super::preflight_pool::G2BackingRoute,
    byte_len: usize,
    header_prefix_len: usize,
    committed_lanes: Vec<RvrG2CommittedLaneV1>,
}

#[derive(Clone, Copy, Debug)]
struct RvrG2CommittedLaneV1 {
    kind: u16,
    source_offset: Option<usize>,
    wire_offset: usize,
    payload_bytes: usize,
}

impl std::fmt::Debug for RvrG2SegmentV1 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RvrG2SegmentV1")
            .field("byte_len", &self.byte_len)
            .finish_non_exhaustive()
    }
}

impl RvrG2SegmentV1 {
    pub fn byte_len(&self) -> usize {
        self.byte_len
    }

    /// Bytes physically staged into the compact device wire. Opaque-final
    /// payloads already live in their owning custom arenas, so only their
    /// descriptors—not logical payload holes—are transferred here.
    pub fn transfer_byte_len(&self) -> usize {
        self.committed_lanes
            .iter()
            .filter(|lane| lane.source_offset.is_some())
            .map(|lane| lane.wire_offset + lane.payload_bytes)
            .fold(self.header_prefix_len, usize::max)
            .next_multiple_of(G2_WIRE_ALIGNMENT)
    }

    /// Return the compact wire as scatter/gather pieces. The generated C
    /// writes each lane once into its capacity-bounded staging range; the
    /// O(lanes) finalizer assigns tight wire offsets without copying payloads.
    /// CUDA uploads these pieces directly to their final device offsets.
    pub fn wire_parts(&self) -> impl Iterator<Item = (usize, &[u8])> {
        std::iter::once((0, self.backing_bytes_prefix(self.header_prefix_len))).chain(
            self.committed_lanes.iter().filter_map(|lane| {
                lane.source_offset.map(|source_offset| {
                    (
                        lane.wire_offset,
                        self.backing_bytes_range(source_offset, lane.payload_bytes),
                    )
                })
            }),
        )
    }

    fn backing_bytes_prefix(&self, len: usize) -> &[u8] {
        self.backing_bytes_range(0, len)
    }

    fn backing_bytes_range(&self, offset: usize, len: usize) -> &[u8] {
        let backing = self.backing.as_ref().expect("G2 backing already recycled");
        debug_assert!(offset + len <= self.backing_capacity);
        unsafe {
            // SAFETY: the backing is initialized before native execution and
            // every committed source range was bounds-checked by finalization.
            std::slice::from_raw_parts(backing.as_ptr().add(self.backing_offset + offset), len)
        }
    }

    pub fn header_acquire(&self) -> Result<G2SegmentHeaderV1, ExecutionError> {
        if self.header_prefix_len < G2_SEGMENT_HEADER_V1_SIZE
            || self.byte_len < G2_SEGMENT_HEADER_V1_SIZE
        {
            return Err(g2_error("segment is shorter than its v1 header"));
        }
        let bytes = self.backing_bytes_prefix(self.header_prefix_len);
        let flags = unsafe {
            // SAFETY: u128 backing is aligned, and offset 14 is u16-aligned.
            (&*bytes.as_ptr().add(14).cast::<AtomicU16>()).load(Ordering::Acquire)
        };
        if flags & G2_FLAG_COMMITTED == 0 {
            return Err(g2_error("segment is not committed"));
        }
        let header = unsafe {
            // SAFETY: the backing is suitably aligned and contains 64 bytes.
            bytes.as_ptr().cast::<G2SegmentHeaderV1>().read()
        };
        if header.flags != flags {
            return Err(g2_error("header flags changed during acquire"));
        }
        Ok(header)
    }

    pub fn validate(
        &self,
        expected_fingerprint: &[u8; 32],
    ) -> Result<Vec<G2LaneDescV1>, ExecutionError> {
        let bytes = self.backing_bytes_prefix(self.header_prefix_len);
        let header = self.header_acquire()?;
        let lane_count = header.lane_count as usize;
        if header.magic != G2_SEGMENT_MAGIC_V1
            || header.version != G2_WIRE_VERSION_V1
            || header.header_bytes as usize
                != G2_SEGMENT_HEADER_V1_SIZE + lane_count * G2_LANE_DESC_V1_SIZE
            || lane_count == 0
            || lane_count != self.committed_lanes.len()
            || header.flags != G2_FLAGS_V1 | G2_FLAG_COMMITTED
            || &header.schema_fingerprint != expected_fingerprint
            || !self.byte_len.is_multiple_of(G2_WIRE_ALIGNMENT)
            || header.header_bytes as usize > self.header_prefix_len
        {
            return Err(g2_error("header capability or schema binding mismatch"));
        }

        let payload_begin = align_up(header.header_bytes as usize, G2_WIRE_ALIGNMENT)?;
        let mut descs: Vec<G2LaneDescV1> = Vec::with_capacity(lane_count);
        for index in 0..lane_count {
            let desc = unsafe {
                // SAFETY: the validated header establishes the complete
                // descriptor table inside the segment.
                bytes
                    .as_ptr()
                    .add(G2_SEGMENT_HEADER_V1_SIZE + index * G2_LANE_DESC_V1_SIZE)
                    .cast::<G2LaneDescV1>()
                    .read()
            };
            if index != 0 && descs[index - 1].kind >= desc.kind {
                return Err(g2_error("lane descriptors are not unique and sorted"));
            }
            let committed = self
                .committed_lanes
                .iter()
                .find(|lane| lane.kind == desc.kind)
                .ok_or_else(|| g2_error("descriptor has no committed producer lane"))?;
            if let Some(spec) = lane_spec(desc.kind) {
                validate_desc(&desc, spec, self.byte_len, payload_begin)?;
            } else if desc.kind & 0x8000 != 0 {
                validate_opaque_desc(&desc, self.byte_len, payload_begin)?;
            } else {
                return Err(g2_error(format!("unknown lane kind {:#06x}", desc.kind)));
            }
            if committed.wire_offset != desc.offset as usize
                || committed.payload_bytes != desc.payload_bytes as usize
            {
                return Err(g2_error("descriptor differs from committed producer lane"));
            }
            if desc.kind == G2_LANE_RUN_BLOCK_ID && desc.count != header.run_count {
                return Err(g2_error("run descriptor/header count mismatch"));
            }
            if matches!(
                desc.kind,
                G2_LANE_RESIDUAL_CTRL | G2_LANE_RESIDUAL_TAG | G2_LANE_RESIDUAL_VALUE
            ) && desc.count != header.residual_event_count
            {
                return Err(g2_error("residual descriptor/header count mismatch"));
            }
            descs.push(desc);
        }
        if descs
            .first()
            .is_none_or(|desc| desc.kind != G2_LANE_RUN_BLOCK_ID)
        {
            return Err(g2_error("RUN_BLOCK_ID lane is required"));
        }
        let mut by_offset = descs.clone();
        by_offset.sort_unstable_by_key(|desc| desc.offset);
        for pair in by_offset.windows(2) {
            let left_end = pair[0].offset + u64::from(pair[0].payload_bytes);
            if left_end > pair[1].offset {
                return Err(g2_error("lane payloads overlap"));
            }
        }
        validate_atomic_members(&descs, header.residual_event_count)?;
        if header.instruction_count != 0 && header.run_count == 0 {
            return Err(g2_error("non-empty segment has no program runs"));
        }
        Ok(descs)
    }

    pub fn lane(&self, descs: &[G2LaneDescV1], kind: u16) -> Option<&[u8]> {
        descs
            .binary_search_by_key(&kind, |desc| desc.kind)
            .ok()
            .map(|index| self.lane_bytes(&descs[index]))
    }

    pub fn lane_bytes<'a>(&'a self, desc: &G2LaneDescV1) -> &'a [u8] {
        let lane = self
            .committed_lanes
            .iter()
            .find(|lane| lane.kind == desc.kind)
            .expect("validated descriptor must have a committed producer lane");
        self.backing_bytes_range(
            lane.source_offset
                .expect("opaque-final lanes are consumed by their existing arena owner"),
            lane.payload_bytes,
        )
    }
}

impl Drop for RvrG2SegmentV1 {
    fn drop(&mut self) {
        if let Some(backing) = self.backing.take() {
            self.pool
                .recycle_g2_backing(backing, self.backing_capacity, self.backing_route);
        }
    }
}

/// Oracle-only CPU decoder for the Phase-1 AddI scaffold. Production G2
/// consumers use the CUDA expansion; this function exists to byte-compare its
/// established compact-consumer form without introducing a CPU assembly pass.
pub fn decode_addi_reference_v1(
    segment: &RvrG2SegmentV1,
    meta: &RvrG2MetaV1,
    decode: &RvrDeltaDecodePrecompute,
    initial_registers: [u64; 32],
    initial_timestamp: u32,
) -> Result<RvrG2AddIReferenceV1, ExecutionError> {
    let descs = segment.validate(&meta.fingerprint)?;
    let run_lane = segment
        .lane(&descs, G2_LANE_RUN_BLOCK_ID)
        .ok_or_else(|| g2_error("missing RUN_BLOCK_ID lane"))?;
    let addi_lane = segment
        .lane(&descs, G2_LANE_ADDI_V0)
        .ok_or_else(|| g2_error("missing AddI lane"))?;
    let addi_count = descs
        .iter()
        .find(|desc| desc.kind == G2_LANE_ADDI_V0)
        .map(|desc| desc.count)
        .unwrap_or(0);
    let header = segment.header_acquire()?;
    let mut registers = initial_registers;
    registers[0] = 0;
    let mut timestamps = [0u32; 32];
    let mut timestamp = initial_timestamp;
    let mut addi_cursor = 0usize;
    let mut compact_records =
        Vec::with_capacity(addi_count as usize * PREFLIGHT_ADDSUB_RECORD_SIZE);
    let mut expanded_program_slots = Vec::with_capacity(header.instruction_count as usize);

    for run in run_lane.chunks_exact(4) {
        let program_slot = u32::from_le_bytes(run.try_into().expect("four-byte run lane"));
        let block = meta
            .blocks
            .binary_search_by_key(&program_slot, |entry| entry.program_slot)
            .ok()
            .map(|index| meta.blocks[index])
            .ok_or_else(|| g2_error(format!("run references unknown block slot {program_slot}")))?;
        for local in 0..block.instruction_count {
            let slot = program_slot
                .checked_add(local)
                .ok_or_else(|| g2_error("expanded program slot overflow"))?;
            expanded_program_slots.push(slot);
            let entry = decode
                .entries
                .get(slot as usize)
                .ok_or_else(|| g2_error(format!("expanded slot {slot} exceeds operand table")))?;
            if entry.air_idx == u8::MAX {
                continue;
            }
            if entry.access_pattern != 8 || Some(entry.air_idx as usize) != meta.air_idx(29) {
                return Err(g2_error(format!("slot {slot} is not a Phase-1 AddI entry")));
            }
            let rs1 = register_index(entry.b)?;
            let rd = register_index(entry.a)?;
            let payload_at = addi_cursor
                .checked_mul(8)
                .ok_or_else(|| g2_error("AddI cursor overflow"))?;
            let payload = addi_lane.get(payload_at..payload_at + 8).ok_or_else(|| {
                g2_error("expanded program chronology over-consumed the AddI lane")
            })?;
            let rs1_value = u64::from_le_bytes(payload.try_into().expect("eight-byte AddI lane"));
            if rs1_value != registers[rs1] {
                return Err(g2_error(format!(
                    "AddI payload mismatch at slot {slot}: lane={rs1_value:#x}, replay={:#x}",
                    registers[rs1]
                )));
            }
            let from_timestamp = timestamp;
            let read_prev_timestamp = timestamps[rs1];
            timestamps[rs1] = timestamp;
            timestamp = timestamp
                .checked_add(1)
                .ok_or_else(|| g2_error("timestamp overflow"))?;
            let write_prev_timestamp = timestamps[rd];
            let write_prev_data = registers[rd];
            timestamps[rd] = timestamp;
            timestamp = timestamp
                .checked_add(1)
                .ok_or_else(|| g2_error("timestamp overflow"))?;
            let immediate = sign_extend_12(entry.c);
            let result = rs1_value.wrapping_add(immediate);
            if rd != 0 {
                registers[rd] = result;
            }

            compact_records.extend_from_slice(&(decode.pc_base + slot * 4).to_le_bytes());
            compact_records.extend_from_slice(&from_timestamp.to_le_bytes());
            compact_records.extend_from_slice(&read_prev_timestamp.to_le_bytes());
            compact_records.extend_from_slice(&0u32.to_le_bytes());
            compact_records.extend_from_slice(&write_prev_timestamp.to_le_bytes());
            compact_records.extend_from_slice(&write_prev_data.to_le_bytes());
            compact_records.extend_from_slice(&rs1_value.to_le_bytes());
            compact_records.extend_from_slice(&immediate.to_le_bytes());
            addi_cursor += 1;
        }
    }
    if expanded_program_slots.len() != header.instruction_count as usize
        || addi_cursor != addi_count as usize
        || compact_records.len() != addi_cursor * PREFLIGHT_ADDSUB_RECORD_SIZE
    {
        return Err(g2_error(
            "expanded instruction or AddI cursor did not finish exactly",
        ));
    }
    Ok(RvrG2AddIReferenceV1 {
        compact_records,
        final_registers: registers,
        final_timestamps: timestamps,
        final_timestamp: timestamp,
        expanded_program_slots,
    })
}

/// Oracle-only G2 decoder. It walks program chronology on the CPU only in
/// tests and reconstructs the established compact consumer records.
pub fn decode_reference_v1(
    segment: &RvrG2SegmentV1,
    meta: &RvrG2MetaV1,
    decode: &RvrDeltaDecodePrecompute,
    initial_registers: [u64; 32],
    initial_blocks: &BTreeMap<(u8, u32), u64>,
    initial_timestamp: u32,
) -> Result<RvrG2ReferenceV1, ExecutionError> {
    let descs = segment.validate(&meta.fingerprint)?;
    let run_lane = segment
        .lane(&descs, G2_LANE_RUN_BLOCK_ID)
        .ok_or_else(|| g2_error("missing RUN_BLOCK_ID lane"))?;
    let residual_ctrl = segment.lane(&descs, G2_LANE_RESIDUAL_CTRL);
    let residual_tag = segment.lane(&descs, G2_LANE_RESIDUAL_TAG);
    let residual_value = segment.lane(&descs, G2_LANE_RESIDUAL_VALUE);
    let opaque_event_counts = segment.lane(&descs, G2_LANE_OPAQUE_EVENT_COUNT);
    let header = segment.header_acquire()?;
    let mut registers = initial_registers;
    registers[0] = 0;
    let mut timestamps = BTreeMap::new();
    let mut blocks = initial_blocks.clone();
    for (reg, &value) in registers.iter().enumerate() {
        blocks.insert((1, reg as u32 * 8), value);
    }
    let mut kind_v0_cursors = [0usize; 31];
    let mut kind_v1_cursors = [0usize; 31];
    let mut residual_cursor = 0usize;
    let mut opaque_event_cursor = 0usize;
    let mut timestamp = initial_timestamp;
    let mut compact_records = BTreeMap::<u8, Vec<u8>>::new();
    let mut expanded_program_slots = Vec::with_capacity(header.instruction_count as usize);

    for run in run_lane.chunks_exact(4) {
        let run_slot = u32::from_le_bytes(run.try_into().expect("four-byte run lane"));
        let block = meta
            .blocks
            .binary_search_by_key(&run_slot, |entry| entry.program_slot)
            .ok()
            .map(|index| meta.blocks[index])
            .ok_or_else(|| g2_error(format!("run references unknown block slot {run_slot}")))?;
        for local in 0..block.instruction_count {
            let slot = run_slot
                .checked_add(local)
                .ok_or_else(|| g2_error("expanded program slot overflow"))?;
            expanded_program_slots.push(slot);
            let entry = decode
                .entries
                .get(slot as usize)
                .ok_or_else(|| g2_error(format!("expanded slot {slot} exceeds operand table")))?;
            if entry.air_idx == u8::MAX {
                continue;
            }
            let binding = meta
                .air_bindings
                .iter()
                .find(|binding| binding.air_idx == entry.air_idx as usize);
            if binding.is_none() && entry.access_pattern == 11 {
                timestamp = timestamp
                    .checked_add(1)
                    .ok_or_else(|| g2_error("timestamp overflow"))?;
                continue;
            }
            if binding.is_none() && entry.access_pattern == 10 {
                let lanes = (
                    residual_ctrl.ok_or_else(|| g2_error("missing residual CTRL lane"))?,
                    residual_tag.ok_or_else(|| g2_error("missing residual TAG lane"))?,
                    residual_value.ok_or_else(|| g2_error("missing residual VALUE lane"))?,
                );
                let event_count = lane_u32(
                    opaque_event_counts
                        .ok_or_else(|| g2_error("missing opaque event-count lane"))?,
                    opaque_event_cursor,
                    "opaque event count",
                )? as usize;
                opaque_event_cursor += 1;
                if event_count == 0
                    || residual_cursor
                        .checked_add(event_count)
                        .is_none_or(|end| end > header.residual_event_count as usize)
                {
                    return Err(g2_error("opaque residual span is invalid"));
                }
                for _ in 0..event_count {
                    let (event_timestamp, address, tag, value) =
                        read_residual(lanes, residual_cursor)?
                            .ok_or_else(|| g2_error("opaque residual span over-consumed"))?;
                    if event_timestamp != timestamp {
                        return Err(g2_error("opaque residual timestamp is not contiguous"));
                    }
                    apply_residual_reference(
                        event_timestamp,
                        address,
                        tag,
                        value,
                        &mut registers,
                        &mut timestamps,
                        &mut blocks,
                    )?;
                    residual_cursor += 1;
                    timestamp = timestamp
                        .checked_add(1)
                        .ok_or_else(|| g2_error("timestamp overflow"))?;
                }
                continue;
            }
            let binding = binding
                .ok_or_else(|| g2_error(format!("slot {slot} references an unbound AIR")))?;
            let kind = binding.kind;
            let from_timestamp = timestamp;
            if kind == 30 && entry.access_pattern == 9 {
                let lanes = (
                    residual_ctrl.ok_or_else(|| g2_error("missing residual CTRL lane"))?,
                    residual_tag.ok_or_else(|| g2_error("missing residual TAG lane"))?,
                    residual_value.ok_or_else(|| g2_error("missing residual VALUE lane"))?,
                );
                let mem_ptr_reg = register_index(entry.b)?;
                let mem_ptr = u32::try_from(registers[mem_ptr_reg])
                    .map_err(|_| g2_error("HintStore memory pointer exceeds u32"))?;
                validate_residual(
                    lanes,
                    residual_cursor,
                    timestamp,
                    entry.b,
                    0x40,
                    Some(u64::from(mem_ptr)),
                )?;
                let mem_ptr_prev = touch(&mut timestamps, (1, entry.b), timestamp);
                residual_cursor += 1;
                let (num_words, num_words_ptr, num_words_prev) = if entry.local_opcode == 0 {
                    (1u32, u32::MAX, 0u32)
                } else {
                    let num_reg = register_index(entry.a)?;
                    let num_words = u32::try_from(registers[num_reg])
                        .map_err(|_| g2_error("HintStore word count exceeds u32"))?;
                    validate_residual(
                        lanes,
                        residual_cursor,
                        timestamp + 1,
                        entry.a,
                        0x40,
                        Some(u64::from(num_words)),
                    )?;
                    let previous = touch(&mut timestamps, (1, entry.a), timestamp + 1);
                    residual_cursor += 1;
                    (num_words, entry.a, previous)
                };
                if !(1..=1023).contains(&num_words) {
                    return Err(g2_error("HintStore word count is outside the frozen bound"));
                }
                for row in 0..num_words {
                    let row_timestamp = timestamp
                        .checked_add(3 * row)
                        .ok_or_else(|| g2_error("HintStore timestamp overflow"))?;
                    let address = mem_ptr
                        .checked_add(8 * row)
                        .ok_or_else(|| g2_error("HintStore address overflow"))?;
                    let data = validate_residual(
                        lanes,
                        residual_cursor,
                        row_timestamp + 2,
                        address,
                        0x45,
                        None,
                    )?;
                    residual_cursor += 1;
                    let key = (2, address);
                    let previous = blocks.get(&key).copied().unwrap_or(0);
                    let write_prev = touch(&mut timestamps, key, row_timestamp + 2);
                    blocks.insert(key, data);
                    let out = compact_records.entry(kind).or_default();
                    append_u32(out, decode.pc_base + slot * 4);
                    append_u32(out, timestamp);
                    append_u32(out, row);
                    append_u32(out, num_words);
                    append_u32(out, entry.b);
                    append_u32(out, mem_ptr);
                    append_u32(out, if row == 0 { mem_ptr_prev } else { 0 });
                    append_u32(out, num_words_ptr);
                    append_u32(out, if row == 0 { num_words_prev } else { 0 });
                    append_u32(out, write_prev);
                    append_u64(out, previous);
                    append_u64(out, data);
                    append_u64(out, 0);
                }
                timestamp = timestamp
                    .checked_add(3 * num_words)
                    .ok_or_else(|| g2_error("HintStore timestamp overflow"))?;
                continue;
            }
            let mut record = Vec::with_capacity(PREFLIGHT_ADDSUB_RECORD_SIZE);
            append_u32(&mut record, decode.pc_base + slot * 4);
            append_u32(&mut record, from_timestamp);

            if kind == 29 && entry.access_pattern == 8 {
                let lane = segment
                    .lane(&descs, G2_LANE_ADDI_V0)
                    .ok_or_else(|| g2_error("missing AddI lane"))?;
                let rs1 = register_index(entry.b)?;
                let rd = register_index(entry.a)?;
                let value = lane_u64(lane, kind_v0_cursors[kind as usize], "AddI")?;
                kind_v0_cursors[kind as usize] += 1;
                if value != registers[rs1] {
                    return Err(g2_error(format!("AddI value mismatch at slot {slot}")));
                }
                let read_prev = touch(&mut timestamps, (1, entry.b), timestamp);
                timestamp = timestamp
                    .checked_add(1)
                    .ok_or_else(|| g2_error("timestamp overflow"))?;
                let write_prev = touch(&mut timestamps, (1, entry.a), timestamp);
                let write_prev_value = registers[rd];
                timestamp = timestamp
                    .checked_add(1)
                    .ok_or_else(|| g2_error("timestamp overflow"))?;
                let immediate = sign_extend_12(entry.c);
                if rd != 0 {
                    registers[rd] = value.wrapping_add(immediate);
                    blocks.insert((1, entry.a), registers[rd]);
                }
                append_u32(&mut record, read_prev);
                append_u32(&mut record, 0);
                append_u32(&mut record, write_prev);
                append_u64(&mut record, write_prev_value);
                append_u64(&mut record, value);
                append_u64(&mut record, immediate);
            } else if matches!(kind, 0..=7 | 15..=19) && matches!(entry.access_pattern, 0 | 1) {
                let rs1 = register_index(entry.b)?;
                let rd = register_index(entry.a)?;
                let v0_lane = segment
                    .lane(&descs, g2_lane_v0(kind))
                    .ok_or_else(|| g2_error(format!("kind {kind} missing V0 lane")))?;
                let v0 = lane_u64(v0_lane, kind_v0_cursors[kind as usize], "standard V0")?;
                kind_v0_cursors[kind as usize] += 1;
                if v0 != registers[rs1] {
                    return Err(g2_error(format!("kind {kind} V0 mismatch at slot {slot}")));
                }
                let (v1, rs2_ptr) = if entry.flags & 1 != 0 {
                    (standard_immediate(entry), None)
                } else {
                    let rs2 = register_index(entry.c)?;
                    let v1_lane = segment
                        .lane(&descs, g2_lane_v1(kind))
                        .ok_or_else(|| g2_error(format!("kind {kind} missing V1 lane")))?;
                    let v1 = lane_u64(v1_lane, kind_v1_cursors[kind as usize], "standard V1")?;
                    kind_v1_cursors[kind as usize] += 1;
                    if v1 != registers[rs2] {
                        return Err(g2_error(format!("kind {kind} V1 mismatch at slot {slot}")));
                    }
                    (v1, Some(entry.c))
                };
                let read0_prev = touch(&mut timestamps, (1, entry.b), timestamp);
                let read1_prev = rs2_ptr
                    .map(|pointer| touch(&mut timestamps, (1, pointer), timestamp + 1))
                    .unwrap_or(0);
                let write_prev = touch(&mut timestamps, (1, entry.a), timestamp + 2);
                let write_prev_value = registers[rd];
                let result = standard_result(kind, entry.local_opcode, v0, v1)?;
                if rd != 0 {
                    registers[rd] = result;
                    blocks.insert((1, entry.a), result);
                }
                timestamp = timestamp
                    .checked_add(3)
                    .ok_or_else(|| g2_error("timestamp overflow"))?;
                append_u32(&mut record, read0_prev);
                append_u32(&mut record, read1_prev);
                append_u32(&mut record, write_prev);
                append_u64(&mut record, write_prev_value);
                append_u64(&mut record, v0);
                append_u64(&mut record, v1);
            } else if matches!(kind, 10 | 11) && entry.access_pattern == 4 {
                let rs1 = register_index(entry.a)?;
                let rs2 = register_index(entry.b)?;
                let v0_lane = segment
                    .lane(&descs, g2_lane_v0(kind))
                    .ok_or_else(|| g2_error(format!("branch kind {kind} missing V0 lane")))?;
                let v1_lane = segment
                    .lane(&descs, g2_lane_v1(kind))
                    .ok_or_else(|| g2_error(format!("branch kind {kind} missing V1 lane")))?;
                let v0 = lane_u64(v0_lane, kind_v0_cursors[kind as usize], "branch V0")?;
                let v1 = lane_u64(v1_lane, kind_v1_cursors[kind as usize], "branch V1")?;
                kind_v0_cursors[kind as usize] += 1;
                kind_v1_cursors[kind as usize] += 1;
                if v0 != registers[rs1] || v1 != registers[rs2] {
                    return Err(g2_error(format!(
                        "branch kind {kind} value mismatch at slot {slot}"
                    )));
                }
                let read0_prev = touch(&mut timestamps, (1, entry.a), timestamp);
                let read1_prev = touch(&mut timestamps, (1, entry.b), timestamp + 1);
                timestamp = timestamp
                    .checked_add(2)
                    .ok_or_else(|| g2_error("timestamp overflow"))?;
                append_u32(&mut record, read0_prev);
                append_u32(&mut record, read1_prev);
                append_u64(&mut record, v0);
                append_u64(&mut record, v1);
            } else if matches!(kind, 12 | 14) && matches!(entry.access_pattern, 5 | 6) {
                let write_enabled = entry.flags & (1 << 2) != 0;
                let (write_prev, write_prev_value) = if write_enabled {
                    let rd = register_index(entry.a)?;
                    let previous = registers[rd];
                    let write_prev = touch(&mut timestamps, (1, entry.a), timestamp);
                    let result = wr1_result(kind, entry, decode.pc_base + slot * 4)?;
                    if rd != 0 {
                        registers[rd] = result;
                        blocks.insert((1, entry.a), result);
                    }
                    (write_prev, previous)
                } else {
                    (0, 0)
                };
                timestamp = timestamp
                    .checked_add(1)
                    .ok_or_else(|| g2_error("timestamp overflow"))?;
                append_u32(&mut record, write_prev);
                append_u64(&mut record, write_prev_value);
            } else if kind == 13 && entry.access_pattern == 7 {
                let rs1 = register_index(entry.b)?;
                let v0_lane = segment
                    .lane(&descs, g2_lane_v0(kind))
                    .ok_or_else(|| g2_error("Jalr missing V0 lane"))?;
                let pointer = lane_u32(v0_lane, kind_v0_cursors[kind as usize], "Jalr pointer")?;
                kind_v0_cursors[kind as usize] += 1;
                if u64::from(pointer) != registers[rs1] {
                    return Err(g2_error(format!("Jalr pointer mismatch at slot {slot}")));
                }
                let read_prev = touch(&mut timestamps, (1, entry.b), timestamp);
                let write_enabled = entry.flags & (1 << 2) != 0;
                let (write_prev, write_prev_value) = if write_enabled {
                    let rd = register_index(entry.a)?;
                    let previous = registers[rd];
                    let write_prev = touch(&mut timestamps, (1, entry.a), timestamp + 1);
                    if rd != 0 {
                        let result = u64::from(decode.pc_base + slot * 4 + 4);
                        registers[rd] = result;
                        blocks.insert((1, entry.a), result);
                    }
                    (write_prev, previous)
                } else {
                    (0, 0)
                };
                timestamp = timestamp
                    .checked_add(2)
                    .ok_or_else(|| g2_error("timestamp overflow"))?;
                append_u32(&mut record, read_prev);
                append_u32(&mut record, write_prev);
                append_u64(&mut record, u64::from(pointer));
                append_u64(&mut record, write_prev_value);
            } else if G2_LOAD_STORE_KINDS.contains(&kind) && matches!(entry.access_pattern, 2 | 3) {
                let a = register_index(entry.a)?;
                let base = register_index(entry.b)?;
                let is_store = entry.access_pattern == 3;
                let v0 = segment
                    .lane(&descs, g2_lane_v0(kind))
                    .ok_or_else(|| g2_error(format!("kind {kind} missing V0 lane")))?;
                let v1 = segment
                    .lane(&descs, g2_lane_v1(kind))
                    .ok_or_else(|| g2_error(format!("kind {kind} missing V1 lane")))?;
                let pointer = lane_u32(v0, kind_v0_cursors[kind as usize], "load/store pointer")?;
                let block0 = lane_u64(v1, kind_v1_cursors[kind as usize], "load/store block")?;
                kind_v0_cursors[kind as usize] += 1;
                kind_v1_cursors[kind as usize] += 1;
                if u64::from(pointer) != registers[base] {
                    return Err(g2_error(format!(
                        "kind {kind} pointer mismatch at slot {slot}"
                    )));
                }

                let source = registers[a];
                let address_space = if entry.flags & (1 << 4) != 0 { 3 } else { 2 };
                let effective = effective_address(pointer, entry);
                let width = if is_store {
                    store_width(entry.local_opcode)?
                } else {
                    load_width(entry.local_opcode)?
                };
                let crossing = u32::from(width) + (effective & 7) > 8;
                let memory_key0 = (address_space, effective & !7);
                let memory_key1 = (address_space, (effective & !7) + 8);
                let current0 = blocks.get(&memory_key0).copied().unwrap_or(0);
                if current0 != block0 {
                    return Err(g2_error(format!(
                        "kind {kind} aligned block mismatch at slot {slot}"
                    )));
                }
                blocks.entry(memory_key0).or_insert(block0);

                let register0_prev = touch(&mut timestamps, (1, entry.b), timestamp);
                let mut memory1_prev = u32::MAX;
                let mut block1 = 0;
                let mut write1_prev = u32::MAX;
                let mut write_prev1_value = 0;
                let (register1_or_memory0_prev, write0_prev, write_prev0_value) = if is_store {
                    let register1_prev = touch(&mut timestamps, (1, entry.a), timestamp + 1);
                    let previous0 = block0;
                    if crossing {
                        let lanes = (
                            residual_ctrl.ok_or_else(|| g2_error("missing residual CTRL lane"))?,
                            residual_tag.ok_or_else(|| g2_error("missing residual TAG lane"))?,
                            residual_value
                                .ok_or_else(|| g2_error("missing residual VALUE lane"))?,
                        );
                        let previous1 = blocks.get(&memory_key1).copied().unwrap_or(0);
                        let (next0, next1) =
                            patch_store_two_blocks(previous0, previous1, effective, width, source);
                        validate_residual(
                            lanes,
                            residual_cursor,
                            timestamp + 2,
                            memory_key0.1,
                            if address_space == 3 { 0x49 } else { 0x45 },
                            Some(next0),
                        )?;
                        validate_residual(
                            lanes,
                            residual_cursor + 1,
                            timestamp + 3,
                            memory_key1.1,
                            if address_space == 3 { 0x49 } else { 0x45 },
                            Some(next1),
                        )?;
                        residual_cursor += 2;
                        let previous_timestamp0 =
                            touch(&mut timestamps, memory_key0, timestamp + 2);
                        write1_prev = touch(&mut timestamps, memory_key1, timestamp + 3);
                        write_prev1_value = previous1;
                        blocks.insert(memory_key0, next0);
                        blocks.insert(memory_key1, next1);
                        (register1_prev, previous_timestamp0, previous0)
                    } else {
                        let previous_timestamp0 =
                            touch(&mut timestamps, memory_key0, timestamp + 2);
                        blocks.insert(
                            memory_key0,
                            patch_store(previous0, effective, width, source),
                        );
                        (register1_prev, previous_timestamp0, previous0)
                    }
                } else {
                    let memory0_prev = touch(&mut timestamps, memory_key0, timestamp + 1);
                    if crossing {
                        let lanes = (
                            residual_ctrl.ok_or_else(|| g2_error("missing residual CTRL lane"))?,
                            residual_tag.ok_or_else(|| g2_error("missing residual TAG lane"))?,
                            residual_value
                                .ok_or_else(|| g2_error("missing residual VALUE lane"))?,
                        );
                        validate_residual(
                            lanes,
                            residual_cursor,
                            timestamp + 1,
                            memory_key0.1,
                            0x44,
                            Some(block0),
                        )?;
                        block1 = validate_residual(
                            lanes,
                            residual_cursor + 1,
                            timestamp + 2,
                            memory_key1.1,
                            0x44,
                            None,
                        )?;
                        residual_cursor += 2;
                        memory1_prev = touch(&mut timestamps, memory_key1, timestamp + 2);
                        blocks.entry(memory_key1).or_insert(block1);
                    }
                    let (write_prev, write_prev_value) = if entry.flags & (1 << 2) != 0 {
                        let previous = registers[a];
                        let write_timestamp =
                            timestamp + if g2_multi_block_kind(kind) { 3 } else { 2 };
                        let write_prev = touch(&mut timestamps, (1, entry.a), write_timestamp);
                        let loaded =
                            decode_load_two_blocks(entry.local_opcode, block0, block1, effective)?;
                        if a != 0 {
                            registers[a] = loaded;
                            blocks.insert((1, entry.a), loaded);
                        }
                        (write_prev, previous)
                    } else {
                        (0, 0)
                    };
                    (memory0_prev, write_prev, write_prev_value)
                };

                timestamp = timestamp
                    .checked_add(if g2_multi_block_kind(kind) { 4 } else { 3 })
                    .ok_or_else(|| g2_error("timestamp overflow"))?;
                if g2_multi_block_kind(kind) {
                    record = if is_store {
                        native_store_record(
                            decode.pc_base + slot * 4,
                            from_timestamp,
                            entry,
                            pointer,
                            register0_prev,
                            register1_or_memory0_prev,
                            write0_prev,
                            write1_prev,
                            source,
                            write_prev0_value,
                            write_prev1_value,
                            address_space,
                        )
                    } else {
                        native_load_record(
                            decode.pc_base + slot * 4,
                            from_timestamp,
                            entry,
                            pointer,
                            register0_prev,
                            register1_or_memory0_prev,
                            memory1_prev,
                            write0_prev,
                            write_prev0_value,
                            block0,
                            block1,
                        )
                    };
                } else {
                    append_u32(&mut record, register0_prev);
                    append_u32(&mut record, register1_or_memory0_prev);
                    append_u32(&mut record, write0_prev);
                    append_u64(&mut record, write_prev0_value);
                    append_u64(&mut record, u64::from(pointer));
                    append_u64(&mut record, if is_store { source } else { block0 });
                }
            } else {
                return Err(g2_error(format!(
                    "slot {slot} is outside the fixed-standard G2 schema"
                )));
            }
            let expected_stride = match entry.access_pattern {
                2 | 3 if g2_multi_block_kind(kind) => 60,
                0..=3 | 8 => PREFLIGHT_ADDSUB_RECORD_SIZE,
                4 | 7 => 32,
                5 | 6 => 20,
                _ => return Err(g2_error("reference decoder saw an invalid access pattern")),
            };
            if record.len() != expected_stride {
                return Err(g2_error(format!(
                    "reference decoder produced stride {} instead of {expected_stride}",
                    record.len()
                )));
            }
            compact_records.entry(kind).or_default().extend(record);
        }
    }

    let opaque_event_count = descs
        .iter()
        .find(|desc| desc.kind == G2_LANE_OPAQUE_EVENT_COUNT)
        .map_or(0, |desc| desc.count as usize);
    if expanded_program_slots.len() != header.instruction_count as usize
        || residual_cursor != header.residual_event_count as usize
        || opaque_event_cursor != opaque_event_count
    {
        return Err(g2_error(
            "program or residual cursor did not finish exactly",
        ));
    }
    for kind in 0u8..30 {
        let v0_count = descs
            .iter()
            .find(|desc| desc.kind == g2_lane_v0(kind))
            .map_or(0, |desc| desc.count as usize);
        let v1_count = descs
            .iter()
            .find(|desc| desc.kind == g2_lane_v1(kind))
            .map_or(0, |desc| desc.count as usize);
        if kind_v0_cursors[kind as usize] != v0_count || kind_v1_cursors[kind as usize] != v1_count
        {
            return Err(g2_error(format!(
                "kind {kind} V0/V1 cursors did not finish exactly"
            )));
        }
    }
    Ok(RvrG2ReferenceV1 {
        compact_records,
        final_registers: registers,
        final_timestamps: timestamps,
        final_blocks: blocks,
        final_timestamp: timestamp,
        expanded_program_slots,
    })
}

fn append_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn append_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_u16(out: &mut [u8], offset: usize, value: u16) {
    out[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

fn put_u32(out: &mut [u8], offset: usize, value: u32) {
    out[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn put_u64(out: &mut [u8], offset: usize, value: u64) {
    out[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

fn g2_multi_block_kind(kind: u8) -> bool {
    matches!(kind, 20..=22 | 24..=28)
}

#[allow(clippy::too_many_arguments)]
fn native_load_record(
    pc: u32,
    timestamp: u32,
    entry: &RvrDeltaDecodeEntry,
    pointer: u32,
    rs1_prev: u32,
    memory0_prev: u32,
    memory1_prev: u32,
    write_prev: u32,
    write_prev_value: u64,
    block0: u64,
    block1: u64,
) -> Vec<u8> {
    let mut out = vec![0; 60];
    put_u32(&mut out, 0, pc);
    put_u32(&mut out, 4, timestamp);
    put_u32(&mut out, 8, pointer);
    put_u32(&mut out, 12, rs1_prev);
    put_u32(&mut out, 16, memory0_prev);
    put_u32(&mut out, 20, memory1_prev);
    put_u16(&mut out, 24, entry.c as u16);
    out[26] = u8::from(entry.flags & (1 << 5) != 0);
    put_u32(&mut out, 28, write_prev);
    put_u64(&mut out, 32, write_prev_value);
    out[40] = entry.b as u8;
    out[41] = if entry.flags & (1 << 2) != 0 {
        entry.a as u8
    } else {
        u8::MAX
    };
    put_u64(&mut out, 44, block0);
    put_u64(&mut out, 52, block1);
    out
}

#[allow(clippy::too_many_arguments)]
fn native_store_record(
    pc: u32,
    timestamp: u32,
    entry: &RvrDeltaDecodeEntry,
    pointer: u32,
    rs1_prev: u32,
    rs2_prev: u32,
    write0_prev: u32,
    write1_prev: u32,
    source: u64,
    previous0: u64,
    previous1: u64,
    address_space: u8,
) -> Vec<u8> {
    let mut out = vec![0; 60];
    put_u32(&mut out, 0, pc);
    put_u32(&mut out, 4, timestamp);
    put_u32(&mut out, 8, pointer);
    put_u32(&mut out, 12, rs1_prev);
    put_u32(&mut out, 16, rs2_prev);
    put_u32(&mut out, 20, write0_prev);
    put_u32(&mut out, 24, write1_prev);
    put_u16(&mut out, 28, entry.c as u16);
    out[30] = entry.b as u8;
    out[31] = entry.a as u8;
    out[32] = u8::from(entry.flags & (1 << 5) != 0);
    out[33] = address_space;
    put_u64(&mut out, 36, source);
    put_u64(&mut out, 44, previous0);
    put_u64(&mut out, 52, previous1);
    out
}

fn lane_u32(lane: &[u8], index: usize, label: &str) -> Result<u32, ExecutionError> {
    let at = index
        .checked_mul(4)
        .ok_or_else(|| g2_error(format!("{label} cursor overflow")))?;
    lane.get(at..at + 4)
        .map(|bytes| u32::from_le_bytes(bytes.try_into().expect("four-byte lane")))
        .ok_or_else(|| g2_error(format!("{label} lane over-consumed")))
}

fn lane_u64(lane: &[u8], index: usize, label: &str) -> Result<u64, ExecutionError> {
    let at = index
        .checked_mul(8)
        .ok_or_else(|| g2_error(format!("{label} cursor overflow")))?;
    lane.get(at..at + 8)
        .map(|bytes| u64::from_le_bytes(bytes.try_into().expect("eight-byte lane")))
        .ok_or_else(|| g2_error(format!("{label} lane over-consumed")))
}

fn touch(timestamps: &mut BTreeMap<(u8, u32), u32>, key: (u8, u32), timestamp: u32) -> u32 {
    timestamps.insert(key, timestamp).unwrap_or(0)
}

fn effective_address(pointer: u32, entry: &RvrDeltaDecodeEntry) -> u32 {
    let offset = if entry.flags & (1 << 5) != 0 {
        i32::from(entry.c as u16 as i16) as u32
    } else {
        u32::from(entry.c as u16)
    };
    pointer.wrapping_add(offset)
}

fn store_width(local_opcode: u8) -> Result<u8, ExecutionError> {
    match local_opcode {
        4 => Ok(8),
        5 => Ok(4),
        6 => Ok(2),
        7 => Ok(1),
        _ => Err(g2_error(format!("invalid store opcode {local_opcode}"))),
    }
}

fn load_width(local_opcode: u8) -> Result<u8, ExecutionError> {
    match local_opcode {
        0 => Ok(8),
        1 | 8 => Ok(1),
        2 | 9 => Ok(2),
        3 | 10 => Ok(4),
        _ => Err(g2_error(format!("invalid load opcode {local_opcode}"))),
    }
}

fn patch_store(block: u64, address: u32, width: u8, source: u64) -> u64 {
    let shift = (address & 7) * 8;
    let mask = if width == 8 {
        u64::MAX
    } else {
        (1u64 << (u32::from(width) * 8)) - 1
    };
    (block & !(mask << shift)) | ((source & mask) << shift)
}

fn patch_store_two_blocks(
    block0: u64,
    block1: u64,
    address: u32,
    width: u8,
    source: u64,
) -> (u64, u64) {
    let mut bytes = [0u8; 16];
    bytes[..8].copy_from_slice(&block0.to_le_bytes());
    bytes[8..].copy_from_slice(&block1.to_le_bytes());
    let offset = (address & 7) as usize;
    bytes[offset..offset + width as usize].copy_from_slice(&source.to_le_bytes()[..width as usize]);
    (
        u64::from_le_bytes(bytes[..8].try_into().expect("first memory block")),
        u64::from_le_bytes(bytes[8..].try_into().expect("second memory block")),
    )
}

fn decode_load_two_blocks(
    local_opcode: u8,
    block0: u64,
    block1: u64,
    address: u32,
) -> Result<u64, ExecutionError> {
    let mut bytes = [0u8; 16];
    bytes[..8].copy_from_slice(&block0.to_le_bytes());
    bytes[8..].copy_from_slice(&block1.to_le_bytes());
    let offset = (address & 7) as usize;
    let shifted = u64::from_le_bytes(
        bytes[offset..offset + 8]
            .try_into()
            .expect("two-block load window"),
    );
    match local_opcode {
        0 => Ok(shifted),
        1 => Ok(shifted as u8 as u64),
        2 => Ok(shifted as u16 as u64),
        3 => Ok(shifted as u32 as u64),
        8 => Ok(shifted as u8 as i8 as i64 as u64),
        9 => Ok(shifted as u16 as i16 as i64 as u64),
        10 => Ok(shifted as u32 as i32 as i64 as u64),
        _ => Err(g2_error(format!("invalid load opcode {local_opcode}"))),
    }
}

fn validate_residual(
    lanes: (&[u8], &[u8], &[u8]),
    index: usize,
    timestamp: u32,
    address: u32,
    tag: u8,
    value: Option<u64>,
) -> Result<u64, ExecutionError> {
    let (ctrl, tags, values) = lanes;
    let control = lane_u64(ctrl, index, "residual control")?;
    let actual_tag = *tags
        .get(index)
        .ok_or_else(|| g2_error("residual tag lane over-consumed"))?;
    let actual_value = lane_u64(values, index, "residual value")?;
    if control as u32 != timestamp
        || (control >> 32) as u32 != address
        || actual_tag != tag
        || value.is_some_and(|value| value != actual_value)
    {
        return Err(g2_error(format!("residual event {index} shape mismatch")));
    }
    Ok(actual_value)
}

fn read_residual(
    lanes: (&[u8], &[u8], &[u8]),
    index: usize,
) -> Result<Option<(u32, u32, u8, u64)>, ExecutionError> {
    let Some(ctrl_at) = index.checked_mul(8) else {
        return Err(g2_error("residual CTRL cursor overflow"));
    };
    let Some(value_at) = index.checked_mul(8) else {
        return Err(g2_error("residual VALUE cursor overflow"));
    };
    let Some(ctrl) = lanes.0.get(ctrl_at..ctrl_at + 8) else {
        return Ok(None);
    };
    let tag = *lanes
        .1
        .get(index)
        .ok_or_else(|| g2_error("residual TAG lanes have unequal lengths"))?;
    let value = lanes
        .2
        .get(value_at..value_at + 8)
        .ok_or_else(|| g2_error("residual VALUE lanes have unequal lengths"))?;
    let ctrl = u64::from_le_bytes(ctrl.try_into().expect("eight-byte residual CTRL"));
    Ok(Some((
        ctrl as u32,
        (ctrl >> 32) as u32,
        tag,
        u64::from_le_bytes(value.try_into().expect("eight-byte residual VALUE")),
    )))
}

fn apply_residual_reference(
    timestamp: u32,
    address: u32,
    tag: u8,
    value: u64,
    registers: &mut [u64; 32],
    timestamps: &mut BTreeMap<(u8, u32), u32>,
    blocks: &mut BTreeMap<(u8, u32), u64>,
) -> Result<(), ExecutionError> {
    let kind = tag & 3;
    let address_space_code = (tag >> 2) & 3;
    let width_code = (tag >> 4) & 7;
    let valid_width = matches!(width_code, 0..=4);
    if tag & 0x80 != 0
        || kind == 3
        || address_space_code == 3
        || !valid_width
        || (kind != 2 && width_code == 0)
    {
        return Err(g2_error("opaque residual tag is outside the frozen schema"));
    }
    let address_space = address_space_code + 1;
    let key = (address_space, address & !7);
    if address_space == 1 && (width_code != 4 || address >= 32 * 8 || address & 7 != 0) {
        return Err(g2_error("opaque register residual is malformed"));
    }
    if kind == 0 {
        if blocks.get(&key).is_some_and(|current| *current != value) {
            return Err(g2_error("opaque residual read value mismatch"));
        }
    } else if kind == 1 {
        blocks.insert(key, value);
        if address_space == 1 && address != 0 {
            registers[(address / 8) as usize] = value;
        }
    }
    touch(timestamps, key, timestamp);
    Ok(())
}

fn register_index(pointer: u32) -> Result<usize, ExecutionError> {
    if !pointer.is_multiple_of(8) || pointer / 8 >= 32 {
        return Err(g2_error(format!(
            "register pointer {pointer} is outside the RV64 register file"
        )));
    }
    Ok((pointer / 8) as usize)
}

fn sign_extend_12(value: u32) -> u64 {
    ((value << 20) as i32 >> 20) as i64 as u64
}

fn standard_immediate(entry: &RvrDeltaDecodeEntry) -> u64 {
    if entry.flags & (1 << 1) != 0 {
        ((entry.c << 8) as i32 >> 8) as i64 as u64
    } else {
        u64::from(entry.c)
    }
}

fn sign_extend_word(value: u32) -> u64 {
    value as i32 as i64 as u64
}

fn standard_result(kind: u8, local_opcode: u8, v0: u64, v1: u64) -> Result<u64, ExecutionError> {
    let invalid = || {
        g2_error(format!(
            "invalid local opcode {local_opcode} for kind {kind}"
        ))
    };
    Ok(match kind {
        0 => match local_opcode {
            0 => v0.wrapping_add(v1),
            1 => v0.wrapping_sub(v1),
            _ => return Err(invalid()),
        },
        1 => match local_opcode {
            2 => v0 ^ v1,
            3 => v0 | v1,
            4 => v0 & v1,
            _ => return Err(invalid()),
        },
        2 => match local_opcode {
            0 => u64::from((v0 as i64) < (v1 as i64)),
            1 => u64::from(v0 < v1),
            _ => return Err(invalid()),
        },
        3 => match local_opcode {
            0 => v0.wrapping_shl((v1 & 63) as u32),
            1 => v0.wrapping_shr((v1 & 63) as u32),
            _ => return Err(invalid()),
        },
        4 if local_opcode == 2 => ((v0 as i64) >> (v1 & 63)) as u64,
        5 => sign_extend_word(match local_opcode {
            0 => (v0 as u32).wrapping_add(v1 as u32),
            1 => (v0 as u32).wrapping_sub(v1 as u32),
            _ => return Err(invalid()),
        }),
        6 => sign_extend_word(match local_opcode {
            0 => (v0 as u32).wrapping_shl((v1 & 31) as u32),
            1 => (v0 as u32).wrapping_shr((v1 & 31) as u32),
            _ => return Err(invalid()),
        }),
        7 if local_opcode == 0 => sign_extend_word(((v0 as u32 as i32) >> (v1 & 31)) as u32),
        15 => v0.wrapping_mul(v1),
        16 => match local_opcode {
            0 => (((v0 as i64 as i128) * (v1 as i64 as i128)) >> 64) as u64,
            1 => (((v0 as i64 as i128) * (v1 as u128 as i128)) >> 64) as u64,
            2 => (((v0 as u128) * (v1 as u128)) >> 64) as u64,
            _ => return Err(invalid()),
        },
        17 => sign_extend_word((v0 as u32).wrapping_mul(v1 as u32)),
        18 => divrem_result(local_opcode, v0, v1).ok_or_else(invalid)?,
        19 => divrem_w_result(local_opcode, v0, v1).ok_or_else(invalid)?,
        _ => return Err(invalid()),
    })
}

fn divrem_result(local_opcode: u8, lhs: u64, rhs: u64) -> Option<u64> {
    Some(match local_opcode {
        0 => {
            let (lhs, rhs) = (lhs as i64, rhs as i64);
            if rhs == 0 {
                u64::MAX
            } else if lhs == i64::MIN && rhs == -1 {
                lhs as u64
            } else {
                (lhs / rhs) as u64
            }
        }
        1 => {
            if rhs == 0 {
                u64::MAX
            } else {
                lhs / rhs
            }
        }
        2 => {
            let (lhs, rhs) = (lhs as i64, rhs as i64);
            if rhs == 0 {
                lhs as u64
            } else if lhs == i64::MIN && rhs == -1 {
                0
            } else {
                (lhs % rhs) as u64
            }
        }
        3 => {
            if rhs == 0 {
                lhs
            } else {
                lhs % rhs
            }
        }
        _ => return None,
    })
}

fn divrem_w_result(local_opcode: u8, lhs: u64, rhs: u64) -> Option<u64> {
    let (lhs, rhs) = (lhs as u32, rhs as u32);
    let word = match local_opcode {
        0 => {
            let (lhs, rhs) = (lhs as i32, rhs as i32);
            if rhs == 0 {
                u32::MAX
            } else if lhs == i32::MIN && rhs == -1 {
                lhs as u32
            } else {
                (lhs / rhs) as u32
            }
        }
        1 => {
            if rhs == 0 {
                u32::MAX
            } else {
                lhs / rhs
            }
        }
        2 => {
            let (lhs, rhs) = (lhs as i32, rhs as i32);
            if rhs == 0 {
                lhs as u32
            } else if lhs == i32::MIN && rhs == -1 {
                0
            } else {
                (lhs % rhs) as u32
            }
        }
        3 => {
            if rhs == 0 {
                lhs
            } else {
                lhs % rhs
            }
        }
        _ => return None,
    };
    Some(sign_extend_word(word))
}

fn wr1_result(kind: u8, entry: &RvrDeltaDecodeEntry, pc: u32) -> Result<u64, ExecutionError> {
    match kind {
        12 if entry.flags & (1 << 3) != 0 => Ok(u64::from(pc.wrapping_add(4))),
        12 => Ok((entry.c << 12) as i32 as i64 as u64),
        14 => Ok(u64::from(pc).wrapping_add((entry.c << 8) as i32 as i64 as u64)),
        _ => Err(g2_error(format!("kind {kind} is not a zero-arity write"))),
    }
}

pub(crate) struct RvrG2PreparedV1 {
    backing: Option<Vec<u8>>,
    backing_offset: usize,
    byte_capacity: usize,
    lanes: Vec<G2ProducerLaneV1>,
    pool: super::RvrPreflightBufferPool,
    backing_route: super::preflight_pool::G2BackingRoute,
    pub producer: G2ProducerV1,
}

impl RvrG2PreparedV1 {
    #[cfg(test)]
    pub fn new(capacities: &RvrG2CapacitiesV1) -> Result<Self, ExecutionError> {
        Self::new_pooled(capacities, &super::RvrPreflightBufferPool::default())
    }

    pub fn capacity_bytes(capacities: &RvrG2CapacitiesV1) -> Result<usize, ExecutionError> {
        let specs = producer_lane_specs();
        debug_assert_eq!(specs.len(), G2_PRODUCER_LANE_COUNT);
        let mut offset = align_up(
            G2_SEGMENT_HEADER_V1_SIZE
                + (G2_PRODUCER_LANE_COUNT + G2_MAX_OPAQUE_LANES) * G2_LANE_DESC_V1_SIZE,
            G2_WIRE_ALIGNMENT,
        )?;
        for (slot, spec) in specs.iter().enumerate() {
            debug_assert_eq!(slot, spec.slot);
            let cap = lane_capacity(*spec, capacities);
            let bytes = (cap as usize)
                .checked_mul(spec.width as usize)
                .ok_or_else(|| g2_error("lane capacity byte count overflow"))?;
            offset = align_up(
                offset
                    .checked_add(bytes)
                    .ok_or_else(|| g2_error("wire capacity overflow"))?,
                G2_WIRE_ALIGNMENT,
            )?;
        }
        Ok(offset)
    }

    #[cfg(test)]
    pub(crate) fn new_pooled(
        capacities: &RvrG2CapacitiesV1,
        pool: &super::RvrPreflightBufferPool,
    ) -> Result<Self, ExecutionError> {
        Self::new_pooled_for_mode(capacities, pool, true)
    }

    pub(crate) fn new_pooled_for_mode(
        capacities: &RvrG2CapacitiesV1,
        pool: &super::RvrPreflightBufferPool,
        checked_emission: bool,
    ) -> Result<Self, ExecutionError> {
        let backing_route = if checked_emission {
            super::preflight_pool::G2BackingRoute::Checked
        } else {
            super::preflight_pool::G2BackingRoute::Production
        };
        let specs = producer_lane_specs();
        let byte_capacity = Self::capacity_bytes(capacities)?;
        let mut lanes = Vec::with_capacity(specs.len());
        let mut offset = align_up(
            G2_SEGMENT_HEADER_V1_SIZE
                + (G2_PRODUCER_LANE_COUNT + G2_MAX_OPAQUE_LANES) * G2_LANE_DESC_V1_SIZE,
            G2_WIRE_ALIGNMENT,
        )?;
        for (slot, spec) in specs.iter().enumerate() {
            debug_assert_eq!(slot, spec.slot);
            let cap = lane_capacity(*spec, capacities);
            lanes.push(G2ProducerLaneV1 {
                offset: offset as u64,
                len: 0,
                cap,
                expected_len: 0,
                reserved: 0,
            });
            let bytes = (cap as usize)
                .checked_mul(spec.width as usize)
                .ok_or_else(|| g2_error("lane capacity byte count overflow"))?;
            offset = align_up(
                offset
                    .checked_add(bytes)
                    .ok_or_else(|| g2_error("wire capacity overflow"))?,
                G2_WIRE_ALIGNMENT,
            )?;
        }
        debug_assert_eq!(offset, byte_capacity);
        let mut backing = pool.take_g2_backing(byte_capacity, backing_route);
        let backing_offset = (32 - backing.as_ptr() as usize % 32) % 32;
        assert!(backing_offset + byte_capacity <= backing.len());
        let base = unsafe { backing.as_mut_ptr().add(backing_offset) };
        let producer = G2ProducerV1 {
            base,
            capacity: byte_capacity as u64,
            lanes: lanes.as_mut_ptr(),
            lane_count: G2_PRODUCER_LANE_COUNT as u32,
            instruction_count: 0,
            overflow: 0,
            reserved: 0,
        };
        Ok(Self {
            backing: Some(backing),
            backing_offset,
            byte_capacity,
            lanes,
            pool: pool.clone(),
            backing_route,
            producer,
        })
    }

    pub(crate) fn residual_capacity(&self) -> usize {
        self.lanes[G2_PRODUCER_RESIDUAL_VALUE_SLOT].cap as usize
    }

    pub(crate) fn producer_lane_len(&self, slot: usize) -> Result<u32, ExecutionError> {
        self.lanes
            .get(slot)
            .map(|lane| lane.len)
            .ok_or_else(|| g2_error(format!("producer lane slot {slot} is out of range")))
    }

    pub(crate) fn producer_u32_lane(&self, slot: usize) -> Result<&[u32], ExecutionError> {
        let lane = self
            .lanes
            .get(slot)
            .ok_or_else(|| g2_error(format!("producer lane slot {slot} is out of range")))?;
        let offset = usize::try_from(lane.offset)
            .map_err(|_| g2_error("producer u32 lane offset exceeds usize"))?;
        let bytes = (lane.len as usize)
            .checked_mul(size_of::<u32>())
            .ok_or_else(|| g2_error("producer u32 lane byte count overflow"))?;
        if offset % align_of::<u32>() != 0
            || offset
                .checked_add(bytes)
                .is_none_or(|end| end > self.byte_capacity)
        {
            return Err(g2_error("producer u32 lane exceeds its backing"));
        }
        let base = self
            .backing
            .as_ref()
            .expect("G2 backing already moved")
            .as_ptr();
        unsafe {
            // SAFETY: the producer layout aligns every fixed lane, the range
            // was checked above, and native execution initialized its written
            // prefix before this read.
            Ok(std::slice::from_raw_parts(
                base.add(self.backing_offset + offset).cast::<u32>(),
                lane.len as usize,
            ))
        }
    }

    pub fn finalize(
        mut self,
        segment_id: u32,
        expected_instruction_count: u32,
        expected_kind_counts: Option<&[u32; 31]>,
        fingerprint: [u8; 32],
        opaque_written: &[(RvrG2OpaqueBindingV1, u32, u32)],
    ) -> Result<RvrG2SegmentV1, ExecutionError> {
        let p = self.producer;
        let checked_emission = expected_kind_counts.is_some();
        if p.overflow != 0
            || p.base
                != unsafe {
                    self.backing
                        .as_mut()
                        .expect("G2 backing already moved")
                        .as_mut_ptr()
                        .add(self.backing_offset)
                }
            || p.capacity as usize != self.byte_capacity
            || p.lanes != self.lanes.as_mut_ptr()
            || p.lane_count as usize != self.lanes.len()
            || (checked_emission && p.instruction_count != expected_instruction_count)
            || (!checked_emission && p.instruction_count != 0)
            || p.reserved != 0
            || self.lanes.iter().any(|lane| lane.len > lane.cap)
            || (expected_instruction_count != 0 && self.lanes[G2_PRODUCER_RUN_SLOT].len == 0)
        {
            return Err(g2_error("native lane cursor/count validation failed"));
        }

        // Checked emission retains an independent host cursor oracle.
        // Production leaves all expected cursors at zero and delegates exact
        // run/lane exhaustion to the fail-hard device replay.
        if checked_emission {
            for (slot, lane) in self.lanes.iter().enumerate() {
                if slot == G2_PRODUCER_RUN_SLOT || (G2_PRODUCER_ADDI_SLOT..=57).contains(&slot) {
                    if lane.len != lane.expected_len {
                        return Err(g2_error(format!(
                            "static lane cursor differs from entered block spans: slot {slot}, actual {}, expected {}",
                            lane.len, lane.expected_len
                        )));
                    }
                } else if lane.expected_len != 0 {
                    return Err(g2_error(format!(
                        "dynamic lane carries invalid static-span metadata: slot {slot}"
                    )));
                }
                if lane.reserved != 0 {
                    return Err(g2_error(format!(
                        "lane carries nonzero reserved metadata: slot {slot}"
                    )));
                }
            }
        } else if self.lanes.iter().any(|lane| lane.expected_len != 0) {
            return Err(g2_error(
                "production lane carries checked-emission cursor metadata",
            ));
        }

        // Cross-check the independently accumulated static spans against the
        // metered per-AIR totals. Kinds 1..=7 share one AIR between immediate
        // and register forms, so V1 is necessarily bounded by that total;
        // exactness for those lanes is supplied by `expected_len` above.
        if let Some(expected_kind_counts) = expected_kind_counts {
            for kind in 0u8..30 {
                for value_lane in [false, true] {
                    let Some(slot) = g2_standard_producer_slot(kind, value_lane) else {
                        continue;
                    };
                    let expected = expected_kind_counts[kind as usize];
                    let actual = self.lanes[slot].len;
                    let valid = if value_lane && (1..=7).contains(&kind) {
                        actual <= expected
                    } else {
                        actual == expected
                    };
                    if !valid {
                        return Err(g2_error(format!(
                            "standard lane cursor differs from metered count: kind {kind}, lane {value_lane}, actual {actual}, expected {expected}"
                        )));
                    }
                }
            }
        }

        let residual_count = self.lanes[G2_PRODUCER_RESIDUAL_CTRL_SLOT].len;
        if self.lanes[G2_PRODUCER_RESIDUAL_TAG_SLOT].len != residual_count
            || self.lanes[G2_PRODUCER_RESIDUAL_VALUE_SLOT].len != residual_count
        {
            return Err(g2_error("residual atomic-group cursors differ"));
        }
        for kind in G2_LOAD_STORE_KINDS {
            let slot = g2_load_store_producer_slot(kind, false)
                .expect("frozen load/store kind has a producer slot");
            if self.lanes[slot].len != self.lanes[slot + 1].len {
                return Err(g2_error(format!(
                    "load/store kind {kind} atomic lane cursors differ"
                )));
            }
        }

        let specs = producer_lane_specs();
        let mut active_lanes = specs
            .iter()
            .filter_map(|spec| {
                let lane = self.lanes[spec.slot];
                (spec.kind == G2_LANE_RUN_BLOCK_ID || lane.len != 0).then_some((*spec, lane))
            })
            .collect::<Vec<_>>();
        active_lanes.sort_unstable_by_key(|(spec, _)| spec.kind);
        let header_bytes = G2_SEGMENT_HEADER_V1_SIZE
            + (active_lanes.len() + opaque_written.len()) * G2_LANE_DESC_V1_SIZE;
        if opaque_written.len() > G2_MAX_OPAQUE_LANES {
            return Err(g2_error(
                "opaque AIR count exceeds the producer descriptor reserve",
            ));
        }
        let payload_begin = align_up(header_bytes, G2_WIRE_ALIGNMENT)?;
        let mut wire_offset = payload_begin;
        let mut committed_lanes = Vec::with_capacity(active_lanes.len());
        let mut descs = Vec::with_capacity(active_lanes.len());
        for (spec, lane) in active_lanes {
            let payload_bytes = lane
                .len
                .checked_mul(u32::from(spec.width))
                .ok_or_else(|| g2_error("lane payload byte count overflow"))?;
            let source_offset = usize::try_from(lane.offset)
                .map_err(|_| g2_error("producer lane offset exceeds usize"))?;
            if source_offset
                .checked_add(payload_bytes as usize)
                .is_none_or(|end| end > self.byte_capacity)
            {
                return Err(g2_error("committed lane exceeds its producer backing"));
            }
            descs.push(G2LaneDescV1 {
                kind: spec.kind,
                elem_width: spec.width,
                encoding: G2_ENCODING_FIXED_LE,
                flags: spec.flags,
                count: lane.len,
                payload_bytes,
                offset: wire_offset as u64,
                group_id: spec.group,
                reserved: 0,
            });
            committed_lanes.push(RvrG2CommittedLaneV1 {
                kind: spec.kind,
                source_offset: Some(source_offset),
                wire_offset,
                payload_bytes: payload_bytes as usize,
            });
            wire_offset = align_up(
                wire_offset
                    .checked_add(payload_bytes as usize)
                    .ok_or_else(|| g2_error("committed segment length overflow"))?,
                G2_WIRE_ALIGNMENT,
            )?;
        }
        for &(binding, count, payload_bytes) in opaque_written {
            let stride = u32::try_from(binding.geometry.stride_dense())
                .map_err(|_| g2_error("opaque AIR stride exceeds the frozen u32 descriptor"))?;
            let expected_payload_bytes = count
                .checked_mul(stride)
                .ok_or_else(|| g2_error("opaque AIR payload byte count overflow"))?;
            if payload_bytes != expected_payload_bytes {
                return Err(g2_error(format!(
                    "opaque AIR {} cursor does not match its fingerprinted geometry",
                    binding.air_idx
                )));
            }
            descs.push(G2LaneDescV1 {
                kind: binding.lane_kind(),
                elem_width: 0,
                encoding: G2_ENCODING_OPAQUE_FINAL,
                flags: G2_LANE_FLAG_OPAQUE_FINAL,
                count,
                payload_bytes,
                offset: wire_offset as u64,
                group_id: 0,
                reserved: 0,
            });
            committed_lanes.push(RvrG2CommittedLaneV1 {
                kind: binding.lane_kind(),
                source_offset: None,
                wire_offset,
                payload_bytes: payload_bytes as usize,
            });
            wire_offset = align_up(
                wire_offset
                    .checked_add(payload_bytes as usize)
                    .ok_or_else(|| g2_error("opaque segment length overflow"))?,
                G2_WIRE_ALIGNMENT,
            )?;
        }
        descs.sort_unstable_by_key(|desc| desc.kind);
        committed_lanes.sort_unstable_by_key(|lane| lane.kind);
        let byte_len = wire_offset;
        let header = G2SegmentHeaderV1 {
            magic: G2_SEGMENT_MAGIC_V1,
            version: G2_WIRE_VERSION_V1,
            header_bytes: u16::try_from(header_bytes)
                .map_err(|_| g2_error("descriptor table exceeds u16 header size"))?,
            lane_count: descs.len() as u16,
            flags: G2_FLAGS_V1,
            segment_id,
            instruction_count: expected_instruction_count,
            run_count: self.lanes[G2_PRODUCER_RUN_SLOT].len,
            residual_event_count: residual_count,
            schema_fingerprint: fingerprint,
        };
        unsafe {
            // SAFETY: the backing base is 32-byte aligned for both POD
            // structs, and the fixed producer layout reserves the maximum
            // opaque descriptor table before the first source payload.
            let base = self
                .backing
                .as_mut()
                .expect("G2 backing already moved")
                .as_mut_ptr()
                .add(self.backing_offset);
            base.cast::<G2SegmentHeaderV1>().write(header);
            let desc_base = base.add(G2_SEGMENT_HEADER_V1_SIZE);
            for (index, desc) in descs.iter().enumerate() {
                desc_base
                    .add(index * G2_LANE_DESC_V1_SIZE)
                    .cast::<G2LaneDescV1>()
                    .write(*desc);
            }
            (&*base.add(14).cast::<AtomicU16>())
                .store(G2_FLAGS_V1 | G2_FLAG_COMMITTED, Ordering::Release);
        }
        let segment = RvrG2SegmentV1 {
            backing: self.backing.take(),
            backing_offset: self.backing_offset,
            backing_capacity: self.byte_capacity,
            pool: self.pool.clone(),
            backing_route: self.backing_route,
            byte_len,
            header_prefix_len: payload_begin,
            committed_lanes,
        };
        segment.validate(&fingerprint)?;
        if std::env::var("OPENVM_RVR_G2_GPU_PROFILE").as_deref() == Ok("1") {
            eprintln!(
                "OPENVM_RVR_G2_CAPACITY segment={} reserved_bytes={} committed_bytes={} transfer_bytes={} residual_events={}",
                segment_id,
                self.byte_capacity,
                segment.byte_len(),
                segment.transfer_byte_len(),
                residual_count,
            );
        }
        Ok(segment)
    }
}

impl Drop for RvrG2PreparedV1 {
    fn drop(&mut self) {
        if let Some(backing) = self.backing.take() {
            self.pool
                .recycle_g2_backing(backing, self.byte_capacity, self.backing_route);
        }
    }
}

#[derive(Clone, Copy)]
struct LaneSpec {
    slot: usize,
    kind: u16,
    width: u8,
    flags: u32,
    group: u32,
}

fn producer_lane_specs() -> Vec<LaneSpec> {
    let required = G2_LANE_FLAG_REQUIRED;
    let atomic = required | G2_LANE_FLAG_ATOMIC_GROUP;
    let mut specs = vec![
        LaneSpec {
            slot: G2_PRODUCER_RUN_SLOT,
            kind: G2_LANE_RUN_BLOCK_ID,
            width: 4,
            flags: required,
            group: 0,
        },
        LaneSpec {
            slot: G2_PRODUCER_RESIDUAL_CTRL_SLOT,
            kind: G2_LANE_RESIDUAL_CTRL,
            width: 8,
            flags: atomic,
            group: G2_GROUP_RESIDUAL,
        },
        LaneSpec {
            slot: G2_PRODUCER_RESIDUAL_TAG_SLOT,
            kind: G2_LANE_RESIDUAL_TAG,
            width: 1,
            flags: atomic,
            group: G2_GROUP_RESIDUAL,
        },
        LaneSpec {
            slot: G2_PRODUCER_RESIDUAL_VALUE_SLOT,
            kind: G2_LANE_RESIDUAL_VALUE,
            width: 8,
            flags: atomic,
            group: G2_GROUP_RESIDUAL,
        },
        LaneSpec {
            slot: G2_PRODUCER_OPAQUE_EVENT_COUNT_SLOT,
            kind: G2_LANE_OPAQUE_EVENT_COUNT,
            width: 4,
            flags: atomic,
            group: G2_GROUP_RESIDUAL,
        },
    ];
    for kind in 0u8..30 {
        for value_lane in [false, true] {
            let Some(width) = g2_standard_lane_width(kind, value_lane) else {
                continue;
            };
            let load_store = G2_LOAD_STORE_KINDS.contains(&kind);
            specs.push(LaneSpec {
                slot: g2_standard_producer_slot(kind, value_lane)
                    .expect("standard lane has a producer slot"),
                kind: if value_lane {
                    g2_lane_v1(kind)
                } else {
                    g2_lane_v0(kind)
                },
                width,
                flags: if load_store { atomic } else { required },
                group: if load_store { G2_GROUP_LOAD_STORE } else { 0 },
            });
        }
    }
    specs.sort_unstable_by_key(|spec| spec.slot);
    specs
}

fn lane_capacity(spec: LaneSpec, capacities: &RvrG2CapacitiesV1) -> u32 {
    match spec.slot {
        G2_PRODUCER_RUN_SLOT => capacities.run,
        G2_PRODUCER_RESIDUAL_CTRL_SLOT
        | G2_PRODUCER_RESIDUAL_TAG_SLOT
        | G2_PRODUCER_RESIDUAL_VALUE_SLOT => capacities.residual,
        G2_PRODUCER_OPAQUE_EVENT_COUNT_SLOT => capacities.opaque_events,
        _ => (0u8..30)
            .find(|&kind| {
                [false, true].into_iter().any(|value_lane| {
                    g2_standard_producer_slot(kind, value_lane) == Some(spec.slot)
                })
            })
            .map(|kind| capacities.kinds[kind as usize])
            .unwrap_or(0),
    }
}

fn lane_spec(kind: u16) -> Option<LaneSpec> {
    producer_lane_specs()
        .into_iter()
        .find(|spec| spec.kind == kind)
}

fn validate_desc(
    desc: &G2LaneDescV1,
    spec: LaneSpec,
    segment_len: usize,
    payload_begin: usize,
) -> Result<(), ExecutionError> {
    let expected_payload = desc
        .count
        .checked_mul(u32::from(spec.width))
        .ok_or_else(|| g2_error("lane payload byte count overflow"))?;
    if desc.kind != spec.kind
        || desc.elem_width != spec.width
        || desc.encoding != G2_ENCODING_FIXED_LE
        || desc.flags != spec.flags
        || desc.payload_bytes != expected_payload
        || desc.offset < payload_begin as u64
        || !(desc.offset as usize).is_multiple_of(G2_WIRE_ALIGNMENT)
        || desc.group_id != spec.group
        || desc.reserved != 0
        || desc
            .offset
            .checked_add(u64::from(desc.payload_bytes))
            .is_none_or(|end| end > segment_len as u64)
    {
        return Err(g2_error("lane descriptor validation failed"));
    }
    Ok(())
}

fn validate_opaque_desc(
    desc: &G2LaneDescV1,
    segment_len: usize,
    payload_begin: usize,
) -> Result<(), ExecutionError> {
    if desc.elem_width != 0
        || desc.encoding != G2_ENCODING_OPAQUE_FINAL
        || desc.flags != G2_LANE_FLAG_OPAQUE_FINAL
        || desc.offset < payload_begin as u64
        || !(desc.offset as usize).is_multiple_of(G2_WIRE_ALIGNMENT)
        || desc.group_id != 0
        || desc.reserved != 0
        || desc
            .offset
            .checked_add(u64::from(desc.payload_bytes))
            .is_none_or(|end| end > segment_len as u64)
    {
        return Err(g2_error("opaque lane descriptor validation failed"));
    }
    Ok(())
}

fn validate_atomic_members(
    descs: &[G2LaneDescV1],
    residual_count: u32,
) -> Result<(), ExecutionError> {
    let by_kind = descs
        .iter()
        .map(|desc| (desc.kind, desc))
        .collect::<BTreeMap<_, _>>();
    for kind in [
        G2_LANE_RESIDUAL_CTRL,
        G2_LANE_RESIDUAL_TAG,
        G2_LANE_RESIDUAL_VALUE,
    ] {
        if (residual_count != 0) != by_kind.contains_key(&kind) {
            return Err(g2_error("residual atomic group is incomplete"));
        }
    }
    for kind in G2_LOAD_STORE_KINDS {
        let pointer = by_kind.get(&g2_lane_v0(kind));
        let block = by_kind.get(&g2_lane_v1(kind));
        if pointer.is_some() != block.is_some()
            || pointer.zip(block).is_some_and(|(a, b)| a.count != b.count)
        {
            return Err(g2_error(format!(
                "load/store kind {kind} atomic group is incomplete"
            )));
        }
    }
    Ok(())
}

fn align_up(value: usize, alignment: usize) -> Result<usize, ExecutionError> {
    value
        .checked_add(alignment - 1)
        .map(|value| value / alignment * alignment)
        .ok_or_else(|| g2_error("wire alignment overflow"))
}

fn g2_error(message: impl Into<String>) -> ExecutionError {
    ExecutionError::RvrExecution(format!("G2 wire v1: {}", message.into()))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use rvr_openvm_ext_ffi_common::G2_PRODUCER_ADDI_SLOT;

    use super::*;

    fn test_meta(bindings: Vec<RvrG2AirBindingV1>) -> RvrG2MetaV1 {
        RvrG2MetaV1 {
            fingerprint: [0; 32],
            producer_schema_fingerprint: [0; 32],
            emission_mode: 0,
            program_fingerprint: [0; 32],
            block_fingerprint: [0; 32],
            air_manifest_fingerprint: [0; 32],
            blocks: Arc::new(Vec::new()),
            air_bindings: Arc::new(bindings),
            opaque_bindings: Arc::new(Vec::new()),
        }
    }

    #[test]
    fn metered_capacity_uses_one_real_joint_shape() {
        let meta = test_meta(vec![
            RvrG2AirBindingV1 {
                kind: 0,
                air_idx: 1,
            },
            RvrG2AirBindingV1 {
                kind: 29,
                air_idx: 2,
            },
            RvrG2AirBindingV1 {
                kind: 30,
                air_idx: 3,
            },
        ]);
        let first =
            RvrG2CapacitiesV1::for_metered_segment(&meta, &[0, 1_000, 0, 0], 1_000).unwrap();
        let second =
            RvrG2CapacitiesV1::for_metered_segment(&meta, &[0, 0, 1_000, 64], 1_000).unwrap();
        assert_eq!(first.residual, 2 * 1_016 + 64);
        assert_eq!(second.residual, 2 * 1_016 + 64 + 3 * 64);

        let mut impossible = first.clone();
        impossible.kinds[29] = second.kinds[29];
        impossible.kinds[30] = second.kinds[30];
        let joint_max = RvrG2PreparedV1::capacity_bytes(&first)
            .unwrap()
            .max(RvrG2PreparedV1::capacity_bytes(&second).unwrap());
        assert!(RvrG2PreparedV1::capacity_bytes(&impossible).unwrap() > joint_max);
    }

    #[test]
    fn metered_capacity_keeps_opaque_residual_bound_fail_closed() {
        let mut meta = test_meta(Vec::new());
        meta.opaque_bindings = Arc::new(vec![RvrG2OpaqueBindingV1 {
            air_idx: 2,
            max_residual_events_per_record: 32,
            geometry: super::super::ArenaNativeGeometry {
                adapter_size: 16,
                adapter_align: 8,
                core_size: 0,
                core_align: 1,
                core_off_matrix: 0,
                layout: super::super::ArenaNativeLayout::Custom {
                    residual_memory_chronology: true,
                    layout_id: "openvm.rvr.test-capacity-opaque.v1",
                },
            },
            air_identity_digest: [0; 32],
            layout_digest: [0; 32],
        }]);
        let capacities = RvrG2CapacitiesV1::for_metered_segment(&meta, &[0, 0, 128], 100).unwrap();
        assert_eq!(capacities.opaque_events, 116);
        assert_eq!(capacities.residual, 2 * 116 + 64 + 32 * 128);
    }

    #[test]
    fn metered_capacity_covers_narrow_reveals_beyond_slack() {
        let meta = test_meta(vec![RvrG2AirBindingV1 {
            kind: 23,
            air_idx: 1,
        }]);
        let capacities = RvrG2CapacitiesV1::for_metered_segment(&meta, &[0, 128], 128).unwrap();

        // 128 possible narrow reveals exceed the old 96-event run/slack
        // allowance. Each may consume a third residual event.
        assert_eq!(capacities.residual, 2 * 144 + 128 + 64);
        assert!(capacities.residual >= 3 * 128);
    }

    #[test]
    fn metered_standard_cursor_mismatch_is_rejected_at_publish() {
        let mut capacities = RvrG2CapacitiesV1::default();
        capacities.kinds[29] = 2;
        let pool = crate::arch::rvr::RvrPreflightBufferPool::default();
        let mut prepared = RvrG2PreparedV1::new_pooled(&capacities, &pool).unwrap();
        prepared.lanes[G2_PRODUCER_ADDI_SLOT].len = 1;
        prepared.lanes[G2_PRODUCER_ADDI_SLOT].expected_len = 1;
        let mut expected = [0u32; 31];
        expected[29] = 2;

        let error = prepared
            .finalize(0, 0, Some(&expected), [0; 32], &[])
            .unwrap_err();
        assert!(error
            .to_string()
            .contains("standard lane cursor differs from metered count"));
    }

    #[test]
    fn entered_block_span_cursor_mismatch_is_rejected_at_publish() {
        let mut capacities = RvrG2CapacitiesV1::default();
        capacities.kinds[29] = 2;
        let pool = crate::arch::rvr::RvrPreflightBufferPool::default();
        let mut prepared = RvrG2PreparedV1::new_pooled(&capacities, &pool).unwrap();
        prepared.lanes[G2_PRODUCER_ADDI_SLOT].len = 1;
        prepared.lanes[G2_PRODUCER_ADDI_SLOT].expected_len = 2;
        let mut expected = [0u32; 31];
        expected[29] = 1;

        let error = prepared
            .finalize(0, 0, Some(&expected), [0; 32], &[])
            .unwrap_err();
        assert!(error
            .to_string()
            .contains("static lane cursor differs from entered block spans"));
    }

    #[test]
    fn phase1_transport_release_acquire_and_exact_cursors() {
        let mut capacities = RvrG2CapacitiesV1 {
            run: 1,
            ..Default::default()
        };
        capacities.kinds[29] = 2;
        let mut prepared = RvrG2PreparedV1::new(&capacities).unwrap();
        let addi_offset = prepared.lanes[G2_PRODUCER_ADDI_SLOT].offset as usize;
        let run_offset = prepared.lanes[G2_PRODUCER_RUN_SLOT].offset as usize;
        unsafe {
            prepared
                .producer
                .base
                .add(addi_offset)
                .cast::<u64>()
                .write(7);
            prepared
                .producer
                .base
                .add(addi_offset + 8)
                .cast::<u64>()
                .write(11);
            prepared
                .producer
                .base
                .add(run_offset)
                .cast::<u32>()
                .write(3);
        }
        prepared.lanes[G2_PRODUCER_ADDI_SLOT].len = 2;
        prepared.lanes[G2_PRODUCER_RUN_SLOT].len = 1;
        prepared.producer.instruction_count = 2;
        let fingerprint = [0x5a; 32];
        let segment = prepared.finalize(9, 2, None, fingerprint, &[]).unwrap();
        let descs = segment.validate(&fingerprint).unwrap();
        assert_eq!(segment.header_acquire().unwrap().segment_id, 9);
        assert_eq!(
            segment.lane(&descs, G2_LANE_RUN_BLOCK_ID).unwrap(),
            3u32.to_le_bytes()
        );
        assert_eq!(
            segment.lane(&descs, G2_LANE_ADDI_V0).unwrap(),
            [7u64, 11].map(u64::to_le_bytes).concat()
        );
    }

    #[test]
    fn phase1_transport_rejects_partial_or_bad_cursor() {
        let prepared = RvrG2PreparedV1::new(&RvrG2CapacitiesV1 {
            run: 1,
            ..Default::default()
        })
        .unwrap();
        let partial = RvrG2SegmentV1 {
            backing: Some(vec![0; prepared.byte_capacity + 32]),
            backing_offset: 0,
            backing_capacity: prepared.byte_capacity,
            pool: crate::arch::rvr::RvrPreflightBufferPool::default(),
            backing_route: crate::arch::rvr::preflight_pool::G2BackingRoute::Checked,
            byte_len: prepared.byte_capacity,
            header_prefix_len: G2_SEGMENT_HEADER_V1_SIZE,
            committed_lanes: Vec::new(),
        };
        assert!(partial.header_acquire().is_err());
        assert!(prepared.finalize(0, 1, None, [0; 32], &[]).is_err());
    }

    #[test]
    fn phase1_transport_rejects_schema_mismatch() {
        let mut prepared = RvrG2PreparedV1::new(&RvrG2CapacitiesV1::default()).unwrap();
        prepared.producer.instruction_count = 0;
        let segment = prepared.finalize(0, 0, None, [1; 32], &[]).unwrap();
        assert!(segment.validate(&[2; 32]).is_err());
    }

    #[test]
    fn phase2a_transport_rejects_partial_atomic_groups() {
        let mut capacities = RvrG2CapacitiesV1 {
            run: 1,
            residual: 1,
            ..Default::default()
        };
        capacities.kinds[26] = 1;

        let mut load_store = RvrG2PreparedV1::new(&capacities).unwrap();
        let store_slot = g2_load_store_producer_slot(26, false).unwrap();
        load_store.lanes[store_slot].len = 1;
        assert!(load_store.finalize(0, 0, None, [0; 32], &[]).is_err());

        let mut residual = RvrG2PreparedV1::new(&capacities).unwrap();
        residual.lanes[G2_PRODUCER_RESIDUAL_CTRL_SLOT].len = 1;
        residual.lanes[G2_PRODUCER_RESIDUAL_VALUE_SLOT].len = 1;
        assert!(residual.finalize(0, 0, None, [0; 32], &[]).is_err());
    }

    #[test]
    fn phase2a_transport_rejects_overflow_before_publish() {
        let mut prepared = RvrG2PreparedV1::new(&RvrG2CapacitiesV1::default()).unwrap();
        prepared.producer.overflow = 1;
        assert!(prepared.finalize(0, 0, None, [0; 32], &[]).is_err());
    }

    #[test]
    fn phase2a_transport_rejects_payload_inside_descriptor_table() {
        let capacities = RvrG2CapacitiesV1 {
            residual: 1,
            ..Default::default()
        };
        let mut prepared = RvrG2PreparedV1::new(&capacities).unwrap();
        prepared.lanes[G2_PRODUCER_RESIDUAL_CTRL_SLOT].len = 1;
        prepared.lanes[G2_PRODUCER_RESIDUAL_TAG_SLOT].len = 1;
        prepared.lanes[G2_PRODUCER_RESIDUAL_VALUE_SLOT].len = 1;
        let fingerprint = [3; 32];
        let mut segment = prepared.finalize(0, 0, None, fingerprint, &[]).unwrap();
        unsafe {
            // SAFETY: the test owns the segment and mutates its second POD
            // descriptor before exposing it to any consumer.
            let descriptor = segment
                .backing
                .as_mut()
                .expect("test segment backing must still be owned")
                .as_mut_ptr()
                .cast::<u8>()
                .add(G2_SEGMENT_HEADER_V1_SIZE + G2_LANE_DESC_V1_SIZE)
                .cast::<G2LaneDescV1>();
            (*descriptor).offset = G2_WIRE_ALIGNMENT as u64;
        }
        assert!(segment.validate(&fingerprint).is_err());
    }

    #[test]
    fn phase2a_wire_length_excludes_unused_lane_capacity() {
        let mut capacities = RvrG2CapacitiesV1 {
            run: 1_000,
            residual: 1_000,
            ..Default::default()
        };
        capacities.kinds[26] = 1_000;
        capacities.kinds[29] = 1_000;
        let mut prepared = RvrG2PreparedV1::new(&capacities).unwrap();
        prepared.lanes[G2_PRODUCER_RUN_SLOT].len = 1;
        prepared.lanes[G2_PRODUCER_RESIDUAL_CTRL_SLOT].len = 3;
        prepared.lanes[G2_PRODUCER_RESIDUAL_TAG_SLOT].len = 3;
        prepared.lanes[G2_PRODUCER_RESIDUAL_VALUE_SLOT].len = 3;
        prepared.lanes[G2_PRODUCER_ADDI_SLOT].len = 2;
        let store_slot = g2_load_store_producer_slot(26, false).unwrap();
        prepared.lanes[store_slot].len = 1;
        prepared.lanes[store_slot + 1].len = 1;
        prepared.producer.instruction_count = 4;
        let fingerprint = [4; 32];
        let segment = prepared.finalize(0, 4, None, fingerprint, &[]).unwrap();
        let descs = segment.validate(&fingerprint).unwrap();

        let mut expected_len = align_up(
            G2_SEGMENT_HEADER_V1_SIZE + descs.len() * G2_LANE_DESC_V1_SIZE,
            G2_WIRE_ALIGNMENT,
        )
        .unwrap();
        for desc in descs {
            assert_eq!(desc.offset as usize, expected_len);
            expected_len = align_up(
                expected_len + desc.payload_bytes as usize,
                G2_WIRE_ALIGNMENT,
            )
            .unwrap();
        }
        assert_eq!(
            segment.byte_len(),
            expected_len,
            "wire transfer must contain only active payload prefixes and frozen alignment"
        );
    }

    #[test]
    fn opaque_final_payload_is_logical_but_not_staged_twice() {
        let mut prepared = RvrG2PreparedV1::new(&RvrG2CapacitiesV1::default()).unwrap();
        prepared.producer.instruction_count = 0;
        let binding = RvrG2OpaqueBindingV1 {
            air_idx: 7,
            geometry: super::super::ArenaNativeGeometry {
                adapter_size: 16,
                adapter_align: 8,
                core_size: 0,
                core_align: 1,
                core_off_matrix: 0,
                layout: super::super::ArenaNativeLayout::Custom {
                    residual_memory_chronology: true,
                    max_residual_events_per_record: 0,
                    layout_id: "openvm.rvr.test-opaque-final.v1",
                },
            },
            max_residual_events_per_record: 0,
            air_identity_digest: [0x3c; 32],
            layout_digest: [0x5a; 32],
        };
        let fingerprint = [7; 32];
        let segment = prepared
            .finalize(0, 0, None, fingerprint, &[(binding, 4, 64)])
            .unwrap();
        segment.validate(&fingerprint).unwrap();
        assert!(segment.transfer_byte_len() < segment.byte_len());
        assert_eq!(
            segment
                .wire_parts()
                .map(|(_, bytes)| bytes.len())
                .sum::<usize>(),
            128
        );
    }
}
