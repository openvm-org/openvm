//! G2 private compact-wire transport.
//!
//! The generated native producer writes lane payloads directly into this
//! backing. Finalization is deliberately O(lanes): it validates cursors,
//! writes the active descriptors and header, and release-publishes COMMITTED.

use std::{
    collections::BTreeMap,
    mem::size_of,
    sync::atomic::{AtomicU16, AtomicU32, Ordering},
};

use rvr_openvm_ext_ffi_common::{
    g2_lane_v0, g2_lane_v1, g2_load_store_producer_slot, G2_ENCODING_FIXED_LE, G2_FLAGS_V1,
    G2_FLAG_COMMITTED, G2_GROUP_LOAD_STORE, G2_GROUP_RESIDUAL, G2_LANE_ADDI_V0,
    G2_LANE_DESC_V1_SIZE, G2_LANE_FLAG_ATOMIC_GROUP, G2_LANE_FLAG_REQUIRED, G2_LANE_RESIDUAL_CTRL,
    G2_LANE_RESIDUAL_TAG, G2_LANE_RESIDUAL_VALUE, G2_LANE_RUN_BLOCK_ID, G2_LOAD_STORE_KINDS,
    G2_PRODUCER_ADDI_SLOT, G2_PRODUCER_LANE_COUNT, G2_PRODUCER_RESIDUAL_CTRL_SLOT,
    G2_PRODUCER_RESIDUAL_TAG_SLOT, G2_PRODUCER_RESIDUAL_VALUE_SLOT, G2_PRODUCER_RUN_SLOT,
    G2_SEGMENT_HEADER_V1_SIZE, G2_SEGMENT_MAGIC_V1, G2_WIRE_ALIGNMENT, G2_WIRE_VERSION_V1,
};
pub use rvr_openvm_ext_ffi_common::{
    G2LaneDescV1, G2ProducerLaneV1, G2ProducerV1, G2SegmentHeaderV1,
};

use super::{RvrDeltaDecodeEntry, RvrDeltaDecodePrecompute, PREFLIGHT_ADDSUB_RECORD_SIZE};
use crate::arch::ExecutionError;

static NEXT_G2_SEGMENT_ID: AtomicU32 = AtomicU32::new(0);

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

#[derive(Clone, Debug)]
pub struct RvrG2MetaV1 {
    pub fingerprint: [u8; 32],
    pub program_fingerprint: [u8; 32],
    pub block_fingerprint: [u8; 32],
    pub air_manifest_fingerprint: [u8; 32],
    pub blocks: std::sync::Arc<Vec<RvrG2BlockEntryV1>>,
    /// Sorted stable decoder-kind to global AIR bindings admitted by this
    /// executable. Phase 2a admits AddI and the complete LoadStore family.
    pub air_bindings: std::sync::Arc<Vec<RvrG2AirBindingV1>>,
}

impl RvrG2MetaV1 {
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

#[derive(Clone, Debug, Default)]
pub struct RvrG2CapacitiesV1 {
    pub run: u32,
    pub residual: u32,
    /// Per `DeltaAirKind`; unsupported entries must remain zero.
    pub kinds: [u32; 30],
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

/// Committed compact segment. The u128 allocation gives native u64 lane
/// stores sufficient alignment; all public offsets remain segment-relative.
pub struct RvrG2SegmentV1 {
    backing: Vec<u128>,
    byte_len: usize,
    header_prefix_len: usize,
    committed_lanes: Vec<RvrG2CommittedLaneV1>,
}

#[derive(Clone, Copy, Debug)]
struct RvrG2CommittedLaneV1 {
    kind: u16,
    source_offset: usize,
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

    /// Return the compact wire as scatter/gather pieces. The generated C
    /// writes each lane once into its capacity-bounded staging range; the
    /// O(lanes) finalizer assigns tight wire offsets without copying payloads.
    /// CUDA uploads these pieces directly to their final device offsets.
    pub fn wire_parts(&self) -> impl Iterator<Item = (usize, &[u8])> {
        std::iter::once((0, self.backing_bytes_prefix(self.header_prefix_len))).chain(
            self.committed_lanes.iter().map(|lane| {
                (
                    lane.wire_offset,
                    self.backing_bytes_range(lane.source_offset, lane.payload_bytes),
                )
            }),
        )
    }

    fn backing_bytes_prefix(&self, len: usize) -> &[u8] {
        self.backing_bytes_range(0, len)
    }

    fn backing_bytes_range(&self, offset: usize, len: usize) -> &[u8] {
        debug_assert!(offset + len <= self.backing.len() * size_of::<u128>());
        unsafe {
            // SAFETY: the backing is initialized before native execution and
            // every committed source range was bounds-checked by finalization.
            std::slice::from_raw_parts(self.backing.as_ptr().cast::<u8>().add(offset), len)
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
            || lane_count > G2_PRODUCER_LANE_COUNT
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
            let spec = lane_spec(desc.kind)
                .ok_or_else(|| g2_error(format!("unknown lane kind {:#06x}", desc.kind)))?;
            validate_desc(&desc, spec, self.byte_len, payload_begin)?;
            let committed = self
                .committed_lanes
                .iter()
                .find(|lane| lane.kind == desc.kind)
                .ok_or_else(|| g2_error("descriptor has no committed producer lane"))?;
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
        self.backing_bytes_range(lane.source_offset, lane.payload_bytes)
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

/// Oracle-only Phase-2a decoder. It walks program chronology on the CPU only
/// in tests and reconstructs the established 44-byte consumer records for
/// AddI and every LoadStore kind, including residual narrow REVEAL rows.
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
    let header = segment.header_acquire()?;
    let mut registers = initial_registers;
    registers[0] = 0;
    let mut timestamps = BTreeMap::new();
    let mut blocks = initial_blocks.clone();
    for (reg, &value) in registers.iter().enumerate() {
        blocks.insert((1, reg as u32 * 8), value);
    }
    let mut kind_cursors = [0usize; 30];
    let mut residual_cursor = 0usize;
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
                .find(|binding| binding.air_idx == entry.air_idx as usize)
                .ok_or_else(|| g2_error(format!("slot {slot} references an unbound AIR")))?;
            let kind = binding.kind;
            let from_timestamp = timestamp;
            let mut record = Vec::with_capacity(PREFLIGHT_ADDSUB_RECORD_SIZE);
            append_u32(&mut record, decode.pc_base + slot * 4);
            append_u32(&mut record, from_timestamp);

            if kind == 29 && entry.access_pattern == 8 {
                let lane = segment
                    .lane(&descs, G2_LANE_ADDI_V0)
                    .ok_or_else(|| g2_error("missing AddI lane"))?;
                let rs1 = register_index(entry.b)?;
                let rd = register_index(entry.a)?;
                let value = lane_u64(lane, kind_cursors[kind as usize], "AddI")?;
                kind_cursors[kind as usize] += 1;
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
            } else if G2_LOAD_STORE_KINDS.contains(&kind) && matches!(entry.access_pattern, 2 | 3) {
                let a = register_index(entry.a)?;
                let base = register_index(entry.b)?;
                let is_store = entry.access_pattern == 3;
                let narrow_reveal =
                    is_store && entry.flags & (1 << 4) != 0 && entry.local_opcode != 4;
                let (pointer, block_value, source, expected_post) = if narrow_reveal {
                    let ctrl =
                        residual_ctrl.ok_or_else(|| g2_error("missing residual CTRL lane"))?;
                    let tags = residual_tag.ok_or_else(|| g2_error("missing residual TAG lane"))?;
                    let values =
                        residual_value.ok_or_else(|| g2_error("missing residual VALUE lane"))?;
                    let pointer = u32::try_from(registers[base])
                        .map_err(|_| g2_error("narrow REVEAL pointer exceeds u32"))?;
                    let source = registers[a];
                    let effective = effective_address(pointer, entry);
                    let residual_lanes = (ctrl, tags, values);
                    validate_residual(
                        residual_lanes,
                        residual_cursor,
                        timestamp,
                        entry.b,
                        0x40,
                        Some(u64::from(pointer)),
                    )?;
                    validate_residual(
                        residual_lanes,
                        residual_cursor + 1,
                        timestamp + 1,
                        entry.a,
                        0x40,
                        Some(source),
                    )?;
                    let post = validate_residual(
                        residual_lanes,
                        residual_cursor + 2,
                        timestamp + 2,
                        effective & !7,
                        0x49,
                        None,
                    )?;
                    residual_cursor += 3;
                    let key = (3, effective & !7);
                    let previous = *blocks.get(&key).ok_or_else(|| {
                        g2_error(format!("narrow REVEAL has no initial block at {key:?}"))
                    })?;
                    (pointer, previous, source, Some(post))
                } else {
                    let v0 = segment
                        .lane(&descs, g2_lane_v0(kind))
                        .ok_or_else(|| g2_error(format!("kind {kind} missing V0 lane")))?;
                    let v1 = segment
                        .lane(&descs, g2_lane_v1(kind))
                        .ok_or_else(|| g2_error(format!("kind {kind} missing V1 lane")))?;
                    let cursor = kind_cursors[kind as usize];
                    kind_cursors[kind as usize] += 1;
                    let pointer = lane_u32(v0, cursor, "load/store pointer")?;
                    if u64::from(pointer) != registers[base] {
                        return Err(g2_error(format!(
                            "kind {kind} pointer mismatch at slot {slot}"
                        )));
                    }
                    let block = lane_u64(v1, cursor, "load/store block")?;
                    (pointer, block, registers[a], None)
                };
                let address_space = if entry.flags & (1 << 4) != 0 { 3 } else { 2 };
                let effective = effective_address(pointer, entry);
                let memory_key = (address_space, effective & !7);
                if let Some(current) = blocks.get(&memory_key) {
                    if *current != block_value {
                        return Err(g2_error(format!(
                            "kind {kind} aligned block mismatch at slot {slot}"
                        )));
                    }
                } else {
                    blocks.insert(memory_key, block_value);
                }
                let read0_prev = touch(&mut timestamps, (1, entry.b), timestamp);
                let read1_key = if is_store { (1, entry.a) } else { memory_key };
                let read1_prev = touch(&mut timestamps, read1_key, timestamp + 1);
                let (write_prev, write_prev_value) = if is_store {
                    let previous = *blocks
                        .get(&memory_key)
                        .ok_or_else(|| g2_error("store block disappeared during replay"))?;
                    let write_prev = touch(&mut timestamps, memory_key, timestamp + 2);
                    let patched = patch_store(
                        previous,
                        effective,
                        store_width(entry.local_opcode)?,
                        source,
                    );
                    if expected_post.is_some_and(|expected| expected != patched) {
                        return Err(g2_error(format!(
                            "narrow REVEAL post-block mismatch at slot {slot}"
                        )));
                    }
                    blocks.insert(memory_key, patched);
                    (write_prev, previous)
                } else if entry.flags & (1 << 2) != 0 {
                    let previous = registers[a];
                    let write_prev = touch(&mut timestamps, (1, entry.a), timestamp + 2);
                    let loaded = decode_load(entry.local_opcode, block_value, effective)?;
                    if a != 0 {
                        registers[a] = loaded;
                        blocks.insert((1, entry.a), loaded);
                    }
                    (write_prev, previous)
                } else {
                    (0, 0)
                };
                timestamp = timestamp
                    .checked_add(3)
                    .ok_or_else(|| g2_error("timestamp overflow"))?;
                append_u32(&mut record, read0_prev);
                append_u32(&mut record, read1_prev);
                append_u32(&mut record, write_prev);
                append_u64(&mut record, write_prev_value);
                append_u64(&mut record, u64::from(pointer));
                append_u64(&mut record, if is_store { source } else { block_value });
            } else {
                return Err(g2_error(format!("slot {slot} is outside Phase 2a")));
            }
            if record.len() != PREFLIGHT_ADDSUB_RECORD_SIZE {
                return Err(g2_error("reference decoder produced a bad compact stride"));
            }
            compact_records.entry(kind).or_default().extend(record);
        }
    }

    if expanded_program_slots.len() != header.instruction_count as usize
        || residual_cursor != header.residual_event_count as usize
    {
        return Err(g2_error(
            "program or residual cursor did not finish exactly",
        ));
    }
    for kind in G2_LOAD_STORE_KINDS {
        let lane_count = descs
            .iter()
            .find(|desc| desc.kind == g2_lane_v0(kind))
            .map(|desc| desc.count as usize)
            .unwrap_or(0);
        if kind_cursors[kind as usize] != lane_count {
            return Err(g2_error(format!(
                "kind {kind} cursor did not finish exactly"
            )));
        }
    }
    let addi_count = descs
        .iter()
        .find(|desc| desc.kind == G2_LANE_ADDI_V0)
        .map(|desc| desc.count as usize)
        .unwrap_or(0);
    if kind_cursors[29] != addi_count {
        return Err(g2_error("AddI cursor did not finish exactly"));
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

fn patch_store(block: u64, address: u32, width: u8, source: u64) -> u64 {
    let shift = (address & 7) * 8;
    let mask = if width == 8 {
        u64::MAX
    } else {
        (1u64 << (u32::from(width) * 8)) - 1
    };
    (block & !(mask << shift)) | ((source & mask) << shift)
}

fn decode_load(local_opcode: u8, block: u64, address: u32) -> Result<u64, ExecutionError> {
    let shifted = block >> ((address & 7) * 8);
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

pub(crate) struct RvrG2PreparedV1 {
    backing: Vec<u128>,
    byte_capacity: usize,
    lanes: Vec<G2ProducerLaneV1>,
    pub producer: G2ProducerV1,
}

impl RvrG2PreparedV1 {
    pub fn new(capacities: &RvrG2CapacitiesV1) -> Result<Self, ExecutionError> {
        let specs = producer_lane_specs();
        debug_assert_eq!(specs.len(), G2_PRODUCER_LANE_COUNT);
        let mut offset = align_up(
            G2_SEGMENT_HEADER_V1_SIZE + G2_PRODUCER_LANE_COUNT * G2_LANE_DESC_V1_SIZE,
            G2_WIRE_ALIGNMENT,
        )?;
        let mut lanes = Vec::with_capacity(specs.len());
        for (slot, spec) in specs.iter().enumerate() {
            debug_assert_eq!(slot, spec.slot);
            let cap = lane_capacity(*spec, capacities);
            lanes.push(G2ProducerLaneV1 {
                offset: offset as u64,
                len: 0,
                cap,
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
        let byte_capacity = offset;
        let words = byte_capacity.div_ceil(size_of::<u128>());
        let mut backing = vec![0u128; words];
        let producer = G2ProducerV1 {
            base: backing.as_mut_ptr().cast(),
            capacity: byte_capacity as u64,
            lanes: lanes.as_mut_ptr(),
            lane_count: G2_PRODUCER_LANE_COUNT as u32,
            instruction_count: 0,
            overflow: 0,
            reserved: 0,
        };
        Ok(Self {
            backing,
            byte_capacity,
            lanes,
            producer,
        })
    }

    pub fn finalize(
        mut self,
        segment_id: u32,
        expected_instruction_count: u32,
        fingerprint: [u8; 32],
    ) -> Result<RvrG2SegmentV1, ExecutionError> {
        let p = self.producer;
        if p.overflow != 0
            || p.base != self.backing.as_mut_ptr().cast()
            || p.capacity as usize != self.byte_capacity
            || p.lanes != self.lanes.as_mut_ptr()
            || p.lane_count as usize != self.lanes.len()
            || p.instruction_count != expected_instruction_count
            || p.reserved != 0
            || self.lanes.iter().any(|lane| lane.len > lane.cap)
            || (p.instruction_count != 0 && self.lanes[G2_PRODUCER_RUN_SLOT].len == 0)
        {
            return Err(g2_error("native lane cursor/count validation failed"));
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
        let header_bytes = G2_SEGMENT_HEADER_V1_SIZE + active_lanes.len() * G2_LANE_DESC_V1_SIZE;
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
                source_offset,
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
        let byte_len = wire_offset;
        let header = G2SegmentHeaderV1 {
            magic: G2_SEGMENT_MAGIC_V1,
            version: G2_WIRE_VERSION_V1,
            header_bytes: u16::try_from(header_bytes)
                .map_err(|_| g2_error("descriptor table exceeds u16 header size"))?,
            lane_count: descs.len() as u16,
            flags: G2_FLAGS_V1,
            segment_id,
            instruction_count: p.instruction_count,
            run_count: self.lanes[G2_PRODUCER_RUN_SLOT].len,
            residual_event_count: residual_count,
            schema_fingerprint: fingerprint,
        };
        unsafe {
            // SAFETY: the u128 backing is aligned for both POD structs and the
            // fixed layout reserves the maximum descriptor table before the
            // first possible payload offset.
            let base = self.backing.as_mut_ptr().cast::<u8>();
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
            backing: self.backing,
            byte_len,
            header_prefix_len: payload_begin,
            committed_lanes,
        };
        segment.validate(&fingerprint)?;
        Ok(segment)
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
            slot: G2_PRODUCER_ADDI_SLOT,
            kind: G2_LANE_ADDI_V0,
            width: 8,
            flags: required,
            group: 0,
        },
    ];
    for kind in G2_LOAD_STORE_KINDS {
        let pointer_slot =
            g2_load_store_producer_slot(kind, false).expect("frozen kind has producer slot");
        specs.push(LaneSpec {
            slot: pointer_slot,
            kind: g2_lane_v0(kind),
            width: 4,
            flags: atomic,
            group: G2_GROUP_LOAD_STORE,
        });
        specs.push(LaneSpec {
            slot: pointer_slot + 1,
            kind: g2_lane_v1(kind),
            width: 8,
            flags: atomic,
            group: G2_GROUP_LOAD_STORE,
        });
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
        G2_PRODUCER_ADDI_SLOT => capacities.kinds[29],
        _ => G2_LOAD_STORE_KINDS
            .iter()
            .copied()
            .find(|&kind| {
                g2_load_store_producer_slot(kind, false)
                    .is_some_and(|slot| spec.slot == slot || spec.slot == slot + 1)
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
    use super::*;

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
        let segment = prepared.finalize(9, 2, fingerprint).unwrap();
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
            backing: prepared.backing.clone(),
            byte_len: prepared.byte_capacity,
            header_prefix_len: G2_SEGMENT_HEADER_V1_SIZE,
            committed_lanes: Vec::new(),
        };
        assert!(partial.header_acquire().is_err());
        assert!(prepared.finalize(0, 1, [0; 32]).is_err());
    }

    #[test]
    fn phase1_transport_rejects_schema_mismatch() {
        let mut prepared = RvrG2PreparedV1::new(&RvrG2CapacitiesV1::default()).unwrap();
        prepared.producer.instruction_count = 0;
        let segment = prepared.finalize(0, 0, [1; 32]).unwrap();
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
        assert!(load_store.finalize(0, 0, [0; 32]).is_err());

        let mut residual = RvrG2PreparedV1::new(&capacities).unwrap();
        residual.lanes[G2_PRODUCER_RESIDUAL_CTRL_SLOT].len = 1;
        residual.lanes[G2_PRODUCER_RESIDUAL_VALUE_SLOT].len = 1;
        assert!(residual.finalize(0, 0, [0; 32]).is_err());
    }

    #[test]
    fn phase2a_transport_rejects_overflow_before_publish() {
        let mut prepared = RvrG2PreparedV1::new(&RvrG2CapacitiesV1::default()).unwrap();
        prepared.producer.overflow = 1;
        assert!(prepared.finalize(0, 0, [0; 32]).is_err());
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
        let mut segment = prepared.finalize(0, 0, fingerprint).unwrap();
        unsafe {
            // SAFETY: the test owns the segment and mutates its second POD
            // descriptor before exposing it to any consumer.
            let descriptor = segment
                .backing
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
        let segment = prepared.finalize(0, 4, fingerprint).unwrap();
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
}
