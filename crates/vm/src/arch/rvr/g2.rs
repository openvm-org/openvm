//! G2 private compact-wire transport.
//!
//! The generated native producer writes lane payloads directly into this
//! backing. Finalization is deliberately O(lanes): it validates cursors,
//! writes the two descriptors and header, and release-publishes COMMITTED.

use std::{
    mem::size_of,
    sync::atomic::{AtomicU16, AtomicU32, Ordering},
};

pub use rvr_openvm_ext_ffi_common::{G2LaneDescV1, G2ProducerV1, G2SegmentHeaderV1};
use rvr_openvm_ext_ffi_common::{
    G2_ENCODING_FIXED_LE, G2_FLAGS_V1, G2_FLAG_COMMITTED, G2_LANE_ADDI_V0, G2_LANE_DESC_V1_SIZE,
    G2_LANE_FLAG_REQUIRED, G2_LANE_RUN_BLOCK_ID, G2_SEGMENT_HEADER_V1_SIZE, G2_SEGMENT_MAGIC_V1,
    G2_WIRE_ALIGNMENT, G2_WIRE_VERSION_V1,
};

use super::{RvrDeltaDecodePrecompute, PREFLIGHT_ADDSUB_RECORD_SIZE};
use crate::arch::ExecutionError;

const G2_PHASE1_LANE_COUNT: usize = 2;
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
    pub addi_air_idx: usize,
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

/// Committed compact segment. The u128 allocation gives native u64 lane
/// stores sufficient alignment; all public offsets remain segment-relative.
pub struct RvrG2SegmentV1 {
    backing: Vec<u128>,
    byte_len: usize,
}

impl std::fmt::Debug for RvrG2SegmentV1 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RvrG2SegmentV1")
            .field("byte_len", &self.byte_len)
            .finish_non_exhaustive()
    }
}

impl RvrG2SegmentV1 {
    pub fn bytes(&self) -> &[u8] {
        // SAFETY: the backing is initialized to zero before native execution,
        // and `byte_len` never exceeds its allocation.
        unsafe { std::slice::from_raw_parts(self.backing.as_ptr().cast(), self.byte_len) }
    }

    pub fn header_acquire(&self) -> Result<G2SegmentHeaderV1, ExecutionError> {
        let bytes = self.bytes();
        if bytes.len() < G2_SEGMENT_HEADER_V1_SIZE {
            return Err(g2_error("segment is shorter than its v1 header"));
        }
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
    ) -> Result<[G2LaneDescV1; G2_PHASE1_LANE_COUNT], ExecutionError> {
        let bytes = self.bytes();
        let header = self.header_acquire()?;
        if header.magic != G2_SEGMENT_MAGIC_V1
            || header.version != G2_WIRE_VERSION_V1
            || header.header_bytes as usize
                != G2_SEGMENT_HEADER_V1_SIZE + G2_PHASE1_LANE_COUNT * G2_LANE_DESC_V1_SIZE
            || header.lane_count as usize != G2_PHASE1_LANE_COUNT
            || header.flags != G2_FLAGS_V1 | G2_FLAG_COMMITTED
            || &header.schema_fingerprint != expected_fingerprint
            || header.residual_event_count != 0
            || !bytes.len().is_multiple_of(G2_WIRE_ALIGNMENT)
        {
            return Err(g2_error("header capability or schema binding mismatch"));
        }

        let desc_ptr = unsafe { bytes.as_ptr().add(G2_SEGMENT_HEADER_V1_SIZE) };
        let descs = unsafe {
            // SAFETY: header_bytes establishes space for two aligned descs.
            [
                desc_ptr.cast::<G2LaneDescV1>().read(),
                desc_ptr
                    .add(G2_LANE_DESC_V1_SIZE)
                    .cast::<G2LaneDescV1>()
                    .read(),
            ]
        };
        validate_desc(
            &descs[0],
            G2_LANE_RUN_BLOCK_ID,
            4,
            header.run_count,
            bytes.len(),
        )?;
        validate_desc(&descs[1], G2_LANE_ADDI_V0, 8, u32::MAX, bytes.len())?;
        if descs[0].offset + u64::from(descs[0].payload_bytes) > descs[1].offset
            && descs[1].offset + u64::from(descs[1].payload_bytes) > descs[0].offset
        {
            return Err(g2_error("lane payloads overlap"));
        }
        Ok(descs)
    }

    pub fn lane_bytes<'a>(&'a self, desc: &G2LaneDescV1) -> &'a [u8] {
        &self.bytes()[desc.offset as usize..(desc.offset + u64::from(desc.payload_bytes)) as usize]
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
    let run_lane = segment.lane_bytes(&descs[0]);
    let addi_lane = segment.lane_bytes(&descs[1]);
    let header = segment.header_acquire()?;
    let mut registers = initial_registers;
    registers[0] = 0;
    let mut timestamps = [0u32; 32];
    let mut timestamp = initial_timestamp;
    let mut addi_cursor = 0usize;
    let mut compact_records =
        Vec::with_capacity(descs[1].count as usize * PREFLIGHT_ADDSUB_RECORD_SIZE);
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
            if entry.access_pattern != 8 || entry.air_idx as usize != meta.addi_air_idx {
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
        || addi_cursor != descs[1].count as usize
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
    pub producer: G2ProducerV1,
}

impl RvrG2PreparedV1 {
    pub fn new(addi_capacity: u32, run_capacity: u32) -> Result<Self, ExecutionError> {
        let addi_offset = G2_WIRE_ALIGNMENT;
        let addi_bytes = (addi_capacity as usize)
            .checked_mul(size_of::<u64>())
            .ok_or_else(|| g2_error("AddI lane capacity overflow"))?;
        let run_offset = align_up(addi_offset + addi_bytes, G2_WIRE_ALIGNMENT)?;
        let run_bytes = (run_capacity as usize)
            .checked_mul(size_of::<u32>())
            .ok_or_else(|| g2_error("run lane capacity overflow"))?;
        let byte_capacity = align_up(run_offset + run_bytes, G2_WIRE_ALIGNMENT)?;
        let words = byte_capacity.div_ceil(size_of::<u128>());
        let mut backing = vec![0u128; words];
        let producer = G2ProducerV1 {
            base: backing.as_mut_ptr().cast(),
            capacity: byte_capacity as u64,
            run_offset: run_offset as u64,
            addi_offset: addi_offset as u64,
            run_len: 0,
            run_cap: run_capacity,
            addi_len: 0,
            addi_cap: addi_capacity,
            instruction_count: 0,
            overflow: 0,
        };
        Ok(Self {
            backing,
            byte_capacity,
            producer,
        })
    }

    pub fn finalize(
        mut self,
        segment_id: u32,
        expected_instruction_count: u32,
        expected_addi_count: u32,
        fingerprint: [u8; 32],
    ) -> Result<RvrG2SegmentV1, ExecutionError> {
        let p = self.producer;
        if p.overflow != 0
            || p.base != self.backing.as_mut_ptr().cast()
            || p.capacity as usize != self.byte_capacity
            || p.run_len > p.run_cap
            || p.addi_len > p.addi_cap
            || p.addi_len != expected_addi_count
            || p.instruction_count != expected_instruction_count
            || (p.instruction_count != 0 && p.run_len == 0)
        {
            return Err(g2_error("native lane cursor/count validation failed"));
        }
        let run_bytes = p
            .run_len
            .checked_mul(4)
            .ok_or_else(|| g2_error("run payload byte count overflow"))?;
        let addi_bytes = p
            .addi_len
            .checked_mul(8)
            .ok_or_else(|| g2_error("AddI payload byte count overflow"))?;
        let byte_len = align_up(
            (p.run_offset as usize)
                .checked_add(run_bytes as usize)
                .ok_or_else(|| g2_error("committed segment length overflow"))?,
            G2_WIRE_ALIGNMENT,
        )?;
        if byte_len > self.byte_capacity {
            return Err(g2_error("committed segment exceeds its backing"));
        }

        let descs = [
            G2LaneDescV1 {
                kind: G2_LANE_RUN_BLOCK_ID,
                elem_width: 4,
                encoding: G2_ENCODING_FIXED_LE,
                flags: G2_LANE_FLAG_REQUIRED,
                count: p.run_len,
                payload_bytes: run_bytes,
                offset: p.run_offset,
                group_id: 0,
                reserved: 0,
            },
            G2LaneDescV1 {
                kind: G2_LANE_ADDI_V0,
                elem_width: 8,
                encoding: G2_ENCODING_FIXED_LE,
                flags: G2_LANE_FLAG_REQUIRED,
                count: p.addi_len,
                payload_bytes: addi_bytes,
                offset: p.addi_offset,
                group_id: 0,
                reserved: 0,
            },
        ];
        let header = G2SegmentHeaderV1 {
            magic: G2_SEGMENT_MAGIC_V1,
            version: G2_WIRE_VERSION_V1,
            header_bytes: (G2_SEGMENT_HEADER_V1_SIZE + G2_PHASE1_LANE_COUNT * G2_LANE_DESC_V1_SIZE)
                as u16,
            lane_count: G2_PHASE1_LANE_COUNT as u16,
            flags: G2_FLAGS_V1,
            segment_id,
            instruction_count: p.instruction_count,
            run_count: p.run_len,
            residual_event_count: 0,
            schema_fingerprint: fingerprint,
        };
        unsafe {
            // SAFETY: the u128 backing is aligned for both POD structs and the
            // fixed layout reserves exactly their bytes before lane offset 128.
            let base = self.backing.as_mut_ptr().cast::<u8>();
            base.cast::<G2SegmentHeaderV1>().write(header);
            let desc_base = base.add(G2_SEGMENT_HEADER_V1_SIZE);
            desc_base.cast::<G2LaneDescV1>().write(descs[0]);
            desc_base
                .add(G2_LANE_DESC_V1_SIZE)
                .cast::<G2LaneDescV1>()
                .write(descs[1]);
            (&*base.add(14).cast::<AtomicU16>())
                .store(G2_FLAGS_V1 | G2_FLAG_COMMITTED, Ordering::Release);
        }
        let segment = RvrG2SegmentV1 {
            backing: self.backing,
            byte_len,
        };
        segment.validate(&fingerprint)?;
        Ok(segment)
    }
}

fn validate_desc(
    desc: &G2LaneDescV1,
    kind: u16,
    elem_width: u8,
    expected_count: u32,
    segment_len: usize,
) -> Result<(), ExecutionError> {
    let expected_payload = desc
        .count
        .checked_mul(u32::from(elem_width))
        .ok_or_else(|| g2_error("lane payload byte count overflow"))?;
    if desc.kind != kind
        || desc.elem_width != elem_width
        || desc.encoding != G2_ENCODING_FIXED_LE
        || desc.flags != G2_LANE_FLAG_REQUIRED
        || (expected_count != u32::MAX && desc.count != expected_count)
        || desc.payload_bytes != expected_payload
        || desc.offset < G2_WIRE_ALIGNMENT as u64
        || !(desc.offset as usize).is_multiple_of(G2_WIRE_ALIGNMENT)
        || desc.group_id != 0
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
        let mut prepared = RvrG2PreparedV1::new(2, 1).unwrap();
        unsafe {
            prepared
                .producer
                .base
                .add(prepared.producer.addi_offset as usize)
                .cast::<u64>()
                .write(7);
            prepared
                .producer
                .base
                .add(prepared.producer.addi_offset as usize + 8)
                .cast::<u64>()
                .write(11);
            prepared
                .producer
                .base
                .add(prepared.producer.run_offset as usize)
                .cast::<u32>()
                .write(3);
        }
        prepared.producer.addi_len = 2;
        prepared.producer.run_len = 1;
        prepared.producer.instruction_count = 2;
        let fingerprint = [0x5a; 32];
        let segment = prepared.finalize(9, 2, 2, fingerprint).unwrap();
        let descs = segment.validate(&fingerprint).unwrap();
        assert_eq!(segment.header_acquire().unwrap().segment_id, 9);
        assert_eq!(segment.lane_bytes(&descs[0]), 3u32.to_le_bytes());
        assert_eq!(
            segment.lane_bytes(&descs[1]),
            [7u64, 11].map(u64::to_le_bytes).concat()
        );
    }

    #[test]
    fn phase1_transport_rejects_partial_or_bad_cursor() {
        let prepared = RvrG2PreparedV1::new(1, 1).unwrap();
        let partial = RvrG2SegmentV1 {
            backing: prepared.backing.clone(),
            byte_len: prepared.byte_capacity,
        };
        assert!(partial.header_acquire().is_err());
        assert!(prepared.finalize(0, 1, 1, [0; 32]).is_err());
    }

    #[test]
    fn phase1_transport_rejects_schema_mismatch() {
        let mut prepared = RvrG2PreparedV1::new(0, 0).unwrap();
        prepared.producer.instruction_count = 0;
        let segment = prepared.finalize(0, 0, 0, [1; 32]).unwrap();
        assert!(segment.validate(&[2; 32]).is_err());
    }
}
