use std::{
    borrow::{Borrow, BorrowMut},
    mem::{align_of, size_of},
};

use openvm_circuit_primitives::AlignedBytesBorrow;

use crate::arch::{CustomBorrow, DenseRecordArena, RecordArena, SizedRecord};

#[repr(C)]
#[derive(Debug, Clone, Copy, AlignedBytesBorrow, PartialEq, Eq, PartialOrd, Ord)]
pub struct AccessRecordHeader {
    /// Iff we need to merge before, this has the `MERGE_AND_NOT_SPLIT_FLAG` bit set
    pub timestamp_and_mask: u32,
    pub address_space: u32,
    pub pointer: u32,
    // PERF: these three are easily mergeable into a single u32
    pub block_size: u32,
    pub lowest_block_size: u32,
    pub type_size: u32,
}

#[repr(C)]
#[derive(Debug)]
pub struct AccessRecordMut<'a> {
    pub header: &'a mut AccessRecordHeader,
    // PERF(AG): optimize with some `Option` serialization stuff
    pub timestamps: &'a mut [u32], // len is block_size / lowest_block_size
    pub data: &'a mut [u8],        // len is block_size * type_size
}

#[derive(Debug, Clone)]
pub struct AccessLayout {
    /// The size of the block in elements.
    pub block_size: usize,
    /// The size of the minimal block we may split into/merge from (usually 1 or 4)
    pub lowest_block_size: usize,
    /// The size of the type in bytes (1 for u8, 4 for F).
    pub type_size: usize,
}

impl AccessLayout {
    pub(crate) fn from_record_header(header: &AccessRecordHeader) -> Self {
        Self {
            block_size: header.block_size as usize,
            lowest_block_size: header.lowest_block_size as usize,
            type_size: header.type_size as usize,
        }
    }
}

pub(crate) const MERGE_AND_NOT_SPLIT_FLAG: u32 = 1 << 31;

pub(crate) fn size_by_layout(layout: &AccessLayout) -> usize {
    size_of::<AccessRecordHeader>() // header struct
    + (layout.block_size / layout.lowest_block_size) * size_of::<u32>() // timestamps
    + (layout.block_size * layout.type_size).next_multiple_of(4) // data
}

impl SizedRecord<AccessLayout> for AccessRecordMut<'_> {
    fn size(layout: &AccessLayout) -> usize {
        size_by_layout(layout)
    }

    fn alignment(_: &AccessLayout) -> usize {
        align_of::<AccessRecordHeader>()
    }
}

impl<'a> CustomBorrow<'a, AccessRecordMut<'a>, AccessLayout> for [u8] {
    fn custom_borrow(&'a mut self, layout: AccessLayout) -> AccessRecordMut<'a> {
        // header: AccessRecordHeader
        // SAFETY: self.len() >= size_of::<AccessRecordHeader>() from size_by_layout()
        let (header_buf, rest) =
            unsafe { self.split_at_mut_unchecked(size_of::<AccessRecordHeader>()) };
        let header = header_buf.borrow_mut();

        let mut offset = 0;

        // timestamps: [u32] (block_size / cell_size * 4 bytes)
        // SAFETY:
        // - size: (layout.block_size / layout.lowest_block_size) * size_of::<u32>() from
        //   size_by_layout()
        // - alignment: u32 aligned due to AccessRecordHeader alignment
        let timestamps = unsafe {
            std::slice::from_raw_parts_mut(
                rest.as_mut_ptr().add(offset) as *mut u32,
                layout.block_size / layout.lowest_block_size,
            )
        };
        offset += layout.block_size / layout.lowest_block_size * size_of::<u32>();

        // data: [u8] (block_size * type_size bytes)
        // SAFETY:
        // - size: layout.block_size * layout.type_size from size_by_layout()
        // - offset points past timestamps section
        let data = unsafe {
            std::slice::from_raw_parts_mut(
                rest.as_mut_ptr().add(offset),
                layout.block_size * layout.type_size,
            )
        };

        AccessRecordMut {
            header,
            data,
            timestamps,
        }
    }

    unsafe fn extract_layout(&self) -> AccessLayout {
        let header: &AccessRecordHeader = self.borrow();
        AccessLayout {
            block_size: header.block_size as usize,
            lowest_block_size: header.lowest_block_size as usize,
            type_size: header.type_size as usize,
        }
    }
}

impl<'a> RecordArena<'a, AccessLayout, AccessRecordMut<'a>> for DenseRecordArena {
    fn alloc(&'a mut self, layout: AccessLayout) -> AccessRecordMut<'a> {
        let bytes = self.alloc_bytes(<AccessRecordMut<'a> as SizedRecord<AccessLayout>>::size(
            &layout,
        ));
        <[u8] as CustomBorrow<AccessRecordMut<'a>, AccessLayout>>::custom_borrow(bytes, layout)
    }
}

/// `trace_heights[i]` is assumed to correspond to `Adapter< 2^(i+1) >`.
pub fn arena_size_bound(trace_heights: &[u32]) -> usize {
    // At the very worst, each row in `Adapter<N>`
    // corresponds to a unique record of `block_size` being `2 * N`,
    // and its `lowest_block_size` is at least 1 and `type_size` is at most 4.
    let size_bound = trace_heights
        .iter()
        .enumerate()
        .map(|(i, &h)| {
            size_by_layout(&AccessLayout {
                block_size: 1 << (i + 1),
                lowest_block_size: 1,
                type_size: 4,
            }) * h as usize
        })
        .sum::<usize>();
    tracing::debug!(
        "Allocating {} bytes for memory adapters arena from heights {:?}",
        size_bound,
        trace_heights
    );
    size_bound
}
