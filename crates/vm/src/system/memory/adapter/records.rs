use crate::arch::{CustomBorrow, DenseRecordArena, RecordArena, SizedRecord};

#[repr(C)]
#[derive(Debug)]
pub(crate) struct AccessRecordMut<'a> {
    /// If we need to merge before, this has the 31st bit set.
    /// If we need to split after, this has the 30th bit set.
    pub timestamp_and_mask: &'a mut u32,
    pub address_space: &'a mut u32,
    pub pointer: &'a mut u32,
    /// The block size is not defined by the access adapter's `N`,
    /// because the record denotes an operation of sort
    /// "split/merge every `block_size` chunk of this data"
    pub block_size: &'a mut u32,
    pub data: &'a mut [u8],
    pub prev_data: &'a mut [u8],
    pub timestamps: &'a mut [u32],
    // TODO(AG): optimize with some `Option` serialization stuff
}

#[derive(Debug, Clone)]
pub(crate) struct AccessLayout {
    /// The size of the block in elements.
    pub block_size: usize,
    /// The size of the minimal block (1 or 4 depending on the address space)
    pub cell_size: usize,
    /// The size of the type in bytes (1 for u8, 4 for F).
    // TODO(AG): is it true that cell_size * type_size = 4?
    pub type_size: usize,
}

pub(crate) const MERGE_BEFORE_FLAG: u32 = 1 << 30;
pub(crate) const SPLIT_AFTER_FLAG: u32 = 1 << 31;

pub(crate) fn size_by_layout(layout: &AccessLayout) -> usize {
    debug_assert_eq!(layout.cell_size * layout.type_size, 4);
    4 * size_of::<u32>() // all individual u32 fields
        + layout.block_size * 2 * layout.type_size // data and prev_data
        + (layout.block_size / layout.cell_size) * size_of::<u32>() // timestamps
}

impl SizedRecord<AccessLayout> for AccessRecordMut<'_> {
    fn size(layout: &AccessLayout) -> usize {
        size_by_layout(layout)
    }

    fn alignment(layout: &AccessLayout) -> usize {
        debug_assert_eq!(layout.cell_size * layout.type_size, 4);
        align_of::<u32>()
    }
}

impl<'a> CustomBorrow<'a, AccessRecordMut<'a>, AccessLayout> for [u8] {
    fn custom_borrow(&'a mut self, layout: AccessLayout) -> AccessRecordMut<'a> {
        let mut offset = 0;

        // timestamp_and_mask: u32 (4 bytes)
        let timestamp_and_mask = unsafe { &mut *(self.as_mut_ptr().add(offset) as *mut u32) };
        offset += 4;

        // address_space: u32 (4 bytes)
        let address_space = unsafe { &mut *(self.as_mut_ptr().add(offset) as *mut u32) };
        offset += 4;

        // pointer: u32 (4 bytes)
        let pointer = unsafe { &mut *(self.as_mut_ptr().add(offset) as *mut u32) };
        offset += 4;

        // block_size: u32 (4 bytes)
        let block_size_field = unsafe { &mut *(self.as_mut_ptr().add(offset) as *mut u32) };
        offset += 4;

        // data: [u8] (block_size * type_size bytes)
        let data = unsafe {
            std::slice::from_raw_parts_mut(
                self.as_mut_ptr().add(offset),
                layout.block_size * layout.type_size,
            )
        };
        offset += layout.block_size * layout.type_size;

        // prev_data: [u8] (block_size * type_size bytes)
        let prev_data = unsafe {
            std::slice::from_raw_parts_mut(
                self.as_mut_ptr().add(offset),
                layout.block_size * layout.type_size,
            )
        };
        offset += layout.block_size * layout.type_size;

        // timestamps: [u32] (block_size / cell_size * 4 bytes)
        let timestamps = unsafe {
            std::slice::from_raw_parts_mut(
                self.as_mut_ptr().add(offset) as *mut u32,
                layout.block_size / layout.cell_size,
            )
        };

        AccessRecordMut {
            timestamp_and_mask,
            address_space,
            pointer,
            block_size: block_size_field,
            data,
            prev_data,
            timestamps,
        }
    }

    unsafe fn extract_layout(&self) -> AccessLayout {
        let address_space = unsafe { std::ptr::read(self.as_ptr().add(4) as *const u32) };
        let block_size = unsafe { std::ptr::read(self.as_ptr().add(12) as *const u32) };
        let type_size = if address_space < 4 { 1 } else { 4 };
        AccessLayout {
            block_size: block_size as usize,
            cell_size: 4 / type_size,
            type_size,
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
