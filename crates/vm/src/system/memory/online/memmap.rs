use std::{fmt::Debug, marker::PhantomData, mem::MaybeUninit, ptr::copy_nonoverlapping};

use memmap2::MmapMut;

pub const CELL_STRIDE: usize = 1;
/// Default mmap page size. Change this if using THB.
const PAGE_SIZE: usize = 4096;

/// Mmap-backed linear memory.
#[derive(Debug)]
pub struct MmapMemory {
    mmap: MmapMut,
}

impl Clone for MmapMemory {
    fn clone(&self) -> Self {
        let mut new_mmap = MmapMut::map_anon(self.mmap.len()).unwrap();
        new_mmap.copy_from_slice(&self.mmap);
        Self { mmap: new_mmap }
    }
}

/// Iterator over MmapWrapper that yields elements of type T
pub struct MmapWrapperIter<'a, T: Copy> {
    wrapper: &'a MmapMemory,
    current_index: usize,
    phantom: PhantomData<T>,
}

impl<'a, T: Copy> MmapWrapperIter<'a, T> {
    fn new(wrapper: &'a MmapMemory) -> Self {
        Self {
            wrapper,
            current_index: 0,
            phantom: PhantomData,
        }
    }
}

impl<T: Copy> Iterator for MmapWrapperIter<'_, T> {
    type Item = (usize, T);

    fn next(&mut self) -> Option<Self::Item> {
        let size = std::mem::size_of::<T>();
        if self.current_index + size <= self.wrapper.len() {
            let value = self.wrapper.get::<T>(self.current_index);
            let index = self.current_index / size;
            self.current_index += size;
            Some((index, value))
        } else {
            None
        }
    }
}

impl MmapMemory {
    /// Create a new MmapMemory with the given `size` in bytes.
    /// We require `size` to be a multiple of the mmap page size (4kb by default) so that OS-level
    /// MMU protection corresponds to out of bounds protection.
    pub fn new(size: usize) -> Self {
        assert_eq!(
            size % PAGE_SIZE,
            0,
            "size {size} is not a multiple of page size {PAGE_SIZE}"
        );
        // anonymous mapping means pages are zero-initialized on first use
        Self {
            mmap: MmapMut::map_anon(size).unwrap(),
        }
    }

    pub fn len(&self) -> usize {
        self.mmap.len()
    }

    pub fn is_empty(&self) -> bool {
        self.mmap.is_empty()
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.mmap.as_ptr()
    }

    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.mmap.as_mut_ptr()
    }

    pub fn as_slice(&self) -> &[u8] {
        &self.mmap
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.mmap
    }

    /// Iterate over MmapWrapper as iterator of elements of type `T`.
    /// Iterator is over `(index, element)` where `index` is the byte index divided by
    /// `size_of::<T>()`.
    ///
    /// `T` must be stack allocated
    pub fn iter<T: Copy>(&self) -> MmapWrapperIter<'_, T> {
        MmapWrapperIter::new(self)
    }

    // Copies a range of length `len` starting at index `start`
    // into the memory pointed to by `dst`. If the relevant range is not
    // initialized, fills that range with `0u8`.
    /// # Safety
    /// - `dst` must be a valid pointer to a memory location
    /// - `start` and `start + len` must be within the bounds of the memory
    #[inline]
    pub unsafe fn read_range_generic(&self, start: usize, len: usize, dst: *mut u8) {
        // Calculate how much we can actually copy
        let copy_len = std::cmp::min(len, self.len() - start);

        // Copy the data
        copy_nonoverlapping(self.as_ptr().add(start), dst, copy_len);

        // If we couldn't copy everything, ensure the rest is zeroed
        if copy_len < len {
            std::slice::from_raw_parts_mut(dst.add(copy_len), len - copy_len).fill(0u8);
        }
    }

    /// # Panics
    /// If `from..from + size_of<BLOCK>()` is out of bounds.
    #[inline(always)]
    pub fn get<BLOCK: Copy>(&self, from: usize) -> BLOCK {
        // Create an uninitialized array of MaybeUninit<BLOCK>
        let mut result: MaybeUninit<BLOCK> = MaybeUninit::uninit();
        unsafe {
            self.read_range_generic(from, size_of::<BLOCK>(), result.as_mut_ptr() as *mut u8);
        }
        // SAFETY:
        // - All elements have been initialized (zero-initialized if didn't exist).
        // - `result` is aligned to `BLOCK`
        unsafe { result.assume_init() }
    }

    /// # Panics
    /// If `start..start + size_of<BLOCK>()` is out of bounds.
    // @dev: `values` is passed by reference since the data is copied into memory. Even though the
    // compiler probably optimizes it, we use reference to avoid any unnecessary copy of `values`
    // onto the stack in the function call.
    #[inline(always)]
    pub fn set<BLOCK: Copy>(&mut self, start: usize, values: &BLOCK) {
        let size = std::mem::size_of::<BLOCK>();

        unsafe {
            copy_nonoverlapping(
                values as *const _ as *const u8,
                self.as_mut_ptr().add(start),
                size,
            );
        }
    }

    /// memcpy of new `values` into from..from + size_of<BLOCK>(), memcpy of old existing values
    /// into new returned value.
    /// # Panics
    /// If `from..from + size_of<BLOCK>()` is out of bounds.
    #[inline(always)]
    pub fn replace<BLOCK: Copy>(&mut self, from: usize, values: &BLOCK) -> BLOCK {
        let size = std::mem::size_of::<BLOCK>();

        // Create an uninitialized array of MaybeUninit<BLOCK>
        let mut result: MaybeUninit<BLOCK> = MaybeUninit::uninit();
        unsafe {
            copy_nonoverlapping(
                self.as_ptr().add(from),
                result.as_mut_ptr() as *mut u8,
                size,
            );
            copy_nonoverlapping(
                values as *const _ as *const u8,
                self.as_mut_ptr().add(from),
                size,
            );
        }
        // SAFETY:
        // - All elements have been initialized (zero-initialized if didn't exist).
        // - `result` is aligned to `BLOCK`
        unsafe { result.assume_init() }
    }
}
