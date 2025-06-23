use super::LinearMemory;

#[derive(Clone, Debug)]
pub struct BasicMemory {
    pub data: Vec<u8>,
}

impl BasicMemory {
    #[inline(always)]
    pub fn as_ptr(&self) -> *const u8 {
        self.data.as_ptr()
    }

    #[inline(always)]
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.data.as_mut_ptr()
    }
}

impl LinearMemory for BasicMemory {
    fn new(size: usize) -> Self {
        Self {
            data: vec![0u8; size],
        }
    }

    fn size(&self) -> usize {
        self.data.len()
    }

    fn as_slice(&self) -> &[u8] {
        &self.data
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.data
    }

    #[inline(always)]
    unsafe fn read<BLOCK: Copy>(&self, from: usize) -> BLOCK {
        let size = size_of::<BLOCK>();
        assert!(from + size <= self.data.len(), "read out of bounds");

        let src = self.as_ptr().add(from) as *const BLOCK;
        // SAFETY:
        // - Bounds check is done via assert above
        // - We assume `src` is aligned to `BLOCK`
        // - We assume `BLOCK` is "plain old data" so the underlying `src` bytes is valid to read as
        //   an initialized value of `BLOCK`
        core::ptr::read(src)
    }

    #[inline(always)]
    unsafe fn read_unaligned<BLOCK: Copy>(&self, from: usize) -> BLOCK {
        let size = size_of::<BLOCK>();
        assert!(
            from + size <= self.data.len(),
            "read_unaligned out of bounds"
        );

        let src = self.as_ptr().add(from) as *const BLOCK;
        // SAFETY:
        // - Bounds check is done via assert above
        // - We assume `BLOCK` is "plain old data" so the underlying `src` bytes is valid to read as
        //   an initialized value of `BLOCK`
        core::ptr::read_unaligned(src)
    }

    #[inline(always)]
    unsafe fn write<BLOCK: Copy>(&mut self, start: usize, values: BLOCK) {
        let size = size_of::<BLOCK>();
        assert!(start + size <= self.data.len(), "write out of bounds");

        let dst = self.as_mut_ptr().add(start) as *mut BLOCK;
        // SAFETY:
        // - Bounds check is done via assert above
        // - We assume `dst` is aligned to `BLOCK`
        core::ptr::write(dst, values);
    }

    #[inline(always)]
    unsafe fn write_unaligned<BLOCK: Copy>(&mut self, start: usize, values: BLOCK) {
        let size = size_of::<BLOCK>();
        assert!(
            start + size <= self.data.len(),
            "write_unaligned out of bounds"
        );

        // Use Vec's copy_from_slice for safe byte-level copy
        let src_bytes = std::slice::from_raw_parts(&values as *const BLOCK as *const u8, size);
        self.data[start..start + size].copy_from_slice(src_bytes);
    }

    #[inline(always)]
    unsafe fn swap<BLOCK: Copy>(&mut self, start: usize, values: &mut BLOCK) {
        let size = size_of::<BLOCK>();
        assert!(start + size <= self.data.len(), "swap out of bounds");

        // SAFETY:
        // - Bounds check is done via assert above
        // - We assume `start` is aligned to `BLOCK`
        core::ptr::swap(
            self.as_mut_ptr().add(start) as *mut BLOCK,
            values as *mut BLOCK,
        );
    }

    #[inline(always)]
    unsafe fn copy_nonoverlapping<T: Copy>(&mut self, to: usize, data: &[T]) {
        let byte_len = size_of_val(data);
        assert!(
            to + byte_len <= self.data.len(),
            "copy_nonoverlapping out of bounds"
        );

        // Use Vec's copy_from_slice for safe byte-level copy
        let src_bytes = std::slice::from_raw_parts(data.as_ptr() as *const u8, byte_len);
        self.data[to..to + byte_len].copy_from_slice(src_bytes);
    }

    #[inline(always)]
    unsafe fn get_aligned_slice<T: Copy>(&self, start: usize, len: usize) -> &[T] {
        let byte_len = len * size_of::<T>();
        assert!(
            start + byte_len <= self.data.len(),
            "get_aligned_slice out of bounds"
        );
        debug_assert!(
            start % align_of::<T>() == 0,
            "get_aligned_slice: misaligned start"
        );

        let data = self.as_ptr().add(start) as *const T;
        // SAFETY:
        // - Bounds check is done via assert above
        // - Alignment check is done via assert above
        // - `T` is "plain old data" (POD), so conversion from underlying bytes is properly
        //   initialized
        // - `self` will not be mutated while borrowed
        core::slice::from_raw_parts(data, len)
    }

    unsafe fn iter<T: Copy>(&self) -> impl Iterator<Item = (usize, T)> {
        BasicMemoryIter::new(self)
    }
}

/// Iterator over BasicMemory that yields elements of type T
pub struct BasicMemoryIter<'a, T: Copy> {
    memory: &'a BasicMemory,
    current_index: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<'a, T: Copy> BasicMemoryIter<'a, T> {
    fn new(memory: &'a BasicMemory) -> Self {
        assert_eq!(memory.as_ptr() as usize % align_of::<T>(), 0);
        Self {
            memory,
            current_index: 0,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: Copy> Iterator for BasicMemoryIter<'_, T> {
    type Item = (usize, T);

    fn next(&mut self) -> Option<Self::Item> {
        let size = size_of::<T>();
        if self.current_index + size <= self.memory.size() {
            // Asserted to be aligned in constructor
            let value = unsafe { self.memory.read::<T>(self.current_index) };
            let index = self.current_index / size;
            self.current_index += size;
            Some((index, value))
        } else {
            None
        }
    }
}
