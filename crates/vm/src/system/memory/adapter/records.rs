use std::io::Cursor;

use openvm_stark_backend::p3_field::PrimeField32;

use crate::arch::RecordArena;

struct AccessRecordMut<'a, F, const N: usize> {
    pub timestamp: &'a mut u32,
    pub address_space: &'a mut u32,
    pub pointer: &'a mut u32,
    pub data: &'a mut [F; N],
    /// If `Some`, contains references to previous values and timestamps.
    pub merge_before: Option<(&'a mut [u32; N], &'a mut [u32])>,
    pub split_after: &'a mut bool,
}

/// If `None`, we don't need to merge before.
/// If `Some(cell_size)`, this defines the length of the corresponding timestamps.
type AccessLayout = Option<usize>;

pub(crate) struct AccessAdapterRecordArena {
    pub cursor: Cursor<Vec<u8>>,
}

impl AccessAdapterRecordArena {
    pub fn with_capacity(size_bytes: usize) -> Self {
        Self {
            cursor: Cursor::new(vec![0; size_bytes]),
        }
    }

    pub fn alloc_one<'a, T>(&mut self) -> &'a mut T {
        let begin = self.cursor.position();
        let width = size_of::<T>();
        debug_assert!(begin as usize + width <= self.cursor.get_ref().len());
        // It is important here not to resize the vector, as the reference can be invalidated in that case.
        self.cursor.set_position(begin + width as u64);
        unsafe { &mut *(self.cursor.get_mut().as_mut_ptr().add(begin as usize) as *mut T) }
    }

    pub fn alloc_many<'a, T>(&mut self, count: usize) -> &'a mut [T] {
        let begin = self.cursor.position();
        let width = size_of::<T>() * count;
        debug_assert!(begin as usize + width <= self.cursor.get_ref().len());
        self.cursor.set_position(begin + width as u64);
        unsafe {
            std::slice::from_raw_parts_mut(
                self.cursor.get_mut().as_mut_ptr().add(begin as usize) as *mut T,
                count,
            )
        }
    }

    pub fn push<T>(&mut self, value: T) {
        let begin = self.cursor.position();
        let width = size_of::<T>();
        debug_assert!(begin as usize + width <= self.cursor.get_ref().len());
        unsafe {
            std::ptr::write(
                self.cursor.get_mut().as_mut_ptr().add(begin as usize) as *mut T,
                value,
            );
        }
        self.cursor.set_position(begin + width as u64);
    }

    pub fn extract_bytes(&self) -> Vec<u8> {
        self.cursor.get_ref()[..self.cursor.position() as usize].to_vec()
    }

    pub fn current_len(&self) -> usize {
        self.cursor.position() as usize
    }
}

impl<'a, F: PrimeField32, const N: usize> RecordArena<'a, AccessLayout, AccessRecordMut<'a, F, N>>
    for AccessAdapterRecordArena
{
    fn alloc(&'a mut self, layout: AccessLayout) -> AccessRecordMut<'a, F, N> {
        let timestamp = self.alloc_one();
        let address_space = self.alloc_one();
        let pointer = self.alloc_one();
        let data = self.alloc_one();
        let merge_before = if let Some(cell_size) = layout {
            self.push(1u8);
            let values = self.alloc_one();
            let timestamps = self.alloc_many(N / cell_size);
            Some((values, timestamps))
        } else {
            self.push(0u8);
            None
        };
        let split_after = self.alloc_one();
        AccessRecordMut {
            timestamp,
            address_space,
            pointer,
            data,
            merge_before,
            split_after,
        }
    }
}
