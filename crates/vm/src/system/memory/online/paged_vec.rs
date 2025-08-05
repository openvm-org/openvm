use std::fmt::Debug;
use std::alloc::{alloc_zeroed, Layout};

use openvm_stark_backend::p3_maybe_rayon::prelude::*;

use crate::system::memory::online::{PAGE_SIZE, PAGE_SIZE_LOG2};

#[derive(Debug, Clone)]
pub struct PagedVec<T> {
    pages: Box<[T]>,
    flags: Box<[bool]>,
}

unsafe impl<T: Send> Send for PagedVec<T> {}
unsafe impl<T: Sync> Sync for PagedVec<T> {}

impl<T: Copy + Default> PagedVec<T> {
    #[inline]
    /// `total_size` is the capacity of elements of type `T`.
    pub fn new(total_size: usize) -> Self {
        let num_pages = total_size.div_ceil(PAGE_SIZE);
        
        // Allocate and zero-initialize pages using unsafe for performance
        let pages = unsafe {
            let layout = Layout::array::<T>(total_size).expect("Layout creation failed");
            let ptr = alloc_zeroed(layout) as *mut T;
            if ptr.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            let slice = std::slice::from_raw_parts_mut(ptr, total_size);
            Box::from_raw(slice)
        };
        
        // Allocate and zero-initialize flags (already false by default for bool)
        let flags = unsafe {
            let layout = Layout::array::<bool>(num_pages).expect("Layout creation failed");
            let ptr = alloc_zeroed(layout) as *mut bool;
            if ptr.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            let slice = std::slice::from_raw_parts_mut(ptr, num_pages);
            Box::from_raw(slice)
        };
        
        Self { pages, flags }
    }

    /// Panics if the index is out of bounds. Creates a new page with default values if no page
    /// exists.
    #[inline]
    pub fn get(&mut self, index: usize) -> &T {
        let page_idx = index >> PAGE_SIZE_LOG2;

        assert!(index < self.pages.len());

        unsafe {
            *self.flags.get_unchecked_mut(page_idx) = true;
            // SAFETY:
            // - We just ensured the page exists and has size `page_size`
            // - offset < page_size by construction
            self.pages.get_unchecked(index)
        }
    }

    /// Panics if the index is out of bounds. Creates new page before write when necessary.
    #[inline]
    pub fn set(&mut self, index: usize, value: T) {
        let page_idx = index >> PAGE_SIZE_LOG2;

        assert!(page_idx < self.pages.len());

        // SAFETY:
        // - If page exists, then it has size `page_size`
        unsafe {
            *self.flags.get_unchecked_mut(page_idx) = true;
            *self.pages.get_unchecked_mut(index) = value;
        }
    }

    pub fn par_iter(&self) -> impl ParallelIterator<Item = (usize, T)> + '_
    where
        T: Send + Sync,
    {
        self.pages
            .par_chunks_exact(PAGE_SIZE)
            .zip(self.flags.par_iter())
            .enumerate()
            .filter_map(move |(page_idx, (page, flag))| {
                if *flag {
                    Some(
                        page.par_iter()
                            .enumerate()
                            .map(move |(offset, &value)| (page_idx * PAGE_SIZE + offset, value)),
                    )
                } else {
                    None
                }
            })
            .flatten()
    }

    pub fn iter(&self) -> impl Iterator<Item = (usize, T)> + '_
    where
        T: Send + Sync,
    {
        self.pages
            .chunks_exact(PAGE_SIZE)
            .zip(self.flags.iter())
            .enumerate()
            .filter_map(move |(page_idx, (page, flag))| {
                if *flag {
                    Some(
                        page.iter()
                            .enumerate()
                            .map(move |(offset, &value)| (page_idx * PAGE_SIZE + offset, value)),
                    )
                } else {
                    None
                }
            })
            .flatten()
    }
}
