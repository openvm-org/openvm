use std::fmt::Debug;

use openvm_stark_backend::p3_maybe_rayon::prelude::*;

use crate::system::memory::online::{PAGE_SIZE, PAGE_SIZE_LOG2};

#[derive(Debug, Clone)]
pub struct PagedVec<T> {
    pages: Vec<Option<Box<[T]>>>,
}

unsafe impl<T: Send> Send for PagedVec<T> {}
unsafe impl<T: Sync> Sync for PagedVec<T> {}

impl<T: Copy + Default> PagedVec<T> {
    #[inline]
    /// `total_size` is the capacity of elements of type `T`.
    pub fn new(total_size: usize) -> Self {
        Self {
            pages: vec![Some(vec![T::default(); total_size].into_boxed_slice()); 1],
        }
    }

    /// Panics if the index is out of bounds. Creates a new page with default values if no page
    /// exists.
    #[inline]
    pub fn get(&mut self, index: usize) -> &T {
        // let page_idx = index >> PAGE_SIZE_LOG2;
        // let offset = index & (PAGE_SIZE - 1);

        // assert!(
        //     page_idx < self.pages.len(),
        //     "PagedVec::get index out of bounds: {} >= {}",
        //     index,
        //     self.pages.len() << PAGE_SIZE_LOG2
        // );

        unsafe {
            // SAFETY:
            // - We just ensured the page exists and has size `page_size`
            // - offset < page_size by construction
            self.pages
                .get_unchecked(0)
                .as_ref()
                .unwrap()
                .get_unchecked(index)
        }
    }

    /// Panics if the index is out of bounds. Creates new page before write when necessary.
    #[inline]
    pub fn set(&mut self, index: usize, value: T) {
        // let page_idx = index >> PAGE_SIZE_LOG2;
        // let offset = index & (PAGE_SIZE - 1);

        // assert!(
        //     page_idx < self.pages.len(),
        //     "PagedVec::set index out of bounds: {} >= {}",
        //     index,
        //     self.pages.len() << PAGE_SIZE_LOG2
        // );

        unsafe {
            let page = self.pages.get_unchecked_mut(0).as_mut().unwrap();
            *page.get_unchecked_mut(index) = value;
        }
    }

    pub fn par_iter(&self) -> impl ParallelIterator<Item = (usize, T)> + '_
    where
        T: Send + Sync,
    {
        self.pages[0].as_ref().unwrap().par_iter().enumerate().map(move |(offset, &value)| (offset, value))
    }

    pub fn iter(&self) -> impl Iterator<Item = (usize, T)> + '_
    where
        T: Send + Sync,
    {
        self.pages[0].as_ref().unwrap().iter().enumerate().map(move |(offset, &value)| (offset, value))
    }
}
