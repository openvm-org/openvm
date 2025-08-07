use std::fmt::Debug;

use openvm_stark_backend::p3_maybe_rayon::prelude::*;

use crate::utils::get_zeroed_array;

#[derive(Debug, Clone)]
pub struct PagedVec<T, const PAGE_SIZE: usize> {
    pages: Vec<Option<Box<[T; PAGE_SIZE]>>>,
}

unsafe impl<T: Send, const PAGE_SIZE: usize> Send for PagedVec<T, PAGE_SIZE> {}
unsafe impl<T: Sync, const PAGE_SIZE: usize> Sync for PagedVec<T, PAGE_SIZE> {}

impl<T: Copy + Default, const PAGE_SIZE: usize> PagedVec<T, PAGE_SIZE> {
    #[inline]
    /// `total_size` is the capacity of elements of type `T`.
    pub fn new(total_size: usize) -> Self {
        let num_pages = total_size.div_ceil(PAGE_SIZE);
        Self {
            pages: vec![None; num_pages],
        }
    }

    /// Panics if the index is out of bounds. Creates a new page with default values if no page
    /// exists.
    #[inline]
    pub fn get(&mut self, index: usize) -> &T {
        let page_idx = index / PAGE_SIZE;
        let offset = index % PAGE_SIZE;

        let page_slot = &mut self.pages[page_idx];
        if page_slot.is_none() {
            let new_page = get_zeroed_array::<_, PAGE_SIZE>();
            *page_slot = Some(Box::new(new_page));
        }

        unsafe {
            // SAFETY:
            // - We just ensured the page exists and has size `PAGE_SIZE`
            // - offset < PAGE_SIZE by construction
            self.pages
                .get_unchecked(page_idx)
                .as_ref()
                .unwrap()
                .get_unchecked(offset)
        }
    }

    /// Panics if the index is out of bounds. Creates new page before write when necessary.
    #[inline]
    pub fn set(&mut self, index: usize, value: T) {
        let page_idx = index / PAGE_SIZE;
        let offset = index % PAGE_SIZE;

        let page_slot = &mut self.pages[page_idx];
        if let Some(page) = page_slot {
            // SAFETY: If page exists, then it has size `PAGE_SIZE`
            unsafe {
                *page.get_unchecked_mut(offset) = value;
            }
        } else {
            let mut new_page = get_zeroed_array::<_, PAGE_SIZE>();
            new_page[offset] = value;
            *page_slot = Some(Box::new(new_page));
        }
    }

    pub fn par_iter(&self) -> impl ParallelIterator<Item = (usize, T)> + '_
    where
        T: Send + Sync,
    {
        self.pages
            .par_iter()
            .enumerate()
            .filter_map(move |(page_idx, page)| {
                page.as_ref().map(move |p| {
                    p.par_iter()
                        .enumerate()
                        .map(move |(offset, &value)| (page_idx * PAGE_SIZE + offset, value))
                })
            })
            .flatten()
    }

    pub fn iter(&self) -> impl Iterator<Item = (usize, T)> + '_
    where
        T: Send + Sync,
    {
        self.pages
            .iter()
            .enumerate()
            .filter_map(move |(page_idx, page)| {
                page.as_ref().map(move |p| {
                    p.iter()
                        .enumerate()
                        .map(move |(offset, &value)| (page_idx * PAGE_SIZE + offset, value))
                })
            })
            .flatten()
    }
}
