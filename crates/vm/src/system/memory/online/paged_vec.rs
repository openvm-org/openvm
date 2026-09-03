use std::{fmt::Debug, mem};

use bytemuck::{zeroed_slice_box, Zeroable};
use openvm_stark_backend::p3_maybe_rayon::prelude::*;

#[derive(Debug, Clone)]
pub struct PagedVec<T, const PAGE_SIZE: usize> {
    pages: Vec<Option<Box<[T; PAGE_SIZE]>>>,
}

impl<T: Copy + Zeroable, const PAGE_SIZE: usize> PagedVec<T, PAGE_SIZE> {
    #[inline]
    /// `total_size` is the capacity of elements of type `T`.
    pub fn new(total_size: usize) -> Self {
        let num_pages = total_size.div_ceil(PAGE_SIZE);
        Self {
            pages: vec![None; num_pages],
        }
    }

    #[cold]
    #[inline(never)]
    fn create_zeroed_page() -> Box<[T; PAGE_SIZE]> {
        let page = zeroed_slice_box(PAGE_SIZE);
        match page.try_into() {
            Ok(page) => page,
            Err(_) => unreachable!("page was allocated with exactly PAGE_SIZE elements"),
        }
    }

    /// Get value at index without allocating new pages.
    /// Panics if index is out of bounds. Returns zero if the page does not exist.
    #[inline]
    pub fn get(&self, index: usize) -> T {
        let page_idx = index / PAGE_SIZE;
        let offset = index % PAGE_SIZE;

        // SAFETY:
        // - offset < PAGE_SIZE by construction (from modulo operation)
        // - page exists when as_ref() returns Some
        self.pages[page_idx]
            .as_ref()
            .map(|page| unsafe { *page.get_unchecked(offset) })
            .unwrap_or_else(T::zeroed)
    }

    /// Panics if the index is out of bounds. Creates new page before write when necessary.
    #[inline]
    pub fn set(&mut self, index: usize, value: T) {
        let page_idx = index / PAGE_SIZE;
        let offset = index % PAGE_SIZE;

        let page = self.pages[page_idx].get_or_insert_with(Self::create_zeroed_page);

        // SAFETY: offset < PAGE_SIZE by construction
        unsafe {
            *page.get_unchecked_mut(offset) = value;
        }
    }

    /// Replaces the value at `index`, returning its previous value.
    ///
    /// Panics if the index is out of bounds. Creates the page before writing
    /// when necessary.
    #[inline]
    pub(crate) fn replace(&mut self, index: usize, value: T) -> T {
        let page_idx = index / PAGE_SIZE;
        let offset = index % PAGE_SIZE;
        let page = self.pages[page_idx].get_or_insert_with(Self::create_zeroed_page);

        // SAFETY: offset < PAGE_SIZE by construction.
        unsafe { mem::replace(page.get_unchecked_mut(offset), value) }
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
