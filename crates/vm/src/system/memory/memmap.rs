use std::fmt::Debug;

use itertools::{zip_eq, Itertools};
use openvm_instructions::exe::SparseMemoryImage;
use openvm_stark_backend::p3_field::PrimeField32;
use serde::{Deserialize, Serialize};

use crate::arch::MemoryConfig;
use memmap2::MmapMut;

/// (address_space, pointer)
pub type Address = (u32, u32);

// impl<const PAGE_SIZE: usize> PagedVec<PAGE_SIZE> {
//     #[inline]
//     fn read_range_generic(&self, _start: usize, _len: usize, _dst: *mut u8) {
//         unimplemented!()
//     }

//     #[inline]
//     fn set_range_generic(&mut self, _start: usize, _len: usize, _new: *const u8, _dst: *mut u8) {
//         unimplemented!()
//     }

//     pub fn new(_num_pages: usize) -> Self {
//         unimplemented!()
//     }

//     pub fn bytes_capacity(&self) -> usize {
//         unimplemented!()
//     }

//     pub fn is_empty(&self) -> bool {
//         unimplemented!()
//     }

//     #[inline(always)]
//     pub fn get<BLOCK: Copy>(&self, _from: usize) -> BLOCK {
//         unimplemented!()
//     }

//     #[inline(always)]
//     pub fn set<BLOCK: Copy>(&mut self, _start: usize, _values: &BLOCK) {
//         unimplemented!()
//     }

//     #[inline(always)]
//     pub fn replace<BLOCK: Copy>(&mut self, _from: usize, _values: &BLOCK) -> BLOCK {
//         unimplemented!()
//     }

//     pub fn iter<T: Copy>(&self) -> PagedVecIter<'_, T, PAGE_SIZE> {
//         unimplemented!()
//     }
// }

// pub struct PagedVecIter<'a, T, const PAGE_SIZE: usize> {
//     vec: &'a PagedVec<PAGE_SIZE>,
//     current_page: usize,
//     current_index_in_page: usize,
//     phantom: PhantomData<T>,
// }

// impl<T: Copy, const PAGE_SIZE: usize> Iterator for PagedVecIter<'_, T, PAGE_SIZE> {
//     type Item = (usize, T);

//     fn next(&mut self) -> Option<Self::Item> {
//         unimplemented!()
//     }
// }

#[derive(Debug)]
pub struct AddressMap {
    pub mem: Vec<MmapMut>,
    pub cell_size: Vec<usize>,
    pub as_offset: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AddressMapSerializeHelper {
    compressed: Vec<Vec<(usize, Vec<u8>)>>,
    lengths: Vec<usize>,
    cell_size: Vec<usize>,
    as_offset: u32,
}

impl AddressMapSerializeHelper {
    fn from(map: &AddressMap) -> Self {
        assert!(map.mem.len() % PAGE_SIZE == 0);
        let compressed = map
            .mem
            .iter()
            .map(|space_mem| {
                space_mem
                    .iter()
                    .chunks(PAGE_SIZE)
                    .into_iter()
                    .enumerate()
                    .map(|(i, chunk)| {
                        let cv = chunk.cloned().collect::<Vec<_>>();
                        if cv.iter().all(|x| x == &0) {
                            None
                        } else {
                            Some((i, cv))
                        }
                    })
                    .flatten()
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let lengths = map.mem.iter().map(|m| m.len()).collect::<Vec<_>>();
        Self {
            compressed,
            lengths,
            cell_size: map.cell_size.clone(),
            as_offset: map.as_offset,
        }
    }

    fn to(&self) -> AddressMap {
        let mut mem = self
            .lengths
            .iter()
            .map(|l| MmapMut::map_anon(*l).unwrap())
            .collect::<Vec<_>>();
        for (space_compr, space_mem) in self.compressed.iter().zip(mem.iter_mut()) {
            for (i, chunk) in space_compr.iter() {
                space_mem[i * PAGE_SIZE..(i + 1) * PAGE_SIZE].copy_from_slice(&chunk);
            }
        }
        AddressMap {
            mem,
            cell_size: self.cell_size.clone(),
            as_offset: self.as_offset,
        }
    }
}

const PAGE_SIZE: usize = 4096;

impl Serialize for AddressMap {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        AddressMapSerializeHelper::from(self).serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for AddressMap {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let helper = AddressMapSerializeHelper::deserialize(deserializer)?;
        Ok(helper.to())
    }
}

impl Default for AddressMap {
    fn default() -> Self {
        unimplemented!()
    }
}

impl AddressMap {
    pub fn new(as_offset: u32, as_cnt: usize, mem_size: usize) -> Self {
        // TMP: hardcoding for now
        let mut cell_size = vec![1, 1, 1];
        cell_size.resize(as_cnt, 4);
        let mem = cell_size
            .iter()
            .map(|&cell_size| MmapMut::map_anon(mem_size.checked_mul(cell_size).unwrap()).unwrap())
            .collect();
        Self {
            mem,
            cell_size,
            as_offset,
        }
    }
    pub fn from_mem_config(mem_config: &MemoryConfig) -> Self {
        Self::new(
            mem_config.as_offset,
            1 << mem_config.as_height,
            1 << mem_config.pointer_max_bits,
        )
    }
    pub fn items<F: PrimeField32>(&self) -> impl Iterator<Item = (Address, F)> + '_ {
        zip_eq(&self.mem, &self.cell_size).enumerate().flat_map(
            move |(as_idx, (space_mem, &cell_size))| {
                // TODO: better way to handle address space conversions to F
                if cell_size == 1 {
                    space_mem
                        .iter()
                        .enumerate()
                        .map(move |(ptr_idx, x)| {
                            (
                                (as_idx as u32 + self.as_offset, ptr_idx as u32),
                                F::from_canonical_u8(*x),
                            )
                        })
                        .collect_vec()
                } else {
                    // TEMP
                    assert_eq!(cell_size, 4);
                    space_mem
                        .iter()
                        .chunks(4)
                        .into_iter()
                        .enumerate()
                        .map(move |(ptr_idx, chunk)| {
                            ((as_idx as u32 + self.as_offset, ptr_idx as u32), unsafe {
                                std::mem::transmute_copy(&chunk)
                            })
                        })
                        .collect_vec()
                }
            },
        )
    }

    pub fn get_f<F: PrimeField32>(&self, addr_space: u32, ptr: u32) -> F {
        debug_assert_ne!(addr_space, 0);
        // TODO: fix this
        unsafe {
            if addr_space <= 3 {
                F::from_canonical_u8(self.get::<u8>((addr_space, ptr)))
            } else {
                self.get::<F>((addr_space, ptr))
            }
        }
    }

    /// # Safety
    /// - `T` **must** be the correct type for a single memory cell for `addr_space`
    /// - Assumes `addr_space` is within the configured memory and not out of bounds
    pub unsafe fn get<T: Copy>(&self, (addr_space, ptr): Address) -> T {
        debug_assert_eq!(
            size_of::<T>(),
            self.cell_size[(addr_space - self.as_offset) as usize]
        );
        {
            let bytes = self
                .mem
                .get_unchecked((addr_space - self.as_offset) as usize)
                .get_unchecked(
                    (ptr as usize) * size_of::<T>()..(ptr as usize + 1) * size_of::<T>(),
                );
            let mut value = std::mem::MaybeUninit::<T>::uninit();
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                value.as_mut_ptr() as *mut u8,
                size_of::<T>(),
            );
            value.assume_init()
        }
    }

    /// # Safety
    /// - `T` **must** be the correct type for a single memory cell for `addr_space`
    /// - Assumes `addr_space` is within the configured memory and not out of bounds
    pub unsafe fn insert<T: Copy>(&mut self, (addr_space, ptr): Address, data: T) -> T {
        debug_assert_eq!(
            size_of::<T>(),
            self.cell_size[(addr_space - self.as_offset) as usize]
        );
        let mut data = data;
        {
            let bytes = self
                .mem
                .get_unchecked_mut((addr_space - self.as_offset) as usize)
                .get_unchecked_mut(
                    (ptr as usize) * size_of::<T>()..(ptr as usize + 1) * size_of::<T>(),
                );
            // Swap contents of data and bytes using std::mem::swap for efficiency
            std::ptr::swap_nonoverlapping(
                &mut *(bytes.as_mut_ptr() as *mut T),
                &mut *(&mut data as *mut T),
                1,
            );
            data
        }
    }
    pub fn is_empty(&self) -> bool {
        unimplemented!()
    }

    // TODO[jpw]: stabilize the boundary memory image format and how to construct
    /// # Safety
    /// - `T` **must** be the correct type for a single memory cell for `addr_space`
    /// - Assumes `addr_space` is within the configured memory and not out of bounds
    pub fn from_sparse(
        as_offset: u32,
        as_cnt: usize,
        mem_size: usize,
        sparse_map: SparseMemoryImage,
    ) -> Self {
        let mut vec = Self::new(as_offset, as_cnt, mem_size);
        for ((addr_space, index), data_byte) in sparse_map.into_iter() {
            vec.mem[(addr_space - as_offset) as usize][index as usize] = data_byte;
        }
        vec
    }
}
