use std::cmp::min;

pub mod air;
/// Approach representing big field elements as big integers with limbs
pub mod bigint;
/// Casting out Primes approach following <https://eprint.iacr.org/2022/1470>
mod cast_primes;
pub mod columns;
pub mod trace;

pub struct LimbDimensions {
    pub io_limb_sizes: Vec<Vec<usize>>,
    pub q_limb_sizes: Vec<usize>,
    pub num_materialized_io_limbs: usize,
}

impl LimbDimensions {
    fn new(io_limb_sizes: Vec<Vec<usize>>, q_limb_sizes: Vec<usize>) -> Self {
        let num_materialized_io_limbs = io_limb_sizes.iter().map(|limbs| limbs.len() - 1).sum();
        Self {
            io_limb_sizes,
            q_limb_sizes,
            num_materialized_io_limbs,
        }
    }

    #[allow(dead_code)]
    fn new_same_sizes(limb_sizes: Vec<usize>, limbs_per_elem: usize) -> Self {
        let mut io_limb_sizes = vec![];
        for i in (0..limb_sizes.len()).step_by(limbs_per_elem) {
            io_limb_sizes.push(limb_sizes[i..min(i + limbs_per_elem, limb_sizes.len())].to_vec());
        }
        Self::new(io_limb_sizes, limb_sizes)
    }
}

/// Essentially `ModularMultiplicationAuxCols` but with the first limbs
pub struct FullLimbs<T> {
    pub a_limbs: Vec<Vec<T>>,
    pub b_limbs: Vec<Vec<T>>,
    pub r_limbs: Vec<Vec<T>>,
    pub q_limbs: Vec<T>,
}
