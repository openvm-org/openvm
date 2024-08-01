mod air;
mod columns;
mod modular_multiplication_bigint;
mod modular_multiplication_primes;
mod trace;

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
}

pub struct FullLimbs<T> {
    pub a_limbs: Vec<Vec<T>>,
    pub b_limbs: Vec<Vec<T>>,
    pub r_limbs: Vec<Vec<T>>,
    pub q_limbs: Vec<T>,
}
