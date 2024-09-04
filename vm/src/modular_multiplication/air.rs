use afs_primitives::modular_multiplication::bigint::air::ModularArithmeticBigIntAir;

#[derive(Clone, Debug)]
pub struct ModularArithmeticVmAir {
    pub air: ModularArithmeticBigIntAir,
}

impl ModularArithmeticVmAir {
    pub fn time_stamp_delta(&self) -> usize {
        let num_elems = self.air.limb_dimensions.io_limb_sizes.len();
        3 * (num_elems + 1)
    }
}
