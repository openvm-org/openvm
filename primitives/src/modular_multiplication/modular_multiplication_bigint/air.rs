use num_bigint::BigUint;

pub struct ModularMultiplicationBigIntAir {
    pub modulus: BigUint,
    pub total_bits: usize,
    pub decomp: usize,
    pub range_bus: usize,

    pub max_limb_bits: usize,
    pub limbs_per_elem: usize,
    pub total_limbs: usize,
    pub modulus_limbs: Vec<usize>,
}

impl ModularMultiplicationBigIntAir {
    pub fn new(
        modulus: BigUint,
        total_bits: usize,
        decomp: usize,
        range_bus: usize,
        repr_bits: usize,
        max_limb_bits: usize,
    ) -> Self {
        assert_eq!(repr_bits % max_limb_bits, 0);
        let limbs_per_elem = repr_bits / max_limb_bits;
        let total_limbs = (total_bits + max_limb_bits - 1) / max_limb_bits;
        let modulus_limbs = (0..total_limbs)
            .map(|i| {
                let mut limb = 0;
                for j in 0..max_limb_bits {
                    limb += modulus.bit(((max_limb_bits * i) + j) as u64) as usize;
                }
                limb
            })
            .collect();
        Self {
            modulus,
            total_bits,
            decomp,
            range_bus,
            max_limb_bits,
            limbs_per_elem,
            total_limbs,
            modulus_limbs,
        }
    }

    fn limb_size(&self, index: usize) -> usize {
        assert!(index < self.total_limbs);
        if index == self.total_limbs - 1 && self.total_bits % self.max_limb_bits != 0 {
            self.total_bits % self.max_limb_bits
        } else {
            self.max_limb_bits
        }
    }
}
