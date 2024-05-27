use crate::xor_lookup::XorLookupChip;
use afs_stark_backend::interaction::Interaction;
use columns::XorLimbsCols;
use p3_air::VirtualPairCol;
use p3_field::PrimeField64;
use parking_lot::Mutex;

pub mod air;
pub mod chip;
pub mod columns;
pub mod trace;

/// This chip gets requests to compute the xor of two numbers x and y of at most N bits.
/// It breaks down those numbers into limbs of at most M bits each, and computes the xor of
/// those limbs by communicating with the `XorLookupChip`.
#[derive(Default)]
pub struct XorLimbsChip<const N: usize, const M: usize> {
    bus_index: usize,

    pairs: Mutex<Vec<(u32, u32)>>,
    pub xor_lookup_chip: XorLookupChip<M>,
}

impl<const N: usize, const M: usize> XorLimbsChip<N, M> {
    pub fn new(bus_index: usize, pairs: Vec<(u32, u32)>) -> Self {
        Self {
            bus_index,
            pairs: Mutex::new(pairs),
            xor_lookup_chip: XorLookupChip::<M>::new(bus_index),
        }
    }

    pub fn bus_index(&self) -> usize {
        self.bus_index
    }

    fn calc_xor(&self, a: u32, b: u32) -> u32 {
        a ^ b
    }

    pub fn request(&self, a: u32, b: u32) -> u32 {
        let mut pairs_locked = self.pairs.lock();
        pairs_locked.push((a, b));
        self.calc_xor(a, b)
    }

    pub fn sends_custom<F: PrimeField64>(
        &self,
        cols: XorLimbsCols<N, M, usize>,
    ) -> Vec<Interaction<F>> {
        let num_limbs = (N + M - 1) / M;

        let mut interactions = vec![];

        for i in 0..num_limbs {
            interactions.push(Interaction {
                fields: vec![
                    VirtualPairCol::single_main(cols.x_limbs[i]),
                    VirtualPairCol::single_main(cols.y_limbs[i]),
                    VirtualPairCol::single_main(cols.z_limbs[i]),
                ],
                count: VirtualPairCol::constant(F::one()),
                argument_index: self.bus_index(),
            });
        }

        interactions
    }

    pub fn receives_custom<F: PrimeField64>(
        &self,
        cols: XorLimbsCols<N, M, usize>,
    ) -> Vec<Interaction<F>> {
        vec![Interaction {
            fields: vec![
                VirtualPairCol::single_main(cols.x),
                VirtualPairCol::single_main(cols.y),
                VirtualPairCol::single_main(cols.z),
            ],
            count: VirtualPairCol::constant(F::one()),
            argument_index: self.bus_index(),
        }]
    }
}
