use std::{borrow::Borrow, iter};

use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_stark_backend::{
    air_builders::sub::SubAirBuilder,
    interaction::{InteractionBuilder, PermutationCheckBus},
    p3_air::{Air, AirBuilder, BaseAir},
    p3_matrix::Matrix,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_keccak_air::{
    KeccakAir, KeccakCols as KeccakPermCols, NUM_KECCAK_COLS as NUM_KECCAK_PERM_COLS, U64_LIMBS,
};

use crate::KECCAK_WIDTH_U64S;

#[repr(C)]
#[derive(Debug, AlignedBorrow)]
pub struct KeccakfPeripheryCols<T> {
    pub inner: KeccakPermCols<T>,
}

/// A periphery AIR that wraps the Plonky3 AIR with a direct interaction on a [PermutationCheckBus].
#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct KeccakfPeripheryAir {
    /// Direct bus with keccakf pre- or post-state. Bus message is `prestate_u16_limbs ||
    /// poststate_u16_limbs`
    pub keccakf_state_bus: PermutationCheckBus,
}

impl<T: Copy> KeccakfPeripheryCols<T> {
    pub fn postimage(&self, y: usize, x: usize, limb: usize) -> T {
        self.inner.a_prime_prime_prime(y, x, limb)
    }

    pub fn is_first_round(&self) -> T {
        *self.inner.step_flags.first().unwrap()
    }

    pub fn is_last_round(&self) -> T {
        *self.inner.step_flags.last().unwrap()
    }
}

pub const NUM_KECCAKF_PERI_COLS: usize = size_of::<KeccakfPeripheryCols<u8>>();

impl<F> BaseAirWithPublicValues<F> for KeccakfPeripheryAir {}
impl<F> PartitionedBaseAir<F> for KeccakfPeripheryAir {}
impl<F> BaseAir<F> for KeccakfPeripheryAir {
    fn width(&self) -> usize {
        NUM_KECCAKF_PERI_COLS
    }
}

impl<AB: InteractionBuilder> Air<AB> for KeccakfPeripheryAir {
    fn eval(&self, builder: &mut AB) {
        self.eval_keccak_f(builder);

        let main = builder.main();
        let local = main.row_slice(0);
        let local: &KeccakfPeripheryCols<_> = (*local).borrow();

        // Only allowed to export on the last round of permutation
        builder
            .when(local.inner.export)
            .assert_one(local.is_last_round());

        // ==== Receive preimage and postimage on direct bus ====
        // reminder: keccakf-air constrains that `preimage` stays the same across steps within 24 rows of the same permutation <https://github.com/Plonky3/Plonky3/blob/539bbc84085efb609f4f62cb03cf49588388abdb/keccak-air/src/air.rs#L62>
        self.keccakf_state_bus.receive(
            builder,
            // state[x + 5 * y] is y-major as in https://keccak.team/keccak_specs_summary.html
            iter::empty()
                .chain((0..KECCAK_WIDTH_U64S).flat_map(|i| {
                    let y = i / 5;
                    let x = i % 5;
                    (0..U64_LIMBS).map(move |limb| local.inner.preimage[y][x][limb].into())
                }))
                .chain((0..KECCAK_WIDTH_U64S).flat_map(|i| {
                    let y = i / 5;
                    let x = i % 5;
                    (0..U64_LIMBS).map(move |limb| local.postimage(y, x, limb).into())
                })),
            local.inner.export,
        );
    }
}

impl KeccakfPeripheryAir {
    /// Evaluate the keccak-f permutation constraints.
    ///
    /// WARNING: The keccak-f AIR columns **must** be the first columns in the main AIR.
    #[inline]
    pub fn eval_keccak_f<AB: AirBuilder>(&self, builder: &mut AB) {
        let keccakf_air = KeccakAir {};
        let mut sub_builder =
            SubAirBuilder::<AB, KeccakAir, AB::Var>::new(builder, 0..NUM_KECCAK_PERM_COLS);
        keccakf_air.eval(&mut sub_builder);
    }
}
