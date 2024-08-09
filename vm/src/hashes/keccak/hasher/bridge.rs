use std::iter::zip;

use afs_stark_backend::interaction::InteractionBuilder;
use itertools::{izip, Itertools};
use p3_field::AbstractField;
use p3_keccak_air::{NUM_ROUNDS, U64_LIMBS};

use super::{
    columns::KeccakVmCols, KeccakVmAir, KECCAK_RATE_BYTES, NUM_ABSORB_ROUNDS, NUM_U64_HASH_ELEMS,
};
use crate::cpu::{KECCAK256_BUS, MEMORY_BUS};

impl KeccakVmAir {
    /// Add new send interaction to lookup (x, y, x ^ y) where x, y, z
    /// will all be range checked to be 8-bits (assuming the bus is
    /// received by an 8-bit xor chip).
    fn send_xor<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        x: impl Into<AB::Expr>,
        y: impl Into<AB::Expr>,
        z: impl Into<AB::Expr>,
        count: impl Into<AB::Expr>,
    ) {
        builder.push_send(self.xor_bus_index, [x.into(), y.into(), z.into()], count);
    }
    /// Constrain state transition between keccak-f permutations is valid absorb of input bytes.
    /// The end-state in last round is given by `a_prime_prime_prime()` in `u16` limbs.
    /// The pre-state is given by `preimage` also in `u16` limbs.
    /// The input `block_bytes` will be given as **bytes**.
    ///
    /// We will XOR `block_bytes` with `a_prime_prime_prime()` and constrain to be `next.preimage`.
    /// This will be done using 8-bit XOR lookup in a separate AIR via interactions.
    /// This will require decomposing `u16` into bytes.
    /// Note that the XOR lookup automatically range checks its inputs to be bytes.
    ///
    /// We use the following trick to keep `u16` limbs and avoid changing
    /// the `keccak-f` AIR itself:
    /// if we already have a 16-bit limb `x` and we also provide a 8-bit limb
    /// `hi = x >> 8`, assuming `x` and `hi` have been range checked,
    /// we can use the expression `lo = x - hi * 256` for the low byte.
    /// If `lo` is range checked to `8`-bits, this constrains a valid byte
    ///  decomposition of `x` into `hi, lo`.
    /// This means in terms of trace cells, it is equivalent to provide
    /// `x, hi` versus `hi, lo`.
    pub fn constrain_absorb<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        local: &KeccakVmCols<AB::Var>,
        next: &KeccakVmCols<AB::Var>,
    ) {
        let updated_state_bytes = (0..NUM_ABSORB_ROUNDS).flat_map(|i| {
            let y = i / 5;
            let x = i % 5;
            (0..U64_LIMBS).flat_map(|limb| {
                let state_limb = local.postimage(y, x, limb);
                let hi = local.sponge.state_hi[i * U64_LIMBS + limb];
                let lo = state_limb - hi * AB::F::from_canonical_u64(1 << 8);
                [hi.into(), lo]
            })
        });
        // State should be initialized to 0s if next.is_new_start
        let pre_absorb_state_bytes = updated_state_bytes.map(|b| next.is_new_start() * b);

        let post_absorb_state_bytes = (0..NUM_ABSORB_ROUNDS).flat_map(|i| {
            let y = i / 5;
            let x = i % 5;
            (0..U64_LIMBS).flat_map(|limb| {
                let state_limb = next.inner.preimage[y][x][limb];
                let hi = next.sponge.state_hi[i * U64_LIMBS + limb];
                let lo = state_limb - hi * AB::F::from_canonical_u64(1 << 8);
                [hi.into(), lo]
            })
        });
        for (input, pre, post) in izip!(
            next.sponge.block_bytes,
            pre_absorb_state_bytes,
            post_absorb_state_bytes
        ) {
            // only absorb if first round
            // this should even work when `local` is the last row since
            // `next` becomes row 0 which `is_new_start`
            self.send_xor(builder, input, pre, post, next.inner.step_flags[0]);
        }
    }

    /*
    pub fn constrain_memory_accesses<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        local: &KeccakVmCols<AB::Var>,
    ) {
        let is_input = local.io.is_opcode * local.inner.step_flags[0];
        // receive the opcode itself
        builder.push_receive(
            KECCAK_PERMUTE_BUS,
            [
                local.io.clk.into(),
                local.io.a.into(),
                AB::Expr::zero(),
                local.io.c.into(),
                local.io.d.into(),
                local.io.e.into(),
            ],
            is_input.clone(),
        );

        let mut timestamp_offset = 0;

        // read addresses
        // dst = word[a]_d, src = word[c]_d
        for (ptr, value) in [local.io.a, local.io.c]
            .into_iter()
            .zip_eq([local.aux.dst, local.aux.src])
        {
            let timestamp = local.io.clk + AB::F::from_canonical_usize(timestamp_offset);
            timestamp_offset += 1;

            let fields = [
                timestamp,
                AB::Expr::from_bool(false), // read
                local.io.d.into(),
                ptr.into(),
                value.into(),
            ];
            builder.push_send(MEMORY_BUS, fields, is_input.clone());
        }

        // read `state` into `word[src + ...]_e`
        // iterator of state as u16:
        let input = local.inner.preimage.into_iter().flatten().flatten();
        for (i, input) in input.enumerate() {
            let timestamp = local.io.clk + AB::F::from_canonical_usize(timestamp_offset);
            timestamp_offset += 1;

            let address = local.aux.src + AB::F::from_canonical_usize(i);

            let fields = [
                timestamp,
                AB::Expr::from_bool(false), // read
                local.io.e.into(),
                address,
                input.into(),
            ];

            builder.push_send(MEMORY_BUS, fields, is_input.clone());
        }

        // write `new_state` into `word[dst + ...]_e`
        let is_output = local.io.is_opcode * local.inner.step_flags[NUM_ROUNDS - 1];
        // iterator of `new_state` as u16, in y-major order:
        let output = (0..5).flat_map(move |y| {
            (0..5).flat_map(move |x| {
                (0..NUM_U64_HASH_ELEMS).map(move |limb| {
                    // TODO: after switching to latest p3 commit, this should be y, x
                    // This is next.a[y][x][limb]
                    local.inner.a_prime_prime_prime(x, y, limb)
                })
            })
        });
        for (i, output) in output.enumerate() {
            let timestamp = local.io.clk + AB::F::from_canonical_usize(timestamp_offset);
            timestamp_offset += 1;

            let address = local.opcode.dst + AB::F::from_canonical_usize(i);

            let fields = [
                timestamp,
                AB::Expr::from_bool(true), // write
                local.io.e.into(),
                address,
                output.into(),
            ];

            builder.push_send(MEMORY_BUS, fields, is_output.clone());
        }
    }
    */
}
