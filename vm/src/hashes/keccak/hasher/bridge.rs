use std::iter::zip;

use afs_primitives::utils::not;
use afs_stark_backend::interaction::InteractionBuilder;
use itertools::izip;
use p3_air::AirBuilder;
use p3_field::AbstractField;
use p3_keccak_air::U64_LIMBS;

use super::{
    columns::KeccakVmColsRef, KeccakVmAir, KECCAK_DIGEST_BYTES, KECCAK_RATE_BYTES,
    KECCAK_RATE_U16S, KECCAK_WIDTH_U16S, NUM_ABSORB_ROUNDS,
};

/// We need three memory accesses to read dst, src, len from memory.
/// It seems harmless to just shift timestamp by this even in blocks
/// where we don't do this memory access.
/// See `eval_opcode_interactions`.
pub(super) const TIMESTAMP_OFFSET_FOR_OPCODE: usize = 3;
// This depends on WORD_SIZE
pub(super) const BLOCK_MEMORY_ACCESSES: usize = KECCAK_RATE_BYTES;

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
        local: KeccakVmColsRef<AB::Var>,
        next: KeccakVmColsRef<AB::Var>,
    ) {
        let updated_state_bytes = (0..NUM_ABSORB_ROUNDS).flat_map(|i| {
            let y = i / 5;
            let x = i % 5;
            (0..U64_LIMBS).flat_map(move |limb| {
                let state_limb = local.postimage(y, x, limb);
                let hi = local.sponge.state_hi[i * U64_LIMBS + limb];
                let lo = state_limb - hi * AB::F::from_canonical_u64(1 << 8);
                // Conversion from bytes to u64 is little-endian
                [lo, hi.into()]
            })
        });

        // TODO: for interaction chunking we want to keep interaction `fields`
        // degree 1 when possible. Currently this makes `fields` degree 2.
        // [jpw] I wanted to keep the property that input bytes are auto-range
        // checked via xor lookup
        let pre_absorb_state_bytes = updated_state_bytes.map(|b| not(next.is_new_start()) * b);

        let post_absorb_state_bytes = (0..NUM_ABSORB_ROUNDS).flat_map(|i| {
            let y = i / 5;
            let x = i % 5;
            (0..U64_LIMBS).flat_map(move |limb| {
                let state_limb = next.inner.preimage[y][x][limb];
                let hi = next.sponge.state_hi[i * U64_LIMBS + limb];
                let lo = state_limb - hi * AB::F::from_canonical_u64(1 << 8);
                [lo, hi.into()]
            })
        });

        // only absorb if next is first round and enabled (so don't constrain absorbs on non-enabled rows)
        let should_absorb = next.is_first_round() * next.opcode.is_enabled;
        for (input, pre, post) in izip!(
            next.sponge.block_bytes,
            pre_absorb_state_bytes,
            post_absorb_state_bytes
        ) {
            // this should even work when `local` is the last row since
            // `next` becomes row 0 which `is_new_start`
            self.send_xor(builder, input, pre, post, should_absorb.clone());
        }
        // constrain transition on the state outside rate
        let mut reset_builder = builder.when(local.is_new_start());
        for i in KECCAK_RATE_U16S..KECCAK_WIDTH_U16S {
            let y = i / U64_LIMBS / 5;
            let x = (i / U64_LIMBS) % 5;
            let limb = i % U64_LIMBS;
            reset_builder.assert_zero(local.inner.preimage[y][x][limb]);
        }
        let mut absorb_builder = builder.when(local.is_last_round() * not(next.is_new_start()));
        for i in KECCAK_RATE_U16S..KECCAK_WIDTH_U16S {
            let y = i / U64_LIMBS / 5;
            let x = (i / U64_LIMBS) % 5;
            let limb = i % U64_LIMBS;
            absorb_builder.assert_eq(local.postimage(y, x, limb), next.inner.preimage[y][x][limb]);
        }
    }

    /// Receive the opcode instruction itself on opcode bus.
    /// Then does memory read to get `dst` and `src` from memory.
    pub fn eval_opcode_interactions<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        local: KeccakVmColsRef<AB::Var>,
    ) {
        let opcode = &local.opcode;
        // Only receive opcode if:
        // - enabled row (not dummy row)
        // - first round of block
        // - is_new_start
        // Note this is degree 3, which results in quotient degree 2 if used
        // as `count` in interaction
        let should_receive = local.opcode.is_enabled * local.sponge.is_new_start;
        // receive the opcode itself
        builder.push_receive(
            KECCAK256_BUS,
            [
                opcode.start_timestamp,
                opcode.a,
                opcode.b,
                opcode.c,
                opcode.d,
                opcode.e,
            ],
            should_receive.clone(),
        );

        // Only when it is an input do we want to do memory read for
        // dst <- word[a]_d, src <- word[b]_d
        for (t_offset, ptr, addr_sp, value) in izip!(
            [0, 1, 2],
            [opcode.a, opcode.b, opcode.c],
            [opcode.d, opcode.d, opcode.d], // TODO use addr_sp = f for len
            [opcode.dst, opcode.src, opcode.len]
        ) {
            let timestamp = opcode.start_timestamp + AB::F::from_canonical_usize(t_offset);

            Self::constrain_memory_read(
                builder,
                timestamp,
                addr_sp,
                ptr,
                value,
                should_receive.clone(),
            );
        }
    }

    /// Constrain reading the input as `block_bytes` from memory.
    /// Reads input based on `is_padding_byte`.
    /// Constrains timestamp transitions between blocks if input crosses blocks.
    pub fn constrain_input_read<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        local: KeccakVmColsRef<AB::Var>,
    ) {
        // Only read input from memory when it is an opcode-related row
        // and only on the first round of block
        let is_input = local.opcode.is_enabled * local.inner.step_flags[0];
        // read `state` into `word[src + ...]_e`
        // iterator of state as u16:
        for (i, (input, is_padding)) in
            zip(local.sponge.block_bytes, local.sponge.is_padding_byte).enumerate()
        {
            // reserve two timestamp advances for opcode dst,src reads in
            // eval_opcode_interactions, even if they don't always happen
            let timestamp = local.opcode.start_timestamp
                + AB::F::from_canonical_usize(TIMESTAMP_OFFSET_FOR_OPCODE + i);
            let ptr = local.opcode.src + AB::F::from_canonical_usize(i);
            // Only read byte i if it is not padding byte
            // This is constraint degree 3, which leads to quotient degree 2
            // if used as `count` in interaction
            let count = is_input.clone() * not(is_padding);

            Self::constrain_memory_read(builder, timestamp, local.opcode.e, ptr, input, count);
        }
    }

    pub fn constrain_output_write<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        local: KeccakVmColsRef<AB::Var>,
    ) {
        let opcode = &local.opcode;

        let is_final_block = *local.sponge.is_padding_byte.last().unwrap();
        // since keccak-f AIR has this column, we might as well use it
        builder.assert_eq(
            local.inner.export,
            opcode.is_enabled * is_final_block * local.is_last_round(),
        );
        for x in 0..KECCAK_DIGEST_BYTES / 8 {
            for limb in 0..U64_LIMBS {
                let index = x * U64_LIMBS + limb;
                let timestamp = local.opcode.start_timestamp
                    + AB::F::from_canonical_usize(
                        TIMESTAMP_OFFSET_FOR_OPCODE + BLOCK_MEMORY_ACCESSES + index,
                    );
                let value = local.postimage(0, x, limb);
                Self::constrain_memory_write(
                    builder,
                    timestamp,
                    local.opcode.e,
                    local.opcode.dst + AB::F::from_canonical_usize(index),
                    value,
                    local.inner.export,
                );
            }
        }
    }

    // TODO: this should be general interface of Memory
    fn constrain_memory_read<AB: InteractionBuilder>(
        builder: &mut AB,
        timestamp: impl Into<AB::Expr>,
        address_space: impl Into<AB::Expr>,
        ptr: impl Into<AB::Expr>,
        value: impl Into<AB::Expr>,
        count: impl Into<AB::Expr>,
    ) {
        builder.push_send(
            MEMORY_BUS,
            [
                timestamp.into(),
                AB::Expr::from_bool(false), // read
                address_space.into(),
                ptr.into(),
                value.into(),
            ],
            count,
        );
    }

    // TODO: this should be general interface of Memory
    fn constrain_memory_write<AB: InteractionBuilder>(
        builder: &mut AB,
        timestamp: impl Into<AB::Expr>,
        address_space: impl Into<AB::Expr>,
        ptr: impl Into<AB::Expr>,
        value: impl Into<AB::Expr>,
        count: impl Into<AB::Expr>,
    ) {
        builder.push_send(
            MEMORY_BUS,
            [
                timestamp.into(),
                AB::Expr::from_bool(true), // write
                address_space.into(),
                ptr.into(),
                value.into(),
            ],
            count,
        );
    }
}
