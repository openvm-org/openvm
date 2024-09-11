use afs_primitives::utils::not;
use afs_stark_backend::interaction::InteractionBuilder;
use itertools::izip;
use p3_air::AirBuilder;
use p3_field::AbstractField;
use p3_keccak_air::U64_LIMBS;

use super::{
    columns::KeccakVmColsRef, KeccakVmAir, KECCAK_ABSORB_READS, KECCAK_DIGEST_WRITES,
    KECCAK_EXECUTION_READS, KECCAK_RATE_U16S, KECCAK_WIDTH_U16S, NUM_ABSORB_ROUNDS,
};
use crate::{
    arch::{
        columns::{ExecutionState, InstructionCols},
        instructions::Opcode,
    },
    memory::{
        offline_checker::{MemoryBridge, MemoryReadAuxCols, MemoryWriteAuxCols},
        MemoryAddress,
    },
};

impl KeccakVmAir {
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
            // Add new send interaction to lookup (x, y, x ^ y) where x, y, z
            // will all be range checked to be 8-bits (assuming the bus is
            // received by an 8-bit xor chip).

            // this should even work when `local` is the last row since
            // `next` becomes row 0 which `is_new_start`
            self.xor_bus
                .send(input, pre, post)
                .eval(builder, should_absorb.clone());
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
    /// Then does memory read to get `dst, src, len` from memory.
    ///
    /// Returns `start_read_timestamp` which is only relevant when `local.opcode.is_enabled`.
    /// Note that `start_read_timestamp` is a linear expression.
    pub fn eval_opcode_interactions<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        local: KeccakVmColsRef<AB::Var>,
        mem_aux: [MemoryReadAuxCols<1, AB::Var>; KECCAK_EXECUTION_READS],
    ) -> AB::Expr {
        let opcode = local.opcode;
        // Only receive opcode if:
        // - enabled row (not dummy row)
        // - first round of block
        // - is_new_start
        // Note this is degree 3, which results in quotient degree 2 if used
        // as `count` in interaction
        let should_receive = local.opcode.is_enabled * local.sponge.is_new_start;

        let timestamp_change: AB::Expr = Self::timestamp_change(opcode.len);
        self.execution_bus.execute_increment_pc(
            builder,
            should_receive.clone(),
            ExecutionState::new(opcode.pc, opcode.start_timestamp),
            timestamp_change,
            InstructionCols::new(
                AB::Expr::from_canonical_usize(Opcode::KECCAK256 as usize),
                [opcode.a, opcode.b, opcode.c, opcode.d, opcode.e, opcode.f],
            ),
        );

        let mut timestamp: AB::Expr = opcode.start_timestamp.into();
        let memory_bridge = MemoryBridge::new(self.mem_oc);
        // Only when it is an input do we want to do memory read for
        // dst <- word[a]_d, src <- word[b]_d
        for (ptr, addr_sp, value, mem_aux) in izip!(
            [opcode.a, opcode.b, opcode.c],
            [opcode.d, opcode.d, opcode.f],
            [opcode.dst, opcode.src, opcode.len],
            mem_aux,
        ) {
            memory_bridge
                .read(
                    MemoryAddress::new(addr_sp, ptr),
                    [value],
                    timestamp.clone(),
                    mem_aux,
                )
                .eval(builder, should_receive.clone());

            timestamp += AB::Expr::one();
        }
        timestamp
    }

    /// Constrain reading the input as `block_bytes` from memory.
    /// Reads input based on `is_padding_byte`.
    /// Constrains timestamp transitions between blocks if input crosses blocks.
    ///
    /// Expects `start_read_timestamp` to be a linear expression.
    /// Returns the `start_write_timestamp` which is the timestamp to start from
    /// for writing digest to memory.
    pub fn constrain_input_read<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        local: KeccakVmColsRef<AB::Var>,
        start_read_timestamp: AB::Expr,
        mem_aux: [MemoryReadAuxCols<1, AB::Var>; KECCAK_ABSORB_READS],
    ) -> AB::Expr {
        let memory_bridge = MemoryBridge::new(self.mem_oc);
        // Only read input from memory when it is an opcode-related row
        // and only on the first round of block
        let is_input = local.opcode.is_enabled * local.inner.step_flags[0];

        let mut timestamp = start_read_timestamp;
        // read `state` into `word[src + ...]_e`
        // iterator of state as u16:
        for (i, (input, is_padding, mem_aux)) in izip!(
            local.sponge.block_bytes,
            local.sponge.is_padding_byte,
            mem_aux
        )
        .enumerate()
        {
            let ptr = local.opcode.src + AB::F::from_canonical_usize(i);
            // Only read byte i if it is not padding byte
            // This is constraint degree 3, which leads to quotient degree 2
            // if used as `count` in interaction
            let count = is_input.clone() * not(is_padding);

            // reminder: input is currently range checked to be 8-bits in `constrain_absorb` by the XOR lookup
            memory_bridge
                .read(
                    MemoryAddress::new(local.opcode.e, ptr),
                    [input],
                    timestamp.clone(),
                    mem_aux,
                )
                .eval(builder, count);

            timestamp += AB::Expr::one();
        }
        timestamp
    }

    pub fn constrain_output_write<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        local: KeccakVmColsRef<AB::Var>,
        start_write_timestamp: AB::Expr,
        mem_aux: [MemoryWriteAuxCols<1, AB::Var>; KECCAK_DIGEST_WRITES],
    ) {
        let opcode = local.opcode;
        let memory_bridge = MemoryBridge::new(self.mem_oc);

        let is_final_block = *local.sponge.is_padding_byte.last().unwrap();
        // since keccak-f AIR has this column, we might as well use it
        builder.assert_eq(
            local.inner.export,
            opcode.is_enabled * is_final_block * local.is_last_round(),
        );
        // See `constrain_absorb` on how we derive the postimage bytes from u16 limbs
        // **SAFETY:** because we never XOR the final state, these bytes are NOT range checked.
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
        for (i, digest_byte) in updated_state_bytes.take(KECCAK_DIGEST_WRITES).enumerate() {
            let timestamp = start_write_timestamp.clone() + AB::Expr::from_canonical_usize(i);
            memory_bridge
                .write(
                    MemoryAddress::new(opcode.e, opcode.dst + AB::F::from_canonical_usize(i)),
                    [digest_byte],
                    timestamp,
                    mem_aux[i].clone(),
                )
                .eval(builder, local.inner.export)
        }
    }

    /// Amount to advance timestamp by after execution of one opcode instruction.
    /// This is an upper bound dependant on the length `len` operand, which is unbounded.
    pub fn timestamp_change<T: AbstractField>(len: impl Into<T>) -> T {
        // actual number is ceil(len / 136) * (3 + 136) + KECCAK_DIGEST_WRITES
        // digest writes only done on last row of multi-block
        // add another KECCAK_ABSORB_READS to round up so we don't deal with padding
        len.into() * T::two()
            + T::from_canonical_usize(
                KECCAK_EXECUTION_READS + KECCAK_ABSORB_READS + KECCAK_DIGEST_WRITES,
            )
    }
}
