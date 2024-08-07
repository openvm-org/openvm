# Spec

## Review of `keccak-f` AIR

The `keccak-air` from Plonky3 is an AIR that does one `keccak-f[1600]` permutation every `NUM_ROUNDS = 24` rows (henceforth we call this the `keccak-f` AIR to avoid confusion). All rows in the round have the same `preimage`, which is the starting state prior to the permutation, represented as `5 * 5 * 4` `u16` limbs (the state in the spec is `5 * 5` `u64`s, but since the AIR uses a 31-bit field, the `u64` is broken into `u16`s).

The `keccak-f` permutation copies `preimage` to `A` and mutates `A` over rounds. The mutations are materialized in the `keccak-f` AIR in `A'` and `A''` arrays. While the bits of `A'` are materialized, the bits of `preimage` and `A` are never materialized (there is an implicit bit compostion in the constraints).

## Review of `keccak256` sponge

The `keccak256` hash function on variable length byte arrays works by two main steps:

1. Padding the input to a multiple of `RATE_IN_BYTES = 136` bytes. The padding can be described as appending a `1` bit, then multiple `0`s, and another `1` to get the length to a multiple of `RATE_IN_BYTES`. In bytes this means appending `0x80`, then multiples `0x00` and a final `0x01`.
2. Absorb the padded input `RATE_IN_BYTES` bytes at a time into the state, and then applying the `keccak-f` permutation. Here absorb means to XOR the input with the state.

The output is "squeezed" by reading the first `32` bytes of the state. The combination of absorb and squeeze is what makes the `keccak256` hash function a sponge construction.

## VM AIR

In our VM's `keccak256` hasher AIR, the AIR will add columns and constraints to the `keccak-f` AIR to make it stateful, meaning that the transition of `preimage` between different `keccak-f` permutations will be constrained based on the instructions received.

We add `KECCAK_RATE_U16S = 136 / 2` columns for the input to be absorbed.

# References

- Official Keccak [spec summary](https://keccak.team/keccak_specs_summary.html)
