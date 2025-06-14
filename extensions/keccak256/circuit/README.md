# Spec

## Review of `keccak-f` AIR

The `keccak-air` from Plonky3 is an AIR that does one `keccak-f[1600]` permutation every `NUM_ROUNDS = 24` rows (henceforth we call this the `keccak-f` AIR to avoid confusion). All rows in the round have the same `preimage`, which is the starting state prior to the permutation, represented as `5 * 5 * 4` `u16` limbs (the state in the spec is `5 * 5` `u64`s, but since the AIR uses a 31-bit field, the `u64` is broken into `u16`s).

The `keccak-f` permutation copies `preimage` to `A` and mutates `A` over rounds. The mutations are materialized in the `keccak-f` AIR in `A'` and `A''` arrays. While the bits of `A'` are materialized, the bits of `preimage` and `A` are never materialized (there is an implicit bit composition in the constraints).

## Review of `keccak256` sponge

The `keccak256` hash function on variable length byte arrays works by two main steps:

1. Padding the input to a multiple of `RATE_IN_BYTES = 136` bytes. The padding can be described as appending a `1` bit, then multiple `0`s, and another `1` to get the length to a multiple of `RATE_IN_BYTES`. In bytes this means appending `0x01`, then multiples `0x00` and a final `0x80` because the keccak-f state conversion uses [little-endian](https://keccak.team/keccak_bits_and_bytes.html).
2. Absorb the padded input `RATE_IN_BYTES` bytes at a time into the state, and then applying the `keccak-f` permutation. Here absorb means to XOR the input with the state.

The output is "squeezed" by reading the first `32` bytes of the state. The combination of absorb and squeeze is what makes the `keccak256` hash function a sponge construction.

## VM AIR

In our VM's `keccak256` hasher AIR, the AIR will add columns and constraints to the `keccak-f` AIR to make it stateful, meaning that the transition of `preimage` between different `keccak-f` permutations will be constrained based on the instructions received.

We add `KECCAK_RATE_U16S = 136 / 2` columns for the input to be absorbed.
It seems to handle padding in a single AIR row there is no alternate to having `136` columns with bits to represent whether it is padding byte or not.

The absorb step must correctly constrain that the input bytes are XORed with the end-state in the last round and equals the next permutation's `preimage`. The end-state is accessed via `a_prime_prime_prime()`. Note that both `preimage` and `a_prime_prime_prime()` are represented as `u16`s. However we can only XOR at most 8-bit limbs. Without changing the `keccak-f` AIR itself, we can use a trick:
if we already have a 16-bit limb `x` and we also provide a 8-bit limb `hi = x >> 8`, assuming `x` and `hi` have been range checked, we can use the expression `lo = x - hi * 256` for the low byte. If `lo` is range checked to `8`-bits, this constrains a valid byte decomposition of `x` into `hi, lo`. This means in terms of trace cells, it is equivalent to provide `x, hi` versus `hi, lo`.

The constraints are in [air.rs](./src/air.rs). Notably we use an XOR lookup table for byte XORs in the absorb step.

## Future Improvement

Currently most of the columns in `KeccakOpcodeCols` and `KeccakSpongeCols` only change every `NUM_ROUNDS = 24` rows for a `keccak-f` block. It will likely save more cells if this part is split out into a separate AIR which communicates with the `keccak-f` AIR via interactions. However this requires some care in matching up rows via timestamps, so it is not currently implemented.

# References

- Official Keccak [spec summary](https://keccak.team/keccak_specs_summary.html)
