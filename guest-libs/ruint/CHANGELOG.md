# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!-- Section names: Added, Changed, Deprecated, Removed, Fixed, Security -->

## [Unreleased]

## [1.14.0] - 2025-03-25

### Added

- Add const `not` function ([#442])
- Make `leading_zeros`, `leading_ones`, `count_ones`, `count_zeros`, `bit_len`, `byte_len`, `is_power_of_two` functions `const` ([#442])
- `random`, `random_with`, `randomize`, `randomize_with` methods ([#444])
- rand 0.9 support ([#445])
- Add `const ONE` ([#448])

### Changed

- Update `try_from_{b,l}e_slice` documentation ([#439])

### Fixed

- Fix `checked_byte` bounds check and make it const ([#438])

[#438]: https://github.com/recmo/uint/pull/438
[#439]: https://github.com/recmo/uint/pull/439
[#442]: https://github.com/recmo/uint/pull/442
[#444]: https://github.com/recmo/uint/pull/444
[#445]: https://github.com/recmo/uint/pull/445
[#448]: https://github.com/recmo/uint/pull/448

## [1.13.1] - 2025-02-18

### Fixed

- Remove unused `hex` dependencies which broke `no_std` ([#433])

[#433]: https://github.com/recmo/uint/pull/433

## [1.13.0] - 2025-02-18

### Added

- Support for borsh @ 1.5 ([#416])
- `copy_le_to_slice` family to allow easier writing to pre-allocated buffers ([#424])
- add `Uint::checked_byte(idx: usize) -> Option<u8>` ([#429])

### Changed

- Unpin proptest ([#426])

### Fixed

- Update documentation related to `Uint::byte` and knuth divison ([#429])
- fix: swap bytes for `as_le_bytes` in big endian world ([#431])

[#416]: https://github.com/recmo/uint/pull/416
[#424]: https://github.com/recmo/uint/pull/424
[#426]: https://github.com/recmo/uint/pull/426
[#429]: https://github.com/recmo/uint/pull/429
[#431]: https://github.com/recmo/uint/pull/431

## [1.12.4] - 2024-12-16

### Added

- Added `Uint::square_redc`. ([#402])
- Support for diesel @ 2.2 ([#404])
- Support for sqlx @ 0.8 ([#400])
- Support for fastrlp @ 0.4 ([#401])
- Added support for [`subtle`](https://docs.rs/subtle) and [`der`](https://docs.rs/der) ([#399])

### Removed

- Support for sqlx @ 0.7. This is a breaking change, outside of
  regular semver policy, as 0.7 contains a security vulnerability  ([#400])

### Fixed

- `Uint::mul_redc` is now alloc free ([#402])

[#399]: https://github.com/recmo/uint/pull/399
[#400]: https://github.com/recmo/uint/pull/400
[#401]: https://github.com/recmo/uint/pull/401
[#404]: https://github.com/recmo/uint/pull/404
[#402]: https://github.com/recmo/uint/pull/402

## [1.12.3] - 2024-06-03

### Changed

- Use borrowing/carrying ops in add/sub, remove bound checks in shifts ([#366])
- Make `mul_mod` non-allocating ([#373])

### Fixed

- Add `alloc` requirement to `num-traits` feature [#363]

[#363]: https://github.com/recmo/uint/pull/363
[#366]: https://github.com/recmo/uint/pull/366
[#373]: https://github.com/recmo/uint/pull/373

## [1.12.1] - 2024-03-12

### Fixed

- docs.rs build ([#356])
- `uint!` in item position ([#360])

[#356]: https://github.com/recmo/uint/pull/356
[#360]: https://github.com/recmo/uint/pull/360

## [1.12.0] - 2024-02-27

### Added

- Wrap the `uint!` macro to allow usage without needing `uint` import ([#350])

### Fixed

- Overflow check in `overflowing_shr` implementation ([#347])

[#347]: https://github.com/recmo/uint/pull/347
[#350]: https://github.com/recmo/uint/pull/350

## [1.11.1] - 2023-11-18

### Fixed

- Typo in `Shr` implementation ([#343])

[#343]: https://github.com/recmo/uint/pull/343

### Added

-   Enable `SSZ` ([#344])

[#344]: https://github.com/recmo/uint/pull/344

## [1.11.0] - 2023-10-31

### Added

- `bytemuck` feature ([#292])
- `Uint::is_zero() -> bool` ([#296])
- `num-traits` features ([#298])
- `U768` alias ([#310])
- Improved `add` and `sub` performance ([#316])
- Made `add` and `sub` functions `const` ([#324])
- Made `{from,to}_{b,l}e_bytes` `const` ([#329])

### Fixed

- Restricted RLP decoding to match the RLP spec and disallow leading zeros ([#335])
- `leading_ones` failed for non-aligned sizes.

[#292]: https://github.com/recmo/uint/pull/292
[#296]: https://github.com/recmo/uint/pull/296
[#298]: https://github.com/recmo/uint/pull/298
[#310]: https://github.com/recmo/uint/pull/310
[#316]: https://github.com/recmo/uint/pull/316
[#324]: https://github.com/recmo/uint/pull/324
[#329]: https://github.com/recmo/uint/pull/329
[#335]: https://github.com/recmo/uint/pull/335

## [1.10.1] - 2023-07-30

### Fixed

- Fixed some support features ([#289])

[#289]: https://github.com/recmo/uint/pull/289

## [1.10.0] - 2023-07-30

### Added

- Support for `no_std` environments ([#274])
- `alloc` feature ([#277])

[#274]: https://github.com/recmo/uint/pull/274
[#277]: https://github.com/recmo/uint/pull/277

## [1.9.0] - 2023-07-25

### Added

- Introduce `ark-ff-04` feature flag for conversion to `ark-ff@0.4` types
- Support for [`alloy-rlp`](https://github.com/alloy-rs/rlp)
- MSRV (Minimum Supported Rust Version) is now set at 1.65.0, from previously undefined
- Implement `TryFrom<bool>` for `Uint`
- New method: `byte`

### Changed

- Make `serde::Deserialize` impl more permissive
- Use Ethereum `Quantity` encoding for serde serialization when human-readable
- Fix error in `from_base_be` that allowed instantiation of overflowing `Uint`
- Updated `fastrlp` to `0.3`, `pyo3` to `0.19`, and `sqlx-core` to `0.7`
- Improved `fastrlp` perfomance
- Improved `proptest` performance
- Made `support` module and its modules public
- Made more `algorithm` functions public
- Constified `as_le_slice` and `as_le_bytes`

### Removed

- Automatic detection of nightly features. Enable them instead with the `nightly` cargo feature
- Dependency on `derive_more`

### Fixed

- `from_base_le` implementation by reversing the input iterator

## [1.8.0] — 2023-04-19

### Added

- Support `bn-rs`, `serde` and `uint!` for `Bits`

### Fixed

- Serde human readable now encodes the empty bitstring as `0x0` and rejects zero prefixes.

## [1.7.0] — 2022-10-31

### Added

- Support `rlp` for `Bits`

### Fixed

- Edge case in which an overflow occurs when parsing a `Uint` with `uint!` ([#199]).

[#199]: https://github.com/recmo/uint/issues/199

## [1.6.0] — 2022-10-28

### Added

- `TryFrom<Uint> for usize`
- Bit type aliases (`B128`, `B160`, `B256`, `B512`, ...)
- `From` and `Into` trait implementations for `primitive-types` bit types
- Support for `bn-rs`
- Derive `Default` for `Bits`

### Changed

- (Breaking) Changed the arguments of `pow` and `log` to `Uint`.
- More efficient `wrapping_pow` implementation.

## [1.5.1] — 2022-10-24

### Changed

- Performance improvements in `wrapping_mul` and `from_bytes_be`.

## [1.5.0] — 2022-10-24

### Added

- Add `parity-scale-codec` support.
- Added unstable `algorithms::div` module and improved div algorithm.

## [1.4.1] — 2022-10-15

### Changed

- Made `primitive-types` version flexible.

## [1.4.0] — 2022-10-02

### Added

- Add `Pyo3` support.
- `from` now supports `Uint` arguments.
- `saturating_from`, `wrapping_from`, `to`, `wrapping_to`, `saturating_to`.
- `wrapping_from_limbs_slice`, `overflowing_from_limbs_slice`, `saturating_from_limbs_slice`.
- Add `zeroize` and `valuable` support.

### Changed

- `ToUintError` and `FromUintError` now contain wrapped value and other context.
- `from_uint` and `checked_from_uint` are now deprecated.

## [1.3.0] — 2022-06-08

### Added

- Added `inv_mod`, `mul_redc`, `gcd`, `gcd_extended`, `lcm`.
- Added `sqlx` support.

### Changed

- Renamed `ring_inverse` to `inv_ring`.

## [1.2.0] — 2022-06-03

### Added

- Added `reduce_mod`, `add_mod`, `mul_mod`, `pow_mod`.
- Added `num-bigint` and `ark-ff` support.
- Made `algorithms` public, but with unstable API for now.

### Changed

- Marked `Uint::as_limbs_mut` as unsafe.
- Unified `mul` implementations and move to `algorithms`.

### Fixed

- `uint!` macro incorrectly accepting hex digits in decimal.

## [1.1.0] — 2022-05-31

### Added

- Added `saturating_shl`.
- Added `approx_log`, `approx_log2`, `approx_log10` for `f64` log approximations.
- Added `approx_pow2` to construct from `f64` log2 approximation.
- Added `root` computing integer roots.

### Changed

- Made logarithms `usize` to match `BITS` in `pow`, `log` functions.
- Applied `track_caller` to div/rem ops to track div-by-zero easier.

## [1.0.0] — 2022-05-28

### Added

- Added comparison.
- Added add, sub, neg and sum functions.
- Added mul functions.
- Added division and remainder functions.
- Added pow and log functions.
- Added `next_power_of_two` and `next_multiple_of` functions.
- Added `checked_from_limb_slice` and `from_uint`.

### Changed

- `from_limb_slice` now handles arbitrary length slices.

## [0.3.0] — 2022-05-23

### Added

- All the binary operations (not, and, or, xor, shifts, rotate, etc)
- `Bits`, a newtype wrapped `Uint` restricted to non-numeric operations.
- Postgres `FromSql` support and JSON column support.
- `from_base_le` and `from_base_be` base conversion.
- `from_str_radix` string base conversion up to base64.
- `FromStr` trait with decimal, hex, octal and binary support.

### Changed

- `reverse_bits` is now by value instead of `&mut self`.

## [0.2.1] — 2022-05-18

### Added

- Extensive Postgres ToSql support supporting many column types.
- `TryFrom<Uint>` for primitive integer types.
- Added `From` conversions to `f32` and `f64`.
- Implement all rust formatting: `Debug`, `Display`, decimal, hex, etc.
- `>>=` operator.
- `to_base_le` and `to_base_le` base conversion spigots
- `reverse_bits`, `most_sigificant_bits` bit methods.
- Optimized `as_le_{slice,bytes)_*` accessors.

### Changed

- Rewrote `to_{be,le}_bytes_*` to use optimized methods. This has trickle-down
  effects for a lot of conversions and formatting.

## [0.2.0] — 2022-05-16

### Added

- Changelog
- CI test on stable Rust.
- Common bit-size aliases and nightly-only `Uint<BITS>` alias.
- Added `to_{be/le}_bytes_vec` and made `try_from_le_byte_iter` public.
- Added `rlp` and `fastrlp` support.
- Added `into_limbs`, `leading_zeros`, `bit_len`, `byte_len`, `checked_log2`.
- Added `primitive-types` support.

### Changed

- Changed to `Uint<BITS, LIMBS>` to get stable compatibility!
- Added generic `BYTES` parameter to `to_{be/le}_bytes`.
- Renamed `try_from_{be/le}_slice`.

## [0.1.0] — 2022-05-15

### Added

- Const-generic `Uint` structure.
- Basic `overflowing_add` implementation.
- Algorithms for division and gcd (currently unused).
- `uint!` and `const_for!` macros.
- Documentation with examples.
- Support for rand, arbitrary, quickcheck, proptest, serde
- Github actions for linting, testing, code coverage, cargo-audit.
- Pushed to crates.io.

<!-- links to version -->

[unreleased]: https://github.com/recmo/uint/compare/v1.14.0...HEAD
[1.14.0]: https://github.com/recmo/uint/releases/tag/v1.14.0
[1.13.1]: https://github.com/recmo/uint/releases/tag/v1.13.1
[1.13.0]: https://github.com/recmo/uint/releases/tag/v1.13.0
[1.12.4]: https://github.com/recmo/uint/releases/tag/v1.12.4
[1.12.3]: https://github.com/recmo/uint/releases/tag/v1.12.3
[1.12.1]: https://github.com/recmo/uint/releases/tag/v1.12.1
[1.12.0]: https://github.com/recmo/uint/releases/tag/v1.12.0
[1.11.1]: https://github.com/recmo/uint/releases/tag/v1.11.1
[1.11.0]: https://github.com/recmo/uint/releases/tag/v1.11.0
[1.10.1]: https://github.com/recmo/uint/releases/tag/v1.10.1
[1.10.0]: https://github.com/recmo/uint/releases/tag/v1.10.0
[1.9.0]: https://github.com/recmo/uint/releases/tag/v1.9.0
[1.8.0]: https://github.com/recmo/uint/releases/tag/v1.8.0
[1.7.0]: https://github.com/recmo/uint/releases/tag/v1.7.0
[1.6.0]: https://github.com/recmo/uint/releases/tag/v1.6.0
[1.5.1]: https://github.com/recmo/uint/releases/tag/v1.5.1
[1.5.0]: https://github.com/recmo/uint/releases/tag/v1.5.0
[1.4.1]: https://github.com/recmo/uint/releases/tag/v1.4.1
[1.4.0]: https://github.com/recmo/uint/releases/tag/v1.4.0
[1.3.0]: https://github.com/recmo/uint/releases/tag/v1.3.0
[1.2.0]: https://github.com/recmo/uint/releases/tag/v1.2.0
[1.1.0]: https://github.com/recmo/uint/releases/tag/v1.1.0
[1.0.0]: https://github.com/recmo/uint/releases/tag/v1.0.0
[0.3.0]: https://github.com/recmo/uint/releases/tag/v0.3.0
[0.2.1]: https://github.com/recmo/uint/releases/tag/v0.2.1
[0.2.0]: https://github.com/recmo/uint/releases/tag/v0.2.0
[0.1.0]: https://github.com/recmo/uint/releases/tag/v0.1.0
