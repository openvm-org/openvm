//! Primitive byte helpers shared by every wire encoder.
//!
//! See `notes/lean-verifier-wire-format.md §A.3`:
//! - `Bool`: 1 byte, `0x00`/`0x01`.
//! - `Option<T>`: 1 byte tag (`0x00`/`0x01`), then `T` when some.
//! - `Nat` / `UInt32`: 4 bytes little-endian `u32`.
//! - `Int`: 4 bytes little-endian sign-extended `i32`.
//! - `List T`: `u32` length prefix, then values.
//!
//! The wire is explicitly `u32`-length-prefixed (not `usize`-prefixed),
//! so `len()` is narrowed via `try_into` and surfaces `io::Error` if a
//! collection exceeds `2^32 − 1`. That cannot legitimately happen for
//! any FibonacciAir / BabyBearPoseidon2 proof at v1, but it is a hard
//! decoder requirement so we fail fast.

use std::io::{Error, ErrorKind, Result, Write};

use openvm_stark_backend::p3_field::PrimeField32;

/// Write a `u32` as 4 little-endian bytes.
#[inline]
pub fn write_u32<W: Write>(writer: &mut W, value: u32) -> Result<()> {
    writer.write_all(&value.to_le_bytes())
}

/// Narrow a `usize` to `u32` for wire emission. Errors with
/// `InvalidData` if the value would not fit in 32 bits.
#[inline]
pub fn narrow_usize(value: usize) -> Result<u32> {
    u32::try_from(value).map_err(|_| {
        Error::new(
            ErrorKind::InvalidData,
            format!("wire-format value {value} exceeds u32 range"),
        )
    })
}

/// Write a `usize` as a 4-byte wire `u32`. Errors on overflow.
#[inline]
pub fn write_usize_as_u32<W: Write>(writer: &mut W, value: usize) -> Result<()> {
    write_u32(writer, narrow_usize(value)?)
}

/// Write a Lean-side `Nat` field as a 4-byte wire `u32`.
#[inline]
pub fn write_nat<W: Write>(writer: &mut W, value: usize) -> Result<()> {
    write_usize_as_u32(writer, value)
}

/// Write a Lean-side `Int` field as a 4-byte wire `i32`. Errors if the
/// upstream `isize` value falls outside `i32` range.
#[inline]
pub fn write_isize_as_i32<W: Write>(writer: &mut W, value: isize) -> Result<()> {
    let narrowed = i32::try_from(value).map_err(|_| {
        Error::new(
            ErrorKind::InvalidData,
            format!("wire-format value {value} exceeds i32 range"),
        )
    })?;
    writer.write_all(&narrowed.to_le_bytes())
}

/// Write a Lean-side `Bool` as a 1-byte tag.
#[inline]
pub fn write_bool<W: Write>(writer: &mut W, value: bool) -> Result<()> {
    let byte = if value { 0x01u8 } else { 0x00u8 };
    writer.write_all(&[byte])
}

/// Write a `u32` length prefix.
#[inline]
pub fn write_length_prefix<W: Write>(writer: &mut W, len: usize) -> Result<()> {
    write_usize_as_u32(writer, len)
}

/// Write a `u32` list of `usize` values.
pub fn write_usize_list<W: Write>(writer: &mut W, items: &[usize]) -> Result<()> {
    write_length_prefix(writer, items.len())?;
    for value in items {
        write_usize_as_u32(writer, *value)?;
    }
    Ok(())
}

/// Write a `u32` list of `u32` values.
pub fn write_u32_list<W: Write>(writer: &mut W, items: &[u32]) -> Result<()> {
    write_length_prefix(writer, items.len())?;
    for value in items {
        write_u32(writer, *value)?;
    }
    Ok(())
}

/// Write an `Option<&T>` with a 1-byte tag and an inner-writer for the
/// `some` payload. The inner writer is invoked only when `value` is
/// `Some(_)`.
pub fn write_option<T, W: Write>(
    writer: &mut W,
    value: Option<&T>,
    write_inner: impl FnOnce(&mut W, &T) -> Result<()>,
) -> Result<()> {
    match value {
        None => writer.write_all(&[0x00u8]),
        Some(inner) => {
            writer.write_all(&[0x01u8])?;
            write_inner(writer, inner)
        }
    }
}

/// Write an `Option<usize>` as `1-byte tag + 4-byte u32 if some`.
pub fn write_option_usize<W: Write>(writer: &mut W, value: Option<usize>) -> Result<()> {
    write_option(writer, value.as_ref(), |w, v| write_usize_as_u32(w, *v))
}

/// Write a canonical `PrimeField32` base value as 4 little-endian bytes.
#[inline]
pub fn write_prime_field32<F: PrimeField32, W: Write>(writer: &mut W, value: &F) -> Result<()> {
    writer.write_all(&value.as_canonical_u32().to_le_bytes())
}

/// Write a length-prefixed list of `usize` values, one Nat per item.
pub fn write_nat_list<W: Write>(writer: &mut W, items: &[usize]) -> Result<()> {
    write_usize_list(writer, items)
}
