//! Magic + version constants for the three wire inputs.
//!
//! Quoting `notes/lean-verifier-wire-format.md §A.1`:
//!
//! > Each of the three top-level inputs begins with an 8-byte header:
//! > 4 bytes magic + 4 bytes version. The version is a `u32` little-endian
//! > word equal to `1` for the v1 schema.
//!
//! The magic bytes are an ASCII mnemonic written verbatim (no endian flip).
//! The Lean side keeps a matching `MagicProof`, `MagicVk`, `MagicPv`
//! triple in `Swirl.Protocol.Noninteractive.Wire` so the literal sequence
//! is the same byte-for-byte.

use std::io::{Result, Write};

/// Schema version word (v1).
pub const WIRE_VERSION: u32 = 1;

/// Magic for the `Proof` blob — ASCII `PROF`.
pub const MAGIC_PROOF: [u8; 4] = *b"PROF";

/// Magic for the `MultiStarkVerifyingKey` blob — ASCII `SVKY`.
pub const MAGIC_VK: [u8; 4] = *b"SVKY";

/// Magic for the `PublicValuesFor vk` blob — ASCII `PUBV`.
pub const MAGIC_PUBLIC_VALUES: [u8; 4] = *b"PUBV";

/// Write the 8-byte header (magic + version) for one wire blob.
pub fn write_header<W: Write>(writer: &mut W, magic: [u8; 4]) -> Result<()> {
    writer.write_all(&magic)?;
    writer.write_all(&WIRE_VERSION.to_le_bytes())
}
