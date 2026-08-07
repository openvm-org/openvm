//! Type and version headers for the three wire blobs.
//!
//! Each blob starts with a four-byte ASCII identifier followed by a
//! little-endian `u32` schema version. These values mirror `magicProof`,
//! `magicVk`, `magicPv`, and `wireVersion` in the upstream Lean
//! `Swirl.Protocol.Noninteractive.Wire.Raw` decoder.

use std::io::{Result, Write};

/// Current wire schema version.
pub const WIRE_VERSION: u32 = 1;

/// Identifier for a proof blob.
pub const MAGIC_PROOF: [u8; 4] = *b"PROF";

/// Identifier for a verifying-key blob.
pub const MAGIC_VK: [u8; 4] = *b"SVKY";

/// Identifier for a public-values blob.
pub const MAGIC_PUBLIC_VALUES: [u8; 4] = *b"PUBV";

/// Write the 8-byte header (magic + version) for one wire blob.
pub fn write_header<W: Write>(writer: &mut W, magic: [u8; 4]) -> Result<()> {
    writer.write_all(&magic)?;
    writer.write_all(&WIRE_VERSION.to_le_bytes())
}
