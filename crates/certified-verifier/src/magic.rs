//! Type and version headers for the certified-verifier wire blobs.
//!
//! Each blob starts with a four-byte ASCII identifier followed by a
//! little-endian `u32` schema version. These values mirror `magicProof`,
//! `magicVk`, `magicPv`, and `wireVersion` in the upstream Lean SWIRL decoder,
//! plus the VM verifier's `baselineMagic` and `userPvsMagic` values.

use std::io::{Result, Write};

/// Current wire schema version.
pub(crate) const WIRE_VERSION: u32 = 1;

/// Identifier for a proof blob.
pub(crate) const MAGIC_PROOF: [u8; 4] = *b"PROF";

/// Identifier for a verifying-key blob.
pub(crate) const MAGIC_VK: [u8; 4] = *b"SVKY";

/// Identifier for a public-values blob.
pub(crate) const MAGIC_PUBLIC_VALUES: [u8; 4] = *b"PUBV";

/// Identifier for a VM verification-baseline blob.
pub const MAGIC_VM_BASELINE: [u8; 4] = *b"VMBL";

/// Identifier for a VM user-public-values proof blob.
pub const MAGIC_USER_PUBLIC_VALUES: [u8; 4] = *b"UPVS";

/// Write the 8-byte header (magic + version) for one wire blob.
pub(crate) fn write_header<W: Write>(writer: &mut W, magic: [u8; 4]) -> Result<()> {
    writer.write_all(&magic)?;
    writer.write_all(&WIRE_VERSION.to_le_bytes())
}
