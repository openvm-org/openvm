//! Encoder for the Lean `PublicValuesFor vk` wire type.
//!
//! The Rust source for the per-AIR public values lives on `Proof<SC>`
//! as `proof.public_values: Vec<Vec<SC::F>>`. The Lean wire treats
//! public values as a separate blob keyed by an already-parsed vk, so
//! the encoder takes the `Vec<Vec<SC::F>>` directly and emits the
//! `airCount` and per-AIR vector lengths exactly as the decoder
//! expects.

use std::io::{Error, ErrorKind, Result, Write};

use openvm_stark_backend::{codec::EncodableConfig, keygen::types::MultiStarkVerifyingKey};

use super::{
    magic::{write_header, MAGIC_PUBLIC_VALUES},
    primitives::{write_length_prefix, write_usize_as_u32},
};

/// Encode per-AIR public values after validating their shape against the key.
///
/// The encoder cross-checks the `public_values` shape against the
/// supplied `vk`: it errors with `InvalidData` if `public_values.len()`
/// does not match `vk.airCount`, or if any per-AIR length does not
/// match `vk.publicValueCount air`.
pub fn write_public_values<SC: EncodableConfig, W: Write>(
    writer: &mut W,
    vk: &MultiStarkVerifyingKey<SC>,
    public_values: &[Vec<SC::F>],
) -> Result<()> {
    write_header(writer, MAGIC_PUBLIC_VALUES)?;
    let air_count = vk.inner.per_air.len();
    if public_values.len() != air_count {
        return Err(Error::new(
            ErrorKind::InvalidData,
            format!(
                "wire-format: public_values.len() = {} but vk.airCount = {}",
                public_values.len(),
                air_count
            ),
        ));
    }
    write_length_prefix(writer, air_count)?;
    for (air_idx, pv) in public_values.iter().enumerate() {
        let expected = vk.inner.per_air[air_idx].params.num_public_values;
        if pv.len() != expected {
            return Err(Error::new(
                ErrorKind::InvalidData,
                format!(
                    "wire-format: public_values[{air_idx}].len() = {} but \
                     vk.publicValueCount = {expected}",
                    pv.len()
                ),
            ));
        }
        write_usize_as_u32(writer, expected)?;
        for value in pv {
            SC::encode_base_field(value, writer)?;
        }
    }
    Ok(())
}
