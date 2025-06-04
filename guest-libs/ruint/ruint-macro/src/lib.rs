#![doc = include_str!("../README.md")]
#![warn(clippy::all, clippy::pedantic, clippy::cargo, clippy::nursery)]

use proc_macro::{Delimiter, Group, Ident, Literal, Punct, Spacing, Span, TokenStream, TokenTree};
use std::fmt::{self, Write};

// Repeat the crate doc.
#[doc = include_str!("../README.md")]
#[proc_macro]
pub fn uint(stream: TokenStream) -> TokenStream {
    Transformer::new(None).transform_stream(stream)
}

/// Same as [`uint`], but with the first token always being a
/// [group](proc_macro::Group) containing the `ruint` crate path.
///
/// This allows the macro to be used in a crates that don't on `ruint` through a
/// wrapper `macro_rules!` that passes `$crate` as the path.
///
/// This is an implementation detail and should not be used directly.
#[proc_macro]
#[doc(hidden)]
pub fn uint_with_path(stream: TokenStream) -> TokenStream {
    let mut stream_iter = stream.into_iter();
    let Some(TokenTree::Group(group)) = stream_iter.next() else {
        return error(
            Span::call_site(),
            "Expected a group containing the `ruint` crate path",
        )
        .into();
    };
    Transformer::new(Some(group.stream())).transform_stream(stream_iter.collect())
}

#[derive(Copy, Clone, PartialEq, Debug)]
enum LiteralBaseType {
    Uint,
    Bits,
}

impl LiteralBaseType {
    const PATTERN: &'static [char] = &['U', 'B'];
}

impl fmt::Display for LiteralBaseType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Uint => f.write_str("Uint"),
            Self::Bits => f.write_str("Bits"),
        }
    }
}

impl std::str::FromStr for LiteralBaseType {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "U" => Ok(Self::Uint),
            "B" => Ok(Self::Bits),
            _ => Err(()),
        }
    }
}

/// Construct a compiler error message.
// FEATURE: (BLOCKED) Replace with Diagnostic API when stable.
// See <https://doc.rust-lang.org/stable/proc_macro/struct.Diagnostic.html>
fn error(span: Span, message: &str) -> TokenTree {
    // See: https://docs.rs/syn/1.0.70/src/syn/error.rs.html#243
    let tokens = TokenStream::from_iter(vec![
        TokenTree::Ident(Ident::new("compile_error", span)),
        TokenTree::Punct(Punct::new('!', Spacing::Alone)),
        TokenTree::Group({
            let mut group = Group::new(
                Delimiter::Brace,
                TokenStream::from_iter(vec![TokenTree::Literal(Literal::string(message))]),
            );
            group.set_span(span);
            group
        }),
    ]);
    TokenTree::Group(Group::new(Delimiter::None, tokens))
}

fn parse_digits(value: &str) -> Result<Vec<u64>, String> {
    // Parse base
    let (base, digits) = if value.len() >= 2 {
        let (prefix, remainder) = value.split_at(2);
        match prefix {
            "0x" => (16_u8, remainder),
            "0o" => (8, remainder),
            "0b" => (2, remainder),
            _ => (10, value),
        }
    } else {
        (10, value)
    };

    // Parse digits in base
    let mut limbs = vec![0_u64];
    for c in digits.chars() {
        // Read next digit
        let digit = match c {
            '0'..='9' => c as u64 - '0' as u64,
            'a'..='f' => c as u64 - 'a' as u64 + 10,
            'A'..='F' => c as u64 - 'A' as u64 + 10,
            '_' => continue,
            _ => return Err(format!("Invalid character '{c}'")),
        };
        #[allow(clippy::cast_lossless)]
        if digit > base as u64 {
            return Err(format!(
                "Invalid digit {c} in base {base} (did you forget the `0x` prefix?)"
            ));
        }

        // Multiply result by base and add digit
        let mut carry = digit;
        #[allow(clippy::cast_lossless)]
        #[allow(clippy::cast_possible_truncation)]
        for limb in &mut limbs {
            let product = (*limb as u128) * (base as u128) + (carry as u128);
            *limb = product as u64;
            carry = (product >> 64) as u64;
        }
        if carry > 0 {
            limbs.push(carry);
        }
    }
    Ok(limbs)
}

fn pad_limbs(bits: usize, mut limbs: Vec<u64>) -> Option<Vec<u64>> {
    // Get limb count and mask
    let num_limbs = (bits + 63) / 64;
    let mask = if bits == 0 {
        0
    } else {
        let bits = bits % 64;
        if bits == 0 {
            u64::MAX
        } else {
            (1 << bits) - 1
        }
    };

    // Remove trailing zeros, pad with zeros
    while limbs.len() > num_limbs && limbs.last() == Some(&0) {
        limbs.pop();
    }
    while limbs.len() < num_limbs {
        limbs.push(0);
    }

    // Validate length
    if limbs.len() > num_limbs || limbs.last().copied().unwrap_or(0) > mask {
        return None;
    }
    Some(limbs)
}

fn parse_suffix(source: &str) -> Option<(LiteralBaseType, usize, &str)> {
    // Parse into value, bits, and base type.
    let suffix_index = source.rfind(LiteralBaseType::PATTERN)?;
    let (value, suffix) = source.split_at(suffix_index);
    let (base_type, bits) = suffix.split_at(1);
    let base_type = base_type.parse::<LiteralBaseType>().ok()?;
    let bits = bits.parse::<usize>().ok()?;

    // Ignore hexadecimal Bits literals without `_` before the suffix.
    if base_type == LiteralBaseType::Bits && value.starts_with("0x") && !value.ends_with('_') {
        return None;
    }
    Some((base_type, bits, value))
}

struct Transformer {
    /// The `ruint` crate path.
    /// Note that this stream's span must be used in order for the `$crate` to
    /// work.
    ruint_crate: TokenStream,
}

impl Transformer {
    fn new(ruint_crate: Option<TokenStream>) -> Self {
        Self {
            ruint_crate: ruint_crate.unwrap_or_else(|| "::ruint".parse().unwrap()),
        }
    }

    /// Construct a `<{base_type}><{bits}>` literal from `limbs`.
    fn construct(&self, base_type: LiteralBaseType, bits: usize, limbs: &[u64]) -> TokenStream {
        let mut limbs_str = String::new();
        for limb in limbs {
            write!(&mut limbs_str, "0x{limb:016x}_u64, ").unwrap();
        }
        let limbs_str = limbs_str.trim_end_matches(", ");
        let limbs = (bits + 63) / 64;
        let source = format!("::{base_type}::<{bits}, {limbs}>::from_limbs([{limbs_str}])");

        let mut tokens = self.ruint_crate.clone();
        tokens.extend(source.parse::<TokenStream>().unwrap());
        tokens
    }

    /// Transforms a [`Literal`] and returns the substitute [`TokenStream`].
    fn transform_literal(&self, source: &str) -> Result<Option<TokenStream>, String> {
        // Check if literal has a suffix we accept.
        let Some((base_type, bits, value)) = parse_suffix(source) else {
            return Ok(None);
        };

        // Parse `value` into limbs.
        // At this point we are confident the literal was for us, so we throw errors.
        let limbs = parse_digits(value)?;

        // Pad limbs to the correct length.
        let Some(limbs) = pad_limbs(bits, limbs) else {
            let value = value.trim_end_matches('_');
            return Err(format!("Value too large for {base_type}<{bits}>: {value}"));
        };

        Ok(Some(self.construct(base_type, bits, &limbs)))
    }

    /// Recurse down tree and transform all literals.
    fn transform_tree(&self, tree: TokenTree) -> TokenTree {
        match tree {
            TokenTree::Group(group) => {
                let delimiter = group.delimiter();
                let span = group.span();
                let stream = self.transform_stream(group.stream());
                let mut transformed = Group::new(delimiter, stream);
                transformed.set_span(span);
                TokenTree::Group(transformed)
            }
            TokenTree::Literal(a) => {
                let span = a.span();
                let source = a.to_string();
                let mut tree = match self.transform_literal(&source) {
                    Ok(Some(stream)) => TokenTree::Group({
                        let mut group = Group::new(Delimiter::None, stream);
                        group.set_span(span);
                        group
                    }),
                    Ok(None) => TokenTree::Literal(a),
                    Err(message) => error(span, &message),
                };
                tree.set_span(span);
                tree
            }
            tree => tree,
        }
    }

    /// Iterate over a [`TokenStream`] and transform all [`TokenTree`]s.
    fn transform_stream(&self, stream: TokenStream) -> TokenStream {
        stream
            .into_iter()
            .map(|tree| self.transform_tree(tree))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_size() {
        assert_eq!(parse_digits("0"), Ok(vec![0]));
        assert_eq!(parse_digits("00000"), Ok(vec![0]));
        assert_eq!(parse_digits("0x00"), Ok(vec![0]));
        assert_eq!(parse_digits("0b0000"), Ok(vec![0]));
        assert_eq!(parse_digits("0b0000000"), Ok(vec![0]));

        assert_eq!(parse_digits("0"), Ok(vec![0]));
        assert_eq!(parse_digits("00000"), Ok(vec![0]));
        assert_eq!(parse_digits("0x00"), Ok(vec![0]));
        assert_eq!(parse_digits("0b0000"), Ok(vec![0]));
        assert_eq!(parse_digits("0b0000000"), Ok(vec![0]));
    }

    #[test]
    fn test_bases() {
        assert_eq!(parse_digits("10"), Ok(vec![10]));
        assert_eq!(parse_digits("0x10"), Ok(vec![16]));
        assert_eq!(parse_digits("0b10"), Ok(vec![2]));
        assert_eq!(parse_digits("0o10"), Ok(vec![8]));
    }

    #[test]
    #[allow(clippy::unreadable_literal)]
    fn test_overflow_during_parsing() {
        assert_eq!(parse_digits("258664426012969093929703085429980814127835149614277183275038967946009968870203535512256352201271898244626862047232"), Ok(vec![0, 15125697203588300800, 6414901478162127871, 13296924585243691235, 13584922160258634318, 121098312706494698]));
        assert_eq!(parse_digits("2135987035920910082395021706169552114602704522356652769947041607822219725780640550022962086936576"), Ok(vec![0, 0, 0, 0, 0, 1]));
    }
}
