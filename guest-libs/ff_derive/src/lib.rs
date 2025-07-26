#![recursion_limit = "1024"]
#![allow(clippy::manual_repeat_n)]

extern crate proc_macro;
extern crate proc_macro2;

use std::{iter, str::FromStr};

use num_bigint::BigUint;
use num_integer::Integer;
use num_traits::{One, ToPrimitive, Zero};
use quote::{quote, TokenStreamExt};

mod pow_fixed;

enum ReprEndianness {
    Big,
    Little,
}

impl FromStr for ReprEndianness {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "big" => Ok(ReprEndianness::Big),
            "little" => Ok(ReprEndianness::Little),
            _ => Err(()),
        }
    }
}

impl ReprEndianness {
    fn modulus_repr(&self, modulus: &BigUint, bytes: usize) -> Vec<u8> {
        match self {
            ReprEndianness::Big => {
                let buf = modulus.to_bytes_be();
                iter::repeat(0).take(bytes - buf.len()).chain(buf).collect()
            }
            ReprEndianness::Little => {
                let mut buf = modulus.to_bytes_le();
                buf.extend(iter::repeat(0).take(bytes - buf.len()));
                buf
            }
        }
    }

    // Clippy things methods named from_* don't take self as a parameter
    #[allow(clippy::wrong_self_convention)]
    fn from_repr(&self, name: &syn::Ident, limbs: usize) -> proc_macro2::TokenStream {
        let read_repr = match self {
            ReprEndianness::Big => quote! {
                <#name as ::openvm_algebra_guest::IntMod>::from_be_bytes(&r.as_ref()).unwrap()
            },
            ReprEndianness::Little => quote! {
                <#name as ::openvm_algebra_guest::IntMod>::from_le_bytes(&r.as_ref()).unwrap()
            },
        };

        let zkvm_impl = quote! {
            #read_repr
        };

        let read_repr = match self {
            ReprEndianness::Big => quote! {
                ::ff::derive::byteorder::BigEndian::read_u64_into(r.as_ref(), &mut inner[..]);
                inner.reverse();
            },
            ReprEndianness::Little => quote! {
                ::ff::derive::byteorder::LittleEndian::read_u64_into(r.as_ref(), &mut inner[..]);
            },
        };

        let non_zkvm_impl = quote! {
            {
                use ::ff::derive::byteorder::ByteOrder;

                let mut inner = [0u64; #limbs];
                #read_repr
                #name(inner)
            }
        };

        quote! {
            #[cfg(target_os = "zkvm")]
                let r = #zkvm_impl;
            #[cfg(not(target_os = "zkvm"))]
                let r = #non_zkvm_impl;
        }
    }

    fn to_repr(
        &self,
        repr: proc_macro2::TokenStream,
        mont_reduce_self_params: &proc_macro2::TokenStream,
        limbs: usize,
    ) -> proc_macro2::TokenStream {
        let bytes = limbs * 8;

        let write_repr = match self {
            ReprEndianness::Big => quote! {
                <Self as ::openvm_algebra_guest::IntMod>::to_be_bytes(self)[..#bytes].try_into().unwrap()
            },
            ReprEndianness::Little => quote! {
                <Self as ::openvm_algebra_guest::IntMod>::as_le_bytes(self)[..#bytes].try_into().unwrap()
            },
        };

        let zkvm_impl = quote! {
            #repr(#write_repr)
        };

        let write_repr = match self {
            ReprEndianness::Big => quote! {
                r.0.reverse();
                ::ff::derive::byteorder::BigEndian::write_u64_into(&r.0, &mut repr[..]);
            },
            ReprEndianness::Little => quote! {
                ::ff::derive::byteorder::LittleEndian::write_u64_into(&r.0, &mut repr[..]);
            },
        };

        let non_zkvm_impl = quote! {
            use ::ff::derive::byteorder::ByteOrder;

            let mut r = *self;
            r.mont_reduce(
                #mont_reduce_self_params
            );

            let mut repr = [0u8; #bytes];
            #write_repr
            #repr(repr)
        };

        quote! {
            #[cfg(target_os = "zkvm")]
            {
                #zkvm_impl
            }
            #[cfg(not(target_os = "zkvm"))]
            {
                #non_zkvm_impl
            }
        }
    }

    fn iter_be(&self) -> proc_macro2::TokenStream {
        // We aren't implementing for zkvm here because this function is only used in the prime
        // field repr impl which is a plain array of bytes
        match self {
            ReprEndianness::Big => quote! {self.0.iter()},
            ReprEndianness::Little => quote! {self.0.iter().rev()},
        }
    }
}

/// Derive the `PrimeField` trait.
/// **Warning**: This macro removes the struct definition and inserts our own zkvm-compatible struct
/// into the token stream.
/// Currently the memory layout of the new struct will always be either 32 or 48-bytes, where the
/// smallest that will fit the field's modulus is used. Moduli that are larger than 48-bytes are not
/// yet supported.
// Required attributes: PrimeFieldModulus, PrimeFieldGenerator, PrimeFieldReprEndianness
// Note: In our fork, we changed the macro from a derive macro to an attribute-style macro because
// we need to be able to remove the struct definition and insert our own into the token stream.
#[proc_macro_attribute]
pub fn openvm_prime_field(
    _: proc_macro::TokenStream,
    input: proc_macro::TokenStream,
) -> proc_macro::TokenStream {
    // Parse the type definition
    let ast: syn::Item = syn::parse_macro_input!(input);

    // The attribute should be applied to a struct.
    let mut ast = match ast {
        syn::Item::Struct(ast) => ast,
        _ => {
            return syn::Error::new_spanned(ast, "PrimeField derive only works for structs.")
                .to_compile_error()
                .into();
        }
    };

    // We're given the modulus p of the prime field
    let modulus: BigUint = fetch_attr("PrimeFieldModulus", &ast.attrs)
        .expect("Please supply a PrimeFieldModulus attribute")
        .parse()
        .expect("PrimeFieldModulus should be a number");

    // We may be provided with a generator of p - 1 order. It is required that this generator be
    // quadratic nonresidue.
    // TODO: Compute this ourselves.
    let generator: BigUint = fetch_attr("PrimeFieldGenerator", &ast.attrs)
        .expect("Please supply a PrimeFieldGenerator attribute")
        .parse()
        .expect("PrimeFieldGenerator should be a number");

    // Field element representations may be in little-endian or big-endian.
    let endianness = fetch_attr("PrimeFieldReprEndianness", &ast.attrs)
        .expect("Please supply a PrimeFieldReprEndianness attribute")
        .parse()
        .expect("PrimeFieldReprEndianness should be 'big' or 'little'");

    // The arithmetic in this library only works if the modulus*2 is smaller than the backing
    // representation. Compute the number of 64-bit limbs we need.
    let mut limbs = 1;
    {
        let mod2 = (&modulus) << 1; // modulus * 2
        let mut cur = BigUint::one() << 64; // always 64-bit limbs for now
        while cur < mod2 {
            limbs += 1;
            cur <<= 64;
        }
    }

    let bytes = modulus.bits().div_ceil(8);
    let zkvm_limbs = if bytes <= 32 {
        32
    } else if bytes <= 48 {
        48
    } else {
        // A limitation of our zkvm implementation is that we only support moduli up to 48 bytes.
        return syn::Error::new_spanned(
            ast,
            "PrimeField modulus is too large. Only 48 byte moduli are supported.",
        )
        .to_compile_error()
        .into();
    };

    // The struct we're deriving for must be a wrapper around `pub [u64; limbs]`.
    if let Some(err) = validate_struct(&ast, limbs) {
        return err.into();
    }

    // Generate the identifier for the "Repr" type we must construct.
    let repr_ident = syn::Ident::new(
        &format!("{}Repr", ast.ident),
        proc_macro2::Span::call_site(),
    );

    let mut gen = proc_macro2::TokenStream::new();

    // Remove the attributes from the struct so we can insert it back into the code
    ast.attrs.clear();

    // Call moduli_declare! to define the struct
    let openvm_struct = openvm_struct_impl(&ast, &modulus);

    // TODO: test the non-zkvm case
    gen.extend(quote! {
        #[cfg(target_os = "zkvm")]
            #openvm_struct
        #[cfg(not(target_os = "zkvm"))]
            #ast
    });

    let (constants_impl, sqrt_impl) =
        prime_field_constants_and_sqrt(&ast.ident, &modulus, limbs, zkvm_limbs, generator);

    gen.extend(constants_impl);
    gen.extend(prime_field_repr_impl(&repr_ident, &endianness, limbs * 8));
    gen.extend(prime_field_impl(
        &ast.ident,
        &repr_ident,
        &modulus,
        &endianness,
        limbs,
        zkvm_limbs,
        sqrt_impl,
    ));

    // Return the generated impl
    gen.into()
}

fn openvm_struct_impl(ast: &syn::ItemStruct, modulus: &BigUint) -> proc_macro2::TokenStream {
    let struct_ident = &ast.ident;
    let modulus_str = modulus.to_str_radix(10);
    quote! {
        ::openvm_algebra_moduli_macros::moduli_declare! {
            #struct_ident {
                modulus = #modulus_str
            }
        }
    }
}

/// Checks that `body` contains `pub [u64; limbs]`.
fn validate_struct(ast: &syn::ItemStruct, limbs: usize) -> Option<proc_macro2::TokenStream> {
    // The struct should contain a single unnamed field.
    let fields = match &ast.fields {
        syn::Fields::Unnamed(x) if x.unnamed.len() == 1 => x,
        _ => {
            return Some(
                syn::Error::new_spanned(
                    &ast.ident,
                    format!(
                        "The struct must contain an array of limbs. Change this to `{}([u64; {}])`",
                        ast.ident, limbs,
                    ),
                )
                .to_compile_error(),
            )
        }
    };
    let field = &fields.unnamed[0];

    // The field should be an array.
    let arr = match &field.ty {
        syn::Type::Array(x) => x,
        _ => {
            return Some(
                syn::Error::new_spanned(
                    field,
                    format!(
                        "The inner field must be an array of limbs. Change this to `[u64; {}]`",
                        limbs,
                    ),
                )
                .to_compile_error(),
            )
        }
    };

    // The array's element type should be `u64`.
    if match arr.elem.as_ref() {
        syn::Type::Path(path) => path.path.get_ident().map(|x| *x != "u64").unwrap_or(true),
        _ => true,
    } {
        return Some(
            syn::Error::new_spanned(
                arr,
                format!(
                    "PrimeField derive requires 64-bit limbs. Change this to `[u64; {}]",
                    limbs
                ),
            )
            .to_compile_error(),
        );
    }

    // The array's length should be a literal int equal to `limbs`.
    let expr_lit = match &arr.len {
        syn::Expr::Lit(expr_lit) => Some(&expr_lit.lit),
        syn::Expr::Group(expr_group) => match &*expr_group.expr {
            syn::Expr::Lit(expr_lit) => Some(&expr_lit.lit),
            _ => None,
        },
        _ => None,
    };
    let lit_int = match match expr_lit {
        Some(syn::Lit::Int(lit_int)) => Some(lit_int),
        _ => None,
    } {
        Some(x) => x,
        _ => {
            return Some(
                syn::Error::new_spanned(
                    arr,
                    format!("To derive PrimeField, change this to `[u64; {}]`.", limbs),
                )
                .to_compile_error(),
            )
        }
    };
    if lit_int.base10_digits() != limbs.to_string() {
        return Some(
            syn::Error::new_spanned(
                lit_int,
                format!("The given modulus requires {} limbs.", limbs),
            )
            .to_compile_error(),
        );
    }

    // The field should not be public.
    match &field.vis {
        syn::Visibility::Inherited => (),
        _ => {
            return Some(
                syn::Error::new_spanned(&field.vis, "Field must not be public.").to_compile_error(),
            )
        }
    }

    // Valid!
    None
}

/// Fetch an attribute string from the derived struct.
fn fetch_attr(name: &str, attrs: &[syn::Attribute]) -> Option<String> {
    for attr in attrs {
        if let Ok(meta) = attr.parse_meta() {
            match meta {
                syn::Meta::NameValue(nv) => {
                    if nv.path.get_ident().map(|i| i.to_string()) == Some(name.to_string()) {
                        match nv.lit {
                            syn::Lit::Str(ref s) => return Some(s.value()),
                            _ => {
                                panic!("attribute {} should be a string", name);
                            }
                        }
                    }
                }
                _ => {
                    panic!("attribute {} should be a string", name);
                }
            }
        }
    }

    None
}

// Implement the wrapped ident `repr` with `bytes` bytes.
fn prime_field_repr_impl(
    repr: &syn::Ident,
    endianness: &ReprEndianness,
    bytes: usize,
) -> proc_macro2::TokenStream {
    let repr_iter_be = endianness.iter_be();

    quote! {
        #[derive(Copy, Clone)]
        pub struct #repr(pub [u8; #bytes]);

        impl ::ff::derive::subtle::ConstantTimeEq for #repr {
            fn ct_eq(&self, other: &#repr) -> ::ff::derive::subtle::Choice {
                self.0
                    .iter()
                    .zip(other.0.iter())
                    .map(|(a, b)| a.ct_eq(b))
                    .fold(1.into(), |acc, x| acc & x)
            }
        }

        impl ::core::cmp::PartialEq for #repr {
            fn eq(&self, other: &#repr) -> bool {
                use ::ff::derive::subtle::ConstantTimeEq;
                self.ct_eq(other).into()
            }
        }

        impl ::core::cmp::Eq for #repr { }

        impl ::core::default::Default for #repr {
            fn default() -> #repr {
                #repr([0u8; #bytes])
            }
        }

        impl ::core::fmt::Debug for #repr
        {
            fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                write!(f, "0x")?;
                for i in #repr_iter_be {
                    write!(f, "{:02x}", *i)?;
                }

                Ok(())
            }
        }

        impl AsRef<[u8]> for #repr {
            #[inline(always)]
            fn as_ref(&self) -> &[u8] {
                &self.0
            }
        }

        impl AsMut<[u8]> for #repr {
            #[inline(always)]
            fn as_mut(&mut self) -> &mut [u8] {
                &mut self.0
            }
        }
    }
}

/// Convert BigUint into a vector of 64-bit limbs.
fn biguint_to_real_u64_vec(mut v: BigUint, limbs: usize) -> Vec<u64> {
    let m = BigUint::one() << 64;
    let mut ret = vec![];

    while v > BigUint::zero() {
        let limb: BigUint = &v % &m;
        ret.push(limb.to_u64().unwrap());
        v >>= 64;
    }

    while ret.len() < limbs {
        ret.push(0);
    }

    assert!(ret.len() == limbs);

    ret
}

/// Convert BigUint into a tokenized vector of 64-bit limbs.
fn biguint_to_u64_vec(v: BigUint, limbs: usize) -> proc_macro2::TokenStream {
    let ret = biguint_to_real_u64_vec(v, limbs);
    quote!([#(#ret,)*])
}

/// Returns a token stream containing a little-endian bytes representation of `v`.
fn biguint_to_u8_vec(v: BigUint, limbs: usize) -> proc_macro2::TokenStream {
    let mut bytes = v.to_bytes_le();
    while bytes.len() < limbs {
        bytes.push(0);
    }
    quote!([#(#bytes,)*])
}

fn biguint_num_bits(mut v: BigUint) -> u32 {
    let mut bits = 0;

    while v != BigUint::zero() {
        v >>= 1;
        bits += 1;
    }

    bits
}

/// BigUint modular exponentiation by square-and-multiply.
fn exp(base: BigUint, exp: &BigUint, modulus: &BigUint) -> BigUint {
    let mut ret = BigUint::one();

    for i in exp
        .to_bytes_be()
        .into_iter()
        .flat_map(|x| (0..8).rev().map(move |i| (x >> i).is_odd()))
    {
        ret = (&ret * &ret) % modulus;
        if i {
            ret = (ret * &base) % modulus;
        }
    }

    ret
}

#[test]
fn test_exp() {
    assert_eq!(
        exp(
            BigUint::from_str("4398572349857239485729348572983472345").unwrap(),
            &BigUint::from_str("5489673498567349856734895").unwrap(),
            &BigUint::from_str(
                "52435875175126190479447740508185965837690552500527637822603658699938581184513"
            )
            .unwrap()
        ),
        BigUint::from_str(
            "4371221214068404307866768905142520595925044802278091865033317963560480051536"
        )
        .unwrap()
    );
}

fn prime_field_constants_and_sqrt(
    name: &syn::Ident,
    modulus: &BigUint,
    limbs: usize,
    zkvm_limbs: usize,
    generator: BigUint,
) -> (proc_macro2::TokenStream, proc_macro2::TokenStream) {
    let bytes = limbs * 8;
    let modulus_num_bits = biguint_num_bits(modulus.clone());

    // The number of bits we should "shave" from a randomly sampled representation, i.e.,
    // if our modulus is 381 bits and our representation is 384 bits, we should shave
    // 3 bits from the beginning of a randomly sampled 384 bit representation to
    // reduce the cost of rejection sampling.
    let repr_shave_bits = (64 * limbs as u32) - biguint_num_bits(modulus.clone());

    // Compute R = 2**(64 * limbs) mod m
    let r = (BigUint::one() << (limbs * 64)) % modulus;
    let to_mont = |v| (v * &r) % modulus;

    let two = BigUint::from_str("2").unwrap();
    let p_minus_2 = modulus - &two;
    let invert = |v| exp(v, &p_minus_2, modulus);

    // 2^-1 mod m
    let two_inv = biguint_to_u64_vec(to_mont(invert(two.clone())), limbs);
    let two_inv_bytes = biguint_to_u8_vec(invert(two), zkvm_limbs);

    // modulus - 1 = 2^s * t
    let mut s: u32 = 0;
    let mut t = modulus - BigUint::from_str("1").unwrap();
    while t.is_even() {
        t >>= 1;
        s += 1;
    }

    // Compute 2^s root of unity given the generator
    let root_of_unity_biguint = exp(generator.clone(), &t, modulus);

    let root_of_unity_inv_biguint = invert(root_of_unity_biguint.clone());
    let root_of_unity_inv = biguint_to_u64_vec(to_mont(root_of_unity_inv_biguint.clone()), limbs);
    let root_of_unity_inv_bytes = biguint_to_u8_vec(root_of_unity_inv_biguint, zkvm_limbs);

    let root_of_unity = biguint_to_u64_vec(to_mont(root_of_unity_biguint.clone()), limbs);
    let root_of_unity_bytes = biguint_to_u8_vec(root_of_unity_biguint, zkvm_limbs);

    let delta_biguint = exp(generator.clone(), &(BigUint::one() << s), modulus);
    let delta = biguint_to_u64_vec(to_mont(delta_biguint.clone()), limbs);
    let delta_bytes = biguint_to_u8_vec(delta_biguint, zkvm_limbs);

    let generator_u64_limbs = biguint_to_u64_vec(to_mont(generator.clone()), limbs);
    let generator_bytes = biguint_to_u8_vec(generator, zkvm_limbs);

    let sqrt_impl =
        if (modulus % BigUint::from_str("4").unwrap()) == BigUint::from_str("3").unwrap() {
            // Addition chain for (r + 1) // 4
            let mod_plus_1_over_4 = pow_fixed::generate(
                &quote! {self},
                (modulus + BigUint::from_str("1").unwrap()) >> 2,
            );

            quote! {
                use ::ff::derive::subtle::ConstantTimeEq;

                // Because r = 3 (mod 4)
                // sqrt can be done with only one exponentiation,
                // via the computation of  self^((r + 1) // 4) (mod r)
                let sqrt = {
                    #mod_plus_1_over_4
                };

                ::ff::derive::subtle::CtOption::new(
                    sqrt,
                    (sqrt * &sqrt).ct_eq(self), // Only return Some if it's the square root.
                )
            }
        } else {
            // Addition chain for (t - 1) // 2
            let t_minus_1_over_2 = if t == BigUint::one() {
                quote!( #name::ONE )
            } else {
                pow_fixed::generate(&quote! {self}, (&t - BigUint::one()) >> 1)
            };

            quote! {
                // Tonelli-Shanks algorithm works for every remaining odd prime.
                // https://eprint.iacr.org/2012/685.pdf (page 12, algorithm 5)
                use ::ff::derive::subtle::{ConditionallySelectable, ConstantTimeEq};

                // w = self^((t - 1) // 2)
                let w = {
                    #t_minus_1_over_2
                };

                let mut v = S;
                let mut x = *self * &w;
                let mut b = x * &w;

                // Initialize z as the 2^S root of unity.
                let mut z = ROOT_OF_UNITY;

                for max_v in (1..=S).rev() {
                    let mut k = 1;
                    let mut tmp = b.square();
                    let mut j_less_than_v: ::ff::derive::subtle::Choice = 1.into();

                    for j in 2..max_v {
                        let tmp_is_one = tmp.ct_eq(&#name::ONE);
                        let squared = #name::conditional_select(&tmp, &z, tmp_is_one).square();
                        tmp = #name::conditional_select(&squared, &tmp, tmp_is_one);
                        let new_z = #name::conditional_select(&z, &squared, tmp_is_one);
                        j_less_than_v &= !j.ct_eq(&v);
                        k = u32::conditional_select(&j, &k, tmp_is_one);
                        z = #name::conditional_select(&z, &new_z, j_less_than_v);
                    }

                    let result = x * &z;
                    x = #name::conditional_select(&result, &x, b.ct_eq(&#name::ONE));
                    z = z.square();
                    b *= &z;
                    v = k;
                }

                ::ff::derive::subtle::CtOption::new(
                    x,
                    (x * &x).ct_eq(self), // Only return Some if it's the square root.
                )
            }
        };

    // Compute R^2 mod m
    let r2 = biguint_to_u64_vec((&r * &r) % modulus, limbs);

    let r = biguint_to_u64_vec(r, limbs);
    let modulus_le_bytes = ReprEndianness::Little.modulus_repr(modulus, limbs * 8);
    let modulus_str = format!("0x{}", modulus.to_str_radix(16));
    let modulus = biguint_to_real_u64_vec(modulus.clone(), limbs);

    // Compute -m^-1 mod 2**64 by exponentiating by totient(2**64) - 1
    let mut inv = 1u64;
    for _ in 0..63 {
        inv = inv.wrapping_mul(inv);
        inv = inv.wrapping_mul(modulus[0]);
    }
    inv = inv.wrapping_neg();

    (
        quote! {
            type REPR_BYTES = [u8; #bytes];
            type REPR_BITS = REPR_BYTES;

            /// This is the modulus m of the prime field
            const MODULUS: REPR_BITS = [#(#modulus_le_bytes,)*];

            /// This is the modulus m of the prime field in limb form
            #[cfg(not(target_os = "zkvm"))]
            const MODULUS_LIMBS: #name = #name([#(#modulus,)*]);

            /// This is the modulus m of the prime field in hex string form
            const MODULUS_STR: &'static str = #modulus_str;

            /// The number of bits needed to represent the modulus.
            const MODULUS_BITS: u32 = #modulus_num_bits;

            /// The number of bits that must be shaved from the beginning of
            /// the representation when randomly sampling.
            const REPR_SHAVE_BITS: u32 = #repr_shave_bits;

            /// 2^{limbs*64} mod m
            #[cfg(not(target_os = "zkvm"))]
            const R: #name = #name(#r);

            /// 2^{limbs*64*2} mod m
            #[cfg(not(target_os = "zkvm"))]
            const R2: #name = #name(#r2);

            /// -(m^{-1} mod m) mod m
            #[cfg(not(target_os = "zkvm"))]
            const INV: u64 = #inv;

            /// 2^{-1} mod m
            #[cfg(target_os = "zkvm")]
            const TWO_INV: #name = <#name>::from_const_bytes(#two_inv_bytes);
            #[cfg(not(target_os = "zkvm"))]
            const TWO_INV: #name = #name(#two_inv);

            /// Multiplicative generator of `MODULUS` - 1 order, also quadratic
            /// nonresidue.
            #[cfg(target_os = "zkvm")]
            const GENERATOR: #name = <#name>::from_const_bytes(#generator_bytes);
            #[cfg(not(target_os = "zkvm"))]
            const GENERATOR: #name = #name(#generator_u64_limbs);

            /// 2^s * t = MODULUS - 1 with t odd
            const S: u32 = #s;

            /// 2^s root of unity computed by GENERATOR^t
            #[cfg(target_os = "zkvm")]
            const ROOT_OF_UNITY: #name = <#name>::from_const_bytes(#root_of_unity_bytes);
            #[cfg(not(target_os = "zkvm"))]
            const ROOT_OF_UNITY: #name = #name(#root_of_unity);

            /// (2^s)^{-1} mod m
            #[cfg(target_os = "zkvm")]
            const ROOT_OF_UNITY_INV: #name = <#name>::from_const_bytes(#root_of_unity_inv_bytes);
            #[cfg(not(target_os = "zkvm"))]
            const ROOT_OF_UNITY_INV: #name = #name(#root_of_unity_inv);

            /// GENERATOR^{2^s}
            #[cfg(target_os = "zkvm")]
            const DELTA: #name = <#name>::from_const_bytes(#delta_bytes);
            #[cfg(not(target_os = "zkvm"))]
            const DELTA: #name = #name(#delta);
        },
        sqrt_impl,
    )
}

/// Implement PrimeField for the derived type.
fn prime_field_impl(
    name: &syn::Ident,
    repr: &syn::Ident,
    modulus: &BigUint,
    endianness: &ReprEndianness,
    limbs: usize,
    zkvm_limbs: usize,
    sqrt_impl: proc_macro2::TokenStream,
) -> proc_macro2::TokenStream {
    // Returns r{n} as an ident.
    fn get_temp(n: usize) -> syn::Ident {
        syn::Ident::new(&format!("r{}", n), proc_macro2::Span::call_site())
    }

    // The parameter list for the mont_reduce() internal method.
    // r0: u64, mut r1: u64, mut r2: u64, ...
    let mut mont_paramlist = proc_macro2::TokenStream::new();
    mont_paramlist.append_separated(
        (0..(limbs * 2)).map(|i| (i, get_temp(i))).map(|(i, x)| {
            if i != 0 {
                quote! {mut #x: u64}
            } else {
                quote! {#x: u64}
            }
        }),
        proc_macro2::Punct::new(',', proc_macro2::Spacing::Alone),
    );

    // Implement montgomery reduction for some number of limbs
    fn mont_impl(limbs: usize) -> proc_macro2::TokenStream {
        #[cfg(target_os = "zkvm")]
        {
            quote! {
                unimplemented!();
            }
        }
        #[cfg(not(target_os = "zkvm"))]
        {
            let mut gen = proc_macro2::TokenStream::new();

            for i in 0..limbs {
                {
                    let temp = get_temp(i);
                    gen.extend(quote! {
                        let k = #temp.wrapping_mul(INV);
                        let (_, carry) = ::ff::derive::mac(#temp, k, MODULUS_LIMBS.0[0], 0);
                    });
                }

                for j in 1..limbs {
                    let temp = get_temp(i + j);
                    gen.extend(quote! {
                    let (#temp, carry) = ::ff::derive::mac(#temp, k, MODULUS_LIMBS.0[#j], carry);
                });
                }

                let temp = get_temp(i + limbs);

                if i == 0 {
                    gen.extend(quote! {
                        let (#temp, carry2) = ::ff::derive::adc(#temp, 0, carry);
                    });
                } else {
                    gen.extend(quote! {
                        let (#temp, carry2) = ::ff::derive::adc(#temp, carry2, carry);
                    });
                }
            }

            for i in 0..limbs {
                let temp = get_temp(limbs + i);

                gen.extend(quote! {
                    self.0[#i] = #temp;
                });
            }

            gen
        }
    }

    fn sqr_impl(a: proc_macro2::TokenStream, limbs: usize) -> proc_macro2::TokenStream {
        let mut gen = proc_macro2::TokenStream::new();

        if limbs > 1 {
            for i in 0..(limbs - 1) {
                gen.extend(quote! {
                    let carry = 0;
                });

                for j in (i + 1)..limbs {
                    let temp = get_temp(i + j);
                    if i == 0 {
                        gen.extend(quote! {
                            let (#temp, carry) = ::ff::derive::mac(0, #a.0[#i], #a.0[#j], carry);
                        });
                    } else {
                        gen.extend(quote! {
                            let (#temp, carry) = ::ff::derive::mac(#temp, #a.0[#i], #a.0[#j], carry);
                        });
                    }
                }

                let temp = get_temp(i + limbs);

                gen.extend(quote! {
                    let #temp = carry;
                });
            }

            for i in 1..(limbs * 2) {
                let temp0 = get_temp(limbs * 2 - i);
                let temp1 = get_temp(limbs * 2 - i - 1);

                if i == 1 {
                    gen.extend(quote! {
                        let #temp0 = #temp1 >> 63;
                    });
                } else if i == (limbs * 2 - 1) {
                    gen.extend(quote! {
                        let #temp0 = #temp0 << 1;
                    });
                } else {
                    gen.extend(quote! {
                        let #temp0 = (#temp0 << 1) | (#temp1 >> 63);
                    });
                }
            }
        } else {
            let temp1 = get_temp(1);
            gen.extend(quote! {
                let #temp1 = 0;
            });
        }

        for i in 0..limbs {
            let temp0 = get_temp(i * 2);
            let temp1 = get_temp(i * 2 + 1);
            if i == 0 {
                gen.extend(quote! {
                    let (#temp0, carry) = ::ff::derive::mac(0, #a.0[#i], #a.0[#i], 0);
                });
            } else {
                gen.extend(quote! {
                    let (#temp0, carry) = ::ff::derive::mac(#temp0, #a.0[#i], #a.0[#i], carry);
                });
            }

            gen.extend(quote! {
                let (#temp1, carry) = ::ff::derive::adc(#temp1, 0, carry);
            });
        }

        let mut mont_calling = proc_macro2::TokenStream::new();
        mont_calling.append_separated(
            (0..(limbs * 2)).map(get_temp),
            proc_macro2::Punct::new(',', proc_macro2::Spacing::Alone),
        );

        gen.extend(quote! {
            let mut ret = *self;
            ret.mont_reduce(#mont_calling);
            ret
        });

        gen
    }

    fn mul_impl(
        a: proc_macro2::TokenStream,
        b: proc_macro2::TokenStream,
        limbs: usize,
    ) -> proc_macro2::TokenStream {
        let mut gen = proc_macro2::TokenStream::new();

        for i in 0..limbs {
            gen.extend(quote! {
                let carry = 0;
            });

            for j in 0..limbs {
                let temp = get_temp(i + j);

                if i == 0 {
                    gen.extend(quote! {
                        let (#temp, carry) = ::ff::derive::mac(0, #a.0[#i], #b.0[#j], carry);
                    });
                } else {
                    gen.extend(quote! {
                        let (#temp, carry) = ::ff::derive::mac(#temp, #a.0[#i], #b.0[#j], carry);
                    });
                }
            }

            let temp = get_temp(i + limbs);

            gen.extend(quote! {
                let #temp = carry;
            });
        }

        let mut mont_calling = proc_macro2::TokenStream::new();
        mont_calling.append_separated(
            (0..(limbs * 2)).map(get_temp),
            proc_macro2::Punct::new(',', proc_macro2::Spacing::Alone),
        );

        gen.extend(quote! {
            self.mont_reduce(#mont_calling);
        });

        gen
    }

    /// Generates an implementation of multiplicative inversion within the target prime
    /// field.
    fn inv_impl(a: proc_macro2::TokenStream, modulus: &BigUint) -> proc_macro2::TokenStream {
        // Addition chain for p - 2
        let mod_minus_2 = pow_fixed::generate(&a, modulus - BigUint::from(2u64));

        quote! {
            use ::ff::derive::subtle::ConstantTimeEq;

            // By Euler's theorem, if `a` is coprime to `p` (i.e. `gcd(a, p) = 1`), then:
            //     a^-1 ≡ a^(phi(p) - 1) mod p
            //
            // `ff_derive` requires that `p` is prime; in this case, `phi(p) = p - 1`, and
            // thus:
            //     a^-1 ≡ a^(p - 2) mod p
            let inv = {
                #mod_minus_2
            };

            ::ff::derive::subtle::CtOption::new(inv, !#a.is_zero())
        }
    }

    let squaring_impl = sqr_impl(quote! {self}, limbs);
    let multiply_impl = mul_impl(quote! {self}, quote! {other}, limbs);
    let invert_impl = inv_impl(quote! {self}, modulus);
    let montgomery_impl = mont_impl(limbs);

    fn mont_reduce_params(a: proc_macro2::TokenStream, limbs: usize) -> proc_macro2::TokenStream {
        // a.0[0], a.0[1], ..., 0, 0, 0, 0, ...
        let mut mont_reduce_params = proc_macro2::TokenStream::new();
        mont_reduce_params.append_separated(
            (0..limbs)
                .map(|i| quote! { #a.0[#i] })
                .chain((0..limbs).map(|_| quote! {0})),
            proc_macro2::Punct::new(',', proc_macro2::Spacing::Alone),
        );
        mont_reduce_params
    }

    let mont_reduce_self_params = mont_reduce_params(quote! {self}, limbs);
    let mont_reduce_other_params = mont_reduce_params(quote! {other}, limbs);

    let from_repr_impl = endianness.from_repr(name, limbs);
    let to_repr_impl = endianness.to_repr(quote! {#repr}, &mont_reduce_self_params, limbs);

    let prime_field_bits_impl = if cfg!(feature = "bits") {
        let to_le_bits_impl = ReprEndianness::Little.to_repr(
            quote! {::ff::derive::bitvec::array::BitArray::new},
            &mont_reduce_self_params,
            limbs,
        );

        Some(quote! {
            impl ::ff::PrimeFieldBits for #name {
                type ReprBits = REPR_BITS;

                fn to_le_bits(&self) -> ::ff::FieldBits<REPR_BITS> {
                    #to_le_bits_impl
                }

                fn char_le_bits() -> ::ff::FieldBits<REPR_BITS> {
                    ::ff::FieldBits::new(MODULUS)
                }
            }
        })
    } else {
        None
    };

    let top_limb_index = limbs - 1;

    // Since moduli_declare! already implements some traits, we need to conditionally
    // compile some of the trait impls depending on whether we're in zkvm or not.
    // So, we create a new module with #[cfg(not(target_os = "zkvm"))] and place the impls in there.
    let impl_module_ident =
        syn::Ident::new(&format!("impl_{}", name), proc_macro2::Span::call_site());

    let zero_impl = quote! {
        #[cfg(target_os = "zkvm")]
            const ZERO: Self = <Self as ::openvm_algebra_guest::IntMod>::ZERO;
        #[cfg(not(target_os = "zkvm"))]
            const ZERO: Self = #name([0; #limbs]);
    };
    let one_impl = quote! {
        #[cfg(target_os = "zkvm")]
            const ONE: Self = <Self as ::openvm_algebra_guest::IntMod>::ONE;
        #[cfg(not(target_os = "zkvm"))]
            const ONE: Self = R;
    };

    quote! {
        impl ::core::marker::Copy for #name { }

        impl ::core::default::Default for #name {
            fn default() -> #name {
                use ::ff::Field;
                #name::ZERO
            }
        }

        impl ::ff::derive::subtle::ConstantTimeEq for #name {
            fn ct_eq(&self, other: &#name) -> ::ff::derive::subtle::Choice {
                use ::ff::PrimeField;
                self.to_repr().ct_eq(&other.to_repr())
            }
        }

        /// Elements are ordered lexicographically.
        impl Ord for #name {
            #[inline(always)]
            fn cmp(&self, other: &#name) -> ::core::cmp::Ordering {
                #[cfg(target_os = "zkvm")]
                {
                    <Self as ::openvm_algebra_guest::IntMod>::assert_reduced(self);
                    <Self as ::openvm_algebra_guest::IntMod>::assert_reduced(other);

                    self.cmp_native(other)
                }
                #[cfg(not(target_os = "zkvm"))]
                {
                    let mut a = *self;
                    a.mont_reduce(
                        #mont_reduce_self_params
                    );

                    let mut b = *other;
                    b.mont_reduce(
                        #mont_reduce_other_params
                    );

                    a.cmp_native(&b)
                }
            }
        }

        impl PartialOrd for #name {
            #[inline(always)]
            fn partial_cmp(&self, other: &#name) -> Option<::core::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }

        impl From<u64> for #name {
            #[inline(always)]
            fn from(val: u64) -> #name {
                #[cfg(target_os = "zkvm")]
                {
                    <#name as ::openvm_algebra_guest::IntMod>::from_u64(val)
                }
                #[cfg(not(target_os = "zkvm"))]
                {
                    let mut raw = [0u64; #limbs];
                    raw[0] = val;
                    #name(raw) * R2
                }
            }
        }

        impl From<#name> for #repr {
            fn from(e: #name) -> #repr {
                use ::ff::PrimeField;
                e.to_repr()
            }
        }

        impl<'a> From<&'a #name> for #repr {
            fn from(e: &'a #name) -> #repr {
                use ::ff::PrimeField;
                e.to_repr()
            }
        }

        impl ::ff::derive::subtle::ConditionallySelectable for #name {
            fn conditional_select(a: &#name, b: &#name, choice: ::ff::derive::subtle::Choice) -> #name {
                #[cfg(target_os = "zkvm")]
                {
                    let mut res = [0u8; #zkvm_limbs];
                    let a_le_bytes = <Self as ::openvm_algebra_guest::IntMod>::as_le_bytes(a);
                    let b_le_bytes = <Self as ::openvm_algebra_guest::IntMod>::as_le_bytes(b);
                    for i in 0..#zkvm_limbs {
                        res[i] = u8::conditional_select(&a_le_bytes[i], &b_le_bytes[i], choice);
                    }
                    #name(res)
                }
                #[cfg(not(target_os = "zkvm"))]
                {
                    let mut res = [0u64; #limbs];
                    for i in 0..#limbs {
                        res[i] = u64::conditional_select(&a.0[i], &b.0[i], choice);
                    }
                    #name(res)
                }
            }
        }

        // All the traits that are implemented in this module are already implemented
        // on our zkvm-compatible struct, so we need to conditionally implement them
        #[cfg(not(target_os = "zkvm"))]
        mod #impl_module_ident {
            use super::{#name, MODULUS_LIMBS};

            impl ::core::clone::Clone for #name {
                fn clone(&self) -> #name {
                    *self
                }
            }

            impl ::core::cmp::PartialEq for #name {
                fn eq(&self, other: &#name) -> bool {
                    use ::ff::derive::subtle::ConstantTimeEq;
                    self.ct_eq(other).into()
                }
            }

            impl ::core::cmp::Eq for #name { }

            impl ::core::fmt::Debug for #name
            {
                fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                    use ::ff::PrimeField;
                    write!(f, "{}({:?})", stringify!(#name), self.to_repr())
                }
            }


            impl ::core::ops::Neg for #name {
                type Output = #name;

                #[inline]
                fn neg(self) -> #name {
                    use ::ff::Field;

                    let mut ret = self;
                    if !ret.is_zero_vartime() {
                        let mut tmp = MODULUS_LIMBS;
                        tmp.sub_noborrow(&ret);
                        ret = tmp;
                    }
                    ret
                }
            }

            impl<'r> ::core::ops::Add<&'r #name> for #name {
                type Output = #name;

                #[inline]
                fn add(self, other: &#name) -> #name {
                    use ::core::ops::AddAssign;

                    let mut ret = self;
                    ret.add_assign(other);
                    ret
                }
            }

            impl ::core::ops::Add for #name {
                type Output = #name;

                #[inline]
                fn add(self, other: #name) -> Self {
                    self + &other
                }
            }

            impl<'r> ::core::ops::AddAssign<&'r #name> for #name {
                #[inline]
                fn add_assign(&mut self, other: &#name) {
                    // This cannot exceed the backing capacity.
                    self.add_nocarry(other);

                    // However, it may need to be reduced.
                    self.reduce();
                }
            }

            impl ::core::ops::AddAssign for #name {
                #[inline]
                fn add_assign(&mut self, other: #name) {
                    self.add_assign(&other);
                }
            }

            impl<'r> ::core::ops::Sub<&'r #name> for #name {
                type Output = #name;

                #[inline]
                fn sub(self, other: &#name) -> Self {
                    use ::core::ops::SubAssign;

                    let mut ret = self;
                    ret.sub_assign(other);
                    ret
                }
            }

            impl ::core::ops::Sub for #name {
                type Output = #name;

                #[inline]
                fn sub(self, other: #name) -> Self {
                    self - &other
                }
            }

            impl<'r> ::core::ops::SubAssign<&'r #name> for #name {
                #[inline]
                fn sub_assign(&mut self, other: &#name) {
                    // If `other` is larger than `self`, we'll need to add the modulus to self first.
                    if other.cmp_native(self) == ::core::cmp::Ordering::Greater {
                        self.add_nocarry(&MODULUS_LIMBS);
                    }

                    self.sub_noborrow(other);
                }
            }

            impl ::core::ops::SubAssign for #name {
                #[inline]
                fn sub_assign(&mut self, other: #name) {
                    self.sub_assign(&other);
                }
            }

            impl<'r> ::core::ops::Mul<&'r #name> for #name {
                type Output = #name;

                #[inline]
                fn mul(self, other: &#name) -> Self {
                    use ::core::ops::MulAssign;

                    let mut ret = self;
                    ret.mul_assign(other);
                    ret
                }
            }

            impl ::core::ops::Mul for #name {
                type Output = #name;

                #[inline]
                fn mul(self, other: #name) -> Self {
                    self * &other
                }
            }

            impl<'r> ::core::ops::MulAssign<&'r #name> for #name {
                #[inline]
                fn mul_assign(&mut self, other: &#name)
                {
                    #multiply_impl
                }
            }

            impl ::core::ops::MulAssign for #name {
                #[inline]
                fn mul_assign(&mut self, other: #name)
                {
                    self.mul_assign(&other);
                }
            }

            impl<T: ::core::borrow::Borrow<#name>> ::core::iter::Sum<T> for #name {
                fn sum<I: Iterator<Item = T>>(iter: I) -> Self {
                    use ::ff::Field;

                    iter.fold(Self::ZERO, |acc, item| acc + item.borrow())
                }
            }

            impl<T: ::core::borrow::Borrow<#name>> ::core::iter::Product<T> for #name {
                fn product<I: Iterator<Item = T>>(iter: I) -> Self {
                    use ::ff::Field;

                    iter.fold(Self::ONE, |acc, item| acc * item.borrow())
                }
            }
        }

        impl ::ff::PrimeField for #name {
            type Repr = #repr;

            fn from_repr(r: #repr) -> ::ff::derive::subtle::CtOption<#name> {
                #from_repr_impl

                #[cfg(target_os = "zkvm")]
                {
                    ::ff::derive::subtle::CtOption::new(r, r.constant_time_is_reduced())
                }
                #[cfg(not(target_os = "zkvm"))]
                {
                    // Try to subtract the modulus
                    let borrow = r.0.iter().zip(MODULUS_LIMBS.0.iter()).fold(0, |borrow, (a, b)| {
                        ::ff::derive::sbb(*a, *b, borrow).1
                    });

                    // If the element is smaller than MODULUS then the
                    // subtraction will underflow, producing a borrow value
                    // of 0xffff...ffff. Otherwise, it'll be zero.
                    let is_some = ::ff::derive::subtle::Choice::from((borrow as u8) & 1);

                    // Convert to Montgomery form by computing
                    // (a.R^0 * R^2) / R = a.R
                    ::ff::derive::subtle::CtOption::new(r * &R2, is_some)
                }
            }

            fn from_repr_vartime(r: #repr) -> Option<#name> {
                #from_repr_impl

                #[cfg(target_os = "zkvm")]
                {
                    if <Self as ::openvm_algebra_guest::IntMod>::is_reduced(&r) {
                        Some(r)
                    } else {
                        None
                    }
                }
                #[cfg(not(target_os = "zkvm"))]
                {
                    if r.is_valid() {
                        Some(r * R2)
                    } else {
                        None
                    }

                }
            }

            fn to_repr(&self) -> #repr {
                #to_repr_impl
            }

            #[inline(always)]
            fn is_odd(&self) -> ::ff::derive::subtle::Choice {
                #[cfg(target_os = "zkvm")]
                {
                    ::ff::derive::subtle::Choice::from((<Self as ::openvm_algebra_guest::IntMod>::as_le_bytes(self)[0] & 1) as u8)
                }
                #[cfg(not(target_os = "zkvm"))]
                {
                    let mut r = *self;
                    r.mont_reduce(
                        #mont_reduce_self_params
                    );

                    // TODO: This looks like a constant-time result, but r.mont_reduce() is
                    // currently implemented using variable-time code.
                    ::ff::derive::subtle::Choice::from((r.0[0] & 1) as u8)
                }
            }

            const MODULUS: &'static str = MODULUS_STR;

            const NUM_BITS: u32 = MODULUS_BITS;

            const CAPACITY: u32 = Self::NUM_BITS - 1;

            const TWO_INV: Self = TWO_INV;

            const MULTIPLICATIVE_GENERATOR: Self = GENERATOR;

            const S: u32 = S;

            const ROOT_OF_UNITY: Self = ROOT_OF_UNITY;

            const ROOT_OF_UNITY_INV: Self = ROOT_OF_UNITY_INV;

            const DELTA: Self = DELTA;
        }

        #prime_field_bits_impl

        impl ::ff::Field for #name {
            #zero_impl
            #one_impl

            /// Computes a uniformly random element using rejection sampling.
            fn random(mut rng: impl ::ff::derive::rand_core::RngCore) -> Self {
                #[cfg(target_os = "zkvm")]
                {
                    panic!("randomn is not implemented for the zkvm");
                }
                #[cfg(not(target_os = "zkvm"))]
                {
                    loop {
                        let mut tmp = {
                            let mut repr = [0u64; #limbs];
                            for i in 0..#limbs {
                                repr[i] = rng.next_u64();
                            }
                            #name(repr)
                        };

                        // Mask away the unused most-significant bits.
                        // Note: In some edge cases, `REPR_SHAVE_BITS` could be 64, in which case
                        // `0xfff... >> REPR_SHAVE_BITS` overflows. So use `checked_shr` instead.
                        // This is always sufficient because we will have at most one spare limb
                        // to accommodate values of up to twice the modulus.
                        tmp.0[#top_limb_index] &= 0xffffffffffffffffu64.checked_shr(REPR_SHAVE_BITS).unwrap_or(0);

                        if tmp.is_valid() {
                            return tmp
                        }
                    }
                }
            }

            #[inline]
            fn is_zero_vartime(&self) -> bool {
                #[cfg(target_os = "zkvm")]
                {
                    self == &Self::ZERO
                }
                #[cfg(not(target_os = "zkvm"))]
                {
                    self.0.iter().all(|&e| e == 0)
                }
            }

            #[inline]
            fn double(&self) -> Self {
                #[cfg(target_os = "zkvm")]
                {
                    <Self as ::openvm_algebra_guest::IntMod>::double(self)
                }
                #[cfg(not(target_os = "zkvm"))]
                {
                    let mut ret = *self;

                    // This cannot exceed the backing capacity.
                    let mut last = 0;
                    for i in &mut ret.0 {
                        let tmp = *i >> 63;
                        *i <<= 1;
                        *i |= last;
                        last = tmp;
                    }

                    // However, it may need to be reduced.
                    ret.reduce();

                    ret
                }
            }

            /// Note that invert is not constant-time in the zkvm.
            fn invert(&self) -> ::ff::derive::subtle::CtOption<Self> {
                #[cfg(target_os = "zkvm")]
                {
                    let is_self_zero = self.is_zero_vartime();
                    let res = if is_self_zero {
                        <Self as ::openvm_algebra_guest::IntMod>::ZERO
                    } else {
                        use ::openvm_algebra_guest::DivUnsafe;
                        <Self as ::openvm_algebra_guest::IntMod>::ONE.div_unsafe(self)
                    };
                    ::ff::derive::subtle::CtOption::new(res, (!is_self_zero as u8).into())
                }
                #[cfg(not(target_os = "zkvm"))]
                {
                    #invert_impl
                }
            }

            #[inline]
            fn square(&self) -> Self
            {
                #[cfg(target_os = "zkvm")]
                {
                    <Self as ::openvm_algebra_guest::IntMod>::square(self)
                }
                #[cfg(not(target_os = "zkvm"))]
                {
                    #squaring_impl
                }
            }

            fn sqrt_ratio(num: &Self, div: &Self) -> (::ff::derive::subtle::Choice, Self) {
                ::ff::helpers::sqrt_ratio_generic(num, div)
            }

            /// Note that sqrt is not constant-time in the zkvm
            fn sqrt(&self) -> ::ff::derive::subtle::CtOption<Self> {
                #[cfg(target_os = "zkvm")]
                {
                    use ::openvm_algebra_guest::Sqrt;
                    match Sqrt::sqrt(self) {
                        Some(sqrt) => ::ff::derive::subtle::CtOption::new(sqrt, 1u8.into()),
                        None => ::ff::derive::subtle::CtOption::new(Self::ZERO, 0u8.into()),
                    }
                }
                #[cfg(not(target_os = "zkvm"))]
                {
                    #sqrt_impl
                }
            }
        }

        impl #name {
            /// Compares two elements in native representation. This is only used
            /// internally.
            #[inline(always)]
            fn cmp_native(&self, other: &#name) -> ::core::cmp::Ordering {
                for (a, b) in self.0.iter().rev().zip(other.0.iter().rev()) {
                    if a < b {
                        return ::core::cmp::Ordering::Less
                    } else if a > b {
                        return ::core::cmp::Ordering::Greater
                    }
                }

                ::core::cmp::Ordering::Equal
            }

            /// Determines if the element is really in the field. This is only used
            /// internally.
            #[inline(always)]
            #[cfg(not(target_os = "zkvm"))]
            fn is_valid(&self) -> bool {
                // The Ord impl calls `reduce`, which in turn calls `is_valid`, so we use
                // this internal function to eliminate the cycle.
                self.cmp_native(&MODULUS_LIMBS) == ::core::cmp::Ordering::Less
            }

            #[inline(always)]
            #[cfg(not(target_os = "zkvm"))]
            fn add_nocarry(&mut self, other: &#name) {
                let mut carry = 0;

                for (a, b) in self.0.iter_mut().zip(other.0.iter()) {
                    let (new_a, new_carry) = ::ff::derive::adc(*a, *b, carry);
                    *a = new_a;
                    carry = new_carry;
                }
            }

            #[inline(always)]
            #[cfg(not(target_os = "zkvm"))]
            fn sub_noborrow(&mut self, other: &#name) {
                let mut borrow = 0;

                for (a, b) in self.0.iter_mut().zip(other.0.iter()) {
                    let (new_a, new_borrow) = ::ff::derive::sbb(*a, *b, borrow);
                    *a = new_a;
                    borrow = new_borrow;
                }
            }

            /// Subtracts the modulus from this element if this element is not in the
            /// field. Only used internally.
            #[inline(always)]
            #[cfg(not(target_os = "zkvm"))]
            fn reduce(&mut self) {
                if !self.is_valid() {
                    self.sub_noborrow(&MODULUS_LIMBS);
                }
            }

            #[allow(clippy::too_many_arguments)]
            #[inline(always)]
            #[cfg(not(target_os = "zkvm"))]
            fn mont_reduce(
                &mut self,
                #mont_paramlist
            )
            {
                // The Montgomery reduction here is based on Algorithm 14.32 in
                // Handbook of Applied Cryptography
                // <http://cacr.uwaterloo.ca/hac/about/chap14.pdf>.

                #montgomery_impl

                self.reduce();
            }

            // A variant of IntMod::is_reduced that runs in constant time
            #[cfg(target_os = "zkvm")]
            fn constant_time_is_reduced(&self) -> ::ff::derive::subtle::Choice {
                let mut is_less = 0u8.into();
                // Iterate over limbs in little endian order and retain the result of the last non-equal comparison.
                for (x_limb, p_limb) in self.0.iter().zip(<Self as ::openvm_algebra_guest::IntMod>::MODULUS.iter()) {
                    if x_limb < p_limb {
                        is_less = 1u8.into();
                    } else if x_limb > p_limb {
                        is_less = 0u8.into();
                    }
                }
                // If all limbs are equal, is_less is false
                is_less
            }
        }
    }
}
