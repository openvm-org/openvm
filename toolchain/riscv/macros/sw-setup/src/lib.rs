#![feature(proc_macro_diagnostic)]

extern crate proc_macro;

use axvm_macros_common::Stmts;
use proc_macro::TokenStream;
use syn::{parse_macro_input, Expr, ExprPath, Ident, Stmt};

// Returns (ec_struct_name, intmod_type, x, y)
fn parse_input(expr: &Expr, span: proc_macro::Span) -> Result<(Ident, ExprPath), &str> {
    let assign = if let syn::Expr::Assign(assign) = expr {
        assign
    } else {
        return Err("Expected an assignment expression");
    };
    let struct_name = if let syn::Expr::Path(path) = &*assign.left {
        let struct_name = path
            .path
            .get_ident()
            .expect("Left hand side must be an identifier")
            .to_string();
        syn::Ident::new(&struct_name, span.into())
    } else {
        return Err("Left hand side must be an identifier");
    };

    let intmod_type = if let syn::Expr::Path(path) = &*assign.right {
        path
    } else {
        return Err("Right hand side must be an identifier");
    };

    Ok((struct_name, intmod_type.clone()))
}

/// This macro generates the code to setup the elliptic curve for a given modular type. Also it places the curve parameters into a special static variable to be later extracted from the ELF and used by the VM.
/// Usage:
/// ```
/// sw_setup! {
///     [TODO]
/// }
/// ```
/// This [TODO add description].
#[proc_macro]
pub fn sw_setup(input: TokenStream) -> TokenStream {
    let Stmts { stmts } = parse_macro_input!(input as Stmts);

    let mut output = Vec::new();
    let mut ec_idx = 0usize;

    let span = proc_macro::Span::call_site();

    for stmt in stmts {
        let result: Result<TokenStream, &str> = match stmt.clone() {
            Stmt::Expr(expr, _) => {
                let (struct_name, intmod_type) = parse_input(&expr, span).unwrap();
                let result = TokenStream::from(quote::quote_spanned! { span.into() =>

                    #[derive(Eq, PartialEq, Clone, Debug)]
                    #[repr(C)]
                    pub struct #struct_name {
                        pub x: #intmod_type,
                        pub y: #intmod_type,
                    }

                    impl #struct_name {
                        pub const EC_IDX: usize = #ec_idx;

                        // Below are wrapper functions for the intrinsic instructions.
                        // Should not be called directly.
                        #[inline(always)]
                        fn add_ne(p1: &#struct_name, p2: &#struct_name) -> #struct_name {
                            #[cfg(not(target_os = "zkvm"))]
                            {
                                use axvm_algebra::DivUnsafe;
                                let lambda = (&p2.y - &p1.y).div_unsafe(&p2.x - &p1.x);
                                let x3 = &lambda * &lambda - &p1.x - &p2.x;
                                let y3 = &lambda * &(&p1.x - &x3) - &p1.y;
                                #struct_name { x: x3, y: y3 }
                            }
                            #[cfg(target_os = "zkvm")]
                            {
                                let mut uninit: MaybeUninit<#struct_name> = MaybeUninit::uninit();
                                custom_insn_r!(
                                    CUSTOM_1,
                                    Custom1Funct3::ShortWeierstrass as usize,
                                    SwBaseFunct7::SwAddNe as usize + Self::EC_IDX
                                        * (axvm_platform::constants::SHORT_WEIERSTRASS_MAX_KINDS as usize),
                                    uninit.as_mut_ptr(),
                                    p1 as *const #struct_name,
                                    p2 as *const #struct_name
                                );
                                unsafe { uninit.assume_init() }
                            }
                        }

                        #[inline(always)]
                        fn add_ne_assign(&mut self, p2: &#struct_name) {
                            #[cfg(not(target_os = "zkvm"))]
                            {
                                use axvm_algebra::DivUnsafe;
                                let lambda = (&p2.y - &self.y).div_unsafe(&p2.x - &self.x);
                                let x3 = &lambda * &lambda - &self.x - &p2.x;
                                let y3 = &lambda * &(&self.x - &x3) - &self.y;
                                self.x = x3;
                                self.y = y3;
                            }
                            #[cfg(target_os = "zkvm")]
                            {
                                custom_insn_r!(
                                    CUSTOM_1,
                                    Custom1Funct3::ShortWeierstrass as usize,
                                    SwBaseFunct7::SwAddNe as usize + Self::EC_IDX
                                        * (axvm_platform::constants::SHORT_WEIERSTRASS_MAX_KINDS as usize),
                                    self as *mut #struct_name,
                                    self as *const #struct_name,
                                    p2 as *const #struct_name
                                );
                            }
                        }

                        /// Assumes that `p` is not identity.
                        #[inline(always)]
                        fn double_impl(p: &#struct_name) -> #struct_name {
                            #[cfg(not(target_os = "zkvm"))]
                            {
                                use axvm_algebra::DivUnsafe;
                                let two = #intmod_type::from_u8(2);
                                let lambda = &p.x * &p.x * #intmod_type::from_u8(3).div_unsafe(&p.y * &two);
                                let x3 = &lambda * &lambda - &p.x * &two;
                                let y3 = &lambda * &(&p.x - &x3) - &p.y;
                                #struct_name { x: x3, y: y3 }
                            }
                            #[cfg(target_os = "zkvm")]
                            {
                                let mut uninit: MaybeUninit<#struct_name> = MaybeUninit::uninit();
                                custom_insn_r!(
                                    CUSTOM_1,
                                    Custom1Funct3::ShortWeierstrass as usize,
                                    SwBaseFunct7::SwDouble as usize + Self::EC_IDX
                                        * (axvm_platform::constants::SHORT_WEIERSTRASS_MAX_KINDS as usize),
                                    uninit.as_mut_ptr(),
                                    p as *const #struct_name,
                                    "x0"
                                );
                                unsafe { uninit.assume_init() }
                            }
                        }

                        #[inline(always)]
                        fn double_assign_impl(&mut self) {
                            #[cfg(not(target_os = "zkvm"))]
                            {
                                use axvm_algebra::DivUnsafe;
                                let two = #intmod_type::from_u8(2);
                                let lambda = &self.x * &self.x * #intmod_type::from_u8(3).div_unsafe(&self.y * &two);
                                let x3 = &lambda * &lambda - &self.x * &two;
                                let y3 = &lambda * &(&self.x - &x3) - &self.y;
                                self.x = x3;
                                self.y = y3;
                            }
                            #[cfg(target_os = "zkvm")]
                            {
                                custom_insn_r!(
                                    CUSTOM_1,
                                    Custom1Funct3::ShortWeierstrass as usize,
                                    SwBaseFunct7::SwDouble as usize + Self::EC_IDX
                                        * (axvm_platform::constants::SHORT_WEIERSTRASS_MAX_KINDS as usize),
                                    self as *mut #struct_name,
                                    self as *const #struct_name,
                                    "x0"
                                );
                            }
                        }
                    }

                    impl SwPoint for #struct_name {
                        type Coordinate = #intmod_type;

                        // Ref: https://docs.rs/k256/latest/src/k256/arithmetic/affine.rs.html#247
                        fn from_encoded_point<C: Curve>(p: &EncodedPoint<C>) -> Self
                        where
                            C::FieldBytesSize: ModulusSize
                        {
                            match p.coordinates() {
                                Coordinates::Identity => Self::identity(),
                                Coordinates::Uncompressed { x, y } => {
                                    // Sec1 bytes are in big endian.
                                    let x = Self::Coordinate::from_be_bytes(x.as_ref());
                                    let y = Self::Coordinate::from_be_bytes(y.as_ref());
                                    // TODO: Verify that the point is on the curve

                                    Self { x, y }

                                }
                                Coordinates::Compact { x } => unimplemented!(),
                                Coordinates::Compressed { x, y_is_odd } => unimplemented!(),
                            }
                        }

                        fn to_sec1_bytes(&self, is_compressed: bool) -> Vec<u8>
                        {
                            let mut bytes = Vec::new();
                            if is_compressed {
                                let y_is_odd = self.y().as_le_bytes()[0] & 1;
                                if y_is_odd == 1 {
                                    bytes.push(0x03);
                                } else {
                                    bytes.push(0x02);
                                }
                                bytes.extend_from_slice(&self.x().as_be_bytes());
                            } else {
                                bytes.push(0x04);
                                bytes.extend_from_slice(&self.x().as_be_bytes());
                                bytes.extend_from_slice(&self.y().as_be_bytes());
                            }
                            bytes
                        }

                        fn x(&self) -> Self::Coordinate {
                            self.x.clone()
                        }

                        fn y(&self) -> Self::Coordinate {
                            self.y.clone()
                        }
                    }

                    impl Group for #struct_name {
                        type SelfRef<'a> = &'a Self;

                        fn identity() -> Self {
                            Self {
                                x: <#intmod_type as IntMod>::ZERO,
                                y: <#intmod_type as IntMod>::ZERO,
                            }
                        }

                        fn is_identity(&self) -> bool {
                            self.x == <#intmod_type as IntMod>::ZERO && self.y == <#intmod_type as IntMod>::ZERO
                        }

                        fn double(&self) -> Self {
                            if self.is_identity() {
                                self.clone()
                            } else {
                                Self::double_impl(self)
                            }
                        }

                        fn double_assign(&mut self) {
                            if !self.is_identity() {
                                Self::double_assign_impl(self);
                            }
                        }
                    }

                    impl Add<&#struct_name> for #struct_name {
                        type Output = Self;

                        fn add(mut self, p2: &#struct_name) -> Self::Output {
                            self.add_assign(p2);
                            self
                        }
                    }

                    impl Add for #struct_name {
                        type Output = Self;

                        fn add(self, rhs: Self) -> Self::Output {
                            self.add(&rhs)
                        }
                    }

                    impl Add<&#struct_name> for &#struct_name {
                        type Output = #struct_name;

                        fn add(self, p2: &#struct_name) -> Self::Output {
                            if self.is_identity() {
                                p2.clone()
                            } else if p2.is_identity() {
                                self.clone()
                            } else if self.x == p2.x {
                                if &self.y + &p2.y == <#intmod_type as IntMod>::ZERO {
                                    #struct_name::identity()
                                } else {
                                    #struct_name::double_impl(self)
                                }
                            } else {
                                #struct_name::add_ne(self, p2)
                            }
                        }
                    }

                    impl AddAssign<&#struct_name> for #struct_name {
                        fn add_assign(&mut self, p2: &#struct_name) {
                            if self.is_identity() {
                                *self = p2.clone();
                            } else if p2.is_identity() {
                                // do nothing
                            } else if self.x == p2.x {
                                if &self.y + &p2.y == <#intmod_type as IntMod>::ZERO {
                                    *self = Self::identity();
                                } else {
                                    Self::double_assign_impl(self);
                                }
                            } else {
                                Self::add_ne_assign(self, p2);
                            }
                        }
                    }

                    impl AddAssign for #struct_name {
                        fn add_assign(&mut self, rhs: Self) {
                            self.add_assign(&rhs);
                        }
                    }

                    impl Neg for #struct_name {
                        type Output = Self;

                        fn neg(self) -> Self::Output {
                            Self {
                                x: self.x,
                                y: -self.y,
                            }
                        }
                    }

                    impl Sub<&#struct_name> for #struct_name {
                        type Output = Self;

                        fn sub(self, rhs: &#struct_name) -> Self::Output {
                            self.sub(rhs.clone())
                        }
                    }

                    impl Sub for #struct_name {
                        type Output = #struct_name;

                        fn sub(self, rhs: Self) -> Self::Output {
                            self.add(rhs.neg())
                        }
                    }

                    impl Sub<&#struct_name> for &#struct_name {
                        type Output = #struct_name;

                        fn sub(self, p2: &#struct_name) -> Self::Output {
                            self.add(&p2.clone().neg())
                        }
                    }

                    impl SubAssign<&#struct_name> for #struct_name {
                        fn sub_assign(&mut self, p2: &#struct_name) {
                            self.sub_assign(p2.clone());
                        }
                    }

                    impl SubAssign for #struct_name {
                        fn sub_assign(&mut self, rhs: Self) {
                            self.add_assign(rhs.neg());
                        }
                    }
                });

                ec_idx += 1;

                Ok(result)
            }
            _ => Err("Only assignments are supported"),
        };
        if let Err(err) = result {
            return syn::Error::new_spanned(stmt, err).to_compile_error().into();
        } else {
            output.push(result.unwrap());
        }
    }

    TokenStream::from_iter(output)
}
