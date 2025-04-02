use proc_macro2::{Literal, TokenStream, TokenTree};
use quote::quote;

use crate::air::air::{
    AirExpression, Bus, Constraint, Direction, Interaction, ScopedConstraint, Term,
};
use crate::air::constructor::AirConstructor;

fn local() -> TokenStream {
    quote! { local }
}

fn next() -> TokenStream {
    quote! { next }
}

fn cell() -> TokenStream {
    quote! { cell }
}

fn builder() -> TokenStream {
    quote! { builder }
}

impl Term {
    pub fn transpile(self) -> TokenStream {
        match self {
            Term::Cell(index) => {
                let as_token = TokenTree::Literal(Literal::usize_unsuffixed(index));
                let cell = cell();
                quote! { #cell(#as_token) }
            }
            Term::Constant(constant) => {
                let as_token =
                    TokenTree::Literal(Literal::usize_unsuffixed(constant.abs() as usize));
                if constant >= 0 {
                    quote! { AB::Expr::from_canonical_usize(#as_token) }
                } else {
                    quote! { -AB::F::from_canonical_usize(#as_token) }
                }
            }
        }
    }
}

impl AirExpression {
    pub fn transpile(&self) -> TokenStream {
        if self.sum.is_empty() {
            quote! { AB::Expr::ZERO }
        } else if self.sum.len() == 1 {
            Self::transpile_product(&self.sum[0], false)
        } else {
            let products = self
                .sum
                .iter()
                .map(|product| Self::transpile_product(product, true));
            quote! { #(#products)+* }
        }
    }

    fn transpile_product(product: &Vec<Term>, parenthesize: bool) -> TokenStream {
        if product.is_empty() {
            quote! { AB::Expr::ONE }
        } else {
            let mut result = product[0].transpile();
            for term in product.iter().skip(1) {
                let term = term.transpile();
                result = quote! { #result * #term };
            }
            if parenthesize && product.len() > 1 {
                quote! { (#result) }
            } else {
                result
            }
        }
    }
}

impl Constraint {
    pub fn transpile(&self) -> TokenStream {
        let left = self.left.transpile();
        let right = self.right.transpile();
        let builder = builder();
        quote! { #builder.assert_eq(#left, #right); }
    }
}

impl ScopedConstraint {
    pub fn transpile(&self, air_constructor: &AirConstructor) -> TokenStream {
        let scope_expression = air_constructor
            .get_scope_expression(&self.scope)
            .transpile();
        let left = self.constraint.left.transpile();
        let right = self.constraint.right.transpile();
        let builder = builder();
        quote! { #builder.when(#scope_expression).assert_eq(#left, #right); }
    }
}

impl Bus {
    pub fn index(self) -> u16 {
        match self {
            Bus::Function => 0,
            Bus::Reference => 1,
            Bus::Array => 2,
        }
    }
}

impl Direction {
    pub fn wrap(self, expression: TokenStream) -> TokenStream {
        match self {
            Direction::Send => quote! { #expression },
            Direction::Receive => quote! { - (#expression) },
        }
    }
}

impl Interaction {
    pub fn transpile(&self) -> TokenStream {
        let builder = builder();
        let bus = self.bus.index();
        let fields = self.fields.iter().map(AirExpression::transpile);
        let multiplicity = self.direction.wrap(self.multiplicity.transpile());
        quote! {
            #builder.push_interaction(
                #bus, [#(#fields),*], #multiplicity, 1
            );
        }
    }
}

impl AirConstructor {
    pub fn transpile_constraints(&self) -> TokenStream {
        let builder = builder();
        let local = local();
        let next = next();
        let cell = cell();

        let mut pieces = vec![];
        pieces.push(quote! {
            let main = #builder.main();
            let #local = main.row_slice(0);
            let #next = main.row_slice(1);
            let #cell = |i: usize| local[i].into();
        });
        if let Some(row_index_cell) = self.row_index_cell {
            let cell = Term::Cell(row_index_cell).transpile();
            pieces.push(quote! {
                builder.when_first_row().assert_zero(#cell);
                builder.when_transition().assert_eq(#next[#row_index_cell], #cell + AB::F::ONE);
            });
        }
        pieces.extend(self.constraints.iter().map(Constraint::transpile));
        pieces.extend(
            self.scoped_constraints
                .iter()
                .map(|scoped_constraint| scoped_constraint.transpile(self)),
        );
        pieces.extend(self.interactions.iter().map(Interaction::transpile));

        quote! {
            #(#pieces)*
        }
    }
}
