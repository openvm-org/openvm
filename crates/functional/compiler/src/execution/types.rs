use proc_macro2::TokenStream;

use crate::{
    execution::{transpilation::VariableNamer, util::*},
    folder1::ir::Type,
};

fn field_type() -> TokenStream {
    ident("F")
}

fn reference_type() -> TokenStream {
    ident("TLRef")
}

fn array_type() -> TokenStream {
    ident("TLArray")
}

fn under_construction_array_type() -> TokenStream {
    ident("UnderConstructionArray")
}

impl VariableNamer {
    pub fn type_to_rust(&self, tipo: &Type) -> TokenStream {
        match tipo {
            Type::Field => ident("F"),
            Type::Reference(inner) => connect([
                reference_type(),
                less_than(),
                self.type_to_rust(inner),
                greater_than(),
            ]),
            Type::Array(inner) => connect([
                array_type(),
                less_than(),
                self.type_to_rust(inner),
                greater_than(),
            ]),
            Type::UnderConstructionArray(inner) => connect([
                under_construction_array_type(),
                less_than(),
                self.type_to_rust(inner),
                greater_than(),
            ]),
            Type::NamedType(name) => self.type_name(name),
            Type::Unmaterialized(inner) => self.type_to_rust(inner),
            Type::ConstArray(elem, length) => bracketed([
                self.type_to_rust(elem),
                semicolon(),
                literal(*length as isize),
            ]),
        }
    }
    pub fn type_to_identifier(&self, tipo: &Type) -> String {
        match tipo {
            Type::Field => "F".to_string(),
            Type::Reference(inner) => format!("Ref_{}", self.type_to_identifier(inner)),
            Type::Array(inner) => format!("Array_{}", self.type_to_identifier(inner)),
            Type::UnderConstructionArray(inner) => {
                format!("UnderConstructionArray_{}", self.type_to_identifier(inner))
            }
            Type::NamedType(name) => name.clone(),
            Type::Unmaterialized(inner) => self.type_to_identifier(inner),
            Type::ConstArray(elem, length) => {
                format!("ConstArray{}_{}", length, self.type_to_identifier(elem))
            }
        }
    }
}
