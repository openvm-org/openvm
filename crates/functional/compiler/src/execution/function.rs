use std::collections::HashSet;

use proc_macro2::TokenStream;
use quote::quote;

use crate::{
    execution::{
        constants::*,
        transpilation::{FieldNamer, VariableNamer},
    },
    folder1::{
        file2_tree::ScopePath,
        file3::{Atom, FlattenedFunction},
        ir::{Material, Statement, Type},
        stage1::Stage2Program,
    },
};

impl FlattenedFunction {
    pub fn transpile_struct_declaration(&self) -> TokenStream {
        let mut fields = vec![];
        let field_namer = FieldNamer::new(&self.declaration_set);
        for (i, argument) in self.arguments.iter().enumerate() {
            let name = field_namer.argument_name(i);
            let tipo = type_to_rust(argument.get_type());
            fields.push(quote! {
                pub #name: #tipo,
            });
        }
        for ((scope, name), tipo) in self.declaration_set.declarations.iter() {
            let name = field_namer.variable_name(scope, name);
            let tipo = type_to_rust(tipo);
            fields.push(quote! {
                pub #name: #tipo,
            });
        }
        let mut scopes = HashSet::new();
        for statement in self.statements.iter() {
            scopes.insert(statement.scope.clone());
        }
        for matchi in self.matches.iter() {
            scopes.insert(matchi.scope.clone());
        }
        for function_call in self.function_calls.iter() {
            scopes.insert(function_call.scope.clone());
        }
        for scope in scopes {
            let name = field_namer.scope_name(&scope);
            fields.push(quote! {
                pub #name: bool,
            });
        }
        let ref_type = reference_type();
        let array_type = array_type();
        for (i, statement) in self.statements.iter().enumerate() {
            if statement.material == Material::Materialized {
                match statement.statement {
                    Statement::Reference { .. } => {
                        let name = field_namer.reference_name(i);
                        fields.push(quote! {
                            pub #name: #ref_type,
                        })
                    }
                    Statement::ArrayFinalization { .. } => {
                        let name = field_namer.finalized_array_name(i);
                        fields.push(quote! {
                            pub #name: #array_type,
                        })
                    }
                    _ => {}
                }
            }
        }
        for (i, function_call) in self.function_calls.iter().enumerate() {
            let name = field_namer.callee_name(i);
            let struct_name = function_struct_name(&function_call.function_name);
            fields.push(quote! {
                pub #name: #struct_name,
            });
        }

        let struct_name = function_struct_name(&self.name);
        quote! {
            pub struct #struct_name {
                #(#fields)*
            }
        }
    }

    pub fn transpile_default_impl(&self) -> TokenStream {
        let struct_name = function_struct_name(&self.name);
        quote! {
            impl Default for #struct_name {
                fn default() -> Self {
                    unsafe {
                        std::mem::zeroed()
                    }
                }
            }
        }
    }

    pub fn transpile_stage(&self, stage_index: usize, program: &Stage2Program) -> TokenStream {
        let mut transpiled_atoms = vec![];
        for atom in self.atoms_staged[stage_index].iter() {
            let transpiled_atom = match atom {
                Atom::Statement(index) => {
                    self.statements[*index].transpile(*index, &program.types, &self.declaration_set)
                }
                Atom::Match(index) => {
                    self.matches[*index].transpile(*index, &program.types, &self.declaration_set)
                }
                Atom::PartialFunctionCall(index, stage) => self.function_calls[*index].transpile(
                    *stage,
                    *index,
                    &program.types,
                    &self.declaration_set,
                ),
                Atom::InArgument(index) => {
                    let mut namer = VariableNamer::new(&self.declaration_set);
                    let argument_name = namer.argument_name(*index);
                    self.arguments[*index].transpile_top_down(
                        &ScopePath::empty(),
                        &argument_name,
                        &mut namer,
                        &program.types,
                    )
                }
                Atom::OutArgument(index) => {
                    let namer = VariableNamer::new(&self.declaration_set);
                    let argument_name = namer.argument_name(*index);
                    let argument_value = self.arguments[*index].transpile_defined(
                        &ScopePath::empty(),
                        &namer,
                        &program.types,
                    );
                    quote! {
                        #argument_name = #argument_value;
                    }
                }
            };
            transpiled_atoms.push(transpiled_atom);
        }
        define_stage(
            stage_index,
            quote! {
                #(#transpiled_atoms)*
            },
        )
    }

    pub fn transpile_impl(&self, program: &Stage2Program) -> TokenStream {
        let mut stages = vec![];
        for i in 0..self.stages.len() {
            stages.push(self.transpile_stage(i, program));
        }
        let struct_name = function_struct_name(&self.name);
        quote! {
            impl #struct_name {
                #(#stages)*
            }
        }
    }

    pub fn transpile(&self, program: &Stage2Program) -> TokenStream {
        let declaration = self.transpile_struct_declaration();
        let default_impl = self.transpile_default_impl();
        let implementation = self.transpile_impl(program);
        quote! {
            #declaration
            #default_impl
            #implementation
        }
    }

    pub fn get_types_in_memory(&self) -> Vec<Type> {
        let mut result = Vec::new();
        for statement in self.statements.iter() {
            match &statement.statement {
                Statement::Reference { data, .. } => {
                    result.push(data.get_type().clone());
                }
                Statement::EmptyUnderConstructionArray { elem_type, .. } => {
                    result.push(elem_type.clone());
                }
                _ => {}
            }
        }
        result
    }
}
