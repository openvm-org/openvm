use proc_macro2::TokenStream;
use quote::quote;

use crate::{
    execution::{
        constants::*,
        transpilation::{FieldNamer, VariableNamer},
    },
    folder1::{
        file3::{Atom, FlattenedFunction},
        ir::{Material, StatementVariant, Type},
        stage1::Stage2Program,
    },
};

impl FlattenedFunction {
    pub fn transpile_struct_declaration(&self) -> TokenStream {
        let mut fields = vec![];
        let field_namer = FieldNamer::new(&self.declaration_set);
        for ((scope, name), tipo) in self.declaration_set.declarations.iter() {
            let name = field_namer.variable_name(scope, name);
            let tipo = type_to_rust(tipo);
            fields.push(quote! {
                pub #name: #tipo,
            });
        }
        for matchi in self.matches.iter() {
            for branch in matchi.branches.iter() {
                let scope = matchi.scope.then(matchi.index, branch.constructor.clone());
                let name = field_namer.scope_name(&scope);
                fields.push(quote! {
                    pub #name: bool,
                });
            }
        }
        let ref_type = reference_type();
        let array_type = array_type();
        for (i, statement) in self.statements.iter().enumerate() {
            if statement.material == Material::Materialized {
                match statement.statement {
                    StatementVariant::Reference { .. } => {
                        let name = field_namer.reference_name(i);
                        fields.push(quote! {
                            pub #name: #ref_type,
                        })
                    }
                    StatementVariant::ArrayFinalization { .. } => {
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
                pub #name: Box<Option<#struct_name>>,
            });
        }

        let struct_name = function_struct_name(&self.name);
        quote! {
            #[derive(Default, Debug)]
            pub struct #struct_name {
                #(#fields)*
            }
        }
    }

    pub fn transpile_stage(&self, stage_index: usize, program: &Stage2Program) -> TokenStream {
        let mut transpiled_atoms = vec![];
        let mut namer = VariableNamer::new(&self.declaration_set);
        for atom in self.atoms_staged[stage_index].iter() {
            let transpiled_atom =
                match atom {
                    Atom::Statement(index) => {
                        self.statements[*index].transpile(*index, program, &mut namer)
                    }
                    Atom::Match(index) => {
                        self.matches[*index].transpile(*index, program, &mut namer)
                    }
                    Atom::PartialFunctionCall(index, stage) => self.function_calls[*index]
                        .transpile(*stage, *index, &program.types, program, &mut namer),
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
        let implementation = self.transpile_impl(program);
        quote! {
            #declaration
            #implementation
        }
    }

    pub fn get_types_in_memory(&self) -> Vec<Type> {
        let mut result = Vec::new();
        for statement in self.statements.iter() {
            match &statement.statement {
                StatementVariant::Reference { data, .. } => {
                    result.push(data.get_type().clone());
                }
                StatementVariant::EmptyUnderConstructionArray { elem_type, .. } => {
                    result.push(elem_type.clone());
                }
                _ => {}
            }
        }
        result
    }
}
