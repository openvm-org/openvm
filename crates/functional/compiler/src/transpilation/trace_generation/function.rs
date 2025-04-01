use crate::air::constructor::AirConstructor;
use crate::air::representation::RepresentationTable;
use crate::core::file3::FlattenedFunction;
use crate::core::ir::{Material, StatementVariant};
use crate::core::stage1::Stage2Program;
use crate::transpilation::execution::constants::*;
use crate::transpilation::execution::transpilation::VariableNamer;
use crate::transpilation::trace_generation::constants::*;
use itertools::Itertools;
use proc_macro2::TokenStream;
use quote::quote;

impl FlattenedFunction {
    pub fn transpile_generate_trace(
        &self,
        namer: &VariableNamer,
        representation_table: &RepresentationTable,
        air_constructor: &AirConstructor,
        program: &Stage2Program,
    ) -> TokenStream {
        let tracker = ident(TRACKER);
        let tracker_struct = tracker_struct_name();
        let trace_set = quote! { trace_set };
        let trace_set_struct = trace_set_struct_name();
        let trace_set_field_name = trace_set_field_name(&self.name);
        let generate_trace = generate_trace_function_name();
        let width_name = trace_width_const_name();

        let mut pieces = vec![];

        for (i, function_call) in self.function_calls.iter().enumerate() {
            if function_call.material == Material::Materialized {
                let callee = namer.callee_name(i);
                pieces.push(quote! {
                    #callee.as_ref().as_ref().unwrap().#generate_trace(#tracker, #trace_set);
                })
            }
        }

        let row = quote! { row };
        let call_index = namer.call_index();
        pieces.push(quote! {
            let #row = &mut #trace_set.#trace_set_field_name(&self.name)[#call_index * Self::#width_name..(#call_index + 1) * Self::#width_name];
        });
        for (scope, cell) in air_constructor.leaf_scopes.iter() {
            let scope_field = namer.scope_name(scope);
            pieces.push(quote! {
                if #scope_field {
                    #row[#cell] = F::ONE;
                }
            });
        }
        for ((scope, name), representation) in representation_table.representations.iter() {
            let cells_to_set = (0..representation.len())
                .filter(|&i| representation.owned[i].is_some())
                .collect_vec();
            if !cells_to_set.is_empty() {
                let scope_field = namer.scope_name(scope);
                let declaration_scope = self.declaration_set.get_declaration_scope(scope, name);
                let tipo = &self.declaration_set.declarations[&(declaration_scope, name.clone())];
                let size = program.types.calc_type_size(tipo);
                let as_cells = quote! { as_cells };

                let mut here = vec![];
                here.push(quote! {
                    let #as_cells = [F; #size];
                });
                for i in cells_to_set {
                    let cell = representation.owned[i].unwrap();
                    here.push(quote! {
                        #row[#cell] = #as_cells[#i];
                    });
                }
                pieces.push(quote! {
                    if #scope_field {
                        #(#here)*
                    }
                });
            }
        }
        for (&index, &cell) in representation_table.reference_multiplicity_cells.iter() {
            let scope = &self.statements[index].scope;
            let scope_field = namer.scope_name(scope);
            let multiplicity = match &self.statements[index].statement {
                StatementVariant::Reference { data, .. } => {
                    let reference = namer.reference_name(index);
                    let tipo = data.get_type();
                    let type_identifier = type_to_identifier_execution(&tipo);
                    let tracker_field_name = ident(&format!("{}_{}", MEMORY, type_identifier));
                    let get_reference_multiplicity = ident(GET_REFERENCE_MULTIPLICITY);

                    quote! { #tracker.#tracker_field_name.#get_reference_multiplicity(#reference) }
                }
                StatementVariant::PrefixAppend { elem, .. } => {
                    let appended_array = namer.appended_array_name(index);
                    let appended_index = namer.appended_index_name(index);
                    let tipo = elem.get_type();
                    let type_identifier = type_to_identifier_execution(&tipo);
                    let tracker_field_name = ident(&format!("{}_{}", MEMORY, type_identifier));
                    let get_array_multiplicity = ident(GET_ARRAY_MULTIPLICITY);

                    quote! { #tracker.#tracker_field_name.#get_array_multiplicity(#appended_array, #appended_index) }
                }
                _ => unreachable!(),
            };
            let multiplicity = usize_to_field_elem(multiplicity);
            pieces.push(quote! {
                if #scope_field {
                    #row[#cell] = #multiplicity;
                }
            });
        }

        quote! {
            fn #generate_trace(&self, #tracker: &#tracker_struct, #trace_set: &mut #trace_set_struct) -> Vec<F> {
                #(#pieces)*
            }
        }
    }

    pub fn transpile_calc_zk_identifier(
        &self,
        namer: &VariableNamer,
        representation_table: &RepresentationTable,
    ) -> TokenStream {
        let mut branches = vec![];
        for (i, offset) in representation_table.reference_offsets.iter() {
            branches.push(quote! {
                #i => #offset,
            });
        }
        branches.push(quote! {
            _ => unreachable!(),
        });
        let calc_zk_identifier = ident(CALC_ZK_IDENTIFIER);
        let function_id = ident(FUNCTION_ID);
        let call_index = namer.call_index();
        let max_trace_height_name = max_trace_height_const_name();
        let num_references_name = num_references_const_name();

        quote! {
            fn #calc_zk_identifier(&self, i: usize) -> usize {
                let offset = match i {
                    #(#branches)*
                };
                (Self::#function_id * #max_trace_height_name) + (#call_index * Self::#num_references_name) + offset
            }
        }
    }

    pub fn transpile_trace_generation_impl(
        &self,
        representation_table: &RepresentationTable,
        air_constructor: &AirConstructor,
        program: &Stage2Program,
    ) -> TokenStream {
        let namer = VariableNamer::new(&self.declaration_set);
        let generate_trace =
            self.transpile_generate_trace(&namer, representation_table, air_constructor, program);
        let calc_zk_identifier = self.transpile_calc_zk_identifier(&namer, representation_table);

        let struct_name = function_struct_name(&self.name);
        let width_name = trace_width_const_name();
        let width_value = air_constructor.width();
        let num_references_name = num_references_const_name();
        let num_references_value = representation_table.reference_offsets.len();
        quote! {
            impl #struct_name {
                const #width_name: usize = #width_value;
                const #num_references_name: usize = #num_references_value;
                #generate_trace
                #calc_zk_identifier
            }
        }
    }
}
