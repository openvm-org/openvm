use itertools::Itertools;
use proc_macro2::TokenStream;
use quote::quote;

use crate::{
    execution::constants::*,
    folder1::{
        file2_tree::{DeclarationSet, ExpressionContainer, ScopePath},
        file3::{FlatFunctionCall, FlatMatch, FlatStatement},
        function_resolution::Stage,
        ir::{ArithmeticOperator, Expression, Material, Statement},
        type_resolution::TypeSet,
    },
};

pub struct FieldNamer<'a> {
    declaration_set: &'a DeclarationSet,
}

impl<'a> FieldNamer<'a> {
    pub(crate) fn new(declaration_set: &'a DeclarationSet) -> Self {
        Self { declaration_set }
    }
}

impl<'a> FieldNamer<'a> {
    pub fn scope_name(&self, scope: &ScopePath) -> TokenStream {
        let mut full_name = "scope".to_string();
        for (i, branch) in scope.0.iter() {
            full_name.extend(format!("_{}_{}", i, branch).chars());
        }
        ident(&full_name)
    }
    pub fn variable_name(&self, curr: &ScopePath, name: &str) -> TokenStream {
        let scope = self.declaration_set.get_declaration_scope(curr, name);
        let mut full_name = name.to_string();
        for (i, branch) in scope.0.iter() {
            full_name.extend(format!("_{}_{}", i, branch).chars());
        }
        ident(&full_name)
    }
    pub fn reference_name(&self, index: usize) -> TokenStream {
        ident(&format!("ref_{}", index))
    }
    pub fn finalized_array_name(&self, index: usize) -> TokenStream {
        ident(&format!("finalized_array_{}", index))
    }
    pub fn callee_name(&self, index: usize) -> TokenStream {
        ident(&format!("callee_{}", index))
    }
    pub fn argument_name(&self, index: usize) -> TokenStream {
        ident(&format!("argument_{}", index))
    }
}

pub struct VariableNamer<'a> {
    counter: usize,
    field_namer: FieldNamer<'a>,
}

impl<'a> VariableNamer<'a> {
    pub fn new(declaration_set: &'a DeclarationSet) -> Self {
        Self {
            counter: 0,
            field_namer: FieldNamer::new(declaration_set),
        }
    }
    pub fn new_temporary_name(&mut self) -> TokenStream {
        self.counter += 1;
        ident(&format!("temp_{}", self.counter))
    }
    pub fn refer_to_field(&self, name: TokenStream) -> TokenStream {
        quote! {
            self.#name
        }
    }
    pub fn scope_name(&self, scope: &ScopePath) -> TokenStream {
        self.refer_to_field(self.field_namer.scope_name(scope))
    }
    pub fn variable_name(&self, curr: &ScopePath, name: &str) -> TokenStream {
        self.refer_to_field(self.field_namer.variable_name(curr, name))
    }
    pub fn reference_name(&self, index: usize) -> TokenStream {
        self.refer_to_field(self.field_namer.reference_name(index))
    }
    pub fn finalized_array_name(&self, index: usize) -> TokenStream {
        self.refer_to_field(self.field_namer.finalized_array_name(index))
    }
    pub fn callee_name(&self, index: usize) -> TokenStream {
        self.refer_to_field(self.field_namer.callee_name(index))
    }
    pub fn argument_name(&self, index: usize) -> TokenStream {
        self.refer_to_field(self.field_namer.argument_name(index))
    }
}

impl ExpressionContainer {
    pub fn transpile_defined(
        &self,
        scope: &ScopePath,
        namer: &VariableNamer,
        type_set: &TypeSet,
    ) -> TokenStream {
        let guard = self.expression.lock().unwrap();
        match &*guard {
            Expression::Constant { value } => isize_to_field_elem(*value),
            Expression::Variable { name } => namer.variable_name(scope, name),
            Expression::Let { .. } => unreachable!(),
            Expression::Define { .. } => unreachable!(),
            Expression::Algebraic {
                constructor,
                fields,
            } => {
                let type_name =
                    type_name(&type_set.get_constructor_type_name(constructor).unwrap());
                let fields = fields
                    .iter()
                    .map(|field| field.transpile_defined(scope, namer, type_set));
                quote! {
                    #type_name(#(#fields),*)
                }
            }
            Expression::Arithmetic {
                left,
                right,
                operator,
            } => {
                let left = left.transpile_defined(scope, namer, type_set);
                let right = right.transpile_defined(scope, namer, type_set);
                match operator {
                    ArithmeticOperator::Plus => quote! { #left + #right },
                    ArithmeticOperator::Minus => quote! { #left - #right },
                    ArithmeticOperator::Times => quote! { #left * #right },
                }
            }
            Expression::Dematerialized { value } => value.transpile_defined(scope, namer, type_set),
            Expression::EqUnmaterialized { left, right } => eq_to_bool(
                left.transpile_defined(scope, namer, type_set),
                right.transpile_defined(scope, namer, type_set),
            ),
            Expression::EmptyConstArray { elem_type } => {
                let elem_type = type_to_rust(elem_type);
                quote! {
                    [#elem_type; 0]
                }
            }
            Expression::ConstArray { elements } => {
                let elements = elements
                    .iter()
                    .map(|element| element.transpile_defined(scope, namer, type_set));
                quote! {
                    [#(#elements),*]
                }
            }
            Expression::ConstArrayConcatenation { left, right } => {
                let (_, left_len) = left
                    .get_type()
                    .get_const_array_type(Material::Dematerialized)
                    .unwrap();
                let left_indices = 0..left_len;
                let left = left.transpile_defined(scope, namer, type_set);

                let (_, right_len) = right
                    .get_type()
                    .get_const_array_type(Material::Dematerialized)
                    .unwrap();
                let right_indices = 0..right_len;
                let right = right.transpile_defined(scope, namer, type_set);

                quote! {
                    [#(#left[#left_indices]),*, #(#right[#right_indices]),*]
                }
            }
            Expression::ConstArrayAccess { array, index } => {
                let array = array.transpile_defined(scope, namer, type_set);
                let index = *index;
                quote! {
                    #array[#index]
                }
            }
            Expression::ConstArraySlice { array, from, to } => {
                let array = array.transpile_defined(scope, namer, type_set);
                let indices = *from..*to;
                quote! {
                    [#(#array[#indices]),*]
                }
            }
        }
    }

    pub fn transpile_top_down(
        &self,
        scope: &ScopePath,
        this: &TokenStream,
        namer: &mut VariableNamer,
        type_set: &TypeSet,
    ) -> TokenStream {
        let this_copy = this.clone();
        let guard = self.expression.lock().unwrap();
        match &*guard {
            Expression::Let { name } => {
                let name = namer.variable_name(scope, name);
                return quote! {
                    #name = #this;
                };
            }
            Expression::Define { name } => {
                let name = namer.variable_name(scope, name);
                return quote! {
                    #name = #this;
                };
            }
            Expression::Algebraic {
                constructor,
                fields,
            } => {
                let type_name =
                    type_name(&type_set.get_constructor_type_name(constructor).unwrap());
                let names = (0..fields.len())
                    .map(|_| namer.new_temporary_name())
                    .collect::<Vec<_>>();
                let insides = fields
                    .iter()
                    .zip_eq(names.iter())
                    .map(|(field, name)| field.transpile_top_down(scope, name, namer, type_set));
                return quote! {
                    if let #type_name(#(#names),*) = #this {
                        #(#insides)*
                    } else {
                        panic!();
                    }
                };
            }
            Expression::Dematerialized { value } => {
                return value.transpile_top_down(scope, this, namer, type_set);
            }
            Expression::ConstArray { elements } => {
                let names = (0..elements.len())
                    .map(|_| namer.new_temporary_name())
                    .collect::<Vec<_>>();
                let insides = elements
                    .iter()
                    .zip_eq(names.iter())
                    .map(|(elem, name)| elem.transpile_top_down(scope, name, namer, type_set));
                return quote! {
                    let [(#(#names),*)] = #this;
                    #(#insides)*
                };
            }
            _ => {}
        }
        let defined = self.transpile_defined(scope, namer, type_set);
        quote! {
            assert_eq!(#this_copy, #defined);
        }
    }
}

impl FlatStatement {
    pub fn transpile(
        &self,
        index: usize,
        type_set: &TypeSet,
        declaration_set: &DeclarationSet,
    ) -> TokenStream {
        let mut namer = VariableNamer::new(declaration_set);
        let scope = &self.scope;
        let material = self.material;
        let block = match &self.statement {
            Statement::VariableDeclaration { .. } => quote! {},
            Statement::Equality { left, right } => {
                let right = right.transpile_defined(scope, &namer, type_set);
                left.transpile_top_down(scope, &right, &mut namer, type_set)
            }
            Statement::Reference {
                reference: reference_expression,
                data,
            } => {
                let type_identifier = type_to_identifier(data.get_type());
                let data = data.transpile_defined(scope, &namer, type_set);
                let reference = create_ref(type_identifier, data);
                let (init, this) = match material {
                    Material::Materialized => {
                        let ref_name = namer.reference_name(index);
                        (
                            quote! {
                                #ref_name = #reference;
                            },
                            ref_name,
                        )
                    }
                    Material::Dematerialized => {
                        let temp_name = namer.new_temporary_name();
                        (
                            quote! {
                                let #temp_name = #reference;
                            },
                            temp_name,
                        )
                    }
                };
                let following =
                    reference_expression.transpile_top_down(scope, &this, &mut namer, type_set);
                quote! {
                    #init
                    #following
                }
            }
            Statement::Dereference {
                data: data_expression,
                reference,
            } => {
                let tipo = reference.get_type();
                let reference = reference.transpile_defined(scope, &namer, type_set);
                data_expression.transpile_top_down(
                    scope,
                    &dereference(tipo, reference),
                    &mut namer,
                    type_set,
                )
            }
            Statement::EmptyUnderConstructionArray { array, elem_type } => array
                .transpile_top_down(
                    scope,
                    &create_empty_under_construction_array(elem_type),
                    &mut namer,
                    type_set,
                ),
            Statement::UnderConstructionArrayPrepend {
                new_array,
                elem,
                old_array,
            } => {
                let tipo = elem.get_type();
                let old_array = old_array.transpile_defined(scope, &namer, type_set);
                let elem = elem.transpile_defined(scope, &namer, type_set);
                new_array.transpile_top_down(
                    scope,
                    &prepend_under_construction_array(tipo, old_array, elem),
                    &mut namer,
                    type_set,
                )
            }
            Statement::ArrayFinalization {
                finalized: finalized_expression,
                under_construction,
            } => {
                let tipo = under_construction.get_type();
                let under_construction =
                    under_construction.transpile_defined(scope, &namer, type_set);
                let finalized = finalize_array(tipo, under_construction);
                let (init, this) = match material {
                    Material::Materialized => {
                        let finalized_array_name = namer.finalized_array_name(index);
                        (
                            quote! {
                                #finalized_array_name = #finalized;
                            },
                            finalized_array_name,
                        )
                    }
                    Material::Dematerialized => {
                        let temp_name = namer.new_temporary_name();
                        (
                            quote! {
                                let #temp_name = #finalized;
                            },
                            temp_name,
                        )
                    }
                };
                let following =
                    finalized_expression.transpile_top_down(scope, &this, &mut namer, type_set);
                quote! {
                    #init
                    #following
                }
            }
            Statement::ArrayAccess { elem, array, index } => {
                let tipo = elem.get_type();
                let array = array.transpile_defined(scope, &namer, type_set);
                let index = index.transpile_defined(scope, &namer, type_set);
                elem.transpile_top_down(
                    scope,
                    &array_access(tipo, array, index),
                    &mut namer,
                    type_set,
                )
            }
        };
        let scope_name = namer.scope_name(scope);
        quote! {
            if #scope_name { #block }
        }
    }
}

impl FlatFunctionCall {
    pub fn transpile(
        &self,
        stage: Stage,
        index: usize,
        type_set: &TypeSet,
        declaration_set: &DeclarationSet,
    ) -> TokenStream {
        let mut namer = VariableNamer::new(declaration_set);
        let mut block = vec![];

        let callee = namer.callee_name(index);
        let struct_name = function_struct_name(&self.function_name);
        if stage.index == 0 {
            block.push(quote! {
                #callee = Box::new(#struct_name::init());
            });
        };
        let empty_declaration_set = DeclarationSet::new();
        let blank_field_namer = FieldNamer::new(&empty_declaration_set);
        for i in stage.start..stage.mid {
            let argument = &self.arguments[i].transpile_defined(&self.scope, &namer, type_set);
            let callee_field = blank_field_namer.argument_name(i);
            block.push(quote! {
                #callee.#callee_field = #argument;
            });
        }
        block.push(execute_stage(&callee, stage.index));
        for i in stage.mid..stage.end {
            let callee_field = blank_field_namer.argument_name(i);
            block.push(self.arguments[i].transpile_top_down(
                &self.scope,
                &quote! { #callee.#callee_field },
                &mut namer,
                type_set,
            ));
        }

        let scope_name = namer.scope_name(&self.scope);
        quote! {
            if #scope_name { #(#block)* }
        }
    }
}

impl FlatMatch {
    pub fn transpile(
        &self,
        _: usize,
        type_set: &TypeSet,
        declaration_set: &DeclarationSet,
    ) -> TokenStream {
        let mut namer = VariableNamer::new(declaration_set);
        let type_name = type_to_rust(&self.value.get_type());

        let mut arms = vec![];
        for (constructor, components) in self.branches.iter() {
            let scope = self.scope.then(self.index, constructor.clone());
            let scope_name = namer.scope_name(&scope);
            let temps = (0..components.len())
                .map(|_| namer.new_temporary_name())
                .collect::<Vec<_>>();
            let fields = components
                .iter()
                .map(|component| namer.variable_name(&scope, component));
            let arm = quote! {
                #type_name::#constructor(#(#temps),*) => {
                    #(#fields = #temps;)*
                    #scope_name = true;
                }
            };
            arms.push(arm);
        }

        let scope_name = namer.scope_name(&self.scope);
        let value = self.value.transpile_defined(&self.scope, &namer, type_set);
        quote! {
            if #scope_name {
                match #type_name::#value {
                    #(#arms)*
                }
            }
        }
    }
}
