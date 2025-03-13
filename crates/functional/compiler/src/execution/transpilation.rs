use itertools::Itertools;
use proc_macro2::TokenStream;
use quote::quote;

use crate::{
    execution::constants::*,
    folder1::{
        file2_tree::{DeclarationSet, ExpressionContainer, ScopePath},
        file3::{FlatFunctionCall, FlatMatch, FlatStatement},
        function_resolution::Stage,
        ir::{
            ArithmeticOperator, BooleanOperator, Expression, Material, Statement,
            FALSE_CONSTRUCTOR_NAME, TRUE_CONSTRUCTOR_NAME,
        },
        stage1::Stage2Program,
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

pub fn name_field_according_to_scope(scope: &ScopePath, name: &str) -> TokenStream {
    let mut full_name = name.to_string();
    for (i, branch) in scope.0.iter() {
        full_name.extend(format!("_{}_{}", i, branch).chars());
    }
    ident(&full_name)
}

impl<'a> FieldNamer<'a> {
    pub fn scope_name(&self, scope: &ScopePath) -> TokenStream {
        assert!(!scope.0.is_empty());
        let mut full_name = "scope".to_string();
        for (i, branch) in scope.0.iter() {
            full_name.extend(format!("_{}_{}", i, branch).chars());
        }
        ident(&full_name)
    }
    pub fn variable_name(&self, curr: &ScopePath, name: &str) -> TokenStream {
        let scope = self.declaration_set.get_declaration_scope(curr, name);
        name_field_according_to_scope(&scope, name)
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
    pub fn scoped(&self, scope: &ScopePath, body: TokenStream) -> TokenStream {
        if scope.0.is_empty() {
            body
        } else {
            let scope_name = self.scope_name(scope);
            quote! {
                if #scope_name {
                    #body
                }
            }
        }
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
}

impl ExpressionContainer {
    pub fn transpile_defined(
        &self,
        scope: &ScopePath,
        namer: &VariableNamer,
        type_set: &TypeSet,
    ) -> TokenStream {
        match self.expression.as_ref() {
            Expression::Constant { value } => isize_to_field_elem(*value),
            Expression::Variable { name, defines, .. } => {
                assert!(!*defines);
                namer.variable_name(scope, name)
            }
            Expression::Algebraic {
                constructor,
                fields,
            } => {
                if constructor == TRUE_CONSTRUCTOR_NAME {
                    quote! { true }
                } else if constructor == FALSE_CONSTRUCTOR_NAME {
                    quote! { false }
                } else {
                    let type_name =
                        type_name(&type_set.get_constructor_type_name(constructor).unwrap());
                    let fields = fields
                        .iter()
                        .map(|field| field.transpile_defined(scope, namer, type_set));
                    quote! {
                        #type_name(#(#fields),*)
                    }
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
                    ArithmeticOperator::Div => quote! { #left / #right },
                }
            }
            Expression::Dematerialized { value } => value.transpile_defined(scope, namer, type_set),
            Expression::Eq { left, right } => {
                let left = left.transpile_defined(scope, namer, type_set);
                let right = right.transpile_defined(scope, namer, type_set);
                quote! { #left == #right }
            }
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
            Expression::ConstArrayRepeated { element, length } => {
                let element = element.transpile_defined(scope, namer, type_set);
                quote! {
                    [#element; #length]
                }
            }
            Expression::BooleanNot { value } => {
                let value = value.transpile_defined(scope, namer, type_set);
                quote! { !#value }
            }
            Expression::BooleanBinary {
                left,
                right,
                operator,
            } => {
                let left = left.transpile_defined(scope, namer, type_set);
                let right = right.transpile_defined(scope, namer, type_set);
                match *operator {
                    BooleanOperator::And => quote! { #left && #right },
                    BooleanOperator::Or => quote! { #left || #right },
                    BooleanOperator::Xor => quote! { #left ^ #right },
                }
            }
            Expression::Ternary {
                condition,
                true_value,
                false_value,
            } => {
                let condition = condition.transpile_defined(scope, namer, type_set);
                let true_value = true_value.transpile_defined(scope, namer, type_set);
                let false_value = false_value.transpile_defined(scope, namer, type_set);
                quote! { if #condition { #true_value } else { #false_value } }
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
        match self.expression.as_ref() {
            Expression::Variable {
                name,
                defines: true,
                ..
            } => {
                let name = namer.variable_name(scope, name);
                quote! {
                    #name = #this;
                }
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
                quote! {
                    if let #type_name(#(#names),*) = #this {
                        #(#insides)*
                    } else {
                        panic!();
                    }
                }
            }
            Expression::Dematerialized { value } => {
                value.transpile_top_down(scope, this, namer, type_set)
            }
            Expression::ConstArray { elements } => {
                let names = (0..elements.len())
                    .map(|_| namer.new_temporary_name())
                    .collect::<Vec<_>>();
                let insides = elements
                    .iter()
                    .zip_eq(names.iter())
                    .map(|(elem, name)| elem.transpile_top_down(scope, name, namer, type_set));
                quote! {
                    let [(#(#names),*)] = #this;
                    #(#insides)*
                }
            }
            _ => {
                let defined = self.transpile_defined(scope, namer, type_set);
                quote! {
                    assert_eq!(#this, #defined);
                }
            }
        }
    }
}

impl FlatStatement {
    pub fn transpile(
        &self,
        index: usize,
        program: &Stage2Program,
        namer: &mut VariableNamer,
    ) -> TokenStream {
        let type_set = &program.types;
        let scope = &self.scope;
        let material = self.material;
        let block = match &self.statement {
            Statement::VariableDeclaration { .. } => quote! {},
            Statement::Equality { left, right } => {
                let right = right.transpile_defined(scope, &namer, type_set);
                left.transpile_top_down(scope, &right, namer, type_set)
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
                    reference_expression.transpile_top_down(scope, &this, namer, type_set);
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
                    namer,
                    type_set,
                )
            }
            Statement::EmptyUnderConstructionArray { array, elem_type } => array
                .transpile_top_down(
                    scope,
                    &create_empty_under_construction_array(elem_type),
                    namer,
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
                    namer,
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
                    finalized_expression.transpile_top_down(scope, &this, namer, type_set);
                quote! {
                    #init
                    #following
                }
            }
            Statement::ArrayAccess { elem, array, index } => {
                let tipo = elem.get_type();
                let array = array.transpile_defined(scope, &namer, type_set);
                let index = index.transpile_defined(scope, &namer, type_set);
                elem.transpile_top_down(scope, &array_access(tipo, array, index), namer, type_set)
            }
        };
        namer.scoped(scope, block)
    }
}

impl FlatFunctionCall {
    pub fn transpile(
        &self,
        stage: Stage,
        index: usize,
        type_set: &TypeSet,
        program: &Stage2Program,
        namer: &mut VariableNamer,
    ) -> TokenStream {
        let mut block = vec![];

        let callee = &program.functions[&self.function_name];

        let callee_field = namer.callee_name(index);
        if stage.index == 0 {
            let struct_name = function_struct_name(&self.function_name);
            block.push(quote! {
                #callee_field = Box::new(Some(#struct_name::default()));
            });
        };

        for i in stage.start..stage.mid {
            let argument = &self.arguments[i].transpile_defined(&self.scope, &namer, type_set);
            let argument_field = name_field_according_to_scope(
                &ScopePath::empty(),
                callee.arguments[i].name.as_str(),
            );
            block.push(quote! {
                #callee_field.as_mut().as_mut().unwrap().#argument_field = #argument;
            });
        }
        block.push(execute_stage(&callee_field, stage.index));
        for i in stage.mid..stage.end {
            let argument_field = name_field_according_to_scope(
                &ScopePath::empty(),
                callee.arguments[i].name.as_str(),
            );
            block.push(self.arguments[i].transpile_top_down(
                &self.scope,
                &quote! { #callee_field.as_ref().as_ref().unwrap().#argument_field },
                namer,
                type_set,
            ));
        }

        namer.scoped(&self.scope, quote! { #(#block)* })
    }
}

impl FlatMatch {
    pub fn transpile(
        &self,
        _: usize,
        program: &Stage2Program,
        namer: &mut VariableNamer,
    ) -> TokenStream {
        let type_set = &program.types;
        let type_name = type_to_rust(self.value.get_type());

        let mut arms = vec![];
        for (constructor, components) in self.branches.iter() {
            let scope = self.scope.then(self.index, constructor.clone());
            let scope_name = namer.scope_name(&scope);
            let arm = if constructor == TRUE_CONSTRUCTOR_NAME {
                quote! { true => #scope_name = true, }
            } else if constructor == FALSE_CONSTRUCTOR_NAME {
                quote! { false => #scope_name = true, }
            } else {
                let temps = (0..components.len())
                    .map(|_| namer.new_temporary_name())
                    .collect::<Vec<_>>();
                let scope = self.scope.then(self.index, constructor.clone());
                let fields = components
                    .iter()
                    .map(|component| namer.variable_name(&scope, &component.name));
                let constructor = ident(&constructor);
                quote! {
                    #type_name::#constructor(#(#temps),*) => {
                        #(#fields = #temps;)*
                        #scope_name = true;
                    }
                }
            };
            arms.push(arm);
        }

        let value = self.value.transpile_defined(&self.scope, namer, type_set);
        namer.scoped(
            &self.scope,
            quote! {
                match #value {
                    #(#arms)*
                }
            },
        )
    }
}
