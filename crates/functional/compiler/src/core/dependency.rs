use itertools::Itertools;

use crate::{
    core::{
        containers::{ExpressionContainer, RootContainer},
        error::CompilationError,
        file3::{Atom, DeclarationResolutionStep, FlattenedFunction},
        ir::{Expression, StatementVariant, Type},
        scope::ScopePath,
    },
    parser::metadata::ParserMetadata,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DependencyType {
    Declaration,
    Definition,
    Representation,
}

impl ExpressionContainer {
    pub fn new(expression: Expression) -> Self {
        Self {
            expression: Box::new(expression),
            tipo: None,
            parser_metadata: Default::default(),
        }
    }

    pub fn children(&self) -> Vec<&ExpressionContainer> {
        match self.expression.as_ref() {
            Expression::Constant { .. } => vec![],
            Expression::Variable { .. } => vec![],
            Expression::Algebraic { fields, .. } => fields.iter().collect(),
            Expression::Arithmetic { left, right, .. } => vec![left, right],
            Expression::Dematerialized { value } => vec![value],
            Expression::ReadableViewOfPrefix {
                appendable_prefix: appendable,
            } => vec![appendable],
            Expression::ConstArray { elements } => elements.iter().collect(),
            Expression::ConstArrayAccess { array, .. } => vec![array],
            Expression::ConstArraySlice { array, .. } => vec![array],
            Expression::ConstArrayConcatenation { left, right } => vec![left, right],
            Expression::Eq { left, right } => vec![left, right],
            Expression::EmptyConstArray { .. } => vec![],
            Expression::ConstArrayRepeated { element, .. } => vec![element],
            Expression::BooleanNot { value } => vec![value],
            Expression::BooleanBinary { left, right, .. } => vec![left, right],
            Expression::Ternary {
                condition,
                true_value,
                false_value,
            } => vec![condition, true_value, false_value],
        }
    }

    pub fn children_mut(&mut self) -> Vec<&mut ExpressionContainer> {
        match self.expression.as_mut() {
            Expression::Constant { .. } => vec![],
            Expression::Variable { .. } => vec![],
            Expression::Algebraic { fields, .. } => fields.iter_mut().collect(),
            Expression::Arithmetic { left, right, .. } => vec![left, right],
            Expression::Dematerialized { value } => vec![value],
            Expression::ReadableViewOfPrefix {
                appendable_prefix: appendable,
            } => vec![appendable],
            Expression::ConstArray { elements } => elements.iter_mut().collect(),
            Expression::ConstArrayAccess { array, .. } => vec![array],
            Expression::ConstArraySlice { array, .. } => vec![array],
            Expression::ConstArrayConcatenation { left, right } => vec![left, right],
            Expression::Eq { left, right } => vec![left, right],
            Expression::EmptyConstArray { .. } => vec![],
            Expression::ConstArrayRepeated { element, .. } => vec![element],
            Expression::BooleanNot { value } => vec![value],
            Expression::BooleanBinary { left, right, .. } => vec![left, right],
            Expression::Ternary {
                condition,
                true_value,
                false_value,
            } => vec![condition, true_value, false_value],
        }
    }

    pub fn dependencies(&self, dependency_type: DependencyType) -> Vec<String> {
        match self.expression.as_ref() {
            Expression::Variable {
                name,
                declares,
                defines,
                represents,
            } => {
                let all_good = match dependency_type {
                    DependencyType::Declaration => *declares,
                    DependencyType::Definition => *defines,
                    DependencyType::Representation => *represents,
                };
                if all_good {
                    vec![]
                } else {
                    vec![name.clone()]
                }
            }
            Expression::EmptyConstArray { elem_type } => {
                if dependency_type == DependencyType::Declaration {
                    elem_type.variable_dependencies()
                } else {
                    vec![]
                }
            }
            _ => self
                .children()
                .iter()
                .flat_map(|child| child.dependencies(dependency_type))
                .collect(),
        }
    }

    pub fn does_representing(&self) -> bool {
        match self.expression.as_ref() {
            Expression::Variable {
                represents: true, ..
            } => true,
            _ => self
                .children()
                .iter()
                .any(|child| child.does_representing()),
        }
    }
}

impl Type {
    pub fn children(&self) -> Vec<&Type> {
        match self {
            Type::Field => vec![],
            Type::NamedType(_) => vec![],
            Type::Reference(inside) => vec![inside],
            Type::AppendablePrefix(elem_type, _) => vec![elem_type],
            Type::ReadablePrefix(elem_type, _) => vec![elem_type],
            Type::Unmaterialized(inside) => vec![inside],
            Type::ConstArray(elem_type, _) => vec![elem_type],
        }
    }
    pub fn children_mut(&mut self) -> Vec<&mut Type> {
        match self {
            Type::Field => vec![],
            Type::NamedType(_) => vec![],
            Type::Reference(inside) => vec![inside],
            Type::AppendablePrefix(elem_type, _) => vec![elem_type],
            Type::ReadablePrefix(elem_type, _) => vec![elem_type],
            Type::Unmaterialized(inside) => vec![inside],
            Type::ConstArray(elem_type, _) => vec![elem_type],
        }
    }

    pub fn variable_dependencies(&self) -> Vec<String> {
        let mut dependencies: Vec<String> = self
            .children()
            .iter()
            .flat_map(|child| child.variable_dependencies())
            .collect();

        match self {
            Type::AppendablePrefix(_, length) => {
                dependencies.extend(length.dependencies(DependencyType::Declaration))
            }
            Type::ReadablePrefix(_, length) => {
                dependencies.extend(length.dependencies(DependencyType::Declaration))
            }
            _ => {}
        }

        dependencies
    }
}

pub struct DeclaredName {
    pub scope: ScopePath,
    pub defines: bool,
    pub represents: bool,
    pub name: String,
    pub parser_metadata: ParserMetadata,
}

impl FlattenedFunction {
    pub fn scope(&self, atom: Atom) -> &ScopePath {
        match atom {
            Atom::Match(index) => &self.matches[index].scope,
            Atom::Statement(index) => &self.statements[index].scope,
            Atom::PartialFunctionCall(index, _) => &self.function_calls[index].scope,
        }
    }
    pub fn viable_for_declaration(
        &self,
        root_container: &RootContainer,
        step: DeclarationResolutionStep,
    ) -> Result<bool, CompilationError> {
        let scope = match step {
            DeclarationResolutionStep::Statement(index) => &self.statements[index].scope,
            DeclarationResolutionStep::Match(index) => &self.matches[index].scope,
            DeclarationResolutionStep::OwnArgument(_) => &ScopePath::empty(),
            DeclarationResolutionStep::CallArgument(index, _) => &self.function_calls[index].scope,
        };
        let dependencies = match step {
            DeclarationResolutionStep::OwnArgument(argument_index) => {
                self.arguments[argument_index].tipo.variable_dependencies()
            }
            DeclarationResolutionStep::CallArgument(index, argument_index) => {
                let call = &self.function_calls[index];
                let callee = root_container
                    .function_set
                    .get_function(&call.function_name, &call.parser_metadata)?
                    .clone();
                call.arguments[argument_index]
                    .dependencies(DependencyType::Declaration)
                    .into_iter()
                    .chain(callee.argument_type(argument_index).variable_dependencies())
                    .collect()
            }
            _ => {
                let atom = match step {
                    DeclarationResolutionStep::Statement(index) => Atom::Statement(index),
                    DeclarationResolutionStep::Match(index) => Atom::Match(index),
                    _ => unreachable!(),
                };
                self.children(atom)
                    .into_iter()
                    .flat_map(|child| child.dependencies(DependencyType::Declaration))
                    .chain(
                        self.contained_types(atom)
                            .into_iter()
                            .flat_map(Type::variable_dependencies),
                    )
                    .collect()
            }
        };
        Ok(dependencies
            .iter()
            .all(|dependency| root_container.root_scope.is_declared(scope, &dependency)))
    }
    pub fn viable_for_definition(&self, root_container: &RootContainer, atom: Atom) -> bool {
        let scope = self.scope(atom);
        root_container.root_scope.scope_is_active(scope)
            && self
                .children(atom)
                .into_iter()
                .flat_map(|child| child.dependencies(DependencyType::Definition))
                .all(|dependency| root_container.root_scope.is_defined(scope, &dependency))
            && self.declared_names(atom).into_iter().all(|declared_name| {
                declared_name.defines
                    || root_container
                        .root_scope
                        .is_defined(&declared_name.scope, &declared_name.name)
            })
    }
    pub fn viable_for_representation(&self, root_container: &RootContainer, atom: Atom) -> bool {
        let scope = self.scope(atom);
        self.children(atom)
            .into_iter()
            .flat_map(|child| child.dependencies(DependencyType::Representation))
            .all(|dependency| {
                root_container.root_scope.is_represented(
                    scope,
                    &dependency,
                    &root_container.type_set,
                )
            })
            && self.declared_names(atom).into_iter().all(|declared_name| {
                declared_name.represents
                    || root_container.root_scope.is_represented(
                        &declared_name.scope,
                        &declared_name.name,
                        &root_container.type_set,
                    )
            })
    }
    pub fn children(&self, atom: Atom) -> Vec<&ExpressionContainer> {
        match atom {
            Atom::Match(index) => vec![&self.matches[index].value],
            Atom::PartialFunctionCall(index, stage) => {
                let call = &self.function_calls[index];
                (stage.start..stage.end)
                    .map(|i| &call.arguments[i])
                    .collect()
            }
            Atom::Statement(index) => match &self.statements[index].statement {
                StatementVariant::VariableDeclaration { .. } => vec![],
                StatementVariant::Equality { left, right } => vec![left, right],
                StatementVariant::Reference { reference, data } => vec![reference, data],
                StatementVariant::Dereference { data, reference } => vec![data, reference],
                StatementVariant::EmptyPrefix { prefix, .. } => vec![prefix],
                StatementVariant::PrefixAppend {
                    new_prefix,
                    old_prefix,
                    elem,
                } => vec![new_prefix, old_prefix, elem],
                StatementVariant::PrefixAccess {
                    elem,
                    prefix,
                    index,
                } => vec![elem, prefix, index],
            },
        }
    }
    pub fn declared_names(&self, atom: Atom) -> Vec<DeclaredName> {
        let scope = self.scope(atom);
        match atom {
            Atom::Match(index) => {
                let matchi = &self.matches[index];
                matchi
                    .branches
                    .iter()
                    .flat_map(|branch| {
                        let scope = scope.then(matchi.index, branch.constructor.clone());
                        branch
                            .components
                            .iter()
                            .map(|component| DeclaredName {
                                scope: scope.clone(),
                                defines: true,
                                represents: component.represents,
                                name: component.name.clone(),
                                parser_metadata: branch.parser_metadata.clone(),
                            })
                            .collect_vec()
                    })
                    .collect()
            }
            Atom::Statement(index) => match &self.statements[index].statement {
                StatementVariant::VariableDeclaration {
                    name, represents, ..
                } => vec![DeclaredName {
                    scope: scope.clone(),
                    defines: false,
                    represents: *represents,
                    name: name.clone(),
                    parser_metadata: self.statements[index].parser_metadata.clone(),
                }],
                _ => vec![],
            },
            Atom::PartialFunctionCall(..) => vec![],
        }
    }
    pub fn contained_types(&self, atom: Atom) -> Vec<&Type> {
        if let Atom::Statement(index) = atom {
            match &self.statements[index].statement {
                StatementVariant::VariableDeclaration { tipo, .. } => vec![tipo],
                StatementVariant::EmptyPrefix { elem_type, .. } => vec![elem_type],
                _ => vec![],
            }
        } else {
            vec![]
        }
    }
}
