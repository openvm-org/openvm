use std::collections::HashMap;

use itertools::Itertools;

use crate::{
    core::{
        containers::{ExpressionContainer, RootContainer},
        error::CompilationError,
        file3::{
            DeclarationResolutionStep, FlatFunctionCall, FlatMatch, FlatStatement,
            FlattenedFunction,
        },
        ir::{
            ArgumentBehavior, ArithmeticOperator, Expression, Function, Material, StatementVariant,
            Type,
        },
        scope::ScopePath,
    },
    parser::metadata::ParserMetadata,
};

impl Type {
    pub fn check_exists(
        &mut self,
        root_container: &mut RootContainer,
        scope: &ScopePath,
        material: Material,
        parser_metadata: &ParserMetadata,
    ) -> Result<(), CompilationError> {
        match self {
            Type::Field => Ok(()),
            Type::NamedType(name) => {
                if root_container.type_set().algebraic_types.contains_key(name) {
                    Ok(())
                } else {
                    println!("oh no");
                    Err(CompilationError::UndefinedType(
                        parser_metadata.clone(),
                        name.clone(),
                    ))
                }
            }
            Type::Reference(inside) => {
                inside.check_exists(root_container, scope, material, parser_metadata)
            }
            Type::ReadablePrefix(inside, length) => {
                length.resolve_bottom_up(root_container, scope, material, false, true)?;
                inside.check_exists(root_container, scope, material, parser_metadata)
            }
            Type::AppendablePrefix(inside, length) => {
                length.resolve_bottom_up(root_container, scope, material, false, true)?;
                inside.check_exists(root_container, scope, material, parser_metadata)
            }
            Type::Unmaterialized(inside) => {
                inside.check_exists(root_container, scope, material, parser_metadata)
            }
            Type::ConstArray(inside, _) => {
                inside.check_exists(root_container, scope, material, parser_metadata)
            }
        }
    }
}

pub struct Substitutor {
    substitutions: HashMap<String, ExpressionContainer>,
}

impl Substitutor {
    pub fn new(callee: &Function, arguments: &Vec<ExpressionContainer>) -> Self {
        let mut substitutions = HashMap::new();
        for (abstract_argument, argument_specified) in
            callee.arguments.iter().zip_eq(arguments.iter())
        {
            substitutions.insert(abstract_argument.name.clone(), argument_specified.clone());
        }
        Substitutor { substitutions }
    }

    pub fn make_substitution(&self, tipo: &Type) -> Type {
        let mut tipo = tipo.clone();
        tipo.substitute(self);
        tipo
    }
}

impl Type {
    fn substitute(&mut self, substitutor: &Substitutor) {
        for child in self.children_mut() {
            child.substitute(substitutor);
        }
        match self {
            Type::AppendablePrefix(_, length) => length.substitute(substitutor),
            Type::ReadablePrefix(_, length) => length.substitute(substitutor),
            _ => {}
        }
    }
}

impl ExpressionContainer {
    fn substitute(&mut self, substitutor: &Substitutor) {
        match self.expression.as_mut() {
            Expression::Variable { name, .. } => {
                *self = substitutor.substitutions[name].clone();
            }
            Expression::EmptyConstArray { elem_type } => {
                elem_type.substitute(substitutor);
            }
            _ => {
                for child in self.children_mut() {
                    child.substitute(substitutor);
                }
            }
        }
    }
}

impl FlattenedFunction {
    pub fn resolve_types(
        &mut self,
        step: DeclarationResolutionStep,
        root_container: &mut RootContainer,
    ) -> Result<(), CompilationError> {
        match step {
            DeclarationResolutionStep::Match(flat_index) => {
                let FlatMatch {
                    check_material: material,
                    scope,
                    index,
                    value,
                    branches,
                    parser_metadata,
                } = &mut self.matches[flat_index];
                let material = *material;
                value.resolve_bottom_up(root_container, scope, material, true, true)?;
                let type_name = value
                    .get_type()
                    .get_named_type(material, &value.parser_metadata)?;
                let type_definition = root_container
                    .type_set
                    .get_algebraic_type(type_name, parser_metadata)?
                    .clone();
                for branch in branches {
                    let component_types = type_definition
                        .variants
                        .iter()
                        .find(|variant| variant.name == branch.constructor)
                        .ok_or(CompilationError::UndefinedConstructor(
                            branch.parser_metadata.clone(),
                            branch.constructor.clone(),
                        ))?
                        .components
                        .clone();
                    let new_path = scope.then(*index, branch.constructor.clone());
                    for (component, tipo) in branch.components.iter().zip_eq(component_types.iter())
                    {
                        root_container.declare(
                            &new_path,
                            &component.name,
                            material.wrap(tipo),
                            &branch.parser_metadata,
                        )?;
                    }
                }
            }
            DeclarationResolutionStep::Statement(index) => {
                let FlatStatement {
                    material,
                    scope: path,
                    statement,
                    parser_metadata,
                } = &mut self.statements[index];
                let material = *material;
                match statement {
                    StatementVariant::VariableDeclaration { name, tipo, .. } => {
                        tipo.check_exists(root_container, path, material, parser_metadata)?;
                        root_container.declare(path, name, tipo.clone(), parser_metadata)?;
                    }
                    StatementVariant::Equality { left, right } => {
                        right.resolve_bottom_up(root_container, path, material, false, true)?;
                        left.resolve_types_top_down(
                            right.get_type(),
                            root_container,
                            path,
                            material,
                        )?;
                    }
                    StatementVariant::Reference { reference, data } => {
                        data.resolve_bottom_up(root_container, path, material, false, true)?;
                        reference.resolve_types_top_down(
                            &Type::Reference(Box::new(data.get_type().clone())),
                            root_container,
                            path,
                            material,
                        )?;
                    }
                    StatementVariant::Dereference { data, reference } => {
                        reference.resolve_bottom_up(root_container, path, material, false, true)?;
                        data.resolve_types_top_down(
                            reference
                                .get_type()
                                .get_reference_type(material, &reference.parser_metadata)?,
                            root_container,
                            path,
                            material,
                        )?;
                    }
                    StatementVariant::EmptyPrefix {
                        prefix: array,
                        elem_type,
                    } => {
                        elem_type.check_exists(root_container, path, material, parser_metadata)?;
                        array.resolve_types_top_down(
                            &Type::AppendablePrefix(
                                Box::new(elem_type.clone()),
                                Box::new(ExpressionContainer::synthetic(Expression::Constant {
                                    value: 0,
                                })),
                            ),
                            root_container,
                            path,
                            material,
                        )?;
                    }
                    StatementVariant::PrefixAppend {
                        new_prefix: new_array,
                        elem,
                        old_prefix: old_array,
                    } => {
                        old_array.resolve_bottom_up(root_container, path, material, false, true)?;
                        elem.resolve_bottom_up(root_container, path, material, false, true)?;

                        let (elem_type, old_length) = old_array
                            .get_type()
                            .get_appendable_prefix_type(material, &old_array.parser_metadata)?;
                        root_container.assert_types_eq(
                            elem_type,
                            elem.get_type(),
                            path,
                            material,
                            &elem.parser_metadata,
                        )?;
                        let new_prefix_type = Type::AppendablePrefix(
                            Box::new(elem_type.clone()),
                            Box::new(ExpressionContainer::synthetic(Expression::Arithmetic {
                                operator: ArithmeticOperator::Plus,
                                left: old_length.clone(),
                                right: ExpressionContainer::synthetic(Expression::Constant {
                                    value: 1,
                                }),
                            })),
                        );

                        new_array.resolve_types_top_down(
                            &new_prefix_type,
                            root_container,
                            &path,
                            material,
                        )?;
                    }
                    StatementVariant::PrefixAccess {
                        elem,
                        prefix,
                        index,
                    } => {
                        index.resolve_bottom_up(root_container, path, material, false, true)?;
                        if !material.same_type(index.get_type(), &Type::Field) {
                            return Err(CompilationError::NotAnIndex(
                                elem.parser_metadata.clone(),
                                elem.get_type().clone(),
                            ));
                        }
                        prefix.resolve_bottom_up(root_container, path, material, false, true)?;
                        let (elem_type, _) = prefix
                            .get_type()
                            .get_readable_prefix_type(material, &prefix.parser_metadata)?;
                        elem.resolve_types_top_down(elem_type, root_container, &path, material)?;
                    }
                }
            }
            DeclarationResolutionStep::CallArgument(index, i) => {
                let FlatFunctionCall {
                    material,
                    scope,
                    function_name,
                    arguments,
                    parser_metadata,
                } = &mut self.function_calls[index];
                let material = *material;

                let callee = root_container
                    .function_set
                    .get_function(function_name, parser_metadata)?
                    .clone();

                let inline = callee.function.inline;
                let substitutor = Substitutor::new(&callee.function, arguments);

                match callee.function.arguments[i].behavior {
                    ArgumentBehavior::In => {
                        let argument = &mut arguments[i];
                        argument.resolve_bottom_up(
                            root_container,
                            scope,
                            material,
                            inline,
                            true,
                        )?;
                        root_container.assert_types_eq(
                            argument.get_type(),
                            &substitutor.make_substitution(callee.argument_type(i)),
                            scope,
                            material,
                            &argument.parser_metadata,
                        )?;
                    }
                    ArgumentBehavior::Out => {
                        arguments[i].resolve_types_top_down(
                            &substitutor.make_substitution(callee.argument_type(i)),
                            root_container,
                            scope,
                            material,
                        )?;
                    }
                }
            }
            DeclarationResolutionStep::OwnArgument(i) => {
                let argument = &self.arguments[i];
                root_container.declare(
                    &ScopePath::empty(),
                    &argument.name,
                    argument.tipo.clone(),
                    &argument.parser_metadata,
                )?;
            }
        }
        Ok(())
    }
}
