use itertools::Itertools;

use super::{
    error::CompilationError,
    ir::{Expression, Type},
};
use crate::core::{
    containers::{ExpressionContainer, RootContainer},
    ir::{ArithmeticOperator, Material},
    scope::ScopePath,
};

impl ExpressionContainer {
    pub fn resolve_bottom_up(
        &mut self,
        function_container: &mut RootContainer,
        path: &ScopePath,
        material: Material,
        can_represent: bool,
        consuming_prefix: bool,
    ) -> Result<(), CompilationError> {
        match self.expression.as_mut() {
            Expression::Constant { .. } => {
                self.tipo = Some(Type::Field);
            }
            Expression::Variable {
                name,
                declares,
                defines,
                represents,
            } => {
                if *defines {
                    return Err(CompilationError::CannotAssignHere(
                        self.parser_metadata.clone(),
                        name.clone(),
                    ));
                }
                if *represents && !can_represent {
                    return Err(CompilationError::CannotRepresentHere(
                        self.parser_metadata.clone(),
                        name.clone(),
                    ));
                }
                assert!(!*declares);
                self.tipo = Some(function_container.get_declaration_type(
                    path,
                    name,
                    &self.parser_metadata,
                )?);
                if consuming_prefix
                    && self
                        .tipo
                        .as_ref()
                        .unwrap()
                        .contains_appendable_prefix(&function_container.type_set())
                {
                    function_container.consume_appendable_prefix(
                        path,
                        name,
                        &self.parser_metadata,
                    )?;
                }
            }
            Expression::Algebraic {
                constructor,
                fields,
            } => {
                let expected_component_types = function_container
                    .type_set()
                    .get_component_types(constructor, &self.parser_metadata)?;
                if fields.len() != expected_component_types.len() {
                    return Err(CompilationError::IncorrectNumberOfComponents(
                        self.parser_metadata.clone(),
                        constructor.clone(),
                        fields.len(),
                        expected_component_types.len(),
                    ));
                }
                for (field, expected_type) in
                    fields.iter_mut().zip_eq(expected_component_types.iter())
                {
                    field.resolve_bottom_up(
                        function_container,
                        path,
                        material,
                        can_represent,
                        consuming_prefix,
                    )?;
                    function_container.assert_types_eq(
                        field.get_type(),
                        expected_type,
                        path,
                        material,
                        &field.parser_metadata,
                    )?;
                }

                self.tipo = Some(
                    function_container
                        .type_set()
                        .get_constructor_type(constructor, &self.parser_metadata)?,
                );
            }
            Expression::Arithmetic {
                operator,
                left,
                right,
            } => {
                if *operator == ArithmeticOperator::Div && material == Material::Materialized {
                    return Err(CompilationError::DivMustBeDematerialized(
                        self.parser_metadata.clone(),
                    ));
                }
                left.resolve_bottom_up(
                    function_container,
                    path,
                    material,
                    false,
                    consuming_prefix,
                )?;
                if !material.same_type(&left.tipo.clone().unwrap(), &Type::Field) {
                    return Err(CompilationError::IncorrectTypeInArithmetic(
                        left.parser_metadata.clone(),
                        left.tipo.clone().unwrap(),
                    ));
                }
                right.resolve_bottom_up(
                    function_container,
                    path,
                    material,
                    false,
                    consuming_prefix,
                )?;
                if !material.same_type(&right.tipo.clone().unwrap(), &Type::Field) {
                    return Err(CompilationError::IncorrectTypeInArithmetic(
                        right.parser_metadata.clone(),
                        right.tipo.clone().unwrap(),
                    ));
                }
                self.tipo = Some(Type::Field);
            }
            Expression::Dematerialized { value } => {
                value.resolve_bottom_up(
                    function_container,
                    path,
                    Material::Dematerialized,
                    can_represent,
                    consuming_prefix,
                )?;
                self.tipo = Some(Type::Unmaterialized(Box::new(value.tipo.clone().unwrap())));
            }
            Expression::ReadableViewOfPrefix { appendable_prefix } => {
                appendable_prefix.resolve_bottom_up(
                    function_container,
                    path,
                    Material::Dematerialized,
                    can_represent,
                    false,
                )?;
                let (elem_type, length) = appendable_prefix
                    .get_type()
                    .get_appendable_prefix_type(material, &self.parser_metadata)?;
                self.tipo = Some(Type::ReadablePrefix(
                    Box::new(elem_type.clone()),
                    Box::new(length.clone()),
                ));
            }
            Expression::Eq { left, right } => {
                if material == Material::Materialized {
                    return Err(CompilationError::EqMustBeDematerialized(
                        self.parser_metadata.clone(),
                    ));
                }
                left.resolve_bottom_up(
                    function_container,
                    path,
                    Material::Dematerialized,
                    false,
                    consuming_prefix,
                )?;
                right.resolve_bottom_up(
                    function_container,
                    path,
                    Material::Dematerialized,
                    false,
                    consuming_prefix,
                )?;
                function_container.assert_types_eq(
                    left.get_type(),
                    right.get_type(),
                    path,
                    Material::Dematerialized,
                    &self.parser_metadata,
                )?;
                if left
                    .get_type()
                    .contains_reference(&function_container.type_set, false)
                {
                    return Err(CompilationError::CannotEquateReferences(
                        self.parser_metadata.clone(),
                        left.get_type().clone(),
                    ));
                }
                self.tipo = Some(Type::boolean());
            }
            Expression::EmptyConstArray { elem_type } => {
                elem_type.check_exists(
                    function_container,
                    path,
                    material,
                    &self.parser_metadata,
                )?;
                self.tipo = Some(Type::ConstArray(Box::new(elem_type.clone()), 0));
            }
            Expression::ConstArray { elements } => {
                if elements.is_empty() {
                    return Err(CompilationError::CannotInferEmptyConstArrayType(
                        self.parser_metadata.clone(),
                    ));
                }
                for element in elements.iter_mut() {
                    element.resolve_bottom_up(
                        function_container,
                        path,
                        material,
                        can_represent,
                        consuming_prefix,
                    )?;
                }
                let elem_type = elements[0].tipo.clone().unwrap();
                for element in elements.iter() {
                    function_container.assert_types_eq(
                        &elem_type,
                        element.get_type(),
                        path,
                        material,
                        &element.parser_metadata,
                    )?;
                }
                self.tipo = Some(Type::ConstArray(Box::new(elem_type), elements.len()));
            }
            Expression::ConstArrayConcatenation { left, right } => {
                left.resolve_bottom_up(
                    function_container,
                    path,
                    material,
                    false,
                    consuming_prefix,
                )?;
                right.resolve_bottom_up(
                    function_container,
                    path,
                    material,
                    false,
                    consuming_prefix,
                )?;
                let (elem_type_1, len_1) = left
                    .get_type()
                    .get_const_array_type(material, &left.parser_metadata)?;
                let (elem_type_2, len_2) = right
                    .get_type()
                    .get_const_array_type(material, &right.parser_metadata)?;
                function_container.assert_types_eq(
                    elem_type_1,
                    elem_type_2,
                    path,
                    material,
                    &right.parser_metadata,
                )?;
                /*
                return Err(
                        CompilationError::ElementTypesInconsistentInConstArrayConcatenation(
                            self.parser_metadata.clone(),
                            elem_type_1.clone(),
                            elem_type_2.clone(),
                        ),
                    );
                 */
                self.tipo = Some(Type::ConstArray(elem_type_1.clone().into(), len_1 + len_2));
            }
            Expression::ConstArrayAccess { array, index } => {
                array.resolve_bottom_up(
                    function_container,
                    path,
                    material,
                    false,
                    consuming_prefix,
                )?;
                let (elem_type, len) = array
                    .get_type()
                    .get_const_array_type(material, &array.parser_metadata)?;
                if *index >= len {
                    return Err(CompilationError::OutOfBoundsConstArrayAccess(
                        self.parser_metadata.clone(),
                        *index,
                        len,
                    ));
                }
                self.tipo = Some(elem_type.clone());
            }
            Expression::ConstArraySlice { array, from, to } => {
                array.resolve_bottom_up(
                    function_container,
                    path,
                    material,
                    false,
                    consuming_prefix,
                )?;
                let (elem_type, len) = array
                    .get_type()
                    .get_const_array_type(material, &array.parser_metadata)?;
                if *from > *to || *to > len {
                    return Err(CompilationError::OutOfBoundsConstArraySlice(
                        self.parser_metadata.clone(),
                        *from,
                        *to,
                        len,
                    ));
                }
                self.tipo = Some(Type::ConstArray(elem_type.clone().into(), *to - *from));
            }
            Expression::ConstArrayRepeated { element, length } => {
                element.resolve_bottom_up(
                    function_container,
                    path,
                    material,
                    false,
                    consuming_prefix,
                )?;
                let elem_type = element.get_type();
                if elem_type.contains_appendable_prefix(&function_container.type_set) {
                    return Err(
                        CompilationError::DuplicateUnderConstructionArrayConsumptionInConstArray(
                            element.parser_metadata.clone(),
                        ),
                    );
                }

                self.tipo = Some(Type::ConstArray(Box::new(elem_type.clone()), *length));
            }
            Expression::BooleanNot { value } => {
                value.resolve_bottom_up(
                    function_container,
                    path,
                    material,
                    false,
                    consuming_prefix,
                )?;
                if !material.same_type(value.get_type(), &Type::boolean()) {
                    return Err(CompilationError::ExpectedBoolean(
                        self.parser_metadata.clone(),
                        value.get_type().clone(),
                    ));
                }
                self.tipo = Some(Type::boolean());
            }
            Expression::BooleanBinary { left, right, .. } => {
                left.resolve_bottom_up(
                    function_container,
                    path,
                    material,
                    false,
                    consuming_prefix,
                )?;
                if !material.same_type(left.get_type(), &Type::boolean()) {
                    return Err(CompilationError::ExpectedBoolean(
                        left.parser_metadata.clone(),
                        left.get_type().clone(),
                    ));
                }
                right.resolve_bottom_up(
                    function_container,
                    path,
                    material,
                    false,
                    consuming_prefix,
                )?;
                if !material.same_type(right.get_type(), &Type::boolean()) {
                    return Err(CompilationError::ExpectedBoolean(
                        right.parser_metadata.clone(),
                        right.get_type().clone(),
                    ));
                }
                self.tipo = Some(Type::boolean());
            }
            Expression::Ternary {
                condition,
                true_value,
                false_value,
            } => {
                condition.resolve_bottom_up(
                    function_container,
                    path,
                    material,
                    false,
                    consuming_prefix,
                )?;
                if !material.same_type(condition.get_type(), &Type::boolean()) {
                    return Err(CompilationError::ExpectedBoolean(
                        condition.parser_metadata.clone(),
                        condition.get_type().clone(),
                    ));
                }
                true_value.resolve_bottom_up(
                    function_container,
                    path,
                    material,
                    false,
                    consuming_prefix,
                )?;
                false_value.resolve_bottom_up(
                    function_container,
                    path,
                    material,
                    false,
                    consuming_prefix,
                )?;
                function_container.assert_types_eq(
                    true_value.get_type(),
                    false_value.get_type(),
                    path,
                    material,
                    &self.parser_metadata,
                )?;
                self.tipo = Some(true_value.get_type().clone());
            }
        }
        Ok(())
    }

    pub fn resolve_types_top_down(
        &mut self,
        expected_type: &Type,
        function_container: &mut RootContainer,
        path: &ScopePath,
        material: Material,
    ) -> Result<(), CompilationError> {
        match self.expression.as_mut() {
            Expression::Variable {
                name,
                declares: true,
                defines,
                ..
            } => {
                assert!(*defines);
                let declaration_type = match material {
                    Material::Materialized => expected_type.clone(),
                    Material::Dematerialized => {
                        Type::Unmaterialized(Box::new(expected_type.clone()))
                    }
                };
                function_container.declare(path, name, declaration_type, &self.parser_metadata)?;
            }
            Expression::Variable {
                name,
                declares: false,
                defines: true,
                ..
            } => {
                function_container.assert_types_eq(
                    &function_container.get_declaration_type(path, name, &self.parser_metadata)?,
                    expected_type,
                    path,
                    material,
                    &self.parser_metadata,
                )?;
                if material == Material::Dematerialized
                    && expected_type.contains_reference(&function_container.type_set(), true)
                {
                    return Err(CompilationError::ReferenceDefinitionMustBeMaterialized(
                        self.parser_metadata.clone(),
                        name.clone(),
                    ));
                }
            }
            Expression::Algebraic {
                constructor,
                fields,
            } => {
                let expected_component_types = function_container
                    .type_set()
                    .get_component_types(constructor, &self.parser_metadata)?;
                if fields.len() != expected_component_types.len() {
                    return Err(CompilationError::IncorrectNumberOfComponents(
                        self.parser_metadata.clone(),
                        constructor.clone(),
                        fields.len(),
                        expected_component_types.len(),
                    ));
                }
                for (field, expected_field_type) in
                    fields.iter_mut().zip_eq(expected_component_types.iter())
                {
                    field.resolve_types_top_down(
                        expected_field_type,
                        function_container,
                        path,
                        material,
                    )?;
                }

                function_container.assert_types_eq(
                    &function_container
                        .type_set()
                        .get_constructor_type(constructor, &self.parser_metadata)?,
                    expected_type,
                    path,
                    material,
                    &self.parser_metadata,
                )?;
            }
            Expression::ConstArray { elements } => {
                let (elem_type, len) =
                    expected_type.get_const_array_type(material, &self.parser_metadata)?;
                if elements.len() != len {
                    return Err(CompilationError::IncorrectNumberOfElementsInConstArray(
                        self.parser_metadata.clone(),
                        elements.len(),
                        len,
                    ));
                }
                for element in elements.iter_mut() {
                    element.resolve_types_top_down(
                        elem_type,
                        function_container,
                        path,
                        material,
                    )?;
                }
            }
            Expression::Dematerialized { value } => {
                if material == Material::Materialized {
                    match expected_type {
                        Type::Unmaterialized(_) => {}
                        _ => {
                            return Err(CompilationError::UnexpectedUnmaterialized(
                                self.parser_metadata.clone(),
                                expected_type.clone(),
                            ))
                        }
                    }
                }
                value.resolve_types_top_down(
                    expected_type,
                    function_container,
                    path,
                    Material::Dematerialized,
                )?;
            }
            _ => {
                if expected_type.contains_reference(&function_container.type_set(), false) {
                    return Err(CompilationError::CannotEquateReferences(
                        self.parser_metadata.clone(),
                        expected_type.clone(),
                    ));
                }
                self.resolve_bottom_up(function_container, path, material, true, true)?;
                function_container.assert_types_eq(
                    self.get_type(),
                    &expected_type,
                    path,
                    material,
                    &self.parser_metadata,
                )?;
            }
        }

        self.tipo = Some(expected_type.clone());
        Ok(())
    }
}
impl ExpressionContainer {
    pub fn resolve_definition(
        &self,
        function_container: &mut RootContainer,
        path: &ScopePath,
    ) -> Result<(), CompilationError> {
        self.resolve_definition_or_representation(function_container, path, true)
    }

    pub fn resolve_representation(
        &self,
        function_container: &mut RootContainer,
        path: &ScopePath,
    ) -> Result<(), CompilationError> {
        self.resolve_definition_or_representation(function_container, path, false)
    }

    fn resolve_definition_or_representation(
        &self,
        function_container: &mut RootContainer,
        path: &ScopePath,
        definition: bool,
    ) -> Result<(), CompilationError> {
        match self.expression.as_ref() {
            Expression::Variable {
                name,
                declares: _,
                defines,
                represents,
            } => match definition {
                true => {
                    if *defines {
                        function_container.define(path, name, &self.parser_metadata)?;
                    }
                }
                false => {
                    if *represents {
                        function_container.represent(path, name, &self.parser_metadata)?;
                    }
                }
            },
            Expression::Algebraic { fields, .. } => {
                for field in fields {
                    field.resolve_definition_or_representation(
                        function_container,
                        path,
                        definition,
                    )?;
                }
            }
            Expression::ConstArray { elements } => {
                for element in elements {
                    element.resolve_definition_or_representation(
                        function_container,
                        path,
                        definition,
                    )?;
                }
            }
            Expression::Dematerialized { value } => {
                value.resolve_definition_or_representation(function_container, path, definition)?;
            }
            _ => {}
        }
        Ok(())
    }
}
