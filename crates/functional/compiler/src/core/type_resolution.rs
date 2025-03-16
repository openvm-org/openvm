use std::collections::{HashMap, HashSet};

use super::{
    error::CompilationError,
    ir::{AlgebraicTypeDeclaration, Type},
};
use crate::{
    core::{containers::ExpressionContainer, ir::Material},
    parser::metadata::ParserMetadata,
};

#[derive(Clone, Debug)]
pub struct TypeSet {
    pub type_order: Vec<String>,
    pub algebraic_types: HashMap<String, AlgebraicTypeDeclaration>,
    pub constructors: HashMap<String, (String, Vec<Type>)>,
}

impl TypeSet {
    pub fn new(
        mut type_declarations: Vec<AlgebraicTypeDeclaration>,
    ) -> Result<Self, CompilationError> {
        let mut type_order = Vec::new();
        let mut defined_types = HashSet::new();
        let mut types = HashMap::new();
        let mut constructors = HashMap::new();

        while !type_declarations.is_empty() {
            let i = type_declarations
                .iter()
                .position(|type_declaration| {
                    type_declaration.variants.iter().all(|variant| {
                        variant.components.iter().all(|component| {
                            component
                                .size_dependencies()
                                .iter()
                                .all(|dependency| defined_types.contains(dependency))
                        })
                    })
                })
                .ok_or(CompilationError::TypesSelfReferential)?;
            let type_declaration = type_declarations.remove(i);
            if type_declaration.variants.is_empty() {
                return Err(CompilationError::AlgebraicTypeCannotBeEmpty(
                    type_declaration.parser_metadata.clone(),
                    type_declaration.name.clone(),
                ));
            }
            type_order.push(type_declaration.name.clone());
            defined_types.insert(type_declaration.name.clone());
            for variant in type_declaration.variants.iter() {
                if constructors.contains_key(&variant.name) {
                    return Err(CompilationError::DuplicateConstructorName(
                        type_declaration.parser_metadata.clone(),
                        variant.name.clone(),
                    ));
                }
                constructors.insert(
                    variant.name.clone(),
                    (type_declaration.name.clone(), variant.components.clone()),
                );
            }
            if types.contains_key(&type_declaration.name) {
                return Err(CompilationError::DuplicateTypeName(
                    type_declaration.parser_metadata.clone(),
                    type_declaration.name.clone(),
                ));
            }
            if type_declaration.name == "F" {
                return Err(CompilationError::TypeNameCannotBeF(
                    type_declaration.parser_metadata.clone(),
                ));
            }
            types.insert(type_declaration.name.clone(), type_declaration);
        }
        Ok(Self {
            type_order,
            algebraic_types: types,
            constructors,
        })
    }
    pub fn get_algebraic_type(
        &self,
        name: &String,
        parser_metadata: &ParserMetadata,
    ) -> Result<&AlgebraicTypeDeclaration, CompilationError> {
        if let Some(algebraic_type) = self.algebraic_types.get(name) {
            Ok(algebraic_type)
        } else {
            Err(CompilationError::UndefinedType(
                parser_metadata.clone(),
                name.clone(),
            ))
        }
    }
    pub fn get_component_types(
        &self,
        constructor: &String,
        parser_metadata: &ParserMetadata,
    ) -> Result<Vec<Type>, CompilationError> {
        if let Some((_, components)) = self.constructors.get(constructor) {
            Ok(components.to_vec())
        } else {
            Err(CompilationError::UndefinedConstructor(
                parser_metadata.clone(),
                constructor.clone(),
            ))
        }
    }

    pub fn get_constructor_type_name(
        &self,
        constructor: &String,
        parser_metadata: &ParserMetadata,
    ) -> Result<String, CompilationError> {
        if let Some((name, _)) = self.constructors.get(constructor) {
            Ok(name.clone())
        } else {
            Err(CompilationError::UndefinedConstructor(
                parser_metadata.clone(),
                constructor.clone(),
            ))
        }
    }

    pub fn get_constructor_type(
        &self,
        constructor: &String,
        parser_metadata: &ParserMetadata,
    ) -> Result<Type, CompilationError> {
        Ok(Type::NamedType(
            self.get_constructor_type_name(constructor, parser_metadata)?,
        ))
    }
}

impl Type {
    pub fn size_dependencies(&self) -> Vec<String> {
        match self {
            Type::NamedType(name) => vec![name.clone()],
            Type::Unmaterialized(inside) => inside.size_dependencies(),
            Type::ConstArray(inside, _) => inside.size_dependencies(),
            _ => vec![],
        }
    }
    pub fn contains_reference(&self, type_set: &TypeSet, only_materialized: bool) -> bool {
        match self {
            Type::Reference(_) => true,
            Type::ReadablePrefix(..) => true,
            Type::AppendablePrefix(..) => true,
            Type::Field => false,
            Type::NamedType(name) => {
                let algebraic_type = type_set
                    .get_algebraic_type(name, &ParserMetadata::default())
                    .unwrap();
                for variant in algebraic_type.variants.iter() {
                    for component in variant.components.iter() {
                        if component.contains_reference(type_set, only_materialized) {
                            return true;
                        }
                    }
                }
                false
            }
            Type::ConstArray(inside, _) => inside.contains_reference(type_set, only_materialized),
            Type::Unmaterialized(inside) => {
                !only_materialized && inside.contains_reference(type_set, false)
            }
        }
    }

    pub fn contains_appendable_prefix(&self, type_set: &TypeSet) -> bool {
        match self {
            Type::AppendablePrefix(..) => true,
            Type::Reference(value) => value.contains_appendable_prefix(type_set),
            Type::ReadablePrefix(elem, _) => elem.contains_appendable_prefix(type_set),
            Type::Field => false,
            Type::NamedType(name) => {
                let algebraic_type = type_set
                    .get_algebraic_type(name, &ParserMetadata::default())
                    .unwrap();
                for variant in algebraic_type.variants.iter() {
                    for component in variant.components.iter() {
                        if component.contains_appendable_prefix(type_set) {
                            return true;
                        }
                    }
                }
                false
            }
            Type::ConstArray(inside, _) => inside.contains_appendable_prefix(type_set),
            Type::Unmaterialized(inside) => inside.contains_appendable_prefix(type_set),
        }
    }

    pub fn get_reference_type(
        &self,
        material: Material,
        parser_metadata: &ParserMetadata,
    ) -> Result<&Type, CompilationError> {
        match (self, material) {
            (Type::Reference(value), _) => Ok(value),
            (Type::Unmaterialized(inside), Material::Dematerialized) => {
                inside.get_reference_type(material, parser_metadata)
            }
            _ => Err(CompilationError::NotAReference(
                parser_metadata.clone(),
                self.clone(),
            )),
        }
    }

    pub fn get_appendable_prefix_type(
        &self,
        material: Material,
        parser_metadata: &ParserMetadata,
    ) -> Result<(&Type, &ExpressionContainer), CompilationError> {
        match (self, material) {
            (Type::AppendablePrefix(elem_type, length), _) => Ok((elem_type, length)),
            (Type::Unmaterialized(inside), Material::Dematerialized) => {
                inside.get_appendable_prefix_type(material, parser_metadata)
            }
            _ => Err(CompilationError::NotAnUnderConstructionArray(
                parser_metadata.clone(),
                self.clone(),
            )),
        }
    }

    pub fn get_readable_prefix_type(
        &self,
        material: Material,
        parser_metadata: &ParserMetadata,
    ) -> Result<(&Type, &ExpressionContainer), CompilationError> {
        match (self, material) {
            (Type::ReadablePrefix(elem_type, length), _) => Ok((elem_type, length)),
            (Type::Unmaterialized(inside), Material::Dematerialized) => {
                inside.get_readable_prefix_type(material, parser_metadata)
            }
            _ => Err(CompilationError::NotAnArray(
                parser_metadata.clone(),
                self.clone(),
            )),
        }
    }

    pub fn get_named_type(
        &self,
        material: Material,
        parser_metadata: &ParserMetadata,
    ) -> Result<&String, CompilationError> {
        match (self, material) {
            (Type::NamedType(value), _) => Ok(value),
            (Type::Unmaterialized(inside), Material::Dematerialized) => {
                inside.get_named_type(material, parser_metadata)
            }
            _ => Err(CompilationError::NotANamedType(
                parser_metadata.clone(),
                self.clone(),
            )),
        }
    }

    pub fn get_const_array_type(
        &self,
        material: Material,
        parser_metadata: &ParserMetadata,
    ) -> Result<(&Type, usize), CompilationError> {
        match (self, material) {
            (Type::ConstArray(value, len), _) => Ok((value.as_ref(), *len)),
            (Type::Unmaterialized(inside), Material::Dematerialized) => {
                inside.get_const_array_type(material, parser_metadata)
            }
            _ => Err(CompilationError::NotAConstArray(
                parser_metadata.clone(),
                self.clone(),
            )),
        }
    }
}
