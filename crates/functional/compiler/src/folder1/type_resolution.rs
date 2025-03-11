use std::collections::{HashMap, HashSet};

use super::{
    error::CompilationError,
    ir::{AlgebraicTypeDeclaration, Type},
};
use crate::folder1::ir::Material;

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
                                .dependencies()
                                .iter()
                                .all(|dependency| defined_types.contains(dependency))
                        })
                    })
                })
                .ok_or(CompilationError::TypesSelfReferential())?;
            let type_declaration = type_declarations.remove(i);
            type_order.push(type_declaration.name.clone());
            defined_types.insert(type_declaration.name.clone());
            for variant in type_declaration.variants.iter() {
                if constructors.contains_key(&variant.name) {
                    return Err(CompilationError::DuplicateConstructorName(
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
                    type_declaration.name.clone(),
                ));
            }
            if type_declaration.name == "F" {
                return Err(CompilationError::TypeNameCannotBeF);
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
    ) -> Result<&AlgebraicTypeDeclaration, CompilationError> {
        if let Some(algebraic_type) = self.algebraic_types.get(name) {
            Ok(algebraic_type)
        } else {
            Err(CompilationError::UndefinedType(name.clone()))
        }
    }
    pub fn get_component_types(&self, constructor: &String) -> Result<Vec<Type>, CompilationError> {
        if let Some((_, components)) = self.constructors.get(constructor) {
            Ok(components.to_vec())
        } else {
            Err(CompilationError::UndefinedConstructor(constructor.clone()))
        }
    }

    pub fn get_constructor_type_name(
        &self,
        constructor: &String,
    ) -> Result<String, CompilationError> {
        if let Some((name, _)) = self.constructors.get(constructor) {
            Ok(name.clone())
        } else {
            Err(CompilationError::UndefinedConstructor(constructor.clone()))
        }
    }

    pub fn get_constructor_type(&self, constructor: &String) -> Result<Type, CompilationError> {
        Ok(Type::NamedType(
            self.get_constructor_type_name(constructor)?,
        ))
    }
}

impl Type {
    pub fn dependencies(&self) -> Vec<String> {
        match self {
            Type::NamedType(name) => vec![name.clone()],
            Type::Unmaterialized(inside) => inside.dependencies(),
            Type::ConstArray(inside, _) => inside.dependencies(),
            _ => vec![],
        }
    }
    pub fn contains_reference(
        &self,
        type_set: &TypeSet,
        only_materialized: bool,
    ) -> Result<bool, CompilationError> {
        Ok(match self {
            Type::Reference(_) => true,
            Type::Array(_) => true,
            Type::UnderConstructionArray(_) => true,
            Type::Field => false,
            Type::NamedType(name) => {
                let algebraic_type = type_set.get_algebraic_type(name)?;
                for variant in algebraic_type.variants.iter() {
                    for component in variant.components.iter() {
                        if component.contains_reference(type_set, only_materialized)? {
                            return Ok(true);
                        }
                    }
                }
                false
            }
            Type::ConstArray(inside, _) => {
                inside.contains_reference(type_set, only_materialized)?
            }
            Type::Unmaterialized(inside) => {
                !only_materialized && inside.contains_reference(type_set, false)?
            }
        })
    }

    pub fn contains_under_construction_array(
        &self,
        type_set: &TypeSet,
    ) -> Result<bool, CompilationError> {
        Ok(match self {
            Type::UnderConstructionArray(_) => true,
            Type::Reference(value) => value.contains_under_construction_array(type_set)?,
            Type::Array(elem) => elem.contains_under_construction_array(type_set)?,
            Type::Field => false,
            Type::NamedType(name) => {
                let algebraic_type = type_set.get_algebraic_type(name)?;
                for variant in algebraic_type.variants.iter() {
                    for component in variant.components.iter() {
                        if component.contains_under_construction_array(type_set)? {
                            return Ok(true);
                        }
                    }
                }
                false
            }
            Type::ConstArray(inside, _) => inside.contains_under_construction_array(type_set)?,
            Type::Unmaterialized(inside) => inside.contains_under_construction_array(type_set)?,
        })
    }

    pub fn get_reference_type(&self, material: Material) -> Result<&Type, CompilationError> {
        match (self, material) {
            (Type::Reference(value), _) => Ok(value),
            (Type::Unmaterialized(inside), Material::Dematerialized) => {
                inside.get_reference_type(material)
            }
            _ => Err(CompilationError::NotAReference(self.clone())),
        }
    }

    pub fn get_under_construction_array_type(
        &self,
        material: Material,
    ) -> Result<&Type, CompilationError> {
        match (self, material) {
            (Type::UnderConstructionArray(value), _) => Ok(value),
            (Type::Unmaterialized(inside), Material::Dematerialized) => {
                inside.get_under_construction_array_type(material)
            }
            _ => Err(CompilationError::NotAnUnderConstructionArray(self.clone())),
        }
    }

    pub fn get_array_type(&self, material: Material) -> Result<&Type, CompilationError> {
        match (self, material) {
            (Type::Array(value), _) => Ok(value),
            (Type::Unmaterialized(inside), Material::Dematerialized) => {
                inside.get_array_type(material)
            }
            _ => Err(CompilationError::NotAnArray(self.clone())),
        }
    }

    pub fn get_named_type(&self, material: Material) -> Result<&String, CompilationError> {
        match (self, material) {
            (Type::NamedType(value), _) => Ok(value),
            (Type::Unmaterialized(inside), Material::Dematerialized) => {
                inside.get_named_type(material)
            }
            _ => Err(CompilationError::NotANamedType(self.clone())),
        }
    }

    pub fn get_const_array_type(
        &self,
        material: Material,
    ) -> Result<(&Type, usize), CompilationError> {
        match (self, material) {
            (Type::ConstArray(value, len), _) => Ok((value.as_ref(), *len)),
            (Type::Unmaterialized(inside), Material::Dematerialized) => {
                inside.get_const_array_type(material)
            }
            _ => Err(CompilationError::NotAConstArray(self.clone())),
        }
    }
}
