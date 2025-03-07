use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, Mutex},
};

use itertools::Itertools;

use super::{
    error::CompilationError,
    function_resolution::{FunctionContainer, FunctionSet},
    ir::{Expression, Type},
    type_resolution::TypeSet,
};
use crate::folder1::ir::{ArithmeticOperator, Material};

#[derive(Clone)]
pub struct ExpressionContainer {
    pub(crate) expression: Box<Expression>,
    pub(crate) tipo: Option<Type>,
}

impl ExpressionContainer {
    pub fn get_type(&self) -> &Type {
        self.tipo.as_ref().unwrap()
    }
}

pub struct RootContainer {
    pub(crate) type_set: Arc<TypeSet>,
    pub(crate) function_set: Arc<FunctionSet>,
    pub(crate) root_scope: ScopeContainer,
    pub(crate) current_function: FunctionContainer,
}

pub struct ScopeContainer {
    pub children: Vec<HashMap<String, ScopeContainer>>,

    declarations: HashMap<String, Type>,
    banned_declarations: HashSet<String>,

    definitions: HashSet<String>,
    under_construction_array_usages: HashSet<String>,
}

#[derive(Clone, Default)]
pub struct DeclarationSet {
    pub declarations: HashMap<(ScopePath, String), Type>,
}

impl DeclarationSet {
    pub fn new() -> Self {
        Self {
            declarations: HashMap::new(),
        }
    }
    pub fn insert(&mut self, path: &ScopePath, name: &String, tipo: &Type) {
        self.declarations
            .insert((path.clone(), name.clone()), tipo.clone());
    }
    pub fn get_declaration_scope(&self, curr: &ScopePath, name: &str) -> ScopePath {
        curr.prefixes()
            .find(|prefix| {
                self.declarations
                    .contains_key(&(prefix.clone(), name.to_string()))
            })
            .unwrap()
            .clone()
    }
}

impl ScopeContainer {
    pub(crate) fn new() -> Self {
        Self {
            children: Vec::new(),
            declarations: HashMap::new(),
            banned_declarations: HashSet::new(),
            definitions: HashSet::new(),
            under_construction_array_usages: HashSet::new(),
        }
    }
    pub fn scope_exists(&self, path: &ScopePath) -> bool {
        let mut scope = self;
        for (i, name) in path.0.iter() {
            if *i >= scope.children.len() {
                return false;
            }
            if !scope.children[*i].contains_key(name) {
                return false;
            }
            scope = &scope.children[*i][name];
        }
        true
    }
    pub fn define_new_scopes(&mut self, path: &ScopePath, index: usize, constructors: Vec<String>) {
        let mut scope = self;
        for (i, name) in path.0.iter() {
            scope = scope.children.get_mut(*i).unwrap().get_mut(name).unwrap();
        }
        while scope.children.len() <= index {
            scope.children.push(HashMap::new());
        }
        for constructor in constructors {
            scope.children[index].insert(constructor, ScopeContainer::new());
        }
    }
    fn define(
        &mut self,
        path: &ScopePath,
        path_index: usize,
        name: &String,
    ) -> Result<(), CompilationError> {
        if path_index == path.0.len() {
            if self.definitions.contains(name) {
                return Err(CompilationError::DuplicateDefinition(name.clone()));
            }
            self.definitions.insert(name.clone());
        } else {
            let (i, constructor) = &path.0[path_index];
            let branches = &mut self.children[*i];
            let branch = branches.get_mut(constructor).unwrap();
            branch.define(path, path_index + 1, name)?;

            let mut present_in_all_branches = true;
            for (_, branch) in branches.iter() {
                if !branch.definitions.contains(name) {
                    present_in_all_branches = false;
                }
            }
            if present_in_all_branches {
                for (_, branch) in branches.iter_mut() {
                    branch.definitions.remove(name);
                }
                if self.definitions.contains(name) {
                    return Err(CompilationError::DuplicateDefinition(name.clone()));
                }
                self.definitions.insert(name.clone());
            }
        }
        Ok(())
    }
    pub fn is_declared(&self, path: &ScopePath, name: &String) -> bool {
        let mut scope = self;
        let mut depth = 0;
        loop {
            if scope.declarations.contains_key(name) {
                return true;
            }
            if depth == path.0.len() {
                return false;
            }
            let (i, constructor) = &path.0[depth];
            scope = scope.children[*i].get(constructor).unwrap();
            depth += 1;
        }
    }
    pub fn is_defined(&self, path: &ScopePath, name: &String) -> bool {
        let mut scope = self;
        let mut depth = 0;
        loop {
            if scope.definitions.contains(name) {
                return true;
            }
            if depth == path.0.len() {
                return false;
            }
            let (i, constructor) = &path.0[depth];
            scope = scope.children[*i].get(constructor).unwrap();
            depth += 1;
        }
    }

    fn use_under_construction_array(
        &mut self,
        path: &ScopePath,
        path_index: usize,
        name: &String,
    ) -> Result<(), CompilationError> {
        if path_index == path.0.len() {
            if self.under_construction_array_usages.contains(name) {
                return Err(CompilationError::DuplicateUnderConstructionArrayUsage(
                    name.clone(),
                ));
            }
            self.under_construction_array_usages.insert(name.clone());
        } else {
            let (i, constructor) = &path.0[path_index];
            let branches = &mut self.children[*i];
            let branch = branches.get_mut(constructor).unwrap();
            branch.define(path, path_index + 1, name)?;

            let mut present_in_all_branches = true;
            for (_, branch) in branches.iter() {
                if !branch.under_construction_array_usages.contains(name) {
                    present_in_all_branches = false;
                }
            }
            if present_in_all_branches {
                for (_, branch) in branches.iter_mut() {
                    branch.under_construction_array_usages.remove(name);
                }
                if self.under_construction_array_usages.contains(name) {
                    return Err(CompilationError::DuplicateUnderConstructionArrayUsage(
                        name.clone(),
                    ));
                }
                self.under_construction_array_usages.insert(name.clone());
            }
        }
        Ok(())
    }

    fn place_in_declaration_set(&self, declaration_set: &mut DeclarationSet, prefix: &ScopePath) {
        for (name, tipo) in self.declarations.iter() {
            declaration_set.insert(prefix, name, tipo);
        }
        for (i, map) in self.children.iter().enumerate() {
            for (constructor, child) in map.iter() {
                child.place_in_declaration_set(
                    declaration_set,
                    &prefix.then(i, constructor.clone()),
                );
            }
        }
    }

    pub fn unpack(&self, declaration_set: &mut DeclarationSet) {
        self.place_in_declaration_set(declaration_set, &ScopePath::empty());
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct ScopePath(pub Vec<(usize, String)>);

impl ScopePath {
    pub fn empty() -> Self {
        Self(Vec::new())
    }
    pub fn then(&self, index: usize, name: String) -> Self {
        Self(
            self.0
                .clone()
                .into_iter()
                .chain(vec![(index, name)])
                .collect(),
        )
    }

    pub fn concat(&self, offset: usize, other: &Self) -> Self {
        let mut result = self.0.clone();
        result.push(other.0[0].clone());
        result.last_mut().unwrap().0 += offset;
        result.extend(other.0.clone().into_iter().skip(1));
        Self(result)
    }

    pub fn prepend(&mut self, other: &Self, offset: usize) {
        *self = other.concat(offset, self);
    }

    pub fn disjoint(&self, other: &Self) -> bool {
        // NOT zip_eq
        for ((index1, constructor1), (index2, constructor2)) in self.0.iter().zip(other.0.iter()) {
            if index1 != index2 {
                return false;
            }
            if constructor1 != constructor2 {
                return true;
            }
        }
        false
    }

    pub fn prefixes<'a>(&'a self) -> impl Iterator<Item = Self> + 'a {
        (0..=self.0.len()).map(move |i| ScopePath(self.0[..i].to_vec()))
    }

    pub fn is_prefix(&self, other: &Self, index: usize) -> bool {
        other.0.len() < self.0.len()
            && self.0[..other.0.len()] == other.0
            && self.0[other.0.len()].0 == index
    }
}

impl RootContainer {
    pub fn new(
        type_set: Arc<TypeSet>,
        function_set: Arc<FunctionSet>,
        current_function: FunctionContainer,
    ) -> Self {
        Self {
            type_set,
            root_scope: ScopeContainer::new(),
            function_set,
            current_function,
        }
    }
    fn get_declaration_type(
        &self,
        path: &ScopePath,
        name: &String,
    ) -> Result<Type, CompilationError> {
        let mut scope = &self.root_scope;
        let mut depth = 0;
        loop {
            if let Some(tipo) = scope.declarations.get(name) {
                return Ok(tipo.clone());
            }
            if depth == path.0.len() {
                return Err(CompilationError::UndeclaredVariable(name.clone()));
            }
            let (x, constructor) = &path.0[depth];
            scope = &scope.children[*x][constructor];

            depth += 1;
        }
    }

    pub fn declare(
        &mut self,
        path: &ScopePath,
        name: &String,
        tipo: Type,
    ) -> Result<(), CompilationError> {
        if self.get_declaration_type(path, name).is_ok() {
            return Err(CompilationError::DuplicateDeclaration(name.clone()));
        }

        let mut scope = &mut self.root_scope;
        for (x, constructor) in path.0.iter() {
            scope.banned_declarations.insert(name.clone());
            scope = scope.children[*x].get_mut(constructor).unwrap();
        }

        if scope.banned_declarations.contains(name) {
            return Err(CompilationError::DuplicateDeclaration(name.clone()));
        }

        scope.declarations.insert(name.clone(), tipo);
        Ok(())
    }

    pub fn define(&mut self, path: &ScopePath, name: &String) -> Result<(), CompilationError> {
        self.root_scope.define(path, 0, name)
    }

    fn use_under_construction_array(
        &mut self,
        path: &ScopePath,
        name: &String,
    ) -> Result<(), CompilationError> {
        self.root_scope.use_under_construction_array(path, 0, name)
    }

    fn type_set(&self) -> Arc<TypeSet> {
        self.type_set.clone()
    }
}

impl ExpressionContainer {
    pub fn new(expression: Expression) -> Self {
        Self {
            expression: Box::new(expression),
            tipo: None,
        }
    }

    fn children(&self) -> Vec<ExpressionContainer> {
        match self.expression.as_ref() {
            Expression::Algebraic { fields, .. } => fields.to_vec(),
            Expression::Arithmetic { left, right, .. } => vec![left.clone(), right.clone()],
            Expression::Dematerialized { value, .. } => vec![value.clone()],
            _ => vec![],
        }
    }

    pub fn dependencies(&self, declaration: bool) -> Vec<String> {
        match self.expression.as_ref() {
            Expression::Variable { name } => vec![name.clone()],
            Expression::Let { name } if declaration => {
                if declaration {
                    vec![name.clone()]
                } else {
                    vec![]
                }
            }
            _ => self
                .children()
                .iter()
                .flat_map(|child| child.dependencies(declaration))
                .collect(),
        }
    }

    pub fn resolve_defined(
        &mut self,
        function_container: &mut RootContainer,
        path: &ScopePath,
        material: Material,
    ) -> Result<(), CompilationError> {
        match self.expression.as_mut() {
            Expression::Constant { .. } => {
                self.tipo = Some(Type::Field);
            }
            Expression::Variable { name } => {
                self.tipo = Some(function_container.get_declaration_type(path, name)?);
                if self
                    .tipo
                    .as_ref()
                    .unwrap()
                    .contains_under_construction_array(&function_container.type_set())?
                {
                    function_container.use_under_construction_array(path, name)?;
                }
            }
            Expression::Let { name } => {
                return Err(CompilationError::CannotAssignHere(name.clone()));
            }
            Expression::Define { name } => {
                return Err(CompilationError::CannotAssignHere(name.clone()));
            }
            Expression::Algebraic {
                constructor,
                fields,
            } => {
                let expected_component_types = function_container
                    .type_set()
                    .get_component_types(constructor)?;
                if fields.len() != expected_component_types.len() {
                    return Err(CompilationError::IncorrectNumberOfComponents(
                        constructor.clone(),
                        fields.len(),
                        expected_component_types.len(),
                    ));
                }
                let mut index = 0;
                for (field, expected_type) in
                    fields.iter_mut().zip_eq(expected_component_types.iter())
                {
                    index += 1;
                    field.resolve_defined(function_container, path, material)?;
                    if !material.same_type(&field.tipo.clone().unwrap(), expected_type) {
                        return Err(CompilationError::IncorrectTypeInComponent(
                            constructor.clone(),
                            index,
                            field.tipo.clone().unwrap(),
                            expected_type.clone(),
                        ));
                    }
                }

                self.tipo = Some(
                    function_container
                        .type_set()
                        .get_constructor_type(constructor)?,
                );
            }
            Expression::Arithmetic {
                operator,
                left,
                right,
            } => {
                if *operator == ArithmeticOperator::Div && material == Material::Materialized {
                    return Err(CompilationError::DivMustBeDematerialized());
                }
                left.resolve_defined(function_container, path, material)?;
                if !material.same_type(&left.tipo.clone().unwrap(), &Type::Field) {
                    return Err(CompilationError::IncorrectTypeInArithmetic(
                        left.tipo.clone().unwrap(),
                    ));
                }
                right.resolve_defined(function_container, path, material)?;
                if !material.same_type(&right.tipo.clone().unwrap(), &Type::Field) {
                    return Err(CompilationError::IncorrectTypeInArithmetic(
                        right.tipo.clone().unwrap(),
                    ));
                }
                self.tipo = Some(Type::Field);
            }
            Expression::Dematerialized { value } => {
                value.resolve_defined(function_container, path, Material::Dematerialized)?;
                self.tipo = Some(Type::Unmaterialized(Arc::new(value.tipo.clone().unwrap())));
            }
            Expression::Eq { left, right } => {
                if material == Material::Materialized {
                    return Err(CompilationError::EqMustBeDematerialized());
                }
                left.resolve_defined(function_container, path, Material::Dematerialized)?;
                right.resolve_defined(function_container, path, Material::Dematerialized)?;
                if !Material::Dematerialized
                    .same_type(&left.tipo.clone().unwrap(), &right.tipo.clone().unwrap())
                {
                    return Err(CompilationError::IncorrectTypesInEquality(
                        left.tipo.clone().unwrap(),
                        right.tipo.clone().unwrap(),
                    ));
                }
                if left
                    .get_type()
                    .contains_reference(&function_container.type_set, false)
                    .unwrap()
                {
                    return Err(CompilationError::CannotEquateReferences(
                        left.get_type().clone(),
                    ));
                }
                self.tipo = Some(Type::NamedType("Bool".to_string()));
            }
            Expression::EmptyConstArray { elem_type } => {
                self.tipo = Some(Type::ConstArray(Arc::new(elem_type.clone()), 0));
            }
            Expression::ConstArray { elements } => {
                if elements.is_empty() {
                    return Err(CompilationError::CannotInferEmptyConstArrayType());
                }
                for element in elements.iter_mut() {
                    element.resolve_defined(function_container, path, material)?;
                }
                let elem_type = elements[0].tipo.clone().unwrap();
                for (i, element) in elements.iter().enumerate() {
                    if !material.same_type(&elem_type, &element.tipo.clone().unwrap()) {
                        return Err(CompilationError::ElementTypesInconsistentInConstArray(
                            elem_type.clone(),
                            element.tipo.clone().unwrap(),
                            0,
                            i,
                        ));
                    }
                }
                self.tipo = Some(Type::ConstArray(Arc::new(elem_type), elements.len()));
            }
            Expression::ConstArrayConcatenation { left, right } => {
                left.resolve_defined(function_container, path, material)?;
                right.resolve_defined(function_container, path, material)?;
                let (elem_type_1, len_1) = left.get_type().get_const_array_type(material)?;
                let (elem_type_2, len_2) = right.get_type().get_const_array_type(material)?;
                if !material.same_type(&elem_type_1, &elem_type_2) {
                    return Err(
                        CompilationError::ElementTypesInconsistentInConstArrayConcatenation(
                            elem_type_1.clone(),
                            elem_type_2.clone(),
                        ),
                    );
                }
                self.tipo = Some(Type::ConstArray(elem_type_1.clone().into(), len_1 + len_2));
            }
            Expression::ConstArrayAccess { array, index } => {
                array.resolve_defined(function_container, path, material)?;
                let (elem_type, len) = array.get_type().get_const_array_type(material)?;
                if *index >= len {
                    return Err(CompilationError::OutOfBoundsConstArrayAccess(*index, len));
                }
                self.tipo = Some(elem_type.clone());
            }
            Expression::ConstArraySlice { array, from, to } => {
                array.resolve_defined(function_container, path, material)?;
                let (elem_type, len) = array.get_type().get_const_array_type(material)?;
                if *from > *to || *to > len {
                    return Err(CompilationError::OutOfBoundsConstArraySlice(
                        *from, *to, len,
                    ));
                }
                self.tipo = Some(Type::ConstArray(elem_type.clone().into(), *to - *from));
            }
            Expression::ConstArrayRepeated { element, length } => {
                element.resolve_defined(function_container, path, material)?;
                let elem_type = element.get_type();
                if elem_type
                    .contains_under_construction_array(&function_container.type_set)
                    .unwrap()
                {
                    return Err(CompilationError::DuplicateUnderConstructionArrayUsageInConstArray);
                }

                self.tipo = Some(Type::ConstArray(Arc::new(elem_type.clone()), *length));
            }
        }
        Ok(())
    }

    pub fn resolve_top_down(
        &mut self,
        expected_type: &Type,
        function_container: &mut RootContainer,
        path: &ScopePath,
        material: Material,
    ) -> Result<(), CompilationError> {
        match self.expression.as_mut() {
            Expression::Let { name } => {
                material.assert_type(
                    &function_container.get_declaration_type(path, name)?,
                    expected_type,
                )?;
                if material == Material::Dematerialized
                    && expected_type.contains_reference(&function_container.type_set(), true)?
                {
                    return Err(CompilationError::ReferenceDefinitionMustBeMaterialized(
                        name.clone(),
                    ));
                }
                function_container.define(path, name)?;
            }
            Expression::Define { name } => {
                let declaration_type = match material {
                    Material::Materialized => expected_type.clone(),
                    Material::Dematerialized => {
                        Type::Unmaterialized(Arc::new(expected_type.clone()))
                    }
                };
                function_container.declare(path, name, declaration_type)?;
                function_container.define(path, name)?;
            }
            Expression::Algebraic {
                constructor,
                fields,
            } => {
                let expected_component_types = function_container
                    .type_set()
                    .get_component_types(constructor)?;
                if fields.len() != expected_component_types.len() {
                    return Err(CompilationError::IncorrectNumberOfComponents(
                        constructor.clone(),
                        fields.len(),
                        expected_component_types.len(),
                    ));
                }
                for (field, expected_field_type) in
                    fields.iter_mut().zip_eq(expected_component_types.iter())
                {
                    field.resolve_top_down(
                        expected_field_type,
                        function_container,
                        path,
                        material,
                    )?;
                }

                material.assert_type(
                    &function_container
                        .type_set()
                        .get_constructor_type(constructor)?,
                    expected_type,
                )?;
            }
            Expression::ConstArray { elements } => {
                let (elem_type, len) = expected_type.get_const_array_type(material)?;
                if elements.len() != len {
                    return Err(CompilationError::IncorrectNumberOfElementsInConstArray(
                        elements.len(),
                        len,
                    ));
                }
                for element in elements.iter_mut() {
                    element.resolve_top_down(elem_type, function_container, path, material)?;
                }
            }
            Expression::Dematerialized { value } => {
                if material == Material::Materialized {
                    match expected_type {
                        Type::Unmaterialized(_) => {}
                        _ => {
                            return Err(CompilationError::UnexpectedUnmaterialized(
                                expected_type.clone(),
                            ))
                        }
                    }
                }
                value.resolve_top_down(
                    expected_type,
                    function_container,
                    path,
                    Material::Dematerialized,
                )?;
            }
            _ => {
                if expected_type.contains_reference(&function_container.type_set(), false)? {
                    return Err(CompilationError::CannotEquateReferences(
                        expected_type.clone(),
                    ));
                }
                self.resolve_defined(function_container, path, material)?;
                if !material.same_type(&self.tipo.clone().unwrap(), &expected_type) {
                    return Err(CompilationError::UnexpectedType(
                        self.tipo.clone().unwrap(),
                        expected_type.clone(),
                    ));
                }
            }
        }

        self.tipo = Some(expected_type.clone());
        Ok(())
    }
}
