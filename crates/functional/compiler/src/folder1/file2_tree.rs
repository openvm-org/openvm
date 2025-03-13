use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use itertools::Itertools;

use super::{
    error::CompilationError,
    function_resolution::{FunctionContainer, FunctionSet},
    ir::{Expression, Type},
    type_resolution::TypeSet,
};
use crate::{
    folder1::ir::{ArithmeticOperator, Material},
    parser::metadata::ParserMetadata,
};

#[derive(Clone, Debug)]
pub struct ExpressionContainer {
    pub(crate) expression: Box<Expression>,
    pub(crate) tipo: Option<Type>,
    pub(crate) parser_metadata: ParserMetadata,
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
    representations: HashSet<String>,
    under_construction_array_usages: HashSet<String>,
}

#[derive(Clone, Default, Debug)]
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
            representations: HashSet::new(),
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
        parser_metadata: &ParserMetadata,
    ) -> Result<(), CompilationError> {
        if path_index == path.0.len() {
            if self.definitions.contains(name) {
                return Err(CompilationError::DuplicateDefinition(
                    parser_metadata.clone(),
                    name.clone(),
                ));
            }
            self.definitions.insert(name.clone());
        } else {
            let (i, constructor) = &path.0[path_index];
            let branches = &mut self.children[*i];
            let branch = branches.get_mut(constructor).unwrap();
            branch.define(path, path_index + 1, name, parser_metadata)?;

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
                    return Err(CompilationError::DuplicateDefinition(
                        parser_metadata.clone(),
                        name.clone(),
                    ));
                }
                self.definitions.insert(name.clone());
            }
        }
        Ok(())
    }
    fn represent(
        &mut self,
        path: &ScopePath,
        path_index: usize,
        name: &String,
        parser_metadata: &ParserMetadata,
    ) -> Result<(), CompilationError> {
        if path_index == path.0.len() {
            if self.representations.contains(name) {
                return Err(CompilationError::DuplicateRepresentation(
                    parser_metadata.clone(),
                    name.clone(),
                ));
            }
            self.representations.insert(name.clone());
        } else {
            let (i, constructor) = &path.0[path_index];
            let branches = &mut self.children[*i];
            let branch = branches.get_mut(constructor).unwrap();
            branch.represent(path, path_index + 1, name, parser_metadata)?;

            let mut present_in_all_branches = true;
            for (_, branch) in branches.iter() {
                if !branch.representations.contains(name) {
                    present_in_all_branches = false;
                }
            }
            if present_in_all_branches {
                for (_, branch) in branches.iter_mut() {
                    branch.representations.remove(name);
                }
                if self.representations.contains(name) {
                    return Err(CompilationError::DuplicateRepresentation(
                        parser_metadata.clone(),
                        name.clone(),
                    ));
                }
                self.representations.insert(name.clone());
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
    pub fn get_declaration_type(&self, path: &ScopePath, name: &String) -> Type {
        let mut scope = self;
        let mut depth = 0;
        loop {
            if scope.declarations.contains_key(name) {
                return scope.declarations[name].clone();
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
    pub fn is_represented(&self, path: &ScopePath, name: &String, type_set: &TypeSet) -> bool {
        if !self.is_declared(path, name) {
            return false;
        }
        let tipo = self.get_declaration_type(path, name);
        if type_set.calc_type_size(&tipo) == 0 {
            return true;
        }

        let mut scope = self;
        let mut depth = 0;
        loop {
            if scope.representations.contains(name) {
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
        parser_metadata: &ParserMetadata,
    ) -> Result<(), CompilationError> {
        if path_index == path.0.len() {
            if self.under_construction_array_usages.contains(name) {
                return Err(CompilationError::DuplicateUnderConstructionArrayUsage(
                    parser_metadata.clone(),
                    name.clone(),
                ));
            }
            self.under_construction_array_usages.insert(name.clone());
        } else {
            let (i, constructor) = &path.0[path_index];
            let branches = &mut self.children[*i];
            let branch = branches.get_mut(constructor).unwrap();
            branch.define(path, path_index + 1, name, parser_metadata)?;

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
                        parser_metadata.clone(),
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

    pub fn verify_completeness(
        &self,
        type_set: &TypeSet,
        parser_metadata: &ParserMetadata,
    ) -> Result<(), CompilationError> {
        for (name, tipo) in self.declarations.iter() {
            if !self.definitions.contains(name) {
                return Err(CompilationError::UndeclaredVariable(
                    parser_metadata.clone(),
                    name.clone(),
                ));
            }
            if type_set.calc_type_size(tipo) > 0 && !self.representations.contains(name) {
                return Err(CompilationError::UnrepresentedVariable(
                    parser_metadata.clone(),
                    name.clone(),
                ));
            }
        }
        for matchi in self.children.iter() {
            for child in matchi.values() {
                child.verify_completeness(type_set, parser_metadata)?;
            }
        }
        Ok(())
    }

    pub fn unpack(&self, declaration_set: &mut DeclarationSet) {
        self.place_in_declaration_set(declaration_set, &ScopePath::empty());
    }
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
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

    pub fn prefixes<'a>(&'a self) -> impl DoubleEndedIterator<Item = Self> + 'a {
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
        parser_metadata: &ParserMetadata,
    ) -> Result<Type, CompilationError> {
        let mut scope = &self.root_scope;
        let mut depth = 0;
        loop {
            if let Some(tipo) = scope.declarations.get(name) {
                return Ok(tipo.clone());
            }
            if depth == path.0.len() {
                return Err(CompilationError::UndeclaredVariable(
                    parser_metadata.clone(),
                    name.clone(),
                ));
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
        parser_metadata: &ParserMetadata,
    ) -> Result<(), CompilationError> {
        if self
            .get_declaration_type(path, name, &ParserMetadata::default())
            .is_ok()
        {
            return Err(CompilationError::DuplicateDeclaration(
                parser_metadata.clone(),
                name.clone(),
            ));
        }

        let mut scope = &mut self.root_scope;
        for (x, constructor) in path.0.iter() {
            scope.banned_declarations.insert(name.clone());
            scope = scope.children[*x].get_mut(constructor).unwrap();
        }

        if scope.banned_declarations.contains(name) {
            return Err(CompilationError::DuplicateDeclaration(
                parser_metadata.clone(),
                name.clone(),
            ));
        }

        scope.declarations.insert(name.clone(), tipo);
        Ok(())
    }

    pub fn define(
        &mut self,
        path: &ScopePath,
        name: &String,
        parser_metadata: &ParserMetadata,
    ) -> Result<(), CompilationError> {
        self.root_scope.define(path, 0, name, parser_metadata)
    }

    pub fn represent(
        &mut self,
        path: &ScopePath,
        name: &String,
        parser_metadata: &ParserMetadata,
    ) -> Result<(), CompilationError> {
        self.root_scope.represent(path, 0, name, parser_metadata)
    }

    fn use_under_construction_array(
        &mut self,
        path: &ScopePath,
        name: &String,
        parser_metadata: &ParserMetadata,
    ) -> Result<(), CompilationError> {
        self.root_scope
            .use_under_construction_array(path, 0, name, parser_metadata)
    }

    fn type_set(&self) -> Arc<TypeSet> {
        self.type_set.clone()
    }
}

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

    pub fn resolve_defined(
        &mut self,
        function_container: &mut RootContainer,
        path: &ScopePath,
        material: Material,
        can_represent: bool,
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
                if self
                    .tipo
                    .as_ref()
                    .unwrap()
                    .contains_under_construction_array(&function_container.type_set())
                {
                    function_container.use_under_construction_array(
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
                let mut index = 0;
                for (field, expected_type) in
                    fields.iter_mut().zip_eq(expected_component_types.iter())
                {
                    index += 1;
                    field.resolve_defined(function_container, path, material, can_represent)?;
                    if !material.same_type(&field.tipo.clone().unwrap(), expected_type) {
                        return Err(CompilationError::IncorrectTypeInComponent(
                            self.parser_metadata.clone(),
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
                left.resolve_defined(function_container, path, material, false)?;
                if !material.same_type(&left.tipo.clone().unwrap(), &Type::Field) {
                    return Err(CompilationError::IncorrectTypeInArithmetic(
                        left.parser_metadata.clone(),
                        left.tipo.clone().unwrap(),
                    ));
                }
                right.resolve_defined(function_container, path, material, false)?;
                if !material.same_type(&right.tipo.clone().unwrap(), &Type::Field) {
                    return Err(CompilationError::IncorrectTypeInArithmetic(
                        right.parser_metadata.clone(),
                        right.tipo.clone().unwrap(),
                    ));
                }
                self.tipo = Some(Type::Field);
            }
            Expression::Dematerialized { value } => {
                value.resolve_defined(
                    function_container,
                    path,
                    Material::Dematerialized,
                    can_represent,
                )?;
                self.tipo = Some(Type::Unmaterialized(Arc::new(value.tipo.clone().unwrap())));
            }
            Expression::Eq { left, right } => {
                if material == Material::Materialized {
                    return Err(CompilationError::EqMustBeDematerialized(
                        self.parser_metadata.clone(),
                    ));
                }
                left.resolve_defined(function_container, path, Material::Dematerialized, false)?;
                right.resolve_defined(function_container, path, Material::Dematerialized, false)?;
                if !left.get_type().eq_unmaterialized(right.get_type()) {
                    return Err(CompilationError::MismatchedTypes(
                        self.parser_metadata.clone(),
                        left.tipo.clone().unwrap(),
                        right.tipo.clone().unwrap(),
                    ));
                }
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
                function_container
                    .type_set
                    .check_type_exists(elem_type, &self.parser_metadata)?;
                self.tipo = Some(Type::ConstArray(Arc::new(elem_type.clone()), 0));
            }
            Expression::ConstArray { elements } => {
                if elements.is_empty() {
                    return Err(CompilationError::CannotInferEmptyConstArrayType(
                        self.parser_metadata.clone(),
                    ));
                }
                for element in elements.iter_mut() {
                    element.resolve_defined(function_container, path, material, can_represent)?;
                }
                let elem_type = elements[0].tipo.clone().unwrap();
                for (i, element) in elements.iter().enumerate() {
                    if !material.same_type(&elem_type, &element.tipo.clone().unwrap()) {
                        return Err(CompilationError::ElementTypesInconsistentInConstArray(
                            self.parser_metadata.clone(),
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
                left.resolve_defined(function_container, path, material, false)?;
                right.resolve_defined(function_container, path, material, false)?;
                let (elem_type_1, len_1) = left
                    .get_type()
                    .get_const_array_type(material, &left.parser_metadata)?;
                let (elem_type_2, len_2) = right
                    .get_type()
                    .get_const_array_type(material, &right.parser_metadata)?;
                if !material.same_type(&elem_type_1, &elem_type_2) {
                    return Err(
                        CompilationError::ElementTypesInconsistentInConstArrayConcatenation(
                            self.parser_metadata.clone(),
                            elem_type_1.clone(),
                            elem_type_2.clone(),
                        ),
                    );
                }
                self.tipo = Some(Type::ConstArray(elem_type_1.clone().into(), len_1 + len_2));
            }
            Expression::ConstArrayAccess { array, index } => {
                array.resolve_defined(function_container, path, material, false)?;
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
                array.resolve_defined(function_container, path, material, false)?;
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
                element.resolve_defined(function_container, path, material, false)?;
                let elem_type = element.get_type();
                if elem_type.contains_under_construction_array(&function_container.type_set) {
                    return Err(
                        CompilationError::DuplicateUnderConstructionArrayUsageInConstArray(
                            element.parser_metadata.clone(),
                        ),
                    );
                }

                self.tipo = Some(Type::ConstArray(Arc::new(elem_type.clone()), *length));
            }
            Expression::BooleanNot { value } => {
                value.resolve_defined(function_container, path, material, false)?;
                if !material.same_type(value.get_type(), &Type::boolean()) {
                    return Err(CompilationError::ExpectedBoolean(
                        self.parser_metadata.clone(),
                        value.get_type().clone(),
                    ));
                }
                self.tipo = Some(Type::boolean());
            }
            Expression::BooleanBinary { left, right, .. } => {
                left.resolve_defined(function_container, path, material, false)?;
                if !material.same_type(left.get_type(), &Type::boolean()) {
                    return Err(CompilationError::ExpectedBoolean(
                        left.parser_metadata.clone(),
                        left.get_type().clone(),
                    ));
                }
                right.resolve_defined(function_container, path, material, false)?;
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
                condition.resolve_defined(function_container, path, material, false)?;
                if !material.same_type(condition.get_type(), &Type::boolean()) {
                    return Err(CompilationError::ExpectedBoolean(
                        condition.parser_metadata.clone(),
                        condition.get_type().clone(),
                    ));
                }
                true_value.resolve_defined(function_container, path, material, false)?;
                false_value.resolve_defined(function_container, path, material, false)?;
                if !material.same_type(true_value.get_type(), false_value.get_type()) {
                    return Err(CompilationError::MismatchedTypes(
                        self.parser_metadata.clone(),
                        true_value.get_type().clone(),
                        false_value.get_type().clone(),
                    ));
                }
                self.tipo = Some(true_value.get_type().clone());
            }
        }
        Ok(())
    }

    pub fn resolve_definition_top_down(
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
                        Type::Unmaterialized(Arc::new(expected_type.clone()))
                    }
                };
                function_container.declare(path, name, declaration_type, &self.parser_metadata)?;
                function_container.define(path, name, &self.parser_metadata)?;
            }
            Expression::Variable {
                name,
                declares: false,
                defines: true,
                ..
            } => {
                material.assert_type(
                    &function_container.get_declaration_type(path, name, &self.parser_metadata)?,
                    expected_type,
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
                function_container.define(path, name, &self.parser_metadata)?;
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
                    field.resolve_definition_top_down(
                        expected_field_type,
                        function_container,
                        path,
                        material,
                    )?;
                }

                material.assert_type(
                    &function_container
                        .type_set()
                        .get_constructor_type(constructor, &self.parser_metadata)?,
                    expected_type,
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
                    element.resolve_definition_top_down(
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
                value.resolve_definition_top_down(
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
                self.resolve_defined(function_container, path, material, true)?;
                if !material.same_type(&self.tipo.clone().unwrap(), &expected_type) {
                    return Err(CompilationError::UnexpectedType(
                        self.parser_metadata.clone(),
                        self.tipo.clone().unwrap(),
                        expected_type.clone(),
                    ));
                }
            }
        }

        self.tipo = Some(expected_type.clone());
        Ok(())
    }

    pub fn resolve_representation(
        &self,
        function_container: &mut RootContainer,
        path: &ScopePath,
    ) -> Result<(), CompilationError> {
        match self.expression.as_ref() {
            Expression::Variable {
                name, represents, ..
            } => {
                if *represents {
                    function_container.represent(path, name, &self.parser_metadata)?;
                }
            }
            Expression::Algebraic { fields, .. } => {
                for field in fields {
                    field.resolve_representation(function_container, path)?;
                }
            }
            Expression::ConstArray { elements } => {
                for element in elements {
                    element.resolve_representation(function_container, path)?;
                }
            }
            Expression::Dematerialized { value } => {
                value.resolve_representation(function_container, path)?;
            }
            _ => {}
        }
        Ok(())
    }
}
