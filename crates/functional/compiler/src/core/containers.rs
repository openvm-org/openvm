use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use crate::{
    core::{
        error::CompilationError,
        file3::Tree,
        function_resolution::{FunctionContainer, FunctionSet},
        ir::{Expression, Material, Type},
        scope::ScopePath,
        type_resolution::TypeSet,
    },
    parser::metadata::ParserMetadata,
};

#[derive(Clone, Debug)]
pub struct ExpressionContainer {
    pub(crate) expression: Box<Expression>,
    pub(crate) tipo: Option<Type>,
    pub(crate) parser_metadata: ParserMetadata,
}

#[derive(Clone, Debug)]
pub struct Assertion {
    pub material: Material,
    pub scope: ScopePath,
    pub left: ExpressionContainer,
    pub right: ExpressionContainer,
}

pub struct RootContainer {
    pub(crate) type_set: Arc<TypeSet>,
    pub(crate) function_set: Arc<FunctionSet>,
    pub(crate) root_scope: ScopeContainer,
    pub(crate) current_function: FunctionContainer,
    pub(crate) assertions: Vec<Assertion>,
}

impl RootContainer {
    pub fn assert_types_eq(
        &mut self,
        left: &Type,
        right: &Type,
        scope: &ScopePath,
        material: Material,
        parser_metadata: &ParserMetadata,
    ) -> Result<(), CompilationError> {
        if self.assert_types_eq_helper(left, right, scope, material) {
            Ok(())
        } else {
            Err(CompilationError::MismatchedTypes(
                parser_metadata.clone(),
                left.clone(),
                right.clone(),
            ))
        }
    }

    fn assert_types_eq_helper(
        &mut self,
        left: &Type,
        right: &Type,
        scope: &ScopePath,
        material: Material,
    ) -> bool {
        if material == Material::Dematerialized {
            if let Type::Unmaterialized(inside) = left {
                return self.assert_types_eq_helper(inside, right, scope, material);
            } else if let Type::Unmaterialized(inside) = right {
                return self.assert_types_eq_helper(left, inside, scope, material);
            }
        }
        match (left, right) {
            (Type::Field, Type::Field) => true,
            (Type::NamedType(left), Type::NamedType(right)) => left == right,
            (Type::Reference(left), Type::Reference(right)) => {
                self.assert_types_eq_helper(left, right, scope, material)
            }
            (Type::ReadablePrefix(left, length1), Type::ReadablePrefix(right, length2)) => {
                self.assertions.push(Assertion {
                    left: length1.as_ref().clone(),
                    right: length2.as_ref().clone(),
                    scope: scope.clone(),
                    material,
                });
                self.assert_types_eq_helper(left, right, scope, material)
            }
            (Type::AppendablePrefix(left, length1), Type::AppendablePrefix(right, length2)) => {
                self.assertions.push(Assertion {
                    left: length1.as_ref().clone(),
                    right: length2.as_ref().clone(),
                    scope: scope.clone(),
                    material,
                });
                self.assert_types_eq_helper(left, right, scope, material)
            }
            (Type::Array(left), Type::Array(right)) => {
                self.assert_types_eq_helper(left, right, scope, material)
            }
            (Type::Unmaterialized(left), Type::Unmaterialized(right)) => {
                self.assert_types_eq_helper(left, right, scope, material)
            }
            (Type::ConstArray(left, len1), Type::ConstArray(right, len2)) => {
                len1 == len2 && self.assert_types_eq_helper(left, right, scope, material)
            }
            _ => false,
        }
    }
}

pub struct ScopeContainer {
    pub children: Vec<HashMap<String, ScopeContainer>>,

    declarations: HashMap<String, Type>,
    banned_declarations: HashSet<String>,

    definitions: HashSet<String>,
    representations: HashSet<String>,
    appendable_prefix_consumptions: HashSet<String>,

    active: bool,
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
    
    pub fn get_declaration_type(&self, curr: &ScopePath, name: &str) -> &Type {
        let scope = self.get_declaration_scope(curr, name);
        &self.declarations[&(scope, name.to_string())]
    }
}

impl ScopeContainer {
    pub(crate) fn new(tree: &Tree) -> Self {
        Self {
            children: tree
                .children
                .iter()
                .map(|children| {
                    let mut scope_children = HashMap::new();
                    for (constructor, subtree) in children.iter() {
                        scope_children.insert(constructor.clone(), Self::new(subtree));
                    }
                    scope_children
                })
                .collect(),
            declarations: HashMap::new(),
            banned_declarations: HashSet::new(),
            definitions: HashSet::new(),
            representations: HashSet::new(),
            appendable_prefix_consumptions: HashSet::new(),
            active: false,
        }
    }
    pub fn scope_is_active(&self, path: &ScopePath) -> bool {
        let mut scope = self;
        for (i, name) in path.0.iter() {
            scope = &scope.children[*i][name];
        }
        scope.active
    }
    pub fn activate(&mut self) {
        self.active = true;
    }
    pub fn activate_children(&mut self, path: &ScopePath, index: usize, constructors: Vec<String>) {
        let mut scope = self;
        for (i, name) in path.0.iter() {
            scope = scope.children.get_mut(*i).unwrap().get_mut(name).unwrap();
        }
        for constructor in constructors {
            scope.children[index]
                .get_mut(&constructor)
                .unwrap()
                .activate();
        }
    }
    pub fn define(
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
    pub fn represent(
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

    pub fn consume_appendable_prefix(
        &mut self,
        path: &ScopePath,
        path_index: usize,
        name: &String,
        parser_metadata: &ParserMetadata,
    ) -> Result<(), CompilationError> {
        if path_index == path.0.len() {
            if self.appendable_prefix_consumptions.contains(name) {
                return Err(CompilationError::DuplicateAppendablePrefixConsumptions(
                    parser_metadata.clone(),
                    name.clone(),
                ));
            }
            self.appendable_prefix_consumptions.insert(name.clone());
        } else {
            let (i, constructor) = &path.0[path_index];
            let branches = &mut self.children[*i];
            let branch = branches.get_mut(constructor).unwrap();
            branch.define(path, path_index + 1, name, parser_metadata)?;

            let mut present_in_all_branches = true;
            for (_, branch) in branches.iter() {
                if !branch.appendable_prefix_consumptions.contains(name) {
                    present_in_all_branches = false;
                }
            }
            if present_in_all_branches {
                for (_, branch) in branches.iter_mut() {
                    branch.appendable_prefix_consumptions.remove(name);
                }
                if self.appendable_prefix_consumptions.contains(name) {
                    return Err(CompilationError::DuplicateAppendablePrefixConsumptions(
                        parser_metadata.clone(),
                        name.clone(),
                    ));
                }
                self.appendable_prefix_consumptions.insert(name.clone());
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
                return Err(CompilationError::UndefinedVariable(
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

impl RootContainer {
    pub fn new(
        type_set: Arc<TypeSet>,
        function_set: Arc<FunctionSet>,
        current_function: FunctionContainer,
        tree: &Tree,
    ) -> Self {
        Self {
            type_set,
            root_scope: ScopeContainer::new(tree),
            function_set,
            current_function,
            assertions: vec![],
        }
    }
    pub fn get_declaration_type(
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

    pub fn consume_appendable_prefix(
        &mut self,
        path: &ScopePath,
        name: &String,
        parser_metadata: &ParserMetadata,
    ) -> Result<(), CompilationError> {
        self.root_scope
            .consume_appendable_prefix(path, 0, name, parser_metadata)
    }

    pub fn type_set(&self) -> Arc<TypeSet> {
        self.type_set.clone()
    }
}

impl ExpressionContainer {
    pub fn get_type(&self) -> &Type {
        self.tipo.as_ref().unwrap()
    }
}
