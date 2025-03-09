use std::{collections::HashMap, sync::Arc};

use itertools::Itertools;

use super::{
    error::CompilationError,
    file2_tree::{DeclarationSet, ExpressionContainer, RootContainer, ScopePath},
    function_resolution::{FunctionContainer, FunctionSet},
    ir::{Body, Statement, Type},
    type_resolution::TypeSet,
};
use crate::folder1::{
    file2_tree::DependencyType,
    function_resolution::Stage,
    ir::{FunctionCall, Material},
};

#[derive(Clone)]
pub enum RepresentationOrder {
    Inline(Vec<Vec<Atom>>),
    NotInline(Vec<Atom>),
}

impl Default for RepresentationOrder {
    fn default() -> Self {
        Self::NotInline(Vec::default())
    }
}

#[derive(Clone)]
pub struct FlattenedFunction {
    pub(crate) inline: bool,
    pub(crate) stages: Vec<Stage>,
    pub(crate) arguments: Vec<ExpressionContainer>,

    pub(crate) statements: Vec<FlatStatement>,
    pub(crate) function_calls: Vec<FlatFunctionCall>,
    pub(crate) matches: Vec<FlatMatch>,

    pub(crate) tree: Tree,

    pub(crate) atoms_staged: Vec<Vec<Atom>>,
    pub(crate) representation_order: RepresentationOrder,
    pub(crate) declaration_set: DeclarationSet,

    pub(crate) name: String,
    pub(crate) uses_timestamp: bool,
    pub(crate) function_id: usize,
}

#[derive(Clone)]
pub struct FlatStatement {
    pub material: Material,
    pub scope: ScopePath,
    pub statement: Statement,
}

#[derive(Clone)]
pub struct FlatMatch {
    pub material: Material,
    pub scope: ScopePath,
    pub index: usize,
    pub value: ExpressionContainer,
    pub branches: Vec<(String, Vec<String>)>,
}

#[derive(Clone)]
pub struct FlatFunctionCall {
    pub material: Material,
    pub scope: ScopePath,
    pub function_name: String,
    pub arguments: Vec<ExpressionContainer>,
}

impl FlatFunctionCall {
    pub fn new(material: Material, scope: ScopePath, function_call: FunctionCall) -> Self {
        Self {
            material,
            scope,
            function_name: function_call.function,
            arguments: function_call.arguments,
        }
    }
}

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
pub enum Atom {
    Match(usize),
    Statement(usize),
    PartialFunctionCall(usize, Stage),
    InArgument(usize),
    OutArgument(usize),
}

#[derive(Clone)]
pub struct Tree {
    pub children: Vec<HashMap<String, Tree>>,
}

impl Tree {
    pub fn new() -> Self {
        Self {
            children: Vec::new(),
        }
    }

    pub fn insert(&mut self, path: &ScopePath, other: &Self) {
        let mut node = self;
        for (i, name) in path.0.iter() {
            node = node.children.get_mut(*i).unwrap().get_mut(name).unwrap();
        }
        node.children.extend(other.children.clone());
    }

    pub fn num_branches(&self, path: &ScopePath) -> usize {
        let mut node = self;
        for (i, name) in path.0.iter() {
            node = node.children.get(*i).unwrap().get(name).unwrap();
        }
        node.children.len()
    }
}

impl FlattenedFunction {
    pub fn create_flattened(
        function: FunctionContainer,
        type_set: Arc<TypeSet>,
        function_set: Arc<FunctionSet>,
        function_id: usize,
    ) -> Result<Self, CompilationError> {
        let arguments = function
            .function
            .arguments
            .iter()
            .map(|argument| ExpressionContainer::new(argument.value.clone()))
            .collect();
        let stages = function.stages.clone();

        let mut statements = Vec::new();
        let mut matches = Vec::new();
        let mut function_calls = Vec::new();
        let mut tree = Tree::new();
        Self::init(
            &mut statements,
            &mut matches,
            &mut function_calls,
            &mut tree,
            &ScopePath(vec![]),
            &function.function.body,
            &function_set,
        )?;

        let mut uses_timestamp = false;
        for statement in statements.iter() {
            match statement.statement {
                Statement::Reference { .. } => uses_timestamp = true,
                Statement::ArrayFinalization { .. } => uses_timestamp = true,
                _ => {}
            }
        }

        let mut flattened_function = Self {
            name: function.function.name.clone(),
            inline: function.function.inline,
            arguments,
            stages,
            statements,
            function_calls,
            matches,
            tree,
            atoms_staged: vec![],
            representation_order: if function.function.inline {
                RepresentationOrder::Inline(vec![])
            } else {
                RepresentationOrder::NotInline(vec![])
            },
            declaration_set: DeclarationSet::new(),
            uses_timestamp,
            function_id,
        };

        let mut root_container = RootContainer::new(type_set, function_set, function);
        flattened_function.order_for_execution(&mut root_container)?;
        flattened_function.order_for_representation(&mut root_container)?;
        root_container
            .root_scope
            .verify_completeness(root_container.type_set.as_ref())?;
        root_container
            .root_scope
            .unpack(&mut flattened_function.declaration_set);

        Ok(flattened_function)
    }

    fn init(
        statements: &mut Vec<FlatStatement>,
        matches: &mut Vec<FlatMatch>,
        function_calls: &mut Vec<FlatFunctionCall>,
        node: &mut Tree,
        path: &ScopePath,
        body: &Body,
        function_set: &FunctionSet,
    ) -> Result<(), CompilationError> {
        for (material, statement) in body.statements.clone() {
            statements.push(FlatStatement {
                material,
                scope: path.clone(),
                statement,
            });
        }
        for (material, function_call) in body.function_calls.clone() {
            function_calls.push(FlatFunctionCall::new(material, path.clone(), function_call));
        }
        for (i, matchi) in body.matches.iter().enumerate() {
            let branches = matchi
                .branches
                .iter()
                .map(|branch| (branch.constructor.clone(), branch.components.clone()))
                .collect();
            matches.push(FlatMatch {
                material: matchi.check_material,
                scope: path.clone(),
                index: i,
                value: matchi.value.clone(),
                branches,
            });
            let mut these_children = HashMap::new();
            for branch in &matchi.branches {
                let mut node = Tree::new();
                Self::init(
                    statements,
                    matches,
                    function_calls,
                    &mut node,
                    &ScopePath(
                        path.0
                            .clone()
                            .into_iter()
                            .chain(vec![(i, branch.constructor.clone())])
                            .collect(),
                    ),
                    &branch.body,
                    function_set,
                )?;
                these_children.insert(branch.constructor.clone(), node);
            }
            node.children.push(these_children);
        }

        Ok(())
    }

    fn order_for_execution(
        &mut self,
        root_container: &mut RootContainer,
    ) -> Result<(), CompilationError> {
        let mut disp_atoms = vec![];
        let mut function_calls_current_stage = vec![0; self.function_calls.len()];
        for i in 0..self.statements.len() {
            disp_atoms.push(Atom::Statement(i));
        }
        for i in 0..self.matches.len() {
            disp_atoms.push(Atom::Match(i));
        }
        for i in 0..self.function_calls.len() {
            disp_atoms.push(Atom::PartialFunctionCall(
                i,
                root_container.current_function.stages[0].clone(),
            ));
        }

        for stage in root_container.current_function.stages.clone() {
            let mut ordered_atoms = Vec::new();
            for argument_index in stage.start..stage.mid {
                let argument_atom = Atom::InArgument(argument_index);
                if !self.viable_for_definition(root_container, argument_atom) {
                    return Err(CompilationError::CannotOrderStatementsForDefinition());
                }
                self.resolve_definition(argument_atom, root_container)?;
                ordered_atoms.push(argument_atom);

                loop {
                    let i = disp_atoms
                        .iter()
                        .position(|&atom| self.viable_for_definition(root_container, atom));
                    if let Some(i) = i {
                        let atom = disp_atoms.remove(i);
                        self.resolve_definition(atom, root_container)?;
                        ordered_atoms.push(atom);

                        if let Atom::PartialFunctionCall(index, _) = atom {
                            function_calls_current_stage[index] += 1;
                            let callee = root_container
                                .function_set
                                .get_function(&self.function_calls[index].function_name)
                                .unwrap();
                            if function_calls_current_stage[index] < callee.stages.len() {
                                disp_atoms.push(Atom::PartialFunctionCall(
                                    index,
                                    callee.stages[function_calls_current_stage[index]],
                                ));
                            }
                        }
                    } else {
                        break;
                    }
                }
            }
            for argument_index in stage.mid..stage.end {
                let argument_atom = Atom::OutArgument(argument_index);
                if !self.viable_for_definition(root_container, argument_atom) {
                    return Err(CompilationError::CannotOrderStatementsForDefinition());
                }
                self.resolve_definition(argument_atom, root_container)?;
                ordered_atoms.push(argument_atom);
            }
            self.atoms_staged.push(ordered_atoms);
        }
        if !disp_atoms.is_empty() {
            return Err(CompilationError::CannotOrderStatementsForDefinition());
        }
        Ok(())
    }

    fn order_for_representation(
        &mut self,
        root_container: &mut RootContainer,
    ) -> Result<(), CompilationError> {
        let mut disp_atoms = vec![];
        let mut function_calls_current_stage = vec![0; self.function_calls.len()];
        for i in 0..self.statements.len() {
            disp_atoms.push(Atom::Statement(i));
        }
        for i in 0..self.matches.len() {
            disp_atoms.push(Atom::Match(i));
        }
        for i in 0..self.function_calls.len() {
            let callee = root_container
                .function_set
                .get_function(&self.function_calls[i].function_name)
                .unwrap();
            if callee.function.inline {
                disp_atoms.push(Atom::PartialFunctionCall(
                    i,
                    root_container.current_function.stages[0].clone(),
                ));
            } else {
                for stage in callee.stages.iter() {
                    disp_atoms.push(Atom::PartialFunctionCall(i, *stage));
                }
            }
        }

        if self.inline {
            let mut atoms_staged = vec![];
            for stage in root_container.current_function.stages.clone() {
                let mut ordered_atoms = Vec::new();
                for argument_index in stage.start..stage.mid {
                    let argument_atom = Atom::InArgument(argument_index);
                    self.resolve_representation(argument_atom, root_container)?;
                    ordered_atoms.push(argument_atom);

                    loop {
                        let i = disp_atoms
                            .iter()
                            .position(|&atom| self.viable_for_representation(root_container, atom));
                        if let Some(i) = i {
                            let atom = disp_atoms.remove(i);
                            self.resolve_representation(atom, root_container)?;
                            ordered_atoms.push(atom);

                            if let Atom::PartialFunctionCall(index, _) = atom {
                                let callee = root_container
                                    .function_set
                                    .get_function(&self.function_calls[index].function_name)
                                    .unwrap();
                                if callee.function.inline {
                                    function_calls_current_stage[index] += 1;
                                    if function_calls_current_stage[index] < callee.stages.len() {
                                        disp_atoms.push(Atom::PartialFunctionCall(
                                            index,
                                            callee.stages[function_calls_current_stage[index]],
                                        ));
                                    }
                                }
                            }
                        } else {
                            break;
                        }
                    }
                }
                for argument_index in stage.mid..stage.end {
                    let argument_atom = Atom::OutArgument(argument_index);
                    if !self.viable_for_representation(root_container, argument_atom) {
                        return Err(CompilationError::CannotOrderStatementsForRepresentation());
                    }
                    ordered_atoms.push(argument_atom);
                }
                atoms_staged.push(ordered_atoms);
            }
            self.representation_order = RepresentationOrder::Inline(atoms_staged);
        } else {
            let mut ordered_atoms = Vec::new();
            for stage in self.stages.iter() {
                for argument_index in stage.start..stage.mid {
                    let argument_atom = Atom::InArgument(argument_index);
                    self.resolve_representation(argument_atom, root_container)?;
                    ordered_atoms.push(argument_atom);
                }
            }
            loop {
                let i = disp_atoms
                    .iter()
                    .position(|&atom| self.viable_for_representation(root_container, atom));
                if let Some(i) = i {
                    let atom = disp_atoms.remove(i);
                    self.resolve_representation(atom, root_container)?;
                    ordered_atoms.push(atom);

                    if let Atom::PartialFunctionCall(index, _) = atom {
                        let callee = root_container
                            .function_set
                            .get_function(&self.function_calls[index].function_name)
                            .unwrap();
                        if callee.function.inline {
                            function_calls_current_stage[index] += 1;
                            if function_calls_current_stage[index] < callee.stages.len() {
                                disp_atoms.push(Atom::PartialFunctionCall(
                                    index,
                                    callee.stages[function_calls_current_stage[index]],
                                ));
                            }
                        }
                    }
                } else {
                    break;
                }
            }
            self.representation_order = RepresentationOrder::NotInline(ordered_atoms);
        }
        if !disp_atoms.is_empty() {
            return Err(CompilationError::CannotOrderStatementsForRepresentation());
        }
        Ok(())
    }
}

impl FlattenedFunction {
    pub(crate) fn scope(&self, atom: Atom) -> ScopePath {
        match atom {
            Atom::Match(index) => self.matches[index].scope.clone(),
            Atom::Statement(index) => self.statements[index].scope.clone(),
            Atom::PartialFunctionCall(index, _) => self.function_calls[index].scope.clone(),
            Atom::InArgument(_) => ScopePath::empty(),
            Atom::OutArgument(_) => ScopePath::empty(),
        }
    }
    fn viable_for_definition(&self, root_container: &RootContainer, atom: Atom) -> bool {
        let scope = &self.scope(atom);
        root_container.root_scope.scope_exists(scope)
            && self
                .dependencies(atom, DependencyType::Definition)
                .iter()
                .all(|dependency| root_container.root_scope.is_defined(scope, dependency))
            && self
                .dependencies(atom, DependencyType::Declaration)
                .iter()
                .all(|dependency| root_container.root_scope.is_declared(scope, dependency))
    }
    fn viable_for_representation(&self, root_container: &RootContainer, atom: Atom) -> bool {
        let scope = &self.scope(atom);
        self.dependencies(atom, DependencyType::Representation)
            .iter()
            .all(|dependency| root_container.root_scope.is_represented(scope, dependency))
    }
    fn dependencies(&self, atom: Atom, dependency_type: DependencyType) -> Vec<String> {
        match atom {
            Atom::Match(index) => self.matches[index].value.dependencies(dependency_type),
            Atom::PartialFunctionCall(index, stage) => {
                let call = &self.function_calls[index];
                (stage.start..stage.end)
                    .flat_map(|i| call.arguments[i].dependencies(dependency_type))
                    .collect()
            }
            Atom::InArgument(argument_index) => {
                self.arguments[argument_index].dependencies(dependency_type)
            }
            Atom::OutArgument(argument_index) => {
                self.arguments[argument_index].dependencies(dependency_type)
            }
            Atom::Statement(index) => match &self.statements[index].statement {
                Statement::VariableDeclaration { .. } => vec![],
                Statement::Equality { left, right } => left
                    .dependencies(dependency_type)
                    .into_iter()
                    .chain(right.dependencies(dependency_type))
                    .collect(),
                Statement::Reference { reference, data } => reference
                    .dependencies(dependency_type)
                    .into_iter()
                    .chain(data.dependencies(dependency_type))
                    .collect(),
                Statement::Dereference { data, reference } => data
                    .dependencies(dependency_type)
                    .into_iter()
                    .chain(reference.dependencies(dependency_type))
                    .collect(),
                Statement::EmptyUnderConstructionArray { array, .. } => {
                    array.dependencies(dependency_type).into_iter().collect()
                }
                Statement::UnderConstructionArrayPrepend {
                    new_array,
                    elem,
                    old_array,
                } => new_array
                    .dependencies(dependency_type)
                    .into_iter()
                    .chain(elem.dependencies(dependency_type))
                    .chain(old_array.dependencies(dependency_type))
                    .collect(),
                Statement::ArrayFinalization {
                    finalized,
                    under_construction,
                } => finalized
                    .dependencies(dependency_type)
                    .into_iter()
                    .chain(under_construction.dependencies(dependency_type))
                    .collect(),
                Statement::ArrayAccess { elem, array, index } => elem
                    .dependencies(dependency_type)
                    .into_iter()
                    .chain(array.dependencies(dependency_type))
                    .chain(index.dependencies(dependency_type))
                    .collect(),
            },
        }
    }

    fn resolve_definition(
        &mut self,
        atom: Atom,
        root_container: &mut RootContainer,
    ) -> Result<(), CompilationError> {
        match atom {
            Atom::Match(flat_index) => {
                let FlatMatch {
                    material,
                    scope,
                    index,
                    value,
                    branches,
                } = &mut self.matches[flat_index];
                let material = *material;
                value.resolve_defined(root_container, scope, material, true)?;
                root_container.root_scope.define_new_scopes(
                    scope,
                    *index,
                    branches.iter().map(|branch| branch.0.clone()).collect(),
                );
                let type_name = value.get_type().get_named_type(material)?;
                let type_definition = root_container
                    .type_set
                    .get_algebraic_type(type_name)?
                    .clone();
                for (constructor, components) in branches {
                    let component_types = type_definition
                        .variants
                        .iter()
                        .find(|variant| &variant.name == constructor)
                        .ok_or(CompilationError::UndefinedConstructor(constructor.clone()))?
                        .components
                        .clone();
                    let new_path = scope.then(*index, constructor.clone());
                    for (component, tipo) in components.iter().zip_eq(component_types.iter()) {
                        root_container.declare(&new_path, component, material.wrap(tipo))?;
                        root_container.define(&new_path, component)?;
                    }
                }
            }
            Atom::InArgument(argument_index) => {
                self.arguments[argument_index].resolve_definition_top_down(
                    &root_container
                        .current_function
                        .argument_type(argument_index)
                        .clone(),
                    root_container,
                    &ScopePath::empty(),
                    Material::Materialized,
                )?;
            }
            Atom::OutArgument(argument_index) => {
                let expression = &mut self.arguments[argument_index];
                expression.resolve_defined(
                    root_container,
                    &ScopePath::empty(),
                    Material::Dematerialized,
                    false,
                )?;
                let expected_type = root_container
                    .current_function
                    .argument_type(argument_index);
                if expression.get_type() != expected_type {
                    return Err(CompilationError::IncorrectTypeForOutArgument(
                        argument_index,
                        expression.get_type().clone(),
                        expected_type.clone(),
                    ));
                }
            }
            Atom::Statement(index) => {
                let FlatStatement {
                    material,
                    scope: path,
                    statement,
                } = &mut self.statements[index];
                let material = *material;
                match statement {
                    Statement::VariableDeclaration { name, tipo, .. } => {
                        root_container.declare(path, name, tipo.clone())?;
                    }
                    Statement::Equality { left, right } => {
                        right.resolve_defined(root_container, path, material, false)?;
                        left.resolve_definition_top_down(
                            right.get_type(),
                            root_container,
                            path,
                            material,
                        )?;
                    }
                    Statement::Reference { reference, data } => {
                        data.resolve_defined(root_container, path, material, false)?;
                        reference.resolve_definition_top_down(
                            &Type::Reference(Arc::new(data.get_type().clone())),
                            root_container,
                            path,
                            material,
                        )?;
                    }
                    Statement::Dereference { data, reference } => {
                        reference.resolve_defined(root_container, path, material, false)?;
                        data.resolve_definition_top_down(
                            reference.get_type().get_reference_type(material)?,
                            root_container,
                            path,
                            material,
                        )?;
                    }
                    Statement::EmptyUnderConstructionArray { array, elem_type } => {
                        array.resolve_definition_top_down(
                            &Type::UnderConstructionArray(Arc::new(elem_type.clone())),
                            root_container,
                            path,
                            material,
                        )?;
                    }
                    Statement::UnderConstructionArrayPrepend {
                        new_array,
                        elem,
                        old_array,
                    } => {
                        old_array.resolve_defined(root_container, path, material, false)?;
                        elem.resolve_defined(root_container, path, material, false)?;
                        if !material.same_type(
                            old_array.get_type(),
                            &Type::UnderConstructionArray(Arc::new(elem.get_type().clone())),
                        ) {
                            return Err(CompilationError::UnexpectedType(
                                old_array.get_type().clone(),
                                Type::UnderConstructionArray(Arc::new(elem.get_type().clone())),
                            ));
                        }
                        new_array.resolve_definition_top_down(
                            old_array.get_type(),
                            root_container,
                            &path,
                            material,
                        )?;
                    }
                    Statement::ArrayFinalization {
                        finalized,
                        under_construction,
                    } => {
                        under_construction.resolve_defined(
                            root_container,
                            path,
                            material,
                            false,
                        )?;
                        finalized.resolve_definition_top_down(
                            &Type::Array(Arc::new(
                                under_construction
                                    .get_type()
                                    .get_under_construction_array_type(material)?
                                    .clone(),
                            )),
                            root_container,
                            &path,
                            material,
                        )?;
                    }
                    Statement::ArrayAccess { elem, array, index } => {
                        index.resolve_defined(root_container, path, material, false)?;
                        if !material.same_type(index.get_type(), &Type::Field) {
                            return Err(CompilationError::NotAnIndex(elem.get_type().clone()));
                        }
                        array.resolve_defined(root_container, path, material, false)?;
                        elem.resolve_definition_top_down(
                            array.get_type().get_array_type(material)?,
                            root_container,
                            &path,
                            material,
                        )?;
                    }
                }
            }
            Atom::PartialFunctionCall(index, stage) => {
                let FlatFunctionCall {
                    material,
                    scope: path,
                    function_name,
                    arguments,
                } = &mut self.function_calls[index];
                let material = *material;

                let inline = root_container
                    .function_set
                    .get_function(function_name)?
                    .function
                    .inline;

                for i in stage.start..stage.mid {
                    let argument = &mut arguments[i];
                    argument.resolve_defined(root_container, path, material, inline)?;
                    if !material.same_type(
                        argument.get_type(),
                        root_container
                            .function_set
                            .get_function(function_name)?
                            .argument_type(i),
                    ) {
                        return Err(CompilationError::IncorrectTypeForArgument(
                            function_name.clone(),
                            i,
                            argument.get_type().clone(),
                            root_container
                                .function_set
                                .get_function(function_name)?
                                .argument_type(i)
                                .clone(),
                        ));
                    }
                }
                for i in stage.mid..stage.end {
                    let expected_type = root_container
                        .function_set
                        .get_function(function_name)?
                        .argument_type(i)
                        .clone();
                    let argument = &mut arguments[i];
                    argument.resolve_definition_top_down(
                        &expected_type,
                        root_container,
                        path,
                        material,
                    )?;
                }
            }
        }
        Ok(())
    }

    fn resolve_representation(
        &self,
        atom: Atom,
        root_container: &mut RootContainer,
    ) -> Result<(), CompilationError> {
        match atom {
            Atom::Match(flat_index) => {
                let FlatMatch {
                    material: _,
                    scope,
                    index,
                    value,
                    branches,
                } = &self.matches[flat_index];
                value.resolve_representation(root_container, scope)?;
                for (constructor, components) in branches {
                    let new_path = scope.then(*index, constructor.clone());
                    for component in components.iter() {
                        root_container.represent(&new_path, component)?;
                    }
                }
            }
            Atom::InArgument(argument_index) => {
                self.arguments[argument_index]
                    .resolve_representation(root_container, &ScopePath::empty())?;
            }
            Atom::OutArgument(_) => unreachable!(),
            Atom::Statement(index) => {
                let FlatStatement {
                    material: _,
                    scope: path,
                    statement,
                } = &self.statements[index];
                match statement {
                    Statement::VariableDeclaration {
                        name, represents, ..
                    } => {
                        if *represents {
                            root_container.represent(path, name)?;
                        }
                    }
                    Statement::Equality { left, right: _ } => {
                        left.resolve_representation(root_container, path)?;
                    }
                    Statement::Reference { reference, data: _ } => {
                        reference.resolve_representation(root_container, path)?;
                    }
                    Statement::Dereference { data, reference: _ } => {
                        data.resolve_representation(root_container, path)?;
                    }
                    Statement::EmptyUnderConstructionArray { array, .. } => {
                        array.resolve_representation(root_container, path)?;
                    }
                    Statement::UnderConstructionArrayPrepend {
                        new_array,
                        elem: _,
                        old_array: _,
                    } => {
                        new_array.resolve_representation(root_container, path)?;
                    }
                    Statement::ArrayFinalization {
                        finalized,
                        under_construction: _,
                    } => {
                        finalized.resolve_representation(root_container, path)?;
                    }
                    Statement::ArrayAccess {
                        elem,
                        array: _,
                        index: _,
                    } => {
                        elem.resolve_representation(root_container, path)?;
                    }
                }
            }
            Atom::PartialFunctionCall(index, stage) => {
                let FlatFunctionCall {
                    material: _,
                    scope: path,
                    function_name,
                    arguments,
                } = &self.function_calls[index];
                let inline = root_container
                    .function_set
                    .get_function(function_name)?
                    .function
                    .inline;

                if inline {
                    for i in stage.start..stage.mid {
                        let argument = &arguments[i];
                        argument.resolve_representation(root_container, path)?;
                    }
                }
                for i in stage.mid..stage.end {
                    let argument = &arguments[i];
                    argument.resolve_representation(root_container, path)?;
                }
            }
        }
        Ok(())
    }
}
