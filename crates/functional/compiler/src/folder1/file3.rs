use std::{collections::HashMap, sync::Arc};

use itertools::Itertools;

use super::{
    error::CompilationError,
    file2_tree::{DeclarationSet, ExpressionContainer, RootContainer, ScopePath},
    function_resolution::{FunctionContainer, FunctionSet},
    ir::{Body, StatementVariant, Type},
    type_resolution::TypeSet,
};
use crate::{
    folder1::{
        file2_tree::DependencyType,
        function_resolution::Stage,
        ir::{Argument, Branch, BranchComponent, FunctionCall, Material, Statement},
    },
    parser::metadata::ParserMetadata,
};

#[derive(Clone, Debug)]
pub enum RepresentationOrder {
    Inline(Vec<Vec<Atom>>),
    NotInline(Vec<Atom>),
}

impl Default for RepresentationOrder {
    fn default() -> Self {
        Self::NotInline(Vec::default())
    }
}

#[derive(Clone, Debug)]
pub struct FlattenedFunction {
    pub(crate) inline: bool,
    pub(crate) stages: Vec<Stage>,
    pub(crate) arguments: Vec<Argument>,

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

    pub(crate) parser_metadata: ParserMetadata,
}

#[derive(Clone, Debug)]
pub struct FlatStatement {
    pub material: Material,
    pub scope: ScopePath,
    pub statement: StatementVariant,
    pub parser_metadata: ParserMetadata,
}

#[derive(Clone, Debug)]
pub struct FlatMatch {
    pub check_material: Material,
    pub scope: ScopePath,
    pub index: usize,
    pub value: ExpressionContainer,
    pub branches: Vec<FlatBranch>,
    pub parser_metadata: ParserMetadata,
}

#[derive(Clone, Debug)]
pub struct FlatBranch {
    pub constructor: String,
    pub components: Vec<BranchComponent>,
    pub parser_metadata: ParserMetadata,
}

#[derive(Clone, Debug)]
pub struct FlatFunctionCall {
    pub material: Material,
    pub scope: ScopePath,
    pub function_name: String,
    pub arguments: Vec<ExpressionContainer>,
    pub parser_metadata: ParserMetadata,
}

impl FlatFunctionCall {
    pub fn new(scope: ScopePath, function_call: FunctionCall) -> Self {
        Self {
            material: function_call.material,
            scope,
            function_name: function_call.function,
            arguments: function_call.arguments,
            parser_metadata: function_call.parser_metadata,
        }
    }
}

#[derive(Clone, Copy, Hash, PartialEq, Eq, Debug)]
pub enum Atom {
    Match(usize),
    Statement(usize),
    PartialFunctionCall(usize, Stage),
}

#[derive(Clone, Debug)]
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
        let arguments = function.function.arguments.clone();
        for argument in arguments.iter() {
            type_set.check_type_exists(&argument.tipo, &argument.parser_metadata)?;
        }
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
                StatementVariant::Reference { .. } => uses_timestamp = true,
                StatementVariant::ArrayFinalization { .. } => uses_timestamp = true,
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
            parser_metadata: function.function.parser_metadata.clone(),
        };

        let mut root_container = RootContainer::new(type_set, function_set, function);
        flattened_function.order_for_execution(&mut root_container)?;
        flattened_function.order_for_representation(&mut root_container)?;
        root_container.root_scope.verify_completeness(
            root_container.type_set.as_ref(),
            &flattened_function.parser_metadata,
        )?;
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
        for Statement {
            variant,
            material,
            parser_metadata,
        } in body.statements.clone()
        {
            statements.push(FlatStatement {
                material,
                scope: path.clone(),
                statement: variant,
                parser_metadata,
            });
        }
        for function_call in body.function_calls.clone() {
            function_calls.push(FlatFunctionCall::new(path.clone(), function_call));
        }
        for (i, matchi) in body.matches.iter().enumerate() {
            let branches = matchi
                .branches
                .iter()
                .map(
                    |Branch {
                         constructor,
                         components,
                         body: _,
                         parser_metadata,
                     }| FlatBranch {
                        constructor: constructor.clone(),
                        components: components.clone(),
                        parser_metadata: parser_metadata.clone(),
                    },
                )
                .collect();
            matches.push(FlatMatch {
                check_material: matchi.check_material,
                scope: path.clone(),
                index: i,
                value: matchi.value.clone(),
                parser_metadata: matchi.parser_metadata.clone(),
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
        for argument in self.arguments.iter() {
            root_container.declare(
                &ScopePath::empty(),
                &argument.name,
                argument.tipo.clone(),
                &argument.parser_metadata,
            )?;
        }

        for stage in root_container.current_function.stages.clone() {
            let mut ordered_atoms = Vec::new();
            for argument_index in stage.start..stage.mid {
                root_container.define(
                    &ScopePath::empty(),
                    &self.arguments[argument_index].name,
                    &self.arguments[argument_index].parser_metadata,
                )?;
            }
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
                            .get_function(
                                &self.function_calls[index].function_name,
                                &self.function_calls[index].parser_metadata,
                            )
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
            for argument_index in stage.mid..stage.end {
                if !root_container
                    .root_scope
                    .is_defined(&ScopePath::empty(), &self.arguments[argument_index].name)
                {
                    return Err(CompilationError::CannotOrderStatementsForDefinition(
                        self.parser_metadata.clone(),
                    ));
                }
            }
            self.atoms_staged.push(ordered_atoms);
        }
        if !disp_atoms.is_empty() {
            return Err(CompilationError::CannotOrderStatementsForDefinition(
                self.parser_metadata.clone(),
            ));
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
                .get_function(
                    &self.function_calls[i].function_name,
                    &self.function_calls[i].parser_metadata,
                )
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
                for i in stage.start..stage.mid {
                    if self.arguments[i].represents {
                        root_container.represent(
                            &ScopePath::empty(),
                            &self.arguments[i].name,
                            &self.arguments[i].parser_metadata,
                        )?;
                    }
                }
                loop {
                    let i = disp_atoms
                        .iter()
                        .position(|&atom| self.viable_for_representation(root_container, atom));
                    if let Some(i) = i {
                        let atom = disp_atoms.remove(i);
                        self.try_resolve_representation(atom, root_container)?;
                        ordered_atoms.push(atom);

                        if let Atom::PartialFunctionCall(index, _) = atom {
                            let callee = root_container
                                .function_set
                                .get_function(
                                    &self.function_calls[index].function_name,
                                    &self.function_calls[index].parser_metadata,
                                )
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
                for argument_index in stage.start..stage.end {
                    if !root_container.root_scope.is_represented(
                        &ScopePath::empty(),
                        &self.arguments[argument_index].name,
                        &root_container.type_set,
                    ) {
                        return Err(CompilationError::CannotOrderStatementsForRepresentation(
                            self.parser_metadata.clone(),
                        ));
                    }
                }
                atoms_staged.push(ordered_atoms);
            }
            self.representation_order = RepresentationOrder::Inline(atoms_staged);
        } else {
            for argument in self.arguments.iter() {
                if argument.represents {
                    root_container.represent(
                        &ScopePath::empty(),
                        &argument.name,
                        &argument.parser_metadata,
                    )?;
                }
            }

            let mut ordered_atoms = Vec::new();
            loop {
                let i = disp_atoms
                    .iter()
                    .position(|&atom| self.viable_for_representation(root_container, atom));
                if let Some(i) = i {
                    let atom = disp_atoms.remove(i);
                    self.try_resolve_representation(atom, root_container)?;
                    ordered_atoms.push(atom);

                    if let Atom::PartialFunctionCall(index, _) = atom {
                        let callee = root_container
                            .function_set
                            .get_function(
                                &self.function_calls[index].function_name,
                                &self.function_calls[index].parser_metadata,
                            )
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
            for argument in self.arguments.iter() {
                if !root_container.root_scope.is_represented(
                    &ScopePath::empty(),
                    &argument.name,
                    &root_container.type_set,
                ) {
                    return Err(CompilationError::CannotOrderStatementsForRepresentation(
                        self.parser_metadata.clone(),
                    ));
                }
            }
            self.representation_order = RepresentationOrder::NotInline(ordered_atoms);
        }
        if !disp_atoms.is_empty() {
            return Err(CompilationError::CannotOrderStatementsForRepresentation(
                self.parser_metadata.clone(),
            ));
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
        if let Atom::Match(index) = atom {
            for branch in self.matches[index].branches.iter() {
                let scope_here = self.scope(atom).then(index, branch.constructor.clone());
                for component in branch.components.iter() {
                    if !component.represents
                        && !root_container.root_scope.is_represented(
                            &scope_here,
                            &component.name,
                            &root_container.type_set,
                        )
                    {
                        return false;
                    }
                }
            }
        }
        let scope = &self.scope(atom);
        self.dependencies(atom, DependencyType::Representation)
            .iter()
            .all(|dependency| {
                root_container.root_scope.is_represented(
                    scope,
                    dependency,
                    &root_container.type_set,
                )
            })
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
            Atom::Statement(index) => match &self.statements[index].statement {
                StatementVariant::VariableDeclaration { .. } => vec![],
                StatementVariant::Equality { left, right } => left
                    .dependencies(dependency_type)
                    .into_iter()
                    .chain(right.dependencies(dependency_type))
                    .collect(),
                StatementVariant::Reference { reference, data } => reference
                    .dependencies(dependency_type)
                    .into_iter()
                    .chain(data.dependencies(dependency_type))
                    .collect(),
                StatementVariant::Dereference { data, reference } => data
                    .dependencies(dependency_type)
                    .into_iter()
                    .chain(reference.dependencies(dependency_type))
                    .collect(),
                StatementVariant::EmptyUnderConstructionArray { array, .. } => {
                    array.dependencies(dependency_type).into_iter().collect()
                }
                StatementVariant::UnderConstructionArrayPrepend {
                    new_array,
                    elem,
                    old_array,
                } => new_array
                    .dependencies(dependency_type)
                    .into_iter()
                    .chain(elem.dependencies(dependency_type))
                    .chain(old_array.dependencies(dependency_type))
                    .collect(),
                StatementVariant::ArrayFinalization {
                    finalized,
                    under_construction,
                } => finalized
                    .dependencies(dependency_type)
                    .into_iter()
                    .chain(under_construction.dependencies(dependency_type))
                    .collect(),
                StatementVariant::ArrayAccess { elem, array, index } => elem
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
                    check_material: material,
                    scope,
                    index,
                    value,
                    branches,
                    parser_metadata,
                } = &mut self.matches[flat_index];
                let material = *material;
                value.resolve_defined(root_container, scope, material, true)?;
                root_container.root_scope.define_new_scopes(
                    scope,
                    *index,
                    branches
                        .iter()
                        .map(|branch| branch.constructor.clone())
                        .collect(),
                );
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
                        root_container.define(
                            &new_path,
                            &component.name,
                            &branch.parser_metadata,
                        )?;
                    }
                }
            }
            Atom::Statement(index) => {
                let FlatStatement {
                    material,
                    scope: path,
                    statement,
                    parser_metadata,
                } = &mut self.statements[index];
                let material = *material;
                match statement {
                    StatementVariant::VariableDeclaration { name, tipo, .. } => {
                        root_container
                            .type_set
                            .check_type_exists(tipo, parser_metadata)?;
                        root_container.declare(path, name, tipo.clone(), parser_metadata)?;
                    }
                    StatementVariant::Equality { left, right } => {
                        right.resolve_defined(root_container, path, material, false)?;
                        left.resolve_definition_top_down(
                            right.get_type(),
                            root_container,
                            path,
                            material,
                        )?;
                    }
                    StatementVariant::Reference { reference, data } => {
                        data.resolve_defined(root_container, path, material, false)?;
                        reference.resolve_definition_top_down(
                            &Type::Reference(Arc::new(data.get_type().clone())),
                            root_container,
                            path,
                            material,
                        )?;
                    }
                    StatementVariant::Dereference { data, reference } => {
                        reference.resolve_defined(root_container, path, material, false)?;
                        data.resolve_definition_top_down(
                            reference
                                .get_type()
                                .get_reference_type(material, &reference.parser_metadata)?,
                            root_container,
                            path,
                            material,
                        )?;
                    }
                    StatementVariant::EmptyUnderConstructionArray { array, elem_type } => {
                        array.resolve_definition_top_down(
                            &Type::UnderConstructionArray(Arc::new(elem_type.clone())),
                            root_container,
                            path,
                            material,
                        )?;
                    }
                    StatementVariant::UnderConstructionArrayPrepend {
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
                                parser_metadata.clone(),
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
                    StatementVariant::ArrayFinalization {
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
                                    .get_under_construction_array_type(
                                        material,
                                        &under_construction.parser_metadata,
                                    )?
                                    .clone(),
                            )),
                            root_container,
                            &path,
                            material,
                        )?;
                    }
                    StatementVariant::ArrayAccess { elem, array, index } => {
                        index.resolve_defined(root_container, path, material, false)?;
                        if !material.same_type(index.get_type(), &Type::Field) {
                            return Err(CompilationError::NotAnIndex(
                                elem.parser_metadata.clone(),
                                elem.get_type().clone(),
                            ));
                        }
                        array.resolve_defined(root_container, path, material, false)?;
                        elem.resolve_definition_top_down(
                            array
                                .get_type()
                                .get_array_type(material, &array.parser_metadata)?,
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
                    parser_metadata,
                } = &mut self.function_calls[index];
                let material = *material;

                let callee = root_container
                    .function_set
                    .get_function(function_name, parser_metadata)?
                    .clone();

                let inline = callee.function.inline;

                for i in stage.start..stage.mid {
                    let argument = &mut arguments[i];
                    argument.resolve_defined(root_container, path, material, inline)?;
                    if !material.same_type(argument.get_type(), callee.argument_type(i)) {
                        return Err(CompilationError::IncorrectTypeForArgument(
                            argument.parser_metadata.clone(),
                            function_name.clone(),
                            i,
                            argument.get_type().clone(),
                            callee.argument_type(i).clone(),
                        ));
                    }
                }
                for i in stage.mid..stage.end {
                    let expected_type = callee.argument_type(i).clone();
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

    fn try_resolve_representation(
        &self,
        atom: Atom,
        root_container: &mut RootContainer,
    ) -> Result<(), CompilationError> {
        match atom {
            Atom::Match(flat_index) => {
                let FlatMatch {
                    check_material: _,
                    scope,
                    index,
                    value,
                    branches,
                    parser_metadata: _,
                } = &self.matches[flat_index];

                value.resolve_representation(root_container, scope)?;
                for branch in branches {
                    let new_path = scope.then(*index, branch.constructor.clone());
                    for component in branch.components.iter() {
                        if component.represents {
                            root_container.represent(
                                &new_path,
                                &component.name,
                                &branch.parser_metadata,
                            )?;
                        }
                    }
                }
            }
            Atom::Statement(index) => {
                let FlatStatement {
                    material: _,
                    scope: path,
                    statement,
                    parser_metadata,
                } = &self.statements[index];
                match statement {
                    StatementVariant::VariableDeclaration {
                        name, represents, ..
                    } => {
                        if *represents {
                            root_container.represent(path, name, parser_metadata)?;
                        }
                    }
                    StatementVariant::Equality { left, right: _ } => {
                        left.resolve_representation(root_container, path)?;
                    }
                    StatementVariant::Reference { reference, data: _ } => {
                        reference.resolve_representation(root_container, path)?;
                    }
                    StatementVariant::Dereference { data, reference: _ } => {
                        data.resolve_representation(root_container, path)?;
                    }
                    StatementVariant::EmptyUnderConstructionArray { array, .. } => {
                        array.resolve_representation(root_container, path)?;
                    }
                    StatementVariant::UnderConstructionArrayPrepend {
                        new_array,
                        elem: _,
                        old_array: _,
                    } => {
                        new_array.resolve_representation(root_container, path)?;
                    }
                    StatementVariant::ArrayFinalization {
                        finalized,
                        under_construction: _,
                    } => {
                        finalized.resolve_representation(root_container, path)?;
                    }
                    StatementVariant::ArrayAccess {
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
                    parser_metadata,
                } = &self.function_calls[index];
                let inline = root_container
                    .function_set
                    .get_function(function_name, parser_metadata)?
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
