use std::{collections::HashMap, mem::take, sync::Arc};

use super::{
    error::CompilationError,
    function_resolution::{FunctionContainer, FunctionSet},
    ir::{Body, StatementVariant},
    type_resolution::TypeSet,
};
use crate::{
    core::{
        containers::{Assertion, DeclarationSet, ExpressionContainer, RootContainer},
        function_resolution::Stage,
        ir::{Argument, Branch, BranchComponent, FunctionCall, Material, Statement},
        scope::ScopePath,
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
    pub(crate) assertions: Vec<Assertion>,

    pub(crate) name: String,
    pub(crate) creates_addresses: bool,
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
                StatementVariant::EmptyPrefix { .. } => uses_timestamp = true,
                _ => {}
            }
        }
        let mut root_container =
            RootContainer::new(type_set, function_set, function.clone(), &tree);

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
            assertions: vec![],
            creates_addresses: uses_timestamp,
            function_id,
            parser_metadata: function.function.parser_metadata.clone(),
        };

        flattened_function.order_for_types(&mut root_container)?;
        flattened_function.order_for_execution(&mut root_container)?;
        flattened_function.order_for_representation(&mut root_container)?;
        root_container.root_scope.verify_completeness(
            root_container.type_set.as_ref(),
            &flattened_function.parser_metadata,
        )?;
        root_container
            .root_scope
            .unpack(&mut flattened_function.declaration_set);
        flattened_function.assertions = take(&mut root_container.assertions);

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
}
#[derive(Clone, Copy)]
pub enum DeclarationResolutionStep {
    Statement(usize),
    Match(usize),
    OwnArgument(usize),
    CallArgument(usize, usize),
}
impl FlattenedFunction {
    fn order_for_types(
        &mut self,
        root_container: &mut RootContainer,
    ) -> Result<(), CompilationError> {
        let mut disp_steps = vec![];
        for i in 0..self.statements.len() {
            disp_steps.push(DeclarationResolutionStep::Statement(i));
        }
        for i in 0..self.matches.len() {
            disp_steps.push(DeclarationResolutionStep::Match(i));
        }
        for (i, function_call) in self.function_calls.iter().enumerate() {
            for j in 0..function_call.arguments.len() {
                disp_steps.push(DeclarationResolutionStep::CallArgument(i, j));
            }
        }
        for i in 0..self.arguments.len() {
            disp_steps.push(DeclarationResolutionStep::OwnArgument(i));
        }
        while !disp_steps.is_empty() {
            let mut i: Option<usize> = None;
            for (j, step) in disp_steps.iter().enumerate() {
                if self.viable_for_declaration(root_container, *step)? {
                    i = Some(j);
                }
            }
            match i {
                None => {
                    return Err(CompilationError::CannotOrderStatementsForDeclaration(
                        self.parser_metadata.clone(),
                    ))
                }
                Some(i) => {
                    let step = disp_steps.remove(i);
                    self.resolve_types(step, root_container)?;
                }
            }
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
                root_container.current_function.stages[0],
            ));
        }
        root_container.root_scope.activate();

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
                        self.resolve_representation(atom, root_container)?;
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
                    self.resolve_representation(atom, root_container)?;
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
