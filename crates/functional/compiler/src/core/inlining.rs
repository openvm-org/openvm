use std::{collections::HashMap, mem::take};

use super::{
    file3::{Atom, FlattenedFunction},
    ir::{Expression, StatementVariant},
};
use crate::{
    core::{
        containers::{Assertion, ExpressionContainer},
        file3::{FlatFunctionCall, FlatMatch, FlatStatement, RepresentationOrder},
        ir::Material,
        scope::ScopePath,
    },
    parser::metadata::ParserMetadata,
};

pub struct Renamer {
    identity: usize,
}

impl Renamer {
    pub fn new(identity: usize) -> Self {
        Self { identity }
    }
    pub fn rename(&self, name: &mut String) {
        *name = format!("inline{}_{}", self.identity, name);
    }
}

enum InlineResult {
    Inlined(FlattenedFunction),
    NotInlined(usize),
}

impl FlattenedFunction {
    pub fn perform_inlining(&mut self, functions: &HashMap<String, FlattenedFunction>) {
        let old_function_calls = take(&mut self.function_calls);
        let mut num_inlines = 0;
        let mut inline_results = vec![];
        for function_call in old_function_calls {
            let FlatFunctionCall {
                material,
                scope,
                function_name,
                arguments,
                parser_metadata: _,
            } = &function_call;
            // hacky, would be nice some other way
            if function_name == &self.name {
                inline_results.push(InlineResult::NotInlined(self.function_calls.len()));
                self.function_calls.push(function_call.clone());
                continue;
            }
            let callee = &functions[function_name];
            if callee.inline {
                let path_offset = self.tree.num_branches(scope);
                self.tree.insert(scope, &callee.tree);

                let mut inlined_callee = callee.clone();
                inlined_callee.inline(
                    self,
                    path_offset,
                    scope,
                    &Renamer::new(num_inlines),
                    *material,
                    arguments.clone(),
                );

                self.declaration_set
                    .declarations
                    .extend(inlined_callee.declaration_set.declarations.clone());
                self.statements.extend(inlined_callee.statements.clone());
                self.matches.extend(inlined_callee.matches.clone());
                self.function_calls
                    .extend(inlined_callee.function_calls.clone());

                inline_results.push(InlineResult::Inlined(inlined_callee));
                num_inlines += 1;
            } else {
                inline_results.push(InlineResult::NotInlined(self.function_calls.len()));
                self.function_calls.push(function_call.clone());
            }
        }

        for atoms in self.atoms_staged.iter_mut() {
            let old_atoms = take(atoms);

            for atom in old_atoms {
                if let Atom::PartialFunctionCall(index, stage) = atom {
                    match &inline_results[index] {
                        InlineResult::Inlined(inlined_callee) => {
                            atoms.extend(inlined_callee.atoms_staged[stage.index].clone());
                        }
                        InlineResult::NotInlined(new_index) => {
                            atoms.push(Atom::PartialFunctionCall(*new_index, stage));
                        }
                    }
                } else {
                    atoms.push(atom);
                }
            }
        }
    }

    pub fn inline(
        &mut self,
        target: &FlattenedFunction,
        path_offset: usize,
        path_prefix: &ScopePath,
        renamer: &Renamer,
        material: Material,
        argument_fulfillments: Vec<ExpressionContainer>,
    ) {
        for argument in self.arguments.iter_mut() {
            renamer.rename(&mut argument.name);
        }

        let old_declaration_set = take(&mut self.declaration_set);
        for ((path, mut name), tipo) in old_declaration_set.declarations {
            renamer.rename(&mut name);
            self.declaration_set.insert(
                &path_prefix.concat(path_offset, &path),
                &name,
                &material.wrap(&tipo),
            );
        }

        for statement in self.statements.iter_mut() {
            statement.inline(renamer, path_offset, path_prefix, material);
        }
        for matchi in self.matches.iter_mut() {
            matchi.inline(renamer, path_offset, path_prefix, material);
        }
        for function_call in self.function_calls.iter_mut() {
            function_call.inline(renamer, path_offset, path_prefix, material);
        }
        for assertion in self.assertions.iter_mut() {
            assertion.inline(renamer, path_offset, path_prefix, material);
        }

        let mut atom_replacements = HashMap::new();
        let mut atoms_staged = take(&mut self.atoms_staged);
        for stage in atoms_staged.iter_mut() {
            for atom in stage.iter_mut() {
                let old_atom = *atom;
                atom.inline(target);
                atom_replacements.insert(old_atom, *atom);
            }
        }
        self.atoms_staged = atoms_staged;

        let mut atoms_staged_representation = match &mut self.representation_order {
            RepresentationOrder::Inline(atoms_staged) => take(atoms_staged),
            RepresentationOrder::NotInline(_) => unreachable!(),
        };

        for stage in atoms_staged_representation.iter_mut() {
            for atom in stage.iter_mut() {
                *atom = atom_replacements[atom];
            }
        }

        for stage in self.stages.iter() {
            for i in stage.start..stage.mid {
                if !self.arguments[i].represents {
                    panic!("representation via inline functions is currently limited to matching definition order (argument {})", i);
                }
                let index = target.statements.len() + self.statements.len();
                let mut argument_expression = ExpressionContainer::new(Expression::Variable {
                    name: self.arguments[i].name.clone(),
                    declares: true,
                    defines: true,
                    represents: self.arguments[i].represents,
                });
                argument_expression.tipo = Some(self.arguments[i].tipo.clone());
                self.statements.push(FlatStatement {
                    material,
                    scope: path_prefix.clone(),
                    statement: StatementVariant::Equality {
                        left: argument_expression,
                        right: argument_fulfillments[i].clone(),
                    },
                    parser_metadata: ParserMetadata::default(),
                });
                self.atoms_staged[stage.index].insert(0, Atom::Statement(index));
                atoms_staged_representation[stage.index].insert(0, Atom::Statement(index));
            }
            for i in stage.mid..stage.end {
                if self.arguments[i].represents {
                    panic!("representation via inline functions is currently limited to matching definition order (argument {})", i);
                }
                let declaration_index = target.statements.len() + self.statements.len();
                let out_index = declaration_index + 1;
                self.statements.push(FlatStatement {
                    material,
                    scope: path_prefix.clone(),
                    statement: StatementVariant::VariableDeclaration {
                        name: self.arguments[i].name.clone(),
                        tipo: self.arguments[i].tipo.clone(),
                        represents: false,
                    },
                    parser_metadata: ParserMetadata::default(),
                });
                let mut argument_expression = ExpressionContainer::new(Expression::Variable {
                    name: self.arguments[i].name.clone(),
                    declares: false,
                    defines: false,
                    represents: self.arguments[i].represents,
                });
                argument_expression.tipo = Some(self.arguments[i].tipo.clone());
                self.atoms_staged[0].insert(0, Atom::Statement(declaration_index));
                self.atoms_staged[stage.index].push(Atom::Statement(out_index));
                atoms_staged_representation[0].insert(0, Atom::Statement(declaration_index));
                atoms_staged_representation[stage.index].push(Atom::Statement(out_index));
            }
        }

        self.representation_order = RepresentationOrder::Inline(atoms_staged_representation);
    }
}

impl FlatStatement {
    pub fn inline(
        &mut self,
        renamer: &Renamer,
        path_offset: usize,
        path_prefix: &ScopePath,
        imposed_material: Material,
    ) {
        let FlatStatement {
            material,
            scope,
            statement,
            parser_metadata: _,
        } = self;
        *material &= imposed_material;
        scope.prepend(path_prefix, path_offset);
        match statement {
            StatementVariant::VariableDeclaration { name, .. } => {
                renamer.rename(name);
            }
            StatementVariant::Equality { left, right } => {
                left.inline(renamer);
                right.inline(renamer);
            }
            StatementVariant::Reference { reference, data } => {
                reference.inline(renamer);
                data.inline(renamer);
            }
            StatementVariant::Dereference { data, reference } => {
                data.inline(renamer);
                reference.inline(renamer);
            }
            StatementVariant::EmptyPrefix { prefix: array, .. } => {
                array.inline(renamer);
            }
            StatementVariant::PrefixAppend {
                new_prefix: new_array,
                elem,
                old_prefix: old_array,
            } => {
                new_array.inline(renamer);
                elem.inline(renamer);
                old_array.inline(renamer);
            }
            StatementVariant::PrefixAccess {
                elem,
                prefix: array,
                index,
            } => {
                elem.inline(renamer);
                array.inline(renamer);
                index.inline(renamer);
            }
        }
    }
}

impl FlatMatch {
    pub fn inline(
        &mut self,
        renamer: &Renamer,
        path_offset: usize,
        path_prefix: &ScopePath,
        imposed_material: Material,
    ) {
        let FlatMatch {
            check_material: material,
            scope,
            index,
            value,
            branches,
            parser_metadata: _,
        } = self;
        scope.prepend(path_prefix, path_offset);
        *material &= imposed_material;
        value.inline(renamer);
        *index += path_offset;

        for branch in branches.iter_mut() {
            for component in branch.components.iter_mut() {
                renamer.rename(&mut component.name);
            }
        }
    }
}

impl FlatFunctionCall {
    pub fn inline(
        &mut self,
        renamer: &Renamer,
        path_offset: usize,
        path_prefix: &ScopePath,
        imposed_material: Material,
    ) {
        let FlatFunctionCall {
            material,
            scope,
            function_name: _,
            arguments,
            parser_metadata: _,
        } = self;
        *material &= imposed_material;
        scope.prepend(path_prefix, path_offset);
        for argument in arguments.iter_mut() {
            argument.inline(renamer);
        }
    }
}

impl Assertion {
    pub fn inline(
        &mut self,
        renamer: &Renamer,
        path_offset: usize,
        path_prefix: &ScopePath,
        imposed_material: Material,
    ) {
        let Assertion {
            left,
            right,
            scope,
            material,
        } = self;
        *material &= imposed_material;
        scope.prepend(path_prefix, path_offset);
        left.inline(renamer);
        right.inline(renamer);
    }
}

impl Atom {
    pub fn inline(&mut self, target: &FlattenedFunction) {
        match self {
            /*Atom::InArgument(argument_index) => {
                home.statements.push(FlatStatement {
                    material: imposed_material,
                    scope: path_prefix.clone(),
                    statement: Statement::Equality {
                        left: home.arguments[*argument_index].clone(),
                        right: argument_fulfillments[*argument_index].clone(),
                    },
                });
                *self = Atom::Statement(target.statements.len() + home.statements.len() - 1);
            }
            Atom::OutArgument(argument_index) => {
                home.statements.push(FlatStatement {
                    material: imposed_material,
                    scope: path_prefix.clone(),
                    statement: Statement::Equality {
                        left: argument_fulfillments[*argument_index].clone(),
                        right: home.arguments[*argument_index].clone(),
                    },
                });
                *self = Atom::Statement(target.statements.len() + home.statements.len() - 1);
            }*/
            Atom::Match(index) => *index += target.matches.len(),
            Atom::Statement(index) => *index += target.statements.len(),
            Atom::PartialFunctionCall(index, _) => *index += target.function_calls.len(),
        }
    }
}

impl ExpressionContainer {
    fn inline(&mut self, renamer: &Renamer) {
        match self.expression.as_mut() {
            Expression::Variable { name, .. } => {
                renamer.rename(name);
            }
            _ => self
                .children_mut()
                .iter_mut()
                .for_each(|child| child.inline(renamer)),
        };
    }
}
