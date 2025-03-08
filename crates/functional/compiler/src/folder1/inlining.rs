use std::{collections::HashMap, mem::take};

use super::{
    file2_tree::{ExpressionContainer, ScopePath},
    file3::{Atom, FlattenedFunction},
    ir::{Expression, Statement},
};
use crate::folder1::{
    file3::{FlatFunctionCall, FlatMatch, FlatStatement, RepresentationOrder},
    ir::Material,
};

struct Renamer {}

impl Renamer {
    pub fn new(identity: usize) -> Self {
        todo!()
    }
    pub fn rename(&self, name: &mut String) {
        todo!()
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
            } = &function_call;
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
                if *material == Material::Materialized && inlined_callee.uses_timestamp {
                    self.uses_timestamp = true;
                }

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
            argument.inline(renamer);
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

        let mut atom_replacements = HashMap::new();
        let mut atoms_staged = take(&mut self.atoms_staged);
        for stage in atoms_staged.iter_mut() {
            for atom in stage.iter_mut() {
                let old_atom = *atom;
                atom.inline(self, target, &argument_fulfillments, path_prefix, material);
                atom_replacements.insert(old_atom, *atom);
            }
        }
        self.atoms_staged = atoms_staged;
        match &mut self.representation_order {
            RepresentationOrder::Inline(atoms_staged) => {
                for stage in atoms_staged.iter_mut() {
                    for atom in stage.iter_mut() {
                        *atom = atom_replacements[atom];
                    }
                }
            }
            RepresentationOrder::NotInline(atoms_staged) => {
                for atom in atoms_staged.iter_mut() {
                    *atom = atom_replacements[atom];
                }
            }
        }
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
        } = self;
        *material &= imposed_material;
        scope.prepend(path_prefix, path_offset);
        match statement {
            Statement::VariableDeclaration { name, .. } => {
                renamer.rename(name);
            }
            Statement::Equality { left, right } => {
                left.inline(renamer);
                right.inline(renamer);
            }
            Statement::Reference { reference, data } => {
                reference.inline(renamer);
                data.inline(renamer);
            }
            Statement::Dereference { data, reference } => {
                data.inline(renamer);
                reference.inline(renamer);
            }
            Statement::EmptyUnderConstructionArray { array, .. } => {
                array.inline(renamer);
            }
            Statement::UnderConstructionArrayPrepend {
                new_array,
                elem,
                old_array,
            } => {
                new_array.inline(renamer);
                elem.inline(renamer);
                old_array.inline(renamer);
            }
            Statement::ArrayFinalization {
                finalized,
                under_construction,
            } => {
                finalized.inline(renamer);
                under_construction.inline(renamer);
            }
            Statement::ArrayAccess { elem, array, index } => {
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
            material,
            scope,
            index,
            value,
            branches,
        } = self;
        scope.prepend(path_prefix, path_offset);
        *material &= imposed_material;
        value.inline(renamer);
        *index += path_offset;

        for (_, components) in branches.iter_mut() {
            for component in components.iter_mut() {
                renamer.rename(component);
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
        } = self;
        *material &= imposed_material;
        scope.prepend(path_prefix, path_offset);
        for argument in arguments.iter_mut() {
            argument.inline(renamer);
        }
    }
}

impl Atom {
    pub fn inline(
        &mut self,
        home: &mut FlattenedFunction,
        target: &FlattenedFunction,
        argument_fulfillments: &Vec<ExpressionContainer>,
        path_prefix: &ScopePath,
        imposed_material: Material,
    ) {
        match self {
            Atom::InArgument(argument_index) => {
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
            }
            Atom::Match(index) => *index += target.matches.len(),
            Atom::Statement(index) => *index += target.statements.len(),
            Atom::PartialFunctionCall(index, _) => *index += target.function_calls.len(),
        }
    }
}

impl ExpressionContainer {
    fn inline(&mut self, renamer: &Renamer) {
        match self.expression.as_mut() {
            Expression::Constant { .. } => {}
            Expression::Variable { name, .. } => {
                renamer.rename(name);
            }
            Expression::Algebraic {
                constructor: _,
                fields,
            } => {
                for field in fields.iter_mut() {
                    field.inline(renamer);
                }
            }
            Expression::Arithmetic {
                operator: _,
                left,
                right,
            } => {
                left.inline(renamer);
                right.inline(renamer);
            }
            Expression::Dematerialized { value } => {
                value.inline(renamer);
            }
            Expression::Eq { left, right } => {
                left.inline(renamer);
                right.inline(renamer);
            }
            Expression::EmptyConstArray { elem_type: _ } => {}
            Expression::ConstArray { elements } => {
                for element in elements.iter_mut() {
                    element.inline(renamer);
                }
            }
            Expression::ConstArrayConcatenation { left, right } => {
                left.inline(renamer);
                right.inline(renamer);
            }
            Expression::ConstArrayAccess { array, index: _ } => {
                array.inline(renamer);
            }
            Expression::ConstArraySlice {
                array,
                from: _,
                to: _,
            } => {
                array.inline(renamer);
            }
            Expression::ConstArrayRepeated { element, length: _ } => {
                element.inline(renamer);
            }
        };
    }
}
