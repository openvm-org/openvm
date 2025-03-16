use crate::core::{
    containers::RootContainer,
    error::CompilationError,
    file3::{Atom, FlatMatch, FlattenedFunction},
};

impl FlattenedFunction {
    pub fn resolve_definition(
        &mut self,
        atom: Atom,
        root_container: &mut RootContainer,
    ) -> Result<(), CompilationError> {
        let scope = self.scope(atom);
        for child in self.children(atom) {
            child.resolve_definition(root_container, scope)?;
        }
        for declared_name in self.declared_names(atom) {
            if declared_name.defines {
                root_container.define(
                    &declared_name.scope,
                    &declared_name.name,
                    &declared_name.parser_metadata,
                )?;
            }
        }
        if let Atom::Match(flat_index) = atom {
            let FlatMatch {
                index, branches, ..
            } = &self.matches[flat_index];
            root_container.root_scope.activate_children(
                scope,
                *index,
                branches
                    .iter()
                    .map(|branch| branch.constructor.clone())
                    .collect(),
            );
        }
        Ok(())
    }

    pub fn resolve_representation(
        &self,
        atom: Atom,
        root_container: &mut RootContainer,
    ) -> Result<(), CompilationError> {
        let scope = self.scope(atom);
        for child in self.children(atom) {
            child.resolve_representation(root_container, scope)?;
        }
        for declared_name in self.declared_names(atom) {
            if declared_name.represents {
                root_container.represent(
                    &declared_name.scope,
                    &declared_name.name,
                    &declared_name.parser_metadata,
                )?;
            }
        }
        Ok(())
    }
}
