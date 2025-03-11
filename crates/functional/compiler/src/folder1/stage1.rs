use std::{collections::HashMap, sync::Arc};

use super::{
    error::CompilationError, file3::FlattenedFunction, function_resolution::FunctionSet,
    ir::Program, type_resolution::TypeSet,
};

#[derive(Debug)]
pub struct Stage2Program {
    pub types: TypeSet,
    pub functions: HashMap<String, FlattenedFunction>,
}

pub fn stage1(program: Program) -> Result<Stage2Program, CompilationError> {
    let types = Arc::new(TypeSet::new(program.algebraic_types)?);
    let function_set = Arc::new(FunctionSet::new(program.functions)?);

    let mut functions = HashMap::new();
    let mut inlined_functions = HashMap::new();

    for function_name in function_set.function_order.iter() {
        let mut function = FlattenedFunction::create_flattened(
            function_set.functions[function_name].clone(),
            types.clone(),
            function_set.clone(),
            functions.len(),
        )?;
        function.perform_inlining(&inlined_functions);
        if function.inline {
            inlined_functions.insert(function_name.clone(), function);
        } else {
            functions.insert(function_name.clone(), function);
        }
    }

    loop {
        let mut updated = false;
        let function_names = functions.keys().cloned().collect::<Vec<_>>();
        for name in function_names {
            let mut function = functions.remove(&name).unwrap();
            updated |= function.update_uses_timestamp(&functions);
            functions.insert(name, function);
        }
        if !updated {
            break;
        }
    }

    Ok(Stage2Program {
        types: (*types).clone(),
        functions,
    })
}

impl FlattenedFunction {
    pub fn update_uses_timestamp(
        &mut self,
        functions: &HashMap<String, FlattenedFunction>,
    ) -> bool {
        if self.uses_timestamp {
            return false;
        }

        for function_call in self.function_calls.iter() {
            let callee = &functions[&function_call.function_name];
            if callee.uses_timestamp {
                self.uses_timestamp = true;
                return true;
            }
        }

        false
    }
}
