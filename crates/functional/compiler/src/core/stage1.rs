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

    Ok(Stage2Program {
        types: (*types).clone(),
        functions,
    })
}
