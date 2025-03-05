use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use super::{
    error::CompilationError,
    ir::{ArgumentBehavior, Body, Function, Type},
};

pub struct FunctionSet {
    pub(crate) functions: HashMap<String, FunctionContainer>,
    pub(crate) function_order: Vec<String>,
}

impl FunctionSet {
    pub fn new(mut functions: Vec<Function>) -> Result<Self, CompilationError> {
        let mut defined_functions = HashSet::new();
        for function in functions.iter() {
            if !function.inline {
                defined_functions.insert(function.name.clone());
            }
        }
        let mut functions_map = HashMap::new();

        let mut function_order = Vec::new();
        while !functions.is_empty() {
            let i = functions
                .iter()
                .position(|function| {
                    Self::body_referenced_functions(&function.body)
                        .iter()
                        .all(|name| defined_functions.contains(name))
                })
                .ok_or(CompilationError::InlineFunctionsSelfReferential())?;
            let function = functions.remove(i);
            function_order.push(function.name.clone());
            defined_functions.insert(function.name.clone());
            functions_map.insert(function.name.clone(), FunctionContainer::new(function));
        }

        Ok(Self {
            functions: functions_map,
            function_order,
        })
    }
    fn body_referenced_functions(body: &Body) -> Vec<String> {
        let mut result = Vec::new();
        for (_, function_call) in body.function_calls.iter() {
            result.push(function_call.function.clone());
        }
        for matchi in body.matches.iter() {
            for branch in matchi.branches.iter() {
                result.extend(Self::body_referenced_functions(&branch.body));
            }
        }
        result
    }
    pub fn get_function(&self, name: &String) -> Result<&FunctionContainer, CompilationError> {
        if let Some(function) = self.functions.get(name) {
            Ok(function)
        } else {
            Err(CompilationError::UndefinedFunction(name.clone()))
        }
    }
}

#[derive(Clone)]
pub struct FunctionContainer {
    pub stages: Vec<Stage>,
    pub function: Arc<Function>,
}

impl FunctionContainer {
    pub fn new(function: Function) -> Self {
        let mut changes = vec![0];
        let mut last = ArgumentBehavior::In;
        for (i, argument) in function.arguments.iter().enumerate() {
            if argument.behavior != last {
                changes.push(i);
                last = argument.behavior;
            }
        }
        changes.push(function.arguments.len());
        if changes.len() % 2 == 0 {
            changes.push(function.arguments.len());
        }
        let mut stages = Vec::new();
        for i in 0..changes.len() / 2 {
            stages.push(Stage {
                index: i,
                start: changes[2 * i],
                mid: changes[2 * i + 1],
                end: changes[2 * i + 2],
            });
        }
        Self {
            stages,
            function: Arc::new(function),
        }
    }

    pub fn argument_type(&self, index: usize) -> &Type {
        &self.function.arguments[index].tipo
    }
}

#[derive(Clone, Copy)]
pub struct Stage {
    pub index: usize,
    pub start: usize,
    pub mid: usize,
    pub end: usize,
}
