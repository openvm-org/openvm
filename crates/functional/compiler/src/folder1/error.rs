use super::ir::Type;

#[derive(Debug)]
pub enum CompilationError {
    UndeclaredVariable(String),
    UndefinedVariable(String),
    UnrepresentedVariable(String),
    CannotAssignHere(String),
    CannotRepresentHere(String),
    CannotEquateReferences(Type),
    ReferenceDefinitionMustBeMaterialized(String),
    IncorrectNumberOfComponents(String, usize, usize),
    IncorrectTypeInComponent(String, usize, Type, Type),
    UnexpectedType(Type, Type),
    UnexpectedUnmaterialized(Type),
    IncorrectTypeInArithmetic(Type),
    IncorrectTypesInEquality(Type, Type),
    DuplicateDeclaration(String),
    DuplicateDefinition(String),
    DuplicateRepresentation(String),
    DuplicateUnderConstructionArrayUsage(String),
    UndefinedType(String),
    UndefinedConstructor(String),

    NotAReference(Type),
    NotAnUnderConstructionArray(Type),
    NotAnArray(Type),
    NotAnIndex(Type),
    NotANamedType(Type),
    NotAConstArray(Type),

    UndefinedFunction(String),
    IncorrectNumberOfArgumentsToFunction(String, usize, usize),
    IncorrectTypeForArgument(String, usize, Type, Type),
    IncorrectTypeForOutArgument(usize, Type, Type),

    CannotOrderStatementsForDefinition(),
    CannotOrderStatementsForRepresentation(),
    TypesSelfReferential(),
    InlineFunctionsSelfReferential(),

    CannotInferEmptyConstArrayType(),
    ElementTypesInconsistentInConstArray(Type, Type, usize, usize),
    ElementTypesInconsistentInConstArrayConcatenation(Type, Type),
    OutOfBoundsConstArrayAccess(usize, usize),
    OutOfBoundsConstArraySlice(usize, usize, usize),
    IncorrectNumberOfElementsInConstArray(usize, usize),
    DuplicateUnderConstructionArrayUsageInConstArray,

    EqMustBeDematerialized(),
    DivMustBeDematerialized(),

    DuplicateConstructorName(String),
    DuplicateTypeName(String),
    TypeNameCannotBeF,
}
