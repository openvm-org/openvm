use super::ir::Type;
use crate::parser::metadata::ParserMetadata;

#[derive(Debug)]
pub enum CompilationError {
    UndeclaredVariable(ParserMetadata, String),
    UndefinedVariable(ParserMetadata, String),
    UnrepresentedVariable(ParserMetadata, String),
    CannotAssignHere(ParserMetadata, String),
    CannotRepresentHere(ParserMetadata, String),
    CannotEquateReferences(ParserMetadata, Type),
    ReferenceDefinitionMustBeMaterialized(ParserMetadata, String),
    IncorrectNumberOfComponents(ParserMetadata, String, usize, usize),
    IncorrectTypeInComponent(ParserMetadata, String, usize, Type, Type),
    UnexpectedType(ParserMetadata, Type, Type),
    UnexpectedUnmaterialized(ParserMetadata, Type),
    IncorrectTypeInArithmetic(ParserMetadata, Type),
    ExpectedBoolean(ParserMetadata, Type),
    MismatchedTypes(ParserMetadata, Type, Type),
    DuplicateDeclaration(ParserMetadata, String),
    DuplicateDefinition(ParserMetadata, String),
    DuplicateRepresentation(ParserMetadata, String),
    DuplicateUnderConstructionArrayUsage(ParserMetadata, String),
    UndefinedType(ParserMetadata, String),
    UndefinedConstructor(ParserMetadata, String),

    NotAReference(ParserMetadata, Type),
    NotAnUnderConstructionArray(ParserMetadata, Type),
    NotAnArray(ParserMetadata, Type),
    NotAnIndex(ParserMetadata, Type),
    NotANamedType(ParserMetadata, Type),
    NotAConstArray(ParserMetadata, Type),

    UndefinedFunction(ParserMetadata, String),
    IncorrectNumberOfArgumentsToFunction(ParserMetadata, String, usize, usize),
    IncorrectTypeForArgument(ParserMetadata, String, usize, Type, Type),
    IncorrectTypeForOutArgument(ParserMetadata, usize, Type, Type),

    CannotOrderStatementsForDefinition(ParserMetadata),
    CannotOrderStatementsForRepresentation(ParserMetadata),
    TypesSelfReferential,
    InlineFunctionsSelfReferential,

    CannotInferEmptyConstArrayType(ParserMetadata),
    ElementTypesInconsistentInConstArray(ParserMetadata, Type, Type, usize, usize),
    ElementTypesInconsistentInConstArrayConcatenation(ParserMetadata, Type, Type),
    OutOfBoundsConstArrayAccess(ParserMetadata, usize, usize),
    OutOfBoundsConstArraySlice(ParserMetadata, usize, usize, usize),
    IncorrectNumberOfElementsInConstArray(ParserMetadata, usize, usize),
    DuplicateUnderConstructionArrayUsageInConstArray(ParserMetadata),

    EqMustBeDematerialized(ParserMetadata),
    DivMustBeDematerialized(ParserMetadata),

    DuplicateConstructorName(ParserMetadata, String),
    DuplicateTypeName(ParserMetadata, String),
    TypeNameCannotBeF(ParserMetadata),
    AlgebraicTypeCannotBeEmpty(ParserMetadata, String),
}
