use ax_stark_backend::air_builders::verifier::GenericVerifierConstraintFolder;
use axvm_native_compiler::ir::{Config, Ext, Felt, SymbolicExt};

type Var<C> = Ext<<C as Config>::F, <C as Config>::EF>;
type Expr<C> = SymbolicExt<<C as Config>::F, <C as Config>::EF>;

pub type RecursiveVerifierConstraintFolder<'a, C> = GenericVerifierConstraintFolder<
    'a,
    <C as Config>::F,
    <C as Config>::EF,
    Felt<<C as Config>::F>,
    Var<C>,
    Expr<C>,
>;
