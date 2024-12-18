use openvm_native_compiler::ir::{Config, Ext, Felt, SymbolicExt};
use openvm_stark_backend::air_builders::verifier::GenericVerifierConstraintFolder;

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
