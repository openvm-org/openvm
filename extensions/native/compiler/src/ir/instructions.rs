use serde::{Deserialize, Serialize};

use super::{Array, Config, Ext, Felt, MemIndex, Ptr, RVar, TracedVec, Usize, Var};

/// An intermediate instruction set for implementing programs.
///
/// Programs written in the DSL can compile both to the recursive zkVM and the R1CS or Plonk-ish
/// circuits.
#[derive(Debug, Clone, strum_macros::Display, Serialize, Deserialize)]
pub enum DslIr<C: Config> {
    // Immediates.
    /// Assigns an immediate to a variable (var = imm).
    ImmV(Var<C::N>, C::N),
    /// Assigns a field immediate to a field element (felt = field imm).
    ImmF(Felt<C::F>, C::F),
    /// Assigns an ext field immediate to an extension field element (ext = ext field imm).
    ImmE(Ext<C::F, C::EF>, C::EF),

    // Additions.
    /// Add two variables (var = var + var).
    AddV(Var<C::N>, Var<C::N>, Var<C::N>),
    /// Add a variable and an immediate (var = var + imm).
    AddVI(Var<C::N>, Var<C::N>, C::N),
    /// Add two field elements (felt = felt + felt).
    AddF(Felt<C::F>, Felt<C::F>, Felt<C::F>),
    /// Add a field element and a field immediate (felt = felt + field imm).
    AddFI(Felt<C::F>, Felt<C::F>, C::F),
    /// Add two extension field elements (ext = ext + ext).
    AddE(Ext<C::F, C::EF>, Ext<C::F, C::EF>, Ext<C::F, C::EF>),
    /// Add an extension field element and an ext field immediate (ext = ext + ext field imm).
    AddEI(Ext<C::F, C::EF>, Ext<C::F, C::EF>, C::EF),
    /// Add an extension field element and a field element (ext = ext + felt).
    AddEF(Ext<C::F, C::EF>, Ext<C::F, C::EF>, Felt<C::F>),
    /// Add an extension field element and a field immediate (ext = ext + field imm).
    AddEFI(Ext<C::F, C::EF>, Ext<C::F, C::EF>, C::F),
    /// Add a field element and an ext field immediate (ext = felt + ext field imm).
    AddEFFI(Ext<C::F, C::EF>, Felt<C::F>, C::EF),

    // Subtractions.
    /// Subtracts two variables (var = var - var).
    SubV(Var<C::N>, Var<C::N>, Var<C::N>),
    /// Subtracts a variable and an immediate (var = var - imm).
    SubVI(Var<C::N>, Var<C::N>, C::N),
    /// Subtracts an immediate and a variable (var = imm - var).
    SubVIN(Var<C::N>, C::N, Var<C::N>),
    /// Subtracts two field elements (felt = felt - felt).
    SubF(Felt<C::F>, Felt<C::F>, Felt<C::F>),
    /// Subtracts a field element and a field immediate (felt = felt - field imm).
    SubFI(Felt<C::F>, Felt<C::F>, C::F),
    /// Subtracts a field immediate and a field element (felt = field imm - felt).
    SubFIN(Felt<C::F>, C::F, Felt<C::F>),
    /// Subtracts two extension field elements (ext = ext - ext).
    SubE(Ext<C::F, C::EF>, Ext<C::F, C::EF>, Ext<C::F, C::EF>),
    /// Subtracts an extension field element and an extension field immediate (ext = ext - ext
    /// field imm).
    SubEI(Ext<C::F, C::EF>, Ext<C::F, C::EF>, C::EF),
    /// Subtracts an extension field immediate and an extension field element (ext = ext field imm
    /// - ext).
    SubEIN(Ext<C::F, C::EF>, C::EF, Ext<C::F, C::EF>),
    /// Subtracts an extension field element and a field immediate (ext = ext - field imm).
    SubEFI(Ext<C::F, C::EF>, Ext<C::F, C::EF>, C::F),
    /// Subtracts an extension field element and a field element (ext = ext - felt).
    SubEF(Ext<C::F, C::EF>, Ext<C::F, C::EF>, Felt<C::F>),

    // Multiplications.
    /// Multiplies two variables (var = var * var).
    MulV(Var<C::N>, Var<C::N>, Var<C::N>),
    /// Multiplies a variable and an immediate (var = var * imm).
    MulVI(Var<C::N>, Var<C::N>, C::N),
    /// Multiplies two field elements (felt = felt * felt).
    MulF(Felt<C::F>, Felt<C::F>, Felt<C::F>),
    /// Multiplies a field element and a field immediate (felt = felt * field imm).
    MulFI(Felt<C::F>, Felt<C::F>, C::F),
    /// Multiplies two extension field elements (ext = ext * ext).
    MulE(Ext<C::F, C::EF>, Ext<C::F, C::EF>, Ext<C::F, C::EF>),
    /// Multiplies an extension field element and an extension field immediate (ext = ext * ext
    /// field imm).
    MulEI(Ext<C::F, C::EF>, Ext<C::F, C::EF>, C::EF),
    /// Multiplies an extension field element and a field immediate (ext = ext * field imm).
    MulEFI(Ext<C::F, C::EF>, Ext<C::F, C::EF>, C::F),
    /// Multiplies an extension field element and a field element (ext = ext * felt).
    MulEF(Ext<C::F, C::EF>, Ext<C::F, C::EF>, Felt<C::F>),

    // Divisions.
    /// Divides two variables (var = var / var).
    DivF(Felt<C::F>, Felt<C::F>, Felt<C::F>),
    /// Divides a field element and a field immediate (felt = felt / field imm).
    DivFI(Felt<C::F>, Felt<C::F>, C::F),
    /// Divides a field immediate and a field element (felt = field imm / felt).
    DivFIN(Felt<C::F>, C::F, Felt<C::F>),
    /// Divides two extension field elements (ext = ext / ext).
    DivE(Ext<C::F, C::EF>, Ext<C::F, C::EF>, Ext<C::F, C::EF>),
    /// Divides an extension field element and an extension field immediate (ext = ext / ext field
    /// imm).
    DivEI(Ext<C::F, C::EF>, Ext<C::F, C::EF>, C::EF),
    /// Divides and extension field immediate and an extension field element (ext = ext field imm /
    /// ext).
    DivEIN(Ext<C::F, C::EF>, C::EF, Ext<C::F, C::EF>),
    /// Divides an extension field element and a field immediate (ext = ext / field imm).
    DivEFI(Ext<C::F, C::EF>, Ext<C::F, C::EF>, C::F),
    /// Divides an extension field element and a field element (ext = ext / felt).
    DivEF(Ext<C::F, C::EF>, Ext<C::F, C::EF>, Felt<C::F>),

    // Negations.
    /// Negates a variable (var = -var).
    NegV(Var<C::N>, Var<C::N>),
    /// Negates a field element (felt = -felt).
    NegF(Felt<C::F>, Felt<C::F>),
    /// Negates an extension field element (ext = -ext).
    NegE(Ext<C::F, C::EF>, Ext<C::F, C::EF>),

    /// Cast a Felt to a Var.
    CastFV(Var<C::N>, Felt<C::F>),
    /// Cast a Var to a Felt. This is unsafe because of possible overflow. Dynamic mode only.
    UnsafeCastVF(Felt<C::F>, Var<C::N>),

    // =======

    // Control flow.
    /// Executes a zipped iterator for loop over pointers with the parameters
    /// (start step values, end step value of first pointer, step sizes, step variables, body).
    ZipFor(
        Vec<RVar<C::N>>,
        RVar<C::N>,
        Vec<C::N>,
        Vec<Var<C::N>>,
        TracedVec<DslIr<C>>,
    ),

    /// Executes an equal conditional branch with the parameters (lhs var, rhs var, then body, else
    /// body).
    IfEq(
        Var<C::N>,
        Var<C::N>,
        TracedVec<DslIr<C>>,
        TracedVec<DslIr<C>>,
    ),
    /// Executes a not equal conditional branch with the parameters (lhs var, rhs var, then body,
    /// else body).
    IfNe(
        Var<C::N>,
        Var<C::N>,
        TracedVec<DslIr<C>>,
        TracedVec<DslIr<C>>,
    ),
    /// Executes an equal conditional branch with the parameters (lhs var, rhs imm, then body, else
    /// body).
    IfEqI(Var<C::N>, C::N, TracedVec<DslIr<C>>, TracedVec<DslIr<C>>),
    /// Executes a not equal conditional branch with the parameters (lhs var, rhs imm, then body,
    /// else body).
    IfNeI(Var<C::N>, C::N, TracedVec<DslIr<C>>, TracedVec<DslIr<C>>),

    // Assertions.
    /// Assert that two variables are equal (var == var).
    AssertEqV(Var<C::N>, Var<C::N>),
    /// Assert that two field elements are equal (felt == felt).
    AssertEqF(Felt<C::F>, Felt<C::F>),
    /// Assert that two extension field elements are equal (ext == ext).
    AssertEqE(Ext<C::F, C::EF>, Ext<C::F, C::EF>),
    /// Assert that a variable is equal to an immediate (var == imm).
    AssertEqVI(Var<C::N>, C::N),
    /// Assert that a field element is equal to a field immediate (felt == field imm).
    AssertEqFI(Felt<C::F>, C::F),
    /// Assert that an extension field element is equal to an extension field immediate (ext == ext
    /// field imm).
    AssertEqEI(Ext<C::F, C::EF>, C::EF),

    /// Assert that a usize is not zero (usize != 0).
    AssertNonZero(Usize<C::N>),

    // Memory instructions.
    /// Allocate (ptr, len, size) a memory slice of length len
    Alloc(Ptr<C::N>, RVar<C::N>, usize),
    /// Load variable (var, ptr, index)
    LoadV(Var<C::N>, Ptr<C::N>, MemIndex<C::N>),
    /// Load field element (var, ptr, index)
    LoadF(Felt<C::F>, Ptr<C::N>, MemIndex<C::N>),
    /// Load extension field
    LoadE(Ext<C::F, C::EF>, Ptr<C::N>, MemIndex<C::N>),
    /// Load heap pointer into a stack variable. ASM only.
    LoadHeapPtr(Ptr<C::N>),
    /// Store variable at address
    StoreV(Var<C::N>, Ptr<C::N>, MemIndex<C::N>),
    /// Store field element at address
    StoreF(Felt<C::F>, Ptr<C::N>, MemIndex<C::N>),
    /// Store extension field at address
    StoreE(Ext<C::F, C::EF>, Ptr<C::N>, MemIndex<C::N>),
    /// Store heap pointer. ASM only.
    StoreHeapPtr(Ptr<C::N>),

    // Bits.
    /// Decompose a field element into bits (bits = num2bits(felt)). Should only be used when
    /// target is a circuit.
    CircuitNum2BitsF(Felt<C::F>, Vec<Var<C::N>>),
    /// Decompose a Var into 16-bit limbs.
    CircuitVarTo64BitsF(Var<C::N>, [Felt<C::F>; 4]),

    // Hashing.
    /// Permutes an array of baby bear elements using Poseidon2 (output = p2_permute(array)).
    Poseidon2PermuteBabyBear(Array<C, Felt<C::F>>, Array<C, Felt<C::F>>),
    /// Compresses two baby bear element arrays using Poseidon2 (output = p2_compress(array1,
    /// array2)).
    Poseidon2CompressBabyBear(
        Array<C, Felt<C::F>>,
        Array<C, Felt<C::F>>,
        Array<C, Felt<C::F>>,
    ),
    /// Permutes an array of Bn254 elements using Poseidon2 (output = p2_permute(array)). Should
    /// only be used when target is a circuit.
    CircuitPoseidon2Permute([Var<C::N>; 3]),

    // Miscellaneous instructions.
    /// Prints a variable.
    PrintV(Var<C::N>),
    /// Prints a field element.
    PrintF(Felt<C::F>),
    /// Prints an extension field element.
    PrintE(Ext<C::F, C::EF>),
    /// Throws an error.
    Error(),

    /// Prepare next input vector (preceded by its length) for hinting.
    HintInputVec(),
    /// Prepare next felt for hinting
    HintFelt(),
    /// Prepare bit decomposition for hinting.
    HintBitsF(Felt<C::F>, u32),

    StoreHintWord(Ptr<C::N>, MemIndex<C::N>),
    /// Move data from input stream into hint space
    HintLoad(),

    /// Witness a variable. Should only be used when target is a circuit.
    WitnessVar(Var<C::N>, u32),
    /// Witness a field element. Should only be used when target is a circuit.
    WitnessFelt(Felt<C::F>, u32),
    /// Witness an extension field element. Should only be used when target is a circuit.
    WitnessExt(Ext<C::F, C::EF>, u32),
    /// Label a field element as the ith public input.
    Publish(Felt<C::F>, Var<C::N>),
    /// Operation to halt the program. Should be the last instruction in the program.
    Halt,

    // Public inputs for circuits.
    /// Publish a field element as the ith public value. Should only be used when target is a
    /// circuit.
    CircuitPublish(Var<C::N>, usize),

    // FRI specific instructions.
    /// Select's a variable based on a condition. (select(cond, true_val, false_val) => output).
    /// Should only be used when target is a circuit.
    CircuitSelectV(Var<C::N>, Var<C::N>, Var<C::N>, Var<C::N>),
    /// Select's a field element based on a condition. (select(cond, true_val, false_val) =>
    /// output). Should only be used when target is a circuit.
    CircuitSelectF(Var<C::N>, Felt<C::F>, Felt<C::F>, Felt<C::F>),
    /// Select's an extension field element based on a condition. (select(cond, true_val,
    /// false_val) => output). Should only be used when target is a circuit.
    CircuitSelectE(
        Var<C::N>,
        Ext<C::F, C::EF>,
        Ext<C::F, C::EF>,
        Ext<C::F, C::EF>,
    ),
    /// Converts an ext to a slice of felts. Should only be used when target is a circuit.
    CircuitExt2Felt([Felt<C::F>; 4], Ext<C::F, C::EF>),
    /// Converts a slice of felts to an ext. Should only be used when target is a circuit.
    CircuitFelts2Ext([Felt<C::F>; 4], Ext<C::F, C::EF>),
    /// Halo2 only. Reduce a Felt so later computation becomes cheaper.
    CircuitFeltReduce(Felt<C::F>),
    /// Halo2 only. Reduce an Ext so later computation becomes cheaper.
    CircuitExtReduce(Ext<C::F, C::EF>),
    /// Halo2 only. Asserts that `a, b` both have <= `C::F::bits()` and then asserts `a < b`.
    /// Assumes that `C::F::bits() < C::N::bits()`.
    CircuitLessThan(Var<C::N>, Var<C::N>),
    /// FriReducedOpening(alpha, hint_id, is_init, at_x_array, at_z_array, result)
    FriReducedOpening(
        Ext<C::F, C::EF>,
        Var<C::N>,
        Var<C::N>,
        Array<C, Felt<C::F>>,
        Array<C, Ext<C::F, C::EF>>,
        Ext<C::F, C::EF>,
    ),
    /// VerifyBatch(dim, opened, proof_id, index, commit)
    /// opened values are Felts
    VerifyBatchFelt(
        Array<C, Usize<C::F>>,
        Array<C, Array<C, Felt<C::F>>>,
        Var<C::N>,
        Array<C, Var<C::N>>,
        Array<C, Felt<C::F>>,
    ),
    /// VerifyBatch(dim, opened, proof_id, index, commit)
    /// opened values are Exts
    VerifyBatchExt(
        Array<C, Usize<C::F>>,
        Array<C, Array<C, Ext<C::F, C::EF>>>,
        Var<C::N>,
        Array<C, Var<C::N>>,
        Array<C, Felt<C::F>>,
    ),
    /// RangeCheckV(v, bit)
    /// Assert that v < 2^bit.
    RangeCheckV(Var<C::N>, usize),

    /// Start the cycle tracker used by a block of code annotated by the string input. Calling this
    /// with the same string will end the open cycle tracker instance and start a new one with
    /// an increasing numeric postfix.
    CycleTrackerStart(String),
    /// End the cycle tracker used by a block of code annotated by the string input.
    CycleTrackerEnd(String),
}

impl<C: Config> Default for DslIr<C> {
    fn default() -> Self {
        Self::Halt
    }
}
