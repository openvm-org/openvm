use serde::{Deserialize, Serialize};

/// Operations that can be constrained inside the circuit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConstraintOpcode {
    ImmV,
    ImmF,
    ImmE,
    AddV,
    AddF,
    AddE,
    AddEF,
    SubV,
    SubF,
    SubE,
    SubEF,
    MulV,
    MulF,
    MulE,
    MulEF,
    DivF,
    DivE,
    DivEF,
    NegV,
    NegF,
    NegE,
    AssertEqV,
    AssertEqF,
    AssertEqE,
    Permute,
    Num2BitsV,
    Num2BitsF,
    SelectV,
    SelectF,
    SelectE,
    Ext2Felt,
    PrintV,
    PrintF,
    PrintE,
    WitnessV,
    WitnessF,
    WitnessE,
    CommitVkeyHash,
    CommitCommittedValuesDigest,
    CircuitFelts2Ext,
}
