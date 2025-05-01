# OpenVM Instruction Mapping

In this document, we provide a mapping between the representation of instructions
in the OpenVM codebase and the instructions in the [ISA specification](../specs/ISA.md).

- Instructions in OpenVM implement the `LocalOpcode` trait. Different groups of `LocalOpcode`s from different VM extensions may be combined to form a set of instructions for a customized VM using several extensions.
- The PHANTOM instruction may be extended in each VM extension by adding new sub-instructions with different `PhantomDiscriminant` values.

In the tables below, we provide the mapping between the `LocalOpcode` and `PhantomDiscriminant` and instructions and phantom sub-instructions in the ISA specification.

## System Instructions

#### Instructions

| VM Extension | `LocalOpcode` | ISA Instruction |
| ------------- | ---------- | ------------- |
| System | `SystemOpcode::TERMINATE` | TERMINATE |
| System | `SystemOpcode::PHANTOM` | PHANTOM |
| System | `PublishOpcode::PUBLISH` | PUBLISH |

#### Phantom Sub-Instructions

| VM Extension | `PhantomDiscriminant` | ISA Phantom Sub-Instruction |
| ------------- | ---------- | ------------- |
| System | `SysPhantom::Nop` | NOP |
| System | `SysPhantom::DebugPanic` | DEBUG_PANIC |
| System | `SysPhantom::CtStart` | CT_START |
| System | `SysPhantom::CtEnd` | CT_END |

## RV32IM Extension

#### Instructions

| VM Extension | `LocalOpcode` | ISA Instruction  |
| ------------- | ---------- |------------------|
| RV32IM | `BaseAluOpcode::ADD` | ADD_RV32         |
| RV32IM | `BaseAluOpcode::SUB` | SUB_RV32         |
| RV32IM | `BaseAluOpcode::XOR` | XOR_RV32         |
| RV32IM | `BaseAluOpcode::OR` | OR_RV32          |
| RV32IM | `BaseAluOpcode::AND` | AND_RV32         |
| RV32IM | `ShiftOpcode::SLL` | SLL_RV32         |
| RV32IM | `ShiftOpcode::SRL` | SRL_RV32         |
| RV32IM | `ShiftOpcode::SRA` | SRA_RV32         |
| RV32IM | `LessThanOpcode::SLT` | SLT_RV32         |
| RV32IM | `LessThanOpcode::SLTU` | SLTU_RV32        |
| RV32IM | `Rv32LoadStoreOpcode::LOADB` | LOADB_RV32       |
| RV32IM | `Rv32LoadStoreOpcode::LOADH` | LOADH_RV32       |
| RV32IM | `Rv32LoadStoreOpcode::LOADW` | LOADW_RV32       |
| RV32IM | `Rv32LoadStoreOpcode::LOADBU` | LOADBU_RV32      |
| RV32IM | `Rv32LoadStoreOpcode::LOADHU` | LOADHU_RV32      |
| RV32IM | `Rv32LoadStoreOpcode::STOREB` | STOREB_RV32      |
| RV32IM | `Rv32LoadStoreOpcode::STOREH` | STOREH_RV32      |
| RV32IM | `Rv32LoadStoreOpcode::STOREW` | STOREW_RV32      |
| RV32IM | `BranchEqualOpcode::BEQ` | BEQ_RV32         |
| RV32IM | `BranchEqualOpcode::BNE` | BNE_RV32         |
| RV32IM | `BranchLessThanOpcode::BLT` | BLT_RV32         |
| RV32IM | `BranchLessThanOpcode::BGE` | BGE_RV32         |
| RV32IM | `BranchLessThanOpcode::BLTU` | BLTU_RV32        |
| RV32IM | `BranchLessThanOpcode::BGEU` | BGEU_RV32        |
| RV32IM | `Rv32JalLuiOpcode::JAL` | JAL_RV32         |
| RV32IM | `Rv32JalrOpcode::JALR` | JALR_RV32        |
| RV32IM | `Rv32JalLuiOpcode::LUI` | LUI_RV32         |
| RV32IM | `Rv32AuipcOpcode::AUIPC` | AUIPC_RV32       |
| RV32IM | `MulOpcode::MUL` | MUL_RV32         |
| RV32IM | `MulHOpcode::MULH` | MULH_RV32        |
| RV32IM | `MulHOpcode::MULHSU` | MULHSU_RV32      |
| RV32IM | `MulHOpcode::MULHU` | MULHU_RV32       |
| RV32IM | `DivRemOpcode::DIV` | DIV_RV32         |
| RV32IM | `DivRemOpcode::DIVU` | DIVU_RV32        |
| RV32IM | `DivRemOpcode::REM` | REM_RV32         |
| RV32IM | `DivRemOpcode::REMU` | REMU_RV32        |
| RV32IM | `Rv32HintStoreOpcode::HINT_STOREW` | HINT_STOREW_RV32 |
| RV32IM | `Rv32HintStoreOpcode::HINT_BUFFER` | HINT_BUFFER_RV32 |
| RV32IM | Pseudo-instruction for `STOREW_RV32` | REVEAL_RV32      |
| RV32IM | Pseudo-instruction for `STOREW_RV32` | NATIVE_STOREW    |   |

#### Phantom Sub-Instructions

| VM Extension | `PhantomDiscriminant`         | ISA Phantom Sub-Instruction |
| ------------- |-------------------------------| ------------- |
| RV32IM | `Rv32Phantom::HintInput`      | Rv32HintInput |
| RV32IM | `Rv32Phantom::PrintStr`       | Rv32PrintStr |
| RV32IM | `Rv32Phantom::HintRandom`     | Rv32HintRandom |
| RV32IM | `Rv32Phantom::HintLoadByKey` | Rv32HintLoadByKey |

## Native Extension

#### Instructions

| VM Extension | `LocalOpcode` | ISA Instruction |
| ------------- | ---------- | ------------- |
| Native | `NativeLoadStoreOpcode::LOADW` | LOADW |
| Native | `NativeLoadStoreOpcode::STOREW` | STOREW |
| Native | `NativeLoadStore4Opcode::LOADW4` | LOADW4 |
| Native | `NativeLoadStore4Opcode::STOREW4` | STOREW4 |
| Native | `NativeJalOpcode::JAL` | JAL |
| Native | `NativeRangeCheckOpcode::RANGE_CHECK` | RANGE_CHECK |
| Native | `NativeBranchEqualOpcode::BEQ` | BEQ |
| Native | `NativeBranchEqualOpcode::BNE` | BNE |
| Native | `NativeLoadStoreOpcode::HINT_STOREW` | HINT_STOREW |
| Native | `NativeLoadStore4Opcode::HINT_STOREW4` | HINT_STOREW4 |
| Native | `CastfOpcode::CASTF` | CASTF |
| Native | `FieldArithmeticOpcode::ADD` | ADDF |
| Native | `FieldArithmeticOpcode::SUB` | SUBF |
| Native | `FieldArithmeticOpcode::MUL` | MULF |
| Native | `FieldArithmeticOpcode::DIV` | DIVF |
| Native | `FieldExtensionOpcode::FE4ADD` | FE4ADD |
| Native | `FieldExtensionOpcode::FE4SUB` | FE4SUB |
| Native | `FieldExtensionOpcode::BBE4MUL` | BBE4MUL |
| Native | `FieldExtensionOpcode::BBE4DIV` | BBE4DIV |
| Native | `Poseidon2Opcode::COMP_POS2` | COMP_POS2 |
| Native | `Poseidon2Opcode::PERM_POS2` | PERM_POS2 |
| Native | `VerifyBatchOpcode::VERIFY_BATCH` | VERIFY_BATCH |
| Native | `FriOpcode::FRI_REDUCED_OPENING` | FRI_REDUCED_OPENING |

#### Phantom Sub-Instructions

| VM Extension | `PhantomDiscriminant` | ISA Phantom Sub-Instruction |
| ------------- | ---------- | ------------- |
| Native | `NativePhantom::Print` | NativePrint |
| Native | `NativePhantom::HintInput` | NativeHintInput |
| Native | `NativePhantom::HintBits` | NativeHintBits |
| Native | `NativePhantom::HintLoad` | NativeHintLoad |

## Keccak Extension

#### Instructions

| VM Extension | `LocalOpcode` | ISA Instruction |
| ------------- | ---------- | ------------- |
| Keccak | `Rv32KeccakOpcode::KECCAK256` | KECCAK256_RV32 |

## SHA2-256 Extension

#### Instructions

| VM Extension | `LocalOpcode` | ISA Instruction |
| ------------- | ---------- | ------------- |
| SHA2-256 | `Rv32Sha256Opcode::SHA256` | SHA256_RV32 |

## BigInt Extension

#### Instructions

| VM Extension | `LocalOpcode` | ISA Instruction |
| ------------- | ---------- | ------------- |
| BigInt | `Rv32BaseAlu256Opcode::ADD256` | ADD256_RV32 |
| BigInt | `Rv32BaseAlu256Opcode::SUB256` | SUB256_RV32 |
| BigInt | `Rv32BaseAlu256Opcode::XOR256` | XOR256_RV32 |
| BigInt | `Rv32BaseAlu256Opcode::OR256` | OR256_RV32 |
| BigInt | `Rv32BaseAlu256Opcode::AND256` | AND256_RV32 |
| BigInt | `Rv32Shift256Opcode::SLL256` | SLL256_RV32 |
| BigInt | `Rv32Shift256Opcode::SRL256` | SRL256_RV32 |
| BigInt | `Rv32Shift256Opcode::SRA256` | SRA256_RV32 |
| BigInt | `Rv32LessThan256Opcode::SLT256` | SLT256_RV32 |
| BigInt | `Rv32LessThan256Opcode::SLTU256` | SLTU256_RV32 |
| BigInt | `Rv32BranchEqual256Opcode::BEQ256` | BEQ256_RV32 |
| BigInt | `Rv32BranchEqual256Opcode::BNE256` | BNE256_RV32 |
| BigInt | `Rv32BranchLessThan256Opcode::BLT256` | BLT256_RV32 |
| BigInt | `Rv32BranchLessThan256Opcode::BGE256` | BGE256_RV32 |
| BigInt | `Rv32BranchLessThan256Opcode::BLTU256` | BLTU256_RV32 |
| BigInt | `Rv32BranchLessThan256Opcode::BGEU256` | BGEU256_RV32 |
| BigInt | `Rv32Mul256Opcode::MUL256` | MUL256_RV32 |

## Algebra Extension

#### Instructions

| VM Extension | `LocalOpcode` | ISA Instruction |
| ------------- | ---------- | ------------- |
| Algebra | `Rv32ModularArithmeticOpcode::ADD` | ADDMOD_RV32\<N\> |
| Algebra | `Rv32ModularArithmeticOpcode::SUB` | SUBMOD_RV32\<N\> |
| Algebra | `Rv32ModularArithmeticOpcode::SETUP_ADDSUB` | SETUP_ADDSUBMOD_RV32\<N\> |
| Algebra | `Rv32ModularArithmeticOpcode::MUL` | MULMOD_RV32\<N\> |
| Algebra | `Rv32ModularArithmeticOpcode::DIV` | DIVMOD_RV32\<N\> |
| Algebra | `Rv32ModularArithmeticOpcode::SETUP_MULDIV` | SETUP_MULDIVMOD_RV32\<N\> |
| Algebra | `Rv32ModularArithmeticOpcode::IS_EQ` | ISEQMOD_RV32\<N\> |
| Algebra | `Rv32ModularArithmeticOpcode::SETUP_ISEQ` | SETUP_ISEQMOD_RV32\<N\> |
| Algebra | `Fp2Opcode::ADD` | ADD\<Fp2\> |
| Algebra | `Fp2Opcode::SUB` | SUB\<Fp2\> |
| Algebra | `Fp2Opcode::SETUP_ADDSUB` | SETUP_ADDSUB\<Fp2\> |
| Algebra | `Fp2Opcode::MUL` | MUL\<Fp2\> |
| Algebra | `Fp2Opcode::DIV` | DIV\<Fp2\> |
| Algebra | `Fp2Opcode::SETUP_MULDIV` | SETUP_MULDIV\<Fp2\> |

## Elliptic Curve Extension

#### Instructions

| VM Extension | `LocalOpcode` | ISA Instruction |
| ------------- | ---------- | ------------- |
| Elliptic Curve | `Rv32WeierstrassOpcode::EC_ADD_NE` | EC_ADD_NE\<C\> |
| Elliptic Curve | `Rv32WeierstrassOpcode::SETUP_EC_ADD_NE` | SETUP_EC_ADD_NE\<C\> |
| Elliptic Curve | `Rv32WeierstrassOpcode::EC_DOUBLE` | EC_DOUBLE\<C\> |
| Elliptic Curve | `Rv32WeierstrassOpcode::SETUP_EC_DOUBLE` | SETUP_EC_DOUBLE\<C\> |

#### Phantom Sub-Instructions

| VM Extension | `PhantomDiscriminant` | ISA Phantom Sub-Instruction |
| ------------- | ---------- | ------------- |
| Elliptic Curve | `EccPhantom::HintDecompress` | HintDecompress |

## Pairing Extension

#### Instructions

#### Phantom Sub-Instructions

| VM Extension | `PhantomDiscriminant` | ISA Phantom Sub-Instruction |
| ------------- | ---------- | ------------- |
| Pairing | `PairingPhantom::HintDecompress` | HintDecompress |
