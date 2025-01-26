# OpenVM Instruction Mapping

Instructions in OpenVM implement the `LocalOpcode` trait. Different groups of 
`LocalOpcode`s from different VM extensions may be combined to form a set of
instructions for a customized VM using several extensions. In this document, we
provide a mapping between the `LocalOpcode` representation of instructions and the
instructions in the [ISA specification](../specs/ISA.md).

## Instruction Mapping

| VM Extension | `LocalOpcode` | ISA Instruction |
| ------------- | ---------- | ------------- |
| System | `SystemOpcode::TERMINATE` | TERMINATE |
| System | `SystemOpcode::PHANTOM` | PHANTOM |
| System | `PublishOpcode::PUBLISH` | PUBLISH |
| RV32IM | `Rv32Opcode::ADD` | ADD_RV32 |
| RV32IM | `Rv32Opcode::SUB` | SUB_RV32 |
| RV32IM | `Rv32Opcode::XOR` | XOR_RV32 |
| RV32IM | `Rv32Opcode::OR` | OR_RV32 |
| RV32IM | `Rv32Opcode::AND` | AND_RV32 |
| RV32IM | `Rv32Opcode::SLL` | SLL_RV32 |
| RV32IM | `Rv32Opcode::SRL` | SRL_RV32 |
| RV32IM | `Rv32Opcode::SRA` | SRA_RV32 |
| RV32IM | `Rv32Opcode::SLT` | SLT_RV32 |
| RV32IM | `Rv32Opcode::SLTU` | SLTU_RV32 |
| RV32IM | `Rv32Opcode::LOADB` | LOADB_RV32 |
| RV32IM | `Rv32Opcode::LOADH` | LOADH_RV32 |
| RV32IM | `Rv32Opcode::LOADW` | LOADW_RV32 |
| RV32IM | `Rv32Opcode::LOADBU` | LOADBU_RV32 |
| RV32IM | `Rv32Opcode::LOADHU` | LOADHU_RV32 |
| RV32IM | `Rv32Opcode::STOREB` | STOREB_RV32 |
| RV32IM | `Rv32Opcode::STOREH` | STOREH_RV32 |
| RV32IM | `Rv32Opcode::STOREW` | STOREW_RV32 |
| RV32IM | `Rv32Opcode::BEQ` | BEQ_RV32 |
| RV32IM | `Rv32Opcode::BNE` | BNE_RV32 |
| RV32IM | `Rv32Opcode::BLT` | BLT_RV32 |
| RV32IM | `Rv32Opcode::BGE` | BGE_RV32 |
| RV32IM | `Rv32Opcode::BLTU` | BLTU_RV32 |
| RV32IM | `Rv32Opcode::BGEU` | BGEU_RV32 |
| RV32IM | `Rv32Opcode::JAL` | JAL_RV32 |
| RV32IM | `Rv32Opcode::JALR` | JALR_RV32 |
| RV32IM | `Rv32Opcode::LUI` | LUI_RV32 |
| RV32IM | `Rv32Opcode::AUIPC` | AUIPC_RV32 |
| RV32IM | `Rv32Opcode::MUL` | MUL_RV32 |
| RV32IM | `Rv32Opcode::MULH` | MULH_RV32 |
| RV32IM | `Rv32Opcode::MULHSU` | MULHSU_RV32 |
| RV32IM | `Rv32Opcode::MULHU` | MULHU_RV32 |
| RV32IM | `Rv32Opcode::DIV` | DIV_RV32 |
| RV32IM | `Rv32Opcode::DIVU` | DIVU_RV32 |
| RV32IM | `Rv32Opcode::REM` | REM_RV32 |
| RV32IM | `Rv32Opcode::REMU` | REMU_RV32 |
| RV32IM | `Rv32Opcode::HINT_STOREW` | HINT_STOREW_RV32 |
| RV32IM | `Rv32Opcode::HINT_BUFFER` | HINT_BUFFER_RV32 |
| RV32IM | `Rv32Opcode::REVEAL` | REVEAL_RV32 |


## Phantom Instruction Mapping
