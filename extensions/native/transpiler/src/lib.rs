pub mod deserialize_instruction;

const LONG_FORM_INSTRUCTION_INDICATOR: u32 = (1 << 31) + 115115115;
const GAP_INDICATOR: u32 = (1 << 31) + 113113113;
const VARIABLE_REGISTER_INDICATOR: u32 = (1 << 31) + 116;
const REGISTER_LIMBS: u32 = 4;
