// Lean compiler output
// Module: Mathlib.Data.Nat.Bits
// Imports: public import Init public meta import Init public import Mathlib.Data.Nat.BinaryRec public import Mathlib.Data.List.Defs
#include <lean/lean.h>
#if defined(__clang__)
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wunused-label"
#elif defined(__GNUC__) && !defined(__CLANG__)
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-label"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#endif
#ifdef __cplusplus
extern "C" {
#endif
lean_object* l_Nat_bitwise(lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_nat_shiftr(lean_object*, lean_object*);
lean_object* lean_nat_land(lean_object*, lean_object*);
lean_object* l_Lean_Name_str___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_num___override(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
uint8_t l_Nat_testBit(lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lp_mathlib_Nat_bit(uint8_t, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Data_Nat_Bits_0__termBxor___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "_private"};
static const lean_object* lp_mathlib___private_Mathlib_Data_Nat_Bits_0__termBxor___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Nat_Bits_0__termBxor___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_Nat_Bits_0__termBxor___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Data_Nat_Bits_0__termBxor___closed__0_value),LEAN_SCALAR_PTR_LITERAL(103, 214, 75, 80, 34, 198, 193, 153)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_Nat_Bits_0__termBxor___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Nat_Bits_0__termBxor___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Data_Nat_Bits_0__termBxor___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib___private_Mathlib_Data_Nat_Bits_0__termBxor___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Nat_Bits_0__termBxor___closed__2_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_Nat_Bits_0__termBxor___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_Nat_Bits_0__termBxor___closed__1_value),((lean_object*)&lp_mathlib___private_Mathlib_Data_Nat_Bits_0__termBxor___closed__2_value),LEAN_SCALAR_PTR_LITERAL(234, 232, 174, 134, 127, 136, 69, 92)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_Nat_Bits_0__termBxor___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Nat_Bits_0__termBxor___closed__3_value;
static const lean_string_object lp_mathlib___private_Mathlib_Data_Nat_Bits_0__termBxor___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Data"};
static const lean_object* lp_mathlib___private_Mathlib_Data_Nat_Bits_0__termBxor___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Nat_Bits_0__termBxor___closed__4_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_Nat_Bits_0__termBxor___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_Nat_Bits_0__termBxor___closed__3_value),((lean_object*)&lp_mathlib___private_Mathlib_Data_Nat_Bits_0__termBxor___closed__4_value),LEAN_SCALAR_PTR_LITERAL(149, 48, 214, 5, 44, 128, 44, 12)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_Nat_Bits_0__termBxor___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Nat_Bits_0__termBxor___closed__5_value;
static const lean_string_object lp_mathlib___private_Mathlib_Data_Nat_Bits_0__termBxor___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Nat"};
static const lean_object* lp_mathlib___private_Mathlib_Data_Nat_Bits_0__termBxor___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Nat_Bits_0__termBxor___closed__6_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_Nat_Bits_0__termBxor___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_Nat_Bits_0__termBxor___closed__5_value),((lean_object*)&lp_mathlib___private_Mathlib_Data_Nat_Bits_0__termBxor___closed__6_value),LEAN_SCALAR_PTR_LITERAL(213, 118, 156, 245, 192, 254, 123, 239)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_Nat_Bits_0__termBxor___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Nat_Bits_0__termBxor___closed__7_value;
static const lean_string_object lp_mathlib___private_Mathlib_Data_Nat_Bits_0__termBxor___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Bits"};
static const lean_object* lp_mathlib___private_Mathlib_Data_Nat_Bits_0__termBxor___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Nat_Bits_0__termBxor___closed__8_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_Nat_Bits_0__termBxor___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_Nat_Bits_0__termBxor___closed__7_value),((lean_object*)&lp_mathlib___private_Mathlib_Data_Nat_Bits_0__termBxor___closed__8_value),LEAN_SCALAR_PTR_LITERAL(8, 209, 1, 43, 5, 63, 76, 116)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_Nat_Bits_0__termBxor___closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Nat_Bits_0__termBxor___closed__9_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_Nat_Bits_0__termBxor___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_Nat_Bits_0__termBxor___closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(97, 216, 228, 51, 130, 199, 64, 48)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_Nat_Bits_0__termBxor___closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Nat_Bits_0__termBxor___closed__10_value;
static const lean_string_object lp_mathlib___private_Mathlib_Data_Nat_Bits_0__termBxor___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "termBxor"};
static const lean_object* lp_mathlib___private_Mathlib_Data_Nat_Bits_0__termBxor___closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Nat_Bits_0__termBxor___closed__11_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_Nat_Bits_0__termBxor___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_Nat_Bits_0__termBxor___closed__10_value),((lean_object*)&lp_mathlib___private_Mathlib_Data_Nat_Bits_0__termBxor___closed__11_value),LEAN_SCALAR_PTR_LITERAL(231, 237, 20, 211, 226, 155, 197, 177)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_Nat_Bits_0__termBxor___closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Nat_Bits_0__termBxor___closed__12_value;
static const lean_string_object lp_mathlib___private_Mathlib_Data_Nat_Bits_0__termBxor___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "bxor"};
static const lean_object* lp_mathlib___private_Mathlib_Data_Nat_Bits_0__termBxor___closed__13 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Nat_Bits_0__termBxor___closed__13_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_Nat_Bits_0__termBxor___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_Nat_Bits_0__termBxor___closed__13_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_Nat_Bits_0__termBxor___closed__14 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Nat_Bits_0__termBxor___closed__14_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_Nat_Bits_0__termBxor___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_Nat_Bits_0__termBxor___closed__12_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Data_Nat_Bits_0__termBxor___closed__14_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_Nat_Bits_0__termBxor___closed__15 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Nat_Bits_0__termBxor___closed__15_value;
LEAN_EXPORT const lean_object* lp_mathlib___private_Mathlib_Data_Nat_Bits_0__termBxor = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Nat_Bits_0__termBxor___closed__15_value;
static const lean_string_object lp_mathlib___private_Mathlib_Data_Nat_Bits_0____aux__Mathlib__Data__Nat__Bits______macroRules____private__Mathlib__Data__Nat__Bits__0__termBxor__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "xor"};
static const lean_object* lp_mathlib___private_Mathlib_Data_Nat_Bits_0____aux__Mathlib__Data__Nat__Bits______macroRules____private__Mathlib__Data__Nat__Bits__0__termBxor__1___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Nat_Bits_0____aux__Mathlib__Data__Nat__Bits______macroRules____private__Mathlib__Data__Nat__Bits__0__termBxor__1___closed__0_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Data_Nat_Bits_0____aux__Mathlib__Data__Nat__Bits______macroRules____private__Mathlib__Data__Nat__Bits__0__termBxor__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Data_Nat_Bits_0____aux__Mathlib__Data__Nat__Bits______macroRules____private__Mathlib__Data__Nat__Bits__0__termBxor__1___closed__1;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_Nat_Bits_0____aux__Mathlib__Data__Nat__Bits______macroRules____private__Mathlib__Data__Nat__Bits__0__termBxor__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Data_Nat_Bits_0____aux__Mathlib__Data__Nat__Bits______macroRules____private__Mathlib__Data__Nat__Bits__0__termBxor__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(202, 242, 219, 132, 101, 186, 164, 72)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_Nat_Bits_0____aux__Mathlib__Data__Nat__Bits______macroRules____private__Mathlib__Data__Nat__Bits__0__termBxor__1___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Nat_Bits_0____aux__Mathlib__Data__Nat__Bits______macroRules____private__Mathlib__Data__Nat__Bits__0__termBxor__1___closed__2_value;
static const lean_string_object lp_mathlib___private_Mathlib_Data_Nat_Bits_0____aux__Mathlib__Data__Nat__Bits______macroRules____private__Mathlib__Data__Nat__Bits__0__termBxor__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Bool"};
static const lean_object* lp_mathlib___private_Mathlib_Data_Nat_Bits_0____aux__Mathlib__Data__Nat__Bits______macroRules____private__Mathlib__Data__Nat__Bits__0__termBxor__1___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Nat_Bits_0____aux__Mathlib__Data__Nat__Bits______macroRules____private__Mathlib__Data__Nat__Bits__0__termBxor__1___closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_Nat_Bits_0____aux__Mathlib__Data__Nat__Bits______macroRules____private__Mathlib__Data__Nat__Bits__0__termBxor__1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Data_Nat_Bits_0____aux__Mathlib__Data__Nat__Bits______macroRules____private__Mathlib__Data__Nat__Bits__0__termBxor__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(250, 44, 198, 216, 184, 195, 199, 178)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_Nat_Bits_0____aux__Mathlib__Data__Nat__Bits______macroRules____private__Mathlib__Data__Nat__Bits__0__termBxor__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_Nat_Bits_0____aux__Mathlib__Data__Nat__Bits______macroRules____private__Mathlib__Data__Nat__Bits__0__termBxor__1___closed__4_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Data_Nat_Bits_0____aux__Mathlib__Data__Nat__Bits______macroRules____private__Mathlib__Data__Nat__Bits__0__termBxor__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(159, 35, 146, 118, 24, 65, 174, 144)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_Nat_Bits_0____aux__Mathlib__Data__Nat__Bits______macroRules____private__Mathlib__Data__Nat__Bits__0__termBxor__1___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Nat_Bits_0____aux__Mathlib__Data__Nat__Bits______macroRules____private__Mathlib__Data__Nat__Bits__0__termBxor__1___closed__4_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_Nat_Bits_0____aux__Mathlib__Data__Nat__Bits______macroRules____private__Mathlib__Data__Nat__Bits__0__termBxor__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_Nat_Bits_0____aux__Mathlib__Data__Nat__Bits______macroRules____private__Mathlib__Data__Nat__Bits__0__termBxor__1___closed__4_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Data_Nat_Bits_0____aux__Mathlib__Data__Nat__Bits______macroRules____private__Mathlib__Data__Nat__Bits__0__termBxor__1___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Nat_Bits_0____aux__Mathlib__Data__Nat__Bits______macroRules____private__Mathlib__Data__Nat__Bits__0__termBxor__1___closed__5_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_Nat_Bits_0____aux__Mathlib__Data__Nat__Bits______macroRules____private__Mathlib__Data__Nat__Bits__0__termBxor__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_Nat_Bits_0____aux__Mathlib__Data__Nat__Bits______macroRules____private__Mathlib__Data__Nat__Bits__0__termBxor__1___closed__5_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Data_Nat_Bits_0____aux__Mathlib__Data__Nat__Bits______macroRules____private__Mathlib__Data__Nat__Bits__0__termBxor__1___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Nat_Bits_0____aux__Mathlib__Data__Nat__Bits______macroRules____private__Mathlib__Data__Nat__Bits__0__termBxor__1___closed__6_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Bits_0____aux__Mathlib__Data__Nat__Bits______macroRules____private__Mathlib__Data__Nat__Bits__0__termBxor__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Bits_0____aux__Mathlib__Data__Nat__Bits______macroRules____private__Mathlib__Data__Nat__Bits__0__termBxor__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Data_Nat_Bits_0____aux__Mathlib__Data__Nat__Bits______unexpand__Bool__xor__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib___private_Mathlib_Data_Nat_Bits_0____aux__Mathlib__Data__Nat__Bits______unexpand__Bool__xor__1___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Nat_Bits_0____aux__Mathlib__Data__Nat__Bits______unexpand__Bool__xor__1___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_Nat_Bits_0____aux__Mathlib__Data__Nat__Bits______unexpand__Bool__xor__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Data_Nat_Bits_0____aux__Mathlib__Data__Nat__Bits______unexpand__Bool__xor__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_Nat_Bits_0____aux__Mathlib__Data__Nat__Bits______unexpand__Bool__xor__1___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Nat_Bits_0____aux__Mathlib__Data__Nat__Bits______unexpand__Bool__xor__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Bits_0____aux__Mathlib__Data__Nat__Bits______unexpand__Bool__xor__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Bits_0____aux__Mathlib__Data__Nat__Bits______unexpand__Bool__xor__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_Nat_boddDiv2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Nat_boddDiv2___closed__0 = (const lean_object*)&lp_mathlib_Nat_boddDiv2___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Nat_boddDiv2(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_boddDiv2___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_div2(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_div2___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Nat_bodd(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_bodd___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_shiftLeft_x27(uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_shiftLeft_x27___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Bits_0__Nat_boddDiv2_match__3_splitter___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Bits_0__Nat_boddDiv2_match__3_splitter___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Bits_0__Nat_boddDiv2_match__3_splitter(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Bits_0__Nat_boddDiv2_match__3_splitter___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_binaryRec___at___00Nat_size_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_binaryRec___at___00Nat_size_spec__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_size(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_size___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_binaryRec___at___00Nat_bits_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_binaryRec___at___00Nat_bits_spec__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_bits(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_bits___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Nat_ldiff___lam__0(uint8_t, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Nat_ldiff___lam__0___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Nat_ldiff___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Nat_ldiff___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Nat_ldiff___closed__0 = (const lean_object*)&lp_mathlib_Nat_ldiff___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Nat_ldiff(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_ldiff___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Bits_0__Nat_boddDiv2_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Bits_0__Nat_boddDiv2_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib___private_Mathlib_Data_Nat_Bits_0____aux__Mathlib__Data__Nat__Bits______macroRules____private__Mathlib__Data__Nat__Bits__0__termBxor__1___closed__1(void){
_start:
{
lean_object* v___x_37_; lean_object* v___x_38_; 
v___x_37_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_Nat_Bits_0____aux__Mathlib__Data__Nat__Bits______macroRules____private__Mathlib__Data__Nat__Bits__0__termBxor__1___closed__0));
v___x_38_ = l_String_toRawSubstring_x27(v___x_37_);
return v___x_38_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Bits_0____aux__Mathlib__Data__Nat__Bits______macroRules____private__Mathlib__Data__Nat__Bits__0__termBxor__1(lean_object* v_x_51_, lean_object* v_a_52_, lean_object* v_a_53_){
_start:
{
lean_object* v___x_54_; uint8_t v___x_55_; 
v___x_54_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_Nat_Bits_0__termBxor___closed__12));
v___x_55_ = l_Lean_Syntax_isOfKind(v_x_51_, v___x_54_);
if (v___x_55_ == 0)
{
lean_object* v___x_56_; lean_object* v___x_57_; 
v___x_56_ = lean_box(1);
v___x_57_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_57_, 0, v___x_56_);
lean_ctor_set(v___x_57_, 1, v_a_53_);
return v___x_57_;
}
else
{
lean_object* v_quotContext_58_; lean_object* v_currMacroScope_59_; lean_object* v_ref_60_; uint8_t v___x_61_; lean_object* v___x_62_; lean_object* v___x_63_; lean_object* v___x_64_; lean_object* v___x_65_; lean_object* v___x_66_; lean_object* v___x_67_; lean_object* v___x_68_; 
v_quotContext_58_ = lean_ctor_get(v_a_52_, 1);
v_currMacroScope_59_ = lean_ctor_get(v_a_52_, 2);
v_ref_60_ = lean_ctor_get(v_a_52_, 5);
v___x_61_ = 0;
v___x_62_ = l_Lean_SourceInfo_fromRef(v_ref_60_, v___x_61_);
v___x_63_ = lean_obj_once(&lp_mathlib___private_Mathlib_Data_Nat_Bits_0____aux__Mathlib__Data__Nat__Bits______macroRules____private__Mathlib__Data__Nat__Bits__0__termBxor__1___closed__1, &lp_mathlib___private_Mathlib_Data_Nat_Bits_0____aux__Mathlib__Data__Nat__Bits______macroRules____private__Mathlib__Data__Nat__Bits__0__termBxor__1___closed__1_once, _init_lp_mathlib___private_Mathlib_Data_Nat_Bits_0____aux__Mathlib__Data__Nat__Bits______macroRules____private__Mathlib__Data__Nat__Bits__0__termBxor__1___closed__1);
v___x_64_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_Nat_Bits_0____aux__Mathlib__Data__Nat__Bits______macroRules____private__Mathlib__Data__Nat__Bits__0__termBxor__1___closed__2));
lean_inc(v_currMacroScope_59_);
lean_inc(v_quotContext_58_);
v___x_65_ = l_Lean_addMacroScope(v_quotContext_58_, v___x_64_, v_currMacroScope_59_);
v___x_66_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_Nat_Bits_0____aux__Mathlib__Data__Nat__Bits______macroRules____private__Mathlib__Data__Nat__Bits__0__termBxor__1___closed__6));
v___x_67_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_67_, 0, v___x_62_);
lean_ctor_set(v___x_67_, 1, v___x_63_);
lean_ctor_set(v___x_67_, 2, v___x_65_);
lean_ctor_set(v___x_67_, 3, v___x_66_);
v___x_68_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_68_, 0, v___x_67_);
lean_ctor_set(v___x_68_, 1, v_a_53_);
return v___x_68_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Bits_0____aux__Mathlib__Data__Nat__Bits______macroRules____private__Mathlib__Data__Nat__Bits__0__termBxor__1___boxed(lean_object* v_x_69_, lean_object* v_a_70_, lean_object* v_a_71_){
_start:
{
lean_object* v_res_72_; 
v_res_72_ = lp_mathlib___private_Mathlib_Data_Nat_Bits_0____aux__Mathlib__Data__Nat__Bits______macroRules____private__Mathlib__Data__Nat__Bits__0__termBxor__1(v_x_69_, v_a_70_, v_a_71_);
lean_dec_ref(v_a_70_);
return v_res_72_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Bits_0____aux__Mathlib__Data__Nat__Bits______unexpand__Bool__xor__1(lean_object* v_x_76_, lean_object* v_a_77_, lean_object* v_a_78_){
_start:
{
lean_object* v___x_79_; uint8_t v___x_80_; 
v___x_79_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_Nat_Bits_0____aux__Mathlib__Data__Nat__Bits______unexpand__Bool__xor__1___closed__1));
lean_inc(v_x_76_);
v___x_80_ = l_Lean_Syntax_isOfKind(v_x_76_, v___x_79_);
if (v___x_80_ == 0)
{
lean_object* v___x_81_; lean_object* v___x_82_; 
lean_dec(v_x_76_);
v___x_81_ = lean_box(0);
v___x_82_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_82_, 0, v___x_81_);
lean_ctor_set(v___x_82_, 1, v_a_78_);
return v___x_82_;
}
else
{
lean_object* v_ref_83_; uint8_t v___x_84_; lean_object* v___x_85_; lean_object* v___x_86_; lean_object* v___x_87_; lean_object* v___x_88_; lean_object* v___x_89_; lean_object* v___x_90_; 
v_ref_83_ = l_Lean_replaceRef(v_x_76_, v_a_77_);
lean_dec(v_x_76_);
v___x_84_ = 0;
v___x_85_ = l_Lean_SourceInfo_fromRef(v_ref_83_, v___x_84_);
lean_dec(v_ref_83_);
v___x_86_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_Nat_Bits_0__termBxor___closed__12));
v___x_87_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_Nat_Bits_0__termBxor___closed__13));
lean_inc(v___x_85_);
v___x_88_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_88_, 0, v___x_85_);
lean_ctor_set(v___x_88_, 1, v___x_87_);
v___x_89_ = l_Lean_Syntax_node1(v___x_85_, v___x_86_, v___x_88_);
v___x_90_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_90_, 0, v___x_89_);
lean_ctor_set(v___x_90_, 1, v_a_78_);
return v___x_90_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Bits_0____aux__Mathlib__Data__Nat__Bits______unexpand__Bool__xor__1___boxed(lean_object* v_x_91_, lean_object* v_a_92_, lean_object* v_a_93_){
_start:
{
lean_object* v_res_94_; 
v_res_94_ = lp_mathlib___private_Mathlib_Data_Nat_Bits_0____aux__Mathlib__Data__Nat__Bits______unexpand__Bool__xor__1(v_x_91_, v_a_92_, v_a_93_);
lean_dec(v_a_92_);
return v_res_94_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_boddDiv2(lean_object* v_x_99_){
_start:
{
lean_object* v_zero_100_; uint8_t v_isZero_101_; 
v_zero_100_ = lean_unsigned_to_nat(0u);
v_isZero_101_ = lean_nat_dec_eq(v_x_99_, v_zero_100_);
if (v_isZero_101_ == 1)
{
lean_object* v___x_102_; 
v___x_102_ = ((lean_object*)(lp_mathlib_Nat_boddDiv2___closed__0));
return v___x_102_;
}
else
{
lean_object* v_one_103_; lean_object* v_n_104_; lean_object* v___x_105_; lean_object* v_fst_106_; uint8_t v___x_107_; 
v_one_103_ = lean_unsigned_to_nat(1u);
v_n_104_ = lean_nat_sub(v_x_99_, v_one_103_);
v___x_105_ = lp_mathlib_Nat_boddDiv2(v_n_104_);
lean_dec(v_n_104_);
v_fst_106_ = lean_ctor_get(v___x_105_, 0);
lean_inc(v_fst_106_);
v___x_107_ = lean_unbox(v_fst_106_);
lean_dec(v_fst_106_);
if (v___x_107_ == 0)
{
lean_object* v_snd_108_; lean_object* v___x_110_; uint8_t v_isShared_111_; uint8_t v_isSharedCheck_117_; 
v_snd_108_ = lean_ctor_get(v___x_105_, 1);
v_isSharedCheck_117_ = !lean_is_exclusive(v___x_105_);
if (v_isSharedCheck_117_ == 0)
{
lean_object* v_unused_118_; 
v_unused_118_ = lean_ctor_get(v___x_105_, 0);
lean_dec(v_unused_118_);
v___x_110_ = v___x_105_;
v_isShared_111_ = v_isSharedCheck_117_;
goto v_resetjp_109_;
}
else
{
lean_inc(v_snd_108_);
lean_dec(v___x_105_);
v___x_110_ = lean_box(0);
v_isShared_111_ = v_isSharedCheck_117_;
goto v_resetjp_109_;
}
v_resetjp_109_:
{
uint8_t v___x_112_; lean_object* v___x_113_; lean_object* v___x_115_; 
v___x_112_ = 1;
v___x_113_ = lean_box(v___x_112_);
if (v_isShared_111_ == 0)
{
lean_ctor_set(v___x_110_, 0, v___x_113_);
v___x_115_ = v___x_110_;
goto v_reusejp_114_;
}
else
{
lean_object* v_reuseFailAlloc_116_; 
v_reuseFailAlloc_116_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_116_, 0, v___x_113_);
lean_ctor_set(v_reuseFailAlloc_116_, 1, v_snd_108_);
v___x_115_ = v_reuseFailAlloc_116_;
goto v_reusejp_114_;
}
v_reusejp_114_:
{
return v___x_115_;
}
}
}
else
{
lean_object* v_snd_119_; lean_object* v___x_121_; uint8_t v_isShared_122_; uint8_t v_isSharedCheck_128_; 
v_snd_119_ = lean_ctor_get(v___x_105_, 1);
v_isSharedCheck_128_ = !lean_is_exclusive(v___x_105_);
if (v_isSharedCheck_128_ == 0)
{
lean_object* v_unused_129_; 
v_unused_129_ = lean_ctor_get(v___x_105_, 0);
lean_dec(v_unused_129_);
v___x_121_ = v___x_105_;
v_isShared_122_ = v_isSharedCheck_128_;
goto v_resetjp_120_;
}
else
{
lean_inc(v_snd_119_);
lean_dec(v___x_105_);
v___x_121_ = lean_box(0);
v_isShared_122_ = v_isSharedCheck_128_;
goto v_resetjp_120_;
}
v_resetjp_120_:
{
lean_object* v___x_123_; lean_object* v___x_124_; lean_object* v___x_126_; 
v___x_123_ = lean_nat_add(v_snd_119_, v_one_103_);
lean_dec(v_snd_119_);
v___x_124_ = lean_box(v_isZero_101_);
if (v_isShared_122_ == 0)
{
lean_ctor_set(v___x_121_, 1, v___x_123_);
lean_ctor_set(v___x_121_, 0, v___x_124_);
v___x_126_ = v___x_121_;
goto v_reusejp_125_;
}
else
{
lean_object* v_reuseFailAlloc_127_; 
v_reuseFailAlloc_127_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_127_, 0, v___x_124_);
lean_ctor_set(v_reuseFailAlloc_127_, 1, v___x_123_);
v___x_126_ = v_reuseFailAlloc_127_;
goto v_reusejp_125_;
}
v_reusejp_125_:
{
return v___x_126_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_boddDiv2___boxed(lean_object* v_x_130_){
_start:
{
lean_object* v_res_131_; 
v_res_131_ = lp_mathlib_Nat_boddDiv2(v_x_130_);
lean_dec(v_x_130_);
return v_res_131_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_div2(lean_object* v_n_132_){
_start:
{
lean_object* v___x_133_; lean_object* v___x_134_; 
v___x_133_ = lean_unsigned_to_nat(1u);
v___x_134_ = lean_nat_shiftr(v_n_132_, v___x_133_);
return v___x_134_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_div2___boxed(lean_object* v_n_135_){
_start:
{
lean_object* v_res_136_; 
v_res_136_ = lp_mathlib_Nat_div2(v_n_135_);
lean_dec(v_n_135_);
return v_res_136_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Nat_bodd(lean_object* v_n_137_){
_start:
{
lean_object* v___x_138_; uint8_t v___x_139_; 
v___x_138_ = lean_unsigned_to_nat(0u);
v___x_139_ = l_Nat_testBit(v_n_137_, v___x_138_);
return v___x_139_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_bodd___boxed(lean_object* v_n_140_){
_start:
{
uint8_t v_res_141_; lean_object* v_r_142_; 
v_res_141_ = lp_mathlib_Nat_bodd(v_n_140_);
lean_dec(v_n_140_);
v_r_142_ = lean_box(v_res_141_);
return v_r_142_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_shiftLeft_x27(uint8_t v_b_143_, lean_object* v_m_144_, lean_object* v_x_145_){
_start:
{
lean_object* v_zero_146_; uint8_t v_isZero_147_; 
v_zero_146_ = lean_unsigned_to_nat(0u);
v_isZero_147_ = lean_nat_dec_eq(v_x_145_, v_zero_146_);
if (v_isZero_147_ == 1)
{
lean_inc(v_m_144_);
return v_m_144_;
}
else
{
lean_object* v_one_148_; lean_object* v_n_149_; lean_object* v___x_150_; lean_object* v___x_151_; 
v_one_148_ = lean_unsigned_to_nat(1u);
v_n_149_ = lean_nat_sub(v_x_145_, v_one_148_);
v___x_150_ = lp_mathlib_Nat_shiftLeft_x27(v_b_143_, v_m_144_, v_n_149_);
lean_dec(v_n_149_);
v___x_151_ = lp_mathlib_Nat_bit(v_b_143_, v___x_150_);
lean_dec(v___x_150_);
return v___x_151_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_shiftLeft_x27___boxed(lean_object* v_b_152_, lean_object* v_m_153_, lean_object* v_x_154_){
_start:
{
uint8_t v_b_boxed_155_; lean_object* v_res_156_; 
v_b_boxed_155_ = lean_unbox(v_b_152_);
v_res_156_ = lp_mathlib_Nat_shiftLeft_x27(v_b_boxed_155_, v_m_153_, v_x_154_);
lean_dec(v_x_154_);
lean_dec(v_m_153_);
return v_res_156_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Bits_0__Nat_boddDiv2_match__3_splitter___redArg(lean_object* v_x_157_, lean_object* v_h__1_158_, lean_object* v_h__2_159_){
_start:
{
lean_object* v_zero_160_; uint8_t v_isZero_161_; 
v_zero_160_ = lean_unsigned_to_nat(0u);
v_isZero_161_ = lean_nat_dec_eq(v_x_157_, v_zero_160_);
if (v_isZero_161_ == 1)
{
lean_object* v___x_162_; lean_object* v___x_163_; 
lean_dec(v_h__2_159_);
v___x_162_ = lean_box(0);
v___x_163_ = lean_apply_1(v_h__1_158_, v___x_162_);
return v___x_163_;
}
else
{
lean_object* v_one_164_; lean_object* v_n_165_; lean_object* v___x_166_; 
lean_dec(v_h__1_158_);
v_one_164_ = lean_unsigned_to_nat(1u);
v_n_165_ = lean_nat_sub(v_x_157_, v_one_164_);
v___x_166_ = lean_apply_1(v_h__2_159_, v_n_165_);
return v___x_166_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Bits_0__Nat_boddDiv2_match__3_splitter___redArg___boxed(lean_object* v_x_167_, lean_object* v_h__1_168_, lean_object* v_h__2_169_){
_start:
{
lean_object* v_res_170_; 
v_res_170_ = lp_mathlib___private_Mathlib_Data_Nat_Bits_0__Nat_boddDiv2_match__3_splitter___redArg(v_x_167_, v_h__1_168_, v_h__2_169_);
lean_dec(v_x_167_);
return v_res_170_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Bits_0__Nat_boddDiv2_match__3_splitter(lean_object* v_motive_171_, lean_object* v_x_172_, lean_object* v_h__1_173_, lean_object* v_h__2_174_){
_start:
{
lean_object* v_zero_175_; uint8_t v_isZero_176_; 
v_zero_175_ = lean_unsigned_to_nat(0u);
v_isZero_176_ = lean_nat_dec_eq(v_x_172_, v_zero_175_);
if (v_isZero_176_ == 1)
{
lean_object* v___x_177_; lean_object* v___x_178_; 
lean_dec(v_h__2_174_);
v___x_177_ = lean_box(0);
v___x_178_ = lean_apply_1(v_h__1_173_, v___x_177_);
return v___x_178_;
}
else
{
lean_object* v_one_179_; lean_object* v_n_180_; lean_object* v___x_181_; 
lean_dec(v_h__1_173_);
v_one_179_ = lean_unsigned_to_nat(1u);
v_n_180_ = lean_nat_sub(v_x_172_, v_one_179_);
v___x_181_ = lean_apply_1(v_h__2_174_, v_n_180_);
return v___x_181_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Bits_0__Nat_boddDiv2_match__3_splitter___boxed(lean_object* v_motive_182_, lean_object* v_x_183_, lean_object* v_h__1_184_, lean_object* v_h__2_185_){
_start:
{
lean_object* v_res_186_; 
v_res_186_ = lp_mathlib___private_Mathlib_Data_Nat_Bits_0__Nat_boddDiv2_match__3_splitter(v_motive_182_, v_x_183_, v_h__1_184_, v_h__2_185_);
lean_dec(v_x_183_);
return v_res_186_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_binaryRec___at___00Nat_size_spec__0(lean_object* v_zero_187_, lean_object* v_n_188_){
_start:
{
lean_object* v___x_189_; uint8_t v___x_190_; 
v___x_189_ = lean_unsigned_to_nat(0u);
v___x_190_ = lean_nat_dec_eq(v_n_188_, v___x_189_);
if (v___x_190_ == 0)
{
lean_object* v___x_191_; lean_object* v___x_192_; lean_object* v___x_193_; lean_object* v___x_194_; 
v___x_191_ = lean_unsigned_to_nat(1u);
v___x_192_ = lean_nat_shiftr(v_n_188_, v___x_191_);
v___x_193_ = lp_mathlib_Nat_binaryRec___at___00Nat_size_spec__0(v_zero_187_, v___x_192_);
lean_dec(v___x_192_);
v___x_194_ = lean_nat_add(v___x_193_, v___x_191_);
lean_dec(v___x_193_);
return v___x_194_;
}
else
{
lean_inc(v_zero_187_);
return v_zero_187_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_binaryRec___at___00Nat_size_spec__0___boxed(lean_object* v_zero_195_, lean_object* v_n_196_){
_start:
{
lean_object* v_res_197_; 
v_res_197_ = lp_mathlib_Nat_binaryRec___at___00Nat_size_spec__0(v_zero_195_, v_n_196_);
lean_dec(v_n_196_);
lean_dec(v_zero_195_);
return v_res_197_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_size(lean_object* v_n_198_){
_start:
{
lean_object* v___x_199_; lean_object* v___x_200_; 
v___x_199_ = lean_unsigned_to_nat(0u);
v___x_200_ = lp_mathlib_Nat_binaryRec___at___00Nat_size_spec__0(v___x_199_, v_n_198_);
return v___x_200_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_size___boxed(lean_object* v_n_201_){
_start:
{
lean_object* v_res_202_; 
v_res_202_ = lp_mathlib_Nat_size(v_n_201_);
lean_dec(v_n_201_);
return v_res_202_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_binaryRec___at___00Nat_bits_spec__0(lean_object* v_zero_203_, lean_object* v_n_204_){
_start:
{
lean_object* v___x_205_; uint8_t v___x_206_; 
v___x_205_ = lean_unsigned_to_nat(0u);
v___x_206_ = lean_nat_dec_eq(v_n_204_, v___x_205_);
if (v___x_206_ == 0)
{
lean_object* v___x_207_; uint8_t v___y_209_; lean_object* v___x_214_; uint8_t v___x_215_; 
v___x_207_ = lean_unsigned_to_nat(1u);
v___x_214_ = lean_nat_land(v___x_207_, v_n_204_);
v___x_215_ = lean_nat_dec_eq(v___x_214_, v___x_205_);
lean_dec(v___x_214_);
if (v___x_215_ == 0)
{
uint8_t v___x_216_; 
v___x_216_ = 1;
v___y_209_ = v___x_216_;
goto v___jp_208_;
}
else
{
v___y_209_ = v___x_206_;
goto v___jp_208_;
}
v___jp_208_:
{
lean_object* v___x_210_; lean_object* v___x_211_; lean_object* v___x_212_; lean_object* v___x_213_; 
v___x_210_ = lean_nat_shiftr(v_n_204_, v___x_207_);
v___x_211_ = lp_mathlib_Nat_binaryRec___at___00Nat_bits_spec__0(v_zero_203_, v___x_210_);
lean_dec(v___x_210_);
v___x_212_ = lean_box(v___y_209_);
v___x_213_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_213_, 0, v___x_212_);
lean_ctor_set(v___x_213_, 1, v___x_211_);
return v___x_213_;
}
}
else
{
lean_inc(v_zero_203_);
return v_zero_203_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_binaryRec___at___00Nat_bits_spec__0___boxed(lean_object* v_zero_217_, lean_object* v_n_218_){
_start:
{
lean_object* v_res_219_; 
v_res_219_ = lp_mathlib_Nat_binaryRec___at___00Nat_bits_spec__0(v_zero_217_, v_n_218_);
lean_dec(v_n_218_);
lean_dec(v_zero_217_);
return v_res_219_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_bits(lean_object* v_n_220_){
_start:
{
lean_object* v___x_221_; lean_object* v___x_222_; 
v___x_221_ = lean_box(0);
v___x_222_ = lp_mathlib_Nat_binaryRec___at___00Nat_bits_spec__0(v___x_221_, v_n_220_);
return v___x_222_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_bits___boxed(lean_object* v_n_223_){
_start:
{
lean_object* v_res_224_; 
v_res_224_ = lp_mathlib_Nat_bits(v_n_223_);
lean_dec(v_n_223_);
return v_res_224_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Nat_ldiff___lam__0(uint8_t v_a_225_, uint8_t v_b_226_){
_start:
{
if (v_a_225_ == 0)
{
return v_a_225_;
}
else
{
if (v_b_226_ == 0)
{
return v_a_225_;
}
else
{
uint8_t v___x_227_; 
v___x_227_ = 0;
return v___x_227_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_ldiff___lam__0___boxed(lean_object* v_a_228_, lean_object* v_b_229_){
_start:
{
uint8_t v_a_boxed_230_; uint8_t v_b_boxed_231_; uint8_t v_res_232_; lean_object* v_r_233_; 
v_a_boxed_230_ = lean_unbox(v_a_228_);
v_b_boxed_231_ = lean_unbox(v_b_229_);
v_res_232_ = lp_mathlib_Nat_ldiff___lam__0(v_a_boxed_230_, v_b_boxed_231_);
v_r_233_ = lean_box(v_res_232_);
return v_r_233_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_ldiff(lean_object* v_n_235_, lean_object* v_m_236_){
_start:
{
lean_object* v___f_237_; lean_object* v___x_238_; 
v___f_237_ = ((lean_object*)(lp_mathlib_Nat_ldiff___closed__0));
v___x_238_ = l_Nat_bitwise(v___f_237_, v_n_235_, v_m_236_);
return v___x_238_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_ldiff___boxed(lean_object* v_n_239_, lean_object* v_m_240_){
_start:
{
lean_object* v_res_241_; 
v_res_241_ = lp_mathlib_Nat_ldiff(v_n_239_, v_m_240_);
lean_dec(v_m_240_);
lean_dec(v_n_239_);
return v_res_241_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Bits_0__Nat_boddDiv2_match__1_splitter___redArg(lean_object* v_x_242_, lean_object* v_h__1_243_, lean_object* v_h__2_244_){
_start:
{
lean_object* v_fst_245_; uint8_t v___x_246_; 
v_fst_245_ = lean_ctor_get(v_x_242_, 0);
v___x_246_ = lean_unbox(v_fst_245_);
if (v___x_246_ == 0)
{
lean_object* v_snd_247_; lean_object* v___x_248_; 
lean_dec(v_h__2_244_);
v_snd_247_ = lean_ctor_get(v_x_242_, 1);
lean_inc(v_snd_247_);
lean_dec_ref(v_x_242_);
v___x_248_ = lean_apply_1(v_h__1_243_, v_snd_247_);
return v___x_248_;
}
else
{
lean_object* v_snd_249_; lean_object* v___x_250_; 
lean_dec(v_h__1_243_);
v_snd_249_ = lean_ctor_get(v_x_242_, 1);
lean_inc(v_snd_249_);
lean_dec_ref(v_x_242_);
v___x_250_ = lean_apply_1(v_h__2_244_, v_snd_249_);
return v___x_250_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Bits_0__Nat_boddDiv2_match__1_splitter(lean_object* v_motive_251_, lean_object* v_x_252_, lean_object* v_h__1_253_, lean_object* v_h__2_254_){
_start:
{
lean_object* v_fst_255_; uint8_t v___x_256_; 
v_fst_255_ = lean_ctor_get(v_x_252_, 0);
v___x_256_ = lean_unbox(v_fst_255_);
if (v___x_256_ == 0)
{
lean_object* v_snd_257_; lean_object* v___x_258_; 
lean_dec(v_h__2_254_);
v_snd_257_ = lean_ctor_get(v_x_252_, 1);
lean_inc(v_snd_257_);
lean_dec_ref(v_x_252_);
v___x_258_ = lean_apply_1(v_h__1_253_, v_snd_257_);
return v___x_258_;
}
else
{
lean_object* v_snd_259_; lean_object* v___x_260_; 
lean_dec(v_h__1_253_);
v_snd_259_ = lean_ctor_get(v_x_252_, 1);
lean_inc(v_snd_259_);
lean_dec_ref(v_x_252_);
v___x_260_ = lean_apply_1(v_h__2_254_, v_snd_259_);
return v___x_260_;
}
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_BinaryRec(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_List_Defs(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_Bits(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Nat_BinaryRec(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_List_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Nat_Bits(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Nat_BinaryRec(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_List_Defs(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Nat_Bits(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_BinaryRec(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_List_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Nat_Bits(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Nat_Bits(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Nat_Bits(builtin);
}
#ifdef __cplusplus
}
#endif
