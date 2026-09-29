// Lean compiler output
// Module: Mathlib.Logic.Function.Iterate
// Imports: public import Init public meta import Init public import Mathlib.Logic.Function.Conjugate public import Mathlib.Data.Nat.Notation
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
lean_object* l_Lean_Name_mkStr1(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_iterate___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_iterate(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_term___x5e_x5b___x5d___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "term_^[_]"};
static const lean_object* lp_mathlib_term___x5e_x5b___x5d___closed__0 = (const lean_object*)&lp_mathlib_term___x5e_x5b___x5d___closed__0_value;
static const lean_ctor_object lp_mathlib_term___x5e_x5b___x5d___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___x5e_x5b___x5d___closed__0_value),LEAN_SCALAR_PTR_LITERAL(7, 113, 219, 91, 197, 155, 87, 183)}};
static const lean_object* lp_mathlib_term___x5e_x5b___x5d___closed__1 = (const lean_object*)&lp_mathlib_term___x5e_x5b___x5d___closed__1_value;
static const lean_string_object lp_mathlib_term___x5e_x5b___x5d___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_term___x5e_x5b___x5d___closed__2 = (const lean_object*)&lp_mathlib_term___x5e_x5b___x5d___closed__2_value;
static const lean_ctor_object lp_mathlib_term___x5e_x5b___x5d___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___x5e_x5b___x5d___closed__2_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_term___x5e_x5b___x5d___closed__3 = (const lean_object*)&lp_mathlib_term___x5e_x5b___x5d___closed__3_value;
static const lean_string_object lp_mathlib_term___x5e_x5b___x5d___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "^["};
static const lean_object* lp_mathlib_term___x5e_x5b___x5d___closed__4 = (const lean_object*)&lp_mathlib_term___x5e_x5b___x5d___closed__4_value;
static const lean_ctor_object lp_mathlib_term___x5e_x5b___x5d___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_term___x5e_x5b___x5d___closed__4_value)}};
static const lean_object* lp_mathlib_term___x5e_x5b___x5d___closed__5 = (const lean_object*)&lp_mathlib_term___x5e_x5b___x5d___closed__5_value;
static const lean_string_object lp_mathlib_term___x5e_x5b___x5d___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_term___x5e_x5b___x5d___closed__6 = (const lean_object*)&lp_mathlib_term___x5e_x5b___x5d___closed__6_value;
static const lean_ctor_object lp_mathlib_term___x5e_x5b___x5d___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___x5e_x5b___x5d___closed__6_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_term___x5e_x5b___x5d___closed__7 = (const lean_object*)&lp_mathlib_term___x5e_x5b___x5d___closed__7_value;
static const lean_ctor_object lp_mathlib_term___x5e_x5b___x5d___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_term___x5e_x5b___x5d___closed__7_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_term___x5e_x5b___x5d___closed__8 = (const lean_object*)&lp_mathlib_term___x5e_x5b___x5d___closed__8_value;
static const lean_ctor_object lp_mathlib_term___x5e_x5b___x5d___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_term___x5e_x5b___x5d___closed__3_value),((lean_object*)&lp_mathlib_term___x5e_x5b___x5d___closed__5_value),((lean_object*)&lp_mathlib_term___x5e_x5b___x5d___closed__8_value)}};
static const lean_object* lp_mathlib_term___x5e_x5b___x5d___closed__9 = (const lean_object*)&lp_mathlib_term___x5e_x5b___x5d___closed__9_value;
static const lean_string_object lp_mathlib_term___x5e_x5b___x5d___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "]"};
static const lean_object* lp_mathlib_term___x5e_x5b___x5d___closed__10 = (const lean_object*)&lp_mathlib_term___x5e_x5b___x5d___closed__10_value;
static const lean_ctor_object lp_mathlib_term___x5e_x5b___x5d___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_term___x5e_x5b___x5d___closed__10_value)}};
static const lean_object* lp_mathlib_term___x5e_x5b___x5d___closed__11 = (const lean_object*)&lp_mathlib_term___x5e_x5b___x5d___closed__11_value;
static const lean_ctor_object lp_mathlib_term___x5e_x5b___x5d___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_term___x5e_x5b___x5d___closed__3_value),((lean_object*)&lp_mathlib_term___x5e_x5b___x5d___closed__9_value),((lean_object*)&lp_mathlib_term___x5e_x5b___x5d___closed__11_value)}};
static const lean_object* lp_mathlib_term___x5e_x5b___x5d___closed__12 = (const lean_object*)&lp_mathlib_term___x5e_x5b___x5d___closed__12_value;
static const lean_ctor_object lp_mathlib_term___x5e_x5b___x5d___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_term___x5e_x5b___x5d___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___x5e_x5b___x5d___closed__12_value)}};
static const lean_object* lp_mathlib_term___x5e_x5b___x5d___closed__13 = (const lean_object*)&lp_mathlib_term___x5e_x5b___x5d___closed__13_value;
LEAN_EXPORT const lean_object* lp_mathlib_term___x5e_x5b___x5d = (const lean_object*)&lp_mathlib_term___x5e_x5b___x5d___closed__13_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Logic__Function__Iterate______macroRules__term___x5e_x5b___x5d__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib___aux__Mathlib__Logic__Function__Iterate______macroRules__term___x5e_x5b___x5d__1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Logic__Function__Iterate______macroRules__term___x5e_x5b___x5d__1___closed__0_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Logic__Function__Iterate______macroRules__term___x5e_x5b___x5d__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib___aux__Mathlib__Logic__Function__Iterate______macroRules__term___x5e_x5b___x5d__1___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__Logic__Function__Iterate______macroRules__term___x5e_x5b___x5d__1___closed__1_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Logic__Function__Iterate______macroRules__term___x5e_x5b___x5d__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib___aux__Mathlib__Logic__Function__Iterate______macroRules__term___x5e_x5b___x5d__1___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__Logic__Function__Iterate______macroRules__term___x5e_x5b___x5d__1___closed__2_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Logic__Function__Iterate______macroRules__term___x5e_x5b___x5d__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib___aux__Mathlib__Logic__Function__Iterate______macroRules__term___x5e_x5b___x5d__1___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__Logic__Function__Iterate______macroRules__term___x5e_x5b___x5d__1___closed__3_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Logic__Function__Iterate______macroRules__term___x5e_x5b___x5d__1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Logic__Function__Iterate______macroRules__term___x5e_x5b___x5d__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Logic__Function__Iterate______macroRules__term___x5e_x5b___x5d__1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Logic__Function__Iterate______macroRules__term___x5e_x5b___x5d__1___closed__4_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Logic__Function__Iterate______macroRules__term___x5e_x5b___x5d__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Logic__Function__Iterate______macroRules__term___x5e_x5b___x5d__1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Logic__Function__Iterate______macroRules__term___x5e_x5b___x5d__1___closed__4_value_aux_1),((lean_object*)&lp_mathlib___aux__Mathlib__Logic__Function__Iterate______macroRules__term___x5e_x5b___x5d__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Logic__Function__Iterate______macroRules__term___x5e_x5b___x5d__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Logic__Function__Iterate______macroRules__term___x5e_x5b___x5d__1___closed__4_value_aux_2),((lean_object*)&lp_mathlib___aux__Mathlib__Logic__Function__Iterate______macroRules__term___x5e_x5b___x5d__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib___aux__Mathlib__Logic__Function__Iterate______macroRules__term___x5e_x5b___x5d__1___closed__4 = (const lean_object*)&lp_mathlib___aux__Mathlib__Logic__Function__Iterate______macroRules__term___x5e_x5b___x5d__1___closed__4_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Logic__Function__Iterate______macroRules__term___x5e_x5b___x5d__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "Nat.iterate"};
static const lean_object* lp_mathlib___aux__Mathlib__Logic__Function__Iterate______macroRules__term___x5e_x5b___x5d__1___closed__5 = (const lean_object*)&lp_mathlib___aux__Mathlib__Logic__Function__Iterate______macroRules__term___x5e_x5b___x5d__1___closed__5_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Logic__Function__Iterate______macroRules__term___x5e_x5b___x5d__1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Logic__Function__Iterate______macroRules__term___x5e_x5b___x5d__1___closed__6;
static const lean_string_object lp_mathlib___aux__Mathlib__Logic__Function__Iterate______macroRules__term___x5e_x5b___x5d__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Nat"};
static const lean_object* lp_mathlib___aux__Mathlib__Logic__Function__Iterate______macroRules__term___x5e_x5b___x5d__1___closed__7 = (const lean_object*)&lp_mathlib___aux__Mathlib__Logic__Function__Iterate______macroRules__term___x5e_x5b___x5d__1___closed__7_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Logic__Function__Iterate______macroRules__term___x5e_x5b___x5d__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "iterate"};
static const lean_object* lp_mathlib___aux__Mathlib__Logic__Function__Iterate______macroRules__term___x5e_x5b___x5d__1___closed__8 = (const lean_object*)&lp_mathlib___aux__Mathlib__Logic__Function__Iterate______macroRules__term___x5e_x5b___x5d__1___closed__8_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Logic__Function__Iterate______macroRules__term___x5e_x5b___x5d__1___closed__9_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Logic__Function__Iterate______macroRules__term___x5e_x5b___x5d__1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(155, 221, 223, 104, 58, 13, 204, 158)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Logic__Function__Iterate______macroRules__term___x5e_x5b___x5d__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Logic__Function__Iterate______macroRules__term___x5e_x5b___x5d__1___closed__9_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Logic__Function__Iterate______macroRules__term___x5e_x5b___x5d__1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(37, 234, 178, 146, 85, 250, 177, 110)}};
static const lean_object* lp_mathlib___aux__Mathlib__Logic__Function__Iterate______macroRules__term___x5e_x5b___x5d__1___closed__9 = (const lean_object*)&lp_mathlib___aux__Mathlib__Logic__Function__Iterate______macroRules__term___x5e_x5b___x5d__1___closed__9_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Logic__Function__Iterate______macroRules__term___x5e_x5b___x5d__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Logic__Function__Iterate______macroRules__term___x5e_x5b___x5d__1___closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Logic__Function__Iterate______macroRules__term___x5e_x5b___x5d__1___closed__10 = (const lean_object*)&lp_mathlib___aux__Mathlib__Logic__Function__Iterate______macroRules__term___x5e_x5b___x5d__1___closed__10_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Logic__Function__Iterate______macroRules__term___x5e_x5b___x5d__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Logic__Function__Iterate______macroRules__term___x5e_x5b___x5d__1___closed__9_value)}};
static const lean_object* lp_mathlib___aux__Mathlib__Logic__Function__Iterate______macroRules__term___x5e_x5b___x5d__1___closed__11 = (const lean_object*)&lp_mathlib___aux__Mathlib__Logic__Function__Iterate______macroRules__term___x5e_x5b___x5d__1___closed__11_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Logic__Function__Iterate______macroRules__term___x5e_x5b___x5d__1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Logic__Function__Iterate______macroRules__term___x5e_x5b___x5d__1___closed__11_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Logic__Function__Iterate______macroRules__term___x5e_x5b___x5d__1___closed__12 = (const lean_object*)&lp_mathlib___aux__Mathlib__Logic__Function__Iterate______macroRules__term___x5e_x5b___x5d__1___closed__12_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Logic__Function__Iterate______macroRules__term___x5e_x5b___x5d__1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Logic__Function__Iterate______macroRules__term___x5e_x5b___x5d__1___closed__10_value),((lean_object*)&lp_mathlib___aux__Mathlib__Logic__Function__Iterate______macroRules__term___x5e_x5b___x5d__1___closed__12_value)}};
static const lean_object* lp_mathlib___aux__Mathlib__Logic__Function__Iterate______macroRules__term___x5e_x5b___x5d__1___closed__13 = (const lean_object*)&lp_mathlib___aux__Mathlib__Logic__Function__Iterate______macroRules__term___x5e_x5b___x5d__1___closed__13_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Logic__Function__Iterate______macroRules__term___x5e_x5b___x5d__1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib___aux__Mathlib__Logic__Function__Iterate______macroRules__term___x5e_x5b___x5d__1___closed__14 = (const lean_object*)&lp_mathlib___aux__Mathlib__Logic__Function__Iterate______macroRules__term___x5e_x5b___x5d__1___closed__14_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Logic__Function__Iterate______macroRules__term___x5e_x5b___x5d__1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Logic__Function__Iterate______macroRules__term___x5e_x5b___x5d__1___closed__14_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib___aux__Mathlib__Logic__Function__Iterate______macroRules__term___x5e_x5b___x5d__1___closed__15 = (const lean_object*)&lp_mathlib___aux__Mathlib__Logic__Function__Iterate______macroRules__term___x5e_x5b___x5d__1___closed__15_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Logic__Function__Iterate______macroRules__term___x5e_x5b___x5d__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Logic__Function__Iterate______macroRules__term___x5e_x5b___x5d__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___aux__Mathlib__Logic__Function__Iterate______unexpand__Nat__iterate__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib___aux__Mathlib__Logic__Function__Iterate______unexpand__Nat__iterate__1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Logic__Function__Iterate______unexpand__Nat__iterate__1___closed__0_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Logic__Function__Iterate______unexpand__Nat__iterate__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Logic__Function__Iterate______unexpand__Nat__iterate__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib___aux__Mathlib__Logic__Function__Iterate______unexpand__Nat__iterate__1___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__Logic__Function__Iterate______unexpand__Nat__iterate__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Logic__Function__Iterate______unexpand__Nat__iterate__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Logic__Function__Iterate______unexpand__Nat__iterate__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Iterate_rec___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Iterate_rec(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_iterate___redArg(lean_object* v_op_1_, lean_object* v_x_2_, lean_object* v_x_3_){
_start:
{
lean_object* v_zero_4_; uint8_t v_isZero_5_; 
v_zero_4_ = lean_unsigned_to_nat(0u);
v_isZero_5_ = lean_nat_dec_eq(v_x_2_, v_zero_4_);
if (v_isZero_5_ == 1)
{
lean_dec(v_x_2_);
lean_dec(v_op_1_);
return v_x_3_;
}
else
{
lean_object* v_one_6_; lean_object* v_n_7_; lean_object* v___x_8_; 
v_one_6_ = lean_unsigned_to_nat(1u);
v_n_7_ = lean_nat_sub(v_x_2_, v_one_6_);
lean_dec(v_x_2_);
lean_inc(v_op_1_);
v___x_8_ = lean_apply_1(v_op_1_, v_x_3_);
v_x_2_ = v_n_7_;
v_x_3_ = v___x_8_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_iterate(lean_object* v_00_u03b1_10_, lean_object* v_op_11_, lean_object* v_x_12_, lean_object* v_x_13_){
_start:
{
lean_object* v___x_14_; 
v___x_14_ = lp_mathlib_Nat_iterate___redArg(v_op_11_, v_x_12_, v_x_13_);
return v___x_14_;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Logic__Function__Iterate______macroRules__term___x5e_x5b___x5d__1___closed__6(void){
_start:
{
lean_object* v___x_57_; lean_object* v___x_58_; 
v___x_57_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Logic__Function__Iterate______macroRules__term___x5e_x5b___x5d__1___closed__5));
v___x_58_ = l_String_toRawSubstring_x27(v___x_57_);
return v___x_58_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Logic__Function__Iterate______macroRules__term___x5e_x5b___x5d__1(lean_object* v_x_78_, lean_object* v_a_79_, lean_object* v_a_80_){
_start:
{
lean_object* v___x_81_; uint8_t v___x_82_; 
v___x_81_ = ((lean_object*)(lp_mathlib_term___x5e_x5b___x5d___closed__1));
lean_inc(v_x_78_);
v___x_82_ = l_Lean_Syntax_isOfKind(v_x_78_, v___x_81_);
if (v___x_82_ == 0)
{
lean_object* v___x_83_; lean_object* v___x_84_; 
lean_dec(v_x_78_);
v___x_83_ = lean_box(1);
v___x_84_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_84_, 0, v___x_83_);
lean_ctor_set(v___x_84_, 1, v_a_80_);
return v___x_84_;
}
else
{
lean_object* v_quotContext_85_; lean_object* v_currMacroScope_86_; lean_object* v_ref_87_; lean_object* v___x_88_; lean_object* v___x_89_; lean_object* v___x_90_; lean_object* v___x_91_; uint8_t v___x_92_; lean_object* v___x_93_; lean_object* v___x_94_; lean_object* v___x_95_; lean_object* v___x_96_; lean_object* v___x_97_; lean_object* v___x_98_; lean_object* v___x_99_; lean_object* v___x_100_; lean_object* v___x_101_; lean_object* v___x_102_; lean_object* v___x_103_; 
v_quotContext_85_ = lean_ctor_get(v_a_79_, 1);
v_currMacroScope_86_ = lean_ctor_get(v_a_79_, 2);
v_ref_87_ = lean_ctor_get(v_a_79_, 5);
v___x_88_ = lean_unsigned_to_nat(0u);
v___x_89_ = l_Lean_Syntax_getArg(v_x_78_, v___x_88_);
v___x_90_ = lean_unsigned_to_nat(2u);
v___x_91_ = l_Lean_Syntax_getArg(v_x_78_, v___x_90_);
lean_dec(v_x_78_);
v___x_92_ = 0;
v___x_93_ = l_Lean_SourceInfo_fromRef(v_ref_87_, v___x_92_);
v___x_94_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Logic__Function__Iterate______macroRules__term___x5e_x5b___x5d__1___closed__4));
v___x_95_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Logic__Function__Iterate______macroRules__term___x5e_x5b___x5d__1___closed__6, &lp_mathlib___aux__Mathlib__Logic__Function__Iterate______macroRules__term___x5e_x5b___x5d__1___closed__6_once, _init_lp_mathlib___aux__Mathlib__Logic__Function__Iterate______macroRules__term___x5e_x5b___x5d__1___closed__6);
v___x_96_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Logic__Function__Iterate______macroRules__term___x5e_x5b___x5d__1___closed__9));
lean_inc(v_currMacroScope_86_);
lean_inc(v_quotContext_85_);
v___x_97_ = l_Lean_addMacroScope(v_quotContext_85_, v___x_96_, v_currMacroScope_86_);
v___x_98_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Logic__Function__Iterate______macroRules__term___x5e_x5b___x5d__1___closed__13));
lean_inc_n(v___x_93_, 2);
v___x_99_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_99_, 0, v___x_93_);
lean_ctor_set(v___x_99_, 1, v___x_95_);
lean_ctor_set(v___x_99_, 2, v___x_97_);
lean_ctor_set(v___x_99_, 3, v___x_98_);
v___x_100_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Logic__Function__Iterate______macroRules__term___x5e_x5b___x5d__1___closed__15));
v___x_101_ = l_Lean_Syntax_node2(v___x_93_, v___x_100_, v___x_89_, v___x_91_);
v___x_102_ = l_Lean_Syntax_node2(v___x_93_, v___x_94_, v___x_99_, v___x_101_);
v___x_103_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_103_, 0, v___x_102_);
lean_ctor_set(v___x_103_, 1, v_a_80_);
return v___x_103_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Logic__Function__Iterate______macroRules__term___x5e_x5b___x5d__1___boxed(lean_object* v_x_104_, lean_object* v_a_105_, lean_object* v_a_106_){
_start:
{
lean_object* v_res_107_; 
v_res_107_ = lp_mathlib___aux__Mathlib__Logic__Function__Iterate______macroRules__term___x5e_x5b___x5d__1(v_x_104_, v_a_105_, v_a_106_);
lean_dec_ref(v_a_105_);
return v_res_107_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Logic__Function__Iterate______unexpand__Nat__iterate__1(lean_object* v_x_111_, lean_object* v_a_112_, lean_object* v_a_113_){
_start:
{
lean_object* v___x_114_; uint8_t v___x_115_; 
v___x_114_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Logic__Function__Iterate______macroRules__term___x5e_x5b___x5d__1___closed__4));
lean_inc(v_x_111_);
v___x_115_ = l_Lean_Syntax_isOfKind(v_x_111_, v___x_114_);
if (v___x_115_ == 0)
{
lean_object* v___x_116_; lean_object* v___x_117_; 
lean_dec(v_x_111_);
v___x_116_ = lean_box(0);
v___x_117_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_117_, 0, v___x_116_);
lean_ctor_set(v___x_117_, 1, v_a_113_);
return v___x_117_;
}
else
{
lean_object* v___x_118_; lean_object* v___x_119_; lean_object* v___x_120_; uint8_t v___x_121_; 
v___x_118_ = lean_unsigned_to_nat(0u);
v___x_119_ = l_Lean_Syntax_getArg(v_x_111_, v___x_118_);
v___x_120_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Logic__Function__Iterate______unexpand__Nat__iterate__1___closed__1));
lean_inc(v___x_119_);
v___x_121_ = l_Lean_Syntax_isOfKind(v___x_119_, v___x_120_);
if (v___x_121_ == 0)
{
lean_object* v___x_122_; lean_object* v___x_123_; 
lean_dec(v___x_119_);
lean_dec(v_x_111_);
v___x_122_ = lean_box(0);
v___x_123_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_123_, 0, v___x_122_);
lean_ctor_set(v___x_123_, 1, v_a_113_);
return v___x_123_;
}
else
{
lean_object* v___x_124_; lean_object* v___x_125_; lean_object* v___x_126_; uint8_t v___x_127_; 
v___x_124_ = lean_unsigned_to_nat(1u);
v___x_125_ = l_Lean_Syntax_getArg(v_x_111_, v___x_124_);
lean_dec(v_x_111_);
v___x_126_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_125_);
v___x_127_ = l_Lean_Syntax_matchesNull(v___x_125_, v___x_126_);
if (v___x_127_ == 0)
{
lean_object* v___x_128_; lean_object* v___x_129_; 
lean_dec(v___x_125_);
lean_dec(v___x_119_);
v___x_128_ = lean_box(0);
v___x_129_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_129_, 0, v___x_128_);
lean_ctor_set(v___x_129_, 1, v_a_113_);
return v___x_129_;
}
else
{
lean_object* v___x_130_; lean_object* v___x_131_; lean_object* v_ref_132_; uint8_t v___x_133_; lean_object* v___x_134_; lean_object* v___x_135_; lean_object* v___x_136_; lean_object* v___x_137_; lean_object* v___x_138_; lean_object* v___x_139_; lean_object* v___x_140_; lean_object* v___x_141_; 
v___x_130_ = l_Lean_Syntax_getArg(v___x_125_, v___x_118_);
v___x_131_ = l_Lean_Syntax_getArg(v___x_125_, v___x_124_);
lean_dec(v___x_125_);
v_ref_132_ = l_Lean_replaceRef(v___x_119_, v_a_112_);
lean_dec(v___x_119_);
v___x_133_ = 0;
v___x_134_ = l_Lean_SourceInfo_fromRef(v_ref_132_, v___x_133_);
lean_dec(v_ref_132_);
v___x_135_ = ((lean_object*)(lp_mathlib_term___x5e_x5b___x5d___closed__1));
v___x_136_ = ((lean_object*)(lp_mathlib_term___x5e_x5b___x5d___closed__4));
lean_inc_n(v___x_134_, 2);
v___x_137_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_137_, 0, v___x_134_);
lean_ctor_set(v___x_137_, 1, v___x_136_);
v___x_138_ = ((lean_object*)(lp_mathlib_term___x5e_x5b___x5d___closed__10));
v___x_139_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_139_, 0, v___x_134_);
lean_ctor_set(v___x_139_, 1, v___x_138_);
v___x_140_ = l_Lean_Syntax_node4(v___x_134_, v___x_135_, v___x_130_, v___x_137_, v___x_131_, v___x_139_);
v___x_141_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_141_, 0, v___x_140_);
lean_ctor_set(v___x_141_, 1, v_a_113_);
return v___x_141_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Logic__Function__Iterate______unexpand__Nat__iterate__1___boxed(lean_object* v_x_142_, lean_object* v_a_143_, lean_object* v_a_144_){
_start:
{
lean_object* v_res_145_; 
v_res_145_ = lp_mathlib___aux__Mathlib__Logic__Function__Iterate______unexpand__Nat__iterate__1(v_x_142_, v_a_143_, v_a_144_);
lean_dec(v_a_143_);
return v_res_145_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Iterate_rec___redArg(lean_object* v_a_146_, lean_object* v_arg_147_, lean_object* v_f_148_, lean_object* v_app_149_, lean_object* v_n_150_){
_start:
{
lean_object* v_zero_151_; uint8_t v_isZero_152_; 
v_zero_151_ = lean_unsigned_to_nat(0u);
v_isZero_152_ = lean_nat_dec_eq(v_n_150_, v_zero_151_);
if (v_isZero_152_ == 1)
{
lean_dec(v_n_150_);
lean_dec(v_app_149_);
lean_dec(v_f_148_);
lean_dec(v_a_146_);
return v_arg_147_;
}
else
{
lean_object* v_one_153_; lean_object* v_n_154_; lean_object* v___x_155_; lean_object* v___x_156_; 
v_one_153_ = lean_unsigned_to_nat(1u);
v_n_154_ = lean_nat_sub(v_n_150_, v_one_153_);
lean_dec(v_n_150_);
lean_inc(v_f_148_);
lean_inc(v_a_146_);
v___x_155_ = lean_apply_1(v_f_148_, v_a_146_);
lean_inc(v_app_149_);
v___x_156_ = lean_apply_2(v_app_149_, v_a_146_, v_arg_147_);
v_a_146_ = v___x_155_;
v_arg_147_ = v___x_156_;
v_n_150_ = v_n_154_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Iterate_rec(lean_object* v_00_u03b1_158_, lean_object* v_motive_159_, lean_object* v_a_160_, lean_object* v_arg_161_, lean_object* v_f_162_, lean_object* v_app_163_, lean_object* v_n_164_){
_start:
{
lean_object* v___x_165_; 
v___x_165_ = lp_mathlib_Function_Iterate_rec___redArg(v_a_160_, v_arg_161_, v_f_162_, v_app_163_, v_n_164_);
return v___x_165_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Logic_Function_Conjugate(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_Notation(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Logic_Function_Iterate(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Logic_Function_Conjugate(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Nat_Notation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Logic_Function_Iterate(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Logic_Function_Conjugate(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Nat_Notation(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Logic_Function_Iterate(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Function_Conjugate(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_Notation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Logic_Function_Iterate(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Logic_Function_Iterate(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Logic_Function_Iterate(builtin);
}
#ifdef __cplusplus
}
#endif
