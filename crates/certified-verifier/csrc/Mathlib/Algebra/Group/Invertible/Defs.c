// Lean compiler output
// Module: Mathlib.Algebra.Group.Invertible.Defs
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Defs
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
lean_object* lp_mathlib_Monoid_toMulOneClass___redArg(lean_object*);
lean_object* lp_mathlib_MulOneClass_toMulOne___redArg(lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_term_u215f___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 6, .m_data = "term⅟_"};
static const lean_object* lp_mathlib_term_u215f___00__closed__0 = (const lean_object*)&lp_mathlib_term_u215f___00__closed__0_value;
static const lean_ctor_object lp_mathlib_term_u215f___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term_u215f___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(81, 63, 8, 86, 196, 9, 84, 126)}};
static const lean_object* lp_mathlib_term_u215f___00__closed__1 = (const lean_object*)&lp_mathlib_term_u215f___00__closed__1_value;
static const lean_string_object lp_mathlib_term_u215f___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_term_u215f___00__closed__2 = (const lean_object*)&lp_mathlib_term_u215f___00__closed__2_value;
static const lean_ctor_object lp_mathlib_term_u215f___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term_u215f___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_term_u215f___00__closed__3 = (const lean_object*)&lp_mathlib_term_u215f___00__closed__3_value;
static const lean_string_object lp_mathlib_term_u215f___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "⅟"};
static const lean_object* lp_mathlib_term_u215f___00__closed__4 = (const lean_object*)&lp_mathlib_term_u215f___00__closed__4_value;
static const lean_ctor_object lp_mathlib_term_u215f___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_term_u215f___00__closed__4_value)}};
static const lean_object* lp_mathlib_term_u215f___00__closed__5 = (const lean_object*)&lp_mathlib_term_u215f___00__closed__5_value;
static const lean_string_object lp_mathlib_term_u215f___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_term_u215f___00__closed__6 = (const lean_object*)&lp_mathlib_term_u215f___00__closed__6_value;
static const lean_ctor_object lp_mathlib_term_u215f___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term_u215f___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_term_u215f___00__closed__7 = (const lean_object*)&lp_mathlib_term_u215f___00__closed__7_value;
static const lean_ctor_object lp_mathlib_term_u215f___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_term_u215f___00__closed__7_value),((lean_object*)(((size_t)(1024) << 1) | 1))}};
static const lean_object* lp_mathlib_term_u215f___00__closed__8 = (const lean_object*)&lp_mathlib_term_u215f___00__closed__8_value;
static const lean_ctor_object lp_mathlib_term_u215f___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_term_u215f___00__closed__3_value),((lean_object*)&lp_mathlib_term_u215f___00__closed__5_value),((lean_object*)&lp_mathlib_term_u215f___00__closed__8_value)}};
static const lean_object* lp_mathlib_term_u215f___00__closed__9 = (const lean_object*)&lp_mathlib_term_u215f___00__closed__9_value;
static const lean_ctor_object lp_mathlib_term_u215f___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_term_u215f___00__closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_term_u215f___00__closed__9_value)}};
static const lean_object* lp_mathlib_term_u215f___00__closed__10 = (const lean_object*)&lp_mathlib_term_u215f___00__closed__10_value;
LEAN_EXPORT const lean_object* lp_mathlib_term_u215f__ = (const lean_object*)&lp_mathlib_term_u215f___00__closed__10_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Group__Invertible__Defs______macroRules__term_u215f____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Invertible__Defs______macroRules__term_u215f____1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Invertible__Defs______macroRules__term_u215f____1___closed__0_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Group__Invertible__Defs______macroRules__term_u215f____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Invertible__Defs______macroRules__term_u215f____1___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Invertible__Defs______macroRules__term_u215f____1___closed__1_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Group__Invertible__Defs______macroRules__term_u215f____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Invertible__Defs______macroRules__term_u215f____1___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Invertible__Defs______macroRules__term_u215f____1___closed__2_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Group__Invertible__Defs______macroRules__term_u215f____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Invertible__Defs______macroRules__term_u215f____1___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Invertible__Defs______macroRules__term_u215f____1___closed__3_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Group__Invertible__Defs______macroRules__term_u215f____1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Invertible__Defs______macroRules__term_u215f____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Group__Invertible__Defs______macroRules__term_u215f____1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Invertible__Defs______macroRules__term_u215f____1___closed__4_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Invertible__Defs______macroRules__term_u215f____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Group__Invertible__Defs______macroRules__term_u215f____1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Invertible__Defs______macroRules__term_u215f____1___closed__4_value_aux_1),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Invertible__Defs______macroRules__term_u215f____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Group__Invertible__Defs______macroRules__term_u215f____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Invertible__Defs______macroRules__term_u215f____1___closed__4_value_aux_2),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Invertible__Defs______macroRules__term_u215f____1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Invertible__Defs______macroRules__term_u215f____1___closed__4 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Invertible__Defs______macroRules__term_u215f____1___closed__4_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Group__Invertible__Defs______macroRules__term_u215f____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "Invertible.invOf"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Invertible__Defs______macroRules__term_u215f____1___closed__5 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Invertible__Defs______macroRules__term_u215f____1___closed__5_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Algebra__Group__Invertible__Defs______macroRules__term_u215f____1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Invertible__Defs______macroRules__term_u215f____1___closed__6;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Group__Invertible__Defs______macroRules__term_u215f____1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "Invertible"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Invertible__Defs______macroRules__term_u215f____1___closed__7 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Invertible__Defs______macroRules__term_u215f____1___closed__7_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Group__Invertible__Defs______macroRules__term_u215f____1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "invOf"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Invertible__Defs______macroRules__term_u215f____1___closed__8 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Invertible__Defs______macroRules__term_u215f____1___closed__8_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Group__Invertible__Defs______macroRules__term_u215f____1___closed__9_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Invertible__Defs______macroRules__term_u215f____1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(141, 210, 35, 207, 7, 165, 52, 176)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Group__Invertible__Defs______macroRules__term_u215f____1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Invertible__Defs______macroRules__term_u215f____1___closed__9_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Invertible__Defs______macroRules__term_u215f____1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(162, 56, 156, 147, 206, 163, 116, 58)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Invertible__Defs______macroRules__term_u215f____1___closed__9 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Invertible__Defs______macroRules__term_u215f____1___closed__9_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Group__Invertible__Defs______macroRules__term_u215f____1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Invertible__Defs______macroRules__term_u215f____1___closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Invertible__Defs______macroRules__term_u215f____1___closed__10 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Invertible__Defs______macroRules__term_u215f____1___closed__10_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Group__Invertible__Defs______macroRules__term_u215f____1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Invertible__Defs______macroRules__term_u215f____1___closed__10_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Invertible__Defs______macroRules__term_u215f____1___closed__11 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Invertible__Defs______macroRules__term_u215f____1___closed__11_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Group__Invertible__Defs______macroRules__term_u215f____1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Invertible__Defs______macroRules__term_u215f____1___closed__12 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Invertible__Defs______macroRules__term_u215f____1___closed__12_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Group__Invertible__Defs______macroRules__term_u215f____1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Invertible__Defs______macroRules__term_u215f____1___closed__12_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Invertible__Defs______macroRules__term_u215f____1___closed__13 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Invertible__Defs______macroRules__term_u215f____1___closed__13_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Invertible__Defs______macroRules__term_u215f____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Invertible__Defs______macroRules__term_u215f____1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Group__Invertible__Defs______unexpand__Invertible__invOf__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Invertible__Defs______unexpand__Invertible__invOf__1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Invertible__Defs______unexpand__Invertible__invOf__1___closed__0_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Group__Invertible__Defs______unexpand__Invertible__invOf__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Invertible__Defs______unexpand__Invertible__invOf__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Invertible__Defs______unexpand__Invertible__invOf__1___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Invertible__Defs______unexpand__Invertible__invOf__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Invertible__Defs______unexpand__Invertible__invOf__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Invertible__Defs______unexpand__Invertible__invOf__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Invertible_copy_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Invertible_copy_x27___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Invertible_copy_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Invertible_copy_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Invertible_copy___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Invertible_copy___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Invertible_copy(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Invertible_copy___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfGroup___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfGroup(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleOne___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleOne___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleOne(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleOne___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleInvOf___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleInvOf___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleInvOf(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleInvOf___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleMul___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleMul___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleMul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleMul___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Invertible_mul___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Invertible_mul___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Invertible_mul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Invertible_mul___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfLeftInverse___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfLeftInverse___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfLeftInverse(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfLeftInverse___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfRightInverse___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfRightInverse___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfRightInverse(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfRightInverse___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib___aux__Mathlib__Algebra__Group__Invertible__Defs______macroRules__term_u215f____1___closed__6(void){
_start:
{
lean_object* v___x_35_; lean_object* v___x_36_; 
v___x_35_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Group__Invertible__Defs______macroRules__term_u215f____1___closed__5));
v___x_36_ = l_String_toRawSubstring_x27(v___x_35_);
return v___x_36_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Invertible__Defs______macroRules__term_u215f____1(lean_object* v_x_51_, lean_object* v_a_52_, lean_object* v_a_53_){
_start:
{
lean_object* v___x_54_; uint8_t v___x_55_; 
v___x_54_ = ((lean_object*)(lp_mathlib_term_u215f___00__closed__1));
lean_inc(v_x_51_);
v___x_55_ = l_Lean_Syntax_isOfKind(v_x_51_, v___x_54_);
if (v___x_55_ == 0)
{
lean_object* v___x_56_; lean_object* v___x_57_; 
lean_dec(v_x_51_);
v___x_56_ = lean_box(1);
v___x_57_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_57_, 0, v___x_56_);
lean_ctor_set(v___x_57_, 1, v_a_53_);
return v___x_57_;
}
else
{
lean_object* v_quotContext_58_; lean_object* v_currMacroScope_59_; lean_object* v_ref_60_; lean_object* v___x_61_; lean_object* v___x_62_; uint8_t v___x_63_; lean_object* v___x_64_; lean_object* v___x_65_; lean_object* v___x_66_; lean_object* v___x_67_; lean_object* v___x_68_; lean_object* v___x_69_; lean_object* v___x_70_; lean_object* v___x_71_; lean_object* v___x_72_; lean_object* v___x_73_; lean_object* v___x_74_; 
v_quotContext_58_ = lean_ctor_get(v_a_52_, 1);
v_currMacroScope_59_ = lean_ctor_get(v_a_52_, 2);
v_ref_60_ = lean_ctor_get(v_a_52_, 5);
v___x_61_ = lean_unsigned_to_nat(1u);
v___x_62_ = l_Lean_Syntax_getArg(v_x_51_, v___x_61_);
lean_dec(v_x_51_);
v___x_63_ = 0;
v___x_64_ = l_Lean_SourceInfo_fromRef(v_ref_60_, v___x_63_);
v___x_65_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Group__Invertible__Defs______macroRules__term_u215f____1___closed__4));
v___x_66_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Algebra__Group__Invertible__Defs______macroRules__term_u215f____1___closed__6, &lp_mathlib___aux__Mathlib__Algebra__Group__Invertible__Defs______macroRules__term_u215f____1___closed__6_once, _init_lp_mathlib___aux__Mathlib__Algebra__Group__Invertible__Defs______macroRules__term_u215f____1___closed__6);
v___x_67_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Group__Invertible__Defs______macroRules__term_u215f____1___closed__9));
lean_inc(v_currMacroScope_59_);
lean_inc(v_quotContext_58_);
v___x_68_ = l_Lean_addMacroScope(v_quotContext_58_, v___x_67_, v_currMacroScope_59_);
v___x_69_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Group__Invertible__Defs______macroRules__term_u215f____1___closed__11));
lean_inc_n(v___x_64_, 2);
v___x_70_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_70_, 0, v___x_64_);
lean_ctor_set(v___x_70_, 1, v___x_66_);
lean_ctor_set(v___x_70_, 2, v___x_68_);
lean_ctor_set(v___x_70_, 3, v___x_69_);
v___x_71_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Group__Invertible__Defs______macroRules__term_u215f____1___closed__13));
v___x_72_ = l_Lean_Syntax_node1(v___x_64_, v___x_71_, v___x_62_);
v___x_73_ = l_Lean_Syntax_node2(v___x_64_, v___x_65_, v___x_70_, v___x_72_);
v___x_74_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_74_, 0, v___x_73_);
lean_ctor_set(v___x_74_, 1, v_a_53_);
return v___x_74_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Invertible__Defs______macroRules__term_u215f____1___boxed(lean_object* v_x_75_, lean_object* v_a_76_, lean_object* v_a_77_){
_start:
{
lean_object* v_res_78_; 
v_res_78_ = lp_mathlib___aux__Mathlib__Algebra__Group__Invertible__Defs______macroRules__term_u215f____1(v_x_75_, v_a_76_, v_a_77_);
lean_dec_ref(v_a_76_);
return v_res_78_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Invertible__Defs______unexpand__Invertible__invOf__1(lean_object* v_x_82_, lean_object* v_a_83_, lean_object* v_a_84_){
_start:
{
lean_object* v___x_85_; uint8_t v___x_86_; 
v___x_85_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Group__Invertible__Defs______macroRules__term_u215f____1___closed__4));
lean_inc(v_x_82_);
v___x_86_ = l_Lean_Syntax_isOfKind(v_x_82_, v___x_85_);
if (v___x_86_ == 0)
{
lean_object* v___x_87_; lean_object* v___x_88_; 
lean_dec(v_x_82_);
v___x_87_ = lean_box(0);
v___x_88_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_88_, 0, v___x_87_);
lean_ctor_set(v___x_88_, 1, v_a_84_);
return v___x_88_;
}
else
{
lean_object* v___x_89_; lean_object* v___x_90_; lean_object* v___x_91_; uint8_t v___x_92_; 
v___x_89_ = lean_unsigned_to_nat(0u);
v___x_90_ = l_Lean_Syntax_getArg(v_x_82_, v___x_89_);
v___x_91_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Group__Invertible__Defs______unexpand__Invertible__invOf__1___closed__1));
lean_inc(v___x_90_);
v___x_92_ = l_Lean_Syntax_isOfKind(v___x_90_, v___x_91_);
if (v___x_92_ == 0)
{
lean_object* v___x_93_; lean_object* v___x_94_; 
lean_dec(v___x_90_);
lean_dec(v_x_82_);
v___x_93_ = lean_box(0);
v___x_94_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_94_, 0, v___x_93_);
lean_ctor_set(v___x_94_, 1, v_a_84_);
return v___x_94_;
}
else
{
lean_object* v___x_95_; lean_object* v___x_96_; uint8_t v___x_97_; 
v___x_95_ = lean_unsigned_to_nat(1u);
v___x_96_ = l_Lean_Syntax_getArg(v_x_82_, v___x_95_);
lean_dec(v_x_82_);
lean_inc(v___x_96_);
v___x_97_ = l_Lean_Syntax_matchesNull(v___x_96_, v___x_95_);
if (v___x_97_ == 0)
{
lean_object* v___x_98_; lean_object* v___x_99_; 
lean_dec(v___x_96_);
lean_dec(v___x_90_);
v___x_98_ = lean_box(0);
v___x_99_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_99_, 0, v___x_98_);
lean_ctor_set(v___x_99_, 1, v_a_84_);
return v___x_99_;
}
else
{
lean_object* v___x_100_; lean_object* v_ref_101_; uint8_t v___x_102_; lean_object* v___x_103_; lean_object* v___x_104_; lean_object* v___x_105_; lean_object* v___x_106_; lean_object* v___x_107_; lean_object* v___x_108_; 
v___x_100_ = l_Lean_Syntax_getArg(v___x_96_, v___x_89_);
lean_dec(v___x_96_);
v_ref_101_ = l_Lean_replaceRef(v___x_90_, v_a_83_);
lean_dec(v___x_90_);
v___x_102_ = 0;
v___x_103_ = l_Lean_SourceInfo_fromRef(v_ref_101_, v___x_102_);
lean_dec(v_ref_101_);
v___x_104_ = ((lean_object*)(lp_mathlib_term_u215f___00__closed__1));
v___x_105_ = ((lean_object*)(lp_mathlib_term_u215f___00__closed__4));
lean_inc(v___x_103_);
v___x_106_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_106_, 0, v___x_103_);
lean_ctor_set(v___x_106_, 1, v___x_105_);
v___x_107_ = l_Lean_Syntax_node2(v___x_103_, v___x_104_, v___x_106_, v___x_100_);
v___x_108_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_108_, 0, v___x_107_);
lean_ctor_set(v___x_108_, 1, v_a_84_);
return v___x_108_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Invertible__Defs______unexpand__Invertible__invOf__1___boxed(lean_object* v_x_109_, lean_object* v_a_110_, lean_object* v_a_111_){
_start:
{
lean_object* v_res_112_; 
v_res_112_ = lp_mathlib___aux__Mathlib__Algebra__Group__Invertible__Defs______unexpand__Invertible__invOf__1(v_x_109_, v_a_110_, v_a_111_);
lean_dec(v_a_110_);
return v_res_112_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Invertible_copy_x27___redArg(lean_object* v_si_113_){
_start:
{
lean_inc(v_si_113_);
return v_si_113_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Invertible_copy_x27___redArg___boxed(lean_object* v_si_114_){
_start:
{
lean_object* v_res_115_; 
v_res_115_ = lp_mathlib_Invertible_copy_x27___redArg(v_si_114_);
lean_dec(v_si_114_);
return v_res_115_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Invertible_copy_x27(lean_object* v_00_u03b1_116_, lean_object* v_inst_117_, lean_object* v_r_118_, lean_object* v_hr_119_, lean_object* v_s_120_, lean_object* v_si_121_, lean_object* v_hs_122_, lean_object* v_hsi_123_){
_start:
{
lean_inc(v_si_121_);
return v_si_121_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Invertible_copy_x27___boxed(lean_object* v_00_u03b1_124_, lean_object* v_inst_125_, lean_object* v_r_126_, lean_object* v_hr_127_, lean_object* v_s_128_, lean_object* v_si_129_, lean_object* v_hs_130_, lean_object* v_hsi_131_){
_start:
{
lean_object* v_res_132_; 
v_res_132_ = lp_mathlib_Invertible_copy_x27(v_00_u03b1_124_, v_inst_125_, v_r_126_, v_hr_127_, v_s_128_, v_si_129_, v_hs_130_, v_hsi_131_);
lean_dec(v_si_129_);
lean_dec(v_s_128_);
lean_dec(v_hr_127_);
lean_dec(v_r_126_);
lean_dec_ref(v_inst_125_);
return v_res_132_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Invertible_copy___redArg(lean_object* v_hr_133_){
_start:
{
lean_inc(v_hr_133_);
return v_hr_133_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Invertible_copy___redArg___boxed(lean_object* v_hr_134_){
_start:
{
lean_object* v_res_135_; 
v_res_135_ = lp_mathlib_Invertible_copy___redArg(v_hr_134_);
lean_dec(v_hr_134_);
return v_res_135_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Invertible_copy(lean_object* v_00_u03b1_136_, lean_object* v_inst_137_, lean_object* v_r_138_, lean_object* v_hr_139_, lean_object* v_s_140_, lean_object* v_hs_141_){
_start:
{
lean_inc(v_hr_139_);
return v_hr_139_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Invertible_copy___boxed(lean_object* v_00_u03b1_142_, lean_object* v_inst_143_, lean_object* v_r_144_, lean_object* v_hr_145_, lean_object* v_s_146_, lean_object* v_hs_147_){
_start:
{
lean_object* v_res_148_; 
v_res_148_ = lp_mathlib_Invertible_copy(v_00_u03b1_142_, v_inst_143_, v_r_144_, v_hr_145_, v_s_146_, v_hs_147_);
lean_dec(v_s_146_);
lean_dec(v_hr_145_);
lean_dec(v_r_144_);
lean_dec_ref(v_inst_143_);
return v_res_148_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfGroup___redArg(lean_object* v_inst_149_, lean_object* v_a_150_){
_start:
{
lean_object* v_toInv_151_; lean_object* v___x_152_; 
v_toInv_151_ = lean_ctor_get(v_inst_149_, 1);
lean_inc(v_toInv_151_);
lean_dec_ref(v_inst_149_);
v___x_152_ = lean_apply_1(v_toInv_151_, v_a_150_);
return v___x_152_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfGroup(lean_object* v_00_u03b1_153_, lean_object* v_inst_154_, lean_object* v_a_155_){
_start:
{
lean_object* v___x_156_; 
v___x_156_ = lp_mathlib_invertibleOfGroup___redArg(v_inst_154_, v_a_155_);
return v___x_156_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleOne___redArg(lean_object* v_inst_157_){
_start:
{
lean_object* v___x_158_; lean_object* v___x_159_; lean_object* v_toOne_160_; 
v___x_158_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_157_);
v___x_159_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_158_);
v_toOne_160_ = lean_ctor_get(v___x_159_, 0);
lean_inc(v_toOne_160_);
lean_dec_ref(v___x_159_);
return v_toOne_160_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleOne___redArg___boxed(lean_object* v_inst_161_){
_start:
{
lean_object* v_res_162_; 
v_res_162_ = lp_mathlib_invertibleOne___redArg(v_inst_161_);
lean_dec_ref(v_inst_161_);
return v_res_162_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleOne(lean_object* v_00_u03b1_163_, lean_object* v_inst_164_){
_start:
{
lean_object* v___x_165_; 
v___x_165_ = lp_mathlib_invertibleOne___redArg(v_inst_164_);
return v___x_165_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleOne___boxed(lean_object* v_00_u03b1_166_, lean_object* v_inst_167_){
_start:
{
lean_object* v_res_168_; 
v_res_168_ = lp_mathlib_invertibleOne(v_00_u03b1_166_, v_inst_167_);
lean_dec_ref(v_inst_167_);
return v_res_168_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleInvOf___redArg(lean_object* v_a_169_){
_start:
{
lean_inc(v_a_169_);
return v_a_169_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleInvOf___redArg___boxed(lean_object* v_a_170_){
_start:
{
lean_object* v_res_171_; 
v_res_171_ = lp_mathlib_invertibleInvOf___redArg(v_a_170_);
lean_dec(v_a_170_);
return v_res_171_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleInvOf(lean_object* v_00_u03b1_172_, lean_object* v_inst_173_, lean_object* v_inst_174_, lean_object* v_a_175_, lean_object* v_inst_176_){
_start:
{
lean_inc(v_a_175_);
return v_a_175_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleInvOf___boxed(lean_object* v_00_u03b1_177_, lean_object* v_inst_178_, lean_object* v_inst_179_, lean_object* v_a_180_, lean_object* v_inst_181_){
_start:
{
lean_object* v_res_182_; 
v_res_182_ = lp_mathlib_invertibleInvOf(v_00_u03b1_177_, v_inst_178_, v_inst_179_, v_a_180_, v_inst_181_);
lean_dec(v_inst_181_);
lean_dec(v_a_180_);
lean_dec(v_inst_179_);
lean_dec(v_inst_178_);
return v_res_182_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleMul___redArg(lean_object* v_inst_183_, lean_object* v_inst_184_, lean_object* v_inst_185_){
_start:
{
lean_object* v___x_186_; lean_object* v___x_187_; lean_object* v_toMul_188_; lean_object* v___x_189_; 
v___x_186_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_183_);
v___x_187_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_186_);
v_toMul_188_ = lean_ctor_get(v___x_187_, 1);
lean_inc(v_toMul_188_);
lean_dec_ref(v___x_187_);
v___x_189_ = lean_apply_2(v_toMul_188_, v_inst_185_, v_inst_184_);
return v___x_189_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleMul___redArg___boxed(lean_object* v_inst_190_, lean_object* v_inst_191_, lean_object* v_inst_192_){
_start:
{
lean_object* v_res_193_; 
v_res_193_ = lp_mathlib_invertibleMul___redArg(v_inst_190_, v_inst_191_, v_inst_192_);
lean_dec_ref(v_inst_190_);
return v_res_193_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleMul(lean_object* v_00_u03b1_194_, lean_object* v_inst_195_, lean_object* v_a_196_, lean_object* v_b_197_, lean_object* v_inst_198_, lean_object* v_inst_199_){
_start:
{
lean_object* v___x_200_; 
v___x_200_ = lp_mathlib_invertibleMul___redArg(v_inst_195_, v_inst_198_, v_inst_199_);
return v___x_200_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleMul___boxed(lean_object* v_00_u03b1_201_, lean_object* v_inst_202_, lean_object* v_a_203_, lean_object* v_b_204_, lean_object* v_inst_205_, lean_object* v_inst_206_){
_start:
{
lean_object* v_res_207_; 
v_res_207_ = lp_mathlib_invertibleMul(v_00_u03b1_201_, v_inst_202_, v_a_203_, v_b_204_, v_inst_205_, v_inst_206_);
lean_dec(v_b_204_);
lean_dec(v_a_203_);
lean_dec_ref(v_inst_202_);
return v_res_207_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Invertible_mul___redArg(lean_object* v_inst_208_, lean_object* v_x_209_, lean_object* v_x_210_){
_start:
{
lean_object* v___x_211_; 
v___x_211_ = lp_mathlib_invertibleMul___redArg(v_inst_208_, v_x_209_, v_x_210_);
return v___x_211_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Invertible_mul___redArg___boxed(lean_object* v_inst_212_, lean_object* v_x_213_, lean_object* v_x_214_){
_start:
{
lean_object* v_res_215_; 
v_res_215_ = lp_mathlib_Invertible_mul___redArg(v_inst_212_, v_x_213_, v_x_214_);
lean_dec_ref(v_inst_212_);
return v_res_215_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Invertible_mul(lean_object* v_00_u03b1_216_, lean_object* v_inst_217_, lean_object* v_a_218_, lean_object* v_b_219_, lean_object* v_x_220_, lean_object* v_x_221_){
_start:
{
lean_object* v___x_222_; 
v___x_222_ = lp_mathlib_invertibleMul___redArg(v_inst_217_, v_x_220_, v_x_221_);
return v___x_222_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Invertible_mul___boxed(lean_object* v_00_u03b1_223_, lean_object* v_inst_224_, lean_object* v_a_225_, lean_object* v_b_226_, lean_object* v_x_227_, lean_object* v_x_228_){
_start:
{
lean_object* v_res_229_; 
v_res_229_ = lp_mathlib_Invertible_mul(v_00_u03b1_223_, v_inst_224_, v_a_225_, v_b_226_, v_x_227_, v_x_228_);
lean_dec(v_b_226_);
lean_dec(v_a_225_);
lean_dec_ref(v_inst_224_);
return v_res_229_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfLeftInverse___redArg(lean_object* v_b_230_){
_start:
{
lean_inc(v_b_230_);
return v_b_230_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfLeftInverse___redArg___boxed(lean_object* v_b_231_){
_start:
{
lean_object* v_res_232_; 
v_res_232_ = lp_mathlib_invertibleOfLeftInverse___redArg(v_b_231_);
lean_dec(v_b_231_);
return v_res_232_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfLeftInverse(lean_object* v_00_u03b1_233_, lean_object* v_inst_234_, lean_object* v_inst_235_, lean_object* v_a_236_, lean_object* v_b_237_, lean_object* v_h_238_){
_start:
{
lean_inc(v_b_237_);
return v_b_237_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfLeftInverse___boxed(lean_object* v_00_u03b1_239_, lean_object* v_inst_240_, lean_object* v_inst_241_, lean_object* v_a_242_, lean_object* v_b_243_, lean_object* v_h_244_){
_start:
{
lean_object* v_res_245_; 
v_res_245_ = lp_mathlib_invertibleOfLeftInverse(v_00_u03b1_239_, v_inst_240_, v_inst_241_, v_a_242_, v_b_243_, v_h_244_);
lean_dec(v_b_243_);
lean_dec(v_a_242_);
lean_dec_ref(v_inst_240_);
return v_res_245_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfRightInverse___redArg(lean_object* v_b_246_){
_start:
{
lean_inc(v_b_246_);
return v_b_246_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfRightInverse___redArg___boxed(lean_object* v_b_247_){
_start:
{
lean_object* v_res_248_; 
v_res_248_ = lp_mathlib_invertibleOfRightInverse___redArg(v_b_247_);
lean_dec(v_b_247_);
return v_res_248_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfRightInverse(lean_object* v_00_u03b1_249_, lean_object* v_inst_250_, lean_object* v_inst_251_, lean_object* v_a_252_, lean_object* v_b_253_, lean_object* v_h_254_){
_start:
{
lean_inc(v_b_253_);
return v_b_253_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfRightInverse___boxed(lean_object* v_00_u03b1_255_, lean_object* v_inst_256_, lean_object* v_inst_257_, lean_object* v_a_258_, lean_object* v_b_259_, lean_object* v_h_260_){
_start:
{
lean_object* v_res_261_; 
v_res_261_ = lp_mathlib_invertibleOfRightInverse(v_00_u03b1_255_, v_inst_256_, v_inst_257_, v_a_258_, v_b_259_, v_h_260_);
lean_dec(v_b_259_);
lean_dec(v_a_258_);
lean_dec_ref(v_inst_256_);
return v_res_261_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Defs(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Invertible_Defs(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Group_Invertible_Defs(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Defs(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Group_Invertible_Defs(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Invertible_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Group_Invertible_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Group_Invertible_Defs(builtin);
}
#ifdef __cplusplus
}
#endif
