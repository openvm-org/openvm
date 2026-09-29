// Lean compiler output
// Module: Mathlib.Order.Heyting.Boundary
// Imports: public import Init public meta import Init public import Mathlib.Order.BooleanAlgebra.Basic public import Mathlib.Tactic.Common public import Mathlib.Tactic.Attr.Core
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
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Coheyting_boundary___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Coheyting_boundary(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Heyting_term_u2202___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Heyting"};
static const lean_object* lp_mathlib_Heyting_term_u2202___00__closed__0 = (const lean_object*)&lp_mathlib_Heyting_term_u2202___00__closed__0_value;
static const lean_string_object lp_mathlib_Heyting_term_u2202___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 6, .m_data = "term∂_"};
static const lean_object* lp_mathlib_Heyting_term_u2202___00__closed__1 = (const lean_object*)&lp_mathlib_Heyting_term_u2202___00__closed__1_value;
static const lean_ctor_object lp_mathlib_Heyting_term_u2202___00__closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Heyting_term_u2202___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(246, 17, 82, 121, 228, 236, 8, 205)}};
static const lean_ctor_object lp_mathlib_Heyting_term_u2202___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Heyting_term_u2202___00__closed__2_value_aux_0),((lean_object*)&lp_mathlib_Heyting_term_u2202___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(168, 149, 46, 213, 15, 200, 137, 159)}};
static const lean_object* lp_mathlib_Heyting_term_u2202___00__closed__2 = (const lean_object*)&lp_mathlib_Heyting_term_u2202___00__closed__2_value;
static const lean_string_object lp_mathlib_Heyting_term_u2202___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Heyting_term_u2202___00__closed__3 = (const lean_object*)&lp_mathlib_Heyting_term_u2202___00__closed__3_value;
static const lean_ctor_object lp_mathlib_Heyting_term_u2202___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Heyting_term_u2202___00__closed__3_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Heyting_term_u2202___00__closed__4 = (const lean_object*)&lp_mathlib_Heyting_term_u2202___00__closed__4_value;
static const lean_string_object lp_mathlib_Heyting_term_u2202___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 2, .m_data = "∂ "};
static const lean_object* lp_mathlib_Heyting_term_u2202___00__closed__5 = (const lean_object*)&lp_mathlib_Heyting_term_u2202___00__closed__5_value;
static const lean_ctor_object lp_mathlib_Heyting_term_u2202___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Heyting_term_u2202___00__closed__5_value)}};
static const lean_object* lp_mathlib_Heyting_term_u2202___00__closed__6 = (const lean_object*)&lp_mathlib_Heyting_term_u2202___00__closed__6_value;
static const lean_string_object lp_mathlib_Heyting_term_u2202___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_Heyting_term_u2202___00__closed__7 = (const lean_object*)&lp_mathlib_Heyting_term_u2202___00__closed__7_value;
static const lean_ctor_object lp_mathlib_Heyting_term_u2202___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Heyting_term_u2202___00__closed__7_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_Heyting_term_u2202___00__closed__8 = (const lean_object*)&lp_mathlib_Heyting_term_u2202___00__closed__8_value;
static const lean_ctor_object lp_mathlib_Heyting_term_u2202___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_Heyting_term_u2202___00__closed__8_value),((lean_object*)(((size_t)(120) << 1) | 1))}};
static const lean_object* lp_mathlib_Heyting_term_u2202___00__closed__9 = (const lean_object*)&lp_mathlib_Heyting_term_u2202___00__closed__9_value;
static const lean_ctor_object lp_mathlib_Heyting_term_u2202___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Heyting_term_u2202___00__closed__4_value),((lean_object*)&lp_mathlib_Heyting_term_u2202___00__closed__6_value),((lean_object*)&lp_mathlib_Heyting_term_u2202___00__closed__9_value)}};
static const lean_object* lp_mathlib_Heyting_term_u2202___00__closed__10 = (const lean_object*)&lp_mathlib_Heyting_term_u2202___00__closed__10_value;
static const lean_ctor_object lp_mathlib_Heyting_term_u2202___00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Heyting_term_u2202___00__closed__2_value),((lean_object*)(((size_t)(120) << 1) | 1)),((lean_object*)&lp_mathlib_Heyting_term_u2202___00__closed__10_value)}};
static const lean_object* lp_mathlib_Heyting_term_u2202___00__closed__11 = (const lean_object*)&lp_mathlib_Heyting_term_u2202___00__closed__11_value;
LEAN_EXPORT const lean_object* lp_mathlib_Heyting_term_u2202__ = (const lean_object*)&lp_mathlib_Heyting_term_u2202___00__closed__11_value;
static const lean_string_object lp_mathlib_Heyting___aux__Mathlib__Order__Heyting__Boundary______macroRules__Heyting__term_u2202____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Heyting___aux__Mathlib__Order__Heyting__Boundary______macroRules__Heyting__term_u2202____1___closed__0 = (const lean_object*)&lp_mathlib_Heyting___aux__Mathlib__Order__Heyting__Boundary______macroRules__Heyting__term_u2202____1___closed__0_value;
static const lean_string_object lp_mathlib_Heyting___aux__Mathlib__Order__Heyting__Boundary______macroRules__Heyting__term_u2202____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Heyting___aux__Mathlib__Order__Heyting__Boundary______macroRules__Heyting__term_u2202____1___closed__1 = (const lean_object*)&lp_mathlib_Heyting___aux__Mathlib__Order__Heyting__Boundary______macroRules__Heyting__term_u2202____1___closed__1_value;
static const lean_string_object lp_mathlib_Heyting___aux__Mathlib__Order__Heyting__Boundary______macroRules__Heyting__term_u2202____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib_Heyting___aux__Mathlib__Order__Heyting__Boundary______macroRules__Heyting__term_u2202____1___closed__2 = (const lean_object*)&lp_mathlib_Heyting___aux__Mathlib__Order__Heyting__Boundary______macroRules__Heyting__term_u2202____1___closed__2_value;
static const lean_string_object lp_mathlib_Heyting___aux__Mathlib__Order__Heyting__Boundary______macroRules__Heyting__term_u2202____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib_Heyting___aux__Mathlib__Order__Heyting__Boundary______macroRules__Heyting__term_u2202____1___closed__3 = (const lean_object*)&lp_mathlib_Heyting___aux__Mathlib__Order__Heyting__Boundary______macroRules__Heyting__term_u2202____1___closed__3_value;
static const lean_ctor_object lp_mathlib_Heyting___aux__Mathlib__Order__Heyting__Boundary______macroRules__Heyting__term_u2202____1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Heyting___aux__Mathlib__Order__Heyting__Boundary______macroRules__Heyting__term_u2202____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Heyting___aux__Mathlib__Order__Heyting__Boundary______macroRules__Heyting__term_u2202____1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Heyting___aux__Mathlib__Order__Heyting__Boundary______macroRules__Heyting__term_u2202____1___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Heyting___aux__Mathlib__Order__Heyting__Boundary______macroRules__Heyting__term_u2202____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Heyting___aux__Mathlib__Order__Heyting__Boundary______macroRules__Heyting__term_u2202____1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Heyting___aux__Mathlib__Order__Heyting__Boundary______macroRules__Heyting__term_u2202____1___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Heyting___aux__Mathlib__Order__Heyting__Boundary______macroRules__Heyting__term_u2202____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Heyting___aux__Mathlib__Order__Heyting__Boundary______macroRules__Heyting__term_u2202____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Heyting___aux__Mathlib__Order__Heyting__Boundary______macroRules__Heyting__term_u2202____1___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Heyting___aux__Mathlib__Order__Heyting__Boundary______macroRules__Heyting__term_u2202____1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib_Heyting___aux__Mathlib__Order__Heyting__Boundary______macroRules__Heyting__term_u2202____1___closed__4 = (const lean_object*)&lp_mathlib_Heyting___aux__Mathlib__Order__Heyting__Boundary______macroRules__Heyting__term_u2202____1___closed__4_value;
static const lean_string_object lp_mathlib_Heyting___aux__Mathlib__Order__Heyting__Boundary______macroRules__Heyting__term_u2202____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "Coheyting.boundary"};
static const lean_object* lp_mathlib_Heyting___aux__Mathlib__Order__Heyting__Boundary______macroRules__Heyting__term_u2202____1___closed__5 = (const lean_object*)&lp_mathlib_Heyting___aux__Mathlib__Order__Heyting__Boundary______macroRules__Heyting__term_u2202____1___closed__5_value;
static lean_once_cell_t lp_mathlib_Heyting___aux__Mathlib__Order__Heyting__Boundary______macroRules__Heyting__term_u2202____1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Heyting___aux__Mathlib__Order__Heyting__Boundary______macroRules__Heyting__term_u2202____1___closed__6;
static const lean_string_object lp_mathlib_Heyting___aux__Mathlib__Order__Heyting__Boundary______macroRules__Heyting__term_u2202____1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Coheyting"};
static const lean_object* lp_mathlib_Heyting___aux__Mathlib__Order__Heyting__Boundary______macroRules__Heyting__term_u2202____1___closed__7 = (const lean_object*)&lp_mathlib_Heyting___aux__Mathlib__Order__Heyting__Boundary______macroRules__Heyting__term_u2202____1___closed__7_value;
static const lean_string_object lp_mathlib_Heyting___aux__Mathlib__Order__Heyting__Boundary______macroRules__Heyting__term_u2202____1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "boundary"};
static const lean_object* lp_mathlib_Heyting___aux__Mathlib__Order__Heyting__Boundary______macroRules__Heyting__term_u2202____1___closed__8 = (const lean_object*)&lp_mathlib_Heyting___aux__Mathlib__Order__Heyting__Boundary______macroRules__Heyting__term_u2202____1___closed__8_value;
static const lean_ctor_object lp_mathlib_Heyting___aux__Mathlib__Order__Heyting__Boundary______macroRules__Heyting__term_u2202____1___closed__9_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Heyting___aux__Mathlib__Order__Heyting__Boundary______macroRules__Heyting__term_u2202____1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(53, 28, 6, 219, 222, 41, 187, 123)}};
static const lean_ctor_object lp_mathlib_Heyting___aux__Mathlib__Order__Heyting__Boundary______macroRules__Heyting__term_u2202____1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Heyting___aux__Mathlib__Order__Heyting__Boundary______macroRules__Heyting__term_u2202____1___closed__9_value_aux_0),((lean_object*)&lp_mathlib_Heyting___aux__Mathlib__Order__Heyting__Boundary______macroRules__Heyting__term_u2202____1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(199, 193, 144, 41, 223, 144, 86, 148)}};
static const lean_object* lp_mathlib_Heyting___aux__Mathlib__Order__Heyting__Boundary______macroRules__Heyting__term_u2202____1___closed__9 = (const lean_object*)&lp_mathlib_Heyting___aux__Mathlib__Order__Heyting__Boundary______macroRules__Heyting__term_u2202____1___closed__9_value;
static const lean_ctor_object lp_mathlib_Heyting___aux__Mathlib__Order__Heyting__Boundary______macroRules__Heyting__term_u2202____1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Heyting___aux__Mathlib__Order__Heyting__Boundary______macroRules__Heyting__term_u2202____1___closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Heyting___aux__Mathlib__Order__Heyting__Boundary______macroRules__Heyting__term_u2202____1___closed__10 = (const lean_object*)&lp_mathlib_Heyting___aux__Mathlib__Order__Heyting__Boundary______macroRules__Heyting__term_u2202____1___closed__10_value;
static const lean_ctor_object lp_mathlib_Heyting___aux__Mathlib__Order__Heyting__Boundary______macroRules__Heyting__term_u2202____1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Heyting___aux__Mathlib__Order__Heyting__Boundary______macroRules__Heyting__term_u2202____1___closed__10_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Heyting___aux__Mathlib__Order__Heyting__Boundary______macroRules__Heyting__term_u2202____1___closed__11 = (const lean_object*)&lp_mathlib_Heyting___aux__Mathlib__Order__Heyting__Boundary______macroRules__Heyting__term_u2202____1___closed__11_value;
static const lean_string_object lp_mathlib_Heyting___aux__Mathlib__Order__Heyting__Boundary______macroRules__Heyting__term_u2202____1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_Heyting___aux__Mathlib__Order__Heyting__Boundary______macroRules__Heyting__term_u2202____1___closed__12 = (const lean_object*)&lp_mathlib_Heyting___aux__Mathlib__Order__Heyting__Boundary______macroRules__Heyting__term_u2202____1___closed__12_value;
static const lean_ctor_object lp_mathlib_Heyting___aux__Mathlib__Order__Heyting__Boundary______macroRules__Heyting__term_u2202____1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Heyting___aux__Mathlib__Order__Heyting__Boundary______macroRules__Heyting__term_u2202____1___closed__12_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_Heyting___aux__Mathlib__Order__Heyting__Boundary______macroRules__Heyting__term_u2202____1___closed__13 = (const lean_object*)&lp_mathlib_Heyting___aux__Mathlib__Order__Heyting__Boundary______macroRules__Heyting__term_u2202____1___closed__13_value;
LEAN_EXPORT lean_object* lp_mathlib_Heyting___aux__Mathlib__Order__Heyting__Boundary______macroRules__Heyting__term_u2202____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Heyting___aux__Mathlib__Order__Heyting__Boundary______macroRules__Heyting__term_u2202____1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Heyting___aux__Mathlib__Order__Heyting__Boundary______unexpand__Coheyting__boundary__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib_Heyting___aux__Mathlib__Order__Heyting__Boundary______unexpand__Coheyting__boundary__1___closed__0 = (const lean_object*)&lp_mathlib_Heyting___aux__Mathlib__Order__Heyting__Boundary______unexpand__Coheyting__boundary__1___closed__0_value;
static const lean_ctor_object lp_mathlib_Heyting___aux__Mathlib__Order__Heyting__Boundary______unexpand__Coheyting__boundary__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Heyting___aux__Mathlib__Order__Heyting__Boundary______unexpand__Coheyting__boundary__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib_Heyting___aux__Mathlib__Order__Heyting__Boundary______unexpand__Coheyting__boundary__1___closed__1 = (const lean_object*)&lp_mathlib_Heyting___aux__Mathlib__Order__Heyting__Boundary______unexpand__Coheyting__boundary__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Heyting___aux__Mathlib__Order__Heyting__Boundary______unexpand__Coheyting__boundary__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Heyting___aux__Mathlib__Order__Heyting__Boundary______unexpand__Coheyting__boundary__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Coheyting_boundary___redArg(lean_object* v_inst_1_, lean_object* v_a_2_){
_start:
{
lean_object* v_toGeneralizedCoheytingAlgebra_3_; lean_object* v_toLattice_4_; lean_object* v_toHNot_5_; lean_object* v_inf_6_; lean_object* v___x_7_; lean_object* v___x_8_; 
v_toGeneralizedCoheytingAlgebra_3_ = lean_ctor_get(v_inst_1_, 0);
v_toLattice_4_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_3_, 0);
lean_inc_ref(v_toLattice_4_);
v_toHNot_5_ = lean_ctor_get(v_inst_1_, 2);
lean_inc(v_toHNot_5_);
lean_dec_ref(v_inst_1_);
v_inf_6_ = lean_ctor_get(v_toLattice_4_, 1);
lean_inc(v_inf_6_);
lean_dec_ref(v_toLattice_4_);
lean_inc(v_a_2_);
v___x_7_ = lean_apply_1(v_toHNot_5_, v_a_2_);
v___x_8_ = lean_apply_2(v_inf_6_, v_a_2_, v___x_7_);
return v___x_8_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Coheyting_boundary(lean_object* v_00_u03b1_9_, lean_object* v_inst_10_, lean_object* v_a_11_){
_start:
{
lean_object* v___x_12_; 
v___x_12_ = lp_mathlib_Coheyting_boundary___redArg(v_inst_10_, v_a_11_);
return v___x_12_;
}
}
static lean_object* _init_lp_mathlib_Heyting___aux__Mathlib__Order__Heyting__Boundary______macroRules__Heyting__term_u2202____1___closed__6(void){
_start:
{
lean_object* v___x_49_; lean_object* v___x_50_; 
v___x_49_ = ((lean_object*)(lp_mathlib_Heyting___aux__Mathlib__Order__Heyting__Boundary______macroRules__Heyting__term_u2202____1___closed__5));
v___x_50_ = l_String_toRawSubstring_x27(v___x_49_);
return v___x_50_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Heyting___aux__Mathlib__Order__Heyting__Boundary______macroRules__Heyting__term_u2202____1(lean_object* v_x_65_, lean_object* v_a_66_, lean_object* v_a_67_){
_start:
{
lean_object* v___x_68_; uint8_t v___x_69_; 
v___x_68_ = ((lean_object*)(lp_mathlib_Heyting_term_u2202___00__closed__2));
lean_inc(v_x_65_);
v___x_69_ = l_Lean_Syntax_isOfKind(v_x_65_, v___x_68_);
if (v___x_69_ == 0)
{
lean_object* v___x_70_; lean_object* v___x_71_; 
lean_dec(v_x_65_);
v___x_70_ = lean_box(1);
v___x_71_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_71_, 0, v___x_70_);
lean_ctor_set(v___x_71_, 1, v_a_67_);
return v___x_71_;
}
else
{
lean_object* v_quotContext_72_; lean_object* v_currMacroScope_73_; lean_object* v_ref_74_; lean_object* v___x_75_; lean_object* v___x_76_; uint8_t v___x_77_; lean_object* v___x_78_; lean_object* v___x_79_; lean_object* v___x_80_; lean_object* v___x_81_; lean_object* v___x_82_; lean_object* v___x_83_; lean_object* v___x_84_; lean_object* v___x_85_; lean_object* v___x_86_; lean_object* v___x_87_; lean_object* v___x_88_; 
v_quotContext_72_ = lean_ctor_get(v_a_66_, 1);
v_currMacroScope_73_ = lean_ctor_get(v_a_66_, 2);
v_ref_74_ = lean_ctor_get(v_a_66_, 5);
v___x_75_ = lean_unsigned_to_nat(1u);
v___x_76_ = l_Lean_Syntax_getArg(v_x_65_, v___x_75_);
lean_dec(v_x_65_);
v___x_77_ = 0;
v___x_78_ = l_Lean_SourceInfo_fromRef(v_ref_74_, v___x_77_);
v___x_79_ = ((lean_object*)(lp_mathlib_Heyting___aux__Mathlib__Order__Heyting__Boundary______macroRules__Heyting__term_u2202____1___closed__4));
v___x_80_ = lean_obj_once(&lp_mathlib_Heyting___aux__Mathlib__Order__Heyting__Boundary______macroRules__Heyting__term_u2202____1___closed__6, &lp_mathlib_Heyting___aux__Mathlib__Order__Heyting__Boundary______macroRules__Heyting__term_u2202____1___closed__6_once, _init_lp_mathlib_Heyting___aux__Mathlib__Order__Heyting__Boundary______macroRules__Heyting__term_u2202____1___closed__6);
v___x_81_ = ((lean_object*)(lp_mathlib_Heyting___aux__Mathlib__Order__Heyting__Boundary______macroRules__Heyting__term_u2202____1___closed__9));
lean_inc(v_currMacroScope_73_);
lean_inc(v_quotContext_72_);
v___x_82_ = l_Lean_addMacroScope(v_quotContext_72_, v___x_81_, v_currMacroScope_73_);
v___x_83_ = ((lean_object*)(lp_mathlib_Heyting___aux__Mathlib__Order__Heyting__Boundary______macroRules__Heyting__term_u2202____1___closed__11));
lean_inc_n(v___x_78_, 2);
v___x_84_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_84_, 0, v___x_78_);
lean_ctor_set(v___x_84_, 1, v___x_80_);
lean_ctor_set(v___x_84_, 2, v___x_82_);
lean_ctor_set(v___x_84_, 3, v___x_83_);
v___x_85_ = ((lean_object*)(lp_mathlib_Heyting___aux__Mathlib__Order__Heyting__Boundary______macroRules__Heyting__term_u2202____1___closed__13));
v___x_86_ = l_Lean_Syntax_node1(v___x_78_, v___x_85_, v___x_76_);
v___x_87_ = l_Lean_Syntax_node2(v___x_78_, v___x_79_, v___x_84_, v___x_86_);
v___x_88_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_88_, 0, v___x_87_);
lean_ctor_set(v___x_88_, 1, v_a_67_);
return v___x_88_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Heyting___aux__Mathlib__Order__Heyting__Boundary______macroRules__Heyting__term_u2202____1___boxed(lean_object* v_x_89_, lean_object* v_a_90_, lean_object* v_a_91_){
_start:
{
lean_object* v_res_92_; 
v_res_92_ = lp_mathlib_Heyting___aux__Mathlib__Order__Heyting__Boundary______macroRules__Heyting__term_u2202____1(v_x_89_, v_a_90_, v_a_91_);
lean_dec_ref(v_a_90_);
return v_res_92_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Heyting___aux__Mathlib__Order__Heyting__Boundary______unexpand__Coheyting__boundary__1(lean_object* v_x_96_, lean_object* v_a_97_, lean_object* v_a_98_){
_start:
{
lean_object* v___x_99_; uint8_t v___x_100_; 
v___x_99_ = ((lean_object*)(lp_mathlib_Heyting___aux__Mathlib__Order__Heyting__Boundary______macroRules__Heyting__term_u2202____1___closed__4));
lean_inc(v_x_96_);
v___x_100_ = l_Lean_Syntax_isOfKind(v_x_96_, v___x_99_);
if (v___x_100_ == 0)
{
lean_object* v___x_101_; lean_object* v___x_102_; 
lean_dec(v_x_96_);
v___x_101_ = lean_box(0);
v___x_102_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_102_, 0, v___x_101_);
lean_ctor_set(v___x_102_, 1, v_a_98_);
return v___x_102_;
}
else
{
lean_object* v___x_103_; lean_object* v___x_104_; lean_object* v___x_105_; uint8_t v___x_106_; 
v___x_103_ = lean_unsigned_to_nat(0u);
v___x_104_ = l_Lean_Syntax_getArg(v_x_96_, v___x_103_);
v___x_105_ = ((lean_object*)(lp_mathlib_Heyting___aux__Mathlib__Order__Heyting__Boundary______unexpand__Coheyting__boundary__1___closed__1));
lean_inc(v___x_104_);
v___x_106_ = l_Lean_Syntax_isOfKind(v___x_104_, v___x_105_);
if (v___x_106_ == 0)
{
lean_object* v___x_107_; lean_object* v___x_108_; 
lean_dec(v___x_104_);
lean_dec(v_x_96_);
v___x_107_ = lean_box(0);
v___x_108_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_108_, 0, v___x_107_);
lean_ctor_set(v___x_108_, 1, v_a_98_);
return v___x_108_;
}
else
{
lean_object* v___x_109_; lean_object* v___x_110_; uint8_t v___x_111_; 
v___x_109_ = lean_unsigned_to_nat(1u);
v___x_110_ = l_Lean_Syntax_getArg(v_x_96_, v___x_109_);
lean_dec(v_x_96_);
lean_inc(v___x_110_);
v___x_111_ = l_Lean_Syntax_matchesNull(v___x_110_, v___x_109_);
if (v___x_111_ == 0)
{
lean_object* v___x_112_; lean_object* v___x_113_; 
lean_dec(v___x_110_);
lean_dec(v___x_104_);
v___x_112_ = lean_box(0);
v___x_113_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_113_, 0, v___x_112_);
lean_ctor_set(v___x_113_, 1, v_a_98_);
return v___x_113_;
}
else
{
lean_object* v___x_114_; lean_object* v_ref_115_; uint8_t v___x_116_; lean_object* v___x_117_; lean_object* v___x_118_; lean_object* v___x_119_; lean_object* v___x_120_; lean_object* v___x_121_; lean_object* v___x_122_; 
v___x_114_ = l_Lean_Syntax_getArg(v___x_110_, v___x_103_);
lean_dec(v___x_110_);
v_ref_115_ = l_Lean_replaceRef(v___x_104_, v_a_97_);
lean_dec(v___x_104_);
v___x_116_ = 0;
v___x_117_ = l_Lean_SourceInfo_fromRef(v_ref_115_, v___x_116_);
lean_dec(v_ref_115_);
v___x_118_ = ((lean_object*)(lp_mathlib_Heyting_term_u2202___00__closed__2));
v___x_119_ = ((lean_object*)(lp_mathlib_Heyting_term_u2202___00__closed__5));
lean_inc(v___x_117_);
v___x_120_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_120_, 0, v___x_117_);
lean_ctor_set(v___x_120_, 1, v___x_119_);
v___x_121_ = l_Lean_Syntax_node2(v___x_117_, v___x_118_, v___x_120_, v___x_114_);
v___x_122_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_122_, 0, v___x_121_);
lean_ctor_set(v___x_122_, 1, v_a_98_);
return v___x_122_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Heyting___aux__Mathlib__Order__Heyting__Boundary______unexpand__Coheyting__boundary__1___boxed(lean_object* v_x_123_, lean_object* v_a_124_, lean_object* v_a_125_){
_start:
{
lean_object* v_res_126_; 
v_res_126_ = lp_mathlib_Heyting___aux__Mathlib__Order__Heyting__Boundary______unexpand__Coheyting__boundary__1(v_x_123_, v_a_124_, v_a_125_);
lean_dec(v_a_124_);
return v_res_126_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_BooleanAlgebra_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Common(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Attr_Core(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Order_Heyting_Boundary(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_BooleanAlgebra_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Common(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Attr_Core(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Order_Heyting_Boundary(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Order_BooleanAlgebra_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Common(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Attr_Core(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_Heyting_Boundary(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_BooleanAlgebra_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Common(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Attr_Core(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Heyting_Boundary(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Order_Heyting_Boundary(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Order_Heyting_Boundary(builtin);
}
#ifdef __cplusplus
}
#endif
