// Lean compiler output
// Module: Mathlib.SetTheory.Cardinal.Aleph
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Order.Monoid.Basic public import Mathlib.SetTheory.Cardinal.Cofinality.Enum public import Mathlib.SetTheory.Cardinal.ToNat public import Mathlib.SetTheory.Cardinal.ENat public import Mathlib.SetTheory.Ordinal.Enum public import Mathlib.SetTheory.Ordinal.Univ import Mathlib.SetTheory.Ordinal.Principal
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
lean_object* l_Lean_Name_mkStr1(lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Ordinal_term_u03c9___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Ordinal"};
static const lean_object* lp_mathlib_Ordinal_term_u03c9___00__closed__0 = (const lean_object*)&lp_mathlib_Ordinal_term_u03c9___00__closed__0_value;
static const lean_string_object lp_mathlib_Ordinal_term_u03c9___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 6, .m_data = "termω_"};
static const lean_object* lp_mathlib_Ordinal_term_u03c9___00__closed__1 = (const lean_object*)&lp_mathlib_Ordinal_term_u03c9___00__closed__1_value;
static const lean_ctor_object lp_mathlib_Ordinal_term_u03c9___00__closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Ordinal_term_u03c9___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(132, 160, 33, 233, 30, 237, 55, 251)}};
static const lean_ctor_object lp_mathlib_Ordinal_term_u03c9___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Ordinal_term_u03c9___00__closed__2_value_aux_0),((lean_object*)&lp_mathlib_Ordinal_term_u03c9___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(13, 158, 36, 235, 230, 32, 127, 110)}};
static const lean_object* lp_mathlib_Ordinal_term_u03c9___00__closed__2 = (const lean_object*)&lp_mathlib_Ordinal_term_u03c9___00__closed__2_value;
static const lean_string_object lp_mathlib_Ordinal_term_u03c9___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 3, .m_data = "ω_ "};
static const lean_object* lp_mathlib_Ordinal_term_u03c9___00__closed__3 = (const lean_object*)&lp_mathlib_Ordinal_term_u03c9___00__closed__3_value;
static const lean_ctor_object lp_mathlib_Ordinal_term_u03c9___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Ordinal_term_u03c9___00__closed__3_value)}};
static const lean_object* lp_mathlib_Ordinal_term_u03c9___00__closed__4 = (const lean_object*)&lp_mathlib_Ordinal_term_u03c9___00__closed__4_value;
static const lean_ctor_object lp_mathlib_Ordinal_term_u03c9___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Ordinal_term_u03c9___00__closed__2_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_Ordinal_term_u03c9___00__closed__4_value)}};
static const lean_object* lp_mathlib_Ordinal_term_u03c9___00__closed__5 = (const lean_object*)&lp_mathlib_Ordinal_term_u03c9___00__closed__5_value;
LEAN_EXPORT const lean_object* lp_mathlib_Ordinal_term_u03c9__ = (const lean_object*)&lp_mathlib_Ordinal_term_u03c9___00__closed__5_value;
static const lean_string_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Ordinal__term_u03c9____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "omega"};
static const lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Ordinal__term_u03c9____1___closed__0 = (const lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Ordinal__term_u03c9____1___closed__0_value;
static lean_once_cell_t lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Ordinal__term_u03c9____1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Ordinal__term_u03c9____1___closed__1;
static const lean_ctor_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Ordinal__term_u03c9____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Ordinal__term_u03c9____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(107, 155, 144, 136, 132, 122, 189, 157)}};
static const lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Ordinal__term_u03c9____1___closed__2 = (const lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Ordinal__term_u03c9____1___closed__2_value;
static const lean_ctor_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Ordinal__term_u03c9____1___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Ordinal_term_u03c9___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(132, 160, 33, 233, 30, 237, 55, 251)}};
static const lean_ctor_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Ordinal__term_u03c9____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Ordinal__term_u03c9____1___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Ordinal__term_u03c9____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(192, 125, 196, 147, 214, 50, 33, 114)}};
static const lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Ordinal__term_u03c9____1___closed__3 = (const lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Ordinal__term_u03c9____1___closed__3_value;
static const lean_ctor_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Ordinal__term_u03c9____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Ordinal__term_u03c9____1___closed__3_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Ordinal__term_u03c9____1___closed__4 = (const lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Ordinal__term_u03c9____1___closed__4_value;
static const lean_ctor_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Ordinal__term_u03c9____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Ordinal__term_u03c9____1___closed__4_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Ordinal__term_u03c9____1___closed__5 = (const lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Ordinal__term_u03c9____1___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Ordinal__term_u03c9____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Ordinal__term_u03c9____1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______unexpand__Ordinal__omega__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______unexpand__Ordinal__omega__1___closed__0 = (const lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______unexpand__Ordinal__omega__1___closed__0_value;
static const lean_ctor_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______unexpand__Ordinal__omega__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______unexpand__Ordinal__omega__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______unexpand__Ordinal__omega__1___closed__1 = (const lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______unexpand__Ordinal__omega__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______unexpand__Ordinal__omega__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______unexpand__Ordinal__omega__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Ordinal_term_u03c9_u2081___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 6, .m_data = "termω₁"};
static const lean_object* lp_mathlib_Ordinal_term_u03c9_u2081___closed__0 = (const lean_object*)&lp_mathlib_Ordinal_term_u03c9_u2081___closed__0_value;
static const lean_ctor_object lp_mathlib_Ordinal_term_u03c9_u2081___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Ordinal_term_u03c9___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(132, 160, 33, 233, 30, 237, 55, 251)}};
static const lean_ctor_object lp_mathlib_Ordinal_term_u03c9_u2081___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Ordinal_term_u03c9_u2081___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Ordinal_term_u03c9_u2081___closed__0_value),LEAN_SCALAR_PTR_LITERAL(235, 6, 240, 21, 167, 79, 162, 154)}};
static const lean_object* lp_mathlib_Ordinal_term_u03c9_u2081___closed__1 = (const lean_object*)&lp_mathlib_Ordinal_term_u03c9_u2081___closed__1_value;
static const lean_string_object lp_mathlib_Ordinal_term_u03c9_u2081___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 2, .m_data = "ω₁"};
static const lean_object* lp_mathlib_Ordinal_term_u03c9_u2081___closed__2 = (const lean_object*)&lp_mathlib_Ordinal_term_u03c9_u2081___closed__2_value;
static const lean_ctor_object lp_mathlib_Ordinal_term_u03c9_u2081___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Ordinal_term_u03c9_u2081___closed__2_value)}};
static const lean_object* lp_mathlib_Ordinal_term_u03c9_u2081___closed__3 = (const lean_object*)&lp_mathlib_Ordinal_term_u03c9_u2081___closed__3_value;
static const lean_ctor_object lp_mathlib_Ordinal_term_u03c9_u2081___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Ordinal_term_u03c9_u2081___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_Ordinal_term_u03c9_u2081___closed__3_value)}};
static const lean_object* lp_mathlib_Ordinal_term_u03c9_u2081___closed__4 = (const lean_object*)&lp_mathlib_Ordinal_term_u03c9_u2081___closed__4_value;
LEAN_EXPORT const lean_object* lp_mathlib_Ordinal_term_u03c9_u2081 = (const lean_object*)&lp_mathlib_Ordinal_term_u03c9_u2081___closed__4_value;
static const lean_string_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Ordinal__term_u03c9_u2081__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Ordinal__term_u03c9_u2081__1___closed__0 = (const lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Ordinal__term_u03c9_u2081__1___closed__0_value;
static const lean_string_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Ordinal__term_u03c9_u2081__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Ordinal__term_u03c9_u2081__1___closed__1 = (const lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Ordinal__term_u03c9_u2081__1___closed__1_value;
static const lean_string_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Ordinal__term_u03c9_u2081__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Ordinal__term_u03c9_u2081__1___closed__2 = (const lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Ordinal__term_u03c9_u2081__1___closed__2_value;
static const lean_string_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Ordinal__term_u03c9_u2081__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Ordinal__term_u03c9_u2081__1___closed__3 = (const lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Ordinal__term_u03c9_u2081__1___closed__3_value;
static const lean_ctor_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Ordinal__term_u03c9_u2081__1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Ordinal__term_u03c9_u2081__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Ordinal__term_u03c9_u2081__1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Ordinal__term_u03c9_u2081__1___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Ordinal__term_u03c9_u2081__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Ordinal__term_u03c9_u2081__1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Ordinal__term_u03c9_u2081__1___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Ordinal__term_u03c9_u2081__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Ordinal__term_u03c9_u2081__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Ordinal__term_u03c9_u2081__1___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Ordinal__term_u03c9_u2081__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Ordinal__term_u03c9_u2081__1___closed__4 = (const lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Ordinal__term_u03c9_u2081__1___closed__4_value;
static const lean_string_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Ordinal__term_u03c9_u2081__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 2, .m_data = "ω_"};
static const lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Ordinal__term_u03c9_u2081__1___closed__5 = (const lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Ordinal__term_u03c9_u2081__1___closed__5_value;
static const lean_string_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Ordinal__term_u03c9_u2081__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Ordinal__term_u03c9_u2081__1___closed__6 = (const lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Ordinal__term_u03c9_u2081__1___closed__6_value;
static const lean_ctor_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Ordinal__term_u03c9_u2081__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Ordinal__term_u03c9_u2081__1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Ordinal__term_u03c9_u2081__1___closed__7 = (const lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Ordinal__term_u03c9_u2081__1___closed__7_value;
static const lean_string_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Ordinal__term_u03c9_u2081__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "num"};
static const lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Ordinal__term_u03c9_u2081__1___closed__8 = (const lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Ordinal__term_u03c9_u2081__1___closed__8_value;
static const lean_ctor_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Ordinal__term_u03c9_u2081__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Ordinal__term_u03c9_u2081__1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(227, 68, 22, 222, 47, 51, 204, 84)}};
static const lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Ordinal__term_u03c9_u2081__1___closed__9 = (const lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Ordinal__term_u03c9_u2081__1___closed__9_value;
static const lean_string_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Ordinal__term_u03c9_u2081__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "1"};
static const lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Ordinal__term_u03c9_u2081__1___closed__10 = (const lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Ordinal__term_u03c9_u2081__1___closed__10_value;
LEAN_EXPORT lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Ordinal__term_u03c9_u2081__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Ordinal__term_u03c9_u2081__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Cardinal_term_u2135___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Cardinal"};
static const lean_object* lp_mathlib_Cardinal_term_u2135___00__closed__0 = (const lean_object*)&lp_mathlib_Cardinal_term_u2135___00__closed__0_value;
static const lean_string_object lp_mathlib_Cardinal_term_u2135___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 6, .m_data = "termℵ_"};
static const lean_object* lp_mathlib_Cardinal_term_u2135___00__closed__1 = (const lean_object*)&lp_mathlib_Cardinal_term_u2135___00__closed__1_value;
static const lean_ctor_object lp_mathlib_Cardinal_term_u2135___00__closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Cardinal_term_u2135___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(247, 176, 10, 15, 146, 217, 130, 31)}};
static const lean_ctor_object lp_mathlib_Cardinal_term_u2135___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Cardinal_term_u2135___00__closed__2_value_aux_0),((lean_object*)&lp_mathlib_Cardinal_term_u2135___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(116, 204, 196, 251, 7, 79, 145, 233)}};
static const lean_object* lp_mathlib_Cardinal_term_u2135___00__closed__2 = (const lean_object*)&lp_mathlib_Cardinal_term_u2135___00__closed__2_value;
static const lean_string_object lp_mathlib_Cardinal_term_u2135___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 3, .m_data = "ℵ_ "};
static const lean_object* lp_mathlib_Cardinal_term_u2135___00__closed__3 = (const lean_object*)&lp_mathlib_Cardinal_term_u2135___00__closed__3_value;
static const lean_ctor_object lp_mathlib_Cardinal_term_u2135___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Cardinal_term_u2135___00__closed__3_value)}};
static const lean_object* lp_mathlib_Cardinal_term_u2135___00__closed__4 = (const lean_object*)&lp_mathlib_Cardinal_term_u2135___00__closed__4_value;
static const lean_ctor_object lp_mathlib_Cardinal_term_u2135___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Cardinal_term_u2135___00__closed__2_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_Cardinal_term_u2135___00__closed__4_value)}};
static const lean_object* lp_mathlib_Cardinal_term_u2135___00__closed__5 = (const lean_object*)&lp_mathlib_Cardinal_term_u2135___00__closed__5_value;
LEAN_EXPORT const lean_object* lp_mathlib_Cardinal_term_u2135__ = (const lean_object*)&lp_mathlib_Cardinal_term_u2135___00__closed__5_value;
static const lean_string_object lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Cardinal__term_u2135____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "aleph"};
static const lean_object* lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Cardinal__term_u2135____1___closed__0 = (const lean_object*)&lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Cardinal__term_u2135____1___closed__0_value;
static lean_once_cell_t lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Cardinal__term_u2135____1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Cardinal__term_u2135____1___closed__1;
static const lean_ctor_object lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Cardinal__term_u2135____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Cardinal__term_u2135____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(19, 193, 45, 177, 210, 226, 26, 37)}};
static const lean_object* lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Cardinal__term_u2135____1___closed__2 = (const lean_object*)&lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Cardinal__term_u2135____1___closed__2_value;
static const lean_ctor_object lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Cardinal__term_u2135____1___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Cardinal_term_u2135___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(247, 176, 10, 15, 146, 217, 130, 31)}};
static const lean_ctor_object lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Cardinal__term_u2135____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Cardinal__term_u2135____1___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Cardinal__term_u2135____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(167, 126, 37, 60, 171, 46, 31, 24)}};
static const lean_object* lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Cardinal__term_u2135____1___closed__3 = (const lean_object*)&lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Cardinal__term_u2135____1___closed__3_value;
static const lean_ctor_object lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Cardinal__term_u2135____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Cardinal__term_u2135____1___closed__3_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Cardinal__term_u2135____1___closed__4 = (const lean_object*)&lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Cardinal__term_u2135____1___closed__4_value;
static const lean_ctor_object lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Cardinal__term_u2135____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Cardinal__term_u2135____1___closed__4_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Cardinal__term_u2135____1___closed__5 = (const lean_object*)&lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Cardinal__term_u2135____1___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Cardinal__term_u2135____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Cardinal__term_u2135____1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Aleph______unexpand__Cardinal__aleph__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Aleph______unexpand__Cardinal__aleph__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Cardinal_term_u2135_u2081___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 6, .m_data = "termℵ₁"};
static const lean_object* lp_mathlib_Cardinal_term_u2135_u2081___closed__0 = (const lean_object*)&lp_mathlib_Cardinal_term_u2135_u2081___closed__0_value;
static const lean_ctor_object lp_mathlib_Cardinal_term_u2135_u2081___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Cardinal_term_u2135___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(247, 176, 10, 15, 146, 217, 130, 31)}};
static const lean_ctor_object lp_mathlib_Cardinal_term_u2135_u2081___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Cardinal_term_u2135_u2081___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Cardinal_term_u2135_u2081___closed__0_value),LEAN_SCALAR_PTR_LITERAL(94, 42, 113, 189, 138, 193, 166, 40)}};
static const lean_object* lp_mathlib_Cardinal_term_u2135_u2081___closed__1 = (const lean_object*)&lp_mathlib_Cardinal_term_u2135_u2081___closed__1_value;
static const lean_string_object lp_mathlib_Cardinal_term_u2135_u2081___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 2, .m_data = "ℵ₁"};
static const lean_object* lp_mathlib_Cardinal_term_u2135_u2081___closed__2 = (const lean_object*)&lp_mathlib_Cardinal_term_u2135_u2081___closed__2_value;
static const lean_ctor_object lp_mathlib_Cardinal_term_u2135_u2081___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Cardinal_term_u2135_u2081___closed__2_value)}};
static const lean_object* lp_mathlib_Cardinal_term_u2135_u2081___closed__3 = (const lean_object*)&lp_mathlib_Cardinal_term_u2135_u2081___closed__3_value;
static const lean_ctor_object lp_mathlib_Cardinal_term_u2135_u2081___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Cardinal_term_u2135_u2081___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_Cardinal_term_u2135_u2081___closed__3_value)}};
static const lean_object* lp_mathlib_Cardinal_term_u2135_u2081___closed__4 = (const lean_object*)&lp_mathlib_Cardinal_term_u2135_u2081___closed__4_value;
LEAN_EXPORT const lean_object* lp_mathlib_Cardinal_term_u2135_u2081 = (const lean_object*)&lp_mathlib_Cardinal_term_u2135_u2081___closed__4_value;
static const lean_string_object lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Cardinal__term_u2135_u2081__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 2, .m_data = "ℵ_"};
static const lean_object* lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Cardinal__term_u2135_u2081__1___closed__0 = (const lean_object*)&lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Cardinal__term_u2135_u2081__1___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Cardinal__term_u2135_u2081__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Cardinal__term_u2135_u2081__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Cardinal_term_u2136___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 6, .m_data = "termℶ_"};
static const lean_object* lp_mathlib_Cardinal_term_u2136___00__closed__0 = (const lean_object*)&lp_mathlib_Cardinal_term_u2136___00__closed__0_value;
static const lean_ctor_object lp_mathlib_Cardinal_term_u2136___00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Cardinal_term_u2135___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(247, 176, 10, 15, 146, 217, 130, 31)}};
static const lean_ctor_object lp_mathlib_Cardinal_term_u2136___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Cardinal_term_u2136___00__closed__1_value_aux_0),((lean_object*)&lp_mathlib_Cardinal_term_u2136___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(42, 184, 73, 147, 160, 173, 76, 213)}};
static const lean_object* lp_mathlib_Cardinal_term_u2136___00__closed__1 = (const lean_object*)&lp_mathlib_Cardinal_term_u2136___00__closed__1_value;
static const lean_string_object lp_mathlib_Cardinal_term_u2136___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 3, .m_data = "ℶ_ "};
static const lean_object* lp_mathlib_Cardinal_term_u2136___00__closed__2 = (const lean_object*)&lp_mathlib_Cardinal_term_u2136___00__closed__2_value;
static const lean_ctor_object lp_mathlib_Cardinal_term_u2136___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Cardinal_term_u2136___00__closed__2_value)}};
static const lean_object* lp_mathlib_Cardinal_term_u2136___00__closed__3 = (const lean_object*)&lp_mathlib_Cardinal_term_u2136___00__closed__3_value;
static const lean_ctor_object lp_mathlib_Cardinal_term_u2136___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Cardinal_term_u2136___00__closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_Cardinal_term_u2136___00__closed__3_value)}};
static const lean_object* lp_mathlib_Cardinal_term_u2136___00__closed__4 = (const lean_object*)&lp_mathlib_Cardinal_term_u2136___00__closed__4_value;
LEAN_EXPORT const lean_object* lp_mathlib_Cardinal_term_u2136__ = (const lean_object*)&lp_mathlib_Cardinal_term_u2136___00__closed__4_value;
static const lean_string_object lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Cardinal__term_u2136____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "beth"};
static const lean_object* lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Cardinal__term_u2136____1___closed__0 = (const lean_object*)&lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Cardinal__term_u2136____1___closed__0_value;
static lean_once_cell_t lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Cardinal__term_u2136____1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Cardinal__term_u2136____1___closed__1;
static const lean_ctor_object lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Cardinal__term_u2136____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Cardinal__term_u2136____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(123, 124, 68, 125, 230, 193, 82, 24)}};
static const lean_object* lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Cardinal__term_u2136____1___closed__2 = (const lean_object*)&lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Cardinal__term_u2136____1___closed__2_value;
static const lean_ctor_object lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Cardinal__term_u2136____1___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Cardinal_term_u2135___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(247, 176, 10, 15, 146, 217, 130, 31)}};
static const lean_ctor_object lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Cardinal__term_u2136____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Cardinal__term_u2136____1___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Cardinal__term_u2136____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(63, 210, 218, 195, 17, 89, 187, 225)}};
static const lean_object* lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Cardinal__term_u2136____1___closed__3 = (const lean_object*)&lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Cardinal__term_u2136____1___closed__3_value;
static const lean_ctor_object lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Cardinal__term_u2136____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Cardinal__term_u2136____1___closed__3_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Cardinal__term_u2136____1___closed__4 = (const lean_object*)&lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Cardinal__term_u2136____1___closed__4_value;
static const lean_ctor_object lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Cardinal__term_u2136____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Cardinal__term_u2136____1___closed__4_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Cardinal__term_u2136____1___closed__5 = (const lean_object*)&lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Cardinal__term_u2136____1___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Cardinal__term_u2136____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Cardinal__term_u2136____1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Aleph______unexpand__Cardinal__beth__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Aleph______unexpand__Cardinal__beth__1___boxed(lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Ordinal__term_u03c9____1___closed__1(void){
_start:
{
lean_object* v___x_15_; lean_object* v___x_16_; 
v___x_15_ = ((lean_object*)(lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Ordinal__term_u03c9____1___closed__0));
v___x_16_ = l_String_toRawSubstring_x27(v___x_15_);
return v___x_16_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Ordinal__term_u03c9____1(lean_object* v_x_28_, lean_object* v_a_29_, lean_object* v_a_30_){
_start:
{
lean_object* v___x_31_; uint8_t v___x_32_; 
v___x_31_ = ((lean_object*)(lp_mathlib_Ordinal_term_u03c9___00__closed__2));
v___x_32_ = l_Lean_Syntax_isOfKind(v_x_28_, v___x_31_);
if (v___x_32_ == 0)
{
lean_object* v___x_33_; lean_object* v___x_34_; 
v___x_33_ = lean_box(1);
v___x_34_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_34_, 0, v___x_33_);
lean_ctor_set(v___x_34_, 1, v_a_30_);
return v___x_34_;
}
else
{
lean_object* v_quotContext_35_; lean_object* v_currMacroScope_36_; lean_object* v_ref_37_; uint8_t v___x_38_; lean_object* v___x_39_; lean_object* v___x_40_; lean_object* v___x_41_; lean_object* v___x_42_; lean_object* v___x_43_; lean_object* v___x_44_; lean_object* v___x_45_; 
v_quotContext_35_ = lean_ctor_get(v_a_29_, 1);
v_currMacroScope_36_ = lean_ctor_get(v_a_29_, 2);
v_ref_37_ = lean_ctor_get(v_a_29_, 5);
v___x_38_ = 0;
v___x_39_ = l_Lean_SourceInfo_fromRef(v_ref_37_, v___x_38_);
v___x_40_ = lean_obj_once(&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Ordinal__term_u03c9____1___closed__1, &lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Ordinal__term_u03c9____1___closed__1_once, _init_lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Ordinal__term_u03c9____1___closed__1);
v___x_41_ = ((lean_object*)(lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Ordinal__term_u03c9____1___closed__2));
lean_inc(v_currMacroScope_36_);
lean_inc(v_quotContext_35_);
v___x_42_ = l_Lean_addMacroScope(v_quotContext_35_, v___x_41_, v_currMacroScope_36_);
v___x_43_ = ((lean_object*)(lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Ordinal__term_u03c9____1___closed__5));
v___x_44_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_44_, 0, v___x_39_);
lean_ctor_set(v___x_44_, 1, v___x_40_);
lean_ctor_set(v___x_44_, 2, v___x_42_);
lean_ctor_set(v___x_44_, 3, v___x_43_);
v___x_45_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_45_, 0, v___x_44_);
lean_ctor_set(v___x_45_, 1, v_a_30_);
return v___x_45_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Ordinal__term_u03c9____1___boxed(lean_object* v_x_46_, lean_object* v_a_47_, lean_object* v_a_48_){
_start:
{
lean_object* v_res_49_; 
v_res_49_ = lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Ordinal__term_u03c9____1(v_x_46_, v_a_47_, v_a_48_);
lean_dec_ref(v_a_47_);
return v_res_49_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______unexpand__Ordinal__omega__1(lean_object* v_x_53_, lean_object* v_a_54_, lean_object* v_a_55_){
_start:
{
lean_object* v___x_56_; uint8_t v___x_57_; 
v___x_56_ = ((lean_object*)(lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______unexpand__Ordinal__omega__1___closed__1));
lean_inc(v_x_53_);
v___x_57_ = l_Lean_Syntax_isOfKind(v_x_53_, v___x_56_);
if (v___x_57_ == 0)
{
lean_object* v___x_58_; lean_object* v___x_59_; 
lean_dec(v_x_53_);
v___x_58_ = lean_box(0);
v___x_59_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_59_, 0, v___x_58_);
lean_ctor_set(v___x_59_, 1, v_a_55_);
return v___x_59_;
}
else
{
lean_object* v_ref_60_; uint8_t v___x_61_; lean_object* v___x_62_; lean_object* v___x_63_; lean_object* v___x_64_; lean_object* v___x_65_; lean_object* v___x_66_; lean_object* v___x_67_; 
v_ref_60_ = l_Lean_replaceRef(v_x_53_, v_a_54_);
lean_dec(v_x_53_);
v___x_61_ = 0;
v___x_62_ = l_Lean_SourceInfo_fromRef(v_ref_60_, v___x_61_);
lean_dec(v_ref_60_);
v___x_63_ = ((lean_object*)(lp_mathlib_Ordinal_term_u03c9___00__closed__2));
v___x_64_ = ((lean_object*)(lp_mathlib_Ordinal_term_u03c9___00__closed__3));
lean_inc(v___x_62_);
v___x_65_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_65_, 0, v___x_62_);
lean_ctor_set(v___x_65_, 1, v___x_64_);
v___x_66_ = l_Lean_Syntax_node1(v___x_62_, v___x_63_, v___x_65_);
v___x_67_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_67_, 0, v___x_66_);
lean_ctor_set(v___x_67_, 1, v_a_55_);
return v___x_67_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______unexpand__Ordinal__omega__1___boxed(lean_object* v_x_68_, lean_object* v_a_69_, lean_object* v_a_70_){
_start:
{
lean_object* v_res_71_; 
v_res_71_ = lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______unexpand__Ordinal__omega__1(v_x_68_, v_a_69_, v_a_70_);
lean_dec(v_a_69_);
return v_res_71_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Ordinal__term_u03c9_u2081__1(lean_object* v_x_101_, lean_object* v_a_102_, lean_object* v_a_103_){
_start:
{
lean_object* v___x_104_; uint8_t v___x_105_; 
v___x_104_ = ((lean_object*)(lp_mathlib_Ordinal_term_u03c9_u2081___closed__1));
v___x_105_ = l_Lean_Syntax_isOfKind(v_x_101_, v___x_104_);
if (v___x_105_ == 0)
{
lean_object* v___x_106_; lean_object* v___x_107_; 
v___x_106_ = lean_box(1);
v___x_107_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_107_, 0, v___x_106_);
lean_ctor_set(v___x_107_, 1, v_a_103_);
return v___x_107_;
}
else
{
lean_object* v_ref_108_; uint8_t v___x_109_; lean_object* v___x_110_; lean_object* v___x_111_; lean_object* v___x_112_; lean_object* v___x_113_; lean_object* v___x_114_; lean_object* v___x_115_; lean_object* v___x_116_; lean_object* v___x_117_; lean_object* v___x_118_; lean_object* v___x_119_; lean_object* v___x_120_; lean_object* v___x_121_; lean_object* v___x_122_; lean_object* v___x_123_; 
v_ref_108_ = lean_ctor_get(v_a_102_, 5);
v___x_109_ = 0;
v___x_110_ = l_Lean_SourceInfo_fromRef(v_ref_108_, v___x_109_);
v___x_111_ = ((lean_object*)(lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Ordinal__term_u03c9_u2081__1___closed__4));
v___x_112_ = ((lean_object*)(lp_mathlib_Ordinal_term_u03c9___00__closed__2));
v___x_113_ = ((lean_object*)(lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Ordinal__term_u03c9_u2081__1___closed__5));
lean_inc_n(v___x_110_, 5);
v___x_114_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_114_, 0, v___x_110_);
lean_ctor_set(v___x_114_, 1, v___x_113_);
v___x_115_ = l_Lean_Syntax_node1(v___x_110_, v___x_112_, v___x_114_);
v___x_116_ = ((lean_object*)(lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Ordinal__term_u03c9_u2081__1___closed__7));
v___x_117_ = ((lean_object*)(lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Ordinal__term_u03c9_u2081__1___closed__9));
v___x_118_ = ((lean_object*)(lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Ordinal__term_u03c9_u2081__1___closed__10));
v___x_119_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_119_, 0, v___x_110_);
lean_ctor_set(v___x_119_, 1, v___x_118_);
v___x_120_ = l_Lean_Syntax_node1(v___x_110_, v___x_117_, v___x_119_);
v___x_121_ = l_Lean_Syntax_node1(v___x_110_, v___x_116_, v___x_120_);
v___x_122_ = l_Lean_Syntax_node2(v___x_110_, v___x_111_, v___x_115_, v___x_121_);
v___x_123_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_123_, 0, v___x_122_);
lean_ctor_set(v___x_123_, 1, v_a_103_);
return v___x_123_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Ordinal__term_u03c9_u2081__1___boxed(lean_object* v_x_124_, lean_object* v_a_125_, lean_object* v_a_126_){
_start:
{
lean_object* v_res_127_; 
v_res_127_ = lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Ordinal__term_u03c9_u2081__1(v_x_124_, v_a_125_, v_a_126_);
lean_dec_ref(v_a_125_);
return v_res_127_;
}
}
static lean_object* _init_lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Cardinal__term_u2135____1___closed__1(void){
_start:
{
lean_object* v___x_142_; lean_object* v___x_143_; 
v___x_142_ = ((lean_object*)(lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Cardinal__term_u2135____1___closed__0));
v___x_143_ = l_String_toRawSubstring_x27(v___x_142_);
return v___x_143_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Cardinal__term_u2135____1(lean_object* v_x_155_, lean_object* v_a_156_, lean_object* v_a_157_){
_start:
{
lean_object* v___x_158_; uint8_t v___x_159_; 
v___x_158_ = ((lean_object*)(lp_mathlib_Cardinal_term_u2135___00__closed__2));
v___x_159_ = l_Lean_Syntax_isOfKind(v_x_155_, v___x_158_);
if (v___x_159_ == 0)
{
lean_object* v___x_160_; lean_object* v___x_161_; 
v___x_160_ = lean_box(1);
v___x_161_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_161_, 0, v___x_160_);
lean_ctor_set(v___x_161_, 1, v_a_157_);
return v___x_161_;
}
else
{
lean_object* v_quotContext_162_; lean_object* v_currMacroScope_163_; lean_object* v_ref_164_; uint8_t v___x_165_; lean_object* v___x_166_; lean_object* v___x_167_; lean_object* v___x_168_; lean_object* v___x_169_; lean_object* v___x_170_; lean_object* v___x_171_; lean_object* v___x_172_; 
v_quotContext_162_ = lean_ctor_get(v_a_156_, 1);
v_currMacroScope_163_ = lean_ctor_get(v_a_156_, 2);
v_ref_164_ = lean_ctor_get(v_a_156_, 5);
v___x_165_ = 0;
v___x_166_ = l_Lean_SourceInfo_fromRef(v_ref_164_, v___x_165_);
v___x_167_ = lean_obj_once(&lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Cardinal__term_u2135____1___closed__1, &lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Cardinal__term_u2135____1___closed__1_once, _init_lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Cardinal__term_u2135____1___closed__1);
v___x_168_ = ((lean_object*)(lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Cardinal__term_u2135____1___closed__2));
lean_inc(v_currMacroScope_163_);
lean_inc(v_quotContext_162_);
v___x_169_ = l_Lean_addMacroScope(v_quotContext_162_, v___x_168_, v_currMacroScope_163_);
v___x_170_ = ((lean_object*)(lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Cardinal__term_u2135____1___closed__5));
v___x_171_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_171_, 0, v___x_166_);
lean_ctor_set(v___x_171_, 1, v___x_167_);
lean_ctor_set(v___x_171_, 2, v___x_169_);
lean_ctor_set(v___x_171_, 3, v___x_170_);
v___x_172_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_172_, 0, v___x_171_);
lean_ctor_set(v___x_172_, 1, v_a_157_);
return v___x_172_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Cardinal__term_u2135____1___boxed(lean_object* v_x_173_, lean_object* v_a_174_, lean_object* v_a_175_){
_start:
{
lean_object* v_res_176_; 
v_res_176_ = lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Cardinal__term_u2135____1(v_x_173_, v_a_174_, v_a_175_);
lean_dec_ref(v_a_174_);
return v_res_176_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Aleph______unexpand__Cardinal__aleph__1(lean_object* v_x_177_, lean_object* v_a_178_, lean_object* v_a_179_){
_start:
{
lean_object* v___x_180_; uint8_t v___x_181_; 
v___x_180_ = ((lean_object*)(lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______unexpand__Ordinal__omega__1___closed__1));
lean_inc(v_x_177_);
v___x_181_ = l_Lean_Syntax_isOfKind(v_x_177_, v___x_180_);
if (v___x_181_ == 0)
{
lean_object* v___x_182_; lean_object* v___x_183_; 
lean_dec(v_x_177_);
v___x_182_ = lean_box(0);
v___x_183_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_183_, 0, v___x_182_);
lean_ctor_set(v___x_183_, 1, v_a_179_);
return v___x_183_;
}
else
{
lean_object* v_ref_184_; uint8_t v___x_185_; lean_object* v___x_186_; lean_object* v___x_187_; lean_object* v___x_188_; lean_object* v___x_189_; lean_object* v___x_190_; lean_object* v___x_191_; 
v_ref_184_ = l_Lean_replaceRef(v_x_177_, v_a_178_);
lean_dec(v_x_177_);
v___x_185_ = 0;
v___x_186_ = l_Lean_SourceInfo_fromRef(v_ref_184_, v___x_185_);
lean_dec(v_ref_184_);
v___x_187_ = ((lean_object*)(lp_mathlib_Cardinal_term_u2135___00__closed__2));
v___x_188_ = ((lean_object*)(lp_mathlib_Cardinal_term_u2135___00__closed__3));
lean_inc(v___x_186_);
v___x_189_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_189_, 0, v___x_186_);
lean_ctor_set(v___x_189_, 1, v___x_188_);
v___x_190_ = l_Lean_Syntax_node1(v___x_186_, v___x_187_, v___x_189_);
v___x_191_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_191_, 0, v___x_190_);
lean_ctor_set(v___x_191_, 1, v_a_179_);
return v___x_191_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Aleph______unexpand__Cardinal__aleph__1___boxed(lean_object* v_x_192_, lean_object* v_a_193_, lean_object* v_a_194_){
_start:
{
lean_object* v_res_195_; 
v_res_195_ = lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Aleph______unexpand__Cardinal__aleph__1(v_x_192_, v_a_193_, v_a_194_);
lean_dec(v_a_193_);
return v_res_195_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Cardinal__term_u2135_u2081__1(lean_object* v_x_209_, lean_object* v_a_210_, lean_object* v_a_211_){
_start:
{
lean_object* v___x_212_; uint8_t v___x_213_; 
v___x_212_ = ((lean_object*)(lp_mathlib_Cardinal_term_u2135_u2081___closed__1));
v___x_213_ = l_Lean_Syntax_isOfKind(v_x_209_, v___x_212_);
if (v___x_213_ == 0)
{
lean_object* v___x_214_; lean_object* v___x_215_; 
v___x_214_ = lean_box(1);
v___x_215_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_215_, 0, v___x_214_);
lean_ctor_set(v___x_215_, 1, v_a_211_);
return v___x_215_;
}
else
{
lean_object* v_ref_216_; uint8_t v___x_217_; lean_object* v___x_218_; lean_object* v___x_219_; lean_object* v___x_220_; lean_object* v___x_221_; lean_object* v___x_222_; lean_object* v___x_223_; lean_object* v___x_224_; lean_object* v___x_225_; lean_object* v___x_226_; lean_object* v___x_227_; lean_object* v___x_228_; lean_object* v___x_229_; lean_object* v___x_230_; lean_object* v___x_231_; 
v_ref_216_ = lean_ctor_get(v_a_210_, 5);
v___x_217_ = 0;
v___x_218_ = l_Lean_SourceInfo_fromRef(v_ref_216_, v___x_217_);
v___x_219_ = ((lean_object*)(lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Ordinal__term_u03c9_u2081__1___closed__4));
v___x_220_ = ((lean_object*)(lp_mathlib_Cardinal_term_u2135___00__closed__2));
v___x_221_ = ((lean_object*)(lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Cardinal__term_u2135_u2081__1___closed__0));
lean_inc_n(v___x_218_, 5);
v___x_222_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_222_, 0, v___x_218_);
lean_ctor_set(v___x_222_, 1, v___x_221_);
v___x_223_ = l_Lean_Syntax_node1(v___x_218_, v___x_220_, v___x_222_);
v___x_224_ = ((lean_object*)(lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Ordinal__term_u03c9_u2081__1___closed__7));
v___x_225_ = ((lean_object*)(lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Ordinal__term_u03c9_u2081__1___closed__9));
v___x_226_ = ((lean_object*)(lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Ordinal__term_u03c9_u2081__1___closed__10));
v___x_227_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_227_, 0, v___x_218_);
lean_ctor_set(v___x_227_, 1, v___x_226_);
v___x_228_ = l_Lean_Syntax_node1(v___x_218_, v___x_225_, v___x_227_);
v___x_229_ = l_Lean_Syntax_node1(v___x_218_, v___x_224_, v___x_228_);
v___x_230_ = l_Lean_Syntax_node2(v___x_218_, v___x_219_, v___x_223_, v___x_229_);
v___x_231_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_231_, 0, v___x_230_);
lean_ctor_set(v___x_231_, 1, v_a_211_);
return v___x_231_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Cardinal__term_u2135_u2081__1___boxed(lean_object* v_x_232_, lean_object* v_a_233_, lean_object* v_a_234_){
_start:
{
lean_object* v_res_235_; 
v_res_235_ = lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Cardinal__term_u2135_u2081__1(v_x_232_, v_a_233_, v_a_234_);
lean_dec_ref(v_a_233_);
return v_res_235_;
}
}
static lean_object* _init_lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Cardinal__term_u2136____1___closed__1(void){
_start:
{
lean_object* v___x_249_; lean_object* v___x_250_; 
v___x_249_ = ((lean_object*)(lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Cardinal__term_u2136____1___closed__0));
v___x_250_ = l_String_toRawSubstring_x27(v___x_249_);
return v___x_250_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Cardinal__term_u2136____1(lean_object* v_x_262_, lean_object* v_a_263_, lean_object* v_a_264_){
_start:
{
lean_object* v___x_265_; uint8_t v___x_266_; 
v___x_265_ = ((lean_object*)(lp_mathlib_Cardinal_term_u2136___00__closed__1));
v___x_266_ = l_Lean_Syntax_isOfKind(v_x_262_, v___x_265_);
if (v___x_266_ == 0)
{
lean_object* v___x_267_; lean_object* v___x_268_; 
v___x_267_ = lean_box(1);
v___x_268_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_268_, 0, v___x_267_);
lean_ctor_set(v___x_268_, 1, v_a_264_);
return v___x_268_;
}
else
{
lean_object* v_quotContext_269_; lean_object* v_currMacroScope_270_; lean_object* v_ref_271_; uint8_t v___x_272_; lean_object* v___x_273_; lean_object* v___x_274_; lean_object* v___x_275_; lean_object* v___x_276_; lean_object* v___x_277_; lean_object* v___x_278_; lean_object* v___x_279_; 
v_quotContext_269_ = lean_ctor_get(v_a_263_, 1);
v_currMacroScope_270_ = lean_ctor_get(v_a_263_, 2);
v_ref_271_ = lean_ctor_get(v_a_263_, 5);
v___x_272_ = 0;
v___x_273_ = l_Lean_SourceInfo_fromRef(v_ref_271_, v___x_272_);
v___x_274_ = lean_obj_once(&lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Cardinal__term_u2136____1___closed__1, &lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Cardinal__term_u2136____1___closed__1_once, _init_lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Cardinal__term_u2136____1___closed__1);
v___x_275_ = ((lean_object*)(lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Cardinal__term_u2136____1___closed__2));
lean_inc(v_currMacroScope_270_);
lean_inc(v_quotContext_269_);
v___x_276_ = l_Lean_addMacroScope(v_quotContext_269_, v___x_275_, v_currMacroScope_270_);
v___x_277_ = ((lean_object*)(lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Cardinal__term_u2136____1___closed__5));
v___x_278_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_278_, 0, v___x_273_);
lean_ctor_set(v___x_278_, 1, v___x_274_);
lean_ctor_set(v___x_278_, 2, v___x_276_);
lean_ctor_set(v___x_278_, 3, v___x_277_);
v___x_279_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_279_, 0, v___x_278_);
lean_ctor_set(v___x_279_, 1, v_a_264_);
return v___x_279_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Cardinal__term_u2136____1___boxed(lean_object* v_x_280_, lean_object* v_a_281_, lean_object* v_a_282_){
_start:
{
lean_object* v_res_283_; 
v_res_283_ = lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Aleph______macroRules__Cardinal__term_u2136____1(v_x_280_, v_a_281_, v_a_282_);
lean_dec_ref(v_a_281_);
return v_res_283_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Aleph______unexpand__Cardinal__beth__1(lean_object* v_x_284_, lean_object* v_a_285_, lean_object* v_a_286_){
_start:
{
lean_object* v___x_287_; uint8_t v___x_288_; 
v___x_287_ = ((lean_object*)(lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Cardinal__Aleph______unexpand__Ordinal__omega__1___closed__1));
lean_inc(v_x_284_);
v___x_288_ = l_Lean_Syntax_isOfKind(v_x_284_, v___x_287_);
if (v___x_288_ == 0)
{
lean_object* v___x_289_; lean_object* v___x_290_; 
lean_dec(v_x_284_);
v___x_289_ = lean_box(0);
v___x_290_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_290_, 0, v___x_289_);
lean_ctor_set(v___x_290_, 1, v_a_286_);
return v___x_290_;
}
else
{
lean_object* v_ref_291_; uint8_t v___x_292_; lean_object* v___x_293_; lean_object* v___x_294_; lean_object* v___x_295_; lean_object* v___x_296_; lean_object* v___x_297_; lean_object* v___x_298_; 
v_ref_291_ = l_Lean_replaceRef(v_x_284_, v_a_285_);
lean_dec(v_x_284_);
v___x_292_ = 0;
v___x_293_ = l_Lean_SourceInfo_fromRef(v_ref_291_, v___x_292_);
lean_dec(v_ref_291_);
v___x_294_ = ((lean_object*)(lp_mathlib_Cardinal_term_u2136___00__closed__1));
v___x_295_ = ((lean_object*)(lp_mathlib_Cardinal_term_u2136___00__closed__2));
lean_inc(v___x_293_);
v___x_296_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_296_, 0, v___x_293_);
lean_ctor_set(v___x_296_, 1, v___x_295_);
v___x_297_ = l_Lean_Syntax_node1(v___x_293_, v___x_294_, v___x_296_);
v___x_298_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_298_, 0, v___x_297_);
lean_ctor_set(v___x_298_, 1, v_a_286_);
return v___x_298_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Aleph______unexpand__Cardinal__beth__1___boxed(lean_object* v_x_299_, lean_object* v_a_300_, lean_object* v_a_301_){
_start:
{
lean_object* v_res_302_; 
v_res_302_ = lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Aleph______unexpand__Cardinal__beth__1(v_x_299_, v_a_300_, v_a_301_);
lean_dec(v_a_300_);
return v_res_302_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Monoid_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_SetTheory_Cardinal_Cofinality_Enum(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_SetTheory_Cardinal_ToNat(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_SetTheory_Cardinal_ENat(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_SetTheory_Ordinal_Enum(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_SetTheory_Ordinal_Univ(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_SetTheory_Ordinal_Principal(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_SetTheory_Cardinal_Aleph(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Monoid_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_SetTheory_Cardinal_Cofinality_Enum(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_SetTheory_Cardinal_ToNat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_SetTheory_Cardinal_ENat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_SetTheory_Ordinal_Enum(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_SetTheory_Ordinal_Univ(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_SetTheory_Ordinal_Principal(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_SetTheory_Cardinal_Aleph(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Monoid_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_SetTheory_Cardinal_Cofinality_Enum(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_SetTheory_Cardinal_ToNat(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_SetTheory_Cardinal_ENat(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_SetTheory_Ordinal_Enum(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_SetTheory_Ordinal_Univ(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_SetTheory_Ordinal_Principal(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_SetTheory_Cardinal_Aleph(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Monoid_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_SetTheory_Cardinal_Cofinality_Enum(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_SetTheory_Cardinal_ToNat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_SetTheory_Cardinal_ENat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_SetTheory_Ordinal_Enum(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_SetTheory_Ordinal_Univ(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_SetTheory_Ordinal_Principal(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_SetTheory_Cardinal_Aleph(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_SetTheory_Cardinal_Aleph(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_SetTheory_Cardinal_Aleph(builtin);
}
#ifdef __cplusplus
}
#endif
