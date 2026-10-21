// Lean compiler output
// Module: Mathlib.Algebra.Algebra.Hom
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Algebra.Basic
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
lean_object* lp_mathlib_Units_map___redArg___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_RingHom_comp___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_NonUnitalRingHom_id___lam__0___boxed(lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_AddZeroClass_toAddZero___redArg(lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_npowBinRecAuto___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_term___u2192_u2090___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 8, .m_data = "term_→ₐ_"};
static const lean_object* lp_mathlib_term___u2192_u2090___00__closed__0 = (const lean_object*)&lp_mathlib_term___u2192_u2090___00__closed__0_value;
static const lean_ctor_object lp_mathlib_term___u2192_u2090___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2192_u2090___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(192, 202, 222, 116, 191, 17, 67, 135)}};
static const lean_object* lp_mathlib_term___u2192_u2090___00__closed__1 = (const lean_object*)&lp_mathlib_term___u2192_u2090___00__closed__1_value;
static const lean_string_object lp_mathlib_term___u2192_u2090___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_term___u2192_u2090___00__closed__2 = (const lean_object*)&lp_mathlib_term___u2192_u2090___00__closed__2_value;
static const lean_ctor_object lp_mathlib_term___u2192_u2090___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2192_u2090___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_term___u2192_u2090___00__closed__3 = (const lean_object*)&lp_mathlib_term___u2192_u2090___00__closed__3_value;
static const lean_string_object lp_mathlib_term___u2192_u2090___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 4, .m_data = " →ₐ "};
static const lean_object* lp_mathlib_term___u2192_u2090___00__closed__4 = (const lean_object*)&lp_mathlib_term___u2192_u2090___00__closed__4_value;
static const lean_ctor_object lp_mathlib_term___u2192_u2090___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192_u2090___00__closed__4_value)}};
static const lean_object* lp_mathlib_term___u2192_u2090___00__closed__5 = (const lean_object*)&lp_mathlib_term___u2192_u2090___00__closed__5_value;
static const lean_string_object lp_mathlib_term___u2192_u2090___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_term___u2192_u2090___00__closed__6 = (const lean_object*)&lp_mathlib_term___u2192_u2090___00__closed__6_value;
static const lean_ctor_object lp_mathlib_term___u2192_u2090___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2192_u2090___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_term___u2192_u2090___00__closed__7 = (const lean_object*)&lp_mathlib_term___u2192_u2090___00__closed__7_value;
static const lean_ctor_object lp_mathlib_term___u2192_u2090___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192_u2090___00__closed__7_value),((lean_object*)(((size_t)(25) << 1) | 1))}};
static const lean_object* lp_mathlib_term___u2192_u2090___00__closed__8 = (const lean_object*)&lp_mathlib_term___u2192_u2090___00__closed__8_value;
static const lean_ctor_object lp_mathlib_term___u2192_u2090___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192_u2090___00__closed__3_value),((lean_object*)&lp_mathlib_term___u2192_u2090___00__closed__5_value),((lean_object*)&lp_mathlib_term___u2192_u2090___00__closed__8_value)}};
static const lean_object* lp_mathlib_term___u2192_u2090___00__closed__9 = (const lean_object*)&lp_mathlib_term___u2192_u2090___00__closed__9_value;
static const lean_ctor_object lp_mathlib_term___u2192_u2090___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192_u2090___00__closed__1_value),((lean_object*)(((size_t)(25) << 1) | 1)),((lean_object*)(((size_t)(26) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2192_u2090___00__closed__9_value)}};
static const lean_object* lp_mathlib_term___u2192_u2090___00__closed__10 = (const lean_object*)&lp_mathlib_term___u2192_u2090___00__closed__10_value;
LEAN_EXPORT const lean_object* lp_mathlib_term___u2192_u2090__ = (const lean_object*)&lp_mathlib_term___u2192_u2090___00__closed__10_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1___closed__0_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1___closed__1_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1___closed__2_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1___closed__3_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1___closed__4_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1___closed__4_value_aux_1),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1___closed__4_value_aux_2),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1___closed__4 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1___closed__4_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "AlgHom"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1___closed__5 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1___closed__5_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1___closed__6;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(108, 55, 29, 249, 16, 81, 0, 42)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1___closed__7 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1___closed__7_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1___closed__7_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1___closed__8 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1___closed__8_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1___closed__7_value)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1___closed__9 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1___closed__9_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1___closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1___closed__10 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1___closed__10_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1___closed__8_value),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1___closed__10_value)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1___closed__11 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1___closed__11_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1___closed__12 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1___closed__12_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1___closed__12_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1___closed__13 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1___closed__13_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "hole"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1___closed__14 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1___closed__14_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1___closed__15_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1___closed__15_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1___closed__15_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1___closed__15_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1___closed__15_value_aux_1),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1___closed__15_value_aux_2),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1___closed__14_value),LEAN_SCALAR_PTR_LITERAL(135, 134, 219, 115, 97, 130, 74, 55)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1___closed__15 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1___closed__15_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "_"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1___closed__16 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1___closed__16_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_term___u2192_u2090_x5b___x5d___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 11, .m_data = "term_→ₐ[_]_"};
static const lean_object* lp_mathlib_term___u2192_u2090_x5b___x5d___00__closed__0 = (const lean_object*)&lp_mathlib_term___u2192_u2090_x5b___x5d___00__closed__0_value;
static const lean_ctor_object lp_mathlib_term___u2192_u2090_x5b___x5d___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2192_u2090_x5b___x5d___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(135, 209, 219, 188, 81, 91, 57, 86)}};
static const lean_object* lp_mathlib_term___u2192_u2090_x5b___x5d___00__closed__1 = (const lean_object*)&lp_mathlib_term___u2192_u2090_x5b___x5d___00__closed__1_value;
static const lean_string_object lp_mathlib_term___u2192_u2090_x5b___x5d___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 4, .m_data = " →ₐ["};
static const lean_object* lp_mathlib_term___u2192_u2090_x5b___x5d___00__closed__2 = (const lean_object*)&lp_mathlib_term___u2192_u2090_x5b___x5d___00__closed__2_value;
static const lean_ctor_object lp_mathlib_term___u2192_u2090_x5b___x5d___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192_u2090_x5b___x5d___00__closed__2_value)}};
static const lean_object* lp_mathlib_term___u2192_u2090_x5b___x5d___00__closed__3 = (const lean_object*)&lp_mathlib_term___u2192_u2090_x5b___x5d___00__closed__3_value;
static const lean_ctor_object lp_mathlib_term___u2192_u2090_x5b___x5d___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192_u2090___00__closed__7_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_term___u2192_u2090_x5b___x5d___00__closed__4 = (const lean_object*)&lp_mathlib_term___u2192_u2090_x5b___x5d___00__closed__4_value;
static const lean_ctor_object lp_mathlib_term___u2192_u2090_x5b___x5d___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192_u2090___00__closed__3_value),((lean_object*)&lp_mathlib_term___u2192_u2090_x5b___x5d___00__closed__3_value),((lean_object*)&lp_mathlib_term___u2192_u2090_x5b___x5d___00__closed__4_value)}};
static const lean_object* lp_mathlib_term___u2192_u2090_x5b___x5d___00__closed__5 = (const lean_object*)&lp_mathlib_term___u2192_u2090_x5b___x5d___00__closed__5_value;
static const lean_string_object lp_mathlib_term___u2192_u2090_x5b___x5d___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "] "};
static const lean_object* lp_mathlib_term___u2192_u2090_x5b___x5d___00__closed__6 = (const lean_object*)&lp_mathlib_term___u2192_u2090_x5b___x5d___00__closed__6_value;
static const lean_ctor_object lp_mathlib_term___u2192_u2090_x5b___x5d___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192_u2090_x5b___x5d___00__closed__6_value)}};
static const lean_object* lp_mathlib_term___u2192_u2090_x5b___x5d___00__closed__7 = (const lean_object*)&lp_mathlib_term___u2192_u2090_x5b___x5d___00__closed__7_value;
static const lean_ctor_object lp_mathlib_term___u2192_u2090_x5b___x5d___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192_u2090___00__closed__3_value),((lean_object*)&lp_mathlib_term___u2192_u2090_x5b___x5d___00__closed__5_value),((lean_object*)&lp_mathlib_term___u2192_u2090_x5b___x5d___00__closed__7_value)}};
static const lean_object* lp_mathlib_term___u2192_u2090_x5b___x5d___00__closed__8 = (const lean_object*)&lp_mathlib_term___u2192_u2090_x5b___x5d___00__closed__8_value;
static const lean_ctor_object lp_mathlib_term___u2192_u2090_x5b___x5d___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192_u2090___00__closed__3_value),((lean_object*)&lp_mathlib_term___u2192_u2090_x5b___x5d___00__closed__8_value),((lean_object*)&lp_mathlib_term___u2192_u2090_x5b___x5d___00__closed__4_value)}};
static const lean_object* lp_mathlib_term___u2192_u2090_x5b___x5d___00__closed__9 = (const lean_object*)&lp_mathlib_term___u2192_u2090_x5b___x5d___00__closed__9_value;
static const lean_ctor_object lp_mathlib_term___u2192_u2090_x5b___x5d___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192_u2090_x5b___x5d___00__closed__1_value),((lean_object*)(((size_t)(25) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2192_u2090_x5b___x5d___00__closed__9_value)}};
static const lean_object* lp_mathlib_term___u2192_u2090_x5b___x5d___00__closed__10 = (const lean_object*)&lp_mathlib_term___u2192_u2090_x5b___x5d___00__closed__10_value;
LEAN_EXPORT const lean_object* lp_mathlib_term___u2192_u2090_x5b___x5d__ = (const lean_object*)&lp_mathlib_term___u2192_u2090_x5b___x5d___00__closed__10_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090_x5b___x5d____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090_x5b___x5d____1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______unexpand__AlgHom__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______unexpand__AlgHom__1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______unexpand__AlgHom__1___closed__0_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______unexpand__AlgHom__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______unexpand__AlgHom__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______unexpand__AlgHom__1___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______unexpand__AlgHom__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______unexpand__AlgHom__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______unexpand__AlgHom__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_ofClass___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_ofClass(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_ofClass___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHomClass_toAlgHom___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHomClass_toAlgHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHomClass_toAlgHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_Simps_apply___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_Simps_apply(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_Simps_apply___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_toMonoidHom_x27___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_toMonoidHom_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_toMonoidHom_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_toMonoidHom_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_coeOutMonoidHom___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_coeOutMonoidHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_toAddMonoidHom_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_toAddMonoidHom_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_toAddMonoidHom_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_coeOutAddMonoidHom___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_coeOutAddMonoidHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_mk_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_mk_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_mk_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_AlgHom_id___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_NonUnitalRingHom_id___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_AlgHom_id___closed__0 = (const lean_object*)&lp_mathlib_AlgHom_id___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_id(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_id___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_comp___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_comp___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_comp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_comp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_toLinearMap___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_toLinearMap___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_toLinearMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_toLinearMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_ofLinearMap___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_ofLinearMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_ofLinearMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_End___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_End(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_toEnd___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_toEnd(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsScalarTower_toAlgHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsScalarTower_toAlgHom___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsScalarTower_toAlgHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsScalarTower_toAlgHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Algebra_algHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Algebra_algHom___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Algebra_algHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Algebra_algHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_toNatAlgHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_toNatAlgHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_toNatAlgHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_equivNatAlgHom___redArg___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_RingHom_equivNatAlgHom___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_RingHom_equivNatAlgHom___redArg___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_RingHom_equivNatAlgHom___redArg___closed__0 = (const lean_object*)&lp_mathlib_RingHom_equivNatAlgHom___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_RingHom_equivNatAlgHom___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_equivNatAlgHom(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_toIntAlgHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_toIntAlgHom___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_toIntAlgHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_toIntAlgHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_equivIntAlgHom___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_equivIntAlgHom(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Algebra_ofId___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Algebra_ofId___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Algebra_ofId(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Algebra_ofId___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Algebra_instMulDistribMulActionAlgHomUnits___lam__1(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Algebra_instMulDistribMulActionAlgHomUnits___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Algebra_instMulDistribMulActionAlgHomUnits___lam__1, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Algebra_instMulDistribMulActionAlgHomUnits___closed__0 = (const lean_object*)&lp_mathlib_Algebra_instMulDistribMulActionAlgHomUnits___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Algebra_instMulDistribMulActionAlgHomUnits(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Algebra_instMulDistribMulActionAlgHomUnits___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulSemiringAction_toAlgHom___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulSemiringAction_toAlgHom___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulSemiringAction_toAlgHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulSemiringAction_toAlgHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_uniqueOfRight___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_uniqueOfRight___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_uniqueOfRight___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_uniqueOfRight___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_uniqueOfRight(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_uniqueOfRight___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1___closed__6(void){
_start:
{
lean_object* v___x_36_; lean_object* v___x_37_; 
v___x_36_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1___closed__5));
v___x_37_ = l_String_toRawSubstring_x27(v___x_36_);
return v___x_37_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1(lean_object* v_x_61_, lean_object* v_a_62_, lean_object* v_a_63_){
_start:
{
lean_object* v___x_64_; uint8_t v___x_65_; 
v___x_64_ = ((lean_object*)(lp_mathlib_term___u2192_u2090___00__closed__1));
lean_inc(v_x_61_);
v___x_65_ = l_Lean_Syntax_isOfKind(v_x_61_, v___x_64_);
if (v___x_65_ == 0)
{
lean_object* v___x_66_; lean_object* v___x_67_; 
lean_dec(v_x_61_);
v___x_66_ = lean_box(1);
v___x_67_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_67_, 0, v___x_66_);
lean_ctor_set(v___x_67_, 1, v_a_63_);
return v___x_67_;
}
else
{
lean_object* v_quotContext_68_; lean_object* v_currMacroScope_69_; lean_object* v_ref_70_; lean_object* v___x_71_; lean_object* v___x_72_; lean_object* v___x_73_; lean_object* v___x_74_; uint8_t v___x_75_; lean_object* v___x_76_; lean_object* v___x_77_; lean_object* v___x_78_; lean_object* v___x_79_; lean_object* v___x_80_; lean_object* v___x_81_; lean_object* v___x_82_; lean_object* v___x_83_; lean_object* v___x_84_; lean_object* v___x_85_; lean_object* v___x_86_; lean_object* v___x_87_; lean_object* v___x_88_; lean_object* v___x_89_; lean_object* v___x_90_; lean_object* v___x_91_; lean_object* v___x_92_; 
v_quotContext_68_ = lean_ctor_get(v_a_62_, 1);
v_currMacroScope_69_ = lean_ctor_get(v_a_62_, 2);
v_ref_70_ = lean_ctor_get(v_a_62_, 5);
v___x_71_ = lean_unsigned_to_nat(0u);
v___x_72_ = l_Lean_Syntax_getArg(v_x_61_, v___x_71_);
v___x_73_ = lean_unsigned_to_nat(2u);
v___x_74_ = l_Lean_Syntax_getArg(v_x_61_, v___x_73_);
lean_dec(v_x_61_);
v___x_75_ = 0;
v___x_76_ = l_Lean_SourceInfo_fromRef(v_ref_70_, v___x_75_);
v___x_77_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1___closed__4));
v___x_78_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1___closed__6, &lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1___closed__6_once, _init_lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1___closed__6);
v___x_79_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1___closed__7));
lean_inc(v_currMacroScope_69_);
lean_inc(v_quotContext_68_);
v___x_80_ = l_Lean_addMacroScope(v_quotContext_68_, v___x_79_, v_currMacroScope_69_);
v___x_81_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1___closed__11));
lean_inc_n(v___x_76_, 6);
v___x_82_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_82_, 0, v___x_76_);
lean_ctor_set(v___x_82_, 1, v___x_78_);
lean_ctor_set(v___x_82_, 2, v___x_80_);
lean_ctor_set(v___x_82_, 3, v___x_81_);
v___x_83_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1___closed__13));
v___x_84_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1___closed__15));
v___x_85_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1___closed__16));
v___x_86_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_86_, 0, v___x_76_);
lean_ctor_set(v___x_86_, 1, v___x_85_);
v___x_87_ = l_Lean_Syntax_node1(v___x_76_, v___x_84_, v___x_86_);
v___x_88_ = l_Lean_Syntax_node1(v___x_76_, v___x_83_, v___x_87_);
v___x_89_ = l_Lean_Syntax_node2(v___x_76_, v___x_77_, v___x_82_, v___x_88_);
v___x_90_ = l_Lean_Syntax_node2(v___x_76_, v___x_83_, v___x_72_, v___x_74_);
v___x_91_ = l_Lean_Syntax_node2(v___x_76_, v___x_77_, v___x_89_, v___x_90_);
v___x_92_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_92_, 0, v___x_91_);
lean_ctor_set(v___x_92_, 1, v_a_63_);
return v___x_92_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1___boxed(lean_object* v_x_93_, lean_object* v_a_94_, lean_object* v_a_95_){
_start:
{
lean_object* v_res_96_; 
v_res_96_ = lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1(v_x_93_, v_a_94_, v_a_95_);
lean_dec_ref(v_a_94_);
return v_res_96_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090_x5b___x5d____1(lean_object* v_x_127_, lean_object* v_a_128_, lean_object* v_a_129_){
_start:
{
lean_object* v___x_130_; uint8_t v___x_131_; 
v___x_130_ = ((lean_object*)(lp_mathlib_term___u2192_u2090_x5b___x5d___00__closed__1));
lean_inc(v_x_127_);
v___x_131_ = l_Lean_Syntax_isOfKind(v_x_127_, v___x_130_);
if (v___x_131_ == 0)
{
lean_object* v___x_132_; lean_object* v___x_133_; 
lean_dec(v_x_127_);
v___x_132_ = lean_box(1);
v___x_133_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_133_, 0, v___x_132_);
lean_ctor_set(v___x_133_, 1, v_a_129_);
return v___x_133_;
}
else
{
lean_object* v_quotContext_134_; lean_object* v_currMacroScope_135_; lean_object* v_ref_136_; lean_object* v___x_137_; lean_object* v___x_138_; lean_object* v___x_139_; lean_object* v___x_140_; lean_object* v___x_141_; lean_object* v___x_142_; uint8_t v___x_143_; lean_object* v___x_144_; lean_object* v___x_145_; lean_object* v___x_146_; lean_object* v___x_147_; lean_object* v___x_148_; lean_object* v___x_149_; lean_object* v___x_150_; lean_object* v___x_151_; lean_object* v___x_152_; lean_object* v___x_153_; lean_object* v___x_154_; 
v_quotContext_134_ = lean_ctor_get(v_a_128_, 1);
v_currMacroScope_135_ = lean_ctor_get(v_a_128_, 2);
v_ref_136_ = lean_ctor_get(v_a_128_, 5);
v___x_137_ = lean_unsigned_to_nat(0u);
v___x_138_ = l_Lean_Syntax_getArg(v_x_127_, v___x_137_);
v___x_139_ = lean_unsigned_to_nat(2u);
v___x_140_ = l_Lean_Syntax_getArg(v_x_127_, v___x_139_);
v___x_141_ = lean_unsigned_to_nat(4u);
v___x_142_ = l_Lean_Syntax_getArg(v_x_127_, v___x_141_);
lean_dec(v_x_127_);
v___x_143_ = 0;
v___x_144_ = l_Lean_SourceInfo_fromRef(v_ref_136_, v___x_143_);
v___x_145_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1___closed__4));
v___x_146_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1___closed__6, &lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1___closed__6_once, _init_lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1___closed__6);
v___x_147_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1___closed__7));
lean_inc(v_currMacroScope_135_);
lean_inc(v_quotContext_134_);
v___x_148_ = l_Lean_addMacroScope(v_quotContext_134_, v___x_147_, v_currMacroScope_135_);
v___x_149_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1___closed__11));
lean_inc_n(v___x_144_, 2);
v___x_150_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_150_, 0, v___x_144_);
lean_ctor_set(v___x_150_, 1, v___x_146_);
lean_ctor_set(v___x_150_, 2, v___x_148_);
lean_ctor_set(v___x_150_, 3, v___x_149_);
v___x_151_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1___closed__13));
v___x_152_ = l_Lean_Syntax_node3(v___x_144_, v___x_151_, v___x_140_, v___x_138_, v___x_142_);
v___x_153_ = l_Lean_Syntax_node2(v___x_144_, v___x_145_, v___x_150_, v___x_152_);
v___x_154_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_154_, 0, v___x_153_);
lean_ctor_set(v___x_154_, 1, v_a_129_);
return v___x_154_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090_x5b___x5d____1___boxed(lean_object* v_x_155_, lean_object* v_a_156_, lean_object* v_a_157_){
_start:
{
lean_object* v_res_158_; 
v_res_158_ = lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090_x5b___x5d____1(v_x_155_, v_a_156_, v_a_157_);
lean_dec_ref(v_a_156_);
return v_res_158_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______unexpand__AlgHom__1(lean_object* v_x_162_, lean_object* v_a_163_, lean_object* v_a_164_){
_start:
{
lean_object* v___x_165_; uint8_t v___x_166_; 
v___x_165_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______macroRules__term___u2192_u2090____1___closed__4));
lean_inc(v_x_162_);
v___x_166_ = l_Lean_Syntax_isOfKind(v_x_162_, v___x_165_);
if (v___x_166_ == 0)
{
lean_object* v___x_167_; lean_object* v___x_168_; 
lean_dec(v_x_162_);
v___x_167_ = lean_box(0);
v___x_168_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_168_, 0, v___x_167_);
lean_ctor_set(v___x_168_, 1, v_a_164_);
return v___x_168_;
}
else
{
lean_object* v___x_169_; lean_object* v___x_170_; lean_object* v___x_171_; uint8_t v___x_172_; 
v___x_169_ = lean_unsigned_to_nat(0u);
v___x_170_ = l_Lean_Syntax_getArg(v_x_162_, v___x_169_);
v___x_171_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______unexpand__AlgHom__1___closed__1));
lean_inc(v___x_170_);
v___x_172_ = l_Lean_Syntax_isOfKind(v___x_170_, v___x_171_);
if (v___x_172_ == 0)
{
lean_object* v___x_173_; lean_object* v___x_174_; 
lean_dec(v___x_170_);
lean_dec(v_x_162_);
v___x_173_ = lean_box(0);
v___x_174_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_174_, 0, v___x_173_);
lean_ctor_set(v___x_174_, 1, v_a_164_);
return v___x_174_;
}
else
{
lean_object* v___x_175_; lean_object* v___x_176_; lean_object* v___x_177_; uint8_t v___x_178_; 
v___x_175_ = lean_unsigned_to_nat(1u);
v___x_176_ = l_Lean_Syntax_getArg(v_x_162_, v___x_175_);
lean_dec(v_x_162_);
v___x_177_ = lean_unsigned_to_nat(3u);
lean_inc(v___x_176_);
v___x_178_ = l_Lean_Syntax_matchesNull(v___x_176_, v___x_177_);
if (v___x_178_ == 0)
{
lean_object* v___x_179_; lean_object* v___x_180_; 
lean_dec(v___x_176_);
lean_dec(v___x_170_);
v___x_179_ = lean_box(0);
v___x_180_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_180_, 0, v___x_179_);
lean_ctor_set(v___x_180_, 1, v_a_164_);
return v___x_180_;
}
else
{
lean_object* v___x_181_; lean_object* v___x_182_; lean_object* v___x_183_; lean_object* v___x_184_; lean_object* v_ref_185_; uint8_t v___x_186_; lean_object* v___x_187_; lean_object* v___x_188_; lean_object* v___x_189_; lean_object* v___x_190_; lean_object* v___x_191_; lean_object* v___x_192_; lean_object* v___x_193_; lean_object* v___x_194_; 
v___x_181_ = l_Lean_Syntax_getArg(v___x_176_, v___x_169_);
v___x_182_ = l_Lean_Syntax_getArg(v___x_176_, v___x_175_);
v___x_183_ = lean_unsigned_to_nat(2u);
v___x_184_ = l_Lean_Syntax_getArg(v___x_176_, v___x_183_);
lean_dec(v___x_176_);
v_ref_185_ = l_Lean_replaceRef(v___x_170_, v_a_163_);
lean_dec(v___x_170_);
v___x_186_ = 0;
v___x_187_ = l_Lean_SourceInfo_fromRef(v_ref_185_, v___x_186_);
lean_dec(v_ref_185_);
v___x_188_ = ((lean_object*)(lp_mathlib_term___u2192_u2090_x5b___x5d___00__closed__1));
v___x_189_ = ((lean_object*)(lp_mathlib_term___u2192_u2090_x5b___x5d___00__closed__2));
lean_inc_n(v___x_187_, 2);
v___x_190_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_190_, 0, v___x_187_);
lean_ctor_set(v___x_190_, 1, v___x_189_);
v___x_191_ = ((lean_object*)(lp_mathlib_term___u2192_u2090_x5b___x5d___00__closed__6));
v___x_192_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_192_, 0, v___x_187_);
lean_ctor_set(v___x_192_, 1, v___x_191_);
v___x_193_ = l_Lean_Syntax_node5(v___x_187_, v___x_188_, v___x_182_, v___x_190_, v___x_181_, v___x_192_, v___x_184_);
v___x_194_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_194_, 0, v___x_193_);
lean_ctor_set(v___x_194_, 1, v_a_164_);
return v___x_194_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______unexpand__AlgHom__1___boxed(lean_object* v_x_195_, lean_object* v_a_196_, lean_object* v_a_197_){
_start:
{
lean_object* v_res_198_; 
v_res_198_ = lp_mathlib___aux__Mathlib__Algebra__Algebra__Hom______unexpand__AlgHom__1(v_x_195_, v_a_196_, v_a_197_);
lean_dec(v_a_196_);
return v_res_198_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_ofClass___redArg(lean_object* v_inst_199_, lean_object* v_f_200_){
_start:
{
lean_object* v___x_201_; 
v___x_201_ = lean_apply_1(v_inst_199_, v_f_200_);
return v___x_201_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_ofClass(lean_object* v_R_202_, lean_object* v_A_203_, lean_object* v_B_204_, lean_object* v_inst_205_, lean_object* v_inst_206_, lean_object* v_inst_207_, lean_object* v_inst_208_, lean_object* v_inst_209_, lean_object* v_F_210_, lean_object* v_inst_211_, lean_object* v_inst_212_, lean_object* v_f_213_){
_start:
{
lean_object* v___x_214_; 
v___x_214_ = lean_apply_1(v_inst_211_, v_f_213_);
return v___x_214_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_ofClass___boxed(lean_object* v_R_215_, lean_object* v_A_216_, lean_object* v_B_217_, lean_object* v_inst_218_, lean_object* v_inst_219_, lean_object* v_inst_220_, lean_object* v_inst_221_, lean_object* v_inst_222_, lean_object* v_F_223_, lean_object* v_inst_224_, lean_object* v_inst_225_, lean_object* v_f_226_){
_start:
{
lean_object* v_res_227_; 
v_res_227_ = lp_mathlib_AlgHom_ofClass(v_R_215_, v_A_216_, v_B_217_, v_inst_218_, v_inst_219_, v_inst_220_, v_inst_221_, v_inst_222_, v_F_223_, v_inst_224_, v_inst_225_, v_f_226_);
lean_dec_ref(v_inst_222_);
lean_dec_ref(v_inst_221_);
lean_dec_ref(v_inst_220_);
lean_dec_ref(v_inst_219_);
lean_dec_ref(v_inst_218_);
return v_res_227_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHomClass_toAlgHom___redArg(lean_object* v_inst_228_, lean_object* v_f_229_){
_start:
{
lean_object* v___x_230_; 
v___x_230_ = lean_apply_1(v_inst_228_, v_f_229_);
return v___x_230_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHomClass_toAlgHom(lean_object* v_R_231_, lean_object* v_A_232_, lean_object* v_B_233_, lean_object* v_inst_234_, lean_object* v_inst_235_, lean_object* v_inst_236_, lean_object* v_inst_237_, lean_object* v_inst_238_, lean_object* v_F_239_, lean_object* v_inst_240_, lean_object* v_inst_241_, lean_object* v_f_242_){
_start:
{
lean_object* v___x_243_; 
v___x_243_ = lean_apply_1(v_inst_240_, v_f_242_);
return v___x_243_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHomClass_toAlgHom___boxed(lean_object* v_R_244_, lean_object* v_A_245_, lean_object* v_B_246_, lean_object* v_inst_247_, lean_object* v_inst_248_, lean_object* v_inst_249_, lean_object* v_inst_250_, lean_object* v_inst_251_, lean_object* v_F_252_, lean_object* v_inst_253_, lean_object* v_inst_254_, lean_object* v_f_255_){
_start:
{
lean_object* v_res_256_; 
v_res_256_ = lp_mathlib_AlgHomClass_toAlgHom(v_R_244_, v_A_245_, v_B_246_, v_inst_247_, v_inst_248_, v_inst_249_, v_inst_250_, v_inst_251_, v_F_252_, v_inst_253_, v_inst_254_, v_f_255_);
lean_dec_ref(v_inst_251_);
lean_dec_ref(v_inst_250_);
lean_dec_ref(v_inst_249_);
lean_dec_ref(v_inst_248_);
lean_dec_ref(v_inst_247_);
return v_res_256_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_Simps_apply___redArg(lean_object* v_f_257_, lean_object* v_a_258_){
_start:
{
lean_object* v___x_259_; 
v___x_259_ = lean_apply_1(v_f_257_, v_a_258_);
return v___x_259_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_Simps_apply(lean_object* v_R_260_, lean_object* v_00_u03b1_261_, lean_object* v_00_u03b2_262_, lean_object* v_inst_263_, lean_object* v_inst_264_, lean_object* v_inst_265_, lean_object* v_inst_266_, lean_object* v_inst_267_, lean_object* v_f_268_, lean_object* v_a_269_){
_start:
{
lean_object* v___x_270_; 
v___x_270_ = lean_apply_1(v_f_268_, v_a_269_);
return v___x_270_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_Simps_apply___boxed(lean_object* v_R_271_, lean_object* v_00_u03b1_272_, lean_object* v_00_u03b2_273_, lean_object* v_inst_274_, lean_object* v_inst_275_, lean_object* v_inst_276_, lean_object* v_inst_277_, lean_object* v_inst_278_, lean_object* v_f_279_, lean_object* v_a_280_){
_start:
{
lean_object* v_res_281_; 
v_res_281_ = lp_mathlib_AlgHom_Simps_apply(v_R_271_, v_00_u03b1_272_, v_00_u03b2_273_, v_inst_274_, v_inst_275_, v_inst_276_, v_inst_277_, v_inst_278_, v_f_279_, v_a_280_);
lean_dec_ref(v_inst_278_);
lean_dec_ref(v_inst_277_);
lean_dec_ref(v_inst_276_);
lean_dec_ref(v_inst_275_);
lean_dec_ref(v_inst_274_);
return v_res_281_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_toMonoidHom_x27___redArg___lam__0(lean_object* v_f_282_, lean_object* v___y_283_){
_start:
{
lean_object* v___x_284_; 
v___x_284_ = lean_apply_1(v_f_282_, v___y_283_);
return v___x_284_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_toMonoidHom_x27___redArg(lean_object* v_f_285_){
_start:
{
lean_object* v___f_286_; 
v___f_286_ = lean_alloc_closure((void*)(lp_mathlib_AlgHom_toMonoidHom_x27___redArg___lam__0), 2, 1);
lean_closure_set(v___f_286_, 0, v_f_285_);
return v___f_286_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_toMonoidHom_x27(lean_object* v_R_287_, lean_object* v_A_288_, lean_object* v_B_289_, lean_object* v_inst_290_, lean_object* v_inst_291_, lean_object* v_inst_292_, lean_object* v_inst_293_, lean_object* v_inst_294_, lean_object* v_f_295_){
_start:
{
lean_object* v___f_296_; 
v___f_296_ = lean_alloc_closure((void*)(lp_mathlib_AlgHom_toMonoidHom_x27___redArg___lam__0), 2, 1);
lean_closure_set(v___f_296_, 0, v_f_295_);
return v___f_296_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_toMonoidHom_x27___boxed(lean_object* v_R_297_, lean_object* v_A_298_, lean_object* v_B_299_, lean_object* v_inst_300_, lean_object* v_inst_301_, lean_object* v_inst_302_, lean_object* v_inst_303_, lean_object* v_inst_304_, lean_object* v_f_305_){
_start:
{
lean_object* v_res_306_; 
v_res_306_ = lp_mathlib_AlgHom_toMonoidHom_x27(v_R_297_, v_A_298_, v_B_299_, v_inst_300_, v_inst_301_, v_inst_302_, v_inst_303_, v_inst_304_, v_f_305_);
lean_dec_ref(v_inst_304_);
lean_dec_ref(v_inst_303_);
lean_dec_ref(v_inst_302_);
lean_dec_ref(v_inst_301_);
lean_dec_ref(v_inst_300_);
return v_res_306_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_coeOutMonoidHom___redArg(lean_object* v_inst_307_, lean_object* v_inst_308_, lean_object* v_inst_309_, lean_object* v_inst_310_, lean_object* v_inst_311_){
_start:
{
lean_object* v___x_312_; 
v___x_312_ = lean_alloc_closure((void*)(lp_mathlib_AlgHom_toMonoidHom_x27___boxed), 9, 8);
lean_closure_set(v___x_312_, 0, lean_box(0));
lean_closure_set(v___x_312_, 1, lean_box(0));
lean_closure_set(v___x_312_, 2, lean_box(0));
lean_closure_set(v___x_312_, 3, v_inst_307_);
lean_closure_set(v___x_312_, 4, v_inst_308_);
lean_closure_set(v___x_312_, 5, v_inst_309_);
lean_closure_set(v___x_312_, 6, v_inst_310_);
lean_closure_set(v___x_312_, 7, v_inst_311_);
return v___x_312_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_coeOutMonoidHom(lean_object* v_R_313_, lean_object* v_A_314_, lean_object* v_B_315_, lean_object* v_inst_316_, lean_object* v_inst_317_, lean_object* v_inst_318_, lean_object* v_inst_319_, lean_object* v_inst_320_){
_start:
{
lean_object* v___x_321_; 
v___x_321_ = lean_alloc_closure((void*)(lp_mathlib_AlgHom_toMonoidHom_x27___boxed), 9, 8);
lean_closure_set(v___x_321_, 0, lean_box(0));
lean_closure_set(v___x_321_, 1, lean_box(0));
lean_closure_set(v___x_321_, 2, lean_box(0));
lean_closure_set(v___x_321_, 3, v_inst_316_);
lean_closure_set(v___x_321_, 4, v_inst_317_);
lean_closure_set(v___x_321_, 5, v_inst_318_);
lean_closure_set(v___x_321_, 6, v_inst_319_);
lean_closure_set(v___x_321_, 7, v_inst_320_);
return v___x_321_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_toAddMonoidHom_x27___redArg(lean_object* v_f_322_){
_start:
{
lean_object* v___f_323_; 
v___f_323_ = lean_alloc_closure((void*)(lp_mathlib_AlgHom_toMonoidHom_x27___redArg___lam__0), 2, 1);
lean_closure_set(v___f_323_, 0, v_f_322_);
return v___f_323_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_toAddMonoidHom_x27(lean_object* v_R_324_, lean_object* v_A_325_, lean_object* v_B_326_, lean_object* v_inst_327_, lean_object* v_inst_328_, lean_object* v_inst_329_, lean_object* v_inst_330_, lean_object* v_inst_331_, lean_object* v_f_332_){
_start:
{
lean_object* v___f_333_; 
v___f_333_ = lean_alloc_closure((void*)(lp_mathlib_AlgHom_toMonoidHom_x27___redArg___lam__0), 2, 1);
lean_closure_set(v___f_333_, 0, v_f_332_);
return v___f_333_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_toAddMonoidHom_x27___boxed(lean_object* v_R_334_, lean_object* v_A_335_, lean_object* v_B_336_, lean_object* v_inst_337_, lean_object* v_inst_338_, lean_object* v_inst_339_, lean_object* v_inst_340_, lean_object* v_inst_341_, lean_object* v_f_342_){
_start:
{
lean_object* v_res_343_; 
v_res_343_ = lp_mathlib_AlgHom_toAddMonoidHom_x27(v_R_334_, v_A_335_, v_B_336_, v_inst_337_, v_inst_338_, v_inst_339_, v_inst_340_, v_inst_341_, v_f_342_);
lean_dec_ref(v_inst_341_);
lean_dec_ref(v_inst_340_);
lean_dec_ref(v_inst_339_);
lean_dec_ref(v_inst_338_);
lean_dec_ref(v_inst_337_);
return v_res_343_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_coeOutAddMonoidHom___redArg(lean_object* v_inst_344_, lean_object* v_inst_345_, lean_object* v_inst_346_, lean_object* v_inst_347_, lean_object* v_inst_348_){
_start:
{
lean_object* v___x_349_; 
v___x_349_ = lean_alloc_closure((void*)(lp_mathlib_AlgHom_toAddMonoidHom_x27___boxed), 9, 8);
lean_closure_set(v___x_349_, 0, lean_box(0));
lean_closure_set(v___x_349_, 1, lean_box(0));
lean_closure_set(v___x_349_, 2, lean_box(0));
lean_closure_set(v___x_349_, 3, v_inst_344_);
lean_closure_set(v___x_349_, 4, v_inst_345_);
lean_closure_set(v___x_349_, 5, v_inst_346_);
lean_closure_set(v___x_349_, 6, v_inst_347_);
lean_closure_set(v___x_349_, 7, v_inst_348_);
return v___x_349_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_coeOutAddMonoidHom(lean_object* v_R_350_, lean_object* v_A_351_, lean_object* v_B_352_, lean_object* v_inst_353_, lean_object* v_inst_354_, lean_object* v_inst_355_, lean_object* v_inst_356_, lean_object* v_inst_357_){
_start:
{
lean_object* v___x_358_; 
v___x_358_ = lean_alloc_closure((void*)(lp_mathlib_AlgHom_toAddMonoidHom_x27___boxed), 9, 8);
lean_closure_set(v___x_358_, 0, lean_box(0));
lean_closure_set(v___x_358_, 1, lean_box(0));
lean_closure_set(v___x_358_, 2, lean_box(0));
lean_closure_set(v___x_358_, 3, v_inst_353_);
lean_closure_set(v___x_358_, 4, v_inst_354_);
lean_closure_set(v___x_358_, 5, v_inst_355_);
lean_closure_set(v___x_358_, 6, v_inst_356_);
lean_closure_set(v___x_358_, 7, v_inst_357_);
return v___x_358_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_mk_x27___redArg(lean_object* v_f_359_){
_start:
{
lean_object* v___f_360_; 
v___f_360_ = lean_alloc_closure((void*)(lp_mathlib_AlgHom_toMonoidHom_x27___redArg___lam__0), 2, 1);
lean_closure_set(v___f_360_, 0, v_f_359_);
return v___f_360_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_mk_x27(lean_object* v_R_361_, lean_object* v_A_362_, lean_object* v_B_363_, lean_object* v_inst_364_, lean_object* v_inst_365_, lean_object* v_inst_366_, lean_object* v_inst_367_, lean_object* v_inst_368_, lean_object* v_f_369_, lean_object* v_h_370_){
_start:
{
lean_object* v___f_371_; 
v___f_371_ = lean_alloc_closure((void*)(lp_mathlib_AlgHom_toMonoidHom_x27___redArg___lam__0), 2, 1);
lean_closure_set(v___f_371_, 0, v_f_369_);
return v___f_371_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_mk_x27___boxed(lean_object* v_R_372_, lean_object* v_A_373_, lean_object* v_B_374_, lean_object* v_inst_375_, lean_object* v_inst_376_, lean_object* v_inst_377_, lean_object* v_inst_378_, lean_object* v_inst_379_, lean_object* v_f_380_, lean_object* v_h_381_){
_start:
{
lean_object* v_res_382_; 
v_res_382_ = lp_mathlib_AlgHom_mk_x27(v_R_372_, v_A_373_, v_B_374_, v_inst_375_, v_inst_376_, v_inst_377_, v_inst_378_, v_inst_379_, v_f_380_, v_h_381_);
lean_dec_ref(v_inst_379_);
lean_dec_ref(v_inst_378_);
lean_dec_ref(v_inst_377_);
lean_dec_ref(v_inst_376_);
lean_dec_ref(v_inst_375_);
return v_res_382_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_id(lean_object* v_R_384_, lean_object* v_A_385_, lean_object* v_inst_386_, lean_object* v_inst_387_, lean_object* v_inst_388_){
_start:
{
lean_object* v___f_389_; 
v___f_389_ = ((lean_object*)(lp_mathlib_AlgHom_id___closed__0));
return v___f_389_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_id___boxed(lean_object* v_R_390_, lean_object* v_A_391_, lean_object* v_inst_392_, lean_object* v_inst_393_, lean_object* v_inst_394_){
_start:
{
lean_object* v_res_395_; 
v_res_395_ = lp_mathlib_AlgHom_id(v_R_390_, v_A_391_, v_inst_392_, v_inst_393_, v_inst_394_);
lean_dec_ref(v_inst_394_);
lean_dec_ref(v_inst_393_);
lean_dec_ref(v_inst_392_);
return v_res_395_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_comp___redArg___lam__0(lean_object* v_00_u03c6_u2082_396_, lean_object* v___y_397_){
_start:
{
lean_object* v___x_398_; 
v___x_398_ = lean_apply_1(v_00_u03c6_u2082_396_, v___y_397_);
return v___x_398_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_comp___redArg(lean_object* v_00_u03c6_u2081_399_, lean_object* v_00_u03c6_u2082_400_){
_start:
{
lean_object* v___f_401_; lean_object* v___f_402_; 
v___f_401_ = lean_alloc_closure((void*)(lp_mathlib_AlgHom_comp___redArg___lam__0), 2, 1);
lean_closure_set(v___f_401_, 0, v_00_u03c6_u2082_400_);
v___f_402_ = lean_alloc_closure((void*)(lp_mathlib_RingHom_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_402_, 0, v___f_401_);
lean_closure_set(v___f_402_, 1, v_00_u03c6_u2081_399_);
return v___f_402_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_comp(lean_object* v_R_403_, lean_object* v_A_404_, lean_object* v_B_405_, lean_object* v_C_406_, lean_object* v_inst_407_, lean_object* v_inst_408_, lean_object* v_inst_409_, lean_object* v_inst_410_, lean_object* v_inst_411_, lean_object* v_inst_412_, lean_object* v_inst_413_, lean_object* v_00_u03c6_u2081_414_, lean_object* v_00_u03c6_u2082_415_){
_start:
{
lean_object* v___x_416_; 
v___x_416_ = lp_mathlib_AlgHom_comp___redArg(v_00_u03c6_u2081_414_, v_00_u03c6_u2082_415_);
return v___x_416_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_comp___boxed(lean_object* v_R_417_, lean_object* v_A_418_, lean_object* v_B_419_, lean_object* v_C_420_, lean_object* v_inst_421_, lean_object* v_inst_422_, lean_object* v_inst_423_, lean_object* v_inst_424_, lean_object* v_inst_425_, lean_object* v_inst_426_, lean_object* v_inst_427_, lean_object* v_00_u03c6_u2081_428_, lean_object* v_00_u03c6_u2082_429_){
_start:
{
lean_object* v_res_430_; 
v_res_430_ = lp_mathlib_AlgHom_comp(v_R_417_, v_A_418_, v_B_419_, v_C_420_, v_inst_421_, v_inst_422_, v_inst_423_, v_inst_424_, v_inst_425_, v_inst_426_, v_inst_427_, v_00_u03c6_u2081_428_, v_00_u03c6_u2082_429_);
lean_dec_ref(v_inst_427_);
lean_dec_ref(v_inst_426_);
lean_dec_ref(v_inst_425_);
lean_dec_ref(v_inst_424_);
lean_dec_ref(v_inst_423_);
lean_dec_ref(v_inst_422_);
lean_dec_ref(v_inst_421_);
return v_res_430_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_toLinearMap___redArg___lam__0(lean_object* v_00_u03c6_431_, lean_object* v___y_432_){
_start:
{
lean_object* v___x_433_; 
v___x_433_ = lean_apply_1(v_00_u03c6_431_, v___y_432_);
return v___x_433_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_toLinearMap___redArg(lean_object* v_00_u03c6_434_){
_start:
{
lean_object* v___f_435_; 
v___f_435_ = lean_alloc_closure((void*)(lp_mathlib_AlgHom_toLinearMap___redArg___lam__0), 2, 1);
lean_closure_set(v___f_435_, 0, v_00_u03c6_434_);
return v___f_435_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_toLinearMap(lean_object* v_R_436_, lean_object* v_A_437_, lean_object* v_B_438_, lean_object* v_inst_439_, lean_object* v_inst_440_, lean_object* v_inst_441_, lean_object* v_inst_442_, lean_object* v_inst_443_, lean_object* v_00_u03c6_444_){
_start:
{
lean_object* v___f_445_; 
v___f_445_ = lean_alloc_closure((void*)(lp_mathlib_AlgHom_toLinearMap___redArg___lam__0), 2, 1);
lean_closure_set(v___f_445_, 0, v_00_u03c6_444_);
return v___f_445_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_toLinearMap___boxed(lean_object* v_R_446_, lean_object* v_A_447_, lean_object* v_B_448_, lean_object* v_inst_449_, lean_object* v_inst_450_, lean_object* v_inst_451_, lean_object* v_inst_452_, lean_object* v_inst_453_, lean_object* v_00_u03c6_454_){
_start:
{
lean_object* v_res_455_; 
v_res_455_ = lp_mathlib_AlgHom_toLinearMap(v_R_446_, v_A_447_, v_B_448_, v_inst_449_, v_inst_450_, v_inst_451_, v_inst_452_, v_inst_453_, v_00_u03c6_454_);
lean_dec_ref(v_inst_453_);
lean_dec_ref(v_inst_452_);
lean_dec_ref(v_inst_451_);
lean_dec_ref(v_inst_450_);
lean_dec_ref(v_inst_449_);
return v_res_455_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_ofLinearMap___redArg(lean_object* v_f_456_){
_start:
{
lean_object* v___f_457_; 
v___f_457_ = lean_alloc_closure((void*)(lp_mathlib_AlgHom_toMonoidHom_x27___redArg___lam__0), 2, 1);
lean_closure_set(v___f_457_, 0, v_f_456_);
return v___f_457_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_ofLinearMap(lean_object* v_R_458_, lean_object* v_A_459_, lean_object* v_B_460_, lean_object* v_inst_461_, lean_object* v_inst_462_, lean_object* v_inst_463_, lean_object* v_inst_464_, lean_object* v_inst_465_, lean_object* v_f_466_, lean_object* v_map__one_467_, lean_object* v_map__mul_468_){
_start:
{
lean_object* v___f_469_; 
v___f_469_ = lean_alloc_closure((void*)(lp_mathlib_AlgHom_toMonoidHom_x27___redArg___lam__0), 2, 1);
lean_closure_set(v___f_469_, 0, v_f_466_);
return v___f_469_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_ofLinearMap___boxed(lean_object* v_R_470_, lean_object* v_A_471_, lean_object* v_B_472_, lean_object* v_inst_473_, lean_object* v_inst_474_, lean_object* v_inst_475_, lean_object* v_inst_476_, lean_object* v_inst_477_, lean_object* v_f_478_, lean_object* v_map__one_479_, lean_object* v_map__mul_480_){
_start:
{
lean_object* v_res_481_; 
v_res_481_ = lp_mathlib_AlgHom_ofLinearMap(v_R_470_, v_A_471_, v_B_472_, v_inst_473_, v_inst_474_, v_inst_475_, v_inst_476_, v_inst_477_, v_f_478_, v_map__one_479_, v_map__mul_480_);
lean_dec_ref(v_inst_477_);
lean_dec_ref(v_inst_476_);
lean_dec_ref(v_inst_475_);
lean_dec_ref(v_inst_474_);
lean_dec_ref(v_inst_473_);
return v_res_481_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_End___redArg(lean_object* v_inst_482_, lean_object* v_inst_483_, lean_object* v_inst_484_){
_start:
{
lean_object* v___f_485_; lean_object* v___x_486_; lean_object* v___x_487_; lean_object* v___x_488_; 
v___f_485_ = ((lean_object*)(lp_mathlib_AlgHom_id___closed__0));
lean_inc_ref_n(v_inst_484_, 2);
lean_inc_ref_n(v_inst_483_, 2);
v___x_486_ = lean_alloc_closure((void*)(lp_mathlib_AlgHom_comp___boxed), 13, 11);
lean_closure_set(v___x_486_, 0, lean_box(0));
lean_closure_set(v___x_486_, 1, lean_box(0));
lean_closure_set(v___x_486_, 2, lean_box(0));
lean_closure_set(v___x_486_, 3, lean_box(0));
lean_closure_set(v___x_486_, 4, v_inst_482_);
lean_closure_set(v___x_486_, 5, v_inst_483_);
lean_closure_set(v___x_486_, 6, v_inst_483_);
lean_closure_set(v___x_486_, 7, v_inst_483_);
lean_closure_set(v___x_486_, 8, v_inst_484_);
lean_closure_set(v___x_486_, 9, v_inst_484_);
lean_closure_set(v___x_486_, 10, v_inst_484_);
lean_inc_ref(v___x_486_);
v___x_487_ = lean_alloc_closure((void*)(lp_mathlib_npowBinRecAuto___boxed), 5, 3);
lean_closure_set(v___x_487_, 0, lean_box(0));
lean_closure_set(v___x_487_, 1, v___x_486_);
lean_closure_set(v___x_487_, 2, v___f_485_);
v___x_488_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_488_, 0, v___f_485_);
lean_ctor_set(v___x_488_, 1, v___x_486_);
lean_ctor_set(v___x_488_, 2, v___x_487_);
return v___x_488_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_End(lean_object* v_R_489_, lean_object* v_A_490_, lean_object* v_inst_491_, lean_object* v_inst_492_, lean_object* v_inst_493_){
_start:
{
lean_object* v___x_494_; 
v___x_494_ = lp_mathlib_AlgHom_End___redArg(v_inst_491_, v_inst_492_, v_inst_493_);
return v___x_494_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_toEnd___redArg(lean_object* v_inst_495_, lean_object* v_inst_496_, lean_object* v_inst_497_){
_start:
{
lean_object* v___x_498_; 
lean_inc_ref(v_inst_497_);
lean_inc_ref(v_inst_496_);
v___x_498_ = lean_alloc_closure((void*)(lp_mathlib_AlgHom_toLinearMap___boxed), 9, 8);
lean_closure_set(v___x_498_, 0, lean_box(0));
lean_closure_set(v___x_498_, 1, lean_box(0));
lean_closure_set(v___x_498_, 2, lean_box(0));
lean_closure_set(v___x_498_, 3, v_inst_495_);
lean_closure_set(v___x_498_, 4, v_inst_496_);
lean_closure_set(v___x_498_, 5, v_inst_496_);
lean_closure_set(v___x_498_, 6, v_inst_497_);
lean_closure_set(v___x_498_, 7, v_inst_497_);
return v___x_498_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_toEnd(lean_object* v_R_499_, lean_object* v_A_500_, lean_object* v_inst_501_, lean_object* v_inst_502_, lean_object* v_inst_503_){
_start:
{
lean_object* v___x_504_; 
lean_inc_ref(v_inst_503_);
lean_inc_ref(v_inst_502_);
v___x_504_ = lean_alloc_closure((void*)(lp_mathlib_AlgHom_toLinearMap___boxed), 9, 8);
lean_closure_set(v___x_504_, 0, lean_box(0));
lean_closure_set(v___x_504_, 1, lean_box(0));
lean_closure_set(v___x_504_, 2, lean_box(0));
lean_closure_set(v___x_504_, 3, v_inst_501_);
lean_closure_set(v___x_504_, 4, v_inst_502_);
lean_closure_set(v___x_504_, 5, v_inst_502_);
lean_closure_set(v___x_504_, 6, v_inst_503_);
lean_closure_set(v___x_504_, 7, v_inst_503_);
return v___x_504_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsScalarTower_toAlgHom___redArg(lean_object* v_inst_505_){
_start:
{
lean_object* v_algebraMap_506_; 
v_algebraMap_506_ = lean_ctor_get(v_inst_505_, 1);
lean_inc(v_algebraMap_506_);
return v_algebraMap_506_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsScalarTower_toAlgHom___redArg___boxed(lean_object* v_inst_507_){
_start:
{
lean_object* v_res_508_; 
v_res_508_ = lp_mathlib_IsScalarTower_toAlgHom___redArg(v_inst_507_);
lean_dec_ref(v_inst_507_);
return v_res_508_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsScalarTower_toAlgHom(lean_object* v_R_509_, lean_object* v_S_510_, lean_object* v_A_511_, lean_object* v_inst_512_, lean_object* v_inst_513_, lean_object* v_inst_514_, lean_object* v_inst_515_, lean_object* v_inst_516_, lean_object* v_inst_517_, lean_object* v_inst_518_){
_start:
{
lean_object* v_algebraMap_519_; 
v_algebraMap_519_ = lean_ctor_get(v_inst_516_, 1);
lean_inc(v_algebraMap_519_);
return v_algebraMap_519_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsScalarTower_toAlgHom___boxed(lean_object* v_R_520_, lean_object* v_S_521_, lean_object* v_A_522_, lean_object* v_inst_523_, lean_object* v_inst_524_, lean_object* v_inst_525_, lean_object* v_inst_526_, lean_object* v_inst_527_, lean_object* v_inst_528_, lean_object* v_inst_529_){
_start:
{
lean_object* v_res_530_; 
v_res_530_ = lp_mathlib_IsScalarTower_toAlgHom(v_R_520_, v_S_521_, v_A_522_, v_inst_523_, v_inst_524_, v_inst_525_, v_inst_526_, v_inst_527_, v_inst_528_, v_inst_529_);
lean_dec_ref(v_inst_528_);
lean_dec_ref(v_inst_527_);
lean_dec_ref(v_inst_526_);
lean_dec_ref(v_inst_525_);
lean_dec_ref(v_inst_524_);
lean_dec_ref(v_inst_523_);
return v_res_530_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra_algHom___redArg(lean_object* v_inst_531_){
_start:
{
lean_object* v_algebraMap_532_; 
v_algebraMap_532_ = lean_ctor_get(v_inst_531_, 1);
lean_inc(v_algebraMap_532_);
return v_algebraMap_532_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra_algHom___redArg___boxed(lean_object* v_inst_533_){
_start:
{
lean_object* v_res_534_; 
v_res_534_ = lp_mathlib_Algebra_algHom___redArg(v_inst_533_);
lean_dec_ref(v_inst_533_);
return v_res_534_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra_algHom(lean_object* v_R_535_, lean_object* v_S_536_, lean_object* v_A_537_, lean_object* v_inst_538_, lean_object* v_inst_539_, lean_object* v_inst_540_, lean_object* v_inst_541_, lean_object* v_inst_542_, lean_object* v_inst_543_, lean_object* v_inst_544_){
_start:
{
lean_object* v_algebraMap_545_; 
v_algebraMap_545_ = lean_ctor_get(v_inst_542_, 1);
lean_inc(v_algebraMap_545_);
return v_algebraMap_545_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra_algHom___boxed(lean_object* v_R_546_, lean_object* v_S_547_, lean_object* v_A_548_, lean_object* v_inst_549_, lean_object* v_inst_550_, lean_object* v_inst_551_, lean_object* v_inst_552_, lean_object* v_inst_553_, lean_object* v_inst_554_, lean_object* v_inst_555_){
_start:
{
lean_object* v_res_556_; 
v_res_556_ = lp_mathlib_Algebra_algHom(v_R_546_, v_S_547_, v_A_548_, v_inst_549_, v_inst_550_, v_inst_551_, v_inst_552_, v_inst_553_, v_inst_554_, v_inst_555_);
lean_dec_ref(v_inst_554_);
lean_dec_ref(v_inst_553_);
lean_dec_ref(v_inst_552_);
lean_dec_ref(v_inst_551_);
lean_dec_ref(v_inst_550_);
lean_dec_ref(v_inst_549_);
return v_res_556_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_toNatAlgHom___redArg(lean_object* v_f_557_){
_start:
{
lean_object* v___f_558_; 
v___f_558_ = lean_alloc_closure((void*)(lp_mathlib_AlgHom_toMonoidHom_x27___redArg___lam__0), 2, 1);
lean_closure_set(v___f_558_, 0, v_f_557_);
return v___f_558_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_toNatAlgHom(lean_object* v_R_559_, lean_object* v_S_560_, lean_object* v_inst_561_, lean_object* v_inst_562_, lean_object* v_f_563_){
_start:
{
lean_object* v___f_564_; 
v___f_564_ = lean_alloc_closure((void*)(lp_mathlib_AlgHom_toMonoidHom_x27___redArg___lam__0), 2, 1);
lean_closure_set(v___f_564_, 0, v_f_563_);
return v___f_564_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_toNatAlgHom___boxed(lean_object* v_R_565_, lean_object* v_S_566_, lean_object* v_inst_567_, lean_object* v_inst_568_, lean_object* v_f_569_){
_start:
{
lean_object* v_res_570_; 
v_res_570_ = lp_mathlib_RingHom_toNatAlgHom(v_R_565_, v_S_566_, v_inst_567_, v_inst_568_, v_f_569_);
lean_dec_ref(v_inst_568_);
lean_dec_ref(v_inst_567_);
return v_res_570_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_equivNatAlgHom___redArg___lam__0(lean_object* v_self_571_, lean_object* v___y_572_){
_start:
{
lean_object* v___x_573_; 
v___x_573_ = lean_apply_1(v_self_571_, v___y_572_);
return v___x_573_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_equivNatAlgHom___redArg(lean_object* v_inst_575_, lean_object* v_inst_576_){
_start:
{
lean_object* v___f_577_; lean_object* v___x_578_; lean_object* v___x_579_; 
v___f_577_ = ((lean_object*)(lp_mathlib_RingHom_equivNatAlgHom___redArg___closed__0));
v___x_578_ = lean_alloc_closure((void*)(lp_mathlib_RingHom_toNatAlgHom___boxed), 5, 4);
lean_closure_set(v___x_578_, 0, lean_box(0));
lean_closure_set(v___x_578_, 1, lean_box(0));
lean_closure_set(v___x_578_, 2, v_inst_575_);
lean_closure_set(v___x_578_, 3, v_inst_576_);
v___x_579_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_579_, 0, v___x_578_);
lean_ctor_set(v___x_579_, 1, v___f_577_);
return v___x_579_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_equivNatAlgHom(lean_object* v_R_580_, lean_object* v_S_581_, lean_object* v_inst_582_, lean_object* v_inst_583_){
_start:
{
lean_object* v___x_584_; 
v___x_584_ = lp_mathlib_RingHom_equivNatAlgHom___redArg(v_inst_582_, v_inst_583_);
return v___x_584_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_toIntAlgHom___redArg(lean_object* v_f_585_){
_start:
{
lean_inc(v_f_585_);
return v_f_585_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_toIntAlgHom___redArg___boxed(lean_object* v_f_586_){
_start:
{
lean_object* v_res_587_; 
v_res_587_ = lp_mathlib_RingHom_toIntAlgHom___redArg(v_f_586_);
lean_dec(v_f_586_);
return v_res_587_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_toIntAlgHom(lean_object* v_R_588_, lean_object* v_S_589_, lean_object* v_inst_590_, lean_object* v_inst_591_, lean_object* v_f_592_){
_start:
{
lean_inc(v_f_592_);
return v_f_592_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_toIntAlgHom___boxed(lean_object* v_R_593_, lean_object* v_S_594_, lean_object* v_inst_595_, lean_object* v_inst_596_, lean_object* v_f_597_){
_start:
{
lean_object* v_res_598_; 
v_res_598_ = lp_mathlib_RingHom_toIntAlgHom(v_R_593_, v_S_594_, v_inst_595_, v_inst_596_, v_f_597_);
lean_dec(v_f_597_);
lean_dec_ref(v_inst_596_);
lean_dec_ref(v_inst_595_);
return v_res_598_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_equivIntAlgHom___redArg(lean_object* v_inst_599_, lean_object* v_inst_600_){
_start:
{
lean_object* v___f_601_; lean_object* v___x_602_; lean_object* v___x_603_; 
v___f_601_ = ((lean_object*)(lp_mathlib_RingHom_equivNatAlgHom___redArg___closed__0));
v___x_602_ = lean_alloc_closure((void*)(lp_mathlib_RingHom_toIntAlgHom___boxed), 5, 4);
lean_closure_set(v___x_602_, 0, lean_box(0));
lean_closure_set(v___x_602_, 1, lean_box(0));
lean_closure_set(v___x_602_, 2, v_inst_599_);
lean_closure_set(v___x_602_, 3, v_inst_600_);
v___x_603_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_603_, 0, v___x_602_);
lean_ctor_set(v___x_603_, 1, v___f_601_);
return v___x_603_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_equivIntAlgHom(lean_object* v_R_604_, lean_object* v_S_605_, lean_object* v_inst_606_, lean_object* v_inst_607_){
_start:
{
lean_object* v___x_608_; 
v___x_608_ = lp_mathlib_RingHom_equivIntAlgHom___redArg(v_inst_606_, v_inst_607_);
return v___x_608_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra_ofId___redArg(lean_object* v_inst_609_){
_start:
{
lean_object* v_algebraMap_610_; 
v_algebraMap_610_ = lean_ctor_get(v_inst_609_, 1);
lean_inc(v_algebraMap_610_);
return v_algebraMap_610_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra_ofId___redArg___boxed(lean_object* v_inst_611_){
_start:
{
lean_object* v_res_612_; 
v_res_612_ = lp_mathlib_Algebra_ofId___redArg(v_inst_611_);
lean_dec_ref(v_inst_611_);
return v_res_612_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra_ofId(lean_object* v_R_613_, lean_object* v_A_614_, lean_object* v_inst_615_, lean_object* v_inst_616_, lean_object* v_inst_617_){
_start:
{
lean_object* v_algebraMap_618_; 
v_algebraMap_618_ = lean_ctor_get(v_inst_617_, 1);
lean_inc(v_algebraMap_618_);
return v_algebraMap_618_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra_ofId___boxed(lean_object* v_R_619_, lean_object* v_A_620_, lean_object* v_inst_621_, lean_object* v_inst_622_, lean_object* v_inst_623_){
_start:
{
lean_object* v_res_624_; 
v_res_624_ = lp_mathlib_Algebra_ofId(v_R_619_, v_A_620_, v_inst_621_, v_inst_622_, v_inst_623_);
lean_dec_ref(v_inst_623_);
lean_dec_ref(v_inst_622_);
lean_dec_ref(v_inst_621_);
return v_res_624_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra_instMulDistribMulActionAlgHomUnits___lam__1(lean_object* v_f_625_, lean_object* v___y_626_){
_start:
{
lean_object* v___f_627_; lean_object* v___x_628_; 
v___f_627_ = lean_alloc_closure((void*)(lp_mathlib_AlgHom_toMonoidHom_x27___redArg___lam__0), 2, 1);
lean_closure_set(v___f_627_, 0, v_f_625_);
v___x_628_ = lp_mathlib_Units_map___redArg___lam__0(v___f_627_, v___y_626_);
return v___x_628_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra_instMulDistribMulActionAlgHomUnits(lean_object* v_R_630_, lean_object* v_A_631_, lean_object* v_inst_632_, lean_object* v_inst_633_, lean_object* v_inst_634_){
_start:
{
lean_object* v___f_635_; 
v___f_635_ = ((lean_object*)(lp_mathlib_Algebra_instMulDistribMulActionAlgHomUnits___closed__0));
return v___f_635_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra_instMulDistribMulActionAlgHomUnits___boxed(lean_object* v_R_636_, lean_object* v_A_637_, lean_object* v_inst_638_, lean_object* v_inst_639_, lean_object* v_inst_640_){
_start:
{
lean_object* v_res_641_; 
v_res_641_ = lp_mathlib_Algebra_instMulDistribMulActionAlgHomUnits(v_R_636_, v_A_637_, v_inst_638_, v_inst_639_, v_inst_640_);
lean_dec_ref(v_inst_640_);
lean_dec_ref(v_inst_639_);
lean_dec_ref(v_inst_638_);
return v_res_641_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulSemiringAction_toAlgHom___redArg___lam__0(lean_object* v_inst_642_, lean_object* v_m_643_, lean_object* v_a_644_){
_start:
{
lean_object* v___x_645_; 
v___x_645_ = lean_apply_2(v_inst_642_, v_m_643_, v_a_644_);
return v___x_645_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulSemiringAction_toAlgHom___redArg(lean_object* v_inst_646_, lean_object* v_m_647_){
_start:
{
lean_object* v___f_648_; 
v___f_648_ = lean_alloc_closure((void*)(lp_mathlib_MulSemiringAction_toAlgHom___redArg___lam__0), 3, 2);
lean_closure_set(v___f_648_, 0, v_inst_646_);
lean_closure_set(v___f_648_, 1, v_m_647_);
return v___f_648_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulSemiringAction_toAlgHom(lean_object* v_M_649_, lean_object* v_R_650_, lean_object* v_A_651_, lean_object* v_inst_652_, lean_object* v_inst_653_, lean_object* v_inst_654_, lean_object* v_inst_655_, lean_object* v_inst_656_, lean_object* v_inst_657_, lean_object* v_m_658_){
_start:
{
lean_object* v___f_659_; 
v___f_659_ = lean_alloc_closure((void*)(lp_mathlib_MulSemiringAction_toAlgHom___redArg___lam__0), 3, 2);
lean_closure_set(v___f_659_, 0, v_inst_656_);
lean_closure_set(v___f_659_, 1, v_m_658_);
return v___f_659_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulSemiringAction_toAlgHom___boxed(lean_object* v_M_660_, lean_object* v_R_661_, lean_object* v_A_662_, lean_object* v_inst_663_, lean_object* v_inst_664_, lean_object* v_inst_665_, lean_object* v_inst_666_, lean_object* v_inst_667_, lean_object* v_inst_668_, lean_object* v_m_669_){
_start:
{
lean_object* v_res_670_; 
v_res_670_ = lp_mathlib_MulSemiringAction_toAlgHom(v_M_660_, v_R_661_, v_A_662_, v_inst_663_, v_inst_664_, v_inst_665_, v_inst_666_, v_inst_667_, v_inst_668_, v_m_669_);
lean_dec_ref(v_inst_666_);
lean_dec_ref(v_inst_665_);
lean_dec_ref(v_inst_664_);
lean_dec_ref(v_inst_663_);
return v_res_670_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_uniqueOfRight___redArg___lam__0(lean_object* v_toZero_671_, lean_object* v_x_672_){
_start:
{
lean_inc(v_toZero_671_);
return v_toZero_671_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_uniqueOfRight___redArg___lam__0___boxed(lean_object* v_toZero_673_, lean_object* v_x_674_){
_start:
{
lean_object* v_res_675_; 
v_res_675_ = lp_mathlib_uniqueOfRight___redArg___lam__0(v_toZero_673_, v_x_674_);
lean_dec(v_x_674_);
lean_dec(v_toZero_673_);
return v_res_675_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_uniqueOfRight___redArg(lean_object* v_inst_676_){
_start:
{
lean_object* v_toAddCommMonoid_677_; lean_object* v___x_678_; lean_object* v___x_679_; lean_object* v_toZero_680_; lean_object* v___f_681_; lean_object* v___f_682_; 
v_toAddCommMonoid_677_ = lean_ctor_get(v_inst_676_, 0);
v___x_678_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_toAddCommMonoid_677_);
v___x_679_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_678_);
v_toZero_680_ = lean_ctor_get(v___x_679_, 0);
lean_inc(v_toZero_680_);
lean_dec_ref(v___x_679_);
v___f_681_ = lean_alloc_closure((void*)(lp_mathlib_uniqueOfRight___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_681_, 0, v_toZero_680_);
v___f_682_ = lean_alloc_closure((void*)(lp_mathlib_AlgHom_toMonoidHom_x27___redArg___lam__0), 2, 1);
lean_closure_set(v___f_682_, 0, v___f_681_);
return v___f_682_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_uniqueOfRight___redArg___boxed(lean_object* v_inst_683_){
_start:
{
lean_object* v_res_684_; 
v_res_684_ = lp_mathlib_uniqueOfRight___redArg(v_inst_683_);
lean_dec_ref(v_inst_683_);
return v_res_684_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_uniqueOfRight(lean_object* v_R_685_, lean_object* v_S_686_, lean_object* v_T_687_, lean_object* v_inst_688_, lean_object* v_inst_689_, lean_object* v_inst_690_, lean_object* v_inst_691_, lean_object* v_inst_692_, lean_object* v_inst_693_){
_start:
{
lean_object* v___x_694_; 
v___x_694_ = lp_mathlib_uniqueOfRight___redArg(v_inst_690_);
return v___x_694_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_uniqueOfRight___boxed(lean_object* v_R_695_, lean_object* v_S_696_, lean_object* v_T_697_, lean_object* v_inst_698_, lean_object* v_inst_699_, lean_object* v_inst_700_, lean_object* v_inst_701_, lean_object* v_inst_702_, lean_object* v_inst_703_){
_start:
{
lean_object* v_res_704_; 
v_res_704_ = lp_mathlib_uniqueOfRight(v_R_695_, v_S_696_, v_T_697_, v_inst_698_, v_inst_699_, v_inst_700_, v_inst_701_, v_inst_702_, v_inst_703_);
lean_dec_ref(v_inst_702_);
lean_dec_ref(v_inst_701_);
lean_dec_ref(v_inst_700_);
lean_dec_ref(v_inst_699_);
lean_dec_ref(v_inst_698_);
return v_res_704_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Algebra_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Algebra_Hom(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Algebra_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Algebra_Hom(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Algebra_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Algebra_Hom(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Algebra_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Algebra_Hom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Algebra_Hom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Algebra_Hom(builtin);
}
#ifdef __cplusplus
}
#endif
