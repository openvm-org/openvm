// Lean compiler output
// Module: Mathlib.Algebra.Algebra.NonUnitalHom
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Algebra.Hom public import Mathlib.Algebra.GroupWithZero.Action.Prod
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
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Function_prod(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_AddZeroClass_toAddZero___redArg(lean_object*);
lean_object* l_id___boxed(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesIdent(lean_object*, lean_object*);
lean_object* lp_mathlib_OneHom_comp___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_toMulHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_toMulHom___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_toMulHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_toMulHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_term___u2192_u2099_u2090___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 9, .m_data = "term_→ₙₐ_"};
static const lean_object* lp_mathlib_term___u2192_u2099_u2090___00__closed__0 = (const lean_object*)&lp_mathlib_term___u2192_u2099_u2090___00__closed__0_value;
static const lean_ctor_object lp_mathlib_term___u2192_u2099_u2090___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2192_u2099_u2090___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(184, 19, 197, 230, 168, 95, 123, 24)}};
static const lean_object* lp_mathlib_term___u2192_u2099_u2090___00__closed__1 = (const lean_object*)&lp_mathlib_term___u2192_u2099_u2090___00__closed__1_value;
static const lean_string_object lp_mathlib_term___u2192_u2099_u2090___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_term___u2192_u2099_u2090___00__closed__2 = (const lean_object*)&lp_mathlib_term___u2192_u2099_u2090___00__closed__2_value;
static const lean_ctor_object lp_mathlib_term___u2192_u2099_u2090___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2192_u2099_u2090___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_term___u2192_u2099_u2090___00__closed__3 = (const lean_object*)&lp_mathlib_term___u2192_u2099_u2090___00__closed__3_value;
static const lean_string_object lp_mathlib_term___u2192_u2099_u2090___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 5, .m_data = " →ₙₐ "};
static const lean_object* lp_mathlib_term___u2192_u2099_u2090___00__closed__4 = (const lean_object*)&lp_mathlib_term___u2192_u2099_u2090___00__closed__4_value;
static const lean_ctor_object lp_mathlib_term___u2192_u2099_u2090___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192_u2099_u2090___00__closed__4_value)}};
static const lean_object* lp_mathlib_term___u2192_u2099_u2090___00__closed__5 = (const lean_object*)&lp_mathlib_term___u2192_u2099_u2090___00__closed__5_value;
static const lean_string_object lp_mathlib_term___u2192_u2099_u2090___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_term___u2192_u2099_u2090___00__closed__6 = (const lean_object*)&lp_mathlib_term___u2192_u2099_u2090___00__closed__6_value;
static const lean_ctor_object lp_mathlib_term___u2192_u2099_u2090___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2192_u2099_u2090___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_term___u2192_u2099_u2090___00__closed__7 = (const lean_object*)&lp_mathlib_term___u2192_u2099_u2090___00__closed__7_value;
static const lean_ctor_object lp_mathlib_term___u2192_u2099_u2090___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192_u2099_u2090___00__closed__7_value),((lean_object*)(((size_t)(25) << 1) | 1))}};
static const lean_object* lp_mathlib_term___u2192_u2099_u2090___00__closed__8 = (const lean_object*)&lp_mathlib_term___u2192_u2099_u2090___00__closed__8_value;
static const lean_ctor_object lp_mathlib_term___u2192_u2099_u2090___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192_u2099_u2090___00__closed__3_value),((lean_object*)&lp_mathlib_term___u2192_u2099_u2090___00__closed__5_value),((lean_object*)&lp_mathlib_term___u2192_u2099_u2090___00__closed__8_value)}};
static const lean_object* lp_mathlib_term___u2192_u2099_u2090___00__closed__9 = (const lean_object*)&lp_mathlib_term___u2192_u2099_u2090___00__closed__9_value;
static const lean_ctor_object lp_mathlib_term___u2192_u2099_u2090___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192_u2099_u2090___00__closed__1_value),((lean_object*)(((size_t)(25) << 1) | 1)),((lean_object*)(((size_t)(26) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2192_u2099_u2090___00__closed__9_value)}};
static const lean_object* lp_mathlib_term___u2192_u2099_u2090___00__closed__10 = (const lean_object*)&lp_mathlib_term___u2192_u2099_u2090___00__closed__10_value;
LEAN_EXPORT const lean_object* lp_mathlib_term___u2192_u2099_u2090__ = (const lean_object*)&lp_mathlib_term___u2192_u2099_u2090___00__closed__10_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__0_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__1_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__2_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__3_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__4_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__4_value_aux_1),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__4_value_aux_2),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__4 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__4_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "NonUnitalAlgHom"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__5 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__5_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__6;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(1, 22, 162, 153, 41, 125, 108, 103)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__7 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__7_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__7_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__8 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__8_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__7_value)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__9 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__9_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__10 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__10_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__8_value),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__10_value)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__11 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__11_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__12 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__12_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__12_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__13 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__13_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "hole"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__14 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__14_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__15_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__15_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__15_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__15_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__15_value_aux_1),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__15_value_aux_2),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__14_value),LEAN_SCALAR_PTR_LITERAL(135, 134, 219, 115, 97, 130, 74, 55)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__15 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__15_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "_"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__16 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__16_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_term___u2192_u209b_u2099_u2090_x5b___x5d___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 22, .m_capacity = 22, .m_length = 13, .m_data = "term_→ₛₙₐ[_]_"};
static const lean_object* lp_mathlib_term___u2192_u209b_u2099_u2090_x5b___x5d___00__closed__0 = (const lean_object*)&lp_mathlib_term___u2192_u209b_u2099_u2090_x5b___x5d___00__closed__0_value;
static const lean_ctor_object lp_mathlib_term___u2192_u209b_u2099_u2090_x5b___x5d___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2192_u209b_u2099_u2090_x5b___x5d___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(222, 64, 232, 67, 129, 150, 65, 112)}};
static const lean_object* lp_mathlib_term___u2192_u209b_u2099_u2090_x5b___x5d___00__closed__1 = (const lean_object*)&lp_mathlib_term___u2192_u209b_u2099_u2090_x5b___x5d___00__closed__1_value;
static const lean_string_object lp_mathlib_term___u2192_u209b_u2099_u2090_x5b___x5d___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 6, .m_data = " →ₛₙₐ["};
static const lean_object* lp_mathlib_term___u2192_u209b_u2099_u2090_x5b___x5d___00__closed__2 = (const lean_object*)&lp_mathlib_term___u2192_u209b_u2099_u2090_x5b___x5d___00__closed__2_value;
static const lean_ctor_object lp_mathlib_term___u2192_u209b_u2099_u2090_x5b___x5d___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192_u209b_u2099_u2090_x5b___x5d___00__closed__2_value)}};
static const lean_object* lp_mathlib_term___u2192_u209b_u2099_u2090_x5b___x5d___00__closed__3 = (const lean_object*)&lp_mathlib_term___u2192_u209b_u2099_u2090_x5b___x5d___00__closed__3_value;
static const lean_ctor_object lp_mathlib_term___u2192_u209b_u2099_u2090_x5b___x5d___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192_u2099_u2090___00__closed__7_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_term___u2192_u209b_u2099_u2090_x5b___x5d___00__closed__4 = (const lean_object*)&lp_mathlib_term___u2192_u209b_u2099_u2090_x5b___x5d___00__closed__4_value;
static const lean_ctor_object lp_mathlib_term___u2192_u209b_u2099_u2090_x5b___x5d___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192_u2099_u2090___00__closed__3_value),((lean_object*)&lp_mathlib_term___u2192_u209b_u2099_u2090_x5b___x5d___00__closed__3_value),((lean_object*)&lp_mathlib_term___u2192_u209b_u2099_u2090_x5b___x5d___00__closed__4_value)}};
static const lean_object* lp_mathlib_term___u2192_u209b_u2099_u2090_x5b___x5d___00__closed__5 = (const lean_object*)&lp_mathlib_term___u2192_u209b_u2099_u2090_x5b___x5d___00__closed__5_value;
static const lean_string_object lp_mathlib_term___u2192_u209b_u2099_u2090_x5b___x5d___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "] "};
static const lean_object* lp_mathlib_term___u2192_u209b_u2099_u2090_x5b___x5d___00__closed__6 = (const lean_object*)&lp_mathlib_term___u2192_u209b_u2099_u2090_x5b___x5d___00__closed__6_value;
static const lean_ctor_object lp_mathlib_term___u2192_u209b_u2099_u2090_x5b___x5d___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192_u209b_u2099_u2090_x5b___x5d___00__closed__6_value)}};
static const lean_object* lp_mathlib_term___u2192_u209b_u2099_u2090_x5b___x5d___00__closed__7 = (const lean_object*)&lp_mathlib_term___u2192_u209b_u2099_u2090_x5b___x5d___00__closed__7_value;
static const lean_ctor_object lp_mathlib_term___u2192_u209b_u2099_u2090_x5b___x5d___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192_u2099_u2090___00__closed__3_value),((lean_object*)&lp_mathlib_term___u2192_u209b_u2099_u2090_x5b___x5d___00__closed__5_value),((lean_object*)&lp_mathlib_term___u2192_u209b_u2099_u2090_x5b___x5d___00__closed__7_value)}};
static const lean_object* lp_mathlib_term___u2192_u209b_u2099_u2090_x5b___x5d___00__closed__8 = (const lean_object*)&lp_mathlib_term___u2192_u209b_u2099_u2090_x5b___x5d___00__closed__8_value;
static const lean_ctor_object lp_mathlib_term___u2192_u209b_u2099_u2090_x5b___x5d___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192_u2099_u2090___00__closed__3_value),((lean_object*)&lp_mathlib_term___u2192_u209b_u2099_u2090_x5b___x5d___00__closed__8_value),((lean_object*)&lp_mathlib_term___u2192_u209b_u2099_u2090_x5b___x5d___00__closed__4_value)}};
static const lean_object* lp_mathlib_term___u2192_u209b_u2099_u2090_x5b___x5d___00__closed__9 = (const lean_object*)&lp_mathlib_term___u2192_u209b_u2099_u2090_x5b___x5d___00__closed__9_value;
static const lean_ctor_object lp_mathlib_term___u2192_u209b_u2099_u2090_x5b___x5d___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192_u209b_u2099_u2090_x5b___x5d___00__closed__1_value),((lean_object*)(((size_t)(25) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2192_u209b_u2099_u2090_x5b___x5d___00__closed__9_value)}};
static const lean_object* lp_mathlib_term___u2192_u209b_u2099_u2090_x5b___x5d___00__closed__10 = (const lean_object*)&lp_mathlib_term___u2192_u209b_u2099_u2090_x5b___x5d___00__closed__10_value;
LEAN_EXPORT const lean_object* lp_mathlib_term___u2192_u209b_u2099_u2090_x5b___x5d__ = (const lean_object*)&lp_mathlib_term___u2192_u209b_u2099_u2090_x5b___x5d___00__closed__10_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u209b_u2099_u2090_x5b___x5d____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u209b_u2099_u2090_x5b___x5d____1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______unexpand__NonUnitalAlgHom__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______unexpand__NonUnitalAlgHom__1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______unexpand__NonUnitalAlgHom__1___closed__0_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______unexpand__NonUnitalAlgHom__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______unexpand__NonUnitalAlgHom__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______unexpand__NonUnitalAlgHom__1___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______unexpand__NonUnitalAlgHom__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______unexpand__NonUnitalAlgHom__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______unexpand__NonUnitalAlgHom__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_term___u2192_u2099_u2090_x5b___x5d___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 12, .m_data = "term_→ₙₐ[_]_"};
static const lean_object* lp_mathlib_term___u2192_u2099_u2090_x5b___x5d___00__closed__0 = (const lean_object*)&lp_mathlib_term___u2192_u2099_u2090_x5b___x5d___00__closed__0_value;
static const lean_ctor_object lp_mathlib_term___u2192_u2099_u2090_x5b___x5d___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2192_u2099_u2090_x5b___x5d___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(71, 130, 232, 177, 159, 201, 163, 147)}};
static const lean_object* lp_mathlib_term___u2192_u2099_u2090_x5b___x5d___00__closed__1 = (const lean_object*)&lp_mathlib_term___u2192_u2099_u2090_x5b___x5d___00__closed__1_value;
static const lean_string_object lp_mathlib_term___u2192_u2099_u2090_x5b___x5d___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 5, .m_data = " →ₙₐ["};
static const lean_object* lp_mathlib_term___u2192_u2099_u2090_x5b___x5d___00__closed__2 = (const lean_object*)&lp_mathlib_term___u2192_u2099_u2090_x5b___x5d___00__closed__2_value;
static const lean_ctor_object lp_mathlib_term___u2192_u2099_u2090_x5b___x5d___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192_u2099_u2090_x5b___x5d___00__closed__2_value)}};
static const lean_object* lp_mathlib_term___u2192_u2099_u2090_x5b___x5d___00__closed__3 = (const lean_object*)&lp_mathlib_term___u2192_u2099_u2090_x5b___x5d___00__closed__3_value;
static const lean_ctor_object lp_mathlib_term___u2192_u2099_u2090_x5b___x5d___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192_u2099_u2090___00__closed__3_value),((lean_object*)&lp_mathlib_term___u2192_u2099_u2090_x5b___x5d___00__closed__3_value),((lean_object*)&lp_mathlib_term___u2192_u209b_u2099_u2090_x5b___x5d___00__closed__4_value)}};
static const lean_object* lp_mathlib_term___u2192_u2099_u2090_x5b___x5d___00__closed__4 = (const lean_object*)&lp_mathlib_term___u2192_u2099_u2090_x5b___x5d___00__closed__4_value;
static const lean_ctor_object lp_mathlib_term___u2192_u2099_u2090_x5b___x5d___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192_u2099_u2090___00__closed__3_value),((lean_object*)&lp_mathlib_term___u2192_u2099_u2090_x5b___x5d___00__closed__4_value),((lean_object*)&lp_mathlib_term___u2192_u209b_u2099_u2090_x5b___x5d___00__closed__7_value)}};
static const lean_object* lp_mathlib_term___u2192_u2099_u2090_x5b___x5d___00__closed__5 = (const lean_object*)&lp_mathlib_term___u2192_u2099_u2090_x5b___x5d___00__closed__5_value;
static const lean_ctor_object lp_mathlib_term___u2192_u2099_u2090_x5b___x5d___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192_u2099_u2090___00__closed__3_value),((lean_object*)&lp_mathlib_term___u2192_u2099_u2090_x5b___x5d___00__closed__5_value),((lean_object*)&lp_mathlib_term___u2192_u209b_u2099_u2090_x5b___x5d___00__closed__4_value)}};
static const lean_object* lp_mathlib_term___u2192_u2099_u2090_x5b___x5d___00__closed__6 = (const lean_object*)&lp_mathlib_term___u2192_u2099_u2090_x5b___x5d___00__closed__6_value;
static const lean_ctor_object lp_mathlib_term___u2192_u2099_u2090_x5b___x5d___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192_u2099_u2090_x5b___x5d___00__closed__1_value),((lean_object*)(((size_t)(25) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2192_u2099_u2090_x5b___x5d___00__closed__6_value)}};
static const lean_object* lp_mathlib_term___u2192_u2099_u2090_x5b___x5d___00__closed__7 = (const lean_object*)&lp_mathlib_term___u2192_u2099_u2090_x5b___x5d___00__closed__7_value;
LEAN_EXPORT const lean_object* lp_mathlib_term___u2192_u2099_u2090_x5b___x5d__ = (const lean_object*)&lp_mathlib_term___u2192_u2099_u2090_x5b___x5d___00__closed__7_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "paren"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1___closed__0_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1___closed__1_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1___closed__1_value_aux_1),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1___closed__1_value_aux_2),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(124, 9, 161, 194, 227, 100, 20, 110)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1___closed__1_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "hygienicLParen"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1___closed__2_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1___closed__3_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1___closed__3_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1___closed__3_value_aux_1),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1___closed__3_value_aux_2),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(41, 104, 206, 51, 21, 254, 100, 101)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1___closed__3_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "("};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1___closed__4 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1___closed__4_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "hygieneInfo"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1___closed__5 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1___closed__5_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(27, 64, 36, 144, 170, 151, 255, 136)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1___closed__6 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1___closed__6_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1___closed__7 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1___closed__7_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1___closed__8;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1___closed__9 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1___closed__9_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1___closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1___closed__10 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1___closed__10_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "MonoidHom.id"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1___closed__11 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1___closed__11_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1___closed__12;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "MonoidHom"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1___closed__13 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1___closed__13_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "id"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1___closed__14 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1___closed__14_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1___closed__15_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1___closed__13_value),LEAN_SCALAR_PTR_LITERAL(42, 146, 241, 17, 119, 0, 235, 30)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1___closed__15_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1___closed__14_value),LEAN_SCALAR_PTR_LITERAL(26, 88, 52, 83, 162, 234, 201, 148)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1___closed__15 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1___closed__15_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1___closed__15_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1___closed__16 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1___closed__16_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1___closed__16_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1___closed__17 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1___closed__17_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ")"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1___closed__18 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1___closed__18_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______unexpand__NonUnitalAlgHom__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______unexpand__NonUnitalAlgHom__2___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHomClass_toNonUnitalAlgSemiHom___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHomClass_toNonUnitalAlgSemiHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHomClass_toNonUnitalAlgSemiHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHomClass_instCoeTCNonUnitalAlgHomOfNonUnitalAlgSemiHomClass___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHomClass_instCoeTCNonUnitalAlgHomOfNonUnitalAlgSemiHomClass(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHomClass_toNonUnitalAlgHom___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHomClass_toNonUnitalAlgHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHomClass_toNonUnitalAlgHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHomClass_instCoeTCNonUnitalAlgHomId___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHomClass_instCoeTCNonUnitalAlgHomId(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_Simps_apply___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_Simps_apply(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_Simps_apply___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_NonUnitalAlgHom_id___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_id___boxed, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_NonUnitalAlgHom_id___closed__0 = (const lean_object*)&lp_mathlib_NonUnitalAlgHom_id___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_id(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_id___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_instZero___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_instZero___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_instZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_instZero___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_instZero(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_instZero___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_instOneId(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_instOneId___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_instInhabited___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_instInhabited___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_instInhabited(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_instInhabited___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_comp___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_comp___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_comp___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_comp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_comp___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_inverse___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_inverse___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_inverse(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_inverse___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_inverse_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_inverse_x27___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_inverse_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_inverse_x27___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_fst___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_fst___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_NonUnitalAlgHom_fst___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_NonUnitalAlgHom_fst___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_NonUnitalAlgHom_fst___closed__0 = (const lean_object*)&lp_mathlib_NonUnitalAlgHom_fst___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_fst(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_fst___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_snd___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_snd___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_NonUnitalAlgHom_snd___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_NonUnitalAlgHom_snd___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_NonUnitalAlgHom_snd___closed__0 = (const lean_object*)&lp_mathlib_NonUnitalAlgHom_snd___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_snd(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_snd___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_prod___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_prod(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_prod___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_prodEquiv___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_prodEquiv___lam__1(lean_object*);
static const lean_closure_object lp_mathlib_NonUnitalAlgHom_prodEquiv___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_NonUnitalAlgHom_prodEquiv___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_NonUnitalAlgHom_prodEquiv___closed__0 = (const lean_object*)&lp_mathlib_NonUnitalAlgHom_prodEquiv___closed__0_value;
static const lean_closure_object lp_mathlib_NonUnitalAlgHom_prodEquiv___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_NonUnitalAlgHom_prodEquiv___lam__1, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_NonUnitalAlgHom_prodEquiv___closed__1 = (const lean_object*)&lp_mathlib_NonUnitalAlgHom_prodEquiv___closed__1_value;
static const lean_ctor_object lp_mathlib_NonUnitalAlgHom_prodEquiv___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_NonUnitalAlgHom_prodEquiv___closed__0_value),((lean_object*)&lp_mathlib_NonUnitalAlgHom_prodEquiv___closed__1_value)}};
static const lean_object* lp_mathlib_NonUnitalAlgHom_prodEquiv___closed__2 = (const lean_object*)&lp_mathlib_NonUnitalAlgHom_prodEquiv___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_prodEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_prodEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_inl___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_inl___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_inl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_inl___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_inr___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_inr___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_inr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_inr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_toNonUnitalAlgHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_toNonUnitalAlgHom___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_toNonUnitalAlgHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_toNonUnitalAlgHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_NonUnitalAlgHom_hasCoe___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_NonUnitalAlgHom_hasCoe(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_restrictScalars___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_restrictScalars(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_restrictScalars___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_toMulHom___redArg(lean_object* v_self_1_){
_start:
{
lean_inc(v_self_1_);
return v_self_1_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_toMulHom___redArg___boxed(lean_object* v_self_2_){
_start:
{
lean_object* v_res_3_; 
v_res_3_ = lp_mathlib_NonUnitalAlgHom_toMulHom___redArg(v_self_2_);
lean_dec(v_self_2_);
return v_res_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_toMulHom(lean_object* v_R_4_, lean_object* v_S_5_, lean_object* v_inst_6_, lean_object* v_inst_7_, lean_object* v_00_u03c6_8_, lean_object* v_A_9_, lean_object* v_B_10_, lean_object* v_inst_11_, lean_object* v_inst_12_, lean_object* v_inst_13_, lean_object* v_inst_14_, lean_object* v_self_15_){
_start:
{
lean_inc(v_self_15_);
return v_self_15_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_toMulHom___boxed(lean_object* v_R_16_, lean_object* v_S_17_, lean_object* v_inst_18_, lean_object* v_inst_19_, lean_object* v_00_u03c6_20_, lean_object* v_A_21_, lean_object* v_B_22_, lean_object* v_inst_23_, lean_object* v_inst_24_, lean_object* v_inst_25_, lean_object* v_inst_26_, lean_object* v_self_27_){
_start:
{
lean_object* v_res_28_; 
v_res_28_ = lp_mathlib_NonUnitalAlgHom_toMulHom(v_R_16_, v_S_17_, v_inst_18_, v_inst_19_, v_00_u03c6_20_, v_A_21_, v_B_22_, v_inst_23_, v_inst_24_, v_inst_25_, v_inst_26_, v_self_27_);
lean_dec(v_self_27_);
lean_dec(v_inst_26_);
lean_dec_ref(v_inst_25_);
lean_dec(v_inst_24_);
lean_dec_ref(v_inst_23_);
lean_dec(v_00_u03c6_20_);
lean_dec_ref(v_inst_19_);
lean_dec_ref(v_inst_18_);
return v_res_28_;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__6(void){
_start:
{
lean_object* v___x_64_; lean_object* v___x_65_; 
v___x_64_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__5));
v___x_65_ = l_String_toRawSubstring_x27(v___x_64_);
return v___x_65_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1(lean_object* v_x_89_, lean_object* v_a_90_, lean_object* v_a_91_){
_start:
{
lean_object* v___x_92_; uint8_t v___x_93_; 
v___x_92_ = ((lean_object*)(lp_mathlib_term___u2192_u2099_u2090___00__closed__1));
lean_inc(v_x_89_);
v___x_93_ = l_Lean_Syntax_isOfKind(v_x_89_, v___x_92_);
if (v___x_93_ == 0)
{
lean_object* v___x_94_; lean_object* v___x_95_; 
lean_dec(v_x_89_);
v___x_94_ = lean_box(1);
v___x_95_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_95_, 0, v___x_94_);
lean_ctor_set(v___x_95_, 1, v_a_91_);
return v___x_95_;
}
else
{
lean_object* v_quotContext_96_; lean_object* v_currMacroScope_97_; lean_object* v_ref_98_; lean_object* v___x_99_; lean_object* v___x_100_; lean_object* v___x_101_; lean_object* v___x_102_; uint8_t v___x_103_; lean_object* v___x_104_; lean_object* v___x_105_; lean_object* v___x_106_; lean_object* v___x_107_; lean_object* v___x_108_; lean_object* v___x_109_; lean_object* v___x_110_; lean_object* v___x_111_; lean_object* v___x_112_; lean_object* v___x_113_; lean_object* v___x_114_; lean_object* v___x_115_; lean_object* v___x_116_; lean_object* v___x_117_; lean_object* v___x_118_; lean_object* v___x_119_; lean_object* v___x_120_; 
v_quotContext_96_ = lean_ctor_get(v_a_90_, 1);
v_currMacroScope_97_ = lean_ctor_get(v_a_90_, 2);
v_ref_98_ = lean_ctor_get(v_a_90_, 5);
v___x_99_ = lean_unsigned_to_nat(0u);
v___x_100_ = l_Lean_Syntax_getArg(v_x_89_, v___x_99_);
v___x_101_ = lean_unsigned_to_nat(2u);
v___x_102_ = l_Lean_Syntax_getArg(v_x_89_, v___x_101_);
lean_dec(v_x_89_);
v___x_103_ = 0;
v___x_104_ = l_Lean_SourceInfo_fromRef(v_ref_98_, v___x_103_);
v___x_105_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__4));
v___x_106_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__6, &lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__6_once, _init_lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__6);
v___x_107_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__7));
lean_inc(v_currMacroScope_97_);
lean_inc(v_quotContext_96_);
v___x_108_ = l_Lean_addMacroScope(v_quotContext_96_, v___x_107_, v_currMacroScope_97_);
v___x_109_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__11));
lean_inc_n(v___x_104_, 6);
v___x_110_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_110_, 0, v___x_104_);
lean_ctor_set(v___x_110_, 1, v___x_106_);
lean_ctor_set(v___x_110_, 2, v___x_108_);
lean_ctor_set(v___x_110_, 3, v___x_109_);
v___x_111_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__13));
v___x_112_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__15));
v___x_113_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__16));
v___x_114_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_114_, 0, v___x_104_);
lean_ctor_set(v___x_114_, 1, v___x_113_);
v___x_115_ = l_Lean_Syntax_node1(v___x_104_, v___x_112_, v___x_114_);
v___x_116_ = l_Lean_Syntax_node1(v___x_104_, v___x_111_, v___x_115_);
v___x_117_ = l_Lean_Syntax_node2(v___x_104_, v___x_105_, v___x_110_, v___x_116_);
v___x_118_ = l_Lean_Syntax_node2(v___x_104_, v___x_111_, v___x_100_, v___x_102_);
v___x_119_ = l_Lean_Syntax_node2(v___x_104_, v___x_105_, v___x_117_, v___x_118_);
v___x_120_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_120_, 0, v___x_119_);
lean_ctor_set(v___x_120_, 1, v_a_91_);
return v___x_120_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___boxed(lean_object* v_x_121_, lean_object* v_a_122_, lean_object* v_a_123_){
_start:
{
lean_object* v_res_124_; 
v_res_124_ = lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1(v_x_121_, v_a_122_, v_a_123_);
lean_dec_ref(v_a_122_);
return v_res_124_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u209b_u2099_u2090_x5b___x5d____1(lean_object* v_x_155_, lean_object* v_a_156_, lean_object* v_a_157_){
_start:
{
lean_object* v___x_158_; uint8_t v___x_159_; 
v___x_158_ = ((lean_object*)(lp_mathlib_term___u2192_u209b_u2099_u2090_x5b___x5d___00__closed__1));
lean_inc(v_x_155_);
v___x_159_ = l_Lean_Syntax_isOfKind(v_x_155_, v___x_158_);
if (v___x_159_ == 0)
{
lean_object* v___x_160_; lean_object* v___x_161_; 
lean_dec(v_x_155_);
v___x_160_ = lean_box(1);
v___x_161_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_161_, 0, v___x_160_);
lean_ctor_set(v___x_161_, 1, v_a_157_);
return v___x_161_;
}
else
{
lean_object* v_quotContext_162_; lean_object* v_currMacroScope_163_; lean_object* v_ref_164_; lean_object* v___x_165_; lean_object* v___x_166_; lean_object* v___x_167_; lean_object* v___x_168_; lean_object* v___x_169_; lean_object* v___x_170_; uint8_t v___x_171_; lean_object* v___x_172_; lean_object* v___x_173_; lean_object* v___x_174_; lean_object* v___x_175_; lean_object* v___x_176_; lean_object* v___x_177_; lean_object* v___x_178_; lean_object* v___x_179_; lean_object* v___x_180_; lean_object* v___x_181_; lean_object* v___x_182_; 
v_quotContext_162_ = lean_ctor_get(v_a_156_, 1);
v_currMacroScope_163_ = lean_ctor_get(v_a_156_, 2);
v_ref_164_ = lean_ctor_get(v_a_156_, 5);
v___x_165_ = lean_unsigned_to_nat(0u);
v___x_166_ = l_Lean_Syntax_getArg(v_x_155_, v___x_165_);
v___x_167_ = lean_unsigned_to_nat(2u);
v___x_168_ = l_Lean_Syntax_getArg(v_x_155_, v___x_167_);
v___x_169_ = lean_unsigned_to_nat(4u);
v___x_170_ = l_Lean_Syntax_getArg(v_x_155_, v___x_169_);
lean_dec(v_x_155_);
v___x_171_ = 0;
v___x_172_ = l_Lean_SourceInfo_fromRef(v_ref_164_, v___x_171_);
v___x_173_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__4));
v___x_174_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__6, &lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__6_once, _init_lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__6);
v___x_175_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__7));
lean_inc(v_currMacroScope_163_);
lean_inc(v_quotContext_162_);
v___x_176_ = l_Lean_addMacroScope(v_quotContext_162_, v___x_175_, v_currMacroScope_163_);
v___x_177_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__11));
lean_inc_n(v___x_172_, 2);
v___x_178_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_178_, 0, v___x_172_);
lean_ctor_set(v___x_178_, 1, v___x_174_);
lean_ctor_set(v___x_178_, 2, v___x_176_);
lean_ctor_set(v___x_178_, 3, v___x_177_);
v___x_179_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__13));
v___x_180_ = l_Lean_Syntax_node3(v___x_172_, v___x_179_, v___x_168_, v___x_166_, v___x_170_);
v___x_181_ = l_Lean_Syntax_node2(v___x_172_, v___x_173_, v___x_178_, v___x_180_);
v___x_182_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_182_, 0, v___x_181_);
lean_ctor_set(v___x_182_, 1, v_a_157_);
return v___x_182_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u209b_u2099_u2090_x5b___x5d____1___boxed(lean_object* v_x_183_, lean_object* v_a_184_, lean_object* v_a_185_){
_start:
{
lean_object* v_res_186_; 
v_res_186_ = lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u209b_u2099_u2090_x5b___x5d____1(v_x_183_, v_a_184_, v_a_185_);
lean_dec_ref(v_a_184_);
return v_res_186_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______unexpand__NonUnitalAlgHom__1(lean_object* v_x_190_, lean_object* v_a_191_, lean_object* v_a_192_){
_start:
{
lean_object* v___x_193_; uint8_t v___x_194_; 
v___x_193_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__4));
lean_inc(v_x_190_);
v___x_194_ = l_Lean_Syntax_isOfKind(v_x_190_, v___x_193_);
if (v___x_194_ == 0)
{
lean_object* v___x_195_; lean_object* v___x_196_; 
lean_dec(v_x_190_);
v___x_195_ = lean_box(0);
v___x_196_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_196_, 0, v___x_195_);
lean_ctor_set(v___x_196_, 1, v_a_192_);
return v___x_196_;
}
else
{
lean_object* v___x_197_; lean_object* v___x_198_; lean_object* v___x_199_; uint8_t v___x_200_; 
v___x_197_ = lean_unsigned_to_nat(0u);
v___x_198_ = l_Lean_Syntax_getArg(v_x_190_, v___x_197_);
v___x_199_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______unexpand__NonUnitalAlgHom__1___closed__1));
lean_inc(v___x_198_);
v___x_200_ = l_Lean_Syntax_isOfKind(v___x_198_, v___x_199_);
if (v___x_200_ == 0)
{
lean_object* v___x_201_; lean_object* v___x_202_; 
lean_dec(v___x_198_);
lean_dec(v_x_190_);
v___x_201_ = lean_box(0);
v___x_202_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_202_, 0, v___x_201_);
lean_ctor_set(v___x_202_, 1, v_a_192_);
return v___x_202_;
}
else
{
lean_object* v___x_203_; lean_object* v___x_204_; lean_object* v___x_205_; uint8_t v___x_206_; 
v___x_203_ = lean_unsigned_to_nat(1u);
v___x_204_ = l_Lean_Syntax_getArg(v_x_190_, v___x_203_);
lean_dec(v_x_190_);
v___x_205_ = lean_unsigned_to_nat(3u);
lean_inc(v___x_204_);
v___x_206_ = l_Lean_Syntax_matchesNull(v___x_204_, v___x_205_);
if (v___x_206_ == 0)
{
lean_object* v___x_207_; lean_object* v___x_208_; 
lean_dec(v___x_204_);
lean_dec(v___x_198_);
v___x_207_ = lean_box(0);
v___x_208_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_208_, 0, v___x_207_);
lean_ctor_set(v___x_208_, 1, v_a_192_);
return v___x_208_;
}
else
{
lean_object* v___x_209_; lean_object* v___x_210_; lean_object* v___x_211_; lean_object* v___x_212_; lean_object* v_ref_213_; uint8_t v___x_214_; lean_object* v___x_215_; lean_object* v___x_216_; lean_object* v___x_217_; lean_object* v___x_218_; lean_object* v___x_219_; lean_object* v___x_220_; lean_object* v___x_221_; lean_object* v___x_222_; 
v___x_209_ = l_Lean_Syntax_getArg(v___x_204_, v___x_197_);
v___x_210_ = l_Lean_Syntax_getArg(v___x_204_, v___x_203_);
v___x_211_ = lean_unsigned_to_nat(2u);
v___x_212_ = l_Lean_Syntax_getArg(v___x_204_, v___x_211_);
lean_dec(v___x_204_);
v_ref_213_ = l_Lean_replaceRef(v___x_198_, v_a_191_);
lean_dec(v___x_198_);
v___x_214_ = 0;
v___x_215_ = l_Lean_SourceInfo_fromRef(v_ref_213_, v___x_214_);
lean_dec(v_ref_213_);
v___x_216_ = ((lean_object*)(lp_mathlib_term___u2192_u209b_u2099_u2090_x5b___x5d___00__closed__1));
v___x_217_ = ((lean_object*)(lp_mathlib_term___u2192_u209b_u2099_u2090_x5b___x5d___00__closed__2));
lean_inc_n(v___x_215_, 2);
v___x_218_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_218_, 0, v___x_215_);
lean_ctor_set(v___x_218_, 1, v___x_217_);
v___x_219_ = ((lean_object*)(lp_mathlib_term___u2192_u209b_u2099_u2090_x5b___x5d___00__closed__6));
v___x_220_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_220_, 0, v___x_215_);
lean_ctor_set(v___x_220_, 1, v___x_219_);
v___x_221_ = l_Lean_Syntax_node5(v___x_215_, v___x_216_, v___x_210_, v___x_218_, v___x_209_, v___x_220_, v___x_212_);
v___x_222_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_222_, 0, v___x_221_);
lean_ctor_set(v___x_222_, 1, v_a_192_);
return v___x_222_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______unexpand__NonUnitalAlgHom__1___boxed(lean_object* v_x_223_, lean_object* v_a_224_, lean_object* v_a_225_){
_start:
{
lean_object* v_res_226_; 
v_res_226_ = lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______unexpand__NonUnitalAlgHom__1(v_x_223_, v_a_224_, v_a_225_);
lean_dec(v_a_224_);
return v_res_226_;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1___closed__8(void){
_start:
{
lean_object* v___x_268_; lean_object* v___x_269_; 
v___x_268_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1___closed__7));
v___x_269_ = l_String_toRawSubstring_x27(v___x_268_);
return v___x_269_;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1___closed__12(void){
_start:
{
lean_object* v___x_276_; lean_object* v___x_277_; 
v___x_276_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1___closed__11));
v___x_277_ = l_String_toRawSubstring_x27(v___x_276_);
return v___x_277_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1(lean_object* v_x_290_, lean_object* v_a_291_, lean_object* v_a_292_){
_start:
{
lean_object* v___x_293_; uint8_t v___x_294_; 
v___x_293_ = ((lean_object*)(lp_mathlib_term___u2192_u2099_u2090_x5b___x5d___00__closed__1));
lean_inc(v_x_290_);
v___x_294_ = l_Lean_Syntax_isOfKind(v_x_290_, v___x_293_);
if (v___x_294_ == 0)
{
lean_object* v___x_295_; lean_object* v___x_296_; 
lean_dec(v_x_290_);
v___x_295_ = lean_box(1);
v___x_296_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_296_, 0, v___x_295_);
lean_ctor_set(v___x_296_, 1, v_a_292_);
return v___x_296_;
}
else
{
lean_object* v_quotContext_297_; lean_object* v_currMacroScope_298_; lean_object* v_ref_299_; lean_object* v___x_300_; lean_object* v___x_301_; lean_object* v___x_302_; lean_object* v___x_303_; lean_object* v___x_304_; lean_object* v___x_305_; uint8_t v___x_306_; lean_object* v___x_307_; lean_object* v___x_308_; lean_object* v___x_309_; lean_object* v___x_310_; lean_object* v___x_311_; lean_object* v___x_312_; lean_object* v___x_313_; lean_object* v___x_314_; lean_object* v___x_315_; lean_object* v___x_316_; lean_object* v___x_317_; lean_object* v___x_318_; lean_object* v___x_319_; lean_object* v___x_320_; lean_object* v___x_321_; lean_object* v___x_322_; lean_object* v___x_323_; lean_object* v___x_324_; lean_object* v___x_325_; lean_object* v___x_326_; lean_object* v___x_327_; lean_object* v___x_328_; lean_object* v___x_329_; lean_object* v___x_330_; lean_object* v___x_331_; lean_object* v___x_332_; lean_object* v___x_333_; lean_object* v___x_334_; lean_object* v___x_335_; lean_object* v___x_336_; lean_object* v___x_337_; lean_object* v___x_338_; lean_object* v___x_339_; 
v_quotContext_297_ = lean_ctor_get(v_a_291_, 1);
v_currMacroScope_298_ = lean_ctor_get(v_a_291_, 2);
v_ref_299_ = lean_ctor_get(v_a_291_, 5);
v___x_300_ = lean_unsigned_to_nat(0u);
v___x_301_ = l_Lean_Syntax_getArg(v_x_290_, v___x_300_);
v___x_302_ = lean_unsigned_to_nat(2u);
v___x_303_ = l_Lean_Syntax_getArg(v_x_290_, v___x_302_);
v___x_304_ = lean_unsigned_to_nat(4u);
v___x_305_ = l_Lean_Syntax_getArg(v_x_290_, v___x_304_);
lean_dec(v_x_290_);
v___x_306_ = 0;
v___x_307_ = l_Lean_SourceInfo_fromRef(v_ref_299_, v___x_306_);
v___x_308_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__4));
v___x_309_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__6, &lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__6_once, _init_lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__6);
v___x_310_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__7));
lean_inc_n(v_currMacroScope_298_, 3);
lean_inc_n(v_quotContext_297_, 3);
v___x_311_ = l_Lean_addMacroScope(v_quotContext_297_, v___x_310_, v_currMacroScope_298_);
v___x_312_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__11));
lean_inc_n(v___x_307_, 11);
v___x_313_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_313_, 0, v___x_307_);
lean_ctor_set(v___x_313_, 1, v___x_309_);
lean_ctor_set(v___x_313_, 2, v___x_311_);
lean_ctor_set(v___x_313_, 3, v___x_312_);
v___x_314_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__13));
v___x_315_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1___closed__1));
v___x_316_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1___closed__3));
v___x_317_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1___closed__4));
v___x_318_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_318_, 0, v___x_307_);
lean_ctor_set(v___x_318_, 1, v___x_317_);
v___x_319_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1___closed__6));
v___x_320_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1___closed__8, &lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1___closed__8_once, _init_lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1___closed__8);
v___x_321_ = lean_box(0);
v___x_322_ = l_Lean_addMacroScope(v_quotContext_297_, v___x_321_, v_currMacroScope_298_);
v___x_323_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1___closed__10));
v___x_324_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_324_, 0, v___x_307_);
lean_ctor_set(v___x_324_, 1, v___x_320_);
lean_ctor_set(v___x_324_, 2, v___x_322_);
lean_ctor_set(v___x_324_, 3, v___x_323_);
v___x_325_ = l_Lean_Syntax_node1(v___x_307_, v___x_319_, v___x_324_);
v___x_326_ = l_Lean_Syntax_node2(v___x_307_, v___x_316_, v___x_318_, v___x_325_);
v___x_327_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1___closed__12, &lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1___closed__12_once, _init_lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1___closed__12);
v___x_328_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1___closed__15));
v___x_329_ = l_Lean_addMacroScope(v_quotContext_297_, v___x_328_, v_currMacroScope_298_);
v___x_330_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1___closed__17));
v___x_331_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_331_, 0, v___x_307_);
lean_ctor_set(v___x_331_, 1, v___x_327_);
lean_ctor_set(v___x_331_, 2, v___x_329_);
lean_ctor_set(v___x_331_, 3, v___x_330_);
v___x_332_ = l_Lean_Syntax_node1(v___x_307_, v___x_314_, v___x_303_);
v___x_333_ = l_Lean_Syntax_node2(v___x_307_, v___x_308_, v___x_331_, v___x_332_);
v___x_334_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1___closed__18));
v___x_335_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_335_, 0, v___x_307_);
lean_ctor_set(v___x_335_, 1, v___x_334_);
v___x_336_ = l_Lean_Syntax_node3(v___x_307_, v___x_315_, v___x_326_, v___x_333_, v___x_335_);
v___x_337_ = l_Lean_Syntax_node3(v___x_307_, v___x_314_, v___x_336_, v___x_301_, v___x_305_);
v___x_338_ = l_Lean_Syntax_node2(v___x_307_, v___x_308_, v___x_313_, v___x_337_);
v___x_339_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_339_, 0, v___x_338_);
lean_ctor_set(v___x_339_, 1, v_a_292_);
return v___x_339_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1___boxed(lean_object* v_x_340_, lean_object* v_a_341_, lean_object* v_a_342_){
_start:
{
lean_object* v_res_343_; 
v_res_343_ = lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1(v_x_340_, v_a_341_, v_a_342_);
lean_dec_ref(v_a_341_);
return v_res_343_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______unexpand__NonUnitalAlgHom__2(lean_object* v_x_344_, lean_object* v_a_345_, lean_object* v_a_346_){
_start:
{
lean_object* v___x_347_; uint8_t v___x_348_; 
v___x_347_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090____1___closed__4));
lean_inc(v_x_344_);
v___x_348_ = l_Lean_Syntax_isOfKind(v_x_344_, v___x_347_);
if (v___x_348_ == 0)
{
lean_object* v___x_349_; lean_object* v___x_350_; 
lean_dec(v_x_344_);
v___x_349_ = lean_box(0);
v___x_350_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_350_, 0, v___x_349_);
lean_ctor_set(v___x_350_, 1, v_a_346_);
return v___x_350_;
}
else
{
lean_object* v___x_351_; lean_object* v___x_352_; lean_object* v___x_353_; uint8_t v___x_354_; 
v___x_351_ = lean_unsigned_to_nat(0u);
v___x_352_ = l_Lean_Syntax_getArg(v_x_344_, v___x_351_);
v___x_353_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______unexpand__NonUnitalAlgHom__1___closed__1));
lean_inc(v___x_352_);
v___x_354_ = l_Lean_Syntax_isOfKind(v___x_352_, v___x_353_);
if (v___x_354_ == 0)
{
lean_object* v___x_355_; lean_object* v___x_356_; 
lean_dec(v___x_352_);
lean_dec(v_x_344_);
v___x_355_ = lean_box(0);
v___x_356_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_356_, 0, v___x_355_);
lean_ctor_set(v___x_356_, 1, v_a_346_);
return v___x_356_;
}
else
{
lean_object* v___x_357_; lean_object* v___x_358_; lean_object* v___x_359_; uint8_t v___x_360_; 
v___x_357_ = lean_unsigned_to_nat(1u);
v___x_358_ = l_Lean_Syntax_getArg(v_x_344_, v___x_357_);
lean_dec(v_x_344_);
v___x_359_ = lean_unsigned_to_nat(3u);
lean_inc(v___x_358_);
v___x_360_ = l_Lean_Syntax_matchesNull(v___x_358_, v___x_359_);
if (v___x_360_ == 0)
{
lean_object* v___x_361_; lean_object* v___x_362_; 
lean_dec(v___x_358_);
lean_dec(v___x_352_);
v___x_361_ = lean_box(0);
v___x_362_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_362_, 0, v___x_361_);
lean_ctor_set(v___x_362_, 1, v_a_346_);
return v___x_362_;
}
else
{
lean_object* v___x_363_; uint8_t v___x_364_; 
v___x_363_ = l_Lean_Syntax_getArg(v___x_358_, v___x_351_);
lean_inc(v___x_363_);
v___x_364_ = l_Lean_Syntax_isOfKind(v___x_363_, v___x_347_);
if (v___x_364_ == 0)
{
lean_object* v___x_365_; lean_object* v___x_366_; 
lean_dec(v___x_363_);
lean_dec(v___x_358_);
lean_dec(v___x_352_);
v___x_365_ = lean_box(0);
v___x_366_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_366_, 0, v___x_365_);
lean_ctor_set(v___x_366_, 1, v_a_346_);
return v___x_366_;
}
else
{
lean_object* v___x_367_; lean_object* v___x_368_; uint8_t v___x_369_; 
v___x_367_ = l_Lean_Syntax_getArg(v___x_363_, v___x_351_);
v___x_368_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______macroRules__term___u2192_u2099_u2090_x5b___x5d____1___closed__15));
v___x_369_ = l_Lean_Syntax_matchesIdent(v___x_367_, v___x_368_);
lean_dec(v___x_367_);
if (v___x_369_ == 0)
{
lean_object* v___x_370_; lean_object* v___x_371_; 
lean_dec(v___x_363_);
lean_dec(v___x_358_);
lean_dec(v___x_352_);
v___x_370_ = lean_box(0);
v___x_371_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_371_, 0, v___x_370_);
lean_ctor_set(v___x_371_, 1, v_a_346_);
return v___x_371_;
}
else
{
lean_object* v___x_372_; uint8_t v___x_373_; 
v___x_372_ = l_Lean_Syntax_getArg(v___x_363_, v___x_357_);
lean_dec(v___x_363_);
lean_inc(v___x_372_);
v___x_373_ = l_Lean_Syntax_matchesNull(v___x_372_, v___x_357_);
if (v___x_373_ == 0)
{
lean_object* v___x_374_; lean_object* v___x_375_; 
lean_dec(v___x_372_);
lean_dec(v___x_358_);
lean_dec(v___x_352_);
v___x_374_ = lean_box(0);
v___x_375_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_375_, 0, v___x_374_);
lean_ctor_set(v___x_375_, 1, v_a_346_);
return v___x_375_;
}
else
{
lean_object* v___x_376_; lean_object* v___x_377_; lean_object* v___x_378_; lean_object* v___x_379_; lean_object* v_ref_380_; uint8_t v___x_381_; lean_object* v___x_382_; lean_object* v___x_383_; lean_object* v___x_384_; lean_object* v___x_385_; lean_object* v___x_386_; lean_object* v___x_387_; lean_object* v___x_388_; lean_object* v___x_389_; 
v___x_376_ = l_Lean_Syntax_getArg(v___x_372_, v___x_351_);
lean_dec(v___x_372_);
v___x_377_ = l_Lean_Syntax_getArg(v___x_358_, v___x_357_);
v___x_378_ = lean_unsigned_to_nat(2u);
v___x_379_ = l_Lean_Syntax_getArg(v___x_358_, v___x_378_);
lean_dec(v___x_358_);
v_ref_380_ = l_Lean_replaceRef(v___x_352_, v_a_345_);
lean_dec(v___x_352_);
v___x_381_ = 0;
v___x_382_ = l_Lean_SourceInfo_fromRef(v_ref_380_, v___x_381_);
lean_dec(v_ref_380_);
v___x_383_ = ((lean_object*)(lp_mathlib_term___u2192_u2099_u2090_x5b___x5d___00__closed__1));
v___x_384_ = ((lean_object*)(lp_mathlib_term___u2192_u2099_u2090_x5b___x5d___00__closed__2));
lean_inc_n(v___x_382_, 2);
v___x_385_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_385_, 0, v___x_382_);
lean_ctor_set(v___x_385_, 1, v___x_384_);
v___x_386_ = ((lean_object*)(lp_mathlib_term___u2192_u209b_u2099_u2090_x5b___x5d___00__closed__6));
v___x_387_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_387_, 0, v___x_382_);
lean_ctor_set(v___x_387_, 1, v___x_386_);
v___x_388_ = l_Lean_Syntax_node5(v___x_382_, v___x_383_, v___x_377_, v___x_385_, v___x_376_, v___x_387_, v___x_379_);
v___x_389_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_389_, 0, v___x_388_);
lean_ctor_set(v___x_389_, 1, v_a_346_);
return v___x_389_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______unexpand__NonUnitalAlgHom__2___boxed(lean_object* v_x_390_, lean_object* v_a_391_, lean_object* v_a_392_){
_start:
{
lean_object* v_res_393_; 
v_res_393_ = lp_mathlib___aux__Mathlib__Algebra__Algebra__NonUnitalHom______unexpand__NonUnitalAlgHom__2(v_x_390_, v_a_391_, v_a_392_);
lean_dec(v_a_391_);
return v_res_393_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHomClass_toNonUnitalAlgSemiHom___redArg(lean_object* v_inst_394_, lean_object* v_f_395_){
_start:
{
lean_object* v___x_396_; 
v___x_396_ = lean_apply_1(v_inst_394_, v_f_395_);
return v___x_396_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHomClass_toNonUnitalAlgSemiHom(lean_object* v_F_397_, lean_object* v_R_398_, lean_object* v_S_399_, lean_object* v_inst_400_, lean_object* v_inst_401_, lean_object* v_00_u03c6_402_, lean_object* v_A_403_, lean_object* v_B_404_, lean_object* v_inst_405_, lean_object* v_inst_406_, lean_object* v_inst_407_, lean_object* v_inst_408_, lean_object* v_inst_409_, lean_object* v_inst_410_, lean_object* v_f_411_){
_start:
{
lean_object* v___x_412_; 
v___x_412_ = lean_apply_1(v_inst_409_, v_f_411_);
return v___x_412_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHomClass_toNonUnitalAlgSemiHom___boxed(lean_object* v_F_413_, lean_object* v_R_414_, lean_object* v_S_415_, lean_object* v_inst_416_, lean_object* v_inst_417_, lean_object* v_00_u03c6_418_, lean_object* v_A_419_, lean_object* v_B_420_, lean_object* v_inst_421_, lean_object* v_inst_422_, lean_object* v_inst_423_, lean_object* v_inst_424_, lean_object* v_inst_425_, lean_object* v_inst_426_, lean_object* v_f_427_){
_start:
{
lean_object* v_res_428_; 
v_res_428_ = lp_mathlib_NonUnitalAlgHomClass_toNonUnitalAlgSemiHom(v_F_413_, v_R_414_, v_S_415_, v_inst_416_, v_inst_417_, v_00_u03c6_418_, v_A_419_, v_B_420_, v_inst_421_, v_inst_422_, v_inst_423_, v_inst_424_, v_inst_425_, v_inst_426_, v_f_427_);
lean_dec(v_inst_424_);
lean_dec_ref(v_inst_423_);
lean_dec(v_inst_422_);
lean_dec_ref(v_inst_421_);
lean_dec(v_00_u03c6_418_);
lean_dec_ref(v_inst_417_);
lean_dec_ref(v_inst_416_);
return v_res_428_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHomClass_instCoeTCNonUnitalAlgHomOfNonUnitalAlgSemiHomClass___redArg(lean_object* v_inst_429_, lean_object* v_inst_430_, lean_object* v_00_u03c6_431_, lean_object* v_inst_432_, lean_object* v_inst_433_, lean_object* v_inst_434_, lean_object* v_inst_435_, lean_object* v_inst_436_){
_start:
{
lean_object* v___x_437_; 
v___x_437_ = lean_alloc_closure((void*)(lp_mathlib_NonUnitalAlgHomClass_toNonUnitalAlgSemiHom___boxed), 15, 14);
lean_closure_set(v___x_437_, 0, lean_box(0));
lean_closure_set(v___x_437_, 1, lean_box(0));
lean_closure_set(v___x_437_, 2, lean_box(0));
lean_closure_set(v___x_437_, 3, v_inst_429_);
lean_closure_set(v___x_437_, 4, v_inst_430_);
lean_closure_set(v___x_437_, 5, v_00_u03c6_431_);
lean_closure_set(v___x_437_, 6, lean_box(0));
lean_closure_set(v___x_437_, 7, lean_box(0));
lean_closure_set(v___x_437_, 8, v_inst_432_);
lean_closure_set(v___x_437_, 9, v_inst_433_);
lean_closure_set(v___x_437_, 10, v_inst_434_);
lean_closure_set(v___x_437_, 11, v_inst_435_);
lean_closure_set(v___x_437_, 12, v_inst_436_);
lean_closure_set(v___x_437_, 13, lean_box(0));
return v___x_437_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHomClass_instCoeTCNonUnitalAlgHomOfNonUnitalAlgSemiHomClass(lean_object* v_F_438_, lean_object* v_R_439_, lean_object* v_S_440_, lean_object* v_A_441_, lean_object* v_B_442_, lean_object* v_inst_443_, lean_object* v_inst_444_, lean_object* v_00_u03c6_445_, lean_object* v_inst_446_, lean_object* v_inst_447_, lean_object* v_inst_448_, lean_object* v_inst_449_, lean_object* v_inst_450_, lean_object* v_inst_451_){
_start:
{
lean_object* v___x_452_; 
v___x_452_ = lean_alloc_closure((void*)(lp_mathlib_NonUnitalAlgHomClass_toNonUnitalAlgSemiHom___boxed), 15, 14);
lean_closure_set(v___x_452_, 0, lean_box(0));
lean_closure_set(v___x_452_, 1, lean_box(0));
lean_closure_set(v___x_452_, 2, lean_box(0));
lean_closure_set(v___x_452_, 3, v_inst_443_);
lean_closure_set(v___x_452_, 4, v_inst_444_);
lean_closure_set(v___x_452_, 5, v_00_u03c6_445_);
lean_closure_set(v___x_452_, 6, lean_box(0));
lean_closure_set(v___x_452_, 7, lean_box(0));
lean_closure_set(v___x_452_, 8, v_inst_446_);
lean_closure_set(v___x_452_, 9, v_inst_447_);
lean_closure_set(v___x_452_, 10, v_inst_448_);
lean_closure_set(v___x_452_, 11, v_inst_449_);
lean_closure_set(v___x_452_, 12, v_inst_450_);
lean_closure_set(v___x_452_, 13, lean_box(0));
return v___x_452_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHomClass_toNonUnitalAlgHom___redArg(lean_object* v_inst_453_, lean_object* v_f_454_){
_start:
{
lean_object* v___x_455_; 
v___x_455_ = lean_apply_1(v_inst_453_, v_f_454_);
return v___x_455_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHomClass_toNonUnitalAlgHom(lean_object* v_F_456_, lean_object* v_R_457_, lean_object* v_inst_458_, lean_object* v_A_459_, lean_object* v_B_460_, lean_object* v_inst_461_, lean_object* v_inst_462_, lean_object* v_inst_463_, lean_object* v_inst_464_, lean_object* v_inst_465_, lean_object* v_inst_466_, lean_object* v_f_467_){
_start:
{
lean_object* v___x_468_; 
v___x_468_ = lean_apply_1(v_inst_465_, v_f_467_);
return v___x_468_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHomClass_toNonUnitalAlgHom___boxed(lean_object* v_F_469_, lean_object* v_R_470_, lean_object* v_inst_471_, lean_object* v_A_472_, lean_object* v_B_473_, lean_object* v_inst_474_, lean_object* v_inst_475_, lean_object* v_inst_476_, lean_object* v_inst_477_, lean_object* v_inst_478_, lean_object* v_inst_479_, lean_object* v_f_480_){
_start:
{
lean_object* v_res_481_; 
v_res_481_ = lp_mathlib_NonUnitalAlgHomClass_toNonUnitalAlgHom(v_F_469_, v_R_470_, v_inst_471_, v_A_472_, v_B_473_, v_inst_474_, v_inst_475_, v_inst_476_, v_inst_477_, v_inst_478_, v_inst_479_, v_f_480_);
lean_dec(v_inst_477_);
lean_dec_ref(v_inst_476_);
lean_dec(v_inst_475_);
lean_dec_ref(v_inst_474_);
lean_dec_ref(v_inst_471_);
return v_res_481_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHomClass_instCoeTCNonUnitalAlgHomId___redArg(lean_object* v_inst_482_, lean_object* v_inst_483_, lean_object* v_inst_484_, lean_object* v_inst_485_, lean_object* v_inst_486_, lean_object* v_inst_487_){
_start:
{
lean_object* v___x_488_; 
v___x_488_ = lean_alloc_closure((void*)(lp_mathlib_NonUnitalAlgHomClass_toNonUnitalAlgHom___boxed), 12, 11);
lean_closure_set(v___x_488_, 0, lean_box(0));
lean_closure_set(v___x_488_, 1, lean_box(0));
lean_closure_set(v___x_488_, 2, v_inst_482_);
lean_closure_set(v___x_488_, 3, lean_box(0));
lean_closure_set(v___x_488_, 4, lean_box(0));
lean_closure_set(v___x_488_, 5, v_inst_483_);
lean_closure_set(v___x_488_, 6, v_inst_484_);
lean_closure_set(v___x_488_, 7, v_inst_485_);
lean_closure_set(v___x_488_, 8, v_inst_486_);
lean_closure_set(v___x_488_, 9, v_inst_487_);
lean_closure_set(v___x_488_, 10, lean_box(0));
return v___x_488_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHomClass_instCoeTCNonUnitalAlgHomId(lean_object* v_F_489_, lean_object* v_R_490_, lean_object* v_inst_491_, lean_object* v_A_492_, lean_object* v_B_493_, lean_object* v_inst_494_, lean_object* v_inst_495_, lean_object* v_inst_496_, lean_object* v_inst_497_, lean_object* v_inst_498_, lean_object* v_inst_499_){
_start:
{
lean_object* v___x_500_; 
v___x_500_ = lean_alloc_closure((void*)(lp_mathlib_NonUnitalAlgHomClass_toNonUnitalAlgHom___boxed), 12, 11);
lean_closure_set(v___x_500_, 0, lean_box(0));
lean_closure_set(v___x_500_, 1, lean_box(0));
lean_closure_set(v___x_500_, 2, v_inst_491_);
lean_closure_set(v___x_500_, 3, lean_box(0));
lean_closure_set(v___x_500_, 4, lean_box(0));
lean_closure_set(v___x_500_, 5, v_inst_494_);
lean_closure_set(v___x_500_, 6, v_inst_495_);
lean_closure_set(v___x_500_, 7, v_inst_496_);
lean_closure_set(v___x_500_, 8, v_inst_497_);
lean_closure_set(v___x_500_, 9, v_inst_498_);
lean_closure_set(v___x_500_, 10, lean_box(0));
return v___x_500_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_Simps_apply___redArg(lean_object* v_f_501_, lean_object* v_a_502_){
_start:
{
lean_object* v___x_503_; 
v___x_503_ = lean_apply_1(v_f_501_, v_a_502_);
return v___x_503_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_Simps_apply(lean_object* v_R_504_, lean_object* v_S_505_, lean_object* v_inst_506_, lean_object* v_inst_507_, lean_object* v_00_u03c6_508_, lean_object* v_A_509_, lean_object* v_B_510_, lean_object* v_inst_511_, lean_object* v_inst_512_, lean_object* v_inst_513_, lean_object* v_inst_514_, lean_object* v_f_515_, lean_object* v_a_516_){
_start:
{
lean_object* v___x_517_; 
v___x_517_ = lean_apply_1(v_f_515_, v_a_516_);
return v___x_517_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_Simps_apply___boxed(lean_object* v_R_518_, lean_object* v_S_519_, lean_object* v_inst_520_, lean_object* v_inst_521_, lean_object* v_00_u03c6_522_, lean_object* v_A_523_, lean_object* v_B_524_, lean_object* v_inst_525_, lean_object* v_inst_526_, lean_object* v_inst_527_, lean_object* v_inst_528_, lean_object* v_f_529_, lean_object* v_a_530_){
_start:
{
lean_object* v_res_531_; 
v_res_531_ = lp_mathlib_NonUnitalAlgHom_Simps_apply(v_R_518_, v_S_519_, v_inst_520_, v_inst_521_, v_00_u03c6_522_, v_A_523_, v_B_524_, v_inst_525_, v_inst_526_, v_inst_527_, v_inst_528_, v_f_529_, v_a_530_);
lean_dec(v_inst_528_);
lean_dec_ref(v_inst_527_);
lean_dec(v_inst_526_);
lean_dec_ref(v_inst_525_);
lean_dec(v_00_u03c6_522_);
lean_dec_ref(v_inst_521_);
lean_dec_ref(v_inst_520_);
return v_res_531_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_id(lean_object* v_R_533_, lean_object* v_A_534_, lean_object* v_inst_535_, lean_object* v_inst_536_, lean_object* v_inst_537_){
_start:
{
lean_object* v___x_538_; 
v___x_538_ = ((lean_object*)(lp_mathlib_NonUnitalAlgHom_id___closed__0));
return v___x_538_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_id___boxed(lean_object* v_R_539_, lean_object* v_A_540_, lean_object* v_inst_541_, lean_object* v_inst_542_, lean_object* v_inst_543_){
_start:
{
lean_object* v_res_544_; 
v_res_544_ = lp_mathlib_NonUnitalAlgHom_id(v_R_539_, v_A_540_, v_inst_541_, v_inst_542_, v_inst_543_);
lean_dec(v_inst_543_);
lean_dec_ref(v_inst_542_);
lean_dec_ref(v_inst_541_);
return v_res_544_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_instZero___redArg___lam__0(lean_object* v_toZero_545_, lean_object* v_x_546_){
_start:
{
lean_inc(v_toZero_545_);
return v_toZero_545_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_instZero___redArg___lam__0___boxed(lean_object* v_toZero_547_, lean_object* v_x_548_){
_start:
{
lean_object* v_res_549_; 
v_res_549_ = lp_mathlib_NonUnitalAlgHom_instZero___redArg___lam__0(v_toZero_547_, v_x_548_);
lean_dec(v_x_548_);
lean_dec(v_toZero_547_);
return v_res_549_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_instZero___redArg(lean_object* v_inst_550_){
_start:
{
lean_object* v_toAddCommMonoid_551_; lean_object* v___x_552_; lean_object* v___x_553_; lean_object* v_toZero_554_; lean_object* v___f_555_; 
v_toAddCommMonoid_551_ = lean_ctor_get(v_inst_550_, 0);
v___x_552_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_toAddCommMonoid_551_);
v___x_553_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_552_);
v_toZero_554_ = lean_ctor_get(v___x_553_, 0);
lean_inc(v_toZero_554_);
lean_dec_ref(v___x_553_);
v___f_555_ = lean_alloc_closure((void*)(lp_mathlib_NonUnitalAlgHom_instZero___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_555_, 0, v_toZero_554_);
return v___f_555_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_instZero___redArg___boxed(lean_object* v_inst_556_){
_start:
{
lean_object* v_res_557_; 
v_res_557_ = lp_mathlib_NonUnitalAlgHom_instZero___redArg(v_inst_556_);
lean_dec_ref(v_inst_556_);
return v_res_557_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_instZero(lean_object* v_R_558_, lean_object* v_S_559_, lean_object* v_inst_560_, lean_object* v_inst_561_, lean_object* v_00_u03c6_562_, lean_object* v_A_563_, lean_object* v_B_564_, lean_object* v_inst_565_, lean_object* v_inst_566_, lean_object* v_inst_567_, lean_object* v_inst_568_){
_start:
{
lean_object* v___x_569_; 
v___x_569_ = lp_mathlib_NonUnitalAlgHom_instZero___redArg(v_inst_567_);
return v___x_569_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_instZero___boxed(lean_object* v_R_570_, lean_object* v_S_571_, lean_object* v_inst_572_, lean_object* v_inst_573_, lean_object* v_00_u03c6_574_, lean_object* v_A_575_, lean_object* v_B_576_, lean_object* v_inst_577_, lean_object* v_inst_578_, lean_object* v_inst_579_, lean_object* v_inst_580_){
_start:
{
lean_object* v_res_581_; 
v_res_581_ = lp_mathlib_NonUnitalAlgHom_instZero(v_R_570_, v_S_571_, v_inst_572_, v_inst_573_, v_00_u03c6_574_, v_A_575_, v_B_576_, v_inst_577_, v_inst_578_, v_inst_579_, v_inst_580_);
lean_dec(v_inst_580_);
lean_dec_ref(v_inst_579_);
lean_dec(v_inst_578_);
lean_dec_ref(v_inst_577_);
lean_dec(v_00_u03c6_574_);
lean_dec_ref(v_inst_573_);
lean_dec_ref(v_inst_572_);
return v_res_581_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_instOneId(lean_object* v_R_582_, lean_object* v_inst_583_, lean_object* v_A_584_, lean_object* v_inst_585_, lean_object* v_inst_586_){
_start:
{
lean_object* v___x_587_; 
v___x_587_ = ((lean_object*)(lp_mathlib_NonUnitalAlgHom_id___closed__0));
return v___x_587_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_instOneId___boxed(lean_object* v_R_588_, lean_object* v_inst_589_, lean_object* v_A_590_, lean_object* v_inst_591_, lean_object* v_inst_592_){
_start:
{
lean_object* v_res_593_; 
v_res_593_ = lp_mathlib_NonUnitalAlgHom_instOneId(v_R_588_, v_inst_589_, v_A_590_, v_inst_591_, v_inst_592_);
lean_dec(v_inst_592_);
lean_dec_ref(v_inst_591_);
lean_dec_ref(v_inst_589_);
return v_res_593_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_instInhabited___redArg(lean_object* v_inst_594_){
_start:
{
lean_object* v_toAddCommMonoid_595_; lean_object* v___x_596_; lean_object* v___x_597_; lean_object* v_toZero_598_; lean_object* v___f_599_; 
v_toAddCommMonoid_595_ = lean_ctor_get(v_inst_594_, 0);
v___x_596_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_toAddCommMonoid_595_);
v___x_597_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_596_);
v_toZero_598_ = lean_ctor_get(v___x_597_, 0);
lean_inc(v_toZero_598_);
lean_dec_ref(v___x_597_);
v___f_599_ = lean_alloc_closure((void*)(lp_mathlib_NonUnitalAlgHom_instZero___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_599_, 0, v_toZero_598_);
return v___f_599_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_instInhabited___redArg___boxed(lean_object* v_inst_600_){
_start:
{
lean_object* v_res_601_; 
v_res_601_ = lp_mathlib_NonUnitalAlgHom_instInhabited___redArg(v_inst_600_);
lean_dec_ref(v_inst_600_);
return v_res_601_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_instInhabited(lean_object* v_R_602_, lean_object* v_S_603_, lean_object* v_inst_604_, lean_object* v_inst_605_, lean_object* v_00_u03c6_606_, lean_object* v_A_607_, lean_object* v_B_608_, lean_object* v_inst_609_, lean_object* v_inst_610_, lean_object* v_inst_611_, lean_object* v_inst_612_){
_start:
{
lean_object* v___x_613_; 
v___x_613_ = lp_mathlib_NonUnitalAlgHom_instInhabited___redArg(v_inst_611_);
return v___x_613_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_instInhabited___boxed(lean_object* v_R_614_, lean_object* v_S_615_, lean_object* v_inst_616_, lean_object* v_inst_617_, lean_object* v_00_u03c6_618_, lean_object* v_A_619_, lean_object* v_B_620_, lean_object* v_inst_621_, lean_object* v_inst_622_, lean_object* v_inst_623_, lean_object* v_inst_624_){
_start:
{
lean_object* v_res_625_; 
v_res_625_ = lp_mathlib_NonUnitalAlgHom_instInhabited(v_R_614_, v_S_615_, v_inst_616_, v_inst_617_, v_00_u03c6_618_, v_A_619_, v_B_620_, v_inst_621_, v_inst_622_, v_inst_623_, v_inst_624_);
lean_dec(v_inst_624_);
lean_dec_ref(v_inst_623_);
lean_dec(v_inst_622_);
lean_dec_ref(v_inst_621_);
lean_dec(v_00_u03c6_618_);
lean_dec_ref(v_inst_617_);
lean_dec_ref(v_inst_616_);
return v_res_625_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_comp___redArg___lam__0(lean_object* v_f_626_, lean_object* v___y_627_){
_start:
{
lean_object* v___x_628_; 
v___x_628_ = lean_apply_1(v_f_626_, v___y_627_);
return v___x_628_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_comp___redArg___lam__1(lean_object* v_g_629_, lean_object* v___y_630_){
_start:
{
lean_object* v___x_631_; 
v___x_631_ = lean_apply_1(v_g_629_, v___y_630_);
return v___x_631_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_comp___redArg(lean_object* v_f_632_, lean_object* v_g_633_){
_start:
{
lean_object* v___f_634_; lean_object* v___f_635_; lean_object* v___f_636_; 
v___f_634_ = lean_alloc_closure((void*)(lp_mathlib_NonUnitalAlgHom_comp___redArg___lam__0), 2, 1);
lean_closure_set(v___f_634_, 0, v_f_632_);
v___f_635_ = lean_alloc_closure((void*)(lp_mathlib_NonUnitalAlgHom_comp___redArg___lam__1), 2, 1);
lean_closure_set(v___f_635_, 0, v_g_633_);
v___f_636_ = lean_alloc_closure((void*)(lp_mathlib_OneHom_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_636_, 0, v___f_635_);
lean_closure_set(v___f_636_, 1, v___f_634_);
return v___f_636_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_comp(lean_object* v_R_637_, lean_object* v_S_638_, lean_object* v_T_639_, lean_object* v_inst_640_, lean_object* v_inst_641_, lean_object* v_inst_642_, lean_object* v_00_u03c6_643_, lean_object* v_A_644_, lean_object* v_B_645_, lean_object* v_C_646_, lean_object* v_inst_647_, lean_object* v_inst_648_, lean_object* v_inst_649_, lean_object* v_inst_650_, lean_object* v_inst_651_, lean_object* v_inst_652_, lean_object* v_00_u03c8_653_, lean_object* v_00_u03c7_654_, lean_object* v_f_655_, lean_object* v_g_656_, lean_object* v_00_u03ba_657_){
_start:
{
lean_object* v___x_658_; 
v___x_658_ = lp_mathlib_NonUnitalAlgHom_comp___redArg(v_f_655_, v_g_656_);
return v___x_658_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_comp___boxed(lean_object** _args){
lean_object* v_R_659_ = _args[0];
lean_object* v_S_660_ = _args[1];
lean_object* v_T_661_ = _args[2];
lean_object* v_inst_662_ = _args[3];
lean_object* v_inst_663_ = _args[4];
lean_object* v_inst_664_ = _args[5];
lean_object* v_00_u03c6_665_ = _args[6];
lean_object* v_A_666_ = _args[7];
lean_object* v_B_667_ = _args[8];
lean_object* v_C_668_ = _args[9];
lean_object* v_inst_669_ = _args[10];
lean_object* v_inst_670_ = _args[11];
lean_object* v_inst_671_ = _args[12];
lean_object* v_inst_672_ = _args[13];
lean_object* v_inst_673_ = _args[14];
lean_object* v_inst_674_ = _args[15];
lean_object* v_00_u03c8_675_ = _args[16];
lean_object* v_00_u03c7_676_ = _args[17];
lean_object* v_f_677_ = _args[18];
lean_object* v_g_678_ = _args[19];
lean_object* v_00_u03ba_679_ = _args[20];
_start:
{
lean_object* v_res_680_; 
v_res_680_ = lp_mathlib_NonUnitalAlgHom_comp(v_R_659_, v_S_660_, v_T_661_, v_inst_662_, v_inst_663_, v_inst_664_, v_00_u03c6_665_, v_A_666_, v_B_667_, v_C_668_, v_inst_669_, v_inst_670_, v_inst_671_, v_inst_672_, v_inst_673_, v_inst_674_, v_00_u03c8_675_, v_00_u03c7_676_, v_f_677_, v_g_678_, v_00_u03ba_679_);
lean_dec(v_00_u03c7_676_);
lean_dec(v_00_u03c8_675_);
lean_dec(v_inst_674_);
lean_dec_ref(v_inst_673_);
lean_dec(v_inst_672_);
lean_dec_ref(v_inst_671_);
lean_dec(v_inst_670_);
lean_dec_ref(v_inst_669_);
lean_dec(v_00_u03c6_665_);
lean_dec_ref(v_inst_664_);
lean_dec_ref(v_inst_663_);
lean_dec_ref(v_inst_662_);
return v_res_680_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_inverse___redArg(lean_object* v_g_681_){
_start:
{
lean_inc(v_g_681_);
return v_g_681_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_inverse___redArg___boxed(lean_object* v_g_682_){
_start:
{
lean_object* v_res_683_; 
v_res_683_ = lp_mathlib_NonUnitalAlgHom_inverse___redArg(v_g_682_);
lean_dec(v_g_682_);
return v_res_683_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_inverse(lean_object* v_R_684_, lean_object* v_inst_685_, lean_object* v_A_686_, lean_object* v_inst_687_, lean_object* v_inst_688_, lean_object* v_B_u2081_689_, lean_object* v_inst_690_, lean_object* v_inst_691_, lean_object* v_f_692_, lean_object* v_g_693_, lean_object* v_h_u2081_694_, lean_object* v_h_u2082_695_){
_start:
{
lean_inc(v_g_693_);
return v_g_693_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_inverse___boxed(lean_object* v_R_696_, lean_object* v_inst_697_, lean_object* v_A_698_, lean_object* v_inst_699_, lean_object* v_inst_700_, lean_object* v_B_u2081_701_, lean_object* v_inst_702_, lean_object* v_inst_703_, lean_object* v_f_704_, lean_object* v_g_705_, lean_object* v_h_u2081_706_, lean_object* v_h_u2082_707_){
_start:
{
lean_object* v_res_708_; 
v_res_708_ = lp_mathlib_NonUnitalAlgHom_inverse(v_R_696_, v_inst_697_, v_A_698_, v_inst_699_, v_inst_700_, v_B_u2081_701_, v_inst_702_, v_inst_703_, v_f_704_, v_g_705_, v_h_u2081_706_, v_h_u2082_707_);
lean_dec(v_g_705_);
lean_dec(v_f_704_);
lean_dec(v_inst_703_);
lean_dec_ref(v_inst_702_);
lean_dec(v_inst_700_);
lean_dec_ref(v_inst_699_);
lean_dec_ref(v_inst_697_);
return v_res_708_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_inverse_x27___redArg(lean_object* v_g_709_){
_start:
{
lean_inc(v_g_709_);
return v_g_709_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_inverse_x27___redArg___boxed(lean_object* v_g_710_){
_start:
{
lean_object* v_res_711_; 
v_res_711_ = lp_mathlib_NonUnitalAlgHom_inverse_x27___redArg(v_g_710_);
lean_dec(v_g_710_);
return v_res_711_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_inverse_x27(lean_object* v_R_712_, lean_object* v_S_713_, lean_object* v_inst_714_, lean_object* v_inst_715_, lean_object* v_00_u03c6_716_, lean_object* v_A_717_, lean_object* v_B_718_, lean_object* v_inst_719_, lean_object* v_inst_720_, lean_object* v_inst_721_, lean_object* v_inst_722_, lean_object* v_00_u03c6_x27_723_, lean_object* v_f_724_, lean_object* v_g_725_, lean_object* v_k_726_, lean_object* v_h_u2081_727_, lean_object* v_h_u2082_728_){
_start:
{
lean_inc(v_g_725_);
return v_g_725_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_inverse_x27___boxed(lean_object** _args){
lean_object* v_R_729_ = _args[0];
lean_object* v_S_730_ = _args[1];
lean_object* v_inst_731_ = _args[2];
lean_object* v_inst_732_ = _args[3];
lean_object* v_00_u03c6_733_ = _args[4];
lean_object* v_A_734_ = _args[5];
lean_object* v_B_735_ = _args[6];
lean_object* v_inst_736_ = _args[7];
lean_object* v_inst_737_ = _args[8];
lean_object* v_inst_738_ = _args[9];
lean_object* v_inst_739_ = _args[10];
lean_object* v_00_u03c6_x27_740_ = _args[11];
lean_object* v_f_741_ = _args[12];
lean_object* v_g_742_ = _args[13];
lean_object* v_k_743_ = _args[14];
lean_object* v_h_u2081_744_ = _args[15];
lean_object* v_h_u2082_745_ = _args[16];
_start:
{
lean_object* v_res_746_; 
v_res_746_ = lp_mathlib_NonUnitalAlgHom_inverse_x27(v_R_729_, v_S_730_, v_inst_731_, v_inst_732_, v_00_u03c6_733_, v_A_734_, v_B_735_, v_inst_736_, v_inst_737_, v_inst_738_, v_inst_739_, v_00_u03c6_x27_740_, v_f_741_, v_g_742_, v_k_743_, v_h_u2081_744_, v_h_u2082_745_);
lean_dec(v_g_742_);
lean_dec(v_f_741_);
lean_dec(v_00_u03c6_x27_740_);
lean_dec(v_inst_739_);
lean_dec_ref(v_inst_738_);
lean_dec(v_inst_737_);
lean_dec_ref(v_inst_736_);
lean_dec(v_00_u03c6_733_);
lean_dec_ref(v_inst_732_);
lean_dec_ref(v_inst_731_);
return v_res_746_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_fst___lam__0(lean_object* v_self_747_){
_start:
{
lean_object* v_fst_748_; 
v_fst_748_ = lean_ctor_get(v_self_747_, 0);
lean_inc(v_fst_748_);
return v_fst_748_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_fst___lam__0___boxed(lean_object* v_self_749_){
_start:
{
lean_object* v_res_750_; 
v_res_750_ = lp_mathlib_NonUnitalAlgHom_fst___lam__0(v_self_749_);
lean_dec_ref(v_self_749_);
return v_res_750_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_fst(lean_object* v_R_752_, lean_object* v_inst_753_, lean_object* v_A_754_, lean_object* v_B_755_, lean_object* v_inst_756_, lean_object* v_inst_757_, lean_object* v_inst_758_, lean_object* v_inst_759_){
_start:
{
lean_object* v___f_760_; 
v___f_760_ = ((lean_object*)(lp_mathlib_NonUnitalAlgHom_fst___closed__0));
return v___f_760_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_fst___boxed(lean_object* v_R_761_, lean_object* v_inst_762_, lean_object* v_A_763_, lean_object* v_B_764_, lean_object* v_inst_765_, lean_object* v_inst_766_, lean_object* v_inst_767_, lean_object* v_inst_768_){
_start:
{
lean_object* v_res_769_; 
v_res_769_ = lp_mathlib_NonUnitalAlgHom_fst(v_R_761_, v_inst_762_, v_A_763_, v_B_764_, v_inst_765_, v_inst_766_, v_inst_767_, v_inst_768_);
lean_dec(v_inst_768_);
lean_dec_ref(v_inst_767_);
lean_dec(v_inst_766_);
lean_dec_ref(v_inst_765_);
lean_dec_ref(v_inst_762_);
return v_res_769_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_snd___lam__0(lean_object* v_self_770_){
_start:
{
lean_object* v_snd_771_; 
v_snd_771_ = lean_ctor_get(v_self_770_, 1);
lean_inc(v_snd_771_);
return v_snd_771_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_snd___lam__0___boxed(lean_object* v_self_772_){
_start:
{
lean_object* v_res_773_; 
v_res_773_ = lp_mathlib_NonUnitalAlgHom_snd___lam__0(v_self_772_);
lean_dec_ref(v_self_772_);
return v_res_773_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_snd(lean_object* v_R_775_, lean_object* v_inst_776_, lean_object* v_A_777_, lean_object* v_B_778_, lean_object* v_inst_779_, lean_object* v_inst_780_, lean_object* v_inst_781_, lean_object* v_inst_782_){
_start:
{
lean_object* v___f_783_; 
v___f_783_ = ((lean_object*)(lp_mathlib_NonUnitalAlgHom_snd___closed__0));
return v___f_783_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_snd___boxed(lean_object* v_R_784_, lean_object* v_inst_785_, lean_object* v_A_786_, lean_object* v_B_787_, lean_object* v_inst_788_, lean_object* v_inst_789_, lean_object* v_inst_790_, lean_object* v_inst_791_){
_start:
{
lean_object* v_res_792_; 
v_res_792_ = lp_mathlib_NonUnitalAlgHom_snd(v_R_784_, v_inst_785_, v_A_786_, v_B_787_, v_inst_788_, v_inst_789_, v_inst_790_, v_inst_791_);
lean_dec(v_inst_791_);
lean_dec_ref(v_inst_790_);
lean_dec(v_inst_789_);
lean_dec_ref(v_inst_788_);
lean_dec_ref(v_inst_785_);
return v_res_792_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_prod___redArg(lean_object* v_f_793_, lean_object* v_g_794_){
_start:
{
lean_object* v___f_795_; lean_object* v___f_796_; lean_object* v___x_797_; 
v___f_795_ = lean_alloc_closure((void*)(lp_mathlib_NonUnitalAlgHom_comp___redArg___lam__1), 2, 1);
lean_closure_set(v___f_795_, 0, v_g_794_);
v___f_796_ = lean_alloc_closure((void*)(lp_mathlib_NonUnitalAlgHom_comp___redArg___lam__0), 2, 1);
lean_closure_set(v___f_796_, 0, v_f_793_);
v___x_797_ = lean_alloc_closure((void*)(lp_mathlib_Function_prod), 6, 5);
lean_closure_set(v___x_797_, 0, lean_box(0));
lean_closure_set(v___x_797_, 1, lean_box(0));
lean_closure_set(v___x_797_, 2, lean_box(0));
lean_closure_set(v___x_797_, 3, v___f_796_);
lean_closure_set(v___x_797_, 4, v___f_795_);
return v___x_797_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_prod(lean_object* v_R_798_, lean_object* v_inst_799_, lean_object* v_A_800_, lean_object* v_B_801_, lean_object* v_C_802_, lean_object* v_inst_803_, lean_object* v_inst_804_, lean_object* v_inst_805_, lean_object* v_inst_806_, lean_object* v_inst_807_, lean_object* v_inst_808_, lean_object* v_f_809_, lean_object* v_g_810_){
_start:
{
lean_object* v___x_811_; 
v___x_811_ = lp_mathlib_NonUnitalAlgHom_prod___redArg(v_f_809_, v_g_810_);
return v___x_811_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_prod___boxed(lean_object* v_R_812_, lean_object* v_inst_813_, lean_object* v_A_814_, lean_object* v_B_815_, lean_object* v_C_816_, lean_object* v_inst_817_, lean_object* v_inst_818_, lean_object* v_inst_819_, lean_object* v_inst_820_, lean_object* v_inst_821_, lean_object* v_inst_822_, lean_object* v_f_823_, lean_object* v_g_824_){
_start:
{
lean_object* v_res_825_; 
v_res_825_ = lp_mathlib_NonUnitalAlgHom_prod(v_R_812_, v_inst_813_, v_A_814_, v_B_815_, v_C_816_, v_inst_817_, v_inst_818_, v_inst_819_, v_inst_820_, v_inst_821_, v_inst_822_, v_f_823_, v_g_824_);
lean_dec(v_inst_822_);
lean_dec(v_inst_821_);
lean_dec_ref(v_inst_820_);
lean_dec_ref(v_inst_819_);
lean_dec(v_inst_818_);
lean_dec_ref(v_inst_817_);
lean_dec_ref(v_inst_813_);
return v_res_825_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_prodEquiv___lam__0(lean_object* v_f_826_, lean_object* v___y_827_){
_start:
{
lean_object* v_fst_828_; lean_object* v_snd_829_; lean_object* v___x_36__overap_830_; lean_object* v___x_831_; 
v_fst_828_ = lean_ctor_get(v_f_826_, 0);
lean_inc(v_fst_828_);
v_snd_829_ = lean_ctor_get(v_f_826_, 1);
lean_inc(v_snd_829_);
lean_dec_ref(v_f_826_);
v___x_36__overap_830_ = lp_mathlib_NonUnitalAlgHom_prod___redArg(v_fst_828_, v_snd_829_);
v___x_831_ = lean_apply_1(v___x_36__overap_830_, v___y_827_);
return v___x_831_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_prodEquiv___lam__1(lean_object* v_f_832_){
_start:
{
lean_object* v___f_833_; lean_object* v___x_834_; lean_object* v___f_835_; lean_object* v___x_836_; lean_object* v___x_837_; 
v___f_833_ = ((lean_object*)(lp_mathlib_NonUnitalAlgHom_fst___closed__0));
lean_inc_ref(v_f_832_);
v___x_834_ = lp_mathlib_NonUnitalAlgHom_comp___redArg(v___f_833_, v_f_832_);
v___f_835_ = ((lean_object*)(lp_mathlib_NonUnitalAlgHom_snd___closed__0));
v___x_836_ = lp_mathlib_NonUnitalAlgHom_comp___redArg(v___f_835_, v_f_832_);
v___x_837_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_837_, 0, v___x_834_);
lean_ctor_set(v___x_837_, 1, v___x_836_);
return v___x_837_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_prodEquiv(lean_object* v_R_843_, lean_object* v_inst_844_, lean_object* v_A_845_, lean_object* v_B_846_, lean_object* v_C_847_, lean_object* v_inst_848_, lean_object* v_inst_849_, lean_object* v_inst_850_, lean_object* v_inst_851_, lean_object* v_inst_852_, lean_object* v_inst_853_){
_start:
{
lean_object* v___x_854_; 
v___x_854_ = ((lean_object*)(lp_mathlib_NonUnitalAlgHom_prodEquiv___closed__2));
return v___x_854_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_prodEquiv___boxed(lean_object* v_R_855_, lean_object* v_inst_856_, lean_object* v_A_857_, lean_object* v_B_858_, lean_object* v_C_859_, lean_object* v_inst_860_, lean_object* v_inst_861_, lean_object* v_inst_862_, lean_object* v_inst_863_, lean_object* v_inst_864_, lean_object* v_inst_865_){
_start:
{
lean_object* v_res_866_; 
v_res_866_ = lp_mathlib_NonUnitalAlgHom_prodEquiv(v_R_855_, v_inst_856_, v_A_857_, v_B_858_, v_C_859_, v_inst_860_, v_inst_861_, v_inst_862_, v_inst_863_, v_inst_864_, v_inst_865_);
lean_dec(v_inst_865_);
lean_dec(v_inst_864_);
lean_dec_ref(v_inst_863_);
lean_dec_ref(v_inst_862_);
lean_dec(v_inst_861_);
lean_dec_ref(v_inst_860_);
lean_dec_ref(v_inst_856_);
return v_res_866_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_inl___redArg(lean_object* v_inst_867_){
_start:
{
lean_object* v_toAddCommMonoid_868_; lean_object* v___x_869_; lean_object* v___x_870_; lean_object* v_toZero_871_; lean_object* v___x_872_; lean_object* v___f_873_; lean_object* v___x_874_; 
v_toAddCommMonoid_868_ = lean_ctor_get(v_inst_867_, 0);
v___x_869_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_toAddCommMonoid_868_);
v___x_870_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_869_);
v_toZero_871_ = lean_ctor_get(v___x_870_, 0);
lean_inc(v_toZero_871_);
lean_dec_ref(v___x_870_);
v___x_872_ = ((lean_object*)(lp_mathlib_NonUnitalAlgHom_id___closed__0));
v___f_873_ = lean_alloc_closure((void*)(lp_mathlib_NonUnitalAlgHom_instZero___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_873_, 0, v_toZero_871_);
v___x_874_ = lp_mathlib_NonUnitalAlgHom_prod___redArg(v___x_872_, v___f_873_);
return v___x_874_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_inl___redArg___boxed(lean_object* v_inst_875_){
_start:
{
lean_object* v_res_876_; 
v_res_876_ = lp_mathlib_NonUnitalAlgHom_inl___redArg(v_inst_875_);
lean_dec_ref(v_inst_875_);
return v_res_876_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_inl(lean_object* v_R_877_, lean_object* v_inst_878_, lean_object* v_A_879_, lean_object* v_B_880_, lean_object* v_inst_881_, lean_object* v_inst_882_, lean_object* v_inst_883_, lean_object* v_inst_884_){
_start:
{
lean_object* v___x_885_; 
v___x_885_ = lp_mathlib_NonUnitalAlgHom_inl___redArg(v_inst_883_);
return v___x_885_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_inl___boxed(lean_object* v_R_886_, lean_object* v_inst_887_, lean_object* v_A_888_, lean_object* v_B_889_, lean_object* v_inst_890_, lean_object* v_inst_891_, lean_object* v_inst_892_, lean_object* v_inst_893_){
_start:
{
lean_object* v_res_894_; 
v_res_894_ = lp_mathlib_NonUnitalAlgHom_inl(v_R_886_, v_inst_887_, v_A_888_, v_B_889_, v_inst_890_, v_inst_891_, v_inst_892_, v_inst_893_);
lean_dec(v_inst_893_);
lean_dec_ref(v_inst_892_);
lean_dec(v_inst_891_);
lean_dec_ref(v_inst_890_);
lean_dec_ref(v_inst_887_);
return v_res_894_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_inr___redArg(lean_object* v_inst_895_){
_start:
{
lean_object* v_toAddCommMonoid_896_; lean_object* v___x_897_; lean_object* v___x_898_; lean_object* v_toZero_899_; lean_object* v___f_900_; lean_object* v___x_901_; lean_object* v___x_902_; 
v_toAddCommMonoid_896_ = lean_ctor_get(v_inst_895_, 0);
v___x_897_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_toAddCommMonoid_896_);
v___x_898_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_897_);
v_toZero_899_ = lean_ctor_get(v___x_898_, 0);
lean_inc(v_toZero_899_);
lean_dec_ref(v___x_898_);
v___f_900_ = lean_alloc_closure((void*)(lp_mathlib_NonUnitalAlgHom_instZero___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_900_, 0, v_toZero_899_);
v___x_901_ = ((lean_object*)(lp_mathlib_NonUnitalAlgHom_id___closed__0));
v___x_902_ = lp_mathlib_NonUnitalAlgHom_prod___redArg(v___f_900_, v___x_901_);
return v___x_902_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_inr___redArg___boxed(lean_object* v_inst_903_){
_start:
{
lean_object* v_res_904_; 
v_res_904_ = lp_mathlib_NonUnitalAlgHom_inr___redArg(v_inst_903_);
lean_dec_ref(v_inst_903_);
return v_res_904_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_inr(lean_object* v_R_905_, lean_object* v_inst_906_, lean_object* v_A_907_, lean_object* v_B_908_, lean_object* v_inst_909_, lean_object* v_inst_910_, lean_object* v_inst_911_, lean_object* v_inst_912_){
_start:
{
lean_object* v___x_913_; 
v___x_913_ = lp_mathlib_NonUnitalAlgHom_inr___redArg(v_inst_909_);
return v___x_913_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_inr___boxed(lean_object* v_R_914_, lean_object* v_inst_915_, lean_object* v_A_916_, lean_object* v_B_917_, lean_object* v_inst_918_, lean_object* v_inst_919_, lean_object* v_inst_920_, lean_object* v_inst_921_){
_start:
{
lean_object* v_res_922_; 
v_res_922_ = lp_mathlib_NonUnitalAlgHom_inr(v_R_914_, v_inst_915_, v_A_916_, v_B_917_, v_inst_918_, v_inst_919_, v_inst_920_, v_inst_921_);
lean_dec(v_inst_921_);
lean_dec_ref(v_inst_920_);
lean_dec(v_inst_919_);
lean_dec_ref(v_inst_918_);
lean_dec_ref(v_inst_915_);
return v_res_922_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_toNonUnitalAlgHom___redArg(lean_object* v_f_923_){
_start:
{
lean_inc(v_f_923_);
return v_f_923_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_toNonUnitalAlgHom___redArg___boxed(lean_object* v_f_924_){
_start:
{
lean_object* v_res_925_; 
v_res_925_ = lp_mathlib_AlgHom_toNonUnitalAlgHom___redArg(v_f_924_);
lean_dec(v_f_924_);
return v_res_925_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_toNonUnitalAlgHom(lean_object* v_R_926_, lean_object* v_inst_927_, lean_object* v_A_928_, lean_object* v_B_929_, lean_object* v_inst_930_, lean_object* v_inst_931_, lean_object* v_inst_932_, lean_object* v_inst_933_, lean_object* v_f_934_){
_start:
{
lean_inc(v_f_934_);
return v_f_934_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_toNonUnitalAlgHom___boxed(lean_object* v_R_935_, lean_object* v_inst_936_, lean_object* v_A_937_, lean_object* v_B_938_, lean_object* v_inst_939_, lean_object* v_inst_940_, lean_object* v_inst_941_, lean_object* v_inst_942_, lean_object* v_f_943_){
_start:
{
lean_object* v_res_944_; 
v_res_944_ = lp_mathlib_AlgHom_toNonUnitalAlgHom(v_R_935_, v_inst_936_, v_A_937_, v_B_938_, v_inst_939_, v_inst_940_, v_inst_941_, v_inst_942_, v_f_943_);
lean_dec(v_f_943_);
lean_dec_ref(v_inst_942_);
lean_dec_ref(v_inst_941_);
lean_dec_ref(v_inst_940_);
lean_dec_ref(v_inst_939_);
lean_dec_ref(v_inst_936_);
return v_res_944_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_NonUnitalAlgHom_hasCoe___redArg(lean_object* v_inst_945_, lean_object* v_inst_946_, lean_object* v_inst_947_, lean_object* v_inst_948_, lean_object* v_inst_949_){
_start:
{
lean_object* v___x_950_; 
v___x_950_ = lean_alloc_closure((void*)(lp_mathlib_AlgHom_toNonUnitalAlgHom___boxed), 9, 8);
lean_closure_set(v___x_950_, 0, lean_box(0));
lean_closure_set(v___x_950_, 1, v_inst_945_);
lean_closure_set(v___x_950_, 2, lean_box(0));
lean_closure_set(v___x_950_, 3, lean_box(0));
lean_closure_set(v___x_950_, 4, v_inst_946_);
lean_closure_set(v___x_950_, 5, v_inst_947_);
lean_closure_set(v___x_950_, 6, v_inst_948_);
lean_closure_set(v___x_950_, 7, v_inst_949_);
return v___x_950_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_NonUnitalAlgHom_hasCoe(lean_object* v_R_951_, lean_object* v_inst_952_, lean_object* v_A_953_, lean_object* v_B_954_, lean_object* v_inst_955_, lean_object* v_inst_956_, lean_object* v_inst_957_, lean_object* v_inst_958_){
_start:
{
lean_object* v___x_959_; 
v___x_959_ = lean_alloc_closure((void*)(lp_mathlib_AlgHom_toNonUnitalAlgHom___boxed), 9, 8);
lean_closure_set(v___x_959_, 0, lean_box(0));
lean_closure_set(v___x_959_, 1, v_inst_952_);
lean_closure_set(v___x_959_, 2, lean_box(0));
lean_closure_set(v___x_959_, 3, lean_box(0));
lean_closure_set(v___x_959_, 4, v_inst_955_);
lean_closure_set(v___x_959_, 5, v_inst_956_);
lean_closure_set(v___x_959_, 6, v_inst_957_);
lean_closure_set(v___x_959_, 7, v_inst_958_);
return v___x_959_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_restrictScalars___redArg(lean_object* v_f_960_){
_start:
{
lean_object* v___f_961_; 
v___f_961_ = lean_alloc_closure((void*)(lp_mathlib_NonUnitalAlgHom_comp___redArg___lam__0), 2, 1);
lean_closure_set(v___f_961_, 0, v_f_960_);
return v___f_961_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_restrictScalars(lean_object* v_R_962_, lean_object* v_S_963_, lean_object* v_A_964_, lean_object* v_B_965_, lean_object* v_inst_966_, lean_object* v_inst_967_, lean_object* v_inst_968_, lean_object* v_inst_969_, lean_object* v_inst_970_, lean_object* v_inst_971_, lean_object* v_inst_972_, lean_object* v_inst_973_, lean_object* v_inst_974_, lean_object* v_inst_975_, lean_object* v_inst_976_, lean_object* v_f_977_){
_start:
{
lean_object* v___f_978_; 
v___f_978_ = lean_alloc_closure((void*)(lp_mathlib_NonUnitalAlgHom_comp___redArg___lam__0), 2, 1);
lean_closure_set(v___f_978_, 0, v_f_977_);
return v___f_978_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_restrictScalars___boxed(lean_object* v_R_979_, lean_object* v_S_980_, lean_object* v_A_981_, lean_object* v_B_982_, lean_object* v_inst_983_, lean_object* v_inst_984_, lean_object* v_inst_985_, lean_object* v_inst_986_, lean_object* v_inst_987_, lean_object* v_inst_988_, lean_object* v_inst_989_, lean_object* v_inst_990_, lean_object* v_inst_991_, lean_object* v_inst_992_, lean_object* v_inst_993_, lean_object* v_f_994_){
_start:
{
lean_object* v_res_995_; 
v_res_995_ = lp_mathlib_NonUnitalAlgHom_restrictScalars(v_R_979_, v_S_980_, v_A_981_, v_B_982_, v_inst_983_, v_inst_984_, v_inst_985_, v_inst_986_, v_inst_987_, v_inst_988_, v_inst_989_, v_inst_990_, v_inst_991_, v_inst_992_, v_inst_993_, v_f_994_);
lean_dec(v_inst_991_);
lean_dec(v_inst_990_);
lean_dec(v_inst_989_);
lean_dec(v_inst_988_);
lean_dec(v_inst_987_);
lean_dec_ref(v_inst_986_);
lean_dec_ref(v_inst_985_);
lean_dec_ref(v_inst_984_);
lean_dec_ref(v_inst_983_);
return v_res_995_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Algebra_Hom(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Prod(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Algebra_NonUnitalHom(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Algebra_Hom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Algebra_NonUnitalHom(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Algebra_Hom(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Prod(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Algebra_NonUnitalHom(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Algebra_Hom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Algebra_NonUnitalHom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Algebra_NonUnitalHom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Algebra_NonUnitalHom(builtin);
}
#ifdef __cplusplus
}
#endif
