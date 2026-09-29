// Lean compiler output
// Module: Mathlib.Algebra.Order.Hom.MonoidWithZero
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Order.GroupWithZero.Canonical public import Mathlib.Algebra.Order.Hom.Monoid
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
lean_object* lp_mathlib_Monoid_toMulOneClass___redArg(lean_object*);
lean_object* lp_mathlib_MulOneClass_toMulOne___redArg(lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* lp_mathlib_MonoidWithZeroHom_id___lam__0___boxed(lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_WithZero_unitsWithZeroEquiv___redArg(lean_object*);
lean_object* lp_mathlib_MulEquiv_withZero___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
lean_object* lp_mathlib_EquivLike_toEquiv___redArg(lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_MonoidWithZeroHom_comp___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_LinearOrderedCommGroupWithZero_toCommGroupWithZero___redArg(lean_object*);
lean_object* lp_mathlib_CommGroupWithZero_toGroupWithZero___redArg(lean_object*);
lean_object* lp_mathlib_WithZero_withZeroUnitsEquiv___redArg(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_term___u2192_x2a_u2080o___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 10, .m_data = "term_→*₀o_"};
static const lean_object* lp_mathlib_term___u2192_x2a_u2080o___00__closed__0 = (const lean_object*)&lp_mathlib_term___u2192_x2a_u2080o___00__closed__0_value;
static const lean_ctor_object lp_mathlib_term___u2192_x2a_u2080o___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2192_x2a_u2080o___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(22, 117, 122, 23, 6, 246, 41, 24)}};
static const lean_object* lp_mathlib_term___u2192_x2a_u2080o___00__closed__1 = (const lean_object*)&lp_mathlib_term___u2192_x2a_u2080o___00__closed__1_value;
static const lean_string_object lp_mathlib_term___u2192_x2a_u2080o___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_term___u2192_x2a_u2080o___00__closed__2 = (const lean_object*)&lp_mathlib_term___u2192_x2a_u2080o___00__closed__2_value;
static const lean_ctor_object lp_mathlib_term___u2192_x2a_u2080o___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2192_x2a_u2080o___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_term___u2192_x2a_u2080o___00__closed__3 = (const lean_object*)&lp_mathlib_term___u2192_x2a_u2080o___00__closed__3_value;
static const lean_string_object lp_mathlib_term___u2192_x2a_u2080o___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 6, .m_data = " →*₀o "};
static const lean_object* lp_mathlib_term___u2192_x2a_u2080o___00__closed__4 = (const lean_object*)&lp_mathlib_term___u2192_x2a_u2080o___00__closed__4_value;
static const lean_ctor_object lp_mathlib_term___u2192_x2a_u2080o___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192_x2a_u2080o___00__closed__4_value)}};
static const lean_object* lp_mathlib_term___u2192_x2a_u2080o___00__closed__5 = (const lean_object*)&lp_mathlib_term___u2192_x2a_u2080o___00__closed__5_value;
static const lean_string_object lp_mathlib_term___u2192_x2a_u2080o___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_term___u2192_x2a_u2080o___00__closed__6 = (const lean_object*)&lp_mathlib_term___u2192_x2a_u2080o___00__closed__6_value;
static const lean_ctor_object lp_mathlib_term___u2192_x2a_u2080o___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2192_x2a_u2080o___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_term___u2192_x2a_u2080o___00__closed__7 = (const lean_object*)&lp_mathlib_term___u2192_x2a_u2080o___00__closed__7_value;
static const lean_ctor_object lp_mathlib_term___u2192_x2a_u2080o___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192_x2a_u2080o___00__closed__7_value),((lean_object*)(((size_t)(25) << 1) | 1))}};
static const lean_object* lp_mathlib_term___u2192_x2a_u2080o___00__closed__8 = (const lean_object*)&lp_mathlib_term___u2192_x2a_u2080o___00__closed__8_value;
static const lean_ctor_object lp_mathlib_term___u2192_x2a_u2080o___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192_x2a_u2080o___00__closed__3_value),((lean_object*)&lp_mathlib_term___u2192_x2a_u2080o___00__closed__5_value),((lean_object*)&lp_mathlib_term___u2192_x2a_u2080o___00__closed__8_value)}};
static const lean_object* lp_mathlib_term___u2192_x2a_u2080o___00__closed__9 = (const lean_object*)&lp_mathlib_term___u2192_x2a_u2080o___00__closed__9_value;
static const lean_ctor_object lp_mathlib_term___u2192_x2a_u2080o___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192_x2a_u2080o___00__closed__1_value),((lean_object*)(((size_t)(25) << 1) | 1)),((lean_object*)(((size_t)(26) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2192_x2a_u2080o___00__closed__9_value)}};
static const lean_object* lp_mathlib_term___u2192_x2a_u2080o___00__closed__10 = (const lean_object*)&lp_mathlib_term___u2192_x2a_u2080o___00__closed__10_value;
LEAN_EXPORT const lean_object* lp_mathlib_term___u2192_x2a_u2080o__ = (const lean_object*)&lp_mathlib_term___u2192_x2a_u2080o___00__closed__10_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Order__Hom__MonoidWithZero______macroRules__term___u2192_x2a_u2080o____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__MonoidWithZero______macroRules__term___u2192_x2a_u2080o____1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__MonoidWithZero______macroRules__term___u2192_x2a_u2080o____1___closed__0_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Order__Hom__MonoidWithZero______macroRules__term___u2192_x2a_u2080o____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__MonoidWithZero______macroRules__term___u2192_x2a_u2080o____1___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__MonoidWithZero______macroRules__term___u2192_x2a_u2080o____1___closed__1_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Order__Hom__MonoidWithZero______macroRules__term___u2192_x2a_u2080o____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__MonoidWithZero______macroRules__term___u2192_x2a_u2080o____1___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__MonoidWithZero______macroRules__term___u2192_x2a_u2080o____1___closed__2_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Order__Hom__MonoidWithZero______macroRules__term___u2192_x2a_u2080o____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__MonoidWithZero______macroRules__term___u2192_x2a_u2080o____1___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__MonoidWithZero______macroRules__term___u2192_x2a_u2080o____1___closed__3_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Order__Hom__MonoidWithZero______macroRules__term___u2192_x2a_u2080o____1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__MonoidWithZero______macroRules__term___u2192_x2a_u2080o____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Order__Hom__MonoidWithZero______macroRules__term___u2192_x2a_u2080o____1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__MonoidWithZero______macroRules__term___u2192_x2a_u2080o____1___closed__4_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__MonoidWithZero______macroRules__term___u2192_x2a_u2080o____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Order__Hom__MonoidWithZero______macroRules__term___u2192_x2a_u2080o____1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__MonoidWithZero______macroRules__term___u2192_x2a_u2080o____1___closed__4_value_aux_1),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__MonoidWithZero______macroRules__term___u2192_x2a_u2080o____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Order__Hom__MonoidWithZero______macroRules__term___u2192_x2a_u2080o____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__MonoidWithZero______macroRules__term___u2192_x2a_u2080o____1___closed__4_value_aux_2),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__MonoidWithZero______macroRules__term___u2192_x2a_u2080o____1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__MonoidWithZero______macroRules__term___u2192_x2a_u2080o____1___closed__4 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__MonoidWithZero______macroRules__term___u2192_x2a_u2080o____1___closed__4_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Order__Hom__MonoidWithZero______macroRules__term___u2192_x2a_u2080o____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "OrderMonoidWithZeroHom"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__MonoidWithZero______macroRules__term___u2192_x2a_u2080o____1___closed__5 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__MonoidWithZero______macroRules__term___u2192_x2a_u2080o____1___closed__5_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Algebra__Order__Hom__MonoidWithZero______macroRules__term___u2192_x2a_u2080o____1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__MonoidWithZero______macroRules__term___u2192_x2a_u2080o____1___closed__6;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Order__Hom__MonoidWithZero______macroRules__term___u2192_x2a_u2080o____1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__MonoidWithZero______macroRules__term___u2192_x2a_u2080o____1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(0, 211, 133, 125, 128, 243, 129, 50)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__MonoidWithZero______macroRules__term___u2192_x2a_u2080o____1___closed__7 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__MonoidWithZero______macroRules__term___u2192_x2a_u2080o____1___closed__7_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Order__Hom__MonoidWithZero______macroRules__term___u2192_x2a_u2080o____1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__MonoidWithZero______macroRules__term___u2192_x2a_u2080o____1___closed__7_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__MonoidWithZero______macroRules__term___u2192_x2a_u2080o____1___closed__8 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__MonoidWithZero______macroRules__term___u2192_x2a_u2080o____1___closed__8_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Order__Hom__MonoidWithZero______macroRules__term___u2192_x2a_u2080o____1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__MonoidWithZero______macroRules__term___u2192_x2a_u2080o____1___closed__7_value)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__MonoidWithZero______macroRules__term___u2192_x2a_u2080o____1___closed__9 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__MonoidWithZero______macroRules__term___u2192_x2a_u2080o____1___closed__9_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Order__Hom__MonoidWithZero______macroRules__term___u2192_x2a_u2080o____1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__MonoidWithZero______macroRules__term___u2192_x2a_u2080o____1___closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__MonoidWithZero______macroRules__term___u2192_x2a_u2080o____1___closed__10 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__MonoidWithZero______macroRules__term___u2192_x2a_u2080o____1___closed__10_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Order__Hom__MonoidWithZero______macroRules__term___u2192_x2a_u2080o____1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__MonoidWithZero______macroRules__term___u2192_x2a_u2080o____1___closed__8_value),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__MonoidWithZero______macroRules__term___u2192_x2a_u2080o____1___closed__10_value)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__MonoidWithZero______macroRules__term___u2192_x2a_u2080o____1___closed__11 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__MonoidWithZero______macroRules__term___u2192_x2a_u2080o____1___closed__11_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Order__Hom__MonoidWithZero______macroRules__term___u2192_x2a_u2080o____1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__MonoidWithZero______macroRules__term___u2192_x2a_u2080o____1___closed__12 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__MonoidWithZero______macroRules__term___u2192_x2a_u2080o____1___closed__12_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Order__Hom__MonoidWithZero______macroRules__term___u2192_x2a_u2080o____1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__MonoidWithZero______macroRules__term___u2192_x2a_u2080o____1___closed__12_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__MonoidWithZero______macroRules__term___u2192_x2a_u2080o____1___closed__13 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__MonoidWithZero______macroRules__term___u2192_x2a_u2080o____1___closed__13_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__MonoidWithZero______macroRules__term___u2192_x2a_u2080o____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__MonoidWithZero______macroRules__term___u2192_x2a_u2080o____1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Order__Hom__MonoidWithZero______unexpand__OrderMonoidWithZeroHom__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__MonoidWithZero______unexpand__OrderMonoidWithZeroHom__1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__MonoidWithZero______unexpand__OrderMonoidWithZeroHom__1___closed__0_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Order__Hom__MonoidWithZero______unexpand__OrderMonoidWithZeroHom__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__MonoidWithZero______unexpand__OrderMonoidWithZeroHom__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__MonoidWithZero______unexpand__OrderMonoidWithZeroHom__1___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__MonoidWithZero______unexpand__OrderMonoidWithZeroHom__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__MonoidWithZero______unexpand__OrderMonoidWithZeroHom__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__MonoidWithZero______unexpand__OrderMonoidWithZeroHom__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidWithZeroHomClass_toOrderMonoidWithZeroHom___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidWithZeroHomClass_toOrderMonoidWithZeroHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidWithZeroHomClass_toOrderMonoidWithZeroHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCOrderMonoidWithZeroHomOfOrderHomClassOfMonoidWithZeroHomClass___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCOrderMonoidWithZeroHomOfOrderHomClassOfMonoidWithZeroHomClass(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidWithZeroHom_toOrderMonoidHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidWithZeroHom_toOrderMonoidHom___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidWithZeroHom_toOrderMonoidHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidWithZeroHom_toOrderMonoidHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidWithZeroHom_copy___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidWithZeroHom_copy___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidWithZeroHom_copy(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidWithZeroHom_copy___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_OrderMonoidWithZeroHom_id___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_MonoidWithZeroHom_id___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_OrderMonoidWithZeroHom_id___closed__0 = (const lean_object*)&lp_mathlib_OrderMonoidWithZeroHom_id___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidWithZeroHom_id(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidWithZeroHom_id___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidWithZeroHom_instInhabited(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidWithZeroHom_instInhabited___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidWithZeroHom_comp___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidWithZeroHom_comp___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidWithZeroHom_comp___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidWithZeroHom_comp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidWithZeroHom_comp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidWithZeroHom_instMul___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidWithZeroHom_instMul___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidWithZeroHom_instMul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidWithZeroHom_instMul(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidWithZeroHom_instMul___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidIso_unitsWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidIso_unitsWithZero(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidIso_unitsWithZero___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidIso_withZero___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidIso_withZero___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidIso_withZero___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidIso_withZero___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidIso_withZero___redArg___lam__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidIso_withZero___redArg___lam__3___boxed(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_OrderMonoidIso_withZero___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_OrderMonoidIso_withZero___redArg___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_OrderMonoidIso_withZero___redArg___closed__0 = (const lean_object*)&lp_mathlib_OrderMonoidIso_withZero___redArg___closed__0_value;
static const lean_closure_object lp_mathlib_OrderMonoidIso_withZero___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_OrderMonoidIso_withZero___redArg___lam__1, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_OrderMonoidIso_withZero___redArg___closed__1 = (const lean_object*)&lp_mathlib_OrderMonoidIso_withZero___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidIso_withZero___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidIso_withZero(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidIso_withZero___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidIso_withZeroUnits___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidIso_withZeroUnits(lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib___aux__Mathlib__Algebra__Order__Hom__MonoidWithZero______macroRules__term___u2192_x2a_u2080o____1___closed__6(void){
_start:
{
lean_object* v___x_36_; lean_object* v___x_37_; 
v___x_36_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Order__Hom__MonoidWithZero______macroRules__term___u2192_x2a_u2080o____1___closed__5));
v___x_37_ = l_String_toRawSubstring_x27(v___x_36_);
return v___x_37_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__MonoidWithZero______macroRules__term___u2192_x2a_u2080o____1(lean_object* v_x_54_, lean_object* v_a_55_, lean_object* v_a_56_){
_start:
{
lean_object* v___x_57_; uint8_t v___x_58_; 
v___x_57_ = ((lean_object*)(lp_mathlib_term___u2192_x2a_u2080o___00__closed__1));
lean_inc(v_x_54_);
v___x_58_ = l_Lean_Syntax_isOfKind(v_x_54_, v___x_57_);
if (v___x_58_ == 0)
{
lean_object* v___x_59_; lean_object* v___x_60_; 
lean_dec(v_x_54_);
v___x_59_ = lean_box(1);
v___x_60_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_60_, 0, v___x_59_);
lean_ctor_set(v___x_60_, 1, v_a_56_);
return v___x_60_;
}
else
{
lean_object* v_quotContext_61_; lean_object* v_currMacroScope_62_; lean_object* v_ref_63_; lean_object* v___x_64_; lean_object* v___x_65_; lean_object* v___x_66_; lean_object* v___x_67_; uint8_t v___x_68_; lean_object* v___x_69_; lean_object* v___x_70_; lean_object* v___x_71_; lean_object* v___x_72_; lean_object* v___x_73_; lean_object* v___x_74_; lean_object* v___x_75_; lean_object* v___x_76_; lean_object* v___x_77_; lean_object* v___x_78_; lean_object* v___x_79_; 
v_quotContext_61_ = lean_ctor_get(v_a_55_, 1);
v_currMacroScope_62_ = lean_ctor_get(v_a_55_, 2);
v_ref_63_ = lean_ctor_get(v_a_55_, 5);
v___x_64_ = lean_unsigned_to_nat(0u);
v___x_65_ = l_Lean_Syntax_getArg(v_x_54_, v___x_64_);
v___x_66_ = lean_unsigned_to_nat(2u);
v___x_67_ = l_Lean_Syntax_getArg(v_x_54_, v___x_66_);
lean_dec(v_x_54_);
v___x_68_ = 0;
v___x_69_ = l_Lean_SourceInfo_fromRef(v_ref_63_, v___x_68_);
v___x_70_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Order__Hom__MonoidWithZero______macroRules__term___u2192_x2a_u2080o____1___closed__4));
v___x_71_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__MonoidWithZero______macroRules__term___u2192_x2a_u2080o____1___closed__6, &lp_mathlib___aux__Mathlib__Algebra__Order__Hom__MonoidWithZero______macroRules__term___u2192_x2a_u2080o____1___closed__6_once, _init_lp_mathlib___aux__Mathlib__Algebra__Order__Hom__MonoidWithZero______macroRules__term___u2192_x2a_u2080o____1___closed__6);
v___x_72_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Order__Hom__MonoidWithZero______macroRules__term___u2192_x2a_u2080o____1___closed__7));
lean_inc(v_currMacroScope_62_);
lean_inc(v_quotContext_61_);
v___x_73_ = l_Lean_addMacroScope(v_quotContext_61_, v___x_72_, v_currMacroScope_62_);
v___x_74_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Order__Hom__MonoidWithZero______macroRules__term___u2192_x2a_u2080o____1___closed__11));
lean_inc_n(v___x_69_, 2);
v___x_75_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_75_, 0, v___x_69_);
lean_ctor_set(v___x_75_, 1, v___x_71_);
lean_ctor_set(v___x_75_, 2, v___x_73_);
lean_ctor_set(v___x_75_, 3, v___x_74_);
v___x_76_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Order__Hom__MonoidWithZero______macroRules__term___u2192_x2a_u2080o____1___closed__13));
v___x_77_ = l_Lean_Syntax_node2(v___x_69_, v___x_76_, v___x_65_, v___x_67_);
v___x_78_ = l_Lean_Syntax_node2(v___x_69_, v___x_70_, v___x_75_, v___x_77_);
v___x_79_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_79_, 0, v___x_78_);
lean_ctor_set(v___x_79_, 1, v_a_56_);
return v___x_79_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__MonoidWithZero______macroRules__term___u2192_x2a_u2080o____1___boxed(lean_object* v_x_80_, lean_object* v_a_81_, lean_object* v_a_82_){
_start:
{
lean_object* v_res_83_; 
v_res_83_ = lp_mathlib___aux__Mathlib__Algebra__Order__Hom__MonoidWithZero______macroRules__term___u2192_x2a_u2080o____1(v_x_80_, v_a_81_, v_a_82_);
lean_dec_ref(v_a_81_);
return v_res_83_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__MonoidWithZero______unexpand__OrderMonoidWithZeroHom__1(lean_object* v_x_87_, lean_object* v_a_88_, lean_object* v_a_89_){
_start:
{
lean_object* v___x_90_; uint8_t v___x_91_; 
v___x_90_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Order__Hom__MonoidWithZero______macroRules__term___u2192_x2a_u2080o____1___closed__4));
lean_inc(v_x_87_);
v___x_91_ = l_Lean_Syntax_isOfKind(v_x_87_, v___x_90_);
if (v___x_91_ == 0)
{
lean_object* v___x_92_; lean_object* v___x_93_; 
lean_dec(v_x_87_);
v___x_92_ = lean_box(0);
v___x_93_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_93_, 0, v___x_92_);
lean_ctor_set(v___x_93_, 1, v_a_89_);
return v___x_93_;
}
else
{
lean_object* v___x_94_; lean_object* v___x_95_; lean_object* v___x_96_; uint8_t v___x_97_; 
v___x_94_ = lean_unsigned_to_nat(0u);
v___x_95_ = l_Lean_Syntax_getArg(v_x_87_, v___x_94_);
v___x_96_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Order__Hom__MonoidWithZero______unexpand__OrderMonoidWithZeroHom__1___closed__1));
lean_inc(v___x_95_);
v___x_97_ = l_Lean_Syntax_isOfKind(v___x_95_, v___x_96_);
if (v___x_97_ == 0)
{
lean_object* v___x_98_; lean_object* v___x_99_; 
lean_dec(v___x_95_);
lean_dec(v_x_87_);
v___x_98_ = lean_box(0);
v___x_99_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_99_, 0, v___x_98_);
lean_ctor_set(v___x_99_, 1, v_a_89_);
return v___x_99_;
}
else
{
lean_object* v___x_100_; lean_object* v___x_101_; lean_object* v___x_102_; uint8_t v___x_103_; 
v___x_100_ = lean_unsigned_to_nat(1u);
v___x_101_ = l_Lean_Syntax_getArg(v_x_87_, v___x_100_);
lean_dec(v_x_87_);
v___x_102_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_101_);
v___x_103_ = l_Lean_Syntax_matchesNull(v___x_101_, v___x_102_);
if (v___x_103_ == 0)
{
lean_object* v___x_104_; lean_object* v___x_105_; 
lean_dec(v___x_101_);
lean_dec(v___x_95_);
v___x_104_ = lean_box(0);
v___x_105_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_105_, 0, v___x_104_);
lean_ctor_set(v___x_105_, 1, v_a_89_);
return v___x_105_;
}
else
{
lean_object* v___x_106_; lean_object* v___x_107_; lean_object* v_ref_108_; uint8_t v___x_109_; lean_object* v___x_110_; lean_object* v___x_111_; lean_object* v___x_112_; lean_object* v___x_113_; lean_object* v___x_114_; lean_object* v___x_115_; 
v___x_106_ = l_Lean_Syntax_getArg(v___x_101_, v___x_94_);
v___x_107_ = l_Lean_Syntax_getArg(v___x_101_, v___x_100_);
lean_dec(v___x_101_);
v_ref_108_ = l_Lean_replaceRef(v___x_95_, v_a_88_);
lean_dec(v___x_95_);
v___x_109_ = 0;
v___x_110_ = l_Lean_SourceInfo_fromRef(v_ref_108_, v___x_109_);
lean_dec(v_ref_108_);
v___x_111_ = ((lean_object*)(lp_mathlib_term___u2192_x2a_u2080o___00__closed__1));
v___x_112_ = ((lean_object*)(lp_mathlib_term___u2192_x2a_u2080o___00__closed__4));
lean_inc(v___x_110_);
v___x_113_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_113_, 0, v___x_110_);
lean_ctor_set(v___x_113_, 1, v___x_112_);
v___x_114_ = l_Lean_Syntax_node3(v___x_110_, v___x_111_, v___x_106_, v___x_113_, v___x_107_);
v___x_115_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_115_, 0, v___x_114_);
lean_ctor_set(v___x_115_, 1, v_a_89_);
return v___x_115_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__MonoidWithZero______unexpand__OrderMonoidWithZeroHom__1___boxed(lean_object* v_x_116_, lean_object* v_a_117_, lean_object* v_a_118_){
_start:
{
lean_object* v_res_119_; 
v_res_119_ = lp_mathlib___aux__Mathlib__Algebra__Order__Hom__MonoidWithZero______unexpand__OrderMonoidWithZeroHom__1(v_x_116_, v_a_117_, v_a_118_);
lean_dec(v_a_117_);
return v_res_119_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidWithZeroHomClass_toOrderMonoidWithZeroHom___redArg(lean_object* v_inst_120_, lean_object* v_f_121_){
_start:
{
lean_object* v___x_122_; 
v___x_122_ = lean_apply_1(v_inst_120_, v_f_121_);
return v___x_122_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidWithZeroHomClass_toOrderMonoidWithZeroHom(lean_object* v_F_123_, lean_object* v_00_u03b1_124_, lean_object* v_00_u03b2_125_, lean_object* v_inst_126_, lean_object* v_inst_127_, lean_object* v_inst_128_, lean_object* v_inst_129_, lean_object* v_inst_130_, lean_object* v_inst_131_, lean_object* v_inst_132_, lean_object* v_f_133_){
_start:
{
lean_object* v___x_134_; 
v___x_134_ = lean_apply_1(v_inst_130_, v_f_133_);
return v___x_134_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidWithZeroHomClass_toOrderMonoidWithZeroHom___boxed(lean_object* v_F_135_, lean_object* v_00_u03b1_136_, lean_object* v_00_u03b2_137_, lean_object* v_inst_138_, lean_object* v_inst_139_, lean_object* v_inst_140_, lean_object* v_inst_141_, lean_object* v_inst_142_, lean_object* v_inst_143_, lean_object* v_inst_144_, lean_object* v_f_145_){
_start:
{
lean_object* v_res_146_; 
v_res_146_ = lp_mathlib_OrderMonoidWithZeroHomClass_toOrderMonoidWithZeroHom(v_F_135_, v_00_u03b1_136_, v_00_u03b2_137_, v_inst_138_, v_inst_139_, v_inst_140_, v_inst_141_, v_inst_142_, v_inst_143_, v_inst_144_, v_f_145_);
lean_dec_ref(v_inst_141_);
lean_dec_ref(v_inst_140_);
lean_dec_ref(v_inst_139_);
lean_dec_ref(v_inst_138_);
return v_res_146_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCOrderMonoidWithZeroHomOfOrderHomClassOfMonoidWithZeroHomClass___redArg(lean_object* v_inst_147_, lean_object* v_inst_148_, lean_object* v_inst_149_, lean_object* v_inst_150_, lean_object* v_inst_151_){
_start:
{
lean_object* v___x_152_; 
v___x_152_ = lean_alloc_closure((void*)(lp_mathlib_OrderMonoidWithZeroHomClass_toOrderMonoidWithZeroHom___boxed), 11, 10);
lean_closure_set(v___x_152_, 0, lean_box(0));
lean_closure_set(v___x_152_, 1, lean_box(0));
lean_closure_set(v___x_152_, 2, lean_box(0));
lean_closure_set(v___x_152_, 3, v_inst_147_);
lean_closure_set(v___x_152_, 4, v_inst_148_);
lean_closure_set(v___x_152_, 5, v_inst_149_);
lean_closure_set(v___x_152_, 6, v_inst_150_);
lean_closure_set(v___x_152_, 7, v_inst_151_);
lean_closure_set(v___x_152_, 8, lean_box(0));
lean_closure_set(v___x_152_, 9, lean_box(0));
return v___x_152_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCOrderMonoidWithZeroHomOfOrderHomClassOfMonoidWithZeroHomClass(lean_object* v_F_153_, lean_object* v_00_u03b1_154_, lean_object* v_00_u03b2_155_, lean_object* v_inst_156_, lean_object* v_inst_157_, lean_object* v_inst_158_, lean_object* v_inst_159_, lean_object* v_inst_160_, lean_object* v_inst_161_, lean_object* v_inst_162_){
_start:
{
lean_object* v___x_163_; 
v___x_163_ = lean_alloc_closure((void*)(lp_mathlib_OrderMonoidWithZeroHomClass_toOrderMonoidWithZeroHom___boxed), 11, 10);
lean_closure_set(v___x_163_, 0, lean_box(0));
lean_closure_set(v___x_163_, 1, lean_box(0));
lean_closure_set(v___x_163_, 2, lean_box(0));
lean_closure_set(v___x_163_, 3, v_inst_156_);
lean_closure_set(v___x_163_, 4, v_inst_157_);
lean_closure_set(v___x_163_, 5, v_inst_158_);
lean_closure_set(v___x_163_, 6, v_inst_159_);
lean_closure_set(v___x_163_, 7, v_inst_160_);
lean_closure_set(v___x_163_, 8, lean_box(0));
lean_closure_set(v___x_163_, 9, lean_box(0));
return v___x_163_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidWithZeroHom_toOrderMonoidHom___redArg(lean_object* v_f_164_){
_start:
{
lean_inc(v_f_164_);
return v_f_164_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidWithZeroHom_toOrderMonoidHom___redArg___boxed(lean_object* v_f_165_){
_start:
{
lean_object* v_res_166_; 
v_res_166_ = lp_mathlib_OrderMonoidWithZeroHom_toOrderMonoidHom___redArg(v_f_165_);
lean_dec(v_f_165_);
return v_res_166_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidWithZeroHom_toOrderMonoidHom(lean_object* v_00_u03b1_167_, lean_object* v_00_u03b2_168_, lean_object* v_inst_169_, lean_object* v_inst_170_, lean_object* v_inst_171_, lean_object* v_inst_172_, lean_object* v_f_173_){
_start:
{
lean_inc(v_f_173_);
return v_f_173_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidWithZeroHom_toOrderMonoidHom___boxed(lean_object* v_00_u03b1_174_, lean_object* v_00_u03b2_175_, lean_object* v_inst_176_, lean_object* v_inst_177_, lean_object* v_inst_178_, lean_object* v_inst_179_, lean_object* v_f_180_){
_start:
{
lean_object* v_res_181_; 
v_res_181_ = lp_mathlib_OrderMonoidWithZeroHom_toOrderMonoidHom(v_00_u03b1_174_, v_00_u03b2_175_, v_inst_176_, v_inst_177_, v_inst_178_, v_inst_179_, v_f_180_);
lean_dec(v_f_180_);
lean_dec_ref(v_inst_179_);
lean_dec_ref(v_inst_178_);
lean_dec_ref(v_inst_177_);
lean_dec_ref(v_inst_176_);
return v_res_181_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidWithZeroHom_copy___redArg(lean_object* v_f_x27_182_){
_start:
{
lean_inc(v_f_x27_182_);
return v_f_x27_182_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidWithZeroHom_copy___redArg___boxed(lean_object* v_f_x27_183_){
_start:
{
lean_object* v_res_184_; 
v_res_184_ = lp_mathlib_OrderMonoidWithZeroHom_copy___redArg(v_f_x27_183_);
lean_dec(v_f_x27_183_);
return v_res_184_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidWithZeroHom_copy(lean_object* v_00_u03b1_185_, lean_object* v_00_u03b2_186_, lean_object* v_inst_187_, lean_object* v_inst_188_, lean_object* v_inst_189_, lean_object* v_inst_190_, lean_object* v_f_191_, lean_object* v_f_x27_192_, lean_object* v_h_193_){
_start:
{
lean_inc(v_f_x27_192_);
return v_f_x27_192_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidWithZeroHom_copy___boxed(lean_object* v_00_u03b1_194_, lean_object* v_00_u03b2_195_, lean_object* v_inst_196_, lean_object* v_inst_197_, lean_object* v_inst_198_, lean_object* v_inst_199_, lean_object* v_f_200_, lean_object* v_f_x27_201_, lean_object* v_h_202_){
_start:
{
lean_object* v_res_203_; 
v_res_203_ = lp_mathlib_OrderMonoidWithZeroHom_copy(v_00_u03b1_194_, v_00_u03b2_195_, v_inst_196_, v_inst_197_, v_inst_198_, v_inst_199_, v_f_200_, v_f_x27_201_, v_h_202_);
lean_dec(v_f_x27_201_);
lean_dec(v_f_200_);
lean_dec_ref(v_inst_199_);
lean_dec_ref(v_inst_198_);
lean_dec_ref(v_inst_197_);
lean_dec_ref(v_inst_196_);
return v_res_203_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidWithZeroHom_id(lean_object* v_00_u03b1_205_, lean_object* v_inst_206_, lean_object* v_inst_207_){
_start:
{
lean_object* v___f_208_; 
v___f_208_ = ((lean_object*)(lp_mathlib_OrderMonoidWithZeroHom_id___closed__0));
return v___f_208_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidWithZeroHom_id___boxed(lean_object* v_00_u03b1_209_, lean_object* v_inst_210_, lean_object* v_inst_211_){
_start:
{
lean_object* v_res_212_; 
v_res_212_ = lp_mathlib_OrderMonoidWithZeroHom_id(v_00_u03b1_209_, v_inst_210_, v_inst_211_);
lean_dec_ref(v_inst_211_);
lean_dec_ref(v_inst_210_);
return v_res_212_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidWithZeroHom_instInhabited(lean_object* v_00_u03b1_213_, lean_object* v_inst_214_, lean_object* v_inst_215_){
_start:
{
lean_object* v___f_216_; 
v___f_216_ = ((lean_object*)(lp_mathlib_OrderMonoidWithZeroHom_id___closed__0));
return v___f_216_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidWithZeroHom_instInhabited___boxed(lean_object* v_00_u03b1_217_, lean_object* v_inst_218_, lean_object* v_inst_219_){
_start:
{
lean_object* v_res_220_; 
v_res_220_ = lp_mathlib_OrderMonoidWithZeroHom_instInhabited(v_00_u03b1_217_, v_inst_218_, v_inst_219_);
lean_dec_ref(v_inst_219_);
lean_dec_ref(v_inst_218_);
return v_res_220_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidWithZeroHom_comp___redArg___lam__0(lean_object* v_f_221_, lean_object* v___y_222_){
_start:
{
lean_object* v___x_223_; 
v___x_223_ = lean_apply_1(v_f_221_, v___y_222_);
return v___x_223_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidWithZeroHom_comp___redArg___lam__1(lean_object* v_g_224_, lean_object* v___y_225_){
_start:
{
lean_object* v___x_226_; 
v___x_226_ = lean_apply_1(v_g_224_, v___y_225_);
return v___x_226_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidWithZeroHom_comp___redArg(lean_object* v_f_227_, lean_object* v_g_228_){
_start:
{
lean_object* v___f_229_; lean_object* v___f_230_; lean_object* v___x_231_; 
v___f_229_ = lean_alloc_closure((void*)(lp_mathlib_OrderMonoidWithZeroHom_comp___redArg___lam__0), 2, 1);
lean_closure_set(v___f_229_, 0, v_f_227_);
v___f_230_ = lean_alloc_closure((void*)(lp_mathlib_OrderMonoidWithZeroHom_comp___redArg___lam__1), 2, 1);
lean_closure_set(v___f_230_, 0, v_g_228_);
v___x_231_ = lp_mathlib_MonoidWithZeroHom_comp___redArg(v___f_229_, v___f_230_);
return v___x_231_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidWithZeroHom_comp(lean_object* v_00_u03b1_232_, lean_object* v_00_u03b2_233_, lean_object* v_00_u03b3_234_, lean_object* v_inst_235_, lean_object* v_inst_236_, lean_object* v_inst_237_, lean_object* v_inst_238_, lean_object* v_inst_239_, lean_object* v_inst_240_, lean_object* v_f_241_, lean_object* v_g_242_){
_start:
{
lean_object* v___x_243_; 
v___x_243_ = lp_mathlib_OrderMonoidWithZeroHom_comp___redArg(v_f_241_, v_g_242_);
return v___x_243_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidWithZeroHom_comp___boxed(lean_object* v_00_u03b1_244_, lean_object* v_00_u03b2_245_, lean_object* v_00_u03b3_246_, lean_object* v_inst_247_, lean_object* v_inst_248_, lean_object* v_inst_249_, lean_object* v_inst_250_, lean_object* v_inst_251_, lean_object* v_inst_252_, lean_object* v_f_253_, lean_object* v_g_254_){
_start:
{
lean_object* v_res_255_; 
v_res_255_ = lp_mathlib_OrderMonoidWithZeroHom_comp(v_00_u03b1_244_, v_00_u03b2_245_, v_00_u03b3_246_, v_inst_247_, v_inst_248_, v_inst_249_, v_inst_250_, v_inst_251_, v_inst_252_, v_f_253_, v_g_254_);
lean_dec_ref(v_inst_252_);
lean_dec_ref(v_inst_251_);
lean_dec_ref(v_inst_250_);
lean_dec_ref(v_inst_249_);
lean_dec_ref(v_inst_248_);
lean_dec_ref(v_inst_247_);
return v_res_255_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidWithZeroHom_instMul___redArg___lam__0(lean_object* v_toCommMonoidWithZero_256_, lean_object* v_f_257_, lean_object* v_g_258_, lean_object* v___y_259_){
_start:
{
lean_object* v_toCommMonoid_260_; lean_object* v___x_261_; lean_object* v___x_262_; lean_object* v_toMul_263_; lean_object* v___x_264_; lean_object* v___x_265_; lean_object* v___x_266_; 
v_toCommMonoid_260_ = lean_ctor_get(v_toCommMonoidWithZero_256_, 0);
v___x_261_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_toCommMonoid_260_);
v___x_262_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_261_);
v_toMul_263_ = lean_ctor_get(v___x_262_, 1);
lean_inc(v_toMul_263_);
lean_dec_ref(v___x_262_);
lean_inc(v___y_259_);
v___x_264_ = lean_apply_1(v_f_257_, v___y_259_);
v___x_265_ = lean_apply_1(v_g_258_, v___y_259_);
v___x_266_ = lean_apply_2(v_toMul_263_, v___x_264_, v___x_265_);
return v___x_266_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidWithZeroHom_instMul___redArg___lam__0___boxed(lean_object* v_toCommMonoidWithZero_267_, lean_object* v_f_268_, lean_object* v_g_269_, lean_object* v___y_270_){
_start:
{
lean_object* v_res_271_; 
v_res_271_ = lp_mathlib_OrderMonoidWithZeroHom_instMul___redArg___lam__0(v_toCommMonoidWithZero_267_, v_f_268_, v_g_269_, v___y_270_);
lean_dec_ref(v_toCommMonoidWithZero_267_);
return v_res_271_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidWithZeroHom_instMul___redArg(lean_object* v_inst_272_){
_start:
{
lean_object* v_toCommMonoidWithZero_273_; lean_object* v___f_274_; 
v_toCommMonoidWithZero_273_ = lean_ctor_get(v_inst_272_, 0);
lean_inc_ref(v_toCommMonoidWithZero_273_);
lean_dec_ref(v_inst_272_);
v___f_274_ = lean_alloc_closure((void*)(lp_mathlib_OrderMonoidWithZeroHom_instMul___redArg___lam__0___boxed), 4, 1);
lean_closure_set(v___f_274_, 0, v_toCommMonoidWithZero_273_);
return v___f_274_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidWithZeroHom_instMul(lean_object* v_00_u03b1_275_, lean_object* v_00_u03b2_276_, lean_object* v_inst_277_, lean_object* v_inst_278_){
_start:
{
lean_object* v___x_279_; 
v___x_279_ = lp_mathlib_OrderMonoidWithZeroHom_instMul___redArg(v_inst_278_);
return v___x_279_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidWithZeroHom_instMul___boxed(lean_object* v_00_u03b1_280_, lean_object* v_00_u03b2_281_, lean_object* v_inst_282_, lean_object* v_inst_283_){
_start:
{
lean_object* v_res_284_; 
v_res_284_ = lp_mathlib_OrderMonoidWithZeroHom_instMul(v_00_u03b1_280_, v_00_u03b2_281_, v_inst_282_, v_inst_283_);
lean_dec_ref(v_inst_282_);
return v_res_284_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidIso_unitsWithZero___redArg(lean_object* v_inst_285_){
_start:
{
lean_object* v___x_286_; 
v___x_286_ = lp_mathlib_WithZero_unitsWithZeroEquiv___redArg(v_inst_285_);
return v___x_286_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidIso_unitsWithZero(lean_object* v_00_u03b1_287_, lean_object* v_inst_288_, lean_object* v_inst_289_){
_start:
{
lean_object* v___x_290_; 
v___x_290_ = lp_mathlib_WithZero_unitsWithZeroEquiv___redArg(v_inst_288_);
return v___x_290_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidIso_unitsWithZero___boxed(lean_object* v_00_u03b1_291_, lean_object* v_inst_292_, lean_object* v_inst_293_){
_start:
{
lean_object* v_res_294_; 
v_res_294_ = lp_mathlib_OrderMonoidIso_unitsWithZero(v_00_u03b1_291_, v_inst_292_, v_inst_293_);
lean_dec_ref(v_inst_293_);
return v_res_294_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidIso_withZero___redArg___lam__0(lean_object* v_f_295_, lean_object* v___y_296_){
_start:
{
lean_object* v_invFun_297_; lean_object* v___x_298_; 
v_invFun_297_ = lean_ctor_get(v_f_295_, 1);
lean_inc(v_invFun_297_);
lean_dec_ref(v_f_295_);
v___x_298_ = lean_apply_1(v_invFun_297_, v___y_296_);
return v___x_298_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidIso_withZero___redArg___lam__1(lean_object* v_f_299_, lean_object* v___y_300_){
_start:
{
lean_object* v_toFun_301_; lean_object* v___x_302_; 
v_toFun_301_ = lean_ctor_get(v_f_299_, 0);
lean_inc(v_toFun_301_);
lean_dec_ref(v_f_299_);
v___x_302_ = lean_apply_1(v_toFun_301_, v___y_300_);
return v___x_302_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidIso_withZero___redArg___lam__2(lean_object* v_inst_303_, lean_object* v_inst_304_, lean_object* v___f_305_, lean_object* v___f_306_, lean_object* v_e_307_){
_start:
{
lean_object* v___x_308_; lean_object* v___x_309_; lean_object* v___x_310_; lean_object* v_toFun_311_; lean_object* v___x_312_; lean_object* v___x_313_; 
v___x_308_ = lp_mathlib_MulEquiv_withZero___redArg(v_inst_303_, v_inst_304_);
v___x_309_ = lp_mathlib_Equiv_symm___redArg(v___x_308_);
v___x_310_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_310_, 0, v___f_305_);
lean_ctor_set(v___x_310_, 1, v___f_306_);
v_toFun_311_ = lean_ctor_get(v___x_309_, 0);
lean_inc(v_toFun_311_);
lean_dec_ref(v___x_309_);
v___x_312_ = lp_mathlib_EquivLike_toEquiv___redArg(v___x_310_, v_e_307_);
v___x_313_ = lean_apply_1(v_toFun_311_, v___x_312_);
return v___x_313_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidIso_withZero___redArg___lam__2___boxed(lean_object* v_inst_314_, lean_object* v_inst_315_, lean_object* v___f_316_, lean_object* v___f_317_, lean_object* v_e_318_){
_start:
{
lean_object* v_res_319_; 
v_res_319_ = lp_mathlib_OrderMonoidIso_withZero___redArg___lam__2(v_inst_314_, v_inst_315_, v___f_316_, v___f_317_, v_e_318_);
lean_dec_ref(v_inst_315_);
lean_dec_ref(v_inst_314_);
return v_res_319_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidIso_withZero___redArg___lam__3(lean_object* v_inst_320_, lean_object* v_inst_321_, lean_object* v_e_322_){
_start:
{
lean_object* v___x_323_; lean_object* v_toFun_324_; lean_object* v___x_325_; 
v___x_323_ = lp_mathlib_MulEquiv_withZero___redArg(v_inst_320_, v_inst_321_);
v_toFun_324_ = lean_ctor_get(v___x_323_, 0);
lean_inc(v_toFun_324_);
lean_dec_ref(v___x_323_);
v___x_325_ = lean_apply_1(v_toFun_324_, v_e_322_);
return v___x_325_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidIso_withZero___redArg___lam__3___boxed(lean_object* v_inst_326_, lean_object* v_inst_327_, lean_object* v_e_328_){
_start:
{
lean_object* v_res_329_; 
v_res_329_ = lp_mathlib_OrderMonoidIso_withZero___redArg___lam__3(v_inst_326_, v_inst_327_, v_e_328_);
lean_dec_ref(v_inst_327_);
lean_dec_ref(v_inst_326_);
return v_res_329_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidIso_withZero___redArg(lean_object* v_inst_332_, lean_object* v_inst_333_){
_start:
{
lean_object* v___f_334_; lean_object* v___f_335_; lean_object* v___f_336_; lean_object* v___f_337_; lean_object* v___x_338_; 
v___f_334_ = ((lean_object*)(lp_mathlib_OrderMonoidIso_withZero___redArg___closed__0));
v___f_335_ = ((lean_object*)(lp_mathlib_OrderMonoidIso_withZero___redArg___closed__1));
lean_inc_ref(v_inst_333_);
lean_inc_ref(v_inst_332_);
v___f_336_ = lean_alloc_closure((void*)(lp_mathlib_OrderMonoidIso_withZero___redArg___lam__2___boxed), 5, 4);
lean_closure_set(v___f_336_, 0, v_inst_332_);
lean_closure_set(v___f_336_, 1, v_inst_333_);
lean_closure_set(v___f_336_, 2, v___f_335_);
lean_closure_set(v___f_336_, 3, v___f_334_);
v___f_337_ = lean_alloc_closure((void*)(lp_mathlib_OrderMonoidIso_withZero___redArg___lam__3___boxed), 3, 2);
lean_closure_set(v___f_337_, 0, v_inst_332_);
lean_closure_set(v___f_337_, 1, v_inst_333_);
v___x_338_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_338_, 0, v___f_337_);
lean_ctor_set(v___x_338_, 1, v___f_336_);
return v___x_338_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidIso_withZero(lean_object* v_G_339_, lean_object* v_H_340_, lean_object* v_inst_341_, lean_object* v_inst_342_, lean_object* v_inst_343_, lean_object* v_inst_344_){
_start:
{
lean_object* v___x_345_; 
v___x_345_ = lp_mathlib_OrderMonoidIso_withZero___redArg(v_inst_341_, v_inst_343_);
return v___x_345_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidIso_withZero___boxed(lean_object* v_G_346_, lean_object* v_H_347_, lean_object* v_inst_348_, lean_object* v_inst_349_, lean_object* v_inst_350_, lean_object* v_inst_351_){
_start:
{
lean_object* v_res_352_; 
v_res_352_ = lp_mathlib_OrderMonoidIso_withZero(v_G_346_, v_H_347_, v_inst_348_, v_inst_349_, v_inst_350_, v_inst_351_);
lean_dec_ref(v_inst_351_);
lean_dec_ref(v_inst_349_);
return v_res_352_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidIso_withZeroUnits___redArg(lean_object* v_inst_353_, lean_object* v_inst_354_){
_start:
{
lean_object* v___x_355_; lean_object* v___x_356_; lean_object* v___x_357_; 
v___x_355_ = lp_mathlib_LinearOrderedCommGroupWithZero_toCommGroupWithZero___redArg(v_inst_353_);
v___x_356_ = lp_mathlib_CommGroupWithZero_toGroupWithZero___redArg(v___x_355_);
v___x_357_ = lp_mathlib_WithZero_withZeroUnitsEquiv___redArg(v___x_356_, v_inst_354_);
return v___x_357_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidIso_withZeroUnits(lean_object* v_00_u03b1_358_, lean_object* v_inst_359_, lean_object* v_inst_360_){
_start:
{
lean_object* v___x_361_; 
v___x_361_ = lp_mathlib_OrderMonoidIso_withZeroUnits___redArg(v_inst_359_, v_inst_360_);
return v___x_361_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_GroupWithZero_Canonical(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Hom_Monoid(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Hom_MonoidWithZero(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_GroupWithZero_Canonical(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Hom_Monoid(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Order_Hom_MonoidWithZero(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Order_GroupWithZero_Canonical(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Hom_Monoid(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Order_Hom_MonoidWithZero(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_GroupWithZero_Canonical(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Hom_Monoid(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Hom_MonoidWithZero(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Order_Hom_MonoidWithZero(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Order_Hom_MonoidWithZero(builtin);
}
#ifdef __cplusplus
}
#endif
