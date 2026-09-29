// Lean compiler output
// Module: Mathlib.Algebra.Order.Hom.Monoid
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Equiv.Defs public import Mathlib.Algebra.Group.Hom.Basic public import Mathlib.Algebra.Order.Group.Unbundled.Basic public import Mathlib.Algebra.Order.Monoid.OrderDual public import Mathlib.Order.Hom.Basic
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
lean_object* lp_mathlib_Equiv_refl(lean_object*);
lean_object* lp_mathlib_EquivLike_toEquiv___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_AddZeroClass_toAddZero___redArg(lean_object*);
lean_object* lp_mathlib_Equiv_trans___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
lean_object* lp_mathlib_OneHom_comp___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* lp_mathlib_Monoid_toMulOneClass___redArg(lean_object*);
lean_object* lp_mathlib_MulOneClass_toMulOne___redArg(lean_object*);
lean_object* lp_mathlib_OneHom_id___lam__0___boxed(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_term___u2192_x2bo___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 9, .m_data = "term_→+o_"};
static const lean_object* lp_mathlib_term___u2192_x2bo___00__closed__0 = (const lean_object*)&lp_mathlib_term___u2192_x2bo___00__closed__0_value;
static const lean_ctor_object lp_mathlib_term___u2192_x2bo___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2192_x2bo___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(55, 176, 40, 210, 53, 180, 65, 117)}};
static const lean_object* lp_mathlib_term___u2192_x2bo___00__closed__1 = (const lean_object*)&lp_mathlib_term___u2192_x2bo___00__closed__1_value;
static const lean_string_object lp_mathlib_term___u2192_x2bo___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_term___u2192_x2bo___00__closed__2 = (const lean_object*)&lp_mathlib_term___u2192_x2bo___00__closed__2_value;
static const lean_ctor_object lp_mathlib_term___u2192_x2bo___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2192_x2bo___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_term___u2192_x2bo___00__closed__3 = (const lean_object*)&lp_mathlib_term___u2192_x2bo___00__closed__3_value;
static const lean_string_object lp_mathlib_term___u2192_x2bo___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 5, .m_data = " →+o "};
static const lean_object* lp_mathlib_term___u2192_x2bo___00__closed__4 = (const lean_object*)&lp_mathlib_term___u2192_x2bo___00__closed__4_value;
static const lean_ctor_object lp_mathlib_term___u2192_x2bo___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192_x2bo___00__closed__4_value)}};
static const lean_object* lp_mathlib_term___u2192_x2bo___00__closed__5 = (const lean_object*)&lp_mathlib_term___u2192_x2bo___00__closed__5_value;
static const lean_string_object lp_mathlib_term___u2192_x2bo___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_term___u2192_x2bo___00__closed__6 = (const lean_object*)&lp_mathlib_term___u2192_x2bo___00__closed__6_value;
static const lean_ctor_object lp_mathlib_term___u2192_x2bo___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2192_x2bo___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_term___u2192_x2bo___00__closed__7 = (const lean_object*)&lp_mathlib_term___u2192_x2bo___00__closed__7_value;
static const lean_ctor_object lp_mathlib_term___u2192_x2bo___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192_x2bo___00__closed__7_value),((lean_object*)(((size_t)(25) << 1) | 1))}};
static const lean_object* lp_mathlib_term___u2192_x2bo___00__closed__8 = (const lean_object*)&lp_mathlib_term___u2192_x2bo___00__closed__8_value;
static const lean_ctor_object lp_mathlib_term___u2192_x2bo___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192_x2bo___00__closed__3_value),((lean_object*)&lp_mathlib_term___u2192_x2bo___00__closed__5_value),((lean_object*)&lp_mathlib_term___u2192_x2bo___00__closed__8_value)}};
static const lean_object* lp_mathlib_term___u2192_x2bo___00__closed__9 = (const lean_object*)&lp_mathlib_term___u2192_x2bo___00__closed__9_value;
static const lean_ctor_object lp_mathlib_term___u2192_x2bo___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192_x2bo___00__closed__1_value),((lean_object*)(((size_t)(25) << 1) | 1)),((lean_object*)(((size_t)(26) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2192_x2bo___00__closed__9_value)}};
static const lean_object* lp_mathlib_term___u2192_x2bo___00__closed__10 = (const lean_object*)&lp_mathlib_term___u2192_x2bo___00__closed__10_value;
LEAN_EXPORT const lean_object* lp_mathlib_term___u2192_x2bo__ = (const lean_object*)&lp_mathlib_term___u2192_x2bo___00__closed__10_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2bo____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2bo____1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2bo____1___closed__0_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2bo____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2bo____1___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2bo____1___closed__1_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2bo____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2bo____1___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2bo____1___closed__2_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2bo____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2bo____1___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2bo____1___closed__3_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2bo____1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2bo____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2bo____1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2bo____1___closed__4_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2bo____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2bo____1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2bo____1___closed__4_value_aux_1),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2bo____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2bo____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2bo____1___closed__4_value_aux_2),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2bo____1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2bo____1___closed__4 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2bo____1___closed__4_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2bo____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "OrderAddMonoidHom"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2bo____1___closed__5 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2bo____1___closed__5_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2bo____1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2bo____1___closed__6;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2bo____1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2bo____1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(254, 18, 58, 229, 252, 105, 124, 213)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2bo____1___closed__7 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2bo____1___closed__7_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2bo____1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2bo____1___closed__7_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2bo____1___closed__8 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2bo____1___closed__8_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2bo____1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2bo____1___closed__7_value)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2bo____1___closed__9 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2bo____1___closed__9_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2bo____1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2bo____1___closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2bo____1___closed__10 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2bo____1___closed__10_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2bo____1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2bo____1___closed__8_value),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2bo____1___closed__10_value)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2bo____1___closed__11 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2bo____1___closed__11_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2bo____1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2bo____1___closed__12 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2bo____1___closed__12_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2bo____1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2bo____1___closed__12_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2bo____1___closed__13 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2bo____1___closed__13_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2bo____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2bo____1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______unexpand__OrderAddMonoidHom__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______unexpand__OrderAddMonoidHom__1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______unexpand__OrderAddMonoidHom__1___closed__0_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______unexpand__OrderAddMonoidHom__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______unexpand__OrderAddMonoidHom__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______unexpand__OrderAddMonoidHom__1___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______unexpand__OrderAddMonoidHom__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______unexpand__OrderAddMonoidHom__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______unexpand__OrderAddMonoidHom__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_term___u2243_x2bo___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 9, .m_data = "term_≃+o_"};
static const lean_object* lp_mathlib_term___u2243_x2bo___00__closed__0 = (const lean_object*)&lp_mathlib_term___u2243_x2bo___00__closed__0_value;
static const lean_ctor_object lp_mathlib_term___u2243_x2bo___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2243_x2bo___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(121, 122, 204, 180, 206, 208, 105, 85)}};
static const lean_object* lp_mathlib_term___u2243_x2bo___00__closed__1 = (const lean_object*)&lp_mathlib_term___u2243_x2bo___00__closed__1_value;
static const lean_string_object lp_mathlib_term___u2243_x2bo___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 5, .m_data = " ≃+o "};
static const lean_object* lp_mathlib_term___u2243_x2bo___00__closed__2 = (const lean_object*)&lp_mathlib_term___u2243_x2bo___00__closed__2_value;
static const lean_ctor_object lp_mathlib_term___u2243_x2bo___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_term___u2243_x2bo___00__closed__2_value)}};
static const lean_object* lp_mathlib_term___u2243_x2bo___00__closed__3 = (const lean_object*)&lp_mathlib_term___u2243_x2bo___00__closed__3_value;
static const lean_ctor_object lp_mathlib_term___u2243_x2bo___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192_x2bo___00__closed__3_value),((lean_object*)&lp_mathlib_term___u2243_x2bo___00__closed__3_value),((lean_object*)&lp_mathlib_term___u2192_x2bo___00__closed__8_value)}};
static const lean_object* lp_mathlib_term___u2243_x2bo___00__closed__4 = (const lean_object*)&lp_mathlib_term___u2243_x2bo___00__closed__4_value;
static const lean_ctor_object lp_mathlib_term___u2243_x2bo___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_term___u2243_x2bo___00__closed__1_value),((lean_object*)(((size_t)(25) << 1) | 1)),((lean_object*)(((size_t)(26) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2243_x2bo___00__closed__4_value)}};
static const lean_object* lp_mathlib_term___u2243_x2bo___00__closed__5 = (const lean_object*)&lp_mathlib_term___u2243_x2bo___00__closed__5_value;
LEAN_EXPORT const lean_object* lp_mathlib_term___u2243_x2bo__ = (const lean_object*)&lp_mathlib_term___u2243_x2bo___00__closed__5_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2243_x2bo____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "OrderAddMonoidIso"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2243_x2bo____1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2243_x2bo____1___closed__0_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2243_x2bo____1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2243_x2bo____1___closed__1;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2243_x2bo____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2243_x2bo____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(24, 63, 79, 213, 198, 94, 177, 14)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2243_x2bo____1___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2243_x2bo____1___closed__2_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2243_x2bo____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2243_x2bo____1___closed__2_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2243_x2bo____1___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2243_x2bo____1___closed__3_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2243_x2bo____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2243_x2bo____1___closed__2_value)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2243_x2bo____1___closed__4 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2243_x2bo____1___closed__4_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2243_x2bo____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2243_x2bo____1___closed__4_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2243_x2bo____1___closed__5 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2243_x2bo____1___closed__5_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2243_x2bo____1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2243_x2bo____1___closed__3_value),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2243_x2bo____1___closed__5_value)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2243_x2bo____1___closed__6 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2243_x2bo____1___closed__6_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2243_x2bo____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2243_x2bo____1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______unexpand__OrderAddMonoidIso__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______unexpand__OrderAddMonoidIso__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_term___u2192_x2ao___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 9, .m_data = "term_→*o_"};
static const lean_object* lp_mathlib_term___u2192_x2ao___00__closed__0 = (const lean_object*)&lp_mathlib_term___u2192_x2ao___00__closed__0_value;
static const lean_ctor_object lp_mathlib_term___u2192_x2ao___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2192_x2ao___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(81, 154, 166, 42, 87, 118, 6, 160)}};
static const lean_object* lp_mathlib_term___u2192_x2ao___00__closed__1 = (const lean_object*)&lp_mathlib_term___u2192_x2ao___00__closed__1_value;
static const lean_string_object lp_mathlib_term___u2192_x2ao___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 5, .m_data = " →*o "};
static const lean_object* lp_mathlib_term___u2192_x2ao___00__closed__2 = (const lean_object*)&lp_mathlib_term___u2192_x2ao___00__closed__2_value;
static const lean_ctor_object lp_mathlib_term___u2192_x2ao___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192_x2ao___00__closed__2_value)}};
static const lean_object* lp_mathlib_term___u2192_x2ao___00__closed__3 = (const lean_object*)&lp_mathlib_term___u2192_x2ao___00__closed__3_value;
static const lean_ctor_object lp_mathlib_term___u2192_x2ao___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192_x2bo___00__closed__3_value),((lean_object*)&lp_mathlib_term___u2192_x2ao___00__closed__3_value),((lean_object*)&lp_mathlib_term___u2192_x2bo___00__closed__8_value)}};
static const lean_object* lp_mathlib_term___u2192_x2ao___00__closed__4 = (const lean_object*)&lp_mathlib_term___u2192_x2ao___00__closed__4_value;
static const lean_ctor_object lp_mathlib_term___u2192_x2ao___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192_x2ao___00__closed__1_value),((lean_object*)(((size_t)(25) << 1) | 1)),((lean_object*)(((size_t)(26) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2192_x2ao___00__closed__4_value)}};
static const lean_object* lp_mathlib_term___u2192_x2ao___00__closed__5 = (const lean_object*)&lp_mathlib_term___u2192_x2ao___00__closed__5_value;
LEAN_EXPORT const lean_object* lp_mathlib_term___u2192_x2ao__ = (const lean_object*)&lp_mathlib_term___u2192_x2ao___00__closed__5_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2ao____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "OrderMonoidHom"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2ao____1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2ao____1___closed__0_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2ao____1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2ao____1___closed__1;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2ao____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2ao____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(227, 65, 132, 70, 187, 28, 247, 17)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2ao____1___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2ao____1___closed__2_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2ao____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2ao____1___closed__2_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2ao____1___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2ao____1___closed__3_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2ao____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2ao____1___closed__2_value)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2ao____1___closed__4 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2ao____1___closed__4_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2ao____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2ao____1___closed__4_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2ao____1___closed__5 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2ao____1___closed__5_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2ao____1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2ao____1___closed__3_value),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2ao____1___closed__5_value)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2ao____1___closed__6 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2ao____1___closed__6_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2ao____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2ao____1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______unexpand__OrderMonoidHom__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______unexpand__OrderMonoidHom__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidHomClass_toOrderMonoidHom___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidHomClass_toOrderMonoidHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidHomClass_toOrderMonoidHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidHomClass_toOrderAddMonoidHom___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidHomClass_toOrderAddMonoidHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidHomClass_toOrderAddMonoidHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCOrderMonoidHomOfOrderHomClassOfMonoidHomClass___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCOrderMonoidHomOfOrderHomClassOfMonoidHomClass(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCOrderAddMonoidHomOfOrderHomClassOfAddMonoidHomClass___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCOrderAddMonoidHomOfOrderHomClassOfAddMonoidHomClass(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_term___u2243_x2ao___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 9, .m_data = "term_≃*o_"};
static const lean_object* lp_mathlib_term___u2243_x2ao___00__closed__0 = (const lean_object*)&lp_mathlib_term___u2243_x2ao___00__closed__0_value;
static const lean_ctor_object lp_mathlib_term___u2243_x2ao___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2243_x2ao___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(162, 168, 132, 255, 152, 10, 124, 71)}};
static const lean_object* lp_mathlib_term___u2243_x2ao___00__closed__1 = (const lean_object*)&lp_mathlib_term___u2243_x2ao___00__closed__1_value;
static const lean_string_object lp_mathlib_term___u2243_x2ao___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 5, .m_data = " ≃*o "};
static const lean_object* lp_mathlib_term___u2243_x2ao___00__closed__2 = (const lean_object*)&lp_mathlib_term___u2243_x2ao___00__closed__2_value;
static const lean_ctor_object lp_mathlib_term___u2243_x2ao___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_term___u2243_x2ao___00__closed__2_value)}};
static const lean_object* lp_mathlib_term___u2243_x2ao___00__closed__3 = (const lean_object*)&lp_mathlib_term___u2243_x2ao___00__closed__3_value;
static const lean_ctor_object lp_mathlib_term___u2243_x2ao___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192_x2bo___00__closed__3_value),((lean_object*)&lp_mathlib_term___u2243_x2ao___00__closed__3_value),((lean_object*)&lp_mathlib_term___u2192_x2bo___00__closed__8_value)}};
static const lean_object* lp_mathlib_term___u2243_x2ao___00__closed__4 = (const lean_object*)&lp_mathlib_term___u2243_x2ao___00__closed__4_value;
static const lean_ctor_object lp_mathlib_term___u2243_x2ao___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_term___u2243_x2ao___00__closed__1_value),((lean_object*)(((size_t)(25) << 1) | 1)),((lean_object*)(((size_t)(26) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2243_x2ao___00__closed__4_value)}};
static const lean_object* lp_mathlib_term___u2243_x2ao___00__closed__5 = (const lean_object*)&lp_mathlib_term___u2243_x2ao___00__closed__5_value;
LEAN_EXPORT const lean_object* lp_mathlib_term___u2243_x2ao__ = (const lean_object*)&lp_mathlib_term___u2243_x2ao___00__closed__5_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2243_x2ao____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "OrderMonoidIso"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2243_x2ao____1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2243_x2ao____1___closed__0_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2243_x2ao____1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2243_x2ao____1___closed__1;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2243_x2ao____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2243_x2ao____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(196, 57, 203, 140, 118, 125, 137, 178)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2243_x2ao____1___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2243_x2ao____1___closed__2_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2243_x2ao____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2243_x2ao____1___closed__2_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2243_x2ao____1___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2243_x2ao____1___closed__3_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2243_x2ao____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2243_x2ao____1___closed__2_value)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2243_x2ao____1___closed__4 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2243_x2ao____1___closed__4_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2243_x2ao____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2243_x2ao____1___closed__4_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2243_x2ao____1___closed__5 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2243_x2ao____1___closed__5_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2243_x2ao____1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2243_x2ao____1___closed__3_value),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2243_x2ao____1___closed__5_value)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2243_x2ao____1___closed__6 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2243_x2ao____1___closed__6_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2243_x2ao____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2243_x2ao____1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______unexpand__OrderMonoidIso__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______unexpand__OrderMonoidIso__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidIsoClass_toOrderMonoidIso___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidIsoClass_toOrderMonoidIso(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidIsoClass_toOrderMonoidIso___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidIsoClass_toOrderAddMonoidIso___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidIsoClass_toOrderAddMonoidIso(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidIsoClass_toOrderAddMonoidIso___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCOrderMonoidIsoOfOrderIsoClassOfMulEquivClass___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCOrderMonoidIsoOfOrderIsoClassOfMulEquivClass(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCOrderAddMonoidIsoOfOrderIsoClassOfAddEquivClass___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCOrderAddMonoidIsoOfOrderIsoClassOfAddEquivClass(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidHom_instFunLike___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_OrderMonoidHom_instFunLike___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_OrderMonoidHom_instFunLike___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_OrderMonoidHom_instFunLike___closed__0 = (const lean_object*)&lp_mathlib_OrderMonoidHom_instFunLike___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidHom_instFunLike(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidHom_instFunLike___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidHom_instFunLike(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidHom_instFunLike___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidHom_toOrderHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidHom_toOrderHom___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidHom_toOrderHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidHom_toOrderHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidHom_toOrderHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidHom_toOrderHom___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidHom_toOrderHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidHom_toOrderHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidHom_copy___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidHom_copy___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidHom_copy(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidHom_copy___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidHom_copy___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidHom_copy___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidHom_copy(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidHom_copy___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_OrderMonoidHom_id___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_OneHom_id___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_OrderMonoidHom_id___closed__0 = (const lean_object*)&lp_mathlib_OrderMonoidHom_id___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidHom_id(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidHom_id___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidHom_id(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidHom_id___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidHom_instInhabited(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidHom_instInhabited___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidHom_instInhabited(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidHom_instInhabited___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidHom_comp___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidHom_comp___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidHom_comp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidHom_comp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidHom_comp___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidHom_comp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidHom_comp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidHom_instOne___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidHom_instOne___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidHom_instOne___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidHom_instOne(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidHom_instOne___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidHom_instZero___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidHom_instZero___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidHom_instZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidHom_instZero(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidHom_instZero___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidHom_instMulOfIsOrderedMonoid___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidHom_instMulOfIsOrderedMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidHom_instMulOfIsOrderedMonoid___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidHom_instMulOfIsOrderedMonoid(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidHom_instMulOfIsOrderedMonoid___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidHom_instAddOfIsOrderedAddMonoid___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidHom_instAddOfIsOrderedAddMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidHom_instAddOfIsOrderedAddMonoid___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidHom_instAddOfIsOrderedAddMonoid(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidHom_instAddOfIsOrderedAddMonoid___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidHom_mk_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidHom_mk_x27___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidHom_mk_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidHom_mk_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidHom_mk_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidHom_mk_x27___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidHom_mk_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidHom_mk_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidIso_instEquivLike___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidIso_instEquivLike___lam__1(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_OrderMonoidIso_instEquivLike___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_OrderMonoidIso_instEquivLike___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_OrderMonoidIso_instEquivLike___closed__0 = (const lean_object*)&lp_mathlib_OrderMonoidIso_instEquivLike___closed__0_value;
static const lean_closure_object lp_mathlib_OrderMonoidIso_instEquivLike___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_OrderMonoidIso_instEquivLike___lam__1, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_OrderMonoidIso_instEquivLike___closed__1 = (const lean_object*)&lp_mathlib_OrderMonoidIso_instEquivLike___closed__1_value;
static const lean_ctor_object lp_mathlib_OrderMonoidIso_instEquivLike___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_OrderMonoidIso_instEquivLike___closed__0_value),((lean_object*)&lp_mathlib_OrderMonoidIso_instEquivLike___closed__1_value)}};
static const lean_object* lp_mathlib_OrderMonoidIso_instEquivLike___closed__2 = (const lean_object*)&lp_mathlib_OrderMonoidIso_instEquivLike___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidIso_instEquivLike(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidIso_instEquivLike___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidIso_instEquivLike(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidIso_instEquivLike___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidIso_toOrderIso___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidIso_toOrderIso___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidIso_toOrderIso(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidIso_toOrderIso___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidIso_toOrderIso___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidIso_toOrderIso___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidIso_toOrderIso(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidIso_toOrderIso___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_OrderMonoidIso_refl___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_OrderMonoidIso_refl___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidIso_refl(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidIso_refl___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidIso_refl(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidIso_refl___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidIso_instInhabited(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidIso_instInhabited___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidIso_instInhabited(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidIso_instInhabited___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidIso_trans___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidIso_trans(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidIso_trans___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidIso_trans___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidIso_trans(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidIso_trans___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidIso_symm___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidIso_symm(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidIso_symm___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidIso_symm___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidIso_symm(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidIso_symm___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidIso_Simps_apply___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidIso_Simps_apply(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidIso_Simps_apply___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidIso_Simps_apply___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidIso_Simps_apply(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidIso_Simps_apply___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidIso_Simps_symm__apply___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidIso_Simps_symm__apply(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidIso_Simps_symm__apply___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidIso_Simps_symm__apply___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidIso_Simps_symm__apply(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidIso_Simps_symm__apply___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidIso_mk_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidIso_mk_x27___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidIso_mk_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidIso_mk_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidIso_mk_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidIso_mk_x27___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidIso_mk_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidIso_mk_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2bo____1___closed__6(void){
_start:
{
lean_object* v___x_36_; lean_object* v___x_37_; 
v___x_36_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2bo____1___closed__5));
v___x_37_ = l_String_toRawSubstring_x27(v___x_36_);
return v___x_37_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2bo____1(lean_object* v_x_54_, lean_object* v_a_55_, lean_object* v_a_56_){
_start:
{
lean_object* v___x_57_; uint8_t v___x_58_; 
v___x_57_ = ((lean_object*)(lp_mathlib_term___u2192_x2bo___00__closed__1));
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
v___x_70_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2bo____1___closed__4));
v___x_71_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2bo____1___closed__6, &lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2bo____1___closed__6_once, _init_lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2bo____1___closed__6);
v___x_72_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2bo____1___closed__7));
lean_inc(v_currMacroScope_62_);
lean_inc(v_quotContext_61_);
v___x_73_ = l_Lean_addMacroScope(v_quotContext_61_, v___x_72_, v_currMacroScope_62_);
v___x_74_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2bo____1___closed__11));
lean_inc_n(v___x_69_, 2);
v___x_75_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_75_, 0, v___x_69_);
lean_ctor_set(v___x_75_, 1, v___x_71_);
lean_ctor_set(v___x_75_, 2, v___x_73_);
lean_ctor_set(v___x_75_, 3, v___x_74_);
v___x_76_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2bo____1___closed__13));
v___x_77_ = l_Lean_Syntax_node2(v___x_69_, v___x_76_, v___x_65_, v___x_67_);
v___x_78_ = l_Lean_Syntax_node2(v___x_69_, v___x_70_, v___x_75_, v___x_77_);
v___x_79_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_79_, 0, v___x_78_);
lean_ctor_set(v___x_79_, 1, v_a_56_);
return v___x_79_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2bo____1___boxed(lean_object* v_x_80_, lean_object* v_a_81_, lean_object* v_a_82_){
_start:
{
lean_object* v_res_83_; 
v_res_83_ = lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2bo____1(v_x_80_, v_a_81_, v_a_82_);
lean_dec_ref(v_a_81_);
return v_res_83_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______unexpand__OrderAddMonoidHom__1(lean_object* v_x_87_, lean_object* v_a_88_, lean_object* v_a_89_){
_start:
{
lean_object* v___x_90_; uint8_t v___x_91_; 
v___x_90_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2bo____1___closed__4));
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
v___x_96_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______unexpand__OrderAddMonoidHom__1___closed__1));
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
v___x_111_ = ((lean_object*)(lp_mathlib_term___u2192_x2bo___00__closed__1));
v___x_112_ = ((lean_object*)(lp_mathlib_term___u2192_x2bo___00__closed__4));
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
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______unexpand__OrderAddMonoidHom__1___boxed(lean_object* v_x_116_, lean_object* v_a_117_, lean_object* v_a_118_){
_start:
{
lean_object* v_res_119_; 
v_res_119_ = lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______unexpand__OrderAddMonoidHom__1(v_x_116_, v_a_117_, v_a_118_);
lean_dec(v_a_117_);
return v_res_119_;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2243_x2bo____1___closed__1(void){
_start:
{
lean_object* v___x_137_; lean_object* v___x_138_; 
v___x_137_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2243_x2bo____1___closed__0));
v___x_138_ = l_String_toRawSubstring_x27(v___x_137_);
return v___x_138_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2243_x2bo____1(lean_object* v_x_152_, lean_object* v_a_153_, lean_object* v_a_154_){
_start:
{
lean_object* v___x_155_; uint8_t v___x_156_; 
v___x_155_ = ((lean_object*)(lp_mathlib_term___u2243_x2bo___00__closed__1));
lean_inc(v_x_152_);
v___x_156_ = l_Lean_Syntax_isOfKind(v_x_152_, v___x_155_);
if (v___x_156_ == 0)
{
lean_object* v___x_157_; lean_object* v___x_158_; 
lean_dec(v_x_152_);
v___x_157_ = lean_box(1);
v___x_158_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_158_, 0, v___x_157_);
lean_ctor_set(v___x_158_, 1, v_a_154_);
return v___x_158_;
}
else
{
lean_object* v_quotContext_159_; lean_object* v_currMacroScope_160_; lean_object* v_ref_161_; lean_object* v___x_162_; lean_object* v___x_163_; lean_object* v___x_164_; lean_object* v___x_165_; uint8_t v___x_166_; lean_object* v___x_167_; lean_object* v___x_168_; lean_object* v___x_169_; lean_object* v___x_170_; lean_object* v___x_171_; lean_object* v___x_172_; lean_object* v___x_173_; lean_object* v___x_174_; lean_object* v___x_175_; lean_object* v___x_176_; lean_object* v___x_177_; 
v_quotContext_159_ = lean_ctor_get(v_a_153_, 1);
v_currMacroScope_160_ = lean_ctor_get(v_a_153_, 2);
v_ref_161_ = lean_ctor_get(v_a_153_, 5);
v___x_162_ = lean_unsigned_to_nat(0u);
v___x_163_ = l_Lean_Syntax_getArg(v_x_152_, v___x_162_);
v___x_164_ = lean_unsigned_to_nat(2u);
v___x_165_ = l_Lean_Syntax_getArg(v_x_152_, v___x_164_);
lean_dec(v_x_152_);
v___x_166_ = 0;
v___x_167_ = l_Lean_SourceInfo_fromRef(v_ref_161_, v___x_166_);
v___x_168_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2bo____1___closed__4));
v___x_169_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2243_x2bo____1___closed__1, &lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2243_x2bo____1___closed__1_once, _init_lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2243_x2bo____1___closed__1);
v___x_170_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2243_x2bo____1___closed__2));
lean_inc(v_currMacroScope_160_);
lean_inc(v_quotContext_159_);
v___x_171_ = l_Lean_addMacroScope(v_quotContext_159_, v___x_170_, v_currMacroScope_160_);
v___x_172_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2243_x2bo____1___closed__6));
lean_inc_n(v___x_167_, 2);
v___x_173_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_173_, 0, v___x_167_);
lean_ctor_set(v___x_173_, 1, v___x_169_);
lean_ctor_set(v___x_173_, 2, v___x_171_);
lean_ctor_set(v___x_173_, 3, v___x_172_);
v___x_174_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2bo____1___closed__13));
v___x_175_ = l_Lean_Syntax_node2(v___x_167_, v___x_174_, v___x_163_, v___x_165_);
v___x_176_ = l_Lean_Syntax_node2(v___x_167_, v___x_168_, v___x_173_, v___x_175_);
v___x_177_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_177_, 0, v___x_176_);
lean_ctor_set(v___x_177_, 1, v_a_154_);
return v___x_177_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2243_x2bo____1___boxed(lean_object* v_x_178_, lean_object* v_a_179_, lean_object* v_a_180_){
_start:
{
lean_object* v_res_181_; 
v_res_181_ = lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2243_x2bo____1(v_x_178_, v_a_179_, v_a_180_);
lean_dec_ref(v_a_179_);
return v_res_181_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______unexpand__OrderAddMonoidIso__1(lean_object* v_x_182_, lean_object* v_a_183_, lean_object* v_a_184_){
_start:
{
lean_object* v___x_185_; uint8_t v___x_186_; 
v___x_185_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2bo____1___closed__4));
lean_inc(v_x_182_);
v___x_186_ = l_Lean_Syntax_isOfKind(v_x_182_, v___x_185_);
if (v___x_186_ == 0)
{
lean_object* v___x_187_; lean_object* v___x_188_; 
lean_dec(v_x_182_);
v___x_187_ = lean_box(0);
v___x_188_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_188_, 0, v___x_187_);
lean_ctor_set(v___x_188_, 1, v_a_184_);
return v___x_188_;
}
else
{
lean_object* v___x_189_; lean_object* v___x_190_; lean_object* v___x_191_; uint8_t v___x_192_; 
v___x_189_ = lean_unsigned_to_nat(0u);
v___x_190_ = l_Lean_Syntax_getArg(v_x_182_, v___x_189_);
v___x_191_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______unexpand__OrderAddMonoidHom__1___closed__1));
lean_inc(v___x_190_);
v___x_192_ = l_Lean_Syntax_isOfKind(v___x_190_, v___x_191_);
if (v___x_192_ == 0)
{
lean_object* v___x_193_; lean_object* v___x_194_; 
lean_dec(v___x_190_);
lean_dec(v_x_182_);
v___x_193_ = lean_box(0);
v___x_194_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_194_, 0, v___x_193_);
lean_ctor_set(v___x_194_, 1, v_a_184_);
return v___x_194_;
}
else
{
lean_object* v___x_195_; lean_object* v___x_196_; lean_object* v___x_197_; uint8_t v___x_198_; 
v___x_195_ = lean_unsigned_to_nat(1u);
v___x_196_ = l_Lean_Syntax_getArg(v_x_182_, v___x_195_);
lean_dec(v_x_182_);
v___x_197_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_196_);
v___x_198_ = l_Lean_Syntax_matchesNull(v___x_196_, v___x_197_);
if (v___x_198_ == 0)
{
lean_object* v___x_199_; lean_object* v___x_200_; 
lean_dec(v___x_196_);
lean_dec(v___x_190_);
v___x_199_ = lean_box(0);
v___x_200_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_200_, 0, v___x_199_);
lean_ctor_set(v___x_200_, 1, v_a_184_);
return v___x_200_;
}
else
{
lean_object* v___x_201_; lean_object* v___x_202_; lean_object* v_ref_203_; uint8_t v___x_204_; lean_object* v___x_205_; lean_object* v___x_206_; lean_object* v___x_207_; lean_object* v___x_208_; lean_object* v___x_209_; lean_object* v___x_210_; 
v___x_201_ = l_Lean_Syntax_getArg(v___x_196_, v___x_189_);
v___x_202_ = l_Lean_Syntax_getArg(v___x_196_, v___x_195_);
lean_dec(v___x_196_);
v_ref_203_ = l_Lean_replaceRef(v___x_190_, v_a_183_);
lean_dec(v___x_190_);
v___x_204_ = 0;
v___x_205_ = l_Lean_SourceInfo_fromRef(v_ref_203_, v___x_204_);
lean_dec(v_ref_203_);
v___x_206_ = ((lean_object*)(lp_mathlib_term___u2243_x2bo___00__closed__1));
v___x_207_ = ((lean_object*)(lp_mathlib_term___u2243_x2bo___00__closed__2));
lean_inc(v___x_205_);
v___x_208_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_208_, 0, v___x_205_);
lean_ctor_set(v___x_208_, 1, v___x_207_);
v___x_209_ = l_Lean_Syntax_node3(v___x_205_, v___x_206_, v___x_201_, v___x_208_, v___x_202_);
v___x_210_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_210_, 0, v___x_209_);
lean_ctor_set(v___x_210_, 1, v_a_184_);
return v___x_210_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______unexpand__OrderAddMonoidIso__1___boxed(lean_object* v_x_211_, lean_object* v_a_212_, lean_object* v_a_213_){
_start:
{
lean_object* v_res_214_; 
v_res_214_ = lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______unexpand__OrderAddMonoidIso__1(v_x_211_, v_a_212_, v_a_213_);
lean_dec(v_a_212_);
return v_res_214_;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2ao____1___closed__1(void){
_start:
{
lean_object* v___x_232_; lean_object* v___x_233_; 
v___x_232_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2ao____1___closed__0));
v___x_233_ = l_String_toRawSubstring_x27(v___x_232_);
return v___x_233_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2ao____1(lean_object* v_x_247_, lean_object* v_a_248_, lean_object* v_a_249_){
_start:
{
lean_object* v___x_250_; uint8_t v___x_251_; 
v___x_250_ = ((lean_object*)(lp_mathlib_term___u2192_x2ao___00__closed__1));
lean_inc(v_x_247_);
v___x_251_ = l_Lean_Syntax_isOfKind(v_x_247_, v___x_250_);
if (v___x_251_ == 0)
{
lean_object* v___x_252_; lean_object* v___x_253_; 
lean_dec(v_x_247_);
v___x_252_ = lean_box(1);
v___x_253_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_253_, 0, v___x_252_);
lean_ctor_set(v___x_253_, 1, v_a_249_);
return v___x_253_;
}
else
{
lean_object* v_quotContext_254_; lean_object* v_currMacroScope_255_; lean_object* v_ref_256_; lean_object* v___x_257_; lean_object* v___x_258_; lean_object* v___x_259_; lean_object* v___x_260_; uint8_t v___x_261_; lean_object* v___x_262_; lean_object* v___x_263_; lean_object* v___x_264_; lean_object* v___x_265_; lean_object* v___x_266_; lean_object* v___x_267_; lean_object* v___x_268_; lean_object* v___x_269_; lean_object* v___x_270_; lean_object* v___x_271_; lean_object* v___x_272_; 
v_quotContext_254_ = lean_ctor_get(v_a_248_, 1);
v_currMacroScope_255_ = lean_ctor_get(v_a_248_, 2);
v_ref_256_ = lean_ctor_get(v_a_248_, 5);
v___x_257_ = lean_unsigned_to_nat(0u);
v___x_258_ = l_Lean_Syntax_getArg(v_x_247_, v___x_257_);
v___x_259_ = lean_unsigned_to_nat(2u);
v___x_260_ = l_Lean_Syntax_getArg(v_x_247_, v___x_259_);
lean_dec(v_x_247_);
v___x_261_ = 0;
v___x_262_ = l_Lean_SourceInfo_fromRef(v_ref_256_, v___x_261_);
v___x_263_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2bo____1___closed__4));
v___x_264_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2ao____1___closed__1, &lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2ao____1___closed__1_once, _init_lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2ao____1___closed__1);
v___x_265_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2ao____1___closed__2));
lean_inc(v_currMacroScope_255_);
lean_inc(v_quotContext_254_);
v___x_266_ = l_Lean_addMacroScope(v_quotContext_254_, v___x_265_, v_currMacroScope_255_);
v___x_267_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2ao____1___closed__6));
lean_inc_n(v___x_262_, 2);
v___x_268_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_268_, 0, v___x_262_);
lean_ctor_set(v___x_268_, 1, v___x_264_);
lean_ctor_set(v___x_268_, 2, v___x_266_);
lean_ctor_set(v___x_268_, 3, v___x_267_);
v___x_269_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2bo____1___closed__13));
v___x_270_ = l_Lean_Syntax_node2(v___x_262_, v___x_269_, v___x_258_, v___x_260_);
v___x_271_ = l_Lean_Syntax_node2(v___x_262_, v___x_263_, v___x_268_, v___x_270_);
v___x_272_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_272_, 0, v___x_271_);
lean_ctor_set(v___x_272_, 1, v_a_249_);
return v___x_272_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2ao____1___boxed(lean_object* v_x_273_, lean_object* v_a_274_, lean_object* v_a_275_){
_start:
{
lean_object* v_res_276_; 
v_res_276_ = lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2ao____1(v_x_273_, v_a_274_, v_a_275_);
lean_dec_ref(v_a_274_);
return v_res_276_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______unexpand__OrderMonoidHom__1(lean_object* v_x_277_, lean_object* v_a_278_, lean_object* v_a_279_){
_start:
{
lean_object* v___x_280_; uint8_t v___x_281_; 
v___x_280_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2bo____1___closed__4));
lean_inc(v_x_277_);
v___x_281_ = l_Lean_Syntax_isOfKind(v_x_277_, v___x_280_);
if (v___x_281_ == 0)
{
lean_object* v___x_282_; lean_object* v___x_283_; 
lean_dec(v_x_277_);
v___x_282_ = lean_box(0);
v___x_283_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_283_, 0, v___x_282_);
lean_ctor_set(v___x_283_, 1, v_a_279_);
return v___x_283_;
}
else
{
lean_object* v___x_284_; lean_object* v___x_285_; lean_object* v___x_286_; uint8_t v___x_287_; 
v___x_284_ = lean_unsigned_to_nat(0u);
v___x_285_ = l_Lean_Syntax_getArg(v_x_277_, v___x_284_);
v___x_286_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______unexpand__OrderAddMonoidHom__1___closed__1));
lean_inc(v___x_285_);
v___x_287_ = l_Lean_Syntax_isOfKind(v___x_285_, v___x_286_);
if (v___x_287_ == 0)
{
lean_object* v___x_288_; lean_object* v___x_289_; 
lean_dec(v___x_285_);
lean_dec(v_x_277_);
v___x_288_ = lean_box(0);
v___x_289_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_289_, 0, v___x_288_);
lean_ctor_set(v___x_289_, 1, v_a_279_);
return v___x_289_;
}
else
{
lean_object* v___x_290_; lean_object* v___x_291_; lean_object* v___x_292_; uint8_t v___x_293_; 
v___x_290_ = lean_unsigned_to_nat(1u);
v___x_291_ = l_Lean_Syntax_getArg(v_x_277_, v___x_290_);
lean_dec(v_x_277_);
v___x_292_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_291_);
v___x_293_ = l_Lean_Syntax_matchesNull(v___x_291_, v___x_292_);
if (v___x_293_ == 0)
{
lean_object* v___x_294_; lean_object* v___x_295_; 
lean_dec(v___x_291_);
lean_dec(v___x_285_);
v___x_294_ = lean_box(0);
v___x_295_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_295_, 0, v___x_294_);
lean_ctor_set(v___x_295_, 1, v_a_279_);
return v___x_295_;
}
else
{
lean_object* v___x_296_; lean_object* v___x_297_; lean_object* v_ref_298_; uint8_t v___x_299_; lean_object* v___x_300_; lean_object* v___x_301_; lean_object* v___x_302_; lean_object* v___x_303_; lean_object* v___x_304_; lean_object* v___x_305_; 
v___x_296_ = l_Lean_Syntax_getArg(v___x_291_, v___x_284_);
v___x_297_ = l_Lean_Syntax_getArg(v___x_291_, v___x_290_);
lean_dec(v___x_291_);
v_ref_298_ = l_Lean_replaceRef(v___x_285_, v_a_278_);
lean_dec(v___x_285_);
v___x_299_ = 0;
v___x_300_ = l_Lean_SourceInfo_fromRef(v_ref_298_, v___x_299_);
lean_dec(v_ref_298_);
v___x_301_ = ((lean_object*)(lp_mathlib_term___u2192_x2ao___00__closed__1));
v___x_302_ = ((lean_object*)(lp_mathlib_term___u2192_x2ao___00__closed__2));
lean_inc(v___x_300_);
v___x_303_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_303_, 0, v___x_300_);
lean_ctor_set(v___x_303_, 1, v___x_302_);
v___x_304_ = l_Lean_Syntax_node3(v___x_300_, v___x_301_, v___x_296_, v___x_303_, v___x_297_);
v___x_305_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_305_, 0, v___x_304_);
lean_ctor_set(v___x_305_, 1, v_a_279_);
return v___x_305_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______unexpand__OrderMonoidHom__1___boxed(lean_object* v_x_306_, lean_object* v_a_307_, lean_object* v_a_308_){
_start:
{
lean_object* v_res_309_; 
v_res_309_ = lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______unexpand__OrderMonoidHom__1(v_x_306_, v_a_307_, v_a_308_);
lean_dec(v_a_307_);
return v_res_309_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidHomClass_toOrderMonoidHom___redArg(lean_object* v_inst_310_, lean_object* v_f_311_){
_start:
{
lean_object* v___x_312_; 
v___x_312_ = lean_apply_1(v_inst_310_, v_f_311_);
return v___x_312_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidHomClass_toOrderMonoidHom(lean_object* v_F_313_, lean_object* v_00_u03b1_314_, lean_object* v_00_u03b2_315_, lean_object* v_inst_316_, lean_object* v_inst_317_, lean_object* v_inst_318_, lean_object* v_inst_319_, lean_object* v_inst_320_, lean_object* v_inst_321_, lean_object* v_inst_322_, lean_object* v_f_323_){
_start:
{
lean_object* v___x_324_; 
v___x_324_ = lean_apply_1(v_inst_320_, v_f_323_);
return v___x_324_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidHomClass_toOrderMonoidHom___boxed(lean_object* v_F_325_, lean_object* v_00_u03b1_326_, lean_object* v_00_u03b2_327_, lean_object* v_inst_328_, lean_object* v_inst_329_, lean_object* v_inst_330_, lean_object* v_inst_331_, lean_object* v_inst_332_, lean_object* v_inst_333_, lean_object* v_inst_334_, lean_object* v_f_335_){
_start:
{
lean_object* v_res_336_; 
v_res_336_ = lp_mathlib_OrderMonoidHomClass_toOrderMonoidHom(v_F_325_, v_00_u03b1_326_, v_00_u03b2_327_, v_inst_328_, v_inst_329_, v_inst_330_, v_inst_331_, v_inst_332_, v_inst_333_, v_inst_334_, v_f_335_);
lean_dec_ref(v_inst_331_);
lean_dec_ref(v_inst_330_);
lean_dec_ref(v_inst_329_);
lean_dec_ref(v_inst_328_);
return v_res_336_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidHomClass_toOrderAddMonoidHom___redArg(lean_object* v_inst_337_, lean_object* v_f_338_){
_start:
{
lean_object* v___x_339_; 
v___x_339_ = lean_apply_1(v_inst_337_, v_f_338_);
return v___x_339_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidHomClass_toOrderAddMonoidHom(lean_object* v_F_340_, lean_object* v_00_u03b1_341_, lean_object* v_00_u03b2_342_, lean_object* v_inst_343_, lean_object* v_inst_344_, lean_object* v_inst_345_, lean_object* v_inst_346_, lean_object* v_inst_347_, lean_object* v_inst_348_, lean_object* v_inst_349_, lean_object* v_f_350_){
_start:
{
lean_object* v___x_351_; 
v___x_351_ = lean_apply_1(v_inst_347_, v_f_350_);
return v___x_351_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidHomClass_toOrderAddMonoidHom___boxed(lean_object* v_F_352_, lean_object* v_00_u03b1_353_, lean_object* v_00_u03b2_354_, lean_object* v_inst_355_, lean_object* v_inst_356_, lean_object* v_inst_357_, lean_object* v_inst_358_, lean_object* v_inst_359_, lean_object* v_inst_360_, lean_object* v_inst_361_, lean_object* v_f_362_){
_start:
{
lean_object* v_res_363_; 
v_res_363_ = lp_mathlib_OrderMonoidHomClass_toOrderAddMonoidHom(v_F_352_, v_00_u03b1_353_, v_00_u03b2_354_, v_inst_355_, v_inst_356_, v_inst_357_, v_inst_358_, v_inst_359_, v_inst_360_, v_inst_361_, v_f_362_);
lean_dec_ref(v_inst_358_);
lean_dec_ref(v_inst_357_);
lean_dec_ref(v_inst_356_);
lean_dec_ref(v_inst_355_);
return v_res_363_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCOrderMonoidHomOfOrderHomClassOfMonoidHomClass___redArg(lean_object* v_inst_364_, lean_object* v_inst_365_, lean_object* v_inst_366_, lean_object* v_inst_367_, lean_object* v_inst_368_){
_start:
{
lean_object* v___x_369_; 
v___x_369_ = lean_alloc_closure((void*)(lp_mathlib_OrderMonoidHomClass_toOrderMonoidHom___boxed), 11, 10);
lean_closure_set(v___x_369_, 0, lean_box(0));
lean_closure_set(v___x_369_, 1, lean_box(0));
lean_closure_set(v___x_369_, 2, lean_box(0));
lean_closure_set(v___x_369_, 3, v_inst_364_);
lean_closure_set(v___x_369_, 4, v_inst_365_);
lean_closure_set(v___x_369_, 5, v_inst_366_);
lean_closure_set(v___x_369_, 6, v_inst_367_);
lean_closure_set(v___x_369_, 7, v_inst_368_);
lean_closure_set(v___x_369_, 8, lean_box(0));
lean_closure_set(v___x_369_, 9, lean_box(0));
return v___x_369_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCOrderMonoidHomOfOrderHomClassOfMonoidHomClass(lean_object* v_F_370_, lean_object* v_00_u03b1_371_, lean_object* v_00_u03b2_372_, lean_object* v_inst_373_, lean_object* v_inst_374_, lean_object* v_inst_375_, lean_object* v_inst_376_, lean_object* v_inst_377_, lean_object* v_inst_378_, lean_object* v_inst_379_){
_start:
{
lean_object* v___x_380_; 
v___x_380_ = lean_alloc_closure((void*)(lp_mathlib_OrderMonoidHomClass_toOrderMonoidHom___boxed), 11, 10);
lean_closure_set(v___x_380_, 0, lean_box(0));
lean_closure_set(v___x_380_, 1, lean_box(0));
lean_closure_set(v___x_380_, 2, lean_box(0));
lean_closure_set(v___x_380_, 3, v_inst_373_);
lean_closure_set(v___x_380_, 4, v_inst_374_);
lean_closure_set(v___x_380_, 5, v_inst_375_);
lean_closure_set(v___x_380_, 6, v_inst_376_);
lean_closure_set(v___x_380_, 7, v_inst_377_);
lean_closure_set(v___x_380_, 8, lean_box(0));
lean_closure_set(v___x_380_, 9, lean_box(0));
return v___x_380_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCOrderAddMonoidHomOfOrderHomClassOfAddMonoidHomClass___redArg(lean_object* v_inst_381_, lean_object* v_inst_382_, lean_object* v_inst_383_, lean_object* v_inst_384_, lean_object* v_inst_385_){
_start:
{
lean_object* v___x_386_; 
v___x_386_ = lean_alloc_closure((void*)(lp_mathlib_OrderMonoidHomClass_toOrderAddMonoidHom___boxed), 11, 10);
lean_closure_set(v___x_386_, 0, lean_box(0));
lean_closure_set(v___x_386_, 1, lean_box(0));
lean_closure_set(v___x_386_, 2, lean_box(0));
lean_closure_set(v___x_386_, 3, v_inst_381_);
lean_closure_set(v___x_386_, 4, v_inst_382_);
lean_closure_set(v___x_386_, 5, v_inst_383_);
lean_closure_set(v___x_386_, 6, v_inst_384_);
lean_closure_set(v___x_386_, 7, v_inst_385_);
lean_closure_set(v___x_386_, 8, lean_box(0));
lean_closure_set(v___x_386_, 9, lean_box(0));
return v___x_386_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCOrderAddMonoidHomOfOrderHomClassOfAddMonoidHomClass(lean_object* v_F_387_, lean_object* v_00_u03b1_388_, lean_object* v_00_u03b2_389_, lean_object* v_inst_390_, lean_object* v_inst_391_, lean_object* v_inst_392_, lean_object* v_inst_393_, lean_object* v_inst_394_, lean_object* v_inst_395_, lean_object* v_inst_396_){
_start:
{
lean_object* v___x_397_; 
v___x_397_ = lean_alloc_closure((void*)(lp_mathlib_OrderMonoidHomClass_toOrderAddMonoidHom___boxed), 11, 10);
lean_closure_set(v___x_397_, 0, lean_box(0));
lean_closure_set(v___x_397_, 1, lean_box(0));
lean_closure_set(v___x_397_, 2, lean_box(0));
lean_closure_set(v___x_397_, 3, v_inst_390_);
lean_closure_set(v___x_397_, 4, v_inst_391_);
lean_closure_set(v___x_397_, 5, v_inst_392_);
lean_closure_set(v___x_397_, 6, v_inst_393_);
lean_closure_set(v___x_397_, 7, v_inst_394_);
lean_closure_set(v___x_397_, 8, lean_box(0));
lean_closure_set(v___x_397_, 9, lean_box(0));
return v___x_397_;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2243_x2ao____1___closed__1(void){
_start:
{
lean_object* v___x_415_; lean_object* v___x_416_; 
v___x_415_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2243_x2ao____1___closed__0));
v___x_416_ = l_String_toRawSubstring_x27(v___x_415_);
return v___x_416_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2243_x2ao____1(lean_object* v_x_430_, lean_object* v_a_431_, lean_object* v_a_432_){
_start:
{
lean_object* v___x_433_; uint8_t v___x_434_; 
v___x_433_ = ((lean_object*)(lp_mathlib_term___u2243_x2ao___00__closed__1));
lean_inc(v_x_430_);
v___x_434_ = l_Lean_Syntax_isOfKind(v_x_430_, v___x_433_);
if (v___x_434_ == 0)
{
lean_object* v___x_435_; lean_object* v___x_436_; 
lean_dec(v_x_430_);
v___x_435_ = lean_box(1);
v___x_436_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_436_, 0, v___x_435_);
lean_ctor_set(v___x_436_, 1, v_a_432_);
return v___x_436_;
}
else
{
lean_object* v_quotContext_437_; lean_object* v_currMacroScope_438_; lean_object* v_ref_439_; lean_object* v___x_440_; lean_object* v___x_441_; lean_object* v___x_442_; lean_object* v___x_443_; uint8_t v___x_444_; lean_object* v___x_445_; lean_object* v___x_446_; lean_object* v___x_447_; lean_object* v___x_448_; lean_object* v___x_449_; lean_object* v___x_450_; lean_object* v___x_451_; lean_object* v___x_452_; lean_object* v___x_453_; lean_object* v___x_454_; lean_object* v___x_455_; 
v_quotContext_437_ = lean_ctor_get(v_a_431_, 1);
v_currMacroScope_438_ = lean_ctor_get(v_a_431_, 2);
v_ref_439_ = lean_ctor_get(v_a_431_, 5);
v___x_440_ = lean_unsigned_to_nat(0u);
v___x_441_ = l_Lean_Syntax_getArg(v_x_430_, v___x_440_);
v___x_442_ = lean_unsigned_to_nat(2u);
v___x_443_ = l_Lean_Syntax_getArg(v_x_430_, v___x_442_);
lean_dec(v_x_430_);
v___x_444_ = 0;
v___x_445_ = l_Lean_SourceInfo_fromRef(v_ref_439_, v___x_444_);
v___x_446_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2bo____1___closed__4));
v___x_447_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2243_x2ao____1___closed__1, &lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2243_x2ao____1___closed__1_once, _init_lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2243_x2ao____1___closed__1);
v___x_448_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2243_x2ao____1___closed__2));
lean_inc(v_currMacroScope_438_);
lean_inc(v_quotContext_437_);
v___x_449_ = l_Lean_addMacroScope(v_quotContext_437_, v___x_448_, v_currMacroScope_438_);
v___x_450_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2243_x2ao____1___closed__6));
lean_inc_n(v___x_445_, 2);
v___x_451_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_451_, 0, v___x_445_);
lean_ctor_set(v___x_451_, 1, v___x_447_);
lean_ctor_set(v___x_451_, 2, v___x_449_);
lean_ctor_set(v___x_451_, 3, v___x_450_);
v___x_452_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2bo____1___closed__13));
v___x_453_ = l_Lean_Syntax_node2(v___x_445_, v___x_452_, v___x_441_, v___x_443_);
v___x_454_ = l_Lean_Syntax_node2(v___x_445_, v___x_446_, v___x_451_, v___x_453_);
v___x_455_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_455_, 0, v___x_454_);
lean_ctor_set(v___x_455_, 1, v_a_432_);
return v___x_455_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2243_x2ao____1___boxed(lean_object* v_x_456_, lean_object* v_a_457_, lean_object* v_a_458_){
_start:
{
lean_object* v_res_459_; 
v_res_459_ = lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2243_x2ao____1(v_x_456_, v_a_457_, v_a_458_);
lean_dec_ref(v_a_457_);
return v_res_459_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______unexpand__OrderMonoidIso__1(lean_object* v_x_460_, lean_object* v_a_461_, lean_object* v_a_462_){
_start:
{
lean_object* v___x_463_; uint8_t v___x_464_; 
v___x_463_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______macroRules__term___u2192_x2bo____1___closed__4));
lean_inc(v_x_460_);
v___x_464_ = l_Lean_Syntax_isOfKind(v_x_460_, v___x_463_);
if (v___x_464_ == 0)
{
lean_object* v___x_465_; lean_object* v___x_466_; 
lean_dec(v_x_460_);
v___x_465_ = lean_box(0);
v___x_466_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_466_, 0, v___x_465_);
lean_ctor_set(v___x_466_, 1, v_a_462_);
return v___x_466_;
}
else
{
lean_object* v___x_467_; lean_object* v___x_468_; lean_object* v___x_469_; uint8_t v___x_470_; 
v___x_467_ = lean_unsigned_to_nat(0u);
v___x_468_ = l_Lean_Syntax_getArg(v_x_460_, v___x_467_);
v___x_469_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______unexpand__OrderAddMonoidHom__1___closed__1));
lean_inc(v___x_468_);
v___x_470_ = l_Lean_Syntax_isOfKind(v___x_468_, v___x_469_);
if (v___x_470_ == 0)
{
lean_object* v___x_471_; lean_object* v___x_472_; 
lean_dec(v___x_468_);
lean_dec(v_x_460_);
v___x_471_ = lean_box(0);
v___x_472_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_472_, 0, v___x_471_);
lean_ctor_set(v___x_472_, 1, v_a_462_);
return v___x_472_;
}
else
{
lean_object* v___x_473_; lean_object* v___x_474_; lean_object* v___x_475_; uint8_t v___x_476_; 
v___x_473_ = lean_unsigned_to_nat(1u);
v___x_474_ = l_Lean_Syntax_getArg(v_x_460_, v___x_473_);
lean_dec(v_x_460_);
v___x_475_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_474_);
v___x_476_ = l_Lean_Syntax_matchesNull(v___x_474_, v___x_475_);
if (v___x_476_ == 0)
{
lean_object* v___x_477_; lean_object* v___x_478_; 
lean_dec(v___x_474_);
lean_dec(v___x_468_);
v___x_477_ = lean_box(0);
v___x_478_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_478_, 0, v___x_477_);
lean_ctor_set(v___x_478_, 1, v_a_462_);
return v___x_478_;
}
else
{
lean_object* v___x_479_; lean_object* v___x_480_; lean_object* v_ref_481_; uint8_t v___x_482_; lean_object* v___x_483_; lean_object* v___x_484_; lean_object* v___x_485_; lean_object* v___x_486_; lean_object* v___x_487_; lean_object* v___x_488_; 
v___x_479_ = l_Lean_Syntax_getArg(v___x_474_, v___x_467_);
v___x_480_ = l_Lean_Syntax_getArg(v___x_474_, v___x_473_);
lean_dec(v___x_474_);
v_ref_481_ = l_Lean_replaceRef(v___x_468_, v_a_461_);
lean_dec(v___x_468_);
v___x_482_ = 0;
v___x_483_ = l_Lean_SourceInfo_fromRef(v_ref_481_, v___x_482_);
lean_dec(v_ref_481_);
v___x_484_ = ((lean_object*)(lp_mathlib_term___u2243_x2ao___00__closed__1));
v___x_485_ = ((lean_object*)(lp_mathlib_term___u2243_x2ao___00__closed__2));
lean_inc(v___x_483_);
v___x_486_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_486_, 0, v___x_483_);
lean_ctor_set(v___x_486_, 1, v___x_485_);
v___x_487_ = l_Lean_Syntax_node3(v___x_483_, v___x_484_, v___x_479_, v___x_486_, v___x_480_);
v___x_488_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_488_, 0, v___x_487_);
lean_ctor_set(v___x_488_, 1, v_a_462_);
return v___x_488_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______unexpand__OrderMonoidIso__1___boxed(lean_object* v_x_489_, lean_object* v_a_490_, lean_object* v_a_491_){
_start:
{
lean_object* v_res_492_; 
v_res_492_ = lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Monoid______unexpand__OrderMonoidIso__1(v_x_489_, v_a_490_, v_a_491_);
lean_dec(v_a_490_);
return v_res_492_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidIsoClass_toOrderMonoidIso___redArg(lean_object* v_inst_493_, lean_object* v_f_494_){
_start:
{
lean_object* v___x_495_; 
v___x_495_ = lp_mathlib_EquivLike_toEquiv___redArg(v_inst_493_, v_f_494_);
return v___x_495_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidIsoClass_toOrderMonoidIso(lean_object* v_F_496_, lean_object* v_00_u03b1_497_, lean_object* v_00_u03b2_498_, lean_object* v_inst_499_, lean_object* v_inst_500_, lean_object* v_inst_501_, lean_object* v_inst_502_, lean_object* v_inst_503_, lean_object* v_inst_504_, lean_object* v_inst_505_, lean_object* v_f_506_){
_start:
{
lean_object* v___x_507_; 
v___x_507_ = lp_mathlib_EquivLike_toEquiv___redArg(v_inst_503_, v_f_506_);
return v___x_507_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidIsoClass_toOrderMonoidIso___boxed(lean_object* v_F_508_, lean_object* v_00_u03b1_509_, lean_object* v_00_u03b2_510_, lean_object* v_inst_511_, lean_object* v_inst_512_, lean_object* v_inst_513_, lean_object* v_inst_514_, lean_object* v_inst_515_, lean_object* v_inst_516_, lean_object* v_inst_517_, lean_object* v_f_518_){
_start:
{
lean_object* v_res_519_; 
v_res_519_ = lp_mathlib_OrderMonoidIsoClass_toOrderMonoidIso(v_F_508_, v_00_u03b1_509_, v_00_u03b2_510_, v_inst_511_, v_inst_512_, v_inst_513_, v_inst_514_, v_inst_515_, v_inst_516_, v_inst_517_, v_f_518_);
lean_dec_ref(v_inst_514_);
lean_dec_ref(v_inst_513_);
lean_dec_ref(v_inst_512_);
lean_dec_ref(v_inst_511_);
return v_res_519_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidIsoClass_toOrderAddMonoidIso___redArg(lean_object* v_inst_520_, lean_object* v_f_521_){
_start:
{
lean_object* v___x_522_; 
v___x_522_ = lp_mathlib_EquivLike_toEquiv___redArg(v_inst_520_, v_f_521_);
return v___x_522_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidIsoClass_toOrderAddMonoidIso(lean_object* v_F_523_, lean_object* v_00_u03b1_524_, lean_object* v_00_u03b2_525_, lean_object* v_inst_526_, lean_object* v_inst_527_, lean_object* v_inst_528_, lean_object* v_inst_529_, lean_object* v_inst_530_, lean_object* v_inst_531_, lean_object* v_inst_532_, lean_object* v_f_533_){
_start:
{
lean_object* v___x_534_; 
v___x_534_ = lp_mathlib_EquivLike_toEquiv___redArg(v_inst_530_, v_f_533_);
return v___x_534_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidIsoClass_toOrderAddMonoidIso___boxed(lean_object* v_F_535_, lean_object* v_00_u03b1_536_, lean_object* v_00_u03b2_537_, lean_object* v_inst_538_, lean_object* v_inst_539_, lean_object* v_inst_540_, lean_object* v_inst_541_, lean_object* v_inst_542_, lean_object* v_inst_543_, lean_object* v_inst_544_, lean_object* v_f_545_){
_start:
{
lean_object* v_res_546_; 
v_res_546_ = lp_mathlib_OrderMonoidIsoClass_toOrderAddMonoidIso(v_F_535_, v_00_u03b1_536_, v_00_u03b2_537_, v_inst_538_, v_inst_539_, v_inst_540_, v_inst_541_, v_inst_542_, v_inst_543_, v_inst_544_, v_f_545_);
lean_dec_ref(v_inst_541_);
lean_dec_ref(v_inst_540_);
lean_dec_ref(v_inst_539_);
lean_dec_ref(v_inst_538_);
return v_res_546_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCOrderMonoidIsoOfOrderIsoClassOfMulEquivClass___redArg(lean_object* v_inst_547_, lean_object* v_inst_548_, lean_object* v_inst_549_, lean_object* v_inst_550_, lean_object* v_inst_551_){
_start:
{
lean_object* v___x_552_; 
v___x_552_ = lean_alloc_closure((void*)(lp_mathlib_OrderMonoidIsoClass_toOrderMonoidIso___boxed), 11, 10);
lean_closure_set(v___x_552_, 0, lean_box(0));
lean_closure_set(v___x_552_, 1, lean_box(0));
lean_closure_set(v___x_552_, 2, lean_box(0));
lean_closure_set(v___x_552_, 3, v_inst_547_);
lean_closure_set(v___x_552_, 4, v_inst_548_);
lean_closure_set(v___x_552_, 5, v_inst_549_);
lean_closure_set(v___x_552_, 6, v_inst_550_);
lean_closure_set(v___x_552_, 7, v_inst_551_);
lean_closure_set(v___x_552_, 8, lean_box(0));
lean_closure_set(v___x_552_, 9, lean_box(0));
return v___x_552_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCOrderMonoidIsoOfOrderIsoClassOfMulEquivClass(lean_object* v_F_553_, lean_object* v_00_u03b1_554_, lean_object* v_00_u03b2_555_, lean_object* v_inst_556_, lean_object* v_inst_557_, lean_object* v_inst_558_, lean_object* v_inst_559_, lean_object* v_inst_560_, lean_object* v_inst_561_, lean_object* v_inst_562_){
_start:
{
lean_object* v___x_563_; 
v___x_563_ = lean_alloc_closure((void*)(lp_mathlib_OrderMonoidIsoClass_toOrderMonoidIso___boxed), 11, 10);
lean_closure_set(v___x_563_, 0, lean_box(0));
lean_closure_set(v___x_563_, 1, lean_box(0));
lean_closure_set(v___x_563_, 2, lean_box(0));
lean_closure_set(v___x_563_, 3, v_inst_556_);
lean_closure_set(v___x_563_, 4, v_inst_557_);
lean_closure_set(v___x_563_, 5, v_inst_558_);
lean_closure_set(v___x_563_, 6, v_inst_559_);
lean_closure_set(v___x_563_, 7, v_inst_560_);
lean_closure_set(v___x_563_, 8, lean_box(0));
lean_closure_set(v___x_563_, 9, lean_box(0));
return v___x_563_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCOrderAddMonoidIsoOfOrderIsoClassOfAddEquivClass___redArg(lean_object* v_inst_564_, lean_object* v_inst_565_, lean_object* v_inst_566_, lean_object* v_inst_567_, lean_object* v_inst_568_){
_start:
{
lean_object* v___x_569_; 
v___x_569_ = lean_alloc_closure((void*)(lp_mathlib_OrderMonoidIsoClass_toOrderAddMonoidIso___boxed), 11, 10);
lean_closure_set(v___x_569_, 0, lean_box(0));
lean_closure_set(v___x_569_, 1, lean_box(0));
lean_closure_set(v___x_569_, 2, lean_box(0));
lean_closure_set(v___x_569_, 3, v_inst_564_);
lean_closure_set(v___x_569_, 4, v_inst_565_);
lean_closure_set(v___x_569_, 5, v_inst_566_);
lean_closure_set(v___x_569_, 6, v_inst_567_);
lean_closure_set(v___x_569_, 7, v_inst_568_);
lean_closure_set(v___x_569_, 8, lean_box(0));
lean_closure_set(v___x_569_, 9, lean_box(0));
return v___x_569_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCOrderAddMonoidIsoOfOrderIsoClassOfAddEquivClass(lean_object* v_F_570_, lean_object* v_00_u03b1_571_, lean_object* v_00_u03b2_572_, lean_object* v_inst_573_, lean_object* v_inst_574_, lean_object* v_inst_575_, lean_object* v_inst_576_, lean_object* v_inst_577_, lean_object* v_inst_578_, lean_object* v_inst_579_){
_start:
{
lean_object* v___x_580_; 
v___x_580_ = lean_alloc_closure((void*)(lp_mathlib_OrderMonoidIsoClass_toOrderAddMonoidIso___boxed), 11, 10);
lean_closure_set(v___x_580_, 0, lean_box(0));
lean_closure_set(v___x_580_, 1, lean_box(0));
lean_closure_set(v___x_580_, 2, lean_box(0));
lean_closure_set(v___x_580_, 3, v_inst_573_);
lean_closure_set(v___x_580_, 4, v_inst_574_);
lean_closure_set(v___x_580_, 5, v_inst_575_);
lean_closure_set(v___x_580_, 6, v_inst_576_);
lean_closure_set(v___x_580_, 7, v_inst_577_);
lean_closure_set(v___x_580_, 8, lean_box(0));
lean_closure_set(v___x_580_, 9, lean_box(0));
return v___x_580_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidHom_instFunLike___lam__0(lean_object* v_f_581_, lean_object* v___y_582_){
_start:
{
lean_object* v___x_583_; 
v___x_583_ = lean_apply_1(v_f_581_, v___y_582_);
return v___x_583_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidHom_instFunLike(lean_object* v_00_u03b1_585_, lean_object* v_00_u03b2_586_, lean_object* v_inst_587_, lean_object* v_inst_588_, lean_object* v_inst_589_, lean_object* v_inst_590_){
_start:
{
lean_object* v___f_591_; 
v___f_591_ = ((lean_object*)(lp_mathlib_OrderMonoidHom_instFunLike___closed__0));
return v___f_591_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidHom_instFunLike___boxed(lean_object* v_00_u03b1_592_, lean_object* v_00_u03b2_593_, lean_object* v_inst_594_, lean_object* v_inst_595_, lean_object* v_inst_596_, lean_object* v_inst_597_){
_start:
{
lean_object* v_res_598_; 
v_res_598_ = lp_mathlib_OrderMonoidHom_instFunLike(v_00_u03b1_592_, v_00_u03b2_593_, v_inst_594_, v_inst_595_, v_inst_596_, v_inst_597_);
lean_dec_ref(v_inst_597_);
lean_dec_ref(v_inst_596_);
lean_dec_ref(v_inst_595_);
lean_dec_ref(v_inst_594_);
return v_res_598_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidHom_instFunLike(lean_object* v_00_u03b1_599_, lean_object* v_00_u03b2_600_, lean_object* v_inst_601_, lean_object* v_inst_602_, lean_object* v_inst_603_, lean_object* v_inst_604_){
_start:
{
lean_object* v___f_605_; 
v___f_605_ = ((lean_object*)(lp_mathlib_OrderMonoidHom_instFunLike___closed__0));
return v___f_605_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidHom_instFunLike___boxed(lean_object* v_00_u03b1_606_, lean_object* v_00_u03b2_607_, lean_object* v_inst_608_, lean_object* v_inst_609_, lean_object* v_inst_610_, lean_object* v_inst_611_){
_start:
{
lean_object* v_res_612_; 
v_res_612_ = lp_mathlib_OrderAddMonoidHom_instFunLike(v_00_u03b1_606_, v_00_u03b2_607_, v_inst_608_, v_inst_609_, v_inst_610_, v_inst_611_);
lean_dec_ref(v_inst_611_);
lean_dec_ref(v_inst_610_);
lean_dec_ref(v_inst_609_);
lean_dec_ref(v_inst_608_);
return v_res_612_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidHom_toOrderHom___redArg(lean_object* v_f_613_){
_start:
{
lean_inc(v_f_613_);
return v_f_613_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidHom_toOrderHom___redArg___boxed(lean_object* v_f_614_){
_start:
{
lean_object* v_res_615_; 
v_res_615_ = lp_mathlib_OrderMonoidHom_toOrderHom___redArg(v_f_614_);
lean_dec(v_f_614_);
return v_res_615_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidHom_toOrderHom(lean_object* v_00_u03b1_616_, lean_object* v_00_u03b2_617_, lean_object* v_inst_618_, lean_object* v_inst_619_, lean_object* v_inst_620_, lean_object* v_inst_621_, lean_object* v_f_622_){
_start:
{
lean_inc(v_f_622_);
return v_f_622_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidHom_toOrderHom___boxed(lean_object* v_00_u03b1_623_, lean_object* v_00_u03b2_624_, lean_object* v_inst_625_, lean_object* v_inst_626_, lean_object* v_inst_627_, lean_object* v_inst_628_, lean_object* v_f_629_){
_start:
{
lean_object* v_res_630_; 
v_res_630_ = lp_mathlib_OrderMonoidHom_toOrderHom(v_00_u03b1_623_, v_00_u03b2_624_, v_inst_625_, v_inst_626_, v_inst_627_, v_inst_628_, v_f_629_);
lean_dec(v_f_629_);
lean_dec_ref(v_inst_628_);
lean_dec_ref(v_inst_627_);
lean_dec_ref(v_inst_626_);
lean_dec_ref(v_inst_625_);
return v_res_630_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidHom_toOrderHom___redArg(lean_object* v_f_631_){
_start:
{
lean_inc(v_f_631_);
return v_f_631_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidHom_toOrderHom___redArg___boxed(lean_object* v_f_632_){
_start:
{
lean_object* v_res_633_; 
v_res_633_ = lp_mathlib_OrderAddMonoidHom_toOrderHom___redArg(v_f_632_);
lean_dec(v_f_632_);
return v_res_633_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidHom_toOrderHom(lean_object* v_00_u03b1_634_, lean_object* v_00_u03b2_635_, lean_object* v_inst_636_, lean_object* v_inst_637_, lean_object* v_inst_638_, lean_object* v_inst_639_, lean_object* v_f_640_){
_start:
{
lean_inc(v_f_640_);
return v_f_640_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidHom_toOrderHom___boxed(lean_object* v_00_u03b1_641_, lean_object* v_00_u03b2_642_, lean_object* v_inst_643_, lean_object* v_inst_644_, lean_object* v_inst_645_, lean_object* v_inst_646_, lean_object* v_f_647_){
_start:
{
lean_object* v_res_648_; 
v_res_648_ = lp_mathlib_OrderAddMonoidHom_toOrderHom(v_00_u03b1_641_, v_00_u03b2_642_, v_inst_643_, v_inst_644_, v_inst_645_, v_inst_646_, v_f_647_);
lean_dec(v_f_647_);
lean_dec_ref(v_inst_646_);
lean_dec_ref(v_inst_645_);
lean_dec_ref(v_inst_644_);
lean_dec_ref(v_inst_643_);
return v_res_648_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidHom_copy___redArg(lean_object* v_f_x27_649_){
_start:
{
lean_inc(v_f_x27_649_);
return v_f_x27_649_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidHom_copy___redArg___boxed(lean_object* v_f_x27_650_){
_start:
{
lean_object* v_res_651_; 
v_res_651_ = lp_mathlib_OrderMonoidHom_copy___redArg(v_f_x27_650_);
lean_dec(v_f_x27_650_);
return v_res_651_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidHom_copy(lean_object* v_00_u03b1_652_, lean_object* v_00_u03b2_653_, lean_object* v_inst_654_, lean_object* v_inst_655_, lean_object* v_inst_656_, lean_object* v_inst_657_, lean_object* v_f_658_, lean_object* v_f_x27_659_, lean_object* v_h_660_){
_start:
{
lean_inc(v_f_x27_659_);
return v_f_x27_659_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidHom_copy___boxed(lean_object* v_00_u03b1_661_, lean_object* v_00_u03b2_662_, lean_object* v_inst_663_, lean_object* v_inst_664_, lean_object* v_inst_665_, lean_object* v_inst_666_, lean_object* v_f_667_, lean_object* v_f_x27_668_, lean_object* v_h_669_){
_start:
{
lean_object* v_res_670_; 
v_res_670_ = lp_mathlib_OrderMonoidHom_copy(v_00_u03b1_661_, v_00_u03b2_662_, v_inst_663_, v_inst_664_, v_inst_665_, v_inst_666_, v_f_667_, v_f_x27_668_, v_h_669_);
lean_dec(v_f_x27_668_);
lean_dec(v_f_667_);
lean_dec_ref(v_inst_666_);
lean_dec_ref(v_inst_665_);
lean_dec_ref(v_inst_664_);
lean_dec_ref(v_inst_663_);
return v_res_670_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidHom_copy___redArg(lean_object* v_f_x27_671_){
_start:
{
lean_inc(v_f_x27_671_);
return v_f_x27_671_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidHom_copy___redArg___boxed(lean_object* v_f_x27_672_){
_start:
{
lean_object* v_res_673_; 
v_res_673_ = lp_mathlib_OrderAddMonoidHom_copy___redArg(v_f_x27_672_);
lean_dec(v_f_x27_672_);
return v_res_673_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidHom_copy(lean_object* v_00_u03b1_674_, lean_object* v_00_u03b2_675_, lean_object* v_inst_676_, lean_object* v_inst_677_, lean_object* v_inst_678_, lean_object* v_inst_679_, lean_object* v_f_680_, lean_object* v_f_x27_681_, lean_object* v_h_682_){
_start:
{
lean_inc(v_f_x27_681_);
return v_f_x27_681_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidHom_copy___boxed(lean_object* v_00_u03b1_683_, lean_object* v_00_u03b2_684_, lean_object* v_inst_685_, lean_object* v_inst_686_, lean_object* v_inst_687_, lean_object* v_inst_688_, lean_object* v_f_689_, lean_object* v_f_x27_690_, lean_object* v_h_691_){
_start:
{
lean_object* v_res_692_; 
v_res_692_ = lp_mathlib_OrderAddMonoidHom_copy(v_00_u03b1_683_, v_00_u03b2_684_, v_inst_685_, v_inst_686_, v_inst_687_, v_inst_688_, v_f_689_, v_f_x27_690_, v_h_691_);
lean_dec(v_f_x27_690_);
lean_dec(v_f_689_);
lean_dec_ref(v_inst_688_);
lean_dec_ref(v_inst_687_);
lean_dec_ref(v_inst_686_);
lean_dec_ref(v_inst_685_);
return v_res_692_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidHom_id(lean_object* v_00_u03b1_694_, lean_object* v_inst_695_, lean_object* v_inst_696_){
_start:
{
lean_object* v___f_697_; 
v___f_697_ = ((lean_object*)(lp_mathlib_OrderMonoidHom_id___closed__0));
return v___f_697_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidHom_id___boxed(lean_object* v_00_u03b1_698_, lean_object* v_inst_699_, lean_object* v_inst_700_){
_start:
{
lean_object* v_res_701_; 
v_res_701_ = lp_mathlib_OrderMonoidHom_id(v_00_u03b1_698_, v_inst_699_, v_inst_700_);
lean_dec_ref(v_inst_700_);
lean_dec_ref(v_inst_699_);
return v_res_701_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidHom_id(lean_object* v_00_u03b1_702_, lean_object* v_inst_703_, lean_object* v_inst_704_){
_start:
{
lean_object* v___f_705_; 
v___f_705_ = ((lean_object*)(lp_mathlib_OrderMonoidHom_id___closed__0));
return v___f_705_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidHom_id___boxed(lean_object* v_00_u03b1_706_, lean_object* v_inst_707_, lean_object* v_inst_708_){
_start:
{
lean_object* v_res_709_; 
v_res_709_ = lp_mathlib_OrderAddMonoidHom_id(v_00_u03b1_706_, v_inst_707_, v_inst_708_);
lean_dec_ref(v_inst_708_);
lean_dec_ref(v_inst_707_);
return v_res_709_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidHom_instInhabited(lean_object* v_00_u03b1_710_, lean_object* v_inst_711_, lean_object* v_inst_712_){
_start:
{
lean_object* v___f_713_; 
v___f_713_ = ((lean_object*)(lp_mathlib_OrderMonoidHom_id___closed__0));
return v___f_713_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidHom_instInhabited___boxed(lean_object* v_00_u03b1_714_, lean_object* v_inst_715_, lean_object* v_inst_716_){
_start:
{
lean_object* v_res_717_; 
v_res_717_ = lp_mathlib_OrderMonoidHom_instInhabited(v_00_u03b1_714_, v_inst_715_, v_inst_716_);
lean_dec_ref(v_inst_716_);
lean_dec_ref(v_inst_715_);
return v_res_717_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidHom_instInhabited(lean_object* v_00_u03b1_718_, lean_object* v_inst_719_, lean_object* v_inst_720_){
_start:
{
lean_object* v___f_721_; 
v___f_721_ = ((lean_object*)(lp_mathlib_OrderMonoidHom_id___closed__0));
return v___f_721_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidHom_instInhabited___boxed(lean_object* v_00_u03b1_722_, lean_object* v_inst_723_, lean_object* v_inst_724_){
_start:
{
lean_object* v_res_725_; 
v_res_725_ = lp_mathlib_OrderAddMonoidHom_instInhabited(v_00_u03b1_722_, v_inst_723_, v_inst_724_);
lean_dec_ref(v_inst_724_);
lean_dec_ref(v_inst_723_);
return v_res_725_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidHom_comp___redArg___lam__0(lean_object* v_g_726_, lean_object* v___y_727_){
_start:
{
lean_object* v___x_728_; 
v___x_728_ = lean_apply_1(v_g_726_, v___y_727_);
return v___x_728_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidHom_comp___redArg(lean_object* v_f_729_, lean_object* v_g_730_){
_start:
{
lean_object* v___f_731_; lean_object* v___f_732_; 
v___f_731_ = lean_alloc_closure((void*)(lp_mathlib_OrderMonoidHom_comp___redArg___lam__0), 2, 1);
lean_closure_set(v___f_731_, 0, v_g_730_);
v___f_732_ = lean_alloc_closure((void*)(lp_mathlib_OneHom_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_732_, 0, v___f_731_);
lean_closure_set(v___f_732_, 1, v_f_729_);
return v___f_732_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidHom_comp(lean_object* v_00_u03b1_733_, lean_object* v_00_u03b2_734_, lean_object* v_00_u03b3_735_, lean_object* v_inst_736_, lean_object* v_inst_737_, lean_object* v_inst_738_, lean_object* v_inst_739_, lean_object* v_inst_740_, lean_object* v_inst_741_, lean_object* v_f_742_, lean_object* v_g_743_){
_start:
{
lean_object* v___x_744_; 
v___x_744_ = lp_mathlib_OrderMonoidHom_comp___redArg(v_f_742_, v_g_743_);
return v___x_744_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidHom_comp___boxed(lean_object* v_00_u03b1_745_, lean_object* v_00_u03b2_746_, lean_object* v_00_u03b3_747_, lean_object* v_inst_748_, lean_object* v_inst_749_, lean_object* v_inst_750_, lean_object* v_inst_751_, lean_object* v_inst_752_, lean_object* v_inst_753_, lean_object* v_f_754_, lean_object* v_g_755_){
_start:
{
lean_object* v_res_756_; 
v_res_756_ = lp_mathlib_OrderMonoidHom_comp(v_00_u03b1_745_, v_00_u03b2_746_, v_00_u03b3_747_, v_inst_748_, v_inst_749_, v_inst_750_, v_inst_751_, v_inst_752_, v_inst_753_, v_f_754_, v_g_755_);
lean_dec_ref(v_inst_753_);
lean_dec_ref(v_inst_752_);
lean_dec_ref(v_inst_751_);
lean_dec_ref(v_inst_750_);
lean_dec_ref(v_inst_749_);
lean_dec_ref(v_inst_748_);
return v_res_756_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidHom_comp___redArg(lean_object* v_f_757_, lean_object* v_g_758_){
_start:
{
lean_object* v___f_759_; lean_object* v___f_760_; 
v___f_759_ = lean_alloc_closure((void*)(lp_mathlib_OrderMonoidHom_comp___redArg___lam__0), 2, 1);
lean_closure_set(v___f_759_, 0, v_g_758_);
v___f_760_ = lean_alloc_closure((void*)(lp_mathlib_OneHom_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_760_, 0, v___f_759_);
lean_closure_set(v___f_760_, 1, v_f_757_);
return v___f_760_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidHom_comp(lean_object* v_00_u03b1_761_, lean_object* v_00_u03b2_762_, lean_object* v_00_u03b3_763_, lean_object* v_inst_764_, lean_object* v_inst_765_, lean_object* v_inst_766_, lean_object* v_inst_767_, lean_object* v_inst_768_, lean_object* v_inst_769_, lean_object* v_f_770_, lean_object* v_g_771_){
_start:
{
lean_object* v___x_772_; 
v___x_772_ = lp_mathlib_OrderAddMonoidHom_comp___redArg(v_f_770_, v_g_771_);
return v___x_772_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidHom_comp___boxed(lean_object* v_00_u03b1_773_, lean_object* v_00_u03b2_774_, lean_object* v_00_u03b3_775_, lean_object* v_inst_776_, lean_object* v_inst_777_, lean_object* v_inst_778_, lean_object* v_inst_779_, lean_object* v_inst_780_, lean_object* v_inst_781_, lean_object* v_f_782_, lean_object* v_g_783_){
_start:
{
lean_object* v_res_784_; 
v_res_784_ = lp_mathlib_OrderAddMonoidHom_comp(v_00_u03b1_773_, v_00_u03b2_774_, v_00_u03b3_775_, v_inst_776_, v_inst_777_, v_inst_778_, v_inst_779_, v_inst_780_, v_inst_781_, v_f_782_, v_g_783_);
lean_dec_ref(v_inst_781_);
lean_dec_ref(v_inst_780_);
lean_dec_ref(v_inst_779_);
lean_dec_ref(v_inst_778_);
lean_dec_ref(v_inst_777_);
lean_dec_ref(v_inst_776_);
return v_res_784_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidHom_instOne___redArg___lam__0(lean_object* v_toOne_785_, lean_object* v_x_786_){
_start:
{
lean_inc(v_toOne_785_);
return v_toOne_785_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidHom_instOne___redArg___lam__0___boxed(lean_object* v_toOne_787_, lean_object* v_x_788_){
_start:
{
lean_object* v_res_789_; 
v_res_789_ = lp_mathlib_OrderMonoidHom_instOne___redArg___lam__0(v_toOne_787_, v_x_788_);
lean_dec(v_x_788_);
lean_dec(v_toOne_787_);
return v_res_789_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidHom_instOne___redArg(lean_object* v_inst_790_){
_start:
{
lean_object* v___x_791_; lean_object* v_toOne_792_; lean_object* v___f_793_; 
v___x_791_ = lp_mathlib_MulOneClass_toMulOne___redArg(v_inst_790_);
v_toOne_792_ = lean_ctor_get(v___x_791_, 0);
lean_inc(v_toOne_792_);
lean_dec_ref(v___x_791_);
v___f_793_ = lean_alloc_closure((void*)(lp_mathlib_OrderMonoidHom_instOne___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_793_, 0, v_toOne_792_);
return v___f_793_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidHom_instOne(lean_object* v_00_u03b1_794_, lean_object* v_00_u03b2_795_, lean_object* v_inst_796_, lean_object* v_inst_797_, lean_object* v_inst_798_, lean_object* v_inst_799_){
_start:
{
lean_object* v___x_800_; 
v___x_800_ = lp_mathlib_OrderMonoidHom_instOne___redArg(v_inst_799_);
return v___x_800_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidHom_instOne___boxed(lean_object* v_00_u03b1_801_, lean_object* v_00_u03b2_802_, lean_object* v_inst_803_, lean_object* v_inst_804_, lean_object* v_inst_805_, lean_object* v_inst_806_){
_start:
{
lean_object* v_res_807_; 
v_res_807_ = lp_mathlib_OrderMonoidHom_instOne(v_00_u03b1_801_, v_00_u03b2_802_, v_inst_803_, v_inst_804_, v_inst_805_, v_inst_806_);
lean_dec_ref(v_inst_805_);
lean_dec_ref(v_inst_804_);
lean_dec_ref(v_inst_803_);
return v_res_807_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidHom_instZero___redArg___lam__0(lean_object* v_toZero_808_, lean_object* v_x_809_){
_start:
{
lean_inc(v_toZero_808_);
return v_toZero_808_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidHom_instZero___redArg___lam__0___boxed(lean_object* v_toZero_810_, lean_object* v_x_811_){
_start:
{
lean_object* v_res_812_; 
v_res_812_ = lp_mathlib_OrderAddMonoidHom_instZero___redArg___lam__0(v_toZero_810_, v_x_811_);
lean_dec(v_x_811_);
lean_dec(v_toZero_810_);
return v_res_812_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidHom_instZero___redArg(lean_object* v_inst_813_){
_start:
{
lean_object* v___x_814_; lean_object* v_toZero_815_; lean_object* v___f_816_; 
v___x_814_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v_inst_813_);
v_toZero_815_ = lean_ctor_get(v___x_814_, 0);
lean_inc(v_toZero_815_);
lean_dec_ref(v___x_814_);
v___f_816_ = lean_alloc_closure((void*)(lp_mathlib_OrderAddMonoidHom_instZero___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_816_, 0, v_toZero_815_);
return v___f_816_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidHom_instZero(lean_object* v_00_u03b1_817_, lean_object* v_00_u03b2_818_, lean_object* v_inst_819_, lean_object* v_inst_820_, lean_object* v_inst_821_, lean_object* v_inst_822_){
_start:
{
lean_object* v___x_823_; 
v___x_823_ = lp_mathlib_OrderAddMonoidHom_instZero___redArg(v_inst_822_);
return v___x_823_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidHom_instZero___boxed(lean_object* v_00_u03b1_824_, lean_object* v_00_u03b2_825_, lean_object* v_inst_826_, lean_object* v_inst_827_, lean_object* v_inst_828_, lean_object* v_inst_829_){
_start:
{
lean_object* v_res_830_; 
v_res_830_ = lp_mathlib_OrderAddMonoidHom_instZero(v_00_u03b1_824_, v_00_u03b2_825_, v_inst_826_, v_inst_827_, v_inst_828_, v_inst_829_);
lean_dec_ref(v_inst_828_);
lean_dec_ref(v_inst_827_);
lean_dec_ref(v_inst_826_);
return v_res_830_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidHom_instMulOfIsOrderedMonoid___redArg___lam__0(lean_object* v___x_831_, lean_object* v_f_832_, lean_object* v_g_833_, lean_object* v___y_834_){
_start:
{
lean_object* v_toMul_835_; lean_object* v___x_836_; lean_object* v___x_837_; lean_object* v___x_838_; 
v_toMul_835_ = lean_ctor_get(v___x_831_, 1);
lean_inc(v_toMul_835_);
lean_dec_ref(v___x_831_);
lean_inc(v___y_834_);
v___x_836_ = lean_apply_1(v_f_832_, v___y_834_);
v___x_837_ = lean_apply_1(v_g_833_, v___y_834_);
v___x_838_ = lean_apply_2(v_toMul_835_, v___x_836_, v___x_837_);
return v___x_838_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidHom_instMulOfIsOrderedMonoid___redArg(lean_object* v_inst_839_){
_start:
{
lean_object* v___x_840_; lean_object* v___x_841_; lean_object* v___f_842_; 
v___x_840_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_839_);
v___x_841_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_840_);
v___f_842_ = lean_alloc_closure((void*)(lp_mathlib_OrderMonoidHom_instMulOfIsOrderedMonoid___redArg___lam__0), 4, 1);
lean_closure_set(v___f_842_, 0, v___x_841_);
return v___f_842_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidHom_instMulOfIsOrderedMonoid___redArg___boxed(lean_object* v_inst_843_){
_start:
{
lean_object* v_res_844_; 
v_res_844_ = lp_mathlib_OrderMonoidHom_instMulOfIsOrderedMonoid___redArg(v_inst_843_);
lean_dec_ref(v_inst_843_);
return v_res_844_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidHom_instMulOfIsOrderedMonoid(lean_object* v_00_u03b1_845_, lean_object* v_00_u03b2_846_, lean_object* v_inst_847_, lean_object* v_inst_848_, lean_object* v_inst_849_, lean_object* v_inst_850_, lean_object* v_inst_851_){
_start:
{
lean_object* v___x_852_; 
v___x_852_ = lp_mathlib_OrderMonoidHom_instMulOfIsOrderedMonoid___redArg(v_inst_849_);
return v___x_852_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidHom_instMulOfIsOrderedMonoid___boxed(lean_object* v_00_u03b1_853_, lean_object* v_00_u03b2_854_, lean_object* v_inst_855_, lean_object* v_inst_856_, lean_object* v_inst_857_, lean_object* v_inst_858_, lean_object* v_inst_859_){
_start:
{
lean_object* v_res_860_; 
v_res_860_ = lp_mathlib_OrderMonoidHom_instMulOfIsOrderedMonoid(v_00_u03b1_853_, v_00_u03b2_854_, v_inst_855_, v_inst_856_, v_inst_857_, v_inst_858_, v_inst_859_);
lean_dec_ref(v_inst_858_);
lean_dec_ref(v_inst_857_);
lean_dec_ref(v_inst_856_);
lean_dec_ref(v_inst_855_);
return v_res_860_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidHom_instAddOfIsOrderedAddMonoid___redArg___lam__0(lean_object* v___x_861_, lean_object* v_f_862_, lean_object* v_g_863_, lean_object* v___y_864_){
_start:
{
lean_object* v_toAdd_865_; lean_object* v___x_866_; lean_object* v___x_867_; lean_object* v___x_868_; 
v_toAdd_865_ = lean_ctor_get(v___x_861_, 1);
lean_inc(v_toAdd_865_);
lean_dec_ref(v___x_861_);
lean_inc(v___y_864_);
v___x_866_ = lean_apply_1(v_f_862_, v___y_864_);
v___x_867_ = lean_apply_1(v_g_863_, v___y_864_);
v___x_868_ = lean_apply_2(v_toAdd_865_, v___x_866_, v___x_867_);
return v___x_868_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidHom_instAddOfIsOrderedAddMonoid___redArg(lean_object* v_inst_869_){
_start:
{
lean_object* v___x_870_; lean_object* v___x_871_; lean_object* v___f_872_; 
v___x_870_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_869_);
v___x_871_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_870_);
v___f_872_ = lean_alloc_closure((void*)(lp_mathlib_OrderAddMonoidHom_instAddOfIsOrderedAddMonoid___redArg___lam__0), 4, 1);
lean_closure_set(v___f_872_, 0, v___x_871_);
return v___f_872_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidHom_instAddOfIsOrderedAddMonoid___redArg___boxed(lean_object* v_inst_873_){
_start:
{
lean_object* v_res_874_; 
v_res_874_ = lp_mathlib_OrderAddMonoidHom_instAddOfIsOrderedAddMonoid___redArg(v_inst_873_);
lean_dec_ref(v_inst_873_);
return v_res_874_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidHom_instAddOfIsOrderedAddMonoid(lean_object* v_00_u03b1_875_, lean_object* v_00_u03b2_876_, lean_object* v_inst_877_, lean_object* v_inst_878_, lean_object* v_inst_879_, lean_object* v_inst_880_, lean_object* v_inst_881_){
_start:
{
lean_object* v___x_882_; 
v___x_882_ = lp_mathlib_OrderAddMonoidHom_instAddOfIsOrderedAddMonoid___redArg(v_inst_879_);
return v___x_882_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidHom_instAddOfIsOrderedAddMonoid___boxed(lean_object* v_00_u03b1_883_, lean_object* v_00_u03b2_884_, lean_object* v_inst_885_, lean_object* v_inst_886_, lean_object* v_inst_887_, lean_object* v_inst_888_, lean_object* v_inst_889_){
_start:
{
lean_object* v_res_890_; 
v_res_890_ = lp_mathlib_OrderAddMonoidHom_instAddOfIsOrderedAddMonoid(v_00_u03b1_883_, v_00_u03b2_884_, v_inst_885_, v_inst_886_, v_inst_887_, v_inst_888_, v_inst_889_);
lean_dec_ref(v_inst_888_);
lean_dec_ref(v_inst_887_);
lean_dec_ref(v_inst_886_);
lean_dec_ref(v_inst_885_);
return v_res_890_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidHom_mk_x27___redArg(lean_object* v_f_891_){
_start:
{
lean_inc(v_f_891_);
return v_f_891_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidHom_mk_x27___redArg___boxed(lean_object* v_f_892_){
_start:
{
lean_object* v_res_893_; 
v_res_893_ = lp_mathlib_OrderMonoidHom_mk_x27___redArg(v_f_892_);
lean_dec(v_f_892_);
return v_res_893_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidHom_mk_x27(lean_object* v_00_u03b1_894_, lean_object* v_00_u03b2_895_, lean_object* v_x_896_, lean_object* v_x_897_, lean_object* v_x_898_, lean_object* v_x_899_, lean_object* v_f_900_, lean_object* v_hf_901_, lean_object* v_map__mul_902_){
_start:
{
lean_inc(v_f_900_);
return v_f_900_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidHom_mk_x27___boxed(lean_object* v_00_u03b1_903_, lean_object* v_00_u03b2_904_, lean_object* v_x_905_, lean_object* v_x_906_, lean_object* v_x_907_, lean_object* v_x_908_, lean_object* v_f_909_, lean_object* v_hf_910_, lean_object* v_map__mul_911_){
_start:
{
lean_object* v_res_912_; 
v_res_912_ = lp_mathlib_OrderMonoidHom_mk_x27(v_00_u03b1_903_, v_00_u03b2_904_, v_x_905_, v_x_906_, v_x_907_, v_x_908_, v_f_909_, v_hf_910_, v_map__mul_911_);
lean_dec(v_f_909_);
lean_dec_ref(v_x_908_);
lean_dec_ref(v_x_907_);
lean_dec_ref(v_x_906_);
lean_dec_ref(v_x_905_);
return v_res_912_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidHom_mk_x27___redArg(lean_object* v_f_913_){
_start:
{
lean_inc(v_f_913_);
return v_f_913_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidHom_mk_x27___redArg___boxed(lean_object* v_f_914_){
_start:
{
lean_object* v_res_915_; 
v_res_915_ = lp_mathlib_OrderAddMonoidHom_mk_x27___redArg(v_f_914_);
lean_dec(v_f_914_);
return v_res_915_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidHom_mk_x27(lean_object* v_00_u03b1_916_, lean_object* v_00_u03b2_917_, lean_object* v_x_918_, lean_object* v_x_919_, lean_object* v_x_920_, lean_object* v_x_921_, lean_object* v_f_922_, lean_object* v_hf_923_, lean_object* v_map__mul_924_){
_start:
{
lean_inc(v_f_922_);
return v_f_922_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidHom_mk_x27___boxed(lean_object* v_00_u03b1_925_, lean_object* v_00_u03b2_926_, lean_object* v_x_927_, lean_object* v_x_928_, lean_object* v_x_929_, lean_object* v_x_930_, lean_object* v_f_931_, lean_object* v_hf_932_, lean_object* v_map__mul_933_){
_start:
{
lean_object* v_res_934_; 
v_res_934_ = lp_mathlib_OrderAddMonoidHom_mk_x27(v_00_u03b1_925_, v_00_u03b2_926_, v_x_927_, v_x_928_, v_x_929_, v_x_930_, v_f_931_, v_hf_932_, v_map__mul_933_);
lean_dec(v_f_931_);
lean_dec_ref(v_x_930_);
lean_dec_ref(v_x_929_);
lean_dec_ref(v_x_928_);
lean_dec_ref(v_x_927_);
return v_res_934_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidIso_instEquivLike___lam__0(lean_object* v_f_935_, lean_object* v___y_936_){
_start:
{
lean_object* v_toFun_937_; lean_object* v___x_938_; 
v_toFun_937_ = lean_ctor_get(v_f_935_, 0);
lean_inc(v_toFun_937_);
lean_dec_ref(v_f_935_);
v___x_938_ = lean_apply_1(v_toFun_937_, v___y_936_);
return v___x_938_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidIso_instEquivLike___lam__1(lean_object* v_f_939_, lean_object* v___y_940_){
_start:
{
lean_object* v_invFun_941_; lean_object* v___x_942_; 
v_invFun_941_ = lean_ctor_get(v_f_939_, 1);
lean_inc(v_invFun_941_);
lean_dec_ref(v_f_939_);
v___x_942_ = lean_apply_1(v_invFun_941_, v___y_940_);
return v___x_942_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidIso_instEquivLike(lean_object* v_00_u03b1_948_, lean_object* v_00_u03b2_949_, lean_object* v_inst_950_, lean_object* v_inst_951_, lean_object* v_inst_952_, lean_object* v_inst_953_){
_start:
{
lean_object* v___x_954_; 
v___x_954_ = ((lean_object*)(lp_mathlib_OrderMonoidIso_instEquivLike___closed__2));
return v___x_954_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidIso_instEquivLike___boxed(lean_object* v_00_u03b1_955_, lean_object* v_00_u03b2_956_, lean_object* v_inst_957_, lean_object* v_inst_958_, lean_object* v_inst_959_, lean_object* v_inst_960_){
_start:
{
lean_object* v_res_961_; 
v_res_961_ = lp_mathlib_OrderMonoidIso_instEquivLike(v_00_u03b1_955_, v_00_u03b2_956_, v_inst_957_, v_inst_958_, v_inst_959_, v_inst_960_);
lean_dec(v_inst_960_);
lean_dec(v_inst_959_);
lean_dec_ref(v_inst_958_);
lean_dec_ref(v_inst_957_);
return v_res_961_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidIso_instEquivLike(lean_object* v_00_u03b1_962_, lean_object* v_00_u03b2_963_, lean_object* v_inst_964_, lean_object* v_inst_965_, lean_object* v_inst_966_, lean_object* v_inst_967_){
_start:
{
lean_object* v___x_968_; 
v___x_968_ = ((lean_object*)(lp_mathlib_OrderMonoidIso_instEquivLike___closed__2));
return v___x_968_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidIso_instEquivLike___boxed(lean_object* v_00_u03b1_969_, lean_object* v_00_u03b2_970_, lean_object* v_inst_971_, lean_object* v_inst_972_, lean_object* v_inst_973_, lean_object* v_inst_974_){
_start:
{
lean_object* v_res_975_; 
v_res_975_ = lp_mathlib_OrderAddMonoidIso_instEquivLike(v_00_u03b1_969_, v_00_u03b2_970_, v_inst_971_, v_inst_972_, v_inst_973_, v_inst_974_);
lean_dec(v_inst_974_);
lean_dec(v_inst_973_);
lean_dec_ref(v_inst_972_);
lean_dec_ref(v_inst_971_);
return v_res_975_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidIso_toOrderIso___redArg(lean_object* v_f_976_){
_start:
{
lean_inc_ref(v_f_976_);
return v_f_976_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidIso_toOrderIso___redArg___boxed(lean_object* v_f_977_){
_start:
{
lean_object* v_res_978_; 
v_res_978_ = lp_mathlib_OrderMonoidIso_toOrderIso___redArg(v_f_977_);
lean_dec_ref(v_f_977_);
return v_res_978_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidIso_toOrderIso(lean_object* v_00_u03b1_979_, lean_object* v_00_u03b2_980_, lean_object* v_inst_981_, lean_object* v_inst_982_, lean_object* v_inst_983_, lean_object* v_inst_984_, lean_object* v_f_985_){
_start:
{
lean_inc_ref(v_f_985_);
return v_f_985_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidIso_toOrderIso___boxed(lean_object* v_00_u03b1_986_, lean_object* v_00_u03b2_987_, lean_object* v_inst_988_, lean_object* v_inst_989_, lean_object* v_inst_990_, lean_object* v_inst_991_, lean_object* v_f_992_){
_start:
{
lean_object* v_res_993_; 
v_res_993_ = lp_mathlib_OrderMonoidIso_toOrderIso(v_00_u03b1_986_, v_00_u03b2_987_, v_inst_988_, v_inst_989_, v_inst_990_, v_inst_991_, v_f_992_);
lean_dec_ref(v_f_992_);
lean_dec(v_inst_991_);
lean_dec(v_inst_990_);
lean_dec_ref(v_inst_989_);
lean_dec_ref(v_inst_988_);
return v_res_993_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidIso_toOrderIso___redArg(lean_object* v_f_994_){
_start:
{
lean_inc_ref(v_f_994_);
return v_f_994_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidIso_toOrderIso___redArg___boxed(lean_object* v_f_995_){
_start:
{
lean_object* v_res_996_; 
v_res_996_ = lp_mathlib_OrderAddMonoidIso_toOrderIso___redArg(v_f_995_);
lean_dec_ref(v_f_995_);
return v_res_996_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidIso_toOrderIso(lean_object* v_00_u03b1_997_, lean_object* v_00_u03b2_998_, lean_object* v_inst_999_, lean_object* v_inst_1000_, lean_object* v_inst_1001_, lean_object* v_inst_1002_, lean_object* v_f_1003_){
_start:
{
lean_inc_ref(v_f_1003_);
return v_f_1003_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidIso_toOrderIso___boxed(lean_object* v_00_u03b1_1004_, lean_object* v_00_u03b2_1005_, lean_object* v_inst_1006_, lean_object* v_inst_1007_, lean_object* v_inst_1008_, lean_object* v_inst_1009_, lean_object* v_f_1010_){
_start:
{
lean_object* v_res_1011_; 
v_res_1011_ = lp_mathlib_OrderAddMonoidIso_toOrderIso(v_00_u03b1_1004_, v_00_u03b2_1005_, v_inst_1006_, v_inst_1007_, v_inst_1008_, v_inst_1009_, v_f_1010_);
lean_dec_ref(v_f_1010_);
lean_dec(v_inst_1009_);
lean_dec(v_inst_1008_);
lean_dec_ref(v_inst_1007_);
lean_dec_ref(v_inst_1006_);
return v_res_1011_;
}
}
static lean_object* _init_lp_mathlib_OrderMonoidIso_refl___closed__0(void){
_start:
{
lean_object* v___x_1012_; 
v___x_1012_ = lp_mathlib_Equiv_refl(lean_box(0));
return v___x_1012_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidIso_refl(lean_object* v_00_u03b1_1013_, lean_object* v_inst_1014_, lean_object* v_inst_1015_){
_start:
{
lean_object* v___x_1016_; 
v___x_1016_ = lean_obj_once(&lp_mathlib_OrderMonoidIso_refl___closed__0, &lp_mathlib_OrderMonoidIso_refl___closed__0_once, _init_lp_mathlib_OrderMonoidIso_refl___closed__0);
return v___x_1016_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidIso_refl___boxed(lean_object* v_00_u03b1_1017_, lean_object* v_inst_1018_, lean_object* v_inst_1019_){
_start:
{
lean_object* v_res_1020_; 
v_res_1020_ = lp_mathlib_OrderMonoidIso_refl(v_00_u03b1_1017_, v_inst_1018_, v_inst_1019_);
lean_dec(v_inst_1019_);
lean_dec_ref(v_inst_1018_);
return v_res_1020_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidIso_refl(lean_object* v_00_u03b1_1021_, lean_object* v_inst_1022_, lean_object* v_inst_1023_){
_start:
{
lean_object* v___x_1024_; 
v___x_1024_ = lean_obj_once(&lp_mathlib_OrderMonoidIso_refl___closed__0, &lp_mathlib_OrderMonoidIso_refl___closed__0_once, _init_lp_mathlib_OrderMonoidIso_refl___closed__0);
return v___x_1024_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidIso_refl___boxed(lean_object* v_00_u03b1_1025_, lean_object* v_inst_1026_, lean_object* v_inst_1027_){
_start:
{
lean_object* v_res_1028_; 
v_res_1028_ = lp_mathlib_OrderAddMonoidIso_refl(v_00_u03b1_1025_, v_inst_1026_, v_inst_1027_);
lean_dec(v_inst_1027_);
lean_dec_ref(v_inst_1026_);
return v_res_1028_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidIso_instInhabited(lean_object* v_00_u03b1_1029_, lean_object* v_inst_1030_, lean_object* v_inst_1031_){
_start:
{
lean_object* v___x_1032_; 
v___x_1032_ = lean_obj_once(&lp_mathlib_OrderMonoidIso_refl___closed__0, &lp_mathlib_OrderMonoidIso_refl___closed__0_once, _init_lp_mathlib_OrderMonoidIso_refl___closed__0);
return v___x_1032_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidIso_instInhabited___boxed(lean_object* v_00_u03b1_1033_, lean_object* v_inst_1034_, lean_object* v_inst_1035_){
_start:
{
lean_object* v_res_1036_; 
v_res_1036_ = lp_mathlib_OrderMonoidIso_instInhabited(v_00_u03b1_1033_, v_inst_1034_, v_inst_1035_);
lean_dec(v_inst_1035_);
lean_dec_ref(v_inst_1034_);
return v_res_1036_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidIso_instInhabited(lean_object* v_00_u03b1_1037_, lean_object* v_inst_1038_, lean_object* v_inst_1039_){
_start:
{
lean_object* v___x_1040_; 
v___x_1040_ = lean_obj_once(&lp_mathlib_OrderMonoidIso_refl___closed__0, &lp_mathlib_OrderMonoidIso_refl___closed__0_once, _init_lp_mathlib_OrderMonoidIso_refl___closed__0);
return v___x_1040_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidIso_instInhabited___boxed(lean_object* v_00_u03b1_1041_, lean_object* v_inst_1042_, lean_object* v_inst_1043_){
_start:
{
lean_object* v_res_1044_; 
v_res_1044_ = lp_mathlib_OrderAddMonoidIso_instInhabited(v_00_u03b1_1041_, v_inst_1042_, v_inst_1043_);
lean_dec(v_inst_1043_);
lean_dec_ref(v_inst_1042_);
return v_res_1044_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidIso_trans___redArg(lean_object* v_f_1045_, lean_object* v_g_1046_){
_start:
{
lean_object* v___x_1047_; lean_object* v___x_1048_; lean_object* v___x_1049_; lean_object* v___x_1050_; 
v___x_1047_ = ((lean_object*)(lp_mathlib_OrderMonoidIso_instEquivLike___closed__2));
v___x_1048_ = lp_mathlib_EquivLike_toEquiv___redArg(v___x_1047_, v_f_1045_);
v___x_1049_ = lp_mathlib_EquivLike_toEquiv___redArg(v___x_1047_, v_g_1046_);
v___x_1050_ = lp_mathlib_Equiv_trans___redArg(v___x_1048_, v___x_1049_);
return v___x_1050_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidIso_trans(lean_object* v_00_u03b1_1051_, lean_object* v_00_u03b2_1052_, lean_object* v_00_u03b3_1053_, lean_object* v_inst_1054_, lean_object* v_inst_1055_, lean_object* v_inst_1056_, lean_object* v_inst_1057_, lean_object* v_inst_1058_, lean_object* v_inst_1059_, lean_object* v_f_1060_, lean_object* v_g_1061_){
_start:
{
lean_object* v___x_1062_; 
v___x_1062_ = lp_mathlib_OrderMonoidIso_trans___redArg(v_f_1060_, v_g_1061_);
return v___x_1062_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidIso_trans___boxed(lean_object* v_00_u03b1_1063_, lean_object* v_00_u03b2_1064_, lean_object* v_00_u03b3_1065_, lean_object* v_inst_1066_, lean_object* v_inst_1067_, lean_object* v_inst_1068_, lean_object* v_inst_1069_, lean_object* v_inst_1070_, lean_object* v_inst_1071_, lean_object* v_f_1072_, lean_object* v_g_1073_){
_start:
{
lean_object* v_res_1074_; 
v_res_1074_ = lp_mathlib_OrderMonoidIso_trans(v_00_u03b1_1063_, v_00_u03b2_1064_, v_00_u03b3_1065_, v_inst_1066_, v_inst_1067_, v_inst_1068_, v_inst_1069_, v_inst_1070_, v_inst_1071_, v_f_1072_, v_g_1073_);
lean_dec(v_inst_1071_);
lean_dec(v_inst_1070_);
lean_dec(v_inst_1069_);
lean_dec_ref(v_inst_1068_);
lean_dec_ref(v_inst_1067_);
lean_dec_ref(v_inst_1066_);
return v_res_1074_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidIso_trans___redArg(lean_object* v_f_1075_, lean_object* v_g_1076_){
_start:
{
lean_object* v___x_1077_; lean_object* v___x_1078_; lean_object* v___x_1079_; lean_object* v___x_1080_; 
v___x_1077_ = ((lean_object*)(lp_mathlib_OrderMonoidIso_instEquivLike___closed__2));
v___x_1078_ = lp_mathlib_EquivLike_toEquiv___redArg(v___x_1077_, v_f_1075_);
v___x_1079_ = lp_mathlib_EquivLike_toEquiv___redArg(v___x_1077_, v_g_1076_);
v___x_1080_ = lp_mathlib_Equiv_trans___redArg(v___x_1078_, v___x_1079_);
return v___x_1080_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidIso_trans(lean_object* v_00_u03b1_1081_, lean_object* v_00_u03b2_1082_, lean_object* v_00_u03b3_1083_, lean_object* v_inst_1084_, lean_object* v_inst_1085_, lean_object* v_inst_1086_, lean_object* v_inst_1087_, lean_object* v_inst_1088_, lean_object* v_inst_1089_, lean_object* v_f_1090_, lean_object* v_g_1091_){
_start:
{
lean_object* v___x_1092_; 
v___x_1092_ = lp_mathlib_OrderAddMonoidIso_trans___redArg(v_f_1090_, v_g_1091_);
return v___x_1092_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidIso_trans___boxed(lean_object* v_00_u03b1_1093_, lean_object* v_00_u03b2_1094_, lean_object* v_00_u03b3_1095_, lean_object* v_inst_1096_, lean_object* v_inst_1097_, lean_object* v_inst_1098_, lean_object* v_inst_1099_, lean_object* v_inst_1100_, lean_object* v_inst_1101_, lean_object* v_f_1102_, lean_object* v_g_1103_){
_start:
{
lean_object* v_res_1104_; 
v_res_1104_ = lp_mathlib_OrderAddMonoidIso_trans(v_00_u03b1_1093_, v_00_u03b2_1094_, v_00_u03b3_1095_, v_inst_1096_, v_inst_1097_, v_inst_1098_, v_inst_1099_, v_inst_1100_, v_inst_1101_, v_f_1102_, v_g_1103_);
lean_dec(v_inst_1101_);
lean_dec(v_inst_1100_);
lean_dec(v_inst_1099_);
lean_dec_ref(v_inst_1098_);
lean_dec_ref(v_inst_1097_);
lean_dec_ref(v_inst_1096_);
return v_res_1104_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidIso_symm___redArg(lean_object* v_f_1105_){
_start:
{
lean_object* v___x_1106_; 
v___x_1106_ = lp_mathlib_Equiv_symm___redArg(v_f_1105_);
return v___x_1106_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidIso_symm(lean_object* v_00_u03b1_1107_, lean_object* v_00_u03b2_1108_, lean_object* v_inst_1109_, lean_object* v_inst_1110_, lean_object* v_inst_1111_, lean_object* v_inst_1112_, lean_object* v_f_1113_){
_start:
{
lean_object* v___x_1114_; 
v___x_1114_ = lp_mathlib_Equiv_symm___redArg(v_f_1113_);
return v___x_1114_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidIso_symm___boxed(lean_object* v_00_u03b1_1115_, lean_object* v_00_u03b2_1116_, lean_object* v_inst_1117_, lean_object* v_inst_1118_, lean_object* v_inst_1119_, lean_object* v_inst_1120_, lean_object* v_f_1121_){
_start:
{
lean_object* v_res_1122_; 
v_res_1122_ = lp_mathlib_OrderMonoidIso_symm(v_00_u03b1_1115_, v_00_u03b2_1116_, v_inst_1117_, v_inst_1118_, v_inst_1119_, v_inst_1120_, v_f_1121_);
lean_dec(v_inst_1120_);
lean_dec(v_inst_1119_);
lean_dec_ref(v_inst_1118_);
lean_dec_ref(v_inst_1117_);
return v_res_1122_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidIso_symm___redArg(lean_object* v_f_1123_){
_start:
{
lean_object* v___x_1124_; 
v___x_1124_ = lp_mathlib_Equiv_symm___redArg(v_f_1123_);
return v___x_1124_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidIso_symm(lean_object* v_00_u03b1_1125_, lean_object* v_00_u03b2_1126_, lean_object* v_inst_1127_, lean_object* v_inst_1128_, lean_object* v_inst_1129_, lean_object* v_inst_1130_, lean_object* v_f_1131_){
_start:
{
lean_object* v___x_1132_; 
v___x_1132_ = lp_mathlib_Equiv_symm___redArg(v_f_1131_);
return v___x_1132_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidIso_symm___boxed(lean_object* v_00_u03b1_1133_, lean_object* v_00_u03b2_1134_, lean_object* v_inst_1135_, lean_object* v_inst_1136_, lean_object* v_inst_1137_, lean_object* v_inst_1138_, lean_object* v_f_1139_){
_start:
{
lean_object* v_res_1140_; 
v_res_1140_ = lp_mathlib_OrderAddMonoidIso_symm(v_00_u03b1_1133_, v_00_u03b2_1134_, v_inst_1135_, v_inst_1136_, v_inst_1137_, v_inst_1138_, v_f_1139_);
lean_dec(v_inst_1138_);
lean_dec(v_inst_1137_);
lean_dec_ref(v_inst_1136_);
lean_dec_ref(v_inst_1135_);
return v_res_1140_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidIso_Simps_apply___redArg(lean_object* v_h_1141_, lean_object* v_a_1142_){
_start:
{
lean_object* v_toFun_1143_; lean_object* v___x_1144_; 
v_toFun_1143_ = lean_ctor_get(v_h_1141_, 0);
lean_inc(v_toFun_1143_);
lean_dec_ref(v_h_1141_);
v___x_1144_ = lean_apply_1(v_toFun_1143_, v_a_1142_);
return v___x_1144_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidIso_Simps_apply(lean_object* v_00_u03b1_1145_, lean_object* v_00_u03b2_1146_, lean_object* v_inst_1147_, lean_object* v_inst_1148_, lean_object* v_inst_1149_, lean_object* v_inst_1150_, lean_object* v_h_1151_, lean_object* v_a_1152_){
_start:
{
lean_object* v___x_1153_; 
v___x_1153_ = lp_mathlib_OrderMonoidIso_Simps_apply___redArg(v_h_1151_, v_a_1152_);
return v___x_1153_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidIso_Simps_apply___boxed(lean_object* v_00_u03b1_1154_, lean_object* v_00_u03b2_1155_, lean_object* v_inst_1156_, lean_object* v_inst_1157_, lean_object* v_inst_1158_, lean_object* v_inst_1159_, lean_object* v_h_1160_, lean_object* v_a_1161_){
_start:
{
lean_object* v_res_1162_; 
v_res_1162_ = lp_mathlib_OrderMonoidIso_Simps_apply(v_00_u03b1_1154_, v_00_u03b2_1155_, v_inst_1156_, v_inst_1157_, v_inst_1158_, v_inst_1159_, v_h_1160_, v_a_1161_);
lean_dec(v_inst_1159_);
lean_dec(v_inst_1158_);
lean_dec_ref(v_inst_1157_);
lean_dec_ref(v_inst_1156_);
return v_res_1162_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidIso_Simps_apply___redArg(lean_object* v_h_1163_, lean_object* v_a_1164_){
_start:
{
lean_object* v_toFun_1165_; lean_object* v___x_1166_; 
v_toFun_1165_ = lean_ctor_get(v_h_1163_, 0);
lean_inc(v_toFun_1165_);
lean_dec_ref(v_h_1163_);
v___x_1166_ = lean_apply_1(v_toFun_1165_, v_a_1164_);
return v___x_1166_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidIso_Simps_apply(lean_object* v_00_u03b1_1167_, lean_object* v_00_u03b2_1168_, lean_object* v_inst_1169_, lean_object* v_inst_1170_, lean_object* v_inst_1171_, lean_object* v_inst_1172_, lean_object* v_h_1173_, lean_object* v_a_1174_){
_start:
{
lean_object* v___x_1175_; 
v___x_1175_ = lp_mathlib_OrderAddMonoidIso_Simps_apply___redArg(v_h_1173_, v_a_1174_);
return v___x_1175_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidIso_Simps_apply___boxed(lean_object* v_00_u03b1_1176_, lean_object* v_00_u03b2_1177_, lean_object* v_inst_1178_, lean_object* v_inst_1179_, lean_object* v_inst_1180_, lean_object* v_inst_1181_, lean_object* v_h_1182_, lean_object* v_a_1183_){
_start:
{
lean_object* v_res_1184_; 
v_res_1184_ = lp_mathlib_OrderAddMonoidIso_Simps_apply(v_00_u03b1_1176_, v_00_u03b2_1177_, v_inst_1178_, v_inst_1179_, v_inst_1180_, v_inst_1181_, v_h_1182_, v_a_1183_);
lean_dec(v_inst_1181_);
lean_dec(v_inst_1180_);
lean_dec_ref(v_inst_1179_);
lean_dec_ref(v_inst_1178_);
return v_res_1184_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidIso_Simps_symm__apply___redArg(lean_object* v_h_1185_, lean_object* v_a_1186_){
_start:
{
lean_object* v___x_1187_; lean_object* v_toFun_1188_; lean_object* v___x_1189_; 
v___x_1187_ = lp_mathlib_Equiv_symm___redArg(v_h_1185_);
v_toFun_1188_ = lean_ctor_get(v___x_1187_, 0);
lean_inc(v_toFun_1188_);
lean_dec_ref(v___x_1187_);
v___x_1189_ = lean_apply_1(v_toFun_1188_, v_a_1186_);
return v___x_1189_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidIso_Simps_symm__apply(lean_object* v_00_u03b1_1190_, lean_object* v_00_u03b2_1191_, lean_object* v_inst_1192_, lean_object* v_inst_1193_, lean_object* v_inst_1194_, lean_object* v_inst_1195_, lean_object* v_h_1196_, lean_object* v_a_1197_){
_start:
{
lean_object* v___x_1198_; 
v___x_1198_ = lp_mathlib_OrderMonoidIso_Simps_symm__apply___redArg(v_h_1196_, v_a_1197_);
return v___x_1198_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidIso_Simps_symm__apply___boxed(lean_object* v_00_u03b1_1199_, lean_object* v_00_u03b2_1200_, lean_object* v_inst_1201_, lean_object* v_inst_1202_, lean_object* v_inst_1203_, lean_object* v_inst_1204_, lean_object* v_h_1205_, lean_object* v_a_1206_){
_start:
{
lean_object* v_res_1207_; 
v_res_1207_ = lp_mathlib_OrderMonoidIso_Simps_symm__apply(v_00_u03b1_1199_, v_00_u03b2_1200_, v_inst_1201_, v_inst_1202_, v_inst_1203_, v_inst_1204_, v_h_1205_, v_a_1206_);
lean_dec(v_inst_1204_);
lean_dec(v_inst_1203_);
lean_dec_ref(v_inst_1202_);
lean_dec_ref(v_inst_1201_);
return v_res_1207_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidIso_Simps_symm__apply___redArg(lean_object* v_h_1208_, lean_object* v_a_1209_){
_start:
{
lean_object* v___x_1210_; lean_object* v_toFun_1211_; lean_object* v___x_1212_; 
v___x_1210_ = lp_mathlib_Equiv_symm___redArg(v_h_1208_);
v_toFun_1211_ = lean_ctor_get(v___x_1210_, 0);
lean_inc(v_toFun_1211_);
lean_dec_ref(v___x_1210_);
v___x_1212_ = lean_apply_1(v_toFun_1211_, v_a_1209_);
return v___x_1212_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidIso_Simps_symm__apply(lean_object* v_00_u03b1_1213_, lean_object* v_00_u03b2_1214_, lean_object* v_inst_1215_, lean_object* v_inst_1216_, lean_object* v_inst_1217_, lean_object* v_inst_1218_, lean_object* v_h_1219_, lean_object* v_a_1220_){
_start:
{
lean_object* v___x_1221_; 
v___x_1221_ = lp_mathlib_OrderAddMonoidIso_Simps_symm__apply___redArg(v_h_1219_, v_a_1220_);
return v___x_1221_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidIso_Simps_symm__apply___boxed(lean_object* v_00_u03b1_1222_, lean_object* v_00_u03b2_1223_, lean_object* v_inst_1224_, lean_object* v_inst_1225_, lean_object* v_inst_1226_, lean_object* v_inst_1227_, lean_object* v_h_1228_, lean_object* v_a_1229_){
_start:
{
lean_object* v_res_1230_; 
v_res_1230_ = lp_mathlib_OrderAddMonoidIso_Simps_symm__apply(v_00_u03b1_1222_, v_00_u03b2_1223_, v_inst_1224_, v_inst_1225_, v_inst_1226_, v_inst_1227_, v_h_1228_, v_a_1229_);
lean_dec(v_inst_1227_);
lean_dec(v_inst_1226_);
lean_dec_ref(v_inst_1225_);
lean_dec_ref(v_inst_1224_);
return v_res_1230_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidIso_mk_x27___redArg(lean_object* v_f_1231_){
_start:
{
lean_inc_ref(v_f_1231_);
return v_f_1231_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidIso_mk_x27___redArg___boxed(lean_object* v_f_1232_){
_start:
{
lean_object* v_res_1233_; 
v_res_1233_ = lp_mathlib_OrderMonoidIso_mk_x27___redArg(v_f_1232_);
lean_dec_ref(v_f_1232_);
return v_res_1233_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidIso_mk_x27(lean_object* v_00_u03b1_1234_, lean_object* v_00_u03b2_1235_, lean_object* v_x_1236_, lean_object* v_x_1237_, lean_object* v_x_1238_, lean_object* v_x_1239_, lean_object* v_f_1240_, lean_object* v_hf_1241_, lean_object* v_map__mul_1242_){
_start:
{
lean_inc_ref(v_f_1240_);
return v_f_1240_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderMonoidIso_mk_x27___boxed(lean_object* v_00_u03b1_1243_, lean_object* v_00_u03b2_1244_, lean_object* v_x_1245_, lean_object* v_x_1246_, lean_object* v_x_1247_, lean_object* v_x_1248_, lean_object* v_f_1249_, lean_object* v_hf_1250_, lean_object* v_map__mul_1251_){
_start:
{
lean_object* v_res_1252_; 
v_res_1252_ = lp_mathlib_OrderMonoidIso_mk_x27(v_00_u03b1_1243_, v_00_u03b2_1244_, v_x_1245_, v_x_1246_, v_x_1247_, v_x_1248_, v_f_1249_, v_hf_1250_, v_map__mul_1251_);
lean_dec_ref(v_f_1249_);
lean_dec_ref(v_x_1248_);
lean_dec_ref(v_x_1247_);
lean_dec_ref(v_x_1246_);
lean_dec_ref(v_x_1245_);
return v_res_1252_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidIso_mk_x27___redArg(lean_object* v_f_1253_){
_start:
{
lean_inc_ref(v_f_1253_);
return v_f_1253_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidIso_mk_x27___redArg___boxed(lean_object* v_f_1254_){
_start:
{
lean_object* v_res_1255_; 
v_res_1255_ = lp_mathlib_OrderAddMonoidIso_mk_x27___redArg(v_f_1254_);
lean_dec_ref(v_f_1254_);
return v_res_1255_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidIso_mk_x27(lean_object* v_00_u03b1_1256_, lean_object* v_00_u03b2_1257_, lean_object* v_x_1258_, lean_object* v_x_1259_, lean_object* v_x_1260_, lean_object* v_x_1261_, lean_object* v_f_1262_, lean_object* v_hf_1263_, lean_object* v_map__mul_1264_){
_start:
{
lean_inc_ref(v_f_1262_);
return v_f_1262_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderAddMonoidIso_mk_x27___boxed(lean_object* v_00_u03b1_1265_, lean_object* v_00_u03b2_1266_, lean_object* v_x_1267_, lean_object* v_x_1268_, lean_object* v_x_1269_, lean_object* v_x_1270_, lean_object* v_f_1271_, lean_object* v_hf_1272_, lean_object* v_map__mul_1273_){
_start:
{
lean_object* v_res_1274_; 
v_res_1274_ = lp_mathlib_OrderAddMonoidIso_mk_x27(v_00_u03b1_1265_, v_00_u03b2_1266_, v_x_1267_, v_x_1268_, v_x_1269_, v_x_1270_, v_f_1271_, v_hf_1272_, v_map__mul_1273_);
lean_dec_ref(v_f_1271_);
lean_dec_ref(v_x_1270_);
lean_dec_ref(v_x_1269_);
lean_dec_ref(v_x_1268_);
lean_dec_ref(v_x_1267_);
return v_res_1274_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Equiv_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Hom_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Group_Unbundled_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Monoid_OrderDual(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Hom_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Hom_Monoid(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Equiv_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Hom_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Group_Unbundled_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Monoid_OrderDual(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Hom_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Order_Hom_Monoid(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Equiv_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Hom_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Group_Unbundled_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Monoid_OrderDual(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Hom_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Order_Hom_Monoid(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Equiv_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Hom_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Group_Unbundled_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Monoid_OrderDual(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Hom_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Hom_Monoid(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Order_Hom_Monoid(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Order_Hom_Monoid(builtin);
}
#ifdef __cplusplus
}
#endif
