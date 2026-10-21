// Lean compiler output
// Module: Mathlib.Algebra.GroupWithZero.Hom
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Hom.Basic public import Mathlib.Algebra.GroupWithZero.Basic
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
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Monoid_toMulOneClass___redArg(lean_object*);
lean_object* lp_mathlib_MulOneClass_toMulOne___redArg(lean_object*);
lean_object* l_Function_comp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* lp_mathlib_powMonoidHom___redArg(lean_object*, lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_toMonoidHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_toMonoidHom___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_toMonoidHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_toMonoidHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_term___u2192_x2a_u2080___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 9, .m_data = "term_→*₀_"};
static const lean_object* lp_mathlib_term___u2192_x2a_u2080___00__closed__0 = (const lean_object*)&lp_mathlib_term___u2192_x2a_u2080___00__closed__0_value;
static const lean_ctor_object lp_mathlib_term___u2192_x2a_u2080___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2192_x2a_u2080___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(11, 50, 120, 3, 171, 163, 94, 70)}};
static const lean_object* lp_mathlib_term___u2192_x2a_u2080___00__closed__1 = (const lean_object*)&lp_mathlib_term___u2192_x2a_u2080___00__closed__1_value;
static const lean_string_object lp_mathlib_term___u2192_x2a_u2080___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_term___u2192_x2a_u2080___00__closed__2 = (const lean_object*)&lp_mathlib_term___u2192_x2a_u2080___00__closed__2_value;
static const lean_ctor_object lp_mathlib_term___u2192_x2a_u2080___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2192_x2a_u2080___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_term___u2192_x2a_u2080___00__closed__3 = (const lean_object*)&lp_mathlib_term___u2192_x2a_u2080___00__closed__3_value;
static const lean_string_object lp_mathlib_term___u2192_x2a_u2080___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 5, .m_data = " →*₀ "};
static const lean_object* lp_mathlib_term___u2192_x2a_u2080___00__closed__4 = (const lean_object*)&lp_mathlib_term___u2192_x2a_u2080___00__closed__4_value;
static const lean_ctor_object lp_mathlib_term___u2192_x2a_u2080___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192_x2a_u2080___00__closed__4_value)}};
static const lean_object* lp_mathlib_term___u2192_x2a_u2080___00__closed__5 = (const lean_object*)&lp_mathlib_term___u2192_x2a_u2080___00__closed__5_value;
static const lean_string_object lp_mathlib_term___u2192_x2a_u2080___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_term___u2192_x2a_u2080___00__closed__6 = (const lean_object*)&lp_mathlib_term___u2192_x2a_u2080___00__closed__6_value;
static const lean_ctor_object lp_mathlib_term___u2192_x2a_u2080___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2192_x2a_u2080___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_term___u2192_x2a_u2080___00__closed__7 = (const lean_object*)&lp_mathlib_term___u2192_x2a_u2080___00__closed__7_value;
static const lean_ctor_object lp_mathlib_term___u2192_x2a_u2080___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192_x2a_u2080___00__closed__7_value),((lean_object*)(((size_t)(25) << 1) | 1))}};
static const lean_object* lp_mathlib_term___u2192_x2a_u2080___00__closed__8 = (const lean_object*)&lp_mathlib_term___u2192_x2a_u2080___00__closed__8_value;
static const lean_ctor_object lp_mathlib_term___u2192_x2a_u2080___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192_x2a_u2080___00__closed__3_value),((lean_object*)&lp_mathlib_term___u2192_x2a_u2080___00__closed__5_value),((lean_object*)&lp_mathlib_term___u2192_x2a_u2080___00__closed__8_value)}};
static const lean_object* lp_mathlib_term___u2192_x2a_u2080___00__closed__9 = (const lean_object*)&lp_mathlib_term___u2192_x2a_u2080___00__closed__9_value;
static const lean_ctor_object lp_mathlib_term___u2192_x2a_u2080___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192_x2a_u2080___00__closed__1_value),((lean_object*)(((size_t)(25) << 1) | 1)),((lean_object*)(((size_t)(26) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2192_x2a_u2080___00__closed__9_value)}};
static const lean_object* lp_mathlib_term___u2192_x2a_u2080___00__closed__10 = (const lean_object*)&lp_mathlib_term___u2192_x2a_u2080___00__closed__10_value;
LEAN_EXPORT const lean_object* lp_mathlib_term___u2192_x2a_u2080__ = (const lean_object*)&lp_mathlib_term___u2192_x2a_u2080___00__closed__10_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__GroupWithZero__Hom______macroRules__term___u2192_x2a_u2080____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__GroupWithZero__Hom______macroRules__term___u2192_x2a_u2080____1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__GroupWithZero__Hom______macroRules__term___u2192_x2a_u2080____1___closed__0_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__GroupWithZero__Hom______macroRules__term___u2192_x2a_u2080____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__GroupWithZero__Hom______macroRules__term___u2192_x2a_u2080____1___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__GroupWithZero__Hom______macroRules__term___u2192_x2a_u2080____1___closed__1_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__GroupWithZero__Hom______macroRules__term___u2192_x2a_u2080____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__GroupWithZero__Hom______macroRules__term___u2192_x2a_u2080____1___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__GroupWithZero__Hom______macroRules__term___u2192_x2a_u2080____1___closed__2_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__GroupWithZero__Hom______macroRules__term___u2192_x2a_u2080____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__GroupWithZero__Hom______macroRules__term___u2192_x2a_u2080____1___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__GroupWithZero__Hom______macroRules__term___u2192_x2a_u2080____1___closed__3_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__GroupWithZero__Hom______macroRules__term___u2192_x2a_u2080____1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__GroupWithZero__Hom______macroRules__term___u2192_x2a_u2080____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__GroupWithZero__Hom______macroRules__term___u2192_x2a_u2080____1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__GroupWithZero__Hom______macroRules__term___u2192_x2a_u2080____1___closed__4_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__GroupWithZero__Hom______macroRules__term___u2192_x2a_u2080____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__GroupWithZero__Hom______macroRules__term___u2192_x2a_u2080____1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__GroupWithZero__Hom______macroRules__term___u2192_x2a_u2080____1___closed__4_value_aux_1),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__GroupWithZero__Hom______macroRules__term___u2192_x2a_u2080____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__GroupWithZero__Hom______macroRules__term___u2192_x2a_u2080____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__GroupWithZero__Hom______macroRules__term___u2192_x2a_u2080____1___closed__4_value_aux_2),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__GroupWithZero__Hom______macroRules__term___u2192_x2a_u2080____1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__GroupWithZero__Hom______macroRules__term___u2192_x2a_u2080____1___closed__4 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__GroupWithZero__Hom______macroRules__term___u2192_x2a_u2080____1___closed__4_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__GroupWithZero__Hom______macroRules__term___u2192_x2a_u2080____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "MonoidWithZeroHom"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__GroupWithZero__Hom______macroRules__term___u2192_x2a_u2080____1___closed__5 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__GroupWithZero__Hom______macroRules__term___u2192_x2a_u2080____1___closed__5_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Algebra__GroupWithZero__Hom______macroRules__term___u2192_x2a_u2080____1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Algebra__GroupWithZero__Hom______macroRules__term___u2192_x2a_u2080____1___closed__6;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__GroupWithZero__Hom______macroRules__term___u2192_x2a_u2080____1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__GroupWithZero__Hom______macroRules__term___u2192_x2a_u2080____1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(155, 250, 85, 25, 242, 33, 169, 136)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__GroupWithZero__Hom______macroRules__term___u2192_x2a_u2080____1___closed__7 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__GroupWithZero__Hom______macroRules__term___u2192_x2a_u2080____1___closed__7_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__GroupWithZero__Hom______macroRules__term___u2192_x2a_u2080____1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__GroupWithZero__Hom______macroRules__term___u2192_x2a_u2080____1___closed__7_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__GroupWithZero__Hom______macroRules__term___u2192_x2a_u2080____1___closed__8 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__GroupWithZero__Hom______macroRules__term___u2192_x2a_u2080____1___closed__8_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__GroupWithZero__Hom______macroRules__term___u2192_x2a_u2080____1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__GroupWithZero__Hom______macroRules__term___u2192_x2a_u2080____1___closed__7_value)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__GroupWithZero__Hom______macroRules__term___u2192_x2a_u2080____1___closed__9 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__GroupWithZero__Hom______macroRules__term___u2192_x2a_u2080____1___closed__9_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__GroupWithZero__Hom______macroRules__term___u2192_x2a_u2080____1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__GroupWithZero__Hom______macroRules__term___u2192_x2a_u2080____1___closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__GroupWithZero__Hom______macroRules__term___u2192_x2a_u2080____1___closed__10 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__GroupWithZero__Hom______macroRules__term___u2192_x2a_u2080____1___closed__10_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__GroupWithZero__Hom______macroRules__term___u2192_x2a_u2080____1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__GroupWithZero__Hom______macroRules__term___u2192_x2a_u2080____1___closed__8_value),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__GroupWithZero__Hom______macroRules__term___u2192_x2a_u2080____1___closed__10_value)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__GroupWithZero__Hom______macroRules__term___u2192_x2a_u2080____1___closed__11 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__GroupWithZero__Hom______macroRules__term___u2192_x2a_u2080____1___closed__11_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__GroupWithZero__Hom______macroRules__term___u2192_x2a_u2080____1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__GroupWithZero__Hom______macroRules__term___u2192_x2a_u2080____1___closed__12 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__GroupWithZero__Hom______macroRules__term___u2192_x2a_u2080____1___closed__12_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__GroupWithZero__Hom______macroRules__term___u2192_x2a_u2080____1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__GroupWithZero__Hom______macroRules__term___u2192_x2a_u2080____1___closed__12_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__GroupWithZero__Hom______macroRules__term___u2192_x2a_u2080____1___closed__13 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__GroupWithZero__Hom______macroRules__term___u2192_x2a_u2080____1___closed__13_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__GroupWithZero__Hom______macroRules__term___u2192_x2a_u2080____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__GroupWithZero__Hom______macroRules__term___u2192_x2a_u2080____1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__GroupWithZero__Hom______unexpand__MonoidWithZeroHom__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__GroupWithZero__Hom______unexpand__MonoidWithZeroHom__1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__GroupWithZero__Hom______unexpand__MonoidWithZeroHom__1___closed__0_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__GroupWithZero__Hom______unexpand__MonoidWithZeroHom__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__GroupWithZero__Hom______unexpand__MonoidWithZeroHom__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__GroupWithZero__Hom______unexpand__MonoidWithZeroHom__1___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__GroupWithZero__Hom______unexpand__MonoidWithZeroHom__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__GroupWithZero__Hom______unexpand__MonoidWithZeroHom__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__GroupWithZero__Hom______unexpand__MonoidWithZeroHom__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_ofClass___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_ofClass(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_ofClass___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_coeToMonoidHom___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_coeToMonoidHom(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_coeToZeroHom___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_MonoidWithZeroHom_coeToZeroHom___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_MonoidWithZeroHom_coeToZeroHom___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_MonoidWithZeroHom_coeToZeroHom___closed__0 = (const lean_object*)&lp_mathlib_MonoidWithZeroHom_coeToZeroHom___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_coeToZeroHom(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_coeToZeroHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_copy___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_copy___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_copy(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_copy___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_id___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_id___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_MonoidWithZeroHom_id___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_MonoidWithZeroHom_id___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_MonoidWithZeroHom_id___closed__0 = (const lean_object*)&lp_mathlib_MonoidWithZeroHom_id___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_id(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_id___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_comp___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_comp___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_comp___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_comp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_comp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_instInhabited(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_instInhabited___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_instMul___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_instMul___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_instMul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_instMul(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_instMul___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_one___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_one___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_one___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_one(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_one___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_powMonoidWithZeroHom___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_powMonoidWithZeroHom(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_toMonoidHom___redArg(lean_object* v_self_1_){
_start:
{
lean_inc(v_self_1_);
return v_self_1_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_toMonoidHom___redArg___boxed(lean_object* v_self_2_){
_start:
{
lean_object* v_res_3_; 
v_res_3_ = lp_mathlib_MonoidWithZeroHom_toMonoidHom___redArg(v_self_2_);
lean_dec(v_self_2_);
return v_res_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_toMonoidHom(lean_object* v_00_u03b1_4_, lean_object* v_00_u03b2_5_, lean_object* v_inst_6_, lean_object* v_inst_7_, lean_object* v_self_8_){
_start:
{
lean_inc(v_self_8_);
return v_self_8_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_toMonoidHom___boxed(lean_object* v_00_u03b1_9_, lean_object* v_00_u03b2_10_, lean_object* v_inst_11_, lean_object* v_inst_12_, lean_object* v_self_13_){
_start:
{
lean_object* v_res_14_; 
v_res_14_ = lp_mathlib_MonoidWithZeroHom_toMonoidHom(v_00_u03b1_9_, v_00_u03b2_10_, v_inst_11_, v_inst_12_, v_self_13_);
lean_dec(v_self_13_);
lean_dec_ref(v_inst_12_);
lean_dec_ref(v_inst_11_);
return v_res_14_;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Algebra__GroupWithZero__Hom______macroRules__term___u2192_x2a_u2080____1___closed__6(void){
_start:
{
lean_object* v___x_50_; lean_object* v___x_51_; 
v___x_50_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__GroupWithZero__Hom______macroRules__term___u2192_x2a_u2080____1___closed__5));
v___x_51_ = l_String_toRawSubstring_x27(v___x_50_);
return v___x_51_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__GroupWithZero__Hom______macroRules__term___u2192_x2a_u2080____1(lean_object* v_x_68_, lean_object* v_a_69_, lean_object* v_a_70_){
_start:
{
lean_object* v___x_71_; uint8_t v___x_72_; 
v___x_71_ = ((lean_object*)(lp_mathlib_term___u2192_x2a_u2080___00__closed__1));
lean_inc(v_x_68_);
v___x_72_ = l_Lean_Syntax_isOfKind(v_x_68_, v___x_71_);
if (v___x_72_ == 0)
{
lean_object* v___x_73_; lean_object* v___x_74_; 
lean_dec(v_x_68_);
v___x_73_ = lean_box(1);
v___x_74_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_74_, 0, v___x_73_);
lean_ctor_set(v___x_74_, 1, v_a_70_);
return v___x_74_;
}
else
{
lean_object* v_quotContext_75_; lean_object* v_currMacroScope_76_; lean_object* v_ref_77_; lean_object* v___x_78_; lean_object* v___x_79_; lean_object* v___x_80_; lean_object* v___x_81_; uint8_t v___x_82_; lean_object* v___x_83_; lean_object* v___x_84_; lean_object* v___x_85_; lean_object* v___x_86_; lean_object* v___x_87_; lean_object* v___x_88_; lean_object* v___x_89_; lean_object* v___x_90_; lean_object* v___x_91_; lean_object* v___x_92_; lean_object* v___x_93_; 
v_quotContext_75_ = lean_ctor_get(v_a_69_, 1);
v_currMacroScope_76_ = lean_ctor_get(v_a_69_, 2);
v_ref_77_ = lean_ctor_get(v_a_69_, 5);
v___x_78_ = lean_unsigned_to_nat(0u);
v___x_79_ = l_Lean_Syntax_getArg(v_x_68_, v___x_78_);
v___x_80_ = lean_unsigned_to_nat(2u);
v___x_81_ = l_Lean_Syntax_getArg(v_x_68_, v___x_80_);
lean_dec(v_x_68_);
v___x_82_ = 0;
v___x_83_ = l_Lean_SourceInfo_fromRef(v_ref_77_, v___x_82_);
v___x_84_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__GroupWithZero__Hom______macroRules__term___u2192_x2a_u2080____1___closed__4));
v___x_85_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Algebra__GroupWithZero__Hom______macroRules__term___u2192_x2a_u2080____1___closed__6, &lp_mathlib___aux__Mathlib__Algebra__GroupWithZero__Hom______macroRules__term___u2192_x2a_u2080____1___closed__6_once, _init_lp_mathlib___aux__Mathlib__Algebra__GroupWithZero__Hom______macroRules__term___u2192_x2a_u2080____1___closed__6);
v___x_86_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__GroupWithZero__Hom______macroRules__term___u2192_x2a_u2080____1___closed__7));
lean_inc(v_currMacroScope_76_);
lean_inc(v_quotContext_75_);
v___x_87_ = l_Lean_addMacroScope(v_quotContext_75_, v___x_86_, v_currMacroScope_76_);
v___x_88_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__GroupWithZero__Hom______macroRules__term___u2192_x2a_u2080____1___closed__11));
lean_inc_n(v___x_83_, 2);
v___x_89_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_89_, 0, v___x_83_);
lean_ctor_set(v___x_89_, 1, v___x_85_);
lean_ctor_set(v___x_89_, 2, v___x_87_);
lean_ctor_set(v___x_89_, 3, v___x_88_);
v___x_90_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__GroupWithZero__Hom______macroRules__term___u2192_x2a_u2080____1___closed__13));
v___x_91_ = l_Lean_Syntax_node2(v___x_83_, v___x_90_, v___x_79_, v___x_81_);
v___x_92_ = l_Lean_Syntax_node2(v___x_83_, v___x_84_, v___x_89_, v___x_91_);
v___x_93_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_93_, 0, v___x_92_);
lean_ctor_set(v___x_93_, 1, v_a_70_);
return v___x_93_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__GroupWithZero__Hom______macroRules__term___u2192_x2a_u2080____1___boxed(lean_object* v_x_94_, lean_object* v_a_95_, lean_object* v_a_96_){
_start:
{
lean_object* v_res_97_; 
v_res_97_ = lp_mathlib___aux__Mathlib__Algebra__GroupWithZero__Hom______macroRules__term___u2192_x2a_u2080____1(v_x_94_, v_a_95_, v_a_96_);
lean_dec_ref(v_a_95_);
return v_res_97_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__GroupWithZero__Hom______unexpand__MonoidWithZeroHom__1(lean_object* v_x_101_, lean_object* v_a_102_, lean_object* v_a_103_){
_start:
{
lean_object* v___x_104_; uint8_t v___x_105_; 
v___x_104_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__GroupWithZero__Hom______macroRules__term___u2192_x2a_u2080____1___closed__4));
lean_inc(v_x_101_);
v___x_105_ = l_Lean_Syntax_isOfKind(v_x_101_, v___x_104_);
if (v___x_105_ == 0)
{
lean_object* v___x_106_; lean_object* v___x_107_; 
lean_dec(v_x_101_);
v___x_106_ = lean_box(0);
v___x_107_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_107_, 0, v___x_106_);
lean_ctor_set(v___x_107_, 1, v_a_103_);
return v___x_107_;
}
else
{
lean_object* v___x_108_; lean_object* v___x_109_; lean_object* v___x_110_; uint8_t v___x_111_; 
v___x_108_ = lean_unsigned_to_nat(0u);
v___x_109_ = l_Lean_Syntax_getArg(v_x_101_, v___x_108_);
v___x_110_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__GroupWithZero__Hom______unexpand__MonoidWithZeroHom__1___closed__1));
lean_inc(v___x_109_);
v___x_111_ = l_Lean_Syntax_isOfKind(v___x_109_, v___x_110_);
if (v___x_111_ == 0)
{
lean_object* v___x_112_; lean_object* v___x_113_; 
lean_dec(v___x_109_);
lean_dec(v_x_101_);
v___x_112_ = lean_box(0);
v___x_113_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_113_, 0, v___x_112_);
lean_ctor_set(v___x_113_, 1, v_a_103_);
return v___x_113_;
}
else
{
lean_object* v___x_114_; lean_object* v___x_115_; lean_object* v___x_116_; uint8_t v___x_117_; 
v___x_114_ = lean_unsigned_to_nat(1u);
v___x_115_ = l_Lean_Syntax_getArg(v_x_101_, v___x_114_);
lean_dec(v_x_101_);
v___x_116_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_115_);
v___x_117_ = l_Lean_Syntax_matchesNull(v___x_115_, v___x_116_);
if (v___x_117_ == 0)
{
lean_object* v___x_118_; lean_object* v___x_119_; 
lean_dec(v___x_115_);
lean_dec(v___x_109_);
v___x_118_ = lean_box(0);
v___x_119_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_119_, 0, v___x_118_);
lean_ctor_set(v___x_119_, 1, v_a_103_);
return v___x_119_;
}
else
{
lean_object* v___x_120_; lean_object* v___x_121_; lean_object* v_ref_122_; uint8_t v___x_123_; lean_object* v___x_124_; lean_object* v___x_125_; lean_object* v___x_126_; lean_object* v___x_127_; lean_object* v___x_128_; lean_object* v___x_129_; 
v___x_120_ = l_Lean_Syntax_getArg(v___x_115_, v___x_108_);
v___x_121_ = l_Lean_Syntax_getArg(v___x_115_, v___x_114_);
lean_dec(v___x_115_);
v_ref_122_ = l_Lean_replaceRef(v___x_109_, v_a_102_);
lean_dec(v___x_109_);
v___x_123_ = 0;
v___x_124_ = l_Lean_SourceInfo_fromRef(v_ref_122_, v___x_123_);
lean_dec(v_ref_122_);
v___x_125_ = ((lean_object*)(lp_mathlib_term___u2192_x2a_u2080___00__closed__1));
v___x_126_ = ((lean_object*)(lp_mathlib_term___u2192_x2a_u2080___00__closed__4));
lean_inc(v___x_124_);
v___x_127_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_127_, 0, v___x_124_);
lean_ctor_set(v___x_127_, 1, v___x_126_);
v___x_128_ = l_Lean_Syntax_node3(v___x_124_, v___x_125_, v___x_120_, v___x_127_, v___x_121_);
v___x_129_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_129_, 0, v___x_128_);
lean_ctor_set(v___x_129_, 1, v_a_103_);
return v___x_129_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__GroupWithZero__Hom______unexpand__MonoidWithZeroHom__1___boxed(lean_object* v_x_130_, lean_object* v_a_131_, lean_object* v_a_132_){
_start:
{
lean_object* v_res_133_; 
v_res_133_ = lp_mathlib___aux__Mathlib__Algebra__GroupWithZero__Hom______unexpand__MonoidWithZeroHom__1(v_x_130_, v_a_131_, v_a_132_);
lean_dec(v_a_131_);
return v_res_133_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_ofClass___redArg(lean_object* v_inst_134_, lean_object* v_f_135_){
_start:
{
lean_object* v___x_136_; 
v___x_136_ = lean_apply_1(v_inst_134_, v_f_135_);
return v___x_136_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_ofClass(lean_object* v_F_137_, lean_object* v_00_u03b1_138_, lean_object* v_00_u03b2_139_, lean_object* v_inst_140_, lean_object* v_inst_141_, lean_object* v_inst_142_, lean_object* v_inst_143_, lean_object* v_f_144_){
_start:
{
lean_object* v___x_145_; 
v___x_145_ = lean_apply_1(v_inst_142_, v_f_144_);
return v___x_145_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_ofClass___boxed(lean_object* v_F_146_, lean_object* v_00_u03b1_147_, lean_object* v_00_u03b2_148_, lean_object* v_inst_149_, lean_object* v_inst_150_, lean_object* v_inst_151_, lean_object* v_inst_152_, lean_object* v_f_153_){
_start:
{
lean_object* v_res_154_; 
v_res_154_ = lp_mathlib_MonoidWithZeroHom_ofClass(v_F_146_, v_00_u03b1_147_, v_00_u03b2_148_, v_inst_149_, v_inst_150_, v_inst_151_, v_inst_152_, v_f_153_);
lean_dec_ref(v_inst_150_);
lean_dec_ref(v_inst_149_);
return v_res_154_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_coeToMonoidHom___redArg(lean_object* v_inst_155_, lean_object* v_inst_156_){
_start:
{
lean_object* v___x_157_; 
v___x_157_ = lean_alloc_closure((void*)(lp_mathlib_MonoidWithZeroHom_toMonoidHom___boxed), 5, 4);
lean_closure_set(v___x_157_, 0, lean_box(0));
lean_closure_set(v___x_157_, 1, lean_box(0));
lean_closure_set(v___x_157_, 2, v_inst_155_);
lean_closure_set(v___x_157_, 3, v_inst_156_);
return v___x_157_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_coeToMonoidHom(lean_object* v_00_u03b1_158_, lean_object* v_00_u03b2_159_, lean_object* v_inst_160_, lean_object* v_inst_161_){
_start:
{
lean_object* v___x_162_; 
v___x_162_ = lean_alloc_closure((void*)(lp_mathlib_MonoidWithZeroHom_toMonoidHom___boxed), 5, 4);
lean_closure_set(v___x_162_, 0, lean_box(0));
lean_closure_set(v___x_162_, 1, lean_box(0));
lean_closure_set(v___x_162_, 2, v_inst_160_);
lean_closure_set(v___x_162_, 3, v_inst_161_);
return v___x_162_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_coeToZeroHom___lam__0(lean_object* v_self_163_, lean_object* v___y_164_){
_start:
{
lean_object* v___x_165_; 
v___x_165_ = lean_apply_1(v_self_163_, v___y_164_);
return v___x_165_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_coeToZeroHom(lean_object* v_00_u03b1_167_, lean_object* v_00_u03b2_168_, lean_object* v_inst_169_, lean_object* v_inst_170_){
_start:
{
lean_object* v___f_171_; 
v___f_171_ = ((lean_object*)(lp_mathlib_MonoidWithZeroHom_coeToZeroHom___closed__0));
return v___f_171_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_coeToZeroHom___boxed(lean_object* v_00_u03b1_172_, lean_object* v_00_u03b2_173_, lean_object* v_inst_174_, lean_object* v_inst_175_){
_start:
{
lean_object* v_res_176_; 
v_res_176_ = lp_mathlib_MonoidWithZeroHom_coeToZeroHom(v_00_u03b1_172_, v_00_u03b2_173_, v_inst_174_, v_inst_175_);
lean_dec_ref(v_inst_175_);
lean_dec_ref(v_inst_174_);
return v_res_176_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_copy___redArg(lean_object* v_f_x27_177_){
_start:
{
lean_inc(v_f_x27_177_);
return v_f_x27_177_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_copy___redArg___boxed(lean_object* v_f_x27_178_){
_start:
{
lean_object* v_res_179_; 
v_res_179_ = lp_mathlib_MonoidWithZeroHom_copy___redArg(v_f_x27_178_);
lean_dec(v_f_x27_178_);
return v_res_179_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_copy(lean_object* v_00_u03b1_180_, lean_object* v_00_u03b2_181_, lean_object* v_inst_182_, lean_object* v_inst_183_, lean_object* v_f_184_, lean_object* v_f_x27_185_, lean_object* v_h_186_){
_start:
{
lean_inc(v_f_x27_185_);
return v_f_x27_185_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_copy___boxed(lean_object* v_00_u03b1_187_, lean_object* v_00_u03b2_188_, lean_object* v_inst_189_, lean_object* v_inst_190_, lean_object* v_f_191_, lean_object* v_f_x27_192_, lean_object* v_h_193_){
_start:
{
lean_object* v_res_194_; 
v_res_194_ = lp_mathlib_MonoidWithZeroHom_copy(v_00_u03b1_187_, v_00_u03b2_188_, v_inst_189_, v_inst_190_, v_f_191_, v_f_x27_192_, v_h_193_);
lean_dec(v_f_x27_192_);
lean_dec(v_f_191_);
lean_dec_ref(v_inst_190_);
lean_dec_ref(v_inst_189_);
return v_res_194_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_id___lam__0(lean_object* v_x_195_){
_start:
{
lean_inc(v_x_195_);
return v_x_195_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_id___lam__0___boxed(lean_object* v_x_196_){
_start:
{
lean_object* v_res_197_; 
v_res_197_ = lp_mathlib_MonoidWithZeroHom_id___lam__0(v_x_196_);
lean_dec(v_x_196_);
return v_res_197_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_id(lean_object* v_00_u03b1_199_, lean_object* v_inst_200_){
_start:
{
lean_object* v___f_201_; 
v___f_201_ = ((lean_object*)(lp_mathlib_MonoidWithZeroHom_id___closed__0));
return v___f_201_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_id___boxed(lean_object* v_00_u03b1_202_, lean_object* v_inst_203_){
_start:
{
lean_object* v_res_204_; 
v_res_204_ = lp_mathlib_MonoidWithZeroHom_id(v_00_u03b1_202_, v_inst_203_);
lean_dec_ref(v_inst_203_);
return v_res_204_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_comp___redArg___lam__0(lean_object* v_hnp_205_, lean_object* v___y_206_){
_start:
{
lean_object* v___x_207_; 
v___x_207_ = lean_apply_1(v_hnp_205_, v___y_206_);
return v___x_207_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_comp___redArg___lam__1(lean_object* v_hmn_208_, lean_object* v___y_209_){
_start:
{
lean_object* v___x_210_; 
v___x_210_ = lean_apply_1(v_hmn_208_, v___y_209_);
return v___x_210_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_comp___redArg(lean_object* v_hnp_211_, lean_object* v_hmn_212_){
_start:
{
lean_object* v___f_213_; lean_object* v___f_214_; lean_object* v___x_215_; 
v___f_213_ = lean_alloc_closure((void*)(lp_mathlib_MonoidWithZeroHom_comp___redArg___lam__0), 2, 1);
lean_closure_set(v___f_213_, 0, v_hnp_211_);
v___f_214_ = lean_alloc_closure((void*)(lp_mathlib_MonoidWithZeroHom_comp___redArg___lam__1), 2, 1);
lean_closure_set(v___f_214_, 0, v_hmn_212_);
v___x_215_ = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(v___x_215_, 0, lean_box(0));
lean_closure_set(v___x_215_, 1, lean_box(0));
lean_closure_set(v___x_215_, 2, lean_box(0));
lean_closure_set(v___x_215_, 3, v___f_213_);
lean_closure_set(v___x_215_, 4, v___f_214_);
return v___x_215_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_comp(lean_object* v_00_u03b1_216_, lean_object* v_00_u03b2_217_, lean_object* v_00_u03b3_218_, lean_object* v_inst_219_, lean_object* v_inst_220_, lean_object* v_inst_221_, lean_object* v_hnp_222_, lean_object* v_hmn_223_){
_start:
{
lean_object* v___x_224_; 
v___x_224_ = lp_mathlib_MonoidWithZeroHom_comp___redArg(v_hnp_222_, v_hmn_223_);
return v___x_224_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_comp___boxed(lean_object* v_00_u03b1_225_, lean_object* v_00_u03b2_226_, lean_object* v_00_u03b3_227_, lean_object* v_inst_228_, lean_object* v_inst_229_, lean_object* v_inst_230_, lean_object* v_hnp_231_, lean_object* v_hmn_232_){
_start:
{
lean_object* v_res_233_; 
v_res_233_ = lp_mathlib_MonoidWithZeroHom_comp(v_00_u03b1_225_, v_00_u03b2_226_, v_00_u03b3_227_, v_inst_228_, v_inst_229_, v_inst_230_, v_hnp_231_, v_hmn_232_);
lean_dec_ref(v_inst_230_);
lean_dec_ref(v_inst_229_);
lean_dec_ref(v_inst_228_);
return v_res_233_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_instInhabited(lean_object* v_00_u03b1_234_, lean_object* v_inst_235_){
_start:
{
lean_object* v___f_236_; 
v___f_236_ = ((lean_object*)(lp_mathlib_MonoidWithZeroHom_id___closed__0));
return v___f_236_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_instInhabited___boxed(lean_object* v_00_u03b1_237_, lean_object* v_inst_238_){
_start:
{
lean_object* v_res_239_; 
v_res_239_ = lp_mathlib_MonoidWithZeroHom_instInhabited(v_00_u03b1_237_, v_inst_238_);
lean_dec_ref(v_inst_238_);
return v_res_239_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_instMul___redArg___lam__0(lean_object* v_toCommMonoid_240_, lean_object* v_f_241_, lean_object* v_g_242_, lean_object* v___y_243_){
_start:
{
lean_object* v___x_244_; lean_object* v___x_245_; lean_object* v_toMul_246_; lean_object* v___x_247_; lean_object* v___x_248_; lean_object* v___x_249_; 
v___x_244_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_toCommMonoid_240_);
v___x_245_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_244_);
v_toMul_246_ = lean_ctor_get(v___x_245_, 1);
lean_inc(v_toMul_246_);
lean_dec_ref(v___x_245_);
lean_inc(v___y_243_);
v___x_247_ = lean_apply_1(v_f_241_, v___y_243_);
v___x_248_ = lean_apply_1(v_g_242_, v___y_243_);
v___x_249_ = lean_apply_2(v_toMul_246_, v___x_247_, v___x_248_);
return v___x_249_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_instMul___redArg___lam__0___boxed(lean_object* v_toCommMonoid_250_, lean_object* v_f_251_, lean_object* v_g_252_, lean_object* v___y_253_){
_start:
{
lean_object* v_res_254_; 
v_res_254_ = lp_mathlib_MonoidWithZeroHom_instMul___redArg___lam__0(v_toCommMonoid_250_, v_f_251_, v_g_252_, v___y_253_);
lean_dec_ref(v_toCommMonoid_250_);
return v_res_254_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_instMul___redArg(lean_object* v_inst_255_){
_start:
{
lean_object* v_toCommMonoid_256_; lean_object* v___f_257_; 
v_toCommMonoid_256_ = lean_ctor_get(v_inst_255_, 0);
lean_inc_ref(v_toCommMonoid_256_);
lean_dec_ref(v_inst_255_);
v___f_257_ = lean_alloc_closure((void*)(lp_mathlib_MonoidWithZeroHom_instMul___redArg___lam__0___boxed), 4, 1);
lean_closure_set(v___f_257_, 0, v_toCommMonoid_256_);
return v___f_257_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_instMul(lean_object* v_00_u03b1_258_, lean_object* v_inst_259_, lean_object* v_00_u03b2_260_, lean_object* v_inst_261_){
_start:
{
lean_object* v___x_262_; 
v___x_262_ = lp_mathlib_MonoidWithZeroHom_instMul___redArg(v_inst_261_);
return v___x_262_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_instMul___boxed(lean_object* v_00_u03b1_263_, lean_object* v_inst_264_, lean_object* v_00_u03b2_265_, lean_object* v_inst_266_){
_start:
{
lean_object* v_res_267_; 
v_res_267_ = lp_mathlib_MonoidWithZeroHom_instMul(v_00_u03b1_263_, v_inst_264_, v_00_u03b2_265_, v_inst_266_);
lean_dec_ref(v_inst_264_);
return v_res_267_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_one___redArg___lam__0(lean_object* v_inst_268_, lean_object* v_inst_269_, lean_object* v_toZero_270_, lean_object* v_x_271_){
_start:
{
lean_object* v___x_272_; uint8_t v___x_273_; 
v___x_272_ = lean_apply_1(v_inst_268_, v_x_271_);
v___x_273_ = lean_unbox(v___x_272_);
if (v___x_273_ == 0)
{
lean_object* v_toMulOneClass_274_; lean_object* v___x_275_; lean_object* v_toOne_276_; 
v_toMulOneClass_274_ = lean_ctor_get(v_inst_269_, 0);
lean_inc_ref(v_toMulOneClass_274_);
lean_dec_ref(v_inst_269_);
v___x_275_ = lp_mathlib_MulOneClass_toMulOne___redArg(v_toMulOneClass_274_);
v_toOne_276_ = lean_ctor_get(v___x_275_, 0);
lean_inc(v_toOne_276_);
lean_dec_ref(v___x_275_);
return v_toOne_276_;
}
else
{
lean_dec_ref(v_inst_269_);
lean_inc(v_toZero_270_);
return v_toZero_270_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_one___redArg___lam__0___boxed(lean_object* v_inst_277_, lean_object* v_inst_278_, lean_object* v_toZero_279_, lean_object* v_x_280_){
_start:
{
lean_object* v_res_281_; 
v_res_281_ = lp_mathlib_MonoidWithZeroHom_one___redArg___lam__0(v_inst_277_, v_inst_278_, v_toZero_279_, v_x_280_);
lean_dec(v_toZero_279_);
return v_res_281_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_one___redArg(lean_object* v_inst_282_, lean_object* v_inst_283_){
_start:
{
lean_object* v___x_284_; lean_object* v_toZero_285_; lean_object* v___f_286_; 
lean_inc_ref(v_inst_282_);
v___x_284_ = lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(v_inst_282_);
v_toZero_285_ = lean_ctor_get(v___x_284_, 1);
lean_inc(v_toZero_285_);
lean_dec_ref(v___x_284_);
v___f_286_ = lean_alloc_closure((void*)(lp_mathlib_MonoidWithZeroHom_one___redArg___lam__0___boxed), 4, 3);
lean_closure_set(v___f_286_, 0, v_inst_283_);
lean_closure_set(v___f_286_, 1, v_inst_282_);
lean_closure_set(v___f_286_, 2, v_toZero_285_);
return v___f_286_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_one(lean_object* v_M_u2080_287_, lean_object* v_N_u2080_288_, lean_object* v_inst_289_, lean_object* v_inst_290_, lean_object* v_inst_291_, lean_object* v_inst_292_, lean_object* v_inst_293_){
_start:
{
lean_object* v___x_294_; 
v___x_294_ = lp_mathlib_MonoidWithZeroHom_one___redArg(v_inst_290_, v_inst_291_);
return v___x_294_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_one___boxed(lean_object* v_M_u2080_295_, lean_object* v_N_u2080_296_, lean_object* v_inst_297_, lean_object* v_inst_298_, lean_object* v_inst_299_, lean_object* v_inst_300_, lean_object* v_inst_301_){
_start:
{
lean_object* v_res_302_; 
v_res_302_ = lp_mathlib_MonoidWithZeroHom_one(v_M_u2080_295_, v_N_u2080_296_, v_inst_297_, v_inst_298_, v_inst_299_, v_inst_300_, v_inst_301_);
lean_dec_ref(v_inst_297_);
return v_res_302_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_powMonoidWithZeroHom___redArg(lean_object* v_inst_303_, lean_object* v_n_304_){
_start:
{
lean_object* v_toCommMonoid_305_; lean_object* v___x_306_; 
v_toCommMonoid_305_ = lean_ctor_get(v_inst_303_, 0);
lean_inc_ref(v_toCommMonoid_305_);
lean_dec_ref(v_inst_303_);
v___x_306_ = lp_mathlib_powMonoidHom___redArg(v_toCommMonoid_305_, v_n_304_);
return v___x_306_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_powMonoidWithZeroHom(lean_object* v_M_u2080_307_, lean_object* v_inst_308_, lean_object* v_n_309_, lean_object* v_hn_310_){
_start:
{
lean_object* v___x_311_; 
v___x_311_ = lp_mathlib_powMonoidWithZeroHom___redArg(v_inst_308_, v_n_309_);
return v___x_311_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Hom_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Hom(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Hom_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Hom(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Hom_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Hom(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Hom_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Hom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Hom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_GroupWithZero_Hom(builtin);
}
#ifdef __cplusplus
}
#endif
