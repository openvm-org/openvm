// Lean compiler output
// Module: Mathlib.Algebra.Ring.Hom.Defs
// Imports: public import Init public meta import Init public import Mathlib.Algebra.GroupWithZero.Hom public import Mathlib.Algebra.Ring.Defs public import Mathlib.Algebra.Ring.Basic
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
lean_object* lp_mathlib_OneHom_comp___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_npowBinRecAuto___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_NonUnitalNonAssocSemiring_toMulZeroClass___redArg(lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Nat_iterate___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_toAddMonoidHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_toAddMonoidHom___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_toAddMonoidHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_toAddMonoidHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_term___u2192_u2099_x2b_x2a___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 10, .m_data = "term_→ₙ+*_"};
static const lean_object* lp_mathlib_term___u2192_u2099_x2b_x2a___00__closed__0 = (const lean_object*)&lp_mathlib_term___u2192_u2099_x2b_x2a___00__closed__0_value;
static const lean_ctor_object lp_mathlib_term___u2192_u2099_x2b_x2a___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2192_u2099_x2b_x2a___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(61, 254, 120, 106, 172, 26, 120, 183)}};
static const lean_object* lp_mathlib_term___u2192_u2099_x2b_x2a___00__closed__1 = (const lean_object*)&lp_mathlib_term___u2192_u2099_x2b_x2a___00__closed__1_value;
static const lean_string_object lp_mathlib_term___u2192_u2099_x2b_x2a___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_term___u2192_u2099_x2b_x2a___00__closed__2 = (const lean_object*)&lp_mathlib_term___u2192_u2099_x2b_x2a___00__closed__2_value;
static const lean_ctor_object lp_mathlib_term___u2192_u2099_x2b_x2a___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2192_u2099_x2b_x2a___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_term___u2192_u2099_x2b_x2a___00__closed__3 = (const lean_object*)&lp_mathlib_term___u2192_u2099_x2b_x2a___00__closed__3_value;
static const lean_string_object lp_mathlib_term___u2192_u2099_x2b_x2a___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 6, .m_data = " →ₙ+* "};
static const lean_object* lp_mathlib_term___u2192_u2099_x2b_x2a___00__closed__4 = (const lean_object*)&lp_mathlib_term___u2192_u2099_x2b_x2a___00__closed__4_value;
static const lean_ctor_object lp_mathlib_term___u2192_u2099_x2b_x2a___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192_u2099_x2b_x2a___00__closed__4_value)}};
static const lean_object* lp_mathlib_term___u2192_u2099_x2b_x2a___00__closed__5 = (const lean_object*)&lp_mathlib_term___u2192_u2099_x2b_x2a___00__closed__5_value;
static const lean_string_object lp_mathlib_term___u2192_u2099_x2b_x2a___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_term___u2192_u2099_x2b_x2a___00__closed__6 = (const lean_object*)&lp_mathlib_term___u2192_u2099_x2b_x2a___00__closed__6_value;
static const lean_ctor_object lp_mathlib_term___u2192_u2099_x2b_x2a___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2192_u2099_x2b_x2a___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_term___u2192_u2099_x2b_x2a___00__closed__7 = (const lean_object*)&lp_mathlib_term___u2192_u2099_x2b_x2a___00__closed__7_value;
static const lean_ctor_object lp_mathlib_term___u2192_u2099_x2b_x2a___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192_u2099_x2b_x2a___00__closed__7_value),((lean_object*)(((size_t)(25) << 1) | 1))}};
static const lean_object* lp_mathlib_term___u2192_u2099_x2b_x2a___00__closed__8 = (const lean_object*)&lp_mathlib_term___u2192_u2099_x2b_x2a___00__closed__8_value;
static const lean_ctor_object lp_mathlib_term___u2192_u2099_x2b_x2a___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192_u2099_x2b_x2a___00__closed__3_value),((lean_object*)&lp_mathlib_term___u2192_u2099_x2b_x2a___00__closed__5_value),((lean_object*)&lp_mathlib_term___u2192_u2099_x2b_x2a___00__closed__8_value)}};
static const lean_object* lp_mathlib_term___u2192_u2099_x2b_x2a___00__closed__9 = (const lean_object*)&lp_mathlib_term___u2192_u2099_x2b_x2a___00__closed__9_value;
static const lean_ctor_object lp_mathlib_term___u2192_u2099_x2b_x2a___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192_u2099_x2b_x2a___00__closed__1_value),((lean_object*)(((size_t)(25) << 1) | 1)),((lean_object*)(((size_t)(26) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2192_u2099_x2b_x2a___00__closed__9_value)}};
static const lean_object* lp_mathlib_term___u2192_u2099_x2b_x2a___00__closed__10 = (const lean_object*)&lp_mathlib_term___u2192_u2099_x2b_x2a___00__closed__10_value;
LEAN_EXPORT const lean_object* lp_mathlib_term___u2192_u2099_x2b_x2a__ = (const lean_object*)&lp_mathlib_term___u2192_u2099_x2b_x2a___00__closed__10_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_u2099_x2b_x2a____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_u2099_x2b_x2a____1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_u2099_x2b_x2a____1___closed__0_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_u2099_x2b_x2a____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_u2099_x2b_x2a____1___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_u2099_x2b_x2a____1___closed__1_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_u2099_x2b_x2a____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_u2099_x2b_x2a____1___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_u2099_x2b_x2a____1___closed__2_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_u2099_x2b_x2a____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_u2099_x2b_x2a____1___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_u2099_x2b_x2a____1___closed__3_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_u2099_x2b_x2a____1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_u2099_x2b_x2a____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_u2099_x2b_x2a____1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_u2099_x2b_x2a____1___closed__4_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_u2099_x2b_x2a____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_u2099_x2b_x2a____1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_u2099_x2b_x2a____1___closed__4_value_aux_1),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_u2099_x2b_x2a____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_u2099_x2b_x2a____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_u2099_x2b_x2a____1___closed__4_value_aux_2),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_u2099_x2b_x2a____1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_u2099_x2b_x2a____1___closed__4 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_u2099_x2b_x2a____1___closed__4_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_u2099_x2b_x2a____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "NonUnitalRingHom"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_u2099_x2b_x2a____1___closed__5 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_u2099_x2b_x2a____1___closed__5_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_u2099_x2b_x2a____1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_u2099_x2b_x2a____1___closed__6;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_u2099_x2b_x2a____1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_u2099_x2b_x2a____1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(3, 93, 156, 189, 234, 33, 252, 205)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_u2099_x2b_x2a____1___closed__7 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_u2099_x2b_x2a____1___closed__7_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_u2099_x2b_x2a____1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_u2099_x2b_x2a____1___closed__7_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_u2099_x2b_x2a____1___closed__8 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_u2099_x2b_x2a____1___closed__8_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_u2099_x2b_x2a____1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_u2099_x2b_x2a____1___closed__7_value)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_u2099_x2b_x2a____1___closed__9 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_u2099_x2b_x2a____1___closed__9_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_u2099_x2b_x2a____1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_u2099_x2b_x2a____1___closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_u2099_x2b_x2a____1___closed__10 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_u2099_x2b_x2a____1___closed__10_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_u2099_x2b_x2a____1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_u2099_x2b_x2a____1___closed__8_value),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_u2099_x2b_x2a____1___closed__10_value)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_u2099_x2b_x2a____1___closed__11 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_u2099_x2b_x2a____1___closed__11_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_u2099_x2b_x2a____1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_u2099_x2b_x2a____1___closed__12 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_u2099_x2b_x2a____1___closed__12_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_u2099_x2b_x2a____1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_u2099_x2b_x2a____1___closed__12_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_u2099_x2b_x2a____1___closed__13 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_u2099_x2b_x2a____1___closed__13_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_u2099_x2b_x2a____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_u2099_x2b_x2a____1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______unexpand__NonUnitalRingHom__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______unexpand__NonUnitalRingHom__1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______unexpand__NonUnitalRingHom__1___closed__0_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______unexpand__NonUnitalRingHom__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______unexpand__NonUnitalRingHom__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______unexpand__NonUnitalRingHom__1___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______unexpand__NonUnitalRingHom__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______unexpand__NonUnitalRingHom__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______unexpand__NonUnitalRingHom__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHomClass_toNonUnitalRingHom___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHomClass_toNonUnitalRingHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHomClass_toNonUnitalRingHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCNonUnitalRingHom___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCNonUnitalRingHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_copy___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_copy___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_copy(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_copy___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_id___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_id___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_NonUnitalRingHom_id___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_NonUnitalRingHom_id___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_NonUnitalRingHom_id___closed__0 = (const lean_object*)&lp_mathlib_NonUnitalRingHom_id___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_id(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_id___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_instZero___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_instZero___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_instZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_instZero(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_instZero___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_instInhabited___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_instInhabited(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_instInhabited___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_comp___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_comp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_comp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_instMonoidWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_instMonoidWithZero(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_toAddMonoidHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_toAddMonoidHom___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_toAddMonoidHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_toAddMonoidHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_toNonUnitalRingHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_toNonUnitalRingHom___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_toNonUnitalRingHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_toNonUnitalRingHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_toMonoidWithZeroHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_toMonoidWithZeroHom___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_toMonoidWithZeroHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_toMonoidWithZeroHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_term___u2192_x2b_x2a___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 9, .m_data = "term_→+*_"};
static const lean_object* lp_mathlib_term___u2192_x2b_x2a___00__closed__0 = (const lean_object*)&lp_mathlib_term___u2192_x2b_x2a___00__closed__0_value;
static const lean_ctor_object lp_mathlib_term___u2192_x2b_x2a___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2192_x2b_x2a___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(108, 80, 127, 194, 217, 137, 214, 251)}};
static const lean_object* lp_mathlib_term___u2192_x2b_x2a___00__closed__1 = (const lean_object*)&lp_mathlib_term___u2192_x2b_x2a___00__closed__1_value;
static const lean_string_object lp_mathlib_term___u2192_x2b_x2a___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 5, .m_data = " →+* "};
static const lean_object* lp_mathlib_term___u2192_x2b_x2a___00__closed__2 = (const lean_object*)&lp_mathlib_term___u2192_x2b_x2a___00__closed__2_value;
static const lean_ctor_object lp_mathlib_term___u2192_x2b_x2a___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192_x2b_x2a___00__closed__2_value)}};
static const lean_object* lp_mathlib_term___u2192_x2b_x2a___00__closed__3 = (const lean_object*)&lp_mathlib_term___u2192_x2b_x2a___00__closed__3_value;
static const lean_ctor_object lp_mathlib_term___u2192_x2b_x2a___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192_u2099_x2b_x2a___00__closed__3_value),((lean_object*)&lp_mathlib_term___u2192_x2b_x2a___00__closed__3_value),((lean_object*)&lp_mathlib_term___u2192_u2099_x2b_x2a___00__closed__8_value)}};
static const lean_object* lp_mathlib_term___u2192_x2b_x2a___00__closed__4 = (const lean_object*)&lp_mathlib_term___u2192_x2b_x2a___00__closed__4_value;
static const lean_ctor_object lp_mathlib_term___u2192_x2b_x2a___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192_x2b_x2a___00__closed__1_value),((lean_object*)(((size_t)(25) << 1) | 1)),((lean_object*)(((size_t)(26) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2192_x2b_x2a___00__closed__4_value)}};
static const lean_object* lp_mathlib_term___u2192_x2b_x2a___00__closed__5 = (const lean_object*)&lp_mathlib_term___u2192_x2b_x2a___00__closed__5_value;
LEAN_EXPORT const lean_object* lp_mathlib_term___u2192_x2b_x2a__ = (const lean_object*)&lp_mathlib_term___u2192_x2b_x2a___00__closed__5_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_x2b_x2a____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "RingHom"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_x2b_x2a____1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_x2b_x2a____1___closed__0_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_x2b_x2a____1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_x2b_x2a____1___closed__1;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_x2b_x2a____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_x2b_x2a____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(193, 71, 107, 83, 214, 46, 125, 66)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_x2b_x2a____1___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_x2b_x2a____1___closed__2_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_x2b_x2a____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_x2b_x2a____1___closed__2_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_x2b_x2a____1___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_x2b_x2a____1___closed__3_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_x2b_x2a____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_x2b_x2a____1___closed__2_value)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_x2b_x2a____1___closed__4 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_x2b_x2a____1___closed__4_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_x2b_x2a____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_x2b_x2a____1___closed__4_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_x2b_x2a____1___closed__5 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_x2b_x2a____1___closed__5_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_x2b_x2a____1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_x2b_x2a____1___closed__3_value),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_x2b_x2a____1___closed__5_value)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_x2b_x2a____1___closed__6 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_x2b_x2a____1___closed__6_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_x2b_x2a____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_x2b_x2a____1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______unexpand__RingHom__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______unexpand__RingHom__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHomClass_toRingHom___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHomClass_toRingHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHomClass_toRingHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCRingHom___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCRingHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_coeToMonoidHom___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_RingHom_coeToMonoidHom___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_RingHom_coeToMonoidHom___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_RingHom_coeToMonoidHom___closed__0 = (const lean_object*)&lp_mathlib_RingHom_coeToMonoidHom___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_RingHom_coeToMonoidHom(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_coeToMonoidHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_copy___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_copy___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_copy(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_copy___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_mk_x27___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_mk_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_mk_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_mk_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_id(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_id___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_instInhabited(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_instInhabited___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_comp___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_comp___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_comp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_comp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_instOne(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_instOne___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_instMul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_instMul(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_instMonoid___redArg___lam__1(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_RingHom_instMonoid___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_RingHom_instMonoid___redArg___lam__1, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_RingHom_instMonoid___redArg___closed__0 = (const lean_object*)&lp_mathlib_RingHom_instMonoid___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_RingHom_instMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_instMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mkRingHomOfMulSelfOfTwoNeZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mkRingHomOfMulSelfOfTwoNeZero___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mkRingHomOfMulSelfOfTwoNeZero(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mkRingHomOfMulSelfOfTwoNeZero___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_toAddMonoidHom___redArg(lean_object* v_self_1_){
_start:
{
lean_inc(v_self_1_);
return v_self_1_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_toAddMonoidHom___redArg___boxed(lean_object* v_self_2_){
_start:
{
lean_object* v_res_3_; 
v_res_3_ = lp_mathlib_NonUnitalRingHom_toAddMonoidHom___redArg(v_self_2_);
lean_dec(v_self_2_);
return v_res_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_toAddMonoidHom(lean_object* v_00_u03b1_4_, lean_object* v_00_u03b2_5_, lean_object* v_inst_6_, lean_object* v_inst_7_, lean_object* v_self_8_){
_start:
{
lean_inc(v_self_8_);
return v_self_8_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_toAddMonoidHom___boxed(lean_object* v_00_u03b1_9_, lean_object* v_00_u03b2_10_, lean_object* v_inst_11_, lean_object* v_inst_12_, lean_object* v_self_13_){
_start:
{
lean_object* v_res_14_; 
v_res_14_ = lp_mathlib_NonUnitalRingHom_toAddMonoidHom(v_00_u03b1_9_, v_00_u03b2_10_, v_inst_11_, v_inst_12_, v_self_13_);
lean_dec(v_self_13_);
lean_dec_ref(v_inst_12_);
lean_dec_ref(v_inst_11_);
return v_res_14_;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_u2099_x2b_x2a____1___closed__6(void){
_start:
{
lean_object* v___x_50_; lean_object* v___x_51_; 
v___x_50_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_u2099_x2b_x2a____1___closed__5));
v___x_51_ = l_String_toRawSubstring_x27(v___x_50_);
return v___x_51_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_u2099_x2b_x2a____1(lean_object* v_x_68_, lean_object* v_a_69_, lean_object* v_a_70_){
_start:
{
lean_object* v___x_71_; uint8_t v___x_72_; 
v___x_71_ = ((lean_object*)(lp_mathlib_term___u2192_u2099_x2b_x2a___00__closed__1));
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
v___x_84_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_u2099_x2b_x2a____1___closed__4));
v___x_85_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_u2099_x2b_x2a____1___closed__6, &lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_u2099_x2b_x2a____1___closed__6_once, _init_lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_u2099_x2b_x2a____1___closed__6);
v___x_86_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_u2099_x2b_x2a____1___closed__7));
lean_inc(v_currMacroScope_76_);
lean_inc(v_quotContext_75_);
v___x_87_ = l_Lean_addMacroScope(v_quotContext_75_, v___x_86_, v_currMacroScope_76_);
v___x_88_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_u2099_x2b_x2a____1___closed__11));
lean_inc_n(v___x_83_, 2);
v___x_89_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_89_, 0, v___x_83_);
lean_ctor_set(v___x_89_, 1, v___x_85_);
lean_ctor_set(v___x_89_, 2, v___x_87_);
lean_ctor_set(v___x_89_, 3, v___x_88_);
v___x_90_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_u2099_x2b_x2a____1___closed__13));
v___x_91_ = l_Lean_Syntax_node2(v___x_83_, v___x_90_, v___x_79_, v___x_81_);
v___x_92_ = l_Lean_Syntax_node2(v___x_83_, v___x_84_, v___x_89_, v___x_91_);
v___x_93_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_93_, 0, v___x_92_);
lean_ctor_set(v___x_93_, 1, v_a_70_);
return v___x_93_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_u2099_x2b_x2a____1___boxed(lean_object* v_x_94_, lean_object* v_a_95_, lean_object* v_a_96_){
_start:
{
lean_object* v_res_97_; 
v_res_97_ = lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_u2099_x2b_x2a____1(v_x_94_, v_a_95_, v_a_96_);
lean_dec_ref(v_a_95_);
return v_res_97_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______unexpand__NonUnitalRingHom__1(lean_object* v_x_101_, lean_object* v_a_102_, lean_object* v_a_103_){
_start:
{
lean_object* v___x_104_; uint8_t v___x_105_; 
v___x_104_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_u2099_x2b_x2a____1___closed__4));
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
v___x_110_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______unexpand__NonUnitalRingHom__1___closed__1));
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
v___x_125_ = ((lean_object*)(lp_mathlib_term___u2192_u2099_x2b_x2a___00__closed__1));
v___x_126_ = ((lean_object*)(lp_mathlib_term___u2192_u2099_x2b_x2a___00__closed__4));
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
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______unexpand__NonUnitalRingHom__1___boxed(lean_object* v_x_130_, lean_object* v_a_131_, lean_object* v_a_132_){
_start:
{
lean_object* v_res_133_; 
v_res_133_ = lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______unexpand__NonUnitalRingHom__1(v_x_130_, v_a_131_, v_a_132_);
lean_dec(v_a_131_);
return v_res_133_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHomClass_toNonUnitalRingHom___redArg(lean_object* v_inst_134_, lean_object* v_f_135_){
_start:
{
lean_object* v___x_136_; 
v___x_136_ = lean_apply_1(v_inst_134_, v_f_135_);
return v___x_136_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHomClass_toNonUnitalRingHom(lean_object* v_F_137_, lean_object* v_00_u03b1_138_, lean_object* v_00_u03b2_139_, lean_object* v_inst_140_, lean_object* v_inst_141_, lean_object* v_inst_142_, lean_object* v_inst_143_, lean_object* v_f_144_){
_start:
{
lean_object* v___x_145_; 
v___x_145_ = lean_apply_1(v_inst_142_, v_f_144_);
return v___x_145_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHomClass_toNonUnitalRingHom___boxed(lean_object* v_F_146_, lean_object* v_00_u03b1_147_, lean_object* v_00_u03b2_148_, lean_object* v_inst_149_, lean_object* v_inst_150_, lean_object* v_inst_151_, lean_object* v_inst_152_, lean_object* v_f_153_){
_start:
{
lean_object* v_res_154_; 
v_res_154_ = lp_mathlib_NonUnitalRingHomClass_toNonUnitalRingHom(v_F_146_, v_00_u03b1_147_, v_00_u03b2_148_, v_inst_149_, v_inst_150_, v_inst_151_, v_inst_152_, v_f_153_);
lean_dec_ref(v_inst_150_);
lean_dec_ref(v_inst_149_);
return v_res_154_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCNonUnitalRingHom___redArg(lean_object* v_inst_155_, lean_object* v_inst_156_, lean_object* v_inst_157_){
_start:
{
lean_object* v___x_158_; 
v___x_158_ = lean_alloc_closure((void*)(lp_mathlib_NonUnitalRingHomClass_toNonUnitalRingHom___boxed), 8, 7);
lean_closure_set(v___x_158_, 0, lean_box(0));
lean_closure_set(v___x_158_, 1, lean_box(0));
lean_closure_set(v___x_158_, 2, lean_box(0));
lean_closure_set(v___x_158_, 3, v_inst_155_);
lean_closure_set(v___x_158_, 4, v_inst_156_);
lean_closure_set(v___x_158_, 5, v_inst_157_);
lean_closure_set(v___x_158_, 6, lean_box(0));
return v___x_158_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCNonUnitalRingHom(lean_object* v_F_159_, lean_object* v_00_u03b1_160_, lean_object* v_00_u03b2_161_, lean_object* v_inst_162_, lean_object* v_inst_163_, lean_object* v_inst_164_, lean_object* v_inst_165_){
_start:
{
lean_object* v___x_166_; 
v___x_166_ = lean_alloc_closure((void*)(lp_mathlib_NonUnitalRingHomClass_toNonUnitalRingHom___boxed), 8, 7);
lean_closure_set(v___x_166_, 0, lean_box(0));
lean_closure_set(v___x_166_, 1, lean_box(0));
lean_closure_set(v___x_166_, 2, lean_box(0));
lean_closure_set(v___x_166_, 3, v_inst_162_);
lean_closure_set(v___x_166_, 4, v_inst_163_);
lean_closure_set(v___x_166_, 5, v_inst_164_);
lean_closure_set(v___x_166_, 6, lean_box(0));
return v___x_166_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_copy___redArg(lean_object* v_f_x27_167_){
_start:
{
lean_inc(v_f_x27_167_);
return v_f_x27_167_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_copy___redArg___boxed(lean_object* v_f_x27_168_){
_start:
{
lean_object* v_res_169_; 
v_res_169_ = lp_mathlib_NonUnitalRingHom_copy___redArg(v_f_x27_168_);
lean_dec(v_f_x27_168_);
return v_res_169_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_copy(lean_object* v_00_u03b1_170_, lean_object* v_00_u03b2_171_, lean_object* v_inst_172_, lean_object* v_inst_173_, lean_object* v_f_174_, lean_object* v_f_x27_175_, lean_object* v_h_176_){
_start:
{
lean_inc(v_f_x27_175_);
return v_f_x27_175_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_copy___boxed(lean_object* v_00_u03b1_177_, lean_object* v_00_u03b2_178_, lean_object* v_inst_179_, lean_object* v_inst_180_, lean_object* v_f_181_, lean_object* v_f_x27_182_, lean_object* v_h_183_){
_start:
{
lean_object* v_res_184_; 
v_res_184_ = lp_mathlib_NonUnitalRingHom_copy(v_00_u03b1_177_, v_00_u03b2_178_, v_inst_179_, v_inst_180_, v_f_181_, v_f_x27_182_, v_h_183_);
lean_dec(v_f_x27_182_);
lean_dec(v_f_181_);
lean_dec_ref(v_inst_180_);
lean_dec_ref(v_inst_179_);
return v_res_184_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_id___lam__0(lean_object* v_x_185_){
_start:
{
lean_inc(v_x_185_);
return v_x_185_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_id___lam__0___boxed(lean_object* v_x_186_){
_start:
{
lean_object* v_res_187_; 
v_res_187_ = lp_mathlib_NonUnitalRingHom_id___lam__0(v_x_186_);
lean_dec(v_x_186_);
return v_res_187_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_id(lean_object* v_00_u03b1_189_, lean_object* v_inst_190_){
_start:
{
lean_object* v___f_191_; 
v___f_191_ = ((lean_object*)(lp_mathlib_NonUnitalRingHom_id___closed__0));
return v___f_191_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_id___boxed(lean_object* v_00_u03b1_192_, lean_object* v_inst_193_){
_start:
{
lean_object* v_res_194_; 
v_res_194_ = lp_mathlib_NonUnitalRingHom_id(v_00_u03b1_192_, v_inst_193_);
lean_dec_ref(v_inst_193_);
return v_res_194_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_instZero___redArg___lam__0(lean_object* v_toZero_195_, lean_object* v_x_196_){
_start:
{
lean_inc(v_toZero_195_);
return v_toZero_195_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_instZero___redArg___lam__0___boxed(lean_object* v_toZero_197_, lean_object* v_x_198_){
_start:
{
lean_object* v_res_199_; 
v_res_199_ = lp_mathlib_NonUnitalRingHom_instZero___redArg___lam__0(v_toZero_197_, v_x_198_);
lean_dec(v_x_198_);
lean_dec(v_toZero_197_);
return v_res_199_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_instZero___redArg(lean_object* v_inst_200_){
_start:
{
lean_object* v___x_201_; lean_object* v_toZero_202_; lean_object* v___f_203_; 
v___x_201_ = lp_mathlib_NonUnitalNonAssocSemiring_toMulZeroClass___redArg(v_inst_200_);
v_toZero_202_ = lean_ctor_get(v___x_201_, 1);
lean_inc(v_toZero_202_);
lean_dec_ref(v___x_201_);
v___f_203_ = lean_alloc_closure((void*)(lp_mathlib_NonUnitalRingHom_instZero___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_203_, 0, v_toZero_202_);
return v___f_203_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_instZero(lean_object* v_00_u03b1_204_, lean_object* v_00_u03b2_205_, lean_object* v_inst_206_, lean_object* v_inst_207_){
_start:
{
lean_object* v___x_208_; 
v___x_208_ = lp_mathlib_NonUnitalRingHom_instZero___redArg(v_inst_207_);
return v___x_208_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_instZero___boxed(lean_object* v_00_u03b1_209_, lean_object* v_00_u03b2_210_, lean_object* v_inst_211_, lean_object* v_inst_212_){
_start:
{
lean_object* v_res_213_; 
v_res_213_ = lp_mathlib_NonUnitalRingHom_instZero(v_00_u03b1_209_, v_00_u03b2_210_, v_inst_211_, v_inst_212_);
lean_dec_ref(v_inst_211_);
return v_res_213_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_instInhabited___redArg(lean_object* v_inst_214_){
_start:
{
lean_object* v___x_215_; lean_object* v_toZero_216_; lean_object* v___f_217_; 
v___x_215_ = lp_mathlib_NonUnitalNonAssocSemiring_toMulZeroClass___redArg(v_inst_214_);
v_toZero_216_ = lean_ctor_get(v___x_215_, 1);
lean_inc(v_toZero_216_);
lean_dec_ref(v___x_215_);
v___f_217_ = lean_alloc_closure((void*)(lp_mathlib_NonUnitalRingHom_instZero___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_217_, 0, v_toZero_216_);
return v___f_217_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_instInhabited(lean_object* v_00_u03b1_218_, lean_object* v_00_u03b2_219_, lean_object* v_inst_220_, lean_object* v_inst_221_){
_start:
{
lean_object* v___x_222_; 
v___x_222_ = lp_mathlib_NonUnitalRingHom_instInhabited___redArg(v_inst_221_);
return v___x_222_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_instInhabited___boxed(lean_object* v_00_u03b1_223_, lean_object* v_00_u03b2_224_, lean_object* v_inst_225_, lean_object* v_inst_226_){
_start:
{
lean_object* v_res_227_; 
v_res_227_ = lp_mathlib_NonUnitalRingHom_instInhabited(v_00_u03b1_223_, v_00_u03b2_224_, v_inst_225_, v_inst_226_);
lean_dec_ref(v_inst_225_);
return v_res_227_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_comp___redArg(lean_object* v_g_228_, lean_object* v_f_229_){
_start:
{
lean_object* v___f_230_; 
v___f_230_ = lean_alloc_closure((void*)(lp_mathlib_OneHom_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_230_, 0, v_f_229_);
lean_closure_set(v___f_230_, 1, v_g_228_);
return v___f_230_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_comp(lean_object* v_00_u03b1_231_, lean_object* v_00_u03b2_232_, lean_object* v_00_u03b3_233_, lean_object* v_inst_234_, lean_object* v_inst_235_, lean_object* v_inst_236_, lean_object* v_g_237_, lean_object* v_f_238_){
_start:
{
lean_object* v___f_239_; 
v___f_239_ = lean_alloc_closure((void*)(lp_mathlib_OneHom_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_239_, 0, v_f_238_);
lean_closure_set(v___f_239_, 1, v_g_237_);
return v___f_239_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_comp___boxed(lean_object* v_00_u03b1_240_, lean_object* v_00_u03b2_241_, lean_object* v_00_u03b3_242_, lean_object* v_inst_243_, lean_object* v_inst_244_, lean_object* v_inst_245_, lean_object* v_g_246_, lean_object* v_f_247_){
_start:
{
lean_object* v_res_248_; 
v_res_248_ = lp_mathlib_NonUnitalRingHom_comp(v_00_u03b1_240_, v_00_u03b2_241_, v_00_u03b3_242_, v_inst_243_, v_inst_244_, v_inst_245_, v_g_246_, v_f_247_);
lean_dec_ref(v_inst_245_);
lean_dec_ref(v_inst_244_);
lean_dec_ref(v_inst_243_);
return v_res_248_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_instMonoidWithZero___redArg(lean_object* v_inst_249_){
_start:
{
lean_object* v___f_250_; lean_object* v___x_251_; lean_object* v___x_252_; lean_object* v___x_253_; lean_object* v___x_254_; lean_object* v___x_255_; 
v___f_250_ = ((lean_object*)(lp_mathlib_NonUnitalRingHom_id___closed__0));
lean_inc_ref_n(v_inst_249_, 3);
v___x_251_ = lean_alloc_closure((void*)(lp_mathlib_NonUnitalRingHom_comp___boxed), 8, 6);
lean_closure_set(v___x_251_, 0, lean_box(0));
lean_closure_set(v___x_251_, 1, lean_box(0));
lean_closure_set(v___x_251_, 2, lean_box(0));
lean_closure_set(v___x_251_, 3, v_inst_249_);
lean_closure_set(v___x_251_, 4, v_inst_249_);
lean_closure_set(v___x_251_, 5, v_inst_249_);
lean_inc_ref(v___x_251_);
v___x_252_ = lean_alloc_closure((void*)(lp_mathlib_npowBinRecAuto___boxed), 5, 3);
lean_closure_set(v___x_252_, 0, lean_box(0));
lean_closure_set(v___x_252_, 1, v___x_251_);
lean_closure_set(v___x_252_, 2, v___f_250_);
v___x_253_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_253_, 0, v___f_250_);
lean_ctor_set(v___x_253_, 1, v___x_251_);
lean_ctor_set(v___x_253_, 2, v___x_252_);
v___x_254_ = lp_mathlib_NonUnitalRingHom_instZero___redArg(v_inst_249_);
v___x_255_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_255_, 0, v___x_253_);
lean_ctor_set(v___x_255_, 1, v___x_254_);
return v___x_255_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_instMonoidWithZero(lean_object* v_00_u03b1_256_, lean_object* v_inst_257_){
_start:
{
lean_object* v___x_258_; 
v___x_258_ = lp_mathlib_NonUnitalRingHom_instMonoidWithZero___redArg(v_inst_257_);
return v___x_258_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_toAddMonoidHom___redArg(lean_object* v_self_259_){
_start:
{
lean_inc(v_self_259_);
return v_self_259_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_toAddMonoidHom___redArg___boxed(lean_object* v_self_260_){
_start:
{
lean_object* v_res_261_; 
v_res_261_ = lp_mathlib_RingHom_toAddMonoidHom___redArg(v_self_260_);
lean_dec(v_self_260_);
return v_res_261_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_toAddMonoidHom(lean_object* v_00_u03b1_262_, lean_object* v_00_u03b2_263_, lean_object* v_inst_264_, lean_object* v_inst_265_, lean_object* v_self_266_){
_start:
{
lean_inc(v_self_266_);
return v_self_266_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_toAddMonoidHom___boxed(lean_object* v_00_u03b1_267_, lean_object* v_00_u03b2_268_, lean_object* v_inst_269_, lean_object* v_inst_270_, lean_object* v_self_271_){
_start:
{
lean_object* v_res_272_; 
v_res_272_ = lp_mathlib_RingHom_toAddMonoidHom(v_00_u03b1_267_, v_00_u03b2_268_, v_inst_269_, v_inst_270_, v_self_271_);
lean_dec(v_self_271_);
lean_dec_ref(v_inst_270_);
lean_dec_ref(v_inst_269_);
return v_res_272_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_toNonUnitalRingHom___redArg(lean_object* v_self_273_){
_start:
{
lean_inc(v_self_273_);
return v_self_273_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_toNonUnitalRingHom___redArg___boxed(lean_object* v_self_274_){
_start:
{
lean_object* v_res_275_; 
v_res_275_ = lp_mathlib_RingHom_toNonUnitalRingHom___redArg(v_self_274_);
lean_dec(v_self_274_);
return v_res_275_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_toNonUnitalRingHom(lean_object* v_00_u03b1_276_, lean_object* v_00_u03b2_277_, lean_object* v_inst_278_, lean_object* v_inst_279_, lean_object* v_self_280_){
_start:
{
lean_inc(v_self_280_);
return v_self_280_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_toNonUnitalRingHom___boxed(lean_object* v_00_u03b1_281_, lean_object* v_00_u03b2_282_, lean_object* v_inst_283_, lean_object* v_inst_284_, lean_object* v_self_285_){
_start:
{
lean_object* v_res_286_; 
v_res_286_ = lp_mathlib_RingHom_toNonUnitalRingHom(v_00_u03b1_281_, v_00_u03b2_282_, v_inst_283_, v_inst_284_, v_self_285_);
lean_dec(v_self_285_);
lean_dec_ref(v_inst_284_);
lean_dec_ref(v_inst_283_);
return v_res_286_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_toMonoidWithZeroHom___redArg(lean_object* v_self_287_){
_start:
{
lean_inc(v_self_287_);
return v_self_287_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_toMonoidWithZeroHom___redArg___boxed(lean_object* v_self_288_){
_start:
{
lean_object* v_res_289_; 
v_res_289_ = lp_mathlib_RingHom_toMonoidWithZeroHom___redArg(v_self_288_);
lean_dec(v_self_288_);
return v_res_289_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_toMonoidWithZeroHom(lean_object* v_00_u03b1_290_, lean_object* v_00_u03b2_291_, lean_object* v_inst_292_, lean_object* v_inst_293_, lean_object* v_self_294_){
_start:
{
lean_inc(v_self_294_);
return v_self_294_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_toMonoidWithZeroHom___boxed(lean_object* v_00_u03b1_295_, lean_object* v_00_u03b2_296_, lean_object* v_inst_297_, lean_object* v_inst_298_, lean_object* v_self_299_){
_start:
{
lean_object* v_res_300_; 
v_res_300_ = lp_mathlib_RingHom_toMonoidWithZeroHom(v_00_u03b1_295_, v_00_u03b2_296_, v_inst_297_, v_inst_298_, v_self_299_);
lean_dec(v_self_299_);
lean_dec_ref(v_inst_298_);
lean_dec_ref(v_inst_297_);
return v_res_300_;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_x2b_x2a____1___closed__1(void){
_start:
{
lean_object* v___x_318_; lean_object* v___x_319_; 
v___x_318_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_x2b_x2a____1___closed__0));
v___x_319_ = l_String_toRawSubstring_x27(v___x_318_);
return v___x_319_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_x2b_x2a____1(lean_object* v_x_333_, lean_object* v_a_334_, lean_object* v_a_335_){
_start:
{
lean_object* v___x_336_; uint8_t v___x_337_; 
v___x_336_ = ((lean_object*)(lp_mathlib_term___u2192_x2b_x2a___00__closed__1));
lean_inc(v_x_333_);
v___x_337_ = l_Lean_Syntax_isOfKind(v_x_333_, v___x_336_);
if (v___x_337_ == 0)
{
lean_object* v___x_338_; lean_object* v___x_339_; 
lean_dec(v_x_333_);
v___x_338_ = lean_box(1);
v___x_339_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_339_, 0, v___x_338_);
lean_ctor_set(v___x_339_, 1, v_a_335_);
return v___x_339_;
}
else
{
lean_object* v_quotContext_340_; lean_object* v_currMacroScope_341_; lean_object* v_ref_342_; lean_object* v___x_343_; lean_object* v___x_344_; lean_object* v___x_345_; lean_object* v___x_346_; uint8_t v___x_347_; lean_object* v___x_348_; lean_object* v___x_349_; lean_object* v___x_350_; lean_object* v___x_351_; lean_object* v___x_352_; lean_object* v___x_353_; lean_object* v___x_354_; lean_object* v___x_355_; lean_object* v___x_356_; lean_object* v___x_357_; lean_object* v___x_358_; 
v_quotContext_340_ = lean_ctor_get(v_a_334_, 1);
v_currMacroScope_341_ = lean_ctor_get(v_a_334_, 2);
v_ref_342_ = lean_ctor_get(v_a_334_, 5);
v___x_343_ = lean_unsigned_to_nat(0u);
v___x_344_ = l_Lean_Syntax_getArg(v_x_333_, v___x_343_);
v___x_345_ = lean_unsigned_to_nat(2u);
v___x_346_ = l_Lean_Syntax_getArg(v_x_333_, v___x_345_);
lean_dec(v_x_333_);
v___x_347_ = 0;
v___x_348_ = l_Lean_SourceInfo_fromRef(v_ref_342_, v___x_347_);
v___x_349_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_u2099_x2b_x2a____1___closed__4));
v___x_350_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_x2b_x2a____1___closed__1, &lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_x2b_x2a____1___closed__1_once, _init_lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_x2b_x2a____1___closed__1);
v___x_351_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_x2b_x2a____1___closed__2));
lean_inc(v_currMacroScope_341_);
lean_inc(v_quotContext_340_);
v___x_352_ = l_Lean_addMacroScope(v_quotContext_340_, v___x_351_, v_currMacroScope_341_);
v___x_353_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_x2b_x2a____1___closed__6));
lean_inc_n(v___x_348_, 2);
v___x_354_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_354_, 0, v___x_348_);
lean_ctor_set(v___x_354_, 1, v___x_350_);
lean_ctor_set(v___x_354_, 2, v___x_352_);
lean_ctor_set(v___x_354_, 3, v___x_353_);
v___x_355_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_u2099_x2b_x2a____1___closed__13));
v___x_356_ = l_Lean_Syntax_node2(v___x_348_, v___x_355_, v___x_344_, v___x_346_);
v___x_357_ = l_Lean_Syntax_node2(v___x_348_, v___x_349_, v___x_354_, v___x_356_);
v___x_358_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_358_, 0, v___x_357_);
lean_ctor_set(v___x_358_, 1, v_a_335_);
return v___x_358_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_x2b_x2a____1___boxed(lean_object* v_x_359_, lean_object* v_a_360_, lean_object* v_a_361_){
_start:
{
lean_object* v_res_362_; 
v_res_362_ = lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_x2b_x2a____1(v_x_359_, v_a_360_, v_a_361_);
lean_dec_ref(v_a_360_);
return v_res_362_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______unexpand__RingHom__1(lean_object* v_x_363_, lean_object* v_a_364_, lean_object* v_a_365_){
_start:
{
lean_object* v___x_366_; uint8_t v___x_367_; 
v___x_366_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______macroRules__term___u2192_u2099_x2b_x2a____1___closed__4));
lean_inc(v_x_363_);
v___x_367_ = l_Lean_Syntax_isOfKind(v_x_363_, v___x_366_);
if (v___x_367_ == 0)
{
lean_object* v___x_368_; lean_object* v___x_369_; 
lean_dec(v_x_363_);
v___x_368_ = lean_box(0);
v___x_369_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_369_, 0, v___x_368_);
lean_ctor_set(v___x_369_, 1, v_a_365_);
return v___x_369_;
}
else
{
lean_object* v___x_370_; lean_object* v___x_371_; lean_object* v___x_372_; uint8_t v___x_373_; 
v___x_370_ = lean_unsigned_to_nat(0u);
v___x_371_ = l_Lean_Syntax_getArg(v_x_363_, v___x_370_);
v___x_372_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______unexpand__NonUnitalRingHom__1___closed__1));
lean_inc(v___x_371_);
v___x_373_ = l_Lean_Syntax_isOfKind(v___x_371_, v___x_372_);
if (v___x_373_ == 0)
{
lean_object* v___x_374_; lean_object* v___x_375_; 
lean_dec(v___x_371_);
lean_dec(v_x_363_);
v___x_374_ = lean_box(0);
v___x_375_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_375_, 0, v___x_374_);
lean_ctor_set(v___x_375_, 1, v_a_365_);
return v___x_375_;
}
else
{
lean_object* v___x_376_; lean_object* v___x_377_; lean_object* v___x_378_; uint8_t v___x_379_; 
v___x_376_ = lean_unsigned_to_nat(1u);
v___x_377_ = l_Lean_Syntax_getArg(v_x_363_, v___x_376_);
lean_dec(v_x_363_);
v___x_378_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_377_);
v___x_379_ = l_Lean_Syntax_matchesNull(v___x_377_, v___x_378_);
if (v___x_379_ == 0)
{
lean_object* v___x_380_; lean_object* v___x_381_; 
lean_dec(v___x_377_);
lean_dec(v___x_371_);
v___x_380_ = lean_box(0);
v___x_381_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_381_, 0, v___x_380_);
lean_ctor_set(v___x_381_, 1, v_a_365_);
return v___x_381_;
}
else
{
lean_object* v___x_382_; lean_object* v___x_383_; lean_object* v_ref_384_; uint8_t v___x_385_; lean_object* v___x_386_; lean_object* v___x_387_; lean_object* v___x_388_; lean_object* v___x_389_; lean_object* v___x_390_; lean_object* v___x_391_; 
v___x_382_ = l_Lean_Syntax_getArg(v___x_377_, v___x_370_);
v___x_383_ = l_Lean_Syntax_getArg(v___x_377_, v___x_376_);
lean_dec(v___x_377_);
v_ref_384_ = l_Lean_replaceRef(v___x_371_, v_a_364_);
lean_dec(v___x_371_);
v___x_385_ = 0;
v___x_386_ = l_Lean_SourceInfo_fromRef(v_ref_384_, v___x_385_);
lean_dec(v_ref_384_);
v___x_387_ = ((lean_object*)(lp_mathlib_term___u2192_x2b_x2a___00__closed__1));
v___x_388_ = ((lean_object*)(lp_mathlib_term___u2192_x2b_x2a___00__closed__2));
lean_inc(v___x_386_);
v___x_389_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_389_, 0, v___x_386_);
lean_ctor_set(v___x_389_, 1, v___x_388_);
v___x_390_ = l_Lean_Syntax_node3(v___x_386_, v___x_387_, v___x_382_, v___x_389_, v___x_383_);
v___x_391_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_391_, 0, v___x_390_);
lean_ctor_set(v___x_391_, 1, v_a_365_);
return v___x_391_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______unexpand__RingHom__1___boxed(lean_object* v_x_392_, lean_object* v_a_393_, lean_object* v_a_394_){
_start:
{
lean_object* v_res_395_; 
v_res_395_ = lp_mathlib___aux__Mathlib__Algebra__Ring__Hom__Defs______unexpand__RingHom__1(v_x_392_, v_a_393_, v_a_394_);
lean_dec(v_a_393_);
return v_res_395_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHomClass_toRingHom___redArg(lean_object* v_inst_396_, lean_object* v_f_397_){
_start:
{
lean_object* v___x_398_; 
v___x_398_ = lean_apply_1(v_inst_396_, v_f_397_);
return v___x_398_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHomClass_toRingHom(lean_object* v_F_399_, lean_object* v_00_u03b1_400_, lean_object* v_00_u03b2_401_, lean_object* v_inst_402_, lean_object* v_x_403_, lean_object* v_x_404_, lean_object* v_inst_405_, lean_object* v_f_406_){
_start:
{
lean_object* v___x_407_; 
v___x_407_ = lean_apply_1(v_inst_402_, v_f_406_);
return v___x_407_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHomClass_toRingHom___boxed(lean_object* v_F_408_, lean_object* v_00_u03b1_409_, lean_object* v_00_u03b2_410_, lean_object* v_inst_411_, lean_object* v_x_412_, lean_object* v_x_413_, lean_object* v_inst_414_, lean_object* v_f_415_){
_start:
{
lean_object* v_res_416_; 
v_res_416_ = lp_mathlib_RingHomClass_toRingHom(v_F_408_, v_00_u03b1_409_, v_00_u03b2_410_, v_inst_411_, v_x_412_, v_x_413_, v_inst_414_, v_f_415_);
lean_dec_ref(v_x_413_);
lean_dec_ref(v_x_412_);
return v_res_416_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCRingHom___redArg(lean_object* v_inst_417_, lean_object* v_x_418_, lean_object* v_x_419_){
_start:
{
lean_object* v___x_420_; 
v___x_420_ = lean_alloc_closure((void*)(lp_mathlib_RingHomClass_toRingHom___boxed), 8, 7);
lean_closure_set(v___x_420_, 0, lean_box(0));
lean_closure_set(v___x_420_, 1, lean_box(0));
lean_closure_set(v___x_420_, 2, lean_box(0));
lean_closure_set(v___x_420_, 3, v_inst_417_);
lean_closure_set(v___x_420_, 4, v_x_418_);
lean_closure_set(v___x_420_, 5, v_x_419_);
lean_closure_set(v___x_420_, 6, lean_box(0));
return v___x_420_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCRingHom(lean_object* v_F_421_, lean_object* v_00_u03b1_422_, lean_object* v_00_u03b2_423_, lean_object* v_inst_424_, lean_object* v_x_425_, lean_object* v_x_426_, lean_object* v_inst_427_){
_start:
{
lean_object* v___x_428_; 
v___x_428_ = lean_alloc_closure((void*)(lp_mathlib_RingHomClass_toRingHom___boxed), 8, 7);
lean_closure_set(v___x_428_, 0, lean_box(0));
lean_closure_set(v___x_428_, 1, lean_box(0));
lean_closure_set(v___x_428_, 2, lean_box(0));
lean_closure_set(v___x_428_, 3, v_inst_424_);
lean_closure_set(v___x_428_, 4, v_x_425_);
lean_closure_set(v___x_428_, 5, v_x_426_);
lean_closure_set(v___x_428_, 6, lean_box(0));
return v___x_428_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_coeToMonoidHom___lam__0(lean_object* v_self_429_, lean_object* v___y_430_){
_start:
{
lean_object* v___x_431_; 
v___x_431_ = lean_apply_1(v_self_429_, v___y_430_);
return v___x_431_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_coeToMonoidHom(lean_object* v_00_u03b1_433_, lean_object* v_00_u03b2_434_, lean_object* v_x_435_, lean_object* v_x_436_){
_start:
{
lean_object* v___f_437_; 
v___f_437_ = ((lean_object*)(lp_mathlib_RingHom_coeToMonoidHom___closed__0));
return v___f_437_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_coeToMonoidHom___boxed(lean_object* v_00_u03b1_438_, lean_object* v_00_u03b2_439_, lean_object* v_x_440_, lean_object* v_x_441_){
_start:
{
lean_object* v_res_442_; 
v_res_442_ = lp_mathlib_RingHom_coeToMonoidHom(v_00_u03b1_438_, v_00_u03b2_439_, v_x_440_, v_x_441_);
lean_dec_ref(v_x_441_);
lean_dec_ref(v_x_440_);
return v_res_442_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_copy___redArg(lean_object* v_f_x27_443_){
_start:
{
lean_inc(v_f_x27_443_);
return v_f_x27_443_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_copy___redArg___boxed(lean_object* v_f_x27_444_){
_start:
{
lean_object* v_res_445_; 
v_res_445_ = lp_mathlib_RingHom_copy___redArg(v_f_x27_444_);
lean_dec(v_f_x27_444_);
return v_res_445_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_copy(lean_object* v_00_u03b1_446_, lean_object* v_00_u03b2_447_, lean_object* v_x_448_, lean_object* v_x_449_, lean_object* v_f_450_, lean_object* v_f_x27_451_, lean_object* v_h_452_){
_start:
{
lean_inc(v_f_x27_451_);
return v_f_x27_451_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_copy___boxed(lean_object* v_00_u03b1_453_, lean_object* v_00_u03b2_454_, lean_object* v_x_455_, lean_object* v_x_456_, lean_object* v_f_457_, lean_object* v_f_x27_458_, lean_object* v_h_459_){
_start:
{
lean_object* v_res_460_; 
v_res_460_ = lp_mathlib_RingHom_copy(v_00_u03b1_453_, v_00_u03b2_454_, v_x_455_, v_x_456_, v_f_457_, v_f_x27_458_, v_h_459_);
lean_dec(v_f_x27_458_);
lean_dec(v_f_457_);
lean_dec_ref(v_x_456_);
lean_dec_ref(v_x_455_);
return v_res_460_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_mk_x27___redArg___lam__0(lean_object* v_f_461_, lean_object* v___y_462_){
_start:
{
lean_object* v___x_463_; 
v___x_463_ = lean_apply_1(v_f_461_, v___y_462_);
return v___x_463_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_mk_x27___redArg(lean_object* v_f_464_){
_start:
{
lean_object* v___f_465_; 
v___f_465_ = lean_alloc_closure((void*)(lp_mathlib_RingHom_mk_x27___redArg___lam__0), 2, 1);
lean_closure_set(v___f_465_, 0, v_f_464_);
return v___f_465_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_mk_x27(lean_object* v_00_u03b1_466_, lean_object* v_00_u03b2_467_, lean_object* v_inst_468_, lean_object* v_inst_469_, lean_object* v_f_470_, lean_object* v_map__add_471_){
_start:
{
lean_object* v___f_472_; 
v___f_472_ = lean_alloc_closure((void*)(lp_mathlib_RingHom_mk_x27___redArg___lam__0), 2, 1);
lean_closure_set(v___f_472_, 0, v_f_470_);
return v___f_472_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_mk_x27___boxed(lean_object* v_00_u03b1_473_, lean_object* v_00_u03b2_474_, lean_object* v_inst_475_, lean_object* v_inst_476_, lean_object* v_f_477_, lean_object* v_map__add_478_){
_start:
{
lean_object* v_res_479_; 
v_res_479_ = lp_mathlib_RingHom_mk_x27(v_00_u03b1_473_, v_00_u03b2_474_, v_inst_475_, v_inst_476_, v_f_477_, v_map__add_478_);
lean_dec_ref(v_inst_476_);
lean_dec_ref(v_inst_475_);
return v_res_479_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_id(lean_object* v_00_u03b1_480_, lean_object* v_inst_481_){
_start:
{
lean_object* v___f_482_; 
v___f_482_ = ((lean_object*)(lp_mathlib_NonUnitalRingHom_id___closed__0));
return v___f_482_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_id___boxed(lean_object* v_00_u03b1_483_, lean_object* v_inst_484_){
_start:
{
lean_object* v_res_485_; 
v_res_485_ = lp_mathlib_RingHom_id(v_00_u03b1_483_, v_inst_484_);
lean_dec_ref(v_inst_484_);
return v_res_485_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_instInhabited(lean_object* v_00_u03b1_486_, lean_object* v_x_487_){
_start:
{
lean_object* v___f_488_; 
v___f_488_ = ((lean_object*)(lp_mathlib_NonUnitalRingHom_id___closed__0));
return v___f_488_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_instInhabited___boxed(lean_object* v_00_u03b1_489_, lean_object* v_x_490_){
_start:
{
lean_object* v_res_491_; 
v_res_491_ = lp_mathlib_RingHom_instInhabited(v_00_u03b1_489_, v_x_490_);
lean_dec_ref(v_x_490_);
return v_res_491_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_comp___redArg___lam__0(lean_object* v_f_492_, lean_object* v_g_493_, lean_object* v_x_494_){
_start:
{
lean_object* v___x_495_; lean_object* v___x_496_; 
v___x_495_ = lean_apply_1(v_f_492_, v_x_494_);
v___x_496_ = lean_apply_1(v_g_493_, v___x_495_);
return v___x_496_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_comp___redArg(lean_object* v_g_497_, lean_object* v_f_498_){
_start:
{
lean_object* v___f_499_; 
v___f_499_ = lean_alloc_closure((void*)(lp_mathlib_RingHom_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_499_, 0, v_f_498_);
lean_closure_set(v___f_499_, 1, v_g_497_);
return v___f_499_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_comp(lean_object* v_00_u03b1_500_, lean_object* v_00_u03b2_501_, lean_object* v_00_u03b3_502_, lean_object* v_x_503_, lean_object* v_x_504_, lean_object* v_x_505_, lean_object* v_g_506_, lean_object* v_f_507_){
_start:
{
lean_object* v___f_508_; 
v___f_508_ = lean_alloc_closure((void*)(lp_mathlib_RingHom_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_508_, 0, v_f_507_);
lean_closure_set(v___f_508_, 1, v_g_506_);
return v___f_508_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_comp___boxed(lean_object* v_00_u03b1_509_, lean_object* v_00_u03b2_510_, lean_object* v_00_u03b3_511_, lean_object* v_x_512_, lean_object* v_x_513_, lean_object* v_x_514_, lean_object* v_g_515_, lean_object* v_f_516_){
_start:
{
lean_object* v_res_517_; 
v_res_517_ = lp_mathlib_RingHom_comp(v_00_u03b1_509_, v_00_u03b2_510_, v_00_u03b3_511_, v_x_512_, v_x_513_, v_x_514_, v_g_515_, v_f_516_);
lean_dec_ref(v_x_514_);
lean_dec_ref(v_x_513_);
lean_dec_ref(v_x_512_);
return v_res_517_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_instOne(lean_object* v_00_u03b1_518_, lean_object* v_x_519_){
_start:
{
lean_object* v___f_520_; 
v___f_520_ = ((lean_object*)(lp_mathlib_NonUnitalRingHom_id___closed__0));
return v___f_520_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_instOne___boxed(lean_object* v_00_u03b1_521_, lean_object* v_x_522_){
_start:
{
lean_object* v_res_523_; 
v_res_523_ = lp_mathlib_RingHom_instOne(v_00_u03b1_521_, v_x_522_);
lean_dec_ref(v_x_522_);
return v_res_523_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_instMul___redArg(lean_object* v_x_524_){
_start:
{
lean_object* v___x_525_; 
lean_inc_ref_n(v_x_524_, 2);
v___x_525_ = lean_alloc_closure((void*)(lp_mathlib_RingHom_comp___boxed), 8, 6);
lean_closure_set(v___x_525_, 0, lean_box(0));
lean_closure_set(v___x_525_, 1, lean_box(0));
lean_closure_set(v___x_525_, 2, lean_box(0));
lean_closure_set(v___x_525_, 3, v_x_524_);
lean_closure_set(v___x_525_, 4, v_x_524_);
lean_closure_set(v___x_525_, 5, v_x_524_);
return v___x_525_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_instMul(lean_object* v_00_u03b1_526_, lean_object* v_x_527_){
_start:
{
lean_object* v___x_528_; 
lean_inc_ref_n(v_x_527_, 2);
v___x_528_ = lean_alloc_closure((void*)(lp_mathlib_RingHom_comp___boxed), 8, 6);
lean_closure_set(v___x_528_, 0, lean_box(0));
lean_closure_set(v___x_528_, 1, lean_box(0));
lean_closure_set(v___x_528_, 2, lean_box(0));
lean_closure_set(v___x_528_, 3, v_x_527_);
lean_closure_set(v___x_528_, 4, v_x_527_);
lean_closure_set(v___x_528_, 5, v_x_527_);
return v___x_528_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_instMonoid___redArg___lam__1(lean_object* v_n_529_, lean_object* v_f_530_, lean_object* v___y_531_){
_start:
{
lean_object* v___f_532_; lean_object* v___x_533_; 
v___f_532_ = lean_alloc_closure((void*)(lp_mathlib_RingHom_mk_x27___redArg___lam__0), 2, 1);
lean_closure_set(v___f_532_, 0, v_f_530_);
v___x_533_ = lp_mathlib_Nat_iterate___redArg(v___f_532_, v_n_529_, v___y_531_);
return v___x_533_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_instMonoid___redArg(lean_object* v_x_535_){
_start:
{
lean_object* v___f_536_; lean_object* v___f_537_; lean_object* v___x_538_; lean_object* v___x_539_; 
v___f_536_ = ((lean_object*)(lp_mathlib_RingHom_instMonoid___redArg___closed__0));
v___f_537_ = ((lean_object*)(lp_mathlib_NonUnitalRingHom_id___closed__0));
lean_inc_ref_n(v_x_535_, 2);
v___x_538_ = lean_alloc_closure((void*)(lp_mathlib_RingHom_comp___boxed), 8, 6);
lean_closure_set(v___x_538_, 0, lean_box(0));
lean_closure_set(v___x_538_, 1, lean_box(0));
lean_closure_set(v___x_538_, 2, lean_box(0));
lean_closure_set(v___x_538_, 3, v_x_535_);
lean_closure_set(v___x_538_, 4, v_x_535_);
lean_closure_set(v___x_538_, 5, v_x_535_);
v___x_539_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_539_, 0, v___f_537_);
lean_ctor_set(v___x_539_, 1, v___x_538_);
lean_ctor_set(v___x_539_, 2, v___f_536_);
return v___x_539_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_instMonoid(lean_object* v_00_u03b1_540_, lean_object* v_x_541_){
_start:
{
lean_object* v___x_542_; 
v___x_542_ = lp_mathlib_RingHom_instMonoid___redArg(v_x_541_);
return v___x_542_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mkRingHomOfMulSelfOfTwoNeZero___redArg(lean_object* v_f_543_){
_start:
{
lean_inc(v_f_543_);
return v_f_543_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mkRingHomOfMulSelfOfTwoNeZero___redArg___boxed(lean_object* v_f_544_){
_start:
{
lean_object* v_res_545_; 
v_res_545_ = lp_mathlib_AddMonoidHom_mkRingHomOfMulSelfOfTwoNeZero___redArg(v_f_544_);
lean_dec(v_f_544_);
return v_res_545_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mkRingHomOfMulSelfOfTwoNeZero(lean_object* v_00_u03b1_546_, lean_object* v_00_u03b2_547_, lean_object* v_inst_548_, lean_object* v_inst_549_, lean_object* v_inst_550_, lean_object* v_f_551_, lean_object* v_h_552_, lean_object* v_h__two_553_, lean_object* v_h__one_554_){
_start:
{
lean_inc(v_f_551_);
return v_f_551_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mkRingHomOfMulSelfOfTwoNeZero___boxed(lean_object* v_00_u03b1_555_, lean_object* v_00_u03b2_556_, lean_object* v_inst_557_, lean_object* v_inst_558_, lean_object* v_inst_559_, lean_object* v_f_560_, lean_object* v_h_561_, lean_object* v_h__two_562_, lean_object* v_h__one_563_){
_start:
{
lean_object* v_res_564_; 
v_res_564_ = lp_mathlib_AddMonoidHom_mkRingHomOfMulSelfOfTwoNeZero(v_00_u03b1_555_, v_00_u03b2_556_, v_inst_557_, v_inst_558_, v_inst_559_, v_f_560_, v_h_561_, v_h__two_562_, v_h__one_563_);
lean_dec(v_f_560_);
lean_dec_ref(v_inst_559_);
lean_dec_ref(v_inst_557_);
return v_res_564_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Hom(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Hom_Defs(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Hom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Ring_Hom_Defs(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Hom(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Hom_Defs(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Hom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Hom_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Ring_Hom_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Ring_Hom_Defs(builtin);
}
#ifdef __cplusplus
}
#endif
