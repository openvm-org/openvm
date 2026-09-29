// Lean compiler output
// Module: Mathlib.Algebra.Group.Hom.Defs
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Defs public import Mathlib.Algebra.Notation.Pi.Defs public import Mathlib.Data.FunLike.Basic public import Mathlib.Logic.Function.Iterate
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
lean_object* lp_mathlib_MulOneClass_toMulOne___redArg(lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddZeroClass_toAddZero___redArg(lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Nat_iterate___redArg(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_term___u2192_u2099_x2b___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 9, .m_data = "term_→ₙ+_"};
static const lean_object* lp_mathlib_term___u2192_u2099_x2b___00__closed__0 = (const lean_object*)&lp_mathlib_term___u2192_u2099_x2b___00__closed__0_value;
static const lean_ctor_object lp_mathlib_term___u2192_u2099_x2b___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2192_u2099_x2b___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(62, 255, 97, 58, 78, 213, 71, 8)}};
static const lean_object* lp_mathlib_term___u2192_u2099_x2b___00__closed__1 = (const lean_object*)&lp_mathlib_term___u2192_u2099_x2b___00__closed__1_value;
static const lean_string_object lp_mathlib_term___u2192_u2099_x2b___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_term___u2192_u2099_x2b___00__closed__2 = (const lean_object*)&lp_mathlib_term___u2192_u2099_x2b___00__closed__2_value;
static const lean_ctor_object lp_mathlib_term___u2192_u2099_x2b___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2192_u2099_x2b___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_term___u2192_u2099_x2b___00__closed__3 = (const lean_object*)&lp_mathlib_term___u2192_u2099_x2b___00__closed__3_value;
static const lean_string_object lp_mathlib_term___u2192_u2099_x2b___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 5, .m_data = " →ₙ+ "};
static const lean_object* lp_mathlib_term___u2192_u2099_x2b___00__closed__4 = (const lean_object*)&lp_mathlib_term___u2192_u2099_x2b___00__closed__4_value;
static const lean_ctor_object lp_mathlib_term___u2192_u2099_x2b___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192_u2099_x2b___00__closed__4_value)}};
static const lean_object* lp_mathlib_term___u2192_u2099_x2b___00__closed__5 = (const lean_object*)&lp_mathlib_term___u2192_u2099_x2b___00__closed__5_value;
static const lean_string_object lp_mathlib_term___u2192_u2099_x2b___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_term___u2192_u2099_x2b___00__closed__6 = (const lean_object*)&lp_mathlib_term___u2192_u2099_x2b___00__closed__6_value;
static const lean_ctor_object lp_mathlib_term___u2192_u2099_x2b___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2192_u2099_x2b___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_term___u2192_u2099_x2b___00__closed__7 = (const lean_object*)&lp_mathlib_term___u2192_u2099_x2b___00__closed__7_value;
static const lean_ctor_object lp_mathlib_term___u2192_u2099_x2b___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192_u2099_x2b___00__closed__7_value),((lean_object*)(((size_t)(25) << 1) | 1))}};
static const lean_object* lp_mathlib_term___u2192_u2099_x2b___00__closed__8 = (const lean_object*)&lp_mathlib_term___u2192_u2099_x2b___00__closed__8_value;
static const lean_ctor_object lp_mathlib_term___u2192_u2099_x2b___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192_u2099_x2b___00__closed__3_value),((lean_object*)&lp_mathlib_term___u2192_u2099_x2b___00__closed__5_value),((lean_object*)&lp_mathlib_term___u2192_u2099_x2b___00__closed__8_value)}};
static const lean_object* lp_mathlib_term___u2192_u2099_x2b___00__closed__9 = (const lean_object*)&lp_mathlib_term___u2192_u2099_x2b___00__closed__9_value;
static const lean_ctor_object lp_mathlib_term___u2192_u2099_x2b___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192_u2099_x2b___00__closed__1_value),((lean_object*)(((size_t)(25) << 1) | 1)),((lean_object*)(((size_t)(26) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2192_u2099_x2b___00__closed__9_value)}};
static const lean_object* lp_mathlib_term___u2192_u2099_x2b___00__closed__10 = (const lean_object*)&lp_mathlib_term___u2192_u2099_x2b___00__closed__10_value;
LEAN_EXPORT const lean_object* lp_mathlib_term___u2192_u2099_x2b__ = (const lean_object*)&lp_mathlib_term___u2192_u2099_x2b___00__closed__10_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2b____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2b____1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2b____1___closed__0_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2b____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2b____1___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2b____1___closed__1_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2b____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2b____1___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2b____1___closed__2_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2b____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2b____1___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2b____1___closed__3_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2b____1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2b____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2b____1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2b____1___closed__4_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2b____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2b____1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2b____1___closed__4_value_aux_1),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2b____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2b____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2b____1___closed__4_value_aux_2),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2b____1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2b____1___closed__4 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2b____1___closed__4_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2b____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "AddHom"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2b____1___closed__5 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2b____1___closed__5_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2b____1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2b____1___closed__6;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2b____1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2b____1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(73, 182, 214, 26, 42, 159, 137, 40)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2b____1___closed__7 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2b____1___closed__7_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2b____1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2b____1___closed__7_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2b____1___closed__8 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2b____1___closed__8_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2b____1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2b____1___closed__7_value)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2b____1___closed__9 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2b____1___closed__9_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2b____1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2b____1___closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2b____1___closed__10 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2b____1___closed__10_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2b____1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2b____1___closed__8_value),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2b____1___closed__10_value)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2b____1___closed__11 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2b____1___closed__11_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2b____1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2b____1___closed__12 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2b____1___closed__12_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2b____1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2b____1___closed__12_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2b____1___closed__13 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2b____1___closed__13_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2b____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2b____1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______unexpand__AddHom__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______unexpand__AddHom__1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______unexpand__AddHom__1___closed__0_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______unexpand__AddHom__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______unexpand__AddHom__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______unexpand__AddHom__1___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______unexpand__AddHom__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______unexpand__AddHom__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______unexpand__AddHom__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_toAddHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_toAddHom___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_toAddHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_toAddHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_term___u2192_x2b___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 8, .m_data = "term_→+_"};
static const lean_object* lp_mathlib_term___u2192_x2b___00__closed__0 = (const lean_object*)&lp_mathlib_term___u2192_x2b___00__closed__0_value;
static const lean_ctor_object lp_mathlib_term___u2192_x2b___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2192_x2b___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(112, 234, 176, 203, 211, 122, 212, 209)}};
static const lean_object* lp_mathlib_term___u2192_x2b___00__closed__1 = (const lean_object*)&lp_mathlib_term___u2192_x2b___00__closed__1_value;
static const lean_string_object lp_mathlib_term___u2192_x2b___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 4, .m_data = " →+ "};
static const lean_object* lp_mathlib_term___u2192_x2b___00__closed__2 = (const lean_object*)&lp_mathlib_term___u2192_x2b___00__closed__2_value;
static const lean_ctor_object lp_mathlib_term___u2192_x2b___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192_x2b___00__closed__2_value)}};
static const lean_object* lp_mathlib_term___u2192_x2b___00__closed__3 = (const lean_object*)&lp_mathlib_term___u2192_x2b___00__closed__3_value;
static const lean_ctor_object lp_mathlib_term___u2192_x2b___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192_u2099_x2b___00__closed__3_value),((lean_object*)&lp_mathlib_term___u2192_x2b___00__closed__3_value),((lean_object*)&lp_mathlib_term___u2192_u2099_x2b___00__closed__8_value)}};
static const lean_object* lp_mathlib_term___u2192_x2b___00__closed__4 = (const lean_object*)&lp_mathlib_term___u2192_x2b___00__closed__4_value;
static const lean_ctor_object lp_mathlib_term___u2192_x2b___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192_x2b___00__closed__1_value),((lean_object*)(((size_t)(25) << 1) | 1)),((lean_object*)(((size_t)(26) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2192_x2b___00__closed__4_value)}};
static const lean_object* lp_mathlib_term___u2192_x2b___00__closed__5 = (const lean_object*)&lp_mathlib_term___u2192_x2b___00__closed__5_value;
LEAN_EXPORT const lean_object* lp_mathlib_term___u2192_x2b__ = (const lean_object*)&lp_mathlib_term___u2192_x2b___00__closed__5_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_x2b____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "AddMonoidHom"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_x2b____1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_x2b____1___closed__0_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_x2b____1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_x2b____1___closed__1;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_x2b____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_x2b____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(126, 52, 81, 78, 72, 34, 241, 210)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_x2b____1___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_x2b____1___closed__2_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_x2b____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_x2b____1___closed__2_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_x2b____1___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_x2b____1___closed__3_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_x2b____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_x2b____1___closed__2_value)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_x2b____1___closed__4 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_x2b____1___closed__4_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_x2b____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_x2b____1___closed__4_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_x2b____1___closed__5 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_x2b____1___closed__5_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_x2b____1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_x2b____1___closed__3_value),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_x2b____1___closed__5_value)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_x2b____1___closed__6 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_x2b____1___closed__6_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_x2b____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_x2b____1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______unexpand__AddMonoidHom__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______unexpand__AddMonoidHom__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OneHom_funLike___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_OneHom_funLike___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_OneHom_funLike___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_OneHom_funLike___closed__0 = (const lean_object*)&lp_mathlib_OneHom_funLike___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_OneHom_funLike(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OneHom_funLike___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_funLike(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_funLike___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LibraryNote_hom__simp__lemma__priority;
LEAN_EXPORT lean_object* lp_mathlib_OneHomClass_toOneHom___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OneHomClass_toOneHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OneHomClass_toOneHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZeroHomClass_toZeroHom___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZeroHomClass_toZeroHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZeroHomClass_toZeroHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCOneHomOfOneHomClass___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCOneHomOfOneHomClass(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCZeroHomOfZeroHomClass___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCZeroHomOfZeroHomClass(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_term___u2192_u2099_x2a___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 9, .m_data = "term_→ₙ*_"};
static const lean_object* lp_mathlib_term___u2192_u2099_x2a___00__closed__0 = (const lean_object*)&lp_mathlib_term___u2192_u2099_x2a___00__closed__0_value;
static const lean_ctor_object lp_mathlib_term___u2192_u2099_x2a___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2192_u2099_x2a___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(99, 91, 40, 159, 31, 36, 139, 87)}};
static const lean_object* lp_mathlib_term___u2192_u2099_x2a___00__closed__1 = (const lean_object*)&lp_mathlib_term___u2192_u2099_x2a___00__closed__1_value;
static const lean_string_object lp_mathlib_term___u2192_u2099_x2a___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 5, .m_data = " →ₙ* "};
static const lean_object* lp_mathlib_term___u2192_u2099_x2a___00__closed__2 = (const lean_object*)&lp_mathlib_term___u2192_u2099_x2a___00__closed__2_value;
static const lean_ctor_object lp_mathlib_term___u2192_u2099_x2a___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192_u2099_x2a___00__closed__2_value)}};
static const lean_object* lp_mathlib_term___u2192_u2099_x2a___00__closed__3 = (const lean_object*)&lp_mathlib_term___u2192_u2099_x2a___00__closed__3_value;
static const lean_ctor_object lp_mathlib_term___u2192_u2099_x2a___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192_u2099_x2b___00__closed__3_value),((lean_object*)&lp_mathlib_term___u2192_u2099_x2a___00__closed__3_value),((lean_object*)&lp_mathlib_term___u2192_u2099_x2b___00__closed__8_value)}};
static const lean_object* lp_mathlib_term___u2192_u2099_x2a___00__closed__4 = (const lean_object*)&lp_mathlib_term___u2192_u2099_x2a___00__closed__4_value;
static const lean_ctor_object lp_mathlib_term___u2192_u2099_x2a___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192_u2099_x2a___00__closed__1_value),((lean_object*)(((size_t)(25) << 1) | 1)),((lean_object*)(((size_t)(26) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2192_u2099_x2a___00__closed__4_value)}};
static const lean_object* lp_mathlib_term___u2192_u2099_x2a___00__closed__5 = (const lean_object*)&lp_mathlib_term___u2192_u2099_x2a___00__closed__5_value;
LEAN_EXPORT const lean_object* lp_mathlib_term___u2192_u2099_x2a__ = (const lean_object*)&lp_mathlib_term___u2192_u2099_x2a___00__closed__5_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2a____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "MulHom"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2a____1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2a____1___closed__0_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2a____1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2a____1___closed__1;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2a____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2a____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(208, 135, 172, 106, 130, 119, 166, 3)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2a____1___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2a____1___closed__2_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2a____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2a____1___closed__2_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2a____1___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2a____1___closed__3_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2a____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2a____1___closed__2_value)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2a____1___closed__4 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2a____1___closed__4_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2a____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2a____1___closed__4_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2a____1___closed__5 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2a____1___closed__5_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2a____1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2a____1___closed__3_value),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2a____1___closed__5_value)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2a____1___closed__6 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2a____1___closed__6_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2a____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2a____1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______unexpand__MulHom__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______unexpand__MulHom__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulHom_funLike(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulHom_funLike___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddHom_funLike(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddHom_funLike___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulHomClass_toMulHom___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulHomClass_toMulHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulHomClass_toMulHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddHomClass_toAddHom___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddHomClass_toAddHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddHomClass_toAddHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCMulHomOfMulHomClass___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCMulHomOfMulHomClass(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCAddHomOfAddHomClass___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCAddHomOfAddHomClass(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_toMulHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_toMulHom___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_toMulHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_toMulHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_term___u2192_x2a___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 8, .m_data = "term_→*_"};
static const lean_object* lp_mathlib_term___u2192_x2a___00__closed__0 = (const lean_object*)&lp_mathlib_term___u2192_x2a___00__closed__0_value;
static const lean_ctor_object lp_mathlib_term___u2192_x2a___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2192_x2a___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(57, 102, 173, 105, 164, 115, 171, 169)}};
static const lean_object* lp_mathlib_term___u2192_x2a___00__closed__1 = (const lean_object*)&lp_mathlib_term___u2192_x2a___00__closed__1_value;
static const lean_string_object lp_mathlib_term___u2192_x2a___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 4, .m_data = " →* "};
static const lean_object* lp_mathlib_term___u2192_x2a___00__closed__2 = (const lean_object*)&lp_mathlib_term___u2192_x2a___00__closed__2_value;
static const lean_ctor_object lp_mathlib_term___u2192_x2a___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192_x2a___00__closed__2_value)}};
static const lean_object* lp_mathlib_term___u2192_x2a___00__closed__3 = (const lean_object*)&lp_mathlib_term___u2192_x2a___00__closed__3_value;
static const lean_ctor_object lp_mathlib_term___u2192_x2a___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192_u2099_x2b___00__closed__3_value),((lean_object*)&lp_mathlib_term___u2192_x2a___00__closed__3_value),((lean_object*)&lp_mathlib_term___u2192_u2099_x2b___00__closed__8_value)}};
static const lean_object* lp_mathlib_term___u2192_x2a___00__closed__4 = (const lean_object*)&lp_mathlib_term___u2192_x2a___00__closed__4_value;
static const lean_ctor_object lp_mathlib_term___u2192_x2a___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192_x2a___00__closed__1_value),((lean_object*)(((size_t)(25) << 1) | 1)),((lean_object*)(((size_t)(26) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2192_x2a___00__closed__4_value)}};
static const lean_object* lp_mathlib_term___u2192_x2a___00__closed__5 = (const lean_object*)&lp_mathlib_term___u2192_x2a___00__closed__5_value;
LEAN_EXPORT const lean_object* lp_mathlib_term___u2192_x2a__ = (const lean_object*)&lp_mathlib_term___u2192_x2a___00__closed__5_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_x2a____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "MonoidHom"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_x2a____1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_x2a____1___closed__0_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_x2a____1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_x2a____1___closed__1;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_x2a____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_x2a____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(42, 146, 241, 17, 119, 0, 235, 30)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_x2a____1___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_x2a____1___closed__2_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_x2a____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_x2a____1___closed__2_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_x2a____1___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_x2a____1___closed__3_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_x2a____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_x2a____1___closed__2_value)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_x2a____1___closed__4 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_x2a____1___closed__4_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_x2a____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_x2a____1___closed__4_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_x2a____1___closed__5 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_x2a____1___closed__5_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_x2a____1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_x2a____1___closed__3_value),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_x2a____1___closed__5_value)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_x2a____1___closed__6 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_x2a____1___closed__6_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_x2a____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_x2a____1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______unexpand__MonoidHom__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______unexpand__MonoidHom__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_instFunLike___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_MonoidHom_instFunLike___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_MonoidHom_instFunLike___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_MonoidHom_instFunLike___closed__0 = (const lean_object*)&lp_mathlib_MonoidHom_instFunLike___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_instFunLike(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_instFunLike___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_instFunLike(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_instFunLike___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHomClass_toMonoidHom___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHomClass_toMonoidHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHomClass_toMonoidHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHomClass_toAddMonoidHom___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHomClass_toAddMonoidHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHomClass_toAddMonoidHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCMonoidHomOfMonoidHomClass___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCMonoidHomOfMonoidHomClass(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCAddMonoidHomOfAddMonoidHomClass___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCAddMonoidHomOfAddMonoidHomClass(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_coeToOneHom(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_coeToOneHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_coeToZeroHom(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_coeToZeroHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_coeToMulHom___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_coeToMulHom(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_coeToAddHom___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_coeToAddHom(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_mk_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_mk_x27___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_mk_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_mk_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mk_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mk_x27___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mk_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mk_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OneHom_copy___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OneHom_copy___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OneHom_copy(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OneHom_copy___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_copy___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_copy___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_copy(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_copy___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulHom_copy___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulHom_copy___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulHom_copy(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulHom_copy___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddHom_copy___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddHom_copy___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddHom_copy(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddHom_copy___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_copy___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_copy___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_copy(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_copy___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_copy___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_copy___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_copy(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_copy___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OneHom_id___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OneHom_id___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_OneHom_id___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_OneHom_id___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_OneHom_id___closed__0 = (const lean_object*)&lp_mathlib_OneHom_id___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_OneHom_id(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OneHom_id___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_id(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_id___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulHom_id(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulHom_id___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddHom_id(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddHom_id___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_id(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_id___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_id(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_id___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OneHom_comp___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OneHom_comp___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OneHom_comp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OneHom_comp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_comp___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_comp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_comp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulHom_comp___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulHom_comp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulHom_comp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddHom_comp___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddHom_comp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddHom_comp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_comp___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_comp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_comp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_comp___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_comp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_comp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OneHom_inverse___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OneHom_inverse___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OneHom_inverse(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OneHom_inverse___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_inverse___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_inverse___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_inverse(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_inverse___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulHom_inverse___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulHom_inverse___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulHom_inverse(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulHom_inverse___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddHom_inverse___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddHom_inverse___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddHom_inverse(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddHom_inverse___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_inverse___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_inverse___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_inverse(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_inverse___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_inverse___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_inverse___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_inverse(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_inverse___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Monoid_End_instFunLike___aux__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Monoid_End_instFunLike___aux__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Monoid_End_instFunLike___aux__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Monoid_End_instFunLike___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Monoid_End_instFunLike(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instFunLike___aux__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instFunLike___aux__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instFunLike___aux__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instFunLike___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instFunLike(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Monoid_End_instOne(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Monoid_End_instOne___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instOne(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instOne___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Monoid_End_instMul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Monoid_End_instMul(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instMul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instMul(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Monoid_End_instMonoid___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Monoid_End_instMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Monoid_End_instMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instMonoid___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Monoid_End_instInhabited(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Monoid_End_instInhabited___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instInhabited(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instInhabited___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instOneOneHom___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instOneOneHom___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instOneOneHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instOneOneHom(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instOneOneHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instZeroZeroHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instZeroZeroHom(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instZeroZeroHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instOneMulHom___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instOneMulHom___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instOneMulHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instOneMulHom(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instOneMulHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instZeroAddHom___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instZeroAddHom___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instZeroAddHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instZeroAddHom(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instZeroAddHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instOneMonoidHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instOneMonoidHom(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instOneMonoidHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instZeroAddMonoidHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instZeroAddMonoidHom(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instZeroAddMonoidHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedOneHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedOneHom(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedOneHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedZeroHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedZeroHom(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedZeroHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedMulHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedMulHom(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedMulHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedAddHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedAddHom(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedAddHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedMonoidHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedMonoidHom(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedMonoidHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedAddMonoidHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedAddMonoidHom(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedAddMonoidHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2b____1___closed__6(void){
_start:
{
lean_object* v___x_36_; lean_object* v___x_37_; 
v___x_36_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2b____1___closed__5));
v___x_37_ = l_String_toRawSubstring_x27(v___x_36_);
return v___x_37_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2b____1(lean_object* v_x_54_, lean_object* v_a_55_, lean_object* v_a_56_){
_start:
{
lean_object* v___x_57_; uint8_t v___x_58_; 
v___x_57_ = ((lean_object*)(lp_mathlib_term___u2192_u2099_x2b___00__closed__1));
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
v___x_70_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2b____1___closed__4));
v___x_71_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2b____1___closed__6, &lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2b____1___closed__6_once, _init_lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2b____1___closed__6);
v___x_72_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2b____1___closed__7));
lean_inc(v_currMacroScope_62_);
lean_inc(v_quotContext_61_);
v___x_73_ = l_Lean_addMacroScope(v_quotContext_61_, v___x_72_, v_currMacroScope_62_);
v___x_74_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2b____1___closed__11));
lean_inc_n(v___x_69_, 2);
v___x_75_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_75_, 0, v___x_69_);
lean_ctor_set(v___x_75_, 1, v___x_71_);
lean_ctor_set(v___x_75_, 2, v___x_73_);
lean_ctor_set(v___x_75_, 3, v___x_74_);
v___x_76_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2b____1___closed__13));
v___x_77_ = l_Lean_Syntax_node2(v___x_69_, v___x_76_, v___x_65_, v___x_67_);
v___x_78_ = l_Lean_Syntax_node2(v___x_69_, v___x_70_, v___x_75_, v___x_77_);
v___x_79_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_79_, 0, v___x_78_);
lean_ctor_set(v___x_79_, 1, v_a_56_);
return v___x_79_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2b____1___boxed(lean_object* v_x_80_, lean_object* v_a_81_, lean_object* v_a_82_){
_start:
{
lean_object* v_res_83_; 
v_res_83_ = lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2b____1(v_x_80_, v_a_81_, v_a_82_);
lean_dec_ref(v_a_81_);
return v_res_83_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______unexpand__AddHom__1(lean_object* v_x_87_, lean_object* v_a_88_, lean_object* v_a_89_){
_start:
{
lean_object* v___x_90_; uint8_t v___x_91_; 
v___x_90_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2b____1___closed__4));
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
v___x_96_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______unexpand__AddHom__1___closed__1));
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
v___x_111_ = ((lean_object*)(lp_mathlib_term___u2192_u2099_x2b___00__closed__1));
v___x_112_ = ((lean_object*)(lp_mathlib_term___u2192_u2099_x2b___00__closed__4));
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
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______unexpand__AddHom__1___boxed(lean_object* v_x_116_, lean_object* v_a_117_, lean_object* v_a_118_){
_start:
{
lean_object* v_res_119_; 
v_res_119_ = lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______unexpand__AddHom__1(v_x_116_, v_a_117_, v_a_118_);
lean_dec(v_a_117_);
return v_res_119_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_toAddHom___redArg(lean_object* v_self_120_){
_start:
{
lean_inc(v_self_120_);
return v_self_120_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_toAddHom___redArg___boxed(lean_object* v_self_121_){
_start:
{
lean_object* v_res_122_; 
v_res_122_ = lp_mathlib_AddMonoidHom_toAddHom___redArg(v_self_121_);
lean_dec(v_self_121_);
return v_res_122_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_toAddHom(lean_object* v_M_123_, lean_object* v_N_124_, lean_object* v_inst_125_, lean_object* v_inst_126_, lean_object* v_self_127_){
_start:
{
lean_inc(v_self_127_);
return v_self_127_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_toAddHom___boxed(lean_object* v_M_128_, lean_object* v_N_129_, lean_object* v_inst_130_, lean_object* v_inst_131_, lean_object* v_self_132_){
_start:
{
lean_object* v_res_133_; 
v_res_133_ = lp_mathlib_AddMonoidHom_toAddHom(v_M_128_, v_N_129_, v_inst_130_, v_inst_131_, v_self_132_);
lean_dec(v_self_132_);
lean_dec_ref(v_inst_131_);
lean_dec_ref(v_inst_130_);
return v_res_133_;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_x2b____1___closed__1(void){
_start:
{
lean_object* v___x_151_; lean_object* v___x_152_; 
v___x_151_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_x2b____1___closed__0));
v___x_152_ = l_String_toRawSubstring_x27(v___x_151_);
return v___x_152_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_x2b____1(lean_object* v_x_166_, lean_object* v_a_167_, lean_object* v_a_168_){
_start:
{
lean_object* v___x_169_; uint8_t v___x_170_; 
v___x_169_ = ((lean_object*)(lp_mathlib_term___u2192_x2b___00__closed__1));
lean_inc(v_x_166_);
v___x_170_ = l_Lean_Syntax_isOfKind(v_x_166_, v___x_169_);
if (v___x_170_ == 0)
{
lean_object* v___x_171_; lean_object* v___x_172_; 
lean_dec(v_x_166_);
v___x_171_ = lean_box(1);
v___x_172_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_172_, 0, v___x_171_);
lean_ctor_set(v___x_172_, 1, v_a_168_);
return v___x_172_;
}
else
{
lean_object* v_quotContext_173_; lean_object* v_currMacroScope_174_; lean_object* v_ref_175_; lean_object* v___x_176_; lean_object* v___x_177_; lean_object* v___x_178_; lean_object* v___x_179_; uint8_t v___x_180_; lean_object* v___x_181_; lean_object* v___x_182_; lean_object* v___x_183_; lean_object* v___x_184_; lean_object* v___x_185_; lean_object* v___x_186_; lean_object* v___x_187_; lean_object* v___x_188_; lean_object* v___x_189_; lean_object* v___x_190_; lean_object* v___x_191_; 
v_quotContext_173_ = lean_ctor_get(v_a_167_, 1);
v_currMacroScope_174_ = lean_ctor_get(v_a_167_, 2);
v_ref_175_ = lean_ctor_get(v_a_167_, 5);
v___x_176_ = lean_unsigned_to_nat(0u);
v___x_177_ = l_Lean_Syntax_getArg(v_x_166_, v___x_176_);
v___x_178_ = lean_unsigned_to_nat(2u);
v___x_179_ = l_Lean_Syntax_getArg(v_x_166_, v___x_178_);
lean_dec(v_x_166_);
v___x_180_ = 0;
v___x_181_ = l_Lean_SourceInfo_fromRef(v_ref_175_, v___x_180_);
v___x_182_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2b____1___closed__4));
v___x_183_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_x2b____1___closed__1, &lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_x2b____1___closed__1_once, _init_lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_x2b____1___closed__1);
v___x_184_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_x2b____1___closed__2));
lean_inc(v_currMacroScope_174_);
lean_inc(v_quotContext_173_);
v___x_185_ = l_Lean_addMacroScope(v_quotContext_173_, v___x_184_, v_currMacroScope_174_);
v___x_186_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_x2b____1___closed__6));
lean_inc_n(v___x_181_, 2);
v___x_187_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_187_, 0, v___x_181_);
lean_ctor_set(v___x_187_, 1, v___x_183_);
lean_ctor_set(v___x_187_, 2, v___x_185_);
lean_ctor_set(v___x_187_, 3, v___x_186_);
v___x_188_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2b____1___closed__13));
v___x_189_ = l_Lean_Syntax_node2(v___x_181_, v___x_188_, v___x_177_, v___x_179_);
v___x_190_ = l_Lean_Syntax_node2(v___x_181_, v___x_182_, v___x_187_, v___x_189_);
v___x_191_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_191_, 0, v___x_190_);
lean_ctor_set(v___x_191_, 1, v_a_168_);
return v___x_191_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_x2b____1___boxed(lean_object* v_x_192_, lean_object* v_a_193_, lean_object* v_a_194_){
_start:
{
lean_object* v_res_195_; 
v_res_195_ = lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_x2b____1(v_x_192_, v_a_193_, v_a_194_);
lean_dec_ref(v_a_193_);
return v_res_195_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______unexpand__AddMonoidHom__1(lean_object* v_x_196_, lean_object* v_a_197_, lean_object* v_a_198_){
_start:
{
lean_object* v___x_199_; uint8_t v___x_200_; 
v___x_199_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2b____1___closed__4));
lean_inc(v_x_196_);
v___x_200_ = l_Lean_Syntax_isOfKind(v_x_196_, v___x_199_);
if (v___x_200_ == 0)
{
lean_object* v___x_201_; lean_object* v___x_202_; 
lean_dec(v_x_196_);
v___x_201_ = lean_box(0);
v___x_202_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_202_, 0, v___x_201_);
lean_ctor_set(v___x_202_, 1, v_a_198_);
return v___x_202_;
}
else
{
lean_object* v___x_203_; lean_object* v___x_204_; lean_object* v___x_205_; uint8_t v___x_206_; 
v___x_203_ = lean_unsigned_to_nat(0u);
v___x_204_ = l_Lean_Syntax_getArg(v_x_196_, v___x_203_);
v___x_205_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______unexpand__AddHom__1___closed__1));
lean_inc(v___x_204_);
v___x_206_ = l_Lean_Syntax_isOfKind(v___x_204_, v___x_205_);
if (v___x_206_ == 0)
{
lean_object* v___x_207_; lean_object* v___x_208_; 
lean_dec(v___x_204_);
lean_dec(v_x_196_);
v___x_207_ = lean_box(0);
v___x_208_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_208_, 0, v___x_207_);
lean_ctor_set(v___x_208_, 1, v_a_198_);
return v___x_208_;
}
else
{
lean_object* v___x_209_; lean_object* v___x_210_; lean_object* v___x_211_; uint8_t v___x_212_; 
v___x_209_ = lean_unsigned_to_nat(1u);
v___x_210_ = l_Lean_Syntax_getArg(v_x_196_, v___x_209_);
lean_dec(v_x_196_);
v___x_211_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_210_);
v___x_212_ = l_Lean_Syntax_matchesNull(v___x_210_, v___x_211_);
if (v___x_212_ == 0)
{
lean_object* v___x_213_; lean_object* v___x_214_; 
lean_dec(v___x_210_);
lean_dec(v___x_204_);
v___x_213_ = lean_box(0);
v___x_214_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_214_, 0, v___x_213_);
lean_ctor_set(v___x_214_, 1, v_a_198_);
return v___x_214_;
}
else
{
lean_object* v___x_215_; lean_object* v___x_216_; lean_object* v_ref_217_; uint8_t v___x_218_; lean_object* v___x_219_; lean_object* v___x_220_; lean_object* v___x_221_; lean_object* v___x_222_; lean_object* v___x_223_; lean_object* v___x_224_; 
v___x_215_ = l_Lean_Syntax_getArg(v___x_210_, v___x_203_);
v___x_216_ = l_Lean_Syntax_getArg(v___x_210_, v___x_209_);
lean_dec(v___x_210_);
v_ref_217_ = l_Lean_replaceRef(v___x_204_, v_a_197_);
lean_dec(v___x_204_);
v___x_218_ = 0;
v___x_219_ = l_Lean_SourceInfo_fromRef(v_ref_217_, v___x_218_);
lean_dec(v_ref_217_);
v___x_220_ = ((lean_object*)(lp_mathlib_term___u2192_x2b___00__closed__1));
v___x_221_ = ((lean_object*)(lp_mathlib_term___u2192_x2b___00__closed__2));
lean_inc(v___x_219_);
v___x_222_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_222_, 0, v___x_219_);
lean_ctor_set(v___x_222_, 1, v___x_221_);
v___x_223_ = l_Lean_Syntax_node3(v___x_219_, v___x_220_, v___x_215_, v___x_222_, v___x_216_);
v___x_224_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_224_, 0, v___x_223_);
lean_ctor_set(v___x_224_, 1, v_a_198_);
return v___x_224_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______unexpand__AddMonoidHom__1___boxed(lean_object* v_x_225_, lean_object* v_a_226_, lean_object* v_a_227_){
_start:
{
lean_object* v_res_228_; 
v_res_228_ = lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______unexpand__AddMonoidHom__1(v_x_225_, v_a_226_, v_a_227_);
lean_dec(v_a_226_);
return v_res_228_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OneHom_funLike___lam__0(lean_object* v_self_229_, lean_object* v___y_230_){
_start:
{
lean_object* v___x_231_; 
v___x_231_ = lean_apply_1(v_self_229_, v___y_230_);
return v___x_231_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OneHom_funLike(lean_object* v_M_233_, lean_object* v_N_234_, lean_object* v_inst_235_, lean_object* v_inst_236_){
_start:
{
lean_object* v___f_237_; 
v___f_237_ = ((lean_object*)(lp_mathlib_OneHom_funLike___closed__0));
return v___f_237_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OneHom_funLike___boxed(lean_object* v_M_238_, lean_object* v_N_239_, lean_object* v_inst_240_, lean_object* v_inst_241_){
_start:
{
lean_object* v_res_242_; 
v_res_242_ = lp_mathlib_OneHom_funLike(v_M_238_, v_N_239_, v_inst_240_, v_inst_241_);
lean_dec(v_inst_241_);
lean_dec(v_inst_240_);
return v_res_242_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_funLike(lean_object* v_M_243_, lean_object* v_N_244_, lean_object* v_inst_245_, lean_object* v_inst_246_){
_start:
{
lean_object* v___f_247_; 
v___f_247_ = ((lean_object*)(lp_mathlib_OneHom_funLike___closed__0));
return v___f_247_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_funLike___boxed(lean_object* v_M_248_, lean_object* v_N_249_, lean_object* v_inst_250_, lean_object* v_inst_251_){
_start:
{
lean_object* v_res_252_; 
v_res_252_ = lp_mathlib_ZeroHom_funLike(v_M_248_, v_N_249_, v_inst_250_, v_inst_251_);
lean_dec(v_inst_251_);
lean_dec(v_inst_250_);
return v_res_252_;
}
}
static lean_object* _init_lp_mathlib_LibraryNote_hom__simp__lemma__priority(void){
_start:
{
lean_object* v___x_253_; 
v___x_253_ = lean_box(0);
return v___x_253_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OneHomClass_toOneHom___redArg(lean_object* v_inst_254_, lean_object* v_f_255_){
_start:
{
lean_object* v___x_256_; 
v___x_256_ = lean_apply_1(v_inst_254_, v_f_255_);
return v___x_256_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OneHomClass_toOneHom(lean_object* v_M_257_, lean_object* v_N_258_, lean_object* v_F_259_, lean_object* v_inst_260_, lean_object* v_inst_261_, lean_object* v_inst_262_, lean_object* v_inst_263_, lean_object* v_f_264_){
_start:
{
lean_object* v___x_265_; 
v___x_265_ = lean_apply_1(v_inst_262_, v_f_264_);
return v___x_265_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OneHomClass_toOneHom___boxed(lean_object* v_M_266_, lean_object* v_N_267_, lean_object* v_F_268_, lean_object* v_inst_269_, lean_object* v_inst_270_, lean_object* v_inst_271_, lean_object* v_inst_272_, lean_object* v_f_273_){
_start:
{
lean_object* v_res_274_; 
v_res_274_ = lp_mathlib_OneHomClass_toOneHom(v_M_266_, v_N_267_, v_F_268_, v_inst_269_, v_inst_270_, v_inst_271_, v_inst_272_, v_f_273_);
lean_dec(v_inst_270_);
lean_dec(v_inst_269_);
return v_res_274_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZeroHomClass_toZeroHom___redArg(lean_object* v_inst_275_, lean_object* v_f_276_){
_start:
{
lean_object* v___x_277_; 
v___x_277_ = lean_apply_1(v_inst_275_, v_f_276_);
return v___x_277_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZeroHomClass_toZeroHom(lean_object* v_M_278_, lean_object* v_N_279_, lean_object* v_F_280_, lean_object* v_inst_281_, lean_object* v_inst_282_, lean_object* v_inst_283_, lean_object* v_inst_284_, lean_object* v_f_285_){
_start:
{
lean_object* v___x_286_; 
v___x_286_ = lean_apply_1(v_inst_283_, v_f_285_);
return v___x_286_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZeroHomClass_toZeroHom___boxed(lean_object* v_M_287_, lean_object* v_N_288_, lean_object* v_F_289_, lean_object* v_inst_290_, lean_object* v_inst_291_, lean_object* v_inst_292_, lean_object* v_inst_293_, lean_object* v_f_294_){
_start:
{
lean_object* v_res_295_; 
v_res_295_ = lp_mathlib_ZeroHomClass_toZeroHom(v_M_287_, v_N_288_, v_F_289_, v_inst_290_, v_inst_291_, v_inst_292_, v_inst_293_, v_f_294_);
lean_dec(v_inst_291_);
lean_dec(v_inst_290_);
return v_res_295_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCOneHomOfOneHomClass___redArg(lean_object* v_inst_296_, lean_object* v_inst_297_, lean_object* v_inst_298_){
_start:
{
lean_object* v___x_299_; 
v___x_299_ = lean_alloc_closure((void*)(lp_mathlib_OneHomClass_toOneHom___boxed), 8, 7);
lean_closure_set(v___x_299_, 0, lean_box(0));
lean_closure_set(v___x_299_, 1, lean_box(0));
lean_closure_set(v___x_299_, 2, lean_box(0));
lean_closure_set(v___x_299_, 3, v_inst_296_);
lean_closure_set(v___x_299_, 4, v_inst_297_);
lean_closure_set(v___x_299_, 5, v_inst_298_);
lean_closure_set(v___x_299_, 6, lean_box(0));
return v___x_299_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCOneHomOfOneHomClass(lean_object* v_M_300_, lean_object* v_N_301_, lean_object* v_F_302_, lean_object* v_inst_303_, lean_object* v_inst_304_, lean_object* v_inst_305_, lean_object* v_inst_306_){
_start:
{
lean_object* v___x_307_; 
v___x_307_ = lean_alloc_closure((void*)(lp_mathlib_OneHomClass_toOneHom___boxed), 8, 7);
lean_closure_set(v___x_307_, 0, lean_box(0));
lean_closure_set(v___x_307_, 1, lean_box(0));
lean_closure_set(v___x_307_, 2, lean_box(0));
lean_closure_set(v___x_307_, 3, v_inst_303_);
lean_closure_set(v___x_307_, 4, v_inst_304_);
lean_closure_set(v___x_307_, 5, v_inst_305_);
lean_closure_set(v___x_307_, 6, lean_box(0));
return v___x_307_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCZeroHomOfZeroHomClass___redArg(lean_object* v_inst_308_, lean_object* v_inst_309_, lean_object* v_inst_310_){
_start:
{
lean_object* v___x_311_; 
v___x_311_ = lean_alloc_closure((void*)(lp_mathlib_ZeroHomClass_toZeroHom___boxed), 8, 7);
lean_closure_set(v___x_311_, 0, lean_box(0));
lean_closure_set(v___x_311_, 1, lean_box(0));
lean_closure_set(v___x_311_, 2, lean_box(0));
lean_closure_set(v___x_311_, 3, v_inst_308_);
lean_closure_set(v___x_311_, 4, v_inst_309_);
lean_closure_set(v___x_311_, 5, v_inst_310_);
lean_closure_set(v___x_311_, 6, lean_box(0));
return v___x_311_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCZeroHomOfZeroHomClass(lean_object* v_M_312_, lean_object* v_N_313_, lean_object* v_F_314_, lean_object* v_inst_315_, lean_object* v_inst_316_, lean_object* v_inst_317_, lean_object* v_inst_318_){
_start:
{
lean_object* v___x_319_; 
v___x_319_ = lean_alloc_closure((void*)(lp_mathlib_ZeroHomClass_toZeroHom___boxed), 8, 7);
lean_closure_set(v___x_319_, 0, lean_box(0));
lean_closure_set(v___x_319_, 1, lean_box(0));
lean_closure_set(v___x_319_, 2, lean_box(0));
lean_closure_set(v___x_319_, 3, v_inst_315_);
lean_closure_set(v___x_319_, 4, v_inst_316_);
lean_closure_set(v___x_319_, 5, v_inst_317_);
lean_closure_set(v___x_319_, 6, lean_box(0));
return v___x_319_;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2a____1___closed__1(void){
_start:
{
lean_object* v___x_337_; lean_object* v___x_338_; 
v___x_337_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2a____1___closed__0));
v___x_338_ = l_String_toRawSubstring_x27(v___x_337_);
return v___x_338_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2a____1(lean_object* v_x_352_, lean_object* v_a_353_, lean_object* v_a_354_){
_start:
{
lean_object* v___x_355_; uint8_t v___x_356_; 
v___x_355_ = ((lean_object*)(lp_mathlib_term___u2192_u2099_x2a___00__closed__1));
lean_inc(v_x_352_);
v___x_356_ = l_Lean_Syntax_isOfKind(v_x_352_, v___x_355_);
if (v___x_356_ == 0)
{
lean_object* v___x_357_; lean_object* v___x_358_; 
lean_dec(v_x_352_);
v___x_357_ = lean_box(1);
v___x_358_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_358_, 0, v___x_357_);
lean_ctor_set(v___x_358_, 1, v_a_354_);
return v___x_358_;
}
else
{
lean_object* v_quotContext_359_; lean_object* v_currMacroScope_360_; lean_object* v_ref_361_; lean_object* v___x_362_; lean_object* v___x_363_; lean_object* v___x_364_; lean_object* v___x_365_; uint8_t v___x_366_; lean_object* v___x_367_; lean_object* v___x_368_; lean_object* v___x_369_; lean_object* v___x_370_; lean_object* v___x_371_; lean_object* v___x_372_; lean_object* v___x_373_; lean_object* v___x_374_; lean_object* v___x_375_; lean_object* v___x_376_; lean_object* v___x_377_; 
v_quotContext_359_ = lean_ctor_get(v_a_353_, 1);
v_currMacroScope_360_ = lean_ctor_get(v_a_353_, 2);
v_ref_361_ = lean_ctor_get(v_a_353_, 5);
v___x_362_ = lean_unsigned_to_nat(0u);
v___x_363_ = l_Lean_Syntax_getArg(v_x_352_, v___x_362_);
v___x_364_ = lean_unsigned_to_nat(2u);
v___x_365_ = l_Lean_Syntax_getArg(v_x_352_, v___x_364_);
lean_dec(v_x_352_);
v___x_366_ = 0;
v___x_367_ = l_Lean_SourceInfo_fromRef(v_ref_361_, v___x_366_);
v___x_368_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2b____1___closed__4));
v___x_369_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2a____1___closed__1, &lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2a____1___closed__1_once, _init_lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2a____1___closed__1);
v___x_370_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2a____1___closed__2));
lean_inc(v_currMacroScope_360_);
lean_inc(v_quotContext_359_);
v___x_371_ = l_Lean_addMacroScope(v_quotContext_359_, v___x_370_, v_currMacroScope_360_);
v___x_372_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2a____1___closed__6));
lean_inc_n(v___x_367_, 2);
v___x_373_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_373_, 0, v___x_367_);
lean_ctor_set(v___x_373_, 1, v___x_369_);
lean_ctor_set(v___x_373_, 2, v___x_371_);
lean_ctor_set(v___x_373_, 3, v___x_372_);
v___x_374_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2b____1___closed__13));
v___x_375_ = l_Lean_Syntax_node2(v___x_367_, v___x_374_, v___x_363_, v___x_365_);
v___x_376_ = l_Lean_Syntax_node2(v___x_367_, v___x_368_, v___x_373_, v___x_375_);
v___x_377_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_377_, 0, v___x_376_);
lean_ctor_set(v___x_377_, 1, v_a_354_);
return v___x_377_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2a____1___boxed(lean_object* v_x_378_, lean_object* v_a_379_, lean_object* v_a_380_){
_start:
{
lean_object* v_res_381_; 
v_res_381_ = lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2a____1(v_x_378_, v_a_379_, v_a_380_);
lean_dec_ref(v_a_379_);
return v_res_381_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______unexpand__MulHom__1(lean_object* v_x_382_, lean_object* v_a_383_, lean_object* v_a_384_){
_start:
{
lean_object* v___x_385_; uint8_t v___x_386_; 
v___x_385_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2b____1___closed__4));
lean_inc(v_x_382_);
v___x_386_ = l_Lean_Syntax_isOfKind(v_x_382_, v___x_385_);
if (v___x_386_ == 0)
{
lean_object* v___x_387_; lean_object* v___x_388_; 
lean_dec(v_x_382_);
v___x_387_ = lean_box(0);
v___x_388_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_388_, 0, v___x_387_);
lean_ctor_set(v___x_388_, 1, v_a_384_);
return v___x_388_;
}
else
{
lean_object* v___x_389_; lean_object* v___x_390_; lean_object* v___x_391_; uint8_t v___x_392_; 
v___x_389_ = lean_unsigned_to_nat(0u);
v___x_390_ = l_Lean_Syntax_getArg(v_x_382_, v___x_389_);
v___x_391_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______unexpand__AddHom__1___closed__1));
lean_inc(v___x_390_);
v___x_392_ = l_Lean_Syntax_isOfKind(v___x_390_, v___x_391_);
if (v___x_392_ == 0)
{
lean_object* v___x_393_; lean_object* v___x_394_; 
lean_dec(v___x_390_);
lean_dec(v_x_382_);
v___x_393_ = lean_box(0);
v___x_394_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_394_, 0, v___x_393_);
lean_ctor_set(v___x_394_, 1, v_a_384_);
return v___x_394_;
}
else
{
lean_object* v___x_395_; lean_object* v___x_396_; lean_object* v___x_397_; uint8_t v___x_398_; 
v___x_395_ = lean_unsigned_to_nat(1u);
v___x_396_ = l_Lean_Syntax_getArg(v_x_382_, v___x_395_);
lean_dec(v_x_382_);
v___x_397_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_396_);
v___x_398_ = l_Lean_Syntax_matchesNull(v___x_396_, v___x_397_);
if (v___x_398_ == 0)
{
lean_object* v___x_399_; lean_object* v___x_400_; 
lean_dec(v___x_396_);
lean_dec(v___x_390_);
v___x_399_ = lean_box(0);
v___x_400_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_400_, 0, v___x_399_);
lean_ctor_set(v___x_400_, 1, v_a_384_);
return v___x_400_;
}
else
{
lean_object* v___x_401_; lean_object* v___x_402_; lean_object* v_ref_403_; uint8_t v___x_404_; lean_object* v___x_405_; lean_object* v___x_406_; lean_object* v___x_407_; lean_object* v___x_408_; lean_object* v___x_409_; lean_object* v___x_410_; 
v___x_401_ = l_Lean_Syntax_getArg(v___x_396_, v___x_389_);
v___x_402_ = l_Lean_Syntax_getArg(v___x_396_, v___x_395_);
lean_dec(v___x_396_);
v_ref_403_ = l_Lean_replaceRef(v___x_390_, v_a_383_);
lean_dec(v___x_390_);
v___x_404_ = 0;
v___x_405_ = l_Lean_SourceInfo_fromRef(v_ref_403_, v___x_404_);
lean_dec(v_ref_403_);
v___x_406_ = ((lean_object*)(lp_mathlib_term___u2192_u2099_x2a___00__closed__1));
v___x_407_ = ((lean_object*)(lp_mathlib_term___u2192_u2099_x2a___00__closed__2));
lean_inc(v___x_405_);
v___x_408_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_408_, 0, v___x_405_);
lean_ctor_set(v___x_408_, 1, v___x_407_);
v___x_409_ = l_Lean_Syntax_node3(v___x_405_, v___x_406_, v___x_401_, v___x_408_, v___x_402_);
v___x_410_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_410_, 0, v___x_409_);
lean_ctor_set(v___x_410_, 1, v_a_384_);
return v___x_410_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______unexpand__MulHom__1___boxed(lean_object* v_x_411_, lean_object* v_a_412_, lean_object* v_a_413_){
_start:
{
lean_object* v_res_414_; 
v_res_414_ = lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______unexpand__MulHom__1(v_x_411_, v_a_412_, v_a_413_);
lean_dec(v_a_412_);
return v_res_414_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulHom_funLike(lean_object* v_M_415_, lean_object* v_N_416_, lean_object* v_inst_417_, lean_object* v_inst_418_){
_start:
{
lean_object* v___f_419_; 
v___f_419_ = ((lean_object*)(lp_mathlib_OneHom_funLike___closed__0));
return v___f_419_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulHom_funLike___boxed(lean_object* v_M_420_, lean_object* v_N_421_, lean_object* v_inst_422_, lean_object* v_inst_423_){
_start:
{
lean_object* v_res_424_; 
v_res_424_ = lp_mathlib_MulHom_funLike(v_M_420_, v_N_421_, v_inst_422_, v_inst_423_);
lean_dec(v_inst_423_);
lean_dec(v_inst_422_);
return v_res_424_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddHom_funLike(lean_object* v_M_425_, lean_object* v_N_426_, lean_object* v_inst_427_, lean_object* v_inst_428_){
_start:
{
lean_object* v___f_429_; 
v___f_429_ = ((lean_object*)(lp_mathlib_OneHom_funLike___closed__0));
return v___f_429_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddHom_funLike___boxed(lean_object* v_M_430_, lean_object* v_N_431_, lean_object* v_inst_432_, lean_object* v_inst_433_){
_start:
{
lean_object* v_res_434_; 
v_res_434_ = lp_mathlib_AddHom_funLike(v_M_430_, v_N_431_, v_inst_432_, v_inst_433_);
lean_dec(v_inst_433_);
lean_dec(v_inst_432_);
return v_res_434_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulHomClass_toMulHom___redArg(lean_object* v_inst_435_, lean_object* v_f_436_){
_start:
{
lean_object* v___x_437_; 
v___x_437_ = lean_apply_1(v_inst_435_, v_f_436_);
return v___x_437_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulHomClass_toMulHom(lean_object* v_M_438_, lean_object* v_N_439_, lean_object* v_F_440_, lean_object* v_inst_441_, lean_object* v_inst_442_, lean_object* v_inst_443_, lean_object* v_inst_444_, lean_object* v_f_445_){
_start:
{
lean_object* v___x_446_; 
v___x_446_ = lean_apply_1(v_inst_443_, v_f_445_);
return v___x_446_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulHomClass_toMulHom___boxed(lean_object* v_M_447_, lean_object* v_N_448_, lean_object* v_F_449_, lean_object* v_inst_450_, lean_object* v_inst_451_, lean_object* v_inst_452_, lean_object* v_inst_453_, lean_object* v_f_454_){
_start:
{
lean_object* v_res_455_; 
v_res_455_ = lp_mathlib_MulHomClass_toMulHom(v_M_447_, v_N_448_, v_F_449_, v_inst_450_, v_inst_451_, v_inst_452_, v_inst_453_, v_f_454_);
lean_dec(v_inst_451_);
lean_dec(v_inst_450_);
return v_res_455_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddHomClass_toAddHom___redArg(lean_object* v_inst_456_, lean_object* v_f_457_){
_start:
{
lean_object* v___x_458_; 
v___x_458_ = lean_apply_1(v_inst_456_, v_f_457_);
return v___x_458_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddHomClass_toAddHom(lean_object* v_M_459_, lean_object* v_N_460_, lean_object* v_F_461_, lean_object* v_inst_462_, lean_object* v_inst_463_, lean_object* v_inst_464_, lean_object* v_inst_465_, lean_object* v_f_466_){
_start:
{
lean_object* v___x_467_; 
v___x_467_ = lean_apply_1(v_inst_464_, v_f_466_);
return v___x_467_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddHomClass_toAddHom___boxed(lean_object* v_M_468_, lean_object* v_N_469_, lean_object* v_F_470_, lean_object* v_inst_471_, lean_object* v_inst_472_, lean_object* v_inst_473_, lean_object* v_inst_474_, lean_object* v_f_475_){
_start:
{
lean_object* v_res_476_; 
v_res_476_ = lp_mathlib_AddHomClass_toAddHom(v_M_468_, v_N_469_, v_F_470_, v_inst_471_, v_inst_472_, v_inst_473_, v_inst_474_, v_f_475_);
lean_dec(v_inst_472_);
lean_dec(v_inst_471_);
return v_res_476_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCMulHomOfMulHomClass___redArg(lean_object* v_inst_477_, lean_object* v_inst_478_, lean_object* v_inst_479_){
_start:
{
lean_object* v___x_480_; 
v___x_480_ = lean_alloc_closure((void*)(lp_mathlib_MulHomClass_toMulHom___boxed), 8, 7);
lean_closure_set(v___x_480_, 0, lean_box(0));
lean_closure_set(v___x_480_, 1, lean_box(0));
lean_closure_set(v___x_480_, 2, lean_box(0));
lean_closure_set(v___x_480_, 3, v_inst_477_);
lean_closure_set(v___x_480_, 4, v_inst_478_);
lean_closure_set(v___x_480_, 5, v_inst_479_);
lean_closure_set(v___x_480_, 6, lean_box(0));
return v___x_480_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCMulHomOfMulHomClass(lean_object* v_M_481_, lean_object* v_N_482_, lean_object* v_F_483_, lean_object* v_inst_484_, lean_object* v_inst_485_, lean_object* v_inst_486_, lean_object* v_inst_487_){
_start:
{
lean_object* v___x_488_; 
v___x_488_ = lean_alloc_closure((void*)(lp_mathlib_MulHomClass_toMulHom___boxed), 8, 7);
lean_closure_set(v___x_488_, 0, lean_box(0));
lean_closure_set(v___x_488_, 1, lean_box(0));
lean_closure_set(v___x_488_, 2, lean_box(0));
lean_closure_set(v___x_488_, 3, v_inst_484_);
lean_closure_set(v___x_488_, 4, v_inst_485_);
lean_closure_set(v___x_488_, 5, v_inst_486_);
lean_closure_set(v___x_488_, 6, lean_box(0));
return v___x_488_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCAddHomOfAddHomClass___redArg(lean_object* v_inst_489_, lean_object* v_inst_490_, lean_object* v_inst_491_){
_start:
{
lean_object* v___x_492_; 
v___x_492_ = lean_alloc_closure((void*)(lp_mathlib_AddHomClass_toAddHom___boxed), 8, 7);
lean_closure_set(v___x_492_, 0, lean_box(0));
lean_closure_set(v___x_492_, 1, lean_box(0));
lean_closure_set(v___x_492_, 2, lean_box(0));
lean_closure_set(v___x_492_, 3, v_inst_489_);
lean_closure_set(v___x_492_, 4, v_inst_490_);
lean_closure_set(v___x_492_, 5, v_inst_491_);
lean_closure_set(v___x_492_, 6, lean_box(0));
return v___x_492_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCAddHomOfAddHomClass(lean_object* v_M_493_, lean_object* v_N_494_, lean_object* v_F_495_, lean_object* v_inst_496_, lean_object* v_inst_497_, lean_object* v_inst_498_, lean_object* v_inst_499_){
_start:
{
lean_object* v___x_500_; 
v___x_500_ = lean_alloc_closure((void*)(lp_mathlib_AddHomClass_toAddHom___boxed), 8, 7);
lean_closure_set(v___x_500_, 0, lean_box(0));
lean_closure_set(v___x_500_, 1, lean_box(0));
lean_closure_set(v___x_500_, 2, lean_box(0));
lean_closure_set(v___x_500_, 3, v_inst_496_);
lean_closure_set(v___x_500_, 4, v_inst_497_);
lean_closure_set(v___x_500_, 5, v_inst_498_);
lean_closure_set(v___x_500_, 6, lean_box(0));
return v___x_500_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_toMulHom___redArg(lean_object* v_self_501_){
_start:
{
lean_inc(v_self_501_);
return v_self_501_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_toMulHom___redArg___boxed(lean_object* v_self_502_){
_start:
{
lean_object* v_res_503_; 
v_res_503_ = lp_mathlib_MonoidHom_toMulHom___redArg(v_self_502_);
lean_dec(v_self_502_);
return v_res_503_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_toMulHom(lean_object* v_M_504_, lean_object* v_N_505_, lean_object* v_inst_506_, lean_object* v_inst_507_, lean_object* v_self_508_){
_start:
{
lean_inc(v_self_508_);
return v_self_508_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_toMulHom___boxed(lean_object* v_M_509_, lean_object* v_N_510_, lean_object* v_inst_511_, lean_object* v_inst_512_, lean_object* v_self_513_){
_start:
{
lean_object* v_res_514_; 
v_res_514_ = lp_mathlib_MonoidHom_toMulHom(v_M_509_, v_N_510_, v_inst_511_, v_inst_512_, v_self_513_);
lean_dec(v_self_513_);
lean_dec_ref(v_inst_512_);
lean_dec_ref(v_inst_511_);
return v_res_514_;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_x2a____1___closed__1(void){
_start:
{
lean_object* v___x_532_; lean_object* v___x_533_; 
v___x_532_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_x2a____1___closed__0));
v___x_533_ = l_String_toRawSubstring_x27(v___x_532_);
return v___x_533_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_x2a____1(lean_object* v_x_547_, lean_object* v_a_548_, lean_object* v_a_549_){
_start:
{
lean_object* v___x_550_; uint8_t v___x_551_; 
v___x_550_ = ((lean_object*)(lp_mathlib_term___u2192_x2a___00__closed__1));
lean_inc(v_x_547_);
v___x_551_ = l_Lean_Syntax_isOfKind(v_x_547_, v___x_550_);
if (v___x_551_ == 0)
{
lean_object* v___x_552_; lean_object* v___x_553_; 
lean_dec(v_x_547_);
v___x_552_ = lean_box(1);
v___x_553_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_553_, 0, v___x_552_);
lean_ctor_set(v___x_553_, 1, v_a_549_);
return v___x_553_;
}
else
{
lean_object* v_quotContext_554_; lean_object* v_currMacroScope_555_; lean_object* v_ref_556_; lean_object* v___x_557_; lean_object* v___x_558_; lean_object* v___x_559_; lean_object* v___x_560_; uint8_t v___x_561_; lean_object* v___x_562_; lean_object* v___x_563_; lean_object* v___x_564_; lean_object* v___x_565_; lean_object* v___x_566_; lean_object* v___x_567_; lean_object* v___x_568_; lean_object* v___x_569_; lean_object* v___x_570_; lean_object* v___x_571_; lean_object* v___x_572_; 
v_quotContext_554_ = lean_ctor_get(v_a_548_, 1);
v_currMacroScope_555_ = lean_ctor_get(v_a_548_, 2);
v_ref_556_ = lean_ctor_get(v_a_548_, 5);
v___x_557_ = lean_unsigned_to_nat(0u);
v___x_558_ = l_Lean_Syntax_getArg(v_x_547_, v___x_557_);
v___x_559_ = lean_unsigned_to_nat(2u);
v___x_560_ = l_Lean_Syntax_getArg(v_x_547_, v___x_559_);
lean_dec(v_x_547_);
v___x_561_ = 0;
v___x_562_ = l_Lean_SourceInfo_fromRef(v_ref_556_, v___x_561_);
v___x_563_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2b____1___closed__4));
v___x_564_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_x2a____1___closed__1, &lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_x2a____1___closed__1_once, _init_lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_x2a____1___closed__1);
v___x_565_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_x2a____1___closed__2));
lean_inc(v_currMacroScope_555_);
lean_inc(v_quotContext_554_);
v___x_566_ = l_Lean_addMacroScope(v_quotContext_554_, v___x_565_, v_currMacroScope_555_);
v___x_567_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_x2a____1___closed__6));
lean_inc_n(v___x_562_, 2);
v___x_568_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_568_, 0, v___x_562_);
lean_ctor_set(v___x_568_, 1, v___x_564_);
lean_ctor_set(v___x_568_, 2, v___x_566_);
lean_ctor_set(v___x_568_, 3, v___x_567_);
v___x_569_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2b____1___closed__13));
v___x_570_ = l_Lean_Syntax_node2(v___x_562_, v___x_569_, v___x_558_, v___x_560_);
v___x_571_ = l_Lean_Syntax_node2(v___x_562_, v___x_563_, v___x_568_, v___x_570_);
v___x_572_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_572_, 0, v___x_571_);
lean_ctor_set(v___x_572_, 1, v_a_549_);
return v___x_572_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_x2a____1___boxed(lean_object* v_x_573_, lean_object* v_a_574_, lean_object* v_a_575_){
_start:
{
lean_object* v_res_576_; 
v_res_576_ = lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_x2a____1(v_x_573_, v_a_574_, v_a_575_);
lean_dec_ref(v_a_574_);
return v_res_576_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______unexpand__MonoidHom__1(lean_object* v_x_577_, lean_object* v_a_578_, lean_object* v_a_579_){
_start:
{
lean_object* v___x_580_; uint8_t v___x_581_; 
v___x_580_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______macroRules__term___u2192_u2099_x2b____1___closed__4));
lean_inc(v_x_577_);
v___x_581_ = l_Lean_Syntax_isOfKind(v_x_577_, v___x_580_);
if (v___x_581_ == 0)
{
lean_object* v___x_582_; lean_object* v___x_583_; 
lean_dec(v_x_577_);
v___x_582_ = lean_box(0);
v___x_583_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_583_, 0, v___x_582_);
lean_ctor_set(v___x_583_, 1, v_a_579_);
return v___x_583_;
}
else
{
lean_object* v___x_584_; lean_object* v___x_585_; lean_object* v___x_586_; uint8_t v___x_587_; 
v___x_584_ = lean_unsigned_to_nat(0u);
v___x_585_ = l_Lean_Syntax_getArg(v_x_577_, v___x_584_);
v___x_586_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______unexpand__AddHom__1___closed__1));
lean_inc(v___x_585_);
v___x_587_ = l_Lean_Syntax_isOfKind(v___x_585_, v___x_586_);
if (v___x_587_ == 0)
{
lean_object* v___x_588_; lean_object* v___x_589_; 
lean_dec(v___x_585_);
lean_dec(v_x_577_);
v___x_588_ = lean_box(0);
v___x_589_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_589_, 0, v___x_588_);
lean_ctor_set(v___x_589_, 1, v_a_579_);
return v___x_589_;
}
else
{
lean_object* v___x_590_; lean_object* v___x_591_; lean_object* v___x_592_; uint8_t v___x_593_; 
v___x_590_ = lean_unsigned_to_nat(1u);
v___x_591_ = l_Lean_Syntax_getArg(v_x_577_, v___x_590_);
lean_dec(v_x_577_);
v___x_592_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_591_);
v___x_593_ = l_Lean_Syntax_matchesNull(v___x_591_, v___x_592_);
if (v___x_593_ == 0)
{
lean_object* v___x_594_; lean_object* v___x_595_; 
lean_dec(v___x_591_);
lean_dec(v___x_585_);
v___x_594_ = lean_box(0);
v___x_595_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_595_, 0, v___x_594_);
lean_ctor_set(v___x_595_, 1, v_a_579_);
return v___x_595_;
}
else
{
lean_object* v___x_596_; lean_object* v___x_597_; lean_object* v_ref_598_; uint8_t v___x_599_; lean_object* v___x_600_; lean_object* v___x_601_; lean_object* v___x_602_; lean_object* v___x_603_; lean_object* v___x_604_; lean_object* v___x_605_; 
v___x_596_ = l_Lean_Syntax_getArg(v___x_591_, v___x_584_);
v___x_597_ = l_Lean_Syntax_getArg(v___x_591_, v___x_590_);
lean_dec(v___x_591_);
v_ref_598_ = l_Lean_replaceRef(v___x_585_, v_a_578_);
lean_dec(v___x_585_);
v___x_599_ = 0;
v___x_600_ = l_Lean_SourceInfo_fromRef(v_ref_598_, v___x_599_);
lean_dec(v_ref_598_);
v___x_601_ = ((lean_object*)(lp_mathlib_term___u2192_x2a___00__closed__1));
v___x_602_ = ((lean_object*)(lp_mathlib_term___u2192_x2a___00__closed__2));
lean_inc(v___x_600_);
v___x_603_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_603_, 0, v___x_600_);
lean_ctor_set(v___x_603_, 1, v___x_602_);
v___x_604_ = l_Lean_Syntax_node3(v___x_600_, v___x_601_, v___x_596_, v___x_603_, v___x_597_);
v___x_605_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_605_, 0, v___x_604_);
lean_ctor_set(v___x_605_, 1, v_a_579_);
return v___x_605_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______unexpand__MonoidHom__1___boxed(lean_object* v_x_606_, lean_object* v_a_607_, lean_object* v_a_608_){
_start:
{
lean_object* v_res_609_; 
v_res_609_ = lp_mathlib___aux__Mathlib__Algebra__Group__Hom__Defs______unexpand__MonoidHom__1(v_x_606_, v_a_607_, v_a_608_);
lean_dec(v_a_607_);
return v_res_609_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_instFunLike___lam__0(lean_object* v_f_610_, lean_object* v___y_611_){
_start:
{
lean_object* v___x_612_; 
v___x_612_ = lean_apply_1(v_f_610_, v___y_611_);
return v___x_612_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_instFunLike(lean_object* v_M_614_, lean_object* v_N_615_, lean_object* v_inst_616_, lean_object* v_inst_617_){
_start:
{
lean_object* v___f_618_; 
v___f_618_ = ((lean_object*)(lp_mathlib_MonoidHom_instFunLike___closed__0));
return v___f_618_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_instFunLike___boxed(lean_object* v_M_619_, lean_object* v_N_620_, lean_object* v_inst_621_, lean_object* v_inst_622_){
_start:
{
lean_object* v_res_623_; 
v_res_623_ = lp_mathlib_MonoidHom_instFunLike(v_M_619_, v_N_620_, v_inst_621_, v_inst_622_);
lean_dec_ref(v_inst_622_);
lean_dec_ref(v_inst_621_);
return v_res_623_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_instFunLike(lean_object* v_M_624_, lean_object* v_N_625_, lean_object* v_inst_626_, lean_object* v_inst_627_){
_start:
{
lean_object* v___f_628_; 
v___f_628_ = ((lean_object*)(lp_mathlib_MonoidHom_instFunLike___closed__0));
return v___f_628_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_instFunLike___boxed(lean_object* v_M_629_, lean_object* v_N_630_, lean_object* v_inst_631_, lean_object* v_inst_632_){
_start:
{
lean_object* v_res_633_; 
v_res_633_ = lp_mathlib_AddMonoidHom_instFunLike(v_M_629_, v_N_630_, v_inst_631_, v_inst_632_);
lean_dec_ref(v_inst_632_);
lean_dec_ref(v_inst_631_);
return v_res_633_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHomClass_toMonoidHom___redArg(lean_object* v_inst_634_, lean_object* v_f_635_){
_start:
{
lean_object* v___x_636_; 
v___x_636_ = lean_apply_1(v_inst_634_, v_f_635_);
return v___x_636_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHomClass_toMonoidHom(lean_object* v_M_637_, lean_object* v_N_638_, lean_object* v_F_639_, lean_object* v_inst_640_, lean_object* v_inst_641_, lean_object* v_inst_642_, lean_object* v_inst_643_, lean_object* v_f_644_){
_start:
{
lean_object* v___x_645_; 
v___x_645_ = lean_apply_1(v_inst_642_, v_f_644_);
return v___x_645_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHomClass_toMonoidHom___boxed(lean_object* v_M_646_, lean_object* v_N_647_, lean_object* v_F_648_, lean_object* v_inst_649_, lean_object* v_inst_650_, lean_object* v_inst_651_, lean_object* v_inst_652_, lean_object* v_f_653_){
_start:
{
lean_object* v_res_654_; 
v_res_654_ = lp_mathlib_MonoidHomClass_toMonoidHom(v_M_646_, v_N_647_, v_F_648_, v_inst_649_, v_inst_650_, v_inst_651_, v_inst_652_, v_f_653_);
lean_dec_ref(v_inst_650_);
lean_dec_ref(v_inst_649_);
return v_res_654_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHomClass_toAddMonoidHom___redArg(lean_object* v_inst_655_, lean_object* v_f_656_){
_start:
{
lean_object* v___x_657_; 
v___x_657_ = lean_apply_1(v_inst_655_, v_f_656_);
return v___x_657_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHomClass_toAddMonoidHom(lean_object* v_M_658_, lean_object* v_N_659_, lean_object* v_F_660_, lean_object* v_inst_661_, lean_object* v_inst_662_, lean_object* v_inst_663_, lean_object* v_inst_664_, lean_object* v_f_665_){
_start:
{
lean_object* v___x_666_; 
v___x_666_ = lean_apply_1(v_inst_663_, v_f_665_);
return v___x_666_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHomClass_toAddMonoidHom___boxed(lean_object* v_M_667_, lean_object* v_N_668_, lean_object* v_F_669_, lean_object* v_inst_670_, lean_object* v_inst_671_, lean_object* v_inst_672_, lean_object* v_inst_673_, lean_object* v_f_674_){
_start:
{
lean_object* v_res_675_; 
v_res_675_ = lp_mathlib_AddMonoidHomClass_toAddMonoidHom(v_M_667_, v_N_668_, v_F_669_, v_inst_670_, v_inst_671_, v_inst_672_, v_inst_673_, v_f_674_);
lean_dec_ref(v_inst_671_);
lean_dec_ref(v_inst_670_);
return v_res_675_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCMonoidHomOfMonoidHomClass___redArg(lean_object* v_inst_676_, lean_object* v_inst_677_, lean_object* v_inst_678_){
_start:
{
lean_object* v___x_679_; 
v___x_679_ = lean_alloc_closure((void*)(lp_mathlib_MonoidHomClass_toMonoidHom___boxed), 8, 7);
lean_closure_set(v___x_679_, 0, lean_box(0));
lean_closure_set(v___x_679_, 1, lean_box(0));
lean_closure_set(v___x_679_, 2, lean_box(0));
lean_closure_set(v___x_679_, 3, v_inst_676_);
lean_closure_set(v___x_679_, 4, v_inst_677_);
lean_closure_set(v___x_679_, 5, v_inst_678_);
lean_closure_set(v___x_679_, 6, lean_box(0));
return v___x_679_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCMonoidHomOfMonoidHomClass(lean_object* v_M_680_, lean_object* v_N_681_, lean_object* v_F_682_, lean_object* v_inst_683_, lean_object* v_inst_684_, lean_object* v_inst_685_, lean_object* v_inst_686_){
_start:
{
lean_object* v___x_687_; 
v___x_687_ = lean_alloc_closure((void*)(lp_mathlib_MonoidHomClass_toMonoidHom___boxed), 8, 7);
lean_closure_set(v___x_687_, 0, lean_box(0));
lean_closure_set(v___x_687_, 1, lean_box(0));
lean_closure_set(v___x_687_, 2, lean_box(0));
lean_closure_set(v___x_687_, 3, v_inst_683_);
lean_closure_set(v___x_687_, 4, v_inst_684_);
lean_closure_set(v___x_687_, 5, v_inst_685_);
lean_closure_set(v___x_687_, 6, lean_box(0));
return v___x_687_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCAddMonoidHomOfAddMonoidHomClass___redArg(lean_object* v_inst_688_, lean_object* v_inst_689_, lean_object* v_inst_690_){
_start:
{
lean_object* v___x_691_; 
v___x_691_ = lean_alloc_closure((void*)(lp_mathlib_AddMonoidHomClass_toAddMonoidHom___boxed), 8, 7);
lean_closure_set(v___x_691_, 0, lean_box(0));
lean_closure_set(v___x_691_, 1, lean_box(0));
lean_closure_set(v___x_691_, 2, lean_box(0));
lean_closure_set(v___x_691_, 3, v_inst_688_);
lean_closure_set(v___x_691_, 4, v_inst_689_);
lean_closure_set(v___x_691_, 5, v_inst_690_);
lean_closure_set(v___x_691_, 6, lean_box(0));
return v___x_691_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCAddMonoidHomOfAddMonoidHomClass(lean_object* v_M_692_, lean_object* v_N_693_, lean_object* v_F_694_, lean_object* v_inst_695_, lean_object* v_inst_696_, lean_object* v_inst_697_, lean_object* v_inst_698_){
_start:
{
lean_object* v___x_699_; 
v___x_699_ = lean_alloc_closure((void*)(lp_mathlib_AddMonoidHomClass_toAddMonoidHom___boxed), 8, 7);
lean_closure_set(v___x_699_, 0, lean_box(0));
lean_closure_set(v___x_699_, 1, lean_box(0));
lean_closure_set(v___x_699_, 2, lean_box(0));
lean_closure_set(v___x_699_, 3, v_inst_695_);
lean_closure_set(v___x_699_, 4, v_inst_696_);
lean_closure_set(v___x_699_, 5, v_inst_697_);
lean_closure_set(v___x_699_, 6, lean_box(0));
return v___x_699_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_coeToOneHom(lean_object* v_M_700_, lean_object* v_N_701_, lean_object* v_inst_702_, lean_object* v_inst_703_){
_start:
{
lean_object* v___f_704_; 
v___f_704_ = ((lean_object*)(lp_mathlib_OneHom_funLike___closed__0));
return v___f_704_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_coeToOneHom___boxed(lean_object* v_M_705_, lean_object* v_N_706_, lean_object* v_inst_707_, lean_object* v_inst_708_){
_start:
{
lean_object* v_res_709_; 
v_res_709_ = lp_mathlib_MonoidHom_coeToOneHom(v_M_705_, v_N_706_, v_inst_707_, v_inst_708_);
lean_dec_ref(v_inst_708_);
lean_dec_ref(v_inst_707_);
return v_res_709_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_coeToZeroHom(lean_object* v_M_710_, lean_object* v_N_711_, lean_object* v_inst_712_, lean_object* v_inst_713_){
_start:
{
lean_object* v___f_714_; 
v___f_714_ = ((lean_object*)(lp_mathlib_OneHom_funLike___closed__0));
return v___f_714_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_coeToZeroHom___boxed(lean_object* v_M_715_, lean_object* v_N_716_, lean_object* v_inst_717_, lean_object* v_inst_718_){
_start:
{
lean_object* v_res_719_; 
v_res_719_ = lp_mathlib_AddMonoidHom_coeToZeroHom(v_M_715_, v_N_716_, v_inst_717_, v_inst_718_);
lean_dec_ref(v_inst_718_);
lean_dec_ref(v_inst_717_);
return v_res_719_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_coeToMulHom___redArg(lean_object* v_inst_720_, lean_object* v_inst_721_){
_start:
{
lean_object* v___x_722_; 
v___x_722_ = lean_alloc_closure((void*)(lp_mathlib_MonoidHom_toMulHom___boxed), 5, 4);
lean_closure_set(v___x_722_, 0, lean_box(0));
lean_closure_set(v___x_722_, 1, lean_box(0));
lean_closure_set(v___x_722_, 2, v_inst_720_);
lean_closure_set(v___x_722_, 3, v_inst_721_);
return v___x_722_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_coeToMulHom(lean_object* v_M_723_, lean_object* v_N_724_, lean_object* v_inst_725_, lean_object* v_inst_726_){
_start:
{
lean_object* v___x_727_; 
v___x_727_ = lean_alloc_closure((void*)(lp_mathlib_MonoidHom_toMulHom___boxed), 5, 4);
lean_closure_set(v___x_727_, 0, lean_box(0));
lean_closure_set(v___x_727_, 1, lean_box(0));
lean_closure_set(v___x_727_, 2, v_inst_725_);
lean_closure_set(v___x_727_, 3, v_inst_726_);
return v___x_727_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_coeToAddHom___redArg(lean_object* v_inst_728_, lean_object* v_inst_729_){
_start:
{
lean_object* v___x_730_; 
v___x_730_ = lean_alloc_closure((void*)(lp_mathlib_AddMonoidHom_toAddHom___boxed), 5, 4);
lean_closure_set(v___x_730_, 0, lean_box(0));
lean_closure_set(v___x_730_, 1, lean_box(0));
lean_closure_set(v___x_730_, 2, v_inst_728_);
lean_closure_set(v___x_730_, 3, v_inst_729_);
return v___x_730_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_coeToAddHom(lean_object* v_M_731_, lean_object* v_N_732_, lean_object* v_inst_733_, lean_object* v_inst_734_){
_start:
{
lean_object* v___x_735_; 
v___x_735_ = lean_alloc_closure((void*)(lp_mathlib_AddMonoidHom_toAddHom___boxed), 5, 4);
lean_closure_set(v___x_735_, 0, lean_box(0));
lean_closure_set(v___x_735_, 1, lean_box(0));
lean_closure_set(v___x_735_, 2, v_inst_733_);
lean_closure_set(v___x_735_, 3, v_inst_734_);
return v___x_735_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_mk_x27___redArg(lean_object* v_f_736_){
_start:
{
lean_inc(v_f_736_);
return v_f_736_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_mk_x27___redArg___boxed(lean_object* v_f_737_){
_start:
{
lean_object* v_res_738_; 
v_res_738_ = lp_mathlib_MonoidHom_mk_x27___redArg(v_f_737_);
lean_dec(v_f_737_);
return v_res_738_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_mk_x27(lean_object* v_M_739_, lean_object* v_G_740_, lean_object* v_inst_741_, lean_object* v_inst_742_, lean_object* v_f_743_, lean_object* v_map__mul_744_){
_start:
{
lean_inc(v_f_743_);
return v_f_743_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_mk_x27___boxed(lean_object* v_M_745_, lean_object* v_G_746_, lean_object* v_inst_747_, lean_object* v_inst_748_, lean_object* v_f_749_, lean_object* v_map__mul_750_){
_start:
{
lean_object* v_res_751_; 
v_res_751_ = lp_mathlib_MonoidHom_mk_x27(v_M_745_, v_G_746_, v_inst_747_, v_inst_748_, v_f_749_, v_map__mul_750_);
lean_dec(v_f_749_);
lean_dec_ref(v_inst_748_);
lean_dec_ref(v_inst_747_);
return v_res_751_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mk_x27___redArg(lean_object* v_f_752_){
_start:
{
lean_inc(v_f_752_);
return v_f_752_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mk_x27___redArg___boxed(lean_object* v_f_753_){
_start:
{
lean_object* v_res_754_; 
v_res_754_ = lp_mathlib_AddMonoidHom_mk_x27___redArg(v_f_753_);
lean_dec(v_f_753_);
return v_res_754_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mk_x27(lean_object* v_M_755_, lean_object* v_G_756_, lean_object* v_inst_757_, lean_object* v_inst_758_, lean_object* v_f_759_, lean_object* v_map__mul_760_){
_start:
{
lean_inc(v_f_759_);
return v_f_759_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mk_x27___boxed(lean_object* v_M_761_, lean_object* v_G_762_, lean_object* v_inst_763_, lean_object* v_inst_764_, lean_object* v_f_765_, lean_object* v_map__mul_766_){
_start:
{
lean_object* v_res_767_; 
v_res_767_ = lp_mathlib_AddMonoidHom_mk_x27(v_M_761_, v_G_762_, v_inst_763_, v_inst_764_, v_f_765_, v_map__mul_766_);
lean_dec(v_f_765_);
lean_dec_ref(v_inst_764_);
lean_dec_ref(v_inst_763_);
return v_res_767_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OneHom_copy___redArg(lean_object* v_f_x27_768_){
_start:
{
lean_inc(v_f_x27_768_);
return v_f_x27_768_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OneHom_copy___redArg___boxed(lean_object* v_f_x27_769_){
_start:
{
lean_object* v_res_770_; 
v_res_770_ = lp_mathlib_OneHom_copy___redArg(v_f_x27_769_);
lean_dec(v_f_x27_769_);
return v_res_770_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OneHom_copy(lean_object* v_M_771_, lean_object* v_N_772_, lean_object* v_inst_773_, lean_object* v_inst_774_, lean_object* v_f_775_, lean_object* v_f_x27_776_, lean_object* v_h_777_){
_start:
{
lean_inc(v_f_x27_776_);
return v_f_x27_776_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OneHom_copy___boxed(lean_object* v_M_778_, lean_object* v_N_779_, lean_object* v_inst_780_, lean_object* v_inst_781_, lean_object* v_f_782_, lean_object* v_f_x27_783_, lean_object* v_h_784_){
_start:
{
lean_object* v_res_785_; 
v_res_785_ = lp_mathlib_OneHom_copy(v_M_778_, v_N_779_, v_inst_780_, v_inst_781_, v_f_782_, v_f_x27_783_, v_h_784_);
lean_dec(v_f_x27_783_);
lean_dec(v_f_782_);
lean_dec(v_inst_781_);
lean_dec(v_inst_780_);
return v_res_785_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_copy___redArg(lean_object* v_f_x27_786_){
_start:
{
lean_inc(v_f_x27_786_);
return v_f_x27_786_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_copy___redArg___boxed(lean_object* v_f_x27_787_){
_start:
{
lean_object* v_res_788_; 
v_res_788_ = lp_mathlib_ZeroHom_copy___redArg(v_f_x27_787_);
lean_dec(v_f_x27_787_);
return v_res_788_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_copy(lean_object* v_M_789_, lean_object* v_N_790_, lean_object* v_inst_791_, lean_object* v_inst_792_, lean_object* v_f_793_, lean_object* v_f_x27_794_, lean_object* v_h_795_){
_start:
{
lean_inc(v_f_x27_794_);
return v_f_x27_794_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_copy___boxed(lean_object* v_M_796_, lean_object* v_N_797_, lean_object* v_inst_798_, lean_object* v_inst_799_, lean_object* v_f_800_, lean_object* v_f_x27_801_, lean_object* v_h_802_){
_start:
{
lean_object* v_res_803_; 
v_res_803_ = lp_mathlib_ZeroHom_copy(v_M_796_, v_N_797_, v_inst_798_, v_inst_799_, v_f_800_, v_f_x27_801_, v_h_802_);
lean_dec(v_f_x27_801_);
lean_dec(v_f_800_);
lean_dec(v_inst_799_);
lean_dec(v_inst_798_);
return v_res_803_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulHom_copy___redArg(lean_object* v_f_x27_804_){
_start:
{
lean_inc(v_f_x27_804_);
return v_f_x27_804_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulHom_copy___redArg___boxed(lean_object* v_f_x27_805_){
_start:
{
lean_object* v_res_806_; 
v_res_806_ = lp_mathlib_MulHom_copy___redArg(v_f_x27_805_);
lean_dec(v_f_x27_805_);
return v_res_806_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulHom_copy(lean_object* v_M_807_, lean_object* v_N_808_, lean_object* v_inst_809_, lean_object* v_inst_810_, lean_object* v_f_811_, lean_object* v_f_x27_812_, lean_object* v_h_813_){
_start:
{
lean_inc(v_f_x27_812_);
return v_f_x27_812_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulHom_copy___boxed(lean_object* v_M_814_, lean_object* v_N_815_, lean_object* v_inst_816_, lean_object* v_inst_817_, lean_object* v_f_818_, lean_object* v_f_x27_819_, lean_object* v_h_820_){
_start:
{
lean_object* v_res_821_; 
v_res_821_ = lp_mathlib_MulHom_copy(v_M_814_, v_N_815_, v_inst_816_, v_inst_817_, v_f_818_, v_f_x27_819_, v_h_820_);
lean_dec(v_f_x27_819_);
lean_dec(v_f_818_);
lean_dec(v_inst_817_);
lean_dec(v_inst_816_);
return v_res_821_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddHom_copy___redArg(lean_object* v_f_x27_822_){
_start:
{
lean_inc(v_f_x27_822_);
return v_f_x27_822_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddHom_copy___redArg___boxed(lean_object* v_f_x27_823_){
_start:
{
lean_object* v_res_824_; 
v_res_824_ = lp_mathlib_AddHom_copy___redArg(v_f_x27_823_);
lean_dec(v_f_x27_823_);
return v_res_824_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddHom_copy(lean_object* v_M_825_, lean_object* v_N_826_, lean_object* v_inst_827_, lean_object* v_inst_828_, lean_object* v_f_829_, lean_object* v_f_x27_830_, lean_object* v_h_831_){
_start:
{
lean_inc(v_f_x27_830_);
return v_f_x27_830_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddHom_copy___boxed(lean_object* v_M_832_, lean_object* v_N_833_, lean_object* v_inst_834_, lean_object* v_inst_835_, lean_object* v_f_836_, lean_object* v_f_x27_837_, lean_object* v_h_838_){
_start:
{
lean_object* v_res_839_; 
v_res_839_ = lp_mathlib_AddHom_copy(v_M_832_, v_N_833_, v_inst_834_, v_inst_835_, v_f_836_, v_f_x27_837_, v_h_838_);
lean_dec(v_f_x27_837_);
lean_dec(v_f_836_);
lean_dec(v_inst_835_);
lean_dec(v_inst_834_);
return v_res_839_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_copy___redArg(lean_object* v_f_x27_840_){
_start:
{
lean_inc(v_f_x27_840_);
return v_f_x27_840_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_copy___redArg___boxed(lean_object* v_f_x27_841_){
_start:
{
lean_object* v_res_842_; 
v_res_842_ = lp_mathlib_MonoidHom_copy___redArg(v_f_x27_841_);
lean_dec(v_f_x27_841_);
return v_res_842_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_copy(lean_object* v_M_843_, lean_object* v_N_844_, lean_object* v_inst_845_, lean_object* v_inst_846_, lean_object* v_f_847_, lean_object* v_f_x27_848_, lean_object* v_h_849_){
_start:
{
lean_inc(v_f_x27_848_);
return v_f_x27_848_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_copy___boxed(lean_object* v_M_850_, lean_object* v_N_851_, lean_object* v_inst_852_, lean_object* v_inst_853_, lean_object* v_f_854_, lean_object* v_f_x27_855_, lean_object* v_h_856_){
_start:
{
lean_object* v_res_857_; 
v_res_857_ = lp_mathlib_MonoidHom_copy(v_M_850_, v_N_851_, v_inst_852_, v_inst_853_, v_f_854_, v_f_x27_855_, v_h_856_);
lean_dec(v_f_x27_855_);
lean_dec(v_f_854_);
lean_dec_ref(v_inst_853_);
lean_dec_ref(v_inst_852_);
return v_res_857_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_copy___redArg(lean_object* v_f_x27_858_){
_start:
{
lean_inc(v_f_x27_858_);
return v_f_x27_858_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_copy___redArg___boxed(lean_object* v_f_x27_859_){
_start:
{
lean_object* v_res_860_; 
v_res_860_ = lp_mathlib_AddMonoidHom_copy___redArg(v_f_x27_859_);
lean_dec(v_f_x27_859_);
return v_res_860_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_copy(lean_object* v_M_861_, lean_object* v_N_862_, lean_object* v_inst_863_, lean_object* v_inst_864_, lean_object* v_f_865_, lean_object* v_f_x27_866_, lean_object* v_h_867_){
_start:
{
lean_inc(v_f_x27_866_);
return v_f_x27_866_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_copy___boxed(lean_object* v_M_868_, lean_object* v_N_869_, lean_object* v_inst_870_, lean_object* v_inst_871_, lean_object* v_f_872_, lean_object* v_f_x27_873_, lean_object* v_h_874_){
_start:
{
lean_object* v_res_875_; 
v_res_875_ = lp_mathlib_AddMonoidHom_copy(v_M_868_, v_N_869_, v_inst_870_, v_inst_871_, v_f_872_, v_f_x27_873_, v_h_874_);
lean_dec(v_f_x27_873_);
lean_dec(v_f_872_);
lean_dec_ref(v_inst_871_);
lean_dec_ref(v_inst_870_);
return v_res_875_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OneHom_id___lam__0(lean_object* v_x_876_){
_start:
{
lean_inc(v_x_876_);
return v_x_876_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OneHom_id___lam__0___boxed(lean_object* v_x_877_){
_start:
{
lean_object* v_res_878_; 
v_res_878_ = lp_mathlib_OneHom_id___lam__0(v_x_877_);
lean_dec(v_x_877_);
return v_res_878_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OneHom_id(lean_object* v_M_880_, lean_object* v_inst_881_){
_start:
{
lean_object* v___f_882_; 
v___f_882_ = ((lean_object*)(lp_mathlib_OneHom_id___closed__0));
return v___f_882_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OneHom_id___boxed(lean_object* v_M_883_, lean_object* v_inst_884_){
_start:
{
lean_object* v_res_885_; 
v_res_885_ = lp_mathlib_OneHom_id(v_M_883_, v_inst_884_);
lean_dec(v_inst_884_);
return v_res_885_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_id(lean_object* v_M_886_, lean_object* v_inst_887_){
_start:
{
lean_object* v___f_888_; 
v___f_888_ = ((lean_object*)(lp_mathlib_OneHom_id___closed__0));
return v___f_888_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_id___boxed(lean_object* v_M_889_, lean_object* v_inst_890_){
_start:
{
lean_object* v_res_891_; 
v_res_891_ = lp_mathlib_ZeroHom_id(v_M_889_, v_inst_890_);
lean_dec(v_inst_890_);
return v_res_891_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulHom_id(lean_object* v_M_892_, lean_object* v_inst_893_){
_start:
{
lean_object* v___f_894_; 
v___f_894_ = ((lean_object*)(lp_mathlib_OneHom_id___closed__0));
return v___f_894_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulHom_id___boxed(lean_object* v_M_895_, lean_object* v_inst_896_){
_start:
{
lean_object* v_res_897_; 
v_res_897_ = lp_mathlib_MulHom_id(v_M_895_, v_inst_896_);
lean_dec(v_inst_896_);
return v_res_897_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddHom_id(lean_object* v_M_898_, lean_object* v_inst_899_){
_start:
{
lean_object* v___f_900_; 
v___f_900_ = ((lean_object*)(lp_mathlib_OneHom_id___closed__0));
return v___f_900_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddHom_id___boxed(lean_object* v_M_901_, lean_object* v_inst_902_){
_start:
{
lean_object* v_res_903_; 
v_res_903_ = lp_mathlib_AddHom_id(v_M_901_, v_inst_902_);
lean_dec(v_inst_902_);
return v_res_903_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_id(lean_object* v_M_904_, lean_object* v_inst_905_){
_start:
{
lean_object* v___f_906_; 
v___f_906_ = ((lean_object*)(lp_mathlib_OneHom_id___closed__0));
return v___f_906_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_id___boxed(lean_object* v_M_907_, lean_object* v_inst_908_){
_start:
{
lean_object* v_res_909_; 
v_res_909_ = lp_mathlib_MonoidHom_id(v_M_907_, v_inst_908_);
lean_dec_ref(v_inst_908_);
return v_res_909_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_id(lean_object* v_M_910_, lean_object* v_inst_911_){
_start:
{
lean_object* v___f_912_; 
v___f_912_ = ((lean_object*)(lp_mathlib_OneHom_id___closed__0));
return v___f_912_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_id___boxed(lean_object* v_M_913_, lean_object* v_inst_914_){
_start:
{
lean_object* v_res_915_; 
v_res_915_ = lp_mathlib_AddMonoidHom_id(v_M_913_, v_inst_914_);
lean_dec_ref(v_inst_914_);
return v_res_915_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OneHom_comp___redArg___lam__0(lean_object* v_hmn_916_, lean_object* v_hnp_917_, lean_object* v_x_918_){
_start:
{
lean_object* v___x_919_; lean_object* v___x_920_; 
v___x_919_ = lean_apply_1(v_hmn_916_, v_x_918_);
v___x_920_ = lean_apply_1(v_hnp_917_, v___x_919_);
return v___x_920_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OneHom_comp___redArg(lean_object* v_hnp_921_, lean_object* v_hmn_922_){
_start:
{
lean_object* v___f_923_; 
v___f_923_ = lean_alloc_closure((void*)(lp_mathlib_OneHom_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_923_, 0, v_hmn_922_);
lean_closure_set(v___f_923_, 1, v_hnp_921_);
return v___f_923_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OneHom_comp(lean_object* v_M_924_, lean_object* v_N_925_, lean_object* v_P_926_, lean_object* v_inst_927_, lean_object* v_inst_928_, lean_object* v_inst_929_, lean_object* v_hnp_930_, lean_object* v_hmn_931_){
_start:
{
lean_object* v___f_932_; 
v___f_932_ = lean_alloc_closure((void*)(lp_mathlib_OneHom_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_932_, 0, v_hmn_931_);
lean_closure_set(v___f_932_, 1, v_hnp_930_);
return v___f_932_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OneHom_comp___boxed(lean_object* v_M_933_, lean_object* v_N_934_, lean_object* v_P_935_, lean_object* v_inst_936_, lean_object* v_inst_937_, lean_object* v_inst_938_, lean_object* v_hnp_939_, lean_object* v_hmn_940_){
_start:
{
lean_object* v_res_941_; 
v_res_941_ = lp_mathlib_OneHom_comp(v_M_933_, v_N_934_, v_P_935_, v_inst_936_, v_inst_937_, v_inst_938_, v_hnp_939_, v_hmn_940_);
lean_dec(v_inst_938_);
lean_dec(v_inst_937_);
lean_dec(v_inst_936_);
return v_res_941_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_comp___redArg(lean_object* v_hnp_942_, lean_object* v_hmn_943_){
_start:
{
lean_object* v___f_944_; 
v___f_944_ = lean_alloc_closure((void*)(lp_mathlib_OneHom_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_944_, 0, v_hmn_943_);
lean_closure_set(v___f_944_, 1, v_hnp_942_);
return v___f_944_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_comp(lean_object* v_M_945_, lean_object* v_N_946_, lean_object* v_P_947_, lean_object* v_inst_948_, lean_object* v_inst_949_, lean_object* v_inst_950_, lean_object* v_hnp_951_, lean_object* v_hmn_952_){
_start:
{
lean_object* v___f_953_; 
v___f_953_ = lean_alloc_closure((void*)(lp_mathlib_OneHom_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_953_, 0, v_hmn_952_);
lean_closure_set(v___f_953_, 1, v_hnp_951_);
return v___f_953_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_comp___boxed(lean_object* v_M_954_, lean_object* v_N_955_, lean_object* v_P_956_, lean_object* v_inst_957_, lean_object* v_inst_958_, lean_object* v_inst_959_, lean_object* v_hnp_960_, lean_object* v_hmn_961_){
_start:
{
lean_object* v_res_962_; 
v_res_962_ = lp_mathlib_ZeroHom_comp(v_M_954_, v_N_955_, v_P_956_, v_inst_957_, v_inst_958_, v_inst_959_, v_hnp_960_, v_hmn_961_);
lean_dec(v_inst_959_);
lean_dec(v_inst_958_);
lean_dec(v_inst_957_);
return v_res_962_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulHom_comp___redArg(lean_object* v_hnp_963_, lean_object* v_hmn_964_){
_start:
{
lean_object* v___f_965_; 
v___f_965_ = lean_alloc_closure((void*)(lp_mathlib_OneHom_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_965_, 0, v_hmn_964_);
lean_closure_set(v___f_965_, 1, v_hnp_963_);
return v___f_965_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulHom_comp(lean_object* v_M_966_, lean_object* v_N_967_, lean_object* v_P_968_, lean_object* v_inst_969_, lean_object* v_inst_970_, lean_object* v_inst_971_, lean_object* v_hnp_972_, lean_object* v_hmn_973_){
_start:
{
lean_object* v___f_974_; 
v___f_974_ = lean_alloc_closure((void*)(lp_mathlib_OneHom_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_974_, 0, v_hmn_973_);
lean_closure_set(v___f_974_, 1, v_hnp_972_);
return v___f_974_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulHom_comp___boxed(lean_object* v_M_975_, lean_object* v_N_976_, lean_object* v_P_977_, lean_object* v_inst_978_, lean_object* v_inst_979_, lean_object* v_inst_980_, lean_object* v_hnp_981_, lean_object* v_hmn_982_){
_start:
{
lean_object* v_res_983_; 
v_res_983_ = lp_mathlib_MulHom_comp(v_M_975_, v_N_976_, v_P_977_, v_inst_978_, v_inst_979_, v_inst_980_, v_hnp_981_, v_hmn_982_);
lean_dec(v_inst_980_);
lean_dec(v_inst_979_);
lean_dec(v_inst_978_);
return v_res_983_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddHom_comp___redArg(lean_object* v_hnp_984_, lean_object* v_hmn_985_){
_start:
{
lean_object* v___f_986_; 
v___f_986_ = lean_alloc_closure((void*)(lp_mathlib_OneHom_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_986_, 0, v_hmn_985_);
lean_closure_set(v___f_986_, 1, v_hnp_984_);
return v___f_986_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddHom_comp(lean_object* v_M_987_, lean_object* v_N_988_, lean_object* v_P_989_, lean_object* v_inst_990_, lean_object* v_inst_991_, lean_object* v_inst_992_, lean_object* v_hnp_993_, lean_object* v_hmn_994_){
_start:
{
lean_object* v___f_995_; 
v___f_995_ = lean_alloc_closure((void*)(lp_mathlib_OneHom_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_995_, 0, v_hmn_994_);
lean_closure_set(v___f_995_, 1, v_hnp_993_);
return v___f_995_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddHom_comp___boxed(lean_object* v_M_996_, lean_object* v_N_997_, lean_object* v_P_998_, lean_object* v_inst_999_, lean_object* v_inst_1000_, lean_object* v_inst_1001_, lean_object* v_hnp_1002_, lean_object* v_hmn_1003_){
_start:
{
lean_object* v_res_1004_; 
v_res_1004_ = lp_mathlib_AddHom_comp(v_M_996_, v_N_997_, v_P_998_, v_inst_999_, v_inst_1000_, v_inst_1001_, v_hnp_1002_, v_hmn_1003_);
lean_dec(v_inst_1001_);
lean_dec(v_inst_1000_);
lean_dec(v_inst_999_);
return v_res_1004_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_comp___redArg(lean_object* v_hnp_1005_, lean_object* v_hmn_1006_){
_start:
{
lean_object* v___f_1007_; 
v___f_1007_ = lean_alloc_closure((void*)(lp_mathlib_OneHom_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_1007_, 0, v_hmn_1006_);
lean_closure_set(v___f_1007_, 1, v_hnp_1005_);
return v___f_1007_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_comp(lean_object* v_M_1008_, lean_object* v_N_1009_, lean_object* v_P_1010_, lean_object* v_inst_1011_, lean_object* v_inst_1012_, lean_object* v_inst_1013_, lean_object* v_hnp_1014_, lean_object* v_hmn_1015_){
_start:
{
lean_object* v___f_1016_; 
v___f_1016_ = lean_alloc_closure((void*)(lp_mathlib_OneHom_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_1016_, 0, v_hmn_1015_);
lean_closure_set(v___f_1016_, 1, v_hnp_1014_);
return v___f_1016_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_comp___boxed(lean_object* v_M_1017_, lean_object* v_N_1018_, lean_object* v_P_1019_, lean_object* v_inst_1020_, lean_object* v_inst_1021_, lean_object* v_inst_1022_, lean_object* v_hnp_1023_, lean_object* v_hmn_1024_){
_start:
{
lean_object* v_res_1025_; 
v_res_1025_ = lp_mathlib_MonoidHom_comp(v_M_1017_, v_N_1018_, v_P_1019_, v_inst_1020_, v_inst_1021_, v_inst_1022_, v_hnp_1023_, v_hmn_1024_);
lean_dec_ref(v_inst_1022_);
lean_dec_ref(v_inst_1021_);
lean_dec_ref(v_inst_1020_);
return v_res_1025_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_comp___redArg(lean_object* v_hnp_1026_, lean_object* v_hmn_1027_){
_start:
{
lean_object* v___f_1028_; 
v___f_1028_ = lean_alloc_closure((void*)(lp_mathlib_OneHom_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_1028_, 0, v_hmn_1027_);
lean_closure_set(v___f_1028_, 1, v_hnp_1026_);
return v___f_1028_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_comp(lean_object* v_M_1029_, lean_object* v_N_1030_, lean_object* v_P_1031_, lean_object* v_inst_1032_, lean_object* v_inst_1033_, lean_object* v_inst_1034_, lean_object* v_hnp_1035_, lean_object* v_hmn_1036_){
_start:
{
lean_object* v___f_1037_; 
v___f_1037_ = lean_alloc_closure((void*)(lp_mathlib_OneHom_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_1037_, 0, v_hmn_1036_);
lean_closure_set(v___f_1037_, 1, v_hnp_1035_);
return v___f_1037_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_comp___boxed(lean_object* v_M_1038_, lean_object* v_N_1039_, lean_object* v_P_1040_, lean_object* v_inst_1041_, lean_object* v_inst_1042_, lean_object* v_inst_1043_, lean_object* v_hnp_1044_, lean_object* v_hmn_1045_){
_start:
{
lean_object* v_res_1046_; 
v_res_1046_ = lp_mathlib_AddMonoidHom_comp(v_M_1038_, v_N_1039_, v_P_1040_, v_inst_1041_, v_inst_1042_, v_inst_1043_, v_hnp_1044_, v_hmn_1045_);
lean_dec_ref(v_inst_1043_);
lean_dec_ref(v_inst_1042_);
lean_dec_ref(v_inst_1041_);
return v_res_1046_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OneHom_inverse___redArg(lean_object* v_g_1047_){
_start:
{
lean_inc(v_g_1047_);
return v_g_1047_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OneHom_inverse___redArg___boxed(lean_object* v_g_1048_){
_start:
{
lean_object* v_res_1049_; 
v_res_1049_ = lp_mathlib_OneHom_inverse___redArg(v_g_1048_);
lean_dec(v_g_1048_);
return v_res_1049_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OneHom_inverse(lean_object* v_M_1050_, lean_object* v_N_1051_, lean_object* v_inst_1052_, lean_object* v_inst_1053_, lean_object* v_f_1054_, lean_object* v_g_1055_, lean_object* v_h_u2081_1056_){
_start:
{
lean_inc(v_g_1055_);
return v_g_1055_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OneHom_inverse___boxed(lean_object* v_M_1057_, lean_object* v_N_1058_, lean_object* v_inst_1059_, lean_object* v_inst_1060_, lean_object* v_f_1061_, lean_object* v_g_1062_, lean_object* v_h_u2081_1063_){
_start:
{
lean_object* v_res_1064_; 
v_res_1064_ = lp_mathlib_OneHom_inverse(v_M_1057_, v_N_1058_, v_inst_1059_, v_inst_1060_, v_f_1061_, v_g_1062_, v_h_u2081_1063_);
lean_dec(v_g_1062_);
lean_dec(v_f_1061_);
lean_dec(v_inst_1060_);
lean_dec(v_inst_1059_);
return v_res_1064_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_inverse___redArg(lean_object* v_g_1065_){
_start:
{
lean_inc(v_g_1065_);
return v_g_1065_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_inverse___redArg___boxed(lean_object* v_g_1066_){
_start:
{
lean_object* v_res_1067_; 
v_res_1067_ = lp_mathlib_ZeroHom_inverse___redArg(v_g_1066_);
lean_dec(v_g_1066_);
return v_res_1067_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_inverse(lean_object* v_M_1068_, lean_object* v_N_1069_, lean_object* v_inst_1070_, lean_object* v_inst_1071_, lean_object* v_f_1072_, lean_object* v_g_1073_, lean_object* v_h_u2081_1074_){
_start:
{
lean_inc(v_g_1073_);
return v_g_1073_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_inverse___boxed(lean_object* v_M_1075_, lean_object* v_N_1076_, lean_object* v_inst_1077_, lean_object* v_inst_1078_, lean_object* v_f_1079_, lean_object* v_g_1080_, lean_object* v_h_u2081_1081_){
_start:
{
lean_object* v_res_1082_; 
v_res_1082_ = lp_mathlib_ZeroHom_inverse(v_M_1075_, v_N_1076_, v_inst_1077_, v_inst_1078_, v_f_1079_, v_g_1080_, v_h_u2081_1081_);
lean_dec(v_g_1080_);
lean_dec(v_f_1079_);
lean_dec(v_inst_1078_);
lean_dec(v_inst_1077_);
return v_res_1082_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulHom_inverse___redArg(lean_object* v_g_1083_){
_start:
{
lean_inc(v_g_1083_);
return v_g_1083_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulHom_inverse___redArg___boxed(lean_object* v_g_1084_){
_start:
{
lean_object* v_res_1085_; 
v_res_1085_ = lp_mathlib_MulHom_inverse___redArg(v_g_1084_);
lean_dec(v_g_1084_);
return v_res_1085_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulHom_inverse(lean_object* v_M_1086_, lean_object* v_N_1087_, lean_object* v_inst_1088_, lean_object* v_inst_1089_, lean_object* v_f_1090_, lean_object* v_g_1091_, lean_object* v_h_u2081_1092_, lean_object* v_h_u2082_1093_){
_start:
{
lean_inc(v_g_1091_);
return v_g_1091_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulHom_inverse___boxed(lean_object* v_M_1094_, lean_object* v_N_1095_, lean_object* v_inst_1096_, lean_object* v_inst_1097_, lean_object* v_f_1098_, lean_object* v_g_1099_, lean_object* v_h_u2081_1100_, lean_object* v_h_u2082_1101_){
_start:
{
lean_object* v_res_1102_; 
v_res_1102_ = lp_mathlib_MulHom_inverse(v_M_1094_, v_N_1095_, v_inst_1096_, v_inst_1097_, v_f_1098_, v_g_1099_, v_h_u2081_1100_, v_h_u2082_1101_);
lean_dec(v_g_1099_);
lean_dec(v_f_1098_);
lean_dec(v_inst_1097_);
lean_dec(v_inst_1096_);
return v_res_1102_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddHom_inverse___redArg(lean_object* v_g_1103_){
_start:
{
lean_inc(v_g_1103_);
return v_g_1103_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddHom_inverse___redArg___boxed(lean_object* v_g_1104_){
_start:
{
lean_object* v_res_1105_; 
v_res_1105_ = lp_mathlib_AddHom_inverse___redArg(v_g_1104_);
lean_dec(v_g_1104_);
return v_res_1105_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddHom_inverse(lean_object* v_M_1106_, lean_object* v_N_1107_, lean_object* v_inst_1108_, lean_object* v_inst_1109_, lean_object* v_f_1110_, lean_object* v_g_1111_, lean_object* v_h_u2081_1112_, lean_object* v_h_u2082_1113_){
_start:
{
lean_inc(v_g_1111_);
return v_g_1111_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddHom_inverse___boxed(lean_object* v_M_1114_, lean_object* v_N_1115_, lean_object* v_inst_1116_, lean_object* v_inst_1117_, lean_object* v_f_1118_, lean_object* v_g_1119_, lean_object* v_h_u2081_1120_, lean_object* v_h_u2082_1121_){
_start:
{
lean_object* v_res_1122_; 
v_res_1122_ = lp_mathlib_AddHom_inverse(v_M_1114_, v_N_1115_, v_inst_1116_, v_inst_1117_, v_f_1118_, v_g_1119_, v_h_u2081_1120_, v_h_u2082_1121_);
lean_dec(v_g_1119_);
lean_dec(v_f_1118_);
lean_dec(v_inst_1117_);
lean_dec(v_inst_1116_);
return v_res_1122_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_inverse___redArg(lean_object* v_g_1123_){
_start:
{
lean_inc(v_g_1123_);
return v_g_1123_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_inverse___redArg___boxed(lean_object* v_g_1124_){
_start:
{
lean_object* v_res_1125_; 
v_res_1125_ = lp_mathlib_MonoidHom_inverse___redArg(v_g_1124_);
lean_dec(v_g_1124_);
return v_res_1125_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_inverse(lean_object* v_A_1126_, lean_object* v_B_1127_, lean_object* v_inst_1128_, lean_object* v_inst_1129_, lean_object* v_f_1130_, lean_object* v_g_1131_, lean_object* v_h_u2081_1132_, lean_object* v_h_u2082_1133_){
_start:
{
lean_inc(v_g_1131_);
return v_g_1131_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_inverse___boxed(lean_object* v_A_1134_, lean_object* v_B_1135_, lean_object* v_inst_1136_, lean_object* v_inst_1137_, lean_object* v_f_1138_, lean_object* v_g_1139_, lean_object* v_h_u2081_1140_, lean_object* v_h_u2082_1141_){
_start:
{
lean_object* v_res_1142_; 
v_res_1142_ = lp_mathlib_MonoidHom_inverse(v_A_1134_, v_B_1135_, v_inst_1136_, v_inst_1137_, v_f_1138_, v_g_1139_, v_h_u2081_1140_, v_h_u2082_1141_);
lean_dec(v_g_1139_);
lean_dec(v_f_1138_);
lean_dec_ref(v_inst_1137_);
lean_dec_ref(v_inst_1136_);
return v_res_1142_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_inverse___redArg(lean_object* v_g_1143_){
_start:
{
lean_inc(v_g_1143_);
return v_g_1143_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_inverse___redArg___boxed(lean_object* v_g_1144_){
_start:
{
lean_object* v_res_1145_; 
v_res_1145_ = lp_mathlib_AddMonoidHom_inverse___redArg(v_g_1144_);
lean_dec(v_g_1144_);
return v_res_1145_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_inverse(lean_object* v_A_1146_, lean_object* v_B_1147_, lean_object* v_inst_1148_, lean_object* v_inst_1149_, lean_object* v_f_1150_, lean_object* v_g_1151_, lean_object* v_h_u2081_1152_, lean_object* v_h_u2082_1153_){
_start:
{
lean_inc(v_g_1151_);
return v_g_1151_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_inverse___boxed(lean_object* v_A_1154_, lean_object* v_B_1155_, lean_object* v_inst_1156_, lean_object* v_inst_1157_, lean_object* v_f_1158_, lean_object* v_g_1159_, lean_object* v_h_u2081_1160_, lean_object* v_h_u2082_1161_){
_start:
{
lean_object* v_res_1162_; 
v_res_1162_ = lp_mathlib_AddMonoidHom_inverse(v_A_1154_, v_B_1155_, v_inst_1156_, v_inst_1157_, v_f_1158_, v_g_1159_, v_h_u2081_1160_, v_h_u2082_1161_);
lean_dec(v_g_1159_);
lean_dec(v_f_1158_);
lean_dec_ref(v_inst_1157_);
lean_dec_ref(v_inst_1156_);
return v_res_1162_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Monoid_End_instFunLike___aux__1___redArg(lean_object* v_f_1163_, lean_object* v_a_1164_){
_start:
{
lean_object* v___x_1165_; 
v___x_1165_ = lean_apply_1(v_f_1163_, v_a_1164_);
return v___x_1165_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Monoid_End_instFunLike___aux__1(lean_object* v_M_1166_, lean_object* v_inst_1167_, lean_object* v_f_1168_, lean_object* v_a_1169_){
_start:
{
lean_object* v___x_1170_; 
v___x_1170_ = lean_apply_1(v_f_1168_, v_a_1169_);
return v___x_1170_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Monoid_End_instFunLike___aux__1___boxed(lean_object* v_M_1171_, lean_object* v_inst_1172_, lean_object* v_f_1173_, lean_object* v_a_1174_){
_start:
{
lean_object* v_res_1175_; 
v_res_1175_ = lp_mathlib_Monoid_End_instFunLike___aux__1(v_M_1171_, v_inst_1172_, v_f_1173_, v_a_1174_);
lean_dec_ref(v_inst_1172_);
return v_res_1175_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Monoid_End_instFunLike___redArg(lean_object* v_inst_1176_){
_start:
{
lean_object* v___x_1177_; 
v___x_1177_ = lean_alloc_closure((void*)(lp_mathlib_Monoid_End_instFunLike___aux__1___boxed), 4, 2);
lean_closure_set(v___x_1177_, 0, lean_box(0));
lean_closure_set(v___x_1177_, 1, v_inst_1176_);
return v___x_1177_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Monoid_End_instFunLike(lean_object* v_M_1178_, lean_object* v_inst_1179_){
_start:
{
lean_object* v___x_1180_; 
v___x_1180_ = lean_alloc_closure((void*)(lp_mathlib_Monoid_End_instFunLike___aux__1___boxed), 4, 2);
lean_closure_set(v___x_1180_, 0, lean_box(0));
lean_closure_set(v___x_1180_, 1, v_inst_1179_);
return v___x_1180_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instFunLike___aux__1___redArg(lean_object* v_f_1181_, lean_object* v_a_1182_){
_start:
{
lean_object* v___x_1183_; 
v___x_1183_ = lean_apply_1(v_f_1181_, v_a_1182_);
return v___x_1183_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instFunLike___aux__1(lean_object* v_M_1184_, lean_object* v_inst_1185_, lean_object* v_f_1186_, lean_object* v_a_1187_){
_start:
{
lean_object* v___x_1188_; 
v___x_1188_ = lean_apply_1(v_f_1186_, v_a_1187_);
return v___x_1188_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instFunLike___aux__1___boxed(lean_object* v_M_1189_, lean_object* v_inst_1190_, lean_object* v_f_1191_, lean_object* v_a_1192_){
_start:
{
lean_object* v_res_1193_; 
v_res_1193_ = lp_mathlib_AddMonoid_End_instFunLike___aux__1(v_M_1189_, v_inst_1190_, v_f_1191_, v_a_1192_);
lean_dec_ref(v_inst_1190_);
return v_res_1193_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instFunLike___redArg(lean_object* v_inst_1194_){
_start:
{
lean_object* v___x_1195_; 
v___x_1195_ = lean_alloc_closure((void*)(lp_mathlib_AddMonoid_End_instFunLike___aux__1___boxed), 4, 2);
lean_closure_set(v___x_1195_, 0, lean_box(0));
lean_closure_set(v___x_1195_, 1, v_inst_1194_);
return v___x_1195_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instFunLike(lean_object* v_M_1196_, lean_object* v_inst_1197_){
_start:
{
lean_object* v___x_1198_; 
v___x_1198_ = lean_alloc_closure((void*)(lp_mathlib_AddMonoid_End_instFunLike___aux__1___boxed), 4, 2);
lean_closure_set(v___x_1198_, 0, lean_box(0));
lean_closure_set(v___x_1198_, 1, v_inst_1197_);
return v___x_1198_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Monoid_End_instOne(lean_object* v_M_1199_, lean_object* v_inst_1200_){
_start:
{
lean_object* v___f_1201_; 
v___f_1201_ = ((lean_object*)(lp_mathlib_OneHom_id___closed__0));
return v___f_1201_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Monoid_End_instOne___boxed(lean_object* v_M_1202_, lean_object* v_inst_1203_){
_start:
{
lean_object* v_res_1204_; 
v_res_1204_ = lp_mathlib_Monoid_End_instOne(v_M_1202_, v_inst_1203_);
lean_dec_ref(v_inst_1203_);
return v_res_1204_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instOne(lean_object* v_M_1205_, lean_object* v_inst_1206_){
_start:
{
lean_object* v___f_1207_; 
v___f_1207_ = ((lean_object*)(lp_mathlib_OneHom_id___closed__0));
return v___f_1207_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instOne___boxed(lean_object* v_M_1208_, lean_object* v_inst_1209_){
_start:
{
lean_object* v_res_1210_; 
v_res_1210_ = lp_mathlib_AddMonoid_End_instOne(v_M_1208_, v_inst_1209_);
lean_dec_ref(v_inst_1209_);
return v_res_1210_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Monoid_End_instMul___redArg(lean_object* v_inst_1211_){
_start:
{
lean_object* v___x_1212_; 
lean_inc_ref_n(v_inst_1211_, 2);
v___x_1212_ = lean_alloc_closure((void*)(lp_mathlib_MonoidHom_comp___boxed), 8, 6);
lean_closure_set(v___x_1212_, 0, lean_box(0));
lean_closure_set(v___x_1212_, 1, lean_box(0));
lean_closure_set(v___x_1212_, 2, lean_box(0));
lean_closure_set(v___x_1212_, 3, v_inst_1211_);
lean_closure_set(v___x_1212_, 4, v_inst_1211_);
lean_closure_set(v___x_1212_, 5, v_inst_1211_);
return v___x_1212_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Monoid_End_instMul(lean_object* v_M_1213_, lean_object* v_inst_1214_){
_start:
{
lean_object* v___x_1215_; 
lean_inc_ref_n(v_inst_1214_, 2);
v___x_1215_ = lean_alloc_closure((void*)(lp_mathlib_MonoidHom_comp___boxed), 8, 6);
lean_closure_set(v___x_1215_, 0, lean_box(0));
lean_closure_set(v___x_1215_, 1, lean_box(0));
lean_closure_set(v___x_1215_, 2, lean_box(0));
lean_closure_set(v___x_1215_, 3, v_inst_1214_);
lean_closure_set(v___x_1215_, 4, v_inst_1214_);
lean_closure_set(v___x_1215_, 5, v_inst_1214_);
return v___x_1215_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instMul___redArg(lean_object* v_inst_1216_){
_start:
{
lean_object* v___x_1217_; 
lean_inc_ref_n(v_inst_1216_, 2);
v___x_1217_ = lean_alloc_closure((void*)(lp_mathlib_AddMonoidHom_comp___boxed), 8, 6);
lean_closure_set(v___x_1217_, 0, lean_box(0));
lean_closure_set(v___x_1217_, 1, lean_box(0));
lean_closure_set(v___x_1217_, 2, lean_box(0));
lean_closure_set(v___x_1217_, 3, v_inst_1216_);
lean_closure_set(v___x_1217_, 4, v_inst_1216_);
lean_closure_set(v___x_1217_, 5, v_inst_1216_);
return v___x_1217_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instMul(lean_object* v_M_1218_, lean_object* v_inst_1219_){
_start:
{
lean_object* v___x_1220_; 
lean_inc_ref_n(v_inst_1219_, 2);
v___x_1220_ = lean_alloc_closure((void*)(lp_mathlib_AddMonoidHom_comp___boxed), 8, 6);
lean_closure_set(v___x_1220_, 0, lean_box(0));
lean_closure_set(v___x_1220_, 1, lean_box(0));
lean_closure_set(v___x_1220_, 2, lean_box(0));
lean_closure_set(v___x_1220_, 3, v_inst_1219_);
lean_closure_set(v___x_1220_, 4, v_inst_1219_);
lean_closure_set(v___x_1220_, 5, v_inst_1219_);
return v___x_1220_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Monoid_End_instMonoid___redArg___lam__0(lean_object* v_inst_1221_, lean_object* v_n_1222_, lean_object* v_f_1223_, lean_object* v___y_1224_){
_start:
{
lean_object* v___x_1225_; lean_object* v___x_1226_; 
v___x_1225_ = lean_alloc_closure((void*)(lp_mathlib_Monoid_End_instFunLike___aux__1___boxed), 4, 3);
lean_closure_set(v___x_1225_, 0, lean_box(0));
lean_closure_set(v___x_1225_, 1, v_inst_1221_);
lean_closure_set(v___x_1225_, 2, v_f_1223_);
v___x_1226_ = lp_mathlib_Nat_iterate___redArg(v___x_1225_, v_n_1222_, v___y_1224_);
return v___x_1226_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Monoid_End_instMonoid___redArg(lean_object* v_inst_1227_){
_start:
{
lean_object* v___f_1228_; lean_object* v___f_1229_; lean_object* v___x_1230_; lean_object* v___x_1231_; 
lean_inc_ref_n(v_inst_1227_, 3);
v___f_1228_ = lean_alloc_closure((void*)(lp_mathlib_Monoid_End_instMonoid___redArg___lam__0), 4, 1);
lean_closure_set(v___f_1228_, 0, v_inst_1227_);
v___f_1229_ = ((lean_object*)(lp_mathlib_OneHom_id___closed__0));
v___x_1230_ = lean_alloc_closure((void*)(lp_mathlib_MonoidHom_comp___boxed), 8, 6);
lean_closure_set(v___x_1230_, 0, lean_box(0));
lean_closure_set(v___x_1230_, 1, lean_box(0));
lean_closure_set(v___x_1230_, 2, lean_box(0));
lean_closure_set(v___x_1230_, 3, v_inst_1227_);
lean_closure_set(v___x_1230_, 4, v_inst_1227_);
lean_closure_set(v___x_1230_, 5, v_inst_1227_);
v___x_1231_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1231_, 0, v___f_1229_);
lean_ctor_set(v___x_1231_, 1, v___x_1230_);
lean_ctor_set(v___x_1231_, 2, v___f_1228_);
return v___x_1231_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Monoid_End_instMonoid(lean_object* v_M_1232_, lean_object* v_inst_1233_){
_start:
{
lean_object* v___x_1234_; 
v___x_1234_ = lp_mathlib_Monoid_End_instMonoid___redArg(v_inst_1233_);
return v___x_1234_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instMonoid___redArg___lam__0(lean_object* v_inst_1235_, lean_object* v_n_1236_, lean_object* v_f_1237_, lean_object* v___y_1238_){
_start:
{
lean_object* v___x_1239_; lean_object* v___x_1240_; 
v___x_1239_ = lean_alloc_closure((void*)(lp_mathlib_AddMonoid_End_instFunLike___aux__1___boxed), 4, 3);
lean_closure_set(v___x_1239_, 0, lean_box(0));
lean_closure_set(v___x_1239_, 1, v_inst_1235_);
lean_closure_set(v___x_1239_, 2, v_f_1237_);
v___x_1240_ = lp_mathlib_Nat_iterate___redArg(v___x_1239_, v_n_1236_, v___y_1238_);
return v___x_1240_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instMonoid___redArg(lean_object* v_inst_1241_){
_start:
{
lean_object* v___f_1242_; lean_object* v___f_1243_; lean_object* v___x_1244_; lean_object* v___x_1245_; 
lean_inc_ref_n(v_inst_1241_, 3);
v___f_1242_ = lean_alloc_closure((void*)(lp_mathlib_AddMonoid_End_instMonoid___redArg___lam__0), 4, 1);
lean_closure_set(v___f_1242_, 0, v_inst_1241_);
v___f_1243_ = ((lean_object*)(lp_mathlib_OneHom_id___closed__0));
v___x_1244_ = lean_alloc_closure((void*)(lp_mathlib_AddMonoidHom_comp___boxed), 8, 6);
lean_closure_set(v___x_1244_, 0, lean_box(0));
lean_closure_set(v___x_1244_, 1, lean_box(0));
lean_closure_set(v___x_1244_, 2, lean_box(0));
lean_closure_set(v___x_1244_, 3, v_inst_1241_);
lean_closure_set(v___x_1244_, 4, v_inst_1241_);
lean_closure_set(v___x_1244_, 5, v_inst_1241_);
v___x_1245_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1245_, 0, v___f_1243_);
lean_ctor_set(v___x_1245_, 1, v___x_1244_);
lean_ctor_set(v___x_1245_, 2, v___f_1242_);
return v___x_1245_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instMonoid(lean_object* v_M_1246_, lean_object* v_inst_1247_){
_start:
{
lean_object* v___x_1248_; 
v___x_1248_ = lp_mathlib_AddMonoid_End_instMonoid___redArg(v_inst_1247_);
return v___x_1248_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Monoid_End_instInhabited(lean_object* v_M_1249_, lean_object* v_inst_1250_){
_start:
{
lean_object* v___f_1251_; 
v___f_1251_ = ((lean_object*)(lp_mathlib_OneHom_id___closed__0));
return v___f_1251_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Monoid_End_instInhabited___boxed(lean_object* v_M_1252_, lean_object* v_inst_1253_){
_start:
{
lean_object* v_res_1254_; 
v_res_1254_ = lp_mathlib_Monoid_End_instInhabited(v_M_1252_, v_inst_1253_);
lean_dec_ref(v_inst_1253_);
return v_res_1254_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instInhabited(lean_object* v_M_1255_, lean_object* v_inst_1256_){
_start:
{
lean_object* v___f_1257_; 
v___f_1257_ = ((lean_object*)(lp_mathlib_OneHom_id___closed__0));
return v___f_1257_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instInhabited___boxed(lean_object* v_M_1258_, lean_object* v_inst_1259_){
_start:
{
lean_object* v_res_1260_; 
v_res_1260_ = lp_mathlib_AddMonoid_End_instInhabited(v_M_1258_, v_inst_1259_);
lean_dec_ref(v_inst_1259_);
return v_res_1260_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instOneOneHom___redArg___lam__0(lean_object* v_inst_1261_, lean_object* v_x_1262_){
_start:
{
lean_inc(v_inst_1261_);
return v_inst_1261_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instOneOneHom___redArg___lam__0___boxed(lean_object* v_inst_1263_, lean_object* v_x_1264_){
_start:
{
lean_object* v_res_1265_; 
v_res_1265_ = lp_mathlib_instOneOneHom___redArg___lam__0(v_inst_1263_, v_x_1264_);
lean_dec(v_x_1264_);
lean_dec(v_inst_1263_);
return v_res_1265_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instOneOneHom___redArg(lean_object* v_inst_1266_){
_start:
{
lean_object* v___f_1267_; 
v___f_1267_ = lean_alloc_closure((void*)(lp_mathlib_instOneOneHom___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_1267_, 0, v_inst_1266_);
return v___f_1267_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instOneOneHom(lean_object* v_M_1268_, lean_object* v_N_1269_, lean_object* v_inst_1270_, lean_object* v_inst_1271_){
_start:
{
lean_object* v___f_1272_; 
v___f_1272_ = lean_alloc_closure((void*)(lp_mathlib_instOneOneHom___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_1272_, 0, v_inst_1271_);
return v___f_1272_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instOneOneHom___boxed(lean_object* v_M_1273_, lean_object* v_N_1274_, lean_object* v_inst_1275_, lean_object* v_inst_1276_){
_start:
{
lean_object* v_res_1277_; 
v_res_1277_ = lp_mathlib_instOneOneHom(v_M_1273_, v_N_1274_, v_inst_1275_, v_inst_1276_);
lean_dec(v_inst_1275_);
return v_res_1277_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instZeroZeroHom___redArg(lean_object* v_inst_1278_){
_start:
{
lean_object* v___f_1279_; 
v___f_1279_ = lean_alloc_closure((void*)(lp_mathlib_instOneOneHom___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_1279_, 0, v_inst_1278_);
return v___f_1279_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instZeroZeroHom(lean_object* v_M_1280_, lean_object* v_N_1281_, lean_object* v_inst_1282_, lean_object* v_inst_1283_){
_start:
{
lean_object* v___f_1284_; 
v___f_1284_ = lean_alloc_closure((void*)(lp_mathlib_instOneOneHom___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_1284_, 0, v_inst_1283_);
return v___f_1284_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instZeroZeroHom___boxed(lean_object* v_M_1285_, lean_object* v_N_1286_, lean_object* v_inst_1287_, lean_object* v_inst_1288_){
_start:
{
lean_object* v_res_1289_; 
v_res_1289_ = lp_mathlib_instZeroZeroHom(v_M_1285_, v_N_1286_, v_inst_1287_, v_inst_1288_);
lean_dec(v_inst_1287_);
return v_res_1289_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instOneMulHom___redArg___lam__0(lean_object* v_toOne_1290_, lean_object* v_x_1291_){
_start:
{
lean_inc(v_toOne_1290_);
return v_toOne_1290_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instOneMulHom___redArg___lam__0___boxed(lean_object* v_toOne_1292_, lean_object* v_x_1293_){
_start:
{
lean_object* v_res_1294_; 
v_res_1294_ = lp_mathlib_instOneMulHom___redArg___lam__0(v_toOne_1292_, v_x_1293_);
lean_dec(v_x_1293_);
lean_dec(v_toOne_1292_);
return v_res_1294_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instOneMulHom___redArg(lean_object* v_inst_1295_){
_start:
{
lean_object* v___x_1296_; lean_object* v_toOne_1297_; lean_object* v___f_1298_; 
v___x_1296_ = lp_mathlib_MulOneClass_toMulOne___redArg(v_inst_1295_);
v_toOne_1297_ = lean_ctor_get(v___x_1296_, 0);
lean_inc(v_toOne_1297_);
lean_dec_ref(v___x_1296_);
v___f_1298_ = lean_alloc_closure((void*)(lp_mathlib_instOneMulHom___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_1298_, 0, v_toOne_1297_);
return v___f_1298_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instOneMulHom(lean_object* v_M_1299_, lean_object* v_N_1300_, lean_object* v_inst_1301_, lean_object* v_inst_1302_){
_start:
{
lean_object* v___x_1303_; 
v___x_1303_ = lp_mathlib_instOneMulHom___redArg(v_inst_1302_);
return v___x_1303_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instOneMulHom___boxed(lean_object* v_M_1304_, lean_object* v_N_1305_, lean_object* v_inst_1306_, lean_object* v_inst_1307_){
_start:
{
lean_object* v_res_1308_; 
v_res_1308_ = lp_mathlib_instOneMulHom(v_M_1304_, v_N_1305_, v_inst_1306_, v_inst_1307_);
lean_dec(v_inst_1306_);
return v_res_1308_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instZeroAddHom___redArg___lam__0(lean_object* v_toZero_1309_, lean_object* v_x_1310_){
_start:
{
lean_inc(v_toZero_1309_);
return v_toZero_1309_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instZeroAddHom___redArg___lam__0___boxed(lean_object* v_toZero_1311_, lean_object* v_x_1312_){
_start:
{
lean_object* v_res_1313_; 
v_res_1313_ = lp_mathlib_instZeroAddHom___redArg___lam__0(v_toZero_1311_, v_x_1312_);
lean_dec(v_x_1312_);
lean_dec(v_toZero_1311_);
return v_res_1313_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instZeroAddHom___redArg(lean_object* v_inst_1314_){
_start:
{
lean_object* v___x_1315_; lean_object* v_toZero_1316_; lean_object* v___f_1317_; 
v___x_1315_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v_inst_1314_);
v_toZero_1316_ = lean_ctor_get(v___x_1315_, 0);
lean_inc(v_toZero_1316_);
lean_dec_ref(v___x_1315_);
v___f_1317_ = lean_alloc_closure((void*)(lp_mathlib_instZeroAddHom___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_1317_, 0, v_toZero_1316_);
return v___f_1317_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instZeroAddHom(lean_object* v_M_1318_, lean_object* v_N_1319_, lean_object* v_inst_1320_, lean_object* v_inst_1321_){
_start:
{
lean_object* v___x_1322_; 
v___x_1322_ = lp_mathlib_instZeroAddHom___redArg(v_inst_1321_);
return v___x_1322_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instZeroAddHom___boxed(lean_object* v_M_1323_, lean_object* v_N_1324_, lean_object* v_inst_1325_, lean_object* v_inst_1326_){
_start:
{
lean_object* v_res_1327_; 
v_res_1327_ = lp_mathlib_instZeroAddHom(v_M_1323_, v_N_1324_, v_inst_1325_, v_inst_1326_);
lean_dec(v_inst_1325_);
return v_res_1327_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instOneMonoidHom___redArg(lean_object* v_inst_1328_){
_start:
{
lean_object* v___x_1329_; lean_object* v_toOne_1330_; lean_object* v___f_1331_; 
v___x_1329_ = lp_mathlib_MulOneClass_toMulOne___redArg(v_inst_1328_);
v_toOne_1330_ = lean_ctor_get(v___x_1329_, 0);
lean_inc(v_toOne_1330_);
lean_dec_ref(v___x_1329_);
v___f_1331_ = lean_alloc_closure((void*)(lp_mathlib_instOneMulHom___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_1331_, 0, v_toOne_1330_);
return v___f_1331_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instOneMonoidHom(lean_object* v_M_1332_, lean_object* v_N_1333_, lean_object* v_inst_1334_, lean_object* v_inst_1335_){
_start:
{
lean_object* v___x_1336_; 
v___x_1336_ = lp_mathlib_instOneMonoidHom___redArg(v_inst_1335_);
return v___x_1336_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instOneMonoidHom___boxed(lean_object* v_M_1337_, lean_object* v_N_1338_, lean_object* v_inst_1339_, lean_object* v_inst_1340_){
_start:
{
lean_object* v_res_1341_; 
v_res_1341_ = lp_mathlib_instOneMonoidHom(v_M_1337_, v_N_1338_, v_inst_1339_, v_inst_1340_);
lean_dec_ref(v_inst_1339_);
return v_res_1341_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instZeroAddMonoidHom___redArg(lean_object* v_inst_1342_){
_start:
{
lean_object* v___x_1343_; lean_object* v_toZero_1344_; lean_object* v___f_1345_; 
v___x_1343_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v_inst_1342_);
v_toZero_1344_ = lean_ctor_get(v___x_1343_, 0);
lean_inc(v_toZero_1344_);
lean_dec_ref(v___x_1343_);
v___f_1345_ = lean_alloc_closure((void*)(lp_mathlib_instZeroAddHom___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_1345_, 0, v_toZero_1344_);
return v___f_1345_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instZeroAddMonoidHom(lean_object* v_M_1346_, lean_object* v_N_1347_, lean_object* v_inst_1348_, lean_object* v_inst_1349_){
_start:
{
lean_object* v___x_1350_; 
v___x_1350_ = lp_mathlib_instZeroAddMonoidHom___redArg(v_inst_1349_);
return v___x_1350_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instZeroAddMonoidHom___boxed(lean_object* v_M_1351_, lean_object* v_N_1352_, lean_object* v_inst_1353_, lean_object* v_inst_1354_){
_start:
{
lean_object* v_res_1355_; 
v_res_1355_ = lp_mathlib_instZeroAddMonoidHom(v_M_1351_, v_N_1352_, v_inst_1353_, v_inst_1354_);
lean_dec_ref(v_inst_1353_);
return v_res_1355_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedOneHom___redArg(lean_object* v_inst_1356_){
_start:
{
lean_object* v___f_1357_; 
v___f_1357_ = lean_alloc_closure((void*)(lp_mathlib_instOneOneHom___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_1357_, 0, v_inst_1356_);
return v___f_1357_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedOneHom(lean_object* v_M_1358_, lean_object* v_N_1359_, lean_object* v_inst_1360_, lean_object* v_inst_1361_){
_start:
{
lean_object* v___f_1362_; 
v___f_1362_ = lean_alloc_closure((void*)(lp_mathlib_instOneOneHom___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_1362_, 0, v_inst_1361_);
return v___f_1362_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedOneHom___boxed(lean_object* v_M_1363_, lean_object* v_N_1364_, lean_object* v_inst_1365_, lean_object* v_inst_1366_){
_start:
{
lean_object* v_res_1367_; 
v_res_1367_ = lp_mathlib_instInhabitedOneHom(v_M_1363_, v_N_1364_, v_inst_1365_, v_inst_1366_);
lean_dec(v_inst_1365_);
return v_res_1367_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedZeroHom___redArg(lean_object* v_inst_1368_){
_start:
{
lean_object* v___f_1369_; 
v___f_1369_ = lean_alloc_closure((void*)(lp_mathlib_instOneOneHom___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_1369_, 0, v_inst_1368_);
return v___f_1369_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedZeroHom(lean_object* v_M_1370_, lean_object* v_N_1371_, lean_object* v_inst_1372_, lean_object* v_inst_1373_){
_start:
{
lean_object* v___f_1374_; 
v___f_1374_ = lean_alloc_closure((void*)(lp_mathlib_instOneOneHom___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_1374_, 0, v_inst_1373_);
return v___f_1374_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedZeroHom___boxed(lean_object* v_M_1375_, lean_object* v_N_1376_, lean_object* v_inst_1377_, lean_object* v_inst_1378_){
_start:
{
lean_object* v_res_1379_; 
v_res_1379_ = lp_mathlib_instInhabitedZeroHom(v_M_1375_, v_N_1376_, v_inst_1377_, v_inst_1378_);
lean_dec(v_inst_1377_);
return v_res_1379_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedMulHom___redArg(lean_object* v_inst_1380_){
_start:
{
lean_object* v___x_1381_; lean_object* v_toOne_1382_; lean_object* v___f_1383_; 
v___x_1381_ = lp_mathlib_MulOneClass_toMulOne___redArg(v_inst_1380_);
v_toOne_1382_ = lean_ctor_get(v___x_1381_, 0);
lean_inc(v_toOne_1382_);
lean_dec_ref(v___x_1381_);
v___f_1383_ = lean_alloc_closure((void*)(lp_mathlib_instOneMulHom___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_1383_, 0, v_toOne_1382_);
return v___f_1383_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedMulHom(lean_object* v_M_1384_, lean_object* v_N_1385_, lean_object* v_inst_1386_, lean_object* v_inst_1387_){
_start:
{
lean_object* v___x_1388_; 
v___x_1388_ = lp_mathlib_instInhabitedMulHom___redArg(v_inst_1387_);
return v___x_1388_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedMulHom___boxed(lean_object* v_M_1389_, lean_object* v_N_1390_, lean_object* v_inst_1391_, lean_object* v_inst_1392_){
_start:
{
lean_object* v_res_1393_; 
v_res_1393_ = lp_mathlib_instInhabitedMulHom(v_M_1389_, v_N_1390_, v_inst_1391_, v_inst_1392_);
lean_dec(v_inst_1391_);
return v_res_1393_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedAddHom___redArg(lean_object* v_inst_1394_){
_start:
{
lean_object* v___x_1395_; lean_object* v_toZero_1396_; lean_object* v___f_1397_; 
v___x_1395_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v_inst_1394_);
v_toZero_1396_ = lean_ctor_get(v___x_1395_, 0);
lean_inc(v_toZero_1396_);
lean_dec_ref(v___x_1395_);
v___f_1397_ = lean_alloc_closure((void*)(lp_mathlib_instZeroAddHom___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_1397_, 0, v_toZero_1396_);
return v___f_1397_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedAddHom(lean_object* v_M_1398_, lean_object* v_N_1399_, lean_object* v_inst_1400_, lean_object* v_inst_1401_){
_start:
{
lean_object* v___x_1402_; 
v___x_1402_ = lp_mathlib_instInhabitedAddHom___redArg(v_inst_1401_);
return v___x_1402_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedAddHom___boxed(lean_object* v_M_1403_, lean_object* v_N_1404_, lean_object* v_inst_1405_, lean_object* v_inst_1406_){
_start:
{
lean_object* v_res_1407_; 
v_res_1407_ = lp_mathlib_instInhabitedAddHom(v_M_1403_, v_N_1404_, v_inst_1405_, v_inst_1406_);
lean_dec(v_inst_1405_);
return v_res_1407_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedMonoidHom___redArg(lean_object* v_inst_1408_){
_start:
{
lean_object* v___x_1409_; lean_object* v_toOne_1410_; lean_object* v___f_1411_; 
v___x_1409_ = lp_mathlib_MulOneClass_toMulOne___redArg(v_inst_1408_);
v_toOne_1410_ = lean_ctor_get(v___x_1409_, 0);
lean_inc(v_toOne_1410_);
lean_dec_ref(v___x_1409_);
v___f_1411_ = lean_alloc_closure((void*)(lp_mathlib_instOneMulHom___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_1411_, 0, v_toOne_1410_);
return v___f_1411_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedMonoidHom(lean_object* v_M_1412_, lean_object* v_N_1413_, lean_object* v_inst_1414_, lean_object* v_inst_1415_){
_start:
{
lean_object* v___x_1416_; 
v___x_1416_ = lp_mathlib_instInhabitedMonoidHom___redArg(v_inst_1415_);
return v___x_1416_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedMonoidHom___boxed(lean_object* v_M_1417_, lean_object* v_N_1418_, lean_object* v_inst_1419_, lean_object* v_inst_1420_){
_start:
{
lean_object* v_res_1421_; 
v_res_1421_ = lp_mathlib_instInhabitedMonoidHom(v_M_1417_, v_N_1418_, v_inst_1419_, v_inst_1420_);
lean_dec_ref(v_inst_1419_);
return v_res_1421_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedAddMonoidHom___redArg(lean_object* v_inst_1422_){
_start:
{
lean_object* v___x_1423_; lean_object* v_toZero_1424_; lean_object* v___f_1425_; 
v___x_1423_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v_inst_1422_);
v_toZero_1424_ = lean_ctor_get(v___x_1423_, 0);
lean_inc(v_toZero_1424_);
lean_dec_ref(v___x_1423_);
v___f_1425_ = lean_alloc_closure((void*)(lp_mathlib_instZeroAddHom___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_1425_, 0, v_toZero_1424_);
return v___f_1425_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedAddMonoidHom(lean_object* v_M_1426_, lean_object* v_N_1427_, lean_object* v_inst_1428_, lean_object* v_inst_1429_){
_start:
{
lean_object* v___x_1430_; 
v___x_1430_ = lp_mathlib_instInhabitedAddMonoidHom___redArg(v_inst_1429_);
return v___x_1430_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedAddMonoidHom___boxed(lean_object* v_M_1431_, lean_object* v_N_1432_, lean_object* v_inst_1433_, lean_object* v_inst_1434_){
_start:
{
lean_object* v_res_1435_; 
v_res_1435_ = lp_mathlib_instInhabitedAddMonoidHom(v_M_1431_, v_N_1432_, v_inst_1433_, v_inst_1434_);
lean_dec_ref(v_inst_1433_);
return v_res_1435_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Notation_Pi_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_FunLike_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Logic_Function_Iterate(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Hom_Defs(uint8_t builtin) {
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
res = runtime_initialize_mathlib_Mathlib_Algebra_Notation_Pi_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_FunLike_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Logic_Function_Iterate(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Group_Hom_Defs(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_LibraryNote_hom__simp__lemma__priority = _init_lp_mathlib_LibraryNote_hom__simp__lemma__priority();
lean_mark_persistent(lp_mathlib_LibraryNote_hom__simp__lemma__priority);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Notation_Pi_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_FunLike_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Logic_Function_Iterate(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Group_Hom_Defs(uint8_t builtin) {
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
res = initialize_mathlib_Mathlib_Algebra_Notation_Pi_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_FunLike_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Function_Iterate(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Hom_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Group_Hom_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Group_Hom_Defs(builtin);
}
#ifdef __cplusplus
}
#endif
