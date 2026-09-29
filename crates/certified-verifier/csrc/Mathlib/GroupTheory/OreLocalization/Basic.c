// Lean compiler output
// Module: Mathlib.GroupTheory.OreLocalization.Basic
// Imports: public import Init public meta import Init public import Mathlib.GroupTheory.OreLocalization.OreSet public import Mathlib.Tactic.Common public import Mathlib.Algebra.Group.Submonoid.MulAction public import Mathlib.Algebra.Group.Units.Defs public import Mathlib.Algebra.Group.Basic public import Mathlib.Tactic.Attr.Core
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
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_AddZeroClass_toAddZero___redArg(lean_object*);
lean_object* lp_mathlib_AddMonoid_toAddAction___redArg(lean_object*);
lean_object* lp_mathlib_AddOreLocalization_oreMin___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddOreLocalization_oreSubtra___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Monoid_toMulOneClass___redArg(lean_object*);
lean_object* lp_mathlib_MulOneClass_toMulOne___redArg(lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* lp_mathlib_OreLocalization_oreNum___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_OreLocalization_oreDenom___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_nsmulRec___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Monoid_toMulAction___redArg(lean_object*);
lean_object* l_npowRec___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_oreEqv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_oreEqv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_oreEqv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_oreEqv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "OreLocalization"};
static const lean_object* lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__0 = (const lean_object*)&lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__0_value;
static const lean_string_object lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 12, .m_data = "term__[_⁻¹_]"};
static const lean_object* lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__1 = (const lean_object*)&lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__1_value;
static const lean_ctor_object lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__0_value),LEAN_SCALAR_PTR_LITERAL(116, 55, 79, 111, 201, 47, 225, 45)}};
static const lean_ctor_object lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__2_value_aux_0),((lean_object*)&lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__1_value),LEAN_SCALAR_PTR_LITERAL(178, 174, 155, 44, 233, 59, 148, 6)}};
static const lean_object* lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__2 = (const lean_object*)&lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__2_value;
static const lean_string_object lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__3 = (const lean_object*)&lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__3_value;
static const lean_ctor_object lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__3_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__4 = (const lean_object*)&lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__4_value;
static const lean_string_object lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "noWs"};
static const lean_object* lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__5 = (const lean_object*)&lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__5_value;
static const lean_ctor_object lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__5_value),LEAN_SCALAR_PTR_LITERAL(92, 29, 204, 148, 167, 109, 242, 21)}};
static const lean_object* lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__6 = (const lean_object*)&lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__6_value;
static const lean_ctor_object lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__6_value)}};
static const lean_object* lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__7 = (const lean_object*)&lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__7_value;
static const lean_string_object lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "atomic"};
static const lean_object* lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__8 = (const lean_object*)&lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__8_value;
static const lean_ctor_object lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__8_value),LEAN_SCALAR_PTR_LITERAL(56, 145, 113, 208, 127, 167, 216, 55)}};
static const lean_object* lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__9 = (const lean_object*)&lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__9_value;
static const lean_string_object lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "["};
static const lean_object* lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__10 = (const lean_object*)&lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__10_value;
static const lean_ctor_object lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__10_value)}};
static const lean_object* lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__11 = (const lean_object*)&lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__11_value;
static const lean_string_object lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__12 = (const lean_object*)&lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__12_value;
static const lean_ctor_object lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__12_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__13 = (const lean_object*)&lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__13_value;
static const lean_ctor_object lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__13_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__14 = (const lean_object*)&lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__14_value;
static const lean_ctor_object lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__4_value),((lean_object*)&lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__11_value),((lean_object*)&lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__14_value)}};
static const lean_object* lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__15 = (const lean_object*)&lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__15_value;
static const lean_string_object lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 2, .m_data = "⁻¹"};
static const lean_object* lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__16 = (const lean_object*)&lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__16_value;
static const lean_ctor_object lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__16_value)}};
static const lean_object* lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__17 = (const lean_object*)&lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__17_value;
static const lean_ctor_object lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__4_value),((lean_object*)&lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__15_value),((lean_object*)&lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__17_value)}};
static const lean_object* lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__18 = (const lean_object*)&lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__18_value;
static const lean_ctor_object lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__4_value),((lean_object*)&lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__18_value),((lean_object*)&lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__7_value)}};
static const lean_object* lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__19 = (const lean_object*)&lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__19_value;
static const lean_string_object lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "]"};
static const lean_object* lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__20 = (const lean_object*)&lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__20_value;
static const lean_ctor_object lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__20_value)}};
static const lean_object* lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__21 = (const lean_object*)&lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__21_value;
static const lean_ctor_object lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__4_value),((lean_object*)&lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__19_value),((lean_object*)&lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__21_value)}};
static const lean_object* lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__22 = (const lean_object*)&lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__22_value;
static const lean_ctor_object lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__9_value),((lean_object*)&lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__22_value)}};
static const lean_object* lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__23 = (const lean_object*)&lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__23_value;
static const lean_ctor_object lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__4_value),((lean_object*)&lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__7_value),((lean_object*)&lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__23_value)}};
static const lean_object* lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__24 = (const lean_object*)&lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__24_value;
static const lean_ctor_object lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__2_value),((lean_object*)(((size_t)(1075) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__24_value)}};
static const lean_object* lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__25 = (const lean_object*)&lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__25_value;
LEAN_EXPORT const lean_object* lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d = (const lean_object*)&lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__25_value;
static const lean_string_object lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__term_____x5b___x5d__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "term__[_]"};
static const lean_object* lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__term_____x5b___x5d__1___closed__0 = (const lean_object*)&lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__term_____x5b___x5d__1___closed__0_value;
static const lean_ctor_object lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__term_____x5b___x5d__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__term_____x5b___x5d__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(167, 68, 146, 84, 128, 183, 70, 246)}};
static const lean_object* lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__term_____x5b___x5d__1___closed__1 = (const lean_object*)&lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__term_____x5b___x5d__1___closed__1_value;
static const lean_string_object lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__term_____x5b___x5d__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 7, .m_data = "term_⁻¹"};
static const lean_object* lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__term_____x5b___x5d__1___closed__2 = (const lean_object*)&lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__term_____x5b___x5d__1___closed__2_value;
static const lean_ctor_object lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__term_____x5b___x5d__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__term_____x5b___x5d__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(42, 245, 188, 102, 250, 83, 210, 162)}};
static const lean_object* lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__term_____x5b___x5d__1___closed__3 = (const lean_object*)&lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__term_____x5b___x5d__1___closed__3_value;
static const lean_string_object lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__term_____x5b___x5d__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__term_____x5b___x5d__1___closed__4 = (const lean_object*)&lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__term_____x5b___x5d__1___closed__4_value;
static const lean_string_object lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__term_____x5b___x5d__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__term_____x5b___x5d__1___closed__5 = (const lean_object*)&lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__term_____x5b___x5d__1___closed__5_value;
static const lean_string_object lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__term_____x5b___x5d__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__term_____x5b___x5d__1___closed__6 = (const lean_object*)&lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__term_____x5b___x5d__1___closed__6_value;
static const lean_string_object lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__term_____x5b___x5d__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__term_____x5b___x5d__1___closed__7 = (const lean_object*)&lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__term_____x5b___x5d__1___closed__7_value;
static const lean_ctor_object lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__term_____x5b___x5d__1___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__term_____x5b___x5d__1___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__term_____x5b___x5d__1___closed__8_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__term_____x5b___x5d__1___closed__8_value_aux_0),((lean_object*)&lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__term_____x5b___x5d__1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__term_____x5b___x5d__1___closed__8_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__term_____x5b___x5d__1___closed__8_value_aux_1),((lean_object*)&lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__term_____x5b___x5d__1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__term_____x5b___x5d__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__term_____x5b___x5d__1___closed__8_value_aux_2),((lean_object*)&lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__term_____x5b___x5d__1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__term_____x5b___x5d__1___closed__8 = (const lean_object*)&lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__term_____x5b___x5d__1___closed__8_value;
static lean_once_cell_t lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__term_____x5b___x5d__1___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__term_____x5b___x5d__1___closed__9;
static const lean_ctor_object lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__term_____x5b___x5d__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__0_value),LEAN_SCALAR_PTR_LITERAL(116, 55, 79, 111, 201, 47, 225, 45)}};
static const lean_object* lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__term_____x5b___x5d__1___closed__10 = (const lean_object*)&lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__term_____x5b___x5d__1___closed__10_value;
static const lean_ctor_object lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__term_____x5b___x5d__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__term_____x5b___x5d__1___closed__10_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__term_____x5b___x5d__1___closed__11 = (const lean_object*)&lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__term_____x5b___x5d__1___closed__11_value;
static const lean_ctor_object lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__term_____x5b___x5d__1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__term_____x5b___x5d__1___closed__10_value)}};
static const lean_object* lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__term_____x5b___x5d__1___closed__12 = (const lean_object*)&lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__term_____x5b___x5d__1___closed__12_value;
static const lean_ctor_object lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__term_____x5b___x5d__1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__term_____x5b___x5d__1___closed__12_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__term_____x5b___x5d__1___closed__13 = (const lean_object*)&lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__term_____x5b___x5d__1___closed__13_value;
static const lean_ctor_object lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__term_____x5b___x5d__1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__term_____x5b___x5d__1___closed__11_value),((lean_object*)&lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__term_____x5b___x5d__1___closed__13_value)}};
static const lean_object* lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__term_____x5b___x5d__1___closed__14 = (const lean_object*)&lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__term_____x5b___x5d__1___closed__14_value;
static const lean_string_object lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__term_____x5b___x5d__1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__term_____x5b___x5d__1___closed__15 = (const lean_object*)&lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__term_____x5b___x5d__1___closed__15_value;
static const lean_ctor_object lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__term_____x5b___x5d__1___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__term_____x5b___x5d__1___closed__15_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__term_____x5b___x5d__1___closed__16 = (const lean_object*)&lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__term_____x5b___x5d__1___closed__16_value;
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__term_____x5b___x5d__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__term_____x5b___x5d__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_oreDiv___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_oreDiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_oreDiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_oreSub___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_oreSub(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_oreSub___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_OreLocalization_term___x2f_u2092___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 8, .m_data = "term_/ₒ_"};
static const lean_object* lp_mathlib_OreLocalization_term___x2f_u2092___00__closed__0 = (const lean_object*)&lp_mathlib_OreLocalization_term___x2f_u2092___00__closed__0_value;
static const lean_ctor_object lp_mathlib_OreLocalization_term___x2f_u2092___00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__0_value),LEAN_SCALAR_PTR_LITERAL(116, 55, 79, 111, 201, 47, 225, 45)}};
static const lean_ctor_object lp_mathlib_OreLocalization_term___x2f_u2092___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_OreLocalization_term___x2f_u2092___00__closed__1_value_aux_0),((lean_object*)&lp_mathlib_OreLocalization_term___x2f_u2092___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(41, 84, 187, 137, 91, 235, 214, 29)}};
static const lean_object* lp_mathlib_OreLocalization_term___x2f_u2092___00__closed__1 = (const lean_object*)&lp_mathlib_OreLocalization_term___x2f_u2092___00__closed__1_value;
static const lean_string_object lp_mathlib_OreLocalization_term___x2f_u2092___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 4, .m_data = " /ₒ "};
static const lean_object* lp_mathlib_OreLocalization_term___x2f_u2092___00__closed__2 = (const lean_object*)&lp_mathlib_OreLocalization_term___x2f_u2092___00__closed__2_value;
static const lean_ctor_object lp_mathlib_OreLocalization_term___x2f_u2092___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_OreLocalization_term___x2f_u2092___00__closed__2_value)}};
static const lean_object* lp_mathlib_OreLocalization_term___x2f_u2092___00__closed__3 = (const lean_object*)&lp_mathlib_OreLocalization_term___x2f_u2092___00__closed__3_value;
static const lean_ctor_object lp_mathlib_OreLocalization_term___x2f_u2092___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__13_value),((lean_object*)(((size_t)(71) << 1) | 1))}};
static const lean_object* lp_mathlib_OreLocalization_term___x2f_u2092___00__closed__4 = (const lean_object*)&lp_mathlib_OreLocalization_term___x2f_u2092___00__closed__4_value;
static const lean_ctor_object lp_mathlib_OreLocalization_term___x2f_u2092___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__4_value),((lean_object*)&lp_mathlib_OreLocalization_term___x2f_u2092___00__closed__3_value),((lean_object*)&lp_mathlib_OreLocalization_term___x2f_u2092___00__closed__4_value)}};
static const lean_object* lp_mathlib_OreLocalization_term___x2f_u2092___00__closed__5 = (const lean_object*)&lp_mathlib_OreLocalization_term___x2f_u2092___00__closed__5_value;
static const lean_ctor_object lp_mathlib_OreLocalization_term___x2f_u2092___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_OreLocalization_term___x2f_u2092___00__closed__1_value),((lean_object*)(((size_t)(70) << 1) | 1)),((lean_object*)(((size_t)(70) << 1) | 1)),((lean_object*)&lp_mathlib_OreLocalization_term___x2f_u2092___00__closed__5_value)}};
static const lean_object* lp_mathlib_OreLocalization_term___x2f_u2092___00__closed__6 = (const lean_object*)&lp_mathlib_OreLocalization_term___x2f_u2092___00__closed__6_value;
LEAN_EXPORT const lean_object* lp_mathlib_OreLocalization_term___x2f_u2092__ = (const lean_object*)&lp_mathlib_OreLocalization_term___x2f_u2092___00__closed__6_value;
static const lean_string_object lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__OreLocalization__term___x2f_u2092____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "oreDiv"};
static const lean_object* lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__OreLocalization__term___x2f_u2092____1___closed__0 = (const lean_object*)&lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__OreLocalization__term___x2f_u2092____1___closed__0_value;
static lean_once_cell_t lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__OreLocalization__term___x2f_u2092____1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__OreLocalization__term___x2f_u2092____1___closed__1;
static const lean_ctor_object lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__OreLocalization__term___x2f_u2092____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__OreLocalization__term___x2f_u2092____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(159, 214, 93, 112, 232, 215, 19, 93)}};
static const lean_object* lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__OreLocalization__term___x2f_u2092____1___closed__2 = (const lean_object*)&lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__OreLocalization__term___x2f_u2092____1___closed__2_value;
static const lean_ctor_object lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__OreLocalization__term___x2f_u2092____1___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__0_value),LEAN_SCALAR_PTR_LITERAL(116, 55, 79, 111, 201, 47, 225, 45)}};
static const lean_ctor_object lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__OreLocalization__term___x2f_u2092____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__OreLocalization__term___x2f_u2092____1___closed__3_value_aux_0),((lean_object*)&lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__OreLocalization__term___x2f_u2092____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(28, 23, 40, 122, 121, 148, 58, 220)}};
static const lean_object* lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__OreLocalization__term___x2f_u2092____1___closed__3 = (const lean_object*)&lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__OreLocalization__term___x2f_u2092____1___closed__3_value;
static const lean_ctor_object lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__OreLocalization__term___x2f_u2092____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__OreLocalization__term___x2f_u2092____1___closed__3_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__OreLocalization__term___x2f_u2092____1___closed__4 = (const lean_object*)&lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__OreLocalization__term___x2f_u2092____1___closed__4_value;
static const lean_ctor_object lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__OreLocalization__term___x2f_u2092____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__OreLocalization__term___x2f_u2092____1___closed__4_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__OreLocalization__term___x2f_u2092____1___closed__5 = (const lean_object*)&lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__OreLocalization__term___x2f_u2092____1___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__OreLocalization__term___x2f_u2092____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__OreLocalization__term___x2f_u2092____1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______unexpand__OreLocalization__oreDiv__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______unexpand__OreLocalization__oreDiv__1___closed__0 = (const lean_object*)&lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______unexpand__OreLocalization__oreDiv__1___closed__0_value;
static const lean_ctor_object lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______unexpand__OreLocalization__oreDiv__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______unexpand__OreLocalization__oreDiv__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______unexpand__OreLocalization__oreDiv__1___closed__1 = (const lean_object*)&lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______unexpand__OreLocalization__oreDiv__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______unexpand__OreLocalization__oreDiv__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______unexpand__OreLocalization__oreDiv__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_OreLocalization_term___x2d_u2092___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 8, .m_data = "term_-ₒ_"};
static const lean_object* lp_mathlib_OreLocalization_term___x2d_u2092___00__closed__0 = (const lean_object*)&lp_mathlib_OreLocalization_term___x2d_u2092___00__closed__0_value;
static const lean_ctor_object lp_mathlib_OreLocalization_term___x2d_u2092___00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__0_value),LEAN_SCALAR_PTR_LITERAL(116, 55, 79, 111, 201, 47, 225, 45)}};
static const lean_ctor_object lp_mathlib_OreLocalization_term___x2d_u2092___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_OreLocalization_term___x2d_u2092___00__closed__1_value_aux_0),((lean_object*)&lp_mathlib_OreLocalization_term___x2d_u2092___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(127, 99, 186, 137, 165, 200, 176, 175)}};
static const lean_object* lp_mathlib_OreLocalization_term___x2d_u2092___00__closed__1 = (const lean_object*)&lp_mathlib_OreLocalization_term___x2d_u2092___00__closed__1_value;
static const lean_string_object lp_mathlib_OreLocalization_term___x2d_u2092___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 4, .m_data = " -ₒ "};
static const lean_object* lp_mathlib_OreLocalization_term___x2d_u2092___00__closed__2 = (const lean_object*)&lp_mathlib_OreLocalization_term___x2d_u2092___00__closed__2_value;
static const lean_ctor_object lp_mathlib_OreLocalization_term___x2d_u2092___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_OreLocalization_term___x2d_u2092___00__closed__2_value)}};
static const lean_object* lp_mathlib_OreLocalization_term___x2d_u2092___00__closed__3 = (const lean_object*)&lp_mathlib_OreLocalization_term___x2d_u2092___00__closed__3_value;
static const lean_ctor_object lp_mathlib_OreLocalization_term___x2d_u2092___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__13_value),((lean_object*)(((size_t)(66) << 1) | 1))}};
static const lean_object* lp_mathlib_OreLocalization_term___x2d_u2092___00__closed__4 = (const lean_object*)&lp_mathlib_OreLocalization_term___x2d_u2092___00__closed__4_value;
static const lean_ctor_object lp_mathlib_OreLocalization_term___x2d_u2092___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__4_value),((lean_object*)&lp_mathlib_OreLocalization_term___x2d_u2092___00__closed__3_value),((lean_object*)&lp_mathlib_OreLocalization_term___x2d_u2092___00__closed__4_value)}};
static const lean_object* lp_mathlib_OreLocalization_term___x2d_u2092___00__closed__5 = (const lean_object*)&lp_mathlib_OreLocalization_term___x2d_u2092___00__closed__5_value;
static const lean_ctor_object lp_mathlib_OreLocalization_term___x2d_u2092___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_OreLocalization_term___x2d_u2092___00__closed__1_value),((lean_object*)(((size_t)(65) << 1) | 1)),((lean_object*)(((size_t)(65) << 1) | 1)),((lean_object*)&lp_mathlib_OreLocalization_term___x2d_u2092___00__closed__5_value)}};
static const lean_object* lp_mathlib_OreLocalization_term___x2d_u2092___00__closed__6 = (const lean_object*)&lp_mathlib_OreLocalization_term___x2d_u2092___00__closed__6_value;
LEAN_EXPORT const lean_object* lp_mathlib_OreLocalization_term___x2d_u2092__ = (const lean_object*)&lp_mathlib_OreLocalization_term___x2d_u2092___00__closed__6_value;
static const lean_string_object lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__OreLocalization__term___x2d_u2092____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 33, .m_capacity = 33, .m_length = 32, .m_data = "_root_.AddOreLocalization.oreSub"};
static const lean_object* lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__OreLocalization__term___x2d_u2092____1___closed__0 = (const lean_object*)&lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__OreLocalization__term___x2d_u2092____1___closed__0_value;
static lean_once_cell_t lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__OreLocalization__term___x2d_u2092____1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__OreLocalization__term___x2d_u2092____1___closed__1;
static const lean_string_object lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__OreLocalization__term___x2d_u2092____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "_root_"};
static const lean_object* lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__OreLocalization__term___x2d_u2092____1___closed__2 = (const lean_object*)&lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__OreLocalization__term___x2d_u2092____1___closed__2_value;
static const lean_string_object lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__OreLocalization__term___x2d_u2092____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "AddOreLocalization"};
static const lean_object* lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__OreLocalization__term___x2d_u2092____1___closed__3 = (const lean_object*)&lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__OreLocalization__term___x2d_u2092____1___closed__3_value;
static const lean_string_object lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__OreLocalization__term___x2d_u2092____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "oreSub"};
static const lean_object* lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__OreLocalization__term___x2d_u2092____1___closed__4 = (const lean_object*)&lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__OreLocalization__term___x2d_u2092____1___closed__4_value;
static const lean_ctor_object lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__OreLocalization__term___x2d_u2092____1___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__OreLocalization__term___x2d_u2092____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(184, 175, 53, 50, 212, 152, 178, 8)}};
static const lean_ctor_object lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__OreLocalization__term___x2d_u2092____1___closed__5_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__OreLocalization__term___x2d_u2092____1___closed__5_value_aux_0),((lean_object*)&lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__OreLocalization__term___x2d_u2092____1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(253, 160, 210, 187, 101, 146, 172, 7)}};
static const lean_ctor_object lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__OreLocalization__term___x2d_u2092____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__OreLocalization__term___x2d_u2092____1___closed__5_value_aux_1),((lean_object*)&lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__OreLocalization__term___x2d_u2092____1___closed__4_value),LEAN_SCALAR_PTR_LITERAL(253, 13, 102, 9, 233, 188, 189, 192)}};
static const lean_object* lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__OreLocalization__term___x2d_u2092____1___closed__5 = (const lean_object*)&lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__OreLocalization__term___x2d_u2092____1___closed__5_value;
static const lean_ctor_object lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__OreLocalization__term___x2d_u2092____1___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__OreLocalization__term___x2d_u2092____1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(146, 109, 111, 187, 167, 118, 236, 113)}};
static const lean_ctor_object lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__OreLocalization__term___x2d_u2092____1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__OreLocalization__term___x2d_u2092____1___closed__6_value_aux_0),((lean_object*)&lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__OreLocalization__term___x2d_u2092____1___closed__4_value),LEAN_SCALAR_PTR_LITERAL(158, 31, 21, 206, 5, 18, 3, 115)}};
static const lean_object* lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__OreLocalization__term___x2d_u2092____1___closed__6 = (const lean_object*)&lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__OreLocalization__term___x2d_u2092____1___closed__6_value;
static const lean_ctor_object lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__OreLocalization__term___x2d_u2092____1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__OreLocalization__term___x2d_u2092____1___closed__6_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__OreLocalization__term___x2d_u2092____1___closed__7 = (const lean_object*)&lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__OreLocalization__term___x2d_u2092____1___closed__7_value;
static const lean_ctor_object lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__OreLocalization__term___x2d_u2092____1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__OreLocalization__term___x2d_u2092____1___closed__7_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__OreLocalization__term___x2d_u2092____1___closed__8 = (const lean_object*)&lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__OreLocalization__term___x2d_u2092____1___closed__8_value;
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__OreLocalization__term___x2d_u2092____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__OreLocalization__term___x2d_u2092____1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______unexpand__AddOreLocalization__oreSub__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______unexpand__AddOreLocalization__oreSub__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_liftExpand___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_liftExpand(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_liftExpand___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_liftExpand___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_liftExpand(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_liftExpand___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_lift_u2082Expand___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_lift_u2082Expand___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_lift_u2082Expand(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_lift_u2082Expand___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_lift_u2082Expand___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_lift_u2082Expand___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_lift_u2082Expand(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_lift_u2082Expand___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_smul___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_smul___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_smul___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_smul___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_smul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_smul___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_vadd___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_vadd___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_vadd___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_vadd___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_vadd(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_vadd___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instSMul___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instSMul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_instVAdd___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_instVAdd(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instMul___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instMul(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_instAdd___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_instAdd(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_oreDivSMulChar_x27___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_oreDivSMulChar_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_oreDivSMulChar_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_oreSubVAddChar_x27___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_oreSubVAddChar_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_oreSubVAddChar_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_oreDivMulChar_x27___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_oreDivMulChar_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_oreDivMulChar_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_oreSubAddChar_x27___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_oreSubAddChar_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_oreSubAddChar_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_one___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_one___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_one(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_one___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_zero___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_zero___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_zero(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_zero___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instOne___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instOne___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instOne(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instOne___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_instZero___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_instZero___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_instZero(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_instZero___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instInhabited___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instInhabited___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instInhabited(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instInhabited___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_instInhabited___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_instInhabited___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_instInhabited(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_instInhabited___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_npow___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_npow___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_npow(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_npow___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_nsmul___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_nsmul___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_nsmul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_nsmul___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instMonoid___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instMonoid(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_instAddMonoid___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_instAddMonoid(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instMulActionOreLocalization___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instMulActionOreLocalization(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_instAddActionOreLocalization___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_instAddActionOreLocalization(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_numeratorUnit___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_numeratorUnit___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_numeratorUnit(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_numeratorUnit___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_numeratorAddUnit___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_numeratorAddUnit___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_numeratorAddUnit(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_numeratorAddUnit___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_numeratorHom___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_numeratorHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_numeratorHom___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_numeratorHom(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_numeratorHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_numeratorHom___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_numeratorHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_numeratorHom___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_numeratorHom(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_numeratorHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_universalMulHom___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_universalMulHom___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_universalMulHom___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_universalMulHom___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_universalMulHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_universalMulHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_universalAddHom___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_universalAddHom___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_universalAddHom___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_universalAddHom___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_universalAddHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_universalAddHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_hsmul___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_hsmul___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_hsmul___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_hsmul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_hsmul___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_hvadd___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_hvadd___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_hvadd___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_hvadd(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_hvadd___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instSMulOfIsScalarTower___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instSMulOfIsScalarTower(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instSMulOfIsScalarTower___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_instVAddOfVAddAssocClass___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_instVAddOfVAddAssocClass(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_instVAddOfVAddAssocClass___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instMulActionOfIsScalarTower___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instMulActionOfIsScalarTower(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instMulActionOfIsScalarTower___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_instAddActionOfVAddAssocClass___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_instAddActionOfVAddAssocClass(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_instAddActionOfVAddAssocClass___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instCommMonoid___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instCommMonoid(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_instAddCommMonoid___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_instAddCommMonoid(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_zero___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_zero___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_zero(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_zero___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instZero___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instZero___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instZero(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instZero___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_oreEqv(lean_object* v_R_1_, lean_object* v_inst_2_, lean_object* v_S_3_, lean_object* v_inst_4_, lean_object* v_X_5_, lean_object* v_inst_6_){
_start:
{
lean_object* v___x_7_; 
v___x_7_ = lean_box(0);
return v___x_7_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_oreEqv___boxed(lean_object* v_R_8_, lean_object* v_inst_9_, lean_object* v_S_10_, lean_object* v_inst_11_, lean_object* v_X_12_, lean_object* v_inst_13_){
_start:
{
lean_object* v_res_14_; 
v_res_14_ = lp_mathlib_OreLocalization_oreEqv(v_R_8_, v_inst_9_, v_S_10_, v_inst_11_, v_X_12_, v_inst_13_);
lean_dec(v_inst_13_);
lean_dec_ref(v_inst_11_);
lean_dec_ref(v_inst_9_);
return v_res_14_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_oreEqv(lean_object* v_R_15_, lean_object* v_inst_16_, lean_object* v_S_17_, lean_object* v_inst_18_, lean_object* v_X_19_, lean_object* v_inst_20_){
_start:
{
lean_object* v___x_21_; 
v___x_21_ = lean_box(0);
return v___x_21_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_oreEqv___boxed(lean_object* v_R_22_, lean_object* v_inst_23_, lean_object* v_S_24_, lean_object* v_inst_25_, lean_object* v_X_26_, lean_object* v_inst_27_){
_start:
{
lean_object* v_res_28_; 
v_res_28_ = lp_mathlib_AddOreLocalization_oreEqv(v_R_22_, v_inst_23_, v_S_24_, v_inst_25_, v_X_26_, v_inst_27_);
lean_dec(v_inst_27_);
lean_dec_ref(v_inst_25_);
lean_dec_ref(v_inst_23_);
return v_res_28_;
}
}
static lean_object* _init_lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__term_____x5b___x5d__1___closed__9(void){
_start:
{
lean_object* v___x_104_; lean_object* v___x_105_; 
v___x_104_ = ((lean_object*)(lp_mathlib_OreLocalization_term_____x5b___u207b_xb9___x5d___closed__0));
v___x_105_ = l_String_toRawSubstring_x27(v___x_104_);
return v___x_105_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__term_____x5b___x5d__1(lean_object* v_x_122_, lean_object* v_a_123_, lean_object* v_a_124_){
_start:
{
lean_object* v___x_125_; uint8_t v___x_126_; 
v___x_125_ = ((lean_object*)(lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__term_____x5b___x5d__1___closed__1));
lean_inc(v_x_122_);
v___x_126_ = l_Lean_Syntax_isOfKind(v_x_122_, v___x_125_);
if (v___x_126_ == 0)
{
lean_object* v___x_127_; lean_object* v___x_128_; 
lean_dec(v_x_122_);
v___x_127_ = lean_box(1);
v___x_128_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_128_, 0, v___x_127_);
lean_ctor_set(v___x_128_, 1, v_a_124_);
return v___x_128_;
}
else
{
lean_object* v___x_129_; lean_object* v___x_130_; lean_object* v___x_131_; uint8_t v___x_132_; 
v___x_129_ = lean_unsigned_to_nat(2u);
v___x_130_ = l_Lean_Syntax_getArg(v_x_122_, v___x_129_);
v___x_131_ = ((lean_object*)(lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__term_____x5b___x5d__1___closed__3));
lean_inc(v___x_130_);
v___x_132_ = l_Lean_Syntax_isOfKind(v___x_130_, v___x_131_);
if (v___x_132_ == 0)
{
lean_object* v___x_133_; lean_object* v___x_134_; 
lean_dec(v___x_130_);
lean_dec(v_x_122_);
v___x_133_ = lean_box(1);
v___x_134_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_134_, 0, v___x_133_);
lean_ctor_set(v___x_134_, 1, v_a_124_);
return v___x_134_;
}
else
{
lean_object* v_quotContext_135_; lean_object* v_currMacroScope_136_; lean_object* v_ref_137_; lean_object* v___x_138_; lean_object* v___x_139_; lean_object* v___x_140_; uint8_t v___x_141_; lean_object* v___x_142_; lean_object* v___x_143_; lean_object* v___x_144_; lean_object* v___x_145_; lean_object* v___x_146_; lean_object* v___x_147_; lean_object* v___x_148_; lean_object* v___x_149_; lean_object* v___x_150_; lean_object* v___x_151_; lean_object* v___x_152_; 
v_quotContext_135_ = lean_ctor_get(v_a_123_, 1);
v_currMacroScope_136_ = lean_ctor_get(v_a_123_, 2);
v_ref_137_ = lean_ctor_get(v_a_123_, 5);
v___x_138_ = lean_unsigned_to_nat(0u);
v___x_139_ = l_Lean_Syntax_getArg(v_x_122_, v___x_138_);
lean_dec(v_x_122_);
v___x_140_ = l_Lean_Syntax_getArg(v___x_130_, v___x_138_);
lean_dec(v___x_130_);
v___x_141_ = 0;
v___x_142_ = l_Lean_SourceInfo_fromRef(v_ref_137_, v___x_141_);
v___x_143_ = ((lean_object*)(lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__term_____x5b___x5d__1___closed__8));
v___x_144_ = lean_obj_once(&lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__term_____x5b___x5d__1___closed__9, &lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__term_____x5b___x5d__1___closed__9_once, _init_lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__term_____x5b___x5d__1___closed__9);
v___x_145_ = ((lean_object*)(lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__term_____x5b___x5d__1___closed__10));
lean_inc(v_currMacroScope_136_);
lean_inc(v_quotContext_135_);
v___x_146_ = l_Lean_addMacroScope(v_quotContext_135_, v___x_145_, v_currMacroScope_136_);
v___x_147_ = ((lean_object*)(lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__term_____x5b___x5d__1___closed__14));
lean_inc_n(v___x_142_, 2);
v___x_148_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_148_, 0, v___x_142_);
lean_ctor_set(v___x_148_, 1, v___x_144_);
lean_ctor_set(v___x_148_, 2, v___x_146_);
lean_ctor_set(v___x_148_, 3, v___x_147_);
v___x_149_ = ((lean_object*)(lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__term_____x5b___x5d__1___closed__16));
v___x_150_ = l_Lean_Syntax_node2(v___x_142_, v___x_149_, v___x_140_, v___x_139_);
v___x_151_ = l_Lean_Syntax_node2(v___x_142_, v___x_143_, v___x_148_, v___x_150_);
v___x_152_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_152_, 0, v___x_151_);
lean_ctor_set(v___x_152_, 1, v_a_124_);
return v___x_152_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__term_____x5b___x5d__1___boxed(lean_object* v_x_153_, lean_object* v_a_154_, lean_object* v_a_155_){
_start:
{
lean_object* v_res_156_; 
v_res_156_ = lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__term_____x5b___x5d__1(v_x_153_, v_a_154_, v_a_155_);
lean_dec_ref(v_a_154_);
return v_res_156_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_oreDiv___redArg(lean_object* v_r_157_, lean_object* v_s_158_){
_start:
{
lean_object* v___x_159_; 
v___x_159_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_159_, 0, v_r_157_);
lean_ctor_set(v___x_159_, 1, v_s_158_);
return v___x_159_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_oreDiv(lean_object* v_R_160_, lean_object* v_inst_161_, lean_object* v_S_162_, lean_object* v_inst_163_, lean_object* v_X_164_, lean_object* v_inst_165_, lean_object* v_r_166_, lean_object* v_s_167_){
_start:
{
lean_object* v___x_168_; 
v___x_168_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_168_, 0, v_r_166_);
lean_ctor_set(v___x_168_, 1, v_s_167_);
return v___x_168_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_oreDiv___boxed(lean_object* v_R_169_, lean_object* v_inst_170_, lean_object* v_S_171_, lean_object* v_inst_172_, lean_object* v_X_173_, lean_object* v_inst_174_, lean_object* v_r_175_, lean_object* v_s_176_){
_start:
{
lean_object* v_res_177_; 
v_res_177_ = lp_mathlib_OreLocalization_oreDiv(v_R_169_, v_inst_170_, v_S_171_, v_inst_172_, v_X_173_, v_inst_174_, v_r_175_, v_s_176_);
lean_dec(v_inst_174_);
lean_dec_ref(v_inst_172_);
lean_dec_ref(v_inst_170_);
return v_res_177_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_oreSub___redArg(lean_object* v_r_178_, lean_object* v_s_179_){
_start:
{
lean_object* v___x_180_; 
v___x_180_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_180_, 0, v_r_178_);
lean_ctor_set(v___x_180_, 1, v_s_179_);
return v___x_180_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_oreSub(lean_object* v_R_181_, lean_object* v_inst_182_, lean_object* v_S_183_, lean_object* v_inst_184_, lean_object* v_X_185_, lean_object* v_inst_186_, lean_object* v_r_187_, lean_object* v_s_188_){
_start:
{
lean_object* v___x_189_; 
v___x_189_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_189_, 0, v_r_187_);
lean_ctor_set(v___x_189_, 1, v_s_188_);
return v___x_189_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_oreSub___boxed(lean_object* v_R_190_, lean_object* v_inst_191_, lean_object* v_S_192_, lean_object* v_inst_193_, lean_object* v_X_194_, lean_object* v_inst_195_, lean_object* v_r_196_, lean_object* v_s_197_){
_start:
{
lean_object* v_res_198_; 
v_res_198_ = lp_mathlib_AddOreLocalization_oreSub(v_R_190_, v_inst_191_, v_S_192_, v_inst_193_, v_X_194_, v_inst_195_, v_r_196_, v_s_197_);
lean_dec(v_inst_195_);
lean_dec_ref(v_inst_193_);
lean_dec_ref(v_inst_191_);
return v_res_198_;
}
}
static lean_object* _init_lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__OreLocalization__term___x2f_u2092____1___closed__1(void){
_start:
{
lean_object* v___x_219_; lean_object* v___x_220_; 
v___x_219_ = ((lean_object*)(lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__OreLocalization__term___x2f_u2092____1___closed__0));
v___x_220_ = l_String_toRawSubstring_x27(v___x_219_);
return v___x_220_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__OreLocalization__term___x2f_u2092____1(lean_object* v_x_232_, lean_object* v_a_233_, lean_object* v_a_234_){
_start:
{
lean_object* v___x_235_; uint8_t v___x_236_; 
v___x_235_ = ((lean_object*)(lp_mathlib_OreLocalization_term___x2f_u2092___00__closed__1));
lean_inc(v_x_232_);
v___x_236_ = l_Lean_Syntax_isOfKind(v_x_232_, v___x_235_);
if (v___x_236_ == 0)
{
lean_object* v___x_237_; lean_object* v___x_238_; 
lean_dec(v_x_232_);
v___x_237_ = lean_box(1);
v___x_238_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_238_, 0, v___x_237_);
lean_ctor_set(v___x_238_, 1, v_a_234_);
return v___x_238_;
}
else
{
lean_object* v_quotContext_239_; lean_object* v_currMacroScope_240_; lean_object* v_ref_241_; lean_object* v___x_242_; lean_object* v___x_243_; lean_object* v___x_244_; lean_object* v___x_245_; uint8_t v___x_246_; lean_object* v___x_247_; lean_object* v___x_248_; lean_object* v___x_249_; lean_object* v___x_250_; lean_object* v___x_251_; lean_object* v___x_252_; lean_object* v___x_253_; lean_object* v___x_254_; lean_object* v___x_255_; lean_object* v___x_256_; lean_object* v___x_257_; 
v_quotContext_239_ = lean_ctor_get(v_a_233_, 1);
v_currMacroScope_240_ = lean_ctor_get(v_a_233_, 2);
v_ref_241_ = lean_ctor_get(v_a_233_, 5);
v___x_242_ = lean_unsigned_to_nat(0u);
v___x_243_ = l_Lean_Syntax_getArg(v_x_232_, v___x_242_);
v___x_244_ = lean_unsigned_to_nat(2u);
v___x_245_ = l_Lean_Syntax_getArg(v_x_232_, v___x_244_);
lean_dec(v_x_232_);
v___x_246_ = 0;
v___x_247_ = l_Lean_SourceInfo_fromRef(v_ref_241_, v___x_246_);
v___x_248_ = ((lean_object*)(lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__term_____x5b___x5d__1___closed__8));
v___x_249_ = lean_obj_once(&lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__OreLocalization__term___x2f_u2092____1___closed__1, &lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__OreLocalization__term___x2f_u2092____1___closed__1_once, _init_lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__OreLocalization__term___x2f_u2092____1___closed__1);
v___x_250_ = ((lean_object*)(lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__OreLocalization__term___x2f_u2092____1___closed__2));
lean_inc(v_currMacroScope_240_);
lean_inc(v_quotContext_239_);
v___x_251_ = l_Lean_addMacroScope(v_quotContext_239_, v___x_250_, v_currMacroScope_240_);
v___x_252_ = ((lean_object*)(lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__OreLocalization__term___x2f_u2092____1___closed__5));
lean_inc_n(v___x_247_, 2);
v___x_253_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_253_, 0, v___x_247_);
lean_ctor_set(v___x_253_, 1, v___x_249_);
lean_ctor_set(v___x_253_, 2, v___x_251_);
lean_ctor_set(v___x_253_, 3, v___x_252_);
v___x_254_ = ((lean_object*)(lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__term_____x5b___x5d__1___closed__16));
v___x_255_ = l_Lean_Syntax_node2(v___x_247_, v___x_254_, v___x_243_, v___x_245_);
v___x_256_ = l_Lean_Syntax_node2(v___x_247_, v___x_248_, v___x_253_, v___x_255_);
v___x_257_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_257_, 0, v___x_256_);
lean_ctor_set(v___x_257_, 1, v_a_234_);
return v___x_257_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__OreLocalization__term___x2f_u2092____1___boxed(lean_object* v_x_258_, lean_object* v_a_259_, lean_object* v_a_260_){
_start:
{
lean_object* v_res_261_; 
v_res_261_ = lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__OreLocalization__term___x2f_u2092____1(v_x_258_, v_a_259_, v_a_260_);
lean_dec_ref(v_a_259_);
return v_res_261_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______unexpand__OreLocalization__oreDiv__1(lean_object* v_x_265_, lean_object* v_a_266_, lean_object* v_a_267_){
_start:
{
lean_object* v___x_268_; uint8_t v___x_269_; 
v___x_268_ = ((lean_object*)(lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__term_____x5b___x5d__1___closed__8));
lean_inc(v_x_265_);
v___x_269_ = l_Lean_Syntax_isOfKind(v_x_265_, v___x_268_);
if (v___x_269_ == 0)
{
lean_object* v___x_270_; lean_object* v___x_271_; 
lean_dec(v_x_265_);
v___x_270_ = lean_box(0);
v___x_271_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_271_, 0, v___x_270_);
lean_ctor_set(v___x_271_, 1, v_a_267_);
return v___x_271_;
}
else
{
lean_object* v___x_272_; lean_object* v___x_273_; lean_object* v___x_274_; uint8_t v___x_275_; 
v___x_272_ = lean_unsigned_to_nat(0u);
v___x_273_ = l_Lean_Syntax_getArg(v_x_265_, v___x_272_);
v___x_274_ = ((lean_object*)(lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______unexpand__OreLocalization__oreDiv__1___closed__1));
lean_inc(v___x_273_);
v___x_275_ = l_Lean_Syntax_isOfKind(v___x_273_, v___x_274_);
if (v___x_275_ == 0)
{
lean_object* v___x_276_; lean_object* v___x_277_; 
lean_dec(v___x_273_);
lean_dec(v_x_265_);
v___x_276_ = lean_box(0);
v___x_277_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_277_, 0, v___x_276_);
lean_ctor_set(v___x_277_, 1, v_a_267_);
return v___x_277_;
}
else
{
lean_object* v___x_278_; lean_object* v___x_279_; lean_object* v___x_280_; uint8_t v___x_281_; 
v___x_278_ = lean_unsigned_to_nat(1u);
v___x_279_ = l_Lean_Syntax_getArg(v_x_265_, v___x_278_);
lean_dec(v_x_265_);
v___x_280_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_279_);
v___x_281_ = l_Lean_Syntax_matchesNull(v___x_279_, v___x_280_);
if (v___x_281_ == 0)
{
lean_object* v___x_282_; lean_object* v___x_283_; 
lean_dec(v___x_279_);
lean_dec(v___x_273_);
v___x_282_ = lean_box(0);
v___x_283_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_283_, 0, v___x_282_);
lean_ctor_set(v___x_283_, 1, v_a_267_);
return v___x_283_;
}
else
{
lean_object* v___x_284_; lean_object* v___x_285_; lean_object* v_ref_286_; uint8_t v___x_287_; lean_object* v___x_288_; lean_object* v___x_289_; lean_object* v___x_290_; lean_object* v___x_291_; lean_object* v___x_292_; lean_object* v___x_293_; 
v___x_284_ = l_Lean_Syntax_getArg(v___x_279_, v___x_272_);
v___x_285_ = l_Lean_Syntax_getArg(v___x_279_, v___x_278_);
lean_dec(v___x_279_);
v_ref_286_ = l_Lean_replaceRef(v___x_273_, v_a_266_);
lean_dec(v___x_273_);
v___x_287_ = 0;
v___x_288_ = l_Lean_SourceInfo_fromRef(v_ref_286_, v___x_287_);
lean_dec(v_ref_286_);
v___x_289_ = ((lean_object*)(lp_mathlib_OreLocalization_term___x2f_u2092___00__closed__1));
v___x_290_ = ((lean_object*)(lp_mathlib_OreLocalization_term___x2f_u2092___00__closed__2));
lean_inc(v___x_288_);
v___x_291_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_291_, 0, v___x_288_);
lean_ctor_set(v___x_291_, 1, v___x_290_);
v___x_292_ = l_Lean_Syntax_node3(v___x_288_, v___x_289_, v___x_284_, v___x_291_, v___x_285_);
v___x_293_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_293_, 0, v___x_292_);
lean_ctor_set(v___x_293_, 1, v_a_267_);
return v___x_293_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______unexpand__OreLocalization__oreDiv__1___boxed(lean_object* v_x_294_, lean_object* v_a_295_, lean_object* v_a_296_){
_start:
{
lean_object* v_res_297_; 
v_res_297_ = lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______unexpand__OreLocalization__oreDiv__1(v_x_294_, v_a_295_, v_a_296_);
lean_dec(v_a_295_);
return v_res_297_;
}
}
static lean_object* _init_lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__OreLocalization__term___x2d_u2092____1___closed__1(void){
_start:
{
lean_object* v___x_318_; lean_object* v___x_319_; 
v___x_318_ = ((lean_object*)(lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__OreLocalization__term___x2d_u2092____1___closed__0));
v___x_319_ = l_String_toRawSubstring_x27(v___x_318_);
return v___x_319_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__OreLocalization__term___x2d_u2092____1(lean_object* v_x_336_, lean_object* v_a_337_, lean_object* v_a_338_){
_start:
{
lean_object* v___x_339_; uint8_t v___x_340_; 
v___x_339_ = ((lean_object*)(lp_mathlib_OreLocalization_term___x2d_u2092___00__closed__1));
lean_inc(v_x_336_);
v___x_340_ = l_Lean_Syntax_isOfKind(v_x_336_, v___x_339_);
if (v___x_340_ == 0)
{
lean_object* v___x_341_; lean_object* v___x_342_; 
lean_dec(v_x_336_);
v___x_341_ = lean_box(1);
v___x_342_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_342_, 0, v___x_341_);
lean_ctor_set(v___x_342_, 1, v_a_338_);
return v___x_342_;
}
else
{
lean_object* v_quotContext_343_; lean_object* v_currMacroScope_344_; lean_object* v_ref_345_; lean_object* v___x_346_; lean_object* v___x_347_; lean_object* v___x_348_; lean_object* v___x_349_; uint8_t v___x_350_; lean_object* v___x_351_; lean_object* v___x_352_; lean_object* v___x_353_; lean_object* v___x_354_; lean_object* v___x_355_; lean_object* v___x_356_; lean_object* v___x_357_; lean_object* v___x_358_; lean_object* v___x_359_; lean_object* v___x_360_; lean_object* v___x_361_; 
v_quotContext_343_ = lean_ctor_get(v_a_337_, 1);
v_currMacroScope_344_ = lean_ctor_get(v_a_337_, 2);
v_ref_345_ = lean_ctor_get(v_a_337_, 5);
v___x_346_ = lean_unsigned_to_nat(0u);
v___x_347_ = l_Lean_Syntax_getArg(v_x_336_, v___x_346_);
v___x_348_ = lean_unsigned_to_nat(2u);
v___x_349_ = l_Lean_Syntax_getArg(v_x_336_, v___x_348_);
lean_dec(v_x_336_);
v___x_350_ = 0;
v___x_351_ = l_Lean_SourceInfo_fromRef(v_ref_345_, v___x_350_);
v___x_352_ = ((lean_object*)(lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__term_____x5b___x5d__1___closed__8));
v___x_353_ = lean_obj_once(&lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__OreLocalization__term___x2d_u2092____1___closed__1, &lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__OreLocalization__term___x2d_u2092____1___closed__1_once, _init_lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__OreLocalization__term___x2d_u2092____1___closed__1);
v___x_354_ = ((lean_object*)(lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__OreLocalization__term___x2d_u2092____1___closed__5));
lean_inc(v_currMacroScope_344_);
lean_inc(v_quotContext_343_);
v___x_355_ = l_Lean_addMacroScope(v_quotContext_343_, v___x_354_, v_currMacroScope_344_);
v___x_356_ = ((lean_object*)(lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__OreLocalization__term___x2d_u2092____1___closed__8));
lean_inc_n(v___x_351_, 2);
v___x_357_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_357_, 0, v___x_351_);
lean_ctor_set(v___x_357_, 1, v___x_353_);
lean_ctor_set(v___x_357_, 2, v___x_355_);
lean_ctor_set(v___x_357_, 3, v___x_356_);
v___x_358_ = ((lean_object*)(lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__term_____x5b___x5d__1___closed__16));
v___x_359_ = l_Lean_Syntax_node2(v___x_351_, v___x_358_, v___x_347_, v___x_349_);
v___x_360_ = l_Lean_Syntax_node2(v___x_351_, v___x_352_, v___x_357_, v___x_359_);
v___x_361_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_361_, 0, v___x_360_);
lean_ctor_set(v___x_361_, 1, v_a_338_);
return v___x_361_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__OreLocalization__term___x2d_u2092____1___boxed(lean_object* v_x_362_, lean_object* v_a_363_, lean_object* v_a_364_){
_start:
{
lean_object* v_res_365_; 
v_res_365_ = lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__OreLocalization__term___x2d_u2092____1(v_x_362_, v_a_363_, v_a_364_);
lean_dec_ref(v_a_363_);
return v_res_365_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______unexpand__AddOreLocalization__oreSub__1(lean_object* v_x_366_, lean_object* v_a_367_, lean_object* v_a_368_){
_start:
{
lean_object* v___x_369_; uint8_t v___x_370_; 
v___x_369_ = ((lean_object*)(lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______macroRules__term_____x5b___x5d__1___closed__8));
lean_inc(v_x_366_);
v___x_370_ = l_Lean_Syntax_isOfKind(v_x_366_, v___x_369_);
if (v___x_370_ == 0)
{
lean_object* v___x_371_; lean_object* v___x_372_; 
lean_dec(v_x_366_);
v___x_371_ = lean_box(0);
v___x_372_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_372_, 0, v___x_371_);
lean_ctor_set(v___x_372_, 1, v_a_368_);
return v___x_372_;
}
else
{
lean_object* v___x_373_; lean_object* v___x_374_; lean_object* v___x_375_; uint8_t v___x_376_; 
v___x_373_ = lean_unsigned_to_nat(0u);
v___x_374_ = l_Lean_Syntax_getArg(v_x_366_, v___x_373_);
v___x_375_ = ((lean_object*)(lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______unexpand__OreLocalization__oreDiv__1___closed__1));
lean_inc(v___x_374_);
v___x_376_ = l_Lean_Syntax_isOfKind(v___x_374_, v___x_375_);
if (v___x_376_ == 0)
{
lean_object* v___x_377_; lean_object* v___x_378_; 
lean_dec(v___x_374_);
lean_dec(v_x_366_);
v___x_377_ = lean_box(0);
v___x_378_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_378_, 0, v___x_377_);
lean_ctor_set(v___x_378_, 1, v_a_368_);
return v___x_378_;
}
else
{
lean_object* v___x_379_; lean_object* v___x_380_; lean_object* v___x_381_; uint8_t v___x_382_; 
v___x_379_ = lean_unsigned_to_nat(1u);
v___x_380_ = l_Lean_Syntax_getArg(v_x_366_, v___x_379_);
lean_dec(v_x_366_);
v___x_381_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_380_);
v___x_382_ = l_Lean_Syntax_matchesNull(v___x_380_, v___x_381_);
if (v___x_382_ == 0)
{
lean_object* v___x_383_; lean_object* v___x_384_; 
lean_dec(v___x_380_);
lean_dec(v___x_374_);
v___x_383_ = lean_box(0);
v___x_384_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_384_, 0, v___x_383_);
lean_ctor_set(v___x_384_, 1, v_a_368_);
return v___x_384_;
}
else
{
lean_object* v___x_385_; lean_object* v___x_386_; lean_object* v_ref_387_; uint8_t v___x_388_; lean_object* v___x_389_; lean_object* v___x_390_; lean_object* v___x_391_; lean_object* v___x_392_; lean_object* v___x_393_; lean_object* v___x_394_; 
v___x_385_ = l_Lean_Syntax_getArg(v___x_380_, v___x_373_);
v___x_386_ = l_Lean_Syntax_getArg(v___x_380_, v___x_379_);
lean_dec(v___x_380_);
v_ref_387_ = l_Lean_replaceRef(v___x_374_, v_a_367_);
lean_dec(v___x_374_);
v___x_388_ = 0;
v___x_389_ = l_Lean_SourceInfo_fromRef(v_ref_387_, v___x_388_);
lean_dec(v_ref_387_);
v___x_390_ = ((lean_object*)(lp_mathlib_OreLocalization_term___x2d_u2092___00__closed__1));
v___x_391_ = ((lean_object*)(lp_mathlib_OreLocalization_term___x2d_u2092___00__closed__2));
lean_inc(v___x_389_);
v___x_392_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_392_, 0, v___x_389_);
lean_ctor_set(v___x_392_, 1, v___x_391_);
v___x_393_ = l_Lean_Syntax_node3(v___x_389_, v___x_390_, v___x_385_, v___x_392_, v___x_386_);
v___x_394_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_394_, 0, v___x_393_);
lean_ctor_set(v___x_394_, 1, v_a_368_);
return v___x_394_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______unexpand__AddOreLocalization__oreSub__1___boxed(lean_object* v_x_395_, lean_object* v_a_396_, lean_object* v_a_397_){
_start:
{
lean_object* v_res_398_; 
v_res_398_ = lp_mathlib_OreLocalization___aux__Mathlib__GroupTheory__OreLocalization__Basic______unexpand__AddOreLocalization__oreSub__1(v_x_395_, v_a_396_, v_a_397_);
lean_dec(v_a_396_);
return v_res_398_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_liftExpand___redArg(lean_object* v_P_399_, lean_object* v_a_400_){
_start:
{
lean_object* v_fst_401_; lean_object* v_snd_402_; lean_object* v___x_403_; 
v_fst_401_ = lean_ctor_get(v_a_400_, 0);
lean_inc(v_fst_401_);
v_snd_402_ = lean_ctor_get(v_a_400_, 1);
lean_inc(v_snd_402_);
lean_dec(v_a_400_);
v___x_403_ = lean_apply_2(v_P_399_, v_fst_401_, v_snd_402_);
return v___x_403_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_liftExpand(lean_object* v_R_404_, lean_object* v_inst_405_, lean_object* v_S_406_, lean_object* v_inst_407_, lean_object* v_X_408_, lean_object* v_inst_409_, lean_object* v_C_410_, lean_object* v_P_411_, lean_object* v_hP_412_, lean_object* v_a_413_){
_start:
{
lean_object* v___x_414_; 
v___x_414_ = lp_mathlib_OreLocalization_liftExpand___redArg(v_P_411_, v_a_413_);
return v___x_414_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_liftExpand___boxed(lean_object* v_R_415_, lean_object* v_inst_416_, lean_object* v_S_417_, lean_object* v_inst_418_, lean_object* v_X_419_, lean_object* v_inst_420_, lean_object* v_C_421_, lean_object* v_P_422_, lean_object* v_hP_423_, lean_object* v_a_424_){
_start:
{
lean_object* v_res_425_; 
v_res_425_ = lp_mathlib_OreLocalization_liftExpand(v_R_415_, v_inst_416_, v_S_417_, v_inst_418_, v_X_419_, v_inst_420_, v_C_421_, v_P_422_, v_hP_423_, v_a_424_);
lean_dec(v_inst_420_);
lean_dec_ref(v_inst_418_);
lean_dec_ref(v_inst_416_);
return v_res_425_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_liftExpand___redArg(lean_object* v_P_426_, lean_object* v_a_427_){
_start:
{
lean_object* v_fst_428_; lean_object* v_snd_429_; lean_object* v___x_430_; 
v_fst_428_ = lean_ctor_get(v_a_427_, 0);
lean_inc(v_fst_428_);
v_snd_429_ = lean_ctor_get(v_a_427_, 1);
lean_inc(v_snd_429_);
lean_dec(v_a_427_);
v___x_430_ = lean_apply_2(v_P_426_, v_fst_428_, v_snd_429_);
return v___x_430_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_liftExpand(lean_object* v_R_431_, lean_object* v_inst_432_, lean_object* v_S_433_, lean_object* v_inst_434_, lean_object* v_X_435_, lean_object* v_inst_436_, lean_object* v_C_437_, lean_object* v_P_438_, lean_object* v_hP_439_, lean_object* v_a_440_){
_start:
{
lean_object* v___x_441_; 
v___x_441_ = lp_mathlib_AddOreLocalization_liftExpand___redArg(v_P_438_, v_a_440_);
return v___x_441_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_liftExpand___boxed(lean_object* v_R_442_, lean_object* v_inst_443_, lean_object* v_S_444_, lean_object* v_inst_445_, lean_object* v_X_446_, lean_object* v_inst_447_, lean_object* v_C_448_, lean_object* v_P_449_, lean_object* v_hP_450_, lean_object* v_a_451_){
_start:
{
lean_object* v_res_452_; 
v_res_452_ = lp_mathlib_AddOreLocalization_liftExpand(v_R_442_, v_inst_443_, v_S_444_, v_inst_445_, v_X_446_, v_inst_447_, v_C_448_, v_P_449_, v_hP_450_, v_a_451_);
lean_dec(v_inst_447_);
lean_dec_ref(v_inst_445_);
lean_dec_ref(v_inst_443_);
return v_res_452_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_lift_u2082Expand___redArg___lam__0(lean_object* v_P_453_, lean_object* v_r_u2081_454_, lean_object* v_s_u2081_455_, lean_object* v___y_456_){
_start:
{
lean_object* v___x_457_; lean_object* v___x_458_; 
v___x_457_ = lean_apply_2(v_P_453_, v_r_u2081_454_, v_s_u2081_455_);
v___x_458_ = lp_mathlib_OreLocalization_liftExpand___redArg(v___x_457_, v___y_456_);
return v___x_458_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_lift_u2082Expand___redArg(lean_object* v_P_459_, lean_object* v_a_460_, lean_object* v_a_461_){
_start:
{
lean_object* v___f_462_; lean_object* v___x_9__overap_463_; lean_object* v___x_464_; 
v___f_462_ = lean_alloc_closure((void*)(lp_mathlib_OreLocalization_lift_u2082Expand___redArg___lam__0), 4, 1);
lean_closure_set(v___f_462_, 0, v_P_459_);
v___x_9__overap_463_ = lp_mathlib_OreLocalization_liftExpand___redArg(v___f_462_, v_a_460_);
v___x_464_ = lean_apply_1(v___x_9__overap_463_, v_a_461_);
return v___x_464_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_lift_u2082Expand(lean_object* v_R_465_, lean_object* v_inst_466_, lean_object* v_S_467_, lean_object* v_inst_468_, lean_object* v_X_469_, lean_object* v_inst_470_, lean_object* v_C_471_, lean_object* v_P_472_, lean_object* v_hP_473_, lean_object* v_a_474_, lean_object* v_a_475_){
_start:
{
lean_object* v___x_476_; 
v___x_476_ = lp_mathlib_OreLocalization_lift_u2082Expand___redArg(v_P_472_, v_a_474_, v_a_475_);
return v___x_476_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_lift_u2082Expand___boxed(lean_object* v_R_477_, lean_object* v_inst_478_, lean_object* v_S_479_, lean_object* v_inst_480_, lean_object* v_X_481_, lean_object* v_inst_482_, lean_object* v_C_483_, lean_object* v_P_484_, lean_object* v_hP_485_, lean_object* v_a_486_, lean_object* v_a_487_){
_start:
{
lean_object* v_res_488_; 
v_res_488_ = lp_mathlib_OreLocalization_lift_u2082Expand(v_R_477_, v_inst_478_, v_S_479_, v_inst_480_, v_X_481_, v_inst_482_, v_C_483_, v_P_484_, v_hP_485_, v_a_486_, v_a_487_);
lean_dec(v_inst_482_);
lean_dec_ref(v_inst_480_);
lean_dec_ref(v_inst_478_);
return v_res_488_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_lift_u2082Expand___redArg___lam__0(lean_object* v_P_489_, lean_object* v_r_u2081_490_, lean_object* v_s_u2081_491_, lean_object* v___y_492_){
_start:
{
lean_object* v___x_493_; lean_object* v___x_494_; 
v___x_493_ = lean_apply_2(v_P_489_, v_r_u2081_490_, v_s_u2081_491_);
v___x_494_ = lp_mathlib_AddOreLocalization_liftExpand___redArg(v___x_493_, v___y_492_);
return v___x_494_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_lift_u2082Expand___redArg(lean_object* v_P_495_, lean_object* v_a_496_, lean_object* v_a_497_){
_start:
{
lean_object* v___f_498_; lean_object* v___x_9__overap_499_; lean_object* v___x_500_; 
v___f_498_ = lean_alloc_closure((void*)(lp_mathlib_AddOreLocalization_lift_u2082Expand___redArg___lam__0), 4, 1);
lean_closure_set(v___f_498_, 0, v_P_495_);
v___x_9__overap_499_ = lp_mathlib_AddOreLocalization_liftExpand___redArg(v___f_498_, v_a_496_);
v___x_500_ = lean_apply_1(v___x_9__overap_499_, v_a_497_);
return v___x_500_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_lift_u2082Expand(lean_object* v_R_501_, lean_object* v_inst_502_, lean_object* v_S_503_, lean_object* v_inst_504_, lean_object* v_X_505_, lean_object* v_inst_506_, lean_object* v_C_507_, lean_object* v_P_508_, lean_object* v_hP_509_, lean_object* v_a_510_, lean_object* v_a_511_){
_start:
{
lean_object* v___x_512_; 
v___x_512_ = lp_mathlib_AddOreLocalization_lift_u2082Expand___redArg(v_P_508_, v_a_510_, v_a_511_);
return v___x_512_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_lift_u2082Expand___boxed(lean_object* v_R_513_, lean_object* v_inst_514_, lean_object* v_S_515_, lean_object* v_inst_516_, lean_object* v_X_517_, lean_object* v_inst_518_, lean_object* v_C_519_, lean_object* v_P_520_, lean_object* v_hP_521_, lean_object* v_a_522_, lean_object* v_a_523_){
_start:
{
lean_object* v_res_524_; 
v_res_524_ = lp_mathlib_AddOreLocalization_lift_u2082Expand(v_R_513_, v_inst_514_, v_S_515_, v_inst_516_, v_X_517_, v_inst_518_, v_C_519_, v_P_520_, v_hP_521_, v_a_522_, v_a_523_);
lean_dec(v_inst_518_);
lean_dec_ref(v_inst_516_);
lean_dec_ref(v_inst_514_);
return v_res_524_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_smul___redArg___lam__0(lean_object* v___x_525_, lean_object* v_inst_526_, lean_object* v_x1_527_, lean_object* v_inst_528_, lean_object* v_x2_529_, lean_object* v___y_530_, lean_object* v___y_531_){
_start:
{
lean_object* v___x_532_; lean_object* v_toMul_533_; lean_object* v___x_535_; uint8_t v_isShared_536_; uint8_t v_isSharedCheck_544_; 
v___x_532_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_525_);
v_toMul_533_ = lean_ctor_get(v___x_532_, 1);
v_isSharedCheck_544_ = !lean_is_exclusive(v___x_532_);
if (v_isSharedCheck_544_ == 0)
{
lean_object* v_unused_545_; 
v_unused_545_ = lean_ctor_get(v___x_532_, 0);
lean_dec(v_unused_545_);
v___x_535_ = v___x_532_;
v_isShared_536_ = v_isSharedCheck_544_;
goto v_resetjp_534_;
}
else
{
lean_inc(v_toMul_533_);
lean_dec(v___x_532_);
v___x_535_ = lean_box(0);
v_isShared_536_ = v_isSharedCheck_544_;
goto v_resetjp_534_;
}
v_resetjp_534_:
{
lean_object* v___x_537_; lean_object* v___x_538_; lean_object* v___x_539_; lean_object* v___x_540_; lean_object* v___x_542_; 
lean_inc(v___y_531_);
lean_inc(v_x1_527_);
lean_inc_ref(v_inst_526_);
v___x_537_ = lp_mathlib_OreLocalization_oreNum___redArg(v_inst_526_, v_x1_527_, v___y_531_);
v___x_538_ = lean_apply_2(v_inst_528_, v___x_537_, v___y_530_);
v___x_539_ = lp_mathlib_OreLocalization_oreDenom___redArg(v_inst_526_, v_x1_527_, v___y_531_);
v___x_540_ = lean_apply_2(v_toMul_533_, v___x_539_, v_x2_529_);
if (v_isShared_536_ == 0)
{
lean_ctor_set(v___x_535_, 1, v___x_540_);
lean_ctor_set(v___x_535_, 0, v___x_538_);
v___x_542_ = v___x_535_;
goto v_reusejp_541_;
}
else
{
lean_object* v_reuseFailAlloc_543_; 
v_reuseFailAlloc_543_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_543_, 0, v___x_538_);
lean_ctor_set(v_reuseFailAlloc_543_, 1, v___x_540_);
v___x_542_ = v_reuseFailAlloc_543_;
goto v_reusejp_541_;
}
v_reusejp_541_:
{
return v___x_542_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_smul___redArg___lam__1(lean_object* v___x_546_, lean_object* v_inst_547_, lean_object* v_inst_548_, lean_object* v_x_549_, lean_object* v_x1_550_, lean_object* v_x2_551_){
_start:
{
lean_object* v___f_552_; lean_object* v___x_553_; 
v___f_552_ = lean_alloc_closure((void*)(lp_mathlib_OreLocalization_smul___redArg___lam__0), 7, 5);
lean_closure_set(v___f_552_, 0, v___x_546_);
lean_closure_set(v___f_552_, 1, v_inst_547_);
lean_closure_set(v___f_552_, 2, v_x1_550_);
lean_closure_set(v___f_552_, 3, v_inst_548_);
lean_closure_set(v___f_552_, 4, v_x2_551_);
v___x_553_ = lp_mathlib_OreLocalization_liftExpand___redArg(v___f_552_, v_x_549_);
return v___x_553_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_smul___redArg(lean_object* v_inst_554_, lean_object* v_inst_555_, lean_object* v_inst_556_, lean_object* v_y_557_, lean_object* v_x_558_){
_start:
{
lean_object* v___x_559_; lean_object* v___f_560_; lean_object* v___x_561_; 
v___x_559_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_554_);
v___f_560_ = lean_alloc_closure((void*)(lp_mathlib_OreLocalization_smul___redArg___lam__1), 6, 4);
lean_closure_set(v___f_560_, 0, v___x_559_);
lean_closure_set(v___f_560_, 1, v_inst_555_);
lean_closure_set(v___f_560_, 2, v_inst_556_);
lean_closure_set(v___f_560_, 3, v_x_558_);
v___x_561_ = lp_mathlib_OreLocalization_liftExpand___redArg(v___f_560_, v_y_557_);
return v___x_561_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_smul___redArg___boxed(lean_object* v_inst_562_, lean_object* v_inst_563_, lean_object* v_inst_564_, lean_object* v_y_565_, lean_object* v_x_566_){
_start:
{
lean_object* v_res_567_; 
v_res_567_ = lp_mathlib_OreLocalization_smul___redArg(v_inst_562_, v_inst_563_, v_inst_564_, v_y_565_, v_x_566_);
lean_dec_ref(v_inst_562_);
return v_res_567_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_smul(lean_object* v_R_568_, lean_object* v_inst_569_, lean_object* v_S_570_, lean_object* v_inst_571_, lean_object* v_X_572_, lean_object* v_inst_573_, lean_object* v_y_574_, lean_object* v_x_575_){
_start:
{
lean_object* v___x_576_; lean_object* v___f_577_; lean_object* v___x_578_; 
v___x_576_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_569_);
v___f_577_ = lean_alloc_closure((void*)(lp_mathlib_OreLocalization_smul___redArg___lam__1), 6, 4);
lean_closure_set(v___f_577_, 0, v___x_576_);
lean_closure_set(v___f_577_, 1, v_inst_571_);
lean_closure_set(v___f_577_, 2, v_inst_573_);
lean_closure_set(v___f_577_, 3, v_x_575_);
v___x_578_ = lp_mathlib_OreLocalization_liftExpand___redArg(v___f_577_, v_y_574_);
return v___x_578_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_smul___boxed(lean_object* v_R_579_, lean_object* v_inst_580_, lean_object* v_S_581_, lean_object* v_inst_582_, lean_object* v_X_583_, lean_object* v_inst_584_, lean_object* v_y_585_, lean_object* v_x_586_){
_start:
{
lean_object* v_res_587_; 
v_res_587_ = lp_mathlib_OreLocalization_smul(v_R_579_, v_inst_580_, v_S_581_, v_inst_582_, v_X_583_, v_inst_584_, v_y_585_, v_x_586_);
lean_dec_ref(v_inst_580_);
return v_res_587_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_vadd___redArg___lam__0(lean_object* v___x_588_, lean_object* v_inst_589_, lean_object* v_x1_590_, lean_object* v_inst_591_, lean_object* v_x2_592_, lean_object* v___y_593_, lean_object* v___y_594_){
_start:
{
lean_object* v___x_595_; lean_object* v_toAdd_596_; lean_object* v___x_598_; uint8_t v_isShared_599_; uint8_t v_isSharedCheck_607_; 
v___x_595_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_588_);
v_toAdd_596_ = lean_ctor_get(v___x_595_, 1);
v_isSharedCheck_607_ = !lean_is_exclusive(v___x_595_);
if (v_isSharedCheck_607_ == 0)
{
lean_object* v_unused_608_; 
v_unused_608_ = lean_ctor_get(v___x_595_, 0);
lean_dec(v_unused_608_);
v___x_598_ = v___x_595_;
v_isShared_599_ = v_isSharedCheck_607_;
goto v_resetjp_597_;
}
else
{
lean_inc(v_toAdd_596_);
lean_dec(v___x_595_);
v___x_598_ = lean_box(0);
v_isShared_599_ = v_isSharedCheck_607_;
goto v_resetjp_597_;
}
v_resetjp_597_:
{
lean_object* v___x_600_; lean_object* v___x_601_; lean_object* v___x_602_; lean_object* v___x_603_; lean_object* v___x_605_; 
lean_inc(v___y_594_);
lean_inc(v_x1_590_);
lean_inc_ref(v_inst_589_);
v___x_600_ = lp_mathlib_AddOreLocalization_oreMin___redArg(v_inst_589_, v_x1_590_, v___y_594_);
v___x_601_ = lean_apply_2(v_inst_591_, v___x_600_, v___y_593_);
v___x_602_ = lp_mathlib_AddOreLocalization_oreSubtra___redArg(v_inst_589_, v_x1_590_, v___y_594_);
v___x_603_ = lean_apply_2(v_toAdd_596_, v___x_602_, v_x2_592_);
if (v_isShared_599_ == 0)
{
lean_ctor_set(v___x_598_, 1, v___x_603_);
lean_ctor_set(v___x_598_, 0, v___x_601_);
v___x_605_ = v___x_598_;
goto v_reusejp_604_;
}
else
{
lean_object* v_reuseFailAlloc_606_; 
v_reuseFailAlloc_606_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_606_, 0, v___x_601_);
lean_ctor_set(v_reuseFailAlloc_606_, 1, v___x_603_);
v___x_605_ = v_reuseFailAlloc_606_;
goto v_reusejp_604_;
}
v_reusejp_604_:
{
return v___x_605_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_vadd___redArg___lam__1(lean_object* v___x_609_, lean_object* v_inst_610_, lean_object* v_inst_611_, lean_object* v_x_612_, lean_object* v_x1_613_, lean_object* v_x2_614_){
_start:
{
lean_object* v___f_615_; lean_object* v___x_616_; 
v___f_615_ = lean_alloc_closure((void*)(lp_mathlib_AddOreLocalization_vadd___redArg___lam__0), 7, 5);
lean_closure_set(v___f_615_, 0, v___x_609_);
lean_closure_set(v___f_615_, 1, v_inst_610_);
lean_closure_set(v___f_615_, 2, v_x1_613_);
lean_closure_set(v___f_615_, 3, v_inst_611_);
lean_closure_set(v___f_615_, 4, v_x2_614_);
v___x_616_ = lp_mathlib_AddOreLocalization_liftExpand___redArg(v___f_615_, v_x_612_);
return v___x_616_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_vadd___redArg(lean_object* v_inst_617_, lean_object* v_inst_618_, lean_object* v_inst_619_, lean_object* v_y_620_, lean_object* v_x_621_){
_start:
{
lean_object* v___x_622_; lean_object* v___f_623_; lean_object* v___x_624_; 
v___x_622_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_617_);
v___f_623_ = lean_alloc_closure((void*)(lp_mathlib_AddOreLocalization_vadd___redArg___lam__1), 6, 4);
lean_closure_set(v___f_623_, 0, v___x_622_);
lean_closure_set(v___f_623_, 1, v_inst_618_);
lean_closure_set(v___f_623_, 2, v_inst_619_);
lean_closure_set(v___f_623_, 3, v_x_621_);
v___x_624_ = lp_mathlib_AddOreLocalization_liftExpand___redArg(v___f_623_, v_y_620_);
return v___x_624_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_vadd___redArg___boxed(lean_object* v_inst_625_, lean_object* v_inst_626_, lean_object* v_inst_627_, lean_object* v_y_628_, lean_object* v_x_629_){
_start:
{
lean_object* v_res_630_; 
v_res_630_ = lp_mathlib_AddOreLocalization_vadd___redArg(v_inst_625_, v_inst_626_, v_inst_627_, v_y_628_, v_x_629_);
lean_dec_ref(v_inst_625_);
return v_res_630_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_vadd(lean_object* v_R_631_, lean_object* v_inst_632_, lean_object* v_S_633_, lean_object* v_inst_634_, lean_object* v_X_635_, lean_object* v_inst_636_, lean_object* v_y_637_, lean_object* v_x_638_){
_start:
{
lean_object* v___x_639_; 
v___x_639_ = lp_mathlib_AddOreLocalization_vadd___redArg(v_inst_632_, v_inst_634_, v_inst_636_, v_y_637_, v_x_638_);
return v___x_639_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_vadd___boxed(lean_object* v_R_640_, lean_object* v_inst_641_, lean_object* v_S_642_, lean_object* v_inst_643_, lean_object* v_X_644_, lean_object* v_inst_645_, lean_object* v_y_646_, lean_object* v_x_647_){
_start:
{
lean_object* v_res_648_; 
v_res_648_ = lp_mathlib_AddOreLocalization_vadd(v_R_640_, v_inst_641_, v_S_642_, v_inst_643_, v_X_644_, v_inst_645_, v_y_646_, v_x_647_);
lean_dec_ref(v_inst_641_);
return v_res_648_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instSMul___redArg(lean_object* v_inst_649_, lean_object* v_S_650_, lean_object* v_inst_651_, lean_object* v_inst_652_){
_start:
{
lean_object* v___x_653_; 
v___x_653_ = lean_alloc_closure((void*)(lp_mathlib_OreLocalization_smul___boxed), 8, 6);
lean_closure_set(v___x_653_, 0, lean_box(0));
lean_closure_set(v___x_653_, 1, v_inst_649_);
lean_closure_set(v___x_653_, 2, v_S_650_);
lean_closure_set(v___x_653_, 3, v_inst_651_);
lean_closure_set(v___x_653_, 4, lean_box(0));
lean_closure_set(v___x_653_, 5, v_inst_652_);
return v___x_653_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instSMul(lean_object* v_R_654_, lean_object* v_inst_655_, lean_object* v_S_656_, lean_object* v_inst_657_, lean_object* v_X_658_, lean_object* v_inst_659_){
_start:
{
lean_object* v___x_660_; 
v___x_660_ = lean_alloc_closure((void*)(lp_mathlib_OreLocalization_smul___boxed), 8, 6);
lean_closure_set(v___x_660_, 0, lean_box(0));
lean_closure_set(v___x_660_, 1, v_inst_655_);
lean_closure_set(v___x_660_, 2, v_S_656_);
lean_closure_set(v___x_660_, 3, v_inst_657_);
lean_closure_set(v___x_660_, 4, lean_box(0));
lean_closure_set(v___x_660_, 5, v_inst_659_);
return v___x_660_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_instVAdd___redArg(lean_object* v_inst_661_, lean_object* v_S_662_, lean_object* v_inst_663_, lean_object* v_inst_664_){
_start:
{
lean_object* v___x_665_; 
v___x_665_ = lean_alloc_closure((void*)(lp_mathlib_AddOreLocalization_vadd___boxed), 8, 6);
lean_closure_set(v___x_665_, 0, lean_box(0));
lean_closure_set(v___x_665_, 1, v_inst_661_);
lean_closure_set(v___x_665_, 2, v_S_662_);
lean_closure_set(v___x_665_, 3, v_inst_663_);
lean_closure_set(v___x_665_, 4, lean_box(0));
lean_closure_set(v___x_665_, 5, v_inst_664_);
return v___x_665_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_instVAdd(lean_object* v_R_666_, lean_object* v_inst_667_, lean_object* v_S_668_, lean_object* v_inst_669_, lean_object* v_X_670_, lean_object* v_inst_671_){
_start:
{
lean_object* v___x_672_; 
v___x_672_ = lean_alloc_closure((void*)(lp_mathlib_AddOreLocalization_vadd___boxed), 8, 6);
lean_closure_set(v___x_672_, 0, lean_box(0));
lean_closure_set(v___x_672_, 1, v_inst_667_);
lean_closure_set(v___x_672_, 2, v_S_668_);
lean_closure_set(v___x_672_, 3, v_inst_669_);
lean_closure_set(v___x_672_, 4, lean_box(0));
lean_closure_set(v___x_672_, 5, v_inst_671_);
return v___x_672_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instMul___redArg(lean_object* v_inst_673_, lean_object* v_S_674_, lean_object* v_inst_675_){
_start:
{
lean_object* v___x_676_; lean_object* v___x_677_; 
v___x_676_ = lp_mathlib_Monoid_toMulAction___redArg(v_inst_673_);
v___x_677_ = lean_alloc_closure((void*)(lp_mathlib_OreLocalization_smul___boxed), 8, 6);
lean_closure_set(v___x_677_, 0, lean_box(0));
lean_closure_set(v___x_677_, 1, v_inst_673_);
lean_closure_set(v___x_677_, 2, v_S_674_);
lean_closure_set(v___x_677_, 3, v_inst_675_);
lean_closure_set(v___x_677_, 4, lean_box(0));
lean_closure_set(v___x_677_, 5, v___x_676_);
return v___x_677_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instMul(lean_object* v_R_678_, lean_object* v_inst_679_, lean_object* v_S_680_, lean_object* v_inst_681_){
_start:
{
lean_object* v___x_682_; 
v___x_682_ = lp_mathlib_OreLocalization_instMul___redArg(v_inst_679_, v_S_680_, v_inst_681_);
return v___x_682_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_instAdd___redArg(lean_object* v_inst_683_, lean_object* v_S_684_, lean_object* v_inst_685_){
_start:
{
lean_object* v___x_686_; lean_object* v___x_687_; 
v___x_686_ = lp_mathlib_AddMonoid_toAddAction___redArg(v_inst_683_);
v___x_687_ = lean_alloc_closure((void*)(lp_mathlib_AddOreLocalization_vadd___boxed), 8, 6);
lean_closure_set(v___x_687_, 0, lean_box(0));
lean_closure_set(v___x_687_, 1, v_inst_683_);
lean_closure_set(v___x_687_, 2, v_S_684_);
lean_closure_set(v___x_687_, 3, v_inst_685_);
lean_closure_set(v___x_687_, 4, lean_box(0));
lean_closure_set(v___x_687_, 5, v___x_686_);
return v___x_687_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_instAdd(lean_object* v_R_688_, lean_object* v_inst_689_, lean_object* v_S_690_, lean_object* v_inst_691_){
_start:
{
lean_object* v___x_692_; 
v___x_692_ = lp_mathlib_AddOreLocalization_instAdd___redArg(v_inst_689_, v_S_690_, v_inst_691_);
return v___x_692_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_oreDivSMulChar_x27___redArg(lean_object* v_inst_693_, lean_object* v_r_u2081_694_, lean_object* v_s_u2082_695_){
_start:
{
lean_object* v___x_696_; lean_object* v___x_697_; lean_object* v___x_698_; lean_object* v___x_699_; 
lean_inc(v_s_u2082_695_);
lean_inc(v_r_u2081_694_);
lean_inc_ref(v_inst_693_);
v___x_696_ = lp_mathlib_OreLocalization_oreNum___redArg(v_inst_693_, v_r_u2081_694_, v_s_u2082_695_);
v___x_697_ = lp_mathlib_OreLocalization_oreDenom___redArg(v_inst_693_, v_r_u2081_694_, v_s_u2082_695_);
v___x_698_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_698_, 0, v___x_697_);
lean_ctor_set(v___x_698_, 1, lean_box(0));
v___x_699_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_699_, 0, v___x_696_);
lean_ctor_set(v___x_699_, 1, v___x_698_);
return v___x_699_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_oreDivSMulChar_x27(lean_object* v_R_700_, lean_object* v_inst_701_, lean_object* v_S_702_, lean_object* v_inst_703_, lean_object* v_X_704_, lean_object* v_inst_705_, lean_object* v_r_u2081_706_, lean_object* v_r_u2082_707_, lean_object* v_s_u2081_708_, lean_object* v_s_u2082_709_){
_start:
{
lean_object* v___x_710_; 
v___x_710_ = lp_mathlib_OreLocalization_oreDivSMulChar_x27___redArg(v_inst_703_, v_r_u2081_706_, v_s_u2082_709_);
return v___x_710_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_oreDivSMulChar_x27___boxed(lean_object* v_R_711_, lean_object* v_inst_712_, lean_object* v_S_713_, lean_object* v_inst_714_, lean_object* v_X_715_, lean_object* v_inst_716_, lean_object* v_r_u2081_717_, lean_object* v_r_u2082_718_, lean_object* v_s_u2081_719_, lean_object* v_s_u2082_720_){
_start:
{
lean_object* v_res_721_; 
v_res_721_ = lp_mathlib_OreLocalization_oreDivSMulChar_x27(v_R_711_, v_inst_712_, v_S_713_, v_inst_714_, v_X_715_, v_inst_716_, v_r_u2081_717_, v_r_u2082_718_, v_s_u2081_719_, v_s_u2082_720_);
lean_dec(v_s_u2081_719_);
lean_dec(v_r_u2082_718_);
lean_dec(v_inst_716_);
lean_dec_ref(v_inst_712_);
return v_res_721_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_oreSubVAddChar_x27___redArg(lean_object* v_inst_722_, lean_object* v_r_u2081_723_, lean_object* v_s_u2082_724_){
_start:
{
lean_object* v___x_725_; lean_object* v___x_726_; lean_object* v___x_727_; lean_object* v___x_728_; 
lean_inc(v_s_u2082_724_);
lean_inc(v_r_u2081_723_);
lean_inc_ref(v_inst_722_);
v___x_725_ = lp_mathlib_AddOreLocalization_oreMin___redArg(v_inst_722_, v_r_u2081_723_, v_s_u2082_724_);
v___x_726_ = lp_mathlib_AddOreLocalization_oreSubtra___redArg(v_inst_722_, v_r_u2081_723_, v_s_u2082_724_);
v___x_727_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_727_, 0, v___x_726_);
lean_ctor_set(v___x_727_, 1, lean_box(0));
v___x_728_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_728_, 0, v___x_725_);
lean_ctor_set(v___x_728_, 1, v___x_727_);
return v___x_728_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_oreSubVAddChar_x27(lean_object* v_R_729_, lean_object* v_inst_730_, lean_object* v_S_731_, lean_object* v_inst_732_, lean_object* v_X_733_, lean_object* v_inst_734_, lean_object* v_r_u2081_735_, lean_object* v_r_u2082_736_, lean_object* v_s_u2081_737_, lean_object* v_s_u2082_738_){
_start:
{
lean_object* v___x_739_; 
v___x_739_ = lp_mathlib_AddOreLocalization_oreSubVAddChar_x27___redArg(v_inst_732_, v_r_u2081_735_, v_s_u2082_738_);
return v___x_739_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_oreSubVAddChar_x27___boxed(lean_object* v_R_740_, lean_object* v_inst_741_, lean_object* v_S_742_, lean_object* v_inst_743_, lean_object* v_X_744_, lean_object* v_inst_745_, lean_object* v_r_u2081_746_, lean_object* v_r_u2082_747_, lean_object* v_s_u2081_748_, lean_object* v_s_u2082_749_){
_start:
{
lean_object* v_res_750_; 
v_res_750_ = lp_mathlib_AddOreLocalization_oreSubVAddChar_x27(v_R_740_, v_inst_741_, v_S_742_, v_inst_743_, v_X_744_, v_inst_745_, v_r_u2081_746_, v_r_u2082_747_, v_s_u2081_748_, v_s_u2082_749_);
lean_dec(v_s_u2081_748_);
lean_dec(v_r_u2082_747_);
lean_dec(v_inst_745_);
lean_dec_ref(v_inst_741_);
return v_res_750_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_oreDivMulChar_x27___redArg(lean_object* v_inst_751_, lean_object* v_r_u2081_752_, lean_object* v_s_u2082_753_){
_start:
{
lean_object* v___x_754_; lean_object* v___x_755_; lean_object* v___x_756_; lean_object* v___x_757_; 
lean_inc(v_s_u2082_753_);
lean_inc(v_r_u2081_752_);
lean_inc_ref(v_inst_751_);
v___x_754_ = lp_mathlib_OreLocalization_oreNum___redArg(v_inst_751_, v_r_u2081_752_, v_s_u2082_753_);
v___x_755_ = lp_mathlib_OreLocalization_oreDenom___redArg(v_inst_751_, v_r_u2081_752_, v_s_u2082_753_);
v___x_756_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_756_, 0, v___x_755_);
lean_ctor_set(v___x_756_, 1, lean_box(0));
v___x_757_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_757_, 0, v___x_754_);
lean_ctor_set(v___x_757_, 1, v___x_756_);
return v___x_757_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_oreDivMulChar_x27(lean_object* v_R_758_, lean_object* v_inst_759_, lean_object* v_S_760_, lean_object* v_inst_761_, lean_object* v_r_u2081_762_, lean_object* v_r_u2082_763_, lean_object* v_s_u2081_764_, lean_object* v_s_u2082_765_){
_start:
{
lean_object* v___x_766_; 
v___x_766_ = lp_mathlib_OreLocalization_oreDivMulChar_x27___redArg(v_inst_761_, v_r_u2081_762_, v_s_u2082_765_);
return v___x_766_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_oreDivMulChar_x27___boxed(lean_object* v_R_767_, lean_object* v_inst_768_, lean_object* v_S_769_, lean_object* v_inst_770_, lean_object* v_r_u2081_771_, lean_object* v_r_u2082_772_, lean_object* v_s_u2081_773_, lean_object* v_s_u2082_774_){
_start:
{
lean_object* v_res_775_; 
v_res_775_ = lp_mathlib_OreLocalization_oreDivMulChar_x27(v_R_767_, v_inst_768_, v_S_769_, v_inst_770_, v_r_u2081_771_, v_r_u2082_772_, v_s_u2081_773_, v_s_u2082_774_);
lean_dec(v_s_u2081_773_);
lean_dec(v_r_u2082_772_);
lean_dec_ref(v_inst_768_);
return v_res_775_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_oreSubAddChar_x27___redArg(lean_object* v_inst_776_, lean_object* v_r_u2081_777_, lean_object* v_s_u2082_778_){
_start:
{
lean_object* v___x_779_; lean_object* v___x_780_; lean_object* v___x_781_; lean_object* v___x_782_; 
lean_inc(v_s_u2082_778_);
lean_inc(v_r_u2081_777_);
lean_inc_ref(v_inst_776_);
v___x_779_ = lp_mathlib_AddOreLocalization_oreMin___redArg(v_inst_776_, v_r_u2081_777_, v_s_u2082_778_);
v___x_780_ = lp_mathlib_AddOreLocalization_oreSubtra___redArg(v_inst_776_, v_r_u2081_777_, v_s_u2082_778_);
v___x_781_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_781_, 0, v___x_780_);
lean_ctor_set(v___x_781_, 1, lean_box(0));
v___x_782_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_782_, 0, v___x_779_);
lean_ctor_set(v___x_782_, 1, v___x_781_);
return v___x_782_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_oreSubAddChar_x27(lean_object* v_R_783_, lean_object* v_inst_784_, lean_object* v_S_785_, lean_object* v_inst_786_, lean_object* v_r_u2081_787_, lean_object* v_r_u2082_788_, lean_object* v_s_u2081_789_, lean_object* v_s_u2082_790_){
_start:
{
lean_object* v___x_791_; 
v___x_791_ = lp_mathlib_AddOreLocalization_oreSubAddChar_x27___redArg(v_inst_786_, v_r_u2081_787_, v_s_u2082_790_);
return v___x_791_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_oreSubAddChar_x27___boxed(lean_object* v_R_792_, lean_object* v_inst_793_, lean_object* v_S_794_, lean_object* v_inst_795_, lean_object* v_r_u2081_796_, lean_object* v_r_u2082_797_, lean_object* v_s_u2081_798_, lean_object* v_s_u2082_799_){
_start:
{
lean_object* v_res_800_; 
v_res_800_ = lp_mathlib_AddOreLocalization_oreSubAddChar_x27(v_R_792_, v_inst_793_, v_S_794_, v_inst_795_, v_r_u2081_796_, v_r_u2082_797_, v_s_u2081_798_, v_s_u2082_799_);
lean_dec(v_s_u2081_798_);
lean_dec(v_r_u2082_797_);
lean_dec_ref(v_inst_793_);
return v_res_800_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_one___redArg(lean_object* v_inst_801_, lean_object* v_inst_802_){
_start:
{
lean_object* v___x_803_; lean_object* v___x_804_; lean_object* v_toOne_805_; lean_object* v___x_807_; uint8_t v_isShared_808_; uint8_t v_isSharedCheck_812_; 
v___x_803_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_801_);
v___x_804_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_803_);
v_toOne_805_ = lean_ctor_get(v___x_804_, 0);
v_isSharedCheck_812_ = !lean_is_exclusive(v___x_804_);
if (v_isSharedCheck_812_ == 0)
{
lean_object* v_unused_813_; 
v_unused_813_ = lean_ctor_get(v___x_804_, 1);
lean_dec(v_unused_813_);
v___x_807_ = v___x_804_;
v_isShared_808_ = v_isSharedCheck_812_;
goto v_resetjp_806_;
}
else
{
lean_inc(v_toOne_805_);
lean_dec(v___x_804_);
v___x_807_ = lean_box(0);
v_isShared_808_ = v_isSharedCheck_812_;
goto v_resetjp_806_;
}
v_resetjp_806_:
{
lean_object* v___x_810_; 
if (v_isShared_808_ == 0)
{
lean_ctor_set(v___x_807_, 1, v_toOne_805_);
lean_ctor_set(v___x_807_, 0, v_inst_802_);
v___x_810_ = v___x_807_;
goto v_reusejp_809_;
}
else
{
lean_object* v_reuseFailAlloc_811_; 
v_reuseFailAlloc_811_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_811_, 0, v_inst_802_);
lean_ctor_set(v_reuseFailAlloc_811_, 1, v_toOne_805_);
v___x_810_ = v_reuseFailAlloc_811_;
goto v_reusejp_809_;
}
v_reusejp_809_:
{
return v___x_810_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_one___redArg___boxed(lean_object* v_inst_814_, lean_object* v_inst_815_){
_start:
{
lean_object* v_res_816_; 
v_res_816_ = lp_mathlib_OreLocalization_one___redArg(v_inst_814_, v_inst_815_);
lean_dec_ref(v_inst_814_);
return v_res_816_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_one(lean_object* v_R_817_, lean_object* v_inst_818_, lean_object* v_S_819_, lean_object* v_inst_820_, lean_object* v_X_821_, lean_object* v_inst_822_, lean_object* v_inst_823_){
_start:
{
lean_object* v___x_824_; 
v___x_824_ = lp_mathlib_OreLocalization_one___redArg(v_inst_818_, v_inst_823_);
return v___x_824_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_one___boxed(lean_object* v_R_825_, lean_object* v_inst_826_, lean_object* v_S_827_, lean_object* v_inst_828_, lean_object* v_X_829_, lean_object* v_inst_830_, lean_object* v_inst_831_){
_start:
{
lean_object* v_res_832_; 
v_res_832_ = lp_mathlib_OreLocalization_one(v_R_825_, v_inst_826_, v_S_827_, v_inst_828_, v_X_829_, v_inst_830_, v_inst_831_);
lean_dec(v_inst_830_);
lean_dec_ref(v_inst_828_);
lean_dec_ref(v_inst_826_);
return v_res_832_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_zero___redArg(lean_object* v_inst_833_, lean_object* v_inst_834_){
_start:
{
lean_object* v___x_835_; lean_object* v___x_836_; lean_object* v_toZero_837_; lean_object* v___x_839_; uint8_t v_isShared_840_; uint8_t v_isSharedCheck_844_; 
v___x_835_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_833_);
v___x_836_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_835_);
v_toZero_837_ = lean_ctor_get(v___x_836_, 0);
v_isSharedCheck_844_ = !lean_is_exclusive(v___x_836_);
if (v_isSharedCheck_844_ == 0)
{
lean_object* v_unused_845_; 
v_unused_845_ = lean_ctor_get(v___x_836_, 1);
lean_dec(v_unused_845_);
v___x_839_ = v___x_836_;
v_isShared_840_ = v_isSharedCheck_844_;
goto v_resetjp_838_;
}
else
{
lean_inc(v_toZero_837_);
lean_dec(v___x_836_);
v___x_839_ = lean_box(0);
v_isShared_840_ = v_isSharedCheck_844_;
goto v_resetjp_838_;
}
v_resetjp_838_:
{
lean_object* v___x_842_; 
if (v_isShared_840_ == 0)
{
lean_ctor_set(v___x_839_, 1, v_toZero_837_);
lean_ctor_set(v___x_839_, 0, v_inst_834_);
v___x_842_ = v___x_839_;
goto v_reusejp_841_;
}
else
{
lean_object* v_reuseFailAlloc_843_; 
v_reuseFailAlloc_843_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_843_, 0, v_inst_834_);
lean_ctor_set(v_reuseFailAlloc_843_, 1, v_toZero_837_);
v___x_842_ = v_reuseFailAlloc_843_;
goto v_reusejp_841_;
}
v_reusejp_841_:
{
return v___x_842_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_zero___redArg___boxed(lean_object* v_inst_846_, lean_object* v_inst_847_){
_start:
{
lean_object* v_res_848_; 
v_res_848_ = lp_mathlib_AddOreLocalization_zero___redArg(v_inst_846_, v_inst_847_);
lean_dec_ref(v_inst_846_);
return v_res_848_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_zero(lean_object* v_R_849_, lean_object* v_inst_850_, lean_object* v_S_851_, lean_object* v_inst_852_, lean_object* v_X_853_, lean_object* v_inst_854_, lean_object* v_inst_855_){
_start:
{
lean_object* v___x_856_; 
v___x_856_ = lp_mathlib_AddOreLocalization_zero___redArg(v_inst_850_, v_inst_855_);
return v___x_856_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_zero___boxed(lean_object* v_R_857_, lean_object* v_inst_858_, lean_object* v_S_859_, lean_object* v_inst_860_, lean_object* v_X_861_, lean_object* v_inst_862_, lean_object* v_inst_863_){
_start:
{
lean_object* v_res_864_; 
v_res_864_ = lp_mathlib_AddOreLocalization_zero(v_R_857_, v_inst_858_, v_S_859_, v_inst_860_, v_X_861_, v_inst_862_, v_inst_863_);
lean_dec(v_inst_862_);
lean_dec_ref(v_inst_860_);
lean_dec_ref(v_inst_858_);
return v_res_864_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instOne___redArg(lean_object* v_inst_865_, lean_object* v_inst_866_){
_start:
{
lean_object* v___x_867_; 
v___x_867_ = lp_mathlib_OreLocalization_one___redArg(v_inst_865_, v_inst_866_);
return v___x_867_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instOne___redArg___boxed(lean_object* v_inst_868_, lean_object* v_inst_869_){
_start:
{
lean_object* v_res_870_; 
v_res_870_ = lp_mathlib_OreLocalization_instOne___redArg(v_inst_868_, v_inst_869_);
lean_dec_ref(v_inst_868_);
return v_res_870_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instOne(lean_object* v_R_871_, lean_object* v_inst_872_, lean_object* v_S_873_, lean_object* v_inst_874_, lean_object* v_X_875_, lean_object* v_inst_876_, lean_object* v_inst_877_){
_start:
{
lean_object* v___x_878_; 
v___x_878_ = lp_mathlib_OreLocalization_one___redArg(v_inst_872_, v_inst_877_);
return v___x_878_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instOne___boxed(lean_object* v_R_879_, lean_object* v_inst_880_, lean_object* v_S_881_, lean_object* v_inst_882_, lean_object* v_X_883_, lean_object* v_inst_884_, lean_object* v_inst_885_){
_start:
{
lean_object* v_res_886_; 
v_res_886_ = lp_mathlib_OreLocalization_instOne(v_R_879_, v_inst_880_, v_S_881_, v_inst_882_, v_X_883_, v_inst_884_, v_inst_885_);
lean_dec(v_inst_884_);
lean_dec_ref(v_inst_882_);
lean_dec_ref(v_inst_880_);
return v_res_886_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_instZero___redArg(lean_object* v_inst_887_, lean_object* v_inst_888_){
_start:
{
lean_object* v___x_889_; 
v___x_889_ = lp_mathlib_AddOreLocalization_zero___redArg(v_inst_887_, v_inst_888_);
return v___x_889_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_instZero___redArg___boxed(lean_object* v_inst_890_, lean_object* v_inst_891_){
_start:
{
lean_object* v_res_892_; 
v_res_892_ = lp_mathlib_AddOreLocalization_instZero___redArg(v_inst_890_, v_inst_891_);
lean_dec_ref(v_inst_890_);
return v_res_892_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_instZero(lean_object* v_R_893_, lean_object* v_inst_894_, lean_object* v_S_895_, lean_object* v_inst_896_, lean_object* v_X_897_, lean_object* v_inst_898_, lean_object* v_inst_899_){
_start:
{
lean_object* v___x_900_; 
v___x_900_ = lp_mathlib_AddOreLocalization_zero___redArg(v_inst_894_, v_inst_899_);
return v___x_900_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_instZero___boxed(lean_object* v_R_901_, lean_object* v_inst_902_, lean_object* v_S_903_, lean_object* v_inst_904_, lean_object* v_X_905_, lean_object* v_inst_906_, lean_object* v_inst_907_){
_start:
{
lean_object* v_res_908_; 
v_res_908_ = lp_mathlib_AddOreLocalization_instZero(v_R_901_, v_inst_902_, v_S_903_, v_inst_904_, v_X_905_, v_inst_906_, v_inst_907_);
lean_dec(v_inst_906_);
lean_dec_ref(v_inst_904_);
lean_dec_ref(v_inst_902_);
return v_res_908_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instInhabited___redArg(lean_object* v_inst_909_){
_start:
{
lean_object* v___x_910_; lean_object* v___x_911_; lean_object* v_toOne_912_; lean_object* v___x_913_; 
v___x_910_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_909_);
v___x_911_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_910_);
v_toOne_912_ = lean_ctor_get(v___x_911_, 0);
lean_inc(v_toOne_912_);
lean_dec_ref(v___x_911_);
v___x_913_ = lp_mathlib_OreLocalization_one___redArg(v_inst_909_, v_toOne_912_);
return v___x_913_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instInhabited___redArg___boxed(lean_object* v_inst_914_){
_start:
{
lean_object* v_res_915_; 
v_res_915_ = lp_mathlib_OreLocalization_instInhabited___redArg(v_inst_914_);
lean_dec_ref(v_inst_914_);
return v_res_915_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instInhabited(lean_object* v_R_916_, lean_object* v_inst_917_, lean_object* v_S_918_, lean_object* v_inst_919_){
_start:
{
lean_object* v___x_920_; 
v___x_920_ = lp_mathlib_OreLocalization_instInhabited___redArg(v_inst_917_);
return v___x_920_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instInhabited___boxed(lean_object* v_R_921_, lean_object* v_inst_922_, lean_object* v_S_923_, lean_object* v_inst_924_){
_start:
{
lean_object* v_res_925_; 
v_res_925_ = lp_mathlib_OreLocalization_instInhabited(v_R_921_, v_inst_922_, v_S_923_, v_inst_924_);
lean_dec_ref(v_inst_924_);
lean_dec_ref(v_inst_922_);
return v_res_925_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_instInhabited___redArg(lean_object* v_inst_926_){
_start:
{
lean_object* v___x_927_; lean_object* v___x_928_; lean_object* v_toZero_929_; lean_object* v___x_930_; 
v___x_927_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_926_);
v___x_928_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_927_);
v_toZero_929_ = lean_ctor_get(v___x_928_, 0);
lean_inc(v_toZero_929_);
lean_dec_ref(v___x_928_);
v___x_930_ = lp_mathlib_AddOreLocalization_zero___redArg(v_inst_926_, v_toZero_929_);
return v___x_930_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_instInhabited___redArg___boxed(lean_object* v_inst_931_){
_start:
{
lean_object* v_res_932_; 
v_res_932_ = lp_mathlib_AddOreLocalization_instInhabited___redArg(v_inst_931_);
lean_dec_ref(v_inst_931_);
return v_res_932_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_instInhabited(lean_object* v_R_933_, lean_object* v_inst_934_, lean_object* v_S_935_, lean_object* v_inst_936_){
_start:
{
lean_object* v___x_937_; 
v___x_937_ = lp_mathlib_AddOreLocalization_instInhabited___redArg(v_inst_934_);
return v___x_937_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_instInhabited___boxed(lean_object* v_R_938_, lean_object* v_inst_939_, lean_object* v_S_940_, lean_object* v_inst_941_){
_start:
{
lean_object* v_res_942_; 
v_res_942_ = lp_mathlib_AddOreLocalization_instInhabited(v_R_938_, v_inst_939_, v_S_940_, v_inst_941_);
lean_dec_ref(v_inst_941_);
lean_dec_ref(v_inst_939_);
return v_res_942_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_npow___redArg(lean_object* v_inst_943_, lean_object* v_S_944_, lean_object* v_inst_945_, lean_object* v_a_946_, lean_object* v_a_947_){
_start:
{
lean_object* v___x_948_; lean_object* v___x_949_; lean_object* v_toOne_950_; lean_object* v___x_951_; lean_object* v___x_952_; lean_object* v___x_953_; 
v___x_948_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_943_);
v___x_949_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_948_);
v_toOne_950_ = lean_ctor_get(v___x_949_, 0);
lean_inc(v_toOne_950_);
lean_dec_ref(v___x_949_);
v___x_951_ = lp_mathlib_OreLocalization_one___redArg(v_inst_943_, v_toOne_950_);
v___x_952_ = lp_mathlib_OreLocalization_instMul___redArg(v_inst_943_, v_S_944_, v_inst_945_);
v___x_953_ = l_npowRec___redArg(v___x_951_, v___x_952_, v_a_946_, v_a_947_);
lean_dec_ref(v___x_951_);
return v___x_953_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_npow___redArg___boxed(lean_object* v_inst_954_, lean_object* v_S_955_, lean_object* v_inst_956_, lean_object* v_a_957_, lean_object* v_a_958_){
_start:
{
lean_object* v_res_959_; 
v_res_959_ = lp_mathlib_OreLocalization_npow___redArg(v_inst_954_, v_S_955_, v_inst_956_, v_a_957_, v_a_958_);
lean_dec(v_a_957_);
return v_res_959_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_npow(lean_object* v_R_960_, lean_object* v_inst_961_, lean_object* v_S_962_, lean_object* v_inst_963_, lean_object* v_a_964_, lean_object* v_a_965_){
_start:
{
lean_object* v___x_966_; lean_object* v___x_967_; lean_object* v_toOne_968_; lean_object* v___x_969_; lean_object* v___x_970_; lean_object* v___x_971_; 
v___x_966_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_961_);
v___x_967_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_966_);
v_toOne_968_ = lean_ctor_get(v___x_967_, 0);
lean_inc(v_toOne_968_);
lean_dec_ref(v___x_967_);
v___x_969_ = lp_mathlib_OreLocalization_one___redArg(v_inst_961_, v_toOne_968_);
v___x_970_ = lp_mathlib_OreLocalization_instMul___redArg(v_inst_961_, v_S_962_, v_inst_963_);
v___x_971_ = l_npowRec___redArg(v___x_969_, v___x_970_, v_a_964_, v_a_965_);
lean_dec_ref(v___x_969_);
return v___x_971_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_npow___boxed(lean_object* v_R_972_, lean_object* v_inst_973_, lean_object* v_S_974_, lean_object* v_inst_975_, lean_object* v_a_976_, lean_object* v_a_977_){
_start:
{
lean_object* v_res_978_; 
v_res_978_ = lp_mathlib_OreLocalization_npow(v_R_972_, v_inst_973_, v_S_974_, v_inst_975_, v_a_976_, v_a_977_);
lean_dec(v_a_976_);
return v_res_978_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_nsmul___redArg(lean_object* v_inst_979_, lean_object* v_S_980_, lean_object* v_inst_981_, lean_object* v_a_982_, lean_object* v_a_983_){
_start:
{
lean_object* v___x_984_; lean_object* v___x_985_; lean_object* v_toZero_986_; lean_object* v___x_987_; lean_object* v___x_988_; lean_object* v___x_989_; 
v___x_984_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_979_);
v___x_985_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_984_);
v_toZero_986_ = lean_ctor_get(v___x_985_, 0);
lean_inc(v_toZero_986_);
lean_dec_ref(v___x_985_);
v___x_987_ = lp_mathlib_AddOreLocalization_zero___redArg(v_inst_979_, v_toZero_986_);
v___x_988_ = lp_mathlib_AddOreLocalization_instAdd___redArg(v_inst_979_, v_S_980_, v_inst_981_);
v___x_989_ = l_nsmulRec___redArg(v___x_987_, v___x_988_, v_a_982_, v_a_983_);
lean_dec_ref(v___x_987_);
return v___x_989_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_nsmul___redArg___boxed(lean_object* v_inst_990_, lean_object* v_S_991_, lean_object* v_inst_992_, lean_object* v_a_993_, lean_object* v_a_994_){
_start:
{
lean_object* v_res_995_; 
v_res_995_ = lp_mathlib_AddOreLocalization_nsmul___redArg(v_inst_990_, v_S_991_, v_inst_992_, v_a_993_, v_a_994_);
lean_dec(v_a_993_);
return v_res_995_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_nsmul(lean_object* v_R_996_, lean_object* v_inst_997_, lean_object* v_S_998_, lean_object* v_inst_999_, lean_object* v_a_1000_, lean_object* v_a_1001_){
_start:
{
lean_object* v___x_1002_; 
v___x_1002_ = lp_mathlib_AddOreLocalization_nsmul___redArg(v_inst_997_, v_S_998_, v_inst_999_, v_a_1000_, v_a_1001_);
return v___x_1002_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_nsmul___boxed(lean_object* v_R_1003_, lean_object* v_inst_1004_, lean_object* v_S_1005_, lean_object* v_inst_1006_, lean_object* v_a_1007_, lean_object* v_a_1008_){
_start:
{
lean_object* v_res_1009_; 
v_res_1009_ = lp_mathlib_AddOreLocalization_nsmul(v_R_1003_, v_inst_1004_, v_S_1005_, v_inst_1006_, v_a_1007_, v_a_1008_);
lean_dec(v_a_1007_);
return v_res_1009_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instMonoid___redArg(lean_object* v_inst_1010_, lean_object* v_S_1011_, lean_object* v_inst_1012_){
_start:
{
lean_object* v___x_1013_; lean_object* v___x_1014_; lean_object* v_toOne_1015_; lean_object* v___x_1016_; lean_object* v___x_1017_; lean_object* v___x_1018_; lean_object* v___x_1019_; 
v___x_1013_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_1010_);
v___x_1014_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_1013_);
v_toOne_1015_ = lean_ctor_get(v___x_1014_, 0);
lean_inc(v_toOne_1015_);
lean_dec_ref(v___x_1014_);
v___x_1016_ = lp_mathlib_OreLocalization_one___redArg(v_inst_1010_, v_toOne_1015_);
lean_inc_ref(v_inst_1012_);
lean_inc_ref(v_inst_1010_);
v___x_1017_ = lp_mathlib_OreLocalization_instMul___redArg(v_inst_1010_, v_S_1011_, v_inst_1012_);
v___x_1018_ = lean_alloc_closure((void*)(lp_mathlib_OreLocalization_npow___boxed), 6, 4);
lean_closure_set(v___x_1018_, 0, lean_box(0));
lean_closure_set(v___x_1018_, 1, v_inst_1010_);
lean_closure_set(v___x_1018_, 2, v_S_1011_);
lean_closure_set(v___x_1018_, 3, v_inst_1012_);
v___x_1019_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1019_, 0, v___x_1016_);
lean_ctor_set(v___x_1019_, 1, v___x_1017_);
lean_ctor_set(v___x_1019_, 2, v___x_1018_);
return v___x_1019_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instMonoid(lean_object* v_R_1020_, lean_object* v_inst_1021_, lean_object* v_S_1022_, lean_object* v_inst_1023_){
_start:
{
lean_object* v___x_1024_; 
v___x_1024_ = lp_mathlib_OreLocalization_instMonoid___redArg(v_inst_1021_, v_S_1022_, v_inst_1023_);
return v___x_1024_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_instAddMonoid___redArg(lean_object* v_inst_1025_, lean_object* v_S_1026_, lean_object* v_inst_1027_){
_start:
{
lean_object* v___x_1028_; lean_object* v___x_1029_; lean_object* v_toZero_1030_; lean_object* v___x_1031_; lean_object* v___x_1032_; lean_object* v___x_1033_; lean_object* v___x_1034_; 
v___x_1028_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_1025_);
v___x_1029_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_1028_);
v_toZero_1030_ = lean_ctor_get(v___x_1029_, 0);
lean_inc(v_toZero_1030_);
lean_dec_ref(v___x_1029_);
v___x_1031_ = lp_mathlib_AddOreLocalization_zero___redArg(v_inst_1025_, v_toZero_1030_);
lean_inc_ref(v_inst_1027_);
lean_inc_ref(v_inst_1025_);
v___x_1032_ = lp_mathlib_AddOreLocalization_instAdd___redArg(v_inst_1025_, v_S_1026_, v_inst_1027_);
v___x_1033_ = lean_alloc_closure((void*)(lp_mathlib_AddOreLocalization_nsmul___boxed), 6, 4);
lean_closure_set(v___x_1033_, 0, lean_box(0));
lean_closure_set(v___x_1033_, 1, v_inst_1025_);
lean_closure_set(v___x_1033_, 2, v_S_1026_);
lean_closure_set(v___x_1033_, 3, v_inst_1027_);
v___x_1034_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1034_, 0, v___x_1031_);
lean_ctor_set(v___x_1034_, 1, v___x_1032_);
lean_ctor_set(v___x_1034_, 2, v___x_1033_);
return v___x_1034_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_instAddMonoid(lean_object* v_R_1035_, lean_object* v_inst_1036_, lean_object* v_S_1037_, lean_object* v_inst_1038_){
_start:
{
lean_object* v___x_1039_; 
v___x_1039_ = lp_mathlib_AddOreLocalization_instAddMonoid___redArg(v_inst_1036_, v_S_1037_, v_inst_1038_);
return v___x_1039_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instMulActionOreLocalization___redArg(lean_object* v_inst_1040_, lean_object* v_S_1041_, lean_object* v_inst_1042_, lean_object* v_inst_1043_){
_start:
{
lean_object* v___x_1044_; 
v___x_1044_ = lean_alloc_closure((void*)(lp_mathlib_OreLocalization_smul___boxed), 8, 6);
lean_closure_set(v___x_1044_, 0, lean_box(0));
lean_closure_set(v___x_1044_, 1, v_inst_1040_);
lean_closure_set(v___x_1044_, 2, v_S_1041_);
lean_closure_set(v___x_1044_, 3, v_inst_1042_);
lean_closure_set(v___x_1044_, 4, lean_box(0));
lean_closure_set(v___x_1044_, 5, v_inst_1043_);
return v___x_1044_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instMulActionOreLocalization(lean_object* v_R_1045_, lean_object* v_inst_1046_, lean_object* v_S_1047_, lean_object* v_inst_1048_, lean_object* v_X_1049_, lean_object* v_inst_1050_){
_start:
{
lean_object* v___x_1051_; 
v___x_1051_ = lean_alloc_closure((void*)(lp_mathlib_OreLocalization_smul___boxed), 8, 6);
lean_closure_set(v___x_1051_, 0, lean_box(0));
lean_closure_set(v___x_1051_, 1, v_inst_1046_);
lean_closure_set(v___x_1051_, 2, v_S_1047_);
lean_closure_set(v___x_1051_, 3, v_inst_1048_);
lean_closure_set(v___x_1051_, 4, lean_box(0));
lean_closure_set(v___x_1051_, 5, v_inst_1050_);
return v___x_1051_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_instAddActionOreLocalization___redArg(lean_object* v_inst_1052_, lean_object* v_S_1053_, lean_object* v_inst_1054_, lean_object* v_inst_1055_){
_start:
{
lean_object* v___x_1056_; 
v___x_1056_ = lean_alloc_closure((void*)(lp_mathlib_AddOreLocalization_vadd___boxed), 8, 6);
lean_closure_set(v___x_1056_, 0, lean_box(0));
lean_closure_set(v___x_1056_, 1, v_inst_1052_);
lean_closure_set(v___x_1056_, 2, v_S_1053_);
lean_closure_set(v___x_1056_, 3, v_inst_1054_);
lean_closure_set(v___x_1056_, 4, lean_box(0));
lean_closure_set(v___x_1056_, 5, v_inst_1055_);
return v___x_1056_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_instAddActionOreLocalization(lean_object* v_R_1057_, lean_object* v_inst_1058_, lean_object* v_S_1059_, lean_object* v_inst_1060_, lean_object* v_X_1061_, lean_object* v_inst_1062_){
_start:
{
lean_object* v___x_1063_; 
v___x_1063_ = lean_alloc_closure((void*)(lp_mathlib_AddOreLocalization_vadd___boxed), 8, 6);
lean_closure_set(v___x_1063_, 0, lean_box(0));
lean_closure_set(v___x_1063_, 1, v_inst_1058_);
lean_closure_set(v___x_1063_, 2, v_S_1059_);
lean_closure_set(v___x_1063_, 3, v_inst_1060_);
lean_closure_set(v___x_1063_, 4, lean_box(0));
lean_closure_set(v___x_1063_, 5, v_inst_1062_);
return v___x_1063_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_numeratorUnit___redArg(lean_object* v_inst_1064_, lean_object* v_s_1065_){
_start:
{
lean_object* v___x_1066_; lean_object* v___x_1067_; lean_object* v_toOne_1068_; lean_object* v___x_1070_; uint8_t v_isShared_1071_; uint8_t v_isSharedCheck_1077_; 
v___x_1066_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_1064_);
v___x_1067_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_1066_);
v_toOne_1068_ = lean_ctor_get(v___x_1067_, 0);
v_isSharedCheck_1077_ = !lean_is_exclusive(v___x_1067_);
if (v_isSharedCheck_1077_ == 0)
{
lean_object* v_unused_1078_; 
v_unused_1078_ = lean_ctor_get(v___x_1067_, 1);
lean_dec(v_unused_1078_);
v___x_1070_ = v___x_1067_;
v_isShared_1071_ = v_isSharedCheck_1077_;
goto v_resetjp_1069_;
}
else
{
lean_inc(v_toOne_1068_);
lean_dec(v___x_1067_);
v___x_1070_ = lean_box(0);
v_isShared_1071_ = v_isSharedCheck_1077_;
goto v_resetjp_1069_;
}
v_resetjp_1069_:
{
lean_object* v___x_1073_; 
lean_inc(v_toOne_1068_);
lean_inc(v_s_1065_);
if (v_isShared_1071_ == 0)
{
lean_ctor_set(v___x_1070_, 1, v_toOne_1068_);
lean_ctor_set(v___x_1070_, 0, v_s_1065_);
v___x_1073_ = v___x_1070_;
goto v_reusejp_1072_;
}
else
{
lean_object* v_reuseFailAlloc_1076_; 
v_reuseFailAlloc_1076_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1076_, 0, v_s_1065_);
lean_ctor_set(v_reuseFailAlloc_1076_, 1, v_toOne_1068_);
v___x_1073_ = v_reuseFailAlloc_1076_;
goto v_reusejp_1072_;
}
v_reusejp_1072_:
{
lean_object* v___x_1074_; lean_object* v___x_1075_; 
v___x_1074_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1074_, 0, v_toOne_1068_);
lean_ctor_set(v___x_1074_, 1, v_s_1065_);
v___x_1075_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1075_, 0, v___x_1073_);
lean_ctor_set(v___x_1075_, 1, v___x_1074_);
return v___x_1075_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_numeratorUnit___redArg___boxed(lean_object* v_inst_1079_, lean_object* v_s_1080_){
_start:
{
lean_object* v_res_1081_; 
v_res_1081_ = lp_mathlib_OreLocalization_numeratorUnit___redArg(v_inst_1079_, v_s_1080_);
lean_dec_ref(v_inst_1079_);
return v_res_1081_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_numeratorUnit(lean_object* v_R_1082_, lean_object* v_inst_1083_, lean_object* v_S_1084_, lean_object* v_inst_1085_, lean_object* v_s_1086_){
_start:
{
lean_object* v___x_1087_; 
v___x_1087_ = lp_mathlib_OreLocalization_numeratorUnit___redArg(v_inst_1083_, v_s_1086_);
return v___x_1087_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_numeratorUnit___boxed(lean_object* v_R_1088_, lean_object* v_inst_1089_, lean_object* v_S_1090_, lean_object* v_inst_1091_, lean_object* v_s_1092_){
_start:
{
lean_object* v_res_1093_; 
v_res_1093_ = lp_mathlib_OreLocalization_numeratorUnit(v_R_1088_, v_inst_1089_, v_S_1090_, v_inst_1091_, v_s_1092_);
lean_dec_ref(v_inst_1091_);
lean_dec_ref(v_inst_1089_);
return v_res_1093_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_numeratorAddUnit___redArg(lean_object* v_inst_1094_, lean_object* v_s_1095_){
_start:
{
lean_object* v___x_1096_; lean_object* v___x_1097_; lean_object* v_toZero_1098_; lean_object* v___x_1100_; uint8_t v_isShared_1101_; uint8_t v_isSharedCheck_1107_; 
v___x_1096_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_1094_);
v___x_1097_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_1096_);
v_toZero_1098_ = lean_ctor_get(v___x_1097_, 0);
v_isSharedCheck_1107_ = !lean_is_exclusive(v___x_1097_);
if (v_isSharedCheck_1107_ == 0)
{
lean_object* v_unused_1108_; 
v_unused_1108_ = lean_ctor_get(v___x_1097_, 1);
lean_dec(v_unused_1108_);
v___x_1100_ = v___x_1097_;
v_isShared_1101_ = v_isSharedCheck_1107_;
goto v_resetjp_1099_;
}
else
{
lean_inc(v_toZero_1098_);
lean_dec(v___x_1097_);
v___x_1100_ = lean_box(0);
v_isShared_1101_ = v_isSharedCheck_1107_;
goto v_resetjp_1099_;
}
v_resetjp_1099_:
{
lean_object* v___x_1103_; 
lean_inc(v_toZero_1098_);
lean_inc(v_s_1095_);
if (v_isShared_1101_ == 0)
{
lean_ctor_set(v___x_1100_, 1, v_toZero_1098_);
lean_ctor_set(v___x_1100_, 0, v_s_1095_);
v___x_1103_ = v___x_1100_;
goto v_reusejp_1102_;
}
else
{
lean_object* v_reuseFailAlloc_1106_; 
v_reuseFailAlloc_1106_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1106_, 0, v_s_1095_);
lean_ctor_set(v_reuseFailAlloc_1106_, 1, v_toZero_1098_);
v___x_1103_ = v_reuseFailAlloc_1106_;
goto v_reusejp_1102_;
}
v_reusejp_1102_:
{
lean_object* v___x_1104_; lean_object* v___x_1105_; 
v___x_1104_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1104_, 0, v_toZero_1098_);
lean_ctor_set(v___x_1104_, 1, v_s_1095_);
v___x_1105_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1105_, 0, v___x_1103_);
lean_ctor_set(v___x_1105_, 1, v___x_1104_);
return v___x_1105_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_numeratorAddUnit___redArg___boxed(lean_object* v_inst_1109_, lean_object* v_s_1110_){
_start:
{
lean_object* v_res_1111_; 
v_res_1111_ = lp_mathlib_AddOreLocalization_numeratorAddUnit___redArg(v_inst_1109_, v_s_1110_);
lean_dec_ref(v_inst_1109_);
return v_res_1111_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_numeratorAddUnit(lean_object* v_R_1112_, lean_object* v_inst_1113_, lean_object* v_S_1114_, lean_object* v_inst_1115_, lean_object* v_s_1116_){
_start:
{
lean_object* v___x_1117_; 
v___x_1117_ = lp_mathlib_AddOreLocalization_numeratorAddUnit___redArg(v_inst_1113_, v_s_1116_);
return v___x_1117_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_numeratorAddUnit___boxed(lean_object* v_R_1118_, lean_object* v_inst_1119_, lean_object* v_S_1120_, lean_object* v_inst_1121_, lean_object* v_s_1122_){
_start:
{
lean_object* v_res_1123_; 
v_res_1123_ = lp_mathlib_AddOreLocalization_numeratorAddUnit(v_R_1118_, v_inst_1119_, v_S_1120_, v_inst_1121_, v_s_1122_);
lean_dec_ref(v_inst_1121_);
lean_dec_ref(v_inst_1119_);
return v_res_1123_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_numeratorHom___redArg___lam__0(lean_object* v_toOne_1124_, lean_object* v_r_1125_){
_start:
{
lean_object* v___x_1126_; 
v___x_1126_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1126_, 0, v_r_1125_);
lean_ctor_set(v___x_1126_, 1, v_toOne_1124_);
return v___x_1126_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_numeratorHom___redArg(lean_object* v_inst_1127_){
_start:
{
lean_object* v___x_1128_; lean_object* v___x_1129_; lean_object* v_toOne_1130_; lean_object* v___f_1131_; 
v___x_1128_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_1127_);
v___x_1129_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_1128_);
v_toOne_1130_ = lean_ctor_get(v___x_1129_, 0);
lean_inc(v_toOne_1130_);
lean_dec_ref(v___x_1129_);
v___f_1131_ = lean_alloc_closure((void*)(lp_mathlib_OreLocalization_numeratorHom___redArg___lam__0), 2, 1);
lean_closure_set(v___f_1131_, 0, v_toOne_1130_);
return v___f_1131_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_numeratorHom___redArg___boxed(lean_object* v_inst_1132_){
_start:
{
lean_object* v_res_1133_; 
v_res_1133_ = lp_mathlib_OreLocalization_numeratorHom___redArg(v_inst_1132_);
lean_dec_ref(v_inst_1132_);
return v_res_1133_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_numeratorHom(lean_object* v_R_1134_, lean_object* v_inst_1135_, lean_object* v_S_1136_, lean_object* v_inst_1137_){
_start:
{
lean_object* v___x_1138_; lean_object* v___x_1139_; lean_object* v_toOne_1140_; lean_object* v___f_1141_; 
v___x_1138_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_1135_);
v___x_1139_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_1138_);
v_toOne_1140_ = lean_ctor_get(v___x_1139_, 0);
lean_inc(v_toOne_1140_);
lean_dec_ref(v___x_1139_);
v___f_1141_ = lean_alloc_closure((void*)(lp_mathlib_OreLocalization_numeratorHom___redArg___lam__0), 2, 1);
lean_closure_set(v___f_1141_, 0, v_toOne_1140_);
return v___f_1141_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_numeratorHom___boxed(lean_object* v_R_1142_, lean_object* v_inst_1143_, lean_object* v_S_1144_, lean_object* v_inst_1145_){
_start:
{
lean_object* v_res_1146_; 
v_res_1146_ = lp_mathlib_OreLocalization_numeratorHom(v_R_1142_, v_inst_1143_, v_S_1144_, v_inst_1145_);
lean_dec_ref(v_inst_1145_);
lean_dec_ref(v_inst_1143_);
return v_res_1146_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_numeratorHom___redArg___lam__0(lean_object* v_toZero_1147_, lean_object* v_r_1148_){
_start:
{
lean_object* v___x_1149_; 
v___x_1149_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1149_, 0, v_r_1148_);
lean_ctor_set(v___x_1149_, 1, v_toZero_1147_);
return v___x_1149_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_numeratorHom___redArg(lean_object* v_inst_1150_){
_start:
{
lean_object* v___x_1151_; lean_object* v___x_1152_; lean_object* v_toZero_1153_; lean_object* v___f_1154_; 
v___x_1151_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_1150_);
v___x_1152_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_1151_);
v_toZero_1153_ = lean_ctor_get(v___x_1152_, 0);
lean_inc(v_toZero_1153_);
lean_dec_ref(v___x_1152_);
v___f_1154_ = lean_alloc_closure((void*)(lp_mathlib_AddOreLocalization_numeratorHom___redArg___lam__0), 2, 1);
lean_closure_set(v___f_1154_, 0, v_toZero_1153_);
return v___f_1154_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_numeratorHom___redArg___boxed(lean_object* v_inst_1155_){
_start:
{
lean_object* v_res_1156_; 
v_res_1156_ = lp_mathlib_AddOreLocalization_numeratorHom___redArg(v_inst_1155_);
lean_dec_ref(v_inst_1155_);
return v_res_1156_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_numeratorHom(lean_object* v_R_1157_, lean_object* v_inst_1158_, lean_object* v_S_1159_, lean_object* v_inst_1160_){
_start:
{
lean_object* v___x_1161_; 
v___x_1161_ = lp_mathlib_AddOreLocalization_numeratorHom___redArg(v_inst_1158_);
return v___x_1161_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_numeratorHom___boxed(lean_object* v_R_1162_, lean_object* v_inst_1163_, lean_object* v_S_1164_, lean_object* v_inst_1165_){
_start:
{
lean_object* v_res_1166_; 
v_res_1166_ = lp_mathlib_AddOreLocalization_numeratorHom(v_R_1162_, v_inst_1163_, v_S_1164_, v_inst_1165_);
lean_dec_ref(v_inst_1165_);
lean_dec_ref(v_inst_1163_);
return v_res_1166_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_universalMulHom___redArg___lam__0(lean_object* v_fS_1167_, lean_object* v_f_1168_, lean_object* v_toMul_1169_, lean_object* v_r_1170_, lean_object* v_s_1171_){
_start:
{
lean_object* v___x_1172_; lean_object* v_inv_1173_; lean_object* v___x_1174_; lean_object* v___x_1175_; 
v___x_1172_ = lean_apply_1(v_fS_1167_, v_s_1171_);
v_inv_1173_ = lean_ctor_get(v___x_1172_, 1);
lean_inc(v_inv_1173_);
lean_dec_ref(v___x_1172_);
v___x_1174_ = lean_apply_1(v_f_1168_, v_r_1170_);
v___x_1175_ = lean_apply_2(v_toMul_1169_, v_inv_1173_, v___x_1174_);
return v___x_1175_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_universalMulHom___redArg___lam__1(lean_object* v___f_1176_, lean_object* v_x_1177_){
_start:
{
lean_object* v___x_1178_; 
v___x_1178_ = lp_mathlib_OreLocalization_liftExpand___redArg(v___f_1176_, v_x_1177_);
return v___x_1178_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_universalMulHom___redArg(lean_object* v_inst_1179_, lean_object* v_f_1180_, lean_object* v_fS_1181_){
_start:
{
lean_object* v___x_1182_; lean_object* v___x_1183_; lean_object* v_toMul_1184_; lean_object* v___f_1185_; lean_object* v___f_1186_; 
v___x_1182_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_1179_);
v___x_1183_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_1182_);
v_toMul_1184_ = lean_ctor_get(v___x_1183_, 1);
lean_inc(v_toMul_1184_);
lean_dec_ref(v___x_1183_);
v___f_1185_ = lean_alloc_closure((void*)(lp_mathlib_OreLocalization_universalMulHom___redArg___lam__0), 5, 3);
lean_closure_set(v___f_1185_, 0, v_fS_1181_);
lean_closure_set(v___f_1185_, 1, v_f_1180_);
lean_closure_set(v___f_1185_, 2, v_toMul_1184_);
v___f_1186_ = lean_alloc_closure((void*)(lp_mathlib_OreLocalization_universalMulHom___redArg___lam__1), 2, 1);
lean_closure_set(v___f_1186_, 0, v___f_1185_);
return v___f_1186_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_universalMulHom___redArg___boxed(lean_object* v_inst_1187_, lean_object* v_f_1188_, lean_object* v_fS_1189_){
_start:
{
lean_object* v_res_1190_; 
v_res_1190_ = lp_mathlib_OreLocalization_universalMulHom___redArg(v_inst_1187_, v_f_1188_, v_fS_1189_);
lean_dec_ref(v_inst_1187_);
return v_res_1190_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_universalMulHom(lean_object* v_R_1191_, lean_object* v_inst_1192_, lean_object* v_S_1193_, lean_object* v_inst_1194_, lean_object* v_T_1195_, lean_object* v_inst_1196_, lean_object* v_f_1197_, lean_object* v_fS_1198_, lean_object* v_hf_1199_){
_start:
{
lean_object* v___x_1200_; 
v___x_1200_ = lp_mathlib_OreLocalization_universalMulHom___redArg(v_inst_1196_, v_f_1197_, v_fS_1198_);
return v___x_1200_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_universalMulHom___boxed(lean_object* v_R_1201_, lean_object* v_inst_1202_, lean_object* v_S_1203_, lean_object* v_inst_1204_, lean_object* v_T_1205_, lean_object* v_inst_1206_, lean_object* v_f_1207_, lean_object* v_fS_1208_, lean_object* v_hf_1209_){
_start:
{
lean_object* v_res_1210_; 
v_res_1210_ = lp_mathlib_OreLocalization_universalMulHom(v_R_1201_, v_inst_1202_, v_S_1203_, v_inst_1204_, v_T_1205_, v_inst_1206_, v_f_1207_, v_fS_1208_, v_hf_1209_);
lean_dec_ref(v_inst_1206_);
lean_dec_ref(v_inst_1204_);
lean_dec_ref(v_inst_1202_);
return v_res_1210_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_universalAddHom___redArg___lam__0(lean_object* v_fS_1211_, lean_object* v_f_1212_, lean_object* v_toAdd_1213_, lean_object* v_r_1214_, lean_object* v_s_1215_){
_start:
{
lean_object* v___x_1216_; lean_object* v_neg_1217_; lean_object* v___x_1218_; lean_object* v___x_1219_; 
v___x_1216_ = lean_apply_1(v_fS_1211_, v_s_1215_);
v_neg_1217_ = lean_ctor_get(v___x_1216_, 1);
lean_inc(v_neg_1217_);
lean_dec_ref(v___x_1216_);
v___x_1218_ = lean_apply_1(v_f_1212_, v_r_1214_);
v___x_1219_ = lean_apply_2(v_toAdd_1213_, v_neg_1217_, v___x_1218_);
return v___x_1219_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_universalAddHom___redArg___lam__1(lean_object* v___f_1220_, lean_object* v_x_1221_){
_start:
{
lean_object* v___x_1222_; 
v___x_1222_ = lp_mathlib_AddOreLocalization_liftExpand___redArg(v___f_1220_, v_x_1221_);
return v___x_1222_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_universalAddHom___redArg(lean_object* v_inst_1223_, lean_object* v_f_1224_, lean_object* v_fS_1225_){
_start:
{
lean_object* v___x_1226_; lean_object* v___x_1227_; lean_object* v_toAdd_1228_; lean_object* v___f_1229_; lean_object* v___f_1230_; 
v___x_1226_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_1223_);
v___x_1227_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_1226_);
v_toAdd_1228_ = lean_ctor_get(v___x_1227_, 1);
lean_inc(v_toAdd_1228_);
lean_dec_ref(v___x_1227_);
v___f_1229_ = lean_alloc_closure((void*)(lp_mathlib_AddOreLocalization_universalAddHom___redArg___lam__0), 5, 3);
lean_closure_set(v___f_1229_, 0, v_fS_1225_);
lean_closure_set(v___f_1229_, 1, v_f_1224_);
lean_closure_set(v___f_1229_, 2, v_toAdd_1228_);
v___f_1230_ = lean_alloc_closure((void*)(lp_mathlib_AddOreLocalization_universalAddHom___redArg___lam__1), 2, 1);
lean_closure_set(v___f_1230_, 0, v___f_1229_);
return v___f_1230_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_universalAddHom___redArg___boxed(lean_object* v_inst_1231_, lean_object* v_f_1232_, lean_object* v_fS_1233_){
_start:
{
lean_object* v_res_1234_; 
v_res_1234_ = lp_mathlib_AddOreLocalization_universalAddHom___redArg(v_inst_1231_, v_f_1232_, v_fS_1233_);
lean_dec_ref(v_inst_1231_);
return v_res_1234_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_universalAddHom(lean_object* v_R_1235_, lean_object* v_inst_1236_, lean_object* v_S_1237_, lean_object* v_inst_1238_, lean_object* v_T_1239_, lean_object* v_inst_1240_, lean_object* v_f_1241_, lean_object* v_fS_1242_, lean_object* v_hf_1243_){
_start:
{
lean_object* v___x_1244_; 
v___x_1244_ = lp_mathlib_AddOreLocalization_universalAddHom___redArg(v_inst_1240_, v_f_1241_, v_fS_1242_);
return v___x_1244_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_universalAddHom___boxed(lean_object* v_R_1245_, lean_object* v_inst_1246_, lean_object* v_S_1247_, lean_object* v_inst_1248_, lean_object* v_T_1249_, lean_object* v_inst_1250_, lean_object* v_f_1251_, lean_object* v_fS_1252_, lean_object* v_hf_1253_){
_start:
{
lean_object* v_res_1254_; 
v_res_1254_ = lp_mathlib_AddOreLocalization_universalAddHom(v_R_1245_, v_inst_1246_, v_S_1247_, v_inst_1248_, v_T_1249_, v_inst_1250_, v_f_1251_, v_fS_1252_, v_hf_1253_);
lean_dec_ref(v_inst_1250_);
lean_dec_ref(v_inst_1248_);
lean_dec_ref(v_inst_1246_);
return v_res_1254_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_hsmul___redArg___lam__0(lean_object* v_inst_1255_, lean_object* v_c_1256_, lean_object* v_toOne_1257_, lean_object* v_inst_1258_, lean_object* v_inst_1259_, lean_object* v_m_1260_, lean_object* v_s_1261_){
_start:
{
lean_object* v___x_1262_; lean_object* v___x_1263_; lean_object* v___x_1264_; lean_object* v___x_1265_; lean_object* v___x_1266_; 
v___x_1262_ = lean_apply_2(v_inst_1255_, v_c_1256_, v_toOne_1257_);
lean_inc(v_s_1261_);
lean_inc(v___x_1262_);
lean_inc_ref(v_inst_1258_);
v___x_1263_ = lp_mathlib_OreLocalization_oreNum___redArg(v_inst_1258_, v___x_1262_, v_s_1261_);
v___x_1264_ = lean_apply_2(v_inst_1259_, v___x_1263_, v_m_1260_);
v___x_1265_ = lp_mathlib_OreLocalization_oreDenom___redArg(v_inst_1258_, v___x_1262_, v_s_1261_);
v___x_1266_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1266_, 0, v___x_1264_);
lean_ctor_set(v___x_1266_, 1, v___x_1265_);
return v___x_1266_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_hsmul___redArg(lean_object* v_inst_1267_, lean_object* v_inst_1268_, lean_object* v_inst_1269_, lean_object* v_inst_1270_, lean_object* v_c_1271_, lean_object* v_a_1272_){
_start:
{
lean_object* v___x_1273_; lean_object* v___x_1274_; lean_object* v_toOne_1275_; lean_object* v___f_1276_; lean_object* v___x_1277_; 
v___x_1273_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_1267_);
v___x_1274_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_1273_);
v_toOne_1275_ = lean_ctor_get(v___x_1274_, 0);
lean_inc(v_toOne_1275_);
lean_dec_ref(v___x_1274_);
v___f_1276_ = lean_alloc_closure((void*)(lp_mathlib_OreLocalization_hsmul___redArg___lam__0), 7, 5);
lean_closure_set(v___f_1276_, 0, v_inst_1270_);
lean_closure_set(v___f_1276_, 1, v_c_1271_);
lean_closure_set(v___f_1276_, 2, v_toOne_1275_);
lean_closure_set(v___f_1276_, 3, v_inst_1268_);
lean_closure_set(v___f_1276_, 4, v_inst_1269_);
v___x_1277_ = lp_mathlib_OreLocalization_liftExpand___redArg(v___f_1276_, v_a_1272_);
return v___x_1277_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_hsmul___redArg___boxed(lean_object* v_inst_1278_, lean_object* v_inst_1279_, lean_object* v_inst_1280_, lean_object* v_inst_1281_, lean_object* v_c_1282_, lean_object* v_a_1283_){
_start:
{
lean_object* v_res_1284_; 
v_res_1284_ = lp_mathlib_OreLocalization_hsmul___redArg(v_inst_1278_, v_inst_1279_, v_inst_1280_, v_inst_1281_, v_c_1282_, v_a_1283_);
lean_dec_ref(v_inst_1278_);
return v_res_1284_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_hsmul(lean_object* v_R_1285_, lean_object* v_M_1286_, lean_object* v_X_1287_, lean_object* v_inst_1288_, lean_object* v_S_1289_, lean_object* v_inst_1290_, lean_object* v_inst_1291_, lean_object* v_inst_1292_, lean_object* v_c_1293_, lean_object* v_a_1294_){
_start:
{
lean_object* v___x_1295_; 
v___x_1295_ = lp_mathlib_OreLocalization_hsmul___redArg(v_inst_1288_, v_inst_1290_, v_inst_1291_, v_inst_1292_, v_c_1293_, v_a_1294_);
return v___x_1295_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_hsmul___boxed(lean_object* v_R_1296_, lean_object* v_M_1297_, lean_object* v_X_1298_, lean_object* v_inst_1299_, lean_object* v_S_1300_, lean_object* v_inst_1301_, lean_object* v_inst_1302_, lean_object* v_inst_1303_, lean_object* v_c_1304_, lean_object* v_a_1305_){
_start:
{
lean_object* v_res_1306_; 
v_res_1306_ = lp_mathlib_OreLocalization_hsmul(v_R_1296_, v_M_1297_, v_X_1298_, v_inst_1299_, v_S_1300_, v_inst_1301_, v_inst_1302_, v_inst_1303_, v_c_1304_, v_a_1305_);
lean_dec_ref(v_inst_1299_);
return v_res_1306_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_hvadd___redArg___lam__0(lean_object* v_inst_1307_, lean_object* v_c_1308_, lean_object* v_toZero_1309_, lean_object* v_inst_1310_, lean_object* v_inst_1311_, lean_object* v_m_1312_, lean_object* v_s_1313_){
_start:
{
lean_object* v___x_1314_; lean_object* v___x_1315_; lean_object* v___x_1316_; lean_object* v___x_1317_; lean_object* v___x_1318_; 
v___x_1314_ = lean_apply_2(v_inst_1307_, v_c_1308_, v_toZero_1309_);
lean_inc(v_s_1313_);
lean_inc(v___x_1314_);
lean_inc_ref(v_inst_1310_);
v___x_1315_ = lp_mathlib_AddOreLocalization_oreMin___redArg(v_inst_1310_, v___x_1314_, v_s_1313_);
v___x_1316_ = lean_apply_2(v_inst_1311_, v___x_1315_, v_m_1312_);
v___x_1317_ = lp_mathlib_AddOreLocalization_oreSubtra___redArg(v_inst_1310_, v___x_1314_, v_s_1313_);
v___x_1318_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1318_, 0, v___x_1316_);
lean_ctor_set(v___x_1318_, 1, v___x_1317_);
return v___x_1318_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_hvadd___redArg(lean_object* v_inst_1319_, lean_object* v_inst_1320_, lean_object* v_inst_1321_, lean_object* v_inst_1322_, lean_object* v_c_1323_, lean_object* v_a_1324_){
_start:
{
lean_object* v___x_1325_; lean_object* v___x_1326_; lean_object* v_toZero_1327_; lean_object* v___f_1328_; lean_object* v___x_1329_; 
v___x_1325_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_1319_);
v___x_1326_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_1325_);
v_toZero_1327_ = lean_ctor_get(v___x_1326_, 0);
lean_inc(v_toZero_1327_);
lean_dec_ref(v___x_1326_);
v___f_1328_ = lean_alloc_closure((void*)(lp_mathlib_AddOreLocalization_hvadd___redArg___lam__0), 7, 5);
lean_closure_set(v___f_1328_, 0, v_inst_1322_);
lean_closure_set(v___f_1328_, 1, v_c_1323_);
lean_closure_set(v___f_1328_, 2, v_toZero_1327_);
lean_closure_set(v___f_1328_, 3, v_inst_1320_);
lean_closure_set(v___f_1328_, 4, v_inst_1321_);
v___x_1329_ = lp_mathlib_AddOreLocalization_liftExpand___redArg(v___f_1328_, v_a_1324_);
return v___x_1329_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_hvadd___redArg___boxed(lean_object* v_inst_1330_, lean_object* v_inst_1331_, lean_object* v_inst_1332_, lean_object* v_inst_1333_, lean_object* v_c_1334_, lean_object* v_a_1335_){
_start:
{
lean_object* v_res_1336_; 
v_res_1336_ = lp_mathlib_AddOreLocalization_hvadd___redArg(v_inst_1330_, v_inst_1331_, v_inst_1332_, v_inst_1333_, v_c_1334_, v_a_1335_);
lean_dec_ref(v_inst_1330_);
return v_res_1336_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_hvadd(lean_object* v_R_1337_, lean_object* v_M_1338_, lean_object* v_X_1339_, lean_object* v_inst_1340_, lean_object* v_S_1341_, lean_object* v_inst_1342_, lean_object* v_inst_1343_, lean_object* v_inst_1344_, lean_object* v_c_1345_, lean_object* v_a_1346_){
_start:
{
lean_object* v___x_1347_; 
v___x_1347_ = lp_mathlib_AddOreLocalization_hvadd___redArg(v_inst_1340_, v_inst_1342_, v_inst_1343_, v_inst_1344_, v_c_1345_, v_a_1346_);
return v___x_1347_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_hvadd___boxed(lean_object* v_R_1348_, lean_object* v_M_1349_, lean_object* v_X_1350_, lean_object* v_inst_1351_, lean_object* v_S_1352_, lean_object* v_inst_1353_, lean_object* v_inst_1354_, lean_object* v_inst_1355_, lean_object* v_c_1356_, lean_object* v_a_1357_){
_start:
{
lean_object* v_res_1358_; 
v_res_1358_ = lp_mathlib_AddOreLocalization_hvadd(v_R_1348_, v_M_1349_, v_X_1350_, v_inst_1351_, v_S_1352_, v_inst_1353_, v_inst_1354_, v_inst_1355_, v_c_1356_, v_a_1357_);
lean_dec_ref(v_inst_1351_);
return v_res_1358_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instSMulOfIsScalarTower___redArg(lean_object* v_inst_1359_, lean_object* v_S_1360_, lean_object* v_inst_1361_, lean_object* v_inst_1362_, lean_object* v_inst_1363_){
_start:
{
lean_object* v___x_1364_; 
v___x_1364_ = lean_alloc_closure((void*)(lp_mathlib_OreLocalization_hsmul___boxed), 10, 8);
lean_closure_set(v___x_1364_, 0, lean_box(0));
lean_closure_set(v___x_1364_, 1, lean_box(0));
lean_closure_set(v___x_1364_, 2, lean_box(0));
lean_closure_set(v___x_1364_, 3, v_inst_1359_);
lean_closure_set(v___x_1364_, 4, v_S_1360_);
lean_closure_set(v___x_1364_, 5, v_inst_1361_);
lean_closure_set(v___x_1364_, 6, v_inst_1362_);
lean_closure_set(v___x_1364_, 7, v_inst_1363_);
return v___x_1364_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instSMulOfIsScalarTower(lean_object* v_R_1365_, lean_object* v_M_1366_, lean_object* v_X_1367_, lean_object* v_inst_1368_, lean_object* v_S_1369_, lean_object* v_inst_1370_, lean_object* v_inst_1371_, lean_object* v_inst_1372_, lean_object* v_inst_1373_, lean_object* v_inst_1374_, lean_object* v_inst_1375_){
_start:
{
lean_object* v___x_1376_; 
v___x_1376_ = lean_alloc_closure((void*)(lp_mathlib_OreLocalization_hsmul___boxed), 10, 8);
lean_closure_set(v___x_1376_, 0, lean_box(0));
lean_closure_set(v___x_1376_, 1, lean_box(0));
lean_closure_set(v___x_1376_, 2, lean_box(0));
lean_closure_set(v___x_1376_, 3, v_inst_1368_);
lean_closure_set(v___x_1376_, 4, v_S_1369_);
lean_closure_set(v___x_1376_, 5, v_inst_1370_);
lean_closure_set(v___x_1376_, 6, v_inst_1371_);
lean_closure_set(v___x_1376_, 7, v_inst_1373_);
return v___x_1376_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instSMulOfIsScalarTower___boxed(lean_object* v_R_1377_, lean_object* v_M_1378_, lean_object* v_X_1379_, lean_object* v_inst_1380_, lean_object* v_S_1381_, lean_object* v_inst_1382_, lean_object* v_inst_1383_, lean_object* v_inst_1384_, lean_object* v_inst_1385_, lean_object* v_inst_1386_, lean_object* v_inst_1387_){
_start:
{
lean_object* v_res_1388_; 
v_res_1388_ = lp_mathlib_OreLocalization_instSMulOfIsScalarTower(v_R_1377_, v_M_1378_, v_X_1379_, v_inst_1380_, v_S_1381_, v_inst_1382_, v_inst_1383_, v_inst_1384_, v_inst_1385_, v_inst_1386_, v_inst_1387_);
lean_dec(v_inst_1384_);
return v_res_1388_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_instVAddOfVAddAssocClass___redArg(lean_object* v_inst_1389_, lean_object* v_S_1390_, lean_object* v_inst_1391_, lean_object* v_inst_1392_, lean_object* v_inst_1393_){
_start:
{
lean_object* v___x_1394_; 
v___x_1394_ = lean_alloc_closure((void*)(lp_mathlib_AddOreLocalization_hvadd___boxed), 10, 8);
lean_closure_set(v___x_1394_, 0, lean_box(0));
lean_closure_set(v___x_1394_, 1, lean_box(0));
lean_closure_set(v___x_1394_, 2, lean_box(0));
lean_closure_set(v___x_1394_, 3, v_inst_1389_);
lean_closure_set(v___x_1394_, 4, v_S_1390_);
lean_closure_set(v___x_1394_, 5, v_inst_1391_);
lean_closure_set(v___x_1394_, 6, v_inst_1392_);
lean_closure_set(v___x_1394_, 7, v_inst_1393_);
return v___x_1394_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_instVAddOfVAddAssocClass(lean_object* v_R_1395_, lean_object* v_M_1396_, lean_object* v_X_1397_, lean_object* v_inst_1398_, lean_object* v_S_1399_, lean_object* v_inst_1400_, lean_object* v_inst_1401_, lean_object* v_inst_1402_, lean_object* v_inst_1403_, lean_object* v_inst_1404_, lean_object* v_inst_1405_){
_start:
{
lean_object* v___x_1406_; 
v___x_1406_ = lean_alloc_closure((void*)(lp_mathlib_AddOreLocalization_hvadd___boxed), 10, 8);
lean_closure_set(v___x_1406_, 0, lean_box(0));
lean_closure_set(v___x_1406_, 1, lean_box(0));
lean_closure_set(v___x_1406_, 2, lean_box(0));
lean_closure_set(v___x_1406_, 3, v_inst_1398_);
lean_closure_set(v___x_1406_, 4, v_S_1399_);
lean_closure_set(v___x_1406_, 5, v_inst_1400_);
lean_closure_set(v___x_1406_, 6, v_inst_1401_);
lean_closure_set(v___x_1406_, 7, v_inst_1403_);
return v___x_1406_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_instVAddOfVAddAssocClass___boxed(lean_object* v_R_1407_, lean_object* v_M_1408_, lean_object* v_X_1409_, lean_object* v_inst_1410_, lean_object* v_S_1411_, lean_object* v_inst_1412_, lean_object* v_inst_1413_, lean_object* v_inst_1414_, lean_object* v_inst_1415_, lean_object* v_inst_1416_, lean_object* v_inst_1417_){
_start:
{
lean_object* v_res_1418_; 
v_res_1418_ = lp_mathlib_AddOreLocalization_instVAddOfVAddAssocClass(v_R_1407_, v_M_1408_, v_X_1409_, v_inst_1410_, v_S_1411_, v_inst_1412_, v_inst_1413_, v_inst_1414_, v_inst_1415_, v_inst_1416_, v_inst_1417_);
lean_dec(v_inst_1414_);
return v_res_1418_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instMulActionOfIsScalarTower___redArg(lean_object* v_inst_1419_, lean_object* v_S_1420_, lean_object* v_inst_1421_, lean_object* v_inst_1422_, lean_object* v_inst_1423_){
_start:
{
lean_object* v___x_1424_; 
v___x_1424_ = lean_alloc_closure((void*)(lp_mathlib_OreLocalization_hsmul___boxed), 10, 8);
lean_closure_set(v___x_1424_, 0, lean_box(0));
lean_closure_set(v___x_1424_, 1, lean_box(0));
lean_closure_set(v___x_1424_, 2, lean_box(0));
lean_closure_set(v___x_1424_, 3, v_inst_1419_);
lean_closure_set(v___x_1424_, 4, v_S_1420_);
lean_closure_set(v___x_1424_, 5, v_inst_1421_);
lean_closure_set(v___x_1424_, 6, v_inst_1422_);
lean_closure_set(v___x_1424_, 7, v_inst_1423_);
return v___x_1424_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instMulActionOfIsScalarTower(lean_object* v_M_1425_, lean_object* v_X_1426_, lean_object* v_inst_1427_, lean_object* v_S_1428_, lean_object* v_inst_1429_, lean_object* v_inst_1430_, lean_object* v_R_1431_, lean_object* v_inst_1432_, lean_object* v_inst_1433_, lean_object* v_inst_1434_, lean_object* v_inst_1435_, lean_object* v_inst_1436_){
_start:
{
lean_object* v___x_1437_; 
v___x_1437_ = lean_alloc_closure((void*)(lp_mathlib_OreLocalization_hsmul___boxed), 10, 8);
lean_closure_set(v___x_1437_, 0, lean_box(0));
lean_closure_set(v___x_1437_, 1, lean_box(0));
lean_closure_set(v___x_1437_, 2, lean_box(0));
lean_closure_set(v___x_1437_, 3, v_inst_1427_);
lean_closure_set(v___x_1437_, 4, v_S_1428_);
lean_closure_set(v___x_1437_, 5, v_inst_1429_);
lean_closure_set(v___x_1437_, 6, v_inst_1430_);
lean_closure_set(v___x_1437_, 7, v_inst_1433_);
return v___x_1437_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instMulActionOfIsScalarTower___boxed(lean_object* v_M_1438_, lean_object* v_X_1439_, lean_object* v_inst_1440_, lean_object* v_S_1441_, lean_object* v_inst_1442_, lean_object* v_inst_1443_, lean_object* v_R_1444_, lean_object* v_inst_1445_, lean_object* v_inst_1446_, lean_object* v_inst_1447_, lean_object* v_inst_1448_, lean_object* v_inst_1449_){
_start:
{
lean_object* v_res_1450_; 
v_res_1450_ = lp_mathlib_OreLocalization_instMulActionOfIsScalarTower(v_M_1438_, v_X_1439_, v_inst_1440_, v_S_1441_, v_inst_1442_, v_inst_1443_, v_R_1444_, v_inst_1445_, v_inst_1446_, v_inst_1447_, v_inst_1448_, v_inst_1449_);
lean_dec(v_inst_1448_);
lean_dec_ref(v_inst_1445_);
return v_res_1450_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_instAddActionOfVAddAssocClass___redArg(lean_object* v_inst_1451_, lean_object* v_S_1452_, lean_object* v_inst_1453_, lean_object* v_inst_1454_, lean_object* v_inst_1455_){
_start:
{
lean_object* v___x_1456_; 
v___x_1456_ = lean_alloc_closure((void*)(lp_mathlib_AddOreLocalization_hvadd___boxed), 10, 8);
lean_closure_set(v___x_1456_, 0, lean_box(0));
lean_closure_set(v___x_1456_, 1, lean_box(0));
lean_closure_set(v___x_1456_, 2, lean_box(0));
lean_closure_set(v___x_1456_, 3, v_inst_1451_);
lean_closure_set(v___x_1456_, 4, v_S_1452_);
lean_closure_set(v___x_1456_, 5, v_inst_1453_);
lean_closure_set(v___x_1456_, 6, v_inst_1454_);
lean_closure_set(v___x_1456_, 7, v_inst_1455_);
return v___x_1456_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_instAddActionOfVAddAssocClass(lean_object* v_M_1457_, lean_object* v_X_1458_, lean_object* v_inst_1459_, lean_object* v_S_1460_, lean_object* v_inst_1461_, lean_object* v_inst_1462_, lean_object* v_R_1463_, lean_object* v_inst_1464_, lean_object* v_inst_1465_, lean_object* v_inst_1466_, lean_object* v_inst_1467_, lean_object* v_inst_1468_){
_start:
{
lean_object* v___x_1469_; 
v___x_1469_ = lean_alloc_closure((void*)(lp_mathlib_AddOreLocalization_hvadd___boxed), 10, 8);
lean_closure_set(v___x_1469_, 0, lean_box(0));
lean_closure_set(v___x_1469_, 1, lean_box(0));
lean_closure_set(v___x_1469_, 2, lean_box(0));
lean_closure_set(v___x_1469_, 3, v_inst_1459_);
lean_closure_set(v___x_1469_, 4, v_S_1460_);
lean_closure_set(v___x_1469_, 5, v_inst_1461_);
lean_closure_set(v___x_1469_, 6, v_inst_1462_);
lean_closure_set(v___x_1469_, 7, v_inst_1465_);
return v___x_1469_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_instAddActionOfVAddAssocClass___boxed(lean_object* v_M_1470_, lean_object* v_X_1471_, lean_object* v_inst_1472_, lean_object* v_S_1473_, lean_object* v_inst_1474_, lean_object* v_inst_1475_, lean_object* v_R_1476_, lean_object* v_inst_1477_, lean_object* v_inst_1478_, lean_object* v_inst_1479_, lean_object* v_inst_1480_, lean_object* v_inst_1481_){
_start:
{
lean_object* v_res_1482_; 
v_res_1482_ = lp_mathlib_AddOreLocalization_instAddActionOfVAddAssocClass(v_M_1470_, v_X_1471_, v_inst_1472_, v_S_1473_, v_inst_1474_, v_inst_1475_, v_R_1476_, v_inst_1477_, v_inst_1478_, v_inst_1479_, v_inst_1480_, v_inst_1481_);
lean_dec(v_inst_1480_);
lean_dec_ref(v_inst_1477_);
return v_res_1482_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instCommMonoid___redArg(lean_object* v_inst_1483_, lean_object* v_S_1484_, lean_object* v_inst_1485_){
_start:
{
lean_object* v___x_1486_; 
v___x_1486_ = lp_mathlib_OreLocalization_instMonoid___redArg(v_inst_1483_, v_S_1484_, v_inst_1485_);
return v___x_1486_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instCommMonoid(lean_object* v_R_1487_, lean_object* v_inst_1488_, lean_object* v_S_1489_, lean_object* v_inst_1490_){
_start:
{
lean_object* v___x_1491_; 
v___x_1491_ = lp_mathlib_OreLocalization_instMonoid___redArg(v_inst_1488_, v_S_1489_, v_inst_1490_);
return v___x_1491_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_instAddCommMonoid___redArg(lean_object* v_inst_1492_, lean_object* v_S_1493_, lean_object* v_inst_1494_){
_start:
{
lean_object* v___x_1495_; 
v___x_1495_ = lp_mathlib_AddOreLocalization_instAddMonoid___redArg(v_inst_1492_, v_S_1493_, v_inst_1494_);
return v___x_1495_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOreLocalization_instAddCommMonoid(lean_object* v_R_1496_, lean_object* v_inst_1497_, lean_object* v_S_1498_, lean_object* v_inst_1499_){
_start:
{
lean_object* v___x_1500_; 
v___x_1500_ = lp_mathlib_AddOreLocalization_instAddMonoid___redArg(v_inst_1497_, v_S_1498_, v_inst_1499_);
return v___x_1500_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_zero___redArg(lean_object* v_inst_1501_, lean_object* v_inst_1502_){
_start:
{
lean_object* v___x_1503_; lean_object* v___x_1504_; lean_object* v_toOne_1505_; lean_object* v___x_1507_; uint8_t v_isShared_1508_; uint8_t v_isSharedCheck_1512_; 
v___x_1503_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_1501_);
v___x_1504_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_1503_);
v_toOne_1505_ = lean_ctor_get(v___x_1504_, 0);
v_isSharedCheck_1512_ = !lean_is_exclusive(v___x_1504_);
if (v_isSharedCheck_1512_ == 0)
{
lean_object* v_unused_1513_; 
v_unused_1513_ = lean_ctor_get(v___x_1504_, 1);
lean_dec(v_unused_1513_);
v___x_1507_ = v___x_1504_;
v_isShared_1508_ = v_isSharedCheck_1512_;
goto v_resetjp_1506_;
}
else
{
lean_inc(v_toOne_1505_);
lean_dec(v___x_1504_);
v___x_1507_ = lean_box(0);
v_isShared_1508_ = v_isSharedCheck_1512_;
goto v_resetjp_1506_;
}
v_resetjp_1506_:
{
lean_object* v___x_1510_; 
if (v_isShared_1508_ == 0)
{
lean_ctor_set(v___x_1507_, 1, v_toOne_1505_);
lean_ctor_set(v___x_1507_, 0, v_inst_1502_);
v___x_1510_ = v___x_1507_;
goto v_reusejp_1509_;
}
else
{
lean_object* v_reuseFailAlloc_1511_; 
v_reuseFailAlloc_1511_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1511_, 0, v_inst_1502_);
lean_ctor_set(v_reuseFailAlloc_1511_, 1, v_toOne_1505_);
v___x_1510_ = v_reuseFailAlloc_1511_;
goto v_reusejp_1509_;
}
v_reusejp_1509_:
{
return v___x_1510_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_zero___redArg___boxed(lean_object* v_inst_1514_, lean_object* v_inst_1515_){
_start:
{
lean_object* v_res_1516_; 
v_res_1516_ = lp_mathlib_OreLocalization_zero___redArg(v_inst_1514_, v_inst_1515_);
lean_dec_ref(v_inst_1514_);
return v_res_1516_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_zero(lean_object* v_R_1517_, lean_object* v_inst_1518_, lean_object* v_S_1519_, lean_object* v_inst_1520_, lean_object* v_X_1521_, lean_object* v_inst_1522_, lean_object* v_inst_1523_){
_start:
{
lean_object* v___x_1524_; 
v___x_1524_ = lp_mathlib_OreLocalization_zero___redArg(v_inst_1518_, v_inst_1522_);
return v___x_1524_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_zero___boxed(lean_object* v_R_1525_, lean_object* v_inst_1526_, lean_object* v_S_1527_, lean_object* v_inst_1528_, lean_object* v_X_1529_, lean_object* v_inst_1530_, lean_object* v_inst_1531_){
_start:
{
lean_object* v_res_1532_; 
v_res_1532_ = lp_mathlib_OreLocalization_zero(v_R_1525_, v_inst_1526_, v_S_1527_, v_inst_1528_, v_X_1529_, v_inst_1530_, v_inst_1531_);
lean_dec(v_inst_1531_);
lean_dec_ref(v_inst_1528_);
lean_dec_ref(v_inst_1526_);
return v_res_1532_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instZero___redArg(lean_object* v_inst_1533_, lean_object* v_inst_1534_){
_start:
{
lean_object* v___x_1535_; 
v___x_1535_ = lp_mathlib_OreLocalization_zero___redArg(v_inst_1533_, v_inst_1534_);
return v___x_1535_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instZero___redArg___boxed(lean_object* v_inst_1536_, lean_object* v_inst_1537_){
_start:
{
lean_object* v_res_1538_; 
v_res_1538_ = lp_mathlib_OreLocalization_instZero___redArg(v_inst_1536_, v_inst_1537_);
lean_dec_ref(v_inst_1536_);
return v_res_1538_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instZero(lean_object* v_R_1539_, lean_object* v_inst_1540_, lean_object* v_S_1541_, lean_object* v_inst_1542_, lean_object* v_X_1543_, lean_object* v_inst_1544_, lean_object* v_inst_1545_){
_start:
{
lean_object* v___x_1546_; 
v___x_1546_ = lp_mathlib_OreLocalization_zero___redArg(v_inst_1540_, v_inst_1544_);
return v___x_1546_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instZero___boxed(lean_object* v_R_1547_, lean_object* v_inst_1548_, lean_object* v_S_1549_, lean_object* v_inst_1550_, lean_object* v_X_1551_, lean_object* v_inst_1552_, lean_object* v_inst_1553_){
_start:
{
lean_object* v_res_1554_; 
v_res_1554_ = lp_mathlib_OreLocalization_instZero(v_R_1547_, v_inst_1548_, v_S_1549_, v_inst_1550_, v_X_1551_, v_inst_1552_, v_inst_1553_);
lean_dec(v_inst_1553_);
lean_dec_ref(v_inst_1550_);
lean_dec_ref(v_inst_1548_);
return v_res_1554_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_GroupTheory_OreLocalization_OreSet(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Common(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Submonoid_MulAction(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Units_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Attr_Core(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_GroupTheory_OreLocalization_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_GroupTheory_OreLocalization_OreSet(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Common(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Submonoid_MulAction(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Units_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Attr_Core(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_GroupTheory_OreLocalization_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_GroupTheory_OreLocalization_OreSet(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Common(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Submonoid_MulAction(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Units_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Attr_Core(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_GroupTheory_OreLocalization_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_GroupTheory_OreLocalization_OreSet(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Common(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Submonoid_MulAction(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Units_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Attr_Core(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_GroupTheory_OreLocalization_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_GroupTheory_OreLocalization_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_GroupTheory_OreLocalization_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
