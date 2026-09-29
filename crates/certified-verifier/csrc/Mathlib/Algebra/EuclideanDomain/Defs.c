// Lean compiler output
// Module: Mathlib.Algebra.EuclideanDomain.Defs
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Ring.Defs public import Mathlib.Order.RelClasses
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
lean_object* l_Lean_Name_str___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_num___override(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_instDistribOfSemiring___redArg(lean_object*);
lean_object* lp_mathlib_instMulZeroClassOfSemiring___redArg(lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* lp_mathlib_Ring_toAddGroupWithOne___redArg(lean_object*);
lean_object* lp_mathlib_AddGroupWithOne_toAddGroup___redArg(lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain_term___u227a___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "_private"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain_term___u227a___00__closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain_term___u227a___00__closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain_term___u227a___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain_term___u227a___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(103, 214, 75, 80, 34, 198, 193, 153)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain_term___u227a___00__closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain_term___u227a___00__closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain_term___u227a___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain_term___u227a___00__closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain_term___u227a___00__closed__2_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain_term___u227a___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain_term___u227a___00__closed__1_value),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain_term___u227a___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(234, 232, 174, 134, 127, 136, 69, 92)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain_term___u227a___00__closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain_term___u227a___00__closed__3_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain_term___u227a___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Algebra"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain_term___u227a___00__closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain_term___u227a___00__closed__4_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain_term___u227a___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain_term___u227a___00__closed__3_value),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain_term___u227a___00__closed__4_value),LEAN_SCALAR_PTR_LITERAL(242, 212, 98, 212, 98, 99, 115, 158)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain_term___u227a___00__closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain_term___u227a___00__closed__5_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain_term___u227a___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "EuclideanDomain"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain_term___u227a___00__closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain_term___u227a___00__closed__6_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain_term___u227a___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain_term___u227a___00__closed__5_value),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain_term___u227a___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(119, 146, 51, 98, 243, 243, 212, 249)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain_term___u227a___00__closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain_term___u227a___00__closed__7_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain_term___u227a___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Defs"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain_term___u227a___00__closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain_term___u227a___00__closed__8_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain_term___u227a___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain_term___u227a___00__closed__7_value),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain_term___u227a___00__closed__8_value),LEAN_SCALAR_PTR_LITERAL(187, 218, 128, 2, 110, 26, 23, 156)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain_term___u227a___00__closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain_term___u227a___00__closed__9_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain_term___u227a___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain_term___u227a___00__closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(198, 167, 101, 6, 166, 88, 154, 44)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain_term___u227a___00__closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain_term___u227a___00__closed__10_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain_term___u227a___00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain_term___u227a___00__closed__10_value),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain_term___u227a___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(27, 65, 254, 14, 36, 93, 208, 54)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain_term___u227a___00__closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain_term___u227a___00__closed__11_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain_term___u227a___00__closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 7, .m_data = "term_≺_"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain_term___u227a___00__closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain_term___u227a___00__closed__12_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain_term___u227a___00__closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain_term___u227a___00__closed__11_value),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain_term___u227a___00__closed__12_value),LEAN_SCALAR_PTR_LITERAL(3, 54, 234, 36, 210, 248, 14, 157)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain_term___u227a___00__closed__13 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain_term___u227a___00__closed__13_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain_term___u227a___00__closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain_term___u227a___00__closed__14 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain_term___u227a___00__closed__14_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain_term___u227a___00__closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain_term___u227a___00__closed__14_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain_term___u227a___00__closed__15 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain_term___u227a___00__closed__15_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain_term___u227a___00__closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 3, .m_data = " ≺ "};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain_term___u227a___00__closed__16 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain_term___u227a___00__closed__16_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain_term___u227a___00__closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain_term___u227a___00__closed__16_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain_term___u227a___00__closed__17 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain_term___u227a___00__closed__17_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain_term___u227a___00__closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain_term___u227a___00__closed__18 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain_term___u227a___00__closed__18_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain_term___u227a___00__closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain_term___u227a___00__closed__18_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain_term___u227a___00__closed__19 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain_term___u227a___00__closed__19_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain_term___u227a___00__closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain_term___u227a___00__closed__19_value),((lean_object*)(((size_t)(51) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain_term___u227a___00__closed__20 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain_term___u227a___00__closed__20_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain_term___u227a___00__closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain_term___u227a___00__closed__15_value),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain_term___u227a___00__closed__17_value),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain_term___u227a___00__closed__20_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain_term___u227a___00__closed__21 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain_term___u227a___00__closed__21_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain_term___u227a___00__closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain_term___u227a___00__closed__13_value),((lean_object*)(((size_t)(50) << 1) | 1)),((lean_object*)(((size_t)(50) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain_term___u227a___00__closed__21_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain_term___u227a___00__closed__22 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain_term___u227a___00__closed__22_value;
LEAN_EXPORT const lean_object* lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain_term___u227a__ = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain_term___u227a___00__closed__22_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Defs______macroRules____private__Mathlib__Algebra__EuclideanDomain__Defs__0__EuclideanDomain__term___u227a____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Defs______macroRules____private__Mathlib__Algebra__EuclideanDomain__Defs__0__EuclideanDomain__term___u227a____1___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Defs______macroRules____private__Mathlib__Algebra__EuclideanDomain__Defs__0__EuclideanDomain__term___u227a____1___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Defs______macroRules____private__Mathlib__Algebra__EuclideanDomain__Defs__0__EuclideanDomain__term___u227a____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Defs______macroRules____private__Mathlib__Algebra__EuclideanDomain__Defs__0__EuclideanDomain__term___u227a____1___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Defs______macroRules____private__Mathlib__Algebra__EuclideanDomain__Defs__0__EuclideanDomain__term___u227a____1___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Defs______macroRules____private__Mathlib__Algebra__EuclideanDomain__Defs__0__EuclideanDomain__term___u227a____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Defs______macroRules____private__Mathlib__Algebra__EuclideanDomain__Defs__0__EuclideanDomain__term___u227a____1___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Defs______macroRules____private__Mathlib__Algebra__EuclideanDomain__Defs__0__EuclideanDomain__term___u227a____1___closed__2_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Defs______macroRules____private__Mathlib__Algebra__EuclideanDomain__Defs__0__EuclideanDomain__term___u227a____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Defs______macroRules____private__Mathlib__Algebra__EuclideanDomain__Defs__0__EuclideanDomain__term___u227a____1___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Defs______macroRules____private__Mathlib__Algebra__EuclideanDomain__Defs__0__EuclideanDomain__term___u227a____1___closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Defs______macroRules____private__Mathlib__Algebra__EuclideanDomain__Defs__0__EuclideanDomain__term___u227a____1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Defs______macroRules____private__Mathlib__Algebra__EuclideanDomain__Defs__0__EuclideanDomain__term___u227a____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Defs______macroRules____private__Mathlib__Algebra__EuclideanDomain__Defs__0__EuclideanDomain__term___u227a____1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Defs______macroRules____private__Mathlib__Algebra__EuclideanDomain__Defs__0__EuclideanDomain__term___u227a____1___closed__4_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Defs______macroRules____private__Mathlib__Algebra__EuclideanDomain__Defs__0__EuclideanDomain__term___u227a____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Defs______macroRules____private__Mathlib__Algebra__EuclideanDomain__Defs__0__EuclideanDomain__term___u227a____1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Defs______macroRules____private__Mathlib__Algebra__EuclideanDomain__Defs__0__EuclideanDomain__term___u227a____1___closed__4_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Defs______macroRules____private__Mathlib__Algebra__EuclideanDomain__Defs__0__EuclideanDomain__term___u227a____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Defs______macroRules____private__Mathlib__Algebra__EuclideanDomain__Defs__0__EuclideanDomain__term___u227a____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Defs______macroRules____private__Mathlib__Algebra__EuclideanDomain__Defs__0__EuclideanDomain__term___u227a____1___closed__4_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Defs______macroRules____private__Mathlib__Algebra__EuclideanDomain__Defs__0__EuclideanDomain__term___u227a____1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Defs______macroRules____private__Mathlib__Algebra__EuclideanDomain__Defs__0__EuclideanDomain__term___u227a____1___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Defs______macroRules____private__Mathlib__Algebra__EuclideanDomain__Defs__0__EuclideanDomain__term___u227a____1___closed__4_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Defs______macroRules____private__Mathlib__Algebra__EuclideanDomain__Defs__0__EuclideanDomain__term___u227a____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "EuclideanDomain.r"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Defs______macroRules____private__Mathlib__Algebra__EuclideanDomain__Defs__0__EuclideanDomain__term___u227a____1___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Defs______macroRules____private__Mathlib__Algebra__EuclideanDomain__Defs__0__EuclideanDomain__term___u227a____1___closed__5_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Defs______macroRules____private__Mathlib__Algebra__EuclideanDomain__Defs__0__EuclideanDomain__term___u227a____1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Defs______macroRules____private__Mathlib__Algebra__EuclideanDomain__Defs__0__EuclideanDomain__term___u227a____1___closed__6;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Defs______macroRules____private__Mathlib__Algebra__EuclideanDomain__Defs__0__EuclideanDomain__term___u227a____1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "r"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Defs______macroRules____private__Mathlib__Algebra__EuclideanDomain__Defs__0__EuclideanDomain__term___u227a____1___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Defs______macroRules____private__Mathlib__Algebra__EuclideanDomain__Defs__0__EuclideanDomain__term___u227a____1___closed__7_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Defs______macroRules____private__Mathlib__Algebra__EuclideanDomain__Defs__0__EuclideanDomain__term___u227a____1___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain_term___u227a___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(250, 176, 222, 37, 22, 220, 155, 159)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Defs______macroRules____private__Mathlib__Algebra__EuclideanDomain__Defs__0__EuclideanDomain__term___u227a____1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Defs______macroRules____private__Mathlib__Algebra__EuclideanDomain__Defs__0__EuclideanDomain__term___u227a____1___closed__8_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Defs______macroRules____private__Mathlib__Algebra__EuclideanDomain__Defs__0__EuclideanDomain__term___u227a____1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(244, 25, 245, 32, 120, 145, 151, 47)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Defs______macroRules____private__Mathlib__Algebra__EuclideanDomain__Defs__0__EuclideanDomain__term___u227a____1___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Defs______macroRules____private__Mathlib__Algebra__EuclideanDomain__Defs__0__EuclideanDomain__term___u227a____1___closed__8_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Defs______macroRules____private__Mathlib__Algebra__EuclideanDomain__Defs__0__EuclideanDomain__term___u227a____1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Defs______macroRules____private__Mathlib__Algebra__EuclideanDomain__Defs__0__EuclideanDomain__term___u227a____1___closed__8_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Defs______macroRules____private__Mathlib__Algebra__EuclideanDomain__Defs__0__EuclideanDomain__term___u227a____1___closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Defs______macroRules____private__Mathlib__Algebra__EuclideanDomain__Defs__0__EuclideanDomain__term___u227a____1___closed__9_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Defs______macroRules____private__Mathlib__Algebra__EuclideanDomain__Defs__0__EuclideanDomain__term___u227a____1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Defs______macroRules____private__Mathlib__Algebra__EuclideanDomain__Defs__0__EuclideanDomain__term___u227a____1___closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Defs______macroRules____private__Mathlib__Algebra__EuclideanDomain__Defs__0__EuclideanDomain__term___u227a____1___closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Defs______macroRules____private__Mathlib__Algebra__EuclideanDomain__Defs__0__EuclideanDomain__term___u227a____1___closed__10_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Defs______macroRules____private__Mathlib__Algebra__EuclideanDomain__Defs__0__EuclideanDomain__term___u227a____1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Defs______macroRules____private__Mathlib__Algebra__EuclideanDomain__Defs__0__EuclideanDomain__term___u227a____1___closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Defs______macroRules____private__Mathlib__Algebra__EuclideanDomain__Defs__0__EuclideanDomain__term___u227a____1___closed__11_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Defs______macroRules____private__Mathlib__Algebra__EuclideanDomain__Defs__0__EuclideanDomain__term___u227a____1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Defs______macroRules____private__Mathlib__Algebra__EuclideanDomain__Defs__0__EuclideanDomain__term___u227a____1___closed__11_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Defs______macroRules____private__Mathlib__Algebra__EuclideanDomain__Defs__0__EuclideanDomain__term___u227a____1___closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Defs______macroRules____private__Mathlib__Algebra__EuclideanDomain__Defs__0__EuclideanDomain__term___u227a____1___closed__12_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Defs______macroRules____private__Mathlib__Algebra__EuclideanDomain__Defs__0__EuclideanDomain__term___u227a____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Defs______macroRules____private__Mathlib__Algebra__EuclideanDomain__Defs__0__EuclideanDomain__term___u227a____1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Defs______unexpand__EuclideanDomain__r__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Defs______unexpand__EuclideanDomain__r__1___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Defs______unexpand__EuclideanDomain__r__1___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Defs______unexpand__EuclideanDomain__r__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Defs______unexpand__EuclideanDomain__r__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Defs______unexpand__EuclideanDomain__r__1___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Defs______unexpand__EuclideanDomain__r__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Defs______unexpand__EuclideanDomain__r__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Defs______unexpand__EuclideanDomain__r__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_EuclideanDomain_wellFoundedRelation(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_EuclideanDomain_wellFoundedRelation___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_EuclideanDomain_instDiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_EuclideanDomain_instDiv___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_EuclideanDomain_instDiv(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_EuclideanDomain_instDiv___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_EuclideanDomain_instMod___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_EuclideanDomain_instMod___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_EuclideanDomain_instMod(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_EuclideanDomain_instMod___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_EuclideanDomain_gcd___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_EuclideanDomain_gcd(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_EuclideanDomain_xgcdAux___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_EuclideanDomain_xgcdAux(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_EuclideanDomain_xgcd___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_EuclideanDomain_xgcd(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_EuclideanDomain_gcdA___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_EuclideanDomain_gcdA(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_EuclideanDomain_gcdB___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_EuclideanDomain_gcdB(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_EuclideanDomain_lcm___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_EuclideanDomain_lcm(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Defs______macroRules____private__Mathlib__Algebra__EuclideanDomain__Defs__0__EuclideanDomain__term___u227a____1___closed__6(void){
_start:
{
lean_object* v___x_62_; lean_object* v___x_63_; 
v___x_62_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Defs______macroRules____private__Mathlib__Algebra__EuclideanDomain__Defs__0__EuclideanDomain__term___u227a____1___closed__5));
v___x_63_ = l_String_toRawSubstring_x27(v___x_62_);
return v___x_63_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Defs______macroRules____private__Mathlib__Algebra__EuclideanDomain__Defs__0__EuclideanDomain__term___u227a____1(lean_object* v_x_77_, lean_object* v_a_78_, lean_object* v_a_79_){
_start:
{
lean_object* v___x_80_; lean_object* v___x_81_; uint8_t v___x_82_; 
v___x_80_ = lean_unsigned_to_nat(0u);
v___x_81_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain_term___u227a___00__closed__13));
lean_inc(v_x_77_);
v___x_82_ = l_Lean_Syntax_isOfKind(v_x_77_, v___x_81_);
if (v___x_82_ == 0)
{
lean_object* v___x_83_; lean_object* v___x_84_; 
lean_dec(v_x_77_);
v___x_83_ = lean_box(1);
v___x_84_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_84_, 0, v___x_83_);
lean_ctor_set(v___x_84_, 1, v_a_79_);
return v___x_84_;
}
else
{
lean_object* v_quotContext_85_; lean_object* v_currMacroScope_86_; lean_object* v_ref_87_; lean_object* v___x_88_; lean_object* v___x_89_; lean_object* v___x_90_; uint8_t v___x_91_; lean_object* v___x_92_; lean_object* v___x_93_; lean_object* v___x_94_; lean_object* v___x_95_; lean_object* v___x_96_; lean_object* v___x_97_; lean_object* v___x_98_; lean_object* v___x_99_; lean_object* v___x_100_; lean_object* v___x_101_; lean_object* v___x_102_; 
v_quotContext_85_ = lean_ctor_get(v_a_78_, 1);
v_currMacroScope_86_ = lean_ctor_get(v_a_78_, 2);
v_ref_87_ = lean_ctor_get(v_a_78_, 5);
v___x_88_ = l_Lean_Syntax_getArg(v_x_77_, v___x_80_);
v___x_89_ = lean_unsigned_to_nat(2u);
v___x_90_ = l_Lean_Syntax_getArg(v_x_77_, v___x_89_);
lean_dec(v_x_77_);
v___x_91_ = 0;
v___x_92_ = l_Lean_SourceInfo_fromRef(v_ref_87_, v___x_91_);
v___x_93_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Defs______macroRules____private__Mathlib__Algebra__EuclideanDomain__Defs__0__EuclideanDomain__term___u227a____1___closed__4));
v___x_94_ = lean_obj_once(&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Defs______macroRules____private__Mathlib__Algebra__EuclideanDomain__Defs__0__EuclideanDomain__term___u227a____1___closed__6, &lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Defs______macroRules____private__Mathlib__Algebra__EuclideanDomain__Defs__0__EuclideanDomain__term___u227a____1___closed__6_once, _init_lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Defs______macroRules____private__Mathlib__Algebra__EuclideanDomain__Defs__0__EuclideanDomain__term___u227a____1___closed__6);
v___x_95_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Defs______macroRules____private__Mathlib__Algebra__EuclideanDomain__Defs__0__EuclideanDomain__term___u227a____1___closed__8));
lean_inc(v_currMacroScope_86_);
lean_inc(v_quotContext_85_);
v___x_96_ = l_Lean_addMacroScope(v_quotContext_85_, v___x_95_, v_currMacroScope_86_);
v___x_97_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Defs______macroRules____private__Mathlib__Algebra__EuclideanDomain__Defs__0__EuclideanDomain__term___u227a____1___closed__10));
lean_inc_n(v___x_92_, 2);
v___x_98_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_98_, 0, v___x_92_);
lean_ctor_set(v___x_98_, 1, v___x_94_);
lean_ctor_set(v___x_98_, 2, v___x_96_);
lean_ctor_set(v___x_98_, 3, v___x_97_);
v___x_99_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Defs______macroRules____private__Mathlib__Algebra__EuclideanDomain__Defs__0__EuclideanDomain__term___u227a____1___closed__12));
v___x_100_ = l_Lean_Syntax_node2(v___x_92_, v___x_99_, v___x_88_, v___x_90_);
v___x_101_ = l_Lean_Syntax_node2(v___x_92_, v___x_93_, v___x_98_, v___x_100_);
v___x_102_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_102_, 0, v___x_101_);
lean_ctor_set(v___x_102_, 1, v_a_79_);
return v___x_102_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Defs______macroRules____private__Mathlib__Algebra__EuclideanDomain__Defs__0__EuclideanDomain__term___u227a____1___boxed(lean_object* v_x_103_, lean_object* v_a_104_, lean_object* v_a_105_){
_start:
{
lean_object* v_res_106_; 
v_res_106_ = lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Defs______macroRules____private__Mathlib__Algebra__EuclideanDomain__Defs__0__EuclideanDomain__term___u227a____1(v_x_103_, v_a_104_, v_a_105_);
lean_dec_ref(v_a_104_);
return v_res_106_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Defs______unexpand__EuclideanDomain__r__1(lean_object* v_x_110_, lean_object* v_a_111_, lean_object* v_a_112_){
_start:
{
lean_object* v___x_113_; uint8_t v___x_114_; 
v___x_113_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Defs______macroRules____private__Mathlib__Algebra__EuclideanDomain__Defs__0__EuclideanDomain__term___u227a____1___closed__4));
lean_inc(v_x_110_);
v___x_114_ = l_Lean_Syntax_isOfKind(v_x_110_, v___x_113_);
if (v___x_114_ == 0)
{
lean_object* v___x_115_; lean_object* v___x_116_; 
lean_dec(v_x_110_);
v___x_115_ = lean_box(0);
v___x_116_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_116_, 0, v___x_115_);
lean_ctor_set(v___x_116_, 1, v_a_112_);
return v___x_116_;
}
else
{
lean_object* v___x_117_; lean_object* v___x_118_; lean_object* v___x_119_; uint8_t v___x_120_; 
v___x_117_ = lean_unsigned_to_nat(0u);
v___x_118_ = l_Lean_Syntax_getArg(v_x_110_, v___x_117_);
v___x_119_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Defs______unexpand__EuclideanDomain__r__1___closed__1));
lean_inc(v___x_118_);
v___x_120_ = l_Lean_Syntax_isOfKind(v___x_118_, v___x_119_);
if (v___x_120_ == 0)
{
lean_object* v___x_121_; lean_object* v___x_122_; 
lean_dec(v___x_118_);
lean_dec(v_x_110_);
v___x_121_ = lean_box(0);
v___x_122_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_122_, 0, v___x_121_);
lean_ctor_set(v___x_122_, 1, v_a_112_);
return v___x_122_;
}
else
{
lean_object* v___x_123_; lean_object* v___x_124_; lean_object* v___x_125_; uint8_t v___x_126_; 
v___x_123_ = lean_unsigned_to_nat(1u);
v___x_124_ = l_Lean_Syntax_getArg(v_x_110_, v___x_123_);
lean_dec(v_x_110_);
v___x_125_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_124_);
v___x_126_ = l_Lean_Syntax_matchesNull(v___x_124_, v___x_125_);
if (v___x_126_ == 0)
{
lean_object* v___x_127_; lean_object* v___x_128_; 
lean_dec(v___x_124_);
lean_dec(v___x_118_);
v___x_127_ = lean_box(0);
v___x_128_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_128_, 0, v___x_127_);
lean_ctor_set(v___x_128_, 1, v_a_112_);
return v___x_128_;
}
else
{
lean_object* v___x_129_; lean_object* v___x_130_; lean_object* v_ref_131_; uint8_t v___x_132_; lean_object* v___x_133_; lean_object* v___x_134_; lean_object* v___x_135_; lean_object* v___x_136_; lean_object* v___x_137_; lean_object* v___x_138_; 
v___x_129_ = l_Lean_Syntax_getArg(v___x_124_, v___x_117_);
v___x_130_ = l_Lean_Syntax_getArg(v___x_124_, v___x_123_);
lean_dec(v___x_124_);
v_ref_131_ = l_Lean_replaceRef(v___x_118_, v_a_111_);
lean_dec(v___x_118_);
v___x_132_ = 0;
v___x_133_ = l_Lean_SourceInfo_fromRef(v_ref_131_, v___x_132_);
lean_dec(v_ref_131_);
v___x_134_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain_term___u227a___00__closed__13));
v___x_135_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain_term___u227a___00__closed__16));
lean_inc(v___x_133_);
v___x_136_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_136_, 0, v___x_133_);
lean_ctor_set(v___x_136_, 1, v___x_135_);
v___x_137_ = l_Lean_Syntax_node3(v___x_133_, v___x_134_, v___x_129_, v___x_136_, v___x_130_);
v___x_138_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_138_, 0, v___x_137_);
lean_ctor_set(v___x_138_, 1, v_a_112_);
return v___x_138_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Defs______unexpand__EuclideanDomain__r__1___boxed(lean_object* v_x_139_, lean_object* v_a_140_, lean_object* v_a_141_){
_start:
{
lean_object* v_res_142_; 
v_res_142_ = lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Defs_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Defs______unexpand__EuclideanDomain__r__1(v_x_139_, v_a_140_, v_a_141_);
lean_dec(v_a_140_);
return v_res_142_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_EuclideanDomain_wellFoundedRelation(lean_object* v_R_143_, lean_object* v_inst_144_){
_start:
{
lean_object* v___x_145_; 
v___x_145_ = lean_box(0);
return v___x_145_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_EuclideanDomain_wellFoundedRelation___boxed(lean_object* v_R_146_, lean_object* v_inst_147_){
_start:
{
lean_object* v_res_148_; 
v_res_148_ = lp_mathlib_EuclideanDomain_wellFoundedRelation(v_R_146_, v_inst_147_);
lean_dec_ref(v_inst_147_);
return v_res_148_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_EuclideanDomain_instDiv___redArg(lean_object* v_inst_149_){
_start:
{
lean_object* v_quotient_150_; 
v_quotient_150_ = lean_ctor_get(v_inst_149_, 1);
lean_inc(v_quotient_150_);
return v_quotient_150_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_EuclideanDomain_instDiv___redArg___boxed(lean_object* v_inst_151_){
_start:
{
lean_object* v_res_152_; 
v_res_152_ = lp_mathlib_EuclideanDomain_instDiv___redArg(v_inst_151_);
lean_dec_ref(v_inst_151_);
return v_res_152_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_EuclideanDomain_instDiv(lean_object* v_R_153_, lean_object* v_inst_154_){
_start:
{
lean_object* v_quotient_155_; 
v_quotient_155_ = lean_ctor_get(v_inst_154_, 1);
lean_inc(v_quotient_155_);
return v_quotient_155_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_EuclideanDomain_instDiv___boxed(lean_object* v_R_156_, lean_object* v_inst_157_){
_start:
{
lean_object* v_res_158_; 
v_res_158_ = lp_mathlib_EuclideanDomain_instDiv(v_R_156_, v_inst_157_);
lean_dec_ref(v_inst_157_);
return v_res_158_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_EuclideanDomain_instMod___redArg(lean_object* v_inst_159_){
_start:
{
lean_object* v_remainder_160_; 
v_remainder_160_ = lean_ctor_get(v_inst_159_, 2);
lean_inc(v_remainder_160_);
return v_remainder_160_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_EuclideanDomain_instMod___redArg___boxed(lean_object* v_inst_161_){
_start:
{
lean_object* v_res_162_; 
v_res_162_ = lp_mathlib_EuclideanDomain_instMod___redArg(v_inst_161_);
lean_dec_ref(v_inst_161_);
return v_res_162_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_EuclideanDomain_instMod(lean_object* v_R_163_, lean_object* v_inst_164_){
_start:
{
lean_object* v_remainder_165_; 
v_remainder_165_ = lean_ctor_get(v_inst_164_, 2);
lean_inc(v_remainder_165_);
return v_remainder_165_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_EuclideanDomain_instMod___boxed(lean_object* v_R_166_, lean_object* v_inst_167_){
_start:
{
lean_object* v_res_168_; 
v_res_168_ = lp_mathlib_EuclideanDomain_instMod(v_R_166_, v_inst_167_);
lean_dec_ref(v_inst_167_);
return v_res_168_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_EuclideanDomain_gcd___redArg(lean_object* v_inst_169_, lean_object* v_inst_170_, lean_object* v_a_171_, lean_object* v_b_172_){
_start:
{
lean_object* v_toCommRing_173_; lean_object* v_remainder_174_; lean_object* v_toSemiring_175_; lean_object* v___x_176_; lean_object* v_toZero_177_; lean_object* v___x_178_; uint8_t v___x_179_; 
v_toCommRing_173_ = lean_ctor_get(v_inst_169_, 0);
v_remainder_174_ = lean_ctor_get(v_inst_169_, 2);
v_toSemiring_175_ = lean_ctor_get(v_toCommRing_173_, 0);
lean_inc_ref(v_toSemiring_175_);
v___x_176_ = lp_mathlib_instMulZeroClassOfSemiring___redArg(v_toSemiring_175_);
v_toZero_177_ = lean_ctor_get(v___x_176_, 1);
lean_inc(v_toZero_177_);
lean_dec_ref(v___x_176_);
lean_inc_ref(v_inst_170_);
lean_inc(v_a_171_);
v___x_178_ = lean_apply_2(v_inst_170_, v_a_171_, v_toZero_177_);
v___x_179_ = lean_unbox(v___x_178_);
if (v___x_179_ == 0)
{
lean_object* v___x_180_; 
lean_inc(v_remainder_174_);
lean_inc(v_a_171_);
v___x_180_ = lean_apply_2(v_remainder_174_, v_b_172_, v_a_171_);
{
lean_object* _tmp_2 = v___x_180_;
lean_object* _tmp_3 = v_a_171_;
v_a_171_ = _tmp_2;
v_b_172_ = _tmp_3;
}
goto _start;
}
else
{
lean_dec(v_a_171_);
lean_dec_ref(v_inst_170_);
lean_dec_ref(v_inst_169_);
return v_b_172_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_EuclideanDomain_gcd(lean_object* v_R_182_, lean_object* v_inst_183_, lean_object* v_inst_184_, lean_object* v_a_185_, lean_object* v_b_186_){
_start:
{
lean_object* v___x_187_; 
v___x_187_ = lp_mathlib_EuclideanDomain_gcd___redArg(v_inst_183_, v_inst_184_, v_a_185_, v_b_186_);
return v___x_187_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_EuclideanDomain_xgcdAux___redArg(lean_object* v_inst_188_, lean_object* v_inst_189_, lean_object* v_r_190_, lean_object* v_s_191_, lean_object* v_t_192_, lean_object* v_r_x27_193_, lean_object* v_s_x27_194_, lean_object* v_t_x27_195_){
_start:
{
lean_object* v_toCommRing_196_; lean_object* v_quotient_197_; lean_object* v_remainder_198_; lean_object* v_toSemiring_199_; lean_object* v___x_200_; lean_object* v_toZero_201_; lean_object* v___x_203_; uint8_t v_isShared_204_; uint8_t v_isSharedCheck_223_; 
v_toCommRing_196_ = lean_ctor_get(v_inst_188_, 0);
v_quotient_197_ = lean_ctor_get(v_inst_188_, 1);
v_remainder_198_ = lean_ctor_get(v_inst_188_, 2);
v_toSemiring_199_ = lean_ctor_get(v_toCommRing_196_, 0);
lean_inc_ref(v_toSemiring_199_);
v___x_200_ = lp_mathlib_instMulZeroClassOfSemiring___redArg(v_toSemiring_199_);
v_toZero_201_ = lean_ctor_get(v___x_200_, 1);
v_isSharedCheck_223_ = !lean_is_exclusive(v___x_200_);
if (v_isSharedCheck_223_ == 0)
{
lean_object* v_unused_224_; 
v_unused_224_ = lean_ctor_get(v___x_200_, 0);
lean_dec(v_unused_224_);
v___x_203_ = v___x_200_;
v_isShared_204_ = v_isSharedCheck_223_;
goto v_resetjp_202_;
}
else
{
lean_inc(v_toZero_201_);
lean_dec(v___x_200_);
v___x_203_ = lean_box(0);
v_isShared_204_ = v_isSharedCheck_223_;
goto v_resetjp_202_;
}
v_resetjp_202_:
{
lean_object* v___x_205_; uint8_t v___x_206_; 
lean_inc_ref(v_inst_189_);
lean_inc(v_r_190_);
v___x_205_ = lean_apply_2(v_inst_189_, v_r_190_, v_toZero_201_);
v___x_206_ = lean_unbox(v___x_205_);
if (v___x_206_ == 0)
{
lean_object* v___x_207_; lean_object* v___x_208_; lean_object* v_toSub_209_; lean_object* v___x_210_; lean_object* v_toMul_211_; lean_object* v_q_212_; lean_object* v___x_213_; lean_object* v___x_214_; lean_object* v___x_215_; lean_object* v___x_216_; lean_object* v___x_217_; 
lean_del_object(v___x_203_);
lean_inc_ref(v_toCommRing_196_);
v___x_207_ = lp_mathlib_Ring_toAddGroupWithOne___redArg(v_toCommRing_196_);
v___x_208_ = lp_mathlib_AddGroupWithOne_toAddGroup___redArg(v___x_207_);
lean_dec_ref(v___x_207_);
v_toSub_209_ = lean_ctor_get(v___x_208_, 2);
lean_inc_n(v_toSub_209_, 2);
lean_dec_ref(v___x_208_);
lean_inc_ref(v_toSemiring_199_);
v___x_210_ = lp_mathlib_instDistribOfSemiring___redArg(v_toSemiring_199_);
v_toMul_211_ = lean_ctor_get(v___x_210_, 0);
lean_inc_n(v_toMul_211_, 2);
lean_dec_ref(v___x_210_);
lean_inc(v_quotient_197_);
lean_inc_n(v_r_190_, 2);
lean_inc(v_r_x27_193_);
v_q_212_ = lean_apply_2(v_quotient_197_, v_r_x27_193_, v_r_190_);
lean_inc(v_remainder_198_);
v___x_213_ = lean_apply_2(v_remainder_198_, v_r_x27_193_, v_r_190_);
lean_inc(v_s_191_);
lean_inc(v_q_212_);
v___x_214_ = lean_apply_2(v_toMul_211_, v_q_212_, v_s_191_);
v___x_215_ = lean_apply_2(v_toSub_209_, v_s_x27_194_, v___x_214_);
lean_inc(v_t_192_);
v___x_216_ = lean_apply_2(v_toMul_211_, v_q_212_, v_t_192_);
v___x_217_ = lean_apply_2(v_toSub_209_, v_t_x27_195_, v___x_216_);
{
lean_object* _tmp_2 = v___x_213_;
lean_object* _tmp_3 = v___x_215_;
lean_object* _tmp_4 = v___x_217_;
lean_object* _tmp_5 = v_r_190_;
lean_object* _tmp_6 = v_s_191_;
lean_object* _tmp_7 = v_t_192_;
v_r_190_ = _tmp_2;
v_s_191_ = _tmp_3;
v_t_192_ = _tmp_4;
v_r_x27_193_ = _tmp_5;
v_s_x27_194_ = _tmp_6;
v_t_x27_195_ = _tmp_7;
}
goto _start;
}
else
{
lean_object* v___x_220_; 
lean_dec(v_t_192_);
lean_dec(v_s_191_);
lean_dec(v_r_190_);
lean_dec_ref(v_inst_189_);
lean_dec_ref(v_inst_188_);
if (v_isShared_204_ == 0)
{
lean_ctor_set(v___x_203_, 1, v_t_x27_195_);
lean_ctor_set(v___x_203_, 0, v_s_x27_194_);
v___x_220_ = v___x_203_;
goto v_reusejp_219_;
}
else
{
lean_object* v_reuseFailAlloc_222_; 
v_reuseFailAlloc_222_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_222_, 0, v_s_x27_194_);
lean_ctor_set(v_reuseFailAlloc_222_, 1, v_t_x27_195_);
v___x_220_ = v_reuseFailAlloc_222_;
goto v_reusejp_219_;
}
v_reusejp_219_:
{
lean_object* v___x_221_; 
v___x_221_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_221_, 0, v_r_x27_193_);
lean_ctor_set(v___x_221_, 1, v___x_220_);
return v___x_221_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_EuclideanDomain_xgcdAux(lean_object* v_R_225_, lean_object* v_inst_226_, lean_object* v_inst_227_, lean_object* v_r_228_, lean_object* v_s_229_, lean_object* v_t_230_, lean_object* v_r_x27_231_, lean_object* v_s_x27_232_, lean_object* v_t_x27_233_){
_start:
{
lean_object* v___x_234_; 
v___x_234_ = lp_mathlib_EuclideanDomain_xgcdAux___redArg(v_inst_226_, v_inst_227_, v_r_228_, v_s_229_, v_t_230_, v_r_x27_231_, v_s_x27_232_, v_t_x27_233_);
return v___x_234_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_EuclideanDomain_xgcd___redArg(lean_object* v_inst_235_, lean_object* v_inst_236_, lean_object* v_x_237_, lean_object* v_y_238_){
_start:
{
lean_object* v_toCommRing_239_; lean_object* v___x_240_; lean_object* v_toAddMonoidWithOne_241_; lean_object* v_toOne_242_; lean_object* v_toSemiring_243_; lean_object* v___x_244_; lean_object* v_toZero_245_; lean_object* v___x_246_; lean_object* v_snd_247_; 
v_toCommRing_239_ = lean_ctor_get(v_inst_235_, 0);
lean_inc_ref(v_toCommRing_239_);
v___x_240_ = lp_mathlib_Ring_toAddGroupWithOne___redArg(v_toCommRing_239_);
v_toAddMonoidWithOne_241_ = lean_ctor_get(v___x_240_, 1);
lean_inc_ref(v_toAddMonoidWithOne_241_);
lean_dec_ref(v___x_240_);
v_toOne_242_ = lean_ctor_get(v_toAddMonoidWithOne_241_, 2);
lean_inc_n(v_toOne_242_, 2);
lean_dec_ref(v_toAddMonoidWithOne_241_);
v_toSemiring_243_ = lean_ctor_get(v_toCommRing_239_, 0);
lean_inc_ref(v_toSemiring_243_);
v___x_244_ = lp_mathlib_instMulZeroClassOfSemiring___redArg(v_toSemiring_243_);
v_toZero_245_ = lean_ctor_get(v___x_244_, 1);
lean_inc_n(v_toZero_245_, 2);
lean_dec_ref(v___x_244_);
v___x_246_ = lp_mathlib_EuclideanDomain_xgcdAux___redArg(v_inst_235_, v_inst_236_, v_x_237_, v_toOne_242_, v_toZero_245_, v_y_238_, v_toZero_245_, v_toOne_242_);
v_snd_247_ = lean_ctor_get(v___x_246_, 1);
lean_inc(v_snd_247_);
lean_dec_ref(v___x_246_);
return v_snd_247_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_EuclideanDomain_xgcd(lean_object* v_R_248_, lean_object* v_inst_249_, lean_object* v_inst_250_, lean_object* v_x_251_, lean_object* v_y_252_){
_start:
{
lean_object* v___x_253_; 
v___x_253_ = lp_mathlib_EuclideanDomain_xgcd___redArg(v_inst_249_, v_inst_250_, v_x_251_, v_y_252_);
return v___x_253_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_EuclideanDomain_gcdA___redArg(lean_object* v_inst_254_, lean_object* v_inst_255_, lean_object* v_x_256_, lean_object* v_y_257_){
_start:
{
lean_object* v___x_258_; lean_object* v_fst_259_; 
v___x_258_ = lp_mathlib_EuclideanDomain_xgcd___redArg(v_inst_254_, v_inst_255_, v_x_256_, v_y_257_);
v_fst_259_ = lean_ctor_get(v___x_258_, 0);
lean_inc(v_fst_259_);
lean_dec_ref(v___x_258_);
return v_fst_259_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_EuclideanDomain_gcdA(lean_object* v_R_260_, lean_object* v_inst_261_, lean_object* v_inst_262_, lean_object* v_x_263_, lean_object* v_y_264_){
_start:
{
lean_object* v___x_265_; 
v___x_265_ = lp_mathlib_EuclideanDomain_gcdA___redArg(v_inst_261_, v_inst_262_, v_x_263_, v_y_264_);
return v___x_265_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_EuclideanDomain_gcdB___redArg(lean_object* v_inst_266_, lean_object* v_inst_267_, lean_object* v_x_268_, lean_object* v_y_269_){
_start:
{
lean_object* v___x_270_; lean_object* v_snd_271_; 
v___x_270_ = lp_mathlib_EuclideanDomain_xgcd___redArg(v_inst_266_, v_inst_267_, v_x_268_, v_y_269_);
v_snd_271_ = lean_ctor_get(v___x_270_, 1);
lean_inc(v_snd_271_);
lean_dec_ref(v___x_270_);
return v_snd_271_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_EuclideanDomain_gcdB(lean_object* v_R_272_, lean_object* v_inst_273_, lean_object* v_inst_274_, lean_object* v_x_275_, lean_object* v_y_276_){
_start:
{
lean_object* v___x_277_; 
v___x_277_ = lp_mathlib_EuclideanDomain_gcdB___redArg(v_inst_273_, v_inst_274_, v_x_275_, v_y_276_);
return v___x_277_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_EuclideanDomain_lcm___redArg(lean_object* v_inst_278_, lean_object* v_inst_279_, lean_object* v_x_280_, lean_object* v_y_281_){
_start:
{
lean_object* v_toCommRing_282_; lean_object* v_quotient_283_; lean_object* v_toSemiring_284_; lean_object* v___x_285_; lean_object* v_toMul_286_; lean_object* v___x_287_; lean_object* v___x_288_; lean_object* v___x_289_; 
v_toCommRing_282_ = lean_ctor_get(v_inst_278_, 0);
v_quotient_283_ = lean_ctor_get(v_inst_278_, 1);
lean_inc(v_quotient_283_);
v_toSemiring_284_ = lean_ctor_get(v_toCommRing_282_, 0);
lean_inc_ref(v_toSemiring_284_);
v___x_285_ = lp_mathlib_instDistribOfSemiring___redArg(v_toSemiring_284_);
v_toMul_286_ = lean_ctor_get(v___x_285_, 0);
lean_inc(v_toMul_286_);
lean_dec_ref(v___x_285_);
lean_inc(v_y_281_);
lean_inc(v_x_280_);
v___x_287_ = lean_apply_2(v_toMul_286_, v_x_280_, v_y_281_);
v___x_288_ = lp_mathlib_EuclideanDomain_gcd___redArg(v_inst_278_, v_inst_279_, v_x_280_, v_y_281_);
v___x_289_ = lean_apply_2(v_quotient_283_, v___x_287_, v___x_288_);
return v___x_289_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_EuclideanDomain_lcm(lean_object* v_R_290_, lean_object* v_inst_291_, lean_object* v_inst_292_, lean_object* v_x_293_, lean_object* v_y_294_){
_start:
{
lean_object* v___x_295_; 
v___x_295_ = lp_mathlib_EuclideanDomain_lcm___redArg(v_inst_291_, v_inst_292_, v_x_293_, v_y_294_);
return v___x_295_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_RelClasses(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_EuclideanDomain_Defs(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_RelClasses(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_EuclideanDomain_Defs(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_RelClasses(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_EuclideanDomain_Defs(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_RelClasses(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_EuclideanDomain_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_EuclideanDomain_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_EuclideanDomain_Defs(builtin);
}
#ifdef __cplusplus
}
#endif
