// Lean compiler output
// Module: Mathlib.Algebra.EuclideanDomain.Basic
// Imports: public import Init public meta import Init public import Mathlib.Algebra.EuclideanDomain.Defs public import Mathlib.Algebra.Ring.Divisibility.Basic public import Mathlib.Algebra.GroupWithZero.Divisibility public import Mathlib.Algebra.Ring.Equiv
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
lean_object* l_Lean_Name_str___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_num___override(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain_term___u227a___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "_private"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain_term___u227a___00__closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain_term___u227a___00__closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain_term___u227a___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain_term___u227a___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(103, 214, 75, 80, 34, 198, 193, 153)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain_term___u227a___00__closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain_term___u227a___00__closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain_term___u227a___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain_term___u227a___00__closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain_term___u227a___00__closed__2_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain_term___u227a___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain_term___u227a___00__closed__1_value),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain_term___u227a___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(234, 232, 174, 134, 127, 136, 69, 92)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain_term___u227a___00__closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain_term___u227a___00__closed__3_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain_term___u227a___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Algebra"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain_term___u227a___00__closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain_term___u227a___00__closed__4_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain_term___u227a___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain_term___u227a___00__closed__3_value),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain_term___u227a___00__closed__4_value),LEAN_SCALAR_PTR_LITERAL(242, 212, 98, 212, 98, 99, 115, 158)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain_term___u227a___00__closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain_term___u227a___00__closed__5_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain_term___u227a___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "EuclideanDomain"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain_term___u227a___00__closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain_term___u227a___00__closed__6_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain_term___u227a___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain_term___u227a___00__closed__5_value),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain_term___u227a___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(119, 146, 51, 98, 243, 243, 212, 249)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain_term___u227a___00__closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain_term___u227a___00__closed__7_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain_term___u227a___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Basic"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain_term___u227a___00__closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain_term___u227a___00__closed__8_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain_term___u227a___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain_term___u227a___00__closed__7_value),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain_term___u227a___00__closed__8_value),LEAN_SCALAR_PTR_LITERAL(216, 248, 249, 121, 149, 45, 40, 135)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain_term___u227a___00__closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain_term___u227a___00__closed__9_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain_term___u227a___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain_term___u227a___00__closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(209, 38, 238, 215, 187, 53, 49, 81)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain_term___u227a___00__closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain_term___u227a___00__closed__10_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain_term___u227a___00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain_term___u227a___00__closed__10_value),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain_term___u227a___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(88, 98, 238, 155, 13, 142, 210, 70)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain_term___u227a___00__closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain_term___u227a___00__closed__11_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain_term___u227a___00__closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 7, .m_data = "term_≺_"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain_term___u227a___00__closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain_term___u227a___00__closed__12_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain_term___u227a___00__closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain_term___u227a___00__closed__11_value),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain_term___u227a___00__closed__12_value),LEAN_SCALAR_PTR_LITERAL(132, 127, 79, 123, 232, 225, 213, 69)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain_term___u227a___00__closed__13 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain_term___u227a___00__closed__13_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain_term___u227a___00__closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain_term___u227a___00__closed__14 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain_term___u227a___00__closed__14_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain_term___u227a___00__closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain_term___u227a___00__closed__14_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain_term___u227a___00__closed__15 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain_term___u227a___00__closed__15_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain_term___u227a___00__closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 3, .m_data = " ≺ "};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain_term___u227a___00__closed__16 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain_term___u227a___00__closed__16_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain_term___u227a___00__closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain_term___u227a___00__closed__16_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain_term___u227a___00__closed__17 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain_term___u227a___00__closed__17_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain_term___u227a___00__closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain_term___u227a___00__closed__18 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain_term___u227a___00__closed__18_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain_term___u227a___00__closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain_term___u227a___00__closed__18_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain_term___u227a___00__closed__19 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain_term___u227a___00__closed__19_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain_term___u227a___00__closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain_term___u227a___00__closed__19_value),((lean_object*)(((size_t)(51) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain_term___u227a___00__closed__20 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain_term___u227a___00__closed__20_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain_term___u227a___00__closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain_term___u227a___00__closed__15_value),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain_term___u227a___00__closed__17_value),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain_term___u227a___00__closed__20_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain_term___u227a___00__closed__21 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain_term___u227a___00__closed__21_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain_term___u227a___00__closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain_term___u227a___00__closed__13_value),((lean_object*)(((size_t)(50) << 1) | 1)),((lean_object*)(((size_t)(50) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain_term___u227a___00__closed__21_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain_term___u227a___00__closed__22 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain_term___u227a___00__closed__22_value;
LEAN_EXPORT const lean_object* lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain_term___u227a__ = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain_term___u227a___00__closed__22_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Basic______macroRules____private__Mathlib__Algebra__EuclideanDomain__Basic__0__EuclideanDomain__term___u227a____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Basic______macroRules____private__Mathlib__Algebra__EuclideanDomain__Basic__0__EuclideanDomain__term___u227a____1___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Basic______macroRules____private__Mathlib__Algebra__EuclideanDomain__Basic__0__EuclideanDomain__term___u227a____1___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Basic______macroRules____private__Mathlib__Algebra__EuclideanDomain__Basic__0__EuclideanDomain__term___u227a____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Basic______macroRules____private__Mathlib__Algebra__EuclideanDomain__Basic__0__EuclideanDomain__term___u227a____1___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Basic______macroRules____private__Mathlib__Algebra__EuclideanDomain__Basic__0__EuclideanDomain__term___u227a____1___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Basic______macroRules____private__Mathlib__Algebra__EuclideanDomain__Basic__0__EuclideanDomain__term___u227a____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Basic______macroRules____private__Mathlib__Algebra__EuclideanDomain__Basic__0__EuclideanDomain__term___u227a____1___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Basic______macroRules____private__Mathlib__Algebra__EuclideanDomain__Basic__0__EuclideanDomain__term___u227a____1___closed__2_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Basic______macroRules____private__Mathlib__Algebra__EuclideanDomain__Basic__0__EuclideanDomain__term___u227a____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Basic______macroRules____private__Mathlib__Algebra__EuclideanDomain__Basic__0__EuclideanDomain__term___u227a____1___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Basic______macroRules____private__Mathlib__Algebra__EuclideanDomain__Basic__0__EuclideanDomain__term___u227a____1___closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Basic______macroRules____private__Mathlib__Algebra__EuclideanDomain__Basic__0__EuclideanDomain__term___u227a____1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Basic______macroRules____private__Mathlib__Algebra__EuclideanDomain__Basic__0__EuclideanDomain__term___u227a____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Basic______macroRules____private__Mathlib__Algebra__EuclideanDomain__Basic__0__EuclideanDomain__term___u227a____1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Basic______macroRules____private__Mathlib__Algebra__EuclideanDomain__Basic__0__EuclideanDomain__term___u227a____1___closed__4_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Basic______macroRules____private__Mathlib__Algebra__EuclideanDomain__Basic__0__EuclideanDomain__term___u227a____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Basic______macroRules____private__Mathlib__Algebra__EuclideanDomain__Basic__0__EuclideanDomain__term___u227a____1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Basic______macroRules____private__Mathlib__Algebra__EuclideanDomain__Basic__0__EuclideanDomain__term___u227a____1___closed__4_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Basic______macroRules____private__Mathlib__Algebra__EuclideanDomain__Basic__0__EuclideanDomain__term___u227a____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Basic______macroRules____private__Mathlib__Algebra__EuclideanDomain__Basic__0__EuclideanDomain__term___u227a____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Basic______macroRules____private__Mathlib__Algebra__EuclideanDomain__Basic__0__EuclideanDomain__term___u227a____1___closed__4_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Basic______macroRules____private__Mathlib__Algebra__EuclideanDomain__Basic__0__EuclideanDomain__term___u227a____1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Basic______macroRules____private__Mathlib__Algebra__EuclideanDomain__Basic__0__EuclideanDomain__term___u227a____1___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Basic______macroRules____private__Mathlib__Algebra__EuclideanDomain__Basic__0__EuclideanDomain__term___u227a____1___closed__4_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Basic______macroRules____private__Mathlib__Algebra__EuclideanDomain__Basic__0__EuclideanDomain__term___u227a____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "EuclideanDomain.r"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Basic______macroRules____private__Mathlib__Algebra__EuclideanDomain__Basic__0__EuclideanDomain__term___u227a____1___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Basic______macroRules____private__Mathlib__Algebra__EuclideanDomain__Basic__0__EuclideanDomain__term___u227a____1___closed__5_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Basic______macroRules____private__Mathlib__Algebra__EuclideanDomain__Basic__0__EuclideanDomain__term___u227a____1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Basic______macroRules____private__Mathlib__Algebra__EuclideanDomain__Basic__0__EuclideanDomain__term___u227a____1___closed__6;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Basic______macroRules____private__Mathlib__Algebra__EuclideanDomain__Basic__0__EuclideanDomain__term___u227a____1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "r"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Basic______macroRules____private__Mathlib__Algebra__EuclideanDomain__Basic__0__EuclideanDomain__term___u227a____1___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Basic______macroRules____private__Mathlib__Algebra__EuclideanDomain__Basic__0__EuclideanDomain__term___u227a____1___closed__7_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Basic______macroRules____private__Mathlib__Algebra__EuclideanDomain__Basic__0__EuclideanDomain__term___u227a____1___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain_term___u227a___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(250, 176, 222, 37, 22, 220, 155, 159)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Basic______macroRules____private__Mathlib__Algebra__EuclideanDomain__Basic__0__EuclideanDomain__term___u227a____1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Basic______macroRules____private__Mathlib__Algebra__EuclideanDomain__Basic__0__EuclideanDomain__term___u227a____1___closed__8_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Basic______macroRules____private__Mathlib__Algebra__EuclideanDomain__Basic__0__EuclideanDomain__term___u227a____1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(244, 25, 245, 32, 120, 145, 151, 47)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Basic______macroRules____private__Mathlib__Algebra__EuclideanDomain__Basic__0__EuclideanDomain__term___u227a____1___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Basic______macroRules____private__Mathlib__Algebra__EuclideanDomain__Basic__0__EuclideanDomain__term___u227a____1___closed__8_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Basic______macroRules____private__Mathlib__Algebra__EuclideanDomain__Basic__0__EuclideanDomain__term___u227a____1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Basic______macroRules____private__Mathlib__Algebra__EuclideanDomain__Basic__0__EuclideanDomain__term___u227a____1___closed__8_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Basic______macroRules____private__Mathlib__Algebra__EuclideanDomain__Basic__0__EuclideanDomain__term___u227a____1___closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Basic______macroRules____private__Mathlib__Algebra__EuclideanDomain__Basic__0__EuclideanDomain__term___u227a____1___closed__9_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Basic______macroRules____private__Mathlib__Algebra__EuclideanDomain__Basic__0__EuclideanDomain__term___u227a____1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Basic______macroRules____private__Mathlib__Algebra__EuclideanDomain__Basic__0__EuclideanDomain__term___u227a____1___closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Basic______macroRules____private__Mathlib__Algebra__EuclideanDomain__Basic__0__EuclideanDomain__term___u227a____1___closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Basic______macroRules____private__Mathlib__Algebra__EuclideanDomain__Basic__0__EuclideanDomain__term___u227a____1___closed__10_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Basic______macroRules____private__Mathlib__Algebra__EuclideanDomain__Basic__0__EuclideanDomain__term___u227a____1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Basic______macroRules____private__Mathlib__Algebra__EuclideanDomain__Basic__0__EuclideanDomain__term___u227a____1___closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Basic______macroRules____private__Mathlib__Algebra__EuclideanDomain__Basic__0__EuclideanDomain__term___u227a____1___closed__11_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Basic______macroRules____private__Mathlib__Algebra__EuclideanDomain__Basic__0__EuclideanDomain__term___u227a____1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Basic______macroRules____private__Mathlib__Algebra__EuclideanDomain__Basic__0__EuclideanDomain__term___u227a____1___closed__11_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Basic______macroRules____private__Mathlib__Algebra__EuclideanDomain__Basic__0__EuclideanDomain__term___u227a____1___closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Basic______macroRules____private__Mathlib__Algebra__EuclideanDomain__Basic__0__EuclideanDomain__term___u227a____1___closed__12_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Basic______macroRules____private__Mathlib__Algebra__EuclideanDomain__Basic__0__EuclideanDomain__term___u227a____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Basic______macroRules____private__Mathlib__Algebra__EuclideanDomain__Basic__0__EuclideanDomain__term___u227a____1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Basic______unexpand__EuclideanDomain__r__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Basic______unexpand__EuclideanDomain__r__1___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Basic______unexpand__EuclideanDomain__r__1___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Basic______unexpand__EuclideanDomain__r__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Basic______unexpand__EuclideanDomain__r__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Basic______unexpand__EuclideanDomain__r__1___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Basic______unexpand__EuclideanDomain__r__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Basic______unexpand__EuclideanDomain__r__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Basic______unexpand__EuclideanDomain__r__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_euclideanDomain___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_euclideanDomain___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_euclideanDomain___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_RingEquiv_euclideanDomain___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_RingEquiv_euclideanDomain___redArg___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_RingEquiv_euclideanDomain___redArg___closed__0 = (const lean_object*)&lp_mathlib_RingEquiv_euclideanDomain___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_euclideanDomain___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_euclideanDomain(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Basic______macroRules____private__Mathlib__Algebra__EuclideanDomain__Basic__0__EuclideanDomain__term___u227a____1___closed__6(void){
_start:
{
lean_object* v___x_62_; lean_object* v___x_63_; 
v___x_62_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Basic______macroRules____private__Mathlib__Algebra__EuclideanDomain__Basic__0__EuclideanDomain__term___u227a____1___closed__5));
v___x_63_ = l_String_toRawSubstring_x27(v___x_62_);
return v___x_63_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Basic______macroRules____private__Mathlib__Algebra__EuclideanDomain__Basic__0__EuclideanDomain__term___u227a____1(lean_object* v_x_77_, lean_object* v_a_78_, lean_object* v_a_79_){
_start:
{
lean_object* v___x_80_; lean_object* v___x_81_; uint8_t v___x_82_; 
v___x_80_ = lean_unsigned_to_nat(0u);
v___x_81_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain_term___u227a___00__closed__13));
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
v___x_93_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Basic______macroRules____private__Mathlib__Algebra__EuclideanDomain__Basic__0__EuclideanDomain__term___u227a____1___closed__4));
v___x_94_ = lean_obj_once(&lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Basic______macroRules____private__Mathlib__Algebra__EuclideanDomain__Basic__0__EuclideanDomain__term___u227a____1___closed__6, &lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Basic______macroRules____private__Mathlib__Algebra__EuclideanDomain__Basic__0__EuclideanDomain__term___u227a____1___closed__6_once, _init_lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Basic______macroRules____private__Mathlib__Algebra__EuclideanDomain__Basic__0__EuclideanDomain__term___u227a____1___closed__6);
v___x_95_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Basic______macroRules____private__Mathlib__Algebra__EuclideanDomain__Basic__0__EuclideanDomain__term___u227a____1___closed__8));
lean_inc(v_currMacroScope_86_);
lean_inc(v_quotContext_85_);
v___x_96_ = l_Lean_addMacroScope(v_quotContext_85_, v___x_95_, v_currMacroScope_86_);
v___x_97_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Basic______macroRules____private__Mathlib__Algebra__EuclideanDomain__Basic__0__EuclideanDomain__term___u227a____1___closed__10));
lean_inc_n(v___x_92_, 2);
v___x_98_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_98_, 0, v___x_92_);
lean_ctor_set(v___x_98_, 1, v___x_94_);
lean_ctor_set(v___x_98_, 2, v___x_96_);
lean_ctor_set(v___x_98_, 3, v___x_97_);
v___x_99_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Basic______macroRules____private__Mathlib__Algebra__EuclideanDomain__Basic__0__EuclideanDomain__term___u227a____1___closed__12));
v___x_100_ = l_Lean_Syntax_node2(v___x_92_, v___x_99_, v___x_88_, v___x_90_);
v___x_101_ = l_Lean_Syntax_node2(v___x_92_, v___x_93_, v___x_98_, v___x_100_);
v___x_102_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_102_, 0, v___x_101_);
lean_ctor_set(v___x_102_, 1, v_a_79_);
return v___x_102_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Basic______macroRules____private__Mathlib__Algebra__EuclideanDomain__Basic__0__EuclideanDomain__term___u227a____1___boxed(lean_object* v_x_103_, lean_object* v_a_104_, lean_object* v_a_105_){
_start:
{
lean_object* v_res_106_; 
v_res_106_ = lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Basic______macroRules____private__Mathlib__Algebra__EuclideanDomain__Basic__0__EuclideanDomain__term___u227a____1(v_x_103_, v_a_104_, v_a_105_);
lean_dec_ref(v_a_104_);
return v_res_106_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Basic______unexpand__EuclideanDomain__r__1(lean_object* v_x_110_, lean_object* v_a_111_, lean_object* v_a_112_){
_start:
{
lean_object* v___x_113_; uint8_t v___x_114_; 
v___x_113_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Basic______macroRules____private__Mathlib__Algebra__EuclideanDomain__Basic__0__EuclideanDomain__term___u227a____1___closed__4));
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
v___x_119_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Basic______unexpand__EuclideanDomain__r__1___closed__1));
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
v___x_134_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain_term___u227a___00__closed__13));
v___x_135_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain_term___u227a___00__closed__16));
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
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Basic______unexpand__EuclideanDomain__r__1___boxed(lean_object* v_x_139_, lean_object* v_a_140_, lean_object* v_a_141_){
_start:
{
lean_object* v_res_142_; 
v_res_142_ = lp_mathlib___private_Mathlib_Algebra_EuclideanDomain_Basic_0__EuclideanDomain___aux__Mathlib__Algebra__EuclideanDomain__Basic______unexpand__EuclideanDomain__r__1(v_x_139_, v_a_140_, v_a_141_);
lean_dec(v_a_140_);
return v_res_142_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_euclideanDomain___redArg___lam__0(lean_object* v_f_143_, lean_object* v___y_144_){
_start:
{
lean_object* v_toFun_145_; lean_object* v___x_146_; 
v_toFun_145_ = lean_ctor_get(v_f_143_, 0);
lean_inc(v_toFun_145_);
lean_dec_ref(v_f_143_);
v___x_146_ = lean_apply_1(v_toFun_145_, v___y_144_);
return v___x_146_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_euclideanDomain___redArg___lam__2(lean_object* v_e_147_, lean_object* v_inst_148_, lean_object* v___f_149_, lean_object* v_a_150_, lean_object* v_b_151_){
_start:
{
lean_object* v___x_152_; lean_object* v_remainder_153_; lean_object* v_toFun_154_; lean_object* v___x_155_; lean_object* v___x_156_; lean_object* v___x_157_; lean_object* v___x_158_; 
lean_inc_ref_n(v_e_147_, 2);
v___x_152_ = lp_mathlib_Equiv_symm___redArg(v_e_147_);
v_remainder_153_ = lean_ctor_get(v_inst_148_, 2);
lean_inc(v_remainder_153_);
lean_dec_ref(v_inst_148_);
v_toFun_154_ = lean_ctor_get(v___x_152_, 0);
lean_inc(v_toFun_154_);
lean_dec_ref(v___x_152_);
lean_inc(v___f_149_);
v___x_155_ = lean_apply_2(v___f_149_, v_e_147_, v_a_150_);
v___x_156_ = lean_apply_2(v___f_149_, v_e_147_, v_b_151_);
v___x_157_ = lean_apply_2(v_remainder_153_, v___x_155_, v___x_156_);
v___x_158_ = lean_apply_1(v_toFun_154_, v___x_157_);
return v___x_158_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_euclideanDomain___redArg___lam__1(lean_object* v_e_159_, lean_object* v_inst_160_, lean_object* v___f_161_, lean_object* v_a_162_, lean_object* v_b_163_){
_start:
{
lean_object* v___x_164_; lean_object* v_quotient_165_; lean_object* v_toFun_166_; lean_object* v___x_167_; lean_object* v___x_168_; lean_object* v___x_169_; lean_object* v___x_170_; 
lean_inc_ref_n(v_e_159_, 2);
v___x_164_ = lp_mathlib_Equiv_symm___redArg(v_e_159_);
v_quotient_165_ = lean_ctor_get(v_inst_160_, 1);
lean_inc(v_quotient_165_);
lean_dec_ref(v_inst_160_);
v_toFun_166_ = lean_ctor_get(v___x_164_, 0);
lean_inc(v_toFun_166_);
lean_dec_ref(v___x_164_);
lean_inc(v___f_161_);
v___x_167_ = lean_apply_2(v___f_161_, v_e_159_, v_a_162_);
v___x_168_ = lean_apply_2(v___f_161_, v_e_159_, v_b_163_);
v___x_169_ = lean_apply_2(v_quotient_165_, v___x_167_, v___x_168_);
v___x_170_ = lean_apply_1(v_toFun_166_, v___x_169_);
return v___x_170_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_euclideanDomain___redArg(lean_object* v_inst_172_, lean_object* v_inst_173_, lean_object* v_e_174_){
_start:
{
lean_object* v___f_175_; lean_object* v___f_176_; lean_object* v___f_177_; lean_object* v___x_178_; 
v___f_175_ = ((lean_object*)(lp_mathlib_RingEquiv_euclideanDomain___redArg___closed__0));
lean_inc_ref(v_inst_172_);
lean_inc_ref(v_e_174_);
v___f_176_ = lean_alloc_closure((void*)(lp_mathlib_RingEquiv_euclideanDomain___redArg___lam__2), 5, 3);
lean_closure_set(v___f_176_, 0, v_e_174_);
lean_closure_set(v___f_176_, 1, v_inst_172_);
lean_closure_set(v___f_176_, 2, v___f_175_);
v___f_177_ = lean_alloc_closure((void*)(lp_mathlib_RingEquiv_euclideanDomain___redArg___lam__1), 5, 3);
lean_closure_set(v___f_177_, 0, v_e_174_);
lean_closure_set(v___f_177_, 1, v_inst_172_);
lean_closure_set(v___f_177_, 2, v___f_175_);
v___x_178_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_178_, 0, v_inst_173_);
lean_ctor_set(v___x_178_, 1, v___f_177_);
lean_ctor_set(v___x_178_, 2, v___f_176_);
return v___x_178_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_euclideanDomain(lean_object* v_R_179_, lean_object* v_S_180_, lean_object* v_inst_181_, lean_object* v_inst_182_, lean_object* v_e_183_){
_start:
{
lean_object* v___f_184_; lean_object* v___f_185_; lean_object* v___f_186_; lean_object* v___x_187_; 
v___f_184_ = ((lean_object*)(lp_mathlib_RingEquiv_euclideanDomain___redArg___closed__0));
lean_inc_ref(v_inst_181_);
lean_inc_ref(v_e_183_);
v___f_185_ = lean_alloc_closure((void*)(lp_mathlib_RingEquiv_euclideanDomain___redArg___lam__2), 5, 3);
lean_closure_set(v___f_185_, 0, v_e_183_);
lean_closure_set(v___f_185_, 1, v_inst_181_);
lean_closure_set(v___f_185_, 2, v___f_184_);
v___f_186_ = lean_alloc_closure((void*)(lp_mathlib_RingEquiv_euclideanDomain___redArg___lam__1), 5, 3);
lean_closure_set(v___f_186_, 0, v_e_183_);
lean_closure_set(v___f_186_, 1, v_inst_181_);
lean_closure_set(v___f_186_, 2, v___f_184_);
v___x_187_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_187_, 0, v_inst_182_);
lean_ctor_set(v___x_187_, 1, v___f_186_);
lean_ctor_set(v___x_187_, 2, v___f_185_);
return v___x_187_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_EuclideanDomain_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Divisibility_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Divisibility(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Equiv(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_EuclideanDomain_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_EuclideanDomain_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Divisibility_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Divisibility(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Equiv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_EuclideanDomain_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_EuclideanDomain_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Divisibility_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Divisibility(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Equiv(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_EuclideanDomain_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_EuclideanDomain_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Divisibility_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Divisibility(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Equiv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_EuclideanDomain_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_EuclideanDomain_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_EuclideanDomain_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
