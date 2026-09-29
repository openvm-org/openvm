// Lean compiler output
// Module: Mathlib.Order.Interval.Set.UnorderedInterval
// Imports: public import Init public meta import Init public import Mathlib.Data.Set.Order public import Mathlib.Order.Bounds.Basic public import Mathlib.Order.Interval.Set.Image public import Mathlib.Order.Interval.Set.LinearOrder public import Mathlib.Tactic.Common public import Mathlib.Order.MinMax public import Mathlib.Tactic.Attr.Core
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
lean_object* l_String_toRawSubstring_x27(lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Interval_term_x5b_x5b___x2c___x5d_x5d___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Interval"};
static const lean_object* lp_mathlib_Interval_term_x5b_x5b___x2c___x5d_x5d___closed__0 = (const lean_object*)&lp_mathlib_Interval_term_x5b_x5b___x2c___x5d_x5d___closed__0_value;
static const lean_string_object lp_mathlib_Interval_term_x5b_x5b___x2c___x5d_x5d___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "term[[_,_]]"};
static const lean_object* lp_mathlib_Interval_term_x5b_x5b___x2c___x5d_x5d___closed__1 = (const lean_object*)&lp_mathlib_Interval_term_x5b_x5b___x2c___x5d_x5d___closed__1_value;
static const lean_ctor_object lp_mathlib_Interval_term_x5b_x5b___x2c___x5d_x5d___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Interval_term_x5b_x5b___x2c___x5d_x5d___closed__0_value),LEAN_SCALAR_PTR_LITERAL(8, 35, 73, 97, 169, 159, 221, 98)}};
static const lean_ctor_object lp_mathlib_Interval_term_x5b_x5b___x2c___x5d_x5d___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Interval_term_x5b_x5b___x2c___x5d_x5d___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Interval_term_x5b_x5b___x2c___x5d_x5d___closed__1_value),LEAN_SCALAR_PTR_LITERAL(212, 21, 64, 116, 242, 115, 80, 213)}};
static const lean_object* lp_mathlib_Interval_term_x5b_x5b___x2c___x5d_x5d___closed__2 = (const lean_object*)&lp_mathlib_Interval_term_x5b_x5b___x2c___x5d_x5d___closed__2_value;
static const lean_string_object lp_mathlib_Interval_term_x5b_x5b___x2c___x5d_x5d___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Interval_term_x5b_x5b___x2c___x5d_x5d___closed__3 = (const lean_object*)&lp_mathlib_Interval_term_x5b_x5b___x2c___x5d_x5d___closed__3_value;
static const lean_ctor_object lp_mathlib_Interval_term_x5b_x5b___x2c___x5d_x5d___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Interval_term_x5b_x5b___x2c___x5d_x5d___closed__3_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Interval_term_x5b_x5b___x2c___x5d_x5d___closed__4 = (const lean_object*)&lp_mathlib_Interval_term_x5b_x5b___x2c___x5d_x5d___closed__4_value;
static const lean_string_object lp_mathlib_Interval_term_x5b_x5b___x2c___x5d_x5d___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "[["};
static const lean_object* lp_mathlib_Interval_term_x5b_x5b___x2c___x5d_x5d___closed__5 = (const lean_object*)&lp_mathlib_Interval_term_x5b_x5b___x2c___x5d_x5d___closed__5_value;
static const lean_ctor_object lp_mathlib_Interval_term_x5b_x5b___x2c___x5d_x5d___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Interval_term_x5b_x5b___x2c___x5d_x5d___closed__5_value)}};
static const lean_object* lp_mathlib_Interval_term_x5b_x5b___x2c___x5d_x5d___closed__6 = (const lean_object*)&lp_mathlib_Interval_term_x5b_x5b___x2c___x5d_x5d___closed__6_value;
static const lean_string_object lp_mathlib_Interval_term_x5b_x5b___x2c___x5d_x5d___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_Interval_term_x5b_x5b___x2c___x5d_x5d___closed__7 = (const lean_object*)&lp_mathlib_Interval_term_x5b_x5b___x2c___x5d_x5d___closed__7_value;
static const lean_ctor_object lp_mathlib_Interval_term_x5b_x5b___x2c___x5d_x5d___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Interval_term_x5b_x5b___x2c___x5d_x5d___closed__7_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_Interval_term_x5b_x5b___x2c___x5d_x5d___closed__8 = (const lean_object*)&lp_mathlib_Interval_term_x5b_x5b___x2c___x5d_x5d___closed__8_value;
static const lean_ctor_object lp_mathlib_Interval_term_x5b_x5b___x2c___x5d_x5d___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_Interval_term_x5b_x5b___x2c___x5d_x5d___closed__8_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Interval_term_x5b_x5b___x2c___x5d_x5d___closed__9 = (const lean_object*)&lp_mathlib_Interval_term_x5b_x5b___x2c___x5d_x5d___closed__9_value;
static const lean_ctor_object lp_mathlib_Interval_term_x5b_x5b___x2c___x5d_x5d___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Interval_term_x5b_x5b___x2c___x5d_x5d___closed__4_value),((lean_object*)&lp_mathlib_Interval_term_x5b_x5b___x2c___x5d_x5d___closed__6_value),((lean_object*)&lp_mathlib_Interval_term_x5b_x5b___x2c___x5d_x5d___closed__9_value)}};
static const lean_object* lp_mathlib_Interval_term_x5b_x5b___x2c___x5d_x5d___closed__10 = (const lean_object*)&lp_mathlib_Interval_term_x5b_x5b___x2c___x5d_x5d___closed__10_value;
static const lean_string_object lp_mathlib_Interval_term_x5b_x5b___x2c___x5d_x5d___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ", "};
static const lean_object* lp_mathlib_Interval_term_x5b_x5b___x2c___x5d_x5d___closed__11 = (const lean_object*)&lp_mathlib_Interval_term_x5b_x5b___x2c___x5d_x5d___closed__11_value;
static const lean_ctor_object lp_mathlib_Interval_term_x5b_x5b___x2c___x5d_x5d___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Interval_term_x5b_x5b___x2c___x5d_x5d___closed__11_value)}};
static const lean_object* lp_mathlib_Interval_term_x5b_x5b___x2c___x5d_x5d___closed__12 = (const lean_object*)&lp_mathlib_Interval_term_x5b_x5b___x2c___x5d_x5d___closed__12_value;
static const lean_ctor_object lp_mathlib_Interval_term_x5b_x5b___x2c___x5d_x5d___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Interval_term_x5b_x5b___x2c___x5d_x5d___closed__4_value),((lean_object*)&lp_mathlib_Interval_term_x5b_x5b___x2c___x5d_x5d___closed__10_value),((lean_object*)&lp_mathlib_Interval_term_x5b_x5b___x2c___x5d_x5d___closed__12_value)}};
static const lean_object* lp_mathlib_Interval_term_x5b_x5b___x2c___x5d_x5d___closed__13 = (const lean_object*)&lp_mathlib_Interval_term_x5b_x5b___x2c___x5d_x5d___closed__13_value;
static const lean_ctor_object lp_mathlib_Interval_term_x5b_x5b___x2c___x5d_x5d___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Interval_term_x5b_x5b___x2c___x5d_x5d___closed__4_value),((lean_object*)&lp_mathlib_Interval_term_x5b_x5b___x2c___x5d_x5d___closed__13_value),((lean_object*)&lp_mathlib_Interval_term_x5b_x5b___x2c___x5d_x5d___closed__9_value)}};
static const lean_object* lp_mathlib_Interval_term_x5b_x5b___x2c___x5d_x5d___closed__14 = (const lean_object*)&lp_mathlib_Interval_term_x5b_x5b___x2c___x5d_x5d___closed__14_value;
static const lean_string_object lp_mathlib_Interval_term_x5b_x5b___x2c___x5d_x5d___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "]]"};
static const lean_object* lp_mathlib_Interval_term_x5b_x5b___x2c___x5d_x5d___closed__15 = (const lean_object*)&lp_mathlib_Interval_term_x5b_x5b___x2c___x5d_x5d___closed__15_value;
static const lean_ctor_object lp_mathlib_Interval_term_x5b_x5b___x2c___x5d_x5d___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Interval_term_x5b_x5b___x2c___x5d_x5d___closed__15_value)}};
static const lean_object* lp_mathlib_Interval_term_x5b_x5b___x2c___x5d_x5d___closed__16 = (const lean_object*)&lp_mathlib_Interval_term_x5b_x5b___x2c___x5d_x5d___closed__16_value;
static const lean_ctor_object lp_mathlib_Interval_term_x5b_x5b___x2c___x5d_x5d___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Interval_term_x5b_x5b___x2c___x5d_x5d___closed__4_value),((lean_object*)&lp_mathlib_Interval_term_x5b_x5b___x2c___x5d_x5d___closed__14_value),((lean_object*)&lp_mathlib_Interval_term_x5b_x5b___x2c___x5d_x5d___closed__16_value)}};
static const lean_object* lp_mathlib_Interval_term_x5b_x5b___x2c___x5d_x5d___closed__17 = (const lean_object*)&lp_mathlib_Interval_term_x5b_x5b___x2c___x5d_x5d___closed__17_value;
static const lean_ctor_object lp_mathlib_Interval_term_x5b_x5b___x2c___x5d_x5d___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Interval_term_x5b_x5b___x2c___x5d_x5d___closed__2_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_Interval_term_x5b_x5b___x2c___x5d_x5d___closed__17_value)}};
static const lean_object* lp_mathlib_Interval_term_x5b_x5b___x2c___x5d_x5d___closed__18 = (const lean_object*)&lp_mathlib_Interval_term_x5b_x5b___x2c___x5d_x5d___closed__18_value;
LEAN_EXPORT const lean_object* lp_mathlib_Interval_term_x5b_x5b___x2c___x5d_x5d = (const lean_object*)&lp_mathlib_Interval_term_x5b_x5b___x2c___x5d_x5d___closed__18_value;
static const lean_string_object lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_x5b_x5b___x2c___x5d_x5d__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_x5b_x5b___x2c___x5d_x5d__1___closed__0 = (const lean_object*)&lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_x5b_x5b___x2c___x5d_x5d__1___closed__0_value;
static const lean_string_object lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_x5b_x5b___x2c___x5d_x5d__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_x5b_x5b___x2c___x5d_x5d__1___closed__1 = (const lean_object*)&lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_x5b_x5b___x2c___x5d_x5d__1___closed__1_value;
static const lean_string_object lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_x5b_x5b___x2c___x5d_x5d__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_x5b_x5b___x2c___x5d_x5d__1___closed__2 = (const lean_object*)&lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_x5b_x5b___x2c___x5d_x5d__1___closed__2_value;
static const lean_string_object lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_x5b_x5b___x2c___x5d_x5d__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_x5b_x5b___x2c___x5d_x5d__1___closed__3 = (const lean_object*)&lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_x5b_x5b___x2c___x5d_x5d__1___closed__3_value;
static const lean_ctor_object lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_x5b_x5b___x2c___x5d_x5d__1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_x5b_x5b___x2c___x5d_x5d__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_x5b_x5b___x2c___x5d_x5d__1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_x5b_x5b___x2c___x5d_x5d__1___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_x5b_x5b___x2c___x5d_x5d__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_x5b_x5b___x2c___x5d_x5d__1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_x5b_x5b___x2c___x5d_x5d__1___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_x5b_x5b___x2c___x5d_x5d__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_x5b_x5b___x2c___x5d_x5d__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_x5b_x5b___x2c___x5d_x5d__1___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_x5b_x5b___x2c___x5d_x5d__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_x5b_x5b___x2c___x5d_x5d__1___closed__4 = (const lean_object*)&lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_x5b_x5b___x2c___x5d_x5d__1___closed__4_value;
static const lean_string_object lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_x5b_x5b___x2c___x5d_x5d__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Set.uIcc"};
static const lean_object* lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_x5b_x5b___x2c___x5d_x5d__1___closed__5 = (const lean_object*)&lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_x5b_x5b___x2c___x5d_x5d__1___closed__5_value;
static lean_once_cell_t lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_x5b_x5b___x2c___x5d_x5d__1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_x5b_x5b___x2c___x5d_x5d__1___closed__6;
static const lean_string_object lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_x5b_x5b___x2c___x5d_x5d__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Set"};
static const lean_object* lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_x5b_x5b___x2c___x5d_x5d__1___closed__7 = (const lean_object*)&lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_x5b_x5b___x2c___x5d_x5d__1___closed__7_value;
static const lean_string_object lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_x5b_x5b___x2c___x5d_x5d__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "uIcc"};
static const lean_object* lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_x5b_x5b___x2c___x5d_x5d__1___closed__8 = (const lean_object*)&lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_x5b_x5b___x2c___x5d_x5d__1___closed__8_value;
static const lean_ctor_object lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_x5b_x5b___x2c___x5d_x5d__1___closed__9_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_x5b_x5b___x2c___x5d_x5d__1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(70, 214, 213, 227, 101, 196, 147, 255)}};
static const lean_ctor_object lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_x5b_x5b___x2c___x5d_x5d__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_x5b_x5b___x2c___x5d_x5d__1___closed__9_value_aux_0),((lean_object*)&lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_x5b_x5b___x2c___x5d_x5d__1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(105, 30, 87, 37, 133, 206, 30, 232)}};
static const lean_object* lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_x5b_x5b___x2c___x5d_x5d__1___closed__9 = (const lean_object*)&lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_x5b_x5b___x2c___x5d_x5d__1___closed__9_value;
static const lean_ctor_object lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_x5b_x5b___x2c___x5d_x5d__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_x5b_x5b___x2c___x5d_x5d__1___closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_x5b_x5b___x2c___x5d_x5d__1___closed__10 = (const lean_object*)&lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_x5b_x5b___x2c___x5d_x5d__1___closed__10_value;
static const lean_ctor_object lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_x5b_x5b___x2c___x5d_x5d__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_x5b_x5b___x2c___x5d_x5d__1___closed__10_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_x5b_x5b___x2c___x5d_x5d__1___closed__11 = (const lean_object*)&lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_x5b_x5b___x2c___x5d_x5d__1___closed__11_value;
static const lean_string_object lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_x5b_x5b___x2c___x5d_x5d__1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_x5b_x5b___x2c___x5d_x5d__1___closed__12 = (const lean_object*)&lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_x5b_x5b___x2c___x5d_x5d__1___closed__12_value;
static const lean_ctor_object lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_x5b_x5b___x2c___x5d_x5d__1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_x5b_x5b___x2c___x5d_x5d__1___closed__12_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_x5b_x5b___x2c___x5d_x5d__1___closed__13 = (const lean_object*)&lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_x5b_x5b___x2c___x5d_x5d__1___closed__13_value;
LEAN_EXPORT lean_object* lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_x5b_x5b___x2c___x5d_x5d__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_x5b_x5b___x2c___x5d_x5d__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______unexpand__Set__uIcc__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______unexpand__Set__uIcc__1___closed__0 = (const lean_object*)&lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______unexpand__Set__uIcc__1___closed__0_value;
static const lean_ctor_object lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______unexpand__Set__uIcc__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______unexpand__Set__uIcc__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______unexpand__Set__uIcc__1___closed__1 = (const lean_object*)&lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______unexpand__Set__uIcc__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______unexpand__Set__uIcc__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______unexpand__Set__uIcc__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Interval_term_u0399___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 5, .m_data = "termΙ"};
static const lean_object* lp_mathlib_Interval_term_u0399___closed__0 = (const lean_object*)&lp_mathlib_Interval_term_u0399___closed__0_value;
static const lean_ctor_object lp_mathlib_Interval_term_u0399___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Interval_term_x5b_x5b___x2c___x5d_x5d___closed__0_value),LEAN_SCALAR_PTR_LITERAL(8, 35, 73, 97, 169, 159, 221, 98)}};
static const lean_ctor_object lp_mathlib_Interval_term_u0399___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Interval_term_u0399___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Interval_term_u0399___closed__0_value),LEAN_SCALAR_PTR_LITERAL(169, 150, 205, 239, 137, 129, 197, 82)}};
static const lean_object* lp_mathlib_Interval_term_u0399___closed__1 = (const lean_object*)&lp_mathlib_Interval_term_u0399___closed__1_value;
static const lean_string_object lp_mathlib_Interval_term_u0399___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 1, .m_data = "Ι"};
static const lean_object* lp_mathlib_Interval_term_u0399___closed__2 = (const lean_object*)&lp_mathlib_Interval_term_u0399___closed__2_value;
static const lean_ctor_object lp_mathlib_Interval_term_u0399___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Interval_term_u0399___closed__2_value)}};
static const lean_object* lp_mathlib_Interval_term_u0399___closed__3 = (const lean_object*)&lp_mathlib_Interval_term_u0399___closed__3_value;
static const lean_ctor_object lp_mathlib_Interval_term_u0399___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Interval_term_u0399___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_Interval_term_u0399___closed__3_value)}};
static const lean_object* lp_mathlib_Interval_term_u0399___closed__4 = (const lean_object*)&lp_mathlib_Interval_term_u0399___closed__4_value;
LEAN_EXPORT const lean_object* lp_mathlib_Interval_term_u0399 = (const lean_object*)&lp_mathlib_Interval_term_u0399___closed__4_value;
static const lean_string_object lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_u0399__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Set.uIoc"};
static const lean_object* lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_u0399__1___closed__0 = (const lean_object*)&lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_u0399__1___closed__0_value;
static lean_once_cell_t lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_u0399__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_u0399__1___closed__1;
static const lean_string_object lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_u0399__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "uIoc"};
static const lean_object* lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_u0399__1___closed__2 = (const lean_object*)&lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_u0399__1___closed__2_value;
static const lean_ctor_object lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_u0399__1___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_x5b_x5b___x2c___x5d_x5d__1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(70, 214, 213, 227, 101, 196, 147, 255)}};
static const lean_ctor_object lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_u0399__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_u0399__1___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_u0399__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(141, 120, 213, 58, 12, 11, 20, 53)}};
static const lean_object* lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_u0399__1___closed__3 = (const lean_object*)&lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_u0399__1___closed__3_value;
static const lean_ctor_object lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_u0399__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_u0399__1___closed__3_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_u0399__1___closed__4 = (const lean_object*)&lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_u0399__1___closed__4_value;
static const lean_ctor_object lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_u0399__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_u0399__1___closed__4_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_u0399__1___closed__5 = (const lean_object*)&lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_u0399__1___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_u0399__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_u0399__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______unexpand__Set__uIoc__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______unexpand__Set__uIoc__1___boxed(lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_x5b_x5b___x2c___x5d_x5d__1___closed__6(void){
_start:
{
lean_object* v___x_55_; lean_object* v___x_56_; 
v___x_55_ = ((lean_object*)(lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_x5b_x5b___x2c___x5d_x5d__1___closed__5));
v___x_56_ = l_String_toRawSubstring_x27(v___x_55_);
return v___x_56_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_x5b_x5b___x2c___x5d_x5d__1(lean_object* v_x_71_, lean_object* v_a_72_, lean_object* v_a_73_){
_start:
{
lean_object* v___x_74_; uint8_t v___x_75_; 
v___x_74_ = ((lean_object*)(lp_mathlib_Interval_term_x5b_x5b___x2c___x5d_x5d___closed__2));
lean_inc(v_x_71_);
v___x_75_ = l_Lean_Syntax_isOfKind(v_x_71_, v___x_74_);
if (v___x_75_ == 0)
{
lean_object* v___x_76_; lean_object* v___x_77_; 
lean_dec(v_x_71_);
v___x_76_ = lean_box(1);
v___x_77_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_77_, 0, v___x_76_);
lean_ctor_set(v___x_77_, 1, v_a_73_);
return v___x_77_;
}
else
{
lean_object* v_quotContext_78_; lean_object* v_currMacroScope_79_; lean_object* v_ref_80_; lean_object* v___x_81_; lean_object* v___x_82_; lean_object* v___x_83_; lean_object* v___x_84_; uint8_t v___x_85_; lean_object* v___x_86_; lean_object* v___x_87_; lean_object* v___x_88_; lean_object* v___x_89_; lean_object* v___x_90_; lean_object* v___x_91_; lean_object* v___x_92_; lean_object* v___x_93_; lean_object* v___x_94_; lean_object* v___x_95_; lean_object* v___x_96_; 
v_quotContext_78_ = lean_ctor_get(v_a_72_, 1);
v_currMacroScope_79_ = lean_ctor_get(v_a_72_, 2);
v_ref_80_ = lean_ctor_get(v_a_72_, 5);
v___x_81_ = lean_unsigned_to_nat(1u);
v___x_82_ = l_Lean_Syntax_getArg(v_x_71_, v___x_81_);
v___x_83_ = lean_unsigned_to_nat(3u);
v___x_84_ = l_Lean_Syntax_getArg(v_x_71_, v___x_83_);
lean_dec(v_x_71_);
v___x_85_ = 0;
v___x_86_ = l_Lean_SourceInfo_fromRef(v_ref_80_, v___x_85_);
v___x_87_ = ((lean_object*)(lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_x5b_x5b___x2c___x5d_x5d__1___closed__4));
v___x_88_ = lean_obj_once(&lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_x5b_x5b___x2c___x5d_x5d__1___closed__6, &lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_x5b_x5b___x2c___x5d_x5d__1___closed__6_once, _init_lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_x5b_x5b___x2c___x5d_x5d__1___closed__6);
v___x_89_ = ((lean_object*)(lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_x5b_x5b___x2c___x5d_x5d__1___closed__9));
lean_inc(v_currMacroScope_79_);
lean_inc(v_quotContext_78_);
v___x_90_ = l_Lean_addMacroScope(v_quotContext_78_, v___x_89_, v_currMacroScope_79_);
v___x_91_ = ((lean_object*)(lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_x5b_x5b___x2c___x5d_x5d__1___closed__11));
lean_inc_n(v___x_86_, 2);
v___x_92_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_92_, 0, v___x_86_);
lean_ctor_set(v___x_92_, 1, v___x_88_);
lean_ctor_set(v___x_92_, 2, v___x_90_);
lean_ctor_set(v___x_92_, 3, v___x_91_);
v___x_93_ = ((lean_object*)(lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_x5b_x5b___x2c___x5d_x5d__1___closed__13));
v___x_94_ = l_Lean_Syntax_node2(v___x_86_, v___x_93_, v___x_82_, v___x_84_);
v___x_95_ = l_Lean_Syntax_node2(v___x_86_, v___x_87_, v___x_92_, v___x_94_);
v___x_96_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_96_, 0, v___x_95_);
lean_ctor_set(v___x_96_, 1, v_a_73_);
return v___x_96_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_x5b_x5b___x2c___x5d_x5d__1___boxed(lean_object* v_x_97_, lean_object* v_a_98_, lean_object* v_a_99_){
_start:
{
lean_object* v_res_100_; 
v_res_100_ = lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_x5b_x5b___x2c___x5d_x5d__1(v_x_97_, v_a_98_, v_a_99_);
lean_dec_ref(v_a_98_);
return v_res_100_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______unexpand__Set__uIcc__1(lean_object* v_x_104_, lean_object* v_a_105_, lean_object* v_a_106_){
_start:
{
lean_object* v___x_107_; uint8_t v___x_108_; 
v___x_107_ = ((lean_object*)(lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_x5b_x5b___x2c___x5d_x5d__1___closed__4));
lean_inc(v_x_104_);
v___x_108_ = l_Lean_Syntax_isOfKind(v_x_104_, v___x_107_);
if (v___x_108_ == 0)
{
lean_object* v___x_109_; lean_object* v___x_110_; 
lean_dec(v_x_104_);
v___x_109_ = lean_box(0);
v___x_110_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_110_, 0, v___x_109_);
lean_ctor_set(v___x_110_, 1, v_a_106_);
return v___x_110_;
}
else
{
lean_object* v___x_111_; lean_object* v___x_112_; lean_object* v___x_113_; uint8_t v___x_114_; 
v___x_111_ = lean_unsigned_to_nat(0u);
v___x_112_ = l_Lean_Syntax_getArg(v_x_104_, v___x_111_);
v___x_113_ = ((lean_object*)(lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______unexpand__Set__uIcc__1___closed__1));
lean_inc(v___x_112_);
v___x_114_ = l_Lean_Syntax_isOfKind(v___x_112_, v___x_113_);
if (v___x_114_ == 0)
{
lean_object* v___x_115_; lean_object* v___x_116_; 
lean_dec(v___x_112_);
lean_dec(v_x_104_);
v___x_115_ = lean_box(0);
v___x_116_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_116_, 0, v___x_115_);
lean_ctor_set(v___x_116_, 1, v_a_106_);
return v___x_116_;
}
else
{
lean_object* v___x_117_; lean_object* v___x_118_; lean_object* v___x_119_; uint8_t v___x_120_; 
v___x_117_ = lean_unsigned_to_nat(1u);
v___x_118_ = l_Lean_Syntax_getArg(v_x_104_, v___x_117_);
lean_dec(v_x_104_);
v___x_119_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_118_);
v___x_120_ = l_Lean_Syntax_matchesNull(v___x_118_, v___x_119_);
if (v___x_120_ == 0)
{
lean_object* v___x_121_; lean_object* v___x_122_; 
lean_dec(v___x_118_);
lean_dec(v___x_112_);
v___x_121_ = lean_box(0);
v___x_122_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_122_, 0, v___x_121_);
lean_ctor_set(v___x_122_, 1, v_a_106_);
return v___x_122_;
}
else
{
lean_object* v___x_123_; lean_object* v___x_124_; lean_object* v_ref_125_; uint8_t v___x_126_; lean_object* v___x_127_; lean_object* v___x_128_; lean_object* v___x_129_; lean_object* v___x_130_; lean_object* v___x_131_; lean_object* v___x_132_; lean_object* v___x_133_; lean_object* v___x_134_; lean_object* v___x_135_; lean_object* v___x_136_; 
v___x_123_ = l_Lean_Syntax_getArg(v___x_118_, v___x_111_);
v___x_124_ = l_Lean_Syntax_getArg(v___x_118_, v___x_117_);
lean_dec(v___x_118_);
v_ref_125_ = l_Lean_replaceRef(v___x_112_, v_a_105_);
lean_dec(v___x_112_);
v___x_126_ = 0;
v___x_127_ = l_Lean_SourceInfo_fromRef(v_ref_125_, v___x_126_);
lean_dec(v_ref_125_);
v___x_128_ = ((lean_object*)(lp_mathlib_Interval_term_x5b_x5b___x2c___x5d_x5d___closed__2));
v___x_129_ = ((lean_object*)(lp_mathlib_Interval_term_x5b_x5b___x2c___x5d_x5d___closed__5));
lean_inc_n(v___x_127_, 3);
v___x_130_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_130_, 0, v___x_127_);
lean_ctor_set(v___x_130_, 1, v___x_129_);
v___x_131_ = ((lean_object*)(lp_mathlib_Interval_term_x5b_x5b___x2c___x5d_x5d___closed__11));
v___x_132_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_132_, 0, v___x_127_);
lean_ctor_set(v___x_132_, 1, v___x_131_);
v___x_133_ = ((lean_object*)(lp_mathlib_Interval_term_x5b_x5b___x2c___x5d_x5d___closed__15));
v___x_134_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_134_, 0, v___x_127_);
lean_ctor_set(v___x_134_, 1, v___x_133_);
v___x_135_ = l_Lean_Syntax_node5(v___x_127_, v___x_128_, v___x_130_, v___x_123_, v___x_132_, v___x_124_, v___x_134_);
v___x_136_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_136_, 0, v___x_135_);
lean_ctor_set(v___x_136_, 1, v_a_106_);
return v___x_136_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______unexpand__Set__uIcc__1___boxed(lean_object* v_x_137_, lean_object* v_a_138_, lean_object* v_a_139_){
_start:
{
lean_object* v_res_140_; 
v_res_140_ = lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______unexpand__Set__uIcc__1(v_x_137_, v_a_138_, v_a_139_);
lean_dec(v_a_138_);
return v_res_140_;
}
}
static lean_object* _init_lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_u0399__1___closed__1(void){
_start:
{
lean_object* v___x_154_; lean_object* v___x_155_; 
v___x_154_ = ((lean_object*)(lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_u0399__1___closed__0));
v___x_155_ = l_String_toRawSubstring_x27(v___x_154_);
return v___x_155_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_u0399__1(lean_object* v_x_166_, lean_object* v_a_167_, lean_object* v_a_168_){
_start:
{
lean_object* v___x_169_; uint8_t v___x_170_; 
v___x_169_ = ((lean_object*)(lp_mathlib_Interval_term_u0399___closed__1));
v___x_170_ = l_Lean_Syntax_isOfKind(v_x_166_, v___x_169_);
if (v___x_170_ == 0)
{
lean_object* v___x_171_; lean_object* v___x_172_; 
v___x_171_ = lean_box(1);
v___x_172_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_172_, 0, v___x_171_);
lean_ctor_set(v___x_172_, 1, v_a_168_);
return v___x_172_;
}
else
{
lean_object* v_quotContext_173_; lean_object* v_currMacroScope_174_; lean_object* v_ref_175_; uint8_t v___x_176_; lean_object* v___x_177_; lean_object* v___x_178_; lean_object* v___x_179_; lean_object* v___x_180_; lean_object* v___x_181_; lean_object* v___x_182_; lean_object* v___x_183_; 
v_quotContext_173_ = lean_ctor_get(v_a_167_, 1);
v_currMacroScope_174_ = lean_ctor_get(v_a_167_, 2);
v_ref_175_ = lean_ctor_get(v_a_167_, 5);
v___x_176_ = 0;
v___x_177_ = l_Lean_SourceInfo_fromRef(v_ref_175_, v___x_176_);
v___x_178_ = lean_obj_once(&lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_u0399__1___closed__1, &lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_u0399__1___closed__1_once, _init_lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_u0399__1___closed__1);
v___x_179_ = ((lean_object*)(lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_u0399__1___closed__3));
lean_inc(v_currMacroScope_174_);
lean_inc(v_quotContext_173_);
v___x_180_ = l_Lean_addMacroScope(v_quotContext_173_, v___x_179_, v_currMacroScope_174_);
v___x_181_ = ((lean_object*)(lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_u0399__1___closed__5));
v___x_182_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_182_, 0, v___x_177_);
lean_ctor_set(v___x_182_, 1, v___x_178_);
lean_ctor_set(v___x_182_, 2, v___x_180_);
lean_ctor_set(v___x_182_, 3, v___x_181_);
v___x_183_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_183_, 0, v___x_182_);
lean_ctor_set(v___x_183_, 1, v_a_168_);
return v___x_183_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_u0399__1___boxed(lean_object* v_x_184_, lean_object* v_a_185_, lean_object* v_a_186_){
_start:
{
lean_object* v_res_187_; 
v_res_187_ = lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______macroRules__Interval__term_u0399__1(v_x_184_, v_a_185_, v_a_186_);
lean_dec_ref(v_a_185_);
return v_res_187_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______unexpand__Set__uIoc__1(lean_object* v_x_188_, lean_object* v_a_189_, lean_object* v_a_190_){
_start:
{
lean_object* v___x_191_; uint8_t v___x_192_; 
v___x_191_ = ((lean_object*)(lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______unexpand__Set__uIcc__1___closed__1));
lean_inc(v_x_188_);
v___x_192_ = l_Lean_Syntax_isOfKind(v_x_188_, v___x_191_);
if (v___x_192_ == 0)
{
lean_object* v___x_193_; lean_object* v___x_194_; 
lean_dec(v_x_188_);
v___x_193_ = lean_box(0);
v___x_194_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_194_, 0, v___x_193_);
lean_ctor_set(v___x_194_, 1, v_a_190_);
return v___x_194_;
}
else
{
lean_object* v_ref_195_; uint8_t v___x_196_; lean_object* v___x_197_; lean_object* v___x_198_; lean_object* v___x_199_; lean_object* v___x_200_; lean_object* v___x_201_; lean_object* v___x_202_; 
v_ref_195_ = l_Lean_replaceRef(v_x_188_, v_a_189_);
lean_dec(v_x_188_);
v___x_196_ = 0;
v___x_197_ = l_Lean_SourceInfo_fromRef(v_ref_195_, v___x_196_);
lean_dec(v_ref_195_);
v___x_198_ = ((lean_object*)(lp_mathlib_Interval_term_u0399___closed__1));
v___x_199_ = ((lean_object*)(lp_mathlib_Interval_term_u0399___closed__2));
lean_inc(v___x_197_);
v___x_200_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_200_, 0, v___x_197_);
lean_ctor_set(v___x_200_, 1, v___x_199_);
v___x_201_ = l_Lean_Syntax_node1(v___x_197_, v___x_198_, v___x_200_);
v___x_202_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_202_, 0, v___x_201_);
lean_ctor_set(v___x_202_, 1, v_a_190_);
return v___x_202_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______unexpand__Set__uIoc__1___boxed(lean_object* v_x_203_, lean_object* v_a_204_, lean_object* v_a_205_){
_start:
{
lean_object* v_res_206_; 
v_res_206_ = lp_mathlib_Interval___aux__Mathlib__Order__Interval__Set__UnorderedInterval______unexpand__Set__uIoc__1(v_x_203_, v_a_204_, v_a_205_);
lean_dec(v_a_204_);
return v_res_206_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Set_Order(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Bounds_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Interval_Set_Image(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Interval_Set_LinearOrder(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Common(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_MinMax(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Attr_Core(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Order_Interval_Set_UnorderedInterval(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Set_Order(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Bounds_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Interval_Set_Image(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Interval_Set_LinearOrder(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Common(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_MinMax(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Attr_Core(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Order_Interval_Set_UnorderedInterval(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Set_Order(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Bounds_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Interval_Set_Image(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Interval_Set_LinearOrder(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Common(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_MinMax(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Attr_Core(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_Interval_Set_UnorderedInterval(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_Order(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Bounds_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Interval_Set_Image(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Interval_Set_LinearOrder(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Common(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_MinMax(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Attr_Core(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Interval_Set_UnorderedInterval(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Order_Interval_Set_UnorderedInterval(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Order_Interval_Set_UnorderedInterval(builtin);
}
#ifdef __cplusplus
}
#endif
