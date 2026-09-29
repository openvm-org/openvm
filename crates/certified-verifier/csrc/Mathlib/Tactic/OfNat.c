// Lean compiler output
// Module: Mathlib.Tactic.OfNat
// Imports: public import Init public meta import Init public import Mathlib.Init
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
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_termOfNat_x28___x29___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "termOfNat(_)"};
static const lean_object* lp_mathlib_termOfNat_x28___x29___closed__0 = (const lean_object*)&lp_mathlib_termOfNat_x28___x29___closed__0_value;
static const lean_ctor_object lp_mathlib_termOfNat_x28___x29___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_termOfNat_x28___x29___closed__0_value),LEAN_SCALAR_PTR_LITERAL(104, 57, 51, 26, 7, 137, 184, 235)}};
static const lean_object* lp_mathlib_termOfNat_x28___x29___closed__1 = (const lean_object*)&lp_mathlib_termOfNat_x28___x29___closed__1_value;
static const lean_string_object lp_mathlib_termOfNat_x28___x29___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_termOfNat_x28___x29___closed__2 = (const lean_object*)&lp_mathlib_termOfNat_x28___x29___closed__2_value;
static const lean_ctor_object lp_mathlib_termOfNat_x28___x29___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_termOfNat_x28___x29___closed__2_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_termOfNat_x28___x29___closed__3 = (const lean_object*)&lp_mathlib_termOfNat_x28___x29___closed__3_value;
static const lean_string_object lp_mathlib_termOfNat_x28___x29___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "ofNat("};
static const lean_object* lp_mathlib_termOfNat_x28___x29___closed__4 = (const lean_object*)&lp_mathlib_termOfNat_x28___x29___closed__4_value;
static const lean_ctor_object lp_mathlib_termOfNat_x28___x29___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_termOfNat_x28___x29___closed__4_value)}};
static const lean_object* lp_mathlib_termOfNat_x28___x29___closed__5 = (const lean_object*)&lp_mathlib_termOfNat_x28___x29___closed__5_value;
static const lean_string_object lp_mathlib_termOfNat_x28___x29___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_termOfNat_x28___x29___closed__6 = (const lean_object*)&lp_mathlib_termOfNat_x28___x29___closed__6_value;
static const lean_ctor_object lp_mathlib_termOfNat_x28___x29___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_termOfNat_x28___x29___closed__6_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_termOfNat_x28___x29___closed__7 = (const lean_object*)&lp_mathlib_termOfNat_x28___x29___closed__7_value;
static const lean_ctor_object lp_mathlib_termOfNat_x28___x29___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_termOfNat_x28___x29___closed__7_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_termOfNat_x28___x29___closed__8 = (const lean_object*)&lp_mathlib_termOfNat_x28___x29___closed__8_value;
static const lean_ctor_object lp_mathlib_termOfNat_x28___x29___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_termOfNat_x28___x29___closed__3_value),((lean_object*)&lp_mathlib_termOfNat_x28___x29___closed__5_value),((lean_object*)&lp_mathlib_termOfNat_x28___x29___closed__8_value)}};
static const lean_object* lp_mathlib_termOfNat_x28___x29___closed__9 = (const lean_object*)&lp_mathlib_termOfNat_x28___x29___closed__9_value;
static const lean_string_object lp_mathlib_termOfNat_x28___x29___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ")"};
static const lean_object* lp_mathlib_termOfNat_x28___x29___closed__10 = (const lean_object*)&lp_mathlib_termOfNat_x28___x29___closed__10_value;
static const lean_ctor_object lp_mathlib_termOfNat_x28___x29___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_termOfNat_x28___x29___closed__10_value)}};
static const lean_object* lp_mathlib_termOfNat_x28___x29___closed__11 = (const lean_object*)&lp_mathlib_termOfNat_x28___x29___closed__11_value;
static const lean_ctor_object lp_mathlib_termOfNat_x28___x29___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_termOfNat_x28___x29___closed__3_value),((lean_object*)&lp_mathlib_termOfNat_x28___x29___closed__9_value),((lean_object*)&lp_mathlib_termOfNat_x28___x29___closed__11_value)}};
static const lean_object* lp_mathlib_termOfNat_x28___x29___closed__12 = (const lean_object*)&lp_mathlib_termOfNat_x28___x29___closed__12_value;
static const lean_ctor_object lp_mathlib_termOfNat_x28___x29___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_termOfNat_x28___x29___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_termOfNat_x28___x29___closed__12_value)}};
static const lean_object* lp_mathlib_termOfNat_x28___x29___closed__13 = (const lean_object*)&lp_mathlib_termOfNat_x28___x29___closed__13_value;
LEAN_EXPORT const lean_object* lp_mathlib_termOfNat_x28___x29 = (const lean_object*)&lp_mathlib_termOfNat_x28___x29___closed__13_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__0_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__1_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__2_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "noindex"};
static const lean_object* lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__3_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__4_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__4_value_aux_1),((lean_object*)&lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__4_value_aux_2),((lean_object*)&lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(0, 143, 63, 201, 38, 174, 32, 127)}};
static const lean_object* lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__4 = (const lean_object*)&lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__4_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "no_index"};
static const lean_object* lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__5 = (const lean_object*)&lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__5_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "paren"};
static const lean_object* lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__6 = (const lean_object*)&lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__6_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__7_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__7_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__7_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__7_value_aux_1),((lean_object*)&lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__7_value_aux_2),((lean_object*)&lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(124, 9, 161, 194, 227, 100, 20, 110)}};
static const lean_object* lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__7 = (const lean_object*)&lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__7_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "hygienicLParen"};
static const lean_object* lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__8 = (const lean_object*)&lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__8_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__9_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__9_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__9_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__9_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__9_value_aux_1),((lean_object*)&lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__9_value_aux_2),((lean_object*)&lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(41, 104, 206, 51, 21, 254, 100, 101)}};
static const lean_object* lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__9 = (const lean_object*)&lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__9_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "("};
static const lean_object* lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__10 = (const lean_object*)&lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__10_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "hygieneInfo"};
static const lean_object* lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__11 = (const lean_object*)&lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__11_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__11_value),LEAN_SCALAR_PTR_LITERAL(27, 64, 36, 144, 170, 151, 255, 136)}};
static const lean_object* lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__12 = (const lean_object*)&lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__12_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__13 = (const lean_object*)&lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__13_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__14;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__15 = (const lean_object*)&lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__15_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__15_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__16 = (const lean_object*)&lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__16_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__17 = (const lean_object*)&lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__17_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__18_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__18_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__18_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__18_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__18_value_aux_1),((lean_object*)&lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__18_value_aux_2),((lean_object*)&lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__17_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__18 = (const lean_object*)&lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__18_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "OfNat.ofNat"};
static const lean_object* lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__19 = (const lean_object*)&lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__19_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__20_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__20;
static const lean_string_object lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "OfNat"};
static const lean_object* lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__21 = (const lean_object*)&lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__21_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ofNat"};
static const lean_object* lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__22 = (const lean_object*)&lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__22_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__23_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__21_value),LEAN_SCALAR_PTR_LITERAL(135, 241, 166, 108, 243, 216, 193, 244)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__23_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__22_value),LEAN_SCALAR_PTR_LITERAL(2, 108, 58, 34, 100, 49, 50, 216)}};
static const lean_object* lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__23 = (const lean_object*)&lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__23_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__23_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__24 = (const lean_object*)&lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__24_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__24_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__25 = (const lean_object*)&lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__25_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__26 = (const lean_object*)&lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__26_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__26_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__27 = (const lean_object*)&lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__27_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___boxed(lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__14(void){
_start:
{
lean_object* v___x_59_; lean_object* v___x_60_; 
v___x_59_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__13));
v___x_60_ = l_String_toRawSubstring_x27(v___x_59_);
return v___x_60_;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__20(void){
_start:
{
lean_object* v___x_73_; lean_object* v___x_74_; 
v___x_73_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__19));
v___x_74_ = l_String_toRawSubstring_x27(v___x_73_);
return v___x_74_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1(lean_object* v_x_89_, lean_object* v_a_90_, lean_object* v_a_91_){
_start:
{
lean_object* v___x_92_; uint8_t v___x_93_; 
v___x_92_ = ((lean_object*)(lp_mathlib_termOfNat_x28___x29___closed__1));
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
lean_object* v_quotContext_96_; lean_object* v_currMacroScope_97_; lean_object* v_ref_98_; lean_object* v___x_99_; lean_object* v___x_100_; uint8_t v___x_101_; lean_object* v___x_102_; lean_object* v___x_103_; lean_object* v___x_104_; lean_object* v___x_105_; lean_object* v___x_106_; lean_object* v___x_107_; lean_object* v___x_108_; lean_object* v___x_109_; lean_object* v___x_110_; lean_object* v___x_111_; lean_object* v___x_112_; lean_object* v___x_113_; lean_object* v___x_114_; lean_object* v___x_115_; lean_object* v___x_116_; lean_object* v___x_117_; lean_object* v___x_118_; lean_object* v___x_119_; lean_object* v___x_120_; lean_object* v___x_121_; lean_object* v___x_122_; lean_object* v___x_123_; lean_object* v___x_124_; lean_object* v___x_125_; lean_object* v___x_126_; lean_object* v___x_127_; lean_object* v___x_128_; lean_object* v___x_129_; lean_object* v___x_130_; lean_object* v___x_131_; 
v_quotContext_96_ = lean_ctor_get(v_a_90_, 1);
v_currMacroScope_97_ = lean_ctor_get(v_a_90_, 2);
v_ref_98_ = lean_ctor_get(v_a_90_, 5);
v___x_99_ = lean_unsigned_to_nat(1u);
v___x_100_ = l_Lean_Syntax_getArg(v_x_89_, v___x_99_);
lean_dec(v_x_89_);
v___x_101_ = 0;
v___x_102_ = l_Lean_SourceInfo_fromRef(v_ref_98_, v___x_101_);
v___x_103_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__4));
v___x_104_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__5));
lean_inc_n(v___x_102_, 10);
v___x_105_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_105_, 0, v___x_102_);
lean_ctor_set(v___x_105_, 1, v___x_104_);
v___x_106_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__7));
v___x_107_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__9));
v___x_108_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__10));
v___x_109_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_109_, 0, v___x_102_);
lean_ctor_set(v___x_109_, 1, v___x_108_);
v___x_110_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__12));
v___x_111_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__14, &lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__14_once, _init_lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__14);
v___x_112_ = lean_box(0);
lean_inc_n(v_currMacroScope_97_, 2);
lean_inc_n(v_quotContext_96_, 2);
v___x_113_ = l_Lean_addMacroScope(v_quotContext_96_, v___x_112_, v_currMacroScope_97_);
v___x_114_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__16));
v___x_115_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_115_, 0, v___x_102_);
lean_ctor_set(v___x_115_, 1, v___x_111_);
lean_ctor_set(v___x_115_, 2, v___x_113_);
lean_ctor_set(v___x_115_, 3, v___x_114_);
v___x_116_ = l_Lean_Syntax_node1(v___x_102_, v___x_110_, v___x_115_);
v___x_117_ = l_Lean_Syntax_node2(v___x_102_, v___x_107_, v___x_109_, v___x_116_);
v___x_118_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__18));
v___x_119_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__20, &lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__20_once, _init_lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__20);
v___x_120_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__23));
v___x_121_ = l_Lean_addMacroScope(v_quotContext_96_, v___x_120_, v_currMacroScope_97_);
v___x_122_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__25));
v___x_123_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_123_, 0, v___x_102_);
lean_ctor_set(v___x_123_, 1, v___x_119_);
lean_ctor_set(v___x_123_, 2, v___x_121_);
lean_ctor_set(v___x_123_, 3, v___x_122_);
v___x_124_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___closed__27));
v___x_125_ = l_Lean_Syntax_node1(v___x_102_, v___x_124_, v___x_100_);
v___x_126_ = l_Lean_Syntax_node2(v___x_102_, v___x_118_, v___x_123_, v___x_125_);
v___x_127_ = ((lean_object*)(lp_mathlib_termOfNat_x28___x29___closed__10));
v___x_128_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_128_, 0, v___x_102_);
lean_ctor_set(v___x_128_, 1, v___x_127_);
v___x_129_ = l_Lean_Syntax_node3(v___x_102_, v___x_106_, v___x_117_, v___x_126_, v___x_128_);
v___x_130_ = l_Lean_Syntax_node2(v___x_102_, v___x_103_, v___x_105_, v___x_129_);
v___x_131_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_131_, 0, v___x_130_);
lean_ctor_set(v___x_131_, 1, v_a_91_);
return v___x_131_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1___boxed(lean_object* v_x_132_, lean_object* v_a_133_, lean_object* v_a_134_){
_start:
{
lean_object* v_res_135_; 
v_res_135_ = lp_mathlib___aux__Mathlib__Tactic__OfNat______macroRules__termOfNat_x28___x29__1(v_x_132_, v_a_133_, v_a_134_);
lean_dec_ref(v_a_133_);
return v_res_135_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_OfNat(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_OfNat(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_OfNat(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_OfNat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_OfNat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_OfNat(builtin);
}
#ifdef __cplusplus
}
#endif
