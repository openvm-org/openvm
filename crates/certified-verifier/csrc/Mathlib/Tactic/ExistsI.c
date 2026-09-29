// Lean compiler output
// Module: Mathlib.Tactic.ExistsI
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
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArgs(lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Array_mkArray0(lean_object*);
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_tacticExistsi___x2c_x2c___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticExistsi___x2c_x2c___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticExistsi___x2c_x2c___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_tacticExistsi___x2c_x2c___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticExistsi___x2c_x2c___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticExistsi___x2c_x2c___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_tacticExistsi___x2c_x2c___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "tacticExistsi_,,"};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticExistsi___x2c_x2c___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticExistsi___x2c_x2c___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticExistsi___x2c_x2c___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticExistsi___x2c_x2c___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticExistsi___x2c_x2c___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticExistsi___x2c_x2c___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticExistsi___x2c_x2c___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticExistsi___x2c_x2c___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticExistsi___x2c_x2c___closed__3_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticExistsi___x2c_x2c___closed__2_value),LEAN_SCALAR_PTR_LITERAL(184, 38, 216, 233, 15, 67, 211, 158)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticExistsi___x2c_x2c___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticExistsi___x2c_x2c___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_tacticExistsi___x2c_x2c___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticExistsi___x2c_x2c___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticExistsi___x2c_x2c___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticExistsi___x2c_x2c___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticExistsi___x2c_x2c___closed__4_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticExistsi___x2c_x2c___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticExistsi___x2c_x2c___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_tacticExistsi___x2c_x2c___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "existsi "};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticExistsi___x2c_x2c___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticExistsi___x2c_x2c___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticExistsi___x2c_x2c___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticExistsi___x2c_x2c___closed__6_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticExistsi___x2c_x2c___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticExistsi___x2c_x2c___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_tacticExistsi___x2c_x2c___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticExistsi___x2c_x2c___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticExistsi___x2c_x2c___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticExistsi___x2c_x2c___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticExistsi___x2c_x2c___closed__8_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticExistsi___x2c_x2c___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticExistsi___x2c_x2c___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticExistsi___x2c_x2c___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticExistsi___x2c_x2c___closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticExistsi___x2c_x2c___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticExistsi___x2c_x2c___closed__10_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_tacticExistsi___x2c_x2c___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ","};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticExistsi___x2c_x2c___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticExistsi___x2c_x2c___closed__11_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_tacticExistsi___x2c_x2c___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ", "};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticExistsi___x2c_x2c___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticExistsi___x2c_x2c___closed__12_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticExistsi___x2c_x2c___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticExistsi___x2c_x2c___closed__12_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticExistsi___x2c_x2c___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticExistsi___x2c_x2c___closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticExistsi___x2c_x2c___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 8, .m_other = 3, .m_tag = 11}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticExistsi___x2c_x2c___closed__10_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticExistsi___x2c_x2c___closed__11_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticExistsi___x2c_x2c___closed__13_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticExistsi___x2c_x2c___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticExistsi___x2c_x2c___closed__14_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticExistsi___x2c_x2c___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticExistsi___x2c_x2c___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticExistsi___x2c_x2c___closed__7_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticExistsi___x2c_x2c___closed__14_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticExistsi___x2c_x2c___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticExistsi___x2c_x2c___closed__15_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticExistsi___x2c_x2c___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticExistsi___x2c_x2c___closed__3_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticExistsi___x2c_x2c___closed__15_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticExistsi___x2c_x2c___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticExistsi___x2c_x2c___closed__16_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_tacticExistsi___x2c_x2c = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticExistsi___x2c_x2c___closed__16_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ExistsI______macroRules__Mathlib__Tactic__tacticExistsi___x2c_x2c__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ExistsI______macroRules__Mathlib__Tactic__tacticExistsi___x2c_x2c__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ExistsI______macroRules__Mathlib__Tactic__tacticExistsi___x2c_x2c__1___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ExistsI______macroRules__Mathlib__Tactic__tacticExistsi___x2c_x2c__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ExistsI______macroRules__Mathlib__Tactic__tacticExistsi___x2c_x2c__1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ExistsI______macroRules__Mathlib__Tactic__tacticExistsi___x2c_x2c__1___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ExistsI______macroRules__Mathlib__Tactic__tacticExistsi___x2c_x2c__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "refine"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ExistsI______macroRules__Mathlib__Tactic__tacticExistsi___x2c_x2c__1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ExistsI______macroRules__Mathlib__Tactic__tacticExistsi___x2c_x2c__1___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ExistsI______macroRules__Mathlib__Tactic__tacticExistsi___x2c_x2c__1___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ExistsI______macroRules__Mathlib__Tactic__tacticExistsi___x2c_x2c__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ExistsI______macroRules__Mathlib__Tactic__tacticExistsi___x2c_x2c__1___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ExistsI______macroRules__Mathlib__Tactic__tacticExistsi___x2c_x2c__1___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ExistsI______macroRules__Mathlib__Tactic__tacticExistsi___x2c_x2c__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ExistsI______macroRules__Mathlib__Tactic__tacticExistsi___x2c_x2c__1___closed__3_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ExistsI______macroRules__Mathlib__Tactic__tacticExistsi___x2c_x2c__1___closed__3_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticExistsi___x2c_x2c___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ExistsI______macroRules__Mathlib__Tactic__tacticExistsi___x2c_x2c__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ExistsI______macroRules__Mathlib__Tactic__tacticExistsi___x2c_x2c__1___closed__3_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ExistsI______macroRules__Mathlib__Tactic__tacticExistsi___x2c_x2c__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(49, 130, 130, 160, 131, 48, 178, 245)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ExistsI______macroRules__Mathlib__Tactic__tacticExistsi___x2c_x2c__1___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ExistsI______macroRules__Mathlib__Tactic__tacticExistsi___x2c_x2c__1___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ExistsI______macroRules__Mathlib__Tactic__tacticExistsi___x2c_x2c__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ExistsI______macroRules__Mathlib__Tactic__tacticExistsi___x2c_x2c__1___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ExistsI______macroRules__Mathlib__Tactic__tacticExistsi___x2c_x2c__1___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ExistsI______macroRules__Mathlib__Tactic__tacticExistsi___x2c_x2c__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "anonymousCtor"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ExistsI______macroRules__Mathlib__Tactic__tacticExistsi___x2c_x2c__1___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ExistsI______macroRules__Mathlib__Tactic__tacticExistsi___x2c_x2c__1___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ExistsI______macroRules__Mathlib__Tactic__tacticExistsi___x2c_x2c__1___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ExistsI______macroRules__Mathlib__Tactic__tacticExistsi___x2c_x2c__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ExistsI______macroRules__Mathlib__Tactic__tacticExistsi___x2c_x2c__1___closed__6_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ExistsI______macroRules__Mathlib__Tactic__tacticExistsi___x2c_x2c__1___closed__6_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ExistsI______macroRules__Mathlib__Tactic__tacticExistsi___x2c_x2c__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ExistsI______macroRules__Mathlib__Tactic__tacticExistsi___x2c_x2c__1___closed__6_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ExistsI______macroRules__Mathlib__Tactic__tacticExistsi___x2c_x2c__1___closed__6_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ExistsI______macroRules__Mathlib__Tactic__tacticExistsi___x2c_x2c__1___closed__4_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ExistsI______macroRules__Mathlib__Tactic__tacticExistsi___x2c_x2c__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ExistsI______macroRules__Mathlib__Tactic__tacticExistsi___x2c_x2c__1___closed__6_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ExistsI______macroRules__Mathlib__Tactic__tacticExistsi___x2c_x2c__1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(56, 53, 154, 97, 179, 232, 94, 186)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ExistsI______macroRules__Mathlib__Tactic__tacticExistsi___x2c_x2c__1___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ExistsI______macroRules__Mathlib__Tactic__tacticExistsi___x2c_x2c__1___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ExistsI______macroRules__Mathlib__Tactic__tacticExistsi___x2c_x2c__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "⟨"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ExistsI______macroRules__Mathlib__Tactic__tacticExistsi___x2c_x2c__1___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ExistsI______macroRules__Mathlib__Tactic__tacticExistsi___x2c_x2c__1___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ExistsI______macroRules__Mathlib__Tactic__tacticExistsi___x2c_x2c__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ExistsI______macroRules__Mathlib__Tactic__tacticExistsi___x2c_x2c__1___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ExistsI______macroRules__Mathlib__Tactic__tacticExistsi___x2c_x2c__1___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ExistsI______macroRules__Mathlib__Tactic__tacticExistsi___x2c_x2c__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ExistsI______macroRules__Mathlib__Tactic__tacticExistsi___x2c_x2c__1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ExistsI______macroRules__Mathlib__Tactic__tacticExistsi___x2c_x2c__1___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ExistsI______macroRules__Mathlib__Tactic__tacticExistsi___x2c_x2c__1___closed__9_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ExistsI______macroRules__Mathlib__Tactic__tacticExistsi___x2c_x2c__1___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ExistsI______macroRules__Mathlib__Tactic__tacticExistsi___x2c_x2c__1___closed__10;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ExistsI______macroRules__Mathlib__Tactic__tacticExistsi___x2c_x2c__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "syntheticHole"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ExistsI______macroRules__Mathlib__Tactic__tacticExistsi___x2c_x2c__1___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ExistsI______macroRules__Mathlib__Tactic__tacticExistsi___x2c_x2c__1___closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ExistsI______macroRules__Mathlib__Tactic__tacticExistsi___x2c_x2c__1___closed__12_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ExistsI______macroRules__Mathlib__Tactic__tacticExistsi___x2c_x2c__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ExistsI______macroRules__Mathlib__Tactic__tacticExistsi___x2c_x2c__1___closed__12_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ExistsI______macroRules__Mathlib__Tactic__tacticExistsi___x2c_x2c__1___closed__12_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ExistsI______macroRules__Mathlib__Tactic__tacticExistsi___x2c_x2c__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ExistsI______macroRules__Mathlib__Tactic__tacticExistsi___x2c_x2c__1___closed__12_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ExistsI______macroRules__Mathlib__Tactic__tacticExistsi___x2c_x2c__1___closed__12_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ExistsI______macroRules__Mathlib__Tactic__tacticExistsi___x2c_x2c__1___closed__4_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ExistsI______macroRules__Mathlib__Tactic__tacticExistsi___x2c_x2c__1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ExistsI______macroRules__Mathlib__Tactic__tacticExistsi___x2c_x2c__1___closed__12_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ExistsI______macroRules__Mathlib__Tactic__tacticExistsi___x2c_x2c__1___closed__11_value),LEAN_SCALAR_PTR_LITERAL(218, 189, 67, 60, 211, 196, 112, 165)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ExistsI______macroRules__Mathlib__Tactic__tacticExistsi___x2c_x2c__1___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ExistsI______macroRules__Mathlib__Tactic__tacticExistsi___x2c_x2c__1___closed__12_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ExistsI______macroRules__Mathlib__Tactic__tacticExistsi___x2c_x2c__1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "\?"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ExistsI______macroRules__Mathlib__Tactic__tacticExistsi___x2c_x2c__1___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ExistsI______macroRules__Mathlib__Tactic__tacticExistsi___x2c_x2c__1___closed__13_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ExistsI______macroRules__Mathlib__Tactic__tacticExistsi___x2c_x2c__1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "_"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ExistsI______macroRules__Mathlib__Tactic__tacticExistsi___x2c_x2c__1___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ExistsI______macroRules__Mathlib__Tactic__tacticExistsi___x2c_x2c__1___closed__14_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ExistsI______macroRules__Mathlib__Tactic__tacticExistsi___x2c_x2c__1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "⟩"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ExistsI______macroRules__Mathlib__Tactic__tacticExistsi___x2c_x2c__1___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ExistsI______macroRules__Mathlib__Tactic__tacticExistsi___x2c_x2c__1___closed__15_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ExistsI______macroRules__Mathlib__Tactic__tacticExistsi___x2c_x2c__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ExistsI______macroRules__Mathlib__Tactic__tacticExistsi___x2c_x2c__1___boxed(lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ExistsI______macroRules__Mathlib__Tactic__tacticExistsi___x2c_x2c__1___closed__10(void){
_start:
{
lean_object* v___x_58_; 
v___x_58_ = l_Array_mkArray0(lean_box(0));
return v___x_58_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ExistsI______macroRules__Mathlib__Tactic__tacticExistsi___x2c_x2c__1(lean_object* v_x_68_, lean_object* v_a_69_, lean_object* v_a_70_){
_start:
{
lean_object* v___x_71_; uint8_t v___x_72_; 
v___x_71_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticExistsi___x2c_x2c___closed__3));
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
lean_object* v_ref_75_; lean_object* v___x_76_; lean_object* v___x_77_; lean_object* v___x_78_; lean_object* v___x_79_; uint8_t v___x_80_; lean_object* v___x_81_; lean_object* v___x_82_; lean_object* v___x_83_; lean_object* v___x_84_; lean_object* v___x_85_; lean_object* v___x_86_; lean_object* v___x_87_; lean_object* v___x_88_; lean_object* v___x_89_; lean_object* v___x_90_; lean_object* v___x_91_; lean_object* v___x_92_; lean_object* v___x_93_; lean_object* v___x_94_; lean_object* v___x_95_; lean_object* v___x_96_; lean_object* v___x_97_; lean_object* v___x_98_; lean_object* v___x_99_; lean_object* v___x_100_; lean_object* v___x_101_; lean_object* v___x_102_; lean_object* v___x_103_; lean_object* v___x_104_; lean_object* v___x_105_; 
v_ref_75_ = lean_ctor_get(v_a_69_, 5);
v___x_76_ = lean_unsigned_to_nat(1u);
v___x_77_ = l_Lean_Syntax_getArg(v_x_68_, v___x_76_);
lean_dec(v_x_68_);
v___x_78_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticExistsi___x2c_x2c___closed__11));
v___x_79_ = l_Lean_Syntax_getArgs(v___x_77_);
lean_dec(v___x_77_);
v___x_80_ = 0;
v___x_81_ = l_Lean_SourceInfo_fromRef(v_ref_75_, v___x_80_);
v___x_82_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ExistsI______macroRules__Mathlib__Tactic__tacticExistsi___x2c_x2c__1___closed__2));
v___x_83_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ExistsI______macroRules__Mathlib__Tactic__tacticExistsi___x2c_x2c__1___closed__3));
lean_inc_n(v___x_81_, 9);
v___x_84_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_84_, 0, v___x_81_);
lean_ctor_set(v___x_84_, 1, v___x_82_);
v___x_85_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ExistsI______macroRules__Mathlib__Tactic__tacticExistsi___x2c_x2c__1___closed__6));
v___x_86_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ExistsI______macroRules__Mathlib__Tactic__tacticExistsi___x2c_x2c__1___closed__7));
v___x_87_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_87_, 0, v___x_81_);
lean_ctor_set(v___x_87_, 1, v___x_86_);
v___x_88_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ExistsI______macroRules__Mathlib__Tactic__tacticExistsi___x2c_x2c__1___closed__9));
v___x_89_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ExistsI______macroRules__Mathlib__Tactic__tacticExistsi___x2c_x2c__1___closed__10, &lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ExistsI______macroRules__Mathlib__Tactic__tacticExistsi___x2c_x2c__1___closed__10_once, _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ExistsI______macroRules__Mathlib__Tactic__tacticExistsi___x2c_x2c__1___closed__10);
v___x_90_ = l_Array_append___redArg(v___x_89_, v___x_79_);
lean_dec_ref(v___x_79_);
v___x_91_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_91_, 0, v___x_81_);
lean_ctor_set(v___x_91_, 1, v___x_78_);
v___x_92_ = lean_array_push(v___x_90_, v___x_91_);
v___x_93_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ExistsI______macroRules__Mathlib__Tactic__tacticExistsi___x2c_x2c__1___closed__12));
v___x_94_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ExistsI______macroRules__Mathlib__Tactic__tacticExistsi___x2c_x2c__1___closed__13));
v___x_95_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_95_, 0, v___x_81_);
lean_ctor_set(v___x_95_, 1, v___x_94_);
v___x_96_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ExistsI______macroRules__Mathlib__Tactic__tacticExistsi___x2c_x2c__1___closed__14));
v___x_97_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_97_, 0, v___x_81_);
lean_ctor_set(v___x_97_, 1, v___x_96_);
v___x_98_ = l_Lean_Syntax_node2(v___x_81_, v___x_93_, v___x_95_, v___x_97_);
v___x_99_ = lean_array_push(v___x_92_, v___x_98_);
v___x_100_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_100_, 0, v___x_81_);
lean_ctor_set(v___x_100_, 1, v___x_88_);
lean_ctor_set(v___x_100_, 2, v___x_99_);
v___x_101_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ExistsI______macroRules__Mathlib__Tactic__tacticExistsi___x2c_x2c__1___closed__15));
v___x_102_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_102_, 0, v___x_81_);
lean_ctor_set(v___x_102_, 1, v___x_101_);
v___x_103_ = l_Lean_Syntax_node3(v___x_81_, v___x_85_, v___x_87_, v___x_100_, v___x_102_);
v___x_104_ = l_Lean_Syntax_node2(v___x_81_, v___x_83_, v___x_84_, v___x_103_);
v___x_105_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_105_, 0, v___x_104_);
lean_ctor_set(v___x_105_, 1, v_a_70_);
return v___x_105_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ExistsI______macroRules__Mathlib__Tactic__tacticExistsi___x2c_x2c__1___boxed(lean_object* v_x_106_, lean_object* v_a_107_, lean_object* v_a_108_){
_start:
{
lean_object* v_res_109_; 
v_res_109_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ExistsI______macroRules__Mathlib__Tactic__tacticExistsi___x2c_x2c__1(v_x_106_, v_a_107_, v_a_108_);
lean_dec_ref(v_a_107_);
return v_res_109_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_ExistsI(uint8_t builtin) {
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
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_ExistsI(uint8_t builtin) {
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
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_ExistsI(uint8_t builtin) {
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
res = runtime_initialize_mathlib_Mathlib_Tactic_ExistsI(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_ExistsI(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_ExistsI(builtin);
}
#ifdef __cplusplus
}
#endif
