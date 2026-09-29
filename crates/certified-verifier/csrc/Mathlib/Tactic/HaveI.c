// Lean compiler output
// Module: Mathlib.Tactic.HaveI
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
lean_object* l_Lean_Name_str___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_Name_num___override(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Array_mkArray0(lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "_private"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(103, 214, 75, 80, 34, 198, 193, 153)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__2_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__1_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(234, 232, 174, 134, 127, 136, 69, 92)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__3_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__4_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__3_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__4_value),LEAN_SCALAR_PTR_LITERAL(191, 70, 156, 159, 11, 54, 216, 94)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__5_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "HaveI"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__6_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__5_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(233, 195, 53, 213, 164, 58, 84, 243)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__7_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__7_value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(44, 150, 179, 152, 248, 30, 117, 130)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__8_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__8_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(189, 166, 148, 76, 127, 31, 141, 198)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__9_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__9_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__4_value),LEAN_SCALAR_PTR_LITERAL(228, 192, 243, 20, 154, 42, 150, 161)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__10_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__10_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(182, 78, 6, 160, 207, 251, 173, 77)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__11_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "termHaveIDummy_"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__12_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__11_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__12_value),LEAN_SCALAR_PTR_LITERAL(59, 250, 249, 203, 241, 191, 34, 240)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__13 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__13_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__14 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__14_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__14_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__15 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__15_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "haveIDummy"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__16 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__16_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__16_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__17 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__17_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "letDecl"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__18 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__18_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__18_value),LEAN_SCALAR_PTR_LITERAL(237, 158, 72, 239, 156, 118, 8, 209)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__19 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__19_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__19_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__20 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__20_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__15_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__17_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__20_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__21 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__21_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__13_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__21_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__22 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__22_value;
LEAN_EXPORT const lean_object* lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy__ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__22_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "assert"};
static const lean_object* lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(193, 125, 133, 119, 190, 55, 66, 188)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__5_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__5_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__5_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__5_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__5_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__18_value),LEAN_SCALAR_PTR_LITERAL(61, 47, 121, 206, 37, 68, 134, 111)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "haveI"};
static const lean_object* lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__7_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__7_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__7_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__7_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__7_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(85, 132, 233, 17, 205, 1, 111, 129)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "letConfig"};
static const lean_object* lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__9_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__9_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__9_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__9_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__9_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__9_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(5, 186, 227, 151, 19, 40, 136, 241)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__10_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__11_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__12;
static const lean_string_object lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ";"};
static const lean_object* lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__13_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_HaveI_doElemHaveI_x27___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "doElemHaveI'_"};
static const lean_object* lp_mathlib_Mathlib_Tactic_HaveI_doElemHaveI_x27___00__closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_HaveI_doElemHaveI_x27___00__closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_HaveI_doElemHaveI_x27___00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_HaveI_doElemHaveI_x27___00__closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_HaveI_doElemHaveI_x27___00__closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__4_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_HaveI_doElemHaveI_x27___00__closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_HaveI_doElemHaveI_x27___00__closed__1_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(133, 97, 20, 47, 11, 241, 1, 129)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_HaveI_doElemHaveI_x27___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_HaveI_doElemHaveI_x27___00__closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_HaveI_doElemHaveI_x27___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(17, 22, 228, 141, 177, 154, 80, 63)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_HaveI_doElemHaveI_x27___00__closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_HaveI_doElemHaveI_x27___00__closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_HaveI_doElemHaveI_x27___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "haveI' "};
static const lean_object* lp_mathlib_Mathlib_Tactic_HaveI_doElemHaveI_x27___00__closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_HaveI_doElemHaveI_x27___00__closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_HaveI_doElemHaveI_x27___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_HaveI_doElemHaveI_x27___00__closed__2_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_HaveI_doElemHaveI_x27___00__closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_HaveI_doElemHaveI_x27___00__closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_HaveI_doElemHaveI_x27___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__15_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_HaveI_doElemHaveI_x27___00__closed__3_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__20_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_HaveI_doElemHaveI_x27___00__closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_HaveI_doElemHaveI_x27___00__closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_HaveI_doElemHaveI_x27___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_HaveI_doElemHaveI_x27___00__closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_HaveI_doElemHaveI_x27___00__closed__4_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_HaveI_doElemHaveI_x27___00__closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_HaveI_doElemHaveI_x27___00__closed__5_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_HaveI_doElemHaveI_x27__ = (const lean_object*)&lp_mathlib_Mathlib_Tactic_HaveI_doElemHaveI_x27___00__closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Mathlib__Tactic__HaveI__doElemHaveI_x27____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "doAssert"};
static const lean_object* lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Mathlib__Tactic__HaveI__doElemHaveI_x27____1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Mathlib__Tactic__HaveI__doElemHaveI_x27____1___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Mathlib__Tactic__HaveI__doElemHaveI_x27____1___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Mathlib__Tactic__HaveI__doElemHaveI_x27____1___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Mathlib__Tactic__HaveI__doElemHaveI_x27____1___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Mathlib__Tactic__HaveI__doElemHaveI_x27____1___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Mathlib__Tactic__HaveI__doElemHaveI_x27____1___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Mathlib__Tactic__HaveI__doElemHaveI_x27____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Mathlib__Tactic__HaveI__doElemHaveI_x27____1___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Mathlib__Tactic__HaveI__doElemHaveI_x27____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(171, 15, 212, 125, 46, 208, 251, 33)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Mathlib__Tactic__HaveI__doElemHaveI_x27____1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Mathlib__Tactic__HaveI__doElemHaveI_x27____1___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Mathlib__Tactic__HaveI__doElemHaveI_x27____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "assert!"};
static const lean_object* lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Mathlib__Tactic__HaveI__doElemHaveI_x27____1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Mathlib__Tactic__HaveI__doElemHaveI_x27____1___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Mathlib__Tactic__HaveI__doElemHaveI_x27____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Mathlib__Tactic__HaveI__doElemHaveI_x27____1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termLetIDummy___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "termLetIDummy_"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termLetIDummy___00__closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termLetIDummy___00__closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termLetIDummy___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__11_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termLetIDummy___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(165, 81, 112, 4, 144, 186, 156, 105)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termLetIDummy___00__closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termLetIDummy___00__closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termLetIDummy___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "letIDummy"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termLetIDummy___00__closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termLetIDummy___00__closed__2_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termLetIDummy___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termLetIDummy___00__closed__2_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termLetIDummy___00__closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termLetIDummy___00__closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termLetIDummy___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__15_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termLetIDummy___00__closed__3_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__20_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termLetIDummy___00__closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termLetIDummy___00__closed__4_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termLetIDummy___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termLetIDummy___00__closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termLetIDummy___00__closed__4_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termLetIDummy___00__closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termLetIDummy___00__closed__5_value;
LEAN_EXPORT const lean_object* lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termLetIDummy__ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termLetIDummy___00__closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "letI"};
static const lean_object* lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__2___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__2___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__2___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__2___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__2___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__2___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__2___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__2___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__2___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__2___closed__0_value),LEAN_SCALAR_PTR_LITERAL(12, 85, 65, 24, 247, 201, 209, 85)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__2___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__2___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__2___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_HaveI_doElemLetI_x27___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "doElemLetI'_"};
static const lean_object* lp_mathlib_Mathlib_Tactic_HaveI_doElemLetI_x27___00__closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_HaveI_doElemLetI_x27___00__closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_HaveI_doElemLetI_x27___00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_HaveI_doElemLetI_x27___00__closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_HaveI_doElemLetI_x27___00__closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__4_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_HaveI_doElemLetI_x27___00__closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_HaveI_doElemLetI_x27___00__closed__1_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(133, 97, 20, 47, 11, 241, 1, 129)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_HaveI_doElemLetI_x27___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_HaveI_doElemLetI_x27___00__closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_HaveI_doElemLetI_x27___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(176, 17, 153, 27, 94, 165, 55, 215)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_HaveI_doElemLetI_x27___00__closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_HaveI_doElemLetI_x27___00__closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_HaveI_doElemLetI_x27___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "letI' "};
static const lean_object* lp_mathlib_Mathlib_Tactic_HaveI_doElemLetI_x27___00__closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_HaveI_doElemLetI_x27___00__closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_HaveI_doElemLetI_x27___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_HaveI_doElemLetI_x27___00__closed__2_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_HaveI_doElemLetI_x27___00__closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_HaveI_doElemLetI_x27___00__closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_HaveI_doElemLetI_x27___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__15_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_HaveI_doElemLetI_x27___00__closed__3_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__20_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_HaveI_doElemLetI_x27___00__closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_HaveI_doElemLetI_x27___00__closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_HaveI_doElemLetI_x27___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_HaveI_doElemLetI_x27___00__closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_HaveI_doElemLetI_x27___00__closed__4_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_HaveI_doElemLetI_x27___00__closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_HaveI_doElemLetI_x27___00__closed__5_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_HaveI_doElemLetI_x27__ = (const lean_object*)&lp_mathlib_Mathlib_Tactic_HaveI_doElemLetI_x27___00__closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Mathlib__Tactic__HaveI__doElemLetI_x27____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Mathlib__Tactic__HaveI__doElemLetI_x27____1___boxed(lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__12(void){
_start:
{
lean_object* v___x_82_; 
v___x_82_ = l_Array_mkArray0(lean_box(0));
return v___x_82_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1(lean_object* v_x_84_, lean_object* v_a_85_, lean_object* v_a_86_){
_start:
{
lean_object* v___x_87_; uint8_t v___x_88_; 
v___x_87_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__4));
lean_inc(v_x_84_);
v___x_88_ = l_Lean_Syntax_isOfKind(v_x_84_, v___x_87_);
if (v___x_88_ == 0)
{
lean_object* v___x_89_; lean_object* v___x_90_; 
lean_dec(v_x_84_);
v___x_89_ = lean_box(1);
v___x_90_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_90_, 0, v___x_89_);
lean_ctor_set(v___x_90_, 1, v_a_86_);
return v___x_90_;
}
else
{
lean_object* v___x_91_; lean_object* v___x_92_; lean_object* v___x_93_; uint8_t v___x_94_; 
v___x_91_ = lean_unsigned_to_nat(1u);
v___x_92_ = l_Lean_Syntax_getArg(v_x_84_, v___x_91_);
v___x_93_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__13));
lean_inc(v___x_92_);
v___x_94_ = l_Lean_Syntax_isOfKind(v___x_92_, v___x_93_);
if (v___x_94_ == 0)
{
lean_object* v___x_95_; lean_object* v___x_96_; 
lean_dec(v___x_92_);
lean_dec(v_x_84_);
v___x_95_ = lean_box(1);
v___x_96_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_96_, 0, v___x_95_);
lean_ctor_set(v___x_96_, 1, v_a_86_);
return v___x_96_;
}
else
{
lean_object* v___x_97_; lean_object* v___x_98_; uint8_t v___x_99_; 
v___x_97_ = l_Lean_Syntax_getArg(v___x_92_, v___x_91_);
lean_dec(v___x_92_);
v___x_98_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__5));
lean_inc(v___x_97_);
v___x_99_ = l_Lean_Syntax_isOfKind(v___x_97_, v___x_98_);
if (v___x_99_ == 0)
{
lean_object* v___x_100_; lean_object* v___x_101_; 
lean_dec(v___x_97_);
lean_dec(v_x_84_);
v___x_100_ = lean_box(1);
v___x_101_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_101_, 0, v___x_100_);
lean_ctor_set(v___x_101_, 1, v_a_86_);
return v___x_101_;
}
else
{
lean_object* v_ref_102_; lean_object* v___x_103_; lean_object* v___x_104_; uint8_t v___x_105_; lean_object* v___x_106_; lean_object* v___x_107_; lean_object* v___x_108_; lean_object* v___x_109_; lean_object* v___x_110_; lean_object* v___x_111_; lean_object* v___x_112_; lean_object* v___x_113_; lean_object* v___x_114_; lean_object* v___x_115_; lean_object* v___x_116_; lean_object* v___x_117_; lean_object* v___x_118_; 
v_ref_102_ = lean_ctor_get(v_a_85_, 5);
v___x_103_ = lean_unsigned_to_nat(3u);
v___x_104_ = l_Lean_Syntax_getArg(v_x_84_, v___x_103_);
lean_dec(v_x_84_);
v___x_105_ = 0;
v___x_106_ = l_Lean_SourceInfo_fromRef(v_ref_102_, v___x_105_);
v___x_107_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__6));
v___x_108_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__7));
lean_inc_n(v___x_106_, 4);
v___x_109_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_109_, 0, v___x_106_);
lean_ctor_set(v___x_109_, 1, v___x_107_);
v___x_110_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__9));
v___x_111_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__11));
v___x_112_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__12, &lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__12_once, _init_lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__12);
v___x_113_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_113_, 0, v___x_106_);
lean_ctor_set(v___x_113_, 1, v___x_111_);
lean_ctor_set(v___x_113_, 2, v___x_112_);
v___x_114_ = l_Lean_Syntax_node1(v___x_106_, v___x_110_, v___x_113_);
v___x_115_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__13));
v___x_116_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_116_, 0, v___x_106_);
lean_ctor_set(v___x_116_, 1, v___x_115_);
v___x_117_ = l_Lean_Syntax_node5(v___x_106_, v___x_108_, v___x_109_, v___x_114_, v___x_97_, v___x_116_, v___x_104_);
v___x_118_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_118_, 0, v___x_117_);
lean_ctor_set(v___x_118_, 1, v_a_86_);
return v___x_118_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___boxed(lean_object* v_x_119_, lean_object* v_a_120_, lean_object* v_a_121_){
_start:
{
lean_object* v_res_122_; 
v_res_122_ = lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1(v_x_119_, v_a_120_, v_a_121_);
lean_dec_ref(v_a_120_);
return v_res_122_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Mathlib__Tactic__HaveI__doElemHaveI_x27____1(lean_object* v_x_148_, lean_object* v_a_149_, lean_object* v_a_150_){
_start:
{
lean_object* v___x_151_; uint8_t v___x_152_; 
v___x_151_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_HaveI_doElemHaveI_x27___00__closed__1));
lean_inc(v_x_148_);
v___x_152_ = l_Lean_Syntax_isOfKind(v_x_148_, v___x_151_);
if (v___x_152_ == 0)
{
lean_object* v___x_153_; lean_object* v___x_154_; 
lean_dec(v_x_148_);
v___x_153_ = lean_box(1);
v___x_154_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_154_, 0, v___x_153_);
lean_ctor_set(v___x_154_, 1, v_a_150_);
return v___x_154_;
}
else
{
lean_object* v_ref_155_; lean_object* v___x_156_; lean_object* v___x_157_; uint8_t v___x_158_; lean_object* v___x_159_; lean_object* v___x_160_; lean_object* v___x_161_; lean_object* v___x_162_; lean_object* v___x_163_; lean_object* v___x_164_; lean_object* v___x_165_; lean_object* v___x_166_; lean_object* v___x_167_; lean_object* v___x_168_; 
v_ref_155_ = lean_ctor_get(v_a_149_, 5);
v___x_156_ = lean_unsigned_to_nat(1u);
v___x_157_ = l_Lean_Syntax_getArg(v_x_148_, v___x_156_);
lean_dec(v_x_148_);
v___x_158_ = 0;
v___x_159_ = l_Lean_SourceInfo_fromRef(v_ref_155_, v___x_158_);
v___x_160_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Mathlib__Tactic__HaveI__doElemHaveI_x27____1___closed__1));
v___x_161_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Mathlib__Tactic__HaveI__doElemHaveI_x27____1___closed__2));
lean_inc_n(v___x_159_, 3);
v___x_162_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_162_, 0, v___x_159_);
lean_ctor_set(v___x_162_, 1, v___x_161_);
v___x_163_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__13));
v___x_164_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termHaveIDummy___00__closed__16));
v___x_165_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_165_, 0, v___x_159_);
lean_ctor_set(v___x_165_, 1, v___x_164_);
v___x_166_ = l_Lean_Syntax_node2(v___x_159_, v___x_163_, v___x_165_, v___x_157_);
v___x_167_ = l_Lean_Syntax_node2(v___x_159_, v___x_160_, v___x_162_, v___x_166_);
v___x_168_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_168_, 0, v___x_167_);
lean_ctor_set(v___x_168_, 1, v_a_150_);
return v___x_168_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Mathlib__Tactic__HaveI__doElemHaveI_x27____1___boxed(lean_object* v_x_169_, lean_object* v_a_170_, lean_object* v_a_171_){
_start:
{
lean_object* v_res_172_; 
v_res_172_ = lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Mathlib__Tactic__HaveI__doElemHaveI_x27____1(v_x_169_, v_a_170_, v_a_171_);
lean_dec_ref(v_a_170_);
return v_res_172_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__2(lean_object* v_x_195_, lean_object* v_a_196_, lean_object* v_a_197_){
_start:
{
lean_object* v___x_198_; uint8_t v___x_199_; 
v___x_198_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__4));
lean_inc(v_x_195_);
v___x_199_ = l_Lean_Syntax_isOfKind(v_x_195_, v___x_198_);
if (v___x_199_ == 0)
{
lean_object* v___x_200_; lean_object* v___x_201_; 
lean_dec(v_x_195_);
v___x_200_ = lean_box(1);
v___x_201_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_201_, 0, v___x_200_);
lean_ctor_set(v___x_201_, 1, v_a_197_);
return v___x_201_;
}
else
{
lean_object* v___x_202_; lean_object* v___x_203_; lean_object* v___x_204_; uint8_t v___x_205_; 
v___x_202_ = lean_unsigned_to_nat(1u);
v___x_203_ = l_Lean_Syntax_getArg(v_x_195_, v___x_202_);
v___x_204_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termLetIDummy___00__closed__1));
lean_inc(v___x_203_);
v___x_205_ = l_Lean_Syntax_isOfKind(v___x_203_, v___x_204_);
if (v___x_205_ == 0)
{
lean_object* v___x_206_; lean_object* v___x_207_; 
lean_dec(v___x_203_);
lean_dec(v_x_195_);
v___x_206_ = lean_box(1);
v___x_207_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_207_, 0, v___x_206_);
lean_ctor_set(v___x_207_, 1, v_a_197_);
return v___x_207_;
}
else
{
lean_object* v___x_208_; lean_object* v___x_209_; uint8_t v___x_210_; 
v___x_208_ = l_Lean_Syntax_getArg(v___x_203_, v___x_202_);
lean_dec(v___x_203_);
v___x_209_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__5));
lean_inc(v___x_208_);
v___x_210_ = l_Lean_Syntax_isOfKind(v___x_208_, v___x_209_);
if (v___x_210_ == 0)
{
lean_object* v___x_211_; lean_object* v___x_212_; 
lean_dec(v___x_208_);
lean_dec(v_x_195_);
v___x_211_ = lean_box(1);
v___x_212_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_212_, 0, v___x_211_);
lean_ctor_set(v___x_212_, 1, v_a_197_);
return v___x_212_;
}
else
{
lean_object* v_ref_213_; lean_object* v___x_214_; lean_object* v___x_215_; uint8_t v___x_216_; lean_object* v___x_217_; lean_object* v___x_218_; lean_object* v___x_219_; lean_object* v___x_220_; lean_object* v___x_221_; lean_object* v___x_222_; lean_object* v___x_223_; lean_object* v___x_224_; lean_object* v___x_225_; lean_object* v___x_226_; lean_object* v___x_227_; lean_object* v___x_228_; lean_object* v___x_229_; 
v_ref_213_ = lean_ctor_get(v_a_196_, 5);
v___x_214_ = lean_unsigned_to_nat(3u);
v___x_215_ = l_Lean_Syntax_getArg(v_x_195_, v___x_214_);
lean_dec(v_x_195_);
v___x_216_ = 0;
v___x_217_ = l_Lean_SourceInfo_fromRef(v_ref_213_, v___x_216_);
v___x_218_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__2___closed__0));
v___x_219_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__2___closed__1));
lean_inc_n(v___x_217_, 4);
v___x_220_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_220_, 0, v___x_217_);
lean_ctor_set(v___x_220_, 1, v___x_218_);
v___x_221_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__9));
v___x_222_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__11));
v___x_223_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__12, &lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__12_once, _init_lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__12);
v___x_224_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_224_, 0, v___x_217_);
lean_ctor_set(v___x_224_, 1, v___x_222_);
lean_ctor_set(v___x_224_, 2, v___x_223_);
v___x_225_ = l_Lean_Syntax_node1(v___x_217_, v___x_221_, v___x_224_);
v___x_226_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__1___closed__13));
v___x_227_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_227_, 0, v___x_217_);
lean_ctor_set(v___x_227_, 1, v___x_226_);
v___x_228_ = l_Lean_Syntax_node5(v___x_217_, v___x_219_, v___x_220_, v___x_225_, v___x_208_, v___x_227_, v___x_215_);
v___x_229_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_229_, 0, v___x_228_);
lean_ctor_set(v___x_229_, 1, v_a_197_);
return v___x_229_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__2___boxed(lean_object* v_x_230_, lean_object* v_a_231_, lean_object* v_a_232_){
_start:
{
lean_object* v_res_233_; 
v_res_233_ = lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Lean__Parser__Term__assert__2(v_x_230_, v_a_231_, v_a_232_);
lean_dec_ref(v_a_231_);
return v_res_233_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Mathlib__Tactic__HaveI__doElemLetI_x27____1(lean_object* v_x_252_, lean_object* v_a_253_, lean_object* v_a_254_){
_start:
{
lean_object* v___x_255_; uint8_t v___x_256_; 
v___x_255_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_HaveI_doElemLetI_x27___00__closed__1));
lean_inc(v_x_252_);
v___x_256_ = l_Lean_Syntax_isOfKind(v_x_252_, v___x_255_);
if (v___x_256_ == 0)
{
lean_object* v___x_257_; lean_object* v___x_258_; 
lean_dec(v_x_252_);
v___x_257_ = lean_box(1);
v___x_258_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_258_, 0, v___x_257_);
lean_ctor_set(v___x_258_, 1, v_a_254_);
return v___x_258_;
}
else
{
lean_object* v_ref_259_; lean_object* v___x_260_; lean_object* v___x_261_; uint8_t v___x_262_; lean_object* v___x_263_; lean_object* v___x_264_; lean_object* v___x_265_; lean_object* v___x_266_; lean_object* v___x_267_; lean_object* v___x_268_; lean_object* v___x_269_; lean_object* v___x_270_; lean_object* v___x_271_; lean_object* v___x_272_; 
v_ref_259_ = lean_ctor_get(v_a_253_, 5);
v___x_260_ = lean_unsigned_to_nat(1u);
v___x_261_ = l_Lean_Syntax_getArg(v_x_252_, v___x_260_);
lean_dec(v_x_252_);
v___x_262_ = 0;
v___x_263_ = l_Lean_SourceInfo_fromRef(v_ref_259_, v___x_262_);
v___x_264_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Mathlib__Tactic__HaveI__doElemHaveI_x27____1___closed__1));
v___x_265_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Mathlib__Tactic__HaveI__doElemHaveI_x27____1___closed__2));
lean_inc_n(v___x_263_, 3);
v___x_266_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_266_, 0, v___x_263_);
lean_ctor_set(v___x_266_, 1, v___x_265_);
v___x_267_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termLetIDummy___00__closed__1));
v___x_268_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_HaveI_0__Mathlib_Tactic_HaveI_termLetIDummy___00__closed__2));
v___x_269_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_269_, 0, v___x_263_);
lean_ctor_set(v___x_269_, 1, v___x_268_);
v___x_270_ = l_Lean_Syntax_node2(v___x_263_, v___x_267_, v___x_269_, v___x_261_);
v___x_271_ = l_Lean_Syntax_node2(v___x_263_, v___x_264_, v___x_266_, v___x_270_);
v___x_272_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_272_, 0, v___x_271_);
lean_ctor_set(v___x_272_, 1, v_a_254_);
return v___x_272_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Mathlib__Tactic__HaveI__doElemLetI_x27____1___boxed(lean_object* v_x_273_, lean_object* v_a_274_, lean_object* v_a_275_){
_start:
{
lean_object* v_res_276_; 
v_res_276_ = lp_mathlib_Mathlib_Tactic_HaveI___aux__Mathlib__Tactic__HaveI______macroRules__Mathlib__Tactic__HaveI__doElemLetI_x27____1(v_x_273_, v_a_274_, v_a_275_);
lean_dec_ref(v_a_274_);
return v_res_276_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_HaveI(uint8_t builtin) {
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
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_HaveI(uint8_t builtin) {
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
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_HaveI(uint8_t builtin) {
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
res = runtime_initialize_mathlib_Mathlib_Tactic_HaveI(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_HaveI(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_HaveI(builtin);
}
#ifdef __cplusplus
}
#endif
