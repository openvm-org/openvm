// Lean compiler output
// Module: Mathlib.Data.List.Sort
// Imports: public import Init public meta import Init public import Batteries.Data.List.Perm public import Mathlib.Data.List.OfFn public import Mathlib.Data.List.Nodup public import Mathlib.Order.Fin.Basic
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
uint8_t lp_batteries_List_instDecidableIsChainOfDecidableRel___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Name_str___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_num___override(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_List_foldrTR___redArg(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "_private"};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(103, 214, 75, 80, 34, 198, 193, 153)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__2_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__1_value),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(234, 232, 174, 134, 127, 136, 69, 92)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__3_value;
static const lean_string_object lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Data"};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__4_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__3_value),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__4_value),LEAN_SCALAR_PTR_LITERAL(149, 48, 214, 5, 44, 128, 44, 12)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__5_value;
static const lean_string_object lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "List"};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__6_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__5_value),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(123, 214, 204, 50, 8, 169, 217, 159)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__7_value;
static const lean_string_object lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Sort"};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__8_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__7_value),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__8_value),LEAN_SCALAR_PTR_LITERAL(208, 225, 136, 231, 132, 214, 35, 89)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__9_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(41, 157, 173, 239, 124, 154, 208, 201)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__10_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__10_value),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(239, 27, 247, 191, 74, 103, 71, 124)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__11_value;
static const lean_string_object lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 7, .m_data = "term_≼_"};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__12_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__11_value),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__12_value),LEAN_SCALAR_PTR_LITERAL(246, 146, 192, 140, 182, 55, 251, 8)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__13 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__13_value;
static const lean_string_object lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__14 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__14_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__14_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__15 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__15_value;
static const lean_string_object lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 3, .m_data = " ≼ "};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__16 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__16_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__16_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__17 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__17_value;
static const lean_string_object lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__18 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__18_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__18_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__19 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__19_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__19_value),((lean_object*)(((size_t)(51) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__20 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__20_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__15_value),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__17_value),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__20_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__21 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__21_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__13_value),((lean_object*)(((size_t)(50) << 1) | 1)),((lean_object*)(((size_t)(50) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__21_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__22 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__22_value;
LEAN_EXPORT const lean_object* lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c__ = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__22_value;
static const lean_string_object lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__2_value;
static const lean_string_object lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__4_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__4_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__4_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__4_value;
static const lean_string_object lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "r"};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__5_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__6;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(201, 206, 29, 183, 206, 15, 98, 41)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__7_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(201, 206, 29, 183, 206, 15, 98, 41)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__8_value;
static const lean_string_object lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "_@"};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__9_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__8_value),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__9_value),LEAN_SCALAR_PTR_LITERAL(100, 2, 144, 119, 127, 225, 14, 168)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__10_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__10_value),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(85, 59, 115, 197, 18, 0, 59, 244)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__11_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__11_value),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__4_value),LEAN_SCALAR_PTR_LITERAL(150, 116, 64, 155, 22, 22, 7, 231)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__12_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__12_value),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(188, 45, 62, 54, 90, 126, 171, 233)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__13 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__13_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__13_value),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__8_value),LEAN_SCALAR_PTR_LITERAL(163, 162, 4, 121, 251, 62, 171, 32)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__14 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__14_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__14_value),((lean_object*)(((size_t)(1913905867) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(173, 218, 5, 91, 192, 38, 98, 147)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__15 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__15_value;
static const lean_string_object lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "_hygCtx"};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__16 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__16_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__15_value),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__16_value),LEAN_SCALAR_PTR_LITERAL(206, 214, 227, 109, 43, 44, 187, 8)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__17 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__17_value;
static const lean_string_object lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "_hyg"};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__18 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__18_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__17_value),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__18_value),LEAN_SCALAR_PTR_LITERAL(162, 38, 18, 137, 155, 208, 183, 145)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__19 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__19_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__19_value),((lean_object*)(((size_t)(17) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(252, 142, 171, 124, 38, 216, 162, 182)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__20 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__20_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__20_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__21 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__21_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__21_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__22 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__22_value;
static const lean_string_object lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__23 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__23_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__23_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__24 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__24_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 9, .m_data = "term_≼__1"};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c____1___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c____1___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__11_value),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(255, 234, 58, 112, 215, 66, 165, 177)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c____1___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c____1___closed__1_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c____1___closed__1_value),((lean_object*)(((size_t)(50) << 1) | 1)),((lean_object*)(((size_t)(50) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__21_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c____1___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c____1___closed__2_value;
LEAN_EXPORT const lean_object* lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c____1 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c____1___closed__2_value;
static const lean_string_object lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "s"};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1__1___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1__1___closed__0_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1__1___closed__1;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(203, 235, 49, 11, 232, 138, 137, 74)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1__1___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1__1___closed__2_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(203, 235, 49, 11, 232, 138, 137, 74)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1__1___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1__1___closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1__1___closed__3_value),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__9_value),LEAN_SCALAR_PTR_LITERAL(142, 140, 34, 115, 116, 115, 217, 196)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1__1___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1__1___closed__4_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1__1___closed__4_value),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(7, 96, 191, 228, 80, 239, 78, 151)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1__1___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1__1___closed__5_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1__1___closed__5_value),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__4_value),LEAN_SCALAR_PTR_LITERAL(92, 116, 32, 124, 103, 130, 151, 178)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1__1___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1__1___closed__6_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1__1___closed__6_value),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(126, 132, 131, 153, 74, 230, 132, 184)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1__1___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1__1___closed__7_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1__1___closed__7_value),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__8_value),LEAN_SCALAR_PTR_LITERAL(185, 215, 208, 223, 155, 252, 197, 244)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1__1___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1__1___closed__8_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1__1___closed__8_value),((lean_object*)(((size_t)(1913905867) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(223, 24, 175, 179, 108, 79, 165, 105)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1__1___closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1__1___closed__9_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1__1___closed__9_value),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__16_value),LEAN_SCALAR_PTR_LITERAL(52, 236, 168, 226, 223, 152, 165, 167)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1__1___closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1__1___closed__10_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1__1___closed__10_value),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__18_value),LEAN_SCALAR_PTR_LITERAL(224, 8, 105, 2, 139, 247, 137, 170)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1__1___closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1__1___closed__11_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1__1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1__1___closed__11_value),((lean_object*)(((size_t)(18) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(72, 137, 206, 31, 171, 31, 218, 22)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1__1___closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1__1___closed__12_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1__1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1__1___closed__12_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1__1___closed__13 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1__1___closed__13_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1__1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1__1___closed__13_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1__1___closed__14 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1__1___closed__14_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_orderedInsert___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_orderedInsert(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_insertionSort___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_insertionSort(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Sort_0__List_filter_match__1_splitter___redArg(uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Sort_0__List_filter_match__1_splitter___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Sort_0__List_filter_match__1_splitter(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Sort_0__List_filter_match__1_splitter___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Sort_0__List_orderedInsert_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Sort_0__List_orderedInsert_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_List_decidableSortedLE___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_decidableSortedLE___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_List_decidableSortedLE(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_decidableSortedLE___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_List_decidableSortedGE___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_decidableSortedGE___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_List_decidableSortedGE___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_decidableSortedGE___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_List_decidableSortedGE(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_decidableSortedGE___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_List_decidableSortedLT___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_decidableSortedLT___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_List_decidableSortedLT(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_decidableSortedLT___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_List_decidableSortedGT___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_decidableSortedGT___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_List_decidableSortedGT(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_decidableSortedGT___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__6(void){
_start:
{
lean_object* v___x_62_; lean_object* v___x_63_; 
v___x_62_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__5));
v___x_63_ = l_String_toRawSubstring_x27(v___x_62_);
return v___x_63_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1(lean_object* v_x_108_, lean_object* v_a_109_, lean_object* v_a_110_){
_start:
{
lean_object* v___x_111_; lean_object* v___x_112_; uint8_t v___x_113_; 
v___x_111_ = lean_unsigned_to_nat(0u);
v___x_112_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c___00__closed__13));
lean_inc(v_x_108_);
v___x_113_ = l_Lean_Syntax_isOfKind(v_x_108_, v___x_112_);
if (v___x_113_ == 0)
{
lean_object* v___x_114_; lean_object* v___x_115_; 
lean_dec(v_x_108_);
v___x_114_ = lean_box(1);
v___x_115_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_115_, 0, v___x_114_);
lean_ctor_set(v___x_115_, 1, v_a_110_);
return v___x_115_;
}
else
{
lean_object* v_quotContext_116_; lean_object* v_currMacroScope_117_; lean_object* v_ref_118_; lean_object* v___x_119_; lean_object* v___x_120_; lean_object* v___x_121_; uint8_t v___x_122_; lean_object* v___x_123_; lean_object* v___x_124_; lean_object* v___x_125_; lean_object* v___x_126_; lean_object* v___x_127_; lean_object* v___x_128_; lean_object* v___x_129_; lean_object* v___x_130_; lean_object* v___x_131_; lean_object* v___x_132_; lean_object* v___x_133_; 
v_quotContext_116_ = lean_ctor_get(v_a_109_, 1);
v_currMacroScope_117_ = lean_ctor_get(v_a_109_, 2);
v_ref_118_ = lean_ctor_get(v_a_109_, 5);
v___x_119_ = l_Lean_Syntax_getArg(v_x_108_, v___x_111_);
v___x_120_ = lean_unsigned_to_nat(2u);
v___x_121_ = l_Lean_Syntax_getArg(v_x_108_, v___x_120_);
lean_dec(v_x_108_);
v___x_122_ = 0;
v___x_123_ = l_Lean_SourceInfo_fromRef(v_ref_118_, v___x_122_);
v___x_124_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__4));
v___x_125_ = lean_obj_once(&lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__6, &lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__6_once, _init_lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__6);
v___x_126_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__7));
lean_inc(v_currMacroScope_117_);
lean_inc(v_quotContext_116_);
v___x_127_ = l_Lean_addMacroScope(v_quotContext_116_, v___x_126_, v_currMacroScope_117_);
v___x_128_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__22));
lean_inc_n(v___x_123_, 2);
v___x_129_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_129_, 0, v___x_123_);
lean_ctor_set(v___x_129_, 1, v___x_125_);
lean_ctor_set(v___x_129_, 2, v___x_127_);
lean_ctor_set(v___x_129_, 3, v___x_128_);
v___x_130_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__24));
v___x_131_ = l_Lean_Syntax_node2(v___x_123_, v___x_130_, v___x_119_, v___x_121_);
v___x_132_ = l_Lean_Syntax_node2(v___x_123_, v___x_124_, v___x_129_, v___x_131_);
v___x_133_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_133_, 0, v___x_132_);
lean_ctor_set(v___x_133_, 1, v_a_110_);
return v___x_133_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___boxed(lean_object* v_x_134_, lean_object* v_a_135_, lean_object* v_a_136_){
_start:
{
lean_object* v_res_137_; 
v_res_137_ = lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1(v_x_134_, v_a_135_, v_a_136_);
lean_dec_ref(v_a_135_);
return v_res_137_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1__1___closed__1(void){
_start:
{
lean_object* v___x_148_; lean_object* v___x_149_; 
v___x_148_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1__1___closed__0));
v___x_149_ = l_String_toRawSubstring_x27(v___x_148_);
return v___x_149_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1__1(lean_object* v_x_188_, lean_object* v_a_189_, lean_object* v_a_190_){
_start:
{
lean_object* v___x_191_; lean_object* v___x_192_; uint8_t v___x_193_; 
v___x_191_ = lean_unsigned_to_nat(0u);
v___x_192_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_List_Sort_0__List_term___u227c____1___closed__1));
lean_inc(v_x_188_);
v___x_193_ = l_Lean_Syntax_isOfKind(v_x_188_, v___x_192_);
if (v___x_193_ == 0)
{
lean_object* v___x_194_; lean_object* v___x_195_; 
lean_dec(v_x_188_);
v___x_194_ = lean_box(1);
v___x_195_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_195_, 0, v___x_194_);
lean_ctor_set(v___x_195_, 1, v_a_190_);
return v___x_195_;
}
else
{
lean_object* v_quotContext_196_; lean_object* v_currMacroScope_197_; lean_object* v_ref_198_; lean_object* v___x_199_; lean_object* v___x_200_; lean_object* v___x_201_; uint8_t v___x_202_; lean_object* v___x_203_; lean_object* v___x_204_; lean_object* v___x_205_; lean_object* v___x_206_; lean_object* v___x_207_; lean_object* v___x_208_; lean_object* v___x_209_; lean_object* v___x_210_; lean_object* v___x_211_; lean_object* v___x_212_; lean_object* v___x_213_; 
v_quotContext_196_ = lean_ctor_get(v_a_189_, 1);
v_currMacroScope_197_ = lean_ctor_get(v_a_189_, 2);
v_ref_198_ = lean_ctor_get(v_a_189_, 5);
v___x_199_ = l_Lean_Syntax_getArg(v_x_188_, v___x_191_);
v___x_200_ = lean_unsigned_to_nat(2u);
v___x_201_ = l_Lean_Syntax_getArg(v_x_188_, v___x_200_);
lean_dec(v_x_188_);
v___x_202_ = 0;
v___x_203_ = l_Lean_SourceInfo_fromRef(v_ref_198_, v___x_202_);
v___x_204_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__4));
v___x_205_ = lean_obj_once(&lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1__1___closed__1, &lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1__1___closed__1_once, _init_lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1__1___closed__1);
v___x_206_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1__1___closed__2));
lean_inc(v_currMacroScope_197_);
lean_inc(v_quotContext_196_);
v___x_207_ = l_Lean_addMacroScope(v_quotContext_196_, v___x_206_, v_currMacroScope_197_);
v___x_208_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1__1___closed__14));
lean_inc_n(v___x_203_, 2);
v___x_209_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_209_, 0, v___x_203_);
lean_ctor_set(v___x_209_, 1, v___x_205_);
lean_ctor_set(v___x_209_, 2, v___x_207_);
lean_ctor_set(v___x_209_, 3, v___x_208_);
v___x_210_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1___closed__24));
v___x_211_ = l_Lean_Syntax_node2(v___x_203_, v___x_210_, v___x_199_, v___x_201_);
v___x_212_ = l_Lean_Syntax_node2(v___x_203_, v___x_204_, v___x_209_, v___x_211_);
v___x_213_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_213_, 0, v___x_212_);
lean_ctor_set(v___x_213_, 1, v_a_190_);
return v___x_213_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1__1___boxed(lean_object* v_x_214_, lean_object* v_a_215_, lean_object* v_a_216_){
_start:
{
lean_object* v_res_217_; 
v_res_217_ = lp_mathlib___private_Mathlib_Data_List_Sort_0__List___aux__Mathlib__Data__List__Sort______macroRules____private__Mathlib__Data__List__Sort__0__List__term___u227c____1__1(v_x_214_, v_a_215_, v_a_216_);
lean_dec_ref(v_a_215_);
return v_res_217_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_orderedInsert___redArg(lean_object* v_inst_218_, lean_object* v_a_219_, lean_object* v_x_220_){
_start:
{
if (lean_obj_tag(v_x_220_) == 0)
{
lean_object* v___x_221_; 
lean_dec_ref(v_inst_218_);
v___x_221_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_221_, 0, v_a_219_);
lean_ctor_set(v___x_221_, 1, v_x_220_);
return v___x_221_;
}
else
{
lean_object* v_head_222_; lean_object* v_tail_223_; lean_object* v___x_224_; uint8_t v___x_225_; 
v_head_222_ = lean_ctor_get(v_x_220_, 0);
v_tail_223_ = lean_ctor_get(v_x_220_, 1);
lean_inc_ref(v_inst_218_);
lean_inc(v_head_222_);
lean_inc(v_a_219_);
v___x_224_ = lean_apply_2(v_inst_218_, v_a_219_, v_head_222_);
v___x_225_ = lean_unbox(v___x_224_);
if (v___x_225_ == 0)
{
lean_object* v___x_227_; uint8_t v_isShared_228_; uint8_t v_isSharedCheck_233_; 
lean_inc(v_tail_223_);
lean_inc(v_head_222_);
v_isSharedCheck_233_ = !lean_is_exclusive(v_x_220_);
if (v_isSharedCheck_233_ == 0)
{
lean_object* v_unused_234_; lean_object* v_unused_235_; 
v_unused_234_ = lean_ctor_get(v_x_220_, 1);
lean_dec(v_unused_234_);
v_unused_235_ = lean_ctor_get(v_x_220_, 0);
lean_dec(v_unused_235_);
v___x_227_ = v_x_220_;
v_isShared_228_ = v_isSharedCheck_233_;
goto v_resetjp_226_;
}
else
{
lean_dec(v_x_220_);
v___x_227_ = lean_box(0);
v_isShared_228_ = v_isSharedCheck_233_;
goto v_resetjp_226_;
}
v_resetjp_226_:
{
lean_object* v___x_229_; lean_object* v___x_231_; 
v___x_229_ = lp_mathlib_List_orderedInsert___redArg(v_inst_218_, v_a_219_, v_tail_223_);
if (v_isShared_228_ == 0)
{
lean_ctor_set(v___x_227_, 1, v___x_229_);
v___x_231_ = v___x_227_;
goto v_reusejp_230_;
}
else
{
lean_object* v_reuseFailAlloc_232_; 
v_reuseFailAlloc_232_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_232_, 0, v_head_222_);
lean_ctor_set(v_reuseFailAlloc_232_, 1, v___x_229_);
v___x_231_ = v_reuseFailAlloc_232_;
goto v_reusejp_230_;
}
v_reusejp_230_:
{
return v___x_231_;
}
}
}
else
{
lean_object* v___x_236_; 
lean_dec_ref(v_inst_218_);
v___x_236_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_236_, 0, v_a_219_);
lean_ctor_set(v___x_236_, 1, v_x_220_);
return v___x_236_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_orderedInsert(lean_object* v_00_u03b1_237_, lean_object* v_r_238_, lean_object* v_inst_239_, lean_object* v_a_240_, lean_object* v_x_241_){
_start:
{
lean_object* v___x_242_; 
v___x_242_ = lp_mathlib_List_orderedInsert___redArg(v_inst_239_, v_a_240_, v_x_241_);
return v___x_242_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_insertionSort___redArg(lean_object* v_inst_243_, lean_object* v_l_244_){
_start:
{
lean_object* v___x_245_; lean_object* v___x_246_; lean_object* v___x_247_; 
v___x_245_ = lean_alloc_closure((void*)(lp_mathlib_List_orderedInsert), 5, 3);
lean_closure_set(v___x_245_, 0, lean_box(0));
lean_closure_set(v___x_245_, 1, lean_box(0));
lean_closure_set(v___x_245_, 2, v_inst_243_);
v___x_246_ = lean_box(0);
v___x_247_ = l_List_foldrTR___redArg(v___x_245_, v___x_246_, v_l_244_);
return v___x_247_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_insertionSort(lean_object* v_00_u03b1_248_, lean_object* v_r_249_, lean_object* v_inst_250_, lean_object* v_l_251_){
_start:
{
lean_object* v___x_252_; 
v___x_252_ = lp_mathlib_List_insertionSort___redArg(v_inst_250_, v_l_251_);
return v___x_252_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Sort_0__List_filter_match__1_splitter___redArg(uint8_t v_x_253_, lean_object* v_h__1_254_, lean_object* v_h__2_255_){
_start:
{
if (v_x_253_ == 0)
{
lean_object* v___x_256_; lean_object* v___x_257_; 
lean_dec(v_h__1_254_);
v___x_256_ = lean_box(0);
v___x_257_ = lean_apply_1(v_h__2_255_, v___x_256_);
return v___x_257_;
}
else
{
lean_object* v___x_258_; lean_object* v___x_259_; 
lean_dec(v_h__2_255_);
v___x_258_ = lean_box(0);
v___x_259_ = lean_apply_1(v_h__1_254_, v___x_258_);
return v___x_259_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Sort_0__List_filter_match__1_splitter___redArg___boxed(lean_object* v_x_260_, lean_object* v_h__1_261_, lean_object* v_h__2_262_){
_start:
{
uint8_t v_x_24__boxed_263_; lean_object* v_res_264_; 
v_x_24__boxed_263_ = lean_unbox(v_x_260_);
v_res_264_ = lp_mathlib___private_Mathlib_Data_List_Sort_0__List_filter_match__1_splitter___redArg(v_x_24__boxed_263_, v_h__1_261_, v_h__2_262_);
return v_res_264_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Sort_0__List_filter_match__1_splitter(lean_object* v_motive_265_, uint8_t v_x_266_, lean_object* v_h__1_267_, lean_object* v_h__2_268_){
_start:
{
if (v_x_266_ == 0)
{
lean_object* v___x_269_; lean_object* v___x_270_; 
lean_dec(v_h__1_267_);
v___x_269_ = lean_box(0);
v___x_270_ = lean_apply_1(v_h__2_268_, v___x_269_);
return v___x_270_;
}
else
{
lean_object* v___x_271_; lean_object* v___x_272_; 
lean_dec(v_h__2_268_);
v___x_271_ = lean_box(0);
v___x_272_ = lean_apply_1(v_h__1_267_, v___x_271_);
return v___x_272_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Sort_0__List_filter_match__1_splitter___boxed(lean_object* v_motive_273_, lean_object* v_x_274_, lean_object* v_h__1_275_, lean_object* v_h__2_276_){
_start:
{
uint8_t v_x_35__boxed_277_; lean_object* v_res_278_; 
v_x_35__boxed_277_ = lean_unbox(v_x_274_);
v_res_278_ = lp_mathlib___private_Mathlib_Data_List_Sort_0__List_filter_match__1_splitter(v_motive_273_, v_x_35__boxed_277_, v_h__1_275_, v_h__2_276_);
return v_res_278_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Sort_0__List_orderedInsert_match__1_splitter___redArg(lean_object* v_x_279_, lean_object* v_h__1_280_, lean_object* v_h__2_281_){
_start:
{
if (lean_obj_tag(v_x_279_) == 0)
{
lean_object* v___x_282_; lean_object* v___x_283_; 
lean_dec(v_h__2_281_);
v___x_282_ = lean_box(0);
v___x_283_ = lean_apply_1(v_h__1_280_, v___x_282_);
return v___x_283_;
}
else
{
lean_object* v_head_284_; lean_object* v_tail_285_; lean_object* v___x_286_; 
lean_dec(v_h__1_280_);
v_head_284_ = lean_ctor_get(v_x_279_, 0);
lean_inc(v_head_284_);
v_tail_285_ = lean_ctor_get(v_x_279_, 1);
lean_inc(v_tail_285_);
lean_dec_ref_known(v_x_279_, 2);
v___x_286_ = lean_apply_2(v_h__2_281_, v_head_284_, v_tail_285_);
return v___x_286_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Sort_0__List_orderedInsert_match__1_splitter(lean_object* v_00_u03b1_287_, lean_object* v_motive_288_, lean_object* v_x_289_, lean_object* v_h__1_290_, lean_object* v_h__2_291_){
_start:
{
if (lean_obj_tag(v_x_289_) == 0)
{
lean_object* v___x_292_; lean_object* v___x_293_; 
lean_dec(v_h__2_291_);
v___x_292_ = lean_box(0);
v___x_293_ = lean_apply_1(v_h__1_290_, v___x_292_);
return v___x_293_;
}
else
{
lean_object* v_head_294_; lean_object* v_tail_295_; lean_object* v___x_296_; 
lean_dec(v_h__1_290_);
v_head_294_ = lean_ctor_get(v_x_289_, 0);
lean_inc(v_head_294_);
v_tail_295_ = lean_ctor_get(v_x_289_, 1);
lean_inc(v_tail_295_);
lean_dec_ref_known(v_x_289_, 2);
v___x_296_ = lean_apply_2(v_h__2_291_, v_head_294_, v_tail_295_);
return v___x_296_;
}
}
}
LEAN_EXPORT uint8_t lp_mathlib_List_decidableSortedLE___redArg(lean_object* v_inst_297_, lean_object* v_x_298_){
_start:
{
uint8_t v___x_299_; 
v___x_299_ = lp_batteries_List_instDecidableIsChainOfDecidableRel___redArg(v_inst_297_, v_x_298_);
return v___x_299_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_decidableSortedLE___redArg___boxed(lean_object* v_inst_300_, lean_object* v_x_301_){
_start:
{
uint8_t v_res_302_; lean_object* v_r_303_; 
v_res_302_ = lp_mathlib_List_decidableSortedLE___redArg(v_inst_300_, v_x_301_);
v_r_303_ = lean_box(v_res_302_);
return v_r_303_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_List_decidableSortedLE(lean_object* v_00_u03b1_304_, lean_object* v_inst_305_, lean_object* v_inst_306_, lean_object* v_x_307_){
_start:
{
uint8_t v___x_308_; 
v___x_308_ = lp_batteries_List_instDecidableIsChainOfDecidableRel___redArg(v_inst_306_, v_x_307_);
return v___x_308_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_decidableSortedLE___boxed(lean_object* v_00_u03b1_309_, lean_object* v_inst_310_, lean_object* v_inst_311_, lean_object* v_x_312_){
_start:
{
uint8_t v_res_313_; lean_object* v_r_314_; 
v_res_313_ = lp_mathlib_List_decidableSortedLE(v_00_u03b1_309_, v_inst_310_, v_inst_311_, v_x_312_);
lean_dec_ref(v_inst_310_);
v_r_314_ = lean_box(v_res_313_);
return v_r_314_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_List_decidableSortedGE___redArg___lam__0(lean_object* v_inst_315_, lean_object* v_a_316_, lean_object* v_b_317_){
_start:
{
lean_object* v___x_318_; uint8_t v___x_319_; 
v___x_318_ = lean_apply_2(v_inst_315_, v_b_317_, v_a_316_);
v___x_319_ = lean_unbox(v___x_318_);
return v___x_319_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_decidableSortedGE___redArg___lam__0___boxed(lean_object* v_inst_320_, lean_object* v_a_321_, lean_object* v_b_322_){
_start:
{
uint8_t v_res_323_; lean_object* v_r_324_; 
v_res_323_ = lp_mathlib_List_decidableSortedGE___redArg___lam__0(v_inst_320_, v_a_321_, v_b_322_);
v_r_324_ = lean_box(v_res_323_);
return v_r_324_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_List_decidableSortedGE___redArg(lean_object* v_inst_325_, lean_object* v_x_326_){
_start:
{
lean_object* v___f_327_; uint8_t v___x_328_; 
v___f_327_ = lean_alloc_closure((void*)(lp_mathlib_List_decidableSortedGE___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_327_, 0, v_inst_325_);
v___x_328_ = lp_batteries_List_instDecidableIsChainOfDecidableRel___redArg(v___f_327_, v_x_326_);
return v___x_328_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_decidableSortedGE___redArg___boxed(lean_object* v_inst_329_, lean_object* v_x_330_){
_start:
{
uint8_t v_res_331_; lean_object* v_r_332_; 
v_res_331_ = lp_mathlib_List_decidableSortedGE___redArg(v_inst_329_, v_x_330_);
v_r_332_ = lean_box(v_res_331_);
return v_r_332_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_List_decidableSortedGE(lean_object* v_00_u03b1_333_, lean_object* v_inst_334_, lean_object* v_inst_335_, lean_object* v_x_336_){
_start:
{
uint8_t v___x_337_; 
v___x_337_ = lp_mathlib_List_decidableSortedGE___redArg(v_inst_335_, v_x_336_);
return v___x_337_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_decidableSortedGE___boxed(lean_object* v_00_u03b1_338_, lean_object* v_inst_339_, lean_object* v_inst_340_, lean_object* v_x_341_){
_start:
{
uint8_t v_res_342_; lean_object* v_r_343_; 
v_res_342_ = lp_mathlib_List_decidableSortedGE(v_00_u03b1_338_, v_inst_339_, v_inst_340_, v_x_341_);
lean_dec_ref(v_inst_339_);
v_r_343_ = lean_box(v_res_342_);
return v_r_343_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_List_decidableSortedLT___redArg(lean_object* v_inst_344_, lean_object* v_x_345_){
_start:
{
uint8_t v___x_346_; 
v___x_346_ = lp_batteries_List_instDecidableIsChainOfDecidableRel___redArg(v_inst_344_, v_x_345_);
return v___x_346_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_decidableSortedLT___redArg___boxed(lean_object* v_inst_347_, lean_object* v_x_348_){
_start:
{
uint8_t v_res_349_; lean_object* v_r_350_; 
v_res_349_ = lp_mathlib_List_decidableSortedLT___redArg(v_inst_347_, v_x_348_);
v_r_350_ = lean_box(v_res_349_);
return v_r_350_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_List_decidableSortedLT(lean_object* v_00_u03b1_351_, lean_object* v_inst_352_, lean_object* v_inst_353_, lean_object* v_x_354_){
_start:
{
uint8_t v___x_355_; 
v___x_355_ = lp_batteries_List_instDecidableIsChainOfDecidableRel___redArg(v_inst_353_, v_x_354_);
return v___x_355_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_decidableSortedLT___boxed(lean_object* v_00_u03b1_356_, lean_object* v_inst_357_, lean_object* v_inst_358_, lean_object* v_x_359_){
_start:
{
uint8_t v_res_360_; lean_object* v_r_361_; 
v_res_360_ = lp_mathlib_List_decidableSortedLT(v_00_u03b1_356_, v_inst_357_, v_inst_358_, v_x_359_);
lean_dec_ref(v_inst_357_);
v_r_361_ = lean_box(v_res_360_);
return v_r_361_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_List_decidableSortedGT___redArg(lean_object* v_inst_362_, lean_object* v_x_363_){
_start:
{
lean_object* v___f_364_; uint8_t v___x_365_; 
v___f_364_ = lean_alloc_closure((void*)(lp_mathlib_List_decidableSortedGE___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_364_, 0, v_inst_362_);
v___x_365_ = lp_batteries_List_instDecidableIsChainOfDecidableRel___redArg(v___f_364_, v_x_363_);
return v___x_365_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_decidableSortedGT___redArg___boxed(lean_object* v_inst_366_, lean_object* v_x_367_){
_start:
{
uint8_t v_res_368_; lean_object* v_r_369_; 
v_res_368_ = lp_mathlib_List_decidableSortedGT___redArg(v_inst_366_, v_x_367_);
v_r_369_ = lean_box(v_res_368_);
return v_r_369_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_List_decidableSortedGT(lean_object* v_00_u03b1_370_, lean_object* v_inst_371_, lean_object* v_inst_372_, lean_object* v_x_373_){
_start:
{
uint8_t v___x_374_; 
v___x_374_ = lp_mathlib_List_decidableSortedGT___redArg(v_inst_372_, v_x_373_);
return v___x_374_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_decidableSortedGT___boxed(lean_object* v_00_u03b1_375_, lean_object* v_inst_376_, lean_object* v_inst_377_, lean_object* v_x_378_){
_start:
{
uint8_t v_res_379_; lean_object* v_r_380_; 
v_res_379_ = lp_mathlib_List_decidableSortedGT(v_00_u03b1_375_, v_inst_376_, v_inst_377_, v_x_378_);
lean_dec_ref(v_inst_376_);
v_r_380_ = lean_box(v_res_379_);
return v_r_380_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Data_List_Perm(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_List_OfFn(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_List_Nodup(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Fin_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_List_Sort(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Data_List_Perm(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_List_OfFn(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_List_Nodup(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Fin_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_List_Sort(uint8_t builtin) {
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
lean_object* initialize_batteries_Batteries_Data_List_Perm(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_List_OfFn(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_List_Nodup(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Fin_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_List_Sort(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Data_List_Perm(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_List_OfFn(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_List_Nodup(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Fin_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_List_Sort(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_List_Sort(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_List_Sort(builtin);
}
#ifdef __cplusplus
}
#endif
