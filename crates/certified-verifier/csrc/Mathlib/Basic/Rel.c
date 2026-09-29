// Lean compiler output
// Module: Mathlib.Basic.Rel
// Imports: public import Init public meta import Init public import Mathlib.Data.Set.Prod public import Mathlib.Order.RelIso.Basic public import Mathlib.Order.SetNotation
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
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_SetRel_term___x7e_x5b___x5d___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "SetRel"};
static const lean_object* lp_mathlib_SetRel_term___x7e_x5b___x5d___00__closed__0 = (const lean_object*)&lp_mathlib_SetRel_term___x7e_x5b___x5d___00__closed__0_value;
static const lean_string_object lp_mathlib_SetRel_term___x7e_x5b___x5d___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "term_~[_]_"};
static const lean_object* lp_mathlib_SetRel_term___x7e_x5b___x5d___00__closed__1 = (const lean_object*)&lp_mathlib_SetRel_term___x7e_x5b___x5d___00__closed__1_value;
static const lean_ctor_object lp_mathlib_SetRel_term___x7e_x5b___x5d___00__closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_SetRel_term___x7e_x5b___x5d___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(34, 199, 61, 66, 87, 66, 228, 27)}};
static const lean_ctor_object lp_mathlib_SetRel_term___x7e_x5b___x5d___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_SetRel_term___x7e_x5b___x5d___00__closed__2_value_aux_0),((lean_object*)&lp_mathlib_SetRel_term___x7e_x5b___x5d___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(183, 150, 181, 171, 34, 229, 156, 86)}};
static const lean_object* lp_mathlib_SetRel_term___x7e_x5b___x5d___00__closed__2 = (const lean_object*)&lp_mathlib_SetRel_term___x7e_x5b___x5d___00__closed__2_value;
static const lean_string_object lp_mathlib_SetRel_term___x7e_x5b___x5d___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_SetRel_term___x7e_x5b___x5d___00__closed__3 = (const lean_object*)&lp_mathlib_SetRel_term___x7e_x5b___x5d___00__closed__3_value;
static const lean_ctor_object lp_mathlib_SetRel_term___x7e_x5b___x5d___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_SetRel_term___x7e_x5b___x5d___00__closed__3_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_SetRel_term___x7e_x5b___x5d___00__closed__4 = (const lean_object*)&lp_mathlib_SetRel_term___x7e_x5b___x5d___00__closed__4_value;
static const lean_string_object lp_mathlib_SetRel_term___x7e_x5b___x5d___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = " ~["};
static const lean_object* lp_mathlib_SetRel_term___x7e_x5b___x5d___00__closed__5 = (const lean_object*)&lp_mathlib_SetRel_term___x7e_x5b___x5d___00__closed__5_value;
static const lean_ctor_object lp_mathlib_SetRel_term___x7e_x5b___x5d___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_SetRel_term___x7e_x5b___x5d___00__closed__5_value)}};
static const lean_object* lp_mathlib_SetRel_term___x7e_x5b___x5d___00__closed__6 = (const lean_object*)&lp_mathlib_SetRel_term___x7e_x5b___x5d___00__closed__6_value;
static const lean_string_object lp_mathlib_SetRel_term___x7e_x5b___x5d___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_SetRel_term___x7e_x5b___x5d___00__closed__7 = (const lean_object*)&lp_mathlib_SetRel_term___x7e_x5b___x5d___00__closed__7_value;
static const lean_ctor_object lp_mathlib_SetRel_term___x7e_x5b___x5d___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_SetRel_term___x7e_x5b___x5d___00__closed__7_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_SetRel_term___x7e_x5b___x5d___00__closed__8 = (const lean_object*)&lp_mathlib_SetRel_term___x7e_x5b___x5d___00__closed__8_value;
static const lean_ctor_object lp_mathlib_SetRel_term___x7e_x5b___x5d___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_SetRel_term___x7e_x5b___x5d___00__closed__8_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_SetRel_term___x7e_x5b___x5d___00__closed__9 = (const lean_object*)&lp_mathlib_SetRel_term___x7e_x5b___x5d___00__closed__9_value;
static const lean_ctor_object lp_mathlib_SetRel_term___x7e_x5b___x5d___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_SetRel_term___x7e_x5b___x5d___00__closed__4_value),((lean_object*)&lp_mathlib_SetRel_term___x7e_x5b___x5d___00__closed__6_value),((lean_object*)&lp_mathlib_SetRel_term___x7e_x5b___x5d___00__closed__9_value)}};
static const lean_object* lp_mathlib_SetRel_term___x7e_x5b___x5d___00__closed__10 = (const lean_object*)&lp_mathlib_SetRel_term___x7e_x5b___x5d___00__closed__10_value;
static const lean_string_object lp_mathlib_SetRel_term___x7e_x5b___x5d___00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "] "};
static const lean_object* lp_mathlib_SetRel_term___x7e_x5b___x5d___00__closed__11 = (const lean_object*)&lp_mathlib_SetRel_term___x7e_x5b___x5d___00__closed__11_value;
static const lean_ctor_object lp_mathlib_SetRel_term___x7e_x5b___x5d___00__closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_SetRel_term___x7e_x5b___x5d___00__closed__11_value)}};
static const lean_object* lp_mathlib_SetRel_term___x7e_x5b___x5d___00__closed__12 = (const lean_object*)&lp_mathlib_SetRel_term___x7e_x5b___x5d___00__closed__12_value;
static const lean_ctor_object lp_mathlib_SetRel_term___x7e_x5b___x5d___00__closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_SetRel_term___x7e_x5b___x5d___00__closed__4_value),((lean_object*)&lp_mathlib_SetRel_term___x7e_x5b___x5d___00__closed__10_value),((lean_object*)&lp_mathlib_SetRel_term___x7e_x5b___x5d___00__closed__12_value)}};
static const lean_object* lp_mathlib_SetRel_term___x7e_x5b___x5d___00__closed__13 = (const lean_object*)&lp_mathlib_SetRel_term___x7e_x5b___x5d___00__closed__13_value;
static const lean_ctor_object lp_mathlib_SetRel_term___x7e_x5b___x5d___00__closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_SetRel_term___x7e_x5b___x5d___00__closed__8_value),((lean_object*)(((size_t)(50) << 1) | 1))}};
static const lean_object* lp_mathlib_SetRel_term___x7e_x5b___x5d___00__closed__14 = (const lean_object*)&lp_mathlib_SetRel_term___x7e_x5b___x5d___00__closed__14_value;
static const lean_ctor_object lp_mathlib_SetRel_term___x7e_x5b___x5d___00__closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_SetRel_term___x7e_x5b___x5d___00__closed__4_value),((lean_object*)&lp_mathlib_SetRel_term___x7e_x5b___x5d___00__closed__13_value),((lean_object*)&lp_mathlib_SetRel_term___x7e_x5b___x5d___00__closed__14_value)}};
static const lean_object* lp_mathlib_SetRel_term___x7e_x5b___x5d___00__closed__15 = (const lean_object*)&lp_mathlib_SetRel_term___x7e_x5b___x5d___00__closed__15_value;
static const lean_ctor_object lp_mathlib_SetRel_term___x7e_x5b___x5d___00__closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_SetRel_term___x7e_x5b___x5d___00__closed__2_value),((lean_object*)(((size_t)(50) << 1) | 1)),((lean_object*)(((size_t)(50) << 1) | 1)),((lean_object*)&lp_mathlib_SetRel_term___x7e_x5b___x5d___00__closed__15_value)}};
static const lean_object* lp_mathlib_SetRel_term___x7e_x5b___x5d___00__closed__16 = (const lean_object*)&lp_mathlib_SetRel_term___x7e_x5b___x5d___00__closed__16_value;
LEAN_EXPORT const lean_object* lp_mathlib_SetRel_term___x7e_x5b___x5d__ = (const lean_object*)&lp_mathlib_SetRel_term___x7e_x5b___x5d___00__closed__16_value;
static const lean_string_object lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 7, .m_data = "term_∈_"};
static const lean_object* lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__0 = (const lean_object*)&lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__0_value;
static const lean_ctor_object lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(145, 149, 102, 29, 65, 152, 113, 144)}};
static const lean_object* lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__1 = (const lean_object*)&lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__1_value;
static const lean_string_object lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__2 = (const lean_object*)&lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__2_value;
static const lean_string_object lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__3 = (const lean_object*)&lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__3_value;
static const lean_string_object lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__4 = (const lean_object*)&lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__4_value;
static const lean_string_object lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "tuple"};
static const lean_object* lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__5 = (const lean_object*)&lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__5_value;
static const lean_ctor_object lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__6_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__6_value_aux_0),((lean_object*)&lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__6_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__6_value_aux_1),((lean_object*)&lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__4_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__6_value_aux_2),((lean_object*)&lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(191, 24, 88, 245, 200, 250, 27, 217)}};
static const lean_object* lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__6 = (const lean_object*)&lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__6_value;
static const lean_string_object lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "hygienicLParen"};
static const lean_object* lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__7 = (const lean_object*)&lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__7_value;
static const lean_ctor_object lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__8_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__8_value_aux_0),((lean_object*)&lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__8_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__8_value_aux_1),((lean_object*)&lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__4_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__8_value_aux_2),((lean_object*)&lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(41, 104, 206, 51, 21, 254, 100, 101)}};
static const lean_object* lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__8 = (const lean_object*)&lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__8_value;
static const lean_string_object lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "("};
static const lean_object* lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__9 = (const lean_object*)&lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__9_value;
static const lean_string_object lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "hygieneInfo"};
static const lean_object* lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__10 = (const lean_object*)&lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__10_value;
static const lean_ctor_object lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__10_value),LEAN_SCALAR_PTR_LITERAL(27, 64, 36, 144, 170, 151, 255, 136)}};
static const lean_object* lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__11 = (const lean_object*)&lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__11_value;
static const lean_string_object lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__12 = (const lean_object*)&lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__12_value;
static lean_once_cell_t lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__13;
static const lean_ctor_object lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_SetRel_term___x7e_x5b___x5d___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(34, 199, 61, 66, 87, 66, 228, 27)}};
static const lean_object* lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__14 = (const lean_object*)&lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__14_value;
static const lean_ctor_object lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__14_value)}};
static const lean_object* lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__15 = (const lean_object*)&lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__15_value;
static const lean_ctor_object lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__15_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__16 = (const lean_object*)&lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__16_value;
static const lean_string_object lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__17 = (const lean_object*)&lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__17_value;
static const lean_ctor_object lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__17_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__18 = (const lean_object*)&lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__18_value;
static const lean_string_object lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ","};
static const lean_object* lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__19 = (const lean_object*)&lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__19_value;
static const lean_string_object lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ")"};
static const lean_object* lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__20 = (const lean_object*)&lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__20_value;
static const lean_string_object lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "∈"};
static const lean_object* lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__21 = (const lean_object*)&lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__21_value;
LEAN_EXPORT lean_object* lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_SetRel_term___u25cb___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 7, .m_data = "term_○_"};
static const lean_object* lp_mathlib_SetRel_term___u25cb___00__closed__0 = (const lean_object*)&lp_mathlib_SetRel_term___u25cb___00__closed__0_value;
static const lean_ctor_object lp_mathlib_SetRel_term___u25cb___00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_SetRel_term___x7e_x5b___x5d___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(34, 199, 61, 66, 87, 66, 228, 27)}};
static const lean_ctor_object lp_mathlib_SetRel_term___u25cb___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_SetRel_term___u25cb___00__closed__1_value_aux_0),((lean_object*)&lp_mathlib_SetRel_term___u25cb___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(14, 33, 114, 184, 174, 144, 104, 72)}};
static const lean_object* lp_mathlib_SetRel_term___u25cb___00__closed__1 = (const lean_object*)&lp_mathlib_SetRel_term___u25cb___00__closed__1_value;
static const lean_string_object lp_mathlib_SetRel_term___u25cb___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 3, .m_data = " ○ "};
static const lean_object* lp_mathlib_SetRel_term___u25cb___00__closed__2 = (const lean_object*)&lp_mathlib_SetRel_term___u25cb___00__closed__2_value;
static const lean_ctor_object lp_mathlib_SetRel_term___u25cb___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_SetRel_term___u25cb___00__closed__2_value)}};
static const lean_object* lp_mathlib_SetRel_term___u25cb___00__closed__3 = (const lean_object*)&lp_mathlib_SetRel_term___u25cb___00__closed__3_value;
static const lean_ctor_object lp_mathlib_SetRel_term___u25cb___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_SetRel_term___x7e_x5b___x5d___00__closed__8_value),((lean_object*)(((size_t)(63) << 1) | 1))}};
static const lean_object* lp_mathlib_SetRel_term___u25cb___00__closed__4 = (const lean_object*)&lp_mathlib_SetRel_term___u25cb___00__closed__4_value;
static const lean_ctor_object lp_mathlib_SetRel_term___u25cb___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_SetRel_term___x7e_x5b___x5d___00__closed__4_value),((lean_object*)&lp_mathlib_SetRel_term___u25cb___00__closed__3_value),((lean_object*)&lp_mathlib_SetRel_term___u25cb___00__closed__4_value)}};
static const lean_object* lp_mathlib_SetRel_term___u25cb___00__closed__5 = (const lean_object*)&lp_mathlib_SetRel_term___u25cb___00__closed__5_value;
static const lean_ctor_object lp_mathlib_SetRel_term___u25cb___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_SetRel_term___u25cb___00__closed__1_value),((lean_object*)(((size_t)(62) << 1) | 1)),((lean_object*)(((size_t)(62) << 1) | 1)),((lean_object*)&lp_mathlib_SetRel_term___u25cb___00__closed__5_value)}};
static const lean_object* lp_mathlib_SetRel_term___u25cb___00__closed__6 = (const lean_object*)&lp_mathlib_SetRel_term___u25cb___00__closed__6_value;
LEAN_EXPORT const lean_object* lp_mathlib_SetRel_term___u25cb__ = (const lean_object*)&lp_mathlib_SetRel_term___u25cb___00__closed__6_value;
static const lean_string_object lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___u25cb____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___u25cb____1___closed__0 = (const lean_object*)&lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___u25cb____1___closed__0_value;
static const lean_ctor_object lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___u25cb____1___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___u25cb____1___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___u25cb____1___closed__1_value_aux_0),((lean_object*)&lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___u25cb____1___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___u25cb____1___closed__1_value_aux_1),((lean_object*)&lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__4_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___u25cb____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___u25cb____1___closed__1_value_aux_2),((lean_object*)&lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___u25cb____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___u25cb____1___closed__1 = (const lean_object*)&lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___u25cb____1___closed__1_value;
static const lean_string_object lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___u25cb____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "comp"};
static const lean_object* lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___u25cb____1___closed__2 = (const lean_object*)&lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___u25cb____1___closed__2_value;
static lean_once_cell_t lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___u25cb____1___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___u25cb____1___closed__3;
static const lean_ctor_object lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___u25cb____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___u25cb____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(228, 239, 123, 62, 2, 27, 64, 57)}};
static const lean_object* lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___u25cb____1___closed__4 = (const lean_object*)&lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___u25cb____1___closed__4_value;
static const lean_ctor_object lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___u25cb____1___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_SetRel_term___x7e_x5b___x5d___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(34, 199, 61, 66, 87, 66, 228, 27)}};
static const lean_ctor_object lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___u25cb____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___u25cb____1___closed__5_value_aux_0),((lean_object*)&lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___u25cb____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(81, 2, 86, 135, 112, 197, 186, 118)}};
static const lean_object* lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___u25cb____1___closed__5 = (const lean_object*)&lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___u25cb____1___closed__5_value;
static const lean_ctor_object lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___u25cb____1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___u25cb____1___closed__5_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___u25cb____1___closed__6 = (const lean_object*)&lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___u25cb____1___closed__6_value;
static const lean_ctor_object lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___u25cb____1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___u25cb____1___closed__5_value)}};
static const lean_object* lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___u25cb____1___closed__7 = (const lean_object*)&lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___u25cb____1___closed__7_value;
static const lean_ctor_object lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___u25cb____1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___u25cb____1___closed__7_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___u25cb____1___closed__8 = (const lean_object*)&lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___u25cb____1___closed__8_value;
static const lean_ctor_object lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___u25cb____1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___u25cb____1___closed__6_value),((lean_object*)&lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___u25cb____1___closed__8_value)}};
static const lean_object* lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___u25cb____1___closed__9 = (const lean_object*)&lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___u25cb____1___closed__9_value;
LEAN_EXPORT lean_object* lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___u25cb____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___u25cb____1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______unexpand__SetRel__comp__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______unexpand__SetRel__comp__1___closed__0 = (const lean_object*)&lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______unexpand__SetRel__comp__1___closed__0_value;
static const lean_ctor_object lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______unexpand__SetRel__comp__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______unexpand__SetRel__comp__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______unexpand__SetRel__comp__1___closed__1 = (const lean_object*)&lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______unexpand__SetRel__comp__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______unexpand__SetRel__comp__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______unexpand__SetRel__comp__1___boxed(lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__13(void){
_start:
{
lean_object* v___x_64_; lean_object* v___x_65_; 
v___x_64_ = ((lean_object*)(lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__12));
v___x_65_ = l_String_toRawSubstring_x27(v___x_64_);
return v___x_65_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1(lean_object* v_x_79_, lean_object* v_a_80_, lean_object* v_a_81_){
_start:
{
lean_object* v___x_82_; uint8_t v___x_83_; 
v___x_82_ = ((lean_object*)(lp_mathlib_SetRel_term___x7e_x5b___x5d___00__closed__2));
lean_inc(v_x_79_);
v___x_83_ = l_Lean_Syntax_isOfKind(v_x_79_, v___x_82_);
if (v___x_83_ == 0)
{
lean_object* v___x_84_; lean_object* v___x_85_; 
lean_dec(v_x_79_);
v___x_84_ = lean_box(1);
v___x_85_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_85_, 0, v___x_84_);
lean_ctor_set(v___x_85_, 1, v_a_81_);
return v___x_85_;
}
else
{
lean_object* v_quotContext_86_; lean_object* v_currMacroScope_87_; lean_object* v_ref_88_; lean_object* v___x_89_; lean_object* v___x_90_; lean_object* v___x_91_; lean_object* v___x_92_; lean_object* v___x_93_; lean_object* v___x_94_; uint8_t v___x_95_; lean_object* v___x_96_; lean_object* v___x_97_; lean_object* v___x_98_; lean_object* v___x_99_; lean_object* v___x_100_; lean_object* v___x_101_; lean_object* v___x_102_; lean_object* v___x_103_; lean_object* v___x_104_; lean_object* v___x_105_; lean_object* v___x_106_; lean_object* v___x_107_; lean_object* v___x_108_; lean_object* v___x_109_; lean_object* v___x_110_; lean_object* v___x_111_; lean_object* v___x_112_; lean_object* v___x_113_; lean_object* v___x_114_; lean_object* v___x_115_; lean_object* v___x_116_; lean_object* v___x_117_; lean_object* v___x_118_; lean_object* v___x_119_; lean_object* v___x_120_; lean_object* v___x_121_; 
v_quotContext_86_ = lean_ctor_get(v_a_80_, 1);
v_currMacroScope_87_ = lean_ctor_get(v_a_80_, 2);
v_ref_88_ = lean_ctor_get(v_a_80_, 5);
v___x_89_ = lean_unsigned_to_nat(0u);
v___x_90_ = l_Lean_Syntax_getArg(v_x_79_, v___x_89_);
v___x_91_ = lean_unsigned_to_nat(2u);
v___x_92_ = l_Lean_Syntax_getArg(v_x_79_, v___x_91_);
v___x_93_ = lean_unsigned_to_nat(4u);
v___x_94_ = l_Lean_Syntax_getArg(v_x_79_, v___x_93_);
lean_dec(v_x_79_);
v___x_95_ = 0;
v___x_96_ = l_Lean_SourceInfo_fromRef(v_ref_88_, v___x_95_);
v___x_97_ = ((lean_object*)(lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__1));
v___x_98_ = ((lean_object*)(lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__6));
v___x_99_ = ((lean_object*)(lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__8));
v___x_100_ = ((lean_object*)(lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__9));
lean_inc_n(v___x_96_, 10);
v___x_101_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_101_, 0, v___x_96_);
lean_ctor_set(v___x_101_, 1, v___x_100_);
v___x_102_ = ((lean_object*)(lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__11));
v___x_103_ = lean_obj_once(&lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__13, &lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__13_once, _init_lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__13);
v___x_104_ = lean_box(0);
lean_inc(v_currMacroScope_87_);
lean_inc(v_quotContext_86_);
v___x_105_ = l_Lean_addMacroScope(v_quotContext_86_, v___x_104_, v_currMacroScope_87_);
v___x_106_ = ((lean_object*)(lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__16));
v___x_107_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_107_, 0, v___x_96_);
lean_ctor_set(v___x_107_, 1, v___x_103_);
lean_ctor_set(v___x_107_, 2, v___x_105_);
lean_ctor_set(v___x_107_, 3, v___x_106_);
v___x_108_ = l_Lean_Syntax_node1(v___x_96_, v___x_102_, v___x_107_);
v___x_109_ = l_Lean_Syntax_node2(v___x_96_, v___x_99_, v___x_101_, v___x_108_);
v___x_110_ = ((lean_object*)(lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__18));
v___x_111_ = ((lean_object*)(lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__19));
v___x_112_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_112_, 0, v___x_96_);
lean_ctor_set(v___x_112_, 1, v___x_111_);
v___x_113_ = l_Lean_Syntax_node1(v___x_96_, v___x_110_, v___x_94_);
v___x_114_ = l_Lean_Syntax_node3(v___x_96_, v___x_110_, v___x_90_, v___x_112_, v___x_113_);
v___x_115_ = ((lean_object*)(lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__20));
v___x_116_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_116_, 0, v___x_96_);
lean_ctor_set(v___x_116_, 1, v___x_115_);
v___x_117_ = l_Lean_Syntax_node3(v___x_96_, v___x_98_, v___x_109_, v___x_114_, v___x_116_);
v___x_118_ = ((lean_object*)(lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__21));
v___x_119_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_119_, 0, v___x_96_);
lean_ctor_set(v___x_119_, 1, v___x_118_);
v___x_120_ = l_Lean_Syntax_node3(v___x_96_, v___x_97_, v___x_117_, v___x_119_, v___x_92_);
v___x_121_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_121_, 0, v___x_120_);
lean_ctor_set(v___x_121_, 1, v_a_81_);
return v___x_121_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___boxed(lean_object* v_x_122_, lean_object* v_a_123_, lean_object* v_a_124_){
_start:
{
lean_object* v_res_125_; 
v_res_125_ = lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1(v_x_122_, v_a_123_, v_a_124_);
lean_dec_ref(v_a_123_);
return v_res_125_;
}
}
static lean_object* _init_lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___u25cb____1___closed__3(void){
_start:
{
lean_object* v___x_152_; lean_object* v___x_153_; 
v___x_152_ = ((lean_object*)(lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___u25cb____1___closed__2));
v___x_153_ = l_String_toRawSubstring_x27(v___x_152_);
return v___x_153_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___u25cb____1(lean_object* v_x_170_, lean_object* v_a_171_, lean_object* v_a_172_){
_start:
{
lean_object* v___x_173_; uint8_t v___x_174_; 
v___x_173_ = ((lean_object*)(lp_mathlib_SetRel_term___u25cb___00__closed__1));
lean_inc(v_x_170_);
v___x_174_ = l_Lean_Syntax_isOfKind(v_x_170_, v___x_173_);
if (v___x_174_ == 0)
{
lean_object* v___x_175_; lean_object* v___x_176_; 
lean_dec(v_x_170_);
v___x_175_ = lean_box(1);
v___x_176_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_176_, 0, v___x_175_);
lean_ctor_set(v___x_176_, 1, v_a_172_);
return v___x_176_;
}
else
{
lean_object* v_quotContext_177_; lean_object* v_currMacroScope_178_; lean_object* v_ref_179_; lean_object* v___x_180_; lean_object* v___x_181_; lean_object* v___x_182_; lean_object* v___x_183_; uint8_t v___x_184_; lean_object* v___x_185_; lean_object* v___x_186_; lean_object* v___x_187_; lean_object* v___x_188_; lean_object* v___x_189_; lean_object* v___x_190_; lean_object* v___x_191_; lean_object* v___x_192_; lean_object* v___x_193_; lean_object* v___x_194_; lean_object* v___x_195_; 
v_quotContext_177_ = lean_ctor_get(v_a_171_, 1);
v_currMacroScope_178_ = lean_ctor_get(v_a_171_, 2);
v_ref_179_ = lean_ctor_get(v_a_171_, 5);
v___x_180_ = lean_unsigned_to_nat(0u);
v___x_181_ = l_Lean_Syntax_getArg(v_x_170_, v___x_180_);
v___x_182_ = lean_unsigned_to_nat(2u);
v___x_183_ = l_Lean_Syntax_getArg(v_x_170_, v___x_182_);
lean_dec(v_x_170_);
v___x_184_ = 0;
v___x_185_ = l_Lean_SourceInfo_fromRef(v_ref_179_, v___x_184_);
v___x_186_ = ((lean_object*)(lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___u25cb____1___closed__1));
v___x_187_ = lean_obj_once(&lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___u25cb____1___closed__3, &lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___u25cb____1___closed__3_once, _init_lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___u25cb____1___closed__3);
v___x_188_ = ((lean_object*)(lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___u25cb____1___closed__4));
lean_inc(v_currMacroScope_178_);
lean_inc(v_quotContext_177_);
v___x_189_ = l_Lean_addMacroScope(v_quotContext_177_, v___x_188_, v_currMacroScope_178_);
v___x_190_ = ((lean_object*)(lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___u25cb____1___closed__9));
lean_inc_n(v___x_185_, 2);
v___x_191_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_191_, 0, v___x_185_);
lean_ctor_set(v___x_191_, 1, v___x_187_);
lean_ctor_set(v___x_191_, 2, v___x_189_);
lean_ctor_set(v___x_191_, 3, v___x_190_);
v___x_192_ = ((lean_object*)(lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___x7e_x5b___x5d____1___closed__18));
v___x_193_ = l_Lean_Syntax_node2(v___x_185_, v___x_192_, v___x_181_, v___x_183_);
v___x_194_ = l_Lean_Syntax_node2(v___x_185_, v___x_186_, v___x_191_, v___x_193_);
v___x_195_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_195_, 0, v___x_194_);
lean_ctor_set(v___x_195_, 1, v_a_172_);
return v___x_195_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___u25cb____1___boxed(lean_object* v_x_196_, lean_object* v_a_197_, lean_object* v_a_198_){
_start:
{
lean_object* v_res_199_; 
v_res_199_ = lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___u25cb____1(v_x_196_, v_a_197_, v_a_198_);
lean_dec_ref(v_a_197_);
return v_res_199_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______unexpand__SetRel__comp__1(lean_object* v_x_203_, lean_object* v_a_204_, lean_object* v_a_205_){
_start:
{
lean_object* v___x_206_; uint8_t v___x_207_; 
v___x_206_ = ((lean_object*)(lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______macroRules__SetRel__term___u25cb____1___closed__1));
lean_inc(v_x_203_);
v___x_207_ = l_Lean_Syntax_isOfKind(v_x_203_, v___x_206_);
if (v___x_207_ == 0)
{
lean_object* v___x_208_; lean_object* v___x_209_; 
lean_dec(v_x_203_);
v___x_208_ = lean_box(0);
v___x_209_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_209_, 0, v___x_208_);
lean_ctor_set(v___x_209_, 1, v_a_205_);
return v___x_209_;
}
else
{
lean_object* v___x_210_; lean_object* v___x_211_; lean_object* v___x_212_; uint8_t v___x_213_; 
v___x_210_ = lean_unsigned_to_nat(0u);
v___x_211_ = l_Lean_Syntax_getArg(v_x_203_, v___x_210_);
v___x_212_ = ((lean_object*)(lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______unexpand__SetRel__comp__1___closed__1));
lean_inc(v___x_211_);
v___x_213_ = l_Lean_Syntax_isOfKind(v___x_211_, v___x_212_);
if (v___x_213_ == 0)
{
lean_object* v___x_214_; lean_object* v___x_215_; 
lean_dec(v___x_211_);
lean_dec(v_x_203_);
v___x_214_ = lean_box(0);
v___x_215_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_215_, 0, v___x_214_);
lean_ctor_set(v___x_215_, 1, v_a_205_);
return v___x_215_;
}
else
{
lean_object* v___x_216_; lean_object* v___x_217_; lean_object* v___x_218_; uint8_t v___x_219_; 
v___x_216_ = lean_unsigned_to_nat(1u);
v___x_217_ = l_Lean_Syntax_getArg(v_x_203_, v___x_216_);
lean_dec(v_x_203_);
v___x_218_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_217_);
v___x_219_ = l_Lean_Syntax_matchesNull(v___x_217_, v___x_218_);
if (v___x_219_ == 0)
{
lean_object* v___x_220_; lean_object* v___x_221_; 
lean_dec(v___x_217_);
lean_dec(v___x_211_);
v___x_220_ = lean_box(0);
v___x_221_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_221_, 0, v___x_220_);
lean_ctor_set(v___x_221_, 1, v_a_205_);
return v___x_221_;
}
else
{
lean_object* v___x_222_; lean_object* v___x_223_; lean_object* v_ref_224_; uint8_t v___x_225_; lean_object* v___x_226_; lean_object* v___x_227_; lean_object* v___x_228_; lean_object* v___x_229_; lean_object* v___x_230_; lean_object* v___x_231_; 
v___x_222_ = l_Lean_Syntax_getArg(v___x_217_, v___x_210_);
v___x_223_ = l_Lean_Syntax_getArg(v___x_217_, v___x_216_);
lean_dec(v___x_217_);
v_ref_224_ = l_Lean_replaceRef(v___x_211_, v_a_204_);
lean_dec(v___x_211_);
v___x_225_ = 0;
v___x_226_ = l_Lean_SourceInfo_fromRef(v_ref_224_, v___x_225_);
lean_dec(v_ref_224_);
v___x_227_ = ((lean_object*)(lp_mathlib_SetRel_term___u25cb___00__closed__1));
v___x_228_ = ((lean_object*)(lp_mathlib_SetRel_term___u25cb___00__closed__2));
lean_inc(v___x_226_);
v___x_229_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_229_, 0, v___x_226_);
lean_ctor_set(v___x_229_, 1, v___x_228_);
v___x_230_ = l_Lean_Syntax_node3(v___x_226_, v___x_227_, v___x_222_, v___x_229_, v___x_223_);
v___x_231_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_231_, 0, v___x_230_);
lean_ctor_set(v___x_231_, 1, v_a_205_);
return v___x_231_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______unexpand__SetRel__comp__1___boxed(lean_object* v_x_232_, lean_object* v_a_233_, lean_object* v_a_234_){
_start:
{
lean_object* v_res_235_; 
v_res_235_ = lp_mathlib_SetRel___aux__Mathlib__Basic__Rel______unexpand__SetRel__comp__1(v_x_232_, v_a_233_, v_a_234_);
lean_dec(v_a_233_);
return v_res_235_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Set_Prod(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_RelIso_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_SetNotation(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Basic_Rel(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Set_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_RelIso_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_SetNotation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Basic_Rel(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Set_Prod(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_RelIso_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_SetNotation(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Basic_Rel(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_RelIso_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_SetNotation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Basic_Rel(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Basic_Rel(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Basic_Rel(builtin);
}
#ifdef __cplusplus
}
#endif
