// Lean compiler output
// Module: Mathlib.Tactic.ByCases
// Imports: public import Init public meta import Init public import Batteries.Tactic.PermuteGoals public import Mathlib.Tactic.Push
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
lean_object* l_Lean_Name_str___override(lean_object*, lean_object*);
extern lean_object* l_Lean_Parser_Tactic_optConfig;
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* l_Lean_Name_num___override(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_Push_elabPushConfig___redArg(lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_Push_push(uint8_t, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
lean_object* l_Array_mkArray0(lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "ByCases"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "byCases!"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__2_value),LEAN_SCALAR_PTR_LITERAL(123, 244, 251, 147, 115, 12, 96, 220)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__3_value),LEAN_SCALAR_PTR_LITERAL(177, 136, 82, 194, 143, 80, 24, 77)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__5_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "by_cases! "};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__7_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__8_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__9;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "optional"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__10_value),LEAN_SCALAR_PTR_LITERAL(233, 141, 154, 50, 143, 135, 42, 252)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__11_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "atomic"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__12_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__12_value),LEAN_SCALAR_PTR_LITERAL(56, 145, 113, 208, 127, 167, 216, 55)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__13_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__14_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__14_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__15_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__15_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__16_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = " : "};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__17_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__17_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__18_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__16_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__18_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__19_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__13_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__19_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__20 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__20_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__11_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__20_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__21 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__21_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__22_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__22;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__23 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__23_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__23_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__24 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__24_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__24_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__25 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__25_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__26_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__26;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__27_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__27;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ByCases_0__Mathlib_Tactic_ByCases_tacticTry__push__neg__at_____00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "_private"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ByCases_0__Mathlib_Tactic_ByCases_tacticTry__push__neg__at_____00__closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ByCases_0__Mathlib_Tactic_ByCases_tacticTry__push__neg__at_____00__closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ByCases_0__Mathlib_Tactic_ByCases_tacticTry__push__neg__at_____00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ByCases_0__Mathlib_Tactic_ByCases_tacticTry__push__neg__at_____00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(103, 214, 75, 80, 34, 198, 193, 153)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ByCases_0__Mathlib_Tactic_ByCases_tacticTry__push__neg__at_____00__closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ByCases_0__Mathlib_Tactic_ByCases_tacticTry__push__neg__at_____00__closed__1_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ByCases_0__Mathlib_Tactic_ByCases_tacticTry__push__neg__at_____00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ByCases_0__Mathlib_Tactic_ByCases_tacticTry__push__neg__at_____00__closed__1_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__0_value),LEAN_SCALAR_PTR_LITERAL(234, 232, 174, 134, 127, 136, 69, 92)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ByCases_0__Mathlib_Tactic_ByCases_tacticTry__push__neg__at_____00__closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ByCases_0__Mathlib_Tactic_ByCases_tacticTry__push__neg__at_____00__closed__2_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ByCases_0__Mathlib_Tactic_ByCases_tacticTry__push__neg__at_____00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ByCases_0__Mathlib_Tactic_ByCases_tacticTry__push__neg__at_____00__closed__2_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__1_value),LEAN_SCALAR_PTR_LITERAL(191, 70, 156, 159, 11, 54, 216, 94)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ByCases_0__Mathlib_Tactic_ByCases_tacticTry__push__neg__at_____00__closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ByCases_0__Mathlib_Tactic_ByCases_tacticTry__push__neg__at_____00__closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ByCases_0__Mathlib_Tactic_ByCases_tacticTry__push__neg__at_____00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ByCases_0__Mathlib_Tactic_ByCases_tacticTry__push__neg__at_____00__closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__2_value),LEAN_SCALAR_PTR_LITERAL(23, 146, 233, 76, 94, 143, 200, 197)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ByCases_0__Mathlib_Tactic_ByCases_tacticTry__push__neg__at_____00__closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ByCases_0__Mathlib_Tactic_ByCases_tacticTry__push__neg__at_____00__closed__4_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ByCases_0__Mathlib_Tactic_ByCases_tacticTry__push__neg__at_____00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ByCases_0__Mathlib_Tactic_ByCases_tacticTry__push__neg__at_____00__closed__4_value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(170, 52, 107, 247, 42, 44, 16, 127)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ByCases_0__Mathlib_Tactic_ByCases_tacticTry__push__neg__at_____00__closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ByCases_0__Mathlib_Tactic_ByCases_tacticTry__push__neg__at_____00__closed__5_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ByCases_0__Mathlib_Tactic_ByCases_tacticTry__push__neg__at_____00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ByCases_0__Mathlib_Tactic_ByCases_tacticTry__push__neg__at_____00__closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__0_value),LEAN_SCALAR_PTR_LITERAL(187, 75, 184, 100, 146, 31, 0, 93)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ByCases_0__Mathlib_Tactic_ByCases_tacticTry__push__neg__at_____00__closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ByCases_0__Mathlib_Tactic_ByCases_tacticTry__push__neg__at_____00__closed__6_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ByCases_0__Mathlib_Tactic_ByCases_tacticTry__push__neg__at_____00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ByCases_0__Mathlib_Tactic_ByCases_tacticTry__push__neg__at_____00__closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__1_value),LEAN_SCALAR_PTR_LITERAL(186, 166, 162, 193, 77, 210, 235, 236)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ByCases_0__Mathlib_Tactic_ByCases_tacticTry__push__neg__at_____00__closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ByCases_0__Mathlib_Tactic_ByCases_tacticTry__push__neg__at_____00__closed__7_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ByCases_0__Mathlib_Tactic_ByCases_tacticTry__push__neg__at_____00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ByCases_0__Mathlib_Tactic_ByCases_tacticTry__push__neg__at_____00__closed__7_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__2_value),LEAN_SCALAR_PTR_LITERAL(214, 174, 67, 93, 160, 206, 44, 206)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ByCases_0__Mathlib_Tactic_ByCases_tacticTry__push__neg__at_____00__closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ByCases_0__Mathlib_Tactic_ByCases_tacticTry__push__neg__at_____00__closed__8_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ByCases_0__Mathlib_Tactic_ByCases_tacticTry__push__neg__at_____00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = "tacticTry_push_neg_at__"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ByCases_0__Mathlib_Tactic_ByCases_tacticTry__push__neg__at_____00__closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ByCases_0__Mathlib_Tactic_ByCases_tacticTry__push__neg__at_____00__closed__9_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ByCases_0__Mathlib_Tactic_ByCases_tacticTry__push__neg__at_____00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ByCases_0__Mathlib_Tactic_ByCases_tacticTry__push__neg__at_____00__closed__8_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ByCases_0__Mathlib_Tactic_ByCases_tacticTry__push__neg__at_____00__closed__9_value),LEAN_SCALAR_PTR_LITERAL(60, 61, 142, 185, 58, 127, 97, 192)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ByCases_0__Mathlib_Tactic_ByCases_tacticTry__push__neg__at_____00__closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ByCases_0__Mathlib_Tactic_ByCases_tacticTry__push__neg__at_____00__closed__10_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ByCases_0__Mathlib_Tactic_ByCases_tacticTry__push__neg__at_____00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "try_push_neg_at"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ByCases_0__Mathlib_Tactic_ByCases_tacticTry__push__neg__at_____00__closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ByCases_0__Mathlib_Tactic_ByCases_tacticTry__push__neg__at_____00__closed__11_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ByCases_0__Mathlib_Tactic_ByCases_tacticTry__push__neg__at_____00__closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ByCases_0__Mathlib_Tactic_ByCases_tacticTry__push__neg__at_____00__closed__11_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ByCases_0__Mathlib_Tactic_ByCases_tacticTry__push__neg__at_____00__closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ByCases_0__Mathlib_Tactic_ByCases_tacticTry__push__neg__at_____00__closed__12_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_ByCases_0__Mathlib_Tactic_ByCases_tacticTry__push__neg__at_____00__closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_ByCases_0__Mathlib_Tactic_ByCases_tacticTry__push__neg__at_____00__closed__13;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_ByCases_0__Mathlib_Tactic_ByCases_tacticTry__push__neg__at_____00__closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_ByCases_0__Mathlib_Tactic_ByCases_tacticTry__push__neg__at_____00__closed__14;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_ByCases_0__Mathlib_Tactic_ByCases_tacticTry__push__neg__at_____00__closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_ByCases_0__Mathlib_Tactic_ByCases_tacticTry__push__neg__at_____00__closed__15;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ByCases_0__Mathlib_Tactic_ByCases_tacticTry__push__neg__at____;
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______elabRules____private__Mathlib__Tactic__ByCases__0__Mathlib__Tactic__ByCases__tacticTry__push__neg__at______1_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______elabRules____private__Mathlib__Tactic__ByCases__0__Mathlib__Tactic__ByCases__tacticTry__push__neg__at______1_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______elabRules____private__Mathlib__Tactic__ByCases__0__Mathlib__Tactic__ByCases__tacticTry__push__neg__at______1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______elabRules____private__Mathlib__Tactic__ByCases__0__Mathlib__Tactic__ByCases__tacticTry__push__neg__at______1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______elabRules____private__Mathlib__Tactic__ByCases__0__Mathlib__Tactic__ByCases__tacticTry__push__neg__at______1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______elabRules____private__Mathlib__Tactic__ByCases__0__Mathlib__Tactic__ByCases__tacticTry__push__neg__at______1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______elabRules____private__Mathlib__Tactic__ByCases__0__Mathlib__Tactic__ByCases__tacticTry__push__neg__at______1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Not"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______elabRules____private__Mathlib__Tactic__ByCases__0__Mathlib__Tactic__ByCases__tacticTry__push__neg__at______1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______elabRules____private__Mathlib__Tactic__ByCases__0__Mathlib__Tactic__ByCases__tacticTry__push__neg__at______1___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______elabRules____private__Mathlib__Tactic__ByCases__0__Mathlib__Tactic__ByCases__tacticTry__push__neg__at______1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______elabRules____private__Mathlib__Tactic__ByCases__0__Mathlib__Tactic__ByCases__tacticTry__push__neg__at______1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(185, 11, 203, 55, 27, 192, 137, 230)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______elabRules____private__Mathlib__Tactic__ByCases__0__Mathlib__Tactic__ByCases__tacticTry__push__neg__at______1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______elabRules____private__Mathlib__Tactic__ByCases__0__Mathlib__Tactic__ByCases__tacticTry__push__neg__at______1___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______elabRules____private__Mathlib__Tactic__ByCases__0__Mathlib__Tactic__ByCases__tacticTry__push__neg__at______1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______elabRules____private__Mathlib__Tactic__ByCases__0__Mathlib__Tactic__ByCases__tacticTry__push__neg__at______1___closed__1_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______elabRules____private__Mathlib__Tactic__ByCases__0__Mathlib__Tactic__ByCases__tacticTry__push__neg__at______1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______elabRules____private__Mathlib__Tactic__ByCases__0__Mathlib__Tactic__ByCases__tacticTry__push__neg__at______1___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______elabRules____private__Mathlib__Tactic__ByCases__0__Mathlib__Tactic__ByCases__tacticTry__push__neg__at______1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______elabRules____private__Mathlib__Tactic__ByCases__0__Mathlib__Tactic__ByCases__tacticTry__push__neg__at______1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "optConfig"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__3_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__3_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__3_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(137, 208, 10, 74, 108, 50, 106, 48)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "seq1"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__5_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__5_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__5_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__5_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__5_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__4_value),LEAN_SCALAR_PTR_LITERAL(242, 140, 137, 56, 141, 11, 143, 117)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "tacticBy_cases_:_"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(252, 65, 31, 128, 134, 243, 21, 139)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "by_cases"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__10_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ":"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__11_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ";"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__12_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Batteries"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__13_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tacticOn_goal-_=>_"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__14_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__15_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__13_value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__15_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__15_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__1_value),LEAN_SCALAR_PTR_LITERAL(219, 16, 21, 14, 34, 244, 116, 98)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__15_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__14_value),LEAN_SCALAR_PTR_LITERAL(243, 56, 227, 189, 147, 207, 104, 76)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__15_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "on_goal"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__16_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__17;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "num"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__18_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__18_value),LEAN_SCALAR_PTR_LITERAL(227, 68, 22, 222, 47, 51, 204, 84)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__19_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "2"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__20 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__20_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "=>"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__21 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__21_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tacticSeq"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__22 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__22_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__23_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__23_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__23_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__23_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__23_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__23_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__22_value),LEAN_SCALAR_PTR_LITERAL(212, 140, 85, 215, 241, 69, 7, 118)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__23 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__23_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tacticSeq1Indented"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__24 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__24_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__25_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__25_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__25_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__25_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__25_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__25_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__24_value),LEAN_SCALAR_PTR_LITERAL(223, 90, 160, 238, 133, 180, 23, 239)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__25 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__25_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "by_cases!"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__26 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__26_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "h"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__27 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__27_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__28_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__28;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__27_value),LEAN_SCALAR_PTR_LITERAL(176, 181, 207, 77, 197, 87, 68, 121)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__29 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__29_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___boxed(lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__9(void){
_start:
{
lean_object* v___x_17_; lean_object* v___x_18_; lean_object* v___x_19_; lean_object* v___x_20_; 
v___x_17_ = l_Lean_Parser_Tactic_optConfig;
v___x_18_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__8));
v___x_19_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__6));
v___x_20_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_20_, 0, v___x_19_);
lean_ctor_set(v___x_20_, 1, v___x_18_);
lean_ctor_set(v___x_20_, 2, v___x_17_);
return v___x_20_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__22(void){
_start:
{
lean_object* v___x_45_; lean_object* v___x_46_; lean_object* v___x_47_; lean_object* v___x_48_; 
v___x_45_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__21));
v___x_46_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__9, &lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__9_once, _init_lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__9);
v___x_47_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__6));
v___x_48_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_48_, 0, v___x_47_);
lean_ctor_set(v___x_48_, 1, v___x_46_);
lean_ctor_set(v___x_48_, 2, v___x_45_);
return v___x_48_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__26(void){
_start:
{
lean_object* v___x_55_; lean_object* v___x_56_; lean_object* v___x_57_; lean_object* v___x_58_; 
v___x_55_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__25));
v___x_56_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__22, &lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__22_once, _init_lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__22);
v___x_57_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__6));
v___x_58_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_58_, 0, v___x_57_);
lean_ctor_set(v___x_58_, 1, v___x_56_);
lean_ctor_set(v___x_58_, 2, v___x_55_);
return v___x_58_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__27(void){
_start:
{
lean_object* v___x_59_; lean_object* v___x_60_; lean_object* v___x_61_; lean_object* v___x_62_; 
v___x_59_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__26, &lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__26_once, _init_lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__26);
v___x_60_ = lean_unsigned_to_nat(1022u);
v___x_61_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__4));
v___x_62_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_62_, 0, v___x_61_);
lean_ctor_set(v___x_62_, 1, v___x_60_);
lean_ctor_set(v___x_62_, 2, v___x_59_);
return v___x_62_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21(void){
_start:
{
lean_object* v___x_63_; 
v___x_63_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__27, &lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__27_once, _init_lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__27);
return v___x_63_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_ByCases_0__Mathlib_Tactic_ByCases_tacticTry__push__neg__at_____00__closed__13(void){
_start:
{
lean_object* v___x_97_; lean_object* v___x_98_; lean_object* v___x_99_; lean_object* v___x_100_; 
v___x_97_ = l_Lean_Parser_Tactic_optConfig;
v___x_98_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ByCases_0__Mathlib_Tactic_ByCases_tacticTry__push__neg__at_____00__closed__12));
v___x_99_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__6));
v___x_100_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_100_, 0, v___x_99_);
lean_ctor_set(v___x_100_, 1, v___x_98_);
lean_ctor_set(v___x_100_, 2, v___x_97_);
return v___x_100_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_ByCases_0__Mathlib_Tactic_ByCases_tacticTry__push__neg__at_____00__closed__14(void){
_start:
{
lean_object* v___x_101_; lean_object* v___x_102_; lean_object* v___x_103_; lean_object* v___x_104_; 
v___x_101_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__16));
v___x_102_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_ByCases_0__Mathlib_Tactic_ByCases_tacticTry__push__neg__at_____00__closed__13, &lp_mathlib___private_Mathlib_Tactic_ByCases_0__Mathlib_Tactic_ByCases_tacticTry__push__neg__at_____00__closed__13_once, _init_lp_mathlib___private_Mathlib_Tactic_ByCases_0__Mathlib_Tactic_ByCases_tacticTry__push__neg__at_____00__closed__13);
v___x_103_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__6));
v___x_104_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_104_, 0, v___x_103_);
lean_ctor_set(v___x_104_, 1, v___x_102_);
lean_ctor_set(v___x_104_, 2, v___x_101_);
return v___x_104_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_ByCases_0__Mathlib_Tactic_ByCases_tacticTry__push__neg__at_____00__closed__15(void){
_start:
{
lean_object* v___x_105_; lean_object* v___x_106_; lean_object* v___x_107_; lean_object* v___x_108_; 
v___x_105_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_ByCases_0__Mathlib_Tactic_ByCases_tacticTry__push__neg__at_____00__closed__14, &lp_mathlib___private_Mathlib_Tactic_ByCases_0__Mathlib_Tactic_ByCases_tacticTry__push__neg__at_____00__closed__14_once, _init_lp_mathlib___private_Mathlib_Tactic_ByCases_0__Mathlib_Tactic_ByCases_tacticTry__push__neg__at_____00__closed__14);
v___x_106_ = lean_unsigned_to_nat(1022u);
v___x_107_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ByCases_0__Mathlib_Tactic_ByCases_tacticTry__push__neg__at_____00__closed__10));
v___x_108_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_108_, 0, v___x_107_);
lean_ctor_set(v___x_108_, 1, v___x_106_);
lean_ctor_set(v___x_108_, 2, v___x_105_);
return v___x_108_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_ByCases_0__Mathlib_Tactic_ByCases_tacticTry__push__neg__at____(void){
_start:
{
lean_object* v___x_109_; 
v___x_109_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_ByCases_0__Mathlib_Tactic_ByCases_tacticTry__push__neg__at_____00__closed__15, &lp_mathlib___private_Mathlib_Tactic_ByCases_0__Mathlib_Tactic_ByCases_tacticTry__push__neg__at_____00__closed__15_once, _init_lp_mathlib___private_Mathlib_Tactic_ByCases_0__Mathlib_Tactic_ByCases_tacticTry__push__neg__at_____00__closed__15);
return v___x_109_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______elabRules____private__Mathlib__Tactic__ByCases__0__Mathlib__Tactic__ByCases__tacticTry__push__neg__at______1_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_110_; lean_object* v___x_111_; lean_object* v___x_112_; 
v___x_110_ = lean_box(0);
v___x_111_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_112_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_112_, 0, v___x_111_);
lean_ctor_set(v___x_112_, 1, v___x_110_);
return v___x_112_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______elabRules____private__Mathlib__Tactic__ByCases__0__Mathlib__Tactic__ByCases__tacticTry__push__neg__at______1_spec__0___redArg(){
_start:
{
lean_object* v___x_114_; lean_object* v___x_115_; 
v___x_114_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______elabRules____private__Mathlib__Tactic__ByCases__0__Mathlib__Tactic__ByCases__tacticTry__push__neg__at______1_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______elabRules____private__Mathlib__Tactic__ByCases__0__Mathlib__Tactic__ByCases__tacticTry__push__neg__at______1_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______elabRules____private__Mathlib__Tactic__ByCases__0__Mathlib__Tactic__ByCases__tacticTry__push__neg__at______1_spec__0___redArg___closed__0);
v___x_115_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_115_, 0, v___x_114_);
return v___x_115_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______elabRules____private__Mathlib__Tactic__ByCases__0__Mathlib__Tactic__ByCases__tacticTry__push__neg__at______1_spec__0___redArg___boxed(lean_object* v___y_116_){
_start:
{
lean_object* v_res_117_; 
v_res_117_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______elabRules____private__Mathlib__Tactic__ByCases__0__Mathlib__Tactic__ByCases__tacticTry__push__neg__at______1_spec__0___redArg();
return v_res_117_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______elabRules____private__Mathlib__Tactic__ByCases__0__Mathlib__Tactic__ByCases__tacticTry__push__neg__at______1_spec__0(lean_object* v_00_u03b1_118_, lean_object* v___y_119_, lean_object* v___y_120_, lean_object* v___y_121_, lean_object* v___y_122_, lean_object* v___y_123_, lean_object* v___y_124_, lean_object* v___y_125_, lean_object* v___y_126_){
_start:
{
lean_object* v___x_128_; 
v___x_128_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______elabRules____private__Mathlib__Tactic__ByCases__0__Mathlib__Tactic__ByCases__tacticTry__push__neg__at______1_spec__0___redArg();
return v___x_128_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______elabRules____private__Mathlib__Tactic__ByCases__0__Mathlib__Tactic__ByCases__tacticTry__push__neg__at______1_spec__0___boxed(lean_object* v_00_u03b1_129_, lean_object* v___y_130_, lean_object* v___y_131_, lean_object* v___y_132_, lean_object* v___y_133_, lean_object* v___y_134_, lean_object* v___y_135_, lean_object* v___y_136_, lean_object* v___y_137_, lean_object* v___y_138_){
_start:
{
lean_object* v_res_139_; 
v_res_139_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______elabRules____private__Mathlib__Tactic__ByCases__0__Mathlib__Tactic__ByCases__tacticTry__push__neg__at______1_spec__0(v_00_u03b1_129_, v___y_130_, v___y_131_, v___y_132_, v___y_133_, v___y_134_, v___y_135_, v___y_136_, v___y_137_);
lean_dec(v___y_137_);
lean_dec_ref(v___y_136_);
lean_dec(v___y_135_);
lean_dec_ref(v___y_134_);
lean_dec(v___y_133_);
lean_dec_ref(v___y_132_);
lean_dec(v___y_131_);
lean_dec_ref(v___y_130_);
return v_res_139_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______elabRules____private__Mathlib__Tactic__ByCases__0__Mathlib__Tactic__ByCases__tacticTry__push__neg__at______1(lean_object* v_x_145_, lean_object* v_a_146_, lean_object* v_a_147_, lean_object* v_a_148_, lean_object* v_a_149_, lean_object* v_a_150_, lean_object* v_a_151_, lean_object* v_a_152_, lean_object* v_a_153_){
_start:
{
lean_object* v___x_155_; uint8_t v___x_156_; 
v___x_155_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ByCases_0__Mathlib_Tactic_ByCases_tacticTry__push__neg__at_____00__closed__10));
lean_inc(v_x_145_);
v___x_156_ = l_Lean_Syntax_isOfKind(v_x_145_, v___x_155_);
if (v___x_156_ == 0)
{
lean_object* v___x_157_; 
lean_dec(v_x_145_);
v___x_157_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______elabRules____private__Mathlib__Tactic__ByCases__0__Mathlib__Tactic__ByCases__tacticTry__push__neg__at______1_spec__0___redArg();
return v___x_157_;
}
else
{
lean_object* v___x_158_; lean_object* v___x_159_; uint8_t v___x_160_; lean_object* v___x_161_; 
v___x_158_ = lean_unsigned_to_nat(1u);
v___x_159_ = l_Lean_Syntax_getArg(v_x_145_, v___x_158_);
v___x_160_ = 0;
v___x_161_ = lp_mathlib_Mathlib_Tactic_Push_elabPushConfig___redArg(v___x_159_, v___x_160_, v___x_156_, v_a_146_, v_a_152_, v_a_153_);
if (lean_obj_tag(v___x_161_) == 0)
{
lean_object* v_a_162_; lean_object* v___x_163_; lean_object* v___x_164_; lean_object* v___x_165_; lean_object* v___x_166_; lean_object* v___x_167_; lean_object* v___x_168_; lean_object* v___x_169_; uint8_t v___x_170_; uint8_t v___x_171_; lean_object* v___x_172_; 
v_a_162_ = lean_ctor_get(v___x_161_, 0);
lean_inc(v_a_162_);
lean_dec_ref_known(v___x_161_, 1);
v___x_163_ = lean_unsigned_to_nat(2u);
v___x_164_ = l_Lean_Syntax_getArg(v_x_145_, v___x_163_);
lean_dec(v_x_145_);
v___x_165_ = lean_box(0);
v___x_166_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______elabRules____private__Mathlib__Tactic__ByCases__0__Mathlib__Tactic__ByCases__tacticTry__push__neg__at______1___closed__2));
v___x_167_ = lean_mk_empty_array_with_capacity(v___x_158_);
v___x_168_ = lean_array_push(v___x_167_, v___x_164_);
v___x_169_ = lean_alloc_ctor(1, 1, 1);
lean_ctor_set(v___x_169_, 0, v___x_168_);
lean_ctor_set_uint8(v___x_169_, sizeof(void*)*1, v___x_160_);
v___x_170_ = 0;
v___x_171_ = lean_unbox(v_a_162_);
lean_dec(v_a_162_);
v___x_172_ = lp_mathlib_Mathlib_Tactic_Push_push(v___x_171_, v___x_165_, v___x_166_, v___x_169_, v___x_170_, v_a_146_, v_a_147_, v_a_148_, v_a_149_, v_a_150_, v_a_151_, v_a_152_, v_a_153_);
lean_dec_ref_known(v___x_169_, 1);
return v___x_172_;
}
else
{
lean_object* v_a_173_; lean_object* v___x_175_; uint8_t v_isShared_176_; uint8_t v_isSharedCheck_180_; 
lean_dec(v_x_145_);
v_a_173_ = lean_ctor_get(v___x_161_, 0);
v_isSharedCheck_180_ = !lean_is_exclusive(v___x_161_);
if (v_isSharedCheck_180_ == 0)
{
v___x_175_ = v___x_161_;
v_isShared_176_ = v_isSharedCheck_180_;
goto v_resetjp_174_;
}
else
{
lean_inc(v_a_173_);
lean_dec(v___x_161_);
v___x_175_ = lean_box(0);
v_isShared_176_ = v_isSharedCheck_180_;
goto v_resetjp_174_;
}
v_resetjp_174_:
{
lean_object* v___x_178_; 
if (v_isShared_176_ == 0)
{
v___x_178_ = v___x_175_;
goto v_reusejp_177_;
}
else
{
lean_object* v_reuseFailAlloc_179_; 
v_reuseFailAlloc_179_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_179_, 0, v_a_173_);
v___x_178_ = v_reuseFailAlloc_179_;
goto v_reusejp_177_;
}
v_reusejp_177_:
{
return v___x_178_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______elabRules____private__Mathlib__Tactic__ByCases__0__Mathlib__Tactic__ByCases__tacticTry__push__neg__at______1___boxed(lean_object* v_x_181_, lean_object* v_a_182_, lean_object* v_a_183_, lean_object* v_a_184_, lean_object* v_a_185_, lean_object* v_a_186_, lean_object* v_a_187_, lean_object* v_a_188_, lean_object* v_a_189_, lean_object* v_a_190_){
_start:
{
lean_object* v_res_191_; 
v_res_191_ = lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______elabRules____private__Mathlib__Tactic__ByCases__0__Mathlib__Tactic__ByCases__tacticTry__push__neg__at______1(v_x_181_, v_a_182_, v_a_183_, v_a_184_, v_a_185_, v_a_186_, v_a_187_, v_a_188_, v_a_189_);
lean_dec(v_a_189_);
lean_dec_ref(v_a_188_);
lean_dec(v_a_187_);
lean_dec_ref(v_a_186_);
lean_dec(v_a_185_);
lean_dec_ref(v_a_184_);
lean_dec(v_a_183_);
lean_dec_ref(v_a_182_);
return v_res_191_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__17(void){
_start:
{
lean_object* v___x_222_; 
v___x_222_ = l_Array_mkArray0(lean_box(0));
return v___x_222_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__28(void){
_start:
{
lean_object* v___x_242_; lean_object* v___x_243_; 
v___x_242_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__27));
v___x_243_ = l_String_toRawSubstring_x27(v___x_242_);
return v___x_243_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1(lean_object* v_x_246_, lean_object* v_a_247_, lean_object* v_a_248_){
_start:
{
lean_object* v___x_249_; uint8_t v___x_250_; 
v___x_249_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21___closed__4));
lean_inc(v_x_246_);
v___x_250_ = l_Lean_Syntax_isOfKind(v_x_246_, v___x_249_);
if (v___x_250_ == 0)
{
lean_object* v___x_251_; lean_object* v___x_252_; 
lean_dec(v_x_246_);
v___x_251_ = lean_box(1);
v___x_252_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_252_, 0, v___x_251_);
lean_ctor_set(v___x_252_, 1, v_a_248_);
return v___x_252_;
}
else
{
lean_object* v___x_253_; lean_object* v___x_254_; lean_object* v___x_255_; uint8_t v___x_256_; 
v___x_253_ = lean_unsigned_to_nat(1u);
v___x_254_ = l_Lean_Syntax_getArg(v_x_246_, v___x_253_);
v___x_255_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__3));
lean_inc(v___x_254_);
v___x_256_ = l_Lean_Syntax_isOfKind(v___x_254_, v___x_255_);
if (v___x_256_ == 0)
{
lean_object* v___x_257_; lean_object* v___x_258_; 
lean_dec(v___x_254_);
lean_dec(v_x_246_);
v___x_257_ = lean_box(1);
v___x_258_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_258_, 0, v___x_257_);
lean_ctor_set(v___x_258_, 1, v_a_248_);
return v___x_258_;
}
else
{
lean_object* v___x_259_; lean_object* v___x_260_; lean_object* v___x_261_; uint8_t v___x_262_; 
v___x_259_ = lean_unsigned_to_nat(0u);
v___x_260_ = lean_unsigned_to_nat(2u);
v___x_261_ = l_Lean_Syntax_getArg(v_x_246_, v___x_260_);
lean_inc(v___x_261_);
v___x_262_ = l_Lean_Syntax_matchesNull(v___x_261_, v___x_259_);
if (v___x_262_ == 0)
{
uint8_t v___x_263_; 
lean_inc(v___x_261_);
v___x_263_ = l_Lean_Syntax_matchesNull(v___x_261_, v___x_260_);
if (v___x_263_ == 0)
{
lean_object* v___x_264_; lean_object* v___x_265_; 
lean_dec(v___x_261_);
lean_dec(v___x_254_);
lean_dec(v_x_246_);
v___x_264_ = lean_box(1);
v___x_265_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_265_, 0, v___x_264_);
lean_ctor_set(v___x_265_, 1, v_a_248_);
return v___x_265_;
}
else
{
lean_object* v_ref_266_; lean_object* v___x_267_; lean_object* v___x_268_; lean_object* v___x_269_; lean_object* v___x_270_; lean_object* v___x_271_; lean_object* v___x_272_; lean_object* v___x_273_; lean_object* v___x_274_; lean_object* v___x_275_; lean_object* v___x_276_; lean_object* v___x_277_; lean_object* v___x_278_; lean_object* v___x_279_; lean_object* v___x_280_; lean_object* v___x_281_; lean_object* v___x_282_; lean_object* v___x_283_; lean_object* v___x_284_; lean_object* v___x_285_; lean_object* v___x_286_; lean_object* v___x_287_; lean_object* v___x_288_; lean_object* v___x_289_; lean_object* v___x_290_; lean_object* v___x_291_; lean_object* v___x_292_; lean_object* v___x_293_; lean_object* v___x_294_; lean_object* v___x_295_; lean_object* v___x_296_; lean_object* v___x_297_; lean_object* v___x_298_; lean_object* v___x_299_; lean_object* v___x_300_; lean_object* v___x_301_; lean_object* v___x_302_; lean_object* v___x_303_; lean_object* v___x_304_; lean_object* v___x_305_; 
v_ref_266_ = lean_ctor_get(v_a_247_, 5);
v___x_267_ = l_Lean_Syntax_getArg(v___x_261_, v___x_259_);
lean_dec(v___x_261_);
v___x_268_ = lean_unsigned_to_nat(3u);
v___x_269_ = l_Lean_Syntax_getArg(v_x_246_, v___x_268_);
lean_dec(v_x_246_);
v___x_270_ = l_Lean_SourceInfo_fromRef(v_ref_266_, v___x_262_);
v___x_271_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__5));
v___x_272_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__7));
v___x_273_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__9));
v___x_274_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__10));
lean_inc_n(v___x_270_, 17);
v___x_275_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_275_, 0, v___x_270_);
lean_ctor_set(v___x_275_, 1, v___x_274_);
v___x_276_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__11));
v___x_277_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_277_, 0, v___x_270_);
lean_ctor_set(v___x_277_, 1, v___x_276_);
lean_inc(v___x_267_);
v___x_278_ = l_Lean_Syntax_node2(v___x_270_, v___x_272_, v___x_267_, v___x_277_);
v___x_279_ = l_Lean_Syntax_node3(v___x_270_, v___x_273_, v___x_275_, v___x_278_, v___x_269_);
v___x_280_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__12));
v___x_281_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_281_, 0, v___x_270_);
lean_ctor_set(v___x_281_, 1, v___x_280_);
v___x_282_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__15));
v___x_283_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__16));
v___x_284_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_284_, 0, v___x_270_);
lean_ctor_set(v___x_284_, 1, v___x_283_);
v___x_285_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__17, &lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__17_once, _init_lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__17);
v___x_286_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_286_, 0, v___x_270_);
lean_ctor_set(v___x_286_, 1, v___x_272_);
lean_ctor_set(v___x_286_, 2, v___x_285_);
v___x_287_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__19));
v___x_288_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__20));
v___x_289_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_289_, 0, v___x_270_);
lean_ctor_set(v___x_289_, 1, v___x_288_);
v___x_290_ = l_Lean_Syntax_node1(v___x_270_, v___x_287_, v___x_289_);
v___x_291_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__21));
v___x_292_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_292_, 0, v___x_270_);
lean_ctor_set(v___x_292_, 1, v___x_291_);
v___x_293_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__23));
v___x_294_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__25));
v___x_295_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ByCases_0__Mathlib_Tactic_ByCases_tacticTry__push__neg__at_____00__closed__10));
v___x_296_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ByCases_0__Mathlib_Tactic_ByCases_tacticTry__push__neg__at_____00__closed__11));
v___x_297_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_297_, 0, v___x_270_);
lean_ctor_set(v___x_297_, 1, v___x_296_);
v___x_298_ = l_Lean_Syntax_node3(v___x_270_, v___x_295_, v___x_297_, v___x_254_, v___x_267_);
v___x_299_ = l_Lean_Syntax_node1(v___x_270_, v___x_272_, v___x_298_);
v___x_300_ = l_Lean_Syntax_node1(v___x_270_, v___x_294_, v___x_299_);
v___x_301_ = l_Lean_Syntax_node1(v___x_270_, v___x_293_, v___x_300_);
v___x_302_ = l_Lean_Syntax_node5(v___x_270_, v___x_282_, v___x_284_, v___x_286_, v___x_290_, v___x_292_, v___x_301_);
v___x_303_ = l_Lean_Syntax_node3(v___x_270_, v___x_272_, v___x_279_, v___x_281_, v___x_302_);
v___x_304_ = l_Lean_Syntax_node1(v___x_270_, v___x_271_, v___x_303_);
v___x_305_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_305_, 0, v___x_304_);
lean_ctor_set(v___x_305_, 1, v_a_248_);
return v___x_305_;
}
}
else
{
lean_object* v_quotContext_306_; lean_object* v_currMacroScope_307_; lean_object* v_ref_308_; lean_object* v___x_309_; lean_object* v___x_310_; uint8_t v___x_311_; lean_object* v___x_312_; lean_object* v___x_313_; lean_object* v___x_314_; lean_object* v___x_315_; lean_object* v___x_316_; lean_object* v___x_317_; lean_object* v___x_318_; lean_object* v___x_319_; lean_object* v___x_320_; lean_object* v___x_321_; lean_object* v___x_322_; lean_object* v___x_323_; lean_object* v___x_324_; lean_object* v___x_325_; 
lean_dec(v___x_261_);
v_quotContext_306_ = lean_ctor_get(v_a_247_, 1);
v_currMacroScope_307_ = lean_ctor_get(v_a_247_, 2);
v_ref_308_ = lean_ctor_get(v_a_247_, 5);
v___x_309_ = lean_unsigned_to_nat(3u);
v___x_310_ = l_Lean_Syntax_getArg(v_x_246_, v___x_309_);
lean_dec(v_x_246_);
v___x_311_ = 0;
v___x_312_ = l_Lean_SourceInfo_fromRef(v_ref_308_, v___x_311_);
v___x_313_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__26));
lean_inc_n(v___x_312_, 4);
v___x_314_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_314_, 0, v___x_312_);
lean_ctor_set(v___x_314_, 1, v___x_313_);
v___x_315_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__7));
v___x_316_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__28, &lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__28_once, _init_lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__28);
v___x_317_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__29));
lean_inc(v_currMacroScope_307_);
lean_inc(v_quotContext_306_);
v___x_318_ = l_Lean_addMacroScope(v_quotContext_306_, v___x_317_, v_currMacroScope_307_);
v___x_319_ = lean_box(0);
v___x_320_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_320_, 0, v___x_312_);
lean_ctor_set(v___x_320_, 1, v___x_316_);
lean_ctor_set(v___x_320_, 2, v___x_318_);
lean_ctor_set(v___x_320_, 3, v___x_319_);
v___x_321_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___closed__11));
v___x_322_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_322_, 0, v___x_312_);
lean_ctor_set(v___x_322_, 1, v___x_321_);
v___x_323_ = l_Lean_Syntax_node2(v___x_312_, v___x_315_, v___x_320_, v___x_322_);
v___x_324_ = l_Lean_Syntax_node4(v___x_312_, v___x_249_, v___x_314_, v___x_254_, v___x_323_, v___x_310_);
v___x_325_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_325_, 0, v___x_324_);
lean_ctor_set(v___x_325_, 1, v_a_248_);
return v___x_325_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1___boxed(lean_object* v_x_326_, lean_object* v_a_327_, lean_object* v_a_328_){
_start:
{
lean_object* v_res_329_; 
v_res_329_ = lp_mathlib_Mathlib_Tactic_ByCases___aux__Mathlib__Tactic__ByCases______macroRules__Mathlib__Tactic__ByCases__byCases_x21__1(v_x_326_, v_a_327_, v_a_328_);
lean_dec_ref(v_a_327_);
return v_res_329_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Tactic_PermuteGoals(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Push(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_ByCases(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Tactic_PermuteGoals(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Push(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_ByCases(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21 = _init_lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_ByCases_byCases_x21);
lp_mathlib___private_Mathlib_Tactic_ByCases_0__Mathlib_Tactic_ByCases_tacticTry__push__neg__at____ = _init_lp_mathlib___private_Mathlib_Tactic_ByCases_0__Mathlib_Tactic_ByCases_tacticTry__push__neg__at____();
lean_mark_persistent(lp_mathlib___private_Mathlib_Tactic_ByCases_0__Mathlib_Tactic_ByCases_tacticTry__push__neg__at____);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Tactic_PermuteGoals(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Push(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_ByCases(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Tactic_PermuteGoals(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Push(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_ByCases(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_ByCases(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_ByCases(builtin);
}
#ifdef __cplusplus
}
#endif
