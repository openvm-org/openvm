// Lean compiler output
// Module: Mathlib.Tactic.ByContra
// Imports: public import Init public meta import Init public import Batteries.Tactic.Init public import Mathlib.Tactic.Push
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
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_Name_str___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_num___override(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_Push_elabPushConfig___redArg(lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_Push_push(uint8_t, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Parser_Tactic_optConfig;
lean_object* l_Array_mkArray0(lean_object*);
extern lean_object* l_Lean_Parser_Tactic_rcasesPatMed;
lean_object* l_Lean_Name_mkStr5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkIdent(lean_object*);
uint8_t l_Lean_Syntax_isNone(lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "ByContra"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "byContra!"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__2_value),LEAN_SCALAR_PTR_LITERAL(81, 42, 251, 71, 231, 71, 43, 201)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__3_value),LEAN_SCALAR_PTR_LITERAL(183, 203, 28, 162, 126, 183, 195, 112)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__5_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "by_contra!"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__7_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__8_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__9;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "optional"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__10_value),LEAN_SCALAR_PTR_LITERAL(233, 141, 154, 50, 143, 135, 42, 252)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__11_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "ppSpace"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__12_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__12_value),LEAN_SCALAR_PTR_LITERAL(207, 47, 58, 43, 30, 240, 125, 246)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__13_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__14_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "colGt"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__15_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__15_value),LEAN_SCALAR_PTR_LITERAL(185, 236, 32, 153, 169, 213, 53, 244)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__16_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__16_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__17_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__14_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__17_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__18_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__19;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__20_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__20;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__21_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__21;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = " : "};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__22 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__22_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__22_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__23 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__23_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__24 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__24_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__24_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__25 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__25_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__25_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__26 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__26_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__23_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__26_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__27 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__27_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__11_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__27_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__28 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__28_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__29_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__29;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__30_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__30;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg__at_____00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "_private"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg__at_____00__closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg__at_____00__closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg__at_____00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg__at_____00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(103, 214, 75, 80, 34, 198, 193, 153)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg__at_____00__closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg__at_____00__closed__1_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg__at_____00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg__at_____00__closed__1_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__0_value),LEAN_SCALAR_PTR_LITERAL(234, 232, 174, 134, 127, 136, 69, 92)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg__at_____00__closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg__at_____00__closed__2_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg__at_____00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg__at_____00__closed__2_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__1_value),LEAN_SCALAR_PTR_LITERAL(191, 70, 156, 159, 11, 54, 216, 94)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg__at_____00__closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg__at_____00__closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg__at_____00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg__at_____00__closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__2_value),LEAN_SCALAR_PTR_LITERAL(237, 199, 81, 170, 203, 202, 220, 24)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg__at_____00__closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg__at_____00__closed__4_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg__at_____00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg__at_____00__closed__4_value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(216, 67, 222, 209, 164, 73, 222, 218)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg__at_____00__closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg__at_____00__closed__5_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg__at_____00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg__at_____00__closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__0_value),LEAN_SCALAR_PTR_LITERAL(193, 4, 62, 120, 50, 132, 208, 253)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg__at_____00__closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg__at_____00__closed__6_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg__at_____00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg__at_____00__closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__1_value),LEAN_SCALAR_PTR_LITERAL(72, 165, 83, 228, 151, 84, 109, 38)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg__at_____00__closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg__at_____00__closed__7_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg__at_____00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg__at_____00__closed__7_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__2_value),LEAN_SCALAR_PTR_LITERAL(38, 231, 71, 147, 170, 116, 227, 113)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg__at_____00__closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg__at_____00__closed__8_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg__at_____00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = "tacticTry_push_neg_at__"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg__at_____00__closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg__at_____00__closed__9_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg__at_____00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg__at_____00__closed__8_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg__at_____00__closed__9_value),LEAN_SCALAR_PTR_LITERAL(108, 193, 128, 169, 80, 85, 102, 180)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg__at_____00__closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg__at_____00__closed__10_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg__at_____00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "try_push_neg_at"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg__at_____00__closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg__at_____00__closed__11_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg__at_____00__closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg__at_____00__closed__11_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg__at_____00__closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg__at_____00__closed__12_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg__at_____00__closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg__at_____00__closed__13;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg__at_____00__closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg__at_____00__closed__14 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg__at_____00__closed__14_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg__at_____00__closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg__at_____00__closed__14_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg__at_____00__closed__15 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg__at_____00__closed__15_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg__at_____00__closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg__at_____00__closed__15_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg__at_____00__closed__16 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg__at_____00__closed__16_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg__at_____00__closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg__at_____00__closed__17;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg__at_____00__closed__18_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg__at_____00__closed__18;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg__at____;
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______elabRules____private__Mathlib__Tactic__ByContra__0__Mathlib__Tactic__ByContra__tacticTry__push__neg__at______1_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______elabRules____private__Mathlib__Tactic__ByContra__0__Mathlib__Tactic__ByContra__tacticTry__push__neg__at______1_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______elabRules____private__Mathlib__Tactic__ByContra__0__Mathlib__Tactic__ByContra__tacticTry__push__neg__at______1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______elabRules____private__Mathlib__Tactic__ByContra__0__Mathlib__Tactic__ByContra__tacticTry__push__neg__at______1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______elabRules____private__Mathlib__Tactic__ByContra__0__Mathlib__Tactic__ByContra__tacticTry__push__neg__at______1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______elabRules____private__Mathlib__Tactic__ByContra__0__Mathlib__Tactic__ByContra__tacticTry__push__neg__at______1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______elabRules____private__Mathlib__Tactic__ByContra__0__Mathlib__Tactic__ByContra__tacticTry__push__neg__at______1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Not"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______elabRules____private__Mathlib__Tactic__ByContra__0__Mathlib__Tactic__ByContra__tacticTry__push__neg__at______1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______elabRules____private__Mathlib__Tactic__ByContra__0__Mathlib__Tactic__ByContra__tacticTry__push__neg__at______1___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______elabRules____private__Mathlib__Tactic__ByContra__0__Mathlib__Tactic__ByContra__tacticTry__push__neg__at______1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______elabRules____private__Mathlib__Tactic__ByContra__0__Mathlib__Tactic__ByContra__tacticTry__push__neg__at______1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(185, 11, 203, 55, 27, 192, 137, 230)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______elabRules____private__Mathlib__Tactic__ByContra__0__Mathlib__Tactic__ByContra__tacticTry__push__neg__at______1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______elabRules____private__Mathlib__Tactic__ByContra__0__Mathlib__Tactic__ByContra__tacticTry__push__neg__at______1___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______elabRules____private__Mathlib__Tactic__ByContra__0__Mathlib__Tactic__ByContra__tacticTry__push__neg__at______1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______elabRules____private__Mathlib__Tactic__ByContra__0__Mathlib__Tactic__ByContra__tacticTry__push__neg__at______1___closed__1_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______elabRules____private__Mathlib__Tactic__ByContra__0__Mathlib__Tactic__ByContra__tacticTry__push__neg__at______1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______elabRules____private__Mathlib__Tactic__ByContra__0__Mathlib__Tactic__ByContra__tacticTry__push__neg__at______1___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______elabRules____private__Mathlib__Tactic__ByContra__0__Mathlib__Tactic__ByContra__tacticTry__push__neg__at______1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______elabRules____private__Mathlib__Tactic__ByContra__0__Mathlib__Tactic__ByContra__tacticTry__push__neg__at______1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "tacticTry_push_neg_"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg___00__closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg___00__closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg__at_____00__closed__8_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(166, 168, 51, 141, 77, 195, 175, 51)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg___00__closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg___00__closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "try_push_neg"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg___00__closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg___00__closed__2_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg___00__closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg___00__closed__3_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg___00__closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg___00__closed__4;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg___00__closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg___00__closed__5;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg__;
static const lean_array_object lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______elabRules____private__Mathlib__Tactic__ByContra__0__Mathlib__Tactic__ByContra__tacticTry__push__neg____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______elabRules____private__Mathlib__Tactic__ByContra__0__Mathlib__Tactic__ByContra__tacticTry__push__neg____1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______elabRules____private__Mathlib__Tactic__ByContra__0__Mathlib__Tactic__ByContra__tacticTry__push__neg____1___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______elabRules____private__Mathlib__Tactic__ByContra__0__Mathlib__Tactic__ByContra__tacticTry__push__neg____1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______elabRules____private__Mathlib__Tactic__ByContra__0__Mathlib__Tactic__ByContra__tacticTry__push__neg____1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___lam__0___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "paren"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "("};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tacticSeq"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tacticSeq1Indented"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__4_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Batteries"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "byContra"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__8_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__8_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__1_value),LEAN_SCALAR_PTR_LITERAL(219, 16, 21, 14, 34, 244, 116, 98)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__8_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(172, 151, 1, 107, 2, 236, 248, 130)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__8_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "by_contra"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "rcasesPatMed"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__10_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "rcasesPat"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__11_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "one"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__12_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "h"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__13_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__14;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__13_value),LEAN_SCALAR_PTR_LITERAL(176, 181, 207, 77, 197, 87, 68, 121)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__15_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__16;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ";"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__17_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "revert"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__18_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "rintro"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__19_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "rintroPat"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__20 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__20_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "rcasesPatLo"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__21 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__21_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ")"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__22 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__22_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "skip"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__23 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__23_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "replace"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__24 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__24_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__25 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__25_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "letDecl"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__26 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__26_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "letIdDecl"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__27 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__27_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "letId"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__28 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__28_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "typeSpec"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__29 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__29_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ":"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__30 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__30_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ":="};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__31 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__31_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "byTactic"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__32 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__32_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__33_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "by"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__33 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__33_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__34_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "exact"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__34 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__34_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__35 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__35_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__36_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__36 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__36_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__37_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__35_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__37_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__37_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__36_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__37_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__37_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__37_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__37_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__10_value),LEAN_SCALAR_PTR_LITERAL(253, 13, 65, 195, 228, 27, 47, 149)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__37 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__37_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__38_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__35_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__38_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__38_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__36_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__38_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__38_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__38_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__38_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__11_value),LEAN_SCALAR_PTR_LITERAL(162, 181, 165, 225, 136, 177, 169, 19)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__38_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__38_value_aux_3),((lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__12_value),LEAN_SCALAR_PTR_LITERAL(186, 152, 172, 228, 11, 240, 156, 168)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__38 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__38_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__39_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "this"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__39 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__39_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__40_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__39_value),LEAN_SCALAR_PTR_LITERAL(38, 116, 214, 236, 212, 160, 188, 150)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__40 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__40_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__41_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__41;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___boxed(lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__9(void){
_start:
{
lean_object* v___x_17_; lean_object* v___x_18_; lean_object* v___x_19_; lean_object* v___x_20_; 
v___x_17_ = l_Lean_Parser_Tactic_optConfig;
v___x_18_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__8));
v___x_19_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__6));
v___x_20_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_20_, 0, v___x_19_);
lean_ctor_set(v___x_20_, 1, v___x_18_);
lean_ctor_set(v___x_20_, 2, v___x_17_);
return v___x_20_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__19(void){
_start:
{
lean_object* v___x_38_; lean_object* v___x_39_; lean_object* v___x_40_; lean_object* v___x_41_; 
v___x_38_ = l_Lean_Parser_Tactic_rcasesPatMed;
v___x_39_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__18));
v___x_40_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__6));
v___x_41_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_41_, 0, v___x_40_);
lean_ctor_set(v___x_41_, 1, v___x_39_);
lean_ctor_set(v___x_41_, 2, v___x_38_);
return v___x_41_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__20(void){
_start:
{
lean_object* v___x_42_; lean_object* v___x_43_; lean_object* v___x_44_; 
v___x_42_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__19, &lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__19_once, _init_lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__19);
v___x_43_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__11));
v___x_44_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_44_, 0, v___x_43_);
lean_ctor_set(v___x_44_, 1, v___x_42_);
return v___x_44_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__21(void){
_start:
{
lean_object* v___x_45_; lean_object* v___x_46_; lean_object* v___x_47_; lean_object* v___x_48_; 
v___x_45_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__20, &lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__20_once, _init_lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__20);
v___x_46_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__9, &lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__9_once, _init_lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__9);
v___x_47_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__6));
v___x_48_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_48_, 0, v___x_47_);
lean_ctor_set(v___x_48_, 1, v___x_46_);
lean_ctor_set(v___x_48_, 2, v___x_45_);
return v___x_48_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__29(void){
_start:
{
lean_object* v___x_65_; lean_object* v___x_66_; lean_object* v___x_67_; lean_object* v___x_68_; 
v___x_65_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__28));
v___x_66_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__21, &lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__21_once, _init_lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__21);
v___x_67_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__6));
v___x_68_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_68_, 0, v___x_67_);
lean_ctor_set(v___x_68_, 1, v___x_66_);
lean_ctor_set(v___x_68_, 2, v___x_65_);
return v___x_68_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__30(void){
_start:
{
lean_object* v___x_69_; lean_object* v___x_70_; lean_object* v___x_71_; lean_object* v___x_72_; 
v___x_69_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__29, &lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__29_once, _init_lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__29);
v___x_70_ = lean_unsigned_to_nat(1022u);
v___x_71_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__4));
v___x_72_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_72_, 0, v___x_71_);
lean_ctor_set(v___x_72_, 1, v___x_70_);
lean_ctor_set(v___x_72_, 2, v___x_69_);
return v___x_72_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21(void){
_start:
{
lean_object* v___x_73_; 
v___x_73_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__30, &lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__30_once, _init_lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__30);
return v___x_73_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg__at_____00__closed__13(void){
_start:
{
lean_object* v___x_107_; lean_object* v___x_108_; lean_object* v___x_109_; lean_object* v___x_110_; 
v___x_107_ = l_Lean_Parser_Tactic_optConfig;
v___x_108_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg__at_____00__closed__12));
v___x_109_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__6));
v___x_110_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_110_, 0, v___x_109_);
lean_ctor_set(v___x_110_, 1, v___x_108_);
lean_ctor_set(v___x_110_, 2, v___x_107_);
return v___x_110_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg__at_____00__closed__17(void){
_start:
{
lean_object* v___x_116_; lean_object* v___x_117_; lean_object* v___x_118_; lean_object* v___x_119_; 
v___x_116_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg__at_____00__closed__16));
v___x_117_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg__at_____00__closed__13, &lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg__at_____00__closed__13_once, _init_lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg__at_____00__closed__13);
v___x_118_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__6));
v___x_119_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_119_, 0, v___x_118_);
lean_ctor_set(v___x_119_, 1, v___x_117_);
lean_ctor_set(v___x_119_, 2, v___x_116_);
return v___x_119_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg__at_____00__closed__18(void){
_start:
{
lean_object* v___x_120_; lean_object* v___x_121_; lean_object* v___x_122_; lean_object* v___x_123_; 
v___x_120_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg__at_____00__closed__17, &lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg__at_____00__closed__17_once, _init_lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg__at_____00__closed__17);
v___x_121_ = lean_unsigned_to_nat(1022u);
v___x_122_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg__at_____00__closed__10));
v___x_123_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_123_, 0, v___x_122_);
lean_ctor_set(v___x_123_, 1, v___x_121_);
lean_ctor_set(v___x_123_, 2, v___x_120_);
return v___x_123_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg__at____(void){
_start:
{
lean_object* v___x_124_; 
v___x_124_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg__at_____00__closed__18, &lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg__at_____00__closed__18_once, _init_lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg__at_____00__closed__18);
return v___x_124_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______elabRules____private__Mathlib__Tactic__ByContra__0__Mathlib__Tactic__ByContra__tacticTry__push__neg__at______1_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_125_; lean_object* v___x_126_; lean_object* v___x_127_; 
v___x_125_ = lean_box(0);
v___x_126_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_127_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_127_, 0, v___x_126_);
lean_ctor_set(v___x_127_, 1, v___x_125_);
return v___x_127_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______elabRules____private__Mathlib__Tactic__ByContra__0__Mathlib__Tactic__ByContra__tacticTry__push__neg__at______1_spec__0___redArg(){
_start:
{
lean_object* v___x_129_; lean_object* v___x_130_; 
v___x_129_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______elabRules____private__Mathlib__Tactic__ByContra__0__Mathlib__Tactic__ByContra__tacticTry__push__neg__at______1_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______elabRules____private__Mathlib__Tactic__ByContra__0__Mathlib__Tactic__ByContra__tacticTry__push__neg__at______1_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______elabRules____private__Mathlib__Tactic__ByContra__0__Mathlib__Tactic__ByContra__tacticTry__push__neg__at______1_spec__0___redArg___closed__0);
v___x_130_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_130_, 0, v___x_129_);
return v___x_130_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______elabRules____private__Mathlib__Tactic__ByContra__0__Mathlib__Tactic__ByContra__tacticTry__push__neg__at______1_spec__0___redArg___boxed(lean_object* v___y_131_){
_start:
{
lean_object* v_res_132_; 
v_res_132_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______elabRules____private__Mathlib__Tactic__ByContra__0__Mathlib__Tactic__ByContra__tacticTry__push__neg__at______1_spec__0___redArg();
return v_res_132_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______elabRules____private__Mathlib__Tactic__ByContra__0__Mathlib__Tactic__ByContra__tacticTry__push__neg__at______1_spec__0(lean_object* v_00_u03b1_133_, lean_object* v___y_134_, lean_object* v___y_135_, lean_object* v___y_136_, lean_object* v___y_137_, lean_object* v___y_138_, lean_object* v___y_139_, lean_object* v___y_140_, lean_object* v___y_141_){
_start:
{
lean_object* v___x_143_; 
v___x_143_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______elabRules____private__Mathlib__Tactic__ByContra__0__Mathlib__Tactic__ByContra__tacticTry__push__neg__at______1_spec__0___redArg();
return v___x_143_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______elabRules____private__Mathlib__Tactic__ByContra__0__Mathlib__Tactic__ByContra__tacticTry__push__neg__at______1_spec__0___boxed(lean_object* v_00_u03b1_144_, lean_object* v___y_145_, lean_object* v___y_146_, lean_object* v___y_147_, lean_object* v___y_148_, lean_object* v___y_149_, lean_object* v___y_150_, lean_object* v___y_151_, lean_object* v___y_152_, lean_object* v___y_153_){
_start:
{
lean_object* v_res_154_; 
v_res_154_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______elabRules____private__Mathlib__Tactic__ByContra__0__Mathlib__Tactic__ByContra__tacticTry__push__neg__at______1_spec__0(v_00_u03b1_144_, v___y_145_, v___y_146_, v___y_147_, v___y_148_, v___y_149_, v___y_150_, v___y_151_, v___y_152_);
lean_dec(v___y_152_);
lean_dec_ref(v___y_151_);
lean_dec(v___y_150_);
lean_dec_ref(v___y_149_);
lean_dec(v___y_148_);
lean_dec_ref(v___y_147_);
lean_dec(v___y_146_);
lean_dec_ref(v___y_145_);
return v_res_154_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______elabRules____private__Mathlib__Tactic__ByContra__0__Mathlib__Tactic__ByContra__tacticTry__push__neg__at______1(lean_object* v_x_160_, lean_object* v_a_161_, lean_object* v_a_162_, lean_object* v_a_163_, lean_object* v_a_164_, lean_object* v_a_165_, lean_object* v_a_166_, lean_object* v_a_167_, lean_object* v_a_168_){
_start:
{
lean_object* v___x_170_; uint8_t v___x_171_; 
v___x_170_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg__at_____00__closed__10));
lean_inc(v_x_160_);
v___x_171_ = l_Lean_Syntax_isOfKind(v_x_160_, v___x_170_);
if (v___x_171_ == 0)
{
lean_object* v___x_172_; 
lean_dec(v_x_160_);
v___x_172_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______elabRules____private__Mathlib__Tactic__ByContra__0__Mathlib__Tactic__ByContra__tacticTry__push__neg__at______1_spec__0___redArg();
return v___x_172_;
}
else
{
lean_object* v___x_173_; lean_object* v___x_174_; uint8_t v___x_175_; lean_object* v___x_176_; 
v___x_173_ = lean_unsigned_to_nat(1u);
v___x_174_ = l_Lean_Syntax_getArg(v_x_160_, v___x_173_);
v___x_175_ = 0;
v___x_176_ = lp_mathlib_Mathlib_Tactic_Push_elabPushConfig___redArg(v___x_174_, v___x_175_, v___x_171_, v_a_161_, v_a_167_, v_a_168_);
if (lean_obj_tag(v___x_176_) == 0)
{
lean_object* v_a_177_; lean_object* v___x_178_; lean_object* v___x_179_; lean_object* v___x_180_; lean_object* v___x_181_; lean_object* v___x_182_; lean_object* v___x_183_; lean_object* v___x_184_; uint8_t v___x_185_; uint8_t v___x_186_; lean_object* v___x_187_; 
v_a_177_ = lean_ctor_get(v___x_176_, 0);
lean_inc(v_a_177_);
lean_dec_ref_known(v___x_176_, 1);
v___x_178_ = lean_unsigned_to_nat(2u);
v___x_179_ = l_Lean_Syntax_getArg(v_x_160_, v___x_178_);
lean_dec(v_x_160_);
v___x_180_ = lean_box(0);
v___x_181_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______elabRules____private__Mathlib__Tactic__ByContra__0__Mathlib__Tactic__ByContra__tacticTry__push__neg__at______1___closed__2));
v___x_182_ = lean_mk_empty_array_with_capacity(v___x_173_);
v___x_183_ = lean_array_push(v___x_182_, v___x_179_);
v___x_184_ = lean_alloc_ctor(1, 1, 1);
lean_ctor_set(v___x_184_, 0, v___x_183_);
lean_ctor_set_uint8(v___x_184_, sizeof(void*)*1, v___x_175_);
v___x_185_ = 0;
v___x_186_ = lean_unbox(v_a_177_);
lean_dec(v_a_177_);
v___x_187_ = lp_mathlib_Mathlib_Tactic_Push_push(v___x_186_, v___x_180_, v___x_181_, v___x_184_, v___x_185_, v_a_161_, v_a_162_, v_a_163_, v_a_164_, v_a_165_, v_a_166_, v_a_167_, v_a_168_);
lean_dec_ref_known(v___x_184_, 1);
return v___x_187_;
}
else
{
lean_object* v_a_188_; lean_object* v___x_190_; uint8_t v_isShared_191_; uint8_t v_isSharedCheck_195_; 
lean_dec(v_x_160_);
v_a_188_ = lean_ctor_get(v___x_176_, 0);
v_isSharedCheck_195_ = !lean_is_exclusive(v___x_176_);
if (v_isSharedCheck_195_ == 0)
{
v___x_190_ = v___x_176_;
v_isShared_191_ = v_isSharedCheck_195_;
goto v_resetjp_189_;
}
else
{
lean_inc(v_a_188_);
lean_dec(v___x_176_);
v___x_190_ = lean_box(0);
v_isShared_191_ = v_isSharedCheck_195_;
goto v_resetjp_189_;
}
v_resetjp_189_:
{
lean_object* v___x_193_; 
if (v_isShared_191_ == 0)
{
v___x_193_ = v___x_190_;
goto v_reusejp_192_;
}
else
{
lean_object* v_reuseFailAlloc_194_; 
v_reuseFailAlloc_194_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_194_, 0, v_a_188_);
v___x_193_ = v_reuseFailAlloc_194_;
goto v_reusejp_192_;
}
v_reusejp_192_:
{
return v___x_193_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______elabRules____private__Mathlib__Tactic__ByContra__0__Mathlib__Tactic__ByContra__tacticTry__push__neg__at______1___boxed(lean_object* v_x_196_, lean_object* v_a_197_, lean_object* v_a_198_, lean_object* v_a_199_, lean_object* v_a_200_, lean_object* v_a_201_, lean_object* v_a_202_, lean_object* v_a_203_, lean_object* v_a_204_, lean_object* v_a_205_){
_start:
{
lean_object* v_res_206_; 
v_res_206_ = lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______elabRules____private__Mathlib__Tactic__ByContra__0__Mathlib__Tactic__ByContra__tacticTry__push__neg__at______1(v_x_196_, v_a_197_, v_a_198_, v_a_199_, v_a_200_, v_a_201_, v_a_202_, v_a_203_, v_a_204_);
lean_dec(v_a_204_);
lean_dec_ref(v_a_203_);
lean_dec(v_a_202_);
lean_dec_ref(v_a_201_);
lean_dec(v_a_200_);
lean_dec_ref(v_a_199_);
lean_dec(v_a_198_);
lean_dec_ref(v_a_197_);
return v_res_206_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg___00__closed__4(void){
_start:
{
lean_object* v___x_215_; lean_object* v___x_216_; lean_object* v___x_217_; lean_object* v___x_218_; 
v___x_215_ = l_Lean_Parser_Tactic_optConfig;
v___x_216_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg___00__closed__3));
v___x_217_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__6));
v___x_218_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_218_, 0, v___x_217_);
lean_ctor_set(v___x_218_, 1, v___x_216_);
lean_ctor_set(v___x_218_, 2, v___x_215_);
return v___x_218_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg___00__closed__5(void){
_start:
{
lean_object* v___x_219_; lean_object* v___x_220_; lean_object* v___x_221_; lean_object* v___x_222_; 
v___x_219_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg___00__closed__4, &lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg___00__closed__4_once, _init_lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg___00__closed__4);
v___x_220_ = lean_unsigned_to_nat(1022u);
v___x_221_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg___00__closed__1));
v___x_222_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_222_, 0, v___x_221_);
lean_ctor_set(v___x_222_, 1, v___x_220_);
lean_ctor_set(v___x_222_, 2, v___x_219_);
return v___x_222_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg__(void){
_start:
{
lean_object* v___x_223_; 
v___x_223_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg___00__closed__5, &lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg___00__closed__5_once, _init_lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg___00__closed__5);
return v___x_223_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______elabRules____private__Mathlib__Tactic__ByContra__0__Mathlib__Tactic__ByContra__tacticTry__push__neg____1(lean_object* v_x_226_, lean_object* v_a_227_, lean_object* v_a_228_, lean_object* v_a_229_, lean_object* v_a_230_, lean_object* v_a_231_, lean_object* v_a_232_, lean_object* v_a_233_, lean_object* v_a_234_){
_start:
{
lean_object* v___x_236_; uint8_t v___x_237_; 
v___x_236_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg___00__closed__1));
lean_inc(v_x_226_);
v___x_237_ = l_Lean_Syntax_isOfKind(v_x_226_, v___x_236_);
if (v___x_237_ == 0)
{
lean_object* v___x_238_; 
lean_dec(v_x_226_);
v___x_238_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______elabRules____private__Mathlib__Tactic__ByContra__0__Mathlib__Tactic__ByContra__tacticTry__push__neg__at______1_spec__0___redArg();
return v___x_238_;
}
else
{
lean_object* v___x_239_; lean_object* v___x_240_; uint8_t v___x_241_; lean_object* v___x_242_; 
v___x_239_ = lean_unsigned_to_nat(1u);
v___x_240_ = l_Lean_Syntax_getArg(v_x_226_, v___x_239_);
lean_dec(v_x_226_);
v___x_241_ = 0;
v___x_242_ = lp_mathlib_Mathlib_Tactic_Push_elabPushConfig___redArg(v___x_240_, v___x_241_, v___x_237_, v_a_227_, v_a_233_, v_a_234_);
if (lean_obj_tag(v___x_242_) == 0)
{
lean_object* v_a_243_; lean_object* v___x_244_; lean_object* v___x_245_; lean_object* v___x_246_; lean_object* v___x_247_; uint8_t v___x_248_; uint8_t v___x_249_; lean_object* v___x_250_; 
v_a_243_ = lean_ctor_get(v___x_242_, 0);
lean_inc(v_a_243_);
lean_dec_ref_known(v___x_242_, 1);
v___x_244_ = lean_box(0);
v___x_245_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______elabRules____private__Mathlib__Tactic__ByContra__0__Mathlib__Tactic__ByContra__tacticTry__push__neg__at______1___closed__2));
v___x_246_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______elabRules____private__Mathlib__Tactic__ByContra__0__Mathlib__Tactic__ByContra__tacticTry__push__neg____1___closed__0));
v___x_247_ = lean_alloc_ctor(1, 1, 1);
lean_ctor_set(v___x_247_, 0, v___x_246_);
lean_ctor_set_uint8(v___x_247_, sizeof(void*)*1, v___x_237_);
v___x_248_ = 0;
v___x_249_ = lean_unbox(v_a_243_);
lean_dec(v_a_243_);
v___x_250_ = lp_mathlib_Mathlib_Tactic_Push_push(v___x_249_, v___x_244_, v___x_245_, v___x_247_, v___x_248_, v_a_227_, v_a_228_, v_a_229_, v_a_230_, v_a_231_, v_a_232_, v_a_233_, v_a_234_);
lean_dec_ref_known(v___x_247_, 1);
return v___x_250_;
}
else
{
lean_object* v_a_251_; lean_object* v___x_253_; uint8_t v_isShared_254_; uint8_t v_isSharedCheck_258_; 
v_a_251_ = lean_ctor_get(v___x_242_, 0);
v_isSharedCheck_258_ = !lean_is_exclusive(v___x_242_);
if (v_isSharedCheck_258_ == 0)
{
v___x_253_ = v___x_242_;
v_isShared_254_ = v_isSharedCheck_258_;
goto v_resetjp_252_;
}
else
{
lean_inc(v_a_251_);
lean_dec(v___x_242_);
v___x_253_ = lean_box(0);
v_isShared_254_ = v_isSharedCheck_258_;
goto v_resetjp_252_;
}
v_resetjp_252_:
{
lean_object* v___x_256_; 
if (v_isShared_254_ == 0)
{
v___x_256_ = v___x_253_;
goto v_reusejp_255_;
}
else
{
lean_object* v_reuseFailAlloc_257_; 
v_reuseFailAlloc_257_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_257_, 0, v_a_251_);
v___x_256_ = v_reuseFailAlloc_257_;
goto v_reusejp_255_;
}
v_reusejp_255_:
{
return v___x_256_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______elabRules____private__Mathlib__Tactic__ByContra__0__Mathlib__Tactic__ByContra__tacticTry__push__neg____1___boxed(lean_object* v_x_259_, lean_object* v_a_260_, lean_object* v_a_261_, lean_object* v_a_262_, lean_object* v_a_263_, lean_object* v_a_264_, lean_object* v_a_265_, lean_object* v_a_266_, lean_object* v_a_267_, lean_object* v_a_268_){
_start:
{
lean_object* v_res_269_; 
v_res_269_ = lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______elabRules____private__Mathlib__Tactic__ByContra__0__Mathlib__Tactic__ByContra__tacticTry__push__neg____1(v_x_259_, v_a_260_, v_a_261_, v_a_262_, v_a_263_, v_a_264_, v_a_265_, v_a_266_, v_a_267_);
lean_dec(v_a_267_);
lean_dec_ref(v_a_266_);
lean_dec(v_a_265_);
lean_dec_ref(v_a_264_);
lean_dec(v_a_263_);
lean_dec_ref(v_a_262_);
lean_dec(v_a_261_);
lean_dec_ref(v_a_260_);
return v_res_269_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___lam__0(lean_object* v_____do__lift_270_, lean_object* v___y_271_, lean_object* v___y_272_){
_start:
{
uint8_t v___x_273_; lean_object* v___x_274_; lean_object* v___x_275_; 
v___x_273_ = 0;
v___x_274_ = l_Lean_SourceInfo_fromRef(v_____do__lift_270_, v___x_273_);
v___x_275_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_275_, 0, v___x_274_);
lean_ctor_set(v___x_275_, 1, v___y_272_);
return v___x_275_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___lam__0___boxed(lean_object* v_____do__lift_276_, lean_object* v___y_277_, lean_object* v___y_278_){
_start:
{
lean_object* v_res_279_; 
v_res_279_ = lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___lam__0(v_____do__lift_276_, v___y_277_, v___y_278_);
lean_dec_ref(v___y_277_);
lean_dec(v_____do__lift_276_);
return v_res_279_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__14(void){
_start:
{
lean_object* v___x_298_; lean_object* v___x_299_; 
v___x_298_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__13));
v___x_299_ = l_String_toRawSubstring_x27(v___x_298_);
return v___x_299_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__16(void){
_start:
{
lean_object* v___x_302_; 
v___x_302_ = l_Array_mkArray0(lean_box(0));
return v___x_302_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__41(void){
_start:
{
lean_object* v___x_337_; lean_object* v___x_338_; 
v___x_337_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__40));
v___x_338_ = l_Lean_mkIdent(v___x_337_);
return v___x_338_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1(lean_object* v_x_339_, lean_object* v_a_340_, lean_object* v_a_341_){
_start:
{
lean_object* v___x_342_; lean_object* v___x_343_; uint8_t v___x_344_; 
v___x_342_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__1));
v___x_343_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21___closed__4));
lean_inc(v_x_339_);
v___x_344_ = l_Lean_Syntax_isOfKind(v_x_339_, v___x_343_);
if (v___x_344_ == 0)
{
lean_object* v___x_345_; lean_object* v___x_346_; 
lean_dec(v_x_339_);
v___x_345_ = lean_box(1);
v___x_346_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_346_, 0, v___x_345_);
lean_ctor_set(v___x_346_, 1, v_a_341_);
return v___x_346_;
}
else
{
lean_object* v___x_347_; lean_object* v___x_348_; lean_object* v___x_349_; lean_object* v___y_351_; lean_object* v___y_352_; lean_object* v___y_353_; lean_object* v_replaceTac_354_; lean_object* v_quotContext_355_; lean_object* v_currMacroScope_356_; lean_object* v_ref_357_; lean_object* v___y_358_; lean_object* v___y_433_; lean_object* v___y_434_; lean_object* v___y_435_; lean_object* v___y_436_; lean_object* v_a_437_; lean_object* v_a_438_; lean_object* v___y_521_; lean_object* v_ty_x3f_522_; lean_object* v___y_523_; lean_object* v___y_524_; lean_object* v___x_539_; lean_object* v_pat_x3f_541_; lean_object* v___y_542_; lean_object* v___y_543_; lean_object* v___x_553_; uint8_t v___x_554_; 
v___x_347_ = lean_unsigned_to_nat(0u);
v___x_348_ = lean_unsigned_to_nat(1u);
v___x_349_ = l_Lean_Syntax_getArg(v_x_339_, v___x_348_);
v___x_539_ = lean_unsigned_to_nat(2u);
v___x_553_ = l_Lean_Syntax_getArg(v_x_339_, v___x_539_);
v___x_554_ = l_Lean_Syntax_isNone(v___x_553_);
if (v___x_554_ == 0)
{
uint8_t v___x_555_; 
lean_inc(v___x_553_);
v___x_555_ = l_Lean_Syntax_matchesNull(v___x_553_, v___x_348_);
if (v___x_555_ == 0)
{
lean_object* v___x_556_; lean_object* v___x_557_; 
lean_dec(v___x_553_);
lean_dec(v___x_349_);
lean_dec(v_x_339_);
v___x_556_ = lean_box(1);
v___x_557_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_557_, 0, v___x_556_);
lean_ctor_set(v___x_557_, 1, v_a_341_);
return v___x_557_;
}
else
{
lean_object* v_pat_x3f_558_; lean_object* v___x_559_; 
v_pat_x3f_558_ = l_Lean_Syntax_getArg(v___x_553_, v___x_347_);
lean_dec(v___x_553_);
v___x_559_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_559_, 0, v_pat_x3f_558_);
v_pat_x3f_541_ = v___x_559_;
v___y_542_ = v_a_340_;
v___y_543_ = v_a_341_;
goto v___jp_540_;
}
}
else
{
lean_object* v___x_560_; 
lean_dec(v___x_553_);
v___x_560_ = lean_box(0);
v_pat_x3f_541_ = v___x_560_;
v___y_542_ = v_a_340_;
v___y_543_ = v_a_341_;
goto v___jp_540_;
}
v___jp_350_:
{
uint8_t v___x_359_; lean_object* v___x_360_; lean_object* v___x_361_; lean_object* v___x_362_; lean_object* v___x_363_; lean_object* v___x_364_; lean_object* v___x_365_; lean_object* v___x_366_; lean_object* v___x_367_; lean_object* v___x_368_; lean_object* v___x_369_; lean_object* v___x_370_; lean_object* v___x_371_; lean_object* v___x_372_; lean_object* v___x_373_; lean_object* v___x_374_; lean_object* v___x_375_; lean_object* v___x_376_; lean_object* v___x_377_; lean_object* v___x_378_; lean_object* v___x_379_; lean_object* v___x_380_; lean_object* v___x_381_; lean_object* v___x_382_; lean_object* v___x_383_; lean_object* v___x_384_; lean_object* v___x_385_; lean_object* v___x_386_; lean_object* v___x_387_; lean_object* v___x_388_; lean_object* v___x_389_; lean_object* v___x_390_; lean_object* v___x_391_; lean_object* v___x_392_; lean_object* v___x_393_; lean_object* v___x_394_; lean_object* v___x_395_; lean_object* v___x_396_; lean_object* v___x_397_; lean_object* v___x_398_; lean_object* v___x_399_; lean_object* v___x_400_; lean_object* v___x_401_; lean_object* v___x_402_; lean_object* v___x_403_; lean_object* v___x_404_; lean_object* v___x_405_; lean_object* v___x_406_; lean_object* v___x_407_; lean_object* v___x_408_; lean_object* v___x_409_; lean_object* v___x_410_; lean_object* v___x_411_; lean_object* v___x_412_; lean_object* v___x_413_; lean_object* v___x_414_; lean_object* v___x_415_; lean_object* v___x_416_; lean_object* v___x_417_; lean_object* v___x_418_; lean_object* v___x_419_; lean_object* v___x_420_; lean_object* v___x_421_; lean_object* v___x_422_; lean_object* v___x_423_; lean_object* v___x_424_; lean_object* v___x_425_; lean_object* v___x_426_; lean_object* v___x_427_; lean_object* v___x_428_; lean_object* v___x_429_; lean_object* v___x_430_; lean_object* v___x_431_; 
v___x_359_ = 0;
v___x_360_ = l_Lean_SourceInfo_fromRef(v_ref_357_, v___x_359_);
v___x_361_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__0));
lean_inc_ref_n(v___y_353_, 10);
lean_inc_ref_n(v___y_352_, 10);
v___x_362_ = l_Lean_Name_mkStr4(v___y_352_, v___y_353_, v___x_342_, v___x_361_);
v___x_363_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__1));
lean_inc_n(v___x_360_, 25);
v___x_364_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_364_, 0, v___x_360_);
lean_ctor_set(v___x_364_, 1, v___x_363_);
v___x_365_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__2));
v___x_366_ = l_Lean_Name_mkStr4(v___y_352_, v___y_353_, v___x_342_, v___x_365_);
v___x_367_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__3));
v___x_368_ = l_Lean_Name_mkStr4(v___y_352_, v___y_353_, v___x_342_, v___x_367_);
v___x_369_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__5));
v___x_370_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__8));
v___x_371_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__9));
v___x_372_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_372_, 0, v___x_360_);
lean_ctor_set(v___x_372_, 1, v___x_371_);
v___x_373_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__10));
v___x_374_ = l_Lean_Name_mkStr4(v___y_352_, v___y_353_, v___x_342_, v___x_373_);
v___x_375_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__11));
v___x_376_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__12));
v___x_377_ = l_Lean_Name_mkStr5(v___y_352_, v___y_353_, v___x_342_, v___x_375_, v___x_376_);
v___x_378_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__14, &lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__14_once, _init_lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__14);
v___x_379_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__15));
v___x_380_ = l_Lean_addMacroScope(v_quotContext_355_, v___x_379_, v_currMacroScope_356_);
v___x_381_ = lean_box(0);
v___x_382_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_382_, 0, v___x_360_);
lean_ctor_set(v___x_382_, 1, v___x_378_);
lean_ctor_set(v___x_382_, 2, v___x_380_);
lean_ctor_set(v___x_382_, 3, v___x_381_);
lean_inc_ref_n(v___x_382_, 2);
v___x_383_ = l_Lean_Syntax_node1(v___x_360_, v___x_377_, v___x_382_);
v___x_384_ = l_Lean_Syntax_node1(v___x_360_, v___x_369_, v___x_383_);
v___x_385_ = l_Lean_Syntax_node1(v___x_360_, v___x_374_, v___x_384_);
v___x_386_ = l_Lean_Syntax_node1(v___x_360_, v___x_369_, v___x_385_);
v___x_387_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__16, &lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__16_once, _init_lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__16);
v___x_388_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_388_, 0, v___x_360_);
lean_ctor_set(v___x_388_, 1, v___x_369_);
lean_ctor_set(v___x_388_, 2, v___x_387_);
lean_inc_ref_n(v___x_388_, 2);
v___x_389_ = l_Lean_Syntax_node3(v___x_360_, v___x_370_, v___x_372_, v___x_386_, v___x_388_);
v___x_390_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__17));
v___x_391_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_391_, 0, v___x_360_);
lean_ctor_set(v___x_391_, 1, v___x_390_);
v___x_392_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg__at_____00__closed__10));
v___x_393_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg__at_____00__closed__11));
v___x_394_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_394_, 0, v___x_360_);
lean_ctor_set(v___x_394_, 1, v___x_393_);
v___x_395_ = l_Lean_Syntax_node3(v___x_360_, v___x_392_, v___x_394_, v___x_349_, v___x_382_);
v___x_396_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__18));
v___x_397_ = l_Lean_Name_mkStr4(v___y_352_, v___y_353_, v___x_342_, v___x_396_);
v___x_398_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_398_, 0, v___x_360_);
lean_ctor_set(v___x_398_, 1, v___x_396_);
v___x_399_ = l_Lean_Syntax_node1(v___x_360_, v___x_369_, v___x_382_);
v___x_400_ = l_Lean_Syntax_node2(v___x_360_, v___x_397_, v___x_398_, v___x_399_);
v___x_401_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__19));
v___x_402_ = l_Lean_Name_mkStr4(v___y_352_, v___y_353_, v___x_342_, v___x_401_);
v___x_403_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_403_, 0, v___x_360_);
lean_ctor_set(v___x_403_, 1, v___x_401_);
v___x_404_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__20));
v___x_405_ = l_Lean_Name_mkStr5(v___y_352_, v___y_353_, v___x_342_, v___x_404_, v___x_376_);
v___x_406_ = l_Lean_Name_mkStr5(v___y_352_, v___y_353_, v___x_342_, v___x_375_, v___x_361_);
v___x_407_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__21));
v___x_408_ = l_Lean_Name_mkStr4(v___y_352_, v___y_353_, v___x_342_, v___x_407_);
v___x_409_ = l_Lean_Syntax_node2(v___x_360_, v___x_408_, v___y_351_, v___x_388_);
v___x_410_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__22));
v___x_411_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_411_, 0, v___x_360_);
lean_ctor_set(v___x_411_, 1, v___x_410_);
lean_inc_ref(v___x_411_);
lean_inc_ref(v___x_364_);
v___x_412_ = l_Lean_Syntax_node3(v___x_360_, v___x_406_, v___x_364_, v___x_409_, v___x_411_);
v___x_413_ = l_Lean_Syntax_node1(v___x_360_, v___x_405_, v___x_412_);
v___x_414_ = l_Lean_Syntax_node1(v___x_360_, v___x_369_, v___x_413_);
v___x_415_ = l_Lean_Syntax_node3(v___x_360_, v___x_402_, v___x_403_, v___x_414_, v___x_388_);
v___x_416_ = lean_unsigned_to_nat(9u);
v___x_417_ = lean_mk_empty_array_with_capacity(v___x_416_);
v___x_418_ = lean_array_push(v___x_417_, v___x_389_);
lean_inc_ref_n(v___x_391_, 3);
v___x_419_ = lean_array_push(v___x_418_, v___x_391_);
v___x_420_ = lean_array_push(v___x_419_, v___x_395_);
v___x_421_ = lean_array_push(v___x_420_, v___x_391_);
v___x_422_ = lean_array_push(v___x_421_, v_replaceTac_354_);
v___x_423_ = lean_array_push(v___x_422_, v___x_391_);
v___x_424_ = lean_array_push(v___x_423_, v___x_400_);
v___x_425_ = lean_array_push(v___x_424_, v___x_391_);
v___x_426_ = lean_array_push(v___x_425_, v___x_415_);
v___x_427_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_427_, 0, v___x_360_);
lean_ctor_set(v___x_427_, 1, v___x_369_);
lean_ctor_set(v___x_427_, 2, v___x_426_);
v___x_428_ = l_Lean_Syntax_node1(v___x_360_, v___x_368_, v___x_427_);
v___x_429_ = l_Lean_Syntax_node1(v___x_360_, v___x_366_, v___x_428_);
v___x_430_ = l_Lean_Syntax_node3(v___x_360_, v___x_362_, v___x_364_, v___x_429_, v___x_411_);
v___x_431_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_431_, 0, v___x_430_);
lean_ctor_set(v___x_431_, 1, v___y_358_);
return v___x_431_;
}
v___jp_432_:
{
if (lean_obj_tag(v___y_434_) == 0)
{
lean_object* v_quotContext_439_; lean_object* v_currMacroScope_440_; lean_object* v_ref_441_; lean_object* v___x_442_; lean_object* v_a_443_; lean_object* v_a_444_; lean_object* v___x_446_; uint8_t v_isShared_447_; uint8_t v_isSharedCheck_454_; 
v_quotContext_439_ = lean_ctor_get(v___y_436_, 1);
v_currMacroScope_440_ = lean_ctor_get(v___y_436_, 2);
v_ref_441_ = lean_ctor_get(v___y_436_, 5);
v___x_442_ = lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___lam__0(v_ref_441_, v___y_436_, v_a_438_);
v_a_443_ = lean_ctor_get(v___x_442_, 0);
v_a_444_ = lean_ctor_get(v___x_442_, 1);
v_isSharedCheck_454_ = !lean_is_exclusive(v___x_442_);
if (v_isSharedCheck_454_ == 0)
{
v___x_446_ = v___x_442_;
v_isShared_447_ = v_isSharedCheck_454_;
goto v_resetjp_445_;
}
else
{
lean_inc(v_a_444_);
lean_inc(v_a_443_);
lean_dec(v___x_442_);
v___x_446_ = lean_box(0);
v_isShared_447_ = v_isSharedCheck_454_;
goto v_resetjp_445_;
}
v_resetjp_445_:
{
lean_object* v___x_448_; lean_object* v___x_449_; lean_object* v___x_451_; 
v___x_448_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__23));
lean_inc_ref(v___y_435_);
lean_inc_ref(v___y_433_);
v___x_449_ = l_Lean_Name_mkStr4(v___y_433_, v___y_435_, v___x_342_, v___x_448_);
lean_inc(v_a_443_);
if (v_isShared_447_ == 0)
{
lean_ctor_set_tag(v___x_446_, 2);
lean_ctor_set(v___x_446_, 1, v___x_448_);
v___x_451_ = v___x_446_;
goto v_reusejp_450_;
}
else
{
lean_object* v_reuseFailAlloc_453_; 
v_reuseFailAlloc_453_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v_reuseFailAlloc_453_, 0, v_a_443_);
lean_ctor_set(v_reuseFailAlloc_453_, 1, v___x_448_);
v___x_451_ = v_reuseFailAlloc_453_;
goto v_reusejp_450_;
}
v_reusejp_450_:
{
lean_object* v___x_452_; 
v___x_452_ = l_Lean_Syntax_node1(v_a_443_, v___x_449_, v___x_451_);
lean_inc(v_currMacroScope_440_);
lean_inc(v_quotContext_439_);
v___y_351_ = v_a_437_;
v___y_352_ = v___y_433_;
v___y_353_ = v___y_435_;
v_replaceTac_354_ = v___x_452_;
v_quotContext_355_ = v_quotContext_439_;
v_currMacroScope_356_ = v_currMacroScope_440_;
v_ref_357_ = v_ref_441_;
v___y_358_ = v_a_444_;
goto v___jp_350_;
}
}
}
else
{
lean_object* v_val_455_; lean_object* v_quotContext_456_; lean_object* v_currMacroScope_457_; lean_object* v_ref_458_; lean_object* v___x_459_; lean_object* v_a_460_; lean_object* v_a_461_; lean_object* v___x_463_; uint8_t v_isShared_464_; uint8_t v_isSharedCheck_519_; 
v_val_455_ = lean_ctor_get(v___y_434_, 0);
lean_inc(v_val_455_);
lean_dec_ref_known(v___y_434_, 1);
v_quotContext_456_ = lean_ctor_get(v___y_436_, 1);
v_currMacroScope_457_ = lean_ctor_get(v___y_436_, 2);
v_ref_458_ = lean_ctor_get(v___y_436_, 5);
v___x_459_ = lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___lam__0(v_ref_458_, v___y_436_, v_a_438_);
v_a_460_ = lean_ctor_get(v___x_459_, 0);
v_a_461_ = lean_ctor_get(v___x_459_, 1);
v_isSharedCheck_519_ = !lean_is_exclusive(v___x_459_);
if (v_isSharedCheck_519_ == 0)
{
v___x_463_ = v___x_459_;
v_isShared_464_ = v_isSharedCheck_519_;
goto v_resetjp_462_;
}
else
{
lean_inc(v_a_461_);
lean_inc(v_a_460_);
lean_dec(v___x_459_);
v___x_463_ = lean_box(0);
v_isShared_464_ = v_isSharedCheck_519_;
goto v_resetjp_462_;
}
v_resetjp_462_:
{
lean_object* v___x_465_; lean_object* v___x_466_; lean_object* v___x_468_; 
v___x_465_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__24));
lean_inc_ref(v___y_435_);
lean_inc_ref(v___y_433_);
v___x_466_ = l_Lean_Name_mkStr4(v___y_433_, v___y_435_, v___x_342_, v___x_465_);
lean_inc(v_a_460_);
if (v_isShared_464_ == 0)
{
lean_ctor_set_tag(v___x_463_, 2);
lean_ctor_set(v___x_463_, 1, v___x_465_);
v___x_468_ = v___x_463_;
goto v_reusejp_467_;
}
else
{
lean_object* v_reuseFailAlloc_518_; 
v_reuseFailAlloc_518_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v_reuseFailAlloc_518_, 0, v_a_460_);
lean_ctor_set(v_reuseFailAlloc_518_, 1, v___x_465_);
v___x_468_ = v_reuseFailAlloc_518_;
goto v_reusejp_467_;
}
v_reusejp_467_:
{
lean_object* v___x_469_; lean_object* v___x_470_; lean_object* v___x_471_; lean_object* v___x_472_; lean_object* v___x_473_; lean_object* v___x_474_; lean_object* v___x_475_; lean_object* v___x_476_; lean_object* v___x_477_; lean_object* v___x_478_; lean_object* v___x_479_; lean_object* v___x_480_; lean_object* v___x_481_; lean_object* v___x_482_; lean_object* v___x_483_; lean_object* v___x_484_; lean_object* v___x_485_; lean_object* v___x_486_; lean_object* v___x_487_; lean_object* v___x_488_; lean_object* v___x_489_; lean_object* v___x_490_; lean_object* v___x_491_; lean_object* v___x_492_; lean_object* v___x_493_; lean_object* v___x_494_; lean_object* v___x_495_; lean_object* v___x_496_; lean_object* v___x_497_; lean_object* v___x_498_; lean_object* v___x_499_; lean_object* v___x_500_; lean_object* v___x_501_; lean_object* v___x_502_; lean_object* v___x_503_; lean_object* v___x_504_; lean_object* v___x_505_; lean_object* v___x_506_; lean_object* v___x_507_; lean_object* v___x_508_; lean_object* v___x_509_; lean_object* v___x_510_; lean_object* v___x_511_; lean_object* v___x_512_; lean_object* v___x_513_; lean_object* v___x_514_; lean_object* v___x_515_; lean_object* v___x_516_; lean_object* v___x_517_; 
v___x_469_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__25));
v___x_470_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__26));
lean_inc_ref_n(v___y_435_, 8);
lean_inc_ref_n(v___y_433_, 8);
v___x_471_ = l_Lean_Name_mkStr4(v___y_433_, v___y_435_, v___x_469_, v___x_470_);
v___x_472_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__27));
v___x_473_ = l_Lean_Name_mkStr4(v___y_433_, v___y_435_, v___x_469_, v___x_472_);
v___x_474_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__28));
v___x_475_ = l_Lean_Name_mkStr4(v___y_433_, v___y_435_, v___x_469_, v___x_474_);
v___x_476_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__14, &lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__14_once, _init_lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__14);
v___x_477_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__15));
lean_inc_n(v_currMacroScope_457_, 2);
lean_inc_n(v_quotContext_456_, 2);
v___x_478_ = l_Lean_addMacroScope(v_quotContext_456_, v___x_477_, v_currMacroScope_457_);
v___x_479_ = lean_box(0);
lean_inc_n(v_a_460_, 19);
v___x_480_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_480_, 0, v_a_460_);
lean_ctor_set(v___x_480_, 1, v___x_476_);
lean_ctor_set(v___x_480_, 2, v___x_478_);
lean_ctor_set(v___x_480_, 3, v___x_479_);
lean_inc_ref(v___x_480_);
v___x_481_ = l_Lean_Syntax_node1(v_a_460_, v___x_475_, v___x_480_);
v___x_482_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__5));
v___x_483_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__16, &lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__16_once, _init_lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__16);
v___x_484_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_484_, 0, v_a_460_);
lean_ctor_set(v___x_484_, 1, v___x_482_);
lean_ctor_set(v___x_484_, 2, v___x_483_);
v___x_485_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__29));
v___x_486_ = l_Lean_Name_mkStr4(v___y_433_, v___y_435_, v___x_469_, v___x_485_);
v___x_487_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__30));
v___x_488_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_488_, 0, v_a_460_);
lean_ctor_set(v___x_488_, 1, v___x_487_);
v___x_489_ = l_Lean_Syntax_node2(v_a_460_, v___x_486_, v___x_488_, v_val_455_);
v___x_490_ = l_Lean_Syntax_node1(v_a_460_, v___x_482_, v___x_489_);
v___x_491_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__31));
v___x_492_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_492_, 0, v_a_460_);
lean_ctor_set(v___x_492_, 1, v___x_491_);
v___x_493_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__32));
v___x_494_ = l_Lean_Name_mkStr4(v___y_433_, v___y_435_, v___x_469_, v___x_493_);
v___x_495_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__33));
v___x_496_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_496_, 0, v_a_460_);
lean_ctor_set(v___x_496_, 1, v___x_495_);
v___x_497_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__2));
v___x_498_ = l_Lean_Name_mkStr4(v___y_433_, v___y_435_, v___x_342_, v___x_497_);
v___x_499_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__3));
v___x_500_ = l_Lean_Name_mkStr4(v___y_433_, v___y_435_, v___x_342_, v___x_499_);
v___x_501_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg___00__closed__1));
v___x_502_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg___00__closed__2));
v___x_503_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_503_, 0, v_a_460_);
lean_ctor_set(v___x_503_, 1, v___x_502_);
lean_inc(v___x_349_);
v___x_504_ = l_Lean_Syntax_node2(v_a_460_, v___x_501_, v___x_503_, v___x_349_);
v___x_505_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__17));
v___x_506_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_506_, 0, v_a_460_);
lean_ctor_set(v___x_506_, 1, v___x_505_);
v___x_507_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__34));
v___x_508_ = l_Lean_Name_mkStr4(v___y_433_, v___y_435_, v___x_342_, v___x_507_);
v___x_509_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_509_, 0, v_a_460_);
lean_ctor_set(v___x_509_, 1, v___x_507_);
v___x_510_ = l_Lean_Syntax_node2(v_a_460_, v___x_508_, v___x_509_, v___x_480_);
v___x_511_ = l_Lean_Syntax_node3(v_a_460_, v___x_482_, v___x_504_, v___x_506_, v___x_510_);
v___x_512_ = l_Lean_Syntax_node1(v_a_460_, v___x_500_, v___x_511_);
v___x_513_ = l_Lean_Syntax_node1(v_a_460_, v___x_498_, v___x_512_);
v___x_514_ = l_Lean_Syntax_node2(v_a_460_, v___x_494_, v___x_496_, v___x_513_);
v___x_515_ = l_Lean_Syntax_node5(v_a_460_, v___x_473_, v___x_481_, v___x_484_, v___x_490_, v___x_492_, v___x_514_);
v___x_516_ = l_Lean_Syntax_node1(v_a_460_, v___x_471_, v___x_515_);
v___x_517_ = l_Lean_Syntax_node2(v_a_460_, v___x_466_, v___x_468_, v___x_516_);
v___y_351_ = v_a_437_;
v___y_352_ = v___y_433_;
v___y_353_ = v___y_435_;
v_replaceTac_354_ = v___x_517_;
v_quotContext_355_ = v_quotContext_456_;
v_currMacroScope_356_ = v_currMacroScope_457_;
v_ref_357_ = v_ref_458_;
v___y_358_ = v_a_461_;
goto v___jp_350_;
}
}
}
}
v___jp_520_:
{
lean_object* v___x_525_; lean_object* v___x_526_; 
v___x_525_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__35));
v___x_526_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__36));
if (lean_obj_tag(v___y_521_) == 0)
{
lean_object* v_ref_527_; lean_object* v___x_528_; lean_object* v_a_529_; lean_object* v_a_530_; lean_object* v___x_531_; lean_object* v___x_532_; lean_object* v___x_533_; lean_object* v___x_534_; lean_object* v___x_535_; lean_object* v___x_536_; lean_object* v___x_537_; 
v_ref_527_ = lean_ctor_get(v___y_523_, 5);
v___x_528_ = lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___lam__0(v_ref_527_, v___y_523_, v___y_524_);
v_a_529_ = lean_ctor_get(v___x_528_, 0);
lean_inc_n(v_a_529_, 3);
v_a_530_ = lean_ctor_get(v___x_528_, 1);
lean_inc(v_a_530_);
lean_dec_ref(v___x_528_);
v___x_531_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__37));
v___x_532_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__5));
v___x_533_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__38));
v___x_534_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__41, &lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__41_once, _init_lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___closed__41);
v___x_535_ = l_Lean_Syntax_node1(v_a_529_, v___x_533_, v___x_534_);
v___x_536_ = l_Lean_Syntax_node1(v_a_529_, v___x_532_, v___x_535_);
v___x_537_ = l_Lean_Syntax_node1(v_a_529_, v___x_531_, v___x_536_);
v___y_433_ = v___x_525_;
v___y_434_ = v_ty_x3f_522_;
v___y_435_ = v___x_526_;
v___y_436_ = v___y_523_;
v_a_437_ = v___x_537_;
v_a_438_ = v_a_530_;
goto v___jp_432_;
}
else
{
lean_object* v_val_538_; 
v_val_538_ = lean_ctor_get(v___y_521_, 0);
lean_inc(v_val_538_);
lean_dec_ref_known(v___y_521_, 1);
v___y_433_ = v___x_525_;
v___y_434_ = v_ty_x3f_522_;
v___y_435_ = v___x_526_;
v___y_436_ = v___y_523_;
v_a_437_ = v_val_538_;
v_a_438_ = v___y_524_;
goto v___jp_432_;
}
}
v___jp_540_:
{
lean_object* v___x_544_; lean_object* v___x_545_; uint8_t v___x_546_; 
v___x_544_ = lean_unsigned_to_nat(3u);
v___x_545_ = l_Lean_Syntax_getArg(v_x_339_, v___x_544_);
lean_dec(v_x_339_);
v___x_546_ = l_Lean_Syntax_isNone(v___x_545_);
if (v___x_546_ == 0)
{
uint8_t v___x_547_; 
lean_inc(v___x_545_);
v___x_547_ = l_Lean_Syntax_matchesNull(v___x_545_, v___x_539_);
if (v___x_547_ == 0)
{
lean_object* v___x_548_; lean_object* v___x_549_; 
lean_dec(v___x_545_);
lean_dec(v_pat_x3f_541_);
lean_dec(v___x_349_);
v___x_548_ = lean_box(1);
v___x_549_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_549_, 0, v___x_548_);
lean_ctor_set(v___x_549_, 1, v___y_543_);
return v___x_549_;
}
else
{
lean_object* v_ty_x3f_550_; lean_object* v___x_551_; 
v_ty_x3f_550_ = l_Lean_Syntax_getArg(v___x_545_, v___x_348_);
lean_dec(v___x_545_);
v___x_551_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_551_, 0, v_ty_x3f_550_);
v___y_521_ = v_pat_x3f_541_;
v_ty_x3f_522_ = v___x_551_;
v___y_523_ = v___y_542_;
v___y_524_ = v___y_543_;
goto v___jp_520_;
}
}
else
{
lean_object* v___x_552_; 
lean_dec(v___x_545_);
v___x_552_ = lean_box(0);
v___y_521_ = v_pat_x3f_541_;
v_ty_x3f_522_ = v___x_552_;
v___y_523_ = v___y_542_;
v___y_524_ = v___y_543_;
goto v___jp_520_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1___boxed(lean_object* v_x_561_, lean_object* v_a_562_, lean_object* v_a_563_){
_start:
{
lean_object* v_res_564_; 
v_res_564_ = lp_mathlib_Mathlib_Tactic_ByContra___aux__Mathlib__Tactic__ByContra______macroRules__Mathlib__Tactic__ByContra__byContra_x21__1(v_x_561_, v_a_562_, v_a_563_);
lean_dec_ref(v_a_562_);
return v_res_564_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Tactic_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Push(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_ByContra(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Tactic_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Push(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_ByContra(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21 = _init_lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_ByContra_byContra_x21);
lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg__at____ = _init_lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg__at____();
lean_mark_persistent(lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg__at____);
lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg__ = _init_lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg__();
lean_mark_persistent(lp_mathlib___private_Mathlib_Tactic_ByContra_0__Mathlib_Tactic_ByContra_tacticTry__push__neg__);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Tactic_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Push(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_ByContra(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Tactic_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Push(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_ByContra(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_ByContra(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_ByContra(builtin);
}
#ifdef __cplusplus
}
#endif
