// Lean compiler output
// Module: Mathlib.Tactic.Set
// Imports: public import Init public meta import Init public import Mathlib.Init public meta import Lean.Elab.Tactic.ElabTerm
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
extern lean_object* l_Lean_binderIdent;
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* l_Lean_Elab_Term_exprToSyntax(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Array_mkArray0(lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_evalTactic(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_getMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_TSyntax_getId(lean_object*);
lean_object* l_Lean_MVarId_define(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_intro1Core(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_replaceMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_withMainContext___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkFVar(lean_object*);
lean_object* l_Lean_Elab_Term_addTermInfo_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_elabTerm(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_infer_type(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_elabType(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_elabTermEnsuringType(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isNone(lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_setArgsRest___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "setArgsRest"};
static const lean_object* lp_mathlib_Mathlib_Tactic_setArgsRest___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_setArgsRest___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_setArgsRest___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib_Mathlib_Tactic_setArgsRest___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_setArgsRest___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_setArgsRest___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_Mathlib_Tactic_setArgsRest___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_setArgsRest___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_setArgsRest___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_setArgsRest___closed__1_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_setArgsRest___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_setArgsRest___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_setArgsRest___closed__2_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_setArgsRest___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_setArgsRest___closed__3_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_setArgsRest___closed__0_value),LEAN_SCALAR_PTR_LITERAL(48, 171, 42, 104, 252, 222, 191, 134)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_setArgsRest___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_setArgsRest___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_setArgsRest___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Mathlib_Tactic_setArgsRest___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_setArgsRest___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_setArgsRest___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_setArgsRest___closed__4_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_setArgsRest___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_setArgsRest___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_setArgsRest___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "ppSpace"};
static const lean_object* lp_mathlib_Mathlib_Tactic_setArgsRest___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_setArgsRest___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_setArgsRest___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_setArgsRest___closed__6_value),LEAN_SCALAR_PTR_LITERAL(207, 47, 58, 43, 30, 240, 125, 246)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_setArgsRest___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_setArgsRest___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_setArgsRest___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_setArgsRest___closed__7_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_setArgsRest___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_setArgsRest___closed__8_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_setArgsRest___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_setArgsRest___closed__9;
static const lean_string_object lp_mathlib_Mathlib_Tactic_setArgsRest___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "optional"};
static const lean_object* lp_mathlib_Mathlib_Tactic_setArgsRest___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_setArgsRest___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_setArgsRest___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_setArgsRest___closed__10_value),LEAN_SCALAR_PTR_LITERAL(233, 141, 154, 50, 143, 135, 42, 252)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_setArgsRest___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_setArgsRest___closed__11_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_setArgsRest___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = " : "};
static const lean_object* lp_mathlib_Mathlib_Tactic_setArgsRest___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_setArgsRest___closed__12_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_setArgsRest___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_setArgsRest___closed__12_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_setArgsRest___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_setArgsRest___closed__13_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_setArgsRest___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_Mathlib_Tactic_setArgsRest___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_setArgsRest___closed__14_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_setArgsRest___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_setArgsRest___closed__14_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_setArgsRest___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_setArgsRest___closed__15_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_setArgsRest___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_setArgsRest___closed__15_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_setArgsRest___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_setArgsRest___closed__16_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_setArgsRest___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_setArgsRest___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_setArgsRest___closed__13_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_setArgsRest___closed__16_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_setArgsRest___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_setArgsRest___closed__17_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_setArgsRest___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_setArgsRest___closed__11_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_setArgsRest___closed__17_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_setArgsRest___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_setArgsRest___closed__18_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_setArgsRest___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_setArgsRest___closed__19;
static const lean_string_object lp_mathlib_Mathlib_Tactic_setArgsRest___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = " := "};
static const lean_object* lp_mathlib_Mathlib_Tactic_setArgsRest___closed__20 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_setArgsRest___closed__20_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_setArgsRest___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_setArgsRest___closed__20_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_setArgsRest___closed__21 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_setArgsRest___closed__21_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_setArgsRest___closed__22_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_setArgsRest___closed__22;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_setArgsRest___closed__23_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_setArgsRest___closed__23;
static const lean_string_object lp_mathlib_Mathlib_Tactic_setArgsRest___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = " with "};
static const lean_object* lp_mathlib_Mathlib_Tactic_setArgsRest___closed__24 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_setArgsRest___closed__24_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_setArgsRest___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_setArgsRest___closed__24_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_setArgsRest___closed__25 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_setArgsRest___closed__25_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_setArgsRest___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 2, .m_data = "← "};
static const lean_object* lp_mathlib_Mathlib_Tactic_setArgsRest___closed__26 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_setArgsRest___closed__26_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_setArgsRest___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_setArgsRest___closed__26_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_setArgsRest___closed__27 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_setArgsRest___closed__27_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_setArgsRest___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_setArgsRest___closed__11_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_setArgsRest___closed__27_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_setArgsRest___closed__28 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_setArgsRest___closed__28_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_setArgsRest___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_setArgsRest___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_setArgsRest___closed__25_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_setArgsRest___closed__28_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_setArgsRest___closed__29 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_setArgsRest___closed__29_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_setArgsRest___closed__30_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_setArgsRest___closed__30;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_setArgsRest___closed__31_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_setArgsRest___closed__31;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_setArgsRest___closed__32_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_setArgsRest___closed__32;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_setArgsRest___closed__33_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_setArgsRest___closed__33;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_setArgsRest;
static const lean_string_object lp_mathlib_Mathlib_Tactic_setTactic___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "setTactic"};
static const lean_object* lp_mathlib_Mathlib_Tactic_setTactic___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_setTactic___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_setTactic___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_setArgsRest___closed__1_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_setTactic___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_setTactic___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_setArgsRest___closed__2_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_setTactic___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_setTactic___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_setTactic___closed__0_value),LEAN_SCALAR_PTR_LITERAL(81, 213, 240, 6, 67, 117, 195, 74)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_setTactic___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_setTactic___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_setTactic___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "set"};
static const lean_object* lp_mathlib_Mathlib_Tactic_setTactic___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_setTactic___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_setTactic___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_setTactic___closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_setTactic___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_setTactic___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_setTactic___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "!"};
static const lean_object* lp_mathlib_Mathlib_Tactic_setTactic___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_setTactic___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_setTactic___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_setTactic___closed__4_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_setTactic___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_setTactic___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_setTactic___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_setArgsRest___closed__11_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_setTactic___closed__5_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_setTactic___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_setTactic___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_setTactic___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_setArgsRest___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_setTactic___closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_setTactic___closed__6_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_setTactic___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_setTactic___closed__7_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_setTactic___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_setTactic___closed__8;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_setTactic___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_setTactic___closed__9;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_setTactic;
static const lean_string_object lp_mathlib_Mathlib_Tactic_tacticSet_x21___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "tacticSet!_"};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticSet_x21___00__closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSet_x21___00__closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticSet_x21___00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_setArgsRest___closed__1_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticSet_x21___00__closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSet_x21___00__closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_setArgsRest___closed__2_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticSet_x21___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSet_x21___00__closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSet_x21___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(94, 189, 205, 110, 125, 123, 91, 174)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticSet_x21___00__closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSet_x21___00__closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_tacticSet_x21___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "set!"};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticSet_x21___00__closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSet_x21___00__closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticSet_x21___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSet_x21___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticSet_x21___00__closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSet_x21___00__closed__3_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_tacticSet_x21___00__closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_tacticSet_x21___00__closed__4;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_tacticSet_x21___00__closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_tacticSet_x21___00__closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_tacticSet_x21__;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______macroRules__Mathlib__Tactic__tacticSet_x21____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______macroRules__Mathlib__Tactic__tacticSet_x21____1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______macroRules__Mathlib__Tactic__tacticSet_x21____1___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______macroRules__Mathlib__Tactic__tacticSet_x21____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______macroRules__Mathlib__Tactic__tacticSet_x21____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______macroRules__Mathlib__Tactic__tacticSet_x21____1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______macroRules__Mathlib__Tactic__tacticSet_x21____1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______macroRules__Mathlib__Tactic__tacticSet_x21____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______macroRules__Mathlib__Tactic__tacticSet_x21____1___boxed(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__0(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "tacticHave__"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "have"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "letConfig"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__4_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__5;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "letDecl"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "letIdDecl"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "letId"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__8_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "typeSpec"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ":"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__10_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "term_=_"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__11_value),LEAN_SCALAR_PTR_LITERAL(167, 251, 107, 62, 223, 239, 203, 78)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__12_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "="};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__13_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "typeAscription"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__14_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "hygienicLParen"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__15_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "("};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__16_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "hygieneInfo"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__17_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__17_value),LEAN_SCALAR_PTR_LITERAL(27, 64, 36, 144, 170, 151, 255, 136)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__18_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__19_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__20_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__20;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Meta"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__21 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__21_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Elab"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__22 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__22_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ")"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__23 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__23_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ":="};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__24 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__24_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "rfl"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__25 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__25_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__26_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__26;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__25_value),LEAN_SCALAR_PTR_LITERAL(77, 42, 253, 71, 61, 132, 173, 240)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__27 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__27_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__27_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__28 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__28_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__28_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__29 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__29_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "tacticTry_"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__30 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__30_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "try"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__31 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__31_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tacticSeq"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__32 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__32_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__33_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tacticSeq1Indented"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__33 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__33_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__34_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "rewriteSeq"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__34 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__34_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "rewrite"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__35 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__35_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__36_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "optConfig"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__36 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__36_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__37_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "rwRuleSeq"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__37 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__37_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__38_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "["};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__38 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__38_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__39_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "rwRule"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__39 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__39_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__40_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "show"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__40 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__40_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__41_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "fromTerm"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__41 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__41_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__42_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "from"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__42 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__42_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__43_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__27_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__43 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__43_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__44_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__43_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__44 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__44_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__45_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "]"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__45 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__45_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__46_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "location"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__46 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__46_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__47_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "at"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__47 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__47_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__48_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "locationWildcard"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__48 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__48_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__49_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "*"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__49 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__49_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__50_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "h"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__50 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__50_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__51_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__51;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__52_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__50_value),LEAN_SCALAR_PTR_LITERAL(176, 181, 207, 77, 197, 87, 68, 121)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__52 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__52_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__53_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__53 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__53_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__54_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__53_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__54 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__54_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___boxed(lean_object**);
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "a"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__3___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__3___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__3___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__3___closed__1;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__3___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__3___closed__0_value),LEAN_SCALAR_PTR_LITERAL(247, 80, 99, 121, 74, 33, 203, 108)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__3___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__3___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__3(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "binderIdent"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(37, 194, 68, 106, 254, 181, 31, 191)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_Mathlib_Tactic_setArgsRest___closed__9(void){
_start:
{
lean_object* v___x_16_; lean_object* v___x_17_; lean_object* v___x_18_; lean_object* v___x_19_; 
v___x_16_ = l_Lean_binderIdent;
v___x_17_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_setArgsRest___closed__8));
v___x_18_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_setArgsRest___closed__5));
v___x_19_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_19_, 0, v___x_18_);
lean_ctor_set(v___x_19_, 1, v___x_17_);
lean_ctor_set(v___x_19_, 2, v___x_16_);
return v___x_19_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_setArgsRest___closed__19(void){
_start:
{
lean_object* v___x_39_; lean_object* v___x_40_; lean_object* v___x_41_; lean_object* v___x_42_; 
v___x_39_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_setArgsRest___closed__18));
v___x_40_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_setArgsRest___closed__9, &lp_mathlib_Mathlib_Tactic_setArgsRest___closed__9_once, _init_lp_mathlib_Mathlib_Tactic_setArgsRest___closed__9);
v___x_41_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_setArgsRest___closed__5));
v___x_42_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_42_, 0, v___x_41_);
lean_ctor_set(v___x_42_, 1, v___x_40_);
lean_ctor_set(v___x_42_, 2, v___x_39_);
return v___x_42_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_setArgsRest___closed__22(void){
_start:
{
lean_object* v___x_46_; lean_object* v___x_47_; lean_object* v___x_48_; lean_object* v___x_49_; 
v___x_46_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_setArgsRest___closed__21));
v___x_47_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_setArgsRest___closed__19, &lp_mathlib_Mathlib_Tactic_setArgsRest___closed__19_once, _init_lp_mathlib_Mathlib_Tactic_setArgsRest___closed__19);
v___x_48_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_setArgsRest___closed__5));
v___x_49_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_49_, 0, v___x_48_);
lean_ctor_set(v___x_49_, 1, v___x_47_);
lean_ctor_set(v___x_49_, 2, v___x_46_);
return v___x_49_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_setArgsRest___closed__23(void){
_start:
{
lean_object* v___x_50_; lean_object* v___x_51_; lean_object* v___x_52_; lean_object* v___x_53_; 
v___x_50_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_setArgsRest___closed__16));
v___x_51_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_setArgsRest___closed__22, &lp_mathlib_Mathlib_Tactic_setArgsRest___closed__22_once, _init_lp_mathlib_Mathlib_Tactic_setArgsRest___closed__22);
v___x_52_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_setArgsRest___closed__5));
v___x_53_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_53_, 0, v___x_52_);
lean_ctor_set(v___x_53_, 1, v___x_51_);
lean_ctor_set(v___x_53_, 2, v___x_50_);
return v___x_53_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_setArgsRest___closed__30(void){
_start:
{
lean_object* v___x_67_; lean_object* v___x_68_; lean_object* v___x_69_; lean_object* v___x_70_; 
v___x_67_ = l_Lean_binderIdent;
v___x_68_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_setArgsRest___closed__29));
v___x_69_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_setArgsRest___closed__5));
v___x_70_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_70_, 0, v___x_69_);
lean_ctor_set(v___x_70_, 1, v___x_68_);
lean_ctor_set(v___x_70_, 2, v___x_67_);
return v___x_70_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_setArgsRest___closed__31(void){
_start:
{
lean_object* v___x_71_; lean_object* v___x_72_; lean_object* v___x_73_; 
v___x_71_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_setArgsRest___closed__30, &lp_mathlib_Mathlib_Tactic_setArgsRest___closed__30_once, _init_lp_mathlib_Mathlib_Tactic_setArgsRest___closed__30);
v___x_72_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_setArgsRest___closed__11));
v___x_73_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_73_, 0, v___x_72_);
lean_ctor_set(v___x_73_, 1, v___x_71_);
return v___x_73_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_setArgsRest___closed__32(void){
_start:
{
lean_object* v___x_74_; lean_object* v___x_75_; lean_object* v___x_76_; lean_object* v___x_77_; 
v___x_74_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_setArgsRest___closed__31, &lp_mathlib_Mathlib_Tactic_setArgsRest___closed__31_once, _init_lp_mathlib_Mathlib_Tactic_setArgsRest___closed__31);
v___x_75_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_setArgsRest___closed__23, &lp_mathlib_Mathlib_Tactic_setArgsRest___closed__23_once, _init_lp_mathlib_Mathlib_Tactic_setArgsRest___closed__23);
v___x_76_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_setArgsRest___closed__5));
v___x_77_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_77_, 0, v___x_76_);
lean_ctor_set(v___x_77_, 1, v___x_75_);
lean_ctor_set(v___x_77_, 2, v___x_74_);
return v___x_77_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_setArgsRest___closed__33(void){
_start:
{
lean_object* v___x_78_; lean_object* v___x_79_; lean_object* v___x_80_; lean_object* v___x_81_; 
v___x_78_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_setArgsRest___closed__32, &lp_mathlib_Mathlib_Tactic_setArgsRest___closed__32_once, _init_lp_mathlib_Mathlib_Tactic_setArgsRest___closed__32);
v___x_79_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_setArgsRest___closed__3));
v___x_80_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_setArgsRest___closed__0));
v___x_81_ = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(v___x_81_, 0, v___x_80_);
lean_ctor_set(v___x_81_, 1, v___x_79_);
lean_ctor_set(v___x_81_, 2, v___x_78_);
return v___x_81_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_setArgsRest(void){
_start:
{
lean_object* v___x_82_; 
v___x_82_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_setArgsRest___closed__33, &lp_mathlib_Mathlib_Tactic_setArgsRest___closed__33_once, _init_lp_mathlib_Mathlib_Tactic_setArgsRest___closed__33);
return v___x_82_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_setTactic___closed__8(void){
_start:
{
lean_object* v___x_102_; lean_object* v___x_103_; lean_object* v___x_104_; lean_object* v___x_105_; 
v___x_102_ = lp_mathlib_Mathlib_Tactic_setArgsRest;
v___x_103_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_setTactic___closed__7));
v___x_104_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_setArgsRest___closed__5));
v___x_105_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_105_, 0, v___x_104_);
lean_ctor_set(v___x_105_, 1, v___x_103_);
lean_ctor_set(v___x_105_, 2, v___x_102_);
return v___x_105_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_setTactic___closed__9(void){
_start:
{
lean_object* v___x_106_; lean_object* v___x_107_; lean_object* v___x_108_; lean_object* v___x_109_; 
v___x_106_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_setTactic___closed__8, &lp_mathlib_Mathlib_Tactic_setTactic___closed__8_once, _init_lp_mathlib_Mathlib_Tactic_setTactic___closed__8);
v___x_107_ = lean_unsigned_to_nat(1022u);
v___x_108_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_setTactic___closed__1));
v___x_109_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_109_, 0, v___x_108_);
lean_ctor_set(v___x_109_, 1, v___x_107_);
lean_ctor_set(v___x_109_, 2, v___x_106_);
return v___x_109_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_setTactic(void){
_start:
{
lean_object* v___x_110_; 
v___x_110_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_setTactic___closed__9, &lp_mathlib_Mathlib_Tactic_setTactic___closed__9_once, _init_lp_mathlib_Mathlib_Tactic_setTactic___closed__9);
return v___x_110_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_tacticSet_x21___00__closed__4(void){
_start:
{
lean_object* v___x_120_; lean_object* v___x_121_; lean_object* v___x_122_; lean_object* v___x_123_; 
v___x_120_ = lp_mathlib_Mathlib_Tactic_setArgsRest;
v___x_121_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticSet_x21___00__closed__3));
v___x_122_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_setArgsRest___closed__5));
v___x_123_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_123_, 0, v___x_122_);
lean_ctor_set(v___x_123_, 1, v___x_121_);
lean_ctor_set(v___x_123_, 2, v___x_120_);
return v___x_123_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_tacticSet_x21___00__closed__5(void){
_start:
{
lean_object* v___x_124_; lean_object* v___x_125_; lean_object* v___x_126_; lean_object* v___x_127_; 
v___x_124_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_tacticSet_x21___00__closed__4, &lp_mathlib_Mathlib_Tactic_tacticSet_x21___00__closed__4_once, _init_lp_mathlib_Mathlib_Tactic_tacticSet_x21___00__closed__4);
v___x_125_ = lean_unsigned_to_nat(1022u);
v___x_126_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticSet_x21___00__closed__1));
v___x_127_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_127_, 0, v___x_126_);
lean_ctor_set(v___x_127_, 1, v___x_125_);
lean_ctor_set(v___x_127_, 2, v___x_124_);
return v___x_127_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_tacticSet_x21__(void){
_start:
{
lean_object* v___x_128_; 
v___x_128_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_tacticSet_x21___00__closed__5, &lp_mathlib_Mathlib_Tactic_tacticSet_x21___00__closed__5_once, _init_lp_mathlib_Mathlib_Tactic_tacticSet_x21___00__closed__5);
return v___x_128_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______macroRules__Mathlib__Tactic__tacticSet_x21____1(lean_object* v_x_132_, lean_object* v_a_133_, lean_object* v_a_134_){
_start:
{
lean_object* v___x_135_; uint8_t v___x_136_; 
v___x_135_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticSet_x21___00__closed__1));
lean_inc(v_x_132_);
v___x_136_ = l_Lean_Syntax_isOfKind(v_x_132_, v___x_135_);
if (v___x_136_ == 0)
{
lean_object* v___x_137_; lean_object* v___x_138_; 
lean_dec(v_x_132_);
v___x_137_ = lean_box(1);
v___x_138_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_138_, 0, v___x_137_);
lean_ctor_set(v___x_138_, 1, v_a_134_);
return v___x_138_;
}
else
{
lean_object* v_ref_139_; lean_object* v___x_140_; lean_object* v___x_141_; uint8_t v___x_142_; lean_object* v___x_143_; lean_object* v___x_144_; lean_object* v___x_145_; lean_object* v___x_146_; lean_object* v___x_147_; lean_object* v___x_148_; lean_object* v___x_149_; lean_object* v___x_150_; lean_object* v___x_151_; lean_object* v___x_152_; 
v_ref_139_ = lean_ctor_get(v_a_133_, 5);
v___x_140_ = lean_unsigned_to_nat(1u);
v___x_141_ = l_Lean_Syntax_getArg(v_x_132_, v___x_140_);
lean_dec(v_x_132_);
v___x_142_ = 0;
v___x_143_ = l_Lean_SourceInfo_fromRef(v_ref_139_, v___x_142_);
v___x_144_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_setTactic___closed__1));
v___x_145_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_setTactic___closed__2));
lean_inc_n(v___x_143_, 3);
v___x_146_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_146_, 0, v___x_143_);
lean_ctor_set(v___x_146_, 1, v___x_145_);
v___x_147_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______macroRules__Mathlib__Tactic__tacticSet_x21____1___closed__1));
v___x_148_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_setTactic___closed__4));
v___x_149_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_149_, 0, v___x_143_);
lean_ctor_set(v___x_149_, 1, v___x_148_);
v___x_150_ = l_Lean_Syntax_node1(v___x_143_, v___x_147_, v___x_149_);
v___x_151_ = l_Lean_Syntax_node3(v___x_143_, v___x_144_, v___x_146_, v___x_150_, v___x_141_);
v___x_152_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_152_, 0, v___x_151_);
lean_ctor_set(v___x_152_, 1, v_a_134_);
return v___x_152_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______macroRules__Mathlib__Tactic__tacticSet_x21____1___boxed(lean_object* v_x_153_, lean_object* v_a_154_, lean_object* v_a_155_){
_start:
{
lean_object* v_res_156_; 
v_res_156_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______macroRules__Mathlib__Tactic__tacticSet_x21____1(v_x_153_, v_a_154_, v_a_155_);
lean_dec_ref(v_a_154_);
return v_res_156_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_157_; lean_object* v___x_158_; lean_object* v___x_159_; 
v___x_157_ = lean_box(0);
v___x_158_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_159_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_159_, 0, v___x_158_);
lean_ctor_set(v___x_159_, 1, v___x_157_);
return v___x_159_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1_spec__0___redArg(){
_start:
{
lean_object* v___x_161_; lean_object* v___x_162_; 
v___x_161_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1_spec__0___redArg___closed__0);
v___x_162_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_162_, 0, v___x_161_);
return v___x_162_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1_spec__0___redArg___boxed(lean_object* v___y_163_){
_start:
{
lean_object* v_res_164_; 
v_res_164_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1_spec__0___redArg();
return v_res_164_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1_spec__0(lean_object* v_00_u03b1_165_, lean_object* v___y_166_, lean_object* v___y_167_, lean_object* v___y_168_, lean_object* v___y_169_, lean_object* v___y_170_, lean_object* v___y_171_, lean_object* v___y_172_, lean_object* v___y_173_){
_start:
{
lean_object* v___x_175_; 
v___x_175_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1_spec__0___redArg();
return v___x_175_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1_spec__0___boxed(lean_object* v_00_u03b1_176_, lean_object* v___y_177_, lean_object* v___y_178_, lean_object* v___y_179_, lean_object* v___y_180_, lean_object* v___y_181_, lean_object* v___y_182_, lean_object* v___y_183_, lean_object* v___y_184_, lean_object* v___y_185_){
_start:
{
lean_object* v_res_186_; 
v_res_186_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1_spec__0(v_00_u03b1_176_, v___y_177_, v___y_178_, v___y_179_, v___y_180_, v___y_181_, v___y_182_, v___y_183_, v___y_184_);
lean_dec(v___y_184_);
lean_dec_ref(v___y_183_);
lean_dec(v___y_182_);
lean_dec_ref(v___y_181_);
lean_dec(v___y_180_);
lean_dec_ref(v___y_179_);
lean_dec(v___y_178_);
lean_dec_ref(v___y_177_);
return v_res_186_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__0(lean_object* v_a_187_, lean_object* v_fst_188_, lean_object* v_snd_189_, uint8_t v___x_190_, lean_object* v___y_191_, lean_object* v___y_192_, lean_object* v___y_193_, lean_object* v___y_194_, lean_object* v___y_195_, lean_object* v___y_196_, lean_object* v___y_197_, lean_object* v___y_198_){
_start:
{
lean_object* v___x_200_; 
v___x_200_ = l_Lean_Elab_Tactic_getMainGoal___redArg(v___y_192_, v___y_195_, v___y_196_, v___y_197_, v___y_198_);
if (lean_obj_tag(v___x_200_) == 0)
{
lean_object* v_a_201_; lean_object* v___x_202_; lean_object* v___x_203_; 
v_a_201_ = lean_ctor_get(v___x_200_, 0);
lean_inc(v_a_201_);
lean_dec_ref_known(v___x_200_, 1);
v___x_202_ = l_Lean_TSyntax_getId(v_a_187_);
v___x_203_ = l_Lean_MVarId_define(v_a_201_, v___x_202_, v_fst_188_, v_snd_189_, v___y_195_, v___y_196_, v___y_197_, v___y_198_);
if (lean_obj_tag(v___x_203_) == 0)
{
lean_object* v_a_204_; lean_object* v___x_205_; 
v_a_204_ = lean_ctor_get(v___x_203_, 0);
lean_inc(v_a_204_);
lean_dec_ref_known(v___x_203_, 1);
v___x_205_ = l_Lean_Meta_intro1Core(v_a_204_, v___x_190_, v___y_195_, v___y_196_, v___y_197_, v___y_198_);
if (lean_obj_tag(v___x_205_) == 0)
{
lean_object* v_a_206_; lean_object* v_fst_207_; lean_object* v_snd_208_; lean_object* v___x_210_; uint8_t v_isShared_211_; uint8_t v_isSharedCheck_233_; 
v_a_206_ = lean_ctor_get(v___x_205_, 0);
lean_inc(v_a_206_);
lean_dec_ref_known(v___x_205_, 1);
v_fst_207_ = lean_ctor_get(v_a_206_, 0);
v_snd_208_ = lean_ctor_get(v_a_206_, 1);
v_isSharedCheck_233_ = !lean_is_exclusive(v_a_206_);
if (v_isSharedCheck_233_ == 0)
{
v___x_210_ = v_a_206_;
v_isShared_211_ = v_isSharedCheck_233_;
goto v_resetjp_209_;
}
else
{
lean_inc(v_snd_208_);
lean_inc(v_fst_207_);
lean_dec(v_a_206_);
v___x_210_ = lean_box(0);
v_isShared_211_ = v_isSharedCheck_233_;
goto v_resetjp_209_;
}
v_resetjp_209_:
{
lean_object* v___x_212_; lean_object* v___x_214_; 
v___x_212_ = lean_box(0);
if (v_isShared_211_ == 0)
{
lean_ctor_set_tag(v___x_210_, 1);
lean_ctor_set(v___x_210_, 1, v___x_212_);
lean_ctor_set(v___x_210_, 0, v_snd_208_);
v___x_214_ = v___x_210_;
goto v_reusejp_213_;
}
else
{
lean_object* v_reuseFailAlloc_232_; 
v_reuseFailAlloc_232_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_232_, 0, v_snd_208_);
lean_ctor_set(v_reuseFailAlloc_232_, 1, v___x_212_);
v___x_214_ = v_reuseFailAlloc_232_;
goto v_reusejp_213_;
}
v_reusejp_213_:
{
lean_object* v___x_215_; 
v___x_215_ = l_Lean_Elab_Tactic_replaceMainGoal___redArg(v___x_214_, v___y_192_, v___y_195_, v___y_196_, v___y_197_, v___y_198_);
if (lean_obj_tag(v___x_215_) == 0)
{
lean_object* v___x_217_; uint8_t v_isShared_218_; uint8_t v_isSharedCheck_222_; 
v_isSharedCheck_222_ = !lean_is_exclusive(v___x_215_);
if (v_isSharedCheck_222_ == 0)
{
lean_object* v_unused_223_; 
v_unused_223_ = lean_ctor_get(v___x_215_, 0);
lean_dec(v_unused_223_);
v___x_217_ = v___x_215_;
v_isShared_218_ = v_isSharedCheck_222_;
goto v_resetjp_216_;
}
else
{
lean_dec(v___x_215_);
v___x_217_ = lean_box(0);
v_isShared_218_ = v_isSharedCheck_222_;
goto v_resetjp_216_;
}
v_resetjp_216_:
{
lean_object* v___x_220_; 
if (v_isShared_218_ == 0)
{
lean_ctor_set(v___x_217_, 0, v_fst_207_);
v___x_220_ = v___x_217_;
goto v_reusejp_219_;
}
else
{
lean_object* v_reuseFailAlloc_221_; 
v_reuseFailAlloc_221_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_221_, 0, v_fst_207_);
v___x_220_ = v_reuseFailAlloc_221_;
goto v_reusejp_219_;
}
v_reusejp_219_:
{
return v___x_220_;
}
}
}
else
{
lean_object* v_a_224_; lean_object* v___x_226_; uint8_t v_isShared_227_; uint8_t v_isSharedCheck_231_; 
lean_dec(v_fst_207_);
v_a_224_ = lean_ctor_get(v___x_215_, 0);
v_isSharedCheck_231_ = !lean_is_exclusive(v___x_215_);
if (v_isSharedCheck_231_ == 0)
{
v___x_226_ = v___x_215_;
v_isShared_227_ = v_isSharedCheck_231_;
goto v_resetjp_225_;
}
else
{
lean_inc(v_a_224_);
lean_dec(v___x_215_);
v___x_226_ = lean_box(0);
v_isShared_227_ = v_isSharedCheck_231_;
goto v_resetjp_225_;
}
v_resetjp_225_:
{
lean_object* v___x_229_; 
if (v_isShared_227_ == 0)
{
v___x_229_ = v___x_226_;
goto v_reusejp_228_;
}
else
{
lean_object* v_reuseFailAlloc_230_; 
v_reuseFailAlloc_230_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_230_, 0, v_a_224_);
v___x_229_ = v_reuseFailAlloc_230_;
goto v_reusejp_228_;
}
v_reusejp_228_:
{
return v___x_229_;
}
}
}
}
}
}
else
{
lean_object* v_a_234_; lean_object* v___x_236_; uint8_t v_isShared_237_; uint8_t v_isSharedCheck_241_; 
v_a_234_ = lean_ctor_get(v___x_205_, 0);
v_isSharedCheck_241_ = !lean_is_exclusive(v___x_205_);
if (v_isSharedCheck_241_ == 0)
{
v___x_236_ = v___x_205_;
v_isShared_237_ = v_isSharedCheck_241_;
goto v_resetjp_235_;
}
else
{
lean_inc(v_a_234_);
lean_dec(v___x_205_);
v___x_236_ = lean_box(0);
v_isShared_237_ = v_isSharedCheck_241_;
goto v_resetjp_235_;
}
v_resetjp_235_:
{
lean_object* v___x_239_; 
if (v_isShared_237_ == 0)
{
v___x_239_ = v___x_236_;
goto v_reusejp_238_;
}
else
{
lean_object* v_reuseFailAlloc_240_; 
v_reuseFailAlloc_240_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_240_, 0, v_a_234_);
v___x_239_ = v_reuseFailAlloc_240_;
goto v_reusejp_238_;
}
v_reusejp_238_:
{
return v___x_239_;
}
}
}
}
else
{
lean_object* v_a_242_; lean_object* v___x_244_; uint8_t v_isShared_245_; uint8_t v_isSharedCheck_249_; 
v_a_242_ = lean_ctor_get(v___x_203_, 0);
v_isSharedCheck_249_ = !lean_is_exclusive(v___x_203_);
if (v_isSharedCheck_249_ == 0)
{
v___x_244_ = v___x_203_;
v_isShared_245_ = v_isSharedCheck_249_;
goto v_resetjp_243_;
}
else
{
lean_inc(v_a_242_);
lean_dec(v___x_203_);
v___x_244_ = lean_box(0);
v_isShared_245_ = v_isSharedCheck_249_;
goto v_resetjp_243_;
}
v_resetjp_243_:
{
lean_object* v___x_247_; 
if (v_isShared_245_ == 0)
{
v___x_247_ = v___x_244_;
goto v_reusejp_246_;
}
else
{
lean_object* v_reuseFailAlloc_248_; 
v_reuseFailAlloc_248_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_248_, 0, v_a_242_);
v___x_247_ = v_reuseFailAlloc_248_;
goto v_reusejp_246_;
}
v_reusejp_246_:
{
return v___x_247_;
}
}
}
}
else
{
lean_object* v_a_250_; lean_object* v___x_252_; uint8_t v_isShared_253_; uint8_t v_isSharedCheck_257_; 
lean_dec_ref(v_snd_189_);
lean_dec_ref(v_fst_188_);
v_a_250_ = lean_ctor_get(v___x_200_, 0);
v_isSharedCheck_257_ = !lean_is_exclusive(v___x_200_);
if (v_isSharedCheck_257_ == 0)
{
v___x_252_ = v___x_200_;
v_isShared_253_ = v_isSharedCheck_257_;
goto v_resetjp_251_;
}
else
{
lean_inc(v_a_250_);
lean_dec(v___x_200_);
v___x_252_ = lean_box(0);
v_isShared_253_ = v_isSharedCheck_257_;
goto v_resetjp_251_;
}
v_resetjp_251_:
{
lean_object* v___x_255_; 
if (v_isShared_253_ == 0)
{
v___x_255_ = v___x_252_;
goto v_reusejp_254_;
}
else
{
lean_object* v_reuseFailAlloc_256_; 
v_reuseFailAlloc_256_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_256_, 0, v_a_250_);
v___x_255_ = v_reuseFailAlloc_256_;
goto v_reusejp_254_;
}
v_reusejp_254_:
{
return v___x_255_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__0___boxed(lean_object* v_a_258_, lean_object* v_fst_259_, lean_object* v_snd_260_, lean_object* v___x_261_, lean_object* v___y_262_, lean_object* v___y_263_, lean_object* v___y_264_, lean_object* v___y_265_, lean_object* v___y_266_, lean_object* v___y_267_, lean_object* v___y_268_, lean_object* v___y_269_, lean_object* v___y_270_){
_start:
{
uint8_t v___x_39699__boxed_271_; lean_object* v_res_272_; 
v___x_39699__boxed_271_ = lean_unbox(v___x_261_);
v_res_272_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__0(v_a_258_, v_fst_259_, v_snd_260_, v___x_39699__boxed_271_, v___y_262_, v___y_263_, v___y_264_, v___y_265_, v___y_266_, v___y_267_, v___y_268_, v___y_269_);
lean_dec(v___y_269_);
lean_dec_ref(v___y_268_);
lean_dec(v___y_267_);
lean_dec_ref(v___y_266_);
lean_dec(v___y_265_);
lean_dec_ref(v___y_264_);
lean_dec(v___y_263_);
lean_dec_ref(v___y_262_);
lean_dec(v_a_258_);
return v_res_272_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__1(lean_object* v_a_273_, lean_object* v___x_274_, lean_object* v___x_275_, lean_object* v___x_276_, lean_object* v___x_277_, uint8_t v___x_278_, uint8_t v___x_279_, lean_object* v___y_280_, lean_object* v___y_281_, lean_object* v___y_282_, lean_object* v___y_283_, lean_object* v___y_284_, lean_object* v___y_285_, lean_object* v___y_286_, lean_object* v___y_287_){
_start:
{
lean_object* v___x_289_; 
v___x_289_ = l_Lean_Elab_Term_addTermInfo_x27(v_a_273_, v___x_274_, v___x_275_, v___x_276_, v___x_277_, v___x_278_, v___x_279_, v___y_282_, v___y_283_, v___y_284_, v___y_285_, v___y_286_, v___y_287_);
return v___x_289_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__1___boxed(lean_object* v_a_290_, lean_object* v___x_291_, lean_object* v___x_292_, lean_object* v___x_293_, lean_object* v___x_294_, lean_object* v___x_295_, lean_object* v___x_296_, lean_object* v___y_297_, lean_object* v___y_298_, lean_object* v___y_299_, lean_object* v___y_300_, lean_object* v___y_301_, lean_object* v___y_302_, lean_object* v___y_303_, lean_object* v___y_304_, lean_object* v___y_305_){
_start:
{
uint8_t v___x_39854__boxed_306_; uint8_t v___x_39855__boxed_307_; lean_object* v_res_308_; 
v___x_39854__boxed_306_ = lean_unbox(v___x_295_);
v___x_39855__boxed_307_ = lean_unbox(v___x_296_);
v_res_308_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__1(v_a_290_, v___x_291_, v___x_292_, v___x_293_, v___x_294_, v___x_39854__boxed_306_, v___x_39855__boxed_307_, v___y_297_, v___y_298_, v___y_299_, v___y_300_, v___y_301_, v___y_302_, v___y_303_, v___y_304_);
lean_dec(v___y_304_);
lean_dec_ref(v___y_303_);
lean_dec(v___y_302_);
lean_dec_ref(v___y_301_);
lean_dec(v___y_300_);
lean_dec_ref(v___y_299_);
lean_dec(v___y_298_);
lean_dec_ref(v___y_297_);
return v_res_308_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__5(void){
_start:
{
lean_object* v___x_314_; 
v___x_314_ = l_Array_mkArray0(lean_box(0));
return v___x_314_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__20(void){
_start:
{
lean_object* v___x_331_; lean_object* v___x_332_; 
v___x_331_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__19));
v___x_332_ = l_String_toRawSubstring_x27(v___x_331_);
return v___x_332_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__26(void){
_start:
{
lean_object* v___x_338_; lean_object* v___x_339_; 
v___x_338_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__25));
v___x_339_ = l_String_toRawSubstring_x27(v___x_338_);
return v___x_339_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__51(void){
_start:
{
lean_object* v___x_373_; lean_object* v___x_374_; 
v___x_373_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__50));
v___x_374_ = l_String_toRawSubstring_x27(v___x_373_);
return v___x_374_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2(lean_object* v_rev_380_, lean_object* v___x_381_, lean_object* v___x_382_, lean_object* v_tk_383_, uint8_t v___x_384_, lean_object* v___x_385_, uint8_t v___x_386_, lean_object* v_rw_387_, lean_object* v_ty_388_, lean_object* v___x_389_, lean_object* v_h_390_, lean_object* v___x_391_, lean_object* v___x_392_, lean_object* v_a_393_, lean_object* v___y_394_, lean_object* v___y_395_, lean_object* v___y_396_, lean_object* v___y_397_, lean_object* v___y_398_, lean_object* v___y_399_, lean_object* v___y_400_, lean_object* v___y_401_){
_start:
{
lean_object* v___y_407_; lean_object* v___y_408_; lean_object* v___y_409_; lean_object* v___y_410_; lean_object* v___y_411_; lean_object* v___y_412_; lean_object* v___y_413_; lean_object* v___y_414_; lean_object* v___y_415_; lean_object* v___y_416_; lean_object* v___y_417_; lean_object* v___y_418_; lean_object* v___y_650_; lean_object* v_fst_651_; lean_object* v_snd_652_; lean_object* v_a_754_; lean_object* v_a_807_; 
if (lean_obj_tag(v_h_390_) == 0)
{
v_a_754_ = v_h_390_;
goto v___jp_753_;
}
else
{
lean_object* v_val_809_; uint8_t v___x_810_; 
v_val_809_ = lean_ctor_get(v_h_390_, 0);
lean_inc_n(v_val_809_, 2);
lean_dec_ref_known(v_h_390_, 1);
v___x_810_ = l_Lean_Syntax_isOfKind(v_val_809_, v___x_391_);
if (v___x_810_ == 0)
{
lean_object* v_ref_811_; lean_object* v_quotContext_812_; lean_object* v_currMacroScope_813_; lean_object* v___x_814_; lean_object* v___x_815_; lean_object* v___x_816_; lean_object* v___x_817_; lean_object* v___x_818_; lean_object* v___x_819_; 
lean_dec(v_val_809_);
v_ref_811_ = lean_ctor_get(v___y_400_, 5);
v_quotContext_812_ = lean_ctor_get(v___y_400_, 10);
v_currMacroScope_813_ = lean_ctor_get(v___y_400_, 11);
v___x_814_ = l_Lean_SourceInfo_fromRef(v_ref_811_, v___x_810_);
v___x_815_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__51, &lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__51_once, _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__51);
v___x_816_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__52));
lean_inc(v_currMacroScope_813_);
lean_inc(v_quotContext_812_);
v___x_817_ = l_Lean_addMacroScope(v_quotContext_812_, v___x_816_, v_currMacroScope_813_);
v___x_818_ = lean_box(0);
v___x_819_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_819_, 0, v___x_814_);
lean_ctor_set(v___x_819_, 1, v___x_815_);
lean_ctor_set(v___x_819_, 2, v___x_817_);
lean_ctor_set(v___x_819_, 3, v___x_818_);
v_a_807_ = v___x_819_;
goto v___jp_806_;
}
else
{
lean_object* v___x_820_; lean_object* v___x_821_; uint8_t v___x_822_; 
v___x_820_ = l_Lean_Syntax_getArg(v_val_809_, v___x_392_);
lean_dec(v_val_809_);
v___x_821_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__54));
lean_inc(v___x_820_);
v___x_822_ = l_Lean_Syntax_isOfKind(v___x_820_, v___x_821_);
if (v___x_822_ == 0)
{
lean_object* v_ref_823_; lean_object* v_quotContext_824_; lean_object* v_currMacroScope_825_; lean_object* v___x_826_; lean_object* v___x_827_; lean_object* v___x_828_; lean_object* v___x_829_; lean_object* v___x_830_; lean_object* v___x_831_; 
lean_dec(v___x_820_);
v_ref_823_ = lean_ctor_get(v___y_400_, 5);
v_quotContext_824_ = lean_ctor_get(v___y_400_, 10);
v_currMacroScope_825_ = lean_ctor_get(v___y_400_, 11);
v___x_826_ = l_Lean_SourceInfo_fromRef(v_ref_823_, v___x_822_);
v___x_827_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__51, &lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__51_once, _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__51);
v___x_828_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__52));
lean_inc(v_currMacroScope_825_);
lean_inc(v_quotContext_824_);
v___x_829_ = l_Lean_addMacroScope(v_quotContext_824_, v___x_828_, v_currMacroScope_825_);
v___x_830_ = lean_box(0);
v___x_831_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_831_, 0, v___x_826_);
lean_ctor_set(v___x_831_, 1, v___x_827_);
lean_ctor_set(v___x_831_, 2, v___x_829_);
lean_ctor_set(v___x_831_, 3, v___x_830_);
v_a_807_ = v___x_831_;
goto v___jp_806_;
}
else
{
v_a_807_ = v___x_820_;
goto v___jp_806_;
}
}
}
v___jp_403_:
{
lean_object* v___x_404_; lean_object* v___x_405_; 
v___x_404_ = lean_box(0);
v___x_405_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_405_, 0, v___x_404_);
return v___x_405_;
}
v___jp_406_:
{
if (lean_obj_tag(v___y_410_) == 1)
{
if (lean_obj_tag(v_rev_380_) == 1)
{
lean_object* v_val_419_; lean_object* v___x_421_; uint8_t v_isShared_422_; uint8_t v_isSharedCheck_648_; 
v_val_419_ = lean_ctor_get(v_rev_380_, 0);
v_isSharedCheck_648_ = !lean_is_exclusive(v_rev_380_);
if (v_isSharedCheck_648_ == 0)
{
v___x_421_ = v_rev_380_;
v_isShared_422_ = v_isSharedCheck_648_;
goto v_resetjp_420_;
}
else
{
lean_inc(v_val_419_);
lean_dec(v_rev_380_);
v___x_421_ = lean_box(0);
v_isShared_422_ = v_isSharedCheck_648_;
goto v_resetjp_420_;
}
v_resetjp_420_:
{
if (lean_obj_tag(v_val_419_) == 0)
{
lean_object* v_val_423_; lean_object* v___x_425_; uint8_t v_isShared_426_; uint8_t v_isSharedCheck_531_; 
v_val_423_ = lean_ctor_get(v___y_410_, 0);
v_isSharedCheck_531_ = !lean_is_exclusive(v___y_410_);
if (v_isSharedCheck_531_ == 0)
{
v___x_425_ = v___y_410_;
v_isShared_426_ = v_isSharedCheck_531_;
goto v_resetjp_424_;
}
else
{
lean_inc(v_val_423_);
lean_dec(v___y_410_);
v___x_425_ = lean_box(0);
v_isShared_426_ = v_isSharedCheck_531_;
goto v_resetjp_424_;
}
v_resetjp_424_:
{
lean_object* v___x_427_; 
v___x_427_ = l_Lean_Elab_Term_exprToSyntax(v___y_409_, v___y_413_, v___y_414_, v___y_415_, v___y_416_, v___y_417_, v___y_418_);
if (lean_obj_tag(v___x_427_) == 0)
{
lean_object* v_a_428_; lean_object* v___x_429_; 
v_a_428_ = lean_ctor_get(v___x_427_, 0);
lean_inc(v_a_428_);
lean_dec_ref_known(v___x_427_, 1);
v___x_429_ = l_Lean_Elab_Term_exprToSyntax(v___y_408_, v___y_413_, v___y_414_, v___y_415_, v___y_416_, v___y_417_, v___y_418_);
if (lean_obj_tag(v___x_429_) == 0)
{
lean_object* v_a_430_; lean_object* v_ref_431_; lean_object* v_quotContext_432_; lean_object* v_currMacroScope_433_; uint8_t v___x_434_; lean_object* v___x_435_; lean_object* v___x_436_; lean_object* v___x_437_; lean_object* v___x_438_; lean_object* v___x_439_; lean_object* v___x_440_; lean_object* v___x_441_; lean_object* v___x_442_; lean_object* v___x_443_; lean_object* v___x_444_; lean_object* v___x_445_; lean_object* v___x_446_; lean_object* v___x_447_; lean_object* v___x_448_; lean_object* v___x_449_; lean_object* v___x_450_; lean_object* v___x_451_; lean_object* v___x_452_; lean_object* v___x_453_; lean_object* v___x_454_; lean_object* v___x_455_; lean_object* v___x_456_; lean_object* v___x_457_; lean_object* v___x_458_; lean_object* v___x_459_; lean_object* v___x_460_; lean_object* v___x_461_; lean_object* v___x_462_; lean_object* v___x_463_; lean_object* v___x_464_; lean_object* v___x_465_; lean_object* v___x_466_; lean_object* v___x_467_; lean_object* v___x_468_; lean_object* v___x_469_; lean_object* v___x_470_; lean_object* v___x_471_; lean_object* v___x_472_; lean_object* v___x_474_; 
v_a_430_ = lean_ctor_get(v___x_429_, 0);
lean_inc(v_a_430_);
lean_dec_ref_known(v___x_429_, 1);
v_ref_431_ = lean_ctor_get(v___y_417_, 5);
v_quotContext_432_ = lean_ctor_get(v___y_417_, 10);
v_currMacroScope_433_ = lean_ctor_get(v___y_417_, 11);
v___x_434_ = 0;
v___x_435_ = l_Lean_SourceInfo_fromRef(v_ref_431_, v___x_434_);
v___x_436_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__0));
v___x_437_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__1));
lean_inc_ref_n(v___x_382_, 2);
lean_inc_ref_n(v___x_381_, 8);
v___x_438_ = l_Lean_Name_mkStr4(v___x_381_, v___x_436_, v___x_382_, v___x_437_);
v___x_439_ = l_Lean_SourceInfo_fromRef(v_tk_383_, v___x_384_);
v___x_440_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__2));
v___x_441_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_441_, 0, v___x_439_);
lean_ctor_set(v___x_441_, 1, v___x_440_);
v___x_442_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__3));
v___x_443_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__4));
v___x_444_ = l_Lean_Name_mkStr4(v___x_381_, v___x_436_, v___x_442_, v___x_443_);
v___x_445_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______macroRules__Mathlib__Tactic__tacticSet_x21____1___closed__1));
v___x_446_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__5, &lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__5_once, _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__5);
lean_inc_n(v___x_435_, 6);
v___x_447_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_447_, 0, v___x_435_);
lean_ctor_set(v___x_447_, 1, v___x_445_);
lean_ctor_set(v___x_447_, 2, v___x_446_);
lean_inc_ref(v___x_447_);
v___x_448_ = l_Lean_Syntax_node1(v___x_435_, v___x_444_, v___x_447_);
v___x_449_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__6));
v___x_450_ = l_Lean_Name_mkStr4(v___x_381_, v___x_436_, v___x_442_, v___x_449_);
v___x_451_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__7));
v___x_452_ = l_Lean_Name_mkStr4(v___x_381_, v___x_436_, v___x_442_, v___x_451_);
v___x_453_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__8));
v___x_454_ = l_Lean_Name_mkStr4(v___x_381_, v___x_436_, v___x_442_, v___x_453_);
v___x_455_ = l_Lean_Syntax_node1(v___x_435_, v___x_454_, v_val_423_);
v___x_456_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__9));
v___x_457_ = l_Lean_Name_mkStr4(v___x_381_, v___x_436_, v___x_442_, v___x_456_);
v___x_458_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__10));
v___x_459_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_459_, 0, v___x_435_);
lean_ctor_set(v___x_459_, 1, v___x_458_);
v___x_460_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__12));
v___x_461_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__13));
v___x_462_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_462_, 0, v___x_435_);
lean_ctor_set(v___x_462_, 1, v___x_461_);
v___x_463_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__14));
v___x_464_ = l_Lean_Name_mkStr4(v___x_381_, v___x_436_, v___x_442_, v___x_463_);
v___x_465_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__15));
v___x_466_ = l_Lean_Name_mkStr4(v___x_381_, v___x_436_, v___x_442_, v___x_465_);
v___x_467_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__16));
v___x_468_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_468_, 0, v___x_435_);
lean_ctor_set(v___x_468_, 1, v___x_467_);
v___x_469_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__18));
v___x_470_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__20, &lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__20_once, _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__20);
lean_inc(v_currMacroScope_433_);
lean_inc(v_quotContext_432_);
v___x_471_ = l_Lean_addMacroScope(v_quotContext_432_, v___y_407_, v_currMacroScope_433_);
v___x_472_ = l_Lean_Name_mkStr2(v___x_385_, v___x_382_);
if (v_isShared_422_ == 0)
{
lean_ctor_set_tag(v___x_421_, 0);
lean_ctor_set(v___x_421_, 0, v___x_472_);
v___x_474_ = v___x_421_;
goto v_reusejp_473_;
}
else
{
lean_object* v_reuseFailAlloc_514_; 
v_reuseFailAlloc_514_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_514_, 0, v___x_472_);
v___x_474_ = v_reuseFailAlloc_514_;
goto v_reusejp_473_;
}
v_reusejp_473_:
{
lean_object* v___x_475_; lean_object* v___x_476_; lean_object* v___x_478_; 
v___x_475_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__21));
lean_inc_ref(v___x_381_);
v___x_476_ = l_Lean_Name_mkStr2(v___x_381_, v___x_475_);
if (v_isShared_426_ == 0)
{
lean_ctor_set_tag(v___x_425_, 0);
lean_ctor_set(v___x_425_, 0, v___x_476_);
v___x_478_ = v___x_425_;
goto v_reusejp_477_;
}
else
{
lean_object* v_reuseFailAlloc_513_; 
v_reuseFailAlloc_513_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_513_, 0, v___x_476_);
v___x_478_ = v_reuseFailAlloc_513_;
goto v_reusejp_477_;
}
v_reusejp_477_:
{
lean_object* v___x_479_; lean_object* v___x_480_; lean_object* v___x_481_; lean_object* v___x_482_; lean_object* v___x_483_; lean_object* v___x_484_; lean_object* v___x_485_; lean_object* v___x_486_; lean_object* v___x_487_; lean_object* v___x_488_; lean_object* v___x_489_; lean_object* v___x_490_; lean_object* v___x_491_; lean_object* v___x_492_; lean_object* v___x_493_; lean_object* v___x_494_; lean_object* v___x_495_; lean_object* v___x_496_; lean_object* v___x_497_; lean_object* v___x_498_; lean_object* v___x_499_; lean_object* v___x_500_; lean_object* v___x_501_; lean_object* v___x_502_; lean_object* v___x_503_; lean_object* v___x_504_; lean_object* v___x_505_; lean_object* v___x_506_; lean_object* v___x_507_; lean_object* v___x_508_; lean_object* v___x_509_; lean_object* v___x_510_; lean_object* v___x_511_; lean_object* v___x_512_; 
v___x_479_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__22));
lean_inc_ref_n(v___x_381_, 2);
v___x_480_ = l_Lean_Name_mkStr3(v___x_381_, v___x_479_, v___x_382_);
v___x_481_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_481_, 0, v___x_480_);
v___x_482_ = l_Lean_Name_mkStr2(v___x_381_, v___x_479_);
v___x_483_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_483_, 0, v___x_482_);
v___x_484_ = l_Lean_Name_mkStr1(v___x_381_);
v___x_485_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_485_, 0, v___x_484_);
v___x_486_ = lean_box(0);
v___x_487_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_487_, 0, v___x_485_);
lean_ctor_set(v___x_487_, 1, v___x_486_);
v___x_488_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_488_, 0, v___x_483_);
lean_ctor_set(v___x_488_, 1, v___x_487_);
v___x_489_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_489_, 0, v___x_481_);
lean_ctor_set(v___x_489_, 1, v___x_488_);
v___x_490_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_490_, 0, v___x_478_);
lean_ctor_set(v___x_490_, 1, v___x_489_);
v___x_491_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_491_, 0, v___x_474_);
lean_ctor_set(v___x_491_, 1, v___x_490_);
lean_inc_n(v___x_435_, 13);
v___x_492_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_492_, 0, v___x_435_);
lean_ctor_set(v___x_492_, 1, v___x_470_);
lean_ctor_set(v___x_492_, 2, v___x_471_);
lean_ctor_set(v___x_492_, 3, v___x_491_);
v___x_493_ = l_Lean_Syntax_node1(v___x_435_, v___x_469_, v___x_492_);
v___x_494_ = l_Lean_Syntax_node2(v___x_435_, v___x_466_, v___x_468_, v___x_493_);
v___x_495_ = l_Lean_Syntax_node1(v___x_435_, v___x_445_, v_a_430_);
v___x_496_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__23));
v___x_497_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_497_, 0, v___x_435_);
lean_ctor_set(v___x_497_, 1, v___x_496_);
lean_inc_ref(v___x_459_);
v___x_498_ = l_Lean_Syntax_node5(v___x_435_, v___x_464_, v___x_494_, v_a_428_, v___x_459_, v___x_495_, v___x_497_);
v___x_499_ = l_Lean_Syntax_node3(v___x_435_, v___x_460_, v_a_393_, v___x_462_, v___x_498_);
v___x_500_ = l_Lean_Syntax_node2(v___x_435_, v___x_457_, v___x_459_, v___x_499_);
v___x_501_ = l_Lean_Syntax_node1(v___x_435_, v___x_445_, v___x_500_);
v___x_502_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__24));
v___x_503_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_503_, 0, v___x_435_);
lean_ctor_set(v___x_503_, 1, v___x_502_);
v___x_504_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__26, &lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__26_once, _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__26);
v___x_505_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__27));
lean_inc(v_currMacroScope_433_);
lean_inc(v_quotContext_432_);
v___x_506_ = l_Lean_addMacroScope(v_quotContext_432_, v___x_505_, v_currMacroScope_433_);
v___x_507_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__29));
v___x_508_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_508_, 0, v___x_435_);
lean_ctor_set(v___x_508_, 1, v___x_504_);
lean_ctor_set(v___x_508_, 2, v___x_506_);
lean_ctor_set(v___x_508_, 3, v___x_507_);
v___x_509_ = l_Lean_Syntax_node5(v___x_435_, v___x_452_, v___x_455_, v___x_447_, v___x_501_, v___x_503_, v___x_508_);
v___x_510_ = l_Lean_Syntax_node1(v___x_435_, v___x_450_, v___x_509_);
v___x_511_ = l_Lean_Syntax_node3(v___x_435_, v___x_438_, v___x_441_, v___x_448_, v___x_510_);
v___x_512_ = l_Lean_Elab_Tactic_evalTactic(v___x_511_, v___y_411_, v___y_412_, v___y_413_, v___y_414_, v___y_415_, v___y_416_, v___y_417_, v___y_418_);
return v___x_512_;
}
}
}
else
{
lean_object* v_a_515_; lean_object* v___x_517_; uint8_t v_isShared_518_; uint8_t v_isSharedCheck_522_; 
lean_dec(v_a_428_);
lean_del_object(v___x_425_);
lean_dec(v_val_423_);
lean_del_object(v___x_421_);
lean_dec(v___y_407_);
lean_dec(v_a_393_);
lean_dec_ref(v___x_385_);
lean_dec_ref(v___x_382_);
lean_dec_ref(v___x_381_);
v_a_515_ = lean_ctor_get(v___x_429_, 0);
v_isSharedCheck_522_ = !lean_is_exclusive(v___x_429_);
if (v_isSharedCheck_522_ == 0)
{
v___x_517_ = v___x_429_;
v_isShared_518_ = v_isSharedCheck_522_;
goto v_resetjp_516_;
}
else
{
lean_inc(v_a_515_);
lean_dec(v___x_429_);
v___x_517_ = lean_box(0);
v_isShared_518_ = v_isSharedCheck_522_;
goto v_resetjp_516_;
}
v_resetjp_516_:
{
lean_object* v___x_520_; 
if (v_isShared_518_ == 0)
{
v___x_520_ = v___x_517_;
goto v_reusejp_519_;
}
else
{
lean_object* v_reuseFailAlloc_521_; 
v_reuseFailAlloc_521_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_521_, 0, v_a_515_);
v___x_520_ = v_reuseFailAlloc_521_;
goto v_reusejp_519_;
}
v_reusejp_519_:
{
return v___x_520_;
}
}
}
}
else
{
lean_object* v_a_523_; lean_object* v___x_525_; uint8_t v_isShared_526_; uint8_t v_isSharedCheck_530_; 
lean_del_object(v___x_425_);
lean_dec(v_val_423_);
lean_del_object(v___x_421_);
lean_dec_ref(v___y_408_);
lean_dec(v___y_407_);
lean_dec(v_a_393_);
lean_dec_ref(v___x_385_);
lean_dec_ref(v___x_382_);
lean_dec_ref(v___x_381_);
v_a_523_ = lean_ctor_get(v___x_427_, 0);
v_isSharedCheck_530_ = !lean_is_exclusive(v___x_427_);
if (v_isSharedCheck_530_ == 0)
{
v___x_525_ = v___x_427_;
v_isShared_526_ = v_isSharedCheck_530_;
goto v_resetjp_524_;
}
else
{
lean_inc(v_a_523_);
lean_dec(v___x_427_);
v___x_525_ = lean_box(0);
v_isShared_526_ = v_isSharedCheck_530_;
goto v_resetjp_524_;
}
v_resetjp_524_:
{
lean_object* v___x_528_; 
if (v_isShared_526_ == 0)
{
v___x_528_ = v___x_525_;
goto v_reusejp_527_;
}
else
{
lean_object* v_reuseFailAlloc_529_; 
v_reuseFailAlloc_529_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_529_, 0, v_a_523_);
v___x_528_ = v_reuseFailAlloc_529_;
goto v_reusejp_527_;
}
v_reusejp_527_:
{
return v___x_528_;
}
}
}
}
}
else
{
lean_object* v_val_532_; lean_object* v___x_534_; uint8_t v_isShared_535_; uint8_t v_isSharedCheck_647_; 
v_val_532_ = lean_ctor_get(v___y_410_, 0);
v_isSharedCheck_647_ = !lean_is_exclusive(v___y_410_);
if (v_isSharedCheck_647_ == 0)
{
v___x_534_ = v___y_410_;
v_isShared_535_ = v_isSharedCheck_647_;
goto v_resetjp_533_;
}
else
{
lean_inc(v_val_532_);
lean_dec(v___y_410_);
v___x_534_ = lean_box(0);
v_isShared_535_ = v_isSharedCheck_647_;
goto v_resetjp_533_;
}
v_resetjp_533_:
{
lean_object* v___x_537_; uint8_t v_isShared_538_; uint8_t v_isSharedCheck_645_; 
v_isSharedCheck_645_ = !lean_is_exclusive(v_val_419_);
if (v_isSharedCheck_645_ == 0)
{
lean_object* v_unused_646_; 
v_unused_646_ = lean_ctor_get(v_val_419_, 0);
lean_dec(v_unused_646_);
v___x_537_ = v_val_419_;
v_isShared_538_ = v_isSharedCheck_645_;
goto v_resetjp_536_;
}
else
{
lean_dec(v_val_419_);
v___x_537_ = lean_box(0);
v_isShared_538_ = v_isSharedCheck_645_;
goto v_resetjp_536_;
}
v_resetjp_536_:
{
lean_object* v___x_539_; 
v___x_539_ = l_Lean_Elab_Term_exprToSyntax(v___y_409_, v___y_413_, v___y_414_, v___y_415_, v___y_416_, v___y_417_, v___y_418_);
if (lean_obj_tag(v___x_539_) == 0)
{
lean_object* v_a_540_; lean_object* v___x_541_; 
v_a_540_ = lean_ctor_get(v___x_539_, 0);
lean_inc(v_a_540_);
lean_dec_ref_known(v___x_539_, 1);
v___x_541_ = l_Lean_Elab_Term_exprToSyntax(v___y_408_, v___y_413_, v___y_414_, v___y_415_, v___y_416_, v___y_417_, v___y_418_);
if (lean_obj_tag(v___x_541_) == 0)
{
lean_object* v_a_542_; lean_object* v_ref_543_; lean_object* v_quotContext_544_; lean_object* v_currMacroScope_545_; uint8_t v___x_546_; lean_object* v___x_547_; lean_object* v___x_548_; lean_object* v___x_549_; lean_object* v___x_550_; lean_object* v___x_551_; lean_object* v___x_552_; lean_object* v___x_553_; lean_object* v___x_554_; lean_object* v___x_555_; lean_object* v___x_556_; lean_object* v___x_557_; lean_object* v___x_558_; lean_object* v___x_559_; lean_object* v___x_560_; lean_object* v___x_561_; lean_object* v___x_562_; lean_object* v___x_563_; lean_object* v___x_564_; lean_object* v___x_565_; lean_object* v___x_566_; lean_object* v___x_567_; lean_object* v___x_568_; lean_object* v___x_569_; lean_object* v___x_570_; lean_object* v___x_571_; lean_object* v___x_572_; lean_object* v___x_573_; lean_object* v___x_574_; lean_object* v___x_575_; lean_object* v___x_576_; lean_object* v___x_577_; lean_object* v___x_578_; lean_object* v___x_579_; lean_object* v___x_580_; lean_object* v___x_581_; lean_object* v___x_582_; lean_object* v___x_584_; 
v_a_542_ = lean_ctor_get(v___x_541_, 0);
lean_inc(v_a_542_);
lean_dec_ref_known(v___x_541_, 1);
v_ref_543_ = lean_ctor_get(v___y_417_, 5);
v_quotContext_544_ = lean_ctor_get(v___y_417_, 10);
v_currMacroScope_545_ = lean_ctor_get(v___y_417_, 11);
v___x_546_ = 0;
v___x_547_ = l_Lean_SourceInfo_fromRef(v_ref_543_, v___x_546_);
v___x_548_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__0));
v___x_549_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__1));
lean_inc_ref_n(v___x_382_, 2);
lean_inc_ref_n(v___x_381_, 8);
v___x_550_ = l_Lean_Name_mkStr4(v___x_381_, v___x_548_, v___x_382_, v___x_549_);
v___x_551_ = l_Lean_SourceInfo_fromRef(v_tk_383_, v___x_384_);
v___x_552_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__2));
v___x_553_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_553_, 0, v___x_551_);
lean_ctor_set(v___x_553_, 1, v___x_552_);
v___x_554_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__3));
v___x_555_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__4));
v___x_556_ = l_Lean_Name_mkStr4(v___x_381_, v___x_548_, v___x_554_, v___x_555_);
v___x_557_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______macroRules__Mathlib__Tactic__tacticSet_x21____1___closed__1));
v___x_558_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__5, &lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__5_once, _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__5);
lean_inc_n(v___x_547_, 5);
v___x_559_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_559_, 0, v___x_547_);
lean_ctor_set(v___x_559_, 1, v___x_557_);
lean_ctor_set(v___x_559_, 2, v___x_558_);
lean_inc_ref(v___x_559_);
v___x_560_ = l_Lean_Syntax_node1(v___x_547_, v___x_556_, v___x_559_);
v___x_561_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__6));
v___x_562_ = l_Lean_Name_mkStr4(v___x_381_, v___x_548_, v___x_554_, v___x_561_);
v___x_563_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__7));
v___x_564_ = l_Lean_Name_mkStr4(v___x_381_, v___x_548_, v___x_554_, v___x_563_);
v___x_565_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__8));
v___x_566_ = l_Lean_Name_mkStr4(v___x_381_, v___x_548_, v___x_554_, v___x_565_);
v___x_567_ = l_Lean_Syntax_node1(v___x_547_, v___x_566_, v_val_532_);
v___x_568_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__9));
v___x_569_ = l_Lean_Name_mkStr4(v___x_381_, v___x_548_, v___x_554_, v___x_568_);
v___x_570_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__10));
v___x_571_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_571_, 0, v___x_547_);
lean_ctor_set(v___x_571_, 1, v___x_570_);
v___x_572_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__12));
v___x_573_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__14));
v___x_574_ = l_Lean_Name_mkStr4(v___x_381_, v___x_548_, v___x_554_, v___x_573_);
v___x_575_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__15));
v___x_576_ = l_Lean_Name_mkStr4(v___x_381_, v___x_548_, v___x_554_, v___x_575_);
v___x_577_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__16));
v___x_578_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_578_, 0, v___x_547_);
lean_ctor_set(v___x_578_, 1, v___x_577_);
v___x_579_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__18));
v___x_580_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__20, &lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__20_once, _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__20);
lean_inc(v_currMacroScope_545_);
lean_inc(v_quotContext_544_);
v___x_581_ = l_Lean_addMacroScope(v_quotContext_544_, v___y_407_, v_currMacroScope_545_);
v___x_582_ = l_Lean_Name_mkStr2(v___x_385_, v___x_382_);
if (v_isShared_538_ == 0)
{
lean_ctor_set_tag(v___x_537_, 0);
lean_ctor_set(v___x_537_, 0, v___x_582_);
v___x_584_ = v___x_537_;
goto v_reusejp_583_;
}
else
{
lean_object* v_reuseFailAlloc_628_; 
v_reuseFailAlloc_628_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_628_, 0, v___x_582_);
v___x_584_ = v_reuseFailAlloc_628_;
goto v_reusejp_583_;
}
v_reusejp_583_:
{
lean_object* v___x_585_; lean_object* v___x_586_; lean_object* v___x_588_; 
v___x_585_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__21));
lean_inc_ref(v___x_381_);
v___x_586_ = l_Lean_Name_mkStr2(v___x_381_, v___x_585_);
if (v_isShared_422_ == 0)
{
lean_ctor_set_tag(v___x_421_, 0);
lean_ctor_set(v___x_421_, 0, v___x_586_);
v___x_588_ = v___x_421_;
goto v_reusejp_587_;
}
else
{
lean_object* v_reuseFailAlloc_627_; 
v_reuseFailAlloc_627_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_627_, 0, v___x_586_);
v___x_588_ = v_reuseFailAlloc_627_;
goto v_reusejp_587_;
}
v_reusejp_587_:
{
lean_object* v___x_589_; lean_object* v___x_590_; lean_object* v___x_592_; 
v___x_589_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__22));
lean_inc_ref(v___x_381_);
v___x_590_ = l_Lean_Name_mkStr3(v___x_381_, v___x_589_, v___x_382_);
if (v_isShared_535_ == 0)
{
lean_ctor_set_tag(v___x_534_, 0);
lean_ctor_set(v___x_534_, 0, v___x_590_);
v___x_592_ = v___x_534_;
goto v_reusejp_591_;
}
else
{
lean_object* v_reuseFailAlloc_626_; 
v_reuseFailAlloc_626_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_626_, 0, v___x_590_);
v___x_592_ = v_reuseFailAlloc_626_;
goto v_reusejp_591_;
}
v_reusejp_591_:
{
lean_object* v___x_593_; lean_object* v___x_594_; lean_object* v___x_595_; lean_object* v___x_596_; lean_object* v___x_597_; lean_object* v___x_598_; lean_object* v___x_599_; lean_object* v___x_600_; lean_object* v___x_601_; lean_object* v___x_602_; lean_object* v___x_603_; lean_object* v___x_604_; lean_object* v___x_605_; lean_object* v___x_606_; lean_object* v___x_607_; lean_object* v___x_608_; lean_object* v___x_609_; lean_object* v___x_610_; lean_object* v___x_611_; lean_object* v___x_612_; lean_object* v___x_613_; lean_object* v___x_614_; lean_object* v___x_615_; lean_object* v___x_616_; lean_object* v___x_617_; lean_object* v___x_618_; lean_object* v___x_619_; lean_object* v___x_620_; lean_object* v___x_621_; lean_object* v___x_622_; lean_object* v___x_623_; lean_object* v___x_624_; lean_object* v___x_625_; 
lean_inc_ref(v___x_381_);
v___x_593_ = l_Lean_Name_mkStr2(v___x_381_, v___x_589_);
v___x_594_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_594_, 0, v___x_593_);
v___x_595_ = l_Lean_Name_mkStr1(v___x_381_);
v___x_596_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_596_, 0, v___x_595_);
v___x_597_ = lean_box(0);
v___x_598_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_598_, 0, v___x_596_);
lean_ctor_set(v___x_598_, 1, v___x_597_);
v___x_599_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_599_, 0, v___x_594_);
lean_ctor_set(v___x_599_, 1, v___x_598_);
v___x_600_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_600_, 0, v___x_592_);
lean_ctor_set(v___x_600_, 1, v___x_599_);
v___x_601_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_601_, 0, v___x_588_);
lean_ctor_set(v___x_601_, 1, v___x_600_);
v___x_602_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_602_, 0, v___x_584_);
lean_ctor_set(v___x_602_, 1, v___x_601_);
lean_inc_n(v___x_547_, 14);
v___x_603_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_603_, 0, v___x_547_);
lean_ctor_set(v___x_603_, 1, v___x_580_);
lean_ctor_set(v___x_603_, 2, v___x_581_);
lean_ctor_set(v___x_603_, 3, v___x_602_);
v___x_604_ = l_Lean_Syntax_node1(v___x_547_, v___x_579_, v___x_603_);
v___x_605_ = l_Lean_Syntax_node2(v___x_547_, v___x_576_, v___x_578_, v___x_604_);
v___x_606_ = l_Lean_Syntax_node1(v___x_547_, v___x_557_, v_a_542_);
v___x_607_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__23));
v___x_608_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_608_, 0, v___x_547_);
lean_ctor_set(v___x_608_, 1, v___x_607_);
lean_inc_ref(v___x_571_);
v___x_609_ = l_Lean_Syntax_node5(v___x_547_, v___x_574_, v___x_605_, v_a_540_, v___x_571_, v___x_606_, v___x_608_);
v___x_610_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__13));
v___x_611_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_611_, 0, v___x_547_);
lean_ctor_set(v___x_611_, 1, v___x_610_);
v___x_612_ = l_Lean_Syntax_node3(v___x_547_, v___x_572_, v___x_609_, v___x_611_, v_a_393_);
v___x_613_ = l_Lean_Syntax_node2(v___x_547_, v___x_569_, v___x_571_, v___x_612_);
v___x_614_ = l_Lean_Syntax_node1(v___x_547_, v___x_557_, v___x_613_);
v___x_615_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__24));
v___x_616_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_616_, 0, v___x_547_);
lean_ctor_set(v___x_616_, 1, v___x_615_);
v___x_617_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__26, &lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__26_once, _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__26);
v___x_618_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__27));
lean_inc(v_currMacroScope_545_);
lean_inc(v_quotContext_544_);
v___x_619_ = l_Lean_addMacroScope(v_quotContext_544_, v___x_618_, v_currMacroScope_545_);
v___x_620_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__29));
v___x_621_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_621_, 0, v___x_547_);
lean_ctor_set(v___x_621_, 1, v___x_617_);
lean_ctor_set(v___x_621_, 2, v___x_619_);
lean_ctor_set(v___x_621_, 3, v___x_620_);
v___x_622_ = l_Lean_Syntax_node5(v___x_547_, v___x_564_, v___x_567_, v___x_559_, v___x_614_, v___x_616_, v___x_621_);
v___x_623_ = l_Lean_Syntax_node1(v___x_547_, v___x_562_, v___x_622_);
v___x_624_ = l_Lean_Syntax_node3(v___x_547_, v___x_550_, v___x_553_, v___x_560_, v___x_623_);
v___x_625_ = l_Lean_Elab_Tactic_evalTactic(v___x_624_, v___y_411_, v___y_412_, v___y_413_, v___y_414_, v___y_415_, v___y_416_, v___y_417_, v___y_418_);
return v___x_625_;
}
}
}
}
else
{
lean_object* v_a_629_; lean_object* v___x_631_; uint8_t v_isShared_632_; uint8_t v_isSharedCheck_636_; 
lean_dec(v_a_540_);
lean_del_object(v___x_537_);
lean_del_object(v___x_534_);
lean_dec(v_val_532_);
lean_del_object(v___x_421_);
lean_dec(v___y_407_);
lean_dec(v_a_393_);
lean_dec_ref(v___x_385_);
lean_dec_ref(v___x_382_);
lean_dec_ref(v___x_381_);
v_a_629_ = lean_ctor_get(v___x_541_, 0);
v_isSharedCheck_636_ = !lean_is_exclusive(v___x_541_);
if (v_isSharedCheck_636_ == 0)
{
v___x_631_ = v___x_541_;
v_isShared_632_ = v_isSharedCheck_636_;
goto v_resetjp_630_;
}
else
{
lean_inc(v_a_629_);
lean_dec(v___x_541_);
v___x_631_ = lean_box(0);
v_isShared_632_ = v_isSharedCheck_636_;
goto v_resetjp_630_;
}
v_resetjp_630_:
{
lean_object* v___x_634_; 
if (v_isShared_632_ == 0)
{
v___x_634_ = v___x_631_;
goto v_reusejp_633_;
}
else
{
lean_object* v_reuseFailAlloc_635_; 
v_reuseFailAlloc_635_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_635_, 0, v_a_629_);
v___x_634_ = v_reuseFailAlloc_635_;
goto v_reusejp_633_;
}
v_reusejp_633_:
{
return v___x_634_;
}
}
}
}
else
{
lean_object* v_a_637_; lean_object* v___x_639_; uint8_t v_isShared_640_; uint8_t v_isSharedCheck_644_; 
lean_del_object(v___x_537_);
lean_del_object(v___x_534_);
lean_dec(v_val_532_);
lean_del_object(v___x_421_);
lean_dec_ref(v___y_408_);
lean_dec(v___y_407_);
lean_dec(v_a_393_);
lean_dec_ref(v___x_385_);
lean_dec_ref(v___x_382_);
lean_dec_ref(v___x_381_);
v_a_637_ = lean_ctor_get(v___x_539_, 0);
v_isSharedCheck_644_ = !lean_is_exclusive(v___x_539_);
if (v_isSharedCheck_644_ == 0)
{
v___x_639_ = v___x_539_;
v_isShared_640_ = v_isSharedCheck_644_;
goto v_resetjp_638_;
}
else
{
lean_inc(v_a_637_);
lean_dec(v___x_539_);
v___x_639_ = lean_box(0);
v_isShared_640_ = v_isSharedCheck_644_;
goto v_resetjp_638_;
}
v_resetjp_638_:
{
lean_object* v___x_642_; 
if (v_isShared_640_ == 0)
{
v___x_642_ = v___x_639_;
goto v_reusejp_641_;
}
else
{
lean_object* v_reuseFailAlloc_643_; 
v_reuseFailAlloc_643_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_643_, 0, v_a_637_);
v___x_642_ = v_reuseFailAlloc_643_;
goto v_reusejp_641_;
}
v_reusejp_641_:
{
return v___x_642_;
}
}
}
}
}
}
}
}
else
{
lean_dec_ref_known(v___y_410_, 1);
lean_dec_ref(v___y_409_);
lean_dec_ref(v___y_408_);
lean_dec(v___y_407_);
lean_dec(v_a_393_);
lean_dec_ref(v___x_385_);
lean_dec_ref(v___x_382_);
lean_dec_ref(v___x_381_);
lean_dec(v_rev_380_);
goto v___jp_403_;
}
}
else
{
lean_dec(v___y_410_);
lean_dec_ref(v___y_409_);
lean_dec_ref(v___y_408_);
lean_dec(v___y_407_);
lean_dec(v_a_393_);
lean_dec_ref(v___x_385_);
lean_dec_ref(v___x_382_);
lean_dec_ref(v___x_381_);
lean_dec(v_rev_380_);
goto v___jp_403_;
}
}
v___jp_649_:
{
lean_object* v___x_653_; lean_object* v___f_654_; lean_object* v___x_655_; 
v___x_653_ = lean_box(v___x_386_);
lean_inc_ref(v_snd_652_);
lean_inc_ref(v_fst_651_);
lean_inc(v_a_393_);
v___f_654_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__0___boxed), 13, 4);
lean_closure_set(v___f_654_, 0, v_a_393_);
lean_closure_set(v___f_654_, 1, v_fst_651_);
lean_closure_set(v___f_654_, 2, v_snd_652_);
lean_closure_set(v___f_654_, 3, v___x_653_);
v___x_655_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___f_654_, v___y_394_, v___y_395_, v___y_396_, v___y_397_, v___y_398_, v___y_399_, v___y_400_, v___y_401_);
if (lean_obj_tag(v___x_655_) == 0)
{
lean_object* v_a_656_; lean_object* v___x_657_; lean_object* v___x_658_; lean_object* v___x_659_; uint8_t v___x_660_; lean_object* v___x_661_; lean_object* v___x_662_; lean_object* v___f_663_; lean_object* v___x_664_; 
v_a_656_ = lean_ctor_get(v___x_655_, 0);
lean_inc(v_a_656_);
lean_dec_ref_known(v___x_655_, 1);
v___x_657_ = l_Lean_mkFVar(v_a_656_);
v___x_658_ = lean_box(0);
v___x_659_ = lean_box(0);
v___x_660_ = 0;
v___x_661_ = lean_box(v___x_384_);
v___x_662_ = lean_box(v___x_660_);
lean_inc(v_a_393_);
v___f_663_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__1___boxed), 16, 7);
lean_closure_set(v___f_663_, 0, v_a_393_);
lean_closure_set(v___f_663_, 1, v___x_657_);
lean_closure_set(v___f_663_, 2, v___x_658_);
lean_closure_set(v___f_663_, 3, v___x_658_);
lean_closure_set(v___f_663_, 4, v___x_659_);
lean_closure_set(v___f_663_, 5, v___x_661_);
lean_closure_set(v___f_663_, 6, v___x_662_);
v___x_664_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___f_663_, v___y_394_, v___y_395_, v___y_396_, v___y_397_, v___y_398_, v___y_399_, v___y_400_, v___y_401_);
if (lean_obj_tag(v___x_664_) == 0)
{
lean_dec_ref_known(v___x_664_, 1);
if (lean_obj_tag(v_rw_387_) == 0)
{
if (v___x_386_ == 0)
{
v___y_407_ = v___x_659_;
v___y_408_ = v_fst_651_;
v___y_409_ = v_snd_652_;
v___y_410_ = v___y_650_;
v___y_411_ = v___y_394_;
v___y_412_ = v___y_395_;
v___y_413_ = v___y_396_;
v___y_414_ = v___y_397_;
v___y_415_ = v___y_398_;
v___y_416_ = v___y_399_;
v___y_417_ = v___y_400_;
v___y_418_ = v___y_401_;
goto v___jp_406_;
}
else
{
lean_object* v___x_665_; 
lean_inc_ref(v_snd_652_);
v___x_665_ = l_Lean_Elab_Term_exprToSyntax(v_snd_652_, v___y_396_, v___y_397_, v___y_398_, v___y_399_, v___y_400_, v___y_401_);
if (lean_obj_tag(v___x_665_) == 0)
{
lean_object* v_a_666_; lean_object* v_ref_667_; lean_object* v_quotContext_668_; lean_object* v_currMacroScope_669_; lean_object* v___x_670_; lean_object* v___x_671_; lean_object* v___x_672_; lean_object* v___x_673_; lean_object* v___x_674_; lean_object* v___x_675_; lean_object* v___x_676_; lean_object* v___x_677_; lean_object* v___x_678_; lean_object* v___x_679_; lean_object* v___x_680_; lean_object* v___x_681_; lean_object* v___x_682_; lean_object* v___x_683_; lean_object* v___x_684_; lean_object* v___x_685_; lean_object* v___x_686_; lean_object* v___x_687_; lean_object* v___x_688_; lean_object* v___x_689_; lean_object* v___x_690_; lean_object* v___x_691_; lean_object* v___x_692_; lean_object* v___x_693_; lean_object* v___x_694_; lean_object* v___x_695_; lean_object* v___x_696_; lean_object* v___x_697_; lean_object* v___x_698_; lean_object* v___x_699_; lean_object* v___x_700_; lean_object* v___x_701_; lean_object* v___x_702_; lean_object* v___x_703_; lean_object* v___x_704_; lean_object* v___x_705_; lean_object* v___x_706_; lean_object* v___x_707_; lean_object* v___x_708_; lean_object* v___x_709_; lean_object* v___x_710_; lean_object* v___x_711_; lean_object* v___x_712_; lean_object* v___x_713_; lean_object* v___x_714_; lean_object* v___x_715_; lean_object* v___x_716_; lean_object* v___x_717_; lean_object* v___x_718_; lean_object* v___x_719_; lean_object* v___x_720_; lean_object* v___x_721_; lean_object* v___x_722_; lean_object* v___x_723_; lean_object* v___x_724_; lean_object* v___x_725_; lean_object* v___x_726_; lean_object* v___x_727_; lean_object* v___x_728_; lean_object* v___x_729_; lean_object* v___x_730_; lean_object* v___x_731_; lean_object* v___x_732_; lean_object* v___x_733_; lean_object* v___x_734_; lean_object* v___x_735_; lean_object* v___x_736_; 
v_a_666_ = lean_ctor_get(v___x_665_, 0);
lean_inc(v_a_666_);
lean_dec_ref_known(v___x_665_, 1);
v_ref_667_ = lean_ctor_get(v___y_400_, 5);
v_quotContext_668_ = lean_ctor_get(v___y_400_, 10);
v_currMacroScope_669_ = lean_ctor_get(v___y_400_, 11);
v___x_670_ = l_Lean_SourceInfo_fromRef(v_ref_667_, v___x_660_);
v___x_671_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__0));
v___x_672_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__30));
lean_inc_ref_n(v___x_382_, 9);
lean_inc_ref_n(v___x_381_, 11);
v___x_673_ = l_Lean_Name_mkStr4(v___x_381_, v___x_671_, v___x_382_, v___x_672_);
v___x_674_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__31));
lean_inc_n(v___x_670_, 25);
v___x_675_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_675_, 0, v___x_670_);
lean_ctor_set(v___x_675_, 1, v___x_674_);
v___x_676_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__32));
v___x_677_ = l_Lean_Name_mkStr4(v___x_381_, v___x_671_, v___x_382_, v___x_676_);
v___x_678_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__33));
v___x_679_ = l_Lean_Name_mkStr4(v___x_381_, v___x_671_, v___x_382_, v___x_678_);
v___x_680_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______macroRules__Mathlib__Tactic__tacticSet_x21____1___closed__1));
v___x_681_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__34));
v___x_682_ = l_Lean_Name_mkStr4(v___x_381_, v___x_671_, v___x_382_, v___x_681_);
v___x_683_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__35));
v___x_684_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_684_, 0, v___x_670_);
lean_ctor_set(v___x_684_, 1, v___x_683_);
v___x_685_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__36));
v___x_686_ = l_Lean_Name_mkStr4(v___x_381_, v___x_671_, v___x_382_, v___x_685_);
v___x_687_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__5, &lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__5_once, _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__5);
v___x_688_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_688_, 0, v___x_670_);
lean_ctor_set(v___x_688_, 1, v___x_680_);
lean_ctor_set(v___x_688_, 2, v___x_687_);
lean_inc_ref(v___x_688_);
v___x_689_ = l_Lean_Syntax_node1(v___x_670_, v___x_686_, v___x_688_);
v___x_690_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__37));
v___x_691_ = l_Lean_Name_mkStr4(v___x_381_, v___x_671_, v___x_382_, v___x_690_);
v___x_692_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__38));
v___x_693_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_693_, 0, v___x_670_);
lean_ctor_set(v___x_693_, 1, v___x_692_);
v___x_694_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__39));
v___x_695_ = l_Lean_Name_mkStr4(v___x_381_, v___x_671_, v___x_382_, v___x_694_);
v___x_696_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__3));
v___x_697_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__40));
v___x_698_ = l_Lean_Name_mkStr4(v___x_381_, v___x_671_, v___x_696_, v___x_697_);
v___x_699_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_699_, 0, v___x_670_);
lean_ctor_set(v___x_699_, 1, v___x_697_);
v___x_700_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__12));
v___x_701_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__13));
v___x_702_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_702_, 0, v___x_670_);
lean_ctor_set(v___x_702_, 1, v___x_701_);
lean_inc(v_a_393_);
v___x_703_ = l_Lean_Syntax_node3(v___x_670_, v___x_700_, v_a_666_, v___x_702_, v_a_393_);
v___x_704_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__41));
v___x_705_ = l_Lean_Name_mkStr4(v___x_381_, v___x_671_, v___x_696_, v___x_704_);
v___x_706_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__42));
v___x_707_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_707_, 0, v___x_670_);
lean_ctor_set(v___x_707_, 1, v___x_706_);
v___x_708_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__26, &lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__26_once, _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__26);
v___x_709_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__27));
lean_inc(v_currMacroScope_669_);
lean_inc(v_quotContext_668_);
v___x_710_ = l_Lean_addMacroScope(v_quotContext_668_, v___x_709_, v_currMacroScope_669_);
v___x_711_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__44));
v___x_712_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_712_, 0, v___x_670_);
lean_ctor_set(v___x_712_, 1, v___x_708_);
lean_ctor_set(v___x_712_, 2, v___x_710_);
lean_ctor_set(v___x_712_, 3, v___x_711_);
v___x_713_ = l_Lean_Syntax_node2(v___x_670_, v___x_705_, v___x_707_, v___x_712_);
v___x_714_ = l_Lean_Syntax_node3(v___x_670_, v___x_698_, v___x_699_, v___x_703_, v___x_713_);
v___x_715_ = l_Lean_Syntax_node2(v___x_670_, v___x_695_, v___x_688_, v___x_714_);
v___x_716_ = l_Lean_Syntax_node1(v___x_670_, v___x_680_, v___x_715_);
v___x_717_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__45));
v___x_718_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_718_, 0, v___x_670_);
lean_ctor_set(v___x_718_, 1, v___x_717_);
v___x_719_ = l_Lean_Syntax_node3(v___x_670_, v___x_691_, v___x_693_, v___x_716_, v___x_718_);
v___x_720_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__46));
v___x_721_ = l_Lean_Name_mkStr4(v___x_381_, v___x_671_, v___x_382_, v___x_720_);
v___x_722_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__47));
v___x_723_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_723_, 0, v___x_670_);
lean_ctor_set(v___x_723_, 1, v___x_722_);
v___x_724_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__48));
v___x_725_ = l_Lean_Name_mkStr4(v___x_381_, v___x_671_, v___x_382_, v___x_724_);
v___x_726_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__49));
v___x_727_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_727_, 0, v___x_670_);
lean_ctor_set(v___x_727_, 1, v___x_726_);
v___x_728_ = l_Lean_Syntax_node1(v___x_670_, v___x_725_, v___x_727_);
v___x_729_ = l_Lean_Syntax_node2(v___x_670_, v___x_721_, v___x_723_, v___x_728_);
v___x_730_ = l_Lean_Syntax_node1(v___x_670_, v___x_680_, v___x_729_);
v___x_731_ = l_Lean_Syntax_node4(v___x_670_, v___x_682_, v___x_684_, v___x_689_, v___x_719_, v___x_730_);
v___x_732_ = l_Lean_Syntax_node1(v___x_670_, v___x_680_, v___x_731_);
v___x_733_ = l_Lean_Syntax_node1(v___x_670_, v___x_679_, v___x_732_);
v___x_734_ = l_Lean_Syntax_node1(v___x_670_, v___x_677_, v___x_733_);
v___x_735_ = l_Lean_Syntax_node2(v___x_670_, v___x_673_, v___x_675_, v___x_734_);
v___x_736_ = l_Lean_Elab_Tactic_evalTactic(v___x_735_, v___y_394_, v___y_395_, v___y_396_, v___y_397_, v___y_398_, v___y_399_, v___y_400_, v___y_401_);
if (lean_obj_tag(v___x_736_) == 0)
{
lean_dec_ref_known(v___x_736_, 1);
v___y_407_ = v___x_659_;
v___y_408_ = v_fst_651_;
v___y_409_ = v_snd_652_;
v___y_410_ = v___y_650_;
v___y_411_ = v___y_394_;
v___y_412_ = v___y_395_;
v___y_413_ = v___y_396_;
v___y_414_ = v___y_397_;
v___y_415_ = v___y_398_;
v___y_416_ = v___y_399_;
v___y_417_ = v___y_400_;
v___y_418_ = v___y_401_;
goto v___jp_406_;
}
else
{
lean_dec_ref(v_snd_652_);
lean_dec_ref(v_fst_651_);
lean_dec(v___y_650_);
lean_dec(v_a_393_);
lean_dec_ref(v___x_385_);
lean_dec_ref(v___x_382_);
lean_dec_ref(v___x_381_);
lean_dec(v_rev_380_);
return v___x_736_;
}
}
else
{
lean_object* v_a_737_; lean_object* v___x_739_; uint8_t v_isShared_740_; uint8_t v_isSharedCheck_744_; 
lean_dec_ref(v_snd_652_);
lean_dec_ref(v_fst_651_);
lean_dec(v___y_650_);
lean_dec(v_a_393_);
lean_dec_ref(v___x_385_);
lean_dec_ref(v___x_382_);
lean_dec_ref(v___x_381_);
lean_dec(v_rev_380_);
v_a_737_ = lean_ctor_get(v___x_665_, 0);
v_isSharedCheck_744_ = !lean_is_exclusive(v___x_665_);
if (v_isSharedCheck_744_ == 0)
{
v___x_739_ = v___x_665_;
v_isShared_740_ = v_isSharedCheck_744_;
goto v_resetjp_738_;
}
else
{
lean_inc(v_a_737_);
lean_dec(v___x_665_);
v___x_739_ = lean_box(0);
v_isShared_740_ = v_isSharedCheck_744_;
goto v_resetjp_738_;
}
v_resetjp_738_:
{
lean_object* v___x_742_; 
if (v_isShared_740_ == 0)
{
v___x_742_ = v___x_739_;
goto v_reusejp_741_;
}
else
{
lean_object* v_reuseFailAlloc_743_; 
v_reuseFailAlloc_743_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_743_, 0, v_a_737_);
v___x_742_ = v_reuseFailAlloc_743_;
goto v_reusejp_741_;
}
v_reusejp_741_:
{
return v___x_742_;
}
}
}
}
}
else
{
v___y_407_ = v___x_659_;
v___y_408_ = v_fst_651_;
v___y_409_ = v_snd_652_;
v___y_410_ = v___y_650_;
v___y_411_ = v___y_394_;
v___y_412_ = v___y_395_;
v___y_413_ = v___y_396_;
v___y_414_ = v___y_397_;
v___y_415_ = v___y_398_;
v___y_416_ = v___y_399_;
v___y_417_ = v___y_400_;
v___y_418_ = v___y_401_;
goto v___jp_406_;
}
}
else
{
lean_dec_ref(v_snd_652_);
lean_dec_ref(v_fst_651_);
lean_dec(v___y_650_);
lean_dec(v_a_393_);
lean_dec_ref(v___x_385_);
lean_dec_ref(v___x_382_);
lean_dec_ref(v___x_381_);
lean_dec(v_rev_380_);
return v___x_664_;
}
}
else
{
lean_object* v_a_745_; lean_object* v___x_747_; uint8_t v_isShared_748_; uint8_t v_isSharedCheck_752_; 
lean_dec_ref(v_snd_652_);
lean_dec_ref(v_fst_651_);
lean_dec(v___y_650_);
lean_dec(v_a_393_);
lean_dec_ref(v___x_385_);
lean_dec_ref(v___x_382_);
lean_dec_ref(v___x_381_);
lean_dec(v_rev_380_);
v_a_745_ = lean_ctor_get(v___x_655_, 0);
v_isSharedCheck_752_ = !lean_is_exclusive(v___x_655_);
if (v_isSharedCheck_752_ == 0)
{
v___x_747_ = v___x_655_;
v_isShared_748_ = v_isSharedCheck_752_;
goto v_resetjp_746_;
}
else
{
lean_inc(v_a_745_);
lean_dec(v___x_655_);
v___x_747_ = lean_box(0);
v_isShared_748_ = v_isSharedCheck_752_;
goto v_resetjp_746_;
}
v_resetjp_746_:
{
lean_object* v___x_750_; 
if (v_isShared_748_ == 0)
{
v___x_750_ = v___x_747_;
goto v_reusejp_749_;
}
else
{
lean_object* v_reuseFailAlloc_751_; 
v_reuseFailAlloc_751_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_751_, 0, v_a_745_);
v___x_750_ = v_reuseFailAlloc_751_;
goto v_reusejp_749_;
}
v_reusejp_749_:
{
return v___x_750_;
}
}
}
}
v___jp_753_:
{
if (lean_obj_tag(v_ty_388_) == 0)
{
lean_object* v___x_755_; uint8_t v___x_756_; lean_object* v___x_757_; 
v___x_755_ = lean_box(0);
v___x_756_ = 0;
v___x_757_ = l_Lean_Elab_Tactic_elabTerm(v___x_389_, v___x_755_, v___x_756_, v___y_394_, v___y_395_, v___y_396_, v___y_397_, v___y_398_, v___y_399_, v___y_400_, v___y_401_);
if (lean_obj_tag(v___x_757_) == 0)
{
lean_object* v_a_758_; lean_object* v___x_759_; 
v_a_758_ = lean_ctor_get(v___x_757_, 0);
lean_inc_n(v_a_758_, 2);
lean_dec_ref_known(v___x_757_, 1);
lean_inc(v___y_401_);
lean_inc_ref(v___y_400_);
lean_inc(v___y_399_);
lean_inc_ref(v___y_398_);
v___x_759_ = lean_infer_type(v_a_758_, v___y_398_, v___y_399_, v___y_400_, v___y_401_);
if (lean_obj_tag(v___x_759_) == 0)
{
lean_object* v_a_760_; 
v_a_760_ = lean_ctor_get(v___x_759_, 0);
lean_inc(v_a_760_);
lean_dec_ref_known(v___x_759_, 1);
v___y_650_ = v_a_754_;
v_fst_651_ = v_a_760_;
v_snd_652_ = v_a_758_;
goto v___jp_649_;
}
else
{
lean_object* v_a_761_; lean_object* v___x_763_; uint8_t v_isShared_764_; uint8_t v_isSharedCheck_768_; 
lean_dec(v_a_758_);
lean_dec(v_a_754_);
lean_dec(v_a_393_);
lean_dec_ref(v___x_385_);
lean_dec_ref(v___x_382_);
lean_dec_ref(v___x_381_);
lean_dec(v_rev_380_);
v_a_761_ = lean_ctor_get(v___x_759_, 0);
v_isSharedCheck_768_ = !lean_is_exclusive(v___x_759_);
if (v_isSharedCheck_768_ == 0)
{
v___x_763_ = v___x_759_;
v_isShared_764_ = v_isSharedCheck_768_;
goto v_resetjp_762_;
}
else
{
lean_inc(v_a_761_);
lean_dec(v___x_759_);
v___x_763_ = lean_box(0);
v_isShared_764_ = v_isSharedCheck_768_;
goto v_resetjp_762_;
}
v_resetjp_762_:
{
lean_object* v___x_766_; 
if (v_isShared_764_ == 0)
{
v___x_766_ = v___x_763_;
goto v_reusejp_765_;
}
else
{
lean_object* v_reuseFailAlloc_767_; 
v_reuseFailAlloc_767_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_767_, 0, v_a_761_);
v___x_766_ = v_reuseFailAlloc_767_;
goto v_reusejp_765_;
}
v_reusejp_765_:
{
return v___x_766_;
}
}
}
}
else
{
lean_object* v_a_769_; lean_object* v___x_771_; uint8_t v_isShared_772_; uint8_t v_isSharedCheck_776_; 
lean_dec(v_a_754_);
lean_dec(v_a_393_);
lean_dec_ref(v___x_385_);
lean_dec_ref(v___x_382_);
lean_dec_ref(v___x_381_);
lean_dec(v_rev_380_);
v_a_769_ = lean_ctor_get(v___x_757_, 0);
v_isSharedCheck_776_ = !lean_is_exclusive(v___x_757_);
if (v_isSharedCheck_776_ == 0)
{
v___x_771_ = v___x_757_;
v_isShared_772_ = v_isSharedCheck_776_;
goto v_resetjp_770_;
}
else
{
lean_inc(v_a_769_);
lean_dec(v___x_757_);
v___x_771_ = lean_box(0);
v_isShared_772_ = v_isSharedCheck_776_;
goto v_resetjp_770_;
}
v_resetjp_770_:
{
lean_object* v___x_774_; 
if (v_isShared_772_ == 0)
{
v___x_774_ = v___x_771_;
goto v_reusejp_773_;
}
else
{
lean_object* v_reuseFailAlloc_775_; 
v_reuseFailAlloc_775_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_775_, 0, v_a_769_);
v___x_774_ = v_reuseFailAlloc_775_;
goto v_reusejp_773_;
}
v_reusejp_773_:
{
return v___x_774_;
}
}
}
}
else
{
lean_object* v_val_777_; lean_object* v___x_779_; uint8_t v_isShared_780_; uint8_t v_isSharedCheck_805_; 
v_val_777_ = lean_ctor_get(v_ty_388_, 0);
v_isSharedCheck_805_ = !lean_is_exclusive(v_ty_388_);
if (v_isSharedCheck_805_ == 0)
{
v___x_779_ = v_ty_388_;
v_isShared_780_ = v_isSharedCheck_805_;
goto v_resetjp_778_;
}
else
{
lean_inc(v_val_777_);
lean_dec(v_ty_388_);
v___x_779_ = lean_box(0);
v_isShared_780_ = v_isSharedCheck_805_;
goto v_resetjp_778_;
}
v_resetjp_778_:
{
lean_object* v___x_781_; 
v___x_781_ = l_Lean_Elab_Term_elabType(v_val_777_, v___y_396_, v___y_397_, v___y_398_, v___y_399_, v___y_400_, v___y_401_);
if (lean_obj_tag(v___x_781_) == 0)
{
lean_object* v_a_782_; lean_object* v___x_784_; 
v_a_782_ = lean_ctor_get(v___x_781_, 0);
lean_inc_n(v_a_782_, 2);
lean_dec_ref_known(v___x_781_, 1);
if (v_isShared_780_ == 0)
{
lean_ctor_set(v___x_779_, 0, v_a_782_);
v___x_784_ = v___x_779_;
goto v_reusejp_783_;
}
else
{
lean_object* v_reuseFailAlloc_796_; 
v_reuseFailAlloc_796_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_796_, 0, v_a_782_);
v___x_784_ = v_reuseFailAlloc_796_;
goto v_reusejp_783_;
}
v_reusejp_783_:
{
uint8_t v___x_785_; lean_object* v___x_786_; 
v___x_785_ = 0;
v___x_786_ = l_Lean_Elab_Tactic_elabTermEnsuringType(v___x_389_, v___x_784_, v___x_785_, v___y_394_, v___y_395_, v___y_396_, v___y_397_, v___y_398_, v___y_399_, v___y_400_, v___y_401_);
if (lean_obj_tag(v___x_786_) == 0)
{
lean_object* v_a_787_; 
v_a_787_ = lean_ctor_get(v___x_786_, 0);
lean_inc(v_a_787_);
lean_dec_ref_known(v___x_786_, 1);
v___y_650_ = v_a_754_;
v_fst_651_ = v_a_782_;
v_snd_652_ = v_a_787_;
goto v___jp_649_;
}
else
{
lean_object* v_a_788_; lean_object* v___x_790_; uint8_t v_isShared_791_; uint8_t v_isSharedCheck_795_; 
lean_dec(v_a_782_);
lean_dec(v_a_754_);
lean_dec(v_a_393_);
lean_dec_ref(v___x_385_);
lean_dec_ref(v___x_382_);
lean_dec_ref(v___x_381_);
lean_dec(v_rev_380_);
v_a_788_ = lean_ctor_get(v___x_786_, 0);
v_isSharedCheck_795_ = !lean_is_exclusive(v___x_786_);
if (v_isSharedCheck_795_ == 0)
{
v___x_790_ = v___x_786_;
v_isShared_791_ = v_isSharedCheck_795_;
goto v_resetjp_789_;
}
else
{
lean_inc(v_a_788_);
lean_dec(v___x_786_);
v___x_790_ = lean_box(0);
v_isShared_791_ = v_isSharedCheck_795_;
goto v_resetjp_789_;
}
v_resetjp_789_:
{
lean_object* v___x_793_; 
if (v_isShared_791_ == 0)
{
v___x_793_ = v___x_790_;
goto v_reusejp_792_;
}
else
{
lean_object* v_reuseFailAlloc_794_; 
v_reuseFailAlloc_794_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_794_, 0, v_a_788_);
v___x_793_ = v_reuseFailAlloc_794_;
goto v_reusejp_792_;
}
v_reusejp_792_:
{
return v___x_793_;
}
}
}
}
}
else
{
lean_object* v_a_797_; lean_object* v___x_799_; uint8_t v_isShared_800_; uint8_t v_isSharedCheck_804_; 
lean_del_object(v___x_779_);
lean_dec(v_a_754_);
lean_dec(v_a_393_);
lean_dec(v___x_389_);
lean_dec_ref(v___x_385_);
lean_dec_ref(v___x_382_);
lean_dec_ref(v___x_381_);
lean_dec(v_rev_380_);
v_a_797_ = lean_ctor_get(v___x_781_, 0);
v_isSharedCheck_804_ = !lean_is_exclusive(v___x_781_);
if (v_isSharedCheck_804_ == 0)
{
v___x_799_ = v___x_781_;
v_isShared_800_ = v_isSharedCheck_804_;
goto v_resetjp_798_;
}
else
{
lean_inc(v_a_797_);
lean_dec(v___x_781_);
v___x_799_ = lean_box(0);
v_isShared_800_ = v_isSharedCheck_804_;
goto v_resetjp_798_;
}
v_resetjp_798_:
{
lean_object* v___x_802_; 
if (v_isShared_800_ == 0)
{
v___x_802_ = v___x_799_;
goto v_reusejp_801_;
}
else
{
lean_object* v_reuseFailAlloc_803_; 
v_reuseFailAlloc_803_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_803_, 0, v_a_797_);
v___x_802_ = v_reuseFailAlloc_803_;
goto v_reusejp_801_;
}
v_reusejp_801_:
{
return v___x_802_;
}
}
}
}
}
}
v___jp_806_:
{
lean_object* v___x_808_; 
v___x_808_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_808_, 0, v_a_807_);
v_a_754_ = v___x_808_;
goto v___jp_753_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___boxed(lean_object** _args){
lean_object* v_rev_832_ = _args[0];
lean_object* v___x_833_ = _args[1];
lean_object* v___x_834_ = _args[2];
lean_object* v_tk_835_ = _args[3];
lean_object* v___x_836_ = _args[4];
lean_object* v___x_837_ = _args[5];
lean_object* v___x_838_ = _args[6];
lean_object* v_rw_839_ = _args[7];
lean_object* v_ty_840_ = _args[8];
lean_object* v___x_841_ = _args[9];
lean_object* v_h_842_ = _args[10];
lean_object* v___x_843_ = _args[11];
lean_object* v___x_844_ = _args[12];
lean_object* v_a_845_ = _args[13];
lean_object* v___y_846_ = _args[14];
lean_object* v___y_847_ = _args[15];
lean_object* v___y_848_ = _args[16];
lean_object* v___y_849_ = _args[17];
lean_object* v___y_850_ = _args[18];
lean_object* v___y_851_ = _args[19];
lean_object* v___y_852_ = _args[20];
lean_object* v___y_853_ = _args[21];
lean_object* v___y_854_ = _args[22];
_start:
{
uint8_t v___x_40106__boxed_855_; uint8_t v___x_40108__boxed_856_; lean_object* v_res_857_; 
v___x_40106__boxed_855_ = lean_unbox(v___x_836_);
v___x_40108__boxed_856_ = lean_unbox(v___x_838_);
v_res_857_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2(v_rev_832_, v___x_833_, v___x_834_, v_tk_835_, v___x_40106__boxed_855_, v___x_837_, v___x_40108__boxed_856_, v_rw_839_, v_ty_840_, v___x_841_, v_h_842_, v___x_843_, v___x_844_, v_a_845_, v___y_846_, v___y_847_, v___y_848_, v___y_849_, v___y_850_, v___y_851_, v___y_852_, v___y_853_);
lean_dec(v___y_853_);
lean_dec_ref(v___y_852_);
lean_dec(v___y_851_);
lean_dec_ref(v___y_850_);
lean_dec(v___y_849_);
lean_dec_ref(v___y_848_);
lean_dec(v___y_847_);
lean_dec_ref(v___y_846_);
lean_dec(v___x_844_);
lean_dec(v___x_843_);
lean_dec(v_rw_839_);
lean_dec(v_tk_835_);
return v_res_857_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__3___closed__1(void){
_start:
{
lean_object* v___x_859_; lean_object* v___x_860_; 
v___x_859_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__3___closed__0));
v___x_860_ = l_String_toRawSubstring_x27(v___x_859_);
return v___x_860_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__3(uint8_t v___x_863_, lean_object* v___f_864_, lean_object* v___x_865_, lean_object* v___x_866_, lean_object* v___y_867_, lean_object* v___y_868_, lean_object* v___y_869_, lean_object* v___y_870_, lean_object* v___y_871_, lean_object* v___y_872_, lean_object* v___y_873_, lean_object* v___y_874_){
_start:
{
if (v___x_863_ == 0)
{
lean_object* v_ref_876_; lean_object* v_quotContext_877_; lean_object* v_currMacroScope_878_; lean_object* v___x_879_; lean_object* v___x_880_; lean_object* v___x_881_; lean_object* v___x_882_; lean_object* v___x_883_; lean_object* v___x_884_; lean_object* v___x_885_; 
v_ref_876_ = lean_ctor_get(v___y_873_, 5);
v_quotContext_877_ = lean_ctor_get(v___y_873_, 10);
v_currMacroScope_878_ = lean_ctor_get(v___y_873_, 11);
v___x_879_ = l_Lean_SourceInfo_fromRef(v_ref_876_, v___x_863_);
v___x_880_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__3___closed__1, &lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__3___closed__1_once, _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__3___closed__1);
v___x_881_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__3___closed__2));
lean_inc(v_currMacroScope_878_);
lean_inc(v_quotContext_877_);
v___x_882_ = l_Lean_addMacroScope(v_quotContext_877_, v___x_881_, v_currMacroScope_878_);
v___x_883_ = lean_box(0);
v___x_884_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_884_, 0, v___x_879_);
lean_ctor_set(v___x_884_, 1, v___x_880_);
lean_ctor_set(v___x_884_, 2, v___x_882_);
lean_ctor_set(v___x_884_, 3, v___x_883_);
lean_inc(v___y_874_);
lean_inc_ref(v___y_873_);
lean_inc(v___y_872_);
lean_inc_ref(v___y_871_);
lean_inc(v___y_870_);
lean_inc_ref(v___y_869_);
lean_inc(v___y_868_);
lean_inc_ref(v___y_867_);
v___x_885_ = lean_apply_10(v___f_864_, v___x_884_, v___y_867_, v___y_868_, v___y_869_, v___y_870_, v___y_871_, v___y_872_, v___y_873_, v___y_874_, lean_box(0));
return v___x_885_;
}
else
{
lean_object* v___x_886_; lean_object* v___x_887_; uint8_t v___x_888_; 
v___x_886_ = l_Lean_Syntax_getArg(v___x_865_, v___x_866_);
v___x_887_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___closed__54));
lean_inc(v___x_886_);
v___x_888_ = l_Lean_Syntax_isOfKind(v___x_886_, v___x_887_);
if (v___x_888_ == 0)
{
lean_object* v_ref_889_; lean_object* v_quotContext_890_; lean_object* v_currMacroScope_891_; lean_object* v___x_892_; lean_object* v___x_893_; lean_object* v___x_894_; lean_object* v___x_895_; lean_object* v___x_896_; lean_object* v___x_897_; lean_object* v___x_898_; 
lean_dec(v___x_886_);
v_ref_889_ = lean_ctor_get(v___y_873_, 5);
v_quotContext_890_ = lean_ctor_get(v___y_873_, 10);
v_currMacroScope_891_ = lean_ctor_get(v___y_873_, 11);
v___x_892_ = l_Lean_SourceInfo_fromRef(v_ref_889_, v___x_888_);
v___x_893_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__3___closed__1, &lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__3___closed__1_once, _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__3___closed__1);
v___x_894_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__3___closed__2));
lean_inc(v_currMacroScope_891_);
lean_inc(v_quotContext_890_);
v___x_895_ = l_Lean_addMacroScope(v_quotContext_890_, v___x_894_, v_currMacroScope_891_);
v___x_896_ = lean_box(0);
v___x_897_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_897_, 0, v___x_892_);
lean_ctor_set(v___x_897_, 1, v___x_893_);
lean_ctor_set(v___x_897_, 2, v___x_895_);
lean_ctor_set(v___x_897_, 3, v___x_896_);
lean_inc(v___y_874_);
lean_inc_ref(v___y_873_);
lean_inc(v___y_872_);
lean_inc_ref(v___y_871_);
lean_inc(v___y_870_);
lean_inc_ref(v___y_869_);
lean_inc(v___y_868_);
lean_inc_ref(v___y_867_);
v___x_898_ = lean_apply_10(v___f_864_, v___x_897_, v___y_867_, v___y_868_, v___y_869_, v___y_870_, v___y_871_, v___y_872_, v___y_873_, v___y_874_, lean_box(0));
return v___x_898_;
}
else
{
lean_object* v___x_899_; 
lean_inc(v___y_874_);
lean_inc_ref(v___y_873_);
lean_inc(v___y_872_);
lean_inc_ref(v___y_871_);
lean_inc(v___y_870_);
lean_inc_ref(v___y_869_);
lean_inc(v___y_868_);
lean_inc_ref(v___y_867_);
v___x_899_ = lean_apply_10(v___f_864_, v___x_886_, v___y_867_, v___y_868_, v___y_869_, v___y_870_, v___y_871_, v___y_872_, v___y_873_, v___y_874_, lean_box(0));
return v___x_899_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__3___boxed(lean_object* v___x_900_, lean_object* v___f_901_, lean_object* v___x_902_, lean_object* v___x_903_, lean_object* v___y_904_, lean_object* v___y_905_, lean_object* v___y_906_, lean_object* v___y_907_, lean_object* v___y_908_, lean_object* v___y_909_, lean_object* v___y_910_, lean_object* v___y_911_, lean_object* v___y_912_){
_start:
{
uint8_t v___x_41077__boxed_913_; lean_object* v_res_914_; 
v___x_41077__boxed_913_ = lean_unbox(v___x_900_);
v_res_914_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__3(v___x_41077__boxed_913_, v___f_901_, v___x_902_, v___x_903_, v___y_904_, v___y_905_, v___y_906_, v___y_907_, v___y_908_, v___y_909_, v___y_910_, v___y_911_);
lean_dec(v___y_911_);
lean_dec_ref(v___y_910_);
lean_dec(v___y_909_);
lean_dec_ref(v___y_908_);
lean_dec(v___y_907_);
lean_dec_ref(v___y_906_);
lean_dec(v___y_905_);
lean_dec_ref(v___y_904_);
lean_dec(v___x_903_);
lean_dec(v___x_902_);
return v_res_914_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1(lean_object* v_x_920_, lean_object* v_a_921_, lean_object* v_a_922_, lean_object* v_a_923_, lean_object* v_a_924_, lean_object* v_a_925_, lean_object* v_a_926_, lean_object* v_a_927_, lean_object* v_a_928_){
_start:
{
lean_object* v___x_930_; lean_object* v___x_931_; lean_object* v___x_932_; uint8_t v___x_933_; 
v___x_930_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_setArgsRest___closed__1));
v___x_931_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_setArgsRest___closed__2));
v___x_932_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_setTactic___closed__1));
lean_inc(v_x_920_);
v___x_933_ = l_Lean_Syntax_isOfKind(v_x_920_, v___x_932_);
if (v___x_933_ == 0)
{
lean_object* v___x_934_; 
lean_dec(v_x_920_);
v___x_934_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1_spec__0___redArg();
return v___x_934_;
}
else
{
lean_object* v___x_935_; lean_object* v_tk_936_; uint8_t v___y_938_; lean_object* v___y_939_; lean_object* v___y_940_; lean_object* v___y_941_; lean_object* v___y_942_; lean_object* v___y_943_; lean_object* v___y_944_; lean_object* v_rev_945_; lean_object* v_h_946_; lean_object* v___y_947_; lean_object* v___y_948_; lean_object* v___y_949_; lean_object* v___y_950_; lean_object* v___y_951_; lean_object* v___y_952_; lean_object* v___y_953_; lean_object* v___y_954_; uint8_t v___y_962_; lean_object* v___y_963_; lean_object* v___y_964_; lean_object* v___y_965_; lean_object* v___y_966_; lean_object* v___y_967_; lean_object* v___y_968_; lean_object* v___y_969_; lean_object* v___y_970_; lean_object* v___y_971_; lean_object* v_rev_972_; lean_object* v___y_973_; lean_object* v___y_974_; lean_object* v___y_975_; lean_object* v___y_976_; lean_object* v___y_977_; lean_object* v___y_978_; lean_object* v___y_979_; lean_object* v___y_980_; lean_object* v___x_986_; uint8_t v___y_988_; lean_object* v___y_989_; lean_object* v___y_990_; lean_object* v___y_991_; lean_object* v___y_992_; lean_object* v___y_993_; lean_object* v___y_994_; lean_object* v___y_995_; lean_object* v_ty_996_; lean_object* v___y_997_; lean_object* v___y_998_; lean_object* v___y_999_; lean_object* v___y_1000_; lean_object* v___y_1001_; lean_object* v___y_1002_; lean_object* v___y_1003_; lean_object* v___y_1004_; lean_object* v_rw_1021_; lean_object* v___y_1022_; lean_object* v___y_1023_; lean_object* v___y_1024_; lean_object* v___y_1025_; lean_object* v___y_1026_; lean_object* v___y_1027_; lean_object* v___y_1028_; lean_object* v___y_1029_; lean_object* v___x_1047_; uint8_t v___x_1048_; 
v___x_935_ = lean_unsigned_to_nat(0u);
v_tk_936_ = l_Lean_Syntax_getArg(v_x_920_, v___x_935_);
v___x_986_ = lean_unsigned_to_nat(1u);
v___x_1047_ = l_Lean_Syntax_getArg(v_x_920_, v___x_986_);
v___x_1048_ = l_Lean_Syntax_isNone(v___x_1047_);
if (v___x_1048_ == 0)
{
uint8_t v___x_1049_; 
lean_inc(v___x_1047_);
v___x_1049_ = l_Lean_Syntax_matchesNull(v___x_1047_, v___x_986_);
if (v___x_1049_ == 0)
{
lean_object* v___x_1050_; 
lean_dec(v___x_1047_);
lean_dec(v_tk_936_);
lean_dec(v_x_920_);
v___x_1050_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1_spec__0___redArg();
return v___x_1050_;
}
else
{
lean_object* v_rw_1051_; lean_object* v___x_1052_; 
v_rw_1051_ = l_Lean_Syntax_getArg(v___x_1047_, v___x_935_);
lean_dec(v___x_1047_);
v___x_1052_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1052_, 0, v_rw_1051_);
v_rw_1021_ = v___x_1052_;
v___y_1022_ = v_a_921_;
v___y_1023_ = v_a_922_;
v___y_1024_ = v_a_923_;
v___y_1025_ = v_a_924_;
v___y_1026_ = v_a_925_;
v___y_1027_ = v_a_926_;
v___y_1028_ = v_a_927_;
v___y_1029_ = v_a_928_;
goto v___jp_1020_;
}
}
else
{
lean_object* v___x_1053_; 
lean_dec(v___x_1047_);
v___x_1053_ = lean_box(0);
v_rw_1021_ = v___x_1053_;
v___y_1022_ = v_a_921_;
v___y_1023_ = v_a_922_;
v___y_1024_ = v_a_923_;
v___y_1025_ = v_a_924_;
v___y_1026_ = v_a_925_;
v___y_1027_ = v_a_926_;
v___y_1028_ = v_a_927_;
v___y_1029_ = v_a_928_;
goto v___jp_1020_;
}
v___jp_937_:
{
lean_object* v___x_955_; lean_object* v___x_956_; lean_object* v___f_957_; lean_object* v___x_958_; lean_object* v___y_959_; lean_object* v___x_960_; 
v___x_955_ = lean_box(v___x_933_);
v___x_956_ = lean_box(v___y_938_);
lean_inc(v___y_941_);
lean_inc_ref(v___y_939_);
v___f_957_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__2___boxed), 23, 13);
lean_closure_set(v___f_957_, 0, v_rev_945_);
lean_closure_set(v___f_957_, 1, v___y_939_);
lean_closure_set(v___f_957_, 2, v___x_931_);
lean_closure_set(v___f_957_, 3, v_tk_936_);
lean_closure_set(v___f_957_, 4, v___x_955_);
lean_closure_set(v___f_957_, 5, v___x_930_);
lean_closure_set(v___f_957_, 6, v___x_956_);
lean_closure_set(v___f_957_, 7, v___y_940_);
lean_closure_set(v___f_957_, 8, v___y_943_);
lean_closure_set(v___f_957_, 9, v___y_942_);
lean_closure_set(v___f_957_, 10, v_h_946_);
lean_closure_set(v___f_957_, 11, v___y_941_);
lean_closure_set(v___f_957_, 12, v___x_935_);
v___x_958_ = lean_box(v___y_938_);
v___y_959_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___lam__3___boxed), 13, 4);
lean_closure_set(v___y_959_, 0, v___x_958_);
lean_closure_set(v___y_959_, 1, v___f_957_);
lean_closure_set(v___y_959_, 2, v___y_944_);
lean_closure_set(v___y_959_, 3, v___x_935_);
v___x_960_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___y_959_, v___y_947_, v___y_948_, v___y_949_, v___y_950_, v___y_951_, v___y_952_, v___y_953_, v___y_954_);
return v___x_960_;
}
v___jp_961_:
{
lean_object* v_h_981_; uint8_t v___x_982_; 
v_h_981_ = l_Lean_Syntax_getArg(v___y_969_, v___y_971_);
lean_dec(v___y_969_);
lean_inc(v_h_981_);
v___x_982_ = l_Lean_Syntax_isOfKind(v_h_981_, v___y_970_);
if (v___x_982_ == 0)
{
lean_object* v___x_983_; 
lean_dec(v_h_981_);
lean_dec(v_rev_972_);
lean_dec(v___y_968_);
lean_dec(v___y_967_);
lean_dec(v___y_966_);
lean_dec(v___y_964_);
lean_dec(v_tk_936_);
v___x_983_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1_spec__0___redArg();
return v___x_983_;
}
else
{
lean_object* v___x_984_; lean_object* v___x_985_; 
v___x_984_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_984_, 0, v_rev_972_);
v___x_985_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_985_, 0, v_h_981_);
v___y_938_ = v___y_962_;
v___y_939_ = v___y_963_;
v___y_940_ = v___y_964_;
v___y_941_ = v___y_965_;
v___y_942_ = v___y_967_;
v___y_943_ = v___y_966_;
v___y_944_ = v___y_968_;
v_rev_945_ = v___x_984_;
v_h_946_ = v___x_985_;
v___y_947_ = v___y_973_;
v___y_948_ = v___y_974_;
v___y_949_ = v___y_975_;
v___y_950_ = v___y_976_;
v___y_951_ = v___y_977_;
v___y_952_ = v___y_978_;
v___y_953_ = v___y_979_;
v___y_954_ = v___y_980_;
goto v___jp_937_;
}
}
v___jp_987_:
{
lean_object* v___x_1005_; lean_object* v___x_1006_; lean_object* v___x_1007_; lean_object* v___x_1008_; uint8_t v___x_1009_; 
v___x_1005_ = lean_unsigned_to_nat(3u);
v___x_1006_ = l_Lean_Syntax_getArg(v___y_993_, v___x_1005_);
v___x_1007_ = lean_unsigned_to_nat(4u);
v___x_1008_ = l_Lean_Syntax_getArg(v___y_993_, v___x_1007_);
lean_dec(v___y_993_);
v___x_1009_ = l_Lean_Syntax_isNone(v___x_1008_);
if (v___x_1009_ == 0)
{
uint8_t v___x_1010_; 
lean_inc(v___x_1008_);
v___x_1010_ = l_Lean_Syntax_matchesNull(v___x_1008_, v___x_1005_);
if (v___x_1010_ == 0)
{
lean_object* v___x_1011_; 
lean_dec(v___x_1008_);
lean_dec(v___x_1006_);
lean_dec(v_ty_996_);
lean_dec(v___y_992_);
lean_dec(v___y_990_);
lean_dec(v_tk_936_);
v___x_1011_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1_spec__0___redArg();
return v___x_1011_;
}
else
{
lean_object* v___x_1012_; uint8_t v___x_1013_; 
v___x_1012_ = l_Lean_Syntax_getArg(v___x_1008_, v___x_986_);
v___x_1013_ = l_Lean_Syntax_isNone(v___x_1012_);
if (v___x_1013_ == 0)
{
uint8_t v___x_1014_; 
lean_inc(v___x_1012_);
v___x_1014_ = l_Lean_Syntax_matchesNull(v___x_1012_, v___x_986_);
if (v___x_1014_ == 0)
{
lean_object* v___x_1015_; 
lean_dec(v___x_1012_);
lean_dec(v___x_1008_);
lean_dec(v___x_1006_);
lean_dec(v_ty_996_);
lean_dec(v___y_992_);
lean_dec(v___y_990_);
lean_dec(v_tk_936_);
v___x_1015_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1_spec__0___redArg();
return v___x_1015_;
}
else
{
lean_object* v_rev_1016_; lean_object* v___x_1017_; 
v_rev_1016_ = l_Lean_Syntax_getArg(v___x_1012_, v___x_935_);
lean_dec(v___x_1012_);
v___x_1017_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1017_, 0, v_rev_1016_);
v___y_962_ = v___y_988_;
v___y_963_ = v___y_989_;
v___y_964_ = v___y_990_;
v___y_965_ = v___y_991_;
v___y_966_ = v_ty_996_;
v___y_967_ = v___x_1006_;
v___y_968_ = v___y_992_;
v___y_969_ = v___x_1008_;
v___y_970_ = v___y_994_;
v___y_971_ = v___y_995_;
v_rev_972_ = v___x_1017_;
v___y_973_ = v___y_997_;
v___y_974_ = v___y_998_;
v___y_975_ = v___y_999_;
v___y_976_ = v___y_1000_;
v___y_977_ = v___y_1001_;
v___y_978_ = v___y_1002_;
v___y_979_ = v___y_1003_;
v___y_980_ = v___y_1004_;
goto v___jp_961_;
}
}
else
{
lean_object* v___x_1018_; 
lean_dec(v___x_1012_);
v___x_1018_ = lean_box(0);
v___y_962_ = v___y_988_;
v___y_963_ = v___y_989_;
v___y_964_ = v___y_990_;
v___y_965_ = v___y_991_;
v___y_966_ = v_ty_996_;
v___y_967_ = v___x_1006_;
v___y_968_ = v___y_992_;
v___y_969_ = v___x_1008_;
v___y_970_ = v___y_994_;
v___y_971_ = v___y_995_;
v_rev_972_ = v___x_1018_;
v___y_973_ = v___y_997_;
v___y_974_ = v___y_998_;
v___y_975_ = v___y_999_;
v___y_976_ = v___y_1000_;
v___y_977_ = v___y_1001_;
v___y_978_ = v___y_1002_;
v___y_979_ = v___y_1003_;
v___y_980_ = v___y_1004_;
goto v___jp_961_;
}
}
}
else
{
lean_object* v___x_1019_; 
lean_dec(v___x_1008_);
v___x_1019_ = lean_box(0);
v___y_938_ = v___y_988_;
v___y_939_ = v___y_989_;
v___y_940_ = v___y_990_;
v___y_941_ = v___y_991_;
v___y_942_ = v___x_1006_;
v___y_943_ = v_ty_996_;
v___y_944_ = v___y_992_;
v_rev_945_ = v___x_1019_;
v_h_946_ = v___x_1019_;
v___y_947_ = v___y_997_;
v___y_948_ = v___y_998_;
v___y_949_ = v___y_999_;
v___y_950_ = v___y_1000_;
v___y_951_ = v___y_1001_;
v___y_952_ = v___y_1002_;
v___y_953_ = v___y_1003_;
v___y_954_ = v___y_1004_;
goto v___jp_937_;
}
}
v___jp_1020_:
{
lean_object* v___x_1030_; lean_object* v___x_1031_; lean_object* v___x_1032_; uint8_t v___x_1033_; 
v___x_1030_ = lean_unsigned_to_nat(2u);
v___x_1031_ = l_Lean_Syntax_getArg(v_x_920_, v___x_1030_);
lean_dec(v_x_920_);
v___x_1032_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_setArgsRest___closed__3));
lean_inc(v___x_1031_);
v___x_1033_ = l_Lean_Syntax_isOfKind(v___x_1031_, v___x_1032_);
if (v___x_1033_ == 0)
{
lean_object* v___x_1034_; 
lean_dec(v___x_1031_);
lean_dec(v_rw_1021_);
lean_dec(v_tk_936_);
v___x_1034_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1_spec__0___redArg();
return v___x_1034_;
}
else
{
lean_object* v___x_1035_; lean_object* v___x_1036_; lean_object* v___x_1037_; uint8_t v___x_1038_; 
v___x_1035_ = l_Lean_Syntax_getArg(v___x_1031_, v___x_935_);
v___x_1036_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___closed__0));
v___x_1037_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___closed__2));
lean_inc(v___x_1035_);
v___x_1038_ = l_Lean_Syntax_isOfKind(v___x_1035_, v___x_1037_);
if (v___x_1038_ == 0)
{
lean_object* v___x_1039_; 
lean_dec(v___x_1035_);
lean_dec(v___x_1031_);
lean_dec(v_rw_1021_);
lean_dec(v_tk_936_);
v___x_1039_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1_spec__0___redArg();
return v___x_1039_;
}
else
{
lean_object* v___x_1040_; uint8_t v___x_1041_; 
v___x_1040_ = l_Lean_Syntax_getArg(v___x_1031_, v___x_986_);
v___x_1041_ = l_Lean_Syntax_isNone(v___x_1040_);
if (v___x_1041_ == 0)
{
uint8_t v___x_1042_; 
lean_inc(v___x_1040_);
v___x_1042_ = l_Lean_Syntax_matchesNull(v___x_1040_, v___x_1030_);
if (v___x_1042_ == 0)
{
lean_object* v___x_1043_; 
lean_dec(v___x_1040_);
lean_dec(v___x_1035_);
lean_dec(v___x_1031_);
lean_dec(v_rw_1021_);
lean_dec(v_tk_936_);
v___x_1043_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1_spec__0___redArg();
return v___x_1043_;
}
else
{
lean_object* v_ty_1044_; lean_object* v___x_1045_; 
v_ty_1044_ = l_Lean_Syntax_getArg(v___x_1040_, v___x_986_);
lean_dec(v___x_1040_);
v___x_1045_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1045_, 0, v_ty_1044_);
v___y_988_ = v___x_1038_;
v___y_989_ = v___x_1036_;
v___y_990_ = v_rw_1021_;
v___y_991_ = v___x_1037_;
v___y_992_ = v___x_1035_;
v___y_993_ = v___x_1031_;
v___y_994_ = v___x_1037_;
v___y_995_ = v___x_1030_;
v_ty_996_ = v___x_1045_;
v___y_997_ = v___y_1022_;
v___y_998_ = v___y_1023_;
v___y_999_ = v___y_1024_;
v___y_1000_ = v___y_1025_;
v___y_1001_ = v___y_1026_;
v___y_1002_ = v___y_1027_;
v___y_1003_ = v___y_1028_;
v___y_1004_ = v___y_1029_;
goto v___jp_987_;
}
}
else
{
lean_object* v___x_1046_; 
lean_dec(v___x_1040_);
v___x_1046_ = lean_box(0);
v___y_988_ = v___x_1038_;
v___y_989_ = v___x_1036_;
v___y_990_ = v_rw_1021_;
v___y_991_ = v___x_1037_;
v___y_992_ = v___x_1035_;
v___y_993_ = v___x_1031_;
v___y_994_ = v___x_1037_;
v___y_995_ = v___x_1030_;
v_ty_996_ = v___x_1046_;
v___y_997_ = v___y_1022_;
v___y_998_ = v___y_1023_;
v___y_999_ = v___y_1024_;
v___y_1000_ = v___y_1025_;
v___y_1001_ = v___y_1026_;
v___y_1002_ = v___y_1027_;
v___y_1003_ = v___y_1028_;
v___y_1004_ = v___y_1029_;
goto v___jp_987_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1___boxed(lean_object* v_x_1054_, lean_object* v_a_1055_, lean_object* v_a_1056_, lean_object* v_a_1057_, lean_object* v_a_1058_, lean_object* v_a_1059_, lean_object* v_a_1060_, lean_object* v_a_1061_, lean_object* v_a_1062_, lean_object* v_a_1063_){
_start:
{
lean_object* v_res_1064_; 
v_res_1064_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Set______elabRules__Mathlib__Tactic__setTactic__1(v_x_1054_, v_a_1055_, v_a_1056_, v_a_1057_, v_a_1058_, v_a_1059_, v_a_1060_, v_a_1061_, v_a_1062_);
lean_dec(v_a_1062_);
lean_dec_ref(v_a_1061_);
lean_dec(v_a_1060_);
lean_dec_ref(v_a_1059_);
lean_dec(v_a_1058_);
lean_dec_ref(v_a_1057_);
lean_dec(v_a_1056_);
lean_dec_ref(v_a_1055_);
return v_res_1064_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Set(uint8_t builtin) {
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
lean_object* runtime_initialize_Lean_Elab_Tactic_ElabTerm(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_Set(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Tactic_ElabTerm(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Mathlib_Tactic_setArgsRest = _init_lp_mathlib_Mathlib_Tactic_setArgsRest();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_setArgsRest);
lp_mathlib_Mathlib_Tactic_setTactic = _init_lp_mathlib_Mathlib_Tactic_setTactic();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_setTactic);
lp_mathlib_Mathlib_Tactic_tacticSet_x21__ = _init_lp_mathlib_Mathlib_Tactic_tacticSet_x21__();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_tacticSet_x21__);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
lean_object* initialize_Lean_Elab_Tactic_ElabTerm(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_Set(uint8_t builtin) {
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
res = initialize_Lean_Elab_Tactic_ElabTerm(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Set(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_Set(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_Set(builtin);
}
#ifdef __cplusplus
}
#endif
