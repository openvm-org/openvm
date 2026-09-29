// Lean compiler output
// Module: Batteries.Tactic.PermuteGoals
// Imports: public import Init public meta import Init public meta import Lean.Elab.Tactic.Basic
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
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_List_splitAt___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_throwNoGoalsToBeSolved___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_getGoals___redArg(lean_object*);
lean_object* l_List_lengthTR___redArg(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_setGoals___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_evalTactic(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_getUnsolvedGoals(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_List_appendTR___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getOptional_x3f(lean_object*);
lean_object* l_Lean_Syntax_toNat(lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Array_mkArray0(lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_addMessageContextFull___at___00Lean_throwError___at___00Batteries_Tactic_splitGoalsAndGetNth_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_addMessageContextFull___at___00Lean_throwError___at___00Batteries_Tactic_splitGoalsAndGetNth_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00Batteries_Tactic_splitGoalsAndGetNth_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00Batteries_Tactic_splitGoalsAndGetNth_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_Batteries_Tactic_splitGoalsAndGetNth___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = "goal index out of bounds"};
static const lean_object* lp_batteries_Batteries_Tactic_splitGoalsAndGetNth___closed__0 = (const lean_object*)&lp_batteries_Batteries_Tactic_splitGoalsAndGetNth___closed__0_value;
static lean_once_cell_t lp_batteries_Batteries_Tactic_splitGoalsAndGetNth___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic_splitGoalsAndGetNth___closed__1;
static const lean_string_object lp_batteries_Batteries_Tactic_splitGoalsAndGetNth___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "goals are 1-indexed"};
static const lean_object* lp_batteries_Batteries_Tactic_splitGoalsAndGetNth___closed__2 = (const lean_object*)&lp_batteries_Batteries_Tactic_splitGoalsAndGetNth___closed__2_value;
static lean_once_cell_t lp_batteries_Batteries_Tactic_splitGoalsAndGetNth___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic_splitGoalsAndGetNth___closed__3;
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_splitGoalsAndGetNth(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_splitGoalsAndGetNth___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00Batteries_Tactic_splitGoalsAndGetNth_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00Batteries_Tactic_splitGoalsAndGetNth_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_Batteries_Tactic_tacticPick__goal_x2d___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Batteries"};
static const lean_object* lp_batteries_Batteries_Tactic_tacticPick__goal_x2d___00__closed__0 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticPick__goal_x2d___00__closed__0_value;
static const lean_string_object lp_batteries_Batteries_Tactic_tacticPick__goal_x2d___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_batteries_Batteries_Tactic_tacticPick__goal_x2d___00__closed__1 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticPick__goal_x2d___00__closed__1_value;
static const lean_string_object lp_batteries_Batteries_Tactic_tacticPick__goal_x2d___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "tacticPick_goal-_"};
static const lean_object* lp_batteries_Batteries_Tactic_tacticPick__goal_x2d___00__closed__2 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticPick__goal_x2d___00__closed__2_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_tacticPick__goal_x2d___00__closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_tacticPick__goal_x2d___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_tacticPick__goal_x2d___00__closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_tacticPick__goal_x2d___00__closed__3_value_aux_0),((lean_object*)&lp_batteries_Batteries_Tactic_tacticPick__goal_x2d___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(219, 16, 21, 14, 34, 244, 116, 98)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_tacticPick__goal_x2d___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_tacticPick__goal_x2d___00__closed__3_value_aux_1),((lean_object*)&lp_batteries_Batteries_Tactic_tacticPick__goal_x2d___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(121, 117, 22, 172, 18, 18, 128, 239)}};
static const lean_object* lp_batteries_Batteries_Tactic_tacticPick__goal_x2d___00__closed__3 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticPick__goal_x2d___00__closed__3_value;
static const lean_string_object lp_batteries_Batteries_Tactic_tacticPick__goal_x2d___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_batteries_Batteries_Tactic_tacticPick__goal_x2d___00__closed__4 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticPick__goal_x2d___00__closed__4_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_tacticPick__goal_x2d___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_tacticPick__goal_x2d___00__closed__4_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_batteries_Batteries_Tactic_tacticPick__goal_x2d___00__closed__5 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticPick__goal_x2d___00__closed__5_value;
static const lean_string_object lp_batteries_Batteries_Tactic_tacticPick__goal_x2d___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "pick_goal "};
static const lean_object* lp_batteries_Batteries_Tactic_tacticPick__goal_x2d___00__closed__6 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticPick__goal_x2d___00__closed__6_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_tacticPick__goal_x2d___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_tacticPick__goal_x2d___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_batteries_Batteries_Tactic_tacticPick__goal_x2d___00__closed__7 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticPick__goal_x2d___00__closed__7_value;
static const lean_string_object lp_batteries_Batteries_Tactic_tacticPick__goal_x2d___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "optional"};
static const lean_object* lp_batteries_Batteries_Tactic_tacticPick__goal_x2d___00__closed__8 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticPick__goal_x2d___00__closed__8_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_tacticPick__goal_x2d___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_tacticPick__goal_x2d___00__closed__8_value),LEAN_SCALAR_PTR_LITERAL(233, 141, 154, 50, 143, 135, 42, 252)}};
static const lean_object* lp_batteries_Batteries_Tactic_tacticPick__goal_x2d___00__closed__9 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticPick__goal_x2d___00__closed__9_value;
static const lean_string_object lp_batteries_Batteries_Tactic_tacticPick__goal_x2d___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "-"};
static const lean_object* lp_batteries_Batteries_Tactic_tacticPick__goal_x2d___00__closed__10 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticPick__goal_x2d___00__closed__10_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_tacticPick__goal_x2d___00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_tacticPick__goal_x2d___00__closed__10_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_tacticPick__goal_x2d___00__closed__11 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticPick__goal_x2d___00__closed__11_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_tacticPick__goal_x2d___00__closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_tacticPick__goal_x2d___00__closed__9_value),((lean_object*)&lp_batteries_Batteries_Tactic_tacticPick__goal_x2d___00__closed__11_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_tacticPick__goal_x2d___00__closed__12 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticPick__goal_x2d___00__closed__12_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_tacticPick__goal_x2d___00__closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_tacticPick__goal_x2d___00__closed__5_value),((lean_object*)&lp_batteries_Batteries_Tactic_tacticPick__goal_x2d___00__closed__7_value),((lean_object*)&lp_batteries_Batteries_Tactic_tacticPick__goal_x2d___00__closed__12_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_tacticPick__goal_x2d___00__closed__13 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticPick__goal_x2d___00__closed__13_value;
static const lean_string_object lp_batteries_Batteries_Tactic_tacticPick__goal_x2d___00__closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "num"};
static const lean_object* lp_batteries_Batteries_Tactic_tacticPick__goal_x2d___00__closed__14 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticPick__goal_x2d___00__closed__14_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_tacticPick__goal_x2d___00__closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_tacticPick__goal_x2d___00__closed__14_value),LEAN_SCALAR_PTR_LITERAL(227, 68, 22, 222, 47, 51, 204, 84)}};
static const lean_object* lp_batteries_Batteries_Tactic_tacticPick__goal_x2d___00__closed__15 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticPick__goal_x2d___00__closed__15_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_tacticPick__goal_x2d___00__closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_tacticPick__goal_x2d___00__closed__15_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_tacticPick__goal_x2d___00__closed__16 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticPick__goal_x2d___00__closed__16_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_tacticPick__goal_x2d___00__closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_tacticPick__goal_x2d___00__closed__5_value),((lean_object*)&lp_batteries_Batteries_Tactic_tacticPick__goal_x2d___00__closed__13_value),((lean_object*)&lp_batteries_Batteries_Tactic_tacticPick__goal_x2d___00__closed__16_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_tacticPick__goal_x2d___00__closed__17 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticPick__goal_x2d___00__closed__17_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_tacticPick__goal_x2d___00__closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_tacticPick__goal_x2d___00__closed__3_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_tacticPick__goal_x2d___00__closed__17_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_tacticPick__goal_x2d___00__closed__18 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticPick__goal_x2d___00__closed__18_value;
LEAN_EXPORT const lean_object* lp_batteries_Batteries_Tactic_tacticPick__goal_x2d__ = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticPick__goal_x2d___00__closed__18_value;
static lean_once_cell_t lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__PermuteGoals______elabRules__Batteries__Tactic__tacticPick__goal_x2d____1_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__PermuteGoals______elabRules__Batteries__Tactic__tacticPick__goal_x2d____1_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__PermuteGoals______elabRules__Batteries__Tactic__tacticPick__goal_x2d____1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__PermuteGoals______elabRules__Batteries__Tactic__tacticPick__goal_x2d____1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__PermuteGoals______elabRules__Batteries__Tactic__tacticPick__goal_x2d____1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__PermuteGoals______elabRules__Batteries__Tactic__tacticPick__goal_x2d____1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__PermuteGoals______elabRules__Batteries__Tactic__tacticPick__goal_x2d____1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__PermuteGoals______elabRules__Batteries__Tactic__tacticPick__goal_x2d____1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_Batteries_Tactic_tacticSwap___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "tacticSwap"};
static const lean_object* lp_batteries_Batteries_Tactic_tacticSwap___closed__0 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticSwap___closed__0_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_tacticSwap___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_tacticPick__goal_x2d___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_tacticSwap___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_tacticSwap___closed__1_value_aux_0),((lean_object*)&lp_batteries_Batteries_Tactic_tacticPick__goal_x2d___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(219, 16, 21, 14, 34, 244, 116, 98)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_tacticSwap___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_tacticSwap___closed__1_value_aux_1),((lean_object*)&lp_batteries_Batteries_Tactic_tacticSwap___closed__0_value),LEAN_SCALAR_PTR_LITERAL(131, 195, 160, 50, 159, 119, 204, 253)}};
static const lean_object* lp_batteries_Batteries_Tactic_tacticSwap___closed__1 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticSwap___closed__1_value;
static const lean_string_object lp_batteries_Batteries_Tactic_tacticSwap___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "swap"};
static const lean_object* lp_batteries_Batteries_Tactic_tacticSwap___closed__2 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticSwap___closed__2_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_tacticSwap___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_tacticSwap___closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_batteries_Batteries_Tactic_tacticSwap___closed__3 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticSwap___closed__3_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_tacticSwap___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_tacticSwap___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_tacticSwap___closed__3_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_tacticSwap___closed__4 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticSwap___closed__4_value;
LEAN_EXPORT const lean_object* lp_batteries_Batteries_Tactic_tacticSwap = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticSwap___closed__4_value;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__PermuteGoals______macroRules__Batteries__Tactic__tacticSwap__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "pick_goal"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__PermuteGoals______macroRules__Batteries__Tactic__tacticSwap__1___closed__0 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__PermuteGoals______macroRules__Batteries__Tactic__tacticSwap__1___closed__0_value;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__PermuteGoals______macroRules__Batteries__Tactic__tacticSwap__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__PermuteGoals______macroRules__Batteries__Tactic__tacticSwap__1___closed__1 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__PermuteGoals______macroRules__Batteries__Tactic__tacticSwap__1___closed__1_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__PermuteGoals______macroRules__Batteries__Tactic__tacticSwap__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__PermuteGoals______macroRules__Batteries__Tactic__tacticSwap__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__PermuteGoals______macroRules__Batteries__Tactic__tacticSwap__1___closed__2 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__PermuteGoals______macroRules__Batteries__Tactic__tacticSwap__1___closed__2_value;
static lean_once_cell_t lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__PermuteGoals______macroRules__Batteries__Tactic__tacticSwap__1___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__PermuteGoals______macroRules__Batteries__Tactic__tacticSwap__1___closed__3;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__PermuteGoals______macroRules__Batteries__Tactic__tacticSwap__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "2"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__PermuteGoals______macroRules__Batteries__Tactic__tacticSwap__1___closed__4 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__PermuteGoals______macroRules__Batteries__Tactic__tacticSwap__1___closed__4_value;
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__PermuteGoals______macroRules__Batteries__Tactic__tacticSwap__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__PermuteGoals______macroRules__Batteries__Tactic__tacticSwap__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_Batteries_Tactic_tacticOn__goal_x2d___x3d_x3e___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tacticOn_goal-_=>_"};
static const lean_object* lp_batteries_Batteries_Tactic_tacticOn__goal_x2d___x3d_x3e___00__closed__0 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticOn__goal_x2d___x3d_x3e___00__closed__0_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_tacticOn__goal_x2d___x3d_x3e___00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_tacticPick__goal_x2d___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_tacticOn__goal_x2d___x3d_x3e___00__closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_tacticOn__goal_x2d___x3d_x3e___00__closed__1_value_aux_0),((lean_object*)&lp_batteries_Batteries_Tactic_tacticPick__goal_x2d___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(219, 16, 21, 14, 34, 244, 116, 98)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_tacticOn__goal_x2d___x3d_x3e___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_tacticOn__goal_x2d___x3d_x3e___00__closed__1_value_aux_1),((lean_object*)&lp_batteries_Batteries_Tactic_tacticOn__goal_x2d___x3d_x3e___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(243, 56, 227, 189, 147, 207, 104, 76)}};
static const lean_object* lp_batteries_Batteries_Tactic_tacticOn__goal_x2d___x3d_x3e___00__closed__1 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticOn__goal_x2d___x3d_x3e___00__closed__1_value;
static const lean_string_object lp_batteries_Batteries_Tactic_tacticOn__goal_x2d___x3d_x3e___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "on_goal "};
static const lean_object* lp_batteries_Batteries_Tactic_tacticOn__goal_x2d___x3d_x3e___00__closed__2 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticOn__goal_x2d___x3d_x3e___00__closed__2_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_tacticOn__goal_x2d___x3d_x3e___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_tacticOn__goal_x2d___x3d_x3e___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_batteries_Batteries_Tactic_tacticOn__goal_x2d___x3d_x3e___00__closed__3 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticOn__goal_x2d___x3d_x3e___00__closed__3_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_tacticOn__goal_x2d___x3d_x3e___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_tacticPick__goal_x2d___00__closed__5_value),((lean_object*)&lp_batteries_Batteries_Tactic_tacticOn__goal_x2d___x3d_x3e___00__closed__3_value),((lean_object*)&lp_batteries_Batteries_Tactic_tacticPick__goal_x2d___00__closed__12_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_tacticOn__goal_x2d___x3d_x3e___00__closed__4 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticOn__goal_x2d___x3d_x3e___00__closed__4_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_tacticOn__goal_x2d___x3d_x3e___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_tacticPick__goal_x2d___00__closed__5_value),((lean_object*)&lp_batteries_Batteries_Tactic_tacticOn__goal_x2d___x3d_x3e___00__closed__4_value),((lean_object*)&lp_batteries_Batteries_Tactic_tacticPick__goal_x2d___00__closed__16_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_tacticOn__goal_x2d___x3d_x3e___00__closed__5 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticOn__goal_x2d___x3d_x3e___00__closed__5_value;
static const lean_string_object lp_batteries_Batteries_Tactic_tacticOn__goal_x2d___x3d_x3e___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = " => "};
static const lean_object* lp_batteries_Batteries_Tactic_tacticOn__goal_x2d___x3d_x3e___00__closed__6 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticOn__goal_x2d___x3d_x3e___00__closed__6_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_tacticOn__goal_x2d___x3d_x3e___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_tacticOn__goal_x2d___x3d_x3e___00__closed__6_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_tacticOn__goal_x2d___x3d_x3e___00__closed__7 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticOn__goal_x2d___x3d_x3e___00__closed__7_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_tacticOn__goal_x2d___x3d_x3e___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_tacticPick__goal_x2d___00__closed__5_value),((lean_object*)&lp_batteries_Batteries_Tactic_tacticOn__goal_x2d___x3d_x3e___00__closed__5_value),((lean_object*)&lp_batteries_Batteries_Tactic_tacticOn__goal_x2d___x3d_x3e___00__closed__7_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_tacticOn__goal_x2d___x3d_x3e___00__closed__8 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticOn__goal_x2d___x3d_x3e___00__closed__8_value;
static const lean_string_object lp_batteries_Batteries_Tactic_tacticOn__goal_x2d___x3d_x3e___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tacticSeq"};
static const lean_object* lp_batteries_Batteries_Tactic_tacticOn__goal_x2d___x3d_x3e___00__closed__9 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticOn__goal_x2d___x3d_x3e___00__closed__9_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_tacticOn__goal_x2d___x3d_x3e___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_tacticOn__goal_x2d___x3d_x3e___00__closed__9_value),LEAN_SCALAR_PTR_LITERAL(13, 106, 54, 236, 164, 218, 24, 154)}};
static const lean_object* lp_batteries_Batteries_Tactic_tacticOn__goal_x2d___x3d_x3e___00__closed__10 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticOn__goal_x2d___x3d_x3e___00__closed__10_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_tacticOn__goal_x2d___x3d_x3e___00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_tacticOn__goal_x2d___x3d_x3e___00__closed__10_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_tacticOn__goal_x2d___x3d_x3e___00__closed__11 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticOn__goal_x2d___x3d_x3e___00__closed__11_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_tacticOn__goal_x2d___x3d_x3e___00__closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_tacticPick__goal_x2d___00__closed__5_value),((lean_object*)&lp_batteries_Batteries_Tactic_tacticOn__goal_x2d___x3d_x3e___00__closed__8_value),((lean_object*)&lp_batteries_Batteries_Tactic_tacticOn__goal_x2d___x3d_x3e___00__closed__11_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_tacticOn__goal_x2d___x3d_x3e___00__closed__12 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticOn__goal_x2d___x3d_x3e___00__closed__12_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_tacticOn__goal_x2d___x3d_x3e___00__closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_tacticOn__goal_x2d___x3d_x3e___00__closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_tacticOn__goal_x2d___x3d_x3e___00__closed__12_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_tacticOn__goal_x2d___x3d_x3e___00__closed__13 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticOn__goal_x2d___x3d_x3e___00__closed__13_value;
LEAN_EXPORT const lean_object* lp_batteries_Batteries_Tactic_tacticOn__goal_x2d___x3d_x3e__ = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticOn__goal_x2d___x3d_x3e___00__closed__13_value;
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__PermuteGoals______elabRules__Batteries__Tactic__tacticOn__goal_x2d___x3d_x3e____1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__PermuteGoals______elabRules__Batteries__Tactic__tacticOn__goal_x2d___x3d_x3e____1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_addMessageContextFull___at___00Lean_throwError___at___00Batteries_Tactic_splitGoalsAndGetNth_spec__0_spec__0(lean_object* v_msgData_1_, lean_object* v___y_2_, lean_object* v___y_3_, lean_object* v___y_4_, lean_object* v___y_5_){
_start:
{
lean_object* v___x_7_; lean_object* v_env_8_; lean_object* v___x_9_; lean_object* v_mctx_10_; lean_object* v_lctx_11_; lean_object* v_options_12_; lean_object* v___x_13_; lean_object* v___x_14_; lean_object* v___x_15_; 
v___x_7_ = lean_st_ref_get(v___y_5_);
v_env_8_ = lean_ctor_get(v___x_7_, 0);
lean_inc_ref(v_env_8_);
lean_dec(v___x_7_);
v___x_9_ = lean_st_ref_get(v___y_3_);
v_mctx_10_ = lean_ctor_get(v___x_9_, 0);
lean_inc_ref(v_mctx_10_);
lean_dec(v___x_9_);
v_lctx_11_ = lean_ctor_get(v___y_2_, 2);
v_options_12_ = lean_ctor_get(v___y_4_, 2);
lean_inc_ref(v_options_12_);
lean_inc_ref(v_lctx_11_);
v___x_13_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_13_, 0, v_env_8_);
lean_ctor_set(v___x_13_, 1, v_mctx_10_);
lean_ctor_set(v___x_13_, 2, v_lctx_11_);
lean_ctor_set(v___x_13_, 3, v_options_12_);
v___x_14_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_14_, 0, v___x_13_);
lean_ctor_set(v___x_14_, 1, v_msgData_1_);
v___x_15_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_15_, 0, v___x_14_);
return v___x_15_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_addMessageContextFull___at___00Lean_throwError___at___00Batteries_Tactic_splitGoalsAndGetNth_spec__0_spec__0___boxed(lean_object* v_msgData_16_, lean_object* v___y_17_, lean_object* v___y_18_, lean_object* v___y_19_, lean_object* v___y_20_, lean_object* v___y_21_){
_start:
{
lean_object* v_res_22_; 
v_res_22_ = lp_batteries_Lean_addMessageContextFull___at___00Lean_throwError___at___00Batteries_Tactic_splitGoalsAndGetNth_spec__0_spec__0(v_msgData_16_, v___y_17_, v___y_18_, v___y_19_, v___y_20_);
lean_dec(v___y_20_);
lean_dec_ref(v___y_19_);
lean_dec(v___y_18_);
lean_dec_ref(v___y_17_);
return v_res_22_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00Batteries_Tactic_splitGoalsAndGetNth_spec__0___redArg(lean_object* v_msg_23_, lean_object* v___y_24_, lean_object* v___y_25_, lean_object* v___y_26_, lean_object* v___y_27_){
_start:
{
lean_object* v_ref_29_; lean_object* v___x_30_; lean_object* v_a_31_; lean_object* v___x_33_; uint8_t v_isShared_34_; uint8_t v_isSharedCheck_39_; 
v_ref_29_ = lean_ctor_get(v___y_26_, 5);
v___x_30_ = lp_batteries_Lean_addMessageContextFull___at___00Lean_throwError___at___00Batteries_Tactic_splitGoalsAndGetNth_spec__0_spec__0(v_msg_23_, v___y_24_, v___y_25_, v___y_26_, v___y_27_);
v_a_31_ = lean_ctor_get(v___x_30_, 0);
v_isSharedCheck_39_ = !lean_is_exclusive(v___x_30_);
if (v_isSharedCheck_39_ == 0)
{
v___x_33_ = v___x_30_;
v_isShared_34_ = v_isSharedCheck_39_;
goto v_resetjp_32_;
}
else
{
lean_inc(v_a_31_);
lean_dec(v___x_30_);
v___x_33_ = lean_box(0);
v_isShared_34_ = v_isSharedCheck_39_;
goto v_resetjp_32_;
}
v_resetjp_32_:
{
lean_object* v___x_35_; lean_object* v___x_37_; 
lean_inc(v_ref_29_);
v___x_35_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_35_, 0, v_ref_29_);
lean_ctor_set(v___x_35_, 1, v_a_31_);
if (v_isShared_34_ == 0)
{
lean_ctor_set_tag(v___x_33_, 1);
lean_ctor_set(v___x_33_, 0, v___x_35_);
v___x_37_ = v___x_33_;
goto v_reusejp_36_;
}
else
{
lean_object* v_reuseFailAlloc_38_; 
v_reuseFailAlloc_38_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_38_, 0, v___x_35_);
v___x_37_ = v_reuseFailAlloc_38_;
goto v_reusejp_36_;
}
v_reusejp_36_:
{
return v___x_37_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00Batteries_Tactic_splitGoalsAndGetNth_spec__0___redArg___boxed(lean_object* v_msg_40_, lean_object* v___y_41_, lean_object* v___y_42_, lean_object* v___y_43_, lean_object* v___y_44_, lean_object* v___y_45_){
_start:
{
lean_object* v_res_46_; 
v_res_46_ = lp_batteries_Lean_throwError___at___00Batteries_Tactic_splitGoalsAndGetNth_spec__0___redArg(v_msg_40_, v___y_41_, v___y_42_, v___y_43_, v___y_44_);
lean_dec(v___y_44_);
lean_dec_ref(v___y_43_);
lean_dec(v___y_42_);
lean_dec_ref(v___y_41_);
return v_res_46_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic_splitGoalsAndGetNth___closed__1(void){
_start:
{
lean_object* v___x_48_; lean_object* v___x_49_; 
v___x_48_ = ((lean_object*)(lp_batteries_Batteries_Tactic_splitGoalsAndGetNth___closed__0));
v___x_49_ = l_Lean_stringToMessageData(v___x_48_);
return v___x_49_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic_splitGoalsAndGetNth___closed__3(void){
_start:
{
lean_object* v___x_51_; lean_object* v___x_52_; 
v___x_51_ = ((lean_object*)(lp_batteries_Batteries_Tactic_splitGoalsAndGetNth___closed__2));
v___x_52_ = l_Lean_stringToMessageData(v___x_51_);
return v___x_52_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_splitGoalsAndGetNth(lean_object* v_nth_53_, uint8_t v_reverse_54_, lean_object* v_a_55_, lean_object* v_a_56_, lean_object* v_a_57_, lean_object* v_a_58_, lean_object* v_a_59_, lean_object* v_a_60_, lean_object* v_a_61_, lean_object* v_a_62_){
_start:
{
lean_object* v___y_65_; lean_object* v___y_66_; lean_object* v___y_67_; lean_object* v___y_68_; lean_object* v___y_69_; lean_object* v___y_70_; lean_object* v___y_94_; lean_object* v___y_95_; lean_object* v___y_96_; lean_object* v___y_97_; lean_object* v___y_98_; lean_object* v___y_99_; lean_object* v___y_104_; lean_object* v___y_105_; lean_object* v___y_106_; lean_object* v___y_107_; lean_object* v___y_108_; lean_object* v___y_109_; lean_object* v___y_110_; lean_object* v___y_111_; lean_object* v___x_134_; uint8_t v___x_135_; 
v___x_134_ = lean_unsigned_to_nat(0u);
v___x_135_ = lean_nat_dec_eq(v_nth_53_, v___x_134_);
if (v___x_135_ == 0)
{
v___y_104_ = v_a_55_;
v___y_105_ = v_a_56_;
v___y_106_ = v_a_57_;
v___y_107_ = v_a_58_;
v___y_108_ = v_a_59_;
v___y_109_ = v_a_60_;
v___y_110_ = v_a_61_;
v___y_111_ = v_a_62_;
goto v___jp_103_;
}
else
{
lean_object* v___x_136_; lean_object* v___x_137_; lean_object* v_a_138_; lean_object* v___x_140_; uint8_t v_isShared_141_; uint8_t v_isSharedCheck_145_; 
v___x_136_ = lean_obj_once(&lp_batteries_Batteries_Tactic_splitGoalsAndGetNth___closed__3, &lp_batteries_Batteries_Tactic_splitGoalsAndGetNth___closed__3_once, _init_lp_batteries_Batteries_Tactic_splitGoalsAndGetNth___closed__3);
v___x_137_ = lp_batteries_Lean_throwError___at___00Batteries_Tactic_splitGoalsAndGetNth_spec__0___redArg(v___x_136_, v_a_59_, v_a_60_, v_a_61_, v_a_62_);
v_a_138_ = lean_ctor_get(v___x_137_, 0);
v_isSharedCheck_145_ = !lean_is_exclusive(v___x_137_);
if (v_isSharedCheck_145_ == 0)
{
v___x_140_ = v___x_137_;
v_isShared_141_ = v_isSharedCheck_145_;
goto v_resetjp_139_;
}
else
{
lean_inc(v_a_138_);
lean_dec(v___x_137_);
v___x_140_ = lean_box(0);
v_isShared_141_ = v_isSharedCheck_145_;
goto v_resetjp_139_;
}
v_resetjp_139_:
{
lean_object* v___x_143_; 
if (v_isShared_141_ == 0)
{
v___x_143_ = v___x_140_;
goto v_reusejp_142_;
}
else
{
lean_object* v_reuseFailAlloc_144_; 
v_reuseFailAlloc_144_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_144_, 0, v_a_138_);
v___x_143_ = v_reuseFailAlloc_144_;
goto v_reusejp_142_;
}
v_reusejp_142_:
{
return v___x_143_;
}
}
}
v___jp_64_:
{
lean_object* v___x_71_; lean_object* v_snd_72_; 
v___x_71_ = l_List_splitAt___redArg(v___y_70_, v___y_68_);
v_snd_72_ = lean_ctor_get(v___x_71_, 1);
lean_inc(v_snd_72_);
if (lean_obj_tag(v_snd_72_) == 1)
{
lean_object* v_fst_73_; lean_object* v___x_75_; uint8_t v_isShared_76_; uint8_t v_isSharedCheck_90_; 
v_fst_73_ = lean_ctor_get(v___x_71_, 0);
v_isSharedCheck_90_ = !lean_is_exclusive(v___x_71_);
if (v_isSharedCheck_90_ == 0)
{
lean_object* v_unused_91_; 
v_unused_91_ = lean_ctor_get(v___x_71_, 1);
lean_dec(v_unused_91_);
v___x_75_ = v___x_71_;
v_isShared_76_ = v_isSharedCheck_90_;
goto v_resetjp_74_;
}
else
{
lean_inc(v_fst_73_);
lean_dec(v___x_71_);
v___x_75_ = lean_box(0);
v_isShared_76_ = v_isSharedCheck_90_;
goto v_resetjp_74_;
}
v_resetjp_74_:
{
lean_object* v_head_77_; lean_object* v_tail_78_; lean_object* v___x_80_; uint8_t v_isShared_81_; uint8_t v_isSharedCheck_89_; 
v_head_77_ = lean_ctor_get(v_snd_72_, 0);
v_tail_78_ = lean_ctor_get(v_snd_72_, 1);
v_isSharedCheck_89_ = !lean_is_exclusive(v_snd_72_);
if (v_isSharedCheck_89_ == 0)
{
v___x_80_ = v_snd_72_;
v_isShared_81_ = v_isSharedCheck_89_;
goto v_resetjp_79_;
}
else
{
lean_inc(v_tail_78_);
lean_inc(v_head_77_);
lean_dec(v_snd_72_);
v___x_80_ = lean_box(0);
v_isShared_81_ = v_isSharedCheck_89_;
goto v_resetjp_79_;
}
v_resetjp_79_:
{
lean_object* v___x_83_; 
if (v_isShared_76_ == 0)
{
lean_ctor_set(v___x_75_, 1, v_tail_78_);
v___x_83_ = v___x_75_;
goto v_reusejp_82_;
}
else
{
lean_object* v_reuseFailAlloc_88_; 
v_reuseFailAlloc_88_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_88_, 0, v_fst_73_);
lean_ctor_set(v_reuseFailAlloc_88_, 1, v_tail_78_);
v___x_83_ = v_reuseFailAlloc_88_;
goto v_reusejp_82_;
}
v_reusejp_82_:
{
lean_object* v___x_85_; 
if (v_isShared_81_ == 0)
{
lean_ctor_set_tag(v___x_80_, 0);
lean_ctor_set(v___x_80_, 1, v___x_83_);
v___x_85_ = v___x_80_;
goto v_reusejp_84_;
}
else
{
lean_object* v_reuseFailAlloc_87_; 
v_reuseFailAlloc_87_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_87_, 0, v_head_77_);
lean_ctor_set(v_reuseFailAlloc_87_, 1, v___x_83_);
v___x_85_ = v_reuseFailAlloc_87_;
goto v_reusejp_84_;
}
v_reusejp_84_:
{
lean_object* v___x_86_; 
v___x_86_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_86_, 0, v___x_85_);
return v___x_86_;
}
}
}
}
}
else
{
lean_object* v___x_92_; 
lean_dec(v_snd_72_);
lean_dec_ref(v___x_71_);
v___x_92_ = l_Lean_Elab_Tactic_throwNoGoalsToBeSolved___redArg(v___y_69_, v___y_67_, v___y_65_, v___y_66_);
return v___x_92_;
}
}
v___jp_93_:
{
if (v_reverse_54_ == 0)
{
lean_object* v___x_100_; lean_object* v___x_101_; 
lean_dec(v___y_94_);
v___x_100_ = lean_unsigned_to_nat(1u);
v___x_101_ = lean_nat_sub(v_nth_53_, v___x_100_);
v___y_65_ = v___y_98_;
v___y_66_ = v___y_99_;
v___y_67_ = v___y_97_;
v___y_68_ = v___y_95_;
v___y_69_ = v___y_96_;
v___y_70_ = v___x_101_;
goto v___jp_64_;
}
else
{
lean_object* v___x_102_; 
v___x_102_ = lean_nat_sub(v___y_94_, v_nth_53_);
lean_dec(v___y_94_);
v___y_65_ = v___y_98_;
v___y_66_ = v___y_99_;
v___y_67_ = v___y_97_;
v___y_68_ = v___y_95_;
v___y_69_ = v___y_96_;
v___y_70_ = v___x_102_;
goto v___jp_64_;
}
}
v___jp_103_:
{
lean_object* v___x_112_; 
v___x_112_ = l_Lean_Elab_Tactic_getGoals___redArg(v___y_105_);
if (lean_obj_tag(v___x_112_) == 0)
{
lean_object* v_a_113_; lean_object* v___x_114_; uint8_t v___x_115_; 
v_a_113_ = lean_ctor_get(v___x_112_, 0);
lean_inc(v_a_113_);
lean_dec_ref_known(v___x_112_, 1);
v___x_114_ = l_List_lengthTR___redArg(v_a_113_);
v___x_115_ = lean_nat_dec_lt(v___x_114_, v_nth_53_);
if (v___x_115_ == 0)
{
v___y_94_ = v___x_114_;
v___y_95_ = v_a_113_;
v___y_96_ = v___y_108_;
v___y_97_ = v___y_109_;
v___y_98_ = v___y_110_;
v___y_99_ = v___y_111_;
goto v___jp_93_;
}
else
{
lean_object* v___x_116_; lean_object* v___x_117_; lean_object* v_a_118_; lean_object* v___x_120_; uint8_t v_isShared_121_; uint8_t v_isSharedCheck_125_; 
lean_dec(v___x_114_);
lean_dec(v_a_113_);
v___x_116_ = lean_obj_once(&lp_batteries_Batteries_Tactic_splitGoalsAndGetNth___closed__1, &lp_batteries_Batteries_Tactic_splitGoalsAndGetNth___closed__1_once, _init_lp_batteries_Batteries_Tactic_splitGoalsAndGetNth___closed__1);
v___x_117_ = lp_batteries_Lean_throwError___at___00Batteries_Tactic_splitGoalsAndGetNth_spec__0___redArg(v___x_116_, v___y_108_, v___y_109_, v___y_110_, v___y_111_);
v_a_118_ = lean_ctor_get(v___x_117_, 0);
v_isSharedCheck_125_ = !lean_is_exclusive(v___x_117_);
if (v_isSharedCheck_125_ == 0)
{
v___x_120_ = v___x_117_;
v_isShared_121_ = v_isSharedCheck_125_;
goto v_resetjp_119_;
}
else
{
lean_inc(v_a_118_);
lean_dec(v___x_117_);
v___x_120_ = lean_box(0);
v_isShared_121_ = v_isSharedCheck_125_;
goto v_resetjp_119_;
}
v_resetjp_119_:
{
lean_object* v___x_123_; 
if (v_isShared_121_ == 0)
{
v___x_123_ = v___x_120_;
goto v_reusejp_122_;
}
else
{
lean_object* v_reuseFailAlloc_124_; 
v_reuseFailAlloc_124_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_124_, 0, v_a_118_);
v___x_123_ = v_reuseFailAlloc_124_;
goto v_reusejp_122_;
}
v_reusejp_122_:
{
return v___x_123_;
}
}
}
}
else
{
lean_object* v_a_126_; lean_object* v___x_128_; uint8_t v_isShared_129_; uint8_t v_isSharedCheck_133_; 
v_a_126_ = lean_ctor_get(v___x_112_, 0);
v_isSharedCheck_133_ = !lean_is_exclusive(v___x_112_);
if (v_isSharedCheck_133_ == 0)
{
v___x_128_ = v___x_112_;
v_isShared_129_ = v_isSharedCheck_133_;
goto v_resetjp_127_;
}
else
{
lean_inc(v_a_126_);
lean_dec(v___x_112_);
v___x_128_ = lean_box(0);
v_isShared_129_ = v_isSharedCheck_133_;
goto v_resetjp_127_;
}
v_resetjp_127_:
{
lean_object* v___x_131_; 
if (v_isShared_129_ == 0)
{
v___x_131_ = v___x_128_;
goto v_reusejp_130_;
}
else
{
lean_object* v_reuseFailAlloc_132_; 
v_reuseFailAlloc_132_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_132_, 0, v_a_126_);
v___x_131_ = v_reuseFailAlloc_132_;
goto v_reusejp_130_;
}
v_reusejp_130_:
{
return v___x_131_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_splitGoalsAndGetNth___boxed(lean_object* v_nth_146_, lean_object* v_reverse_147_, lean_object* v_a_148_, lean_object* v_a_149_, lean_object* v_a_150_, lean_object* v_a_151_, lean_object* v_a_152_, lean_object* v_a_153_, lean_object* v_a_154_, lean_object* v_a_155_, lean_object* v_a_156_){
_start:
{
uint8_t v_reverse_boxed_157_; lean_object* v_res_158_; 
v_reverse_boxed_157_ = lean_unbox(v_reverse_147_);
v_res_158_ = lp_batteries_Batteries_Tactic_splitGoalsAndGetNth(v_nth_146_, v_reverse_boxed_157_, v_a_148_, v_a_149_, v_a_150_, v_a_151_, v_a_152_, v_a_153_, v_a_154_, v_a_155_);
lean_dec(v_a_155_);
lean_dec_ref(v_a_154_);
lean_dec(v_a_153_);
lean_dec_ref(v_a_152_);
lean_dec(v_a_151_);
lean_dec_ref(v_a_150_);
lean_dec(v_a_149_);
lean_dec_ref(v_a_148_);
lean_dec(v_nth_146_);
return v_res_158_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00Batteries_Tactic_splitGoalsAndGetNth_spec__0(lean_object* v_00_u03b1_159_, lean_object* v_msg_160_, lean_object* v___y_161_, lean_object* v___y_162_, lean_object* v___y_163_, lean_object* v___y_164_, lean_object* v___y_165_, lean_object* v___y_166_, lean_object* v___y_167_, lean_object* v___y_168_){
_start:
{
lean_object* v___x_170_; 
v___x_170_ = lp_batteries_Lean_throwError___at___00Batteries_Tactic_splitGoalsAndGetNth_spec__0___redArg(v_msg_160_, v___y_165_, v___y_166_, v___y_167_, v___y_168_);
return v___x_170_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00Batteries_Tactic_splitGoalsAndGetNth_spec__0___boxed(lean_object* v_00_u03b1_171_, lean_object* v_msg_172_, lean_object* v___y_173_, lean_object* v___y_174_, lean_object* v___y_175_, lean_object* v___y_176_, lean_object* v___y_177_, lean_object* v___y_178_, lean_object* v___y_179_, lean_object* v___y_180_, lean_object* v___y_181_){
_start:
{
lean_object* v_res_182_; 
v_res_182_ = lp_batteries_Lean_throwError___at___00Batteries_Tactic_splitGoalsAndGetNth_spec__0(v_00_u03b1_171_, v_msg_172_, v___y_173_, v___y_174_, v___y_175_, v___y_176_, v___y_177_, v___y_178_, v___y_179_, v___y_180_);
lean_dec(v___y_180_);
lean_dec_ref(v___y_179_);
lean_dec(v___y_178_);
lean_dec_ref(v___y_177_);
lean_dec(v___y_176_);
lean_dec_ref(v___y_175_);
lean_dec(v___y_174_);
lean_dec_ref(v___y_173_);
return v_res_182_;
}
}
static lean_object* _init_lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__PermuteGoals______elabRules__Batteries__Tactic__tacticPick__goal_x2d____1_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_224_; lean_object* v___x_225_; lean_object* v___x_226_; 
v___x_224_ = lean_box(0);
v___x_225_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_226_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_226_, 0, v___x_225_);
lean_ctor_set(v___x_226_, 1, v___x_224_);
return v___x_226_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__PermuteGoals______elabRules__Batteries__Tactic__tacticPick__goal_x2d____1_spec__0___redArg(){
_start:
{
lean_object* v___x_228_; lean_object* v___x_229_; 
v___x_228_ = lean_obj_once(&lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__PermuteGoals______elabRules__Batteries__Tactic__tacticPick__goal_x2d____1_spec__0___redArg___closed__0, &lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__PermuteGoals______elabRules__Batteries__Tactic__tacticPick__goal_x2d____1_spec__0___redArg___closed__0_once, _init_lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__PermuteGoals______elabRules__Batteries__Tactic__tacticPick__goal_x2d____1_spec__0___redArg___closed__0);
v___x_229_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_229_, 0, v___x_228_);
return v___x_229_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__PermuteGoals______elabRules__Batteries__Tactic__tacticPick__goal_x2d____1_spec__0___redArg___boxed(lean_object* v___y_230_){
_start:
{
lean_object* v_res_231_; 
v_res_231_ = lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__PermuteGoals______elabRules__Batteries__Tactic__tacticPick__goal_x2d____1_spec__0___redArg();
return v_res_231_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__PermuteGoals______elabRules__Batteries__Tactic__tacticPick__goal_x2d____1_spec__0(lean_object* v_00_u03b1_232_, lean_object* v___y_233_, lean_object* v___y_234_, lean_object* v___y_235_, lean_object* v___y_236_, lean_object* v___y_237_, lean_object* v___y_238_, lean_object* v___y_239_, lean_object* v___y_240_){
_start:
{
lean_object* v___x_242_; 
v___x_242_ = lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__PermuteGoals______elabRules__Batteries__Tactic__tacticPick__goal_x2d____1_spec__0___redArg();
return v___x_242_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__PermuteGoals______elabRules__Batteries__Tactic__tacticPick__goal_x2d____1_spec__0___boxed(lean_object* v_00_u03b1_243_, lean_object* v___y_244_, lean_object* v___y_245_, lean_object* v___y_246_, lean_object* v___y_247_, lean_object* v___y_248_, lean_object* v___y_249_, lean_object* v___y_250_, lean_object* v___y_251_, lean_object* v___y_252_){
_start:
{
lean_object* v_res_253_; 
v_res_253_ = lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__PermuteGoals______elabRules__Batteries__Tactic__tacticPick__goal_x2d____1_spec__0(v_00_u03b1_243_, v___y_244_, v___y_245_, v___y_246_, v___y_247_, v___y_248_, v___y_249_, v___y_250_, v___y_251_);
lean_dec(v___y_251_);
lean_dec_ref(v___y_250_);
lean_dec(v___y_249_);
lean_dec_ref(v___y_248_);
lean_dec(v___y_247_);
lean_dec_ref(v___y_246_);
lean_dec(v___y_245_);
lean_dec_ref(v___y_244_);
return v_res_253_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__PermuteGoals______elabRules__Batteries__Tactic__tacticPick__goal_x2d____1(lean_object* v_x_254_, lean_object* v_a_255_, lean_object* v_a_256_, lean_object* v_a_257_, lean_object* v_a_258_, lean_object* v_a_259_, lean_object* v_a_260_, lean_object* v_a_261_, lean_object* v_a_262_){
_start:
{
lean_object* v___y_265_; uint8_t v___y_266_; lean_object* v___x_290_; uint8_t v___x_291_; 
v___x_290_ = ((lean_object*)(lp_batteries_Batteries_Tactic_tacticPick__goal_x2d___00__closed__3));
lean_inc(v_x_254_);
v___x_291_ = l_Lean_Syntax_isOfKind(v_x_254_, v___x_290_);
if (v___x_291_ == 0)
{
lean_object* v___x_292_; 
lean_dec(v_x_254_);
v___x_292_ = lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__PermuteGoals______elabRules__Batteries__Tactic__tacticPick__goal_x2d____1_spec__0___redArg();
return v___x_292_;
}
else
{
lean_object* v___x_293_; lean_object* v___x_294_; lean_object* v___x_295_; lean_object* v___x_296_; lean_object* v___x_297_; 
v___x_293_ = lean_unsigned_to_nat(1u);
v___x_294_ = l_Lean_Syntax_getArg(v_x_254_, v___x_293_);
v___x_295_ = lean_unsigned_to_nat(2u);
v___x_296_ = l_Lean_Syntax_getArg(v_x_254_, v___x_295_);
lean_dec(v_x_254_);
v___x_297_ = l_Lean_Syntax_getOptional_x3f(v___x_294_);
lean_dec(v___x_294_);
if (lean_obj_tag(v___x_297_) == 0)
{
lean_object* v___x_298_; 
v___x_298_ = l_Lean_Syntax_toNat(v___x_296_);
lean_dec(v___x_296_);
if (v___x_291_ == 0)
{
v___y_265_ = v___x_298_;
v___y_266_ = v___x_291_;
goto v___jp_264_;
}
else
{
uint8_t v___x_299_; 
v___x_299_ = 0;
v___y_265_ = v___x_298_;
v___y_266_ = v___x_299_;
goto v___jp_264_;
}
}
else
{
lean_object* v___x_300_; 
lean_dec_ref_known(v___x_297_, 1);
v___x_300_ = l_Lean_Syntax_toNat(v___x_296_);
lean_dec(v___x_296_);
v___y_265_ = v___x_300_;
v___y_266_ = v___x_291_;
goto v___jp_264_;
}
}
v___jp_264_:
{
lean_object* v___x_267_; 
v___x_267_ = lp_batteries_Batteries_Tactic_splitGoalsAndGetNth(v___y_265_, v___y_266_, v_a_255_, v_a_256_, v_a_257_, v_a_258_, v_a_259_, v_a_260_, v_a_261_, v_a_262_);
lean_dec(v___y_265_);
if (lean_obj_tag(v___x_267_) == 0)
{
lean_object* v_a_268_; lean_object* v_snd_269_; lean_object* v_fst_270_; lean_object* v_fst_271_; lean_object* v_snd_272_; lean_object* v___x_274_; uint8_t v_isShared_275_; uint8_t v_isSharedCheck_281_; 
v_a_268_ = lean_ctor_get(v___x_267_, 0);
lean_inc(v_a_268_);
lean_dec_ref_known(v___x_267_, 1);
v_snd_269_ = lean_ctor_get(v_a_268_, 1);
lean_inc(v_snd_269_);
v_fst_270_ = lean_ctor_get(v_a_268_, 0);
lean_inc(v_fst_270_);
lean_dec(v_a_268_);
v_fst_271_ = lean_ctor_get(v_snd_269_, 0);
v_snd_272_ = lean_ctor_get(v_snd_269_, 1);
v_isSharedCheck_281_ = !lean_is_exclusive(v_snd_269_);
if (v_isSharedCheck_281_ == 0)
{
v___x_274_ = v_snd_269_;
v_isShared_275_ = v_isSharedCheck_281_;
goto v_resetjp_273_;
}
else
{
lean_inc(v_snd_272_);
lean_inc(v_fst_271_);
lean_dec(v_snd_269_);
v___x_274_ = lean_box(0);
v_isShared_275_ = v_isSharedCheck_281_;
goto v_resetjp_273_;
}
v_resetjp_273_:
{
lean_object* v___x_276_; lean_object* v___x_278_; 
v___x_276_ = l_List_appendTR___redArg(v_fst_271_, v_snd_272_);
if (v_isShared_275_ == 0)
{
lean_ctor_set_tag(v___x_274_, 1);
lean_ctor_set(v___x_274_, 1, v___x_276_);
lean_ctor_set(v___x_274_, 0, v_fst_270_);
v___x_278_ = v___x_274_;
goto v_reusejp_277_;
}
else
{
lean_object* v_reuseFailAlloc_280_; 
v_reuseFailAlloc_280_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_280_, 0, v_fst_270_);
lean_ctor_set(v_reuseFailAlloc_280_, 1, v___x_276_);
v___x_278_ = v_reuseFailAlloc_280_;
goto v_reusejp_277_;
}
v_reusejp_277_:
{
lean_object* v___x_279_; 
v___x_279_ = l_Lean_Elab_Tactic_setGoals___redArg(v___x_278_, v_a_256_);
return v___x_279_;
}
}
}
else
{
lean_object* v_a_282_; lean_object* v___x_284_; uint8_t v_isShared_285_; uint8_t v_isSharedCheck_289_; 
v_a_282_ = lean_ctor_get(v___x_267_, 0);
v_isSharedCheck_289_ = !lean_is_exclusive(v___x_267_);
if (v_isSharedCheck_289_ == 0)
{
v___x_284_ = v___x_267_;
v_isShared_285_ = v_isSharedCheck_289_;
goto v_resetjp_283_;
}
else
{
lean_inc(v_a_282_);
lean_dec(v___x_267_);
v___x_284_ = lean_box(0);
v_isShared_285_ = v_isSharedCheck_289_;
goto v_resetjp_283_;
}
v_resetjp_283_:
{
lean_object* v___x_287_; 
if (v_isShared_285_ == 0)
{
v___x_287_ = v___x_284_;
goto v_reusejp_286_;
}
else
{
lean_object* v_reuseFailAlloc_288_; 
v_reuseFailAlloc_288_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_288_, 0, v_a_282_);
v___x_287_ = v_reuseFailAlloc_288_;
goto v_reusejp_286_;
}
v_reusejp_286_:
{
return v___x_287_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__PermuteGoals______elabRules__Batteries__Tactic__tacticPick__goal_x2d____1___boxed(lean_object* v_x_301_, lean_object* v_a_302_, lean_object* v_a_303_, lean_object* v_a_304_, lean_object* v_a_305_, lean_object* v_a_306_, lean_object* v_a_307_, lean_object* v_a_308_, lean_object* v_a_309_, lean_object* v_a_310_){
_start:
{
lean_object* v_res_311_; 
v_res_311_ = lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__PermuteGoals______elabRules__Batteries__Tactic__tacticPick__goal_x2d____1(v_x_301_, v_a_302_, v_a_303_, v_a_304_, v_a_305_, v_a_306_, v_a_307_, v_a_308_, v_a_309_);
lean_dec(v_a_309_);
lean_dec_ref(v_a_308_);
lean_dec(v_a_307_);
lean_dec_ref(v_a_306_);
lean_dec(v_a_305_);
lean_dec_ref(v_a_304_);
lean_dec(v_a_303_);
lean_dec_ref(v_a_302_);
return v_res_311_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__PermuteGoals______macroRules__Batteries__Tactic__tacticSwap__1___closed__3(void){
_start:
{
lean_object* v___x_330_; 
v___x_330_ = l_Array_mkArray0(lean_box(0));
return v___x_330_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__PermuteGoals______macroRules__Batteries__Tactic__tacticSwap__1(lean_object* v_x_332_, lean_object* v_a_333_, lean_object* v_a_334_){
_start:
{
lean_object* v___x_335_; uint8_t v___x_336_; 
v___x_335_ = ((lean_object*)(lp_batteries_Batteries_Tactic_tacticSwap___closed__1));
v___x_336_ = l_Lean_Syntax_isOfKind(v_x_332_, v___x_335_);
if (v___x_336_ == 0)
{
lean_object* v___x_337_; lean_object* v___x_338_; 
v___x_337_ = lean_box(1);
v___x_338_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_338_, 0, v___x_337_);
lean_ctor_set(v___x_338_, 1, v_a_334_);
return v___x_338_;
}
else
{
lean_object* v_ref_339_; uint8_t v___x_340_; lean_object* v___x_341_; lean_object* v___x_342_; lean_object* v___x_343_; lean_object* v___x_344_; lean_object* v___x_345_; lean_object* v___x_346_; lean_object* v___x_347_; lean_object* v___x_348_; lean_object* v___x_349_; lean_object* v___x_350_; lean_object* v___x_351_; lean_object* v___x_352_; lean_object* v___x_353_; 
v_ref_339_ = lean_ctor_get(v_a_333_, 5);
v___x_340_ = 0;
v___x_341_ = l_Lean_SourceInfo_fromRef(v_ref_339_, v___x_340_);
v___x_342_ = ((lean_object*)(lp_batteries_Batteries_Tactic_tacticPick__goal_x2d___00__closed__3));
v___x_343_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__PermuteGoals______macroRules__Batteries__Tactic__tacticSwap__1___closed__0));
lean_inc_n(v___x_341_, 4);
v___x_344_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_344_, 0, v___x_341_);
lean_ctor_set(v___x_344_, 1, v___x_343_);
v___x_345_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__PermuteGoals______macroRules__Batteries__Tactic__tacticSwap__1___closed__2));
v___x_346_ = lean_obj_once(&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__PermuteGoals______macroRules__Batteries__Tactic__tacticSwap__1___closed__3, &lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__PermuteGoals______macroRules__Batteries__Tactic__tacticSwap__1___closed__3_once, _init_lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__PermuteGoals______macroRules__Batteries__Tactic__tacticSwap__1___closed__3);
v___x_347_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_347_, 0, v___x_341_);
lean_ctor_set(v___x_347_, 1, v___x_345_);
lean_ctor_set(v___x_347_, 2, v___x_346_);
v___x_348_ = ((lean_object*)(lp_batteries_Batteries_Tactic_tacticPick__goal_x2d___00__closed__15));
v___x_349_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__PermuteGoals______macroRules__Batteries__Tactic__tacticSwap__1___closed__4));
v___x_350_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_350_, 0, v___x_341_);
lean_ctor_set(v___x_350_, 1, v___x_349_);
v___x_351_ = l_Lean_Syntax_node1(v___x_341_, v___x_348_, v___x_350_);
v___x_352_ = l_Lean_Syntax_node3(v___x_341_, v___x_342_, v___x_344_, v___x_347_, v___x_351_);
v___x_353_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_353_, 0, v___x_352_);
lean_ctor_set(v___x_353_, 1, v_a_334_);
return v___x_353_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__PermuteGoals______macroRules__Batteries__Tactic__tacticSwap__1___boxed(lean_object* v_x_354_, lean_object* v_a_355_, lean_object* v_a_356_){
_start:
{
lean_object* v_res_357_; 
v_res_357_ = lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__PermuteGoals______macroRules__Batteries__Tactic__tacticSwap__1(v_x_354_, v_a_355_, v_a_356_);
lean_dec_ref(v_a_355_);
return v_res_357_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__PermuteGoals______elabRules__Batteries__Tactic__tacticOn__goal_x2d___x3d_x3e____1(lean_object* v_x_396_, lean_object* v_a_397_, lean_object* v_a_398_, lean_object* v_a_399_, lean_object* v_a_400_, lean_object* v_a_401_, lean_object* v_a_402_, lean_object* v_a_403_, lean_object* v_a_404_){
_start:
{
lean_object* v___x_406_; uint8_t v___x_407_; 
v___x_406_ = ((lean_object*)(lp_batteries_Batteries_Tactic_tacticOn__goal_x2d___x3d_x3e___00__closed__1));
lean_inc(v_x_396_);
v___x_407_ = l_Lean_Syntax_isOfKind(v_x_396_, v___x_406_);
if (v___x_407_ == 0)
{
lean_object* v___x_408_; 
lean_dec(v_x_396_);
v___x_408_ = lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__PermuteGoals______elabRules__Batteries__Tactic__tacticPick__goal_x2d____1_spec__0___redArg();
return v___x_408_;
}
else
{
lean_object* v___x_409_; lean_object* v___x_410_; lean_object* v___x_411_; lean_object* v___x_412_; lean_object* v___x_413_; lean_object* v___x_414_; lean_object* v___y_416_; uint8_t v___y_417_; lean_object* v___x_455_; 
v___x_409_ = lean_unsigned_to_nat(1u);
v___x_410_ = l_Lean_Syntax_getArg(v_x_396_, v___x_409_);
v___x_411_ = lean_unsigned_to_nat(2u);
v___x_412_ = l_Lean_Syntax_getArg(v_x_396_, v___x_411_);
v___x_413_ = lean_unsigned_to_nat(4u);
v___x_414_ = l_Lean_Syntax_getArg(v_x_396_, v___x_413_);
lean_dec(v_x_396_);
v___x_455_ = l_Lean_Syntax_getOptional_x3f(v___x_410_);
lean_dec(v___x_410_);
if (lean_obj_tag(v___x_455_) == 0)
{
lean_object* v___x_456_; 
v___x_456_ = l_Lean_Syntax_toNat(v___x_412_);
lean_dec(v___x_412_);
if (v___x_407_ == 0)
{
v___y_416_ = v___x_456_;
v___y_417_ = v___x_407_;
goto v___jp_415_;
}
else
{
uint8_t v___x_457_; 
v___x_457_ = 0;
v___y_416_ = v___x_456_;
v___y_417_ = v___x_457_;
goto v___jp_415_;
}
}
else
{
lean_object* v___x_458_; 
lean_dec_ref_known(v___x_455_, 1);
v___x_458_ = l_Lean_Syntax_toNat(v___x_412_);
lean_dec(v___x_412_);
v___y_416_ = v___x_458_;
v___y_417_ = v___x_407_;
goto v___jp_415_;
}
v___jp_415_:
{
lean_object* v___x_418_; 
v___x_418_ = lp_batteries_Batteries_Tactic_splitGoalsAndGetNth(v___y_416_, v___y_417_, v_a_397_, v_a_398_, v_a_399_, v_a_400_, v_a_401_, v_a_402_, v_a_403_, v_a_404_);
lean_dec(v___y_416_);
if (lean_obj_tag(v___x_418_) == 0)
{
lean_object* v_a_419_; lean_object* v_snd_420_; lean_object* v_fst_421_; lean_object* v_fst_422_; lean_object* v_snd_423_; lean_object* v___x_425_; uint8_t v_isShared_426_; uint8_t v_isSharedCheck_446_; 
v_a_419_ = lean_ctor_get(v___x_418_, 0);
lean_inc(v_a_419_);
lean_dec_ref_known(v___x_418_, 1);
v_snd_420_ = lean_ctor_get(v_a_419_, 1);
lean_inc(v_snd_420_);
v_fst_421_ = lean_ctor_get(v_a_419_, 0);
lean_inc(v_fst_421_);
lean_dec(v_a_419_);
v_fst_422_ = lean_ctor_get(v_snd_420_, 0);
v_snd_423_ = lean_ctor_get(v_snd_420_, 1);
v_isSharedCheck_446_ = !lean_is_exclusive(v_snd_420_);
if (v_isSharedCheck_446_ == 0)
{
v___x_425_ = v_snd_420_;
v_isShared_426_ = v_isSharedCheck_446_;
goto v_resetjp_424_;
}
else
{
lean_inc(v_snd_423_);
lean_inc(v_fst_422_);
lean_dec(v_snd_420_);
v___x_425_ = lean_box(0);
v_isShared_426_ = v_isSharedCheck_446_;
goto v_resetjp_424_;
}
v_resetjp_424_:
{
lean_object* v___x_427_; lean_object* v___x_429_; 
v___x_427_ = lean_box(0);
if (v_isShared_426_ == 0)
{
lean_ctor_set_tag(v___x_425_, 1);
lean_ctor_set(v___x_425_, 1, v___x_427_);
lean_ctor_set(v___x_425_, 0, v_fst_421_);
v___x_429_ = v___x_425_;
goto v_reusejp_428_;
}
else
{
lean_object* v_reuseFailAlloc_445_; 
v_reuseFailAlloc_445_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_445_, 0, v_fst_421_);
lean_ctor_set(v_reuseFailAlloc_445_, 1, v___x_427_);
v___x_429_ = v_reuseFailAlloc_445_;
goto v_reusejp_428_;
}
v_reusejp_428_:
{
lean_object* v___x_430_; 
v___x_430_ = l_Lean_Elab_Tactic_setGoals___redArg(v___x_429_, v_a_398_);
if (lean_obj_tag(v___x_430_) == 0)
{
lean_object* v___x_431_; 
lean_dec_ref_known(v___x_430_, 1);
v___x_431_ = l_Lean_Elab_Tactic_evalTactic(v___x_414_, v_a_397_, v_a_398_, v_a_399_, v_a_400_, v_a_401_, v_a_402_, v_a_403_, v_a_404_);
if (lean_obj_tag(v___x_431_) == 0)
{
lean_object* v___x_432_; 
lean_dec_ref_known(v___x_431_, 1);
v___x_432_ = l_Lean_Elab_Tactic_getUnsolvedGoals(v_a_397_, v_a_398_, v_a_399_, v_a_400_, v_a_401_, v_a_402_, v_a_403_, v_a_404_);
if (lean_obj_tag(v___x_432_) == 0)
{
lean_object* v_a_433_; lean_object* v___x_434_; lean_object* v___x_435_; lean_object* v___x_436_; 
v_a_433_ = lean_ctor_get(v___x_432_, 0);
lean_inc(v_a_433_);
lean_dec_ref_known(v___x_432_, 1);
v___x_434_ = l_List_appendTR___redArg(v_fst_422_, v_a_433_);
v___x_435_ = l_List_appendTR___redArg(v___x_434_, v_snd_423_);
v___x_436_ = l_Lean_Elab_Tactic_setGoals___redArg(v___x_435_, v_a_398_);
return v___x_436_;
}
else
{
lean_object* v_a_437_; lean_object* v___x_439_; uint8_t v_isShared_440_; uint8_t v_isSharedCheck_444_; 
lean_dec(v_snd_423_);
lean_dec(v_fst_422_);
v_a_437_ = lean_ctor_get(v___x_432_, 0);
v_isSharedCheck_444_ = !lean_is_exclusive(v___x_432_);
if (v_isSharedCheck_444_ == 0)
{
v___x_439_ = v___x_432_;
v_isShared_440_ = v_isSharedCheck_444_;
goto v_resetjp_438_;
}
else
{
lean_inc(v_a_437_);
lean_dec(v___x_432_);
v___x_439_ = lean_box(0);
v_isShared_440_ = v_isSharedCheck_444_;
goto v_resetjp_438_;
}
v_resetjp_438_:
{
lean_object* v___x_442_; 
if (v_isShared_440_ == 0)
{
v___x_442_ = v___x_439_;
goto v_reusejp_441_;
}
else
{
lean_object* v_reuseFailAlloc_443_; 
v_reuseFailAlloc_443_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_443_, 0, v_a_437_);
v___x_442_ = v_reuseFailAlloc_443_;
goto v_reusejp_441_;
}
v_reusejp_441_:
{
return v___x_442_;
}
}
}
}
else
{
lean_dec(v_snd_423_);
lean_dec(v_fst_422_);
return v___x_431_;
}
}
else
{
lean_dec(v_snd_423_);
lean_dec(v_fst_422_);
lean_dec(v___x_414_);
return v___x_430_;
}
}
}
}
else
{
lean_object* v_a_447_; lean_object* v___x_449_; uint8_t v_isShared_450_; uint8_t v_isSharedCheck_454_; 
lean_dec(v___x_414_);
v_a_447_ = lean_ctor_get(v___x_418_, 0);
v_isSharedCheck_454_ = !lean_is_exclusive(v___x_418_);
if (v_isSharedCheck_454_ == 0)
{
v___x_449_ = v___x_418_;
v_isShared_450_ = v_isSharedCheck_454_;
goto v_resetjp_448_;
}
else
{
lean_inc(v_a_447_);
lean_dec(v___x_418_);
v___x_449_ = lean_box(0);
v_isShared_450_ = v_isSharedCheck_454_;
goto v_resetjp_448_;
}
v_resetjp_448_:
{
lean_object* v___x_452_; 
if (v_isShared_450_ == 0)
{
v___x_452_ = v___x_449_;
goto v_reusejp_451_;
}
else
{
lean_object* v_reuseFailAlloc_453_; 
v_reuseFailAlloc_453_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_453_, 0, v_a_447_);
v___x_452_ = v_reuseFailAlloc_453_;
goto v_reusejp_451_;
}
v_reusejp_451_:
{
return v___x_452_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__PermuteGoals______elabRules__Batteries__Tactic__tacticOn__goal_x2d___x3d_x3e____1___boxed(lean_object* v_x_459_, lean_object* v_a_460_, lean_object* v_a_461_, lean_object* v_a_462_, lean_object* v_a_463_, lean_object* v_a_464_, lean_object* v_a_465_, lean_object* v_a_466_, lean_object* v_a_467_, lean_object* v_a_468_){
_start:
{
lean_object* v_res_469_; 
v_res_469_ = lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__PermuteGoals______elabRules__Batteries__Tactic__tacticOn__goal_x2d___x3d_x3e____1(v_x_459_, v_a_460_, v_a_461_, v_a_462_, v_a_463_, v_a_464_, v_a_465_, v_a_466_, v_a_467_);
lean_dec(v_a_467_);
lean_dec_ref(v_a_466_);
lean_dec(v_a_465_);
lean_dec_ref(v_a_464_);
lean_dec(v_a_463_);
lean_dec_ref(v_a_462_);
lean_dec(v_a_461_);
lean_dec_ref(v_a_460_);
return v_res_469_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
void lean_initialize_runtime_module();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_batteries_Batteries_Tactic_PermuteGoals(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize_runtime_module();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_Tactic_Basic(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_batteries_Batteries_Tactic_PermuteGoals(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Tactic_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Lean_Elab_Tactic_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_batteries_Batteries_Tactic_PermuteGoals(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_Tactic_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Tactic_PermuteGoals(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_batteries_Batteries_Tactic_PermuteGoals(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_batteries_Batteries_Tactic_PermuteGoals(builtin);
}
#ifdef __cplusplus
}
#endif
