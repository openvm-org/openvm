// Lean compiler output
// Module: Aesop.Script.UScript
// Imports: public import Init public meta import Init public import Aesop.Script.Step import Batteries.Lean.Meta.SavedState
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
lean_object* lean_st_ref_get(lean_object*);
lean_object* lp_aesop_Aesop_Script_TacticState_getVisibleGoalIndex_x3f(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_MessageData_ofName(lean_object*);
lean_object* lp_aesop_Aesop_Script_Step_render___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_Script_TacticState_mkInitial___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_batteries_Lean_Meta_SavedState_runMetaM_x27___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* lean_mk_array(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
size_t lean_usize_add(size_t, size_t);
uint8_t l_Lean_instBEqMVarId_beq(lean_object*, lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_Script_TacticState_eraseSolvedGoals(lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_Script_mkOnGoal(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Array_mkArray0(lean_object*);
lean_object* l_Lean_Syntax_SepArray_ofElems(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* lp_aesop_Aesop_Script_Step_validate(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_UScript_render___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_UScript_render___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_UScript_render___redArg___lam__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_UScript_render___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_UScript_render(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__1_spec__2_spec__4_spec__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__1_spec__2_spec__4_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00__private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__1_spec__2_spec__4___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00__private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__1_spec__2_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__1_spec__2___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "internal error: "};
static const lean_object* lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__1_spec__2___redArg___closed__0 = (const lean_object*)&lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__1_spec__2___redArg___closed__0_value;
static lean_once_cell_t lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__1_spec__2___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__1_spec__2___redArg___closed__1;
static const lean_string_object lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__1_spec__2___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = ": unknown goal '\?"};
static const lean_object* lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__1_spec__2___redArg___closed__2 = (const lean_object*)&lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__1_spec__2___redArg___closed__2_value;
static lean_once_cell_t lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__1_spec__2___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__1_spec__2___redArg___closed__3;
static const lean_string_object lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__1_spec__2___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "'"};
static const lean_object* lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__1_spec__2___redArg___closed__4 = (const lean_object*)&lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__1_spec__2___redArg___closed__4_value;
static lean_once_cell_t lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__1_spec__2___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__1_spec__2___redArg___closed__5;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__1_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__1_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_replaceWithArray___at___00Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__2_spec__4_spec__7_spec__9(lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_replaceWithArray___at___00Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__2_spec__4_spec__7_spec__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_replaceWithArray___at___00Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__2_spec__4_spec__7(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_replaceWithArray___at___00Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__2_spec__4_spec__7___boxed(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop_Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__2_spec__4___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__2_spec__4___closed__0;
static lean_once_cell_t lp_aesop_Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__2_spec__4___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__2_spec__4___closed__1;
static const lean_string_object lp_aesop_Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__2_spec__4___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "applyTactic"};
static const lean_object* lp_aesop_Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__2_spec__4___closed__2 = (const lean_object*)&lp_aesop_Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__2_spec__4___closed__2_value;
static const lean_ctor_object lp_aesop_Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__2_spec__4___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__2_spec__4___closed__2_value)}};
static const lean_object* lp_aesop_Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__2_spec__4___closed__3 = (const lean_object*)&lp_aesop_Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__2_spec__4___closed__3_value;
static lean_once_cell_t lp_aesop_Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__2_spec__4___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__2_spec__4___closed__4;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__2_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__2_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_applyStep___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_applyStep___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "getVisibleGoalIndex"};
static const lean_object* lp_aesop_Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__1___closed__0 = (const lean_object*)&lp_aesop_Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__1___closed__0_value;
static const lean_ctor_object lp_aesop_Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__1___closed__0_value)}};
static const lean_object* lp_aesop_Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__1___closed__1 = (const lean_object*)&lp_aesop_Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__1___closed__1_value;
static lean_once_cell_t lp_aesop_Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__1___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__1___closed__2;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__1(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_Script_UScript_renderTacticSeq___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_aesop_Aesop_Script_UScript_renderTacticSeq___closed__0 = (const lean_object*)&lp_aesop_Aesop_Script_UScript_renderTacticSeq___closed__0_value;
static const lean_string_object lp_aesop_Aesop_Script_UScript_renderTacticSeq___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_aesop_Aesop_Script_UScript_renderTacticSeq___closed__1 = (const lean_object*)&lp_aesop_Aesop_Script_UScript_renderTacticSeq___closed__1_value;
static const lean_string_object lp_aesop_Aesop_Script_UScript_renderTacticSeq___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_aesop_Aesop_Script_UScript_renderTacticSeq___closed__2 = (const lean_object*)&lp_aesop_Aesop_Script_UScript_renderTacticSeq___closed__2_value;
static const lean_string_object lp_aesop_Aesop_Script_UScript_renderTacticSeq___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tacticSeq"};
static const lean_object* lp_aesop_Aesop_Script_UScript_renderTacticSeq___closed__3 = (const lean_object*)&lp_aesop_Aesop_Script_UScript_renderTacticSeq___closed__3_value;
static const lean_ctor_object lp_aesop_Aesop_Script_UScript_renderTacticSeq___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Script_UScript_renderTacticSeq___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop_Aesop_Script_UScript_renderTacticSeq___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_UScript_renderTacticSeq___closed__4_value_aux_0),((lean_object*)&lp_aesop_Aesop_Script_UScript_renderTacticSeq___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop_Aesop_Script_UScript_renderTacticSeq___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_UScript_renderTacticSeq___closed__4_value_aux_1),((lean_object*)&lp_aesop_Aesop_Script_UScript_renderTacticSeq___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_aesop_Aesop_Script_UScript_renderTacticSeq___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_UScript_renderTacticSeq___closed__4_value_aux_2),((lean_object*)&lp_aesop_Aesop_Script_UScript_renderTacticSeq___closed__3_value),LEAN_SCALAR_PTR_LITERAL(212, 140, 85, 215, 241, 69, 7, 118)}};
static const lean_object* lp_aesop_Aesop_Script_UScript_renderTacticSeq___closed__4 = (const lean_object*)&lp_aesop_Aesop_Script_UScript_renderTacticSeq___closed__4_value;
static const lean_string_object lp_aesop_Aesop_Script_UScript_renderTacticSeq___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tacticSeq1Indented"};
static const lean_object* lp_aesop_Aesop_Script_UScript_renderTacticSeq___closed__5 = (const lean_object*)&lp_aesop_Aesop_Script_UScript_renderTacticSeq___closed__5_value;
static const lean_ctor_object lp_aesop_Aesop_Script_UScript_renderTacticSeq___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Script_UScript_renderTacticSeq___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop_Aesop_Script_UScript_renderTacticSeq___closed__6_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_UScript_renderTacticSeq___closed__6_value_aux_0),((lean_object*)&lp_aesop_Aesop_Script_UScript_renderTacticSeq___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop_Aesop_Script_UScript_renderTacticSeq___closed__6_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_UScript_renderTacticSeq___closed__6_value_aux_1),((lean_object*)&lp_aesop_Aesop_Script_UScript_renderTacticSeq___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_aesop_Aesop_Script_UScript_renderTacticSeq___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_UScript_renderTacticSeq___closed__6_value_aux_2),((lean_object*)&lp_aesop_Aesop_Script_UScript_renderTacticSeq___closed__5_value),LEAN_SCALAR_PTR_LITERAL(223, 90, 160, 238, 133, 180, 23, 239)}};
static const lean_object* lp_aesop_Aesop_Script_UScript_renderTacticSeq___closed__6 = (const lean_object*)&lp_aesop_Aesop_Script_UScript_renderTacticSeq___closed__6_value;
static const lean_string_object lp_aesop_Aesop_Script_UScript_renderTacticSeq___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_aesop_Aesop_Script_UScript_renderTacticSeq___closed__7 = (const lean_object*)&lp_aesop_Aesop_Script_UScript_renderTacticSeq___closed__7_value;
static const lean_ctor_object lp_aesop_Aesop_Script_UScript_renderTacticSeq___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Script_UScript_renderTacticSeq___closed__7_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_aesop_Aesop_Script_UScript_renderTacticSeq___closed__8 = (const lean_object*)&lp_aesop_Aesop_Script_UScript_renderTacticSeq___closed__8_value;
static lean_once_cell_t lp_aesop_Aesop_Script_UScript_renderTacticSeq___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Script_UScript_renderTacticSeq___closed__9;
static const lean_string_object lp_aesop_Aesop_Script_UScript_renderTacticSeq___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_aesop_Aesop_Script_UScript_renderTacticSeq___closed__10 = (const lean_object*)&lp_aesop_Aesop_Script_UScript_renderTacticSeq___closed__10_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_UScript_renderTacticSeq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_UScript_renderTacticSeq___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00__private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__1_spec__2_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00__private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__1_spec__2_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Script_UScript_validate_spec__0(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Script_UScript_validate_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_UScript_validate(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_UScript_validate___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_UScript_render___redArg___lam__0(lean_object* v_toPure_1_, lean_object* v_____x_2_){
_start:
{
lean_object* v___x_3_; lean_object* v___x_4_; 
v___x_3_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3_, 0, v_____x_2_);
v___x_4_ = lean_apply_2(v_toPure_1_, lean_box(0), v___x_3_);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_UScript_render___redArg___lam__1(lean_object* v_inst_5_, lean_object* v_inst_6_, lean_object* v_toBind_7_, lean_object* v___f_8_, lean_object* v_a_9_, lean_object* v_x_10_, lean_object* v___y_11_){
_start:
{
lean_object* v_fst_12_; lean_object* v_snd_13_; lean_object* v___x_14_; lean_object* v___x_15_; 
v_fst_12_ = lean_ctor_get(v___y_11_, 0);
lean_inc(v_fst_12_);
v_snd_13_ = lean_ctor_get(v___y_11_, 1);
lean_inc(v_snd_13_);
lean_dec_ref(v___y_11_);
v___x_14_ = lp_aesop_Aesop_Script_Step_render___redArg(v_inst_5_, v_inst_6_, v_fst_12_, v_a_9_, v_snd_13_);
v___x_15_ = lean_apply_4(v_toBind_7_, lean_box(0), lean_box(0), v___x_14_, v___f_8_);
return v___x_15_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_UScript_render___redArg___lam__2(lean_object* v_toPure_16_, lean_object* v_____s_17_){
_start:
{
lean_object* v_fst_18_; lean_object* v___x_19_; 
v_fst_18_ = lean_ctor_get(v_____s_17_, 0);
lean_inc(v_fst_18_);
lean_dec_ref(v_____s_17_);
v___x_19_ = lean_apply_2(v_toPure_16_, lean_box(0), v_fst_18_);
return v___x_19_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_UScript_render___redArg(lean_object* v_inst_20_, lean_object* v_inst_21_, lean_object* v_tacticState_22_, lean_object* v_s_23_){
_start:
{
lean_object* v_toApplicative_24_; lean_object* v_toBind_25_; lean_object* v_toPure_26_; lean_object* v___x_27_; lean_object* v_script_28_; lean_object* v___x_29_; lean_object* v___f_30_; lean_object* v___f_31_; lean_object* v___f_32_; size_t v_sz_33_; size_t v___x_34_; lean_object* v___x_35_; lean_object* v___x_36_; 
v_toApplicative_24_ = lean_ctor_get(v_inst_20_, 0);
v_toBind_25_ = lean_ctor_get(v_inst_20_, 1);
lean_inc_n(v_toBind_25_, 2);
v_toPure_26_ = lean_ctor_get(v_toApplicative_24_, 1);
v___x_27_ = lean_array_get_size(v_s_23_);
v_script_28_ = lean_mk_empty_array_with_capacity(v___x_27_);
v___x_29_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_29_, 0, v_script_28_);
lean_ctor_set(v___x_29_, 1, v_tacticState_22_);
lean_inc_n(v_toPure_26_, 2);
v___f_30_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Script_UScript_render___redArg___lam__0), 2, 1);
lean_closure_set(v___f_30_, 0, v_toPure_26_);
lean_inc_ref(v_inst_20_);
v___f_31_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Script_UScript_render___redArg___lam__1), 7, 4);
lean_closure_set(v___f_31_, 0, v_inst_20_);
lean_closure_set(v___f_31_, 1, v_inst_21_);
lean_closure_set(v___f_31_, 2, v_toBind_25_);
lean_closure_set(v___f_31_, 3, v___f_30_);
v___f_32_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Script_UScript_render___redArg___lam__2), 2, 1);
lean_closure_set(v___f_32_, 0, v_toPure_26_);
v_sz_33_ = lean_array_size(v_s_23_);
v___x_34_ = ((size_t)0ULL);
v___x_35_ = l___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop(lean_box(0), lean_box(0), lean_box(0), v_inst_20_, v_s_23_, v___f_31_, v_sz_33_, v___x_34_, v___x_29_);
v___x_36_ = lean_apply_4(v_toBind_25_, lean_box(0), lean_box(0), v___x_35_, v___f_32_);
return v___x_36_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_UScript_render(lean_object* v_m_37_, lean_object* v_inst_38_, lean_object* v_inst_39_, lean_object* v_tacticState_40_, lean_object* v_s_41_){
_start:
{
lean_object* v___x_42_; 
v___x_42_ = lp_aesop_Aesop_Script_UScript_render___redArg(v_inst_38_, v_inst_39_, v_tacticState_40_, v_s_41_);
return v___x_42_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__1_spec__2_spec__4_spec__6(lean_object* v_msgData_43_, lean_object* v___y_44_, lean_object* v___y_45_, lean_object* v___y_46_, lean_object* v___y_47_){
_start:
{
lean_object* v___x_49_; lean_object* v_env_50_; lean_object* v___x_51_; lean_object* v_mctx_52_; lean_object* v_lctx_53_; lean_object* v_options_54_; lean_object* v___x_55_; lean_object* v___x_56_; lean_object* v___x_57_; 
v___x_49_ = lean_st_ref_get(v___y_47_);
v_env_50_ = lean_ctor_get(v___x_49_, 0);
lean_inc_ref(v_env_50_);
lean_dec(v___x_49_);
v___x_51_ = lean_st_ref_get(v___y_45_);
v_mctx_52_ = lean_ctor_get(v___x_51_, 0);
lean_inc_ref(v_mctx_52_);
lean_dec(v___x_51_);
v_lctx_53_ = lean_ctor_get(v___y_44_, 2);
v_options_54_ = lean_ctor_get(v___y_46_, 2);
lean_inc_ref(v_options_54_);
lean_inc_ref(v_lctx_53_);
v___x_55_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_55_, 0, v_env_50_);
lean_ctor_set(v___x_55_, 1, v_mctx_52_);
lean_ctor_set(v___x_55_, 2, v_lctx_53_);
lean_ctor_set(v___x_55_, 3, v_options_54_);
v___x_56_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_56_, 0, v___x_55_);
lean_ctor_set(v___x_56_, 1, v_msgData_43_);
v___x_57_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_57_, 0, v___x_56_);
return v___x_57_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__1_spec__2_spec__4_spec__6___boxed(lean_object* v_msgData_58_, lean_object* v___y_59_, lean_object* v___y_60_, lean_object* v___y_61_, lean_object* v___y_62_, lean_object* v___y_63_){
_start:
{
lean_object* v_res_64_; 
v_res_64_ = lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__1_spec__2_spec__4_spec__6(v_msgData_58_, v___y_59_, v___y_60_, v___y_61_, v___y_62_);
lean_dec(v___y_62_);
lean_dec_ref(v___y_61_);
lean_dec(v___y_60_);
lean_dec_ref(v___y_59_);
return v_res_64_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00__private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__1_spec__2_spec__4___redArg(lean_object* v_msg_65_, lean_object* v___y_66_, lean_object* v___y_67_, lean_object* v___y_68_, lean_object* v___y_69_){
_start:
{
lean_object* v_ref_71_; lean_object* v___x_72_; lean_object* v_a_73_; lean_object* v___x_75_; uint8_t v_isShared_76_; uint8_t v_isSharedCheck_81_; 
v_ref_71_ = lean_ctor_get(v___y_68_, 5);
v___x_72_ = lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__1_spec__2_spec__4_spec__6(v_msg_65_, v___y_66_, v___y_67_, v___y_68_, v___y_69_);
v_a_73_ = lean_ctor_get(v___x_72_, 0);
v_isSharedCheck_81_ = !lean_is_exclusive(v___x_72_);
if (v_isSharedCheck_81_ == 0)
{
v___x_75_ = v___x_72_;
v_isShared_76_ = v_isSharedCheck_81_;
goto v_resetjp_74_;
}
else
{
lean_inc(v_a_73_);
lean_dec(v___x_72_);
v___x_75_ = lean_box(0);
v_isShared_76_ = v_isSharedCheck_81_;
goto v_resetjp_74_;
}
v_resetjp_74_:
{
lean_object* v___x_77_; lean_object* v___x_79_; 
lean_inc(v_ref_71_);
v___x_77_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_77_, 0, v_ref_71_);
lean_ctor_set(v___x_77_, 1, v_a_73_);
if (v_isShared_76_ == 0)
{
lean_ctor_set_tag(v___x_75_, 1);
lean_ctor_set(v___x_75_, 0, v___x_77_);
v___x_79_ = v___x_75_;
goto v_reusejp_78_;
}
else
{
lean_object* v_reuseFailAlloc_80_; 
v_reuseFailAlloc_80_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_80_, 0, v___x_77_);
v___x_79_ = v_reuseFailAlloc_80_;
goto v_reusejp_78_;
}
v_reusejp_78_:
{
return v___x_79_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00__private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__1_spec__2_spec__4___redArg___boxed(lean_object* v_msg_82_, lean_object* v___y_83_, lean_object* v___y_84_, lean_object* v___y_85_, lean_object* v___y_86_, lean_object* v___y_87_){
_start:
{
lean_object* v_res_88_; 
v_res_88_ = lp_aesop_Lean_throwError___at___00__private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__1_spec__2_spec__4___redArg(v_msg_82_, v___y_83_, v___y_84_, v___y_85_, v___y_86_);
lean_dec(v___y_86_);
lean_dec_ref(v___y_85_);
lean_dec(v___y_84_);
lean_dec_ref(v___y_83_);
return v_res_88_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__1_spec__2___redArg___closed__1(void){
_start:
{
lean_object* v___x_90_; lean_object* v___x_91_; 
v___x_90_ = ((lean_object*)(lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__1_spec__2___redArg___closed__0));
v___x_91_ = l_Lean_stringToMessageData(v___x_90_);
return v___x_91_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__1_spec__2___redArg___closed__3(void){
_start:
{
lean_object* v___x_93_; lean_object* v___x_94_; 
v___x_93_ = ((lean_object*)(lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__1_spec__2___redArg___closed__2));
v___x_94_ = l_Lean_stringToMessageData(v___x_93_);
return v___x_94_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__1_spec__2___redArg___closed__5(void){
_start:
{
lean_object* v___x_96_; lean_object* v___x_97_; 
v___x_96_ = ((lean_object*)(lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__1_spec__2___redArg___closed__4));
v___x_97_ = l_Lean_stringToMessageData(v___x_96_);
return v___x_97_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__1_spec__2___redArg(lean_object* v_goal_98_, lean_object* v_pre_99_, lean_object* v___y_100_, lean_object* v___y_101_, lean_object* v___y_102_, lean_object* v___y_103_){
_start:
{
lean_object* v___x_105_; lean_object* v___x_106_; lean_object* v___x_107_; lean_object* v___x_108_; lean_object* v___x_109_; lean_object* v___x_110_; lean_object* v___x_111_; lean_object* v___x_112_; lean_object* v___x_113_; 
v___x_105_ = lean_obj_once(&lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__1_spec__2___redArg___closed__1, &lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__1_spec__2___redArg___closed__1_once, _init_lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__1_spec__2___redArg___closed__1);
v___x_106_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_106_, 0, v___x_105_);
lean_ctor_set(v___x_106_, 1, v_pre_99_);
v___x_107_ = lean_obj_once(&lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__1_spec__2___redArg___closed__3, &lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__1_spec__2___redArg___closed__3_once, _init_lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__1_spec__2___redArg___closed__3);
v___x_108_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_108_, 0, v___x_106_);
lean_ctor_set(v___x_108_, 1, v___x_107_);
v___x_109_ = l_Lean_MessageData_ofName(v_goal_98_);
v___x_110_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_110_, 0, v___x_108_);
lean_ctor_set(v___x_110_, 1, v___x_109_);
v___x_111_ = lean_obj_once(&lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__1_spec__2___redArg___closed__5, &lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__1_spec__2___redArg___closed__5_once, _init_lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__1_spec__2___redArg___closed__5);
v___x_112_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_112_, 0, v___x_110_);
lean_ctor_set(v___x_112_, 1, v___x_111_);
v___x_113_ = lp_aesop_Lean_throwError___at___00__private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__1_spec__2_spec__4___redArg(v___x_112_, v___y_100_, v___y_101_, v___y_102_, v___y_103_);
return v___x_113_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__1_spec__2___redArg___boxed(lean_object* v_goal_114_, lean_object* v_pre_115_, lean_object* v___y_116_, lean_object* v___y_117_, lean_object* v___y_118_, lean_object* v___y_119_, lean_object* v___y_120_){
_start:
{
lean_object* v_res_121_; 
v_res_121_ = lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__1_spec__2___redArg(v_goal_114_, v_pre_115_, v___y_116_, v___y_117_, v___y_118_, v___y_119_);
lean_dec(v___y_119_);
lean_dec_ref(v___y_118_);
lean_dec(v___y_117_);
lean_dec_ref(v___y_116_);
return v_res_121_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_replaceWithArray___at___00Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__2_spec__4_spec__7_spec__9(lean_object* v_x_122_, lean_object* v_r_123_, lean_object* v_as_124_, size_t v_sz_125_, size_t v_i_126_, lean_object* v_b_127_){
_start:
{
lean_object* v_a_129_; uint8_t v___x_133_; 
v___x_133_ = lean_usize_dec_lt(v_i_126_, v_sz_125_);
if (v___x_133_ == 0)
{
return v_b_127_;
}
else
{
lean_object* v_fst_134_; lean_object* v_snd_135_; lean_object* v___x_137_; uint8_t v_isShared_138_; uint8_t v_isSharedCheck_152_; 
v_fst_134_ = lean_ctor_get(v_b_127_, 0);
v_snd_135_ = lean_ctor_get(v_b_127_, 1);
v_isSharedCheck_152_ = !lean_is_exclusive(v_b_127_);
if (v_isSharedCheck_152_ == 0)
{
v___x_137_ = v_b_127_;
v_isShared_138_ = v_isSharedCheck_152_;
goto v_resetjp_136_;
}
else
{
lean_inc(v_snd_135_);
lean_inc(v_fst_134_);
lean_dec(v_b_127_);
v___x_137_ = lean_box(0);
v_isShared_138_ = v_isSharedCheck_152_;
goto v_resetjp_136_;
}
v_resetjp_136_:
{
lean_object* v_a_139_; lean_object* v_goal_140_; lean_object* v_goal_141_; uint8_t v___x_142_; 
v_a_139_ = lean_array_uget_borrowed(v_as_124_, v_i_126_);
v_goal_140_ = lean_ctor_get(v_a_139_, 0);
v_goal_141_ = lean_ctor_get(v_x_122_, 0);
v___x_142_ = l_Lean_instBEqMVarId_beq(v_goal_140_, v_goal_141_);
if (v___x_142_ == 0)
{
lean_object* v___x_143_; lean_object* v___x_145_; 
lean_inc(v_a_139_);
v___x_143_ = lean_array_push(v_snd_135_, v_a_139_);
if (v_isShared_138_ == 0)
{
lean_ctor_set(v___x_137_, 1, v___x_143_);
v___x_145_ = v___x_137_;
goto v_reusejp_144_;
}
else
{
lean_object* v_reuseFailAlloc_146_; 
v_reuseFailAlloc_146_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_146_, 0, v_fst_134_);
lean_ctor_set(v_reuseFailAlloc_146_, 1, v___x_143_);
v___x_145_ = v_reuseFailAlloc_146_;
goto v_reusejp_144_;
}
v_reusejp_144_:
{
v_a_129_ = v___x_145_;
goto v___jp_128_;
}
}
else
{
lean_object* v___x_147_; lean_object* v___x_148_; lean_object* v___x_150_; 
lean_dec(v_fst_134_);
v___x_147_ = l_Array_append___redArg(v_snd_135_, v_r_123_);
v___x_148_ = lean_box(v___x_142_);
if (v_isShared_138_ == 0)
{
lean_ctor_set(v___x_137_, 1, v___x_147_);
lean_ctor_set(v___x_137_, 0, v___x_148_);
v___x_150_ = v___x_137_;
goto v_reusejp_149_;
}
else
{
lean_object* v_reuseFailAlloc_151_; 
v_reuseFailAlloc_151_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_151_, 0, v___x_148_);
lean_ctor_set(v_reuseFailAlloc_151_, 1, v___x_147_);
v___x_150_ = v_reuseFailAlloc_151_;
goto v_reusejp_149_;
}
v_reusejp_149_:
{
v_a_129_ = v___x_150_;
goto v___jp_128_;
}
}
}
}
v___jp_128_:
{
size_t v___x_130_; size_t v___x_131_; 
v___x_130_ = ((size_t)1ULL);
v___x_131_ = lean_usize_add(v_i_126_, v___x_130_);
v_i_126_ = v___x_131_;
v_b_127_ = v_a_129_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_replaceWithArray___at___00Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__2_spec__4_spec__7_spec__9___boxed(lean_object* v_x_153_, lean_object* v_r_154_, lean_object* v_as_155_, lean_object* v_sz_156_, lean_object* v_i_157_, lean_object* v_b_158_){
_start:
{
size_t v_sz_boxed_159_; size_t v_i_boxed_160_; lean_object* v_res_161_; 
v_sz_boxed_159_ = lean_unbox_usize(v_sz_156_);
lean_dec(v_sz_156_);
v_i_boxed_160_ = lean_unbox_usize(v_i_157_);
lean_dec(v_i_157_);
v_res_161_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_replaceWithArray___at___00Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__2_spec__4_spec__7_spec__9(v_x_153_, v_r_154_, v_as_155_, v_sz_boxed_159_, v_i_boxed_160_, v_b_158_);
lean_dec_ref(v_as_155_);
lean_dec_ref(v_r_154_);
lean_dec_ref(v_x_153_);
return v_res_161_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_replaceWithArray___at___00Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__2_spec__4_spec__7(lean_object* v_xs_162_, lean_object* v_x_163_, lean_object* v_r_164_){
_start:
{
uint8_t v_found_165_; lean_object* v___x_166_; lean_object* v___x_167_; lean_object* v___x_168_; lean_object* v___x_169_; lean_object* v___x_170_; lean_object* v_ys_171_; lean_object* v___x_172_; lean_object* v___x_173_; size_t v_sz_174_; size_t v___x_175_; lean_object* v___x_176_; lean_object* v_fst_177_; uint8_t v___x_178_; 
v_found_165_ = 0;
v___x_166_ = lean_array_get_size(v_xs_162_);
v___x_167_ = lean_unsigned_to_nat(1u);
v___x_168_ = lean_nat_sub(v___x_166_, v___x_167_);
v___x_169_ = lean_array_get_size(v_r_164_);
v___x_170_ = lean_nat_add(v___x_168_, v___x_169_);
lean_dec(v___x_168_);
v_ys_171_ = lean_mk_empty_array_with_capacity(v___x_170_);
lean_dec(v___x_170_);
v___x_172_ = lean_box(v_found_165_);
v___x_173_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_173_, 0, v___x_172_);
lean_ctor_set(v___x_173_, 1, v_ys_171_);
v_sz_174_ = lean_array_size(v_xs_162_);
v___x_175_ = ((size_t)0ULL);
v___x_176_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_replaceWithArray___at___00Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__2_spec__4_spec__7_spec__9(v_x_163_, v_r_164_, v_xs_162_, v_sz_174_, v___x_175_, v___x_173_);
v_fst_177_ = lean_ctor_get(v___x_176_, 0);
lean_inc(v_fst_177_);
v___x_178_ = lean_unbox(v_fst_177_);
lean_dec(v_fst_177_);
if (v___x_178_ == 0)
{
lean_object* v___x_179_; 
lean_dec_ref(v___x_176_);
v___x_179_ = lean_box(0);
return v___x_179_;
}
else
{
lean_object* v_snd_180_; lean_object* v___x_181_; 
v_snd_180_ = lean_ctor_get(v___x_176_, 1);
lean_inc(v_snd_180_);
lean_dec_ref(v___x_176_);
v___x_181_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_181_, 0, v_snd_180_);
return v___x_181_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_replaceWithArray___at___00Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__2_spec__4_spec__7___boxed(lean_object* v_xs_182_, lean_object* v_x_183_, lean_object* v_r_184_){
_start:
{
lean_object* v_res_185_; 
v_res_185_ = lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_replaceWithArray___at___00Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__2_spec__4_spec__7(v_xs_182_, v_x_183_, v_r_184_);
lean_dec_ref(v_r_184_);
lean_dec_ref(v_x_183_);
lean_dec_ref(v_xs_182_);
return v_res_185_;
}
}
static lean_object* _init_lp_aesop_Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__2_spec__4___closed__0(void){
_start:
{
lean_object* v___x_186_; lean_object* v___x_187_; lean_object* v___x_188_; 
v___x_186_ = lean_box(0);
v___x_187_ = lean_unsigned_to_nat(16u);
v___x_188_ = lean_mk_array(v___x_187_, v___x_186_);
return v___x_188_;
}
}
static lean_object* _init_lp_aesop_Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__2_spec__4___closed__1(void){
_start:
{
lean_object* v___x_189_; lean_object* v___x_190_; lean_object* v___x_191_; 
v___x_189_ = lean_obj_once(&lp_aesop_Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__2_spec__4___closed__0, &lp_aesop_Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__2_spec__4___closed__0_once, _init_lp_aesop_Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__2_spec__4___closed__0);
v___x_190_ = lean_unsigned_to_nat(0u);
v___x_191_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_191_, 0, v___x_190_);
lean_ctor_set(v___x_191_, 1, v___x_189_);
return v___x_191_;
}
}
static lean_object* _init_lp_aesop_Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__2_spec__4___closed__4(void){
_start:
{
lean_object* v___x_195_; lean_object* v___x_196_; 
v___x_195_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__2_spec__4___closed__3));
v___x_196_ = l_Lean_MessageData_ofFormat(v___x_195_);
return v___x_196_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__2_spec__4(lean_object* v_ts_197_, lean_object* v_inGoal_198_, lean_object* v_outGoals_199_, lean_object* v_preMCtx_200_, lean_object* v_postMCtx_201_, lean_object* v___y_202_, lean_object* v___y_203_, lean_object* v___y_204_, lean_object* v___y_205_){
_start:
{
lean_object* v_visibleGoals_207_; lean_object* v_invisibleGoals_208_; lean_object* v___x_210_; uint8_t v_isShared_211_; uint8_t v_isSharedCheck_229_; 
v_visibleGoals_207_ = lean_ctor_get(v_ts_197_, 0);
v_invisibleGoals_208_ = lean_ctor_get(v_ts_197_, 1);
v_isSharedCheck_229_ = !lean_is_exclusive(v_ts_197_);
if (v_isSharedCheck_229_ == 0)
{
v___x_210_ = v_ts_197_;
v_isShared_211_ = v_isSharedCheck_229_;
goto v_resetjp_209_;
}
else
{
lean_inc(v_invisibleGoals_208_);
lean_inc(v_visibleGoals_207_);
lean_dec(v_ts_197_);
v___x_210_ = lean_box(0);
v_isShared_211_ = v_isSharedCheck_229_;
goto v_resetjp_209_;
}
v_resetjp_209_:
{
lean_object* v___x_212_; lean_object* v___x_213_; lean_object* v___x_214_; 
v___x_212_ = lean_obj_once(&lp_aesop_Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__2_spec__4___closed__1, &lp_aesop_Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__2_spec__4___closed__1_once, _init_lp_aesop_Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__2_spec__4___closed__1);
lean_inc(v_inGoal_198_);
v___x_213_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_213_, 0, v_inGoal_198_);
lean_ctor_set(v___x_213_, 1, v___x_212_);
v___x_214_ = lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_replaceWithArray___at___00Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__2_spec__4_spec__7(v_visibleGoals_207_, v___x_213_, v_outGoals_199_);
lean_dec_ref_known(v___x_213_, 2);
lean_dec_ref(v_visibleGoals_207_);
if (lean_obj_tag(v___x_214_) == 1)
{
lean_object* v_val_215_; lean_object* v___x_217_; uint8_t v_isShared_218_; uint8_t v_isSharedCheck_226_; 
lean_dec(v_inGoal_198_);
v_val_215_ = lean_ctor_get(v___x_214_, 0);
v_isSharedCheck_226_ = !lean_is_exclusive(v___x_214_);
if (v_isSharedCheck_226_ == 0)
{
v___x_217_ = v___x_214_;
v_isShared_218_ = v_isSharedCheck_226_;
goto v_resetjp_216_;
}
else
{
lean_inc(v_val_215_);
lean_dec(v___x_214_);
v___x_217_ = lean_box(0);
v_isShared_218_ = v_isSharedCheck_226_;
goto v_resetjp_216_;
}
v_resetjp_216_:
{
lean_object* v_ts_220_; 
if (v_isShared_211_ == 0)
{
lean_ctor_set(v___x_210_, 0, v_val_215_);
v_ts_220_ = v___x_210_;
goto v_reusejp_219_;
}
else
{
lean_object* v_reuseFailAlloc_225_; 
v_reuseFailAlloc_225_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_225_, 0, v_val_215_);
lean_ctor_set(v_reuseFailAlloc_225_, 1, v_invisibleGoals_208_);
v_ts_220_ = v_reuseFailAlloc_225_;
goto v_reusejp_219_;
}
v_reusejp_219_:
{
lean_object* v___x_221_; lean_object* v___x_223_; 
v___x_221_ = lp_aesop_Aesop_Script_TacticState_eraseSolvedGoals(v_ts_220_, v_preMCtx_200_, v_postMCtx_201_);
if (v_isShared_218_ == 0)
{
lean_ctor_set_tag(v___x_217_, 0);
lean_ctor_set(v___x_217_, 0, v___x_221_);
v___x_223_ = v___x_217_;
goto v_reusejp_222_;
}
else
{
lean_object* v_reuseFailAlloc_224_; 
v_reuseFailAlloc_224_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_224_, 0, v___x_221_);
v___x_223_ = v_reuseFailAlloc_224_;
goto v_reusejp_222_;
}
v_reusejp_222_:
{
return v___x_223_;
}
}
}
}
else
{
lean_object* v___x_227_; lean_object* v___x_228_; 
lean_dec(v___x_214_);
lean_del_object(v___x_210_);
lean_dec_ref(v_invisibleGoals_208_);
v___x_227_ = lean_obj_once(&lp_aesop_Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__2_spec__4___closed__4, &lp_aesop_Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__2_spec__4___closed__4_once, _init_lp_aesop_Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__2_spec__4___closed__4);
v___x_228_ = lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__1_spec__2___redArg(v_inGoal_198_, v___x_227_, v___y_202_, v___y_203_, v___y_204_, v___y_205_);
return v___x_228_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__2_spec__4___boxed(lean_object* v_ts_230_, lean_object* v_inGoal_231_, lean_object* v_outGoals_232_, lean_object* v_preMCtx_233_, lean_object* v_postMCtx_234_, lean_object* v___y_235_, lean_object* v___y_236_, lean_object* v___y_237_, lean_object* v___y_238_, lean_object* v___y_239_){
_start:
{
lean_object* v_res_240_; 
v_res_240_ = lp_aesop_Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__2_spec__4(v_ts_230_, v_inGoal_231_, v_outGoals_232_, v_preMCtx_233_, v_postMCtx_234_, v___y_235_, v___y_236_, v___y_237_, v___y_238_);
lean_dec(v___y_238_);
lean_dec_ref(v___y_237_);
lean_dec(v___y_236_);
lean_dec_ref(v___y_235_);
lean_dec_ref(v_postMCtx_234_);
lean_dec_ref(v_preMCtx_233_);
lean_dec_ref(v_outGoals_232_);
return v_res_240_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_applyStep___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__2(lean_object* v_tacticState_241_, lean_object* v_step_242_, lean_object* v___y_243_, lean_object* v___y_244_, lean_object* v___y_245_, lean_object* v___y_246_){
_start:
{
lean_object* v_preState_248_; lean_object* v_meta_249_; lean_object* v_postState_250_; lean_object* v_meta_251_; lean_object* v_preGoal_252_; lean_object* v_postGoals_253_; lean_object* v_mctx_254_; lean_object* v_mctx_255_; lean_object* v___x_256_; 
v_preState_248_ = lean_ctor_get(v_step_242_, 0);
v_meta_249_ = lean_ctor_get(v_preState_248_, 1);
lean_inc_ref(v_meta_249_);
v_postState_250_ = lean_ctor_get(v_step_242_, 3);
v_meta_251_ = lean_ctor_get(v_postState_250_, 1);
lean_inc_ref(v_meta_251_);
v_preGoal_252_ = lean_ctor_get(v_step_242_, 1);
lean_inc(v_preGoal_252_);
v_postGoals_253_ = lean_ctor_get(v_step_242_, 4);
lean_inc_ref(v_postGoals_253_);
lean_dec_ref(v_step_242_);
v_mctx_254_ = lean_ctor_get(v_meta_249_, 0);
lean_inc_ref(v_mctx_254_);
lean_dec_ref(v_meta_249_);
v_mctx_255_ = lean_ctor_get(v_meta_251_, 0);
lean_inc_ref(v_mctx_255_);
lean_dec_ref(v_meta_251_);
v___x_256_ = lp_aesop_Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__2_spec__4(v_tacticState_241_, v_preGoal_252_, v_postGoals_253_, v_mctx_254_, v_mctx_255_, v___y_243_, v___y_244_, v___y_245_, v___y_246_);
lean_dec_ref(v_mctx_255_);
lean_dec_ref(v_mctx_254_);
lean_dec_ref(v_postGoals_253_);
return v___x_256_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_applyStep___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__2___boxed(lean_object* v_tacticState_257_, lean_object* v_step_258_, lean_object* v___y_259_, lean_object* v___y_260_, lean_object* v___y_261_, lean_object* v___y_262_, lean_object* v___y_263_){
_start:
{
lean_object* v_res_264_; 
v_res_264_ = lp_aesop_Aesop_Script_TacticState_applyStep___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__2(v_tacticState_257_, v_step_258_, v___y_259_, v___y_260_, v___y_261_, v___y_262_);
lean_dec(v___y_262_);
lean_dec_ref(v___y_261_);
lean_dec(v___y_260_);
lean_dec_ref(v___y_259_);
return v_res_264_;
}
}
static lean_object* _init_lp_aesop_Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__1___closed__2(void){
_start:
{
lean_object* v___x_268_; lean_object* v___x_269_; 
v___x_268_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__1___closed__1));
v___x_269_ = l_Lean_MessageData_ofFormat(v___x_268_);
return v___x_269_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__1(lean_object* v_ts_270_, lean_object* v_goal_271_, lean_object* v___y_272_, lean_object* v___y_273_, lean_object* v___y_274_, lean_object* v___y_275_){
_start:
{
lean_object* v___x_277_; 
v___x_277_ = lp_aesop_Aesop_Script_TacticState_getVisibleGoalIndex_x3f(v_ts_270_, v_goal_271_);
if (lean_obj_tag(v___x_277_) == 1)
{
lean_object* v_val_278_; lean_object* v___x_280_; uint8_t v_isShared_281_; uint8_t v_isSharedCheck_285_; 
lean_dec(v_goal_271_);
v_val_278_ = lean_ctor_get(v___x_277_, 0);
v_isSharedCheck_285_ = !lean_is_exclusive(v___x_277_);
if (v_isSharedCheck_285_ == 0)
{
v___x_280_ = v___x_277_;
v_isShared_281_ = v_isSharedCheck_285_;
goto v_resetjp_279_;
}
else
{
lean_inc(v_val_278_);
lean_dec(v___x_277_);
v___x_280_ = lean_box(0);
v_isShared_281_ = v_isSharedCheck_285_;
goto v_resetjp_279_;
}
v_resetjp_279_:
{
lean_object* v___x_283_; 
if (v_isShared_281_ == 0)
{
lean_ctor_set_tag(v___x_280_, 0);
v___x_283_ = v___x_280_;
goto v_reusejp_282_;
}
else
{
lean_object* v_reuseFailAlloc_284_; 
v_reuseFailAlloc_284_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_284_, 0, v_val_278_);
v___x_283_ = v_reuseFailAlloc_284_;
goto v_reusejp_282_;
}
v_reusejp_282_:
{
return v___x_283_;
}
}
}
else
{
lean_object* v___x_286_; lean_object* v___x_287_; 
lean_dec(v___x_277_);
v___x_286_ = lean_obj_once(&lp_aesop_Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__1___closed__2, &lp_aesop_Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__1___closed__2_once, _init_lp_aesop_Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__1___closed__2);
v___x_287_ = lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__1_spec__2___redArg(v_goal_271_, v___x_286_, v___y_272_, v___y_273_, v___y_274_, v___y_275_);
return v___x_287_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__1___boxed(lean_object* v_ts_288_, lean_object* v_goal_289_, lean_object* v___y_290_, lean_object* v___y_291_, lean_object* v___y_292_, lean_object* v___y_293_, lean_object* v___y_294_){
_start:
{
lean_object* v_res_295_; 
v_res_295_ = lp_aesop_Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__1(v_ts_288_, v_goal_289_, v___y_290_, v___y_291_, v___y_292_, v___y_293_);
lean_dec(v___y_293_);
lean_dec_ref(v___y_292_);
lean_dec(v___y_291_);
lean_dec_ref(v___y_290_);
lean_dec_ref(v_ts_288_);
return v_res_295_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0(lean_object* v_acc_296_, lean_object* v_step_297_, lean_object* v_tacticState_298_, lean_object* v___y_299_, lean_object* v___y_300_, lean_object* v___y_301_, lean_object* v___y_302_){
_start:
{
lean_object* v_preGoal_304_; lean_object* v_tactic_305_; lean_object* v___x_306_; 
v_preGoal_304_ = lean_ctor_get(v_step_297_, 1);
v_tactic_305_ = lean_ctor_get(v_step_297_, 2);
lean_inc_ref(v_tactic_305_);
lean_inc(v_preGoal_304_);
v___x_306_ = lp_aesop_Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__1(v_tacticState_298_, v_preGoal_304_, v___y_299_, v___y_300_, v___y_301_, v___y_302_);
if (lean_obj_tag(v___x_306_) == 0)
{
lean_object* v_a_307_; lean_object* v___x_308_; 
v_a_307_ = lean_ctor_get(v___x_306_, 0);
lean_inc(v_a_307_);
lean_dec_ref_known(v___x_306_, 1);
v___x_308_ = lp_aesop_Aesop_Script_TacticState_applyStep___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__2(v_tacticState_298_, v_step_297_, v___y_299_, v___y_300_, v___y_301_, v___y_302_);
if (lean_obj_tag(v___x_308_) == 0)
{
lean_object* v_a_309_; lean_object* v___x_311_; uint8_t v_isShared_312_; uint8_t v_isSharedCheck_327_; 
v_a_309_ = lean_ctor_get(v___x_308_, 0);
v_isSharedCheck_327_ = !lean_is_exclusive(v___x_308_);
if (v_isSharedCheck_327_ == 0)
{
v___x_311_ = v___x_308_;
v_isShared_312_ = v_isSharedCheck_327_;
goto v_resetjp_310_;
}
else
{
lean_inc(v_a_309_);
lean_dec(v___x_308_);
v___x_311_ = lean_box(0);
v_isShared_312_ = v_isSharedCheck_327_;
goto v_resetjp_310_;
}
v_resetjp_310_:
{
lean_object* v_uTactic_313_; lean_object* v___x_315_; uint8_t v_isShared_316_; uint8_t v_isSharedCheck_325_; 
v_uTactic_313_ = lean_ctor_get(v_tactic_305_, 0);
v_isSharedCheck_325_ = !lean_is_exclusive(v_tactic_305_);
if (v_isSharedCheck_325_ == 0)
{
lean_object* v_unused_326_; 
v_unused_326_ = lean_ctor_get(v_tactic_305_, 1);
lean_dec(v_unused_326_);
v___x_315_ = v_tactic_305_;
v_isShared_316_ = v_isSharedCheck_325_;
goto v_resetjp_314_;
}
else
{
lean_inc(v_uTactic_313_);
lean_dec(v_tactic_305_);
v___x_315_ = lean_box(0);
v_isShared_316_ = v_isSharedCheck_325_;
goto v_resetjp_314_;
}
v_resetjp_314_:
{
lean_object* v___x_317_; lean_object* v_acc_318_; lean_object* v___x_320_; 
v___x_317_ = lp_aesop_Aesop_Script_mkOnGoal(v_a_307_, v_uTactic_313_);
lean_dec(v_a_307_);
v_acc_318_ = lean_array_push(v_acc_296_, v___x_317_);
if (v_isShared_316_ == 0)
{
lean_ctor_set(v___x_315_, 1, v_a_309_);
lean_ctor_set(v___x_315_, 0, v_acc_318_);
v___x_320_ = v___x_315_;
goto v_reusejp_319_;
}
else
{
lean_object* v_reuseFailAlloc_324_; 
v_reuseFailAlloc_324_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_324_, 0, v_acc_318_);
lean_ctor_set(v_reuseFailAlloc_324_, 1, v_a_309_);
v___x_320_ = v_reuseFailAlloc_324_;
goto v_reusejp_319_;
}
v_reusejp_319_:
{
lean_object* v___x_322_; 
if (v_isShared_312_ == 0)
{
lean_ctor_set(v___x_311_, 0, v___x_320_);
v___x_322_ = v___x_311_;
goto v_reusejp_321_;
}
else
{
lean_object* v_reuseFailAlloc_323_; 
v_reuseFailAlloc_323_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_323_, 0, v___x_320_);
v___x_322_ = v_reuseFailAlloc_323_;
goto v_reusejp_321_;
}
v_reusejp_321_:
{
return v___x_322_;
}
}
}
}
}
else
{
lean_object* v_a_328_; lean_object* v___x_330_; uint8_t v_isShared_331_; uint8_t v_isSharedCheck_335_; 
lean_dec(v_a_307_);
lean_dec_ref(v_tactic_305_);
lean_dec_ref(v_acc_296_);
v_a_328_ = lean_ctor_get(v___x_308_, 0);
v_isSharedCheck_335_ = !lean_is_exclusive(v___x_308_);
if (v_isSharedCheck_335_ == 0)
{
v___x_330_ = v___x_308_;
v_isShared_331_ = v_isSharedCheck_335_;
goto v_resetjp_329_;
}
else
{
lean_inc(v_a_328_);
lean_dec(v___x_308_);
v___x_330_ = lean_box(0);
v_isShared_331_ = v_isSharedCheck_335_;
goto v_resetjp_329_;
}
v_resetjp_329_:
{
lean_object* v___x_333_; 
if (v_isShared_331_ == 0)
{
v___x_333_ = v___x_330_;
goto v_reusejp_332_;
}
else
{
lean_object* v_reuseFailAlloc_334_; 
v_reuseFailAlloc_334_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_334_, 0, v_a_328_);
v___x_333_ = v_reuseFailAlloc_334_;
goto v_reusejp_332_;
}
v_reusejp_332_:
{
return v___x_333_;
}
}
}
}
else
{
lean_object* v_a_336_; lean_object* v___x_338_; uint8_t v_isShared_339_; uint8_t v_isSharedCheck_343_; 
lean_dec_ref(v_tactic_305_);
lean_dec_ref(v_tacticState_298_);
lean_dec_ref(v_step_297_);
lean_dec_ref(v_acc_296_);
v_a_336_ = lean_ctor_get(v___x_306_, 0);
v_isSharedCheck_343_ = !lean_is_exclusive(v___x_306_);
if (v_isSharedCheck_343_ == 0)
{
v___x_338_ = v___x_306_;
v_isShared_339_ = v_isSharedCheck_343_;
goto v_resetjp_337_;
}
else
{
lean_inc(v_a_336_);
lean_dec(v___x_306_);
v___x_338_ = lean_box(0);
v_isShared_339_ = v_isSharedCheck_343_;
goto v_resetjp_337_;
}
v_resetjp_337_:
{
lean_object* v___x_341_; 
if (v_isShared_339_ == 0)
{
v___x_341_ = v___x_338_;
goto v_reusejp_340_;
}
else
{
lean_object* v_reuseFailAlloc_342_; 
v_reuseFailAlloc_342_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_342_, 0, v_a_336_);
v___x_341_ = v_reuseFailAlloc_342_;
goto v_reusejp_340_;
}
v_reusejp_340_:
{
return v___x_341_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0___boxed(lean_object* v_acc_344_, lean_object* v_step_345_, lean_object* v_tacticState_346_, lean_object* v___y_347_, lean_object* v___y_348_, lean_object* v___y_349_, lean_object* v___y_350_, lean_object* v___y_351_){
_start:
{
lean_object* v_res_352_; 
v_res_352_ = lp_aesop_Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0(v_acc_344_, v_step_345_, v_tacticState_346_, v___y_347_, v___y_348_, v___y_349_, v___y_350_);
lean_dec(v___y_350_);
lean_dec_ref(v___y_349_);
lean_dec(v___y_348_);
lean_dec_ref(v___y_347_);
return v_res_352_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__1(lean_object* v_as_353_, size_t v_sz_354_, size_t v_i_355_, lean_object* v_b_356_, lean_object* v___y_357_, lean_object* v___y_358_, lean_object* v___y_359_, lean_object* v___y_360_){
_start:
{
uint8_t v___x_362_; 
v___x_362_ = lean_usize_dec_lt(v_i_355_, v_sz_354_);
if (v___x_362_ == 0)
{
lean_object* v___x_363_; 
v___x_363_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_363_, 0, v_b_356_);
return v___x_363_;
}
else
{
lean_object* v_fst_364_; lean_object* v_snd_365_; lean_object* v_a_366_; lean_object* v___x_367_; 
v_fst_364_ = lean_ctor_get(v_b_356_, 0);
lean_inc(v_fst_364_);
v_snd_365_ = lean_ctor_get(v_b_356_, 1);
lean_inc(v_snd_365_);
lean_dec_ref(v_b_356_);
v_a_366_ = lean_array_uget_borrowed(v_as_353_, v_i_355_);
lean_inc(v_a_366_);
v___x_367_ = lp_aesop_Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0(v_fst_364_, v_a_366_, v_snd_365_, v___y_357_, v___y_358_, v___y_359_, v___y_360_);
if (lean_obj_tag(v___x_367_) == 0)
{
lean_object* v_a_368_; size_t v___x_369_; size_t v___x_370_; 
v_a_368_ = lean_ctor_get(v___x_367_, 0);
lean_inc(v_a_368_);
lean_dec_ref_known(v___x_367_, 1);
v___x_369_ = ((size_t)1ULL);
v___x_370_ = lean_usize_add(v_i_355_, v___x_369_);
v_i_355_ = v___x_370_;
v_b_356_ = v_a_368_;
goto _start;
}
else
{
return v___x_367_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__1___boxed(lean_object* v_as_372_, lean_object* v_sz_373_, lean_object* v_i_374_, lean_object* v_b_375_, lean_object* v___y_376_, lean_object* v___y_377_, lean_object* v___y_378_, lean_object* v___y_379_, lean_object* v___y_380_){
_start:
{
size_t v_sz_boxed_381_; size_t v_i_boxed_382_; lean_object* v_res_383_; 
v_sz_boxed_381_ = lean_unbox_usize(v_sz_373_);
lean_dec(v_sz_373_);
v_i_boxed_382_ = lean_unbox_usize(v_i_374_);
lean_dec(v_i_374_);
v_res_383_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__1(v_as_372_, v_sz_boxed_381_, v_i_boxed_382_, v_b_375_, v___y_376_, v___y_377_, v___y_378_, v___y_379_);
lean_dec(v___y_379_);
lean_dec_ref(v___y_378_);
lean_dec(v___y_377_);
lean_dec_ref(v___y_376_);
lean_dec_ref(v_as_372_);
return v_res_383_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0(lean_object* v_tacticState_384_, lean_object* v_s_385_, lean_object* v___y_386_, lean_object* v___y_387_, lean_object* v___y_388_, lean_object* v___y_389_){
_start:
{
lean_object* v___x_391_; lean_object* v_script_392_; lean_object* v___x_393_; size_t v_sz_394_; size_t v___x_395_; lean_object* v___x_396_; 
v___x_391_ = lean_array_get_size(v_s_385_);
v_script_392_ = lean_mk_empty_array_with_capacity(v___x_391_);
v___x_393_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_393_, 0, v_script_392_);
lean_ctor_set(v___x_393_, 1, v_tacticState_384_);
v_sz_394_ = lean_array_size(v_s_385_);
v___x_395_ = ((size_t)0ULL);
v___x_396_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__1(v_s_385_, v_sz_394_, v___x_395_, v___x_393_, v___y_386_, v___y_387_, v___y_388_, v___y_389_);
if (lean_obj_tag(v___x_396_) == 0)
{
lean_object* v_a_397_; lean_object* v___x_399_; uint8_t v_isShared_400_; uint8_t v_isSharedCheck_405_; 
v_a_397_ = lean_ctor_get(v___x_396_, 0);
v_isSharedCheck_405_ = !lean_is_exclusive(v___x_396_);
if (v_isSharedCheck_405_ == 0)
{
v___x_399_ = v___x_396_;
v_isShared_400_ = v_isSharedCheck_405_;
goto v_resetjp_398_;
}
else
{
lean_inc(v_a_397_);
lean_dec(v___x_396_);
v___x_399_ = lean_box(0);
v_isShared_400_ = v_isSharedCheck_405_;
goto v_resetjp_398_;
}
v_resetjp_398_:
{
lean_object* v_fst_401_; lean_object* v___x_403_; 
v_fst_401_ = lean_ctor_get(v_a_397_, 0);
lean_inc(v_fst_401_);
lean_dec(v_a_397_);
if (v_isShared_400_ == 0)
{
lean_ctor_set(v___x_399_, 0, v_fst_401_);
v___x_403_ = v___x_399_;
goto v_reusejp_402_;
}
else
{
lean_object* v_reuseFailAlloc_404_; 
v_reuseFailAlloc_404_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_404_, 0, v_fst_401_);
v___x_403_ = v_reuseFailAlloc_404_;
goto v_reusejp_402_;
}
v_reusejp_402_:
{
return v___x_403_;
}
}
}
else
{
lean_object* v_a_406_; lean_object* v___x_408_; uint8_t v_isShared_409_; uint8_t v_isSharedCheck_413_; 
v_a_406_ = lean_ctor_get(v___x_396_, 0);
v_isSharedCheck_413_ = !lean_is_exclusive(v___x_396_);
if (v_isSharedCheck_413_ == 0)
{
v___x_408_ = v___x_396_;
v_isShared_409_ = v_isSharedCheck_413_;
goto v_resetjp_407_;
}
else
{
lean_inc(v_a_406_);
lean_dec(v___x_396_);
v___x_408_ = lean_box(0);
v_isShared_409_ = v_isSharedCheck_413_;
goto v_resetjp_407_;
}
v_resetjp_407_:
{
lean_object* v___x_411_; 
if (v_isShared_409_ == 0)
{
v___x_411_ = v___x_408_;
goto v_reusejp_410_;
}
else
{
lean_object* v_reuseFailAlloc_412_; 
v_reuseFailAlloc_412_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_412_, 0, v_a_406_);
v___x_411_ = v_reuseFailAlloc_412_;
goto v_reusejp_410_;
}
v_reusejp_410_:
{
return v___x_411_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0___boxed(lean_object* v_tacticState_414_, lean_object* v_s_415_, lean_object* v___y_416_, lean_object* v___y_417_, lean_object* v___y_418_, lean_object* v___y_419_, lean_object* v___y_420_){
_start:
{
lean_object* v_res_421_; 
v_res_421_ = lp_aesop_Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0(v_tacticState_414_, v_s_415_, v___y_416_, v___y_417_, v___y_418_, v___y_419_);
lean_dec(v___y_419_);
lean_dec_ref(v___y_418_);
lean_dec(v___y_417_);
lean_dec_ref(v___y_416_);
lean_dec_ref(v_s_415_);
return v_res_421_;
}
}
static lean_object* _init_lp_aesop_Aesop_Script_UScript_renderTacticSeq___closed__9(void){
_start:
{
lean_object* v___x_440_; 
v___x_440_ = l_Array_mkArray0(lean_box(0));
return v___x_440_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_UScript_renderTacticSeq(lean_object* v_uscript_442_, lean_object* v_preState_443_, lean_object* v_goal_444_, lean_object* v_a_445_, lean_object* v_a_446_, lean_object* v_a_447_, lean_object* v_a_448_){
_start:
{
lean_object* v___x_450_; lean_object* v___x_451_; 
v___x_450_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Script_TacticState_mkInitial___boxed), 6, 1);
lean_closure_set(v___x_450_, 0, v_goal_444_);
v___x_451_ = lp_batteries_Lean_Meta_SavedState_runMetaM_x27___redArg(v_preState_443_, v___x_450_, v_a_445_, v_a_446_, v_a_447_, v_a_448_);
if (lean_obj_tag(v___x_451_) == 0)
{
lean_object* v_a_452_; lean_object* v___x_453_; 
v_a_452_ = lean_ctor_get(v___x_451_, 0);
lean_inc(v_a_452_);
lean_dec_ref_known(v___x_451_, 1);
v___x_453_ = lp_aesop_Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0(v_a_452_, v_uscript_442_, v_a_445_, v_a_446_, v_a_447_, v_a_448_);
if (lean_obj_tag(v___x_453_) == 0)
{
lean_object* v_a_454_; lean_object* v___x_456_; uint8_t v_isShared_457_; uint8_t v_isSharedCheck_474_; 
v_a_454_ = lean_ctor_get(v___x_453_, 0);
v_isSharedCheck_474_ = !lean_is_exclusive(v___x_453_);
if (v_isSharedCheck_474_ == 0)
{
v___x_456_ = v___x_453_;
v_isShared_457_ = v_isSharedCheck_474_;
goto v_resetjp_455_;
}
else
{
lean_inc(v_a_454_);
lean_dec(v___x_453_);
v___x_456_ = lean_box(0);
v_isShared_457_ = v_isSharedCheck_474_;
goto v_resetjp_455_;
}
v_resetjp_455_:
{
lean_object* v_ref_458_; uint8_t v___x_459_; lean_object* v___x_460_; lean_object* v___x_461_; lean_object* v___x_462_; lean_object* v___x_463_; lean_object* v___x_464_; lean_object* v___x_465_; lean_object* v___x_466_; lean_object* v___x_467_; lean_object* v___x_468_; lean_object* v___x_469_; lean_object* v___x_470_; lean_object* v___x_472_; 
v_ref_458_ = lean_ctor_get(v_a_447_, 5);
v___x_459_ = 0;
v___x_460_ = l_Lean_SourceInfo_fromRef(v_ref_458_, v___x_459_);
v___x_461_ = ((lean_object*)(lp_aesop_Aesop_Script_UScript_renderTacticSeq___closed__4));
v___x_462_ = ((lean_object*)(lp_aesop_Aesop_Script_UScript_renderTacticSeq___closed__6));
v___x_463_ = ((lean_object*)(lp_aesop_Aesop_Script_UScript_renderTacticSeq___closed__8));
v___x_464_ = lean_obj_once(&lp_aesop_Aesop_Script_UScript_renderTacticSeq___closed__9, &lp_aesop_Aesop_Script_UScript_renderTacticSeq___closed__9_once, _init_lp_aesop_Aesop_Script_UScript_renderTacticSeq___closed__9);
v___x_465_ = ((lean_object*)(lp_aesop_Aesop_Script_UScript_renderTacticSeq___closed__10));
v___x_466_ = l_Lean_Syntax_SepArray_ofElems(v___x_465_, v_a_454_);
lean_dec(v_a_454_);
v___x_467_ = l_Array_append___redArg(v___x_464_, v___x_466_);
lean_dec_ref(v___x_466_);
lean_inc_n(v___x_460_, 2);
v___x_468_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_468_, 0, v___x_460_);
lean_ctor_set(v___x_468_, 1, v___x_463_);
lean_ctor_set(v___x_468_, 2, v___x_467_);
v___x_469_ = l_Lean_Syntax_node1(v___x_460_, v___x_462_, v___x_468_);
v___x_470_ = l_Lean_Syntax_node1(v___x_460_, v___x_461_, v___x_469_);
if (v_isShared_457_ == 0)
{
lean_ctor_set(v___x_456_, 0, v___x_470_);
v___x_472_ = v___x_456_;
goto v_reusejp_471_;
}
else
{
lean_object* v_reuseFailAlloc_473_; 
v_reuseFailAlloc_473_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_473_, 0, v___x_470_);
v___x_472_ = v_reuseFailAlloc_473_;
goto v_reusejp_471_;
}
v_reusejp_471_:
{
return v___x_472_;
}
}
}
else
{
lean_object* v_a_475_; lean_object* v___x_477_; uint8_t v_isShared_478_; uint8_t v_isSharedCheck_482_; 
v_a_475_ = lean_ctor_get(v___x_453_, 0);
v_isSharedCheck_482_ = !lean_is_exclusive(v___x_453_);
if (v_isSharedCheck_482_ == 0)
{
v___x_477_ = v___x_453_;
v_isShared_478_ = v_isSharedCheck_482_;
goto v_resetjp_476_;
}
else
{
lean_inc(v_a_475_);
lean_dec(v___x_453_);
v___x_477_ = lean_box(0);
v_isShared_478_ = v_isSharedCheck_482_;
goto v_resetjp_476_;
}
v_resetjp_476_:
{
lean_object* v___x_480_; 
if (v_isShared_478_ == 0)
{
v___x_480_ = v___x_477_;
goto v_reusejp_479_;
}
else
{
lean_object* v_reuseFailAlloc_481_; 
v_reuseFailAlloc_481_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_481_, 0, v_a_475_);
v___x_480_ = v_reuseFailAlloc_481_;
goto v_reusejp_479_;
}
v_reusejp_479_:
{
return v___x_480_;
}
}
}
}
else
{
lean_object* v_a_483_; lean_object* v___x_485_; uint8_t v_isShared_486_; uint8_t v_isSharedCheck_490_; 
v_a_483_ = lean_ctor_get(v___x_451_, 0);
v_isSharedCheck_490_ = !lean_is_exclusive(v___x_451_);
if (v_isSharedCheck_490_ == 0)
{
v___x_485_ = v___x_451_;
v_isShared_486_ = v_isSharedCheck_490_;
goto v_resetjp_484_;
}
else
{
lean_inc(v_a_483_);
lean_dec(v___x_451_);
v___x_485_ = lean_box(0);
v_isShared_486_ = v_isSharedCheck_490_;
goto v_resetjp_484_;
}
v_resetjp_484_:
{
lean_object* v___x_488_; 
if (v_isShared_486_ == 0)
{
v___x_488_ = v___x_485_;
goto v_reusejp_487_;
}
else
{
lean_object* v_reuseFailAlloc_489_; 
v_reuseFailAlloc_489_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_489_, 0, v_a_483_);
v___x_488_ = v_reuseFailAlloc_489_;
goto v_reusejp_487_;
}
v_reusejp_487_:
{
return v___x_488_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_UScript_renderTacticSeq___boxed(lean_object* v_uscript_491_, lean_object* v_preState_492_, lean_object* v_goal_493_, lean_object* v_a_494_, lean_object* v_a_495_, lean_object* v_a_496_, lean_object* v_a_497_, lean_object* v_a_498_){
_start:
{
lean_object* v_res_499_; 
v_res_499_ = lp_aesop_Aesop_Script_UScript_renderTacticSeq(v_uscript_491_, v_preState_492_, v_goal_493_, v_a_494_, v_a_495_, v_a_496_, v_a_497_);
lean_dec(v_a_497_);
lean_dec_ref(v_a_496_);
lean_dec(v_a_495_);
lean_dec_ref(v_a_494_);
lean_dec_ref(v_uscript_491_);
return v_res_499_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__1_spec__2(lean_object* v_00_u03b1_500_, lean_object* v_goal_501_, lean_object* v_pre_502_, lean_object* v___y_503_, lean_object* v___y_504_, lean_object* v___y_505_, lean_object* v___y_506_){
_start:
{
lean_object* v___x_508_; 
v___x_508_ = lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__1_spec__2___redArg(v_goal_501_, v_pre_502_, v___y_503_, v___y_504_, v___y_505_, v___y_506_);
return v___x_508_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__1_spec__2___boxed(lean_object* v_00_u03b1_509_, lean_object* v_goal_510_, lean_object* v_pre_511_, lean_object* v___y_512_, lean_object* v___y_513_, lean_object* v___y_514_, lean_object* v___y_515_, lean_object* v___y_516_){
_start:
{
lean_object* v_res_517_; 
v_res_517_ = lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__1_spec__2(v_00_u03b1_509_, v_goal_510_, v_pre_511_, v___y_512_, v___y_513_, v___y_514_, v___y_515_);
lean_dec(v___y_515_);
lean_dec_ref(v___y_514_);
lean_dec(v___y_513_);
lean_dec_ref(v___y_512_);
return v_res_517_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00__private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__1_spec__2_spec__4(lean_object* v_00_u03b1_518_, lean_object* v_msg_519_, lean_object* v___y_520_, lean_object* v___y_521_, lean_object* v___y_522_, lean_object* v___y_523_){
_start:
{
lean_object* v___x_525_; 
v___x_525_ = lp_aesop_Lean_throwError___at___00__private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__1_spec__2_spec__4___redArg(v_msg_519_, v___y_520_, v___y_521_, v___y_522_, v___y_523_);
return v___x_525_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00__private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__1_spec__2_spec__4___boxed(lean_object* v_00_u03b1_526_, lean_object* v_msg_527_, lean_object* v___y_528_, lean_object* v___y_529_, lean_object* v___y_530_, lean_object* v___y_531_, lean_object* v___y_532_){
_start:
{
lean_object* v_res_533_; 
v_res_533_ = lp_aesop_Lean_throwError___at___00__private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_Script_UScript_renderTacticSeq_spec__0_spec__0_spec__1_spec__2_spec__4(v_00_u03b1_526_, v_msg_527_, v___y_528_, v___y_529_, v___y_530_, v___y_531_);
lean_dec(v___y_531_);
lean_dec_ref(v___y_530_);
lean_dec(v___y_529_);
lean_dec_ref(v___y_528_);
return v_res_533_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Script_UScript_validate_spec__0(lean_object* v_as_534_, size_t v_i_535_, size_t v_stop_536_, lean_object* v_b_537_, lean_object* v___y_538_, lean_object* v___y_539_, lean_object* v___y_540_, lean_object* v___y_541_){
_start:
{
uint8_t v___x_543_; 
v___x_543_ = lean_usize_dec_eq(v_i_535_, v_stop_536_);
if (v___x_543_ == 0)
{
lean_object* v___x_544_; lean_object* v___x_545_; 
v___x_544_ = lean_array_uget_borrowed(v_as_534_, v_i_535_);
lean_inc(v___x_544_);
v___x_545_ = lp_aesop_Aesop_Script_Step_validate(v___x_544_, v___y_538_, v___y_539_, v___y_540_, v___y_541_);
if (lean_obj_tag(v___x_545_) == 0)
{
lean_object* v_a_546_; size_t v___x_547_; size_t v___x_548_; 
v_a_546_ = lean_ctor_get(v___x_545_, 0);
lean_inc(v_a_546_);
lean_dec_ref_known(v___x_545_, 1);
v___x_547_ = ((size_t)1ULL);
v___x_548_ = lean_usize_add(v_i_535_, v___x_547_);
v_i_535_ = v___x_548_;
v_b_537_ = v_a_546_;
goto _start;
}
else
{
return v___x_545_;
}
}
else
{
lean_object* v___x_550_; 
v___x_550_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_550_, 0, v_b_537_);
return v___x_550_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Script_UScript_validate_spec__0___boxed(lean_object* v_as_551_, lean_object* v_i_552_, lean_object* v_stop_553_, lean_object* v_b_554_, lean_object* v___y_555_, lean_object* v___y_556_, lean_object* v___y_557_, lean_object* v___y_558_, lean_object* v___y_559_){
_start:
{
size_t v_i_boxed_560_; size_t v_stop_boxed_561_; lean_object* v_res_562_; 
v_i_boxed_560_ = lean_unbox_usize(v_i_552_);
lean_dec(v_i_552_);
v_stop_boxed_561_ = lean_unbox_usize(v_stop_553_);
lean_dec(v_stop_553_);
v_res_562_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Script_UScript_validate_spec__0(v_as_551_, v_i_boxed_560_, v_stop_boxed_561_, v_b_554_, v___y_555_, v___y_556_, v___y_557_, v___y_558_);
lean_dec(v___y_558_);
lean_dec_ref(v___y_557_);
lean_dec(v___y_556_);
lean_dec_ref(v___y_555_);
lean_dec_ref(v_as_551_);
return v_res_562_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_UScript_validate(lean_object* v_s_563_, lean_object* v_a_564_, lean_object* v_a_565_, lean_object* v_a_566_, lean_object* v_a_567_){
_start:
{
lean_object* v___x_569_; lean_object* v___x_570_; lean_object* v___x_571_; uint8_t v___x_572_; 
v___x_569_ = lean_unsigned_to_nat(0u);
v___x_570_ = lean_array_get_size(v_s_563_);
v___x_571_ = lean_box(0);
v___x_572_ = lean_nat_dec_lt(v___x_569_, v___x_570_);
if (v___x_572_ == 0)
{
lean_object* v___x_573_; 
v___x_573_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_573_, 0, v___x_571_);
return v___x_573_;
}
else
{
uint8_t v___x_574_; 
v___x_574_ = lean_nat_dec_le(v___x_570_, v___x_570_);
if (v___x_574_ == 0)
{
if (v___x_572_ == 0)
{
lean_object* v___x_575_; 
v___x_575_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_575_, 0, v___x_571_);
return v___x_575_;
}
else
{
size_t v___x_576_; size_t v___x_577_; lean_object* v___x_578_; 
v___x_576_ = ((size_t)0ULL);
v___x_577_ = lean_usize_of_nat(v___x_570_);
v___x_578_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Script_UScript_validate_spec__0(v_s_563_, v___x_576_, v___x_577_, v___x_571_, v_a_564_, v_a_565_, v_a_566_, v_a_567_);
return v___x_578_;
}
}
else
{
size_t v___x_579_; size_t v___x_580_; lean_object* v___x_581_; 
v___x_579_ = ((size_t)0ULL);
v___x_580_ = lean_usize_of_nat(v___x_570_);
v___x_581_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Script_UScript_validate_spec__0(v_s_563_, v___x_579_, v___x_580_, v___x_571_, v_a_564_, v_a_565_, v_a_566_, v_a_567_);
return v___x_581_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_UScript_validate___boxed(lean_object* v_s_582_, lean_object* v_a_583_, lean_object* v_a_584_, lean_object* v_a_585_, lean_object* v_a_586_, lean_object* v_a_587_){
_start:
{
lean_object* v_res_588_; 
v_res_588_ = lp_aesop_Aesop_Script_UScript_validate(v_s_582_, v_a_583_, v_a_584_, v_a_585_, v_a_586_);
lean_dec(v_a_586_);
lean_dec_ref(v_a_585_);
lean_dec(v_a_584_);
lean_dec_ref(v_a_583_);
lean_dec_ref(v_s_582_);
return v_res_588_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Script_Step(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Lean_Meta_SavedState(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_Script_UScript(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Script_Step(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Lean_Meta_SavedState(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_Script_UScript(uint8_t builtin) {
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
lean_object* initialize_aesop_Aesop_Script_Step(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Lean_Meta_SavedState(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_Script_UScript(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Script_Step(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Lean_Meta_SavedState(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Script_UScript(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_Script_UScript(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_Script_UScript(builtin);
}
#ifdef __cplusplus
}
#endif
