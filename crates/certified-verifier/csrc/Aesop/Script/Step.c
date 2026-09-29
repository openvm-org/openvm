// Lean compiler output
// Module: Aesop.Script.Step
// Imports: public import Init public meta import Init public import Aesop.Script.Tactic public import Aesop.Script.TacticState public import Aesop.Tracing import Batteries.Tactic.PermuteGoals import Batteries.Lean.Meta.SavedState import Aesop.Script.Util import Aesop.Util.EqualUpToIds
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
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Id_instMonad___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_mkAtom(lean_object*);
lean_object* l_Lean_MessageData_ofSyntax(lean_object*);
double lean_float_of_nat(lean_object*);
lean_object* l_Lean_Meta_saveState___redArg(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_PersistentArray_toArray___redArg(lean_object*);
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* lean_st_ref_take(lean_object*);
lean_object* l_Lean_PersistentArray_push___redArg(lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
extern lean_object* l_Lean_trace_profiler;
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_PersistentArray_append___redArg(lean_object*, lean_object*);
double lean_float_sub(double, double);
uint8_t lean_float_decLt(double, double);
extern lean_object* l_Lean_trace_profiler_useHeartbeats;
extern lean_object* l_Lean_trace_profiler_threshold;
double lean_float_div(double, double);
extern lean_object* lp_aesop_Aesop_TraceOption_script;
lean_object* lean_array_get_size(lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* l_Array_toSubarray___redArg(lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_runTacticCapturingPostState___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_SavedState_restore___redArg(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
uint8_t l_Lean_Exception_isRuntime(lean_object*);
lean_object* lean_array_mk(lean_object*);
lean_object* lp_aesop_Aesop_Script_matchGoals(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_append(lean_object*, lean_object*);
uint8_t l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(lean_object*, lean_object*, lean_object*);
lean_object* lean_io_mono_nanos_now();
lean_object* lean_io_get_num_heartbeats();
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Std_DTreeMap_Internal_Impl_insert___at___00Lean_NameMap_insert_spec__0___redArg(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Name_isPrefixOf(lean_object*, lean_object*);
extern lean_object* l_Lean_diagnostics;
extern lean_object* l_Lean_maxRecDepth;
lean_object* l_Lean_Kernel_enableDiag(lean_object*, uint8_t);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
uint8_t l_Lean_Kernel_isDiagnosticsEnabled(lean_object*);
lean_object* lp_aesop_Aesop_GoalWithMVars_ofMVarId(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__6(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__2___boxed(lean_object*, lean_object*);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* l_Array_mkArray0(lean_object*);
lean_object* l_Lean_MVarId_admit___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_batteries_Lean_Meta_SavedState_runMetaM___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_Script_Tactic_unstructured(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* l_Nat_reprFast(lean_object*);
lean_object* l_Lean_Syntax_mkNumLit(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_tacticStatesEqualUpToIds(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* l_Lean_MessageData_joinSep(lean_object*, lean_object*);
lean_object* lp_batteries_Lean_Meta_SavedState_runMetaM_x27___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_indentD(lean_object*);
lean_object* l_Lean_Elab_Tactic_evalTactic___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_runTacticMCapturingPostState(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Exception_toMessageData(lean_object*);
lean_object* l_Lean_MessageData_ofName(lean_object*);
lean_object* l___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
lean_object* l_List_mapTR_loop___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofList(lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* lp_batteries_Lean_MetavarContext_eraseExprMVarAssignment(lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_Script_TacticState_applyTactic___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_Script_TacticState_getVisibleGoalIndex___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_mkOneBasedNumLit(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_mkOneBasedNumLit___boxed(lean_object*);
static const lean_string_object lp_aesop_Aesop_Script_mkOnGoal___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Batteries"};
static const lean_object* lp_aesop_Aesop_Script_mkOnGoal___closed__0 = (const lean_object*)&lp_aesop_Aesop_Script_mkOnGoal___closed__0_value;
static const lean_string_object lp_aesop_Aesop_Script_mkOnGoal___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_aesop_Aesop_Script_mkOnGoal___closed__1 = (const lean_object*)&lp_aesop_Aesop_Script_mkOnGoal___closed__1_value;
static const lean_string_object lp_aesop_Aesop_Script_mkOnGoal___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tacticOn_goal-_=>_"};
static const lean_object* lp_aesop_Aesop_Script_mkOnGoal___closed__2 = (const lean_object*)&lp_aesop_Aesop_Script_mkOnGoal___closed__2_value;
static const lean_ctor_object lp_aesop_Aesop_Script_mkOnGoal___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Script_mkOnGoal___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_aesop_Aesop_Script_mkOnGoal___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_mkOnGoal___closed__3_value_aux_0),((lean_object*)&lp_aesop_Aesop_Script_mkOnGoal___closed__1_value),LEAN_SCALAR_PTR_LITERAL(219, 16, 21, 14, 34, 244, 116, 98)}};
static const lean_ctor_object lp_aesop_Aesop_Script_mkOnGoal___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_mkOnGoal___closed__3_value_aux_1),((lean_object*)&lp_aesop_Aesop_Script_mkOnGoal___closed__2_value),LEAN_SCALAR_PTR_LITERAL(243, 56, 227, 189, 147, 207, 104, 76)}};
static const lean_object* lp_aesop_Aesop_Script_mkOnGoal___closed__3 = (const lean_object*)&lp_aesop_Aesop_Script_mkOnGoal___closed__3_value;
static const lean_string_object lp_aesop_Aesop_Script_mkOnGoal___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "on_goal"};
static const lean_object* lp_aesop_Aesop_Script_mkOnGoal___closed__4 = (const lean_object*)&lp_aesop_Aesop_Script_mkOnGoal___closed__4_value;
static const lean_string_object lp_aesop_Aesop_Script_mkOnGoal___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_aesop_Aesop_Script_mkOnGoal___closed__5 = (const lean_object*)&lp_aesop_Aesop_Script_mkOnGoal___closed__5_value;
static const lean_ctor_object lp_aesop_Aesop_Script_mkOnGoal___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Script_mkOnGoal___closed__5_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_aesop_Aesop_Script_mkOnGoal___closed__6 = (const lean_object*)&lp_aesop_Aesop_Script_mkOnGoal___closed__6_value;
static lean_once_cell_t lp_aesop_Aesop_Script_mkOnGoal___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Script_mkOnGoal___closed__7;
static const lean_string_object lp_aesop_Aesop_Script_mkOnGoal___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "=>"};
static const lean_object* lp_aesop_Aesop_Script_mkOnGoal___closed__8 = (const lean_object*)&lp_aesop_Aesop_Script_mkOnGoal___closed__8_value;
static const lean_string_object lp_aesop_Aesop_Script_mkOnGoal___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_aesop_Aesop_Script_mkOnGoal___closed__9 = (const lean_object*)&lp_aesop_Aesop_Script_mkOnGoal___closed__9_value;
static const lean_string_object lp_aesop_Aesop_Script_mkOnGoal___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_aesop_Aesop_Script_mkOnGoal___closed__10 = (const lean_object*)&lp_aesop_Aesop_Script_mkOnGoal___closed__10_value;
static const lean_string_object lp_aesop_Aesop_Script_mkOnGoal___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tacticSeq"};
static const lean_object* lp_aesop_Aesop_Script_mkOnGoal___closed__11 = (const lean_object*)&lp_aesop_Aesop_Script_mkOnGoal___closed__11_value;
static const lean_ctor_object lp_aesop_Aesop_Script_mkOnGoal___closed__12_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Script_mkOnGoal___closed__9_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop_Aesop_Script_mkOnGoal___closed__12_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_mkOnGoal___closed__12_value_aux_0),((lean_object*)&lp_aesop_Aesop_Script_mkOnGoal___closed__10_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop_Aesop_Script_mkOnGoal___closed__12_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_mkOnGoal___closed__12_value_aux_1),((lean_object*)&lp_aesop_Aesop_Script_mkOnGoal___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_aesop_Aesop_Script_mkOnGoal___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_mkOnGoal___closed__12_value_aux_2),((lean_object*)&lp_aesop_Aesop_Script_mkOnGoal___closed__11_value),LEAN_SCALAR_PTR_LITERAL(212, 140, 85, 215, 241, 69, 7, 118)}};
static const lean_object* lp_aesop_Aesop_Script_mkOnGoal___closed__12 = (const lean_object*)&lp_aesop_Aesop_Script_mkOnGoal___closed__12_value;
static const lean_string_object lp_aesop_Aesop_Script_mkOnGoal___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tacticSeq1Indented"};
static const lean_object* lp_aesop_Aesop_Script_mkOnGoal___closed__13 = (const lean_object*)&lp_aesop_Aesop_Script_mkOnGoal___closed__13_value;
static const lean_ctor_object lp_aesop_Aesop_Script_mkOnGoal___closed__14_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Script_mkOnGoal___closed__9_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop_Aesop_Script_mkOnGoal___closed__14_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_mkOnGoal___closed__14_value_aux_0),((lean_object*)&lp_aesop_Aesop_Script_mkOnGoal___closed__10_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop_Aesop_Script_mkOnGoal___closed__14_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_mkOnGoal___closed__14_value_aux_1),((lean_object*)&lp_aesop_Aesop_Script_mkOnGoal___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_aesop_Aesop_Script_mkOnGoal___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_mkOnGoal___closed__14_value_aux_2),((lean_object*)&lp_aesop_Aesop_Script_mkOnGoal___closed__13_value),LEAN_SCALAR_PTR_LITERAL(223, 90, 160, 238, 133, 180, 23, 239)}};
static const lean_object* lp_aesop_Aesop_Script_mkOnGoal___closed__14 = (const lean_object*)&lp_aesop_Aesop_Script_mkOnGoal___closed__14_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_mkOnGoal(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_mkOnGoal___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_applyStep___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_applyStep(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_Step_uTactic(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_Step_uTactic___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_Step_sTactic_x3f(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_Step_sTactic_x3f___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_Step_instToMessageData___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_Step_instToMessageData___lam__0___boxed(lean_object*);
static const lean_closure_object lp_aesop_Aesop_Script_Step_instToMessageData___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__0, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Script_Step_instToMessageData___lam__1___closed__0 = (const lean_object*)&lp_aesop_Aesop_Script_Step_instToMessageData___lam__1___closed__0_value;
static const lean_closure_object lp_aesop_Aesop_Script_Step_instToMessageData___lam__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__1___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Script_Step_instToMessageData___lam__1___closed__1 = (const lean_object*)&lp_aesop_Aesop_Script_Step_instToMessageData___lam__1___closed__1_value;
static const lean_closure_object lp_aesop_Aesop_Script_Step_instToMessageData___lam__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__2___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Script_Step_instToMessageData___lam__1___closed__2 = (const lean_object*)&lp_aesop_Aesop_Script_Step_instToMessageData___lam__1___closed__2_value;
static const lean_closure_object lp_aesop_Aesop_Script_Step_instToMessageData___lam__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__3, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Script_Step_instToMessageData___lam__1___closed__3 = (const lean_object*)&lp_aesop_Aesop_Script_Step_instToMessageData___lam__1___closed__3_value;
static const lean_closure_object lp_aesop_Aesop_Script_Step_instToMessageData___lam__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__4___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Script_Step_instToMessageData___lam__1___closed__4 = (const lean_object*)&lp_aesop_Aesop_Script_Step_instToMessageData___lam__1___closed__4_value;
static const lean_closure_object lp_aesop_Aesop_Script_Step_instToMessageData___lam__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__5___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Script_Step_instToMessageData___lam__1___closed__5 = (const lean_object*)&lp_aesop_Aesop_Script_Step_instToMessageData___lam__1___closed__5_value;
static const lean_closure_object lp_aesop_Aesop_Script_Step_instToMessageData___lam__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__6, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Script_Step_instToMessageData___lam__1___closed__6 = (const lean_object*)&lp_aesop_Aesop_Script_Step_instToMessageData___lam__1___closed__6_value;
static const lean_ctor_object lp_aesop_Aesop_Script_Step_instToMessageData___lam__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_Step_instToMessageData___lam__1___closed__0_value),((lean_object*)&lp_aesop_Aesop_Script_Step_instToMessageData___lam__1___closed__1_value)}};
static const lean_object* lp_aesop_Aesop_Script_Step_instToMessageData___lam__1___closed__7 = (const lean_object*)&lp_aesop_Aesop_Script_Step_instToMessageData___lam__1___closed__7_value;
static const lean_ctor_object lp_aesop_Aesop_Script_Step_instToMessageData___lam__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*5 + 0, .m_other = 5, .m_tag = 0}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_Step_instToMessageData___lam__1___closed__7_value),((lean_object*)&lp_aesop_Aesop_Script_Step_instToMessageData___lam__1___closed__2_value),((lean_object*)&lp_aesop_Aesop_Script_Step_instToMessageData___lam__1___closed__3_value),((lean_object*)&lp_aesop_Aesop_Script_Step_instToMessageData___lam__1___closed__4_value),((lean_object*)&lp_aesop_Aesop_Script_Step_instToMessageData___lam__1___closed__5_value)}};
static const lean_object* lp_aesop_Aesop_Script_Step_instToMessageData___lam__1___closed__8 = (const lean_object*)&lp_aesop_Aesop_Script_Step_instToMessageData___lam__1___closed__8_value;
static const lean_ctor_object lp_aesop_Aesop_Script_Step_instToMessageData___lam__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_Step_instToMessageData___lam__1___closed__8_value),((lean_object*)&lp_aesop_Aesop_Script_Step_instToMessageData___lam__1___closed__6_value)}};
static const lean_object* lp_aesop_Aesop_Script_Step_instToMessageData___lam__1___closed__9 = (const lean_object*)&lp_aesop_Aesop_Script_Step_instToMessageData___lam__1___closed__9_value;
static const lean_string_object lp_aesop_Aesop_Script_Step_instToMessageData___lam__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 3, .m_data = " → "};
static const lean_object* lp_aesop_Aesop_Script_Step_instToMessageData___lam__1___closed__10 = (const lean_object*)&lp_aesop_Aesop_Script_Step_instToMessageData___lam__1___closed__10_value;
static lean_once_cell_t lp_aesop_Aesop_Script_Step_instToMessageData___lam__1___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Script_Step_instToMessageData___lam__1___closed__11;
static const lean_closure_object lp_aesop_Aesop_Script_Step_instToMessageData___lam__1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_MessageData_ofName, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Script_Step_instToMessageData___lam__1___closed__12 = (const lean_object*)&lp_aesop_Aesop_Script_Step_instToMessageData___lam__1___closed__12_value;
static const lean_string_object lp_aesop_Aesop_Script_Step_instToMessageData___lam__1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ":"};
static const lean_object* lp_aesop_Aesop_Script_Step_instToMessageData___lam__1___closed__13 = (const lean_object*)&lp_aesop_Aesop_Script_Step_instToMessageData___lam__1___closed__13_value;
static lean_once_cell_t lp_aesop_Aesop_Script_Step_instToMessageData___lam__1___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Script_Step_instToMessageData___lam__1___closed__14;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_Step_instToMessageData___lam__1(lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_Script_Step_instToMessageData___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_Script_Step_instToMessageData___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Script_Step_instToMessageData___closed__0 = (const lean_object*)&lp_aesop_Aesop_Script_Step_instToMessageData___closed__0_value;
static const lean_closure_object lp_aesop_Aesop_Script_Step_instToMessageData___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_Script_Step_instToMessageData___lam__1, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_Step_instToMessageData___closed__0_value)} };
static const lean_object* lp_aesop_Aesop_Script_Step_instToMessageData___closed__1 = (const lean_object*)&lp_aesop_Aesop_Script_Step_instToMessageData___closed__1_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_Script_Step_instToMessageData = (const lean_object*)&lp_aesop_Aesop_Script_Step_instToMessageData___closed__1_value;
static const lean_string_object lp_aesop_Aesop_Script_Step_mkSorry___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "tacticSorry"};
static const lean_object* lp_aesop_Aesop_Script_Step_mkSorry___closed__0 = (const lean_object*)&lp_aesop_Aesop_Script_Step_mkSorry___closed__0_value;
static const lean_ctor_object lp_aesop_Aesop_Script_Step_mkSorry___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Script_mkOnGoal___closed__9_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop_Aesop_Script_Step_mkSorry___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_Step_mkSorry___closed__1_value_aux_0),((lean_object*)&lp_aesop_Aesop_Script_mkOnGoal___closed__10_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop_Aesop_Script_Step_mkSorry___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_Step_mkSorry___closed__1_value_aux_1),((lean_object*)&lp_aesop_Aesop_Script_mkOnGoal___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_aesop_Aesop_Script_Step_mkSorry___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_Step_mkSorry___closed__1_value_aux_2),((lean_object*)&lp_aesop_Aesop_Script_Step_mkSorry___closed__0_value),LEAN_SCALAR_PTR_LITERAL(254, 186, 126, 140, 105, 148, 113, 102)}};
static const lean_object* lp_aesop_Aesop_Script_Step_mkSorry___closed__1 = (const lean_object*)&lp_aesop_Aesop_Script_Step_mkSorry___closed__1_value;
static const lean_string_object lp_aesop_Aesop_Script_Step_mkSorry___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "sorry"};
static const lean_object* lp_aesop_Aesop_Script_Step_mkSorry___closed__2 = (const lean_object*)&lp_aesop_Aesop_Script_Step_mkSorry___closed__2_value;
static const lean_array_object lp_aesop_Aesop_Script_Step_mkSorry___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Aesop_Script_Step_mkSorry___closed__3 = (const lean_object*)&lp_aesop_Aesop_Script_Step_mkSorry___closed__3_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_Step_mkSorry(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_Step_mkSorry___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_Step_render___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_Step_render___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_Step_render___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_Step_render___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_Step_render(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00__private_Aesop_Script_Step_0__Aesop_Script_Step_validate_fmtGoals_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00__private_Aesop_Script_Step_0__Aesop_Script_Step_validate_fmtGoals_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_Script_Step_0__Aesop_Script_Step_validate_fmtGoals_spec__0___redArg(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_Script_Step_0__Aesop_Script_Step_validate_fmtGoals_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop___private_Aesop_Script_Step_0__Aesop_Script_Step_validate_fmtGoals___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "\n"};
static const lean_object* lp_aesop___private_Aesop_Script_Step_0__Aesop_Script_Step_validate_fmtGoals___lam__0___closed__0 = (const lean_object*)&lp_aesop___private_Aesop_Script_Step_0__Aesop_Script_Step_validate_fmtGoals___lam__0___closed__0_value;
static const lean_ctor_object lp_aesop___private_Aesop_Script_Step_0__Aesop_Script_Step_validate_fmtGoals___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Script_Step_0__Aesop_Script_Step_validate_fmtGoals___lam__0___closed__0_value)}};
static const lean_object* lp_aesop___private_Aesop_Script_Step_0__Aesop_Script_Step_validate_fmtGoals___lam__0___closed__1 = (const lean_object*)&lp_aesop___private_Aesop_Script_Step_0__Aesop_Script_Step_validate_fmtGoals___lam__0___closed__1_value;
static lean_once_cell_t lp_aesop___private_Aesop_Script_Step_0__Aesop_Script_Step_validate_fmtGoals___lam__0___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Script_Step_0__Aesop_Script_Step_validate_fmtGoals___lam__0___closed__2;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_Step_0__Aesop_Script_Step_validate_fmtGoals___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_Step_0__Aesop_Script_Step_validate_fmtGoals___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_Step_0__Aesop_Script_Step_validate_fmtGoals(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_Step_0__Aesop_Script_Step_validate_fmtGoals___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_Script_Step_0__Aesop_Script_Step_validate_fmtGoals_spec__0(size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_Script_Step_0__Aesop_Script_Step_validate_fmtGoals_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_Script_Step_validate_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_Script_Step_validate_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_Step_validate_spec__0(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_Step_validate_spec__0___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_Script_Step_validate___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "tactic"};
static const lean_object* lp_aesop_Aesop_Script_Step_validate___closed__0 = (const lean_object*)&lp_aesop_Aesop_Script_Step_validate___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_Script_Step_validate___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Script_Step_validate___closed__1;
static const lean_string_object lp_aesop_Aesop_Script_Step_validate___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 62, .m_capacity = 62, .m_length = 61, .m_data = "\nsucceeded but did not generate expected state. Initial goal:"};
static const lean_object* lp_aesop_Aesop_Script_Step_validate___closed__2 = (const lean_object*)&lp_aesop_Aesop_Script_Step_validate___closed__2_value;
static lean_once_cell_t lp_aesop_Aesop_Script_Step_validate___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Script_Step_validate___closed__3;
static const lean_string_object lp_aesop_Aesop_Script_Step_validate___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "\nExpected goals:"};
static const lean_object* lp_aesop_Aesop_Script_Step_validate___closed__4 = (const lean_object*)&lp_aesop_Aesop_Script_Step_validate___closed__4_value;
static lean_once_cell_t lp_aesop_Aesop_Script_Step_validate___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Script_Step_validate___closed__5;
static const lean_string_object lp_aesop_Aesop_Script_Step_validate___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "\nActual goals:"};
static const lean_object* lp_aesop_Aesop_Script_Step_validate___closed__6 = (const lean_object*)&lp_aesop_Aesop_Script_Step_validate___closed__6_value;
static lean_once_cell_t lp_aesop_Aesop_Script_Step_validate___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Script_Step_validate___closed__7;
static const lean_string_object lp_aesop_Aesop_Script_Step_validate___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "\nfailed with error"};
static const lean_object* lp_aesop_Aesop_Script_Step_validate___closed__8 = (const lean_object*)&lp_aesop_Aesop_Script_Step_validate___closed__8_value;
static lean_once_cell_t lp_aesop_Aesop_Script_Step_validate___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Script_Step_validate___closed__9;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_Step_validate(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_Step_validate___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_Script_Step_validate_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_Script_Step_validate_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__0 = (const lean_object*)&lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__0_value;
static const lean_string_object lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "simp"};
static const lean_object* lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__1 = (const lean_object*)&lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__1_value;
static const lean_ctor_object lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Script_mkOnGoal___closed__9_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__2_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__2_value_aux_0),((lean_object*)&lp_aesop_Aesop_Script_mkOnGoal___closed__10_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__2_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__2_value_aux_1),((lean_object*)&lp_aesop_Aesop_Script_mkOnGoal___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__2_value_aux_2),((lean_object*)&lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__1_value),LEAN_SCALAR_PTR_LITERAL(50, 13, 241, 145, 67, 153, 105, 177)}};
static const lean_object* lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__2 = (const lean_object*)&lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__2_value;
static lean_once_cell_t lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__3;
static lean_once_cell_t lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__4;
static const lean_string_object lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "optConfig"};
static const lean_object* lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__5 = (const lean_object*)&lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__5_value;
static const lean_ctor_object lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Script_mkOnGoal___closed__9_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__6_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__6_value_aux_0),((lean_object*)&lp_aesop_Aesop_Script_mkOnGoal___closed__10_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__6_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__6_value_aux_1),((lean_object*)&lp_aesop_Aesop_Script_mkOnGoal___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__6_value_aux_2),((lean_object*)&lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__5_value),LEAN_SCALAR_PTR_LITERAL(137, 208, 10, 74, 108, 50, 106, 48)}};
static const lean_object* lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__6 = (const lean_object*)&lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__6_value;
static const lean_ctor_object lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(2) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Script_mkOnGoal___closed__6_value),((lean_object*)&lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__0_value)}};
static const lean_object* lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__7 = (const lean_object*)&lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__7_value;
static lean_once_cell_t lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__8;
static lean_once_cell_t lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__9;
static lean_once_cell_t lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__10;
static lean_once_cell_t lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__11;
static lean_once_cell_t lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__12;
static lean_once_cell_t lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__13;
static lean_once_cell_t lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__14;
static lean_once_cell_t lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__15;
static lean_once_cell_t lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__16;
static lean_once_cell_t lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__17;
static lean_once_cell_t lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__18_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__18;
static lean_once_cell_t lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__19;
static lean_once_cell_t lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__20_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__20;
static lean_once_cell_t lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__21_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__21;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam;
LEAN_EXPORT lean_object* lp_aesop_Lean_observing_x3f___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_observing_x3f___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_observing_x3f___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_observing_x3f___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__1___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__1___redArg___closed__0;
static lean_once_cell_t lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__1___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__1___redArg___closed__1;
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__1___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__1___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_Option_get___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__2___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__3_spec__6(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__3_spec__6___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__3_spec__5(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__3_spec__5___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__3_spec__3_spec__4(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__3_spec__3_spec__4___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__3_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__3_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__3_spec__4___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__3_spec__4___redArg___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__3___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__3___closed__0;
static const lean_string_object lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__3___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 54, .m_capacity = 54, .m_length = 53, .m_data = "<exception thrown while producing trace node message>"};
static const lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__3___closed__1 = (const lean_object*)&lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__3___closed__1_value;
static lean_once_cell_t lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__3___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__3___closed__2;
static lean_once_cell_t lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__3___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__3___closed__3;
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__3(lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop___private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_aesop___private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder___closed__0 = (const lean_object*)&lp_aesop___private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder___closed__0_value;
static const lean_string_object lp_aesop___private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_aesop___private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder___closed__1 = (const lean_object*)&lp_aesop___private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder___closed__1_value;
static const lean_ctor_object lp_aesop___private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder___closed__1_value),LEAN_SCALAR_PTR_LITERAL(212, 145, 141, 177, 67, 149, 127, 197)}};
static const lean_object* lp_aesop___private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder___closed__2 = (const lean_object*)&lp_aesop___private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder___closed__2_value;
static lean_once_cell_t lp_aesop___private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_aesop___private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder___closed__3;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__3_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__3_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_aesop_WellFounded_opaqueFix_u2083___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__0___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_aesop_WellFounded_opaqueFix_u2083___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__0___redArg___closed__0 = (const lean_object*)&lp_aesop_WellFounded_opaqueFix_u2083___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__0___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_WellFounded_opaqueFix_u2083___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_WellFounded_opaqueFix_u2083___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_aesop_Lean_addTrace___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Lean_addTrace___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__2___closed__0 = (const lean_object*)&lp_aesop_Lean_addTrace___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__2___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Lean_addTrace___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addTrace___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "fallback: "};
static const lean_object* lp_aesop_Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder___lam__0___closed__0 = (const lean_object*)&lp_aesop_Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder___lam__0___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder___lam__0___closed__1;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__4_spec__5(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__4_spec__5___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__4(lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Options_set___at___00Aesop_withPPAnalyze___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__3_spec__3(lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Lean_Options_set___at___00Aesop_withPPAnalyze___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__3_spec__3___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_withPPAnalyze___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__3___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "pp"};
static const lean_object* lp_aesop_Aesop_withPPAnalyze___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__3___redArg___closed__0 = (const lean_object*)&lp_aesop_Aesop_withPPAnalyze___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__3___redArg___closed__0_value;
static const lean_string_object lp_aesop_Aesop_withPPAnalyze___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__3___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "analyze"};
static const lean_object* lp_aesop_Aesop_withPPAnalyze___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__3___redArg___closed__1 = (const lean_object*)&lp_aesop_Aesop_withPPAnalyze___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__3___redArg___closed__1_value;
static const lean_ctor_object lp_aesop_Aesop_withPPAnalyze___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__3___redArg___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_withPPAnalyze___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__3___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(249, 51, 192, 169, 230, 180, 160, 93)}};
static const lean_ctor_object lp_aesop_Aesop_withPPAnalyze___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__3___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_withPPAnalyze___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__3___redArg___closed__2_value_aux_0),((lean_object*)&lp_aesop_Aesop_withPPAnalyze___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__3___redArg___closed__1_value),LEAN_SCALAR_PTR_LITERAL(219, 114, 68, 229, 251, 70, 44, 204)}};
static const lean_object* lp_aesop_Aesop_withPPAnalyze___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__3___redArg___closed__2 = (const lean_object*)&lp_aesop_Aesop_withPPAnalyze___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__3___redArg___closed__2_value;
static const lean_string_object lp_aesop_Aesop_withPPAnalyze___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__3___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "proofs"};
static const lean_object* lp_aesop_Aesop_withPPAnalyze___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__3___redArg___closed__3 = (const lean_object*)&lp_aesop_Aesop_withPPAnalyze___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__3___redArg___closed__3_value;
static const lean_ctor_object lp_aesop_Aesop_withPPAnalyze___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__3___redArg___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_withPPAnalyze___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__3___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(249, 51, 192, 169, 230, 180, 160, 93)}};
static const lean_ctor_object lp_aesop_Aesop_withPPAnalyze___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__3___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_withPPAnalyze___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__3___redArg___closed__4_value_aux_0),((lean_object*)&lp_aesop_Aesop_withPPAnalyze___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__3___redArg___closed__3_value),LEAN_SCALAR_PTR_LITERAL(16, 182, 23, 133, 244, 85, 246, 31)}};
static const lean_object* lp_aesop_Aesop_withPPAnalyze___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__3___redArg___closed__4 = (const lean_object*)&lp_aesop_Aesop_withPPAnalyze___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__3___redArg___closed__4_value;
static lean_once_cell_t lp_aesop_Aesop_withPPAnalyze___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__3___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_withPPAnalyze___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__3___redArg___closed__5;
static lean_once_cell_t lp_aesop_Aesop_withPPAnalyze___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__3___redArg___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_withPPAnalyze___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__3___redArg___closed__6;
static lean_once_cell_t lp_aesop_Aesop_withPPAnalyze___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__3___redArg___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_withPPAnalyze___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__3___redArg___closed__7;
LEAN_EXPORT lean_object* lp_aesop_Aesop_withPPAnalyze___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__3___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_withPPAnalyze___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 29, .m_capacity = 29, .m_length = 28, .m_data = "converting lazy step to step"};
static const lean_object* lp_aesop_Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder___closed__0 = (const lean_object*)&lp_aesop_Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder___closed__1;
static lean_once_cell_t lp_aesop_Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder___closed__2;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_WellFounded_opaqueFix_u2083___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_WellFounded_opaqueFix_u2083___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_withPPAnalyze___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_withPPAnalyze___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_LazyStep_toStep_spec__0(size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_LazyStep_toStep_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_LazyStep_toStep___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_LazyStep_toStep___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_LazyStep_toStep(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_LazyStep_toStep___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_LazyStep_build___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_LazyStep_build___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_LazyStep_build(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_LazyStep_build___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Script_LazyStep_erasePostStateAssignments_spec__0(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Script_LazyStep_erasePostStateAssignments_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_LazyStep_erasePostStateAssignments(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_LazyStep_erasePostStateAssignments___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_mkOneBasedNumLit(lean_object* v_n_1_){
_start:
{
lean_object* v___x_2_; lean_object* v___x_3_; lean_object* v___x_4_; lean_object* v___x_5_; lean_object* v___x_6_; 
v___x_2_ = lean_unsigned_to_nat(1u);
v___x_3_ = lean_nat_add(v_n_1_, v___x_2_);
v___x_4_ = l_Nat_reprFast(v___x_3_);
v___x_5_ = lean_box(2);
v___x_6_ = l_Lean_Syntax_mkNumLit(v___x_4_, v___x_5_);
return v___x_6_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_mkOneBasedNumLit___boxed(lean_object* v_n_7_){
_start:
{
lean_object* v_res_8_; 
v_res_8_ = lp_aesop_Aesop_Script_mkOneBasedNumLit(v_n_7_);
lean_dec(v_n_7_);
return v_res_8_;
}
}
static lean_object* _init_lp_aesop_Aesop_Script_mkOnGoal___closed__7(void){
_start:
{
lean_object* v___x_20_; 
v___x_20_ = l_Array_mkArray0(lean_box(0));
return v___x_20_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_mkOnGoal(lean_object* v_goalPos_36_, lean_object* v_tac_37_){
_start:
{
lean_object* v___x_38_; uint8_t v___x_39_; 
v___x_38_ = lean_unsigned_to_nat(0u);
v___x_39_ = lean_nat_dec_eq(v_goalPos_36_, v___x_38_);
if (v___x_39_ == 0)
{
lean_object* v___x_40_; lean_object* v___x_41_; lean_object* v___x_42_; lean_object* v___x_43_; lean_object* v_posLit_44_; lean_object* v___x_45_; lean_object* v___x_46_; lean_object* v___x_47_; lean_object* v___x_48_; lean_object* v___x_49_; lean_object* v___x_50_; lean_object* v___x_51_; lean_object* v___x_52_; lean_object* v___x_53_; lean_object* v___x_54_; lean_object* v___x_55_; lean_object* v___x_56_; lean_object* v___x_57_; lean_object* v___x_58_; lean_object* v___x_59_; lean_object* v___x_60_; 
v___x_40_ = lean_unsigned_to_nat(1u);
v___x_41_ = lean_nat_add(v_goalPos_36_, v___x_40_);
v___x_42_ = l_Nat_reprFast(v___x_41_);
v___x_43_ = lean_box(2);
v_posLit_44_ = l_Lean_Syntax_mkNumLit(v___x_42_, v___x_43_);
v___x_45_ = lean_box(0);
v___x_46_ = l_Lean_SourceInfo_fromRef(v___x_45_, v___x_39_);
v___x_47_ = ((lean_object*)(lp_aesop_Aesop_Script_mkOnGoal___closed__3));
v___x_48_ = ((lean_object*)(lp_aesop_Aesop_Script_mkOnGoal___closed__4));
lean_inc_n(v___x_46_, 6);
v___x_49_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_49_, 0, v___x_46_);
lean_ctor_set(v___x_49_, 1, v___x_48_);
v___x_50_ = ((lean_object*)(lp_aesop_Aesop_Script_mkOnGoal___closed__6));
v___x_51_ = lean_obj_once(&lp_aesop_Aesop_Script_mkOnGoal___closed__7, &lp_aesop_Aesop_Script_mkOnGoal___closed__7_once, _init_lp_aesop_Aesop_Script_mkOnGoal___closed__7);
v___x_52_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_52_, 0, v___x_46_);
lean_ctor_set(v___x_52_, 1, v___x_50_);
lean_ctor_set(v___x_52_, 2, v___x_51_);
v___x_53_ = ((lean_object*)(lp_aesop_Aesop_Script_mkOnGoal___closed__8));
v___x_54_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_54_, 0, v___x_46_);
lean_ctor_set(v___x_54_, 1, v___x_53_);
v___x_55_ = ((lean_object*)(lp_aesop_Aesop_Script_mkOnGoal___closed__12));
v___x_56_ = ((lean_object*)(lp_aesop_Aesop_Script_mkOnGoal___closed__14));
v___x_57_ = l_Lean_Syntax_node1(v___x_46_, v___x_50_, v_tac_37_);
v___x_58_ = l_Lean_Syntax_node1(v___x_46_, v___x_56_, v___x_57_);
v___x_59_ = l_Lean_Syntax_node1(v___x_46_, v___x_55_, v___x_58_);
v___x_60_ = l_Lean_Syntax_node5(v___x_46_, v___x_47_, v___x_49_, v___x_52_, v_posLit_44_, v___x_54_, v___x_59_);
return v___x_60_;
}
else
{
return v_tac_37_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_mkOnGoal___boxed(lean_object* v_goalPos_61_, lean_object* v_tac_62_){
_start:
{
lean_object* v_res_63_; 
v_res_63_ = lp_aesop_Aesop_Script_mkOnGoal(v_goalPos_61_, v_tac_62_);
lean_dec(v_goalPos_61_);
return v_res_63_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_applyStep___redArg(lean_object* v_inst_64_, lean_object* v_inst_65_, lean_object* v_tacticState_66_, lean_object* v_step_67_){
_start:
{
lean_object* v_preState_68_; lean_object* v_meta_69_; lean_object* v_postState_70_; lean_object* v_meta_71_; lean_object* v_preGoal_72_; lean_object* v_postGoals_73_; lean_object* v_mctx_74_; lean_object* v_mctx_75_; lean_object* v___x_76_; 
v_preState_68_ = lean_ctor_get(v_step_67_, 0);
v_meta_69_ = lean_ctor_get(v_preState_68_, 1);
lean_inc_ref(v_meta_69_);
v_postState_70_ = lean_ctor_get(v_step_67_, 3);
v_meta_71_ = lean_ctor_get(v_postState_70_, 1);
lean_inc_ref(v_meta_71_);
v_preGoal_72_ = lean_ctor_get(v_step_67_, 1);
lean_inc(v_preGoal_72_);
v_postGoals_73_ = lean_ctor_get(v_step_67_, 4);
lean_inc_ref(v_postGoals_73_);
lean_dec_ref(v_step_67_);
v_mctx_74_ = lean_ctor_get(v_meta_69_, 0);
lean_inc_ref(v_mctx_74_);
lean_dec_ref(v_meta_69_);
v_mctx_75_ = lean_ctor_get(v_meta_71_, 0);
lean_inc_ref(v_mctx_75_);
lean_dec_ref(v_meta_71_);
v___x_76_ = lp_aesop_Aesop_Script_TacticState_applyTactic___redArg(v_inst_64_, v_inst_65_, v_tacticState_66_, v_preGoal_72_, v_postGoals_73_, v_mctx_74_, v_mctx_75_);
lean_dec_ref(v_mctx_75_);
lean_dec_ref(v_mctx_74_);
return v___x_76_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_applyStep(lean_object* v_m_77_, lean_object* v_inst_78_, lean_object* v_inst_79_, lean_object* v_tacticState_80_, lean_object* v_step_81_){
_start:
{
lean_object* v___x_82_; 
v___x_82_ = lp_aesop_Aesop_Script_TacticState_applyStep___redArg(v_inst_78_, v_inst_79_, v_tacticState_80_, v_step_81_);
return v___x_82_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_Step_uTactic(lean_object* v_s_83_){
_start:
{
lean_object* v_tactic_84_; lean_object* v_uTactic_85_; 
v_tactic_84_ = lean_ctor_get(v_s_83_, 2);
v_uTactic_85_ = lean_ctor_get(v_tactic_84_, 0);
lean_inc(v_uTactic_85_);
return v_uTactic_85_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_Step_uTactic___boxed(lean_object* v_s_86_){
_start:
{
lean_object* v_res_87_; 
v_res_87_ = lp_aesop_Aesop_Script_Step_uTactic(v_s_86_);
lean_dec_ref(v_s_86_);
return v_res_87_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_Step_sTactic_x3f(lean_object* v_s_88_){
_start:
{
lean_object* v_tactic_89_; lean_object* v_sTactic_x3f_90_; 
v_tactic_89_ = lean_ctor_get(v_s_88_, 2);
v_sTactic_x3f_90_ = lean_ctor_get(v_tactic_89_, 1);
lean_inc(v_sTactic_x3f_90_);
return v_sTactic_x3f_90_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_Step_sTactic_x3f___boxed(lean_object* v_s_91_){
_start:
{
lean_object* v_res_92_; 
v_res_92_ = lp_aesop_Aesop_Script_Step_sTactic_x3f(v_s_91_);
lean_dec_ref(v_s_91_);
return v_res_92_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_Step_instToMessageData___lam__0(lean_object* v_x_93_){
_start:
{
lean_object* v_goal_94_; 
v_goal_94_ = lean_ctor_get(v_x_93_, 0);
lean_inc(v_goal_94_);
return v_goal_94_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_Step_instToMessageData___lam__0___boxed(lean_object* v_x_95_){
_start:
{
lean_object* v_res_96_; 
v_res_96_ = lp_aesop_Aesop_Script_Step_instToMessageData___lam__0(v_x_95_);
lean_dec_ref(v_x_95_);
return v_res_96_;
}
}
static lean_object* _init_lp_aesop_Aesop_Script_Step_instToMessageData___lam__1___closed__11(void){
_start:
{
lean_object* v___x_117_; lean_object* v___x_118_; 
v___x_117_ = ((lean_object*)(lp_aesop_Aesop_Script_Step_instToMessageData___lam__1___closed__10));
v___x_118_ = l_Lean_stringToMessageData(v___x_117_);
return v___x_118_;
}
}
static lean_object* _init_lp_aesop_Aesop_Script_Step_instToMessageData___lam__1___closed__14(void){
_start:
{
lean_object* v___x_121_; lean_object* v___x_122_; 
v___x_121_ = ((lean_object*)(lp_aesop_Aesop_Script_Step_instToMessageData___lam__1___closed__13));
v___x_122_ = l_Lean_stringToMessageData(v___x_121_);
return v___x_122_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_Step_instToMessageData___lam__1(lean_object* v___f_123_, lean_object* v_step_124_){
_start:
{
lean_object* v_preGoal_125_; lean_object* v_tactic_126_; lean_object* v_postGoals_127_; lean_object* v___x_128_; lean_object* v_uTactic_129_; lean_object* v___x_131_; uint8_t v_isShared_132_; uint8_t v_isSharedCheck_152_; 
v_preGoal_125_ = lean_ctor_get(v_step_124_, 1);
lean_inc(v_preGoal_125_);
v_tactic_126_ = lean_ctor_get(v_step_124_, 2);
lean_inc_ref(v_tactic_126_);
v_postGoals_127_ = lean_ctor_get(v_step_124_, 4);
lean_inc_ref(v_postGoals_127_);
lean_dec_ref(v_step_124_);
v___x_128_ = ((lean_object*)(lp_aesop_Aesop_Script_Step_instToMessageData___lam__1___closed__9));
v_uTactic_129_ = lean_ctor_get(v_tactic_126_, 0);
v_isSharedCheck_152_ = !lean_is_exclusive(v_tactic_126_);
if (v_isSharedCheck_152_ == 0)
{
lean_object* v_unused_153_; 
v_unused_153_ = lean_ctor_get(v_tactic_126_, 1);
lean_dec(v_unused_153_);
v___x_131_ = v_tactic_126_;
v_isShared_132_ = v_isSharedCheck_152_;
goto v_resetjp_130_;
}
else
{
lean_inc(v_uTactic_129_);
lean_dec(v_tactic_126_);
v___x_131_ = lean_box(0);
v_isShared_132_ = v_isSharedCheck_152_;
goto v_resetjp_130_;
}
v_resetjp_130_:
{
size_t v_sz_133_; lean_object* v___x_134_; lean_object* v___x_135_; lean_object* v___x_137_; 
v_sz_133_ = lean_array_size(v_postGoals_127_);
v___x_134_ = l_Lean_MessageData_ofName(v_preGoal_125_);
v___x_135_ = lean_obj_once(&lp_aesop_Aesop_Script_Step_instToMessageData___lam__1___closed__11, &lp_aesop_Aesop_Script_Step_instToMessageData___lam__1___closed__11_once, _init_lp_aesop_Aesop_Script_Step_instToMessageData___lam__1___closed__11);
if (v_isShared_132_ == 0)
{
lean_ctor_set_tag(v___x_131_, 7);
lean_ctor_set(v___x_131_, 1, v___x_135_);
lean_ctor_set(v___x_131_, 0, v___x_134_);
v___x_137_ = v___x_131_;
goto v_reusejp_136_;
}
else
{
lean_object* v_reuseFailAlloc_151_; 
v_reuseFailAlloc_151_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_151_, 0, v___x_134_);
lean_ctor_set(v_reuseFailAlloc_151_, 1, v___x_135_);
v___x_137_ = v_reuseFailAlloc_151_;
goto v_reusejp_136_;
}
v_reusejp_136_:
{
size_t v___x_138_; lean_object* v___x_139_; lean_object* v___x_140_; lean_object* v___x_141_; lean_object* v___x_142_; lean_object* v___x_143_; lean_object* v___x_144_; lean_object* v___x_145_; lean_object* v___x_146_; lean_object* v___x_147_; lean_object* v___x_148_; lean_object* v___x_149_; lean_object* v___x_150_; 
v___x_138_ = ((size_t)0ULL);
v___x_139_ = l___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map(lean_box(0), lean_box(0), lean_box(0), v___x_128_, v___f_123_, v_sz_133_, v___x_138_, v_postGoals_127_);
v___x_140_ = lean_array_to_list(v___x_139_);
v___x_141_ = ((lean_object*)(lp_aesop_Aesop_Script_Step_instToMessageData___lam__1___closed__12));
v___x_142_ = lean_box(0);
v___x_143_ = l_List_mapTR_loop___redArg(v___x_141_, v___x_140_, v___x_142_);
v___x_144_ = l_Lean_MessageData_ofList(v___x_143_);
v___x_145_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_145_, 0, v___x_137_);
lean_ctor_set(v___x_145_, 1, v___x_144_);
v___x_146_ = lean_obj_once(&lp_aesop_Aesop_Script_Step_instToMessageData___lam__1___closed__14, &lp_aesop_Aesop_Script_Step_instToMessageData___lam__1___closed__14_once, _init_lp_aesop_Aesop_Script_Step_instToMessageData___lam__1___closed__14);
v___x_147_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_147_, 0, v___x_145_);
lean_ctor_set(v___x_147_, 1, v___x_146_);
v___x_148_ = l_Lean_MessageData_ofSyntax(v_uTactic_129_);
v___x_149_ = l_Lean_indentD(v___x_148_);
v___x_150_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_150_, 0, v___x_147_);
lean_ctor_set(v___x_150_, 1, v___x_149_);
return v___x_150_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_Step_mkSorry(lean_object* v_preGoal_167_, lean_object* v_preState_168_, lean_object* v_a_169_, lean_object* v_a_170_, lean_object* v_a_171_, lean_object* v_a_172_){
_start:
{
uint8_t v___x_174_; lean_object* v___x_175_; lean_object* v___x_176_; lean_object* v___x_177_; 
v___x_174_ = 0;
v___x_175_ = lean_box(v___x_174_);
lean_inc(v_preGoal_167_);
v___x_176_ = lean_alloc_closure((void*)(l_Lean_MVarId_admit___boxed), 7, 2);
lean_closure_set(v___x_176_, 0, v_preGoal_167_);
lean_closure_set(v___x_176_, 1, v___x_175_);
lean_inc_ref(v_preState_168_);
v___x_177_ = lp_batteries_Lean_Meta_SavedState_runMetaM___redArg(v_preState_168_, v___x_176_, v_a_169_, v_a_170_, v_a_171_, v_a_172_);
if (lean_obj_tag(v___x_177_) == 0)
{
lean_object* v_a_178_; lean_object* v___x_180_; uint8_t v_isShared_181_; uint8_t v_isSharedCheck_202_; 
v_a_178_ = lean_ctor_get(v___x_177_, 0);
v_isSharedCheck_202_ = !lean_is_exclusive(v___x_177_);
if (v_isSharedCheck_202_ == 0)
{
v___x_180_ = v___x_177_;
v_isShared_181_ = v_isSharedCheck_202_;
goto v_resetjp_179_;
}
else
{
lean_inc(v_a_178_);
lean_dec(v___x_177_);
v___x_180_ = lean_box(0);
v_isShared_181_ = v_isSharedCheck_202_;
goto v_resetjp_179_;
}
v_resetjp_179_:
{
lean_object* v_snd_182_; lean_object* v___x_184_; uint8_t v_isShared_185_; uint8_t v_isSharedCheck_200_; 
v_snd_182_ = lean_ctor_get(v_a_178_, 1);
v_isSharedCheck_200_ = !lean_is_exclusive(v_a_178_);
if (v_isSharedCheck_200_ == 0)
{
lean_object* v_unused_201_; 
v_unused_201_ = lean_ctor_get(v_a_178_, 0);
lean_dec(v_unused_201_);
v___x_184_ = v_a_178_;
v_isShared_185_ = v_isSharedCheck_200_;
goto v_resetjp_183_;
}
else
{
lean_inc(v_snd_182_);
lean_dec(v_a_178_);
v___x_184_ = lean_box(0);
v_isShared_185_ = v_isSharedCheck_200_;
goto v_resetjp_183_;
}
v_resetjp_183_:
{
lean_object* v_ref_186_; lean_object* v___x_187_; lean_object* v___x_188_; lean_object* v___x_189_; lean_object* v___x_191_; 
v_ref_186_ = lean_ctor_get(v_a_171_, 5);
v___x_187_ = l_Lean_SourceInfo_fromRef(v_ref_186_, v___x_174_);
v___x_188_ = ((lean_object*)(lp_aesop_Aesop_Script_Step_mkSorry___closed__1));
v___x_189_ = ((lean_object*)(lp_aesop_Aesop_Script_Step_mkSorry___closed__2));
lean_inc(v___x_187_);
if (v_isShared_185_ == 0)
{
lean_ctor_set_tag(v___x_184_, 2);
lean_ctor_set(v___x_184_, 1, v___x_189_);
lean_ctor_set(v___x_184_, 0, v___x_187_);
v___x_191_ = v___x_184_;
goto v_reusejp_190_;
}
else
{
lean_object* v_reuseFailAlloc_199_; 
v_reuseFailAlloc_199_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v_reuseFailAlloc_199_, 0, v___x_187_);
lean_ctor_set(v_reuseFailAlloc_199_, 1, v___x_189_);
v___x_191_ = v_reuseFailAlloc_199_;
goto v_reusejp_190_;
}
v_reusejp_190_:
{
lean_object* v___x_192_; lean_object* v___x_193_; lean_object* v___x_194_; lean_object* v___x_195_; lean_object* v___x_197_; 
v___x_192_ = l_Lean_Syntax_node1(v___x_187_, v___x_188_, v___x_191_);
v___x_193_ = lp_aesop_Aesop_Script_Tactic_unstructured(v___x_192_);
v___x_194_ = ((lean_object*)(lp_aesop_Aesop_Script_Step_mkSorry___closed__3));
v___x_195_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_195_, 0, v_preState_168_);
lean_ctor_set(v___x_195_, 1, v_preGoal_167_);
lean_ctor_set(v___x_195_, 2, v___x_193_);
lean_ctor_set(v___x_195_, 3, v_snd_182_);
lean_ctor_set(v___x_195_, 4, v___x_194_);
if (v_isShared_181_ == 0)
{
lean_ctor_set(v___x_180_, 0, v___x_195_);
v___x_197_ = v___x_180_;
goto v_reusejp_196_;
}
else
{
lean_object* v_reuseFailAlloc_198_; 
v_reuseFailAlloc_198_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_198_, 0, v___x_195_);
v___x_197_ = v_reuseFailAlloc_198_;
goto v_reusejp_196_;
}
v_reusejp_196_:
{
return v___x_197_;
}
}
}
}
}
else
{
lean_object* v_a_203_; lean_object* v___x_205_; uint8_t v_isShared_206_; uint8_t v_isSharedCheck_210_; 
lean_dec_ref(v_preState_168_);
lean_dec(v_preGoal_167_);
v_a_203_ = lean_ctor_get(v___x_177_, 0);
v_isSharedCheck_210_ = !lean_is_exclusive(v___x_177_);
if (v_isSharedCheck_210_ == 0)
{
v___x_205_ = v___x_177_;
v_isShared_206_ = v_isSharedCheck_210_;
goto v_resetjp_204_;
}
else
{
lean_inc(v_a_203_);
lean_dec(v___x_177_);
v___x_205_ = lean_box(0);
v_isShared_206_ = v_isSharedCheck_210_;
goto v_resetjp_204_;
}
v_resetjp_204_:
{
lean_object* v___x_208_; 
if (v_isShared_206_ == 0)
{
v___x_208_ = v___x_205_;
goto v_reusejp_207_;
}
else
{
lean_object* v_reuseFailAlloc_209_; 
v_reuseFailAlloc_209_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_209_, 0, v_a_203_);
v___x_208_ = v_reuseFailAlloc_209_;
goto v_reusejp_207_;
}
v_reusejp_207_:
{
return v___x_208_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_Step_mkSorry___boxed(lean_object* v_preGoal_211_, lean_object* v_preState_212_, lean_object* v_a_213_, lean_object* v_a_214_, lean_object* v_a_215_, lean_object* v_a_216_, lean_object* v_a_217_){
_start:
{
lean_object* v_res_218_; 
v_res_218_ = lp_aesop_Aesop_Script_Step_mkSorry(v_preGoal_211_, v_preState_212_, v_a_213_, v_a_214_, v_a_215_, v_a_216_);
lean_dec(v_a_216_);
lean_dec_ref(v_a_215_);
lean_dec(v_a_214_);
lean_dec_ref(v_a_213_);
return v_res_218_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_Step_render___redArg___lam__0(lean_object* v_tactic_219_, lean_object* v_pos_220_, lean_object* v_acc_221_, lean_object* v_toPure_222_, lean_object* v_tacticState_223_){
_start:
{
lean_object* v_uTactic_224_; lean_object* v___x_226_; uint8_t v_isShared_227_; uint8_t v_isSharedCheck_234_; 
v_uTactic_224_ = lean_ctor_get(v_tactic_219_, 0);
v_isSharedCheck_234_ = !lean_is_exclusive(v_tactic_219_);
if (v_isSharedCheck_234_ == 0)
{
lean_object* v_unused_235_; 
v_unused_235_ = lean_ctor_get(v_tactic_219_, 1);
lean_dec(v_unused_235_);
v___x_226_ = v_tactic_219_;
v_isShared_227_ = v_isSharedCheck_234_;
goto v_resetjp_225_;
}
else
{
lean_inc(v_uTactic_224_);
lean_dec(v_tactic_219_);
v___x_226_ = lean_box(0);
v_isShared_227_ = v_isSharedCheck_234_;
goto v_resetjp_225_;
}
v_resetjp_225_:
{
lean_object* v___x_228_; lean_object* v_acc_229_; lean_object* v___x_231_; 
v___x_228_ = lp_aesop_Aesop_Script_mkOnGoal(v_pos_220_, v_uTactic_224_);
v_acc_229_ = lean_array_push(v_acc_221_, v___x_228_);
if (v_isShared_227_ == 0)
{
lean_ctor_set(v___x_226_, 1, v_tacticState_223_);
lean_ctor_set(v___x_226_, 0, v_acc_229_);
v___x_231_ = v___x_226_;
goto v_reusejp_230_;
}
else
{
lean_object* v_reuseFailAlloc_233_; 
v_reuseFailAlloc_233_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_233_, 0, v_acc_229_);
lean_ctor_set(v_reuseFailAlloc_233_, 1, v_tacticState_223_);
v___x_231_ = v_reuseFailAlloc_233_;
goto v_reusejp_230_;
}
v_reusejp_230_:
{
lean_object* v___x_232_; 
v___x_232_ = lean_apply_2(v_toPure_222_, lean_box(0), v___x_231_);
return v___x_232_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_Step_render___redArg___lam__0___boxed(lean_object* v_tactic_236_, lean_object* v_pos_237_, lean_object* v_acc_238_, lean_object* v_toPure_239_, lean_object* v_tacticState_240_){
_start:
{
lean_object* v_res_241_; 
v_res_241_ = lp_aesop_Aesop_Script_Step_render___redArg___lam__0(v_tactic_236_, v_pos_237_, v_acc_238_, v_toPure_239_, v_tacticState_240_);
lean_dec(v_pos_237_);
return v_res_241_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_Step_render___redArg___lam__1(lean_object* v_tactic_242_, lean_object* v_acc_243_, lean_object* v_toPure_244_, lean_object* v_inst_245_, lean_object* v_inst_246_, lean_object* v_tacticState_247_, lean_object* v_step_248_, lean_object* v_toBind_249_, lean_object* v_pos_250_){
_start:
{
lean_object* v___f_251_; lean_object* v___x_252_; lean_object* v___x_253_; 
v___f_251_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Script_Step_render___redArg___lam__0___boxed), 5, 4);
lean_closure_set(v___f_251_, 0, v_tactic_242_);
lean_closure_set(v___f_251_, 1, v_pos_250_);
lean_closure_set(v___f_251_, 2, v_acc_243_);
lean_closure_set(v___f_251_, 3, v_toPure_244_);
v___x_252_ = lp_aesop_Aesop_Script_TacticState_applyStep___redArg(v_inst_245_, v_inst_246_, v_tacticState_247_, v_step_248_);
v___x_253_ = lean_apply_4(v_toBind_249_, lean_box(0), lean_box(0), v___x_252_, v___f_251_);
return v___x_253_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_Step_render___redArg(lean_object* v_inst_254_, lean_object* v_inst_255_, lean_object* v_acc_256_, lean_object* v_step_257_, lean_object* v_tacticState_258_){
_start:
{
lean_object* v_toApplicative_259_; lean_object* v_toBind_260_; lean_object* v_preGoal_261_; lean_object* v_tactic_262_; lean_object* v_toPure_263_; lean_object* v___x_264_; lean_object* v___f_265_; lean_object* v___x_266_; 
v_toApplicative_259_ = lean_ctor_get(v_inst_254_, 0);
v_toBind_260_ = lean_ctor_get(v_inst_254_, 1);
lean_inc_n(v_toBind_260_, 2);
v_preGoal_261_ = lean_ctor_get(v_step_257_, 1);
v_tactic_262_ = lean_ctor_get(v_step_257_, 2);
lean_inc_ref(v_tactic_262_);
v_toPure_263_ = lean_ctor_get(v_toApplicative_259_, 1);
lean_inc(v_toPure_263_);
lean_inc(v_preGoal_261_);
lean_inc_ref(v_inst_255_);
lean_inc_ref(v_inst_254_);
v___x_264_ = lp_aesop_Aesop_Script_TacticState_getVisibleGoalIndex___redArg(v_inst_254_, v_inst_255_, v_tacticState_258_, v_preGoal_261_);
v___f_265_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Script_Step_render___redArg___lam__1), 9, 8);
lean_closure_set(v___f_265_, 0, v_tactic_262_);
lean_closure_set(v___f_265_, 1, v_acc_256_);
lean_closure_set(v___f_265_, 2, v_toPure_263_);
lean_closure_set(v___f_265_, 3, v_inst_254_);
lean_closure_set(v___f_265_, 4, v_inst_255_);
lean_closure_set(v___f_265_, 5, v_tacticState_258_);
lean_closure_set(v___f_265_, 6, v_step_257_);
lean_closure_set(v___f_265_, 7, v_toBind_260_);
v___x_266_ = lean_apply_4(v_toBind_260_, lean_box(0), lean_box(0), v___x_264_, v___f_265_);
return v___x_266_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_Step_render(lean_object* v_m_267_, lean_object* v_inst_268_, lean_object* v_inst_269_, lean_object* v_acc_270_, lean_object* v_step_271_, lean_object* v_tacticState_272_){
_start:
{
lean_object* v___x_273_; 
v___x_273_ = lp_aesop_Aesop_Script_Step_render___redArg(v_inst_268_, v_inst_269_, v_acc_270_, v_step_271_, v_tacticState_272_);
return v___x_273_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00__private_Aesop_Script_Step_0__Aesop_Script_Step_validate_fmtGoals_spec__1(lean_object* v_msgData_274_, lean_object* v___y_275_, lean_object* v___y_276_, lean_object* v___y_277_, lean_object* v___y_278_){
_start:
{
lean_object* v___x_280_; lean_object* v_env_281_; lean_object* v___x_282_; lean_object* v_mctx_283_; lean_object* v_lctx_284_; lean_object* v_options_285_; lean_object* v___x_286_; lean_object* v___x_287_; lean_object* v___x_288_; 
v___x_280_ = lean_st_ref_get(v___y_278_);
v_env_281_ = lean_ctor_get(v___x_280_, 0);
lean_inc_ref(v_env_281_);
lean_dec(v___x_280_);
v___x_282_ = lean_st_ref_get(v___y_276_);
v_mctx_283_ = lean_ctor_get(v___x_282_, 0);
lean_inc_ref(v_mctx_283_);
lean_dec(v___x_282_);
v_lctx_284_ = lean_ctor_get(v___y_275_, 2);
v_options_285_ = lean_ctor_get(v___y_277_, 2);
lean_inc_ref(v_options_285_);
lean_inc_ref(v_lctx_284_);
v___x_286_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_286_, 0, v_env_281_);
lean_ctor_set(v___x_286_, 1, v_mctx_283_);
lean_ctor_set(v___x_286_, 2, v_lctx_284_);
lean_ctor_set(v___x_286_, 3, v_options_285_);
v___x_287_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_287_, 0, v___x_286_);
lean_ctor_set(v___x_287_, 1, v_msgData_274_);
v___x_288_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_288_, 0, v___x_287_);
return v___x_288_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00__private_Aesop_Script_Step_0__Aesop_Script_Step_validate_fmtGoals_spec__1___boxed(lean_object* v_msgData_289_, lean_object* v___y_290_, lean_object* v___y_291_, lean_object* v___y_292_, lean_object* v___y_293_, lean_object* v___y_294_){
_start:
{
lean_object* v_res_295_; 
v_res_295_ = lp_aesop_Lean_addMessageContextFull___at___00__private_Aesop_Script_Step_0__Aesop_Script_Step_validate_fmtGoals_spec__1(v_msgData_289_, v___y_290_, v___y_291_, v___y_292_, v___y_293_);
lean_dec(v___y_293_);
lean_dec_ref(v___y_292_);
lean_dec(v___y_291_);
lean_dec_ref(v___y_290_);
return v_res_295_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_Script_Step_0__Aesop_Script_Step_validate_fmtGoals_spec__0___redArg(size_t v_sz_296_, size_t v_i_297_, lean_object* v_bs_298_){
_start:
{
uint8_t v___x_300_; 
v___x_300_ = lean_usize_dec_lt(v_i_297_, v_sz_296_);
if (v___x_300_ == 0)
{
lean_object* v___x_301_; 
v___x_301_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_301_, 0, v_bs_298_);
return v___x_301_;
}
else
{
lean_object* v_v_302_; lean_object* v___x_303_; lean_object* v_bs_x27_304_; lean_object* v___x_305_; size_t v___x_306_; size_t v___x_307_; lean_object* v___x_308_; 
v_v_302_ = lean_array_uget(v_bs_298_, v_i_297_);
v___x_303_ = lean_unsigned_to_nat(0u);
v_bs_x27_304_ = lean_array_uset(v_bs_298_, v_i_297_, v___x_303_);
v___x_305_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_305_, 0, v_v_302_);
v___x_306_ = ((size_t)1ULL);
v___x_307_ = lean_usize_add(v_i_297_, v___x_306_);
v___x_308_ = lean_array_uset(v_bs_x27_304_, v_i_297_, v___x_305_);
v_i_297_ = v___x_307_;
v_bs_298_ = v___x_308_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_Script_Step_0__Aesop_Script_Step_validate_fmtGoals_spec__0___redArg___boxed(lean_object* v_sz_310_, lean_object* v_i_311_, lean_object* v_bs_312_, lean_object* v___y_313_){
_start:
{
size_t v_sz_boxed_314_; size_t v_i_boxed_315_; lean_object* v_res_316_; 
v_sz_boxed_314_ = lean_unbox_usize(v_sz_310_);
lean_dec(v_sz_310_);
v_i_boxed_315_ = lean_unbox_usize(v_i_311_);
lean_dec(v_i_311_);
v_res_316_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_Script_Step_0__Aesop_Script_Step_validate_fmtGoals_spec__0___redArg(v_sz_boxed_314_, v_i_boxed_315_, v_bs_312_);
return v_res_316_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Script_Step_0__Aesop_Script_Step_validate_fmtGoals___lam__0___closed__2(void){
_start:
{
lean_object* v___x_320_; lean_object* v___x_321_; 
v___x_320_ = ((lean_object*)(lp_aesop___private_Aesop_Script_Step_0__Aesop_Script_Step_validate_fmtGoals___lam__0___closed__1));
v___x_321_ = l_Lean_MessageData_ofFormat(v___x_320_);
return v___x_321_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_Step_0__Aesop_Script_Step_validate_fmtGoals___lam__0(lean_object* v_goals_322_, lean_object* v___y_323_, lean_object* v___y_324_, lean_object* v___y_325_, lean_object* v___y_326_){
_start:
{
size_t v_sz_328_; size_t v___x_329_; lean_object* v___x_330_; 
v_sz_328_ = lean_array_size(v_goals_322_);
v___x_329_ = ((size_t)0ULL);
v___x_330_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_Script_Step_0__Aesop_Script_Step_validate_fmtGoals_spec__0___redArg(v_sz_328_, v___x_329_, v_goals_322_);
if (lean_obj_tag(v___x_330_) == 0)
{
lean_object* v_a_331_; lean_object* v___x_332_; lean_object* v___x_333_; lean_object* v___x_334_; lean_object* v___x_335_; 
v_a_331_ = lean_ctor_get(v___x_330_, 0);
lean_inc(v_a_331_);
lean_dec_ref_known(v___x_330_, 1);
v___x_332_ = lean_array_to_list(v_a_331_);
v___x_333_ = lean_obj_once(&lp_aesop___private_Aesop_Script_Step_0__Aesop_Script_Step_validate_fmtGoals___lam__0___closed__2, &lp_aesop___private_Aesop_Script_Step_0__Aesop_Script_Step_validate_fmtGoals___lam__0___closed__2_once, _init_lp_aesop___private_Aesop_Script_Step_0__Aesop_Script_Step_validate_fmtGoals___lam__0___closed__2);
v___x_334_ = l_Lean_MessageData_joinSep(v___x_332_, v___x_333_);
v___x_335_ = lp_aesop_Lean_addMessageContextFull___at___00__private_Aesop_Script_Step_0__Aesop_Script_Step_validate_fmtGoals_spec__1(v___x_334_, v___y_323_, v___y_324_, v___y_325_, v___y_326_);
return v___x_335_;
}
else
{
lean_object* v_a_336_; lean_object* v___x_338_; uint8_t v_isShared_339_; uint8_t v_isSharedCheck_343_; 
v_a_336_ = lean_ctor_get(v___x_330_, 0);
v_isSharedCheck_343_ = !lean_is_exclusive(v___x_330_);
if (v_isSharedCheck_343_ == 0)
{
v___x_338_ = v___x_330_;
v_isShared_339_ = v_isSharedCheck_343_;
goto v_resetjp_337_;
}
else
{
lean_inc(v_a_336_);
lean_dec(v___x_330_);
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
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_Step_0__Aesop_Script_Step_validate_fmtGoals___lam__0___boxed(lean_object* v_goals_344_, lean_object* v___y_345_, lean_object* v___y_346_, lean_object* v___y_347_, lean_object* v___y_348_, lean_object* v___y_349_){
_start:
{
lean_object* v_res_350_; 
v_res_350_ = lp_aesop___private_Aesop_Script_Step_0__Aesop_Script_Step_validate_fmtGoals___lam__0(v_goals_344_, v___y_345_, v___y_346_, v___y_347_, v___y_348_);
lean_dec(v___y_348_);
lean_dec_ref(v___y_347_);
lean_dec(v___y_346_);
lean_dec_ref(v___y_345_);
return v_res_350_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_Step_0__Aesop_Script_Step_validate_fmtGoals(lean_object* v_state_351_, lean_object* v_goals_352_, lean_object* v_a_353_, lean_object* v_a_354_, lean_object* v_a_355_, lean_object* v_a_356_){
_start:
{
lean_object* v___f_358_; lean_object* v___x_359_; 
v___f_358_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Script_Step_0__Aesop_Script_Step_validate_fmtGoals___lam__0___boxed), 6, 1);
lean_closure_set(v___f_358_, 0, v_goals_352_);
v___x_359_ = lp_batteries_Lean_Meta_SavedState_runMetaM_x27___redArg(v_state_351_, v___f_358_, v_a_353_, v_a_354_, v_a_355_, v_a_356_);
return v___x_359_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_Step_0__Aesop_Script_Step_validate_fmtGoals___boxed(lean_object* v_state_360_, lean_object* v_goals_361_, lean_object* v_a_362_, lean_object* v_a_363_, lean_object* v_a_364_, lean_object* v_a_365_, lean_object* v_a_366_){
_start:
{
lean_object* v_res_367_; 
v_res_367_ = lp_aesop___private_Aesop_Script_Step_0__Aesop_Script_Step_validate_fmtGoals(v_state_360_, v_goals_361_, v_a_362_, v_a_363_, v_a_364_, v_a_365_);
lean_dec(v_a_365_);
lean_dec_ref(v_a_364_);
lean_dec(v_a_363_);
lean_dec_ref(v_a_362_);
return v_res_367_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_Script_Step_0__Aesop_Script_Step_validate_fmtGoals_spec__0(size_t v_sz_368_, size_t v_i_369_, lean_object* v_bs_370_, lean_object* v___y_371_, lean_object* v___y_372_, lean_object* v___y_373_, lean_object* v___y_374_){
_start:
{
lean_object* v___x_376_; 
v___x_376_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_Script_Step_0__Aesop_Script_Step_validate_fmtGoals_spec__0___redArg(v_sz_368_, v_i_369_, v_bs_370_);
return v___x_376_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_Script_Step_0__Aesop_Script_Step_validate_fmtGoals_spec__0___boxed(lean_object* v_sz_377_, lean_object* v_i_378_, lean_object* v_bs_379_, lean_object* v___y_380_, lean_object* v___y_381_, lean_object* v___y_382_, lean_object* v___y_383_, lean_object* v___y_384_){
_start:
{
size_t v_sz_boxed_385_; size_t v_i_boxed_386_; lean_object* v_res_387_; 
v_sz_boxed_385_ = lean_unbox_usize(v_sz_377_);
lean_dec(v_sz_377_);
v_i_boxed_386_ = lean_unbox_usize(v_i_378_);
lean_dec(v_i_378_);
v_res_387_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_Script_Step_0__Aesop_Script_Step_validate_fmtGoals_spec__0(v_sz_boxed_385_, v_i_boxed_386_, v_bs_379_, v___y_380_, v___y_381_, v___y_382_, v___y_383_);
lean_dec(v___y_383_);
lean_dec_ref(v___y_382_);
lean_dec(v___y_381_);
lean_dec_ref(v___y_380_);
return v_res_387_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_Script_Step_validate_spec__1___redArg(lean_object* v_msg_388_, lean_object* v___y_389_, lean_object* v___y_390_, lean_object* v___y_391_, lean_object* v___y_392_){
_start:
{
lean_object* v_ref_394_; lean_object* v___x_395_; lean_object* v_a_396_; lean_object* v___x_398_; uint8_t v_isShared_399_; uint8_t v_isSharedCheck_404_; 
v_ref_394_ = lean_ctor_get(v___y_391_, 5);
v___x_395_ = lp_aesop_Lean_addMessageContextFull___at___00__private_Aesop_Script_Step_0__Aesop_Script_Step_validate_fmtGoals_spec__1(v_msg_388_, v___y_389_, v___y_390_, v___y_391_, v___y_392_);
v_a_396_ = lean_ctor_get(v___x_395_, 0);
v_isSharedCheck_404_ = !lean_is_exclusive(v___x_395_);
if (v_isSharedCheck_404_ == 0)
{
v___x_398_ = v___x_395_;
v_isShared_399_ = v_isSharedCheck_404_;
goto v_resetjp_397_;
}
else
{
lean_inc(v_a_396_);
lean_dec(v___x_395_);
v___x_398_ = lean_box(0);
v_isShared_399_ = v_isSharedCheck_404_;
goto v_resetjp_397_;
}
v_resetjp_397_:
{
lean_object* v___x_400_; lean_object* v___x_402_; 
lean_inc(v_ref_394_);
v___x_400_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_400_, 0, v_ref_394_);
lean_ctor_set(v___x_400_, 1, v_a_396_);
if (v_isShared_399_ == 0)
{
lean_ctor_set_tag(v___x_398_, 1);
lean_ctor_set(v___x_398_, 0, v___x_400_);
v___x_402_ = v___x_398_;
goto v_reusejp_401_;
}
else
{
lean_object* v_reuseFailAlloc_403_; 
v_reuseFailAlloc_403_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_403_, 0, v___x_400_);
v___x_402_ = v_reuseFailAlloc_403_;
goto v_reusejp_401_;
}
v_reusejp_401_:
{
return v___x_402_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_Script_Step_validate_spec__1___redArg___boxed(lean_object* v_msg_405_, lean_object* v___y_406_, lean_object* v___y_407_, lean_object* v___y_408_, lean_object* v___y_409_, lean_object* v___y_410_){
_start:
{
lean_object* v_res_411_; 
v_res_411_ = lp_aesop_Lean_throwError___at___00Aesop_Script_Step_validate_spec__1___redArg(v_msg_405_, v___y_406_, v___y_407_, v___y_408_, v___y_409_);
lean_dec(v___y_409_);
lean_dec_ref(v___y_408_);
lean_dec(v___y_407_);
lean_dec_ref(v___y_406_);
return v_res_411_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_Step_validate_spec__0(size_t v_sz_412_, size_t v_i_413_, lean_object* v_bs_414_){
_start:
{
uint8_t v___x_415_; 
v___x_415_ = lean_usize_dec_lt(v_i_413_, v_sz_412_);
if (v___x_415_ == 0)
{
return v_bs_414_;
}
else
{
lean_object* v_v_416_; lean_object* v_goal_417_; lean_object* v___x_418_; lean_object* v_bs_x27_419_; size_t v___x_420_; size_t v___x_421_; lean_object* v___x_422_; 
v_v_416_ = lean_array_uget_borrowed(v_bs_414_, v_i_413_);
v_goal_417_ = lean_ctor_get(v_v_416_, 0);
lean_inc(v_goal_417_);
v___x_418_ = lean_unsigned_to_nat(0u);
v_bs_x27_419_ = lean_array_uset(v_bs_414_, v_i_413_, v___x_418_);
v___x_420_ = ((size_t)1ULL);
v___x_421_ = lean_usize_add(v_i_413_, v___x_420_);
v___x_422_ = lean_array_uset(v_bs_x27_419_, v_i_413_, v_goal_417_);
v_i_413_ = v___x_421_;
v_bs_414_ = v___x_422_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_Step_validate_spec__0___boxed(lean_object* v_sz_424_, lean_object* v_i_425_, lean_object* v_bs_426_){
_start:
{
size_t v_sz_boxed_427_; size_t v_i_boxed_428_; lean_object* v_res_429_; 
v_sz_boxed_427_ = lean_unbox_usize(v_sz_424_);
lean_dec(v_sz_424_);
v_i_boxed_428_ = lean_unbox_usize(v_i_425_);
lean_dec(v_i_425_);
v_res_429_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_Step_validate_spec__0(v_sz_boxed_427_, v_i_boxed_428_, v_bs_426_);
return v_res_429_;
}
}
static lean_object* _init_lp_aesop_Aesop_Script_Step_validate___closed__1(void){
_start:
{
lean_object* v___x_431_; lean_object* v___x_432_; 
v___x_431_ = ((lean_object*)(lp_aesop_Aesop_Script_Step_validate___closed__0));
v___x_432_ = l_Lean_stringToMessageData(v___x_431_);
return v___x_432_;
}
}
static lean_object* _init_lp_aesop_Aesop_Script_Step_validate___closed__3(void){
_start:
{
lean_object* v___x_434_; lean_object* v___x_435_; 
v___x_434_ = ((lean_object*)(lp_aesop_Aesop_Script_Step_validate___closed__2));
v___x_435_ = l_Lean_stringToMessageData(v___x_434_);
return v___x_435_;
}
}
static lean_object* _init_lp_aesop_Aesop_Script_Step_validate___closed__5(void){
_start:
{
lean_object* v___x_437_; lean_object* v___x_438_; 
v___x_437_ = ((lean_object*)(lp_aesop_Aesop_Script_Step_validate___closed__4));
v___x_438_ = l_Lean_stringToMessageData(v___x_437_);
return v___x_438_;
}
}
static lean_object* _init_lp_aesop_Aesop_Script_Step_validate___closed__7(void){
_start:
{
lean_object* v___x_440_; lean_object* v___x_441_; 
v___x_440_ = ((lean_object*)(lp_aesop_Aesop_Script_Step_validate___closed__6));
v___x_441_ = l_Lean_stringToMessageData(v___x_440_);
return v___x_441_;
}
}
static lean_object* _init_lp_aesop_Aesop_Script_Step_validate___closed__9(void){
_start:
{
lean_object* v___x_443_; lean_object* v___x_444_; 
v___x_443_ = ((lean_object*)(lp_aesop_Aesop_Script_Step_validate___closed__8));
v___x_444_ = l_Lean_stringToMessageData(v___x_443_);
return v___x_444_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_Step_validate(lean_object* v_step_445_, lean_object* v_a_446_, lean_object* v_a_447_, lean_object* v_a_448_, lean_object* v_a_449_){
_start:
{
lean_object* v_preState_451_; lean_object* v_meta_452_; lean_object* v_postState_453_; lean_object* v_meta_454_; lean_object* v_preGoal_455_; lean_object* v_postGoals_456_; lean_object* v_mctx_457_; lean_object* v_mctx_458_; size_t v_sz_459_; size_t v___x_460_; lean_object* v_expectedPostGoals_461_; lean_object* v_tac_462_; lean_object* v___y_464_; lean_object* v___x_557_; lean_object* v___x_558_; lean_object* v___x_559_; lean_object* v___x_560_; 
v_preState_451_ = lean_ctor_get(v_step_445_, 0);
lean_inc_ref_n(v_preState_451_, 2);
v_meta_452_ = lean_ctor_get(v_preState_451_, 1);
v_postState_453_ = lean_ctor_get(v_step_445_, 3);
lean_inc_ref(v_postState_453_);
v_meta_454_ = lean_ctor_get(v_postState_453_, 1);
v_preGoal_455_ = lean_ctor_get(v_step_445_, 1);
lean_inc_n(v_preGoal_455_, 2);
v_postGoals_456_ = lean_ctor_get(v_step_445_, 4);
v_mctx_457_ = lean_ctor_get(v_meta_452_, 0);
v_mctx_458_ = lean_ctor_get(v_meta_454_, 0);
v_sz_459_ = lean_array_size(v_postGoals_456_);
v___x_460_ = ((size_t)0ULL);
lean_inc_ref(v_postGoals_456_);
v_expectedPostGoals_461_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_Step_validate_spec__0(v_sz_459_, v___x_460_, v_postGoals_456_);
v_tac_462_ = lp_aesop_Aesop_Script_Step_uTactic(v_step_445_);
lean_dec_ref(v_step_445_);
lean_inc(v_tac_462_);
v___x_557_ = lean_alloc_closure((void*)(l_Lean_Elab_Tactic_evalTactic___boxed), 10, 1);
lean_closure_set(v___x_557_, 0, v_tac_462_);
v___x_558_ = lean_box(0);
v___x_559_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_559_, 0, v_preGoal_455_);
lean_ctor_set(v___x_559_, 1, v___x_558_);
v___x_560_ = lp_aesop_Aesop_runTacticMCapturingPostState(v___x_557_, v_preState_451_, v___x_559_, v_a_446_, v_a_447_, v_a_448_, v_a_449_);
if (lean_obj_tag(v___x_560_) == 0)
{
v___y_464_ = v___x_560_;
goto v___jp_463_;
}
else
{
lean_object* v_a_561_; uint8_t v___y_563_; uint8_t v___x_574_; 
v_a_561_ = lean_ctor_get(v___x_560_, 0);
lean_inc(v_a_561_);
v___x_574_ = l_Lean_Exception_isInterrupt(v_a_561_);
if (v___x_574_ == 0)
{
uint8_t v___x_575_; 
lean_inc(v_a_561_);
v___x_575_ = l_Lean_Exception_isRuntime(v_a_561_);
v___y_563_ = v___x_575_;
goto v___jp_562_;
}
else
{
v___y_563_ = v___x_574_;
goto v___jp_562_;
}
v___jp_562_:
{
if (v___y_563_ == 0)
{
lean_object* v___x_564_; lean_object* v___x_565_; lean_object* v___x_566_; lean_object* v___x_567_; lean_object* v___x_568_; lean_object* v___x_569_; lean_object* v___x_570_; lean_object* v___x_571_; lean_object* v___x_572_; lean_object* v___x_573_; 
lean_dec_ref_known(v___x_560_, 1);
v___x_564_ = lean_obj_once(&lp_aesop_Aesop_Script_Step_validate___closed__1, &lp_aesop_Aesop_Script_Step_validate___closed__1_once, _init_lp_aesop_Aesop_Script_Step_validate___closed__1);
lean_inc(v_tac_462_);
v___x_565_ = l_Lean_MessageData_ofSyntax(v_tac_462_);
v___x_566_ = l_Lean_indentD(v___x_565_);
v___x_567_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_567_, 0, v___x_564_);
lean_ctor_set(v___x_567_, 1, v___x_566_);
v___x_568_ = lean_obj_once(&lp_aesop_Aesop_Script_Step_validate___closed__9, &lp_aesop_Aesop_Script_Step_validate___closed__9_once, _init_lp_aesop_Aesop_Script_Step_validate___closed__9);
v___x_569_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_569_, 0, v___x_567_);
lean_ctor_set(v___x_569_, 1, v___x_568_);
v___x_570_ = l_Lean_Exception_toMessageData(v_a_561_);
v___x_571_ = l_Lean_indentD(v___x_570_);
v___x_572_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_572_, 0, v___x_569_);
lean_ctor_set(v___x_572_, 1, v___x_571_);
v___x_573_ = lp_aesop_Lean_throwError___at___00Aesop_Script_Step_validate_spec__1___redArg(v___x_572_, v_a_446_, v_a_447_, v_a_448_, v_a_449_);
v___y_464_ = v___x_573_;
goto v___jp_463_;
}
else
{
lean_dec(v_a_561_);
v___y_464_ = v___x_560_;
goto v___jp_463_;
}
}
}
v___jp_463_:
{
if (lean_obj_tag(v___y_464_) == 0)
{
lean_object* v_a_465_; lean_object* v_fst_466_; lean_object* v_meta_467_; lean_object* v_snd_468_; lean_object* v___x_470_; uint8_t v_isShared_471_; uint8_t v_isSharedCheck_547_; 
v_a_465_ = lean_ctor_get(v___y_464_, 0);
lean_inc(v_a_465_);
lean_dec_ref_known(v___y_464_, 1);
v_fst_466_ = lean_ctor_get(v_a_465_, 0);
lean_inc(v_fst_466_);
v_meta_467_ = lean_ctor_get(v_fst_466_, 1);
v_snd_468_ = lean_ctor_get(v_a_465_, 1);
v_isSharedCheck_547_ = !lean_is_exclusive(v_a_465_);
if (v_isSharedCheck_547_ == 0)
{
lean_object* v_unused_548_; 
v_unused_548_ = lean_ctor_get(v_a_465_, 0);
lean_dec(v_unused_548_);
v___x_470_ = v_a_465_;
v_isShared_471_ = v_isSharedCheck_547_;
goto v_resetjp_469_;
}
else
{
lean_inc(v_snd_468_);
lean_dec(v_a_465_);
v___x_470_ = lean_box(0);
v_isShared_471_ = v_isSharedCheck_547_;
goto v_resetjp_469_;
}
v_resetjp_469_:
{
lean_object* v_mctx_472_; lean_object* v___x_473_; lean_object* v___x_474_; uint8_t v___x_475_; lean_object* v___x_476_; 
v_mctx_472_ = lean_ctor_get(v_meta_467_, 0);
v___x_473_ = lean_array_mk(v_snd_468_);
lean_inc_ref(v_mctx_457_);
v___x_474_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_474_, 0, v_mctx_457_);
v___x_475_ = 0;
lean_inc_ref(v___x_473_);
lean_inc_ref(v_expectedPostGoals_461_);
lean_inc_ref(v_mctx_472_);
lean_inc_ref(v_mctx_458_);
v___x_476_ = lp_aesop_Aesop_tacticStatesEqualUpToIds(v___x_474_, v_mctx_458_, v_mctx_472_, v_expectedPostGoals_461_, v___x_473_, v___x_475_, v_a_446_, v_a_447_, v_a_448_, v_a_449_);
if (lean_obj_tag(v___x_476_) == 0)
{
lean_object* v_a_477_; lean_object* v___x_479_; uint8_t v_isShared_480_; uint8_t v_isSharedCheck_538_; 
v_a_477_ = lean_ctor_get(v___x_476_, 0);
v_isSharedCheck_538_ = !lean_is_exclusive(v___x_476_);
if (v_isSharedCheck_538_ == 0)
{
v___x_479_ = v___x_476_;
v_isShared_480_ = v_isSharedCheck_538_;
goto v_resetjp_478_;
}
else
{
lean_inc(v_a_477_);
lean_dec(v___x_476_);
v___x_479_ = lean_box(0);
v_isShared_480_ = v_isSharedCheck_538_;
goto v_resetjp_478_;
}
v_resetjp_478_:
{
uint8_t v___x_481_; 
v___x_481_ = lean_unbox(v_a_477_);
lean_dec(v_a_477_);
if (v___x_481_ == 0)
{
lean_object* v___x_482_; lean_object* v___x_483_; lean_object* v___x_484_; lean_object* v___x_485_; 
lean_del_object(v___x_479_);
v___x_482_ = lean_unsigned_to_nat(1u);
v___x_483_ = lean_mk_empty_array_with_capacity(v___x_482_);
v___x_484_ = lean_array_push(v___x_483_, v_preGoal_455_);
v___x_485_ = lp_aesop___private_Aesop_Script_Step_0__Aesop_Script_Step_validate_fmtGoals(v_preState_451_, v___x_484_, v_a_446_, v_a_447_, v_a_448_, v_a_449_);
if (lean_obj_tag(v___x_485_) == 0)
{
lean_object* v_a_486_; lean_object* v___x_487_; 
v_a_486_ = lean_ctor_get(v___x_485_, 0);
lean_inc(v_a_486_);
lean_dec_ref_known(v___x_485_, 1);
v___x_487_ = lp_aesop___private_Aesop_Script_Step_0__Aesop_Script_Step_validate_fmtGoals(v_postState_453_, v_expectedPostGoals_461_, v_a_446_, v_a_447_, v_a_448_, v_a_449_);
if (lean_obj_tag(v___x_487_) == 0)
{
lean_object* v_a_488_; lean_object* v___x_489_; 
v_a_488_ = lean_ctor_get(v___x_487_, 0);
lean_inc(v_a_488_);
lean_dec_ref_known(v___x_487_, 1);
v___x_489_ = lp_aesop___private_Aesop_Script_Step_0__Aesop_Script_Step_validate_fmtGoals(v_fst_466_, v___x_473_, v_a_446_, v_a_447_, v_a_448_, v_a_449_);
if (lean_obj_tag(v___x_489_) == 0)
{
lean_object* v_a_490_; lean_object* v___x_491_; lean_object* v___x_492_; lean_object* v___x_493_; lean_object* v___x_495_; 
v_a_490_ = lean_ctor_get(v___x_489_, 0);
lean_inc(v_a_490_);
lean_dec_ref_known(v___x_489_, 1);
v___x_491_ = lean_obj_once(&lp_aesop_Aesop_Script_Step_validate___closed__1, &lp_aesop_Aesop_Script_Step_validate___closed__1_once, _init_lp_aesop_Aesop_Script_Step_validate___closed__1);
v___x_492_ = l_Lean_MessageData_ofSyntax(v_tac_462_);
v___x_493_ = l_Lean_indentD(v___x_492_);
if (v_isShared_471_ == 0)
{
lean_ctor_set_tag(v___x_470_, 7);
lean_ctor_set(v___x_470_, 1, v___x_493_);
lean_ctor_set(v___x_470_, 0, v___x_491_);
v___x_495_ = v___x_470_;
goto v_reusejp_494_;
}
else
{
lean_object* v_reuseFailAlloc_509_; 
v_reuseFailAlloc_509_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_509_, 0, v___x_491_);
lean_ctor_set(v_reuseFailAlloc_509_, 1, v___x_493_);
v___x_495_ = v_reuseFailAlloc_509_;
goto v_reusejp_494_;
}
v_reusejp_494_:
{
lean_object* v___x_496_; lean_object* v___x_497_; lean_object* v___x_498_; lean_object* v___x_499_; lean_object* v___x_500_; lean_object* v___x_501_; lean_object* v___x_502_; lean_object* v___x_503_; lean_object* v___x_504_; lean_object* v___x_505_; lean_object* v___x_506_; lean_object* v___x_507_; lean_object* v___x_508_; 
v___x_496_ = lean_obj_once(&lp_aesop_Aesop_Script_Step_validate___closed__3, &lp_aesop_Aesop_Script_Step_validate___closed__3_once, _init_lp_aesop_Aesop_Script_Step_validate___closed__3);
v___x_497_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_497_, 0, v___x_495_);
lean_ctor_set(v___x_497_, 1, v___x_496_);
v___x_498_ = l_Lean_indentD(v_a_486_);
v___x_499_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_499_, 0, v___x_497_);
lean_ctor_set(v___x_499_, 1, v___x_498_);
v___x_500_ = lean_obj_once(&lp_aesop_Aesop_Script_Step_validate___closed__5, &lp_aesop_Aesop_Script_Step_validate___closed__5_once, _init_lp_aesop_Aesop_Script_Step_validate___closed__5);
v___x_501_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_501_, 0, v___x_499_);
lean_ctor_set(v___x_501_, 1, v___x_500_);
v___x_502_ = l_Lean_indentD(v_a_488_);
v___x_503_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_503_, 0, v___x_501_);
lean_ctor_set(v___x_503_, 1, v___x_502_);
v___x_504_ = lean_obj_once(&lp_aesop_Aesop_Script_Step_validate___closed__7, &lp_aesop_Aesop_Script_Step_validate___closed__7_once, _init_lp_aesop_Aesop_Script_Step_validate___closed__7);
v___x_505_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_505_, 0, v___x_503_);
lean_ctor_set(v___x_505_, 1, v___x_504_);
v___x_506_ = l_Lean_indentD(v_a_490_);
v___x_507_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_507_, 0, v___x_505_);
lean_ctor_set(v___x_507_, 1, v___x_506_);
v___x_508_ = lp_aesop_Lean_throwError___at___00Aesop_Script_Step_validate_spec__1___redArg(v___x_507_, v_a_446_, v_a_447_, v_a_448_, v_a_449_);
return v___x_508_;
}
}
else
{
lean_object* v_a_510_; lean_object* v___x_512_; uint8_t v_isShared_513_; uint8_t v_isSharedCheck_517_; 
lean_dec(v_a_488_);
lean_dec(v_a_486_);
lean_del_object(v___x_470_);
lean_dec(v_tac_462_);
v_a_510_ = lean_ctor_get(v___x_489_, 0);
v_isSharedCheck_517_ = !lean_is_exclusive(v___x_489_);
if (v_isSharedCheck_517_ == 0)
{
v___x_512_ = v___x_489_;
v_isShared_513_ = v_isSharedCheck_517_;
goto v_resetjp_511_;
}
else
{
lean_inc(v_a_510_);
lean_dec(v___x_489_);
v___x_512_ = lean_box(0);
v_isShared_513_ = v_isSharedCheck_517_;
goto v_resetjp_511_;
}
v_resetjp_511_:
{
lean_object* v___x_515_; 
if (v_isShared_513_ == 0)
{
v___x_515_ = v___x_512_;
goto v_reusejp_514_;
}
else
{
lean_object* v_reuseFailAlloc_516_; 
v_reuseFailAlloc_516_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_516_, 0, v_a_510_);
v___x_515_ = v_reuseFailAlloc_516_;
goto v_reusejp_514_;
}
v_reusejp_514_:
{
return v___x_515_;
}
}
}
}
else
{
lean_object* v_a_518_; lean_object* v___x_520_; uint8_t v_isShared_521_; uint8_t v_isSharedCheck_525_; 
lean_dec(v_a_486_);
lean_dec_ref(v___x_473_);
lean_del_object(v___x_470_);
lean_dec(v_fst_466_);
lean_dec(v_tac_462_);
v_a_518_ = lean_ctor_get(v___x_487_, 0);
v_isSharedCheck_525_ = !lean_is_exclusive(v___x_487_);
if (v_isSharedCheck_525_ == 0)
{
v___x_520_ = v___x_487_;
v_isShared_521_ = v_isSharedCheck_525_;
goto v_resetjp_519_;
}
else
{
lean_inc(v_a_518_);
lean_dec(v___x_487_);
v___x_520_ = lean_box(0);
v_isShared_521_ = v_isSharedCheck_525_;
goto v_resetjp_519_;
}
v_resetjp_519_:
{
lean_object* v___x_523_; 
if (v_isShared_521_ == 0)
{
v___x_523_ = v___x_520_;
goto v_reusejp_522_;
}
else
{
lean_object* v_reuseFailAlloc_524_; 
v_reuseFailAlloc_524_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_524_, 0, v_a_518_);
v___x_523_ = v_reuseFailAlloc_524_;
goto v_reusejp_522_;
}
v_reusejp_522_:
{
return v___x_523_;
}
}
}
}
else
{
lean_object* v_a_526_; lean_object* v___x_528_; uint8_t v_isShared_529_; uint8_t v_isSharedCheck_533_; 
lean_dec_ref(v___x_473_);
lean_del_object(v___x_470_);
lean_dec(v_fst_466_);
lean_dec(v_tac_462_);
lean_dec_ref(v_expectedPostGoals_461_);
lean_dec_ref(v_postState_453_);
v_a_526_ = lean_ctor_get(v___x_485_, 0);
v_isSharedCheck_533_ = !lean_is_exclusive(v___x_485_);
if (v_isSharedCheck_533_ == 0)
{
v___x_528_ = v___x_485_;
v_isShared_529_ = v_isSharedCheck_533_;
goto v_resetjp_527_;
}
else
{
lean_inc(v_a_526_);
lean_dec(v___x_485_);
v___x_528_ = lean_box(0);
v_isShared_529_ = v_isSharedCheck_533_;
goto v_resetjp_527_;
}
v_resetjp_527_:
{
lean_object* v___x_531_; 
if (v_isShared_529_ == 0)
{
v___x_531_ = v___x_528_;
goto v_reusejp_530_;
}
else
{
lean_object* v_reuseFailAlloc_532_; 
v_reuseFailAlloc_532_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_532_, 0, v_a_526_);
v___x_531_ = v_reuseFailAlloc_532_;
goto v_reusejp_530_;
}
v_reusejp_530_:
{
return v___x_531_;
}
}
}
}
else
{
lean_object* v___x_534_; lean_object* v___x_536_; 
lean_dec_ref(v___x_473_);
lean_del_object(v___x_470_);
lean_dec(v_fst_466_);
lean_dec(v_tac_462_);
lean_dec_ref(v_expectedPostGoals_461_);
lean_dec(v_preGoal_455_);
lean_dec_ref(v_postState_453_);
lean_dec_ref(v_preState_451_);
v___x_534_ = lean_box(0);
if (v_isShared_480_ == 0)
{
lean_ctor_set(v___x_479_, 0, v___x_534_);
v___x_536_ = v___x_479_;
goto v_reusejp_535_;
}
else
{
lean_object* v_reuseFailAlloc_537_; 
v_reuseFailAlloc_537_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_537_, 0, v___x_534_);
v___x_536_ = v_reuseFailAlloc_537_;
goto v_reusejp_535_;
}
v_reusejp_535_:
{
return v___x_536_;
}
}
}
}
else
{
lean_object* v_a_539_; lean_object* v___x_541_; uint8_t v_isShared_542_; uint8_t v_isSharedCheck_546_; 
lean_dec_ref(v___x_473_);
lean_del_object(v___x_470_);
lean_dec(v_fst_466_);
lean_dec(v_tac_462_);
lean_dec_ref(v_expectedPostGoals_461_);
lean_dec(v_preGoal_455_);
lean_dec_ref(v_postState_453_);
lean_dec_ref(v_preState_451_);
v_a_539_ = lean_ctor_get(v___x_476_, 0);
v_isSharedCheck_546_ = !lean_is_exclusive(v___x_476_);
if (v_isSharedCheck_546_ == 0)
{
v___x_541_ = v___x_476_;
v_isShared_542_ = v_isSharedCheck_546_;
goto v_resetjp_540_;
}
else
{
lean_inc(v_a_539_);
lean_dec(v___x_476_);
v___x_541_ = lean_box(0);
v_isShared_542_ = v_isSharedCheck_546_;
goto v_resetjp_540_;
}
v_resetjp_540_:
{
lean_object* v___x_544_; 
if (v_isShared_542_ == 0)
{
v___x_544_ = v___x_541_;
goto v_reusejp_543_;
}
else
{
lean_object* v_reuseFailAlloc_545_; 
v_reuseFailAlloc_545_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_545_, 0, v_a_539_);
v___x_544_ = v_reuseFailAlloc_545_;
goto v_reusejp_543_;
}
v_reusejp_543_:
{
return v___x_544_;
}
}
}
}
}
else
{
lean_object* v_a_549_; lean_object* v___x_551_; uint8_t v_isShared_552_; uint8_t v_isSharedCheck_556_; 
lean_dec(v_tac_462_);
lean_dec_ref(v_expectedPostGoals_461_);
lean_dec(v_preGoal_455_);
lean_dec_ref(v_postState_453_);
lean_dec_ref(v_preState_451_);
v_a_549_ = lean_ctor_get(v___y_464_, 0);
v_isSharedCheck_556_ = !lean_is_exclusive(v___y_464_);
if (v_isSharedCheck_556_ == 0)
{
v___x_551_ = v___y_464_;
v_isShared_552_ = v_isSharedCheck_556_;
goto v_resetjp_550_;
}
else
{
lean_inc(v_a_549_);
lean_dec(v___y_464_);
v___x_551_ = lean_box(0);
v_isShared_552_ = v_isSharedCheck_556_;
goto v_resetjp_550_;
}
v_resetjp_550_:
{
lean_object* v___x_554_; 
if (v_isShared_552_ == 0)
{
v___x_554_ = v___x_551_;
goto v_reusejp_553_;
}
else
{
lean_object* v_reuseFailAlloc_555_; 
v_reuseFailAlloc_555_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_555_, 0, v_a_549_);
v___x_554_ = v_reuseFailAlloc_555_;
goto v_reusejp_553_;
}
v_reusejp_553_:
{
return v___x_554_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_Step_validate___boxed(lean_object* v_step_576_, lean_object* v_a_577_, lean_object* v_a_578_, lean_object* v_a_579_, lean_object* v_a_580_, lean_object* v_a_581_){
_start:
{
lean_object* v_res_582_; 
v_res_582_ = lp_aesop_Aesop_Script_Step_validate(v_step_576_, v_a_577_, v_a_578_, v_a_579_, v_a_580_);
lean_dec(v_a_580_);
lean_dec_ref(v_a_579_);
lean_dec(v_a_578_);
lean_dec_ref(v_a_577_);
return v_res_582_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_Script_Step_validate_spec__1(lean_object* v_00_u03b1_583_, lean_object* v_msg_584_, lean_object* v___y_585_, lean_object* v___y_586_, lean_object* v___y_587_, lean_object* v___y_588_){
_start:
{
lean_object* v___x_590_; 
v___x_590_ = lp_aesop_Lean_throwError___at___00Aesop_Script_Step_validate_spec__1___redArg(v_msg_584_, v___y_585_, v___y_586_, v___y_587_, v___y_588_);
return v___x_590_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_Script_Step_validate_spec__1___boxed(lean_object* v_00_u03b1_591_, lean_object* v_msg_592_, lean_object* v___y_593_, lean_object* v___y_594_, lean_object* v___y_595_, lean_object* v___y_596_, lean_object* v___y_597_){
_start:
{
lean_object* v_res_598_; 
v_res_598_ = lp_aesop_Lean_throwError___at___00Aesop_Script_Step_validate_spec__1(v_00_u03b1_591_, v_msg_592_, v___y_593_, v___y_594_, v___y_595_, v___y_596_);
lean_dec(v___y_596_);
lean_dec_ref(v___y_595_);
lean_dec(v___y_594_);
lean_dec_ref(v___y_593_);
return v_res_598_;
}
}
static lean_object* _init_lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__3(void){
_start:
{
lean_object* v___x_607_; lean_object* v___x_608_; 
v___x_607_ = ((lean_object*)(lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__1));
v___x_608_ = l_Lean_mkAtom(v___x_607_);
return v___x_608_;
}
}
static lean_object* _init_lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__4(void){
_start:
{
lean_object* v___x_609_; lean_object* v___x_610_; lean_object* v___x_611_; 
v___x_609_ = lean_obj_once(&lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__3, &lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__3_once, _init_lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__3);
v___x_610_ = ((lean_object*)(lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__0));
v___x_611_ = lean_array_push(v___x_610_, v___x_609_);
return v___x_611_;
}
}
static lean_object* _init_lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__8(void){
_start:
{
lean_object* v___x_622_; lean_object* v___x_623_; lean_object* v___x_624_; 
v___x_622_ = ((lean_object*)(lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__7));
v___x_623_ = ((lean_object*)(lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__0));
v___x_624_ = lean_array_push(v___x_623_, v___x_622_);
return v___x_624_;
}
}
static lean_object* _init_lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__9(void){
_start:
{
lean_object* v___x_625_; lean_object* v___x_626_; lean_object* v___x_627_; lean_object* v___x_628_; 
v___x_625_ = lean_obj_once(&lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__8, &lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__8_once, _init_lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__8);
v___x_626_ = ((lean_object*)(lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__6));
v___x_627_ = lean_box(2);
v___x_628_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_628_, 0, v___x_627_);
lean_ctor_set(v___x_628_, 1, v___x_626_);
lean_ctor_set(v___x_628_, 2, v___x_625_);
return v___x_628_;
}
}
static lean_object* _init_lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__10(void){
_start:
{
lean_object* v___x_629_; lean_object* v___x_630_; lean_object* v___x_631_; 
v___x_629_ = lean_obj_once(&lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__9, &lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__9_once, _init_lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__9);
v___x_630_ = lean_obj_once(&lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__4, &lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__4_once, _init_lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__4);
v___x_631_ = lean_array_push(v___x_630_, v___x_629_);
return v___x_631_;
}
}
static lean_object* _init_lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__11(void){
_start:
{
lean_object* v___x_632_; lean_object* v___x_633_; lean_object* v___x_634_; 
v___x_632_ = ((lean_object*)(lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__7));
v___x_633_ = lean_obj_once(&lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__10, &lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__10_once, _init_lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__10);
v___x_634_ = lean_array_push(v___x_633_, v___x_632_);
return v___x_634_;
}
}
static lean_object* _init_lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__12(void){
_start:
{
lean_object* v___x_635_; lean_object* v___x_636_; lean_object* v___x_637_; 
v___x_635_ = ((lean_object*)(lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__7));
v___x_636_ = lean_obj_once(&lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__11, &lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__11_once, _init_lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__11);
v___x_637_ = lean_array_push(v___x_636_, v___x_635_);
return v___x_637_;
}
}
static lean_object* _init_lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__13(void){
_start:
{
lean_object* v___x_638_; lean_object* v___x_639_; lean_object* v___x_640_; 
v___x_638_ = ((lean_object*)(lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__7));
v___x_639_ = lean_obj_once(&lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__12, &lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__12_once, _init_lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__12);
v___x_640_ = lean_array_push(v___x_639_, v___x_638_);
return v___x_640_;
}
}
static lean_object* _init_lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__14(void){
_start:
{
lean_object* v___x_641_; lean_object* v___x_642_; lean_object* v___x_643_; 
v___x_641_ = ((lean_object*)(lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__7));
v___x_642_ = lean_obj_once(&lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__13, &lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__13_once, _init_lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__13);
v___x_643_ = lean_array_push(v___x_642_, v___x_641_);
return v___x_643_;
}
}
static lean_object* _init_lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__15(void){
_start:
{
lean_object* v___x_644_; lean_object* v___x_645_; lean_object* v___x_646_; lean_object* v___x_647_; 
v___x_644_ = lean_obj_once(&lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__14, &lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__14_once, _init_lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__14);
v___x_645_ = ((lean_object*)(lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__2));
v___x_646_ = lean_box(2);
v___x_647_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_647_, 0, v___x_646_);
lean_ctor_set(v___x_647_, 1, v___x_645_);
lean_ctor_set(v___x_647_, 2, v___x_644_);
return v___x_647_;
}
}
static lean_object* _init_lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__16(void){
_start:
{
lean_object* v___x_648_; lean_object* v___x_649_; lean_object* v___x_650_; 
v___x_648_ = lean_obj_once(&lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__15, &lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__15_once, _init_lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__15);
v___x_649_ = ((lean_object*)(lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__0));
v___x_650_ = lean_array_push(v___x_649_, v___x_648_);
return v___x_650_;
}
}
static lean_object* _init_lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__17(void){
_start:
{
lean_object* v___x_651_; lean_object* v___x_652_; lean_object* v___x_653_; lean_object* v___x_654_; 
v___x_651_ = lean_obj_once(&lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__16, &lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__16_once, _init_lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__16);
v___x_652_ = ((lean_object*)(lp_aesop_Aesop_Script_mkOnGoal___closed__6));
v___x_653_ = lean_box(2);
v___x_654_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_654_, 0, v___x_653_);
lean_ctor_set(v___x_654_, 1, v___x_652_);
lean_ctor_set(v___x_654_, 2, v___x_651_);
return v___x_654_;
}
}
static lean_object* _init_lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__18(void){
_start:
{
lean_object* v___x_655_; lean_object* v___x_656_; lean_object* v___x_657_; 
v___x_655_ = lean_obj_once(&lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__17, &lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__17_once, _init_lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__17);
v___x_656_ = ((lean_object*)(lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__0));
v___x_657_ = lean_array_push(v___x_656_, v___x_655_);
return v___x_657_;
}
}
static lean_object* _init_lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__19(void){
_start:
{
lean_object* v___x_658_; lean_object* v___x_659_; lean_object* v___x_660_; lean_object* v___x_661_; 
v___x_658_ = lean_obj_once(&lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__18, &lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__18_once, _init_lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__18);
v___x_659_ = ((lean_object*)(lp_aesop_Aesop_Script_mkOnGoal___closed__14));
v___x_660_ = lean_box(2);
v___x_661_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_661_, 0, v___x_660_);
lean_ctor_set(v___x_661_, 1, v___x_659_);
lean_ctor_set(v___x_661_, 2, v___x_658_);
return v___x_661_;
}
}
static lean_object* _init_lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__20(void){
_start:
{
lean_object* v___x_662_; lean_object* v___x_663_; lean_object* v___x_664_; 
v___x_662_ = lean_obj_once(&lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__19, &lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__19_once, _init_lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__19);
v___x_663_ = ((lean_object*)(lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__0));
v___x_664_ = lean_array_push(v___x_663_, v___x_662_);
return v___x_664_;
}
}
static lean_object* _init_lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__21(void){
_start:
{
lean_object* v___x_665_; lean_object* v___x_666_; lean_object* v___x_667_; lean_object* v___x_668_; 
v___x_665_ = lean_obj_once(&lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__20, &lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__20_once, _init_lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__20);
v___x_666_ = ((lean_object*)(lp_aesop_Aesop_Script_mkOnGoal___closed__12));
v___x_667_ = lean_box(2);
v___x_668_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_668_, 0, v___x_667_);
lean_ctor_set(v___x_668_, 1, v___x_666_);
lean_ctor_set(v___x_668_, 2, v___x_665_);
return v___x_668_;
}
}
static lean_object* _init_lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam(void){
_start:
{
lean_object* v___x_669_; 
v___x_669_ = lean_obj_once(&lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__21, &lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__21_once, _init_lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam___closed__21);
return v___x_669_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_observing_x3f___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__0___redArg(lean_object* v_x_670_, lean_object* v___y_671_, lean_object* v___y_672_, lean_object* v___y_673_, lean_object* v___y_674_){
_start:
{
lean_object* v___x_676_; 
v___x_676_ = l_Lean_Meta_saveState___redArg(v___y_672_, v___y_674_);
if (lean_obj_tag(v___x_676_) == 0)
{
lean_object* v_a_677_; lean_object* v___x_678_; 
v_a_677_ = lean_ctor_get(v___x_676_, 0);
lean_inc(v_a_677_);
lean_dec_ref_known(v___x_676_, 1);
lean_inc(v___y_674_);
lean_inc_ref(v___y_673_);
lean_inc(v___y_672_);
lean_inc_ref(v___y_671_);
v___x_678_ = lean_apply_5(v_x_670_, v___y_671_, v___y_672_, v___y_673_, v___y_674_, lean_box(0));
if (lean_obj_tag(v___x_678_) == 0)
{
lean_object* v_a_679_; lean_object* v___x_681_; uint8_t v_isShared_682_; uint8_t v_isSharedCheck_687_; 
lean_dec(v_a_677_);
v_a_679_ = lean_ctor_get(v___x_678_, 0);
v_isSharedCheck_687_ = !lean_is_exclusive(v___x_678_);
if (v_isSharedCheck_687_ == 0)
{
v___x_681_ = v___x_678_;
v_isShared_682_ = v_isSharedCheck_687_;
goto v_resetjp_680_;
}
else
{
lean_inc(v_a_679_);
lean_dec(v___x_678_);
v___x_681_ = lean_box(0);
v_isShared_682_ = v_isSharedCheck_687_;
goto v_resetjp_680_;
}
v_resetjp_680_:
{
lean_object* v___x_683_; lean_object* v___x_685_; 
v___x_683_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_683_, 0, v_a_679_);
if (v_isShared_682_ == 0)
{
lean_ctor_set(v___x_681_, 0, v___x_683_);
v___x_685_ = v___x_681_;
goto v_reusejp_684_;
}
else
{
lean_object* v_reuseFailAlloc_686_; 
v_reuseFailAlloc_686_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_686_, 0, v___x_683_);
v___x_685_ = v_reuseFailAlloc_686_;
goto v_reusejp_684_;
}
v_reusejp_684_:
{
return v___x_685_;
}
}
}
else
{
lean_object* v_a_688_; lean_object* v___x_690_; uint8_t v_isShared_691_; uint8_t v_isSharedCheck_717_; 
v_a_688_ = lean_ctor_get(v___x_678_, 0);
v_isSharedCheck_717_ = !lean_is_exclusive(v___x_678_);
if (v_isSharedCheck_717_ == 0)
{
v___x_690_ = v___x_678_;
v_isShared_691_ = v_isSharedCheck_717_;
goto v_resetjp_689_;
}
else
{
lean_inc(v_a_688_);
lean_dec(v___x_678_);
v___x_690_ = lean_box(0);
v_isShared_691_ = v_isSharedCheck_717_;
goto v_resetjp_689_;
}
v_resetjp_689_:
{
uint8_t v___y_693_; uint8_t v___x_715_; 
v___x_715_ = l_Lean_Exception_isInterrupt(v_a_688_);
if (v___x_715_ == 0)
{
uint8_t v___x_716_; 
lean_inc(v_a_688_);
v___x_716_ = l_Lean_Exception_isRuntime(v_a_688_);
v___y_693_ = v___x_716_;
goto v___jp_692_;
}
else
{
v___y_693_ = v___x_715_;
goto v___jp_692_;
}
v___jp_692_:
{
if (v___y_693_ == 0)
{
lean_object* v___x_694_; 
lean_del_object(v___x_690_);
lean_dec(v_a_688_);
v___x_694_ = l_Lean_Meta_SavedState_restore___redArg(v_a_677_, v___y_672_, v___y_674_);
lean_dec(v_a_677_);
if (lean_obj_tag(v___x_694_) == 0)
{
lean_object* v___x_696_; uint8_t v_isShared_697_; uint8_t v_isSharedCheck_702_; 
v_isSharedCheck_702_ = !lean_is_exclusive(v___x_694_);
if (v_isSharedCheck_702_ == 0)
{
lean_object* v_unused_703_; 
v_unused_703_ = lean_ctor_get(v___x_694_, 0);
lean_dec(v_unused_703_);
v___x_696_ = v___x_694_;
v_isShared_697_ = v_isSharedCheck_702_;
goto v_resetjp_695_;
}
else
{
lean_dec(v___x_694_);
v___x_696_ = lean_box(0);
v_isShared_697_ = v_isSharedCheck_702_;
goto v_resetjp_695_;
}
v_resetjp_695_:
{
lean_object* v___x_698_; lean_object* v___x_700_; 
v___x_698_ = lean_box(0);
if (v_isShared_697_ == 0)
{
lean_ctor_set(v___x_696_, 0, v___x_698_);
v___x_700_ = v___x_696_;
goto v_reusejp_699_;
}
else
{
lean_object* v_reuseFailAlloc_701_; 
v_reuseFailAlloc_701_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_701_, 0, v___x_698_);
v___x_700_ = v_reuseFailAlloc_701_;
goto v_reusejp_699_;
}
v_reusejp_699_:
{
return v___x_700_;
}
}
}
else
{
lean_object* v_a_704_; lean_object* v___x_706_; uint8_t v_isShared_707_; uint8_t v_isSharedCheck_711_; 
v_a_704_ = lean_ctor_get(v___x_694_, 0);
v_isSharedCheck_711_ = !lean_is_exclusive(v___x_694_);
if (v_isSharedCheck_711_ == 0)
{
v___x_706_ = v___x_694_;
v_isShared_707_ = v_isSharedCheck_711_;
goto v_resetjp_705_;
}
else
{
lean_inc(v_a_704_);
lean_dec(v___x_694_);
v___x_706_ = lean_box(0);
v_isShared_707_ = v_isSharedCheck_711_;
goto v_resetjp_705_;
}
v_resetjp_705_:
{
lean_object* v___x_709_; 
if (v_isShared_707_ == 0)
{
v___x_709_ = v___x_706_;
goto v_reusejp_708_;
}
else
{
lean_object* v_reuseFailAlloc_710_; 
v_reuseFailAlloc_710_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_710_, 0, v_a_704_);
v___x_709_ = v_reuseFailAlloc_710_;
goto v_reusejp_708_;
}
v_reusejp_708_:
{
return v___x_709_;
}
}
}
}
else
{
lean_object* v___x_713_; 
lean_dec(v_a_677_);
if (v_isShared_691_ == 0)
{
v___x_713_ = v___x_690_;
goto v_reusejp_712_;
}
else
{
lean_object* v_reuseFailAlloc_714_; 
v_reuseFailAlloc_714_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_714_, 0, v_a_688_);
v___x_713_ = v_reuseFailAlloc_714_;
goto v_reusejp_712_;
}
v_reusejp_712_:
{
return v___x_713_;
}
}
}
}
}
}
else
{
lean_object* v_a_718_; lean_object* v___x_720_; uint8_t v_isShared_721_; uint8_t v_isSharedCheck_725_; 
lean_dec_ref(v_x_670_);
v_a_718_ = lean_ctor_get(v___x_676_, 0);
v_isSharedCheck_725_ = !lean_is_exclusive(v___x_676_);
if (v_isSharedCheck_725_ == 0)
{
v___x_720_ = v___x_676_;
v_isShared_721_ = v_isSharedCheck_725_;
goto v_resetjp_719_;
}
else
{
lean_inc(v_a_718_);
lean_dec(v___x_676_);
v___x_720_ = lean_box(0);
v_isShared_721_ = v_isSharedCheck_725_;
goto v_resetjp_719_;
}
v_resetjp_719_:
{
lean_object* v___x_723_; 
if (v_isShared_721_ == 0)
{
v___x_723_ = v___x_720_;
goto v_reusejp_722_;
}
else
{
lean_object* v_reuseFailAlloc_724_; 
v_reuseFailAlloc_724_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_724_, 0, v_a_718_);
v___x_723_ = v_reuseFailAlloc_724_;
goto v_reusejp_722_;
}
v_reusejp_722_:
{
return v___x_723_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_observing_x3f___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__0___redArg___boxed(lean_object* v_x_726_, lean_object* v___y_727_, lean_object* v___y_728_, lean_object* v___y_729_, lean_object* v___y_730_, lean_object* v___y_731_){
_start:
{
lean_object* v_res_732_; 
v_res_732_ = lp_aesop_Lean_observing_x3f___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__0___redArg(v_x_726_, v___y_727_, v___y_728_, v___y_729_, v___y_730_);
lean_dec(v___y_730_);
lean_dec_ref(v___y_729_);
lean_dec(v___y_728_);
lean_dec_ref(v___y_727_);
return v_res_732_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_observing_x3f___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__0(lean_object* v_00_u03b1_733_, lean_object* v_x_734_, lean_object* v___y_735_, lean_object* v___y_736_, lean_object* v___y_737_, lean_object* v___y_738_){
_start:
{
lean_object* v___x_740_; 
v___x_740_ = lp_aesop_Lean_observing_x3f___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__0___redArg(v_x_734_, v___y_735_, v___y_736_, v___y_737_, v___y_738_);
return v___x_740_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_observing_x3f___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__0___boxed(lean_object* v_00_u03b1_741_, lean_object* v_x_742_, lean_object* v___y_743_, lean_object* v___y_744_, lean_object* v___y_745_, lean_object* v___y_746_, lean_object* v___y_747_){
_start:
{
lean_object* v_res_748_; 
v_res_748_ = lp_aesop_Lean_observing_x3f___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__0(v_00_u03b1_741_, v_x_742_, v___y_743_, v___y_744_, v___y_745_, v___y_746_);
lean_dec(v___y_746_);
lean_dec_ref(v___y_745_);
lean_dec(v___y_744_);
lean_dec_ref(v___y_743_);
return v_res_748_;
}
}
static lean_object* _init_lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__1___redArg___closed__0(void){
_start:
{
lean_object* v___x_749_; lean_object* v___x_750_; lean_object* v___x_751_; 
v___x_749_ = lean_unsigned_to_nat(32u);
v___x_750_ = lean_mk_empty_array_with_capacity(v___x_749_);
v___x_751_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_751_, 0, v___x_750_);
return v___x_751_;
}
}
static lean_object* _init_lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__1___redArg___closed__1(void){
_start:
{
size_t v___x_752_; lean_object* v___x_753_; lean_object* v___x_754_; lean_object* v___x_755_; lean_object* v___x_756_; lean_object* v___x_757_; 
v___x_752_ = ((size_t)5ULL);
v___x_753_ = lean_unsigned_to_nat(0u);
v___x_754_ = lean_unsigned_to_nat(32u);
v___x_755_ = lean_mk_empty_array_with_capacity(v___x_754_);
v___x_756_ = lean_obj_once(&lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__1___redArg___closed__0, &lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__1___redArg___closed__0_once, _init_lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__1___redArg___closed__0);
v___x_757_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_757_, 0, v___x_756_);
lean_ctor_set(v___x_757_, 1, v___x_755_);
lean_ctor_set(v___x_757_, 2, v___x_753_);
lean_ctor_set(v___x_757_, 3, v___x_753_);
lean_ctor_set_usize(v___x_757_, 4, v___x_752_);
return v___x_757_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__1___redArg(lean_object* v___y_758_){
_start:
{
lean_object* v___x_760_; lean_object* v_traceState_761_; lean_object* v_traces_762_; lean_object* v___x_763_; lean_object* v_traceState_764_; lean_object* v_env_765_; lean_object* v_nextMacroScope_766_; lean_object* v_ngen_767_; lean_object* v_auxDeclNGen_768_; lean_object* v_cache_769_; lean_object* v_messages_770_; lean_object* v_infoState_771_; lean_object* v_snapshotTasks_772_; lean_object* v___x_774_; uint8_t v_isShared_775_; uint8_t v_isSharedCheck_791_; 
v___x_760_ = lean_st_ref_get(v___y_758_);
v_traceState_761_ = lean_ctor_get(v___x_760_, 4);
lean_inc_ref(v_traceState_761_);
lean_dec(v___x_760_);
v_traces_762_ = lean_ctor_get(v_traceState_761_, 0);
lean_inc_ref(v_traces_762_);
lean_dec_ref(v_traceState_761_);
v___x_763_ = lean_st_ref_take(v___y_758_);
v_traceState_764_ = lean_ctor_get(v___x_763_, 4);
v_env_765_ = lean_ctor_get(v___x_763_, 0);
v_nextMacroScope_766_ = lean_ctor_get(v___x_763_, 1);
v_ngen_767_ = lean_ctor_get(v___x_763_, 2);
v_auxDeclNGen_768_ = lean_ctor_get(v___x_763_, 3);
v_cache_769_ = lean_ctor_get(v___x_763_, 5);
v_messages_770_ = lean_ctor_get(v___x_763_, 6);
v_infoState_771_ = lean_ctor_get(v___x_763_, 7);
v_snapshotTasks_772_ = lean_ctor_get(v___x_763_, 8);
v_isSharedCheck_791_ = !lean_is_exclusive(v___x_763_);
if (v_isSharedCheck_791_ == 0)
{
v___x_774_ = v___x_763_;
v_isShared_775_ = v_isSharedCheck_791_;
goto v_resetjp_773_;
}
else
{
lean_inc(v_snapshotTasks_772_);
lean_inc(v_infoState_771_);
lean_inc(v_messages_770_);
lean_inc(v_cache_769_);
lean_inc(v_traceState_764_);
lean_inc(v_auxDeclNGen_768_);
lean_inc(v_ngen_767_);
lean_inc(v_nextMacroScope_766_);
lean_inc(v_env_765_);
lean_dec(v___x_763_);
v___x_774_ = lean_box(0);
v_isShared_775_ = v_isSharedCheck_791_;
goto v_resetjp_773_;
}
v_resetjp_773_:
{
uint64_t v_tid_776_; lean_object* v___x_778_; uint8_t v_isShared_779_; uint8_t v_isSharedCheck_789_; 
v_tid_776_ = lean_ctor_get_uint64(v_traceState_764_, sizeof(void*)*1);
v_isSharedCheck_789_ = !lean_is_exclusive(v_traceState_764_);
if (v_isSharedCheck_789_ == 0)
{
lean_object* v_unused_790_; 
v_unused_790_ = lean_ctor_get(v_traceState_764_, 0);
lean_dec(v_unused_790_);
v___x_778_ = v_traceState_764_;
v_isShared_779_ = v_isSharedCheck_789_;
goto v_resetjp_777_;
}
else
{
lean_dec(v_traceState_764_);
v___x_778_ = lean_box(0);
v_isShared_779_ = v_isSharedCheck_789_;
goto v_resetjp_777_;
}
v_resetjp_777_:
{
lean_object* v___x_780_; lean_object* v___x_782_; 
v___x_780_ = lean_obj_once(&lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__1___redArg___closed__1, &lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__1___redArg___closed__1_once, _init_lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__1___redArg___closed__1);
if (v_isShared_779_ == 0)
{
lean_ctor_set(v___x_778_, 0, v___x_780_);
v___x_782_ = v___x_778_;
goto v_reusejp_781_;
}
else
{
lean_object* v_reuseFailAlloc_788_; 
v_reuseFailAlloc_788_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_788_, 0, v___x_780_);
lean_ctor_set_uint64(v_reuseFailAlloc_788_, sizeof(void*)*1, v_tid_776_);
v___x_782_ = v_reuseFailAlloc_788_;
goto v_reusejp_781_;
}
v_reusejp_781_:
{
lean_object* v___x_784_; 
if (v_isShared_775_ == 0)
{
lean_ctor_set(v___x_774_, 4, v___x_782_);
v___x_784_ = v___x_774_;
goto v_reusejp_783_;
}
else
{
lean_object* v_reuseFailAlloc_787_; 
v_reuseFailAlloc_787_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_787_, 0, v_env_765_);
lean_ctor_set(v_reuseFailAlloc_787_, 1, v_nextMacroScope_766_);
lean_ctor_set(v_reuseFailAlloc_787_, 2, v_ngen_767_);
lean_ctor_set(v_reuseFailAlloc_787_, 3, v_auxDeclNGen_768_);
lean_ctor_set(v_reuseFailAlloc_787_, 4, v___x_782_);
lean_ctor_set(v_reuseFailAlloc_787_, 5, v_cache_769_);
lean_ctor_set(v_reuseFailAlloc_787_, 6, v_messages_770_);
lean_ctor_set(v_reuseFailAlloc_787_, 7, v_infoState_771_);
lean_ctor_set(v_reuseFailAlloc_787_, 8, v_snapshotTasks_772_);
v___x_784_ = v_reuseFailAlloc_787_;
goto v_reusejp_783_;
}
v_reusejp_783_:
{
lean_object* v___x_785_; lean_object* v___x_786_; 
v___x_785_ = lean_st_ref_set(v___y_758_, v___x_784_);
v___x_786_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_786_, 0, v_traces_762_);
return v___x_786_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__1___redArg___boxed(lean_object* v___y_792_, lean_object* v___y_793_){
_start:
{
lean_object* v_res_794_; 
v_res_794_ = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__1___redArg(v___y_792_);
lean_dec(v___y_792_);
return v_res_794_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__1(lean_object* v___y_795_, lean_object* v___y_796_, lean_object* v___y_797_, lean_object* v___y_798_){
_start:
{
lean_object* v___x_800_; 
v___x_800_ = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__1___redArg(v___y_798_);
return v___x_800_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__1___boxed(lean_object* v___y_801_, lean_object* v___y_802_, lean_object* v___y_803_, lean_object* v___y_804_, lean_object* v___y_805_){
_start:
{
lean_object* v_res_806_; 
v_res_806_ = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__1(v___y_801_, v___y_802_, v___y_803_, v___y_804_);
lean_dec(v___y_804_);
lean_dec_ref(v___y_803_);
lean_dec(v___y_802_);
lean_dec_ref(v___y_801_);
return v_res_806_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_Option_get___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__2(lean_object* v_opts_807_, lean_object* v_opt_808_){
_start:
{
lean_object* v_name_809_; lean_object* v_defValue_810_; lean_object* v_map_811_; lean_object* v___x_812_; 
v_name_809_ = lean_ctor_get(v_opt_808_, 0);
v_defValue_810_ = lean_ctor_get(v_opt_808_, 1);
v_map_811_ = lean_ctor_get(v_opts_807_, 0);
v___x_812_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_811_, v_name_809_);
if (lean_obj_tag(v___x_812_) == 0)
{
uint8_t v___x_813_; 
v___x_813_ = lean_unbox(v_defValue_810_);
return v___x_813_;
}
else
{
lean_object* v_val_814_; 
v_val_814_ = lean_ctor_get(v___x_812_, 0);
lean_inc(v_val_814_);
lean_dec_ref_known(v___x_812_, 1);
if (lean_obj_tag(v_val_814_) == 1)
{
uint8_t v_v_815_; 
v_v_815_ = lean_ctor_get_uint8(v_val_814_, 0);
lean_dec_ref_known(v_val_814_, 0);
return v_v_815_;
}
else
{
uint8_t v___x_816_; 
lean_dec(v_val_814_);
v___x_816_ = lean_unbox(v_defValue_810_);
return v___x_816_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__2___boxed(lean_object* v_opts_817_, lean_object* v_opt_818_){
_start:
{
uint8_t v_res_819_; lean_object* v_r_820_; 
v_res_819_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__2(v_opts_817_, v_opt_818_);
lean_dec_ref(v_opt_818_);
lean_dec_ref(v_opts_817_);
v_r_820_ = lean_box(v_res_819_);
return v_r_820_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder___lam__0(lean_object* v_uTactic_821_, lean_object* v_x_822_, lean_object* v___y_823_, lean_object* v___y_824_, lean_object* v___y_825_, lean_object* v___y_826_){
_start:
{
lean_object* v___x_828_; lean_object* v___x_829_; 
v___x_828_ = l_Lean_MessageData_ofSyntax(v_uTactic_821_);
v___x_829_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_829_, 0, v___x_828_);
return v___x_829_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder___lam__0___boxed(lean_object* v_uTactic_830_, lean_object* v_x_831_, lean_object* v___y_832_, lean_object* v___y_833_, lean_object* v___y_834_, lean_object* v___y_835_, lean_object* v___y_836_){
_start:
{
lean_object* v_res_837_; 
v_res_837_ = lp_aesop___private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder___lam__0(v_uTactic_830_, v_x_831_, v___y_832_, v___y_833_, v___y_834_, v___y_835_);
lean_dec(v___y_835_);
lean_dec_ref(v___y_834_);
lean_dec(v___y_833_);
lean_dec_ref(v___y_832_);
lean_dec_ref(v_x_831_);
return v_res_837_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__3_spec__6(lean_object* v_opts_838_, lean_object* v_opt_839_){
_start:
{
lean_object* v_name_840_; lean_object* v_defValue_841_; lean_object* v_map_842_; lean_object* v___x_843_; 
v_name_840_ = lean_ctor_get(v_opt_839_, 0);
v_defValue_841_ = lean_ctor_get(v_opt_839_, 1);
v_map_842_ = lean_ctor_get(v_opts_838_, 0);
v___x_843_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_842_, v_name_840_);
if (lean_obj_tag(v___x_843_) == 0)
{
lean_inc(v_defValue_841_);
return v_defValue_841_;
}
else
{
lean_object* v_val_844_; 
v_val_844_ = lean_ctor_get(v___x_843_, 0);
lean_inc(v_val_844_);
lean_dec_ref_known(v___x_843_, 1);
if (lean_obj_tag(v_val_844_) == 3)
{
lean_object* v_v_845_; 
v_v_845_ = lean_ctor_get(v_val_844_, 0);
lean_inc(v_v_845_);
lean_dec_ref_known(v_val_844_, 1);
return v_v_845_;
}
else
{
lean_dec(v_val_844_);
lean_inc(v_defValue_841_);
return v_defValue_841_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__3_spec__6___boxed(lean_object* v_opts_846_, lean_object* v_opt_847_){
_start:
{
lean_object* v_res_848_; 
v_res_848_ = lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__3_spec__6(v_opts_846_, v_opt_847_);
lean_dec_ref(v_opt_847_);
lean_dec_ref(v_opts_846_);
return v_res_848_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__3_spec__5(lean_object* v_e_849_){
_start:
{
if (lean_obj_tag(v_e_849_) == 0)
{
uint8_t v___x_850_; 
v___x_850_ = 2;
return v___x_850_;
}
else
{
lean_object* v_a_851_; 
v_a_851_ = lean_ctor_get(v_e_849_, 0);
if (lean_obj_tag(v_a_851_) == 0)
{
uint8_t v___x_852_; 
v___x_852_ = 1;
return v___x_852_;
}
else
{
uint8_t v___x_853_; 
v___x_853_ = 0;
return v___x_853_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__3_spec__5___boxed(lean_object* v_e_854_){
_start:
{
uint8_t v_res_855_; lean_object* v_r_856_; 
v_res_855_ = lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__3_spec__5(v_e_854_);
lean_dec_ref(v_e_854_);
v_r_856_ = lean_box(v_res_855_);
return v_r_856_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__3_spec__3_spec__4(size_t v_sz_857_, size_t v_i_858_, lean_object* v_bs_859_){
_start:
{
uint8_t v___x_860_; 
v___x_860_ = lean_usize_dec_lt(v_i_858_, v_sz_857_);
if (v___x_860_ == 0)
{
return v_bs_859_;
}
else
{
lean_object* v_v_861_; lean_object* v_msg_862_; lean_object* v___x_863_; lean_object* v_bs_x27_864_; size_t v___x_865_; size_t v___x_866_; lean_object* v___x_867_; 
v_v_861_ = lean_array_uget_borrowed(v_bs_859_, v_i_858_);
v_msg_862_ = lean_ctor_get(v_v_861_, 1);
lean_inc_ref(v_msg_862_);
v___x_863_ = lean_unsigned_to_nat(0u);
v_bs_x27_864_ = lean_array_uset(v_bs_859_, v_i_858_, v___x_863_);
v___x_865_ = ((size_t)1ULL);
v___x_866_ = lean_usize_add(v_i_858_, v___x_865_);
v___x_867_ = lean_array_uset(v_bs_x27_864_, v_i_858_, v_msg_862_);
v_i_858_ = v___x_866_;
v_bs_859_ = v___x_867_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__3_spec__3_spec__4___boxed(lean_object* v_sz_869_, lean_object* v_i_870_, lean_object* v_bs_871_){
_start:
{
size_t v_sz_boxed_872_; size_t v_i_boxed_873_; lean_object* v_res_874_; 
v_sz_boxed_872_ = lean_unbox_usize(v_sz_869_);
lean_dec(v_sz_869_);
v_i_boxed_873_ = lean_unbox_usize(v_i_870_);
lean_dec(v_i_870_);
v_res_874_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__3_spec__3_spec__4(v_sz_boxed_872_, v_i_boxed_873_, v_bs_871_);
return v_res_874_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__3_spec__3(lean_object* v_oldTraces_875_, lean_object* v_data_876_, lean_object* v_ref_877_, lean_object* v_msg_878_, lean_object* v___y_879_, lean_object* v___y_880_, lean_object* v___y_881_, lean_object* v___y_882_){
_start:
{
lean_object* v_fileName_884_; lean_object* v_fileMap_885_; lean_object* v_options_886_; lean_object* v_currRecDepth_887_; lean_object* v_maxRecDepth_888_; lean_object* v_ref_889_; lean_object* v_currNamespace_890_; lean_object* v_openDecls_891_; lean_object* v_initHeartbeats_892_; lean_object* v_maxHeartbeats_893_; lean_object* v_quotContext_894_; lean_object* v_currMacroScope_895_; uint8_t v_diag_896_; lean_object* v_cancelTk_x3f_897_; uint8_t v_suppressElabErrors_898_; lean_object* v_inheritedTraceOptions_899_; lean_object* v___x_900_; lean_object* v_traceState_901_; lean_object* v_traces_902_; lean_object* v_ref_903_; lean_object* v___x_904_; lean_object* v___x_905_; size_t v_sz_906_; size_t v___x_907_; lean_object* v___x_908_; lean_object* v_msg_909_; lean_object* v___x_910_; lean_object* v_a_911_; lean_object* v___x_913_; uint8_t v_isShared_914_; uint8_t v_isSharedCheck_948_; 
v_fileName_884_ = lean_ctor_get(v___y_881_, 0);
v_fileMap_885_ = lean_ctor_get(v___y_881_, 1);
v_options_886_ = lean_ctor_get(v___y_881_, 2);
v_currRecDepth_887_ = lean_ctor_get(v___y_881_, 3);
v_maxRecDepth_888_ = lean_ctor_get(v___y_881_, 4);
v_ref_889_ = lean_ctor_get(v___y_881_, 5);
v_currNamespace_890_ = lean_ctor_get(v___y_881_, 6);
v_openDecls_891_ = lean_ctor_get(v___y_881_, 7);
v_initHeartbeats_892_ = lean_ctor_get(v___y_881_, 8);
v_maxHeartbeats_893_ = lean_ctor_get(v___y_881_, 9);
v_quotContext_894_ = lean_ctor_get(v___y_881_, 10);
v_currMacroScope_895_ = lean_ctor_get(v___y_881_, 11);
v_diag_896_ = lean_ctor_get_uint8(v___y_881_, sizeof(void*)*14);
v_cancelTk_x3f_897_ = lean_ctor_get(v___y_881_, 12);
v_suppressElabErrors_898_ = lean_ctor_get_uint8(v___y_881_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_899_ = lean_ctor_get(v___y_881_, 13);
v___x_900_ = lean_st_ref_get(v___y_882_);
v_traceState_901_ = lean_ctor_get(v___x_900_, 4);
lean_inc_ref(v_traceState_901_);
lean_dec(v___x_900_);
v_traces_902_ = lean_ctor_get(v_traceState_901_, 0);
lean_inc_ref(v_traces_902_);
lean_dec_ref(v_traceState_901_);
v_ref_903_ = l_Lean_replaceRef(v_ref_877_, v_ref_889_);
lean_inc_ref(v_inheritedTraceOptions_899_);
lean_inc(v_cancelTk_x3f_897_);
lean_inc(v_currMacroScope_895_);
lean_inc(v_quotContext_894_);
lean_inc(v_maxHeartbeats_893_);
lean_inc(v_initHeartbeats_892_);
lean_inc(v_openDecls_891_);
lean_inc(v_currNamespace_890_);
lean_inc(v_maxRecDepth_888_);
lean_inc(v_currRecDepth_887_);
lean_inc_ref(v_options_886_);
lean_inc_ref(v_fileMap_885_);
lean_inc_ref(v_fileName_884_);
v___x_904_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_904_, 0, v_fileName_884_);
lean_ctor_set(v___x_904_, 1, v_fileMap_885_);
lean_ctor_set(v___x_904_, 2, v_options_886_);
lean_ctor_set(v___x_904_, 3, v_currRecDepth_887_);
lean_ctor_set(v___x_904_, 4, v_maxRecDepth_888_);
lean_ctor_set(v___x_904_, 5, v_ref_903_);
lean_ctor_set(v___x_904_, 6, v_currNamespace_890_);
lean_ctor_set(v___x_904_, 7, v_openDecls_891_);
lean_ctor_set(v___x_904_, 8, v_initHeartbeats_892_);
lean_ctor_set(v___x_904_, 9, v_maxHeartbeats_893_);
lean_ctor_set(v___x_904_, 10, v_quotContext_894_);
lean_ctor_set(v___x_904_, 11, v_currMacroScope_895_);
lean_ctor_set(v___x_904_, 12, v_cancelTk_x3f_897_);
lean_ctor_set(v___x_904_, 13, v_inheritedTraceOptions_899_);
lean_ctor_set_uint8(v___x_904_, sizeof(void*)*14, v_diag_896_);
lean_ctor_set_uint8(v___x_904_, sizeof(void*)*14 + 1, v_suppressElabErrors_898_);
v___x_905_ = l_Lean_PersistentArray_toArray___redArg(v_traces_902_);
lean_dec_ref(v_traces_902_);
v_sz_906_ = lean_array_size(v___x_905_);
v___x_907_ = ((size_t)0ULL);
v___x_908_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__3_spec__3_spec__4(v_sz_906_, v___x_907_, v___x_905_);
v_msg_909_ = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(v_msg_909_, 0, v_data_876_);
lean_ctor_set(v_msg_909_, 1, v_msg_878_);
lean_ctor_set(v_msg_909_, 2, v___x_908_);
v___x_910_ = lp_aesop_Lean_addMessageContextFull___at___00__private_Aesop_Script_Step_0__Aesop_Script_Step_validate_fmtGoals_spec__1(v_msg_909_, v___y_879_, v___y_880_, v___x_904_, v___y_882_);
lean_dec_ref_known(v___x_904_, 14);
v_a_911_ = lean_ctor_get(v___x_910_, 0);
v_isSharedCheck_948_ = !lean_is_exclusive(v___x_910_);
if (v_isSharedCheck_948_ == 0)
{
v___x_913_ = v___x_910_;
v_isShared_914_ = v_isSharedCheck_948_;
goto v_resetjp_912_;
}
else
{
lean_inc(v_a_911_);
lean_dec(v___x_910_);
v___x_913_ = lean_box(0);
v_isShared_914_ = v_isSharedCheck_948_;
goto v_resetjp_912_;
}
v_resetjp_912_:
{
lean_object* v___x_915_; lean_object* v_traceState_916_; lean_object* v_env_917_; lean_object* v_nextMacroScope_918_; lean_object* v_ngen_919_; lean_object* v_auxDeclNGen_920_; lean_object* v_cache_921_; lean_object* v_messages_922_; lean_object* v_infoState_923_; lean_object* v_snapshotTasks_924_; lean_object* v___x_926_; uint8_t v_isShared_927_; uint8_t v_isSharedCheck_947_; 
v___x_915_ = lean_st_ref_take(v___y_882_);
v_traceState_916_ = lean_ctor_get(v___x_915_, 4);
v_env_917_ = lean_ctor_get(v___x_915_, 0);
v_nextMacroScope_918_ = lean_ctor_get(v___x_915_, 1);
v_ngen_919_ = lean_ctor_get(v___x_915_, 2);
v_auxDeclNGen_920_ = lean_ctor_get(v___x_915_, 3);
v_cache_921_ = lean_ctor_get(v___x_915_, 5);
v_messages_922_ = lean_ctor_get(v___x_915_, 6);
v_infoState_923_ = lean_ctor_get(v___x_915_, 7);
v_snapshotTasks_924_ = lean_ctor_get(v___x_915_, 8);
v_isSharedCheck_947_ = !lean_is_exclusive(v___x_915_);
if (v_isSharedCheck_947_ == 0)
{
v___x_926_ = v___x_915_;
v_isShared_927_ = v_isSharedCheck_947_;
goto v_resetjp_925_;
}
else
{
lean_inc(v_snapshotTasks_924_);
lean_inc(v_infoState_923_);
lean_inc(v_messages_922_);
lean_inc(v_cache_921_);
lean_inc(v_traceState_916_);
lean_inc(v_auxDeclNGen_920_);
lean_inc(v_ngen_919_);
lean_inc(v_nextMacroScope_918_);
lean_inc(v_env_917_);
lean_dec(v___x_915_);
v___x_926_ = lean_box(0);
v_isShared_927_ = v_isSharedCheck_947_;
goto v_resetjp_925_;
}
v_resetjp_925_:
{
uint64_t v_tid_928_; lean_object* v___x_930_; uint8_t v_isShared_931_; uint8_t v_isSharedCheck_945_; 
v_tid_928_ = lean_ctor_get_uint64(v_traceState_916_, sizeof(void*)*1);
v_isSharedCheck_945_ = !lean_is_exclusive(v_traceState_916_);
if (v_isSharedCheck_945_ == 0)
{
lean_object* v_unused_946_; 
v_unused_946_ = lean_ctor_get(v_traceState_916_, 0);
lean_dec(v_unused_946_);
v___x_930_ = v_traceState_916_;
v_isShared_931_ = v_isSharedCheck_945_;
goto v_resetjp_929_;
}
else
{
lean_dec(v_traceState_916_);
v___x_930_ = lean_box(0);
v_isShared_931_ = v_isSharedCheck_945_;
goto v_resetjp_929_;
}
v_resetjp_929_:
{
lean_object* v___x_932_; lean_object* v___x_933_; lean_object* v___x_935_; 
v___x_932_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_932_, 0, v_ref_877_);
lean_ctor_set(v___x_932_, 1, v_a_911_);
v___x_933_ = l_Lean_PersistentArray_push___redArg(v_oldTraces_875_, v___x_932_);
if (v_isShared_931_ == 0)
{
lean_ctor_set(v___x_930_, 0, v___x_933_);
v___x_935_ = v___x_930_;
goto v_reusejp_934_;
}
else
{
lean_object* v_reuseFailAlloc_944_; 
v_reuseFailAlloc_944_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_944_, 0, v___x_933_);
lean_ctor_set_uint64(v_reuseFailAlloc_944_, sizeof(void*)*1, v_tid_928_);
v___x_935_ = v_reuseFailAlloc_944_;
goto v_reusejp_934_;
}
v_reusejp_934_:
{
lean_object* v___x_937_; 
if (v_isShared_927_ == 0)
{
lean_ctor_set(v___x_926_, 4, v___x_935_);
v___x_937_ = v___x_926_;
goto v_reusejp_936_;
}
else
{
lean_object* v_reuseFailAlloc_943_; 
v_reuseFailAlloc_943_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_943_, 0, v_env_917_);
lean_ctor_set(v_reuseFailAlloc_943_, 1, v_nextMacroScope_918_);
lean_ctor_set(v_reuseFailAlloc_943_, 2, v_ngen_919_);
lean_ctor_set(v_reuseFailAlloc_943_, 3, v_auxDeclNGen_920_);
lean_ctor_set(v_reuseFailAlloc_943_, 4, v___x_935_);
lean_ctor_set(v_reuseFailAlloc_943_, 5, v_cache_921_);
lean_ctor_set(v_reuseFailAlloc_943_, 6, v_messages_922_);
lean_ctor_set(v_reuseFailAlloc_943_, 7, v_infoState_923_);
lean_ctor_set(v_reuseFailAlloc_943_, 8, v_snapshotTasks_924_);
v___x_937_ = v_reuseFailAlloc_943_;
goto v_reusejp_936_;
}
v_reusejp_936_:
{
lean_object* v___x_938_; lean_object* v___x_939_; lean_object* v___x_941_; 
v___x_938_ = lean_st_ref_set(v___y_882_, v___x_937_);
v___x_939_ = lean_box(0);
if (v_isShared_914_ == 0)
{
lean_ctor_set(v___x_913_, 0, v___x_939_);
v___x_941_ = v___x_913_;
goto v_reusejp_940_;
}
else
{
lean_object* v_reuseFailAlloc_942_; 
v_reuseFailAlloc_942_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_942_, 0, v___x_939_);
v___x_941_ = v_reuseFailAlloc_942_;
goto v_reusejp_940_;
}
v_reusejp_940_:
{
return v___x_941_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__3_spec__3___boxed(lean_object* v_oldTraces_949_, lean_object* v_data_950_, lean_object* v_ref_951_, lean_object* v_msg_952_, lean_object* v___y_953_, lean_object* v___y_954_, lean_object* v___y_955_, lean_object* v___y_956_, lean_object* v___y_957_){
_start:
{
lean_object* v_res_958_; 
v_res_958_ = lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__3_spec__3(v_oldTraces_949_, v_data_950_, v_ref_951_, v_msg_952_, v___y_953_, v___y_954_, v___y_955_, v___y_956_);
lean_dec(v___y_956_);
lean_dec_ref(v___y_955_);
lean_dec(v___y_954_);
lean_dec_ref(v___y_953_);
return v_res_958_;
}
}
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__3_spec__4___redArg(lean_object* v_x_959_){
_start:
{
if (lean_obj_tag(v_x_959_) == 0)
{
lean_object* v_a_961_; lean_object* v___x_963_; uint8_t v_isShared_964_; uint8_t v_isSharedCheck_968_; 
v_a_961_ = lean_ctor_get(v_x_959_, 0);
v_isSharedCheck_968_ = !lean_is_exclusive(v_x_959_);
if (v_isSharedCheck_968_ == 0)
{
v___x_963_ = v_x_959_;
v_isShared_964_ = v_isSharedCheck_968_;
goto v_resetjp_962_;
}
else
{
lean_inc(v_a_961_);
lean_dec(v_x_959_);
v___x_963_ = lean_box(0);
v_isShared_964_ = v_isSharedCheck_968_;
goto v_resetjp_962_;
}
v_resetjp_962_:
{
lean_object* v___x_966_; 
if (v_isShared_964_ == 0)
{
lean_ctor_set_tag(v___x_963_, 1);
v___x_966_ = v___x_963_;
goto v_reusejp_965_;
}
else
{
lean_object* v_reuseFailAlloc_967_; 
v_reuseFailAlloc_967_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_967_, 0, v_a_961_);
v___x_966_ = v_reuseFailAlloc_967_;
goto v_reusejp_965_;
}
v_reusejp_965_:
{
return v___x_966_;
}
}
}
else
{
lean_object* v_a_969_; lean_object* v___x_971_; uint8_t v_isShared_972_; uint8_t v_isSharedCheck_976_; 
v_a_969_ = lean_ctor_get(v_x_959_, 0);
v_isSharedCheck_976_ = !lean_is_exclusive(v_x_959_);
if (v_isSharedCheck_976_ == 0)
{
v___x_971_ = v_x_959_;
v_isShared_972_ = v_isSharedCheck_976_;
goto v_resetjp_970_;
}
else
{
lean_inc(v_a_969_);
lean_dec(v_x_959_);
v___x_971_ = lean_box(0);
v_isShared_972_ = v_isSharedCheck_976_;
goto v_resetjp_970_;
}
v_resetjp_970_:
{
lean_object* v___x_974_; 
if (v_isShared_972_ == 0)
{
lean_ctor_set_tag(v___x_971_, 0);
v___x_974_ = v___x_971_;
goto v_reusejp_973_;
}
else
{
lean_object* v_reuseFailAlloc_975_; 
v_reuseFailAlloc_975_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_975_, 0, v_a_969_);
v___x_974_ = v_reuseFailAlloc_975_;
goto v_reusejp_973_;
}
v_reusejp_973_:
{
return v___x_974_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__3_spec__4___redArg___boxed(lean_object* v_x_977_, lean_object* v___y_978_){
_start:
{
lean_object* v_res_979_; 
v_res_979_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__3_spec__4___redArg(v_x_977_);
return v_res_979_;
}
}
static double _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__3___closed__0(void){
_start:
{
lean_object* v___x_980_; double v___x_981_; 
v___x_980_ = lean_unsigned_to_nat(0u);
v___x_981_ = lean_float_of_nat(v___x_980_);
return v___x_981_;
}
}
static lean_object* _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__3___closed__2(void){
_start:
{
lean_object* v___x_983_; lean_object* v___x_984_; 
v___x_983_ = ((lean_object*)(lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__3___closed__1));
v___x_984_ = l_Lean_stringToMessageData(v___x_983_);
return v___x_984_;
}
}
static double _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__3___closed__3(void){
_start:
{
lean_object* v___x_985_; double v___x_986_; 
v___x_985_ = lean_unsigned_to_nat(1000u);
v___x_986_ = lean_float_of_nat(v___x_985_);
return v___x_986_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__3(lean_object* v_cls_987_, uint8_t v_collapsed_988_, lean_object* v_tag_989_, lean_object* v_opts_990_, uint8_t v_clsEnabled_991_, lean_object* v_oldTraces_992_, lean_object* v_msg_993_, lean_object* v_resStartStop_994_, lean_object* v___y_995_, lean_object* v___y_996_, lean_object* v___y_997_, lean_object* v___y_998_){
_start:
{
lean_object* v_fst_1000_; lean_object* v_snd_1001_; lean_object* v___y_1003_; lean_object* v___y_1004_; lean_object* v_data_1005_; lean_object* v_fst_1016_; lean_object* v_snd_1017_; lean_object* v___x_1018_; uint8_t v___x_1019_; lean_object* v___y_1021_; lean_object* v_a_1022_; uint8_t v___y_1037_; double v___y_1068_; 
v_fst_1000_ = lean_ctor_get(v_resStartStop_994_, 0);
lean_inc(v_fst_1000_);
v_snd_1001_ = lean_ctor_get(v_resStartStop_994_, 1);
lean_inc(v_snd_1001_);
lean_dec_ref(v_resStartStop_994_);
v_fst_1016_ = lean_ctor_get(v_snd_1001_, 0);
lean_inc(v_fst_1016_);
v_snd_1017_ = lean_ctor_get(v_snd_1001_, 1);
lean_inc(v_snd_1017_);
lean_dec(v_snd_1001_);
v___x_1018_ = l_Lean_trace_profiler;
v___x_1019_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__2(v_opts_990_, v___x_1018_);
if (v___x_1019_ == 0)
{
v___y_1037_ = v___x_1019_;
goto v___jp_1036_;
}
else
{
lean_object* v___x_1073_; uint8_t v___x_1074_; 
v___x_1073_ = l_Lean_trace_profiler_useHeartbeats;
v___x_1074_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__2(v_opts_990_, v___x_1073_);
if (v___x_1074_ == 0)
{
lean_object* v___x_1075_; lean_object* v___x_1076_; double v___x_1077_; double v___x_1078_; double v___x_1079_; 
v___x_1075_ = l_Lean_trace_profiler_threshold;
v___x_1076_ = lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__3_spec__6(v_opts_990_, v___x_1075_);
v___x_1077_ = lean_float_of_nat(v___x_1076_);
v___x_1078_ = lean_float_once(&lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__3___closed__3, &lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__3___closed__3_once, _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__3___closed__3);
v___x_1079_ = lean_float_div(v___x_1077_, v___x_1078_);
v___y_1068_ = v___x_1079_;
goto v___jp_1067_;
}
else
{
lean_object* v___x_1080_; lean_object* v___x_1081_; double v___x_1082_; 
v___x_1080_ = l_Lean_trace_profiler_threshold;
v___x_1081_ = lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__3_spec__6(v_opts_990_, v___x_1080_);
v___x_1082_ = lean_float_of_nat(v___x_1081_);
v___y_1068_ = v___x_1082_;
goto v___jp_1067_;
}
}
v___jp_1002_:
{
lean_object* v___x_1006_; 
lean_inc(v___y_1004_);
v___x_1006_ = lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__3_spec__3(v_oldTraces_992_, v_data_1005_, v___y_1004_, v___y_1003_, v___y_995_, v___y_996_, v___y_997_, v___y_998_);
if (lean_obj_tag(v___x_1006_) == 0)
{
lean_object* v___x_1007_; 
lean_dec_ref_known(v___x_1006_, 1);
v___x_1007_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__3_spec__4___redArg(v_fst_1000_);
return v___x_1007_;
}
else
{
lean_object* v_a_1008_; lean_object* v___x_1010_; uint8_t v_isShared_1011_; uint8_t v_isSharedCheck_1015_; 
lean_dec(v_fst_1000_);
v_a_1008_ = lean_ctor_get(v___x_1006_, 0);
v_isSharedCheck_1015_ = !lean_is_exclusive(v___x_1006_);
if (v_isSharedCheck_1015_ == 0)
{
v___x_1010_ = v___x_1006_;
v_isShared_1011_ = v_isSharedCheck_1015_;
goto v_resetjp_1009_;
}
else
{
lean_inc(v_a_1008_);
lean_dec(v___x_1006_);
v___x_1010_ = lean_box(0);
v_isShared_1011_ = v_isSharedCheck_1015_;
goto v_resetjp_1009_;
}
v_resetjp_1009_:
{
lean_object* v___x_1013_; 
if (v_isShared_1011_ == 0)
{
v___x_1013_ = v___x_1010_;
goto v_reusejp_1012_;
}
else
{
lean_object* v_reuseFailAlloc_1014_; 
v_reuseFailAlloc_1014_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1014_, 0, v_a_1008_);
v___x_1013_ = v_reuseFailAlloc_1014_;
goto v_reusejp_1012_;
}
v_reusejp_1012_:
{
return v___x_1013_;
}
}
}
}
v___jp_1020_:
{
uint8_t v_result_1023_; lean_object* v___x_1024_; lean_object* v___x_1025_; double v___x_1026_; lean_object* v_data_1027_; 
v_result_1023_ = lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__3_spec__5(v_fst_1000_);
v___x_1024_ = lean_box(v_result_1023_);
v___x_1025_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1025_, 0, v___x_1024_);
v___x_1026_ = lean_float_once(&lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__3___closed__0, &lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__3___closed__0_once, _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__3___closed__0);
lean_inc_ref(v_tag_989_);
lean_inc_ref(v___x_1025_);
lean_inc(v_cls_987_);
v_data_1027_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v_data_1027_, 0, v_cls_987_);
lean_ctor_set(v_data_1027_, 1, v___x_1025_);
lean_ctor_set(v_data_1027_, 2, v_tag_989_);
lean_ctor_set_float(v_data_1027_, sizeof(void*)*3, v___x_1026_);
lean_ctor_set_float(v_data_1027_, sizeof(void*)*3 + 8, v___x_1026_);
lean_ctor_set_uint8(v_data_1027_, sizeof(void*)*3 + 16, v_collapsed_988_);
if (v___x_1019_ == 0)
{
lean_dec_ref_known(v___x_1025_, 1);
lean_dec(v_snd_1017_);
lean_dec(v_fst_1016_);
lean_dec_ref(v_tag_989_);
lean_dec(v_cls_987_);
v___y_1003_ = v_a_1022_;
v___y_1004_ = v___y_1021_;
v_data_1005_ = v_data_1027_;
goto v___jp_1002_;
}
else
{
lean_object* v_data_1028_; double v___x_1029_; double v___x_1030_; 
lean_dec_ref_known(v_data_1027_, 3);
v_data_1028_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v_data_1028_, 0, v_cls_987_);
lean_ctor_set(v_data_1028_, 1, v___x_1025_);
lean_ctor_set(v_data_1028_, 2, v_tag_989_);
v___x_1029_ = lean_unbox_float(v_fst_1016_);
lean_dec(v_fst_1016_);
lean_ctor_set_float(v_data_1028_, sizeof(void*)*3, v___x_1029_);
v___x_1030_ = lean_unbox_float(v_snd_1017_);
lean_dec(v_snd_1017_);
lean_ctor_set_float(v_data_1028_, sizeof(void*)*3 + 8, v___x_1030_);
lean_ctor_set_uint8(v_data_1028_, sizeof(void*)*3 + 16, v_collapsed_988_);
v___y_1003_ = v_a_1022_;
v___y_1004_ = v___y_1021_;
v_data_1005_ = v_data_1028_;
goto v___jp_1002_;
}
}
v___jp_1031_:
{
lean_object* v_ref_1032_; lean_object* v___x_1033_; 
v_ref_1032_ = lean_ctor_get(v___y_997_, 5);
lean_inc(v___y_998_);
lean_inc_ref(v___y_997_);
lean_inc(v___y_996_);
lean_inc_ref(v___y_995_);
lean_inc(v_fst_1000_);
v___x_1033_ = lean_apply_6(v_msg_993_, v_fst_1000_, v___y_995_, v___y_996_, v___y_997_, v___y_998_, lean_box(0));
if (lean_obj_tag(v___x_1033_) == 0)
{
lean_object* v_a_1034_; 
v_a_1034_ = lean_ctor_get(v___x_1033_, 0);
lean_inc(v_a_1034_);
lean_dec_ref_known(v___x_1033_, 1);
v___y_1021_ = v_ref_1032_;
v_a_1022_ = v_a_1034_;
goto v___jp_1020_;
}
else
{
lean_object* v___x_1035_; 
lean_dec_ref_known(v___x_1033_, 1);
v___x_1035_ = lean_obj_once(&lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__3___closed__2, &lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__3___closed__2_once, _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__3___closed__2);
v___y_1021_ = v_ref_1032_;
v_a_1022_ = v___x_1035_;
goto v___jp_1020_;
}
}
v___jp_1036_:
{
if (v_clsEnabled_991_ == 0)
{
if (v___y_1037_ == 0)
{
lean_object* v___x_1038_; lean_object* v_traceState_1039_; lean_object* v_env_1040_; lean_object* v_nextMacroScope_1041_; lean_object* v_ngen_1042_; lean_object* v_auxDeclNGen_1043_; lean_object* v_cache_1044_; lean_object* v_messages_1045_; lean_object* v_infoState_1046_; lean_object* v_snapshotTasks_1047_; lean_object* v___x_1049_; uint8_t v_isShared_1050_; uint8_t v_isSharedCheck_1066_; 
lean_dec(v_snd_1017_);
lean_dec(v_fst_1016_);
lean_dec_ref(v_msg_993_);
lean_dec_ref(v_tag_989_);
lean_dec(v_cls_987_);
v___x_1038_ = lean_st_ref_take(v___y_998_);
v_traceState_1039_ = lean_ctor_get(v___x_1038_, 4);
v_env_1040_ = lean_ctor_get(v___x_1038_, 0);
v_nextMacroScope_1041_ = lean_ctor_get(v___x_1038_, 1);
v_ngen_1042_ = lean_ctor_get(v___x_1038_, 2);
v_auxDeclNGen_1043_ = lean_ctor_get(v___x_1038_, 3);
v_cache_1044_ = lean_ctor_get(v___x_1038_, 5);
v_messages_1045_ = lean_ctor_get(v___x_1038_, 6);
v_infoState_1046_ = lean_ctor_get(v___x_1038_, 7);
v_snapshotTasks_1047_ = lean_ctor_get(v___x_1038_, 8);
v_isSharedCheck_1066_ = !lean_is_exclusive(v___x_1038_);
if (v_isSharedCheck_1066_ == 0)
{
v___x_1049_ = v___x_1038_;
v_isShared_1050_ = v_isSharedCheck_1066_;
goto v_resetjp_1048_;
}
else
{
lean_inc(v_snapshotTasks_1047_);
lean_inc(v_infoState_1046_);
lean_inc(v_messages_1045_);
lean_inc(v_cache_1044_);
lean_inc(v_traceState_1039_);
lean_inc(v_auxDeclNGen_1043_);
lean_inc(v_ngen_1042_);
lean_inc(v_nextMacroScope_1041_);
lean_inc(v_env_1040_);
lean_dec(v___x_1038_);
v___x_1049_ = lean_box(0);
v_isShared_1050_ = v_isSharedCheck_1066_;
goto v_resetjp_1048_;
}
v_resetjp_1048_:
{
uint64_t v_tid_1051_; lean_object* v_traces_1052_; lean_object* v___x_1054_; uint8_t v_isShared_1055_; uint8_t v_isSharedCheck_1065_; 
v_tid_1051_ = lean_ctor_get_uint64(v_traceState_1039_, sizeof(void*)*1);
v_traces_1052_ = lean_ctor_get(v_traceState_1039_, 0);
v_isSharedCheck_1065_ = !lean_is_exclusive(v_traceState_1039_);
if (v_isSharedCheck_1065_ == 0)
{
v___x_1054_ = v_traceState_1039_;
v_isShared_1055_ = v_isSharedCheck_1065_;
goto v_resetjp_1053_;
}
else
{
lean_inc(v_traces_1052_);
lean_dec(v_traceState_1039_);
v___x_1054_ = lean_box(0);
v_isShared_1055_ = v_isSharedCheck_1065_;
goto v_resetjp_1053_;
}
v_resetjp_1053_:
{
lean_object* v___x_1056_; lean_object* v___x_1058_; 
v___x_1056_ = l_Lean_PersistentArray_append___redArg(v_oldTraces_992_, v_traces_1052_);
lean_dec_ref(v_traces_1052_);
if (v_isShared_1055_ == 0)
{
lean_ctor_set(v___x_1054_, 0, v___x_1056_);
v___x_1058_ = v___x_1054_;
goto v_reusejp_1057_;
}
else
{
lean_object* v_reuseFailAlloc_1064_; 
v_reuseFailAlloc_1064_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_1064_, 0, v___x_1056_);
lean_ctor_set_uint64(v_reuseFailAlloc_1064_, sizeof(void*)*1, v_tid_1051_);
v___x_1058_ = v_reuseFailAlloc_1064_;
goto v_reusejp_1057_;
}
v_reusejp_1057_:
{
lean_object* v___x_1060_; 
if (v_isShared_1050_ == 0)
{
lean_ctor_set(v___x_1049_, 4, v___x_1058_);
v___x_1060_ = v___x_1049_;
goto v_reusejp_1059_;
}
else
{
lean_object* v_reuseFailAlloc_1063_; 
v_reuseFailAlloc_1063_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_1063_, 0, v_env_1040_);
lean_ctor_set(v_reuseFailAlloc_1063_, 1, v_nextMacroScope_1041_);
lean_ctor_set(v_reuseFailAlloc_1063_, 2, v_ngen_1042_);
lean_ctor_set(v_reuseFailAlloc_1063_, 3, v_auxDeclNGen_1043_);
lean_ctor_set(v_reuseFailAlloc_1063_, 4, v___x_1058_);
lean_ctor_set(v_reuseFailAlloc_1063_, 5, v_cache_1044_);
lean_ctor_set(v_reuseFailAlloc_1063_, 6, v_messages_1045_);
lean_ctor_set(v_reuseFailAlloc_1063_, 7, v_infoState_1046_);
lean_ctor_set(v_reuseFailAlloc_1063_, 8, v_snapshotTasks_1047_);
v___x_1060_ = v_reuseFailAlloc_1063_;
goto v_reusejp_1059_;
}
v_reusejp_1059_:
{
lean_object* v___x_1061_; lean_object* v___x_1062_; 
v___x_1061_ = lean_st_ref_set(v___y_998_, v___x_1060_);
v___x_1062_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__3_spec__4___redArg(v_fst_1000_);
return v___x_1062_;
}
}
}
}
}
else
{
goto v___jp_1031_;
}
}
else
{
goto v___jp_1031_;
}
}
v___jp_1067_:
{
double v___x_1069_; double v___x_1070_; double v___x_1071_; uint8_t v___x_1072_; 
v___x_1069_ = lean_unbox_float(v_snd_1017_);
v___x_1070_ = lean_unbox_float(v_fst_1016_);
v___x_1071_ = lean_float_sub(v___x_1069_, v___x_1070_);
v___x_1072_ = lean_float_decLt(v___y_1068_, v___x_1071_);
v___y_1037_ = v___x_1072_;
goto v___jp_1036_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__3___boxed(lean_object* v_cls_1083_, lean_object* v_collapsed_1084_, lean_object* v_tag_1085_, lean_object* v_opts_1086_, lean_object* v_clsEnabled_1087_, lean_object* v_oldTraces_1088_, lean_object* v_msg_1089_, lean_object* v_resStartStop_1090_, lean_object* v___y_1091_, lean_object* v___y_1092_, lean_object* v___y_1093_, lean_object* v___y_1094_, lean_object* v___y_1095_){
_start:
{
uint8_t v_collapsed_boxed_1096_; uint8_t v_clsEnabled_boxed_1097_; lean_object* v_res_1098_; 
v_collapsed_boxed_1096_ = lean_unbox(v_collapsed_1084_);
v_clsEnabled_boxed_1097_ = lean_unbox(v_clsEnabled_1087_);
v_res_1098_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__3(v_cls_1083_, v_collapsed_boxed_1096_, v_tag_1085_, v_opts_1086_, v_clsEnabled_boxed_1097_, v_oldTraces_1088_, v_msg_1089_, v_resStartStop_1090_, v___y_1091_, v___y_1092_, v___y_1093_, v___y_1094_);
lean_dec(v___y_1094_);
lean_dec_ref(v___y_1093_);
lean_dec(v___y_1092_);
lean_dec_ref(v___y_1091_);
lean_dec_ref(v_opts_1086_);
return v_res_1098_;
}
}
static double _init_lp_aesop___private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder___closed__3(void){
_start:
{
lean_object* v___x_1103_; double v___x_1104_; 
v___x_1103_ = lean_unsigned_to_nat(1000000000u);
v___x_1104_ = lean_float_of_nat(v___x_1103_);
return v___x_1104_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder(lean_object* v_s_1105_, lean_object* v_b_1106_, lean_object* v_a_1107_, lean_object* v_a_1108_, lean_object* v_a_1109_, lean_object* v_a_1110_){
_start:
{
lean_object* v___x_1112_; 
lean_inc(v_a_1110_);
lean_inc_ref(v_a_1109_);
lean_inc(v_a_1108_);
lean_inc_ref(v_a_1107_);
v___x_1112_ = lean_apply_5(v_b_1106_, v_a_1107_, v_a_1108_, v_a_1109_, v_a_1110_, lean_box(0));
if (lean_obj_tag(v___x_1112_) == 0)
{
lean_object* v_a_1113_; lean_object* v_options_1114_; lean_object* v_uTactic_1115_; lean_object* v_preState_1116_; lean_object* v_preGoal_1117_; lean_object* v_postState_1118_; lean_object* v_postGoals_1119_; lean_object* v_inheritedTraceOptions_1120_; uint8_t v_hasTrace_1121_; lean_object* v___x_1122_; lean_object* v___x_1123_; lean_object* v___x_1124_; 
v_a_1113_ = lean_ctor_get(v___x_1112_, 0);
lean_inc(v_a_1113_);
lean_dec_ref_known(v___x_1112_, 1);
v_options_1114_ = lean_ctor_get(v_a_1109_, 2);
v_uTactic_1115_ = lean_ctor_get(v_a_1113_, 0);
v_preState_1116_ = lean_ctor_get(v_s_1105_, 0);
lean_inc_ref(v_preState_1116_);
v_preGoal_1117_ = lean_ctor_get(v_s_1105_, 1);
lean_inc(v_preGoal_1117_);
v_postState_1118_ = lean_ctor_get(v_s_1105_, 3);
lean_inc_ref(v_postState_1118_);
v_postGoals_1119_ = lean_ctor_get(v_s_1105_, 4);
lean_inc_ref(v_postGoals_1119_);
lean_dec_ref(v_s_1105_);
v_inheritedTraceOptions_1120_ = lean_ctor_get(v_a_1109_, 13);
v_hasTrace_1121_ = lean_ctor_get_uint8(v_options_1114_, sizeof(void*)*1);
v___x_1122_ = lean_box(0);
v___x_1123_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1123_, 0, v_preGoal_1117_);
lean_ctor_set(v___x_1123_, 1, v___x_1122_);
lean_inc(v_uTactic_1115_);
v___x_1124_ = lean_alloc_closure((void*)(lp_aesop_Aesop_runTacticCapturingPostState___boxed), 8, 3);
lean_closure_set(v___x_1124_, 0, v_uTactic_1115_);
lean_closure_set(v___x_1124_, 1, v_preState_1116_);
lean_closure_set(v___x_1124_, 2, v___x_1123_);
if (v_hasTrace_1121_ == 0)
{
lean_object* v___x_1125_; 
v___x_1125_ = lp_aesop_Lean_observing_x3f___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__0___redArg(v___x_1124_, v_a_1107_, v_a_1108_, v_a_1109_, v_a_1110_);
if (lean_obj_tag(v___x_1125_) == 0)
{
lean_object* v_a_1126_; lean_object* v___x_1128_; uint8_t v_isShared_1129_; uint8_t v_isSharedCheck_1167_; 
v_a_1126_ = lean_ctor_get(v___x_1125_, 0);
v_isSharedCheck_1167_ = !lean_is_exclusive(v___x_1125_);
if (v_isSharedCheck_1167_ == 0)
{
v___x_1128_ = v___x_1125_;
v_isShared_1129_ = v_isSharedCheck_1167_;
goto v_resetjp_1127_;
}
else
{
lean_inc(v_a_1126_);
lean_dec(v___x_1125_);
v___x_1128_ = lean_box(0);
v_isShared_1129_ = v_isSharedCheck_1167_;
goto v_resetjp_1127_;
}
v_resetjp_1127_:
{
if (lean_obj_tag(v_a_1126_) == 1)
{
lean_object* v_val_1130_; lean_object* v_fst_1131_; lean_object* v_snd_1132_; lean_object* v___x_1133_; lean_object* v___x_1134_; 
lean_del_object(v___x_1128_);
v_val_1130_ = lean_ctor_get(v_a_1126_, 0);
lean_inc(v_val_1130_);
lean_dec_ref_known(v_a_1126_, 1);
v_fst_1131_ = lean_ctor_get(v_val_1130_, 0);
lean_inc(v_fst_1131_);
v_snd_1132_ = lean_ctor_get(v_val_1130_, 1);
lean_inc(v_snd_1132_);
lean_dec(v_val_1130_);
v___x_1133_ = lean_array_mk(v_snd_1132_);
v___x_1134_ = lp_aesop_Aesop_Script_matchGoals(v_postState_1118_, v_fst_1131_, v_postGoals_1119_, v___x_1133_, v_a_1107_, v_a_1108_, v_a_1109_, v_a_1110_);
if (lean_obj_tag(v___x_1134_) == 0)
{
lean_object* v_a_1135_; lean_object* v___x_1137_; uint8_t v_isShared_1138_; uint8_t v_isSharedCheck_1154_; 
v_a_1135_ = lean_ctor_get(v___x_1134_, 0);
v_isSharedCheck_1154_ = !lean_is_exclusive(v___x_1134_);
if (v_isSharedCheck_1154_ == 0)
{
v___x_1137_ = v___x_1134_;
v_isShared_1138_ = v_isSharedCheck_1154_;
goto v_resetjp_1136_;
}
else
{
lean_inc(v_a_1135_);
lean_dec(v___x_1134_);
v___x_1137_ = lean_box(0);
v_isShared_1138_ = v_isSharedCheck_1154_;
goto v_resetjp_1136_;
}
v_resetjp_1136_:
{
if (lean_obj_tag(v_a_1135_) == 1)
{
lean_object* v___x_1140_; uint8_t v_isShared_1141_; uint8_t v_isSharedCheck_1148_; 
v_isSharedCheck_1148_ = !lean_is_exclusive(v_a_1135_);
if (v_isSharedCheck_1148_ == 0)
{
lean_object* v_unused_1149_; 
v_unused_1149_ = lean_ctor_get(v_a_1135_, 0);
lean_dec(v_unused_1149_);
v___x_1140_ = v_a_1135_;
v_isShared_1141_ = v_isSharedCheck_1148_;
goto v_resetjp_1139_;
}
else
{
lean_dec(v_a_1135_);
v___x_1140_ = lean_box(0);
v_isShared_1141_ = v_isSharedCheck_1148_;
goto v_resetjp_1139_;
}
v_resetjp_1139_:
{
lean_object* v___x_1143_; 
if (v_isShared_1141_ == 0)
{
lean_ctor_set(v___x_1140_, 0, v_a_1113_);
v___x_1143_ = v___x_1140_;
goto v_reusejp_1142_;
}
else
{
lean_object* v_reuseFailAlloc_1147_; 
v_reuseFailAlloc_1147_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1147_, 0, v_a_1113_);
v___x_1143_ = v_reuseFailAlloc_1147_;
goto v_reusejp_1142_;
}
v_reusejp_1142_:
{
lean_object* v___x_1145_; 
if (v_isShared_1138_ == 0)
{
lean_ctor_set(v___x_1137_, 0, v___x_1143_);
v___x_1145_ = v___x_1137_;
goto v_reusejp_1144_;
}
else
{
lean_object* v_reuseFailAlloc_1146_; 
v_reuseFailAlloc_1146_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1146_, 0, v___x_1143_);
v___x_1145_ = v_reuseFailAlloc_1146_;
goto v_reusejp_1144_;
}
v_reusejp_1144_:
{
return v___x_1145_;
}
}
}
}
else
{
lean_object* v___x_1150_; lean_object* v___x_1152_; 
lean_dec(v_a_1135_);
lean_dec(v_a_1113_);
v___x_1150_ = lean_box(0);
if (v_isShared_1138_ == 0)
{
lean_ctor_set(v___x_1137_, 0, v___x_1150_);
v___x_1152_ = v___x_1137_;
goto v_reusejp_1151_;
}
else
{
lean_object* v_reuseFailAlloc_1153_; 
v_reuseFailAlloc_1153_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1153_, 0, v___x_1150_);
v___x_1152_ = v_reuseFailAlloc_1153_;
goto v_reusejp_1151_;
}
v_reusejp_1151_:
{
return v___x_1152_;
}
}
}
}
else
{
lean_object* v_a_1155_; lean_object* v___x_1157_; uint8_t v_isShared_1158_; uint8_t v_isSharedCheck_1162_; 
lean_dec(v_a_1113_);
v_a_1155_ = lean_ctor_get(v___x_1134_, 0);
v_isSharedCheck_1162_ = !lean_is_exclusive(v___x_1134_);
if (v_isSharedCheck_1162_ == 0)
{
v___x_1157_ = v___x_1134_;
v_isShared_1158_ = v_isSharedCheck_1162_;
goto v_resetjp_1156_;
}
else
{
lean_inc(v_a_1155_);
lean_dec(v___x_1134_);
v___x_1157_ = lean_box(0);
v_isShared_1158_ = v_isSharedCheck_1162_;
goto v_resetjp_1156_;
}
v_resetjp_1156_:
{
lean_object* v___x_1160_; 
if (v_isShared_1158_ == 0)
{
v___x_1160_ = v___x_1157_;
goto v_reusejp_1159_;
}
else
{
lean_object* v_reuseFailAlloc_1161_; 
v_reuseFailAlloc_1161_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1161_, 0, v_a_1155_);
v___x_1160_ = v_reuseFailAlloc_1161_;
goto v_reusejp_1159_;
}
v_reusejp_1159_:
{
return v___x_1160_;
}
}
}
}
else
{
lean_object* v___x_1163_; lean_object* v___x_1165_; 
lean_dec(v_a_1126_);
lean_dec_ref(v_postGoals_1119_);
lean_dec_ref(v_postState_1118_);
lean_dec(v_a_1113_);
v___x_1163_ = lean_box(0);
if (v_isShared_1129_ == 0)
{
lean_ctor_set(v___x_1128_, 0, v___x_1163_);
v___x_1165_ = v___x_1128_;
goto v_reusejp_1164_;
}
else
{
lean_object* v_reuseFailAlloc_1166_; 
v_reuseFailAlloc_1166_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1166_, 0, v___x_1163_);
v___x_1165_ = v_reuseFailAlloc_1166_;
goto v_reusejp_1164_;
}
v_reusejp_1164_:
{
return v___x_1165_;
}
}
}
}
else
{
lean_object* v_a_1168_; lean_object* v___x_1170_; uint8_t v_isShared_1171_; uint8_t v_isSharedCheck_1175_; 
lean_dec_ref(v_postGoals_1119_);
lean_dec_ref(v_postState_1118_);
lean_dec(v_a_1113_);
v_a_1168_ = lean_ctor_get(v___x_1125_, 0);
v_isSharedCheck_1175_ = !lean_is_exclusive(v___x_1125_);
if (v_isSharedCheck_1175_ == 0)
{
v___x_1170_ = v___x_1125_;
v_isShared_1171_ = v_isSharedCheck_1175_;
goto v_resetjp_1169_;
}
else
{
lean_inc(v_a_1168_);
lean_dec(v___x_1125_);
v___x_1170_ = lean_box(0);
v_isShared_1171_ = v_isSharedCheck_1175_;
goto v_resetjp_1169_;
}
v_resetjp_1169_:
{
lean_object* v___x_1173_; 
if (v_isShared_1171_ == 0)
{
v___x_1173_ = v___x_1170_;
goto v_reusejp_1172_;
}
else
{
lean_object* v_reuseFailAlloc_1174_; 
v_reuseFailAlloc_1174_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1174_, 0, v_a_1168_);
v___x_1173_ = v_reuseFailAlloc_1174_;
goto v_reusejp_1172_;
}
v_reusejp_1172_:
{
return v___x_1173_;
}
}
}
}
else
{
lean_object* v___x_1176_; lean_object* v_traceClass_1177_; lean_object* v___f_1178_; lean_object* v___x_1179_; lean_object* v___x_1180_; lean_object* v___x_1181_; uint8_t v___x_1182_; lean_object* v___y_1184_; lean_object* v___y_1185_; lean_object* v_a_1186_; lean_object* v___y_1199_; lean_object* v___y_1200_; lean_object* v_a_1201_; lean_object* v___y_1204_; lean_object* v___y_1205_; lean_object* v_a_1206_; lean_object* v___y_1209_; lean_object* v___y_1210_; lean_object* v_a_1211_; lean_object* v___y_1221_; lean_object* v___y_1222_; lean_object* v_a_1223_; lean_object* v___y_1226_; lean_object* v___y_1227_; lean_object* v_a_1228_; 
v___x_1176_ = lp_aesop_Aesop_TraceOption_script;
v_traceClass_1177_ = lean_ctor_get(v___x_1176_, 0);
lean_inc(v_uTactic_1115_);
v___f_1178_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder___lam__0___boxed), 7, 1);
lean_closure_set(v___f_1178_, 0, v_uTactic_1115_);
v___x_1179_ = ((lean_object*)(lp_aesop___private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder___closed__0));
v___x_1180_ = ((lean_object*)(lp_aesop___private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder___closed__2));
lean_inc(v_traceClass_1177_);
v___x_1181_ = l_Lean_Name_append(v___x_1180_, v_traceClass_1177_);
v___x_1182_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_1120_, v_options_1114_, v___x_1181_);
lean_dec(v___x_1181_);
if (v___x_1182_ == 0)
{
lean_object* v___x_1277_; uint8_t v___x_1278_; 
v___x_1277_ = l_Lean_trace_profiler;
v___x_1278_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__2(v_options_1114_, v___x_1277_);
if (v___x_1278_ == 0)
{
lean_object* v___x_1279_; 
lean_dec_ref(v___f_1178_);
v___x_1279_ = lp_aesop_Lean_observing_x3f___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__0___redArg(v___x_1124_, v_a_1107_, v_a_1108_, v_a_1109_, v_a_1110_);
if (lean_obj_tag(v___x_1279_) == 0)
{
lean_object* v_a_1280_; lean_object* v___x_1282_; uint8_t v_isShared_1283_; uint8_t v_isSharedCheck_1321_; 
v_a_1280_ = lean_ctor_get(v___x_1279_, 0);
v_isSharedCheck_1321_ = !lean_is_exclusive(v___x_1279_);
if (v_isSharedCheck_1321_ == 0)
{
v___x_1282_ = v___x_1279_;
v_isShared_1283_ = v_isSharedCheck_1321_;
goto v_resetjp_1281_;
}
else
{
lean_inc(v_a_1280_);
lean_dec(v___x_1279_);
v___x_1282_ = lean_box(0);
v_isShared_1283_ = v_isSharedCheck_1321_;
goto v_resetjp_1281_;
}
v_resetjp_1281_:
{
if (lean_obj_tag(v_a_1280_) == 1)
{
lean_object* v_val_1284_; lean_object* v_fst_1285_; lean_object* v_snd_1286_; lean_object* v___x_1287_; lean_object* v___x_1288_; 
lean_del_object(v___x_1282_);
v_val_1284_ = lean_ctor_get(v_a_1280_, 0);
lean_inc(v_val_1284_);
lean_dec_ref_known(v_a_1280_, 1);
v_fst_1285_ = lean_ctor_get(v_val_1284_, 0);
lean_inc(v_fst_1285_);
v_snd_1286_ = lean_ctor_get(v_val_1284_, 1);
lean_inc(v_snd_1286_);
lean_dec(v_val_1284_);
v___x_1287_ = lean_array_mk(v_snd_1286_);
v___x_1288_ = lp_aesop_Aesop_Script_matchGoals(v_postState_1118_, v_fst_1285_, v_postGoals_1119_, v___x_1287_, v_a_1107_, v_a_1108_, v_a_1109_, v_a_1110_);
if (lean_obj_tag(v___x_1288_) == 0)
{
lean_object* v_a_1289_; lean_object* v___x_1291_; uint8_t v_isShared_1292_; uint8_t v_isSharedCheck_1308_; 
v_a_1289_ = lean_ctor_get(v___x_1288_, 0);
v_isSharedCheck_1308_ = !lean_is_exclusive(v___x_1288_);
if (v_isSharedCheck_1308_ == 0)
{
v___x_1291_ = v___x_1288_;
v_isShared_1292_ = v_isSharedCheck_1308_;
goto v_resetjp_1290_;
}
else
{
lean_inc(v_a_1289_);
lean_dec(v___x_1288_);
v___x_1291_ = lean_box(0);
v_isShared_1292_ = v_isSharedCheck_1308_;
goto v_resetjp_1290_;
}
v_resetjp_1290_:
{
if (lean_obj_tag(v_a_1289_) == 1)
{
lean_object* v___x_1294_; uint8_t v_isShared_1295_; uint8_t v_isSharedCheck_1302_; 
v_isSharedCheck_1302_ = !lean_is_exclusive(v_a_1289_);
if (v_isSharedCheck_1302_ == 0)
{
lean_object* v_unused_1303_; 
v_unused_1303_ = lean_ctor_get(v_a_1289_, 0);
lean_dec(v_unused_1303_);
v___x_1294_ = v_a_1289_;
v_isShared_1295_ = v_isSharedCheck_1302_;
goto v_resetjp_1293_;
}
else
{
lean_dec(v_a_1289_);
v___x_1294_ = lean_box(0);
v_isShared_1295_ = v_isSharedCheck_1302_;
goto v_resetjp_1293_;
}
v_resetjp_1293_:
{
lean_object* v___x_1297_; 
if (v_isShared_1295_ == 0)
{
lean_ctor_set(v___x_1294_, 0, v_a_1113_);
v___x_1297_ = v___x_1294_;
goto v_reusejp_1296_;
}
else
{
lean_object* v_reuseFailAlloc_1301_; 
v_reuseFailAlloc_1301_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1301_, 0, v_a_1113_);
v___x_1297_ = v_reuseFailAlloc_1301_;
goto v_reusejp_1296_;
}
v_reusejp_1296_:
{
lean_object* v___x_1299_; 
if (v_isShared_1292_ == 0)
{
lean_ctor_set(v___x_1291_, 0, v___x_1297_);
v___x_1299_ = v___x_1291_;
goto v_reusejp_1298_;
}
else
{
lean_object* v_reuseFailAlloc_1300_; 
v_reuseFailAlloc_1300_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1300_, 0, v___x_1297_);
v___x_1299_ = v_reuseFailAlloc_1300_;
goto v_reusejp_1298_;
}
v_reusejp_1298_:
{
return v___x_1299_;
}
}
}
}
else
{
lean_object* v___x_1304_; lean_object* v___x_1306_; 
lean_dec(v_a_1289_);
lean_dec(v_a_1113_);
v___x_1304_ = lean_box(0);
if (v_isShared_1292_ == 0)
{
lean_ctor_set(v___x_1291_, 0, v___x_1304_);
v___x_1306_ = v___x_1291_;
goto v_reusejp_1305_;
}
else
{
lean_object* v_reuseFailAlloc_1307_; 
v_reuseFailAlloc_1307_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1307_, 0, v___x_1304_);
v___x_1306_ = v_reuseFailAlloc_1307_;
goto v_reusejp_1305_;
}
v_reusejp_1305_:
{
return v___x_1306_;
}
}
}
}
else
{
lean_object* v_a_1309_; lean_object* v___x_1311_; uint8_t v_isShared_1312_; uint8_t v_isSharedCheck_1316_; 
lean_dec(v_a_1113_);
v_a_1309_ = lean_ctor_get(v___x_1288_, 0);
v_isSharedCheck_1316_ = !lean_is_exclusive(v___x_1288_);
if (v_isSharedCheck_1316_ == 0)
{
v___x_1311_ = v___x_1288_;
v_isShared_1312_ = v_isSharedCheck_1316_;
goto v_resetjp_1310_;
}
else
{
lean_inc(v_a_1309_);
lean_dec(v___x_1288_);
v___x_1311_ = lean_box(0);
v_isShared_1312_ = v_isSharedCheck_1316_;
goto v_resetjp_1310_;
}
v_resetjp_1310_:
{
lean_object* v___x_1314_; 
if (v_isShared_1312_ == 0)
{
v___x_1314_ = v___x_1311_;
goto v_reusejp_1313_;
}
else
{
lean_object* v_reuseFailAlloc_1315_; 
v_reuseFailAlloc_1315_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1315_, 0, v_a_1309_);
v___x_1314_ = v_reuseFailAlloc_1315_;
goto v_reusejp_1313_;
}
v_reusejp_1313_:
{
return v___x_1314_;
}
}
}
}
else
{
lean_object* v___x_1317_; lean_object* v___x_1319_; 
lean_dec(v_a_1280_);
lean_dec_ref(v_postGoals_1119_);
lean_dec_ref(v_postState_1118_);
lean_dec(v_a_1113_);
v___x_1317_ = lean_box(0);
if (v_isShared_1283_ == 0)
{
lean_ctor_set(v___x_1282_, 0, v___x_1317_);
v___x_1319_ = v___x_1282_;
goto v_reusejp_1318_;
}
else
{
lean_object* v_reuseFailAlloc_1320_; 
v_reuseFailAlloc_1320_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1320_, 0, v___x_1317_);
v___x_1319_ = v_reuseFailAlloc_1320_;
goto v_reusejp_1318_;
}
v_reusejp_1318_:
{
return v___x_1319_;
}
}
}
}
else
{
lean_object* v_a_1322_; lean_object* v___x_1324_; uint8_t v_isShared_1325_; uint8_t v_isSharedCheck_1329_; 
lean_dec_ref(v_postGoals_1119_);
lean_dec_ref(v_postState_1118_);
lean_dec(v_a_1113_);
v_a_1322_ = lean_ctor_get(v___x_1279_, 0);
v_isSharedCheck_1329_ = !lean_is_exclusive(v___x_1279_);
if (v_isSharedCheck_1329_ == 0)
{
v___x_1324_ = v___x_1279_;
v_isShared_1325_ = v_isSharedCheck_1329_;
goto v_resetjp_1323_;
}
else
{
lean_inc(v_a_1322_);
lean_dec(v___x_1279_);
v___x_1324_ = lean_box(0);
v_isShared_1325_ = v_isSharedCheck_1329_;
goto v_resetjp_1323_;
}
v_resetjp_1323_:
{
lean_object* v___x_1327_; 
if (v_isShared_1325_ == 0)
{
v___x_1327_ = v___x_1324_;
goto v_reusejp_1326_;
}
else
{
lean_object* v_reuseFailAlloc_1328_; 
v_reuseFailAlloc_1328_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1328_, 0, v_a_1322_);
v___x_1327_ = v_reuseFailAlloc_1328_;
goto v_reusejp_1326_;
}
v_reusejp_1326_:
{
return v___x_1327_;
}
}
}
}
else
{
goto v___jp_1230_;
}
}
else
{
goto v___jp_1230_;
}
v___jp_1183_:
{
lean_object* v___x_1187_; double v___x_1188_; double v___x_1189_; double v___x_1190_; double v___x_1191_; double v___x_1192_; lean_object* v___x_1193_; lean_object* v___x_1194_; lean_object* v___x_1195_; lean_object* v___x_1196_; lean_object* v___x_1197_; 
v___x_1187_ = lean_io_mono_nanos_now();
v___x_1188_ = lean_float_of_nat(v___y_1184_);
v___x_1189_ = lean_float_once(&lp_aesop___private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder___closed__3, &lp_aesop___private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder___closed__3_once, _init_lp_aesop___private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder___closed__3);
v___x_1190_ = lean_float_div(v___x_1188_, v___x_1189_);
v___x_1191_ = lean_float_of_nat(v___x_1187_);
v___x_1192_ = lean_float_div(v___x_1191_, v___x_1189_);
v___x_1193_ = lean_box_float(v___x_1190_);
v___x_1194_ = lean_box_float(v___x_1192_);
v___x_1195_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1195_, 0, v___x_1193_);
lean_ctor_set(v___x_1195_, 1, v___x_1194_);
v___x_1196_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1196_, 0, v_a_1186_);
lean_ctor_set(v___x_1196_, 1, v___x_1195_);
lean_inc(v_traceClass_1177_);
v___x_1197_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__3(v_traceClass_1177_, v_hasTrace_1121_, v___x_1179_, v_options_1114_, v___x_1182_, v___y_1185_, v___f_1178_, v___x_1196_, v_a_1107_, v_a_1108_, v_a_1109_, v_a_1110_);
return v___x_1197_;
}
v___jp_1198_:
{
lean_object* v___x_1202_; 
v___x_1202_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1202_, 0, v_a_1201_);
v___y_1184_ = v___y_1199_;
v___y_1185_ = v___y_1200_;
v_a_1186_ = v___x_1202_;
goto v___jp_1183_;
}
v___jp_1203_:
{
lean_object* v___x_1207_; 
v___x_1207_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1207_, 0, v_a_1206_);
v___y_1184_ = v___y_1204_;
v___y_1185_ = v___y_1205_;
v_a_1186_ = v___x_1207_;
goto v___jp_1183_;
}
v___jp_1208_:
{
lean_object* v___x_1212_; double v___x_1213_; double v___x_1214_; lean_object* v___x_1215_; lean_object* v___x_1216_; lean_object* v___x_1217_; lean_object* v___x_1218_; lean_object* v___x_1219_; 
v___x_1212_ = lean_io_get_num_heartbeats();
v___x_1213_ = lean_float_of_nat(v___y_1209_);
v___x_1214_ = lean_float_of_nat(v___x_1212_);
v___x_1215_ = lean_box_float(v___x_1213_);
v___x_1216_ = lean_box_float(v___x_1214_);
v___x_1217_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1217_, 0, v___x_1215_);
lean_ctor_set(v___x_1217_, 1, v___x_1216_);
v___x_1218_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1218_, 0, v_a_1211_);
lean_ctor_set(v___x_1218_, 1, v___x_1217_);
lean_inc(v_traceClass_1177_);
v___x_1219_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__3(v_traceClass_1177_, v_hasTrace_1121_, v___x_1179_, v_options_1114_, v___x_1182_, v___y_1210_, v___f_1178_, v___x_1218_, v_a_1107_, v_a_1108_, v_a_1109_, v_a_1110_);
return v___x_1219_;
}
v___jp_1220_:
{
lean_object* v___x_1224_; 
v___x_1224_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1224_, 0, v_a_1223_);
v___y_1209_ = v___y_1221_;
v___y_1210_ = v___y_1222_;
v_a_1211_ = v___x_1224_;
goto v___jp_1208_;
}
v___jp_1225_:
{
lean_object* v___x_1229_; 
v___x_1229_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1229_, 0, v_a_1228_);
v___y_1209_ = v___y_1226_;
v___y_1210_ = v___y_1227_;
v_a_1211_ = v___x_1229_;
goto v___jp_1208_;
}
v___jp_1230_:
{
lean_object* v___x_1231_; lean_object* v_a_1232_; lean_object* v___x_1233_; uint8_t v___x_1234_; 
v___x_1231_ = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__1___redArg(v_a_1110_);
v_a_1232_ = lean_ctor_get(v___x_1231_, 0);
lean_inc(v_a_1232_);
lean_dec_ref(v___x_1231_);
v___x_1233_ = l_Lean_trace_profiler_useHeartbeats;
v___x_1234_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__2(v_options_1114_, v___x_1233_);
if (v___x_1234_ == 0)
{
lean_object* v___x_1235_; lean_object* v___x_1236_; 
v___x_1235_ = lean_io_mono_nanos_now();
v___x_1236_ = lp_aesop_Lean_observing_x3f___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__0___redArg(v___x_1124_, v_a_1107_, v_a_1108_, v_a_1109_, v_a_1110_);
if (lean_obj_tag(v___x_1236_) == 0)
{
lean_object* v_a_1237_; 
v_a_1237_ = lean_ctor_get(v___x_1236_, 0);
lean_inc(v_a_1237_);
lean_dec_ref_known(v___x_1236_, 1);
if (lean_obj_tag(v_a_1237_) == 1)
{
lean_object* v_val_1238_; lean_object* v_fst_1239_; lean_object* v_snd_1240_; lean_object* v___x_1241_; lean_object* v___x_1242_; 
v_val_1238_ = lean_ctor_get(v_a_1237_, 0);
lean_inc(v_val_1238_);
lean_dec_ref_known(v_a_1237_, 1);
v_fst_1239_ = lean_ctor_get(v_val_1238_, 0);
lean_inc(v_fst_1239_);
v_snd_1240_ = lean_ctor_get(v_val_1238_, 1);
lean_inc(v_snd_1240_);
lean_dec(v_val_1238_);
v___x_1241_ = lean_array_mk(v_snd_1240_);
v___x_1242_ = lp_aesop_Aesop_Script_matchGoals(v_postState_1118_, v_fst_1239_, v_postGoals_1119_, v___x_1241_, v_a_1107_, v_a_1108_, v_a_1109_, v_a_1110_);
if (lean_obj_tag(v___x_1242_) == 0)
{
lean_object* v_a_1243_; 
v_a_1243_ = lean_ctor_get(v___x_1242_, 0);
lean_inc(v_a_1243_);
lean_dec_ref_known(v___x_1242_, 1);
if (lean_obj_tag(v_a_1243_) == 1)
{
lean_object* v___x_1245_; uint8_t v_isShared_1246_; uint8_t v_isSharedCheck_1250_; 
v_isSharedCheck_1250_ = !lean_is_exclusive(v_a_1243_);
if (v_isSharedCheck_1250_ == 0)
{
lean_object* v_unused_1251_; 
v_unused_1251_ = lean_ctor_get(v_a_1243_, 0);
lean_dec(v_unused_1251_);
v___x_1245_ = v_a_1243_;
v_isShared_1246_ = v_isSharedCheck_1250_;
goto v_resetjp_1244_;
}
else
{
lean_dec(v_a_1243_);
v___x_1245_ = lean_box(0);
v_isShared_1246_ = v_isSharedCheck_1250_;
goto v_resetjp_1244_;
}
v_resetjp_1244_:
{
lean_object* v___x_1248_; 
if (v_isShared_1246_ == 0)
{
lean_ctor_set(v___x_1245_, 0, v_a_1113_);
v___x_1248_ = v___x_1245_;
goto v_reusejp_1247_;
}
else
{
lean_object* v_reuseFailAlloc_1249_; 
v_reuseFailAlloc_1249_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1249_, 0, v_a_1113_);
v___x_1248_ = v_reuseFailAlloc_1249_;
goto v_reusejp_1247_;
}
v_reusejp_1247_:
{
v___y_1204_ = v___x_1235_;
v___y_1205_ = v_a_1232_;
v_a_1206_ = v___x_1248_;
goto v___jp_1203_;
}
}
}
else
{
lean_object* v___x_1252_; 
lean_dec(v_a_1243_);
lean_dec(v_a_1113_);
v___x_1252_ = lean_box(0);
v___y_1204_ = v___x_1235_;
v___y_1205_ = v_a_1232_;
v_a_1206_ = v___x_1252_;
goto v___jp_1203_;
}
}
else
{
lean_object* v_a_1253_; 
lean_dec(v_a_1113_);
v_a_1253_ = lean_ctor_get(v___x_1242_, 0);
lean_inc(v_a_1253_);
lean_dec_ref_known(v___x_1242_, 1);
v___y_1199_ = v___x_1235_;
v___y_1200_ = v_a_1232_;
v_a_1201_ = v_a_1253_;
goto v___jp_1198_;
}
}
else
{
lean_object* v___x_1254_; 
lean_dec(v_a_1237_);
lean_dec_ref(v_postGoals_1119_);
lean_dec_ref(v_postState_1118_);
lean_dec(v_a_1113_);
v___x_1254_ = lean_box(0);
v___y_1204_ = v___x_1235_;
v___y_1205_ = v_a_1232_;
v_a_1206_ = v___x_1254_;
goto v___jp_1203_;
}
}
else
{
lean_object* v_a_1255_; 
lean_dec_ref(v_postGoals_1119_);
lean_dec_ref(v_postState_1118_);
lean_dec(v_a_1113_);
v_a_1255_ = lean_ctor_get(v___x_1236_, 0);
lean_inc(v_a_1255_);
lean_dec_ref_known(v___x_1236_, 1);
v___y_1199_ = v___x_1235_;
v___y_1200_ = v_a_1232_;
v_a_1201_ = v_a_1255_;
goto v___jp_1198_;
}
}
else
{
lean_object* v___x_1256_; lean_object* v___x_1257_; 
v___x_1256_ = lean_io_get_num_heartbeats();
v___x_1257_ = lp_aesop_Lean_observing_x3f___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__0___redArg(v___x_1124_, v_a_1107_, v_a_1108_, v_a_1109_, v_a_1110_);
if (lean_obj_tag(v___x_1257_) == 0)
{
lean_object* v_a_1258_; 
v_a_1258_ = lean_ctor_get(v___x_1257_, 0);
lean_inc(v_a_1258_);
lean_dec_ref_known(v___x_1257_, 1);
if (lean_obj_tag(v_a_1258_) == 1)
{
lean_object* v_val_1259_; lean_object* v_fst_1260_; lean_object* v_snd_1261_; lean_object* v___x_1262_; lean_object* v___x_1263_; 
v_val_1259_ = lean_ctor_get(v_a_1258_, 0);
lean_inc(v_val_1259_);
lean_dec_ref_known(v_a_1258_, 1);
v_fst_1260_ = lean_ctor_get(v_val_1259_, 0);
lean_inc(v_fst_1260_);
v_snd_1261_ = lean_ctor_get(v_val_1259_, 1);
lean_inc(v_snd_1261_);
lean_dec(v_val_1259_);
v___x_1262_ = lean_array_mk(v_snd_1261_);
v___x_1263_ = lp_aesop_Aesop_Script_matchGoals(v_postState_1118_, v_fst_1260_, v_postGoals_1119_, v___x_1262_, v_a_1107_, v_a_1108_, v_a_1109_, v_a_1110_);
if (lean_obj_tag(v___x_1263_) == 0)
{
lean_object* v_a_1264_; 
v_a_1264_ = lean_ctor_get(v___x_1263_, 0);
lean_inc(v_a_1264_);
lean_dec_ref_known(v___x_1263_, 1);
if (lean_obj_tag(v_a_1264_) == 1)
{
lean_object* v___x_1266_; uint8_t v_isShared_1267_; uint8_t v_isSharedCheck_1271_; 
v_isSharedCheck_1271_ = !lean_is_exclusive(v_a_1264_);
if (v_isSharedCheck_1271_ == 0)
{
lean_object* v_unused_1272_; 
v_unused_1272_ = lean_ctor_get(v_a_1264_, 0);
lean_dec(v_unused_1272_);
v___x_1266_ = v_a_1264_;
v_isShared_1267_ = v_isSharedCheck_1271_;
goto v_resetjp_1265_;
}
else
{
lean_dec(v_a_1264_);
v___x_1266_ = lean_box(0);
v_isShared_1267_ = v_isSharedCheck_1271_;
goto v_resetjp_1265_;
}
v_resetjp_1265_:
{
lean_object* v___x_1269_; 
if (v_isShared_1267_ == 0)
{
lean_ctor_set(v___x_1266_, 0, v_a_1113_);
v___x_1269_ = v___x_1266_;
goto v_reusejp_1268_;
}
else
{
lean_object* v_reuseFailAlloc_1270_; 
v_reuseFailAlloc_1270_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1270_, 0, v_a_1113_);
v___x_1269_ = v_reuseFailAlloc_1270_;
goto v_reusejp_1268_;
}
v_reusejp_1268_:
{
v___y_1226_ = v___x_1256_;
v___y_1227_ = v_a_1232_;
v_a_1228_ = v___x_1269_;
goto v___jp_1225_;
}
}
}
else
{
lean_object* v___x_1273_; 
lean_dec(v_a_1264_);
lean_dec(v_a_1113_);
v___x_1273_ = lean_box(0);
v___y_1226_ = v___x_1256_;
v___y_1227_ = v_a_1232_;
v_a_1228_ = v___x_1273_;
goto v___jp_1225_;
}
}
else
{
lean_object* v_a_1274_; 
lean_dec(v_a_1113_);
v_a_1274_ = lean_ctor_get(v___x_1263_, 0);
lean_inc(v_a_1274_);
lean_dec_ref_known(v___x_1263_, 1);
v___y_1221_ = v___x_1256_;
v___y_1222_ = v_a_1232_;
v_a_1223_ = v_a_1274_;
goto v___jp_1220_;
}
}
else
{
lean_object* v___x_1275_; 
lean_dec(v_a_1258_);
lean_dec_ref(v_postGoals_1119_);
lean_dec_ref(v_postState_1118_);
lean_dec(v_a_1113_);
v___x_1275_ = lean_box(0);
v___y_1226_ = v___x_1256_;
v___y_1227_ = v_a_1232_;
v_a_1228_ = v___x_1275_;
goto v___jp_1225_;
}
}
else
{
lean_object* v_a_1276_; 
lean_dec_ref(v_postGoals_1119_);
lean_dec_ref(v_postState_1118_);
lean_dec(v_a_1113_);
v_a_1276_ = lean_ctor_get(v___x_1257_, 0);
lean_inc(v_a_1276_);
lean_dec_ref_known(v___x_1257_, 1);
v___y_1221_ = v___x_1256_;
v___y_1222_ = v_a_1232_;
v_a_1223_ = v_a_1276_;
goto v___jp_1220_;
}
}
}
}
}
else
{
lean_object* v_a_1330_; lean_object* v___x_1332_; uint8_t v_isShared_1333_; uint8_t v_isSharedCheck_1337_; 
lean_dec_ref(v_s_1105_);
v_a_1330_ = lean_ctor_get(v___x_1112_, 0);
v_isSharedCheck_1337_ = !lean_is_exclusive(v___x_1112_);
if (v_isSharedCheck_1337_ == 0)
{
v___x_1332_ = v___x_1112_;
v_isShared_1333_ = v_isSharedCheck_1337_;
goto v_resetjp_1331_;
}
else
{
lean_inc(v_a_1330_);
lean_dec(v___x_1112_);
v___x_1332_ = lean_box(0);
v_isShared_1333_ = v_isSharedCheck_1337_;
goto v_resetjp_1331_;
}
v_resetjp_1331_:
{
lean_object* v___x_1335_; 
if (v_isShared_1333_ == 0)
{
v___x_1335_ = v___x_1332_;
goto v_reusejp_1334_;
}
else
{
lean_object* v_reuseFailAlloc_1336_; 
v_reuseFailAlloc_1336_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1336_, 0, v_a_1330_);
v___x_1335_ = v_reuseFailAlloc_1336_;
goto v_reusejp_1334_;
}
v_reusejp_1334_:
{
return v___x_1335_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder___boxed(lean_object* v_s_1338_, lean_object* v_b_1339_, lean_object* v_a_1340_, lean_object* v_a_1341_, lean_object* v_a_1342_, lean_object* v_a_1343_, lean_object* v_a_1344_){
_start:
{
lean_object* v_res_1345_; 
v_res_1345_ = lp_aesop___private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder(v_s_1338_, v_b_1339_, v_a_1340_, v_a_1341_, v_a_1342_, v_a_1343_);
lean_dec(v_a_1343_);
lean_dec_ref(v_a_1342_);
lean_dec(v_a_1341_);
lean_dec_ref(v_a_1340_);
return v_res_1345_;
}
}
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__3_spec__4(lean_object* v_00_u03b1_1346_, lean_object* v_x_1347_, lean_object* v___y_1348_, lean_object* v___y_1349_, lean_object* v___y_1350_, lean_object* v___y_1351_){
_start:
{
lean_object* v___x_1353_; 
v___x_1353_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__3_spec__4___redArg(v_x_1347_);
return v___x_1353_;
}
}
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__3_spec__4___boxed(lean_object* v_00_u03b1_1354_, lean_object* v_x_1355_, lean_object* v___y_1356_, lean_object* v___y_1357_, lean_object* v___y_1358_, lean_object* v___y_1359_, lean_object* v___y_1360_){
_start:
{
lean_object* v_res_1361_; 
v_res_1361_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__3_spec__4(v_00_u03b1_1354_, v_x_1355_, v___y_1356_, v___y_1357_, v___y_1358_, v___y_1359_);
lean_dec(v___y_1359_);
lean_dec_ref(v___y_1358_);
lean_dec(v___y_1357_);
lean_dec_ref(v___y_1356_);
return v_res_1361_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__1___redArg(lean_object* v_opt_1362_, lean_object* v___y_1363_){
_start:
{
lean_object* v_options_1365_; lean_object* v_option_1366_; uint8_t v___x_1367_; lean_object* v___x_1368_; lean_object* v___x_1369_; 
v_options_1365_ = lean_ctor_get(v___y_1363_, 2);
v_option_1366_ = lean_ctor_get(v_opt_1362_, 1);
v___x_1367_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__2(v_options_1365_, v_option_1366_);
v___x_1368_ = lean_box(v___x_1367_);
v___x_1369_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1369_, 0, v___x_1368_);
return v___x_1369_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__1___redArg___boxed(lean_object* v_opt_1370_, lean_object* v___y_1371_, lean_object* v___y_1372_){
_start:
{
lean_object* v_res_1373_; 
v_res_1373_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__1___redArg(v_opt_1370_, v___y_1371_);
lean_dec_ref(v___y_1371_);
lean_dec_ref(v_opt_1370_);
return v_res_1373_;
}
}
LEAN_EXPORT lean_object* lp_aesop_WellFounded_opaqueFix_u2083___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__0___redArg(lean_object* v_s_1377_, lean_object* v_a_1378_, lean_object* v_a_1379_, lean_object* v_b_1380_, lean_object* v___y_1381_, lean_object* v___y_1382_, lean_object* v___y_1383_, lean_object* v___y_1384_){
_start:
{
lean_object* v_array_1386_; lean_object* v_start_1387_; lean_object* v_stop_1388_; lean_object* v___x_1390_; uint8_t v_isShared_1391_; uint8_t v_isSharedCheck_1430_; 
v_array_1386_ = lean_ctor_get(v_a_1379_, 0);
v_start_1387_ = lean_ctor_get(v_a_1379_, 1);
v_stop_1388_ = lean_ctor_get(v_a_1379_, 2);
v_isSharedCheck_1430_ = !lean_is_exclusive(v_a_1379_);
if (v_isSharedCheck_1430_ == 0)
{
v___x_1390_ = v_a_1379_;
v_isShared_1391_ = v_isSharedCheck_1430_;
goto v_resetjp_1389_;
}
else
{
lean_inc(v_stop_1388_);
lean_inc(v_start_1387_);
lean_inc(v_array_1386_);
lean_dec(v_a_1379_);
v___x_1390_ = lean_box(0);
v_isShared_1391_ = v_isSharedCheck_1430_;
goto v_resetjp_1389_;
}
v_resetjp_1389_:
{
uint8_t v___x_1392_; 
v___x_1392_ = lean_nat_dec_lt(v_start_1387_, v_stop_1388_);
if (v___x_1392_ == 0)
{
lean_object* v___x_1393_; 
lean_del_object(v___x_1390_);
lean_dec(v_stop_1388_);
lean_dec(v_start_1387_);
lean_dec_ref(v_array_1386_);
lean_dec_ref(v_s_1377_);
v___x_1393_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1393_, 0, v_b_1380_);
return v___x_1393_;
}
else
{
lean_object* v___x_1394_; lean_object* v___x_1395_; 
lean_dec_ref(v_b_1380_);
v___x_1394_ = lean_array_fget_borrowed(v_array_1386_, v_start_1387_);
lean_inc(v___x_1394_);
lean_inc_ref(v_s_1377_);
v___x_1395_ = lp_aesop___private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder(v_s_1377_, v___x_1394_, v___y_1381_, v___y_1382_, v___y_1383_, v___y_1384_);
if (lean_obj_tag(v___x_1395_) == 0)
{
lean_object* v_a_1396_; lean_object* v___x_1398_; uint8_t v_isShared_1399_; uint8_t v_isSharedCheck_1421_; 
v_a_1396_ = lean_ctor_get(v___x_1395_, 0);
v_isSharedCheck_1421_ = !lean_is_exclusive(v___x_1395_);
if (v_isSharedCheck_1421_ == 0)
{
v___x_1398_ = v___x_1395_;
v_isShared_1399_ = v_isSharedCheck_1421_;
goto v_resetjp_1397_;
}
else
{
lean_inc(v_a_1396_);
lean_dec(v___x_1395_);
v___x_1398_ = lean_box(0);
v_isShared_1399_ = v_isSharedCheck_1421_;
goto v_resetjp_1397_;
}
v_resetjp_1397_:
{
lean_object* v___x_1400_; 
v___x_1400_ = lean_box(0);
if (lean_obj_tag(v_a_1396_) == 1)
{
lean_object* v___x_1401_; lean_object* v___x_1403_; 
lean_del_object(v___x_1390_);
lean_dec(v_stop_1388_);
lean_dec(v_start_1387_);
lean_dec_ref(v_array_1386_);
lean_dec_ref(v_s_1377_);
v___x_1401_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1401_, 0, v_a_1396_);
lean_ctor_set(v___x_1401_, 1, v___x_1400_);
if (v_isShared_1399_ == 0)
{
lean_ctor_set(v___x_1398_, 0, v___x_1401_);
v___x_1403_ = v___x_1398_;
goto v_reusejp_1402_;
}
else
{
lean_object* v_reuseFailAlloc_1404_; 
v_reuseFailAlloc_1404_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1404_, 0, v___x_1401_);
v___x_1403_ = v_reuseFailAlloc_1404_;
goto v_reusejp_1402_;
}
v_reusejp_1402_:
{
return v___x_1403_;
}
}
else
{
lean_object* v___x_1405_; 
lean_del_object(v___x_1398_);
lean_dec(v_a_1396_);
v___x_1405_ = l_Lean_Meta_SavedState_restore___redArg(v_a_1378_, v___y_1382_, v___y_1384_);
if (lean_obj_tag(v___x_1405_) == 0)
{
lean_object* v___x_1406_; lean_object* v___x_1407_; lean_object* v___x_1408_; lean_object* v___x_1410_; 
lean_dec_ref_known(v___x_1405_, 1);
v___x_1406_ = ((lean_object*)(lp_aesop_WellFounded_opaqueFix_u2083___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__0___redArg___closed__0));
v___x_1407_ = lean_unsigned_to_nat(1u);
v___x_1408_ = lean_nat_add(v_start_1387_, v___x_1407_);
lean_dec(v_start_1387_);
if (v_isShared_1391_ == 0)
{
lean_ctor_set(v___x_1390_, 1, v___x_1408_);
v___x_1410_ = v___x_1390_;
goto v_reusejp_1409_;
}
else
{
lean_object* v_reuseFailAlloc_1412_; 
v_reuseFailAlloc_1412_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1412_, 0, v_array_1386_);
lean_ctor_set(v_reuseFailAlloc_1412_, 1, v___x_1408_);
lean_ctor_set(v_reuseFailAlloc_1412_, 2, v_stop_1388_);
v___x_1410_ = v_reuseFailAlloc_1412_;
goto v_reusejp_1409_;
}
v_reusejp_1409_:
{
v_a_1379_ = v___x_1410_;
v_b_1380_ = v___x_1406_;
goto _start;
}
}
else
{
lean_object* v_a_1413_; lean_object* v___x_1415_; uint8_t v_isShared_1416_; uint8_t v_isSharedCheck_1420_; 
lean_del_object(v___x_1390_);
lean_dec(v_stop_1388_);
lean_dec(v_start_1387_);
lean_dec_ref(v_array_1386_);
lean_dec_ref(v_s_1377_);
v_a_1413_ = lean_ctor_get(v___x_1405_, 0);
v_isSharedCheck_1420_ = !lean_is_exclusive(v___x_1405_);
if (v_isSharedCheck_1420_ == 0)
{
v___x_1415_ = v___x_1405_;
v_isShared_1416_ = v_isSharedCheck_1420_;
goto v_resetjp_1414_;
}
else
{
lean_inc(v_a_1413_);
lean_dec(v___x_1405_);
v___x_1415_ = lean_box(0);
v_isShared_1416_ = v_isSharedCheck_1420_;
goto v_resetjp_1414_;
}
v_resetjp_1414_:
{
lean_object* v___x_1418_; 
if (v_isShared_1416_ == 0)
{
v___x_1418_ = v___x_1415_;
goto v_reusejp_1417_;
}
else
{
lean_object* v_reuseFailAlloc_1419_; 
v_reuseFailAlloc_1419_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1419_, 0, v_a_1413_);
v___x_1418_ = v_reuseFailAlloc_1419_;
goto v_reusejp_1417_;
}
v_reusejp_1417_:
{
return v___x_1418_;
}
}
}
}
}
}
else
{
lean_object* v_a_1422_; lean_object* v___x_1424_; uint8_t v_isShared_1425_; uint8_t v_isSharedCheck_1429_; 
lean_del_object(v___x_1390_);
lean_dec(v_stop_1388_);
lean_dec(v_start_1387_);
lean_dec_ref(v_array_1386_);
lean_dec_ref(v_s_1377_);
v_a_1422_ = lean_ctor_get(v___x_1395_, 0);
v_isSharedCheck_1429_ = !lean_is_exclusive(v___x_1395_);
if (v_isSharedCheck_1429_ == 0)
{
v___x_1424_ = v___x_1395_;
v_isShared_1425_ = v_isSharedCheck_1429_;
goto v_resetjp_1423_;
}
else
{
lean_inc(v_a_1422_);
lean_dec(v___x_1395_);
v___x_1424_ = lean_box(0);
v_isShared_1425_ = v_isSharedCheck_1429_;
goto v_resetjp_1423_;
}
v_resetjp_1423_:
{
lean_object* v___x_1427_; 
if (v_isShared_1425_ == 0)
{
v___x_1427_ = v___x_1424_;
goto v_reusejp_1426_;
}
else
{
lean_object* v_reuseFailAlloc_1428_; 
v_reuseFailAlloc_1428_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1428_, 0, v_a_1422_);
v___x_1427_ = v_reuseFailAlloc_1428_;
goto v_reusejp_1426_;
}
v_reusejp_1426_:
{
return v___x_1427_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_WellFounded_opaqueFix_u2083___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__0___redArg___boxed(lean_object* v_s_1431_, lean_object* v_a_1432_, lean_object* v_a_1433_, lean_object* v_b_1434_, lean_object* v___y_1435_, lean_object* v___y_1436_, lean_object* v___y_1437_, lean_object* v___y_1438_, lean_object* v___y_1439_){
_start:
{
lean_object* v_res_1440_; 
v_res_1440_ = lp_aesop_WellFounded_opaqueFix_u2083___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__0___redArg(v_s_1431_, v_a_1432_, v_a_1433_, v_b_1434_, v___y_1435_, v___y_1436_, v___y_1437_, v___y_1438_);
lean_dec(v___y_1438_);
lean_dec_ref(v___y_1437_);
lean_dec(v___y_1436_);
lean_dec_ref(v___y_1435_);
lean_dec_ref(v_a_1432_);
return v_res_1440_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addTrace___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__2(lean_object* v_cls_1443_, lean_object* v_msg_1444_, lean_object* v___y_1445_, lean_object* v___y_1446_, lean_object* v___y_1447_, lean_object* v___y_1448_){
_start:
{
lean_object* v_ref_1450_; lean_object* v___x_1451_; lean_object* v_a_1452_; lean_object* v___x_1454_; uint8_t v_isShared_1455_; uint8_t v_isSharedCheck_1496_; 
v_ref_1450_ = lean_ctor_get(v___y_1447_, 5);
v___x_1451_ = lp_aesop_Lean_addMessageContextFull___at___00__private_Aesop_Script_Step_0__Aesop_Script_Step_validate_fmtGoals_spec__1(v_msg_1444_, v___y_1445_, v___y_1446_, v___y_1447_, v___y_1448_);
v_a_1452_ = lean_ctor_get(v___x_1451_, 0);
v_isSharedCheck_1496_ = !lean_is_exclusive(v___x_1451_);
if (v_isSharedCheck_1496_ == 0)
{
v___x_1454_ = v___x_1451_;
v_isShared_1455_ = v_isSharedCheck_1496_;
goto v_resetjp_1453_;
}
else
{
lean_inc(v_a_1452_);
lean_dec(v___x_1451_);
v___x_1454_ = lean_box(0);
v_isShared_1455_ = v_isSharedCheck_1496_;
goto v_resetjp_1453_;
}
v_resetjp_1453_:
{
lean_object* v___x_1456_; lean_object* v_traceState_1457_; lean_object* v_env_1458_; lean_object* v_nextMacroScope_1459_; lean_object* v_ngen_1460_; lean_object* v_auxDeclNGen_1461_; lean_object* v_cache_1462_; lean_object* v_messages_1463_; lean_object* v_infoState_1464_; lean_object* v_snapshotTasks_1465_; lean_object* v___x_1467_; uint8_t v_isShared_1468_; uint8_t v_isSharedCheck_1495_; 
v___x_1456_ = lean_st_ref_take(v___y_1448_);
v_traceState_1457_ = lean_ctor_get(v___x_1456_, 4);
v_env_1458_ = lean_ctor_get(v___x_1456_, 0);
v_nextMacroScope_1459_ = lean_ctor_get(v___x_1456_, 1);
v_ngen_1460_ = lean_ctor_get(v___x_1456_, 2);
v_auxDeclNGen_1461_ = lean_ctor_get(v___x_1456_, 3);
v_cache_1462_ = lean_ctor_get(v___x_1456_, 5);
v_messages_1463_ = lean_ctor_get(v___x_1456_, 6);
v_infoState_1464_ = lean_ctor_get(v___x_1456_, 7);
v_snapshotTasks_1465_ = lean_ctor_get(v___x_1456_, 8);
v_isSharedCheck_1495_ = !lean_is_exclusive(v___x_1456_);
if (v_isSharedCheck_1495_ == 0)
{
v___x_1467_ = v___x_1456_;
v_isShared_1468_ = v_isSharedCheck_1495_;
goto v_resetjp_1466_;
}
else
{
lean_inc(v_snapshotTasks_1465_);
lean_inc(v_infoState_1464_);
lean_inc(v_messages_1463_);
lean_inc(v_cache_1462_);
lean_inc(v_traceState_1457_);
lean_inc(v_auxDeclNGen_1461_);
lean_inc(v_ngen_1460_);
lean_inc(v_nextMacroScope_1459_);
lean_inc(v_env_1458_);
lean_dec(v___x_1456_);
v___x_1467_ = lean_box(0);
v_isShared_1468_ = v_isSharedCheck_1495_;
goto v_resetjp_1466_;
}
v_resetjp_1466_:
{
uint64_t v_tid_1469_; lean_object* v_traces_1470_; lean_object* v___x_1472_; uint8_t v_isShared_1473_; uint8_t v_isSharedCheck_1494_; 
v_tid_1469_ = lean_ctor_get_uint64(v_traceState_1457_, sizeof(void*)*1);
v_traces_1470_ = lean_ctor_get(v_traceState_1457_, 0);
v_isSharedCheck_1494_ = !lean_is_exclusive(v_traceState_1457_);
if (v_isSharedCheck_1494_ == 0)
{
v___x_1472_ = v_traceState_1457_;
v_isShared_1473_ = v_isSharedCheck_1494_;
goto v_resetjp_1471_;
}
else
{
lean_inc(v_traces_1470_);
lean_dec(v_traceState_1457_);
v___x_1472_ = lean_box(0);
v_isShared_1473_ = v_isSharedCheck_1494_;
goto v_resetjp_1471_;
}
v_resetjp_1471_:
{
lean_object* v___x_1474_; double v___x_1475_; uint8_t v___x_1476_; lean_object* v___x_1477_; lean_object* v___x_1478_; lean_object* v___x_1479_; lean_object* v___x_1480_; lean_object* v___x_1481_; lean_object* v___x_1482_; lean_object* v___x_1484_; 
v___x_1474_ = lean_box(0);
v___x_1475_ = lean_float_once(&lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__3___closed__0, &lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__3___closed__0_once, _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__3___closed__0);
v___x_1476_ = 0;
v___x_1477_ = ((lean_object*)(lp_aesop___private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder___closed__0));
v___x_1478_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v___x_1478_, 0, v_cls_1443_);
lean_ctor_set(v___x_1478_, 1, v___x_1474_);
lean_ctor_set(v___x_1478_, 2, v___x_1477_);
lean_ctor_set_float(v___x_1478_, sizeof(void*)*3, v___x_1475_);
lean_ctor_set_float(v___x_1478_, sizeof(void*)*3 + 8, v___x_1475_);
lean_ctor_set_uint8(v___x_1478_, sizeof(void*)*3 + 16, v___x_1476_);
v___x_1479_ = ((lean_object*)(lp_aesop_Lean_addTrace___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__2___closed__0));
v___x_1480_ = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(v___x_1480_, 0, v___x_1478_);
lean_ctor_set(v___x_1480_, 1, v_a_1452_);
lean_ctor_set(v___x_1480_, 2, v___x_1479_);
lean_inc(v_ref_1450_);
v___x_1481_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1481_, 0, v_ref_1450_);
lean_ctor_set(v___x_1481_, 1, v___x_1480_);
v___x_1482_ = l_Lean_PersistentArray_push___redArg(v_traces_1470_, v___x_1481_);
if (v_isShared_1473_ == 0)
{
lean_ctor_set(v___x_1472_, 0, v___x_1482_);
v___x_1484_ = v___x_1472_;
goto v_reusejp_1483_;
}
else
{
lean_object* v_reuseFailAlloc_1493_; 
v_reuseFailAlloc_1493_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_1493_, 0, v___x_1482_);
lean_ctor_set_uint64(v_reuseFailAlloc_1493_, sizeof(void*)*1, v_tid_1469_);
v___x_1484_ = v_reuseFailAlloc_1493_;
goto v_reusejp_1483_;
}
v_reusejp_1483_:
{
lean_object* v___x_1486_; 
if (v_isShared_1468_ == 0)
{
lean_ctor_set(v___x_1467_, 4, v___x_1484_);
v___x_1486_ = v___x_1467_;
goto v_reusejp_1485_;
}
else
{
lean_object* v_reuseFailAlloc_1492_; 
v_reuseFailAlloc_1492_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_1492_, 0, v_env_1458_);
lean_ctor_set(v_reuseFailAlloc_1492_, 1, v_nextMacroScope_1459_);
lean_ctor_set(v_reuseFailAlloc_1492_, 2, v_ngen_1460_);
lean_ctor_set(v_reuseFailAlloc_1492_, 3, v_auxDeclNGen_1461_);
lean_ctor_set(v_reuseFailAlloc_1492_, 4, v___x_1484_);
lean_ctor_set(v_reuseFailAlloc_1492_, 5, v_cache_1462_);
lean_ctor_set(v_reuseFailAlloc_1492_, 6, v_messages_1463_);
lean_ctor_set(v_reuseFailAlloc_1492_, 7, v_infoState_1464_);
lean_ctor_set(v_reuseFailAlloc_1492_, 8, v_snapshotTasks_1465_);
v___x_1486_ = v_reuseFailAlloc_1492_;
goto v_reusejp_1485_;
}
v_reusejp_1485_:
{
lean_object* v___x_1487_; lean_object* v___x_1488_; lean_object* v___x_1490_; 
v___x_1487_ = lean_st_ref_set(v___y_1448_, v___x_1486_);
v___x_1488_ = lean_box(0);
if (v_isShared_1455_ == 0)
{
lean_ctor_set(v___x_1454_, 0, v___x_1488_);
v___x_1490_ = v___x_1454_;
goto v_reusejp_1489_;
}
else
{
lean_object* v_reuseFailAlloc_1491_; 
v_reuseFailAlloc_1491_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1491_, 0, v___x_1488_);
v___x_1490_ = v_reuseFailAlloc_1491_;
goto v_reusejp_1489_;
}
v_reusejp_1489_:
{
return v___x_1490_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addTrace___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__2___boxed(lean_object* v_cls_1497_, lean_object* v_msg_1498_, lean_object* v___y_1499_, lean_object* v___y_1500_, lean_object* v___y_1501_, lean_object* v___y_1502_, lean_object* v___y_1503_){
_start:
{
lean_object* v_res_1504_; 
v_res_1504_ = lp_aesop_Lean_addTrace___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__2(v_cls_1497_, v_msg_1498_, v___y_1499_, v___y_1500_, v___y_1501_, v___y_1502_);
lean_dec(v___y_1502_);
lean_dec_ref(v___y_1501_);
lean_dec(v___y_1500_);
lean_dec_ref(v___y_1499_);
return v_res_1504_;
}
}
static lean_object* _init_lp_aesop_Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder___lam__0___closed__1(void){
_start:
{
lean_object* v___x_1506_; lean_object* v___x_1507_; 
v___x_1506_ = ((lean_object*)(lp_aesop_Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder___lam__0___closed__0));
v___x_1507_ = l_Lean_stringToMessageData(v___x_1506_);
return v___x_1507_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder___lam__0(lean_object* v_s_1508_, lean_object* v___x_1509_, lean_object* v___y_1510_, lean_object* v___y_1511_, lean_object* v___y_1512_, lean_object* v___y_1513_){
_start:
{
lean_object* v___x_1515_; 
v___x_1515_ = l_Lean_Meta_saveState___redArg(v___y_1511_, v___y_1513_);
if (lean_obj_tag(v___x_1515_) == 0)
{
lean_object* v_a_1516_; lean_object* v_tacticBuilders_1517_; lean_object* v___x_1518_; lean_object* v___x_1519_; lean_object* v___x_1520_; lean_object* v___x_1521_; lean_object* v___x_1522_; lean_object* v___x_1523_; lean_object* v___x_1524_; 
v_a_1516_ = lean_ctor_get(v___x_1515_, 0);
lean_inc(v_a_1516_);
lean_dec_ref_known(v___x_1515_, 1);
v_tacticBuilders_1517_ = lean_ctor_get(v_s_1508_, 2);
lean_inc_ref_n(v_tacticBuilders_1517_, 2);
v___x_1518_ = lean_unsigned_to_nat(0u);
v___x_1519_ = lean_array_get_size(v_tacticBuilders_1517_);
v___x_1520_ = lean_unsigned_to_nat(1u);
v___x_1521_ = lean_nat_sub(v___x_1519_, v___x_1520_);
lean_inc(v___x_1521_);
v___x_1522_ = l_Array_toSubarray___redArg(v_tacticBuilders_1517_, v___x_1518_, v___x_1521_);
v___x_1523_ = ((lean_object*)(lp_aesop_WellFounded_opaqueFix_u2083___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__0___redArg___closed__0));
v___x_1524_ = lp_aesop_WellFounded_opaqueFix_u2083___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__0___redArg(v_s_1508_, v_a_1516_, v___x_1522_, v___x_1523_, v___y_1510_, v___y_1511_, v___y_1512_, v___y_1513_);
lean_dec(v_a_1516_);
if (lean_obj_tag(v___x_1524_) == 0)
{
lean_object* v_a_1525_; lean_object* v___x_1527_; uint8_t v_isShared_1528_; uint8_t v_isSharedCheck_1576_; 
v_a_1525_ = lean_ctor_get(v___x_1524_, 0);
v_isSharedCheck_1576_ = !lean_is_exclusive(v___x_1524_);
if (v_isSharedCheck_1576_ == 0)
{
v___x_1527_ = v___x_1524_;
v_isShared_1528_ = v_isSharedCheck_1576_;
goto v_resetjp_1526_;
}
else
{
lean_inc(v_a_1525_);
lean_dec(v___x_1524_);
v___x_1527_ = lean_box(0);
v_isShared_1528_ = v_isSharedCheck_1576_;
goto v_resetjp_1526_;
}
v_resetjp_1526_:
{
lean_object* v_fst_1529_; 
v_fst_1529_ = lean_ctor_get(v_a_1525_, 0);
lean_inc(v_fst_1529_);
lean_dec(v_a_1525_);
if (lean_obj_tag(v_fst_1529_) == 0)
{
lean_object* v___x_8054__overap_1530_; lean_object* v___x_1531_; 
lean_del_object(v___x_1527_);
v___x_8054__overap_1530_ = lean_array_fget(v_tacticBuilders_1517_, v___x_1521_);
lean_dec(v___x_1521_);
lean_dec_ref(v_tacticBuilders_1517_);
lean_inc(v___y_1513_);
lean_inc_ref(v___y_1512_);
lean_inc(v___y_1511_);
lean_inc_ref(v___y_1510_);
v___x_1531_ = lean_apply_5(v___x_8054__overap_1530_, v___y_1510_, v___y_1511_, v___y_1512_, v___y_1513_, lean_box(0));
if (lean_obj_tag(v___x_1531_) == 0)
{
lean_object* v_a_1532_; lean_object* v___x_1533_; lean_object* v_a_1534_; lean_object* v___x_1536_; uint8_t v_isShared_1537_; uint8_t v_isSharedCheck_1571_; 
v_a_1532_ = lean_ctor_get(v___x_1531_, 0);
lean_inc(v_a_1532_);
lean_dec_ref_known(v___x_1531_, 1);
v___x_1533_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__1___redArg(v___x_1509_, v___y_1512_);
v_a_1534_ = lean_ctor_get(v___x_1533_, 0);
v_isSharedCheck_1571_ = !lean_is_exclusive(v___x_1533_);
if (v_isSharedCheck_1571_ == 0)
{
v___x_1536_ = v___x_1533_;
v_isShared_1537_ = v_isSharedCheck_1571_;
goto v_resetjp_1535_;
}
else
{
lean_inc(v_a_1534_);
lean_dec(v___x_1533_);
v___x_1536_ = lean_box(0);
v_isShared_1537_ = v_isSharedCheck_1571_;
goto v_resetjp_1535_;
}
v_resetjp_1535_:
{
uint8_t v___x_1538_; 
v___x_1538_ = lean_unbox(v_a_1534_);
lean_dec(v_a_1534_);
if (v___x_1538_ == 0)
{
lean_object* v___x_1540_; 
lean_dec(v___y_1513_);
lean_dec_ref(v___y_1512_);
lean_dec(v___y_1511_);
lean_dec_ref(v___y_1510_);
lean_dec_ref(v___x_1509_);
if (v_isShared_1537_ == 0)
{
lean_ctor_set(v___x_1536_, 0, v_a_1532_);
v___x_1540_ = v___x_1536_;
goto v_reusejp_1539_;
}
else
{
lean_object* v_reuseFailAlloc_1541_; 
v_reuseFailAlloc_1541_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1541_, 0, v_a_1532_);
v___x_1540_ = v_reuseFailAlloc_1541_;
goto v_reusejp_1539_;
}
v_reusejp_1539_:
{
return v___x_1540_;
}
}
else
{
lean_object* v_traceClass_1542_; lean_object* v___x_1544_; uint8_t v_isShared_1545_; uint8_t v_isSharedCheck_1569_; 
lean_del_object(v___x_1536_);
v_traceClass_1542_ = lean_ctor_get(v___x_1509_, 0);
v_isSharedCheck_1569_ = !lean_is_exclusive(v___x_1509_);
if (v_isSharedCheck_1569_ == 0)
{
lean_object* v_unused_1570_; 
v_unused_1570_ = lean_ctor_get(v___x_1509_, 1);
lean_dec(v_unused_1570_);
v___x_1544_ = v___x_1509_;
v_isShared_1545_ = v_isSharedCheck_1569_;
goto v_resetjp_1543_;
}
else
{
lean_inc(v_traceClass_1542_);
lean_dec(v___x_1509_);
v___x_1544_ = lean_box(0);
v_isShared_1545_ = v_isSharedCheck_1569_;
goto v_resetjp_1543_;
}
v_resetjp_1543_:
{
lean_object* v_uTactic_1546_; lean_object* v___x_1547_; lean_object* v___x_1548_; lean_object* v___x_1550_; 
v_uTactic_1546_ = lean_ctor_get(v_a_1532_, 0);
v___x_1547_ = lean_obj_once(&lp_aesop_Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder___lam__0___closed__1, &lp_aesop_Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder___lam__0___closed__1_once, _init_lp_aesop_Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder___lam__0___closed__1);
lean_inc(v_uTactic_1546_);
v___x_1548_ = l_Lean_MessageData_ofSyntax(v_uTactic_1546_);
if (v_isShared_1545_ == 0)
{
lean_ctor_set_tag(v___x_1544_, 7);
lean_ctor_set(v___x_1544_, 1, v___x_1548_);
lean_ctor_set(v___x_1544_, 0, v___x_1547_);
v___x_1550_ = v___x_1544_;
goto v_reusejp_1549_;
}
else
{
lean_object* v_reuseFailAlloc_1568_; 
v_reuseFailAlloc_1568_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1568_, 0, v___x_1547_);
lean_ctor_set(v_reuseFailAlloc_1568_, 1, v___x_1548_);
v___x_1550_ = v_reuseFailAlloc_1568_;
goto v_reusejp_1549_;
}
v_reusejp_1549_:
{
lean_object* v___x_1551_; 
v___x_1551_ = lp_aesop_Lean_addTrace___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__2(v_traceClass_1542_, v___x_1550_, v___y_1510_, v___y_1511_, v___y_1512_, v___y_1513_);
lean_dec(v___y_1513_);
lean_dec_ref(v___y_1512_);
lean_dec(v___y_1511_);
lean_dec_ref(v___y_1510_);
if (lean_obj_tag(v___x_1551_) == 0)
{
lean_object* v___x_1553_; uint8_t v_isShared_1554_; uint8_t v_isSharedCheck_1558_; 
v_isSharedCheck_1558_ = !lean_is_exclusive(v___x_1551_);
if (v_isSharedCheck_1558_ == 0)
{
lean_object* v_unused_1559_; 
v_unused_1559_ = lean_ctor_get(v___x_1551_, 0);
lean_dec(v_unused_1559_);
v___x_1553_ = v___x_1551_;
v_isShared_1554_ = v_isSharedCheck_1558_;
goto v_resetjp_1552_;
}
else
{
lean_dec(v___x_1551_);
v___x_1553_ = lean_box(0);
v_isShared_1554_ = v_isSharedCheck_1558_;
goto v_resetjp_1552_;
}
v_resetjp_1552_:
{
lean_object* v___x_1556_; 
if (v_isShared_1554_ == 0)
{
lean_ctor_set(v___x_1553_, 0, v_a_1532_);
v___x_1556_ = v___x_1553_;
goto v_reusejp_1555_;
}
else
{
lean_object* v_reuseFailAlloc_1557_; 
v_reuseFailAlloc_1557_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1557_, 0, v_a_1532_);
v___x_1556_ = v_reuseFailAlloc_1557_;
goto v_reusejp_1555_;
}
v_reusejp_1555_:
{
return v___x_1556_;
}
}
}
else
{
lean_object* v_a_1560_; lean_object* v___x_1562_; uint8_t v_isShared_1563_; uint8_t v_isSharedCheck_1567_; 
lean_dec(v_a_1532_);
v_a_1560_ = lean_ctor_get(v___x_1551_, 0);
v_isSharedCheck_1567_ = !lean_is_exclusive(v___x_1551_);
if (v_isSharedCheck_1567_ == 0)
{
v___x_1562_ = v___x_1551_;
v_isShared_1563_ = v_isSharedCheck_1567_;
goto v_resetjp_1561_;
}
else
{
lean_inc(v_a_1560_);
lean_dec(v___x_1551_);
v___x_1562_ = lean_box(0);
v_isShared_1563_ = v_isSharedCheck_1567_;
goto v_resetjp_1561_;
}
v_resetjp_1561_:
{
lean_object* v___x_1565_; 
if (v_isShared_1563_ == 0)
{
v___x_1565_ = v___x_1562_;
goto v_reusejp_1564_;
}
else
{
lean_object* v_reuseFailAlloc_1566_; 
v_reuseFailAlloc_1566_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1566_, 0, v_a_1560_);
v___x_1565_ = v_reuseFailAlloc_1566_;
goto v_reusejp_1564_;
}
v_reusejp_1564_:
{
return v___x_1565_;
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
lean_dec(v___y_1513_);
lean_dec_ref(v___y_1512_);
lean_dec(v___y_1511_);
lean_dec_ref(v___y_1510_);
lean_dec_ref(v___x_1509_);
return v___x_1531_;
}
}
else
{
lean_object* v_val_1572_; lean_object* v___x_1574_; 
lean_dec(v___x_1521_);
lean_dec_ref(v_tacticBuilders_1517_);
lean_dec(v___y_1513_);
lean_dec_ref(v___y_1512_);
lean_dec(v___y_1511_);
lean_dec_ref(v___y_1510_);
lean_dec_ref(v___x_1509_);
v_val_1572_ = lean_ctor_get(v_fst_1529_, 0);
lean_inc(v_val_1572_);
lean_dec_ref_known(v_fst_1529_, 1);
if (v_isShared_1528_ == 0)
{
lean_ctor_set(v___x_1527_, 0, v_val_1572_);
v___x_1574_ = v___x_1527_;
goto v_reusejp_1573_;
}
else
{
lean_object* v_reuseFailAlloc_1575_; 
v_reuseFailAlloc_1575_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1575_, 0, v_val_1572_);
v___x_1574_ = v_reuseFailAlloc_1575_;
goto v_reusejp_1573_;
}
v_reusejp_1573_:
{
return v___x_1574_;
}
}
}
}
else
{
lean_object* v_a_1577_; lean_object* v___x_1579_; uint8_t v_isShared_1580_; uint8_t v_isSharedCheck_1584_; 
lean_dec(v___x_1521_);
lean_dec_ref(v_tacticBuilders_1517_);
lean_dec(v___y_1513_);
lean_dec_ref(v___y_1512_);
lean_dec(v___y_1511_);
lean_dec_ref(v___y_1510_);
lean_dec_ref(v___x_1509_);
v_a_1577_ = lean_ctor_get(v___x_1524_, 0);
v_isSharedCheck_1584_ = !lean_is_exclusive(v___x_1524_);
if (v_isSharedCheck_1584_ == 0)
{
v___x_1579_ = v___x_1524_;
v_isShared_1580_ = v_isSharedCheck_1584_;
goto v_resetjp_1578_;
}
else
{
lean_inc(v_a_1577_);
lean_dec(v___x_1524_);
v___x_1579_ = lean_box(0);
v_isShared_1580_ = v_isSharedCheck_1584_;
goto v_resetjp_1578_;
}
v_resetjp_1578_:
{
lean_object* v___x_1582_; 
if (v_isShared_1580_ == 0)
{
v___x_1582_ = v___x_1579_;
goto v_reusejp_1581_;
}
else
{
lean_object* v_reuseFailAlloc_1583_; 
v_reuseFailAlloc_1583_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1583_, 0, v_a_1577_);
v___x_1582_ = v_reuseFailAlloc_1583_;
goto v_reusejp_1581_;
}
v_reusejp_1581_:
{
return v___x_1582_;
}
}
}
}
else
{
lean_object* v_a_1585_; lean_object* v___x_1587_; uint8_t v_isShared_1588_; uint8_t v_isSharedCheck_1592_; 
lean_dec(v___y_1513_);
lean_dec_ref(v___y_1512_);
lean_dec(v___y_1511_);
lean_dec_ref(v___y_1510_);
lean_dec_ref(v___x_1509_);
lean_dec_ref(v_s_1508_);
v_a_1585_ = lean_ctor_get(v___x_1515_, 0);
v_isSharedCheck_1592_ = !lean_is_exclusive(v___x_1515_);
if (v_isSharedCheck_1592_ == 0)
{
v___x_1587_ = v___x_1515_;
v_isShared_1588_ = v_isSharedCheck_1592_;
goto v_resetjp_1586_;
}
else
{
lean_inc(v_a_1585_);
lean_dec(v___x_1515_);
v___x_1587_ = lean_box(0);
v_isShared_1588_ = v_isSharedCheck_1592_;
goto v_resetjp_1586_;
}
v_resetjp_1586_:
{
lean_object* v___x_1590_; 
if (v_isShared_1588_ == 0)
{
v___x_1590_ = v___x_1587_;
goto v_reusejp_1589_;
}
else
{
lean_object* v_reuseFailAlloc_1591_; 
v_reuseFailAlloc_1591_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1591_, 0, v_a_1585_);
v___x_1590_ = v_reuseFailAlloc_1591_;
goto v_reusejp_1589_;
}
v_reusejp_1589_:
{
return v___x_1590_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder___lam__0___boxed(lean_object* v_s_1593_, lean_object* v___x_1594_, lean_object* v___y_1595_, lean_object* v___y_1596_, lean_object* v___y_1597_, lean_object* v___y_1598_, lean_object* v___y_1599_){
_start:
{
lean_object* v_res_1600_; 
v_res_1600_ = lp_aesop_Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder___lam__0(v_s_1593_, v___x_1594_, v___y_1595_, v___y_1596_, v___y_1597_, v___y_1598_);
return v_res_1600_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder___lam__1(lean_object* v___x_1601_, lean_object* v_x_1602_, lean_object* v___y_1603_, lean_object* v___y_1604_, lean_object* v___y_1605_, lean_object* v___y_1606_){
_start:
{
lean_object* v___x_1608_; 
v___x_1608_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1608_, 0, v___x_1601_);
return v___x_1608_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder___lam__1___boxed(lean_object* v___x_1609_, lean_object* v_x_1610_, lean_object* v___y_1611_, lean_object* v___y_1612_, lean_object* v___y_1613_, lean_object* v___y_1614_, lean_object* v___y_1615_){
_start:
{
lean_object* v_res_1616_; 
v_res_1616_ = lp_aesop_Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder___lam__1(v___x_1609_, v_x_1610_, v___y_1611_, v___y_1612_, v___y_1613_, v___y_1614_);
lean_dec(v___y_1614_);
lean_dec_ref(v___y_1613_);
lean_dec(v___y_1612_);
lean_dec_ref(v___y_1611_);
lean_dec_ref(v_x_1610_);
return v_res_1616_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__4_spec__5(lean_object* v_e_1617_){
_start:
{
if (lean_obj_tag(v_e_1617_) == 0)
{
uint8_t v___x_1618_; 
v___x_1618_ = 2;
return v___x_1618_;
}
else
{
uint8_t v___x_1619_; 
v___x_1619_ = 0;
return v___x_1619_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__4_spec__5___boxed(lean_object* v_e_1620_){
_start:
{
uint8_t v_res_1621_; lean_object* v_r_1622_; 
v_res_1621_ = lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__4_spec__5(v_e_1620_);
lean_dec_ref(v_e_1620_);
v_r_1622_ = lean_box(v_res_1621_);
return v_r_1622_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__4(lean_object* v_cls_1623_, uint8_t v_collapsed_1624_, lean_object* v_tag_1625_, lean_object* v_opts_1626_, uint8_t v_clsEnabled_1627_, lean_object* v_oldTraces_1628_, lean_object* v_msg_1629_, lean_object* v_resStartStop_1630_, lean_object* v___y_1631_, lean_object* v___y_1632_, lean_object* v___y_1633_, lean_object* v___y_1634_){
_start:
{
lean_object* v_fst_1636_; lean_object* v_snd_1637_; lean_object* v___y_1639_; lean_object* v___y_1640_; lean_object* v_data_1641_; lean_object* v_fst_1652_; lean_object* v_snd_1653_; lean_object* v___x_1654_; uint8_t v___x_1655_; lean_object* v___y_1657_; lean_object* v_a_1658_; uint8_t v___y_1673_; double v___y_1704_; 
v_fst_1636_ = lean_ctor_get(v_resStartStop_1630_, 0);
lean_inc(v_fst_1636_);
v_snd_1637_ = lean_ctor_get(v_resStartStop_1630_, 1);
lean_inc(v_snd_1637_);
lean_dec_ref(v_resStartStop_1630_);
v_fst_1652_ = lean_ctor_get(v_snd_1637_, 0);
lean_inc(v_fst_1652_);
v_snd_1653_ = lean_ctor_get(v_snd_1637_, 1);
lean_inc(v_snd_1653_);
lean_dec(v_snd_1637_);
v___x_1654_ = l_Lean_trace_profiler;
v___x_1655_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__2(v_opts_1626_, v___x_1654_);
if (v___x_1655_ == 0)
{
v___y_1673_ = v___x_1655_;
goto v___jp_1672_;
}
else
{
lean_object* v___x_1709_; uint8_t v___x_1710_; 
v___x_1709_ = l_Lean_trace_profiler_useHeartbeats;
v___x_1710_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__2(v_opts_1626_, v___x_1709_);
if (v___x_1710_ == 0)
{
lean_object* v___x_1711_; lean_object* v___x_1712_; double v___x_1713_; double v___x_1714_; double v___x_1715_; 
v___x_1711_ = l_Lean_trace_profiler_threshold;
v___x_1712_ = lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__3_spec__6(v_opts_1626_, v___x_1711_);
v___x_1713_ = lean_float_of_nat(v___x_1712_);
v___x_1714_ = lean_float_once(&lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__3___closed__3, &lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__3___closed__3_once, _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__3___closed__3);
v___x_1715_ = lean_float_div(v___x_1713_, v___x_1714_);
v___y_1704_ = v___x_1715_;
goto v___jp_1703_;
}
else
{
lean_object* v___x_1716_; lean_object* v___x_1717_; double v___x_1718_; 
v___x_1716_ = l_Lean_trace_profiler_threshold;
v___x_1717_ = lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__3_spec__6(v_opts_1626_, v___x_1716_);
v___x_1718_ = lean_float_of_nat(v___x_1717_);
v___y_1704_ = v___x_1718_;
goto v___jp_1703_;
}
}
v___jp_1638_:
{
lean_object* v___x_1642_; 
lean_inc(v___y_1640_);
v___x_1642_ = lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__3_spec__3(v_oldTraces_1628_, v_data_1641_, v___y_1640_, v___y_1639_, v___y_1631_, v___y_1632_, v___y_1633_, v___y_1634_);
if (lean_obj_tag(v___x_1642_) == 0)
{
lean_object* v___x_1643_; 
lean_dec_ref_known(v___x_1642_, 1);
v___x_1643_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__3_spec__4___redArg(v_fst_1636_);
return v___x_1643_;
}
else
{
lean_object* v_a_1644_; lean_object* v___x_1646_; uint8_t v_isShared_1647_; uint8_t v_isSharedCheck_1651_; 
lean_dec(v_fst_1636_);
v_a_1644_ = lean_ctor_get(v___x_1642_, 0);
v_isSharedCheck_1651_ = !lean_is_exclusive(v___x_1642_);
if (v_isSharedCheck_1651_ == 0)
{
v___x_1646_ = v___x_1642_;
v_isShared_1647_ = v_isSharedCheck_1651_;
goto v_resetjp_1645_;
}
else
{
lean_inc(v_a_1644_);
lean_dec(v___x_1642_);
v___x_1646_ = lean_box(0);
v_isShared_1647_ = v_isSharedCheck_1651_;
goto v_resetjp_1645_;
}
v_resetjp_1645_:
{
lean_object* v___x_1649_; 
if (v_isShared_1647_ == 0)
{
v___x_1649_ = v___x_1646_;
goto v_reusejp_1648_;
}
else
{
lean_object* v_reuseFailAlloc_1650_; 
v_reuseFailAlloc_1650_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1650_, 0, v_a_1644_);
v___x_1649_ = v_reuseFailAlloc_1650_;
goto v_reusejp_1648_;
}
v_reusejp_1648_:
{
return v___x_1649_;
}
}
}
}
v___jp_1656_:
{
uint8_t v_result_1659_; lean_object* v___x_1660_; lean_object* v___x_1661_; double v___x_1662_; lean_object* v_data_1663_; 
v_result_1659_ = lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__4_spec__5(v_fst_1636_);
v___x_1660_ = lean_box(v_result_1659_);
v___x_1661_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1661_, 0, v___x_1660_);
v___x_1662_ = lean_float_once(&lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__3___closed__0, &lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__3___closed__0_once, _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__3___closed__0);
lean_inc_ref(v_tag_1625_);
lean_inc_ref(v___x_1661_);
lean_inc(v_cls_1623_);
v_data_1663_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v_data_1663_, 0, v_cls_1623_);
lean_ctor_set(v_data_1663_, 1, v___x_1661_);
lean_ctor_set(v_data_1663_, 2, v_tag_1625_);
lean_ctor_set_float(v_data_1663_, sizeof(void*)*3, v___x_1662_);
lean_ctor_set_float(v_data_1663_, sizeof(void*)*3 + 8, v___x_1662_);
lean_ctor_set_uint8(v_data_1663_, sizeof(void*)*3 + 16, v_collapsed_1624_);
if (v___x_1655_ == 0)
{
lean_dec_ref_known(v___x_1661_, 1);
lean_dec(v_snd_1653_);
lean_dec(v_fst_1652_);
lean_dec_ref(v_tag_1625_);
lean_dec(v_cls_1623_);
v___y_1639_ = v_a_1658_;
v___y_1640_ = v___y_1657_;
v_data_1641_ = v_data_1663_;
goto v___jp_1638_;
}
else
{
lean_object* v_data_1664_; double v___x_1665_; double v___x_1666_; 
lean_dec_ref_known(v_data_1663_, 3);
v_data_1664_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v_data_1664_, 0, v_cls_1623_);
lean_ctor_set(v_data_1664_, 1, v___x_1661_);
lean_ctor_set(v_data_1664_, 2, v_tag_1625_);
v___x_1665_ = lean_unbox_float(v_fst_1652_);
lean_dec(v_fst_1652_);
lean_ctor_set_float(v_data_1664_, sizeof(void*)*3, v___x_1665_);
v___x_1666_ = lean_unbox_float(v_snd_1653_);
lean_dec(v_snd_1653_);
lean_ctor_set_float(v_data_1664_, sizeof(void*)*3 + 8, v___x_1666_);
lean_ctor_set_uint8(v_data_1664_, sizeof(void*)*3 + 16, v_collapsed_1624_);
v___y_1639_ = v_a_1658_;
v___y_1640_ = v___y_1657_;
v_data_1641_ = v_data_1664_;
goto v___jp_1638_;
}
}
v___jp_1667_:
{
lean_object* v_ref_1668_; lean_object* v___x_1669_; 
v_ref_1668_ = lean_ctor_get(v___y_1633_, 5);
lean_inc(v___y_1634_);
lean_inc_ref(v___y_1633_);
lean_inc(v___y_1632_);
lean_inc_ref(v___y_1631_);
lean_inc(v_fst_1636_);
v___x_1669_ = lean_apply_6(v_msg_1629_, v_fst_1636_, v___y_1631_, v___y_1632_, v___y_1633_, v___y_1634_, lean_box(0));
if (lean_obj_tag(v___x_1669_) == 0)
{
lean_object* v_a_1670_; 
v_a_1670_ = lean_ctor_get(v___x_1669_, 0);
lean_inc(v_a_1670_);
lean_dec_ref_known(v___x_1669_, 1);
v___y_1657_ = v_ref_1668_;
v_a_1658_ = v_a_1670_;
goto v___jp_1656_;
}
else
{
lean_object* v___x_1671_; 
lean_dec_ref_known(v___x_1669_, 1);
v___x_1671_ = lean_obj_once(&lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__3___closed__2, &lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__3___closed__2_once, _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__3___closed__2);
v___y_1657_ = v_ref_1668_;
v_a_1658_ = v___x_1671_;
goto v___jp_1656_;
}
}
v___jp_1672_:
{
if (v_clsEnabled_1627_ == 0)
{
if (v___y_1673_ == 0)
{
lean_object* v___x_1674_; lean_object* v_traceState_1675_; lean_object* v_env_1676_; lean_object* v_nextMacroScope_1677_; lean_object* v_ngen_1678_; lean_object* v_auxDeclNGen_1679_; lean_object* v_cache_1680_; lean_object* v_messages_1681_; lean_object* v_infoState_1682_; lean_object* v_snapshotTasks_1683_; lean_object* v___x_1685_; uint8_t v_isShared_1686_; uint8_t v_isSharedCheck_1702_; 
lean_dec(v_snd_1653_);
lean_dec(v_fst_1652_);
lean_dec_ref(v_msg_1629_);
lean_dec_ref(v_tag_1625_);
lean_dec(v_cls_1623_);
v___x_1674_ = lean_st_ref_take(v___y_1634_);
v_traceState_1675_ = lean_ctor_get(v___x_1674_, 4);
v_env_1676_ = lean_ctor_get(v___x_1674_, 0);
v_nextMacroScope_1677_ = lean_ctor_get(v___x_1674_, 1);
v_ngen_1678_ = lean_ctor_get(v___x_1674_, 2);
v_auxDeclNGen_1679_ = lean_ctor_get(v___x_1674_, 3);
v_cache_1680_ = lean_ctor_get(v___x_1674_, 5);
v_messages_1681_ = lean_ctor_get(v___x_1674_, 6);
v_infoState_1682_ = lean_ctor_get(v___x_1674_, 7);
v_snapshotTasks_1683_ = lean_ctor_get(v___x_1674_, 8);
v_isSharedCheck_1702_ = !lean_is_exclusive(v___x_1674_);
if (v_isSharedCheck_1702_ == 0)
{
v___x_1685_ = v___x_1674_;
v_isShared_1686_ = v_isSharedCheck_1702_;
goto v_resetjp_1684_;
}
else
{
lean_inc(v_snapshotTasks_1683_);
lean_inc(v_infoState_1682_);
lean_inc(v_messages_1681_);
lean_inc(v_cache_1680_);
lean_inc(v_traceState_1675_);
lean_inc(v_auxDeclNGen_1679_);
lean_inc(v_ngen_1678_);
lean_inc(v_nextMacroScope_1677_);
lean_inc(v_env_1676_);
lean_dec(v___x_1674_);
v___x_1685_ = lean_box(0);
v_isShared_1686_ = v_isSharedCheck_1702_;
goto v_resetjp_1684_;
}
v_resetjp_1684_:
{
uint64_t v_tid_1687_; lean_object* v_traces_1688_; lean_object* v___x_1690_; uint8_t v_isShared_1691_; uint8_t v_isSharedCheck_1701_; 
v_tid_1687_ = lean_ctor_get_uint64(v_traceState_1675_, sizeof(void*)*1);
v_traces_1688_ = lean_ctor_get(v_traceState_1675_, 0);
v_isSharedCheck_1701_ = !lean_is_exclusive(v_traceState_1675_);
if (v_isSharedCheck_1701_ == 0)
{
v___x_1690_ = v_traceState_1675_;
v_isShared_1691_ = v_isSharedCheck_1701_;
goto v_resetjp_1689_;
}
else
{
lean_inc(v_traces_1688_);
lean_dec(v_traceState_1675_);
v___x_1690_ = lean_box(0);
v_isShared_1691_ = v_isSharedCheck_1701_;
goto v_resetjp_1689_;
}
v_resetjp_1689_:
{
lean_object* v___x_1692_; lean_object* v___x_1694_; 
v___x_1692_ = l_Lean_PersistentArray_append___redArg(v_oldTraces_1628_, v_traces_1688_);
lean_dec_ref(v_traces_1688_);
if (v_isShared_1691_ == 0)
{
lean_ctor_set(v___x_1690_, 0, v___x_1692_);
v___x_1694_ = v___x_1690_;
goto v_reusejp_1693_;
}
else
{
lean_object* v_reuseFailAlloc_1700_; 
v_reuseFailAlloc_1700_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_1700_, 0, v___x_1692_);
lean_ctor_set_uint64(v_reuseFailAlloc_1700_, sizeof(void*)*1, v_tid_1687_);
v___x_1694_ = v_reuseFailAlloc_1700_;
goto v_reusejp_1693_;
}
v_reusejp_1693_:
{
lean_object* v___x_1696_; 
if (v_isShared_1686_ == 0)
{
lean_ctor_set(v___x_1685_, 4, v___x_1694_);
v___x_1696_ = v___x_1685_;
goto v_reusejp_1695_;
}
else
{
lean_object* v_reuseFailAlloc_1699_; 
v_reuseFailAlloc_1699_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_1699_, 0, v_env_1676_);
lean_ctor_set(v_reuseFailAlloc_1699_, 1, v_nextMacroScope_1677_);
lean_ctor_set(v_reuseFailAlloc_1699_, 2, v_ngen_1678_);
lean_ctor_set(v_reuseFailAlloc_1699_, 3, v_auxDeclNGen_1679_);
lean_ctor_set(v_reuseFailAlloc_1699_, 4, v___x_1694_);
lean_ctor_set(v_reuseFailAlloc_1699_, 5, v_cache_1680_);
lean_ctor_set(v_reuseFailAlloc_1699_, 6, v_messages_1681_);
lean_ctor_set(v_reuseFailAlloc_1699_, 7, v_infoState_1682_);
lean_ctor_set(v_reuseFailAlloc_1699_, 8, v_snapshotTasks_1683_);
v___x_1696_ = v_reuseFailAlloc_1699_;
goto v_reusejp_1695_;
}
v_reusejp_1695_:
{
lean_object* v___x_1697_; lean_object* v___x_1698_; 
v___x_1697_ = lean_st_ref_set(v___y_1634_, v___x_1696_);
v___x_1698_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__3_spec__4___redArg(v_fst_1636_);
return v___x_1698_;
}
}
}
}
}
else
{
goto v___jp_1667_;
}
}
else
{
goto v___jp_1667_;
}
}
v___jp_1703_:
{
double v___x_1705_; double v___x_1706_; double v___x_1707_; uint8_t v___x_1708_; 
v___x_1705_ = lean_unbox_float(v_snd_1653_);
v___x_1706_ = lean_unbox_float(v_fst_1652_);
v___x_1707_ = lean_float_sub(v___x_1705_, v___x_1706_);
v___x_1708_ = lean_float_decLt(v___y_1704_, v___x_1707_);
v___y_1673_ = v___x_1708_;
goto v___jp_1672_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__4___boxed(lean_object* v_cls_1719_, lean_object* v_collapsed_1720_, lean_object* v_tag_1721_, lean_object* v_opts_1722_, lean_object* v_clsEnabled_1723_, lean_object* v_oldTraces_1724_, lean_object* v_msg_1725_, lean_object* v_resStartStop_1726_, lean_object* v___y_1727_, lean_object* v___y_1728_, lean_object* v___y_1729_, lean_object* v___y_1730_, lean_object* v___y_1731_){
_start:
{
uint8_t v_collapsed_boxed_1732_; uint8_t v_clsEnabled_boxed_1733_; lean_object* v_res_1734_; 
v_collapsed_boxed_1732_ = lean_unbox(v_collapsed_1720_);
v_clsEnabled_boxed_1733_ = lean_unbox(v_clsEnabled_1723_);
v_res_1734_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__4(v_cls_1719_, v_collapsed_boxed_1732_, v_tag_1721_, v_opts_1722_, v_clsEnabled_boxed_1733_, v_oldTraces_1724_, v_msg_1725_, v_resStartStop_1726_, v___y_1727_, v___y_1728_, v___y_1729_, v___y_1730_);
lean_dec(v___y_1730_);
lean_dec_ref(v___y_1729_);
lean_dec(v___y_1728_);
lean_dec_ref(v___y_1727_);
lean_dec_ref(v_opts_1722_);
return v_res_1734_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Options_set___at___00Aesop_withPPAnalyze___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__3_spec__3(lean_object* v_o_1735_, lean_object* v_k_1736_, uint8_t v_v_1737_){
_start:
{
lean_object* v_map_1738_; uint8_t v_hasTrace_1739_; lean_object* v___x_1741_; uint8_t v_isShared_1742_; uint8_t v_isSharedCheck_1753_; 
v_map_1738_ = lean_ctor_get(v_o_1735_, 0);
v_hasTrace_1739_ = lean_ctor_get_uint8(v_o_1735_, sizeof(void*)*1);
v_isSharedCheck_1753_ = !lean_is_exclusive(v_o_1735_);
if (v_isSharedCheck_1753_ == 0)
{
v___x_1741_ = v_o_1735_;
v_isShared_1742_ = v_isSharedCheck_1753_;
goto v_resetjp_1740_;
}
else
{
lean_inc(v_map_1738_);
lean_dec(v_o_1735_);
v___x_1741_ = lean_box(0);
v_isShared_1742_ = v_isSharedCheck_1753_;
goto v_resetjp_1740_;
}
v_resetjp_1740_:
{
lean_object* v___x_1743_; lean_object* v___x_1744_; 
v___x_1743_ = lean_alloc_ctor(1, 0, 1);
lean_ctor_set_uint8(v___x_1743_, 0, v_v_1737_);
lean_inc(v_k_1736_);
v___x_1744_ = l_Std_DTreeMap_Internal_Impl_insert___at___00Lean_NameMap_insert_spec__0___redArg(v_k_1736_, v___x_1743_, v_map_1738_);
if (v_hasTrace_1739_ == 0)
{
lean_object* v___x_1745_; uint8_t v___x_1746_; lean_object* v___x_1748_; 
v___x_1745_ = ((lean_object*)(lp_aesop___private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder___closed__2));
v___x_1746_ = l_Lean_Name_isPrefixOf(v___x_1745_, v_k_1736_);
lean_dec(v_k_1736_);
if (v_isShared_1742_ == 0)
{
lean_ctor_set(v___x_1741_, 0, v___x_1744_);
v___x_1748_ = v___x_1741_;
goto v_reusejp_1747_;
}
else
{
lean_object* v_reuseFailAlloc_1749_; 
v_reuseFailAlloc_1749_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v_reuseFailAlloc_1749_, 0, v___x_1744_);
v___x_1748_ = v_reuseFailAlloc_1749_;
goto v_reusejp_1747_;
}
v_reusejp_1747_:
{
lean_ctor_set_uint8(v___x_1748_, sizeof(void*)*1, v___x_1746_);
return v___x_1748_;
}
}
else
{
lean_object* v___x_1751_; 
lean_dec(v_k_1736_);
if (v_isShared_1742_ == 0)
{
lean_ctor_set(v___x_1741_, 0, v___x_1744_);
v___x_1751_ = v___x_1741_;
goto v_reusejp_1750_;
}
else
{
lean_object* v_reuseFailAlloc_1752_; 
v_reuseFailAlloc_1752_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v_reuseFailAlloc_1752_, 0, v___x_1744_);
lean_ctor_set_uint8(v_reuseFailAlloc_1752_, sizeof(void*)*1, v_hasTrace_1739_);
v___x_1751_ = v_reuseFailAlloc_1752_;
goto v_reusejp_1750_;
}
v_reusejp_1750_:
{
return v___x_1751_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Options_set___at___00Aesop_withPPAnalyze___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__3_spec__3___boxed(lean_object* v_o_1754_, lean_object* v_k_1755_, lean_object* v_v_1756_){
_start:
{
uint8_t v_v_boxed_1757_; lean_object* v_res_1758_; 
v_v_boxed_1757_ = lean_unbox(v_v_1756_);
v_res_1758_ = lp_aesop_Lean_Options_set___at___00Aesop_withPPAnalyze___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__3_spec__3(v_o_1754_, v_k_1755_, v_v_boxed_1757_);
return v_res_1758_;
}
}
static lean_object* _init_lp_aesop_Aesop_withPPAnalyze___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__3___redArg___closed__5(void){
_start:
{
lean_object* v___x_1768_; 
v___x_1768_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_1768_;
}
}
static lean_object* _init_lp_aesop_Aesop_withPPAnalyze___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__3___redArg___closed__6(void){
_start:
{
lean_object* v___x_1769_; lean_object* v___x_1770_; 
v___x_1769_ = lean_obj_once(&lp_aesop_Aesop_withPPAnalyze___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__3___redArg___closed__5, &lp_aesop_Aesop_withPPAnalyze___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__3___redArg___closed__5_once, _init_lp_aesop_Aesop_withPPAnalyze___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__3___redArg___closed__5);
v___x_1770_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1770_, 0, v___x_1769_);
return v___x_1770_;
}
}
static lean_object* _init_lp_aesop_Aesop_withPPAnalyze___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__3___redArg___closed__7(void){
_start:
{
lean_object* v___x_1771_; lean_object* v___x_1772_; 
v___x_1771_ = lean_obj_once(&lp_aesop_Aesop_withPPAnalyze___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__3___redArg___closed__6, &lp_aesop_Aesop_withPPAnalyze___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__3___redArg___closed__6_once, _init_lp_aesop_Aesop_withPPAnalyze___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__3___redArg___closed__6);
v___x_1772_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1772_, 0, v___x_1771_);
lean_ctor_set(v___x_1772_, 1, v___x_1771_);
return v___x_1772_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_withPPAnalyze___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__3___redArg(lean_object* v_x_1773_, lean_object* v___y_1774_, lean_object* v___y_1775_, lean_object* v___y_1776_, lean_object* v___y_1777_){
_start:
{
lean_object* v___x_1779_; lean_object* v_fileName_1780_; lean_object* v_fileMap_1781_; lean_object* v_options_1782_; lean_object* v_currRecDepth_1783_; lean_object* v_ref_1784_; lean_object* v_currNamespace_1785_; lean_object* v_openDecls_1786_; lean_object* v_initHeartbeats_1787_; lean_object* v_maxHeartbeats_1788_; lean_object* v_quotContext_1789_; lean_object* v_currMacroScope_1790_; lean_object* v_cancelTk_x3f_1791_; uint8_t v_suppressElabErrors_1792_; lean_object* v_inheritedTraceOptions_1793_; lean_object* v_env_1794_; lean_object* v___x_1795_; uint8_t v___x_1796_; lean_object* v___x_1797_; lean_object* v___x_1798_; lean_object* v___x_1799_; lean_object* v___x_1800_; uint8_t v___x_1801_; lean_object* v_fileName_1803_; lean_object* v_fileMap_1804_; lean_object* v_currRecDepth_1805_; lean_object* v_ref_1806_; lean_object* v_currNamespace_1807_; lean_object* v_openDecls_1808_; lean_object* v_initHeartbeats_1809_; lean_object* v_maxHeartbeats_1810_; lean_object* v_quotContext_1811_; lean_object* v_currMacroScope_1812_; lean_object* v_cancelTk_x3f_1813_; uint8_t v_suppressElabErrors_1814_; lean_object* v_inheritedTraceOptions_1815_; lean_object* v___y_1816_; uint8_t v___y_1822_; uint8_t v___x_1843_; 
v___x_1779_ = lean_st_ref_get(v___y_1777_);
v_fileName_1780_ = lean_ctor_get(v___y_1776_, 0);
v_fileMap_1781_ = lean_ctor_get(v___y_1776_, 1);
v_options_1782_ = lean_ctor_get(v___y_1776_, 2);
v_currRecDepth_1783_ = lean_ctor_get(v___y_1776_, 3);
v_ref_1784_ = lean_ctor_get(v___y_1776_, 5);
v_currNamespace_1785_ = lean_ctor_get(v___y_1776_, 6);
v_openDecls_1786_ = lean_ctor_get(v___y_1776_, 7);
v_initHeartbeats_1787_ = lean_ctor_get(v___y_1776_, 8);
v_maxHeartbeats_1788_ = lean_ctor_get(v___y_1776_, 9);
v_quotContext_1789_ = lean_ctor_get(v___y_1776_, 10);
v_currMacroScope_1790_ = lean_ctor_get(v___y_1776_, 11);
v_cancelTk_x3f_1791_ = lean_ctor_get(v___y_1776_, 12);
v_suppressElabErrors_1792_ = lean_ctor_get_uint8(v___y_1776_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_1793_ = lean_ctor_get(v___y_1776_, 13);
v_env_1794_ = lean_ctor_get(v___x_1779_, 0);
lean_inc_ref(v_env_1794_);
lean_dec(v___x_1779_);
v___x_1795_ = ((lean_object*)(lp_aesop_Aesop_withPPAnalyze___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__3___redArg___closed__2));
v___x_1796_ = 1;
lean_inc_ref(v_options_1782_);
v___x_1797_ = lp_aesop_Lean_Options_set___at___00Aesop_withPPAnalyze___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__3_spec__3(v_options_1782_, v___x_1795_, v___x_1796_);
v___x_1798_ = ((lean_object*)(lp_aesop_Aesop_withPPAnalyze___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__3___redArg___closed__4));
v___x_1799_ = lp_aesop_Lean_Options_set___at___00Aesop_withPPAnalyze___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__3_spec__3(v___x_1797_, v___x_1798_, v___x_1796_);
v___x_1800_ = l_Lean_diagnostics;
v___x_1801_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__2(v___x_1799_, v___x_1800_);
v___x_1843_ = l_Lean_Kernel_isDiagnosticsEnabled(v_env_1794_);
lean_dec_ref(v_env_1794_);
if (v___x_1843_ == 0)
{
if (v___x_1801_ == 0)
{
v_fileName_1803_ = v_fileName_1780_;
v_fileMap_1804_ = v_fileMap_1781_;
v_currRecDepth_1805_ = v_currRecDepth_1783_;
v_ref_1806_ = v_ref_1784_;
v_currNamespace_1807_ = v_currNamespace_1785_;
v_openDecls_1808_ = v_openDecls_1786_;
v_initHeartbeats_1809_ = v_initHeartbeats_1787_;
v_maxHeartbeats_1810_ = v_maxHeartbeats_1788_;
v_quotContext_1811_ = v_quotContext_1789_;
v_currMacroScope_1812_ = v_currMacroScope_1790_;
v_cancelTk_x3f_1813_ = v_cancelTk_x3f_1791_;
v_suppressElabErrors_1814_ = v_suppressElabErrors_1792_;
v_inheritedTraceOptions_1815_ = v_inheritedTraceOptions_1793_;
v___y_1816_ = v___y_1777_;
goto v___jp_1802_;
}
else
{
v___y_1822_ = v___x_1843_;
goto v___jp_1821_;
}
}
else
{
v___y_1822_ = v___x_1801_;
goto v___jp_1821_;
}
v___jp_1802_:
{
lean_object* v___x_1817_; lean_object* v___x_1818_; lean_object* v___x_1819_; lean_object* v___x_1820_; 
v___x_1817_ = l_Lean_maxRecDepth;
v___x_1818_ = lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__3_spec__6(v___x_1799_, v___x_1817_);
lean_inc_ref(v_inheritedTraceOptions_1815_);
lean_inc(v_cancelTk_x3f_1813_);
lean_inc(v_currMacroScope_1812_);
lean_inc(v_quotContext_1811_);
lean_inc(v_maxHeartbeats_1810_);
lean_inc(v_initHeartbeats_1809_);
lean_inc(v_openDecls_1808_);
lean_inc(v_currNamespace_1807_);
lean_inc(v_ref_1806_);
lean_inc(v_currRecDepth_1805_);
lean_inc_ref(v_fileMap_1804_);
lean_inc_ref(v_fileName_1803_);
v___x_1819_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_1819_, 0, v_fileName_1803_);
lean_ctor_set(v___x_1819_, 1, v_fileMap_1804_);
lean_ctor_set(v___x_1819_, 2, v___x_1799_);
lean_ctor_set(v___x_1819_, 3, v_currRecDepth_1805_);
lean_ctor_set(v___x_1819_, 4, v___x_1818_);
lean_ctor_set(v___x_1819_, 5, v_ref_1806_);
lean_ctor_set(v___x_1819_, 6, v_currNamespace_1807_);
lean_ctor_set(v___x_1819_, 7, v_openDecls_1808_);
lean_ctor_set(v___x_1819_, 8, v_initHeartbeats_1809_);
lean_ctor_set(v___x_1819_, 9, v_maxHeartbeats_1810_);
lean_ctor_set(v___x_1819_, 10, v_quotContext_1811_);
lean_ctor_set(v___x_1819_, 11, v_currMacroScope_1812_);
lean_ctor_set(v___x_1819_, 12, v_cancelTk_x3f_1813_);
lean_ctor_set(v___x_1819_, 13, v_inheritedTraceOptions_1815_);
lean_ctor_set_uint8(v___x_1819_, sizeof(void*)*14, v___x_1801_);
lean_ctor_set_uint8(v___x_1819_, sizeof(void*)*14 + 1, v_suppressElabErrors_1814_);
lean_inc(v___y_1816_);
lean_inc(v___y_1775_);
lean_inc_ref(v___y_1774_);
v___x_1820_ = lean_apply_5(v_x_1773_, v___y_1774_, v___y_1775_, v___x_1819_, v___y_1816_, lean_box(0));
return v___x_1820_;
}
v___jp_1821_:
{
if (v___y_1822_ == 0)
{
lean_object* v___x_1823_; lean_object* v_env_1824_; lean_object* v_nextMacroScope_1825_; lean_object* v_ngen_1826_; lean_object* v_auxDeclNGen_1827_; lean_object* v_traceState_1828_; lean_object* v_messages_1829_; lean_object* v_infoState_1830_; lean_object* v_snapshotTasks_1831_; lean_object* v___x_1833_; uint8_t v_isShared_1834_; uint8_t v_isSharedCheck_1841_; 
v___x_1823_ = lean_st_ref_take(v___y_1777_);
v_env_1824_ = lean_ctor_get(v___x_1823_, 0);
v_nextMacroScope_1825_ = lean_ctor_get(v___x_1823_, 1);
v_ngen_1826_ = lean_ctor_get(v___x_1823_, 2);
v_auxDeclNGen_1827_ = lean_ctor_get(v___x_1823_, 3);
v_traceState_1828_ = lean_ctor_get(v___x_1823_, 4);
v_messages_1829_ = lean_ctor_get(v___x_1823_, 6);
v_infoState_1830_ = lean_ctor_get(v___x_1823_, 7);
v_snapshotTasks_1831_ = lean_ctor_get(v___x_1823_, 8);
v_isSharedCheck_1841_ = !lean_is_exclusive(v___x_1823_);
if (v_isSharedCheck_1841_ == 0)
{
lean_object* v_unused_1842_; 
v_unused_1842_ = lean_ctor_get(v___x_1823_, 5);
lean_dec(v_unused_1842_);
v___x_1833_ = v___x_1823_;
v_isShared_1834_ = v_isSharedCheck_1841_;
goto v_resetjp_1832_;
}
else
{
lean_inc(v_snapshotTasks_1831_);
lean_inc(v_infoState_1830_);
lean_inc(v_messages_1829_);
lean_inc(v_traceState_1828_);
lean_inc(v_auxDeclNGen_1827_);
lean_inc(v_ngen_1826_);
lean_inc(v_nextMacroScope_1825_);
lean_inc(v_env_1824_);
lean_dec(v___x_1823_);
v___x_1833_ = lean_box(0);
v_isShared_1834_ = v_isSharedCheck_1841_;
goto v_resetjp_1832_;
}
v_resetjp_1832_:
{
lean_object* v___x_1835_; lean_object* v___x_1836_; lean_object* v___x_1838_; 
v___x_1835_ = l_Lean_Kernel_enableDiag(v_env_1824_, v___x_1801_);
v___x_1836_ = lean_obj_once(&lp_aesop_Aesop_withPPAnalyze___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__3___redArg___closed__7, &lp_aesop_Aesop_withPPAnalyze___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__3___redArg___closed__7_once, _init_lp_aesop_Aesop_withPPAnalyze___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__3___redArg___closed__7);
if (v_isShared_1834_ == 0)
{
lean_ctor_set(v___x_1833_, 5, v___x_1836_);
lean_ctor_set(v___x_1833_, 0, v___x_1835_);
v___x_1838_ = v___x_1833_;
goto v_reusejp_1837_;
}
else
{
lean_object* v_reuseFailAlloc_1840_; 
v_reuseFailAlloc_1840_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_1840_, 0, v___x_1835_);
lean_ctor_set(v_reuseFailAlloc_1840_, 1, v_nextMacroScope_1825_);
lean_ctor_set(v_reuseFailAlloc_1840_, 2, v_ngen_1826_);
lean_ctor_set(v_reuseFailAlloc_1840_, 3, v_auxDeclNGen_1827_);
lean_ctor_set(v_reuseFailAlloc_1840_, 4, v_traceState_1828_);
lean_ctor_set(v_reuseFailAlloc_1840_, 5, v___x_1836_);
lean_ctor_set(v_reuseFailAlloc_1840_, 6, v_messages_1829_);
lean_ctor_set(v_reuseFailAlloc_1840_, 7, v_infoState_1830_);
lean_ctor_set(v_reuseFailAlloc_1840_, 8, v_snapshotTasks_1831_);
v___x_1838_ = v_reuseFailAlloc_1840_;
goto v_reusejp_1837_;
}
v_reusejp_1837_:
{
lean_object* v___x_1839_; 
v___x_1839_ = lean_st_ref_set(v___y_1777_, v___x_1838_);
v_fileName_1803_ = v_fileName_1780_;
v_fileMap_1804_ = v_fileMap_1781_;
v_currRecDepth_1805_ = v_currRecDepth_1783_;
v_ref_1806_ = v_ref_1784_;
v_currNamespace_1807_ = v_currNamespace_1785_;
v_openDecls_1808_ = v_openDecls_1786_;
v_initHeartbeats_1809_ = v_initHeartbeats_1787_;
v_maxHeartbeats_1810_ = v_maxHeartbeats_1788_;
v_quotContext_1811_ = v_quotContext_1789_;
v_currMacroScope_1812_ = v_currMacroScope_1790_;
v_cancelTk_x3f_1813_ = v_cancelTk_x3f_1791_;
v_suppressElabErrors_1814_ = v_suppressElabErrors_1792_;
v_inheritedTraceOptions_1815_ = v_inheritedTraceOptions_1793_;
v___y_1816_ = v___y_1777_;
goto v___jp_1802_;
}
}
}
else
{
v_fileName_1803_ = v_fileName_1780_;
v_fileMap_1804_ = v_fileMap_1781_;
v_currRecDepth_1805_ = v_currRecDepth_1783_;
v_ref_1806_ = v_ref_1784_;
v_currNamespace_1807_ = v_currNamespace_1785_;
v_openDecls_1808_ = v_openDecls_1786_;
v_initHeartbeats_1809_ = v_initHeartbeats_1787_;
v_maxHeartbeats_1810_ = v_maxHeartbeats_1788_;
v_quotContext_1811_ = v_quotContext_1789_;
v_currMacroScope_1812_ = v_currMacroScope_1790_;
v_cancelTk_x3f_1813_ = v_cancelTk_x3f_1791_;
v_suppressElabErrors_1814_ = v_suppressElabErrors_1792_;
v_inheritedTraceOptions_1815_ = v_inheritedTraceOptions_1793_;
v___y_1816_ = v___y_1777_;
goto v___jp_1802_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_withPPAnalyze___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__3___redArg___boxed(lean_object* v_x_1844_, lean_object* v___y_1845_, lean_object* v___y_1846_, lean_object* v___y_1847_, lean_object* v___y_1848_, lean_object* v___y_1849_){
_start:
{
lean_object* v_res_1850_; 
v_res_1850_ = lp_aesop_Aesop_withPPAnalyze___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__3___redArg(v_x_1844_, v___y_1845_, v___y_1846_, v___y_1847_, v___y_1848_);
lean_dec(v___y_1848_);
lean_dec_ref(v___y_1847_);
lean_dec(v___y_1846_);
lean_dec_ref(v___y_1845_);
return v_res_1850_;
}
}
static lean_object* _init_lp_aesop_Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder___closed__1(void){
_start:
{
lean_object* v___x_1852_; lean_object* v___x_1853_; 
v___x_1852_ = ((lean_object*)(lp_aesop_Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder___closed__0));
v___x_1853_ = l_Lean_stringToMessageData(v___x_1852_);
return v___x_1853_;
}
}
static lean_object* _init_lp_aesop_Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder___closed__2(void){
_start:
{
lean_object* v___x_1854_; lean_object* v___f_1855_; 
v___x_1854_ = lean_obj_once(&lp_aesop_Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder___closed__1, &lp_aesop_Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder___closed__1_once, _init_lp_aesop_Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder___closed__1);
v___f_1855_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder___lam__1___boxed), 7, 1);
lean_closure_set(v___f_1855_, 0, v___x_1854_);
return v___f_1855_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder(lean_object* v_s_1856_, lean_object* v_a_1857_, lean_object* v_a_1858_, lean_object* v_a_1859_, lean_object* v_a_1860_){
_start:
{
lean_object* v_options_1862_; lean_object* v_inheritedTraceOptions_1863_; uint8_t v_hasTrace_1864_; lean_object* v___x_1865_; lean_object* v___f_1866_; 
v_options_1862_ = lean_ctor_get(v_a_1859_, 2);
v_inheritedTraceOptions_1863_ = lean_ctor_get(v_a_1859_, 13);
v_hasTrace_1864_ = lean_ctor_get_uint8(v_options_1862_, sizeof(void*)*1);
v___x_1865_ = lp_aesop_Aesop_TraceOption_script;
v___f_1866_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder___lam__0___boxed), 7, 2);
lean_closure_set(v___f_1866_, 0, v_s_1856_);
lean_closure_set(v___f_1866_, 1, v___x_1865_);
if (v_hasTrace_1864_ == 0)
{
lean_object* v___x_1867_; 
v___x_1867_ = lp_aesop_Aesop_withPPAnalyze___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__3___redArg(v___f_1866_, v_a_1857_, v_a_1858_, v_a_1859_, v_a_1860_);
return v___x_1867_;
}
else
{
lean_object* v_traceClass_1868_; lean_object* v___f_1869_; lean_object* v___x_1870_; lean_object* v___x_1871_; lean_object* v___x_1872_; uint8_t v___x_1873_; lean_object* v___y_1875_; lean_object* v___y_1876_; lean_object* v_a_1877_; lean_object* v___y_1890_; lean_object* v___y_1891_; lean_object* v_a_1892_; 
v_traceClass_1868_ = lean_ctor_get(v___x_1865_, 0);
v___f_1869_ = lean_obj_once(&lp_aesop_Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder___closed__2, &lp_aesop_Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder___closed__2_once, _init_lp_aesop_Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder___closed__2);
v___x_1870_ = ((lean_object*)(lp_aesop___private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder___closed__0));
v___x_1871_ = ((lean_object*)(lp_aesop___private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder___closed__2));
lean_inc(v_traceClass_1868_);
v___x_1872_ = l_Lean_Name_append(v___x_1871_, v_traceClass_1868_);
v___x_1873_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_1863_, v_options_1862_, v___x_1872_);
lean_dec(v___x_1872_);
if (v___x_1873_ == 0)
{
lean_object* v___x_1942_; uint8_t v___x_1943_; 
v___x_1942_ = l_Lean_trace_profiler;
v___x_1943_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__2(v_options_1862_, v___x_1942_);
if (v___x_1943_ == 0)
{
lean_object* v___x_1944_; 
v___x_1944_ = lp_aesop_Aesop_withPPAnalyze___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__3___redArg(v___f_1866_, v_a_1857_, v_a_1858_, v_a_1859_, v_a_1860_);
return v___x_1944_;
}
else
{
goto v___jp_1901_;
}
}
else
{
goto v___jp_1901_;
}
v___jp_1874_:
{
lean_object* v___x_1878_; double v___x_1879_; double v___x_1880_; double v___x_1881_; double v___x_1882_; double v___x_1883_; lean_object* v___x_1884_; lean_object* v___x_1885_; lean_object* v___x_1886_; lean_object* v___x_1887_; lean_object* v___x_1888_; 
v___x_1878_ = lean_io_mono_nanos_now();
v___x_1879_ = lean_float_of_nat(v___y_1876_);
v___x_1880_ = lean_float_once(&lp_aesop___private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder___closed__3, &lp_aesop___private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder___closed__3_once, _init_lp_aesop___private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder___closed__3);
v___x_1881_ = lean_float_div(v___x_1879_, v___x_1880_);
v___x_1882_ = lean_float_of_nat(v___x_1878_);
v___x_1883_ = lean_float_div(v___x_1882_, v___x_1880_);
v___x_1884_ = lean_box_float(v___x_1881_);
v___x_1885_ = lean_box_float(v___x_1883_);
v___x_1886_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1886_, 0, v___x_1884_);
lean_ctor_set(v___x_1886_, 1, v___x_1885_);
v___x_1887_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1887_, 0, v_a_1877_);
lean_ctor_set(v___x_1887_, 1, v___x_1886_);
lean_inc(v_traceClass_1868_);
v___x_1888_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__4(v_traceClass_1868_, v_hasTrace_1864_, v___x_1870_, v_options_1862_, v___x_1873_, v___y_1875_, v___f_1869_, v___x_1887_, v_a_1857_, v_a_1858_, v_a_1859_, v_a_1860_);
return v___x_1888_;
}
v___jp_1889_:
{
lean_object* v___x_1893_; double v___x_1894_; double v___x_1895_; lean_object* v___x_1896_; lean_object* v___x_1897_; lean_object* v___x_1898_; lean_object* v___x_1899_; lean_object* v___x_1900_; 
v___x_1893_ = lean_io_get_num_heartbeats();
v___x_1894_ = lean_float_of_nat(v___y_1890_);
v___x_1895_ = lean_float_of_nat(v___x_1893_);
v___x_1896_ = lean_box_float(v___x_1894_);
v___x_1897_ = lean_box_float(v___x_1895_);
v___x_1898_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1898_, 0, v___x_1896_);
lean_ctor_set(v___x_1898_, 1, v___x_1897_);
v___x_1899_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1899_, 0, v_a_1892_);
lean_ctor_set(v___x_1899_, 1, v___x_1898_);
lean_inc(v_traceClass_1868_);
v___x_1900_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__4(v_traceClass_1868_, v_hasTrace_1864_, v___x_1870_, v_options_1862_, v___x_1873_, v___y_1891_, v___f_1869_, v___x_1899_, v_a_1857_, v_a_1858_, v_a_1859_, v_a_1860_);
return v___x_1900_;
}
v___jp_1901_:
{
lean_object* v___x_1902_; lean_object* v_a_1903_; lean_object* v___x_1904_; uint8_t v___x_1905_; 
v___x_1902_ = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__1___redArg(v_a_1860_);
v_a_1903_ = lean_ctor_get(v___x_1902_, 0);
lean_inc(v_a_1903_);
lean_dec_ref(v___x_1902_);
v___x_1904_ = l_Lean_trace_profiler_useHeartbeats;
v___x_1905_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Script_Step_0__Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_tryTacticBuilder_spec__2(v_options_1862_, v___x_1904_);
if (v___x_1905_ == 0)
{
lean_object* v___x_1906_; lean_object* v___x_1907_; 
v___x_1906_ = lean_io_mono_nanos_now();
v___x_1907_ = lp_aesop_Aesop_withPPAnalyze___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__3___redArg(v___f_1866_, v_a_1857_, v_a_1858_, v_a_1859_, v_a_1860_);
if (lean_obj_tag(v___x_1907_) == 0)
{
lean_object* v_a_1908_; lean_object* v___x_1910_; uint8_t v_isShared_1911_; uint8_t v_isSharedCheck_1915_; 
v_a_1908_ = lean_ctor_get(v___x_1907_, 0);
v_isSharedCheck_1915_ = !lean_is_exclusive(v___x_1907_);
if (v_isSharedCheck_1915_ == 0)
{
v___x_1910_ = v___x_1907_;
v_isShared_1911_ = v_isSharedCheck_1915_;
goto v_resetjp_1909_;
}
else
{
lean_inc(v_a_1908_);
lean_dec(v___x_1907_);
v___x_1910_ = lean_box(0);
v_isShared_1911_ = v_isSharedCheck_1915_;
goto v_resetjp_1909_;
}
v_resetjp_1909_:
{
lean_object* v___x_1913_; 
if (v_isShared_1911_ == 0)
{
lean_ctor_set_tag(v___x_1910_, 1);
v___x_1913_ = v___x_1910_;
goto v_reusejp_1912_;
}
else
{
lean_object* v_reuseFailAlloc_1914_; 
v_reuseFailAlloc_1914_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1914_, 0, v_a_1908_);
v___x_1913_ = v_reuseFailAlloc_1914_;
goto v_reusejp_1912_;
}
v_reusejp_1912_:
{
v___y_1875_ = v_a_1903_;
v___y_1876_ = v___x_1906_;
v_a_1877_ = v___x_1913_;
goto v___jp_1874_;
}
}
}
else
{
lean_object* v_a_1916_; lean_object* v___x_1918_; uint8_t v_isShared_1919_; uint8_t v_isSharedCheck_1923_; 
v_a_1916_ = lean_ctor_get(v___x_1907_, 0);
v_isSharedCheck_1923_ = !lean_is_exclusive(v___x_1907_);
if (v_isSharedCheck_1923_ == 0)
{
v___x_1918_ = v___x_1907_;
v_isShared_1919_ = v_isSharedCheck_1923_;
goto v_resetjp_1917_;
}
else
{
lean_inc(v_a_1916_);
lean_dec(v___x_1907_);
v___x_1918_ = lean_box(0);
v_isShared_1919_ = v_isSharedCheck_1923_;
goto v_resetjp_1917_;
}
v_resetjp_1917_:
{
lean_object* v___x_1921_; 
if (v_isShared_1919_ == 0)
{
lean_ctor_set_tag(v___x_1918_, 0);
v___x_1921_ = v___x_1918_;
goto v_reusejp_1920_;
}
else
{
lean_object* v_reuseFailAlloc_1922_; 
v_reuseFailAlloc_1922_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1922_, 0, v_a_1916_);
v___x_1921_ = v_reuseFailAlloc_1922_;
goto v_reusejp_1920_;
}
v_reusejp_1920_:
{
v___y_1875_ = v_a_1903_;
v___y_1876_ = v___x_1906_;
v_a_1877_ = v___x_1921_;
goto v___jp_1874_;
}
}
}
}
else
{
lean_object* v___x_1924_; lean_object* v___x_1925_; 
v___x_1924_ = lean_io_get_num_heartbeats();
v___x_1925_ = lp_aesop_Aesop_withPPAnalyze___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__3___redArg(v___f_1866_, v_a_1857_, v_a_1858_, v_a_1859_, v_a_1860_);
if (lean_obj_tag(v___x_1925_) == 0)
{
lean_object* v_a_1926_; lean_object* v___x_1928_; uint8_t v_isShared_1929_; uint8_t v_isSharedCheck_1933_; 
v_a_1926_ = lean_ctor_get(v___x_1925_, 0);
v_isSharedCheck_1933_ = !lean_is_exclusive(v___x_1925_);
if (v_isSharedCheck_1933_ == 0)
{
v___x_1928_ = v___x_1925_;
v_isShared_1929_ = v_isSharedCheck_1933_;
goto v_resetjp_1927_;
}
else
{
lean_inc(v_a_1926_);
lean_dec(v___x_1925_);
v___x_1928_ = lean_box(0);
v_isShared_1929_ = v_isSharedCheck_1933_;
goto v_resetjp_1927_;
}
v_resetjp_1927_:
{
lean_object* v___x_1931_; 
if (v_isShared_1929_ == 0)
{
lean_ctor_set_tag(v___x_1928_, 1);
v___x_1931_ = v___x_1928_;
goto v_reusejp_1930_;
}
else
{
lean_object* v_reuseFailAlloc_1932_; 
v_reuseFailAlloc_1932_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1932_, 0, v_a_1926_);
v___x_1931_ = v_reuseFailAlloc_1932_;
goto v_reusejp_1930_;
}
v_reusejp_1930_:
{
v___y_1890_ = v___x_1924_;
v___y_1891_ = v_a_1903_;
v_a_1892_ = v___x_1931_;
goto v___jp_1889_;
}
}
}
else
{
lean_object* v_a_1934_; lean_object* v___x_1936_; uint8_t v_isShared_1937_; uint8_t v_isSharedCheck_1941_; 
v_a_1934_ = lean_ctor_get(v___x_1925_, 0);
v_isSharedCheck_1941_ = !lean_is_exclusive(v___x_1925_);
if (v_isSharedCheck_1941_ == 0)
{
v___x_1936_ = v___x_1925_;
v_isShared_1937_ = v_isSharedCheck_1941_;
goto v_resetjp_1935_;
}
else
{
lean_inc(v_a_1934_);
lean_dec(v___x_1925_);
v___x_1936_ = lean_box(0);
v_isShared_1937_ = v_isSharedCheck_1941_;
goto v_resetjp_1935_;
}
v_resetjp_1935_:
{
lean_object* v___x_1939_; 
if (v_isShared_1937_ == 0)
{
lean_ctor_set_tag(v___x_1936_, 0);
v___x_1939_ = v___x_1936_;
goto v_reusejp_1938_;
}
else
{
lean_object* v_reuseFailAlloc_1940_; 
v_reuseFailAlloc_1940_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1940_, 0, v_a_1934_);
v___x_1939_ = v_reuseFailAlloc_1940_;
goto v_reusejp_1938_;
}
v_reusejp_1938_:
{
v___y_1890_ = v___x_1924_;
v___y_1891_ = v_a_1903_;
v_a_1892_ = v___x_1939_;
goto v___jp_1889_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder___boxed(lean_object* v_s_1945_, lean_object* v_a_1946_, lean_object* v_a_1947_, lean_object* v_a_1948_, lean_object* v_a_1949_, lean_object* v_a_1950_){
_start:
{
lean_object* v_res_1951_; 
v_res_1951_ = lp_aesop_Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder(v_s_1945_, v_a_1946_, v_a_1947_, v_a_1948_, v_a_1949_);
lean_dec(v_a_1949_);
lean_dec_ref(v_a_1948_);
lean_dec(v_a_1947_);
lean_dec_ref(v_a_1946_);
return v_res_1951_;
}
}
LEAN_EXPORT lean_object* lp_aesop_WellFounded_opaqueFix_u2083___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__0(lean_object* v_s_1952_, lean_object* v_a_1953_, lean_object* v_inst_1954_, lean_object* v_R_1955_, lean_object* v_a_1956_, lean_object* v_b_1957_, lean_object* v_c_1958_, lean_object* v___y_1959_, lean_object* v___y_1960_, lean_object* v___y_1961_, lean_object* v___y_1962_){
_start:
{
lean_object* v___x_1964_; 
v___x_1964_ = lp_aesop_WellFounded_opaqueFix_u2083___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__0___redArg(v_s_1952_, v_a_1953_, v_a_1956_, v_b_1957_, v___y_1959_, v___y_1960_, v___y_1961_, v___y_1962_);
return v___x_1964_;
}
}
LEAN_EXPORT lean_object* lp_aesop_WellFounded_opaqueFix_u2083___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__0___boxed(lean_object* v_s_1965_, lean_object* v_a_1966_, lean_object* v_inst_1967_, lean_object* v_R_1968_, lean_object* v_a_1969_, lean_object* v_b_1970_, lean_object* v_c_1971_, lean_object* v___y_1972_, lean_object* v___y_1973_, lean_object* v___y_1974_, lean_object* v___y_1975_, lean_object* v___y_1976_){
_start:
{
lean_object* v_res_1977_; 
v_res_1977_ = lp_aesop_WellFounded_opaqueFix_u2083___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__0(v_s_1965_, v_a_1966_, v_inst_1967_, v_R_1968_, v_a_1969_, v_b_1970_, v_c_1971_, v___y_1972_, v___y_1973_, v___y_1974_, v___y_1975_);
lean_dec(v___y_1975_);
lean_dec_ref(v___y_1974_);
lean_dec(v___y_1973_);
lean_dec_ref(v___y_1972_);
lean_dec_ref(v_a_1966_);
return v_res_1977_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__1(lean_object* v_opt_1978_, lean_object* v___y_1979_, lean_object* v___y_1980_, lean_object* v___y_1981_, lean_object* v___y_1982_){
_start:
{
lean_object* v___x_1984_; 
v___x_1984_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__1___redArg(v_opt_1978_, v___y_1981_);
return v___x_1984_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__1___boxed(lean_object* v_opt_1985_, lean_object* v___y_1986_, lean_object* v___y_1987_, lean_object* v___y_1988_, lean_object* v___y_1989_, lean_object* v___y_1990_){
_start:
{
lean_object* v_res_1991_; 
v_res_1991_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__1(v_opt_1985_, v___y_1986_, v___y_1987_, v___y_1988_, v___y_1989_);
lean_dec(v___y_1989_);
lean_dec_ref(v___y_1988_);
lean_dec(v___y_1987_);
lean_dec_ref(v___y_1986_);
lean_dec_ref(v_opt_1985_);
return v_res_1991_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_withPPAnalyze___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__3(lean_object* v_00_u03b1_1992_, lean_object* v_x_1993_, lean_object* v___y_1994_, lean_object* v___y_1995_, lean_object* v___y_1996_, lean_object* v___y_1997_){
_start:
{
lean_object* v___x_1999_; 
v___x_1999_ = lp_aesop_Aesop_withPPAnalyze___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__3___redArg(v_x_1993_, v___y_1994_, v___y_1995_, v___y_1996_, v___y_1997_);
return v___x_1999_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_withPPAnalyze___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__3___boxed(lean_object* v_00_u03b1_2000_, lean_object* v_x_2001_, lean_object* v___y_2002_, lean_object* v___y_2003_, lean_object* v___y_2004_, lean_object* v___y_2005_, lean_object* v___y_2006_){
_start:
{
lean_object* v_res_2007_; 
v_res_2007_ = lp_aesop_Aesop_withPPAnalyze___at___00Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder_spec__3(v_00_u03b1_2000_, v_x_2001_, v___y_2002_, v___y_2003_, v___y_2004_, v___y_2005_);
lean_dec(v___y_2005_);
lean_dec_ref(v___y_2004_);
lean_dec(v___y_2003_);
lean_dec_ref(v___y_2002_);
return v_res_2007_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_LazyStep_toStep_spec__0(size_t v_sz_2008_, size_t v_i_2009_, lean_object* v_bs_2010_, lean_object* v___y_2011_, lean_object* v___y_2012_, lean_object* v___y_2013_, lean_object* v___y_2014_){
_start:
{
uint8_t v___x_2016_; 
v___x_2016_ = lean_usize_dec_lt(v_i_2009_, v_sz_2008_);
if (v___x_2016_ == 0)
{
lean_object* v___x_2017_; 
v___x_2017_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2017_, 0, v_bs_2010_);
return v___x_2017_;
}
else
{
lean_object* v_v_2018_; lean_object* v___x_2019_; 
v_v_2018_ = lean_array_uget_borrowed(v_bs_2010_, v_i_2009_);
lean_inc(v_v_2018_);
v___x_2019_ = lp_aesop_Aesop_GoalWithMVars_ofMVarId(v_v_2018_, v___y_2011_, v___y_2012_, v___y_2013_, v___y_2014_);
if (lean_obj_tag(v___x_2019_) == 0)
{
lean_object* v_a_2020_; lean_object* v___x_2021_; lean_object* v_bs_x27_2022_; size_t v___x_2023_; size_t v___x_2024_; lean_object* v___x_2025_; 
v_a_2020_ = lean_ctor_get(v___x_2019_, 0);
lean_inc(v_a_2020_);
lean_dec_ref_known(v___x_2019_, 1);
v___x_2021_ = lean_unsigned_to_nat(0u);
v_bs_x27_2022_ = lean_array_uset(v_bs_2010_, v_i_2009_, v___x_2021_);
v___x_2023_ = ((size_t)1ULL);
v___x_2024_ = lean_usize_add(v_i_2009_, v___x_2023_);
v___x_2025_ = lean_array_uset(v_bs_x27_2022_, v_i_2009_, v_a_2020_);
v_i_2009_ = v___x_2024_;
v_bs_2010_ = v___x_2025_;
goto _start;
}
else
{
lean_object* v_a_2027_; lean_object* v___x_2029_; uint8_t v_isShared_2030_; uint8_t v_isSharedCheck_2034_; 
lean_dec_ref(v_bs_2010_);
v_a_2027_ = lean_ctor_get(v___x_2019_, 0);
v_isSharedCheck_2034_ = !lean_is_exclusive(v___x_2019_);
if (v_isSharedCheck_2034_ == 0)
{
v___x_2029_ = v___x_2019_;
v_isShared_2030_ = v_isSharedCheck_2034_;
goto v_resetjp_2028_;
}
else
{
lean_inc(v_a_2027_);
lean_dec(v___x_2019_);
v___x_2029_ = lean_box(0);
v_isShared_2030_ = v_isSharedCheck_2034_;
goto v_resetjp_2028_;
}
v_resetjp_2028_:
{
lean_object* v___x_2032_; 
if (v_isShared_2030_ == 0)
{
v___x_2032_ = v___x_2029_;
goto v_reusejp_2031_;
}
else
{
lean_object* v_reuseFailAlloc_2033_; 
v_reuseFailAlloc_2033_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2033_, 0, v_a_2027_);
v___x_2032_ = v_reuseFailAlloc_2033_;
goto v_reusejp_2031_;
}
v_reusejp_2031_:
{
return v___x_2032_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_LazyStep_toStep_spec__0___boxed(lean_object* v_sz_2035_, lean_object* v_i_2036_, lean_object* v_bs_2037_, lean_object* v___y_2038_, lean_object* v___y_2039_, lean_object* v___y_2040_, lean_object* v___y_2041_, lean_object* v___y_2042_){
_start:
{
size_t v_sz_boxed_2043_; size_t v_i_boxed_2044_; lean_object* v_res_2045_; 
v_sz_boxed_2043_ = lean_unbox_usize(v_sz_2035_);
lean_dec(v_sz_2035_);
v_i_boxed_2044_ = lean_unbox_usize(v_i_2036_);
lean_dec(v_i_2036_);
v_res_2045_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_LazyStep_toStep_spec__0(v_sz_boxed_2043_, v_i_boxed_2044_, v_bs_2037_, v___y_2038_, v___y_2039_, v___y_2040_, v___y_2041_);
lean_dec(v___y_2041_);
lean_dec_ref(v___y_2040_);
lean_dec(v___y_2039_);
lean_dec_ref(v___y_2038_);
return v_res_2045_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_LazyStep_toStep___lam__0(lean_object* v_s_2046_, lean_object* v_postGoals_2047_, lean_object* v_preState_2048_, lean_object* v_preGoal_2049_, lean_object* v_postState_2050_, lean_object* v___y_2051_, lean_object* v___y_2052_, lean_object* v___y_2053_, lean_object* v___y_2054_){
_start:
{
lean_object* v___x_2056_; 
v___x_2056_ = lp_aesop_Aesop_Script_LazyStep_runFirstSuccessfulTacticBuilder(v_s_2046_, v___y_2051_, v___y_2052_, v___y_2053_, v___y_2054_);
if (lean_obj_tag(v___x_2056_) == 0)
{
lean_object* v_a_2057_; size_t v_sz_2058_; size_t v___x_2059_; lean_object* v___x_2060_; 
v_a_2057_ = lean_ctor_get(v___x_2056_, 0);
lean_inc(v_a_2057_);
lean_dec_ref_known(v___x_2056_, 1);
v_sz_2058_ = lean_array_size(v_postGoals_2047_);
v___x_2059_ = ((size_t)0ULL);
v___x_2060_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_LazyStep_toStep_spec__0(v_sz_2058_, v___x_2059_, v_postGoals_2047_, v___y_2051_, v___y_2052_, v___y_2053_, v___y_2054_);
if (lean_obj_tag(v___x_2060_) == 0)
{
lean_object* v_a_2061_; lean_object* v___x_2063_; uint8_t v_isShared_2064_; uint8_t v_isSharedCheck_2069_; 
v_a_2061_ = lean_ctor_get(v___x_2060_, 0);
v_isSharedCheck_2069_ = !lean_is_exclusive(v___x_2060_);
if (v_isSharedCheck_2069_ == 0)
{
v___x_2063_ = v___x_2060_;
v_isShared_2064_ = v_isSharedCheck_2069_;
goto v_resetjp_2062_;
}
else
{
lean_inc(v_a_2061_);
lean_dec(v___x_2060_);
v___x_2063_ = lean_box(0);
v_isShared_2064_ = v_isSharedCheck_2069_;
goto v_resetjp_2062_;
}
v_resetjp_2062_:
{
lean_object* v___x_2065_; lean_object* v___x_2067_; 
v___x_2065_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_2065_, 0, v_preState_2048_);
lean_ctor_set(v___x_2065_, 1, v_preGoal_2049_);
lean_ctor_set(v___x_2065_, 2, v_a_2057_);
lean_ctor_set(v___x_2065_, 3, v_postState_2050_);
lean_ctor_set(v___x_2065_, 4, v_a_2061_);
if (v_isShared_2064_ == 0)
{
lean_ctor_set(v___x_2063_, 0, v___x_2065_);
v___x_2067_ = v___x_2063_;
goto v_reusejp_2066_;
}
else
{
lean_object* v_reuseFailAlloc_2068_; 
v_reuseFailAlloc_2068_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2068_, 0, v___x_2065_);
v___x_2067_ = v_reuseFailAlloc_2068_;
goto v_reusejp_2066_;
}
v_reusejp_2066_:
{
return v___x_2067_;
}
}
}
else
{
lean_object* v_a_2070_; lean_object* v___x_2072_; uint8_t v_isShared_2073_; uint8_t v_isSharedCheck_2077_; 
lean_dec(v_a_2057_);
lean_dec_ref(v_postState_2050_);
lean_dec(v_preGoal_2049_);
lean_dec_ref(v_preState_2048_);
v_a_2070_ = lean_ctor_get(v___x_2060_, 0);
v_isSharedCheck_2077_ = !lean_is_exclusive(v___x_2060_);
if (v_isSharedCheck_2077_ == 0)
{
v___x_2072_ = v___x_2060_;
v_isShared_2073_ = v_isSharedCheck_2077_;
goto v_resetjp_2071_;
}
else
{
lean_inc(v_a_2070_);
lean_dec(v___x_2060_);
v___x_2072_ = lean_box(0);
v_isShared_2073_ = v_isSharedCheck_2077_;
goto v_resetjp_2071_;
}
v_resetjp_2071_:
{
lean_object* v___x_2075_; 
if (v_isShared_2073_ == 0)
{
v___x_2075_ = v___x_2072_;
goto v_reusejp_2074_;
}
else
{
lean_object* v_reuseFailAlloc_2076_; 
v_reuseFailAlloc_2076_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2076_, 0, v_a_2070_);
v___x_2075_ = v_reuseFailAlloc_2076_;
goto v_reusejp_2074_;
}
v_reusejp_2074_:
{
return v___x_2075_;
}
}
}
}
else
{
lean_object* v_a_2078_; lean_object* v___x_2080_; uint8_t v_isShared_2081_; uint8_t v_isSharedCheck_2085_; 
lean_dec_ref(v_postState_2050_);
lean_dec(v_preGoal_2049_);
lean_dec_ref(v_preState_2048_);
lean_dec_ref(v_postGoals_2047_);
v_a_2078_ = lean_ctor_get(v___x_2056_, 0);
v_isSharedCheck_2085_ = !lean_is_exclusive(v___x_2056_);
if (v_isSharedCheck_2085_ == 0)
{
v___x_2080_ = v___x_2056_;
v_isShared_2081_ = v_isSharedCheck_2085_;
goto v_resetjp_2079_;
}
else
{
lean_inc(v_a_2078_);
lean_dec(v___x_2056_);
v___x_2080_ = lean_box(0);
v_isShared_2081_ = v_isSharedCheck_2085_;
goto v_resetjp_2079_;
}
v_resetjp_2079_:
{
lean_object* v___x_2083_; 
if (v_isShared_2081_ == 0)
{
v___x_2083_ = v___x_2080_;
goto v_reusejp_2082_;
}
else
{
lean_object* v_reuseFailAlloc_2084_; 
v_reuseFailAlloc_2084_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2084_, 0, v_a_2078_);
v___x_2083_ = v_reuseFailAlloc_2084_;
goto v_reusejp_2082_;
}
v_reusejp_2082_:
{
return v___x_2083_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_LazyStep_toStep___lam__0___boxed(lean_object* v_s_2086_, lean_object* v_postGoals_2087_, lean_object* v_preState_2088_, lean_object* v_preGoal_2089_, lean_object* v_postState_2090_, lean_object* v___y_2091_, lean_object* v___y_2092_, lean_object* v___y_2093_, lean_object* v___y_2094_, lean_object* v___y_2095_){
_start:
{
lean_object* v_res_2096_; 
v_res_2096_ = lp_aesop_Aesop_Script_LazyStep_toStep___lam__0(v_s_2086_, v_postGoals_2087_, v_preState_2088_, v_preGoal_2089_, v_postState_2090_, v___y_2091_, v___y_2092_, v___y_2093_, v___y_2094_);
lean_dec(v___y_2094_);
lean_dec_ref(v___y_2093_);
lean_dec(v___y_2092_);
lean_dec_ref(v___y_2091_);
return v_res_2096_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_LazyStep_toStep(lean_object* v_s_2097_, lean_object* v_a_2098_, lean_object* v_a_2099_, lean_object* v_a_2100_, lean_object* v_a_2101_){
_start:
{
lean_object* v_preState_2103_; lean_object* v_preGoal_2104_; lean_object* v_postState_2105_; lean_object* v_postGoals_2106_; lean_object* v___f_2107_; lean_object* v___x_2108_; 
v_preState_2103_ = lean_ctor_get(v_s_2097_, 0);
lean_inc_ref(v_preState_2103_);
v_preGoal_2104_ = lean_ctor_get(v_s_2097_, 1);
lean_inc(v_preGoal_2104_);
v_postState_2105_ = lean_ctor_get(v_s_2097_, 3);
lean_inc_ref_n(v_postState_2105_, 2);
v_postGoals_2106_ = lean_ctor_get(v_s_2097_, 4);
lean_inc_ref(v_postGoals_2106_);
v___f_2107_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Script_LazyStep_toStep___lam__0___boxed), 10, 5);
lean_closure_set(v___f_2107_, 0, v_s_2097_);
lean_closure_set(v___f_2107_, 1, v_postGoals_2106_);
lean_closure_set(v___f_2107_, 2, v_preState_2103_);
lean_closure_set(v___f_2107_, 3, v_preGoal_2104_);
lean_closure_set(v___f_2107_, 4, v_postState_2105_);
v___x_2108_ = lp_batteries_Lean_Meta_SavedState_runMetaM_x27___redArg(v_postState_2105_, v___f_2107_, v_a_2098_, v_a_2099_, v_a_2100_, v_a_2101_);
return v___x_2108_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_LazyStep_toStep___boxed(lean_object* v_s_2109_, lean_object* v_a_2110_, lean_object* v_a_2111_, lean_object* v_a_2112_, lean_object* v_a_2113_, lean_object* v_a_2114_){
_start:
{
lean_object* v_res_2115_; 
v_res_2115_ = lp_aesop_Aesop_Script_LazyStep_toStep(v_s_2109_, v_a_2110_, v_a_2111_, v_a_2112_, v_a_2113_);
lean_dec(v_a_2113_);
lean_dec_ref(v_a_2112_);
lean_dec(v_a_2111_);
lean_dec_ref(v_a_2110_);
return v_res_2115_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_LazyStep_build___redArg(lean_object* v_preGoal_2116_, lean_object* v_i_2117_, lean_object* v_a_2118_, lean_object* v_a_2119_, lean_object* v_a_2120_, lean_object* v_a_2121_){
_start:
{
lean_object* v___x_2123_; 
v___x_2123_ = l_Lean_Meta_saveState___redArg(v_a_2119_, v_a_2121_);
if (lean_obj_tag(v___x_2123_) == 0)
{
lean_object* v_a_2124_; lean_object* v_tac_2125_; lean_object* v_postGoals_2126_; lean_object* v_tacticBuilder_2127_; lean_object* v___x_2128_; 
v_a_2124_ = lean_ctor_get(v___x_2123_, 0);
lean_inc(v_a_2124_);
lean_dec_ref_known(v___x_2123_, 1);
v_tac_2125_ = lean_ctor_get(v_i_2117_, 0);
lean_inc_ref(v_tac_2125_);
v_postGoals_2126_ = lean_ctor_get(v_i_2117_, 1);
lean_inc_ref(v_postGoals_2126_);
v_tacticBuilder_2127_ = lean_ctor_get(v_i_2117_, 2);
lean_inc_ref(v_tacticBuilder_2127_);
lean_dec_ref(v_i_2117_);
lean_inc(v_a_2121_);
lean_inc_ref(v_a_2120_);
lean_inc(v_a_2119_);
lean_inc_ref(v_a_2118_);
v___x_2128_ = lean_apply_5(v_tac_2125_, v_a_2118_, v_a_2119_, v_a_2120_, v_a_2121_, lean_box(0));
if (lean_obj_tag(v___x_2128_) == 0)
{
lean_object* v_a_2129_; lean_object* v___x_2130_; 
v_a_2129_ = lean_ctor_get(v___x_2128_, 0);
lean_inc(v_a_2129_);
lean_dec_ref_known(v___x_2128_, 1);
v___x_2130_ = l_Lean_Meta_saveState___redArg(v_a_2119_, v_a_2121_);
if (lean_obj_tag(v___x_2130_) == 0)
{
lean_object* v_a_2131_; lean_object* v___x_2133_; uint8_t v_isShared_2134_; uint8_t v_isSharedCheck_2145_; 
v_a_2131_ = lean_ctor_get(v___x_2130_, 0);
v_isSharedCheck_2145_ = !lean_is_exclusive(v___x_2130_);
if (v_isSharedCheck_2145_ == 0)
{
v___x_2133_ = v___x_2130_;
v_isShared_2134_ = v_isSharedCheck_2145_;
goto v_resetjp_2132_;
}
else
{
lean_inc(v_a_2131_);
lean_dec(v___x_2130_);
v___x_2133_ = lean_box(0);
v_isShared_2134_ = v_isSharedCheck_2145_;
goto v_resetjp_2132_;
}
v_resetjp_2132_:
{
lean_object* v___x_2135_; lean_object* v___x_2136_; lean_object* v___x_2137_; lean_object* v___x_2138_; lean_object* v___x_2139_; lean_object* v___x_2140_; lean_object* v___x_2141_; lean_object* v___x_2143_; 
lean_inc_n(v_a_2129_, 2);
v___x_2135_ = lean_apply_1(v_tacticBuilder_2127_, v_a_2129_);
v___x_2136_ = lean_unsigned_to_nat(1u);
v___x_2137_ = lean_mk_empty_array_with_capacity(v___x_2136_);
v___x_2138_ = lean_array_push(v___x_2137_, v___x_2135_);
v___x_2139_ = lean_apply_1(v_postGoals_2126_, v_a_2129_);
v___x_2140_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_2140_, 0, v_a_2124_);
lean_ctor_set(v___x_2140_, 1, v_preGoal_2116_);
lean_ctor_set(v___x_2140_, 2, v___x_2138_);
lean_ctor_set(v___x_2140_, 3, v_a_2131_);
lean_ctor_set(v___x_2140_, 4, v___x_2139_);
v___x_2141_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2141_, 0, v___x_2140_);
lean_ctor_set(v___x_2141_, 1, v_a_2129_);
if (v_isShared_2134_ == 0)
{
lean_ctor_set(v___x_2133_, 0, v___x_2141_);
v___x_2143_ = v___x_2133_;
goto v_reusejp_2142_;
}
else
{
lean_object* v_reuseFailAlloc_2144_; 
v_reuseFailAlloc_2144_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2144_, 0, v___x_2141_);
v___x_2143_ = v_reuseFailAlloc_2144_;
goto v_reusejp_2142_;
}
v_reusejp_2142_:
{
return v___x_2143_;
}
}
}
else
{
lean_object* v_a_2146_; lean_object* v___x_2148_; uint8_t v_isShared_2149_; uint8_t v_isSharedCheck_2153_; 
lean_dec(v_a_2129_);
lean_dec_ref(v_tacticBuilder_2127_);
lean_dec_ref(v_postGoals_2126_);
lean_dec(v_a_2124_);
lean_dec(v_preGoal_2116_);
v_a_2146_ = lean_ctor_get(v___x_2130_, 0);
v_isSharedCheck_2153_ = !lean_is_exclusive(v___x_2130_);
if (v_isSharedCheck_2153_ == 0)
{
v___x_2148_ = v___x_2130_;
v_isShared_2149_ = v_isSharedCheck_2153_;
goto v_resetjp_2147_;
}
else
{
lean_inc(v_a_2146_);
lean_dec(v___x_2130_);
v___x_2148_ = lean_box(0);
v_isShared_2149_ = v_isSharedCheck_2153_;
goto v_resetjp_2147_;
}
v_resetjp_2147_:
{
lean_object* v___x_2151_; 
if (v_isShared_2149_ == 0)
{
v___x_2151_ = v___x_2148_;
goto v_reusejp_2150_;
}
else
{
lean_object* v_reuseFailAlloc_2152_; 
v_reuseFailAlloc_2152_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2152_, 0, v_a_2146_);
v___x_2151_ = v_reuseFailAlloc_2152_;
goto v_reusejp_2150_;
}
v_reusejp_2150_:
{
return v___x_2151_;
}
}
}
}
else
{
lean_object* v_a_2154_; lean_object* v___x_2156_; uint8_t v_isShared_2157_; uint8_t v_isSharedCheck_2161_; 
lean_dec_ref(v_tacticBuilder_2127_);
lean_dec_ref(v_postGoals_2126_);
lean_dec(v_a_2124_);
lean_dec(v_preGoal_2116_);
v_a_2154_ = lean_ctor_get(v___x_2128_, 0);
v_isSharedCheck_2161_ = !lean_is_exclusive(v___x_2128_);
if (v_isSharedCheck_2161_ == 0)
{
v___x_2156_ = v___x_2128_;
v_isShared_2157_ = v_isSharedCheck_2161_;
goto v_resetjp_2155_;
}
else
{
lean_inc(v_a_2154_);
lean_dec(v___x_2128_);
v___x_2156_ = lean_box(0);
v_isShared_2157_ = v_isSharedCheck_2161_;
goto v_resetjp_2155_;
}
v_resetjp_2155_:
{
lean_object* v___x_2159_; 
if (v_isShared_2157_ == 0)
{
v___x_2159_ = v___x_2156_;
goto v_reusejp_2158_;
}
else
{
lean_object* v_reuseFailAlloc_2160_; 
v_reuseFailAlloc_2160_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2160_, 0, v_a_2154_);
v___x_2159_ = v_reuseFailAlloc_2160_;
goto v_reusejp_2158_;
}
v_reusejp_2158_:
{
return v___x_2159_;
}
}
}
}
else
{
lean_object* v_a_2162_; lean_object* v___x_2164_; uint8_t v_isShared_2165_; uint8_t v_isSharedCheck_2169_; 
lean_dec_ref(v_i_2117_);
lean_dec(v_preGoal_2116_);
v_a_2162_ = lean_ctor_get(v___x_2123_, 0);
v_isSharedCheck_2169_ = !lean_is_exclusive(v___x_2123_);
if (v_isSharedCheck_2169_ == 0)
{
v___x_2164_ = v___x_2123_;
v_isShared_2165_ = v_isSharedCheck_2169_;
goto v_resetjp_2163_;
}
else
{
lean_inc(v_a_2162_);
lean_dec(v___x_2123_);
v___x_2164_ = lean_box(0);
v_isShared_2165_ = v_isSharedCheck_2169_;
goto v_resetjp_2163_;
}
v_resetjp_2163_:
{
lean_object* v___x_2167_; 
if (v_isShared_2165_ == 0)
{
v___x_2167_ = v___x_2164_;
goto v_reusejp_2166_;
}
else
{
lean_object* v_reuseFailAlloc_2168_; 
v_reuseFailAlloc_2168_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2168_, 0, v_a_2162_);
v___x_2167_ = v_reuseFailAlloc_2168_;
goto v_reusejp_2166_;
}
v_reusejp_2166_:
{
return v___x_2167_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_LazyStep_build___redArg___boxed(lean_object* v_preGoal_2170_, lean_object* v_i_2171_, lean_object* v_a_2172_, lean_object* v_a_2173_, lean_object* v_a_2174_, lean_object* v_a_2175_, lean_object* v_a_2176_){
_start:
{
lean_object* v_res_2177_; 
v_res_2177_ = lp_aesop_Aesop_Script_LazyStep_build___redArg(v_preGoal_2170_, v_i_2171_, v_a_2172_, v_a_2173_, v_a_2174_, v_a_2175_);
lean_dec(v_a_2175_);
lean_dec_ref(v_a_2174_);
lean_dec(v_a_2173_);
lean_dec_ref(v_a_2172_);
return v_res_2177_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_LazyStep_build(lean_object* v_00_u03b1_2178_, lean_object* v_preGoal_2179_, lean_object* v_i_2180_, lean_object* v_a_2181_, lean_object* v_a_2182_, lean_object* v_a_2183_, lean_object* v_a_2184_){
_start:
{
lean_object* v___x_2186_; 
v___x_2186_ = l_Lean_Meta_saveState___redArg(v_a_2182_, v_a_2184_);
if (lean_obj_tag(v___x_2186_) == 0)
{
lean_object* v_a_2187_; lean_object* v_tac_2188_; lean_object* v_postGoals_2189_; lean_object* v_tacticBuilder_2190_; lean_object* v___x_2191_; 
v_a_2187_ = lean_ctor_get(v___x_2186_, 0);
lean_inc(v_a_2187_);
lean_dec_ref_known(v___x_2186_, 1);
v_tac_2188_ = lean_ctor_get(v_i_2180_, 0);
lean_inc_ref(v_tac_2188_);
v_postGoals_2189_ = lean_ctor_get(v_i_2180_, 1);
lean_inc_ref(v_postGoals_2189_);
v_tacticBuilder_2190_ = lean_ctor_get(v_i_2180_, 2);
lean_inc_ref(v_tacticBuilder_2190_);
lean_dec_ref(v_i_2180_);
lean_inc(v_a_2184_);
lean_inc_ref(v_a_2183_);
lean_inc(v_a_2182_);
lean_inc_ref(v_a_2181_);
v___x_2191_ = lean_apply_5(v_tac_2188_, v_a_2181_, v_a_2182_, v_a_2183_, v_a_2184_, lean_box(0));
if (lean_obj_tag(v___x_2191_) == 0)
{
lean_object* v_a_2192_; lean_object* v___x_2193_; 
v_a_2192_ = lean_ctor_get(v___x_2191_, 0);
lean_inc(v_a_2192_);
lean_dec_ref_known(v___x_2191_, 1);
v___x_2193_ = l_Lean_Meta_saveState___redArg(v_a_2182_, v_a_2184_);
if (lean_obj_tag(v___x_2193_) == 0)
{
lean_object* v_a_2194_; lean_object* v___x_2196_; uint8_t v_isShared_2197_; uint8_t v_isSharedCheck_2208_; 
v_a_2194_ = lean_ctor_get(v___x_2193_, 0);
v_isSharedCheck_2208_ = !lean_is_exclusive(v___x_2193_);
if (v_isSharedCheck_2208_ == 0)
{
v___x_2196_ = v___x_2193_;
v_isShared_2197_ = v_isSharedCheck_2208_;
goto v_resetjp_2195_;
}
else
{
lean_inc(v_a_2194_);
lean_dec(v___x_2193_);
v___x_2196_ = lean_box(0);
v_isShared_2197_ = v_isSharedCheck_2208_;
goto v_resetjp_2195_;
}
v_resetjp_2195_:
{
lean_object* v___x_2198_; lean_object* v___x_2199_; lean_object* v___x_2200_; lean_object* v___x_2201_; lean_object* v___x_2202_; lean_object* v___x_2203_; lean_object* v___x_2204_; lean_object* v___x_2206_; 
lean_inc_n(v_a_2192_, 2);
v___x_2198_ = lean_apply_1(v_tacticBuilder_2190_, v_a_2192_);
v___x_2199_ = lean_unsigned_to_nat(1u);
v___x_2200_ = lean_mk_empty_array_with_capacity(v___x_2199_);
v___x_2201_ = lean_array_push(v___x_2200_, v___x_2198_);
v___x_2202_ = lean_apply_1(v_postGoals_2189_, v_a_2192_);
v___x_2203_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_2203_, 0, v_a_2187_);
lean_ctor_set(v___x_2203_, 1, v_preGoal_2179_);
lean_ctor_set(v___x_2203_, 2, v___x_2201_);
lean_ctor_set(v___x_2203_, 3, v_a_2194_);
lean_ctor_set(v___x_2203_, 4, v___x_2202_);
v___x_2204_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2204_, 0, v___x_2203_);
lean_ctor_set(v___x_2204_, 1, v_a_2192_);
if (v_isShared_2197_ == 0)
{
lean_ctor_set(v___x_2196_, 0, v___x_2204_);
v___x_2206_ = v___x_2196_;
goto v_reusejp_2205_;
}
else
{
lean_object* v_reuseFailAlloc_2207_; 
v_reuseFailAlloc_2207_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2207_, 0, v___x_2204_);
v___x_2206_ = v_reuseFailAlloc_2207_;
goto v_reusejp_2205_;
}
v_reusejp_2205_:
{
return v___x_2206_;
}
}
}
else
{
lean_object* v_a_2209_; lean_object* v___x_2211_; uint8_t v_isShared_2212_; uint8_t v_isSharedCheck_2216_; 
lean_dec(v_a_2192_);
lean_dec_ref(v_tacticBuilder_2190_);
lean_dec_ref(v_postGoals_2189_);
lean_dec(v_a_2187_);
lean_dec(v_preGoal_2179_);
v_a_2209_ = lean_ctor_get(v___x_2193_, 0);
v_isSharedCheck_2216_ = !lean_is_exclusive(v___x_2193_);
if (v_isSharedCheck_2216_ == 0)
{
v___x_2211_ = v___x_2193_;
v_isShared_2212_ = v_isSharedCheck_2216_;
goto v_resetjp_2210_;
}
else
{
lean_inc(v_a_2209_);
lean_dec(v___x_2193_);
v___x_2211_ = lean_box(0);
v_isShared_2212_ = v_isSharedCheck_2216_;
goto v_resetjp_2210_;
}
v_resetjp_2210_:
{
lean_object* v___x_2214_; 
if (v_isShared_2212_ == 0)
{
v___x_2214_ = v___x_2211_;
goto v_reusejp_2213_;
}
else
{
lean_object* v_reuseFailAlloc_2215_; 
v_reuseFailAlloc_2215_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2215_, 0, v_a_2209_);
v___x_2214_ = v_reuseFailAlloc_2215_;
goto v_reusejp_2213_;
}
v_reusejp_2213_:
{
return v___x_2214_;
}
}
}
}
else
{
lean_object* v_a_2217_; lean_object* v___x_2219_; uint8_t v_isShared_2220_; uint8_t v_isSharedCheck_2224_; 
lean_dec_ref(v_tacticBuilder_2190_);
lean_dec_ref(v_postGoals_2189_);
lean_dec(v_a_2187_);
lean_dec(v_preGoal_2179_);
v_a_2217_ = lean_ctor_get(v___x_2191_, 0);
v_isSharedCheck_2224_ = !lean_is_exclusive(v___x_2191_);
if (v_isSharedCheck_2224_ == 0)
{
v___x_2219_ = v___x_2191_;
v_isShared_2220_ = v_isSharedCheck_2224_;
goto v_resetjp_2218_;
}
else
{
lean_inc(v_a_2217_);
lean_dec(v___x_2191_);
v___x_2219_ = lean_box(0);
v_isShared_2220_ = v_isSharedCheck_2224_;
goto v_resetjp_2218_;
}
v_resetjp_2218_:
{
lean_object* v___x_2222_; 
if (v_isShared_2220_ == 0)
{
v___x_2222_ = v___x_2219_;
goto v_reusejp_2221_;
}
else
{
lean_object* v_reuseFailAlloc_2223_; 
v_reuseFailAlloc_2223_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2223_, 0, v_a_2217_);
v___x_2222_ = v_reuseFailAlloc_2223_;
goto v_reusejp_2221_;
}
v_reusejp_2221_:
{
return v___x_2222_;
}
}
}
}
else
{
lean_object* v_a_2225_; lean_object* v___x_2227_; uint8_t v_isShared_2228_; uint8_t v_isSharedCheck_2232_; 
lean_dec_ref(v_i_2180_);
lean_dec(v_preGoal_2179_);
v_a_2225_ = lean_ctor_get(v___x_2186_, 0);
v_isSharedCheck_2232_ = !lean_is_exclusive(v___x_2186_);
if (v_isSharedCheck_2232_ == 0)
{
v___x_2227_ = v___x_2186_;
v_isShared_2228_ = v_isSharedCheck_2232_;
goto v_resetjp_2226_;
}
else
{
lean_inc(v_a_2225_);
lean_dec(v___x_2186_);
v___x_2227_ = lean_box(0);
v_isShared_2228_ = v_isSharedCheck_2232_;
goto v_resetjp_2226_;
}
v_resetjp_2226_:
{
lean_object* v___x_2230_; 
if (v_isShared_2228_ == 0)
{
v___x_2230_ = v___x_2227_;
goto v_reusejp_2229_;
}
else
{
lean_object* v_reuseFailAlloc_2231_; 
v_reuseFailAlloc_2231_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2231_, 0, v_a_2225_);
v___x_2230_ = v_reuseFailAlloc_2231_;
goto v_reusejp_2229_;
}
v_reusejp_2229_:
{
return v___x_2230_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_LazyStep_build___boxed(lean_object* v_00_u03b1_2233_, lean_object* v_preGoal_2234_, lean_object* v_i_2235_, lean_object* v_a_2236_, lean_object* v_a_2237_, lean_object* v_a_2238_, lean_object* v_a_2239_, lean_object* v_a_2240_){
_start:
{
lean_object* v_res_2241_; 
v_res_2241_ = lp_aesop_Aesop_Script_LazyStep_build(v_00_u03b1_2233_, v_preGoal_2234_, v_i_2235_, v_a_2236_, v_a_2237_, v_a_2238_, v_a_2239_);
lean_dec(v_a_2239_);
lean_dec_ref(v_a_2238_);
lean_dec(v_a_2237_);
lean_dec_ref(v_a_2236_);
return v_res_2241_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Script_LazyStep_erasePostStateAssignments_spec__0(lean_object* v_as_2242_, size_t v_i_2243_, size_t v_stop_2244_, lean_object* v_b_2245_){
_start:
{
uint8_t v___x_2246_; 
v___x_2246_ = lean_usize_dec_eq(v_i_2243_, v_stop_2244_);
if (v___x_2246_ == 0)
{
lean_object* v___x_2247_; lean_object* v___x_2248_; size_t v___x_2249_; size_t v___x_2250_; 
v___x_2247_ = lean_array_uget_borrowed(v_as_2242_, v_i_2243_);
v___x_2248_ = lp_batteries_Lean_MetavarContext_eraseExprMVarAssignment(v_b_2245_, v___x_2247_);
v___x_2249_ = ((size_t)1ULL);
v___x_2250_ = lean_usize_add(v_i_2243_, v___x_2249_);
v_i_2243_ = v___x_2250_;
v_b_2245_ = v___x_2248_;
goto _start;
}
else
{
return v_b_2245_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Script_LazyStep_erasePostStateAssignments_spec__0___boxed(lean_object* v_as_2252_, lean_object* v_i_2253_, lean_object* v_stop_2254_, lean_object* v_b_2255_){
_start:
{
size_t v_i_boxed_2256_; size_t v_stop_boxed_2257_; lean_object* v_res_2258_; 
v_i_boxed_2256_ = lean_unbox_usize(v_i_2253_);
lean_dec(v_i_2253_);
v_stop_boxed_2257_ = lean_unbox_usize(v_stop_2254_);
lean_dec(v_stop_2254_);
v_res_2258_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Script_LazyStep_erasePostStateAssignments_spec__0(v_as_2252_, v_i_boxed_2256_, v_stop_boxed_2257_, v_b_2255_);
lean_dec_ref(v_as_2252_);
return v_res_2258_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_LazyStep_erasePostStateAssignments(lean_object* v_s_2259_, lean_object* v_gs_2260_){
_start:
{
lean_object* v_postState_2261_; lean_object* v_meta_2262_; lean_object* v_preState_2263_; lean_object* v_preGoal_2264_; lean_object* v_tacticBuilders_2265_; lean_object* v_postGoals_2266_; lean_object* v___x_2268_; uint8_t v_isShared_2269_; uint8_t v_isSharedCheck_2306_; 
v_postState_2261_ = lean_ctor_get(v_s_2259_, 3);
lean_inc_ref(v_postState_2261_);
v_meta_2262_ = lean_ctor_get(v_postState_2261_, 1);
lean_inc_ref(v_meta_2262_);
v_preState_2263_ = lean_ctor_get(v_s_2259_, 0);
v_preGoal_2264_ = lean_ctor_get(v_s_2259_, 1);
v_tacticBuilders_2265_ = lean_ctor_get(v_s_2259_, 2);
v_postGoals_2266_ = lean_ctor_get(v_s_2259_, 4);
v_isSharedCheck_2306_ = !lean_is_exclusive(v_s_2259_);
if (v_isSharedCheck_2306_ == 0)
{
lean_object* v_unused_2307_; 
v_unused_2307_ = lean_ctor_get(v_s_2259_, 3);
lean_dec(v_unused_2307_);
v___x_2268_ = v_s_2259_;
v_isShared_2269_ = v_isSharedCheck_2306_;
goto v_resetjp_2267_;
}
else
{
lean_inc(v_postGoals_2266_);
lean_inc(v_tacticBuilders_2265_);
lean_inc(v_preGoal_2264_);
lean_inc(v_preState_2263_);
lean_dec(v_s_2259_);
v___x_2268_ = lean_box(0);
v_isShared_2269_ = v_isSharedCheck_2306_;
goto v_resetjp_2267_;
}
v_resetjp_2267_:
{
lean_object* v_core_2270_; lean_object* v___x_2272_; uint8_t v_isShared_2273_; uint8_t v_isSharedCheck_2304_; 
v_core_2270_ = lean_ctor_get(v_postState_2261_, 0);
v_isSharedCheck_2304_ = !lean_is_exclusive(v_postState_2261_);
if (v_isSharedCheck_2304_ == 0)
{
lean_object* v_unused_2305_; 
v_unused_2305_ = lean_ctor_get(v_postState_2261_, 1);
lean_dec(v_unused_2305_);
v___x_2272_ = v_postState_2261_;
v_isShared_2273_ = v_isSharedCheck_2304_;
goto v_resetjp_2271_;
}
else
{
lean_inc(v_core_2270_);
lean_dec(v_postState_2261_);
v___x_2272_ = lean_box(0);
v_isShared_2273_ = v_isSharedCheck_2304_;
goto v_resetjp_2271_;
}
v_resetjp_2271_:
{
lean_object* v_mctx_2274_; lean_object* v_cache_2275_; lean_object* v_zetaDeltaFVarIds_2276_; lean_object* v_postponed_2277_; lean_object* v_diag_2278_; lean_object* v___x_2280_; uint8_t v_isShared_2281_; uint8_t v_isSharedCheck_2303_; 
v_mctx_2274_ = lean_ctor_get(v_meta_2262_, 0);
v_cache_2275_ = lean_ctor_get(v_meta_2262_, 1);
v_zetaDeltaFVarIds_2276_ = lean_ctor_get(v_meta_2262_, 2);
v_postponed_2277_ = lean_ctor_get(v_meta_2262_, 3);
v_diag_2278_ = lean_ctor_get(v_meta_2262_, 4);
v_isSharedCheck_2303_ = !lean_is_exclusive(v_meta_2262_);
if (v_isSharedCheck_2303_ == 0)
{
v___x_2280_ = v_meta_2262_;
v_isShared_2281_ = v_isSharedCheck_2303_;
goto v_resetjp_2279_;
}
else
{
lean_inc(v_diag_2278_);
lean_inc(v_postponed_2277_);
lean_inc(v_zetaDeltaFVarIds_2276_);
lean_inc(v_cache_2275_);
lean_inc(v_mctx_2274_);
lean_dec(v_meta_2262_);
v___x_2280_ = lean_box(0);
v_isShared_2281_ = v_isSharedCheck_2303_;
goto v_resetjp_2279_;
}
v_resetjp_2279_:
{
lean_object* v___y_2283_; lean_object* v___x_2293_; lean_object* v___x_2294_; uint8_t v___x_2295_; 
v___x_2293_ = lean_unsigned_to_nat(0u);
v___x_2294_ = lean_array_get_size(v_gs_2260_);
v___x_2295_ = lean_nat_dec_lt(v___x_2293_, v___x_2294_);
if (v___x_2295_ == 0)
{
v___y_2283_ = v_mctx_2274_;
goto v___jp_2282_;
}
else
{
uint8_t v___x_2296_; 
v___x_2296_ = lean_nat_dec_le(v___x_2294_, v___x_2294_);
if (v___x_2296_ == 0)
{
if (v___x_2295_ == 0)
{
v___y_2283_ = v_mctx_2274_;
goto v___jp_2282_;
}
else
{
size_t v___x_2297_; size_t v___x_2298_; lean_object* v___x_2299_; 
v___x_2297_ = ((size_t)0ULL);
v___x_2298_ = lean_usize_of_nat(v___x_2294_);
v___x_2299_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Script_LazyStep_erasePostStateAssignments_spec__0(v_gs_2260_, v___x_2297_, v___x_2298_, v_mctx_2274_);
v___y_2283_ = v___x_2299_;
goto v___jp_2282_;
}
}
else
{
size_t v___x_2300_; size_t v___x_2301_; lean_object* v___x_2302_; 
v___x_2300_ = ((size_t)0ULL);
v___x_2301_ = lean_usize_of_nat(v___x_2294_);
v___x_2302_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Script_LazyStep_erasePostStateAssignments_spec__0(v_gs_2260_, v___x_2300_, v___x_2301_, v_mctx_2274_);
v___y_2283_ = v___x_2302_;
goto v___jp_2282_;
}
}
v___jp_2282_:
{
lean_object* v___x_2285_; 
if (v_isShared_2281_ == 0)
{
lean_ctor_set(v___x_2280_, 0, v___y_2283_);
v___x_2285_ = v___x_2280_;
goto v_reusejp_2284_;
}
else
{
lean_object* v_reuseFailAlloc_2292_; 
v_reuseFailAlloc_2292_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_2292_, 0, v___y_2283_);
lean_ctor_set(v_reuseFailAlloc_2292_, 1, v_cache_2275_);
lean_ctor_set(v_reuseFailAlloc_2292_, 2, v_zetaDeltaFVarIds_2276_);
lean_ctor_set(v_reuseFailAlloc_2292_, 3, v_postponed_2277_);
lean_ctor_set(v_reuseFailAlloc_2292_, 4, v_diag_2278_);
v___x_2285_ = v_reuseFailAlloc_2292_;
goto v_reusejp_2284_;
}
v_reusejp_2284_:
{
lean_object* v___x_2287_; 
if (v_isShared_2273_ == 0)
{
lean_ctor_set(v___x_2272_, 1, v___x_2285_);
v___x_2287_ = v___x_2272_;
goto v_reusejp_2286_;
}
else
{
lean_object* v_reuseFailAlloc_2291_; 
v_reuseFailAlloc_2291_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2291_, 0, v_core_2270_);
lean_ctor_set(v_reuseFailAlloc_2291_, 1, v___x_2285_);
v___x_2287_ = v_reuseFailAlloc_2291_;
goto v_reusejp_2286_;
}
v_reusejp_2286_:
{
lean_object* v___x_2289_; 
if (v_isShared_2269_ == 0)
{
lean_ctor_set(v___x_2268_, 3, v___x_2287_);
v___x_2289_ = v___x_2268_;
goto v_reusejp_2288_;
}
else
{
lean_object* v_reuseFailAlloc_2290_; 
v_reuseFailAlloc_2290_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_2290_, 0, v_preState_2263_);
lean_ctor_set(v_reuseFailAlloc_2290_, 1, v_preGoal_2264_);
lean_ctor_set(v_reuseFailAlloc_2290_, 2, v_tacticBuilders_2265_);
lean_ctor_set(v_reuseFailAlloc_2290_, 3, v___x_2287_);
lean_ctor_set(v_reuseFailAlloc_2290_, 4, v_postGoals_2266_);
v___x_2289_ = v_reuseFailAlloc_2290_;
goto v_reusejp_2288_;
}
v_reusejp_2288_:
{
return v___x_2289_;
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_LazyStep_erasePostStateAssignments___boxed(lean_object* v_s_2308_, lean_object* v_gs_2309_){
_start:
{
lean_object* v_res_2310_; 
v_res_2310_ = lp_aesop_Aesop_Script_LazyStep_erasePostStateAssignments(v_s_2308_, v_gs_2309_);
lean_dec_ref(v_gs_2309_);
return v_res_2310_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Script_Tactic(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Script_TacticState(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Tracing(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Tactic_PermuteGoals(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Lean_Meta_SavedState(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Script_Util(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Util_EqualUpToIds(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_Script_Step(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Script_Tactic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Script_TacticState(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Tracing(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Tactic_PermuteGoals(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Lean_Meta_SavedState(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Script_Util(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Util_EqualUpToIds(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_Script_Step(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam = _init_lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam();
lean_mark_persistent(lp_aesop_Aesop_Script_LazyStep_tacticBuilders__ne___autoParam);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Script_Tactic(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Script_TacticState(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Tracing(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Tactic_PermuteGoals(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Lean_Meta_SavedState(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Script_Util(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Util_EqualUpToIds(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_Script_Step(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Script_Tactic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Script_TacticState(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Tracing(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Tactic_PermuteGoals(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Lean_Meta_SavedState(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Script_Util(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Util_EqualUpToIds(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Script_Step(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_Script_Step(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_Script_Step(builtin);
}
#ifdef __cplusplus
}
#endif
