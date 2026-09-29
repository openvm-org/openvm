// Lean compiler output
// Module: Aesop.Forward.State.ApplyGoalDiff
// Imports: public import Init public meta import Init public import Aesop.Forward.State public import Aesop.RuleSet
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
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_PersistentArray_toArray___redArg(lean_object*);
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* l_Lean_PersistentArray_push___redArg(lean_object*, lean_object*);
extern lean_object* l_Lean_trace_profiler;
double lean_float_of_nat(lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_PersistentArray_append___redArg(lean_object*, lean_object*);
double lean_float_sub(double, double);
uint8_t lean_float_decLt(double, double);
extern lean_object* l_Lean_trace_profiler_useHeartbeats;
extern lean_object* l_Lean_trace_profiler_threshold;
double lean_float_div(double, double);
lean_object* l_Lean_FVarId_getType___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_LocalRuleSet_applicableForwardRulesWith(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_FVarId_getDecl___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_LocalRuleSet_forwardRulePatternSubstsInLocalDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_ForwardState_enqueueHypWithPatSubsts(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_fvar___override(lean_object*);
lean_object* lp_aesop_Aesop_ForwardState_eraseHyp(lean_object*, lean_object*);
extern lean_object* lp_aesop_Aesop_TraceOption_forward;
lean_object* l_Lean_MessageData_ofExpr(lean_object*);
lean_object* l_Lean_MessageData_ofName(lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Name_append(lean_object*, lean_object*);
uint8_t l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(lean_object*, lean_object*, lean_object*);
lean_object* lean_io_mono_nanos_now();
lean_object* lean_io_get_num_heartbeats();
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
extern lean_object* lp_aesop_Aesop_aesop_dev_statefulForward;
lean_object* l_Lean_MVarId_getType(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_LocalRuleSet_forwardRulePatternSubstsInExpr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_ForwardState_enqueueTargetPatSubsts(lean_object*, lean_object*);
extern lean_object* lp_aesop_Aesop_aesop_collectStats;
extern lean_object* lp_aesop_Aesop_TraceOption_stats;
extern lean_object* lp_aesop_Aesop_aesop_stats_file;
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
static lean_once_cell_t lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__0___redArg___closed__0;
static lean_once_cell_t lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__0___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__0___redArg___closed__1;
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__0___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_Option_get___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__2_spec__2_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__2_spec__2_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__2_spec__2_spec__3(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__2_spec__2_spec__3___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__2_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__2_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__2_spec__5(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__2_spec__5___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__2_spec__3___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__2_spec__3___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__2_spec__4(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__2_spec__4___boxed(lean_object*);
static lean_once_cell_t lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__2___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__2___closed__0;
static const lean_string_object lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__2___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 54, .m_capacity = 54, .m_length = 53, .m_data = "<exception thrown while producing trace node message>"};
static const lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__2___closed__1 = (const lean_object*)&lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__2___closed__1_value;
static lean_once_cell_t lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__2___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__2___closed__2;
static lean_once_cell_t lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__2___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__2___closed__3;
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__2(lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop___private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "erase hyp "};
static const lean_object* lp_aesop___private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp___closed__0 = (const lean_object*)&lp_aesop___private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp___closed__0_value;
static lean_once_cell_t lp_aesop___private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp___closed__1;
static const lean_string_object lp_aesop___private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = " ("};
static const lean_object* lp_aesop___private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp___closed__2 = (const lean_object*)&lp_aesop___private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp___closed__2_value;
static lean_once_cell_t lp_aesop___private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp___closed__3;
static const lean_string_object lp_aesop___private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ")"};
static const lean_object* lp_aesop___private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp___closed__4 = (const lean_object*)&lp_aesop___private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp___closed__4_value;
static lean_once_cell_t lp_aesop___private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp___closed__5;
static const lean_string_object lp_aesop___private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_aesop___private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp___closed__6 = (const lean_object*)&lp_aesop___private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp___closed__6_value;
static const lean_string_object lp_aesop___private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_aesop___private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp___closed__7 = (const lean_object*)&lp_aesop___private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp___closed__7_value;
static const lean_ctor_object lp_aesop___private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp___closed__7_value),LEAN_SCALAR_PTR_LITERAL(212, 145, 141, 177, 67, 149, 127, 197)}};
static const lean_object* lp_aesop___private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp___closed__8 = (const lean_object*)&lp_aesop___private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp___closed__8_value;
static lean_once_cell_t lp_aesop___private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_aesop___private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp___closed__9;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__2_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__2_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__2_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__2_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop___private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_addHyp___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_addHyp___lam__0___boxed(lean_object*);
static const lean_closure_object lp_aesop___private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_addHyp___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop___private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_addHyp___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop___private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_addHyp___closed__0 = (const lean_object*)&lp_aesop___private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_addHyp___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_addHyp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_addHyp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_updateTarget(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_updateTarget___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_ForwardState_applyGoalDiff_spec__3___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_ForwardState_applyGoalDiff_spec__3___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_ForwardState_applyGoalDiff_spec__3___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_ForwardState_applyGoalDiff_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_ForwardState_applyGoalDiff_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_ForwardState_applyGoalDiff_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00Aesop_ForwardState_applyGoalDiff_spec__6(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00Aesop_ForwardState_applyGoalDiff_spec__6___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00Aesop_ForwardState_applyGoalDiff_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00Aesop_ForwardState_applyGoalDiff_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardState_applyGoalDiff_spec__2(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardState_applyGoalDiff_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardState_applyGoalDiff___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardState_applyGoalDiff___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00Aesop_ForwardState_applyGoalDiff_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00Aesop_ForwardState_applyGoalDiff_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardState_applyGoalDiff_spec__4(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardState_applyGoalDiff_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardState_applyGoalDiff___lam__1(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardState_applyGoalDiff___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_ForwardState_applyGoalDiff_spec__5___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_ForwardState_applyGoalDiff_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardState_applyGoalDiff(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardState_applyGoalDiff___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_ForwardState_applyGoalDiff_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_ForwardState_applyGoalDiff_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_1_; lean_object* v___x_2_; lean_object* v___x_3_; 
v___x_1_ = lean_unsigned_to_nat(32u);
v___x_2_ = lean_mk_empty_array_with_capacity(v___x_1_);
v___x_3_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3_, 0, v___x_2_);
return v___x_3_;
}
}
static lean_object* _init_lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__0___redArg___closed__1(void){
_start:
{
size_t v___x_4_; lean_object* v___x_5_; lean_object* v___x_6_; lean_object* v___x_7_; lean_object* v___x_8_; lean_object* v___x_9_; 
v___x_4_ = ((size_t)5ULL);
v___x_5_ = lean_unsigned_to_nat(0u);
v___x_6_ = lean_unsigned_to_nat(32u);
v___x_7_ = lean_mk_empty_array_with_capacity(v___x_6_);
v___x_8_ = lean_obj_once(&lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__0___redArg___closed__0, &lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__0___redArg___closed__0_once, _init_lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__0___redArg___closed__0);
v___x_9_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_9_, 0, v___x_8_);
lean_ctor_set(v___x_9_, 1, v___x_7_);
lean_ctor_set(v___x_9_, 2, v___x_5_);
lean_ctor_set(v___x_9_, 3, v___x_5_);
lean_ctor_set_usize(v___x_9_, 4, v___x_4_);
return v___x_9_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__0___redArg(lean_object* v___y_10_){
_start:
{
lean_object* v___x_12_; lean_object* v_traceState_13_; lean_object* v_traces_14_; lean_object* v___x_15_; lean_object* v_traceState_16_; lean_object* v_env_17_; lean_object* v_nextMacroScope_18_; lean_object* v_ngen_19_; lean_object* v_auxDeclNGen_20_; lean_object* v_cache_21_; lean_object* v_messages_22_; lean_object* v_infoState_23_; lean_object* v_snapshotTasks_24_; lean_object* v___x_26_; uint8_t v_isShared_27_; uint8_t v_isSharedCheck_43_; 
v___x_12_ = lean_st_ref_get(v___y_10_);
v_traceState_13_ = lean_ctor_get(v___x_12_, 4);
lean_inc_ref(v_traceState_13_);
lean_dec(v___x_12_);
v_traces_14_ = lean_ctor_get(v_traceState_13_, 0);
lean_inc_ref(v_traces_14_);
lean_dec_ref(v_traceState_13_);
v___x_15_ = lean_st_ref_take(v___y_10_);
v_traceState_16_ = lean_ctor_get(v___x_15_, 4);
v_env_17_ = lean_ctor_get(v___x_15_, 0);
v_nextMacroScope_18_ = lean_ctor_get(v___x_15_, 1);
v_ngen_19_ = lean_ctor_get(v___x_15_, 2);
v_auxDeclNGen_20_ = lean_ctor_get(v___x_15_, 3);
v_cache_21_ = lean_ctor_get(v___x_15_, 5);
v_messages_22_ = lean_ctor_get(v___x_15_, 6);
v_infoState_23_ = lean_ctor_get(v___x_15_, 7);
v_snapshotTasks_24_ = lean_ctor_get(v___x_15_, 8);
v_isSharedCheck_43_ = !lean_is_exclusive(v___x_15_);
if (v_isSharedCheck_43_ == 0)
{
v___x_26_ = v___x_15_;
v_isShared_27_ = v_isSharedCheck_43_;
goto v_resetjp_25_;
}
else
{
lean_inc(v_snapshotTasks_24_);
lean_inc(v_infoState_23_);
lean_inc(v_messages_22_);
lean_inc(v_cache_21_);
lean_inc(v_traceState_16_);
lean_inc(v_auxDeclNGen_20_);
lean_inc(v_ngen_19_);
lean_inc(v_nextMacroScope_18_);
lean_inc(v_env_17_);
lean_dec(v___x_15_);
v___x_26_ = lean_box(0);
v_isShared_27_ = v_isSharedCheck_43_;
goto v_resetjp_25_;
}
v_resetjp_25_:
{
uint64_t v_tid_28_; lean_object* v___x_30_; uint8_t v_isShared_31_; uint8_t v_isSharedCheck_41_; 
v_tid_28_ = lean_ctor_get_uint64(v_traceState_16_, sizeof(void*)*1);
v_isSharedCheck_41_ = !lean_is_exclusive(v_traceState_16_);
if (v_isSharedCheck_41_ == 0)
{
lean_object* v_unused_42_; 
v_unused_42_ = lean_ctor_get(v_traceState_16_, 0);
lean_dec(v_unused_42_);
v___x_30_ = v_traceState_16_;
v_isShared_31_ = v_isSharedCheck_41_;
goto v_resetjp_29_;
}
else
{
lean_dec(v_traceState_16_);
v___x_30_ = lean_box(0);
v_isShared_31_ = v_isSharedCheck_41_;
goto v_resetjp_29_;
}
v_resetjp_29_:
{
lean_object* v___x_32_; lean_object* v___x_34_; 
v___x_32_ = lean_obj_once(&lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__0___redArg___closed__1, &lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__0___redArg___closed__1_once, _init_lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__0___redArg___closed__1);
if (v_isShared_31_ == 0)
{
lean_ctor_set(v___x_30_, 0, v___x_32_);
v___x_34_ = v___x_30_;
goto v_reusejp_33_;
}
else
{
lean_object* v_reuseFailAlloc_40_; 
v_reuseFailAlloc_40_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_40_, 0, v___x_32_);
lean_ctor_set_uint64(v_reuseFailAlloc_40_, sizeof(void*)*1, v_tid_28_);
v___x_34_ = v_reuseFailAlloc_40_;
goto v_reusejp_33_;
}
v_reusejp_33_:
{
lean_object* v___x_36_; 
if (v_isShared_27_ == 0)
{
lean_ctor_set(v___x_26_, 4, v___x_34_);
v___x_36_ = v___x_26_;
goto v_reusejp_35_;
}
else
{
lean_object* v_reuseFailAlloc_39_; 
v_reuseFailAlloc_39_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_39_, 0, v_env_17_);
lean_ctor_set(v_reuseFailAlloc_39_, 1, v_nextMacroScope_18_);
lean_ctor_set(v_reuseFailAlloc_39_, 2, v_ngen_19_);
lean_ctor_set(v_reuseFailAlloc_39_, 3, v_auxDeclNGen_20_);
lean_ctor_set(v_reuseFailAlloc_39_, 4, v___x_34_);
lean_ctor_set(v_reuseFailAlloc_39_, 5, v_cache_21_);
lean_ctor_set(v_reuseFailAlloc_39_, 6, v_messages_22_);
lean_ctor_set(v_reuseFailAlloc_39_, 7, v_infoState_23_);
lean_ctor_set(v_reuseFailAlloc_39_, 8, v_snapshotTasks_24_);
v___x_36_ = v_reuseFailAlloc_39_;
goto v_reusejp_35_;
}
v_reusejp_35_:
{
lean_object* v___x_37_; lean_object* v___x_38_; 
v___x_37_ = lean_st_ref_set(v___y_10_, v___x_36_);
v___x_38_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_38_, 0, v_traces_14_);
return v___x_38_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__0___redArg___boxed(lean_object* v___y_44_, lean_object* v___y_45_){
_start:
{
lean_object* v_res_46_; 
v_res_46_ = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__0___redArg(v___y_44_);
lean_dec(v___y_44_);
return v_res_46_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__0(lean_object* v___y_47_, lean_object* v___y_48_, lean_object* v___y_49_, lean_object* v___y_50_, lean_object* v___y_51_){
_start:
{
lean_object* v___x_53_; 
v___x_53_ = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__0___redArg(v___y_51_);
return v___x_53_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__0___boxed(lean_object* v___y_54_, lean_object* v___y_55_, lean_object* v___y_56_, lean_object* v___y_57_, lean_object* v___y_58_, lean_object* v___y_59_){
_start:
{
lean_object* v_res_60_; 
v_res_60_ = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__0(v___y_54_, v___y_55_, v___y_56_, v___y_57_, v___y_58_);
lean_dec(v___y_58_);
lean_dec_ref(v___y_57_);
lean_dec(v___y_56_);
lean_dec_ref(v___y_55_);
lean_dec(v___y_54_);
return v_res_60_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_Option_get___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__1(lean_object* v_opts_61_, lean_object* v_opt_62_){
_start:
{
lean_object* v_name_63_; lean_object* v_defValue_64_; lean_object* v_map_65_; lean_object* v___x_66_; 
v_name_63_ = lean_ctor_get(v_opt_62_, 0);
v_defValue_64_ = lean_ctor_get(v_opt_62_, 1);
v_map_65_ = lean_ctor_get(v_opts_61_, 0);
v___x_66_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_65_, v_name_63_);
if (lean_obj_tag(v___x_66_) == 0)
{
uint8_t v___x_67_; 
v___x_67_ = lean_unbox(v_defValue_64_);
return v___x_67_;
}
else
{
lean_object* v_val_68_; 
v_val_68_ = lean_ctor_get(v___x_66_, 0);
lean_inc(v_val_68_);
lean_dec_ref_known(v___x_66_, 1);
if (lean_obj_tag(v_val_68_) == 1)
{
uint8_t v_v_69_; 
v_v_69_ = lean_ctor_get_uint8(v_val_68_, 0);
lean_dec_ref_known(v_val_68_, 0);
return v_v_69_;
}
else
{
uint8_t v___x_70_; 
lean_dec(v_val_68_);
v___x_70_ = lean_unbox(v_defValue_64_);
return v___x_70_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__1___boxed(lean_object* v_opts_71_, lean_object* v_opt_72_){
_start:
{
uint8_t v_res_73_; lean_object* v_r_74_; 
v_res_73_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__1(v_opts_71_, v_opt_72_);
lean_dec_ref(v_opt_72_);
lean_dec_ref(v_opts_71_);
v_r_74_ = lean_box(v_res_73_);
return v_r_74_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp___lam__0(lean_object* v___x_75_, lean_object* v_x_76_, lean_object* v___y_77_, lean_object* v___y_78_, lean_object* v___y_79_, lean_object* v___y_80_, lean_object* v___y_81_){
_start:
{
lean_object* v___x_83_; 
v___x_83_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_83_, 0, v___x_75_);
return v___x_83_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp___lam__0___boxed(lean_object* v___x_84_, lean_object* v_x_85_, lean_object* v___y_86_, lean_object* v___y_87_, lean_object* v___y_88_, lean_object* v___y_89_, lean_object* v___y_90_, lean_object* v___y_91_){
_start:
{
lean_object* v_res_92_; 
v_res_92_ = lp_aesop___private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp___lam__0(v___x_84_, v_x_85_, v___y_86_, v___y_87_, v___y_88_, v___y_89_, v___y_90_);
lean_dec(v___y_90_);
lean_dec_ref(v___y_89_);
lean_dec(v___y_88_);
lean_dec_ref(v___y_87_);
lean_dec(v___y_86_);
lean_dec_ref(v_x_85_);
return v_res_92_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__2_spec__2_spec__4(lean_object* v_msgData_93_, lean_object* v___y_94_, lean_object* v___y_95_, lean_object* v___y_96_, lean_object* v___y_97_){
_start:
{
lean_object* v___x_99_; lean_object* v_env_100_; lean_object* v___x_101_; lean_object* v_mctx_102_; lean_object* v_lctx_103_; lean_object* v_options_104_; lean_object* v___x_105_; lean_object* v___x_106_; lean_object* v___x_107_; 
v___x_99_ = lean_st_ref_get(v___y_97_);
v_env_100_ = lean_ctor_get(v___x_99_, 0);
lean_inc_ref(v_env_100_);
lean_dec(v___x_99_);
v___x_101_ = lean_st_ref_get(v___y_95_);
v_mctx_102_ = lean_ctor_get(v___x_101_, 0);
lean_inc_ref(v_mctx_102_);
lean_dec(v___x_101_);
v_lctx_103_ = lean_ctor_get(v___y_94_, 2);
v_options_104_ = lean_ctor_get(v___y_96_, 2);
lean_inc_ref(v_options_104_);
lean_inc_ref(v_lctx_103_);
v___x_105_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_105_, 0, v_env_100_);
lean_ctor_set(v___x_105_, 1, v_mctx_102_);
lean_ctor_set(v___x_105_, 2, v_lctx_103_);
lean_ctor_set(v___x_105_, 3, v_options_104_);
v___x_106_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_106_, 0, v___x_105_);
lean_ctor_set(v___x_106_, 1, v_msgData_93_);
v___x_107_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_107_, 0, v___x_106_);
return v___x_107_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__2_spec__2_spec__4___boxed(lean_object* v_msgData_108_, lean_object* v___y_109_, lean_object* v___y_110_, lean_object* v___y_111_, lean_object* v___y_112_, lean_object* v___y_113_){
_start:
{
lean_object* v_res_114_; 
v_res_114_ = lp_aesop_Lean_addMessageContextFull___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__2_spec__2_spec__4(v_msgData_108_, v___y_109_, v___y_110_, v___y_111_, v___y_112_);
lean_dec(v___y_112_);
lean_dec_ref(v___y_111_);
lean_dec(v___y_110_);
lean_dec_ref(v___y_109_);
return v_res_114_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__2_spec__2_spec__3(size_t v_sz_115_, size_t v_i_116_, lean_object* v_bs_117_){
_start:
{
uint8_t v___x_118_; 
v___x_118_ = lean_usize_dec_lt(v_i_116_, v_sz_115_);
if (v___x_118_ == 0)
{
return v_bs_117_;
}
else
{
lean_object* v_v_119_; lean_object* v_msg_120_; lean_object* v___x_121_; lean_object* v_bs_x27_122_; size_t v___x_123_; size_t v___x_124_; lean_object* v___x_125_; 
v_v_119_ = lean_array_uget_borrowed(v_bs_117_, v_i_116_);
v_msg_120_ = lean_ctor_get(v_v_119_, 1);
lean_inc_ref(v_msg_120_);
v___x_121_ = lean_unsigned_to_nat(0u);
v_bs_x27_122_ = lean_array_uset(v_bs_117_, v_i_116_, v___x_121_);
v___x_123_ = ((size_t)1ULL);
v___x_124_ = lean_usize_add(v_i_116_, v___x_123_);
v___x_125_ = lean_array_uset(v_bs_x27_122_, v_i_116_, v_msg_120_);
v_i_116_ = v___x_124_;
v_bs_117_ = v___x_125_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__2_spec__2_spec__3___boxed(lean_object* v_sz_127_, lean_object* v_i_128_, lean_object* v_bs_129_){
_start:
{
size_t v_sz_boxed_130_; size_t v_i_boxed_131_; lean_object* v_res_132_; 
v_sz_boxed_130_ = lean_unbox_usize(v_sz_127_);
lean_dec(v_sz_127_);
v_i_boxed_131_ = lean_unbox_usize(v_i_128_);
lean_dec(v_i_128_);
v_res_132_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__2_spec__2_spec__3(v_sz_boxed_130_, v_i_boxed_131_, v_bs_129_);
return v_res_132_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__2_spec__2___redArg(lean_object* v_oldTraces_133_, lean_object* v_data_134_, lean_object* v_ref_135_, lean_object* v_msg_136_, lean_object* v___y_137_, lean_object* v___y_138_, lean_object* v___y_139_, lean_object* v___y_140_){
_start:
{
lean_object* v_fileName_142_; lean_object* v_fileMap_143_; lean_object* v_options_144_; lean_object* v_currRecDepth_145_; lean_object* v_maxRecDepth_146_; lean_object* v_ref_147_; lean_object* v_currNamespace_148_; lean_object* v_openDecls_149_; lean_object* v_initHeartbeats_150_; lean_object* v_maxHeartbeats_151_; lean_object* v_quotContext_152_; lean_object* v_currMacroScope_153_; uint8_t v_diag_154_; lean_object* v_cancelTk_x3f_155_; uint8_t v_suppressElabErrors_156_; lean_object* v_inheritedTraceOptions_157_; lean_object* v___x_158_; lean_object* v_traceState_159_; lean_object* v_traces_160_; lean_object* v_ref_161_; lean_object* v___x_162_; lean_object* v___x_163_; size_t v_sz_164_; size_t v___x_165_; lean_object* v___x_166_; lean_object* v_msg_167_; lean_object* v___x_168_; lean_object* v_a_169_; lean_object* v___x_171_; uint8_t v_isShared_172_; uint8_t v_isSharedCheck_206_; 
v_fileName_142_ = lean_ctor_get(v___y_139_, 0);
v_fileMap_143_ = lean_ctor_get(v___y_139_, 1);
v_options_144_ = lean_ctor_get(v___y_139_, 2);
v_currRecDepth_145_ = lean_ctor_get(v___y_139_, 3);
v_maxRecDepth_146_ = lean_ctor_get(v___y_139_, 4);
v_ref_147_ = lean_ctor_get(v___y_139_, 5);
v_currNamespace_148_ = lean_ctor_get(v___y_139_, 6);
v_openDecls_149_ = lean_ctor_get(v___y_139_, 7);
v_initHeartbeats_150_ = lean_ctor_get(v___y_139_, 8);
v_maxHeartbeats_151_ = lean_ctor_get(v___y_139_, 9);
v_quotContext_152_ = lean_ctor_get(v___y_139_, 10);
v_currMacroScope_153_ = lean_ctor_get(v___y_139_, 11);
v_diag_154_ = lean_ctor_get_uint8(v___y_139_, sizeof(void*)*14);
v_cancelTk_x3f_155_ = lean_ctor_get(v___y_139_, 12);
v_suppressElabErrors_156_ = lean_ctor_get_uint8(v___y_139_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_157_ = lean_ctor_get(v___y_139_, 13);
v___x_158_ = lean_st_ref_get(v___y_140_);
v_traceState_159_ = lean_ctor_get(v___x_158_, 4);
lean_inc_ref(v_traceState_159_);
lean_dec(v___x_158_);
v_traces_160_ = lean_ctor_get(v_traceState_159_, 0);
lean_inc_ref(v_traces_160_);
lean_dec_ref(v_traceState_159_);
v_ref_161_ = l_Lean_replaceRef(v_ref_135_, v_ref_147_);
lean_inc_ref(v_inheritedTraceOptions_157_);
lean_inc(v_cancelTk_x3f_155_);
lean_inc(v_currMacroScope_153_);
lean_inc(v_quotContext_152_);
lean_inc(v_maxHeartbeats_151_);
lean_inc(v_initHeartbeats_150_);
lean_inc(v_openDecls_149_);
lean_inc(v_currNamespace_148_);
lean_inc(v_maxRecDepth_146_);
lean_inc(v_currRecDepth_145_);
lean_inc_ref(v_options_144_);
lean_inc_ref(v_fileMap_143_);
lean_inc_ref(v_fileName_142_);
v___x_162_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_162_, 0, v_fileName_142_);
lean_ctor_set(v___x_162_, 1, v_fileMap_143_);
lean_ctor_set(v___x_162_, 2, v_options_144_);
lean_ctor_set(v___x_162_, 3, v_currRecDepth_145_);
lean_ctor_set(v___x_162_, 4, v_maxRecDepth_146_);
lean_ctor_set(v___x_162_, 5, v_ref_161_);
lean_ctor_set(v___x_162_, 6, v_currNamespace_148_);
lean_ctor_set(v___x_162_, 7, v_openDecls_149_);
lean_ctor_set(v___x_162_, 8, v_initHeartbeats_150_);
lean_ctor_set(v___x_162_, 9, v_maxHeartbeats_151_);
lean_ctor_set(v___x_162_, 10, v_quotContext_152_);
lean_ctor_set(v___x_162_, 11, v_currMacroScope_153_);
lean_ctor_set(v___x_162_, 12, v_cancelTk_x3f_155_);
lean_ctor_set(v___x_162_, 13, v_inheritedTraceOptions_157_);
lean_ctor_set_uint8(v___x_162_, sizeof(void*)*14, v_diag_154_);
lean_ctor_set_uint8(v___x_162_, sizeof(void*)*14 + 1, v_suppressElabErrors_156_);
v___x_163_ = l_Lean_PersistentArray_toArray___redArg(v_traces_160_);
lean_dec_ref(v_traces_160_);
v_sz_164_ = lean_array_size(v___x_163_);
v___x_165_ = ((size_t)0ULL);
v___x_166_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__2_spec__2_spec__3(v_sz_164_, v___x_165_, v___x_163_);
v_msg_167_ = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(v_msg_167_, 0, v_data_134_);
lean_ctor_set(v_msg_167_, 1, v_msg_136_);
lean_ctor_set(v_msg_167_, 2, v___x_166_);
v___x_168_ = lp_aesop_Lean_addMessageContextFull___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__2_spec__2_spec__4(v_msg_167_, v___y_137_, v___y_138_, v___x_162_, v___y_140_);
lean_dec_ref_known(v___x_162_, 14);
v_a_169_ = lean_ctor_get(v___x_168_, 0);
v_isSharedCheck_206_ = !lean_is_exclusive(v___x_168_);
if (v_isSharedCheck_206_ == 0)
{
v___x_171_ = v___x_168_;
v_isShared_172_ = v_isSharedCheck_206_;
goto v_resetjp_170_;
}
else
{
lean_inc(v_a_169_);
lean_dec(v___x_168_);
v___x_171_ = lean_box(0);
v_isShared_172_ = v_isSharedCheck_206_;
goto v_resetjp_170_;
}
v_resetjp_170_:
{
lean_object* v___x_173_; lean_object* v_traceState_174_; lean_object* v_env_175_; lean_object* v_nextMacroScope_176_; lean_object* v_ngen_177_; lean_object* v_auxDeclNGen_178_; lean_object* v_cache_179_; lean_object* v_messages_180_; lean_object* v_infoState_181_; lean_object* v_snapshotTasks_182_; lean_object* v___x_184_; uint8_t v_isShared_185_; uint8_t v_isSharedCheck_205_; 
v___x_173_ = lean_st_ref_take(v___y_140_);
v_traceState_174_ = lean_ctor_get(v___x_173_, 4);
v_env_175_ = lean_ctor_get(v___x_173_, 0);
v_nextMacroScope_176_ = lean_ctor_get(v___x_173_, 1);
v_ngen_177_ = lean_ctor_get(v___x_173_, 2);
v_auxDeclNGen_178_ = lean_ctor_get(v___x_173_, 3);
v_cache_179_ = lean_ctor_get(v___x_173_, 5);
v_messages_180_ = lean_ctor_get(v___x_173_, 6);
v_infoState_181_ = lean_ctor_get(v___x_173_, 7);
v_snapshotTasks_182_ = lean_ctor_get(v___x_173_, 8);
v_isSharedCheck_205_ = !lean_is_exclusive(v___x_173_);
if (v_isSharedCheck_205_ == 0)
{
v___x_184_ = v___x_173_;
v_isShared_185_ = v_isSharedCheck_205_;
goto v_resetjp_183_;
}
else
{
lean_inc(v_snapshotTasks_182_);
lean_inc(v_infoState_181_);
lean_inc(v_messages_180_);
lean_inc(v_cache_179_);
lean_inc(v_traceState_174_);
lean_inc(v_auxDeclNGen_178_);
lean_inc(v_ngen_177_);
lean_inc(v_nextMacroScope_176_);
lean_inc(v_env_175_);
lean_dec(v___x_173_);
v___x_184_ = lean_box(0);
v_isShared_185_ = v_isSharedCheck_205_;
goto v_resetjp_183_;
}
v_resetjp_183_:
{
uint64_t v_tid_186_; lean_object* v___x_188_; uint8_t v_isShared_189_; uint8_t v_isSharedCheck_203_; 
v_tid_186_ = lean_ctor_get_uint64(v_traceState_174_, sizeof(void*)*1);
v_isSharedCheck_203_ = !lean_is_exclusive(v_traceState_174_);
if (v_isSharedCheck_203_ == 0)
{
lean_object* v_unused_204_; 
v_unused_204_ = lean_ctor_get(v_traceState_174_, 0);
lean_dec(v_unused_204_);
v___x_188_ = v_traceState_174_;
v_isShared_189_ = v_isSharedCheck_203_;
goto v_resetjp_187_;
}
else
{
lean_dec(v_traceState_174_);
v___x_188_ = lean_box(0);
v_isShared_189_ = v_isSharedCheck_203_;
goto v_resetjp_187_;
}
v_resetjp_187_:
{
lean_object* v___x_190_; lean_object* v___x_191_; lean_object* v___x_193_; 
v___x_190_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_190_, 0, v_ref_135_);
lean_ctor_set(v___x_190_, 1, v_a_169_);
v___x_191_ = l_Lean_PersistentArray_push___redArg(v_oldTraces_133_, v___x_190_);
if (v_isShared_189_ == 0)
{
lean_ctor_set(v___x_188_, 0, v___x_191_);
v___x_193_ = v___x_188_;
goto v_reusejp_192_;
}
else
{
lean_object* v_reuseFailAlloc_202_; 
v_reuseFailAlloc_202_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_202_, 0, v___x_191_);
lean_ctor_set_uint64(v_reuseFailAlloc_202_, sizeof(void*)*1, v_tid_186_);
v___x_193_ = v_reuseFailAlloc_202_;
goto v_reusejp_192_;
}
v_reusejp_192_:
{
lean_object* v___x_195_; 
if (v_isShared_185_ == 0)
{
lean_ctor_set(v___x_184_, 4, v___x_193_);
v___x_195_ = v___x_184_;
goto v_reusejp_194_;
}
else
{
lean_object* v_reuseFailAlloc_201_; 
v_reuseFailAlloc_201_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_201_, 0, v_env_175_);
lean_ctor_set(v_reuseFailAlloc_201_, 1, v_nextMacroScope_176_);
lean_ctor_set(v_reuseFailAlloc_201_, 2, v_ngen_177_);
lean_ctor_set(v_reuseFailAlloc_201_, 3, v_auxDeclNGen_178_);
lean_ctor_set(v_reuseFailAlloc_201_, 4, v___x_193_);
lean_ctor_set(v_reuseFailAlloc_201_, 5, v_cache_179_);
lean_ctor_set(v_reuseFailAlloc_201_, 6, v_messages_180_);
lean_ctor_set(v_reuseFailAlloc_201_, 7, v_infoState_181_);
lean_ctor_set(v_reuseFailAlloc_201_, 8, v_snapshotTasks_182_);
v___x_195_ = v_reuseFailAlloc_201_;
goto v_reusejp_194_;
}
v_reusejp_194_:
{
lean_object* v___x_196_; lean_object* v___x_197_; lean_object* v___x_199_; 
v___x_196_ = lean_st_ref_set(v___y_140_, v___x_195_);
v___x_197_ = lean_box(0);
if (v_isShared_172_ == 0)
{
lean_ctor_set(v___x_171_, 0, v___x_197_);
v___x_199_ = v___x_171_;
goto v_reusejp_198_;
}
else
{
lean_object* v_reuseFailAlloc_200_; 
v_reuseFailAlloc_200_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_200_, 0, v___x_197_);
v___x_199_ = v_reuseFailAlloc_200_;
goto v_reusejp_198_;
}
v_reusejp_198_:
{
return v___x_199_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__2_spec__2___redArg___boxed(lean_object* v_oldTraces_207_, lean_object* v_data_208_, lean_object* v_ref_209_, lean_object* v_msg_210_, lean_object* v___y_211_, lean_object* v___y_212_, lean_object* v___y_213_, lean_object* v___y_214_, lean_object* v___y_215_){
_start:
{
lean_object* v_res_216_; 
v_res_216_ = lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__2_spec__2___redArg(v_oldTraces_207_, v_data_208_, v_ref_209_, v_msg_210_, v___y_211_, v___y_212_, v___y_213_, v___y_214_);
lean_dec(v___y_214_);
lean_dec_ref(v___y_213_);
lean_dec(v___y_212_);
lean_dec_ref(v___y_211_);
return v_res_216_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__2_spec__5(lean_object* v_opts_217_, lean_object* v_opt_218_){
_start:
{
lean_object* v_name_219_; lean_object* v_defValue_220_; lean_object* v_map_221_; lean_object* v___x_222_; 
v_name_219_ = lean_ctor_get(v_opt_218_, 0);
v_defValue_220_ = lean_ctor_get(v_opt_218_, 1);
v_map_221_ = lean_ctor_get(v_opts_217_, 0);
v___x_222_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_221_, v_name_219_);
if (lean_obj_tag(v___x_222_) == 0)
{
lean_inc(v_defValue_220_);
return v_defValue_220_;
}
else
{
lean_object* v_val_223_; 
v_val_223_ = lean_ctor_get(v___x_222_, 0);
lean_inc(v_val_223_);
lean_dec_ref_known(v___x_222_, 1);
if (lean_obj_tag(v_val_223_) == 3)
{
lean_object* v_v_224_; 
v_v_224_ = lean_ctor_get(v_val_223_, 0);
lean_inc(v_v_224_);
lean_dec_ref_known(v_val_223_, 1);
return v_v_224_;
}
else
{
lean_dec(v_val_223_);
lean_inc(v_defValue_220_);
return v_defValue_220_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__2_spec__5___boxed(lean_object* v_opts_225_, lean_object* v_opt_226_){
_start:
{
lean_object* v_res_227_; 
v_res_227_ = lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__2_spec__5(v_opts_225_, v_opt_226_);
lean_dec_ref(v_opt_226_);
lean_dec_ref(v_opts_225_);
return v_res_227_;
}
}
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__2_spec__3___redArg(lean_object* v_x_228_){
_start:
{
if (lean_obj_tag(v_x_228_) == 0)
{
lean_object* v_a_230_; lean_object* v___x_232_; uint8_t v_isShared_233_; uint8_t v_isSharedCheck_237_; 
v_a_230_ = lean_ctor_get(v_x_228_, 0);
v_isSharedCheck_237_ = !lean_is_exclusive(v_x_228_);
if (v_isSharedCheck_237_ == 0)
{
v___x_232_ = v_x_228_;
v_isShared_233_ = v_isSharedCheck_237_;
goto v_resetjp_231_;
}
else
{
lean_inc(v_a_230_);
lean_dec(v_x_228_);
v___x_232_ = lean_box(0);
v_isShared_233_ = v_isSharedCheck_237_;
goto v_resetjp_231_;
}
v_resetjp_231_:
{
lean_object* v___x_235_; 
if (v_isShared_233_ == 0)
{
lean_ctor_set_tag(v___x_232_, 1);
v___x_235_ = v___x_232_;
goto v_reusejp_234_;
}
else
{
lean_object* v_reuseFailAlloc_236_; 
v_reuseFailAlloc_236_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_236_, 0, v_a_230_);
v___x_235_ = v_reuseFailAlloc_236_;
goto v_reusejp_234_;
}
v_reusejp_234_:
{
return v___x_235_;
}
}
}
else
{
lean_object* v_a_238_; lean_object* v___x_240_; uint8_t v_isShared_241_; uint8_t v_isSharedCheck_245_; 
v_a_238_ = lean_ctor_get(v_x_228_, 0);
v_isSharedCheck_245_ = !lean_is_exclusive(v_x_228_);
if (v_isSharedCheck_245_ == 0)
{
v___x_240_ = v_x_228_;
v_isShared_241_ = v_isSharedCheck_245_;
goto v_resetjp_239_;
}
else
{
lean_inc(v_a_238_);
lean_dec(v_x_228_);
v___x_240_ = lean_box(0);
v_isShared_241_ = v_isSharedCheck_245_;
goto v_resetjp_239_;
}
v_resetjp_239_:
{
lean_object* v___x_243_; 
if (v_isShared_241_ == 0)
{
lean_ctor_set_tag(v___x_240_, 0);
v___x_243_ = v___x_240_;
goto v_reusejp_242_;
}
else
{
lean_object* v_reuseFailAlloc_244_; 
v_reuseFailAlloc_244_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_244_, 0, v_a_238_);
v___x_243_ = v_reuseFailAlloc_244_;
goto v_reusejp_242_;
}
v_reusejp_242_:
{
return v___x_243_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__2_spec__3___redArg___boxed(lean_object* v_x_246_, lean_object* v___y_247_){
_start:
{
lean_object* v_res_248_; 
v_res_248_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__2_spec__3___redArg(v_x_246_);
return v_res_248_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__2_spec__4(lean_object* v_e_249_){
_start:
{
if (lean_obj_tag(v_e_249_) == 0)
{
uint8_t v___x_250_; 
v___x_250_ = 2;
return v___x_250_;
}
else
{
uint8_t v___x_251_; 
v___x_251_ = 0;
return v___x_251_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__2_spec__4___boxed(lean_object* v_e_252_){
_start:
{
uint8_t v_res_253_; lean_object* v_r_254_; 
v_res_253_ = lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__2_spec__4(v_e_252_);
lean_dec_ref(v_e_252_);
v_r_254_ = lean_box(v_res_253_);
return v_r_254_;
}
}
static double _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__2___closed__0(void){
_start:
{
lean_object* v___x_255_; double v___x_256_; 
v___x_255_ = lean_unsigned_to_nat(0u);
v___x_256_ = lean_float_of_nat(v___x_255_);
return v___x_256_;
}
}
static lean_object* _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__2___closed__2(void){
_start:
{
lean_object* v___x_258_; lean_object* v___x_259_; 
v___x_258_ = ((lean_object*)(lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__2___closed__1));
v___x_259_ = l_Lean_stringToMessageData(v___x_258_);
return v___x_259_;
}
}
static double _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__2___closed__3(void){
_start:
{
lean_object* v___x_260_; double v___x_261_; 
v___x_260_ = lean_unsigned_to_nat(1000u);
v___x_261_ = lean_float_of_nat(v___x_260_);
return v___x_261_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__2(lean_object* v_cls_262_, uint8_t v_collapsed_263_, lean_object* v_tag_264_, lean_object* v_opts_265_, uint8_t v_clsEnabled_266_, lean_object* v_oldTraces_267_, lean_object* v_msg_268_, lean_object* v_resStartStop_269_, lean_object* v___y_270_, lean_object* v___y_271_, lean_object* v___y_272_, lean_object* v___y_273_, lean_object* v___y_274_){
_start:
{
lean_object* v_fst_276_; lean_object* v_snd_277_; lean_object* v___y_279_; lean_object* v___y_280_; lean_object* v_data_281_; lean_object* v_fst_292_; lean_object* v_snd_293_; lean_object* v___x_294_; uint8_t v___x_295_; lean_object* v___y_297_; lean_object* v_a_298_; uint8_t v___y_313_; double v___y_344_; 
v_fst_276_ = lean_ctor_get(v_resStartStop_269_, 0);
lean_inc(v_fst_276_);
v_snd_277_ = lean_ctor_get(v_resStartStop_269_, 1);
lean_inc(v_snd_277_);
lean_dec_ref(v_resStartStop_269_);
v_fst_292_ = lean_ctor_get(v_snd_277_, 0);
lean_inc(v_fst_292_);
v_snd_293_ = lean_ctor_get(v_snd_277_, 1);
lean_inc(v_snd_293_);
lean_dec(v_snd_277_);
v___x_294_ = l_Lean_trace_profiler;
v___x_295_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__1(v_opts_265_, v___x_294_);
if (v___x_295_ == 0)
{
v___y_313_ = v___x_295_;
goto v___jp_312_;
}
else
{
lean_object* v___x_349_; uint8_t v___x_350_; 
v___x_349_ = l_Lean_trace_profiler_useHeartbeats;
v___x_350_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__1(v_opts_265_, v___x_349_);
if (v___x_350_ == 0)
{
lean_object* v___x_351_; lean_object* v___x_352_; double v___x_353_; double v___x_354_; double v___x_355_; 
v___x_351_ = l_Lean_trace_profiler_threshold;
v___x_352_ = lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__2_spec__5(v_opts_265_, v___x_351_);
v___x_353_ = lean_float_of_nat(v___x_352_);
v___x_354_ = lean_float_once(&lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__2___closed__3, &lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__2___closed__3_once, _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__2___closed__3);
v___x_355_ = lean_float_div(v___x_353_, v___x_354_);
v___y_344_ = v___x_355_;
goto v___jp_343_;
}
else
{
lean_object* v___x_356_; lean_object* v___x_357_; double v___x_358_; 
v___x_356_ = l_Lean_trace_profiler_threshold;
v___x_357_ = lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__2_spec__5(v_opts_265_, v___x_356_);
v___x_358_ = lean_float_of_nat(v___x_357_);
v___y_344_ = v___x_358_;
goto v___jp_343_;
}
}
v___jp_278_:
{
lean_object* v___x_282_; 
lean_inc(v___y_280_);
v___x_282_ = lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__2_spec__2___redArg(v_oldTraces_267_, v_data_281_, v___y_280_, v___y_279_, v___y_271_, v___y_272_, v___y_273_, v___y_274_);
if (lean_obj_tag(v___x_282_) == 0)
{
lean_object* v___x_283_; 
lean_dec_ref_known(v___x_282_, 1);
v___x_283_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__2_spec__3___redArg(v_fst_276_);
return v___x_283_;
}
else
{
lean_object* v_a_284_; lean_object* v___x_286_; uint8_t v_isShared_287_; uint8_t v_isSharedCheck_291_; 
lean_dec(v_fst_276_);
v_a_284_ = lean_ctor_get(v___x_282_, 0);
v_isSharedCheck_291_ = !lean_is_exclusive(v___x_282_);
if (v_isSharedCheck_291_ == 0)
{
v___x_286_ = v___x_282_;
v_isShared_287_ = v_isSharedCheck_291_;
goto v_resetjp_285_;
}
else
{
lean_inc(v_a_284_);
lean_dec(v___x_282_);
v___x_286_ = lean_box(0);
v_isShared_287_ = v_isSharedCheck_291_;
goto v_resetjp_285_;
}
v_resetjp_285_:
{
lean_object* v___x_289_; 
if (v_isShared_287_ == 0)
{
v___x_289_ = v___x_286_;
goto v_reusejp_288_;
}
else
{
lean_object* v_reuseFailAlloc_290_; 
v_reuseFailAlloc_290_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_290_, 0, v_a_284_);
v___x_289_ = v_reuseFailAlloc_290_;
goto v_reusejp_288_;
}
v_reusejp_288_:
{
return v___x_289_;
}
}
}
}
v___jp_296_:
{
uint8_t v_result_299_; lean_object* v___x_300_; lean_object* v___x_301_; double v___x_302_; lean_object* v_data_303_; 
v_result_299_ = lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__2_spec__4(v_fst_276_);
v___x_300_ = lean_box(v_result_299_);
v___x_301_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_301_, 0, v___x_300_);
v___x_302_ = lean_float_once(&lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__2___closed__0, &lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__2___closed__0_once, _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__2___closed__0);
lean_inc_ref(v_tag_264_);
lean_inc_ref(v___x_301_);
lean_inc(v_cls_262_);
v_data_303_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v_data_303_, 0, v_cls_262_);
lean_ctor_set(v_data_303_, 1, v___x_301_);
lean_ctor_set(v_data_303_, 2, v_tag_264_);
lean_ctor_set_float(v_data_303_, sizeof(void*)*3, v___x_302_);
lean_ctor_set_float(v_data_303_, sizeof(void*)*3 + 8, v___x_302_);
lean_ctor_set_uint8(v_data_303_, sizeof(void*)*3 + 16, v_collapsed_263_);
if (v___x_295_ == 0)
{
lean_dec_ref_known(v___x_301_, 1);
lean_dec(v_snd_293_);
lean_dec(v_fst_292_);
lean_dec_ref(v_tag_264_);
lean_dec(v_cls_262_);
v___y_279_ = v_a_298_;
v___y_280_ = v___y_297_;
v_data_281_ = v_data_303_;
goto v___jp_278_;
}
else
{
lean_object* v_data_304_; double v___x_305_; double v___x_306_; 
lean_dec_ref_known(v_data_303_, 3);
v_data_304_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v_data_304_, 0, v_cls_262_);
lean_ctor_set(v_data_304_, 1, v___x_301_);
lean_ctor_set(v_data_304_, 2, v_tag_264_);
v___x_305_ = lean_unbox_float(v_fst_292_);
lean_dec(v_fst_292_);
lean_ctor_set_float(v_data_304_, sizeof(void*)*3, v___x_305_);
v___x_306_ = lean_unbox_float(v_snd_293_);
lean_dec(v_snd_293_);
lean_ctor_set_float(v_data_304_, sizeof(void*)*3 + 8, v___x_306_);
lean_ctor_set_uint8(v_data_304_, sizeof(void*)*3 + 16, v_collapsed_263_);
v___y_279_ = v_a_298_;
v___y_280_ = v___y_297_;
v_data_281_ = v_data_304_;
goto v___jp_278_;
}
}
v___jp_307_:
{
lean_object* v_ref_308_; lean_object* v___x_309_; 
v_ref_308_ = lean_ctor_get(v___y_273_, 5);
lean_inc(v___y_274_);
lean_inc_ref(v___y_273_);
lean_inc(v___y_272_);
lean_inc_ref(v___y_271_);
lean_inc(v___y_270_);
lean_inc(v_fst_276_);
v___x_309_ = lean_apply_7(v_msg_268_, v_fst_276_, v___y_270_, v___y_271_, v___y_272_, v___y_273_, v___y_274_, lean_box(0));
if (lean_obj_tag(v___x_309_) == 0)
{
lean_object* v_a_310_; 
v_a_310_ = lean_ctor_get(v___x_309_, 0);
lean_inc(v_a_310_);
lean_dec_ref_known(v___x_309_, 1);
v___y_297_ = v_ref_308_;
v_a_298_ = v_a_310_;
goto v___jp_296_;
}
else
{
lean_object* v___x_311_; 
lean_dec_ref_known(v___x_309_, 1);
v___x_311_ = lean_obj_once(&lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__2___closed__2, &lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__2___closed__2_once, _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__2___closed__2);
v___y_297_ = v_ref_308_;
v_a_298_ = v___x_311_;
goto v___jp_296_;
}
}
v___jp_312_:
{
if (v_clsEnabled_266_ == 0)
{
if (v___y_313_ == 0)
{
lean_object* v___x_314_; lean_object* v_traceState_315_; lean_object* v_env_316_; lean_object* v_nextMacroScope_317_; lean_object* v_ngen_318_; lean_object* v_auxDeclNGen_319_; lean_object* v_cache_320_; lean_object* v_messages_321_; lean_object* v_infoState_322_; lean_object* v_snapshotTasks_323_; lean_object* v___x_325_; uint8_t v_isShared_326_; uint8_t v_isSharedCheck_342_; 
lean_dec(v_snd_293_);
lean_dec(v_fst_292_);
lean_dec_ref(v_msg_268_);
lean_dec_ref(v_tag_264_);
lean_dec(v_cls_262_);
v___x_314_ = lean_st_ref_take(v___y_274_);
v_traceState_315_ = lean_ctor_get(v___x_314_, 4);
v_env_316_ = lean_ctor_get(v___x_314_, 0);
v_nextMacroScope_317_ = lean_ctor_get(v___x_314_, 1);
v_ngen_318_ = lean_ctor_get(v___x_314_, 2);
v_auxDeclNGen_319_ = lean_ctor_get(v___x_314_, 3);
v_cache_320_ = lean_ctor_get(v___x_314_, 5);
v_messages_321_ = lean_ctor_get(v___x_314_, 6);
v_infoState_322_ = lean_ctor_get(v___x_314_, 7);
v_snapshotTasks_323_ = lean_ctor_get(v___x_314_, 8);
v_isSharedCheck_342_ = !lean_is_exclusive(v___x_314_);
if (v_isSharedCheck_342_ == 0)
{
v___x_325_ = v___x_314_;
v_isShared_326_ = v_isSharedCheck_342_;
goto v_resetjp_324_;
}
else
{
lean_inc(v_snapshotTasks_323_);
lean_inc(v_infoState_322_);
lean_inc(v_messages_321_);
lean_inc(v_cache_320_);
lean_inc(v_traceState_315_);
lean_inc(v_auxDeclNGen_319_);
lean_inc(v_ngen_318_);
lean_inc(v_nextMacroScope_317_);
lean_inc(v_env_316_);
lean_dec(v___x_314_);
v___x_325_ = lean_box(0);
v_isShared_326_ = v_isSharedCheck_342_;
goto v_resetjp_324_;
}
v_resetjp_324_:
{
uint64_t v_tid_327_; lean_object* v_traces_328_; lean_object* v___x_330_; uint8_t v_isShared_331_; uint8_t v_isSharedCheck_341_; 
v_tid_327_ = lean_ctor_get_uint64(v_traceState_315_, sizeof(void*)*1);
v_traces_328_ = lean_ctor_get(v_traceState_315_, 0);
v_isSharedCheck_341_ = !lean_is_exclusive(v_traceState_315_);
if (v_isSharedCheck_341_ == 0)
{
v___x_330_ = v_traceState_315_;
v_isShared_331_ = v_isSharedCheck_341_;
goto v_resetjp_329_;
}
else
{
lean_inc(v_traces_328_);
lean_dec(v_traceState_315_);
v___x_330_ = lean_box(0);
v_isShared_331_ = v_isSharedCheck_341_;
goto v_resetjp_329_;
}
v_resetjp_329_:
{
lean_object* v___x_332_; lean_object* v___x_334_; 
v___x_332_ = l_Lean_PersistentArray_append___redArg(v_oldTraces_267_, v_traces_328_);
lean_dec_ref(v_traces_328_);
if (v_isShared_331_ == 0)
{
lean_ctor_set(v___x_330_, 0, v___x_332_);
v___x_334_ = v___x_330_;
goto v_reusejp_333_;
}
else
{
lean_object* v_reuseFailAlloc_340_; 
v_reuseFailAlloc_340_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_340_, 0, v___x_332_);
lean_ctor_set_uint64(v_reuseFailAlloc_340_, sizeof(void*)*1, v_tid_327_);
v___x_334_ = v_reuseFailAlloc_340_;
goto v_reusejp_333_;
}
v_reusejp_333_:
{
lean_object* v___x_336_; 
if (v_isShared_326_ == 0)
{
lean_ctor_set(v___x_325_, 4, v___x_334_);
v___x_336_ = v___x_325_;
goto v_reusejp_335_;
}
else
{
lean_object* v_reuseFailAlloc_339_; 
v_reuseFailAlloc_339_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_339_, 0, v_env_316_);
lean_ctor_set(v_reuseFailAlloc_339_, 1, v_nextMacroScope_317_);
lean_ctor_set(v_reuseFailAlloc_339_, 2, v_ngen_318_);
lean_ctor_set(v_reuseFailAlloc_339_, 3, v_auxDeclNGen_319_);
lean_ctor_set(v_reuseFailAlloc_339_, 4, v___x_334_);
lean_ctor_set(v_reuseFailAlloc_339_, 5, v_cache_320_);
lean_ctor_set(v_reuseFailAlloc_339_, 6, v_messages_321_);
lean_ctor_set(v_reuseFailAlloc_339_, 7, v_infoState_322_);
lean_ctor_set(v_reuseFailAlloc_339_, 8, v_snapshotTasks_323_);
v___x_336_ = v_reuseFailAlloc_339_;
goto v_reusejp_335_;
}
v_reusejp_335_:
{
lean_object* v___x_337_; lean_object* v___x_338_; 
v___x_337_ = lean_st_ref_set(v___y_274_, v___x_336_);
v___x_338_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__2_spec__3___redArg(v_fst_276_);
return v___x_338_;
}
}
}
}
}
else
{
goto v___jp_307_;
}
}
else
{
goto v___jp_307_;
}
}
v___jp_343_:
{
double v___x_345_; double v___x_346_; double v___x_347_; uint8_t v___x_348_; 
v___x_345_ = lean_unbox_float(v_snd_293_);
v___x_346_ = lean_unbox_float(v_fst_292_);
v___x_347_ = lean_float_sub(v___x_345_, v___x_346_);
v___x_348_ = lean_float_decLt(v___y_344_, v___x_347_);
v___y_313_ = v___x_348_;
goto v___jp_312_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__2___boxed(lean_object* v_cls_359_, lean_object* v_collapsed_360_, lean_object* v_tag_361_, lean_object* v_opts_362_, lean_object* v_clsEnabled_363_, lean_object* v_oldTraces_364_, lean_object* v_msg_365_, lean_object* v_resStartStop_366_, lean_object* v___y_367_, lean_object* v___y_368_, lean_object* v___y_369_, lean_object* v___y_370_, lean_object* v___y_371_, lean_object* v___y_372_){
_start:
{
uint8_t v_collapsed_boxed_373_; uint8_t v_clsEnabled_boxed_374_; lean_object* v_res_375_; 
v_collapsed_boxed_373_ = lean_unbox(v_collapsed_360_);
v_clsEnabled_boxed_374_ = lean_unbox(v_clsEnabled_363_);
v_res_375_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__2(v_cls_359_, v_collapsed_boxed_373_, v_tag_361_, v_opts_362_, v_clsEnabled_boxed_374_, v_oldTraces_364_, v_msg_365_, v_resStartStop_366_, v___y_367_, v___y_368_, v___y_369_, v___y_370_, v___y_371_);
lean_dec(v___y_371_);
lean_dec_ref(v___y_370_);
lean_dec(v___y_369_);
lean_dec_ref(v___y_368_);
lean_dec(v___y_367_);
lean_dec_ref(v_opts_362_);
return v_res_375_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp___closed__1(void){
_start:
{
lean_object* v___x_377_; lean_object* v___x_378_; 
v___x_377_ = ((lean_object*)(lp_aesop___private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp___closed__0));
v___x_378_ = l_Lean_stringToMessageData(v___x_377_);
return v___x_378_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp___closed__3(void){
_start:
{
lean_object* v___x_380_; lean_object* v___x_381_; 
v___x_380_ = ((lean_object*)(lp_aesop___private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp___closed__2));
v___x_381_ = l_Lean_stringToMessageData(v___x_380_);
return v___x_381_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp___closed__5(void){
_start:
{
lean_object* v___x_383_; lean_object* v___x_384_; 
v___x_383_ = ((lean_object*)(lp_aesop___private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp___closed__4));
v___x_384_ = l_Lean_stringToMessageData(v___x_383_);
return v___x_384_;
}
}
static double _init_lp_aesop___private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp___closed__9(void){
_start:
{
lean_object* v___x_389_; double v___x_390_; 
v___x_389_ = lean_unsigned_to_nat(1000000000u);
v___x_390_ = lean_float_of_nat(v___x_389_);
return v___x_390_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp(lean_object* v_h_391_, lean_object* v_fs_392_, lean_object* v_a_393_, lean_object* v_a_394_, lean_object* v_a_395_, lean_object* v_a_396_, lean_object* v_a_397_){
_start:
{
lean_object* v_options_399_; lean_object* v_inheritedTraceOptions_400_; uint8_t v_hasTrace_401_; lean_object* v___x_402_; lean_object* v___x_403_; 
v_options_399_ = lean_ctor_get(v_a_396_, 2);
v_inheritedTraceOptions_400_ = lean_ctor_get(v_a_396_, 13);
v_hasTrace_401_ = lean_ctor_get_uint8(v_options_399_, sizeof(void*)*1);
lean_inc_n(v_h_391_, 2);
v___x_402_ = l_Lean_Expr_fvar___override(v_h_391_);
v___x_403_ = lp_aesop_Aesop_ForwardState_eraseHyp(v_h_391_, v_fs_392_);
if (v_hasTrace_401_ == 0)
{
lean_object* v___x_404_; 
lean_dec_ref(v___x_402_);
lean_dec(v_h_391_);
v___x_404_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_404_, 0, v___x_403_);
return v___x_404_;
}
else
{
lean_object* v___x_405_; lean_object* v_traceClass_406_; lean_object* v___x_407_; lean_object* v___x_408_; lean_object* v___x_409_; lean_object* v___x_410_; lean_object* v___x_411_; lean_object* v___x_412_; lean_object* v___x_413_; lean_object* v___x_414_; lean_object* v___x_415_; lean_object* v___f_416_; lean_object* v___x_417_; lean_object* v___x_418_; lean_object* v___x_419_; uint8_t v___x_420_; 
v___x_405_ = lp_aesop_Aesop_TraceOption_forward;
v_traceClass_406_ = lean_ctor_get(v___x_405_, 0);
v___x_407_ = lean_obj_once(&lp_aesop___private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp___closed__1, &lp_aesop___private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp___closed__1_once, _init_lp_aesop___private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp___closed__1);
v___x_408_ = l_Lean_MessageData_ofExpr(v___x_402_);
v___x_409_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_409_, 0, v___x_407_);
lean_ctor_set(v___x_409_, 1, v___x_408_);
v___x_410_ = lean_obj_once(&lp_aesop___private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp___closed__3, &lp_aesop___private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp___closed__3_once, _init_lp_aesop___private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp___closed__3);
v___x_411_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_411_, 0, v___x_409_);
lean_ctor_set(v___x_411_, 1, v___x_410_);
v___x_412_ = l_Lean_MessageData_ofName(v_h_391_);
v___x_413_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_413_, 0, v___x_411_);
lean_ctor_set(v___x_413_, 1, v___x_412_);
v___x_414_ = lean_obj_once(&lp_aesop___private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp___closed__5, &lp_aesop___private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp___closed__5_once, _init_lp_aesop___private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp___closed__5);
v___x_415_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_415_, 0, v___x_413_);
lean_ctor_set(v___x_415_, 1, v___x_414_);
v___f_416_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp___lam__0___boxed), 8, 1);
lean_closure_set(v___f_416_, 0, v___x_415_);
v___x_417_ = ((lean_object*)(lp_aesop___private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp___closed__6));
v___x_418_ = ((lean_object*)(lp_aesop___private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp___closed__8));
lean_inc(v_traceClass_406_);
v___x_419_ = l_Lean_Name_append(v___x_418_, v_traceClass_406_);
v___x_420_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_400_, v_options_399_, v___x_419_);
lean_dec(v___x_419_);
if (v___x_420_ == 0)
{
lean_object* v___x_457_; uint8_t v___x_458_; 
v___x_457_ = l_Lean_trace_profiler;
v___x_458_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__1(v_options_399_, v___x_457_);
if (v___x_458_ == 0)
{
lean_object* v___x_459_; 
lean_dec_ref(v___f_416_);
v___x_459_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_459_, 0, v___x_403_);
return v___x_459_;
}
else
{
goto v___jp_421_;
}
}
else
{
goto v___jp_421_;
}
v___jp_421_:
{
lean_object* v___x_422_; lean_object* v_a_423_; lean_object* v___x_425_; uint8_t v_isShared_426_; uint8_t v_isSharedCheck_456_; 
v___x_422_ = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__0___redArg(v_a_397_);
v_a_423_ = lean_ctor_get(v___x_422_, 0);
v_isSharedCheck_456_ = !lean_is_exclusive(v___x_422_);
if (v_isSharedCheck_456_ == 0)
{
v___x_425_ = v___x_422_;
v_isShared_426_ = v_isSharedCheck_456_;
goto v_resetjp_424_;
}
else
{
lean_inc(v_a_423_);
lean_dec(v___x_422_);
v___x_425_ = lean_box(0);
v_isShared_426_ = v_isSharedCheck_456_;
goto v_resetjp_424_;
}
v_resetjp_424_:
{
lean_object* v___x_427_; uint8_t v___x_428_; 
v___x_427_ = l_Lean_trace_profiler_useHeartbeats;
v___x_428_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__1(v_options_399_, v___x_427_);
if (v___x_428_ == 0)
{
lean_object* v___x_429_; lean_object* v___x_430_; lean_object* v___x_432_; 
v___x_429_ = lean_io_mono_nanos_now();
v___x_430_ = lean_io_mono_nanos_now();
if (v_isShared_426_ == 0)
{
lean_ctor_set_tag(v___x_425_, 1);
lean_ctor_set(v___x_425_, 0, v___x_403_);
v___x_432_ = v___x_425_;
goto v_reusejp_431_;
}
else
{
lean_object* v_reuseFailAlloc_443_; 
v_reuseFailAlloc_443_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_443_, 0, v___x_403_);
v___x_432_ = v_reuseFailAlloc_443_;
goto v_reusejp_431_;
}
v_reusejp_431_:
{
double v___x_433_; double v___x_434_; double v___x_435_; double v___x_436_; double v___x_437_; lean_object* v___x_438_; lean_object* v___x_439_; lean_object* v___x_440_; lean_object* v___x_441_; lean_object* v___x_442_; 
v___x_433_ = lean_float_of_nat(v___x_429_);
v___x_434_ = lean_float_once(&lp_aesop___private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp___closed__9, &lp_aesop___private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp___closed__9_once, _init_lp_aesop___private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp___closed__9);
v___x_435_ = lean_float_div(v___x_433_, v___x_434_);
v___x_436_ = lean_float_of_nat(v___x_430_);
v___x_437_ = lean_float_div(v___x_436_, v___x_434_);
v___x_438_ = lean_box_float(v___x_435_);
v___x_439_ = lean_box_float(v___x_437_);
v___x_440_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_440_, 0, v___x_438_);
lean_ctor_set(v___x_440_, 1, v___x_439_);
v___x_441_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_441_, 0, v___x_432_);
lean_ctor_set(v___x_441_, 1, v___x_440_);
lean_inc(v_traceClass_406_);
v___x_442_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__2(v_traceClass_406_, v_hasTrace_401_, v___x_417_, v_options_399_, v___x_420_, v_a_423_, v___f_416_, v___x_441_, v_a_393_, v_a_394_, v_a_395_, v_a_396_, v_a_397_);
return v___x_442_;
}
}
else
{
lean_object* v___x_444_; lean_object* v___x_445_; lean_object* v___x_447_; 
v___x_444_ = lean_io_get_num_heartbeats();
v___x_445_ = lean_io_get_num_heartbeats();
if (v_isShared_426_ == 0)
{
lean_ctor_set_tag(v___x_425_, 1);
lean_ctor_set(v___x_425_, 0, v___x_403_);
v___x_447_ = v___x_425_;
goto v_reusejp_446_;
}
else
{
lean_object* v_reuseFailAlloc_455_; 
v_reuseFailAlloc_455_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_455_, 0, v___x_403_);
v___x_447_ = v_reuseFailAlloc_455_;
goto v_reusejp_446_;
}
v_reusejp_446_:
{
double v___x_448_; double v___x_449_; lean_object* v___x_450_; lean_object* v___x_451_; lean_object* v___x_452_; lean_object* v___x_453_; lean_object* v___x_454_; 
v___x_448_ = lean_float_of_nat(v___x_444_);
v___x_449_ = lean_float_of_nat(v___x_445_);
v___x_450_ = lean_box_float(v___x_448_);
v___x_451_ = lean_box_float(v___x_449_);
v___x_452_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_452_, 0, v___x_450_);
lean_ctor_set(v___x_452_, 1, v___x_451_);
v___x_453_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_453_, 0, v___x_447_);
lean_ctor_set(v___x_453_, 1, v___x_452_);
lean_inc(v_traceClass_406_);
v___x_454_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__2(v_traceClass_406_, v_hasTrace_401_, v___x_417_, v_options_399_, v___x_420_, v_a_423_, v___f_416_, v___x_453_, v_a_393_, v_a_394_, v_a_395_, v_a_396_, v_a_397_);
return v___x_454_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp___boxed(lean_object* v_h_460_, lean_object* v_fs_461_, lean_object* v_a_462_, lean_object* v_a_463_, lean_object* v_a_464_, lean_object* v_a_465_, lean_object* v_a_466_, lean_object* v_a_467_){
_start:
{
lean_object* v_res_468_; 
v_res_468_ = lp_aesop___private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp(v_h_460_, v_fs_461_, v_a_462_, v_a_463_, v_a_464_, v_a_465_, v_a_466_);
lean_dec(v_a_466_);
lean_dec_ref(v_a_465_);
lean_dec(v_a_464_);
lean_dec_ref(v_a_463_);
lean_dec(v_a_462_);
return v_res_468_;
}
}
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__2_spec__3(lean_object* v_00_u03b1_469_, lean_object* v_x_470_, lean_object* v___y_471_, lean_object* v___y_472_, lean_object* v___y_473_, lean_object* v___y_474_, lean_object* v___y_475_){
_start:
{
lean_object* v___x_477_; 
v___x_477_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__2_spec__3___redArg(v_x_470_);
return v___x_477_;
}
}
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__2_spec__3___boxed(lean_object* v_00_u03b1_478_, lean_object* v_x_479_, lean_object* v___y_480_, lean_object* v___y_481_, lean_object* v___y_482_, lean_object* v___y_483_, lean_object* v___y_484_, lean_object* v___y_485_){
_start:
{
lean_object* v_res_486_; 
v_res_486_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__2_spec__3(v_00_u03b1_478_, v_x_479_, v___y_480_, v___y_481_, v___y_482_, v___y_483_, v___y_484_);
lean_dec(v___y_484_);
lean_dec_ref(v___y_483_);
lean_dec(v___y_482_);
lean_dec_ref(v___y_481_);
lean_dec(v___y_480_);
return v_res_486_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__2_spec__2(lean_object* v_oldTraces_487_, lean_object* v_data_488_, lean_object* v_ref_489_, lean_object* v_msg_490_, lean_object* v___y_491_, lean_object* v___y_492_, lean_object* v___y_493_, lean_object* v___y_494_, lean_object* v___y_495_){
_start:
{
lean_object* v___x_497_; 
v___x_497_ = lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__2_spec__2___redArg(v_oldTraces_487_, v_data_488_, v_ref_489_, v_msg_490_, v___y_492_, v___y_493_, v___y_494_, v___y_495_);
return v___x_497_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__2_spec__2___boxed(lean_object* v_oldTraces_498_, lean_object* v_data_499_, lean_object* v_ref_500_, lean_object* v_msg_501_, lean_object* v___y_502_, lean_object* v___y_503_, lean_object* v___y_504_, lean_object* v___y_505_, lean_object* v___y_506_, lean_object* v___y_507_){
_start:
{
lean_object* v_res_508_; 
v_res_508_ = lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__2_spec__2(v_oldTraces_498_, v_data_499_, v_ref_500_, v_msg_501_, v___y_502_, v___y_503_, v___y_504_, v___y_505_, v___y_506_);
lean_dec(v___y_506_);
lean_dec_ref(v___y_505_);
lean_dec(v___y_504_);
lean_dec_ref(v___y_503_);
lean_dec(v___y_502_);
return v_res_508_;
}
}
LEAN_EXPORT uint8_t lp_aesop___private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_addHyp___lam__0(lean_object* v_x_509_){
_start:
{
uint8_t v___x_510_; 
v___x_510_ = 1;
return v___x_510_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_addHyp___lam__0___boxed(lean_object* v_x_511_){
_start:
{
uint8_t v_res_512_; lean_object* v_r_513_; 
v_res_512_ = lp_aesop___private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_addHyp___lam__0(v_x_511_);
lean_dec_ref(v_x_511_);
v_r_513_ = lean_box(v_res_512_);
return v_r_513_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_addHyp(lean_object* v_rs_515_, lean_object* v_h_516_, lean_object* v_fs_517_, lean_object* v_a_518_, lean_object* v_a_519_, lean_object* v_a_520_, lean_object* v_a_521_, lean_object* v_a_522_){
_start:
{
lean_object* v___x_524_; 
lean_inc(v_h_516_);
v___x_524_ = l_Lean_FVarId_getType___redArg(v_h_516_, v_a_519_, v_a_521_, v_a_522_);
if (lean_obj_tag(v___x_524_) == 0)
{
lean_object* v_a_525_; lean_object* v___f_526_; lean_object* v___x_527_; 
v_a_525_ = lean_ctor_get(v___x_524_, 0);
lean_inc(v_a_525_);
lean_dec_ref_known(v___x_524_, 1);
v___f_526_ = ((lean_object*)(lp_aesop___private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_addHyp___closed__0));
lean_inc_ref(v_rs_515_);
v___x_527_ = lp_aesop_Aesop_LocalRuleSet_applicableForwardRulesWith(v_rs_515_, v_a_525_, v___f_526_, v_a_519_, v_a_520_, v_a_521_, v_a_522_);
if (lean_obj_tag(v___x_527_) == 0)
{
lean_object* v_a_528_; lean_object* v___x_529_; 
v_a_528_ = lean_ctor_get(v___x_527_, 0);
lean_inc(v_a_528_);
lean_dec_ref_known(v___x_527_, 1);
lean_inc(v_h_516_);
v___x_529_ = l_Lean_FVarId_getDecl___redArg(v_h_516_, v_a_519_, v_a_521_, v_a_522_);
if (lean_obj_tag(v___x_529_) == 0)
{
lean_object* v_a_530_; lean_object* v___x_531_; 
v_a_530_ = lean_ctor_get(v___x_529_, 0);
lean_inc(v_a_530_);
lean_dec_ref_known(v___x_529_, 1);
v___x_531_ = lp_aesop_Aesop_LocalRuleSet_forwardRulePatternSubstsInLocalDecl(v_rs_515_, v_a_530_, v_a_518_, v_a_519_, v_a_520_, v_a_521_, v_a_522_);
if (lean_obj_tag(v___x_531_) == 0)
{
lean_object* v_a_532_; lean_object* v___x_534_; uint8_t v_isShared_535_; uint8_t v_isSharedCheck_540_; 
v_a_532_ = lean_ctor_get(v___x_531_, 0);
v_isSharedCheck_540_ = !lean_is_exclusive(v___x_531_);
if (v_isSharedCheck_540_ == 0)
{
v___x_534_ = v___x_531_;
v_isShared_535_ = v_isSharedCheck_540_;
goto v_resetjp_533_;
}
else
{
lean_inc(v_a_532_);
lean_dec(v___x_531_);
v___x_534_ = lean_box(0);
v_isShared_535_ = v_isSharedCheck_540_;
goto v_resetjp_533_;
}
v_resetjp_533_:
{
lean_object* v___x_536_; lean_object* v___x_538_; 
v___x_536_ = lp_aesop_Aesop_ForwardState_enqueueHypWithPatSubsts(v_h_516_, v_a_528_, v_a_532_, v_fs_517_);
lean_dec(v_a_532_);
if (v_isShared_535_ == 0)
{
lean_ctor_set(v___x_534_, 0, v___x_536_);
v___x_538_ = v___x_534_;
goto v_reusejp_537_;
}
else
{
lean_object* v_reuseFailAlloc_539_; 
v_reuseFailAlloc_539_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_539_, 0, v___x_536_);
v___x_538_ = v_reuseFailAlloc_539_;
goto v_reusejp_537_;
}
v_reusejp_537_:
{
return v___x_538_;
}
}
}
else
{
lean_object* v_a_541_; lean_object* v___x_543_; uint8_t v_isShared_544_; uint8_t v_isSharedCheck_548_; 
lean_dec(v_a_528_);
lean_dec_ref(v_fs_517_);
lean_dec(v_h_516_);
v_a_541_ = lean_ctor_get(v___x_531_, 0);
v_isSharedCheck_548_ = !lean_is_exclusive(v___x_531_);
if (v_isSharedCheck_548_ == 0)
{
v___x_543_ = v___x_531_;
v_isShared_544_ = v_isSharedCheck_548_;
goto v_resetjp_542_;
}
else
{
lean_inc(v_a_541_);
lean_dec(v___x_531_);
v___x_543_ = lean_box(0);
v_isShared_544_ = v_isSharedCheck_548_;
goto v_resetjp_542_;
}
v_resetjp_542_:
{
lean_object* v___x_546_; 
if (v_isShared_544_ == 0)
{
v___x_546_ = v___x_543_;
goto v_reusejp_545_;
}
else
{
lean_object* v_reuseFailAlloc_547_; 
v_reuseFailAlloc_547_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_547_, 0, v_a_541_);
v___x_546_ = v_reuseFailAlloc_547_;
goto v_reusejp_545_;
}
v_reusejp_545_:
{
return v___x_546_;
}
}
}
}
else
{
lean_object* v_a_549_; lean_object* v___x_551_; uint8_t v_isShared_552_; uint8_t v_isSharedCheck_556_; 
lean_dec(v_a_528_);
lean_dec_ref(v_fs_517_);
lean_dec(v_h_516_);
lean_dec_ref(v_rs_515_);
v_a_549_ = lean_ctor_get(v___x_529_, 0);
v_isSharedCheck_556_ = !lean_is_exclusive(v___x_529_);
if (v_isSharedCheck_556_ == 0)
{
v___x_551_ = v___x_529_;
v_isShared_552_ = v_isSharedCheck_556_;
goto v_resetjp_550_;
}
else
{
lean_inc(v_a_549_);
lean_dec(v___x_529_);
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
else
{
lean_object* v_a_557_; lean_object* v___x_559_; uint8_t v_isShared_560_; uint8_t v_isSharedCheck_564_; 
lean_dec_ref(v_fs_517_);
lean_dec(v_h_516_);
lean_dec_ref(v_rs_515_);
v_a_557_ = lean_ctor_get(v___x_527_, 0);
v_isSharedCheck_564_ = !lean_is_exclusive(v___x_527_);
if (v_isSharedCheck_564_ == 0)
{
v___x_559_ = v___x_527_;
v_isShared_560_ = v_isSharedCheck_564_;
goto v_resetjp_558_;
}
else
{
lean_inc(v_a_557_);
lean_dec(v___x_527_);
v___x_559_ = lean_box(0);
v_isShared_560_ = v_isSharedCheck_564_;
goto v_resetjp_558_;
}
v_resetjp_558_:
{
lean_object* v___x_562_; 
if (v_isShared_560_ == 0)
{
v___x_562_ = v___x_559_;
goto v_reusejp_561_;
}
else
{
lean_object* v_reuseFailAlloc_563_; 
v_reuseFailAlloc_563_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_563_, 0, v_a_557_);
v___x_562_ = v_reuseFailAlloc_563_;
goto v_reusejp_561_;
}
v_reusejp_561_:
{
return v___x_562_;
}
}
}
}
else
{
lean_object* v_a_565_; lean_object* v___x_567_; uint8_t v_isShared_568_; uint8_t v_isSharedCheck_572_; 
lean_dec_ref(v_fs_517_);
lean_dec(v_h_516_);
lean_dec_ref(v_rs_515_);
v_a_565_ = lean_ctor_get(v___x_524_, 0);
v_isSharedCheck_572_ = !lean_is_exclusive(v___x_524_);
if (v_isSharedCheck_572_ == 0)
{
v___x_567_ = v___x_524_;
v_isShared_568_ = v_isSharedCheck_572_;
goto v_resetjp_566_;
}
else
{
lean_inc(v_a_565_);
lean_dec(v___x_524_);
v___x_567_ = lean_box(0);
v_isShared_568_ = v_isSharedCheck_572_;
goto v_resetjp_566_;
}
v_resetjp_566_:
{
lean_object* v___x_570_; 
if (v_isShared_568_ == 0)
{
v___x_570_ = v___x_567_;
goto v_reusejp_569_;
}
else
{
lean_object* v_reuseFailAlloc_571_; 
v_reuseFailAlloc_571_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_571_, 0, v_a_565_);
v___x_570_ = v_reuseFailAlloc_571_;
goto v_reusejp_569_;
}
v_reusejp_569_:
{
return v___x_570_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_addHyp___boxed(lean_object* v_rs_573_, lean_object* v_h_574_, lean_object* v_fs_575_, lean_object* v_a_576_, lean_object* v_a_577_, lean_object* v_a_578_, lean_object* v_a_579_, lean_object* v_a_580_, lean_object* v_a_581_){
_start:
{
lean_object* v_res_582_; 
v_res_582_ = lp_aesop___private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_addHyp(v_rs_573_, v_h_574_, v_fs_575_, v_a_576_, v_a_577_, v_a_578_, v_a_579_, v_a_580_);
lean_dec(v_a_580_);
lean_dec_ref(v_a_579_);
lean_dec(v_a_578_);
lean_dec_ref(v_a_577_);
lean_dec(v_a_576_);
return v_res_582_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_updateTarget(lean_object* v_rs_583_, lean_object* v_diff_584_, lean_object* v_fs_585_, lean_object* v_a_586_, lean_object* v_a_587_, lean_object* v_a_588_, lean_object* v_a_589_, lean_object* v_a_590_){
_start:
{
lean_object* v_newGoal_592_; lean_object* v___x_593_; 
v_newGoal_592_ = lean_ctor_get(v_diff_584_, 1);
lean_inc(v_newGoal_592_);
lean_dec_ref(v_diff_584_);
v___x_593_ = l_Lean_MVarId_getType(v_newGoal_592_, v_a_587_, v_a_588_, v_a_589_, v_a_590_);
if (lean_obj_tag(v___x_593_) == 0)
{
lean_object* v_a_594_; lean_object* v___x_595_; 
v_a_594_ = lean_ctor_get(v___x_593_, 0);
lean_inc(v_a_594_);
lean_dec_ref_known(v___x_593_, 1);
v___x_595_ = lp_aesop_Aesop_LocalRuleSet_forwardRulePatternSubstsInExpr(v_rs_583_, v_a_594_, v_a_586_, v_a_587_, v_a_588_, v_a_589_, v_a_590_);
if (lean_obj_tag(v___x_595_) == 0)
{
lean_object* v_a_596_; lean_object* v___x_598_; uint8_t v_isShared_599_; uint8_t v_isSharedCheck_604_; 
v_a_596_ = lean_ctor_get(v___x_595_, 0);
v_isSharedCheck_604_ = !lean_is_exclusive(v___x_595_);
if (v_isSharedCheck_604_ == 0)
{
v___x_598_ = v___x_595_;
v_isShared_599_ = v_isSharedCheck_604_;
goto v_resetjp_597_;
}
else
{
lean_inc(v_a_596_);
lean_dec(v___x_595_);
v___x_598_ = lean_box(0);
v_isShared_599_ = v_isSharedCheck_604_;
goto v_resetjp_597_;
}
v_resetjp_597_:
{
lean_object* v___x_600_; lean_object* v___x_602_; 
v___x_600_ = lp_aesop_Aesop_ForwardState_enqueueTargetPatSubsts(v_a_596_, v_fs_585_);
lean_dec(v_a_596_);
if (v_isShared_599_ == 0)
{
lean_ctor_set(v___x_598_, 0, v___x_600_);
v___x_602_ = v___x_598_;
goto v_reusejp_601_;
}
else
{
lean_object* v_reuseFailAlloc_603_; 
v_reuseFailAlloc_603_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_603_, 0, v___x_600_);
v___x_602_ = v_reuseFailAlloc_603_;
goto v_reusejp_601_;
}
v_reusejp_601_:
{
return v___x_602_;
}
}
}
else
{
lean_object* v_a_605_; lean_object* v___x_607_; uint8_t v_isShared_608_; uint8_t v_isSharedCheck_612_; 
lean_dec_ref(v_fs_585_);
v_a_605_ = lean_ctor_get(v___x_595_, 0);
v_isSharedCheck_612_ = !lean_is_exclusive(v___x_595_);
if (v_isSharedCheck_612_ == 0)
{
v___x_607_ = v___x_595_;
v_isShared_608_ = v_isSharedCheck_612_;
goto v_resetjp_606_;
}
else
{
lean_inc(v_a_605_);
lean_dec(v___x_595_);
v___x_607_ = lean_box(0);
v_isShared_608_ = v_isSharedCheck_612_;
goto v_resetjp_606_;
}
v_resetjp_606_:
{
lean_object* v___x_610_; 
if (v_isShared_608_ == 0)
{
v___x_610_ = v___x_607_;
goto v_reusejp_609_;
}
else
{
lean_object* v_reuseFailAlloc_611_; 
v_reuseFailAlloc_611_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_611_, 0, v_a_605_);
v___x_610_ = v_reuseFailAlloc_611_;
goto v_reusejp_609_;
}
v_reusejp_609_:
{
return v___x_610_;
}
}
}
}
else
{
lean_object* v_a_613_; lean_object* v___x_615_; uint8_t v_isShared_616_; uint8_t v_isSharedCheck_620_; 
lean_dec_ref(v_fs_585_);
lean_dec_ref(v_rs_583_);
v_a_613_ = lean_ctor_get(v___x_593_, 0);
v_isSharedCheck_620_ = !lean_is_exclusive(v___x_593_);
if (v_isSharedCheck_620_ == 0)
{
v___x_615_ = v___x_593_;
v_isShared_616_ = v_isSharedCheck_620_;
goto v_resetjp_614_;
}
else
{
lean_inc(v_a_613_);
lean_dec(v___x_593_);
v___x_615_ = lean_box(0);
v_isShared_616_ = v_isSharedCheck_620_;
goto v_resetjp_614_;
}
v_resetjp_614_:
{
lean_object* v___x_618_; 
if (v_isShared_616_ == 0)
{
v___x_618_ = v___x_615_;
goto v_reusejp_617_;
}
else
{
lean_object* v_reuseFailAlloc_619_; 
v_reuseFailAlloc_619_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_619_, 0, v_a_613_);
v___x_618_ = v_reuseFailAlloc_619_;
goto v_reusejp_617_;
}
v_reusejp_617_:
{
return v___x_618_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_updateTarget___boxed(lean_object* v_rs_621_, lean_object* v_diff_622_, lean_object* v_fs_623_, lean_object* v_a_624_, lean_object* v_a_625_, lean_object* v_a_626_, lean_object* v_a_627_, lean_object* v_a_628_, lean_object* v_a_629_){
_start:
{
lean_object* v_res_630_; 
v_res_630_ = lp_aesop___private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_updateTarget(v_rs_621_, v_diff_622_, v_fs_623_, v_a_624_, v_a_625_, v_a_626_, v_a_627_, v_a_628_);
lean_dec(v_a_628_);
lean_dec_ref(v_a_627_);
lean_dec(v_a_626_);
lean_dec_ref(v_a_625_);
lean_dec(v_a_624_);
return v_res_630_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_ForwardState_applyGoalDiff_spec__3___redArg___lam__0(lean_object* v_x_631_, lean_object* v___y_632_, lean_object* v___y_633_, lean_object* v___y_634_, lean_object* v___y_635_, lean_object* v___y_636_){
_start:
{
lean_object* v___x_638_; 
lean_inc(v___y_632_);
v___x_638_ = lean_apply_6(v_x_631_, v___y_632_, v___y_633_, v___y_634_, v___y_635_, v___y_636_, lean_box(0));
return v___x_638_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_ForwardState_applyGoalDiff_spec__3___redArg___lam__0___boxed(lean_object* v_x_639_, lean_object* v___y_640_, lean_object* v___y_641_, lean_object* v___y_642_, lean_object* v___y_643_, lean_object* v___y_644_, lean_object* v___y_645_){
_start:
{
lean_object* v_res_646_; 
v_res_646_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_ForwardState_applyGoalDiff_spec__3___redArg___lam__0(v_x_639_, v___y_640_, v___y_641_, v___y_642_, v___y_643_, v___y_644_);
lean_dec(v___y_640_);
return v_res_646_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_ForwardState_applyGoalDiff_spec__3___redArg(lean_object* v_mvarId_647_, lean_object* v_x_648_, lean_object* v___y_649_, lean_object* v___y_650_, lean_object* v___y_651_, lean_object* v___y_652_, lean_object* v___y_653_){
_start:
{
lean_object* v___f_655_; lean_object* v___x_656_; 
lean_inc(v___y_649_);
v___f_655_ = lean_alloc_closure((void*)(lp_aesop_Lean_MVarId_withContext___at___00Aesop_ForwardState_applyGoalDiff_spec__3___redArg___lam__0___boxed), 7, 2);
lean_closure_set(v___f_655_, 0, v_x_648_);
lean_closure_set(v___f_655_, 1, v___y_649_);
v___x_656_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_box(0), v_mvarId_647_, v___f_655_, v___y_650_, v___y_651_, v___y_652_, v___y_653_);
if (lean_obj_tag(v___x_656_) == 0)
{
return v___x_656_;
}
else
{
lean_object* v_a_657_; lean_object* v___x_659_; uint8_t v_isShared_660_; uint8_t v_isSharedCheck_664_; 
v_a_657_ = lean_ctor_get(v___x_656_, 0);
v_isSharedCheck_664_ = !lean_is_exclusive(v___x_656_);
if (v_isSharedCheck_664_ == 0)
{
v___x_659_ = v___x_656_;
v_isShared_660_ = v_isSharedCheck_664_;
goto v_resetjp_658_;
}
else
{
lean_inc(v_a_657_);
lean_dec(v___x_656_);
v___x_659_ = lean_box(0);
v_isShared_660_ = v_isSharedCheck_664_;
goto v_resetjp_658_;
}
v_resetjp_658_:
{
lean_object* v___x_662_; 
if (v_isShared_660_ == 0)
{
v___x_662_ = v___x_659_;
goto v_reusejp_661_;
}
else
{
lean_object* v_reuseFailAlloc_663_; 
v_reuseFailAlloc_663_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_663_, 0, v_a_657_);
v___x_662_ = v_reuseFailAlloc_663_;
goto v_reusejp_661_;
}
v_reusejp_661_:
{
return v___x_662_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_ForwardState_applyGoalDiff_spec__3___redArg___boxed(lean_object* v_mvarId_665_, lean_object* v_x_666_, lean_object* v___y_667_, lean_object* v___y_668_, lean_object* v___y_669_, lean_object* v___y_670_, lean_object* v___y_671_, lean_object* v___y_672_){
_start:
{
lean_object* v_res_673_; 
v_res_673_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_ForwardState_applyGoalDiff_spec__3___redArg(v_mvarId_665_, v_x_666_, v___y_667_, v___y_668_, v___y_669_, v___y_670_, v___y_671_);
lean_dec(v___y_671_);
lean_dec_ref(v___y_670_);
lean_dec(v___y_669_);
lean_dec_ref(v___y_668_);
lean_dec(v___y_667_);
return v_res_673_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_ForwardState_applyGoalDiff_spec__3(lean_object* v_00_u03b1_674_, lean_object* v_mvarId_675_, lean_object* v_x_676_, lean_object* v___y_677_, lean_object* v___y_678_, lean_object* v___y_679_, lean_object* v___y_680_, lean_object* v___y_681_){
_start:
{
lean_object* v___x_683_; 
v___x_683_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_ForwardState_applyGoalDiff_spec__3___redArg(v_mvarId_675_, v_x_676_, v___y_677_, v___y_678_, v___y_679_, v___y_680_, v___y_681_);
return v___x_683_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_ForwardState_applyGoalDiff_spec__3___boxed(lean_object* v_00_u03b1_684_, lean_object* v_mvarId_685_, lean_object* v_x_686_, lean_object* v___y_687_, lean_object* v___y_688_, lean_object* v___y_689_, lean_object* v___y_690_, lean_object* v___y_691_, lean_object* v___y_692_){
_start:
{
lean_object* v_res_693_; 
v_res_693_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_ForwardState_applyGoalDiff_spec__3(v_00_u03b1_684_, v_mvarId_685_, v_x_686_, v___y_687_, v___y_688_, v___y_689_, v___y_690_, v___y_691_);
lean_dec(v___y_691_);
lean_dec_ref(v___y_690_);
lean_dec(v___y_689_);
lean_dec_ref(v___y_688_);
lean_dec(v___y_687_);
return v_res_693_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00Aesop_ForwardState_applyGoalDiff_spec__6(lean_object* v_opts_694_, lean_object* v_opt_695_){
_start:
{
lean_object* v_name_696_; lean_object* v_defValue_697_; lean_object* v_map_698_; lean_object* v___x_699_; 
v_name_696_ = lean_ctor_get(v_opt_695_, 0);
v_defValue_697_ = lean_ctor_get(v_opt_695_, 1);
v_map_698_ = lean_ctor_get(v_opts_694_, 0);
v___x_699_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_698_, v_name_696_);
if (lean_obj_tag(v___x_699_) == 0)
{
lean_inc(v_defValue_697_);
return v_defValue_697_;
}
else
{
lean_object* v_val_700_; 
v_val_700_ = lean_ctor_get(v___x_699_, 0);
lean_inc(v_val_700_);
lean_dec_ref_known(v___x_699_, 1);
if (lean_obj_tag(v_val_700_) == 0)
{
lean_object* v_v_701_; 
v_v_701_ = lean_ctor_get(v_val_700_, 0);
lean_inc_ref(v_v_701_);
lean_dec_ref_known(v_val_700_, 1);
return v_v_701_;
}
else
{
lean_dec(v_val_700_);
lean_inc(v_defValue_697_);
return v_defValue_697_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00Aesop_ForwardState_applyGoalDiff_spec__6___boxed(lean_object* v_opts_702_, lean_object* v_opt_703_){
_start:
{
lean_object* v_res_704_; 
v_res_704_ = lp_aesop_Lean_Option_get___at___00Aesop_ForwardState_applyGoalDiff_spec__6(v_opts_702_, v_opt_703_);
lean_dec_ref(v_opt_703_);
lean_dec_ref(v_opts_702_);
return v_res_704_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00Aesop_ForwardState_applyGoalDiff_spec__1(lean_object* v_x_705_, lean_object* v_x_706_, lean_object* v___y_707_, lean_object* v___y_708_, lean_object* v___y_709_, lean_object* v___y_710_, lean_object* v___y_711_){
_start:
{
if (lean_obj_tag(v_x_706_) == 0)
{
lean_object* v___x_713_; 
v___x_713_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_713_, 0, v_x_705_);
return v___x_713_;
}
else
{
lean_object* v_key_714_; lean_object* v_tail_715_; lean_object* v___x_716_; 
v_key_714_ = lean_ctor_get(v_x_706_, 0);
lean_inc(v_key_714_);
v_tail_715_ = lean_ctor_get(v_x_706_, 2);
lean_inc(v_tail_715_);
lean_dec_ref_known(v_x_706_, 3);
v___x_716_ = lp_aesop___private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp(v_key_714_, v_x_705_, v___y_707_, v___y_708_, v___y_709_, v___y_710_, v___y_711_);
if (lean_obj_tag(v___x_716_) == 0)
{
lean_object* v_a_717_; 
v_a_717_ = lean_ctor_get(v___x_716_, 0);
lean_inc(v_a_717_);
lean_dec_ref_known(v___x_716_, 1);
v_x_705_ = v_a_717_;
v_x_706_ = v_tail_715_;
goto _start;
}
else
{
lean_dec(v_tail_715_);
return v___x_716_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00Aesop_ForwardState_applyGoalDiff_spec__1___boxed(lean_object* v_x_719_, lean_object* v_x_720_, lean_object* v___y_721_, lean_object* v___y_722_, lean_object* v___y_723_, lean_object* v___y_724_, lean_object* v___y_725_, lean_object* v___y_726_){
_start:
{
lean_object* v_res_727_; 
v_res_727_ = lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00Aesop_ForwardState_applyGoalDiff_spec__1(v_x_719_, v_x_720_, v___y_721_, v___y_722_, v___y_723_, v___y_724_, v___y_725_);
lean_dec(v___y_725_);
lean_dec_ref(v___y_724_);
lean_dec(v___y_723_);
lean_dec_ref(v___y_722_);
lean_dec(v___y_721_);
return v_res_727_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardState_applyGoalDiff_spec__2(lean_object* v_as_728_, size_t v_i_729_, size_t v_stop_730_, lean_object* v_b_731_, lean_object* v___y_732_, lean_object* v___y_733_, lean_object* v___y_734_, lean_object* v___y_735_, lean_object* v___y_736_){
_start:
{
uint8_t v___x_738_; 
v___x_738_ = lean_usize_dec_eq(v_i_729_, v_stop_730_);
if (v___x_738_ == 0)
{
lean_object* v___x_739_; lean_object* v___x_740_; 
v___x_739_ = lean_array_uget_borrowed(v_as_728_, v_i_729_);
lean_inc(v___x_739_);
v___x_740_ = lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00Aesop_ForwardState_applyGoalDiff_spec__1(v_b_731_, v___x_739_, v___y_732_, v___y_733_, v___y_734_, v___y_735_, v___y_736_);
if (lean_obj_tag(v___x_740_) == 0)
{
lean_object* v_a_741_; size_t v___x_742_; size_t v___x_743_; 
v_a_741_ = lean_ctor_get(v___x_740_, 0);
lean_inc(v_a_741_);
lean_dec_ref_known(v___x_740_, 1);
v___x_742_ = ((size_t)1ULL);
v___x_743_ = lean_usize_add(v_i_729_, v___x_742_);
v_i_729_ = v___x_743_;
v_b_731_ = v_a_741_;
goto _start;
}
else
{
return v___x_740_;
}
}
else
{
lean_object* v___x_745_; 
v___x_745_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_745_, 0, v_b_731_);
return v___x_745_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardState_applyGoalDiff_spec__2___boxed(lean_object* v_as_746_, lean_object* v_i_747_, lean_object* v_stop_748_, lean_object* v_b_749_, lean_object* v___y_750_, lean_object* v___y_751_, lean_object* v___y_752_, lean_object* v___y_753_, lean_object* v___y_754_, lean_object* v___y_755_){
_start:
{
size_t v_i_boxed_756_; size_t v_stop_boxed_757_; lean_object* v_res_758_; 
v_i_boxed_756_ = lean_unbox_usize(v_i_747_);
lean_dec(v_i_747_);
v_stop_boxed_757_ = lean_unbox_usize(v_stop_748_);
lean_dec(v_stop_748_);
v_res_758_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardState_applyGoalDiff_spec__2(v_as_746_, v_i_boxed_756_, v_stop_boxed_757_, v_b_749_, v___y_750_, v___y_751_, v___y_752_, v___y_753_, v___y_754_);
lean_dec(v___y_754_);
lean_dec_ref(v___y_753_);
lean_dec(v___y_752_);
lean_dec_ref(v___y_751_);
lean_dec(v___y_750_);
lean_dec_ref(v_as_746_);
return v_res_758_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardState_applyGoalDiff___lam__0(lean_object* v_removedFVars_759_, lean_object* v_fs_760_, lean_object* v___y_761_, lean_object* v___y_762_, lean_object* v___y_763_, lean_object* v___y_764_, lean_object* v___y_765_){
_start:
{
lean_object* v_buckets_767_; lean_object* v___x_768_; lean_object* v___x_769_; uint8_t v___x_770_; 
v_buckets_767_ = lean_ctor_get(v_removedFVars_759_, 1);
v___x_768_ = lean_unsigned_to_nat(0u);
v___x_769_ = lean_array_get_size(v_buckets_767_);
v___x_770_ = lean_nat_dec_lt(v___x_768_, v___x_769_);
if (v___x_770_ == 0)
{
lean_object* v___x_771_; 
v___x_771_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_771_, 0, v_fs_760_);
return v___x_771_;
}
else
{
uint8_t v___x_772_; 
v___x_772_ = lean_nat_dec_le(v___x_769_, v___x_769_);
if (v___x_772_ == 0)
{
if (v___x_770_ == 0)
{
lean_object* v___x_773_; 
v___x_773_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_773_, 0, v_fs_760_);
return v___x_773_;
}
else
{
size_t v___x_774_; size_t v___x_775_; lean_object* v___x_776_; 
v___x_774_ = ((size_t)0ULL);
v___x_775_ = lean_usize_of_nat(v___x_769_);
v___x_776_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardState_applyGoalDiff_spec__2(v_buckets_767_, v___x_774_, v___x_775_, v_fs_760_, v___y_761_, v___y_762_, v___y_763_, v___y_764_, v___y_765_);
return v___x_776_;
}
}
else
{
size_t v___x_777_; size_t v___x_778_; lean_object* v___x_779_; 
v___x_777_ = ((size_t)0ULL);
v___x_778_ = lean_usize_of_nat(v___x_769_);
v___x_779_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardState_applyGoalDiff_spec__2(v_buckets_767_, v___x_777_, v___x_778_, v_fs_760_, v___y_761_, v___y_762_, v___y_763_, v___y_764_, v___y_765_);
return v___x_779_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardState_applyGoalDiff___lam__0___boxed(lean_object* v_removedFVars_780_, lean_object* v_fs_781_, lean_object* v___y_782_, lean_object* v___y_783_, lean_object* v___y_784_, lean_object* v___y_785_, lean_object* v___y_786_, lean_object* v___y_787_){
_start:
{
lean_object* v_res_788_; 
v_res_788_ = lp_aesop_Aesop_ForwardState_applyGoalDiff___lam__0(v_removedFVars_780_, v_fs_781_, v___y_782_, v___y_783_, v___y_784_, v___y_785_, v___y_786_);
lean_dec(v___y_786_);
lean_dec_ref(v___y_785_);
lean_dec(v___y_784_);
lean_dec_ref(v___y_783_);
lean_dec(v___y_782_);
lean_dec_ref(v_removedFVars_780_);
return v_res_788_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00Aesop_ForwardState_applyGoalDiff_spec__0(lean_object* v_rs_789_, lean_object* v_x_790_, lean_object* v_x_791_, lean_object* v___y_792_, lean_object* v___y_793_, lean_object* v___y_794_, lean_object* v___y_795_, lean_object* v___y_796_){
_start:
{
if (lean_obj_tag(v_x_791_) == 0)
{
lean_object* v___x_798_; 
lean_dec_ref(v_rs_789_);
v___x_798_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_798_, 0, v_x_790_);
return v___x_798_;
}
else
{
lean_object* v_key_799_; lean_object* v_tail_800_; lean_object* v___x_801_; 
v_key_799_ = lean_ctor_get(v_x_791_, 0);
lean_inc(v_key_799_);
v_tail_800_ = lean_ctor_get(v_x_791_, 2);
lean_inc(v_tail_800_);
lean_dec_ref_known(v_x_791_, 3);
lean_inc_ref(v_rs_789_);
v___x_801_ = lp_aesop___private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_addHyp(v_rs_789_, v_key_799_, v_x_790_, v___y_792_, v___y_793_, v___y_794_, v___y_795_, v___y_796_);
if (lean_obj_tag(v___x_801_) == 0)
{
lean_object* v_a_802_; 
v_a_802_ = lean_ctor_get(v___x_801_, 0);
lean_inc(v_a_802_);
lean_dec_ref_known(v___x_801_, 1);
v_x_790_ = v_a_802_;
v_x_791_ = v_tail_800_;
goto _start;
}
else
{
lean_dec(v_tail_800_);
lean_dec_ref(v_rs_789_);
return v___x_801_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00Aesop_ForwardState_applyGoalDiff_spec__0___boxed(lean_object* v_rs_804_, lean_object* v_x_805_, lean_object* v_x_806_, lean_object* v___y_807_, lean_object* v___y_808_, lean_object* v___y_809_, lean_object* v___y_810_, lean_object* v___y_811_, lean_object* v___y_812_){
_start:
{
lean_object* v_res_813_; 
v_res_813_ = lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00Aesop_ForwardState_applyGoalDiff_spec__0(v_rs_804_, v_x_805_, v_x_806_, v___y_807_, v___y_808_, v___y_809_, v___y_810_, v___y_811_);
lean_dec(v___y_811_);
lean_dec_ref(v___y_810_);
lean_dec(v___y_809_);
lean_dec_ref(v___y_808_);
lean_dec(v___y_807_);
return v_res_813_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardState_applyGoalDiff_spec__4(lean_object* v_rs_814_, lean_object* v_as_815_, size_t v_i_816_, size_t v_stop_817_, lean_object* v_b_818_, lean_object* v___y_819_, lean_object* v___y_820_, lean_object* v___y_821_, lean_object* v___y_822_, lean_object* v___y_823_){
_start:
{
uint8_t v___x_825_; 
v___x_825_ = lean_usize_dec_eq(v_i_816_, v_stop_817_);
if (v___x_825_ == 0)
{
lean_object* v___x_826_; lean_object* v___x_827_; 
v___x_826_ = lean_array_uget_borrowed(v_as_815_, v_i_816_);
lean_inc(v___x_826_);
lean_inc_ref(v_rs_814_);
v___x_827_ = lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00Aesop_ForwardState_applyGoalDiff_spec__0(v_rs_814_, v_b_818_, v___x_826_, v___y_819_, v___y_820_, v___y_821_, v___y_822_, v___y_823_);
if (lean_obj_tag(v___x_827_) == 0)
{
lean_object* v_a_828_; size_t v___x_829_; size_t v___x_830_; 
v_a_828_ = lean_ctor_get(v___x_827_, 0);
lean_inc(v_a_828_);
lean_dec_ref_known(v___x_827_, 1);
v___x_829_ = ((size_t)1ULL);
v___x_830_ = lean_usize_add(v_i_816_, v___x_829_);
v_i_816_ = v___x_830_;
v_b_818_ = v_a_828_;
goto _start;
}
else
{
lean_dec_ref(v_rs_814_);
return v___x_827_;
}
}
else
{
lean_object* v___x_832_; 
lean_dec_ref(v_rs_814_);
v___x_832_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_832_, 0, v_b_818_);
return v___x_832_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardState_applyGoalDiff_spec__4___boxed(lean_object* v_rs_833_, lean_object* v_as_834_, lean_object* v_i_835_, lean_object* v_stop_836_, lean_object* v_b_837_, lean_object* v___y_838_, lean_object* v___y_839_, lean_object* v___y_840_, lean_object* v___y_841_, lean_object* v___y_842_, lean_object* v___y_843_){
_start:
{
size_t v_i_boxed_844_; size_t v_stop_boxed_845_; lean_object* v_res_846_; 
v_i_boxed_844_ = lean_unbox_usize(v_i_835_);
lean_dec(v_i_835_);
v_stop_boxed_845_ = lean_unbox_usize(v_stop_836_);
lean_dec(v_stop_836_);
v_res_846_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardState_applyGoalDiff_spec__4(v_rs_833_, v_as_834_, v_i_boxed_844_, v_stop_boxed_845_, v_b_837_, v___y_838_, v___y_839_, v___y_840_, v___y_841_, v___y_842_);
lean_dec(v___y_842_);
lean_dec_ref(v___y_841_);
lean_dec(v___y_840_);
lean_dec_ref(v___y_839_);
lean_dec(v___y_838_);
lean_dec_ref(v_as_834_);
return v_res_846_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardState_applyGoalDiff___lam__1(lean_object* v_addedFVars_847_, uint8_t v_targetChanged_848_, lean_object* v_rs_849_, lean_object* v_diff_850_, lean_object* v_a_851_, lean_object* v___y_852_, lean_object* v___y_853_, lean_object* v___y_854_, lean_object* v___y_855_, lean_object* v___y_856_){
_start:
{
lean_object* v___y_859_; lean_object* v_a_860_; lean_object* v___y_863_; lean_object* v_buckets_865_; lean_object* v___x_866_; lean_object* v___x_867_; uint8_t v___x_868_; 
v_buckets_865_ = lean_ctor_get(v_addedFVars_847_, 1);
v___x_866_ = lean_unsigned_to_nat(0u);
v___x_867_ = lean_array_get_size(v_buckets_865_);
v___x_868_ = lean_nat_dec_lt(v___x_866_, v___x_867_);
if (v___x_868_ == 0)
{
lean_object* v___x_869_; 
lean_inc_ref(v_a_851_);
v___x_869_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_869_, 0, v_a_851_);
v___y_859_ = v___x_869_;
v_a_860_ = v_a_851_;
goto v___jp_858_;
}
else
{
uint8_t v___x_870_; 
v___x_870_ = lean_nat_dec_le(v___x_867_, v___x_867_);
if (v___x_870_ == 0)
{
if (v___x_868_ == 0)
{
lean_object* v___x_871_; 
lean_inc_ref(v_a_851_);
v___x_871_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_871_, 0, v_a_851_);
v___y_859_ = v___x_871_;
v_a_860_ = v_a_851_;
goto v___jp_858_;
}
else
{
size_t v___x_872_; size_t v___x_873_; lean_object* v___x_874_; 
v___x_872_ = ((size_t)0ULL);
v___x_873_ = lean_usize_of_nat(v___x_867_);
lean_inc_ref(v_rs_849_);
v___x_874_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardState_applyGoalDiff_spec__4(v_rs_849_, v_buckets_865_, v___x_872_, v___x_873_, v_a_851_, v___y_852_, v___y_853_, v___y_854_, v___y_855_, v___y_856_);
v___y_863_ = v___x_874_;
goto v___jp_862_;
}
}
else
{
size_t v___x_875_; size_t v___x_876_; lean_object* v___x_877_; 
v___x_875_ = ((size_t)0ULL);
v___x_876_ = lean_usize_of_nat(v___x_867_);
lean_inc_ref(v_rs_849_);
v___x_877_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardState_applyGoalDiff_spec__4(v_rs_849_, v_buckets_865_, v___x_875_, v___x_876_, v_a_851_, v___y_852_, v___y_853_, v___y_854_, v___y_855_, v___y_856_);
v___y_863_ = v___x_877_;
goto v___jp_862_;
}
}
v___jp_858_:
{
if (v_targetChanged_848_ == 0)
{
lean_dec_ref(v_a_860_);
lean_dec_ref(v_diff_850_);
lean_dec_ref(v_rs_849_);
return v___y_859_;
}
else
{
lean_object* v___x_861_; 
lean_dec_ref(v___y_859_);
v___x_861_ = lp_aesop___private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_updateTarget(v_rs_849_, v_diff_850_, v_a_860_, v___y_852_, v___y_853_, v___y_854_, v___y_855_, v___y_856_);
return v___x_861_;
}
}
v___jp_862_:
{
if (lean_obj_tag(v___y_863_) == 0)
{
lean_object* v_a_864_; 
v_a_864_ = lean_ctor_get(v___y_863_, 0);
lean_inc(v_a_864_);
v___y_859_ = v___y_863_;
v_a_860_ = v_a_864_;
goto v___jp_858_;
}
else
{
lean_dec_ref(v_diff_850_);
lean_dec_ref(v_rs_849_);
return v___y_863_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardState_applyGoalDiff___lam__1___boxed(lean_object* v_addedFVars_878_, lean_object* v_targetChanged_879_, lean_object* v_rs_880_, lean_object* v_diff_881_, lean_object* v_a_882_, lean_object* v___y_883_, lean_object* v___y_884_, lean_object* v___y_885_, lean_object* v___y_886_, lean_object* v___y_887_, lean_object* v___y_888_){
_start:
{
uint8_t v_targetChanged_boxed_889_; lean_object* v_res_890_; 
v_targetChanged_boxed_889_ = lean_unbox(v_targetChanged_879_);
v_res_890_ = lp_aesop_Aesop_ForwardState_applyGoalDiff___lam__1(v_addedFVars_878_, v_targetChanged_boxed_889_, v_rs_880_, v_diff_881_, v_a_882_, v___y_883_, v___y_884_, v___y_885_, v___y_886_, v___y_887_);
lean_dec(v___y_887_);
lean_dec_ref(v___y_886_);
lean_dec(v___y_885_);
lean_dec_ref(v___y_884_);
lean_dec(v___y_883_);
lean_dec_ref(v_addedFVars_878_);
return v_res_890_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_ForwardState_applyGoalDiff_spec__5___redArg(lean_object* v_opt_891_, lean_object* v___y_892_){
_start:
{
lean_object* v_options_894_; lean_object* v_option_895_; uint8_t v___x_896_; lean_object* v___x_897_; lean_object* v___x_898_; 
v_options_894_ = lean_ctor_get(v___y_892_, 2);
v_option_895_ = lean_ctor_get(v_opt_891_, 1);
v___x_896_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__1(v_options_894_, v_option_895_);
v___x_897_ = lean_box(v___x_896_);
v___x_898_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_898_, 0, v___x_897_);
return v___x_898_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_ForwardState_applyGoalDiff_spec__5___redArg___boxed(lean_object* v_opt_899_, lean_object* v___y_900_, lean_object* v___y_901_){
_start:
{
lean_object* v_res_902_; 
v_res_902_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_ForwardState_applyGoalDiff_spec__5___redArg(v_opt_899_, v___y_900_);
lean_dec_ref(v___y_900_);
lean_dec_ref(v_opt_899_);
return v_res_902_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardState_applyGoalDiff(lean_object* v_rs_903_, lean_object* v_diff_904_, lean_object* v_fs_905_, lean_object* v_a_906_, lean_object* v_a_907_, lean_object* v_a_908_, lean_object* v_a_909_, lean_object* v_a_910_){
_start:
{
lean_object* v___y_913_; lean_object* v_a_914_; lean_object* v___y_948_; lean_object* v___y_949_; lean_object* v_options_951_; uint8_t v_a_953_; lean_object* v___y_985_; lean_object* v___x_989_; uint8_t v___x_990_; 
v_options_951_ = lean_ctor_get(v_a_909_, 2);
v___x_989_ = lp_aesop_Aesop_aesop_collectStats;
v___x_990_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__1(v_options_951_, v___x_989_);
if (v___x_990_ == 0)
{
lean_object* v___x_991_; lean_object* v___x_992_; lean_object* v_a_993_; uint8_t v___x_994_; 
v___x_991_ = lp_aesop_Aesop_TraceOption_stats;
v___x_992_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_ForwardState_applyGoalDiff_spec__5___redArg(v___x_991_, v_a_909_);
v_a_993_ = lean_ctor_get(v___x_992_, 0);
lean_inc(v_a_993_);
v___x_994_ = lean_unbox(v_a_993_);
if (v___x_994_ == 0)
{
lean_object* v___x_995_; lean_object* v___x_996_; lean_object* v___x_997_; uint8_t v___x_998_; 
lean_dec_ref(v___x_992_);
v___x_995_ = lp_aesop_Aesop_aesop_stats_file;
v___x_996_ = lp_aesop_Lean_Option_get___at___00Aesop_ForwardState_applyGoalDiff_spec__6(v_options_951_, v___x_995_);
v___x_997_ = ((lean_object*)(lp_aesop___private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp___closed__6));
v___x_998_ = lean_string_dec_eq(v___x_996_, v___x_997_);
lean_dec_ref(v___x_996_);
if (v___x_998_ == 0)
{
lean_dec(v_a_993_);
goto v___jp_969_;
}
else
{
uint8_t v___x_999_; 
v___x_999_ = lean_unbox(v_a_993_);
lean_dec(v_a_993_);
v_a_953_ = v___x_999_;
goto v___jp_952_;
}
}
else
{
lean_dec(v_a_993_);
v___y_985_ = v___x_992_;
goto v___jp_984_;
}
}
else
{
goto v___jp_969_;
}
v___jp_912_:
{
lean_object* v___x_915_; lean_object* v___x_916_; lean_object* v_stats_917_; lean_object* v_rulePatternCache_918_; lean_object* v___x_920_; uint8_t v_isShared_921_; uint8_t v_isSharedCheck_946_; 
v___x_915_ = lean_io_mono_nanos_now();
v___x_916_ = lean_st_ref_take(v_a_906_);
v_stats_917_ = lean_ctor_get(v___x_916_, 1);
v_rulePatternCache_918_ = lean_ctor_get(v___x_916_, 0);
v_isSharedCheck_946_ = !lean_is_exclusive(v___x_916_);
if (v_isSharedCheck_946_ == 0)
{
v___x_920_ = v___x_916_;
v_isShared_921_ = v_isSharedCheck_946_;
goto v_resetjp_919_;
}
else
{
lean_inc(v_stats_917_);
lean_inc(v_rulePatternCache_918_);
lean_dec(v___x_916_);
v___x_920_ = lean_box(0);
v_isShared_921_ = v_isSharedCheck_946_;
goto v_resetjp_919_;
}
v_resetjp_919_:
{
lean_object* v_total_922_; lean_object* v_configParsing_923_; lean_object* v_ruleSetConstruction_924_; lean_object* v_search_925_; lean_object* v_ruleSelection_926_; lean_object* v_script_927_; lean_object* v_forwardState_928_; lean_object* v_scriptGenerated_929_; lean_object* v_ruleStats_930_; lean_object* v_goalStats_931_; lean_object* v___x_933_; uint8_t v_isShared_934_; uint8_t v_isSharedCheck_945_; 
v_total_922_ = lean_ctor_get(v_stats_917_, 0);
v_configParsing_923_ = lean_ctor_get(v_stats_917_, 1);
v_ruleSetConstruction_924_ = lean_ctor_get(v_stats_917_, 2);
v_search_925_ = lean_ctor_get(v_stats_917_, 3);
v_ruleSelection_926_ = lean_ctor_get(v_stats_917_, 4);
v_script_927_ = lean_ctor_get(v_stats_917_, 5);
v_forwardState_928_ = lean_ctor_get(v_stats_917_, 6);
v_scriptGenerated_929_ = lean_ctor_get(v_stats_917_, 7);
v_ruleStats_930_ = lean_ctor_get(v_stats_917_, 8);
v_goalStats_931_ = lean_ctor_get(v_stats_917_, 9);
v_isSharedCheck_945_ = !lean_is_exclusive(v_stats_917_);
if (v_isSharedCheck_945_ == 0)
{
v___x_933_ = v_stats_917_;
v_isShared_934_ = v_isSharedCheck_945_;
goto v_resetjp_932_;
}
else
{
lean_inc(v_goalStats_931_);
lean_inc(v_ruleStats_930_);
lean_inc(v_scriptGenerated_929_);
lean_inc(v_forwardState_928_);
lean_inc(v_script_927_);
lean_inc(v_ruleSelection_926_);
lean_inc(v_search_925_);
lean_inc(v_ruleSetConstruction_924_);
lean_inc(v_configParsing_923_);
lean_inc(v_total_922_);
lean_dec(v_stats_917_);
v___x_933_ = lean_box(0);
v_isShared_934_ = v_isSharedCheck_945_;
goto v_resetjp_932_;
}
v_resetjp_932_:
{
lean_object* v___x_935_; lean_object* v___x_936_; lean_object* v___x_938_; 
v___x_935_ = lean_nat_sub(v___x_915_, v___y_913_);
lean_dec(v___y_913_);
lean_dec(v___x_915_);
v___x_936_ = lean_nat_add(v_forwardState_928_, v___x_935_);
lean_dec(v___x_935_);
lean_dec(v_forwardState_928_);
if (v_isShared_934_ == 0)
{
lean_ctor_set(v___x_933_, 6, v___x_936_);
v___x_938_ = v___x_933_;
goto v_reusejp_937_;
}
else
{
lean_object* v_reuseFailAlloc_944_; 
v_reuseFailAlloc_944_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_944_, 0, v_total_922_);
lean_ctor_set(v_reuseFailAlloc_944_, 1, v_configParsing_923_);
lean_ctor_set(v_reuseFailAlloc_944_, 2, v_ruleSetConstruction_924_);
lean_ctor_set(v_reuseFailAlloc_944_, 3, v_search_925_);
lean_ctor_set(v_reuseFailAlloc_944_, 4, v_ruleSelection_926_);
lean_ctor_set(v_reuseFailAlloc_944_, 5, v_script_927_);
lean_ctor_set(v_reuseFailAlloc_944_, 6, v___x_936_);
lean_ctor_set(v_reuseFailAlloc_944_, 7, v_scriptGenerated_929_);
lean_ctor_set(v_reuseFailAlloc_944_, 8, v_ruleStats_930_);
lean_ctor_set(v_reuseFailAlloc_944_, 9, v_goalStats_931_);
v___x_938_ = v_reuseFailAlloc_944_;
goto v_reusejp_937_;
}
v_reusejp_937_:
{
lean_object* v___x_940_; 
if (v_isShared_921_ == 0)
{
lean_ctor_set(v___x_920_, 1, v___x_938_);
v___x_940_ = v___x_920_;
goto v_reusejp_939_;
}
else
{
lean_object* v_reuseFailAlloc_943_; 
v_reuseFailAlloc_943_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_943_, 0, v_rulePatternCache_918_);
lean_ctor_set(v_reuseFailAlloc_943_, 1, v___x_938_);
v___x_940_ = v_reuseFailAlloc_943_;
goto v_reusejp_939_;
}
v_reusejp_939_:
{
lean_object* v___x_941_; lean_object* v___x_942_; 
v___x_941_ = lean_st_ref_set(v_a_906_, v___x_940_);
v___x_942_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_942_, 0, v_a_914_);
return v___x_942_;
}
}
}
}
}
v___jp_947_:
{
if (lean_obj_tag(v___y_949_) == 0)
{
lean_object* v_a_950_; 
v_a_950_ = lean_ctor_get(v___y_949_, 0);
lean_inc(v_a_950_);
lean_dec_ref_known(v___y_949_, 1);
v___y_913_ = v___y_948_;
v_a_914_ = v_a_950_;
goto v___jp_912_;
}
else
{
lean_dec(v___y_948_);
return v___y_949_;
}
}
v___jp_952_:
{
lean_object* v___x_954_; uint8_t v___x_955_; 
v___x_954_ = lp_aesop_Aesop_aesop_dev_statefulForward;
v___x_955_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__1(v_options_951_, v___x_954_);
if (v___x_955_ == 0)
{
lean_object* v___x_956_; 
lean_dec_ref(v_diff_904_);
lean_dec_ref(v_rs_903_);
v___x_956_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_956_, 0, v_fs_905_);
return v___x_956_;
}
else
{
if (v_a_953_ == 0)
{
lean_object* v_oldGoal_957_; lean_object* v_newGoal_958_; lean_object* v_addedFVars_959_; lean_object* v_removedFVars_960_; uint8_t v_targetChanged_961_; lean_object* v___f_962_; lean_object* v___x_963_; 
v_oldGoal_957_ = lean_ctor_get(v_diff_904_, 0);
v_newGoal_958_ = lean_ctor_get(v_diff_904_, 1);
lean_inc(v_newGoal_958_);
v_addedFVars_959_ = lean_ctor_get(v_diff_904_, 2);
lean_inc_ref(v_addedFVars_959_);
v_removedFVars_960_ = lean_ctor_get(v_diff_904_, 3);
v_targetChanged_961_ = lean_ctor_get_uint8(v_diff_904_, sizeof(void*)*4);
lean_inc_ref(v_removedFVars_960_);
v___f_962_ = lean_alloc_closure((void*)(lp_aesop_Aesop_ForwardState_applyGoalDiff___lam__0___boxed), 8, 2);
lean_closure_set(v___f_962_, 0, v_removedFVars_960_);
lean_closure_set(v___f_962_, 1, v_fs_905_);
lean_inc(v_oldGoal_957_);
v___x_963_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_ForwardState_applyGoalDiff_spec__3___redArg(v_oldGoal_957_, v___f_962_, v_a_906_, v_a_907_, v_a_908_, v_a_909_, v_a_910_);
if (lean_obj_tag(v___x_963_) == 0)
{
lean_object* v_a_964_; lean_object* v___x_965_; lean_object* v___f_966_; lean_object* v___x_967_; 
v_a_964_ = lean_ctor_get(v___x_963_, 0);
lean_inc(v_a_964_);
lean_dec_ref_known(v___x_963_, 1);
v___x_965_ = lean_box(v_targetChanged_961_);
v___f_966_ = lean_alloc_closure((void*)(lp_aesop_Aesop_ForwardState_applyGoalDiff___lam__1___boxed), 11, 5);
lean_closure_set(v___f_966_, 0, v_addedFVars_959_);
lean_closure_set(v___f_966_, 1, v___x_965_);
lean_closure_set(v___f_966_, 2, v_rs_903_);
lean_closure_set(v___f_966_, 3, v_diff_904_);
lean_closure_set(v___f_966_, 4, v_a_964_);
v___x_967_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_ForwardState_applyGoalDiff_spec__3___redArg(v_newGoal_958_, v___f_966_, v_a_906_, v_a_907_, v_a_908_, v_a_909_, v_a_910_);
return v___x_967_;
}
else
{
lean_dec_ref(v_addedFVars_959_);
lean_dec(v_newGoal_958_);
lean_dec_ref(v_diff_904_);
lean_dec_ref(v_rs_903_);
return v___x_963_;
}
}
else
{
lean_object* v___x_968_; 
lean_dec_ref(v_diff_904_);
lean_dec_ref(v_rs_903_);
v___x_968_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_968_, 0, v_fs_905_);
return v___x_968_;
}
}
}
v___jp_969_:
{
lean_object* v___x_970_; lean_object* v___x_971_; uint8_t v___x_972_; 
v___x_970_ = lean_io_mono_nanos_now();
v___x_971_ = lp_aesop_Aesop_aesop_dev_statefulForward;
v___x_972_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Forward_State_ApplyGoalDiff_0__Aesop_ForwardState_applyGoalDiff_eraseHyp_spec__1(v_options_951_, v___x_971_);
if (v___x_972_ == 0)
{
lean_dec_ref(v_diff_904_);
lean_dec_ref(v_rs_903_);
v___y_913_ = v___x_970_;
v_a_914_ = v_fs_905_;
goto v___jp_912_;
}
else
{
lean_object* v_oldGoal_973_; lean_object* v_newGoal_974_; lean_object* v_addedFVars_975_; lean_object* v_removedFVars_976_; uint8_t v_targetChanged_977_; lean_object* v___f_978_; lean_object* v___x_979_; 
v_oldGoal_973_ = lean_ctor_get(v_diff_904_, 0);
v_newGoal_974_ = lean_ctor_get(v_diff_904_, 1);
lean_inc(v_newGoal_974_);
v_addedFVars_975_ = lean_ctor_get(v_diff_904_, 2);
lean_inc_ref(v_addedFVars_975_);
v_removedFVars_976_ = lean_ctor_get(v_diff_904_, 3);
v_targetChanged_977_ = lean_ctor_get_uint8(v_diff_904_, sizeof(void*)*4);
lean_inc_ref(v_removedFVars_976_);
v___f_978_ = lean_alloc_closure((void*)(lp_aesop_Aesop_ForwardState_applyGoalDiff___lam__0___boxed), 8, 2);
lean_closure_set(v___f_978_, 0, v_removedFVars_976_);
lean_closure_set(v___f_978_, 1, v_fs_905_);
lean_inc(v_oldGoal_973_);
v___x_979_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_ForwardState_applyGoalDiff_spec__3___redArg(v_oldGoal_973_, v___f_978_, v_a_906_, v_a_907_, v_a_908_, v_a_909_, v_a_910_);
if (lean_obj_tag(v___x_979_) == 0)
{
lean_object* v_a_980_; lean_object* v___x_981_; lean_object* v___f_982_; lean_object* v___x_983_; 
v_a_980_ = lean_ctor_get(v___x_979_, 0);
lean_inc(v_a_980_);
lean_dec_ref_known(v___x_979_, 1);
v___x_981_ = lean_box(v_targetChanged_977_);
v___f_982_ = lean_alloc_closure((void*)(lp_aesop_Aesop_ForwardState_applyGoalDiff___lam__1___boxed), 11, 5);
lean_closure_set(v___f_982_, 0, v_addedFVars_975_);
lean_closure_set(v___f_982_, 1, v___x_981_);
lean_closure_set(v___f_982_, 2, v_rs_903_);
lean_closure_set(v___f_982_, 3, v_diff_904_);
lean_closure_set(v___f_982_, 4, v_a_980_);
v___x_983_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_ForwardState_applyGoalDiff_spec__3___redArg(v_newGoal_974_, v___f_982_, v_a_906_, v_a_907_, v_a_908_, v_a_909_, v_a_910_);
v___y_948_ = v___x_970_;
v___y_949_ = v___x_983_;
goto v___jp_947_;
}
else
{
lean_dec_ref(v_addedFVars_975_);
lean_dec(v_newGoal_974_);
lean_dec_ref(v_diff_904_);
lean_dec_ref(v_rs_903_);
v___y_948_ = v___x_970_;
v___y_949_ = v___x_979_;
goto v___jp_947_;
}
}
}
v___jp_984_:
{
lean_object* v_a_986_; uint8_t v___x_987_; 
v_a_986_ = lean_ctor_get(v___y_985_, 0);
lean_inc(v_a_986_);
lean_dec_ref(v___y_985_);
v___x_987_ = lean_unbox(v_a_986_);
if (v___x_987_ == 0)
{
uint8_t v___x_988_; 
v___x_988_ = lean_unbox(v_a_986_);
lean_dec(v_a_986_);
v_a_953_ = v___x_988_;
goto v___jp_952_;
}
else
{
lean_dec(v_a_986_);
goto v___jp_969_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardState_applyGoalDiff___boxed(lean_object* v_rs_1000_, lean_object* v_diff_1001_, lean_object* v_fs_1002_, lean_object* v_a_1003_, lean_object* v_a_1004_, lean_object* v_a_1005_, lean_object* v_a_1006_, lean_object* v_a_1007_, lean_object* v_a_1008_){
_start:
{
lean_object* v_res_1009_; 
v_res_1009_ = lp_aesop_Aesop_ForwardState_applyGoalDiff(v_rs_1000_, v_diff_1001_, v_fs_1002_, v_a_1003_, v_a_1004_, v_a_1005_, v_a_1006_, v_a_1007_);
lean_dec(v_a_1007_);
lean_dec_ref(v_a_1006_);
lean_dec(v_a_1005_);
lean_dec_ref(v_a_1004_);
lean_dec(v_a_1003_);
return v_res_1009_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_ForwardState_applyGoalDiff_spec__5(lean_object* v_opt_1010_, lean_object* v___y_1011_, lean_object* v___y_1012_, lean_object* v___y_1013_, lean_object* v___y_1014_, lean_object* v___y_1015_){
_start:
{
lean_object* v___x_1017_; 
v___x_1017_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_ForwardState_applyGoalDiff_spec__5___redArg(v_opt_1010_, v___y_1014_);
return v___x_1017_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_ForwardState_applyGoalDiff_spec__5___boxed(lean_object* v_opt_1018_, lean_object* v___y_1019_, lean_object* v___y_1020_, lean_object* v___y_1021_, lean_object* v___y_1022_, lean_object* v___y_1023_, lean_object* v___y_1024_){
_start:
{
lean_object* v_res_1025_; 
v_res_1025_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_ForwardState_applyGoalDiff_spec__5(v_opt_1018_, v___y_1019_, v___y_1020_, v___y_1021_, v___y_1022_, v___y_1023_);
lean_dec(v___y_1023_);
lean_dec_ref(v___y_1022_);
lean_dec(v___y_1021_);
lean_dec_ref(v___y_1020_);
lean_dec(v___y_1019_);
lean_dec_ref(v_opt_1018_);
return v_res_1025_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Forward_State(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_RuleSet(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_Forward_State_ApplyGoalDiff(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Forward_State(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_RuleSet(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_Forward_State_ApplyGoalDiff(uint8_t builtin) {
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
lean_object* initialize_aesop_Aesop_Forward_State(uint8_t builtin);
lean_object* initialize_aesop_Aesop_RuleSet(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_Forward_State_ApplyGoalDiff(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Forward_State(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_RuleSet(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Forward_State_ApplyGoalDiff(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_Forward_State_ApplyGoalDiff(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_Forward_State_ApplyGoalDiff(builtin);
}
#ifdef __cplusplus
}
#endif
