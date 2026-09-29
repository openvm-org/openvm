// Lean compiler output
// Module: Aesop.Tree.Tracing
// Imports: public import Init public meta import Init public import Aesop.Tree.RunMetaM import Batteries.Lean.Meta.SavedState import Batteries.Data.Array.Basic
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
lean_object* lean_array_get_size(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* lp_aesop_Aesop_RegularRule_name(lean_object*);
double lp_aesop_Aesop_RegularRule_successProbability(lean_object*);
lean_object* lp_aesop_Aesop_Percent_toHumanString(double);
lean_object* lean_string_append(lean_object*, lean_object*);
lean_object* l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(lean_object*, uint8_t);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* lean_st_ref_take(lean_object*);
double lean_float_of_nat(lean_object*);
lean_object* l_Lean_PersistentArray_push___redArg(lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
size_t lean_usize_add(size_t, size_t);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* l_Lean_MessageData_ofExpr(lean_object*);
lean_object* l_Lean_indentD(lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Name_append(lean_object*, lean_object*);
uint8_t l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(lean_object*, lean_object*, lean_object*);
lean_object* lean_io_mono_nanos_now();
double lean_float_div(double, double);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_PersistentArray_toArray___redArg(lean_object*);
size_t lean_array_size(lean_object*);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
extern lean_object* l_Lean_trace_profiler;
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
lean_object* l_Lean_PersistentArray_append___redArg(lean_object*, lean_object*);
double lean_float_sub(double, double);
uint8_t lean_float_decLt(double, double);
extern lean_object* l_Lean_trace_profiler_useHeartbeats;
extern lean_object* l_Lean_trace_profiler_threshold;
lean_object* lean_io_get_num_heartbeats();
extern lean_object* lp_aesop_Aesop_treeImpl;
lean_object* lp_aesop_Aesop_Goal_parentRapp_x3f(lean_object*);
lean_object* l_Lean_Meta_saveState___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Meta_SavedState_restore___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* l_Nat_reprFast(lean_object*);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* lean_usize_to_nat(size_t);
lean_object* l_Lean_MessageData_ofLevel(lean_object*);
lean_object* l_Lean_MessageData_joinSep(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_bracket(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_fvar___override(lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* l_Lean_MessageData_ofList(lean_object*);
lean_object* l_Lean_MessageData_paren(lean_object*);
lean_object* lp_aesop_Aesop_EMap_mapM___at___00Aesop_EMap_map_spec__0___redArg(lean_object*, lean_object*);
lean_object* l_Array_zipIdx___redArg(lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_NodeState_toEmoji(uint8_t);
lean_object* l_Lean_MVarId_getType(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_GoalState_toEmoji(uint8_t);
double lp_aesop_Aesop_Goal_priority(lean_object*);
lean_object* l_Lean_MessageData_ofName(lean_object*);
lean_object* l_Int_repr(lean_object*);
lean_object* lp_batteries_Lean_Meta_SavedState_runMetaM_x27___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_UnsafeQueueEntry_name(lean_object*);
double lp_aesop_Aesop_UnsafeQueueEntry_successProbability(lean_object*);
lean_object* lp_aesop_Aesop_ForwardRuleMatches_size(lean_object*);
extern lean_object* lp_aesop_Aesop_Iteration_none;
lean_object* lp_aesop_Aesop_GoalOrigin_toString(lean_object*);
static const lean_string_object lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_toYesNo___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "no"};
static const lean_object* lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_toYesNo___closed__0 = (const lean_object*)&lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_toYesNo___closed__0_value;
static const lean_string_object lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_toYesNo___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "yes"};
static const lean_object* lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_toYesNo___closed__1 = (const lean_object*)&lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_toYesNo___closed__1_value;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_toYesNo(uint8_t);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_toYesNo___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00__private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00__private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00__private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00__private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00__private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00__private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = " G"};
static const lean_object* lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt___lam__0___closed__0 = (const lean_object*)&lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt___lam__0___closed__0_value;
static lean_once_cell_t lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt___lam__0___closed__1;
static const lean_string_object lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = " ["};
static const lean_object* lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt___lam__0___closed__2 = (const lean_object*)&lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt___lam__0___closed__2_value;
static lean_once_cell_t lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt___lam__0___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt___lam__0___closed__3;
static const lean_string_object lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 6, .m_data = "] ⋯ ⊢ "};
static const lean_object* lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt___lam__0___closed__4 = (const lean_object*)&lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt___lam__0___closed__4_value;
static lean_once_cell_t lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt___lam__0___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt___lam__0___closed__5;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt___lam__0(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_runInMetaState___at___00Aesop_Rapp_runMetaM_x27___at___00Aesop_Goal_runMetaMInParentState_x27___at___00__private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt_spec__2_spec__2_spec__3___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_runInMetaState___at___00Aesop_Rapp_runMetaM_x27___at___00Aesop_Goal_runMetaMInParentState_x27___at___00__private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt_spec__2_spec__2_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_runMetaM_x27___at___00Aesop_Goal_runMetaMInParentState_x27___at___00__private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt_spec__2_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_runMetaM_x27___at___00Aesop_Goal_runMetaMInParentState_x27___at___00__private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt_spec__2_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_runMetaMInParentState_x27___at___00__private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_runMetaMInParentState_x27___at___00__private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_runMetaMInParentState_x27___at___00__private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_runMetaMInParentState_x27___at___00__private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_runInMetaState___at___00Aesop_Rapp_runMetaM_x27___at___00Aesop_Goal_runMetaMInParentState_x27___at___00__private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt_spec__2_spec__2_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_runInMetaState___at___00Aesop_Rapp_runMetaM_x27___at___00Aesop_Goal_runMetaMInParentState_x27___at___00__private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt_spec__2_spec__2_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_runMetaM_x27___at___00Aesop_Goal_runMetaMInParentState_x27___at___00__private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt_spec__2_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_runMetaM_x27___at___00Aesop_Goal_runMetaMInParentState_x27___at___00__private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt_spec__2_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_Goal_withHeadlineTraceNode_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_Goal_withHeadlineTraceNode_spec__0___redArg___closed__0;
static lean_once_cell_t lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_Goal_withHeadlineTraceNode_spec__0___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_Goal_withHeadlineTraceNode_spec__0___redArg___closed__1;
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_Goal_withHeadlineTraceNode_spec__0___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_Goal_withHeadlineTraceNode_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_Goal_withHeadlineTraceNode_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_Goal_withHeadlineTraceNode_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_Option_get___at___00Aesop_Goal_withHeadlineTraceNode_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00Aesop_Goal_withHeadlineTraceNode_spec__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_withHeadlineTraceNode___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_withHeadlineTraceNode___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Goal_withHeadlineTraceNode_spec__2_spec__3___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Goal_withHeadlineTraceNode_spec__2_spec__3___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Goal_withHeadlineTraceNode_spec__2_spec__5(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Goal_withHeadlineTraceNode_spec__2_spec__5___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Goal_withHeadlineTraceNode_spec__2_spec__2_spec__3(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Goal_withHeadlineTraceNode_spec__2_spec__2_spec__3___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Goal_withHeadlineTraceNode_spec__2_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Goal_withHeadlineTraceNode_spec__2_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Goal_withHeadlineTraceNode_spec__2_spec__4___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Goal_withHeadlineTraceNode_spec__2_spec__4___redArg___boxed(lean_object*);
static lean_once_cell_t lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Goal_withHeadlineTraceNode_spec__2___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Goal_withHeadlineTraceNode_spec__2___redArg___closed__0;
static const lean_string_object lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Goal_withHeadlineTraceNode_spec__2___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 54, .m_capacity = 54, .m_length = 53, .m_data = "<exception thrown while producing trace node message>"};
static const lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Goal_withHeadlineTraceNode_spec__2___redArg___closed__1 = (const lean_object*)&lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Goal_withHeadlineTraceNode_spec__2___redArg___closed__1_value;
static lean_once_cell_t lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Goal_withHeadlineTraceNode_spec__2___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Goal_withHeadlineTraceNode_spec__2___redArg___closed__2;
static lean_once_cell_t lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Goal_withHeadlineTraceNode_spec__2___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Goal_withHeadlineTraceNode_spec__2___redArg___closed__3;
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Goal_withHeadlineTraceNode_spec__2___redArg(lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Goal_withHeadlineTraceNode_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_Goal_withHeadlineTraceNode___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_aesop_Aesop_Goal_withHeadlineTraceNode___redArg___closed__0 = (const lean_object*)&lp_aesop_Aesop_Goal_withHeadlineTraceNode___redArg___closed__0_value;
static const lean_string_object lp_aesop_Aesop_Goal_withHeadlineTraceNode___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_aesop_Aesop_Goal_withHeadlineTraceNode___redArg___closed__1 = (const lean_object*)&lp_aesop_Aesop_Goal_withHeadlineTraceNode___redArg___closed__1_value;
static const lean_ctor_object lp_aesop_Aesop_Goal_withHeadlineTraceNode___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Goal_withHeadlineTraceNode___redArg___closed__1_value),LEAN_SCALAR_PTR_LITERAL(212, 145, 141, 177, 67, 149, 127, 197)}};
static const lean_object* lp_aesop_Aesop_Goal_withHeadlineTraceNode___redArg___closed__2 = (const lean_object*)&lp_aesop_Aesop_Goal_withHeadlineTraceNode___redArg___closed__2_value;
static lean_once_cell_t lp_aesop_Aesop_Goal_withHeadlineTraceNode___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_aesop_Aesop_Goal_withHeadlineTraceNode___redArg___closed__3;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_withHeadlineTraceNode___redArg(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_withHeadlineTraceNode___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_withHeadlineTraceNode(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_withHeadlineTraceNode___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Goal_withHeadlineTraceNode_spec__2_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Goal_withHeadlineTraceNode_spec__2_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Goal_withHeadlineTraceNode_spec__2_spec__4(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Goal_withHeadlineTraceNode_spec__2_spec__4___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Goal_withHeadlineTraceNode_spec__2(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Goal_withHeadlineTraceNode_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_aesop_Lean_addTrace___at___00__private_Aesop_Tree_Tracing_0__Aesop_Goal_traceMetadata_trc_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Lean_addTrace___at___00__private_Aesop_Tree_Tracing_0__Aesop_Goal_traceMetadata_trc_spec__0___closed__0 = (const lean_object*)&lp_aesop_Lean_addTrace___at___00__private_Aesop_Tree_Tracing_0__Aesop_Goal_traceMetadata_trc_spec__0___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Lean_addTrace___at___00__private_Aesop_Tree_Tracing_0__Aesop_Goal_traceMetadata_trc_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addTrace___at___00__private_Aesop_Tree_Tracing_0__Aesop_Goal_traceMetadata_trc_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Goal_traceMetadata_trc(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Goal_traceMetadata_trc___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Goal_traceMetadata_trcNode___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Goal_traceMetadata_trcNode___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Tree_Tracing_0__Aesop_Goal_traceMetadata_trcNode_spec__0_spec__0(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Tree_Tracing_0__Aesop_Goal_traceMetadata_trcNode_spec__0_spec__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Tree_Tracing_0__Aesop_Goal_traceMetadata_trcNode_spec__0(lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Tree_Tracing_0__Aesop_Goal_traceMetadata_trcNode_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Goal_traceMetadata_trcNode(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Goal_traceMetadata_trcNode___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "["};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__0 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__0_value;
static lean_once_cell_t lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__1;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "] "};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__2 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__2_value;
static lean_once_cell_t lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__3;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "global"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__4 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__4_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "local"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__5 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__5_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "|"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__6 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__6_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "apply"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__7 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__7_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "cases"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__8 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__8_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "constructors"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__9 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__9_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "destruct"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__10 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__10_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "forward"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__11 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__11_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "simp"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__12 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__12_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "tactic"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__13 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__13_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "unfold"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__14 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__14_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "norm"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__15 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__15_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "safe"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__16 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__16_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "unsafe"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__17 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__17_value;
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6(uint8_t, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_traceMetadata___lam__0(uint8_t, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_traceMetadata___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Aesop_Goal_traceMetadata_spec__1___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 3, .m_data = " ↦ "};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Aesop_Goal_traceMetadata_spec__1___redArg___closed__0 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Aesop_Goal_traceMetadata_spec__1___redArg___closed__0_value;
static lean_once_cell_t lp_aesop___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Aesop_Goal_traceMetadata_spec__1___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Aesop_Goal_traceMetadata_spec__1___redArg___closed__1;
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Aesop_Goal_traceMetadata_spec__1___redArg(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Aesop_Goal_traceMetadata_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Aesop_Goal_traceMetadata_spec__3___redArg(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Aesop_Goal_traceMetadata_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Aesop_Goal_traceMetadata_spec__0_spec__0(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Aesop_Goal_traceMetadata_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_aesop_Array_filterMapM___at___00Aesop_Goal_traceMetadata_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Array_filterMapM___at___00Aesop_Goal_traceMetadata_spec__0___closed__0 = (const lean_object*)&lp_aesop_Array_filterMapM___at___00Aesop_Goal_traceMetadata_spec__0___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Array_filterMapM___at___00Aesop_Goal_traceMetadata_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_filterMapM___at___00Aesop_Goal_traceMetadata_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Aesop_Goal_traceMetadata_spec__2_spec__3(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Aesop_Goal_traceMetadata_spec__2_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_aesop_Array_filterMapM___at___00Aesop_Goal_traceMetadata_spec__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Array_filterMapM___at___00Aesop_Goal_traceMetadata_spec__2___closed__0 = (const lean_object*)&lp_aesop_Array_filterMapM___at___00Aesop_Goal_traceMetadata_spec__2___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Array_filterMapM___at___00Aesop_Goal_traceMetadata_spec__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_filterMapM___at___00Aesop_Goal_traceMetadata_spec__2___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_List_mapTR_loop___at___00Aesop_Goal_traceMetadata_spec__4___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "{"};
static const lean_object* lp_aesop_List_mapTR_loop___at___00Aesop_Goal_traceMetadata_spec__4___closed__0 = (const lean_object*)&lp_aesop_List_mapTR_loop___at___00Aesop_Goal_traceMetadata_spec__4___closed__0_value;
static const lean_string_object lp_aesop_List_mapTR_loop___at___00Aesop_Goal_traceMetadata_spec__4___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ", "};
static const lean_object* lp_aesop_List_mapTR_loop___at___00Aesop_Goal_traceMetadata_spec__4___closed__1 = (const lean_object*)&lp_aesop_List_mapTR_loop___at___00Aesop_Goal_traceMetadata_spec__4___closed__1_value;
static const lean_ctor_object lp_aesop_List_mapTR_loop___at___00Aesop_Goal_traceMetadata_spec__4___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_List_mapTR_loop___at___00Aesop_Goal_traceMetadata_spec__4___closed__1_value)}};
static const lean_object* lp_aesop_List_mapTR_loop___at___00Aesop_Goal_traceMetadata_spec__4___closed__2 = (const lean_object*)&lp_aesop_List_mapTR_loop___at___00Aesop_Goal_traceMetadata_spec__4___closed__2_value;
static lean_once_cell_t lp_aesop_List_mapTR_loop___at___00Aesop_Goal_traceMetadata_spec__4___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_List_mapTR_loop___at___00Aesop_Goal_traceMetadata_spec__4___closed__3;
static const lean_string_object lp_aesop_List_mapTR_loop___at___00Aesop_Goal_traceMetadata_spec__4___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = " | "};
static const lean_object* lp_aesop_List_mapTR_loop___at___00Aesop_Goal_traceMetadata_spec__4___closed__4 = (const lean_object*)&lp_aesop_List_mapTR_loop___at___00Aesop_Goal_traceMetadata_spec__4___closed__4_value;
static const lean_ctor_object lp_aesop_List_mapTR_loop___at___00Aesop_Goal_traceMetadata_spec__4___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_List_mapTR_loop___at___00Aesop_Goal_traceMetadata_spec__4___closed__4_value)}};
static const lean_object* lp_aesop_List_mapTR_loop___at___00Aesop_Goal_traceMetadata_spec__4___closed__5 = (const lean_object*)&lp_aesop_List_mapTR_loop___at___00Aesop_Goal_traceMetadata_spec__4___closed__5_value;
static lean_once_cell_t lp_aesop_List_mapTR_loop___at___00Aesop_Goal_traceMetadata_spec__4___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_List_mapTR_loop___at___00Aesop_Goal_traceMetadata_spec__4___closed__6;
static const lean_string_object lp_aesop_List_mapTR_loop___at___00Aesop_Goal_traceMetadata_spec__4___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "}"};
static const lean_object* lp_aesop_List_mapTR_loop___at___00Aesop_Goal_traceMetadata_spec__4___closed__7 = (const lean_object*)&lp_aesop_List_mapTR_loop___at___00Aesop_Goal_traceMetadata_spec__4___closed__7_value;
LEAN_EXPORT lean_object* lp_aesop_List_mapTR_loop___at___00Aesop_Goal_traceMetadata_spec__4(lean_object*, lean_object*);
static const lean_string_object lp_aesop_List_mapTR_loop___at___00Aesop_Goal_traceMetadata_spec__8___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "slot "};
static const lean_object* lp_aesop_List_mapTR_loop___at___00Aesop_Goal_traceMetadata_spec__8___closed__0 = (const lean_object*)&lp_aesop_List_mapTR_loop___at___00Aesop_Goal_traceMetadata_spec__8___closed__0_value;
static lean_once_cell_t lp_aesop_List_mapTR_loop___at___00Aesop_Goal_traceMetadata_spec__8___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_List_mapTR_loop___at___00Aesop_Goal_traceMetadata_spec__8___closed__1;
static const lean_string_object lp_aesop_List_mapTR_loop___at___00Aesop_Goal_traceMetadata_spec__8___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ": "};
static const lean_object* lp_aesop_List_mapTR_loop___at___00Aesop_Goal_traceMetadata_spec__8___closed__2 = (const lean_object*)&lp_aesop_List_mapTR_loop___at___00Aesop_Goal_traceMetadata_spec__8___closed__2_value;
static lean_once_cell_t lp_aesop_List_mapTR_loop___at___00Aesop_Goal_traceMetadata_spec__8___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_List_mapTR_loop___at___00Aesop_Goal_traceMetadata_spec__8___closed__3;
LEAN_EXPORT lean_object* lp_aesop_List_mapTR_loop___at___00Aesop_Goal_traceMetadata_spec__8(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__15_spec__20_spec__38_spec__48___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__15_spec__20_spec__38_spec__48___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__15_spec__20_spec__38___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__15_spec__20_spec__38_spec__47___redArg(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__15_spec__20_spec__38_spec__47___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__15___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__15___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__15___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_List_mapTR_loop___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__11(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_toList___at___00Lean_PersistentHashSet_toList___at___00__private_Aesop_Forward_State_0__Aesop_ppPHashSet___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__10_spec__11_spec__23___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_Goal_traceMetadata_spec__9_spec__18_spec__25___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_Goal_traceMetadata_spec__9_spec__18_spec__25___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_Goal_traceMetadata_spec__9_spec__18___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_Goal_traceMetadata_spec__9_spec__18_spec__24___redArg(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_Goal_traceMetadata_spec__9_spec__18_spec__24___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_Goal_traceMetadata_spec__9_spec__18___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PersistentHashSet_toList___at___00__private_Aesop_Forward_State_0__Aesop_ppPHashSet___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__10_spec__11_spec__23_spec__31___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PersistentHashSet_toList___at___00__private_Aesop_Forward_State_0__Aesop_ppPHashSet___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__10_spec__11_spec__23_spec__31___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PersistentHashSet_toList___at___00__private_Aesop_Forward_State_0__Aesop_ppPHashSet___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__10_spec__11_spec__23_spec__31___redArg___boxed(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Lean_PersistentHashMap_toList___at___00Lean_PersistentHashSet_toList___at___00__private_Aesop_Forward_State_0__Aesop_ppPHashSet___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__10_spec__11_spec__23___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Lean_PersistentHashMap_toList___at___00Lean_PersistentHashSet_toList___at___00__private_Aesop_Forward_State_0__Aesop_ppPHashSet___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__10_spec__11_spec__23___redArg___lam__0, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Lean_PersistentHashMap_toList___at___00Lean_PersistentHashSet_toList___at___00__private_Aesop_Forward_State_0__Aesop_ppPHashSet___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__10_spec__11_spec__23___redArg___closed__0 = (const lean_object*)&lp_aesop_Lean_PersistentHashMap_toList___at___00Lean_PersistentHashSet_toList___at___00__private_Aesop_Forward_State_0__Aesop_ppPHashSet___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__10_spec__11_spec__23___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_toList___at___00Lean_PersistentHashSet_toList___at___00__private_Aesop_Forward_State_0__Aesop_ppPHashSet___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__10_spec__11_spec__23___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_toList___at___00Lean_PersistentHashSet_toList___at___00__private_Aesop_Forward_State_0__Aesop_ppPHashSet___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__10_spec__11_spec__23___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_List_mapTR_loop___at___00Lean_PersistentHashSet_toList___at___00__private_Aesop_Forward_State_0__Aesop_ppPHashSet___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__10_spec__11_spec__24(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashSet_toList___at___00__private_Aesop_Forward_State_0__Aesop_ppPHashSet___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__10_spec__11(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashSet_toList___at___00__private_Aesop_Forward_State_0__Aesop_ppPHashSet___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__10_spec__11___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_List_mapTR_loop___at___00Aesop_Goal_traceMetadata_spec__10(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Forward_State_0__Aesop_ppPHashSet___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__10(lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Forward_State_0__Aesop_ppPHashSet___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__10___boxed(lean_object*);
static const lean_string_object lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ","};
static const lean_object* lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7___lam__1___closed__0 = (const lean_object*)&lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7___lam__1___closed__0_value;
static const lean_ctor_object lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7___lam__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7___lam__1___closed__0_value)}};
static const lean_object* lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7___lam__1___closed__1 = (const lean_object*)&lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7___lam__1___closed__1_value;
static lean_once_cell_t lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7___lam__1___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7___lam__1___closed__2;
static lean_once_cell_t lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7___lam__1___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7___lam__1___closed__3;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7___lam__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12_spec__14_spec__29_spec__38___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = " => "};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12_spec__14_spec__29_spec__38___closed__0 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12_spec__14_spec__29_spec__38___closed__0_value;
static lean_once_cell_t lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12_spec__14_spec__29_spec__38___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12_spec__14_spec__29_spec__38___closed__1;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12_spec__14_spec__29_spec__38___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = " =>"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12_spec__14_spec__29_spec__38___closed__2 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12_spec__14_spec__29_spec__38___closed__2_value;
static lean_once_cell_t lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12_spec__14_spec__29_spec__38___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12_spec__14_spec__29_spec__38___closed__3;
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12_spec__14_spec__29_spec__38(uint8_t, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12_spec__14_spec__29_spec__38___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12_spec__14_spec__29(uint8_t, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12_spec__14_spec__29___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12_spec__14_spec__28_spec__36_spec__42(uint8_t, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12_spec__14_spec__28_spec__36_spec__42___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12_spec__14_spec__28_spec__36(uint8_t, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12_spec__14_spec__28_spec__36___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12_spec__14_spec__28(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12_spec__14_spec__28_spec__35(lean_object*, uint8_t, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12_spec__14_spec__28_spec__35___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12_spec__14_spec__28___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12_spec__14(uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12_spec__14___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "\n"};
static const lean_object* lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12___closed__0 = (const lean_object*)&lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12___closed__0_value;
static lean_once_cell_t lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12___closed__1;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12(uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7___lam__5(lean_object*, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7___lam__5___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_map___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__13___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_BasicAux_0__Array_mapM_x27_go___at___00Array_mapM_x27___at___00Lean_PersistentHashMap_mapMAux___at___00Lean_PersistentHashMap_mapM___at___00Lean_PersistentHashMap_map___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__13_spec__16_spec__32_spec__42_spec__48___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_BasicAux_0__Array_mapM_x27_go___at___00Array_mapM_x27___at___00Lean_PersistentHashMap_mapMAux___at___00Lean_PersistentHashMap_mapM___at___00Lean_PersistentHashMap_map___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__13_spec__16_spec__32_spec__42_spec__48___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_mapM_x27___at___00Lean_PersistentHashMap_mapMAux___at___00Lean_PersistentHashMap_mapM___at___00Lean_PersistentHashMap_map___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__13_spec__16_spec__32_spec__42___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_mapM_x27___at___00Lean_PersistentHashMap_mapMAux___at___00Lean_PersistentHashMap_mapM___at___00Lean_PersistentHashMap_map___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__13_spec__16_spec__32_spec__42___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_PersistentHashMap_mapMAux___at___00Lean_PersistentHashMap_mapM___at___00Lean_PersistentHashMap_map___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__13_spec__16_spec__32_spec__41___redArg(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_mapMAux___at___00Lean_PersistentHashMap_mapM___at___00Lean_PersistentHashMap_map___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__13_spec__16_spec__32___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_PersistentHashMap_mapMAux___at___00Lean_PersistentHashMap_mapM___at___00Lean_PersistentHashMap_map___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__13_spec__16_spec__32_spec__41___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_map___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__13___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__14___lam__0(uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__14___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__14_spec__18___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__14_spec__18___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__14(uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__14___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7___lam__3(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7___lam__2(lean_object*, lean_object*);
static const lean_closure_object lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7___lam__0, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7___closed__0 = (const lean_object*)&lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7___closed__0_value;
static const lean_closure_object lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7___lam__1___boxed, .m_arity = 3, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7___closed__0_value)} };
static const lean_object* lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7___closed__1 = (const lean_object*)&lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7___closed__1_value;
static const lean_closure_object lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7___lam__2, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7___closed__1_value)} };
static const lean_object* lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7___closed__2 = (const lean_object*)&lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7___closed__2_value;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7(uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_List_mapIdx_go___at___00Aesop_Goal_traceMetadata_spec__11___lam__0(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop_List_mapIdx_go___at___00Aesop_Goal_traceMetadata_spec__11___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_List_mapIdx_go___at___00Aesop_Goal_traceMetadata_spec__11___lam__0, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_List_mapIdx_go___at___00Aesop_Goal_traceMetadata_spec__11___closed__0 = (const lean_object*)&lp_aesop_List_mapIdx_go___at___00Aesop_Goal_traceMetadata_spec__11___closed__0_value;
static const lean_string_object lp_aesop_List_mapIdx_go___at___00Aesop_Goal_traceMetadata_spec__11___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "cluster "};
static const lean_object* lp_aesop_List_mapIdx_go___at___00Aesop_Goal_traceMetadata_spec__11___closed__1 = (const lean_object*)&lp_aesop_List_mapIdx_go___at___00Aesop_Goal_traceMetadata_spec__11___closed__1_value;
static lean_once_cell_t lp_aesop_List_mapIdx_go___at___00Aesop_Goal_traceMetadata_spec__11___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_List_mapIdx_go___at___00Aesop_Goal_traceMetadata_spec__11___closed__2;
static const lean_string_object lp_aesop_List_mapIdx_go___at___00Aesop_Goal_traceMetadata_spec__11___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ":"};
static const lean_object* lp_aesop_List_mapIdx_go___at___00Aesop_Goal_traceMetadata_spec__11___closed__3 = (const lean_object*)&lp_aesop_List_mapIdx_go___at___00Aesop_Goal_traceMetadata_spec__11___closed__3_value;
static lean_once_cell_t lp_aesop_List_mapIdx_go___at___00Aesop_Goal_traceMetadata_spec__11___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_List_mapIdx_go___at___00Aesop_Goal_traceMetadata_spec__11___closed__4;
static const lean_string_object lp_aesop_List_mapIdx_go___at___00Aesop_Goal_traceMetadata_spec__11___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "variables:"};
static const lean_object* lp_aesop_List_mapIdx_go___at___00Aesop_Goal_traceMetadata_spec__11___closed__5 = (const lean_object*)&lp_aesop_List_mapIdx_go___at___00Aesop_Goal_traceMetadata_spec__11___closed__5_value;
static lean_once_cell_t lp_aesop_List_mapIdx_go___at___00Aesop_Goal_traceMetadata_spec__11___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_List_mapIdx_go___at___00Aesop_Goal_traceMetadata_spec__11___closed__6;
static const lean_string_object lp_aesop_List_mapIdx_go___at___00Aesop_Goal_traceMetadata_spec__11___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "\nslot queues:"};
static const lean_object* lp_aesop_List_mapIdx_go___at___00Aesop_Goal_traceMetadata_spec__11___closed__7 = (const lean_object*)&lp_aesop_List_mapIdx_go___at___00Aesop_Goal_traceMetadata_spec__11___closed__7_value;
static lean_once_cell_t lp_aesop_List_mapIdx_go___at___00Aesop_Goal_traceMetadata_spec__11___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_List_mapIdx_go___at___00Aesop_Goal_traceMetadata_spec__11___closed__8;
static const lean_string_object lp_aesop_List_mapIdx_go___at___00Aesop_Goal_traceMetadata_spec__11___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "\ncomplete matches:"};
static const lean_object* lp_aesop_List_mapIdx_go___at___00Aesop_Goal_traceMetadata_spec__11___closed__9 = (const lean_object*)&lp_aesop_List_mapIdx_go___at___00Aesop_Goal_traceMetadata_spec__11___closed__9_value;
static lean_once_cell_t lp_aesop_List_mapIdx_go___at___00Aesop_Goal_traceMetadata_spec__11___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_List_mapIdx_go___at___00Aesop_Goal_traceMetadata_spec__11___closed__10;
LEAN_EXPORT lean_object* lp_aesop_List_mapIdx_go___at___00Aesop_Goal_traceMetadata_spec__11(uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_List_mapIdx_go___at___00Aesop_Goal_traceMetadata_spec__11___boxed(lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_aesop_Aesop_Goal_traceMetadata___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12___closed__0_value)}};
static const lean_object* lp_aesop_Aesop_Goal_traceMetadata___lam__1___closed__0 = (const lean_object*)&lp_aesop_Aesop_Goal_traceMetadata___lam__1___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_Goal_traceMetadata___lam__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Goal_traceMetadata___lam__1___closed__1;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_traceMetadata___lam__1(uint8_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_traceMetadata___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_traceMetadata___lam__2(uint8_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_traceMetadata___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldl___at___00Aesop_Goal_traceMetadata_spec__16___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldl___at___00Aesop_Goal_traceMetadata_spec__16___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldl___at___00Aesop_Goal_traceMetadata_spec__16___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_traceMetadata___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_traceMetadata___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__18(uint8_t, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__18___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_traceMetadata___lam__4(uint8_t, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_traceMetadata___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_WFExtrinsicFix_0__WellFounded_opaqueFix_u2082___at___00Aesop_Goal_traceMetadata_spec__17___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Goal_traceMetadata_spec__14___redArg(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Goal_traceMetadata_spec__14___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_List_mapTR_loop___at___00Aesop_Goal_traceMetadata_spec__13(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_Goal_traceMetadata_spec__5___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_Goal_traceMetadata_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Goal_traceMetadata_spec__12(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Goal_traceMetadata_spec__12___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_List_mapTR_loop___at___00Aesop_Goal_traceMetadata_spec__15(lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_Goal_traceMetadata___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "Failed rules:"};
static const lean_object* lp_aesop_Aesop_Goal_traceMetadata___closed__0 = (const lean_object*)&lp_aesop_Aesop_Goal_traceMetadata___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_Goal_traceMetadata___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Goal_traceMetadata___closed__1;
static const lean_string_object lp_aesop_Aesop_Goal_traceMetadata___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "Failed rules: <none>"};
static const lean_object* lp_aesop_Aesop_Goal_traceMetadata___closed__2 = (const lean_object*)&lp_aesop_Aesop_Goal_traceMetadata___closed__2_value;
static lean_once_cell_t lp_aesop_Aesop_Goal_traceMetadata___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Goal_traceMetadata___closed__3;
static const lean_string_object lp_aesop_Aesop_Goal_traceMetadata___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "ID: "};
static const lean_object* lp_aesop_Aesop_Goal_traceMetadata___closed__4 = (const lean_object*)&lp_aesop_Aesop_Goal_traceMetadata___closed__4_value;
static lean_once_cell_t lp_aesop_Aesop_Goal_traceMetadata___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Goal_traceMetadata___closed__5;
static const lean_string_object lp_aesop_Aesop_Goal_traceMetadata___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = "Pre-normalisation goal ("};
static const lean_object* lp_aesop_Aesop_Goal_traceMetadata___closed__6 = (const lean_object*)&lp_aesop_Aesop_Goal_traceMetadata___closed__6_value;
static lean_once_cell_t lp_aesop_Aesop_Goal_traceMetadata___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Goal_traceMetadata___closed__7;
static const lean_string_object lp_aesop_Aesop_Goal_traceMetadata___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "):"};
static const lean_object* lp_aesop_Aesop_Goal_traceMetadata___closed__8 = (const lean_object*)&lp_aesop_Aesop_Goal_traceMetadata___closed__8_value;
static lean_once_cell_t lp_aesop_Aesop_Goal_traceMetadata___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Goal_traceMetadata___closed__9;
static const lean_string_object lp_aesop_Aesop_Goal_traceMetadata___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 34, .m_capacity = 34, .m_length = 33, .m_data = "Unsafe rule queue: <not selected>"};
static const lean_object* lp_aesop_Aesop_Goal_traceMetadata___closed__10 = (const lean_object*)&lp_aesop_Aesop_Goal_traceMetadata___closed__10_value;
static lean_once_cell_t lp_aesop_Aesop_Goal_traceMetadata___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Goal_traceMetadata___closed__11;
static const lean_string_object lp_aesop_Aesop_Goal_traceMetadata___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "Unsafe rule queue:"};
static const lean_object* lp_aesop_Aesop_Goal_traceMetadata___closed__12 = (const lean_object*)&lp_aesop_Aesop_Goal_traceMetadata___closed__12_value;
static lean_once_cell_t lp_aesop_Aesop_Goal_traceMetadata___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Goal_traceMetadata___closed__13;
static const lean_array_object lp_aesop_Aesop_Goal_traceMetadata___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Aesop_Goal_traceMetadata___closed__14 = (const lean_object*)&lp_aesop_Aesop_Goal_traceMetadata___closed__14_value;
static const lean_string_object lp_aesop_Aesop_Goal_traceMetadata___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 27, .m_capacity = 27, .m_length = 26, .m_data = "Unsafe rule queue: <empty>"};
static const lean_object* lp_aesop_Aesop_Goal_traceMetadata___closed__15 = (const lean_object*)&lp_aesop_Aesop_Goal_traceMetadata___closed__15_value;
static lean_once_cell_t lp_aesop_Aesop_Goal_traceMetadata___closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Goal_traceMetadata___closed__16;
static const lean_string_object lp_aesop_Aesop_Goal_traceMetadata___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "Forward state"};
static const lean_object* lp_aesop_Aesop_Goal_traceMetadata___closed__17 = (const lean_object*)&lp_aesop_Aesop_Goal_traceMetadata___closed__17_value;
static lean_once_cell_t lp_aesop_Aesop_Goal_traceMetadata___closed__18_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Goal_traceMetadata___closed__18;
static const lean_string_object lp_aesop_Aesop_Goal_traceMetadata___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "Forward rule matches: "};
static const lean_object* lp_aesop_Aesop_Goal_traceMetadata___closed__19 = (const lean_object*)&lp_aesop_Aesop_Goal_traceMetadata___closed__19_value;
static lean_once_cell_t lp_aesop_Aesop_Goal_traceMetadata___closed__20_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Goal_traceMetadata___closed__20;
static const lean_string_object lp_aesop_Aesop_Goal_traceMetadata___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "Irrelevant: "};
static const lean_object* lp_aesop_Aesop_Goal_traceMetadata___closed__21 = (const lean_object*)&lp_aesop_Aesop_Goal_traceMetadata___closed__21_value;
static lean_once_cell_t lp_aesop_Aesop_Goal_traceMetadata___closed__22_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Goal_traceMetadata___closed__22;
static const lean_string_object lp_aesop_Aesop_Goal_traceMetadata___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "Forced unprovable: "};
static const lean_object* lp_aesop_Aesop_Goal_traceMetadata___closed__23 = (const lean_object*)&lp_aesop_Aesop_Goal_traceMetadata___closed__23_value;
static lean_once_cell_t lp_aesop_Aesop_Goal_traceMetadata___closed__24_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Goal_traceMetadata___closed__24;
static const lean_string_object lp_aesop_Aesop_Goal_traceMetadata___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "Added in iteration: "};
static const lean_object* lp_aesop_Aesop_Goal_traceMetadata___closed__25 = (const lean_object*)&lp_aesop_Aesop_Goal_traceMetadata___closed__25_value;
static lean_once_cell_t lp_aesop_Aesop_Goal_traceMetadata___closed__26_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Goal_traceMetadata___closed__26;
static const lean_string_object lp_aesop_Aesop_Goal_traceMetadata___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 29, .m_capacity = 29, .m_length = 28, .m_data = "Last expanded in iteration: "};
static const lean_object* lp_aesop_Aesop_Goal_traceMetadata___closed__27 = (const lean_object*)&lp_aesop_Aesop_Goal_traceMetadata___closed__27_value;
static lean_once_cell_t lp_aesop_Aesop_Goal_traceMetadata___closed__28_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Goal_traceMetadata___closed__28;
static const lean_string_object lp_aesop_Aesop_Goal_traceMetadata___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "never"};
static const lean_object* lp_aesop_Aesop_Goal_traceMetadata___closed__29 = (const lean_object*)&lp_aesop_Aesop_Goal_traceMetadata___closed__29_value;
static const lean_string_object lp_aesop_Aesop_Goal_traceMetadata___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "Child rapps:  "};
static const lean_object* lp_aesop_Aesop_Goal_traceMetadata___closed__30 = (const lean_object*)&lp_aesop_Aesop_Goal_traceMetadata___closed__30_value;
static lean_once_cell_t lp_aesop_Aesop_Goal_traceMetadata___closed__31_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Goal_traceMetadata___closed__31;
static const lean_string_object lp_aesop_Aesop_Goal_traceMetadata___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Origin: "};
static const lean_object* lp_aesop_Aesop_Goal_traceMetadata___closed__32 = (const lean_object*)&lp_aesop_Aesop_Goal_traceMetadata___closed__32_value;
static lean_once_cell_t lp_aesop_Aesop_Goal_traceMetadata___closed__33_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Goal_traceMetadata___closed__33;
static const lean_string_object lp_aesop_Aesop_Goal_traceMetadata___closed__34_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Depth: "};
static const lean_object* lp_aesop_Aesop_Goal_traceMetadata___closed__34 = (const lean_object*)&lp_aesop_Aesop_Goal_traceMetadata___closed__34_value;
static lean_once_cell_t lp_aesop_Aesop_Goal_traceMetadata___closed__35_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Goal_traceMetadata___closed__35;
static const lean_string_object lp_aesop_Aesop_Goal_traceMetadata___closed__36_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "State: "};
static const lean_object* lp_aesop_Aesop_Goal_traceMetadata___closed__36 = (const lean_object*)&lp_aesop_Aesop_Goal_traceMetadata___closed__36_value;
static lean_once_cell_t lp_aesop_Aesop_Goal_traceMetadata___closed__37_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Goal_traceMetadata___closed__37;
static const lean_string_object lp_aesop_Aesop_Goal_traceMetadata___closed__38_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = " "};
static const lean_object* lp_aesop_Aesop_Goal_traceMetadata___closed__38 = (const lean_object*)&lp_aesop_Aesop_Goal_traceMetadata___closed__38_value;
static lean_once_cell_t lp_aesop_Aesop_Goal_traceMetadata___closed__39_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Goal_traceMetadata___closed__39;
static const lean_string_object lp_aesop_Aesop_Goal_traceMetadata___closed__40_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "unknown"};
static const lean_object* lp_aesop_Aesop_Goal_traceMetadata___closed__40 = (const lean_object*)&lp_aesop_Aesop_Goal_traceMetadata___closed__40_value;
static const lean_string_object lp_aesop_Aesop_Goal_traceMetadata___closed__41_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = "provenByRuleApplication"};
static const lean_object* lp_aesop_Aesop_Goal_traceMetadata___closed__41 = (const lean_object*)&lp_aesop_Aesop_Goal_traceMetadata___closed__41_value;
static const lean_string_object lp_aesop_Aesop_Goal_traceMetadata___closed__42_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 22, .m_capacity = 22, .m_length = 21, .m_data = "provenByNormalization"};
static const lean_object* lp_aesop_Aesop_Goal_traceMetadata___closed__42 = (const lean_object*)&lp_aesop_Aesop_Goal_traceMetadata___closed__42_value;
static const lean_string_object lp_aesop_Aesop_Goal_traceMetadata___closed__43_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "unprovable"};
static const lean_object* lp_aesop_Aesop_Goal_traceMetadata___closed__43 = (const lean_object*)&lp_aesop_Aesop_Goal_traceMetadata___closed__43_value;
static const lean_string_object lp_aesop_Aesop_Goal_traceMetadata___closed__44_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "Parent rapp:  "};
static const lean_object* lp_aesop_Aesop_Goal_traceMetadata___closed__44 = (const lean_object*)&lp_aesop_Aesop_Goal_traceMetadata___closed__44_value;
static lean_once_cell_t lp_aesop_Aesop_Goal_traceMetadata___closed__45_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Goal_traceMetadata___closed__45;
static const lean_string_object lp_aesop_Aesop_Goal_traceMetadata___closed__46_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "none"};
static const lean_object* lp_aesop_Aesop_Goal_traceMetadata___closed__46 = (const lean_object*)&lp_aesop_Aesop_Goal_traceMetadata___closed__46_value;
static const lean_ctor_object lp_aesop_Aesop_Goal_traceMetadata___closed__47_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Goal_traceMetadata___closed__46_value)}};
static const lean_object* lp_aesop_Aesop_Goal_traceMetadata___closed__47 = (const lean_object*)&lp_aesop_Aesop_Goal_traceMetadata___closed__47_value;
static lean_once_cell_t lp_aesop_Aesop_Goal_traceMetadata___closed__48_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Goal_traceMetadata___closed__48;
static const lean_string_object lp_aesop_Aesop_Goal_traceMetadata___closed__49_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "some ("};
static const lean_object* lp_aesop_Aesop_Goal_traceMetadata___closed__49 = (const lean_object*)&lp_aesop_Aesop_Goal_traceMetadata___closed__49_value;
static const lean_ctor_object lp_aesop_Aesop_Goal_traceMetadata___closed__50_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Goal_traceMetadata___closed__49_value)}};
static const lean_object* lp_aesop_Aesop_Goal_traceMetadata___closed__50 = (const lean_object*)&lp_aesop_Aesop_Goal_traceMetadata___closed__50_value;
static lean_once_cell_t lp_aesop_Aesop_Goal_traceMetadata___closed__51_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Goal_traceMetadata___closed__51;
static const lean_string_object lp_aesop_Aesop_Goal_traceMetadata___closed__52_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ")"};
static const lean_object* lp_aesop_Aesop_Goal_traceMetadata___closed__52 = (const lean_object*)&lp_aesop_Aesop_Goal_traceMetadata___closed__52_value;
static const lean_ctor_object lp_aesop_Aesop_Goal_traceMetadata___closed__53_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Goal_traceMetadata___closed__52_value)}};
static const lean_object* lp_aesop_Aesop_Goal_traceMetadata___closed__53 = (const lean_object*)&lp_aesop_Aesop_Goal_traceMetadata___closed__53_value;
static lean_once_cell_t lp_aesop_Aesop_Goal_traceMetadata___closed__54_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Goal_traceMetadata___closed__54;
static const lean_string_object lp_aesop_Aesop_Goal_traceMetadata___closed__55_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "Metavariables: "};
static const lean_object* lp_aesop_Aesop_Goal_traceMetadata___closed__55 = (const lean_object*)&lp_aesop_Aesop_Goal_traceMetadata___closed__55_value;
static lean_once_cell_t lp_aesop_Aesop_Goal_traceMetadata___closed__56_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Goal_traceMetadata___closed__56;
static const lean_string_object lp_aesop_Aesop_Goal_traceMetadata___closed__57_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 26, .m_capacity = 26, .m_length = 25, .m_data = "Post-normalisation goal ("};
static const lean_object* lp_aesop_Aesop_Goal_traceMetadata___closed__57 = (const lean_object*)&lp_aesop_Aesop_Goal_traceMetadata___closed__57_value;
static lean_once_cell_t lp_aesop_Aesop_Goal_traceMetadata___closed__58_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Goal_traceMetadata___closed__58;
static const lean_string_object lp_aesop_Aesop_Goal_traceMetadata___closed__59_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 47, .m_capacity = 47, .m_length = 46, .m_data = "Post-normalisation goal: <goal not normalised>"};
static const lean_object* lp_aesop_Aesop_Goal_traceMetadata___closed__59 = (const lean_object*)&lp_aesop_Aesop_Goal_traceMetadata___closed__59_value;
static lean_once_cell_t lp_aesop_Aesop_Goal_traceMetadata___closed__60_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Goal_traceMetadata___closed__60;
static const lean_ctor_object lp_aesop_Aesop_Goal_traceMetadata___boxed__const__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*0 + sizeof(size_t)*1, .m_other = 0, .m_tag = 0}, .m_objs = {(lean_object*)(size_t)(0ULL)}};
LEAN_EXPORT const lean_object* lp_aesop_Aesop_Goal_traceMetadata___boxed__const__1 = (const lean_object*)&lp_aesop_Aesop_Goal_traceMetadata___boxed__const__1_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_traceMetadata(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_traceMetadata___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Aesop_Goal_traceMetadata_spec__1(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Aesop_Goal_traceMetadata_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Aesop_Goal_traceMetadata_spec__3(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Aesop_Goal_traceMetadata_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_Goal_traceMetadata_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_Goal_traceMetadata_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Aesop_Goal_traceMetadata_spec__9___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Aesop_Goal_traceMetadata_spec__9___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Aesop_Goal_traceMetadata_spec__9(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Aesop_Goal_traceMetadata_spec__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Goal_traceMetadata_spec__14(size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Goal_traceMetadata_spec__14___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldl___at___00Aesop_Goal_traceMetadata_spec__16(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldl___at___00Aesop_Goal_traceMetadata_spec__16___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_WFExtrinsicFix_0__WellFounded_opaqueFix_u2082___at___00Aesop_Goal_traceMetadata_spec__17(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__9___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__9___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__9(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_map___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__13(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__15(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__15___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_Goal_traceMetadata_spec__9_spec__18(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_Goal_traceMetadata_spec__9_spec__18___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Aesop_Goal_traceMetadata_spec__16_spec__26___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Aesop_Goal_traceMetadata_spec__16_spec__26___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Aesop_Goal_traceMetadata_spec__16_spec__26(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Aesop_Goal_traceMetadata_spec__16_spec__26___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_mapM___at___00Lean_PersistentHashMap_map___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__13_spec__16___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_mapM___at___00Lean_PersistentHashMap_map___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__13_spec__16(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__14_spec__18(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__14_spec__18___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__15_spec__20___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__15_spec__20(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_Goal_traceMetadata_spec__9_spec__18_spec__24(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_Goal_traceMetadata_spec__9_spec__18_spec__24___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_Goal_traceMetadata_spec__9_spec__18_spec__25(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_Goal_traceMetadata_spec__9_spec__18_spec__25___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_toList___at___00Lean_PersistentHashSet_toList___at___00__private_Aesop_Forward_State_0__Aesop_ppPHashSet___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__10_spec__11_spec__23(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_toList___at___00Lean_PersistentHashSet_toList___at___00__private_Aesop_Forward_State_0__Aesop_ppPHashSet___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__10_spec__11_spec__23___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_mapMAux___at___00Lean_PersistentHashMap_mapM___at___00Lean_PersistentHashMap_map___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__13_spec__16_spec__32(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__14_spec__18_spec__35___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__14_spec__18_spec__35(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__15_spec__20_spec__38(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PersistentHashSet_toList___at___00__private_Aesop_Forward_State_0__Aesop_ppPHashSet___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__10_spec__11_spec__23_spec__31(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PersistentHashSet_toList___at___00__private_Aesop_Forward_State_0__Aesop_ppPHashSet___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__10_spec__11_spec__23_spec__31___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_PersistentHashMap_mapMAux___at___00Lean_PersistentHashMap_mapM___at___00Lean_PersistentHashMap_map___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__13_spec__16_spec__32_spec__41(lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_PersistentHashMap_mapMAux___at___00Lean_PersistentHashMap_mapM___at___00Lean_PersistentHashMap_map___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__13_spec__16_spec__32_spec__41___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_mapM_x27___at___00Lean_PersistentHashMap_mapMAux___at___00Lean_PersistentHashMap_mapM___at___00Lean_PersistentHashMap_map___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__13_spec__16_spec__32_spec__42(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_mapM_x27___at___00Lean_PersistentHashMap_mapMAux___at___00Lean_PersistentHashMap_mapM___at___00Lean_PersistentHashMap_map___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__13_spec__16_spec__32_spec__42___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__15_spec__20_spec__38_spec__47(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__15_spec__20_spec__38_spec__47___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__15_spec__20_spec__38_spec__48(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__15_spec__20_spec__38_spec__48___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_BasicAux_0__Array_mapM_x27_go___at___00Array_mapM_x27___at___00Lean_PersistentHashMap_mapMAux___at___00Lean_PersistentHashMap_mapM___at___00Lean_PersistentHashMap_map___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__13_spec__16_spec__32_spec__42_spec__48(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_BasicAux_0__Array_mapM_x27_go___at___00Array_mapM_x27___at___00Lean_PersistentHashMap_mapMAux___at___00Lean_PersistentHashMap_mapM___at___00Lean_PersistentHashMap_map___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__13_spec__16_spec__32_spec__42_spec__48___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_withHeadlineTraceNode___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_withHeadlineTraceNode___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_Rapp_withHeadlineTraceNode___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = " R"};
static const lean_object* lp_aesop_Aesop_Rapp_withHeadlineTraceNode___redArg___closed__0 = (const lean_object*)&lp_aesop_Aesop_Rapp_withHeadlineTraceNode___redArg___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_Rapp_withHeadlineTraceNode___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Rapp_withHeadlineTraceNode___redArg___closed__1;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_withHeadlineTraceNode___redArg(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_withHeadlineTraceNode___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_withHeadlineTraceNode(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_withHeadlineTraceNode___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Rapp_traceMetadata_trc(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Rapp_traceMetadata_trc___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Rapp_traceMetadata_spec__1___redArg(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Rapp_traceMetadata_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Rapp_foldSubgoalsM___at___00Aesop_Rapp_subgoals___at___00Aesop_Rapp_traceMetadata_spec__0_spec__0_spec__1___redArg(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Rapp_foldSubgoalsM___at___00Aesop_Rapp_subgoals___at___00Aesop_Rapp_traceMetadata_spec__0_spec__0_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Rapp_foldSubgoalsM___at___00Aesop_Rapp_subgoals___at___00Aesop_Rapp_traceMetadata_spec__0_spec__0_spec__2___redArg(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Rapp_foldSubgoalsM___at___00Aesop_Rapp_subgoals___at___00Aesop_Rapp_traceMetadata_spec__0_spec__0_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_foldSubgoalsM___at___00Aesop_Rapp_subgoals___at___00Aesop_Rapp_traceMetadata_spec__0_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_foldSubgoalsM___at___00Aesop_Rapp_subgoals___at___00Aesop_Rapp_traceMetadata_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_subgoals___at___00Aesop_Rapp_traceMetadata_spec__0___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_subgoals___at___00Aesop_Rapp_traceMetadata_spec__0___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_Rapp_subgoals___at___00Aesop_Rapp_traceMetadata_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_Rapp_subgoals___at___00Aesop_Rapp_traceMetadata_spec__0___lam__0___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Rapp_subgoals___at___00Aesop_Rapp_traceMetadata_spec__0___closed__0 = (const lean_object*)&lp_aesop_Aesop_Rapp_subgoals___at___00Aesop_Rapp_traceMetadata_spec__0___closed__0_value;
static const lean_array_object lp_aesop_Aesop_Rapp_subgoals___at___00Aesop_Rapp_traceMetadata_spec__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Aesop_Rapp_subgoals___at___00Aesop_Rapp_traceMetadata_spec__0___closed__1 = (const lean_object*)&lp_aesop_Aesop_Rapp_subgoals___at___00Aesop_Rapp_traceMetadata_spec__0___closed__1_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_subgoals___at___00Aesop_Rapp_traceMetadata_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_subgoals___at___00Aesop_Rapp_traceMetadata_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_List_mapTR_loop___at___00Aesop_Rapp_traceMetadata_spec__2(lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_Rapp_traceMetadata___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 27, .m_capacity = 27, .m_length = 26, .m_data = "Introduced metavariables: "};
static const lean_object* lp_aesop_Aesop_Rapp_traceMetadata___closed__0 = (const lean_object*)&lp_aesop_Aesop_Rapp_traceMetadata___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_Rapp_traceMetadata___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Rapp_traceMetadata___closed__1;
static const lean_string_object lp_aesop_Aesop_Rapp_traceMetadata___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 27, .m_capacity = 27, .m_length = 26, .m_data = "Assigned   metavariables: "};
static const lean_object* lp_aesop_Aesop_Rapp_traceMetadata___closed__2 = (const lean_object*)&lp_aesop_Aesop_Rapp_traceMetadata___closed__2_value;
static lean_once_cell_t lp_aesop_Aesop_Rapp_traceMetadata___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Rapp_traceMetadata___closed__3;
static const lean_string_object lp_aesop_Aesop_Rapp_traceMetadata___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "false"};
static const lean_object* lp_aesop_Aesop_Rapp_traceMetadata___closed__4 = (const lean_object*)&lp_aesop_Aesop_Rapp_traceMetadata___closed__4_value;
static const lean_string_object lp_aesop_Aesop_Rapp_traceMetadata___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "true"};
static const lean_object* lp_aesop_Aesop_Rapp_traceMetadata___closed__5 = (const lean_object*)&lp_aesop_Aesop_Rapp_traceMetadata___closed__5_value;
static const lean_string_object lp_aesop_Aesop_Rapp_traceMetadata___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Rule: "};
static const lean_object* lp_aesop_Aesop_Rapp_traceMetadata___closed__6 = (const lean_object*)&lp_aesop_Aesop_Rapp_traceMetadata___closed__6_value;
static lean_once_cell_t lp_aesop_Aesop_Rapp_traceMetadata___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Rapp_traceMetadata___closed__7;
static const lean_string_object lp_aesop_Aesop_Rapp_traceMetadata___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 22, .m_capacity = 22, .m_length = 21, .m_data = "Success probability: "};
static const lean_object* lp_aesop_Aesop_Rapp_traceMetadata___closed__8 = (const lean_object*)&lp_aesop_Aesop_Rapp_traceMetadata___closed__8_value;
static lean_once_cell_t lp_aesop_Aesop_Rapp_traceMetadata___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Rapp_traceMetadata___closed__9;
static const lean_string_object lp_aesop_Aesop_Rapp_traceMetadata___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "Parent goal: "};
static const lean_object* lp_aesop_Aesop_Rapp_traceMetadata___closed__10 = (const lean_object*)&lp_aesop_Aesop_Rapp_traceMetadata___closed__10_value;
static lean_once_cell_t lp_aesop_Aesop_Rapp_traceMetadata___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Rapp_traceMetadata___closed__11;
static const lean_string_object lp_aesop_Aesop_Rapp_traceMetadata___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "Child goals: "};
static const lean_object* lp_aesop_Aesop_Rapp_traceMetadata___closed__12 = (const lean_object*)&lp_aesop_Aesop_Rapp_traceMetadata___closed__12_value;
static lean_once_cell_t lp_aesop_Aesop_Rapp_traceMetadata___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Rapp_traceMetadata___closed__13;
static const lean_string_object lp_aesop_Aesop_Rapp_traceMetadata___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "proven"};
static const lean_object* lp_aesop_Aesop_Rapp_traceMetadata___closed__14 = (const lean_object*)&lp_aesop_Aesop_Rapp_traceMetadata___closed__14_value;
static const lean_string_object lp_aesop_Aesop_Rapp_traceMetadata___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "/"};
static const lean_object* lp_aesop_Aesop_Rapp_traceMetadata___closed__15 = (const lean_object*)&lp_aesop_Aesop_Rapp_traceMetadata___closed__15_value;
static const lean_string_object lp_aesop_Aesop_Rapp_traceMetadata___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "almostSafe"};
static const lean_object* lp_aesop_Aesop_Rapp_traceMetadata___closed__16 = (const lean_object*)&lp_aesop_Aesop_Rapp_traceMetadata___closed__16_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_traceMetadata(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_traceMetadata___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Rapp_traceMetadata_spec__1(size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Rapp_traceMetadata_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_foldSubgoalsM___at___00Aesop_Rapp_subgoals___at___00Aesop_Rapp_traceMetadata_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_foldSubgoalsM___at___00Aesop_Rapp_subgoals___at___00Aesop_Rapp_traceMetadata_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Rapp_foldSubgoalsM___at___00Aesop_Rapp_subgoals___at___00Aesop_Rapp_traceMetadata_spec__0_spec__0_spec__1(lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Rapp_foldSubgoalsM___at___00Aesop_Rapp_subgoals___at___00Aesop_Rapp_traceMetadata_spec__0_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Rapp_foldSubgoalsM___at___00Aesop_Rapp_subgoals___at___00Aesop_Rapp_traceMetadata_spec__0_spec__0_spec__2(lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Rapp_foldSubgoalsM___at___00Aesop_Rapp_subgoals___at___00Aesop_Rapp_traceMetadata_spec__0_spec__0_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_traceTreeCore___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_traceTreeCore___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_traceTreeCore___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_traceTreeCore___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Rapp_forSubgoalsM___at___00Aesop_Rapp_traceTreeCore_spec__2_spec__2(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Rapp_forSubgoalsM___at___00Aesop_Rapp_traceTreeCore_spec__2_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Rapp_forSubgoalsM___at___00Aesop_Rapp_traceTreeCore_spec__2_spec__3(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Rapp_forSubgoalsM___at___00Aesop_Rapp_traceTreeCore_spec__2_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_forSubgoalsM___at___00Aesop_Rapp_traceTreeCore_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_forSubgoalsM___at___00Aesop_Rapp_traceTreeCore_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_traceTreeCore___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_traceTreeCore___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_Rapp_traceTreeCore___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_Goal_traceTreeCore___lam__0___boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Rapp_traceTreeCore___closed__0 = (const lean_object*)&lp_aesop_Aesop_Rapp_traceTreeCore___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_traceTreeCore___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_Rapp_traceTreeCore___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Metadata"};
static const lean_object* lp_aesop_Aesop_Rapp_traceTreeCore___closed__1 = (const lean_object*)&lp_aesop_Aesop_Rapp_traceTreeCore___closed__1_value;
static const lean_ctor_object lp_aesop_Aesop_Rapp_traceTreeCore___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Rapp_traceTreeCore___closed__1_value)}};
static const lean_object* lp_aesop_Aesop_Rapp_traceTreeCore___closed__2 = (const lean_object*)&lp_aesop_Aesop_Rapp_traceTreeCore___closed__2_value;
static lean_once_cell_t lp_aesop_Aesop_Rapp_traceTreeCore___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Rapp_traceTreeCore___closed__3;
static lean_once_cell_t lp_aesop_Aesop_Rapp_traceTreeCore___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Rapp_traceTreeCore___closed__4;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_traceTreeCore(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceTreeCore_spec__0(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_traceTreeCore___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_traceTreeCore___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_traceTreeCore(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_traceTreeCore___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_traceTreeCore___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_traceTreeCore___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceTreeCore_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_traceTree(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_traceTree___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_traceTree(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_traceTree___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_toYesNo(uint8_t v_x_3_){
_start:
{
if (v_x_3_ == 0)
{
lean_object* v___x_4_; 
v___x_4_ = ((lean_object*)(lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_toYesNo___closed__0));
return v___x_4_;
}
else
{
lean_object* v___x_5_; 
v___x_5_ = ((lean_object*)(lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_toYesNo___closed__1));
return v___x_5_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_toYesNo___boxed(lean_object* v_x_6_){
_start:
{
uint8_t v_x_22__boxed_7_; lean_object* v_res_8_; 
v_x_22__boxed_7_ = lean_unbox(v_x_6_);
v_res_8_ = lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_toYesNo(v_x_22__boxed_7_);
return v_res_8_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00__private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt_spec__0(lean_object* v_msgData_9_, lean_object* v___y_10_, lean_object* v___y_11_, lean_object* v___y_12_, lean_object* v___y_13_){
_start:
{
lean_object* v___x_15_; lean_object* v_env_16_; lean_object* v___x_17_; lean_object* v_mctx_18_; lean_object* v_lctx_19_; lean_object* v_options_20_; lean_object* v___x_21_; lean_object* v___x_22_; lean_object* v___x_23_; 
v___x_15_ = lean_st_ref_get(v___y_13_);
v_env_16_ = lean_ctor_get(v___x_15_, 0);
lean_inc_ref(v_env_16_);
lean_dec(v___x_15_);
v___x_17_ = lean_st_ref_get(v___y_11_);
v_mctx_18_ = lean_ctor_get(v___x_17_, 0);
lean_inc_ref(v_mctx_18_);
lean_dec(v___x_17_);
v_lctx_19_ = lean_ctor_get(v___y_10_, 2);
v_options_20_ = lean_ctor_get(v___y_12_, 2);
lean_inc_ref(v_options_20_);
lean_inc_ref(v_lctx_19_);
v___x_21_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_21_, 0, v_env_16_);
lean_ctor_set(v___x_21_, 1, v_mctx_18_);
lean_ctor_set(v___x_21_, 2, v_lctx_19_);
lean_ctor_set(v___x_21_, 3, v_options_20_);
v___x_22_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_22_, 0, v___x_21_);
lean_ctor_set(v___x_22_, 1, v_msgData_9_);
v___x_23_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_23_, 0, v___x_22_);
return v___x_23_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00__private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt_spec__0___boxed(lean_object* v_msgData_24_, lean_object* v___y_25_, lean_object* v___y_26_, lean_object* v___y_27_, lean_object* v___y_28_, lean_object* v___y_29_){
_start:
{
lean_object* v_res_30_; 
v_res_30_ = lp_aesop_Lean_addMessageContextFull___at___00__private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt_spec__0(v_msgData_24_, v___y_25_, v___y_26_, v___y_27_, v___y_28_);
lean_dec(v___y_28_);
lean_dec_ref(v___y_27_);
lean_dec(v___y_26_);
lean_dec_ref(v___y_25_);
return v_res_30_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00__private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt_spec__1___redArg(lean_object* v_mvarId_31_, lean_object* v_x_32_, lean_object* v___y_33_, lean_object* v___y_34_, lean_object* v___y_35_, lean_object* v___y_36_){
_start:
{
lean_object* v___x_38_; 
v___x_38_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_box(0), v_mvarId_31_, v_x_32_, v___y_33_, v___y_34_, v___y_35_, v___y_36_);
if (lean_obj_tag(v___x_38_) == 0)
{
lean_object* v_a_39_; lean_object* v___x_41_; uint8_t v_isShared_42_; uint8_t v_isSharedCheck_46_; 
v_a_39_ = lean_ctor_get(v___x_38_, 0);
v_isSharedCheck_46_ = !lean_is_exclusive(v___x_38_);
if (v_isSharedCheck_46_ == 0)
{
v___x_41_ = v___x_38_;
v_isShared_42_ = v_isSharedCheck_46_;
goto v_resetjp_40_;
}
else
{
lean_inc(v_a_39_);
lean_dec(v___x_38_);
v___x_41_ = lean_box(0);
v_isShared_42_ = v_isSharedCheck_46_;
goto v_resetjp_40_;
}
v_resetjp_40_:
{
lean_object* v___x_44_; 
if (v_isShared_42_ == 0)
{
v___x_44_ = v___x_41_;
goto v_reusejp_43_;
}
else
{
lean_object* v_reuseFailAlloc_45_; 
v_reuseFailAlloc_45_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_45_, 0, v_a_39_);
v___x_44_ = v_reuseFailAlloc_45_;
goto v_reusejp_43_;
}
v_reusejp_43_:
{
return v___x_44_;
}
}
}
else
{
lean_object* v_a_47_; lean_object* v___x_49_; uint8_t v_isShared_50_; uint8_t v_isSharedCheck_54_; 
v_a_47_ = lean_ctor_get(v___x_38_, 0);
v_isSharedCheck_54_ = !lean_is_exclusive(v___x_38_);
if (v_isSharedCheck_54_ == 0)
{
v___x_49_ = v___x_38_;
v_isShared_50_ = v_isSharedCheck_54_;
goto v_resetjp_48_;
}
else
{
lean_inc(v_a_47_);
lean_dec(v___x_38_);
v___x_49_ = lean_box(0);
v_isShared_50_ = v_isSharedCheck_54_;
goto v_resetjp_48_;
}
v_resetjp_48_:
{
lean_object* v___x_52_; 
if (v_isShared_50_ == 0)
{
v___x_52_ = v___x_49_;
goto v_reusejp_51_;
}
else
{
lean_object* v_reuseFailAlloc_53_; 
v_reuseFailAlloc_53_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_53_, 0, v_a_47_);
v___x_52_ = v_reuseFailAlloc_53_;
goto v_reusejp_51_;
}
v_reusejp_51_:
{
return v___x_52_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00__private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt_spec__1___redArg___boxed(lean_object* v_mvarId_55_, lean_object* v_x_56_, lean_object* v___y_57_, lean_object* v___y_58_, lean_object* v___y_59_, lean_object* v___y_60_, lean_object* v___y_61_){
_start:
{
lean_object* v_res_62_; 
v_res_62_ = lp_aesop_Lean_MVarId_withContext___at___00__private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt_spec__1___redArg(v_mvarId_55_, v_x_56_, v___y_57_, v___y_58_, v___y_59_, v___y_60_);
lean_dec(v___y_60_);
lean_dec_ref(v___y_59_);
lean_dec(v___y_58_);
lean_dec_ref(v___y_57_);
return v_res_62_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00__private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt_spec__1(lean_object* v_00_u03b1_63_, lean_object* v_mvarId_64_, lean_object* v_x_65_, lean_object* v___y_66_, lean_object* v___y_67_, lean_object* v___y_68_, lean_object* v___y_69_){
_start:
{
lean_object* v___x_71_; 
v___x_71_ = lp_aesop_Lean_MVarId_withContext___at___00__private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt_spec__1___redArg(v_mvarId_64_, v_x_65_, v___y_66_, v___y_67_, v___y_68_, v___y_69_);
return v___x_71_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00__private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt_spec__1___boxed(lean_object* v_00_u03b1_72_, lean_object* v_mvarId_73_, lean_object* v_x_74_, lean_object* v___y_75_, lean_object* v___y_76_, lean_object* v___y_77_, lean_object* v___y_78_, lean_object* v___y_79_){
_start:
{
lean_object* v_res_80_; 
v_res_80_ = lp_aesop_Lean_MVarId_withContext___at___00__private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt_spec__1(v_00_u03b1_72_, v_mvarId_73_, v_x_74_, v___y_75_, v___y_76_, v___y_77_, v___y_78_);
lean_dec(v___y_78_);
lean_dec_ref(v___y_77_);
lean_dec(v___y_76_);
lean_dec_ref(v___y_75_);
return v_res_80_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt___lam__0___closed__1(void){
_start:
{
lean_object* v___x_82_; lean_object* v___x_83_; 
v___x_82_ = ((lean_object*)(lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt___lam__0___closed__0));
v___x_83_ = l_Lean_stringToMessageData(v___x_82_);
return v___x_83_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt___lam__0___closed__3(void){
_start:
{
lean_object* v___x_85_; lean_object* v___x_86_; 
v___x_85_ = ((lean_object*)(lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt___lam__0___closed__2));
v___x_86_ = l_Lean_stringToMessageData(v___x_85_);
return v___x_86_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt___lam__0___closed__5(void){
_start:
{
lean_object* v___x_88_; lean_object* v___x_89_; 
v___x_88_ = ((lean_object*)(lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt___lam__0___closed__4));
v___x_89_ = l_Lean_stringToMessageData(v___x_88_);
return v___x_89_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt___lam__0(lean_object* v_preNormGoal_90_, uint8_t v_state_91_, lean_object* v_id_92_, lean_object* v_g_93_, lean_object* v_transform_94_, lean_object* v___y_95_, lean_object* v___y_96_, lean_object* v___y_97_, lean_object* v___y_98_){
_start:
{
lean_object* v___x_100_; 
v___x_100_ = l_Lean_MVarId_getType(v_preNormGoal_90_, v___y_95_, v___y_96_, v___y_97_, v___y_98_);
if (lean_obj_tag(v___x_100_) == 0)
{
lean_object* v_a_101_; lean_object* v___x_102_; lean_object* v___x_103_; lean_object* v___x_104_; lean_object* v___x_105_; lean_object* v___x_106_; lean_object* v___x_107_; lean_object* v___x_108_; lean_object* v___x_109_; lean_object* v___x_110_; lean_object* v___x_111_; double v___x_112_; lean_object* v___x_113_; lean_object* v___x_114_; lean_object* v___x_115_; lean_object* v___x_116_; lean_object* v___x_117_; lean_object* v___x_118_; lean_object* v___x_119_; lean_object* v___x_120_; 
v_a_101_ = lean_ctor_get(v___x_100_, 0);
lean_inc(v_a_101_);
lean_dec_ref_known(v___x_100_, 1);
v___x_102_ = lp_aesop_Aesop_GoalState_toEmoji(v_state_91_);
v___x_103_ = l_Lean_stringToMessageData(v___x_102_);
v___x_104_ = lean_obj_once(&lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt___lam__0___closed__1, &lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt___lam__0___closed__1_once, _init_lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt___lam__0___closed__1);
v___x_105_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_105_, 0, v___x_103_);
lean_ctor_set(v___x_105_, 1, v___x_104_);
v___x_106_ = l_Nat_reprFast(v_id_92_);
v___x_107_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_107_, 0, v___x_106_);
v___x_108_ = l_Lean_MessageData_ofFormat(v___x_107_);
v___x_109_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_109_, 0, v___x_105_);
lean_ctor_set(v___x_109_, 1, v___x_108_);
v___x_110_ = lean_obj_once(&lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt___lam__0___closed__3, &lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt___lam__0___closed__3_once, _init_lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt___lam__0___closed__3);
v___x_111_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_111_, 0, v___x_109_);
lean_ctor_set(v___x_111_, 1, v___x_110_);
v___x_112_ = lp_aesop_Aesop_Goal_priority(v_g_93_);
v___x_113_ = lp_aesop_Aesop_Percent_toHumanString(v___x_112_);
v___x_114_ = l_Lean_stringToMessageData(v___x_113_);
v___x_115_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_115_, 0, v___x_111_);
lean_ctor_set(v___x_115_, 1, v___x_114_);
v___x_116_ = lean_obj_once(&lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt___lam__0___closed__5, &lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt___lam__0___closed__5_once, _init_lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt___lam__0___closed__5);
v___x_117_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_117_, 0, v___x_115_);
lean_ctor_set(v___x_117_, 1, v___x_116_);
v___x_118_ = l_Lean_MessageData_ofExpr(v_a_101_);
v___x_119_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_119_, 0, v___x_117_);
lean_ctor_set(v___x_119_, 1, v___x_118_);
lean_inc(v___y_98_);
lean_inc_ref(v___y_97_);
lean_inc(v___y_96_);
lean_inc_ref(v___y_95_);
v___x_120_ = lean_apply_6(v_transform_94_, v___x_119_, v___y_95_, v___y_96_, v___y_97_, v___y_98_, lean_box(0));
if (lean_obj_tag(v___x_120_) == 0)
{
lean_object* v_a_121_; lean_object* v___x_122_; 
v_a_121_ = lean_ctor_get(v___x_120_, 0);
lean_inc(v_a_121_);
lean_dec_ref_known(v___x_120_, 1);
v___x_122_ = lp_aesop_Lean_addMessageContextFull___at___00__private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt_spec__0(v_a_121_, v___y_95_, v___y_96_, v___y_97_, v___y_98_);
lean_dec(v___y_98_);
lean_dec_ref(v___y_97_);
lean_dec(v___y_96_);
lean_dec_ref(v___y_95_);
return v___x_122_;
}
else
{
lean_dec(v___y_98_);
lean_dec_ref(v___y_97_);
lean_dec(v___y_96_);
lean_dec_ref(v___y_95_);
return v___x_120_;
}
}
else
{
lean_object* v_a_123_; lean_object* v___x_125_; uint8_t v_isShared_126_; uint8_t v_isSharedCheck_130_; 
lean_dec(v___y_98_);
lean_dec_ref(v___y_97_);
lean_dec(v___y_96_);
lean_dec_ref(v___y_95_);
lean_dec_ref(v_transform_94_);
lean_dec(v_g_93_);
lean_dec(v_id_92_);
v_a_123_ = lean_ctor_get(v___x_100_, 0);
v_isSharedCheck_130_ = !lean_is_exclusive(v___x_100_);
if (v_isSharedCheck_130_ == 0)
{
v___x_125_ = v___x_100_;
v_isShared_126_ = v_isSharedCheck_130_;
goto v_resetjp_124_;
}
else
{
lean_inc(v_a_123_);
lean_dec(v___x_100_);
v___x_125_ = lean_box(0);
v_isShared_126_ = v_isSharedCheck_130_;
goto v_resetjp_124_;
}
v_resetjp_124_:
{
lean_object* v___x_128_; 
if (v_isShared_126_ == 0)
{
v___x_128_ = v___x_125_;
goto v_reusejp_127_;
}
else
{
lean_object* v_reuseFailAlloc_129_; 
v_reuseFailAlloc_129_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_129_, 0, v_a_123_);
v___x_128_ = v_reuseFailAlloc_129_;
goto v_reusejp_127_;
}
v_reusejp_127_:
{
return v___x_128_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt___lam__0___boxed(lean_object* v_preNormGoal_131_, lean_object* v_state_132_, lean_object* v_id_133_, lean_object* v_g_134_, lean_object* v_transform_135_, lean_object* v___y_136_, lean_object* v___y_137_, lean_object* v___y_138_, lean_object* v___y_139_, lean_object* v___y_140_){
_start:
{
uint8_t v_state_boxed_141_; lean_object* v_res_142_; 
v_state_boxed_141_ = lean_unbox(v_state_132_);
v_res_142_ = lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt___lam__0(v_preNormGoal_131_, v_state_boxed_141_, v_id_133_, v_g_134_, v_transform_135_, v___y_136_, v___y_137_, v___y_138_, v___y_139_);
return v_res_142_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runInMetaState___at___00Aesop_Rapp_runMetaM_x27___at___00Aesop_Goal_runMetaMInParentState_x27___at___00__private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt_spec__2_spec__2_spec__3___redArg(lean_object* v_s_143_, lean_object* v_x_144_, lean_object* v___y_145_, lean_object* v___y_146_, lean_object* v___y_147_, lean_object* v___y_148_){
_start:
{
lean_object* v___x_150_; 
v___x_150_ = l_Lean_Meta_saveState___redArg(v___y_146_, v___y_148_);
if (lean_obj_tag(v___x_150_) == 0)
{
lean_object* v_a_151_; lean_object* v_a_153_; lean_object* v___x_171_; 
v_a_151_ = lean_ctor_get(v___x_150_, 0);
lean_inc(v_a_151_);
lean_dec_ref_known(v___x_150_, 1);
v___x_171_ = l_Lean_Meta_SavedState_restore___redArg(v_s_143_, v___y_146_, v___y_148_);
if (lean_obj_tag(v___x_171_) == 0)
{
lean_object* v___x_172_; 
lean_dec_ref_known(v___x_171_, 1);
lean_inc(v___y_148_);
lean_inc_ref(v___y_147_);
lean_inc(v___y_146_);
lean_inc_ref(v___y_145_);
v___x_172_ = lean_apply_5(v_x_144_, v___y_145_, v___y_146_, v___y_147_, v___y_148_, lean_box(0));
if (lean_obj_tag(v___x_172_) == 0)
{
lean_object* v_a_173_; lean_object* v___x_174_; 
v_a_173_ = lean_ctor_get(v___x_172_, 0);
lean_inc(v_a_173_);
lean_dec_ref_known(v___x_172_, 1);
v___x_174_ = l_Lean_Meta_SavedState_restore___redArg(v_a_151_, v___y_146_, v___y_148_);
lean_dec(v_a_151_);
if (lean_obj_tag(v___x_174_) == 0)
{
lean_object* v___x_176_; uint8_t v_isShared_177_; uint8_t v_isSharedCheck_181_; 
v_isSharedCheck_181_ = !lean_is_exclusive(v___x_174_);
if (v_isSharedCheck_181_ == 0)
{
lean_object* v_unused_182_; 
v_unused_182_ = lean_ctor_get(v___x_174_, 0);
lean_dec(v_unused_182_);
v___x_176_ = v___x_174_;
v_isShared_177_ = v_isSharedCheck_181_;
goto v_resetjp_175_;
}
else
{
lean_dec(v___x_174_);
v___x_176_ = lean_box(0);
v_isShared_177_ = v_isSharedCheck_181_;
goto v_resetjp_175_;
}
v_resetjp_175_:
{
lean_object* v___x_179_; 
if (v_isShared_177_ == 0)
{
lean_ctor_set(v___x_176_, 0, v_a_173_);
v___x_179_ = v___x_176_;
goto v_reusejp_178_;
}
else
{
lean_object* v_reuseFailAlloc_180_; 
v_reuseFailAlloc_180_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_180_, 0, v_a_173_);
v___x_179_ = v_reuseFailAlloc_180_;
goto v_reusejp_178_;
}
v_reusejp_178_:
{
return v___x_179_;
}
}
}
else
{
lean_object* v_a_183_; lean_object* v___x_185_; uint8_t v_isShared_186_; uint8_t v_isSharedCheck_190_; 
lean_dec(v_a_173_);
v_a_183_ = lean_ctor_get(v___x_174_, 0);
v_isSharedCheck_190_ = !lean_is_exclusive(v___x_174_);
if (v_isSharedCheck_190_ == 0)
{
v___x_185_ = v___x_174_;
v_isShared_186_ = v_isSharedCheck_190_;
goto v_resetjp_184_;
}
else
{
lean_inc(v_a_183_);
lean_dec(v___x_174_);
v___x_185_ = lean_box(0);
v_isShared_186_ = v_isSharedCheck_190_;
goto v_resetjp_184_;
}
v_resetjp_184_:
{
lean_object* v___x_188_; 
if (v_isShared_186_ == 0)
{
v___x_188_ = v___x_185_;
goto v_reusejp_187_;
}
else
{
lean_object* v_reuseFailAlloc_189_; 
v_reuseFailAlloc_189_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_189_, 0, v_a_183_);
v___x_188_ = v_reuseFailAlloc_189_;
goto v_reusejp_187_;
}
v_reusejp_187_:
{
return v___x_188_;
}
}
}
}
else
{
lean_object* v_a_191_; 
v_a_191_ = lean_ctor_get(v___x_172_, 0);
lean_inc(v_a_191_);
lean_dec_ref_known(v___x_172_, 1);
v_a_153_ = v_a_191_;
goto v___jp_152_;
}
}
else
{
lean_object* v_a_192_; 
lean_dec_ref(v_x_144_);
v_a_192_ = lean_ctor_get(v___x_171_, 0);
lean_inc(v_a_192_);
lean_dec_ref_known(v___x_171_, 1);
v_a_153_ = v_a_192_;
goto v___jp_152_;
}
v___jp_152_:
{
lean_object* v___x_154_; 
v___x_154_ = l_Lean_Meta_SavedState_restore___redArg(v_a_151_, v___y_146_, v___y_148_);
lean_dec(v_a_151_);
if (lean_obj_tag(v___x_154_) == 0)
{
lean_object* v___x_156_; uint8_t v_isShared_157_; uint8_t v_isSharedCheck_161_; 
v_isSharedCheck_161_ = !lean_is_exclusive(v___x_154_);
if (v_isSharedCheck_161_ == 0)
{
lean_object* v_unused_162_; 
v_unused_162_ = lean_ctor_get(v___x_154_, 0);
lean_dec(v_unused_162_);
v___x_156_ = v___x_154_;
v_isShared_157_ = v_isSharedCheck_161_;
goto v_resetjp_155_;
}
else
{
lean_dec(v___x_154_);
v___x_156_ = lean_box(0);
v_isShared_157_ = v_isSharedCheck_161_;
goto v_resetjp_155_;
}
v_resetjp_155_:
{
lean_object* v___x_159_; 
if (v_isShared_157_ == 0)
{
lean_ctor_set_tag(v___x_156_, 1);
lean_ctor_set(v___x_156_, 0, v_a_153_);
v___x_159_ = v___x_156_;
goto v_reusejp_158_;
}
else
{
lean_object* v_reuseFailAlloc_160_; 
v_reuseFailAlloc_160_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_160_, 0, v_a_153_);
v___x_159_ = v_reuseFailAlloc_160_;
goto v_reusejp_158_;
}
v_reusejp_158_:
{
return v___x_159_;
}
}
}
else
{
lean_object* v_a_163_; lean_object* v___x_165_; uint8_t v_isShared_166_; uint8_t v_isSharedCheck_170_; 
lean_dec_ref(v_a_153_);
v_a_163_ = lean_ctor_get(v___x_154_, 0);
v_isSharedCheck_170_ = !lean_is_exclusive(v___x_154_);
if (v_isSharedCheck_170_ == 0)
{
v___x_165_ = v___x_154_;
v_isShared_166_ = v_isSharedCheck_170_;
goto v_resetjp_164_;
}
else
{
lean_inc(v_a_163_);
lean_dec(v___x_154_);
v___x_165_ = lean_box(0);
v_isShared_166_ = v_isSharedCheck_170_;
goto v_resetjp_164_;
}
v_resetjp_164_:
{
lean_object* v___x_168_; 
if (v_isShared_166_ == 0)
{
v___x_168_ = v___x_165_;
goto v_reusejp_167_;
}
else
{
lean_object* v_reuseFailAlloc_169_; 
v_reuseFailAlloc_169_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_169_, 0, v_a_163_);
v___x_168_ = v_reuseFailAlloc_169_;
goto v_reusejp_167_;
}
v_reusejp_167_:
{
return v___x_168_;
}
}
}
}
}
else
{
lean_object* v_a_193_; lean_object* v___x_195_; uint8_t v_isShared_196_; uint8_t v_isSharedCheck_200_; 
lean_dec_ref(v_x_144_);
v_a_193_ = lean_ctor_get(v___x_150_, 0);
v_isSharedCheck_200_ = !lean_is_exclusive(v___x_150_);
if (v_isSharedCheck_200_ == 0)
{
v___x_195_ = v___x_150_;
v_isShared_196_ = v_isSharedCheck_200_;
goto v_resetjp_194_;
}
else
{
lean_inc(v_a_193_);
lean_dec(v___x_150_);
v___x_195_ = lean_box(0);
v_isShared_196_ = v_isSharedCheck_200_;
goto v_resetjp_194_;
}
v_resetjp_194_:
{
lean_object* v___x_198_; 
if (v_isShared_196_ == 0)
{
v___x_198_ = v___x_195_;
goto v_reusejp_197_;
}
else
{
lean_object* v_reuseFailAlloc_199_; 
v_reuseFailAlloc_199_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_199_, 0, v_a_193_);
v___x_198_ = v_reuseFailAlloc_199_;
goto v_reusejp_197_;
}
v_reusejp_197_:
{
return v___x_198_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runInMetaState___at___00Aesop_Rapp_runMetaM_x27___at___00Aesop_Goal_runMetaMInParentState_x27___at___00__private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt_spec__2_spec__2_spec__3___redArg___boxed(lean_object* v_s_201_, lean_object* v_x_202_, lean_object* v___y_203_, lean_object* v___y_204_, lean_object* v___y_205_, lean_object* v___y_206_, lean_object* v___y_207_){
_start:
{
lean_object* v_res_208_; 
v_res_208_ = lp_aesop_Aesop_runInMetaState___at___00Aesop_Rapp_runMetaM_x27___at___00Aesop_Goal_runMetaMInParentState_x27___at___00__private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt_spec__2_spec__2_spec__3___redArg(v_s_201_, v_x_202_, v___y_203_, v___y_204_, v___y_205_, v___y_206_);
lean_dec(v___y_206_);
lean_dec_ref(v___y_205_);
lean_dec(v___y_204_);
lean_dec_ref(v___y_203_);
lean_dec_ref(v_s_201_);
return v_res_208_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_runMetaM_x27___at___00Aesop_Goal_runMetaMInParentState_x27___at___00__private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt_spec__2_spec__2___redArg(lean_object* v_x_209_, lean_object* v_r_210_, lean_object* v___y_211_, lean_object* v___y_212_, lean_object* v___y_213_, lean_object* v___y_214_){
_start:
{
lean_object* v___x_216_; lean_object* v_elimRapp_217_; lean_object* v___x_218_; lean_object* v_metaState_219_; lean_object* v___x_220_; 
v___x_216_ = lp_aesop_Aesop_treeImpl;
v_elimRapp_217_ = lean_ctor_get(v___x_216_, 3);
lean_inc_ref(v_elimRapp_217_);
v___x_218_ = lean_apply_1(v_elimRapp_217_, v_r_210_);
v_metaState_219_ = lean_ctor_get(v___x_218_, 6);
lean_inc_ref(v_metaState_219_);
lean_dec_ref(v___x_218_);
v___x_220_ = lp_aesop_Aesop_runInMetaState___at___00Aesop_Rapp_runMetaM_x27___at___00Aesop_Goal_runMetaMInParentState_x27___at___00__private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt_spec__2_spec__2_spec__3___redArg(v_metaState_219_, v_x_209_, v___y_211_, v___y_212_, v___y_213_, v___y_214_);
lean_dec_ref(v_metaState_219_);
return v___x_220_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_runMetaM_x27___at___00Aesop_Goal_runMetaMInParentState_x27___at___00__private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt_spec__2_spec__2___redArg___boxed(lean_object* v_x_221_, lean_object* v_r_222_, lean_object* v___y_223_, lean_object* v___y_224_, lean_object* v___y_225_, lean_object* v___y_226_, lean_object* v___y_227_){
_start:
{
lean_object* v_res_228_; 
v_res_228_ = lp_aesop_Aesop_Rapp_runMetaM_x27___at___00Aesop_Goal_runMetaMInParentState_x27___at___00__private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt_spec__2_spec__2___redArg(v_x_221_, v_r_222_, v___y_223_, v___y_224_, v___y_225_, v___y_226_);
lean_dec(v___y_226_);
lean_dec_ref(v___y_225_);
lean_dec(v___y_224_);
lean_dec_ref(v___y_223_);
return v_res_228_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_runMetaMInParentState_x27___at___00__private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt_spec__2___redArg(lean_object* v_x_229_, lean_object* v_g_230_, lean_object* v___y_231_, lean_object* v___y_232_, lean_object* v___y_233_, lean_object* v___y_234_){
_start:
{
lean_object* v___x_236_; 
v___x_236_ = lp_aesop_Aesop_Goal_parentRapp_x3f(v_g_230_);
if (lean_obj_tag(v___x_236_) == 0)
{
lean_object* v___x_237_; 
v___x_237_ = l_Lean_Meta_saveState___redArg(v___y_232_, v___y_234_);
if (lean_obj_tag(v___x_237_) == 0)
{
lean_object* v_a_238_; lean_object* v_r_239_; 
v_a_238_ = lean_ctor_get(v___x_237_, 0);
lean_inc(v_a_238_);
lean_dec_ref_known(v___x_237_, 1);
lean_inc(v___y_234_);
lean_inc_ref(v___y_233_);
lean_inc(v___y_232_);
lean_inc_ref(v___y_231_);
v_r_239_ = lean_apply_5(v_x_229_, v___y_231_, v___y_232_, v___y_233_, v___y_234_, lean_box(0));
if (lean_obj_tag(v_r_239_) == 0)
{
lean_object* v_a_240_; lean_object* v___x_241_; 
v_a_240_ = lean_ctor_get(v_r_239_, 0);
lean_inc(v_a_240_);
lean_dec_ref_known(v_r_239_, 1);
v___x_241_ = l_Lean_Meta_SavedState_restore___redArg(v_a_238_, v___y_232_, v___y_234_);
lean_dec(v_a_238_);
if (lean_obj_tag(v___x_241_) == 0)
{
lean_object* v___x_243_; uint8_t v_isShared_244_; uint8_t v_isSharedCheck_248_; 
v_isSharedCheck_248_ = !lean_is_exclusive(v___x_241_);
if (v_isSharedCheck_248_ == 0)
{
lean_object* v_unused_249_; 
v_unused_249_ = lean_ctor_get(v___x_241_, 0);
lean_dec(v_unused_249_);
v___x_243_ = v___x_241_;
v_isShared_244_ = v_isSharedCheck_248_;
goto v_resetjp_242_;
}
else
{
lean_dec(v___x_241_);
v___x_243_ = lean_box(0);
v_isShared_244_ = v_isSharedCheck_248_;
goto v_resetjp_242_;
}
v_resetjp_242_:
{
lean_object* v___x_246_; 
if (v_isShared_244_ == 0)
{
lean_ctor_set(v___x_243_, 0, v_a_240_);
v___x_246_ = v___x_243_;
goto v_reusejp_245_;
}
else
{
lean_object* v_reuseFailAlloc_247_; 
v_reuseFailAlloc_247_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_247_, 0, v_a_240_);
v___x_246_ = v_reuseFailAlloc_247_;
goto v_reusejp_245_;
}
v_reusejp_245_:
{
return v___x_246_;
}
}
}
else
{
lean_object* v_a_250_; lean_object* v___x_252_; uint8_t v_isShared_253_; uint8_t v_isSharedCheck_257_; 
lean_dec(v_a_240_);
v_a_250_ = lean_ctor_get(v___x_241_, 0);
v_isSharedCheck_257_ = !lean_is_exclusive(v___x_241_);
if (v_isSharedCheck_257_ == 0)
{
v___x_252_ = v___x_241_;
v_isShared_253_ = v_isSharedCheck_257_;
goto v_resetjp_251_;
}
else
{
lean_inc(v_a_250_);
lean_dec(v___x_241_);
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
else
{
lean_object* v_a_258_; lean_object* v___x_259_; 
v_a_258_ = lean_ctor_get(v_r_239_, 0);
lean_inc(v_a_258_);
lean_dec_ref_known(v_r_239_, 1);
v___x_259_ = l_Lean_Meta_SavedState_restore___redArg(v_a_238_, v___y_232_, v___y_234_);
lean_dec(v_a_238_);
if (lean_obj_tag(v___x_259_) == 0)
{
lean_object* v___x_261_; uint8_t v_isShared_262_; uint8_t v_isSharedCheck_266_; 
v_isSharedCheck_266_ = !lean_is_exclusive(v___x_259_);
if (v_isSharedCheck_266_ == 0)
{
lean_object* v_unused_267_; 
v_unused_267_ = lean_ctor_get(v___x_259_, 0);
lean_dec(v_unused_267_);
v___x_261_ = v___x_259_;
v_isShared_262_ = v_isSharedCheck_266_;
goto v_resetjp_260_;
}
else
{
lean_dec(v___x_259_);
v___x_261_ = lean_box(0);
v_isShared_262_ = v_isSharedCheck_266_;
goto v_resetjp_260_;
}
v_resetjp_260_:
{
lean_object* v___x_264_; 
if (v_isShared_262_ == 0)
{
lean_ctor_set_tag(v___x_261_, 1);
lean_ctor_set(v___x_261_, 0, v_a_258_);
v___x_264_ = v___x_261_;
goto v_reusejp_263_;
}
else
{
lean_object* v_reuseFailAlloc_265_; 
v_reuseFailAlloc_265_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_265_, 0, v_a_258_);
v___x_264_ = v_reuseFailAlloc_265_;
goto v_reusejp_263_;
}
v_reusejp_263_:
{
return v___x_264_;
}
}
}
else
{
lean_object* v_a_268_; lean_object* v___x_270_; uint8_t v_isShared_271_; uint8_t v_isSharedCheck_275_; 
lean_dec(v_a_258_);
v_a_268_ = lean_ctor_get(v___x_259_, 0);
v_isSharedCheck_275_ = !lean_is_exclusive(v___x_259_);
if (v_isSharedCheck_275_ == 0)
{
v___x_270_ = v___x_259_;
v_isShared_271_ = v_isSharedCheck_275_;
goto v_resetjp_269_;
}
else
{
lean_inc(v_a_268_);
lean_dec(v___x_259_);
v___x_270_ = lean_box(0);
v_isShared_271_ = v_isSharedCheck_275_;
goto v_resetjp_269_;
}
v_resetjp_269_:
{
lean_object* v___x_273_; 
if (v_isShared_271_ == 0)
{
v___x_273_ = v___x_270_;
goto v_reusejp_272_;
}
else
{
lean_object* v_reuseFailAlloc_274_; 
v_reuseFailAlloc_274_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_274_, 0, v_a_268_);
v___x_273_ = v_reuseFailAlloc_274_;
goto v_reusejp_272_;
}
v_reusejp_272_:
{
return v___x_273_;
}
}
}
}
}
else
{
lean_object* v_a_276_; lean_object* v___x_278_; uint8_t v_isShared_279_; uint8_t v_isSharedCheck_283_; 
lean_dec_ref(v_x_229_);
v_a_276_ = lean_ctor_get(v___x_237_, 0);
v_isSharedCheck_283_ = !lean_is_exclusive(v___x_237_);
if (v_isSharedCheck_283_ == 0)
{
v___x_278_ = v___x_237_;
v_isShared_279_ = v_isSharedCheck_283_;
goto v_resetjp_277_;
}
else
{
lean_inc(v_a_276_);
lean_dec(v___x_237_);
v___x_278_ = lean_box(0);
v_isShared_279_ = v_isSharedCheck_283_;
goto v_resetjp_277_;
}
v_resetjp_277_:
{
lean_object* v___x_281_; 
if (v_isShared_279_ == 0)
{
v___x_281_ = v___x_278_;
goto v_reusejp_280_;
}
else
{
lean_object* v_reuseFailAlloc_282_; 
v_reuseFailAlloc_282_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_282_, 0, v_a_276_);
v___x_281_ = v_reuseFailAlloc_282_;
goto v_reusejp_280_;
}
v_reusejp_280_:
{
return v___x_281_;
}
}
}
}
else
{
lean_object* v_val_284_; lean_object* v___x_285_; lean_object* v___x_286_; 
v_val_284_ = lean_ctor_get(v___x_236_, 0);
lean_inc(v_val_284_);
lean_dec_ref_known(v___x_236_, 1);
v___x_285_ = lean_st_ref_get(v_val_284_);
lean_dec(v_val_284_);
v___x_286_ = lp_aesop_Aesop_Rapp_runMetaM_x27___at___00Aesop_Goal_runMetaMInParentState_x27___at___00__private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt_spec__2_spec__2___redArg(v_x_229_, v___x_285_, v___y_231_, v___y_232_, v___y_233_, v___y_234_);
return v___x_286_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_runMetaMInParentState_x27___at___00__private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt_spec__2___redArg___boxed(lean_object* v_x_287_, lean_object* v_g_288_, lean_object* v___y_289_, lean_object* v___y_290_, lean_object* v___y_291_, lean_object* v___y_292_, lean_object* v___y_293_){
_start:
{
lean_object* v_res_294_; 
v_res_294_ = lp_aesop_Aesop_Goal_runMetaMInParentState_x27___at___00__private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt_spec__2___redArg(v_x_287_, v_g_288_, v___y_289_, v___y_290_, v___y_291_, v___y_292_);
lean_dec(v___y_292_);
lean_dec_ref(v___y_291_);
lean_dec(v___y_290_);
lean_dec_ref(v___y_289_);
return v_res_294_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt(lean_object* v_g_295_, lean_object* v_transform_296_, lean_object* v_a_297_, lean_object* v_a_298_, lean_object* v_a_299_, lean_object* v_a_300_){
_start:
{
lean_object* v___x_302_; lean_object* v_elimGoal_303_; lean_object* v___x_304_; lean_object* v_id_305_; uint8_t v_state_306_; lean_object* v_preNormGoal_307_; lean_object* v___x_308_; lean_object* v___f_309_; lean_object* v___x_310_; lean_object* v___x_311_; 
v___x_302_ = lp_aesop_Aesop_treeImpl;
v_elimGoal_303_ = lean_ctor_get(v___x_302_, 1);
lean_inc_ref(v_elimGoal_303_);
lean_inc_n(v_g_295_, 2);
v___x_304_ = lean_apply_1(v_elimGoal_303_, v_g_295_);
v_id_305_ = lean_ctor_get(v___x_304_, 0);
lean_inc(v_id_305_);
v_state_306_ = lean_ctor_get_uint8(v___x_304_, sizeof(void*)*14 + 8);
v_preNormGoal_307_ = lean_ctor_get(v___x_304_, 5);
lean_inc_n(v_preNormGoal_307_, 2);
lean_dec_ref(v___x_304_);
v___x_308_ = lean_box(v_state_306_);
v___f_309_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt___lam__0___boxed), 10, 5);
lean_closure_set(v___f_309_, 0, v_preNormGoal_307_);
lean_closure_set(v___f_309_, 1, v___x_308_);
lean_closure_set(v___f_309_, 2, v_id_305_);
lean_closure_set(v___f_309_, 3, v_g_295_);
lean_closure_set(v___f_309_, 4, v_transform_296_);
v___x_310_ = lean_alloc_closure((void*)(lp_aesop_Lean_MVarId_withContext___at___00__private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt_spec__1___boxed), 8, 3);
lean_closure_set(v___x_310_, 0, lean_box(0));
lean_closure_set(v___x_310_, 1, v_preNormGoal_307_);
lean_closure_set(v___x_310_, 2, v___f_309_);
v___x_311_ = lp_aesop_Aesop_Goal_runMetaMInParentState_x27___at___00__private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt_spec__2___redArg(v___x_310_, v_g_295_, v_a_297_, v_a_298_, v_a_299_, v_a_300_);
return v___x_311_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt___boxed(lean_object* v_g_312_, lean_object* v_transform_313_, lean_object* v_a_314_, lean_object* v_a_315_, lean_object* v_a_316_, lean_object* v_a_317_, lean_object* v_a_318_){
_start:
{
lean_object* v_res_319_; 
v_res_319_ = lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt(v_g_312_, v_transform_313_, v_a_314_, v_a_315_, v_a_316_, v_a_317_);
lean_dec(v_a_317_);
lean_dec_ref(v_a_316_);
lean_dec(v_a_315_);
lean_dec_ref(v_a_314_);
return v_res_319_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_runMetaMInParentState_x27___at___00__private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt_spec__2(lean_object* v_00_u03b1_320_, lean_object* v_x_321_, lean_object* v_g_322_, lean_object* v___y_323_, lean_object* v___y_324_, lean_object* v___y_325_, lean_object* v___y_326_){
_start:
{
lean_object* v___x_328_; 
v___x_328_ = lp_aesop_Aesop_Goal_runMetaMInParentState_x27___at___00__private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt_spec__2___redArg(v_x_321_, v_g_322_, v___y_323_, v___y_324_, v___y_325_, v___y_326_);
return v___x_328_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_runMetaMInParentState_x27___at___00__private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt_spec__2___boxed(lean_object* v_00_u03b1_329_, lean_object* v_x_330_, lean_object* v_g_331_, lean_object* v___y_332_, lean_object* v___y_333_, lean_object* v___y_334_, lean_object* v___y_335_, lean_object* v___y_336_){
_start:
{
lean_object* v_res_337_; 
v_res_337_ = lp_aesop_Aesop_Goal_runMetaMInParentState_x27___at___00__private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt_spec__2(v_00_u03b1_329_, v_x_330_, v_g_331_, v___y_332_, v___y_333_, v___y_334_, v___y_335_);
lean_dec(v___y_335_);
lean_dec_ref(v___y_334_);
lean_dec(v___y_333_);
lean_dec_ref(v___y_332_);
return v_res_337_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runInMetaState___at___00Aesop_Rapp_runMetaM_x27___at___00Aesop_Goal_runMetaMInParentState_x27___at___00__private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt_spec__2_spec__2_spec__3(lean_object* v_00_u03b1_338_, lean_object* v_s_339_, lean_object* v_x_340_, lean_object* v___y_341_, lean_object* v___y_342_, lean_object* v___y_343_, lean_object* v___y_344_){
_start:
{
lean_object* v___x_346_; 
v___x_346_ = lp_aesop_Aesop_runInMetaState___at___00Aesop_Rapp_runMetaM_x27___at___00Aesop_Goal_runMetaMInParentState_x27___at___00__private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt_spec__2_spec__2_spec__3___redArg(v_s_339_, v_x_340_, v___y_341_, v___y_342_, v___y_343_, v___y_344_);
return v___x_346_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runInMetaState___at___00Aesop_Rapp_runMetaM_x27___at___00Aesop_Goal_runMetaMInParentState_x27___at___00__private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt_spec__2_spec__2_spec__3___boxed(lean_object* v_00_u03b1_347_, lean_object* v_s_348_, lean_object* v_x_349_, lean_object* v___y_350_, lean_object* v___y_351_, lean_object* v___y_352_, lean_object* v___y_353_, lean_object* v___y_354_){
_start:
{
lean_object* v_res_355_; 
v_res_355_ = lp_aesop_Aesop_runInMetaState___at___00Aesop_Rapp_runMetaM_x27___at___00Aesop_Goal_runMetaMInParentState_x27___at___00__private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt_spec__2_spec__2_spec__3(v_00_u03b1_347_, v_s_348_, v_x_349_, v___y_350_, v___y_351_, v___y_352_, v___y_353_);
lean_dec(v___y_353_);
lean_dec_ref(v___y_352_);
lean_dec(v___y_351_);
lean_dec_ref(v___y_350_);
lean_dec_ref(v_s_348_);
return v_res_355_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_runMetaM_x27___at___00Aesop_Goal_runMetaMInParentState_x27___at___00__private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt_spec__2_spec__2(lean_object* v_00_u03b1_356_, lean_object* v_x_357_, lean_object* v_r_358_, lean_object* v___y_359_, lean_object* v___y_360_, lean_object* v___y_361_, lean_object* v___y_362_){
_start:
{
lean_object* v___x_364_; 
v___x_364_ = lp_aesop_Aesop_Rapp_runMetaM_x27___at___00Aesop_Goal_runMetaMInParentState_x27___at___00__private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt_spec__2_spec__2___redArg(v_x_357_, v_r_358_, v___y_359_, v___y_360_, v___y_361_, v___y_362_);
return v___x_364_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_runMetaM_x27___at___00Aesop_Goal_runMetaMInParentState_x27___at___00__private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt_spec__2_spec__2___boxed(lean_object* v_00_u03b1_365_, lean_object* v_x_366_, lean_object* v_r_367_, lean_object* v___y_368_, lean_object* v___y_369_, lean_object* v___y_370_, lean_object* v___y_371_, lean_object* v___y_372_){
_start:
{
lean_object* v_res_373_; 
v_res_373_ = lp_aesop_Aesop_Rapp_runMetaM_x27___at___00Aesop_Goal_runMetaMInParentState_x27___at___00__private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt_spec__2_spec__2(v_00_u03b1_365_, v_x_366_, v_r_367_, v___y_368_, v___y_369_, v___y_370_, v___y_371_);
lean_dec(v___y_371_);
lean_dec_ref(v___y_370_);
lean_dec(v___y_369_);
lean_dec_ref(v___y_368_);
return v_res_373_;
}
}
static lean_object* _init_lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_Goal_withHeadlineTraceNode_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_374_; lean_object* v___x_375_; lean_object* v___x_376_; 
v___x_374_ = lean_unsigned_to_nat(32u);
v___x_375_ = lean_mk_empty_array_with_capacity(v___x_374_);
v___x_376_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_376_, 0, v___x_375_);
return v___x_376_;
}
}
static lean_object* _init_lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_Goal_withHeadlineTraceNode_spec__0___redArg___closed__1(void){
_start:
{
size_t v___x_377_; lean_object* v___x_378_; lean_object* v___x_379_; lean_object* v___x_380_; lean_object* v___x_381_; lean_object* v___x_382_; 
v___x_377_ = ((size_t)5ULL);
v___x_378_ = lean_unsigned_to_nat(0u);
v___x_379_ = lean_unsigned_to_nat(32u);
v___x_380_ = lean_mk_empty_array_with_capacity(v___x_379_);
v___x_381_ = lean_obj_once(&lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_Goal_withHeadlineTraceNode_spec__0___redArg___closed__0, &lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_Goal_withHeadlineTraceNode_spec__0___redArg___closed__0_once, _init_lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_Goal_withHeadlineTraceNode_spec__0___redArg___closed__0);
v___x_382_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_382_, 0, v___x_381_);
lean_ctor_set(v___x_382_, 1, v___x_380_);
lean_ctor_set(v___x_382_, 2, v___x_378_);
lean_ctor_set(v___x_382_, 3, v___x_378_);
lean_ctor_set_usize(v___x_382_, 4, v___x_377_);
return v___x_382_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_Goal_withHeadlineTraceNode_spec__0___redArg(lean_object* v___y_383_){
_start:
{
lean_object* v___x_385_; lean_object* v_traceState_386_; lean_object* v_traces_387_; lean_object* v___x_388_; lean_object* v_traceState_389_; lean_object* v_env_390_; lean_object* v_nextMacroScope_391_; lean_object* v_ngen_392_; lean_object* v_auxDeclNGen_393_; lean_object* v_cache_394_; lean_object* v_messages_395_; lean_object* v_infoState_396_; lean_object* v_snapshotTasks_397_; lean_object* v___x_399_; uint8_t v_isShared_400_; uint8_t v_isSharedCheck_416_; 
v___x_385_ = lean_st_ref_get(v___y_383_);
v_traceState_386_ = lean_ctor_get(v___x_385_, 4);
lean_inc_ref(v_traceState_386_);
lean_dec(v___x_385_);
v_traces_387_ = lean_ctor_get(v_traceState_386_, 0);
lean_inc_ref(v_traces_387_);
lean_dec_ref(v_traceState_386_);
v___x_388_ = lean_st_ref_take(v___y_383_);
v_traceState_389_ = lean_ctor_get(v___x_388_, 4);
v_env_390_ = lean_ctor_get(v___x_388_, 0);
v_nextMacroScope_391_ = lean_ctor_get(v___x_388_, 1);
v_ngen_392_ = lean_ctor_get(v___x_388_, 2);
v_auxDeclNGen_393_ = lean_ctor_get(v___x_388_, 3);
v_cache_394_ = lean_ctor_get(v___x_388_, 5);
v_messages_395_ = lean_ctor_get(v___x_388_, 6);
v_infoState_396_ = lean_ctor_get(v___x_388_, 7);
v_snapshotTasks_397_ = lean_ctor_get(v___x_388_, 8);
v_isSharedCheck_416_ = !lean_is_exclusive(v___x_388_);
if (v_isSharedCheck_416_ == 0)
{
v___x_399_ = v___x_388_;
v_isShared_400_ = v_isSharedCheck_416_;
goto v_resetjp_398_;
}
else
{
lean_inc(v_snapshotTasks_397_);
lean_inc(v_infoState_396_);
lean_inc(v_messages_395_);
lean_inc(v_cache_394_);
lean_inc(v_traceState_389_);
lean_inc(v_auxDeclNGen_393_);
lean_inc(v_ngen_392_);
lean_inc(v_nextMacroScope_391_);
lean_inc(v_env_390_);
lean_dec(v___x_388_);
v___x_399_ = lean_box(0);
v_isShared_400_ = v_isSharedCheck_416_;
goto v_resetjp_398_;
}
v_resetjp_398_:
{
uint64_t v_tid_401_; lean_object* v___x_403_; uint8_t v_isShared_404_; uint8_t v_isSharedCheck_414_; 
v_tid_401_ = lean_ctor_get_uint64(v_traceState_389_, sizeof(void*)*1);
v_isSharedCheck_414_ = !lean_is_exclusive(v_traceState_389_);
if (v_isSharedCheck_414_ == 0)
{
lean_object* v_unused_415_; 
v_unused_415_ = lean_ctor_get(v_traceState_389_, 0);
lean_dec(v_unused_415_);
v___x_403_ = v_traceState_389_;
v_isShared_404_ = v_isSharedCheck_414_;
goto v_resetjp_402_;
}
else
{
lean_dec(v_traceState_389_);
v___x_403_ = lean_box(0);
v_isShared_404_ = v_isSharedCheck_414_;
goto v_resetjp_402_;
}
v_resetjp_402_:
{
lean_object* v___x_405_; lean_object* v___x_407_; 
v___x_405_ = lean_obj_once(&lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_Goal_withHeadlineTraceNode_spec__0___redArg___closed__1, &lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_Goal_withHeadlineTraceNode_spec__0___redArg___closed__1_once, _init_lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_Goal_withHeadlineTraceNode_spec__0___redArg___closed__1);
if (v_isShared_404_ == 0)
{
lean_ctor_set(v___x_403_, 0, v___x_405_);
v___x_407_ = v___x_403_;
goto v_reusejp_406_;
}
else
{
lean_object* v_reuseFailAlloc_413_; 
v_reuseFailAlloc_413_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_413_, 0, v___x_405_);
lean_ctor_set_uint64(v_reuseFailAlloc_413_, sizeof(void*)*1, v_tid_401_);
v___x_407_ = v_reuseFailAlloc_413_;
goto v_reusejp_406_;
}
v_reusejp_406_:
{
lean_object* v___x_409_; 
if (v_isShared_400_ == 0)
{
lean_ctor_set(v___x_399_, 4, v___x_407_);
v___x_409_ = v___x_399_;
goto v_reusejp_408_;
}
else
{
lean_object* v_reuseFailAlloc_412_; 
v_reuseFailAlloc_412_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_412_, 0, v_env_390_);
lean_ctor_set(v_reuseFailAlloc_412_, 1, v_nextMacroScope_391_);
lean_ctor_set(v_reuseFailAlloc_412_, 2, v_ngen_392_);
lean_ctor_set(v_reuseFailAlloc_412_, 3, v_auxDeclNGen_393_);
lean_ctor_set(v_reuseFailAlloc_412_, 4, v___x_407_);
lean_ctor_set(v_reuseFailAlloc_412_, 5, v_cache_394_);
lean_ctor_set(v_reuseFailAlloc_412_, 6, v_messages_395_);
lean_ctor_set(v_reuseFailAlloc_412_, 7, v_infoState_396_);
lean_ctor_set(v_reuseFailAlloc_412_, 8, v_snapshotTasks_397_);
v___x_409_ = v_reuseFailAlloc_412_;
goto v_reusejp_408_;
}
v_reusejp_408_:
{
lean_object* v___x_410_; lean_object* v___x_411_; 
v___x_410_ = lean_st_ref_set(v___y_383_, v___x_409_);
v___x_411_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_411_, 0, v_traces_387_);
return v___x_411_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_Goal_withHeadlineTraceNode_spec__0___redArg___boxed(lean_object* v___y_417_, lean_object* v___y_418_){
_start:
{
lean_object* v_res_419_; 
v_res_419_ = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_Goal_withHeadlineTraceNode_spec__0___redArg(v___y_417_);
lean_dec(v___y_417_);
return v_res_419_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_Goal_withHeadlineTraceNode_spec__0(lean_object* v___y_420_, lean_object* v___y_421_, lean_object* v___y_422_, lean_object* v___y_423_){
_start:
{
lean_object* v___x_425_; 
v___x_425_ = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_Goal_withHeadlineTraceNode_spec__0___redArg(v___y_423_);
return v___x_425_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_Goal_withHeadlineTraceNode_spec__0___boxed(lean_object* v___y_426_, lean_object* v___y_427_, lean_object* v___y_428_, lean_object* v___y_429_, lean_object* v___y_430_){
_start:
{
lean_object* v_res_431_; 
v_res_431_ = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_Goal_withHeadlineTraceNode_spec__0(v___y_426_, v___y_427_, v___y_428_, v___y_429_);
lean_dec(v___y_429_);
lean_dec_ref(v___y_428_);
lean_dec(v___y_427_);
lean_dec_ref(v___y_426_);
return v_res_431_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_Option_get___at___00Aesop_Goal_withHeadlineTraceNode_spec__1(lean_object* v_opts_432_, lean_object* v_opt_433_){
_start:
{
lean_object* v_name_434_; lean_object* v_defValue_435_; lean_object* v_map_436_; lean_object* v___x_437_; 
v_name_434_ = lean_ctor_get(v_opt_433_, 0);
v_defValue_435_ = lean_ctor_get(v_opt_433_, 1);
v_map_436_ = lean_ctor_get(v_opts_432_, 0);
v___x_437_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_436_, v_name_434_);
if (lean_obj_tag(v___x_437_) == 0)
{
uint8_t v___x_438_; 
v___x_438_ = lean_unbox(v_defValue_435_);
return v___x_438_;
}
else
{
lean_object* v_val_439_; 
v_val_439_ = lean_ctor_get(v___x_437_, 0);
lean_inc(v_val_439_);
lean_dec_ref_known(v___x_437_, 1);
if (lean_obj_tag(v_val_439_) == 1)
{
uint8_t v_v_440_; 
v_v_440_ = lean_ctor_get_uint8(v_val_439_, 0);
lean_dec_ref_known(v_val_439_, 0);
return v_v_440_;
}
else
{
uint8_t v___x_441_; 
lean_dec(v_val_439_);
v___x_441_ = lean_unbox(v_defValue_435_);
return v___x_441_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00Aesop_Goal_withHeadlineTraceNode_spec__1___boxed(lean_object* v_opts_442_, lean_object* v_opt_443_){
_start:
{
uint8_t v_res_444_; lean_object* v_r_445_; 
v_res_444_ = lp_aesop_Lean_Option_get___at___00Aesop_Goal_withHeadlineTraceNode_spec__1(v_opts_442_, v_opt_443_);
lean_dec_ref(v_opt_443_);
lean_dec_ref(v_opts_442_);
v_r_445_ = lean_box(v_res_444_);
return v_r_445_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_withHeadlineTraceNode___redArg___lam__0(lean_object* v_g_446_, lean_object* v_transform_447_, lean_object* v_x_448_, lean_object* v___y_449_, lean_object* v___y_450_, lean_object* v___y_451_, lean_object* v___y_452_){
_start:
{
lean_object* v___x_454_; 
v___x_454_ = lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt(v_g_446_, v_transform_447_, v___y_449_, v___y_450_, v___y_451_, v___y_452_);
return v___x_454_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_withHeadlineTraceNode___redArg___lam__0___boxed(lean_object* v_g_455_, lean_object* v_transform_456_, lean_object* v_x_457_, lean_object* v___y_458_, lean_object* v___y_459_, lean_object* v___y_460_, lean_object* v___y_461_, lean_object* v___y_462_){
_start:
{
lean_object* v_res_463_; 
v_res_463_ = lp_aesop_Aesop_Goal_withHeadlineTraceNode___redArg___lam__0(v_g_455_, v_transform_456_, v_x_457_, v___y_458_, v___y_459_, v___y_460_, v___y_461_);
lean_dec(v___y_461_);
lean_dec_ref(v___y_460_);
lean_dec(v___y_459_);
lean_dec_ref(v___y_458_);
lean_dec_ref(v_x_457_);
return v_res_463_;
}
}
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Goal_withHeadlineTraceNode_spec__2_spec__3___redArg(lean_object* v_x_464_){
_start:
{
if (lean_obj_tag(v_x_464_) == 0)
{
lean_object* v_a_466_; lean_object* v___x_468_; uint8_t v_isShared_469_; uint8_t v_isSharedCheck_473_; 
v_a_466_ = lean_ctor_get(v_x_464_, 0);
v_isSharedCheck_473_ = !lean_is_exclusive(v_x_464_);
if (v_isSharedCheck_473_ == 0)
{
v___x_468_ = v_x_464_;
v_isShared_469_ = v_isSharedCheck_473_;
goto v_resetjp_467_;
}
else
{
lean_inc(v_a_466_);
lean_dec(v_x_464_);
v___x_468_ = lean_box(0);
v_isShared_469_ = v_isSharedCheck_473_;
goto v_resetjp_467_;
}
v_resetjp_467_:
{
lean_object* v___x_471_; 
if (v_isShared_469_ == 0)
{
lean_ctor_set_tag(v___x_468_, 1);
v___x_471_ = v___x_468_;
goto v_reusejp_470_;
}
else
{
lean_object* v_reuseFailAlloc_472_; 
v_reuseFailAlloc_472_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_472_, 0, v_a_466_);
v___x_471_ = v_reuseFailAlloc_472_;
goto v_reusejp_470_;
}
v_reusejp_470_:
{
return v___x_471_;
}
}
}
else
{
lean_object* v_a_474_; lean_object* v___x_476_; uint8_t v_isShared_477_; uint8_t v_isSharedCheck_481_; 
v_a_474_ = lean_ctor_get(v_x_464_, 0);
v_isSharedCheck_481_ = !lean_is_exclusive(v_x_464_);
if (v_isSharedCheck_481_ == 0)
{
v___x_476_ = v_x_464_;
v_isShared_477_ = v_isSharedCheck_481_;
goto v_resetjp_475_;
}
else
{
lean_inc(v_a_474_);
lean_dec(v_x_464_);
v___x_476_ = lean_box(0);
v_isShared_477_ = v_isSharedCheck_481_;
goto v_resetjp_475_;
}
v_resetjp_475_:
{
lean_object* v___x_479_; 
if (v_isShared_477_ == 0)
{
lean_ctor_set_tag(v___x_476_, 0);
v___x_479_ = v___x_476_;
goto v_reusejp_478_;
}
else
{
lean_object* v_reuseFailAlloc_480_; 
v_reuseFailAlloc_480_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_480_, 0, v_a_474_);
v___x_479_ = v_reuseFailAlloc_480_;
goto v_reusejp_478_;
}
v_reusejp_478_:
{
return v___x_479_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Goal_withHeadlineTraceNode_spec__2_spec__3___redArg___boxed(lean_object* v_x_482_, lean_object* v___y_483_){
_start:
{
lean_object* v_res_484_; 
v_res_484_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Goal_withHeadlineTraceNode_spec__2_spec__3___redArg(v_x_482_);
return v_res_484_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Goal_withHeadlineTraceNode_spec__2_spec__5(lean_object* v_opts_485_, lean_object* v_opt_486_){
_start:
{
lean_object* v_name_487_; lean_object* v_defValue_488_; lean_object* v_map_489_; lean_object* v___x_490_; 
v_name_487_ = lean_ctor_get(v_opt_486_, 0);
v_defValue_488_ = lean_ctor_get(v_opt_486_, 1);
v_map_489_ = lean_ctor_get(v_opts_485_, 0);
v___x_490_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_489_, v_name_487_);
if (lean_obj_tag(v___x_490_) == 0)
{
lean_inc(v_defValue_488_);
return v_defValue_488_;
}
else
{
lean_object* v_val_491_; 
v_val_491_ = lean_ctor_get(v___x_490_, 0);
lean_inc(v_val_491_);
lean_dec_ref_known(v___x_490_, 1);
if (lean_obj_tag(v_val_491_) == 3)
{
lean_object* v_v_492_; 
v_v_492_ = lean_ctor_get(v_val_491_, 0);
lean_inc(v_v_492_);
lean_dec_ref_known(v_val_491_, 1);
return v_v_492_;
}
else
{
lean_dec(v_val_491_);
lean_inc(v_defValue_488_);
return v_defValue_488_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Goal_withHeadlineTraceNode_spec__2_spec__5___boxed(lean_object* v_opts_493_, lean_object* v_opt_494_){
_start:
{
lean_object* v_res_495_; 
v_res_495_ = lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Goal_withHeadlineTraceNode_spec__2_spec__5(v_opts_493_, v_opt_494_);
lean_dec_ref(v_opt_494_);
lean_dec_ref(v_opts_493_);
return v_res_495_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Goal_withHeadlineTraceNode_spec__2_spec__2_spec__3(size_t v_sz_496_, size_t v_i_497_, lean_object* v_bs_498_){
_start:
{
uint8_t v___x_499_; 
v___x_499_ = lean_usize_dec_lt(v_i_497_, v_sz_496_);
if (v___x_499_ == 0)
{
return v_bs_498_;
}
else
{
lean_object* v_v_500_; lean_object* v_msg_501_; lean_object* v___x_502_; lean_object* v_bs_x27_503_; size_t v___x_504_; size_t v___x_505_; lean_object* v___x_506_; 
v_v_500_ = lean_array_uget_borrowed(v_bs_498_, v_i_497_);
v_msg_501_ = lean_ctor_get(v_v_500_, 1);
lean_inc_ref(v_msg_501_);
v___x_502_ = lean_unsigned_to_nat(0u);
v_bs_x27_503_ = lean_array_uset(v_bs_498_, v_i_497_, v___x_502_);
v___x_504_ = ((size_t)1ULL);
v___x_505_ = lean_usize_add(v_i_497_, v___x_504_);
v___x_506_ = lean_array_uset(v_bs_x27_503_, v_i_497_, v_msg_501_);
v_i_497_ = v___x_505_;
v_bs_498_ = v___x_506_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Goal_withHeadlineTraceNode_spec__2_spec__2_spec__3___boxed(lean_object* v_sz_508_, lean_object* v_i_509_, lean_object* v_bs_510_){
_start:
{
size_t v_sz_boxed_511_; size_t v_i_boxed_512_; lean_object* v_res_513_; 
v_sz_boxed_511_ = lean_unbox_usize(v_sz_508_);
lean_dec(v_sz_508_);
v_i_boxed_512_ = lean_unbox_usize(v_i_509_);
lean_dec(v_i_509_);
v_res_513_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Goal_withHeadlineTraceNode_spec__2_spec__2_spec__3(v_sz_boxed_511_, v_i_boxed_512_, v_bs_510_);
return v_res_513_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Goal_withHeadlineTraceNode_spec__2_spec__2(lean_object* v_oldTraces_514_, lean_object* v_data_515_, lean_object* v_ref_516_, lean_object* v_msg_517_, lean_object* v___y_518_, lean_object* v___y_519_, lean_object* v___y_520_, lean_object* v___y_521_){
_start:
{
lean_object* v_fileName_523_; lean_object* v_fileMap_524_; lean_object* v_options_525_; lean_object* v_currRecDepth_526_; lean_object* v_maxRecDepth_527_; lean_object* v_ref_528_; lean_object* v_currNamespace_529_; lean_object* v_openDecls_530_; lean_object* v_initHeartbeats_531_; lean_object* v_maxHeartbeats_532_; lean_object* v_quotContext_533_; lean_object* v_currMacroScope_534_; uint8_t v_diag_535_; lean_object* v_cancelTk_x3f_536_; uint8_t v_suppressElabErrors_537_; lean_object* v_inheritedTraceOptions_538_; lean_object* v___x_539_; lean_object* v_traceState_540_; lean_object* v_traces_541_; lean_object* v_ref_542_; lean_object* v___x_543_; lean_object* v___x_544_; size_t v_sz_545_; size_t v___x_546_; lean_object* v___x_547_; lean_object* v_msg_548_; lean_object* v___x_549_; lean_object* v_a_550_; lean_object* v___x_552_; uint8_t v_isShared_553_; uint8_t v_isSharedCheck_587_; 
v_fileName_523_ = lean_ctor_get(v___y_520_, 0);
v_fileMap_524_ = lean_ctor_get(v___y_520_, 1);
v_options_525_ = lean_ctor_get(v___y_520_, 2);
v_currRecDepth_526_ = lean_ctor_get(v___y_520_, 3);
v_maxRecDepth_527_ = lean_ctor_get(v___y_520_, 4);
v_ref_528_ = lean_ctor_get(v___y_520_, 5);
v_currNamespace_529_ = lean_ctor_get(v___y_520_, 6);
v_openDecls_530_ = lean_ctor_get(v___y_520_, 7);
v_initHeartbeats_531_ = lean_ctor_get(v___y_520_, 8);
v_maxHeartbeats_532_ = lean_ctor_get(v___y_520_, 9);
v_quotContext_533_ = lean_ctor_get(v___y_520_, 10);
v_currMacroScope_534_ = lean_ctor_get(v___y_520_, 11);
v_diag_535_ = lean_ctor_get_uint8(v___y_520_, sizeof(void*)*14);
v_cancelTk_x3f_536_ = lean_ctor_get(v___y_520_, 12);
v_suppressElabErrors_537_ = lean_ctor_get_uint8(v___y_520_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_538_ = lean_ctor_get(v___y_520_, 13);
v___x_539_ = lean_st_ref_get(v___y_521_);
v_traceState_540_ = lean_ctor_get(v___x_539_, 4);
lean_inc_ref(v_traceState_540_);
lean_dec(v___x_539_);
v_traces_541_ = lean_ctor_get(v_traceState_540_, 0);
lean_inc_ref(v_traces_541_);
lean_dec_ref(v_traceState_540_);
v_ref_542_ = l_Lean_replaceRef(v_ref_516_, v_ref_528_);
lean_inc_ref(v_inheritedTraceOptions_538_);
lean_inc(v_cancelTk_x3f_536_);
lean_inc(v_currMacroScope_534_);
lean_inc(v_quotContext_533_);
lean_inc(v_maxHeartbeats_532_);
lean_inc(v_initHeartbeats_531_);
lean_inc(v_openDecls_530_);
lean_inc(v_currNamespace_529_);
lean_inc(v_maxRecDepth_527_);
lean_inc(v_currRecDepth_526_);
lean_inc_ref(v_options_525_);
lean_inc_ref(v_fileMap_524_);
lean_inc_ref(v_fileName_523_);
v___x_543_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_543_, 0, v_fileName_523_);
lean_ctor_set(v___x_543_, 1, v_fileMap_524_);
lean_ctor_set(v___x_543_, 2, v_options_525_);
lean_ctor_set(v___x_543_, 3, v_currRecDepth_526_);
lean_ctor_set(v___x_543_, 4, v_maxRecDepth_527_);
lean_ctor_set(v___x_543_, 5, v_ref_542_);
lean_ctor_set(v___x_543_, 6, v_currNamespace_529_);
lean_ctor_set(v___x_543_, 7, v_openDecls_530_);
lean_ctor_set(v___x_543_, 8, v_initHeartbeats_531_);
lean_ctor_set(v___x_543_, 9, v_maxHeartbeats_532_);
lean_ctor_set(v___x_543_, 10, v_quotContext_533_);
lean_ctor_set(v___x_543_, 11, v_currMacroScope_534_);
lean_ctor_set(v___x_543_, 12, v_cancelTk_x3f_536_);
lean_ctor_set(v___x_543_, 13, v_inheritedTraceOptions_538_);
lean_ctor_set_uint8(v___x_543_, sizeof(void*)*14, v_diag_535_);
lean_ctor_set_uint8(v___x_543_, sizeof(void*)*14 + 1, v_suppressElabErrors_537_);
v___x_544_ = l_Lean_PersistentArray_toArray___redArg(v_traces_541_);
lean_dec_ref(v_traces_541_);
v_sz_545_ = lean_array_size(v___x_544_);
v___x_546_ = ((size_t)0ULL);
v___x_547_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Goal_withHeadlineTraceNode_spec__2_spec__2_spec__3(v_sz_545_, v___x_546_, v___x_544_);
v_msg_548_ = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(v_msg_548_, 0, v_data_515_);
lean_ctor_set(v_msg_548_, 1, v_msg_517_);
lean_ctor_set(v_msg_548_, 2, v___x_547_);
v___x_549_ = lp_aesop_Lean_addMessageContextFull___at___00__private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt_spec__0(v_msg_548_, v___y_518_, v___y_519_, v___x_543_, v___y_521_);
lean_dec_ref_known(v___x_543_, 14);
v_a_550_ = lean_ctor_get(v___x_549_, 0);
v_isSharedCheck_587_ = !lean_is_exclusive(v___x_549_);
if (v_isSharedCheck_587_ == 0)
{
v___x_552_ = v___x_549_;
v_isShared_553_ = v_isSharedCheck_587_;
goto v_resetjp_551_;
}
else
{
lean_inc(v_a_550_);
lean_dec(v___x_549_);
v___x_552_ = lean_box(0);
v_isShared_553_ = v_isSharedCheck_587_;
goto v_resetjp_551_;
}
v_resetjp_551_:
{
lean_object* v___x_554_; lean_object* v_traceState_555_; lean_object* v_env_556_; lean_object* v_nextMacroScope_557_; lean_object* v_ngen_558_; lean_object* v_auxDeclNGen_559_; lean_object* v_cache_560_; lean_object* v_messages_561_; lean_object* v_infoState_562_; lean_object* v_snapshotTasks_563_; lean_object* v___x_565_; uint8_t v_isShared_566_; uint8_t v_isSharedCheck_586_; 
v___x_554_ = lean_st_ref_take(v___y_521_);
v_traceState_555_ = lean_ctor_get(v___x_554_, 4);
v_env_556_ = lean_ctor_get(v___x_554_, 0);
v_nextMacroScope_557_ = lean_ctor_get(v___x_554_, 1);
v_ngen_558_ = lean_ctor_get(v___x_554_, 2);
v_auxDeclNGen_559_ = lean_ctor_get(v___x_554_, 3);
v_cache_560_ = lean_ctor_get(v___x_554_, 5);
v_messages_561_ = lean_ctor_get(v___x_554_, 6);
v_infoState_562_ = lean_ctor_get(v___x_554_, 7);
v_snapshotTasks_563_ = lean_ctor_get(v___x_554_, 8);
v_isSharedCheck_586_ = !lean_is_exclusive(v___x_554_);
if (v_isSharedCheck_586_ == 0)
{
v___x_565_ = v___x_554_;
v_isShared_566_ = v_isSharedCheck_586_;
goto v_resetjp_564_;
}
else
{
lean_inc(v_snapshotTasks_563_);
lean_inc(v_infoState_562_);
lean_inc(v_messages_561_);
lean_inc(v_cache_560_);
lean_inc(v_traceState_555_);
lean_inc(v_auxDeclNGen_559_);
lean_inc(v_ngen_558_);
lean_inc(v_nextMacroScope_557_);
lean_inc(v_env_556_);
lean_dec(v___x_554_);
v___x_565_ = lean_box(0);
v_isShared_566_ = v_isSharedCheck_586_;
goto v_resetjp_564_;
}
v_resetjp_564_:
{
uint64_t v_tid_567_; lean_object* v___x_569_; uint8_t v_isShared_570_; uint8_t v_isSharedCheck_584_; 
v_tid_567_ = lean_ctor_get_uint64(v_traceState_555_, sizeof(void*)*1);
v_isSharedCheck_584_ = !lean_is_exclusive(v_traceState_555_);
if (v_isSharedCheck_584_ == 0)
{
lean_object* v_unused_585_; 
v_unused_585_ = lean_ctor_get(v_traceState_555_, 0);
lean_dec(v_unused_585_);
v___x_569_ = v_traceState_555_;
v_isShared_570_ = v_isSharedCheck_584_;
goto v_resetjp_568_;
}
else
{
lean_dec(v_traceState_555_);
v___x_569_ = lean_box(0);
v_isShared_570_ = v_isSharedCheck_584_;
goto v_resetjp_568_;
}
v_resetjp_568_:
{
lean_object* v___x_571_; lean_object* v___x_572_; lean_object* v___x_574_; 
v___x_571_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_571_, 0, v_ref_516_);
lean_ctor_set(v___x_571_, 1, v_a_550_);
v___x_572_ = l_Lean_PersistentArray_push___redArg(v_oldTraces_514_, v___x_571_);
if (v_isShared_570_ == 0)
{
lean_ctor_set(v___x_569_, 0, v___x_572_);
v___x_574_ = v___x_569_;
goto v_reusejp_573_;
}
else
{
lean_object* v_reuseFailAlloc_583_; 
v_reuseFailAlloc_583_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_583_, 0, v___x_572_);
lean_ctor_set_uint64(v_reuseFailAlloc_583_, sizeof(void*)*1, v_tid_567_);
v___x_574_ = v_reuseFailAlloc_583_;
goto v_reusejp_573_;
}
v_reusejp_573_:
{
lean_object* v___x_576_; 
if (v_isShared_566_ == 0)
{
lean_ctor_set(v___x_565_, 4, v___x_574_);
v___x_576_ = v___x_565_;
goto v_reusejp_575_;
}
else
{
lean_object* v_reuseFailAlloc_582_; 
v_reuseFailAlloc_582_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_582_, 0, v_env_556_);
lean_ctor_set(v_reuseFailAlloc_582_, 1, v_nextMacroScope_557_);
lean_ctor_set(v_reuseFailAlloc_582_, 2, v_ngen_558_);
lean_ctor_set(v_reuseFailAlloc_582_, 3, v_auxDeclNGen_559_);
lean_ctor_set(v_reuseFailAlloc_582_, 4, v___x_574_);
lean_ctor_set(v_reuseFailAlloc_582_, 5, v_cache_560_);
lean_ctor_set(v_reuseFailAlloc_582_, 6, v_messages_561_);
lean_ctor_set(v_reuseFailAlloc_582_, 7, v_infoState_562_);
lean_ctor_set(v_reuseFailAlloc_582_, 8, v_snapshotTasks_563_);
v___x_576_ = v_reuseFailAlloc_582_;
goto v_reusejp_575_;
}
v_reusejp_575_:
{
lean_object* v___x_577_; lean_object* v___x_578_; lean_object* v___x_580_; 
v___x_577_ = lean_st_ref_set(v___y_521_, v___x_576_);
v___x_578_ = lean_box(0);
if (v_isShared_553_ == 0)
{
lean_ctor_set(v___x_552_, 0, v___x_578_);
v___x_580_ = v___x_552_;
goto v_reusejp_579_;
}
else
{
lean_object* v_reuseFailAlloc_581_; 
v_reuseFailAlloc_581_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_581_, 0, v___x_578_);
v___x_580_ = v_reuseFailAlloc_581_;
goto v_reusejp_579_;
}
v_reusejp_579_:
{
return v___x_580_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Goal_withHeadlineTraceNode_spec__2_spec__2___boxed(lean_object* v_oldTraces_588_, lean_object* v_data_589_, lean_object* v_ref_590_, lean_object* v_msg_591_, lean_object* v___y_592_, lean_object* v___y_593_, lean_object* v___y_594_, lean_object* v___y_595_, lean_object* v___y_596_){
_start:
{
lean_object* v_res_597_; 
v_res_597_ = lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Goal_withHeadlineTraceNode_spec__2_spec__2(v_oldTraces_588_, v_data_589_, v_ref_590_, v_msg_591_, v___y_592_, v___y_593_, v___y_594_, v___y_595_);
lean_dec(v___y_595_);
lean_dec_ref(v___y_594_);
lean_dec(v___y_593_);
lean_dec_ref(v___y_592_);
return v_res_597_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Goal_withHeadlineTraceNode_spec__2_spec__4___redArg(lean_object* v_e_598_){
_start:
{
if (lean_obj_tag(v_e_598_) == 0)
{
uint8_t v___x_599_; 
v___x_599_ = 2;
return v___x_599_;
}
else
{
uint8_t v___x_600_; 
v___x_600_ = 0;
return v___x_600_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Goal_withHeadlineTraceNode_spec__2_spec__4___redArg___boxed(lean_object* v_e_601_){
_start:
{
uint8_t v_res_602_; lean_object* v_r_603_; 
v_res_602_ = lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Goal_withHeadlineTraceNode_spec__2_spec__4___redArg(v_e_601_);
lean_dec_ref(v_e_601_);
v_r_603_ = lean_box(v_res_602_);
return v_r_603_;
}
}
static double _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Goal_withHeadlineTraceNode_spec__2___redArg___closed__0(void){
_start:
{
lean_object* v___x_604_; double v___x_605_; 
v___x_604_ = lean_unsigned_to_nat(0u);
v___x_605_ = lean_float_of_nat(v___x_604_);
return v___x_605_;
}
}
static lean_object* _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Goal_withHeadlineTraceNode_spec__2___redArg___closed__2(void){
_start:
{
lean_object* v___x_607_; lean_object* v___x_608_; 
v___x_607_ = ((lean_object*)(lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Goal_withHeadlineTraceNode_spec__2___redArg___closed__1));
v___x_608_ = l_Lean_stringToMessageData(v___x_607_);
return v___x_608_;
}
}
static double _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Goal_withHeadlineTraceNode_spec__2___redArg___closed__3(void){
_start:
{
lean_object* v___x_609_; double v___x_610_; 
v___x_609_ = lean_unsigned_to_nat(1000u);
v___x_610_ = lean_float_of_nat(v___x_609_);
return v___x_610_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Goal_withHeadlineTraceNode_spec__2___redArg(lean_object* v_cls_611_, uint8_t v_collapsed_612_, lean_object* v_tag_613_, lean_object* v_opts_614_, uint8_t v_clsEnabled_615_, lean_object* v_oldTraces_616_, lean_object* v_msg_617_, lean_object* v_resStartStop_618_, lean_object* v___y_619_, lean_object* v___y_620_, lean_object* v___y_621_, lean_object* v___y_622_){
_start:
{
lean_object* v_fst_624_; lean_object* v_snd_625_; lean_object* v___y_627_; lean_object* v___y_628_; lean_object* v_data_629_; lean_object* v_fst_640_; lean_object* v_snd_641_; lean_object* v___x_642_; uint8_t v___x_643_; lean_object* v___y_645_; lean_object* v_a_646_; uint8_t v___y_661_; double v___y_692_; 
v_fst_624_ = lean_ctor_get(v_resStartStop_618_, 0);
lean_inc(v_fst_624_);
v_snd_625_ = lean_ctor_get(v_resStartStop_618_, 1);
lean_inc(v_snd_625_);
lean_dec_ref(v_resStartStop_618_);
v_fst_640_ = lean_ctor_get(v_snd_625_, 0);
lean_inc(v_fst_640_);
v_snd_641_ = lean_ctor_get(v_snd_625_, 1);
lean_inc(v_snd_641_);
lean_dec(v_snd_625_);
v___x_642_ = l_Lean_trace_profiler;
v___x_643_ = lp_aesop_Lean_Option_get___at___00Aesop_Goal_withHeadlineTraceNode_spec__1(v_opts_614_, v___x_642_);
if (v___x_643_ == 0)
{
v___y_661_ = v___x_643_;
goto v___jp_660_;
}
else
{
lean_object* v___x_697_; uint8_t v___x_698_; 
v___x_697_ = l_Lean_trace_profiler_useHeartbeats;
v___x_698_ = lp_aesop_Lean_Option_get___at___00Aesop_Goal_withHeadlineTraceNode_spec__1(v_opts_614_, v___x_697_);
if (v___x_698_ == 0)
{
lean_object* v___x_699_; lean_object* v___x_700_; double v___x_701_; double v___x_702_; double v___x_703_; 
v___x_699_ = l_Lean_trace_profiler_threshold;
v___x_700_ = lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Goal_withHeadlineTraceNode_spec__2_spec__5(v_opts_614_, v___x_699_);
v___x_701_ = lean_float_of_nat(v___x_700_);
v___x_702_ = lean_float_once(&lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Goal_withHeadlineTraceNode_spec__2___redArg___closed__3, &lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Goal_withHeadlineTraceNode_spec__2___redArg___closed__3_once, _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Goal_withHeadlineTraceNode_spec__2___redArg___closed__3);
v___x_703_ = lean_float_div(v___x_701_, v___x_702_);
v___y_692_ = v___x_703_;
goto v___jp_691_;
}
else
{
lean_object* v___x_704_; lean_object* v___x_705_; double v___x_706_; 
v___x_704_ = l_Lean_trace_profiler_threshold;
v___x_705_ = lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Goal_withHeadlineTraceNode_spec__2_spec__5(v_opts_614_, v___x_704_);
v___x_706_ = lean_float_of_nat(v___x_705_);
v___y_692_ = v___x_706_;
goto v___jp_691_;
}
}
v___jp_626_:
{
lean_object* v___x_630_; 
lean_inc(v___y_627_);
v___x_630_ = lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Goal_withHeadlineTraceNode_spec__2_spec__2(v_oldTraces_616_, v_data_629_, v___y_627_, v___y_628_, v___y_619_, v___y_620_, v___y_621_, v___y_622_);
if (lean_obj_tag(v___x_630_) == 0)
{
lean_object* v___x_631_; 
lean_dec_ref_known(v___x_630_, 1);
v___x_631_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Goal_withHeadlineTraceNode_spec__2_spec__3___redArg(v_fst_624_);
return v___x_631_;
}
else
{
lean_object* v_a_632_; lean_object* v___x_634_; uint8_t v_isShared_635_; uint8_t v_isSharedCheck_639_; 
lean_dec(v_fst_624_);
v_a_632_ = lean_ctor_get(v___x_630_, 0);
v_isSharedCheck_639_ = !lean_is_exclusive(v___x_630_);
if (v_isSharedCheck_639_ == 0)
{
v___x_634_ = v___x_630_;
v_isShared_635_ = v_isSharedCheck_639_;
goto v_resetjp_633_;
}
else
{
lean_inc(v_a_632_);
lean_dec(v___x_630_);
v___x_634_ = lean_box(0);
v_isShared_635_ = v_isSharedCheck_639_;
goto v_resetjp_633_;
}
v_resetjp_633_:
{
lean_object* v___x_637_; 
if (v_isShared_635_ == 0)
{
v___x_637_ = v___x_634_;
goto v_reusejp_636_;
}
else
{
lean_object* v_reuseFailAlloc_638_; 
v_reuseFailAlloc_638_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_638_, 0, v_a_632_);
v___x_637_ = v_reuseFailAlloc_638_;
goto v_reusejp_636_;
}
v_reusejp_636_:
{
return v___x_637_;
}
}
}
}
v___jp_644_:
{
uint8_t v_result_647_; lean_object* v___x_648_; lean_object* v___x_649_; double v___x_650_; lean_object* v_data_651_; 
v_result_647_ = lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Goal_withHeadlineTraceNode_spec__2_spec__4___redArg(v_fst_624_);
v___x_648_ = lean_box(v_result_647_);
v___x_649_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_649_, 0, v___x_648_);
v___x_650_ = lean_float_once(&lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Goal_withHeadlineTraceNode_spec__2___redArg___closed__0, &lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Goal_withHeadlineTraceNode_spec__2___redArg___closed__0_once, _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Goal_withHeadlineTraceNode_spec__2___redArg___closed__0);
lean_inc_ref(v_tag_613_);
lean_inc_ref(v___x_649_);
lean_inc(v_cls_611_);
v_data_651_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v_data_651_, 0, v_cls_611_);
lean_ctor_set(v_data_651_, 1, v___x_649_);
lean_ctor_set(v_data_651_, 2, v_tag_613_);
lean_ctor_set_float(v_data_651_, sizeof(void*)*3, v___x_650_);
lean_ctor_set_float(v_data_651_, sizeof(void*)*3 + 8, v___x_650_);
lean_ctor_set_uint8(v_data_651_, sizeof(void*)*3 + 16, v_collapsed_612_);
if (v___x_643_ == 0)
{
lean_dec_ref_known(v___x_649_, 1);
lean_dec(v_snd_641_);
lean_dec(v_fst_640_);
lean_dec_ref(v_tag_613_);
lean_dec(v_cls_611_);
v___y_627_ = v___y_645_;
v___y_628_ = v_a_646_;
v_data_629_ = v_data_651_;
goto v___jp_626_;
}
else
{
lean_object* v_data_652_; double v___x_653_; double v___x_654_; 
lean_dec_ref_known(v_data_651_, 3);
v_data_652_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v_data_652_, 0, v_cls_611_);
lean_ctor_set(v_data_652_, 1, v___x_649_);
lean_ctor_set(v_data_652_, 2, v_tag_613_);
v___x_653_ = lean_unbox_float(v_fst_640_);
lean_dec(v_fst_640_);
lean_ctor_set_float(v_data_652_, sizeof(void*)*3, v___x_653_);
v___x_654_ = lean_unbox_float(v_snd_641_);
lean_dec(v_snd_641_);
lean_ctor_set_float(v_data_652_, sizeof(void*)*3 + 8, v___x_654_);
lean_ctor_set_uint8(v_data_652_, sizeof(void*)*3 + 16, v_collapsed_612_);
v___y_627_ = v___y_645_;
v___y_628_ = v_a_646_;
v_data_629_ = v_data_652_;
goto v___jp_626_;
}
}
v___jp_655_:
{
lean_object* v_ref_656_; lean_object* v___x_657_; 
v_ref_656_ = lean_ctor_get(v___y_621_, 5);
lean_inc(v___y_622_);
lean_inc_ref(v___y_621_);
lean_inc(v___y_620_);
lean_inc_ref(v___y_619_);
lean_inc(v_fst_624_);
v___x_657_ = lean_apply_6(v_msg_617_, v_fst_624_, v___y_619_, v___y_620_, v___y_621_, v___y_622_, lean_box(0));
if (lean_obj_tag(v___x_657_) == 0)
{
lean_object* v_a_658_; 
v_a_658_ = lean_ctor_get(v___x_657_, 0);
lean_inc(v_a_658_);
lean_dec_ref_known(v___x_657_, 1);
v___y_645_ = v_ref_656_;
v_a_646_ = v_a_658_;
goto v___jp_644_;
}
else
{
lean_object* v___x_659_; 
lean_dec_ref_known(v___x_657_, 1);
v___x_659_ = lean_obj_once(&lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Goal_withHeadlineTraceNode_spec__2___redArg___closed__2, &lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Goal_withHeadlineTraceNode_spec__2___redArg___closed__2_once, _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Goal_withHeadlineTraceNode_spec__2___redArg___closed__2);
v___y_645_ = v_ref_656_;
v_a_646_ = v___x_659_;
goto v___jp_644_;
}
}
v___jp_660_:
{
if (v_clsEnabled_615_ == 0)
{
if (v___y_661_ == 0)
{
lean_object* v___x_662_; lean_object* v_traceState_663_; lean_object* v_env_664_; lean_object* v_nextMacroScope_665_; lean_object* v_ngen_666_; lean_object* v_auxDeclNGen_667_; lean_object* v_cache_668_; lean_object* v_messages_669_; lean_object* v_infoState_670_; lean_object* v_snapshotTasks_671_; lean_object* v___x_673_; uint8_t v_isShared_674_; uint8_t v_isSharedCheck_690_; 
lean_dec(v_snd_641_);
lean_dec(v_fst_640_);
lean_dec_ref(v_msg_617_);
lean_dec_ref(v_tag_613_);
lean_dec(v_cls_611_);
v___x_662_ = lean_st_ref_take(v___y_622_);
v_traceState_663_ = lean_ctor_get(v___x_662_, 4);
v_env_664_ = lean_ctor_get(v___x_662_, 0);
v_nextMacroScope_665_ = lean_ctor_get(v___x_662_, 1);
v_ngen_666_ = lean_ctor_get(v___x_662_, 2);
v_auxDeclNGen_667_ = lean_ctor_get(v___x_662_, 3);
v_cache_668_ = lean_ctor_get(v___x_662_, 5);
v_messages_669_ = lean_ctor_get(v___x_662_, 6);
v_infoState_670_ = lean_ctor_get(v___x_662_, 7);
v_snapshotTasks_671_ = lean_ctor_get(v___x_662_, 8);
v_isSharedCheck_690_ = !lean_is_exclusive(v___x_662_);
if (v_isSharedCheck_690_ == 0)
{
v___x_673_ = v___x_662_;
v_isShared_674_ = v_isSharedCheck_690_;
goto v_resetjp_672_;
}
else
{
lean_inc(v_snapshotTasks_671_);
lean_inc(v_infoState_670_);
lean_inc(v_messages_669_);
lean_inc(v_cache_668_);
lean_inc(v_traceState_663_);
lean_inc(v_auxDeclNGen_667_);
lean_inc(v_ngen_666_);
lean_inc(v_nextMacroScope_665_);
lean_inc(v_env_664_);
lean_dec(v___x_662_);
v___x_673_ = lean_box(0);
v_isShared_674_ = v_isSharedCheck_690_;
goto v_resetjp_672_;
}
v_resetjp_672_:
{
uint64_t v_tid_675_; lean_object* v_traces_676_; lean_object* v___x_678_; uint8_t v_isShared_679_; uint8_t v_isSharedCheck_689_; 
v_tid_675_ = lean_ctor_get_uint64(v_traceState_663_, sizeof(void*)*1);
v_traces_676_ = lean_ctor_get(v_traceState_663_, 0);
v_isSharedCheck_689_ = !lean_is_exclusive(v_traceState_663_);
if (v_isSharedCheck_689_ == 0)
{
v___x_678_ = v_traceState_663_;
v_isShared_679_ = v_isSharedCheck_689_;
goto v_resetjp_677_;
}
else
{
lean_inc(v_traces_676_);
lean_dec(v_traceState_663_);
v___x_678_ = lean_box(0);
v_isShared_679_ = v_isSharedCheck_689_;
goto v_resetjp_677_;
}
v_resetjp_677_:
{
lean_object* v___x_680_; lean_object* v___x_682_; 
v___x_680_ = l_Lean_PersistentArray_append___redArg(v_oldTraces_616_, v_traces_676_);
lean_dec_ref(v_traces_676_);
if (v_isShared_679_ == 0)
{
lean_ctor_set(v___x_678_, 0, v___x_680_);
v___x_682_ = v___x_678_;
goto v_reusejp_681_;
}
else
{
lean_object* v_reuseFailAlloc_688_; 
v_reuseFailAlloc_688_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_688_, 0, v___x_680_);
lean_ctor_set_uint64(v_reuseFailAlloc_688_, sizeof(void*)*1, v_tid_675_);
v___x_682_ = v_reuseFailAlloc_688_;
goto v_reusejp_681_;
}
v_reusejp_681_:
{
lean_object* v___x_684_; 
if (v_isShared_674_ == 0)
{
lean_ctor_set(v___x_673_, 4, v___x_682_);
v___x_684_ = v___x_673_;
goto v_reusejp_683_;
}
else
{
lean_object* v_reuseFailAlloc_687_; 
v_reuseFailAlloc_687_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_687_, 0, v_env_664_);
lean_ctor_set(v_reuseFailAlloc_687_, 1, v_nextMacroScope_665_);
lean_ctor_set(v_reuseFailAlloc_687_, 2, v_ngen_666_);
lean_ctor_set(v_reuseFailAlloc_687_, 3, v_auxDeclNGen_667_);
lean_ctor_set(v_reuseFailAlloc_687_, 4, v___x_682_);
lean_ctor_set(v_reuseFailAlloc_687_, 5, v_cache_668_);
lean_ctor_set(v_reuseFailAlloc_687_, 6, v_messages_669_);
lean_ctor_set(v_reuseFailAlloc_687_, 7, v_infoState_670_);
lean_ctor_set(v_reuseFailAlloc_687_, 8, v_snapshotTasks_671_);
v___x_684_ = v_reuseFailAlloc_687_;
goto v_reusejp_683_;
}
v_reusejp_683_:
{
lean_object* v___x_685_; lean_object* v___x_686_; 
v___x_685_ = lean_st_ref_set(v___y_622_, v___x_684_);
v___x_686_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Goal_withHeadlineTraceNode_spec__2_spec__3___redArg(v_fst_624_);
return v___x_686_;
}
}
}
}
}
else
{
goto v___jp_655_;
}
}
else
{
goto v___jp_655_;
}
}
v___jp_691_:
{
double v___x_693_; double v___x_694_; double v___x_695_; uint8_t v___x_696_; 
v___x_693_ = lean_unbox_float(v_snd_641_);
v___x_694_ = lean_unbox_float(v_fst_640_);
v___x_695_ = lean_float_sub(v___x_693_, v___x_694_);
v___x_696_ = lean_float_decLt(v___y_692_, v___x_695_);
v___y_661_ = v___x_696_;
goto v___jp_660_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Goal_withHeadlineTraceNode_spec__2___redArg___boxed(lean_object* v_cls_707_, lean_object* v_collapsed_708_, lean_object* v_tag_709_, lean_object* v_opts_710_, lean_object* v_clsEnabled_711_, lean_object* v_oldTraces_712_, lean_object* v_msg_713_, lean_object* v_resStartStop_714_, lean_object* v___y_715_, lean_object* v___y_716_, lean_object* v___y_717_, lean_object* v___y_718_, lean_object* v___y_719_){
_start:
{
uint8_t v_collapsed_boxed_720_; uint8_t v_clsEnabled_boxed_721_; lean_object* v_res_722_; 
v_collapsed_boxed_720_ = lean_unbox(v_collapsed_708_);
v_clsEnabled_boxed_721_ = lean_unbox(v_clsEnabled_711_);
v_res_722_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Goal_withHeadlineTraceNode_spec__2___redArg(v_cls_707_, v_collapsed_boxed_720_, v_tag_709_, v_opts_710_, v_clsEnabled_boxed_721_, v_oldTraces_712_, v_msg_713_, v_resStartStop_714_, v___y_715_, v___y_716_, v___y_717_, v___y_718_);
lean_dec(v___y_718_);
lean_dec_ref(v___y_717_);
lean_dec(v___y_716_);
lean_dec_ref(v___y_715_);
lean_dec_ref(v_opts_710_);
return v_res_722_;
}
}
static double _init_lp_aesop_Aesop_Goal_withHeadlineTraceNode___redArg___closed__3(void){
_start:
{
lean_object* v___x_727_; double v___x_728_; 
v___x_727_ = lean_unsigned_to_nat(1000000000u);
v___x_728_ = lean_float_of_nat(v___x_727_);
return v___x_728_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_withHeadlineTraceNode___redArg(lean_object* v_g_729_, lean_object* v_traceOpt_730_, lean_object* v_k_731_, uint8_t v_collapsed_732_, lean_object* v_transform_733_, lean_object* v_a_734_, lean_object* v_a_735_, lean_object* v_a_736_, lean_object* v_a_737_){
_start:
{
lean_object* v_options_739_; uint8_t v_hasTrace_740_; 
v_options_739_ = lean_ctor_get(v_a_736_, 2);
v_hasTrace_740_ = lean_ctor_get_uint8(v_options_739_, sizeof(void*)*1);
if (v_hasTrace_740_ == 0)
{
lean_object* v___x_741_; 
lean_dec_ref(v_transform_733_);
lean_dec_ref(v_traceOpt_730_);
lean_dec(v_g_729_);
lean_inc(v_a_737_);
lean_inc_ref(v_a_736_);
lean_inc(v_a_735_);
lean_inc_ref(v_a_734_);
v___x_741_ = lean_apply_5(v_k_731_, v_a_734_, v_a_735_, v_a_736_, v_a_737_, lean_box(0));
return v___x_741_;
}
else
{
lean_object* v_inheritedTraceOptions_742_; lean_object* v_traceClass_743_; lean_object* v___x_745_; uint8_t v_isShared_746_; uint8_t v_isSharedCheck_825_; 
v_inheritedTraceOptions_742_ = lean_ctor_get(v_a_736_, 13);
v_traceClass_743_ = lean_ctor_get(v_traceOpt_730_, 0);
v_isSharedCheck_825_ = !lean_is_exclusive(v_traceOpt_730_);
if (v_isSharedCheck_825_ == 0)
{
lean_object* v_unused_826_; 
v_unused_826_ = lean_ctor_get(v_traceOpt_730_, 1);
lean_dec(v_unused_826_);
v___x_745_ = v_traceOpt_730_;
v_isShared_746_ = v_isSharedCheck_825_;
goto v_resetjp_744_;
}
else
{
lean_inc(v_traceClass_743_);
lean_dec(v_traceOpt_730_);
v___x_745_ = lean_box(0);
v_isShared_746_ = v_isSharedCheck_825_;
goto v_resetjp_744_;
}
v_resetjp_744_:
{
lean_object* v___f_747_; lean_object* v___x_748_; lean_object* v___x_749_; lean_object* v___x_750_; uint8_t v___x_751_; lean_object* v___y_753_; lean_object* v___y_754_; lean_object* v_a_755_; lean_object* v___y_770_; lean_object* v___y_771_; lean_object* v_a_772_; 
v___f_747_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Goal_withHeadlineTraceNode___redArg___lam__0___boxed), 8, 2);
lean_closure_set(v___f_747_, 0, v_g_729_);
lean_closure_set(v___f_747_, 1, v_transform_733_);
v___x_748_ = ((lean_object*)(lp_aesop_Aesop_Goal_withHeadlineTraceNode___redArg___closed__0));
v___x_749_ = ((lean_object*)(lp_aesop_Aesop_Goal_withHeadlineTraceNode___redArg___closed__2));
lean_inc(v_traceClass_743_);
v___x_750_ = l_Lean_Name_append(v___x_749_, v_traceClass_743_);
v___x_751_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_742_, v_options_739_, v___x_750_);
lean_dec(v___x_750_);
if (v___x_751_ == 0)
{
lean_object* v___x_822_; uint8_t v___x_823_; 
v___x_822_ = l_Lean_trace_profiler;
v___x_823_ = lp_aesop_Lean_Option_get___at___00Aesop_Goal_withHeadlineTraceNode_spec__1(v_options_739_, v___x_822_);
if (v___x_823_ == 0)
{
lean_object* v___x_824_; 
lean_dec_ref(v___f_747_);
lean_del_object(v___x_745_);
lean_dec(v_traceClass_743_);
lean_inc(v_a_737_);
lean_inc_ref(v_a_736_);
lean_inc(v_a_735_);
lean_inc_ref(v_a_734_);
v___x_824_ = lean_apply_5(v_k_731_, v_a_734_, v_a_735_, v_a_736_, v_a_737_, lean_box(0));
return v___x_824_;
}
else
{
goto v___jp_781_;
}
}
else
{
goto v___jp_781_;
}
v___jp_752_:
{
lean_object* v___x_756_; double v___x_757_; double v___x_758_; double v___x_759_; double v___x_760_; double v___x_761_; lean_object* v___x_762_; lean_object* v___x_763_; lean_object* v___x_765_; 
v___x_756_ = lean_io_mono_nanos_now();
v___x_757_ = lean_float_of_nat(v___y_754_);
v___x_758_ = lean_float_once(&lp_aesop_Aesop_Goal_withHeadlineTraceNode___redArg___closed__3, &lp_aesop_Aesop_Goal_withHeadlineTraceNode___redArg___closed__3_once, _init_lp_aesop_Aesop_Goal_withHeadlineTraceNode___redArg___closed__3);
v___x_759_ = lean_float_div(v___x_757_, v___x_758_);
v___x_760_ = lean_float_of_nat(v___x_756_);
v___x_761_ = lean_float_div(v___x_760_, v___x_758_);
v___x_762_ = lean_box_float(v___x_759_);
v___x_763_ = lean_box_float(v___x_761_);
if (v_isShared_746_ == 0)
{
lean_ctor_set(v___x_745_, 1, v___x_763_);
lean_ctor_set(v___x_745_, 0, v___x_762_);
v___x_765_ = v___x_745_;
goto v_reusejp_764_;
}
else
{
lean_object* v_reuseFailAlloc_768_; 
v_reuseFailAlloc_768_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_768_, 0, v___x_762_);
lean_ctor_set(v_reuseFailAlloc_768_, 1, v___x_763_);
v___x_765_ = v_reuseFailAlloc_768_;
goto v_reusejp_764_;
}
v_reusejp_764_:
{
lean_object* v___x_766_; lean_object* v___x_767_; 
v___x_766_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_766_, 0, v_a_755_);
lean_ctor_set(v___x_766_, 1, v___x_765_);
v___x_767_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Goal_withHeadlineTraceNode_spec__2___redArg(v_traceClass_743_, v_collapsed_732_, v___x_748_, v_options_739_, v___x_751_, v___y_753_, v___f_747_, v___x_766_, v_a_734_, v_a_735_, v_a_736_, v_a_737_);
return v___x_767_;
}
}
v___jp_769_:
{
lean_object* v___x_773_; double v___x_774_; double v___x_775_; lean_object* v___x_776_; lean_object* v___x_777_; lean_object* v___x_778_; lean_object* v___x_779_; lean_object* v___x_780_; 
v___x_773_ = lean_io_get_num_heartbeats();
v___x_774_ = lean_float_of_nat(v___y_770_);
v___x_775_ = lean_float_of_nat(v___x_773_);
v___x_776_ = lean_box_float(v___x_774_);
v___x_777_ = lean_box_float(v___x_775_);
v___x_778_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_778_, 0, v___x_776_);
lean_ctor_set(v___x_778_, 1, v___x_777_);
v___x_779_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_779_, 0, v_a_772_);
lean_ctor_set(v___x_779_, 1, v___x_778_);
v___x_780_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Goal_withHeadlineTraceNode_spec__2___redArg(v_traceClass_743_, v_collapsed_732_, v___x_748_, v_options_739_, v___x_751_, v___y_771_, v___f_747_, v___x_779_, v_a_734_, v_a_735_, v_a_736_, v_a_737_);
return v___x_780_;
}
v___jp_781_:
{
lean_object* v___x_782_; lean_object* v_a_783_; lean_object* v___x_784_; uint8_t v___x_785_; 
v___x_782_ = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_Goal_withHeadlineTraceNode_spec__0___redArg(v_a_737_);
v_a_783_ = lean_ctor_get(v___x_782_, 0);
lean_inc(v_a_783_);
lean_dec_ref(v___x_782_);
v___x_784_ = l_Lean_trace_profiler_useHeartbeats;
v___x_785_ = lp_aesop_Lean_Option_get___at___00Aesop_Goal_withHeadlineTraceNode_spec__1(v_options_739_, v___x_784_);
if (v___x_785_ == 0)
{
lean_object* v___x_786_; lean_object* v___x_787_; 
v___x_786_ = lean_io_mono_nanos_now();
lean_inc(v_a_737_);
lean_inc_ref(v_a_736_);
lean_inc(v_a_735_);
lean_inc_ref(v_a_734_);
v___x_787_ = lean_apply_5(v_k_731_, v_a_734_, v_a_735_, v_a_736_, v_a_737_, lean_box(0));
if (lean_obj_tag(v___x_787_) == 0)
{
lean_object* v_a_788_; lean_object* v___x_790_; uint8_t v_isShared_791_; uint8_t v_isSharedCheck_795_; 
v_a_788_ = lean_ctor_get(v___x_787_, 0);
v_isSharedCheck_795_ = !lean_is_exclusive(v___x_787_);
if (v_isSharedCheck_795_ == 0)
{
v___x_790_ = v___x_787_;
v_isShared_791_ = v_isSharedCheck_795_;
goto v_resetjp_789_;
}
else
{
lean_inc(v_a_788_);
lean_dec(v___x_787_);
v___x_790_ = lean_box(0);
v_isShared_791_ = v_isSharedCheck_795_;
goto v_resetjp_789_;
}
v_resetjp_789_:
{
lean_object* v___x_793_; 
if (v_isShared_791_ == 0)
{
lean_ctor_set_tag(v___x_790_, 1);
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
v___y_753_ = v_a_783_;
v___y_754_ = v___x_786_;
v_a_755_ = v___x_793_;
goto v___jp_752_;
}
}
}
else
{
lean_object* v_a_796_; lean_object* v___x_798_; uint8_t v_isShared_799_; uint8_t v_isSharedCheck_803_; 
v_a_796_ = lean_ctor_get(v___x_787_, 0);
v_isSharedCheck_803_ = !lean_is_exclusive(v___x_787_);
if (v_isSharedCheck_803_ == 0)
{
v___x_798_ = v___x_787_;
v_isShared_799_ = v_isSharedCheck_803_;
goto v_resetjp_797_;
}
else
{
lean_inc(v_a_796_);
lean_dec(v___x_787_);
v___x_798_ = lean_box(0);
v_isShared_799_ = v_isSharedCheck_803_;
goto v_resetjp_797_;
}
v_resetjp_797_:
{
lean_object* v___x_801_; 
if (v_isShared_799_ == 0)
{
lean_ctor_set_tag(v___x_798_, 0);
v___x_801_ = v___x_798_;
goto v_reusejp_800_;
}
else
{
lean_object* v_reuseFailAlloc_802_; 
v_reuseFailAlloc_802_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_802_, 0, v_a_796_);
v___x_801_ = v_reuseFailAlloc_802_;
goto v_reusejp_800_;
}
v_reusejp_800_:
{
v___y_753_ = v_a_783_;
v___y_754_ = v___x_786_;
v_a_755_ = v___x_801_;
goto v___jp_752_;
}
}
}
}
else
{
lean_object* v___x_804_; lean_object* v___x_805_; 
lean_del_object(v___x_745_);
v___x_804_ = lean_io_get_num_heartbeats();
lean_inc(v_a_737_);
lean_inc_ref(v_a_736_);
lean_inc(v_a_735_);
lean_inc_ref(v_a_734_);
v___x_805_ = lean_apply_5(v_k_731_, v_a_734_, v_a_735_, v_a_736_, v_a_737_, lean_box(0));
if (lean_obj_tag(v___x_805_) == 0)
{
lean_object* v_a_806_; lean_object* v___x_808_; uint8_t v_isShared_809_; uint8_t v_isSharedCheck_813_; 
v_a_806_ = lean_ctor_get(v___x_805_, 0);
v_isSharedCheck_813_ = !lean_is_exclusive(v___x_805_);
if (v_isSharedCheck_813_ == 0)
{
v___x_808_ = v___x_805_;
v_isShared_809_ = v_isSharedCheck_813_;
goto v_resetjp_807_;
}
else
{
lean_inc(v_a_806_);
lean_dec(v___x_805_);
v___x_808_ = lean_box(0);
v_isShared_809_ = v_isSharedCheck_813_;
goto v_resetjp_807_;
}
v_resetjp_807_:
{
lean_object* v___x_811_; 
if (v_isShared_809_ == 0)
{
lean_ctor_set_tag(v___x_808_, 1);
v___x_811_ = v___x_808_;
goto v_reusejp_810_;
}
else
{
lean_object* v_reuseFailAlloc_812_; 
v_reuseFailAlloc_812_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_812_, 0, v_a_806_);
v___x_811_ = v_reuseFailAlloc_812_;
goto v_reusejp_810_;
}
v_reusejp_810_:
{
v___y_770_ = v___x_804_;
v___y_771_ = v_a_783_;
v_a_772_ = v___x_811_;
goto v___jp_769_;
}
}
}
else
{
lean_object* v_a_814_; lean_object* v___x_816_; uint8_t v_isShared_817_; uint8_t v_isSharedCheck_821_; 
v_a_814_ = lean_ctor_get(v___x_805_, 0);
v_isSharedCheck_821_ = !lean_is_exclusive(v___x_805_);
if (v_isSharedCheck_821_ == 0)
{
v___x_816_ = v___x_805_;
v_isShared_817_ = v_isSharedCheck_821_;
goto v_resetjp_815_;
}
else
{
lean_inc(v_a_814_);
lean_dec(v___x_805_);
v___x_816_ = lean_box(0);
v_isShared_817_ = v_isSharedCheck_821_;
goto v_resetjp_815_;
}
v_resetjp_815_:
{
lean_object* v___x_819_; 
if (v_isShared_817_ == 0)
{
lean_ctor_set_tag(v___x_816_, 0);
v___x_819_ = v___x_816_;
goto v_reusejp_818_;
}
else
{
lean_object* v_reuseFailAlloc_820_; 
v_reuseFailAlloc_820_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_820_, 0, v_a_814_);
v___x_819_ = v_reuseFailAlloc_820_;
goto v_reusejp_818_;
}
v_reusejp_818_:
{
v___y_770_ = v___x_804_;
v___y_771_ = v_a_783_;
v_a_772_ = v___x_819_;
goto v___jp_769_;
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_withHeadlineTraceNode___redArg___boxed(lean_object* v_g_827_, lean_object* v_traceOpt_828_, lean_object* v_k_829_, lean_object* v_collapsed_830_, lean_object* v_transform_831_, lean_object* v_a_832_, lean_object* v_a_833_, lean_object* v_a_834_, lean_object* v_a_835_, lean_object* v_a_836_){
_start:
{
uint8_t v_collapsed_boxed_837_; lean_object* v_res_838_; 
v_collapsed_boxed_837_ = lean_unbox(v_collapsed_830_);
v_res_838_ = lp_aesop_Aesop_Goal_withHeadlineTraceNode___redArg(v_g_827_, v_traceOpt_828_, v_k_829_, v_collapsed_boxed_837_, v_transform_831_, v_a_832_, v_a_833_, v_a_834_, v_a_835_);
lean_dec(v_a_835_);
lean_dec_ref(v_a_834_);
lean_dec(v_a_833_);
lean_dec_ref(v_a_832_);
return v_res_838_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_withHeadlineTraceNode(lean_object* v_00_u03b1_839_, lean_object* v_g_840_, lean_object* v_traceOpt_841_, lean_object* v_k_842_, uint8_t v_collapsed_843_, lean_object* v_transform_844_, lean_object* v_a_845_, lean_object* v_a_846_, lean_object* v_a_847_, lean_object* v_a_848_){
_start:
{
lean_object* v___x_850_; 
v___x_850_ = lp_aesop_Aesop_Goal_withHeadlineTraceNode___redArg(v_g_840_, v_traceOpt_841_, v_k_842_, v_collapsed_843_, v_transform_844_, v_a_845_, v_a_846_, v_a_847_, v_a_848_);
return v___x_850_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_withHeadlineTraceNode___boxed(lean_object* v_00_u03b1_851_, lean_object* v_g_852_, lean_object* v_traceOpt_853_, lean_object* v_k_854_, lean_object* v_collapsed_855_, lean_object* v_transform_856_, lean_object* v_a_857_, lean_object* v_a_858_, lean_object* v_a_859_, lean_object* v_a_860_, lean_object* v_a_861_){
_start:
{
uint8_t v_collapsed_boxed_862_; lean_object* v_res_863_; 
v_collapsed_boxed_862_ = lean_unbox(v_collapsed_855_);
v_res_863_ = lp_aesop_Aesop_Goal_withHeadlineTraceNode(v_00_u03b1_851_, v_g_852_, v_traceOpt_853_, v_k_854_, v_collapsed_boxed_862_, v_transform_856_, v_a_857_, v_a_858_, v_a_859_, v_a_860_);
lean_dec(v_a_860_);
lean_dec_ref(v_a_859_);
lean_dec(v_a_858_);
lean_dec_ref(v_a_857_);
return v_res_863_;
}
}
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Goal_withHeadlineTraceNode_spec__2_spec__3(lean_object* v_00_u03b1_864_, lean_object* v_x_865_, lean_object* v___y_866_, lean_object* v___y_867_, lean_object* v___y_868_, lean_object* v___y_869_){
_start:
{
lean_object* v___x_871_; 
v___x_871_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Goal_withHeadlineTraceNode_spec__2_spec__3___redArg(v_x_865_);
return v___x_871_;
}
}
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Goal_withHeadlineTraceNode_spec__2_spec__3___boxed(lean_object* v_00_u03b1_872_, lean_object* v_x_873_, lean_object* v___y_874_, lean_object* v___y_875_, lean_object* v___y_876_, lean_object* v___y_877_, lean_object* v___y_878_){
_start:
{
lean_object* v_res_879_; 
v_res_879_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Goal_withHeadlineTraceNode_spec__2_spec__3(v_00_u03b1_872_, v_x_873_, v___y_874_, v___y_875_, v___y_876_, v___y_877_);
lean_dec(v___y_877_);
lean_dec_ref(v___y_876_);
lean_dec(v___y_875_);
lean_dec_ref(v___y_874_);
return v_res_879_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Goal_withHeadlineTraceNode_spec__2_spec__4(lean_object* v_00_u03b1_880_, lean_object* v_e_881_){
_start:
{
uint8_t v___x_882_; 
v___x_882_ = lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Goal_withHeadlineTraceNode_spec__2_spec__4___redArg(v_e_881_);
return v___x_882_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Goal_withHeadlineTraceNode_spec__2_spec__4___boxed(lean_object* v_00_u03b1_883_, lean_object* v_e_884_){
_start:
{
uint8_t v_res_885_; lean_object* v_r_886_; 
v_res_885_ = lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Goal_withHeadlineTraceNode_spec__2_spec__4(v_00_u03b1_883_, v_e_884_);
lean_dec_ref(v_e_884_);
v_r_886_ = lean_box(v_res_885_);
return v_r_886_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Goal_withHeadlineTraceNode_spec__2(lean_object* v_00_u03b1_887_, lean_object* v_cls_888_, uint8_t v_collapsed_889_, lean_object* v_tag_890_, lean_object* v_opts_891_, uint8_t v_clsEnabled_892_, lean_object* v_oldTraces_893_, lean_object* v_msg_894_, lean_object* v_resStartStop_895_, lean_object* v___y_896_, lean_object* v___y_897_, lean_object* v___y_898_, lean_object* v___y_899_){
_start:
{
lean_object* v___x_901_; 
v___x_901_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Goal_withHeadlineTraceNode_spec__2___redArg(v_cls_888_, v_collapsed_889_, v_tag_890_, v_opts_891_, v_clsEnabled_892_, v_oldTraces_893_, v_msg_894_, v_resStartStop_895_, v___y_896_, v___y_897_, v___y_898_, v___y_899_);
return v___x_901_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Goal_withHeadlineTraceNode_spec__2___boxed(lean_object* v_00_u03b1_902_, lean_object* v_cls_903_, lean_object* v_collapsed_904_, lean_object* v_tag_905_, lean_object* v_opts_906_, lean_object* v_clsEnabled_907_, lean_object* v_oldTraces_908_, lean_object* v_msg_909_, lean_object* v_resStartStop_910_, lean_object* v___y_911_, lean_object* v___y_912_, lean_object* v___y_913_, lean_object* v___y_914_, lean_object* v___y_915_){
_start:
{
uint8_t v_collapsed_boxed_916_; uint8_t v_clsEnabled_boxed_917_; lean_object* v_res_918_; 
v_collapsed_boxed_916_ = lean_unbox(v_collapsed_904_);
v_clsEnabled_boxed_917_ = lean_unbox(v_clsEnabled_907_);
v_res_918_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Goal_withHeadlineTraceNode_spec__2(v_00_u03b1_902_, v_cls_903_, v_collapsed_boxed_916_, v_tag_905_, v_opts_906_, v_clsEnabled_boxed_917_, v_oldTraces_908_, v_msg_909_, v_resStartStop_910_, v___y_911_, v___y_912_, v___y_913_, v___y_914_);
lean_dec(v___y_914_);
lean_dec_ref(v___y_913_);
lean_dec(v___y_912_);
lean_dec_ref(v___y_911_);
lean_dec_ref(v_opts_906_);
return v_res_918_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addTrace___at___00__private_Aesop_Tree_Tracing_0__Aesop_Goal_traceMetadata_trc_spec__0(lean_object* v_cls_921_, lean_object* v_msg_922_, lean_object* v___y_923_, lean_object* v___y_924_, lean_object* v___y_925_, lean_object* v___y_926_){
_start:
{
lean_object* v_ref_928_; lean_object* v___x_929_; lean_object* v_a_930_; lean_object* v___x_932_; uint8_t v_isShared_933_; uint8_t v_isSharedCheck_974_; 
v_ref_928_ = lean_ctor_get(v___y_925_, 5);
v___x_929_ = lp_aesop_Lean_addMessageContextFull___at___00__private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt_spec__0(v_msg_922_, v___y_923_, v___y_924_, v___y_925_, v___y_926_);
v_a_930_ = lean_ctor_get(v___x_929_, 0);
v_isSharedCheck_974_ = !lean_is_exclusive(v___x_929_);
if (v_isSharedCheck_974_ == 0)
{
v___x_932_ = v___x_929_;
v_isShared_933_ = v_isSharedCheck_974_;
goto v_resetjp_931_;
}
else
{
lean_inc(v_a_930_);
lean_dec(v___x_929_);
v___x_932_ = lean_box(0);
v_isShared_933_ = v_isSharedCheck_974_;
goto v_resetjp_931_;
}
v_resetjp_931_:
{
lean_object* v___x_934_; lean_object* v_traceState_935_; lean_object* v_env_936_; lean_object* v_nextMacroScope_937_; lean_object* v_ngen_938_; lean_object* v_auxDeclNGen_939_; lean_object* v_cache_940_; lean_object* v_messages_941_; lean_object* v_infoState_942_; lean_object* v_snapshotTasks_943_; lean_object* v___x_945_; uint8_t v_isShared_946_; uint8_t v_isSharedCheck_973_; 
v___x_934_ = lean_st_ref_take(v___y_926_);
v_traceState_935_ = lean_ctor_get(v___x_934_, 4);
v_env_936_ = lean_ctor_get(v___x_934_, 0);
v_nextMacroScope_937_ = lean_ctor_get(v___x_934_, 1);
v_ngen_938_ = lean_ctor_get(v___x_934_, 2);
v_auxDeclNGen_939_ = lean_ctor_get(v___x_934_, 3);
v_cache_940_ = lean_ctor_get(v___x_934_, 5);
v_messages_941_ = lean_ctor_get(v___x_934_, 6);
v_infoState_942_ = lean_ctor_get(v___x_934_, 7);
v_snapshotTasks_943_ = lean_ctor_get(v___x_934_, 8);
v_isSharedCheck_973_ = !lean_is_exclusive(v___x_934_);
if (v_isSharedCheck_973_ == 0)
{
v___x_945_ = v___x_934_;
v_isShared_946_ = v_isSharedCheck_973_;
goto v_resetjp_944_;
}
else
{
lean_inc(v_snapshotTasks_943_);
lean_inc(v_infoState_942_);
lean_inc(v_messages_941_);
lean_inc(v_cache_940_);
lean_inc(v_traceState_935_);
lean_inc(v_auxDeclNGen_939_);
lean_inc(v_ngen_938_);
lean_inc(v_nextMacroScope_937_);
lean_inc(v_env_936_);
lean_dec(v___x_934_);
v___x_945_ = lean_box(0);
v_isShared_946_ = v_isSharedCheck_973_;
goto v_resetjp_944_;
}
v_resetjp_944_:
{
uint64_t v_tid_947_; lean_object* v_traces_948_; lean_object* v___x_950_; uint8_t v_isShared_951_; uint8_t v_isSharedCheck_972_; 
v_tid_947_ = lean_ctor_get_uint64(v_traceState_935_, sizeof(void*)*1);
v_traces_948_ = lean_ctor_get(v_traceState_935_, 0);
v_isSharedCheck_972_ = !lean_is_exclusive(v_traceState_935_);
if (v_isSharedCheck_972_ == 0)
{
v___x_950_ = v_traceState_935_;
v_isShared_951_ = v_isSharedCheck_972_;
goto v_resetjp_949_;
}
else
{
lean_inc(v_traces_948_);
lean_dec(v_traceState_935_);
v___x_950_ = lean_box(0);
v_isShared_951_ = v_isSharedCheck_972_;
goto v_resetjp_949_;
}
v_resetjp_949_:
{
lean_object* v___x_952_; double v___x_953_; uint8_t v___x_954_; lean_object* v___x_955_; lean_object* v___x_956_; lean_object* v___x_957_; lean_object* v___x_958_; lean_object* v___x_959_; lean_object* v___x_960_; lean_object* v___x_962_; 
v___x_952_ = lean_box(0);
v___x_953_ = lean_float_once(&lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Goal_withHeadlineTraceNode_spec__2___redArg___closed__0, &lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Goal_withHeadlineTraceNode_spec__2___redArg___closed__0_once, _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Goal_withHeadlineTraceNode_spec__2___redArg___closed__0);
v___x_954_ = 0;
v___x_955_ = ((lean_object*)(lp_aesop_Aesop_Goal_withHeadlineTraceNode___redArg___closed__0));
v___x_956_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v___x_956_, 0, v_cls_921_);
lean_ctor_set(v___x_956_, 1, v___x_952_);
lean_ctor_set(v___x_956_, 2, v___x_955_);
lean_ctor_set_float(v___x_956_, sizeof(void*)*3, v___x_953_);
lean_ctor_set_float(v___x_956_, sizeof(void*)*3 + 8, v___x_953_);
lean_ctor_set_uint8(v___x_956_, sizeof(void*)*3 + 16, v___x_954_);
v___x_957_ = ((lean_object*)(lp_aesop_Lean_addTrace___at___00__private_Aesop_Tree_Tracing_0__Aesop_Goal_traceMetadata_trc_spec__0___closed__0));
v___x_958_ = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(v___x_958_, 0, v___x_956_);
lean_ctor_set(v___x_958_, 1, v_a_930_);
lean_ctor_set(v___x_958_, 2, v___x_957_);
lean_inc(v_ref_928_);
v___x_959_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_959_, 0, v_ref_928_);
lean_ctor_set(v___x_959_, 1, v___x_958_);
v___x_960_ = l_Lean_PersistentArray_push___redArg(v_traces_948_, v___x_959_);
if (v_isShared_951_ == 0)
{
lean_ctor_set(v___x_950_, 0, v___x_960_);
v___x_962_ = v___x_950_;
goto v_reusejp_961_;
}
else
{
lean_object* v_reuseFailAlloc_971_; 
v_reuseFailAlloc_971_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_971_, 0, v___x_960_);
lean_ctor_set_uint64(v_reuseFailAlloc_971_, sizeof(void*)*1, v_tid_947_);
v___x_962_ = v_reuseFailAlloc_971_;
goto v_reusejp_961_;
}
v_reusejp_961_:
{
lean_object* v___x_964_; 
if (v_isShared_946_ == 0)
{
lean_ctor_set(v___x_945_, 4, v___x_962_);
v___x_964_ = v___x_945_;
goto v_reusejp_963_;
}
else
{
lean_object* v_reuseFailAlloc_970_; 
v_reuseFailAlloc_970_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_970_, 0, v_env_936_);
lean_ctor_set(v_reuseFailAlloc_970_, 1, v_nextMacroScope_937_);
lean_ctor_set(v_reuseFailAlloc_970_, 2, v_ngen_938_);
lean_ctor_set(v_reuseFailAlloc_970_, 3, v_auxDeclNGen_939_);
lean_ctor_set(v_reuseFailAlloc_970_, 4, v___x_962_);
lean_ctor_set(v_reuseFailAlloc_970_, 5, v_cache_940_);
lean_ctor_set(v_reuseFailAlloc_970_, 6, v_messages_941_);
lean_ctor_set(v_reuseFailAlloc_970_, 7, v_infoState_942_);
lean_ctor_set(v_reuseFailAlloc_970_, 8, v_snapshotTasks_943_);
v___x_964_ = v_reuseFailAlloc_970_;
goto v_reusejp_963_;
}
v_reusejp_963_:
{
lean_object* v___x_965_; lean_object* v___x_966_; lean_object* v___x_968_; 
v___x_965_ = lean_st_ref_set(v___y_926_, v___x_964_);
v___x_966_ = lean_box(0);
if (v_isShared_933_ == 0)
{
lean_ctor_set(v___x_932_, 0, v___x_966_);
v___x_968_ = v___x_932_;
goto v_reusejp_967_;
}
else
{
lean_object* v_reuseFailAlloc_969_; 
v_reuseFailAlloc_969_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_969_, 0, v___x_966_);
v___x_968_ = v_reuseFailAlloc_969_;
goto v_reusejp_967_;
}
v_reusejp_967_:
{
return v___x_968_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addTrace___at___00__private_Aesop_Tree_Tracing_0__Aesop_Goal_traceMetadata_trc_spec__0___boxed(lean_object* v_cls_975_, lean_object* v_msg_976_, lean_object* v___y_977_, lean_object* v___y_978_, lean_object* v___y_979_, lean_object* v___y_980_, lean_object* v___y_981_){
_start:
{
lean_object* v_res_982_; 
v_res_982_ = lp_aesop_Lean_addTrace___at___00__private_Aesop_Tree_Tracing_0__Aesop_Goal_traceMetadata_trc_spec__0(v_cls_975_, v_msg_976_, v___y_977_, v___y_978_, v___y_979_, v___y_980_);
lean_dec(v___y_980_);
lean_dec_ref(v___y_979_);
lean_dec(v___y_978_);
lean_dec_ref(v___y_977_);
return v_res_982_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Goal_traceMetadata_trc(lean_object* v_traceOpt_983_, lean_object* v_msg_984_, lean_object* v_a_985_, lean_object* v_a_986_, lean_object* v_a_987_, lean_object* v_a_988_){
_start:
{
lean_object* v_traceClass_990_; lean_object* v___x_991_; 
v_traceClass_990_ = lean_ctor_get(v_traceOpt_983_, 0);
lean_inc(v_traceClass_990_);
lean_dec_ref(v_traceOpt_983_);
v___x_991_ = lp_aesop_Lean_addTrace___at___00__private_Aesop_Tree_Tracing_0__Aesop_Goal_traceMetadata_trc_spec__0(v_traceClass_990_, v_msg_984_, v_a_985_, v_a_986_, v_a_987_, v_a_988_);
return v___x_991_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Goal_traceMetadata_trc___boxed(lean_object* v_traceOpt_992_, lean_object* v_msg_993_, lean_object* v_a_994_, lean_object* v_a_995_, lean_object* v_a_996_, lean_object* v_a_997_, lean_object* v_a_998_){
_start:
{
lean_object* v_res_999_; 
v_res_999_ = lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Goal_traceMetadata_trc(v_traceOpt_992_, v_msg_993_, v_a_994_, v_a_995_, v_a_996_, v_a_997_);
lean_dec(v_a_997_);
lean_dec_ref(v_a_996_);
lean_dec(v_a_995_);
lean_dec_ref(v_a_994_);
return v_res_999_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Goal_traceMetadata_trcNode___lam__0(lean_object* v_msg_1000_, lean_object* v_x_1001_, lean_object* v___y_1002_, lean_object* v___y_1003_, lean_object* v___y_1004_, lean_object* v___y_1005_){
_start:
{
lean_object* v___x_1007_; 
v___x_1007_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1007_, 0, v_msg_1000_);
return v___x_1007_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Goal_traceMetadata_trcNode___lam__0___boxed(lean_object* v_msg_1008_, lean_object* v_x_1009_, lean_object* v___y_1010_, lean_object* v___y_1011_, lean_object* v___y_1012_, lean_object* v___y_1013_, lean_object* v___y_1014_){
_start:
{
lean_object* v_res_1015_; 
v_res_1015_ = lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Goal_traceMetadata_trcNode___lam__0(v_msg_1008_, v_x_1009_, v___y_1010_, v___y_1011_, v___y_1012_, v___y_1013_);
lean_dec(v___y_1013_);
lean_dec_ref(v___y_1012_);
lean_dec(v___y_1011_);
lean_dec_ref(v___y_1010_);
lean_dec_ref(v_x_1009_);
return v_res_1015_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Tree_Tracing_0__Aesop_Goal_traceMetadata_trcNode_spec__0_spec__0(lean_object* v_e_1016_){
_start:
{
if (lean_obj_tag(v_e_1016_) == 0)
{
uint8_t v___x_1017_; 
v___x_1017_ = 2;
return v___x_1017_;
}
else
{
uint8_t v___x_1018_; 
v___x_1018_ = 0;
return v___x_1018_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Tree_Tracing_0__Aesop_Goal_traceMetadata_trcNode_spec__0_spec__0___boxed(lean_object* v_e_1019_){
_start:
{
uint8_t v_res_1020_; lean_object* v_r_1021_; 
v_res_1020_ = lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Tree_Tracing_0__Aesop_Goal_traceMetadata_trcNode_spec__0_spec__0(v_e_1019_);
lean_dec_ref(v_e_1019_);
v_r_1021_ = lean_box(v_res_1020_);
return v_r_1021_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Tree_Tracing_0__Aesop_Goal_traceMetadata_trcNode_spec__0(lean_object* v_cls_1022_, uint8_t v_collapsed_1023_, lean_object* v_tag_1024_, lean_object* v_opts_1025_, uint8_t v_clsEnabled_1026_, lean_object* v_oldTraces_1027_, lean_object* v_msg_1028_, lean_object* v_resStartStop_1029_, lean_object* v___y_1030_, lean_object* v___y_1031_, lean_object* v___y_1032_, lean_object* v___y_1033_){
_start:
{
lean_object* v_fst_1035_; lean_object* v_snd_1036_; lean_object* v___y_1038_; lean_object* v___y_1039_; lean_object* v_data_1040_; lean_object* v_fst_1043_; lean_object* v_snd_1044_; lean_object* v___x_1045_; uint8_t v___x_1046_; lean_object* v___y_1048_; lean_object* v_a_1049_; uint8_t v___y_1064_; double v___y_1095_; 
v_fst_1035_ = lean_ctor_get(v_resStartStop_1029_, 0);
lean_inc(v_fst_1035_);
v_snd_1036_ = lean_ctor_get(v_resStartStop_1029_, 1);
lean_inc(v_snd_1036_);
lean_dec_ref(v_resStartStop_1029_);
v_fst_1043_ = lean_ctor_get(v_snd_1036_, 0);
lean_inc(v_fst_1043_);
v_snd_1044_ = lean_ctor_get(v_snd_1036_, 1);
lean_inc(v_snd_1044_);
lean_dec(v_snd_1036_);
v___x_1045_ = l_Lean_trace_profiler;
v___x_1046_ = lp_aesop_Lean_Option_get___at___00Aesop_Goal_withHeadlineTraceNode_spec__1(v_opts_1025_, v___x_1045_);
if (v___x_1046_ == 0)
{
v___y_1064_ = v___x_1046_;
goto v___jp_1063_;
}
else
{
lean_object* v___x_1100_; uint8_t v___x_1101_; 
v___x_1100_ = l_Lean_trace_profiler_useHeartbeats;
v___x_1101_ = lp_aesop_Lean_Option_get___at___00Aesop_Goal_withHeadlineTraceNode_spec__1(v_opts_1025_, v___x_1100_);
if (v___x_1101_ == 0)
{
lean_object* v___x_1102_; lean_object* v___x_1103_; double v___x_1104_; double v___x_1105_; double v___x_1106_; 
v___x_1102_ = l_Lean_trace_profiler_threshold;
v___x_1103_ = lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Goal_withHeadlineTraceNode_spec__2_spec__5(v_opts_1025_, v___x_1102_);
v___x_1104_ = lean_float_of_nat(v___x_1103_);
v___x_1105_ = lean_float_once(&lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Goal_withHeadlineTraceNode_spec__2___redArg___closed__3, &lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Goal_withHeadlineTraceNode_spec__2___redArg___closed__3_once, _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Goal_withHeadlineTraceNode_spec__2___redArg___closed__3);
v___x_1106_ = lean_float_div(v___x_1104_, v___x_1105_);
v___y_1095_ = v___x_1106_;
goto v___jp_1094_;
}
else
{
lean_object* v___x_1107_; lean_object* v___x_1108_; double v___x_1109_; 
v___x_1107_ = l_Lean_trace_profiler_threshold;
v___x_1108_ = lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Goal_withHeadlineTraceNode_spec__2_spec__5(v_opts_1025_, v___x_1107_);
v___x_1109_ = lean_float_of_nat(v___x_1108_);
v___y_1095_ = v___x_1109_;
goto v___jp_1094_;
}
}
v___jp_1037_:
{
lean_object* v___x_1041_; 
lean_inc(v___y_1038_);
v___x_1041_ = lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Goal_withHeadlineTraceNode_spec__2_spec__2(v_oldTraces_1027_, v_data_1040_, v___y_1038_, v___y_1039_, v___y_1030_, v___y_1031_, v___y_1032_, v___y_1033_);
if (lean_obj_tag(v___x_1041_) == 0)
{
lean_object* v___x_1042_; 
lean_dec_ref_known(v___x_1041_, 1);
v___x_1042_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Goal_withHeadlineTraceNode_spec__2_spec__3___redArg(v_fst_1035_);
return v___x_1042_;
}
else
{
lean_dec(v_fst_1035_);
return v___x_1041_;
}
}
v___jp_1047_:
{
uint8_t v_result_1050_; lean_object* v___x_1051_; lean_object* v___x_1052_; double v___x_1053_; lean_object* v_data_1054_; 
v_result_1050_ = lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Tree_Tracing_0__Aesop_Goal_traceMetadata_trcNode_spec__0_spec__0(v_fst_1035_);
v___x_1051_ = lean_box(v_result_1050_);
v___x_1052_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1052_, 0, v___x_1051_);
v___x_1053_ = lean_float_once(&lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Goal_withHeadlineTraceNode_spec__2___redArg___closed__0, &lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Goal_withHeadlineTraceNode_spec__2___redArg___closed__0_once, _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Goal_withHeadlineTraceNode_spec__2___redArg___closed__0);
lean_inc_ref(v_tag_1024_);
lean_inc_ref(v___x_1052_);
lean_inc(v_cls_1022_);
v_data_1054_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v_data_1054_, 0, v_cls_1022_);
lean_ctor_set(v_data_1054_, 1, v___x_1052_);
lean_ctor_set(v_data_1054_, 2, v_tag_1024_);
lean_ctor_set_float(v_data_1054_, sizeof(void*)*3, v___x_1053_);
lean_ctor_set_float(v_data_1054_, sizeof(void*)*3 + 8, v___x_1053_);
lean_ctor_set_uint8(v_data_1054_, sizeof(void*)*3 + 16, v_collapsed_1023_);
if (v___x_1046_ == 0)
{
lean_dec_ref_known(v___x_1052_, 1);
lean_dec(v_snd_1044_);
lean_dec(v_fst_1043_);
lean_dec_ref(v_tag_1024_);
lean_dec(v_cls_1022_);
v___y_1038_ = v___y_1048_;
v___y_1039_ = v_a_1049_;
v_data_1040_ = v_data_1054_;
goto v___jp_1037_;
}
else
{
lean_object* v_data_1055_; double v___x_1056_; double v___x_1057_; 
lean_dec_ref_known(v_data_1054_, 3);
v_data_1055_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v_data_1055_, 0, v_cls_1022_);
lean_ctor_set(v_data_1055_, 1, v___x_1052_);
lean_ctor_set(v_data_1055_, 2, v_tag_1024_);
v___x_1056_ = lean_unbox_float(v_fst_1043_);
lean_dec(v_fst_1043_);
lean_ctor_set_float(v_data_1055_, sizeof(void*)*3, v___x_1056_);
v___x_1057_ = lean_unbox_float(v_snd_1044_);
lean_dec(v_snd_1044_);
lean_ctor_set_float(v_data_1055_, sizeof(void*)*3 + 8, v___x_1057_);
lean_ctor_set_uint8(v_data_1055_, sizeof(void*)*3 + 16, v_collapsed_1023_);
v___y_1038_ = v___y_1048_;
v___y_1039_ = v_a_1049_;
v_data_1040_ = v_data_1055_;
goto v___jp_1037_;
}
}
v___jp_1058_:
{
lean_object* v_ref_1059_; lean_object* v___x_1060_; 
v_ref_1059_ = lean_ctor_get(v___y_1032_, 5);
lean_inc(v___y_1033_);
lean_inc_ref(v___y_1032_);
lean_inc(v___y_1031_);
lean_inc_ref(v___y_1030_);
lean_inc(v_fst_1035_);
v___x_1060_ = lean_apply_6(v_msg_1028_, v_fst_1035_, v___y_1030_, v___y_1031_, v___y_1032_, v___y_1033_, lean_box(0));
if (lean_obj_tag(v___x_1060_) == 0)
{
lean_object* v_a_1061_; 
v_a_1061_ = lean_ctor_get(v___x_1060_, 0);
lean_inc(v_a_1061_);
lean_dec_ref_known(v___x_1060_, 1);
v___y_1048_ = v_ref_1059_;
v_a_1049_ = v_a_1061_;
goto v___jp_1047_;
}
else
{
lean_object* v___x_1062_; 
lean_dec_ref_known(v___x_1060_, 1);
v___x_1062_ = lean_obj_once(&lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Goal_withHeadlineTraceNode_spec__2___redArg___closed__2, &lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Goal_withHeadlineTraceNode_spec__2___redArg___closed__2_once, _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Goal_withHeadlineTraceNode_spec__2___redArg___closed__2);
v___y_1048_ = v_ref_1059_;
v_a_1049_ = v___x_1062_;
goto v___jp_1047_;
}
}
v___jp_1063_:
{
if (v_clsEnabled_1026_ == 0)
{
if (v___y_1064_ == 0)
{
lean_object* v___x_1065_; lean_object* v_traceState_1066_; lean_object* v_env_1067_; lean_object* v_nextMacroScope_1068_; lean_object* v_ngen_1069_; lean_object* v_auxDeclNGen_1070_; lean_object* v_cache_1071_; lean_object* v_messages_1072_; lean_object* v_infoState_1073_; lean_object* v_snapshotTasks_1074_; lean_object* v___x_1076_; uint8_t v_isShared_1077_; uint8_t v_isSharedCheck_1093_; 
lean_dec(v_snd_1044_);
lean_dec(v_fst_1043_);
lean_dec_ref(v_msg_1028_);
lean_dec_ref(v_tag_1024_);
lean_dec(v_cls_1022_);
v___x_1065_ = lean_st_ref_take(v___y_1033_);
v_traceState_1066_ = lean_ctor_get(v___x_1065_, 4);
v_env_1067_ = lean_ctor_get(v___x_1065_, 0);
v_nextMacroScope_1068_ = lean_ctor_get(v___x_1065_, 1);
v_ngen_1069_ = lean_ctor_get(v___x_1065_, 2);
v_auxDeclNGen_1070_ = lean_ctor_get(v___x_1065_, 3);
v_cache_1071_ = lean_ctor_get(v___x_1065_, 5);
v_messages_1072_ = lean_ctor_get(v___x_1065_, 6);
v_infoState_1073_ = lean_ctor_get(v___x_1065_, 7);
v_snapshotTasks_1074_ = lean_ctor_get(v___x_1065_, 8);
v_isSharedCheck_1093_ = !lean_is_exclusive(v___x_1065_);
if (v_isSharedCheck_1093_ == 0)
{
v___x_1076_ = v___x_1065_;
v_isShared_1077_ = v_isSharedCheck_1093_;
goto v_resetjp_1075_;
}
else
{
lean_inc(v_snapshotTasks_1074_);
lean_inc(v_infoState_1073_);
lean_inc(v_messages_1072_);
lean_inc(v_cache_1071_);
lean_inc(v_traceState_1066_);
lean_inc(v_auxDeclNGen_1070_);
lean_inc(v_ngen_1069_);
lean_inc(v_nextMacroScope_1068_);
lean_inc(v_env_1067_);
lean_dec(v___x_1065_);
v___x_1076_ = lean_box(0);
v_isShared_1077_ = v_isSharedCheck_1093_;
goto v_resetjp_1075_;
}
v_resetjp_1075_:
{
uint64_t v_tid_1078_; lean_object* v_traces_1079_; lean_object* v___x_1081_; uint8_t v_isShared_1082_; uint8_t v_isSharedCheck_1092_; 
v_tid_1078_ = lean_ctor_get_uint64(v_traceState_1066_, sizeof(void*)*1);
v_traces_1079_ = lean_ctor_get(v_traceState_1066_, 0);
v_isSharedCheck_1092_ = !lean_is_exclusive(v_traceState_1066_);
if (v_isSharedCheck_1092_ == 0)
{
v___x_1081_ = v_traceState_1066_;
v_isShared_1082_ = v_isSharedCheck_1092_;
goto v_resetjp_1080_;
}
else
{
lean_inc(v_traces_1079_);
lean_dec(v_traceState_1066_);
v___x_1081_ = lean_box(0);
v_isShared_1082_ = v_isSharedCheck_1092_;
goto v_resetjp_1080_;
}
v_resetjp_1080_:
{
lean_object* v___x_1083_; lean_object* v___x_1085_; 
v___x_1083_ = l_Lean_PersistentArray_append___redArg(v_oldTraces_1027_, v_traces_1079_);
lean_dec_ref(v_traces_1079_);
if (v_isShared_1082_ == 0)
{
lean_ctor_set(v___x_1081_, 0, v___x_1083_);
v___x_1085_ = v___x_1081_;
goto v_reusejp_1084_;
}
else
{
lean_object* v_reuseFailAlloc_1091_; 
v_reuseFailAlloc_1091_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_1091_, 0, v___x_1083_);
lean_ctor_set_uint64(v_reuseFailAlloc_1091_, sizeof(void*)*1, v_tid_1078_);
v___x_1085_ = v_reuseFailAlloc_1091_;
goto v_reusejp_1084_;
}
v_reusejp_1084_:
{
lean_object* v___x_1087_; 
if (v_isShared_1077_ == 0)
{
lean_ctor_set(v___x_1076_, 4, v___x_1085_);
v___x_1087_ = v___x_1076_;
goto v_reusejp_1086_;
}
else
{
lean_object* v_reuseFailAlloc_1090_; 
v_reuseFailAlloc_1090_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_1090_, 0, v_env_1067_);
lean_ctor_set(v_reuseFailAlloc_1090_, 1, v_nextMacroScope_1068_);
lean_ctor_set(v_reuseFailAlloc_1090_, 2, v_ngen_1069_);
lean_ctor_set(v_reuseFailAlloc_1090_, 3, v_auxDeclNGen_1070_);
lean_ctor_set(v_reuseFailAlloc_1090_, 4, v___x_1085_);
lean_ctor_set(v_reuseFailAlloc_1090_, 5, v_cache_1071_);
lean_ctor_set(v_reuseFailAlloc_1090_, 6, v_messages_1072_);
lean_ctor_set(v_reuseFailAlloc_1090_, 7, v_infoState_1073_);
lean_ctor_set(v_reuseFailAlloc_1090_, 8, v_snapshotTasks_1074_);
v___x_1087_ = v_reuseFailAlloc_1090_;
goto v_reusejp_1086_;
}
v_reusejp_1086_:
{
lean_object* v___x_1088_; lean_object* v___x_1089_; 
v___x_1088_ = lean_st_ref_set(v___y_1033_, v___x_1087_);
v___x_1089_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Goal_withHeadlineTraceNode_spec__2_spec__3___redArg(v_fst_1035_);
return v___x_1089_;
}
}
}
}
}
else
{
goto v___jp_1058_;
}
}
else
{
goto v___jp_1058_;
}
}
v___jp_1094_:
{
double v___x_1096_; double v___x_1097_; double v___x_1098_; uint8_t v___x_1099_; 
v___x_1096_ = lean_unbox_float(v_snd_1044_);
v___x_1097_ = lean_unbox_float(v_fst_1043_);
v___x_1098_ = lean_float_sub(v___x_1096_, v___x_1097_);
v___x_1099_ = lean_float_decLt(v___y_1095_, v___x_1098_);
v___y_1064_ = v___x_1099_;
goto v___jp_1063_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Tree_Tracing_0__Aesop_Goal_traceMetadata_trcNode_spec__0___boxed(lean_object* v_cls_1110_, lean_object* v_collapsed_1111_, lean_object* v_tag_1112_, lean_object* v_opts_1113_, lean_object* v_clsEnabled_1114_, lean_object* v_oldTraces_1115_, lean_object* v_msg_1116_, lean_object* v_resStartStop_1117_, lean_object* v___y_1118_, lean_object* v___y_1119_, lean_object* v___y_1120_, lean_object* v___y_1121_, lean_object* v___y_1122_){
_start:
{
uint8_t v_collapsed_boxed_1123_; uint8_t v_clsEnabled_boxed_1124_; lean_object* v_res_1125_; 
v_collapsed_boxed_1123_ = lean_unbox(v_collapsed_1111_);
v_clsEnabled_boxed_1124_ = lean_unbox(v_clsEnabled_1114_);
v_res_1125_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Tree_Tracing_0__Aesop_Goal_traceMetadata_trcNode_spec__0(v_cls_1110_, v_collapsed_boxed_1123_, v_tag_1112_, v_opts_1113_, v_clsEnabled_boxed_1124_, v_oldTraces_1115_, v_msg_1116_, v_resStartStop_1117_, v___y_1118_, v___y_1119_, v___y_1120_, v___y_1121_);
lean_dec(v___y_1121_);
lean_dec_ref(v___y_1120_);
lean_dec(v___y_1119_);
lean_dec_ref(v___y_1118_);
lean_dec_ref(v_opts_1113_);
return v_res_1125_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Goal_traceMetadata_trcNode(lean_object* v_traceOpt_1126_, lean_object* v_msg_1127_, lean_object* v_k_1128_, lean_object* v_a_1129_, lean_object* v_a_1130_, lean_object* v_a_1131_, lean_object* v_a_1132_){
_start:
{
lean_object* v_options_1134_; uint8_t v_hasTrace_1135_; 
v_options_1134_ = lean_ctor_get(v_a_1131_, 2);
v_hasTrace_1135_ = lean_ctor_get_uint8(v_options_1134_, sizeof(void*)*1);
if (v_hasTrace_1135_ == 0)
{
lean_object* v___x_1136_; 
lean_dec_ref(v_msg_1127_);
lean_dec_ref(v_traceOpt_1126_);
lean_inc(v_a_1132_);
lean_inc_ref(v_a_1131_);
lean_inc(v_a_1130_);
lean_inc_ref(v_a_1129_);
v___x_1136_ = lean_apply_5(v_k_1128_, v_a_1129_, v_a_1130_, v_a_1131_, v_a_1132_, lean_box(0));
return v___x_1136_;
}
else
{
lean_object* v_inheritedTraceOptions_1137_; lean_object* v_traceClass_1138_; lean_object* v___x_1140_; uint8_t v_isShared_1141_; uint8_t v_isSharedCheck_1220_; 
v_inheritedTraceOptions_1137_ = lean_ctor_get(v_a_1131_, 13);
v_traceClass_1138_ = lean_ctor_get(v_traceOpt_1126_, 0);
v_isSharedCheck_1220_ = !lean_is_exclusive(v_traceOpt_1126_);
if (v_isSharedCheck_1220_ == 0)
{
lean_object* v_unused_1221_; 
v_unused_1221_ = lean_ctor_get(v_traceOpt_1126_, 1);
lean_dec(v_unused_1221_);
v___x_1140_ = v_traceOpt_1126_;
v_isShared_1141_ = v_isSharedCheck_1220_;
goto v_resetjp_1139_;
}
else
{
lean_inc(v_traceClass_1138_);
lean_dec(v_traceOpt_1126_);
v___x_1140_ = lean_box(0);
v_isShared_1141_ = v_isSharedCheck_1220_;
goto v_resetjp_1139_;
}
v_resetjp_1139_:
{
lean_object* v___f_1142_; lean_object* v___x_1143_; lean_object* v___x_1144_; lean_object* v___x_1145_; uint8_t v___x_1146_; lean_object* v___y_1148_; lean_object* v___y_1149_; lean_object* v_a_1150_; lean_object* v___y_1165_; lean_object* v___y_1166_; lean_object* v_a_1167_; 
v___f_1142_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Goal_traceMetadata_trcNode___lam__0___boxed), 7, 1);
lean_closure_set(v___f_1142_, 0, v_msg_1127_);
v___x_1143_ = ((lean_object*)(lp_aesop_Aesop_Goal_withHeadlineTraceNode___redArg___closed__0));
v___x_1144_ = ((lean_object*)(lp_aesop_Aesop_Goal_withHeadlineTraceNode___redArg___closed__2));
lean_inc(v_traceClass_1138_);
v___x_1145_ = l_Lean_Name_append(v___x_1144_, v_traceClass_1138_);
v___x_1146_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_1137_, v_options_1134_, v___x_1145_);
lean_dec(v___x_1145_);
if (v___x_1146_ == 0)
{
lean_object* v___x_1217_; uint8_t v___x_1218_; 
v___x_1217_ = l_Lean_trace_profiler;
v___x_1218_ = lp_aesop_Lean_Option_get___at___00Aesop_Goal_withHeadlineTraceNode_spec__1(v_options_1134_, v___x_1217_);
if (v___x_1218_ == 0)
{
lean_object* v___x_1219_; 
lean_dec_ref(v___f_1142_);
lean_del_object(v___x_1140_);
lean_dec(v_traceClass_1138_);
lean_inc(v_a_1132_);
lean_inc_ref(v_a_1131_);
lean_inc(v_a_1130_);
lean_inc_ref(v_a_1129_);
v___x_1219_ = lean_apply_5(v_k_1128_, v_a_1129_, v_a_1130_, v_a_1131_, v_a_1132_, lean_box(0));
return v___x_1219_;
}
else
{
goto v___jp_1176_;
}
}
else
{
goto v___jp_1176_;
}
v___jp_1147_:
{
lean_object* v___x_1151_; double v___x_1152_; double v___x_1153_; double v___x_1154_; double v___x_1155_; double v___x_1156_; lean_object* v___x_1157_; lean_object* v___x_1158_; lean_object* v___x_1160_; 
v___x_1151_ = lean_io_mono_nanos_now();
v___x_1152_ = lean_float_of_nat(v___y_1148_);
v___x_1153_ = lean_float_once(&lp_aesop_Aesop_Goal_withHeadlineTraceNode___redArg___closed__3, &lp_aesop_Aesop_Goal_withHeadlineTraceNode___redArg___closed__3_once, _init_lp_aesop_Aesop_Goal_withHeadlineTraceNode___redArg___closed__3);
v___x_1154_ = lean_float_div(v___x_1152_, v___x_1153_);
v___x_1155_ = lean_float_of_nat(v___x_1151_);
v___x_1156_ = lean_float_div(v___x_1155_, v___x_1153_);
v___x_1157_ = lean_box_float(v___x_1154_);
v___x_1158_ = lean_box_float(v___x_1156_);
if (v_isShared_1141_ == 0)
{
lean_ctor_set(v___x_1140_, 1, v___x_1158_);
lean_ctor_set(v___x_1140_, 0, v___x_1157_);
v___x_1160_ = v___x_1140_;
goto v_reusejp_1159_;
}
else
{
lean_object* v_reuseFailAlloc_1163_; 
v_reuseFailAlloc_1163_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1163_, 0, v___x_1157_);
lean_ctor_set(v_reuseFailAlloc_1163_, 1, v___x_1158_);
v___x_1160_ = v_reuseFailAlloc_1163_;
goto v_reusejp_1159_;
}
v_reusejp_1159_:
{
lean_object* v___x_1161_; lean_object* v___x_1162_; 
v___x_1161_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1161_, 0, v_a_1150_);
lean_ctor_set(v___x_1161_, 1, v___x_1160_);
v___x_1162_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Tree_Tracing_0__Aesop_Goal_traceMetadata_trcNode_spec__0(v_traceClass_1138_, v_hasTrace_1135_, v___x_1143_, v_options_1134_, v___x_1146_, v___y_1149_, v___f_1142_, v___x_1161_, v_a_1129_, v_a_1130_, v_a_1131_, v_a_1132_);
return v___x_1162_;
}
}
v___jp_1164_:
{
lean_object* v___x_1168_; double v___x_1169_; double v___x_1170_; lean_object* v___x_1171_; lean_object* v___x_1172_; lean_object* v___x_1173_; lean_object* v___x_1174_; lean_object* v___x_1175_; 
v___x_1168_ = lean_io_get_num_heartbeats();
v___x_1169_ = lean_float_of_nat(v___y_1166_);
v___x_1170_ = lean_float_of_nat(v___x_1168_);
v___x_1171_ = lean_box_float(v___x_1169_);
v___x_1172_ = lean_box_float(v___x_1170_);
v___x_1173_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1173_, 0, v___x_1171_);
lean_ctor_set(v___x_1173_, 1, v___x_1172_);
v___x_1174_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1174_, 0, v_a_1167_);
lean_ctor_set(v___x_1174_, 1, v___x_1173_);
v___x_1175_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Tree_Tracing_0__Aesop_Goal_traceMetadata_trcNode_spec__0(v_traceClass_1138_, v_hasTrace_1135_, v___x_1143_, v_options_1134_, v___x_1146_, v___y_1165_, v___f_1142_, v___x_1174_, v_a_1129_, v_a_1130_, v_a_1131_, v_a_1132_);
return v___x_1175_;
}
v___jp_1176_:
{
lean_object* v___x_1177_; lean_object* v_a_1178_; lean_object* v___x_1179_; uint8_t v___x_1180_; 
v___x_1177_ = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_Goal_withHeadlineTraceNode_spec__0___redArg(v_a_1132_);
v_a_1178_ = lean_ctor_get(v___x_1177_, 0);
lean_inc(v_a_1178_);
lean_dec_ref(v___x_1177_);
v___x_1179_ = l_Lean_trace_profiler_useHeartbeats;
v___x_1180_ = lp_aesop_Lean_Option_get___at___00Aesop_Goal_withHeadlineTraceNode_spec__1(v_options_1134_, v___x_1179_);
if (v___x_1180_ == 0)
{
lean_object* v___x_1181_; lean_object* v___x_1182_; 
v___x_1181_ = lean_io_mono_nanos_now();
lean_inc(v_a_1132_);
lean_inc_ref(v_a_1131_);
lean_inc(v_a_1130_);
lean_inc_ref(v_a_1129_);
v___x_1182_ = lean_apply_5(v_k_1128_, v_a_1129_, v_a_1130_, v_a_1131_, v_a_1132_, lean_box(0));
if (lean_obj_tag(v___x_1182_) == 0)
{
lean_object* v_a_1183_; lean_object* v___x_1185_; uint8_t v_isShared_1186_; uint8_t v_isSharedCheck_1190_; 
v_a_1183_ = lean_ctor_get(v___x_1182_, 0);
v_isSharedCheck_1190_ = !lean_is_exclusive(v___x_1182_);
if (v_isSharedCheck_1190_ == 0)
{
v___x_1185_ = v___x_1182_;
v_isShared_1186_ = v_isSharedCheck_1190_;
goto v_resetjp_1184_;
}
else
{
lean_inc(v_a_1183_);
lean_dec(v___x_1182_);
v___x_1185_ = lean_box(0);
v_isShared_1186_ = v_isSharedCheck_1190_;
goto v_resetjp_1184_;
}
v_resetjp_1184_:
{
lean_object* v___x_1188_; 
if (v_isShared_1186_ == 0)
{
lean_ctor_set_tag(v___x_1185_, 1);
v___x_1188_ = v___x_1185_;
goto v_reusejp_1187_;
}
else
{
lean_object* v_reuseFailAlloc_1189_; 
v_reuseFailAlloc_1189_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1189_, 0, v_a_1183_);
v___x_1188_ = v_reuseFailAlloc_1189_;
goto v_reusejp_1187_;
}
v_reusejp_1187_:
{
v___y_1148_ = v___x_1181_;
v___y_1149_ = v_a_1178_;
v_a_1150_ = v___x_1188_;
goto v___jp_1147_;
}
}
}
else
{
lean_object* v_a_1191_; lean_object* v___x_1193_; uint8_t v_isShared_1194_; uint8_t v_isSharedCheck_1198_; 
v_a_1191_ = lean_ctor_get(v___x_1182_, 0);
v_isSharedCheck_1198_ = !lean_is_exclusive(v___x_1182_);
if (v_isSharedCheck_1198_ == 0)
{
v___x_1193_ = v___x_1182_;
v_isShared_1194_ = v_isSharedCheck_1198_;
goto v_resetjp_1192_;
}
else
{
lean_inc(v_a_1191_);
lean_dec(v___x_1182_);
v___x_1193_ = lean_box(0);
v_isShared_1194_ = v_isSharedCheck_1198_;
goto v_resetjp_1192_;
}
v_resetjp_1192_:
{
lean_object* v___x_1196_; 
if (v_isShared_1194_ == 0)
{
lean_ctor_set_tag(v___x_1193_, 0);
v___x_1196_ = v___x_1193_;
goto v_reusejp_1195_;
}
else
{
lean_object* v_reuseFailAlloc_1197_; 
v_reuseFailAlloc_1197_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1197_, 0, v_a_1191_);
v___x_1196_ = v_reuseFailAlloc_1197_;
goto v_reusejp_1195_;
}
v_reusejp_1195_:
{
v___y_1148_ = v___x_1181_;
v___y_1149_ = v_a_1178_;
v_a_1150_ = v___x_1196_;
goto v___jp_1147_;
}
}
}
}
else
{
lean_object* v___x_1199_; lean_object* v___x_1200_; 
lean_del_object(v___x_1140_);
v___x_1199_ = lean_io_get_num_heartbeats();
lean_inc(v_a_1132_);
lean_inc_ref(v_a_1131_);
lean_inc(v_a_1130_);
lean_inc_ref(v_a_1129_);
v___x_1200_ = lean_apply_5(v_k_1128_, v_a_1129_, v_a_1130_, v_a_1131_, v_a_1132_, lean_box(0));
if (lean_obj_tag(v___x_1200_) == 0)
{
lean_object* v_a_1201_; lean_object* v___x_1203_; uint8_t v_isShared_1204_; uint8_t v_isSharedCheck_1208_; 
v_a_1201_ = lean_ctor_get(v___x_1200_, 0);
v_isSharedCheck_1208_ = !lean_is_exclusive(v___x_1200_);
if (v_isSharedCheck_1208_ == 0)
{
v___x_1203_ = v___x_1200_;
v_isShared_1204_ = v_isSharedCheck_1208_;
goto v_resetjp_1202_;
}
else
{
lean_inc(v_a_1201_);
lean_dec(v___x_1200_);
v___x_1203_ = lean_box(0);
v_isShared_1204_ = v_isSharedCheck_1208_;
goto v_resetjp_1202_;
}
v_resetjp_1202_:
{
lean_object* v___x_1206_; 
if (v_isShared_1204_ == 0)
{
lean_ctor_set_tag(v___x_1203_, 1);
v___x_1206_ = v___x_1203_;
goto v_reusejp_1205_;
}
else
{
lean_object* v_reuseFailAlloc_1207_; 
v_reuseFailAlloc_1207_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1207_, 0, v_a_1201_);
v___x_1206_ = v_reuseFailAlloc_1207_;
goto v_reusejp_1205_;
}
v_reusejp_1205_:
{
v___y_1165_ = v_a_1178_;
v___y_1166_ = v___x_1199_;
v_a_1167_ = v___x_1206_;
goto v___jp_1164_;
}
}
}
else
{
lean_object* v_a_1209_; lean_object* v___x_1211_; uint8_t v_isShared_1212_; uint8_t v_isSharedCheck_1216_; 
v_a_1209_ = lean_ctor_get(v___x_1200_, 0);
v_isSharedCheck_1216_ = !lean_is_exclusive(v___x_1200_);
if (v_isSharedCheck_1216_ == 0)
{
v___x_1211_ = v___x_1200_;
v_isShared_1212_ = v_isSharedCheck_1216_;
goto v_resetjp_1210_;
}
else
{
lean_inc(v_a_1209_);
lean_dec(v___x_1200_);
v___x_1211_ = lean_box(0);
v_isShared_1212_ = v_isSharedCheck_1216_;
goto v_resetjp_1210_;
}
v_resetjp_1210_:
{
lean_object* v___x_1214_; 
if (v_isShared_1212_ == 0)
{
lean_ctor_set_tag(v___x_1211_, 0);
v___x_1214_ = v___x_1211_;
goto v_reusejp_1213_;
}
else
{
lean_object* v_reuseFailAlloc_1215_; 
v_reuseFailAlloc_1215_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1215_, 0, v_a_1209_);
v___x_1214_ = v_reuseFailAlloc_1215_;
goto v_reusejp_1213_;
}
v_reusejp_1213_:
{
v___y_1165_ = v_a_1178_;
v___y_1166_ = v___x_1199_;
v_a_1167_ = v___x_1214_;
goto v___jp_1164_;
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Goal_traceMetadata_trcNode___boxed(lean_object* v_traceOpt_1222_, lean_object* v_msg_1223_, lean_object* v_k_1224_, lean_object* v_a_1225_, lean_object* v_a_1226_, lean_object* v_a_1227_, lean_object* v_a_1228_, lean_object* v_a_1229_){
_start:
{
lean_object* v_res_1230_; 
v_res_1230_ = lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Goal_traceMetadata_trcNode(v_traceOpt_1222_, v_msg_1223_, v_k_1224_, v_a_1225_, v_a_1226_, v_a_1227_, v_a_1228_);
lean_dec(v_a_1228_);
lean_dec_ref(v_a_1227_);
lean_dec(v_a_1226_);
lean_dec_ref(v_a_1225_);
return v_res_1230_;
}
}
static lean_object* _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__1(void){
_start:
{
lean_object* v___x_1232_; lean_object* v___x_1233_; 
v___x_1232_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__0));
v___x_1233_ = l_Lean_stringToMessageData(v___x_1232_);
return v___x_1233_;
}
}
static lean_object* _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__3(void){
_start:
{
lean_object* v___x_1235_; lean_object* v___x_1236_; 
v___x_1235_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__2));
v___x_1236_ = l_Lean_stringToMessageData(v___x_1235_);
return v___x_1236_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6(uint8_t v_a_1251_, lean_object* v_traceOpt_1252_, lean_object* v_as_1253_, size_t v_sz_1254_, size_t v_i_1255_, lean_object* v_b_1256_, lean_object* v___y_1257_, lean_object* v___y_1258_, lean_object* v___y_1259_, lean_object* v___y_1260_){
_start:
{
uint8_t v___x_1262_; 
v___x_1262_ = lean_usize_dec_lt(v_i_1255_, v_sz_1254_);
if (v___x_1262_ == 0)
{
lean_object* v___x_1263_; 
lean_dec_ref(v_traceOpt_1252_);
v___x_1263_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1263_, 0, v_b_1256_);
return v___x_1263_;
}
else
{
lean_object* v_a_1264_; lean_object* v___x_1265_; uint8_t v_phase_1266_; lean_object* v___x_1267_; lean_object* v___x_1268_; double v___x_1269_; lean_object* v___x_1270_; lean_object* v___x_1271_; lean_object* v___x_1272_; lean_object* v___x_1273_; lean_object* v___x_1274_; lean_object* v___y_1276_; lean_object* v___y_1277_; lean_object* v___y_1278_; lean_object* v___y_1292_; lean_object* v___y_1293_; lean_object* v___y_1294_; lean_object* v___y_1301_; 
v_a_1264_ = lean_array_uget_borrowed(v_as_1253_, v_i_1255_);
v___x_1265_ = lp_aesop_Aesop_RegularRule_name(v_a_1264_);
v_phase_1266_ = lean_ctor_get_uint8(v___x_1265_, sizeof(void*)*1 + 9);
v___x_1267_ = lean_box(0);
v___x_1268_ = lean_obj_once(&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__1, &lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__1_once, _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__1);
v___x_1269_ = lp_aesop_Aesop_RegularRule_successProbability(v_a_1264_);
v___x_1270_ = lp_aesop_Aesop_Percent_toHumanString(v___x_1269_);
v___x_1271_ = l_Lean_stringToMessageData(v___x_1270_);
v___x_1272_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1272_, 0, v___x_1268_);
lean_ctor_set(v___x_1272_, 1, v___x_1271_);
v___x_1273_ = lean_obj_once(&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__3, &lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__3_once, _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__3);
v___x_1274_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1274_, 0, v___x_1272_);
lean_ctor_set(v___x_1274_, 1, v___x_1273_);
switch(v_phase_1266_)
{
case 0:
{
lean_object* v___x_1313_; 
v___x_1313_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__15));
v___y_1301_ = v___x_1313_;
goto v___jp_1300_;
}
case 1:
{
lean_object* v___x_1314_; 
v___x_1314_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__16));
v___y_1301_ = v___x_1314_;
goto v___jp_1300_;
}
default: 
{
lean_object* v___x_1315_; 
v___x_1315_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__17));
v___y_1301_ = v___x_1315_;
goto v___jp_1300_;
}
}
v___jp_1275_:
{
lean_object* v_name_1279_; lean_object* v___x_1280_; lean_object* v___x_1281_; lean_object* v___x_1282_; lean_object* v___x_1283_; lean_object* v___x_1284_; lean_object* v___x_1285_; lean_object* v___x_1286_; lean_object* v___x_1287_; 
v_name_1279_ = lean_ctor_get(v___x_1265_, 0);
lean_inc(v_name_1279_);
lean_dec_ref(v___x_1265_);
v___x_1280_ = lean_string_append(v___y_1276_, v___y_1278_);
v___x_1281_ = lean_string_append(v___x_1280_, v___y_1277_);
v___x_1282_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_name_1279_, v_a_1251_);
v___x_1283_ = lean_string_append(v___x_1281_, v___x_1282_);
lean_dec_ref(v___x_1282_);
v___x_1284_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1284_, 0, v___x_1283_);
v___x_1285_ = l_Lean_MessageData_ofFormat(v___x_1284_);
v___x_1286_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1286_, 0, v___x_1274_);
lean_ctor_set(v___x_1286_, 1, v___x_1285_);
lean_inc_ref(v_traceOpt_1252_);
v___x_1287_ = lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Goal_traceMetadata_trc(v_traceOpt_1252_, v___x_1286_, v___y_1257_, v___y_1258_, v___y_1259_, v___y_1260_);
if (lean_obj_tag(v___x_1287_) == 0)
{
size_t v___x_1288_; size_t v___x_1289_; 
lean_dec_ref_known(v___x_1287_, 1);
v___x_1288_ = ((size_t)1ULL);
v___x_1289_ = lean_usize_add(v_i_1255_, v___x_1288_);
v_i_1255_ = v___x_1289_;
v_b_1256_ = v___x_1267_;
goto _start;
}
else
{
lean_dec_ref(v_traceOpt_1252_);
return v___x_1287_;
}
}
v___jp_1291_:
{
uint8_t v_scope_1295_; lean_object* v___x_1296_; lean_object* v___x_1297_; 
v_scope_1295_ = lean_ctor_get_uint8(v___x_1265_, sizeof(void*)*1 + 10);
v___x_1296_ = lean_string_append(v___y_1293_, v___y_1294_);
v___x_1297_ = lean_string_append(v___x_1296_, v___y_1292_);
if (v_scope_1295_ == 0)
{
lean_object* v___x_1298_; 
v___x_1298_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__4));
v___y_1276_ = v___x_1297_;
v___y_1277_ = v___y_1292_;
v___y_1278_ = v___x_1298_;
goto v___jp_1275_;
}
else
{
lean_object* v___x_1299_; 
v___x_1299_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__5));
v___y_1276_ = v___x_1297_;
v___y_1277_ = v___y_1292_;
v___y_1278_ = v___x_1299_;
goto v___jp_1275_;
}
}
v___jp_1300_:
{
uint8_t v_builder_1302_; lean_object* v___x_1303_; lean_object* v___x_1304_; 
v_builder_1302_ = lean_ctor_get_uint8(v___x_1265_, sizeof(void*)*1 + 8);
v___x_1303_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__6));
lean_inc_ref(v___y_1301_);
v___x_1304_ = lean_string_append(v___y_1301_, v___x_1303_);
switch(v_builder_1302_)
{
case 0:
{
lean_object* v___x_1305_; 
v___x_1305_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__7));
v___y_1292_ = v___x_1303_;
v___y_1293_ = v___x_1304_;
v___y_1294_ = v___x_1305_;
goto v___jp_1291_;
}
case 1:
{
lean_object* v___x_1306_; 
v___x_1306_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__8));
v___y_1292_ = v___x_1303_;
v___y_1293_ = v___x_1304_;
v___y_1294_ = v___x_1306_;
goto v___jp_1291_;
}
case 2:
{
lean_object* v___x_1307_; 
v___x_1307_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__9));
v___y_1292_ = v___x_1303_;
v___y_1293_ = v___x_1304_;
v___y_1294_ = v___x_1307_;
goto v___jp_1291_;
}
case 3:
{
lean_object* v___x_1308_; 
v___x_1308_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__10));
v___y_1292_ = v___x_1303_;
v___y_1293_ = v___x_1304_;
v___y_1294_ = v___x_1308_;
goto v___jp_1291_;
}
case 4:
{
lean_object* v___x_1309_; 
v___x_1309_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__11));
v___y_1292_ = v___x_1303_;
v___y_1293_ = v___x_1304_;
v___y_1294_ = v___x_1309_;
goto v___jp_1291_;
}
case 5:
{
lean_object* v___x_1310_; 
v___x_1310_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__12));
v___y_1292_ = v___x_1303_;
v___y_1293_ = v___x_1304_;
v___y_1294_ = v___x_1310_;
goto v___jp_1291_;
}
case 6:
{
lean_object* v___x_1311_; 
v___x_1311_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__13));
v___y_1292_ = v___x_1303_;
v___y_1293_ = v___x_1304_;
v___y_1294_ = v___x_1311_;
goto v___jp_1291_;
}
default: 
{
lean_object* v___x_1312_; 
v___x_1312_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__14));
v___y_1292_ = v___x_1303_;
v___y_1293_ = v___x_1304_;
v___y_1294_ = v___x_1312_;
goto v___jp_1291_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___boxed(lean_object* v_a_1316_, lean_object* v_traceOpt_1317_, lean_object* v_as_1318_, lean_object* v_sz_1319_, lean_object* v_i_1320_, lean_object* v_b_1321_, lean_object* v___y_1322_, lean_object* v___y_1323_, lean_object* v___y_1324_, lean_object* v___y_1325_, lean_object* v___y_1326_){
_start:
{
uint8_t v_a_51579__boxed_1327_; size_t v_sz_boxed_1328_; size_t v_i_boxed_1329_; lean_object* v_res_1330_; 
v_a_51579__boxed_1327_ = lean_unbox(v_a_1316_);
v_sz_boxed_1328_ = lean_unbox_usize(v_sz_1319_);
lean_dec(v_sz_1319_);
v_i_boxed_1329_ = lean_unbox_usize(v_i_1320_);
lean_dec(v_i_1320_);
v_res_1330_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6(v_a_51579__boxed_1327_, v_traceOpt_1317_, v_as_1318_, v_sz_boxed_1328_, v_i_boxed_1329_, v_b_1321_, v___y_1322_, v___y_1323_, v___y_1324_, v___y_1325_);
lean_dec(v___y_1325_);
lean_dec_ref(v___y_1324_);
lean_dec(v___y_1323_);
lean_dec_ref(v___y_1322_);
lean_dec_ref(v_as_1318_);
return v_res_1330_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_traceMetadata___lam__0(uint8_t v_a_1331_, lean_object* v_traceOpt_1332_, lean_object* v_failedRapps_1333_, size_t v_sz_1334_, size_t v___x_1335_, lean_object* v___x_1336_, lean_object* v___y_1337_, lean_object* v___y_1338_, lean_object* v___y_1339_, lean_object* v___y_1340_){
_start:
{
lean_object* v___x_1342_; 
v___x_1342_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6(v_a_1331_, v_traceOpt_1332_, v_failedRapps_1333_, v_sz_1334_, v___x_1335_, v___x_1336_, v___y_1337_, v___y_1338_, v___y_1339_, v___y_1340_);
if (lean_obj_tag(v___x_1342_) == 0)
{
lean_object* v___x_1344_; uint8_t v_isShared_1345_; uint8_t v_isSharedCheck_1349_; 
v_isSharedCheck_1349_ = !lean_is_exclusive(v___x_1342_);
if (v_isSharedCheck_1349_ == 0)
{
lean_object* v_unused_1350_; 
v_unused_1350_ = lean_ctor_get(v___x_1342_, 0);
lean_dec(v_unused_1350_);
v___x_1344_ = v___x_1342_;
v_isShared_1345_ = v_isSharedCheck_1349_;
goto v_resetjp_1343_;
}
else
{
lean_dec(v___x_1342_);
v___x_1344_ = lean_box(0);
v_isShared_1345_ = v_isSharedCheck_1349_;
goto v_resetjp_1343_;
}
v_resetjp_1343_:
{
lean_object* v___x_1347_; 
if (v_isShared_1345_ == 0)
{
lean_ctor_set(v___x_1344_, 0, v___x_1336_);
v___x_1347_ = v___x_1344_;
goto v_reusejp_1346_;
}
else
{
lean_object* v_reuseFailAlloc_1348_; 
v_reuseFailAlloc_1348_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1348_, 0, v___x_1336_);
v___x_1347_ = v_reuseFailAlloc_1348_;
goto v_reusejp_1346_;
}
v_reusejp_1346_:
{
return v___x_1347_;
}
}
}
else
{
return v___x_1342_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_traceMetadata___lam__0___boxed(lean_object* v_a_1351_, lean_object* v_traceOpt_1352_, lean_object* v_failedRapps_1353_, lean_object* v_sz_1354_, lean_object* v___x_1355_, lean_object* v___x_1356_, lean_object* v___y_1357_, lean_object* v___y_1358_, lean_object* v___y_1359_, lean_object* v___y_1360_, lean_object* v___y_1361_){
_start:
{
uint8_t v_a_51717__boxed_1362_; size_t v_sz_boxed_1363_; size_t v___x_51718__boxed_1364_; lean_object* v_res_1365_; 
v_a_51717__boxed_1362_ = lean_unbox(v_a_1351_);
v_sz_boxed_1363_ = lean_unbox_usize(v_sz_1354_);
lean_dec(v_sz_1354_);
v___x_51718__boxed_1364_ = lean_unbox_usize(v___x_1355_);
lean_dec(v___x_1355_);
v_res_1365_ = lp_aesop_Aesop_Goal_traceMetadata___lam__0(v_a_51717__boxed_1362_, v_traceOpt_1352_, v_failedRapps_1353_, v_sz_boxed_1363_, v___x_51718__boxed_1364_, v___x_1356_, v___y_1357_, v___y_1358_, v___y_1359_, v___y_1360_);
lean_dec(v___y_1360_);
lean_dec_ref(v___y_1359_);
lean_dec(v___y_1358_);
lean_dec_ref(v___y_1357_);
lean_dec_ref(v_failedRapps_1353_);
return v_res_1365_;
}
}
static lean_object* _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Aesop_Goal_traceMetadata_spec__1___redArg___closed__1(void){
_start:
{
lean_object* v___x_1367_; lean_object* v___x_1368_; 
v___x_1367_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Aesop_Goal_traceMetadata_spec__1___redArg___closed__0));
v___x_1368_ = l_Lean_stringToMessageData(v___x_1367_);
return v___x_1368_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Aesop_Goal_traceMetadata_spec__1___redArg(size_t v_sz_1369_, size_t v_i_1370_, lean_object* v_bs_1371_){
_start:
{
uint8_t v___x_1372_; 
v___x_1372_ = lean_usize_dec_lt(v_i_1370_, v_sz_1369_);
if (v___x_1372_ == 0)
{
return v_bs_1371_;
}
else
{
lean_object* v_v_1373_; lean_object* v___x_1374_; lean_object* v_bs_x27_1375_; lean_object* v___x_1376_; lean_object* v___x_1377_; lean_object* v___x_1378_; lean_object* v___x_1379_; lean_object* v___x_1380_; lean_object* v___x_1381_; lean_object* v___x_1382_; lean_object* v___x_1383_; size_t v___x_1384_; size_t v___x_1385_; lean_object* v___x_1386_; 
v_v_1373_ = lean_array_uget(v_bs_1371_, v_i_1370_);
v___x_1374_ = lean_unsigned_to_nat(0u);
v_bs_x27_1375_ = lean_array_uset(v_bs_1371_, v_i_1370_, v___x_1374_);
v___x_1376_ = lean_usize_to_nat(v_i_1370_);
v___x_1377_ = l_Nat_reprFast(v___x_1376_);
v___x_1378_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1378_, 0, v___x_1377_);
v___x_1379_ = l_Lean_MessageData_ofFormat(v___x_1378_);
v___x_1380_ = lean_obj_once(&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Aesop_Goal_traceMetadata_spec__1___redArg___closed__1, &lp_aesop___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Aesop_Goal_traceMetadata_spec__1___redArg___closed__1_once, _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Aesop_Goal_traceMetadata_spec__1___redArg___closed__1);
v___x_1381_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1381_, 0, v___x_1379_);
lean_ctor_set(v___x_1381_, 1, v___x_1380_);
v___x_1382_ = l_Lean_MessageData_ofExpr(v_v_1373_);
v___x_1383_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1383_, 0, v___x_1381_);
lean_ctor_set(v___x_1383_, 1, v___x_1382_);
v___x_1384_ = ((size_t)1ULL);
v___x_1385_ = lean_usize_add(v_i_1370_, v___x_1384_);
v___x_1386_ = lean_array_uset(v_bs_x27_1375_, v_i_1370_, v___x_1383_);
v_i_1370_ = v___x_1385_;
v_bs_1371_ = v___x_1386_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Aesop_Goal_traceMetadata_spec__1___redArg___boxed(lean_object* v_sz_1388_, lean_object* v_i_1389_, lean_object* v_bs_1390_){
_start:
{
size_t v_sz_boxed_1391_; size_t v_i_boxed_1392_; lean_object* v_res_1393_; 
v_sz_boxed_1391_ = lean_unbox_usize(v_sz_1388_);
lean_dec(v_sz_1388_);
v_i_boxed_1392_ = lean_unbox_usize(v_i_1389_);
lean_dec(v_i_1389_);
v_res_1393_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Aesop_Goal_traceMetadata_spec__1___redArg(v_sz_boxed_1391_, v_i_boxed_1392_, v_bs_1390_);
return v_res_1393_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Aesop_Goal_traceMetadata_spec__3___redArg(size_t v_sz_1394_, size_t v_i_1395_, lean_object* v_bs_1396_){
_start:
{
uint8_t v___x_1397_; 
v___x_1397_ = lean_usize_dec_lt(v_i_1395_, v_sz_1394_);
if (v___x_1397_ == 0)
{
return v_bs_1396_;
}
else
{
lean_object* v_v_1398_; lean_object* v___x_1399_; lean_object* v_bs_x27_1400_; lean_object* v___x_1401_; lean_object* v___x_1402_; lean_object* v___x_1403_; lean_object* v___x_1404_; lean_object* v___x_1405_; lean_object* v___x_1406_; lean_object* v___x_1407_; lean_object* v___x_1408_; size_t v___x_1409_; size_t v___x_1410_; lean_object* v___x_1411_; 
v_v_1398_ = lean_array_uget(v_bs_1396_, v_i_1395_);
v___x_1399_ = lean_unsigned_to_nat(0u);
v_bs_x27_1400_ = lean_array_uset(v_bs_1396_, v_i_1395_, v___x_1399_);
v___x_1401_ = lean_usize_to_nat(v_i_1395_);
v___x_1402_ = l_Nat_reprFast(v___x_1401_);
v___x_1403_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1403_, 0, v___x_1402_);
v___x_1404_ = l_Lean_MessageData_ofFormat(v___x_1403_);
v___x_1405_ = lean_obj_once(&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Aesop_Goal_traceMetadata_spec__1___redArg___closed__1, &lp_aesop___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Aesop_Goal_traceMetadata_spec__1___redArg___closed__1_once, _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Aesop_Goal_traceMetadata_spec__1___redArg___closed__1);
v___x_1406_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1406_, 0, v___x_1404_);
lean_ctor_set(v___x_1406_, 1, v___x_1405_);
v___x_1407_ = l_Lean_MessageData_ofLevel(v_v_1398_);
v___x_1408_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1408_, 0, v___x_1406_);
lean_ctor_set(v___x_1408_, 1, v___x_1407_);
v___x_1409_ = ((size_t)1ULL);
v___x_1410_ = lean_usize_add(v_i_1395_, v___x_1409_);
v___x_1411_ = lean_array_uset(v_bs_x27_1400_, v_i_1395_, v___x_1408_);
v_i_1395_ = v___x_1410_;
v_bs_1396_ = v___x_1411_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Aesop_Goal_traceMetadata_spec__3___redArg___boxed(lean_object* v_sz_1413_, lean_object* v_i_1414_, lean_object* v_bs_1415_){
_start:
{
size_t v_sz_boxed_1416_; size_t v_i_boxed_1417_; lean_object* v_res_1418_; 
v_sz_boxed_1416_ = lean_unbox_usize(v_sz_1413_);
lean_dec(v_sz_1413_);
v_i_boxed_1417_ = lean_unbox_usize(v_i_1414_);
lean_dec(v_i_1414_);
v_res_1418_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Aesop_Goal_traceMetadata_spec__3___redArg(v_sz_boxed_1416_, v_i_boxed_1417_, v_bs_1415_);
return v_res_1418_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Aesop_Goal_traceMetadata_spec__0_spec__0(lean_object* v_as_1419_, size_t v_i_1420_, size_t v_stop_1421_, lean_object* v_b_1422_){
_start:
{
lean_object* v___y_1424_; uint8_t v___x_1428_; 
v___x_1428_ = lean_usize_dec_eq(v_i_1420_, v_stop_1421_);
if (v___x_1428_ == 0)
{
lean_object* v___x_1429_; 
v___x_1429_ = lean_array_uget_borrowed(v_as_1419_, v_i_1420_);
if (lean_obj_tag(v___x_1429_) == 0)
{
v___y_1424_ = v_b_1422_;
goto v___jp_1423_;
}
else
{
lean_object* v_val_1430_; lean_object* v___x_1431_; 
v_val_1430_ = lean_ctor_get(v___x_1429_, 0);
lean_inc(v_val_1430_);
v___x_1431_ = lean_array_push(v_b_1422_, v_val_1430_);
v___y_1424_ = v___x_1431_;
goto v___jp_1423_;
}
}
else
{
return v_b_1422_;
}
v___jp_1423_:
{
size_t v___x_1425_; size_t v___x_1426_; 
v___x_1425_ = ((size_t)1ULL);
v___x_1426_ = lean_usize_add(v_i_1420_, v___x_1425_);
v_i_1420_ = v___x_1426_;
v_b_1422_ = v___y_1424_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Aesop_Goal_traceMetadata_spec__0_spec__0___boxed(lean_object* v_as_1432_, lean_object* v_i_1433_, lean_object* v_stop_1434_, lean_object* v_b_1435_){
_start:
{
size_t v_i_boxed_1436_; size_t v_stop_boxed_1437_; lean_object* v_res_1438_; 
v_i_boxed_1436_ = lean_unbox_usize(v_i_1433_);
lean_dec(v_i_1433_);
v_stop_boxed_1437_ = lean_unbox_usize(v_stop_1434_);
lean_dec(v_stop_1434_);
v_res_1438_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Aesop_Goal_traceMetadata_spec__0_spec__0(v_as_1432_, v_i_boxed_1436_, v_stop_boxed_1437_, v_b_1435_);
lean_dec_ref(v_as_1432_);
return v_res_1438_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_filterMapM___at___00Aesop_Goal_traceMetadata_spec__0(lean_object* v_as_1441_, lean_object* v_start_1442_, lean_object* v_stop_1443_){
_start:
{
lean_object* v___x_1444_; uint8_t v___x_1445_; 
v___x_1444_ = ((lean_object*)(lp_aesop_Array_filterMapM___at___00Aesop_Goal_traceMetadata_spec__0___closed__0));
v___x_1445_ = lean_nat_dec_lt(v_start_1442_, v_stop_1443_);
if (v___x_1445_ == 0)
{
return v___x_1444_;
}
else
{
lean_object* v___x_1446_; uint8_t v___x_1447_; 
v___x_1446_ = lean_array_get_size(v_as_1441_);
v___x_1447_ = lean_nat_dec_le(v_stop_1443_, v___x_1446_);
if (v___x_1447_ == 0)
{
uint8_t v___x_1448_; 
v___x_1448_ = lean_nat_dec_lt(v_start_1442_, v___x_1446_);
if (v___x_1448_ == 0)
{
return v___x_1444_;
}
else
{
size_t v___x_1449_; size_t v___x_1450_; lean_object* v___x_1451_; 
v___x_1449_ = lean_usize_of_nat(v_start_1442_);
v___x_1450_ = lean_usize_of_nat(v___x_1446_);
v___x_1451_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Aesop_Goal_traceMetadata_spec__0_spec__0(v_as_1441_, v___x_1449_, v___x_1450_, v___x_1444_);
return v___x_1451_;
}
}
else
{
size_t v___x_1452_; size_t v___x_1453_; lean_object* v___x_1454_; 
v___x_1452_ = lean_usize_of_nat(v_start_1442_);
v___x_1453_ = lean_usize_of_nat(v_stop_1443_);
v___x_1454_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Aesop_Goal_traceMetadata_spec__0_spec__0(v_as_1441_, v___x_1452_, v___x_1453_, v___x_1444_);
return v___x_1454_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_filterMapM___at___00Aesop_Goal_traceMetadata_spec__0___boxed(lean_object* v_as_1455_, lean_object* v_start_1456_, lean_object* v_stop_1457_){
_start:
{
lean_object* v_res_1458_; 
v_res_1458_ = lp_aesop_Array_filterMapM___at___00Aesop_Goal_traceMetadata_spec__0(v_as_1455_, v_start_1456_, v_stop_1457_);
lean_dec(v_stop_1457_);
lean_dec(v_start_1456_);
lean_dec_ref(v_as_1455_);
return v_res_1458_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Aesop_Goal_traceMetadata_spec__2_spec__3(lean_object* v_as_1459_, size_t v_i_1460_, size_t v_stop_1461_, lean_object* v_b_1462_){
_start:
{
lean_object* v___y_1464_; uint8_t v___x_1468_; 
v___x_1468_ = lean_usize_dec_eq(v_i_1460_, v_stop_1461_);
if (v___x_1468_ == 0)
{
lean_object* v___x_1469_; 
v___x_1469_ = lean_array_uget_borrowed(v_as_1459_, v_i_1460_);
if (lean_obj_tag(v___x_1469_) == 0)
{
v___y_1464_ = v_b_1462_;
goto v___jp_1463_;
}
else
{
lean_object* v_val_1470_; lean_object* v___x_1471_; 
v_val_1470_ = lean_ctor_get(v___x_1469_, 0);
lean_inc(v_val_1470_);
v___x_1471_ = lean_array_push(v_b_1462_, v_val_1470_);
v___y_1464_ = v___x_1471_;
goto v___jp_1463_;
}
}
else
{
return v_b_1462_;
}
v___jp_1463_:
{
size_t v___x_1465_; size_t v___x_1466_; 
v___x_1465_ = ((size_t)1ULL);
v___x_1466_ = lean_usize_add(v_i_1460_, v___x_1465_);
v_i_1460_ = v___x_1466_;
v_b_1462_ = v___y_1464_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Aesop_Goal_traceMetadata_spec__2_spec__3___boxed(lean_object* v_as_1472_, lean_object* v_i_1473_, lean_object* v_stop_1474_, lean_object* v_b_1475_){
_start:
{
size_t v_i_boxed_1476_; size_t v_stop_boxed_1477_; lean_object* v_res_1478_; 
v_i_boxed_1476_ = lean_unbox_usize(v_i_1473_);
lean_dec(v_i_1473_);
v_stop_boxed_1477_ = lean_unbox_usize(v_stop_1474_);
lean_dec(v_stop_1474_);
v_res_1478_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Aesop_Goal_traceMetadata_spec__2_spec__3(v_as_1472_, v_i_boxed_1476_, v_stop_boxed_1477_, v_b_1475_);
lean_dec_ref(v_as_1472_);
return v_res_1478_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_filterMapM___at___00Aesop_Goal_traceMetadata_spec__2(lean_object* v_as_1481_, lean_object* v_start_1482_, lean_object* v_stop_1483_){
_start:
{
lean_object* v___x_1484_; uint8_t v___x_1485_; 
v___x_1484_ = ((lean_object*)(lp_aesop_Array_filterMapM___at___00Aesop_Goal_traceMetadata_spec__2___closed__0));
v___x_1485_ = lean_nat_dec_lt(v_start_1482_, v_stop_1483_);
if (v___x_1485_ == 0)
{
return v___x_1484_;
}
else
{
lean_object* v___x_1486_; uint8_t v___x_1487_; 
v___x_1486_ = lean_array_get_size(v_as_1481_);
v___x_1487_ = lean_nat_dec_le(v_stop_1483_, v___x_1486_);
if (v___x_1487_ == 0)
{
uint8_t v___x_1488_; 
v___x_1488_ = lean_nat_dec_lt(v_start_1482_, v___x_1486_);
if (v___x_1488_ == 0)
{
return v___x_1484_;
}
else
{
size_t v___x_1489_; size_t v___x_1490_; lean_object* v___x_1491_; 
v___x_1489_ = lean_usize_of_nat(v_start_1482_);
v___x_1490_ = lean_usize_of_nat(v___x_1486_);
v___x_1491_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Aesop_Goal_traceMetadata_spec__2_spec__3(v_as_1481_, v___x_1489_, v___x_1490_, v___x_1484_);
return v___x_1491_;
}
}
else
{
size_t v___x_1492_; size_t v___x_1493_; lean_object* v___x_1494_; 
v___x_1492_ = lean_usize_of_nat(v_start_1482_);
v___x_1493_ = lean_usize_of_nat(v_stop_1483_);
v___x_1494_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Aesop_Goal_traceMetadata_spec__2_spec__3(v_as_1481_, v___x_1492_, v___x_1493_, v___x_1484_);
return v___x_1494_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_filterMapM___at___00Aesop_Goal_traceMetadata_spec__2___boxed(lean_object* v_as_1495_, lean_object* v_start_1496_, lean_object* v_stop_1497_){
_start:
{
lean_object* v_res_1498_; 
v_res_1498_ = lp_aesop_Array_filterMapM___at___00Aesop_Goal_traceMetadata_spec__2(v_as_1495_, v_start_1496_, v_stop_1497_);
lean_dec(v_stop_1497_);
lean_dec(v_start_1496_);
lean_dec_ref(v_as_1495_);
return v_res_1498_;
}
}
static lean_object* _init_lp_aesop_List_mapTR_loop___at___00Aesop_Goal_traceMetadata_spec__4___closed__3(void){
_start:
{
lean_object* v___x_1503_; lean_object* v___x_1504_; 
v___x_1503_ = ((lean_object*)(lp_aesop_List_mapTR_loop___at___00Aesop_Goal_traceMetadata_spec__4___closed__2));
v___x_1504_ = l_Lean_MessageData_ofFormat(v___x_1503_);
return v___x_1504_;
}
}
static lean_object* _init_lp_aesop_List_mapTR_loop___at___00Aesop_Goal_traceMetadata_spec__4___closed__6(void){
_start:
{
lean_object* v___x_1508_; lean_object* v___x_1509_; 
v___x_1508_ = ((lean_object*)(lp_aesop_List_mapTR_loop___at___00Aesop_Goal_traceMetadata_spec__4___closed__5));
v___x_1509_ = l_Lean_MessageData_ofFormat(v___x_1508_);
return v___x_1509_;
}
}
LEAN_EXPORT lean_object* lp_aesop_List_mapTR_loop___at___00Aesop_Goal_traceMetadata_spec__4(lean_object* v_a_1511_, lean_object* v_a_1512_){
_start:
{
if (lean_obj_tag(v_a_1511_) == 0)
{
lean_object* v___x_1513_; 
v___x_1513_ = l_List_reverse___redArg(v_a_1512_);
return v___x_1513_;
}
else
{
lean_object* v_head_1514_; lean_object* v_tail_1515_; lean_object* v___x_1517_; uint8_t v_isShared_1518_; uint8_t v_isSharedCheck_1558_; 
v_head_1514_ = lean_ctor_get(v_a_1511_, 0);
v_tail_1515_ = lean_ctor_get(v_a_1511_, 1);
v_isSharedCheck_1558_ = !lean_is_exclusive(v_a_1511_);
if (v_isSharedCheck_1558_ == 0)
{
v___x_1517_ = v_a_1511_;
v_isShared_1518_ = v_isSharedCheck_1558_;
goto v_resetjp_1516_;
}
else
{
lean_inc(v_tail_1515_);
lean_inc(v_head_1514_);
lean_dec(v_a_1511_);
v___x_1517_ = lean_box(0);
v_isShared_1518_ = v_isSharedCheck_1558_;
goto v_resetjp_1516_;
}
v_resetjp_1516_:
{
lean_object* v___y_1520_; 
if (lean_obj_tag(v_head_1514_) == 0)
{
lean_object* v_fvarId_1525_; lean_object* v___x_1526_; lean_object* v___x_1527_; 
v_fvarId_1525_ = lean_ctor_get(v_head_1514_, 0);
lean_inc(v_fvarId_1525_);
lean_dec_ref_known(v_head_1514_, 1);
v___x_1526_ = l_Lean_Expr_fvar___override(v_fvarId_1525_);
v___x_1527_ = l_Lean_MessageData_ofExpr(v___x_1526_);
v___y_1520_ = v___x_1527_;
goto v___jp_1519_;
}
else
{
lean_object* v_subst_1528_; lean_object* v_premises_1529_; lean_object* v_levels_1530_; lean_object* v___x_1532_; uint8_t v_isShared_1533_; uint8_t v_isSharedCheck_1557_; 
v_subst_1528_ = lean_ctor_get(v_head_1514_, 0);
lean_inc_ref(v_subst_1528_);
lean_dec_ref_known(v_head_1514_, 1);
v_premises_1529_ = lean_ctor_get(v_subst_1528_, 0);
v_levels_1530_ = lean_ctor_get(v_subst_1528_, 1);
v_isSharedCheck_1557_ = !lean_is_exclusive(v_subst_1528_);
if (v_isSharedCheck_1557_ == 0)
{
v___x_1532_ = v_subst_1528_;
v_isShared_1533_ = v_isSharedCheck_1557_;
goto v_resetjp_1531_;
}
else
{
lean_inc(v_levels_1530_);
lean_inc(v_premises_1529_);
lean_dec(v_subst_1528_);
v___x_1532_ = lean_box(0);
v_isShared_1533_ = v_isSharedCheck_1557_;
goto v_resetjp_1531_;
}
v_resetjp_1531_:
{
lean_object* v___x_1534_; lean_object* v___x_1535_; lean_object* v___x_1536_; size_t v_sz_1537_; size_t v___x_1538_; lean_object* v___x_1539_; lean_object* v_ps_1540_; lean_object* v___x_1541_; lean_object* v___x_1542_; size_t v_sz_1543_; lean_object* v___x_1544_; lean_object* v_ls_1545_; lean_object* v___x_1546_; lean_object* v___x_1547_; lean_object* v___x_1548_; lean_object* v___x_1549_; lean_object* v___x_1551_; 
v___x_1534_ = lean_unsigned_to_nat(0u);
v___x_1535_ = lean_array_get_size(v_premises_1529_);
v___x_1536_ = lp_aesop_Array_filterMapM___at___00Aesop_Goal_traceMetadata_spec__0(v_premises_1529_, v___x_1534_, v___x_1535_);
lean_dec_ref(v_premises_1529_);
v_sz_1537_ = lean_array_size(v___x_1536_);
v___x_1538_ = ((size_t)0ULL);
v___x_1539_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Aesop_Goal_traceMetadata_spec__1___redArg(v_sz_1537_, v___x_1538_, v___x_1536_);
v_ps_1540_ = lean_array_to_list(v___x_1539_);
v___x_1541_ = lean_array_get_size(v_levels_1530_);
v___x_1542_ = lp_aesop_Array_filterMapM___at___00Aesop_Goal_traceMetadata_spec__2(v_levels_1530_, v___x_1534_, v___x_1541_);
lean_dec_ref(v_levels_1530_);
v_sz_1543_ = lean_array_size(v___x_1542_);
v___x_1544_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Aesop_Goal_traceMetadata_spec__3___redArg(v_sz_1543_, v___x_1538_, v___x_1542_);
v_ls_1545_ = lean_array_to_list(v___x_1544_);
v___x_1546_ = ((lean_object*)(lp_aesop_List_mapTR_loop___at___00Aesop_Goal_traceMetadata_spec__4___closed__0));
v___x_1547_ = lean_obj_once(&lp_aesop_List_mapTR_loop___at___00Aesop_Goal_traceMetadata_spec__4___closed__3, &lp_aesop_List_mapTR_loop___at___00Aesop_Goal_traceMetadata_spec__4___closed__3_once, _init_lp_aesop_List_mapTR_loop___at___00Aesop_Goal_traceMetadata_spec__4___closed__3);
v___x_1548_ = l_Lean_MessageData_joinSep(v_ps_1540_, v___x_1547_);
v___x_1549_ = lean_obj_once(&lp_aesop_List_mapTR_loop___at___00Aesop_Goal_traceMetadata_spec__4___closed__6, &lp_aesop_List_mapTR_loop___at___00Aesop_Goal_traceMetadata_spec__4___closed__6_once, _init_lp_aesop_List_mapTR_loop___at___00Aesop_Goal_traceMetadata_spec__4___closed__6);
if (v_isShared_1533_ == 0)
{
lean_ctor_set_tag(v___x_1532_, 7);
lean_ctor_set(v___x_1532_, 1, v___x_1549_);
lean_ctor_set(v___x_1532_, 0, v___x_1548_);
v___x_1551_ = v___x_1532_;
goto v_reusejp_1550_;
}
else
{
lean_object* v_reuseFailAlloc_1556_; 
v_reuseFailAlloc_1556_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1556_, 0, v___x_1548_);
lean_ctor_set(v_reuseFailAlloc_1556_, 1, v___x_1549_);
v___x_1551_ = v_reuseFailAlloc_1556_;
goto v_reusejp_1550_;
}
v_reusejp_1550_:
{
lean_object* v___x_1552_; lean_object* v___x_1553_; lean_object* v___x_1554_; lean_object* v___x_1555_; 
v___x_1552_ = l_Lean_MessageData_joinSep(v_ls_1545_, v___x_1547_);
v___x_1553_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1553_, 0, v___x_1551_);
lean_ctor_set(v___x_1553_, 1, v___x_1552_);
v___x_1554_ = ((lean_object*)(lp_aesop_List_mapTR_loop___at___00Aesop_Goal_traceMetadata_spec__4___closed__7));
v___x_1555_ = l_Lean_MessageData_bracket(v___x_1546_, v___x_1553_, v___x_1554_);
v___y_1520_ = v___x_1555_;
goto v___jp_1519_;
}
}
}
v___jp_1519_:
{
lean_object* v___x_1522_; 
if (v_isShared_1518_ == 0)
{
lean_ctor_set(v___x_1517_, 1, v_a_1512_);
lean_ctor_set(v___x_1517_, 0, v___y_1520_);
v___x_1522_ = v___x_1517_;
goto v_reusejp_1521_;
}
else
{
lean_object* v_reuseFailAlloc_1524_; 
v_reuseFailAlloc_1524_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1524_, 0, v___y_1520_);
lean_ctor_set(v_reuseFailAlloc_1524_, 1, v_a_1512_);
v___x_1522_ = v_reuseFailAlloc_1524_;
goto v_reusejp_1521_;
}
v_reusejp_1521_:
{
v_a_1511_ = v_tail_1515_;
v_a_1512_ = v___x_1522_;
goto _start;
}
}
}
}
}
}
static lean_object* _init_lp_aesop_List_mapTR_loop___at___00Aesop_Goal_traceMetadata_spec__8___closed__1(void){
_start:
{
lean_object* v___x_1560_; lean_object* v___x_1561_; 
v___x_1560_ = ((lean_object*)(lp_aesop_List_mapTR_loop___at___00Aesop_Goal_traceMetadata_spec__8___closed__0));
v___x_1561_ = l_Lean_stringToMessageData(v___x_1560_);
return v___x_1561_;
}
}
static lean_object* _init_lp_aesop_List_mapTR_loop___at___00Aesop_Goal_traceMetadata_spec__8___closed__3(void){
_start:
{
lean_object* v___x_1563_; lean_object* v___x_1564_; 
v___x_1563_ = ((lean_object*)(lp_aesop_List_mapTR_loop___at___00Aesop_Goal_traceMetadata_spec__8___closed__2));
v___x_1564_ = l_Lean_stringToMessageData(v___x_1563_);
return v___x_1564_;
}
}
LEAN_EXPORT lean_object* lp_aesop_List_mapTR_loop___at___00Aesop_Goal_traceMetadata_spec__8(lean_object* v_a_1565_, lean_object* v_a_1566_){
_start:
{
if (lean_obj_tag(v_a_1565_) == 0)
{
lean_object* v___x_1567_; 
v___x_1567_ = l_List_reverse___redArg(v_a_1566_);
return v___x_1567_;
}
else
{
lean_object* v_head_1568_; lean_object* v_tail_1569_; lean_object* v___x_1571_; uint8_t v_isShared_1572_; uint8_t v_isSharedCheck_1597_; 
v_head_1568_ = lean_ctor_get(v_a_1565_, 0);
v_tail_1569_ = lean_ctor_get(v_a_1565_, 1);
v_isSharedCheck_1597_ = !lean_is_exclusive(v_a_1565_);
if (v_isSharedCheck_1597_ == 0)
{
v___x_1571_ = v_a_1565_;
v_isShared_1572_ = v_isSharedCheck_1597_;
goto v_resetjp_1570_;
}
else
{
lean_inc(v_tail_1569_);
lean_inc(v_head_1568_);
lean_dec(v_a_1565_);
v___x_1571_ = lean_box(0);
v_isShared_1572_ = v_isSharedCheck_1597_;
goto v_resetjp_1570_;
}
v_resetjp_1570_:
{
lean_object* v_fst_1573_; lean_object* v_snd_1574_; lean_object* v___x_1576_; uint8_t v_isShared_1577_; uint8_t v_isSharedCheck_1596_; 
v_fst_1573_ = lean_ctor_get(v_head_1568_, 0);
v_snd_1574_ = lean_ctor_get(v_head_1568_, 1);
v_isSharedCheck_1596_ = !lean_is_exclusive(v_head_1568_);
if (v_isSharedCheck_1596_ == 0)
{
v___x_1576_ = v_head_1568_;
v_isShared_1577_ = v_isSharedCheck_1596_;
goto v_resetjp_1575_;
}
else
{
lean_inc(v_snd_1574_);
lean_inc(v_fst_1573_);
lean_dec(v_head_1568_);
v___x_1576_ = lean_box(0);
v_isShared_1577_ = v_isSharedCheck_1596_;
goto v_resetjp_1575_;
}
v_resetjp_1575_:
{
lean_object* v___x_1578_; lean_object* v___x_1579_; lean_object* v___x_1580_; lean_object* v___x_1581_; lean_object* v___x_1583_; 
v___x_1578_ = lean_obj_once(&lp_aesop_List_mapTR_loop___at___00Aesop_Goal_traceMetadata_spec__8___closed__1, &lp_aesop_List_mapTR_loop___at___00Aesop_Goal_traceMetadata_spec__8___closed__1_once, _init_lp_aesop_List_mapTR_loop___at___00Aesop_Goal_traceMetadata_spec__8___closed__1);
v___x_1579_ = l_Nat_reprFast(v_snd_1574_);
v___x_1580_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1580_, 0, v___x_1579_);
v___x_1581_ = l_Lean_MessageData_ofFormat(v___x_1580_);
if (v_isShared_1577_ == 0)
{
lean_ctor_set_tag(v___x_1576_, 7);
lean_ctor_set(v___x_1576_, 1, v___x_1581_);
lean_ctor_set(v___x_1576_, 0, v___x_1578_);
v___x_1583_ = v___x_1576_;
goto v_reusejp_1582_;
}
else
{
lean_object* v_reuseFailAlloc_1595_; 
v_reuseFailAlloc_1595_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1595_, 0, v___x_1578_);
lean_ctor_set(v_reuseFailAlloc_1595_, 1, v___x_1581_);
v___x_1583_ = v_reuseFailAlloc_1595_;
goto v_reusejp_1582_;
}
v_reusejp_1582_:
{
lean_object* v___x_1584_; lean_object* v___x_1585_; lean_object* v___x_1586_; lean_object* v___x_1587_; lean_object* v___x_1588_; lean_object* v___x_1589_; lean_object* v___x_1590_; lean_object* v___x_1592_; 
v___x_1584_ = lean_obj_once(&lp_aesop_List_mapTR_loop___at___00Aesop_Goal_traceMetadata_spec__8___closed__3, &lp_aesop_List_mapTR_loop___at___00Aesop_Goal_traceMetadata_spec__8___closed__3_once, _init_lp_aesop_List_mapTR_loop___at___00Aesop_Goal_traceMetadata_spec__8___closed__3);
v___x_1585_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1585_, 0, v___x_1583_);
lean_ctor_set(v___x_1585_, 1, v___x_1584_);
v___x_1586_ = lean_array_to_list(v_fst_1573_);
v___x_1587_ = lean_box(0);
v___x_1588_ = lp_aesop_List_mapTR_loop___at___00Aesop_Goal_traceMetadata_spec__4(v___x_1586_, v___x_1587_);
v___x_1589_ = l_Lean_MessageData_ofList(v___x_1588_);
v___x_1590_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1590_, 0, v___x_1585_);
lean_ctor_set(v___x_1590_, 1, v___x_1589_);
if (v_isShared_1572_ == 0)
{
lean_ctor_set(v___x_1571_, 1, v_a_1566_);
lean_ctor_set(v___x_1571_, 0, v___x_1590_);
v___x_1592_ = v___x_1571_;
goto v_reusejp_1591_;
}
else
{
lean_object* v_reuseFailAlloc_1594_; 
v_reuseFailAlloc_1594_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1594_, 0, v___x_1590_);
lean_ctor_set(v_reuseFailAlloc_1594_, 1, v_a_1566_);
v___x_1592_ = v_reuseFailAlloc_1594_;
goto v_reusejp_1591_;
}
v_reusejp_1591_:
{
v_a_1565_ = v_tail_1569_;
v_a_1566_ = v___x_1592_;
goto _start;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__15_spec__20_spec__38_spec__48___redArg(lean_object* v_f_1598_, lean_object* v_keys_1599_, lean_object* v_vals_1600_, lean_object* v_i_1601_, lean_object* v_acc_1602_){
_start:
{
lean_object* v___x_1603_; uint8_t v___x_1604_; 
v___x_1603_ = lean_array_get_size(v_keys_1599_);
v___x_1604_ = lean_nat_dec_lt(v_i_1601_, v___x_1603_);
if (v___x_1604_ == 0)
{
lean_object* v___x_1605_; 
lean_dec(v_i_1601_);
lean_dec_ref(v_f_1598_);
v___x_1605_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1605_, 0, v_acc_1602_);
return v___x_1605_;
}
else
{
lean_object* v_k_1606_; lean_object* v_v_1607_; lean_object* v___x_1608_; 
v_k_1606_ = lean_array_fget_borrowed(v_keys_1599_, v_i_1601_);
v_v_1607_ = lean_array_fget_borrowed(v_vals_1600_, v_i_1601_);
lean_inc_ref(v_f_1598_);
lean_inc(v_v_1607_);
lean_inc(v_k_1606_);
v___x_1608_ = lean_apply_3(v_f_1598_, v_acc_1602_, v_k_1606_, v_v_1607_);
if (lean_obj_tag(v___x_1608_) == 0)
{
lean_dec(v_i_1601_);
lean_dec_ref(v_f_1598_);
return v___x_1608_;
}
else
{
lean_object* v_a_1609_; lean_object* v___x_1610_; lean_object* v___x_1611_; 
v_a_1609_ = lean_ctor_get(v___x_1608_, 0);
lean_inc(v_a_1609_);
lean_dec_ref_known(v___x_1608_, 1);
v___x_1610_ = lean_unsigned_to_nat(1u);
v___x_1611_ = lean_nat_add(v_i_1601_, v___x_1610_);
lean_dec(v_i_1601_);
v_i_1601_ = v___x_1611_;
v_acc_1602_ = v_a_1609_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__15_spec__20_spec__38_spec__48___redArg___boxed(lean_object* v_f_1613_, lean_object* v_keys_1614_, lean_object* v_vals_1615_, lean_object* v_i_1616_, lean_object* v_acc_1617_){
_start:
{
lean_object* v_res_1618_; 
v_res_1618_ = lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__15_spec__20_spec__38_spec__48___redArg(v_f_1613_, v_keys_1614_, v_vals_1615_, v_i_1616_, v_acc_1617_);
lean_dec_ref(v_vals_1615_);
lean_dec_ref(v_keys_1614_);
return v_res_1618_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__15_spec__20_spec__38___redArg(lean_object* v_f_1619_, lean_object* v_x_1620_, lean_object* v_x_1621_){
_start:
{
if (lean_obj_tag(v_x_1620_) == 0)
{
lean_object* v_es_1622_; lean_object* v___x_1624_; uint8_t v_isShared_1625_; uint8_t v_isSharedCheck_1642_; 
v_es_1622_ = lean_ctor_get(v_x_1620_, 0);
v_isSharedCheck_1642_ = !lean_is_exclusive(v_x_1620_);
if (v_isSharedCheck_1642_ == 0)
{
v___x_1624_ = v_x_1620_;
v_isShared_1625_ = v_isSharedCheck_1642_;
goto v_resetjp_1623_;
}
else
{
lean_inc(v_es_1622_);
lean_dec(v_x_1620_);
v___x_1624_ = lean_box(0);
v_isShared_1625_ = v_isSharedCheck_1642_;
goto v_resetjp_1623_;
}
v_resetjp_1623_:
{
lean_object* v___x_1626_; lean_object* v___x_1627_; uint8_t v___x_1628_; 
v___x_1626_ = lean_unsigned_to_nat(0u);
v___x_1627_ = lean_array_get_size(v_es_1622_);
v___x_1628_ = lean_nat_dec_lt(v___x_1626_, v___x_1627_);
if (v___x_1628_ == 0)
{
lean_object* v___x_1630_; 
lean_dec_ref(v_es_1622_);
lean_dec_ref(v_f_1619_);
if (v_isShared_1625_ == 0)
{
lean_ctor_set_tag(v___x_1624_, 1);
lean_ctor_set(v___x_1624_, 0, v_x_1621_);
v___x_1630_ = v___x_1624_;
goto v_reusejp_1629_;
}
else
{
lean_object* v_reuseFailAlloc_1631_; 
v_reuseFailAlloc_1631_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1631_, 0, v_x_1621_);
v___x_1630_ = v_reuseFailAlloc_1631_;
goto v_reusejp_1629_;
}
v_reusejp_1629_:
{
return v___x_1630_;
}
}
else
{
uint8_t v___x_1632_; 
v___x_1632_ = lean_nat_dec_le(v___x_1627_, v___x_1627_);
if (v___x_1632_ == 0)
{
if (v___x_1628_ == 0)
{
lean_object* v___x_1634_; 
lean_dec_ref(v_es_1622_);
lean_dec_ref(v_f_1619_);
if (v_isShared_1625_ == 0)
{
lean_ctor_set_tag(v___x_1624_, 1);
lean_ctor_set(v___x_1624_, 0, v_x_1621_);
v___x_1634_ = v___x_1624_;
goto v_reusejp_1633_;
}
else
{
lean_object* v_reuseFailAlloc_1635_; 
v_reuseFailAlloc_1635_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1635_, 0, v_x_1621_);
v___x_1634_ = v_reuseFailAlloc_1635_;
goto v_reusejp_1633_;
}
v_reusejp_1633_:
{
return v___x_1634_;
}
}
else
{
size_t v___x_1636_; size_t v___x_1637_; lean_object* v___x_1638_; 
lean_del_object(v___x_1624_);
v___x_1636_ = ((size_t)0ULL);
v___x_1637_ = lean_usize_of_nat(v___x_1627_);
v___x_1638_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__15_spec__20_spec__38_spec__47___redArg(v_f_1619_, v_es_1622_, v___x_1636_, v___x_1637_, v_x_1621_);
lean_dec_ref(v_es_1622_);
return v___x_1638_;
}
}
else
{
size_t v___x_1639_; size_t v___x_1640_; lean_object* v___x_1641_; 
lean_del_object(v___x_1624_);
v___x_1639_ = ((size_t)0ULL);
v___x_1640_ = lean_usize_of_nat(v___x_1627_);
v___x_1641_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__15_spec__20_spec__38_spec__47___redArg(v_f_1619_, v_es_1622_, v___x_1639_, v___x_1640_, v_x_1621_);
lean_dec_ref(v_es_1622_);
return v___x_1641_;
}
}
}
}
else
{
lean_object* v_ks_1643_; lean_object* v_vs_1644_; lean_object* v___x_1645_; lean_object* v___x_1646_; 
v_ks_1643_ = lean_ctor_get(v_x_1620_, 0);
lean_inc_ref(v_ks_1643_);
v_vs_1644_ = lean_ctor_get(v_x_1620_, 1);
lean_inc_ref(v_vs_1644_);
lean_dec_ref_known(v_x_1620_, 2);
v___x_1645_ = lean_unsigned_to_nat(0u);
v___x_1646_ = lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__15_spec__20_spec__38_spec__48___redArg(v_f_1619_, v_ks_1643_, v_vs_1644_, v___x_1645_, v_x_1621_);
lean_dec_ref(v_vs_1644_);
lean_dec_ref(v_ks_1643_);
return v___x_1646_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__15_spec__20_spec__38_spec__47___redArg(lean_object* v_f_1647_, lean_object* v_as_1648_, size_t v_i_1649_, size_t v_stop_1650_, lean_object* v_b_1651_){
_start:
{
lean_object* v_a_1653_; lean_object* v___y_1658_; uint8_t v___x_1660_; 
v___x_1660_ = lean_usize_dec_eq(v_i_1649_, v_stop_1650_);
if (v___x_1660_ == 0)
{
lean_object* v___x_1661_; 
v___x_1661_ = lean_array_uget_borrowed(v_as_1648_, v_i_1649_);
switch(lean_obj_tag(v___x_1661_))
{
case 0:
{
lean_object* v_key_1662_; lean_object* v_val_1663_; lean_object* v___x_1664_; 
v_key_1662_ = lean_ctor_get(v___x_1661_, 0);
v_val_1663_ = lean_ctor_get(v___x_1661_, 1);
lean_inc_ref(v_f_1647_);
lean_inc(v_val_1663_);
lean_inc(v_key_1662_);
v___x_1664_ = lean_apply_3(v_f_1647_, v_b_1651_, v_key_1662_, v_val_1663_);
v___y_1658_ = v___x_1664_;
goto v___jp_1657_;
}
case 1:
{
lean_object* v_node_1665_; lean_object* v___x_1666_; 
v_node_1665_ = lean_ctor_get(v___x_1661_, 0);
lean_inc(v_node_1665_);
lean_inc_ref(v_f_1647_);
v___x_1666_ = lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__15_spec__20_spec__38___redArg(v_f_1647_, v_node_1665_, v_b_1651_);
v___y_1658_ = v___x_1666_;
goto v___jp_1657_;
}
default: 
{
v_a_1653_ = v_b_1651_;
goto v___jp_1652_;
}
}
}
else
{
lean_object* v___x_1667_; 
lean_dec_ref(v_f_1647_);
v___x_1667_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1667_, 0, v_b_1651_);
return v___x_1667_;
}
v___jp_1652_:
{
size_t v___x_1654_; size_t v___x_1655_; 
v___x_1654_ = ((size_t)1ULL);
v___x_1655_ = lean_usize_add(v_i_1649_, v___x_1654_);
v_i_1649_ = v___x_1655_;
v_b_1651_ = v_a_1653_;
goto _start;
}
v___jp_1657_:
{
if (lean_obj_tag(v___y_1658_) == 0)
{
lean_dec_ref(v_f_1647_);
return v___y_1658_;
}
else
{
lean_object* v_a_1659_; 
v_a_1659_ = lean_ctor_get(v___y_1658_, 0);
lean_inc(v_a_1659_);
lean_dec_ref_known(v___y_1658_, 1);
v_a_1653_ = v_a_1659_;
goto v___jp_1652_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__15_spec__20_spec__38_spec__47___redArg___boxed(lean_object* v_f_1668_, lean_object* v_as_1669_, lean_object* v_i_1670_, lean_object* v_stop_1671_, lean_object* v_b_1672_){
_start:
{
size_t v_i_boxed_1673_; size_t v_stop_boxed_1674_; lean_object* v_res_1675_; 
v_i_boxed_1673_ = lean_unbox_usize(v_i_1670_);
lean_dec(v_i_1670_);
v_stop_boxed_1674_ = lean_unbox_usize(v_stop_1671_);
lean_dec(v_stop_1671_);
v_res_1675_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__15_spec__20_spec__38_spec__47___redArg(v_f_1668_, v_as_1669_, v_i_boxed_1673_, v_stop_boxed_1674_, v_b_1672_);
lean_dec_ref(v_as_1669_);
return v_res_1675_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__15___redArg___lam__0(lean_object* v_f_1676_, lean_object* v_s_1677_, lean_object* v_a_1678_, lean_object* v_b_1679_){
_start:
{
lean_object* v___x_1680_; lean_object* v___x_1681_; 
v___x_1680_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1680_, 0, v_a_1678_);
lean_ctor_set(v___x_1680_, 1, v_b_1679_);
v___x_1681_ = lean_apply_2(v_f_1676_, v___x_1680_, v_s_1677_);
if (lean_obj_tag(v___x_1681_) == 0)
{
lean_object* v_a_1682_; lean_object* v___x_1684_; uint8_t v_isShared_1685_; uint8_t v_isSharedCheck_1689_; 
v_a_1682_ = lean_ctor_get(v___x_1681_, 0);
v_isSharedCheck_1689_ = !lean_is_exclusive(v___x_1681_);
if (v_isSharedCheck_1689_ == 0)
{
v___x_1684_ = v___x_1681_;
v_isShared_1685_ = v_isSharedCheck_1689_;
goto v_resetjp_1683_;
}
else
{
lean_inc(v_a_1682_);
lean_dec(v___x_1681_);
v___x_1684_ = lean_box(0);
v_isShared_1685_ = v_isSharedCheck_1689_;
goto v_resetjp_1683_;
}
v_resetjp_1683_:
{
lean_object* v___x_1687_; 
if (v_isShared_1685_ == 0)
{
v___x_1687_ = v___x_1684_;
goto v_reusejp_1686_;
}
else
{
lean_object* v_reuseFailAlloc_1688_; 
v_reuseFailAlloc_1688_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1688_, 0, v_a_1682_);
v___x_1687_ = v_reuseFailAlloc_1688_;
goto v_reusejp_1686_;
}
v_reusejp_1686_:
{
return v___x_1687_;
}
}
}
else
{
lean_object* v_a_1690_; lean_object* v___x_1692_; uint8_t v_isShared_1693_; uint8_t v_isSharedCheck_1697_; 
v_a_1690_ = lean_ctor_get(v___x_1681_, 0);
v_isSharedCheck_1697_ = !lean_is_exclusive(v___x_1681_);
if (v_isSharedCheck_1697_ == 0)
{
v___x_1692_ = v___x_1681_;
v_isShared_1693_ = v_isSharedCheck_1697_;
goto v_resetjp_1691_;
}
else
{
lean_inc(v_a_1690_);
lean_dec(v___x_1681_);
v___x_1692_ = lean_box(0);
v_isShared_1693_ = v_isSharedCheck_1697_;
goto v_resetjp_1691_;
}
v_resetjp_1691_:
{
lean_object* v___x_1695_; 
if (v_isShared_1693_ == 0)
{
v___x_1695_ = v___x_1692_;
goto v_reusejp_1694_;
}
else
{
lean_object* v_reuseFailAlloc_1696_; 
v_reuseFailAlloc_1696_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1696_, 0, v_a_1690_);
v___x_1695_ = v_reuseFailAlloc_1696_;
goto v_reusejp_1694_;
}
v_reusejp_1694_:
{
return v___x_1695_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__15___redArg(lean_object* v_map_1698_, lean_object* v_init_1699_, lean_object* v_f_1700_){
_start:
{
lean_object* v___f_1701_; lean_object* v___x_1702_; lean_object* v_a_1703_; 
v___f_1701_ = lean_alloc_closure((void*)(lp_aesop_Lean_PersistentHashMap_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__15___redArg___lam__0), 4, 1);
lean_closure_set(v___f_1701_, 0, v_f_1700_);
lean_inc_ref(v_map_1698_);
v___x_1702_ = lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__15_spec__20_spec__38___redArg(v___f_1701_, v_map_1698_, v_init_1699_);
v_a_1703_ = lean_ctor_get(v___x_1702_, 0);
lean_inc(v_a_1703_);
lean_dec_ref(v___x_1702_);
return v_a_1703_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__15___redArg___boxed(lean_object* v_map_1704_, lean_object* v_init_1705_, lean_object* v_f_1706_){
_start:
{
lean_object* v_res_1707_; 
v_res_1707_ = lp_aesop_Lean_PersistentHashMap_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__15___redArg(v_map_1704_, v_init_1705_, v_f_1706_);
lean_dec_ref(v_map_1704_);
return v_res_1707_;
}
}
LEAN_EXPORT lean_object* lp_aesop_List_mapTR_loop___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__11(lean_object* v_a_1708_, lean_object* v_a_1709_){
_start:
{
if (lean_obj_tag(v_a_1708_) == 0)
{
lean_object* v___x_1710_; 
v___x_1710_ = l_List_reverse___redArg(v_a_1709_);
return v___x_1710_;
}
else
{
lean_object* v_head_1711_; lean_object* v_tail_1712_; lean_object* v___x_1714_; uint8_t v_isShared_1715_; uint8_t v_isSharedCheck_1720_; 
v_head_1711_ = lean_ctor_get(v_a_1708_, 0);
v_tail_1712_ = lean_ctor_get(v_a_1708_, 1);
v_isSharedCheck_1720_ = !lean_is_exclusive(v_a_1708_);
if (v_isSharedCheck_1720_ == 0)
{
v___x_1714_ = v_a_1708_;
v_isShared_1715_ = v_isSharedCheck_1720_;
goto v_resetjp_1713_;
}
else
{
lean_inc(v_tail_1712_);
lean_inc(v_head_1711_);
lean_dec(v_a_1708_);
v___x_1714_ = lean_box(0);
v_isShared_1715_ = v_isSharedCheck_1720_;
goto v_resetjp_1713_;
}
v_resetjp_1713_:
{
lean_object* v___x_1717_; 
if (v_isShared_1715_ == 0)
{
lean_ctor_set(v___x_1714_, 1, v_a_1709_);
v___x_1717_ = v___x_1714_;
goto v_reusejp_1716_;
}
else
{
lean_object* v_reuseFailAlloc_1719_; 
v_reuseFailAlloc_1719_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1719_, 0, v_head_1711_);
lean_ctor_set(v_reuseFailAlloc_1719_, 1, v_a_1709_);
v___x_1717_ = v_reuseFailAlloc_1719_;
goto v_reusejp_1716_;
}
v_reusejp_1716_:
{
v_a_1708_ = v_tail_1712_;
v_a_1709_ = v___x_1717_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_toList___at___00Lean_PersistentHashSet_toList___at___00__private_Aesop_Forward_State_0__Aesop_ppPHashSet___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__10_spec__11_spec__23___redArg___lam__0(lean_object* v_ps_1721_, lean_object* v_k_1722_, lean_object* v_v_1723_){
_start:
{
lean_object* v___x_1724_; lean_object* v___x_1725_; 
v___x_1724_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1724_, 0, v_k_1722_);
lean_ctor_set(v___x_1724_, 1, v_v_1723_);
v___x_1725_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1725_, 0, v___x_1724_);
lean_ctor_set(v___x_1725_, 1, v_ps_1721_);
return v___x_1725_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_Goal_traceMetadata_spec__9_spec__18_spec__25___redArg(lean_object* v_f_1726_, lean_object* v_keys_1727_, lean_object* v_vals_1728_, lean_object* v_i_1729_, lean_object* v_acc_1730_){
_start:
{
lean_object* v___x_1731_; uint8_t v___x_1732_; 
v___x_1731_ = lean_array_get_size(v_keys_1727_);
v___x_1732_ = lean_nat_dec_lt(v_i_1729_, v___x_1731_);
if (v___x_1732_ == 0)
{
lean_dec(v_i_1729_);
lean_dec(v_f_1726_);
return v_acc_1730_;
}
else
{
lean_object* v_k_1733_; lean_object* v_v_1734_; lean_object* v___x_1735_; lean_object* v___x_1736_; lean_object* v___x_1737_; 
v_k_1733_ = lean_array_fget_borrowed(v_keys_1727_, v_i_1729_);
v_v_1734_ = lean_array_fget_borrowed(v_vals_1728_, v_i_1729_);
lean_inc(v_f_1726_);
lean_inc(v_v_1734_);
lean_inc(v_k_1733_);
v___x_1735_ = lean_apply_3(v_f_1726_, v_acc_1730_, v_k_1733_, v_v_1734_);
v___x_1736_ = lean_unsigned_to_nat(1u);
v___x_1737_ = lean_nat_add(v_i_1729_, v___x_1736_);
lean_dec(v_i_1729_);
v_i_1729_ = v___x_1737_;
v_acc_1730_ = v___x_1735_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_Goal_traceMetadata_spec__9_spec__18_spec__25___redArg___boxed(lean_object* v_f_1739_, lean_object* v_keys_1740_, lean_object* v_vals_1741_, lean_object* v_i_1742_, lean_object* v_acc_1743_){
_start:
{
lean_object* v_res_1744_; 
v_res_1744_ = lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_Goal_traceMetadata_spec__9_spec__18_spec__25___redArg(v_f_1739_, v_keys_1740_, v_vals_1741_, v_i_1742_, v_acc_1743_);
lean_dec_ref(v_vals_1741_);
lean_dec_ref(v_keys_1740_);
return v_res_1744_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_Goal_traceMetadata_spec__9_spec__18___redArg(lean_object* v_f_1745_, lean_object* v_x_1746_, lean_object* v_x_1747_){
_start:
{
if (lean_obj_tag(v_x_1746_) == 0)
{
lean_object* v_es_1748_; lean_object* v___x_1749_; lean_object* v___x_1750_; uint8_t v___x_1751_; 
v_es_1748_ = lean_ctor_get(v_x_1746_, 0);
v___x_1749_ = lean_unsigned_to_nat(0u);
v___x_1750_ = lean_array_get_size(v_es_1748_);
v___x_1751_ = lean_nat_dec_lt(v___x_1749_, v___x_1750_);
if (v___x_1751_ == 0)
{
lean_dec(v_f_1745_);
return v_x_1747_;
}
else
{
uint8_t v___x_1752_; 
v___x_1752_ = lean_nat_dec_le(v___x_1750_, v___x_1750_);
if (v___x_1752_ == 0)
{
if (v___x_1751_ == 0)
{
lean_dec(v_f_1745_);
return v_x_1747_;
}
else
{
size_t v___x_1753_; size_t v___x_1754_; lean_object* v___x_1755_; 
v___x_1753_ = ((size_t)0ULL);
v___x_1754_ = lean_usize_of_nat(v___x_1750_);
v___x_1755_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_Goal_traceMetadata_spec__9_spec__18_spec__24___redArg(v_f_1745_, v_es_1748_, v___x_1753_, v___x_1754_, v_x_1747_);
return v___x_1755_;
}
}
else
{
size_t v___x_1756_; size_t v___x_1757_; lean_object* v___x_1758_; 
v___x_1756_ = ((size_t)0ULL);
v___x_1757_ = lean_usize_of_nat(v___x_1750_);
v___x_1758_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_Goal_traceMetadata_spec__9_spec__18_spec__24___redArg(v_f_1745_, v_es_1748_, v___x_1756_, v___x_1757_, v_x_1747_);
return v___x_1758_;
}
}
}
else
{
lean_object* v_ks_1759_; lean_object* v_vs_1760_; lean_object* v___x_1761_; lean_object* v___x_1762_; 
v_ks_1759_ = lean_ctor_get(v_x_1746_, 0);
v_vs_1760_ = lean_ctor_get(v_x_1746_, 1);
v___x_1761_ = lean_unsigned_to_nat(0u);
v___x_1762_ = lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_Goal_traceMetadata_spec__9_spec__18_spec__25___redArg(v_f_1745_, v_ks_1759_, v_vs_1760_, v___x_1761_, v_x_1747_);
return v___x_1762_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_Goal_traceMetadata_spec__9_spec__18_spec__24___redArg(lean_object* v_f_1763_, lean_object* v_as_1764_, size_t v_i_1765_, size_t v_stop_1766_, lean_object* v_b_1767_){
_start:
{
lean_object* v___y_1769_; uint8_t v___x_1773_; 
v___x_1773_ = lean_usize_dec_eq(v_i_1765_, v_stop_1766_);
if (v___x_1773_ == 0)
{
lean_object* v___x_1774_; 
v___x_1774_ = lean_array_uget_borrowed(v_as_1764_, v_i_1765_);
switch(lean_obj_tag(v___x_1774_))
{
case 0:
{
lean_object* v_key_1775_; lean_object* v_val_1776_; lean_object* v___x_1777_; 
v_key_1775_ = lean_ctor_get(v___x_1774_, 0);
v_val_1776_ = lean_ctor_get(v___x_1774_, 1);
lean_inc(v_f_1763_);
lean_inc(v_val_1776_);
lean_inc(v_key_1775_);
v___x_1777_ = lean_apply_3(v_f_1763_, v_b_1767_, v_key_1775_, v_val_1776_);
v___y_1769_ = v___x_1777_;
goto v___jp_1768_;
}
case 1:
{
lean_object* v_node_1778_; lean_object* v___x_1779_; 
v_node_1778_ = lean_ctor_get(v___x_1774_, 0);
lean_inc(v_f_1763_);
v___x_1779_ = lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_Goal_traceMetadata_spec__9_spec__18___redArg(v_f_1763_, v_node_1778_, v_b_1767_);
v___y_1769_ = v___x_1779_;
goto v___jp_1768_;
}
default: 
{
v___y_1769_ = v_b_1767_;
goto v___jp_1768_;
}
}
}
else
{
lean_dec(v_f_1763_);
return v_b_1767_;
}
v___jp_1768_:
{
size_t v___x_1770_; size_t v___x_1771_; 
v___x_1770_ = ((size_t)1ULL);
v___x_1771_ = lean_usize_add(v_i_1765_, v___x_1770_);
v_i_1765_ = v___x_1771_;
v_b_1767_ = v___y_1769_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_Goal_traceMetadata_spec__9_spec__18_spec__24___redArg___boxed(lean_object* v_f_1780_, lean_object* v_as_1781_, lean_object* v_i_1782_, lean_object* v_stop_1783_, lean_object* v_b_1784_){
_start:
{
size_t v_i_boxed_1785_; size_t v_stop_boxed_1786_; lean_object* v_res_1787_; 
v_i_boxed_1785_ = lean_unbox_usize(v_i_1782_);
lean_dec(v_i_1782_);
v_stop_boxed_1786_ = lean_unbox_usize(v_stop_1783_);
lean_dec(v_stop_1783_);
v_res_1787_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_Goal_traceMetadata_spec__9_spec__18_spec__24___redArg(v_f_1780_, v_as_1781_, v_i_boxed_1785_, v_stop_boxed_1786_, v_b_1784_);
lean_dec_ref(v_as_1781_);
return v_res_1787_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_Goal_traceMetadata_spec__9_spec__18___redArg___boxed(lean_object* v_f_1788_, lean_object* v_x_1789_, lean_object* v_x_1790_){
_start:
{
lean_object* v_res_1791_; 
v_res_1791_ = lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_Goal_traceMetadata_spec__9_spec__18___redArg(v_f_1788_, v_x_1789_, v_x_1790_);
lean_dec_ref(v_x_1789_);
return v_res_1791_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PersistentHashSet_toList___at___00__private_Aesop_Forward_State_0__Aesop_ppPHashSet___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__10_spec__11_spec__23_spec__31___redArg___lam__0(lean_object* v_f_1792_, lean_object* v_x1_1793_, lean_object* v_x2_1794_, lean_object* v_x3_1795_){
_start:
{
lean_object* v___x_1796_; 
v___x_1796_ = lean_apply_3(v_f_1792_, v_x1_1793_, v_x2_1794_, v_x3_1795_);
return v___x_1796_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PersistentHashSet_toList___at___00__private_Aesop_Forward_State_0__Aesop_ppPHashSet___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__10_spec__11_spec__23_spec__31___redArg(lean_object* v_map_1797_, lean_object* v_f_1798_, lean_object* v_init_1799_){
_start:
{
lean_object* v___f_1800_; lean_object* v___x_1801_; 
v___f_1800_ = lean_alloc_closure((void*)(lp_aesop_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PersistentHashSet_toList___at___00__private_Aesop_Forward_State_0__Aesop_ppPHashSet___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__10_spec__11_spec__23_spec__31___redArg___lam__0), 4, 1);
lean_closure_set(v___f_1800_, 0, v_f_1798_);
v___x_1801_ = lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_Goal_traceMetadata_spec__9_spec__18___redArg(v___f_1800_, v_map_1797_, v_init_1799_);
return v___x_1801_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PersistentHashSet_toList___at___00__private_Aesop_Forward_State_0__Aesop_ppPHashSet___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__10_spec__11_spec__23_spec__31___redArg___boxed(lean_object* v_map_1802_, lean_object* v_f_1803_, lean_object* v_init_1804_){
_start:
{
lean_object* v_res_1805_; 
v_res_1805_ = lp_aesop_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PersistentHashSet_toList___at___00__private_Aesop_Forward_State_0__Aesop_ppPHashSet___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__10_spec__11_spec__23_spec__31___redArg(v_map_1802_, v_f_1803_, v_init_1804_);
lean_dec_ref(v_map_1802_);
return v_res_1805_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_toList___at___00Lean_PersistentHashSet_toList___at___00__private_Aesop_Forward_State_0__Aesop_ppPHashSet___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__10_spec__11_spec__23___redArg(lean_object* v_m_1807_){
_start:
{
lean_object* v___f_1808_; lean_object* v___x_1809_; lean_object* v___x_1810_; 
v___f_1808_ = ((lean_object*)(lp_aesop_Lean_PersistentHashMap_toList___at___00Lean_PersistentHashSet_toList___at___00__private_Aesop_Forward_State_0__Aesop_ppPHashSet___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__10_spec__11_spec__23___redArg___closed__0));
v___x_1809_ = lean_box(0);
v___x_1810_ = lp_aesop_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PersistentHashSet_toList___at___00__private_Aesop_Forward_State_0__Aesop_ppPHashSet___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__10_spec__11_spec__23_spec__31___redArg(v_m_1807_, v___f_1808_, v___x_1809_);
return v___x_1810_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_toList___at___00Lean_PersistentHashSet_toList___at___00__private_Aesop_Forward_State_0__Aesop_ppPHashSet___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__10_spec__11_spec__23___redArg___boxed(lean_object* v_m_1811_){
_start:
{
lean_object* v_res_1812_; 
v_res_1812_ = lp_aesop_Lean_PersistentHashMap_toList___at___00Lean_PersistentHashSet_toList___at___00__private_Aesop_Forward_State_0__Aesop_ppPHashSet___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__10_spec__11_spec__23___redArg(v_m_1811_);
lean_dec_ref(v_m_1811_);
return v_res_1812_;
}
}
LEAN_EXPORT lean_object* lp_aesop_List_mapTR_loop___at___00Lean_PersistentHashSet_toList___at___00__private_Aesop_Forward_State_0__Aesop_ppPHashSet___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__10_spec__11_spec__24(lean_object* v_a_1813_, lean_object* v_a_1814_){
_start:
{
if (lean_obj_tag(v_a_1813_) == 0)
{
lean_object* v___x_1815_; 
v___x_1815_ = l_List_reverse___redArg(v_a_1814_);
return v___x_1815_;
}
else
{
lean_object* v_head_1816_; lean_object* v_tail_1817_; lean_object* v___x_1819_; uint8_t v_isShared_1820_; uint8_t v_isSharedCheck_1826_; 
v_head_1816_ = lean_ctor_get(v_a_1813_, 0);
v_tail_1817_ = lean_ctor_get(v_a_1813_, 1);
v_isSharedCheck_1826_ = !lean_is_exclusive(v_a_1813_);
if (v_isSharedCheck_1826_ == 0)
{
v___x_1819_ = v_a_1813_;
v_isShared_1820_ = v_isSharedCheck_1826_;
goto v_resetjp_1818_;
}
else
{
lean_inc(v_tail_1817_);
lean_inc(v_head_1816_);
lean_dec(v_a_1813_);
v___x_1819_ = lean_box(0);
v_isShared_1820_ = v_isSharedCheck_1826_;
goto v_resetjp_1818_;
}
v_resetjp_1818_:
{
lean_object* v_fst_1821_; lean_object* v___x_1823_; 
v_fst_1821_ = lean_ctor_get(v_head_1816_, 0);
lean_inc(v_fst_1821_);
lean_dec(v_head_1816_);
if (v_isShared_1820_ == 0)
{
lean_ctor_set(v___x_1819_, 1, v_a_1814_);
lean_ctor_set(v___x_1819_, 0, v_fst_1821_);
v___x_1823_ = v___x_1819_;
goto v_reusejp_1822_;
}
else
{
lean_object* v_reuseFailAlloc_1825_; 
v_reuseFailAlloc_1825_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1825_, 0, v_fst_1821_);
lean_ctor_set(v_reuseFailAlloc_1825_, 1, v_a_1814_);
v___x_1823_ = v_reuseFailAlloc_1825_;
goto v_reusejp_1822_;
}
v_reusejp_1822_:
{
v_a_1813_ = v_tail_1817_;
v_a_1814_ = v___x_1823_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashSet_toList___at___00__private_Aesop_Forward_State_0__Aesop_ppPHashSet___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__10_spec__11(lean_object* v_s_1827_){
_start:
{
lean_object* v___x_1828_; lean_object* v___x_1829_; lean_object* v___x_1830_; 
v___x_1828_ = lp_aesop_Lean_PersistentHashMap_toList___at___00Lean_PersistentHashSet_toList___at___00__private_Aesop_Forward_State_0__Aesop_ppPHashSet___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__10_spec__11_spec__23___redArg(v_s_1827_);
v___x_1829_ = lean_box(0);
v___x_1830_ = lp_aesop_List_mapTR_loop___at___00Lean_PersistentHashSet_toList___at___00__private_Aesop_Forward_State_0__Aesop_ppPHashSet___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__10_spec__11_spec__24(v___x_1828_, v___x_1829_);
return v___x_1830_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashSet_toList___at___00__private_Aesop_Forward_State_0__Aesop_ppPHashSet___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__10_spec__11___boxed(lean_object* v_s_1831_){
_start:
{
lean_object* v_res_1832_; 
v_res_1832_ = lp_aesop_Lean_PersistentHashSet_toList___at___00__private_Aesop_Forward_State_0__Aesop_ppPHashSet___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__10_spec__11(v_s_1831_);
lean_dec_ref(v_s_1831_);
return v_res_1832_;
}
}
LEAN_EXPORT lean_object* lp_aesop_List_mapTR_loop___at___00Aesop_Goal_traceMetadata_spec__10(lean_object* v_a_1833_, lean_object* v_a_1834_){
_start:
{
if (lean_obj_tag(v_a_1833_) == 0)
{
lean_object* v___x_1835_; 
v___x_1835_ = l_List_reverse___redArg(v_a_1834_);
return v___x_1835_;
}
else
{
lean_object* v_head_1836_; lean_object* v_subst_1837_; lean_object* v_tail_1838_; lean_object* v___x_1840_; uint8_t v_isShared_1841_; uint8_t v_isSharedCheck_1875_; 
v_head_1836_ = lean_ctor_get(v_a_1833_, 0);
v_subst_1837_ = lean_ctor_get(v_head_1836_, 0);
lean_inc_ref(v_subst_1837_);
v_tail_1838_ = lean_ctor_get(v_a_1833_, 1);
v_isSharedCheck_1875_ = !lean_is_exclusive(v_a_1833_);
if (v_isSharedCheck_1875_ == 0)
{
lean_object* v_unused_1876_; 
v_unused_1876_ = lean_ctor_get(v_a_1833_, 0);
lean_dec(v_unused_1876_);
v___x_1840_ = v_a_1833_;
v_isShared_1841_ = v_isSharedCheck_1875_;
goto v_resetjp_1839_;
}
else
{
lean_inc(v_tail_1838_);
lean_dec(v_a_1833_);
v___x_1840_ = lean_box(0);
v_isShared_1841_ = v_isSharedCheck_1875_;
goto v_resetjp_1839_;
}
v_resetjp_1839_:
{
lean_object* v_premises_1842_; lean_object* v_levels_1843_; lean_object* v___x_1845_; uint8_t v_isShared_1846_; uint8_t v_isSharedCheck_1874_; 
v_premises_1842_ = lean_ctor_get(v_subst_1837_, 0);
v_levels_1843_ = lean_ctor_get(v_subst_1837_, 1);
v_isSharedCheck_1874_ = !lean_is_exclusive(v_subst_1837_);
if (v_isSharedCheck_1874_ == 0)
{
v___x_1845_ = v_subst_1837_;
v_isShared_1846_ = v_isSharedCheck_1874_;
goto v_resetjp_1844_;
}
else
{
lean_inc(v_levels_1843_);
lean_inc(v_premises_1842_);
lean_dec(v_subst_1837_);
v___x_1845_ = lean_box(0);
v_isShared_1846_ = v_isSharedCheck_1874_;
goto v_resetjp_1844_;
}
v_resetjp_1844_:
{
lean_object* v___x_1847_; lean_object* v___x_1848_; lean_object* v___x_1849_; size_t v_sz_1850_; size_t v___x_1851_; lean_object* v___x_1852_; lean_object* v_ps_1853_; lean_object* v___x_1854_; lean_object* v___x_1855_; size_t v_sz_1856_; lean_object* v___x_1857_; lean_object* v_ls_1858_; lean_object* v___x_1859_; lean_object* v___x_1860_; lean_object* v___x_1861_; lean_object* v___x_1862_; lean_object* v___x_1864_; 
v___x_1847_ = lean_unsigned_to_nat(0u);
v___x_1848_ = lean_array_get_size(v_premises_1842_);
v___x_1849_ = lp_aesop_Array_filterMapM___at___00Aesop_Goal_traceMetadata_spec__0(v_premises_1842_, v___x_1847_, v___x_1848_);
lean_dec_ref(v_premises_1842_);
v_sz_1850_ = lean_array_size(v___x_1849_);
v___x_1851_ = ((size_t)0ULL);
v___x_1852_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Aesop_Goal_traceMetadata_spec__1___redArg(v_sz_1850_, v___x_1851_, v___x_1849_);
v_ps_1853_ = lean_array_to_list(v___x_1852_);
v___x_1854_ = lean_array_get_size(v_levels_1843_);
v___x_1855_ = lp_aesop_Array_filterMapM___at___00Aesop_Goal_traceMetadata_spec__2(v_levels_1843_, v___x_1847_, v___x_1854_);
lean_dec_ref(v_levels_1843_);
v_sz_1856_ = lean_array_size(v___x_1855_);
v___x_1857_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Aesop_Goal_traceMetadata_spec__3___redArg(v_sz_1856_, v___x_1851_, v___x_1855_);
v_ls_1858_ = lean_array_to_list(v___x_1857_);
v___x_1859_ = ((lean_object*)(lp_aesop_List_mapTR_loop___at___00Aesop_Goal_traceMetadata_spec__4___closed__0));
v___x_1860_ = lean_obj_once(&lp_aesop_List_mapTR_loop___at___00Aesop_Goal_traceMetadata_spec__4___closed__3, &lp_aesop_List_mapTR_loop___at___00Aesop_Goal_traceMetadata_spec__4___closed__3_once, _init_lp_aesop_List_mapTR_loop___at___00Aesop_Goal_traceMetadata_spec__4___closed__3);
v___x_1861_ = l_Lean_MessageData_joinSep(v_ps_1853_, v___x_1860_);
v___x_1862_ = lean_obj_once(&lp_aesop_List_mapTR_loop___at___00Aesop_Goal_traceMetadata_spec__4___closed__6, &lp_aesop_List_mapTR_loop___at___00Aesop_Goal_traceMetadata_spec__4___closed__6_once, _init_lp_aesop_List_mapTR_loop___at___00Aesop_Goal_traceMetadata_spec__4___closed__6);
if (v_isShared_1846_ == 0)
{
lean_ctor_set_tag(v___x_1845_, 7);
lean_ctor_set(v___x_1845_, 1, v___x_1862_);
lean_ctor_set(v___x_1845_, 0, v___x_1861_);
v___x_1864_ = v___x_1845_;
goto v_reusejp_1863_;
}
else
{
lean_object* v_reuseFailAlloc_1873_; 
v_reuseFailAlloc_1873_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1873_, 0, v___x_1861_);
lean_ctor_set(v_reuseFailAlloc_1873_, 1, v___x_1862_);
v___x_1864_ = v_reuseFailAlloc_1873_;
goto v_reusejp_1863_;
}
v_reusejp_1863_:
{
lean_object* v___x_1865_; lean_object* v___x_1866_; lean_object* v___x_1867_; lean_object* v___x_1868_; lean_object* v___x_1870_; 
v___x_1865_ = l_Lean_MessageData_joinSep(v_ls_1858_, v___x_1860_);
v___x_1866_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1866_, 0, v___x_1864_);
lean_ctor_set(v___x_1866_, 1, v___x_1865_);
v___x_1867_ = ((lean_object*)(lp_aesop_List_mapTR_loop___at___00Aesop_Goal_traceMetadata_spec__4___closed__7));
v___x_1868_ = l_Lean_MessageData_bracket(v___x_1859_, v___x_1866_, v___x_1867_);
if (v_isShared_1841_ == 0)
{
lean_ctor_set(v___x_1840_, 1, v_a_1834_);
lean_ctor_set(v___x_1840_, 0, v___x_1868_);
v___x_1870_ = v___x_1840_;
goto v_reusejp_1869_;
}
else
{
lean_object* v_reuseFailAlloc_1872_; 
v_reuseFailAlloc_1872_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1872_, 0, v___x_1868_);
lean_ctor_set(v_reuseFailAlloc_1872_, 1, v_a_1834_);
v___x_1870_ = v_reuseFailAlloc_1872_;
goto v_reusejp_1869_;
}
v_reusejp_1869_:
{
v_a_1833_ = v_tail_1838_;
v_a_1834_ = v___x_1870_;
goto _start;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Forward_State_0__Aesop_ppPHashSet___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__10(lean_object* v_s_1877_){
_start:
{
lean_object* v___x_1878_; lean_object* v___x_1879_; lean_object* v___x_1880_; lean_object* v___x_1881_; 
v___x_1878_ = lp_aesop_Lean_PersistentHashSet_toList___at___00__private_Aesop_Forward_State_0__Aesop_ppPHashSet___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__10_spec__11(v_s_1877_);
v___x_1879_ = lean_box(0);
v___x_1880_ = lp_aesop_List_mapTR_loop___at___00Aesop_Goal_traceMetadata_spec__10(v___x_1878_, v___x_1879_);
v___x_1881_ = l_Lean_MessageData_ofList(v___x_1880_);
return v___x_1881_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Forward_State_0__Aesop_ppPHashSet___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__10___boxed(lean_object* v_s_1882_){
_start:
{
lean_object* v_res_1883_; 
v_res_1883_ = lp_aesop___private_Aesop_Forward_State_0__Aesop_ppPHashSet___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__10(v_s_1882_);
lean_dec_ref(v_s_1882_);
return v_res_1883_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7___lam__1___closed__2(void){
_start:
{
lean_object* v___x_1887_; lean_object* v___x_1888_; 
v___x_1887_ = ((lean_object*)(lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7___lam__1___closed__1));
v___x_1888_ = l_Lean_MessageData_ofFormat(v___x_1887_);
return v___x_1888_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7___lam__1___closed__3(void){
_start:
{
lean_object* v___x_1889_; lean_object* v___x_1890_; 
v___x_1889_ = lean_box(1);
v___x_1890_ = l_Lean_MessageData_ofFormat(v___x_1889_);
return v___x_1890_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7___lam__1(lean_object* v___f_1891_, lean_object* v_x_1892_, lean_object* v_x_1893_){
_start:
{
lean_object* v_fst_1894_; lean_object* v_snd_1895_; lean_object* v___x_1897_; uint8_t v_isShared_1898_; uint8_t v_isSharedCheck_1914_; 
v_fst_1894_ = lean_ctor_get(v_x_1893_, 0);
v_snd_1895_ = lean_ctor_get(v_x_1893_, 1);
v_isSharedCheck_1914_ = !lean_is_exclusive(v_x_1893_);
if (v_isSharedCheck_1914_ == 0)
{
v___x_1897_ = v_x_1893_;
v_isShared_1898_ = v_isSharedCheck_1914_;
goto v_resetjp_1896_;
}
else
{
lean_inc(v_snd_1895_);
lean_inc(v_fst_1894_);
lean_dec(v_x_1893_);
v___x_1897_ = lean_box(0);
v_isShared_1898_ = v_isSharedCheck_1914_;
goto v_resetjp_1896_;
}
v_resetjp_1896_:
{
lean_object* v___x_1899_; lean_object* v_hs_1900_; lean_object* v___x_1901_; lean_object* v___x_1902_; lean_object* v___x_1904_; 
v___x_1899_ = ((lean_object*)(lp_aesop_Lean_addTrace___at___00__private_Aesop_Tree_Tracing_0__Aesop_Goal_traceMetadata_trc_spec__0___closed__0));
v_hs_1900_ = lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_Goal_traceMetadata_spec__9_spec__18___redArg(v___f_1891_, v_snd_1895_, v___x_1899_);
lean_dec(v_snd_1895_);
v___x_1901_ = lp_aesop___private_Aesop_Forward_State_0__Aesop_ppPHashSet___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__10(v_fst_1894_);
lean_dec(v_fst_1894_);
v___x_1902_ = lean_obj_once(&lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7___lam__1___closed__2, &lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7___lam__1___closed__2_once, _init_lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7___lam__1___closed__2);
if (v_isShared_1898_ == 0)
{
lean_ctor_set_tag(v___x_1897_, 7);
lean_ctor_set(v___x_1897_, 1, v___x_1902_);
lean_ctor_set(v___x_1897_, 0, v___x_1901_);
v___x_1904_ = v___x_1897_;
goto v_reusejp_1903_;
}
else
{
lean_object* v_reuseFailAlloc_1913_; 
v_reuseFailAlloc_1913_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1913_, 0, v___x_1901_);
lean_ctor_set(v_reuseFailAlloc_1913_, 1, v___x_1902_);
v___x_1904_ = v_reuseFailAlloc_1913_;
goto v_reusejp_1903_;
}
v_reusejp_1903_:
{
lean_object* v___x_1905_; lean_object* v___x_1906_; lean_object* v___x_1907_; lean_object* v___x_1908_; lean_object* v___x_1909_; lean_object* v___x_1910_; lean_object* v___x_1911_; lean_object* v___x_1912_; 
v___x_1905_ = lean_obj_once(&lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7___lam__1___closed__3, &lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7___lam__1___closed__3_once, _init_lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7___lam__1___closed__3);
v___x_1906_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1906_, 0, v___x_1904_);
lean_ctor_set(v___x_1906_, 1, v___x_1905_);
v___x_1907_ = lean_array_to_list(v_hs_1900_);
v___x_1908_ = lean_box(0);
v___x_1909_ = lp_aesop_List_mapTR_loop___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__11(v___x_1907_, v___x_1908_);
v___x_1910_ = l_Lean_MessageData_ofList(v___x_1909_);
v___x_1911_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1911_, 0, v___x_1906_);
lean_ctor_set(v___x_1911_, 1, v___x_1910_);
v___x_1912_ = l_Lean_MessageData_paren(v___x_1911_);
return v___x_1912_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7___lam__1___boxed(lean_object* v___f_1915_, lean_object* v_x_1916_, lean_object* v_x_1917_){
_start:
{
lean_object* v_res_1918_; 
v_res_1918_ = lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7___lam__1(v___f_1915_, v_x_1916_, v_x_1917_);
lean_dec_ref(v_x_1916_);
return v_res_1918_;
}
}
static lean_object* _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12_spec__14_spec__29_spec__38___closed__1(void){
_start:
{
lean_object* v___x_1920_; lean_object* v___x_1921_; 
v___x_1920_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12_spec__14_spec__29_spec__38___closed__0));
v___x_1921_ = l_Lean_stringToMessageData(v___x_1920_);
return v___x_1921_;
}
}
static lean_object* _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12_spec__14_spec__29_spec__38___closed__3(void){
_start:
{
lean_object* v___x_1923_; lean_object* v___x_1924_; 
v___x_1923_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12_spec__14_spec__29_spec__38___closed__2));
v___x_1924_ = l_Lean_stringToMessageData(v___x_1923_);
return v___x_1924_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12_spec__14_spec__29_spec__38(uint8_t v_indent_1925_, lean_object* v_as_1926_, size_t v_sz_1927_, size_t v_i_1928_, lean_object* v_b_1929_){
_start:
{
uint8_t v___x_1930_; 
v___x_1930_ = lean_usize_dec_lt(v_i_1928_, v_sz_1927_);
if (v___x_1930_ == 0)
{
return v_b_1929_;
}
else
{
lean_object* v_snd_1931_; lean_object* v___x_1933_; uint8_t v_isShared_1934_; uint8_t v_isSharedCheck_1974_; 
v_snd_1931_ = lean_ctor_get(v_b_1929_, 1);
v_isSharedCheck_1974_ = !lean_is_exclusive(v_b_1929_);
if (v_isSharedCheck_1974_ == 0)
{
lean_object* v_unused_1975_; 
v_unused_1975_ = lean_ctor_get(v_b_1929_, 0);
lean_dec(v_unused_1975_);
v___x_1933_ = v_b_1929_;
v_isShared_1934_ = v_isSharedCheck_1974_;
goto v_resetjp_1932_;
}
else
{
lean_inc(v_snd_1931_);
lean_dec(v_b_1929_);
v___x_1933_ = lean_box(0);
v_isShared_1934_ = v_isSharedCheck_1974_;
goto v_resetjp_1932_;
}
v_resetjp_1932_:
{
lean_object* v___x_1935_; lean_object* v_a_1937_; lean_object* v___y_1945_; lean_object* v_a_1947_; 
v___x_1935_ = lean_box(0);
v_a_1947_ = lean_array_uget_borrowed(v_as_1926_, v_i_1928_);
if (lean_obj_tag(v_a_1947_) == 0)
{
v_a_1937_ = v_snd_1931_;
goto v___jp_1936_;
}
else
{
lean_object* v_val_1948_; 
v_val_1948_ = lean_ctor_get(v_a_1947_, 0);
lean_inc(v_val_1948_);
if (v_indent_1925_ == 0)
{
lean_object* v_fst_1949_; lean_object* v_snd_1950_; lean_object* v___x_1952_; uint8_t v_isShared_1953_; uint8_t v_isSharedCheck_1960_; 
v_fst_1949_ = lean_ctor_get(v_val_1948_, 0);
v_snd_1950_ = lean_ctor_get(v_val_1948_, 1);
v_isSharedCheck_1960_ = !lean_is_exclusive(v_val_1948_);
if (v_isSharedCheck_1960_ == 0)
{
v___x_1952_ = v_val_1948_;
v_isShared_1953_ = v_isSharedCheck_1960_;
goto v_resetjp_1951_;
}
else
{
lean_inc(v_snd_1950_);
lean_inc(v_fst_1949_);
lean_dec(v_val_1948_);
v___x_1952_ = lean_box(0);
v_isShared_1953_ = v_isSharedCheck_1960_;
goto v_resetjp_1951_;
}
v_resetjp_1951_:
{
lean_object* v___x_1954_; lean_object* v___x_1955_; lean_object* v___x_1957_; 
v___x_1954_ = l_Lean_MessageData_ofExpr(v_fst_1949_);
v___x_1955_ = lean_obj_once(&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12_spec__14_spec__29_spec__38___closed__1, &lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12_spec__14_spec__29_spec__38___closed__1_once, _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12_spec__14_spec__29_spec__38___closed__1);
if (v_isShared_1953_ == 0)
{
lean_ctor_set_tag(v___x_1952_, 7);
lean_ctor_set(v___x_1952_, 1, v___x_1955_);
lean_ctor_set(v___x_1952_, 0, v___x_1954_);
v___x_1957_ = v___x_1952_;
goto v_reusejp_1956_;
}
else
{
lean_object* v_reuseFailAlloc_1959_; 
v_reuseFailAlloc_1959_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1959_, 0, v___x_1954_);
lean_ctor_set(v_reuseFailAlloc_1959_, 1, v___x_1955_);
v___x_1957_ = v_reuseFailAlloc_1959_;
goto v_reusejp_1956_;
}
v_reusejp_1956_:
{
lean_object* v___x_1958_; 
v___x_1958_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1958_, 0, v___x_1957_);
lean_ctor_set(v___x_1958_, 1, v_snd_1950_);
v___y_1945_ = v___x_1958_;
goto v___jp_1944_;
}
}
}
else
{
lean_object* v_fst_1961_; lean_object* v_snd_1962_; lean_object* v___x_1964_; uint8_t v_isShared_1965_; uint8_t v_isSharedCheck_1973_; 
v_fst_1961_ = lean_ctor_get(v_val_1948_, 0);
v_snd_1962_ = lean_ctor_get(v_val_1948_, 1);
v_isSharedCheck_1973_ = !lean_is_exclusive(v_val_1948_);
if (v_isSharedCheck_1973_ == 0)
{
v___x_1964_ = v_val_1948_;
v_isShared_1965_ = v_isSharedCheck_1973_;
goto v_resetjp_1963_;
}
else
{
lean_inc(v_snd_1962_);
lean_inc(v_fst_1961_);
lean_dec(v_val_1948_);
v___x_1964_ = lean_box(0);
v_isShared_1965_ = v_isSharedCheck_1973_;
goto v_resetjp_1963_;
}
v_resetjp_1963_:
{
lean_object* v___x_1966_; lean_object* v___x_1967_; lean_object* v___x_1969_; 
v___x_1966_ = l_Lean_MessageData_ofExpr(v_fst_1961_);
v___x_1967_ = lean_obj_once(&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12_spec__14_spec__29_spec__38___closed__3, &lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12_spec__14_spec__29_spec__38___closed__3_once, _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12_spec__14_spec__29_spec__38___closed__3);
if (v_isShared_1965_ == 0)
{
lean_ctor_set_tag(v___x_1964_, 7);
lean_ctor_set(v___x_1964_, 1, v___x_1967_);
lean_ctor_set(v___x_1964_, 0, v___x_1966_);
v___x_1969_ = v___x_1964_;
goto v_reusejp_1968_;
}
else
{
lean_object* v_reuseFailAlloc_1972_; 
v_reuseFailAlloc_1972_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1972_, 0, v___x_1966_);
lean_ctor_set(v_reuseFailAlloc_1972_, 1, v___x_1967_);
v___x_1969_ = v_reuseFailAlloc_1972_;
goto v_reusejp_1968_;
}
v_reusejp_1968_:
{
lean_object* v___x_1970_; lean_object* v___x_1971_; 
v___x_1970_ = l_Lean_indentD(v_snd_1962_);
v___x_1971_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1971_, 0, v___x_1969_);
lean_ctor_set(v___x_1971_, 1, v___x_1970_);
v___y_1945_ = v___x_1971_;
goto v___jp_1944_;
}
}
}
}
v___jp_1936_:
{
lean_object* v___x_1939_; 
if (v_isShared_1934_ == 0)
{
lean_ctor_set(v___x_1933_, 1, v_a_1937_);
lean_ctor_set(v___x_1933_, 0, v___x_1935_);
v___x_1939_ = v___x_1933_;
goto v_reusejp_1938_;
}
else
{
lean_object* v_reuseFailAlloc_1943_; 
v_reuseFailAlloc_1943_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1943_, 0, v___x_1935_);
lean_ctor_set(v_reuseFailAlloc_1943_, 1, v_a_1937_);
v___x_1939_ = v_reuseFailAlloc_1943_;
goto v_reusejp_1938_;
}
v_reusejp_1938_:
{
size_t v___x_1940_; size_t v___x_1941_; 
v___x_1940_ = ((size_t)1ULL);
v___x_1941_ = lean_usize_add(v_i_1928_, v___x_1940_);
v_i_1928_ = v___x_1941_;
v_b_1929_ = v___x_1939_;
goto _start;
}
}
v___jp_1944_:
{
lean_object* v_entries_1946_; 
v_entries_1946_ = lean_array_push(v_snd_1931_, v___y_1945_);
v_a_1937_ = v_entries_1946_;
goto v___jp_1936_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12_spec__14_spec__29_spec__38___boxed(lean_object* v_indent_1976_, lean_object* v_as_1977_, lean_object* v_sz_1978_, lean_object* v_i_1979_, lean_object* v_b_1980_){
_start:
{
uint8_t v_indent_boxed_1981_; size_t v_sz_boxed_1982_; size_t v_i_boxed_1983_; lean_object* v_res_1984_; 
v_indent_boxed_1981_ = lean_unbox(v_indent_1976_);
v_sz_boxed_1982_ = lean_unbox_usize(v_sz_1978_);
lean_dec(v_sz_1978_);
v_i_boxed_1983_ = lean_unbox_usize(v_i_1979_);
lean_dec(v_i_1979_);
v_res_1984_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12_spec__14_spec__29_spec__38(v_indent_boxed_1981_, v_as_1977_, v_sz_boxed_1982_, v_i_boxed_1983_, v_b_1980_);
lean_dec_ref(v_as_1977_);
return v_res_1984_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12_spec__14_spec__29(uint8_t v_indent_1985_, lean_object* v_as_1986_, size_t v_sz_1987_, size_t v_i_1988_, lean_object* v_b_1989_){
_start:
{
uint8_t v___x_1990_; 
v___x_1990_ = lean_usize_dec_lt(v_i_1988_, v_sz_1987_);
if (v___x_1990_ == 0)
{
return v_b_1989_;
}
else
{
lean_object* v_snd_1991_; lean_object* v___x_1993_; uint8_t v_isShared_1994_; uint8_t v_isSharedCheck_2034_; 
v_snd_1991_ = lean_ctor_get(v_b_1989_, 1);
v_isSharedCheck_2034_ = !lean_is_exclusive(v_b_1989_);
if (v_isSharedCheck_2034_ == 0)
{
lean_object* v_unused_2035_; 
v_unused_2035_ = lean_ctor_get(v_b_1989_, 0);
lean_dec(v_unused_2035_);
v___x_1993_ = v_b_1989_;
v_isShared_1994_ = v_isSharedCheck_2034_;
goto v_resetjp_1992_;
}
else
{
lean_inc(v_snd_1991_);
lean_dec(v_b_1989_);
v___x_1993_ = lean_box(0);
v_isShared_1994_ = v_isSharedCheck_2034_;
goto v_resetjp_1992_;
}
v_resetjp_1992_:
{
lean_object* v___x_1995_; lean_object* v_a_1997_; lean_object* v___y_2005_; lean_object* v_a_2007_; 
v___x_1995_ = lean_box(0);
v_a_2007_ = lean_array_uget_borrowed(v_as_1986_, v_i_1988_);
if (lean_obj_tag(v_a_2007_) == 0)
{
v_a_1997_ = v_snd_1991_;
goto v___jp_1996_;
}
else
{
lean_object* v_val_2008_; 
v_val_2008_ = lean_ctor_get(v_a_2007_, 0);
lean_inc(v_val_2008_);
if (v_indent_1985_ == 0)
{
lean_object* v_fst_2009_; lean_object* v_snd_2010_; lean_object* v___x_2012_; uint8_t v_isShared_2013_; uint8_t v_isSharedCheck_2020_; 
v_fst_2009_ = lean_ctor_get(v_val_2008_, 0);
v_snd_2010_ = lean_ctor_get(v_val_2008_, 1);
v_isSharedCheck_2020_ = !lean_is_exclusive(v_val_2008_);
if (v_isSharedCheck_2020_ == 0)
{
v___x_2012_ = v_val_2008_;
v_isShared_2013_ = v_isSharedCheck_2020_;
goto v_resetjp_2011_;
}
else
{
lean_inc(v_snd_2010_);
lean_inc(v_fst_2009_);
lean_dec(v_val_2008_);
v___x_2012_ = lean_box(0);
v_isShared_2013_ = v_isSharedCheck_2020_;
goto v_resetjp_2011_;
}
v_resetjp_2011_:
{
lean_object* v___x_2014_; lean_object* v___x_2015_; lean_object* v___x_2017_; 
v___x_2014_ = l_Lean_MessageData_ofExpr(v_fst_2009_);
v___x_2015_ = lean_obj_once(&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12_spec__14_spec__29_spec__38___closed__1, &lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12_spec__14_spec__29_spec__38___closed__1_once, _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12_spec__14_spec__29_spec__38___closed__1);
if (v_isShared_2013_ == 0)
{
lean_ctor_set_tag(v___x_2012_, 7);
lean_ctor_set(v___x_2012_, 1, v___x_2015_);
lean_ctor_set(v___x_2012_, 0, v___x_2014_);
v___x_2017_ = v___x_2012_;
goto v_reusejp_2016_;
}
else
{
lean_object* v_reuseFailAlloc_2019_; 
v_reuseFailAlloc_2019_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2019_, 0, v___x_2014_);
lean_ctor_set(v_reuseFailAlloc_2019_, 1, v___x_2015_);
v___x_2017_ = v_reuseFailAlloc_2019_;
goto v_reusejp_2016_;
}
v_reusejp_2016_:
{
lean_object* v___x_2018_; 
v___x_2018_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2018_, 0, v___x_2017_);
lean_ctor_set(v___x_2018_, 1, v_snd_2010_);
v___y_2005_ = v___x_2018_;
goto v___jp_2004_;
}
}
}
else
{
lean_object* v_fst_2021_; lean_object* v_snd_2022_; lean_object* v___x_2024_; uint8_t v_isShared_2025_; uint8_t v_isSharedCheck_2033_; 
v_fst_2021_ = lean_ctor_get(v_val_2008_, 0);
v_snd_2022_ = lean_ctor_get(v_val_2008_, 1);
v_isSharedCheck_2033_ = !lean_is_exclusive(v_val_2008_);
if (v_isSharedCheck_2033_ == 0)
{
v___x_2024_ = v_val_2008_;
v_isShared_2025_ = v_isSharedCheck_2033_;
goto v_resetjp_2023_;
}
else
{
lean_inc(v_snd_2022_);
lean_inc(v_fst_2021_);
lean_dec(v_val_2008_);
v___x_2024_ = lean_box(0);
v_isShared_2025_ = v_isSharedCheck_2033_;
goto v_resetjp_2023_;
}
v_resetjp_2023_:
{
lean_object* v___x_2026_; lean_object* v___x_2027_; lean_object* v___x_2029_; 
v___x_2026_ = l_Lean_MessageData_ofExpr(v_fst_2021_);
v___x_2027_ = lean_obj_once(&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12_spec__14_spec__29_spec__38___closed__3, &lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12_spec__14_spec__29_spec__38___closed__3_once, _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12_spec__14_spec__29_spec__38___closed__3);
if (v_isShared_2025_ == 0)
{
lean_ctor_set_tag(v___x_2024_, 7);
lean_ctor_set(v___x_2024_, 1, v___x_2027_);
lean_ctor_set(v___x_2024_, 0, v___x_2026_);
v___x_2029_ = v___x_2024_;
goto v_reusejp_2028_;
}
else
{
lean_object* v_reuseFailAlloc_2032_; 
v_reuseFailAlloc_2032_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2032_, 0, v___x_2026_);
lean_ctor_set(v_reuseFailAlloc_2032_, 1, v___x_2027_);
v___x_2029_ = v_reuseFailAlloc_2032_;
goto v_reusejp_2028_;
}
v_reusejp_2028_:
{
lean_object* v___x_2030_; lean_object* v___x_2031_; 
v___x_2030_ = l_Lean_indentD(v_snd_2022_);
v___x_2031_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2031_, 0, v___x_2029_);
lean_ctor_set(v___x_2031_, 1, v___x_2030_);
v___y_2005_ = v___x_2031_;
goto v___jp_2004_;
}
}
}
}
v___jp_1996_:
{
lean_object* v___x_1999_; 
if (v_isShared_1994_ == 0)
{
lean_ctor_set(v___x_1993_, 1, v_a_1997_);
lean_ctor_set(v___x_1993_, 0, v___x_1995_);
v___x_1999_ = v___x_1993_;
goto v_reusejp_1998_;
}
else
{
lean_object* v_reuseFailAlloc_2003_; 
v_reuseFailAlloc_2003_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2003_, 0, v___x_1995_);
lean_ctor_set(v_reuseFailAlloc_2003_, 1, v_a_1997_);
v___x_1999_ = v_reuseFailAlloc_2003_;
goto v_reusejp_1998_;
}
v_reusejp_1998_:
{
size_t v___x_2000_; size_t v___x_2001_; lean_object* v___x_2002_; 
v___x_2000_ = ((size_t)1ULL);
v___x_2001_ = lean_usize_add(v_i_1988_, v___x_2000_);
v___x_2002_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12_spec__14_spec__29_spec__38(v_indent_1985_, v_as_1986_, v_sz_1987_, v___x_2001_, v___x_1999_);
return v___x_2002_;
}
}
v___jp_2004_:
{
lean_object* v_entries_2006_; 
v_entries_2006_ = lean_array_push(v_snd_1991_, v___y_2005_);
v_a_1997_ = v_entries_2006_;
goto v___jp_1996_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12_spec__14_spec__29___boxed(lean_object* v_indent_2036_, lean_object* v_as_2037_, lean_object* v_sz_2038_, lean_object* v_i_2039_, lean_object* v_b_2040_){
_start:
{
uint8_t v_indent_boxed_2041_; size_t v_sz_boxed_2042_; size_t v_i_boxed_2043_; lean_object* v_res_2044_; 
v_indent_boxed_2041_ = lean_unbox(v_indent_2036_);
v_sz_boxed_2042_ = lean_unbox_usize(v_sz_2038_);
lean_dec(v_sz_2038_);
v_i_boxed_2043_ = lean_unbox_usize(v_i_2039_);
lean_dec(v_i_2039_);
v_res_2044_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12_spec__14_spec__29(v_indent_boxed_2041_, v_as_2037_, v_sz_boxed_2042_, v_i_boxed_2043_, v_b_2040_);
lean_dec_ref(v_as_2037_);
return v_res_2044_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12_spec__14_spec__28_spec__36_spec__42(uint8_t v_indent_2045_, lean_object* v_as_2046_, size_t v_sz_2047_, size_t v_i_2048_, lean_object* v_b_2049_){
_start:
{
uint8_t v___x_2050_; 
v___x_2050_ = lean_usize_dec_lt(v_i_2048_, v_sz_2047_);
if (v___x_2050_ == 0)
{
return v_b_2049_;
}
else
{
lean_object* v_snd_2051_; lean_object* v___x_2053_; uint8_t v_isShared_2054_; uint8_t v_isSharedCheck_2094_; 
v_snd_2051_ = lean_ctor_get(v_b_2049_, 1);
v_isSharedCheck_2094_ = !lean_is_exclusive(v_b_2049_);
if (v_isSharedCheck_2094_ == 0)
{
lean_object* v_unused_2095_; 
v_unused_2095_ = lean_ctor_get(v_b_2049_, 0);
lean_dec(v_unused_2095_);
v___x_2053_ = v_b_2049_;
v_isShared_2054_ = v_isSharedCheck_2094_;
goto v_resetjp_2052_;
}
else
{
lean_inc(v_snd_2051_);
lean_dec(v_b_2049_);
v___x_2053_ = lean_box(0);
v_isShared_2054_ = v_isSharedCheck_2094_;
goto v_resetjp_2052_;
}
v_resetjp_2052_:
{
lean_object* v___x_2055_; lean_object* v_a_2057_; lean_object* v___y_2065_; lean_object* v_a_2067_; 
v___x_2055_ = lean_box(0);
v_a_2067_ = lean_array_uget_borrowed(v_as_2046_, v_i_2048_);
if (lean_obj_tag(v_a_2067_) == 0)
{
v_a_2057_ = v_snd_2051_;
goto v___jp_2056_;
}
else
{
lean_object* v_val_2068_; 
v_val_2068_ = lean_ctor_get(v_a_2067_, 0);
lean_inc(v_val_2068_);
if (v_indent_2045_ == 0)
{
lean_object* v_fst_2069_; lean_object* v_snd_2070_; lean_object* v___x_2072_; uint8_t v_isShared_2073_; uint8_t v_isSharedCheck_2080_; 
v_fst_2069_ = lean_ctor_get(v_val_2068_, 0);
v_snd_2070_ = lean_ctor_get(v_val_2068_, 1);
v_isSharedCheck_2080_ = !lean_is_exclusive(v_val_2068_);
if (v_isSharedCheck_2080_ == 0)
{
v___x_2072_ = v_val_2068_;
v_isShared_2073_ = v_isSharedCheck_2080_;
goto v_resetjp_2071_;
}
else
{
lean_inc(v_snd_2070_);
lean_inc(v_fst_2069_);
lean_dec(v_val_2068_);
v___x_2072_ = lean_box(0);
v_isShared_2073_ = v_isSharedCheck_2080_;
goto v_resetjp_2071_;
}
v_resetjp_2071_:
{
lean_object* v___x_2074_; lean_object* v___x_2075_; lean_object* v___x_2077_; 
v___x_2074_ = l_Lean_MessageData_ofExpr(v_fst_2069_);
v___x_2075_ = lean_obj_once(&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12_spec__14_spec__29_spec__38___closed__1, &lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12_spec__14_spec__29_spec__38___closed__1_once, _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12_spec__14_spec__29_spec__38___closed__1);
if (v_isShared_2073_ == 0)
{
lean_ctor_set_tag(v___x_2072_, 7);
lean_ctor_set(v___x_2072_, 1, v___x_2075_);
lean_ctor_set(v___x_2072_, 0, v___x_2074_);
v___x_2077_ = v___x_2072_;
goto v_reusejp_2076_;
}
else
{
lean_object* v_reuseFailAlloc_2079_; 
v_reuseFailAlloc_2079_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2079_, 0, v___x_2074_);
lean_ctor_set(v_reuseFailAlloc_2079_, 1, v___x_2075_);
v___x_2077_ = v_reuseFailAlloc_2079_;
goto v_reusejp_2076_;
}
v_reusejp_2076_:
{
lean_object* v___x_2078_; 
v___x_2078_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2078_, 0, v___x_2077_);
lean_ctor_set(v___x_2078_, 1, v_snd_2070_);
v___y_2065_ = v___x_2078_;
goto v___jp_2064_;
}
}
}
else
{
lean_object* v_fst_2081_; lean_object* v_snd_2082_; lean_object* v___x_2084_; uint8_t v_isShared_2085_; uint8_t v_isSharedCheck_2093_; 
v_fst_2081_ = lean_ctor_get(v_val_2068_, 0);
v_snd_2082_ = lean_ctor_get(v_val_2068_, 1);
v_isSharedCheck_2093_ = !lean_is_exclusive(v_val_2068_);
if (v_isSharedCheck_2093_ == 0)
{
v___x_2084_ = v_val_2068_;
v_isShared_2085_ = v_isSharedCheck_2093_;
goto v_resetjp_2083_;
}
else
{
lean_inc(v_snd_2082_);
lean_inc(v_fst_2081_);
lean_dec(v_val_2068_);
v___x_2084_ = lean_box(0);
v_isShared_2085_ = v_isSharedCheck_2093_;
goto v_resetjp_2083_;
}
v_resetjp_2083_:
{
lean_object* v___x_2086_; lean_object* v___x_2087_; lean_object* v___x_2089_; 
v___x_2086_ = l_Lean_MessageData_ofExpr(v_fst_2081_);
v___x_2087_ = lean_obj_once(&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12_spec__14_spec__29_spec__38___closed__3, &lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12_spec__14_spec__29_spec__38___closed__3_once, _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12_spec__14_spec__29_spec__38___closed__3);
if (v_isShared_2085_ == 0)
{
lean_ctor_set_tag(v___x_2084_, 7);
lean_ctor_set(v___x_2084_, 1, v___x_2087_);
lean_ctor_set(v___x_2084_, 0, v___x_2086_);
v___x_2089_ = v___x_2084_;
goto v_reusejp_2088_;
}
else
{
lean_object* v_reuseFailAlloc_2092_; 
v_reuseFailAlloc_2092_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2092_, 0, v___x_2086_);
lean_ctor_set(v_reuseFailAlloc_2092_, 1, v___x_2087_);
v___x_2089_ = v_reuseFailAlloc_2092_;
goto v_reusejp_2088_;
}
v_reusejp_2088_:
{
lean_object* v___x_2090_; lean_object* v___x_2091_; 
v___x_2090_ = l_Lean_indentD(v_snd_2082_);
v___x_2091_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2091_, 0, v___x_2089_);
lean_ctor_set(v___x_2091_, 1, v___x_2090_);
v___y_2065_ = v___x_2091_;
goto v___jp_2064_;
}
}
}
}
v___jp_2056_:
{
lean_object* v___x_2059_; 
if (v_isShared_2054_ == 0)
{
lean_ctor_set(v___x_2053_, 1, v_a_2057_);
lean_ctor_set(v___x_2053_, 0, v___x_2055_);
v___x_2059_ = v___x_2053_;
goto v_reusejp_2058_;
}
else
{
lean_object* v_reuseFailAlloc_2063_; 
v_reuseFailAlloc_2063_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2063_, 0, v___x_2055_);
lean_ctor_set(v_reuseFailAlloc_2063_, 1, v_a_2057_);
v___x_2059_ = v_reuseFailAlloc_2063_;
goto v_reusejp_2058_;
}
v_reusejp_2058_:
{
size_t v___x_2060_; size_t v___x_2061_; 
v___x_2060_ = ((size_t)1ULL);
v___x_2061_ = lean_usize_add(v_i_2048_, v___x_2060_);
v_i_2048_ = v___x_2061_;
v_b_2049_ = v___x_2059_;
goto _start;
}
}
v___jp_2064_:
{
lean_object* v_entries_2066_; 
v_entries_2066_ = lean_array_push(v_snd_2051_, v___y_2065_);
v_a_2057_ = v_entries_2066_;
goto v___jp_2056_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12_spec__14_spec__28_spec__36_spec__42___boxed(lean_object* v_indent_2096_, lean_object* v_as_2097_, lean_object* v_sz_2098_, lean_object* v_i_2099_, lean_object* v_b_2100_){
_start:
{
uint8_t v_indent_boxed_2101_; size_t v_sz_boxed_2102_; size_t v_i_boxed_2103_; lean_object* v_res_2104_; 
v_indent_boxed_2101_ = lean_unbox(v_indent_2096_);
v_sz_boxed_2102_ = lean_unbox_usize(v_sz_2098_);
lean_dec(v_sz_2098_);
v_i_boxed_2103_ = lean_unbox_usize(v_i_2099_);
lean_dec(v_i_2099_);
v_res_2104_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12_spec__14_spec__28_spec__36_spec__42(v_indent_boxed_2101_, v_as_2097_, v_sz_boxed_2102_, v_i_boxed_2103_, v_b_2100_);
lean_dec_ref(v_as_2097_);
return v_res_2104_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12_spec__14_spec__28_spec__36(uint8_t v_indent_2105_, lean_object* v_as_2106_, size_t v_sz_2107_, size_t v_i_2108_, lean_object* v_b_2109_){
_start:
{
uint8_t v___x_2110_; 
v___x_2110_ = lean_usize_dec_lt(v_i_2108_, v_sz_2107_);
if (v___x_2110_ == 0)
{
return v_b_2109_;
}
else
{
lean_object* v_snd_2111_; lean_object* v___x_2113_; uint8_t v_isShared_2114_; uint8_t v_isSharedCheck_2154_; 
v_snd_2111_ = lean_ctor_get(v_b_2109_, 1);
v_isSharedCheck_2154_ = !lean_is_exclusive(v_b_2109_);
if (v_isSharedCheck_2154_ == 0)
{
lean_object* v_unused_2155_; 
v_unused_2155_ = lean_ctor_get(v_b_2109_, 0);
lean_dec(v_unused_2155_);
v___x_2113_ = v_b_2109_;
v_isShared_2114_ = v_isSharedCheck_2154_;
goto v_resetjp_2112_;
}
else
{
lean_inc(v_snd_2111_);
lean_dec(v_b_2109_);
v___x_2113_ = lean_box(0);
v_isShared_2114_ = v_isSharedCheck_2154_;
goto v_resetjp_2112_;
}
v_resetjp_2112_:
{
lean_object* v___x_2115_; lean_object* v_a_2117_; lean_object* v___y_2125_; lean_object* v_a_2127_; 
v___x_2115_ = lean_box(0);
v_a_2127_ = lean_array_uget_borrowed(v_as_2106_, v_i_2108_);
if (lean_obj_tag(v_a_2127_) == 0)
{
v_a_2117_ = v_snd_2111_;
goto v___jp_2116_;
}
else
{
lean_object* v_val_2128_; 
v_val_2128_ = lean_ctor_get(v_a_2127_, 0);
lean_inc(v_val_2128_);
if (v_indent_2105_ == 0)
{
lean_object* v_fst_2129_; lean_object* v_snd_2130_; lean_object* v___x_2132_; uint8_t v_isShared_2133_; uint8_t v_isSharedCheck_2140_; 
v_fst_2129_ = lean_ctor_get(v_val_2128_, 0);
v_snd_2130_ = lean_ctor_get(v_val_2128_, 1);
v_isSharedCheck_2140_ = !lean_is_exclusive(v_val_2128_);
if (v_isSharedCheck_2140_ == 0)
{
v___x_2132_ = v_val_2128_;
v_isShared_2133_ = v_isSharedCheck_2140_;
goto v_resetjp_2131_;
}
else
{
lean_inc(v_snd_2130_);
lean_inc(v_fst_2129_);
lean_dec(v_val_2128_);
v___x_2132_ = lean_box(0);
v_isShared_2133_ = v_isSharedCheck_2140_;
goto v_resetjp_2131_;
}
v_resetjp_2131_:
{
lean_object* v___x_2134_; lean_object* v___x_2135_; lean_object* v___x_2137_; 
v___x_2134_ = l_Lean_MessageData_ofExpr(v_fst_2129_);
v___x_2135_ = lean_obj_once(&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12_spec__14_spec__29_spec__38___closed__1, &lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12_spec__14_spec__29_spec__38___closed__1_once, _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12_spec__14_spec__29_spec__38___closed__1);
if (v_isShared_2133_ == 0)
{
lean_ctor_set_tag(v___x_2132_, 7);
lean_ctor_set(v___x_2132_, 1, v___x_2135_);
lean_ctor_set(v___x_2132_, 0, v___x_2134_);
v___x_2137_ = v___x_2132_;
goto v_reusejp_2136_;
}
else
{
lean_object* v_reuseFailAlloc_2139_; 
v_reuseFailAlloc_2139_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2139_, 0, v___x_2134_);
lean_ctor_set(v_reuseFailAlloc_2139_, 1, v___x_2135_);
v___x_2137_ = v_reuseFailAlloc_2139_;
goto v_reusejp_2136_;
}
v_reusejp_2136_:
{
lean_object* v___x_2138_; 
v___x_2138_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2138_, 0, v___x_2137_);
lean_ctor_set(v___x_2138_, 1, v_snd_2130_);
v___y_2125_ = v___x_2138_;
goto v___jp_2124_;
}
}
}
else
{
lean_object* v_fst_2141_; lean_object* v_snd_2142_; lean_object* v___x_2144_; uint8_t v_isShared_2145_; uint8_t v_isSharedCheck_2153_; 
v_fst_2141_ = lean_ctor_get(v_val_2128_, 0);
v_snd_2142_ = lean_ctor_get(v_val_2128_, 1);
v_isSharedCheck_2153_ = !lean_is_exclusive(v_val_2128_);
if (v_isSharedCheck_2153_ == 0)
{
v___x_2144_ = v_val_2128_;
v_isShared_2145_ = v_isSharedCheck_2153_;
goto v_resetjp_2143_;
}
else
{
lean_inc(v_snd_2142_);
lean_inc(v_fst_2141_);
lean_dec(v_val_2128_);
v___x_2144_ = lean_box(0);
v_isShared_2145_ = v_isSharedCheck_2153_;
goto v_resetjp_2143_;
}
v_resetjp_2143_:
{
lean_object* v___x_2146_; lean_object* v___x_2147_; lean_object* v___x_2149_; 
v___x_2146_ = l_Lean_MessageData_ofExpr(v_fst_2141_);
v___x_2147_ = lean_obj_once(&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12_spec__14_spec__29_spec__38___closed__3, &lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12_spec__14_spec__29_spec__38___closed__3_once, _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12_spec__14_spec__29_spec__38___closed__3);
if (v_isShared_2145_ == 0)
{
lean_ctor_set_tag(v___x_2144_, 7);
lean_ctor_set(v___x_2144_, 1, v___x_2147_);
lean_ctor_set(v___x_2144_, 0, v___x_2146_);
v___x_2149_ = v___x_2144_;
goto v_reusejp_2148_;
}
else
{
lean_object* v_reuseFailAlloc_2152_; 
v_reuseFailAlloc_2152_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2152_, 0, v___x_2146_);
lean_ctor_set(v_reuseFailAlloc_2152_, 1, v___x_2147_);
v___x_2149_ = v_reuseFailAlloc_2152_;
goto v_reusejp_2148_;
}
v_reusejp_2148_:
{
lean_object* v___x_2150_; lean_object* v___x_2151_; 
v___x_2150_ = l_Lean_indentD(v_snd_2142_);
v___x_2151_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2151_, 0, v___x_2149_);
lean_ctor_set(v___x_2151_, 1, v___x_2150_);
v___y_2125_ = v___x_2151_;
goto v___jp_2124_;
}
}
}
}
v___jp_2116_:
{
lean_object* v___x_2119_; 
if (v_isShared_2114_ == 0)
{
lean_ctor_set(v___x_2113_, 1, v_a_2117_);
lean_ctor_set(v___x_2113_, 0, v___x_2115_);
v___x_2119_ = v___x_2113_;
goto v_reusejp_2118_;
}
else
{
lean_object* v_reuseFailAlloc_2123_; 
v_reuseFailAlloc_2123_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2123_, 0, v___x_2115_);
lean_ctor_set(v_reuseFailAlloc_2123_, 1, v_a_2117_);
v___x_2119_ = v_reuseFailAlloc_2123_;
goto v_reusejp_2118_;
}
v_reusejp_2118_:
{
size_t v___x_2120_; size_t v___x_2121_; lean_object* v___x_2122_; 
v___x_2120_ = ((size_t)1ULL);
v___x_2121_ = lean_usize_add(v_i_2108_, v___x_2120_);
v___x_2122_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12_spec__14_spec__28_spec__36_spec__42(v_indent_2105_, v_as_2106_, v_sz_2107_, v___x_2121_, v___x_2119_);
return v___x_2122_;
}
}
v___jp_2124_:
{
lean_object* v_entries_2126_; 
v_entries_2126_ = lean_array_push(v_snd_2111_, v___y_2125_);
v_a_2117_ = v_entries_2126_;
goto v___jp_2116_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12_spec__14_spec__28_spec__36___boxed(lean_object* v_indent_2156_, lean_object* v_as_2157_, lean_object* v_sz_2158_, lean_object* v_i_2159_, lean_object* v_b_2160_){
_start:
{
uint8_t v_indent_boxed_2161_; size_t v_sz_boxed_2162_; size_t v_i_boxed_2163_; lean_object* v_res_2164_; 
v_indent_boxed_2161_ = lean_unbox(v_indent_2156_);
v_sz_boxed_2162_ = lean_unbox_usize(v_sz_2158_);
lean_dec(v_sz_2158_);
v_i_boxed_2163_ = lean_unbox_usize(v_i_2159_);
lean_dec(v_i_2159_);
v_res_2164_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12_spec__14_spec__28_spec__36(v_indent_boxed_2161_, v_as_2157_, v_sz_boxed_2162_, v_i_boxed_2163_, v_b_2160_);
lean_dec_ref(v_as_2157_);
return v_res_2164_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12_spec__14_spec__28(lean_object* v_init_2165_, uint8_t v_indent_2166_, lean_object* v_n_2167_, lean_object* v_b_2168_){
_start:
{
if (lean_obj_tag(v_n_2167_) == 0)
{
lean_object* v_cs_2169_; lean_object* v___x_2170_; lean_object* v___x_2171_; size_t v_sz_2172_; size_t v___x_2173_; lean_object* v___x_2174_; lean_object* v_fst_2175_; 
v_cs_2169_ = lean_ctor_get(v_n_2167_, 0);
v___x_2170_ = lean_box(0);
v___x_2171_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2171_, 0, v___x_2170_);
lean_ctor_set(v___x_2171_, 1, v_b_2168_);
v_sz_2172_ = lean_array_size(v_cs_2169_);
v___x_2173_ = ((size_t)0ULL);
v___x_2174_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12_spec__14_spec__28_spec__35(v_init_2165_, v_indent_2166_, v_cs_2169_, v_sz_2172_, v___x_2173_, v___x_2171_);
v_fst_2175_ = lean_ctor_get(v___x_2174_, 0);
lean_inc(v_fst_2175_);
if (lean_obj_tag(v_fst_2175_) == 0)
{
lean_object* v_snd_2176_; lean_object* v___x_2177_; 
v_snd_2176_ = lean_ctor_get(v___x_2174_, 1);
lean_inc(v_snd_2176_);
lean_dec_ref(v___x_2174_);
v___x_2177_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2177_, 0, v_snd_2176_);
return v___x_2177_;
}
else
{
lean_object* v_val_2178_; 
lean_dec_ref(v___x_2174_);
v_val_2178_ = lean_ctor_get(v_fst_2175_, 0);
lean_inc(v_val_2178_);
lean_dec_ref_known(v_fst_2175_, 1);
return v_val_2178_;
}
}
else
{
lean_object* v_vs_2179_; lean_object* v___x_2180_; lean_object* v___x_2181_; size_t v_sz_2182_; size_t v___x_2183_; lean_object* v___x_2184_; lean_object* v_fst_2185_; 
v_vs_2179_ = lean_ctor_get(v_n_2167_, 0);
v___x_2180_ = lean_box(0);
v___x_2181_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2181_, 0, v___x_2180_);
lean_ctor_set(v___x_2181_, 1, v_b_2168_);
v_sz_2182_ = lean_array_size(v_vs_2179_);
v___x_2183_ = ((size_t)0ULL);
v___x_2184_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12_spec__14_spec__28_spec__36(v_indent_2166_, v_vs_2179_, v_sz_2182_, v___x_2183_, v___x_2181_);
v_fst_2185_ = lean_ctor_get(v___x_2184_, 0);
lean_inc(v_fst_2185_);
if (lean_obj_tag(v_fst_2185_) == 0)
{
lean_object* v_snd_2186_; lean_object* v___x_2187_; 
v_snd_2186_ = lean_ctor_get(v___x_2184_, 1);
lean_inc(v_snd_2186_);
lean_dec_ref(v___x_2184_);
v___x_2187_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2187_, 0, v_snd_2186_);
return v___x_2187_;
}
else
{
lean_object* v_val_2188_; 
lean_dec_ref(v___x_2184_);
v_val_2188_ = lean_ctor_get(v_fst_2185_, 0);
lean_inc(v_val_2188_);
lean_dec_ref_known(v_fst_2185_, 1);
return v_val_2188_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12_spec__14_spec__28_spec__35(lean_object* v_init_2189_, uint8_t v_indent_2190_, lean_object* v_as_2191_, size_t v_sz_2192_, size_t v_i_2193_, lean_object* v_b_2194_){
_start:
{
uint8_t v___x_2195_; 
v___x_2195_ = lean_usize_dec_lt(v_i_2193_, v_sz_2192_);
if (v___x_2195_ == 0)
{
return v_b_2194_;
}
else
{
lean_object* v_snd_2196_; lean_object* v___x_2198_; uint8_t v_isShared_2199_; uint8_t v_isSharedCheck_2214_; 
v_snd_2196_ = lean_ctor_get(v_b_2194_, 1);
v_isSharedCheck_2214_ = !lean_is_exclusive(v_b_2194_);
if (v_isSharedCheck_2214_ == 0)
{
lean_object* v_unused_2215_; 
v_unused_2215_ = lean_ctor_get(v_b_2194_, 0);
lean_dec(v_unused_2215_);
v___x_2198_ = v_b_2194_;
v_isShared_2199_ = v_isSharedCheck_2214_;
goto v_resetjp_2197_;
}
else
{
lean_inc(v_snd_2196_);
lean_dec(v_b_2194_);
v___x_2198_ = lean_box(0);
v_isShared_2199_ = v_isSharedCheck_2214_;
goto v_resetjp_2197_;
}
v_resetjp_2197_:
{
lean_object* v_a_2200_; lean_object* v___x_2201_; 
v_a_2200_ = lean_array_uget_borrowed(v_as_2191_, v_i_2193_);
lean_inc(v_snd_2196_);
v___x_2201_ = lp_aesop_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12_spec__14_spec__28(v_init_2189_, v_indent_2190_, v_a_2200_, v_snd_2196_);
if (lean_obj_tag(v___x_2201_) == 0)
{
lean_object* v___x_2202_; lean_object* v___x_2204_; 
v___x_2202_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2202_, 0, v___x_2201_);
if (v_isShared_2199_ == 0)
{
lean_ctor_set(v___x_2198_, 0, v___x_2202_);
v___x_2204_ = v___x_2198_;
goto v_reusejp_2203_;
}
else
{
lean_object* v_reuseFailAlloc_2205_; 
v_reuseFailAlloc_2205_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2205_, 0, v___x_2202_);
lean_ctor_set(v_reuseFailAlloc_2205_, 1, v_snd_2196_);
v___x_2204_ = v_reuseFailAlloc_2205_;
goto v_reusejp_2203_;
}
v_reusejp_2203_:
{
return v___x_2204_;
}
}
else
{
lean_object* v_a_2206_; lean_object* v___x_2207_; lean_object* v___x_2209_; 
lean_dec(v_snd_2196_);
v_a_2206_ = lean_ctor_get(v___x_2201_, 0);
lean_inc(v_a_2206_);
lean_dec_ref_known(v___x_2201_, 1);
v___x_2207_ = lean_box(0);
if (v_isShared_2199_ == 0)
{
lean_ctor_set(v___x_2198_, 1, v_a_2206_);
lean_ctor_set(v___x_2198_, 0, v___x_2207_);
v___x_2209_ = v___x_2198_;
goto v_reusejp_2208_;
}
else
{
lean_object* v_reuseFailAlloc_2213_; 
v_reuseFailAlloc_2213_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2213_, 0, v___x_2207_);
lean_ctor_set(v_reuseFailAlloc_2213_, 1, v_a_2206_);
v___x_2209_ = v_reuseFailAlloc_2213_;
goto v_reusejp_2208_;
}
v_reusejp_2208_:
{
size_t v___x_2210_; size_t v___x_2211_; 
v___x_2210_ = ((size_t)1ULL);
v___x_2211_ = lean_usize_add(v_i_2193_, v___x_2210_);
v_i_2193_ = v___x_2211_;
v_b_2194_ = v___x_2209_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12_spec__14_spec__28_spec__35___boxed(lean_object* v_init_2216_, lean_object* v_indent_2217_, lean_object* v_as_2218_, lean_object* v_sz_2219_, lean_object* v_i_2220_, lean_object* v_b_2221_){
_start:
{
uint8_t v_indent_boxed_2222_; size_t v_sz_boxed_2223_; size_t v_i_boxed_2224_; lean_object* v_res_2225_; 
v_indent_boxed_2222_ = lean_unbox(v_indent_2217_);
v_sz_boxed_2223_ = lean_unbox_usize(v_sz_2219_);
lean_dec(v_sz_2219_);
v_i_boxed_2224_ = lean_unbox_usize(v_i_2220_);
lean_dec(v_i_2220_);
v_res_2225_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12_spec__14_spec__28_spec__35(v_init_2216_, v_indent_boxed_2222_, v_as_2218_, v_sz_boxed_2223_, v_i_boxed_2224_, v_b_2221_);
lean_dec_ref(v_as_2218_);
lean_dec_ref(v_init_2216_);
return v_res_2225_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12_spec__14_spec__28___boxed(lean_object* v_init_2226_, lean_object* v_indent_2227_, lean_object* v_n_2228_, lean_object* v_b_2229_){
_start:
{
uint8_t v_indent_boxed_2230_; lean_object* v_res_2231_; 
v_indent_boxed_2230_ = lean_unbox(v_indent_2227_);
v_res_2231_ = lp_aesop_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12_spec__14_spec__28(v_init_2226_, v_indent_boxed_2230_, v_n_2228_, v_b_2229_);
lean_dec_ref(v_n_2228_);
lean_dec_ref(v_init_2226_);
return v_res_2231_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12_spec__14(uint8_t v_indent_2232_, lean_object* v_t_2233_, lean_object* v_init_2234_){
_start:
{
lean_object* v_root_2235_; lean_object* v_tail_2236_; lean_object* v___x_2237_; 
v_root_2235_ = lean_ctor_get(v_t_2233_, 0);
v_tail_2236_ = lean_ctor_get(v_t_2233_, 1);
lean_inc_ref(v_init_2234_);
v___x_2237_ = lp_aesop_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12_spec__14_spec__28(v_init_2234_, v_indent_2232_, v_root_2235_, v_init_2234_);
lean_dec_ref(v_init_2234_);
if (lean_obj_tag(v___x_2237_) == 0)
{
lean_object* v_a_2238_; 
v_a_2238_ = lean_ctor_get(v___x_2237_, 0);
lean_inc(v_a_2238_);
lean_dec_ref_known(v___x_2237_, 1);
return v_a_2238_;
}
else
{
lean_object* v_a_2239_; lean_object* v___x_2240_; lean_object* v___x_2241_; size_t v_sz_2242_; size_t v___x_2243_; lean_object* v___x_2244_; lean_object* v_fst_2245_; 
v_a_2239_ = lean_ctor_get(v___x_2237_, 0);
lean_inc(v_a_2239_);
lean_dec_ref_known(v___x_2237_, 1);
v___x_2240_ = lean_box(0);
v___x_2241_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2241_, 0, v___x_2240_);
lean_ctor_set(v___x_2241_, 1, v_a_2239_);
v_sz_2242_ = lean_array_size(v_tail_2236_);
v___x_2243_ = ((size_t)0ULL);
v___x_2244_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12_spec__14_spec__29(v_indent_2232_, v_tail_2236_, v_sz_2242_, v___x_2243_, v___x_2241_);
v_fst_2245_ = lean_ctor_get(v___x_2244_, 0);
lean_inc(v_fst_2245_);
if (lean_obj_tag(v_fst_2245_) == 0)
{
lean_object* v_snd_2246_; 
v_snd_2246_ = lean_ctor_get(v___x_2244_, 1);
lean_inc(v_snd_2246_);
lean_dec_ref(v___x_2244_);
return v_snd_2246_;
}
else
{
lean_object* v_val_2247_; 
lean_dec_ref(v___x_2244_);
v_val_2247_ = lean_ctor_get(v_fst_2245_, 0);
lean_inc(v_val_2247_);
lean_dec_ref_known(v_fst_2245_, 1);
return v_val_2247_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12_spec__14___boxed(lean_object* v_indent_2248_, lean_object* v_t_2249_, lean_object* v_init_2250_){
_start:
{
uint8_t v_indent_boxed_2251_; lean_object* v_res_2252_; 
v_indent_boxed_2251_ = lean_unbox(v_indent_2248_);
v_res_2252_ = lp_aesop_Lean_PersistentArray_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12_spec__14(v_indent_boxed_2251_, v_t_2249_, v_init_2250_);
lean_dec_ref(v_t_2249_);
return v_res_2252_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12___closed__1(void){
_start:
{
lean_object* v___x_2254_; lean_object* v___x_2255_; 
v___x_2254_ = ((lean_object*)(lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12___closed__0));
v___x_2255_ = l_Lean_stringToMessageData(v___x_2254_);
return v___x_2255_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12(uint8_t v_indent_2256_, lean_object* v_m_2257_){
_start:
{
lean_object* v_rep_2258_; lean_object* v_entries_2259_; lean_object* v___x_2260_; lean_object* v___x_2261_; lean_object* v___x_2262_; lean_object* v___x_2263_; 
v_rep_2258_ = lean_ctor_get(v_m_2257_, 0);
v_entries_2259_ = ((lean_object*)(lp_aesop_Lean_addTrace___at___00__private_Aesop_Tree_Tracing_0__Aesop_Goal_traceMetadata_trc_spec__0___closed__0));
v___x_2260_ = lp_aesop_Lean_PersistentArray_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12_spec__14(v_indent_2256_, v_rep_2258_, v_entries_2259_);
v___x_2261_ = lean_array_to_list(v___x_2260_);
v___x_2262_ = lean_obj_once(&lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12___closed__1, &lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12___closed__1_once, _init_lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12___closed__1);
v___x_2263_ = l_Lean_MessageData_joinSep(v___x_2261_, v___x_2262_);
return v___x_2263_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12___boxed(lean_object* v_indent_2264_, lean_object* v_m_2265_){
_start:
{
uint8_t v_indent_boxed_2266_; lean_object* v_res_2267_; 
v_indent_boxed_2266_ = lean_unbox(v_indent_2264_);
v_res_2267_ = lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12(v_indent_boxed_2266_, v_m_2265_);
lean_dec_ref(v_m_2265_);
return v_res_2267_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7___lam__5(lean_object* v___f_2268_, uint8_t v_indent_2269_, lean_object* v_instMap_2270_){
_start:
{
lean_object* v___x_2271_; lean_object* v___x_2272_; 
v___x_2271_ = lp_aesop_Aesop_EMap_mapM___at___00Aesop_EMap_map_spec__0___redArg(v___f_2268_, v_instMap_2270_);
v___x_2272_ = lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12(v_indent_2269_, v___x_2271_);
lean_dec_ref(v___x_2271_);
return v___x_2272_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7___lam__5___boxed(lean_object* v___f_2273_, lean_object* v_indent_2274_, lean_object* v_instMap_2275_){
_start:
{
uint8_t v_indent_boxed_2276_; lean_object* v_res_2277_; 
v_indent_boxed_2276_ = lean_unbox(v_indent_2274_);
v_res_2277_ = lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7___lam__5(v___f_2273_, v_indent_boxed_2276_, v_instMap_2275_);
return v_res_2277_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_map___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__13___redArg___lam__0(lean_object* v_f_2278_, lean_object* v_x_2279_){
_start:
{
lean_object* v___x_2280_; 
v___x_2280_ = lean_apply_1(v_f_2278_, v_x_2279_);
return v___x_2280_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_BasicAux_0__Array_mapM_x27_go___at___00Array_mapM_x27___at___00Lean_PersistentHashMap_mapMAux___at___00Lean_PersistentHashMap_mapM___at___00Lean_PersistentHashMap_map___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__13_spec__16_spec__32_spec__42_spec__48___redArg(lean_object* v_f_2281_, lean_object* v_as_2282_, lean_object* v_i_2283_, lean_object* v_acc_2284_){
_start:
{
lean_object* v___x_2285_; uint8_t v___x_2286_; 
v___x_2285_ = lean_array_get_size(v_as_2282_);
v___x_2286_ = lean_nat_dec_eq(v_i_2283_, v___x_2285_);
if (v___x_2286_ == 0)
{
lean_object* v___x_2287_; lean_object* v___x_2288_; lean_object* v___x_2289_; lean_object* v___x_2290_; lean_object* v___x_2291_; 
v___x_2287_ = lean_array_fget_borrowed(v_as_2282_, v_i_2283_);
lean_inc(v_f_2281_);
lean_inc(v___x_2287_);
v___x_2288_ = lean_apply_1(v_f_2281_, v___x_2287_);
v___x_2289_ = lean_unsigned_to_nat(1u);
v___x_2290_ = lean_nat_add(v_i_2283_, v___x_2289_);
lean_dec(v_i_2283_);
v___x_2291_ = lean_array_push(v_acc_2284_, v___x_2288_);
v_i_2283_ = v___x_2290_;
v_acc_2284_ = v___x_2291_;
goto _start;
}
else
{
lean_dec(v_i_2283_);
lean_dec(v_f_2281_);
return v_acc_2284_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_BasicAux_0__Array_mapM_x27_go___at___00Array_mapM_x27___at___00Lean_PersistentHashMap_mapMAux___at___00Lean_PersistentHashMap_mapM___at___00Lean_PersistentHashMap_map___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__13_spec__16_spec__32_spec__42_spec__48___redArg___boxed(lean_object* v_f_2293_, lean_object* v_as_2294_, lean_object* v_i_2295_, lean_object* v_acc_2296_){
_start:
{
lean_object* v_res_2297_; 
v_res_2297_ = lp_aesop___private_Init_Data_Array_BasicAux_0__Array_mapM_x27_go___at___00Array_mapM_x27___at___00Lean_PersistentHashMap_mapMAux___at___00Lean_PersistentHashMap_mapM___at___00Lean_PersistentHashMap_map___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__13_spec__16_spec__32_spec__42_spec__48___redArg(v_f_2293_, v_as_2294_, v_i_2295_, v_acc_2296_);
lean_dec_ref(v_as_2294_);
return v_res_2297_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_mapM_x27___at___00Lean_PersistentHashMap_mapMAux___at___00Lean_PersistentHashMap_mapM___at___00Lean_PersistentHashMap_map___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__13_spec__16_spec__32_spec__42___redArg(lean_object* v_f_2298_, lean_object* v_as_2299_){
_start:
{
lean_object* v___x_2300_; lean_object* v___x_2301_; lean_object* v___x_2302_; lean_object* v___x_2303_; 
v___x_2300_ = lean_unsigned_to_nat(0u);
v___x_2301_ = lean_array_get_size(v_as_2299_);
v___x_2302_ = lean_mk_empty_array_with_capacity(v___x_2301_);
v___x_2303_ = lp_aesop___private_Init_Data_Array_BasicAux_0__Array_mapM_x27_go___at___00Array_mapM_x27___at___00Lean_PersistentHashMap_mapMAux___at___00Lean_PersistentHashMap_mapM___at___00Lean_PersistentHashMap_map___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__13_spec__16_spec__32_spec__42_spec__48___redArg(v_f_2298_, v_as_2299_, v___x_2300_, v___x_2302_);
return v___x_2303_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_mapM_x27___at___00Lean_PersistentHashMap_mapMAux___at___00Lean_PersistentHashMap_mapM___at___00Lean_PersistentHashMap_map___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__13_spec__16_spec__32_spec__42___redArg___boxed(lean_object* v_f_2304_, lean_object* v_as_2305_){
_start:
{
lean_object* v_res_2306_; 
v_res_2306_ = lp_aesop_Array_mapM_x27___at___00Lean_PersistentHashMap_mapMAux___at___00Lean_PersistentHashMap_mapM___at___00Lean_PersistentHashMap_map___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__13_spec__16_spec__32_spec__42___redArg(v_f_2304_, v_as_2305_);
lean_dec_ref(v_as_2305_);
return v_res_2306_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_PersistentHashMap_mapMAux___at___00Lean_PersistentHashMap_mapM___at___00Lean_PersistentHashMap_map___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__13_spec__16_spec__32_spec__41___redArg(lean_object* v_f_2307_, size_t v_sz_2308_, size_t v_i_2309_, lean_object* v_bs_2310_){
_start:
{
uint8_t v___x_2311_; 
v___x_2311_ = lean_usize_dec_lt(v_i_2309_, v_sz_2308_);
if (v___x_2311_ == 0)
{
lean_dec(v_f_2307_);
return v_bs_2310_;
}
else
{
lean_object* v_v_2312_; lean_object* v___x_2313_; lean_object* v_bs_x27_2314_; lean_object* v___y_2316_; 
v_v_2312_ = lean_array_uget(v_bs_2310_, v_i_2309_);
v___x_2313_ = lean_unsigned_to_nat(0u);
v_bs_x27_2314_ = lean_array_uset(v_bs_2310_, v_i_2309_, v___x_2313_);
switch(lean_obj_tag(v_v_2312_))
{
case 0:
{
lean_object* v_key_2321_; lean_object* v_val_2322_; lean_object* v___x_2324_; uint8_t v_isShared_2325_; uint8_t v_isSharedCheck_2330_; 
v_key_2321_ = lean_ctor_get(v_v_2312_, 0);
v_val_2322_ = lean_ctor_get(v_v_2312_, 1);
v_isSharedCheck_2330_ = !lean_is_exclusive(v_v_2312_);
if (v_isSharedCheck_2330_ == 0)
{
v___x_2324_ = v_v_2312_;
v_isShared_2325_ = v_isSharedCheck_2330_;
goto v_resetjp_2323_;
}
else
{
lean_inc(v_val_2322_);
lean_inc(v_key_2321_);
lean_dec(v_v_2312_);
v___x_2324_ = lean_box(0);
v_isShared_2325_ = v_isSharedCheck_2330_;
goto v_resetjp_2323_;
}
v_resetjp_2323_:
{
lean_object* v___x_2326_; lean_object* v___x_2328_; 
lean_inc(v_f_2307_);
v___x_2326_ = lean_apply_1(v_f_2307_, v_val_2322_);
if (v_isShared_2325_ == 0)
{
lean_ctor_set(v___x_2324_, 1, v___x_2326_);
v___x_2328_ = v___x_2324_;
goto v_reusejp_2327_;
}
else
{
lean_object* v_reuseFailAlloc_2329_; 
v_reuseFailAlloc_2329_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2329_, 0, v_key_2321_);
lean_ctor_set(v_reuseFailAlloc_2329_, 1, v___x_2326_);
v___x_2328_ = v_reuseFailAlloc_2329_;
goto v_reusejp_2327_;
}
v_reusejp_2327_:
{
v___y_2316_ = v___x_2328_;
goto v___jp_2315_;
}
}
}
case 1:
{
lean_object* v_node_2331_; lean_object* v___x_2333_; uint8_t v_isShared_2334_; uint8_t v_isSharedCheck_2339_; 
v_node_2331_ = lean_ctor_get(v_v_2312_, 0);
v_isSharedCheck_2339_ = !lean_is_exclusive(v_v_2312_);
if (v_isSharedCheck_2339_ == 0)
{
v___x_2333_ = v_v_2312_;
v_isShared_2334_ = v_isSharedCheck_2339_;
goto v_resetjp_2332_;
}
else
{
lean_inc(v_node_2331_);
lean_dec(v_v_2312_);
v___x_2333_ = lean_box(0);
v_isShared_2334_ = v_isSharedCheck_2339_;
goto v_resetjp_2332_;
}
v_resetjp_2332_:
{
lean_object* v___x_2335_; lean_object* v___x_2337_; 
lean_inc(v_f_2307_);
v___x_2335_ = lp_aesop_Lean_PersistentHashMap_mapMAux___at___00Lean_PersistentHashMap_mapM___at___00Lean_PersistentHashMap_map___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__13_spec__16_spec__32___redArg(v_f_2307_, v_node_2331_);
if (v_isShared_2334_ == 0)
{
lean_ctor_set(v___x_2333_, 0, v___x_2335_);
v___x_2337_ = v___x_2333_;
goto v_reusejp_2336_;
}
else
{
lean_object* v_reuseFailAlloc_2338_; 
v_reuseFailAlloc_2338_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2338_, 0, v___x_2335_);
v___x_2337_ = v_reuseFailAlloc_2338_;
goto v_reusejp_2336_;
}
v_reusejp_2336_:
{
v___y_2316_ = v___x_2337_;
goto v___jp_2315_;
}
}
}
default: 
{
lean_object* v___x_2340_; 
v___x_2340_ = lean_box(2);
v___y_2316_ = v___x_2340_;
goto v___jp_2315_;
}
}
v___jp_2315_:
{
size_t v___x_2317_; size_t v___x_2318_; lean_object* v___x_2319_; 
v___x_2317_ = ((size_t)1ULL);
v___x_2318_ = lean_usize_add(v_i_2309_, v___x_2317_);
v___x_2319_ = lean_array_uset(v_bs_x27_2314_, v_i_2309_, v___y_2316_);
v_i_2309_ = v___x_2318_;
v_bs_2310_ = v___x_2319_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_mapMAux___at___00Lean_PersistentHashMap_mapM___at___00Lean_PersistentHashMap_map___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__13_spec__16_spec__32___redArg(lean_object* v_f_2341_, lean_object* v_n_2342_){
_start:
{
if (lean_obj_tag(v_n_2342_) == 0)
{
lean_object* v_es_2343_; lean_object* v___x_2345_; uint8_t v_isShared_2346_; uint8_t v_isSharedCheck_2353_; 
v_es_2343_ = lean_ctor_get(v_n_2342_, 0);
v_isSharedCheck_2353_ = !lean_is_exclusive(v_n_2342_);
if (v_isSharedCheck_2353_ == 0)
{
v___x_2345_ = v_n_2342_;
v_isShared_2346_ = v_isSharedCheck_2353_;
goto v_resetjp_2344_;
}
else
{
lean_inc(v_es_2343_);
lean_dec(v_n_2342_);
v___x_2345_ = lean_box(0);
v_isShared_2346_ = v_isSharedCheck_2353_;
goto v_resetjp_2344_;
}
v_resetjp_2344_:
{
size_t v_sz_2347_; size_t v___x_2348_; lean_object* v___x_2349_; lean_object* v___x_2351_; 
v_sz_2347_ = lean_array_size(v_es_2343_);
v___x_2348_ = ((size_t)0ULL);
v___x_2349_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_PersistentHashMap_mapMAux___at___00Lean_PersistentHashMap_mapM___at___00Lean_PersistentHashMap_map___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__13_spec__16_spec__32_spec__41___redArg(v_f_2341_, v_sz_2347_, v___x_2348_, v_es_2343_);
if (v_isShared_2346_ == 0)
{
lean_ctor_set(v___x_2345_, 0, v___x_2349_);
v___x_2351_ = v___x_2345_;
goto v_reusejp_2350_;
}
else
{
lean_object* v_reuseFailAlloc_2352_; 
v_reuseFailAlloc_2352_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2352_, 0, v___x_2349_);
v___x_2351_ = v_reuseFailAlloc_2352_;
goto v_reusejp_2350_;
}
v_reusejp_2350_:
{
return v___x_2351_;
}
}
}
else
{
lean_object* v_ks_2354_; lean_object* v_vs_2355_; lean_object* v___x_2357_; uint8_t v_isShared_2358_; uint8_t v_isSharedCheck_2363_; 
v_ks_2354_ = lean_ctor_get(v_n_2342_, 0);
v_vs_2355_ = lean_ctor_get(v_n_2342_, 1);
v_isSharedCheck_2363_ = !lean_is_exclusive(v_n_2342_);
if (v_isSharedCheck_2363_ == 0)
{
v___x_2357_ = v_n_2342_;
v_isShared_2358_ = v_isSharedCheck_2363_;
goto v_resetjp_2356_;
}
else
{
lean_inc(v_vs_2355_);
lean_inc(v_ks_2354_);
lean_dec(v_n_2342_);
v___x_2357_ = lean_box(0);
v_isShared_2358_ = v_isSharedCheck_2363_;
goto v_resetjp_2356_;
}
v_resetjp_2356_:
{
lean_object* v_val_2359_; lean_object* v___x_2361_; 
v_val_2359_ = lp_aesop_Array_mapM_x27___at___00Lean_PersistentHashMap_mapMAux___at___00Lean_PersistentHashMap_mapM___at___00Lean_PersistentHashMap_map___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__13_spec__16_spec__32_spec__42___redArg(v_f_2341_, v_vs_2355_);
lean_dec_ref(v_vs_2355_);
if (v_isShared_2358_ == 0)
{
lean_ctor_set(v___x_2357_, 1, v_val_2359_);
v___x_2361_ = v___x_2357_;
goto v_reusejp_2360_;
}
else
{
lean_object* v_reuseFailAlloc_2362_; 
v_reuseFailAlloc_2362_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2362_, 0, v_ks_2354_);
lean_ctor_set(v_reuseFailAlloc_2362_, 1, v_val_2359_);
v___x_2361_ = v_reuseFailAlloc_2362_;
goto v_reusejp_2360_;
}
v_reusejp_2360_:
{
return v___x_2361_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_PersistentHashMap_mapMAux___at___00Lean_PersistentHashMap_mapM___at___00Lean_PersistentHashMap_map___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__13_spec__16_spec__32_spec__41___redArg___boxed(lean_object* v_f_2364_, lean_object* v_sz_2365_, lean_object* v_i_2366_, lean_object* v_bs_2367_){
_start:
{
size_t v_sz_boxed_2368_; size_t v_i_boxed_2369_; lean_object* v_res_2370_; 
v_sz_boxed_2368_ = lean_unbox_usize(v_sz_2365_);
lean_dec(v_sz_2365_);
v_i_boxed_2369_ = lean_unbox_usize(v_i_2366_);
lean_dec(v_i_2366_);
v_res_2370_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_PersistentHashMap_mapMAux___at___00Lean_PersistentHashMap_mapM___at___00Lean_PersistentHashMap_map___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__13_spec__16_spec__32_spec__41___redArg(v_f_2364_, v_sz_boxed_2368_, v_i_boxed_2369_, v_bs_2367_);
return v_res_2370_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_map___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__13___redArg(lean_object* v_pm_2371_, lean_object* v_f_2372_){
_start:
{
lean_object* v___f_2373_; lean_object* v___x_2374_; 
v___f_2373_ = lean_alloc_closure((void*)(lp_aesop_Lean_PersistentHashMap_map___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__13___redArg___lam__0), 2, 1);
lean_closure_set(v___f_2373_, 0, v_f_2372_);
v___x_2374_ = lp_aesop_Lean_PersistentHashMap_mapMAux___at___00Lean_PersistentHashMap_mapM___at___00Lean_PersistentHashMap_map___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__13_spec__16_spec__32___redArg(v___f_2373_, v_pm_2371_);
return v___x_2374_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__14___lam__0(uint8_t v_indent_2375_, lean_object* v_x_2376_, lean_object* v_____s_2377_){
_start:
{
lean_object* v___y_2379_; 
if (v_indent_2375_ == 0)
{
lean_object* v_fst_2382_; lean_object* v_snd_2383_; lean_object* v___x_2385_; uint8_t v_isShared_2386_; uint8_t v_isSharedCheck_2395_; 
v_fst_2382_ = lean_ctor_get(v_x_2376_, 0);
v_snd_2383_ = lean_ctor_get(v_x_2376_, 1);
v_isSharedCheck_2395_ = !lean_is_exclusive(v_x_2376_);
if (v_isSharedCheck_2395_ == 0)
{
v___x_2385_ = v_x_2376_;
v_isShared_2386_ = v_isSharedCheck_2395_;
goto v_resetjp_2384_;
}
else
{
lean_inc(v_snd_2383_);
lean_inc(v_fst_2382_);
lean_dec(v_x_2376_);
v___x_2385_ = lean_box(0);
v_isShared_2386_ = v_isSharedCheck_2395_;
goto v_resetjp_2384_;
}
v_resetjp_2384_:
{
lean_object* v___x_2387_; lean_object* v___x_2388_; lean_object* v___x_2389_; lean_object* v___x_2390_; lean_object* v___x_2392_; 
v___x_2387_ = l_Nat_reprFast(v_fst_2382_);
v___x_2388_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_2388_, 0, v___x_2387_);
v___x_2389_ = l_Lean_MessageData_ofFormat(v___x_2388_);
v___x_2390_ = lean_obj_once(&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12_spec__14_spec__29_spec__38___closed__1, &lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12_spec__14_spec__29_spec__38___closed__1_once, _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12_spec__14_spec__29_spec__38___closed__1);
if (v_isShared_2386_ == 0)
{
lean_ctor_set_tag(v___x_2385_, 7);
lean_ctor_set(v___x_2385_, 1, v___x_2390_);
lean_ctor_set(v___x_2385_, 0, v___x_2389_);
v___x_2392_ = v___x_2385_;
goto v_reusejp_2391_;
}
else
{
lean_object* v_reuseFailAlloc_2394_; 
v_reuseFailAlloc_2394_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2394_, 0, v___x_2389_);
lean_ctor_set(v_reuseFailAlloc_2394_, 1, v___x_2390_);
v___x_2392_ = v_reuseFailAlloc_2394_;
goto v_reusejp_2391_;
}
v_reusejp_2391_:
{
lean_object* v___x_2393_; 
v___x_2393_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2393_, 0, v___x_2392_);
lean_ctor_set(v___x_2393_, 1, v_snd_2383_);
v___y_2379_ = v___x_2393_;
goto v___jp_2378_;
}
}
}
else
{
lean_object* v_fst_2396_; lean_object* v_snd_2397_; lean_object* v___x_2399_; uint8_t v_isShared_2400_; uint8_t v_isSharedCheck_2410_; 
v_fst_2396_ = lean_ctor_get(v_x_2376_, 0);
v_snd_2397_ = lean_ctor_get(v_x_2376_, 1);
v_isSharedCheck_2410_ = !lean_is_exclusive(v_x_2376_);
if (v_isSharedCheck_2410_ == 0)
{
v___x_2399_ = v_x_2376_;
v_isShared_2400_ = v_isSharedCheck_2410_;
goto v_resetjp_2398_;
}
else
{
lean_inc(v_snd_2397_);
lean_inc(v_fst_2396_);
lean_dec(v_x_2376_);
v___x_2399_ = lean_box(0);
v_isShared_2400_ = v_isSharedCheck_2410_;
goto v_resetjp_2398_;
}
v_resetjp_2398_:
{
lean_object* v___x_2401_; lean_object* v___x_2402_; lean_object* v___x_2403_; lean_object* v___x_2404_; lean_object* v___x_2406_; 
v___x_2401_ = l_Nat_reprFast(v_fst_2396_);
v___x_2402_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_2402_, 0, v___x_2401_);
v___x_2403_ = l_Lean_MessageData_ofFormat(v___x_2402_);
v___x_2404_ = lean_obj_once(&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12_spec__14_spec__29_spec__38___closed__3, &lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12_spec__14_spec__29_spec__38___closed__3_once, _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12_spec__14_spec__29_spec__38___closed__3);
if (v_isShared_2400_ == 0)
{
lean_ctor_set_tag(v___x_2399_, 7);
lean_ctor_set(v___x_2399_, 1, v___x_2404_);
lean_ctor_set(v___x_2399_, 0, v___x_2403_);
v___x_2406_ = v___x_2399_;
goto v_reusejp_2405_;
}
else
{
lean_object* v_reuseFailAlloc_2409_; 
v_reuseFailAlloc_2409_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2409_, 0, v___x_2403_);
lean_ctor_set(v_reuseFailAlloc_2409_, 1, v___x_2404_);
v___x_2406_ = v_reuseFailAlloc_2409_;
goto v_reusejp_2405_;
}
v_reusejp_2405_:
{
lean_object* v___x_2407_; lean_object* v___x_2408_; 
v___x_2407_ = l_Lean_indentD(v_snd_2397_);
v___x_2408_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2408_, 0, v___x_2406_);
lean_ctor_set(v___x_2408_, 1, v___x_2407_);
v___y_2379_ = v___x_2408_;
goto v___jp_2378_;
}
}
}
v___jp_2378_:
{
lean_object* v_entries_2380_; lean_object* v___x_2381_; 
v_entries_2380_ = lean_array_push(v_____s_2377_, v___y_2379_);
v___x_2381_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2381_, 0, v_entries_2380_);
return v___x_2381_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__14___lam__0___boxed(lean_object* v_indent_2411_, lean_object* v_x_2412_, lean_object* v_____s_2413_){
_start:
{
uint8_t v_indent_boxed_2414_; lean_object* v_res_2415_; 
v_indent_boxed_2414_ = lean_unbox(v_indent_2411_);
v_res_2415_ = lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__14___lam__0(v_indent_boxed_2414_, v_x_2412_, v_____s_2413_);
return v_res_2415_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__14_spec__18___redArg(lean_object* v_map_2416_, lean_object* v_init_2417_, lean_object* v_f_2418_){
_start:
{
lean_object* v___f_2419_; lean_object* v___x_2420_; lean_object* v_a_2421_; 
v___f_2419_ = lean_alloc_closure((void*)(lp_aesop_Lean_PersistentHashMap_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__15___redArg___lam__0), 4, 1);
lean_closure_set(v___f_2419_, 0, v_f_2418_);
lean_inc_ref(v_map_2416_);
v___x_2420_ = lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__15_spec__20_spec__38___redArg(v___f_2419_, v_map_2416_, v_init_2417_);
v_a_2421_ = lean_ctor_get(v___x_2420_, 0);
lean_inc(v_a_2421_);
lean_dec_ref(v___x_2420_);
return v_a_2421_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__14_spec__18___redArg___boxed(lean_object* v_map_2422_, lean_object* v_init_2423_, lean_object* v_f_2424_){
_start:
{
lean_object* v_res_2425_; 
v_res_2425_ = lp_aesop_Lean_PersistentHashMap_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__14_spec__18___redArg(v_map_2422_, v_init_2423_, v_f_2424_);
lean_dec_ref(v_map_2422_);
return v_res_2425_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__14(uint8_t v_indent_2426_, lean_object* v_m_2427_){
_start:
{
lean_object* v___x_2428_; lean_object* v___f_2429_; lean_object* v_entries_2430_; lean_object* v___x_2431_; lean_object* v___x_2432_; lean_object* v___x_2433_; lean_object* v___x_2434_; 
v___x_2428_ = lean_box(v_indent_2426_);
v___f_2429_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__14___lam__0___boxed), 3, 1);
lean_closure_set(v___f_2429_, 0, v___x_2428_);
v_entries_2430_ = ((lean_object*)(lp_aesop_Lean_addTrace___at___00__private_Aesop_Tree_Tracing_0__Aesop_Goal_traceMetadata_trc_spec__0___closed__0));
v___x_2431_ = lp_aesop_Lean_PersistentHashMap_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__14_spec__18___redArg(v_m_2427_, v_entries_2430_, v___f_2429_);
v___x_2432_ = lean_array_to_list(v___x_2431_);
v___x_2433_ = lean_obj_once(&lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12___closed__1, &lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12___closed__1_once, _init_lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12___closed__1);
v___x_2434_ = l_Lean_MessageData_joinSep(v___x_2432_, v___x_2433_);
return v___x_2434_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__14___boxed(lean_object* v_indent_2435_, lean_object* v_m_2436_){
_start:
{
uint8_t v_indent_boxed_2437_; lean_object* v_res_2438_; 
v_indent_boxed_2437_ = lean_unbox(v_indent_2435_);
v_res_2438_ = lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__14(v_indent_boxed_2437_, v_m_2436_);
lean_dec_ref(v_m_2436_);
return v_res_2438_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7___lam__3(uint8_t v_indent_2439_, lean_object* v___f_2440_, lean_object* v___f_2441_, lean_object* v_x_2442_, lean_object* v_____s_2443_){
_start:
{
lean_object* v___y_2445_; 
if (v_indent_2439_ == 0)
{
lean_object* v_fst_2448_; lean_object* v_snd_2449_; lean_object* v___x_2451_; uint8_t v_isShared_2452_; uint8_t v_isSharedCheck_2464_; 
lean_dec_ref(v___f_2441_);
v_fst_2448_ = lean_ctor_get(v_x_2442_, 0);
v_snd_2449_ = lean_ctor_get(v_x_2442_, 1);
v_isSharedCheck_2464_ = !lean_is_exclusive(v_x_2442_);
if (v_isSharedCheck_2464_ == 0)
{
v___x_2451_ = v_x_2442_;
v_isShared_2452_ = v_isSharedCheck_2464_;
goto v_resetjp_2450_;
}
else
{
lean_inc(v_snd_2449_);
lean_inc(v_fst_2448_);
lean_dec(v_x_2442_);
v___x_2451_ = lean_box(0);
v_isShared_2452_ = v_isSharedCheck_2464_;
goto v_resetjp_2450_;
}
v_resetjp_2450_:
{
lean_object* v___x_2453_; lean_object* v___x_2454_; lean_object* v___x_2455_; lean_object* v___x_2456_; lean_object* v___x_2458_; 
v___x_2453_ = l_Nat_reprFast(v_fst_2448_);
v___x_2454_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_2454_, 0, v___x_2453_);
v___x_2455_ = l_Lean_MessageData_ofFormat(v___x_2454_);
v___x_2456_ = lean_obj_once(&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12_spec__14_spec__29_spec__38___closed__1, &lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12_spec__14_spec__29_spec__38___closed__1_once, _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12_spec__14_spec__29_spec__38___closed__1);
if (v_isShared_2452_ == 0)
{
lean_ctor_set_tag(v___x_2451_, 7);
lean_ctor_set(v___x_2451_, 1, v___x_2456_);
lean_ctor_set(v___x_2451_, 0, v___x_2455_);
v___x_2458_ = v___x_2451_;
goto v_reusejp_2457_;
}
else
{
lean_object* v_reuseFailAlloc_2463_; 
v_reuseFailAlloc_2463_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2463_, 0, v___x_2455_);
lean_ctor_set(v_reuseFailAlloc_2463_, 1, v___x_2456_);
v___x_2458_ = v_reuseFailAlloc_2463_;
goto v_reusejp_2457_;
}
v_reusejp_2457_:
{
uint8_t v___x_2459_; lean_object* v___x_2460_; lean_object* v___x_2461_; lean_object* v___x_2462_; 
v___x_2459_ = 1;
v___x_2460_ = lp_aesop_Lean_PersistentHashMap_map___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__13___redArg(v_snd_2449_, v___f_2440_);
v___x_2461_ = lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__14(v___x_2459_, v___x_2460_);
lean_dec_ref(v___x_2460_);
v___x_2462_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2462_, 0, v___x_2458_);
lean_ctor_set(v___x_2462_, 1, v___x_2461_);
v___y_2445_ = v___x_2462_;
goto v___jp_2444_;
}
}
}
else
{
lean_object* v_fst_2465_; lean_object* v_snd_2466_; lean_object* v___x_2468_; uint8_t v_isShared_2469_; uint8_t v_isSharedCheck_2481_; 
lean_dec_ref(v___f_2440_);
v_fst_2465_ = lean_ctor_get(v_x_2442_, 0);
v_snd_2466_ = lean_ctor_get(v_x_2442_, 1);
v_isSharedCheck_2481_ = !lean_is_exclusive(v_x_2442_);
if (v_isSharedCheck_2481_ == 0)
{
v___x_2468_ = v_x_2442_;
v_isShared_2469_ = v_isSharedCheck_2481_;
goto v_resetjp_2467_;
}
else
{
lean_inc(v_snd_2466_);
lean_inc(v_fst_2465_);
lean_dec(v_x_2442_);
v___x_2468_ = lean_box(0);
v_isShared_2469_ = v_isSharedCheck_2481_;
goto v_resetjp_2467_;
}
v_resetjp_2467_:
{
lean_object* v___x_2470_; lean_object* v___x_2471_; lean_object* v___x_2472_; lean_object* v___x_2473_; lean_object* v___x_2475_; 
v___x_2470_ = l_Nat_reprFast(v_fst_2465_);
v___x_2471_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_2471_, 0, v___x_2470_);
v___x_2472_ = l_Lean_MessageData_ofFormat(v___x_2471_);
v___x_2473_ = lean_obj_once(&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12_spec__14_spec__29_spec__38___closed__3, &lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12_spec__14_spec__29_spec__38___closed__3_once, _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12_spec__14_spec__29_spec__38___closed__3);
if (v_isShared_2469_ == 0)
{
lean_ctor_set_tag(v___x_2468_, 7);
lean_ctor_set(v___x_2468_, 1, v___x_2473_);
lean_ctor_set(v___x_2468_, 0, v___x_2472_);
v___x_2475_ = v___x_2468_;
goto v_reusejp_2474_;
}
else
{
lean_object* v_reuseFailAlloc_2480_; 
v_reuseFailAlloc_2480_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2480_, 0, v___x_2472_);
lean_ctor_set(v_reuseFailAlloc_2480_, 1, v___x_2473_);
v___x_2475_ = v_reuseFailAlloc_2480_;
goto v_reusejp_2474_;
}
v_reusejp_2474_:
{
lean_object* v___x_2476_; lean_object* v___x_2477_; lean_object* v___x_2478_; lean_object* v___x_2479_; 
v___x_2476_ = lp_aesop_Lean_PersistentHashMap_map___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__13___redArg(v_snd_2466_, v___f_2441_);
v___x_2477_ = lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__14(v_indent_2439_, v___x_2476_);
lean_dec_ref(v___x_2476_);
v___x_2478_ = l_Lean_indentD(v___x_2477_);
v___x_2479_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2479_, 0, v___x_2475_);
lean_ctor_set(v___x_2479_, 1, v___x_2478_);
v___y_2445_ = v___x_2479_;
goto v___jp_2444_;
}
}
}
v___jp_2444_:
{
lean_object* v_entries_2446_; lean_object* v___x_2447_; 
v_entries_2446_ = lean_array_push(v_____s_2443_, v___y_2445_);
v___x_2447_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2447_, 0, v_entries_2446_);
return v___x_2447_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7___lam__3___boxed(lean_object* v_indent_2482_, lean_object* v___f_2483_, lean_object* v___f_2484_, lean_object* v_x_2485_, lean_object* v_____s_2486_){
_start:
{
uint8_t v_indent_boxed_2487_; lean_object* v_res_2488_; 
v_indent_boxed_2487_ = lean_unbox(v_indent_2482_);
v_res_2488_ = lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7___lam__3(v_indent_boxed_2487_, v___f_2483_, v___f_2484_, v_x_2485_, v_____s_2486_);
return v_res_2488_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7___lam__0(lean_object* v_d_2489_, lean_object* v_a_2490_, lean_object* v_x_2491_){
_start:
{
lean_object* v_fvarId_x3f_2492_; 
v_fvarId_x3f_2492_ = lean_ctor_get(v_a_2490_, 0);
if (lean_obj_tag(v_fvarId_x3f_2492_) == 0)
{
lean_object* v_subst_2493_; lean_object* v___x_2495_; uint8_t v_isShared_2496_; uint8_t v_isSharedCheck_2529_; 
v_subst_2493_ = lean_ctor_get(v_a_2490_, 1);
v_isSharedCheck_2529_ = !lean_is_exclusive(v_a_2490_);
if (v_isSharedCheck_2529_ == 0)
{
lean_object* v_unused_2530_; 
v_unused_2530_ = lean_ctor_get(v_a_2490_, 0);
lean_dec(v_unused_2530_);
v___x_2495_ = v_a_2490_;
v_isShared_2496_ = v_isSharedCheck_2529_;
goto v_resetjp_2494_;
}
else
{
lean_inc(v_subst_2493_);
lean_dec(v_a_2490_);
v___x_2495_ = lean_box(0);
v_isShared_2496_ = v_isSharedCheck_2529_;
goto v_resetjp_2494_;
}
v_resetjp_2494_:
{
lean_object* v_premises_2497_; lean_object* v_levels_2498_; lean_object* v___x_2500_; uint8_t v_isShared_2501_; uint8_t v_isSharedCheck_2528_; 
v_premises_2497_ = lean_ctor_get(v_subst_2493_, 0);
v_levels_2498_ = lean_ctor_get(v_subst_2493_, 1);
v_isSharedCheck_2528_ = !lean_is_exclusive(v_subst_2493_);
if (v_isSharedCheck_2528_ == 0)
{
v___x_2500_ = v_subst_2493_;
v_isShared_2501_ = v_isSharedCheck_2528_;
goto v_resetjp_2499_;
}
else
{
lean_inc(v_levels_2498_);
lean_inc(v_premises_2497_);
lean_dec(v_subst_2493_);
v___x_2500_ = lean_box(0);
v_isShared_2501_ = v_isSharedCheck_2528_;
goto v_resetjp_2499_;
}
v_resetjp_2499_:
{
lean_object* v___x_2502_; lean_object* v___x_2503_; lean_object* v___x_2504_; size_t v_sz_2505_; size_t v___x_2506_; lean_object* v___x_2507_; lean_object* v_ps_2508_; lean_object* v___x_2509_; lean_object* v___x_2510_; size_t v_sz_2511_; lean_object* v___x_2512_; lean_object* v_ls_2513_; lean_object* v___x_2514_; lean_object* v___x_2515_; lean_object* v___x_2516_; lean_object* v___x_2517_; lean_object* v___x_2519_; 
v___x_2502_ = lean_unsigned_to_nat(0u);
v___x_2503_ = lean_array_get_size(v_premises_2497_);
v___x_2504_ = lp_aesop_Array_filterMapM___at___00Aesop_Goal_traceMetadata_spec__0(v_premises_2497_, v___x_2502_, v___x_2503_);
lean_dec_ref(v_premises_2497_);
v_sz_2505_ = lean_array_size(v___x_2504_);
v___x_2506_ = ((size_t)0ULL);
v___x_2507_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Aesop_Goal_traceMetadata_spec__1___redArg(v_sz_2505_, v___x_2506_, v___x_2504_);
v_ps_2508_ = lean_array_to_list(v___x_2507_);
v___x_2509_ = lean_array_get_size(v_levels_2498_);
v___x_2510_ = lp_aesop_Array_filterMapM___at___00Aesop_Goal_traceMetadata_spec__2(v_levels_2498_, v___x_2502_, v___x_2509_);
lean_dec_ref(v_levels_2498_);
v_sz_2511_ = lean_array_size(v___x_2510_);
v___x_2512_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Aesop_Goal_traceMetadata_spec__3___redArg(v_sz_2511_, v___x_2506_, v___x_2510_);
v_ls_2513_ = lean_array_to_list(v___x_2512_);
v___x_2514_ = ((lean_object*)(lp_aesop_List_mapTR_loop___at___00Aesop_Goal_traceMetadata_spec__4___closed__0));
v___x_2515_ = lean_obj_once(&lp_aesop_List_mapTR_loop___at___00Aesop_Goal_traceMetadata_spec__4___closed__3, &lp_aesop_List_mapTR_loop___at___00Aesop_Goal_traceMetadata_spec__4___closed__3_once, _init_lp_aesop_List_mapTR_loop___at___00Aesop_Goal_traceMetadata_spec__4___closed__3);
v___x_2516_ = l_Lean_MessageData_joinSep(v_ps_2508_, v___x_2515_);
v___x_2517_ = lean_obj_once(&lp_aesop_List_mapTR_loop___at___00Aesop_Goal_traceMetadata_spec__4___closed__6, &lp_aesop_List_mapTR_loop___at___00Aesop_Goal_traceMetadata_spec__4___closed__6_once, _init_lp_aesop_List_mapTR_loop___at___00Aesop_Goal_traceMetadata_spec__4___closed__6);
if (v_isShared_2501_ == 0)
{
lean_ctor_set_tag(v___x_2500_, 7);
lean_ctor_set(v___x_2500_, 1, v___x_2517_);
lean_ctor_set(v___x_2500_, 0, v___x_2516_);
v___x_2519_ = v___x_2500_;
goto v_reusejp_2518_;
}
else
{
lean_object* v_reuseFailAlloc_2527_; 
v_reuseFailAlloc_2527_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2527_, 0, v___x_2516_);
lean_ctor_set(v_reuseFailAlloc_2527_, 1, v___x_2517_);
v___x_2519_ = v_reuseFailAlloc_2527_;
goto v_reusejp_2518_;
}
v_reusejp_2518_:
{
lean_object* v___x_2520_; lean_object* v___x_2522_; 
v___x_2520_ = l_Lean_MessageData_joinSep(v_ls_2513_, v___x_2515_);
if (v_isShared_2496_ == 0)
{
lean_ctor_set_tag(v___x_2495_, 7);
lean_ctor_set(v___x_2495_, 1, v___x_2520_);
lean_ctor_set(v___x_2495_, 0, v___x_2519_);
v___x_2522_ = v___x_2495_;
goto v_reusejp_2521_;
}
else
{
lean_object* v_reuseFailAlloc_2526_; 
v_reuseFailAlloc_2526_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2526_, 0, v___x_2519_);
lean_ctor_set(v_reuseFailAlloc_2526_, 1, v___x_2520_);
v___x_2522_ = v_reuseFailAlloc_2526_;
goto v_reusejp_2521_;
}
v_reusejp_2521_:
{
lean_object* v___x_2523_; lean_object* v___x_2524_; lean_object* v___x_2525_; 
v___x_2523_ = ((lean_object*)(lp_aesop_List_mapTR_loop___at___00Aesop_Goal_traceMetadata_spec__4___closed__7));
v___x_2524_ = l_Lean_MessageData_bracket(v___x_2514_, v___x_2522_, v___x_2523_);
v___x_2525_ = lean_array_push(v_d_2489_, v___x_2524_);
return v___x_2525_;
}
}
}
}
}
else
{
lean_object* v_val_2531_; lean_object* v___x_2532_; lean_object* v___x_2533_; lean_object* v___x_2534_; 
lean_inc_ref(v_fvarId_x3f_2492_);
lean_dec_ref(v_a_2490_);
v_val_2531_ = lean_ctor_get(v_fvarId_x3f_2492_, 0);
lean_inc(v_val_2531_);
lean_dec_ref_known(v_fvarId_x3f_2492_, 1);
v___x_2532_ = l_Lean_Expr_fvar___override(v_val_2531_);
v___x_2533_ = l_Lean_MessageData_ofExpr(v___x_2532_);
v___x_2534_ = lean_array_push(v_d_2489_, v___x_2533_);
return v___x_2534_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7___lam__2(lean_object* v___f_2535_, lean_object* v_instMap_2536_){
_start:
{
uint8_t v___x_2537_; lean_object* v___x_2538_; lean_object* v___x_2539_; 
v___x_2537_ = 0;
v___x_2538_ = lp_aesop_Aesop_EMap_mapM___at___00Aesop_EMap_map_spec__0___redArg(v___f_2535_, v_instMap_2536_);
v___x_2539_ = lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12(v___x_2537_, v___x_2538_);
lean_dec_ref(v___x_2538_);
return v___x_2539_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7(uint8_t v_indent_2545_, lean_object* v_m_2546_){
_start:
{
lean_object* v___f_2547_; lean_object* v___f_2548_; lean_object* v___x_2549_; lean_object* v___f_2550_; lean_object* v___x_2551_; lean_object* v___f_2552_; lean_object* v_entries_2553_; lean_object* v___x_2554_; lean_object* v___x_2555_; lean_object* v___x_2556_; lean_object* v___x_2557_; 
v___f_2547_ = ((lean_object*)(lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7___closed__1));
v___f_2548_ = ((lean_object*)(lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7___closed__2));
v___x_2549_ = lean_box(v_indent_2545_);
v___f_2550_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7___lam__5___boxed), 3, 2);
lean_closure_set(v___f_2550_, 0, v___f_2547_);
lean_closure_set(v___f_2550_, 1, v___x_2549_);
v___x_2551_ = lean_box(v_indent_2545_);
v___f_2552_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7___lam__3___boxed), 5, 3);
lean_closure_set(v___f_2552_, 0, v___x_2551_);
lean_closure_set(v___f_2552_, 1, v___f_2550_);
lean_closure_set(v___f_2552_, 2, v___f_2548_);
v_entries_2553_ = ((lean_object*)(lp_aesop_Lean_addTrace___at___00__private_Aesop_Tree_Tracing_0__Aesop_Goal_traceMetadata_trc_spec__0___closed__0));
v___x_2554_ = lp_aesop_Lean_PersistentHashMap_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__15___redArg(v_m_2546_, v_entries_2553_, v___f_2552_);
v___x_2555_ = lean_array_to_list(v___x_2554_);
v___x_2556_ = lean_obj_once(&lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12___closed__1, &lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12___closed__1_once, _init_lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12___closed__1);
v___x_2557_ = l_Lean_MessageData_joinSep(v___x_2555_, v___x_2556_);
return v___x_2557_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7___boxed(lean_object* v_indent_2558_, lean_object* v_m_2559_){
_start:
{
uint8_t v_indent_boxed_2560_; lean_object* v_res_2561_; 
v_indent_boxed_2560_ = lean_unbox(v_indent_2558_);
v_res_2561_ = lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7(v_indent_boxed_2560_, v_m_2559_);
lean_dec_ref(v_m_2559_);
return v_res_2561_;
}
}
LEAN_EXPORT lean_object* lp_aesop_List_mapIdx_go___at___00Aesop_Goal_traceMetadata_spec__11___lam__0(lean_object* v_d_2562_, lean_object* v_a_2563_, lean_object* v_x_2564_){
_start:
{
lean_object* v___x_2565_; 
v___x_2565_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2565_, 0, v_a_2563_);
lean_ctor_set(v___x_2565_, 1, v_d_2562_);
return v___x_2565_;
}
}
static lean_object* _init_lp_aesop_List_mapIdx_go___at___00Aesop_Goal_traceMetadata_spec__11___closed__2(void){
_start:
{
lean_object* v___x_2568_; lean_object* v___x_2569_; 
v___x_2568_ = ((lean_object*)(lp_aesop_List_mapIdx_go___at___00Aesop_Goal_traceMetadata_spec__11___closed__1));
v___x_2569_ = l_Lean_stringToMessageData(v___x_2568_);
return v___x_2569_;
}
}
static lean_object* _init_lp_aesop_List_mapIdx_go___at___00Aesop_Goal_traceMetadata_spec__11___closed__4(void){
_start:
{
lean_object* v___x_2571_; lean_object* v___x_2572_; 
v___x_2571_ = ((lean_object*)(lp_aesop_List_mapIdx_go___at___00Aesop_Goal_traceMetadata_spec__11___closed__3));
v___x_2572_ = l_Lean_stringToMessageData(v___x_2571_);
return v___x_2572_;
}
}
static lean_object* _init_lp_aesop_List_mapIdx_go___at___00Aesop_Goal_traceMetadata_spec__11___closed__6(void){
_start:
{
lean_object* v___x_2574_; lean_object* v___x_2575_; 
v___x_2574_ = ((lean_object*)(lp_aesop_List_mapIdx_go___at___00Aesop_Goal_traceMetadata_spec__11___closed__5));
v___x_2575_ = l_Lean_stringToMessageData(v___x_2574_);
return v___x_2575_;
}
}
static lean_object* _init_lp_aesop_List_mapIdx_go___at___00Aesop_Goal_traceMetadata_spec__11___closed__8(void){
_start:
{
lean_object* v___x_2577_; lean_object* v___x_2578_; 
v___x_2577_ = ((lean_object*)(lp_aesop_List_mapIdx_go___at___00Aesop_Goal_traceMetadata_spec__11___closed__7));
v___x_2578_ = l_Lean_stringToMessageData(v___x_2577_);
return v___x_2578_;
}
}
static lean_object* _init_lp_aesop_List_mapIdx_go___at___00Aesop_Goal_traceMetadata_spec__11___closed__10(void){
_start:
{
lean_object* v___x_2580_; lean_object* v___x_2581_; 
v___x_2580_ = ((lean_object*)(lp_aesop_List_mapIdx_go___at___00Aesop_Goal_traceMetadata_spec__11___closed__9));
v___x_2581_ = l_Lean_stringToMessageData(v___x_2580_);
return v___x_2581_;
}
}
LEAN_EXPORT lean_object* lp_aesop_List_mapIdx_go___at___00Aesop_Goal_traceMetadata_spec__11(uint8_t v_a_2582_, lean_object* v_a_2583_, lean_object* v_a_2584_){
_start:
{
if (lean_obj_tag(v_a_2583_) == 0)
{
lean_object* v___x_2585_; 
v___x_2585_ = lean_array_to_list(v_a_2584_);
return v___x_2585_;
}
else
{
lean_object* v_head_2586_; lean_object* v_tail_2587_; lean_object* v___x_2589_; uint8_t v_isShared_2590_; uint8_t v_isSharedCheck_2631_; 
v_head_2586_ = lean_ctor_get(v_a_2583_, 0);
v_tail_2587_ = lean_ctor_get(v_a_2583_, 1);
v_isSharedCheck_2631_ = !lean_is_exclusive(v_a_2583_);
if (v_isSharedCheck_2631_ == 0)
{
v___x_2589_ = v_a_2583_;
v_isShared_2590_ = v_isSharedCheck_2631_;
goto v_resetjp_2588_;
}
else
{
lean_inc(v_tail_2587_);
lean_inc(v_head_2586_);
lean_dec(v_a_2583_);
v___x_2589_ = lean_box(0);
v_isShared_2590_ = v_isSharedCheck_2631_;
goto v_resetjp_2588_;
}
v_resetjp_2588_:
{
lean_object* v_variableMap_2591_; lean_object* v_completeMatches_2592_; lean_object* v_slotQueues_2593_; lean_object* v___f_2594_; lean_object* v___x_2595_; lean_object* v___x_2596_; lean_object* v___x_2597_; lean_object* v___x_2598_; lean_object* v___x_2599_; lean_object* v___x_2601_; 
v_variableMap_2591_ = lean_ctor_get(v_head_2586_, 2);
lean_inc_ref(v_variableMap_2591_);
v_completeMatches_2592_ = lean_ctor_get(v_head_2586_, 3);
lean_inc_ref(v_completeMatches_2592_);
v_slotQueues_2593_ = lean_ctor_get(v_head_2586_, 4);
lean_inc_ref(v_slotQueues_2593_);
lean_dec(v_head_2586_);
v___f_2594_ = ((lean_object*)(lp_aesop_List_mapIdx_go___at___00Aesop_Goal_traceMetadata_spec__11___closed__0));
v___x_2595_ = lean_array_get_size(v_a_2584_);
v___x_2596_ = lean_obj_once(&lp_aesop_List_mapIdx_go___at___00Aesop_Goal_traceMetadata_spec__11___closed__2, &lp_aesop_List_mapIdx_go___at___00Aesop_Goal_traceMetadata_spec__11___closed__2_once, _init_lp_aesop_List_mapIdx_go___at___00Aesop_Goal_traceMetadata_spec__11___closed__2);
v___x_2597_ = l_Nat_reprFast(v___x_2595_);
v___x_2598_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_2598_, 0, v___x_2597_);
v___x_2599_ = l_Lean_MessageData_ofFormat(v___x_2598_);
if (v_isShared_2590_ == 0)
{
lean_ctor_set_tag(v___x_2589_, 7);
lean_ctor_set(v___x_2589_, 1, v___x_2599_);
lean_ctor_set(v___x_2589_, 0, v___x_2596_);
v___x_2601_ = v___x_2589_;
goto v_reusejp_2600_;
}
else
{
lean_object* v_reuseFailAlloc_2630_; 
v_reuseFailAlloc_2630_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2630_, 0, v___x_2596_);
lean_ctor_set(v_reuseFailAlloc_2630_, 1, v___x_2599_);
v___x_2601_ = v_reuseFailAlloc_2630_;
goto v_reusejp_2600_;
}
v_reusejp_2600_:
{
lean_object* v___x_2602_; lean_object* v___x_2603_; lean_object* v___x_2604_; lean_object* v___x_2605_; lean_object* v___x_2606_; lean_object* v___x_2607_; lean_object* v___x_2608_; lean_object* v___x_2609_; lean_object* v___x_2610_; lean_object* v___x_2611_; lean_object* v___x_2612_; lean_object* v___x_2613_; lean_object* v___x_2614_; lean_object* v___x_2615_; lean_object* v___x_2616_; lean_object* v___x_2617_; lean_object* v___x_2618_; lean_object* v___x_2619_; lean_object* v___x_2620_; lean_object* v___x_2621_; lean_object* v___x_2622_; lean_object* v___x_2623_; lean_object* v___x_2624_; lean_object* v___x_2625_; lean_object* v___x_2626_; lean_object* v___x_2627_; lean_object* v___x_2628_; 
v___x_2602_ = lean_obj_once(&lp_aesop_List_mapIdx_go___at___00Aesop_Goal_traceMetadata_spec__11___closed__4, &lp_aesop_List_mapIdx_go___at___00Aesop_Goal_traceMetadata_spec__11___closed__4_once, _init_lp_aesop_List_mapIdx_go___at___00Aesop_Goal_traceMetadata_spec__11___closed__4);
v___x_2603_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2603_, 0, v___x_2601_);
lean_ctor_set(v___x_2603_, 1, v___x_2602_);
v___x_2604_ = lean_obj_once(&lp_aesop_List_mapIdx_go___at___00Aesop_Goal_traceMetadata_spec__11___closed__6, &lp_aesop_List_mapIdx_go___at___00Aesop_Goal_traceMetadata_spec__11___closed__6_once, _init_lp_aesop_List_mapIdx_go___at___00Aesop_Goal_traceMetadata_spec__11___closed__6);
v___x_2605_ = lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7(v_a_2582_, v_variableMap_2591_);
lean_dec_ref(v_variableMap_2591_);
v___x_2606_ = l_Lean_indentD(v___x_2605_);
v___x_2607_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2607_, 0, v___x_2604_);
lean_ctor_set(v___x_2607_, 1, v___x_2606_);
v___x_2608_ = lean_obj_once(&lp_aesop_List_mapIdx_go___at___00Aesop_Goal_traceMetadata_spec__11___closed__8, &lp_aesop_List_mapIdx_go___at___00Aesop_Goal_traceMetadata_spec__11___closed__8_once, _init_lp_aesop_List_mapIdx_go___at___00Aesop_Goal_traceMetadata_spec__11___closed__8);
v___x_2609_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2609_, 0, v___x_2607_);
lean_ctor_set(v___x_2609_, 1, v___x_2608_);
v___x_2610_ = lean_unsigned_to_nat(0u);
v___x_2611_ = l_Array_zipIdx___redArg(v_slotQueues_2593_, v___x_2610_);
v___x_2612_ = lean_array_to_list(v___x_2611_);
v___x_2613_ = lean_box(0);
v___x_2614_ = lp_aesop_List_mapTR_loop___at___00Aesop_Goal_traceMetadata_spec__8(v___x_2612_, v___x_2613_);
v___x_2615_ = lean_obj_once(&lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12___closed__1, &lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12___closed__1_once, _init_lp_aesop___private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__12___closed__1);
v___x_2616_ = l_Lean_MessageData_joinSep(v___x_2614_, v___x_2615_);
v___x_2617_ = l_Lean_indentD(v___x_2616_);
v___x_2618_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2618_, 0, v___x_2609_);
lean_ctor_set(v___x_2618_, 1, v___x_2617_);
v___x_2619_ = lean_obj_once(&lp_aesop_List_mapIdx_go___at___00Aesop_Goal_traceMetadata_spec__11___closed__10, &lp_aesop_List_mapIdx_go___at___00Aesop_Goal_traceMetadata_spec__11___closed__10_once, _init_lp_aesop_List_mapIdx_go___at___00Aesop_Goal_traceMetadata_spec__11___closed__10);
v___x_2620_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2620_, 0, v___x_2618_);
lean_ctor_set(v___x_2620_, 1, v___x_2619_);
v___x_2621_ = lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_Goal_traceMetadata_spec__9_spec__18___redArg(v___f_2594_, v_completeMatches_2592_, v___x_2613_);
lean_dec_ref(v_completeMatches_2592_);
v___x_2622_ = lp_aesop_List_mapTR_loop___at___00Aesop_Goal_traceMetadata_spec__10(v___x_2621_, v___x_2613_);
v___x_2623_ = l_Lean_MessageData_joinSep(v___x_2622_, v___x_2615_);
v___x_2624_ = l_Lean_indentD(v___x_2623_);
v___x_2625_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2625_, 0, v___x_2620_);
lean_ctor_set(v___x_2625_, 1, v___x_2624_);
v___x_2626_ = l_Lean_indentD(v___x_2625_);
v___x_2627_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2627_, 0, v___x_2603_);
lean_ctor_set(v___x_2627_, 1, v___x_2626_);
v___x_2628_ = lean_array_push(v_a_2584_, v___x_2627_);
v_a_2583_ = v_tail_2587_;
v_a_2584_ = v___x_2628_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_List_mapIdx_go___at___00Aesop_Goal_traceMetadata_spec__11___boxed(lean_object* v_a_2632_, lean_object* v_a_2633_, lean_object* v_a_2634_){
_start:
{
uint8_t v_a_53591__boxed_2635_; lean_object* v_res_2636_; 
v_a_53591__boxed_2635_ = lean_unbox(v_a_2632_);
v_res_2636_ = lp_aesop_List_mapIdx_go___at___00Aesop_Goal_traceMetadata_spec__11(v_a_53591__boxed_2635_, v_a_2633_, v_a_2634_);
return v_res_2636_;
}
}
static lean_object* _init_lp_aesop_Aesop_Goal_traceMetadata___lam__1___closed__1(void){
_start:
{
lean_object* v___x_2639_; lean_object* v___x_2640_; 
v___x_2639_ = ((lean_object*)(lp_aesop_Aesop_Goal_traceMetadata___lam__1___closed__0));
v___x_2640_ = l_Lean_MessageData_ofFormat(v___x_2639_);
return v___x_2640_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_traceMetadata___lam__1(uint8_t v_a_2641_, lean_object* v_result_2642_, lean_object* v_r_2643_, lean_object* v_rs_2644_){
_start:
{
lean_object* v_name_2645_; uint8_t v_builder_2646_; uint8_t v_phase_2647_; uint8_t v_scope_2648_; lean_object* v___y_2650_; lean_object* v___y_2651_; lean_object* v___y_2652_; lean_object* v___y_2671_; lean_object* v___y_2672_; lean_object* v___y_2673_; lean_object* v___y_2679_; 
v_name_2645_ = lean_ctor_get(v_r_2643_, 0);
lean_inc(v_name_2645_);
v_builder_2646_ = lean_ctor_get_uint8(v_r_2643_, sizeof(void*)*1 + 8);
v_phase_2647_ = lean_ctor_get_uint8(v_r_2643_, sizeof(void*)*1 + 9);
v_scope_2648_ = lean_ctor_get_uint8(v_r_2643_, sizeof(void*)*1 + 10);
lean_dec_ref(v_r_2643_);
switch(v_phase_2647_)
{
case 0:
{
lean_object* v___x_2690_; 
v___x_2690_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__15));
v___y_2679_ = v___x_2690_;
goto v___jp_2678_;
}
case 1:
{
lean_object* v___x_2691_; 
v___x_2691_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__16));
v___y_2679_ = v___x_2691_;
goto v___jp_2678_;
}
default: 
{
lean_object* v___x_2692_; 
v___x_2692_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__17));
v___y_2679_ = v___x_2692_;
goto v___jp_2678_;
}
}
v___jp_2649_:
{
lean_object* v_clusterStates_2653_; lean_object* v___x_2654_; lean_object* v___x_2655_; lean_object* v___x_2656_; lean_object* v___x_2657_; lean_object* v___x_2658_; lean_object* v___x_2659_; lean_object* v___x_2660_; lean_object* v___x_2661_; lean_object* v___x_2662_; lean_object* v___x_2663_; lean_object* v___x_2664_; lean_object* v___x_2665_; lean_object* v___x_2666_; lean_object* v___x_2667_; lean_object* v___x_2668_; lean_object* v___x_2669_; 
v_clusterStates_2653_ = lean_ctor_get(v_rs_2644_, 1);
lean_inc_ref(v_clusterStates_2653_);
lean_dec_ref(v_rs_2644_);
v___x_2654_ = lean_string_append(v___y_2651_, v___y_2652_);
v___x_2655_ = lean_string_append(v___x_2654_, v___y_2650_);
v___x_2656_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_name_2645_, v_a_2641_);
v___x_2657_ = lean_string_append(v___x_2655_, v___x_2656_);
lean_dec_ref(v___x_2656_);
v___x_2658_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_2658_, 0, v___x_2657_);
v___x_2659_ = l_Lean_MessageData_ofFormat(v___x_2658_);
v___x_2660_ = lean_obj_once(&lp_aesop_List_mapIdx_go___at___00Aesop_Goal_traceMetadata_spec__11___closed__4, &lp_aesop_List_mapIdx_go___at___00Aesop_Goal_traceMetadata_spec__11___closed__4_once, _init_lp_aesop_List_mapIdx_go___at___00Aesop_Goal_traceMetadata_spec__11___closed__4);
v___x_2661_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2661_, 0, v___x_2659_);
lean_ctor_set(v___x_2661_, 1, v___x_2660_);
v___x_2662_ = lean_obj_once(&lp_aesop_Aesop_Goal_traceMetadata___lam__1___closed__1, &lp_aesop_Aesop_Goal_traceMetadata___lam__1___closed__1_once, _init_lp_aesop_Aesop_Goal_traceMetadata___lam__1___closed__1);
v___x_2663_ = lean_array_to_list(v_clusterStates_2653_);
v___x_2664_ = ((lean_object*)(lp_aesop_Lean_addTrace___at___00__private_Aesop_Tree_Tracing_0__Aesop_Goal_traceMetadata_trc_spec__0___closed__0));
v___x_2665_ = lp_aesop_List_mapIdx_go___at___00Aesop_Goal_traceMetadata_spec__11(v_a_2641_, v___x_2663_, v___x_2664_);
v___x_2666_ = l_Lean_MessageData_joinSep(v___x_2665_, v___x_2662_);
v___x_2667_ = l_Lean_indentD(v___x_2666_);
v___x_2668_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2668_, 0, v___x_2661_);
lean_ctor_set(v___x_2668_, 1, v___x_2667_);
v___x_2669_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2669_, 0, v___x_2668_);
lean_ctor_set(v___x_2669_, 1, v_result_2642_);
return v___x_2669_;
}
v___jp_2670_:
{
lean_object* v___x_2674_; lean_object* v___x_2675_; 
v___x_2674_ = lean_string_append(v___y_2671_, v___y_2673_);
v___x_2675_ = lean_string_append(v___x_2674_, v___y_2672_);
if (v_scope_2648_ == 0)
{
lean_object* v___x_2676_; 
v___x_2676_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__4));
v___y_2650_ = v___y_2672_;
v___y_2651_ = v___x_2675_;
v___y_2652_ = v___x_2676_;
goto v___jp_2649_;
}
else
{
lean_object* v___x_2677_; 
v___x_2677_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__5));
v___y_2650_ = v___y_2672_;
v___y_2651_ = v___x_2675_;
v___y_2652_ = v___x_2677_;
goto v___jp_2649_;
}
}
v___jp_2678_:
{
lean_object* v___x_2680_; lean_object* v___x_2681_; 
v___x_2680_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__6));
lean_inc_ref(v___y_2679_);
v___x_2681_ = lean_string_append(v___y_2679_, v___x_2680_);
switch(v_builder_2646_)
{
case 0:
{
lean_object* v___x_2682_; 
v___x_2682_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__7));
v___y_2671_ = v___x_2681_;
v___y_2672_ = v___x_2680_;
v___y_2673_ = v___x_2682_;
goto v___jp_2670_;
}
case 1:
{
lean_object* v___x_2683_; 
v___x_2683_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__8));
v___y_2671_ = v___x_2681_;
v___y_2672_ = v___x_2680_;
v___y_2673_ = v___x_2683_;
goto v___jp_2670_;
}
case 2:
{
lean_object* v___x_2684_; 
v___x_2684_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__9));
v___y_2671_ = v___x_2681_;
v___y_2672_ = v___x_2680_;
v___y_2673_ = v___x_2684_;
goto v___jp_2670_;
}
case 3:
{
lean_object* v___x_2685_; 
v___x_2685_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__10));
v___y_2671_ = v___x_2681_;
v___y_2672_ = v___x_2680_;
v___y_2673_ = v___x_2685_;
goto v___jp_2670_;
}
case 4:
{
lean_object* v___x_2686_; 
v___x_2686_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__11));
v___y_2671_ = v___x_2681_;
v___y_2672_ = v___x_2680_;
v___y_2673_ = v___x_2686_;
goto v___jp_2670_;
}
case 5:
{
lean_object* v___x_2687_; 
v___x_2687_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__12));
v___y_2671_ = v___x_2681_;
v___y_2672_ = v___x_2680_;
v___y_2673_ = v___x_2687_;
goto v___jp_2670_;
}
case 6:
{
lean_object* v___x_2688_; 
v___x_2688_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__13));
v___y_2671_ = v___x_2681_;
v___y_2672_ = v___x_2680_;
v___y_2673_ = v___x_2688_;
goto v___jp_2670_;
}
default: 
{
lean_object* v___x_2689_; 
v___x_2689_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__14));
v___y_2671_ = v___x_2681_;
v___y_2672_ = v___x_2680_;
v___y_2673_ = v___x_2689_;
goto v___jp_2670_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_traceMetadata___lam__1___boxed(lean_object* v_a_2693_, lean_object* v_result_2694_, lean_object* v_r_2695_, lean_object* v_rs_2696_){
_start:
{
uint8_t v_a_53732__boxed_2697_; lean_object* v_res_2698_; 
v_a_53732__boxed_2697_ = lean_unbox(v_a_2693_);
v_res_2698_ = lp_aesop_Aesop_Goal_traceMetadata___lam__1(v_a_53732__boxed_2697_, v_result_2694_, v_r_2695_, v_rs_2696_);
return v_res_2698_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_traceMetadata___lam__2(uint8_t v_a_2699_, lean_object* v_result_2700_, lean_object* v_r_2701_, lean_object* v_rs_2702_){
_start:
{
lean_object* v_name_2703_; uint8_t v_builder_2704_; uint8_t v_phase_2705_; uint8_t v_scope_2706_; lean_object* v___y_2708_; lean_object* v___y_2709_; lean_object* v___y_2710_; lean_object* v___y_2729_; lean_object* v___y_2730_; lean_object* v___y_2731_; lean_object* v___y_2737_; 
v_name_2703_ = lean_ctor_get(v_r_2701_, 0);
lean_inc(v_name_2703_);
v_builder_2704_ = lean_ctor_get_uint8(v_r_2701_, sizeof(void*)*1 + 8);
v_phase_2705_ = lean_ctor_get_uint8(v_r_2701_, sizeof(void*)*1 + 9);
v_scope_2706_ = lean_ctor_get_uint8(v_r_2701_, sizeof(void*)*1 + 10);
lean_dec_ref(v_r_2701_);
switch(v_phase_2705_)
{
case 0:
{
lean_object* v___x_2748_; 
v___x_2748_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__15));
v___y_2737_ = v___x_2748_;
goto v___jp_2736_;
}
case 1:
{
lean_object* v___x_2749_; 
v___x_2749_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__16));
v___y_2737_ = v___x_2749_;
goto v___jp_2736_;
}
default: 
{
lean_object* v___x_2750_; 
v___x_2750_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__17));
v___y_2737_ = v___x_2750_;
goto v___jp_2736_;
}
}
v___jp_2707_:
{
lean_object* v_clusterStates_2711_; lean_object* v___x_2712_; lean_object* v___x_2713_; lean_object* v___x_2714_; lean_object* v___x_2715_; lean_object* v___x_2716_; lean_object* v___x_2717_; lean_object* v___x_2718_; lean_object* v___x_2719_; lean_object* v___x_2720_; lean_object* v___x_2721_; lean_object* v___x_2722_; lean_object* v___x_2723_; lean_object* v___x_2724_; lean_object* v___x_2725_; lean_object* v___x_2726_; lean_object* v___x_2727_; 
v_clusterStates_2711_ = lean_ctor_get(v_rs_2702_, 1);
lean_inc_ref(v_clusterStates_2711_);
lean_dec_ref(v_rs_2702_);
v___x_2712_ = lean_string_append(v___y_2708_, v___y_2710_);
v___x_2713_ = lean_string_append(v___x_2712_, v___y_2709_);
v___x_2714_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_name_2703_, v_a_2699_);
v___x_2715_ = lean_string_append(v___x_2713_, v___x_2714_);
lean_dec_ref(v___x_2714_);
v___x_2716_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_2716_, 0, v___x_2715_);
v___x_2717_ = l_Lean_MessageData_ofFormat(v___x_2716_);
v___x_2718_ = lean_obj_once(&lp_aesop_List_mapIdx_go___at___00Aesop_Goal_traceMetadata_spec__11___closed__4, &lp_aesop_List_mapIdx_go___at___00Aesop_Goal_traceMetadata_spec__11___closed__4_once, _init_lp_aesop_List_mapIdx_go___at___00Aesop_Goal_traceMetadata_spec__11___closed__4);
v___x_2719_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2719_, 0, v___x_2717_);
lean_ctor_set(v___x_2719_, 1, v___x_2718_);
v___x_2720_ = lean_obj_once(&lp_aesop_Aesop_Goal_traceMetadata___lam__1___closed__1, &lp_aesop_Aesop_Goal_traceMetadata___lam__1___closed__1_once, _init_lp_aesop_Aesop_Goal_traceMetadata___lam__1___closed__1);
v___x_2721_ = lean_array_to_list(v_clusterStates_2711_);
v___x_2722_ = ((lean_object*)(lp_aesop_Lean_addTrace___at___00__private_Aesop_Tree_Tracing_0__Aesop_Goal_traceMetadata_trc_spec__0___closed__0));
v___x_2723_ = lp_aesop_List_mapIdx_go___at___00Aesop_Goal_traceMetadata_spec__11(v_a_2699_, v___x_2721_, v___x_2722_);
v___x_2724_ = l_Lean_MessageData_joinSep(v___x_2723_, v___x_2720_);
v___x_2725_ = l_Lean_indentD(v___x_2724_);
v___x_2726_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2726_, 0, v___x_2719_);
lean_ctor_set(v___x_2726_, 1, v___x_2725_);
v___x_2727_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2727_, 0, v___x_2726_);
lean_ctor_set(v___x_2727_, 1, v_result_2700_);
return v___x_2727_;
}
v___jp_2728_:
{
lean_object* v___x_2732_; lean_object* v___x_2733_; 
v___x_2732_ = lean_string_append(v___y_2729_, v___y_2731_);
v___x_2733_ = lean_string_append(v___x_2732_, v___y_2730_);
if (v_scope_2706_ == 0)
{
lean_object* v___x_2734_; 
v___x_2734_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__4));
v___y_2708_ = v___x_2733_;
v___y_2709_ = v___y_2730_;
v___y_2710_ = v___x_2734_;
goto v___jp_2707_;
}
else
{
lean_object* v___x_2735_; 
v___x_2735_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__5));
v___y_2708_ = v___x_2733_;
v___y_2709_ = v___y_2730_;
v___y_2710_ = v___x_2735_;
goto v___jp_2707_;
}
}
v___jp_2736_:
{
lean_object* v___x_2738_; lean_object* v___x_2739_; 
v___x_2738_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__6));
lean_inc_ref(v___y_2737_);
v___x_2739_ = lean_string_append(v___y_2737_, v___x_2738_);
switch(v_builder_2704_)
{
case 0:
{
lean_object* v___x_2740_; 
v___x_2740_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__7));
v___y_2729_ = v___x_2739_;
v___y_2730_ = v___x_2738_;
v___y_2731_ = v___x_2740_;
goto v___jp_2728_;
}
case 1:
{
lean_object* v___x_2741_; 
v___x_2741_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__8));
v___y_2729_ = v___x_2739_;
v___y_2730_ = v___x_2738_;
v___y_2731_ = v___x_2741_;
goto v___jp_2728_;
}
case 2:
{
lean_object* v___x_2742_; 
v___x_2742_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__9));
v___y_2729_ = v___x_2739_;
v___y_2730_ = v___x_2738_;
v___y_2731_ = v___x_2742_;
goto v___jp_2728_;
}
case 3:
{
lean_object* v___x_2743_; 
v___x_2743_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__10));
v___y_2729_ = v___x_2739_;
v___y_2730_ = v___x_2738_;
v___y_2731_ = v___x_2743_;
goto v___jp_2728_;
}
case 4:
{
lean_object* v___x_2744_; 
v___x_2744_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__11));
v___y_2729_ = v___x_2739_;
v___y_2730_ = v___x_2738_;
v___y_2731_ = v___x_2744_;
goto v___jp_2728_;
}
case 5:
{
lean_object* v___x_2745_; 
v___x_2745_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__12));
v___y_2729_ = v___x_2739_;
v___y_2730_ = v___x_2738_;
v___y_2731_ = v___x_2745_;
goto v___jp_2728_;
}
case 6:
{
lean_object* v___x_2746_; 
v___x_2746_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__13));
v___y_2729_ = v___x_2739_;
v___y_2730_ = v___x_2738_;
v___y_2731_ = v___x_2746_;
goto v___jp_2728_;
}
default: 
{
lean_object* v___x_2747_; 
v___x_2747_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__14));
v___y_2729_ = v___x_2739_;
v___y_2730_ = v___x_2738_;
v___y_2731_ = v___x_2747_;
goto v___jp_2728_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_traceMetadata___lam__2___boxed(lean_object* v_a_2751_, lean_object* v_result_2752_, lean_object* v_r_2753_, lean_object* v_rs_2754_){
_start:
{
uint8_t v_a_53849__boxed_2755_; lean_object* v_res_2756_; 
v_a_53849__boxed_2755_ = lean_unbox(v_a_2751_);
v_res_2756_ = lp_aesop_Aesop_Goal_traceMetadata___lam__2(v_a_53849__boxed_2755_, v_result_2752_, v_r_2753_, v_rs_2754_);
return v_res_2756_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldl___at___00Aesop_Goal_traceMetadata_spec__16___redArg___lam__0(lean_object* v_f_2757_, lean_object* v_x1_2758_, lean_object* v_x2_2759_, lean_object* v_x3_2760_){
_start:
{
lean_object* v___x_2761_; 
v___x_2761_ = lean_apply_3(v_f_2757_, v_x1_2758_, v_x2_2759_, v_x3_2760_);
return v___x_2761_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldl___at___00Aesop_Goal_traceMetadata_spec__16___redArg(lean_object* v_map_2762_, lean_object* v_f_2763_, lean_object* v_init_2764_){
_start:
{
lean_object* v___f_2765_; lean_object* v___x_2766_; 
v___f_2765_ = lean_alloc_closure((void*)(lp_aesop_Lean_PersistentHashMap_foldl___at___00Aesop_Goal_traceMetadata_spec__16___redArg___lam__0), 4, 1);
lean_closure_set(v___f_2765_, 0, v_f_2763_);
v___x_2766_ = lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_Goal_traceMetadata_spec__9_spec__18___redArg(v___f_2765_, v_map_2762_, v_init_2764_);
return v___x_2766_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldl___at___00Aesop_Goal_traceMetadata_spec__16___redArg___boxed(lean_object* v_map_2767_, lean_object* v_f_2768_, lean_object* v_init_2769_){
_start:
{
lean_object* v_res_2770_; 
v_res_2770_ = lp_aesop_Lean_PersistentHashMap_foldl___at___00Aesop_Goal_traceMetadata_spec__16___redArg(v_map_2767_, v_f_2768_, v_init_2769_);
lean_dec_ref(v_map_2767_);
return v_res_2770_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_traceMetadata___lam__3(lean_object* v___y_2771_, lean_object* v_forwardState_2772_, lean_object* v___f_2773_, lean_object* v___x_2774_, lean_object* v_traceOpt_2775_, lean_object* v_preNormGoal_2776_, lean_object* v_g_2777_, lean_object* v___f_2778_, lean_object* v___y_2779_, lean_object* v___y_2780_, lean_object* v___y_2781_, lean_object* v___y_2782_){
_start:
{
if (lean_obj_tag(v___y_2771_) == 0)
{
lean_object* v_ruleStates_2784_; lean_object* v___x_2785_; lean_object* v___x_2786_; lean_object* v___x_2787_; lean_object* v___x_2788_; lean_object* v___x_2789_; lean_object* v___x_2790_; 
lean_dec_ref(v___f_2778_);
v_ruleStates_2784_ = lean_ctor_get(v_forwardState_2772_, 0);
v___x_2785_ = lean_obj_once(&lp_aesop_Aesop_Goal_traceMetadata___lam__1___closed__1, &lp_aesop_Aesop_Goal_traceMetadata___lam__1___closed__1_once, _init_lp_aesop_Aesop_Goal_traceMetadata___lam__1___closed__1);
v___x_2786_ = lp_aesop_Lean_PersistentHashMap_foldl___at___00Aesop_Goal_traceMetadata_spec__16___redArg(v_ruleStates_2784_, v___f_2773_, v___x_2774_);
v___x_2787_ = l_Lean_MessageData_joinSep(v___x_2786_, v___x_2785_);
v___x_2788_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Goal_traceMetadata_trc___boxed), 7, 2);
lean_closure_set(v___x_2788_, 0, v_traceOpt_2775_);
lean_closure_set(v___x_2788_, 1, v___x_2787_);
v___x_2789_ = lean_alloc_closure((void*)(lp_aesop_Lean_MVarId_withContext___at___00__private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt_spec__1___boxed), 8, 3);
lean_closure_set(v___x_2789_, 0, lean_box(0));
lean_closure_set(v___x_2789_, 1, v_preNormGoal_2776_);
lean_closure_set(v___x_2789_, 2, v___x_2788_);
v___x_2790_ = lp_aesop_Aesop_Goal_runMetaMInParentState_x27___at___00__private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt_spec__2___redArg(v___x_2789_, v_g_2777_, v___y_2779_, v___y_2780_, v___y_2781_, v___y_2782_);
return v___x_2790_;
}
else
{
lean_object* v_val_2791_; lean_object* v_fst_2792_; lean_object* v_snd_2793_; lean_object* v_ruleStates_2794_; lean_object* v___x_2795_; lean_object* v___x_2796_; lean_object* v___x_2797_; lean_object* v___x_2798_; lean_object* v___x_2799_; lean_object* v___x_2800_; 
lean_dec(v_g_2777_);
lean_dec(v_preNormGoal_2776_);
lean_dec_ref(v___f_2773_);
v_val_2791_ = lean_ctor_get(v___y_2771_, 0);
lean_inc(v_val_2791_);
lean_dec_ref_known(v___y_2771_, 1);
v_fst_2792_ = lean_ctor_get(v_val_2791_, 0);
lean_inc(v_fst_2792_);
v_snd_2793_ = lean_ctor_get(v_val_2791_, 1);
lean_inc(v_snd_2793_);
lean_dec(v_val_2791_);
v_ruleStates_2794_ = lean_ctor_get(v_forwardState_2772_, 0);
v___x_2795_ = lean_obj_once(&lp_aesop_Aesop_Goal_traceMetadata___lam__1___closed__1, &lp_aesop_Aesop_Goal_traceMetadata___lam__1___closed__1_once, _init_lp_aesop_Aesop_Goal_traceMetadata___lam__1___closed__1);
v___x_2796_ = lp_aesop_Lean_PersistentHashMap_foldl___at___00Aesop_Goal_traceMetadata_spec__16___redArg(v_ruleStates_2794_, v___f_2778_, v___x_2774_);
v___x_2797_ = l_Lean_MessageData_joinSep(v___x_2796_, v___x_2795_);
v___x_2798_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Goal_traceMetadata_trc___boxed), 7, 2);
lean_closure_set(v___x_2798_, 0, v_traceOpt_2775_);
lean_closure_set(v___x_2798_, 1, v___x_2797_);
v___x_2799_ = lean_alloc_closure((void*)(lp_aesop_Lean_MVarId_withContext___at___00__private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt_spec__1___boxed), 8, 3);
lean_closure_set(v___x_2799_, 0, lean_box(0));
lean_closure_set(v___x_2799_, 1, v_fst_2792_);
lean_closure_set(v___x_2799_, 2, v___x_2798_);
v___x_2800_ = lp_batteries_Lean_Meta_SavedState_runMetaM_x27___redArg(v_snd_2793_, v___x_2799_, v___y_2779_, v___y_2780_, v___y_2781_, v___y_2782_);
return v___x_2800_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_traceMetadata___lam__3___boxed(lean_object* v___y_2801_, lean_object* v_forwardState_2802_, lean_object* v___f_2803_, lean_object* v___x_2804_, lean_object* v_traceOpt_2805_, lean_object* v_preNormGoal_2806_, lean_object* v_g_2807_, lean_object* v___f_2808_, lean_object* v___y_2809_, lean_object* v___y_2810_, lean_object* v___y_2811_, lean_object* v___y_2812_, lean_object* v___y_2813_){
_start:
{
lean_object* v_res_2814_; 
v_res_2814_ = lp_aesop_Aesop_Goal_traceMetadata___lam__3(v___y_2801_, v_forwardState_2802_, v___f_2803_, v___x_2804_, v_traceOpt_2805_, v_preNormGoal_2806_, v_g_2807_, v___f_2808_, v___y_2809_, v___y_2810_, v___y_2811_, v___y_2812_);
lean_dec(v___y_2812_);
lean_dec_ref(v___y_2811_);
lean_dec(v___y_2810_);
lean_dec_ref(v___y_2809_);
lean_dec_ref(v_forwardState_2802_);
return v_res_2814_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__18(uint8_t v___x_2815_, lean_object* v_traceOpt_2816_, lean_object* v_as_2817_, size_t v_sz_2818_, size_t v_i_2819_, lean_object* v_b_2820_, lean_object* v___y_2821_, lean_object* v___y_2822_, lean_object* v___y_2823_, lean_object* v___y_2824_){
_start:
{
uint8_t v___x_2826_; 
v___x_2826_ = lean_usize_dec_lt(v_i_2819_, v_sz_2818_);
if (v___x_2826_ == 0)
{
lean_object* v___x_2827_; 
lean_dec_ref(v_traceOpt_2816_);
v___x_2827_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2827_, 0, v_b_2820_);
return v___x_2827_;
}
else
{
lean_object* v_a_2828_; lean_object* v___x_2829_; uint8_t v_phase_2830_; lean_object* v___x_2831_; lean_object* v___x_2832_; double v___x_2833_; lean_object* v___x_2834_; lean_object* v___x_2835_; lean_object* v___x_2836_; lean_object* v___x_2837_; lean_object* v___x_2838_; lean_object* v___y_2840_; lean_object* v___y_2841_; lean_object* v___y_2842_; lean_object* v___y_2856_; lean_object* v___y_2857_; lean_object* v___y_2858_; lean_object* v___y_2865_; 
v_a_2828_ = lean_array_uget_borrowed(v_as_2817_, v_i_2819_);
v___x_2829_ = lp_aesop_Aesop_UnsafeQueueEntry_name(v_a_2828_);
v_phase_2830_ = lean_ctor_get_uint8(v___x_2829_, sizeof(void*)*1 + 9);
v___x_2831_ = lean_box(0);
v___x_2832_ = lean_obj_once(&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__1, &lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__1_once, _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__1);
v___x_2833_ = lp_aesop_Aesop_UnsafeQueueEntry_successProbability(v_a_2828_);
v___x_2834_ = lp_aesop_Aesop_Percent_toHumanString(v___x_2833_);
v___x_2835_ = l_Lean_stringToMessageData(v___x_2834_);
v___x_2836_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2836_, 0, v___x_2832_);
lean_ctor_set(v___x_2836_, 1, v___x_2835_);
v___x_2837_ = lean_obj_once(&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__3, &lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__3_once, _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__3);
v___x_2838_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2838_, 0, v___x_2836_);
lean_ctor_set(v___x_2838_, 1, v___x_2837_);
switch(v_phase_2830_)
{
case 0:
{
lean_object* v___x_2877_; 
v___x_2877_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__15));
v___y_2865_ = v___x_2877_;
goto v___jp_2864_;
}
case 1:
{
lean_object* v___x_2878_; 
v___x_2878_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__16));
v___y_2865_ = v___x_2878_;
goto v___jp_2864_;
}
default: 
{
lean_object* v___x_2879_; 
v___x_2879_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__17));
v___y_2865_ = v___x_2879_;
goto v___jp_2864_;
}
}
v___jp_2839_:
{
lean_object* v_name_2843_; lean_object* v___x_2844_; lean_object* v___x_2845_; lean_object* v___x_2846_; lean_object* v___x_2847_; lean_object* v___x_2848_; lean_object* v___x_2849_; lean_object* v___x_2850_; lean_object* v___x_2851_; 
v_name_2843_ = lean_ctor_get(v___x_2829_, 0);
lean_inc(v_name_2843_);
lean_dec_ref(v___x_2829_);
v___x_2844_ = lean_string_append(v___y_2841_, v___y_2842_);
v___x_2845_ = lean_string_append(v___x_2844_, v___y_2840_);
v___x_2846_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_name_2843_, v___x_2815_);
v___x_2847_ = lean_string_append(v___x_2845_, v___x_2846_);
lean_dec_ref(v___x_2846_);
v___x_2848_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_2848_, 0, v___x_2847_);
v___x_2849_ = l_Lean_MessageData_ofFormat(v___x_2848_);
v___x_2850_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2850_, 0, v___x_2838_);
lean_ctor_set(v___x_2850_, 1, v___x_2849_);
lean_inc_ref(v_traceOpt_2816_);
v___x_2851_ = lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Goal_traceMetadata_trc(v_traceOpt_2816_, v___x_2850_, v___y_2821_, v___y_2822_, v___y_2823_, v___y_2824_);
if (lean_obj_tag(v___x_2851_) == 0)
{
size_t v___x_2852_; size_t v___x_2853_; 
lean_dec_ref_known(v___x_2851_, 1);
v___x_2852_ = ((size_t)1ULL);
v___x_2853_ = lean_usize_add(v_i_2819_, v___x_2852_);
v_i_2819_ = v___x_2853_;
v_b_2820_ = v___x_2831_;
goto _start;
}
else
{
lean_dec_ref(v_traceOpt_2816_);
return v___x_2851_;
}
}
v___jp_2855_:
{
uint8_t v_scope_2859_; lean_object* v___x_2860_; lean_object* v___x_2861_; 
v_scope_2859_ = lean_ctor_get_uint8(v___x_2829_, sizeof(void*)*1 + 10);
v___x_2860_ = lean_string_append(v___y_2857_, v___y_2858_);
v___x_2861_ = lean_string_append(v___x_2860_, v___y_2856_);
if (v_scope_2859_ == 0)
{
lean_object* v___x_2862_; 
v___x_2862_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__4));
v___y_2840_ = v___y_2856_;
v___y_2841_ = v___x_2861_;
v___y_2842_ = v___x_2862_;
goto v___jp_2839_;
}
else
{
lean_object* v___x_2863_; 
v___x_2863_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__5));
v___y_2840_ = v___y_2856_;
v___y_2841_ = v___x_2861_;
v___y_2842_ = v___x_2863_;
goto v___jp_2839_;
}
}
v___jp_2864_:
{
uint8_t v_builder_2866_; lean_object* v___x_2867_; lean_object* v___x_2868_; 
v_builder_2866_ = lean_ctor_get_uint8(v___x_2829_, sizeof(void*)*1 + 8);
v___x_2867_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__6));
lean_inc_ref(v___y_2865_);
v___x_2868_ = lean_string_append(v___y_2865_, v___x_2867_);
switch(v_builder_2866_)
{
case 0:
{
lean_object* v___x_2869_; 
v___x_2869_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__7));
v___y_2856_ = v___x_2867_;
v___y_2857_ = v___x_2868_;
v___y_2858_ = v___x_2869_;
goto v___jp_2855_;
}
case 1:
{
lean_object* v___x_2870_; 
v___x_2870_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__8));
v___y_2856_ = v___x_2867_;
v___y_2857_ = v___x_2868_;
v___y_2858_ = v___x_2870_;
goto v___jp_2855_;
}
case 2:
{
lean_object* v___x_2871_; 
v___x_2871_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__9));
v___y_2856_ = v___x_2867_;
v___y_2857_ = v___x_2868_;
v___y_2858_ = v___x_2871_;
goto v___jp_2855_;
}
case 3:
{
lean_object* v___x_2872_; 
v___x_2872_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__10));
v___y_2856_ = v___x_2867_;
v___y_2857_ = v___x_2868_;
v___y_2858_ = v___x_2872_;
goto v___jp_2855_;
}
case 4:
{
lean_object* v___x_2873_; 
v___x_2873_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__11));
v___y_2856_ = v___x_2867_;
v___y_2857_ = v___x_2868_;
v___y_2858_ = v___x_2873_;
goto v___jp_2855_;
}
case 5:
{
lean_object* v___x_2874_; 
v___x_2874_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__12));
v___y_2856_ = v___x_2867_;
v___y_2857_ = v___x_2868_;
v___y_2858_ = v___x_2874_;
goto v___jp_2855_;
}
case 6:
{
lean_object* v___x_2875_; 
v___x_2875_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__13));
v___y_2856_ = v___x_2867_;
v___y_2857_ = v___x_2868_;
v___y_2858_ = v___x_2875_;
goto v___jp_2855_;
}
default: 
{
lean_object* v___x_2876_; 
v___x_2876_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__14));
v___y_2856_ = v___x_2867_;
v___y_2857_ = v___x_2868_;
v___y_2858_ = v___x_2876_;
goto v___jp_2855_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__18___boxed(lean_object* v___x_2880_, lean_object* v_traceOpt_2881_, lean_object* v_as_2882_, lean_object* v_sz_2883_, lean_object* v_i_2884_, lean_object* v_b_2885_, lean_object* v___y_2886_, lean_object* v___y_2887_, lean_object* v___y_2888_, lean_object* v___y_2889_, lean_object* v___y_2890_){
_start:
{
uint8_t v___x_54045__boxed_2891_; size_t v_sz_boxed_2892_; size_t v_i_boxed_2893_; lean_object* v_res_2894_; 
v___x_54045__boxed_2891_ = lean_unbox(v___x_2880_);
v_sz_boxed_2892_ = lean_unbox_usize(v_sz_2883_);
lean_dec(v_sz_2883_);
v_i_boxed_2893_ = lean_unbox_usize(v_i_2884_);
lean_dec(v_i_2884_);
v_res_2894_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__18(v___x_54045__boxed_2891_, v_traceOpt_2881_, v_as_2882_, v_sz_boxed_2892_, v_i_boxed_2893_, v_b_2885_, v___y_2886_, v___y_2887_, v___y_2888_, v___y_2889_);
lean_dec(v___y_2889_);
lean_dec_ref(v___y_2888_);
lean_dec(v___y_2887_);
lean_dec_ref(v___y_2886_);
lean_dec_ref(v_as_2882_);
return v_res_2894_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_traceMetadata___lam__4(uint8_t v_unsafeRulesSelected_2895_, lean_object* v_traceOpt_2896_, lean_object* v___x_2897_, size_t v_sz_2898_, size_t v___x_2899_, lean_object* v___x_2900_, lean_object* v___y_2901_, lean_object* v___y_2902_, lean_object* v___y_2903_, lean_object* v___y_2904_){
_start:
{
lean_object* v___x_2906_; 
v___x_2906_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__18(v_unsafeRulesSelected_2895_, v_traceOpt_2896_, v___x_2897_, v_sz_2898_, v___x_2899_, v___x_2900_, v___y_2901_, v___y_2902_, v___y_2903_, v___y_2904_);
if (lean_obj_tag(v___x_2906_) == 0)
{
lean_object* v___x_2908_; uint8_t v_isShared_2909_; uint8_t v_isSharedCheck_2913_; 
v_isSharedCheck_2913_ = !lean_is_exclusive(v___x_2906_);
if (v_isSharedCheck_2913_ == 0)
{
lean_object* v_unused_2914_; 
v_unused_2914_ = lean_ctor_get(v___x_2906_, 0);
lean_dec(v_unused_2914_);
v___x_2908_ = v___x_2906_;
v_isShared_2909_ = v_isSharedCheck_2913_;
goto v_resetjp_2907_;
}
else
{
lean_dec(v___x_2906_);
v___x_2908_ = lean_box(0);
v_isShared_2909_ = v_isSharedCheck_2913_;
goto v_resetjp_2907_;
}
v_resetjp_2907_:
{
lean_object* v___x_2911_; 
if (v_isShared_2909_ == 0)
{
lean_ctor_set(v___x_2908_, 0, v___x_2900_);
v___x_2911_ = v___x_2908_;
goto v_reusejp_2910_;
}
else
{
lean_object* v_reuseFailAlloc_2912_; 
v_reuseFailAlloc_2912_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2912_, 0, v___x_2900_);
v___x_2911_ = v_reuseFailAlloc_2912_;
goto v_reusejp_2910_;
}
v_reusejp_2910_:
{
return v___x_2911_;
}
}
}
else
{
return v___x_2906_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_traceMetadata___lam__4___boxed(lean_object* v_unsafeRulesSelected_2915_, lean_object* v_traceOpt_2916_, lean_object* v___x_2917_, lean_object* v_sz_2918_, lean_object* v___x_2919_, lean_object* v___x_2920_, lean_object* v___y_2921_, lean_object* v___y_2922_, lean_object* v___y_2923_, lean_object* v___y_2924_, lean_object* v___y_2925_){
_start:
{
uint8_t v_unsafeRulesSelected_boxed_2926_; size_t v_sz_boxed_2927_; size_t v___x_54166__boxed_2928_; lean_object* v_res_2929_; 
v_unsafeRulesSelected_boxed_2926_ = lean_unbox(v_unsafeRulesSelected_2915_);
v_sz_boxed_2927_ = lean_unbox_usize(v_sz_2918_);
lean_dec(v_sz_2918_);
v___x_54166__boxed_2928_ = lean_unbox_usize(v___x_2919_);
lean_dec(v___x_2919_);
v_res_2929_ = lp_aesop_Aesop_Goal_traceMetadata___lam__4(v_unsafeRulesSelected_boxed_2926_, v_traceOpt_2916_, v___x_2917_, v_sz_boxed_2927_, v___x_54166__boxed_2928_, v___x_2920_, v___y_2921_, v___y_2922_, v___y_2923_, v___y_2924_);
lean_dec(v___y_2924_);
lean_dec_ref(v___y_2923_);
lean_dec(v___y_2922_);
lean_dec_ref(v___y_2921_);
lean_dec_ref(v___x_2917_);
return v_res_2929_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_WFExtrinsicFix_0__WellFounded_opaqueFix_u2082___at___00Aesop_Goal_traceMetadata_spec__17___redArg(lean_object* v_a_2930_, lean_object* v_b_2931_){
_start:
{
lean_object* v_array_2932_; lean_object* v_start_2933_; lean_object* v_stop_2934_; lean_object* v___x_2936_; uint8_t v_isShared_2937_; uint8_t v_isSharedCheck_2947_; 
v_array_2932_ = lean_ctor_get(v_a_2930_, 0);
v_start_2933_ = lean_ctor_get(v_a_2930_, 1);
v_stop_2934_ = lean_ctor_get(v_a_2930_, 2);
v_isSharedCheck_2947_ = !lean_is_exclusive(v_a_2930_);
if (v_isSharedCheck_2947_ == 0)
{
v___x_2936_ = v_a_2930_;
v_isShared_2937_ = v_isSharedCheck_2947_;
goto v_resetjp_2935_;
}
else
{
lean_inc(v_stop_2934_);
lean_inc(v_start_2933_);
lean_inc(v_array_2932_);
lean_dec(v_a_2930_);
v___x_2936_ = lean_box(0);
v_isShared_2937_ = v_isSharedCheck_2947_;
goto v_resetjp_2935_;
}
v_resetjp_2935_:
{
uint8_t v___x_2938_; 
v___x_2938_ = lean_nat_dec_lt(v_start_2933_, v_stop_2934_);
if (v___x_2938_ == 0)
{
lean_del_object(v___x_2936_);
lean_dec(v_stop_2934_);
lean_dec(v_start_2933_);
lean_dec_ref(v_array_2932_);
return v_b_2931_;
}
else
{
lean_object* v___x_2939_; lean_object* v___x_2940_; lean_object* v___x_2942_; 
v___x_2939_ = lean_unsigned_to_nat(1u);
v___x_2940_ = lean_nat_add(v_start_2933_, v___x_2939_);
lean_inc_ref(v_array_2932_);
if (v_isShared_2937_ == 0)
{
lean_ctor_set(v___x_2936_, 1, v___x_2940_);
v___x_2942_ = v___x_2936_;
goto v_reusejp_2941_;
}
else
{
lean_object* v_reuseFailAlloc_2946_; 
v_reuseFailAlloc_2946_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_2946_, 0, v_array_2932_);
lean_ctor_set(v_reuseFailAlloc_2946_, 1, v___x_2940_);
lean_ctor_set(v_reuseFailAlloc_2946_, 2, v_stop_2934_);
v___x_2942_ = v_reuseFailAlloc_2946_;
goto v_reusejp_2941_;
}
v_reusejp_2941_:
{
lean_object* v___x_2943_; lean_object* v___x_2944_; 
v___x_2943_ = lean_array_fget(v_array_2932_, v_start_2933_);
lean_dec(v_start_2933_);
lean_dec_ref(v_array_2932_);
v___x_2944_ = lean_array_push(v_b_2931_, v___x_2943_);
v_a_2930_ = v___x_2942_;
v_b_2931_ = v___x_2944_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Goal_traceMetadata_spec__14___redArg(size_t v_sz_2948_, size_t v_i_2949_, lean_object* v_bs_2950_){
_start:
{
uint8_t v___x_2952_; 
v___x_2952_ = lean_usize_dec_lt(v_i_2949_, v_sz_2948_);
if (v___x_2952_ == 0)
{
lean_object* v___x_2953_; 
v___x_2953_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2953_, 0, v_bs_2950_);
return v___x_2953_;
}
else
{
lean_object* v_v_2954_; lean_object* v___x_2955_; lean_object* v___x_2956_; lean_object* v_elimRapp_2957_; lean_object* v___x_2958_; lean_object* v_id_2959_; lean_object* v___x_2960_; lean_object* v_bs_x27_2961_; size_t v___x_2962_; size_t v___x_2963_; lean_object* v___x_2964_; 
v_v_2954_ = lean_array_uget_borrowed(v_bs_2950_, v_i_2949_);
v___x_2955_ = lean_st_ref_get(v_v_2954_);
v___x_2956_ = lp_aesop_Aesop_treeImpl;
v_elimRapp_2957_ = lean_ctor_get(v___x_2956_, 3);
lean_inc_ref(v_elimRapp_2957_);
v___x_2958_ = lean_apply_1(v_elimRapp_2957_, v___x_2955_);
v_id_2959_ = lean_ctor_get(v___x_2958_, 0);
lean_inc(v_id_2959_);
lean_dec_ref(v___x_2958_);
v___x_2960_ = lean_unsigned_to_nat(0u);
v_bs_x27_2961_ = lean_array_uset(v_bs_2950_, v_i_2949_, v___x_2960_);
v___x_2962_ = ((size_t)1ULL);
v___x_2963_ = lean_usize_add(v_i_2949_, v___x_2962_);
v___x_2964_ = lean_array_uset(v_bs_x27_2961_, v_i_2949_, v_id_2959_);
v_i_2949_ = v___x_2963_;
v_bs_2950_ = v___x_2964_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Goal_traceMetadata_spec__14___redArg___boxed(lean_object* v_sz_2966_, lean_object* v_i_2967_, lean_object* v_bs_2968_, lean_object* v___y_2969_){
_start:
{
size_t v_sz_boxed_2970_; size_t v_i_boxed_2971_; lean_object* v_res_2972_; 
v_sz_boxed_2970_ = lean_unbox_usize(v_sz_2966_);
lean_dec(v_sz_2966_);
v_i_boxed_2971_ = lean_unbox_usize(v_i_2967_);
lean_dec(v_i_2967_);
v_res_2972_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Goal_traceMetadata_spec__14___redArg(v_sz_boxed_2970_, v_i_boxed_2971_, v_bs_2968_);
return v_res_2972_;
}
}
LEAN_EXPORT lean_object* lp_aesop_List_mapTR_loop___at___00Aesop_Goal_traceMetadata_spec__13(lean_object* v_a_2973_, lean_object* v_a_2974_){
_start:
{
if (lean_obj_tag(v_a_2973_) == 0)
{
lean_object* v___x_2975_; 
v___x_2975_ = l_List_reverse___redArg(v_a_2974_);
return v___x_2975_;
}
else
{
lean_object* v_head_2976_; lean_object* v_tail_2977_; lean_object* v___x_2979_; uint8_t v_isShared_2980_; uint8_t v_isSharedCheck_2986_; 
v_head_2976_ = lean_ctor_get(v_a_2973_, 0);
v_tail_2977_ = lean_ctor_get(v_a_2973_, 1);
v_isSharedCheck_2986_ = !lean_is_exclusive(v_a_2973_);
if (v_isSharedCheck_2986_ == 0)
{
v___x_2979_ = v_a_2973_;
v_isShared_2980_ = v_isSharedCheck_2986_;
goto v_resetjp_2978_;
}
else
{
lean_inc(v_tail_2977_);
lean_inc(v_head_2976_);
lean_dec(v_a_2973_);
v___x_2979_ = lean_box(0);
v_isShared_2980_ = v_isSharedCheck_2986_;
goto v_resetjp_2978_;
}
v_resetjp_2978_:
{
lean_object* v___x_2981_; lean_object* v___x_2983_; 
v___x_2981_ = l_Lean_MessageData_ofName(v_head_2976_);
if (v_isShared_2980_ == 0)
{
lean_ctor_set(v___x_2979_, 1, v_a_2974_);
lean_ctor_set(v___x_2979_, 0, v___x_2981_);
v___x_2983_ = v___x_2979_;
goto v_reusejp_2982_;
}
else
{
lean_object* v_reuseFailAlloc_2985_; 
v_reuseFailAlloc_2985_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2985_, 0, v___x_2981_);
lean_ctor_set(v_reuseFailAlloc_2985_, 1, v_a_2974_);
v___x_2983_ = v_reuseFailAlloc_2985_;
goto v_reusejp_2982_;
}
v_reusejp_2982_:
{
v_a_2973_ = v_tail_2977_;
v_a_2974_ = v___x_2983_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_Goal_traceMetadata_spec__5___redArg(lean_object* v_opt_2987_, lean_object* v___y_2988_){
_start:
{
lean_object* v_options_2990_; lean_object* v_option_2991_; uint8_t v___x_2992_; lean_object* v___x_2993_; lean_object* v___x_2994_; 
v_options_2990_ = lean_ctor_get(v___y_2988_, 2);
v_option_2991_ = lean_ctor_get(v_opt_2987_, 1);
v___x_2992_ = lp_aesop_Lean_Option_get___at___00Aesop_Goal_withHeadlineTraceNode_spec__1(v_options_2990_, v_option_2991_);
v___x_2993_ = lean_box(v___x_2992_);
v___x_2994_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2994_, 0, v___x_2993_);
return v___x_2994_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_Goal_traceMetadata_spec__5___redArg___boxed(lean_object* v_opt_2995_, lean_object* v___y_2996_, lean_object* v___y_2997_){
_start:
{
lean_object* v_res_2998_; 
v_res_2998_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_Goal_traceMetadata_spec__5___redArg(v_opt_2995_, v___y_2996_);
lean_dec_ref(v___y_2996_);
lean_dec_ref(v_opt_2995_);
return v_res_2998_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Goal_traceMetadata_spec__12(size_t v_sz_2999_, size_t v_i_3000_, lean_object* v_bs_3001_){
_start:
{
uint8_t v___x_3002_; 
v___x_3002_ = lean_usize_dec_lt(v_i_3000_, v_sz_2999_);
if (v___x_3002_ == 0)
{
return v_bs_3001_;
}
else
{
lean_object* v_v_3003_; lean_object* v___x_3004_; lean_object* v_bs_x27_3005_; size_t v___x_3006_; size_t v___x_3007_; lean_object* v___x_3008_; 
v_v_3003_ = lean_array_uget(v_bs_3001_, v_i_3000_);
v___x_3004_ = lean_unsigned_to_nat(0u);
v_bs_x27_3005_ = lean_array_uset(v_bs_3001_, v_i_3000_, v___x_3004_);
v___x_3006_ = ((size_t)1ULL);
v___x_3007_ = lean_usize_add(v_i_3000_, v___x_3006_);
v___x_3008_ = lean_array_uset(v_bs_x27_3005_, v_i_3000_, v_v_3003_);
v_i_3000_ = v___x_3007_;
v_bs_3001_ = v___x_3008_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Goal_traceMetadata_spec__12___boxed(lean_object* v_sz_3010_, lean_object* v_i_3011_, lean_object* v_bs_3012_){
_start:
{
size_t v_sz_boxed_3013_; size_t v_i_boxed_3014_; lean_object* v_res_3015_; 
v_sz_boxed_3013_ = lean_unbox_usize(v_sz_3010_);
lean_dec(v_sz_3010_);
v_i_boxed_3014_ = lean_unbox_usize(v_i_3011_);
lean_dec(v_i_3011_);
v_res_3015_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Goal_traceMetadata_spec__12(v_sz_boxed_3013_, v_i_boxed_3014_, v_bs_3012_);
return v_res_3015_;
}
}
LEAN_EXPORT lean_object* lp_aesop_List_mapTR_loop___at___00Aesop_Goal_traceMetadata_spec__15(lean_object* v_a_3016_, lean_object* v_a_3017_){
_start:
{
if (lean_obj_tag(v_a_3016_) == 0)
{
lean_object* v___x_3018_; 
v___x_3018_ = l_List_reverse___redArg(v_a_3017_);
return v___x_3018_;
}
else
{
lean_object* v_head_3019_; lean_object* v_tail_3020_; lean_object* v___x_3022_; uint8_t v_isShared_3023_; uint8_t v_isSharedCheck_3031_; 
v_head_3019_ = lean_ctor_get(v_a_3016_, 0);
v_tail_3020_ = lean_ctor_get(v_a_3016_, 1);
v_isSharedCheck_3031_ = !lean_is_exclusive(v_a_3016_);
if (v_isSharedCheck_3031_ == 0)
{
v___x_3022_ = v_a_3016_;
v_isShared_3023_ = v_isSharedCheck_3031_;
goto v_resetjp_3021_;
}
else
{
lean_inc(v_tail_3020_);
lean_inc(v_head_3019_);
lean_dec(v_a_3016_);
v___x_3022_ = lean_box(0);
v_isShared_3023_ = v_isSharedCheck_3031_;
goto v_resetjp_3021_;
}
v_resetjp_3021_:
{
lean_object* v___x_3024_; lean_object* v___x_3025_; lean_object* v___x_3026_; lean_object* v___x_3028_; 
v___x_3024_ = l_Nat_reprFast(v_head_3019_);
v___x_3025_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_3025_, 0, v___x_3024_);
v___x_3026_ = l_Lean_MessageData_ofFormat(v___x_3025_);
if (v_isShared_3023_ == 0)
{
lean_ctor_set(v___x_3022_, 1, v_a_3017_);
lean_ctor_set(v___x_3022_, 0, v___x_3026_);
v___x_3028_ = v___x_3022_;
goto v_reusejp_3027_;
}
else
{
lean_object* v_reuseFailAlloc_3030_; 
v_reuseFailAlloc_3030_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3030_, 0, v___x_3026_);
lean_ctor_set(v_reuseFailAlloc_3030_, 1, v_a_3017_);
v___x_3028_ = v_reuseFailAlloc_3030_;
goto v_reusejp_3027_;
}
v_reusejp_3027_:
{
v_a_3016_ = v_tail_3020_;
v_a_3017_ = v___x_3028_;
goto _start;
}
}
}
}
}
static lean_object* _init_lp_aesop_Aesop_Goal_traceMetadata___closed__1(void){
_start:
{
lean_object* v___x_3033_; lean_object* v___x_3034_; 
v___x_3033_ = ((lean_object*)(lp_aesop_Aesop_Goal_traceMetadata___closed__0));
v___x_3034_ = l_Lean_stringToMessageData(v___x_3033_);
return v___x_3034_;
}
}
static lean_object* _init_lp_aesop_Aesop_Goal_traceMetadata___closed__3(void){
_start:
{
lean_object* v___x_3036_; lean_object* v___x_3037_; 
v___x_3036_ = ((lean_object*)(lp_aesop_Aesop_Goal_traceMetadata___closed__2));
v___x_3037_ = l_Lean_stringToMessageData(v___x_3036_);
return v___x_3037_;
}
}
static lean_object* _init_lp_aesop_Aesop_Goal_traceMetadata___closed__5(void){
_start:
{
lean_object* v___x_3039_; lean_object* v___x_3040_; 
v___x_3039_ = ((lean_object*)(lp_aesop_Aesop_Goal_traceMetadata___closed__4));
v___x_3040_ = l_Lean_stringToMessageData(v___x_3039_);
return v___x_3040_;
}
}
static lean_object* _init_lp_aesop_Aesop_Goal_traceMetadata___closed__7(void){
_start:
{
lean_object* v___x_3042_; lean_object* v___x_3043_; 
v___x_3042_ = ((lean_object*)(lp_aesop_Aesop_Goal_traceMetadata___closed__6));
v___x_3043_ = l_Lean_stringToMessageData(v___x_3042_);
return v___x_3043_;
}
}
static lean_object* _init_lp_aesop_Aesop_Goal_traceMetadata___closed__9(void){
_start:
{
lean_object* v___x_3045_; lean_object* v___x_3046_; 
v___x_3045_ = ((lean_object*)(lp_aesop_Aesop_Goal_traceMetadata___closed__8));
v___x_3046_ = l_Lean_stringToMessageData(v___x_3045_);
return v___x_3046_;
}
}
static lean_object* _init_lp_aesop_Aesop_Goal_traceMetadata___closed__11(void){
_start:
{
lean_object* v___x_3048_; lean_object* v___x_3049_; 
v___x_3048_ = ((lean_object*)(lp_aesop_Aesop_Goal_traceMetadata___closed__10));
v___x_3049_ = l_Lean_stringToMessageData(v___x_3048_);
return v___x_3049_;
}
}
static lean_object* _init_lp_aesop_Aesop_Goal_traceMetadata___closed__13(void){
_start:
{
lean_object* v___x_3051_; lean_object* v___x_3052_; 
v___x_3051_ = ((lean_object*)(lp_aesop_Aesop_Goal_traceMetadata___closed__12));
v___x_3052_ = l_Lean_stringToMessageData(v___x_3051_);
return v___x_3052_;
}
}
static lean_object* _init_lp_aesop_Aesop_Goal_traceMetadata___closed__16(void){
_start:
{
lean_object* v___x_3056_; lean_object* v___x_3057_; 
v___x_3056_ = ((lean_object*)(lp_aesop_Aesop_Goal_traceMetadata___closed__15));
v___x_3057_ = l_Lean_stringToMessageData(v___x_3056_);
return v___x_3057_;
}
}
static lean_object* _init_lp_aesop_Aesop_Goal_traceMetadata___closed__18(void){
_start:
{
lean_object* v___x_3059_; lean_object* v___x_3060_; 
v___x_3059_ = ((lean_object*)(lp_aesop_Aesop_Goal_traceMetadata___closed__17));
v___x_3060_ = l_Lean_stringToMessageData(v___x_3059_);
return v___x_3060_;
}
}
static lean_object* _init_lp_aesop_Aesop_Goal_traceMetadata___closed__20(void){
_start:
{
lean_object* v___x_3062_; lean_object* v___x_3063_; 
v___x_3062_ = ((lean_object*)(lp_aesop_Aesop_Goal_traceMetadata___closed__19));
v___x_3063_ = l_Lean_stringToMessageData(v___x_3062_);
return v___x_3063_;
}
}
static lean_object* _init_lp_aesop_Aesop_Goal_traceMetadata___closed__22(void){
_start:
{
lean_object* v___x_3065_; lean_object* v___x_3066_; 
v___x_3065_ = ((lean_object*)(lp_aesop_Aesop_Goal_traceMetadata___closed__21));
v___x_3066_ = l_Lean_stringToMessageData(v___x_3065_);
return v___x_3066_;
}
}
static lean_object* _init_lp_aesop_Aesop_Goal_traceMetadata___closed__24(void){
_start:
{
lean_object* v___x_3068_; lean_object* v___x_3069_; 
v___x_3068_ = ((lean_object*)(lp_aesop_Aesop_Goal_traceMetadata___closed__23));
v___x_3069_ = l_Lean_stringToMessageData(v___x_3068_);
return v___x_3069_;
}
}
static lean_object* _init_lp_aesop_Aesop_Goal_traceMetadata___closed__26(void){
_start:
{
lean_object* v___x_3071_; lean_object* v___x_3072_; 
v___x_3071_ = ((lean_object*)(lp_aesop_Aesop_Goal_traceMetadata___closed__25));
v___x_3072_ = l_Lean_stringToMessageData(v___x_3071_);
return v___x_3072_;
}
}
static lean_object* _init_lp_aesop_Aesop_Goal_traceMetadata___closed__28(void){
_start:
{
lean_object* v___x_3074_; lean_object* v___x_3075_; 
v___x_3074_ = ((lean_object*)(lp_aesop_Aesop_Goal_traceMetadata___closed__27));
v___x_3075_ = l_Lean_stringToMessageData(v___x_3074_);
return v___x_3075_;
}
}
static lean_object* _init_lp_aesop_Aesop_Goal_traceMetadata___closed__31(void){
_start:
{
lean_object* v___x_3078_; lean_object* v___x_3079_; 
v___x_3078_ = ((lean_object*)(lp_aesop_Aesop_Goal_traceMetadata___closed__30));
v___x_3079_ = l_Lean_stringToMessageData(v___x_3078_);
return v___x_3079_;
}
}
static lean_object* _init_lp_aesop_Aesop_Goal_traceMetadata___closed__33(void){
_start:
{
lean_object* v___x_3081_; lean_object* v___x_3082_; 
v___x_3081_ = ((lean_object*)(lp_aesop_Aesop_Goal_traceMetadata___closed__32));
v___x_3082_ = l_Lean_stringToMessageData(v___x_3081_);
return v___x_3082_;
}
}
static lean_object* _init_lp_aesop_Aesop_Goal_traceMetadata___closed__35(void){
_start:
{
lean_object* v___x_3084_; lean_object* v___x_3085_; 
v___x_3084_ = ((lean_object*)(lp_aesop_Aesop_Goal_traceMetadata___closed__34));
v___x_3085_ = l_Lean_stringToMessageData(v___x_3084_);
return v___x_3085_;
}
}
static lean_object* _init_lp_aesop_Aesop_Goal_traceMetadata___closed__37(void){
_start:
{
lean_object* v___x_3087_; lean_object* v___x_3088_; 
v___x_3087_ = ((lean_object*)(lp_aesop_Aesop_Goal_traceMetadata___closed__36));
v___x_3088_ = l_Lean_stringToMessageData(v___x_3087_);
return v___x_3088_;
}
}
static lean_object* _init_lp_aesop_Aesop_Goal_traceMetadata___closed__39(void){
_start:
{
lean_object* v___x_3090_; lean_object* v___x_3091_; 
v___x_3090_ = ((lean_object*)(lp_aesop_Aesop_Goal_traceMetadata___closed__38));
v___x_3091_ = l_Lean_stringToMessageData(v___x_3090_);
return v___x_3091_;
}
}
static lean_object* _init_lp_aesop_Aesop_Goal_traceMetadata___closed__45(void){
_start:
{
lean_object* v___x_3097_; lean_object* v___x_3098_; 
v___x_3097_ = ((lean_object*)(lp_aesop_Aesop_Goal_traceMetadata___closed__44));
v___x_3098_ = l_Lean_stringToMessageData(v___x_3097_);
return v___x_3098_;
}
}
static lean_object* _init_lp_aesop_Aesop_Goal_traceMetadata___closed__48(void){
_start:
{
lean_object* v___x_3102_; lean_object* v___x_3103_; 
v___x_3102_ = ((lean_object*)(lp_aesop_Aesop_Goal_traceMetadata___closed__47));
v___x_3103_ = l_Lean_MessageData_ofFormat(v___x_3102_);
return v___x_3103_;
}
}
static lean_object* _init_lp_aesop_Aesop_Goal_traceMetadata___closed__51(void){
_start:
{
lean_object* v___x_3107_; lean_object* v___x_3108_; 
v___x_3107_ = ((lean_object*)(lp_aesop_Aesop_Goal_traceMetadata___closed__50));
v___x_3108_ = l_Lean_MessageData_ofFormat(v___x_3107_);
return v___x_3108_;
}
}
static lean_object* _init_lp_aesop_Aesop_Goal_traceMetadata___closed__54(void){
_start:
{
lean_object* v___x_3112_; lean_object* v___x_3113_; 
v___x_3112_ = ((lean_object*)(lp_aesop_Aesop_Goal_traceMetadata___closed__53));
v___x_3113_ = l_Lean_MessageData_ofFormat(v___x_3112_);
return v___x_3113_;
}
}
static lean_object* _init_lp_aesop_Aesop_Goal_traceMetadata___closed__56(void){
_start:
{
lean_object* v___x_3115_; lean_object* v___x_3116_; 
v___x_3115_ = ((lean_object*)(lp_aesop_Aesop_Goal_traceMetadata___closed__55));
v___x_3116_ = l_Lean_stringToMessageData(v___x_3115_);
return v___x_3116_;
}
}
static lean_object* _init_lp_aesop_Aesop_Goal_traceMetadata___closed__58(void){
_start:
{
lean_object* v___x_3118_; lean_object* v___x_3119_; 
v___x_3118_ = ((lean_object*)(lp_aesop_Aesop_Goal_traceMetadata___closed__57));
v___x_3119_ = l_Lean_stringToMessageData(v___x_3118_);
return v___x_3119_;
}
}
static lean_object* _init_lp_aesop_Aesop_Goal_traceMetadata___closed__60(void){
_start:
{
lean_object* v___x_3121_; lean_object* v___x_3122_; 
v___x_3121_ = ((lean_object*)(lp_aesop_Aesop_Goal_traceMetadata___closed__59));
v___x_3122_ = l_Lean_stringToMessageData(v___x_3121_);
return v___x_3122_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_traceMetadata(lean_object* v_g_3125_, lean_object* v_traceOpt_3126_, lean_object* v_a_3127_, lean_object* v_a_3128_, lean_object* v_a_3129_, lean_object* v_a_3130_){
_start:
{
lean_object* v___x_3132_; lean_object* v_a_3133_; lean_object* v___x_3135_; uint8_t v_isShared_3136_; uint8_t v_isSharedCheck_3446_; 
v___x_3132_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_Goal_traceMetadata_spec__5___redArg(v_traceOpt_3126_, v_a_3129_);
v_a_3133_ = lean_ctor_get(v___x_3132_, 0);
v_isSharedCheck_3446_ = !lean_is_exclusive(v___x_3132_);
if (v_isSharedCheck_3446_ == 0)
{
v___x_3135_ = v___x_3132_;
v_isShared_3136_ = v_isSharedCheck_3446_;
goto v_resetjp_3134_;
}
else
{
lean_inc(v_a_3133_);
lean_dec(v___x_3132_);
v___x_3135_ = lean_box(0);
v_isShared_3136_ = v_isSharedCheck_3446_;
goto v_resetjp_3134_;
}
v_resetjp_3134_:
{
lean_object* v___y_3138_; lean_object* v___y_3139_; lean_object* v___y_3140_; lean_object* v___y_3141_; uint8_t v___x_3158_; 
v___x_3158_ = lean_unbox(v_a_3133_);
if (v___x_3158_ == 0)
{
lean_object* v___x_3159_; lean_object* v___x_3161_; 
lean_dec(v_a_3133_);
lean_dec_ref(v_traceOpt_3126_);
lean_dec(v_g_3125_);
v___x_3159_ = lean_box(0);
if (v_isShared_3136_ == 0)
{
lean_ctor_set(v___x_3135_, 0, v___x_3159_);
v___x_3161_ = v___x_3135_;
goto v_reusejp_3160_;
}
else
{
lean_object* v_reuseFailAlloc_3162_; 
v_reuseFailAlloc_3162_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3162_, 0, v___x_3159_);
v___x_3161_ = v_reuseFailAlloc_3162_;
goto v_reusejp_3160_;
}
v_reusejp_3160_:
{
return v___x_3161_;
}
}
else
{
lean_object* v___x_3163_; lean_object* v_elimGoal_3164_; lean_object* v_elimRapp_3165_; lean_object* v___x_3166_; lean_object* v_id_3167_; lean_object* v_children_3168_; lean_object* v_origin_3169_; lean_object* v_depth_3170_; uint8_t v_state_3171_; uint8_t v_isIrrelevant_3172_; uint8_t v_isForcedUnprovable_3173_; lean_object* v_preNormGoal_3174_; lean_object* v_normalizationState_3175_; lean_object* v_mvars_3176_; lean_object* v_forwardState_3177_; lean_object* v_forwardRuleMatches_3178_; lean_object* v_addedInIteration_3179_; lean_object* v_lastExpandedInIteration_3180_; uint8_t v_unsafeRulesSelected_3181_; lean_object* v_unsafeQueue_3182_; lean_object* v___x_3183_; lean_object* v___x_3184_; lean_object* v___x_3185_; lean_object* v___x_3186_; lean_object* v___x_3187_; lean_object* v___x_3188_; 
lean_del_object(v___x_3135_);
v___x_3163_ = lp_aesop_Aesop_treeImpl;
v_elimGoal_3164_ = lean_ctor_get(v___x_3163_, 1);
v_elimRapp_3165_ = lean_ctor_get(v___x_3163_, 3);
lean_inc_ref(v_elimGoal_3164_);
lean_inc(v_g_3125_);
v___x_3166_ = lean_apply_1(v_elimGoal_3164_, v_g_3125_);
v_id_3167_ = lean_ctor_get(v___x_3166_, 0);
lean_inc(v_id_3167_);
v_children_3168_ = lean_ctor_get(v___x_3166_, 2);
lean_inc_ref(v_children_3168_);
v_origin_3169_ = lean_ctor_get(v___x_3166_, 3);
lean_inc(v_origin_3169_);
v_depth_3170_ = lean_ctor_get(v___x_3166_, 4);
lean_inc(v_depth_3170_);
v_state_3171_ = lean_ctor_get_uint8(v___x_3166_, sizeof(void*)*14 + 8);
v_isIrrelevant_3172_ = lean_ctor_get_uint8(v___x_3166_, sizeof(void*)*14 + 9);
v_isForcedUnprovable_3173_ = lean_ctor_get_uint8(v___x_3166_, sizeof(void*)*14 + 10);
v_preNormGoal_3174_ = lean_ctor_get(v___x_3166_, 5);
lean_inc(v_preNormGoal_3174_);
v_normalizationState_3175_ = lean_ctor_get(v___x_3166_, 6);
lean_inc(v_normalizationState_3175_);
v_mvars_3176_ = lean_ctor_get(v___x_3166_, 7);
lean_inc_ref(v_mvars_3176_);
v_forwardState_3177_ = lean_ctor_get(v___x_3166_, 8);
lean_inc_ref(v_forwardState_3177_);
v_forwardRuleMatches_3178_ = lean_ctor_get(v___x_3166_, 9);
lean_inc_ref(v_forwardRuleMatches_3178_);
v_addedInIteration_3179_ = lean_ctor_get(v___x_3166_, 10);
lean_inc(v_addedInIteration_3179_);
v_lastExpandedInIteration_3180_ = lean_ctor_get(v___x_3166_, 11);
lean_inc(v_lastExpandedInIteration_3180_);
v_unsafeRulesSelected_3181_ = lean_ctor_get_uint8(v___x_3166_, sizeof(void*)*14 + 11);
v_unsafeQueue_3182_ = lean_ctor_get(v___x_3166_, 12);
lean_inc_ref(v_unsafeQueue_3182_);
lean_dec_ref(v___x_3166_);
v___x_3183_ = lean_obj_once(&lp_aesop_Aesop_Goal_traceMetadata___closed__5, &lp_aesop_Aesop_Goal_traceMetadata___closed__5_once, _init_lp_aesop_Aesop_Goal_traceMetadata___closed__5);
v___x_3184_ = l_Nat_reprFast(v_id_3167_);
v___x_3185_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_3185_, 0, v___x_3184_);
v___x_3186_ = l_Lean_MessageData_ofFormat(v___x_3185_);
v___x_3187_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3187_, 0, v___x_3183_);
lean_ctor_set(v___x_3187_, 1, v___x_3186_);
lean_inc_ref(v_traceOpt_3126_);
v___x_3188_ = lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Goal_traceMetadata_trc(v_traceOpt_3126_, v___x_3187_, v_a_3127_, v_a_3128_, v_a_3129_, v_a_3130_);
if (lean_obj_tag(v___x_3188_) == 0)
{
lean_object* v___x_3190_; uint8_t v_isShared_3191_; uint8_t v_isSharedCheck_3444_; 
v_isSharedCheck_3444_ = !lean_is_exclusive(v___x_3188_);
if (v_isSharedCheck_3444_ == 0)
{
lean_object* v_unused_3445_; 
v_unused_3445_ = lean_ctor_get(v___x_3188_, 0);
lean_dec(v_unused_3445_);
v___x_3190_ = v___x_3188_;
v_isShared_3191_ = v_isSharedCheck_3444_;
goto v_resetjp_3189_;
}
else
{
lean_dec(v___x_3188_);
v___x_3190_ = lean_box(0);
v_isShared_3191_ = v_isSharedCheck_3444_;
goto v_resetjp_3189_;
}
v_resetjp_3189_:
{
lean_object* v___x_3192_; lean_object* v___x_3193_; lean_object* v___x_3194_; lean_object* v___x_3195_; lean_object* v___x_3196_; lean_object* v___x_3198_; 
v___x_3192_ = lean_obj_once(&lp_aesop_Aesop_Goal_traceMetadata___closed__7, &lp_aesop_Aesop_Goal_traceMetadata___closed__7_once, _init_lp_aesop_Aesop_Goal_traceMetadata___closed__7);
lean_inc_n(v_preNormGoal_3174_, 2);
v___x_3193_ = l_Lean_MessageData_ofName(v_preNormGoal_3174_);
v___x_3194_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3194_, 0, v___x_3192_);
lean_ctor_set(v___x_3194_, 1, v___x_3193_);
v___x_3195_ = lean_obj_once(&lp_aesop_Aesop_Goal_traceMetadata___closed__9, &lp_aesop_Aesop_Goal_traceMetadata___closed__9_once, _init_lp_aesop_Aesop_Goal_traceMetadata___closed__9);
v___x_3196_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3196_, 0, v___x_3194_);
lean_ctor_set(v___x_3196_, 1, v___x_3195_);
if (v_isShared_3191_ == 0)
{
lean_ctor_set_tag(v___x_3190_, 1);
lean_ctor_set(v___x_3190_, 0, v_preNormGoal_3174_);
v___x_3198_ = v___x_3190_;
goto v_reusejp_3197_;
}
else
{
lean_object* v_reuseFailAlloc_3443_; 
v_reuseFailAlloc_3443_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3443_, 0, v_preNormGoal_3174_);
v___x_3198_ = v_reuseFailAlloc_3443_;
goto v_reusejp_3197_;
}
v_reusejp_3197_:
{
lean_object* v___x_3199_; lean_object* v___x_3200_; lean_object* v___x_3201_; 
lean_inc_ref_n(v_traceOpt_3126_, 2);
v___x_3199_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Goal_traceMetadata_trc___boxed), 7, 2);
lean_closure_set(v___x_3199_, 0, v_traceOpt_3126_);
lean_closure_set(v___x_3199_, 1, v___x_3198_);
lean_inc(v_g_3125_);
v___x_3200_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Goal_runMetaMInParentState_x27___at___00__private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt_spec__2___boxed), 8, 3);
lean_closure_set(v___x_3200_, 0, lean_box(0));
lean_closure_set(v___x_3200_, 1, v___x_3199_);
lean_closure_set(v___x_3200_, 2, v_g_3125_);
v___x_3201_ = lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Goal_traceMetadata_trcNode(v_traceOpt_3126_, v___x_3196_, v___x_3200_, v_a_3127_, v_a_3128_, v_a_3129_, v_a_3130_);
if (lean_obj_tag(v___x_3201_) == 0)
{
lean_object* v___x_3203_; uint8_t v_isShared_3204_; uint8_t v_isSharedCheck_3441_; 
v_isSharedCheck_3441_ = !lean_is_exclusive(v___x_3201_);
if (v_isSharedCheck_3441_ == 0)
{
lean_object* v_unused_3442_; 
v_unused_3442_ = lean_ctor_get(v___x_3201_, 0);
lean_dec(v_unused_3442_);
v___x_3203_ = v___x_3201_;
v_isShared_3204_ = v_isSharedCheck_3441_;
goto v_resetjp_3202_;
}
else
{
lean_dec(v___x_3201_);
v___x_3203_ = lean_box(0);
v_isShared_3204_ = v_isSharedCheck_3441_;
goto v_resetjp_3202_;
}
v_resetjp_3202_:
{
lean_object* v___f_3205_; lean_object* v___f_3206_; size_t v___y_3208_; lean_object* v___y_3209_; lean_object* v___y_3210_; lean_object* v___y_3211_; lean_object* v___y_3212_; lean_object* v___y_3213_; lean_object* v___y_3214_; lean_object* v___y_3215_; size_t v___y_3236_; lean_object* v___y_3237_; lean_object* v___y_3238_; lean_object* v___y_3239_; lean_object* v___y_3240_; lean_object* v___y_3241_; lean_object* v___y_3242_; lean_object* v___y_3243_; size_t v___y_3261_; lean_object* v___y_3262_; lean_object* v___y_3263_; lean_object* v___y_3264_; lean_object* v___y_3265_; lean_object* v___y_3266_; lean_object* v___y_3267_; lean_object* v___y_3268_; size_t v___y_3318_; lean_object* v___y_3319_; lean_object* v___y_3320_; size_t v___y_3321_; lean_object* v___y_3322_; lean_object* v___y_3323_; lean_object* v___y_3324_; lean_object* v___y_3325_; lean_object* v___y_3326_; lean_object* v___y_3327_; size_t v___y_3376_; lean_object* v___y_3377_; lean_object* v___y_3378_; size_t v___y_3379_; lean_object* v___y_3380_; lean_object* v___y_3381_; lean_object* v___y_3382_; lean_object* v___y_3383_; lean_object* v_a_3384_; lean_object* v___y_3402_; lean_object* v___y_3403_; lean_object* v___y_3404_; lean_object* v___y_3405_; 
lean_inc_n(v_a_3133_, 2);
v___f_3205_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Goal_traceMetadata___lam__1___boxed), 4, 1);
lean_closure_set(v___f_3205_, 0, v_a_3133_);
v___f_3206_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Goal_traceMetadata___lam__2___boxed), 4, 1);
lean_closure_set(v___f_3206_, 0, v_a_3133_);
if (lean_obj_tag(v_normalizationState_3175_) == 1)
{
lean_object* v_postGoal_3429_; lean_object* v_postState_3430_; lean_object* v___x_3431_; lean_object* v___x_3432_; lean_object* v___x_3433_; lean_object* v___x_3434_; lean_object* v___x_3435_; lean_object* v___x_3436_; lean_object* v___x_3437_; lean_object* v___x_3438_; 
v_postGoal_3429_ = lean_ctor_get(v_normalizationState_3175_, 0);
v_postState_3430_ = lean_ctor_get(v_normalizationState_3175_, 1);
v___x_3431_ = lean_obj_once(&lp_aesop_Aesop_Goal_traceMetadata___closed__58, &lp_aesop_Aesop_Goal_traceMetadata___closed__58_once, _init_lp_aesop_Aesop_Goal_traceMetadata___closed__58);
lean_inc_n(v_postGoal_3429_, 2);
v___x_3432_ = l_Lean_MessageData_ofName(v_postGoal_3429_);
v___x_3433_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3433_, 0, v___x_3431_);
lean_ctor_set(v___x_3433_, 1, v___x_3432_);
v___x_3434_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3434_, 0, v___x_3433_);
lean_ctor_set(v___x_3434_, 1, v___x_3195_);
v___x_3435_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3435_, 0, v_postGoal_3429_);
lean_inc_ref_n(v_traceOpt_3126_, 2);
v___x_3436_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Goal_traceMetadata_trc___boxed), 7, 2);
lean_closure_set(v___x_3436_, 0, v_traceOpt_3126_);
lean_closure_set(v___x_3436_, 1, v___x_3435_);
v___x_3437_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Goal_traceMetadata_trcNode___boxed), 8, 3);
lean_closure_set(v___x_3437_, 0, v_traceOpt_3126_);
lean_closure_set(v___x_3437_, 1, v___x_3434_);
lean_closure_set(v___x_3437_, 2, v___x_3436_);
lean_inc_ref(v_postState_3430_);
v___x_3438_ = lp_batteries_Lean_Meta_SavedState_runMetaM_x27___redArg(v_postState_3430_, v___x_3437_, v_a_3127_, v_a_3128_, v_a_3129_, v_a_3130_);
if (lean_obj_tag(v___x_3438_) == 0)
{
lean_dec_ref_known(v___x_3438_, 1);
v___y_3402_ = v_a_3127_;
v___y_3403_ = v_a_3128_;
v___y_3404_ = v_a_3129_;
v___y_3405_ = v_a_3130_;
goto v___jp_3401_;
}
else
{
lean_dec_ref_known(v_normalizationState_3175_, 3);
lean_dec_ref(v___f_3206_);
lean_dec_ref(v___f_3205_);
lean_del_object(v___x_3203_);
lean_dec_ref(v_unsafeQueue_3182_);
lean_dec(v_lastExpandedInIteration_3180_);
lean_dec(v_addedInIteration_3179_);
lean_dec_ref(v_forwardRuleMatches_3178_);
lean_dec_ref(v_forwardState_3177_);
lean_dec_ref(v_mvars_3176_);
lean_dec(v_preNormGoal_3174_);
lean_dec(v_depth_3170_);
lean_dec(v_origin_3169_);
lean_dec_ref(v_children_3168_);
lean_dec(v_a_3133_);
lean_dec_ref(v_traceOpt_3126_);
lean_dec(v_g_3125_);
return v___x_3438_;
}
}
else
{
lean_object* v___x_3439_; lean_object* v___x_3440_; 
v___x_3439_ = lean_obj_once(&lp_aesop_Aesop_Goal_traceMetadata___closed__60, &lp_aesop_Aesop_Goal_traceMetadata___closed__60_once, _init_lp_aesop_Aesop_Goal_traceMetadata___closed__60);
lean_inc_ref(v_traceOpt_3126_);
v___x_3440_ = lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Goal_traceMetadata_trc(v_traceOpt_3126_, v___x_3439_, v_a_3127_, v_a_3128_, v_a_3129_, v_a_3130_);
if (lean_obj_tag(v___x_3440_) == 0)
{
lean_dec_ref_known(v___x_3440_, 1);
v___y_3402_ = v_a_3127_;
v___y_3403_ = v_a_3128_;
v___y_3404_ = v_a_3129_;
v___y_3405_ = v_a_3130_;
goto v___jp_3401_;
}
else
{
lean_dec_ref(v___f_3206_);
lean_dec_ref(v___f_3205_);
lean_del_object(v___x_3203_);
lean_dec_ref(v_unsafeQueue_3182_);
lean_dec(v_lastExpandedInIteration_3180_);
lean_dec(v_addedInIteration_3179_);
lean_dec_ref(v_forwardRuleMatches_3178_);
lean_dec_ref(v_forwardState_3177_);
lean_dec_ref(v_mvars_3176_);
lean_dec(v_normalizationState_3175_);
lean_dec(v_preNormGoal_3174_);
lean_dec(v_depth_3170_);
lean_dec(v_origin_3169_);
lean_dec_ref(v_children_3168_);
lean_dec(v_a_3133_);
lean_dec_ref(v_traceOpt_3126_);
lean_dec(v_g_3125_);
return v___x_3440_;
}
}
v___jp_3207_:
{
lean_object* v___y_3216_; lean_object* v___x_3217_; 
lean_inc(v_g_3125_);
lean_inc_ref_n(v_traceOpt_3126_, 2);
v___y_3216_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Goal_traceMetadata___lam__3___boxed), 13, 8);
lean_closure_set(v___y_3216_, 0, v___y_3215_);
lean_closure_set(v___y_3216_, 1, v_forwardState_3177_);
lean_closure_set(v___y_3216_, 2, v___f_3206_);
lean_closure_set(v___y_3216_, 3, v___y_3209_);
lean_closure_set(v___y_3216_, 4, v_traceOpt_3126_);
lean_closure_set(v___y_3216_, 5, v_preNormGoal_3174_);
lean_closure_set(v___y_3216_, 6, v_g_3125_);
lean_closure_set(v___y_3216_, 7, v___f_3205_);
lean_inc_ref(v___y_3211_);
v___x_3217_ = lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Goal_traceMetadata_trcNode(v_traceOpt_3126_, v___y_3211_, v___y_3216_, v___y_3213_, v___y_3210_, v___y_3212_, v___y_3214_);
if (lean_obj_tag(v___x_3217_) == 0)
{
lean_dec_ref_known(v___x_3217_, 1);
if (v_unsafeRulesSelected_3181_ == 0)
{
lean_object* v___x_3218_; lean_object* v___x_3219_; 
lean_dec_ref(v_unsafeQueue_3182_);
v___x_3218_ = lean_obj_once(&lp_aesop_Aesop_Goal_traceMetadata___closed__11, &lp_aesop_Aesop_Goal_traceMetadata___closed__11_once, _init_lp_aesop_Aesop_Goal_traceMetadata___closed__11);
lean_inc_ref(v_traceOpt_3126_);
v___x_3219_ = lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Goal_traceMetadata_trc(v_traceOpt_3126_, v___x_3218_, v___y_3213_, v___y_3210_, v___y_3212_, v___y_3214_);
if (lean_obj_tag(v___x_3219_) == 0)
{
lean_dec_ref_known(v___x_3219_, 1);
v___y_3138_ = v___y_3213_;
v___y_3139_ = v___y_3210_;
v___y_3140_ = v___y_3212_;
v___y_3141_ = v___y_3214_;
goto v___jp_3137_;
}
else
{
lean_dec(v_a_3133_);
lean_dec_ref(v_traceOpt_3126_);
lean_dec(v_g_3125_);
return v___x_3219_;
}
}
else
{
lean_object* v_start_3220_; lean_object* v_stop_3221_; uint8_t v___x_3222_; 
v_start_3220_ = lean_ctor_get(v_unsafeQueue_3182_, 1);
v_stop_3221_ = lean_ctor_get(v_unsafeQueue_3182_, 2);
v___x_3222_ = lean_nat_dec_eq(v_start_3220_, v_stop_3221_);
if (v___x_3222_ == 0)
{
lean_object* v___x_3223_; lean_object* v___x_3224_; lean_object* v___x_3225_; lean_object* v___x_3226_; size_t v_sz_3227_; lean_object* v___x_3228_; lean_object* v___x_3229_; lean_object* v___x_3230_; lean_object* v___f_3231_; lean_object* v___x_3232_; 
v___x_3223_ = lean_obj_once(&lp_aesop_Aesop_Goal_traceMetadata___closed__13, &lp_aesop_Aesop_Goal_traceMetadata___closed__13_once, _init_lp_aesop_Aesop_Goal_traceMetadata___closed__13);
v___x_3224_ = ((lean_object*)(lp_aesop_Aesop_Goal_traceMetadata___closed__14));
v___x_3225_ = lp_aesop___private_Init_WFExtrinsicFix_0__WellFounded_opaqueFix_u2082___at___00Aesop_Goal_traceMetadata_spec__17___redArg(v_unsafeQueue_3182_, v___x_3224_);
v___x_3226_ = lean_box(0);
v_sz_3227_ = lean_array_size(v___x_3225_);
v___x_3228_ = lean_box(v_unsafeRulesSelected_3181_);
v___x_3229_ = lean_box_usize(v_sz_3227_);
v___x_3230_ = lean_box_usize(v___y_3208_);
lean_inc_ref_n(v_traceOpt_3126_, 2);
v___f_3231_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Goal_traceMetadata___lam__4___boxed), 11, 6);
lean_closure_set(v___f_3231_, 0, v___x_3228_);
lean_closure_set(v___f_3231_, 1, v_traceOpt_3126_);
lean_closure_set(v___f_3231_, 2, v___x_3225_);
lean_closure_set(v___f_3231_, 3, v___x_3229_);
lean_closure_set(v___f_3231_, 4, v___x_3230_);
lean_closure_set(v___f_3231_, 5, v___x_3226_);
v___x_3232_ = lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Goal_traceMetadata_trcNode(v_traceOpt_3126_, v___x_3223_, v___f_3231_, v___y_3213_, v___y_3210_, v___y_3212_, v___y_3214_);
if (lean_obj_tag(v___x_3232_) == 0)
{
lean_dec_ref_known(v___x_3232_, 1);
v___y_3138_ = v___y_3213_;
v___y_3139_ = v___y_3210_;
v___y_3140_ = v___y_3212_;
v___y_3141_ = v___y_3214_;
goto v___jp_3137_;
}
else
{
lean_dec(v_a_3133_);
lean_dec_ref(v_traceOpt_3126_);
lean_dec(v_g_3125_);
return v___x_3232_;
}
}
else
{
lean_object* v___x_3233_; lean_object* v___x_3234_; 
lean_dec_ref(v_unsafeQueue_3182_);
v___x_3233_ = lean_obj_once(&lp_aesop_Aesop_Goal_traceMetadata___closed__16, &lp_aesop_Aesop_Goal_traceMetadata___closed__16_once, _init_lp_aesop_Aesop_Goal_traceMetadata___closed__16);
lean_inc_ref(v_traceOpt_3126_);
v___x_3234_ = lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Goal_traceMetadata_trc(v_traceOpt_3126_, v___x_3233_, v___y_3213_, v___y_3210_, v___y_3212_, v___y_3214_);
if (lean_obj_tag(v___x_3234_) == 0)
{
lean_dec_ref_known(v___x_3234_, 1);
v___y_3138_ = v___y_3213_;
v___y_3139_ = v___y_3210_;
v___y_3140_ = v___y_3212_;
v___y_3141_ = v___y_3214_;
goto v___jp_3137_;
}
else
{
lean_dec(v_a_3133_);
lean_dec_ref(v_traceOpt_3126_);
lean_dec(v_g_3125_);
return v___x_3234_;
}
}
}
}
else
{
lean_dec_ref(v_unsafeQueue_3182_);
lean_dec(v_a_3133_);
lean_dec_ref(v_traceOpt_3126_);
lean_dec(v_g_3125_);
return v___x_3217_;
}
}
v___jp_3235_:
{
lean_object* v___x_3244_; lean_object* v___x_3245_; lean_object* v___x_3246_; 
v___x_3244_ = l_Lean_stringToMessageData(v___y_3243_);
lean_inc_ref(v___y_3238_);
v___x_3245_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3245_, 0, v___y_3238_);
lean_ctor_set(v___x_3245_, 1, v___x_3244_);
lean_inc_ref(v_traceOpt_3126_);
v___x_3246_ = lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Goal_traceMetadata_trc(v_traceOpt_3126_, v___x_3245_, v___y_3241_, v___y_3239_, v___y_3240_, v___y_3242_);
if (lean_obj_tag(v___x_3246_) == 0)
{
lean_object* v___x_3248_; uint8_t v_isShared_3249_; uint8_t v_isSharedCheck_3258_; 
v_isSharedCheck_3258_ = !lean_is_exclusive(v___x_3246_);
if (v_isSharedCheck_3258_ == 0)
{
lean_object* v_unused_3259_; 
v_unused_3259_ = lean_ctor_get(v___x_3246_, 0);
lean_dec(v_unused_3259_);
v___x_3248_ = v___x_3246_;
v_isShared_3249_ = v_isSharedCheck_3258_;
goto v_resetjp_3247_;
}
else
{
lean_dec(v___x_3246_);
v___x_3248_ = lean_box(0);
v_isShared_3249_ = v_isSharedCheck_3258_;
goto v_resetjp_3247_;
}
v_resetjp_3247_:
{
lean_object* v___x_3250_; 
v___x_3250_ = lean_obj_once(&lp_aesop_Aesop_Goal_traceMetadata___closed__18, &lp_aesop_Aesop_Goal_traceMetadata___closed__18_once, _init_lp_aesop_Aesop_Goal_traceMetadata___closed__18);
if (lean_obj_tag(v_normalizationState_3175_) == 1)
{
lean_object* v_postGoal_3251_; lean_object* v_postState_3252_; lean_object* v___x_3253_; lean_object* v___x_3255_; 
v_postGoal_3251_ = lean_ctor_get(v_normalizationState_3175_, 0);
lean_inc(v_postGoal_3251_);
v_postState_3252_ = lean_ctor_get(v_normalizationState_3175_, 1);
lean_inc_ref(v_postState_3252_);
lean_dec_ref_known(v_normalizationState_3175_, 3);
v___x_3253_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3253_, 0, v_postGoal_3251_);
lean_ctor_set(v___x_3253_, 1, v_postState_3252_);
if (v_isShared_3249_ == 0)
{
lean_ctor_set_tag(v___x_3248_, 1);
lean_ctor_set(v___x_3248_, 0, v___x_3253_);
v___x_3255_ = v___x_3248_;
goto v_reusejp_3254_;
}
else
{
lean_object* v_reuseFailAlloc_3256_; 
v_reuseFailAlloc_3256_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3256_, 0, v___x_3253_);
v___x_3255_ = v_reuseFailAlloc_3256_;
goto v_reusejp_3254_;
}
v_reusejp_3254_:
{
v___y_3208_ = v___y_3236_;
v___y_3209_ = v___y_3237_;
v___y_3210_ = v___y_3239_;
v___y_3211_ = v___x_3250_;
v___y_3212_ = v___y_3240_;
v___y_3213_ = v___y_3241_;
v___y_3214_ = v___y_3242_;
v___y_3215_ = v___x_3255_;
goto v___jp_3207_;
}
}
else
{
lean_object* v___x_3257_; 
lean_del_object(v___x_3248_);
lean_dec(v_normalizationState_3175_);
v___x_3257_ = lean_box(0);
v___y_3208_ = v___y_3236_;
v___y_3209_ = v___y_3237_;
v___y_3210_ = v___y_3239_;
v___y_3211_ = v___x_3250_;
v___y_3212_ = v___y_3240_;
v___y_3213_ = v___y_3241_;
v___y_3214_ = v___y_3242_;
v___y_3215_ = v___x_3257_;
goto v___jp_3207_;
}
}
}
else
{
lean_dec(v___y_3237_);
lean_dec_ref(v___f_3206_);
lean_dec_ref(v___f_3205_);
lean_dec_ref(v_unsafeQueue_3182_);
lean_dec_ref(v_forwardState_3177_);
lean_dec(v_normalizationState_3175_);
lean_dec(v_preNormGoal_3174_);
lean_dec(v_a_3133_);
lean_dec_ref(v_traceOpt_3126_);
lean_dec(v_g_3125_);
return v___x_3246_;
}
}
v___jp_3260_:
{
lean_object* v___x_3270_; 
lean_inc_ref(v___y_3268_);
if (v_isShared_3204_ == 0)
{
lean_ctor_set_tag(v___x_3203_, 3);
lean_ctor_set(v___x_3203_, 0, v___y_3268_);
v___x_3270_ = v___x_3203_;
goto v_reusejp_3269_;
}
else
{
lean_object* v_reuseFailAlloc_3316_; 
v_reuseFailAlloc_3316_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3316_, 0, v___y_3268_);
v___x_3270_ = v_reuseFailAlloc_3316_;
goto v_reusejp_3269_;
}
v_reusejp_3269_:
{
lean_object* v___x_3271_; lean_object* v___x_3272_; lean_object* v___x_3273_; 
v___x_3271_ = l_Lean_MessageData_ofFormat(v___x_3270_);
v___x_3272_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3272_, 0, v___y_3264_);
lean_ctor_set(v___x_3272_, 1, v___x_3271_);
lean_inc_ref(v_traceOpt_3126_);
v___x_3273_ = lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Goal_traceMetadata_trc(v_traceOpt_3126_, v___x_3272_, v___y_3266_, v___y_3263_, v___y_3265_, v___y_3267_);
if (lean_obj_tag(v___x_3273_) == 0)
{
lean_object* v___x_3275_; uint8_t v_isShared_3276_; uint8_t v_isSharedCheck_3314_; 
v_isSharedCheck_3314_ = !lean_is_exclusive(v___x_3273_);
if (v_isSharedCheck_3314_ == 0)
{
lean_object* v_unused_3315_; 
v_unused_3315_ = lean_ctor_get(v___x_3273_, 0);
lean_dec(v_unused_3315_);
v___x_3275_ = v___x_3273_;
v_isShared_3276_ = v_isSharedCheck_3314_;
goto v_resetjp_3274_;
}
else
{
lean_dec(v___x_3273_);
v___x_3275_ = lean_box(0);
v_isShared_3276_ = v_isSharedCheck_3314_;
goto v_resetjp_3274_;
}
v_resetjp_3274_:
{
lean_object* v___x_3277_; lean_object* v___x_3278_; lean_object* v___x_3279_; lean_object* v___x_3281_; 
v___x_3277_ = lean_obj_once(&lp_aesop_Aesop_Goal_traceMetadata___closed__20, &lp_aesop_Aesop_Goal_traceMetadata___closed__20_once, _init_lp_aesop_Aesop_Goal_traceMetadata___closed__20);
v___x_3278_ = lp_aesop_Aesop_ForwardRuleMatches_size(v_forwardRuleMatches_3178_);
lean_dec_ref(v_forwardRuleMatches_3178_);
v___x_3279_ = l_Nat_reprFast(v___x_3278_);
if (v_isShared_3276_ == 0)
{
lean_ctor_set_tag(v___x_3275_, 3);
lean_ctor_set(v___x_3275_, 0, v___x_3279_);
v___x_3281_ = v___x_3275_;
goto v_reusejp_3280_;
}
else
{
lean_object* v_reuseFailAlloc_3313_; 
v_reuseFailAlloc_3313_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3313_, 0, v___x_3279_);
v___x_3281_ = v_reuseFailAlloc_3313_;
goto v_reusejp_3280_;
}
v_reusejp_3280_:
{
lean_object* v___x_3282_; lean_object* v___x_3283_; lean_object* v___x_3284_; 
v___x_3282_ = l_Lean_MessageData_ofFormat(v___x_3281_);
v___x_3283_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3283_, 0, v___x_3277_);
lean_ctor_set(v___x_3283_, 1, v___x_3282_);
lean_inc_ref(v_traceOpt_3126_);
v___x_3284_ = lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Goal_traceMetadata_trc(v_traceOpt_3126_, v___x_3283_, v___y_3266_, v___y_3263_, v___y_3265_, v___y_3267_);
if (lean_obj_tag(v___x_3284_) == 0)
{
lean_object* v___x_3285_; lean_object* v___x_3286_; lean_object* v___x_3287_; lean_object* v___x_3288_; lean_object* v___x_3289_; 
lean_dec_ref_known(v___x_3284_, 1);
v___x_3285_ = lean_obj_once(&lp_aesop_Aesop_Goal_traceMetadata___closed__22, &lp_aesop_Aesop_Goal_traceMetadata___closed__22_once, _init_lp_aesop_Aesop_Goal_traceMetadata___closed__22);
v___x_3286_ = lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_toYesNo(v_isIrrelevant_3172_);
v___x_3287_ = l_Lean_stringToMessageData(v___x_3286_);
v___x_3288_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3288_, 0, v___x_3285_);
lean_ctor_set(v___x_3288_, 1, v___x_3287_);
lean_inc_ref(v_traceOpt_3126_);
v___x_3289_ = lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Goal_traceMetadata_trc(v_traceOpt_3126_, v___x_3288_, v___y_3266_, v___y_3263_, v___y_3265_, v___y_3267_);
if (lean_obj_tag(v___x_3289_) == 0)
{
lean_object* v___x_3290_; lean_object* v___x_3291_; lean_object* v___x_3292_; lean_object* v___x_3293_; lean_object* v___x_3294_; 
lean_dec_ref_known(v___x_3289_, 1);
v___x_3290_ = lean_obj_once(&lp_aesop_Aesop_Goal_traceMetadata___closed__24, &lp_aesop_Aesop_Goal_traceMetadata___closed__24_once, _init_lp_aesop_Aesop_Goal_traceMetadata___closed__24);
v___x_3291_ = lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_toYesNo(v_isForcedUnprovable_3173_);
v___x_3292_ = l_Lean_stringToMessageData(v___x_3291_);
v___x_3293_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3293_, 0, v___x_3290_);
lean_ctor_set(v___x_3293_, 1, v___x_3292_);
lean_inc_ref(v_traceOpt_3126_);
v___x_3294_ = lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Goal_traceMetadata_trc(v_traceOpt_3126_, v___x_3293_, v___y_3266_, v___y_3263_, v___y_3265_, v___y_3267_);
if (lean_obj_tag(v___x_3294_) == 0)
{
lean_object* v___x_3296_; uint8_t v_isShared_3297_; uint8_t v_isSharedCheck_3311_; 
v_isSharedCheck_3311_ = !lean_is_exclusive(v___x_3294_);
if (v_isSharedCheck_3311_ == 0)
{
lean_object* v_unused_3312_; 
v_unused_3312_ = lean_ctor_get(v___x_3294_, 0);
lean_dec(v_unused_3312_);
v___x_3296_ = v___x_3294_;
v_isShared_3297_ = v_isSharedCheck_3311_;
goto v_resetjp_3295_;
}
else
{
lean_dec(v___x_3294_);
v___x_3296_ = lean_box(0);
v_isShared_3297_ = v_isSharedCheck_3311_;
goto v_resetjp_3295_;
}
v_resetjp_3295_:
{
lean_object* v___x_3298_; lean_object* v___x_3299_; lean_object* v___x_3301_; 
v___x_3298_ = lean_obj_once(&lp_aesop_Aesop_Goal_traceMetadata___closed__26, &lp_aesop_Aesop_Goal_traceMetadata___closed__26_once, _init_lp_aesop_Aesop_Goal_traceMetadata___closed__26);
v___x_3299_ = l_Nat_reprFast(v_addedInIteration_3179_);
if (v_isShared_3297_ == 0)
{
lean_ctor_set_tag(v___x_3296_, 3);
lean_ctor_set(v___x_3296_, 0, v___x_3299_);
v___x_3301_ = v___x_3296_;
goto v_reusejp_3300_;
}
else
{
lean_object* v_reuseFailAlloc_3310_; 
v_reuseFailAlloc_3310_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3310_, 0, v___x_3299_);
v___x_3301_ = v_reuseFailAlloc_3310_;
goto v_reusejp_3300_;
}
v_reusejp_3300_:
{
lean_object* v___x_3302_; lean_object* v___x_3303_; lean_object* v___x_3304_; 
v___x_3302_ = l_Lean_MessageData_ofFormat(v___x_3301_);
v___x_3303_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3303_, 0, v___x_3298_);
lean_ctor_set(v___x_3303_, 1, v___x_3302_);
lean_inc_ref(v_traceOpt_3126_);
v___x_3304_ = lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Goal_traceMetadata_trc(v_traceOpt_3126_, v___x_3303_, v___y_3266_, v___y_3263_, v___y_3265_, v___y_3267_);
if (lean_obj_tag(v___x_3304_) == 0)
{
lean_object* v___x_3305_; lean_object* v___x_3306_; uint8_t v___x_3307_; 
lean_dec_ref_known(v___x_3304_, 1);
v___x_3305_ = lean_obj_once(&lp_aesop_Aesop_Goal_traceMetadata___closed__28, &lp_aesop_Aesop_Goal_traceMetadata___closed__28_once, _init_lp_aesop_Aesop_Goal_traceMetadata___closed__28);
v___x_3306_ = lp_aesop_Aesop_Iteration_none;
v___x_3307_ = lean_nat_dec_eq(v_lastExpandedInIteration_3180_, v___x_3306_);
if (v___x_3307_ == 0)
{
lean_object* v___x_3308_; 
v___x_3308_ = l_Nat_reprFast(v_lastExpandedInIteration_3180_);
v___y_3236_ = v___y_3261_;
v___y_3237_ = v___y_3262_;
v___y_3238_ = v___x_3305_;
v___y_3239_ = v___y_3263_;
v___y_3240_ = v___y_3265_;
v___y_3241_ = v___y_3266_;
v___y_3242_ = v___y_3267_;
v___y_3243_ = v___x_3308_;
goto v___jp_3235_;
}
else
{
lean_object* v___x_3309_; 
lean_dec(v_lastExpandedInIteration_3180_);
v___x_3309_ = ((lean_object*)(lp_aesop_Aesop_Goal_traceMetadata___closed__29));
v___y_3236_ = v___y_3261_;
v___y_3237_ = v___y_3262_;
v___y_3238_ = v___x_3305_;
v___y_3239_ = v___y_3263_;
v___y_3240_ = v___y_3265_;
v___y_3241_ = v___y_3266_;
v___y_3242_ = v___y_3267_;
v___y_3243_ = v___x_3309_;
goto v___jp_3235_;
}
}
else
{
lean_dec(v___y_3262_);
lean_dec_ref(v___f_3206_);
lean_dec_ref(v___f_3205_);
lean_dec_ref(v_unsafeQueue_3182_);
lean_dec(v_lastExpandedInIteration_3180_);
lean_dec_ref(v_forwardState_3177_);
lean_dec(v_normalizationState_3175_);
lean_dec(v_preNormGoal_3174_);
lean_dec(v_a_3133_);
lean_dec_ref(v_traceOpt_3126_);
lean_dec(v_g_3125_);
return v___x_3304_;
}
}
}
}
else
{
lean_dec(v___y_3262_);
lean_dec_ref(v___f_3206_);
lean_dec_ref(v___f_3205_);
lean_dec_ref(v_unsafeQueue_3182_);
lean_dec(v_lastExpandedInIteration_3180_);
lean_dec(v_addedInIteration_3179_);
lean_dec_ref(v_forwardState_3177_);
lean_dec(v_normalizationState_3175_);
lean_dec(v_preNormGoal_3174_);
lean_dec(v_a_3133_);
lean_dec_ref(v_traceOpt_3126_);
lean_dec(v_g_3125_);
return v___x_3294_;
}
}
else
{
lean_dec(v___y_3262_);
lean_dec_ref(v___f_3206_);
lean_dec_ref(v___f_3205_);
lean_dec_ref(v_unsafeQueue_3182_);
lean_dec(v_lastExpandedInIteration_3180_);
lean_dec(v_addedInIteration_3179_);
lean_dec_ref(v_forwardState_3177_);
lean_dec(v_normalizationState_3175_);
lean_dec(v_preNormGoal_3174_);
lean_dec(v_a_3133_);
lean_dec_ref(v_traceOpt_3126_);
lean_dec(v_g_3125_);
return v___x_3289_;
}
}
else
{
lean_dec(v___y_3262_);
lean_dec_ref(v___f_3206_);
lean_dec_ref(v___f_3205_);
lean_dec_ref(v_unsafeQueue_3182_);
lean_dec(v_lastExpandedInIteration_3180_);
lean_dec(v_addedInIteration_3179_);
lean_dec_ref(v_forwardState_3177_);
lean_dec(v_normalizationState_3175_);
lean_dec(v_preNormGoal_3174_);
lean_dec(v_a_3133_);
lean_dec_ref(v_traceOpt_3126_);
lean_dec(v_g_3125_);
return v___x_3284_;
}
}
}
}
else
{
lean_dec(v___y_3262_);
lean_dec_ref(v___f_3206_);
lean_dec_ref(v___f_3205_);
lean_dec_ref(v_unsafeQueue_3182_);
lean_dec(v_lastExpandedInIteration_3180_);
lean_dec(v_addedInIteration_3179_);
lean_dec_ref(v_forwardRuleMatches_3178_);
lean_dec_ref(v_forwardState_3177_);
lean_dec(v_normalizationState_3175_);
lean_dec(v_preNormGoal_3174_);
lean_dec(v_a_3133_);
lean_dec_ref(v_traceOpt_3126_);
lean_dec(v_g_3125_);
return v___x_3273_;
}
}
}
v___jp_3317_:
{
lean_object* v___x_3328_; lean_object* v___x_3329_; 
lean_inc_ref(v___y_3325_);
v___x_3328_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3328_, 0, v___y_3325_);
lean_ctor_set(v___x_3328_, 1, v___y_3327_);
lean_inc_ref(v_traceOpt_3126_);
v___x_3329_ = lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Goal_traceMetadata_trc(v_traceOpt_3126_, v___x_3328_, v___y_3324_, v___y_3320_, v___y_3323_, v___y_3326_);
if (lean_obj_tag(v___x_3329_) == 0)
{
size_t v_sz_3330_; lean_object* v___x_3331_; 
lean_dec_ref_known(v___x_3329_, 1);
v_sz_3330_ = lean_array_size(v_children_3168_);
v___x_3331_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Goal_traceMetadata_spec__14___redArg(v_sz_3330_, v___y_3321_, v_children_3168_);
if (lean_obj_tag(v___x_3331_) == 0)
{
lean_object* v_a_3332_; lean_object* v___x_3333_; lean_object* v___x_3334_; lean_object* v___x_3335_; lean_object* v___x_3336_; lean_object* v___x_3337_; lean_object* v___x_3338_; 
v_a_3332_ = lean_ctor_get(v___x_3331_, 0);
lean_inc(v_a_3332_);
lean_dec_ref_known(v___x_3331_, 1);
v___x_3333_ = lean_obj_once(&lp_aesop_Aesop_Goal_traceMetadata___closed__31, &lp_aesop_Aesop_Goal_traceMetadata___closed__31_once, _init_lp_aesop_Aesop_Goal_traceMetadata___closed__31);
v___x_3334_ = lean_array_to_list(v_a_3332_);
v___x_3335_ = lp_aesop_List_mapTR_loop___at___00Aesop_Goal_traceMetadata_spec__15(v___x_3334_, v___y_3322_);
v___x_3336_ = l_Lean_MessageData_ofList(v___x_3335_);
v___x_3337_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3337_, 0, v___x_3333_);
lean_ctor_set(v___x_3337_, 1, v___x_3336_);
lean_inc_ref(v_traceOpt_3126_);
v___x_3338_ = lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Goal_traceMetadata_trc(v_traceOpt_3126_, v___x_3337_, v___y_3324_, v___y_3320_, v___y_3323_, v___y_3326_);
if (lean_obj_tag(v___x_3338_) == 0)
{
lean_object* v___x_3339_; lean_object* v___x_3340_; lean_object* v___x_3341_; lean_object* v___x_3342_; lean_object* v___x_3343_; 
lean_dec_ref_known(v___x_3338_, 1);
v___x_3339_ = lean_obj_once(&lp_aesop_Aesop_Goal_traceMetadata___closed__33, &lp_aesop_Aesop_Goal_traceMetadata___closed__33_once, _init_lp_aesop_Aesop_Goal_traceMetadata___closed__33);
v___x_3340_ = lp_aesop_Aesop_GoalOrigin_toString(v_origin_3169_);
v___x_3341_ = l_Lean_stringToMessageData(v___x_3340_);
v___x_3342_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3342_, 0, v___x_3339_);
lean_ctor_set(v___x_3342_, 1, v___x_3341_);
lean_inc_ref(v_traceOpt_3126_);
v___x_3343_ = lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Goal_traceMetadata_trc(v_traceOpt_3126_, v___x_3342_, v___y_3324_, v___y_3320_, v___y_3323_, v___y_3326_);
if (lean_obj_tag(v___x_3343_) == 0)
{
lean_object* v___x_3345_; uint8_t v_isShared_3346_; uint8_t v_isSharedCheck_3365_; 
v_isSharedCheck_3365_ = !lean_is_exclusive(v___x_3343_);
if (v_isSharedCheck_3365_ == 0)
{
lean_object* v_unused_3366_; 
v_unused_3366_ = lean_ctor_get(v___x_3343_, 0);
lean_dec(v_unused_3366_);
v___x_3345_ = v___x_3343_;
v_isShared_3346_ = v_isSharedCheck_3365_;
goto v_resetjp_3344_;
}
else
{
lean_dec(v___x_3343_);
v___x_3345_ = lean_box(0);
v_isShared_3346_ = v_isSharedCheck_3365_;
goto v_resetjp_3344_;
}
v_resetjp_3344_:
{
lean_object* v___x_3347_; lean_object* v___x_3348_; lean_object* v___x_3350_; 
v___x_3347_ = lean_obj_once(&lp_aesop_Aesop_Goal_traceMetadata___closed__35, &lp_aesop_Aesop_Goal_traceMetadata___closed__35_once, _init_lp_aesop_Aesop_Goal_traceMetadata___closed__35);
v___x_3348_ = l_Nat_reprFast(v_depth_3170_);
if (v_isShared_3346_ == 0)
{
lean_ctor_set_tag(v___x_3345_, 3);
lean_ctor_set(v___x_3345_, 0, v___x_3348_);
v___x_3350_ = v___x_3345_;
goto v_reusejp_3349_;
}
else
{
lean_object* v_reuseFailAlloc_3364_; 
v_reuseFailAlloc_3364_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3364_, 0, v___x_3348_);
v___x_3350_ = v_reuseFailAlloc_3364_;
goto v_reusejp_3349_;
}
v_reusejp_3349_:
{
lean_object* v___x_3351_; lean_object* v___x_3352_; lean_object* v___x_3353_; 
v___x_3351_ = l_Lean_MessageData_ofFormat(v___x_3350_);
v___x_3352_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3352_, 0, v___x_3347_);
lean_ctor_set(v___x_3352_, 1, v___x_3351_);
lean_inc_ref(v_traceOpt_3126_);
v___x_3353_ = lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Goal_traceMetadata_trc(v_traceOpt_3126_, v___x_3352_, v___y_3324_, v___y_3320_, v___y_3323_, v___y_3326_);
if (lean_obj_tag(v___x_3353_) == 0)
{
lean_object* v___x_3354_; lean_object* v___x_3355_; lean_object* v___x_3356_; lean_object* v___x_3357_; lean_object* v___x_3358_; lean_object* v___x_3359_; 
lean_dec_ref_known(v___x_3353_, 1);
v___x_3354_ = lean_obj_once(&lp_aesop_Aesop_Goal_traceMetadata___closed__37, &lp_aesop_Aesop_Goal_traceMetadata___closed__37_once, _init_lp_aesop_Aesop_Goal_traceMetadata___closed__37);
v___x_3355_ = lp_aesop_Aesop_GoalState_toEmoji(v_state_3171_);
v___x_3356_ = l_Lean_stringToMessageData(v___x_3355_);
v___x_3357_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3357_, 0, v___x_3354_);
lean_ctor_set(v___x_3357_, 1, v___x_3356_);
v___x_3358_ = lean_obj_once(&lp_aesop_Aesop_Goal_traceMetadata___closed__39, &lp_aesop_Aesop_Goal_traceMetadata___closed__39_once, _init_lp_aesop_Aesop_Goal_traceMetadata___closed__39);
v___x_3359_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3359_, 0, v___x_3357_);
lean_ctor_set(v___x_3359_, 1, v___x_3358_);
switch(v_state_3171_)
{
case 0:
{
lean_object* v___x_3360_; 
v___x_3360_ = ((lean_object*)(lp_aesop_Aesop_Goal_traceMetadata___closed__40));
v___y_3261_ = v___y_3318_;
v___y_3262_ = v___y_3319_;
v___y_3263_ = v___y_3320_;
v___y_3264_ = v___x_3359_;
v___y_3265_ = v___y_3323_;
v___y_3266_ = v___y_3324_;
v___y_3267_ = v___y_3326_;
v___y_3268_ = v___x_3360_;
goto v___jp_3260_;
}
case 1:
{
lean_object* v___x_3361_; 
v___x_3361_ = ((lean_object*)(lp_aesop_Aesop_Goal_traceMetadata___closed__41));
v___y_3261_ = v___y_3318_;
v___y_3262_ = v___y_3319_;
v___y_3263_ = v___y_3320_;
v___y_3264_ = v___x_3359_;
v___y_3265_ = v___y_3323_;
v___y_3266_ = v___y_3324_;
v___y_3267_ = v___y_3326_;
v___y_3268_ = v___x_3361_;
goto v___jp_3260_;
}
case 2:
{
lean_object* v___x_3362_; 
v___x_3362_ = ((lean_object*)(lp_aesop_Aesop_Goal_traceMetadata___closed__42));
v___y_3261_ = v___y_3318_;
v___y_3262_ = v___y_3319_;
v___y_3263_ = v___y_3320_;
v___y_3264_ = v___x_3359_;
v___y_3265_ = v___y_3323_;
v___y_3266_ = v___y_3324_;
v___y_3267_ = v___y_3326_;
v___y_3268_ = v___x_3362_;
goto v___jp_3260_;
}
default: 
{
lean_object* v___x_3363_; 
v___x_3363_ = ((lean_object*)(lp_aesop_Aesop_Goal_traceMetadata___closed__43));
v___y_3261_ = v___y_3318_;
v___y_3262_ = v___y_3319_;
v___y_3263_ = v___y_3320_;
v___y_3264_ = v___x_3359_;
v___y_3265_ = v___y_3323_;
v___y_3266_ = v___y_3324_;
v___y_3267_ = v___y_3326_;
v___y_3268_ = v___x_3363_;
goto v___jp_3260_;
}
}
}
else
{
lean_dec(v___y_3319_);
lean_dec_ref(v___f_3206_);
lean_dec_ref(v___f_3205_);
lean_del_object(v___x_3203_);
lean_dec_ref(v_unsafeQueue_3182_);
lean_dec(v_lastExpandedInIteration_3180_);
lean_dec(v_addedInIteration_3179_);
lean_dec_ref(v_forwardRuleMatches_3178_);
lean_dec_ref(v_forwardState_3177_);
lean_dec(v_normalizationState_3175_);
lean_dec(v_preNormGoal_3174_);
lean_dec(v_a_3133_);
lean_dec_ref(v_traceOpt_3126_);
lean_dec(v_g_3125_);
return v___x_3353_;
}
}
}
}
else
{
lean_dec(v___y_3319_);
lean_dec_ref(v___f_3206_);
lean_dec_ref(v___f_3205_);
lean_del_object(v___x_3203_);
lean_dec_ref(v_unsafeQueue_3182_);
lean_dec(v_lastExpandedInIteration_3180_);
lean_dec(v_addedInIteration_3179_);
lean_dec_ref(v_forwardRuleMatches_3178_);
lean_dec_ref(v_forwardState_3177_);
lean_dec(v_normalizationState_3175_);
lean_dec(v_preNormGoal_3174_);
lean_dec(v_depth_3170_);
lean_dec(v_a_3133_);
lean_dec_ref(v_traceOpt_3126_);
lean_dec(v_g_3125_);
return v___x_3343_;
}
}
else
{
lean_dec(v___y_3319_);
lean_dec_ref(v___f_3206_);
lean_dec_ref(v___f_3205_);
lean_del_object(v___x_3203_);
lean_dec_ref(v_unsafeQueue_3182_);
lean_dec(v_lastExpandedInIteration_3180_);
lean_dec(v_addedInIteration_3179_);
lean_dec_ref(v_forwardRuleMatches_3178_);
lean_dec_ref(v_forwardState_3177_);
lean_dec(v_normalizationState_3175_);
lean_dec(v_preNormGoal_3174_);
lean_dec(v_depth_3170_);
lean_dec(v_origin_3169_);
lean_dec(v_a_3133_);
lean_dec_ref(v_traceOpt_3126_);
lean_dec(v_g_3125_);
return v___x_3338_;
}
}
else
{
lean_object* v_a_3367_; lean_object* v___x_3369_; uint8_t v_isShared_3370_; uint8_t v_isSharedCheck_3374_; 
lean_dec(v___y_3322_);
lean_dec(v___y_3319_);
lean_dec_ref(v___f_3206_);
lean_dec_ref(v___f_3205_);
lean_del_object(v___x_3203_);
lean_dec_ref(v_unsafeQueue_3182_);
lean_dec(v_lastExpandedInIteration_3180_);
lean_dec(v_addedInIteration_3179_);
lean_dec_ref(v_forwardRuleMatches_3178_);
lean_dec_ref(v_forwardState_3177_);
lean_dec(v_normalizationState_3175_);
lean_dec(v_preNormGoal_3174_);
lean_dec(v_depth_3170_);
lean_dec(v_origin_3169_);
lean_dec(v_a_3133_);
lean_dec_ref(v_traceOpt_3126_);
lean_dec(v_g_3125_);
v_a_3367_ = lean_ctor_get(v___x_3331_, 0);
v_isSharedCheck_3374_ = !lean_is_exclusive(v___x_3331_);
if (v_isSharedCheck_3374_ == 0)
{
v___x_3369_ = v___x_3331_;
v_isShared_3370_ = v_isSharedCheck_3374_;
goto v_resetjp_3368_;
}
else
{
lean_inc(v_a_3367_);
lean_dec(v___x_3331_);
v___x_3369_ = lean_box(0);
v_isShared_3370_ = v_isSharedCheck_3374_;
goto v_resetjp_3368_;
}
v_resetjp_3368_:
{
lean_object* v___x_3372_; 
if (v_isShared_3370_ == 0)
{
v___x_3372_ = v___x_3369_;
goto v_reusejp_3371_;
}
else
{
lean_object* v_reuseFailAlloc_3373_; 
v_reuseFailAlloc_3373_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3373_, 0, v_a_3367_);
v___x_3372_ = v_reuseFailAlloc_3373_;
goto v_reusejp_3371_;
}
v_reusejp_3371_:
{
return v___x_3372_;
}
}
}
}
else
{
lean_dec(v___y_3322_);
lean_dec(v___y_3319_);
lean_dec_ref(v___f_3206_);
lean_dec_ref(v___f_3205_);
lean_del_object(v___x_3203_);
lean_dec_ref(v_unsafeQueue_3182_);
lean_dec(v_lastExpandedInIteration_3180_);
lean_dec(v_addedInIteration_3179_);
lean_dec_ref(v_forwardRuleMatches_3178_);
lean_dec_ref(v_forwardState_3177_);
lean_dec(v_normalizationState_3175_);
lean_dec(v_preNormGoal_3174_);
lean_dec(v_depth_3170_);
lean_dec(v_origin_3169_);
lean_dec_ref(v_children_3168_);
lean_dec(v_a_3133_);
lean_dec_ref(v_traceOpt_3126_);
lean_dec(v_g_3125_);
return v___x_3329_;
}
}
v___jp_3375_:
{
lean_object* v___x_3385_; 
v___x_3385_ = lean_obj_once(&lp_aesop_Aesop_Goal_traceMetadata___closed__45, &lp_aesop_Aesop_Goal_traceMetadata___closed__45_once, _init_lp_aesop_Aesop_Goal_traceMetadata___closed__45);
if (lean_obj_tag(v_a_3384_) == 0)
{
lean_object* v___x_3386_; 
v___x_3386_ = lean_obj_once(&lp_aesop_Aesop_Goal_traceMetadata___closed__48, &lp_aesop_Aesop_Goal_traceMetadata___closed__48_once, _init_lp_aesop_Aesop_Goal_traceMetadata___closed__48);
v___y_3318_ = v___y_3376_;
v___y_3319_ = v___y_3377_;
v___y_3320_ = v___y_3378_;
v___y_3321_ = v___y_3379_;
v___y_3322_ = v___y_3380_;
v___y_3323_ = v___y_3382_;
v___y_3324_ = v___y_3381_;
v___y_3325_ = v___x_3385_;
v___y_3326_ = v___y_3383_;
v___y_3327_ = v___x_3386_;
goto v___jp_3317_;
}
else
{
lean_object* v_val_3387_; lean_object* v___x_3389_; uint8_t v_isShared_3390_; uint8_t v_isSharedCheck_3400_; 
v_val_3387_ = lean_ctor_get(v_a_3384_, 0);
v_isSharedCheck_3400_ = !lean_is_exclusive(v_a_3384_);
if (v_isSharedCheck_3400_ == 0)
{
v___x_3389_ = v_a_3384_;
v_isShared_3390_ = v_isSharedCheck_3400_;
goto v_resetjp_3388_;
}
else
{
lean_inc(v_val_3387_);
lean_dec(v_a_3384_);
v___x_3389_ = lean_box(0);
v_isShared_3390_ = v_isSharedCheck_3400_;
goto v_resetjp_3388_;
}
v_resetjp_3388_:
{
lean_object* v___x_3391_; lean_object* v___x_3392_; lean_object* v___x_3394_; 
v___x_3391_ = lean_obj_once(&lp_aesop_Aesop_Goal_traceMetadata___closed__51, &lp_aesop_Aesop_Goal_traceMetadata___closed__51_once, _init_lp_aesop_Aesop_Goal_traceMetadata___closed__51);
v___x_3392_ = l_Nat_reprFast(v_val_3387_);
if (v_isShared_3390_ == 0)
{
lean_ctor_set_tag(v___x_3389_, 3);
lean_ctor_set(v___x_3389_, 0, v___x_3392_);
v___x_3394_ = v___x_3389_;
goto v_reusejp_3393_;
}
else
{
lean_object* v_reuseFailAlloc_3399_; 
v_reuseFailAlloc_3399_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3399_, 0, v___x_3392_);
v___x_3394_ = v_reuseFailAlloc_3399_;
goto v_reusejp_3393_;
}
v_reusejp_3393_:
{
lean_object* v___x_3395_; lean_object* v___x_3396_; lean_object* v___x_3397_; lean_object* v___x_3398_; 
v___x_3395_ = l_Lean_MessageData_ofFormat(v___x_3394_);
v___x_3396_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3396_, 0, v___x_3391_);
lean_ctor_set(v___x_3396_, 1, v___x_3395_);
v___x_3397_ = lean_obj_once(&lp_aesop_Aesop_Goal_traceMetadata___closed__54, &lp_aesop_Aesop_Goal_traceMetadata___closed__54_once, _init_lp_aesop_Aesop_Goal_traceMetadata___closed__54);
v___x_3398_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3398_, 0, v___x_3396_);
lean_ctor_set(v___x_3398_, 1, v___x_3397_);
v___y_3318_ = v___y_3376_;
v___y_3319_ = v___y_3377_;
v___y_3320_ = v___y_3378_;
v___y_3321_ = v___y_3379_;
v___y_3322_ = v___y_3380_;
v___y_3323_ = v___y_3382_;
v___y_3324_ = v___y_3381_;
v___y_3325_ = v___x_3385_;
v___y_3326_ = v___y_3383_;
v___y_3327_ = v___x_3398_;
goto v___jp_3317_;
}
}
}
}
v___jp_3401_:
{
lean_object* v___x_3406_; size_t v_sz_3407_; size_t v___x_3408_; lean_object* v___x_3409_; lean_object* v___x_3410_; lean_object* v___x_3411_; lean_object* v___x_3412_; lean_object* v___x_3413_; lean_object* v___x_3414_; lean_object* v___x_3415_; 
v___x_3406_ = lean_obj_once(&lp_aesop_Aesop_Goal_traceMetadata___closed__56, &lp_aesop_Aesop_Goal_traceMetadata___closed__56_once, _init_lp_aesop_Aesop_Goal_traceMetadata___closed__56);
v_sz_3407_ = lean_array_size(v_mvars_3176_);
v___x_3408_ = ((size_t)0ULL);
v___x_3409_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Goal_traceMetadata_spec__12(v_sz_3407_, v___x_3408_, v_mvars_3176_);
v___x_3410_ = lean_array_to_list(v___x_3409_);
v___x_3411_ = lean_box(0);
v___x_3412_ = lp_aesop_List_mapTR_loop___at___00Aesop_Goal_traceMetadata_spec__13(v___x_3410_, v___x_3411_);
v___x_3413_ = l_Lean_MessageData_ofList(v___x_3412_);
v___x_3414_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3414_, 0, v___x_3406_);
lean_ctor_set(v___x_3414_, 1, v___x_3413_);
lean_inc_ref(v_traceOpt_3126_);
v___x_3415_ = lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Goal_traceMetadata_trc(v_traceOpt_3126_, v___x_3414_, v___y_3402_, v___y_3403_, v___y_3404_, v___y_3405_);
if (lean_obj_tag(v___x_3415_) == 0)
{
lean_object* v___x_3416_; 
lean_dec_ref_known(v___x_3415_, 1);
lean_inc(v_g_3125_);
v___x_3416_ = lp_aesop_Aesop_Goal_parentRapp_x3f(v_g_3125_);
if (lean_obj_tag(v___x_3416_) == 0)
{
lean_object* v___x_3417_; 
v___x_3417_ = lean_box(0);
v___y_3376_ = v___x_3408_;
v___y_3377_ = v___x_3411_;
v___y_3378_ = v___y_3403_;
v___y_3379_ = v___x_3408_;
v___y_3380_ = v___x_3411_;
v___y_3381_ = v___y_3402_;
v___y_3382_ = v___y_3404_;
v___y_3383_ = v___y_3405_;
v_a_3384_ = v___x_3417_;
goto v___jp_3375_;
}
else
{
lean_object* v_val_3418_; lean_object* v___x_3420_; uint8_t v_isShared_3421_; uint8_t v_isSharedCheck_3428_; 
v_val_3418_ = lean_ctor_get(v___x_3416_, 0);
v_isSharedCheck_3428_ = !lean_is_exclusive(v___x_3416_);
if (v_isSharedCheck_3428_ == 0)
{
v___x_3420_ = v___x_3416_;
v_isShared_3421_ = v_isSharedCheck_3428_;
goto v_resetjp_3419_;
}
else
{
lean_inc(v_val_3418_);
lean_dec(v___x_3416_);
v___x_3420_ = lean_box(0);
v_isShared_3421_ = v_isSharedCheck_3428_;
goto v_resetjp_3419_;
}
v_resetjp_3419_:
{
lean_object* v___x_3422_; lean_object* v___x_3423_; lean_object* v_id_3424_; lean_object* v___x_3426_; 
v___x_3422_ = lean_st_ref_get(v_val_3418_);
lean_dec(v_val_3418_);
lean_inc_ref(v_elimRapp_3165_);
v___x_3423_ = lean_apply_1(v_elimRapp_3165_, v___x_3422_);
v_id_3424_ = lean_ctor_get(v___x_3423_, 0);
lean_inc(v_id_3424_);
lean_dec_ref(v___x_3423_);
if (v_isShared_3421_ == 0)
{
lean_ctor_set(v___x_3420_, 0, v_id_3424_);
v___x_3426_ = v___x_3420_;
goto v_reusejp_3425_;
}
else
{
lean_object* v_reuseFailAlloc_3427_; 
v_reuseFailAlloc_3427_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3427_, 0, v_id_3424_);
v___x_3426_ = v_reuseFailAlloc_3427_;
goto v_reusejp_3425_;
}
v_reusejp_3425_:
{
v___y_3376_ = v___x_3408_;
v___y_3377_ = v___x_3411_;
v___y_3378_ = v___y_3403_;
v___y_3379_ = v___x_3408_;
v___y_3380_ = v___x_3411_;
v___y_3381_ = v___y_3402_;
v___y_3382_ = v___y_3404_;
v___y_3383_ = v___y_3405_;
v_a_3384_ = v___x_3426_;
goto v___jp_3375_;
}
}
}
}
else
{
lean_dec_ref(v___f_3206_);
lean_dec_ref(v___f_3205_);
lean_del_object(v___x_3203_);
lean_dec_ref(v_unsafeQueue_3182_);
lean_dec(v_lastExpandedInIteration_3180_);
lean_dec(v_addedInIteration_3179_);
lean_dec_ref(v_forwardRuleMatches_3178_);
lean_dec_ref(v_forwardState_3177_);
lean_dec(v_normalizationState_3175_);
lean_dec(v_preNormGoal_3174_);
lean_dec(v_depth_3170_);
lean_dec(v_origin_3169_);
lean_dec_ref(v_children_3168_);
lean_dec(v_a_3133_);
lean_dec_ref(v_traceOpt_3126_);
lean_dec(v_g_3125_);
return v___x_3415_;
}
}
}
}
else
{
lean_dec_ref(v_unsafeQueue_3182_);
lean_dec(v_lastExpandedInIteration_3180_);
lean_dec(v_addedInIteration_3179_);
lean_dec_ref(v_forwardRuleMatches_3178_);
lean_dec_ref(v_forwardState_3177_);
lean_dec_ref(v_mvars_3176_);
lean_dec(v_normalizationState_3175_);
lean_dec(v_preNormGoal_3174_);
lean_dec(v_depth_3170_);
lean_dec(v_origin_3169_);
lean_dec_ref(v_children_3168_);
lean_dec(v_a_3133_);
lean_dec_ref(v_traceOpt_3126_);
lean_dec(v_g_3125_);
return v___x_3201_;
}
}
}
}
else
{
lean_dec_ref(v_unsafeQueue_3182_);
lean_dec(v_lastExpandedInIteration_3180_);
lean_dec(v_addedInIteration_3179_);
lean_dec_ref(v_forwardRuleMatches_3178_);
lean_dec_ref(v_forwardState_3177_);
lean_dec_ref(v_mvars_3176_);
lean_dec(v_normalizationState_3175_);
lean_dec(v_preNormGoal_3174_);
lean_dec(v_depth_3170_);
lean_dec(v_origin_3169_);
lean_dec_ref(v_children_3168_);
lean_dec(v_a_3133_);
lean_dec_ref(v_traceOpt_3126_);
lean_dec(v_g_3125_);
return v___x_3188_;
}
}
v___jp_3137_:
{
lean_object* v___x_3142_; lean_object* v_elimGoal_3143_; lean_object* v___x_3144_; lean_object* v_failedRapps_3145_; lean_object* v___x_3146_; lean_object* v___x_3147_; uint8_t v___x_3148_; 
v___x_3142_ = lp_aesop_Aesop_treeImpl;
v_elimGoal_3143_ = lean_ctor_get(v___x_3142_, 1);
lean_inc_ref(v_elimGoal_3143_);
v___x_3144_ = lean_apply_1(v_elimGoal_3143_, v_g_3125_);
v_failedRapps_3145_ = lean_ctor_get(v___x_3144_, 13);
lean_inc_ref(v_failedRapps_3145_);
lean_dec_ref(v___x_3144_);
v___x_3146_ = lean_array_get_size(v_failedRapps_3145_);
v___x_3147_ = lean_unsigned_to_nat(0u);
v___x_3148_ = lean_nat_dec_eq(v___x_3146_, v___x_3147_);
if (v___x_3148_ == 0)
{
lean_object* v___x_3149_; lean_object* v___x_3150_; size_t v_sz_3151_; lean_object* v___x_3152_; lean_object* v___x_3153_; lean_object* v___f_3154_; lean_object* v___x_3155_; 
v___x_3149_ = lean_obj_once(&lp_aesop_Aesop_Goal_traceMetadata___closed__1, &lp_aesop_Aesop_Goal_traceMetadata___closed__1_once, _init_lp_aesop_Aesop_Goal_traceMetadata___closed__1);
v___x_3150_ = lean_box(0);
v_sz_3151_ = lean_array_size(v_failedRapps_3145_);
v___x_3152_ = lean_box_usize(v_sz_3151_);
v___x_3153_ = ((lean_object*)(lp_aesop_Aesop_Goal_traceMetadata___boxed__const__1));
lean_inc_ref(v_traceOpt_3126_);
v___f_3154_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Goal_traceMetadata___lam__0___boxed), 11, 6);
lean_closure_set(v___f_3154_, 0, v_a_3133_);
lean_closure_set(v___f_3154_, 1, v_traceOpt_3126_);
lean_closure_set(v___f_3154_, 2, v_failedRapps_3145_);
lean_closure_set(v___f_3154_, 3, v___x_3152_);
lean_closure_set(v___f_3154_, 4, v___x_3153_);
lean_closure_set(v___f_3154_, 5, v___x_3150_);
v___x_3155_ = lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Goal_traceMetadata_trcNode(v_traceOpt_3126_, v___x_3149_, v___f_3154_, v___y_3138_, v___y_3139_, v___y_3140_, v___y_3141_);
return v___x_3155_;
}
else
{
lean_object* v___x_3156_; lean_object* v___x_3157_; 
lean_dec_ref(v_failedRapps_3145_);
lean_dec(v_a_3133_);
v___x_3156_ = lean_obj_once(&lp_aesop_Aesop_Goal_traceMetadata___closed__3, &lp_aesop_Aesop_Goal_traceMetadata___closed__3_once, _init_lp_aesop_Aesop_Goal_traceMetadata___closed__3);
v___x_3157_ = lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Goal_traceMetadata_trc(v_traceOpt_3126_, v___x_3156_, v___y_3138_, v___y_3139_, v___y_3140_, v___y_3141_);
return v___x_3157_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_traceMetadata___boxed(lean_object* v_g_3447_, lean_object* v_traceOpt_3448_, lean_object* v_a_3449_, lean_object* v_a_3450_, lean_object* v_a_3451_, lean_object* v_a_3452_, lean_object* v_a_3453_){
_start:
{
lean_object* v_res_3454_; 
v_res_3454_ = lp_aesop_Aesop_Goal_traceMetadata(v_g_3447_, v_traceOpt_3448_, v_a_3449_, v_a_3450_, v_a_3451_, v_a_3452_);
lean_dec(v_a_3452_);
lean_dec_ref(v_a_3451_);
lean_dec(v_a_3450_);
lean_dec_ref(v_a_3449_);
return v_res_3454_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Aesop_Goal_traceMetadata_spec__1(lean_object* v_as_3455_, size_t v_sz_3456_, size_t v_i_3457_, lean_object* v_bs_3458_){
_start:
{
lean_object* v___x_3459_; 
v___x_3459_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Aesop_Goal_traceMetadata_spec__1___redArg(v_sz_3456_, v_i_3457_, v_bs_3458_);
return v___x_3459_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Aesop_Goal_traceMetadata_spec__1___boxed(lean_object* v_as_3460_, lean_object* v_sz_3461_, lean_object* v_i_3462_, lean_object* v_bs_3463_){
_start:
{
size_t v_sz_boxed_3464_; size_t v_i_boxed_3465_; lean_object* v_res_3466_; 
v_sz_boxed_3464_ = lean_unbox_usize(v_sz_3461_);
lean_dec(v_sz_3461_);
v_i_boxed_3465_ = lean_unbox_usize(v_i_3462_);
lean_dec(v_i_3462_);
v_res_3466_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Aesop_Goal_traceMetadata_spec__1(v_as_3460_, v_sz_boxed_3464_, v_i_boxed_3465_, v_bs_3463_);
lean_dec_ref(v_as_3460_);
return v_res_3466_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Aesop_Goal_traceMetadata_spec__3(lean_object* v_as_3467_, size_t v_sz_3468_, size_t v_i_3469_, lean_object* v_bs_3470_){
_start:
{
lean_object* v___x_3471_; 
v___x_3471_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Aesop_Goal_traceMetadata_spec__3___redArg(v_sz_3468_, v_i_3469_, v_bs_3470_);
return v___x_3471_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Aesop_Goal_traceMetadata_spec__3___boxed(lean_object* v_as_3472_, lean_object* v_sz_3473_, lean_object* v_i_3474_, lean_object* v_bs_3475_){
_start:
{
size_t v_sz_boxed_3476_; size_t v_i_boxed_3477_; lean_object* v_res_3478_; 
v_sz_boxed_3476_ = lean_unbox_usize(v_sz_3473_);
lean_dec(v_sz_3473_);
v_i_boxed_3477_ = lean_unbox_usize(v_i_3474_);
lean_dec(v_i_3474_);
v_res_3478_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Aesop_Goal_traceMetadata_spec__3(v_as_3472_, v_sz_boxed_3476_, v_i_boxed_3477_, v_bs_3475_);
lean_dec_ref(v_as_3472_);
return v_res_3478_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_Goal_traceMetadata_spec__5(lean_object* v_opt_3479_, lean_object* v___y_3480_, lean_object* v___y_3481_, lean_object* v___y_3482_, lean_object* v___y_3483_){
_start:
{
lean_object* v___x_3485_; 
v___x_3485_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_Goal_traceMetadata_spec__5___redArg(v_opt_3479_, v___y_3482_);
return v___x_3485_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_Goal_traceMetadata_spec__5___boxed(lean_object* v_opt_3486_, lean_object* v___y_3487_, lean_object* v___y_3488_, lean_object* v___y_3489_, lean_object* v___y_3490_, lean_object* v___y_3491_){
_start:
{
lean_object* v_res_3492_; 
v_res_3492_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_Goal_traceMetadata_spec__5(v_opt_3486_, v___y_3487_, v___y_3488_, v___y_3489_, v___y_3490_);
lean_dec(v___y_3490_);
lean_dec_ref(v___y_3489_);
lean_dec(v___y_3488_);
lean_dec_ref(v___y_3487_);
lean_dec_ref(v_opt_3486_);
return v_res_3492_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Aesop_Goal_traceMetadata_spec__9___redArg(lean_object* v_map_3493_, lean_object* v_f_3494_, lean_object* v_init_3495_){
_start:
{
lean_object* v___x_3496_; 
v___x_3496_ = lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_Goal_traceMetadata_spec__9_spec__18___redArg(v_f_3494_, v_map_3493_, v_init_3495_);
return v___x_3496_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Aesop_Goal_traceMetadata_spec__9___redArg___boxed(lean_object* v_map_3497_, lean_object* v_f_3498_, lean_object* v_init_3499_){
_start:
{
lean_object* v_res_3500_; 
v_res_3500_ = lp_aesop_Lean_PersistentHashMap_foldlM___at___00Aesop_Goal_traceMetadata_spec__9___redArg(v_map_3497_, v_f_3498_, v_init_3499_);
lean_dec_ref(v_map_3497_);
return v_res_3500_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Aesop_Goal_traceMetadata_spec__9(lean_object* v_00_u03c3_3501_, lean_object* v_00_u03b2_3502_, lean_object* v_map_3503_, lean_object* v_f_3504_, lean_object* v_init_3505_){
_start:
{
lean_object* v___x_3506_; 
v___x_3506_ = lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_Goal_traceMetadata_spec__9_spec__18___redArg(v_f_3504_, v_map_3503_, v_init_3505_);
return v___x_3506_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Aesop_Goal_traceMetadata_spec__9___boxed(lean_object* v_00_u03c3_3507_, lean_object* v_00_u03b2_3508_, lean_object* v_map_3509_, lean_object* v_f_3510_, lean_object* v_init_3511_){
_start:
{
lean_object* v_res_3512_; 
v_res_3512_ = lp_aesop_Lean_PersistentHashMap_foldlM___at___00Aesop_Goal_traceMetadata_spec__9(v_00_u03c3_3507_, v_00_u03b2_3508_, v_map_3509_, v_f_3510_, v_init_3511_);
lean_dec_ref(v_map_3509_);
return v_res_3512_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Goal_traceMetadata_spec__14(size_t v_sz_3513_, size_t v_i_3514_, lean_object* v_bs_3515_, lean_object* v___y_3516_, lean_object* v___y_3517_, lean_object* v___y_3518_, lean_object* v___y_3519_){
_start:
{
lean_object* v___x_3521_; 
v___x_3521_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Goal_traceMetadata_spec__14___redArg(v_sz_3513_, v_i_3514_, v_bs_3515_);
return v___x_3521_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Goal_traceMetadata_spec__14___boxed(lean_object* v_sz_3522_, lean_object* v_i_3523_, lean_object* v_bs_3524_, lean_object* v___y_3525_, lean_object* v___y_3526_, lean_object* v___y_3527_, lean_object* v___y_3528_, lean_object* v___y_3529_){
_start:
{
size_t v_sz_boxed_3530_; size_t v_i_boxed_3531_; lean_object* v_res_3532_; 
v_sz_boxed_3530_ = lean_unbox_usize(v_sz_3522_);
lean_dec(v_sz_3522_);
v_i_boxed_3531_ = lean_unbox_usize(v_i_3523_);
lean_dec(v_i_3523_);
v_res_3532_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Goal_traceMetadata_spec__14(v_sz_boxed_3530_, v_i_boxed_3531_, v_bs_3524_, v___y_3525_, v___y_3526_, v___y_3527_, v___y_3528_);
lean_dec(v___y_3528_);
lean_dec_ref(v___y_3527_);
lean_dec(v___y_3526_);
lean_dec_ref(v___y_3525_);
return v_res_3532_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldl___at___00Aesop_Goal_traceMetadata_spec__16(lean_object* v_00_u03c3_3533_, lean_object* v_00_u03b2_3534_, lean_object* v_map_3535_, lean_object* v_f_3536_, lean_object* v_init_3537_){
_start:
{
lean_object* v___x_3538_; 
v___x_3538_ = lp_aesop_Lean_PersistentHashMap_foldl___at___00Aesop_Goal_traceMetadata_spec__16___redArg(v_map_3535_, v_f_3536_, v_init_3537_);
return v___x_3538_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldl___at___00Aesop_Goal_traceMetadata_spec__16___boxed(lean_object* v_00_u03c3_3539_, lean_object* v_00_u03b2_3540_, lean_object* v_map_3541_, lean_object* v_f_3542_, lean_object* v_init_3543_){
_start:
{
lean_object* v_res_3544_; 
v_res_3544_ = lp_aesop_Lean_PersistentHashMap_foldl___at___00Aesop_Goal_traceMetadata_spec__16(v_00_u03c3_3539_, v_00_u03b2_3540_, v_map_3541_, v_f_3542_, v_init_3543_);
lean_dec_ref(v_map_3541_);
return v_res_3544_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_WFExtrinsicFix_0__WellFounded_opaqueFix_u2082___at___00Aesop_Goal_traceMetadata_spec__17(lean_object* v_inst_3545_, lean_object* v_R_3546_, lean_object* v_a_3547_, lean_object* v_b_3548_){
_start:
{
lean_object* v___x_3549_; 
v___x_3549_ = lp_aesop___private_Init_WFExtrinsicFix_0__WellFounded_opaqueFix_u2082___at___00Aesop_Goal_traceMetadata_spec__17___redArg(v_a_3547_, v_b_3548_);
return v___x_3549_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__9___redArg(lean_object* v_map_3550_, lean_object* v_f_3551_, lean_object* v_init_3552_){
_start:
{
lean_object* v___x_3553_; 
v___x_3553_ = lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_Goal_traceMetadata_spec__9_spec__18___redArg(v_f_3551_, v_map_3550_, v_init_3552_);
return v___x_3553_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__9___redArg___boxed(lean_object* v_map_3554_, lean_object* v_f_3555_, lean_object* v_init_3556_){
_start:
{
lean_object* v_res_3557_; 
v_res_3557_ = lp_aesop_Lean_PersistentHashMap_foldlM___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__9___redArg(v_map_3554_, v_f_3555_, v_init_3556_);
lean_dec_ref(v_map_3554_);
return v_res_3557_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__9(lean_object* v_00_u03c3_3558_, lean_object* v_00_u03b2_3559_, lean_object* v_map_3560_, lean_object* v_f_3561_, lean_object* v_init_3562_){
_start:
{
lean_object* v___x_3563_; 
v___x_3563_ = lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_Goal_traceMetadata_spec__9_spec__18___redArg(v_f_3561_, v_map_3560_, v_init_3562_);
return v___x_3563_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__9___boxed(lean_object* v_00_u03c3_3564_, lean_object* v_00_u03b2_3565_, lean_object* v_map_3566_, lean_object* v_f_3567_, lean_object* v_init_3568_){
_start:
{
lean_object* v_res_3569_; 
v_res_3569_ = lp_aesop_Lean_PersistentHashMap_foldlM___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__9(v_00_u03c3_3564_, v_00_u03b2_3565_, v_map_3566_, v_f_3567_, v_init_3568_);
lean_dec_ref(v_map_3566_);
return v_res_3569_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_map___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__13(lean_object* v_00_u03b2_3570_, lean_object* v_00_u03c3_3571_, lean_object* v_pm_3572_, lean_object* v_f_3573_){
_start:
{
lean_object* v___x_3574_; 
v___x_3574_ = lp_aesop_Lean_PersistentHashMap_map___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__13___redArg(v_pm_3572_, v_f_3573_);
return v___x_3574_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__15(lean_object* v_00_u03c3_3575_, lean_object* v_00_u03b2_3576_, lean_object* v_map_3577_, lean_object* v_init_3578_, lean_object* v_f_3579_){
_start:
{
lean_object* v___x_3580_; 
v___x_3580_ = lp_aesop_Lean_PersistentHashMap_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__15___redArg(v_map_3577_, v_init_3578_, v_f_3579_);
return v___x_3580_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__15___boxed(lean_object* v_00_u03c3_3581_, lean_object* v_00_u03b2_3582_, lean_object* v_map_3583_, lean_object* v_init_3584_, lean_object* v_f_3585_){
_start:
{
lean_object* v_res_3586_; 
v_res_3586_ = lp_aesop_Lean_PersistentHashMap_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__15(v_00_u03c3_3581_, v_00_u03b2_3582_, v_map_3583_, v_init_3584_, v_f_3585_);
lean_dec_ref(v_map_3583_);
return v_res_3586_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_Goal_traceMetadata_spec__9_spec__18(lean_object* v_00_u03c3_3587_, lean_object* v_00_u03b1_3588_, lean_object* v_00_u03b2_3589_, lean_object* v_f_3590_, lean_object* v_x_3591_, lean_object* v_x_3592_){
_start:
{
lean_object* v___x_3593_; 
v___x_3593_ = lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_Goal_traceMetadata_spec__9_spec__18___redArg(v_f_3590_, v_x_3591_, v_x_3592_);
return v___x_3593_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_Goal_traceMetadata_spec__9_spec__18___boxed(lean_object* v_00_u03c3_3594_, lean_object* v_00_u03b1_3595_, lean_object* v_00_u03b2_3596_, lean_object* v_f_3597_, lean_object* v_x_3598_, lean_object* v_x_3599_){
_start:
{
lean_object* v_res_3600_; 
v_res_3600_ = lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_Goal_traceMetadata_spec__9_spec__18(v_00_u03c3_3594_, v_00_u03b1_3595_, v_00_u03b2_3596_, v_f_3597_, v_x_3598_, v_x_3599_);
lean_dec_ref(v_x_3598_);
return v_res_3600_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Aesop_Goal_traceMetadata_spec__16_spec__26___redArg(lean_object* v_map_3601_, lean_object* v_f_3602_, lean_object* v_init_3603_){
_start:
{
lean_object* v___x_3604_; 
v___x_3604_ = lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_Goal_traceMetadata_spec__9_spec__18___redArg(v_f_3602_, v_map_3601_, v_init_3603_);
return v___x_3604_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Aesop_Goal_traceMetadata_spec__16_spec__26___redArg___boxed(lean_object* v_map_3605_, lean_object* v_f_3606_, lean_object* v_init_3607_){
_start:
{
lean_object* v_res_3608_; 
v_res_3608_ = lp_aesop_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Aesop_Goal_traceMetadata_spec__16_spec__26___redArg(v_map_3605_, v_f_3606_, v_init_3607_);
lean_dec_ref(v_map_3605_);
return v_res_3608_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Aesop_Goal_traceMetadata_spec__16_spec__26(lean_object* v_00_u03c3_3609_, lean_object* v_00_u03b2_3610_, lean_object* v_map_3611_, lean_object* v_f_3612_, lean_object* v_init_3613_){
_start:
{
lean_object* v___x_3614_; 
v___x_3614_ = lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_Goal_traceMetadata_spec__9_spec__18___redArg(v_f_3612_, v_map_3611_, v_init_3613_);
return v___x_3614_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Aesop_Goal_traceMetadata_spec__16_spec__26___boxed(lean_object* v_00_u03c3_3615_, lean_object* v_00_u03b2_3616_, lean_object* v_map_3617_, lean_object* v_f_3618_, lean_object* v_init_3619_){
_start:
{
lean_object* v_res_3620_; 
v_res_3620_ = lp_aesop_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Aesop_Goal_traceMetadata_spec__16_spec__26(v_00_u03c3_3615_, v_00_u03b2_3616_, v_map_3617_, v_f_3618_, v_init_3619_);
lean_dec_ref(v_map_3617_);
return v_res_3620_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_mapM___at___00Lean_PersistentHashMap_map___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__13_spec__16___redArg(lean_object* v_pm_3621_, lean_object* v_f_3622_){
_start:
{
lean_object* v___x_3623_; 
v___x_3623_ = lp_aesop_Lean_PersistentHashMap_mapMAux___at___00Lean_PersistentHashMap_mapM___at___00Lean_PersistentHashMap_map___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__13_spec__16_spec__32___redArg(v_f_3622_, v_pm_3621_);
return v___x_3623_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_mapM___at___00Lean_PersistentHashMap_map___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__13_spec__16(lean_object* v_00_u03b2_3624_, lean_object* v_00_u03c3_3625_, lean_object* v_pm_3626_, lean_object* v_f_3627_){
_start:
{
lean_object* v___x_3628_; 
v___x_3628_ = lp_aesop_Lean_PersistentHashMap_mapMAux___at___00Lean_PersistentHashMap_mapM___at___00Lean_PersistentHashMap_map___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__13_spec__16_spec__32___redArg(v_f_3627_, v_pm_3626_);
return v___x_3628_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__14_spec__18(lean_object* v_00_u03c3_3629_, lean_object* v_00_u03b2_3630_, lean_object* v_map_3631_, lean_object* v_init_3632_, lean_object* v_f_3633_){
_start:
{
lean_object* v___x_3634_; 
v___x_3634_ = lp_aesop_Lean_PersistentHashMap_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__14_spec__18___redArg(v_map_3631_, v_init_3632_, v_f_3633_);
return v___x_3634_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__14_spec__18___boxed(lean_object* v_00_u03c3_3635_, lean_object* v_00_u03b2_3636_, lean_object* v_map_3637_, lean_object* v_init_3638_, lean_object* v_f_3639_){
_start:
{
lean_object* v_res_3640_; 
v_res_3640_ = lp_aesop_Lean_PersistentHashMap_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__14_spec__18(v_00_u03c3_3635_, v_00_u03b2_3636_, v_map_3637_, v_init_3638_, v_f_3639_);
lean_dec_ref(v_map_3637_);
return v_res_3640_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__15_spec__20___redArg(lean_object* v_map_3641_, lean_object* v_f_3642_, lean_object* v_init_3643_){
_start:
{
lean_object* v___x_3644_; 
v___x_3644_ = lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__15_spec__20_spec__38___redArg(v_f_3642_, v_map_3641_, v_init_3643_);
return v___x_3644_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__15_spec__20(lean_object* v_00_u03c3_3645_, lean_object* v_00_u03c3_3646_, lean_object* v_00_u03b2_3647_, lean_object* v_map_3648_, lean_object* v_f_3649_, lean_object* v_init_3650_){
_start:
{
lean_object* v___x_3651_; 
v___x_3651_ = lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__15_spec__20_spec__38___redArg(v_f_3649_, v_map_3648_, v_init_3650_);
return v___x_3651_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_Goal_traceMetadata_spec__9_spec__18_spec__24(lean_object* v_00_u03b1_3652_, lean_object* v_00_u03b2_3653_, lean_object* v_00_u03c3_3654_, lean_object* v_f_3655_, lean_object* v_as_3656_, size_t v_i_3657_, size_t v_stop_3658_, lean_object* v_b_3659_){
_start:
{
lean_object* v___x_3660_; 
v___x_3660_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_Goal_traceMetadata_spec__9_spec__18_spec__24___redArg(v_f_3655_, v_as_3656_, v_i_3657_, v_stop_3658_, v_b_3659_);
return v___x_3660_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_Goal_traceMetadata_spec__9_spec__18_spec__24___boxed(lean_object* v_00_u03b1_3661_, lean_object* v_00_u03b2_3662_, lean_object* v_00_u03c3_3663_, lean_object* v_f_3664_, lean_object* v_as_3665_, lean_object* v_i_3666_, lean_object* v_stop_3667_, lean_object* v_b_3668_){
_start:
{
size_t v_i_boxed_3669_; size_t v_stop_boxed_3670_; lean_object* v_res_3671_; 
v_i_boxed_3669_ = lean_unbox_usize(v_i_3666_);
lean_dec(v_i_3666_);
v_stop_boxed_3670_ = lean_unbox_usize(v_stop_3667_);
lean_dec(v_stop_3667_);
v_res_3671_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_Goal_traceMetadata_spec__9_spec__18_spec__24(v_00_u03b1_3661_, v_00_u03b2_3662_, v_00_u03c3_3663_, v_f_3664_, v_as_3665_, v_i_boxed_3669_, v_stop_boxed_3670_, v_b_3668_);
lean_dec_ref(v_as_3665_);
return v_res_3671_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_Goal_traceMetadata_spec__9_spec__18_spec__25(lean_object* v_00_u03c3_3672_, lean_object* v_00_u03b1_3673_, lean_object* v_00_u03b2_3674_, lean_object* v_f_3675_, lean_object* v_keys_3676_, lean_object* v_vals_3677_, lean_object* v_heq_3678_, lean_object* v_i_3679_, lean_object* v_acc_3680_){
_start:
{
lean_object* v___x_3681_; 
v___x_3681_ = lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_Goal_traceMetadata_spec__9_spec__18_spec__25___redArg(v_f_3675_, v_keys_3676_, v_vals_3677_, v_i_3679_, v_acc_3680_);
return v___x_3681_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_Goal_traceMetadata_spec__9_spec__18_spec__25___boxed(lean_object* v_00_u03c3_3682_, lean_object* v_00_u03b1_3683_, lean_object* v_00_u03b2_3684_, lean_object* v_f_3685_, lean_object* v_keys_3686_, lean_object* v_vals_3687_, lean_object* v_heq_3688_, lean_object* v_i_3689_, lean_object* v_acc_3690_){
_start:
{
lean_object* v_res_3691_; 
v_res_3691_ = lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_Goal_traceMetadata_spec__9_spec__18_spec__25(v_00_u03c3_3682_, v_00_u03b1_3683_, v_00_u03b2_3684_, v_f_3685_, v_keys_3686_, v_vals_3687_, v_heq_3688_, v_i_3689_, v_acc_3690_);
lean_dec_ref(v_vals_3687_);
lean_dec_ref(v_keys_3686_);
return v_res_3691_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_toList___at___00Lean_PersistentHashSet_toList___at___00__private_Aesop_Forward_State_0__Aesop_ppPHashSet___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__10_spec__11_spec__23(lean_object* v_00_u03b2_3692_, lean_object* v_m_3693_){
_start:
{
lean_object* v___x_3694_; 
v___x_3694_ = lp_aesop_Lean_PersistentHashMap_toList___at___00Lean_PersistentHashSet_toList___at___00__private_Aesop_Forward_State_0__Aesop_ppPHashSet___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__10_spec__11_spec__23___redArg(v_m_3693_);
return v___x_3694_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_toList___at___00Lean_PersistentHashSet_toList___at___00__private_Aesop_Forward_State_0__Aesop_ppPHashSet___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__10_spec__11_spec__23___boxed(lean_object* v_00_u03b2_3695_, lean_object* v_m_3696_){
_start:
{
lean_object* v_res_3697_; 
v_res_3697_ = lp_aesop_Lean_PersistentHashMap_toList___at___00Lean_PersistentHashSet_toList___at___00__private_Aesop_Forward_State_0__Aesop_ppPHashSet___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__10_spec__11_spec__23(v_00_u03b2_3695_, v_m_3696_);
lean_dec_ref(v_m_3696_);
return v_res_3697_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_mapMAux___at___00Lean_PersistentHashMap_mapM___at___00Lean_PersistentHashMap_map___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__13_spec__16_spec__32(lean_object* v_00_u03b1_3698_, lean_object* v_00_u03b2_3699_, lean_object* v_00_u03c3_3700_, lean_object* v_f_3701_, lean_object* v_n_3702_){
_start:
{
lean_object* v___x_3703_; 
v___x_3703_ = lp_aesop_Lean_PersistentHashMap_mapMAux___at___00Lean_PersistentHashMap_mapM___at___00Lean_PersistentHashMap_map___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__13_spec__16_spec__32___redArg(v_f_3701_, v_n_3702_);
return v___x_3703_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__14_spec__18_spec__35___redArg(lean_object* v_map_3704_, lean_object* v_f_3705_, lean_object* v_init_3706_){
_start:
{
lean_object* v___x_3707_; 
v___x_3707_ = lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__15_spec__20_spec__38___redArg(v_f_3705_, v_map_3704_, v_init_3706_);
return v___x_3707_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__14_spec__18_spec__35(lean_object* v_00_u03c3_3708_, lean_object* v_00_u03c3_3709_, lean_object* v_00_u03b2_3710_, lean_object* v_map_3711_, lean_object* v_f_3712_, lean_object* v_init_3713_){
_start:
{
lean_object* v___x_3714_; 
v___x_3714_ = lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__15_spec__20_spec__38___redArg(v_f_3712_, v_map_3711_, v_init_3713_);
return v___x_3714_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__15_spec__20_spec__38(lean_object* v_00_u03c3_3715_, lean_object* v_00_u03c3_3716_, lean_object* v_00_u03b1_3717_, lean_object* v_00_u03b2_3718_, lean_object* v_f_3719_, lean_object* v_x_3720_, lean_object* v_x_3721_){
_start:
{
lean_object* v___x_3722_; 
v___x_3722_ = lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__15_spec__20_spec__38___redArg(v_f_3719_, v_x_3720_, v_x_3721_);
return v___x_3722_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PersistentHashSet_toList___at___00__private_Aesop_Forward_State_0__Aesop_ppPHashSet___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__10_spec__11_spec__23_spec__31(lean_object* v_00_u03c3_3723_, lean_object* v_00_u03b2_3724_, lean_object* v_map_3725_, lean_object* v_f_3726_, lean_object* v_init_3727_){
_start:
{
lean_object* v___x_3728_; 
v___x_3728_ = lp_aesop_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PersistentHashSet_toList___at___00__private_Aesop_Forward_State_0__Aesop_ppPHashSet___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__10_spec__11_spec__23_spec__31___redArg(v_map_3725_, v_f_3726_, v_init_3727_);
return v___x_3728_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PersistentHashSet_toList___at___00__private_Aesop_Forward_State_0__Aesop_ppPHashSet___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__10_spec__11_spec__23_spec__31___boxed(lean_object* v_00_u03c3_3729_, lean_object* v_00_u03b2_3730_, lean_object* v_map_3731_, lean_object* v_f_3732_, lean_object* v_init_3733_){
_start:
{
lean_object* v_res_3734_; 
v_res_3734_ = lp_aesop_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PersistentHashSet_toList___at___00__private_Aesop_Forward_State_0__Aesop_ppPHashSet___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__10_spec__11_spec__23_spec__31(v_00_u03c3_3729_, v_00_u03b2_3730_, v_map_3731_, v_f_3732_, v_init_3733_);
lean_dec_ref(v_map_3731_);
return v_res_3734_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_PersistentHashMap_mapMAux___at___00Lean_PersistentHashMap_mapM___at___00Lean_PersistentHashMap_map___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__13_spec__16_spec__32_spec__41(lean_object* v_00_u03b1_3735_, lean_object* v_00_u03b2_3736_, lean_object* v_00_u03c3_3737_, lean_object* v_f_3738_, size_t v_sz_3739_, size_t v_i_3740_, lean_object* v_bs_3741_){
_start:
{
lean_object* v___x_3742_; 
v___x_3742_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_PersistentHashMap_mapMAux___at___00Lean_PersistentHashMap_mapM___at___00Lean_PersistentHashMap_map___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__13_spec__16_spec__32_spec__41___redArg(v_f_3738_, v_sz_3739_, v_i_3740_, v_bs_3741_);
return v___x_3742_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_PersistentHashMap_mapMAux___at___00Lean_PersistentHashMap_mapM___at___00Lean_PersistentHashMap_map___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__13_spec__16_spec__32_spec__41___boxed(lean_object* v_00_u03b1_3743_, lean_object* v_00_u03b2_3744_, lean_object* v_00_u03c3_3745_, lean_object* v_f_3746_, lean_object* v_sz_3747_, lean_object* v_i_3748_, lean_object* v_bs_3749_){
_start:
{
size_t v_sz_boxed_3750_; size_t v_i_boxed_3751_; lean_object* v_res_3752_; 
v_sz_boxed_3750_ = lean_unbox_usize(v_sz_3747_);
lean_dec(v_sz_3747_);
v_i_boxed_3751_ = lean_unbox_usize(v_i_3748_);
lean_dec(v_i_3748_);
v_res_3752_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_PersistentHashMap_mapMAux___at___00Lean_PersistentHashMap_mapM___at___00Lean_PersistentHashMap_map___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__13_spec__16_spec__32_spec__41(v_00_u03b1_3743_, v_00_u03b2_3744_, v_00_u03c3_3745_, v_f_3746_, v_sz_boxed_3750_, v_i_boxed_3751_, v_bs_3749_);
return v_res_3752_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_mapM_x27___at___00Lean_PersistentHashMap_mapMAux___at___00Lean_PersistentHashMap_mapM___at___00Lean_PersistentHashMap_map___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__13_spec__16_spec__32_spec__42(lean_object* v_00_u03b1_3753_, lean_object* v_00_u03b2_3754_, lean_object* v_f_3755_, lean_object* v_as_3756_){
_start:
{
lean_object* v___x_3757_; 
v___x_3757_ = lp_aesop_Array_mapM_x27___at___00Lean_PersistentHashMap_mapMAux___at___00Lean_PersistentHashMap_mapM___at___00Lean_PersistentHashMap_map___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__13_spec__16_spec__32_spec__42___redArg(v_f_3755_, v_as_3756_);
return v___x_3757_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_mapM_x27___at___00Lean_PersistentHashMap_mapMAux___at___00Lean_PersistentHashMap_mapM___at___00Lean_PersistentHashMap_map___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__13_spec__16_spec__32_spec__42___boxed(lean_object* v_00_u03b1_3758_, lean_object* v_00_u03b2_3759_, lean_object* v_f_3760_, lean_object* v_as_3761_){
_start:
{
lean_object* v_res_3762_; 
v_res_3762_ = lp_aesop_Array_mapM_x27___at___00Lean_PersistentHashMap_mapMAux___at___00Lean_PersistentHashMap_mapM___at___00Lean_PersistentHashMap_map___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__13_spec__16_spec__32_spec__42(v_00_u03b1_3758_, v_00_u03b2_3759_, v_f_3760_, v_as_3761_);
lean_dec_ref(v_as_3761_);
return v_res_3762_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__15_spec__20_spec__38_spec__47(lean_object* v_00_u03b1_3763_, lean_object* v_00_u03b2_3764_, lean_object* v_00_u03c3_3765_, lean_object* v_00_u03c3_3766_, lean_object* v_f_3767_, lean_object* v_as_3768_, size_t v_i_3769_, size_t v_stop_3770_, lean_object* v_b_3771_){
_start:
{
lean_object* v___x_3772_; 
v___x_3772_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__15_spec__20_spec__38_spec__47___redArg(v_f_3767_, v_as_3768_, v_i_3769_, v_stop_3770_, v_b_3771_);
return v___x_3772_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__15_spec__20_spec__38_spec__47___boxed(lean_object* v_00_u03b1_3773_, lean_object* v_00_u03b2_3774_, lean_object* v_00_u03c3_3775_, lean_object* v_00_u03c3_3776_, lean_object* v_f_3777_, lean_object* v_as_3778_, lean_object* v_i_3779_, lean_object* v_stop_3780_, lean_object* v_b_3781_){
_start:
{
size_t v_i_boxed_3782_; size_t v_stop_boxed_3783_; lean_object* v_res_3784_; 
v_i_boxed_3782_ = lean_unbox_usize(v_i_3779_);
lean_dec(v_i_3779_);
v_stop_boxed_3783_ = lean_unbox_usize(v_stop_3780_);
lean_dec(v_stop_3780_);
v_res_3784_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__15_spec__20_spec__38_spec__47(v_00_u03b1_3773_, v_00_u03b2_3774_, v_00_u03c3_3775_, v_00_u03c3_3776_, v_f_3777_, v_as_3778_, v_i_boxed_3782_, v_stop_boxed_3783_, v_b_3781_);
lean_dec_ref(v_as_3778_);
return v_res_3784_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__15_spec__20_spec__38_spec__48(lean_object* v_00_u03c3_3785_, lean_object* v_00_u03c3_3786_, lean_object* v_00_u03b1_3787_, lean_object* v_00_u03b2_3788_, lean_object* v_f_3789_, lean_object* v_keys_3790_, lean_object* v_vals_3791_, lean_object* v_heq_3792_, lean_object* v_i_3793_, lean_object* v_acc_3794_){
_start:
{
lean_object* v___x_3795_; 
v___x_3795_ = lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__15_spec__20_spec__38_spec__48___redArg(v_f_3789_, v_keys_3790_, v_vals_3791_, v_i_3793_, v_acc_3794_);
return v___x_3795_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__15_spec__20_spec__38_spec__48___boxed(lean_object* v_00_u03c3_3796_, lean_object* v_00_u03c3_3797_, lean_object* v_00_u03b1_3798_, lean_object* v_00_u03b2_3799_, lean_object* v_f_3800_, lean_object* v_keys_3801_, lean_object* v_vals_3802_, lean_object* v_heq_3803_, lean_object* v_i_3804_, lean_object* v_acc_3805_){
_start:
{
lean_object* v_res_3806_; 
v_res_3806_ = lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__15_spec__20_spec__38_spec__48(v_00_u03c3_3796_, v_00_u03c3_3797_, v_00_u03b1_3798_, v_00_u03b2_3799_, v_f_3800_, v_keys_3801_, v_vals_3802_, v_heq_3803_, v_i_3804_, v_acc_3805_);
lean_dec_ref(v_vals_3802_);
lean_dec_ref(v_keys_3801_);
return v_res_3806_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_BasicAux_0__Array_mapM_x27_go___at___00Array_mapM_x27___at___00Lean_PersistentHashMap_mapMAux___at___00Lean_PersistentHashMap_mapM___at___00Lean_PersistentHashMap_map___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__13_spec__16_spec__32_spec__42_spec__48(lean_object* v_00_u03b1_3807_, lean_object* v_00_u03b2_3808_, lean_object* v_f_3809_, lean_object* v_as_3810_, lean_object* v_i_3811_, lean_object* v_acc_3812_, lean_object* v_hle_3813_){
_start:
{
lean_object* v___x_3814_; 
v___x_3814_ = lp_aesop___private_Init_Data_Array_BasicAux_0__Array_mapM_x27_go___at___00Array_mapM_x27___at___00Lean_PersistentHashMap_mapMAux___at___00Lean_PersistentHashMap_mapM___at___00Lean_PersistentHashMap_map___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__13_spec__16_spec__32_spec__42_spec__48___redArg(v_f_3809_, v_as_3810_, v_i_3811_, v_acc_3812_);
return v___x_3814_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_BasicAux_0__Array_mapM_x27_go___at___00Array_mapM_x27___at___00Lean_PersistentHashMap_mapMAux___at___00Lean_PersistentHashMap_mapM___at___00Lean_PersistentHashMap_map___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__13_spec__16_spec__32_spec__42_spec__48___boxed(lean_object* v_00_u03b1_3815_, lean_object* v_00_u03b2_3816_, lean_object* v_f_3817_, lean_object* v_as_3818_, lean_object* v_i_3819_, lean_object* v_acc_3820_, lean_object* v_hle_3821_){
_start:
{
lean_object* v_res_3822_; 
v_res_3822_ = lp_aesop___private_Init_Data_Array_BasicAux_0__Array_mapM_x27_go___at___00Array_mapM_x27___at___00Lean_PersistentHashMap_mapMAux___at___00Lean_PersistentHashMap_mapM___at___00Lean_PersistentHashMap_map___at___00__private_Aesop_Forward_State_0__Aesop_ppMap___at___00Aesop_Goal_traceMetadata_spec__7_spec__13_spec__16_spec__32_spec__42_spec__48(v_00_u03b1_3815_, v_00_u03b2_3816_, v_f_3817_, v_as_3818_, v_i_3819_, v_acc_3820_, v_hle_3821_);
lean_dec_ref(v_as_3818_);
return v_res_3822_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_withHeadlineTraceNode___redArg___lam__0(lean_object* v_transform_3823_, lean_object* v___x_3824_, lean_object* v_x_3825_, lean_object* v___y_3826_, lean_object* v___y_3827_, lean_object* v___y_3828_, lean_object* v___y_3829_){
_start:
{
lean_object* v___x_3831_; 
lean_inc(v___y_3829_);
lean_inc_ref(v___y_3828_);
lean_inc(v___y_3827_);
lean_inc_ref(v___y_3826_);
v___x_3831_ = lean_apply_6(v_transform_3823_, v___x_3824_, v___y_3826_, v___y_3827_, v___y_3828_, v___y_3829_, lean_box(0));
return v___x_3831_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_withHeadlineTraceNode___redArg___lam__0___boxed(lean_object* v_transform_3832_, lean_object* v___x_3833_, lean_object* v_x_3834_, lean_object* v___y_3835_, lean_object* v___y_3836_, lean_object* v___y_3837_, lean_object* v___y_3838_, lean_object* v___y_3839_){
_start:
{
lean_object* v_res_3840_; 
v_res_3840_ = lp_aesop_Aesop_Rapp_withHeadlineTraceNode___redArg___lam__0(v_transform_3832_, v___x_3833_, v_x_3834_, v___y_3835_, v___y_3836_, v___y_3837_, v___y_3838_);
lean_dec(v___y_3838_);
lean_dec_ref(v___y_3837_);
lean_dec(v___y_3836_);
lean_dec_ref(v___y_3835_);
lean_dec_ref(v_x_3834_);
return v_res_3840_;
}
}
static lean_object* _init_lp_aesop_Aesop_Rapp_withHeadlineTraceNode___redArg___closed__1(void){
_start:
{
lean_object* v___x_3842_; lean_object* v___x_3843_; 
v___x_3842_ = ((lean_object*)(lp_aesop_Aesop_Rapp_withHeadlineTraceNode___redArg___closed__0));
v___x_3843_ = l_Lean_stringToMessageData(v___x_3842_);
return v___x_3843_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_withHeadlineTraceNode___redArg(lean_object* v_r_3844_, lean_object* v_traceOpt_3845_, lean_object* v_k_3846_, uint8_t v_collapsed_3847_, lean_object* v_transform_3848_, lean_object* v_a_3849_, lean_object* v_a_3850_, lean_object* v_a_3851_, lean_object* v_a_3852_){
_start:
{
lean_object* v___y_3855_; lean_object* v___y_3856_; lean_object* v___y_3857_; lean_object* v___y_3858_; lean_object* v___y_3859_; lean_object* v___y_3860_; uint8_t v___y_3861_; lean_object* v_a_3862_; lean_object* v___y_3872_; lean_object* v___y_3873_; lean_object* v___y_3874_; lean_object* v___y_3875_; lean_object* v___y_3876_; lean_object* v___y_3877_; uint8_t v___y_3878_; lean_object* v_a_3879_; lean_object* v___y_3892_; lean_object* v___y_3893_; lean_object* v___y_3894_; lean_object* v___y_3895_; uint8_t v___y_3896_; lean_object* v___x_3937_; lean_object* v_elimRapp_3938_; lean_object* v___x_3939_; lean_object* v_id_3940_; uint8_t v_state_3941_; lean_object* v_appliedRule_3942_; double v_successProbability_3943_; lean_object* v___x_3944_; uint8_t v_phase_3945_; lean_object* v___x_3946_; lean_object* v___x_3947_; lean_object* v___x_3948_; lean_object* v___x_3949_; lean_object* v___x_3950_; lean_object* v___x_3951_; lean_object* v___x_3952_; lean_object* v___x_3953_; lean_object* v___x_3954_; lean_object* v___x_3955_; lean_object* v___x_3956_; lean_object* v___x_3957_; lean_object* v___x_3958_; lean_object* v___x_3959_; lean_object* v___x_3960_; lean_object* v___y_3962_; lean_object* v___y_3963_; lean_object* v___y_3964_; lean_object* v___y_3994_; lean_object* v___y_3995_; lean_object* v___y_3996_; lean_object* v___y_4003_; 
v___x_3937_ = lp_aesop_Aesop_treeImpl;
v_elimRapp_3938_ = lean_ctor_get(v___x_3937_, 3);
lean_inc_ref(v_elimRapp_3938_);
v___x_3939_ = lean_apply_1(v_elimRapp_3938_, v_r_3844_);
v_id_3940_ = lean_ctor_get(v___x_3939_, 0);
lean_inc(v_id_3940_);
v_state_3941_ = lean_ctor_get_uint8(v___x_3939_, sizeof(void*)*9 + 8);
v_appliedRule_3942_ = lean_ctor_get(v___x_3939_, 3);
lean_inc_ref(v_appliedRule_3942_);
v_successProbability_3943_ = lean_ctor_get_float(v___x_3939_, sizeof(void*)*9);
lean_dec_ref(v___x_3939_);
v___x_3944_ = lp_aesop_Aesop_RegularRule_name(v_appliedRule_3942_);
lean_dec_ref(v_appliedRule_3942_);
v_phase_3945_ = lean_ctor_get_uint8(v___x_3944_, sizeof(void*)*1 + 9);
v___x_3946_ = lp_aesop_Aesop_NodeState_toEmoji(v_state_3941_);
v___x_3947_ = l_Lean_stringToMessageData(v___x_3946_);
v___x_3948_ = lean_obj_once(&lp_aesop_Aesop_Rapp_withHeadlineTraceNode___redArg___closed__1, &lp_aesop_Aesop_Rapp_withHeadlineTraceNode___redArg___closed__1_once, _init_lp_aesop_Aesop_Rapp_withHeadlineTraceNode___redArg___closed__1);
v___x_3949_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3949_, 0, v___x_3947_);
lean_ctor_set(v___x_3949_, 1, v___x_3948_);
v___x_3950_ = l_Nat_reprFast(v_id_3940_);
v___x_3951_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_3951_, 0, v___x_3950_);
v___x_3952_ = l_Lean_MessageData_ofFormat(v___x_3951_);
v___x_3953_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3953_, 0, v___x_3949_);
lean_ctor_set(v___x_3953_, 1, v___x_3952_);
v___x_3954_ = lean_obj_once(&lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt___lam__0___closed__3, &lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt___lam__0___closed__3_once, _init_lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt___lam__0___closed__3);
v___x_3955_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3955_, 0, v___x_3953_);
lean_ctor_set(v___x_3955_, 1, v___x_3954_);
v___x_3956_ = lp_aesop_Aesop_Percent_toHumanString(v_successProbability_3943_);
v___x_3957_ = l_Lean_stringToMessageData(v___x_3956_);
v___x_3958_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3958_, 0, v___x_3955_);
lean_ctor_set(v___x_3958_, 1, v___x_3957_);
v___x_3959_ = lean_obj_once(&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__3, &lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__3_once, _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__3);
v___x_3960_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3960_, 0, v___x_3958_);
lean_ctor_set(v___x_3960_, 1, v___x_3959_);
switch(v_phase_3945_)
{
case 0:
{
lean_object* v___x_4015_; 
v___x_4015_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__15));
v___y_4003_ = v___x_4015_;
goto v___jp_4002_;
}
case 1:
{
lean_object* v___x_4016_; 
v___x_4016_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__16));
v___y_4003_ = v___x_4016_;
goto v___jp_4002_;
}
default: 
{
lean_object* v___x_4017_; 
v___x_4017_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__17));
v___y_4003_ = v___x_4017_;
goto v___jp_4002_;
}
}
v___jp_3854_:
{
lean_object* v___x_3863_; double v___x_3864_; double v___x_3865_; lean_object* v___x_3866_; lean_object* v___x_3867_; lean_object* v___x_3868_; lean_object* v___x_3869_; lean_object* v___x_3870_; 
v___x_3863_ = lean_io_get_num_heartbeats();
v___x_3864_ = lean_float_of_nat(v___y_3858_);
v___x_3865_ = lean_float_of_nat(v___x_3863_);
v___x_3866_ = lean_box_float(v___x_3864_);
v___x_3867_ = lean_box_float(v___x_3865_);
v___x_3868_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3868_, 0, v___x_3866_);
lean_ctor_set(v___x_3868_, 1, v___x_3867_);
v___x_3869_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3869_, 0, v_a_3862_);
lean_ctor_set(v___x_3869_, 1, v___x_3868_);
lean_inc_ref(v___y_3860_);
v___x_3870_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Goal_withHeadlineTraceNode_spec__2___redArg(v___y_3856_, v_collapsed_3847_, v___y_3860_, v___y_3857_, v___y_3861_, v___y_3859_, v___y_3855_, v___x_3869_, v_a_3849_, v_a_3850_, v_a_3851_, v_a_3852_);
return v___x_3870_;
}
v___jp_3871_:
{
lean_object* v___x_3880_; double v___x_3881_; double v___x_3882_; double v___x_3883_; double v___x_3884_; double v___x_3885_; lean_object* v___x_3886_; lean_object* v___x_3887_; lean_object* v___x_3888_; lean_object* v___x_3889_; lean_object* v___x_3890_; 
v___x_3880_ = lean_io_mono_nanos_now();
v___x_3881_ = lean_float_of_nat(v___y_3877_);
v___x_3882_ = lean_float_once(&lp_aesop_Aesop_Goal_withHeadlineTraceNode___redArg___closed__3, &lp_aesop_Aesop_Goal_withHeadlineTraceNode___redArg___closed__3_once, _init_lp_aesop_Aesop_Goal_withHeadlineTraceNode___redArg___closed__3);
v___x_3883_ = lean_float_div(v___x_3881_, v___x_3882_);
v___x_3884_ = lean_float_of_nat(v___x_3880_);
v___x_3885_ = lean_float_div(v___x_3884_, v___x_3882_);
v___x_3886_ = lean_box_float(v___x_3883_);
v___x_3887_ = lean_box_float(v___x_3885_);
v___x_3888_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3888_, 0, v___x_3886_);
lean_ctor_set(v___x_3888_, 1, v___x_3887_);
v___x_3889_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3889_, 0, v_a_3879_);
lean_ctor_set(v___x_3889_, 1, v___x_3888_);
lean_inc_ref(v___y_3876_);
v___x_3890_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Goal_withHeadlineTraceNode_spec__2___redArg(v___y_3873_, v_collapsed_3847_, v___y_3876_, v___y_3874_, v___y_3878_, v___y_3875_, v___y_3872_, v___x_3889_, v_a_3849_, v_a_3850_, v_a_3851_, v_a_3852_);
return v___x_3890_;
}
v___jp_3891_:
{
lean_object* v___x_3897_; lean_object* v_a_3898_; lean_object* v___x_3899_; uint8_t v___x_3900_; 
v___x_3897_ = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_Goal_withHeadlineTraceNode_spec__0___redArg(v_a_3852_);
v_a_3898_ = lean_ctor_get(v___x_3897_, 0);
lean_inc(v_a_3898_);
lean_dec_ref(v___x_3897_);
v___x_3899_ = l_Lean_trace_profiler_useHeartbeats;
v___x_3900_ = lp_aesop_Lean_Option_get___at___00Aesop_Goal_withHeadlineTraceNode_spec__1(v___y_3894_, v___x_3899_);
if (v___x_3900_ == 0)
{
lean_object* v___x_3901_; lean_object* v___x_3902_; 
v___x_3901_ = lean_io_mono_nanos_now();
lean_inc(v_a_3852_);
lean_inc_ref(v_a_3851_);
lean_inc(v_a_3850_);
lean_inc_ref(v_a_3849_);
v___x_3902_ = lean_apply_5(v_k_3846_, v_a_3849_, v_a_3850_, v_a_3851_, v_a_3852_, lean_box(0));
if (lean_obj_tag(v___x_3902_) == 0)
{
lean_object* v_a_3903_; lean_object* v___x_3905_; uint8_t v_isShared_3906_; uint8_t v_isSharedCheck_3910_; 
v_a_3903_ = lean_ctor_get(v___x_3902_, 0);
v_isSharedCheck_3910_ = !lean_is_exclusive(v___x_3902_);
if (v_isSharedCheck_3910_ == 0)
{
v___x_3905_ = v___x_3902_;
v_isShared_3906_ = v_isSharedCheck_3910_;
goto v_resetjp_3904_;
}
else
{
lean_inc(v_a_3903_);
lean_dec(v___x_3902_);
v___x_3905_ = lean_box(0);
v_isShared_3906_ = v_isSharedCheck_3910_;
goto v_resetjp_3904_;
}
v_resetjp_3904_:
{
lean_object* v___x_3908_; 
if (v_isShared_3906_ == 0)
{
lean_ctor_set_tag(v___x_3905_, 1);
v___x_3908_ = v___x_3905_;
goto v_reusejp_3907_;
}
else
{
lean_object* v_reuseFailAlloc_3909_; 
v_reuseFailAlloc_3909_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3909_, 0, v_a_3903_);
v___x_3908_ = v_reuseFailAlloc_3909_;
goto v_reusejp_3907_;
}
v_reusejp_3907_:
{
v___y_3872_ = v___y_3892_;
v___y_3873_ = v___y_3893_;
v___y_3874_ = v___y_3894_;
v___y_3875_ = v_a_3898_;
v___y_3876_ = v___y_3895_;
v___y_3877_ = v___x_3901_;
v___y_3878_ = v___y_3896_;
v_a_3879_ = v___x_3908_;
goto v___jp_3871_;
}
}
}
else
{
lean_object* v_a_3911_; lean_object* v___x_3913_; uint8_t v_isShared_3914_; uint8_t v_isSharedCheck_3918_; 
v_a_3911_ = lean_ctor_get(v___x_3902_, 0);
v_isSharedCheck_3918_ = !lean_is_exclusive(v___x_3902_);
if (v_isSharedCheck_3918_ == 0)
{
v___x_3913_ = v___x_3902_;
v_isShared_3914_ = v_isSharedCheck_3918_;
goto v_resetjp_3912_;
}
else
{
lean_inc(v_a_3911_);
lean_dec(v___x_3902_);
v___x_3913_ = lean_box(0);
v_isShared_3914_ = v_isSharedCheck_3918_;
goto v_resetjp_3912_;
}
v_resetjp_3912_:
{
lean_object* v___x_3916_; 
if (v_isShared_3914_ == 0)
{
lean_ctor_set_tag(v___x_3913_, 0);
v___x_3916_ = v___x_3913_;
goto v_reusejp_3915_;
}
else
{
lean_object* v_reuseFailAlloc_3917_; 
v_reuseFailAlloc_3917_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3917_, 0, v_a_3911_);
v___x_3916_ = v_reuseFailAlloc_3917_;
goto v_reusejp_3915_;
}
v_reusejp_3915_:
{
v___y_3872_ = v___y_3892_;
v___y_3873_ = v___y_3893_;
v___y_3874_ = v___y_3894_;
v___y_3875_ = v_a_3898_;
v___y_3876_ = v___y_3895_;
v___y_3877_ = v___x_3901_;
v___y_3878_ = v___y_3896_;
v_a_3879_ = v___x_3916_;
goto v___jp_3871_;
}
}
}
}
else
{
lean_object* v___x_3919_; lean_object* v___x_3920_; 
v___x_3919_ = lean_io_get_num_heartbeats();
lean_inc(v_a_3852_);
lean_inc_ref(v_a_3851_);
lean_inc(v_a_3850_);
lean_inc_ref(v_a_3849_);
v___x_3920_ = lean_apply_5(v_k_3846_, v_a_3849_, v_a_3850_, v_a_3851_, v_a_3852_, lean_box(0));
if (lean_obj_tag(v___x_3920_) == 0)
{
lean_object* v_a_3921_; lean_object* v___x_3923_; uint8_t v_isShared_3924_; uint8_t v_isSharedCheck_3928_; 
v_a_3921_ = lean_ctor_get(v___x_3920_, 0);
v_isSharedCheck_3928_ = !lean_is_exclusive(v___x_3920_);
if (v_isSharedCheck_3928_ == 0)
{
v___x_3923_ = v___x_3920_;
v_isShared_3924_ = v_isSharedCheck_3928_;
goto v_resetjp_3922_;
}
else
{
lean_inc(v_a_3921_);
lean_dec(v___x_3920_);
v___x_3923_ = lean_box(0);
v_isShared_3924_ = v_isSharedCheck_3928_;
goto v_resetjp_3922_;
}
v_resetjp_3922_:
{
lean_object* v___x_3926_; 
if (v_isShared_3924_ == 0)
{
lean_ctor_set_tag(v___x_3923_, 1);
v___x_3926_ = v___x_3923_;
goto v_reusejp_3925_;
}
else
{
lean_object* v_reuseFailAlloc_3927_; 
v_reuseFailAlloc_3927_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3927_, 0, v_a_3921_);
v___x_3926_ = v_reuseFailAlloc_3927_;
goto v_reusejp_3925_;
}
v_reusejp_3925_:
{
v___y_3855_ = v___y_3892_;
v___y_3856_ = v___y_3893_;
v___y_3857_ = v___y_3894_;
v___y_3858_ = v___x_3919_;
v___y_3859_ = v_a_3898_;
v___y_3860_ = v___y_3895_;
v___y_3861_ = v___y_3896_;
v_a_3862_ = v___x_3926_;
goto v___jp_3854_;
}
}
}
else
{
lean_object* v_a_3929_; lean_object* v___x_3931_; uint8_t v_isShared_3932_; uint8_t v_isSharedCheck_3936_; 
v_a_3929_ = lean_ctor_get(v___x_3920_, 0);
v_isSharedCheck_3936_ = !lean_is_exclusive(v___x_3920_);
if (v_isSharedCheck_3936_ == 0)
{
v___x_3931_ = v___x_3920_;
v_isShared_3932_ = v_isSharedCheck_3936_;
goto v_resetjp_3930_;
}
else
{
lean_inc(v_a_3929_);
lean_dec(v___x_3920_);
v___x_3931_ = lean_box(0);
v_isShared_3932_ = v_isSharedCheck_3936_;
goto v_resetjp_3930_;
}
v_resetjp_3930_:
{
lean_object* v___x_3934_; 
if (v_isShared_3932_ == 0)
{
lean_ctor_set_tag(v___x_3931_, 0);
v___x_3934_ = v___x_3931_;
goto v_reusejp_3933_;
}
else
{
lean_object* v_reuseFailAlloc_3935_; 
v_reuseFailAlloc_3935_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3935_, 0, v_a_3929_);
v___x_3934_ = v_reuseFailAlloc_3935_;
goto v_reusejp_3933_;
}
v_reusejp_3933_:
{
v___y_3855_ = v___y_3892_;
v___y_3856_ = v___y_3893_;
v___y_3857_ = v___y_3894_;
v___y_3858_ = v___x_3919_;
v___y_3859_ = v_a_3898_;
v___y_3860_ = v___y_3895_;
v___y_3861_ = v___y_3896_;
v_a_3862_ = v___x_3934_;
goto v___jp_3854_;
}
}
}
}
}
v___jp_3961_:
{
lean_object* v_options_3965_; uint8_t v_hasTrace_3966_; 
v_options_3965_ = lean_ctor_get(v_a_3851_, 2);
v_hasTrace_3966_ = lean_ctor_get_uint8(v_options_3965_, sizeof(void*)*1);
if (v_hasTrace_3966_ == 0)
{
lean_object* v___x_3967_; 
lean_dec_ref(v___y_3963_);
lean_dec_ref_known(v___x_3960_, 2);
lean_dec_ref(v___x_3944_);
lean_dec_ref(v_transform_3848_);
lean_dec_ref(v_traceOpt_3845_);
lean_inc(v_a_3852_);
lean_inc_ref(v_a_3851_);
lean_inc(v_a_3850_);
lean_inc_ref(v_a_3849_);
v___x_3967_ = lean_apply_5(v_k_3846_, v_a_3849_, v_a_3850_, v_a_3851_, v_a_3852_, lean_box(0));
return v___x_3967_;
}
else
{
lean_object* v_inheritedTraceOptions_3968_; lean_object* v_name_3969_; lean_object* v_traceClass_3970_; lean_object* v___x_3972_; uint8_t v_isShared_3973_; uint8_t v_isSharedCheck_3991_; 
v_inheritedTraceOptions_3968_ = lean_ctor_get(v_a_3851_, 13);
v_name_3969_ = lean_ctor_get(v___x_3944_, 0);
lean_inc(v_name_3969_);
lean_dec_ref(v___x_3944_);
v_traceClass_3970_ = lean_ctor_get(v_traceOpt_3845_, 0);
v_isSharedCheck_3991_ = !lean_is_exclusive(v_traceOpt_3845_);
if (v_isSharedCheck_3991_ == 0)
{
lean_object* v_unused_3992_; 
v_unused_3992_ = lean_ctor_get(v_traceOpt_3845_, 1);
lean_dec(v_unused_3992_);
v___x_3972_ = v_traceOpt_3845_;
v_isShared_3973_ = v_isSharedCheck_3991_;
goto v_resetjp_3971_;
}
else
{
lean_inc(v_traceClass_3970_);
lean_dec(v_traceOpt_3845_);
v___x_3972_ = lean_box(0);
v_isShared_3973_ = v_isSharedCheck_3991_;
goto v_resetjp_3971_;
}
v_resetjp_3971_:
{
lean_object* v___x_3974_; lean_object* v___x_3975_; lean_object* v___x_3976_; lean_object* v___x_3977_; lean_object* v___x_3978_; lean_object* v___x_3979_; lean_object* v___x_3981_; 
v___x_3974_ = lean_string_append(v___y_3963_, v___y_3964_);
v___x_3975_ = lean_string_append(v___x_3974_, v___y_3962_);
v___x_3976_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_name_3969_, v_hasTrace_3966_);
v___x_3977_ = lean_string_append(v___x_3975_, v___x_3976_);
lean_dec_ref(v___x_3976_);
v___x_3978_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_3978_, 0, v___x_3977_);
v___x_3979_ = l_Lean_MessageData_ofFormat(v___x_3978_);
if (v_isShared_3973_ == 0)
{
lean_ctor_set_tag(v___x_3972_, 7);
lean_ctor_set(v___x_3972_, 1, v___x_3979_);
lean_ctor_set(v___x_3972_, 0, v___x_3960_);
v___x_3981_ = v___x_3972_;
goto v_reusejp_3980_;
}
else
{
lean_object* v_reuseFailAlloc_3990_; 
v_reuseFailAlloc_3990_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3990_, 0, v___x_3960_);
lean_ctor_set(v_reuseFailAlloc_3990_, 1, v___x_3979_);
v___x_3981_ = v_reuseFailAlloc_3990_;
goto v_reusejp_3980_;
}
v_reusejp_3980_:
{
lean_object* v___f_3982_; lean_object* v___x_3983_; lean_object* v___x_3984_; lean_object* v___x_3985_; uint8_t v___x_3986_; 
v___f_3982_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Rapp_withHeadlineTraceNode___redArg___lam__0___boxed), 8, 2);
lean_closure_set(v___f_3982_, 0, v_transform_3848_);
lean_closure_set(v___f_3982_, 1, v___x_3981_);
v___x_3983_ = ((lean_object*)(lp_aesop_Aesop_Goal_withHeadlineTraceNode___redArg___closed__0));
v___x_3984_ = ((lean_object*)(lp_aesop_Aesop_Goal_withHeadlineTraceNode___redArg___closed__2));
lean_inc(v_traceClass_3970_);
v___x_3985_ = l_Lean_Name_append(v___x_3984_, v_traceClass_3970_);
v___x_3986_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_3968_, v_options_3965_, v___x_3985_);
lean_dec(v___x_3985_);
if (v___x_3986_ == 0)
{
lean_object* v___x_3987_; uint8_t v___x_3988_; 
v___x_3987_ = l_Lean_trace_profiler;
v___x_3988_ = lp_aesop_Lean_Option_get___at___00Aesop_Goal_withHeadlineTraceNode_spec__1(v_options_3965_, v___x_3987_);
if (v___x_3988_ == 0)
{
lean_object* v___x_3989_; 
lean_dec_ref(v___f_3982_);
lean_dec(v_traceClass_3970_);
lean_inc(v_a_3852_);
lean_inc_ref(v_a_3851_);
lean_inc(v_a_3850_);
lean_inc_ref(v_a_3849_);
v___x_3989_ = lean_apply_5(v_k_3846_, v_a_3849_, v_a_3850_, v_a_3851_, v_a_3852_, lean_box(0));
return v___x_3989_;
}
else
{
v___y_3892_ = v___f_3982_;
v___y_3893_ = v_traceClass_3970_;
v___y_3894_ = v_options_3965_;
v___y_3895_ = v___x_3983_;
v___y_3896_ = v___x_3986_;
goto v___jp_3891_;
}
}
else
{
v___y_3892_ = v___f_3982_;
v___y_3893_ = v_traceClass_3970_;
v___y_3894_ = v_options_3965_;
v___y_3895_ = v___x_3983_;
v___y_3896_ = v___x_3986_;
goto v___jp_3891_;
}
}
}
}
}
v___jp_3993_:
{
uint8_t v_scope_3997_; lean_object* v___x_3998_; lean_object* v___x_3999_; 
v_scope_3997_ = lean_ctor_get_uint8(v___x_3944_, sizeof(void*)*1 + 10);
v___x_3998_ = lean_string_append(v___y_3995_, v___y_3996_);
v___x_3999_ = lean_string_append(v___x_3998_, v___y_3994_);
if (v_scope_3997_ == 0)
{
lean_object* v___x_4000_; 
v___x_4000_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__4));
v___y_3962_ = v___y_3994_;
v___y_3963_ = v___x_3999_;
v___y_3964_ = v___x_4000_;
goto v___jp_3961_;
}
else
{
lean_object* v___x_4001_; 
v___x_4001_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__5));
v___y_3962_ = v___y_3994_;
v___y_3963_ = v___x_3999_;
v___y_3964_ = v___x_4001_;
goto v___jp_3961_;
}
}
v___jp_4002_:
{
uint8_t v_builder_4004_; lean_object* v___x_4005_; lean_object* v___x_4006_; 
v_builder_4004_ = lean_ctor_get_uint8(v___x_3944_, sizeof(void*)*1 + 8);
v___x_4005_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__6));
lean_inc_ref(v___y_4003_);
v___x_4006_ = lean_string_append(v___y_4003_, v___x_4005_);
switch(v_builder_4004_)
{
case 0:
{
lean_object* v___x_4007_; 
v___x_4007_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__7));
v___y_3994_ = v___x_4005_;
v___y_3995_ = v___x_4006_;
v___y_3996_ = v___x_4007_;
goto v___jp_3993_;
}
case 1:
{
lean_object* v___x_4008_; 
v___x_4008_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__8));
v___y_3994_ = v___x_4005_;
v___y_3995_ = v___x_4006_;
v___y_3996_ = v___x_4008_;
goto v___jp_3993_;
}
case 2:
{
lean_object* v___x_4009_; 
v___x_4009_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__9));
v___y_3994_ = v___x_4005_;
v___y_3995_ = v___x_4006_;
v___y_3996_ = v___x_4009_;
goto v___jp_3993_;
}
case 3:
{
lean_object* v___x_4010_; 
v___x_4010_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__10));
v___y_3994_ = v___x_4005_;
v___y_3995_ = v___x_4006_;
v___y_3996_ = v___x_4010_;
goto v___jp_3993_;
}
case 4:
{
lean_object* v___x_4011_; 
v___x_4011_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__11));
v___y_3994_ = v___x_4005_;
v___y_3995_ = v___x_4006_;
v___y_3996_ = v___x_4011_;
goto v___jp_3993_;
}
case 5:
{
lean_object* v___x_4012_; 
v___x_4012_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__12));
v___y_3994_ = v___x_4005_;
v___y_3995_ = v___x_4006_;
v___y_3996_ = v___x_4012_;
goto v___jp_3993_;
}
case 6:
{
lean_object* v___x_4013_; 
v___x_4013_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__13));
v___y_3994_ = v___x_4005_;
v___y_3995_ = v___x_4006_;
v___y_3996_ = v___x_4013_;
goto v___jp_3993_;
}
default: 
{
lean_object* v___x_4014_; 
v___x_4014_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__14));
v___y_3994_ = v___x_4005_;
v___y_3995_ = v___x_4006_;
v___y_3996_ = v___x_4014_;
goto v___jp_3993_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_withHeadlineTraceNode___redArg___boxed(lean_object* v_r_4018_, lean_object* v_traceOpt_4019_, lean_object* v_k_4020_, lean_object* v_collapsed_4021_, lean_object* v_transform_4022_, lean_object* v_a_4023_, lean_object* v_a_4024_, lean_object* v_a_4025_, lean_object* v_a_4026_, lean_object* v_a_4027_){
_start:
{
uint8_t v_collapsed_boxed_4028_; lean_object* v_res_4029_; 
v_collapsed_boxed_4028_ = lean_unbox(v_collapsed_4021_);
v_res_4029_ = lp_aesop_Aesop_Rapp_withHeadlineTraceNode___redArg(v_r_4018_, v_traceOpt_4019_, v_k_4020_, v_collapsed_boxed_4028_, v_transform_4022_, v_a_4023_, v_a_4024_, v_a_4025_, v_a_4026_);
lean_dec(v_a_4026_);
lean_dec_ref(v_a_4025_);
lean_dec(v_a_4024_);
lean_dec_ref(v_a_4023_);
return v_res_4029_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_withHeadlineTraceNode(lean_object* v_00_u03b1_4030_, lean_object* v_r_4031_, lean_object* v_traceOpt_4032_, lean_object* v_k_4033_, uint8_t v_collapsed_4034_, lean_object* v_transform_4035_, lean_object* v_a_4036_, lean_object* v_a_4037_, lean_object* v_a_4038_, lean_object* v_a_4039_){
_start:
{
lean_object* v___x_4041_; 
v___x_4041_ = lp_aesop_Aesop_Rapp_withHeadlineTraceNode___redArg(v_r_4031_, v_traceOpt_4032_, v_k_4033_, v_collapsed_4034_, v_transform_4035_, v_a_4036_, v_a_4037_, v_a_4038_, v_a_4039_);
return v___x_4041_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_withHeadlineTraceNode___boxed(lean_object* v_00_u03b1_4042_, lean_object* v_r_4043_, lean_object* v_traceOpt_4044_, lean_object* v_k_4045_, lean_object* v_collapsed_4046_, lean_object* v_transform_4047_, lean_object* v_a_4048_, lean_object* v_a_4049_, lean_object* v_a_4050_, lean_object* v_a_4051_, lean_object* v_a_4052_){
_start:
{
uint8_t v_collapsed_boxed_4053_; lean_object* v_res_4054_; 
v_collapsed_boxed_4053_ = lean_unbox(v_collapsed_4046_);
v_res_4054_ = lp_aesop_Aesop_Rapp_withHeadlineTraceNode(v_00_u03b1_4042_, v_r_4043_, v_traceOpt_4044_, v_k_4045_, v_collapsed_boxed_4053_, v_transform_4047_, v_a_4048_, v_a_4049_, v_a_4050_, v_a_4051_);
lean_dec(v_a_4051_);
lean_dec_ref(v_a_4050_);
lean_dec(v_a_4049_);
lean_dec_ref(v_a_4048_);
return v_res_4054_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Rapp_traceMetadata_trc(lean_object* v_traceOpt_4055_, lean_object* v_m_4056_, lean_object* v_a_4057_, lean_object* v_a_4058_, lean_object* v_a_4059_, lean_object* v_a_4060_){
_start:
{
lean_object* v_traceClass_4062_; lean_object* v___x_4063_; 
v_traceClass_4062_ = lean_ctor_get(v_traceOpt_4055_, 0);
lean_inc(v_traceClass_4062_);
lean_dec_ref(v_traceOpt_4055_);
v___x_4063_ = lp_aesop_Lean_addTrace___at___00__private_Aesop_Tree_Tracing_0__Aesop_Goal_traceMetadata_trc_spec__0(v_traceClass_4062_, v_m_4056_, v_a_4057_, v_a_4058_, v_a_4059_, v_a_4060_);
return v___x_4063_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Rapp_traceMetadata_trc___boxed(lean_object* v_traceOpt_4064_, lean_object* v_m_4065_, lean_object* v_a_4066_, lean_object* v_a_4067_, lean_object* v_a_4068_, lean_object* v_a_4069_, lean_object* v_a_4070_){
_start:
{
lean_object* v_res_4071_; 
v_res_4071_ = lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Rapp_traceMetadata_trc(v_traceOpt_4064_, v_m_4065_, v_a_4066_, v_a_4067_, v_a_4068_, v_a_4069_);
lean_dec(v_a_4069_);
lean_dec_ref(v_a_4068_);
lean_dec(v_a_4067_);
lean_dec_ref(v_a_4066_);
return v_res_4071_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Rapp_traceMetadata_spec__1___redArg(size_t v_sz_4072_, size_t v_i_4073_, lean_object* v_bs_4074_){
_start:
{
uint8_t v___x_4076_; 
v___x_4076_ = lean_usize_dec_lt(v_i_4073_, v_sz_4072_);
if (v___x_4076_ == 0)
{
lean_object* v___x_4077_; 
v___x_4077_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4077_, 0, v_bs_4074_);
return v___x_4077_;
}
else
{
lean_object* v_v_4078_; lean_object* v___x_4079_; lean_object* v___x_4080_; lean_object* v_elimGoal_4081_; lean_object* v___x_4082_; lean_object* v_id_4083_; lean_object* v___x_4084_; lean_object* v_bs_x27_4085_; size_t v___x_4086_; size_t v___x_4087_; lean_object* v___x_4088_; 
v_v_4078_ = lean_array_uget_borrowed(v_bs_4074_, v_i_4073_);
v___x_4079_ = lean_st_ref_get(v_v_4078_);
v___x_4080_ = lp_aesop_Aesop_treeImpl;
v_elimGoal_4081_ = lean_ctor_get(v___x_4080_, 1);
lean_inc_ref(v_elimGoal_4081_);
v___x_4082_ = lean_apply_1(v_elimGoal_4081_, v___x_4079_);
v_id_4083_ = lean_ctor_get(v___x_4082_, 0);
lean_inc(v_id_4083_);
lean_dec_ref(v___x_4082_);
v___x_4084_ = lean_unsigned_to_nat(0u);
v_bs_x27_4085_ = lean_array_uset(v_bs_4074_, v_i_4073_, v___x_4084_);
v___x_4086_ = ((size_t)1ULL);
v___x_4087_ = lean_usize_add(v_i_4073_, v___x_4086_);
v___x_4088_ = lean_array_uset(v_bs_x27_4085_, v_i_4073_, v_id_4083_);
v_i_4073_ = v___x_4087_;
v_bs_4074_ = v___x_4088_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Rapp_traceMetadata_spec__1___redArg___boxed(lean_object* v_sz_4090_, lean_object* v_i_4091_, lean_object* v_bs_4092_, lean_object* v___y_4093_){
_start:
{
size_t v_sz_boxed_4094_; size_t v_i_boxed_4095_; lean_object* v_res_4096_; 
v_sz_boxed_4094_ = lean_unbox_usize(v_sz_4090_);
lean_dec(v_sz_4090_);
v_i_boxed_4095_ = lean_unbox_usize(v_i_4091_);
lean_dec(v_i_4091_);
v_res_4096_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Rapp_traceMetadata_spec__1___redArg(v_sz_boxed_4094_, v_i_boxed_4095_, v_bs_4092_);
return v_res_4096_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Rapp_foldSubgoalsM___at___00Aesop_Rapp_subgoals___at___00Aesop_Rapp_traceMetadata_spec__0_spec__0_spec__1___redArg(lean_object* v_f_4097_, lean_object* v_as_4098_, size_t v_i_4099_, size_t v_stop_4100_, lean_object* v_b_4101_, lean_object* v___y_4102_, lean_object* v___y_4103_, lean_object* v___y_4104_, lean_object* v___y_4105_){
_start:
{
uint8_t v___x_4107_; 
v___x_4107_ = lean_usize_dec_eq(v_i_4099_, v_stop_4100_);
if (v___x_4107_ == 0)
{
lean_object* v___x_4108_; lean_object* v___x_4109_; 
v___x_4108_ = lean_array_uget_borrowed(v_as_4098_, v_i_4099_);
lean_inc_ref(v_f_4097_);
lean_inc(v___y_4105_);
lean_inc_ref(v___y_4104_);
lean_inc(v___y_4103_);
lean_inc_ref(v___y_4102_);
lean_inc(v___x_4108_);
v___x_4109_ = lean_apply_7(v_f_4097_, v_b_4101_, v___x_4108_, v___y_4102_, v___y_4103_, v___y_4104_, v___y_4105_, lean_box(0));
if (lean_obj_tag(v___x_4109_) == 0)
{
lean_object* v_a_4110_; size_t v___x_4111_; size_t v___x_4112_; 
v_a_4110_ = lean_ctor_get(v___x_4109_, 0);
lean_inc(v_a_4110_);
lean_dec_ref_known(v___x_4109_, 1);
v___x_4111_ = ((size_t)1ULL);
v___x_4112_ = lean_usize_add(v_i_4099_, v___x_4111_);
v_i_4099_ = v___x_4112_;
v_b_4101_ = v_a_4110_;
goto _start;
}
else
{
lean_dec_ref(v_f_4097_);
return v___x_4109_;
}
}
else
{
lean_object* v___x_4114_; 
lean_dec_ref(v_f_4097_);
v___x_4114_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4114_, 0, v_b_4101_);
return v___x_4114_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Rapp_foldSubgoalsM___at___00Aesop_Rapp_subgoals___at___00Aesop_Rapp_traceMetadata_spec__0_spec__0_spec__1___redArg___boxed(lean_object* v_f_4115_, lean_object* v_as_4116_, lean_object* v_i_4117_, lean_object* v_stop_4118_, lean_object* v_b_4119_, lean_object* v___y_4120_, lean_object* v___y_4121_, lean_object* v___y_4122_, lean_object* v___y_4123_, lean_object* v___y_4124_){
_start:
{
size_t v_i_boxed_4125_; size_t v_stop_boxed_4126_; lean_object* v_res_4127_; 
v_i_boxed_4125_ = lean_unbox_usize(v_i_4117_);
lean_dec(v_i_4117_);
v_stop_boxed_4126_ = lean_unbox_usize(v_stop_4118_);
lean_dec(v_stop_4118_);
v_res_4127_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Rapp_foldSubgoalsM___at___00Aesop_Rapp_subgoals___at___00Aesop_Rapp_traceMetadata_spec__0_spec__0_spec__1___redArg(v_f_4115_, v_as_4116_, v_i_boxed_4125_, v_stop_boxed_4126_, v_b_4119_, v___y_4120_, v___y_4121_, v___y_4122_, v___y_4123_);
lean_dec(v___y_4123_);
lean_dec_ref(v___y_4122_);
lean_dec(v___y_4121_);
lean_dec_ref(v___y_4120_);
lean_dec_ref(v_as_4116_);
return v_res_4127_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Rapp_foldSubgoalsM___at___00Aesop_Rapp_subgoals___at___00Aesop_Rapp_traceMetadata_spec__0_spec__0_spec__2___redArg(lean_object* v_f_4128_, lean_object* v_as_4129_, size_t v_i_4130_, size_t v_stop_4131_, lean_object* v_b_4132_, lean_object* v___y_4133_, lean_object* v___y_4134_, lean_object* v___y_4135_, lean_object* v___y_4136_){
_start:
{
lean_object* v_a_4139_; lean_object* v___y_4144_; uint8_t v___x_4146_; 
v___x_4146_ = lean_usize_dec_eq(v_i_4130_, v_stop_4131_);
if (v___x_4146_ == 0)
{
lean_object* v___x_4147_; lean_object* v___x_4148_; lean_object* v___x_4149_; lean_object* v_elimMVarCluster_4150_; lean_object* v___x_4151_; lean_object* v_goals_4152_; lean_object* v___x_4153_; lean_object* v___x_4154_; uint8_t v___x_4155_; 
v___x_4147_ = lean_array_uget_borrowed(v_as_4129_, v_i_4130_);
v___x_4148_ = lean_st_ref_get(v___x_4147_);
v___x_4149_ = lp_aesop_Aesop_treeImpl;
v_elimMVarCluster_4150_ = lean_ctor_get(v___x_4149_, 5);
lean_inc_ref(v_elimMVarCluster_4150_);
v___x_4151_ = lean_apply_1(v_elimMVarCluster_4150_, v___x_4148_);
v_goals_4152_ = lean_ctor_get(v___x_4151_, 1);
lean_inc_ref(v_goals_4152_);
lean_dec_ref(v___x_4151_);
v___x_4153_ = lean_unsigned_to_nat(0u);
v___x_4154_ = lean_array_get_size(v_goals_4152_);
v___x_4155_ = lean_nat_dec_lt(v___x_4153_, v___x_4154_);
if (v___x_4155_ == 0)
{
lean_dec_ref(v_goals_4152_);
v_a_4139_ = v_b_4132_;
goto v___jp_4138_;
}
else
{
uint8_t v___x_4156_; 
v___x_4156_ = lean_nat_dec_le(v___x_4154_, v___x_4154_);
if (v___x_4156_ == 0)
{
if (v___x_4155_ == 0)
{
lean_dec_ref(v_goals_4152_);
v_a_4139_ = v_b_4132_;
goto v___jp_4138_;
}
else
{
size_t v___x_4157_; size_t v___x_4158_; lean_object* v___x_4159_; 
v___x_4157_ = ((size_t)0ULL);
v___x_4158_ = lean_usize_of_nat(v___x_4154_);
lean_inc_ref(v_f_4128_);
v___x_4159_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Rapp_foldSubgoalsM___at___00Aesop_Rapp_subgoals___at___00Aesop_Rapp_traceMetadata_spec__0_spec__0_spec__1___redArg(v_f_4128_, v_goals_4152_, v___x_4157_, v___x_4158_, v_b_4132_, v___y_4133_, v___y_4134_, v___y_4135_, v___y_4136_);
lean_dec_ref(v_goals_4152_);
v___y_4144_ = v___x_4159_;
goto v___jp_4143_;
}
}
else
{
size_t v___x_4160_; size_t v___x_4161_; lean_object* v___x_4162_; 
v___x_4160_ = ((size_t)0ULL);
v___x_4161_ = lean_usize_of_nat(v___x_4154_);
lean_inc_ref(v_f_4128_);
v___x_4162_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Rapp_foldSubgoalsM___at___00Aesop_Rapp_subgoals___at___00Aesop_Rapp_traceMetadata_spec__0_spec__0_spec__1___redArg(v_f_4128_, v_goals_4152_, v___x_4160_, v___x_4161_, v_b_4132_, v___y_4133_, v___y_4134_, v___y_4135_, v___y_4136_);
lean_dec_ref(v_goals_4152_);
v___y_4144_ = v___x_4162_;
goto v___jp_4143_;
}
}
}
else
{
lean_object* v___x_4163_; 
lean_dec_ref(v_f_4128_);
v___x_4163_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4163_, 0, v_b_4132_);
return v___x_4163_;
}
v___jp_4138_:
{
size_t v___x_4140_; size_t v___x_4141_; 
v___x_4140_ = ((size_t)1ULL);
v___x_4141_ = lean_usize_add(v_i_4130_, v___x_4140_);
v_i_4130_ = v___x_4141_;
v_b_4132_ = v_a_4139_;
goto _start;
}
v___jp_4143_:
{
if (lean_obj_tag(v___y_4144_) == 0)
{
lean_object* v_a_4145_; 
v_a_4145_ = lean_ctor_get(v___y_4144_, 0);
lean_inc(v_a_4145_);
lean_dec_ref_known(v___y_4144_, 1);
v_a_4139_ = v_a_4145_;
goto v___jp_4138_;
}
else
{
lean_dec_ref(v_f_4128_);
return v___y_4144_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Rapp_foldSubgoalsM___at___00Aesop_Rapp_subgoals___at___00Aesop_Rapp_traceMetadata_spec__0_spec__0_spec__2___redArg___boxed(lean_object* v_f_4164_, lean_object* v_as_4165_, lean_object* v_i_4166_, lean_object* v_stop_4167_, lean_object* v_b_4168_, lean_object* v___y_4169_, lean_object* v___y_4170_, lean_object* v___y_4171_, lean_object* v___y_4172_, lean_object* v___y_4173_){
_start:
{
size_t v_i_boxed_4174_; size_t v_stop_boxed_4175_; lean_object* v_res_4176_; 
v_i_boxed_4174_ = lean_unbox_usize(v_i_4166_);
lean_dec(v_i_4166_);
v_stop_boxed_4175_ = lean_unbox_usize(v_stop_4167_);
lean_dec(v_stop_4167_);
v_res_4176_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Rapp_foldSubgoalsM___at___00Aesop_Rapp_subgoals___at___00Aesop_Rapp_traceMetadata_spec__0_spec__0_spec__2___redArg(v_f_4164_, v_as_4165_, v_i_boxed_4174_, v_stop_boxed_4175_, v_b_4168_, v___y_4169_, v___y_4170_, v___y_4171_, v___y_4172_);
lean_dec(v___y_4172_);
lean_dec_ref(v___y_4171_);
lean_dec(v___y_4170_);
lean_dec_ref(v___y_4169_);
lean_dec_ref(v_as_4165_);
return v_res_4176_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_foldSubgoalsM___at___00Aesop_Rapp_subgoals___at___00Aesop_Rapp_traceMetadata_spec__0_spec__0___redArg(lean_object* v_init_4177_, lean_object* v_f_4178_, lean_object* v_r_4179_, lean_object* v___y_4180_, lean_object* v___y_4181_, lean_object* v___y_4182_, lean_object* v___y_4183_){
_start:
{
lean_object* v___x_4185_; lean_object* v_elimRapp_4186_; lean_object* v___x_4187_; lean_object* v_children_4188_; lean_object* v___x_4189_; lean_object* v___x_4190_; uint8_t v___x_4191_; 
v___x_4185_ = lp_aesop_Aesop_treeImpl;
v_elimRapp_4186_ = lean_ctor_get(v___x_4185_, 3);
lean_inc_ref(v_elimRapp_4186_);
v___x_4187_ = lean_apply_1(v_elimRapp_4186_, v_r_4179_);
v_children_4188_ = lean_ctor_get(v___x_4187_, 2);
lean_inc_ref(v_children_4188_);
lean_dec_ref(v___x_4187_);
v___x_4189_ = lean_unsigned_to_nat(0u);
v___x_4190_ = lean_array_get_size(v_children_4188_);
v___x_4191_ = lean_nat_dec_lt(v___x_4189_, v___x_4190_);
if (v___x_4191_ == 0)
{
lean_object* v___x_4192_; 
lean_dec_ref(v_children_4188_);
lean_dec_ref(v_f_4178_);
v___x_4192_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4192_, 0, v_init_4177_);
return v___x_4192_;
}
else
{
uint8_t v___x_4193_; 
v___x_4193_ = lean_nat_dec_le(v___x_4190_, v___x_4190_);
if (v___x_4193_ == 0)
{
if (v___x_4191_ == 0)
{
lean_object* v___x_4194_; 
lean_dec_ref(v_children_4188_);
lean_dec_ref(v_f_4178_);
v___x_4194_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4194_, 0, v_init_4177_);
return v___x_4194_;
}
else
{
size_t v___x_4195_; size_t v___x_4196_; lean_object* v___x_4197_; 
v___x_4195_ = ((size_t)0ULL);
v___x_4196_ = lean_usize_of_nat(v___x_4190_);
v___x_4197_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Rapp_foldSubgoalsM___at___00Aesop_Rapp_subgoals___at___00Aesop_Rapp_traceMetadata_spec__0_spec__0_spec__2___redArg(v_f_4178_, v_children_4188_, v___x_4195_, v___x_4196_, v_init_4177_, v___y_4180_, v___y_4181_, v___y_4182_, v___y_4183_);
lean_dec_ref(v_children_4188_);
return v___x_4197_;
}
}
else
{
size_t v___x_4198_; size_t v___x_4199_; lean_object* v___x_4200_; 
v___x_4198_ = ((size_t)0ULL);
v___x_4199_ = lean_usize_of_nat(v___x_4190_);
v___x_4200_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Rapp_foldSubgoalsM___at___00Aesop_Rapp_subgoals___at___00Aesop_Rapp_traceMetadata_spec__0_spec__0_spec__2___redArg(v_f_4178_, v_children_4188_, v___x_4198_, v___x_4199_, v_init_4177_, v___y_4180_, v___y_4181_, v___y_4182_, v___y_4183_);
lean_dec_ref(v_children_4188_);
return v___x_4200_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_foldSubgoalsM___at___00Aesop_Rapp_subgoals___at___00Aesop_Rapp_traceMetadata_spec__0_spec__0___redArg___boxed(lean_object* v_init_4201_, lean_object* v_f_4202_, lean_object* v_r_4203_, lean_object* v___y_4204_, lean_object* v___y_4205_, lean_object* v___y_4206_, lean_object* v___y_4207_, lean_object* v___y_4208_){
_start:
{
lean_object* v_res_4209_; 
v_res_4209_ = lp_aesop_Aesop_Rapp_foldSubgoalsM___at___00Aesop_Rapp_subgoals___at___00Aesop_Rapp_traceMetadata_spec__0_spec__0___redArg(v_init_4201_, v_f_4202_, v_r_4203_, v___y_4204_, v___y_4205_, v___y_4206_, v___y_4207_);
lean_dec(v___y_4207_);
lean_dec_ref(v___y_4206_);
lean_dec(v___y_4205_);
lean_dec_ref(v___y_4204_);
return v_res_4209_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_subgoals___at___00Aesop_Rapp_traceMetadata_spec__0___lam__0(lean_object* v_subgoals_4210_, lean_object* v_gref_4211_, lean_object* v___y_4212_, lean_object* v___y_4213_, lean_object* v___y_4214_, lean_object* v___y_4215_){
_start:
{
lean_object* v___x_4217_; lean_object* v___x_4218_; 
v___x_4217_ = lean_array_push(v_subgoals_4210_, v_gref_4211_);
v___x_4218_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4218_, 0, v___x_4217_);
return v___x_4218_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_subgoals___at___00Aesop_Rapp_traceMetadata_spec__0___lam__0___boxed(lean_object* v_subgoals_4219_, lean_object* v_gref_4220_, lean_object* v___y_4221_, lean_object* v___y_4222_, lean_object* v___y_4223_, lean_object* v___y_4224_, lean_object* v___y_4225_){
_start:
{
lean_object* v_res_4226_; 
v_res_4226_ = lp_aesop_Aesop_Rapp_subgoals___at___00Aesop_Rapp_traceMetadata_spec__0___lam__0(v_subgoals_4219_, v_gref_4220_, v___y_4221_, v___y_4222_, v___y_4223_, v___y_4224_);
lean_dec(v___y_4224_);
lean_dec_ref(v___y_4223_);
lean_dec(v___y_4222_);
lean_dec_ref(v___y_4221_);
return v_res_4226_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_subgoals___at___00Aesop_Rapp_traceMetadata_spec__0(lean_object* v_r_4230_, lean_object* v___y_4231_, lean_object* v___y_4232_, lean_object* v___y_4233_, lean_object* v___y_4234_){
_start:
{
lean_object* v___f_4236_; lean_object* v___x_4237_; lean_object* v___x_4238_; 
v___f_4236_ = ((lean_object*)(lp_aesop_Aesop_Rapp_subgoals___at___00Aesop_Rapp_traceMetadata_spec__0___closed__0));
v___x_4237_ = ((lean_object*)(lp_aesop_Aesop_Rapp_subgoals___at___00Aesop_Rapp_traceMetadata_spec__0___closed__1));
v___x_4238_ = lp_aesop_Aesop_Rapp_foldSubgoalsM___at___00Aesop_Rapp_subgoals___at___00Aesop_Rapp_traceMetadata_spec__0_spec__0___redArg(v___x_4237_, v___f_4236_, v_r_4230_, v___y_4231_, v___y_4232_, v___y_4233_, v___y_4234_);
return v___x_4238_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_subgoals___at___00Aesop_Rapp_traceMetadata_spec__0___boxed(lean_object* v_r_4239_, lean_object* v___y_4240_, lean_object* v___y_4241_, lean_object* v___y_4242_, lean_object* v___y_4243_, lean_object* v___y_4244_){
_start:
{
lean_object* v_res_4245_; 
v_res_4245_ = lp_aesop_Aesop_Rapp_subgoals___at___00Aesop_Rapp_traceMetadata_spec__0(v_r_4239_, v___y_4240_, v___y_4241_, v___y_4242_, v___y_4243_);
lean_dec(v___y_4243_);
lean_dec_ref(v___y_4242_);
lean_dec(v___y_4241_);
lean_dec_ref(v___y_4240_);
return v_res_4245_;
}
}
LEAN_EXPORT lean_object* lp_aesop_List_mapTR_loop___at___00Aesop_Rapp_traceMetadata_spec__2(lean_object* v_a_4246_, lean_object* v_a_4247_){
_start:
{
if (lean_obj_tag(v_a_4246_) == 0)
{
lean_object* v___x_4248_; 
v___x_4248_ = l_List_reverse___redArg(v_a_4247_);
return v___x_4248_;
}
else
{
lean_object* v_head_4249_; lean_object* v_tail_4250_; lean_object* v___x_4252_; uint8_t v_isShared_4253_; uint8_t v_isSharedCheck_4261_; 
v_head_4249_ = lean_ctor_get(v_a_4246_, 0);
v_tail_4250_ = lean_ctor_get(v_a_4246_, 1);
v_isSharedCheck_4261_ = !lean_is_exclusive(v_a_4246_);
if (v_isSharedCheck_4261_ == 0)
{
v___x_4252_ = v_a_4246_;
v_isShared_4253_ = v_isSharedCheck_4261_;
goto v_resetjp_4251_;
}
else
{
lean_inc(v_tail_4250_);
lean_inc(v_head_4249_);
lean_dec(v_a_4246_);
v___x_4252_ = lean_box(0);
v_isShared_4253_ = v_isSharedCheck_4261_;
goto v_resetjp_4251_;
}
v_resetjp_4251_:
{
lean_object* v___x_4254_; lean_object* v___x_4255_; lean_object* v___x_4256_; lean_object* v___x_4258_; 
v___x_4254_ = l_Nat_reprFast(v_head_4249_);
v___x_4255_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_4255_, 0, v___x_4254_);
v___x_4256_ = l_Lean_MessageData_ofFormat(v___x_4255_);
if (v_isShared_4253_ == 0)
{
lean_ctor_set(v___x_4252_, 1, v_a_4247_);
lean_ctor_set(v___x_4252_, 0, v___x_4256_);
v___x_4258_ = v___x_4252_;
goto v_reusejp_4257_;
}
else
{
lean_object* v_reuseFailAlloc_4260_; 
v_reuseFailAlloc_4260_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4260_, 0, v___x_4256_);
lean_ctor_set(v_reuseFailAlloc_4260_, 1, v_a_4247_);
v___x_4258_ = v_reuseFailAlloc_4260_;
goto v_reusejp_4257_;
}
v_reusejp_4257_:
{
v_a_4246_ = v_tail_4250_;
v_a_4247_ = v___x_4258_;
goto _start;
}
}
}
}
}
static lean_object* _init_lp_aesop_Aesop_Rapp_traceMetadata___closed__1(void){
_start:
{
lean_object* v___x_4263_; lean_object* v___x_4264_; 
v___x_4263_ = ((lean_object*)(lp_aesop_Aesop_Rapp_traceMetadata___closed__0));
v___x_4264_ = l_Lean_stringToMessageData(v___x_4263_);
return v___x_4264_;
}
}
static lean_object* _init_lp_aesop_Aesop_Rapp_traceMetadata___closed__3(void){
_start:
{
lean_object* v___x_4266_; lean_object* v___x_4267_; 
v___x_4266_ = ((lean_object*)(lp_aesop_Aesop_Rapp_traceMetadata___closed__2));
v___x_4267_ = l_Lean_stringToMessageData(v___x_4266_);
return v___x_4267_;
}
}
static lean_object* _init_lp_aesop_Aesop_Rapp_traceMetadata___closed__7(void){
_start:
{
lean_object* v___x_4271_; lean_object* v___x_4272_; 
v___x_4271_ = ((lean_object*)(lp_aesop_Aesop_Rapp_traceMetadata___closed__6));
v___x_4272_ = l_Lean_stringToMessageData(v___x_4271_);
return v___x_4272_;
}
}
static lean_object* _init_lp_aesop_Aesop_Rapp_traceMetadata___closed__9(void){
_start:
{
lean_object* v___x_4274_; lean_object* v___x_4275_; 
v___x_4274_ = ((lean_object*)(lp_aesop_Aesop_Rapp_traceMetadata___closed__8));
v___x_4275_ = l_Lean_stringToMessageData(v___x_4274_);
return v___x_4275_;
}
}
static lean_object* _init_lp_aesop_Aesop_Rapp_traceMetadata___closed__11(void){
_start:
{
lean_object* v___x_4277_; lean_object* v___x_4278_; 
v___x_4277_ = ((lean_object*)(lp_aesop_Aesop_Rapp_traceMetadata___closed__10));
v___x_4278_ = l_Lean_stringToMessageData(v___x_4277_);
return v___x_4278_;
}
}
static lean_object* _init_lp_aesop_Aesop_Rapp_traceMetadata___closed__13(void){
_start:
{
lean_object* v___x_4280_; lean_object* v___x_4281_; 
v___x_4280_ = ((lean_object*)(lp_aesop_Aesop_Rapp_traceMetadata___closed__12));
v___x_4281_ = l_Lean_stringToMessageData(v___x_4280_);
return v___x_4281_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_traceMetadata(lean_object* v_r_4285_, lean_object* v_traceOpt_4286_, lean_object* v_a_4287_, lean_object* v_a_4288_, lean_object* v_a_4289_, lean_object* v_a_4290_){
_start:
{
lean_object* v___x_4292_; lean_object* v_a_4293_; lean_object* v___x_4295_; uint8_t v_isShared_4296_; uint8_t v_isSharedCheck_4534_; 
v___x_4292_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_Goal_traceMetadata_spec__5___redArg(v_traceOpt_4286_, v_a_4289_);
v_a_4293_ = lean_ctor_get(v___x_4292_, 0);
v_isSharedCheck_4534_ = !lean_is_exclusive(v___x_4292_);
if (v_isSharedCheck_4534_ == 0)
{
v___x_4295_ = v___x_4292_;
v_isShared_4296_ = v_isSharedCheck_4534_;
goto v_resetjp_4294_;
}
else
{
lean_inc(v_a_4293_);
lean_dec(v___x_4292_);
v___x_4295_ = lean_box(0);
v_isShared_4296_ = v_isSharedCheck_4534_;
goto v_resetjp_4294_;
}
v_resetjp_4294_:
{
uint8_t v___x_4297_; 
v___x_4297_ = lean_unbox(v_a_4293_);
if (v___x_4297_ == 0)
{
lean_object* v___x_4298_; lean_object* v___x_4300_; 
lean_dec(v_a_4293_);
lean_dec_ref(v_traceOpt_4286_);
lean_dec(v_r_4285_);
v___x_4298_ = lean_box(0);
if (v_isShared_4296_ == 0)
{
lean_ctor_set(v___x_4295_, 0, v___x_4298_);
v___x_4300_ = v___x_4295_;
goto v_reusejp_4299_;
}
else
{
lean_object* v_reuseFailAlloc_4301_; 
v_reuseFailAlloc_4301_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4301_, 0, v___x_4298_);
v___x_4300_ = v_reuseFailAlloc_4301_;
goto v_reusejp_4299_;
}
v_reusejp_4299_:
{
return v___x_4300_;
}
}
else
{
lean_object* v___x_4302_; lean_object* v_elimGoal_4303_; lean_object* v_elimRapp_4304_; lean_object* v___x_4305_; lean_object* v_id_4306_; lean_object* v_parent_4307_; uint8_t v_state_4308_; uint8_t v_isIrrelevant_4309_; lean_object* v_appliedRule_4310_; double v_successProbability_4311_; lean_object* v_introducedMVars_4312_; lean_object* v_assignedMVars_4313_; lean_object* v___y_4315_; lean_object* v___y_4316_; size_t v___y_4317_; lean_object* v___y_4318_; lean_object* v___y_4340_; lean_object* v___y_4341_; size_t v___y_4342_; lean_object* v___y_4343_; lean_object* v___x_4351_; lean_object* v___x_4352_; lean_object* v___x_4353_; lean_object* v___x_4354_; lean_object* v___x_4355_; lean_object* v___x_4356_; 
lean_del_object(v___x_4295_);
v___x_4302_ = lp_aesop_Aesop_treeImpl;
v_elimGoal_4303_ = lean_ctor_get(v___x_4302_, 1);
v_elimRapp_4304_ = lean_ctor_get(v___x_4302_, 3);
lean_inc_ref(v_elimRapp_4304_);
lean_inc(v_r_4285_);
v___x_4305_ = lean_apply_1(v_elimRapp_4304_, v_r_4285_);
v_id_4306_ = lean_ctor_get(v___x_4305_, 0);
lean_inc(v_id_4306_);
v_parent_4307_ = lean_ctor_get(v___x_4305_, 1);
lean_inc(v_parent_4307_);
v_state_4308_ = lean_ctor_get_uint8(v___x_4305_, sizeof(void*)*9 + 8);
v_isIrrelevant_4309_ = lean_ctor_get_uint8(v___x_4305_, sizeof(void*)*9 + 9);
v_appliedRule_4310_ = lean_ctor_get(v___x_4305_, 3);
lean_inc_ref(v_appliedRule_4310_);
v_successProbability_4311_ = lean_ctor_get_float(v___x_4305_, sizeof(void*)*9);
v_introducedMVars_4312_ = lean_ctor_get(v___x_4305_, 7);
lean_inc_ref(v_introducedMVars_4312_);
v_assignedMVars_4313_ = lean_ctor_get(v___x_4305_, 8);
lean_inc_ref(v_assignedMVars_4313_);
lean_dec_ref(v___x_4305_);
v___x_4351_ = lean_obj_once(&lp_aesop_Aesop_Goal_traceMetadata___closed__5, &lp_aesop_Aesop_Goal_traceMetadata___closed__5_once, _init_lp_aesop_Aesop_Goal_traceMetadata___closed__5);
v___x_4352_ = l_Nat_reprFast(v_id_4306_);
v___x_4353_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_4353_, 0, v___x_4352_);
v___x_4354_ = l_Lean_MessageData_ofFormat(v___x_4353_);
v___x_4355_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4355_, 0, v___x_4351_);
lean_ctor_set(v___x_4355_, 1, v___x_4354_);
lean_inc_ref(v_traceOpt_4286_);
v___x_4356_ = lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Rapp_traceMetadata_trc(v_traceOpt_4286_, v___x_4355_, v_a_4287_, v_a_4288_, v_a_4289_, v_a_4290_);
if (lean_obj_tag(v___x_4356_) == 0)
{
lean_object* v___x_4357_; lean_object* v___y_4359_; 
lean_dec_ref_known(v___x_4356_, 1);
v___x_4357_ = lean_obj_once(&lp_aesop_Aesop_Rapp_traceMetadata___closed__7, &lp_aesop_Aesop_Rapp_traceMetadata___closed__7_once, _init_lp_aesop_Aesop_Rapp_traceMetadata___closed__7);
if (lean_obj_tag(v_appliedRule_4310_) == 0)
{
lean_object* v_r_4417_; lean_object* v___x_4419_; uint8_t v_isShared_4420_; uint8_t v_isSharedCheck_4480_; 
v_r_4417_ = lean_ctor_get(v_appliedRule_4310_, 0);
v_isSharedCheck_4480_ = !lean_is_exclusive(v_appliedRule_4310_);
if (v_isSharedCheck_4480_ == 0)
{
v___x_4419_ = v_appliedRule_4310_;
v_isShared_4420_ = v_isSharedCheck_4480_;
goto v_resetjp_4418_;
}
else
{
lean_inc(v_r_4417_);
lean_dec(v_appliedRule_4310_);
v___x_4419_ = lean_box(0);
v_isShared_4420_ = v_isSharedCheck_4480_;
goto v_resetjp_4418_;
}
v_resetjp_4418_:
{
lean_object* v_name_4421_; lean_object* v_extra_4422_; lean_object* v___y_4424_; lean_object* v___y_4425_; lean_object* v___y_4426_; lean_object* v___y_4427_; lean_object* v___y_4439_; lean_object* v___y_4440_; lean_object* v___y_4441_; lean_object* v___y_4442_; lean_object* v___y_4449_; lean_object* v___y_4450_; lean_object* v_penalty_4462_; uint8_t v_safety_4463_; lean_object* v___x_4464_; lean_object* v___x_4465_; lean_object* v___x_4466_; lean_object* v___x_4467_; lean_object* v___x_4468_; lean_object* v___y_4470_; 
v_name_4421_ = lean_ctor_get(v_r_4417_, 0);
lean_inc_ref(v_name_4421_);
v_extra_4422_ = lean_ctor_get(v_r_4417_, 3);
lean_inc(v_extra_4422_);
lean_dec_ref(v_r_4417_);
v_penalty_4462_ = lean_ctor_get(v_extra_4422_, 0);
lean_inc(v_penalty_4462_);
v_safety_4463_ = lean_ctor_get_uint8(v_extra_4422_, sizeof(void*)*1);
lean_dec(v_extra_4422_);
v___x_4464_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__0));
v___x_4465_ = l_Int_repr(v_penalty_4462_);
lean_dec(v_penalty_4462_);
v___x_4466_ = lean_string_append(v___x_4464_, v___x_4465_);
lean_dec_ref(v___x_4465_);
v___x_4467_ = ((lean_object*)(lp_aesop_Aesop_Rapp_traceMetadata___closed__15));
v___x_4468_ = lean_string_append(v___x_4466_, v___x_4467_);
if (v_safety_4463_ == 0)
{
lean_object* v___x_4478_; 
v___x_4478_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__16));
v___y_4470_ = v___x_4478_;
goto v___jp_4469_;
}
else
{
lean_object* v___x_4479_; 
v___x_4479_ = ((lean_object*)(lp_aesop_Aesop_Rapp_traceMetadata___closed__16));
v___y_4470_ = v___x_4479_;
goto v___jp_4469_;
}
v___jp_4423_:
{
lean_object* v_name_4428_; lean_object* v___x_4429_; lean_object* v___x_4430_; uint8_t v___x_4431_; lean_object* v___x_4432_; lean_object* v___x_4433_; lean_object* v___x_4434_; lean_object* v___x_4436_; 
v_name_4428_ = lean_ctor_get(v_name_4421_, 0);
lean_inc(v_name_4428_);
lean_dec_ref(v_name_4421_);
v___x_4429_ = lean_string_append(v___y_4425_, v___y_4427_);
v___x_4430_ = lean_string_append(v___x_4429_, v___y_4424_);
v___x_4431_ = lean_unbox(v_a_4293_);
lean_dec(v_a_4293_);
v___x_4432_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_name_4428_, v___x_4431_);
v___x_4433_ = lean_string_append(v___x_4430_, v___x_4432_);
lean_dec_ref(v___x_4432_);
v___x_4434_ = lean_string_append(v___y_4426_, v___x_4433_);
lean_dec_ref(v___x_4433_);
if (v_isShared_4420_ == 0)
{
lean_ctor_set_tag(v___x_4419_, 3);
lean_ctor_set(v___x_4419_, 0, v___x_4434_);
v___x_4436_ = v___x_4419_;
goto v_reusejp_4435_;
}
else
{
lean_object* v_reuseFailAlloc_4437_; 
v_reuseFailAlloc_4437_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4437_, 0, v___x_4434_);
v___x_4436_ = v_reuseFailAlloc_4437_;
goto v_reusejp_4435_;
}
v_reusejp_4435_:
{
v___y_4359_ = v___x_4436_;
goto v___jp_4358_;
}
}
v___jp_4438_:
{
uint8_t v_scope_4443_; lean_object* v___x_4444_; lean_object* v___x_4445_; 
v_scope_4443_ = lean_ctor_get_uint8(v_name_4421_, sizeof(void*)*1 + 10);
v___x_4444_ = lean_string_append(v___y_4441_, v___y_4442_);
v___x_4445_ = lean_string_append(v___x_4444_, v___y_4439_);
if (v_scope_4443_ == 0)
{
lean_object* v___x_4446_; 
v___x_4446_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__4));
v___y_4424_ = v___y_4439_;
v___y_4425_ = v___x_4445_;
v___y_4426_ = v___y_4440_;
v___y_4427_ = v___x_4446_;
goto v___jp_4423_;
}
else
{
lean_object* v___x_4447_; 
v___x_4447_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__5));
v___y_4424_ = v___y_4439_;
v___y_4425_ = v___x_4445_;
v___y_4426_ = v___y_4440_;
v___y_4427_ = v___x_4447_;
goto v___jp_4423_;
}
}
v___jp_4448_:
{
uint8_t v_builder_4451_; lean_object* v___x_4452_; lean_object* v___x_4453_; 
v_builder_4451_ = lean_ctor_get_uint8(v_name_4421_, sizeof(void*)*1 + 8);
v___x_4452_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__6));
lean_inc_ref(v___y_4450_);
v___x_4453_ = lean_string_append(v___y_4450_, v___x_4452_);
switch(v_builder_4451_)
{
case 0:
{
lean_object* v___x_4454_; 
v___x_4454_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__7));
v___y_4439_ = v___x_4452_;
v___y_4440_ = v___y_4449_;
v___y_4441_ = v___x_4453_;
v___y_4442_ = v___x_4454_;
goto v___jp_4438_;
}
case 1:
{
lean_object* v___x_4455_; 
v___x_4455_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__8));
v___y_4439_ = v___x_4452_;
v___y_4440_ = v___y_4449_;
v___y_4441_ = v___x_4453_;
v___y_4442_ = v___x_4455_;
goto v___jp_4438_;
}
case 2:
{
lean_object* v___x_4456_; 
v___x_4456_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__9));
v___y_4439_ = v___x_4452_;
v___y_4440_ = v___y_4449_;
v___y_4441_ = v___x_4453_;
v___y_4442_ = v___x_4456_;
goto v___jp_4438_;
}
case 3:
{
lean_object* v___x_4457_; 
v___x_4457_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__10));
v___y_4439_ = v___x_4452_;
v___y_4440_ = v___y_4449_;
v___y_4441_ = v___x_4453_;
v___y_4442_ = v___x_4457_;
goto v___jp_4438_;
}
case 4:
{
lean_object* v___x_4458_; 
v___x_4458_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__11));
v___y_4439_ = v___x_4452_;
v___y_4440_ = v___y_4449_;
v___y_4441_ = v___x_4453_;
v___y_4442_ = v___x_4458_;
goto v___jp_4438_;
}
case 5:
{
lean_object* v___x_4459_; 
v___x_4459_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__12));
v___y_4439_ = v___x_4452_;
v___y_4440_ = v___y_4449_;
v___y_4441_ = v___x_4453_;
v___y_4442_ = v___x_4459_;
goto v___jp_4438_;
}
case 6:
{
lean_object* v___x_4460_; 
v___x_4460_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__13));
v___y_4439_ = v___x_4452_;
v___y_4440_ = v___y_4449_;
v___y_4441_ = v___x_4453_;
v___y_4442_ = v___x_4460_;
goto v___jp_4438_;
}
default: 
{
lean_object* v___x_4461_; 
v___x_4461_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__14));
v___y_4439_ = v___x_4452_;
v___y_4440_ = v___y_4449_;
v___y_4441_ = v___x_4453_;
v___y_4442_ = v___x_4461_;
goto v___jp_4438_;
}
}
}
v___jp_4469_:
{
uint8_t v_phase_4471_; lean_object* v___x_4472_; lean_object* v___x_4473_; lean_object* v___x_4474_; 
v_phase_4471_ = lean_ctor_get_uint8(v_name_4421_, sizeof(void*)*1 + 9);
v___x_4472_ = lean_string_append(v___x_4468_, v___y_4470_);
v___x_4473_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__2));
v___x_4474_ = lean_string_append(v___x_4472_, v___x_4473_);
switch(v_phase_4471_)
{
case 0:
{
lean_object* v___x_4475_; 
v___x_4475_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__15));
v___y_4449_ = v___x_4474_;
v___y_4450_ = v___x_4475_;
goto v___jp_4448_;
}
case 1:
{
lean_object* v___x_4476_; 
v___x_4476_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__16));
v___y_4449_ = v___x_4474_;
v___y_4450_ = v___x_4476_;
goto v___jp_4448_;
}
default: 
{
lean_object* v___x_4477_; 
v___x_4477_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__17));
v___y_4449_ = v___x_4474_;
v___y_4450_ = v___x_4477_;
goto v___jp_4448_;
}
}
}
}
}
else
{
lean_object* v_r_4481_; lean_object* v___x_4483_; uint8_t v_isShared_4484_; uint8_t v_isSharedCheck_4533_; 
v_r_4481_ = lean_ctor_get(v_appliedRule_4310_, 0);
v_isSharedCheck_4533_ = !lean_is_exclusive(v_appliedRule_4310_);
if (v_isSharedCheck_4533_ == 0)
{
v___x_4483_ = v_appliedRule_4310_;
v_isShared_4484_ = v_isSharedCheck_4533_;
goto v_resetjp_4482_;
}
else
{
lean_inc(v_r_4481_);
lean_dec(v_appliedRule_4310_);
v___x_4483_ = lean_box(0);
v_isShared_4484_ = v_isSharedCheck_4533_;
goto v_resetjp_4482_;
}
v_resetjp_4482_:
{
lean_object* v_name_4485_; lean_object* v_extra_4486_; lean_object* v_name_4487_; uint8_t v_builder_4488_; uint8_t v_phase_4489_; uint8_t v_scope_4490_; lean_object* v___x_4491_; double v___x_4492_; lean_object* v___x_4493_; lean_object* v___x_4494_; lean_object* v___x_4495_; lean_object* v___x_4496_; lean_object* v___y_4498_; lean_object* v___y_4499_; lean_object* v___y_4500_; lean_object* v___y_4511_; lean_object* v___y_4512_; lean_object* v___y_4513_; lean_object* v___y_4519_; 
v_name_4485_ = lean_ctor_get(v_r_4481_, 0);
lean_inc_ref(v_name_4485_);
v_extra_4486_ = lean_ctor_get(v_r_4481_, 3);
lean_inc(v_extra_4486_);
lean_dec_ref(v_r_4481_);
v_name_4487_ = lean_ctor_get(v_name_4485_, 0);
lean_inc(v_name_4487_);
v_builder_4488_ = lean_ctor_get_uint8(v_name_4485_, sizeof(void*)*1 + 8);
v_phase_4489_ = lean_ctor_get_uint8(v_name_4485_, sizeof(void*)*1 + 9);
v_scope_4490_ = lean_ctor_get_uint8(v_name_4485_, sizeof(void*)*1 + 10);
lean_dec_ref(v_name_4485_);
v___x_4491_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__0));
v___x_4492_ = lean_unbox_float(v_extra_4486_);
lean_dec(v_extra_4486_);
v___x_4493_ = lp_aesop_Aesop_Percent_toHumanString(v___x_4492_);
v___x_4494_ = lean_string_append(v___x_4491_, v___x_4493_);
lean_dec_ref(v___x_4493_);
v___x_4495_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__2));
v___x_4496_ = lean_string_append(v___x_4494_, v___x_4495_);
switch(v_phase_4489_)
{
case 0:
{
lean_object* v___x_4530_; 
v___x_4530_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__15));
v___y_4519_ = v___x_4530_;
goto v___jp_4518_;
}
case 1:
{
lean_object* v___x_4531_; 
v___x_4531_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__16));
v___y_4519_ = v___x_4531_;
goto v___jp_4518_;
}
default: 
{
lean_object* v___x_4532_; 
v___x_4532_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__17));
v___y_4519_ = v___x_4532_;
goto v___jp_4518_;
}
}
v___jp_4497_:
{
lean_object* v___x_4501_; lean_object* v___x_4502_; uint8_t v___x_4503_; lean_object* v___x_4504_; lean_object* v___x_4505_; lean_object* v___x_4506_; lean_object* v___x_4508_; 
v___x_4501_ = lean_string_append(v___y_4499_, v___y_4500_);
v___x_4502_ = lean_string_append(v___x_4501_, v___y_4498_);
v___x_4503_ = lean_unbox(v_a_4293_);
lean_dec(v_a_4293_);
v___x_4504_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_name_4487_, v___x_4503_);
v___x_4505_ = lean_string_append(v___x_4502_, v___x_4504_);
lean_dec_ref(v___x_4504_);
v___x_4506_ = lean_string_append(v___x_4496_, v___x_4505_);
lean_dec_ref(v___x_4505_);
if (v_isShared_4484_ == 0)
{
lean_ctor_set_tag(v___x_4483_, 3);
lean_ctor_set(v___x_4483_, 0, v___x_4506_);
v___x_4508_ = v___x_4483_;
goto v_reusejp_4507_;
}
else
{
lean_object* v_reuseFailAlloc_4509_; 
v_reuseFailAlloc_4509_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4509_, 0, v___x_4506_);
v___x_4508_ = v_reuseFailAlloc_4509_;
goto v_reusejp_4507_;
}
v_reusejp_4507_:
{
v___y_4359_ = v___x_4508_;
goto v___jp_4358_;
}
}
v___jp_4510_:
{
lean_object* v___x_4514_; lean_object* v___x_4515_; 
v___x_4514_ = lean_string_append(v___y_4511_, v___y_4513_);
v___x_4515_ = lean_string_append(v___x_4514_, v___y_4512_);
if (v_scope_4490_ == 0)
{
lean_object* v___x_4516_; 
v___x_4516_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__4));
v___y_4498_ = v___y_4512_;
v___y_4499_ = v___x_4515_;
v___y_4500_ = v___x_4516_;
goto v___jp_4497_;
}
else
{
lean_object* v___x_4517_; 
v___x_4517_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__5));
v___y_4498_ = v___y_4512_;
v___y_4499_ = v___x_4515_;
v___y_4500_ = v___x_4517_;
goto v___jp_4497_;
}
}
v___jp_4518_:
{
lean_object* v___x_4520_; lean_object* v___x_4521_; 
v___x_4520_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__6));
lean_inc_ref(v___y_4519_);
v___x_4521_ = lean_string_append(v___y_4519_, v___x_4520_);
switch(v_builder_4488_)
{
case 0:
{
lean_object* v___x_4522_; 
v___x_4522_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__7));
v___y_4511_ = v___x_4521_;
v___y_4512_ = v___x_4520_;
v___y_4513_ = v___x_4522_;
goto v___jp_4510_;
}
case 1:
{
lean_object* v___x_4523_; 
v___x_4523_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__8));
v___y_4511_ = v___x_4521_;
v___y_4512_ = v___x_4520_;
v___y_4513_ = v___x_4523_;
goto v___jp_4510_;
}
case 2:
{
lean_object* v___x_4524_; 
v___x_4524_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__9));
v___y_4511_ = v___x_4521_;
v___y_4512_ = v___x_4520_;
v___y_4513_ = v___x_4524_;
goto v___jp_4510_;
}
case 3:
{
lean_object* v___x_4525_; 
v___x_4525_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__10));
v___y_4511_ = v___x_4521_;
v___y_4512_ = v___x_4520_;
v___y_4513_ = v___x_4525_;
goto v___jp_4510_;
}
case 4:
{
lean_object* v___x_4526_; 
v___x_4526_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__11));
v___y_4511_ = v___x_4521_;
v___y_4512_ = v___x_4520_;
v___y_4513_ = v___x_4526_;
goto v___jp_4510_;
}
case 5:
{
lean_object* v___x_4527_; 
v___x_4527_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__12));
v___y_4511_ = v___x_4521_;
v___y_4512_ = v___x_4520_;
v___y_4513_ = v___x_4527_;
goto v___jp_4510_;
}
case 6:
{
lean_object* v___x_4528_; 
v___x_4528_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__13));
v___y_4511_ = v___x_4521_;
v___y_4512_ = v___x_4520_;
v___y_4513_ = v___x_4528_;
goto v___jp_4510_;
}
default: 
{
lean_object* v___x_4529_; 
v___x_4529_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceMetadata_spec__6___closed__14));
v___y_4511_ = v___x_4521_;
v___y_4512_ = v___x_4520_;
v___y_4513_ = v___x_4529_;
goto v___jp_4510_;
}
}
}
}
}
v___jp_4358_:
{
lean_object* v___x_4360_; lean_object* v___x_4361_; lean_object* v___x_4362_; 
v___x_4360_ = l_Lean_MessageData_ofFormat(v___y_4359_);
v___x_4361_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4361_, 0, v___x_4357_);
lean_ctor_set(v___x_4361_, 1, v___x_4360_);
lean_inc_ref(v_traceOpt_4286_);
v___x_4362_ = lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Rapp_traceMetadata_trc(v_traceOpt_4286_, v___x_4361_, v_a_4287_, v_a_4288_, v_a_4289_, v_a_4290_);
if (lean_obj_tag(v___x_4362_) == 0)
{
lean_object* v___x_4363_; lean_object* v___x_4364_; lean_object* v___x_4365_; lean_object* v___x_4366_; lean_object* v___x_4367_; 
lean_dec_ref_known(v___x_4362_, 1);
v___x_4363_ = lean_obj_once(&lp_aesop_Aesop_Rapp_traceMetadata___closed__9, &lp_aesop_Aesop_Rapp_traceMetadata___closed__9_once, _init_lp_aesop_Aesop_Rapp_traceMetadata___closed__9);
v___x_4364_ = lp_aesop_Aesop_Percent_toHumanString(v_successProbability_4311_);
v___x_4365_ = l_Lean_stringToMessageData(v___x_4364_);
v___x_4366_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4366_, 0, v___x_4363_);
lean_ctor_set(v___x_4366_, 1, v___x_4365_);
lean_inc_ref(v_traceOpt_4286_);
v___x_4367_ = lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Rapp_traceMetadata_trc(v_traceOpt_4286_, v___x_4366_, v_a_4287_, v_a_4288_, v_a_4289_, v_a_4290_);
if (lean_obj_tag(v___x_4367_) == 0)
{
lean_object* v___x_4369_; uint8_t v_isShared_4370_; uint8_t v_isSharedCheck_4415_; 
v_isSharedCheck_4415_ = !lean_is_exclusive(v___x_4367_);
if (v_isSharedCheck_4415_ == 0)
{
lean_object* v_unused_4416_; 
v_unused_4416_ = lean_ctor_get(v___x_4367_, 0);
lean_dec(v_unused_4416_);
v___x_4369_ = v___x_4367_;
v_isShared_4370_ = v_isSharedCheck_4415_;
goto v_resetjp_4368_;
}
else
{
lean_dec(v___x_4367_);
v___x_4369_ = lean_box(0);
v_isShared_4370_ = v_isSharedCheck_4415_;
goto v_resetjp_4368_;
}
v_resetjp_4368_:
{
lean_object* v___x_4371_; lean_object* v___x_4372_; lean_object* v_id_4373_; lean_object* v___x_4374_; lean_object* v___x_4375_; lean_object* v___x_4377_; 
v___x_4371_ = lean_st_ref_get(v_parent_4307_);
lean_dec(v_parent_4307_);
lean_inc_ref(v_elimGoal_4303_);
v___x_4372_ = lean_apply_1(v_elimGoal_4303_, v___x_4371_);
v_id_4373_ = lean_ctor_get(v___x_4372_, 0);
lean_inc(v_id_4373_);
lean_dec_ref(v___x_4372_);
v___x_4374_ = lean_obj_once(&lp_aesop_Aesop_Rapp_traceMetadata___closed__11, &lp_aesop_Aesop_Rapp_traceMetadata___closed__11_once, _init_lp_aesop_Aesop_Rapp_traceMetadata___closed__11);
v___x_4375_ = l_Nat_reprFast(v_id_4373_);
if (v_isShared_4370_ == 0)
{
lean_ctor_set_tag(v___x_4369_, 3);
lean_ctor_set(v___x_4369_, 0, v___x_4375_);
v___x_4377_ = v___x_4369_;
goto v_reusejp_4376_;
}
else
{
lean_object* v_reuseFailAlloc_4414_; 
v_reuseFailAlloc_4414_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4414_, 0, v___x_4375_);
v___x_4377_ = v_reuseFailAlloc_4414_;
goto v_reusejp_4376_;
}
v_reusejp_4376_:
{
lean_object* v___x_4378_; lean_object* v___x_4379_; lean_object* v___x_4380_; 
v___x_4378_ = l_Lean_MessageData_ofFormat(v___x_4377_);
v___x_4379_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4379_, 0, v___x_4374_);
lean_ctor_set(v___x_4379_, 1, v___x_4378_);
lean_inc_ref(v_traceOpt_4286_);
v___x_4380_ = lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Rapp_traceMetadata_trc(v_traceOpt_4286_, v___x_4379_, v_a_4287_, v_a_4288_, v_a_4289_, v_a_4290_);
if (lean_obj_tag(v___x_4380_) == 0)
{
lean_object* v___x_4381_; 
lean_dec_ref_known(v___x_4380_, 1);
v___x_4381_ = lp_aesop_Aesop_Rapp_subgoals___at___00Aesop_Rapp_traceMetadata_spec__0(v_r_4285_, v_a_4287_, v_a_4288_, v_a_4289_, v_a_4290_);
if (lean_obj_tag(v___x_4381_) == 0)
{
lean_object* v_a_4382_; size_t v_sz_4383_; size_t v___x_4384_; lean_object* v___x_4385_; 
v_a_4382_ = lean_ctor_get(v___x_4381_, 0);
lean_inc(v_a_4382_);
lean_dec_ref_known(v___x_4381_, 1);
v_sz_4383_ = lean_array_size(v_a_4382_);
v___x_4384_ = ((size_t)0ULL);
v___x_4385_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Rapp_traceMetadata_spec__1___redArg(v_sz_4383_, v___x_4384_, v_a_4382_);
if (lean_obj_tag(v___x_4385_) == 0)
{
lean_object* v_a_4386_; lean_object* v___x_4387_; lean_object* v___x_4388_; lean_object* v___x_4389_; lean_object* v___x_4390_; lean_object* v___x_4391_; lean_object* v___x_4392_; lean_object* v___x_4393_; 
v_a_4386_ = lean_ctor_get(v___x_4385_, 0);
lean_inc(v_a_4386_);
lean_dec_ref_known(v___x_4385_, 1);
v___x_4387_ = lean_obj_once(&lp_aesop_Aesop_Rapp_traceMetadata___closed__13, &lp_aesop_Aesop_Rapp_traceMetadata___closed__13_once, _init_lp_aesop_Aesop_Rapp_traceMetadata___closed__13);
v___x_4388_ = lean_array_to_list(v_a_4386_);
v___x_4389_ = lean_box(0);
v___x_4390_ = lp_aesop_List_mapTR_loop___at___00Aesop_Rapp_traceMetadata_spec__2(v___x_4388_, v___x_4389_);
v___x_4391_ = l_Lean_MessageData_ofList(v___x_4390_);
v___x_4392_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4392_, 0, v___x_4387_);
lean_ctor_set(v___x_4392_, 1, v___x_4391_);
lean_inc_ref(v_traceOpt_4286_);
v___x_4393_ = lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Rapp_traceMetadata_trc(v_traceOpt_4286_, v___x_4392_, v_a_4287_, v_a_4288_, v_a_4289_, v_a_4290_);
if (lean_obj_tag(v___x_4393_) == 0)
{
lean_object* v___x_4394_; 
lean_dec_ref_known(v___x_4393_, 1);
v___x_4394_ = lean_obj_once(&lp_aesop_Aesop_Goal_traceMetadata___closed__37, &lp_aesop_Aesop_Goal_traceMetadata___closed__37_once, _init_lp_aesop_Aesop_Goal_traceMetadata___closed__37);
switch(v_state_4308_)
{
case 0:
{
lean_object* v___x_4395_; 
v___x_4395_ = ((lean_object*)(lp_aesop_Aesop_Goal_traceMetadata___closed__40));
v___y_4340_ = v___x_4389_;
v___y_4341_ = v___x_4394_;
v___y_4342_ = v___x_4384_;
v___y_4343_ = v___x_4395_;
goto v___jp_4339_;
}
case 1:
{
lean_object* v___x_4396_; 
v___x_4396_ = ((lean_object*)(lp_aesop_Aesop_Rapp_traceMetadata___closed__14));
v___y_4340_ = v___x_4389_;
v___y_4341_ = v___x_4394_;
v___y_4342_ = v___x_4384_;
v___y_4343_ = v___x_4396_;
goto v___jp_4339_;
}
default: 
{
lean_object* v___x_4397_; 
v___x_4397_ = ((lean_object*)(lp_aesop_Aesop_Goal_traceMetadata___closed__43));
v___y_4340_ = v___x_4389_;
v___y_4341_ = v___x_4394_;
v___y_4342_ = v___x_4384_;
v___y_4343_ = v___x_4397_;
goto v___jp_4339_;
}
}
}
else
{
lean_dec_ref(v_assignedMVars_4313_);
lean_dec_ref(v_introducedMVars_4312_);
lean_dec_ref(v_traceOpt_4286_);
return v___x_4393_;
}
}
else
{
lean_object* v_a_4398_; lean_object* v___x_4400_; uint8_t v_isShared_4401_; uint8_t v_isSharedCheck_4405_; 
lean_dec_ref(v_assignedMVars_4313_);
lean_dec_ref(v_introducedMVars_4312_);
lean_dec_ref(v_traceOpt_4286_);
v_a_4398_ = lean_ctor_get(v___x_4385_, 0);
v_isSharedCheck_4405_ = !lean_is_exclusive(v___x_4385_);
if (v_isSharedCheck_4405_ == 0)
{
v___x_4400_ = v___x_4385_;
v_isShared_4401_ = v_isSharedCheck_4405_;
goto v_resetjp_4399_;
}
else
{
lean_inc(v_a_4398_);
lean_dec(v___x_4385_);
v___x_4400_ = lean_box(0);
v_isShared_4401_ = v_isSharedCheck_4405_;
goto v_resetjp_4399_;
}
v_resetjp_4399_:
{
lean_object* v___x_4403_; 
if (v_isShared_4401_ == 0)
{
v___x_4403_ = v___x_4400_;
goto v_reusejp_4402_;
}
else
{
lean_object* v_reuseFailAlloc_4404_; 
v_reuseFailAlloc_4404_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4404_, 0, v_a_4398_);
v___x_4403_ = v_reuseFailAlloc_4404_;
goto v_reusejp_4402_;
}
v_reusejp_4402_:
{
return v___x_4403_;
}
}
}
}
else
{
lean_object* v_a_4406_; lean_object* v___x_4408_; uint8_t v_isShared_4409_; uint8_t v_isSharedCheck_4413_; 
lean_dec_ref(v_assignedMVars_4313_);
lean_dec_ref(v_introducedMVars_4312_);
lean_dec_ref(v_traceOpt_4286_);
v_a_4406_ = lean_ctor_get(v___x_4381_, 0);
v_isSharedCheck_4413_ = !lean_is_exclusive(v___x_4381_);
if (v_isSharedCheck_4413_ == 0)
{
v___x_4408_ = v___x_4381_;
v_isShared_4409_ = v_isSharedCheck_4413_;
goto v_resetjp_4407_;
}
else
{
lean_inc(v_a_4406_);
lean_dec(v___x_4381_);
v___x_4408_ = lean_box(0);
v_isShared_4409_ = v_isSharedCheck_4413_;
goto v_resetjp_4407_;
}
v_resetjp_4407_:
{
lean_object* v___x_4411_; 
if (v_isShared_4409_ == 0)
{
v___x_4411_ = v___x_4408_;
goto v_reusejp_4410_;
}
else
{
lean_object* v_reuseFailAlloc_4412_; 
v_reuseFailAlloc_4412_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4412_, 0, v_a_4406_);
v___x_4411_ = v_reuseFailAlloc_4412_;
goto v_reusejp_4410_;
}
v_reusejp_4410_:
{
return v___x_4411_;
}
}
}
}
else
{
lean_dec_ref(v_assignedMVars_4313_);
lean_dec_ref(v_introducedMVars_4312_);
lean_dec_ref(v_traceOpt_4286_);
lean_dec(v_r_4285_);
return v___x_4380_;
}
}
}
}
else
{
lean_dec_ref(v_assignedMVars_4313_);
lean_dec_ref(v_introducedMVars_4312_);
lean_dec(v_parent_4307_);
lean_dec_ref(v_traceOpt_4286_);
lean_dec(v_r_4285_);
return v___x_4367_;
}
}
else
{
lean_dec_ref(v_assignedMVars_4313_);
lean_dec_ref(v_introducedMVars_4312_);
lean_dec(v_parent_4307_);
lean_dec_ref(v_traceOpt_4286_);
lean_dec(v_r_4285_);
return v___x_4362_;
}
}
}
else
{
lean_dec_ref(v_assignedMVars_4313_);
lean_dec_ref(v_introducedMVars_4312_);
lean_dec_ref(v_appliedRule_4310_);
lean_dec(v_parent_4307_);
lean_dec(v_a_4293_);
lean_dec_ref(v_traceOpt_4286_);
lean_dec(v_r_4285_);
return v___x_4356_;
}
v___jp_4314_:
{
lean_object* v___x_4319_; lean_object* v___x_4320_; lean_object* v___x_4321_; lean_object* v___x_4322_; 
lean_inc_ref(v___y_4318_);
v___x_4319_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_4319_, 0, v___y_4318_);
v___x_4320_ = l_Lean_MessageData_ofFormat(v___x_4319_);
lean_inc_ref(v___y_4315_);
v___x_4321_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4321_, 0, v___y_4315_);
lean_ctor_set(v___x_4321_, 1, v___x_4320_);
lean_inc_ref(v_traceOpt_4286_);
v___x_4322_ = lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Rapp_traceMetadata_trc(v_traceOpt_4286_, v___x_4321_, v_a_4287_, v_a_4288_, v_a_4289_, v_a_4290_);
if (lean_obj_tag(v___x_4322_) == 0)
{
lean_object* v___x_4323_; size_t v_sz_4324_; lean_object* v___x_4325_; lean_object* v___x_4326_; lean_object* v___x_4327_; lean_object* v___x_4328_; lean_object* v___x_4329_; lean_object* v___x_4330_; 
lean_dec_ref_known(v___x_4322_, 1);
v___x_4323_ = lean_obj_once(&lp_aesop_Aesop_Rapp_traceMetadata___closed__1, &lp_aesop_Aesop_Rapp_traceMetadata___closed__1_once, _init_lp_aesop_Aesop_Rapp_traceMetadata___closed__1);
v_sz_4324_ = lean_array_size(v_introducedMVars_4312_);
v___x_4325_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Goal_traceMetadata_spec__12(v_sz_4324_, v___y_4317_, v_introducedMVars_4312_);
v___x_4326_ = lean_array_to_list(v___x_4325_);
lean_inc(v___y_4316_);
v___x_4327_ = lp_aesop_List_mapTR_loop___at___00Aesop_Goal_traceMetadata_spec__13(v___x_4326_, v___y_4316_);
v___x_4328_ = l_Lean_MessageData_ofList(v___x_4327_);
v___x_4329_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4329_, 0, v___x_4323_);
lean_ctor_set(v___x_4329_, 1, v___x_4328_);
lean_inc_ref(v_traceOpt_4286_);
v___x_4330_ = lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Rapp_traceMetadata_trc(v_traceOpt_4286_, v___x_4329_, v_a_4287_, v_a_4288_, v_a_4289_, v_a_4290_);
if (lean_obj_tag(v___x_4330_) == 0)
{
lean_object* v___x_4331_; size_t v_sz_4332_; lean_object* v___x_4333_; lean_object* v___x_4334_; lean_object* v___x_4335_; lean_object* v___x_4336_; lean_object* v___x_4337_; lean_object* v___x_4338_; 
lean_dec_ref_known(v___x_4330_, 1);
v___x_4331_ = lean_obj_once(&lp_aesop_Aesop_Rapp_traceMetadata___closed__3, &lp_aesop_Aesop_Rapp_traceMetadata___closed__3_once, _init_lp_aesop_Aesop_Rapp_traceMetadata___closed__3);
v_sz_4332_ = lean_array_size(v_assignedMVars_4313_);
v___x_4333_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Goal_traceMetadata_spec__12(v_sz_4332_, v___y_4317_, v_assignedMVars_4313_);
v___x_4334_ = lean_array_to_list(v___x_4333_);
v___x_4335_ = lp_aesop_List_mapTR_loop___at___00Aesop_Goal_traceMetadata_spec__13(v___x_4334_, v___y_4316_);
v___x_4336_ = l_Lean_MessageData_ofList(v___x_4335_);
v___x_4337_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4337_, 0, v___x_4331_);
lean_ctor_set(v___x_4337_, 1, v___x_4336_);
v___x_4338_ = lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Rapp_traceMetadata_trc(v_traceOpt_4286_, v___x_4337_, v_a_4287_, v_a_4288_, v_a_4289_, v_a_4290_);
return v___x_4338_;
}
else
{
lean_dec(v___y_4316_);
lean_dec_ref(v_assignedMVars_4313_);
lean_dec_ref(v_traceOpt_4286_);
return v___x_4330_;
}
}
else
{
lean_dec(v___y_4316_);
lean_dec_ref(v_assignedMVars_4313_);
lean_dec_ref(v_introducedMVars_4312_);
lean_dec_ref(v_traceOpt_4286_);
return v___x_4322_;
}
}
v___jp_4339_:
{
lean_object* v___x_4344_; lean_object* v___x_4345_; lean_object* v___x_4346_; lean_object* v___x_4347_; 
lean_inc_ref(v___y_4343_);
v___x_4344_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_4344_, 0, v___y_4343_);
v___x_4345_ = l_Lean_MessageData_ofFormat(v___x_4344_);
lean_inc_ref(v___y_4341_);
v___x_4346_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4346_, 0, v___y_4341_);
lean_ctor_set(v___x_4346_, 1, v___x_4345_);
lean_inc_ref(v_traceOpt_4286_);
v___x_4347_ = lp_aesop___private_Aesop_Tree_Tracing_0__Aesop_Rapp_traceMetadata_trc(v_traceOpt_4286_, v___x_4346_, v_a_4287_, v_a_4288_, v_a_4289_, v_a_4290_);
if (lean_obj_tag(v___x_4347_) == 0)
{
lean_object* v___x_4348_; 
lean_dec_ref_known(v___x_4347_, 1);
v___x_4348_ = lean_obj_once(&lp_aesop_Aesop_Goal_traceMetadata___closed__22, &lp_aesop_Aesop_Goal_traceMetadata___closed__22_once, _init_lp_aesop_Aesop_Goal_traceMetadata___closed__22);
if (v_isIrrelevant_4309_ == 0)
{
lean_object* v___x_4349_; 
v___x_4349_ = ((lean_object*)(lp_aesop_Aesop_Rapp_traceMetadata___closed__4));
v___y_4315_ = v___x_4348_;
v___y_4316_ = v___y_4340_;
v___y_4317_ = v___y_4342_;
v___y_4318_ = v___x_4349_;
goto v___jp_4314_;
}
else
{
lean_object* v___x_4350_; 
v___x_4350_ = ((lean_object*)(lp_aesop_Aesop_Rapp_traceMetadata___closed__5));
v___y_4315_ = v___x_4348_;
v___y_4316_ = v___y_4340_;
v___y_4317_ = v___y_4342_;
v___y_4318_ = v___x_4350_;
goto v___jp_4314_;
}
}
else
{
lean_dec(v___y_4340_);
lean_dec_ref(v_assignedMVars_4313_);
lean_dec_ref(v_introducedMVars_4312_);
lean_dec_ref(v_traceOpt_4286_);
return v___x_4347_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_traceMetadata___boxed(lean_object* v_r_4535_, lean_object* v_traceOpt_4536_, lean_object* v_a_4537_, lean_object* v_a_4538_, lean_object* v_a_4539_, lean_object* v_a_4540_, lean_object* v_a_4541_){
_start:
{
lean_object* v_res_4542_; 
v_res_4542_ = lp_aesop_Aesop_Rapp_traceMetadata(v_r_4535_, v_traceOpt_4536_, v_a_4537_, v_a_4538_, v_a_4539_, v_a_4540_);
lean_dec(v_a_4540_);
lean_dec_ref(v_a_4539_);
lean_dec(v_a_4538_);
lean_dec_ref(v_a_4537_);
return v_res_4542_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Rapp_traceMetadata_spec__1(size_t v_sz_4543_, size_t v_i_4544_, lean_object* v_bs_4545_, lean_object* v___y_4546_, lean_object* v___y_4547_, lean_object* v___y_4548_, lean_object* v___y_4549_){
_start:
{
lean_object* v___x_4551_; 
v___x_4551_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Rapp_traceMetadata_spec__1___redArg(v_sz_4543_, v_i_4544_, v_bs_4545_);
return v___x_4551_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Rapp_traceMetadata_spec__1___boxed(lean_object* v_sz_4552_, lean_object* v_i_4553_, lean_object* v_bs_4554_, lean_object* v___y_4555_, lean_object* v___y_4556_, lean_object* v___y_4557_, lean_object* v___y_4558_, lean_object* v___y_4559_){
_start:
{
size_t v_sz_boxed_4560_; size_t v_i_boxed_4561_; lean_object* v_res_4562_; 
v_sz_boxed_4560_ = lean_unbox_usize(v_sz_4552_);
lean_dec(v_sz_4552_);
v_i_boxed_4561_ = lean_unbox_usize(v_i_4553_);
lean_dec(v_i_4553_);
v_res_4562_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Rapp_traceMetadata_spec__1(v_sz_boxed_4560_, v_i_boxed_4561_, v_bs_4554_, v___y_4555_, v___y_4556_, v___y_4557_, v___y_4558_);
lean_dec(v___y_4558_);
lean_dec_ref(v___y_4557_);
lean_dec(v___y_4556_);
lean_dec_ref(v___y_4555_);
return v_res_4562_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_foldSubgoalsM___at___00Aesop_Rapp_subgoals___at___00Aesop_Rapp_traceMetadata_spec__0_spec__0(lean_object* v_00_u03c3_4563_, lean_object* v_init_4564_, lean_object* v_f_4565_, lean_object* v_r_4566_, lean_object* v___y_4567_, lean_object* v___y_4568_, lean_object* v___y_4569_, lean_object* v___y_4570_){
_start:
{
lean_object* v___x_4572_; 
v___x_4572_ = lp_aesop_Aesop_Rapp_foldSubgoalsM___at___00Aesop_Rapp_subgoals___at___00Aesop_Rapp_traceMetadata_spec__0_spec__0___redArg(v_init_4564_, v_f_4565_, v_r_4566_, v___y_4567_, v___y_4568_, v___y_4569_, v___y_4570_);
return v___x_4572_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_foldSubgoalsM___at___00Aesop_Rapp_subgoals___at___00Aesop_Rapp_traceMetadata_spec__0_spec__0___boxed(lean_object* v_00_u03c3_4573_, lean_object* v_init_4574_, lean_object* v_f_4575_, lean_object* v_r_4576_, lean_object* v___y_4577_, lean_object* v___y_4578_, lean_object* v___y_4579_, lean_object* v___y_4580_, lean_object* v___y_4581_){
_start:
{
lean_object* v_res_4582_; 
v_res_4582_ = lp_aesop_Aesop_Rapp_foldSubgoalsM___at___00Aesop_Rapp_subgoals___at___00Aesop_Rapp_traceMetadata_spec__0_spec__0(v_00_u03c3_4573_, v_init_4574_, v_f_4575_, v_r_4576_, v___y_4577_, v___y_4578_, v___y_4579_, v___y_4580_);
lean_dec(v___y_4580_);
lean_dec_ref(v___y_4579_);
lean_dec(v___y_4578_);
lean_dec_ref(v___y_4577_);
return v_res_4582_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Rapp_foldSubgoalsM___at___00Aesop_Rapp_subgoals___at___00Aesop_Rapp_traceMetadata_spec__0_spec__0_spec__1(lean_object* v_00_u03c3_4583_, lean_object* v_f_4584_, lean_object* v_as_4585_, size_t v_i_4586_, size_t v_stop_4587_, lean_object* v_b_4588_, lean_object* v___y_4589_, lean_object* v___y_4590_, lean_object* v___y_4591_, lean_object* v___y_4592_){
_start:
{
lean_object* v___x_4594_; 
v___x_4594_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Rapp_foldSubgoalsM___at___00Aesop_Rapp_subgoals___at___00Aesop_Rapp_traceMetadata_spec__0_spec__0_spec__1___redArg(v_f_4584_, v_as_4585_, v_i_4586_, v_stop_4587_, v_b_4588_, v___y_4589_, v___y_4590_, v___y_4591_, v___y_4592_);
return v___x_4594_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Rapp_foldSubgoalsM___at___00Aesop_Rapp_subgoals___at___00Aesop_Rapp_traceMetadata_spec__0_spec__0_spec__1___boxed(lean_object* v_00_u03c3_4595_, lean_object* v_f_4596_, lean_object* v_as_4597_, lean_object* v_i_4598_, lean_object* v_stop_4599_, lean_object* v_b_4600_, lean_object* v___y_4601_, lean_object* v___y_4602_, lean_object* v___y_4603_, lean_object* v___y_4604_, lean_object* v___y_4605_){
_start:
{
size_t v_i_boxed_4606_; size_t v_stop_boxed_4607_; lean_object* v_res_4608_; 
v_i_boxed_4606_ = lean_unbox_usize(v_i_4598_);
lean_dec(v_i_4598_);
v_stop_boxed_4607_ = lean_unbox_usize(v_stop_4599_);
lean_dec(v_stop_4599_);
v_res_4608_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Rapp_foldSubgoalsM___at___00Aesop_Rapp_subgoals___at___00Aesop_Rapp_traceMetadata_spec__0_spec__0_spec__1(v_00_u03c3_4595_, v_f_4596_, v_as_4597_, v_i_boxed_4606_, v_stop_boxed_4607_, v_b_4600_, v___y_4601_, v___y_4602_, v___y_4603_, v___y_4604_);
lean_dec(v___y_4604_);
lean_dec_ref(v___y_4603_);
lean_dec(v___y_4602_);
lean_dec_ref(v___y_4601_);
lean_dec_ref(v_as_4597_);
return v_res_4608_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Rapp_foldSubgoalsM___at___00Aesop_Rapp_subgoals___at___00Aesop_Rapp_traceMetadata_spec__0_spec__0_spec__2(lean_object* v_00_u03c3_4609_, lean_object* v_f_4610_, lean_object* v_as_4611_, size_t v_i_4612_, size_t v_stop_4613_, lean_object* v_b_4614_, lean_object* v___y_4615_, lean_object* v___y_4616_, lean_object* v___y_4617_, lean_object* v___y_4618_){
_start:
{
lean_object* v___x_4620_; 
v___x_4620_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Rapp_foldSubgoalsM___at___00Aesop_Rapp_subgoals___at___00Aesop_Rapp_traceMetadata_spec__0_spec__0_spec__2___redArg(v_f_4610_, v_as_4611_, v_i_4612_, v_stop_4613_, v_b_4614_, v___y_4615_, v___y_4616_, v___y_4617_, v___y_4618_);
return v___x_4620_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Rapp_foldSubgoalsM___at___00Aesop_Rapp_subgoals___at___00Aesop_Rapp_traceMetadata_spec__0_spec__0_spec__2___boxed(lean_object* v_00_u03c3_4621_, lean_object* v_f_4622_, lean_object* v_as_4623_, lean_object* v_i_4624_, lean_object* v_stop_4625_, lean_object* v_b_4626_, lean_object* v___y_4627_, lean_object* v___y_4628_, lean_object* v___y_4629_, lean_object* v___y_4630_, lean_object* v___y_4631_){
_start:
{
size_t v_i_boxed_4632_; size_t v_stop_boxed_4633_; lean_object* v_res_4634_; 
v_i_boxed_4632_ = lean_unbox_usize(v_i_4624_);
lean_dec(v_i_4624_);
v_stop_boxed_4633_ = lean_unbox_usize(v_stop_4625_);
lean_dec(v_stop_4625_);
v_res_4634_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Rapp_foldSubgoalsM___at___00Aesop_Rapp_subgoals___at___00Aesop_Rapp_traceMetadata_spec__0_spec__0_spec__2(v_00_u03c3_4621_, v_f_4622_, v_as_4623_, v_i_boxed_4632_, v_stop_boxed_4633_, v_b_4626_, v___y_4627_, v___y_4628_, v___y_4629_, v___y_4630_);
lean_dec(v___y_4630_);
lean_dec_ref(v___y_4629_);
lean_dec(v___y_4628_);
lean_dec_ref(v___y_4627_);
lean_dec_ref(v_as_4623_);
return v_res_4634_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_traceTreeCore___lam__0(lean_object* v___y_4635_, lean_object* v___y_4636_, lean_object* v___y_4637_, lean_object* v___y_4638_, lean_object* v___y_4639_){
_start:
{
lean_object* v___x_4641_; 
v___x_4641_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4641_, 0, v___y_4635_);
return v___x_4641_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_traceTreeCore___lam__0___boxed(lean_object* v___y_4642_, lean_object* v___y_4643_, lean_object* v___y_4644_, lean_object* v___y_4645_, lean_object* v___y_4646_, lean_object* v___y_4647_){
_start:
{
lean_object* v_res_4648_; 
v_res_4648_ = lp_aesop_Aesop_Goal_traceTreeCore___lam__0(v___y_4642_, v___y_4643_, v___y_4644_, v___y_4645_, v___y_4646_);
lean_dec(v___y_4646_);
lean_dec_ref(v___y_4645_);
lean_dec(v___y_4644_);
lean_dec_ref(v___y_4643_);
return v_res_4648_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_traceTreeCore___lam__1(lean_object* v___x_4649_, lean_object* v_x_4650_, lean_object* v___y_4651_, lean_object* v___y_4652_, lean_object* v___y_4653_, lean_object* v___y_4654_){
_start:
{
lean_object* v___x_4656_; 
v___x_4656_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4656_, 0, v___x_4649_);
return v___x_4656_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_traceTreeCore___lam__1___boxed(lean_object* v___x_4657_, lean_object* v_x_4658_, lean_object* v___y_4659_, lean_object* v___y_4660_, lean_object* v___y_4661_, lean_object* v___y_4662_, lean_object* v___y_4663_){
_start:
{
lean_object* v_res_4664_; 
v_res_4664_ = lp_aesop_Aesop_Goal_traceTreeCore___lam__1(v___x_4657_, v_x_4658_, v___y_4659_, v___y_4660_, v___y_4661_, v___y_4662_);
lean_dec(v___y_4662_);
lean_dec_ref(v___y_4661_);
lean_dec(v___y_4660_);
lean_dec_ref(v___y_4659_);
lean_dec_ref(v_x_4658_);
return v_res_4664_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Rapp_forSubgoalsM___at___00Aesop_Rapp_traceTreeCore_spec__2_spec__2(lean_object* v_f_4665_, lean_object* v_as_4666_, size_t v_i_4667_, size_t v_stop_4668_, lean_object* v_b_4669_, lean_object* v___y_4670_, lean_object* v___y_4671_, lean_object* v___y_4672_, lean_object* v___y_4673_){
_start:
{
uint8_t v___x_4675_; 
v___x_4675_ = lean_usize_dec_eq(v_i_4667_, v_stop_4668_);
if (v___x_4675_ == 0)
{
lean_object* v___x_4676_; lean_object* v___x_4677_; 
v___x_4676_ = lean_array_uget_borrowed(v_as_4666_, v_i_4667_);
lean_inc_ref(v_f_4665_);
lean_inc(v___y_4673_);
lean_inc_ref(v___y_4672_);
lean_inc(v___y_4671_);
lean_inc_ref(v___y_4670_);
lean_inc(v___x_4676_);
v___x_4677_ = lean_apply_6(v_f_4665_, v___x_4676_, v___y_4670_, v___y_4671_, v___y_4672_, v___y_4673_, lean_box(0));
if (lean_obj_tag(v___x_4677_) == 0)
{
lean_object* v_a_4678_; size_t v___x_4679_; size_t v___x_4680_; 
v_a_4678_ = lean_ctor_get(v___x_4677_, 0);
lean_inc(v_a_4678_);
lean_dec_ref_known(v___x_4677_, 1);
v___x_4679_ = ((size_t)1ULL);
v___x_4680_ = lean_usize_add(v_i_4667_, v___x_4679_);
v_i_4667_ = v___x_4680_;
v_b_4669_ = v_a_4678_;
goto _start;
}
else
{
lean_dec_ref(v_f_4665_);
return v___x_4677_;
}
}
else
{
lean_object* v___x_4682_; 
lean_dec_ref(v_f_4665_);
v___x_4682_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4682_, 0, v_b_4669_);
return v___x_4682_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Rapp_forSubgoalsM___at___00Aesop_Rapp_traceTreeCore_spec__2_spec__2___boxed(lean_object* v_f_4683_, lean_object* v_as_4684_, lean_object* v_i_4685_, lean_object* v_stop_4686_, lean_object* v_b_4687_, lean_object* v___y_4688_, lean_object* v___y_4689_, lean_object* v___y_4690_, lean_object* v___y_4691_, lean_object* v___y_4692_){
_start:
{
size_t v_i_boxed_4693_; size_t v_stop_boxed_4694_; lean_object* v_res_4695_; 
v_i_boxed_4693_ = lean_unbox_usize(v_i_4685_);
lean_dec(v_i_4685_);
v_stop_boxed_4694_ = lean_unbox_usize(v_stop_4686_);
lean_dec(v_stop_4686_);
v_res_4695_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Rapp_forSubgoalsM___at___00Aesop_Rapp_traceTreeCore_spec__2_spec__2(v_f_4683_, v_as_4684_, v_i_boxed_4693_, v_stop_boxed_4694_, v_b_4687_, v___y_4688_, v___y_4689_, v___y_4690_, v___y_4691_);
lean_dec(v___y_4691_);
lean_dec_ref(v___y_4690_);
lean_dec(v___y_4689_);
lean_dec_ref(v___y_4688_);
lean_dec_ref(v_as_4684_);
return v_res_4695_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Rapp_forSubgoalsM___at___00Aesop_Rapp_traceTreeCore_spec__2_spec__3(lean_object* v_f_4696_, lean_object* v_as_4697_, size_t v_i_4698_, size_t v_stop_4699_, lean_object* v_b_4700_, lean_object* v___y_4701_, lean_object* v___y_4702_, lean_object* v___y_4703_, lean_object* v___y_4704_){
_start:
{
lean_object* v_a_4707_; lean_object* v___y_4712_; uint8_t v___x_4714_; 
v___x_4714_ = lean_usize_dec_eq(v_i_4698_, v_stop_4699_);
if (v___x_4714_ == 0)
{
lean_object* v___x_4715_; lean_object* v___x_4716_; lean_object* v___x_4717_; lean_object* v_elimMVarCluster_4718_; lean_object* v___x_4719_; lean_object* v_goals_4720_; lean_object* v___x_4721_; lean_object* v___x_4722_; lean_object* v___x_4723_; uint8_t v___x_4724_; 
v___x_4715_ = lean_array_uget_borrowed(v_as_4697_, v_i_4698_);
v___x_4716_ = lean_st_ref_get(v___x_4715_);
v___x_4717_ = lp_aesop_Aesop_treeImpl;
v_elimMVarCluster_4718_ = lean_ctor_get(v___x_4717_, 5);
lean_inc_ref(v_elimMVarCluster_4718_);
v___x_4719_ = lean_apply_1(v_elimMVarCluster_4718_, v___x_4716_);
v_goals_4720_ = lean_ctor_get(v___x_4719_, 1);
lean_inc_ref(v_goals_4720_);
lean_dec_ref(v___x_4719_);
v___x_4721_ = lean_unsigned_to_nat(0u);
v___x_4722_ = lean_array_get_size(v_goals_4720_);
v___x_4723_ = lean_box(0);
v___x_4724_ = lean_nat_dec_lt(v___x_4721_, v___x_4722_);
if (v___x_4724_ == 0)
{
lean_dec_ref(v_goals_4720_);
v_a_4707_ = v___x_4723_;
goto v___jp_4706_;
}
else
{
uint8_t v___x_4725_; 
v___x_4725_ = lean_nat_dec_le(v___x_4722_, v___x_4722_);
if (v___x_4725_ == 0)
{
if (v___x_4724_ == 0)
{
lean_dec_ref(v_goals_4720_);
v_a_4707_ = v___x_4723_;
goto v___jp_4706_;
}
else
{
size_t v___x_4726_; size_t v___x_4727_; lean_object* v___x_4728_; 
v___x_4726_ = ((size_t)0ULL);
v___x_4727_ = lean_usize_of_nat(v___x_4722_);
lean_inc_ref(v_f_4696_);
v___x_4728_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Rapp_forSubgoalsM___at___00Aesop_Rapp_traceTreeCore_spec__2_spec__2(v_f_4696_, v_goals_4720_, v___x_4726_, v___x_4727_, v___x_4723_, v___y_4701_, v___y_4702_, v___y_4703_, v___y_4704_);
lean_dec_ref(v_goals_4720_);
v___y_4712_ = v___x_4728_;
goto v___jp_4711_;
}
}
else
{
size_t v___x_4729_; size_t v___x_4730_; lean_object* v___x_4731_; 
v___x_4729_ = ((size_t)0ULL);
v___x_4730_ = lean_usize_of_nat(v___x_4722_);
lean_inc_ref(v_f_4696_);
v___x_4731_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Rapp_forSubgoalsM___at___00Aesop_Rapp_traceTreeCore_spec__2_spec__2(v_f_4696_, v_goals_4720_, v___x_4729_, v___x_4730_, v___x_4723_, v___y_4701_, v___y_4702_, v___y_4703_, v___y_4704_);
lean_dec_ref(v_goals_4720_);
v___y_4712_ = v___x_4731_;
goto v___jp_4711_;
}
}
}
else
{
lean_object* v___x_4732_; 
lean_dec_ref(v_f_4696_);
v___x_4732_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4732_, 0, v_b_4700_);
return v___x_4732_;
}
v___jp_4706_:
{
size_t v___x_4708_; size_t v___x_4709_; 
v___x_4708_ = ((size_t)1ULL);
v___x_4709_ = lean_usize_add(v_i_4698_, v___x_4708_);
v_i_4698_ = v___x_4709_;
v_b_4700_ = v_a_4707_;
goto _start;
}
v___jp_4711_:
{
if (lean_obj_tag(v___y_4712_) == 0)
{
lean_object* v_a_4713_; 
v_a_4713_ = lean_ctor_get(v___y_4712_, 0);
lean_inc(v_a_4713_);
lean_dec_ref_known(v___y_4712_, 1);
v_a_4707_ = v_a_4713_;
goto v___jp_4706_;
}
else
{
lean_dec_ref(v_f_4696_);
return v___y_4712_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Rapp_forSubgoalsM___at___00Aesop_Rapp_traceTreeCore_spec__2_spec__3___boxed(lean_object* v_f_4733_, lean_object* v_as_4734_, lean_object* v_i_4735_, lean_object* v_stop_4736_, lean_object* v_b_4737_, lean_object* v___y_4738_, lean_object* v___y_4739_, lean_object* v___y_4740_, lean_object* v___y_4741_, lean_object* v___y_4742_){
_start:
{
size_t v_i_boxed_4743_; size_t v_stop_boxed_4744_; lean_object* v_res_4745_; 
v_i_boxed_4743_ = lean_unbox_usize(v_i_4735_);
lean_dec(v_i_4735_);
v_stop_boxed_4744_ = lean_unbox_usize(v_stop_4736_);
lean_dec(v_stop_4736_);
v_res_4745_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Rapp_forSubgoalsM___at___00Aesop_Rapp_traceTreeCore_spec__2_spec__3(v_f_4733_, v_as_4734_, v_i_boxed_4743_, v_stop_boxed_4744_, v_b_4737_, v___y_4738_, v___y_4739_, v___y_4740_, v___y_4741_);
lean_dec(v___y_4741_);
lean_dec_ref(v___y_4740_);
lean_dec(v___y_4739_);
lean_dec_ref(v___y_4738_);
lean_dec_ref(v_as_4734_);
return v_res_4745_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_forSubgoalsM___at___00Aesop_Rapp_traceTreeCore_spec__2(lean_object* v_f_4746_, lean_object* v_r_4747_, lean_object* v___y_4748_, lean_object* v___y_4749_, lean_object* v___y_4750_, lean_object* v___y_4751_){
_start:
{
lean_object* v___x_4753_; lean_object* v_elimRapp_4754_; lean_object* v___x_4755_; lean_object* v_children_4756_; lean_object* v___x_4757_; lean_object* v___x_4758_; lean_object* v___x_4759_; uint8_t v___x_4760_; 
v___x_4753_ = lp_aesop_Aesop_treeImpl;
v_elimRapp_4754_ = lean_ctor_get(v___x_4753_, 3);
lean_inc_ref(v_elimRapp_4754_);
v___x_4755_ = lean_apply_1(v_elimRapp_4754_, v_r_4747_);
v_children_4756_ = lean_ctor_get(v___x_4755_, 2);
lean_inc_ref(v_children_4756_);
lean_dec_ref(v___x_4755_);
v___x_4757_ = lean_unsigned_to_nat(0u);
v___x_4758_ = lean_array_get_size(v_children_4756_);
v___x_4759_ = lean_box(0);
v___x_4760_ = lean_nat_dec_lt(v___x_4757_, v___x_4758_);
if (v___x_4760_ == 0)
{
lean_object* v___x_4761_; 
lean_dec_ref(v_children_4756_);
lean_dec_ref(v_f_4746_);
v___x_4761_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4761_, 0, v___x_4759_);
return v___x_4761_;
}
else
{
uint8_t v___x_4762_; 
v___x_4762_ = lean_nat_dec_le(v___x_4758_, v___x_4758_);
if (v___x_4762_ == 0)
{
if (v___x_4760_ == 0)
{
lean_object* v___x_4763_; 
lean_dec_ref(v_children_4756_);
lean_dec_ref(v_f_4746_);
v___x_4763_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4763_, 0, v___x_4759_);
return v___x_4763_;
}
else
{
size_t v___x_4764_; size_t v___x_4765_; lean_object* v___x_4766_; 
v___x_4764_ = ((size_t)0ULL);
v___x_4765_ = lean_usize_of_nat(v___x_4758_);
v___x_4766_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Rapp_forSubgoalsM___at___00Aesop_Rapp_traceTreeCore_spec__2_spec__3(v_f_4746_, v_children_4756_, v___x_4764_, v___x_4765_, v___x_4759_, v___y_4748_, v___y_4749_, v___y_4750_, v___y_4751_);
lean_dec_ref(v_children_4756_);
return v___x_4766_;
}
}
else
{
size_t v___x_4767_; size_t v___x_4768_; lean_object* v___x_4769_; 
v___x_4767_ = ((size_t)0ULL);
v___x_4768_ = lean_usize_of_nat(v___x_4758_);
v___x_4769_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Rapp_forSubgoalsM___at___00Aesop_Rapp_traceTreeCore_spec__2_spec__3(v_f_4746_, v_children_4756_, v___x_4767_, v___x_4768_, v___x_4759_, v___y_4748_, v___y_4749_, v___y_4750_, v___y_4751_);
lean_dec_ref(v_children_4756_);
return v___x_4769_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_forSubgoalsM___at___00Aesop_Rapp_traceTreeCore_spec__2___boxed(lean_object* v_f_4770_, lean_object* v_r_4771_, lean_object* v___y_4772_, lean_object* v___y_4773_, lean_object* v___y_4774_, lean_object* v___y_4775_, lean_object* v___y_4776_){
_start:
{
lean_object* v_res_4777_; 
v_res_4777_ = lp_aesop_Aesop_Rapp_forSubgoalsM___at___00Aesop_Rapp_traceTreeCore_spec__2(v_f_4770_, v_r_4771_, v___y_4772_, v___y_4773_, v___y_4774_, v___y_4775_);
lean_dec(v___y_4775_);
lean_dec_ref(v___y_4774_);
lean_dec(v___y_4773_);
lean_dec_ref(v___y_4772_);
return v_res_4777_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_traceTreeCore___lam__3(lean_object* v___f_4778_, lean_object* v_r_4779_, lean_object* v_traceOpt_4780_, lean_object* v_traceClass_4781_, uint8_t v___x_4782_, lean_object* v___x_4783_, lean_object* v___f_4784_, lean_object* v___y_4785_, lean_object* v___y_4786_, lean_object* v___y_4787_, lean_object* v___y_4788_){
_start:
{
lean_object* v___y_4791_; lean_object* v_options_4793_; uint8_t v_hasTrace_4794_; 
v_options_4793_ = lean_ctor_get(v___y_4787_, 2);
v_hasTrace_4794_ = lean_ctor_get_uint8(v_options_4793_, sizeof(void*)*1);
if (v_hasTrace_4794_ == 0)
{
lean_object* v___x_4795_; 
lean_dec_ref(v___f_4784_);
lean_dec_ref(v___x_4783_);
lean_dec(v_traceClass_4781_);
lean_inc(v_r_4779_);
v___x_4795_ = lp_aesop_Aesop_Rapp_traceMetadata(v_r_4779_, v_traceOpt_4780_, v___y_4785_, v___y_4786_, v___y_4787_, v___y_4788_);
v___y_4791_ = v___x_4795_;
goto v___jp_4790_;
}
else
{
lean_object* v_inheritedTraceOptions_4796_; lean_object* v___x_4797_; lean_object* v___x_4798_; uint8_t v___x_4799_; lean_object* v___y_4801_; lean_object* v___y_4802_; lean_object* v_a_4803_; lean_object* v___y_4816_; lean_object* v___y_4817_; lean_object* v_a_4818_; 
v_inheritedTraceOptions_4796_ = lean_ctor_get(v___y_4787_, 13);
v___x_4797_ = ((lean_object*)(lp_aesop_Aesop_Goal_withHeadlineTraceNode___redArg___closed__2));
lean_inc(v_traceClass_4781_);
v___x_4798_ = l_Lean_Name_append(v___x_4797_, v_traceClass_4781_);
v___x_4799_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_4796_, v_options_4793_, v___x_4798_);
lean_dec(v___x_4798_);
if (v___x_4799_ == 0)
{
lean_object* v___x_4876_; uint8_t v___x_4877_; 
v___x_4876_ = l_Lean_trace_profiler;
v___x_4877_ = lp_aesop_Lean_Option_get___at___00Aesop_Goal_withHeadlineTraceNode_spec__1(v_options_4793_, v___x_4876_);
if (v___x_4877_ == 0)
{
lean_object* v___x_4878_; 
lean_dec_ref(v___f_4784_);
lean_dec_ref(v___x_4783_);
lean_dec(v_traceClass_4781_);
lean_inc(v_r_4779_);
v___x_4878_ = lp_aesop_Aesop_Rapp_traceMetadata(v_r_4779_, v_traceOpt_4780_, v___y_4785_, v___y_4786_, v___y_4787_, v___y_4788_);
v___y_4791_ = v___x_4878_;
goto v___jp_4790_;
}
else
{
goto v___jp_4827_;
}
}
else
{
goto v___jp_4827_;
}
v___jp_4800_:
{
lean_object* v___x_4804_; double v___x_4805_; double v___x_4806_; double v___x_4807_; double v___x_4808_; double v___x_4809_; lean_object* v___x_4810_; lean_object* v___x_4811_; lean_object* v___x_4812_; lean_object* v___x_4813_; lean_object* v___x_4814_; 
v___x_4804_ = lean_io_mono_nanos_now();
v___x_4805_ = lean_float_of_nat(v___y_4802_);
v___x_4806_ = lean_float_once(&lp_aesop_Aesop_Goal_withHeadlineTraceNode___redArg___closed__3, &lp_aesop_Aesop_Goal_withHeadlineTraceNode___redArg___closed__3_once, _init_lp_aesop_Aesop_Goal_withHeadlineTraceNode___redArg___closed__3);
v___x_4807_ = lean_float_div(v___x_4805_, v___x_4806_);
v___x_4808_ = lean_float_of_nat(v___x_4804_);
v___x_4809_ = lean_float_div(v___x_4808_, v___x_4806_);
v___x_4810_ = lean_box_float(v___x_4807_);
v___x_4811_ = lean_box_float(v___x_4809_);
v___x_4812_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4812_, 0, v___x_4810_);
lean_ctor_set(v___x_4812_, 1, v___x_4811_);
v___x_4813_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4813_, 0, v_a_4803_);
lean_ctor_set(v___x_4813_, 1, v___x_4812_);
v___x_4814_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Tree_Tracing_0__Aesop_Goal_traceMetadata_trcNode_spec__0(v_traceClass_4781_, v___x_4782_, v___x_4783_, v_options_4793_, v___x_4799_, v___y_4801_, v___f_4784_, v___x_4813_, v___y_4785_, v___y_4786_, v___y_4787_, v___y_4788_);
v___y_4791_ = v___x_4814_;
goto v___jp_4790_;
}
v___jp_4815_:
{
lean_object* v___x_4819_; double v___x_4820_; double v___x_4821_; lean_object* v___x_4822_; lean_object* v___x_4823_; lean_object* v___x_4824_; lean_object* v___x_4825_; lean_object* v___x_4826_; 
v___x_4819_ = lean_io_get_num_heartbeats();
v___x_4820_ = lean_float_of_nat(v___y_4817_);
v___x_4821_ = lean_float_of_nat(v___x_4819_);
v___x_4822_ = lean_box_float(v___x_4820_);
v___x_4823_ = lean_box_float(v___x_4821_);
v___x_4824_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4824_, 0, v___x_4822_);
lean_ctor_set(v___x_4824_, 1, v___x_4823_);
v___x_4825_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4825_, 0, v_a_4818_);
lean_ctor_set(v___x_4825_, 1, v___x_4824_);
v___x_4826_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Tree_Tracing_0__Aesop_Goal_traceMetadata_trcNode_spec__0(v_traceClass_4781_, v___x_4782_, v___x_4783_, v_options_4793_, v___x_4799_, v___y_4816_, v___f_4784_, v___x_4825_, v___y_4785_, v___y_4786_, v___y_4787_, v___y_4788_);
v___y_4791_ = v___x_4826_;
goto v___jp_4790_;
}
v___jp_4827_:
{
lean_object* v___x_4828_; 
v___x_4828_ = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_Goal_withHeadlineTraceNode_spec__0___redArg(v___y_4788_);
if (lean_obj_tag(v___x_4828_) == 0)
{
lean_object* v_a_4829_; lean_object* v___x_4830_; uint8_t v___x_4831_; 
v_a_4829_ = lean_ctor_get(v___x_4828_, 0);
lean_inc(v_a_4829_);
lean_dec_ref_known(v___x_4828_, 1);
v___x_4830_ = l_Lean_trace_profiler_useHeartbeats;
v___x_4831_ = lp_aesop_Lean_Option_get___at___00Aesop_Goal_withHeadlineTraceNode_spec__1(v_options_4793_, v___x_4830_);
if (v___x_4831_ == 0)
{
lean_object* v___x_4832_; lean_object* v___x_4833_; 
v___x_4832_ = lean_io_mono_nanos_now();
lean_inc(v_r_4779_);
v___x_4833_ = lp_aesop_Aesop_Rapp_traceMetadata(v_r_4779_, v_traceOpt_4780_, v___y_4785_, v___y_4786_, v___y_4787_, v___y_4788_);
if (lean_obj_tag(v___x_4833_) == 0)
{
lean_object* v_a_4834_; lean_object* v___x_4836_; uint8_t v_isShared_4837_; uint8_t v_isSharedCheck_4841_; 
v_a_4834_ = lean_ctor_get(v___x_4833_, 0);
v_isSharedCheck_4841_ = !lean_is_exclusive(v___x_4833_);
if (v_isSharedCheck_4841_ == 0)
{
v___x_4836_ = v___x_4833_;
v_isShared_4837_ = v_isSharedCheck_4841_;
goto v_resetjp_4835_;
}
else
{
lean_inc(v_a_4834_);
lean_dec(v___x_4833_);
v___x_4836_ = lean_box(0);
v_isShared_4837_ = v_isSharedCheck_4841_;
goto v_resetjp_4835_;
}
v_resetjp_4835_:
{
lean_object* v___x_4839_; 
if (v_isShared_4837_ == 0)
{
lean_ctor_set_tag(v___x_4836_, 1);
v___x_4839_ = v___x_4836_;
goto v_reusejp_4838_;
}
else
{
lean_object* v_reuseFailAlloc_4840_; 
v_reuseFailAlloc_4840_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4840_, 0, v_a_4834_);
v___x_4839_ = v_reuseFailAlloc_4840_;
goto v_reusejp_4838_;
}
v_reusejp_4838_:
{
v___y_4801_ = v_a_4829_;
v___y_4802_ = v___x_4832_;
v_a_4803_ = v___x_4839_;
goto v___jp_4800_;
}
}
}
else
{
lean_object* v_a_4842_; lean_object* v___x_4844_; uint8_t v_isShared_4845_; uint8_t v_isSharedCheck_4849_; 
v_a_4842_ = lean_ctor_get(v___x_4833_, 0);
v_isSharedCheck_4849_ = !lean_is_exclusive(v___x_4833_);
if (v_isSharedCheck_4849_ == 0)
{
v___x_4844_ = v___x_4833_;
v_isShared_4845_ = v_isSharedCheck_4849_;
goto v_resetjp_4843_;
}
else
{
lean_inc(v_a_4842_);
lean_dec(v___x_4833_);
v___x_4844_ = lean_box(0);
v_isShared_4845_ = v_isSharedCheck_4849_;
goto v_resetjp_4843_;
}
v_resetjp_4843_:
{
lean_object* v___x_4847_; 
if (v_isShared_4845_ == 0)
{
lean_ctor_set_tag(v___x_4844_, 0);
v___x_4847_ = v___x_4844_;
goto v_reusejp_4846_;
}
else
{
lean_object* v_reuseFailAlloc_4848_; 
v_reuseFailAlloc_4848_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4848_, 0, v_a_4842_);
v___x_4847_ = v_reuseFailAlloc_4848_;
goto v_reusejp_4846_;
}
v_reusejp_4846_:
{
v___y_4801_ = v_a_4829_;
v___y_4802_ = v___x_4832_;
v_a_4803_ = v___x_4847_;
goto v___jp_4800_;
}
}
}
}
else
{
lean_object* v___x_4850_; lean_object* v___x_4851_; 
v___x_4850_ = lean_io_get_num_heartbeats();
lean_inc(v_r_4779_);
v___x_4851_ = lp_aesop_Aesop_Rapp_traceMetadata(v_r_4779_, v_traceOpt_4780_, v___y_4785_, v___y_4786_, v___y_4787_, v___y_4788_);
if (lean_obj_tag(v___x_4851_) == 0)
{
lean_object* v_a_4852_; lean_object* v___x_4854_; uint8_t v_isShared_4855_; uint8_t v_isSharedCheck_4859_; 
v_a_4852_ = lean_ctor_get(v___x_4851_, 0);
v_isSharedCheck_4859_ = !lean_is_exclusive(v___x_4851_);
if (v_isSharedCheck_4859_ == 0)
{
v___x_4854_ = v___x_4851_;
v_isShared_4855_ = v_isSharedCheck_4859_;
goto v_resetjp_4853_;
}
else
{
lean_inc(v_a_4852_);
lean_dec(v___x_4851_);
v___x_4854_ = lean_box(0);
v_isShared_4855_ = v_isSharedCheck_4859_;
goto v_resetjp_4853_;
}
v_resetjp_4853_:
{
lean_object* v___x_4857_; 
if (v_isShared_4855_ == 0)
{
lean_ctor_set_tag(v___x_4854_, 1);
v___x_4857_ = v___x_4854_;
goto v_reusejp_4856_;
}
else
{
lean_object* v_reuseFailAlloc_4858_; 
v_reuseFailAlloc_4858_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4858_, 0, v_a_4852_);
v___x_4857_ = v_reuseFailAlloc_4858_;
goto v_reusejp_4856_;
}
v_reusejp_4856_:
{
v___y_4816_ = v_a_4829_;
v___y_4817_ = v___x_4850_;
v_a_4818_ = v___x_4857_;
goto v___jp_4815_;
}
}
}
else
{
lean_object* v_a_4860_; lean_object* v___x_4862_; uint8_t v_isShared_4863_; uint8_t v_isSharedCheck_4867_; 
v_a_4860_ = lean_ctor_get(v___x_4851_, 0);
v_isSharedCheck_4867_ = !lean_is_exclusive(v___x_4851_);
if (v_isSharedCheck_4867_ == 0)
{
v___x_4862_ = v___x_4851_;
v_isShared_4863_ = v_isSharedCheck_4867_;
goto v_resetjp_4861_;
}
else
{
lean_inc(v_a_4860_);
lean_dec(v___x_4851_);
v___x_4862_ = lean_box(0);
v_isShared_4863_ = v_isSharedCheck_4867_;
goto v_resetjp_4861_;
}
v_resetjp_4861_:
{
lean_object* v___x_4865_; 
if (v_isShared_4863_ == 0)
{
lean_ctor_set_tag(v___x_4862_, 0);
v___x_4865_ = v___x_4862_;
goto v_reusejp_4864_;
}
else
{
lean_object* v_reuseFailAlloc_4866_; 
v_reuseFailAlloc_4866_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4866_, 0, v_a_4860_);
v___x_4865_ = v_reuseFailAlloc_4866_;
goto v_reusejp_4864_;
}
v_reusejp_4864_:
{
v___y_4816_ = v_a_4829_;
v___y_4817_ = v___x_4850_;
v_a_4818_ = v___x_4865_;
goto v___jp_4815_;
}
}
}
}
}
else
{
lean_object* v_a_4868_; lean_object* v___x_4870_; uint8_t v_isShared_4871_; uint8_t v_isSharedCheck_4875_; 
lean_dec_ref(v___f_4784_);
lean_dec_ref(v___x_4783_);
lean_dec(v_traceClass_4781_);
lean_dec_ref(v_traceOpt_4780_);
lean_dec(v_r_4779_);
lean_dec_ref(v___f_4778_);
v_a_4868_ = lean_ctor_get(v___x_4828_, 0);
v_isSharedCheck_4875_ = !lean_is_exclusive(v___x_4828_);
if (v_isSharedCheck_4875_ == 0)
{
v___x_4870_ = v___x_4828_;
v_isShared_4871_ = v_isSharedCheck_4875_;
goto v_resetjp_4869_;
}
else
{
lean_inc(v_a_4868_);
lean_dec(v___x_4828_);
v___x_4870_ = lean_box(0);
v_isShared_4871_ = v_isSharedCheck_4875_;
goto v_resetjp_4869_;
}
v_resetjp_4869_:
{
lean_object* v___x_4873_; 
if (v_isShared_4871_ == 0)
{
v___x_4873_ = v___x_4870_;
goto v_reusejp_4872_;
}
else
{
lean_object* v_reuseFailAlloc_4874_; 
v_reuseFailAlloc_4874_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4874_, 0, v_a_4868_);
v___x_4873_ = v_reuseFailAlloc_4874_;
goto v_reusejp_4872_;
}
v_reusejp_4872_:
{
return v___x_4873_;
}
}
}
}
}
v___jp_4790_:
{
if (lean_obj_tag(v___y_4791_) == 0)
{
lean_object* v___x_4792_; 
lean_dec_ref_known(v___y_4791_, 1);
v___x_4792_ = lp_aesop_Aesop_Rapp_forSubgoalsM___at___00Aesop_Rapp_traceTreeCore_spec__2(v___f_4778_, v_r_4779_, v___y_4785_, v___y_4786_, v___y_4787_, v___y_4788_);
return v___x_4792_;
}
else
{
lean_dec(v_r_4779_);
lean_dec_ref(v___f_4778_);
return v___y_4791_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_traceTreeCore___lam__3___boxed(lean_object* v___f_4879_, lean_object* v_r_4880_, lean_object* v_traceOpt_4881_, lean_object* v_traceClass_4882_, lean_object* v___x_4883_, lean_object* v___x_4884_, lean_object* v___f_4885_, lean_object* v___y_4886_, lean_object* v___y_4887_, lean_object* v___y_4888_, lean_object* v___y_4889_, lean_object* v___y_4890_){
_start:
{
uint8_t v___x_9808__boxed_4891_; lean_object* v_res_4892_; 
v___x_9808__boxed_4891_ = lean_unbox(v___x_4883_);
v_res_4892_ = lp_aesop_Aesop_Rapp_traceTreeCore___lam__3(v___f_4879_, v_r_4880_, v_traceOpt_4881_, v_traceClass_4882_, v___x_9808__boxed_4891_, v___x_4884_, v___f_4885_, v___y_4886_, v___y_4887_, v___y_4888_, v___y_4889_);
lean_dec(v___y_4889_);
lean_dec_ref(v___y_4888_);
lean_dec(v___y_4887_);
lean_dec_ref(v___y_4886_);
return v_res_4892_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_traceTreeCore___lam__0___boxed(lean_object* v_traceOpt_4894_, lean_object* v_gref_4895_, lean_object* v___y_4896_, lean_object* v___y_4897_, lean_object* v___y_4898_, lean_object* v___y_4899_, lean_object* v___y_4900_){
_start:
{
lean_object* v_res_4901_; 
v_res_4901_ = lp_aesop_Aesop_Rapp_traceTreeCore___lam__0(v_traceOpt_4894_, v_gref_4895_, v___y_4896_, v___y_4897_, v___y_4898_, v___y_4899_);
lean_dec(v___y_4899_);
lean_dec_ref(v___y_4898_);
lean_dec(v___y_4897_);
lean_dec_ref(v___y_4896_);
lean_dec(v_gref_4895_);
return v_res_4901_;
}
}
static lean_object* _init_lp_aesop_Aesop_Rapp_traceTreeCore___closed__3(void){
_start:
{
lean_object* v___x_4905_; lean_object* v___x_4906_; 
v___x_4905_ = ((lean_object*)(lp_aesop_Aesop_Rapp_traceTreeCore___closed__2));
v___x_4906_ = l_Lean_MessageData_ofFormat(v___x_4905_);
return v___x_4906_;
}
}
static lean_object* _init_lp_aesop_Aesop_Rapp_traceTreeCore___closed__4(void){
_start:
{
lean_object* v___x_4907_; lean_object* v___f_4908_; 
v___x_4907_ = lean_obj_once(&lp_aesop_Aesop_Rapp_traceTreeCore___closed__3, &lp_aesop_Aesop_Rapp_traceTreeCore___closed__3_once, _init_lp_aesop_Aesop_Rapp_traceTreeCore___closed__3);
v___f_4908_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Goal_traceTreeCore___lam__1___boxed), 7, 1);
lean_closure_set(v___f_4908_, 0, v___x_4907_);
return v___f_4908_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_traceTreeCore(lean_object* v_r_4909_, lean_object* v_traceOpt_4910_, lean_object* v_a_4911_, lean_object* v_a_4912_, lean_object* v_a_4913_, lean_object* v_a_4914_){
_start:
{
lean_object* v_traceClass_4916_; lean_object* v___f_4917_; lean_object* v___f_4918_; lean_object* v___f_4919_; uint8_t v___x_4920_; lean_object* v___x_4921_; lean_object* v___x_4922_; lean_object* v___f_4923_; uint8_t v___x_4924_; lean_object* v___x_4925_; 
v_traceClass_4916_ = lean_ctor_get(v_traceOpt_4910_, 0);
lean_inc_ref_n(v_traceOpt_4910_, 2);
v___f_4917_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Rapp_traceTreeCore___lam__0___boxed), 7, 1);
lean_closure_set(v___f_4917_, 0, v_traceOpt_4910_);
v___f_4918_ = ((lean_object*)(lp_aesop_Aesop_Rapp_traceTreeCore___closed__0));
v___f_4919_ = lean_obj_once(&lp_aesop_Aesop_Rapp_traceTreeCore___closed__4, &lp_aesop_Aesop_Rapp_traceTreeCore___closed__4_once, _init_lp_aesop_Aesop_Rapp_traceTreeCore___closed__4);
v___x_4920_ = 1;
v___x_4921_ = ((lean_object*)(lp_aesop_Aesop_Goal_withHeadlineTraceNode___redArg___closed__0));
v___x_4922_ = lean_box(v___x_4920_);
lean_inc(v_traceClass_4916_);
lean_inc(v_r_4909_);
v___f_4923_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Rapp_traceTreeCore___lam__3___boxed), 12, 7);
lean_closure_set(v___f_4923_, 0, v___f_4917_);
lean_closure_set(v___f_4923_, 1, v_r_4909_);
lean_closure_set(v___f_4923_, 2, v_traceOpt_4910_);
lean_closure_set(v___f_4923_, 3, v_traceClass_4916_);
lean_closure_set(v___f_4923_, 4, v___x_4922_);
lean_closure_set(v___f_4923_, 5, v___x_4921_);
lean_closure_set(v___f_4923_, 6, v___f_4919_);
v___x_4924_ = 0;
v___x_4925_ = lp_aesop_Aesop_Rapp_withHeadlineTraceNode___redArg(v_r_4909_, v_traceOpt_4910_, v___f_4923_, v___x_4924_, v___f_4918_, v_a_4911_, v_a_4912_, v_a_4913_, v_a_4914_);
return v___x_4925_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceTreeCore_spec__0(lean_object* v_traceOpt_4926_, lean_object* v_as_4927_, size_t v_sz_4928_, size_t v_i_4929_, lean_object* v_b_4930_, lean_object* v___y_4931_, lean_object* v___y_4932_, lean_object* v___y_4933_, lean_object* v___y_4934_){
_start:
{
uint8_t v___x_4936_; 
v___x_4936_ = lean_usize_dec_lt(v_i_4929_, v_sz_4928_);
if (v___x_4936_ == 0)
{
lean_object* v___x_4937_; 
lean_dec_ref(v_traceOpt_4926_);
v___x_4937_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4937_, 0, v_b_4930_);
return v___x_4937_;
}
else
{
lean_object* v_a_4938_; lean_object* v___x_4939_; lean_object* v___x_4940_; 
v_a_4938_ = lean_array_uget_borrowed(v_as_4927_, v_i_4929_);
v___x_4939_ = lean_st_ref_get(v_a_4938_);
lean_inc_ref(v_traceOpt_4926_);
v___x_4940_ = lp_aesop_Aesop_Rapp_traceTreeCore(v___x_4939_, v_traceOpt_4926_, v___y_4931_, v___y_4932_, v___y_4933_, v___y_4934_);
if (lean_obj_tag(v___x_4940_) == 0)
{
lean_object* v___x_4941_; size_t v___x_4942_; size_t v___x_4943_; 
lean_dec_ref_known(v___x_4940_, 1);
v___x_4941_ = lean_box(0);
v___x_4942_ = ((size_t)1ULL);
v___x_4943_ = lean_usize_add(v_i_4929_, v___x_4942_);
v_i_4929_ = v___x_4943_;
v_b_4930_ = v___x_4941_;
goto _start;
}
else
{
lean_dec_ref(v_traceOpt_4926_);
return v___x_4940_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_traceTreeCore___lam__2(lean_object* v___x_4945_, lean_object* v_g_4946_, lean_object* v_children_4947_, lean_object* v_traceOpt_4948_, lean_object* v_traceClass_4949_, lean_object* v___y_4950_, lean_object* v___y_4951_, lean_object* v___y_4952_, lean_object* v___y_4953_){
_start:
{
lean_object* v___y_4956_; lean_object* v___x_4969_; 
lean_inc(v_g_4946_);
v___x_4969_ = lp_aesop_Aesop_Goal_runMetaMInParentState_x27___at___00__private_Aesop_Tree_Tracing_0__Aesop_Goal_withHeadlineTraceNode_fmt_spec__2___redArg(v___x_4945_, v_g_4946_, v___y_4950_, v___y_4951_, v___y_4952_, v___y_4953_);
if (lean_obj_tag(v___x_4969_) == 0)
{
lean_object* v_options_4970_; uint8_t v_hasTrace_4971_; 
lean_dec_ref_known(v___x_4969_, 1);
v_options_4970_ = lean_ctor_get(v___y_4952_, 2);
v_hasTrace_4971_ = lean_ctor_get_uint8(v_options_4970_, sizeof(void*)*1);
if (v_hasTrace_4971_ == 0)
{
lean_object* v___x_4972_; 
lean_dec(v_traceClass_4949_);
lean_inc_ref(v_traceOpt_4948_);
v___x_4972_ = lp_aesop_Aesop_Goal_traceMetadata(v_g_4946_, v_traceOpt_4948_, v___y_4950_, v___y_4951_, v___y_4952_, v___y_4953_);
v___y_4956_ = v___x_4972_;
goto v___jp_4955_;
}
else
{
lean_object* v_inheritedTraceOptions_4973_; lean_object* v___f_4974_; lean_object* v___x_4975_; lean_object* v___x_4976_; lean_object* v___x_4977_; uint8_t v___x_4978_; lean_object* v___y_4980_; lean_object* v___y_4981_; lean_object* v_a_4982_; lean_object* v___y_4995_; lean_object* v___y_4996_; lean_object* v_a_4997_; 
v_inheritedTraceOptions_4973_ = lean_ctor_get(v___y_4952_, 13);
v___f_4974_ = lean_obj_once(&lp_aesop_Aesop_Rapp_traceTreeCore___closed__4, &lp_aesop_Aesop_Rapp_traceTreeCore___closed__4_once, _init_lp_aesop_Aesop_Rapp_traceTreeCore___closed__4);
v___x_4975_ = ((lean_object*)(lp_aesop_Aesop_Goal_withHeadlineTraceNode___redArg___closed__0));
v___x_4976_ = ((lean_object*)(lp_aesop_Aesop_Goal_withHeadlineTraceNode___redArg___closed__2));
lean_inc(v_traceClass_4949_);
v___x_4977_ = l_Lean_Name_append(v___x_4976_, v_traceClass_4949_);
v___x_4978_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_4973_, v_options_4970_, v___x_4977_);
lean_dec(v___x_4977_);
if (v___x_4978_ == 0)
{
lean_object* v___x_5055_; uint8_t v___x_5056_; 
v___x_5055_ = l_Lean_trace_profiler;
v___x_5056_ = lp_aesop_Lean_Option_get___at___00Aesop_Goal_withHeadlineTraceNode_spec__1(v_options_4970_, v___x_5055_);
if (v___x_5056_ == 0)
{
lean_object* v___x_5057_; 
lean_dec(v_traceClass_4949_);
lean_inc_ref(v_traceOpt_4948_);
v___x_5057_ = lp_aesop_Aesop_Goal_traceMetadata(v_g_4946_, v_traceOpt_4948_, v___y_4950_, v___y_4951_, v___y_4952_, v___y_4953_);
v___y_4956_ = v___x_5057_;
goto v___jp_4955_;
}
else
{
goto v___jp_5006_;
}
}
else
{
goto v___jp_5006_;
}
v___jp_4979_:
{
lean_object* v___x_4983_; double v___x_4984_; double v___x_4985_; double v___x_4986_; double v___x_4987_; double v___x_4988_; lean_object* v___x_4989_; lean_object* v___x_4990_; lean_object* v___x_4991_; lean_object* v___x_4992_; lean_object* v___x_4993_; 
v___x_4983_ = lean_io_mono_nanos_now();
v___x_4984_ = lean_float_of_nat(v___y_4981_);
v___x_4985_ = lean_float_once(&lp_aesop_Aesop_Goal_withHeadlineTraceNode___redArg___closed__3, &lp_aesop_Aesop_Goal_withHeadlineTraceNode___redArg___closed__3_once, _init_lp_aesop_Aesop_Goal_withHeadlineTraceNode___redArg___closed__3);
v___x_4986_ = lean_float_div(v___x_4984_, v___x_4985_);
v___x_4987_ = lean_float_of_nat(v___x_4983_);
v___x_4988_ = lean_float_div(v___x_4987_, v___x_4985_);
v___x_4989_ = lean_box_float(v___x_4986_);
v___x_4990_ = lean_box_float(v___x_4988_);
v___x_4991_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4991_, 0, v___x_4989_);
lean_ctor_set(v___x_4991_, 1, v___x_4990_);
v___x_4992_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4992_, 0, v_a_4982_);
lean_ctor_set(v___x_4992_, 1, v___x_4991_);
v___x_4993_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Tree_Tracing_0__Aesop_Goal_traceMetadata_trcNode_spec__0(v_traceClass_4949_, v_hasTrace_4971_, v___x_4975_, v_options_4970_, v___x_4978_, v___y_4980_, v___f_4974_, v___x_4992_, v___y_4950_, v___y_4951_, v___y_4952_, v___y_4953_);
v___y_4956_ = v___x_4993_;
goto v___jp_4955_;
}
v___jp_4994_:
{
lean_object* v___x_4998_; double v___x_4999_; double v___x_5000_; lean_object* v___x_5001_; lean_object* v___x_5002_; lean_object* v___x_5003_; lean_object* v___x_5004_; lean_object* v___x_5005_; 
v___x_4998_ = lean_io_get_num_heartbeats();
v___x_4999_ = lean_float_of_nat(v___y_4996_);
v___x_5000_ = lean_float_of_nat(v___x_4998_);
v___x_5001_ = lean_box_float(v___x_4999_);
v___x_5002_ = lean_box_float(v___x_5000_);
v___x_5003_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_5003_, 0, v___x_5001_);
lean_ctor_set(v___x_5003_, 1, v___x_5002_);
v___x_5004_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_5004_, 0, v_a_4997_);
lean_ctor_set(v___x_5004_, 1, v___x_5003_);
v___x_5005_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Tree_Tracing_0__Aesop_Goal_traceMetadata_trcNode_spec__0(v_traceClass_4949_, v_hasTrace_4971_, v___x_4975_, v_options_4970_, v___x_4978_, v___y_4995_, v___f_4974_, v___x_5004_, v___y_4950_, v___y_4951_, v___y_4952_, v___y_4953_);
v___y_4956_ = v___x_5005_;
goto v___jp_4955_;
}
v___jp_5006_:
{
lean_object* v___x_5007_; 
v___x_5007_ = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_Goal_withHeadlineTraceNode_spec__0___redArg(v___y_4953_);
if (lean_obj_tag(v___x_5007_) == 0)
{
lean_object* v_a_5008_; lean_object* v___x_5009_; uint8_t v___x_5010_; 
v_a_5008_ = lean_ctor_get(v___x_5007_, 0);
lean_inc(v_a_5008_);
lean_dec_ref_known(v___x_5007_, 1);
v___x_5009_ = l_Lean_trace_profiler_useHeartbeats;
v___x_5010_ = lp_aesop_Lean_Option_get___at___00Aesop_Goal_withHeadlineTraceNode_spec__1(v_options_4970_, v___x_5009_);
if (v___x_5010_ == 0)
{
lean_object* v___x_5011_; lean_object* v___x_5012_; 
v___x_5011_ = lean_io_mono_nanos_now();
lean_inc_ref(v_traceOpt_4948_);
v___x_5012_ = lp_aesop_Aesop_Goal_traceMetadata(v_g_4946_, v_traceOpt_4948_, v___y_4950_, v___y_4951_, v___y_4952_, v___y_4953_);
if (lean_obj_tag(v___x_5012_) == 0)
{
lean_object* v_a_5013_; lean_object* v___x_5015_; uint8_t v_isShared_5016_; uint8_t v_isSharedCheck_5020_; 
v_a_5013_ = lean_ctor_get(v___x_5012_, 0);
v_isSharedCheck_5020_ = !lean_is_exclusive(v___x_5012_);
if (v_isSharedCheck_5020_ == 0)
{
v___x_5015_ = v___x_5012_;
v_isShared_5016_ = v_isSharedCheck_5020_;
goto v_resetjp_5014_;
}
else
{
lean_inc(v_a_5013_);
lean_dec(v___x_5012_);
v___x_5015_ = lean_box(0);
v_isShared_5016_ = v_isSharedCheck_5020_;
goto v_resetjp_5014_;
}
v_resetjp_5014_:
{
lean_object* v___x_5018_; 
if (v_isShared_5016_ == 0)
{
lean_ctor_set_tag(v___x_5015_, 1);
v___x_5018_ = v___x_5015_;
goto v_reusejp_5017_;
}
else
{
lean_object* v_reuseFailAlloc_5019_; 
v_reuseFailAlloc_5019_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5019_, 0, v_a_5013_);
v___x_5018_ = v_reuseFailAlloc_5019_;
goto v_reusejp_5017_;
}
v_reusejp_5017_:
{
v___y_4980_ = v_a_5008_;
v___y_4981_ = v___x_5011_;
v_a_4982_ = v___x_5018_;
goto v___jp_4979_;
}
}
}
else
{
lean_object* v_a_5021_; lean_object* v___x_5023_; uint8_t v_isShared_5024_; uint8_t v_isSharedCheck_5028_; 
v_a_5021_ = lean_ctor_get(v___x_5012_, 0);
v_isSharedCheck_5028_ = !lean_is_exclusive(v___x_5012_);
if (v_isSharedCheck_5028_ == 0)
{
v___x_5023_ = v___x_5012_;
v_isShared_5024_ = v_isSharedCheck_5028_;
goto v_resetjp_5022_;
}
else
{
lean_inc(v_a_5021_);
lean_dec(v___x_5012_);
v___x_5023_ = lean_box(0);
v_isShared_5024_ = v_isSharedCheck_5028_;
goto v_resetjp_5022_;
}
v_resetjp_5022_:
{
lean_object* v___x_5026_; 
if (v_isShared_5024_ == 0)
{
lean_ctor_set_tag(v___x_5023_, 0);
v___x_5026_ = v___x_5023_;
goto v_reusejp_5025_;
}
else
{
lean_object* v_reuseFailAlloc_5027_; 
v_reuseFailAlloc_5027_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5027_, 0, v_a_5021_);
v___x_5026_ = v_reuseFailAlloc_5027_;
goto v_reusejp_5025_;
}
v_reusejp_5025_:
{
v___y_4980_ = v_a_5008_;
v___y_4981_ = v___x_5011_;
v_a_4982_ = v___x_5026_;
goto v___jp_4979_;
}
}
}
}
else
{
lean_object* v___x_5029_; lean_object* v___x_5030_; 
v___x_5029_ = lean_io_get_num_heartbeats();
lean_inc_ref(v_traceOpt_4948_);
v___x_5030_ = lp_aesop_Aesop_Goal_traceMetadata(v_g_4946_, v_traceOpt_4948_, v___y_4950_, v___y_4951_, v___y_4952_, v___y_4953_);
if (lean_obj_tag(v___x_5030_) == 0)
{
lean_object* v_a_5031_; lean_object* v___x_5033_; uint8_t v_isShared_5034_; uint8_t v_isSharedCheck_5038_; 
v_a_5031_ = lean_ctor_get(v___x_5030_, 0);
v_isSharedCheck_5038_ = !lean_is_exclusive(v___x_5030_);
if (v_isSharedCheck_5038_ == 0)
{
v___x_5033_ = v___x_5030_;
v_isShared_5034_ = v_isSharedCheck_5038_;
goto v_resetjp_5032_;
}
else
{
lean_inc(v_a_5031_);
lean_dec(v___x_5030_);
v___x_5033_ = lean_box(0);
v_isShared_5034_ = v_isSharedCheck_5038_;
goto v_resetjp_5032_;
}
v_resetjp_5032_:
{
lean_object* v___x_5036_; 
if (v_isShared_5034_ == 0)
{
lean_ctor_set_tag(v___x_5033_, 1);
v___x_5036_ = v___x_5033_;
goto v_reusejp_5035_;
}
else
{
lean_object* v_reuseFailAlloc_5037_; 
v_reuseFailAlloc_5037_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5037_, 0, v_a_5031_);
v___x_5036_ = v_reuseFailAlloc_5037_;
goto v_reusejp_5035_;
}
v_reusejp_5035_:
{
v___y_4995_ = v_a_5008_;
v___y_4996_ = v___x_5029_;
v_a_4997_ = v___x_5036_;
goto v___jp_4994_;
}
}
}
else
{
lean_object* v_a_5039_; lean_object* v___x_5041_; uint8_t v_isShared_5042_; uint8_t v_isSharedCheck_5046_; 
v_a_5039_ = lean_ctor_get(v___x_5030_, 0);
v_isSharedCheck_5046_ = !lean_is_exclusive(v___x_5030_);
if (v_isSharedCheck_5046_ == 0)
{
v___x_5041_ = v___x_5030_;
v_isShared_5042_ = v_isSharedCheck_5046_;
goto v_resetjp_5040_;
}
else
{
lean_inc(v_a_5039_);
lean_dec(v___x_5030_);
v___x_5041_ = lean_box(0);
v_isShared_5042_ = v_isSharedCheck_5046_;
goto v_resetjp_5040_;
}
v_resetjp_5040_:
{
lean_object* v___x_5044_; 
if (v_isShared_5042_ == 0)
{
lean_ctor_set_tag(v___x_5041_, 0);
v___x_5044_ = v___x_5041_;
goto v_reusejp_5043_;
}
else
{
lean_object* v_reuseFailAlloc_5045_; 
v_reuseFailAlloc_5045_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5045_, 0, v_a_5039_);
v___x_5044_ = v_reuseFailAlloc_5045_;
goto v_reusejp_5043_;
}
v_reusejp_5043_:
{
v___y_4995_ = v_a_5008_;
v___y_4996_ = v___x_5029_;
v_a_4997_ = v___x_5044_;
goto v___jp_4994_;
}
}
}
}
}
else
{
lean_object* v_a_5047_; lean_object* v___x_5049_; uint8_t v_isShared_5050_; uint8_t v_isSharedCheck_5054_; 
lean_dec(v_traceClass_4949_);
lean_dec_ref(v_traceOpt_4948_);
lean_dec(v_g_4946_);
v_a_5047_ = lean_ctor_get(v___x_5007_, 0);
v_isSharedCheck_5054_ = !lean_is_exclusive(v___x_5007_);
if (v_isSharedCheck_5054_ == 0)
{
v___x_5049_ = v___x_5007_;
v_isShared_5050_ = v_isSharedCheck_5054_;
goto v_resetjp_5048_;
}
else
{
lean_inc(v_a_5047_);
lean_dec(v___x_5007_);
v___x_5049_ = lean_box(0);
v_isShared_5050_ = v_isSharedCheck_5054_;
goto v_resetjp_5048_;
}
v_resetjp_5048_:
{
lean_object* v___x_5052_; 
if (v_isShared_5050_ == 0)
{
v___x_5052_ = v___x_5049_;
goto v_reusejp_5051_;
}
else
{
lean_object* v_reuseFailAlloc_5053_; 
v_reuseFailAlloc_5053_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5053_, 0, v_a_5047_);
v___x_5052_ = v_reuseFailAlloc_5053_;
goto v_reusejp_5051_;
}
v_reusejp_5051_:
{
return v___x_5052_;
}
}
}
}
}
}
else
{
lean_dec(v_traceClass_4949_);
lean_dec_ref(v_traceOpt_4948_);
lean_dec(v_g_4946_);
return v___x_4969_;
}
v___jp_4955_:
{
if (lean_obj_tag(v___y_4956_) == 0)
{
lean_object* v___x_4957_; size_t v_sz_4958_; size_t v___x_4959_; lean_object* v___x_4960_; 
lean_dec_ref_known(v___y_4956_, 1);
v___x_4957_ = lean_box(0);
v_sz_4958_ = lean_array_size(v_children_4947_);
v___x_4959_ = ((size_t)0ULL);
v___x_4960_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceTreeCore_spec__0(v_traceOpt_4948_, v_children_4947_, v_sz_4958_, v___x_4959_, v___x_4957_, v___y_4950_, v___y_4951_, v___y_4952_, v___y_4953_);
if (lean_obj_tag(v___x_4960_) == 0)
{
lean_object* v___x_4962_; uint8_t v_isShared_4963_; uint8_t v_isSharedCheck_4967_; 
v_isSharedCheck_4967_ = !lean_is_exclusive(v___x_4960_);
if (v_isSharedCheck_4967_ == 0)
{
lean_object* v_unused_4968_; 
v_unused_4968_ = lean_ctor_get(v___x_4960_, 0);
lean_dec(v_unused_4968_);
v___x_4962_ = v___x_4960_;
v_isShared_4963_ = v_isSharedCheck_4967_;
goto v_resetjp_4961_;
}
else
{
lean_dec(v___x_4960_);
v___x_4962_ = lean_box(0);
v_isShared_4963_ = v_isSharedCheck_4967_;
goto v_resetjp_4961_;
}
v_resetjp_4961_:
{
lean_object* v___x_4965_; 
if (v_isShared_4963_ == 0)
{
lean_ctor_set(v___x_4962_, 0, v___x_4957_);
v___x_4965_ = v___x_4962_;
goto v_reusejp_4964_;
}
else
{
lean_object* v_reuseFailAlloc_4966_; 
v_reuseFailAlloc_4966_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4966_, 0, v___x_4957_);
v___x_4965_ = v_reuseFailAlloc_4966_;
goto v_reusejp_4964_;
}
v_reusejp_4964_:
{
return v___x_4965_;
}
}
}
else
{
return v___x_4960_;
}
}
else
{
lean_dec_ref(v_traceOpt_4948_);
return v___y_4956_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_traceTreeCore___lam__2___boxed(lean_object* v___x_5058_, lean_object* v_g_5059_, lean_object* v_children_5060_, lean_object* v_traceOpt_5061_, lean_object* v_traceClass_5062_, lean_object* v___y_5063_, lean_object* v___y_5064_, lean_object* v___y_5065_, lean_object* v___y_5066_, lean_object* v___y_5067_){
_start:
{
lean_object* v_res_5068_; 
v_res_5068_ = lp_aesop_Aesop_Goal_traceTreeCore___lam__2(v___x_5058_, v_g_5059_, v_children_5060_, v_traceOpt_5061_, v_traceClass_5062_, v___y_5063_, v___y_5064_, v___y_5065_, v___y_5066_);
lean_dec(v___y_5066_);
lean_dec_ref(v___y_5065_);
lean_dec(v___y_5064_);
lean_dec_ref(v___y_5063_);
lean_dec_ref(v_children_5060_);
return v_res_5068_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_traceTreeCore(lean_object* v_g_5069_, lean_object* v_traceOpt_5070_, lean_object* v_a_5071_, lean_object* v_a_5072_, lean_object* v_a_5073_, lean_object* v_a_5074_){
_start:
{
lean_object* v_traceClass_5076_; lean_object* v___x_5077_; lean_object* v_elimGoal_5078_; lean_object* v___x_5079_; lean_object* v_children_5080_; lean_object* v_preNormGoal_5081_; lean_object* v___f_5082_; lean_object* v___x_5083_; lean_object* v___x_5084_; lean_object* v___f_5085_; uint8_t v___x_5086_; lean_object* v___x_5087_; 
v_traceClass_5076_ = lean_ctor_get(v_traceOpt_5070_, 0);
v___x_5077_ = lp_aesop_Aesop_treeImpl;
v_elimGoal_5078_ = lean_ctor_get(v___x_5077_, 1);
lean_inc_ref(v_elimGoal_5078_);
lean_inc_n(v_g_5069_, 2);
v___x_5079_ = lean_apply_1(v_elimGoal_5078_, v_g_5069_);
v_children_5080_ = lean_ctor_get(v___x_5079_, 2);
lean_inc_ref(v_children_5080_);
v_preNormGoal_5081_ = lean_ctor_get(v___x_5079_, 5);
lean_inc(v_preNormGoal_5081_);
lean_dec_ref(v___x_5079_);
v___f_5082_ = ((lean_object*)(lp_aesop_Aesop_Rapp_traceTreeCore___closed__0));
v___x_5083_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_5083_, 0, v_preNormGoal_5081_);
lean_inc_n(v_traceClass_5076_, 2);
v___x_5084_ = lean_alloc_closure((void*)(lp_aesop_Lean_addTrace___at___00__private_Aesop_Tree_Tracing_0__Aesop_Goal_traceMetadata_trc_spec__0___boxed), 7, 2);
lean_closure_set(v___x_5084_, 0, v_traceClass_5076_);
lean_closure_set(v___x_5084_, 1, v___x_5083_);
lean_inc_ref(v_traceOpt_5070_);
v___f_5085_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Goal_traceTreeCore___lam__2___boxed), 10, 5);
lean_closure_set(v___f_5085_, 0, v___x_5084_);
lean_closure_set(v___f_5085_, 1, v_g_5069_);
lean_closure_set(v___f_5085_, 2, v_children_5080_);
lean_closure_set(v___f_5085_, 3, v_traceOpt_5070_);
lean_closure_set(v___f_5085_, 4, v_traceClass_5076_);
v___x_5086_ = 0;
v___x_5087_ = lp_aesop_Aesop_Goal_withHeadlineTraceNode___redArg(v_g_5069_, v_traceOpt_5070_, v___f_5085_, v___x_5086_, v___f_5082_, v_a_5071_, v_a_5072_, v_a_5073_, v_a_5074_);
return v___x_5087_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_traceTreeCore___lam__0(lean_object* v_traceOpt_5088_, lean_object* v_gref_5089_, lean_object* v___y_5090_, lean_object* v___y_5091_, lean_object* v___y_5092_, lean_object* v___y_5093_){
_start:
{
lean_object* v___x_5095_; lean_object* v___x_5096_; 
v___x_5095_ = lean_st_ref_get(v_gref_5089_);
v___x_5096_ = lp_aesop_Aesop_Goal_traceTreeCore(v___x_5095_, v_traceOpt_5088_, v___y_5090_, v___y_5091_, v___y_5092_, v___y_5093_);
return v___x_5096_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_traceTreeCore___boxed(lean_object* v_r_5097_, lean_object* v_traceOpt_5098_, lean_object* v_a_5099_, lean_object* v_a_5100_, lean_object* v_a_5101_, lean_object* v_a_5102_, lean_object* v_a_5103_){
_start:
{
lean_object* v_res_5104_; 
v_res_5104_ = lp_aesop_Aesop_Rapp_traceTreeCore(v_r_5097_, v_traceOpt_5098_, v_a_5099_, v_a_5100_, v_a_5101_, v_a_5102_);
lean_dec(v_a_5102_);
lean_dec_ref(v_a_5101_);
lean_dec(v_a_5100_);
lean_dec_ref(v_a_5099_);
return v_res_5104_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_traceTreeCore___boxed(lean_object* v_g_5105_, lean_object* v_traceOpt_5106_, lean_object* v_a_5107_, lean_object* v_a_5108_, lean_object* v_a_5109_, lean_object* v_a_5110_, lean_object* v_a_5111_){
_start:
{
lean_object* v_res_5112_; 
v_res_5112_ = lp_aesop_Aesop_Goal_traceTreeCore(v_g_5105_, v_traceOpt_5106_, v_a_5107_, v_a_5108_, v_a_5109_, v_a_5110_);
lean_dec(v_a_5110_);
lean_dec_ref(v_a_5109_);
lean_dec(v_a_5108_);
lean_dec_ref(v_a_5107_);
return v_res_5112_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceTreeCore_spec__0___boxed(lean_object* v_traceOpt_5113_, lean_object* v_as_5114_, lean_object* v_sz_5115_, lean_object* v_i_5116_, lean_object* v_b_5117_, lean_object* v___y_5118_, lean_object* v___y_5119_, lean_object* v___y_5120_, lean_object* v___y_5121_, lean_object* v___y_5122_){
_start:
{
size_t v_sz_boxed_5123_; size_t v_i_boxed_5124_; lean_object* v_res_5125_; 
v_sz_boxed_5123_ = lean_unbox_usize(v_sz_5115_);
lean_dec(v_sz_5115_);
v_i_boxed_5124_ = lean_unbox_usize(v_i_5116_);
lean_dec(v_i_5116_);
v_res_5125_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_traceTreeCore_spec__0(v_traceOpt_5113_, v_as_5114_, v_sz_boxed_5123_, v_i_boxed_5124_, v_b_5117_, v___y_5118_, v___y_5119_, v___y_5120_, v___y_5121_);
lean_dec(v___y_5121_);
lean_dec_ref(v___y_5120_);
lean_dec(v___y_5119_);
lean_dec_ref(v___y_5118_);
lean_dec_ref(v_as_5114_);
return v_res_5125_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_traceTree(lean_object* v_g_5126_, lean_object* v_traceOpt_5127_, lean_object* v_a_5128_, lean_object* v_a_5129_, lean_object* v_a_5130_, lean_object* v_a_5131_){
_start:
{
lean_object* v___x_5133_; lean_object* v_a_5134_; lean_object* v___x_5136_; uint8_t v_isShared_5137_; uint8_t v_isSharedCheck_5144_; 
v___x_5133_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_Goal_traceMetadata_spec__5___redArg(v_traceOpt_5127_, v_a_5130_);
v_a_5134_ = lean_ctor_get(v___x_5133_, 0);
v_isSharedCheck_5144_ = !lean_is_exclusive(v___x_5133_);
if (v_isSharedCheck_5144_ == 0)
{
v___x_5136_ = v___x_5133_;
v_isShared_5137_ = v_isSharedCheck_5144_;
goto v_resetjp_5135_;
}
else
{
lean_inc(v_a_5134_);
lean_dec(v___x_5133_);
v___x_5136_ = lean_box(0);
v_isShared_5137_ = v_isSharedCheck_5144_;
goto v_resetjp_5135_;
}
v_resetjp_5135_:
{
uint8_t v___x_5138_; 
v___x_5138_ = lean_unbox(v_a_5134_);
lean_dec(v_a_5134_);
if (v___x_5138_ == 0)
{
lean_object* v___x_5139_; lean_object* v___x_5141_; 
lean_dec_ref(v_traceOpt_5127_);
lean_dec(v_g_5126_);
v___x_5139_ = lean_box(0);
if (v_isShared_5137_ == 0)
{
lean_ctor_set(v___x_5136_, 0, v___x_5139_);
v___x_5141_ = v___x_5136_;
goto v_reusejp_5140_;
}
else
{
lean_object* v_reuseFailAlloc_5142_; 
v_reuseFailAlloc_5142_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5142_, 0, v___x_5139_);
v___x_5141_ = v_reuseFailAlloc_5142_;
goto v_reusejp_5140_;
}
v_reusejp_5140_:
{
return v___x_5141_;
}
}
else
{
lean_object* v___x_5143_; 
lean_del_object(v___x_5136_);
v___x_5143_ = lp_aesop_Aesop_Goal_traceTreeCore(v_g_5126_, v_traceOpt_5127_, v_a_5128_, v_a_5129_, v_a_5130_, v_a_5131_);
return v___x_5143_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_traceTree___boxed(lean_object* v_g_5145_, lean_object* v_traceOpt_5146_, lean_object* v_a_5147_, lean_object* v_a_5148_, lean_object* v_a_5149_, lean_object* v_a_5150_, lean_object* v_a_5151_){
_start:
{
lean_object* v_res_5152_; 
v_res_5152_ = lp_aesop_Aesop_Goal_traceTree(v_g_5145_, v_traceOpt_5146_, v_a_5147_, v_a_5148_, v_a_5149_, v_a_5150_);
lean_dec(v_a_5150_);
lean_dec_ref(v_a_5149_);
lean_dec(v_a_5148_);
lean_dec_ref(v_a_5147_);
return v_res_5152_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_traceTree(lean_object* v_r_5153_, lean_object* v_traceOpt_5154_, lean_object* v_a_5155_, lean_object* v_a_5156_, lean_object* v_a_5157_, lean_object* v_a_5158_){
_start:
{
lean_object* v___x_5160_; lean_object* v_a_5161_; lean_object* v___x_5163_; uint8_t v_isShared_5164_; uint8_t v_isSharedCheck_5171_; 
v___x_5160_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_Goal_traceMetadata_spec__5___redArg(v_traceOpt_5154_, v_a_5157_);
v_a_5161_ = lean_ctor_get(v___x_5160_, 0);
v_isSharedCheck_5171_ = !lean_is_exclusive(v___x_5160_);
if (v_isSharedCheck_5171_ == 0)
{
v___x_5163_ = v___x_5160_;
v_isShared_5164_ = v_isSharedCheck_5171_;
goto v_resetjp_5162_;
}
else
{
lean_inc(v_a_5161_);
lean_dec(v___x_5160_);
v___x_5163_ = lean_box(0);
v_isShared_5164_ = v_isSharedCheck_5171_;
goto v_resetjp_5162_;
}
v_resetjp_5162_:
{
uint8_t v___x_5165_; 
v___x_5165_ = lean_unbox(v_a_5161_);
lean_dec(v_a_5161_);
if (v___x_5165_ == 0)
{
lean_object* v___x_5166_; lean_object* v___x_5168_; 
lean_dec_ref(v_traceOpt_5154_);
lean_dec(v_r_5153_);
v___x_5166_ = lean_box(0);
if (v_isShared_5164_ == 0)
{
lean_ctor_set(v___x_5163_, 0, v___x_5166_);
v___x_5168_ = v___x_5163_;
goto v_reusejp_5167_;
}
else
{
lean_object* v_reuseFailAlloc_5169_; 
v_reuseFailAlloc_5169_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5169_, 0, v___x_5166_);
v___x_5168_ = v_reuseFailAlloc_5169_;
goto v_reusejp_5167_;
}
v_reusejp_5167_:
{
return v___x_5168_;
}
}
else
{
lean_object* v___x_5170_; 
lean_del_object(v___x_5163_);
v___x_5170_ = lp_aesop_Aesop_Rapp_traceTreeCore(v_r_5153_, v_traceOpt_5154_, v_a_5155_, v_a_5156_, v_a_5157_, v_a_5158_);
return v___x_5170_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_traceTree___boxed(lean_object* v_r_5172_, lean_object* v_traceOpt_5173_, lean_object* v_a_5174_, lean_object* v_a_5175_, lean_object* v_a_5176_, lean_object* v_a_5177_, lean_object* v_a_5178_){
_start:
{
lean_object* v_res_5179_; 
v_res_5179_ = lp_aesop_Aesop_Rapp_traceTree(v_r_5172_, v_traceOpt_5173_, v_a_5174_, v_a_5175_, v_a_5176_, v_a_5177_);
lean_dec(v_a_5177_);
lean_dec_ref(v_a_5176_);
lean_dec(v_a_5175_);
lean_dec_ref(v_a_5174_);
return v_res_5179_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Tree_RunMetaM(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Lean_Meta_SavedState(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Data_Array_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_Tree_Tracing(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Tree_RunMetaM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Lean_Meta_SavedState(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Data_Array_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_Tree_Tracing(uint8_t builtin) {
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
lean_object* initialize_aesop_Aesop_Tree_RunMetaM(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Lean_Meta_SavedState(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Data_Array_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_Tree_Tracing(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Tree_RunMetaM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Lean_Meta_SavedState(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Data_Array_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Tree_Tracing(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_Tree_Tracing(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_Tree_Tracing(builtin);
}
#ifdef __cplusplus
}
#endif
