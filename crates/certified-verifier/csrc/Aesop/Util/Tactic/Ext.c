// Lean compiler output
// Module: Aesop.Util.Tactic.Ext
// Imports: public import Init public meta import Init public import Aesop.Tracing import Lean.Elab.Tactic.Ext import Lean.Meta.Tactic.Intro
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
lean_object* l_Lean_Meta_saveState___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Meta_SavedState_restore___redArg(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
uint8_t l_Lean_Exception_isRuntime(lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* lean_array_mk(lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* l_Lean_MVarId_intros(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
extern lean_object* lp_aesop_Aesop_TraceOption_debug;
lean_object* l_Lean_Elab_Tactic_Ext_applyExtTheoremAt___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Elab_goalsToMessageData(lean_object*);
lean_object* l_Lean_indentD(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* lean_st_ref_take(lean_object*);
double lean_float_of_nat(lean_object*);
lean_object* l_Lean_PersistentArray_push___redArg(lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Name_append(lean_object*, lean_object*);
uint8_t l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(lean_object*, lean_object*, lean_object*);
lean_object* lean_io_mono_nanos_now();
double lean_float_div(double, double);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_PersistentArray_toArray___redArg(lean_object*);
extern lean_object* l_Lean_trace_profiler;
lean_object* l_Lean_PersistentArray_append___redArg(lean_object*, lean_object*);
double lean_float_sub(double, double);
uint8_t lean_float_decLt(double, double);
extern lean_object* l_Lean_trace_profiler_useHeartbeats;
extern lean_object* l_Lean_trace_profiler_threshold;
lean_object* lean_io_get_num_heartbeats();
LEAN_EXPORT lean_object* lp_aesop_Lean_observing_x3f___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_observing_x3f___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_observing_x3f___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_observing_x3f___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__4___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__4___redArg___closed__0;
static lean_once_cell_t lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__4___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__4___redArg___closed__1;
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__4___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__4___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__4(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_Option_get___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__5(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__5___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__6_spec__8___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__6_spec__8___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__6_spec__10(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__6_spec__10___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__6_spec__7_spec__8(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__6_spec__7_spec__8___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_addTrace___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__3_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_addTrace___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__3_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__6_spec__7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__6_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__6_spec__9(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__6_spec__9___boxed(lean_object*);
static lean_once_cell_t lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__6___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__6___closed__0;
static const lean_string_object lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__6___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 54, .m_capacity = 54, .m_length = 53, .m_data = "<exception thrown while producing trace node message>"};
static const lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__6___closed__1 = (const lean_object*)&lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__6___closed__1_value;
static lean_once_cell_t lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__6___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__6___closed__2;
static lean_once_cell_t lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__6___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__6___closed__3;
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__6(lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__2___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Lean_addTrace___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_aesop_Lean_addTrace___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__3___closed__0 = (const lean_object*)&lp_aesop_Lean_addTrace___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__3___closed__0_value;
static const lean_array_object lp_aesop_Lean_addTrace___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__3___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Lean_addTrace___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__3___closed__1 = (const lean_object*)&lp_aesop_Lean_addTrace___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__3___closed__1_value;
LEAN_EXPORT lean_object* lp_aesop_Lean_addTrace___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addTrace___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__1(size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___lam__0___closed__0 = (const lean_object*)&lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___lam__0___closed__0_value;
static const lean_string_object lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 30, .m_capacity = 30, .m_length = 29, .m_data = "ext lemma applied; new goals:"};
static const lean_object* lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___lam__0___closed__1 = (const lean_object*)&lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___lam__0___closed__1_value;
static lean_once_cell_t lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___lam__0___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___lam__0___closed__2;
static const lean_string_object lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = "no applicable ext lemma"};
static const lean_object* lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___lam__0___closed__3 = (const lean_object*)&lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___lam__0___closed__3_value;
static lean_once_cell_t lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___lam__0___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___lam__0___closed__4;
static const lean_string_object lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "goal:"};
static const lean_object* lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___closed__0 = (const lean_object*)&lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___closed__0_value;
static lean_once_cell_t lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___closed__1;
static const lean_string_object lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "straightLineExt"};
static const lean_object* lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___closed__2 = (const lean_object*)&lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___closed__2_value;
static lean_once_cell_t lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___closed__3;
static lean_once_cell_t lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___closed__4;
static const lean_string_object lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___closed__5 = (const lean_object*)&lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___closed__5_value;
static const lean_ctor_object lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___closed__5_value),LEAN_SCALAR_PTR_LITERAL(212, 145, 141, 177, 67, 149, 127, 197)}};
static const lean_object* lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___closed__6 = (const lean_object*)&lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___closed__6_value;
static lean_once_cell_t lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___closed__7;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__6_spec__8(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__6_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_straightLineExt(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_straightLineExt___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_straightLineExtProgress_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_straightLineExtProgress_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_straightLineExtProgress___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 43, .m_capacity = 43, .m_length = 42, .m_data = "no applicable extensionality theorem found"};
static const lean_object* lp_aesop_Aesop_straightLineExtProgress___closed__0 = (const lean_object*)&lp_aesop_Aesop_straightLineExtProgress___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_straightLineExtProgress___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_straightLineExtProgress___closed__1;
LEAN_EXPORT lean_object* lp_aesop_Aesop_straightLineExtProgress(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_straightLineExtProgress___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_straightLineExtProgress_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_straightLineExtProgress_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_observing_x3f___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__0___redArg(lean_object* v_x_1_, lean_object* v___y_2_, lean_object* v___y_3_, lean_object* v___y_4_, lean_object* v___y_5_){
_start:
{
lean_object* v___x_7_; 
v___x_7_ = l_Lean_Meta_saveState___redArg(v___y_3_, v___y_5_);
if (lean_obj_tag(v___x_7_) == 0)
{
lean_object* v_a_8_; lean_object* v___x_9_; 
v_a_8_ = lean_ctor_get(v___x_7_, 0);
lean_inc(v_a_8_);
lean_dec_ref_known(v___x_7_, 1);
lean_inc(v___y_5_);
lean_inc_ref(v___y_4_);
lean_inc(v___y_3_);
lean_inc_ref(v___y_2_);
v___x_9_ = lean_apply_5(v_x_1_, v___y_2_, v___y_3_, v___y_4_, v___y_5_, lean_box(0));
if (lean_obj_tag(v___x_9_) == 0)
{
lean_object* v_a_10_; lean_object* v___x_12_; uint8_t v_isShared_13_; uint8_t v_isSharedCheck_18_; 
lean_dec(v_a_8_);
v_a_10_ = lean_ctor_get(v___x_9_, 0);
v_isSharedCheck_18_ = !lean_is_exclusive(v___x_9_);
if (v_isSharedCheck_18_ == 0)
{
v___x_12_ = v___x_9_;
v_isShared_13_ = v_isSharedCheck_18_;
goto v_resetjp_11_;
}
else
{
lean_inc(v_a_10_);
lean_dec(v___x_9_);
v___x_12_ = lean_box(0);
v_isShared_13_ = v_isSharedCheck_18_;
goto v_resetjp_11_;
}
v_resetjp_11_:
{
lean_object* v___x_14_; lean_object* v___x_16_; 
v___x_14_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_14_, 0, v_a_10_);
if (v_isShared_13_ == 0)
{
lean_ctor_set(v___x_12_, 0, v___x_14_);
v___x_16_ = v___x_12_;
goto v_reusejp_15_;
}
else
{
lean_object* v_reuseFailAlloc_17_; 
v_reuseFailAlloc_17_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_17_, 0, v___x_14_);
v___x_16_ = v_reuseFailAlloc_17_;
goto v_reusejp_15_;
}
v_reusejp_15_:
{
return v___x_16_;
}
}
}
else
{
lean_object* v_a_19_; lean_object* v___x_21_; uint8_t v_isShared_22_; uint8_t v_isSharedCheck_48_; 
v_a_19_ = lean_ctor_get(v___x_9_, 0);
v_isSharedCheck_48_ = !lean_is_exclusive(v___x_9_);
if (v_isSharedCheck_48_ == 0)
{
v___x_21_ = v___x_9_;
v_isShared_22_ = v_isSharedCheck_48_;
goto v_resetjp_20_;
}
else
{
lean_inc(v_a_19_);
lean_dec(v___x_9_);
v___x_21_ = lean_box(0);
v_isShared_22_ = v_isSharedCheck_48_;
goto v_resetjp_20_;
}
v_resetjp_20_:
{
uint8_t v___y_24_; uint8_t v___x_46_; 
v___x_46_ = l_Lean_Exception_isInterrupt(v_a_19_);
if (v___x_46_ == 0)
{
uint8_t v___x_47_; 
lean_inc(v_a_19_);
v___x_47_ = l_Lean_Exception_isRuntime(v_a_19_);
v___y_24_ = v___x_47_;
goto v___jp_23_;
}
else
{
v___y_24_ = v___x_46_;
goto v___jp_23_;
}
v___jp_23_:
{
if (v___y_24_ == 0)
{
lean_object* v___x_25_; 
lean_del_object(v___x_21_);
lean_dec(v_a_19_);
v___x_25_ = l_Lean_Meta_SavedState_restore___redArg(v_a_8_, v___y_3_, v___y_5_);
lean_dec(v_a_8_);
if (lean_obj_tag(v___x_25_) == 0)
{
lean_object* v___x_27_; uint8_t v_isShared_28_; uint8_t v_isSharedCheck_33_; 
v_isSharedCheck_33_ = !lean_is_exclusive(v___x_25_);
if (v_isSharedCheck_33_ == 0)
{
lean_object* v_unused_34_; 
v_unused_34_ = lean_ctor_get(v___x_25_, 0);
lean_dec(v_unused_34_);
v___x_27_ = v___x_25_;
v_isShared_28_ = v_isSharedCheck_33_;
goto v_resetjp_26_;
}
else
{
lean_dec(v___x_25_);
v___x_27_ = lean_box(0);
v_isShared_28_ = v_isSharedCheck_33_;
goto v_resetjp_26_;
}
v_resetjp_26_:
{
lean_object* v___x_29_; lean_object* v___x_31_; 
v___x_29_ = lean_box(0);
if (v_isShared_28_ == 0)
{
lean_ctor_set(v___x_27_, 0, v___x_29_);
v___x_31_ = v___x_27_;
goto v_reusejp_30_;
}
else
{
lean_object* v_reuseFailAlloc_32_; 
v_reuseFailAlloc_32_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_32_, 0, v___x_29_);
v___x_31_ = v_reuseFailAlloc_32_;
goto v_reusejp_30_;
}
v_reusejp_30_:
{
return v___x_31_;
}
}
}
else
{
lean_object* v_a_35_; lean_object* v___x_37_; uint8_t v_isShared_38_; uint8_t v_isSharedCheck_42_; 
v_a_35_ = lean_ctor_get(v___x_25_, 0);
v_isSharedCheck_42_ = !lean_is_exclusive(v___x_25_);
if (v_isSharedCheck_42_ == 0)
{
v___x_37_ = v___x_25_;
v_isShared_38_ = v_isSharedCheck_42_;
goto v_resetjp_36_;
}
else
{
lean_inc(v_a_35_);
lean_dec(v___x_25_);
v___x_37_ = lean_box(0);
v_isShared_38_ = v_isSharedCheck_42_;
goto v_resetjp_36_;
}
v_resetjp_36_:
{
lean_object* v___x_40_; 
if (v_isShared_38_ == 0)
{
v___x_40_ = v___x_37_;
goto v_reusejp_39_;
}
else
{
lean_object* v_reuseFailAlloc_41_; 
v_reuseFailAlloc_41_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_41_, 0, v_a_35_);
v___x_40_ = v_reuseFailAlloc_41_;
goto v_reusejp_39_;
}
v_reusejp_39_:
{
return v___x_40_;
}
}
}
}
else
{
lean_object* v___x_44_; 
lean_dec(v_a_8_);
if (v_isShared_22_ == 0)
{
v___x_44_ = v___x_21_;
goto v_reusejp_43_;
}
else
{
lean_object* v_reuseFailAlloc_45_; 
v_reuseFailAlloc_45_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_45_, 0, v_a_19_);
v___x_44_ = v_reuseFailAlloc_45_;
goto v_reusejp_43_;
}
v_reusejp_43_:
{
return v___x_44_;
}
}
}
}
}
}
else
{
lean_object* v_a_49_; lean_object* v___x_51_; uint8_t v_isShared_52_; uint8_t v_isSharedCheck_56_; 
lean_dec_ref(v_x_1_);
v_a_49_ = lean_ctor_get(v___x_7_, 0);
v_isSharedCheck_56_ = !lean_is_exclusive(v___x_7_);
if (v_isSharedCheck_56_ == 0)
{
v___x_51_ = v___x_7_;
v_isShared_52_ = v_isSharedCheck_56_;
goto v_resetjp_50_;
}
else
{
lean_inc(v_a_49_);
lean_dec(v___x_7_);
v___x_51_ = lean_box(0);
v_isShared_52_ = v_isSharedCheck_56_;
goto v_resetjp_50_;
}
v_resetjp_50_:
{
lean_object* v___x_54_; 
if (v_isShared_52_ == 0)
{
v___x_54_ = v___x_51_;
goto v_reusejp_53_;
}
else
{
lean_object* v_reuseFailAlloc_55_; 
v_reuseFailAlloc_55_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_55_, 0, v_a_49_);
v___x_54_ = v_reuseFailAlloc_55_;
goto v_reusejp_53_;
}
v_reusejp_53_:
{
return v___x_54_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_observing_x3f___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__0___redArg___boxed(lean_object* v_x_57_, lean_object* v___y_58_, lean_object* v___y_59_, lean_object* v___y_60_, lean_object* v___y_61_, lean_object* v___y_62_){
_start:
{
lean_object* v_res_63_; 
v_res_63_ = lp_aesop_Lean_observing_x3f___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__0___redArg(v_x_57_, v___y_58_, v___y_59_, v___y_60_, v___y_61_);
lean_dec(v___y_61_);
lean_dec_ref(v___y_60_);
lean_dec(v___y_59_);
lean_dec_ref(v___y_58_);
return v_res_63_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_observing_x3f___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__0(lean_object* v_00_u03b1_64_, lean_object* v_x_65_, lean_object* v___y_66_, lean_object* v___y_67_, lean_object* v___y_68_, lean_object* v___y_69_){
_start:
{
lean_object* v___x_71_; 
v___x_71_ = lp_aesop_Lean_observing_x3f___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__0___redArg(v_x_65_, v___y_66_, v___y_67_, v___y_68_, v___y_69_);
return v___x_71_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_observing_x3f___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__0___boxed(lean_object* v_00_u03b1_72_, lean_object* v_x_73_, lean_object* v___y_74_, lean_object* v___y_75_, lean_object* v___y_76_, lean_object* v___y_77_, lean_object* v___y_78_){
_start:
{
lean_object* v_res_79_; 
v_res_79_ = lp_aesop_Lean_observing_x3f___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__0(v_00_u03b1_72_, v_x_73_, v___y_74_, v___y_75_, v___y_76_, v___y_77_);
lean_dec(v___y_77_);
lean_dec_ref(v___y_76_);
lean_dec(v___y_75_);
lean_dec_ref(v___y_74_);
return v_res_79_;
}
}
static lean_object* _init_lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__4___redArg___closed__0(void){
_start:
{
lean_object* v___x_80_; lean_object* v___x_81_; lean_object* v___x_82_; 
v___x_80_ = lean_unsigned_to_nat(32u);
v___x_81_ = lean_mk_empty_array_with_capacity(v___x_80_);
v___x_82_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_82_, 0, v___x_81_);
return v___x_82_;
}
}
static lean_object* _init_lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__4___redArg___closed__1(void){
_start:
{
size_t v___x_83_; lean_object* v___x_84_; lean_object* v___x_85_; lean_object* v___x_86_; lean_object* v___x_87_; lean_object* v___x_88_; 
v___x_83_ = ((size_t)5ULL);
v___x_84_ = lean_unsigned_to_nat(0u);
v___x_85_ = lean_unsigned_to_nat(32u);
v___x_86_ = lean_mk_empty_array_with_capacity(v___x_85_);
v___x_87_ = lean_obj_once(&lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__4___redArg___closed__0, &lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__4___redArg___closed__0_once, _init_lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__4___redArg___closed__0);
v___x_88_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_88_, 0, v___x_87_);
lean_ctor_set(v___x_88_, 1, v___x_86_);
lean_ctor_set(v___x_88_, 2, v___x_84_);
lean_ctor_set(v___x_88_, 3, v___x_84_);
lean_ctor_set_usize(v___x_88_, 4, v___x_83_);
return v___x_88_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__4___redArg(lean_object* v___y_89_){
_start:
{
lean_object* v___x_91_; lean_object* v_traceState_92_; lean_object* v_traces_93_; lean_object* v___x_94_; lean_object* v_traceState_95_; lean_object* v_env_96_; lean_object* v_nextMacroScope_97_; lean_object* v_ngen_98_; lean_object* v_auxDeclNGen_99_; lean_object* v_cache_100_; lean_object* v_messages_101_; lean_object* v_infoState_102_; lean_object* v_snapshotTasks_103_; lean_object* v___x_105_; uint8_t v_isShared_106_; uint8_t v_isSharedCheck_122_; 
v___x_91_ = lean_st_ref_get(v___y_89_);
v_traceState_92_ = lean_ctor_get(v___x_91_, 4);
lean_inc_ref(v_traceState_92_);
lean_dec(v___x_91_);
v_traces_93_ = lean_ctor_get(v_traceState_92_, 0);
lean_inc_ref(v_traces_93_);
lean_dec_ref(v_traceState_92_);
v___x_94_ = lean_st_ref_take(v___y_89_);
v_traceState_95_ = lean_ctor_get(v___x_94_, 4);
v_env_96_ = lean_ctor_get(v___x_94_, 0);
v_nextMacroScope_97_ = lean_ctor_get(v___x_94_, 1);
v_ngen_98_ = lean_ctor_get(v___x_94_, 2);
v_auxDeclNGen_99_ = lean_ctor_get(v___x_94_, 3);
v_cache_100_ = lean_ctor_get(v___x_94_, 5);
v_messages_101_ = lean_ctor_get(v___x_94_, 6);
v_infoState_102_ = lean_ctor_get(v___x_94_, 7);
v_snapshotTasks_103_ = lean_ctor_get(v___x_94_, 8);
v_isSharedCheck_122_ = !lean_is_exclusive(v___x_94_);
if (v_isSharedCheck_122_ == 0)
{
v___x_105_ = v___x_94_;
v_isShared_106_ = v_isSharedCheck_122_;
goto v_resetjp_104_;
}
else
{
lean_inc(v_snapshotTasks_103_);
lean_inc(v_infoState_102_);
lean_inc(v_messages_101_);
lean_inc(v_cache_100_);
lean_inc(v_traceState_95_);
lean_inc(v_auxDeclNGen_99_);
lean_inc(v_ngen_98_);
lean_inc(v_nextMacroScope_97_);
lean_inc(v_env_96_);
lean_dec(v___x_94_);
v___x_105_ = lean_box(0);
v_isShared_106_ = v_isSharedCheck_122_;
goto v_resetjp_104_;
}
v_resetjp_104_:
{
uint64_t v_tid_107_; lean_object* v___x_109_; uint8_t v_isShared_110_; uint8_t v_isSharedCheck_120_; 
v_tid_107_ = lean_ctor_get_uint64(v_traceState_95_, sizeof(void*)*1);
v_isSharedCheck_120_ = !lean_is_exclusive(v_traceState_95_);
if (v_isSharedCheck_120_ == 0)
{
lean_object* v_unused_121_; 
v_unused_121_ = lean_ctor_get(v_traceState_95_, 0);
lean_dec(v_unused_121_);
v___x_109_ = v_traceState_95_;
v_isShared_110_ = v_isSharedCheck_120_;
goto v_resetjp_108_;
}
else
{
lean_dec(v_traceState_95_);
v___x_109_ = lean_box(0);
v_isShared_110_ = v_isSharedCheck_120_;
goto v_resetjp_108_;
}
v_resetjp_108_:
{
lean_object* v___x_111_; lean_object* v___x_113_; 
v___x_111_ = lean_obj_once(&lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__4___redArg___closed__1, &lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__4___redArg___closed__1_once, _init_lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__4___redArg___closed__1);
if (v_isShared_110_ == 0)
{
lean_ctor_set(v___x_109_, 0, v___x_111_);
v___x_113_ = v___x_109_;
goto v_reusejp_112_;
}
else
{
lean_object* v_reuseFailAlloc_119_; 
v_reuseFailAlloc_119_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_119_, 0, v___x_111_);
lean_ctor_set_uint64(v_reuseFailAlloc_119_, sizeof(void*)*1, v_tid_107_);
v___x_113_ = v_reuseFailAlloc_119_;
goto v_reusejp_112_;
}
v_reusejp_112_:
{
lean_object* v___x_115_; 
if (v_isShared_106_ == 0)
{
lean_ctor_set(v___x_105_, 4, v___x_113_);
v___x_115_ = v___x_105_;
goto v_reusejp_114_;
}
else
{
lean_object* v_reuseFailAlloc_118_; 
v_reuseFailAlloc_118_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_118_, 0, v_env_96_);
lean_ctor_set(v_reuseFailAlloc_118_, 1, v_nextMacroScope_97_);
lean_ctor_set(v_reuseFailAlloc_118_, 2, v_ngen_98_);
lean_ctor_set(v_reuseFailAlloc_118_, 3, v_auxDeclNGen_99_);
lean_ctor_set(v_reuseFailAlloc_118_, 4, v___x_113_);
lean_ctor_set(v_reuseFailAlloc_118_, 5, v_cache_100_);
lean_ctor_set(v_reuseFailAlloc_118_, 6, v_messages_101_);
lean_ctor_set(v_reuseFailAlloc_118_, 7, v_infoState_102_);
lean_ctor_set(v_reuseFailAlloc_118_, 8, v_snapshotTasks_103_);
v___x_115_ = v_reuseFailAlloc_118_;
goto v_reusejp_114_;
}
v_reusejp_114_:
{
lean_object* v___x_116_; lean_object* v___x_117_; 
v___x_116_ = lean_st_ref_set(v___y_89_, v___x_115_);
v___x_117_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_117_, 0, v_traces_93_);
return v___x_117_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__4___redArg___boxed(lean_object* v___y_123_, lean_object* v___y_124_){
_start:
{
lean_object* v_res_125_; 
v_res_125_ = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__4___redArg(v___y_123_);
lean_dec(v___y_123_);
return v_res_125_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__4(lean_object* v___y_126_, lean_object* v___y_127_, lean_object* v___y_128_, lean_object* v___y_129_){
_start:
{
lean_object* v___x_131_; 
v___x_131_ = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__4___redArg(v___y_129_);
return v___x_131_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__4___boxed(lean_object* v___y_132_, lean_object* v___y_133_, lean_object* v___y_134_, lean_object* v___y_135_, lean_object* v___y_136_){
_start:
{
lean_object* v_res_137_; 
v_res_137_ = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__4(v___y_132_, v___y_133_, v___y_134_, v___y_135_);
lean_dec(v___y_135_);
lean_dec_ref(v___y_134_);
lean_dec(v___y_133_);
lean_dec_ref(v___y_132_);
return v_res_137_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_Option_get___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__5(lean_object* v_opts_138_, lean_object* v_opt_139_){
_start:
{
lean_object* v_name_140_; lean_object* v_defValue_141_; lean_object* v_map_142_; lean_object* v___x_143_; 
v_name_140_ = lean_ctor_get(v_opt_139_, 0);
v_defValue_141_ = lean_ctor_get(v_opt_139_, 1);
v_map_142_ = lean_ctor_get(v_opts_138_, 0);
v___x_143_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_142_, v_name_140_);
if (lean_obj_tag(v___x_143_) == 0)
{
uint8_t v___x_144_; 
v___x_144_ = lean_unbox(v_defValue_141_);
return v___x_144_;
}
else
{
lean_object* v_val_145_; 
v_val_145_ = lean_ctor_get(v___x_143_, 0);
lean_inc(v_val_145_);
lean_dec_ref_known(v___x_143_, 1);
if (lean_obj_tag(v_val_145_) == 1)
{
uint8_t v_v_146_; 
v_v_146_ = lean_ctor_get_uint8(v_val_145_, 0);
lean_dec_ref_known(v_val_145_, 0);
return v_v_146_;
}
else
{
uint8_t v___x_147_; 
lean_dec(v_val_145_);
v___x_147_ = lean_unbox(v_defValue_141_);
return v___x_147_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__5___boxed(lean_object* v_opts_148_, lean_object* v_opt_149_){
_start:
{
uint8_t v_res_150_; lean_object* v_r_151_; 
v_res_150_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__5(v_opts_148_, v_opt_149_);
lean_dec_ref(v_opt_149_);
lean_dec_ref(v_opts_148_);
v_r_151_ = lean_box(v_res_150_);
return v_r_151_;
}
}
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__6_spec__8___redArg(lean_object* v_x_152_){
_start:
{
if (lean_obj_tag(v_x_152_) == 0)
{
lean_object* v_a_154_; lean_object* v___x_156_; uint8_t v_isShared_157_; uint8_t v_isSharedCheck_161_; 
v_a_154_ = lean_ctor_get(v_x_152_, 0);
v_isSharedCheck_161_ = !lean_is_exclusive(v_x_152_);
if (v_isSharedCheck_161_ == 0)
{
v___x_156_ = v_x_152_;
v_isShared_157_ = v_isSharedCheck_161_;
goto v_resetjp_155_;
}
else
{
lean_inc(v_a_154_);
lean_dec(v_x_152_);
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
v___x_159_ = v___x_156_;
goto v_reusejp_158_;
}
else
{
lean_object* v_reuseFailAlloc_160_; 
v_reuseFailAlloc_160_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_160_, 0, v_a_154_);
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
lean_object* v_a_162_; lean_object* v___x_164_; uint8_t v_isShared_165_; uint8_t v_isSharedCheck_169_; 
v_a_162_ = lean_ctor_get(v_x_152_, 0);
v_isSharedCheck_169_ = !lean_is_exclusive(v_x_152_);
if (v_isSharedCheck_169_ == 0)
{
v___x_164_ = v_x_152_;
v_isShared_165_ = v_isSharedCheck_169_;
goto v_resetjp_163_;
}
else
{
lean_inc(v_a_162_);
lean_dec(v_x_152_);
v___x_164_ = lean_box(0);
v_isShared_165_ = v_isSharedCheck_169_;
goto v_resetjp_163_;
}
v_resetjp_163_:
{
lean_object* v___x_167_; 
if (v_isShared_165_ == 0)
{
lean_ctor_set_tag(v___x_164_, 0);
v___x_167_ = v___x_164_;
goto v_reusejp_166_;
}
else
{
lean_object* v_reuseFailAlloc_168_; 
v_reuseFailAlloc_168_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_168_, 0, v_a_162_);
v___x_167_ = v_reuseFailAlloc_168_;
goto v_reusejp_166_;
}
v_reusejp_166_:
{
return v___x_167_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__6_spec__8___redArg___boxed(lean_object* v_x_170_, lean_object* v___y_171_){
_start:
{
lean_object* v_res_172_; 
v_res_172_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__6_spec__8___redArg(v_x_170_);
return v_res_172_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__6_spec__10(lean_object* v_opts_173_, lean_object* v_opt_174_){
_start:
{
lean_object* v_name_175_; lean_object* v_defValue_176_; lean_object* v_map_177_; lean_object* v___x_178_; 
v_name_175_ = lean_ctor_get(v_opt_174_, 0);
v_defValue_176_ = lean_ctor_get(v_opt_174_, 1);
v_map_177_ = lean_ctor_get(v_opts_173_, 0);
v___x_178_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_177_, v_name_175_);
if (lean_obj_tag(v___x_178_) == 0)
{
lean_inc(v_defValue_176_);
return v_defValue_176_;
}
else
{
lean_object* v_val_179_; 
v_val_179_ = lean_ctor_get(v___x_178_, 0);
lean_inc(v_val_179_);
lean_dec_ref_known(v___x_178_, 1);
if (lean_obj_tag(v_val_179_) == 3)
{
lean_object* v_v_180_; 
v_v_180_ = lean_ctor_get(v_val_179_, 0);
lean_inc(v_v_180_);
lean_dec_ref_known(v_val_179_, 1);
return v_v_180_;
}
else
{
lean_dec(v_val_179_);
lean_inc(v_defValue_176_);
return v_defValue_176_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__6_spec__10___boxed(lean_object* v_opts_181_, lean_object* v_opt_182_){
_start:
{
lean_object* v_res_183_; 
v_res_183_ = lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__6_spec__10(v_opts_181_, v_opt_182_);
lean_dec_ref(v_opt_182_);
lean_dec_ref(v_opts_181_);
return v_res_183_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__6_spec__7_spec__8(size_t v_sz_184_, size_t v_i_185_, lean_object* v_bs_186_){
_start:
{
uint8_t v___x_187_; 
v___x_187_ = lean_usize_dec_lt(v_i_185_, v_sz_184_);
if (v___x_187_ == 0)
{
return v_bs_186_;
}
else
{
lean_object* v_v_188_; lean_object* v_msg_189_; lean_object* v___x_190_; lean_object* v_bs_x27_191_; size_t v___x_192_; size_t v___x_193_; lean_object* v___x_194_; 
v_v_188_ = lean_array_uget_borrowed(v_bs_186_, v_i_185_);
v_msg_189_ = lean_ctor_get(v_v_188_, 1);
lean_inc_ref(v_msg_189_);
v___x_190_ = lean_unsigned_to_nat(0u);
v_bs_x27_191_ = lean_array_uset(v_bs_186_, v_i_185_, v___x_190_);
v___x_192_ = ((size_t)1ULL);
v___x_193_ = lean_usize_add(v_i_185_, v___x_192_);
v___x_194_ = lean_array_uset(v_bs_x27_191_, v_i_185_, v_msg_189_);
v_i_185_ = v___x_193_;
v_bs_186_ = v___x_194_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__6_spec__7_spec__8___boxed(lean_object* v_sz_196_, lean_object* v_i_197_, lean_object* v_bs_198_){
_start:
{
size_t v_sz_boxed_199_; size_t v_i_boxed_200_; lean_object* v_res_201_; 
v_sz_boxed_199_ = lean_unbox_usize(v_sz_196_);
lean_dec(v_sz_196_);
v_i_boxed_200_ = lean_unbox_usize(v_i_197_);
lean_dec(v_i_197_);
v_res_201_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__6_spec__7_spec__8(v_sz_boxed_199_, v_i_boxed_200_, v_bs_198_);
return v_res_201_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_addTrace___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__3_spec__3(lean_object* v_msgData_202_, lean_object* v___y_203_, lean_object* v___y_204_, lean_object* v___y_205_, lean_object* v___y_206_){
_start:
{
lean_object* v___x_208_; lean_object* v_env_209_; lean_object* v___x_210_; lean_object* v_mctx_211_; lean_object* v_lctx_212_; lean_object* v_options_213_; lean_object* v___x_214_; lean_object* v___x_215_; lean_object* v___x_216_; 
v___x_208_ = lean_st_ref_get(v___y_206_);
v_env_209_ = lean_ctor_get(v___x_208_, 0);
lean_inc_ref(v_env_209_);
lean_dec(v___x_208_);
v___x_210_ = lean_st_ref_get(v___y_204_);
v_mctx_211_ = lean_ctor_get(v___x_210_, 0);
lean_inc_ref(v_mctx_211_);
lean_dec(v___x_210_);
v_lctx_212_ = lean_ctor_get(v___y_203_, 2);
v_options_213_ = lean_ctor_get(v___y_205_, 2);
lean_inc_ref(v_options_213_);
lean_inc_ref(v_lctx_212_);
v___x_214_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_214_, 0, v_env_209_);
lean_ctor_set(v___x_214_, 1, v_mctx_211_);
lean_ctor_set(v___x_214_, 2, v_lctx_212_);
lean_ctor_set(v___x_214_, 3, v_options_213_);
v___x_215_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_215_, 0, v___x_214_);
lean_ctor_set(v___x_215_, 1, v_msgData_202_);
v___x_216_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_216_, 0, v___x_215_);
return v___x_216_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_addTrace___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__3_spec__3___boxed(lean_object* v_msgData_217_, lean_object* v___y_218_, lean_object* v___y_219_, lean_object* v___y_220_, lean_object* v___y_221_, lean_object* v___y_222_){
_start:
{
lean_object* v_res_223_; 
v_res_223_ = lp_aesop_Lean_addMessageContextFull___at___00Lean_addTrace___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__3_spec__3(v_msgData_217_, v___y_218_, v___y_219_, v___y_220_, v___y_221_);
lean_dec(v___y_221_);
lean_dec_ref(v___y_220_);
lean_dec(v___y_219_);
lean_dec_ref(v___y_218_);
return v_res_223_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__6_spec__7(lean_object* v_oldTraces_224_, lean_object* v_data_225_, lean_object* v_ref_226_, lean_object* v_msg_227_, lean_object* v___y_228_, lean_object* v___y_229_, lean_object* v___y_230_, lean_object* v___y_231_){
_start:
{
lean_object* v_fileName_233_; lean_object* v_fileMap_234_; lean_object* v_options_235_; lean_object* v_currRecDepth_236_; lean_object* v_maxRecDepth_237_; lean_object* v_ref_238_; lean_object* v_currNamespace_239_; lean_object* v_openDecls_240_; lean_object* v_initHeartbeats_241_; lean_object* v_maxHeartbeats_242_; lean_object* v_quotContext_243_; lean_object* v_currMacroScope_244_; uint8_t v_diag_245_; lean_object* v_cancelTk_x3f_246_; uint8_t v_suppressElabErrors_247_; lean_object* v_inheritedTraceOptions_248_; lean_object* v___x_249_; lean_object* v_traceState_250_; lean_object* v_traces_251_; lean_object* v_ref_252_; lean_object* v___x_253_; lean_object* v___x_254_; size_t v_sz_255_; size_t v___x_256_; lean_object* v___x_257_; lean_object* v_msg_258_; lean_object* v___x_259_; lean_object* v_a_260_; lean_object* v___x_262_; uint8_t v_isShared_263_; uint8_t v_isSharedCheck_297_; 
v_fileName_233_ = lean_ctor_get(v___y_230_, 0);
v_fileMap_234_ = lean_ctor_get(v___y_230_, 1);
v_options_235_ = lean_ctor_get(v___y_230_, 2);
v_currRecDepth_236_ = lean_ctor_get(v___y_230_, 3);
v_maxRecDepth_237_ = lean_ctor_get(v___y_230_, 4);
v_ref_238_ = lean_ctor_get(v___y_230_, 5);
v_currNamespace_239_ = lean_ctor_get(v___y_230_, 6);
v_openDecls_240_ = lean_ctor_get(v___y_230_, 7);
v_initHeartbeats_241_ = lean_ctor_get(v___y_230_, 8);
v_maxHeartbeats_242_ = lean_ctor_get(v___y_230_, 9);
v_quotContext_243_ = lean_ctor_get(v___y_230_, 10);
v_currMacroScope_244_ = lean_ctor_get(v___y_230_, 11);
v_diag_245_ = lean_ctor_get_uint8(v___y_230_, sizeof(void*)*14);
v_cancelTk_x3f_246_ = lean_ctor_get(v___y_230_, 12);
v_suppressElabErrors_247_ = lean_ctor_get_uint8(v___y_230_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_248_ = lean_ctor_get(v___y_230_, 13);
v___x_249_ = lean_st_ref_get(v___y_231_);
v_traceState_250_ = lean_ctor_get(v___x_249_, 4);
lean_inc_ref(v_traceState_250_);
lean_dec(v___x_249_);
v_traces_251_ = lean_ctor_get(v_traceState_250_, 0);
lean_inc_ref(v_traces_251_);
lean_dec_ref(v_traceState_250_);
v_ref_252_ = l_Lean_replaceRef(v_ref_226_, v_ref_238_);
lean_inc_ref(v_inheritedTraceOptions_248_);
lean_inc(v_cancelTk_x3f_246_);
lean_inc(v_currMacroScope_244_);
lean_inc(v_quotContext_243_);
lean_inc(v_maxHeartbeats_242_);
lean_inc(v_initHeartbeats_241_);
lean_inc(v_openDecls_240_);
lean_inc(v_currNamespace_239_);
lean_inc(v_maxRecDepth_237_);
lean_inc(v_currRecDepth_236_);
lean_inc_ref(v_options_235_);
lean_inc_ref(v_fileMap_234_);
lean_inc_ref(v_fileName_233_);
v___x_253_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_253_, 0, v_fileName_233_);
lean_ctor_set(v___x_253_, 1, v_fileMap_234_);
lean_ctor_set(v___x_253_, 2, v_options_235_);
lean_ctor_set(v___x_253_, 3, v_currRecDepth_236_);
lean_ctor_set(v___x_253_, 4, v_maxRecDepth_237_);
lean_ctor_set(v___x_253_, 5, v_ref_252_);
lean_ctor_set(v___x_253_, 6, v_currNamespace_239_);
lean_ctor_set(v___x_253_, 7, v_openDecls_240_);
lean_ctor_set(v___x_253_, 8, v_initHeartbeats_241_);
lean_ctor_set(v___x_253_, 9, v_maxHeartbeats_242_);
lean_ctor_set(v___x_253_, 10, v_quotContext_243_);
lean_ctor_set(v___x_253_, 11, v_currMacroScope_244_);
lean_ctor_set(v___x_253_, 12, v_cancelTk_x3f_246_);
lean_ctor_set(v___x_253_, 13, v_inheritedTraceOptions_248_);
lean_ctor_set_uint8(v___x_253_, sizeof(void*)*14, v_diag_245_);
lean_ctor_set_uint8(v___x_253_, sizeof(void*)*14 + 1, v_suppressElabErrors_247_);
v___x_254_ = l_Lean_PersistentArray_toArray___redArg(v_traces_251_);
lean_dec_ref(v_traces_251_);
v_sz_255_ = lean_array_size(v___x_254_);
v___x_256_ = ((size_t)0ULL);
v___x_257_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__6_spec__7_spec__8(v_sz_255_, v___x_256_, v___x_254_);
v_msg_258_ = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(v_msg_258_, 0, v_data_225_);
lean_ctor_set(v_msg_258_, 1, v_msg_227_);
lean_ctor_set(v_msg_258_, 2, v___x_257_);
v___x_259_ = lp_aesop_Lean_addMessageContextFull___at___00Lean_addTrace___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__3_spec__3(v_msg_258_, v___y_228_, v___y_229_, v___x_253_, v___y_231_);
lean_dec_ref_known(v___x_253_, 14);
v_a_260_ = lean_ctor_get(v___x_259_, 0);
v_isSharedCheck_297_ = !lean_is_exclusive(v___x_259_);
if (v_isSharedCheck_297_ == 0)
{
v___x_262_ = v___x_259_;
v_isShared_263_ = v_isSharedCheck_297_;
goto v_resetjp_261_;
}
else
{
lean_inc(v_a_260_);
lean_dec(v___x_259_);
v___x_262_ = lean_box(0);
v_isShared_263_ = v_isSharedCheck_297_;
goto v_resetjp_261_;
}
v_resetjp_261_:
{
lean_object* v___x_264_; lean_object* v_traceState_265_; lean_object* v_env_266_; lean_object* v_nextMacroScope_267_; lean_object* v_ngen_268_; lean_object* v_auxDeclNGen_269_; lean_object* v_cache_270_; lean_object* v_messages_271_; lean_object* v_infoState_272_; lean_object* v_snapshotTasks_273_; lean_object* v___x_275_; uint8_t v_isShared_276_; uint8_t v_isSharedCheck_296_; 
v___x_264_ = lean_st_ref_take(v___y_231_);
v_traceState_265_ = lean_ctor_get(v___x_264_, 4);
v_env_266_ = lean_ctor_get(v___x_264_, 0);
v_nextMacroScope_267_ = lean_ctor_get(v___x_264_, 1);
v_ngen_268_ = lean_ctor_get(v___x_264_, 2);
v_auxDeclNGen_269_ = lean_ctor_get(v___x_264_, 3);
v_cache_270_ = lean_ctor_get(v___x_264_, 5);
v_messages_271_ = lean_ctor_get(v___x_264_, 6);
v_infoState_272_ = lean_ctor_get(v___x_264_, 7);
v_snapshotTasks_273_ = lean_ctor_get(v___x_264_, 8);
v_isSharedCheck_296_ = !lean_is_exclusive(v___x_264_);
if (v_isSharedCheck_296_ == 0)
{
v___x_275_ = v___x_264_;
v_isShared_276_ = v_isSharedCheck_296_;
goto v_resetjp_274_;
}
else
{
lean_inc(v_snapshotTasks_273_);
lean_inc(v_infoState_272_);
lean_inc(v_messages_271_);
lean_inc(v_cache_270_);
lean_inc(v_traceState_265_);
lean_inc(v_auxDeclNGen_269_);
lean_inc(v_ngen_268_);
lean_inc(v_nextMacroScope_267_);
lean_inc(v_env_266_);
lean_dec(v___x_264_);
v___x_275_ = lean_box(0);
v_isShared_276_ = v_isSharedCheck_296_;
goto v_resetjp_274_;
}
v_resetjp_274_:
{
uint64_t v_tid_277_; lean_object* v___x_279_; uint8_t v_isShared_280_; uint8_t v_isSharedCheck_294_; 
v_tid_277_ = lean_ctor_get_uint64(v_traceState_265_, sizeof(void*)*1);
v_isSharedCheck_294_ = !lean_is_exclusive(v_traceState_265_);
if (v_isSharedCheck_294_ == 0)
{
lean_object* v_unused_295_; 
v_unused_295_ = lean_ctor_get(v_traceState_265_, 0);
lean_dec(v_unused_295_);
v___x_279_ = v_traceState_265_;
v_isShared_280_ = v_isSharedCheck_294_;
goto v_resetjp_278_;
}
else
{
lean_dec(v_traceState_265_);
v___x_279_ = lean_box(0);
v_isShared_280_ = v_isSharedCheck_294_;
goto v_resetjp_278_;
}
v_resetjp_278_:
{
lean_object* v___x_281_; lean_object* v___x_282_; lean_object* v___x_284_; 
v___x_281_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_281_, 0, v_ref_226_);
lean_ctor_set(v___x_281_, 1, v_a_260_);
v___x_282_ = l_Lean_PersistentArray_push___redArg(v_oldTraces_224_, v___x_281_);
if (v_isShared_280_ == 0)
{
lean_ctor_set(v___x_279_, 0, v___x_282_);
v___x_284_ = v___x_279_;
goto v_reusejp_283_;
}
else
{
lean_object* v_reuseFailAlloc_293_; 
v_reuseFailAlloc_293_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_293_, 0, v___x_282_);
lean_ctor_set_uint64(v_reuseFailAlloc_293_, sizeof(void*)*1, v_tid_277_);
v___x_284_ = v_reuseFailAlloc_293_;
goto v_reusejp_283_;
}
v_reusejp_283_:
{
lean_object* v___x_286_; 
if (v_isShared_276_ == 0)
{
lean_ctor_set(v___x_275_, 4, v___x_284_);
v___x_286_ = v___x_275_;
goto v_reusejp_285_;
}
else
{
lean_object* v_reuseFailAlloc_292_; 
v_reuseFailAlloc_292_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_292_, 0, v_env_266_);
lean_ctor_set(v_reuseFailAlloc_292_, 1, v_nextMacroScope_267_);
lean_ctor_set(v_reuseFailAlloc_292_, 2, v_ngen_268_);
lean_ctor_set(v_reuseFailAlloc_292_, 3, v_auxDeclNGen_269_);
lean_ctor_set(v_reuseFailAlloc_292_, 4, v___x_284_);
lean_ctor_set(v_reuseFailAlloc_292_, 5, v_cache_270_);
lean_ctor_set(v_reuseFailAlloc_292_, 6, v_messages_271_);
lean_ctor_set(v_reuseFailAlloc_292_, 7, v_infoState_272_);
lean_ctor_set(v_reuseFailAlloc_292_, 8, v_snapshotTasks_273_);
v___x_286_ = v_reuseFailAlloc_292_;
goto v_reusejp_285_;
}
v_reusejp_285_:
{
lean_object* v___x_287_; lean_object* v___x_288_; lean_object* v___x_290_; 
v___x_287_ = lean_st_ref_set(v___y_231_, v___x_286_);
v___x_288_ = lean_box(0);
if (v_isShared_263_ == 0)
{
lean_ctor_set(v___x_262_, 0, v___x_288_);
v___x_290_ = v___x_262_;
goto v_reusejp_289_;
}
else
{
lean_object* v_reuseFailAlloc_291_; 
v_reuseFailAlloc_291_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_291_, 0, v___x_288_);
v___x_290_ = v_reuseFailAlloc_291_;
goto v_reusejp_289_;
}
v_reusejp_289_:
{
return v___x_290_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__6_spec__7___boxed(lean_object* v_oldTraces_298_, lean_object* v_data_299_, lean_object* v_ref_300_, lean_object* v_msg_301_, lean_object* v___y_302_, lean_object* v___y_303_, lean_object* v___y_304_, lean_object* v___y_305_, lean_object* v___y_306_){
_start:
{
lean_object* v_res_307_; 
v_res_307_ = lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__6_spec__7(v_oldTraces_298_, v_data_299_, v_ref_300_, v_msg_301_, v___y_302_, v___y_303_, v___y_304_, v___y_305_);
lean_dec(v___y_305_);
lean_dec_ref(v___y_304_);
lean_dec(v___y_303_);
lean_dec_ref(v___y_302_);
return v_res_307_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__6_spec__9(lean_object* v_e_308_){
_start:
{
if (lean_obj_tag(v_e_308_) == 0)
{
uint8_t v___x_309_; 
v___x_309_ = 2;
return v___x_309_;
}
else
{
uint8_t v___x_310_; 
v___x_310_ = 0;
return v___x_310_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__6_spec__9___boxed(lean_object* v_e_311_){
_start:
{
uint8_t v_res_312_; lean_object* v_r_313_; 
v_res_312_ = lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__6_spec__9(v_e_311_);
lean_dec_ref(v_e_311_);
v_r_313_ = lean_box(v_res_312_);
return v_r_313_;
}
}
static double _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__6___closed__0(void){
_start:
{
lean_object* v___x_314_; double v___x_315_; 
v___x_314_ = lean_unsigned_to_nat(0u);
v___x_315_ = lean_float_of_nat(v___x_314_);
return v___x_315_;
}
}
static lean_object* _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__6___closed__2(void){
_start:
{
lean_object* v___x_317_; lean_object* v___x_318_; 
v___x_317_ = ((lean_object*)(lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__6___closed__1));
v___x_318_ = l_Lean_stringToMessageData(v___x_317_);
return v___x_318_;
}
}
static double _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__6___closed__3(void){
_start:
{
lean_object* v___x_319_; double v___x_320_; 
v___x_319_ = lean_unsigned_to_nat(1000u);
v___x_320_ = lean_float_of_nat(v___x_319_);
return v___x_320_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__6(lean_object* v_cls_321_, uint8_t v_collapsed_322_, lean_object* v_tag_323_, lean_object* v_opts_324_, uint8_t v_clsEnabled_325_, lean_object* v_oldTraces_326_, lean_object* v_msg_327_, lean_object* v_resStartStop_328_, lean_object* v___y_329_, lean_object* v___y_330_, lean_object* v___y_331_, lean_object* v___y_332_){
_start:
{
lean_object* v_fst_334_; lean_object* v_snd_335_; lean_object* v___y_337_; lean_object* v___y_338_; lean_object* v_data_339_; lean_object* v_fst_350_; lean_object* v_snd_351_; lean_object* v___x_352_; uint8_t v___x_353_; lean_object* v___y_355_; lean_object* v_a_356_; uint8_t v___y_371_; double v___y_402_; 
v_fst_334_ = lean_ctor_get(v_resStartStop_328_, 0);
lean_inc(v_fst_334_);
v_snd_335_ = lean_ctor_get(v_resStartStop_328_, 1);
lean_inc(v_snd_335_);
lean_dec_ref(v_resStartStop_328_);
v_fst_350_ = lean_ctor_get(v_snd_335_, 0);
lean_inc(v_fst_350_);
v_snd_351_ = lean_ctor_get(v_snd_335_, 1);
lean_inc(v_snd_351_);
lean_dec(v_snd_335_);
v___x_352_ = l_Lean_trace_profiler;
v___x_353_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__5(v_opts_324_, v___x_352_);
if (v___x_353_ == 0)
{
v___y_371_ = v___x_353_;
goto v___jp_370_;
}
else
{
lean_object* v___x_407_; uint8_t v___x_408_; 
v___x_407_ = l_Lean_trace_profiler_useHeartbeats;
v___x_408_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__5(v_opts_324_, v___x_407_);
if (v___x_408_ == 0)
{
lean_object* v___x_409_; lean_object* v___x_410_; double v___x_411_; double v___x_412_; double v___x_413_; 
v___x_409_ = l_Lean_trace_profiler_threshold;
v___x_410_ = lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__6_spec__10(v_opts_324_, v___x_409_);
v___x_411_ = lean_float_of_nat(v___x_410_);
v___x_412_ = lean_float_once(&lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__6___closed__3, &lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__6___closed__3_once, _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__6___closed__3);
v___x_413_ = lean_float_div(v___x_411_, v___x_412_);
v___y_402_ = v___x_413_;
goto v___jp_401_;
}
else
{
lean_object* v___x_414_; lean_object* v___x_415_; double v___x_416_; 
v___x_414_ = l_Lean_trace_profiler_threshold;
v___x_415_ = lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__6_spec__10(v_opts_324_, v___x_414_);
v___x_416_ = lean_float_of_nat(v___x_415_);
v___y_402_ = v___x_416_;
goto v___jp_401_;
}
}
v___jp_336_:
{
lean_object* v___x_340_; 
lean_inc(v___y_338_);
v___x_340_ = lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__6_spec__7(v_oldTraces_326_, v_data_339_, v___y_338_, v___y_337_, v___y_329_, v___y_330_, v___y_331_, v___y_332_);
if (lean_obj_tag(v___x_340_) == 0)
{
lean_object* v___x_341_; 
lean_dec_ref_known(v___x_340_, 1);
v___x_341_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__6_spec__8___redArg(v_fst_334_);
return v___x_341_;
}
else
{
lean_object* v_a_342_; lean_object* v___x_344_; uint8_t v_isShared_345_; uint8_t v_isSharedCheck_349_; 
lean_dec(v_fst_334_);
v_a_342_ = lean_ctor_get(v___x_340_, 0);
v_isSharedCheck_349_ = !lean_is_exclusive(v___x_340_);
if (v_isSharedCheck_349_ == 0)
{
v___x_344_ = v___x_340_;
v_isShared_345_ = v_isSharedCheck_349_;
goto v_resetjp_343_;
}
else
{
lean_inc(v_a_342_);
lean_dec(v___x_340_);
v___x_344_ = lean_box(0);
v_isShared_345_ = v_isSharedCheck_349_;
goto v_resetjp_343_;
}
v_resetjp_343_:
{
lean_object* v___x_347_; 
if (v_isShared_345_ == 0)
{
v___x_347_ = v___x_344_;
goto v_reusejp_346_;
}
else
{
lean_object* v_reuseFailAlloc_348_; 
v_reuseFailAlloc_348_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_348_, 0, v_a_342_);
v___x_347_ = v_reuseFailAlloc_348_;
goto v_reusejp_346_;
}
v_reusejp_346_:
{
return v___x_347_;
}
}
}
}
v___jp_354_:
{
uint8_t v_result_357_; lean_object* v___x_358_; lean_object* v___x_359_; double v___x_360_; lean_object* v_data_361_; 
v_result_357_ = lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__6_spec__9(v_fst_334_);
v___x_358_ = lean_box(v_result_357_);
v___x_359_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_359_, 0, v___x_358_);
v___x_360_ = lean_float_once(&lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__6___closed__0, &lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__6___closed__0_once, _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__6___closed__0);
lean_inc_ref(v_tag_323_);
lean_inc_ref(v___x_359_);
lean_inc(v_cls_321_);
v_data_361_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v_data_361_, 0, v_cls_321_);
lean_ctor_set(v_data_361_, 1, v___x_359_);
lean_ctor_set(v_data_361_, 2, v_tag_323_);
lean_ctor_set_float(v_data_361_, sizeof(void*)*3, v___x_360_);
lean_ctor_set_float(v_data_361_, sizeof(void*)*3 + 8, v___x_360_);
lean_ctor_set_uint8(v_data_361_, sizeof(void*)*3 + 16, v_collapsed_322_);
if (v___x_353_ == 0)
{
lean_dec_ref_known(v___x_359_, 1);
lean_dec(v_snd_351_);
lean_dec(v_fst_350_);
lean_dec_ref(v_tag_323_);
lean_dec(v_cls_321_);
v___y_337_ = v_a_356_;
v___y_338_ = v___y_355_;
v_data_339_ = v_data_361_;
goto v___jp_336_;
}
else
{
lean_object* v_data_362_; double v___x_363_; double v___x_364_; 
lean_dec_ref_known(v_data_361_, 3);
v_data_362_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v_data_362_, 0, v_cls_321_);
lean_ctor_set(v_data_362_, 1, v___x_359_);
lean_ctor_set(v_data_362_, 2, v_tag_323_);
v___x_363_ = lean_unbox_float(v_fst_350_);
lean_dec(v_fst_350_);
lean_ctor_set_float(v_data_362_, sizeof(void*)*3, v___x_363_);
v___x_364_ = lean_unbox_float(v_snd_351_);
lean_dec(v_snd_351_);
lean_ctor_set_float(v_data_362_, sizeof(void*)*3 + 8, v___x_364_);
lean_ctor_set_uint8(v_data_362_, sizeof(void*)*3 + 16, v_collapsed_322_);
v___y_337_ = v_a_356_;
v___y_338_ = v___y_355_;
v_data_339_ = v_data_362_;
goto v___jp_336_;
}
}
v___jp_365_:
{
lean_object* v_ref_366_; lean_object* v___x_367_; 
v_ref_366_ = lean_ctor_get(v___y_331_, 5);
lean_inc(v___y_332_);
lean_inc_ref(v___y_331_);
lean_inc(v___y_330_);
lean_inc_ref(v___y_329_);
lean_inc(v_fst_334_);
v___x_367_ = lean_apply_6(v_msg_327_, v_fst_334_, v___y_329_, v___y_330_, v___y_331_, v___y_332_, lean_box(0));
if (lean_obj_tag(v___x_367_) == 0)
{
lean_object* v_a_368_; 
v_a_368_ = lean_ctor_get(v___x_367_, 0);
lean_inc(v_a_368_);
lean_dec_ref_known(v___x_367_, 1);
v___y_355_ = v_ref_366_;
v_a_356_ = v_a_368_;
goto v___jp_354_;
}
else
{
lean_object* v___x_369_; 
lean_dec_ref_known(v___x_367_, 1);
v___x_369_ = lean_obj_once(&lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__6___closed__2, &lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__6___closed__2_once, _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__6___closed__2);
v___y_355_ = v_ref_366_;
v_a_356_ = v___x_369_;
goto v___jp_354_;
}
}
v___jp_370_:
{
if (v_clsEnabled_325_ == 0)
{
if (v___y_371_ == 0)
{
lean_object* v___x_372_; lean_object* v_traceState_373_; lean_object* v_env_374_; lean_object* v_nextMacroScope_375_; lean_object* v_ngen_376_; lean_object* v_auxDeclNGen_377_; lean_object* v_cache_378_; lean_object* v_messages_379_; lean_object* v_infoState_380_; lean_object* v_snapshotTasks_381_; lean_object* v___x_383_; uint8_t v_isShared_384_; uint8_t v_isSharedCheck_400_; 
lean_dec(v_snd_351_);
lean_dec(v_fst_350_);
lean_dec_ref(v_msg_327_);
lean_dec_ref(v_tag_323_);
lean_dec(v_cls_321_);
v___x_372_ = lean_st_ref_take(v___y_332_);
v_traceState_373_ = lean_ctor_get(v___x_372_, 4);
v_env_374_ = lean_ctor_get(v___x_372_, 0);
v_nextMacroScope_375_ = lean_ctor_get(v___x_372_, 1);
v_ngen_376_ = lean_ctor_get(v___x_372_, 2);
v_auxDeclNGen_377_ = lean_ctor_get(v___x_372_, 3);
v_cache_378_ = lean_ctor_get(v___x_372_, 5);
v_messages_379_ = lean_ctor_get(v___x_372_, 6);
v_infoState_380_ = lean_ctor_get(v___x_372_, 7);
v_snapshotTasks_381_ = lean_ctor_get(v___x_372_, 8);
v_isSharedCheck_400_ = !lean_is_exclusive(v___x_372_);
if (v_isSharedCheck_400_ == 0)
{
v___x_383_ = v___x_372_;
v_isShared_384_ = v_isSharedCheck_400_;
goto v_resetjp_382_;
}
else
{
lean_inc(v_snapshotTasks_381_);
lean_inc(v_infoState_380_);
lean_inc(v_messages_379_);
lean_inc(v_cache_378_);
lean_inc(v_traceState_373_);
lean_inc(v_auxDeclNGen_377_);
lean_inc(v_ngen_376_);
lean_inc(v_nextMacroScope_375_);
lean_inc(v_env_374_);
lean_dec(v___x_372_);
v___x_383_ = lean_box(0);
v_isShared_384_ = v_isSharedCheck_400_;
goto v_resetjp_382_;
}
v_resetjp_382_:
{
uint64_t v_tid_385_; lean_object* v_traces_386_; lean_object* v___x_388_; uint8_t v_isShared_389_; uint8_t v_isSharedCheck_399_; 
v_tid_385_ = lean_ctor_get_uint64(v_traceState_373_, sizeof(void*)*1);
v_traces_386_ = lean_ctor_get(v_traceState_373_, 0);
v_isSharedCheck_399_ = !lean_is_exclusive(v_traceState_373_);
if (v_isSharedCheck_399_ == 0)
{
v___x_388_ = v_traceState_373_;
v_isShared_389_ = v_isSharedCheck_399_;
goto v_resetjp_387_;
}
else
{
lean_inc(v_traces_386_);
lean_dec(v_traceState_373_);
v___x_388_ = lean_box(0);
v_isShared_389_ = v_isSharedCheck_399_;
goto v_resetjp_387_;
}
v_resetjp_387_:
{
lean_object* v___x_390_; lean_object* v___x_392_; 
v___x_390_ = l_Lean_PersistentArray_append___redArg(v_oldTraces_326_, v_traces_386_);
lean_dec_ref(v_traces_386_);
if (v_isShared_389_ == 0)
{
lean_ctor_set(v___x_388_, 0, v___x_390_);
v___x_392_ = v___x_388_;
goto v_reusejp_391_;
}
else
{
lean_object* v_reuseFailAlloc_398_; 
v_reuseFailAlloc_398_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_398_, 0, v___x_390_);
lean_ctor_set_uint64(v_reuseFailAlloc_398_, sizeof(void*)*1, v_tid_385_);
v___x_392_ = v_reuseFailAlloc_398_;
goto v_reusejp_391_;
}
v_reusejp_391_:
{
lean_object* v___x_394_; 
if (v_isShared_384_ == 0)
{
lean_ctor_set(v___x_383_, 4, v___x_392_);
v___x_394_ = v___x_383_;
goto v_reusejp_393_;
}
else
{
lean_object* v_reuseFailAlloc_397_; 
v_reuseFailAlloc_397_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_397_, 0, v_env_374_);
lean_ctor_set(v_reuseFailAlloc_397_, 1, v_nextMacroScope_375_);
lean_ctor_set(v_reuseFailAlloc_397_, 2, v_ngen_376_);
lean_ctor_set(v_reuseFailAlloc_397_, 3, v_auxDeclNGen_377_);
lean_ctor_set(v_reuseFailAlloc_397_, 4, v___x_392_);
lean_ctor_set(v_reuseFailAlloc_397_, 5, v_cache_378_);
lean_ctor_set(v_reuseFailAlloc_397_, 6, v_messages_379_);
lean_ctor_set(v_reuseFailAlloc_397_, 7, v_infoState_380_);
lean_ctor_set(v_reuseFailAlloc_397_, 8, v_snapshotTasks_381_);
v___x_394_ = v_reuseFailAlloc_397_;
goto v_reusejp_393_;
}
v_reusejp_393_:
{
lean_object* v___x_395_; lean_object* v___x_396_; 
v___x_395_ = lean_st_ref_set(v___y_332_, v___x_394_);
v___x_396_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__6_spec__8___redArg(v_fst_334_);
return v___x_396_;
}
}
}
}
}
else
{
goto v___jp_365_;
}
}
else
{
goto v___jp_365_;
}
}
v___jp_401_:
{
double v___x_403_; double v___x_404_; double v___x_405_; uint8_t v___x_406_; 
v___x_403_ = lean_unbox_float(v_snd_351_);
v___x_404_ = lean_unbox_float(v_fst_350_);
v___x_405_ = lean_float_sub(v___x_403_, v___x_404_);
v___x_406_ = lean_float_decLt(v___y_402_, v___x_405_);
v___y_371_ = v___x_406_;
goto v___jp_370_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__6___boxed(lean_object* v_cls_417_, lean_object* v_collapsed_418_, lean_object* v_tag_419_, lean_object* v_opts_420_, lean_object* v_clsEnabled_421_, lean_object* v_oldTraces_422_, lean_object* v_msg_423_, lean_object* v_resStartStop_424_, lean_object* v___y_425_, lean_object* v___y_426_, lean_object* v___y_427_, lean_object* v___y_428_, lean_object* v___y_429_){
_start:
{
uint8_t v_collapsed_boxed_430_; uint8_t v_clsEnabled_boxed_431_; lean_object* v_res_432_; 
v_collapsed_boxed_430_ = lean_unbox(v_collapsed_418_);
v_clsEnabled_boxed_431_ = lean_unbox(v_clsEnabled_421_);
v_res_432_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__6(v_cls_417_, v_collapsed_boxed_430_, v_tag_419_, v_opts_420_, v_clsEnabled_boxed_431_, v_oldTraces_422_, v_msg_423_, v_resStartStop_424_, v___y_425_, v___y_426_, v___y_427_, v___y_428_);
lean_dec(v___y_428_);
lean_dec_ref(v___y_427_);
lean_dec(v___y_426_);
lean_dec_ref(v___y_425_);
lean_dec_ref(v_opts_420_);
return v_res_432_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__2___redArg(lean_object* v_opt_433_, lean_object* v___y_434_){
_start:
{
lean_object* v_options_436_; lean_object* v_option_437_; uint8_t v___x_438_; lean_object* v___x_439_; lean_object* v___x_440_; 
v_options_436_ = lean_ctor_get(v___y_434_, 2);
v_option_437_ = lean_ctor_get(v_opt_433_, 1);
v___x_438_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__5(v_options_436_, v_option_437_);
v___x_439_ = lean_box(v___x_438_);
v___x_440_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_440_, 0, v___x_439_);
return v___x_440_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__2___redArg___boxed(lean_object* v_opt_441_, lean_object* v___y_442_, lean_object* v___y_443_){
_start:
{
lean_object* v_res_444_; 
v_res_444_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__2___redArg(v_opt_441_, v___y_442_);
lean_dec_ref(v___y_442_);
lean_dec_ref(v_opt_441_);
return v_res_444_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addTrace___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__3(lean_object* v_cls_448_, lean_object* v_msg_449_, lean_object* v___y_450_, lean_object* v___y_451_, lean_object* v___y_452_, lean_object* v___y_453_){
_start:
{
lean_object* v_ref_455_; lean_object* v___x_456_; lean_object* v_a_457_; lean_object* v___x_459_; uint8_t v_isShared_460_; uint8_t v_isSharedCheck_501_; 
v_ref_455_ = lean_ctor_get(v___y_452_, 5);
v___x_456_ = lp_aesop_Lean_addMessageContextFull___at___00Lean_addTrace___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__3_spec__3(v_msg_449_, v___y_450_, v___y_451_, v___y_452_, v___y_453_);
v_a_457_ = lean_ctor_get(v___x_456_, 0);
v_isSharedCheck_501_ = !lean_is_exclusive(v___x_456_);
if (v_isSharedCheck_501_ == 0)
{
v___x_459_ = v___x_456_;
v_isShared_460_ = v_isSharedCheck_501_;
goto v_resetjp_458_;
}
else
{
lean_inc(v_a_457_);
lean_dec(v___x_456_);
v___x_459_ = lean_box(0);
v_isShared_460_ = v_isSharedCheck_501_;
goto v_resetjp_458_;
}
v_resetjp_458_:
{
lean_object* v___x_461_; lean_object* v_traceState_462_; lean_object* v_env_463_; lean_object* v_nextMacroScope_464_; lean_object* v_ngen_465_; lean_object* v_auxDeclNGen_466_; lean_object* v_cache_467_; lean_object* v_messages_468_; lean_object* v_infoState_469_; lean_object* v_snapshotTasks_470_; lean_object* v___x_472_; uint8_t v_isShared_473_; uint8_t v_isSharedCheck_500_; 
v___x_461_ = lean_st_ref_take(v___y_453_);
v_traceState_462_ = lean_ctor_get(v___x_461_, 4);
v_env_463_ = lean_ctor_get(v___x_461_, 0);
v_nextMacroScope_464_ = lean_ctor_get(v___x_461_, 1);
v_ngen_465_ = lean_ctor_get(v___x_461_, 2);
v_auxDeclNGen_466_ = lean_ctor_get(v___x_461_, 3);
v_cache_467_ = lean_ctor_get(v___x_461_, 5);
v_messages_468_ = lean_ctor_get(v___x_461_, 6);
v_infoState_469_ = lean_ctor_get(v___x_461_, 7);
v_snapshotTasks_470_ = lean_ctor_get(v___x_461_, 8);
v_isSharedCheck_500_ = !lean_is_exclusive(v___x_461_);
if (v_isSharedCheck_500_ == 0)
{
v___x_472_ = v___x_461_;
v_isShared_473_ = v_isSharedCheck_500_;
goto v_resetjp_471_;
}
else
{
lean_inc(v_snapshotTasks_470_);
lean_inc(v_infoState_469_);
lean_inc(v_messages_468_);
lean_inc(v_cache_467_);
lean_inc(v_traceState_462_);
lean_inc(v_auxDeclNGen_466_);
lean_inc(v_ngen_465_);
lean_inc(v_nextMacroScope_464_);
lean_inc(v_env_463_);
lean_dec(v___x_461_);
v___x_472_ = lean_box(0);
v_isShared_473_ = v_isSharedCheck_500_;
goto v_resetjp_471_;
}
v_resetjp_471_:
{
uint64_t v_tid_474_; lean_object* v_traces_475_; lean_object* v___x_477_; uint8_t v_isShared_478_; uint8_t v_isSharedCheck_499_; 
v_tid_474_ = lean_ctor_get_uint64(v_traceState_462_, sizeof(void*)*1);
v_traces_475_ = lean_ctor_get(v_traceState_462_, 0);
v_isSharedCheck_499_ = !lean_is_exclusive(v_traceState_462_);
if (v_isSharedCheck_499_ == 0)
{
v___x_477_ = v_traceState_462_;
v_isShared_478_ = v_isSharedCheck_499_;
goto v_resetjp_476_;
}
else
{
lean_inc(v_traces_475_);
lean_dec(v_traceState_462_);
v___x_477_ = lean_box(0);
v_isShared_478_ = v_isSharedCheck_499_;
goto v_resetjp_476_;
}
v_resetjp_476_:
{
lean_object* v___x_479_; double v___x_480_; uint8_t v___x_481_; lean_object* v___x_482_; lean_object* v___x_483_; lean_object* v___x_484_; lean_object* v___x_485_; lean_object* v___x_486_; lean_object* v___x_487_; lean_object* v___x_489_; 
v___x_479_ = lean_box(0);
v___x_480_ = lean_float_once(&lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__6___closed__0, &lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__6___closed__0_once, _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__6___closed__0);
v___x_481_ = 0;
v___x_482_ = ((lean_object*)(lp_aesop_Lean_addTrace___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__3___closed__0));
v___x_483_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v___x_483_, 0, v_cls_448_);
lean_ctor_set(v___x_483_, 1, v___x_479_);
lean_ctor_set(v___x_483_, 2, v___x_482_);
lean_ctor_set_float(v___x_483_, sizeof(void*)*3, v___x_480_);
lean_ctor_set_float(v___x_483_, sizeof(void*)*3 + 8, v___x_480_);
lean_ctor_set_uint8(v___x_483_, sizeof(void*)*3 + 16, v___x_481_);
v___x_484_ = ((lean_object*)(lp_aesop_Lean_addTrace___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__3___closed__1));
v___x_485_ = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(v___x_485_, 0, v___x_483_);
lean_ctor_set(v___x_485_, 1, v_a_457_);
lean_ctor_set(v___x_485_, 2, v___x_484_);
lean_inc(v_ref_455_);
v___x_486_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_486_, 0, v_ref_455_);
lean_ctor_set(v___x_486_, 1, v___x_485_);
v___x_487_ = l_Lean_PersistentArray_push___redArg(v_traces_475_, v___x_486_);
if (v_isShared_478_ == 0)
{
lean_ctor_set(v___x_477_, 0, v___x_487_);
v___x_489_ = v___x_477_;
goto v_reusejp_488_;
}
else
{
lean_object* v_reuseFailAlloc_498_; 
v_reuseFailAlloc_498_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_498_, 0, v___x_487_);
lean_ctor_set_uint64(v_reuseFailAlloc_498_, sizeof(void*)*1, v_tid_474_);
v___x_489_ = v_reuseFailAlloc_498_;
goto v_reusejp_488_;
}
v_reusejp_488_:
{
lean_object* v___x_491_; 
if (v_isShared_473_ == 0)
{
lean_ctor_set(v___x_472_, 4, v___x_489_);
v___x_491_ = v___x_472_;
goto v_reusejp_490_;
}
else
{
lean_object* v_reuseFailAlloc_497_; 
v_reuseFailAlloc_497_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_497_, 0, v_env_463_);
lean_ctor_set(v_reuseFailAlloc_497_, 1, v_nextMacroScope_464_);
lean_ctor_set(v_reuseFailAlloc_497_, 2, v_ngen_465_);
lean_ctor_set(v_reuseFailAlloc_497_, 3, v_auxDeclNGen_466_);
lean_ctor_set(v_reuseFailAlloc_497_, 4, v___x_489_);
lean_ctor_set(v_reuseFailAlloc_497_, 5, v_cache_467_);
lean_ctor_set(v_reuseFailAlloc_497_, 6, v_messages_468_);
lean_ctor_set(v_reuseFailAlloc_497_, 7, v_infoState_469_);
lean_ctor_set(v_reuseFailAlloc_497_, 8, v_snapshotTasks_470_);
v___x_491_ = v_reuseFailAlloc_497_;
goto v_reusejp_490_;
}
v_reusejp_490_:
{
lean_object* v___x_492_; lean_object* v___x_493_; lean_object* v___x_495_; 
v___x_492_ = lean_st_ref_set(v___y_453_, v___x_491_);
v___x_493_ = lean_box(0);
if (v_isShared_460_ == 0)
{
lean_ctor_set(v___x_459_, 0, v___x_493_);
v___x_495_ = v___x_459_;
goto v_reusejp_494_;
}
else
{
lean_object* v_reuseFailAlloc_496_; 
v_reuseFailAlloc_496_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_496_, 0, v___x_493_);
v___x_495_ = v_reuseFailAlloc_496_;
goto v_reusejp_494_;
}
v_reusejp_494_:
{
return v___x_495_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addTrace___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__3___boxed(lean_object* v_cls_502_, lean_object* v_msg_503_, lean_object* v___y_504_, lean_object* v___y_505_, lean_object* v___y_506_, lean_object* v___y_507_, lean_object* v___y_508_){
_start:
{
lean_object* v_res_509_; 
v_res_509_ = lp_aesop_Lean_addTrace___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__3(v_cls_502_, v_msg_503_, v___y_504_, v___y_505_, v___y_506_, v___y_507_);
lean_dec(v___y_507_);
lean_dec_ref(v___y_506_);
lean_dec(v___y_505_);
lean_dec_ref(v___y_504_);
return v_res_509_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___lam__2(lean_object* v___x_510_, lean_object* v_x_511_, lean_object* v___y_512_, lean_object* v___y_513_, lean_object* v___y_514_, lean_object* v___y_515_){
_start:
{
lean_object* v___x_517_; 
v___x_517_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_517_, 0, v___x_510_);
return v___x_517_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___lam__2___boxed(lean_object* v___x_518_, lean_object* v_x_519_, lean_object* v___y_520_, lean_object* v___y_521_, lean_object* v___y_522_, lean_object* v___y_523_, lean_object* v___y_524_){
_start:
{
lean_object* v_res_525_; 
v_res_525_ = lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___lam__2(v___x_518_, v_x_519_, v___y_520_, v___y_521_, v___y_522_, v___y_523_);
lean_dec(v___y_523_);
lean_dec_ref(v___y_522_);
lean_dec(v___y_521_);
lean_dec_ref(v___y_520_);
lean_dec_ref(v_x_519_);
return v_res_525_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__1(size_t v_sz_526_, size_t v_i_527_, lean_object* v_bs_528_, lean_object* v___y_529_, lean_object* v___y_530_, lean_object* v___y_531_, lean_object* v___y_532_){
_start:
{
uint8_t v___x_534_; 
v___x_534_ = lean_usize_dec_lt(v_i_527_, v_sz_526_);
if (v___x_534_ == 0)
{
lean_object* v___x_535_; 
v___x_535_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_535_, 0, v_bs_528_);
return v___x_535_;
}
else
{
lean_object* v_v_536_; lean_object* v___x_537_; 
v_v_536_ = lean_array_uget_borrowed(v_bs_528_, v_i_527_);
lean_inc(v_v_536_);
v___x_537_ = l_Lean_MVarId_intros(v_v_536_, v___y_529_, v___y_530_, v___y_531_, v___y_532_);
if (lean_obj_tag(v___x_537_) == 0)
{
lean_object* v_a_538_; lean_object* v_fst_539_; lean_object* v_snd_540_; lean_object* v___x_542_; uint8_t v_isShared_543_; uint8_t v_isSharedCheck_553_; 
v_a_538_ = lean_ctor_get(v___x_537_, 0);
lean_inc(v_a_538_);
lean_dec_ref_known(v___x_537_, 1);
v_fst_539_ = lean_ctor_get(v_a_538_, 0);
v_snd_540_ = lean_ctor_get(v_a_538_, 1);
v_isSharedCheck_553_ = !lean_is_exclusive(v_a_538_);
if (v_isSharedCheck_553_ == 0)
{
v___x_542_ = v_a_538_;
v_isShared_543_ = v_isSharedCheck_553_;
goto v_resetjp_541_;
}
else
{
lean_inc(v_snd_540_);
lean_inc(v_fst_539_);
lean_dec(v_a_538_);
v___x_542_ = lean_box(0);
v_isShared_543_ = v_isSharedCheck_553_;
goto v_resetjp_541_;
}
v_resetjp_541_:
{
lean_object* v___x_544_; lean_object* v_bs_x27_545_; lean_object* v___x_547_; 
v___x_544_ = lean_unsigned_to_nat(0u);
v_bs_x27_545_ = lean_array_uset(v_bs_528_, v_i_527_, v___x_544_);
if (v_isShared_543_ == 0)
{
lean_ctor_set(v___x_542_, 1, v_fst_539_);
lean_ctor_set(v___x_542_, 0, v_snd_540_);
v___x_547_ = v___x_542_;
goto v_reusejp_546_;
}
else
{
lean_object* v_reuseFailAlloc_552_; 
v_reuseFailAlloc_552_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_552_, 0, v_snd_540_);
lean_ctor_set(v_reuseFailAlloc_552_, 1, v_fst_539_);
v___x_547_ = v_reuseFailAlloc_552_;
goto v_reusejp_546_;
}
v_reusejp_546_:
{
size_t v___x_548_; size_t v___x_549_; lean_object* v___x_550_; 
v___x_548_ = ((size_t)1ULL);
v___x_549_ = lean_usize_add(v_i_527_, v___x_548_);
v___x_550_ = lean_array_uset(v_bs_x27_545_, v_i_527_, v___x_547_);
v_i_527_ = v___x_549_;
v_bs_528_ = v___x_550_;
goto _start;
}
}
}
else
{
lean_object* v_a_554_; lean_object* v___x_556_; uint8_t v_isShared_557_; uint8_t v_isSharedCheck_561_; 
lean_dec_ref(v_bs_528_);
v_a_554_ = lean_ctor_get(v___x_537_, 0);
v_isSharedCheck_561_ = !lean_is_exclusive(v___x_537_);
if (v_isSharedCheck_561_ == 0)
{
v___x_556_ = v___x_537_;
v_isShared_557_ = v_isSharedCheck_561_;
goto v_resetjp_555_;
}
else
{
lean_inc(v_a_554_);
lean_dec(v___x_537_);
v___x_556_ = lean_box(0);
v_isShared_557_ = v_isSharedCheck_561_;
goto v_resetjp_555_;
}
v_resetjp_555_:
{
lean_object* v___x_559_; 
if (v_isShared_557_ == 0)
{
v___x_559_ = v___x_556_;
goto v_reusejp_558_;
}
else
{
lean_object* v_reuseFailAlloc_560_; 
v_reuseFailAlloc_560_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_560_, 0, v_a_554_);
v___x_559_ = v_reuseFailAlloc_560_;
goto v_reusejp_558_;
}
v_reusejp_558_:
{
return v___x_559_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__1___boxed(lean_object* v_sz_562_, lean_object* v_i_563_, lean_object* v_bs_564_, lean_object* v___y_565_, lean_object* v___y_566_, lean_object* v___y_567_, lean_object* v___y_568_, lean_object* v___y_569_){
_start:
{
size_t v_sz_boxed_570_; size_t v_i_boxed_571_; lean_object* v_res_572_; 
v_sz_boxed_570_ = lean_unbox_usize(v_sz_562_);
lean_dec(v_sz_562_);
v_i_boxed_571_ = lean_unbox_usize(v_i_563_);
lean_dec(v_i_563_);
v_res_572_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__1(v_sz_boxed_570_, v_i_boxed_571_, v_bs_564_, v___y_565_, v___y_566_, v___y_567_, v___y_568_);
lean_dec(v___y_568_);
lean_dec_ref(v___y_567_);
lean_dec(v___y_566_);
lean_dec_ref(v___y_565_);
return v_res_572_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___lam__0___closed__2(void){
_start:
{
lean_object* v___x_576_; lean_object* v___x_577_; 
v___x_576_ = ((lean_object*)(lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___lam__0___closed__1));
v___x_577_ = l_Lean_stringToMessageData(v___x_576_);
return v___x_577_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___lam__0___closed__4(void){
_start:
{
lean_object* v___x_579_; lean_object* v___x_580_; 
v___x_579_ = ((lean_object*)(lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___lam__0___closed__3));
v___x_580_ = l_Lean_stringToMessageData(v___x_579_);
return v___x_580_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___closed__1(void){
_start:
{
lean_object* v___x_582_; lean_object* v___x_583_; 
v___x_582_ = ((lean_object*)(lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___closed__0));
v___x_583_ = l_Lean_stringToMessageData(v___x_582_);
return v___x_583_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___closed__3(void){
_start:
{
lean_object* v___x_585_; lean_object* v___x_586_; 
v___x_585_ = ((lean_object*)(lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___closed__2));
v___x_586_ = l_Lean_stringToMessageData(v___x_585_);
return v___x_586_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___closed__4(void){
_start:
{
lean_object* v___x_587_; lean_object* v___f_588_; 
v___x_587_ = lean_obj_once(&lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___closed__3, &lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___closed__3_once, _init_lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___closed__3);
v___f_588_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___lam__2___boxed), 7, 1);
lean_closure_set(v___f_588_, 0, v___x_587_);
return v___f_588_;
}
}
static double _init_lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___closed__7(void){
_start:
{
lean_object* v___x_592_; double v___x_593_; 
v___x_592_ = lean_unsigned_to_nat(1000000000u);
v___x_593_ = lean_float_of_nat(v___x_592_);
return v___x_593_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go(lean_object* v_goal_594_, lean_object* v_depth_595_, lean_object* v_commonFVarIds_596_, lean_object* v_a_597_, lean_object* v_a_598_, lean_object* v_a_599_, lean_object* v_a_600_){
_start:
{
lean_object* v___y_603_; lean_object* v___y_604_; lean_object* v___y_605_; lean_object* v___y_606_; lean_object* v___y_607_; lean_object* v___y_656_; lean_object* v___y_657_; lean_object* v___y_658_; lean_object* v___y_659_; lean_object* v___y_660_; lean_object* v_options_708_; lean_object* v_inheritedTraceOptions_709_; uint8_t v_hasTrace_710_; lean_object* v___x_711_; lean_object* v___y_713_; lean_object* v___y_714_; lean_object* v___y_715_; lean_object* v___y_716_; lean_object* v___y_777_; lean_object* v___y_778_; lean_object* v___y_779_; lean_object* v___y_780_; 
v_options_708_ = lean_ctor_get(v_a_599_, 2);
v_inheritedTraceOptions_709_ = lean_ctor_get(v_a_599_, 13);
v_hasTrace_710_ = lean_ctor_get_uint8(v_options_708_, sizeof(void*)*1);
v___x_711_ = lp_aesop_Aesop_TraceOption_debug;
if (v_hasTrace_710_ == 0)
{
lean_object* v___x_840_; 
v___x_840_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__2___redArg(v___x_711_, v_a_599_);
if (lean_obj_tag(v___x_840_) == 0)
{
lean_object* v_a_841_; uint8_t v___x_842_; 
v_a_841_ = lean_ctor_get(v___x_840_, 0);
lean_inc(v_a_841_);
lean_dec_ref_known(v___x_840_, 1);
v___x_842_ = lean_unbox(v_a_841_);
lean_dec(v_a_841_);
if (v___x_842_ == 0)
{
v___y_777_ = v_a_597_;
v___y_778_ = v_a_598_;
v___y_779_ = v_a_599_;
v___y_780_ = v_a_600_;
goto v___jp_776_;
}
else
{
lean_object* v_traceClass_843_; lean_object* v___x_844_; lean_object* v___x_845_; lean_object* v___x_846_; lean_object* v___x_847_; lean_object* v___x_848_; 
v_traceClass_843_ = lean_ctor_get(v___x_711_, 0);
v___x_844_ = lean_obj_once(&lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___closed__1, &lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___closed__1_once, _init_lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___closed__1);
lean_inc(v_goal_594_);
v___x_845_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_845_, 0, v_goal_594_);
v___x_846_ = l_Lean_indentD(v___x_845_);
v___x_847_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_847_, 0, v___x_844_);
lean_ctor_set(v___x_847_, 1, v___x_846_);
lean_inc(v_traceClass_843_);
v___x_848_ = lp_aesop_Lean_addTrace___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__3(v_traceClass_843_, v___x_847_, v_a_597_, v_a_598_, v_a_599_, v_a_600_);
if (lean_obj_tag(v___x_848_) == 0)
{
lean_dec_ref_known(v___x_848_, 1);
v___y_777_ = v_a_597_;
v___y_778_ = v_a_598_;
v___y_779_ = v_a_599_;
v___y_780_ = v_a_600_;
goto v___jp_776_;
}
else
{
lean_object* v_a_849_; lean_object* v___x_851_; uint8_t v_isShared_852_; uint8_t v_isSharedCheck_856_; 
lean_dec_ref(v_commonFVarIds_596_);
lean_dec(v_depth_595_);
lean_dec(v_goal_594_);
v_a_849_ = lean_ctor_get(v___x_848_, 0);
v_isSharedCheck_856_ = !lean_is_exclusive(v___x_848_);
if (v_isSharedCheck_856_ == 0)
{
v___x_851_ = v___x_848_;
v_isShared_852_ = v_isSharedCheck_856_;
goto v_resetjp_850_;
}
else
{
lean_inc(v_a_849_);
lean_dec(v___x_848_);
v___x_851_ = lean_box(0);
v_isShared_852_ = v_isSharedCheck_856_;
goto v_resetjp_850_;
}
v_resetjp_850_:
{
lean_object* v___x_854_; 
if (v_isShared_852_ == 0)
{
v___x_854_ = v___x_851_;
goto v_reusejp_853_;
}
else
{
lean_object* v_reuseFailAlloc_855_; 
v_reuseFailAlloc_855_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_855_, 0, v_a_849_);
v___x_854_ = v_reuseFailAlloc_855_;
goto v_reusejp_853_;
}
v_reusejp_853_:
{
return v___x_854_;
}
}
}
}
}
else
{
lean_object* v_a_857_; lean_object* v___x_859_; uint8_t v_isShared_860_; uint8_t v_isSharedCheck_864_; 
lean_dec_ref(v_commonFVarIds_596_);
lean_dec(v_depth_595_);
lean_dec(v_goal_594_);
v_a_857_ = lean_ctor_get(v___x_840_, 0);
v_isSharedCheck_864_ = !lean_is_exclusive(v___x_840_);
if (v_isSharedCheck_864_ == 0)
{
v___x_859_ = v___x_840_;
v_isShared_860_ = v_isSharedCheck_864_;
goto v_resetjp_858_;
}
else
{
lean_inc(v_a_857_);
lean_dec(v___x_840_);
v___x_859_ = lean_box(0);
v_isShared_860_ = v_isSharedCheck_864_;
goto v_resetjp_858_;
}
v_resetjp_858_:
{
lean_object* v___x_862_; 
if (v_isShared_860_ == 0)
{
v___x_862_ = v___x_859_;
goto v_reusejp_861_;
}
else
{
lean_object* v_reuseFailAlloc_863_; 
v_reuseFailAlloc_863_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_863_, 0, v_a_857_);
v___x_862_ = v_reuseFailAlloc_863_;
goto v_reusejp_861_;
}
v_reusejp_861_:
{
return v___x_862_;
}
}
}
}
else
{
lean_object* v_traceClass_865_; lean_object* v___f_866_; lean_object* v___x_867_; lean_object* v___x_868_; lean_object* v___x_869_; uint8_t v___x_870_; lean_object* v___y_872_; lean_object* v___y_873_; lean_object* v_a_874_; lean_object* v___y_887_; lean_object* v___y_888_; lean_object* v_a_889_; lean_object* v___y_892_; lean_object* v___y_893_; lean_object* v___y_894_; lean_object* v___y_905_; lean_object* v___y_906_; lean_object* v_a_907_; lean_object* v___y_917_; lean_object* v___y_918_; lean_object* v_a_919_; lean_object* v___y_922_; lean_object* v___y_923_; lean_object* v___y_924_; 
v_traceClass_865_ = lean_ctor_get(v___x_711_, 0);
v___f_866_ = lean_obj_once(&lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___closed__4, &lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___closed__4_once, _init_lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___closed__4);
v___x_867_ = ((lean_object*)(lp_aesop_Lean_addTrace___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__3___closed__0));
v___x_868_ = ((lean_object*)(lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___closed__6));
lean_inc(v_traceClass_865_);
v___x_869_ = l_Lean_Name_append(v___x_868_, v_traceClass_865_);
v___x_870_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_709_, v_options_708_, v___x_869_);
lean_dec(v___x_869_);
if (v___x_870_ == 0)
{
lean_object* v___x_989_; uint8_t v___x_990_; 
v___x_989_ = l_Lean_trace_profiler;
v___x_990_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__5(v_options_708_, v___x_989_);
if (v___x_990_ == 0)
{
lean_object* v___x_991_; 
v___x_991_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__2___redArg(v___x_711_, v_a_599_);
if (lean_obj_tag(v___x_991_) == 0)
{
lean_object* v_a_992_; uint8_t v___x_993_; 
v_a_992_ = lean_ctor_get(v___x_991_, 0);
lean_inc(v_a_992_);
lean_dec_ref_known(v___x_991_, 1);
v___x_993_ = lean_unbox(v_a_992_);
lean_dec(v_a_992_);
if (v___x_993_ == 0)
{
v___y_713_ = v_a_597_;
v___y_714_ = v_a_598_;
v___y_715_ = v_a_599_;
v___y_716_ = v_a_600_;
goto v___jp_712_;
}
else
{
lean_object* v___x_994_; lean_object* v___x_995_; lean_object* v___x_996_; lean_object* v___x_997_; lean_object* v___x_998_; 
v___x_994_ = lean_obj_once(&lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___closed__1, &lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___closed__1_once, _init_lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___closed__1);
lean_inc(v_goal_594_);
v___x_995_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_995_, 0, v_goal_594_);
v___x_996_ = l_Lean_indentD(v___x_995_);
v___x_997_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_997_, 0, v___x_994_);
lean_ctor_set(v___x_997_, 1, v___x_996_);
lean_inc(v_traceClass_865_);
v___x_998_ = lp_aesop_Lean_addTrace___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__3(v_traceClass_865_, v___x_997_, v_a_597_, v_a_598_, v_a_599_, v_a_600_);
if (lean_obj_tag(v___x_998_) == 0)
{
lean_dec_ref_known(v___x_998_, 1);
v___y_713_ = v_a_597_;
v___y_714_ = v_a_598_;
v___y_715_ = v_a_599_;
v___y_716_ = v_a_600_;
goto v___jp_712_;
}
else
{
lean_object* v_a_999_; lean_object* v___x_1001_; uint8_t v_isShared_1002_; uint8_t v_isSharedCheck_1006_; 
lean_dec_ref(v_commonFVarIds_596_);
lean_dec(v_depth_595_);
lean_dec(v_goal_594_);
v_a_999_ = lean_ctor_get(v___x_998_, 0);
v_isSharedCheck_1006_ = !lean_is_exclusive(v___x_998_);
if (v_isSharedCheck_1006_ == 0)
{
v___x_1001_ = v___x_998_;
v_isShared_1002_ = v_isSharedCheck_1006_;
goto v_resetjp_1000_;
}
else
{
lean_inc(v_a_999_);
lean_dec(v___x_998_);
v___x_1001_ = lean_box(0);
v_isShared_1002_ = v_isSharedCheck_1006_;
goto v_resetjp_1000_;
}
v_resetjp_1000_:
{
lean_object* v___x_1004_; 
if (v_isShared_1002_ == 0)
{
v___x_1004_ = v___x_1001_;
goto v_reusejp_1003_;
}
else
{
lean_object* v_reuseFailAlloc_1005_; 
v_reuseFailAlloc_1005_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1005_, 0, v_a_999_);
v___x_1004_ = v_reuseFailAlloc_1005_;
goto v_reusejp_1003_;
}
v_reusejp_1003_:
{
return v___x_1004_;
}
}
}
}
}
else
{
lean_object* v_a_1007_; lean_object* v___x_1009_; uint8_t v_isShared_1010_; uint8_t v_isSharedCheck_1014_; 
lean_dec_ref(v_commonFVarIds_596_);
lean_dec(v_depth_595_);
lean_dec(v_goal_594_);
v_a_1007_ = lean_ctor_get(v___x_991_, 0);
v_isSharedCheck_1014_ = !lean_is_exclusive(v___x_991_);
if (v_isSharedCheck_1014_ == 0)
{
v___x_1009_ = v___x_991_;
v_isShared_1010_ = v_isSharedCheck_1014_;
goto v_resetjp_1008_;
}
else
{
lean_inc(v_a_1007_);
lean_dec(v___x_991_);
v___x_1009_ = lean_box(0);
v_isShared_1010_ = v_isSharedCheck_1014_;
goto v_resetjp_1008_;
}
v_resetjp_1008_:
{
lean_object* v___x_1012_; 
if (v_isShared_1010_ == 0)
{
v___x_1012_ = v___x_1009_;
goto v_reusejp_1011_;
}
else
{
lean_object* v_reuseFailAlloc_1013_; 
v_reuseFailAlloc_1013_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1013_, 0, v_a_1007_);
v___x_1012_ = v_reuseFailAlloc_1013_;
goto v_reusejp_1011_;
}
v_reusejp_1011_:
{
return v___x_1012_;
}
}
}
}
else
{
goto v___jp_934_;
}
}
else
{
goto v___jp_934_;
}
v___jp_871_:
{
lean_object* v___x_875_; double v___x_876_; double v___x_877_; double v___x_878_; double v___x_879_; double v___x_880_; lean_object* v___x_881_; lean_object* v___x_882_; lean_object* v___x_883_; lean_object* v___x_884_; lean_object* v___x_885_; 
v___x_875_ = lean_io_mono_nanos_now();
v___x_876_ = lean_float_of_nat(v___y_872_);
v___x_877_ = lean_float_once(&lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___closed__7, &lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___closed__7_once, _init_lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___closed__7);
v___x_878_ = lean_float_div(v___x_876_, v___x_877_);
v___x_879_ = lean_float_of_nat(v___x_875_);
v___x_880_ = lean_float_div(v___x_879_, v___x_877_);
v___x_881_ = lean_box_float(v___x_878_);
v___x_882_ = lean_box_float(v___x_880_);
v___x_883_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_883_, 0, v___x_881_);
lean_ctor_set(v___x_883_, 1, v___x_882_);
v___x_884_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_884_, 0, v_a_874_);
lean_ctor_set(v___x_884_, 1, v___x_883_);
lean_inc(v_traceClass_865_);
v___x_885_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__6(v_traceClass_865_, v_hasTrace_710_, v___x_867_, v_options_708_, v___x_870_, v___y_873_, v___f_866_, v___x_884_, v_a_597_, v_a_598_, v_a_599_, v_a_600_);
return v___x_885_;
}
v___jp_886_:
{
lean_object* v___x_890_; 
v___x_890_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_890_, 0, v_a_889_);
v___y_872_ = v___y_887_;
v___y_873_ = v___y_888_;
v_a_874_ = v___x_890_;
goto v___jp_871_;
}
v___jp_891_:
{
if (lean_obj_tag(v___y_894_) == 0)
{
lean_object* v_a_895_; lean_object* v___x_897_; uint8_t v_isShared_898_; uint8_t v_isSharedCheck_902_; 
v_a_895_ = lean_ctor_get(v___y_894_, 0);
v_isSharedCheck_902_ = !lean_is_exclusive(v___y_894_);
if (v_isSharedCheck_902_ == 0)
{
v___x_897_ = v___y_894_;
v_isShared_898_ = v_isSharedCheck_902_;
goto v_resetjp_896_;
}
else
{
lean_inc(v_a_895_);
lean_dec(v___y_894_);
v___x_897_ = lean_box(0);
v_isShared_898_ = v_isSharedCheck_902_;
goto v_resetjp_896_;
}
v_resetjp_896_:
{
lean_object* v___x_900_; 
if (v_isShared_898_ == 0)
{
lean_ctor_set_tag(v___x_897_, 1);
v___x_900_ = v___x_897_;
goto v_reusejp_899_;
}
else
{
lean_object* v_reuseFailAlloc_901_; 
v_reuseFailAlloc_901_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_901_, 0, v_a_895_);
v___x_900_ = v_reuseFailAlloc_901_;
goto v_reusejp_899_;
}
v_reusejp_899_:
{
v___y_872_ = v___y_892_;
v___y_873_ = v___y_893_;
v_a_874_ = v___x_900_;
goto v___jp_871_;
}
}
}
else
{
lean_object* v_a_903_; 
v_a_903_ = lean_ctor_get(v___y_894_, 0);
lean_inc(v_a_903_);
lean_dec_ref_known(v___y_894_, 1);
v___y_887_ = v___y_892_;
v___y_888_ = v___y_893_;
v_a_889_ = v_a_903_;
goto v___jp_886_;
}
}
v___jp_904_:
{
lean_object* v___x_908_; double v___x_909_; double v___x_910_; lean_object* v___x_911_; lean_object* v___x_912_; lean_object* v___x_913_; lean_object* v___x_914_; lean_object* v___x_915_; 
v___x_908_ = lean_io_get_num_heartbeats();
v___x_909_ = lean_float_of_nat(v___y_906_);
v___x_910_ = lean_float_of_nat(v___x_908_);
v___x_911_ = lean_box_float(v___x_909_);
v___x_912_ = lean_box_float(v___x_910_);
v___x_913_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_913_, 0, v___x_911_);
lean_ctor_set(v___x_913_, 1, v___x_912_);
v___x_914_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_914_, 0, v_a_907_);
lean_ctor_set(v___x_914_, 1, v___x_913_);
lean_inc(v_traceClass_865_);
v___x_915_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__6(v_traceClass_865_, v_hasTrace_710_, v___x_867_, v_options_708_, v___x_870_, v___y_905_, v___f_866_, v___x_914_, v_a_597_, v_a_598_, v_a_599_, v_a_600_);
return v___x_915_;
}
v___jp_916_:
{
lean_object* v___x_920_; 
v___x_920_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_920_, 0, v_a_919_);
v___y_905_ = v___y_918_;
v___y_906_ = v___y_917_;
v_a_907_ = v___x_920_;
goto v___jp_904_;
}
v___jp_921_:
{
if (lean_obj_tag(v___y_924_) == 0)
{
lean_object* v_a_925_; lean_object* v___x_927_; uint8_t v_isShared_928_; uint8_t v_isSharedCheck_932_; 
v_a_925_ = lean_ctor_get(v___y_924_, 0);
v_isSharedCheck_932_ = !lean_is_exclusive(v___y_924_);
if (v_isSharedCheck_932_ == 0)
{
v___x_927_ = v___y_924_;
v_isShared_928_ = v_isSharedCheck_932_;
goto v_resetjp_926_;
}
else
{
lean_inc(v_a_925_);
lean_dec(v___y_924_);
v___x_927_ = lean_box(0);
v_isShared_928_ = v_isSharedCheck_932_;
goto v_resetjp_926_;
}
v_resetjp_926_:
{
lean_object* v___x_930_; 
if (v_isShared_928_ == 0)
{
lean_ctor_set_tag(v___x_927_, 1);
v___x_930_ = v___x_927_;
goto v_reusejp_929_;
}
else
{
lean_object* v_reuseFailAlloc_931_; 
v_reuseFailAlloc_931_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_931_, 0, v_a_925_);
v___x_930_ = v_reuseFailAlloc_931_;
goto v_reusejp_929_;
}
v_reusejp_929_:
{
v___y_905_ = v___y_923_;
v___y_906_ = v___y_922_;
v_a_907_ = v___x_930_;
goto v___jp_904_;
}
}
}
else
{
lean_object* v_a_933_; 
v_a_933_ = lean_ctor_get(v___y_924_, 0);
lean_inc(v_a_933_);
lean_dec_ref_known(v___y_924_, 1);
v___y_917_ = v___y_922_;
v___y_918_ = v___y_923_;
v_a_919_ = v_a_933_;
goto v___jp_916_;
}
}
v___jp_934_:
{
lean_object* v___x_935_; 
v___x_935_ = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__4___redArg(v_a_600_);
if (lean_obj_tag(v___x_935_) == 0)
{
lean_object* v_a_936_; lean_object* v___x_937_; uint8_t v___x_938_; 
v_a_936_ = lean_ctor_get(v___x_935_, 0);
lean_inc(v_a_936_);
lean_dec_ref_known(v___x_935_, 1);
v___x_937_ = l_Lean_trace_profiler_useHeartbeats;
v___x_938_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__5(v_options_708_, v___x_937_);
if (v___x_938_ == 0)
{
lean_object* v___x_939_; lean_object* v___x_940_; 
v___x_939_ = lean_io_mono_nanos_now();
v___x_940_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__2___redArg(v___x_711_, v_a_599_);
if (lean_obj_tag(v___x_940_) == 0)
{
lean_object* v_a_941_; lean_object* v___x_943_; uint8_t v_isShared_944_; uint8_t v_isSharedCheck_958_; 
v_a_941_ = lean_ctor_get(v___x_940_, 0);
v_isSharedCheck_958_ = !lean_is_exclusive(v___x_940_);
if (v_isSharedCheck_958_ == 0)
{
v___x_943_ = v___x_940_;
v_isShared_944_ = v_isSharedCheck_958_;
goto v_resetjp_942_;
}
else
{
lean_inc(v_a_941_);
lean_dec(v___x_940_);
v___x_943_ = lean_box(0);
v_isShared_944_ = v_isSharedCheck_958_;
goto v_resetjp_942_;
}
v_resetjp_942_:
{
uint8_t v___x_945_; 
v___x_945_ = lean_unbox(v_a_941_);
lean_dec(v_a_941_);
if (v___x_945_ == 0)
{
lean_object* v___x_946_; lean_object* v___x_947_; 
lean_del_object(v___x_943_);
v___x_946_ = lean_box(0);
v___x_947_ = lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___lam__0(v_goal_594_, v___x_711_, v_depth_595_, v_commonFVarIds_596_, v___x_946_, v_a_597_, v_a_598_, v_a_599_, v_a_600_);
v___y_892_ = v___x_939_;
v___y_893_ = v_a_936_;
v___y_894_ = v___x_947_;
goto v___jp_891_;
}
else
{
lean_object* v___x_948_; lean_object* v___x_950_; 
v___x_948_ = lean_obj_once(&lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___closed__1, &lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___closed__1_once, _init_lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___closed__1);
lean_inc(v_goal_594_);
if (v_isShared_944_ == 0)
{
lean_ctor_set_tag(v___x_943_, 1);
lean_ctor_set(v___x_943_, 0, v_goal_594_);
v___x_950_ = v___x_943_;
goto v_reusejp_949_;
}
else
{
lean_object* v_reuseFailAlloc_957_; 
v_reuseFailAlloc_957_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_957_, 0, v_goal_594_);
v___x_950_ = v_reuseFailAlloc_957_;
goto v_reusejp_949_;
}
v_reusejp_949_:
{
lean_object* v___x_951_; lean_object* v___x_952_; lean_object* v___x_953_; 
v___x_951_ = l_Lean_indentD(v___x_950_);
v___x_952_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_952_, 0, v___x_948_);
lean_ctor_set(v___x_952_, 1, v___x_951_);
lean_inc(v_traceClass_865_);
v___x_953_ = lp_aesop_Lean_addTrace___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__3(v_traceClass_865_, v___x_952_, v_a_597_, v_a_598_, v_a_599_, v_a_600_);
if (lean_obj_tag(v___x_953_) == 0)
{
lean_object* v_a_954_; lean_object* v___x_955_; 
v_a_954_ = lean_ctor_get(v___x_953_, 0);
lean_inc(v_a_954_);
lean_dec_ref_known(v___x_953_, 1);
v___x_955_ = lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___lam__0(v_goal_594_, v___x_711_, v_depth_595_, v_commonFVarIds_596_, v_a_954_, v_a_597_, v_a_598_, v_a_599_, v_a_600_);
v___y_892_ = v___x_939_;
v___y_893_ = v_a_936_;
v___y_894_ = v___x_955_;
goto v___jp_891_;
}
else
{
lean_object* v_a_956_; 
lean_dec_ref(v_commonFVarIds_596_);
lean_dec(v_depth_595_);
lean_dec(v_goal_594_);
v_a_956_ = lean_ctor_get(v___x_953_, 0);
lean_inc(v_a_956_);
lean_dec_ref_known(v___x_953_, 1);
v___y_887_ = v___x_939_;
v___y_888_ = v_a_936_;
v_a_889_ = v_a_956_;
goto v___jp_886_;
}
}
}
}
}
else
{
lean_object* v_a_959_; 
lean_dec_ref(v_commonFVarIds_596_);
lean_dec(v_depth_595_);
lean_dec(v_goal_594_);
v_a_959_ = lean_ctor_get(v___x_940_, 0);
lean_inc(v_a_959_);
lean_dec_ref_known(v___x_940_, 1);
v___y_887_ = v___x_939_;
v___y_888_ = v_a_936_;
v_a_889_ = v_a_959_;
goto v___jp_886_;
}
}
else
{
lean_object* v___x_960_; lean_object* v___x_961_; 
v___x_960_ = lean_io_get_num_heartbeats();
v___x_961_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__2___redArg(v___x_711_, v_a_599_);
if (lean_obj_tag(v___x_961_) == 0)
{
lean_object* v_a_962_; lean_object* v___x_964_; uint8_t v_isShared_965_; uint8_t v_isSharedCheck_979_; 
v_a_962_ = lean_ctor_get(v___x_961_, 0);
v_isSharedCheck_979_ = !lean_is_exclusive(v___x_961_);
if (v_isSharedCheck_979_ == 0)
{
v___x_964_ = v___x_961_;
v_isShared_965_ = v_isSharedCheck_979_;
goto v_resetjp_963_;
}
else
{
lean_inc(v_a_962_);
lean_dec(v___x_961_);
v___x_964_ = lean_box(0);
v_isShared_965_ = v_isSharedCheck_979_;
goto v_resetjp_963_;
}
v_resetjp_963_:
{
uint8_t v___x_966_; 
v___x_966_ = lean_unbox(v_a_962_);
lean_dec(v_a_962_);
if (v___x_966_ == 0)
{
lean_object* v___x_967_; lean_object* v___x_968_; 
lean_del_object(v___x_964_);
v___x_967_ = lean_box(0);
v___x_968_ = lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___lam__0(v_goal_594_, v___x_711_, v_depth_595_, v_commonFVarIds_596_, v___x_967_, v_a_597_, v_a_598_, v_a_599_, v_a_600_);
v___y_922_ = v___x_960_;
v___y_923_ = v_a_936_;
v___y_924_ = v___x_968_;
goto v___jp_921_;
}
else
{
lean_object* v___x_969_; lean_object* v___x_971_; 
v___x_969_ = lean_obj_once(&lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___closed__1, &lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___closed__1_once, _init_lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___closed__1);
lean_inc(v_goal_594_);
if (v_isShared_965_ == 0)
{
lean_ctor_set_tag(v___x_964_, 1);
lean_ctor_set(v___x_964_, 0, v_goal_594_);
v___x_971_ = v___x_964_;
goto v_reusejp_970_;
}
else
{
lean_object* v_reuseFailAlloc_978_; 
v_reuseFailAlloc_978_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_978_, 0, v_goal_594_);
v___x_971_ = v_reuseFailAlloc_978_;
goto v_reusejp_970_;
}
v_reusejp_970_:
{
lean_object* v___x_972_; lean_object* v___x_973_; lean_object* v___x_974_; 
v___x_972_ = l_Lean_indentD(v___x_971_);
v___x_973_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_973_, 0, v___x_969_);
lean_ctor_set(v___x_973_, 1, v___x_972_);
lean_inc(v_traceClass_865_);
v___x_974_ = lp_aesop_Lean_addTrace___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__3(v_traceClass_865_, v___x_973_, v_a_597_, v_a_598_, v_a_599_, v_a_600_);
if (lean_obj_tag(v___x_974_) == 0)
{
lean_object* v_a_975_; lean_object* v___x_976_; 
v_a_975_ = lean_ctor_get(v___x_974_, 0);
lean_inc(v_a_975_);
lean_dec_ref_known(v___x_974_, 1);
v___x_976_ = lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___lam__0(v_goal_594_, v___x_711_, v_depth_595_, v_commonFVarIds_596_, v_a_975_, v_a_597_, v_a_598_, v_a_599_, v_a_600_);
v___y_922_ = v___x_960_;
v___y_923_ = v_a_936_;
v___y_924_ = v___x_976_;
goto v___jp_921_;
}
else
{
lean_object* v_a_977_; 
lean_dec_ref(v_commonFVarIds_596_);
lean_dec(v_depth_595_);
lean_dec(v_goal_594_);
v_a_977_ = lean_ctor_get(v___x_974_, 0);
lean_inc(v_a_977_);
lean_dec_ref_known(v___x_974_, 1);
v___y_917_ = v___x_960_;
v___y_918_ = v_a_936_;
v_a_919_ = v_a_977_;
goto v___jp_916_;
}
}
}
}
}
else
{
lean_object* v_a_980_; 
lean_dec_ref(v_commonFVarIds_596_);
lean_dec(v_depth_595_);
lean_dec(v_goal_594_);
v_a_980_ = lean_ctor_get(v___x_961_, 0);
lean_inc(v_a_980_);
lean_dec_ref_known(v___x_961_, 1);
v___y_917_ = v___x_960_;
v___y_918_ = v_a_936_;
v_a_919_ = v_a_980_;
goto v___jp_916_;
}
}
}
else
{
lean_object* v_a_981_; lean_object* v___x_983_; uint8_t v_isShared_984_; uint8_t v_isSharedCheck_988_; 
lean_dec_ref(v_commonFVarIds_596_);
lean_dec(v_depth_595_);
lean_dec(v_goal_594_);
v_a_981_ = lean_ctor_get(v___x_935_, 0);
v_isSharedCheck_988_ = !lean_is_exclusive(v___x_935_);
if (v_isSharedCheck_988_ == 0)
{
v___x_983_ = v___x_935_;
v_isShared_984_ = v_isSharedCheck_988_;
goto v_resetjp_982_;
}
else
{
lean_inc(v_a_981_);
lean_dec(v___x_935_);
v___x_983_ = lean_box(0);
v_isShared_984_ = v_isSharedCheck_988_;
goto v_resetjp_982_;
}
v_resetjp_982_:
{
lean_object* v___x_986_; 
if (v_isShared_984_ == 0)
{
v___x_986_ = v___x_983_;
goto v_reusejp_985_;
}
else
{
lean_object* v_reuseFailAlloc_987_; 
v_reuseFailAlloc_987_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_987_, 0, v_a_981_);
v___x_986_ = v_reuseFailAlloc_987_;
goto v_reusejp_985_;
}
v_reusejp_985_:
{
return v___x_986_;
}
}
}
}
}
v___jp_602_:
{
lean_object* v___x_608_; lean_object* v___x_609_; lean_object* v___x_610_; uint8_t v___x_611_; 
v___x_608_ = lean_array_mk(v___y_603_);
v___x_609_ = lean_array_get_size(v___x_608_);
v___x_610_ = lean_unsigned_to_nat(0u);
v___x_611_ = lean_nat_dec_eq(v___x_609_, v___x_610_);
if (v___x_611_ == 0)
{
size_t v_sz_612_; size_t v___x_613_; lean_object* v___x_614_; 
v_sz_612_ = lean_array_size(v___x_608_);
v___x_613_ = ((size_t)0ULL);
v___x_614_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__1(v_sz_612_, v___x_613_, v___x_608_, v___y_604_, v___y_605_, v___y_606_, v___y_607_);
if (lean_obj_tag(v___x_614_) == 0)
{
lean_object* v_a_615_; lean_object* v___x_617_; uint8_t v_isShared_618_; uint8_t v_isSharedCheck_633_; 
v_a_615_ = lean_ctor_get(v___x_614_, 0);
v_isSharedCheck_633_ = !lean_is_exclusive(v___x_614_);
if (v_isSharedCheck_633_ == 0)
{
v___x_617_ = v___x_614_;
v_isShared_618_ = v_isSharedCheck_633_;
goto v_resetjp_616_;
}
else
{
lean_inc(v_a_615_);
lean_dec(v___x_614_);
v___x_617_ = lean_box(0);
v_isShared_618_ = v_isSharedCheck_633_;
goto v_resetjp_616_;
}
v_resetjp_616_:
{
lean_object* v___x_619_; lean_object* v___x_620_; uint8_t v___x_621_; 
v___x_619_ = lean_array_get_size(v_a_615_);
v___x_620_ = lean_unsigned_to_nat(1u);
v___x_621_ = lean_nat_dec_eq(v___x_619_, v___x_620_);
if (v___x_621_ == 0)
{
lean_object* v___x_622_; lean_object* v___x_623_; lean_object* v___x_625_; 
v___x_622_ = lean_nat_add(v_depth_595_, v___x_620_);
lean_dec(v_depth_595_);
v___x_623_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_623_, 0, v___x_622_);
lean_ctor_set(v___x_623_, 1, v_commonFVarIds_596_);
lean_ctor_set(v___x_623_, 2, v_a_615_);
if (v_isShared_618_ == 0)
{
lean_ctor_set(v___x_617_, 0, v___x_623_);
v___x_625_ = v___x_617_;
goto v_reusejp_624_;
}
else
{
lean_object* v_reuseFailAlloc_626_; 
v_reuseFailAlloc_626_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_626_, 0, v___x_623_);
v___x_625_ = v_reuseFailAlloc_626_;
goto v_reusejp_624_;
}
v_reusejp_624_:
{
return v___x_625_;
}
}
else
{
lean_object* v___x_627_; lean_object* v_fst_628_; lean_object* v_snd_629_; lean_object* v___x_630_; lean_object* v___x_631_; 
lean_del_object(v___x_617_);
v___x_627_ = lean_array_fget(v_a_615_, v___x_610_);
lean_dec(v_a_615_);
v_fst_628_ = lean_ctor_get(v___x_627_, 0);
lean_inc(v_fst_628_);
v_snd_629_ = lean_ctor_get(v___x_627_, 1);
lean_inc(v_snd_629_);
lean_dec(v___x_627_);
v___x_630_ = lean_nat_add(v_depth_595_, v___x_620_);
lean_dec(v_depth_595_);
v___x_631_ = l_Array_append___redArg(v_commonFVarIds_596_, v_snd_629_);
lean_dec(v_snd_629_);
v_goal_594_ = v_fst_628_;
v_depth_595_ = v___x_630_;
v_commonFVarIds_596_ = v___x_631_;
v_a_597_ = v___y_604_;
v_a_598_ = v___y_605_;
v_a_599_ = v___y_606_;
v_a_600_ = v___y_607_;
goto _start;
}
}
}
else
{
lean_object* v_a_634_; lean_object* v___x_636_; uint8_t v_isShared_637_; uint8_t v_isSharedCheck_641_; 
lean_dec_ref(v_commonFVarIds_596_);
lean_dec(v_depth_595_);
v_a_634_ = lean_ctor_get(v___x_614_, 0);
v_isSharedCheck_641_ = !lean_is_exclusive(v___x_614_);
if (v_isSharedCheck_641_ == 0)
{
v___x_636_ = v___x_614_;
v_isShared_637_ = v_isSharedCheck_641_;
goto v_resetjp_635_;
}
else
{
lean_inc(v_a_634_);
lean_dec(v___x_614_);
v___x_636_ = lean_box(0);
v_isShared_637_ = v_isSharedCheck_641_;
goto v_resetjp_635_;
}
v_resetjp_635_:
{
lean_object* v___x_639_; 
if (v_isShared_637_ == 0)
{
v___x_639_ = v___x_636_;
goto v_reusejp_638_;
}
else
{
lean_object* v_reuseFailAlloc_640_; 
v_reuseFailAlloc_640_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_640_, 0, v_a_634_);
v___x_639_ = v_reuseFailAlloc_640_;
goto v_reusejp_638_;
}
v_reusejp_638_:
{
return v___x_639_;
}
}
}
}
else
{
lean_object* v___x_642_; lean_object* v___x_643_; lean_object* v___x_644_; lean_object* v___x_645_; lean_object* v___x_646_; 
lean_dec_ref(v___x_608_);
lean_dec_ref(v_commonFVarIds_596_);
v___x_642_ = lean_unsigned_to_nat(1u);
v___x_643_ = lean_nat_add(v_depth_595_, v___x_642_);
lean_dec(v_depth_595_);
v___x_644_ = ((lean_object*)(lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___lam__0___closed__0));
v___x_645_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_645_, 0, v___x_643_);
lean_ctor_set(v___x_645_, 1, v___x_644_);
lean_ctor_set(v___x_645_, 2, v___x_644_);
v___x_646_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_646_, 0, v___x_645_);
return v___x_646_;
}
}
v___jp_647_:
{
lean_object* v___x_648_; lean_object* v___x_649_; lean_object* v___x_650_; lean_object* v___x_651_; lean_object* v___x_652_; lean_object* v___x_653_; lean_object* v___x_654_; 
v___x_648_ = ((lean_object*)(lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___lam__0___closed__0));
v___x_649_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_649_, 0, v_goal_594_);
lean_ctor_set(v___x_649_, 1, v___x_648_);
v___x_650_ = lean_unsigned_to_nat(1u);
v___x_651_ = lean_mk_empty_array_with_capacity(v___x_650_);
v___x_652_ = lean_array_push(v___x_651_, v___x_649_);
v___x_653_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_653_, 0, v_depth_595_);
lean_ctor_set(v___x_653_, 1, v_commonFVarIds_596_);
lean_ctor_set(v___x_653_, 2, v___x_652_);
v___x_654_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_654_, 0, v___x_653_);
return v___x_654_;
}
v___jp_655_:
{
lean_object* v___x_661_; lean_object* v___x_662_; lean_object* v___x_663_; uint8_t v___x_664_; 
v___x_661_ = lean_array_mk(v___y_656_);
v___x_662_ = lean_array_get_size(v___x_661_);
v___x_663_ = lean_unsigned_to_nat(0u);
v___x_664_ = lean_nat_dec_eq(v___x_662_, v___x_663_);
if (v___x_664_ == 0)
{
size_t v_sz_665_; size_t v___x_666_; lean_object* v___x_667_; 
v_sz_665_ = lean_array_size(v___x_661_);
v___x_666_ = ((size_t)0ULL);
v___x_667_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__1(v_sz_665_, v___x_666_, v___x_661_, v___y_657_, v___y_658_, v___y_659_, v___y_660_);
if (lean_obj_tag(v___x_667_) == 0)
{
lean_object* v_a_668_; lean_object* v___x_670_; uint8_t v_isShared_671_; uint8_t v_isSharedCheck_686_; 
v_a_668_ = lean_ctor_get(v___x_667_, 0);
v_isSharedCheck_686_ = !lean_is_exclusive(v___x_667_);
if (v_isSharedCheck_686_ == 0)
{
v___x_670_ = v___x_667_;
v_isShared_671_ = v_isSharedCheck_686_;
goto v_resetjp_669_;
}
else
{
lean_inc(v_a_668_);
lean_dec(v___x_667_);
v___x_670_ = lean_box(0);
v_isShared_671_ = v_isSharedCheck_686_;
goto v_resetjp_669_;
}
v_resetjp_669_:
{
lean_object* v___x_672_; lean_object* v___x_673_; uint8_t v___x_674_; 
v___x_672_ = lean_array_get_size(v_a_668_);
v___x_673_ = lean_unsigned_to_nat(1u);
v___x_674_ = lean_nat_dec_eq(v___x_672_, v___x_673_);
if (v___x_674_ == 0)
{
lean_object* v___x_675_; lean_object* v___x_676_; lean_object* v___x_678_; 
v___x_675_ = lean_nat_add(v_depth_595_, v___x_673_);
lean_dec(v_depth_595_);
v___x_676_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_676_, 0, v___x_675_);
lean_ctor_set(v___x_676_, 1, v_commonFVarIds_596_);
lean_ctor_set(v___x_676_, 2, v_a_668_);
if (v_isShared_671_ == 0)
{
lean_ctor_set(v___x_670_, 0, v___x_676_);
v___x_678_ = v___x_670_;
goto v_reusejp_677_;
}
else
{
lean_object* v_reuseFailAlloc_679_; 
v_reuseFailAlloc_679_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_679_, 0, v___x_676_);
v___x_678_ = v_reuseFailAlloc_679_;
goto v_reusejp_677_;
}
v_reusejp_677_:
{
return v___x_678_;
}
}
else
{
lean_object* v___x_680_; lean_object* v_fst_681_; lean_object* v_snd_682_; lean_object* v___x_683_; lean_object* v___x_684_; 
lean_del_object(v___x_670_);
v___x_680_ = lean_array_fget(v_a_668_, v___x_663_);
lean_dec(v_a_668_);
v_fst_681_ = lean_ctor_get(v___x_680_, 0);
lean_inc(v_fst_681_);
v_snd_682_ = lean_ctor_get(v___x_680_, 1);
lean_inc(v_snd_682_);
lean_dec(v___x_680_);
v___x_683_ = lean_nat_add(v_depth_595_, v___x_673_);
lean_dec(v_depth_595_);
v___x_684_ = l_Array_append___redArg(v_commonFVarIds_596_, v_snd_682_);
lean_dec(v_snd_682_);
v_goal_594_ = v_fst_681_;
v_depth_595_ = v___x_683_;
v_commonFVarIds_596_ = v___x_684_;
v_a_597_ = v___y_657_;
v_a_598_ = v___y_658_;
v_a_599_ = v___y_659_;
v_a_600_ = v___y_660_;
goto _start;
}
}
}
else
{
lean_object* v_a_687_; lean_object* v___x_689_; uint8_t v_isShared_690_; uint8_t v_isSharedCheck_694_; 
lean_dec_ref(v_commonFVarIds_596_);
lean_dec(v_depth_595_);
v_a_687_ = lean_ctor_get(v___x_667_, 0);
v_isSharedCheck_694_ = !lean_is_exclusive(v___x_667_);
if (v_isSharedCheck_694_ == 0)
{
v___x_689_ = v___x_667_;
v_isShared_690_ = v_isSharedCheck_694_;
goto v_resetjp_688_;
}
else
{
lean_inc(v_a_687_);
lean_dec(v___x_667_);
v___x_689_ = lean_box(0);
v_isShared_690_ = v_isSharedCheck_694_;
goto v_resetjp_688_;
}
v_resetjp_688_:
{
lean_object* v___x_692_; 
if (v_isShared_690_ == 0)
{
v___x_692_ = v___x_689_;
goto v_reusejp_691_;
}
else
{
lean_object* v_reuseFailAlloc_693_; 
v_reuseFailAlloc_693_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_693_, 0, v_a_687_);
v___x_692_ = v_reuseFailAlloc_693_;
goto v_reusejp_691_;
}
v_reusejp_691_:
{
return v___x_692_;
}
}
}
}
else
{
lean_object* v___x_695_; lean_object* v___x_696_; lean_object* v___x_697_; lean_object* v___x_698_; lean_object* v___x_699_; 
lean_dec_ref(v___x_661_);
lean_dec_ref(v_commonFVarIds_596_);
v___x_695_ = lean_unsigned_to_nat(1u);
v___x_696_ = lean_nat_add(v_depth_595_, v___x_695_);
lean_dec(v_depth_595_);
v___x_697_ = ((lean_object*)(lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___lam__0___closed__0));
v___x_698_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_698_, 0, v___x_696_);
lean_ctor_set(v___x_698_, 1, v___x_697_);
lean_ctor_set(v___x_698_, 2, v___x_697_);
v___x_699_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_699_, 0, v___x_698_);
return v___x_699_;
}
}
v___jp_700_:
{
lean_object* v___x_701_; lean_object* v___x_702_; lean_object* v___x_703_; lean_object* v___x_704_; lean_object* v___x_705_; lean_object* v___x_706_; lean_object* v___x_707_; 
v___x_701_ = ((lean_object*)(lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___lam__0___closed__0));
v___x_702_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_702_, 0, v_goal_594_);
lean_ctor_set(v___x_702_, 1, v___x_701_);
v___x_703_ = lean_unsigned_to_nat(1u);
v___x_704_ = lean_mk_empty_array_with_capacity(v___x_703_);
v___x_705_ = lean_array_push(v___x_704_, v___x_702_);
v___x_706_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_706_, 0, v_depth_595_);
lean_ctor_set(v___x_706_, 1, v_commonFVarIds_596_);
lean_ctor_set(v___x_706_, 2, v___x_705_);
v___x_707_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_707_, 0, v___x_706_);
return v___x_707_;
}
v___jp_712_:
{
lean_object* v___x_717_; lean_object* v___x_718_; 
lean_inc(v_goal_594_);
v___x_717_ = lean_alloc_closure((void*)(l_Lean_Elab_Tactic_Ext_applyExtTheoremAt___boxed), 6, 1);
lean_closure_set(v___x_717_, 0, v_goal_594_);
v___x_718_ = lp_aesop_Lean_observing_x3f___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__0___redArg(v___x_717_, v___y_713_, v___y_714_, v___y_715_, v___y_716_);
if (lean_obj_tag(v___x_718_) == 0)
{
lean_object* v_a_719_; 
v_a_719_ = lean_ctor_get(v___x_718_, 0);
lean_inc(v_a_719_);
lean_dec_ref_known(v___x_718_, 1);
if (lean_obj_tag(v_a_719_) == 1)
{
lean_object* v_val_720_; lean_object* v___x_721_; 
lean_dec(v_goal_594_);
v_val_720_ = lean_ctor_get(v_a_719_, 0);
lean_inc(v_val_720_);
lean_dec_ref_known(v_a_719_, 1);
v___x_721_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__2___redArg(v___x_711_, v___y_715_);
if (lean_obj_tag(v___x_721_) == 0)
{
lean_object* v_a_722_; uint8_t v___x_723_; 
v_a_722_ = lean_ctor_get(v___x_721_, 0);
lean_inc(v_a_722_);
lean_dec_ref_known(v___x_721_, 1);
v___x_723_ = lean_unbox(v_a_722_);
lean_dec(v_a_722_);
if (v___x_723_ == 0)
{
v___y_603_ = v_val_720_;
v___y_604_ = v___y_713_;
v___y_605_ = v___y_714_;
v___y_606_ = v___y_715_;
v___y_607_ = v___y_716_;
goto v___jp_602_;
}
else
{
lean_object* v_traceClass_724_; lean_object* v___x_725_; lean_object* v___x_726_; lean_object* v___x_727_; lean_object* v___x_728_; lean_object* v___x_729_; 
v_traceClass_724_ = lean_ctor_get(v___x_711_, 0);
v___x_725_ = lean_obj_once(&lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___lam__0___closed__2, &lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___lam__0___closed__2_once, _init_lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___lam__0___closed__2);
lean_inc(v_val_720_);
v___x_726_ = l_Lean_Elab_goalsToMessageData(v_val_720_);
v___x_727_ = l_Lean_indentD(v___x_726_);
v___x_728_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_728_, 0, v___x_725_);
lean_ctor_set(v___x_728_, 1, v___x_727_);
lean_inc(v_traceClass_724_);
v___x_729_ = lp_aesop_Lean_addTrace___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__3(v_traceClass_724_, v___x_728_, v___y_713_, v___y_714_, v___y_715_, v___y_716_);
if (lean_obj_tag(v___x_729_) == 0)
{
lean_dec_ref_known(v___x_729_, 1);
v___y_603_ = v_val_720_;
v___y_604_ = v___y_713_;
v___y_605_ = v___y_714_;
v___y_606_ = v___y_715_;
v___y_607_ = v___y_716_;
goto v___jp_602_;
}
else
{
lean_object* v_a_730_; lean_object* v___x_732_; uint8_t v_isShared_733_; uint8_t v_isSharedCheck_737_; 
lean_dec(v_val_720_);
lean_dec_ref(v_commonFVarIds_596_);
lean_dec(v_depth_595_);
v_a_730_ = lean_ctor_get(v___x_729_, 0);
v_isSharedCheck_737_ = !lean_is_exclusive(v___x_729_);
if (v_isSharedCheck_737_ == 0)
{
v___x_732_ = v___x_729_;
v_isShared_733_ = v_isSharedCheck_737_;
goto v_resetjp_731_;
}
else
{
lean_inc(v_a_730_);
lean_dec(v___x_729_);
v___x_732_ = lean_box(0);
v_isShared_733_ = v_isSharedCheck_737_;
goto v_resetjp_731_;
}
v_resetjp_731_:
{
lean_object* v___x_735_; 
if (v_isShared_733_ == 0)
{
v___x_735_ = v___x_732_;
goto v_reusejp_734_;
}
else
{
lean_object* v_reuseFailAlloc_736_; 
v_reuseFailAlloc_736_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_736_, 0, v_a_730_);
v___x_735_ = v_reuseFailAlloc_736_;
goto v_reusejp_734_;
}
v_reusejp_734_:
{
return v___x_735_;
}
}
}
}
}
else
{
lean_object* v_a_738_; lean_object* v___x_740_; uint8_t v_isShared_741_; uint8_t v_isSharedCheck_745_; 
lean_dec(v_val_720_);
lean_dec_ref(v_commonFVarIds_596_);
lean_dec(v_depth_595_);
v_a_738_ = lean_ctor_get(v___x_721_, 0);
v_isSharedCheck_745_ = !lean_is_exclusive(v___x_721_);
if (v_isSharedCheck_745_ == 0)
{
v___x_740_ = v___x_721_;
v_isShared_741_ = v_isSharedCheck_745_;
goto v_resetjp_739_;
}
else
{
lean_inc(v_a_738_);
lean_dec(v___x_721_);
v___x_740_ = lean_box(0);
v_isShared_741_ = v_isSharedCheck_745_;
goto v_resetjp_739_;
}
v_resetjp_739_:
{
lean_object* v___x_743_; 
if (v_isShared_741_ == 0)
{
v___x_743_ = v___x_740_;
goto v_reusejp_742_;
}
else
{
lean_object* v_reuseFailAlloc_744_; 
v_reuseFailAlloc_744_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_744_, 0, v_a_738_);
v___x_743_ = v_reuseFailAlloc_744_;
goto v_reusejp_742_;
}
v_reusejp_742_:
{
return v___x_743_;
}
}
}
}
else
{
lean_object* v___x_746_; 
lean_dec(v_a_719_);
v___x_746_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__2___redArg(v___x_711_, v___y_715_);
if (lean_obj_tag(v___x_746_) == 0)
{
lean_object* v_a_747_; uint8_t v___x_748_; 
v_a_747_ = lean_ctor_get(v___x_746_, 0);
lean_inc(v_a_747_);
lean_dec_ref_known(v___x_746_, 1);
v___x_748_ = lean_unbox(v_a_747_);
lean_dec(v_a_747_);
if (v___x_748_ == 0)
{
goto v___jp_647_;
}
else
{
lean_object* v_traceClass_749_; lean_object* v___x_750_; lean_object* v___x_751_; 
v_traceClass_749_ = lean_ctor_get(v___x_711_, 0);
v___x_750_ = lean_obj_once(&lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___lam__0___closed__4, &lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___lam__0___closed__4_once, _init_lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___lam__0___closed__4);
lean_inc(v_traceClass_749_);
v___x_751_ = lp_aesop_Lean_addTrace___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__3(v_traceClass_749_, v___x_750_, v___y_713_, v___y_714_, v___y_715_, v___y_716_);
if (lean_obj_tag(v___x_751_) == 0)
{
lean_dec_ref_known(v___x_751_, 1);
goto v___jp_647_;
}
else
{
lean_object* v_a_752_; lean_object* v___x_754_; uint8_t v_isShared_755_; uint8_t v_isSharedCheck_759_; 
lean_dec_ref(v_commonFVarIds_596_);
lean_dec(v_depth_595_);
lean_dec(v_goal_594_);
v_a_752_ = lean_ctor_get(v___x_751_, 0);
v_isSharedCheck_759_ = !lean_is_exclusive(v___x_751_);
if (v_isSharedCheck_759_ == 0)
{
v___x_754_ = v___x_751_;
v_isShared_755_ = v_isSharedCheck_759_;
goto v_resetjp_753_;
}
else
{
lean_inc(v_a_752_);
lean_dec(v___x_751_);
v___x_754_ = lean_box(0);
v_isShared_755_ = v_isSharedCheck_759_;
goto v_resetjp_753_;
}
v_resetjp_753_:
{
lean_object* v___x_757_; 
if (v_isShared_755_ == 0)
{
v___x_757_ = v___x_754_;
goto v_reusejp_756_;
}
else
{
lean_object* v_reuseFailAlloc_758_; 
v_reuseFailAlloc_758_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_758_, 0, v_a_752_);
v___x_757_ = v_reuseFailAlloc_758_;
goto v_reusejp_756_;
}
v_reusejp_756_:
{
return v___x_757_;
}
}
}
}
}
else
{
lean_object* v_a_760_; lean_object* v___x_762_; uint8_t v_isShared_763_; uint8_t v_isSharedCheck_767_; 
lean_dec_ref(v_commonFVarIds_596_);
lean_dec(v_depth_595_);
lean_dec(v_goal_594_);
v_a_760_ = lean_ctor_get(v___x_746_, 0);
v_isSharedCheck_767_ = !lean_is_exclusive(v___x_746_);
if (v_isSharedCheck_767_ == 0)
{
v___x_762_ = v___x_746_;
v_isShared_763_ = v_isSharedCheck_767_;
goto v_resetjp_761_;
}
else
{
lean_inc(v_a_760_);
lean_dec(v___x_746_);
v___x_762_ = lean_box(0);
v_isShared_763_ = v_isSharedCheck_767_;
goto v_resetjp_761_;
}
v_resetjp_761_:
{
lean_object* v___x_765_; 
if (v_isShared_763_ == 0)
{
v___x_765_ = v___x_762_;
goto v_reusejp_764_;
}
else
{
lean_object* v_reuseFailAlloc_766_; 
v_reuseFailAlloc_766_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_766_, 0, v_a_760_);
v___x_765_ = v_reuseFailAlloc_766_;
goto v_reusejp_764_;
}
v_reusejp_764_:
{
return v___x_765_;
}
}
}
}
}
else
{
lean_object* v_a_768_; lean_object* v___x_770_; uint8_t v_isShared_771_; uint8_t v_isSharedCheck_775_; 
lean_dec_ref(v_commonFVarIds_596_);
lean_dec(v_depth_595_);
lean_dec(v_goal_594_);
v_a_768_ = lean_ctor_get(v___x_718_, 0);
v_isSharedCheck_775_ = !lean_is_exclusive(v___x_718_);
if (v_isSharedCheck_775_ == 0)
{
v___x_770_ = v___x_718_;
v_isShared_771_ = v_isSharedCheck_775_;
goto v_resetjp_769_;
}
else
{
lean_inc(v_a_768_);
lean_dec(v___x_718_);
v___x_770_ = lean_box(0);
v_isShared_771_ = v_isSharedCheck_775_;
goto v_resetjp_769_;
}
v_resetjp_769_:
{
lean_object* v___x_773_; 
if (v_isShared_771_ == 0)
{
v___x_773_ = v___x_770_;
goto v_reusejp_772_;
}
else
{
lean_object* v_reuseFailAlloc_774_; 
v_reuseFailAlloc_774_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_774_, 0, v_a_768_);
v___x_773_ = v_reuseFailAlloc_774_;
goto v_reusejp_772_;
}
v_reusejp_772_:
{
return v___x_773_;
}
}
}
}
v___jp_776_:
{
lean_object* v___x_781_; lean_object* v___x_782_; 
lean_inc(v_goal_594_);
v___x_781_ = lean_alloc_closure((void*)(l_Lean_Elab_Tactic_Ext_applyExtTheoremAt___boxed), 6, 1);
lean_closure_set(v___x_781_, 0, v_goal_594_);
v___x_782_ = lp_aesop_Lean_observing_x3f___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__0___redArg(v___x_781_, v___y_777_, v___y_778_, v___y_779_, v___y_780_);
if (lean_obj_tag(v___x_782_) == 0)
{
lean_object* v_a_783_; 
v_a_783_ = lean_ctor_get(v___x_782_, 0);
lean_inc(v_a_783_);
lean_dec_ref_known(v___x_782_, 1);
if (lean_obj_tag(v_a_783_) == 1)
{
lean_object* v_val_784_; lean_object* v___x_785_; 
lean_dec(v_goal_594_);
v_val_784_ = lean_ctor_get(v_a_783_, 0);
lean_inc(v_val_784_);
lean_dec_ref_known(v_a_783_, 1);
v___x_785_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__2___redArg(v___x_711_, v___y_779_);
if (lean_obj_tag(v___x_785_) == 0)
{
lean_object* v_a_786_; uint8_t v___x_787_; 
v_a_786_ = lean_ctor_get(v___x_785_, 0);
lean_inc(v_a_786_);
lean_dec_ref_known(v___x_785_, 1);
v___x_787_ = lean_unbox(v_a_786_);
lean_dec(v_a_786_);
if (v___x_787_ == 0)
{
v___y_656_ = v_val_784_;
v___y_657_ = v___y_777_;
v___y_658_ = v___y_778_;
v___y_659_ = v___y_779_;
v___y_660_ = v___y_780_;
goto v___jp_655_;
}
else
{
lean_object* v_traceClass_788_; lean_object* v___x_789_; lean_object* v___x_790_; lean_object* v___x_791_; lean_object* v___x_792_; lean_object* v___x_793_; 
v_traceClass_788_ = lean_ctor_get(v___x_711_, 0);
v___x_789_ = lean_obj_once(&lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___lam__0___closed__2, &lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___lam__0___closed__2_once, _init_lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___lam__0___closed__2);
lean_inc(v_val_784_);
v___x_790_ = l_Lean_Elab_goalsToMessageData(v_val_784_);
v___x_791_ = l_Lean_indentD(v___x_790_);
v___x_792_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_792_, 0, v___x_789_);
lean_ctor_set(v___x_792_, 1, v___x_791_);
lean_inc(v_traceClass_788_);
v___x_793_ = lp_aesop_Lean_addTrace___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__3(v_traceClass_788_, v___x_792_, v___y_777_, v___y_778_, v___y_779_, v___y_780_);
if (lean_obj_tag(v___x_793_) == 0)
{
lean_dec_ref_known(v___x_793_, 1);
v___y_656_ = v_val_784_;
v___y_657_ = v___y_777_;
v___y_658_ = v___y_778_;
v___y_659_ = v___y_779_;
v___y_660_ = v___y_780_;
goto v___jp_655_;
}
else
{
lean_object* v_a_794_; lean_object* v___x_796_; uint8_t v_isShared_797_; uint8_t v_isSharedCheck_801_; 
lean_dec(v_val_784_);
lean_dec_ref(v_commonFVarIds_596_);
lean_dec(v_depth_595_);
v_a_794_ = lean_ctor_get(v___x_793_, 0);
v_isSharedCheck_801_ = !lean_is_exclusive(v___x_793_);
if (v_isSharedCheck_801_ == 0)
{
v___x_796_ = v___x_793_;
v_isShared_797_ = v_isSharedCheck_801_;
goto v_resetjp_795_;
}
else
{
lean_inc(v_a_794_);
lean_dec(v___x_793_);
v___x_796_ = lean_box(0);
v_isShared_797_ = v_isSharedCheck_801_;
goto v_resetjp_795_;
}
v_resetjp_795_:
{
lean_object* v___x_799_; 
if (v_isShared_797_ == 0)
{
v___x_799_ = v___x_796_;
goto v_reusejp_798_;
}
else
{
lean_object* v_reuseFailAlloc_800_; 
v_reuseFailAlloc_800_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_800_, 0, v_a_794_);
v___x_799_ = v_reuseFailAlloc_800_;
goto v_reusejp_798_;
}
v_reusejp_798_:
{
return v___x_799_;
}
}
}
}
}
else
{
lean_object* v_a_802_; lean_object* v___x_804_; uint8_t v_isShared_805_; uint8_t v_isSharedCheck_809_; 
lean_dec(v_val_784_);
lean_dec_ref(v_commonFVarIds_596_);
lean_dec(v_depth_595_);
v_a_802_ = lean_ctor_get(v___x_785_, 0);
v_isSharedCheck_809_ = !lean_is_exclusive(v___x_785_);
if (v_isSharedCheck_809_ == 0)
{
v___x_804_ = v___x_785_;
v_isShared_805_ = v_isSharedCheck_809_;
goto v_resetjp_803_;
}
else
{
lean_inc(v_a_802_);
lean_dec(v___x_785_);
v___x_804_ = lean_box(0);
v_isShared_805_ = v_isSharedCheck_809_;
goto v_resetjp_803_;
}
v_resetjp_803_:
{
lean_object* v___x_807_; 
if (v_isShared_805_ == 0)
{
v___x_807_ = v___x_804_;
goto v_reusejp_806_;
}
else
{
lean_object* v_reuseFailAlloc_808_; 
v_reuseFailAlloc_808_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_808_, 0, v_a_802_);
v___x_807_ = v_reuseFailAlloc_808_;
goto v_reusejp_806_;
}
v_reusejp_806_:
{
return v___x_807_;
}
}
}
}
else
{
lean_object* v___x_810_; 
lean_dec(v_a_783_);
v___x_810_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__2___redArg(v___x_711_, v___y_779_);
if (lean_obj_tag(v___x_810_) == 0)
{
lean_object* v_a_811_; uint8_t v___x_812_; 
v_a_811_ = lean_ctor_get(v___x_810_, 0);
lean_inc(v_a_811_);
lean_dec_ref_known(v___x_810_, 1);
v___x_812_ = lean_unbox(v_a_811_);
lean_dec(v_a_811_);
if (v___x_812_ == 0)
{
goto v___jp_700_;
}
else
{
lean_object* v_traceClass_813_; lean_object* v___x_814_; lean_object* v___x_815_; 
v_traceClass_813_ = lean_ctor_get(v___x_711_, 0);
v___x_814_ = lean_obj_once(&lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___lam__0___closed__4, &lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___lam__0___closed__4_once, _init_lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___lam__0___closed__4);
lean_inc(v_traceClass_813_);
v___x_815_ = lp_aesop_Lean_addTrace___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__3(v_traceClass_813_, v___x_814_, v___y_777_, v___y_778_, v___y_779_, v___y_780_);
if (lean_obj_tag(v___x_815_) == 0)
{
lean_dec_ref_known(v___x_815_, 1);
goto v___jp_700_;
}
else
{
lean_object* v_a_816_; lean_object* v___x_818_; uint8_t v_isShared_819_; uint8_t v_isSharedCheck_823_; 
lean_dec_ref(v_commonFVarIds_596_);
lean_dec(v_depth_595_);
lean_dec(v_goal_594_);
v_a_816_ = lean_ctor_get(v___x_815_, 0);
v_isSharedCheck_823_ = !lean_is_exclusive(v___x_815_);
if (v_isSharedCheck_823_ == 0)
{
v___x_818_ = v___x_815_;
v_isShared_819_ = v_isSharedCheck_823_;
goto v_resetjp_817_;
}
else
{
lean_inc(v_a_816_);
lean_dec(v___x_815_);
v___x_818_ = lean_box(0);
v_isShared_819_ = v_isSharedCheck_823_;
goto v_resetjp_817_;
}
v_resetjp_817_:
{
lean_object* v___x_821_; 
if (v_isShared_819_ == 0)
{
v___x_821_ = v___x_818_;
goto v_reusejp_820_;
}
else
{
lean_object* v_reuseFailAlloc_822_; 
v_reuseFailAlloc_822_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_822_, 0, v_a_816_);
v___x_821_ = v_reuseFailAlloc_822_;
goto v_reusejp_820_;
}
v_reusejp_820_:
{
return v___x_821_;
}
}
}
}
}
else
{
lean_object* v_a_824_; lean_object* v___x_826_; uint8_t v_isShared_827_; uint8_t v_isSharedCheck_831_; 
lean_dec_ref(v_commonFVarIds_596_);
lean_dec(v_depth_595_);
lean_dec(v_goal_594_);
v_a_824_ = lean_ctor_get(v___x_810_, 0);
v_isSharedCheck_831_ = !lean_is_exclusive(v___x_810_);
if (v_isSharedCheck_831_ == 0)
{
v___x_826_ = v___x_810_;
v_isShared_827_ = v_isSharedCheck_831_;
goto v_resetjp_825_;
}
else
{
lean_inc(v_a_824_);
lean_dec(v___x_810_);
v___x_826_ = lean_box(0);
v_isShared_827_ = v_isSharedCheck_831_;
goto v_resetjp_825_;
}
v_resetjp_825_:
{
lean_object* v___x_829_; 
if (v_isShared_827_ == 0)
{
v___x_829_ = v___x_826_;
goto v_reusejp_828_;
}
else
{
lean_object* v_reuseFailAlloc_830_; 
v_reuseFailAlloc_830_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_830_, 0, v_a_824_);
v___x_829_ = v_reuseFailAlloc_830_;
goto v_reusejp_828_;
}
v_reusejp_828_:
{
return v___x_829_;
}
}
}
}
}
else
{
lean_object* v_a_832_; lean_object* v___x_834_; uint8_t v_isShared_835_; uint8_t v_isSharedCheck_839_; 
lean_dec_ref(v_commonFVarIds_596_);
lean_dec(v_depth_595_);
lean_dec(v_goal_594_);
v_a_832_ = lean_ctor_get(v___x_782_, 0);
v_isSharedCheck_839_ = !lean_is_exclusive(v___x_782_);
if (v_isSharedCheck_839_ == 0)
{
v___x_834_ = v___x_782_;
v_isShared_835_ = v_isSharedCheck_839_;
goto v_resetjp_833_;
}
else
{
lean_inc(v_a_832_);
lean_dec(v___x_782_);
v___x_834_ = lean_box(0);
v_isShared_835_ = v_isSharedCheck_839_;
goto v_resetjp_833_;
}
v_resetjp_833_:
{
lean_object* v___x_837_; 
if (v_isShared_835_ == 0)
{
v___x_837_ = v___x_834_;
goto v_reusejp_836_;
}
else
{
lean_object* v_reuseFailAlloc_838_; 
v_reuseFailAlloc_838_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_838_, 0, v_a_832_);
v___x_837_ = v_reuseFailAlloc_838_;
goto v_reusejp_836_;
}
v_reusejp_836_:
{
return v___x_837_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___lam__0(lean_object* v_goal_1015_, lean_object* v___x_1016_, lean_object* v_depth_1017_, lean_object* v_commonFVarIds_1018_, lean_object* v_____r_1019_, lean_object* v___y_1020_, lean_object* v___y_1021_, lean_object* v___y_1022_, lean_object* v___y_1023_){
_start:
{
lean_object* v___x_1033_; lean_object* v___x_1034_; 
lean_inc(v_goal_1015_);
v___x_1033_ = lean_alloc_closure((void*)(l_Lean_Elab_Tactic_Ext_applyExtTheoremAt___boxed), 6, 1);
lean_closure_set(v___x_1033_, 0, v_goal_1015_);
v___x_1034_ = lp_aesop_Lean_observing_x3f___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__0___redArg(v___x_1033_, v___y_1020_, v___y_1021_, v___y_1022_, v___y_1023_);
if (lean_obj_tag(v___x_1034_) == 0)
{
lean_object* v_a_1035_; lean_object* v___x_1037_; uint8_t v_isShared_1038_; uint8_t v_isSharedCheck_1140_; 
v_a_1035_ = lean_ctor_get(v___x_1034_, 0);
v_isSharedCheck_1140_ = !lean_is_exclusive(v___x_1034_);
if (v_isSharedCheck_1140_ == 0)
{
v___x_1037_ = v___x_1034_;
v_isShared_1038_ = v_isSharedCheck_1140_;
goto v_resetjp_1036_;
}
else
{
lean_inc(v_a_1035_);
lean_dec(v___x_1034_);
v___x_1037_ = lean_box(0);
v_isShared_1038_ = v_isSharedCheck_1140_;
goto v_resetjp_1036_;
}
v_resetjp_1036_:
{
if (lean_obj_tag(v_a_1035_) == 1)
{
lean_object* v_val_1039_; lean_object* v___y_1041_; lean_object* v___y_1042_; lean_object* v___y_1043_; lean_object* v___y_1044_; lean_object* v___x_1086_; 
lean_dec(v_goal_1015_);
v_val_1039_ = lean_ctor_get(v_a_1035_, 0);
lean_inc(v_val_1039_);
lean_dec_ref_known(v_a_1035_, 1);
v___x_1086_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__2___redArg(v___x_1016_, v___y_1022_);
if (lean_obj_tag(v___x_1086_) == 0)
{
lean_object* v_a_1087_; uint8_t v___x_1088_; 
v_a_1087_ = lean_ctor_get(v___x_1086_, 0);
lean_inc(v_a_1087_);
lean_dec_ref_known(v___x_1086_, 1);
v___x_1088_ = lean_unbox(v_a_1087_);
lean_dec(v_a_1087_);
if (v___x_1088_ == 0)
{
lean_dec_ref(v___x_1016_);
v___y_1041_ = v___y_1020_;
v___y_1042_ = v___y_1021_;
v___y_1043_ = v___y_1022_;
v___y_1044_ = v___y_1023_;
goto v___jp_1040_;
}
else
{
lean_object* v_traceClass_1089_; lean_object* v___x_1091_; uint8_t v_isShared_1092_; uint8_t v_isSharedCheck_1108_; 
v_traceClass_1089_ = lean_ctor_get(v___x_1016_, 0);
v_isSharedCheck_1108_ = !lean_is_exclusive(v___x_1016_);
if (v_isSharedCheck_1108_ == 0)
{
lean_object* v_unused_1109_; 
v_unused_1109_ = lean_ctor_get(v___x_1016_, 1);
lean_dec(v_unused_1109_);
v___x_1091_ = v___x_1016_;
v_isShared_1092_ = v_isSharedCheck_1108_;
goto v_resetjp_1090_;
}
else
{
lean_inc(v_traceClass_1089_);
lean_dec(v___x_1016_);
v___x_1091_ = lean_box(0);
v_isShared_1092_ = v_isSharedCheck_1108_;
goto v_resetjp_1090_;
}
v_resetjp_1090_:
{
lean_object* v___x_1093_; lean_object* v___x_1094_; lean_object* v___x_1095_; lean_object* v___x_1097_; 
v___x_1093_ = lean_obj_once(&lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___lam__0___closed__2, &lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___lam__0___closed__2_once, _init_lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___lam__0___closed__2);
lean_inc(v_val_1039_);
v___x_1094_ = l_Lean_Elab_goalsToMessageData(v_val_1039_);
v___x_1095_ = l_Lean_indentD(v___x_1094_);
if (v_isShared_1092_ == 0)
{
lean_ctor_set_tag(v___x_1091_, 7);
lean_ctor_set(v___x_1091_, 1, v___x_1095_);
lean_ctor_set(v___x_1091_, 0, v___x_1093_);
v___x_1097_ = v___x_1091_;
goto v_reusejp_1096_;
}
else
{
lean_object* v_reuseFailAlloc_1107_; 
v_reuseFailAlloc_1107_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1107_, 0, v___x_1093_);
lean_ctor_set(v_reuseFailAlloc_1107_, 1, v___x_1095_);
v___x_1097_ = v_reuseFailAlloc_1107_;
goto v_reusejp_1096_;
}
v_reusejp_1096_:
{
lean_object* v___x_1098_; 
v___x_1098_ = lp_aesop_Lean_addTrace___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__3(v_traceClass_1089_, v___x_1097_, v___y_1020_, v___y_1021_, v___y_1022_, v___y_1023_);
if (lean_obj_tag(v___x_1098_) == 0)
{
lean_dec_ref_known(v___x_1098_, 1);
v___y_1041_ = v___y_1020_;
v___y_1042_ = v___y_1021_;
v___y_1043_ = v___y_1022_;
v___y_1044_ = v___y_1023_;
goto v___jp_1040_;
}
else
{
lean_object* v_a_1099_; lean_object* v___x_1101_; uint8_t v_isShared_1102_; uint8_t v_isSharedCheck_1106_; 
lean_dec(v_val_1039_);
lean_del_object(v___x_1037_);
lean_dec_ref(v_commonFVarIds_1018_);
lean_dec(v_depth_1017_);
v_a_1099_ = lean_ctor_get(v___x_1098_, 0);
v_isSharedCheck_1106_ = !lean_is_exclusive(v___x_1098_);
if (v_isSharedCheck_1106_ == 0)
{
v___x_1101_ = v___x_1098_;
v_isShared_1102_ = v_isSharedCheck_1106_;
goto v_resetjp_1100_;
}
else
{
lean_inc(v_a_1099_);
lean_dec(v___x_1098_);
v___x_1101_ = lean_box(0);
v_isShared_1102_ = v_isSharedCheck_1106_;
goto v_resetjp_1100_;
}
v_resetjp_1100_:
{
lean_object* v___x_1104_; 
if (v_isShared_1102_ == 0)
{
v___x_1104_ = v___x_1101_;
goto v_reusejp_1103_;
}
else
{
lean_object* v_reuseFailAlloc_1105_; 
v_reuseFailAlloc_1105_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1105_, 0, v_a_1099_);
v___x_1104_ = v_reuseFailAlloc_1105_;
goto v_reusejp_1103_;
}
v_reusejp_1103_:
{
return v___x_1104_;
}
}
}
}
}
}
}
else
{
lean_object* v_a_1110_; lean_object* v___x_1112_; uint8_t v_isShared_1113_; uint8_t v_isSharedCheck_1117_; 
lean_dec(v_val_1039_);
lean_del_object(v___x_1037_);
lean_dec_ref(v_commonFVarIds_1018_);
lean_dec(v_depth_1017_);
lean_dec_ref(v___x_1016_);
v_a_1110_ = lean_ctor_get(v___x_1086_, 0);
v_isSharedCheck_1117_ = !lean_is_exclusive(v___x_1086_);
if (v_isSharedCheck_1117_ == 0)
{
v___x_1112_ = v___x_1086_;
v_isShared_1113_ = v_isSharedCheck_1117_;
goto v_resetjp_1111_;
}
else
{
lean_inc(v_a_1110_);
lean_dec(v___x_1086_);
v___x_1112_ = lean_box(0);
v_isShared_1113_ = v_isSharedCheck_1117_;
goto v_resetjp_1111_;
}
v_resetjp_1111_:
{
lean_object* v___x_1115_; 
if (v_isShared_1113_ == 0)
{
v___x_1115_ = v___x_1112_;
goto v_reusejp_1114_;
}
else
{
lean_object* v_reuseFailAlloc_1116_; 
v_reuseFailAlloc_1116_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1116_, 0, v_a_1110_);
v___x_1115_ = v_reuseFailAlloc_1116_;
goto v_reusejp_1114_;
}
v_reusejp_1114_:
{
return v___x_1115_;
}
}
}
v___jp_1040_:
{
lean_object* v___x_1045_; lean_object* v___x_1046_; lean_object* v___x_1047_; uint8_t v___x_1048_; 
v___x_1045_ = lean_array_mk(v_val_1039_);
v___x_1046_ = lean_array_get_size(v___x_1045_);
v___x_1047_ = lean_unsigned_to_nat(0u);
v___x_1048_ = lean_nat_dec_eq(v___x_1046_, v___x_1047_);
if (v___x_1048_ == 0)
{
size_t v_sz_1049_; size_t v___x_1050_; lean_object* v___x_1051_; 
lean_del_object(v___x_1037_);
v_sz_1049_ = lean_array_size(v___x_1045_);
v___x_1050_ = ((size_t)0ULL);
v___x_1051_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__1(v_sz_1049_, v___x_1050_, v___x_1045_, v___y_1041_, v___y_1042_, v___y_1043_, v___y_1044_);
if (lean_obj_tag(v___x_1051_) == 0)
{
lean_object* v_a_1052_; lean_object* v___x_1054_; uint8_t v_isShared_1055_; uint8_t v_isSharedCheck_1070_; 
v_a_1052_ = lean_ctor_get(v___x_1051_, 0);
v_isSharedCheck_1070_ = !lean_is_exclusive(v___x_1051_);
if (v_isSharedCheck_1070_ == 0)
{
v___x_1054_ = v___x_1051_;
v_isShared_1055_ = v_isSharedCheck_1070_;
goto v_resetjp_1053_;
}
else
{
lean_inc(v_a_1052_);
lean_dec(v___x_1051_);
v___x_1054_ = lean_box(0);
v_isShared_1055_ = v_isSharedCheck_1070_;
goto v_resetjp_1053_;
}
v_resetjp_1053_:
{
lean_object* v___x_1056_; lean_object* v___x_1057_; uint8_t v___x_1058_; 
v___x_1056_ = lean_array_get_size(v_a_1052_);
v___x_1057_ = lean_unsigned_to_nat(1u);
v___x_1058_ = lean_nat_dec_eq(v___x_1056_, v___x_1057_);
if (v___x_1058_ == 0)
{
lean_object* v___x_1059_; lean_object* v___x_1060_; lean_object* v___x_1062_; 
v___x_1059_ = lean_nat_add(v_depth_1017_, v___x_1057_);
lean_dec(v_depth_1017_);
v___x_1060_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1060_, 0, v___x_1059_);
lean_ctor_set(v___x_1060_, 1, v_commonFVarIds_1018_);
lean_ctor_set(v___x_1060_, 2, v_a_1052_);
if (v_isShared_1055_ == 0)
{
lean_ctor_set(v___x_1054_, 0, v___x_1060_);
v___x_1062_ = v___x_1054_;
goto v_reusejp_1061_;
}
else
{
lean_object* v_reuseFailAlloc_1063_; 
v_reuseFailAlloc_1063_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1063_, 0, v___x_1060_);
v___x_1062_ = v_reuseFailAlloc_1063_;
goto v_reusejp_1061_;
}
v_reusejp_1061_:
{
return v___x_1062_;
}
}
else
{
lean_object* v___x_1064_; lean_object* v_fst_1065_; lean_object* v_snd_1066_; lean_object* v___x_1067_; lean_object* v___x_1068_; lean_object* v___x_1069_; 
lean_del_object(v___x_1054_);
v___x_1064_ = lean_array_fget(v_a_1052_, v___x_1047_);
lean_dec(v_a_1052_);
v_fst_1065_ = lean_ctor_get(v___x_1064_, 0);
lean_inc(v_fst_1065_);
v_snd_1066_ = lean_ctor_get(v___x_1064_, 1);
lean_inc(v_snd_1066_);
lean_dec(v___x_1064_);
v___x_1067_ = lean_nat_add(v_depth_1017_, v___x_1057_);
lean_dec(v_depth_1017_);
v___x_1068_ = l_Array_append___redArg(v_commonFVarIds_1018_, v_snd_1066_);
lean_dec(v_snd_1066_);
v___x_1069_ = lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go(v_fst_1065_, v___x_1067_, v___x_1068_, v___y_1041_, v___y_1042_, v___y_1043_, v___y_1044_);
return v___x_1069_;
}
}
}
else
{
lean_object* v_a_1071_; lean_object* v___x_1073_; uint8_t v_isShared_1074_; uint8_t v_isSharedCheck_1078_; 
lean_dec_ref(v_commonFVarIds_1018_);
lean_dec(v_depth_1017_);
v_a_1071_ = lean_ctor_get(v___x_1051_, 0);
v_isSharedCheck_1078_ = !lean_is_exclusive(v___x_1051_);
if (v_isSharedCheck_1078_ == 0)
{
v___x_1073_ = v___x_1051_;
v_isShared_1074_ = v_isSharedCheck_1078_;
goto v_resetjp_1072_;
}
else
{
lean_inc(v_a_1071_);
lean_dec(v___x_1051_);
v___x_1073_ = lean_box(0);
v_isShared_1074_ = v_isSharedCheck_1078_;
goto v_resetjp_1072_;
}
v_resetjp_1072_:
{
lean_object* v___x_1076_; 
if (v_isShared_1074_ == 0)
{
v___x_1076_ = v___x_1073_;
goto v_reusejp_1075_;
}
else
{
lean_object* v_reuseFailAlloc_1077_; 
v_reuseFailAlloc_1077_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1077_, 0, v_a_1071_);
v___x_1076_ = v_reuseFailAlloc_1077_;
goto v_reusejp_1075_;
}
v_reusejp_1075_:
{
return v___x_1076_;
}
}
}
}
else
{
lean_object* v___x_1079_; lean_object* v___x_1080_; lean_object* v___x_1081_; lean_object* v___x_1082_; lean_object* v___x_1084_; 
lean_dec_ref(v___x_1045_);
lean_dec_ref(v_commonFVarIds_1018_);
v___x_1079_ = lean_unsigned_to_nat(1u);
v___x_1080_ = lean_nat_add(v_depth_1017_, v___x_1079_);
lean_dec(v_depth_1017_);
v___x_1081_ = ((lean_object*)(lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___lam__0___closed__0));
v___x_1082_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1082_, 0, v___x_1080_);
lean_ctor_set(v___x_1082_, 1, v___x_1081_);
lean_ctor_set(v___x_1082_, 2, v___x_1081_);
if (v_isShared_1038_ == 0)
{
lean_ctor_set(v___x_1037_, 0, v___x_1082_);
v___x_1084_ = v___x_1037_;
goto v_reusejp_1083_;
}
else
{
lean_object* v_reuseFailAlloc_1085_; 
v_reuseFailAlloc_1085_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1085_, 0, v___x_1082_);
v___x_1084_ = v_reuseFailAlloc_1085_;
goto v_reusejp_1083_;
}
v_reusejp_1083_:
{
return v___x_1084_;
}
}
}
}
else
{
lean_object* v___x_1118_; 
lean_del_object(v___x_1037_);
lean_dec(v_a_1035_);
v___x_1118_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__2___redArg(v___x_1016_, v___y_1022_);
if (lean_obj_tag(v___x_1118_) == 0)
{
lean_object* v_a_1119_; uint8_t v___x_1120_; 
v_a_1119_ = lean_ctor_get(v___x_1118_, 0);
lean_inc(v_a_1119_);
lean_dec_ref_known(v___x_1118_, 1);
v___x_1120_ = lean_unbox(v_a_1119_);
lean_dec(v_a_1119_);
if (v___x_1120_ == 0)
{
lean_dec_ref(v___x_1016_);
goto v___jp_1025_;
}
else
{
lean_object* v_traceClass_1121_; lean_object* v___x_1122_; lean_object* v___x_1123_; 
v_traceClass_1121_ = lean_ctor_get(v___x_1016_, 0);
lean_inc(v_traceClass_1121_);
lean_dec_ref(v___x_1016_);
v___x_1122_ = lean_obj_once(&lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___lam__0___closed__4, &lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___lam__0___closed__4_once, _init_lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___lam__0___closed__4);
v___x_1123_ = lp_aesop_Lean_addTrace___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__3(v_traceClass_1121_, v___x_1122_, v___y_1020_, v___y_1021_, v___y_1022_, v___y_1023_);
if (lean_obj_tag(v___x_1123_) == 0)
{
lean_dec_ref_known(v___x_1123_, 1);
goto v___jp_1025_;
}
else
{
lean_object* v_a_1124_; lean_object* v___x_1126_; uint8_t v_isShared_1127_; uint8_t v_isSharedCheck_1131_; 
lean_dec_ref(v_commonFVarIds_1018_);
lean_dec(v_depth_1017_);
lean_dec(v_goal_1015_);
v_a_1124_ = lean_ctor_get(v___x_1123_, 0);
v_isSharedCheck_1131_ = !lean_is_exclusive(v___x_1123_);
if (v_isSharedCheck_1131_ == 0)
{
v___x_1126_ = v___x_1123_;
v_isShared_1127_ = v_isSharedCheck_1131_;
goto v_resetjp_1125_;
}
else
{
lean_inc(v_a_1124_);
lean_dec(v___x_1123_);
v___x_1126_ = lean_box(0);
v_isShared_1127_ = v_isSharedCheck_1131_;
goto v_resetjp_1125_;
}
v_resetjp_1125_:
{
lean_object* v___x_1129_; 
if (v_isShared_1127_ == 0)
{
v___x_1129_ = v___x_1126_;
goto v_reusejp_1128_;
}
else
{
lean_object* v_reuseFailAlloc_1130_; 
v_reuseFailAlloc_1130_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1130_, 0, v_a_1124_);
v___x_1129_ = v_reuseFailAlloc_1130_;
goto v_reusejp_1128_;
}
v_reusejp_1128_:
{
return v___x_1129_;
}
}
}
}
}
else
{
lean_object* v_a_1132_; lean_object* v___x_1134_; uint8_t v_isShared_1135_; uint8_t v_isSharedCheck_1139_; 
lean_dec_ref(v_commonFVarIds_1018_);
lean_dec(v_depth_1017_);
lean_dec_ref(v___x_1016_);
lean_dec(v_goal_1015_);
v_a_1132_ = lean_ctor_get(v___x_1118_, 0);
v_isSharedCheck_1139_ = !lean_is_exclusive(v___x_1118_);
if (v_isSharedCheck_1139_ == 0)
{
v___x_1134_ = v___x_1118_;
v_isShared_1135_ = v_isSharedCheck_1139_;
goto v_resetjp_1133_;
}
else
{
lean_inc(v_a_1132_);
lean_dec(v___x_1118_);
v___x_1134_ = lean_box(0);
v_isShared_1135_ = v_isSharedCheck_1139_;
goto v_resetjp_1133_;
}
v_resetjp_1133_:
{
lean_object* v___x_1137_; 
if (v_isShared_1135_ == 0)
{
v___x_1137_ = v___x_1134_;
goto v_reusejp_1136_;
}
else
{
lean_object* v_reuseFailAlloc_1138_; 
v_reuseFailAlloc_1138_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1138_, 0, v_a_1132_);
v___x_1137_ = v_reuseFailAlloc_1138_;
goto v_reusejp_1136_;
}
v_reusejp_1136_:
{
return v___x_1137_;
}
}
}
}
}
}
else
{
lean_object* v_a_1141_; lean_object* v___x_1143_; uint8_t v_isShared_1144_; uint8_t v_isSharedCheck_1148_; 
lean_dec_ref(v_commonFVarIds_1018_);
lean_dec(v_depth_1017_);
lean_dec_ref(v___x_1016_);
lean_dec(v_goal_1015_);
v_a_1141_ = lean_ctor_get(v___x_1034_, 0);
v_isSharedCheck_1148_ = !lean_is_exclusive(v___x_1034_);
if (v_isSharedCheck_1148_ == 0)
{
v___x_1143_ = v___x_1034_;
v_isShared_1144_ = v_isSharedCheck_1148_;
goto v_resetjp_1142_;
}
else
{
lean_inc(v_a_1141_);
lean_dec(v___x_1034_);
v___x_1143_ = lean_box(0);
v_isShared_1144_ = v_isSharedCheck_1148_;
goto v_resetjp_1142_;
}
v_resetjp_1142_:
{
lean_object* v___x_1146_; 
if (v_isShared_1144_ == 0)
{
v___x_1146_ = v___x_1143_;
goto v_reusejp_1145_;
}
else
{
lean_object* v_reuseFailAlloc_1147_; 
v_reuseFailAlloc_1147_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1147_, 0, v_a_1141_);
v___x_1146_ = v_reuseFailAlloc_1147_;
goto v_reusejp_1145_;
}
v_reusejp_1145_:
{
return v___x_1146_;
}
}
}
v___jp_1025_:
{
lean_object* v___x_1026_; lean_object* v___x_1027_; lean_object* v___x_1028_; lean_object* v___x_1029_; lean_object* v___x_1030_; lean_object* v___x_1031_; lean_object* v___x_1032_; 
v___x_1026_ = ((lean_object*)(lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___lam__0___closed__0));
v___x_1027_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1027_, 0, v_goal_1015_);
lean_ctor_set(v___x_1027_, 1, v___x_1026_);
v___x_1028_ = lean_unsigned_to_nat(1u);
v___x_1029_ = lean_mk_empty_array_with_capacity(v___x_1028_);
v___x_1030_ = lean_array_push(v___x_1029_, v___x_1027_);
v___x_1031_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1031_, 0, v_depth_1017_);
lean_ctor_set(v___x_1031_, 1, v_commonFVarIds_1018_);
lean_ctor_set(v___x_1031_, 2, v___x_1030_);
v___x_1032_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1032_, 0, v___x_1031_);
return v___x_1032_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___lam__0___boxed(lean_object* v_goal_1149_, lean_object* v___x_1150_, lean_object* v_depth_1151_, lean_object* v_commonFVarIds_1152_, lean_object* v_____r_1153_, lean_object* v___y_1154_, lean_object* v___y_1155_, lean_object* v___y_1156_, lean_object* v___y_1157_, lean_object* v___y_1158_){
_start:
{
lean_object* v_res_1159_; 
v_res_1159_ = lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___lam__0(v_goal_1149_, v___x_1150_, v_depth_1151_, v_commonFVarIds_1152_, v_____r_1153_, v___y_1154_, v___y_1155_, v___y_1156_, v___y_1157_);
lean_dec(v___y_1157_);
lean_dec_ref(v___y_1156_);
lean_dec(v___y_1155_);
lean_dec_ref(v___y_1154_);
return v_res_1159_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___boxed(lean_object* v_goal_1160_, lean_object* v_depth_1161_, lean_object* v_commonFVarIds_1162_, lean_object* v_a_1163_, lean_object* v_a_1164_, lean_object* v_a_1165_, lean_object* v_a_1166_, lean_object* v_a_1167_){
_start:
{
lean_object* v_res_1168_; 
v_res_1168_ = lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go(v_goal_1160_, v_depth_1161_, v_commonFVarIds_1162_, v_a_1163_, v_a_1164_, v_a_1165_, v_a_1166_);
lean_dec(v_a_1166_);
lean_dec_ref(v_a_1165_);
lean_dec(v_a_1164_);
lean_dec_ref(v_a_1163_);
return v_res_1168_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__2(lean_object* v_opt_1169_, lean_object* v___y_1170_, lean_object* v___y_1171_, lean_object* v___y_1172_, lean_object* v___y_1173_){
_start:
{
lean_object* v___x_1175_; 
v___x_1175_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__2___redArg(v_opt_1169_, v___y_1172_);
return v___x_1175_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__2___boxed(lean_object* v_opt_1176_, lean_object* v___y_1177_, lean_object* v___y_1178_, lean_object* v___y_1179_, lean_object* v___y_1180_, lean_object* v___y_1181_){
_start:
{
lean_object* v_res_1182_; 
v_res_1182_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__2(v_opt_1176_, v___y_1177_, v___y_1178_, v___y_1179_, v___y_1180_);
lean_dec(v___y_1180_);
lean_dec_ref(v___y_1179_);
lean_dec(v___y_1178_);
lean_dec_ref(v___y_1177_);
lean_dec_ref(v_opt_1176_);
return v_res_1182_;
}
}
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__6_spec__8(lean_object* v_00_u03b1_1183_, lean_object* v_x_1184_, lean_object* v___y_1185_, lean_object* v___y_1186_, lean_object* v___y_1187_, lean_object* v___y_1188_){
_start:
{
lean_object* v___x_1190_; 
v___x_1190_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__6_spec__8___redArg(v_x_1184_);
return v___x_1190_;
}
}
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__6_spec__8___boxed(lean_object* v_00_u03b1_1191_, lean_object* v_x_1192_, lean_object* v___y_1193_, lean_object* v___y_1194_, lean_object* v___y_1195_, lean_object* v___y_1196_, lean_object* v___y_1197_){
_start:
{
lean_object* v_res_1198_; 
v_res_1198_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__6_spec__8(v_00_u03b1_1191_, v_x_1192_, v___y_1193_, v___y_1194_, v___y_1195_, v___y_1196_);
lean_dec(v___y_1196_);
lean_dec_ref(v___y_1195_);
lean_dec(v___y_1194_);
lean_dec_ref(v___y_1193_);
return v_res_1198_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_straightLineExt(lean_object* v_goal_1199_, lean_object* v_a_1200_, lean_object* v_a_1201_, lean_object* v_a_1202_, lean_object* v_a_1203_){
_start:
{
lean_object* v___x_1205_; lean_object* v___x_1206_; lean_object* v___x_1207_; 
v___x_1205_ = lean_unsigned_to_nat(0u);
v___x_1206_ = ((lean_object*)(lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go___lam__0___closed__0));
v___x_1207_ = lp_aesop___private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go(v_goal_1199_, v___x_1205_, v___x_1206_, v_a_1200_, v_a_1201_, v_a_1202_, v_a_1203_);
return v___x_1207_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_straightLineExt___boxed(lean_object* v_goal_1208_, lean_object* v_a_1209_, lean_object* v_a_1210_, lean_object* v_a_1211_, lean_object* v_a_1212_, lean_object* v_a_1213_){
_start:
{
lean_object* v_res_1214_; 
v_res_1214_ = lp_aesop_Aesop_straightLineExt(v_goal_1208_, v_a_1209_, v_a_1210_, v_a_1211_, v_a_1212_);
lean_dec(v_a_1212_);
lean_dec_ref(v_a_1211_);
lean_dec(v_a_1210_);
lean_dec_ref(v_a_1209_);
return v_res_1214_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_straightLineExtProgress_spec__0___redArg(lean_object* v_msg_1215_, lean_object* v___y_1216_, lean_object* v___y_1217_, lean_object* v___y_1218_, lean_object* v___y_1219_){
_start:
{
lean_object* v_ref_1221_; lean_object* v___x_1222_; lean_object* v_a_1223_; lean_object* v___x_1225_; uint8_t v_isShared_1226_; uint8_t v_isSharedCheck_1231_; 
v_ref_1221_ = lean_ctor_get(v___y_1218_, 5);
v___x_1222_ = lp_aesop_Lean_addMessageContextFull___at___00Lean_addTrace___at___00__private_Aesop_Util_Tactic_Ext_0__Aesop_straightLineExt_go_spec__3_spec__3(v_msg_1215_, v___y_1216_, v___y_1217_, v___y_1218_, v___y_1219_);
v_a_1223_ = lean_ctor_get(v___x_1222_, 0);
v_isSharedCheck_1231_ = !lean_is_exclusive(v___x_1222_);
if (v_isSharedCheck_1231_ == 0)
{
v___x_1225_ = v___x_1222_;
v_isShared_1226_ = v_isSharedCheck_1231_;
goto v_resetjp_1224_;
}
else
{
lean_inc(v_a_1223_);
lean_dec(v___x_1222_);
v___x_1225_ = lean_box(0);
v_isShared_1226_ = v_isSharedCheck_1231_;
goto v_resetjp_1224_;
}
v_resetjp_1224_:
{
lean_object* v___x_1227_; lean_object* v___x_1229_; 
lean_inc(v_ref_1221_);
v___x_1227_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1227_, 0, v_ref_1221_);
lean_ctor_set(v___x_1227_, 1, v_a_1223_);
if (v_isShared_1226_ == 0)
{
lean_ctor_set_tag(v___x_1225_, 1);
lean_ctor_set(v___x_1225_, 0, v___x_1227_);
v___x_1229_ = v___x_1225_;
goto v_reusejp_1228_;
}
else
{
lean_object* v_reuseFailAlloc_1230_; 
v_reuseFailAlloc_1230_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1230_, 0, v___x_1227_);
v___x_1229_ = v_reuseFailAlloc_1230_;
goto v_reusejp_1228_;
}
v_reusejp_1228_:
{
return v___x_1229_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_straightLineExtProgress_spec__0___redArg___boxed(lean_object* v_msg_1232_, lean_object* v___y_1233_, lean_object* v___y_1234_, lean_object* v___y_1235_, lean_object* v___y_1236_, lean_object* v___y_1237_){
_start:
{
lean_object* v_res_1238_; 
v_res_1238_ = lp_aesop_Lean_throwError___at___00Aesop_straightLineExtProgress_spec__0___redArg(v_msg_1232_, v___y_1233_, v___y_1234_, v___y_1235_, v___y_1236_);
lean_dec(v___y_1236_);
lean_dec_ref(v___y_1235_);
lean_dec(v___y_1234_);
lean_dec_ref(v___y_1233_);
return v_res_1238_;
}
}
static lean_object* _init_lp_aesop_Aesop_straightLineExtProgress___closed__1(void){
_start:
{
lean_object* v___x_1240_; lean_object* v___x_1241_; 
v___x_1240_ = ((lean_object*)(lp_aesop_Aesop_straightLineExtProgress___closed__0));
v___x_1241_ = l_Lean_stringToMessageData(v___x_1240_);
return v___x_1241_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_straightLineExtProgress(lean_object* v_goal_1242_, lean_object* v_a_1243_, lean_object* v_a_1244_, lean_object* v_a_1245_, lean_object* v_a_1246_){
_start:
{
lean_object* v___x_1248_; 
v___x_1248_ = lp_aesop_Aesop_straightLineExt(v_goal_1242_, v_a_1243_, v_a_1244_, v_a_1245_, v_a_1246_);
if (lean_obj_tag(v___x_1248_) == 0)
{
lean_object* v_a_1249_; lean_object* v_depth_1250_; lean_object* v___x_1251_; uint8_t v___x_1252_; 
v_a_1249_ = lean_ctor_get(v___x_1248_, 0);
lean_inc(v_a_1249_);
v_depth_1250_ = lean_ctor_get(v_a_1249_, 0);
lean_inc(v_depth_1250_);
lean_dec(v_a_1249_);
v___x_1251_ = lean_unsigned_to_nat(0u);
v___x_1252_ = lean_nat_dec_eq(v_depth_1250_, v___x_1251_);
lean_dec(v_depth_1250_);
if (v___x_1252_ == 0)
{
return v___x_1248_;
}
else
{
lean_object* v___x_1253_; lean_object* v___x_1254_; lean_object* v_a_1255_; lean_object* v___x_1257_; uint8_t v_isShared_1258_; uint8_t v_isSharedCheck_1262_; 
lean_dec_ref_known(v___x_1248_, 1);
v___x_1253_ = lean_obj_once(&lp_aesop_Aesop_straightLineExtProgress___closed__1, &lp_aesop_Aesop_straightLineExtProgress___closed__1_once, _init_lp_aesop_Aesop_straightLineExtProgress___closed__1);
v___x_1254_ = lp_aesop_Lean_throwError___at___00Aesop_straightLineExtProgress_spec__0___redArg(v___x_1253_, v_a_1243_, v_a_1244_, v_a_1245_, v_a_1246_);
v_a_1255_ = lean_ctor_get(v___x_1254_, 0);
v_isSharedCheck_1262_ = !lean_is_exclusive(v___x_1254_);
if (v_isSharedCheck_1262_ == 0)
{
v___x_1257_ = v___x_1254_;
v_isShared_1258_ = v_isSharedCheck_1262_;
goto v_resetjp_1256_;
}
else
{
lean_inc(v_a_1255_);
lean_dec(v___x_1254_);
v___x_1257_ = lean_box(0);
v_isShared_1258_ = v_isSharedCheck_1262_;
goto v_resetjp_1256_;
}
v_resetjp_1256_:
{
lean_object* v___x_1260_; 
if (v_isShared_1258_ == 0)
{
v___x_1260_ = v___x_1257_;
goto v_reusejp_1259_;
}
else
{
lean_object* v_reuseFailAlloc_1261_; 
v_reuseFailAlloc_1261_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1261_, 0, v_a_1255_);
v___x_1260_ = v_reuseFailAlloc_1261_;
goto v_reusejp_1259_;
}
v_reusejp_1259_:
{
return v___x_1260_;
}
}
}
}
else
{
return v___x_1248_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_straightLineExtProgress___boxed(lean_object* v_goal_1263_, lean_object* v_a_1264_, lean_object* v_a_1265_, lean_object* v_a_1266_, lean_object* v_a_1267_, lean_object* v_a_1268_){
_start:
{
lean_object* v_res_1269_; 
v_res_1269_ = lp_aesop_Aesop_straightLineExtProgress(v_goal_1263_, v_a_1264_, v_a_1265_, v_a_1266_, v_a_1267_);
lean_dec(v_a_1267_);
lean_dec_ref(v_a_1266_);
lean_dec(v_a_1265_);
lean_dec_ref(v_a_1264_);
return v_res_1269_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_straightLineExtProgress_spec__0(lean_object* v_00_u03b1_1270_, lean_object* v_msg_1271_, lean_object* v___y_1272_, lean_object* v___y_1273_, lean_object* v___y_1274_, lean_object* v___y_1275_){
_start:
{
lean_object* v___x_1277_; 
v___x_1277_ = lp_aesop_Lean_throwError___at___00Aesop_straightLineExtProgress_spec__0___redArg(v_msg_1271_, v___y_1272_, v___y_1273_, v___y_1274_, v___y_1275_);
return v___x_1277_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_straightLineExtProgress_spec__0___boxed(lean_object* v_00_u03b1_1278_, lean_object* v_msg_1279_, lean_object* v___y_1280_, lean_object* v___y_1281_, lean_object* v___y_1282_, lean_object* v___y_1283_, lean_object* v___y_1284_){
_start:
{
lean_object* v_res_1285_; 
v_res_1285_ = lp_aesop_Lean_throwError___at___00Aesop_straightLineExtProgress_spec__0(v_00_u03b1_1278_, v_msg_1279_, v___y_1280_, v___y_1281_, v___y_1282_, v___y_1283_);
lean_dec(v___y_1283_);
lean_dec_ref(v___y_1282_);
lean_dec(v___y_1281_);
lean_dec_ref(v___y_1280_);
return v_res_1285_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Tracing(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_Tactic_Ext(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_Tactic_Intro(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_Util_Tactic_Ext(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Tracing(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Tactic_Ext(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_Tactic_Intro(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_Util_Tactic_Ext(uint8_t builtin) {
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
lean_object* initialize_aesop_Aesop_Tracing(uint8_t builtin);
lean_object* initialize_Lean_Elab_Tactic_Ext(uint8_t builtin);
lean_object* initialize_Lean_Meta_Tactic_Intro(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_Util_Tactic_Ext(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Tracing(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_Tactic_Ext(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_Tactic_Intro(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Util_Tactic_Ext(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_Util_Tactic_Ext(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_Util_Tactic_Ext(builtin);
}
#ifdef __cplusplus
}
#endif
