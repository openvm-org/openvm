// Lean compiler output
// Module: Aesop.Main
// Imports: public import Init public meta import Init public meta import Aesop.Search.Main public meta import Aesop.Frontend.Tactic public meta import Aesop.Stats.Extension public meta import Aesop.Stats.File
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
lean_object* l_Lean_profileitIOUnsafe___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_getDeclName_x3f___redArg(lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_PrettyPrinter_ppCategory(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Std_Format_pretty(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getPos_x3f(lean_object*, uint8_t);
lean_object* l_Lean_FileMap_toPosition(lean_object*, lean_object*);
double lean_float_of_nat(lean_object*);
extern lean_object* lp_aesop_Aesop_aesop_collectStats;
lean_object* lean_st_ref_take(lean_object*);
extern lean_object* lp_aesop_Aesop_statsExtension;
lean_object* l_Lean_PersistentEnvExtension_addEntry___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_getMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* lp_aesop_Aesop_Stats_empty;
lean_object* lean_st_mk_ref(lean_object*);
lean_object* lean_io_mono_nanos_now();
lean_object* lean_st_ref_get(lean_object*);
lean_object* lp_aesop_Aesop_search(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* l_Lean_Elab_Tactic_replaceMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
extern lean_object* lp_aesop_Aesop_TraceOption_stats;
extern lean_object* lp_aesop_Aesop_aesop_stats_file;
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
lean_object* lean_io_get_num_heartbeats();
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_PersistentArray_toArray___redArg(lean_object*);
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* l_Lean_PersistentArray_push___redArg(lean_object*, lean_object*);
extern lean_object* l_Lean_trace_profiler;
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_PersistentArray_append___redArg(lean_object*, lean_object*);
double lean_float_sub(double, double);
uint8_t lean_float_decLt(double, double);
extern lean_object* l_Lean_trace_profiler_useHeartbeats;
extern lean_object* l_Lean_trace_profiler_threshold;
double lean_float_div(double, double);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lp_aesop_Aesop_LocalRuleSet_trace(lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* lp_aesop_Aesop_TraceOption_ruleSet;
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* l_Lean_Name_append(lean_object*, lean_object*);
uint8_t l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_Frontend_TacticConfig_getRuleSet(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_Frontend_TacticConfig_parse(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_Stats_trace(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_io_error_to_string(lean_object*);
lean_object* lean_io_prim_handle_mk(lean_object*, uint8_t);
lean_object* lean_io_prim_handle_lock(lean_object*, uint8_t);
lean_object* lp_aesop_Aesop_instToJsonStatsFileRecord_toJson(lean_object*);
lean_object* l_Lean_Json_compress(lean_object*);
lean_object* l_IO_FS_Handle_putStrLn(lean_object*, lean_object*);
lean_object* lean_io_prim_handle_unlock(lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_Option_get___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__2___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__3___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__3___redArg___closed__0;
static lean_once_cell_t lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__3___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__3___redArg___closed__1;
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__3___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__3___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Main_0__Aesop_evalAesop_go___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Main_0__Aesop_evalAesop_go___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Main_0__Aesop_evalAesop_go___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Main_0__Aesop_evalAesop_go___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__4_spec__4_spec__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__4_spec__4_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__4_spec__4_spec__5(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__4_spec__4_spec__5___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__4_spec__4___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__4_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__4_spec__5___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__4_spec__5___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__4_spec__7(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__4_spec__7___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__4_spec__6(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__4_spec__6___boxed(lean_object*);
static lean_once_cell_t lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__4___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__4___closed__0;
static const lean_string_object lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__4___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 54, .m_capacity = 54, .m_length = 53, .m_data = "<exception thrown while producing trace node message>"};
static const lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__4___closed__1 = (const lean_object*)&lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__4___closed__1_value;
static lean_once_cell_t lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__4___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__4___closed__2;
static lean_once_cell_t lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__4___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__4___closed__3;
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__4(lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__4___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop___private_Aesop_Main_0__Aesop_evalAesop_go___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_aesop___private_Aesop_Main_0__Aesop_evalAesop_go___closed__0 = (const lean_object*)&lp_aesop___private_Aesop_Main_0__Aesop_evalAesop_go___closed__0_value;
static lean_once_cell_t lp_aesop___private_Aesop_Main_0__Aesop_evalAesop_go___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_aesop___private_Aesop_Main_0__Aesop_evalAesop_go___closed__1;
static const lean_string_object lp_aesop___private_Aesop_Main_0__Aesop_evalAesop_go___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Rule set"};
static const lean_object* lp_aesop___private_Aesop_Main_0__Aesop_evalAesop_go___closed__2 = (const lean_object*)&lp_aesop___private_Aesop_Main_0__Aesop_evalAesop_go___closed__2_value;
static const lean_ctor_object lp_aesop___private_Aesop_Main_0__Aesop_evalAesop_go___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Main_0__Aesop_evalAesop_go___closed__2_value)}};
static const lean_object* lp_aesop___private_Aesop_Main_0__Aesop_evalAesop_go___closed__3 = (const lean_object*)&lp_aesop___private_Aesop_Main_0__Aesop_evalAesop_go___closed__3_value;
static lean_once_cell_t lp_aesop___private_Aesop_Main_0__Aesop_evalAesop_go___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Main_0__Aesop_evalAesop_go___closed__4;
static lean_once_cell_t lp_aesop___private_Aesop_Main_0__Aesop_evalAesop_go___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Main_0__Aesop_evalAesop_go___closed__5;
static const lean_string_object lp_aesop___private_Aesop_Main_0__Aesop_evalAesop_go___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_aesop___private_Aesop_Main_0__Aesop_evalAesop_go___closed__6 = (const lean_object*)&lp_aesop___private_Aesop_Main_0__Aesop_evalAesop_go___closed__6_value;
static const lean_ctor_object lp_aesop___private_Aesop_Main_0__Aesop_evalAesop_go___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Main_0__Aesop_evalAesop_go___closed__6_value),LEAN_SCALAR_PTR_LITERAL(212, 145, 141, 177, 67, 149, 127, 197)}};
static const lean_object* lp_aesop___private_Aesop_Main_0__Aesop_evalAesop_go___closed__7 = (const lean_object*)&lp_aesop___private_Aesop_Main_0__Aesop_evalAesop_go___closed__7_value;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Main_0__Aesop_evalAesop_go(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Main_0__Aesop_evalAesop_go___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__4_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__4_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__4_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__4_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_evalAesop_spec__2___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_evalAesop_spec__2___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_evalAesop_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_evalAesop_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_evalAesop_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_evalAesop_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_profileitM___at___00Aesop_evalAesop_spec__3___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_profileitM___at___00Aesop_evalAesop_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_profileitM___at___00Aesop_evalAesop_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_profileitM___at___00Aesop_evalAesop_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_StatsExtensionEntry_forCurrentFile___at___00Aesop_recordStatsForCurrentFileIfEnabled___at___00Aesop_evalAesop_spec__0_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_StatsExtensionEntry_forCurrentFile___at___00Aesop_recordStatsForCurrentFileIfEnabled___at___00Aesop_evalAesop_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop_Aesop_recordStatsForCurrentFileIfEnabled___at___00Aesop_evalAesop_spec__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_recordStatsForCurrentFileIfEnabled___at___00Aesop_evalAesop_spec__0___closed__0;
static lean_once_cell_t lp_aesop_Aesop_recordStatsForCurrentFileIfEnabled___at___00Aesop_evalAesop_spec__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_recordStatsForCurrentFileIfEnabled___at___00Aesop_evalAesop_spec__0___closed__1;
static lean_once_cell_t lp_aesop_Aesop_recordStatsForCurrentFileIfEnabled___at___00Aesop_evalAesop_spec__0___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_recordStatsForCurrentFileIfEnabled___at___00Aesop_evalAesop_spec__0___closed__2;
static lean_once_cell_t lp_aesop_Aesop_recordStatsForCurrentFileIfEnabled___at___00Aesop_evalAesop_spec__0___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_recordStatsForCurrentFileIfEnabled___at___00Aesop_evalAesop_spec__0___closed__3;
LEAN_EXPORT lean_object* lp_aesop_Aesop_recordStatsForCurrentFileIfEnabled___at___00Aesop_evalAesop_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_recordStatsForCurrentFileIfEnabled___at___00Aesop_evalAesop_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_StatsFileRecord_ofStats___at___00Aesop_appendStatsToStatsFileIfEnabled___at___00Aesop_evalAesop_spec__1_spec__2___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "tactic"};
static const lean_object* lp_aesop_Aesop_StatsFileRecord_ofStats___at___00Aesop_appendStatsToStatsFileIfEnabled___at___00Aesop_evalAesop_spec__1_spec__2___redArg___closed__0 = (const lean_object*)&lp_aesop_Aesop_StatsFileRecord_ofStats___at___00Aesop_appendStatsToStatsFileIfEnabled___at___00Aesop_evalAesop_spec__1_spec__2___redArg___closed__0_value;
static const lean_ctor_object lp_aesop_Aesop_StatsFileRecord_ofStats___at___00Aesop_appendStatsToStatsFileIfEnabled___at___00Aesop_evalAesop_spec__1_spec__2___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_StatsFileRecord_ofStats___at___00Aesop_appendStatsToStatsFileIfEnabled___at___00Aesop_evalAesop_spec__1_spec__2___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(99, 76, 33, 121, 85, 143, 17, 224)}};
static const lean_object* lp_aesop_Aesop_StatsFileRecord_ofStats___at___00Aesop_appendStatsToStatsFileIfEnabled___at___00Aesop_evalAesop_spec__1_spec__2___redArg___closed__1 = (const lean_object*)&lp_aesop_Aesop_StatsFileRecord_ofStats___at___00Aesop_appendStatsToStatsFileIfEnabled___at___00Aesop_evalAesop_spec__1_spec__2___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_StatsFileRecord_ofStats___at___00Aesop_appendStatsToStatsFileIfEnabled___at___00Aesop_evalAesop_spec__1_spec__2___redArg(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_StatsFileRecord_ofStats___at___00Aesop_appendStatsToStatsFileIfEnabled___at___00Aesop_evalAesop_spec__1_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_appendStatsToStatsFileIfEnabled___at___00Aesop_evalAesop_spec__1(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_appendStatsToStatsFileIfEnabled___at___00Aesop_evalAesop_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_evalAesop___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_evalAesop___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_evalAesop___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_evalAesop___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_evalAesop___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "aesop"};
static const lean_object* lp_aesop_Aesop_evalAesop___closed__0 = (const lean_object*)&lp_aesop_Aesop_evalAesop___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_evalAesop(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_evalAesop___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_StatsExtensionEntry_forCurrentFile___at___00Aesop_recordStatsForCurrentFileIfEnabled___at___00Aesop_evalAesop_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_StatsExtensionEntry_forCurrentFile___at___00Aesop_recordStatsForCurrentFileIfEnabled___at___00Aesop_evalAesop_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_StatsFileRecord_ofStats___at___00Aesop_appendStatsToStatsFileIfEnabled___at___00Aesop_evalAesop_spec__1_spec__2(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_StatsFileRecord_ofStats___at___00Aesop_appendStatsToStatsFileIfEnabled___at___00Aesop_evalAesop_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_Option_get___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__0(lean_object* v_opts_1_, lean_object* v_opt_2_){
_start:
{
lean_object* v_name_3_; lean_object* v_defValue_4_; lean_object* v_map_5_; lean_object* v___x_6_; 
v_name_3_ = lean_ctor_get(v_opt_2_, 0);
v_defValue_4_ = lean_ctor_get(v_opt_2_, 1);
v_map_5_ = lean_ctor_get(v_opts_1_, 0);
v___x_6_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_5_, v_name_3_);
if (lean_obj_tag(v___x_6_) == 0)
{
uint8_t v___x_7_; 
v___x_7_ = lean_unbox(v_defValue_4_);
return v___x_7_;
}
else
{
lean_object* v_val_8_; 
v_val_8_ = lean_ctor_get(v___x_6_, 0);
lean_inc(v_val_8_);
lean_dec_ref_known(v___x_6_, 1);
if (lean_obj_tag(v_val_8_) == 1)
{
uint8_t v_v_9_; 
v_v_9_ = lean_ctor_get_uint8(v_val_8_, 0);
lean_dec_ref_known(v_val_8_, 0);
return v_v_9_;
}
else
{
uint8_t v___x_10_; 
lean_dec(v_val_8_);
v___x_10_ = lean_unbox(v_defValue_4_);
return v___x_10_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__0___boxed(lean_object* v_opts_11_, lean_object* v_opt_12_){
_start:
{
uint8_t v_res_13_; lean_object* v_r_14_; 
v_res_13_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__0(v_opts_11_, v_opt_12_);
lean_dec_ref(v_opt_12_);
lean_dec_ref(v_opts_11_);
v_r_14_ = lean_box(v_res_13_);
return v_r_14_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__2(lean_object* v_opts_15_, lean_object* v_opt_16_){
_start:
{
lean_object* v_name_17_; lean_object* v_defValue_18_; lean_object* v_map_19_; lean_object* v___x_20_; 
v_name_17_ = lean_ctor_get(v_opt_16_, 0);
v_defValue_18_ = lean_ctor_get(v_opt_16_, 1);
v_map_19_ = lean_ctor_get(v_opts_15_, 0);
v___x_20_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_19_, v_name_17_);
if (lean_obj_tag(v___x_20_) == 0)
{
lean_inc(v_defValue_18_);
return v_defValue_18_;
}
else
{
lean_object* v_val_21_; 
v_val_21_ = lean_ctor_get(v___x_20_, 0);
lean_inc(v_val_21_);
lean_dec_ref_known(v___x_20_, 1);
if (lean_obj_tag(v_val_21_) == 0)
{
lean_object* v_v_22_; 
v_v_22_ = lean_ctor_get(v_val_21_, 0);
lean_inc_ref(v_v_22_);
lean_dec_ref_known(v_val_21_, 1);
return v_v_22_;
}
else
{
lean_dec(v_val_21_);
lean_inc(v_defValue_18_);
return v_defValue_18_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__2___boxed(lean_object* v_opts_23_, lean_object* v_opt_24_){
_start:
{
lean_object* v_res_25_; 
v_res_25_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__2(v_opts_23_, v_opt_24_);
lean_dec_ref(v_opt_24_);
lean_dec_ref(v_opts_23_);
return v_res_25_;
}
}
static lean_object* _init_lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__3___redArg___closed__0(void){
_start:
{
lean_object* v___x_26_; lean_object* v___x_27_; lean_object* v___x_28_; 
v___x_26_ = lean_unsigned_to_nat(32u);
v___x_27_ = lean_mk_empty_array_with_capacity(v___x_26_);
v___x_28_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_28_, 0, v___x_27_);
return v___x_28_;
}
}
static lean_object* _init_lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__3___redArg___closed__1(void){
_start:
{
size_t v___x_29_; lean_object* v___x_30_; lean_object* v___x_31_; lean_object* v___x_32_; lean_object* v___x_33_; lean_object* v___x_34_; 
v___x_29_ = ((size_t)5ULL);
v___x_30_ = lean_unsigned_to_nat(0u);
v___x_31_ = lean_unsigned_to_nat(32u);
v___x_32_ = lean_mk_empty_array_with_capacity(v___x_31_);
v___x_33_ = lean_obj_once(&lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__3___redArg___closed__0, &lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__3___redArg___closed__0_once, _init_lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__3___redArg___closed__0);
v___x_34_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_34_, 0, v___x_33_);
lean_ctor_set(v___x_34_, 1, v___x_32_);
lean_ctor_set(v___x_34_, 2, v___x_30_);
lean_ctor_set(v___x_34_, 3, v___x_30_);
lean_ctor_set_usize(v___x_34_, 4, v___x_29_);
return v___x_34_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__3___redArg(lean_object* v___y_35_){
_start:
{
lean_object* v___x_37_; lean_object* v_traceState_38_; lean_object* v_traces_39_; lean_object* v___x_40_; lean_object* v_traceState_41_; lean_object* v_env_42_; lean_object* v_nextMacroScope_43_; lean_object* v_ngen_44_; lean_object* v_auxDeclNGen_45_; lean_object* v_cache_46_; lean_object* v_messages_47_; lean_object* v_infoState_48_; lean_object* v_snapshotTasks_49_; lean_object* v___x_51_; uint8_t v_isShared_52_; uint8_t v_isSharedCheck_68_; 
v___x_37_ = lean_st_ref_get(v___y_35_);
v_traceState_38_ = lean_ctor_get(v___x_37_, 4);
lean_inc_ref(v_traceState_38_);
lean_dec(v___x_37_);
v_traces_39_ = lean_ctor_get(v_traceState_38_, 0);
lean_inc_ref(v_traces_39_);
lean_dec_ref(v_traceState_38_);
v___x_40_ = lean_st_ref_take(v___y_35_);
v_traceState_41_ = lean_ctor_get(v___x_40_, 4);
v_env_42_ = lean_ctor_get(v___x_40_, 0);
v_nextMacroScope_43_ = lean_ctor_get(v___x_40_, 1);
v_ngen_44_ = lean_ctor_get(v___x_40_, 2);
v_auxDeclNGen_45_ = lean_ctor_get(v___x_40_, 3);
v_cache_46_ = lean_ctor_get(v___x_40_, 5);
v_messages_47_ = lean_ctor_get(v___x_40_, 6);
v_infoState_48_ = lean_ctor_get(v___x_40_, 7);
v_snapshotTasks_49_ = lean_ctor_get(v___x_40_, 8);
v_isSharedCheck_68_ = !lean_is_exclusive(v___x_40_);
if (v_isSharedCheck_68_ == 0)
{
v___x_51_ = v___x_40_;
v_isShared_52_ = v_isSharedCheck_68_;
goto v_resetjp_50_;
}
else
{
lean_inc(v_snapshotTasks_49_);
lean_inc(v_infoState_48_);
lean_inc(v_messages_47_);
lean_inc(v_cache_46_);
lean_inc(v_traceState_41_);
lean_inc(v_auxDeclNGen_45_);
lean_inc(v_ngen_44_);
lean_inc(v_nextMacroScope_43_);
lean_inc(v_env_42_);
lean_dec(v___x_40_);
v___x_51_ = lean_box(0);
v_isShared_52_ = v_isSharedCheck_68_;
goto v_resetjp_50_;
}
v_resetjp_50_:
{
uint64_t v_tid_53_; lean_object* v___x_55_; uint8_t v_isShared_56_; uint8_t v_isSharedCheck_66_; 
v_tid_53_ = lean_ctor_get_uint64(v_traceState_41_, sizeof(void*)*1);
v_isSharedCheck_66_ = !lean_is_exclusive(v_traceState_41_);
if (v_isSharedCheck_66_ == 0)
{
lean_object* v_unused_67_; 
v_unused_67_ = lean_ctor_get(v_traceState_41_, 0);
lean_dec(v_unused_67_);
v___x_55_ = v_traceState_41_;
v_isShared_56_ = v_isSharedCheck_66_;
goto v_resetjp_54_;
}
else
{
lean_dec(v_traceState_41_);
v___x_55_ = lean_box(0);
v_isShared_56_ = v_isSharedCheck_66_;
goto v_resetjp_54_;
}
v_resetjp_54_:
{
lean_object* v___x_57_; lean_object* v___x_59_; 
v___x_57_ = lean_obj_once(&lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__3___redArg___closed__1, &lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__3___redArg___closed__1_once, _init_lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__3___redArg___closed__1);
if (v_isShared_56_ == 0)
{
lean_ctor_set(v___x_55_, 0, v___x_57_);
v___x_59_ = v___x_55_;
goto v_reusejp_58_;
}
else
{
lean_object* v_reuseFailAlloc_65_; 
v_reuseFailAlloc_65_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_65_, 0, v___x_57_);
lean_ctor_set_uint64(v_reuseFailAlloc_65_, sizeof(void*)*1, v_tid_53_);
v___x_59_ = v_reuseFailAlloc_65_;
goto v_reusejp_58_;
}
v_reusejp_58_:
{
lean_object* v___x_61_; 
if (v_isShared_52_ == 0)
{
lean_ctor_set(v___x_51_, 4, v___x_59_);
v___x_61_ = v___x_51_;
goto v_reusejp_60_;
}
else
{
lean_object* v_reuseFailAlloc_64_; 
v_reuseFailAlloc_64_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_64_, 0, v_env_42_);
lean_ctor_set(v_reuseFailAlloc_64_, 1, v_nextMacroScope_43_);
lean_ctor_set(v_reuseFailAlloc_64_, 2, v_ngen_44_);
lean_ctor_set(v_reuseFailAlloc_64_, 3, v_auxDeclNGen_45_);
lean_ctor_set(v_reuseFailAlloc_64_, 4, v___x_59_);
lean_ctor_set(v_reuseFailAlloc_64_, 5, v_cache_46_);
lean_ctor_set(v_reuseFailAlloc_64_, 6, v_messages_47_);
lean_ctor_set(v_reuseFailAlloc_64_, 7, v_infoState_48_);
lean_ctor_set(v_reuseFailAlloc_64_, 8, v_snapshotTasks_49_);
v___x_61_ = v_reuseFailAlloc_64_;
goto v_reusejp_60_;
}
v_reusejp_60_:
{
lean_object* v___x_62_; lean_object* v___x_63_; 
v___x_62_ = lean_st_ref_set(v___y_35_, v___x_61_);
v___x_63_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_63_, 0, v_traces_39_);
return v___x_63_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__3___redArg___boxed(lean_object* v___y_69_, lean_object* v___y_70_){
_start:
{
lean_object* v_res_71_; 
v_res_71_ = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__3___redArg(v___y_69_);
lean_dec(v___y_69_);
return v_res_71_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__3(lean_object* v___y_72_, lean_object* v___y_73_, lean_object* v___y_74_, lean_object* v___y_75_, lean_object* v___y_76_, lean_object* v___y_77_, lean_object* v___y_78_, lean_object* v___y_79_, lean_object* v___y_80_){
_start:
{
lean_object* v___x_82_; 
v___x_82_ = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__3___redArg(v___y_80_);
return v___x_82_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__3___boxed(lean_object* v___y_83_, lean_object* v___y_84_, lean_object* v___y_85_, lean_object* v___y_86_, lean_object* v___y_87_, lean_object* v___y_88_, lean_object* v___y_89_, lean_object* v___y_90_, lean_object* v___y_91_, lean_object* v___y_92_){
_start:
{
lean_object* v_res_93_; 
v_res_93_ = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__3(v___y_83_, v___y_84_, v___y_85_, v___y_86_, v___y_87_, v___y_88_, v___y_89_, v___y_90_, v___y_91_);
lean_dec(v___y_91_);
lean_dec_ref(v___y_90_);
lean_dec(v___y_89_);
lean_dec_ref(v___y_88_);
lean_dec(v___y_87_);
lean_dec_ref(v___y_86_);
lean_dec(v___y_85_);
lean_dec_ref(v___y_84_);
lean_dec(v___y_83_);
return v_res_93_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Main_0__Aesop_evalAesop_go___lam__0(lean_object* v_s_94_, lean_object* v_x_95_, lean_object* v_t_96_){
_start:
{
lean_object* v_total_97_; lean_object* v_ruleSetConstruction_98_; lean_object* v_search_99_; lean_object* v_ruleSelection_100_; lean_object* v_script_101_; lean_object* v_forwardState_102_; lean_object* v_scriptGenerated_103_; lean_object* v_ruleStats_104_; lean_object* v_goalStats_105_; lean_object* v___x_107_; uint8_t v_isShared_108_; uint8_t v_isSharedCheck_112_; 
v_total_97_ = lean_ctor_get(v_s_94_, 0);
v_ruleSetConstruction_98_ = lean_ctor_get(v_s_94_, 2);
v_search_99_ = lean_ctor_get(v_s_94_, 3);
v_ruleSelection_100_ = lean_ctor_get(v_s_94_, 4);
v_script_101_ = lean_ctor_get(v_s_94_, 5);
v_forwardState_102_ = lean_ctor_get(v_s_94_, 6);
v_scriptGenerated_103_ = lean_ctor_get(v_s_94_, 7);
v_ruleStats_104_ = lean_ctor_get(v_s_94_, 8);
v_goalStats_105_ = lean_ctor_get(v_s_94_, 9);
v_isSharedCheck_112_ = !lean_is_exclusive(v_s_94_);
if (v_isSharedCheck_112_ == 0)
{
lean_object* v_unused_113_; 
v_unused_113_ = lean_ctor_get(v_s_94_, 1);
lean_dec(v_unused_113_);
v___x_107_ = v_s_94_;
v_isShared_108_ = v_isSharedCheck_112_;
goto v_resetjp_106_;
}
else
{
lean_inc(v_goalStats_105_);
lean_inc(v_ruleStats_104_);
lean_inc(v_scriptGenerated_103_);
lean_inc(v_forwardState_102_);
lean_inc(v_script_101_);
lean_inc(v_ruleSelection_100_);
lean_inc(v_search_99_);
lean_inc(v_ruleSetConstruction_98_);
lean_inc(v_total_97_);
lean_dec(v_s_94_);
v___x_107_ = lean_box(0);
v_isShared_108_ = v_isSharedCheck_112_;
goto v_resetjp_106_;
}
v_resetjp_106_:
{
lean_object* v___x_110_; 
if (v_isShared_108_ == 0)
{
lean_ctor_set(v___x_107_, 1, v_t_96_);
v___x_110_ = v___x_107_;
goto v_reusejp_109_;
}
else
{
lean_object* v_reuseFailAlloc_111_; 
v_reuseFailAlloc_111_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_111_, 0, v_total_97_);
lean_ctor_set(v_reuseFailAlloc_111_, 1, v_t_96_);
lean_ctor_set(v_reuseFailAlloc_111_, 2, v_ruleSetConstruction_98_);
lean_ctor_set(v_reuseFailAlloc_111_, 3, v_search_99_);
lean_ctor_set(v_reuseFailAlloc_111_, 4, v_ruleSelection_100_);
lean_ctor_set(v_reuseFailAlloc_111_, 5, v_script_101_);
lean_ctor_set(v_reuseFailAlloc_111_, 6, v_forwardState_102_);
lean_ctor_set(v_reuseFailAlloc_111_, 7, v_scriptGenerated_103_);
lean_ctor_set(v_reuseFailAlloc_111_, 8, v_ruleStats_104_);
lean_ctor_set(v_reuseFailAlloc_111_, 9, v_goalStats_105_);
v___x_110_ = v_reuseFailAlloc_111_;
goto v_reusejp_109_;
}
v_reusejp_109_:
{
return v___x_110_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Main_0__Aesop_evalAesop_go___lam__0___boxed(lean_object* v_s_114_, lean_object* v_x_115_, lean_object* v_t_116_){
_start:
{
lean_object* v_res_117_; 
v_res_117_ = lp_aesop___private_Aesop_Main_0__Aesop_evalAesop_go___lam__0(v_s_114_, v_x_115_, v_t_116_);
lean_dec_ref(v_x_115_);
return v_res_117_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Main_0__Aesop_evalAesop_go___lam__1(lean_object* v___x_118_, lean_object* v_x_119_, lean_object* v___y_120_, lean_object* v___y_121_, lean_object* v___y_122_, lean_object* v___y_123_, lean_object* v___y_124_, lean_object* v___y_125_, lean_object* v___y_126_, lean_object* v___y_127_, lean_object* v___y_128_){
_start:
{
lean_object* v___x_130_; 
v___x_130_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_130_, 0, v___x_118_);
return v___x_130_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Main_0__Aesop_evalAesop_go___lam__1___boxed(lean_object* v___x_131_, lean_object* v_x_132_, lean_object* v___y_133_, lean_object* v___y_134_, lean_object* v___y_135_, lean_object* v___y_136_, lean_object* v___y_137_, lean_object* v___y_138_, lean_object* v___y_139_, lean_object* v___y_140_, lean_object* v___y_141_, lean_object* v___y_142_){
_start:
{
lean_object* v_res_143_; 
v_res_143_ = lp_aesop___private_Aesop_Main_0__Aesop_evalAesop_go___lam__1(v___x_131_, v_x_132_, v___y_133_, v___y_134_, v___y_135_, v___y_136_, v___y_137_, v___y_138_, v___y_139_, v___y_140_, v___y_141_);
lean_dec(v___y_141_);
lean_dec_ref(v___y_140_);
lean_dec(v___y_139_);
lean_dec_ref(v___y_138_);
lean_dec(v___y_137_);
lean_dec_ref(v___y_136_);
lean_dec(v___y_135_);
lean_dec_ref(v___y_134_);
lean_dec(v___y_133_);
lean_dec_ref(v_x_132_);
return v_res_143_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__4_spec__4_spec__6(lean_object* v_msgData_144_, lean_object* v___y_145_, lean_object* v___y_146_, lean_object* v___y_147_, lean_object* v___y_148_){
_start:
{
lean_object* v___x_150_; lean_object* v_env_151_; lean_object* v___x_152_; lean_object* v_mctx_153_; lean_object* v_lctx_154_; lean_object* v_options_155_; lean_object* v___x_156_; lean_object* v___x_157_; lean_object* v___x_158_; 
v___x_150_ = lean_st_ref_get(v___y_148_);
v_env_151_ = lean_ctor_get(v___x_150_, 0);
lean_inc_ref(v_env_151_);
lean_dec(v___x_150_);
v___x_152_ = lean_st_ref_get(v___y_146_);
v_mctx_153_ = lean_ctor_get(v___x_152_, 0);
lean_inc_ref(v_mctx_153_);
lean_dec(v___x_152_);
v_lctx_154_ = lean_ctor_get(v___y_145_, 2);
v_options_155_ = lean_ctor_get(v___y_147_, 2);
lean_inc_ref(v_options_155_);
lean_inc_ref(v_lctx_154_);
v___x_156_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_156_, 0, v_env_151_);
lean_ctor_set(v___x_156_, 1, v_mctx_153_);
lean_ctor_set(v___x_156_, 2, v_lctx_154_);
lean_ctor_set(v___x_156_, 3, v_options_155_);
v___x_157_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_157_, 0, v___x_156_);
lean_ctor_set(v___x_157_, 1, v_msgData_144_);
v___x_158_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_158_, 0, v___x_157_);
return v___x_158_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__4_spec__4_spec__6___boxed(lean_object* v_msgData_159_, lean_object* v___y_160_, lean_object* v___y_161_, lean_object* v___y_162_, lean_object* v___y_163_, lean_object* v___y_164_){
_start:
{
lean_object* v_res_165_; 
v_res_165_ = lp_aesop_Lean_addMessageContextFull___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__4_spec__4_spec__6(v_msgData_159_, v___y_160_, v___y_161_, v___y_162_, v___y_163_);
lean_dec(v___y_163_);
lean_dec_ref(v___y_162_);
lean_dec(v___y_161_);
lean_dec_ref(v___y_160_);
return v_res_165_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__4_spec__4_spec__5(size_t v_sz_166_, size_t v_i_167_, lean_object* v_bs_168_){
_start:
{
uint8_t v___x_169_; 
v___x_169_ = lean_usize_dec_lt(v_i_167_, v_sz_166_);
if (v___x_169_ == 0)
{
return v_bs_168_;
}
else
{
lean_object* v_v_170_; lean_object* v_msg_171_; lean_object* v___x_172_; lean_object* v_bs_x27_173_; size_t v___x_174_; size_t v___x_175_; lean_object* v___x_176_; 
v_v_170_ = lean_array_uget_borrowed(v_bs_168_, v_i_167_);
v_msg_171_ = lean_ctor_get(v_v_170_, 1);
lean_inc_ref(v_msg_171_);
v___x_172_ = lean_unsigned_to_nat(0u);
v_bs_x27_173_ = lean_array_uset(v_bs_168_, v_i_167_, v___x_172_);
v___x_174_ = ((size_t)1ULL);
v___x_175_ = lean_usize_add(v_i_167_, v___x_174_);
v___x_176_ = lean_array_uset(v_bs_x27_173_, v_i_167_, v_msg_171_);
v_i_167_ = v___x_175_;
v_bs_168_ = v___x_176_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__4_spec__4_spec__5___boxed(lean_object* v_sz_178_, lean_object* v_i_179_, lean_object* v_bs_180_){
_start:
{
size_t v_sz_boxed_181_; size_t v_i_boxed_182_; lean_object* v_res_183_; 
v_sz_boxed_181_ = lean_unbox_usize(v_sz_178_);
lean_dec(v_sz_178_);
v_i_boxed_182_ = lean_unbox_usize(v_i_179_);
lean_dec(v_i_179_);
v_res_183_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__4_spec__4_spec__5(v_sz_boxed_181_, v_i_boxed_182_, v_bs_180_);
return v_res_183_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__4_spec__4___redArg(lean_object* v_oldTraces_184_, lean_object* v_data_185_, lean_object* v_ref_186_, lean_object* v_msg_187_, lean_object* v___y_188_, lean_object* v___y_189_, lean_object* v___y_190_, lean_object* v___y_191_){
_start:
{
lean_object* v_fileName_193_; lean_object* v_fileMap_194_; lean_object* v_options_195_; lean_object* v_currRecDepth_196_; lean_object* v_maxRecDepth_197_; lean_object* v_ref_198_; lean_object* v_currNamespace_199_; lean_object* v_openDecls_200_; lean_object* v_initHeartbeats_201_; lean_object* v_maxHeartbeats_202_; lean_object* v_quotContext_203_; lean_object* v_currMacroScope_204_; uint8_t v_diag_205_; lean_object* v_cancelTk_x3f_206_; uint8_t v_suppressElabErrors_207_; lean_object* v_inheritedTraceOptions_208_; lean_object* v___x_209_; lean_object* v_traceState_210_; lean_object* v_traces_211_; lean_object* v_ref_212_; lean_object* v___x_213_; lean_object* v___x_214_; size_t v_sz_215_; size_t v___x_216_; lean_object* v___x_217_; lean_object* v_msg_218_; lean_object* v___x_219_; lean_object* v_a_220_; lean_object* v___x_222_; uint8_t v_isShared_223_; uint8_t v_isSharedCheck_257_; 
v_fileName_193_ = lean_ctor_get(v___y_190_, 0);
v_fileMap_194_ = lean_ctor_get(v___y_190_, 1);
v_options_195_ = lean_ctor_get(v___y_190_, 2);
v_currRecDepth_196_ = lean_ctor_get(v___y_190_, 3);
v_maxRecDepth_197_ = lean_ctor_get(v___y_190_, 4);
v_ref_198_ = lean_ctor_get(v___y_190_, 5);
v_currNamespace_199_ = lean_ctor_get(v___y_190_, 6);
v_openDecls_200_ = lean_ctor_get(v___y_190_, 7);
v_initHeartbeats_201_ = lean_ctor_get(v___y_190_, 8);
v_maxHeartbeats_202_ = lean_ctor_get(v___y_190_, 9);
v_quotContext_203_ = lean_ctor_get(v___y_190_, 10);
v_currMacroScope_204_ = lean_ctor_get(v___y_190_, 11);
v_diag_205_ = lean_ctor_get_uint8(v___y_190_, sizeof(void*)*14);
v_cancelTk_x3f_206_ = lean_ctor_get(v___y_190_, 12);
v_suppressElabErrors_207_ = lean_ctor_get_uint8(v___y_190_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_208_ = lean_ctor_get(v___y_190_, 13);
v___x_209_ = lean_st_ref_get(v___y_191_);
v_traceState_210_ = lean_ctor_get(v___x_209_, 4);
lean_inc_ref(v_traceState_210_);
lean_dec(v___x_209_);
v_traces_211_ = lean_ctor_get(v_traceState_210_, 0);
lean_inc_ref(v_traces_211_);
lean_dec_ref(v_traceState_210_);
v_ref_212_ = l_Lean_replaceRef(v_ref_186_, v_ref_198_);
lean_inc_ref(v_inheritedTraceOptions_208_);
lean_inc(v_cancelTk_x3f_206_);
lean_inc(v_currMacroScope_204_);
lean_inc(v_quotContext_203_);
lean_inc(v_maxHeartbeats_202_);
lean_inc(v_initHeartbeats_201_);
lean_inc(v_openDecls_200_);
lean_inc(v_currNamespace_199_);
lean_inc(v_maxRecDepth_197_);
lean_inc(v_currRecDepth_196_);
lean_inc_ref(v_options_195_);
lean_inc_ref(v_fileMap_194_);
lean_inc_ref(v_fileName_193_);
v___x_213_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_213_, 0, v_fileName_193_);
lean_ctor_set(v___x_213_, 1, v_fileMap_194_);
lean_ctor_set(v___x_213_, 2, v_options_195_);
lean_ctor_set(v___x_213_, 3, v_currRecDepth_196_);
lean_ctor_set(v___x_213_, 4, v_maxRecDepth_197_);
lean_ctor_set(v___x_213_, 5, v_ref_212_);
lean_ctor_set(v___x_213_, 6, v_currNamespace_199_);
lean_ctor_set(v___x_213_, 7, v_openDecls_200_);
lean_ctor_set(v___x_213_, 8, v_initHeartbeats_201_);
lean_ctor_set(v___x_213_, 9, v_maxHeartbeats_202_);
lean_ctor_set(v___x_213_, 10, v_quotContext_203_);
lean_ctor_set(v___x_213_, 11, v_currMacroScope_204_);
lean_ctor_set(v___x_213_, 12, v_cancelTk_x3f_206_);
lean_ctor_set(v___x_213_, 13, v_inheritedTraceOptions_208_);
lean_ctor_set_uint8(v___x_213_, sizeof(void*)*14, v_diag_205_);
lean_ctor_set_uint8(v___x_213_, sizeof(void*)*14 + 1, v_suppressElabErrors_207_);
v___x_214_ = l_Lean_PersistentArray_toArray___redArg(v_traces_211_);
lean_dec_ref(v_traces_211_);
v_sz_215_ = lean_array_size(v___x_214_);
v___x_216_ = ((size_t)0ULL);
v___x_217_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__4_spec__4_spec__5(v_sz_215_, v___x_216_, v___x_214_);
v_msg_218_ = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(v_msg_218_, 0, v_data_185_);
lean_ctor_set(v_msg_218_, 1, v_msg_187_);
lean_ctor_set(v_msg_218_, 2, v___x_217_);
v___x_219_ = lp_aesop_Lean_addMessageContextFull___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__4_spec__4_spec__6(v_msg_218_, v___y_188_, v___y_189_, v___x_213_, v___y_191_);
lean_dec_ref_known(v___x_213_, 14);
v_a_220_ = lean_ctor_get(v___x_219_, 0);
v_isSharedCheck_257_ = !lean_is_exclusive(v___x_219_);
if (v_isSharedCheck_257_ == 0)
{
v___x_222_ = v___x_219_;
v_isShared_223_ = v_isSharedCheck_257_;
goto v_resetjp_221_;
}
else
{
lean_inc(v_a_220_);
lean_dec(v___x_219_);
v___x_222_ = lean_box(0);
v_isShared_223_ = v_isSharedCheck_257_;
goto v_resetjp_221_;
}
v_resetjp_221_:
{
lean_object* v___x_224_; lean_object* v_traceState_225_; lean_object* v_env_226_; lean_object* v_nextMacroScope_227_; lean_object* v_ngen_228_; lean_object* v_auxDeclNGen_229_; lean_object* v_cache_230_; lean_object* v_messages_231_; lean_object* v_infoState_232_; lean_object* v_snapshotTasks_233_; lean_object* v___x_235_; uint8_t v_isShared_236_; uint8_t v_isSharedCheck_256_; 
v___x_224_ = lean_st_ref_take(v___y_191_);
v_traceState_225_ = lean_ctor_get(v___x_224_, 4);
v_env_226_ = lean_ctor_get(v___x_224_, 0);
v_nextMacroScope_227_ = lean_ctor_get(v___x_224_, 1);
v_ngen_228_ = lean_ctor_get(v___x_224_, 2);
v_auxDeclNGen_229_ = lean_ctor_get(v___x_224_, 3);
v_cache_230_ = lean_ctor_get(v___x_224_, 5);
v_messages_231_ = lean_ctor_get(v___x_224_, 6);
v_infoState_232_ = lean_ctor_get(v___x_224_, 7);
v_snapshotTasks_233_ = lean_ctor_get(v___x_224_, 8);
v_isSharedCheck_256_ = !lean_is_exclusive(v___x_224_);
if (v_isSharedCheck_256_ == 0)
{
v___x_235_ = v___x_224_;
v_isShared_236_ = v_isSharedCheck_256_;
goto v_resetjp_234_;
}
else
{
lean_inc(v_snapshotTasks_233_);
lean_inc(v_infoState_232_);
lean_inc(v_messages_231_);
lean_inc(v_cache_230_);
lean_inc(v_traceState_225_);
lean_inc(v_auxDeclNGen_229_);
lean_inc(v_ngen_228_);
lean_inc(v_nextMacroScope_227_);
lean_inc(v_env_226_);
lean_dec(v___x_224_);
v___x_235_ = lean_box(0);
v_isShared_236_ = v_isSharedCheck_256_;
goto v_resetjp_234_;
}
v_resetjp_234_:
{
uint64_t v_tid_237_; lean_object* v___x_239_; uint8_t v_isShared_240_; uint8_t v_isSharedCheck_254_; 
v_tid_237_ = lean_ctor_get_uint64(v_traceState_225_, sizeof(void*)*1);
v_isSharedCheck_254_ = !lean_is_exclusive(v_traceState_225_);
if (v_isSharedCheck_254_ == 0)
{
lean_object* v_unused_255_; 
v_unused_255_ = lean_ctor_get(v_traceState_225_, 0);
lean_dec(v_unused_255_);
v___x_239_ = v_traceState_225_;
v_isShared_240_ = v_isSharedCheck_254_;
goto v_resetjp_238_;
}
else
{
lean_dec(v_traceState_225_);
v___x_239_ = lean_box(0);
v_isShared_240_ = v_isSharedCheck_254_;
goto v_resetjp_238_;
}
v_resetjp_238_:
{
lean_object* v___x_241_; lean_object* v___x_242_; lean_object* v___x_244_; 
v___x_241_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_241_, 0, v_ref_186_);
lean_ctor_set(v___x_241_, 1, v_a_220_);
v___x_242_ = l_Lean_PersistentArray_push___redArg(v_oldTraces_184_, v___x_241_);
if (v_isShared_240_ == 0)
{
lean_ctor_set(v___x_239_, 0, v___x_242_);
v___x_244_ = v___x_239_;
goto v_reusejp_243_;
}
else
{
lean_object* v_reuseFailAlloc_253_; 
v_reuseFailAlloc_253_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_253_, 0, v___x_242_);
lean_ctor_set_uint64(v_reuseFailAlloc_253_, sizeof(void*)*1, v_tid_237_);
v___x_244_ = v_reuseFailAlloc_253_;
goto v_reusejp_243_;
}
v_reusejp_243_:
{
lean_object* v___x_246_; 
if (v_isShared_236_ == 0)
{
lean_ctor_set(v___x_235_, 4, v___x_244_);
v___x_246_ = v___x_235_;
goto v_reusejp_245_;
}
else
{
lean_object* v_reuseFailAlloc_252_; 
v_reuseFailAlloc_252_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_252_, 0, v_env_226_);
lean_ctor_set(v_reuseFailAlloc_252_, 1, v_nextMacroScope_227_);
lean_ctor_set(v_reuseFailAlloc_252_, 2, v_ngen_228_);
lean_ctor_set(v_reuseFailAlloc_252_, 3, v_auxDeclNGen_229_);
lean_ctor_set(v_reuseFailAlloc_252_, 4, v___x_244_);
lean_ctor_set(v_reuseFailAlloc_252_, 5, v_cache_230_);
lean_ctor_set(v_reuseFailAlloc_252_, 6, v_messages_231_);
lean_ctor_set(v_reuseFailAlloc_252_, 7, v_infoState_232_);
lean_ctor_set(v_reuseFailAlloc_252_, 8, v_snapshotTasks_233_);
v___x_246_ = v_reuseFailAlloc_252_;
goto v_reusejp_245_;
}
v_reusejp_245_:
{
lean_object* v___x_247_; lean_object* v___x_248_; lean_object* v___x_250_; 
v___x_247_ = lean_st_ref_set(v___y_191_, v___x_246_);
v___x_248_ = lean_box(0);
if (v_isShared_223_ == 0)
{
lean_ctor_set(v___x_222_, 0, v___x_248_);
v___x_250_ = v___x_222_;
goto v_reusejp_249_;
}
else
{
lean_object* v_reuseFailAlloc_251_; 
v_reuseFailAlloc_251_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_251_, 0, v___x_248_);
v___x_250_ = v_reuseFailAlloc_251_;
goto v_reusejp_249_;
}
v_reusejp_249_:
{
return v___x_250_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__4_spec__4___redArg___boxed(lean_object* v_oldTraces_258_, lean_object* v_data_259_, lean_object* v_ref_260_, lean_object* v_msg_261_, lean_object* v___y_262_, lean_object* v___y_263_, lean_object* v___y_264_, lean_object* v___y_265_, lean_object* v___y_266_){
_start:
{
lean_object* v_res_267_; 
v_res_267_ = lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__4_spec__4___redArg(v_oldTraces_258_, v_data_259_, v_ref_260_, v_msg_261_, v___y_262_, v___y_263_, v___y_264_, v___y_265_);
lean_dec(v___y_265_);
lean_dec_ref(v___y_264_);
lean_dec(v___y_263_);
lean_dec_ref(v___y_262_);
return v_res_267_;
}
}
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__4_spec__5___redArg(lean_object* v_x_268_){
_start:
{
if (lean_obj_tag(v_x_268_) == 0)
{
lean_object* v_a_270_; lean_object* v___x_272_; uint8_t v_isShared_273_; uint8_t v_isSharedCheck_277_; 
v_a_270_ = lean_ctor_get(v_x_268_, 0);
v_isSharedCheck_277_ = !lean_is_exclusive(v_x_268_);
if (v_isSharedCheck_277_ == 0)
{
v___x_272_ = v_x_268_;
v_isShared_273_ = v_isSharedCheck_277_;
goto v_resetjp_271_;
}
else
{
lean_inc(v_a_270_);
lean_dec(v_x_268_);
v___x_272_ = lean_box(0);
v_isShared_273_ = v_isSharedCheck_277_;
goto v_resetjp_271_;
}
v_resetjp_271_:
{
lean_object* v___x_275_; 
if (v_isShared_273_ == 0)
{
lean_ctor_set_tag(v___x_272_, 1);
v___x_275_ = v___x_272_;
goto v_reusejp_274_;
}
else
{
lean_object* v_reuseFailAlloc_276_; 
v_reuseFailAlloc_276_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_276_, 0, v_a_270_);
v___x_275_ = v_reuseFailAlloc_276_;
goto v_reusejp_274_;
}
v_reusejp_274_:
{
return v___x_275_;
}
}
}
else
{
lean_object* v_a_278_; lean_object* v___x_280_; uint8_t v_isShared_281_; uint8_t v_isSharedCheck_285_; 
v_a_278_ = lean_ctor_get(v_x_268_, 0);
v_isSharedCheck_285_ = !lean_is_exclusive(v_x_268_);
if (v_isSharedCheck_285_ == 0)
{
v___x_280_ = v_x_268_;
v_isShared_281_ = v_isSharedCheck_285_;
goto v_resetjp_279_;
}
else
{
lean_inc(v_a_278_);
lean_dec(v_x_268_);
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
lean_ctor_set(v_reuseFailAlloc_284_, 0, v_a_278_);
v___x_283_ = v_reuseFailAlloc_284_;
goto v_reusejp_282_;
}
v_reusejp_282_:
{
return v___x_283_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__4_spec__5___redArg___boxed(lean_object* v_x_286_, lean_object* v___y_287_){
_start:
{
lean_object* v_res_288_; 
v_res_288_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__4_spec__5___redArg(v_x_286_);
return v_res_288_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__4_spec__7(lean_object* v_opts_289_, lean_object* v_opt_290_){
_start:
{
lean_object* v_name_291_; lean_object* v_defValue_292_; lean_object* v_map_293_; lean_object* v___x_294_; 
v_name_291_ = lean_ctor_get(v_opt_290_, 0);
v_defValue_292_ = lean_ctor_get(v_opt_290_, 1);
v_map_293_ = lean_ctor_get(v_opts_289_, 0);
v___x_294_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_293_, v_name_291_);
if (lean_obj_tag(v___x_294_) == 0)
{
lean_inc(v_defValue_292_);
return v_defValue_292_;
}
else
{
lean_object* v_val_295_; 
v_val_295_ = lean_ctor_get(v___x_294_, 0);
lean_inc(v_val_295_);
lean_dec_ref_known(v___x_294_, 1);
if (lean_obj_tag(v_val_295_) == 3)
{
lean_object* v_v_296_; 
v_v_296_ = lean_ctor_get(v_val_295_, 0);
lean_inc(v_v_296_);
lean_dec_ref_known(v_val_295_, 1);
return v_v_296_;
}
else
{
lean_dec(v_val_295_);
lean_inc(v_defValue_292_);
return v_defValue_292_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__4_spec__7___boxed(lean_object* v_opts_297_, lean_object* v_opt_298_){
_start:
{
lean_object* v_res_299_; 
v_res_299_ = lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__4_spec__7(v_opts_297_, v_opt_298_);
lean_dec_ref(v_opt_298_);
lean_dec_ref(v_opts_297_);
return v_res_299_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__4_spec__6(lean_object* v_e_300_){
_start:
{
if (lean_obj_tag(v_e_300_) == 0)
{
uint8_t v___x_301_; 
v___x_301_ = 2;
return v___x_301_;
}
else
{
uint8_t v___x_302_; 
v___x_302_ = 0;
return v___x_302_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__4_spec__6___boxed(lean_object* v_e_303_){
_start:
{
uint8_t v_res_304_; lean_object* v_r_305_; 
v_res_304_ = lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__4_spec__6(v_e_303_);
lean_dec_ref(v_e_303_);
v_r_305_ = lean_box(v_res_304_);
return v_r_305_;
}
}
static double _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__4___closed__0(void){
_start:
{
lean_object* v___x_306_; double v___x_307_; 
v___x_306_ = lean_unsigned_to_nat(0u);
v___x_307_ = lean_float_of_nat(v___x_306_);
return v___x_307_;
}
}
static lean_object* _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__4___closed__2(void){
_start:
{
lean_object* v___x_309_; lean_object* v___x_310_; 
v___x_309_ = ((lean_object*)(lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__4___closed__1));
v___x_310_ = l_Lean_stringToMessageData(v___x_309_);
return v___x_310_;
}
}
static double _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__4___closed__3(void){
_start:
{
lean_object* v___x_311_; double v___x_312_; 
v___x_311_ = lean_unsigned_to_nat(1000u);
v___x_312_ = lean_float_of_nat(v___x_311_);
return v___x_312_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__4(lean_object* v_cls_313_, uint8_t v_collapsed_314_, lean_object* v_tag_315_, lean_object* v_opts_316_, uint8_t v_clsEnabled_317_, lean_object* v_oldTraces_318_, lean_object* v_msg_319_, lean_object* v_resStartStop_320_, lean_object* v___y_321_, lean_object* v___y_322_, lean_object* v___y_323_, lean_object* v___y_324_, lean_object* v___y_325_, lean_object* v___y_326_, lean_object* v___y_327_, lean_object* v___y_328_, lean_object* v___y_329_){
_start:
{
lean_object* v_fst_331_; lean_object* v_snd_332_; lean_object* v___y_334_; lean_object* v___y_335_; lean_object* v_data_336_; lean_object* v_fst_339_; lean_object* v_snd_340_; lean_object* v___x_341_; uint8_t v___x_342_; lean_object* v___y_344_; lean_object* v_a_345_; uint8_t v___y_360_; double v___y_391_; 
v_fst_331_ = lean_ctor_get(v_resStartStop_320_, 0);
lean_inc(v_fst_331_);
v_snd_332_ = lean_ctor_get(v_resStartStop_320_, 1);
lean_inc(v_snd_332_);
lean_dec_ref(v_resStartStop_320_);
v_fst_339_ = lean_ctor_get(v_snd_332_, 0);
lean_inc(v_fst_339_);
v_snd_340_ = lean_ctor_get(v_snd_332_, 1);
lean_inc(v_snd_340_);
lean_dec(v_snd_332_);
v___x_341_ = l_Lean_trace_profiler;
v___x_342_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__0(v_opts_316_, v___x_341_);
if (v___x_342_ == 0)
{
v___y_360_ = v___x_342_;
goto v___jp_359_;
}
else
{
lean_object* v___x_396_; uint8_t v___x_397_; 
v___x_396_ = l_Lean_trace_profiler_useHeartbeats;
v___x_397_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__0(v_opts_316_, v___x_396_);
if (v___x_397_ == 0)
{
lean_object* v___x_398_; lean_object* v___x_399_; double v___x_400_; double v___x_401_; double v___x_402_; 
v___x_398_ = l_Lean_trace_profiler_threshold;
v___x_399_ = lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__4_spec__7(v_opts_316_, v___x_398_);
v___x_400_ = lean_float_of_nat(v___x_399_);
v___x_401_ = lean_float_once(&lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__4___closed__3, &lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__4___closed__3_once, _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__4___closed__3);
v___x_402_ = lean_float_div(v___x_400_, v___x_401_);
v___y_391_ = v___x_402_;
goto v___jp_390_;
}
else
{
lean_object* v___x_403_; lean_object* v___x_404_; double v___x_405_; 
v___x_403_ = l_Lean_trace_profiler_threshold;
v___x_404_ = lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__4_spec__7(v_opts_316_, v___x_403_);
v___x_405_ = lean_float_of_nat(v___x_404_);
v___y_391_ = v___x_405_;
goto v___jp_390_;
}
}
v___jp_333_:
{
lean_object* v___x_337_; 
lean_inc(v___y_335_);
v___x_337_ = lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__4_spec__4___redArg(v_oldTraces_318_, v_data_336_, v___y_335_, v___y_334_, v___y_326_, v___y_327_, v___y_328_, v___y_329_);
if (lean_obj_tag(v___x_337_) == 0)
{
lean_object* v___x_338_; 
lean_dec_ref_known(v___x_337_, 1);
v___x_338_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__4_spec__5___redArg(v_fst_331_);
return v___x_338_;
}
else
{
lean_dec(v_fst_331_);
return v___x_337_;
}
}
v___jp_343_:
{
uint8_t v_result_346_; lean_object* v___x_347_; lean_object* v___x_348_; double v___x_349_; lean_object* v_data_350_; 
v_result_346_ = lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__4_spec__6(v_fst_331_);
v___x_347_ = lean_box(v_result_346_);
v___x_348_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_348_, 0, v___x_347_);
v___x_349_ = lean_float_once(&lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__4___closed__0, &lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__4___closed__0_once, _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__4___closed__0);
lean_inc_ref(v_tag_315_);
lean_inc_ref(v___x_348_);
lean_inc(v_cls_313_);
v_data_350_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v_data_350_, 0, v_cls_313_);
lean_ctor_set(v_data_350_, 1, v___x_348_);
lean_ctor_set(v_data_350_, 2, v_tag_315_);
lean_ctor_set_float(v_data_350_, sizeof(void*)*3, v___x_349_);
lean_ctor_set_float(v_data_350_, sizeof(void*)*3 + 8, v___x_349_);
lean_ctor_set_uint8(v_data_350_, sizeof(void*)*3 + 16, v_collapsed_314_);
if (v___x_342_ == 0)
{
lean_dec_ref_known(v___x_348_, 1);
lean_dec(v_snd_340_);
lean_dec(v_fst_339_);
lean_dec_ref(v_tag_315_);
lean_dec(v_cls_313_);
v___y_334_ = v_a_345_;
v___y_335_ = v___y_344_;
v_data_336_ = v_data_350_;
goto v___jp_333_;
}
else
{
lean_object* v_data_351_; double v___x_352_; double v___x_353_; 
lean_dec_ref_known(v_data_350_, 3);
v_data_351_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v_data_351_, 0, v_cls_313_);
lean_ctor_set(v_data_351_, 1, v___x_348_);
lean_ctor_set(v_data_351_, 2, v_tag_315_);
v___x_352_ = lean_unbox_float(v_fst_339_);
lean_dec(v_fst_339_);
lean_ctor_set_float(v_data_351_, sizeof(void*)*3, v___x_352_);
v___x_353_ = lean_unbox_float(v_snd_340_);
lean_dec(v_snd_340_);
lean_ctor_set_float(v_data_351_, sizeof(void*)*3 + 8, v___x_353_);
lean_ctor_set_uint8(v_data_351_, sizeof(void*)*3 + 16, v_collapsed_314_);
v___y_334_ = v_a_345_;
v___y_335_ = v___y_344_;
v_data_336_ = v_data_351_;
goto v___jp_333_;
}
}
v___jp_354_:
{
lean_object* v_ref_355_; lean_object* v___x_356_; 
v_ref_355_ = lean_ctor_get(v___y_328_, 5);
lean_inc(v___y_329_);
lean_inc_ref(v___y_328_);
lean_inc(v___y_327_);
lean_inc_ref(v___y_326_);
lean_inc(v___y_325_);
lean_inc_ref(v___y_324_);
lean_inc(v___y_323_);
lean_inc_ref(v___y_322_);
lean_inc(v___y_321_);
lean_inc(v_fst_331_);
v___x_356_ = lean_apply_11(v_msg_319_, v_fst_331_, v___y_321_, v___y_322_, v___y_323_, v___y_324_, v___y_325_, v___y_326_, v___y_327_, v___y_328_, v___y_329_, lean_box(0));
if (lean_obj_tag(v___x_356_) == 0)
{
lean_object* v_a_357_; 
v_a_357_ = lean_ctor_get(v___x_356_, 0);
lean_inc(v_a_357_);
lean_dec_ref_known(v___x_356_, 1);
v___y_344_ = v_ref_355_;
v_a_345_ = v_a_357_;
goto v___jp_343_;
}
else
{
lean_object* v___x_358_; 
lean_dec_ref_known(v___x_356_, 1);
v___x_358_ = lean_obj_once(&lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__4___closed__2, &lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__4___closed__2_once, _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__4___closed__2);
v___y_344_ = v_ref_355_;
v_a_345_ = v___x_358_;
goto v___jp_343_;
}
}
v___jp_359_:
{
if (v_clsEnabled_317_ == 0)
{
if (v___y_360_ == 0)
{
lean_object* v___x_361_; lean_object* v_traceState_362_; lean_object* v_env_363_; lean_object* v_nextMacroScope_364_; lean_object* v_ngen_365_; lean_object* v_auxDeclNGen_366_; lean_object* v_cache_367_; lean_object* v_messages_368_; lean_object* v_infoState_369_; lean_object* v_snapshotTasks_370_; lean_object* v___x_372_; uint8_t v_isShared_373_; uint8_t v_isSharedCheck_389_; 
lean_dec(v_snd_340_);
lean_dec(v_fst_339_);
lean_dec_ref(v_msg_319_);
lean_dec_ref(v_tag_315_);
lean_dec(v_cls_313_);
v___x_361_ = lean_st_ref_take(v___y_329_);
v_traceState_362_ = lean_ctor_get(v___x_361_, 4);
v_env_363_ = lean_ctor_get(v___x_361_, 0);
v_nextMacroScope_364_ = lean_ctor_get(v___x_361_, 1);
v_ngen_365_ = lean_ctor_get(v___x_361_, 2);
v_auxDeclNGen_366_ = lean_ctor_get(v___x_361_, 3);
v_cache_367_ = lean_ctor_get(v___x_361_, 5);
v_messages_368_ = lean_ctor_get(v___x_361_, 6);
v_infoState_369_ = lean_ctor_get(v___x_361_, 7);
v_snapshotTasks_370_ = lean_ctor_get(v___x_361_, 8);
v_isSharedCheck_389_ = !lean_is_exclusive(v___x_361_);
if (v_isSharedCheck_389_ == 0)
{
v___x_372_ = v___x_361_;
v_isShared_373_ = v_isSharedCheck_389_;
goto v_resetjp_371_;
}
else
{
lean_inc(v_snapshotTasks_370_);
lean_inc(v_infoState_369_);
lean_inc(v_messages_368_);
lean_inc(v_cache_367_);
lean_inc(v_traceState_362_);
lean_inc(v_auxDeclNGen_366_);
lean_inc(v_ngen_365_);
lean_inc(v_nextMacroScope_364_);
lean_inc(v_env_363_);
lean_dec(v___x_361_);
v___x_372_ = lean_box(0);
v_isShared_373_ = v_isSharedCheck_389_;
goto v_resetjp_371_;
}
v_resetjp_371_:
{
uint64_t v_tid_374_; lean_object* v_traces_375_; lean_object* v___x_377_; uint8_t v_isShared_378_; uint8_t v_isSharedCheck_388_; 
v_tid_374_ = lean_ctor_get_uint64(v_traceState_362_, sizeof(void*)*1);
v_traces_375_ = lean_ctor_get(v_traceState_362_, 0);
v_isSharedCheck_388_ = !lean_is_exclusive(v_traceState_362_);
if (v_isSharedCheck_388_ == 0)
{
v___x_377_ = v_traceState_362_;
v_isShared_378_ = v_isSharedCheck_388_;
goto v_resetjp_376_;
}
else
{
lean_inc(v_traces_375_);
lean_dec(v_traceState_362_);
v___x_377_ = lean_box(0);
v_isShared_378_ = v_isSharedCheck_388_;
goto v_resetjp_376_;
}
v_resetjp_376_:
{
lean_object* v___x_379_; lean_object* v___x_381_; 
v___x_379_ = l_Lean_PersistentArray_append___redArg(v_oldTraces_318_, v_traces_375_);
lean_dec_ref(v_traces_375_);
if (v_isShared_378_ == 0)
{
lean_ctor_set(v___x_377_, 0, v___x_379_);
v___x_381_ = v___x_377_;
goto v_reusejp_380_;
}
else
{
lean_object* v_reuseFailAlloc_387_; 
v_reuseFailAlloc_387_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_387_, 0, v___x_379_);
lean_ctor_set_uint64(v_reuseFailAlloc_387_, sizeof(void*)*1, v_tid_374_);
v___x_381_ = v_reuseFailAlloc_387_;
goto v_reusejp_380_;
}
v_reusejp_380_:
{
lean_object* v___x_383_; 
if (v_isShared_373_ == 0)
{
lean_ctor_set(v___x_372_, 4, v___x_381_);
v___x_383_ = v___x_372_;
goto v_reusejp_382_;
}
else
{
lean_object* v_reuseFailAlloc_386_; 
v_reuseFailAlloc_386_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_386_, 0, v_env_363_);
lean_ctor_set(v_reuseFailAlloc_386_, 1, v_nextMacroScope_364_);
lean_ctor_set(v_reuseFailAlloc_386_, 2, v_ngen_365_);
lean_ctor_set(v_reuseFailAlloc_386_, 3, v_auxDeclNGen_366_);
lean_ctor_set(v_reuseFailAlloc_386_, 4, v___x_381_);
lean_ctor_set(v_reuseFailAlloc_386_, 5, v_cache_367_);
lean_ctor_set(v_reuseFailAlloc_386_, 6, v_messages_368_);
lean_ctor_set(v_reuseFailAlloc_386_, 7, v_infoState_369_);
lean_ctor_set(v_reuseFailAlloc_386_, 8, v_snapshotTasks_370_);
v___x_383_ = v_reuseFailAlloc_386_;
goto v_reusejp_382_;
}
v_reusejp_382_:
{
lean_object* v___x_384_; lean_object* v___x_385_; 
v___x_384_ = lean_st_ref_set(v___y_329_, v___x_383_);
v___x_385_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__4_spec__5___redArg(v_fst_331_);
return v___x_385_;
}
}
}
}
}
else
{
goto v___jp_354_;
}
}
else
{
goto v___jp_354_;
}
}
v___jp_390_:
{
double v___x_392_; double v___x_393_; double v___x_394_; uint8_t v___x_395_; 
v___x_392_ = lean_unbox_float(v_snd_340_);
v___x_393_ = lean_unbox_float(v_fst_339_);
v___x_394_ = lean_float_sub(v___x_392_, v___x_393_);
v___x_395_ = lean_float_decLt(v___y_391_, v___x_394_);
v___y_360_ = v___x_395_;
goto v___jp_359_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__4___boxed(lean_object** _args){
lean_object* v_cls_406_ = _args[0];
lean_object* v_collapsed_407_ = _args[1];
lean_object* v_tag_408_ = _args[2];
lean_object* v_opts_409_ = _args[3];
lean_object* v_clsEnabled_410_ = _args[4];
lean_object* v_oldTraces_411_ = _args[5];
lean_object* v_msg_412_ = _args[6];
lean_object* v_resStartStop_413_ = _args[7];
lean_object* v___y_414_ = _args[8];
lean_object* v___y_415_ = _args[9];
lean_object* v___y_416_ = _args[10];
lean_object* v___y_417_ = _args[11];
lean_object* v___y_418_ = _args[12];
lean_object* v___y_419_ = _args[13];
lean_object* v___y_420_ = _args[14];
lean_object* v___y_421_ = _args[15];
lean_object* v___y_422_ = _args[16];
lean_object* v___y_423_ = _args[17];
_start:
{
uint8_t v_collapsed_boxed_424_; uint8_t v_clsEnabled_boxed_425_; lean_object* v_res_426_; 
v_collapsed_boxed_424_ = lean_unbox(v_collapsed_407_);
v_clsEnabled_boxed_425_ = lean_unbox(v_clsEnabled_410_);
v_res_426_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__4(v_cls_406_, v_collapsed_boxed_424_, v_tag_408_, v_opts_409_, v_clsEnabled_boxed_425_, v_oldTraces_411_, v_msg_412_, v_resStartStop_413_, v___y_414_, v___y_415_, v___y_416_, v___y_417_, v___y_418_, v___y_419_, v___y_420_, v___y_421_, v___y_422_);
lean_dec(v___y_422_);
lean_dec_ref(v___y_421_);
lean_dec(v___y_420_);
lean_dec_ref(v___y_419_);
lean_dec(v___y_418_);
lean_dec_ref(v___y_417_);
lean_dec(v___y_416_);
lean_dec_ref(v___y_415_);
lean_dec(v___y_414_);
lean_dec_ref(v_opts_409_);
return v_res_426_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__1___redArg(lean_object* v_opt_427_, lean_object* v___y_428_){
_start:
{
lean_object* v_options_430_; lean_object* v_option_431_; uint8_t v___x_432_; lean_object* v___x_433_; lean_object* v___x_434_; 
v_options_430_ = lean_ctor_get(v___y_428_, 2);
v_option_431_ = lean_ctor_get(v_opt_427_, 1);
v___x_432_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__0(v_options_430_, v_option_431_);
v___x_433_ = lean_box(v___x_432_);
v___x_434_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_434_, 0, v___x_433_);
return v___x_434_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__1___redArg___boxed(lean_object* v_opt_435_, lean_object* v___y_436_, lean_object* v___y_437_){
_start:
{
lean_object* v_res_438_; 
v_res_438_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__1___redArg(v_opt_435_, v___y_436_);
lean_dec_ref(v___y_436_);
lean_dec_ref(v_opt_435_);
return v_res_438_;
}
}
static double _init_lp_aesop___private_Aesop_Main_0__Aesop_evalAesop_go___closed__1(void){
_start:
{
lean_object* v___x_440_; double v___x_441_; 
v___x_440_ = lean_unsigned_to_nat(1000000000u);
v___x_441_ = lean_float_of_nat(v___x_440_);
return v___x_441_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Main_0__Aesop_evalAesop_go___closed__4(void){
_start:
{
lean_object* v___x_445_; lean_object* v___x_446_; 
v___x_445_ = ((lean_object*)(lp_aesop___private_Aesop_Main_0__Aesop_evalAesop_go___closed__3));
v___x_446_ = l_Lean_MessageData_ofFormat(v___x_445_);
return v___x_446_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Main_0__Aesop_evalAesop_go___closed__5(void){
_start:
{
lean_object* v___x_447_; lean_object* v___f_448_; 
v___x_447_ = lean_obj_once(&lp_aesop___private_Aesop_Main_0__Aesop_evalAesop_go___closed__4, &lp_aesop___private_Aesop_Main_0__Aesop_evalAesop_go___closed__4_once, _init_lp_aesop___private_Aesop_Main_0__Aesop_evalAesop_go___closed__4);
v___f_448_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Main_0__Aesop_evalAesop_go___lam__1___boxed), 12, 1);
lean_closure_set(v___f_448_, 0, v___x_447_);
return v___f_448_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Main_0__Aesop_evalAesop_go(lean_object* v_stx_452_, lean_object* v_goal_453_, lean_object* v_a_454_, lean_object* v_a_455_, lean_object* v_a_456_, lean_object* v_a_457_, lean_object* v_a_458_, lean_object* v_a_459_, lean_object* v_a_460_, lean_object* v_a_461_, lean_object* v_a_462_){
_start:
{
lean_object* v___y_465_; lean_object* v___y_466_; lean_object* v___y_531_; lean_object* v___y_532_; uint8_t v_a_533_; lean_object* v___y_576_; lean_object* v___y_577_; lean_object* v___y_578_; lean_object* v___y_582_; uint8_t v_a_583_; lean_object* v___y_608_; lean_object* v___y_609_; lean_object* v___y_610_; lean_object* v___y_644_; lean_object* v___y_645_; lean_object* v___y_646_; lean_object* v___y_702_; lean_object* v___y_703_; lean_object* v___y_704_; lean_object* v___y_705_; lean_object* v_options_708_; lean_object* v_inheritedTraceOptions_709_; lean_object* v___y_711_; lean_object* v___y_712_; lean_object* v___y_713_; lean_object* v___y_714_; lean_object* v___y_734_; lean_object* v___y_735_; lean_object* v___y_736_; uint8_t v___y_737_; lean_object* v___y_738_; lean_object* v___y_739_; lean_object* v___y_740_; lean_object* v___y_741_; lean_object* v___y_742_; uint8_t v___y_743_; lean_object* v___y_744_; lean_object* v_a_745_; lean_object* v___y_755_; lean_object* v___y_756_; lean_object* v___y_757_; uint8_t v___y_758_; lean_object* v___y_759_; lean_object* v___y_760_; lean_object* v___y_761_; lean_object* v___y_762_; lean_object* v___y_763_; uint8_t v___y_764_; lean_object* v___y_765_; lean_object* v_a_766_; lean_object* v___y_779_; lean_object* v___y_780_; lean_object* v___y_781_; uint8_t v___y_782_; lean_object* v___y_783_; lean_object* v___y_784_; lean_object* v___y_785_; lean_object* v___y_786_; uint8_t v___y_787_; lean_object* v___y_788_; lean_object* v___y_830_; lean_object* v___y_831_; lean_object* v_a_832_; lean_object* v___y_846_; lean_object* v___y_847_; lean_object* v___y_859_; lean_object* v___y_860_; lean_object* v___y_894_; lean_object* v___y_895_; lean_object* v___y_896_; lean_object* v___y_900_; lean_object* v_a_901_; lean_object* v___y_913_; lean_object* v___y_914_; lean_object* v___y_925_; lean_object* v___y_928_; lean_object* v___y_929_; lean_object* v___y_930_; lean_object* v___y_950_; lean_object* v___y_951_; uint8_t v___y_952_; lean_object* v___y_953_; uint8_t v___y_954_; lean_object* v___y_955_; lean_object* v___y_956_; lean_object* v___y_957_; lean_object* v___y_958_; lean_object* v___y_959_; lean_object* v_a_960_; lean_object* v___y_970_; lean_object* v___y_971_; lean_object* v___y_972_; uint8_t v___y_973_; lean_object* v___y_974_; uint8_t v___y_975_; lean_object* v___y_976_; lean_object* v___y_977_; lean_object* v___y_978_; lean_object* v___y_979_; lean_object* v_a_980_; lean_object* v___y_993_; lean_object* v___y_994_; uint8_t v___y_995_; lean_object* v___y_996_; uint8_t v___y_997_; lean_object* v___y_998_; lean_object* v___y_999_; lean_object* v___y_1000_; lean_object* v___y_1001_; lean_object* v___y_1043_; lean_object* v_a_1044_; lean_object* v___y_1058_; lean_object* v___y_1070_; lean_object* v___y_1104_; lean_object* v___y_1105_; lean_object* v_a_1109_; lean_object* v___y_1149_; lean_object* v___y_1164_; lean_object* v___y_1174_; uint8_t v_a_1175_; lean_object* v___y_1177_; lean_object* v___y_1178_; lean_object* v___y_1194_; lean_object* v___x_1197_; uint8_t v___x_1198_; 
v_options_708_ = lean_ctor_get(v_a_461_, 2);
v_inheritedTraceOptions_709_ = lean_ctor_get(v_a_461_, 13);
v___x_1197_ = lp_aesop_Aesop_aesop_collectStats;
v___x_1198_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__0(v_options_708_, v___x_1197_);
if (v___x_1198_ == 0)
{
lean_object* v___x_1199_; lean_object* v___x_1200_; lean_object* v_a_1201_; uint8_t v___x_1202_; 
v___x_1199_ = lp_aesop_Aesop_TraceOption_stats;
v___x_1200_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__1___redArg(v___x_1199_, v_a_461_);
v_a_1201_ = lean_ctor_get(v___x_1200_, 0);
lean_inc(v_a_1201_);
v___x_1202_ = lean_unbox(v_a_1201_);
lean_dec(v_a_1201_);
if (v___x_1202_ == 0)
{
lean_object* v___x_1203_; lean_object* v___x_1204_; lean_object* v___x_1205_; uint8_t v___x_1206_; 
lean_dec_ref(v___x_1200_);
v___x_1203_ = lp_aesop_Aesop_aesop_stats_file;
v___x_1204_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__2(v_options_708_, v___x_1203_);
v___x_1205_ = ((lean_object*)(lp_aesop___private_Aesop_Main_0__Aesop_evalAesop_go___closed__0));
v___x_1206_ = lean_string_dec_eq(v___x_1204_, v___x_1205_);
lean_dec_ref(v___x_1204_);
if (v___x_1206_ == 0)
{
goto v___jp_1181_;
}
else
{
goto v___jp_1152_;
}
}
else
{
v___y_1194_ = v___x_1200_;
goto v___jp_1193_;
}
}
else
{
goto v___jp_1181_;
}
v___jp_464_:
{
lean_object* v___x_467_; lean_object* v___x_468_; lean_object* v_options_469_; lean_object* v_simpConfig_470_; lean_object* v_simpConfigSyntax_x3f_471_; lean_object* v___x_472_; lean_object* v___x_473_; 
v___x_467_ = lean_io_mono_nanos_now();
v___x_468_ = lean_st_ref_get(v_a_454_);
v_options_469_ = lean_ctor_get(v___y_466_, 3);
lean_inc_ref(v_options_469_);
v_simpConfig_470_ = lean_ctor_get(v___y_466_, 4);
lean_inc_ref(v_simpConfig_470_);
v_simpConfigSyntax_x3f_471_ = lean_ctor_get(v___y_466_, 5);
lean_inc(v_simpConfigSyntax_x3f_471_);
lean_dec_ref(v___y_466_);
v___x_472_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_472_, 0, v___y_465_);
v___x_473_ = lp_aesop_Aesop_search(v_goal_453_, v___x_472_, v_options_469_, v_simpConfig_470_, v_simpConfigSyntax_x3f_471_, v___x_468_, v_a_459_, v_a_460_, v_a_461_, v_a_462_);
if (lean_obj_tag(v___x_473_) == 0)
{
lean_object* v_a_474_; lean_object* v_fst_475_; lean_object* v_snd_476_; lean_object* v___x_477_; lean_object* v___x_478_; lean_object* v___x_479_; 
v_a_474_ = lean_ctor_get(v___x_473_, 0);
lean_inc(v_a_474_);
lean_dec_ref_known(v___x_473_, 1);
v_fst_475_ = lean_ctor_get(v_a_474_, 0);
lean_inc(v_fst_475_);
v_snd_476_ = lean_ctor_get(v_a_474_, 1);
lean_inc(v_snd_476_);
lean_dec(v_a_474_);
v___x_477_ = lean_array_get_size(v_fst_475_);
v___x_478_ = lean_array_to_list(v_fst_475_);
v___x_479_ = l_Lean_Elab_Tactic_replaceMainGoal___redArg(v___x_478_, v_a_456_, v_a_459_, v_a_460_, v_a_461_, v_a_462_);
if (lean_obj_tag(v___x_479_) == 0)
{
lean_object* v___x_481_; uint8_t v_isShared_482_; uint8_t v_isSharedCheck_512_; 
v_isSharedCheck_512_ = !lean_is_exclusive(v___x_479_);
if (v_isSharedCheck_512_ == 0)
{
lean_object* v_unused_513_; 
v_unused_513_ = lean_ctor_get(v___x_479_, 0);
lean_dec(v_unused_513_);
v___x_481_ = v___x_479_;
v_isShared_482_ = v_isSharedCheck_512_;
goto v_resetjp_480_;
}
else
{
lean_dec(v___x_479_);
v___x_481_ = lean_box(0);
v_isShared_482_ = v_isSharedCheck_512_;
goto v_resetjp_480_;
}
v_resetjp_480_:
{
lean_object* v___x_483_; lean_object* v___x_484_; lean_object* v___x_485_; lean_object* v___x_486_; lean_object* v_total_487_; lean_object* v_configParsing_488_; lean_object* v_ruleSetConstruction_489_; lean_object* v_ruleSelection_490_; lean_object* v_script_491_; lean_object* v_forwardState_492_; lean_object* v_scriptGenerated_493_; lean_object* v_ruleStats_494_; lean_object* v_goalStats_495_; lean_object* v___x_497_; uint8_t v_isShared_498_; uint8_t v_isSharedCheck_510_; 
v___x_483_ = lean_st_ref_take(v_a_454_);
lean_dec(v___x_483_);
v___x_484_ = lean_st_ref_set(v_a_454_, v_snd_476_);
v___x_485_ = lean_io_mono_nanos_now();
v___x_486_ = lean_st_ref_take(v_a_454_);
v_total_487_ = lean_ctor_get(v___x_486_, 0);
v_configParsing_488_ = lean_ctor_get(v___x_486_, 1);
v_ruleSetConstruction_489_ = lean_ctor_get(v___x_486_, 2);
v_ruleSelection_490_ = lean_ctor_get(v___x_486_, 4);
v_script_491_ = lean_ctor_get(v___x_486_, 5);
v_forwardState_492_ = lean_ctor_get(v___x_486_, 6);
v_scriptGenerated_493_ = lean_ctor_get(v___x_486_, 7);
v_ruleStats_494_ = lean_ctor_get(v___x_486_, 8);
v_goalStats_495_ = lean_ctor_get(v___x_486_, 9);
v_isSharedCheck_510_ = !lean_is_exclusive(v___x_486_);
if (v_isSharedCheck_510_ == 0)
{
lean_object* v_unused_511_; 
v_unused_511_ = lean_ctor_get(v___x_486_, 3);
lean_dec(v_unused_511_);
v___x_497_ = v___x_486_;
v_isShared_498_ = v_isSharedCheck_510_;
goto v_resetjp_496_;
}
else
{
lean_inc(v_goalStats_495_);
lean_inc(v_ruleStats_494_);
lean_inc(v_scriptGenerated_493_);
lean_inc(v_forwardState_492_);
lean_inc(v_script_491_);
lean_inc(v_ruleSelection_490_);
lean_inc(v_ruleSetConstruction_489_);
lean_inc(v_configParsing_488_);
lean_inc(v_total_487_);
lean_dec(v___x_486_);
v___x_497_ = lean_box(0);
v_isShared_498_ = v_isSharedCheck_510_;
goto v_resetjp_496_;
}
v_resetjp_496_:
{
lean_object* v___x_499_; lean_object* v___x_501_; 
v___x_499_ = lean_nat_sub(v___x_485_, v___x_467_);
lean_dec(v___x_467_);
lean_dec(v___x_485_);
if (v_isShared_498_ == 0)
{
lean_ctor_set(v___x_497_, 3, v___x_499_);
v___x_501_ = v___x_497_;
goto v_reusejp_500_;
}
else
{
lean_object* v_reuseFailAlloc_509_; 
v_reuseFailAlloc_509_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_509_, 0, v_total_487_);
lean_ctor_set(v_reuseFailAlloc_509_, 1, v_configParsing_488_);
lean_ctor_set(v_reuseFailAlloc_509_, 2, v_ruleSetConstruction_489_);
lean_ctor_set(v_reuseFailAlloc_509_, 3, v___x_499_);
lean_ctor_set(v_reuseFailAlloc_509_, 4, v_ruleSelection_490_);
lean_ctor_set(v_reuseFailAlloc_509_, 5, v_script_491_);
lean_ctor_set(v_reuseFailAlloc_509_, 6, v_forwardState_492_);
lean_ctor_set(v_reuseFailAlloc_509_, 7, v_scriptGenerated_493_);
lean_ctor_set(v_reuseFailAlloc_509_, 8, v_ruleStats_494_);
lean_ctor_set(v_reuseFailAlloc_509_, 9, v_goalStats_495_);
v___x_501_ = v_reuseFailAlloc_509_;
goto v_reusejp_500_;
}
v_reusejp_500_:
{
lean_object* v___x_502_; lean_object* v___x_503_; uint8_t v___x_504_; lean_object* v___x_505_; lean_object* v___x_507_; 
v___x_502_ = lean_st_ref_set(v_a_454_, v___x_501_);
v___x_503_ = lean_unsigned_to_nat(0u);
v___x_504_ = lean_nat_dec_eq(v___x_477_, v___x_503_);
v___x_505_ = lean_box(v___x_504_);
if (v_isShared_482_ == 0)
{
lean_ctor_set(v___x_481_, 0, v___x_505_);
v___x_507_ = v___x_481_;
goto v_reusejp_506_;
}
else
{
lean_object* v_reuseFailAlloc_508_; 
v_reuseFailAlloc_508_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_508_, 0, v___x_505_);
v___x_507_ = v_reuseFailAlloc_508_;
goto v_reusejp_506_;
}
v_reusejp_506_:
{
return v___x_507_;
}
}
}
}
}
else
{
lean_object* v_a_514_; lean_object* v___x_516_; uint8_t v_isShared_517_; uint8_t v_isSharedCheck_521_; 
lean_dec(v_snd_476_);
lean_dec(v___x_467_);
v_a_514_ = lean_ctor_get(v___x_479_, 0);
v_isSharedCheck_521_ = !lean_is_exclusive(v___x_479_);
if (v_isSharedCheck_521_ == 0)
{
v___x_516_ = v___x_479_;
v_isShared_517_ = v_isSharedCheck_521_;
goto v_resetjp_515_;
}
else
{
lean_inc(v_a_514_);
lean_dec(v___x_479_);
v___x_516_ = lean_box(0);
v_isShared_517_ = v_isSharedCheck_521_;
goto v_resetjp_515_;
}
v_resetjp_515_:
{
lean_object* v___x_519_; 
if (v_isShared_517_ == 0)
{
v___x_519_ = v___x_516_;
goto v_reusejp_518_;
}
else
{
lean_object* v_reuseFailAlloc_520_; 
v_reuseFailAlloc_520_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_520_, 0, v_a_514_);
v___x_519_ = v_reuseFailAlloc_520_;
goto v_reusejp_518_;
}
v_reusejp_518_:
{
return v___x_519_;
}
}
}
}
else
{
lean_object* v_a_522_; lean_object* v___x_524_; uint8_t v_isShared_525_; uint8_t v_isSharedCheck_529_; 
lean_dec(v___x_467_);
v_a_522_ = lean_ctor_get(v___x_473_, 0);
v_isSharedCheck_529_ = !lean_is_exclusive(v___x_473_);
if (v_isSharedCheck_529_ == 0)
{
v___x_524_ = v___x_473_;
v_isShared_525_ = v_isSharedCheck_529_;
goto v_resetjp_523_;
}
else
{
lean_inc(v_a_522_);
lean_dec(v___x_473_);
v___x_524_ = lean_box(0);
v_isShared_525_ = v_isSharedCheck_529_;
goto v_resetjp_523_;
}
v_resetjp_523_:
{
lean_object* v___x_527_; 
if (v_isShared_525_ == 0)
{
v___x_527_ = v___x_524_;
goto v_reusejp_526_;
}
else
{
lean_object* v_reuseFailAlloc_528_; 
v_reuseFailAlloc_528_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_528_, 0, v_a_522_);
v___x_527_ = v_reuseFailAlloc_528_;
goto v_reusejp_526_;
}
v_reusejp_526_:
{
return v___x_527_;
}
}
}
}
v___jp_530_:
{
if (v_a_533_ == 0)
{
lean_object* v___x_534_; lean_object* v_options_535_; lean_object* v_simpConfig_536_; lean_object* v_simpConfigSyntax_x3f_537_; lean_object* v___x_538_; lean_object* v___x_539_; 
v___x_534_ = lean_st_ref_get(v_a_454_);
v_options_535_ = lean_ctor_get(v___y_532_, 3);
lean_inc_ref(v_options_535_);
v_simpConfig_536_ = lean_ctor_get(v___y_532_, 4);
lean_inc_ref(v_simpConfig_536_);
v_simpConfigSyntax_x3f_537_ = lean_ctor_get(v___y_532_, 5);
lean_inc(v_simpConfigSyntax_x3f_537_);
lean_dec_ref(v___y_532_);
v___x_538_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_538_, 0, v___y_531_);
v___x_539_ = lp_aesop_Aesop_search(v_goal_453_, v___x_538_, v_options_535_, v_simpConfig_536_, v_simpConfigSyntax_x3f_537_, v___x_534_, v_a_459_, v_a_460_, v_a_461_, v_a_462_);
if (lean_obj_tag(v___x_539_) == 0)
{
lean_object* v_a_540_; lean_object* v_fst_541_; lean_object* v_snd_542_; lean_object* v___x_543_; lean_object* v___x_544_; lean_object* v___x_545_; 
v_a_540_ = lean_ctor_get(v___x_539_, 0);
lean_inc(v_a_540_);
lean_dec_ref_known(v___x_539_, 1);
v_fst_541_ = lean_ctor_get(v_a_540_, 0);
lean_inc(v_fst_541_);
v_snd_542_ = lean_ctor_get(v_a_540_, 1);
lean_inc(v_snd_542_);
lean_dec(v_a_540_);
v___x_543_ = lean_array_get_size(v_fst_541_);
v___x_544_ = lean_array_to_list(v_fst_541_);
v___x_545_ = l_Lean_Elab_Tactic_replaceMainGoal___redArg(v___x_544_, v_a_456_, v_a_459_, v_a_460_, v_a_461_, v_a_462_);
if (lean_obj_tag(v___x_545_) == 0)
{
lean_object* v___x_547_; uint8_t v_isShared_548_; uint8_t v_isSharedCheck_557_; 
v_isSharedCheck_557_ = !lean_is_exclusive(v___x_545_);
if (v_isSharedCheck_557_ == 0)
{
lean_object* v_unused_558_; 
v_unused_558_ = lean_ctor_get(v___x_545_, 0);
lean_dec(v_unused_558_);
v___x_547_ = v___x_545_;
v_isShared_548_ = v_isSharedCheck_557_;
goto v_resetjp_546_;
}
else
{
lean_dec(v___x_545_);
v___x_547_ = lean_box(0);
v_isShared_548_ = v_isSharedCheck_557_;
goto v_resetjp_546_;
}
v_resetjp_546_:
{
lean_object* v___x_549_; lean_object* v___x_550_; lean_object* v___x_551_; uint8_t v___x_552_; lean_object* v___x_553_; lean_object* v___x_555_; 
v___x_549_ = lean_st_ref_take(v_a_454_);
lean_dec(v___x_549_);
v___x_550_ = lean_st_ref_set(v_a_454_, v_snd_542_);
v___x_551_ = lean_unsigned_to_nat(0u);
v___x_552_ = lean_nat_dec_eq(v___x_543_, v___x_551_);
v___x_553_ = lean_box(v___x_552_);
if (v_isShared_548_ == 0)
{
lean_ctor_set(v___x_547_, 0, v___x_553_);
v___x_555_ = v___x_547_;
goto v_reusejp_554_;
}
else
{
lean_object* v_reuseFailAlloc_556_; 
v_reuseFailAlloc_556_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_556_, 0, v___x_553_);
v___x_555_ = v_reuseFailAlloc_556_;
goto v_reusejp_554_;
}
v_reusejp_554_:
{
return v___x_555_;
}
}
}
else
{
lean_object* v_a_559_; lean_object* v___x_561_; uint8_t v_isShared_562_; uint8_t v_isSharedCheck_566_; 
lean_dec(v_snd_542_);
v_a_559_ = lean_ctor_get(v___x_545_, 0);
v_isSharedCheck_566_ = !lean_is_exclusive(v___x_545_);
if (v_isSharedCheck_566_ == 0)
{
v___x_561_ = v___x_545_;
v_isShared_562_ = v_isSharedCheck_566_;
goto v_resetjp_560_;
}
else
{
lean_inc(v_a_559_);
lean_dec(v___x_545_);
v___x_561_ = lean_box(0);
v_isShared_562_ = v_isSharedCheck_566_;
goto v_resetjp_560_;
}
v_resetjp_560_:
{
lean_object* v___x_564_; 
if (v_isShared_562_ == 0)
{
v___x_564_ = v___x_561_;
goto v_reusejp_563_;
}
else
{
lean_object* v_reuseFailAlloc_565_; 
v_reuseFailAlloc_565_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_565_, 0, v_a_559_);
v___x_564_ = v_reuseFailAlloc_565_;
goto v_reusejp_563_;
}
v_reusejp_563_:
{
return v___x_564_;
}
}
}
}
else
{
lean_object* v_a_567_; lean_object* v___x_569_; uint8_t v_isShared_570_; uint8_t v_isSharedCheck_574_; 
v_a_567_ = lean_ctor_get(v___x_539_, 0);
v_isSharedCheck_574_ = !lean_is_exclusive(v___x_539_);
if (v_isSharedCheck_574_ == 0)
{
v___x_569_ = v___x_539_;
v_isShared_570_ = v_isSharedCheck_574_;
goto v_resetjp_568_;
}
else
{
lean_inc(v_a_567_);
lean_dec(v___x_539_);
v___x_569_ = lean_box(0);
v_isShared_570_ = v_isSharedCheck_574_;
goto v_resetjp_568_;
}
v_resetjp_568_:
{
lean_object* v___x_572_; 
if (v_isShared_570_ == 0)
{
v___x_572_ = v___x_569_;
goto v_reusejp_571_;
}
else
{
lean_object* v_reuseFailAlloc_573_; 
v_reuseFailAlloc_573_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_573_, 0, v_a_567_);
v___x_572_ = v_reuseFailAlloc_573_;
goto v_reusejp_571_;
}
v_reusejp_571_:
{
return v___x_572_;
}
}
}
}
else
{
v___y_465_ = v___y_531_;
v___y_466_ = v___y_532_;
goto v___jp_464_;
}
}
v___jp_575_:
{
lean_object* v_a_579_; uint8_t v___x_580_; 
v_a_579_ = lean_ctor_get(v___y_578_, 0);
lean_inc(v_a_579_);
lean_dec_ref(v___y_578_);
v___x_580_ = lean_unbox(v_a_579_);
lean_dec(v_a_579_);
v___y_531_ = v___y_576_;
v___y_532_ = v___y_577_;
v_a_533_ = v___x_580_;
goto v___jp_530_;
}
v___jp_581_:
{
lean_object* v___x_584_; lean_object* v___x_585_; lean_object* v_configParsing_586_; lean_object* v_ruleSetConstruction_587_; lean_object* v_search_588_; lean_object* v_ruleSelection_589_; lean_object* v_script_590_; lean_object* v_forwardState_591_; lean_object* v_scriptGenerated_592_; lean_object* v_ruleStats_593_; lean_object* v_goalStats_594_; lean_object* v___x_596_; uint8_t v_isShared_597_; uint8_t v_isSharedCheck_605_; 
v___x_584_ = lean_io_mono_nanos_now();
v___x_585_ = lean_st_ref_take(v_a_454_);
v_configParsing_586_ = lean_ctor_get(v___x_585_, 1);
v_ruleSetConstruction_587_ = lean_ctor_get(v___x_585_, 2);
v_search_588_ = lean_ctor_get(v___x_585_, 3);
v_ruleSelection_589_ = lean_ctor_get(v___x_585_, 4);
v_script_590_ = lean_ctor_get(v___x_585_, 5);
v_forwardState_591_ = lean_ctor_get(v___x_585_, 6);
v_scriptGenerated_592_ = lean_ctor_get(v___x_585_, 7);
v_ruleStats_593_ = lean_ctor_get(v___x_585_, 8);
v_goalStats_594_ = lean_ctor_get(v___x_585_, 9);
v_isSharedCheck_605_ = !lean_is_exclusive(v___x_585_);
if (v_isSharedCheck_605_ == 0)
{
lean_object* v_unused_606_; 
v_unused_606_ = lean_ctor_get(v___x_585_, 0);
lean_dec(v_unused_606_);
v___x_596_ = v___x_585_;
v_isShared_597_ = v_isSharedCheck_605_;
goto v_resetjp_595_;
}
else
{
lean_inc(v_goalStats_594_);
lean_inc(v_ruleStats_593_);
lean_inc(v_scriptGenerated_592_);
lean_inc(v_forwardState_591_);
lean_inc(v_script_590_);
lean_inc(v_ruleSelection_589_);
lean_inc(v_search_588_);
lean_inc(v_ruleSetConstruction_587_);
lean_inc(v_configParsing_586_);
lean_dec(v___x_585_);
v___x_596_ = lean_box(0);
v_isShared_597_ = v_isSharedCheck_605_;
goto v_resetjp_595_;
}
v_resetjp_595_:
{
lean_object* v___x_598_; lean_object* v___x_600_; 
v___x_598_ = lean_nat_sub(v___x_584_, v___y_582_);
lean_dec(v___y_582_);
lean_dec(v___x_584_);
if (v_isShared_597_ == 0)
{
lean_ctor_set(v___x_596_, 0, v___x_598_);
v___x_600_ = v___x_596_;
goto v_reusejp_599_;
}
else
{
lean_object* v_reuseFailAlloc_604_; 
v_reuseFailAlloc_604_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_604_, 0, v___x_598_);
lean_ctor_set(v_reuseFailAlloc_604_, 1, v_configParsing_586_);
lean_ctor_set(v_reuseFailAlloc_604_, 2, v_ruleSetConstruction_587_);
lean_ctor_set(v_reuseFailAlloc_604_, 3, v_search_588_);
lean_ctor_set(v_reuseFailAlloc_604_, 4, v_ruleSelection_589_);
lean_ctor_set(v_reuseFailAlloc_604_, 5, v_script_590_);
lean_ctor_set(v_reuseFailAlloc_604_, 6, v_forwardState_591_);
lean_ctor_set(v_reuseFailAlloc_604_, 7, v_scriptGenerated_592_);
lean_ctor_set(v_reuseFailAlloc_604_, 8, v_ruleStats_593_);
lean_ctor_set(v_reuseFailAlloc_604_, 9, v_goalStats_594_);
v___x_600_ = v_reuseFailAlloc_604_;
goto v_reusejp_599_;
}
v_reusejp_599_:
{
lean_object* v___x_601_; lean_object* v___x_602_; lean_object* v___x_603_; 
v___x_601_ = lean_st_ref_set(v_a_454_, v___x_600_);
v___x_602_ = lean_box(v_a_583_);
v___x_603_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_603_, 0, v___x_602_);
return v___x_603_;
}
}
}
v___jp_607_:
{
lean_object* v___x_611_; lean_object* v_options_612_; lean_object* v_simpConfig_613_; lean_object* v_simpConfigSyntax_x3f_614_; lean_object* v___x_615_; lean_object* v___x_616_; 
v___x_611_ = lean_st_ref_get(v_a_454_);
v_options_612_ = lean_ctor_get(v___y_609_, 3);
lean_inc_ref(v_options_612_);
v_simpConfig_613_ = lean_ctor_get(v___y_609_, 4);
lean_inc_ref(v_simpConfig_613_);
v_simpConfigSyntax_x3f_614_ = lean_ctor_get(v___y_609_, 5);
lean_inc(v_simpConfigSyntax_x3f_614_);
lean_dec_ref(v___y_609_);
v___x_615_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_615_, 0, v___y_608_);
v___x_616_ = lp_aesop_Aesop_search(v_goal_453_, v___x_615_, v_options_612_, v_simpConfig_613_, v_simpConfigSyntax_x3f_614_, v___x_611_, v_a_459_, v_a_460_, v_a_461_, v_a_462_);
if (lean_obj_tag(v___x_616_) == 0)
{
lean_object* v_a_617_; lean_object* v_fst_618_; lean_object* v_snd_619_; lean_object* v___x_620_; lean_object* v___x_621_; lean_object* v___x_622_; 
v_a_617_ = lean_ctor_get(v___x_616_, 0);
lean_inc(v_a_617_);
lean_dec_ref_known(v___x_616_, 1);
v_fst_618_ = lean_ctor_get(v_a_617_, 0);
lean_inc(v_fst_618_);
v_snd_619_ = lean_ctor_get(v_a_617_, 1);
lean_inc(v_snd_619_);
lean_dec(v_a_617_);
v___x_620_ = lean_array_get_size(v_fst_618_);
v___x_621_ = lean_array_to_list(v_fst_618_);
v___x_622_ = l_Lean_Elab_Tactic_replaceMainGoal___redArg(v___x_621_, v_a_456_, v_a_459_, v_a_460_, v_a_461_, v_a_462_);
if (lean_obj_tag(v___x_622_) == 0)
{
lean_object* v___x_623_; lean_object* v___x_624_; lean_object* v___x_625_; uint8_t v___x_626_; 
lean_dec_ref_known(v___x_622_, 1);
v___x_623_ = lean_st_ref_take(v_a_454_);
lean_dec(v___x_623_);
v___x_624_ = lean_st_ref_set(v_a_454_, v_snd_619_);
v___x_625_ = lean_unsigned_to_nat(0u);
v___x_626_ = lean_nat_dec_eq(v___x_620_, v___x_625_);
v___y_582_ = v___y_610_;
v_a_583_ = v___x_626_;
goto v___jp_581_;
}
else
{
lean_object* v_a_627_; lean_object* v___x_629_; uint8_t v_isShared_630_; uint8_t v_isSharedCheck_634_; 
lean_dec(v_snd_619_);
lean_dec(v___y_610_);
v_a_627_ = lean_ctor_get(v___x_622_, 0);
v_isSharedCheck_634_ = !lean_is_exclusive(v___x_622_);
if (v_isSharedCheck_634_ == 0)
{
v___x_629_ = v___x_622_;
v_isShared_630_ = v_isSharedCheck_634_;
goto v_resetjp_628_;
}
else
{
lean_inc(v_a_627_);
lean_dec(v___x_622_);
v___x_629_ = lean_box(0);
v_isShared_630_ = v_isSharedCheck_634_;
goto v_resetjp_628_;
}
v_resetjp_628_:
{
lean_object* v___x_632_; 
if (v_isShared_630_ == 0)
{
v___x_632_ = v___x_629_;
goto v_reusejp_631_;
}
else
{
lean_object* v_reuseFailAlloc_633_; 
v_reuseFailAlloc_633_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_633_, 0, v_a_627_);
v___x_632_ = v_reuseFailAlloc_633_;
goto v_reusejp_631_;
}
v_reusejp_631_:
{
return v___x_632_;
}
}
}
}
else
{
lean_object* v_a_635_; lean_object* v___x_637_; uint8_t v_isShared_638_; uint8_t v_isSharedCheck_642_; 
lean_dec(v___y_610_);
v_a_635_ = lean_ctor_get(v___x_616_, 0);
v_isSharedCheck_642_ = !lean_is_exclusive(v___x_616_);
if (v_isSharedCheck_642_ == 0)
{
v___x_637_ = v___x_616_;
v_isShared_638_ = v_isSharedCheck_642_;
goto v_resetjp_636_;
}
else
{
lean_inc(v_a_635_);
lean_dec(v___x_616_);
v___x_637_ = lean_box(0);
v_isShared_638_ = v_isSharedCheck_642_;
goto v_resetjp_636_;
}
v_resetjp_636_:
{
lean_object* v___x_640_; 
if (v_isShared_638_ == 0)
{
v___x_640_ = v___x_637_;
goto v_reusejp_639_;
}
else
{
lean_object* v_reuseFailAlloc_641_; 
v_reuseFailAlloc_641_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_641_, 0, v_a_635_);
v___x_640_ = v_reuseFailAlloc_641_;
goto v_reusejp_639_;
}
v_reusejp_639_:
{
return v___x_640_;
}
}
}
}
v___jp_643_:
{
lean_object* v___x_647_; lean_object* v___x_648_; lean_object* v_options_649_; lean_object* v_simpConfig_650_; lean_object* v_simpConfigSyntax_x3f_651_; lean_object* v___x_652_; lean_object* v___x_653_; 
v___x_647_ = lean_io_mono_nanos_now();
v___x_648_ = lean_st_ref_get(v_a_454_);
v_options_649_ = lean_ctor_get(v___y_645_, 3);
lean_inc_ref(v_options_649_);
v_simpConfig_650_ = lean_ctor_get(v___y_645_, 4);
lean_inc_ref(v_simpConfig_650_);
v_simpConfigSyntax_x3f_651_ = lean_ctor_get(v___y_645_, 5);
lean_inc(v_simpConfigSyntax_x3f_651_);
lean_dec_ref(v___y_645_);
v___x_652_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_652_, 0, v___y_644_);
v___x_653_ = lp_aesop_Aesop_search(v_goal_453_, v___x_652_, v_options_649_, v_simpConfig_650_, v_simpConfigSyntax_x3f_651_, v___x_648_, v_a_459_, v_a_460_, v_a_461_, v_a_462_);
if (lean_obj_tag(v___x_653_) == 0)
{
lean_object* v_a_654_; lean_object* v_fst_655_; lean_object* v_snd_656_; lean_object* v___x_657_; lean_object* v___x_658_; lean_object* v___x_659_; 
v_a_654_ = lean_ctor_get(v___x_653_, 0);
lean_inc(v_a_654_);
lean_dec_ref_known(v___x_653_, 1);
v_fst_655_ = lean_ctor_get(v_a_654_, 0);
lean_inc(v_fst_655_);
v_snd_656_ = lean_ctor_get(v_a_654_, 1);
lean_inc(v_snd_656_);
lean_dec(v_a_654_);
v___x_657_ = lean_array_get_size(v_fst_655_);
v___x_658_ = lean_array_to_list(v_fst_655_);
v___x_659_ = l_Lean_Elab_Tactic_replaceMainGoal___redArg(v___x_658_, v_a_456_, v_a_459_, v_a_460_, v_a_461_, v_a_462_);
if (lean_obj_tag(v___x_659_) == 0)
{
lean_object* v___x_660_; lean_object* v___x_661_; lean_object* v___x_662_; lean_object* v___x_663_; lean_object* v_total_664_; lean_object* v_configParsing_665_; lean_object* v_ruleSetConstruction_666_; lean_object* v_ruleSelection_667_; lean_object* v_script_668_; lean_object* v_forwardState_669_; lean_object* v_scriptGenerated_670_; lean_object* v_ruleStats_671_; lean_object* v_goalStats_672_; lean_object* v___x_674_; uint8_t v_isShared_675_; uint8_t v_isSharedCheck_683_; 
lean_dec_ref_known(v___x_659_, 1);
v___x_660_ = lean_st_ref_take(v_a_454_);
lean_dec(v___x_660_);
v___x_661_ = lean_st_ref_set(v_a_454_, v_snd_656_);
v___x_662_ = lean_io_mono_nanos_now();
v___x_663_ = lean_st_ref_take(v_a_454_);
v_total_664_ = lean_ctor_get(v___x_663_, 0);
v_configParsing_665_ = lean_ctor_get(v___x_663_, 1);
v_ruleSetConstruction_666_ = lean_ctor_get(v___x_663_, 2);
v_ruleSelection_667_ = lean_ctor_get(v___x_663_, 4);
v_script_668_ = lean_ctor_get(v___x_663_, 5);
v_forwardState_669_ = lean_ctor_get(v___x_663_, 6);
v_scriptGenerated_670_ = lean_ctor_get(v___x_663_, 7);
v_ruleStats_671_ = lean_ctor_get(v___x_663_, 8);
v_goalStats_672_ = lean_ctor_get(v___x_663_, 9);
v_isSharedCheck_683_ = !lean_is_exclusive(v___x_663_);
if (v_isSharedCheck_683_ == 0)
{
lean_object* v_unused_684_; 
v_unused_684_ = lean_ctor_get(v___x_663_, 3);
lean_dec(v_unused_684_);
v___x_674_ = v___x_663_;
v_isShared_675_ = v_isSharedCheck_683_;
goto v_resetjp_673_;
}
else
{
lean_inc(v_goalStats_672_);
lean_inc(v_ruleStats_671_);
lean_inc(v_scriptGenerated_670_);
lean_inc(v_forwardState_669_);
lean_inc(v_script_668_);
lean_inc(v_ruleSelection_667_);
lean_inc(v_ruleSetConstruction_666_);
lean_inc(v_configParsing_665_);
lean_inc(v_total_664_);
lean_dec(v___x_663_);
v___x_674_ = lean_box(0);
v_isShared_675_ = v_isSharedCheck_683_;
goto v_resetjp_673_;
}
v_resetjp_673_:
{
lean_object* v___x_676_; lean_object* v___x_678_; 
v___x_676_ = lean_nat_sub(v___x_662_, v___x_647_);
lean_dec(v___x_647_);
lean_dec(v___x_662_);
if (v_isShared_675_ == 0)
{
lean_ctor_set(v___x_674_, 3, v___x_676_);
v___x_678_ = v___x_674_;
goto v_reusejp_677_;
}
else
{
lean_object* v_reuseFailAlloc_682_; 
v_reuseFailAlloc_682_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_682_, 0, v_total_664_);
lean_ctor_set(v_reuseFailAlloc_682_, 1, v_configParsing_665_);
lean_ctor_set(v_reuseFailAlloc_682_, 2, v_ruleSetConstruction_666_);
lean_ctor_set(v_reuseFailAlloc_682_, 3, v___x_676_);
lean_ctor_set(v_reuseFailAlloc_682_, 4, v_ruleSelection_667_);
lean_ctor_set(v_reuseFailAlloc_682_, 5, v_script_668_);
lean_ctor_set(v_reuseFailAlloc_682_, 6, v_forwardState_669_);
lean_ctor_set(v_reuseFailAlloc_682_, 7, v_scriptGenerated_670_);
lean_ctor_set(v_reuseFailAlloc_682_, 8, v_ruleStats_671_);
lean_ctor_set(v_reuseFailAlloc_682_, 9, v_goalStats_672_);
v___x_678_ = v_reuseFailAlloc_682_;
goto v_reusejp_677_;
}
v_reusejp_677_:
{
lean_object* v___x_679_; lean_object* v___x_680_; uint8_t v___x_681_; 
v___x_679_ = lean_st_ref_set(v_a_454_, v___x_678_);
v___x_680_ = lean_unsigned_to_nat(0u);
v___x_681_ = lean_nat_dec_eq(v___x_657_, v___x_680_);
v___y_582_ = v___y_646_;
v_a_583_ = v___x_681_;
goto v___jp_581_;
}
}
}
else
{
lean_object* v_a_685_; lean_object* v___x_687_; uint8_t v_isShared_688_; uint8_t v_isSharedCheck_692_; 
lean_dec(v_snd_656_);
lean_dec(v___x_647_);
lean_dec(v___y_646_);
v_a_685_ = lean_ctor_get(v___x_659_, 0);
v_isSharedCheck_692_ = !lean_is_exclusive(v___x_659_);
if (v_isSharedCheck_692_ == 0)
{
v___x_687_ = v___x_659_;
v_isShared_688_ = v_isSharedCheck_692_;
goto v_resetjp_686_;
}
else
{
lean_inc(v_a_685_);
lean_dec(v___x_659_);
v___x_687_ = lean_box(0);
v_isShared_688_ = v_isSharedCheck_692_;
goto v_resetjp_686_;
}
v_resetjp_686_:
{
lean_object* v___x_690_; 
if (v_isShared_688_ == 0)
{
v___x_690_ = v___x_687_;
goto v_reusejp_689_;
}
else
{
lean_object* v_reuseFailAlloc_691_; 
v_reuseFailAlloc_691_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_691_, 0, v_a_685_);
v___x_690_ = v_reuseFailAlloc_691_;
goto v_reusejp_689_;
}
v_reusejp_689_:
{
return v___x_690_;
}
}
}
}
else
{
lean_object* v_a_693_; lean_object* v___x_695_; uint8_t v_isShared_696_; uint8_t v_isSharedCheck_700_; 
lean_dec(v___x_647_);
lean_dec(v___y_646_);
v_a_693_ = lean_ctor_get(v___x_653_, 0);
v_isSharedCheck_700_ = !lean_is_exclusive(v___x_653_);
if (v_isSharedCheck_700_ == 0)
{
v___x_695_ = v___x_653_;
v_isShared_696_ = v_isSharedCheck_700_;
goto v_resetjp_694_;
}
else
{
lean_inc(v_a_693_);
lean_dec(v___x_653_);
v___x_695_ = lean_box(0);
v_isShared_696_ = v_isSharedCheck_700_;
goto v_resetjp_694_;
}
v_resetjp_694_:
{
lean_object* v___x_698_; 
if (v_isShared_696_ == 0)
{
v___x_698_ = v___x_695_;
goto v_reusejp_697_;
}
else
{
lean_object* v_reuseFailAlloc_699_; 
v_reuseFailAlloc_699_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_699_, 0, v_a_693_);
v___x_698_ = v_reuseFailAlloc_699_;
goto v_reusejp_697_;
}
v_reusejp_697_:
{
return v___x_698_;
}
}
}
}
v___jp_701_:
{
lean_object* v_a_706_; uint8_t v___x_707_; 
v_a_706_ = lean_ctor_get(v___y_705_, 0);
lean_inc(v_a_706_);
lean_dec_ref(v___y_705_);
v___x_707_ = lean_unbox(v_a_706_);
lean_dec(v_a_706_);
if (v___x_707_ == 0)
{
v___y_608_ = v___y_702_;
v___y_609_ = v___y_703_;
v___y_610_ = v___y_704_;
goto v___jp_607_;
}
else
{
v___y_644_ = v___y_702_;
v___y_645_ = v___y_703_;
v___y_646_ = v___y_704_;
goto v___jp_643_;
}
}
v___jp_710_:
{
if (lean_obj_tag(v___y_714_) == 0)
{
lean_object* v___x_715_; uint8_t v___x_716_; 
lean_dec_ref_known(v___y_714_, 1);
v___x_715_ = lp_aesop_Aesop_aesop_collectStats;
v___x_716_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__0(v_options_708_, v___x_715_);
if (v___x_716_ == 0)
{
lean_object* v___x_717_; lean_object* v___x_718_; lean_object* v_a_719_; uint8_t v___x_720_; 
v___x_717_ = lp_aesop_Aesop_TraceOption_stats;
v___x_718_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__1___redArg(v___x_717_, v_a_461_);
v_a_719_ = lean_ctor_get(v___x_718_, 0);
lean_inc(v_a_719_);
v___x_720_ = lean_unbox(v_a_719_);
lean_dec(v_a_719_);
if (v___x_720_ == 0)
{
lean_object* v___x_721_; lean_object* v___x_722_; lean_object* v___x_723_; uint8_t v___x_724_; 
lean_dec_ref(v___x_718_);
v___x_721_ = lp_aesop_Aesop_aesop_stats_file;
v___x_722_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__2(v_options_708_, v___x_721_);
v___x_723_ = ((lean_object*)(lp_aesop___private_Aesop_Main_0__Aesop_evalAesop_go___closed__0));
v___x_724_ = lean_string_dec_eq(v___x_722_, v___x_723_);
lean_dec_ref(v___x_722_);
if (v___x_724_ == 0)
{
v___y_644_ = v___y_711_;
v___y_645_ = v___y_712_;
v___y_646_ = v___y_713_;
goto v___jp_643_;
}
else
{
v___y_608_ = v___y_711_;
v___y_609_ = v___y_712_;
v___y_610_ = v___y_713_;
goto v___jp_607_;
}
}
else
{
v___y_702_ = v___y_711_;
v___y_703_ = v___y_712_;
v___y_704_ = v___y_713_;
v___y_705_ = v___x_718_;
goto v___jp_701_;
}
}
else
{
v___y_644_ = v___y_711_;
v___y_645_ = v___y_712_;
v___y_646_ = v___y_713_;
goto v___jp_643_;
}
}
else
{
lean_object* v_a_725_; lean_object* v___x_727_; uint8_t v_isShared_728_; uint8_t v_isSharedCheck_732_; 
lean_dec(v___y_713_);
lean_dec_ref(v___y_712_);
lean_dec_ref(v___y_711_);
lean_dec(v_goal_453_);
v_a_725_ = lean_ctor_get(v___y_714_, 0);
v_isSharedCheck_732_ = !lean_is_exclusive(v___y_714_);
if (v_isSharedCheck_732_ == 0)
{
v___x_727_ = v___y_714_;
v_isShared_728_ = v_isSharedCheck_732_;
goto v_resetjp_726_;
}
else
{
lean_inc(v_a_725_);
lean_dec(v___y_714_);
v___x_727_ = lean_box(0);
v_isShared_728_ = v_isSharedCheck_732_;
goto v_resetjp_726_;
}
v_resetjp_726_:
{
lean_object* v___x_730_; 
if (v_isShared_728_ == 0)
{
v___x_730_ = v___x_727_;
goto v_reusejp_729_;
}
else
{
lean_object* v_reuseFailAlloc_731_; 
v_reuseFailAlloc_731_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_731_, 0, v_a_725_);
v___x_730_ = v_reuseFailAlloc_731_;
goto v_reusejp_729_;
}
v_reusejp_729_:
{
return v___x_730_;
}
}
}
}
v___jp_733_:
{
lean_object* v___x_746_; double v___x_747_; double v___x_748_; lean_object* v___x_749_; lean_object* v___x_750_; lean_object* v___x_751_; lean_object* v___x_752_; lean_object* v___x_753_; 
v___x_746_ = lean_io_get_num_heartbeats();
v___x_747_ = lean_float_of_nat(v___y_738_);
v___x_748_ = lean_float_of_nat(v___x_746_);
v___x_749_ = lean_box_float(v___x_747_);
v___x_750_ = lean_box_float(v___x_748_);
v___x_751_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_751_, 0, v___x_749_);
lean_ctor_set(v___x_751_, 1, v___x_750_);
v___x_752_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_752_, 0, v_a_745_);
lean_ctor_set(v___x_752_, 1, v___x_751_);
lean_inc_ref(v___y_744_);
lean_inc_ref(v___y_740_);
v___x_753_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__4(v___y_735_, v___y_737_, v___y_740_, v___y_736_, v___y_743_, v___y_742_, v___y_744_, v___x_752_, v_a_454_, v_a_455_, v_a_456_, v_a_457_, v_a_458_, v_a_459_, v_a_460_, v_a_461_, v_a_462_);
v___y_711_ = v___y_734_;
v___y_712_ = v___y_739_;
v___y_713_ = v___y_741_;
v___y_714_ = v___x_753_;
goto v___jp_710_;
}
v___jp_754_:
{
lean_object* v___x_767_; double v___x_768_; double v___x_769_; double v___x_770_; double v___x_771_; double v___x_772_; lean_object* v___x_773_; lean_object* v___x_774_; lean_object* v___x_775_; lean_object* v___x_776_; lean_object* v___x_777_; 
v___x_767_ = lean_io_mono_nanos_now();
v___x_768_ = lean_float_of_nat(v___y_759_);
v___x_769_ = lean_float_once(&lp_aesop___private_Aesop_Main_0__Aesop_evalAesop_go___closed__1, &lp_aesop___private_Aesop_Main_0__Aesop_evalAesop_go___closed__1_once, _init_lp_aesop___private_Aesop_Main_0__Aesop_evalAesop_go___closed__1);
v___x_770_ = lean_float_div(v___x_768_, v___x_769_);
v___x_771_ = lean_float_of_nat(v___x_767_);
v___x_772_ = lean_float_div(v___x_771_, v___x_769_);
v___x_773_ = lean_box_float(v___x_770_);
v___x_774_ = lean_box_float(v___x_772_);
v___x_775_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_775_, 0, v___x_773_);
lean_ctor_set(v___x_775_, 1, v___x_774_);
v___x_776_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_776_, 0, v_a_766_);
lean_ctor_set(v___x_776_, 1, v___x_775_);
lean_inc_ref(v___y_765_);
lean_inc_ref(v___y_761_);
v___x_777_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__4(v___y_756_, v___y_758_, v___y_761_, v___y_757_, v___y_764_, v___y_763_, v___y_765_, v___x_776_, v_a_454_, v_a_455_, v_a_456_, v_a_457_, v_a_458_, v_a_459_, v_a_460_, v_a_461_, v_a_462_);
v___y_711_ = v___y_755_;
v___y_712_ = v___y_760_;
v___y_713_ = v___y_762_;
v___y_714_ = v___x_777_;
goto v___jp_710_;
}
v___jp_778_:
{
lean_object* v___x_789_; lean_object* v_a_790_; lean_object* v___x_791_; uint8_t v___x_792_; 
v___x_789_ = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__3___redArg(v_a_462_);
v_a_790_ = lean_ctor_get(v___x_789_, 0);
lean_inc(v_a_790_);
lean_dec_ref(v___x_789_);
v___x_791_ = l_Lean_trace_profiler_useHeartbeats;
v___x_792_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__0(v___y_781_, v___x_791_);
if (v___x_792_ == 0)
{
lean_object* v___x_793_; lean_object* v___x_794_; 
v___x_793_ = lean_io_mono_nanos_now();
lean_inc_ref(v___y_779_);
v___x_794_ = lp_aesop_Aesop_LocalRuleSet_trace(v___y_779_, v___y_785_, v_a_461_, v_a_462_);
if (lean_obj_tag(v___x_794_) == 0)
{
lean_object* v_a_795_; lean_object* v___x_797_; uint8_t v_isShared_798_; uint8_t v_isSharedCheck_802_; 
v_a_795_ = lean_ctor_get(v___x_794_, 0);
v_isSharedCheck_802_ = !lean_is_exclusive(v___x_794_);
if (v_isSharedCheck_802_ == 0)
{
v___x_797_ = v___x_794_;
v_isShared_798_ = v_isSharedCheck_802_;
goto v_resetjp_796_;
}
else
{
lean_inc(v_a_795_);
lean_dec(v___x_794_);
v___x_797_ = lean_box(0);
v_isShared_798_ = v_isSharedCheck_802_;
goto v_resetjp_796_;
}
v_resetjp_796_:
{
lean_object* v___x_800_; 
if (v_isShared_798_ == 0)
{
lean_ctor_set_tag(v___x_797_, 1);
v___x_800_ = v___x_797_;
goto v_reusejp_799_;
}
else
{
lean_object* v_reuseFailAlloc_801_; 
v_reuseFailAlloc_801_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_801_, 0, v_a_795_);
v___x_800_ = v_reuseFailAlloc_801_;
goto v_reusejp_799_;
}
v_reusejp_799_:
{
v___y_755_ = v___y_779_;
v___y_756_ = v___y_780_;
v___y_757_ = v___y_781_;
v___y_758_ = v___y_782_;
v___y_759_ = v___x_793_;
v___y_760_ = v___y_783_;
v___y_761_ = v___y_784_;
v___y_762_ = v___y_786_;
v___y_763_ = v_a_790_;
v___y_764_ = v___y_787_;
v___y_765_ = v___y_788_;
v_a_766_ = v___x_800_;
goto v___jp_754_;
}
}
}
else
{
lean_object* v_a_803_; lean_object* v___x_805_; uint8_t v_isShared_806_; uint8_t v_isSharedCheck_810_; 
v_a_803_ = lean_ctor_get(v___x_794_, 0);
v_isSharedCheck_810_ = !lean_is_exclusive(v___x_794_);
if (v_isSharedCheck_810_ == 0)
{
v___x_805_ = v___x_794_;
v_isShared_806_ = v_isSharedCheck_810_;
goto v_resetjp_804_;
}
else
{
lean_inc(v_a_803_);
lean_dec(v___x_794_);
v___x_805_ = lean_box(0);
v_isShared_806_ = v_isSharedCheck_810_;
goto v_resetjp_804_;
}
v_resetjp_804_:
{
lean_object* v___x_808_; 
if (v_isShared_806_ == 0)
{
lean_ctor_set_tag(v___x_805_, 0);
v___x_808_ = v___x_805_;
goto v_reusejp_807_;
}
else
{
lean_object* v_reuseFailAlloc_809_; 
v_reuseFailAlloc_809_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_809_, 0, v_a_803_);
v___x_808_ = v_reuseFailAlloc_809_;
goto v_reusejp_807_;
}
v_reusejp_807_:
{
v___y_755_ = v___y_779_;
v___y_756_ = v___y_780_;
v___y_757_ = v___y_781_;
v___y_758_ = v___y_782_;
v___y_759_ = v___x_793_;
v___y_760_ = v___y_783_;
v___y_761_ = v___y_784_;
v___y_762_ = v___y_786_;
v___y_763_ = v_a_790_;
v___y_764_ = v___y_787_;
v___y_765_ = v___y_788_;
v_a_766_ = v___x_808_;
goto v___jp_754_;
}
}
}
}
else
{
lean_object* v___x_811_; lean_object* v___x_812_; 
v___x_811_ = lean_io_get_num_heartbeats();
lean_inc_ref(v___y_779_);
v___x_812_ = lp_aesop_Aesop_LocalRuleSet_trace(v___y_779_, v___y_785_, v_a_461_, v_a_462_);
if (lean_obj_tag(v___x_812_) == 0)
{
lean_object* v_a_813_; lean_object* v___x_815_; uint8_t v_isShared_816_; uint8_t v_isSharedCheck_820_; 
v_a_813_ = lean_ctor_get(v___x_812_, 0);
v_isSharedCheck_820_ = !lean_is_exclusive(v___x_812_);
if (v_isSharedCheck_820_ == 0)
{
v___x_815_ = v___x_812_;
v_isShared_816_ = v_isSharedCheck_820_;
goto v_resetjp_814_;
}
else
{
lean_inc(v_a_813_);
lean_dec(v___x_812_);
v___x_815_ = lean_box(0);
v_isShared_816_ = v_isSharedCheck_820_;
goto v_resetjp_814_;
}
v_resetjp_814_:
{
lean_object* v___x_818_; 
if (v_isShared_816_ == 0)
{
lean_ctor_set_tag(v___x_815_, 1);
v___x_818_ = v___x_815_;
goto v_reusejp_817_;
}
else
{
lean_object* v_reuseFailAlloc_819_; 
v_reuseFailAlloc_819_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_819_, 0, v_a_813_);
v___x_818_ = v_reuseFailAlloc_819_;
goto v_reusejp_817_;
}
v_reusejp_817_:
{
v___y_734_ = v___y_779_;
v___y_735_ = v___y_780_;
v___y_736_ = v___y_781_;
v___y_737_ = v___y_782_;
v___y_738_ = v___x_811_;
v___y_739_ = v___y_783_;
v___y_740_ = v___y_784_;
v___y_741_ = v___y_786_;
v___y_742_ = v_a_790_;
v___y_743_ = v___y_787_;
v___y_744_ = v___y_788_;
v_a_745_ = v___x_818_;
goto v___jp_733_;
}
}
}
else
{
lean_object* v_a_821_; lean_object* v___x_823_; uint8_t v_isShared_824_; uint8_t v_isSharedCheck_828_; 
v_a_821_ = lean_ctor_get(v___x_812_, 0);
v_isSharedCheck_828_ = !lean_is_exclusive(v___x_812_);
if (v_isSharedCheck_828_ == 0)
{
v___x_823_ = v___x_812_;
v_isShared_824_ = v_isSharedCheck_828_;
goto v_resetjp_822_;
}
else
{
lean_inc(v_a_821_);
lean_dec(v___x_812_);
v___x_823_ = lean_box(0);
v_isShared_824_ = v_isSharedCheck_828_;
goto v_resetjp_822_;
}
v_resetjp_822_:
{
lean_object* v___x_826_; 
if (v_isShared_824_ == 0)
{
lean_ctor_set_tag(v___x_823_, 0);
v___x_826_ = v___x_823_;
goto v_reusejp_825_;
}
else
{
lean_object* v_reuseFailAlloc_827_; 
v_reuseFailAlloc_827_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_827_, 0, v_a_821_);
v___x_826_ = v_reuseFailAlloc_827_;
goto v_reusejp_825_;
}
v_reusejp_825_:
{
v___y_734_ = v___y_779_;
v___y_735_ = v___y_780_;
v___y_736_ = v___y_781_;
v___y_737_ = v___y_782_;
v___y_738_ = v___x_811_;
v___y_739_ = v___y_783_;
v___y_740_ = v___y_784_;
v___y_741_ = v___y_786_;
v___y_742_ = v_a_790_;
v___y_743_ = v___y_787_;
v___y_744_ = v___y_788_;
v_a_745_ = v___x_826_;
goto v___jp_733_;
}
}
}
}
}
v___jp_829_:
{
uint8_t v_hasTrace_833_; lean_object* v___x_834_; 
v_hasTrace_833_ = lean_ctor_get_uint8(v_options_708_, sizeof(void*)*1);
v___x_834_ = lp_aesop_Aesop_TraceOption_ruleSet;
if (v_hasTrace_833_ == 0)
{
lean_object* v___x_835_; 
lean_inc_ref(v_a_832_);
v___x_835_ = lp_aesop_Aesop_LocalRuleSet_trace(v_a_832_, v___x_834_, v_a_461_, v_a_462_);
v___y_711_ = v_a_832_;
v___y_712_ = v___y_830_;
v___y_713_ = v___y_831_;
v___y_714_ = v___x_835_;
goto v___jp_710_;
}
else
{
lean_object* v_traceClass_836_; lean_object* v___f_837_; lean_object* v___x_838_; lean_object* v___x_839_; lean_object* v___x_840_; uint8_t v___x_841_; 
v_traceClass_836_ = lean_ctor_get(v___x_834_, 0);
v___f_837_ = lean_obj_once(&lp_aesop___private_Aesop_Main_0__Aesop_evalAesop_go___closed__5, &lp_aesop___private_Aesop_Main_0__Aesop_evalAesop_go___closed__5_once, _init_lp_aesop___private_Aesop_Main_0__Aesop_evalAesop_go___closed__5);
v___x_838_ = ((lean_object*)(lp_aesop___private_Aesop_Main_0__Aesop_evalAesop_go___closed__0));
v___x_839_ = ((lean_object*)(lp_aesop___private_Aesop_Main_0__Aesop_evalAesop_go___closed__7));
lean_inc(v_traceClass_836_);
v___x_840_ = l_Lean_Name_append(v___x_839_, v_traceClass_836_);
v___x_841_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_709_, v_options_708_, v___x_840_);
lean_dec(v___x_840_);
if (v___x_841_ == 0)
{
lean_object* v___x_842_; uint8_t v___x_843_; 
v___x_842_ = l_Lean_trace_profiler;
v___x_843_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__0(v_options_708_, v___x_842_);
if (v___x_843_ == 0)
{
lean_object* v___x_844_; 
lean_inc_ref(v_a_832_);
v___x_844_ = lp_aesop_Aesop_LocalRuleSet_trace(v_a_832_, v___x_834_, v_a_461_, v_a_462_);
v___y_711_ = v_a_832_;
v___y_712_ = v___y_830_;
v___y_713_ = v___y_831_;
v___y_714_ = v___x_844_;
goto v___jp_710_;
}
else
{
lean_inc(v_traceClass_836_);
v___y_779_ = v_a_832_;
v___y_780_ = v_traceClass_836_;
v___y_781_ = v_options_708_;
v___y_782_ = v_hasTrace_833_;
v___y_783_ = v___y_830_;
v___y_784_ = v___x_838_;
v___y_785_ = v___x_834_;
v___y_786_ = v___y_831_;
v___y_787_ = v___x_841_;
v___y_788_ = v___f_837_;
goto v___jp_778_;
}
}
else
{
lean_inc(v_traceClass_836_);
v___y_779_ = v_a_832_;
v___y_780_ = v_traceClass_836_;
v___y_781_ = v_options_708_;
v___y_782_ = v_hasTrace_833_;
v___y_783_ = v___y_830_;
v___y_784_ = v___x_838_;
v___y_785_ = v___x_834_;
v___y_786_ = v___y_831_;
v___y_787_ = v___x_841_;
v___y_788_ = v___f_837_;
goto v___jp_778_;
}
}
}
v___jp_845_:
{
lean_object* v___x_848_; 
lean_inc_ref(v___y_846_);
lean_inc(v_goal_453_);
v___x_848_ = lp_aesop_Aesop_Frontend_TacticConfig_getRuleSet(v_goal_453_, v___y_846_, v_a_457_, v_a_458_, v_a_459_, v_a_460_, v_a_461_, v_a_462_);
if (lean_obj_tag(v___x_848_) == 0)
{
lean_object* v_a_849_; 
v_a_849_ = lean_ctor_get(v___x_848_, 0);
lean_inc(v_a_849_);
lean_dec_ref_known(v___x_848_, 1);
v___y_830_ = v___y_846_;
v___y_831_ = v___y_847_;
v_a_832_ = v_a_849_;
goto v___jp_829_;
}
else
{
lean_object* v_a_850_; lean_object* v___x_852_; uint8_t v_isShared_853_; uint8_t v_isSharedCheck_857_; 
lean_dec(v___y_847_);
lean_dec_ref(v___y_846_);
lean_dec(v_goal_453_);
v_a_850_ = lean_ctor_get(v___x_848_, 0);
v_isSharedCheck_857_ = !lean_is_exclusive(v___x_848_);
if (v_isSharedCheck_857_ == 0)
{
v___x_852_ = v___x_848_;
v_isShared_853_ = v_isSharedCheck_857_;
goto v_resetjp_851_;
}
else
{
lean_inc(v_a_850_);
lean_dec(v___x_848_);
v___x_852_ = lean_box(0);
v_isShared_853_ = v_isSharedCheck_857_;
goto v_resetjp_851_;
}
v_resetjp_851_:
{
lean_object* v___x_855_; 
if (v_isShared_853_ == 0)
{
v___x_855_ = v___x_852_;
goto v_reusejp_854_;
}
else
{
lean_object* v_reuseFailAlloc_856_; 
v_reuseFailAlloc_856_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_856_, 0, v_a_850_);
v___x_855_ = v_reuseFailAlloc_856_;
goto v_reusejp_854_;
}
v_reusejp_854_:
{
return v___x_855_;
}
}
}
}
v___jp_858_:
{
lean_object* v___x_861_; lean_object* v___x_862_; 
v___x_861_ = lean_io_mono_nanos_now();
lean_inc_ref(v___y_859_);
lean_inc(v_goal_453_);
v___x_862_ = lp_aesop_Aesop_Frontend_TacticConfig_getRuleSet(v_goal_453_, v___y_859_, v_a_457_, v_a_458_, v_a_459_, v_a_460_, v_a_461_, v_a_462_);
if (lean_obj_tag(v___x_862_) == 0)
{
lean_object* v_a_863_; lean_object* v___x_864_; lean_object* v___x_865_; lean_object* v_total_866_; lean_object* v_configParsing_867_; lean_object* v_search_868_; lean_object* v_ruleSelection_869_; lean_object* v_script_870_; lean_object* v_forwardState_871_; lean_object* v_scriptGenerated_872_; lean_object* v_ruleStats_873_; lean_object* v_goalStats_874_; lean_object* v___x_876_; uint8_t v_isShared_877_; uint8_t v_isSharedCheck_883_; 
v_a_863_ = lean_ctor_get(v___x_862_, 0);
lean_inc(v_a_863_);
lean_dec_ref_known(v___x_862_, 1);
v___x_864_ = lean_io_mono_nanos_now();
v___x_865_ = lean_st_ref_take(v_a_454_);
v_total_866_ = lean_ctor_get(v___x_865_, 0);
v_configParsing_867_ = lean_ctor_get(v___x_865_, 1);
v_search_868_ = lean_ctor_get(v___x_865_, 3);
v_ruleSelection_869_ = lean_ctor_get(v___x_865_, 4);
v_script_870_ = lean_ctor_get(v___x_865_, 5);
v_forwardState_871_ = lean_ctor_get(v___x_865_, 6);
v_scriptGenerated_872_ = lean_ctor_get(v___x_865_, 7);
v_ruleStats_873_ = lean_ctor_get(v___x_865_, 8);
v_goalStats_874_ = lean_ctor_get(v___x_865_, 9);
v_isSharedCheck_883_ = !lean_is_exclusive(v___x_865_);
if (v_isSharedCheck_883_ == 0)
{
lean_object* v_unused_884_; 
v_unused_884_ = lean_ctor_get(v___x_865_, 2);
lean_dec(v_unused_884_);
v___x_876_ = v___x_865_;
v_isShared_877_ = v_isSharedCheck_883_;
goto v_resetjp_875_;
}
else
{
lean_inc(v_goalStats_874_);
lean_inc(v_ruleStats_873_);
lean_inc(v_scriptGenerated_872_);
lean_inc(v_forwardState_871_);
lean_inc(v_script_870_);
lean_inc(v_ruleSelection_869_);
lean_inc(v_search_868_);
lean_inc(v_configParsing_867_);
lean_inc(v_total_866_);
lean_dec(v___x_865_);
v___x_876_ = lean_box(0);
v_isShared_877_ = v_isSharedCheck_883_;
goto v_resetjp_875_;
}
v_resetjp_875_:
{
lean_object* v___x_878_; lean_object* v___x_880_; 
v___x_878_ = lean_nat_sub(v___x_864_, v___x_861_);
lean_dec(v___x_861_);
lean_dec(v___x_864_);
if (v_isShared_877_ == 0)
{
lean_ctor_set(v___x_876_, 2, v___x_878_);
v___x_880_ = v___x_876_;
goto v_reusejp_879_;
}
else
{
lean_object* v_reuseFailAlloc_882_; 
v_reuseFailAlloc_882_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_882_, 0, v_total_866_);
lean_ctor_set(v_reuseFailAlloc_882_, 1, v_configParsing_867_);
lean_ctor_set(v_reuseFailAlloc_882_, 2, v___x_878_);
lean_ctor_set(v_reuseFailAlloc_882_, 3, v_search_868_);
lean_ctor_set(v_reuseFailAlloc_882_, 4, v_ruleSelection_869_);
lean_ctor_set(v_reuseFailAlloc_882_, 5, v_script_870_);
lean_ctor_set(v_reuseFailAlloc_882_, 6, v_forwardState_871_);
lean_ctor_set(v_reuseFailAlloc_882_, 7, v_scriptGenerated_872_);
lean_ctor_set(v_reuseFailAlloc_882_, 8, v_ruleStats_873_);
lean_ctor_set(v_reuseFailAlloc_882_, 9, v_goalStats_874_);
v___x_880_ = v_reuseFailAlloc_882_;
goto v_reusejp_879_;
}
v_reusejp_879_:
{
lean_object* v___x_881_; 
v___x_881_ = lean_st_ref_set(v_a_454_, v___x_880_);
v___y_830_ = v___y_859_;
v___y_831_ = v___y_860_;
v_a_832_ = v_a_863_;
goto v___jp_829_;
}
}
}
else
{
lean_object* v_a_885_; lean_object* v___x_887_; uint8_t v_isShared_888_; uint8_t v_isSharedCheck_892_; 
lean_dec(v___x_861_);
lean_dec(v___y_860_);
lean_dec_ref(v___y_859_);
lean_dec(v_goal_453_);
v_a_885_ = lean_ctor_get(v___x_862_, 0);
v_isSharedCheck_892_ = !lean_is_exclusive(v___x_862_);
if (v_isSharedCheck_892_ == 0)
{
v___x_887_ = v___x_862_;
v_isShared_888_ = v_isSharedCheck_892_;
goto v_resetjp_886_;
}
else
{
lean_inc(v_a_885_);
lean_dec(v___x_862_);
v___x_887_ = lean_box(0);
v_isShared_888_ = v_isSharedCheck_892_;
goto v_resetjp_886_;
}
v_resetjp_886_:
{
lean_object* v___x_890_; 
if (v_isShared_888_ == 0)
{
v___x_890_ = v___x_887_;
goto v_reusejp_889_;
}
else
{
lean_object* v_reuseFailAlloc_891_; 
v_reuseFailAlloc_891_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_891_, 0, v_a_885_);
v___x_890_ = v_reuseFailAlloc_891_;
goto v_reusejp_889_;
}
v_reusejp_889_:
{
return v___x_890_;
}
}
}
}
v___jp_893_:
{
lean_object* v_a_897_; uint8_t v___x_898_; 
v_a_897_ = lean_ctor_get(v___y_896_, 0);
lean_inc(v_a_897_);
lean_dec_ref(v___y_896_);
v___x_898_ = lean_unbox(v_a_897_);
lean_dec(v_a_897_);
if (v___x_898_ == 0)
{
v___y_846_ = v___y_894_;
v___y_847_ = v___y_895_;
goto v___jp_845_;
}
else
{
v___y_859_ = v___y_894_;
v___y_860_ = v___y_895_;
goto v___jp_858_;
}
}
v___jp_899_:
{
lean_object* v___x_902_; uint8_t v___x_903_; 
v___x_902_ = lp_aesop_Aesop_aesop_collectStats;
v___x_903_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__0(v_options_708_, v___x_902_);
if (v___x_903_ == 0)
{
lean_object* v___x_904_; lean_object* v___x_905_; lean_object* v_a_906_; uint8_t v___x_907_; 
v___x_904_ = lp_aesop_Aesop_TraceOption_stats;
v___x_905_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__1___redArg(v___x_904_, v_a_461_);
v_a_906_ = lean_ctor_get(v___x_905_, 0);
lean_inc(v_a_906_);
v___x_907_ = lean_unbox(v_a_906_);
lean_dec(v_a_906_);
if (v___x_907_ == 0)
{
lean_object* v___x_908_; lean_object* v___x_909_; lean_object* v___x_910_; uint8_t v___x_911_; 
lean_dec_ref(v___x_905_);
v___x_908_ = lp_aesop_Aesop_aesop_stats_file;
v___x_909_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__2(v_options_708_, v___x_908_);
v___x_910_ = ((lean_object*)(lp_aesop___private_Aesop_Main_0__Aesop_evalAesop_go___closed__0));
v___x_911_ = lean_string_dec_eq(v___x_909_, v___x_910_);
lean_dec_ref(v___x_909_);
if (v___x_911_ == 0)
{
v___y_859_ = v_a_901_;
v___y_860_ = v___y_900_;
goto v___jp_858_;
}
else
{
v___y_846_ = v_a_901_;
v___y_847_ = v___y_900_;
goto v___jp_845_;
}
}
else
{
v___y_894_ = v_a_901_;
v___y_895_ = v___y_900_;
v___y_896_ = v___x_905_;
goto v___jp_893_;
}
}
else
{
v___y_859_ = v_a_901_;
v___y_860_ = v___y_900_;
goto v___jp_858_;
}
}
v___jp_912_:
{
if (lean_obj_tag(v___y_914_) == 0)
{
lean_object* v_a_915_; 
v_a_915_ = lean_ctor_get(v___y_914_, 0);
lean_inc(v_a_915_);
lean_dec_ref_known(v___y_914_, 1);
v___y_900_ = v___y_913_;
v_a_901_ = v_a_915_;
goto v___jp_899_;
}
else
{
lean_object* v_a_916_; lean_object* v___x_918_; uint8_t v_isShared_919_; uint8_t v_isSharedCheck_923_; 
lean_dec(v___y_913_);
lean_dec(v_goal_453_);
v_a_916_ = lean_ctor_get(v___y_914_, 0);
v_isSharedCheck_923_ = !lean_is_exclusive(v___y_914_);
if (v_isSharedCheck_923_ == 0)
{
v___x_918_ = v___y_914_;
v_isShared_919_ = v_isSharedCheck_923_;
goto v_resetjp_917_;
}
else
{
lean_inc(v_a_916_);
lean_dec(v___y_914_);
v___x_918_ = lean_box(0);
v_isShared_919_ = v_isSharedCheck_923_;
goto v_resetjp_917_;
}
v_resetjp_917_:
{
lean_object* v___x_921_; 
if (v_isShared_919_ == 0)
{
v___x_921_ = v___x_918_;
goto v_reusejp_920_;
}
else
{
lean_object* v_reuseFailAlloc_922_; 
v_reuseFailAlloc_922_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_922_, 0, v_a_916_);
v___x_921_ = v_reuseFailAlloc_922_;
goto v_reusejp_920_;
}
v_reusejp_920_:
{
return v___x_921_;
}
}
}
}
v___jp_924_:
{
lean_object* v___x_926_; 
lean_inc(v_goal_453_);
v___x_926_ = lp_aesop_Aesop_Frontend_TacticConfig_parse(v_stx_452_, v_goal_453_, v_a_457_, v_a_458_, v_a_459_, v_a_460_, v_a_461_, v_a_462_);
v___y_913_ = v___y_925_;
v___y_914_ = v___x_926_;
goto v___jp_912_;
}
v___jp_927_:
{
if (lean_obj_tag(v___y_930_) == 0)
{
lean_object* v___x_931_; uint8_t v___x_932_; 
lean_dec_ref_known(v___y_930_, 1);
v___x_931_ = lp_aesop_Aesop_aesop_collectStats;
v___x_932_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__0(v_options_708_, v___x_931_);
if (v___x_932_ == 0)
{
lean_object* v___x_933_; lean_object* v___x_934_; lean_object* v_a_935_; uint8_t v___x_936_; 
v___x_933_ = lp_aesop_Aesop_TraceOption_stats;
v___x_934_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__1___redArg(v___x_933_, v_a_461_);
v_a_935_ = lean_ctor_get(v___x_934_, 0);
lean_inc(v_a_935_);
v___x_936_ = lean_unbox(v_a_935_);
lean_dec(v_a_935_);
if (v___x_936_ == 0)
{
lean_object* v___x_937_; lean_object* v___x_938_; lean_object* v___x_939_; uint8_t v___x_940_; 
v___x_937_ = lp_aesop_Aesop_aesop_stats_file;
v___x_938_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__2(v_options_708_, v___x_937_);
v___x_939_ = ((lean_object*)(lp_aesop___private_Aesop_Main_0__Aesop_evalAesop_go___closed__0));
v___x_940_ = lean_string_dec_eq(v___x_938_, v___x_939_);
lean_dec_ref(v___x_938_);
if (v___x_940_ == 0)
{
lean_dec_ref(v___x_934_);
v___y_465_ = v___y_928_;
v___y_466_ = v___y_929_;
goto v___jp_464_;
}
else
{
v___y_576_ = v___y_928_;
v___y_577_ = v___y_929_;
v___y_578_ = v___x_934_;
goto v___jp_575_;
}
}
else
{
v___y_576_ = v___y_928_;
v___y_577_ = v___y_929_;
v___y_578_ = v___x_934_;
goto v___jp_575_;
}
}
else
{
v___y_531_ = v___y_928_;
v___y_532_ = v___y_929_;
v_a_533_ = v___x_932_;
goto v___jp_530_;
}
}
else
{
lean_object* v_a_941_; lean_object* v___x_943_; uint8_t v_isShared_944_; uint8_t v_isSharedCheck_948_; 
lean_dec_ref(v___y_929_);
lean_dec_ref(v___y_928_);
lean_dec(v_goal_453_);
v_a_941_ = lean_ctor_get(v___y_930_, 0);
v_isSharedCheck_948_ = !lean_is_exclusive(v___y_930_);
if (v_isSharedCheck_948_ == 0)
{
v___x_943_ = v___y_930_;
v_isShared_944_ = v_isSharedCheck_948_;
goto v_resetjp_942_;
}
else
{
lean_inc(v_a_941_);
lean_dec(v___y_930_);
v___x_943_ = lean_box(0);
v_isShared_944_ = v_isSharedCheck_948_;
goto v_resetjp_942_;
}
v_resetjp_942_:
{
lean_object* v___x_946_; 
if (v_isShared_944_ == 0)
{
v___x_946_ = v___x_943_;
goto v_reusejp_945_;
}
else
{
lean_object* v_reuseFailAlloc_947_; 
v_reuseFailAlloc_947_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_947_, 0, v_a_941_);
v___x_946_ = v_reuseFailAlloc_947_;
goto v_reusejp_945_;
}
v_reusejp_945_:
{
return v___x_946_;
}
}
}
}
v___jp_949_:
{
lean_object* v___x_961_; double v___x_962_; double v___x_963_; lean_object* v___x_964_; lean_object* v___x_965_; lean_object* v___x_966_; lean_object* v___x_967_; lean_object* v___x_968_; 
v___x_961_ = lean_io_get_num_heartbeats();
v___x_962_ = lean_float_of_nat(v___y_959_);
v___x_963_ = lean_float_of_nat(v___x_961_);
v___x_964_ = lean_box_float(v___x_962_);
v___x_965_ = lean_box_float(v___x_963_);
v___x_966_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_966_, 0, v___x_964_);
lean_ctor_set(v___x_966_, 1, v___x_965_);
v___x_967_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_967_, 0, v_a_960_);
lean_ctor_set(v___x_967_, 1, v___x_966_);
lean_inc_ref(v___y_950_);
lean_inc_ref(v___y_957_);
v___x_968_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__4(v___y_953_, v___y_952_, v___y_957_, v___y_956_, v___y_954_, v___y_955_, v___y_950_, v___x_967_, v_a_454_, v_a_455_, v_a_456_, v_a_457_, v_a_458_, v_a_459_, v_a_460_, v_a_461_, v_a_462_);
v___y_928_ = v___y_951_;
v___y_929_ = v___y_958_;
v___y_930_ = v___x_968_;
goto v___jp_927_;
}
v___jp_969_:
{
lean_object* v___x_981_; double v___x_982_; double v___x_983_; double v___x_984_; double v___x_985_; double v___x_986_; lean_object* v___x_987_; lean_object* v___x_988_; lean_object* v___x_989_; lean_object* v___x_990_; lean_object* v___x_991_; 
v___x_981_ = lean_io_mono_nanos_now();
v___x_982_ = lean_float_of_nat(v___y_971_);
v___x_983_ = lean_float_once(&lp_aesop___private_Aesop_Main_0__Aesop_evalAesop_go___closed__1, &lp_aesop___private_Aesop_Main_0__Aesop_evalAesop_go___closed__1_once, _init_lp_aesop___private_Aesop_Main_0__Aesop_evalAesop_go___closed__1);
v___x_984_ = lean_float_div(v___x_982_, v___x_983_);
v___x_985_ = lean_float_of_nat(v___x_981_);
v___x_986_ = lean_float_div(v___x_985_, v___x_983_);
v___x_987_ = lean_box_float(v___x_984_);
v___x_988_ = lean_box_float(v___x_986_);
v___x_989_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_989_, 0, v___x_987_);
lean_ctor_set(v___x_989_, 1, v___x_988_);
v___x_990_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_990_, 0, v_a_980_);
lean_ctor_set(v___x_990_, 1, v___x_989_);
lean_inc_ref(v___y_970_);
lean_inc_ref(v___y_978_);
v___x_991_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__4(v___y_974_, v___y_973_, v___y_978_, v___y_977_, v___y_975_, v___y_976_, v___y_970_, v___x_990_, v_a_454_, v_a_455_, v_a_456_, v_a_457_, v_a_458_, v_a_459_, v_a_460_, v_a_461_, v_a_462_);
v___y_928_ = v___y_972_;
v___y_929_ = v___y_979_;
v___y_930_ = v___x_991_;
goto v___jp_927_;
}
v___jp_992_:
{
lean_object* v___x_1002_; lean_object* v_a_1003_; lean_object* v___x_1004_; uint8_t v___x_1005_; 
v___x_1002_ = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__3___redArg(v_a_462_);
v_a_1003_ = lean_ctor_get(v___x_1002_, 0);
lean_inc(v_a_1003_);
lean_dec_ref(v___x_1002_);
v___x_1004_ = l_Lean_trace_profiler_useHeartbeats;
v___x_1005_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__0(v___y_998_, v___x_1004_);
if (v___x_1005_ == 0)
{
lean_object* v___x_1006_; lean_object* v___x_1007_; 
v___x_1006_ = lean_io_mono_nanos_now();
lean_inc_ref(v___y_994_);
v___x_1007_ = lp_aesop_Aesop_LocalRuleSet_trace(v___y_994_, v___y_1001_, v_a_461_, v_a_462_);
if (lean_obj_tag(v___x_1007_) == 0)
{
lean_object* v_a_1008_; lean_object* v___x_1010_; uint8_t v_isShared_1011_; uint8_t v_isSharedCheck_1015_; 
v_a_1008_ = lean_ctor_get(v___x_1007_, 0);
v_isSharedCheck_1015_ = !lean_is_exclusive(v___x_1007_);
if (v_isSharedCheck_1015_ == 0)
{
v___x_1010_ = v___x_1007_;
v_isShared_1011_ = v_isSharedCheck_1015_;
goto v_resetjp_1009_;
}
else
{
lean_inc(v_a_1008_);
lean_dec(v___x_1007_);
v___x_1010_ = lean_box(0);
v_isShared_1011_ = v_isSharedCheck_1015_;
goto v_resetjp_1009_;
}
v_resetjp_1009_:
{
lean_object* v___x_1013_; 
if (v_isShared_1011_ == 0)
{
lean_ctor_set_tag(v___x_1010_, 1);
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
v___y_970_ = v___y_993_;
v___y_971_ = v___x_1006_;
v___y_972_ = v___y_994_;
v___y_973_ = v___y_995_;
v___y_974_ = v___y_996_;
v___y_975_ = v___y_997_;
v___y_976_ = v_a_1003_;
v___y_977_ = v___y_998_;
v___y_978_ = v___y_999_;
v___y_979_ = v___y_1000_;
v_a_980_ = v___x_1013_;
goto v___jp_969_;
}
}
}
else
{
lean_object* v_a_1016_; lean_object* v___x_1018_; uint8_t v_isShared_1019_; uint8_t v_isSharedCheck_1023_; 
v_a_1016_ = lean_ctor_get(v___x_1007_, 0);
v_isSharedCheck_1023_ = !lean_is_exclusive(v___x_1007_);
if (v_isSharedCheck_1023_ == 0)
{
v___x_1018_ = v___x_1007_;
v_isShared_1019_ = v_isSharedCheck_1023_;
goto v_resetjp_1017_;
}
else
{
lean_inc(v_a_1016_);
lean_dec(v___x_1007_);
v___x_1018_ = lean_box(0);
v_isShared_1019_ = v_isSharedCheck_1023_;
goto v_resetjp_1017_;
}
v_resetjp_1017_:
{
lean_object* v___x_1021_; 
if (v_isShared_1019_ == 0)
{
lean_ctor_set_tag(v___x_1018_, 0);
v___x_1021_ = v___x_1018_;
goto v_reusejp_1020_;
}
else
{
lean_object* v_reuseFailAlloc_1022_; 
v_reuseFailAlloc_1022_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1022_, 0, v_a_1016_);
v___x_1021_ = v_reuseFailAlloc_1022_;
goto v_reusejp_1020_;
}
v_reusejp_1020_:
{
v___y_970_ = v___y_993_;
v___y_971_ = v___x_1006_;
v___y_972_ = v___y_994_;
v___y_973_ = v___y_995_;
v___y_974_ = v___y_996_;
v___y_975_ = v___y_997_;
v___y_976_ = v_a_1003_;
v___y_977_ = v___y_998_;
v___y_978_ = v___y_999_;
v___y_979_ = v___y_1000_;
v_a_980_ = v___x_1021_;
goto v___jp_969_;
}
}
}
}
else
{
lean_object* v___x_1024_; lean_object* v___x_1025_; 
v___x_1024_ = lean_io_get_num_heartbeats();
lean_inc_ref(v___y_994_);
v___x_1025_ = lp_aesop_Aesop_LocalRuleSet_trace(v___y_994_, v___y_1001_, v_a_461_, v_a_462_);
if (lean_obj_tag(v___x_1025_) == 0)
{
lean_object* v_a_1026_; lean_object* v___x_1028_; uint8_t v_isShared_1029_; uint8_t v_isSharedCheck_1033_; 
v_a_1026_ = lean_ctor_get(v___x_1025_, 0);
v_isSharedCheck_1033_ = !lean_is_exclusive(v___x_1025_);
if (v_isSharedCheck_1033_ == 0)
{
v___x_1028_ = v___x_1025_;
v_isShared_1029_ = v_isSharedCheck_1033_;
goto v_resetjp_1027_;
}
else
{
lean_inc(v_a_1026_);
lean_dec(v___x_1025_);
v___x_1028_ = lean_box(0);
v_isShared_1029_ = v_isSharedCheck_1033_;
goto v_resetjp_1027_;
}
v_resetjp_1027_:
{
lean_object* v___x_1031_; 
if (v_isShared_1029_ == 0)
{
lean_ctor_set_tag(v___x_1028_, 1);
v___x_1031_ = v___x_1028_;
goto v_reusejp_1030_;
}
else
{
lean_object* v_reuseFailAlloc_1032_; 
v_reuseFailAlloc_1032_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1032_, 0, v_a_1026_);
v___x_1031_ = v_reuseFailAlloc_1032_;
goto v_reusejp_1030_;
}
v_reusejp_1030_:
{
v___y_950_ = v___y_993_;
v___y_951_ = v___y_994_;
v___y_952_ = v___y_995_;
v___y_953_ = v___y_996_;
v___y_954_ = v___y_997_;
v___y_955_ = v_a_1003_;
v___y_956_ = v___y_998_;
v___y_957_ = v___y_999_;
v___y_958_ = v___y_1000_;
v___y_959_ = v___x_1024_;
v_a_960_ = v___x_1031_;
goto v___jp_949_;
}
}
}
else
{
lean_object* v_a_1034_; lean_object* v___x_1036_; uint8_t v_isShared_1037_; uint8_t v_isSharedCheck_1041_; 
v_a_1034_ = lean_ctor_get(v___x_1025_, 0);
v_isSharedCheck_1041_ = !lean_is_exclusive(v___x_1025_);
if (v_isSharedCheck_1041_ == 0)
{
v___x_1036_ = v___x_1025_;
v_isShared_1037_ = v_isSharedCheck_1041_;
goto v_resetjp_1035_;
}
else
{
lean_inc(v_a_1034_);
lean_dec(v___x_1025_);
v___x_1036_ = lean_box(0);
v_isShared_1037_ = v_isSharedCheck_1041_;
goto v_resetjp_1035_;
}
v_resetjp_1035_:
{
lean_object* v___x_1039_; 
if (v_isShared_1037_ == 0)
{
lean_ctor_set_tag(v___x_1036_, 0);
v___x_1039_ = v___x_1036_;
goto v_reusejp_1038_;
}
else
{
lean_object* v_reuseFailAlloc_1040_; 
v_reuseFailAlloc_1040_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1040_, 0, v_a_1034_);
v___x_1039_ = v_reuseFailAlloc_1040_;
goto v_reusejp_1038_;
}
v_reusejp_1038_:
{
v___y_950_ = v___y_993_;
v___y_951_ = v___y_994_;
v___y_952_ = v___y_995_;
v___y_953_ = v___y_996_;
v___y_954_ = v___y_997_;
v___y_955_ = v_a_1003_;
v___y_956_ = v___y_998_;
v___y_957_ = v___y_999_;
v___y_958_ = v___y_1000_;
v___y_959_ = v___x_1024_;
v_a_960_ = v___x_1039_;
goto v___jp_949_;
}
}
}
}
}
v___jp_1042_:
{
uint8_t v_hasTrace_1045_; lean_object* v___x_1046_; 
v_hasTrace_1045_ = lean_ctor_get_uint8(v_options_708_, sizeof(void*)*1);
v___x_1046_ = lp_aesop_Aesop_TraceOption_ruleSet;
if (v_hasTrace_1045_ == 0)
{
lean_object* v___x_1047_; 
lean_inc_ref(v_a_1044_);
v___x_1047_ = lp_aesop_Aesop_LocalRuleSet_trace(v_a_1044_, v___x_1046_, v_a_461_, v_a_462_);
v___y_928_ = v_a_1044_;
v___y_929_ = v___y_1043_;
v___y_930_ = v___x_1047_;
goto v___jp_927_;
}
else
{
lean_object* v_traceClass_1048_; lean_object* v___f_1049_; lean_object* v___x_1050_; lean_object* v___x_1051_; lean_object* v___x_1052_; uint8_t v___x_1053_; 
v_traceClass_1048_ = lean_ctor_get(v___x_1046_, 0);
v___f_1049_ = lean_obj_once(&lp_aesop___private_Aesop_Main_0__Aesop_evalAesop_go___closed__5, &lp_aesop___private_Aesop_Main_0__Aesop_evalAesop_go___closed__5_once, _init_lp_aesop___private_Aesop_Main_0__Aesop_evalAesop_go___closed__5);
v___x_1050_ = ((lean_object*)(lp_aesop___private_Aesop_Main_0__Aesop_evalAesop_go___closed__0));
v___x_1051_ = ((lean_object*)(lp_aesop___private_Aesop_Main_0__Aesop_evalAesop_go___closed__7));
lean_inc(v_traceClass_1048_);
v___x_1052_ = l_Lean_Name_append(v___x_1051_, v_traceClass_1048_);
v___x_1053_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_709_, v_options_708_, v___x_1052_);
lean_dec(v___x_1052_);
if (v___x_1053_ == 0)
{
lean_object* v___x_1054_; uint8_t v___x_1055_; 
v___x_1054_ = l_Lean_trace_profiler;
v___x_1055_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__0(v_options_708_, v___x_1054_);
if (v___x_1055_ == 0)
{
lean_object* v___x_1056_; 
lean_inc_ref(v_a_1044_);
v___x_1056_ = lp_aesop_Aesop_LocalRuleSet_trace(v_a_1044_, v___x_1046_, v_a_461_, v_a_462_);
v___y_928_ = v_a_1044_;
v___y_929_ = v___y_1043_;
v___y_930_ = v___x_1056_;
goto v___jp_927_;
}
else
{
lean_inc(v_traceClass_1048_);
v___y_993_ = v___f_1049_;
v___y_994_ = v_a_1044_;
v___y_995_ = v_hasTrace_1045_;
v___y_996_ = v_traceClass_1048_;
v___y_997_ = v___x_1053_;
v___y_998_ = v_options_708_;
v___y_999_ = v___x_1050_;
v___y_1000_ = v___y_1043_;
v___y_1001_ = v___x_1046_;
goto v___jp_992_;
}
}
else
{
lean_inc(v_traceClass_1048_);
v___y_993_ = v___f_1049_;
v___y_994_ = v_a_1044_;
v___y_995_ = v_hasTrace_1045_;
v___y_996_ = v_traceClass_1048_;
v___y_997_ = v___x_1053_;
v___y_998_ = v_options_708_;
v___y_999_ = v___x_1050_;
v___y_1000_ = v___y_1043_;
v___y_1001_ = v___x_1046_;
goto v___jp_992_;
}
}
}
v___jp_1057_:
{
lean_object* v___x_1059_; 
lean_inc_ref(v___y_1058_);
lean_inc(v_goal_453_);
v___x_1059_ = lp_aesop_Aesop_Frontend_TacticConfig_getRuleSet(v_goal_453_, v___y_1058_, v_a_457_, v_a_458_, v_a_459_, v_a_460_, v_a_461_, v_a_462_);
if (lean_obj_tag(v___x_1059_) == 0)
{
lean_object* v_a_1060_; 
v_a_1060_ = lean_ctor_get(v___x_1059_, 0);
lean_inc(v_a_1060_);
lean_dec_ref_known(v___x_1059_, 1);
v___y_1043_ = v___y_1058_;
v_a_1044_ = v_a_1060_;
goto v___jp_1042_;
}
else
{
lean_object* v_a_1061_; lean_object* v___x_1063_; uint8_t v_isShared_1064_; uint8_t v_isSharedCheck_1068_; 
lean_dec_ref(v___y_1058_);
lean_dec(v_goal_453_);
v_a_1061_ = lean_ctor_get(v___x_1059_, 0);
v_isSharedCheck_1068_ = !lean_is_exclusive(v___x_1059_);
if (v_isSharedCheck_1068_ == 0)
{
v___x_1063_ = v___x_1059_;
v_isShared_1064_ = v_isSharedCheck_1068_;
goto v_resetjp_1062_;
}
else
{
lean_inc(v_a_1061_);
lean_dec(v___x_1059_);
v___x_1063_ = lean_box(0);
v_isShared_1064_ = v_isSharedCheck_1068_;
goto v_resetjp_1062_;
}
v_resetjp_1062_:
{
lean_object* v___x_1066_; 
if (v_isShared_1064_ == 0)
{
v___x_1066_ = v___x_1063_;
goto v_reusejp_1065_;
}
else
{
lean_object* v_reuseFailAlloc_1067_; 
v_reuseFailAlloc_1067_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1067_, 0, v_a_1061_);
v___x_1066_ = v_reuseFailAlloc_1067_;
goto v_reusejp_1065_;
}
v_reusejp_1065_:
{
return v___x_1066_;
}
}
}
}
v___jp_1069_:
{
lean_object* v___x_1071_; lean_object* v___x_1072_; 
v___x_1071_ = lean_io_mono_nanos_now();
lean_inc_ref(v___y_1070_);
lean_inc(v_goal_453_);
v___x_1072_ = lp_aesop_Aesop_Frontend_TacticConfig_getRuleSet(v_goal_453_, v___y_1070_, v_a_457_, v_a_458_, v_a_459_, v_a_460_, v_a_461_, v_a_462_);
if (lean_obj_tag(v___x_1072_) == 0)
{
lean_object* v_a_1073_; lean_object* v___x_1074_; lean_object* v___x_1075_; lean_object* v_total_1076_; lean_object* v_configParsing_1077_; lean_object* v_search_1078_; lean_object* v_ruleSelection_1079_; lean_object* v_script_1080_; lean_object* v_forwardState_1081_; lean_object* v_scriptGenerated_1082_; lean_object* v_ruleStats_1083_; lean_object* v_goalStats_1084_; lean_object* v___x_1086_; uint8_t v_isShared_1087_; uint8_t v_isSharedCheck_1093_; 
v_a_1073_ = lean_ctor_get(v___x_1072_, 0);
lean_inc(v_a_1073_);
lean_dec_ref_known(v___x_1072_, 1);
v___x_1074_ = lean_io_mono_nanos_now();
v___x_1075_ = lean_st_ref_take(v_a_454_);
v_total_1076_ = lean_ctor_get(v___x_1075_, 0);
v_configParsing_1077_ = lean_ctor_get(v___x_1075_, 1);
v_search_1078_ = lean_ctor_get(v___x_1075_, 3);
v_ruleSelection_1079_ = lean_ctor_get(v___x_1075_, 4);
v_script_1080_ = lean_ctor_get(v___x_1075_, 5);
v_forwardState_1081_ = lean_ctor_get(v___x_1075_, 6);
v_scriptGenerated_1082_ = lean_ctor_get(v___x_1075_, 7);
v_ruleStats_1083_ = lean_ctor_get(v___x_1075_, 8);
v_goalStats_1084_ = lean_ctor_get(v___x_1075_, 9);
v_isSharedCheck_1093_ = !lean_is_exclusive(v___x_1075_);
if (v_isSharedCheck_1093_ == 0)
{
lean_object* v_unused_1094_; 
v_unused_1094_ = lean_ctor_get(v___x_1075_, 2);
lean_dec(v_unused_1094_);
v___x_1086_ = v___x_1075_;
v_isShared_1087_ = v_isSharedCheck_1093_;
goto v_resetjp_1085_;
}
else
{
lean_inc(v_goalStats_1084_);
lean_inc(v_ruleStats_1083_);
lean_inc(v_scriptGenerated_1082_);
lean_inc(v_forwardState_1081_);
lean_inc(v_script_1080_);
lean_inc(v_ruleSelection_1079_);
lean_inc(v_search_1078_);
lean_inc(v_configParsing_1077_);
lean_inc(v_total_1076_);
lean_dec(v___x_1075_);
v___x_1086_ = lean_box(0);
v_isShared_1087_ = v_isSharedCheck_1093_;
goto v_resetjp_1085_;
}
v_resetjp_1085_:
{
lean_object* v___x_1088_; lean_object* v___x_1090_; 
v___x_1088_ = lean_nat_sub(v___x_1074_, v___x_1071_);
lean_dec(v___x_1071_);
lean_dec(v___x_1074_);
if (v_isShared_1087_ == 0)
{
lean_ctor_set(v___x_1086_, 2, v___x_1088_);
v___x_1090_ = v___x_1086_;
goto v_reusejp_1089_;
}
else
{
lean_object* v_reuseFailAlloc_1092_; 
v_reuseFailAlloc_1092_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_1092_, 0, v_total_1076_);
lean_ctor_set(v_reuseFailAlloc_1092_, 1, v_configParsing_1077_);
lean_ctor_set(v_reuseFailAlloc_1092_, 2, v___x_1088_);
lean_ctor_set(v_reuseFailAlloc_1092_, 3, v_search_1078_);
lean_ctor_set(v_reuseFailAlloc_1092_, 4, v_ruleSelection_1079_);
lean_ctor_set(v_reuseFailAlloc_1092_, 5, v_script_1080_);
lean_ctor_set(v_reuseFailAlloc_1092_, 6, v_forwardState_1081_);
lean_ctor_set(v_reuseFailAlloc_1092_, 7, v_scriptGenerated_1082_);
lean_ctor_set(v_reuseFailAlloc_1092_, 8, v_ruleStats_1083_);
lean_ctor_set(v_reuseFailAlloc_1092_, 9, v_goalStats_1084_);
v___x_1090_ = v_reuseFailAlloc_1092_;
goto v_reusejp_1089_;
}
v_reusejp_1089_:
{
lean_object* v___x_1091_; 
v___x_1091_ = lean_st_ref_set(v_a_454_, v___x_1090_);
v___y_1043_ = v___y_1070_;
v_a_1044_ = v_a_1073_;
goto v___jp_1042_;
}
}
}
else
{
lean_object* v_a_1095_; lean_object* v___x_1097_; uint8_t v_isShared_1098_; uint8_t v_isSharedCheck_1102_; 
lean_dec(v___x_1071_);
lean_dec_ref(v___y_1070_);
lean_dec(v_goal_453_);
v_a_1095_ = lean_ctor_get(v___x_1072_, 0);
v_isSharedCheck_1102_ = !lean_is_exclusive(v___x_1072_);
if (v_isSharedCheck_1102_ == 0)
{
v___x_1097_ = v___x_1072_;
v_isShared_1098_ = v_isSharedCheck_1102_;
goto v_resetjp_1096_;
}
else
{
lean_inc(v_a_1095_);
lean_dec(v___x_1072_);
v___x_1097_ = lean_box(0);
v_isShared_1098_ = v_isSharedCheck_1102_;
goto v_resetjp_1096_;
}
v_resetjp_1096_:
{
lean_object* v___x_1100_; 
if (v_isShared_1098_ == 0)
{
v___x_1100_ = v___x_1097_;
goto v_reusejp_1099_;
}
else
{
lean_object* v_reuseFailAlloc_1101_; 
v_reuseFailAlloc_1101_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1101_, 0, v_a_1095_);
v___x_1100_ = v_reuseFailAlloc_1101_;
goto v_reusejp_1099_;
}
v_reusejp_1099_:
{
return v___x_1100_;
}
}
}
}
v___jp_1103_:
{
lean_object* v_a_1106_; uint8_t v___x_1107_; 
v_a_1106_ = lean_ctor_get(v___y_1105_, 0);
lean_inc(v_a_1106_);
lean_dec_ref(v___y_1105_);
v___x_1107_ = lean_unbox(v_a_1106_);
lean_dec(v_a_1106_);
if (v___x_1107_ == 0)
{
v___y_1058_ = v___y_1104_;
goto v___jp_1057_;
}
else
{
v___y_1070_ = v___y_1104_;
goto v___jp_1069_;
}
}
v___jp_1108_:
{
lean_object* v___x_1110_; uint8_t v___x_1111_; 
v___x_1110_ = lp_aesop_Aesop_aesop_collectStats;
v___x_1111_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__0(v_options_708_, v___x_1110_);
if (v___x_1111_ == 0)
{
lean_object* v___x_1112_; lean_object* v___x_1113_; lean_object* v_a_1114_; uint8_t v___x_1115_; 
v___x_1112_ = lp_aesop_Aesop_TraceOption_stats;
v___x_1113_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__1___redArg(v___x_1112_, v_a_461_);
v_a_1114_ = lean_ctor_get(v___x_1113_, 0);
lean_inc(v_a_1114_);
v___x_1115_ = lean_unbox(v_a_1114_);
lean_dec(v_a_1114_);
if (v___x_1115_ == 0)
{
lean_object* v___x_1116_; lean_object* v___x_1117_; lean_object* v___x_1118_; uint8_t v___x_1119_; 
lean_dec_ref(v___x_1113_);
v___x_1116_ = lp_aesop_Aesop_aesop_stats_file;
v___x_1117_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__2(v_options_708_, v___x_1116_);
v___x_1118_ = ((lean_object*)(lp_aesop___private_Aesop_Main_0__Aesop_evalAesop_go___closed__0));
v___x_1119_ = lean_string_dec_eq(v___x_1117_, v___x_1118_);
lean_dec_ref(v___x_1117_);
if (v___x_1119_ == 0)
{
v___y_1070_ = v_a_1109_;
goto v___jp_1069_;
}
else
{
v___y_1058_ = v_a_1109_;
goto v___jp_1057_;
}
}
else
{
v___y_1104_ = v_a_1109_;
v___y_1105_ = v___x_1113_;
goto v___jp_1103_;
}
}
else
{
v___y_1070_ = v_a_1109_;
goto v___jp_1069_;
}
}
v___jp_1120_:
{
lean_object* v___x_1121_; 
lean_inc(v_goal_453_);
v___x_1121_ = lp_aesop_Aesop_Frontend_TacticConfig_parse(v_stx_452_, v_goal_453_, v_a_457_, v_a_458_, v_a_459_, v_a_460_, v_a_461_, v_a_462_);
if (lean_obj_tag(v___x_1121_) == 0)
{
lean_object* v_a_1122_; 
v_a_1122_ = lean_ctor_get(v___x_1121_, 0);
lean_inc(v_a_1122_);
lean_dec_ref_known(v___x_1121_, 1);
v_a_1109_ = v_a_1122_;
goto v___jp_1108_;
}
else
{
lean_object* v_a_1123_; lean_object* v___x_1125_; uint8_t v_isShared_1126_; uint8_t v_isSharedCheck_1130_; 
lean_dec(v_goal_453_);
v_a_1123_ = lean_ctor_get(v___x_1121_, 0);
v_isSharedCheck_1130_ = !lean_is_exclusive(v___x_1121_);
if (v_isSharedCheck_1130_ == 0)
{
v___x_1125_ = v___x_1121_;
v_isShared_1126_ = v_isSharedCheck_1130_;
goto v_resetjp_1124_;
}
else
{
lean_inc(v_a_1123_);
lean_dec(v___x_1121_);
v___x_1125_ = lean_box(0);
v_isShared_1126_ = v_isSharedCheck_1130_;
goto v_resetjp_1124_;
}
v_resetjp_1124_:
{
lean_object* v___x_1128_; 
if (v_isShared_1126_ == 0)
{
v___x_1128_ = v___x_1125_;
goto v_reusejp_1127_;
}
else
{
lean_object* v_reuseFailAlloc_1129_; 
v_reuseFailAlloc_1129_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1129_, 0, v_a_1123_);
v___x_1128_ = v_reuseFailAlloc_1129_;
goto v_reusejp_1127_;
}
v_reusejp_1127_:
{
return v___x_1128_;
}
}
}
}
v___jp_1131_:
{
lean_object* v___x_1132_; lean_object* v___x_1133_; 
v___x_1132_ = lean_io_mono_nanos_now();
lean_inc(v_goal_453_);
v___x_1133_ = lp_aesop_Aesop_Frontend_TacticConfig_parse(v_stx_452_, v_goal_453_, v_a_457_, v_a_458_, v_a_459_, v_a_460_, v_a_461_, v_a_462_);
if (lean_obj_tag(v___x_1133_) == 0)
{
lean_object* v_a_1134_; lean_object* v___x_1135_; lean_object* v___x_1136_; lean_object* v___x_1137_; lean_object* v___x_1138_; lean_object* v___x_1139_; 
v_a_1134_ = lean_ctor_get(v___x_1133_, 0);
lean_inc(v_a_1134_);
lean_dec_ref_known(v___x_1133_, 1);
v___x_1135_ = lean_io_mono_nanos_now();
v___x_1136_ = lean_st_ref_take(v_a_454_);
v___x_1137_ = lean_nat_sub(v___x_1135_, v___x_1132_);
lean_dec(v___x_1132_);
lean_dec(v___x_1135_);
v___x_1138_ = lp_aesop___private_Aesop_Main_0__Aesop_evalAesop_go___lam__0(v___x_1136_, v_a_1134_, v___x_1137_);
v___x_1139_ = lean_st_ref_set(v_a_454_, v___x_1138_);
v_a_1109_ = v_a_1134_;
goto v___jp_1108_;
}
else
{
lean_object* v_a_1140_; lean_object* v___x_1142_; uint8_t v_isShared_1143_; uint8_t v_isSharedCheck_1147_; 
lean_dec(v___x_1132_);
lean_dec(v_goal_453_);
v_a_1140_ = lean_ctor_get(v___x_1133_, 0);
v_isSharedCheck_1147_ = !lean_is_exclusive(v___x_1133_);
if (v_isSharedCheck_1147_ == 0)
{
v___x_1142_ = v___x_1133_;
v_isShared_1143_ = v_isSharedCheck_1147_;
goto v_resetjp_1141_;
}
else
{
lean_inc(v_a_1140_);
lean_dec(v___x_1133_);
v___x_1142_ = lean_box(0);
v_isShared_1143_ = v_isSharedCheck_1147_;
goto v_resetjp_1141_;
}
v_resetjp_1141_:
{
lean_object* v___x_1145_; 
if (v_isShared_1143_ == 0)
{
v___x_1145_ = v___x_1142_;
goto v_reusejp_1144_;
}
else
{
lean_object* v_reuseFailAlloc_1146_; 
v_reuseFailAlloc_1146_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1146_, 0, v_a_1140_);
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
v___jp_1148_:
{
lean_object* v_a_1150_; uint8_t v___x_1151_; 
v_a_1150_ = lean_ctor_get(v___y_1149_, 0);
lean_inc(v_a_1150_);
lean_dec_ref(v___y_1149_);
v___x_1151_ = lean_unbox(v_a_1150_);
lean_dec(v_a_1150_);
if (v___x_1151_ == 0)
{
goto v___jp_1120_;
}
else
{
goto v___jp_1131_;
}
}
v___jp_1152_:
{
lean_object* v___x_1153_; uint8_t v___x_1154_; 
v___x_1153_ = lp_aesop_Aesop_aesop_collectStats;
v___x_1154_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__0(v_options_708_, v___x_1153_);
if (v___x_1154_ == 0)
{
lean_object* v___x_1155_; lean_object* v___x_1156_; lean_object* v_a_1157_; uint8_t v___x_1158_; 
v___x_1155_ = lp_aesop_Aesop_TraceOption_stats;
v___x_1156_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__1___redArg(v___x_1155_, v_a_461_);
v_a_1157_ = lean_ctor_get(v___x_1156_, 0);
lean_inc(v_a_1157_);
v___x_1158_ = lean_unbox(v_a_1157_);
lean_dec(v_a_1157_);
if (v___x_1158_ == 0)
{
lean_object* v___x_1159_; lean_object* v___x_1160_; lean_object* v___x_1161_; uint8_t v___x_1162_; 
lean_dec_ref(v___x_1156_);
v___x_1159_ = lp_aesop_Aesop_aesop_stats_file;
v___x_1160_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__2(v_options_708_, v___x_1159_);
v___x_1161_ = ((lean_object*)(lp_aesop___private_Aesop_Main_0__Aesop_evalAesop_go___closed__0));
v___x_1162_ = lean_string_dec_eq(v___x_1160_, v___x_1161_);
lean_dec_ref(v___x_1160_);
if (v___x_1162_ == 0)
{
goto v___jp_1131_;
}
else
{
goto v___jp_1120_;
}
}
else
{
v___y_1149_ = v___x_1156_;
goto v___jp_1148_;
}
}
else
{
goto v___jp_1131_;
}
}
v___jp_1163_:
{
lean_object* v___x_1165_; lean_object* v___x_1166_; 
v___x_1165_ = lean_io_mono_nanos_now();
lean_inc(v_goal_453_);
v___x_1166_ = lp_aesop_Aesop_Frontend_TacticConfig_parse(v_stx_452_, v_goal_453_, v_a_457_, v_a_458_, v_a_459_, v_a_460_, v_a_461_, v_a_462_);
if (lean_obj_tag(v___x_1166_) == 0)
{
lean_object* v_a_1167_; lean_object* v___x_1168_; lean_object* v___x_1169_; lean_object* v___x_1170_; lean_object* v___x_1171_; lean_object* v___x_1172_; 
v_a_1167_ = lean_ctor_get(v___x_1166_, 0);
lean_inc(v_a_1167_);
lean_dec_ref_known(v___x_1166_, 1);
v___x_1168_ = lean_io_mono_nanos_now();
v___x_1169_ = lean_st_ref_take(v_a_454_);
v___x_1170_ = lean_nat_sub(v___x_1168_, v___x_1165_);
lean_dec(v___x_1165_);
lean_dec(v___x_1168_);
v___x_1171_ = lp_aesop___private_Aesop_Main_0__Aesop_evalAesop_go___lam__0(v___x_1169_, v_a_1167_, v___x_1170_);
v___x_1172_ = lean_st_ref_set(v_a_454_, v___x_1171_);
v___y_900_ = v___y_1164_;
v_a_901_ = v_a_1167_;
goto v___jp_899_;
}
else
{
lean_dec(v___x_1165_);
v___y_913_ = v___y_1164_;
v___y_914_ = v___x_1166_;
goto v___jp_912_;
}
}
v___jp_1173_:
{
if (v_a_1175_ == 0)
{
v___y_925_ = v___y_1174_;
goto v___jp_924_;
}
else
{
v___y_1164_ = v___y_1174_;
goto v___jp_1163_;
}
}
v___jp_1176_:
{
lean_object* v_a_1179_; uint8_t v___x_1180_; 
v_a_1179_ = lean_ctor_get(v___y_1178_, 0);
lean_inc(v_a_1179_);
lean_dec_ref(v___y_1178_);
v___x_1180_ = lean_unbox(v_a_1179_);
lean_dec(v_a_1179_);
v___y_1174_ = v___y_1177_;
v_a_1175_ = v___x_1180_;
goto v___jp_1173_;
}
v___jp_1181_:
{
lean_object* v___x_1182_; lean_object* v___x_1183_; uint8_t v___x_1184_; 
v___x_1182_ = lean_io_mono_nanos_now();
v___x_1183_ = lp_aesop_Aesop_aesop_collectStats;
v___x_1184_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__0(v_options_708_, v___x_1183_);
if (v___x_1184_ == 0)
{
lean_object* v___x_1185_; lean_object* v___x_1186_; lean_object* v_a_1187_; uint8_t v___x_1188_; 
v___x_1185_ = lp_aesop_Aesop_TraceOption_stats;
v___x_1186_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__1___redArg(v___x_1185_, v_a_461_);
v_a_1187_ = lean_ctor_get(v___x_1186_, 0);
lean_inc(v_a_1187_);
v___x_1188_ = lean_unbox(v_a_1187_);
lean_dec(v_a_1187_);
if (v___x_1188_ == 0)
{
lean_object* v___x_1189_; lean_object* v___x_1190_; lean_object* v___x_1191_; uint8_t v___x_1192_; 
lean_dec_ref(v___x_1186_);
v___x_1189_ = lp_aesop_Aesop_aesop_stats_file;
v___x_1190_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__2(v_options_708_, v___x_1189_);
v___x_1191_ = ((lean_object*)(lp_aesop___private_Aesop_Main_0__Aesop_evalAesop_go___closed__0));
v___x_1192_ = lean_string_dec_eq(v___x_1190_, v___x_1191_);
lean_dec_ref(v___x_1190_);
if (v___x_1192_ == 0)
{
v___y_1164_ = v___x_1182_;
goto v___jp_1163_;
}
else
{
v___y_925_ = v___x_1182_;
goto v___jp_924_;
}
}
else
{
v___y_1177_ = v___x_1182_;
v___y_1178_ = v___x_1186_;
goto v___jp_1176_;
}
}
else
{
v___y_1174_ = v___x_1182_;
v_a_1175_ = v___x_1184_;
goto v___jp_1173_;
}
}
v___jp_1193_:
{
lean_object* v_a_1195_; uint8_t v___x_1196_; 
v_a_1195_ = lean_ctor_get(v___y_1194_, 0);
lean_inc(v_a_1195_);
lean_dec_ref(v___y_1194_);
v___x_1196_ = lean_unbox(v_a_1195_);
lean_dec(v_a_1195_);
if (v___x_1196_ == 0)
{
goto v___jp_1152_;
}
else
{
goto v___jp_1181_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Main_0__Aesop_evalAesop_go___boxed(lean_object* v_stx_1207_, lean_object* v_goal_1208_, lean_object* v_a_1209_, lean_object* v_a_1210_, lean_object* v_a_1211_, lean_object* v_a_1212_, lean_object* v_a_1213_, lean_object* v_a_1214_, lean_object* v_a_1215_, lean_object* v_a_1216_, lean_object* v_a_1217_, lean_object* v_a_1218_){
_start:
{
lean_object* v_res_1219_; 
v_res_1219_ = lp_aesop___private_Aesop_Main_0__Aesop_evalAesop_go(v_stx_1207_, v_goal_1208_, v_a_1209_, v_a_1210_, v_a_1211_, v_a_1212_, v_a_1213_, v_a_1214_, v_a_1215_, v_a_1216_, v_a_1217_);
lean_dec(v_a_1217_);
lean_dec_ref(v_a_1216_);
lean_dec(v_a_1215_);
lean_dec_ref(v_a_1214_);
lean_dec(v_a_1213_);
lean_dec_ref(v_a_1212_);
lean_dec(v_a_1211_);
lean_dec_ref(v_a_1210_);
lean_dec(v_a_1209_);
return v_res_1219_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__1(lean_object* v_opt_1220_, lean_object* v___y_1221_, lean_object* v___y_1222_, lean_object* v___y_1223_, lean_object* v___y_1224_, lean_object* v___y_1225_, lean_object* v___y_1226_, lean_object* v___y_1227_, lean_object* v___y_1228_, lean_object* v___y_1229_){
_start:
{
lean_object* v___x_1231_; 
v___x_1231_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__1___redArg(v_opt_1220_, v___y_1228_);
return v___x_1231_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__1___boxed(lean_object* v_opt_1232_, lean_object* v___y_1233_, lean_object* v___y_1234_, lean_object* v___y_1235_, lean_object* v___y_1236_, lean_object* v___y_1237_, lean_object* v___y_1238_, lean_object* v___y_1239_, lean_object* v___y_1240_, lean_object* v___y_1241_, lean_object* v___y_1242_){
_start:
{
lean_object* v_res_1243_; 
v_res_1243_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__1(v_opt_1232_, v___y_1233_, v___y_1234_, v___y_1235_, v___y_1236_, v___y_1237_, v___y_1238_, v___y_1239_, v___y_1240_, v___y_1241_);
lean_dec(v___y_1241_);
lean_dec_ref(v___y_1240_);
lean_dec(v___y_1239_);
lean_dec_ref(v___y_1238_);
lean_dec(v___y_1237_);
lean_dec_ref(v___y_1236_);
lean_dec(v___y_1235_);
lean_dec_ref(v___y_1234_);
lean_dec(v___y_1233_);
lean_dec_ref(v_opt_1232_);
return v_res_1243_;
}
}
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__4_spec__5(lean_object* v_00_u03b1_1244_, lean_object* v_x_1245_, lean_object* v___y_1246_, lean_object* v___y_1247_, lean_object* v___y_1248_, lean_object* v___y_1249_, lean_object* v___y_1250_, lean_object* v___y_1251_, lean_object* v___y_1252_, lean_object* v___y_1253_, lean_object* v___y_1254_){
_start:
{
lean_object* v___x_1256_; 
v___x_1256_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__4_spec__5___redArg(v_x_1245_);
return v___x_1256_;
}
}
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__4_spec__5___boxed(lean_object* v_00_u03b1_1257_, lean_object* v_x_1258_, lean_object* v___y_1259_, lean_object* v___y_1260_, lean_object* v___y_1261_, lean_object* v___y_1262_, lean_object* v___y_1263_, lean_object* v___y_1264_, lean_object* v___y_1265_, lean_object* v___y_1266_, lean_object* v___y_1267_, lean_object* v___y_1268_){
_start:
{
lean_object* v_res_1269_; 
v_res_1269_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__4_spec__5(v_00_u03b1_1257_, v_x_1258_, v___y_1259_, v___y_1260_, v___y_1261_, v___y_1262_, v___y_1263_, v___y_1264_, v___y_1265_, v___y_1266_, v___y_1267_);
lean_dec(v___y_1267_);
lean_dec_ref(v___y_1266_);
lean_dec(v___y_1265_);
lean_dec_ref(v___y_1264_);
lean_dec(v___y_1263_);
lean_dec_ref(v___y_1262_);
lean_dec(v___y_1261_);
lean_dec_ref(v___y_1260_);
lean_dec(v___y_1259_);
return v_res_1269_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__4_spec__4(lean_object* v_oldTraces_1270_, lean_object* v_data_1271_, lean_object* v_ref_1272_, lean_object* v_msg_1273_, lean_object* v___y_1274_, lean_object* v___y_1275_, lean_object* v___y_1276_, lean_object* v___y_1277_, lean_object* v___y_1278_, lean_object* v___y_1279_, lean_object* v___y_1280_, lean_object* v___y_1281_, lean_object* v___y_1282_){
_start:
{
lean_object* v___x_1284_; 
v___x_1284_ = lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__4_spec__4___redArg(v_oldTraces_1270_, v_data_1271_, v_ref_1272_, v_msg_1273_, v___y_1279_, v___y_1280_, v___y_1281_, v___y_1282_);
return v___x_1284_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__4_spec__4___boxed(lean_object* v_oldTraces_1285_, lean_object* v_data_1286_, lean_object* v_ref_1287_, lean_object* v_msg_1288_, lean_object* v___y_1289_, lean_object* v___y_1290_, lean_object* v___y_1291_, lean_object* v___y_1292_, lean_object* v___y_1293_, lean_object* v___y_1294_, lean_object* v___y_1295_, lean_object* v___y_1296_, lean_object* v___y_1297_, lean_object* v___y_1298_){
_start:
{
lean_object* v_res_1299_; 
v_res_1299_ = lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__4_spec__4(v_oldTraces_1285_, v_data_1286_, v_ref_1287_, v_msg_1288_, v___y_1289_, v___y_1290_, v___y_1291_, v___y_1292_, v___y_1293_, v___y_1294_, v___y_1295_, v___y_1296_, v___y_1297_);
lean_dec(v___y_1297_);
lean_dec_ref(v___y_1296_);
lean_dec(v___y_1295_);
lean_dec_ref(v___y_1294_);
lean_dec(v___y_1293_);
lean_dec_ref(v___y_1292_);
lean_dec(v___y_1291_);
lean_dec_ref(v___y_1290_);
lean_dec(v___y_1289_);
return v_res_1299_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_evalAesop_spec__2___redArg___lam__0(lean_object* v_x_1300_, lean_object* v___y_1301_, lean_object* v___y_1302_, lean_object* v___y_1303_, lean_object* v___y_1304_, lean_object* v___y_1305_, lean_object* v___y_1306_, lean_object* v___y_1307_, lean_object* v___y_1308_){
_start:
{
lean_object* v___x_1310_; 
lean_inc(v___y_1304_);
lean_inc_ref(v___y_1303_);
lean_inc(v___y_1302_);
lean_inc_ref(v___y_1301_);
v___x_1310_ = lean_apply_9(v_x_1300_, v___y_1301_, v___y_1302_, v___y_1303_, v___y_1304_, v___y_1305_, v___y_1306_, v___y_1307_, v___y_1308_, lean_box(0));
return v___x_1310_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_evalAesop_spec__2___redArg___lam__0___boxed(lean_object* v_x_1311_, lean_object* v___y_1312_, lean_object* v___y_1313_, lean_object* v___y_1314_, lean_object* v___y_1315_, lean_object* v___y_1316_, lean_object* v___y_1317_, lean_object* v___y_1318_, lean_object* v___y_1319_, lean_object* v___y_1320_){
_start:
{
lean_object* v_res_1321_; 
v_res_1321_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_evalAesop_spec__2___redArg___lam__0(v_x_1311_, v___y_1312_, v___y_1313_, v___y_1314_, v___y_1315_, v___y_1316_, v___y_1317_, v___y_1318_, v___y_1319_);
lean_dec(v___y_1315_);
lean_dec_ref(v___y_1314_);
lean_dec(v___y_1313_);
lean_dec_ref(v___y_1312_);
return v_res_1321_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_evalAesop_spec__2___redArg(lean_object* v_mvarId_1322_, lean_object* v_x_1323_, lean_object* v___y_1324_, lean_object* v___y_1325_, lean_object* v___y_1326_, lean_object* v___y_1327_, lean_object* v___y_1328_, lean_object* v___y_1329_, lean_object* v___y_1330_, lean_object* v___y_1331_){
_start:
{
lean_object* v___f_1333_; lean_object* v___x_1334_; 
lean_inc(v___y_1327_);
lean_inc_ref(v___y_1326_);
lean_inc(v___y_1325_);
lean_inc_ref(v___y_1324_);
v___f_1333_ = lean_alloc_closure((void*)(lp_aesop_Lean_MVarId_withContext___at___00Aesop_evalAesop_spec__2___redArg___lam__0___boxed), 10, 5);
lean_closure_set(v___f_1333_, 0, v_x_1323_);
lean_closure_set(v___f_1333_, 1, v___y_1324_);
lean_closure_set(v___f_1333_, 2, v___y_1325_);
lean_closure_set(v___f_1333_, 3, v___y_1326_);
lean_closure_set(v___f_1333_, 4, v___y_1327_);
v___x_1334_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_box(0), v_mvarId_1322_, v___f_1333_, v___y_1328_, v___y_1329_, v___y_1330_, v___y_1331_);
if (lean_obj_tag(v___x_1334_) == 0)
{
return v___x_1334_;
}
else
{
lean_object* v_a_1335_; lean_object* v___x_1337_; uint8_t v_isShared_1338_; uint8_t v_isSharedCheck_1342_; 
v_a_1335_ = lean_ctor_get(v___x_1334_, 0);
v_isSharedCheck_1342_ = !lean_is_exclusive(v___x_1334_);
if (v_isSharedCheck_1342_ == 0)
{
v___x_1337_ = v___x_1334_;
v_isShared_1338_ = v_isSharedCheck_1342_;
goto v_resetjp_1336_;
}
else
{
lean_inc(v_a_1335_);
lean_dec(v___x_1334_);
v___x_1337_ = lean_box(0);
v_isShared_1338_ = v_isSharedCheck_1342_;
goto v_resetjp_1336_;
}
v_resetjp_1336_:
{
lean_object* v___x_1340_; 
if (v_isShared_1338_ == 0)
{
v___x_1340_ = v___x_1337_;
goto v_reusejp_1339_;
}
else
{
lean_object* v_reuseFailAlloc_1341_; 
v_reuseFailAlloc_1341_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1341_, 0, v_a_1335_);
v___x_1340_ = v_reuseFailAlloc_1341_;
goto v_reusejp_1339_;
}
v_reusejp_1339_:
{
return v___x_1340_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_evalAesop_spec__2___redArg___boxed(lean_object* v_mvarId_1343_, lean_object* v_x_1344_, lean_object* v___y_1345_, lean_object* v___y_1346_, lean_object* v___y_1347_, lean_object* v___y_1348_, lean_object* v___y_1349_, lean_object* v___y_1350_, lean_object* v___y_1351_, lean_object* v___y_1352_, lean_object* v___y_1353_){
_start:
{
lean_object* v_res_1354_; 
v_res_1354_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_evalAesop_spec__2___redArg(v_mvarId_1343_, v_x_1344_, v___y_1345_, v___y_1346_, v___y_1347_, v___y_1348_, v___y_1349_, v___y_1350_, v___y_1351_, v___y_1352_);
lean_dec(v___y_1352_);
lean_dec_ref(v___y_1351_);
lean_dec(v___y_1350_);
lean_dec_ref(v___y_1349_);
lean_dec(v___y_1348_);
lean_dec_ref(v___y_1347_);
lean_dec(v___y_1346_);
lean_dec_ref(v___y_1345_);
return v_res_1354_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_evalAesop_spec__2(lean_object* v_00_u03b1_1355_, lean_object* v_mvarId_1356_, lean_object* v_x_1357_, lean_object* v___y_1358_, lean_object* v___y_1359_, lean_object* v___y_1360_, lean_object* v___y_1361_, lean_object* v___y_1362_, lean_object* v___y_1363_, lean_object* v___y_1364_, lean_object* v___y_1365_){
_start:
{
lean_object* v___x_1367_; 
v___x_1367_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_evalAesop_spec__2___redArg(v_mvarId_1356_, v_x_1357_, v___y_1358_, v___y_1359_, v___y_1360_, v___y_1361_, v___y_1362_, v___y_1363_, v___y_1364_, v___y_1365_);
return v___x_1367_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_evalAesop_spec__2___boxed(lean_object* v_00_u03b1_1368_, lean_object* v_mvarId_1369_, lean_object* v_x_1370_, lean_object* v___y_1371_, lean_object* v___y_1372_, lean_object* v___y_1373_, lean_object* v___y_1374_, lean_object* v___y_1375_, lean_object* v___y_1376_, lean_object* v___y_1377_, lean_object* v___y_1378_, lean_object* v___y_1379_){
_start:
{
lean_object* v_res_1380_; 
v_res_1380_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_evalAesop_spec__2(v_00_u03b1_1368_, v_mvarId_1369_, v_x_1370_, v___y_1371_, v___y_1372_, v___y_1373_, v___y_1374_, v___y_1375_, v___y_1376_, v___y_1377_, v___y_1378_);
lean_dec(v___y_1378_);
lean_dec_ref(v___y_1377_);
lean_dec(v___y_1376_);
lean_dec_ref(v___y_1375_);
lean_dec(v___y_1374_);
lean_dec_ref(v___y_1373_);
lean_dec(v___y_1372_);
lean_dec_ref(v___y_1371_);
return v_res_1380_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_profileitM___at___00Aesop_evalAesop_spec__3___redArg(lean_object* v_category_1381_, lean_object* v_opts_1382_, lean_object* v_act_1383_, lean_object* v_decl_1384_, lean_object* v___y_1385_, lean_object* v___y_1386_, lean_object* v___y_1387_, lean_object* v___y_1388_, lean_object* v___y_1389_, lean_object* v___y_1390_, lean_object* v___y_1391_, lean_object* v___y_1392_){
_start:
{
lean_object* v___x_1394_; lean_object* v___x_1395_; 
lean_inc(v___y_1392_);
lean_inc_ref(v___y_1391_);
lean_inc(v___y_1390_);
lean_inc_ref(v___y_1389_);
lean_inc(v___y_1388_);
lean_inc_ref(v___y_1387_);
lean_inc(v___y_1386_);
lean_inc_ref(v___y_1385_);
v___x_1394_ = lean_apply_8(v_act_1383_, v___y_1385_, v___y_1386_, v___y_1387_, v___y_1388_, v___y_1389_, v___y_1390_, v___y_1391_, v___y_1392_);
v___x_1395_ = l_Lean_profileitIOUnsafe___redArg(v_category_1381_, v_opts_1382_, v___x_1394_, v_decl_1384_);
return v___x_1395_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_profileitM___at___00Aesop_evalAesop_spec__3___redArg___boxed(lean_object* v_category_1396_, lean_object* v_opts_1397_, lean_object* v_act_1398_, lean_object* v_decl_1399_, lean_object* v___y_1400_, lean_object* v___y_1401_, lean_object* v___y_1402_, lean_object* v___y_1403_, lean_object* v___y_1404_, lean_object* v___y_1405_, lean_object* v___y_1406_, lean_object* v___y_1407_, lean_object* v___y_1408_){
_start:
{
lean_object* v_res_1409_; 
v_res_1409_ = lp_aesop_Lean_profileitM___at___00Aesop_evalAesop_spec__3___redArg(v_category_1396_, v_opts_1397_, v_act_1398_, v_decl_1399_, v___y_1400_, v___y_1401_, v___y_1402_, v___y_1403_, v___y_1404_, v___y_1405_, v___y_1406_, v___y_1407_);
lean_dec(v___y_1407_);
lean_dec_ref(v___y_1406_);
lean_dec(v___y_1405_);
lean_dec_ref(v___y_1404_);
lean_dec(v___y_1403_);
lean_dec_ref(v___y_1402_);
lean_dec(v___y_1401_);
lean_dec_ref(v___y_1400_);
lean_dec_ref(v_opts_1397_);
lean_dec_ref(v_category_1396_);
return v_res_1409_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_profileitM___at___00Aesop_evalAesop_spec__3(lean_object* v_00_u03b1_1410_, lean_object* v_category_1411_, lean_object* v_opts_1412_, lean_object* v_act_1413_, lean_object* v_decl_1414_, lean_object* v___y_1415_, lean_object* v___y_1416_, lean_object* v___y_1417_, lean_object* v___y_1418_, lean_object* v___y_1419_, lean_object* v___y_1420_, lean_object* v___y_1421_, lean_object* v___y_1422_){
_start:
{
lean_object* v___x_1424_; 
v___x_1424_ = lp_aesop_Lean_profileitM___at___00Aesop_evalAesop_spec__3___redArg(v_category_1411_, v_opts_1412_, v_act_1413_, v_decl_1414_, v___y_1415_, v___y_1416_, v___y_1417_, v___y_1418_, v___y_1419_, v___y_1420_, v___y_1421_, v___y_1422_);
return v___x_1424_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_profileitM___at___00Aesop_evalAesop_spec__3___boxed(lean_object* v_00_u03b1_1425_, lean_object* v_category_1426_, lean_object* v_opts_1427_, lean_object* v_act_1428_, lean_object* v_decl_1429_, lean_object* v___y_1430_, lean_object* v___y_1431_, lean_object* v___y_1432_, lean_object* v___y_1433_, lean_object* v___y_1434_, lean_object* v___y_1435_, lean_object* v___y_1436_, lean_object* v___y_1437_, lean_object* v___y_1438_){
_start:
{
lean_object* v_res_1439_; 
v_res_1439_ = lp_aesop_Lean_profileitM___at___00Aesop_evalAesop_spec__3(v_00_u03b1_1425_, v_category_1426_, v_opts_1427_, v_act_1428_, v_decl_1429_, v___y_1430_, v___y_1431_, v___y_1432_, v___y_1433_, v___y_1434_, v___y_1435_, v___y_1436_, v___y_1437_);
lean_dec(v___y_1437_);
lean_dec_ref(v___y_1436_);
lean_dec(v___y_1435_);
lean_dec_ref(v___y_1434_);
lean_dec(v___y_1433_);
lean_dec_ref(v___y_1432_);
lean_dec(v___y_1431_);
lean_dec_ref(v___y_1430_);
lean_dec_ref(v_opts_1427_);
lean_dec_ref(v_category_1426_);
return v_res_1439_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_StatsExtensionEntry_forCurrentFile___at___00Aesop_recordStatsForCurrentFileIfEnabled___at___00Aesop_evalAesop_spec__0_spec__0___redArg(lean_object* v_stx_1440_, lean_object* v_stats_1441_, lean_object* v___y_1442_){
_start:
{
lean_object* v_fileName_1444_; lean_object* v_fileMap_1445_; lean_object* v___y_1447_; uint8_t v___x_1450_; lean_object* v___x_1451_; 
v_fileName_1444_ = lean_ctor_get(v___y_1442_, 0);
v_fileMap_1445_ = lean_ctor_get(v___y_1442_, 1);
v___x_1450_ = 0;
v___x_1451_ = l_Lean_Syntax_getPos_x3f(v_stx_1440_, v___x_1450_);
if (lean_obj_tag(v___x_1451_) == 0)
{
lean_object* v___x_1452_; 
v___x_1452_ = lean_box(0);
v___y_1447_ = v___x_1452_;
goto v___jp_1446_;
}
else
{
lean_object* v_val_1453_; lean_object* v___x_1455_; uint8_t v_isShared_1456_; uint8_t v_isSharedCheck_1461_; 
v_val_1453_ = lean_ctor_get(v___x_1451_, 0);
v_isSharedCheck_1461_ = !lean_is_exclusive(v___x_1451_);
if (v_isSharedCheck_1461_ == 0)
{
v___x_1455_ = v___x_1451_;
v_isShared_1456_ = v_isSharedCheck_1461_;
goto v_resetjp_1454_;
}
else
{
lean_inc(v_val_1453_);
lean_dec(v___x_1451_);
v___x_1455_ = lean_box(0);
v_isShared_1456_ = v_isSharedCheck_1461_;
goto v_resetjp_1454_;
}
v_resetjp_1454_:
{
lean_object* v___x_1457_; lean_object* v___x_1459_; 
lean_inc_ref(v_fileMap_1445_);
v___x_1457_ = l_Lean_FileMap_toPosition(v_fileMap_1445_, v_val_1453_);
lean_dec(v_val_1453_);
if (v_isShared_1456_ == 0)
{
lean_ctor_set(v___x_1455_, 0, v___x_1457_);
v___x_1459_ = v___x_1455_;
goto v_reusejp_1458_;
}
else
{
lean_object* v_reuseFailAlloc_1460_; 
v_reuseFailAlloc_1460_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1460_, 0, v___x_1457_);
v___x_1459_ = v_reuseFailAlloc_1460_;
goto v_reusejp_1458_;
}
v_reusejp_1458_:
{
v___y_1447_ = v___x_1459_;
goto v___jp_1446_;
}
}
}
v___jp_1446_:
{
lean_object* v___x_1448_; lean_object* v___x_1449_; 
lean_inc_ref(v_fileName_1444_);
v___x_1448_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_1448_, 0, v_stx_1440_);
lean_ctor_set(v___x_1448_, 1, v_fileName_1444_);
lean_ctor_set(v___x_1448_, 2, v___y_1447_);
lean_ctor_set(v___x_1448_, 3, v_stats_1441_);
v___x_1449_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1449_, 0, v___x_1448_);
return v___x_1449_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_StatsExtensionEntry_forCurrentFile___at___00Aesop_recordStatsForCurrentFileIfEnabled___at___00Aesop_evalAesop_spec__0_spec__0___redArg___boxed(lean_object* v_stx_1462_, lean_object* v_stats_1463_, lean_object* v___y_1464_, lean_object* v___y_1465_){
_start:
{
lean_object* v_res_1466_; 
v_res_1466_ = lp_aesop_Aesop_StatsExtensionEntry_forCurrentFile___at___00Aesop_recordStatsForCurrentFileIfEnabled___at___00Aesop_evalAesop_spec__0_spec__0___redArg(v_stx_1462_, v_stats_1463_, v___y_1464_);
lean_dec_ref(v___y_1464_);
return v_res_1466_;
}
}
static lean_object* _init_lp_aesop_Aesop_recordStatsForCurrentFileIfEnabled___at___00Aesop_evalAesop_spec__0___closed__0(void){
_start:
{
lean_object* v___x_1467_; 
v___x_1467_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_1467_;
}
}
static lean_object* _init_lp_aesop_Aesop_recordStatsForCurrentFileIfEnabled___at___00Aesop_evalAesop_spec__0___closed__1(void){
_start:
{
lean_object* v___x_1468_; lean_object* v___x_1469_; 
v___x_1468_ = lean_obj_once(&lp_aesop_Aesop_recordStatsForCurrentFileIfEnabled___at___00Aesop_evalAesop_spec__0___closed__0, &lp_aesop_Aesop_recordStatsForCurrentFileIfEnabled___at___00Aesop_evalAesop_spec__0___closed__0_once, _init_lp_aesop_Aesop_recordStatsForCurrentFileIfEnabled___at___00Aesop_evalAesop_spec__0___closed__0);
v___x_1469_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1469_, 0, v___x_1468_);
return v___x_1469_;
}
}
static lean_object* _init_lp_aesop_Aesop_recordStatsForCurrentFileIfEnabled___at___00Aesop_evalAesop_spec__0___closed__2(void){
_start:
{
lean_object* v___x_1470_; lean_object* v___x_1471_; 
v___x_1470_ = lean_obj_once(&lp_aesop_Aesop_recordStatsForCurrentFileIfEnabled___at___00Aesop_evalAesop_spec__0___closed__1, &lp_aesop_Aesop_recordStatsForCurrentFileIfEnabled___at___00Aesop_evalAesop_spec__0___closed__1_once, _init_lp_aesop_Aesop_recordStatsForCurrentFileIfEnabled___at___00Aesop_evalAesop_spec__0___closed__1);
v___x_1471_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1471_, 0, v___x_1470_);
lean_ctor_set(v___x_1471_, 1, v___x_1470_);
return v___x_1471_;
}
}
static lean_object* _init_lp_aesop_Aesop_recordStatsForCurrentFileIfEnabled___at___00Aesop_evalAesop_spec__0___closed__3(void){
_start:
{
lean_object* v___x_1472_; lean_object* v___x_1473_; 
v___x_1472_ = lean_obj_once(&lp_aesop_Aesop_recordStatsForCurrentFileIfEnabled___at___00Aesop_evalAesop_spec__0___closed__1, &lp_aesop_Aesop_recordStatsForCurrentFileIfEnabled___at___00Aesop_evalAesop_spec__0___closed__1_once, _init_lp_aesop_Aesop_recordStatsForCurrentFileIfEnabled___at___00Aesop_evalAesop_spec__0___closed__1);
v___x_1473_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_1473_, 0, v___x_1472_);
lean_ctor_set(v___x_1473_, 1, v___x_1472_);
lean_ctor_set(v___x_1473_, 2, v___x_1472_);
lean_ctor_set(v___x_1473_, 3, v___x_1472_);
lean_ctor_set(v___x_1473_, 4, v___x_1472_);
lean_ctor_set(v___x_1473_, 5, v___x_1472_);
return v___x_1473_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_recordStatsForCurrentFileIfEnabled___at___00Aesop_evalAesop_spec__0(lean_object* v_aesopStx_1474_, lean_object* v_stats_1475_, lean_object* v___y_1476_, lean_object* v___y_1477_, lean_object* v___y_1478_, lean_object* v___y_1479_, lean_object* v___y_1480_, lean_object* v___y_1481_, lean_object* v___y_1482_, lean_object* v___y_1483_){
_start:
{
lean_object* v_options_1485_; lean_object* v___x_1486_; uint8_t v___x_1487_; 
v_options_1485_ = lean_ctor_get(v___y_1482_, 2);
v___x_1486_ = lp_aesop_Aesop_aesop_collectStats;
v___x_1487_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__0(v_options_1485_, v___x_1486_);
if (v___x_1487_ == 0)
{
lean_object* v___x_1488_; lean_object* v___x_1489_; 
lean_dec_ref(v_stats_1475_);
lean_dec(v_aesopStx_1474_);
v___x_1488_ = lean_box(0);
v___x_1489_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1489_, 0, v___x_1488_);
return v___x_1489_;
}
else
{
lean_object* v___x_1490_; lean_object* v_a_1491_; lean_object* v___x_1493_; uint8_t v_isShared_1494_; uint8_t v_isSharedCheck_1538_; 
v___x_1490_ = lp_aesop_Aesop_StatsExtensionEntry_forCurrentFile___at___00Aesop_recordStatsForCurrentFileIfEnabled___at___00Aesop_evalAesop_spec__0_spec__0___redArg(v_aesopStx_1474_, v_stats_1475_, v___y_1482_);
v_a_1491_ = lean_ctor_get(v___x_1490_, 0);
v_isSharedCheck_1538_ = !lean_is_exclusive(v___x_1490_);
if (v_isSharedCheck_1538_ == 0)
{
v___x_1493_ = v___x_1490_;
v_isShared_1494_ = v_isSharedCheck_1538_;
goto v_resetjp_1492_;
}
else
{
lean_inc(v_a_1491_);
lean_dec(v___x_1490_);
v___x_1493_ = lean_box(0);
v_isShared_1494_ = v_isSharedCheck_1538_;
goto v_resetjp_1492_;
}
v_resetjp_1492_:
{
lean_object* v___x_1495_; lean_object* v_env_1496_; lean_object* v_nextMacroScope_1497_; lean_object* v_ngen_1498_; lean_object* v_auxDeclNGen_1499_; lean_object* v_traceState_1500_; lean_object* v_messages_1501_; lean_object* v_infoState_1502_; lean_object* v_snapshotTasks_1503_; lean_object* v___x_1505_; uint8_t v_isShared_1506_; uint8_t v_isSharedCheck_1536_; 
v___x_1495_ = lean_st_ref_take(v___y_1483_);
v_env_1496_ = lean_ctor_get(v___x_1495_, 0);
v_nextMacroScope_1497_ = lean_ctor_get(v___x_1495_, 1);
v_ngen_1498_ = lean_ctor_get(v___x_1495_, 2);
v_auxDeclNGen_1499_ = lean_ctor_get(v___x_1495_, 3);
v_traceState_1500_ = lean_ctor_get(v___x_1495_, 4);
v_messages_1501_ = lean_ctor_get(v___x_1495_, 6);
v_infoState_1502_ = lean_ctor_get(v___x_1495_, 7);
v_snapshotTasks_1503_ = lean_ctor_get(v___x_1495_, 8);
v_isSharedCheck_1536_ = !lean_is_exclusive(v___x_1495_);
if (v_isSharedCheck_1536_ == 0)
{
lean_object* v_unused_1537_; 
v_unused_1537_ = lean_ctor_get(v___x_1495_, 5);
lean_dec(v_unused_1537_);
v___x_1505_ = v___x_1495_;
v_isShared_1506_ = v_isSharedCheck_1536_;
goto v_resetjp_1504_;
}
else
{
lean_inc(v_snapshotTasks_1503_);
lean_inc(v_infoState_1502_);
lean_inc(v_messages_1501_);
lean_inc(v_traceState_1500_);
lean_inc(v_auxDeclNGen_1499_);
lean_inc(v_ngen_1498_);
lean_inc(v_nextMacroScope_1497_);
lean_inc(v_env_1496_);
lean_dec(v___x_1495_);
v___x_1505_ = lean_box(0);
v_isShared_1506_ = v_isSharedCheck_1536_;
goto v_resetjp_1504_;
}
v_resetjp_1504_:
{
lean_object* v___x_1507_; lean_object* v_toEnvExtension_1508_; lean_object* v_asyncMode_1509_; lean_object* v___x_1510_; lean_object* v___x_1511_; lean_object* v___x_1512_; lean_object* v___x_1514_; 
v___x_1507_ = lp_aesop_Aesop_statsExtension;
v_toEnvExtension_1508_ = lean_ctor_get(v___x_1507_, 0);
v_asyncMode_1509_ = lean_ctor_get(v_toEnvExtension_1508_, 2);
v___x_1510_ = lean_box(0);
v___x_1511_ = l_Lean_PersistentEnvExtension_addEntry___redArg(v___x_1507_, v_env_1496_, v_a_1491_, v_asyncMode_1509_, v___x_1510_);
v___x_1512_ = lean_obj_once(&lp_aesop_Aesop_recordStatsForCurrentFileIfEnabled___at___00Aesop_evalAesop_spec__0___closed__2, &lp_aesop_Aesop_recordStatsForCurrentFileIfEnabled___at___00Aesop_evalAesop_spec__0___closed__2_once, _init_lp_aesop_Aesop_recordStatsForCurrentFileIfEnabled___at___00Aesop_evalAesop_spec__0___closed__2);
if (v_isShared_1506_ == 0)
{
lean_ctor_set(v___x_1505_, 5, v___x_1512_);
lean_ctor_set(v___x_1505_, 0, v___x_1511_);
v___x_1514_ = v___x_1505_;
goto v_reusejp_1513_;
}
else
{
lean_object* v_reuseFailAlloc_1535_; 
v_reuseFailAlloc_1535_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_1535_, 0, v___x_1511_);
lean_ctor_set(v_reuseFailAlloc_1535_, 1, v_nextMacroScope_1497_);
lean_ctor_set(v_reuseFailAlloc_1535_, 2, v_ngen_1498_);
lean_ctor_set(v_reuseFailAlloc_1535_, 3, v_auxDeclNGen_1499_);
lean_ctor_set(v_reuseFailAlloc_1535_, 4, v_traceState_1500_);
lean_ctor_set(v_reuseFailAlloc_1535_, 5, v___x_1512_);
lean_ctor_set(v_reuseFailAlloc_1535_, 6, v_messages_1501_);
lean_ctor_set(v_reuseFailAlloc_1535_, 7, v_infoState_1502_);
lean_ctor_set(v_reuseFailAlloc_1535_, 8, v_snapshotTasks_1503_);
v___x_1514_ = v_reuseFailAlloc_1535_;
goto v_reusejp_1513_;
}
v_reusejp_1513_:
{
lean_object* v___x_1515_; lean_object* v___x_1516_; lean_object* v_mctx_1517_; lean_object* v_zetaDeltaFVarIds_1518_; lean_object* v_postponed_1519_; lean_object* v_diag_1520_; lean_object* v___x_1522_; uint8_t v_isShared_1523_; uint8_t v_isSharedCheck_1533_; 
v___x_1515_ = lean_st_ref_set(v___y_1483_, v___x_1514_);
v___x_1516_ = lean_st_ref_take(v___y_1481_);
v_mctx_1517_ = lean_ctor_get(v___x_1516_, 0);
v_zetaDeltaFVarIds_1518_ = lean_ctor_get(v___x_1516_, 2);
v_postponed_1519_ = lean_ctor_get(v___x_1516_, 3);
v_diag_1520_ = lean_ctor_get(v___x_1516_, 4);
v_isSharedCheck_1533_ = !lean_is_exclusive(v___x_1516_);
if (v_isSharedCheck_1533_ == 0)
{
lean_object* v_unused_1534_; 
v_unused_1534_ = lean_ctor_get(v___x_1516_, 1);
lean_dec(v_unused_1534_);
v___x_1522_ = v___x_1516_;
v_isShared_1523_ = v_isSharedCheck_1533_;
goto v_resetjp_1521_;
}
else
{
lean_inc(v_diag_1520_);
lean_inc(v_postponed_1519_);
lean_inc(v_zetaDeltaFVarIds_1518_);
lean_inc(v_mctx_1517_);
lean_dec(v___x_1516_);
v___x_1522_ = lean_box(0);
v_isShared_1523_ = v_isSharedCheck_1533_;
goto v_resetjp_1521_;
}
v_resetjp_1521_:
{
lean_object* v___x_1524_; lean_object* v___x_1526_; 
v___x_1524_ = lean_obj_once(&lp_aesop_Aesop_recordStatsForCurrentFileIfEnabled___at___00Aesop_evalAesop_spec__0___closed__3, &lp_aesop_Aesop_recordStatsForCurrentFileIfEnabled___at___00Aesop_evalAesop_spec__0___closed__3_once, _init_lp_aesop_Aesop_recordStatsForCurrentFileIfEnabled___at___00Aesop_evalAesop_spec__0___closed__3);
if (v_isShared_1523_ == 0)
{
lean_ctor_set(v___x_1522_, 1, v___x_1524_);
v___x_1526_ = v___x_1522_;
goto v_reusejp_1525_;
}
else
{
lean_object* v_reuseFailAlloc_1532_; 
v_reuseFailAlloc_1532_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1532_, 0, v_mctx_1517_);
lean_ctor_set(v_reuseFailAlloc_1532_, 1, v___x_1524_);
lean_ctor_set(v_reuseFailAlloc_1532_, 2, v_zetaDeltaFVarIds_1518_);
lean_ctor_set(v_reuseFailAlloc_1532_, 3, v_postponed_1519_);
lean_ctor_set(v_reuseFailAlloc_1532_, 4, v_diag_1520_);
v___x_1526_ = v_reuseFailAlloc_1532_;
goto v_reusejp_1525_;
}
v_reusejp_1525_:
{
lean_object* v___x_1527_; lean_object* v___x_1528_; lean_object* v___x_1530_; 
v___x_1527_ = lean_st_ref_set(v___y_1481_, v___x_1526_);
v___x_1528_ = lean_box(0);
if (v_isShared_1494_ == 0)
{
lean_ctor_set(v___x_1493_, 0, v___x_1528_);
v___x_1530_ = v___x_1493_;
goto v_reusejp_1529_;
}
else
{
lean_object* v_reuseFailAlloc_1531_; 
v_reuseFailAlloc_1531_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1531_, 0, v___x_1528_);
v___x_1530_ = v_reuseFailAlloc_1531_;
goto v_reusejp_1529_;
}
v_reusejp_1529_:
{
return v___x_1530_;
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_recordStatsForCurrentFileIfEnabled___at___00Aesop_evalAesop_spec__0___boxed(lean_object* v_aesopStx_1539_, lean_object* v_stats_1540_, lean_object* v___y_1541_, lean_object* v___y_1542_, lean_object* v___y_1543_, lean_object* v___y_1544_, lean_object* v___y_1545_, lean_object* v___y_1546_, lean_object* v___y_1547_, lean_object* v___y_1548_, lean_object* v___y_1549_){
_start:
{
lean_object* v_res_1550_; 
v_res_1550_ = lp_aesop_Aesop_recordStatsForCurrentFileIfEnabled___at___00Aesop_evalAesop_spec__0(v_aesopStx_1539_, v_stats_1540_, v___y_1541_, v___y_1542_, v___y_1543_, v___y_1544_, v___y_1545_, v___y_1546_, v___y_1547_, v___y_1548_);
lean_dec(v___y_1548_);
lean_dec_ref(v___y_1547_);
lean_dec(v___y_1546_);
lean_dec_ref(v___y_1545_);
lean_dec(v___y_1544_);
lean_dec_ref(v___y_1543_);
lean_dec(v___y_1542_);
lean_dec_ref(v___y_1541_);
return v_res_1550_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_StatsFileRecord_ofStats___at___00Aesop_appendStatsToStatsFileIfEnabled___at___00Aesop_evalAesop_spec__1_spec__2___redArg(lean_object* v_aesopStx_1554_, uint8_t v_goalSolved_1555_, lean_object* v_stats_1556_, lean_object* v___y_1557_, lean_object* v___y_1558_, lean_object* v___y_1559_){
_start:
{
lean_object* v_fileName_1561_; lean_object* v_fileMap_1562_; lean_object* v___y_1564_; uint8_t v___x_1597_; lean_object* v___x_1598_; 
v_fileName_1561_ = lean_ctor_get(v___y_1558_, 0);
v_fileMap_1562_ = lean_ctor_get(v___y_1558_, 1);
v___x_1597_ = 0;
v___x_1598_ = l_Lean_Syntax_getPos_x3f(v_aesopStx_1554_, v___x_1597_);
if (lean_obj_tag(v___x_1598_) == 0)
{
lean_object* v___x_1599_; 
v___x_1599_ = lean_box(0);
v___y_1564_ = v___x_1599_;
goto v___jp_1563_;
}
else
{
lean_object* v_val_1600_; lean_object* v___x_1602_; uint8_t v_isShared_1603_; uint8_t v_isSharedCheck_1608_; 
v_val_1600_ = lean_ctor_get(v___x_1598_, 0);
v_isSharedCheck_1608_ = !lean_is_exclusive(v___x_1598_);
if (v_isSharedCheck_1608_ == 0)
{
v___x_1602_ = v___x_1598_;
v_isShared_1603_ = v_isSharedCheck_1608_;
goto v_resetjp_1601_;
}
else
{
lean_inc(v_val_1600_);
lean_dec(v___x_1598_);
v___x_1602_ = lean_box(0);
v_isShared_1603_ = v_isSharedCheck_1608_;
goto v_resetjp_1601_;
}
v_resetjp_1601_:
{
lean_object* v___x_1604_; lean_object* v___x_1606_; 
lean_inc_ref(v_fileMap_1562_);
v___x_1604_ = l_Lean_FileMap_toPosition(v_fileMap_1562_, v_val_1600_);
lean_dec(v_val_1600_);
if (v_isShared_1603_ == 0)
{
lean_ctor_set(v___x_1602_, 0, v___x_1604_);
v___x_1606_ = v___x_1602_;
goto v_reusejp_1605_;
}
else
{
lean_object* v_reuseFailAlloc_1607_; 
v_reuseFailAlloc_1607_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1607_, 0, v___x_1604_);
v___x_1606_ = v_reuseFailAlloc_1607_;
goto v_reusejp_1605_;
}
v_reusejp_1605_:
{
v___y_1564_ = v___x_1606_;
goto v___jp_1563_;
}
}
}
v___jp_1563_:
{
lean_object* v___x_1565_; 
v___x_1565_ = l_Lean_Elab_Term_getDeclName_x3f___redArg(v___y_1557_);
if (lean_obj_tag(v___x_1565_) == 0)
{
lean_object* v_a_1566_; lean_object* v___x_1567_; lean_object* v___x_1568_; 
v_a_1566_ = lean_ctor_get(v___x_1565_, 0);
lean_inc(v_a_1566_);
lean_dec_ref_known(v___x_1565_, 1);
v___x_1567_ = ((lean_object*)(lp_aesop_Aesop_StatsFileRecord_ofStats___at___00Aesop_appendStatsToStatsFileIfEnabled___at___00Aesop_evalAesop_spec__1_spec__2___redArg___closed__1));
v___x_1568_ = l_Lean_PrettyPrinter_ppCategory(v___x_1567_, v_aesopStx_1554_, v___y_1558_, v___y_1559_);
if (lean_obj_tag(v___x_1568_) == 0)
{
lean_object* v_a_1569_; lean_object* v___x_1571_; uint8_t v_isShared_1572_; uint8_t v_isSharedCheck_1580_; 
v_a_1569_ = lean_ctor_get(v___x_1568_, 0);
v_isSharedCheck_1580_ = !lean_is_exclusive(v___x_1568_);
if (v_isSharedCheck_1580_ == 0)
{
v___x_1571_ = v___x_1568_;
v_isShared_1572_ = v_isSharedCheck_1580_;
goto v_resetjp_1570_;
}
else
{
lean_inc(v_a_1569_);
lean_dec(v___x_1568_);
v___x_1571_ = lean_box(0);
v_isShared_1572_ = v_isSharedCheck_1580_;
goto v_resetjp_1570_;
}
v_resetjp_1570_:
{
lean_object* v___x_1573_; lean_object* v___x_1574_; lean_object* v_syntax_1575_; lean_object* v___x_1576_; lean_object* v___x_1578_; 
v___x_1573_ = lean_cstr_to_nat("100000000000");
v___x_1574_ = lean_unsigned_to_nat(0u);
v_syntax_1575_ = l_Std_Format_pretty(v_a_1569_, v___x_1573_, v___x_1574_, v___x_1574_);
lean_inc_ref(v_fileName_1561_);
v___x_1576_ = lean_alloc_ctor(0, 5, 1);
lean_ctor_set(v___x_1576_, 0, v_stats_1556_);
lean_ctor_set(v___x_1576_, 1, v_syntax_1575_);
lean_ctor_set(v___x_1576_, 2, v_fileName_1561_);
lean_ctor_set(v___x_1576_, 3, v___y_1564_);
lean_ctor_set(v___x_1576_, 4, v_a_1566_);
lean_ctor_set_uint8(v___x_1576_, sizeof(void*)*5, v_goalSolved_1555_);
if (v_isShared_1572_ == 0)
{
lean_ctor_set(v___x_1571_, 0, v___x_1576_);
v___x_1578_ = v___x_1571_;
goto v_reusejp_1577_;
}
else
{
lean_object* v_reuseFailAlloc_1579_; 
v_reuseFailAlloc_1579_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1579_, 0, v___x_1576_);
v___x_1578_ = v_reuseFailAlloc_1579_;
goto v_reusejp_1577_;
}
v_reusejp_1577_:
{
return v___x_1578_;
}
}
}
else
{
lean_object* v_a_1581_; lean_object* v___x_1583_; uint8_t v_isShared_1584_; uint8_t v_isSharedCheck_1588_; 
lean_dec(v_a_1566_);
lean_dec(v___y_1564_);
lean_dec_ref(v_stats_1556_);
v_a_1581_ = lean_ctor_get(v___x_1568_, 0);
v_isSharedCheck_1588_ = !lean_is_exclusive(v___x_1568_);
if (v_isSharedCheck_1588_ == 0)
{
v___x_1583_ = v___x_1568_;
v_isShared_1584_ = v_isSharedCheck_1588_;
goto v_resetjp_1582_;
}
else
{
lean_inc(v_a_1581_);
lean_dec(v___x_1568_);
v___x_1583_ = lean_box(0);
v_isShared_1584_ = v_isSharedCheck_1588_;
goto v_resetjp_1582_;
}
v_resetjp_1582_:
{
lean_object* v___x_1586_; 
if (v_isShared_1584_ == 0)
{
v___x_1586_ = v___x_1583_;
goto v_reusejp_1585_;
}
else
{
lean_object* v_reuseFailAlloc_1587_; 
v_reuseFailAlloc_1587_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1587_, 0, v_a_1581_);
v___x_1586_ = v_reuseFailAlloc_1587_;
goto v_reusejp_1585_;
}
v_reusejp_1585_:
{
return v___x_1586_;
}
}
}
}
else
{
lean_object* v_a_1589_; lean_object* v___x_1591_; uint8_t v_isShared_1592_; uint8_t v_isSharedCheck_1596_; 
lean_dec(v___y_1564_);
lean_dec_ref(v_stats_1556_);
lean_dec(v_aesopStx_1554_);
v_a_1589_ = lean_ctor_get(v___x_1565_, 0);
v_isSharedCheck_1596_ = !lean_is_exclusive(v___x_1565_);
if (v_isSharedCheck_1596_ == 0)
{
v___x_1591_ = v___x_1565_;
v_isShared_1592_ = v_isSharedCheck_1596_;
goto v_resetjp_1590_;
}
else
{
lean_inc(v_a_1589_);
lean_dec(v___x_1565_);
v___x_1591_ = lean_box(0);
v_isShared_1592_ = v_isSharedCheck_1596_;
goto v_resetjp_1590_;
}
v_resetjp_1590_:
{
lean_object* v___x_1594_; 
if (v_isShared_1592_ == 0)
{
v___x_1594_ = v___x_1591_;
goto v_reusejp_1593_;
}
else
{
lean_object* v_reuseFailAlloc_1595_; 
v_reuseFailAlloc_1595_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1595_, 0, v_a_1589_);
v___x_1594_ = v_reuseFailAlloc_1595_;
goto v_reusejp_1593_;
}
v_reusejp_1593_:
{
return v___x_1594_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_StatsFileRecord_ofStats___at___00Aesop_appendStatsToStatsFileIfEnabled___at___00Aesop_evalAesop_spec__1_spec__2___redArg___boxed(lean_object* v_aesopStx_1609_, lean_object* v_goalSolved_1610_, lean_object* v_stats_1611_, lean_object* v___y_1612_, lean_object* v___y_1613_, lean_object* v___y_1614_, lean_object* v___y_1615_){
_start:
{
uint8_t v_goalSolved_boxed_1616_; lean_object* v_res_1617_; 
v_goalSolved_boxed_1616_ = lean_unbox(v_goalSolved_1610_);
v_res_1617_ = lp_aesop_Aesop_StatsFileRecord_ofStats___at___00Aesop_appendStatsToStatsFileIfEnabled___at___00Aesop_evalAesop_spec__1_spec__2___redArg(v_aesopStx_1609_, v_goalSolved_boxed_1616_, v_stats_1611_, v___y_1612_, v___y_1613_, v___y_1614_);
lean_dec(v___y_1614_);
lean_dec_ref(v___y_1613_);
lean_dec_ref(v___y_1612_);
return v_res_1617_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_appendStatsToStatsFileIfEnabled___at___00Aesop_evalAesop_spec__1(lean_object* v_aesopStx_1618_, lean_object* v_stats_1619_, uint8_t v_allGoalsSolved_1620_, lean_object* v___y_1621_, lean_object* v___y_1622_, lean_object* v___y_1623_, lean_object* v___y_1624_, lean_object* v___y_1625_, lean_object* v___y_1626_, lean_object* v___y_1627_, lean_object* v___y_1628_){
_start:
{
lean_object* v_options_1630_; lean_object* v_ref_1631_; lean_object* v_a_1633_; lean_object* v___y_1640_; lean_object* v___x_1650_; lean_object* v_file_1651_; lean_object* v___x_1652_; uint8_t v___x_1653_; 
v_options_1630_ = lean_ctor_get(v___y_1627_, 2);
v_ref_1631_ = lean_ctor_get(v___y_1627_, 5);
v___x_1650_ = lp_aesop_Aesop_aesop_stats_file;
v_file_1651_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Main_0__Aesop_evalAesop_go_spec__2(v_options_1630_, v___x_1650_);
v___x_1652_ = ((lean_object*)(lp_aesop___private_Aesop_Main_0__Aesop_evalAesop_go___closed__0));
v___x_1653_ = lean_string_dec_eq(v_file_1651_, v___x_1652_);
if (v___x_1653_ == 0)
{
lean_object* v___x_1654_; 
v___x_1654_ = lp_aesop_Aesop_StatsFileRecord_ofStats___at___00Aesop_appendStatsToStatsFileIfEnabled___at___00Aesop_evalAesop_spec__1_spec__2___redArg(v_aesopStx_1618_, v_allGoalsSolved_1620_, v_stats_1619_, v___y_1623_, v___y_1627_, v___y_1628_);
if (lean_obj_tag(v___x_1654_) == 0)
{
lean_object* v_a_1655_; uint8_t v___x_1656_; lean_object* v___x_1657_; 
v_a_1655_ = lean_ctor_get(v___x_1654_, 0);
lean_inc(v_a_1655_);
lean_dec_ref_known(v___x_1654_, 1);
v___x_1656_ = 4;
v___x_1657_ = lean_io_prim_handle_mk(v_file_1651_, v___x_1656_);
lean_dec_ref(v_file_1651_);
if (lean_obj_tag(v___x_1657_) == 0)
{
lean_object* v_a_1658_; uint8_t v___x_1659_; lean_object* v___x_1660_; 
v_a_1658_ = lean_ctor_get(v___x_1657_, 0);
lean_inc(v_a_1658_);
lean_dec_ref_known(v___x_1657_, 1);
v___x_1659_ = 1;
v___x_1660_ = lean_io_prim_handle_lock(v_a_1658_, v___x_1659_);
if (lean_obj_tag(v___x_1660_) == 0)
{
lean_object* v___x_1661_; lean_object* v___x_1662_; lean_object* v_r_1663_; 
lean_dec_ref_known(v___x_1660_, 1);
v___x_1661_ = lp_aesop_Aesop_instToJsonStatsFileRecord_toJson(v_a_1655_);
v___x_1662_ = l_Lean_Json_compress(v___x_1661_);
v_r_1663_ = l_IO_FS_Handle_putStrLn(v_a_1658_, v___x_1662_);
if (lean_obj_tag(v_r_1663_) == 0)
{
lean_object* v_a_1664_; lean_object* v___x_1665_; 
v_a_1664_ = lean_ctor_get(v_r_1663_, 0);
lean_inc(v_a_1664_);
lean_dec_ref_known(v_r_1663_, 1);
v___x_1665_ = lean_io_prim_handle_unlock(v_a_1658_);
lean_dec(v_a_1658_);
if (lean_obj_tag(v___x_1665_) == 0)
{
lean_object* v___x_1667_; uint8_t v_isShared_1668_; uint8_t v_isSharedCheck_1672_; 
v_isSharedCheck_1672_ = !lean_is_exclusive(v___x_1665_);
if (v_isSharedCheck_1672_ == 0)
{
lean_object* v_unused_1673_; 
v_unused_1673_ = lean_ctor_get(v___x_1665_, 0);
lean_dec(v_unused_1673_);
v___x_1667_ = v___x_1665_;
v_isShared_1668_ = v_isSharedCheck_1672_;
goto v_resetjp_1666_;
}
else
{
lean_dec(v___x_1665_);
v___x_1667_ = lean_box(0);
v_isShared_1668_ = v_isSharedCheck_1672_;
goto v_resetjp_1666_;
}
v_resetjp_1666_:
{
lean_object* v___x_1670_; 
if (v_isShared_1668_ == 0)
{
lean_ctor_set(v___x_1667_, 0, v_a_1664_);
v___x_1670_ = v___x_1667_;
goto v_reusejp_1669_;
}
else
{
lean_object* v_reuseFailAlloc_1671_; 
v_reuseFailAlloc_1671_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1671_, 0, v_a_1664_);
v___x_1670_ = v_reuseFailAlloc_1671_;
goto v_reusejp_1669_;
}
v_reusejp_1669_:
{
return v___x_1670_;
}
}
}
else
{
lean_dec(v_a_1664_);
v___y_1640_ = v___x_1665_;
goto v___jp_1639_;
}
}
else
{
lean_object* v_a_1674_; lean_object* v___x_1675_; 
v_a_1674_ = lean_ctor_get(v_r_1663_, 0);
lean_inc(v_a_1674_);
lean_dec_ref_known(v_r_1663_, 1);
v___x_1675_ = lean_io_prim_handle_unlock(v_a_1658_);
lean_dec(v_a_1658_);
if (lean_obj_tag(v___x_1675_) == 0)
{
lean_dec_ref_known(v___x_1675_, 1);
v_a_1633_ = v_a_1674_;
goto v___jp_1632_;
}
else
{
lean_dec(v_a_1674_);
v___y_1640_ = v___x_1675_;
goto v___jp_1639_;
}
}
}
else
{
lean_dec(v_a_1658_);
lean_dec(v_a_1655_);
v___y_1640_ = v___x_1660_;
goto v___jp_1639_;
}
}
else
{
lean_object* v_a_1676_; 
lean_dec(v_a_1655_);
v_a_1676_ = lean_ctor_get(v___x_1657_, 0);
lean_inc(v_a_1676_);
lean_dec_ref_known(v___x_1657_, 1);
v_a_1633_ = v_a_1676_;
goto v___jp_1632_;
}
}
else
{
lean_object* v_a_1677_; lean_object* v___x_1679_; uint8_t v_isShared_1680_; uint8_t v_isSharedCheck_1684_; 
lean_dec_ref(v_file_1651_);
v_a_1677_ = lean_ctor_get(v___x_1654_, 0);
v_isSharedCheck_1684_ = !lean_is_exclusive(v___x_1654_);
if (v_isSharedCheck_1684_ == 0)
{
v___x_1679_ = v___x_1654_;
v_isShared_1680_ = v_isSharedCheck_1684_;
goto v_resetjp_1678_;
}
else
{
lean_inc(v_a_1677_);
lean_dec(v___x_1654_);
v___x_1679_ = lean_box(0);
v_isShared_1680_ = v_isSharedCheck_1684_;
goto v_resetjp_1678_;
}
v_resetjp_1678_:
{
lean_object* v___x_1682_; 
if (v_isShared_1680_ == 0)
{
v___x_1682_ = v___x_1679_;
goto v_reusejp_1681_;
}
else
{
lean_object* v_reuseFailAlloc_1683_; 
v_reuseFailAlloc_1683_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1683_, 0, v_a_1677_);
v___x_1682_ = v_reuseFailAlloc_1683_;
goto v_reusejp_1681_;
}
v_reusejp_1681_:
{
return v___x_1682_;
}
}
}
}
else
{
lean_object* v___x_1685_; lean_object* v___x_1686_; 
lean_dec_ref(v_file_1651_);
lean_dec_ref(v_stats_1619_);
lean_dec(v_aesopStx_1618_);
v___x_1685_ = lean_box(0);
v___x_1686_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1686_, 0, v___x_1685_);
return v___x_1686_;
}
v___jp_1632_:
{
lean_object* v___x_1634_; lean_object* v___x_1635_; lean_object* v___x_1636_; lean_object* v___x_1637_; lean_object* v___x_1638_; 
v___x_1634_ = lean_io_error_to_string(v_a_1633_);
v___x_1635_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1635_, 0, v___x_1634_);
v___x_1636_ = l_Lean_MessageData_ofFormat(v___x_1635_);
lean_inc(v_ref_1631_);
v___x_1637_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1637_, 0, v_ref_1631_);
lean_ctor_set(v___x_1637_, 1, v___x_1636_);
v___x_1638_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1638_, 0, v___x_1637_);
return v___x_1638_;
}
v___jp_1639_:
{
if (lean_obj_tag(v___y_1640_) == 0)
{
lean_object* v_a_1641_; lean_object* v___x_1643_; uint8_t v_isShared_1644_; uint8_t v_isSharedCheck_1648_; 
v_a_1641_ = lean_ctor_get(v___y_1640_, 0);
v_isSharedCheck_1648_ = !lean_is_exclusive(v___y_1640_);
if (v_isSharedCheck_1648_ == 0)
{
v___x_1643_ = v___y_1640_;
v_isShared_1644_ = v_isSharedCheck_1648_;
goto v_resetjp_1642_;
}
else
{
lean_inc(v_a_1641_);
lean_dec(v___y_1640_);
v___x_1643_ = lean_box(0);
v_isShared_1644_ = v_isSharedCheck_1648_;
goto v_resetjp_1642_;
}
v_resetjp_1642_:
{
lean_object* v___x_1646_; 
if (v_isShared_1644_ == 0)
{
v___x_1646_ = v___x_1643_;
goto v_reusejp_1645_;
}
else
{
lean_object* v_reuseFailAlloc_1647_; 
v_reuseFailAlloc_1647_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1647_, 0, v_a_1641_);
v___x_1646_ = v_reuseFailAlloc_1647_;
goto v_reusejp_1645_;
}
v_reusejp_1645_:
{
return v___x_1646_;
}
}
}
else
{
lean_object* v_a_1649_; 
v_a_1649_ = lean_ctor_get(v___y_1640_, 0);
lean_inc(v_a_1649_);
lean_dec_ref_known(v___y_1640_, 1);
v_a_1633_ = v_a_1649_;
goto v___jp_1632_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_appendStatsToStatsFileIfEnabled___at___00Aesop_evalAesop_spec__1___boxed(lean_object* v_aesopStx_1687_, lean_object* v_stats_1688_, lean_object* v_allGoalsSolved_1689_, lean_object* v___y_1690_, lean_object* v___y_1691_, lean_object* v___y_1692_, lean_object* v___y_1693_, lean_object* v___y_1694_, lean_object* v___y_1695_, lean_object* v___y_1696_, lean_object* v___y_1697_, lean_object* v___y_1698_){
_start:
{
uint8_t v_allGoalsSolved_boxed_1699_; lean_object* v_res_1700_; 
v_allGoalsSolved_boxed_1699_ = lean_unbox(v_allGoalsSolved_1689_);
v_res_1700_ = lp_aesop_Aesop_appendStatsToStatsFileIfEnabled___at___00Aesop_evalAesop_spec__1(v_aesopStx_1687_, v_stats_1688_, v_allGoalsSolved_boxed_1699_, v___y_1690_, v___y_1691_, v___y_1692_, v___y_1693_, v___y_1694_, v___y_1695_, v___y_1696_, v___y_1697_);
lean_dec(v___y_1697_);
lean_dec_ref(v___y_1696_);
lean_dec(v___y_1695_);
lean_dec_ref(v___y_1694_);
lean_dec(v___y_1693_);
lean_dec_ref(v___y_1692_);
lean_dec(v___y_1691_);
lean_dec_ref(v___y_1690_);
return v_res_1700_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_evalAesop___lam__0(lean_object* v___x_1701_, lean_object* v_stx_1702_, lean_object* v_a_1703_, lean_object* v___y_1704_, lean_object* v___y_1705_, lean_object* v___y_1706_, lean_object* v___y_1707_, lean_object* v___y_1708_, lean_object* v___y_1709_, lean_object* v___y_1710_, lean_object* v___y_1711_){
_start:
{
lean_object* v___x_1713_; lean_object* v___x_1714_; 
v___x_1713_ = lean_st_mk_ref(v___x_1701_);
lean_inc(v_stx_1702_);
v___x_1714_ = lp_aesop___private_Aesop_Main_0__Aesop_evalAesop_go(v_stx_1702_, v_a_1703_, v___x_1713_, v___y_1704_, v___y_1705_, v___y_1706_, v___y_1707_, v___y_1708_, v___y_1709_, v___y_1710_, v___y_1711_);
if (lean_obj_tag(v___x_1714_) == 0)
{
lean_object* v_a_1715_; lean_object* v___x_1716_; lean_object* v___x_1717_; lean_object* v___x_1718_; 
v_a_1715_ = lean_ctor_get(v___x_1714_, 0);
lean_inc(v_a_1715_);
lean_dec_ref_known(v___x_1714_, 1);
v___x_1716_ = lean_st_ref_get(v___x_1713_);
lean_dec(v___x_1713_);
v___x_1717_ = lp_aesop_Aesop_TraceOption_stats;
lean_inc(v___x_1716_);
v___x_1718_ = lp_aesop_Aesop_Stats_trace(v___x_1716_, v___x_1717_, v___y_1710_, v___y_1711_);
if (lean_obj_tag(v___x_1718_) == 0)
{
lean_object* v___x_1719_; 
lean_dec_ref_known(v___x_1718_, 1);
lean_inc(v___x_1716_);
lean_inc(v_stx_1702_);
v___x_1719_ = lp_aesop_Aesop_recordStatsForCurrentFileIfEnabled___at___00Aesop_evalAesop_spec__0(v_stx_1702_, v___x_1716_, v___y_1704_, v___y_1705_, v___y_1706_, v___y_1707_, v___y_1708_, v___y_1709_, v___y_1710_, v___y_1711_);
if (lean_obj_tag(v___x_1719_) == 0)
{
uint8_t v___x_1720_; lean_object* v___x_1721_; 
lean_dec_ref_known(v___x_1719_, 1);
v___x_1720_ = lean_unbox(v_a_1715_);
lean_dec(v_a_1715_);
v___x_1721_ = lp_aesop_Aesop_appendStatsToStatsFileIfEnabled___at___00Aesop_evalAesop_spec__1(v_stx_1702_, v___x_1716_, v___x_1720_, v___y_1704_, v___y_1705_, v___y_1706_, v___y_1707_, v___y_1708_, v___y_1709_, v___y_1710_, v___y_1711_);
return v___x_1721_;
}
else
{
lean_dec(v___x_1716_);
lean_dec(v_a_1715_);
lean_dec(v_stx_1702_);
return v___x_1719_;
}
}
else
{
lean_dec(v___x_1716_);
lean_dec(v_a_1715_);
lean_dec(v_stx_1702_);
return v___x_1718_;
}
}
else
{
lean_object* v_a_1722_; lean_object* v___x_1724_; uint8_t v_isShared_1725_; uint8_t v_isSharedCheck_1729_; 
lean_dec(v___x_1713_);
lean_dec(v_stx_1702_);
v_a_1722_ = lean_ctor_get(v___x_1714_, 0);
v_isSharedCheck_1729_ = !lean_is_exclusive(v___x_1714_);
if (v_isSharedCheck_1729_ == 0)
{
v___x_1724_ = v___x_1714_;
v_isShared_1725_ = v_isSharedCheck_1729_;
goto v_resetjp_1723_;
}
else
{
lean_inc(v_a_1722_);
lean_dec(v___x_1714_);
v___x_1724_ = lean_box(0);
v_isShared_1725_ = v_isSharedCheck_1729_;
goto v_resetjp_1723_;
}
v_resetjp_1723_:
{
lean_object* v___x_1727_; 
if (v_isShared_1725_ == 0)
{
v___x_1727_ = v___x_1724_;
goto v_reusejp_1726_;
}
else
{
lean_object* v_reuseFailAlloc_1728_; 
v_reuseFailAlloc_1728_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1728_, 0, v_a_1722_);
v___x_1727_ = v_reuseFailAlloc_1728_;
goto v_reusejp_1726_;
}
v_reusejp_1726_:
{
return v___x_1727_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_evalAesop___lam__0___boxed(lean_object* v___x_1730_, lean_object* v_stx_1731_, lean_object* v_a_1732_, lean_object* v___y_1733_, lean_object* v___y_1734_, lean_object* v___y_1735_, lean_object* v___y_1736_, lean_object* v___y_1737_, lean_object* v___y_1738_, lean_object* v___y_1739_, lean_object* v___y_1740_, lean_object* v___y_1741_){
_start:
{
lean_object* v_res_1742_; 
v_res_1742_ = lp_aesop_Aesop_evalAesop___lam__0(v___x_1730_, v_stx_1731_, v_a_1732_, v___y_1733_, v___y_1734_, v___y_1735_, v___y_1736_, v___y_1737_, v___y_1738_, v___y_1739_, v___y_1740_);
lean_dec(v___y_1740_);
lean_dec_ref(v___y_1739_);
lean_dec(v___y_1738_);
lean_dec_ref(v___y_1737_);
lean_dec(v___y_1736_);
lean_dec_ref(v___y_1735_);
lean_dec(v___y_1734_);
lean_dec_ref(v___y_1733_);
return v_res_1742_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_evalAesop___lam__1(lean_object* v_stx_1743_, lean_object* v___y_1744_, lean_object* v___y_1745_, lean_object* v___y_1746_, lean_object* v___y_1747_, lean_object* v___y_1748_, lean_object* v___y_1749_, lean_object* v___y_1750_, lean_object* v___y_1751_){
_start:
{
lean_object* v___x_1753_; 
v___x_1753_ = l_Lean_Elab_Tactic_getMainGoal___redArg(v___y_1745_, v___y_1748_, v___y_1749_, v___y_1750_, v___y_1751_);
if (lean_obj_tag(v___x_1753_) == 0)
{
lean_object* v_a_1754_; lean_object* v___x_1755_; lean_object* v___f_1756_; lean_object* v___x_1757_; 
v_a_1754_ = lean_ctor_get(v___x_1753_, 0);
lean_inc_n(v_a_1754_, 2);
lean_dec_ref_known(v___x_1753_, 1);
v___x_1755_ = lp_aesop_Aesop_Stats_empty;
v___f_1756_ = lean_alloc_closure((void*)(lp_aesop_Aesop_evalAesop___lam__0___boxed), 12, 3);
lean_closure_set(v___f_1756_, 0, v___x_1755_);
lean_closure_set(v___f_1756_, 1, v_stx_1743_);
lean_closure_set(v___f_1756_, 2, v_a_1754_);
v___x_1757_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_evalAesop_spec__2___redArg(v_a_1754_, v___f_1756_, v___y_1744_, v___y_1745_, v___y_1746_, v___y_1747_, v___y_1748_, v___y_1749_, v___y_1750_, v___y_1751_);
return v___x_1757_;
}
else
{
lean_object* v_a_1758_; lean_object* v___x_1760_; uint8_t v_isShared_1761_; uint8_t v_isSharedCheck_1765_; 
lean_dec(v_stx_1743_);
v_a_1758_ = lean_ctor_get(v___x_1753_, 0);
v_isSharedCheck_1765_ = !lean_is_exclusive(v___x_1753_);
if (v_isSharedCheck_1765_ == 0)
{
v___x_1760_ = v___x_1753_;
v_isShared_1761_ = v_isSharedCheck_1765_;
goto v_resetjp_1759_;
}
else
{
lean_inc(v_a_1758_);
lean_dec(v___x_1753_);
v___x_1760_ = lean_box(0);
v_isShared_1761_ = v_isSharedCheck_1765_;
goto v_resetjp_1759_;
}
v_resetjp_1759_:
{
lean_object* v___x_1763_; 
if (v_isShared_1761_ == 0)
{
v___x_1763_ = v___x_1760_;
goto v_reusejp_1762_;
}
else
{
lean_object* v_reuseFailAlloc_1764_; 
v_reuseFailAlloc_1764_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1764_, 0, v_a_1758_);
v___x_1763_ = v_reuseFailAlloc_1764_;
goto v_reusejp_1762_;
}
v_reusejp_1762_:
{
return v___x_1763_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_evalAesop___lam__1___boxed(lean_object* v_stx_1766_, lean_object* v___y_1767_, lean_object* v___y_1768_, lean_object* v___y_1769_, lean_object* v___y_1770_, lean_object* v___y_1771_, lean_object* v___y_1772_, lean_object* v___y_1773_, lean_object* v___y_1774_, lean_object* v___y_1775_){
_start:
{
lean_object* v_res_1776_; 
v_res_1776_ = lp_aesop_Aesop_evalAesop___lam__1(v_stx_1766_, v___y_1767_, v___y_1768_, v___y_1769_, v___y_1770_, v___y_1771_, v___y_1772_, v___y_1773_, v___y_1774_);
lean_dec(v___y_1774_);
lean_dec_ref(v___y_1773_);
lean_dec(v___y_1772_);
lean_dec_ref(v___y_1771_);
lean_dec(v___y_1770_);
lean_dec_ref(v___y_1769_);
lean_dec(v___y_1768_);
lean_dec_ref(v___y_1767_);
return v_res_1776_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_evalAesop(lean_object* v_stx_1778_, lean_object* v_a_1779_, lean_object* v_a_1780_, lean_object* v_a_1781_, lean_object* v_a_1782_, lean_object* v_a_1783_, lean_object* v_a_1784_, lean_object* v_a_1785_, lean_object* v_a_1786_){
_start:
{
lean_object* v_options_1788_; lean_object* v___f_1789_; lean_object* v___x_1790_; lean_object* v___x_1791_; lean_object* v___x_1792_; 
v_options_1788_ = lean_ctor_get(v_a_1785_, 2);
v___f_1789_ = lean_alloc_closure((void*)(lp_aesop_Aesop_evalAesop___lam__1___boxed), 10, 1);
lean_closure_set(v___f_1789_, 0, v_stx_1778_);
v___x_1790_ = ((lean_object*)(lp_aesop_Aesop_evalAesop___closed__0));
v___x_1791_ = lean_box(0);
v___x_1792_ = lp_aesop_Lean_profileitM___at___00Aesop_evalAesop_spec__3___redArg(v___x_1790_, v_options_1788_, v___f_1789_, v___x_1791_, v_a_1779_, v_a_1780_, v_a_1781_, v_a_1782_, v_a_1783_, v_a_1784_, v_a_1785_, v_a_1786_);
return v___x_1792_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_evalAesop___boxed(lean_object* v_stx_1793_, lean_object* v_a_1794_, lean_object* v_a_1795_, lean_object* v_a_1796_, lean_object* v_a_1797_, lean_object* v_a_1798_, lean_object* v_a_1799_, lean_object* v_a_1800_, lean_object* v_a_1801_, lean_object* v_a_1802_){
_start:
{
lean_object* v_res_1803_; 
v_res_1803_ = lp_aesop_Aesop_evalAesop(v_stx_1793_, v_a_1794_, v_a_1795_, v_a_1796_, v_a_1797_, v_a_1798_, v_a_1799_, v_a_1800_, v_a_1801_);
lean_dec(v_a_1801_);
lean_dec_ref(v_a_1800_);
lean_dec(v_a_1799_);
lean_dec_ref(v_a_1798_);
lean_dec(v_a_1797_);
lean_dec_ref(v_a_1796_);
lean_dec(v_a_1795_);
lean_dec_ref(v_a_1794_);
return v_res_1803_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_StatsExtensionEntry_forCurrentFile___at___00Aesop_recordStatsForCurrentFileIfEnabled___at___00Aesop_evalAesop_spec__0_spec__0(lean_object* v_stx_1804_, lean_object* v_stats_1805_, lean_object* v___y_1806_, lean_object* v___y_1807_, lean_object* v___y_1808_, lean_object* v___y_1809_, lean_object* v___y_1810_, lean_object* v___y_1811_, lean_object* v___y_1812_, lean_object* v___y_1813_){
_start:
{
lean_object* v___x_1815_; 
v___x_1815_ = lp_aesop_Aesop_StatsExtensionEntry_forCurrentFile___at___00Aesop_recordStatsForCurrentFileIfEnabled___at___00Aesop_evalAesop_spec__0_spec__0___redArg(v_stx_1804_, v_stats_1805_, v___y_1812_);
return v___x_1815_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_StatsExtensionEntry_forCurrentFile___at___00Aesop_recordStatsForCurrentFileIfEnabled___at___00Aesop_evalAesop_spec__0_spec__0___boxed(lean_object* v_stx_1816_, lean_object* v_stats_1817_, lean_object* v___y_1818_, lean_object* v___y_1819_, lean_object* v___y_1820_, lean_object* v___y_1821_, lean_object* v___y_1822_, lean_object* v___y_1823_, lean_object* v___y_1824_, lean_object* v___y_1825_, lean_object* v___y_1826_){
_start:
{
lean_object* v_res_1827_; 
v_res_1827_ = lp_aesop_Aesop_StatsExtensionEntry_forCurrentFile___at___00Aesop_recordStatsForCurrentFileIfEnabled___at___00Aesop_evalAesop_spec__0_spec__0(v_stx_1816_, v_stats_1817_, v___y_1818_, v___y_1819_, v___y_1820_, v___y_1821_, v___y_1822_, v___y_1823_, v___y_1824_, v___y_1825_);
lean_dec(v___y_1825_);
lean_dec_ref(v___y_1824_);
lean_dec(v___y_1823_);
lean_dec_ref(v___y_1822_);
lean_dec(v___y_1821_);
lean_dec_ref(v___y_1820_);
lean_dec(v___y_1819_);
lean_dec_ref(v___y_1818_);
return v_res_1827_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_StatsFileRecord_ofStats___at___00Aesop_appendStatsToStatsFileIfEnabled___at___00Aesop_evalAesop_spec__1_spec__2(lean_object* v_aesopStx_1828_, uint8_t v_goalSolved_1829_, lean_object* v_stats_1830_, lean_object* v___y_1831_, lean_object* v___y_1832_, lean_object* v___y_1833_, lean_object* v___y_1834_, lean_object* v___y_1835_, lean_object* v___y_1836_, lean_object* v___y_1837_, lean_object* v___y_1838_){
_start:
{
lean_object* v___x_1840_; 
v___x_1840_ = lp_aesop_Aesop_StatsFileRecord_ofStats___at___00Aesop_appendStatsToStatsFileIfEnabled___at___00Aesop_evalAesop_spec__1_spec__2___redArg(v_aesopStx_1828_, v_goalSolved_1829_, v_stats_1830_, v___y_1833_, v___y_1837_, v___y_1838_);
return v___x_1840_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_StatsFileRecord_ofStats___at___00Aesop_appendStatsToStatsFileIfEnabled___at___00Aesop_evalAesop_spec__1_spec__2___boxed(lean_object* v_aesopStx_1841_, lean_object* v_goalSolved_1842_, lean_object* v_stats_1843_, lean_object* v___y_1844_, lean_object* v___y_1845_, lean_object* v___y_1846_, lean_object* v___y_1847_, lean_object* v___y_1848_, lean_object* v___y_1849_, lean_object* v___y_1850_, lean_object* v___y_1851_, lean_object* v___y_1852_){
_start:
{
uint8_t v_goalSolved_boxed_1853_; lean_object* v_res_1854_; 
v_goalSolved_boxed_1853_ = lean_unbox(v_goalSolved_1842_);
v_res_1854_ = lp_aesop_Aesop_StatsFileRecord_ofStats___at___00Aesop_appendStatsToStatsFileIfEnabled___at___00Aesop_evalAesop_spec__1_spec__2(v_aesopStx_1841_, v_goalSolved_boxed_1853_, v_stats_1843_, v___y_1844_, v___y_1845_, v___y_1846_, v___y_1847_, v___y_1848_, v___y_1849_, v___y_1850_, v___y_1851_);
lean_dec(v___y_1851_);
lean_dec_ref(v___y_1850_);
lean_dec(v___y_1849_);
lean_dec_ref(v___y_1848_);
lean_dec(v___y_1847_);
lean_dec_ref(v___y_1846_);
lean_dec(v___y_1845_);
lean_dec_ref(v___y_1844_);
return v_res_1854_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
void lean_initialize_runtime_module();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_Main(uint8_t builtin) {
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
lean_object* runtime_initialize_aesop_Aesop_Search_Main(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Frontend_Tactic(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Stats_Extension(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Stats_File(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_Main(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Search_Main(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Frontend_Tactic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Stats_Extension(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Stats_File(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Search_Main(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Frontend_Tactic(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Stats_Extension(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Stats_File(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_Main(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Search_Main(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Frontend_Tactic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Stats_Extension(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Stats_File(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Main(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_Main(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_Main(builtin);
}
#ifdef __cplusplus
}
#endif
