// Lean compiler output
// Module: Aesop.Tree.TreeM
// Imports: public import Init public meta import Init public import Aesop.RuleSet public import Aesop.Tree.Data import Aesop.Forward.State.Initial
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
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_instMonadEIO(lean_object*);
lean_object* l_StateRefT_x27_instMonad___redArg(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* lp_aesop_Aesop_GoalId_succ(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
extern lean_object* l_Lean_Core_instMonadQuotationCoreM;
lean_object* l_StateRefT_x27_lift___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_StateRefT_x27_instMonadFunctor___aux__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_instMonadQuotationOfMonadFunctorOfMonadLift___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instMonadLift___lam__0___boxed(lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instMonadFunctor___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
double lean_float_of_nat(lean_object*);
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
extern lean_object* l_Lean_instMonadExceptOfExceptionCoreM;
lean_object* l_StateRefT_x27_instMonadExceptOf___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_StateRefT_x27_instMonadExceptOf___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instMonadExceptOf___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instMonadExceptOf___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* lp_aesop_Aesop_treeImpl;
lean_object* lean_st_mk_ref(lean_object*);
lean_object* l_Lean_MVarId_getMVarDependencies(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* lp_aesop_Aesop_GoalId_zero;
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
size_t lean_usize_add(size_t, size_t);
lean_object* lp_aesop_Aesop_ForwardRuleMatches_ofArray(lean_object*);
extern double lp_aesop_Aesop_Percent_hundred;
extern lean_object* lp_aesop_Aesop_Iteration_none;
lean_object* l_Subarray_empty(lean_object*);
lean_object* l_Lean_Meta_saveState___redArg(lean_object*, lean_object*);
extern lean_object* lp_aesop_Aesop_GoalId_one;
extern lean_object* lp_aesop_Aesop_RappId_zero;
lean_object* lean_mk_array(lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_LocalRuleSet_mkInitialForwardState(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* lp_aesop_Aesop_TraceOption_forward;
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_Name_append(lean_object*, lean_object*);
uint8_t l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(lean_object*, lean_object*, lean_object*);
lean_object* lean_io_mono_nanos_now();
double lean_float_div(double, double);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_PersistentArray_toArray___redArg(lean_object*);
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* l_Lean_PersistentArray_push___redArg(lean_object*, lean_object*);
extern lean_object* l_Lean_trace_profiler;
lean_object* l_Lean_PersistentArray_append___redArg(lean_object*, lean_object*);
double lean_float_sub(double, double);
uint8_t lean_float_decLt(double, double);
extern lean_object* l_Lean_trace_profiler_useHeartbeats;
extern lean_object* l_Lean_trace_profiler_threshold;
lean_object* lean_io_get_num_heartbeats();
lean_object* lean_nat_add(lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* l_Nat_reprFast(lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* l_Lean_Meta_instMonadMetaM___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Core_instMonadCoreM___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Core_instMonadCoreM___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instFunctorOfMonad___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instFunctorOfMonad___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instApplicativeOfMonad___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instApplicativeOfMonad___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instApplicativeOfMonad___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_instMonadMetaM___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Meta_instAddMessageContextMetaM;
lean_object* l_Lean_instAddErrorMessageContextOfAddMessageContextOfMonad___redArg(lean_object*, lean_object*);
lean_object* l_Lean_throwError___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_RappId_succ(lean_object*);
lean_object* l_ReaderT_bind___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instMonad___redArg(lean_object*);
static lean_once_cell_t lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_mkInitialTree_spec__1___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_mkInitialTree_spec__1___redArg___closed__0;
static lean_once_cell_t lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_mkInitialTree_spec__1___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_mkInitialTree_spec__1___redArg___closed__1;
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_mkInitialTree_spec__1___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_mkInitialTree_spec__1___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_mkInitialTree_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_mkInitialTree_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_Option_get___at___00Aesop_mkInitialTree_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00Aesop_mkInitialTree_spec__2___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_mkInitialTree___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_mkInitialTree___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00Aesop_UnorderedArraySet_ofHashSet___at___00Aesop_mkInitialTree_spec__0_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_UnorderedArraySet_ofHashSet___at___00Aesop_mkInitialTree_spec__0_spec__1(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_UnorderedArraySet_ofHashSet___at___00Aesop_mkInitialTree_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_ofHashSet___at___00Aesop_mkInitialTree_spec__0(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_ofHashSet___at___00Aesop_mkInitialTree_spec__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_mkInitialTree_spec__3_spec__5_spec__6(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_mkInitialTree_spec__3_spec__5_spec__6___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_mkInitialTree_spec__3_spec__5_spec__7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_mkInitialTree_spec__3_spec__5_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_mkInitialTree_spec__3_spec__5___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_mkInitialTree_spec__3_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_mkInitialTree_spec__3_spec__7(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_mkInitialTree_spec__3_spec__7___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_mkInitialTree_spec__3_spec__6___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_mkInitialTree_spec__3_spec__6___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_mkInitialTree_spec__3_spec__8(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_mkInitialTree_spec__3_spec__8___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_mkInitialTree_spec__3___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_mkInitialTree_spec__3___closed__0;
static const lean_string_object lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_mkInitialTree_spec__3___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 54, .m_capacity = 54, .m_length = 53, .m_data = "<exception thrown while producing trace node message>"};
static const lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_mkInitialTree_spec__3___closed__1 = (const lean_object*)&lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_mkInitialTree_spec__3___closed__1_value;
static lean_once_cell_t lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_mkInitialTree_spec__3___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_mkInitialTree_spec__3___closed__2;
static lean_once_cell_t lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_mkInitialTree_spec__3___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_mkInitialTree_spec__3___closed__3;
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_mkInitialTree_spec__3(lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_mkInitialTree_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_aesop_Aesop_mkInitialTree___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Aesop_mkInitialTree___closed__0 = (const lean_object*)&lp_aesop_Aesop_mkInitialTree___closed__0_value;
static const lean_ctor_object lp_aesop_Aesop_mkInitialTree___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_mkInitialTree___closed__0_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_aesop_Aesop_mkInitialTree___closed__1 = (const lean_object*)&lp_aesop_Aesop_mkInitialTree___closed__1_value;
static lean_once_cell_t lp_aesop_Aesop_mkInitialTree___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_mkInitialTree___closed__2;
static lean_once_cell_t lp_aesop_Aesop_mkInitialTree___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_mkInitialTree___closed__3;
static lean_once_cell_t lp_aesop_Aesop_mkInitialTree___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_mkInitialTree___closed__4;
static const lean_string_object lp_aesop_Aesop_mkInitialTree___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 31, .m_capacity = 31, .m_length = 30, .m_data = "building initial forward state"};
static const lean_object* lp_aesop_Aesop_mkInitialTree___closed__5 = (const lean_object*)&lp_aesop_Aesop_mkInitialTree___closed__5_value;
static lean_once_cell_t lp_aesop_Aesop_mkInitialTree___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_mkInitialTree___closed__6;
static lean_once_cell_t lp_aesop_Aesop_mkInitialTree___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_mkInitialTree___closed__7;
static const lean_string_object lp_aesop_Aesop_mkInitialTree___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_aesop_Aesop_mkInitialTree___closed__8 = (const lean_object*)&lp_aesop_Aesop_mkInitialTree___closed__8_value;
static const lean_string_object lp_aesop_Aesop_mkInitialTree___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_aesop_Aesop_mkInitialTree___closed__9 = (const lean_object*)&lp_aesop_Aesop_mkInitialTree___closed__9_value;
static const lean_ctor_object lp_aesop_Aesop_mkInitialTree___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_mkInitialTree___closed__9_value),LEAN_SCALAR_PTR_LITERAL(212, 145, 141, 177, 67, 149, 127, 197)}};
static const lean_object* lp_aesop_Aesop_mkInitialTree___closed__10 = (const lean_object*)&lp_aesop_Aesop_mkInitialTree___closed__10_value;
static lean_once_cell_t lp_aesop_Aesop_mkInitialTree___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_aesop_Aesop_mkInitialTree___closed__11;
LEAN_EXPORT lean_object* lp_aesop_Aesop_mkInitialTree(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_mkInitialTree___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_mkInitialTree_spec__3_spec__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_mkInitialTree_spec__3_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_mkInitialTree_spec__3_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_mkInitialTree_spec__3_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop_Aesop_TreeM_instMonad___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_TreeM_instMonad___closed__0;
static lean_once_cell_t lp_aesop_Aesop_TreeM_instMonad___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_TreeM_instMonad___closed__1;
static const lean_closure_object lp_aesop_Aesop_TreeM_instMonad___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Core_instMonadCoreM___lam__0___boxed, .m_arity = 5, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_TreeM_instMonad___closed__2 = (const lean_object*)&lp_aesop_Aesop_TreeM_instMonad___closed__2_value;
static const lean_closure_object lp_aesop_Aesop_TreeM_instMonad___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Core_instMonadCoreM___lam__1___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_TreeM_instMonad___closed__3 = (const lean_object*)&lp_aesop_Aesop_TreeM_instMonad___closed__3_value;
static const lean_closure_object lp_aesop_Aesop_TreeM_instMonad___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Meta_instMonadMetaM___lam__0___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_TreeM_instMonad___closed__4 = (const lean_object*)&lp_aesop_Aesop_TreeM_instMonad___closed__4_value;
static const lean_closure_object lp_aesop_Aesop_TreeM_instMonad___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Meta_instMonadMetaM___lam__1___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_TreeM_instMonad___closed__5 = (const lean_object*)&lp_aesop_Aesop_TreeM_instMonad___closed__5_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_TreeM_instMonad;
LEAN_EXPORT lean_object* lp_aesop_Aesop_TreeM_instMonadStateOfTree___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TreeM_instMonadStateOfTree___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TreeM_instMonadStateOfTree___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TreeM_instMonadStateOfTree___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TreeM_instMonadStateOfTree___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TreeM_instMonadStateOfTree___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TreeM_instMonadStateOfTree___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TreeM_instMonadStateOfTree___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_TreeM_instMonadStateOfTree___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_TreeM_instMonadStateOfTree___lam__0___boxed, .m_arity = 8, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_TreeM_instMonadStateOfTree___closed__0 = (const lean_object*)&lp_aesop_Aesop_TreeM_instMonadStateOfTree___closed__0_value;
static const lean_closure_object lp_aesop_Aesop_TreeM_instMonadStateOfTree___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_TreeM_instMonadStateOfTree___lam__1___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_TreeM_instMonadStateOfTree___closed__1 = (const lean_object*)&lp_aesop_Aesop_TreeM_instMonadStateOfTree___closed__1_value;
static const lean_closure_object lp_aesop_Aesop_TreeM_instMonadStateOfTree___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_TreeM_instMonadStateOfTree___lam__2___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_TreeM_instMonadStateOfTree___closed__2 = (const lean_object*)&lp_aesop_Aesop_TreeM_instMonadStateOfTree___closed__2_value;
static const lean_closure_object lp_aesop_Aesop_TreeM_instMonadStateOfTree___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_TreeM_instMonadStateOfTree___lam__3___boxed, .m_arity = 10, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_TreeM_instMonadStateOfTree___closed__3 = (const lean_object*)&lp_aesop_Aesop_TreeM_instMonadStateOfTree___closed__3_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_TreeM_instMonadStateOfTree;
static lean_once_cell_t lp_aesop_Aesop_TreeM_instInhabited___lam__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_TreeM_instInhabited___lam__0___closed__0;
static lean_once_cell_t lp_aesop_Aesop_TreeM_instInhabited___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_TreeM_instInhabited___lam__0___closed__1;
static lean_once_cell_t lp_aesop_Aesop_TreeM_instInhabited___lam__0___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_TreeM_instInhabited___lam__0___closed__2;
static lean_once_cell_t lp_aesop_Aesop_TreeM_instInhabited___lam__0___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_TreeM_instInhabited___lam__0___closed__3;
static lean_once_cell_t lp_aesop_Aesop_TreeM_instInhabited___lam__0___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_TreeM_instInhabited___lam__0___closed__4;
static lean_once_cell_t lp_aesop_Aesop_TreeM_instInhabited___lam__0___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_TreeM_instInhabited___lam__0___closed__5;
static const lean_closure_object lp_aesop_Aesop_TreeM_instInhabited___lam__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_ReaderT_instMonadFunctor___lam__0, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_TreeM_instInhabited___lam__0___closed__6 = (const lean_object*)&lp_aesop_Aesop_TreeM_instInhabited___lam__0___closed__6_value;
static const lean_closure_object lp_aesop_Aesop_TreeM_instInhabited___lam__0___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_ReaderT_instMonadLift___lam__0___boxed, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_TreeM_instInhabited___lam__0___closed__7 = (const lean_object*)&lp_aesop_Aesop_TreeM_instInhabited___lam__0___closed__7_value;
static const lean_closure_object lp_aesop_Aesop_TreeM_instInhabited___lam__0___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*3, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_StateRefT_x27_instMonadFunctor___aux__1___boxed, .m_arity = 7, .m_num_fixed = 3, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_aesop_Aesop_TreeM_instInhabited___lam__0___closed__8 = (const lean_object*)&lp_aesop_Aesop_TreeM_instInhabited___lam__0___closed__8_value;
static const lean_closure_object lp_aesop_Aesop_TreeM_instInhabited___lam__0___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*3, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_StateRefT_x27_lift___boxed, .m_arity = 6, .m_num_fixed = 3, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_aesop_Aesop_TreeM_instInhabited___lam__0___closed__9 = (const lean_object*)&lp_aesop_Aesop_TreeM_instInhabited___lam__0___closed__9_value;
static lean_once_cell_t lp_aesop_Aesop_TreeM_instInhabited___lam__0___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_TreeM_instInhabited___lam__0___closed__10;
static lean_once_cell_t lp_aesop_Aesop_TreeM_instInhabited___lam__0___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_TreeM_instInhabited___lam__0___closed__11;
static const lean_string_object lp_aesop_Aesop_TreeM_instInhabited___lam__0___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "failed"};
static const lean_object* lp_aesop_Aesop_TreeM_instInhabited___lam__0___closed__12 = (const lean_object*)&lp_aesop_Aesop_TreeM_instInhabited___lam__0___closed__12_value;
static lean_once_cell_t lp_aesop_Aesop_TreeM_instInhabited___lam__0___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_TreeM_instInhabited___lam__0___closed__13;
LEAN_EXPORT lean_object* lp_aesop_Aesop_TreeM_instInhabited___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TreeM_instInhabited___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_TreeM_instInhabited___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_TreeM_instInhabited___lam__0___boxed, .m_arity = 8, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_TreeM_instInhabited___closed__0 = (const lean_object*)&lp_aesop_Aesop_TreeM_instInhabited___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_TreeM_instInhabited(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TreeM_run_x27___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TreeM_run_x27___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TreeM_run_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TreeM_run_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_getRootMVarCluster___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_getRootMVarCluster___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_getRootMVarCluster(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_getRootMVarCluster___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_getRootMetaState___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_getRootMetaState___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_getRootMetaState(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_getRootMetaState___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_getRootGoal_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_getRootGoal_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_getRootGoal___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 73, .m_capacity = 73, .m_length = 72, .m_data = "aesop: internal error: unexpected number of goals in root mvar cluster: "};
static const lean_object* lp_aesop_Aesop_getRootGoal___closed__0 = (const lean_object*)&lp_aesop_Aesop_getRootGoal___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_getRootGoal___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_getRootGoal___closed__1;
LEAN_EXPORT lean_object* lp_aesop_Aesop_getRootGoal(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_getRootGoal___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_getRootGoal_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_getRootGoal_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_getRootMVarId(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_getRootMVarId___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_incrementNumGoals___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_incrementNumGoals___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_incrementNumGoals(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_incrementNumGoals___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_incrementNumRapps___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_incrementNumRapps___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_incrementNumRapps(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_incrementNumRapps___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_getAllIntroducedMVars___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_getAllIntroducedMVars___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_getAllIntroducedMVars(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_getAllIntroducedMVars___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_getAndIncrementNextGoalId___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_getAndIncrementNextGoalId___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_getAndIncrementNextGoalId(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_getAndIncrementNextGoalId___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_getAndIncrementNextRappId___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_getAndIncrementNextRappId___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_getAndIncrementNextRappId(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_getAndIncrementNextRappId___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_mkInitialTree_spec__1___redArg___closed__0(void){
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
static lean_object* _init_lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_mkInitialTree_spec__1___redArg___closed__1(void){
_start:
{
size_t v___x_4_; lean_object* v___x_5_; lean_object* v___x_6_; lean_object* v___x_7_; lean_object* v___x_8_; lean_object* v___x_9_; 
v___x_4_ = ((size_t)5ULL);
v___x_5_ = lean_unsigned_to_nat(0u);
v___x_6_ = lean_unsigned_to_nat(32u);
v___x_7_ = lean_mk_empty_array_with_capacity(v___x_6_);
v___x_8_ = lean_obj_once(&lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_mkInitialTree_spec__1___redArg___closed__0, &lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_mkInitialTree_spec__1___redArg___closed__0_once, _init_lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_mkInitialTree_spec__1___redArg___closed__0);
v___x_9_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_9_, 0, v___x_8_);
lean_ctor_set(v___x_9_, 1, v___x_7_);
lean_ctor_set(v___x_9_, 2, v___x_5_);
lean_ctor_set(v___x_9_, 3, v___x_5_);
lean_ctor_set_usize(v___x_9_, 4, v___x_4_);
return v___x_9_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_mkInitialTree_spec__1___redArg(lean_object* v___y_10_){
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
v___x_32_ = lean_obj_once(&lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_mkInitialTree_spec__1___redArg___closed__1, &lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_mkInitialTree_spec__1___redArg___closed__1_once, _init_lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_mkInitialTree_spec__1___redArg___closed__1);
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
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_mkInitialTree_spec__1___redArg___boxed(lean_object* v___y_44_, lean_object* v___y_45_){
_start:
{
lean_object* v_res_46_; 
v_res_46_ = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_mkInitialTree_spec__1___redArg(v___y_44_);
lean_dec(v___y_44_);
return v_res_46_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_mkInitialTree_spec__1(lean_object* v___y_47_, lean_object* v___y_48_, lean_object* v___y_49_, lean_object* v___y_50_, lean_object* v___y_51_){
_start:
{
lean_object* v___x_53_; 
v___x_53_ = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_mkInitialTree_spec__1___redArg(v___y_51_);
return v___x_53_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_mkInitialTree_spec__1___boxed(lean_object* v___y_54_, lean_object* v___y_55_, lean_object* v___y_56_, lean_object* v___y_57_, lean_object* v___y_58_, lean_object* v___y_59_){
_start:
{
lean_object* v_res_60_; 
v_res_60_ = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_mkInitialTree_spec__1(v___y_54_, v___y_55_, v___y_56_, v___y_57_, v___y_58_);
lean_dec(v___y_58_);
lean_dec_ref(v___y_57_);
lean_dec(v___y_56_);
lean_dec_ref(v___y_55_);
lean_dec(v___y_54_);
return v_res_60_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_Option_get___at___00Aesop_mkInitialTree_spec__2(lean_object* v_opts_61_, lean_object* v_opt_62_){
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
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00Aesop_mkInitialTree_spec__2___boxed(lean_object* v_opts_71_, lean_object* v_opt_72_){
_start:
{
uint8_t v_res_73_; lean_object* v_r_74_; 
v_res_73_ = lp_aesop_Lean_Option_get___at___00Aesop_mkInitialTree_spec__2(v_opts_71_, v_opt_72_);
lean_dec_ref(v_opt_72_);
lean_dec_ref(v_opts_71_);
v_r_74_ = lean_box(v_res_73_);
return v_r_74_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_mkInitialTree___lam__0(lean_object* v___x_75_, lean_object* v_x_76_, lean_object* v___y_77_, lean_object* v___y_78_, lean_object* v___y_79_, lean_object* v___y_80_, lean_object* v___y_81_){
_start:
{
lean_object* v___x_83_; 
v___x_83_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_83_, 0, v___x_75_);
return v___x_83_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_mkInitialTree___lam__0___boxed(lean_object* v___x_84_, lean_object* v_x_85_, lean_object* v___y_86_, lean_object* v___y_87_, lean_object* v___y_88_, lean_object* v___y_89_, lean_object* v___y_90_, lean_object* v___y_91_){
_start:
{
lean_object* v_res_92_; 
v_res_92_ = lp_aesop_Aesop_mkInitialTree___lam__0(v___x_84_, v_x_85_, v___y_86_, v___y_87_, v___y_88_, v___y_89_, v___y_90_);
lean_dec(v___y_90_);
lean_dec_ref(v___y_89_);
lean_dec(v___y_88_);
lean_dec_ref(v___y_87_);
lean_dec(v___y_86_);
lean_dec_ref(v_x_85_);
return v_res_92_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00Aesop_UnorderedArraySet_ofHashSet___at___00Aesop_mkInitialTree_spec__0_spec__0(lean_object* v_x_93_, lean_object* v_x_94_){
_start:
{
if (lean_obj_tag(v_x_94_) == 0)
{
return v_x_93_;
}
else
{
lean_object* v_key_95_; lean_object* v_tail_96_; lean_object* v___x_97_; 
v_key_95_ = lean_ctor_get(v_x_94_, 0);
lean_inc(v_key_95_);
v_tail_96_ = lean_ctor_get(v_x_94_, 2);
lean_inc(v_tail_96_);
lean_dec_ref_known(v_x_94_, 3);
v___x_97_ = lean_array_push(v_x_93_, v_key_95_);
v_x_93_ = v___x_97_;
v_x_94_ = v_tail_96_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_UnorderedArraySet_ofHashSet___at___00Aesop_mkInitialTree_spec__0_spec__1(lean_object* v_as_99_, size_t v_i_100_, size_t v_stop_101_, lean_object* v_b_102_){
_start:
{
uint8_t v___x_103_; 
v___x_103_ = lean_usize_dec_eq(v_i_100_, v_stop_101_);
if (v___x_103_ == 0)
{
lean_object* v___x_104_; lean_object* v___x_105_; size_t v___x_106_; size_t v___x_107_; 
v___x_104_ = lean_array_uget_borrowed(v_as_99_, v_i_100_);
lean_inc(v___x_104_);
v___x_105_ = lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00Aesop_UnorderedArraySet_ofHashSet___at___00Aesop_mkInitialTree_spec__0_spec__0(v_b_102_, v___x_104_);
v___x_106_ = ((size_t)1ULL);
v___x_107_ = lean_usize_add(v_i_100_, v___x_106_);
v_i_100_ = v___x_107_;
v_b_102_ = v___x_105_;
goto _start;
}
else
{
return v_b_102_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_UnorderedArraySet_ofHashSet___at___00Aesop_mkInitialTree_spec__0_spec__1___boxed(lean_object* v_as_109_, lean_object* v_i_110_, lean_object* v_stop_111_, lean_object* v_b_112_){
_start:
{
size_t v_i_boxed_113_; size_t v_stop_boxed_114_; lean_object* v_res_115_; 
v_i_boxed_113_ = lean_unbox_usize(v_i_110_);
lean_dec(v_i_110_);
v_stop_boxed_114_ = lean_unbox_usize(v_stop_111_);
lean_dec(v_stop_111_);
v_res_115_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_UnorderedArraySet_ofHashSet___at___00Aesop_mkInitialTree_spec__0_spec__1(v_as_109_, v_i_boxed_113_, v_stop_boxed_114_, v_b_112_);
lean_dec_ref(v_as_109_);
return v_res_115_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_ofHashSet___at___00Aesop_mkInitialTree_spec__0(lean_object* v_xs_116_){
_start:
{
lean_object* v_size_117_; lean_object* v_buckets_118_; lean_object* v___x_119_; lean_object* v___x_120_; lean_object* v___x_121_; uint8_t v___x_122_; 
v_size_117_ = lean_ctor_get(v_xs_116_, 0);
v_buckets_118_ = lean_ctor_get(v_xs_116_, 1);
v___x_119_ = lean_mk_empty_array_with_capacity(v_size_117_);
v___x_120_ = lean_unsigned_to_nat(0u);
v___x_121_ = lean_array_get_size(v_buckets_118_);
v___x_122_ = lean_nat_dec_lt(v___x_120_, v___x_121_);
if (v___x_122_ == 0)
{
return v___x_119_;
}
else
{
uint8_t v___x_123_; 
v___x_123_ = lean_nat_dec_le(v___x_121_, v___x_121_);
if (v___x_123_ == 0)
{
if (v___x_122_ == 0)
{
return v___x_119_;
}
else
{
size_t v___x_124_; size_t v___x_125_; lean_object* v___x_126_; 
v___x_124_ = ((size_t)0ULL);
v___x_125_ = lean_usize_of_nat(v___x_121_);
v___x_126_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_UnorderedArraySet_ofHashSet___at___00Aesop_mkInitialTree_spec__0_spec__1(v_buckets_118_, v___x_124_, v___x_125_, v___x_119_);
return v___x_126_;
}
}
else
{
size_t v___x_127_; size_t v___x_128_; lean_object* v___x_129_; 
v___x_127_ = ((size_t)0ULL);
v___x_128_ = lean_usize_of_nat(v___x_121_);
v___x_129_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_UnorderedArraySet_ofHashSet___at___00Aesop_mkInitialTree_spec__0_spec__1(v_buckets_118_, v___x_127_, v___x_128_, v___x_119_);
return v___x_129_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_ofHashSet___at___00Aesop_mkInitialTree_spec__0___boxed(lean_object* v_xs_130_){
_start:
{
lean_object* v_res_131_; 
v_res_131_ = lp_aesop_Aesop_UnorderedArraySet_ofHashSet___at___00Aesop_mkInitialTree_spec__0(v_xs_130_);
lean_dec_ref(v_xs_130_);
return v_res_131_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_mkInitialTree_spec__3_spec__5_spec__6(size_t v_sz_132_, size_t v_i_133_, lean_object* v_bs_134_){
_start:
{
uint8_t v___x_135_; 
v___x_135_ = lean_usize_dec_lt(v_i_133_, v_sz_132_);
if (v___x_135_ == 0)
{
return v_bs_134_;
}
else
{
lean_object* v_v_136_; lean_object* v_msg_137_; lean_object* v___x_138_; lean_object* v_bs_x27_139_; size_t v___x_140_; size_t v___x_141_; lean_object* v___x_142_; 
v_v_136_ = lean_array_uget_borrowed(v_bs_134_, v_i_133_);
v_msg_137_ = lean_ctor_get(v_v_136_, 1);
lean_inc_ref(v_msg_137_);
v___x_138_ = lean_unsigned_to_nat(0u);
v_bs_x27_139_ = lean_array_uset(v_bs_134_, v_i_133_, v___x_138_);
v___x_140_ = ((size_t)1ULL);
v___x_141_ = lean_usize_add(v_i_133_, v___x_140_);
v___x_142_ = lean_array_uset(v_bs_x27_139_, v_i_133_, v_msg_137_);
v_i_133_ = v___x_141_;
v_bs_134_ = v___x_142_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_mkInitialTree_spec__3_spec__5_spec__6___boxed(lean_object* v_sz_144_, lean_object* v_i_145_, lean_object* v_bs_146_){
_start:
{
size_t v_sz_boxed_147_; size_t v_i_boxed_148_; lean_object* v_res_149_; 
v_sz_boxed_147_ = lean_unbox_usize(v_sz_144_);
lean_dec(v_sz_144_);
v_i_boxed_148_ = lean_unbox_usize(v_i_145_);
lean_dec(v_i_145_);
v_res_149_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_mkInitialTree_spec__3_spec__5_spec__6(v_sz_boxed_147_, v_i_boxed_148_, v_bs_146_);
return v_res_149_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_mkInitialTree_spec__3_spec__5_spec__7(lean_object* v_msgData_150_, lean_object* v___y_151_, lean_object* v___y_152_, lean_object* v___y_153_, lean_object* v___y_154_){
_start:
{
lean_object* v___x_156_; lean_object* v_env_157_; lean_object* v___x_158_; lean_object* v_mctx_159_; lean_object* v_lctx_160_; lean_object* v_options_161_; lean_object* v___x_162_; lean_object* v___x_163_; lean_object* v___x_164_; 
v___x_156_ = lean_st_ref_get(v___y_154_);
v_env_157_ = lean_ctor_get(v___x_156_, 0);
lean_inc_ref(v_env_157_);
lean_dec(v___x_156_);
v___x_158_ = lean_st_ref_get(v___y_152_);
v_mctx_159_ = lean_ctor_get(v___x_158_, 0);
lean_inc_ref(v_mctx_159_);
lean_dec(v___x_158_);
v_lctx_160_ = lean_ctor_get(v___y_151_, 2);
v_options_161_ = lean_ctor_get(v___y_153_, 2);
lean_inc_ref(v_options_161_);
lean_inc_ref(v_lctx_160_);
v___x_162_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_162_, 0, v_env_157_);
lean_ctor_set(v___x_162_, 1, v_mctx_159_);
lean_ctor_set(v___x_162_, 2, v_lctx_160_);
lean_ctor_set(v___x_162_, 3, v_options_161_);
v___x_163_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_163_, 0, v___x_162_);
lean_ctor_set(v___x_163_, 1, v_msgData_150_);
v___x_164_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_164_, 0, v___x_163_);
return v___x_164_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_mkInitialTree_spec__3_spec__5_spec__7___boxed(lean_object* v_msgData_165_, lean_object* v___y_166_, lean_object* v___y_167_, lean_object* v___y_168_, lean_object* v___y_169_, lean_object* v___y_170_){
_start:
{
lean_object* v_res_171_; 
v_res_171_ = lp_aesop_Lean_addMessageContextFull___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_mkInitialTree_spec__3_spec__5_spec__7(v_msgData_165_, v___y_166_, v___y_167_, v___y_168_, v___y_169_);
lean_dec(v___y_169_);
lean_dec_ref(v___y_168_);
lean_dec(v___y_167_);
lean_dec_ref(v___y_166_);
return v_res_171_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_mkInitialTree_spec__3_spec__5___redArg(lean_object* v_oldTraces_172_, lean_object* v_data_173_, lean_object* v_ref_174_, lean_object* v_msg_175_, lean_object* v___y_176_, lean_object* v___y_177_, lean_object* v___y_178_, lean_object* v___y_179_){
_start:
{
lean_object* v_fileName_181_; lean_object* v_fileMap_182_; lean_object* v_options_183_; lean_object* v_currRecDepth_184_; lean_object* v_maxRecDepth_185_; lean_object* v_ref_186_; lean_object* v_currNamespace_187_; lean_object* v_openDecls_188_; lean_object* v_initHeartbeats_189_; lean_object* v_maxHeartbeats_190_; lean_object* v_quotContext_191_; lean_object* v_currMacroScope_192_; uint8_t v_diag_193_; lean_object* v_cancelTk_x3f_194_; uint8_t v_suppressElabErrors_195_; lean_object* v_inheritedTraceOptions_196_; lean_object* v___x_197_; lean_object* v_traceState_198_; lean_object* v_traces_199_; lean_object* v_ref_200_; lean_object* v___x_201_; lean_object* v___x_202_; size_t v_sz_203_; size_t v___x_204_; lean_object* v___x_205_; lean_object* v_msg_206_; lean_object* v___x_207_; lean_object* v_a_208_; lean_object* v___x_210_; uint8_t v_isShared_211_; uint8_t v_isSharedCheck_245_; 
v_fileName_181_ = lean_ctor_get(v___y_178_, 0);
v_fileMap_182_ = lean_ctor_get(v___y_178_, 1);
v_options_183_ = lean_ctor_get(v___y_178_, 2);
v_currRecDepth_184_ = lean_ctor_get(v___y_178_, 3);
v_maxRecDepth_185_ = lean_ctor_get(v___y_178_, 4);
v_ref_186_ = lean_ctor_get(v___y_178_, 5);
v_currNamespace_187_ = lean_ctor_get(v___y_178_, 6);
v_openDecls_188_ = lean_ctor_get(v___y_178_, 7);
v_initHeartbeats_189_ = lean_ctor_get(v___y_178_, 8);
v_maxHeartbeats_190_ = lean_ctor_get(v___y_178_, 9);
v_quotContext_191_ = lean_ctor_get(v___y_178_, 10);
v_currMacroScope_192_ = lean_ctor_get(v___y_178_, 11);
v_diag_193_ = lean_ctor_get_uint8(v___y_178_, sizeof(void*)*14);
v_cancelTk_x3f_194_ = lean_ctor_get(v___y_178_, 12);
v_suppressElabErrors_195_ = lean_ctor_get_uint8(v___y_178_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_196_ = lean_ctor_get(v___y_178_, 13);
v___x_197_ = lean_st_ref_get(v___y_179_);
v_traceState_198_ = lean_ctor_get(v___x_197_, 4);
lean_inc_ref(v_traceState_198_);
lean_dec(v___x_197_);
v_traces_199_ = lean_ctor_get(v_traceState_198_, 0);
lean_inc_ref(v_traces_199_);
lean_dec_ref(v_traceState_198_);
v_ref_200_ = l_Lean_replaceRef(v_ref_174_, v_ref_186_);
lean_inc_ref(v_inheritedTraceOptions_196_);
lean_inc(v_cancelTk_x3f_194_);
lean_inc(v_currMacroScope_192_);
lean_inc(v_quotContext_191_);
lean_inc(v_maxHeartbeats_190_);
lean_inc(v_initHeartbeats_189_);
lean_inc(v_openDecls_188_);
lean_inc(v_currNamespace_187_);
lean_inc(v_maxRecDepth_185_);
lean_inc(v_currRecDepth_184_);
lean_inc_ref(v_options_183_);
lean_inc_ref(v_fileMap_182_);
lean_inc_ref(v_fileName_181_);
v___x_201_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_201_, 0, v_fileName_181_);
lean_ctor_set(v___x_201_, 1, v_fileMap_182_);
lean_ctor_set(v___x_201_, 2, v_options_183_);
lean_ctor_set(v___x_201_, 3, v_currRecDepth_184_);
lean_ctor_set(v___x_201_, 4, v_maxRecDepth_185_);
lean_ctor_set(v___x_201_, 5, v_ref_200_);
lean_ctor_set(v___x_201_, 6, v_currNamespace_187_);
lean_ctor_set(v___x_201_, 7, v_openDecls_188_);
lean_ctor_set(v___x_201_, 8, v_initHeartbeats_189_);
lean_ctor_set(v___x_201_, 9, v_maxHeartbeats_190_);
lean_ctor_set(v___x_201_, 10, v_quotContext_191_);
lean_ctor_set(v___x_201_, 11, v_currMacroScope_192_);
lean_ctor_set(v___x_201_, 12, v_cancelTk_x3f_194_);
lean_ctor_set(v___x_201_, 13, v_inheritedTraceOptions_196_);
lean_ctor_set_uint8(v___x_201_, sizeof(void*)*14, v_diag_193_);
lean_ctor_set_uint8(v___x_201_, sizeof(void*)*14 + 1, v_suppressElabErrors_195_);
v___x_202_ = l_Lean_PersistentArray_toArray___redArg(v_traces_199_);
lean_dec_ref(v_traces_199_);
v_sz_203_ = lean_array_size(v___x_202_);
v___x_204_ = ((size_t)0ULL);
v___x_205_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_mkInitialTree_spec__3_spec__5_spec__6(v_sz_203_, v___x_204_, v___x_202_);
v_msg_206_ = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(v_msg_206_, 0, v_data_173_);
lean_ctor_set(v_msg_206_, 1, v_msg_175_);
lean_ctor_set(v_msg_206_, 2, v___x_205_);
v___x_207_ = lp_aesop_Lean_addMessageContextFull___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_mkInitialTree_spec__3_spec__5_spec__7(v_msg_206_, v___y_176_, v___y_177_, v___x_201_, v___y_179_);
lean_dec_ref_known(v___x_201_, 14);
v_a_208_ = lean_ctor_get(v___x_207_, 0);
v_isSharedCheck_245_ = !lean_is_exclusive(v___x_207_);
if (v_isSharedCheck_245_ == 0)
{
v___x_210_ = v___x_207_;
v_isShared_211_ = v_isSharedCheck_245_;
goto v_resetjp_209_;
}
else
{
lean_inc(v_a_208_);
lean_dec(v___x_207_);
v___x_210_ = lean_box(0);
v_isShared_211_ = v_isSharedCheck_245_;
goto v_resetjp_209_;
}
v_resetjp_209_:
{
lean_object* v___x_212_; lean_object* v_traceState_213_; lean_object* v_env_214_; lean_object* v_nextMacroScope_215_; lean_object* v_ngen_216_; lean_object* v_auxDeclNGen_217_; lean_object* v_cache_218_; lean_object* v_messages_219_; lean_object* v_infoState_220_; lean_object* v_snapshotTasks_221_; lean_object* v___x_223_; uint8_t v_isShared_224_; uint8_t v_isSharedCheck_244_; 
v___x_212_ = lean_st_ref_take(v___y_179_);
v_traceState_213_ = lean_ctor_get(v___x_212_, 4);
v_env_214_ = lean_ctor_get(v___x_212_, 0);
v_nextMacroScope_215_ = lean_ctor_get(v___x_212_, 1);
v_ngen_216_ = lean_ctor_get(v___x_212_, 2);
v_auxDeclNGen_217_ = lean_ctor_get(v___x_212_, 3);
v_cache_218_ = lean_ctor_get(v___x_212_, 5);
v_messages_219_ = lean_ctor_get(v___x_212_, 6);
v_infoState_220_ = lean_ctor_get(v___x_212_, 7);
v_snapshotTasks_221_ = lean_ctor_get(v___x_212_, 8);
v_isSharedCheck_244_ = !lean_is_exclusive(v___x_212_);
if (v_isSharedCheck_244_ == 0)
{
v___x_223_ = v___x_212_;
v_isShared_224_ = v_isSharedCheck_244_;
goto v_resetjp_222_;
}
else
{
lean_inc(v_snapshotTasks_221_);
lean_inc(v_infoState_220_);
lean_inc(v_messages_219_);
lean_inc(v_cache_218_);
lean_inc(v_traceState_213_);
lean_inc(v_auxDeclNGen_217_);
lean_inc(v_ngen_216_);
lean_inc(v_nextMacroScope_215_);
lean_inc(v_env_214_);
lean_dec(v___x_212_);
v___x_223_ = lean_box(0);
v_isShared_224_ = v_isSharedCheck_244_;
goto v_resetjp_222_;
}
v_resetjp_222_:
{
uint64_t v_tid_225_; lean_object* v___x_227_; uint8_t v_isShared_228_; uint8_t v_isSharedCheck_242_; 
v_tid_225_ = lean_ctor_get_uint64(v_traceState_213_, sizeof(void*)*1);
v_isSharedCheck_242_ = !lean_is_exclusive(v_traceState_213_);
if (v_isSharedCheck_242_ == 0)
{
lean_object* v_unused_243_; 
v_unused_243_ = lean_ctor_get(v_traceState_213_, 0);
lean_dec(v_unused_243_);
v___x_227_ = v_traceState_213_;
v_isShared_228_ = v_isSharedCheck_242_;
goto v_resetjp_226_;
}
else
{
lean_dec(v_traceState_213_);
v___x_227_ = lean_box(0);
v_isShared_228_ = v_isSharedCheck_242_;
goto v_resetjp_226_;
}
v_resetjp_226_:
{
lean_object* v___x_229_; lean_object* v___x_230_; lean_object* v___x_232_; 
v___x_229_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_229_, 0, v_ref_174_);
lean_ctor_set(v___x_229_, 1, v_a_208_);
v___x_230_ = l_Lean_PersistentArray_push___redArg(v_oldTraces_172_, v___x_229_);
if (v_isShared_228_ == 0)
{
lean_ctor_set(v___x_227_, 0, v___x_230_);
v___x_232_ = v___x_227_;
goto v_reusejp_231_;
}
else
{
lean_object* v_reuseFailAlloc_241_; 
v_reuseFailAlloc_241_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_241_, 0, v___x_230_);
lean_ctor_set_uint64(v_reuseFailAlloc_241_, sizeof(void*)*1, v_tid_225_);
v___x_232_ = v_reuseFailAlloc_241_;
goto v_reusejp_231_;
}
v_reusejp_231_:
{
lean_object* v___x_234_; 
if (v_isShared_224_ == 0)
{
lean_ctor_set(v___x_223_, 4, v___x_232_);
v___x_234_ = v___x_223_;
goto v_reusejp_233_;
}
else
{
lean_object* v_reuseFailAlloc_240_; 
v_reuseFailAlloc_240_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_240_, 0, v_env_214_);
lean_ctor_set(v_reuseFailAlloc_240_, 1, v_nextMacroScope_215_);
lean_ctor_set(v_reuseFailAlloc_240_, 2, v_ngen_216_);
lean_ctor_set(v_reuseFailAlloc_240_, 3, v_auxDeclNGen_217_);
lean_ctor_set(v_reuseFailAlloc_240_, 4, v___x_232_);
lean_ctor_set(v_reuseFailAlloc_240_, 5, v_cache_218_);
lean_ctor_set(v_reuseFailAlloc_240_, 6, v_messages_219_);
lean_ctor_set(v_reuseFailAlloc_240_, 7, v_infoState_220_);
lean_ctor_set(v_reuseFailAlloc_240_, 8, v_snapshotTasks_221_);
v___x_234_ = v_reuseFailAlloc_240_;
goto v_reusejp_233_;
}
v_reusejp_233_:
{
lean_object* v___x_235_; lean_object* v___x_236_; lean_object* v___x_238_; 
v___x_235_ = lean_st_ref_set(v___y_179_, v___x_234_);
v___x_236_ = lean_box(0);
if (v_isShared_211_ == 0)
{
lean_ctor_set(v___x_210_, 0, v___x_236_);
v___x_238_ = v___x_210_;
goto v_reusejp_237_;
}
else
{
lean_object* v_reuseFailAlloc_239_; 
v_reuseFailAlloc_239_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_239_, 0, v___x_236_);
v___x_238_ = v_reuseFailAlloc_239_;
goto v_reusejp_237_;
}
v_reusejp_237_:
{
return v___x_238_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_mkInitialTree_spec__3_spec__5___redArg___boxed(lean_object* v_oldTraces_246_, lean_object* v_data_247_, lean_object* v_ref_248_, lean_object* v_msg_249_, lean_object* v___y_250_, lean_object* v___y_251_, lean_object* v___y_252_, lean_object* v___y_253_, lean_object* v___y_254_){
_start:
{
lean_object* v_res_255_; 
v_res_255_ = lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_mkInitialTree_spec__3_spec__5___redArg(v_oldTraces_246_, v_data_247_, v_ref_248_, v_msg_249_, v___y_250_, v___y_251_, v___y_252_, v___y_253_);
lean_dec(v___y_253_);
lean_dec_ref(v___y_252_);
lean_dec(v___y_251_);
lean_dec_ref(v___y_250_);
return v_res_255_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_mkInitialTree_spec__3_spec__7(lean_object* v_e_256_){
_start:
{
if (lean_obj_tag(v_e_256_) == 0)
{
uint8_t v___x_257_; 
v___x_257_ = 2;
return v___x_257_;
}
else
{
uint8_t v___x_258_; 
v___x_258_ = 0;
return v___x_258_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_mkInitialTree_spec__3_spec__7___boxed(lean_object* v_e_259_){
_start:
{
uint8_t v_res_260_; lean_object* v_r_261_; 
v_res_260_ = lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_mkInitialTree_spec__3_spec__7(v_e_259_);
lean_dec_ref(v_e_259_);
v_r_261_ = lean_box(v_res_260_);
return v_r_261_;
}
}
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_mkInitialTree_spec__3_spec__6___redArg(lean_object* v_x_262_){
_start:
{
if (lean_obj_tag(v_x_262_) == 0)
{
lean_object* v_a_264_; lean_object* v___x_266_; uint8_t v_isShared_267_; uint8_t v_isSharedCheck_271_; 
v_a_264_ = lean_ctor_get(v_x_262_, 0);
v_isSharedCheck_271_ = !lean_is_exclusive(v_x_262_);
if (v_isSharedCheck_271_ == 0)
{
v___x_266_ = v_x_262_;
v_isShared_267_ = v_isSharedCheck_271_;
goto v_resetjp_265_;
}
else
{
lean_inc(v_a_264_);
lean_dec(v_x_262_);
v___x_266_ = lean_box(0);
v_isShared_267_ = v_isSharedCheck_271_;
goto v_resetjp_265_;
}
v_resetjp_265_:
{
lean_object* v___x_269_; 
if (v_isShared_267_ == 0)
{
lean_ctor_set_tag(v___x_266_, 1);
v___x_269_ = v___x_266_;
goto v_reusejp_268_;
}
else
{
lean_object* v_reuseFailAlloc_270_; 
v_reuseFailAlloc_270_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_270_, 0, v_a_264_);
v___x_269_ = v_reuseFailAlloc_270_;
goto v_reusejp_268_;
}
v_reusejp_268_:
{
return v___x_269_;
}
}
}
else
{
lean_object* v_a_272_; lean_object* v___x_274_; uint8_t v_isShared_275_; uint8_t v_isSharedCheck_279_; 
v_a_272_ = lean_ctor_get(v_x_262_, 0);
v_isSharedCheck_279_ = !lean_is_exclusive(v_x_262_);
if (v_isSharedCheck_279_ == 0)
{
v___x_274_ = v_x_262_;
v_isShared_275_ = v_isSharedCheck_279_;
goto v_resetjp_273_;
}
else
{
lean_inc(v_a_272_);
lean_dec(v_x_262_);
v___x_274_ = lean_box(0);
v_isShared_275_ = v_isSharedCheck_279_;
goto v_resetjp_273_;
}
v_resetjp_273_:
{
lean_object* v___x_277_; 
if (v_isShared_275_ == 0)
{
lean_ctor_set_tag(v___x_274_, 0);
v___x_277_ = v___x_274_;
goto v_reusejp_276_;
}
else
{
lean_object* v_reuseFailAlloc_278_; 
v_reuseFailAlloc_278_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_278_, 0, v_a_272_);
v___x_277_ = v_reuseFailAlloc_278_;
goto v_reusejp_276_;
}
v_reusejp_276_:
{
return v___x_277_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_mkInitialTree_spec__3_spec__6___redArg___boxed(lean_object* v_x_280_, lean_object* v___y_281_){
_start:
{
lean_object* v_res_282_; 
v_res_282_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_mkInitialTree_spec__3_spec__6___redArg(v_x_280_);
return v_res_282_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_mkInitialTree_spec__3_spec__8(lean_object* v_opts_283_, lean_object* v_opt_284_){
_start:
{
lean_object* v_name_285_; lean_object* v_defValue_286_; lean_object* v_map_287_; lean_object* v___x_288_; 
v_name_285_ = lean_ctor_get(v_opt_284_, 0);
v_defValue_286_ = lean_ctor_get(v_opt_284_, 1);
v_map_287_ = lean_ctor_get(v_opts_283_, 0);
v___x_288_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_287_, v_name_285_);
if (lean_obj_tag(v___x_288_) == 0)
{
lean_inc(v_defValue_286_);
return v_defValue_286_;
}
else
{
lean_object* v_val_289_; 
v_val_289_ = lean_ctor_get(v___x_288_, 0);
lean_inc(v_val_289_);
lean_dec_ref_known(v___x_288_, 1);
if (lean_obj_tag(v_val_289_) == 3)
{
lean_object* v_v_290_; 
v_v_290_ = lean_ctor_get(v_val_289_, 0);
lean_inc(v_v_290_);
lean_dec_ref_known(v_val_289_, 1);
return v_v_290_;
}
else
{
lean_dec(v_val_289_);
lean_inc(v_defValue_286_);
return v_defValue_286_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_mkInitialTree_spec__3_spec__8___boxed(lean_object* v_opts_291_, lean_object* v_opt_292_){
_start:
{
lean_object* v_res_293_; 
v_res_293_ = lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_mkInitialTree_spec__3_spec__8(v_opts_291_, v_opt_292_);
lean_dec_ref(v_opt_292_);
lean_dec_ref(v_opts_291_);
return v_res_293_;
}
}
static double _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_mkInitialTree_spec__3___closed__0(void){
_start:
{
lean_object* v___x_294_; double v___x_295_; 
v___x_294_ = lean_unsigned_to_nat(0u);
v___x_295_ = lean_float_of_nat(v___x_294_);
return v___x_295_;
}
}
static lean_object* _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_mkInitialTree_spec__3___closed__2(void){
_start:
{
lean_object* v___x_297_; lean_object* v___x_298_; 
v___x_297_ = ((lean_object*)(lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_mkInitialTree_spec__3___closed__1));
v___x_298_ = l_Lean_stringToMessageData(v___x_297_);
return v___x_298_;
}
}
static double _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_mkInitialTree_spec__3___closed__3(void){
_start:
{
lean_object* v___x_299_; double v___x_300_; 
v___x_299_ = lean_unsigned_to_nat(1000u);
v___x_300_ = lean_float_of_nat(v___x_299_);
return v___x_300_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_mkInitialTree_spec__3(lean_object* v_cls_301_, uint8_t v_collapsed_302_, lean_object* v_tag_303_, lean_object* v_opts_304_, uint8_t v_clsEnabled_305_, lean_object* v_oldTraces_306_, lean_object* v_msg_307_, lean_object* v_resStartStop_308_, lean_object* v___y_309_, lean_object* v___y_310_, lean_object* v___y_311_, lean_object* v___y_312_, lean_object* v___y_313_){
_start:
{
lean_object* v_fst_315_; lean_object* v_snd_316_; lean_object* v___y_318_; lean_object* v___y_319_; lean_object* v_data_320_; lean_object* v_fst_331_; lean_object* v_snd_332_; lean_object* v___x_333_; uint8_t v___x_334_; lean_object* v___y_336_; lean_object* v_a_337_; uint8_t v___y_352_; double v___y_383_; 
v_fst_315_ = lean_ctor_get(v_resStartStop_308_, 0);
lean_inc(v_fst_315_);
v_snd_316_ = lean_ctor_get(v_resStartStop_308_, 1);
lean_inc(v_snd_316_);
lean_dec_ref(v_resStartStop_308_);
v_fst_331_ = lean_ctor_get(v_snd_316_, 0);
lean_inc(v_fst_331_);
v_snd_332_ = lean_ctor_get(v_snd_316_, 1);
lean_inc(v_snd_332_);
lean_dec(v_snd_316_);
v___x_333_ = l_Lean_trace_profiler;
v___x_334_ = lp_aesop_Lean_Option_get___at___00Aesop_mkInitialTree_spec__2(v_opts_304_, v___x_333_);
if (v___x_334_ == 0)
{
v___y_352_ = v___x_334_;
goto v___jp_351_;
}
else
{
lean_object* v___x_388_; uint8_t v___x_389_; 
v___x_388_ = l_Lean_trace_profiler_useHeartbeats;
v___x_389_ = lp_aesop_Lean_Option_get___at___00Aesop_mkInitialTree_spec__2(v_opts_304_, v___x_388_);
if (v___x_389_ == 0)
{
lean_object* v___x_390_; lean_object* v___x_391_; double v___x_392_; double v___x_393_; double v___x_394_; 
v___x_390_ = l_Lean_trace_profiler_threshold;
v___x_391_ = lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_mkInitialTree_spec__3_spec__8(v_opts_304_, v___x_390_);
v___x_392_ = lean_float_of_nat(v___x_391_);
v___x_393_ = lean_float_once(&lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_mkInitialTree_spec__3___closed__3, &lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_mkInitialTree_spec__3___closed__3_once, _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_mkInitialTree_spec__3___closed__3);
v___x_394_ = lean_float_div(v___x_392_, v___x_393_);
v___y_383_ = v___x_394_;
goto v___jp_382_;
}
else
{
lean_object* v___x_395_; lean_object* v___x_396_; double v___x_397_; 
v___x_395_ = l_Lean_trace_profiler_threshold;
v___x_396_ = lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_mkInitialTree_spec__3_spec__8(v_opts_304_, v___x_395_);
v___x_397_ = lean_float_of_nat(v___x_396_);
v___y_383_ = v___x_397_;
goto v___jp_382_;
}
}
v___jp_317_:
{
lean_object* v___x_321_; 
lean_inc(v___y_318_);
v___x_321_ = lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_mkInitialTree_spec__3_spec__5___redArg(v_oldTraces_306_, v_data_320_, v___y_318_, v___y_319_, v___y_310_, v___y_311_, v___y_312_, v___y_313_);
if (lean_obj_tag(v___x_321_) == 0)
{
lean_object* v___x_322_; 
lean_dec_ref_known(v___x_321_, 1);
v___x_322_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_mkInitialTree_spec__3_spec__6___redArg(v_fst_315_);
return v___x_322_;
}
else
{
lean_object* v_a_323_; lean_object* v___x_325_; uint8_t v_isShared_326_; uint8_t v_isSharedCheck_330_; 
lean_dec(v_fst_315_);
v_a_323_ = lean_ctor_get(v___x_321_, 0);
v_isSharedCheck_330_ = !lean_is_exclusive(v___x_321_);
if (v_isSharedCheck_330_ == 0)
{
v___x_325_ = v___x_321_;
v_isShared_326_ = v_isSharedCheck_330_;
goto v_resetjp_324_;
}
else
{
lean_inc(v_a_323_);
lean_dec(v___x_321_);
v___x_325_ = lean_box(0);
v_isShared_326_ = v_isSharedCheck_330_;
goto v_resetjp_324_;
}
v_resetjp_324_:
{
lean_object* v___x_328_; 
if (v_isShared_326_ == 0)
{
v___x_328_ = v___x_325_;
goto v_reusejp_327_;
}
else
{
lean_object* v_reuseFailAlloc_329_; 
v_reuseFailAlloc_329_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_329_, 0, v_a_323_);
v___x_328_ = v_reuseFailAlloc_329_;
goto v_reusejp_327_;
}
v_reusejp_327_:
{
return v___x_328_;
}
}
}
}
v___jp_335_:
{
uint8_t v_result_338_; lean_object* v___x_339_; lean_object* v___x_340_; double v___x_341_; lean_object* v_data_342_; 
v_result_338_ = lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_mkInitialTree_spec__3_spec__7(v_fst_315_);
v___x_339_ = lean_box(v_result_338_);
v___x_340_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_340_, 0, v___x_339_);
v___x_341_ = lean_float_once(&lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_mkInitialTree_spec__3___closed__0, &lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_mkInitialTree_spec__3___closed__0_once, _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_mkInitialTree_spec__3___closed__0);
lean_inc_ref(v_tag_303_);
lean_inc_ref(v___x_340_);
lean_inc(v_cls_301_);
v_data_342_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v_data_342_, 0, v_cls_301_);
lean_ctor_set(v_data_342_, 1, v___x_340_);
lean_ctor_set(v_data_342_, 2, v_tag_303_);
lean_ctor_set_float(v_data_342_, sizeof(void*)*3, v___x_341_);
lean_ctor_set_float(v_data_342_, sizeof(void*)*3 + 8, v___x_341_);
lean_ctor_set_uint8(v_data_342_, sizeof(void*)*3 + 16, v_collapsed_302_);
if (v___x_334_ == 0)
{
lean_dec_ref_known(v___x_340_, 1);
lean_dec(v_snd_332_);
lean_dec(v_fst_331_);
lean_dec_ref(v_tag_303_);
lean_dec(v_cls_301_);
v___y_318_ = v___y_336_;
v___y_319_ = v_a_337_;
v_data_320_ = v_data_342_;
goto v___jp_317_;
}
else
{
lean_object* v_data_343_; double v___x_344_; double v___x_345_; 
lean_dec_ref_known(v_data_342_, 3);
v_data_343_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v_data_343_, 0, v_cls_301_);
lean_ctor_set(v_data_343_, 1, v___x_340_);
lean_ctor_set(v_data_343_, 2, v_tag_303_);
v___x_344_ = lean_unbox_float(v_fst_331_);
lean_dec(v_fst_331_);
lean_ctor_set_float(v_data_343_, sizeof(void*)*3, v___x_344_);
v___x_345_ = lean_unbox_float(v_snd_332_);
lean_dec(v_snd_332_);
lean_ctor_set_float(v_data_343_, sizeof(void*)*3 + 8, v___x_345_);
lean_ctor_set_uint8(v_data_343_, sizeof(void*)*3 + 16, v_collapsed_302_);
v___y_318_ = v___y_336_;
v___y_319_ = v_a_337_;
v_data_320_ = v_data_343_;
goto v___jp_317_;
}
}
v___jp_346_:
{
lean_object* v_ref_347_; lean_object* v___x_348_; 
v_ref_347_ = lean_ctor_get(v___y_312_, 5);
lean_inc(v___y_313_);
lean_inc_ref(v___y_312_);
lean_inc(v___y_311_);
lean_inc_ref(v___y_310_);
lean_inc(v___y_309_);
lean_inc(v_fst_315_);
v___x_348_ = lean_apply_7(v_msg_307_, v_fst_315_, v___y_309_, v___y_310_, v___y_311_, v___y_312_, v___y_313_, lean_box(0));
if (lean_obj_tag(v___x_348_) == 0)
{
lean_object* v_a_349_; 
v_a_349_ = lean_ctor_get(v___x_348_, 0);
lean_inc(v_a_349_);
lean_dec_ref_known(v___x_348_, 1);
v___y_336_ = v_ref_347_;
v_a_337_ = v_a_349_;
goto v___jp_335_;
}
else
{
lean_object* v___x_350_; 
lean_dec_ref_known(v___x_348_, 1);
v___x_350_ = lean_obj_once(&lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_mkInitialTree_spec__3___closed__2, &lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_mkInitialTree_spec__3___closed__2_once, _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_mkInitialTree_spec__3___closed__2);
v___y_336_ = v_ref_347_;
v_a_337_ = v___x_350_;
goto v___jp_335_;
}
}
v___jp_351_:
{
if (v_clsEnabled_305_ == 0)
{
if (v___y_352_ == 0)
{
lean_object* v___x_353_; lean_object* v_traceState_354_; lean_object* v_env_355_; lean_object* v_nextMacroScope_356_; lean_object* v_ngen_357_; lean_object* v_auxDeclNGen_358_; lean_object* v_cache_359_; lean_object* v_messages_360_; lean_object* v_infoState_361_; lean_object* v_snapshotTasks_362_; lean_object* v___x_364_; uint8_t v_isShared_365_; uint8_t v_isSharedCheck_381_; 
lean_dec(v_snd_332_);
lean_dec(v_fst_331_);
lean_dec_ref(v_msg_307_);
lean_dec_ref(v_tag_303_);
lean_dec(v_cls_301_);
v___x_353_ = lean_st_ref_take(v___y_313_);
v_traceState_354_ = lean_ctor_get(v___x_353_, 4);
v_env_355_ = lean_ctor_get(v___x_353_, 0);
v_nextMacroScope_356_ = lean_ctor_get(v___x_353_, 1);
v_ngen_357_ = lean_ctor_get(v___x_353_, 2);
v_auxDeclNGen_358_ = lean_ctor_get(v___x_353_, 3);
v_cache_359_ = lean_ctor_get(v___x_353_, 5);
v_messages_360_ = lean_ctor_get(v___x_353_, 6);
v_infoState_361_ = lean_ctor_get(v___x_353_, 7);
v_snapshotTasks_362_ = lean_ctor_get(v___x_353_, 8);
v_isSharedCheck_381_ = !lean_is_exclusive(v___x_353_);
if (v_isSharedCheck_381_ == 0)
{
v___x_364_ = v___x_353_;
v_isShared_365_ = v_isSharedCheck_381_;
goto v_resetjp_363_;
}
else
{
lean_inc(v_snapshotTasks_362_);
lean_inc(v_infoState_361_);
lean_inc(v_messages_360_);
lean_inc(v_cache_359_);
lean_inc(v_traceState_354_);
lean_inc(v_auxDeclNGen_358_);
lean_inc(v_ngen_357_);
lean_inc(v_nextMacroScope_356_);
lean_inc(v_env_355_);
lean_dec(v___x_353_);
v___x_364_ = lean_box(0);
v_isShared_365_ = v_isSharedCheck_381_;
goto v_resetjp_363_;
}
v_resetjp_363_:
{
uint64_t v_tid_366_; lean_object* v_traces_367_; lean_object* v___x_369_; uint8_t v_isShared_370_; uint8_t v_isSharedCheck_380_; 
v_tid_366_ = lean_ctor_get_uint64(v_traceState_354_, sizeof(void*)*1);
v_traces_367_ = lean_ctor_get(v_traceState_354_, 0);
v_isSharedCheck_380_ = !lean_is_exclusive(v_traceState_354_);
if (v_isSharedCheck_380_ == 0)
{
v___x_369_ = v_traceState_354_;
v_isShared_370_ = v_isSharedCheck_380_;
goto v_resetjp_368_;
}
else
{
lean_inc(v_traces_367_);
lean_dec(v_traceState_354_);
v___x_369_ = lean_box(0);
v_isShared_370_ = v_isSharedCheck_380_;
goto v_resetjp_368_;
}
v_resetjp_368_:
{
lean_object* v___x_371_; lean_object* v___x_373_; 
v___x_371_ = l_Lean_PersistentArray_append___redArg(v_oldTraces_306_, v_traces_367_);
lean_dec_ref(v_traces_367_);
if (v_isShared_370_ == 0)
{
lean_ctor_set(v___x_369_, 0, v___x_371_);
v___x_373_ = v___x_369_;
goto v_reusejp_372_;
}
else
{
lean_object* v_reuseFailAlloc_379_; 
v_reuseFailAlloc_379_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_379_, 0, v___x_371_);
lean_ctor_set_uint64(v_reuseFailAlloc_379_, sizeof(void*)*1, v_tid_366_);
v___x_373_ = v_reuseFailAlloc_379_;
goto v_reusejp_372_;
}
v_reusejp_372_:
{
lean_object* v___x_375_; 
if (v_isShared_365_ == 0)
{
lean_ctor_set(v___x_364_, 4, v___x_373_);
v___x_375_ = v___x_364_;
goto v_reusejp_374_;
}
else
{
lean_object* v_reuseFailAlloc_378_; 
v_reuseFailAlloc_378_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_378_, 0, v_env_355_);
lean_ctor_set(v_reuseFailAlloc_378_, 1, v_nextMacroScope_356_);
lean_ctor_set(v_reuseFailAlloc_378_, 2, v_ngen_357_);
lean_ctor_set(v_reuseFailAlloc_378_, 3, v_auxDeclNGen_358_);
lean_ctor_set(v_reuseFailAlloc_378_, 4, v___x_373_);
lean_ctor_set(v_reuseFailAlloc_378_, 5, v_cache_359_);
lean_ctor_set(v_reuseFailAlloc_378_, 6, v_messages_360_);
lean_ctor_set(v_reuseFailAlloc_378_, 7, v_infoState_361_);
lean_ctor_set(v_reuseFailAlloc_378_, 8, v_snapshotTasks_362_);
v___x_375_ = v_reuseFailAlloc_378_;
goto v_reusejp_374_;
}
v_reusejp_374_:
{
lean_object* v___x_376_; lean_object* v___x_377_; 
v___x_376_ = lean_st_ref_set(v___y_313_, v___x_375_);
v___x_377_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_mkInitialTree_spec__3_spec__6___redArg(v_fst_315_);
return v___x_377_;
}
}
}
}
}
else
{
goto v___jp_346_;
}
}
else
{
goto v___jp_346_;
}
}
v___jp_382_:
{
double v___x_384_; double v___x_385_; double v___x_386_; uint8_t v___x_387_; 
v___x_384_ = lean_unbox_float(v_snd_332_);
v___x_385_ = lean_unbox_float(v_fst_331_);
v___x_386_ = lean_float_sub(v___x_384_, v___x_385_);
v___x_387_ = lean_float_decLt(v___y_383_, v___x_386_);
v___y_352_ = v___x_387_;
goto v___jp_351_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_mkInitialTree_spec__3___boxed(lean_object* v_cls_398_, lean_object* v_collapsed_399_, lean_object* v_tag_400_, lean_object* v_opts_401_, lean_object* v_clsEnabled_402_, lean_object* v_oldTraces_403_, lean_object* v_msg_404_, lean_object* v_resStartStop_405_, lean_object* v___y_406_, lean_object* v___y_407_, lean_object* v___y_408_, lean_object* v___y_409_, lean_object* v___y_410_, lean_object* v___y_411_){
_start:
{
uint8_t v_collapsed_boxed_412_; uint8_t v_clsEnabled_boxed_413_; lean_object* v_res_414_; 
v_collapsed_boxed_412_ = lean_unbox(v_collapsed_399_);
v_clsEnabled_boxed_413_ = lean_unbox(v_clsEnabled_402_);
v_res_414_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_mkInitialTree_spec__3(v_cls_398_, v_collapsed_boxed_412_, v_tag_400_, v_opts_401_, v_clsEnabled_boxed_413_, v_oldTraces_403_, v_msg_404_, v_resStartStop_405_, v___y_406_, v___y_407_, v___y_408_, v___y_409_, v___y_410_);
lean_dec(v___y_410_);
lean_dec_ref(v___y_409_);
lean_dec(v___y_408_);
lean_dec_ref(v___y_407_);
lean_dec(v___y_406_);
lean_dec_ref(v_opts_401_);
return v_res_414_;
}
}
static lean_object* _init_lp_aesop_Aesop_mkInitialTree___closed__2(void){
_start:
{
lean_object* v___x_422_; 
v___x_422_ = l_Subarray_empty(lean_box(0));
return v___x_422_;
}
}
static lean_object* _init_lp_aesop_Aesop_mkInitialTree___closed__3(void){
_start:
{
lean_object* v___x_423_; lean_object* v___x_424_; lean_object* v___x_425_; 
v___x_423_ = lean_box(0);
v___x_424_ = lean_unsigned_to_nat(16u);
v___x_425_ = lean_mk_array(v___x_424_, v___x_423_);
return v___x_425_;
}
}
static lean_object* _init_lp_aesop_Aesop_mkInitialTree___closed__4(void){
_start:
{
lean_object* v___x_426_; lean_object* v___x_427_; lean_object* v___x_428_; 
v___x_426_ = lean_obj_once(&lp_aesop_Aesop_mkInitialTree___closed__3, &lp_aesop_Aesop_mkInitialTree___closed__3_once, _init_lp_aesop_Aesop_mkInitialTree___closed__3);
v___x_427_ = lean_unsigned_to_nat(0u);
v___x_428_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_428_, 0, v___x_427_);
lean_ctor_set(v___x_428_, 1, v___x_426_);
return v___x_428_;
}
}
static lean_object* _init_lp_aesop_Aesop_mkInitialTree___closed__6(void){
_start:
{
lean_object* v___x_430_; lean_object* v___x_431_; 
v___x_430_ = ((lean_object*)(lp_aesop_Aesop_mkInitialTree___closed__5));
v___x_431_ = l_Lean_stringToMessageData(v___x_430_);
return v___x_431_;
}
}
static lean_object* _init_lp_aesop_Aesop_mkInitialTree___closed__7(void){
_start:
{
lean_object* v___x_432_; lean_object* v___f_433_; 
v___x_432_ = lean_obj_once(&lp_aesop_Aesop_mkInitialTree___closed__6, &lp_aesop_Aesop_mkInitialTree___closed__6_once, _init_lp_aesop_Aesop_mkInitialTree___closed__6);
v___f_433_ = lean_alloc_closure((void*)(lp_aesop_Aesop_mkInitialTree___lam__0___boxed), 8, 1);
lean_closure_set(v___f_433_, 0, v___x_432_);
return v___f_433_;
}
}
static double _init_lp_aesop_Aesop_mkInitialTree___closed__11(void){
_start:
{
lean_object* v___x_438_; double v___x_439_; 
v___x_438_ = lean_unsigned_to_nat(1000000000u);
v___x_439_ = lean_float_of_nat(v___x_438_);
return v___x_439_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_mkInitialTree(lean_object* v_goal_440_, lean_object* v_rs_441_, lean_object* v_a_442_, lean_object* v_a_443_, lean_object* v_a_444_, lean_object* v_a_445_, lean_object* v_a_446_){
_start:
{
lean_object* v___x_448_; lean_object* v_introGoal_449_; lean_object* v_introMVarCluster_450_; lean_object* v_elimMVarCluster_451_; lean_object* v___x_452_; lean_object* v___x_453_; uint8_t v___x_454_; lean_object* v___x_455_; lean_object* v___x_456_; lean_object* v___x_457_; lean_object* v___y_459_; lean_object* v_options_532_; uint8_t v_hasTrace_533_; 
v___x_448_ = lp_aesop_Aesop_treeImpl;
v_introGoal_449_ = lean_ctor_get(v___x_448_, 0);
v_introMVarCluster_450_ = lean_ctor_get(v___x_448_, 4);
v_elimMVarCluster_451_ = lean_ctor_get(v___x_448_, 5);
v___x_452_ = lean_unsigned_to_nat(0u);
v___x_453_ = ((lean_object*)(lp_aesop_Aesop_mkInitialTree___closed__0));
v___x_454_ = 0;
v___x_455_ = ((lean_object*)(lp_aesop_Aesop_mkInitialTree___closed__1));
lean_inc(v_introMVarCluster_450_);
v___x_456_ = lean_apply_1(v_introMVarCluster_450_, v___x_455_);
v___x_457_ = lean_st_mk_ref(v___x_456_);
v_options_532_ = lean_ctor_get(v_a_445_, 2);
v_hasTrace_533_ = lean_ctor_get_uint8(v_options_532_, sizeof(void*)*1);
if (v_hasTrace_533_ == 0)
{
lean_object* v___x_534_; 
lean_inc(v_goal_440_);
v___x_534_ = lp_aesop_Aesop_LocalRuleSet_mkInitialForwardState(v_goal_440_, v_rs_441_, v_a_442_, v_a_443_, v_a_444_, v_a_445_, v_a_446_);
v___y_459_ = v___x_534_;
goto v___jp_458_;
}
else
{
lean_object* v_inheritedTraceOptions_535_; lean_object* v___x_536_; lean_object* v_traceClass_537_; lean_object* v___f_538_; lean_object* v___x_539_; lean_object* v___x_540_; lean_object* v___x_541_; uint8_t v___x_542_; lean_object* v___y_544_; lean_object* v___y_545_; lean_object* v_a_546_; lean_object* v___y_559_; lean_object* v___y_560_; lean_object* v_a_561_; 
v_inheritedTraceOptions_535_ = lean_ctor_get(v_a_445_, 13);
v___x_536_ = lp_aesop_Aesop_TraceOption_forward;
v_traceClass_537_ = lean_ctor_get(v___x_536_, 0);
v___f_538_ = lean_obj_once(&lp_aesop_Aesop_mkInitialTree___closed__7, &lp_aesop_Aesop_mkInitialTree___closed__7_once, _init_lp_aesop_Aesop_mkInitialTree___closed__7);
v___x_539_ = ((lean_object*)(lp_aesop_Aesop_mkInitialTree___closed__8));
v___x_540_ = ((lean_object*)(lp_aesop_Aesop_mkInitialTree___closed__10));
lean_inc(v_traceClass_537_);
v___x_541_ = l_Lean_Name_append(v___x_540_, v_traceClass_537_);
v___x_542_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_535_, v_options_532_, v___x_541_);
lean_dec(v___x_541_);
if (v___x_542_ == 0)
{
lean_object* v___x_611_; uint8_t v___x_612_; 
v___x_611_ = l_Lean_trace_profiler;
v___x_612_ = lp_aesop_Lean_Option_get___at___00Aesop_mkInitialTree_spec__2(v_options_532_, v___x_611_);
if (v___x_612_ == 0)
{
lean_object* v___x_613_; 
lean_inc(v_goal_440_);
v___x_613_ = lp_aesop_Aesop_LocalRuleSet_mkInitialForwardState(v_goal_440_, v_rs_441_, v_a_442_, v_a_443_, v_a_444_, v_a_445_, v_a_446_);
v___y_459_ = v___x_613_;
goto v___jp_458_;
}
else
{
goto v___jp_570_;
}
}
else
{
goto v___jp_570_;
}
v___jp_543_:
{
lean_object* v___x_547_; double v___x_548_; double v___x_549_; double v___x_550_; double v___x_551_; double v___x_552_; lean_object* v___x_553_; lean_object* v___x_554_; lean_object* v___x_555_; lean_object* v___x_556_; lean_object* v___x_557_; 
v___x_547_ = lean_io_mono_nanos_now();
v___x_548_ = lean_float_of_nat(v___y_545_);
v___x_549_ = lean_float_once(&lp_aesop_Aesop_mkInitialTree___closed__11, &lp_aesop_Aesop_mkInitialTree___closed__11_once, _init_lp_aesop_Aesop_mkInitialTree___closed__11);
v___x_550_ = lean_float_div(v___x_548_, v___x_549_);
v___x_551_ = lean_float_of_nat(v___x_547_);
v___x_552_ = lean_float_div(v___x_551_, v___x_549_);
v___x_553_ = lean_box_float(v___x_550_);
v___x_554_ = lean_box_float(v___x_552_);
v___x_555_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_555_, 0, v___x_553_);
lean_ctor_set(v___x_555_, 1, v___x_554_);
v___x_556_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_556_, 0, v_a_546_);
lean_ctor_set(v___x_556_, 1, v___x_555_);
lean_inc(v_traceClass_537_);
v___x_557_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_mkInitialTree_spec__3(v_traceClass_537_, v_hasTrace_533_, v___x_539_, v_options_532_, v___x_542_, v___y_544_, v___f_538_, v___x_556_, v_a_442_, v_a_443_, v_a_444_, v_a_445_, v_a_446_);
v___y_459_ = v___x_557_;
goto v___jp_458_;
}
v___jp_558_:
{
lean_object* v___x_562_; double v___x_563_; double v___x_564_; lean_object* v___x_565_; lean_object* v___x_566_; lean_object* v___x_567_; lean_object* v___x_568_; lean_object* v___x_569_; 
v___x_562_ = lean_io_get_num_heartbeats();
v___x_563_ = lean_float_of_nat(v___y_560_);
v___x_564_ = lean_float_of_nat(v___x_562_);
v___x_565_ = lean_box_float(v___x_563_);
v___x_566_ = lean_box_float(v___x_564_);
v___x_567_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_567_, 0, v___x_565_);
lean_ctor_set(v___x_567_, 1, v___x_566_);
v___x_568_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_568_, 0, v_a_561_);
lean_ctor_set(v___x_568_, 1, v___x_567_);
lean_inc(v_traceClass_537_);
v___x_569_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_mkInitialTree_spec__3(v_traceClass_537_, v_hasTrace_533_, v___x_539_, v_options_532_, v___x_542_, v___y_559_, v___f_538_, v___x_568_, v_a_442_, v_a_443_, v_a_444_, v_a_445_, v_a_446_);
v___y_459_ = v___x_569_;
goto v___jp_458_;
}
v___jp_570_:
{
lean_object* v___x_571_; lean_object* v_a_572_; lean_object* v___x_573_; uint8_t v___x_574_; 
v___x_571_ = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_mkInitialTree_spec__1___redArg(v_a_446_);
v_a_572_ = lean_ctor_get(v___x_571_, 0);
lean_inc(v_a_572_);
lean_dec_ref(v___x_571_);
v___x_573_ = l_Lean_trace_profiler_useHeartbeats;
v___x_574_ = lp_aesop_Lean_Option_get___at___00Aesop_mkInitialTree_spec__2(v_options_532_, v___x_573_);
if (v___x_574_ == 0)
{
lean_object* v___x_575_; lean_object* v___x_576_; 
v___x_575_ = lean_io_mono_nanos_now();
lean_inc(v_goal_440_);
v___x_576_ = lp_aesop_Aesop_LocalRuleSet_mkInitialForwardState(v_goal_440_, v_rs_441_, v_a_442_, v_a_443_, v_a_444_, v_a_445_, v_a_446_);
if (lean_obj_tag(v___x_576_) == 0)
{
lean_object* v_a_577_; lean_object* v___x_579_; uint8_t v_isShared_580_; uint8_t v_isSharedCheck_584_; 
v_a_577_ = lean_ctor_get(v___x_576_, 0);
v_isSharedCheck_584_ = !lean_is_exclusive(v___x_576_);
if (v_isSharedCheck_584_ == 0)
{
v___x_579_ = v___x_576_;
v_isShared_580_ = v_isSharedCheck_584_;
goto v_resetjp_578_;
}
else
{
lean_inc(v_a_577_);
lean_dec(v___x_576_);
v___x_579_ = lean_box(0);
v_isShared_580_ = v_isSharedCheck_584_;
goto v_resetjp_578_;
}
v_resetjp_578_:
{
lean_object* v___x_582_; 
if (v_isShared_580_ == 0)
{
lean_ctor_set_tag(v___x_579_, 1);
v___x_582_ = v___x_579_;
goto v_reusejp_581_;
}
else
{
lean_object* v_reuseFailAlloc_583_; 
v_reuseFailAlloc_583_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_583_, 0, v_a_577_);
v___x_582_ = v_reuseFailAlloc_583_;
goto v_reusejp_581_;
}
v_reusejp_581_:
{
v___y_544_ = v_a_572_;
v___y_545_ = v___x_575_;
v_a_546_ = v___x_582_;
goto v___jp_543_;
}
}
}
else
{
lean_object* v_a_585_; lean_object* v___x_587_; uint8_t v_isShared_588_; uint8_t v_isSharedCheck_592_; 
v_a_585_ = lean_ctor_get(v___x_576_, 0);
v_isSharedCheck_592_ = !lean_is_exclusive(v___x_576_);
if (v_isSharedCheck_592_ == 0)
{
v___x_587_ = v___x_576_;
v_isShared_588_ = v_isSharedCheck_592_;
goto v_resetjp_586_;
}
else
{
lean_inc(v_a_585_);
lean_dec(v___x_576_);
v___x_587_ = lean_box(0);
v_isShared_588_ = v_isSharedCheck_592_;
goto v_resetjp_586_;
}
v_resetjp_586_:
{
lean_object* v___x_590_; 
if (v_isShared_588_ == 0)
{
lean_ctor_set_tag(v___x_587_, 0);
v___x_590_ = v___x_587_;
goto v_reusejp_589_;
}
else
{
lean_object* v_reuseFailAlloc_591_; 
v_reuseFailAlloc_591_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_591_, 0, v_a_585_);
v___x_590_ = v_reuseFailAlloc_591_;
goto v_reusejp_589_;
}
v_reusejp_589_:
{
v___y_544_ = v_a_572_;
v___y_545_ = v___x_575_;
v_a_546_ = v___x_590_;
goto v___jp_543_;
}
}
}
}
else
{
lean_object* v___x_593_; lean_object* v___x_594_; 
v___x_593_ = lean_io_get_num_heartbeats();
lean_inc(v_goal_440_);
v___x_594_ = lp_aesop_Aesop_LocalRuleSet_mkInitialForwardState(v_goal_440_, v_rs_441_, v_a_442_, v_a_443_, v_a_444_, v_a_445_, v_a_446_);
if (lean_obj_tag(v___x_594_) == 0)
{
lean_object* v_a_595_; lean_object* v___x_597_; uint8_t v_isShared_598_; uint8_t v_isSharedCheck_602_; 
v_a_595_ = lean_ctor_get(v___x_594_, 0);
v_isSharedCheck_602_ = !lean_is_exclusive(v___x_594_);
if (v_isSharedCheck_602_ == 0)
{
v___x_597_ = v___x_594_;
v_isShared_598_ = v_isSharedCheck_602_;
goto v_resetjp_596_;
}
else
{
lean_inc(v_a_595_);
lean_dec(v___x_594_);
v___x_597_ = lean_box(0);
v_isShared_598_ = v_isSharedCheck_602_;
goto v_resetjp_596_;
}
v_resetjp_596_:
{
lean_object* v___x_600_; 
if (v_isShared_598_ == 0)
{
lean_ctor_set_tag(v___x_597_, 1);
v___x_600_ = v___x_597_;
goto v_reusejp_599_;
}
else
{
lean_object* v_reuseFailAlloc_601_; 
v_reuseFailAlloc_601_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_601_, 0, v_a_595_);
v___x_600_ = v_reuseFailAlloc_601_;
goto v_reusejp_599_;
}
v_reusejp_599_:
{
v___y_559_ = v_a_572_;
v___y_560_ = v___x_593_;
v_a_561_ = v___x_600_;
goto v___jp_558_;
}
}
}
else
{
lean_object* v_a_603_; lean_object* v___x_605_; uint8_t v_isShared_606_; uint8_t v_isSharedCheck_610_; 
v_a_603_ = lean_ctor_get(v___x_594_, 0);
v_isSharedCheck_610_ = !lean_is_exclusive(v___x_594_);
if (v_isSharedCheck_610_ == 0)
{
v___x_605_ = v___x_594_;
v_isShared_606_ = v_isSharedCheck_610_;
goto v_resetjp_604_;
}
else
{
lean_inc(v_a_603_);
lean_dec(v___x_594_);
v___x_605_ = lean_box(0);
v_isShared_606_ = v_isSharedCheck_610_;
goto v_resetjp_604_;
}
v_resetjp_604_:
{
lean_object* v___x_608_; 
if (v_isShared_606_ == 0)
{
lean_ctor_set_tag(v___x_605_, 0);
v___x_608_ = v___x_605_;
goto v_reusejp_607_;
}
else
{
lean_object* v_reuseFailAlloc_609_; 
v_reuseFailAlloc_609_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_609_, 0, v_a_603_);
v___x_608_ = v_reuseFailAlloc_609_;
goto v_reusejp_607_;
}
v_reusejp_607_:
{
v___y_559_ = v_a_572_;
v___y_560_ = v___x_593_;
v_a_561_ = v___x_608_;
goto v___jp_558_;
}
}
}
}
}
}
v___jp_458_:
{
if (lean_obj_tag(v___y_459_) == 0)
{
lean_object* v_a_460_; lean_object* v_fst_461_; lean_object* v_snd_462_; lean_object* v___x_463_; 
v_a_460_ = lean_ctor_get(v___y_459_, 0);
lean_inc(v_a_460_);
lean_dec_ref_known(v___y_459_, 1);
v_fst_461_ = lean_ctor_get(v_a_460_, 0);
lean_inc(v_fst_461_);
v_snd_462_ = lean_ctor_get(v_a_460_, 1);
lean_inc(v_snd_462_);
lean_dec(v_a_460_);
lean_inc(v_goal_440_);
v___x_463_ = l_Lean_MVarId_getMVarDependencies(v_goal_440_, v___x_454_, v_a_443_, v_a_444_, v_a_445_, v_a_446_);
if (lean_obj_tag(v___x_463_) == 0)
{
lean_object* v_a_464_; lean_object* v___x_465_; lean_object* v___x_466_; uint8_t v___x_467_; lean_object* v___x_468_; lean_object* v___x_469_; lean_object* v___x_470_; double v___x_471_; lean_object* v___x_472_; lean_object* v___x_473_; lean_object* v___x_474_; lean_object* v___x_475_; lean_object* v___x_476_; lean_object* v___x_477_; lean_object* v___x_478_; lean_object* v___x_479_; lean_object* v_parent_x3f_480_; uint8_t v_isIrrelevant_481_; uint8_t v_state_482_; lean_object* v___x_484_; uint8_t v_isShared_485_; uint8_t v_isSharedCheck_514_; 
v_a_464_ = lean_ctor_get(v___x_463_, 0);
lean_inc(v_a_464_);
lean_dec_ref_known(v___x_463_, 1);
v___x_465_ = lp_aesop_Aesop_GoalId_zero;
v___x_466_ = lean_box(0);
v___x_467_ = 0;
v___x_468_ = lean_box(0);
v___x_469_ = lp_aesop_Aesop_UnorderedArraySet_ofHashSet___at___00Aesop_mkInitialTree_spec__0(v_a_464_);
lean_dec(v_a_464_);
v___x_470_ = lp_aesop_Aesop_ForwardRuleMatches_ofArray(v_snd_462_);
lean_dec(v_snd_462_);
v___x_471_ = lp_aesop_Aesop_Percent_hundred;
v___x_472_ = lean_unsigned_to_nat(1u);
v___x_473_ = lp_aesop_Aesop_Iteration_none;
v___x_474_ = lean_obj_once(&lp_aesop_Aesop_mkInitialTree___closed__2, &lp_aesop_Aesop_mkInitialTree___closed__2_once, _init_lp_aesop_Aesop_mkInitialTree___closed__2);
lean_inc(v___x_457_);
v___x_475_ = lean_alloc_ctor(0, 14, 12);
lean_ctor_set(v___x_475_, 0, v___x_465_);
lean_ctor_set(v___x_475_, 1, v___x_457_);
lean_ctor_set(v___x_475_, 2, v___x_453_);
lean_ctor_set(v___x_475_, 3, v___x_466_);
lean_ctor_set(v___x_475_, 4, v___x_452_);
lean_ctor_set(v___x_475_, 5, v_goal_440_);
lean_ctor_set(v___x_475_, 6, v___x_468_);
lean_ctor_set(v___x_475_, 7, v___x_469_);
lean_ctor_set(v___x_475_, 8, v_fst_461_);
lean_ctor_set(v___x_475_, 9, v___x_470_);
lean_ctor_set(v___x_475_, 10, v___x_472_);
lean_ctor_set(v___x_475_, 11, v___x_473_);
lean_ctor_set(v___x_475_, 12, v___x_474_);
lean_ctor_set(v___x_475_, 13, v___x_453_);
lean_ctor_set_uint8(v___x_475_, sizeof(void*)*14 + 8, v___x_467_);
lean_ctor_set_uint8(v___x_475_, sizeof(void*)*14 + 9, v___x_454_);
lean_ctor_set_uint8(v___x_475_, sizeof(void*)*14 + 10, v___x_454_);
lean_ctor_set_float(v___x_475_, sizeof(void*)*14, v___x_471_);
lean_ctor_set_uint8(v___x_475_, sizeof(void*)*14 + 11, v___x_454_);
lean_inc(v_introGoal_449_);
v___x_476_ = lean_apply_1(v_introGoal_449_, v___x_475_);
v___x_477_ = lean_st_mk_ref(v___x_476_);
v___x_478_ = lean_st_ref_take(v___x_457_);
lean_inc_ref(v_elimMVarCluster_451_);
v___x_479_ = lean_apply_1(v_elimMVarCluster_451_, v___x_478_);
v_parent_x3f_480_ = lean_ctor_get(v___x_479_, 0);
v_isIrrelevant_481_ = lean_ctor_get_uint8(v___x_479_, sizeof(void*)*2);
v_state_482_ = lean_ctor_get_uint8(v___x_479_, sizeof(void*)*2 + 1);
v_isSharedCheck_514_ = !lean_is_exclusive(v___x_479_);
if (v_isSharedCheck_514_ == 0)
{
lean_object* v_unused_515_; 
v_unused_515_ = lean_ctor_get(v___x_479_, 1);
lean_dec(v_unused_515_);
v___x_484_ = v___x_479_;
v_isShared_485_ = v_isSharedCheck_514_;
goto v_resetjp_483_;
}
else
{
lean_inc(v_parent_x3f_480_);
lean_dec(v___x_479_);
v___x_484_ = lean_box(0);
v_isShared_485_ = v_isSharedCheck_514_;
goto v_resetjp_483_;
}
v_resetjp_483_:
{
lean_object* v___x_486_; lean_object* v___x_487_; lean_object* v___x_489_; 
v___x_486_ = lean_mk_empty_array_with_capacity(v___x_472_);
v___x_487_ = lean_array_push(v___x_486_, v___x_477_);
if (v_isShared_485_ == 0)
{
lean_ctor_set(v___x_484_, 1, v___x_487_);
v___x_489_ = v___x_484_;
goto v_reusejp_488_;
}
else
{
lean_object* v_reuseFailAlloc_513_; 
v_reuseFailAlloc_513_ = lean_alloc_ctor(0, 2, 2);
lean_ctor_set(v_reuseFailAlloc_513_, 0, v_parent_x3f_480_);
lean_ctor_set(v_reuseFailAlloc_513_, 1, v___x_487_);
lean_ctor_set_uint8(v_reuseFailAlloc_513_, sizeof(void*)*2, v_isIrrelevant_481_);
lean_ctor_set_uint8(v_reuseFailAlloc_513_, sizeof(void*)*2 + 1, v_state_482_);
v___x_489_ = v_reuseFailAlloc_513_;
goto v_reusejp_488_;
}
v_reusejp_488_:
{
lean_object* v___x_490_; lean_object* v___x_491_; lean_object* v___x_492_; 
lean_inc(v_introMVarCluster_450_);
v___x_490_ = lean_apply_1(v_introMVarCluster_450_, v___x_489_);
v___x_491_ = lean_st_ref_set(v___x_457_, v___x_490_);
v___x_492_ = l_Lean_Meta_saveState___redArg(v_a_444_, v_a_446_);
if (lean_obj_tag(v___x_492_) == 0)
{
lean_object* v_a_493_; lean_object* v___x_495_; uint8_t v_isShared_496_; uint8_t v_isSharedCheck_504_; 
v_a_493_ = lean_ctor_get(v___x_492_, 0);
v_isSharedCheck_504_ = !lean_is_exclusive(v___x_492_);
if (v_isSharedCheck_504_ == 0)
{
v___x_495_ = v___x_492_;
v_isShared_496_ = v_isSharedCheck_504_;
goto v_resetjp_494_;
}
else
{
lean_inc(v_a_493_);
lean_dec(v___x_492_);
v___x_495_ = lean_box(0);
v_isShared_496_ = v_isSharedCheck_504_;
goto v_resetjp_494_;
}
v_resetjp_494_:
{
lean_object* v___x_497_; lean_object* v___x_498_; lean_object* v___x_499_; lean_object* v___x_500_; lean_object* v___x_502_; 
v___x_497_ = lp_aesop_Aesop_GoalId_one;
v___x_498_ = lp_aesop_Aesop_RappId_zero;
v___x_499_ = lean_obj_once(&lp_aesop_Aesop_mkInitialTree___closed__4, &lp_aesop_Aesop_mkInitialTree___closed__4_once, _init_lp_aesop_Aesop_mkInitialTree___closed__4);
v___x_500_ = lean_alloc_ctor(0, 7, 0);
lean_ctor_set(v___x_500_, 0, v___x_457_);
lean_ctor_set(v___x_500_, 1, v_a_493_);
lean_ctor_set(v___x_500_, 2, v___x_472_);
lean_ctor_set(v___x_500_, 3, v___x_452_);
lean_ctor_set(v___x_500_, 4, v___x_497_);
lean_ctor_set(v___x_500_, 5, v___x_498_);
lean_ctor_set(v___x_500_, 6, v___x_499_);
if (v_isShared_496_ == 0)
{
lean_ctor_set(v___x_495_, 0, v___x_500_);
v___x_502_ = v___x_495_;
goto v_reusejp_501_;
}
else
{
lean_object* v_reuseFailAlloc_503_; 
v_reuseFailAlloc_503_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_503_, 0, v___x_500_);
v___x_502_ = v_reuseFailAlloc_503_;
goto v_reusejp_501_;
}
v_reusejp_501_:
{
return v___x_502_;
}
}
}
else
{
lean_object* v_a_505_; lean_object* v___x_507_; uint8_t v_isShared_508_; uint8_t v_isSharedCheck_512_; 
lean_dec(v___x_457_);
v_a_505_ = lean_ctor_get(v___x_492_, 0);
v_isSharedCheck_512_ = !lean_is_exclusive(v___x_492_);
if (v_isSharedCheck_512_ == 0)
{
v___x_507_ = v___x_492_;
v_isShared_508_ = v_isSharedCheck_512_;
goto v_resetjp_506_;
}
else
{
lean_inc(v_a_505_);
lean_dec(v___x_492_);
v___x_507_ = lean_box(0);
v_isShared_508_ = v_isSharedCheck_512_;
goto v_resetjp_506_;
}
v_resetjp_506_:
{
lean_object* v___x_510_; 
if (v_isShared_508_ == 0)
{
v___x_510_ = v___x_507_;
goto v_reusejp_509_;
}
else
{
lean_object* v_reuseFailAlloc_511_; 
v_reuseFailAlloc_511_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_511_, 0, v_a_505_);
v___x_510_ = v_reuseFailAlloc_511_;
goto v_reusejp_509_;
}
v_reusejp_509_:
{
return v___x_510_;
}
}
}
}
}
}
else
{
lean_object* v_a_516_; lean_object* v___x_518_; uint8_t v_isShared_519_; uint8_t v_isSharedCheck_523_; 
lean_dec(v_snd_462_);
lean_dec(v_fst_461_);
lean_dec(v___x_457_);
lean_dec(v_goal_440_);
v_a_516_ = lean_ctor_get(v___x_463_, 0);
v_isSharedCheck_523_ = !lean_is_exclusive(v___x_463_);
if (v_isSharedCheck_523_ == 0)
{
v___x_518_ = v___x_463_;
v_isShared_519_ = v_isSharedCheck_523_;
goto v_resetjp_517_;
}
else
{
lean_inc(v_a_516_);
lean_dec(v___x_463_);
v___x_518_ = lean_box(0);
v_isShared_519_ = v_isSharedCheck_523_;
goto v_resetjp_517_;
}
v_resetjp_517_:
{
lean_object* v___x_521_; 
if (v_isShared_519_ == 0)
{
v___x_521_ = v___x_518_;
goto v_reusejp_520_;
}
else
{
lean_object* v_reuseFailAlloc_522_; 
v_reuseFailAlloc_522_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_522_, 0, v_a_516_);
v___x_521_ = v_reuseFailAlloc_522_;
goto v_reusejp_520_;
}
v_reusejp_520_:
{
return v___x_521_;
}
}
}
}
else
{
lean_object* v_a_524_; lean_object* v___x_526_; uint8_t v_isShared_527_; uint8_t v_isSharedCheck_531_; 
lean_dec(v___x_457_);
lean_dec(v_goal_440_);
v_a_524_ = lean_ctor_get(v___y_459_, 0);
v_isSharedCheck_531_ = !lean_is_exclusive(v___y_459_);
if (v_isSharedCheck_531_ == 0)
{
v___x_526_ = v___y_459_;
v_isShared_527_ = v_isSharedCheck_531_;
goto v_resetjp_525_;
}
else
{
lean_inc(v_a_524_);
lean_dec(v___y_459_);
v___x_526_ = lean_box(0);
v_isShared_527_ = v_isSharedCheck_531_;
goto v_resetjp_525_;
}
v_resetjp_525_:
{
lean_object* v___x_529_; 
if (v_isShared_527_ == 0)
{
v___x_529_ = v___x_526_;
goto v_reusejp_528_;
}
else
{
lean_object* v_reuseFailAlloc_530_; 
v_reuseFailAlloc_530_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_530_, 0, v_a_524_);
v___x_529_ = v_reuseFailAlloc_530_;
goto v_reusejp_528_;
}
v_reusejp_528_:
{
return v___x_529_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_mkInitialTree___boxed(lean_object* v_goal_614_, lean_object* v_rs_615_, lean_object* v_a_616_, lean_object* v_a_617_, lean_object* v_a_618_, lean_object* v_a_619_, lean_object* v_a_620_, lean_object* v_a_621_){
_start:
{
lean_object* v_res_622_; 
v_res_622_ = lp_aesop_Aesop_mkInitialTree(v_goal_614_, v_rs_615_, v_a_616_, v_a_617_, v_a_618_, v_a_619_, v_a_620_);
lean_dec(v_a_620_);
lean_dec_ref(v_a_619_);
lean_dec(v_a_618_);
lean_dec_ref(v_a_617_);
lean_dec(v_a_616_);
return v_res_622_;
}
}
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_mkInitialTree_spec__3_spec__6(lean_object* v_00_u03b1_623_, lean_object* v_x_624_, lean_object* v___y_625_, lean_object* v___y_626_, lean_object* v___y_627_, lean_object* v___y_628_, lean_object* v___y_629_){
_start:
{
lean_object* v___x_631_; 
v___x_631_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_mkInitialTree_spec__3_spec__6___redArg(v_x_624_);
return v___x_631_;
}
}
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_mkInitialTree_spec__3_spec__6___boxed(lean_object* v_00_u03b1_632_, lean_object* v_x_633_, lean_object* v___y_634_, lean_object* v___y_635_, lean_object* v___y_636_, lean_object* v___y_637_, lean_object* v___y_638_, lean_object* v___y_639_){
_start:
{
lean_object* v_res_640_; 
v_res_640_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_mkInitialTree_spec__3_spec__6(v_00_u03b1_632_, v_x_633_, v___y_634_, v___y_635_, v___y_636_, v___y_637_, v___y_638_);
lean_dec(v___y_638_);
lean_dec_ref(v___y_637_);
lean_dec(v___y_636_);
lean_dec_ref(v___y_635_);
lean_dec(v___y_634_);
return v_res_640_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_mkInitialTree_spec__3_spec__5(lean_object* v_oldTraces_641_, lean_object* v_data_642_, lean_object* v_ref_643_, lean_object* v_msg_644_, lean_object* v___y_645_, lean_object* v___y_646_, lean_object* v___y_647_, lean_object* v___y_648_, lean_object* v___y_649_){
_start:
{
lean_object* v___x_651_; 
v___x_651_ = lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_mkInitialTree_spec__3_spec__5___redArg(v_oldTraces_641_, v_data_642_, v_ref_643_, v_msg_644_, v___y_646_, v___y_647_, v___y_648_, v___y_649_);
return v___x_651_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_mkInitialTree_spec__3_spec__5___boxed(lean_object* v_oldTraces_652_, lean_object* v_data_653_, lean_object* v_ref_654_, lean_object* v_msg_655_, lean_object* v___y_656_, lean_object* v___y_657_, lean_object* v___y_658_, lean_object* v___y_659_, lean_object* v___y_660_, lean_object* v___y_661_){
_start:
{
lean_object* v_res_662_; 
v_res_662_ = lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_mkInitialTree_spec__3_spec__5(v_oldTraces_652_, v_data_653_, v_ref_654_, v_msg_655_, v___y_656_, v___y_657_, v___y_658_, v___y_659_, v___y_660_);
lean_dec(v___y_660_);
lean_dec_ref(v___y_659_);
lean_dec(v___y_658_);
lean_dec_ref(v___y_657_);
lean_dec(v___y_656_);
return v_res_662_;
}
}
static lean_object* _init_lp_aesop_Aesop_TreeM_instMonad___closed__0(void){
_start:
{
lean_object* v___x_663_; 
v___x_663_ = l_instMonadEIO(lean_box(0));
return v___x_663_;
}
}
static lean_object* _init_lp_aesop_Aesop_TreeM_instMonad___closed__1(void){
_start:
{
lean_object* v___x_664_; lean_object* v___x_665_; 
v___x_664_ = lean_obj_once(&lp_aesop_Aesop_TreeM_instMonad___closed__0, &lp_aesop_Aesop_TreeM_instMonad___closed__0_once, _init_lp_aesop_Aesop_TreeM_instMonad___closed__0);
v___x_665_ = l_StateRefT_x27_instMonad___redArg(v___x_664_);
return v___x_665_;
}
}
static lean_object* _init_lp_aesop_Aesop_TreeM_instMonad(void){
_start:
{
lean_object* v___x_670_; lean_object* v_toApplicative_671_; lean_object* v_toFunctor_672_; lean_object* v_toSeq_673_; lean_object* v_toSeqLeft_674_; lean_object* v_toSeqRight_675_; lean_object* v___f_676_; lean_object* v___f_677_; lean_object* v___f_678_; lean_object* v___f_679_; lean_object* v___x_680_; lean_object* v___f_681_; lean_object* v___f_682_; lean_object* v___f_683_; lean_object* v___x_684_; lean_object* v___x_685_; lean_object* v___x_686_; lean_object* v_toApplicative_687_; lean_object* v___x_689_; uint8_t v_isShared_690_; uint8_t v_isSharedCheck_717_; 
v___x_670_ = lean_obj_once(&lp_aesop_Aesop_TreeM_instMonad___closed__1, &lp_aesop_Aesop_TreeM_instMonad___closed__1_once, _init_lp_aesop_Aesop_TreeM_instMonad___closed__1);
v_toApplicative_671_ = lean_ctor_get(v___x_670_, 0);
v_toFunctor_672_ = lean_ctor_get(v_toApplicative_671_, 0);
v_toSeq_673_ = lean_ctor_get(v_toApplicative_671_, 2);
v_toSeqLeft_674_ = lean_ctor_get(v_toApplicative_671_, 3);
v_toSeqRight_675_ = lean_ctor_get(v_toApplicative_671_, 4);
v___f_676_ = ((lean_object*)(lp_aesop_Aesop_TreeM_instMonad___closed__2));
v___f_677_ = ((lean_object*)(lp_aesop_Aesop_TreeM_instMonad___closed__3));
lean_inc_ref_n(v_toFunctor_672_, 2);
v___f_678_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_678_, 0, v_toFunctor_672_);
v___f_679_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_679_, 0, v_toFunctor_672_);
v___x_680_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_680_, 0, v___f_678_);
lean_ctor_set(v___x_680_, 1, v___f_679_);
lean_inc(v_toSeqRight_675_);
v___f_681_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_681_, 0, v_toSeqRight_675_);
lean_inc(v_toSeqLeft_674_);
v___f_682_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_682_, 0, v_toSeqLeft_674_);
lean_inc(v_toSeq_673_);
v___f_683_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_683_, 0, v_toSeq_673_);
v___x_684_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_684_, 0, v___x_680_);
lean_ctor_set(v___x_684_, 1, v___f_676_);
lean_ctor_set(v___x_684_, 2, v___f_683_);
lean_ctor_set(v___x_684_, 3, v___f_682_);
lean_ctor_set(v___x_684_, 4, v___f_681_);
v___x_685_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_685_, 0, v___x_684_);
lean_ctor_set(v___x_685_, 1, v___f_677_);
v___x_686_ = l_StateRefT_x27_instMonad___redArg(v___x_685_);
v_toApplicative_687_ = lean_ctor_get(v___x_686_, 0);
v_isSharedCheck_717_ = !lean_is_exclusive(v___x_686_);
if (v_isSharedCheck_717_ == 0)
{
lean_object* v_unused_718_; 
v_unused_718_ = lean_ctor_get(v___x_686_, 1);
lean_dec(v_unused_718_);
v___x_689_ = v___x_686_;
v_isShared_690_ = v_isSharedCheck_717_;
goto v_resetjp_688_;
}
else
{
lean_inc(v_toApplicative_687_);
lean_dec(v___x_686_);
v___x_689_ = lean_box(0);
v_isShared_690_ = v_isSharedCheck_717_;
goto v_resetjp_688_;
}
v_resetjp_688_:
{
lean_object* v_toFunctor_691_; lean_object* v_toSeq_692_; lean_object* v_toSeqLeft_693_; lean_object* v_toSeqRight_694_; lean_object* v___x_696_; uint8_t v_isShared_697_; uint8_t v_isSharedCheck_715_; 
v_toFunctor_691_ = lean_ctor_get(v_toApplicative_687_, 0);
v_toSeq_692_ = lean_ctor_get(v_toApplicative_687_, 2);
v_toSeqLeft_693_ = lean_ctor_get(v_toApplicative_687_, 3);
v_toSeqRight_694_ = lean_ctor_get(v_toApplicative_687_, 4);
v_isSharedCheck_715_ = !lean_is_exclusive(v_toApplicative_687_);
if (v_isSharedCheck_715_ == 0)
{
lean_object* v_unused_716_; 
v_unused_716_ = lean_ctor_get(v_toApplicative_687_, 1);
lean_dec(v_unused_716_);
v___x_696_ = v_toApplicative_687_;
v_isShared_697_ = v_isSharedCheck_715_;
goto v_resetjp_695_;
}
else
{
lean_inc(v_toSeqRight_694_);
lean_inc(v_toSeqLeft_693_);
lean_inc(v_toSeq_692_);
lean_inc(v_toFunctor_691_);
lean_dec(v_toApplicative_687_);
v___x_696_ = lean_box(0);
v_isShared_697_ = v_isSharedCheck_715_;
goto v_resetjp_695_;
}
v_resetjp_695_:
{
lean_object* v___f_698_; lean_object* v___f_699_; lean_object* v___f_700_; lean_object* v___f_701_; lean_object* v___x_702_; lean_object* v___f_703_; lean_object* v___f_704_; lean_object* v___f_705_; lean_object* v___x_707_; 
v___f_698_ = ((lean_object*)(lp_aesop_Aesop_TreeM_instMonad___closed__4));
v___f_699_ = ((lean_object*)(lp_aesop_Aesop_TreeM_instMonad___closed__5));
lean_inc_ref(v_toFunctor_691_);
v___f_700_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_700_, 0, v_toFunctor_691_);
v___f_701_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_701_, 0, v_toFunctor_691_);
v___x_702_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_702_, 0, v___f_700_);
lean_ctor_set(v___x_702_, 1, v___f_701_);
v___f_703_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_703_, 0, v_toSeqRight_694_);
v___f_704_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_704_, 0, v_toSeqLeft_693_);
v___f_705_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_705_, 0, v_toSeq_692_);
if (v_isShared_697_ == 0)
{
lean_ctor_set(v___x_696_, 4, v___f_703_);
lean_ctor_set(v___x_696_, 3, v___f_704_);
lean_ctor_set(v___x_696_, 2, v___f_705_);
lean_ctor_set(v___x_696_, 1, v___f_698_);
lean_ctor_set(v___x_696_, 0, v___x_702_);
v___x_707_ = v___x_696_;
goto v_reusejp_706_;
}
else
{
lean_object* v_reuseFailAlloc_714_; 
v_reuseFailAlloc_714_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_714_, 0, v___x_702_);
lean_ctor_set(v_reuseFailAlloc_714_, 1, v___f_698_);
lean_ctor_set(v_reuseFailAlloc_714_, 2, v___f_705_);
lean_ctor_set(v_reuseFailAlloc_714_, 3, v___f_704_);
lean_ctor_set(v_reuseFailAlloc_714_, 4, v___f_703_);
v___x_707_ = v_reuseFailAlloc_714_;
goto v_reusejp_706_;
}
v_reusejp_706_:
{
lean_object* v___x_709_; 
if (v_isShared_690_ == 0)
{
lean_ctor_set(v___x_689_, 1, v___f_699_);
lean_ctor_set(v___x_689_, 0, v___x_707_);
v___x_709_ = v___x_689_;
goto v_reusejp_708_;
}
else
{
lean_object* v_reuseFailAlloc_713_; 
v_reuseFailAlloc_713_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_713_, 0, v___x_707_);
lean_ctor_set(v_reuseFailAlloc_713_, 1, v___f_699_);
v___x_709_ = v_reuseFailAlloc_713_;
goto v_reusejp_708_;
}
v_reusejp_708_:
{
lean_object* v___x_710_; lean_object* v___x_711_; lean_object* v___x_712_; 
v___x_710_ = l_StateRefT_x27_instMonad___redArg(v___x_709_);
v___x_711_ = l_StateRefT_x27_instMonad___redArg(v___x_710_);
v___x_712_ = l_ReaderT_instMonad___redArg(v___x_711_);
return v___x_712_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TreeM_instMonadStateOfTree___lam__0(lean_object* v___y_719_, lean_object* v___y_720_, lean_object* v___y_721_, lean_object* v___y_722_, lean_object* v___y_723_, lean_object* v___y_724_, lean_object* v___y_725_){
_start:
{
lean_object* v___x_727_; lean_object* v___x_728_; 
v___x_727_ = lean_st_ref_get(v___y_720_);
v___x_728_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_728_, 0, v___x_727_);
return v___x_728_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TreeM_instMonadStateOfTree___lam__0___boxed(lean_object* v___y_729_, lean_object* v___y_730_, lean_object* v___y_731_, lean_object* v___y_732_, lean_object* v___y_733_, lean_object* v___y_734_, lean_object* v___y_735_, lean_object* v___y_736_){
_start:
{
lean_object* v_res_737_; 
v_res_737_ = lp_aesop_Aesop_TreeM_instMonadStateOfTree___lam__0(v___y_729_, v___y_730_, v___y_731_, v___y_732_, v___y_733_, v___y_734_, v___y_735_);
lean_dec(v___y_735_);
lean_dec_ref(v___y_734_);
lean_dec(v___y_733_);
lean_dec_ref(v___y_732_);
lean_dec(v___y_731_);
lean_dec(v___y_730_);
lean_dec_ref(v___y_729_);
return v_res_737_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TreeM_instMonadStateOfTree___lam__1(lean_object* v_____do__lift_738_, lean_object* v___y_739_, lean_object* v___y_740_, lean_object* v___y_741_, lean_object* v___y_742_, lean_object* v___y_743_, lean_object* v___y_744_, lean_object* v___y_745_){
_start:
{
lean_object* v___x_747_; 
v___x_747_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_747_, 0, v_____do__lift_738_);
return v___x_747_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TreeM_instMonadStateOfTree___lam__1___boxed(lean_object* v_____do__lift_748_, lean_object* v___y_749_, lean_object* v___y_750_, lean_object* v___y_751_, lean_object* v___y_752_, lean_object* v___y_753_, lean_object* v___y_754_, lean_object* v___y_755_, lean_object* v___y_756_){
_start:
{
lean_object* v_res_757_; 
v_res_757_ = lp_aesop_Aesop_TreeM_instMonadStateOfTree___lam__1(v_____do__lift_748_, v___y_749_, v___y_750_, v___y_751_, v___y_752_, v___y_753_, v___y_754_, v___y_755_);
lean_dec(v___y_755_);
lean_dec_ref(v___y_754_);
lean_dec(v___y_753_);
lean_dec_ref(v___y_752_);
lean_dec(v___y_751_);
lean_dec(v___y_750_);
lean_dec_ref(v___y_749_);
return v_res_757_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TreeM_instMonadStateOfTree___lam__2(lean_object* v_tree_758_, lean_object* v___y_759_, lean_object* v___y_760_, lean_object* v___y_761_, lean_object* v___y_762_, lean_object* v___y_763_, lean_object* v___y_764_, lean_object* v___y_765_){
_start:
{
lean_object* v___x_767_; lean_object* v___x_768_; lean_object* v___x_769_; lean_object* v___x_770_; 
v___x_767_ = lean_st_ref_take(v___y_760_);
lean_dec(v___x_767_);
v___x_768_ = lean_st_ref_set(v___y_760_, v_tree_758_);
v___x_769_ = lean_box(0);
v___x_770_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_770_, 0, v___x_769_);
return v___x_770_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TreeM_instMonadStateOfTree___lam__2___boxed(lean_object* v_tree_771_, lean_object* v___y_772_, lean_object* v___y_773_, lean_object* v___y_774_, lean_object* v___y_775_, lean_object* v___y_776_, lean_object* v___y_777_, lean_object* v___y_778_, lean_object* v___y_779_){
_start:
{
lean_object* v_res_780_; 
v_res_780_ = lp_aesop_Aesop_TreeM_instMonadStateOfTree___lam__2(v_tree_771_, v___y_772_, v___y_773_, v___y_774_, v___y_775_, v___y_776_, v___y_777_, v___y_778_);
lean_dec(v___y_778_);
lean_dec_ref(v___y_777_);
lean_dec(v___y_776_);
lean_dec_ref(v___y_775_);
lean_dec(v___y_774_);
lean_dec(v___y_773_);
lean_dec_ref(v___y_772_);
return v_res_780_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TreeM_instMonadStateOfTree___lam__3(lean_object* v_00_u03b1_781_, lean_object* v_f_782_, lean_object* v___y_783_, lean_object* v___y_784_, lean_object* v___y_785_, lean_object* v___y_786_, lean_object* v___y_787_, lean_object* v___y_788_, lean_object* v___y_789_){
_start:
{
lean_object* v___x_791_; lean_object* v___x_792_; lean_object* v_fst_793_; lean_object* v_snd_794_; lean_object* v___x_795_; lean_object* v___x_796_; 
v___x_791_ = lean_st_ref_take(v___y_784_);
v___x_792_ = lean_apply_1(v_f_782_, v___x_791_);
v_fst_793_ = lean_ctor_get(v___x_792_, 0);
lean_inc(v_fst_793_);
v_snd_794_ = lean_ctor_get(v___x_792_, 1);
lean_inc(v_snd_794_);
lean_dec_ref(v___x_792_);
v___x_795_ = lean_st_ref_set(v___y_784_, v_snd_794_);
v___x_796_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_796_, 0, v_fst_793_);
return v___x_796_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TreeM_instMonadStateOfTree___lam__3___boxed(lean_object* v_00_u03b1_797_, lean_object* v_f_798_, lean_object* v___y_799_, lean_object* v___y_800_, lean_object* v___y_801_, lean_object* v___y_802_, lean_object* v___y_803_, lean_object* v___y_804_, lean_object* v___y_805_, lean_object* v___y_806_){
_start:
{
lean_object* v_res_807_; 
v_res_807_ = lp_aesop_Aesop_TreeM_instMonadStateOfTree___lam__3(v_00_u03b1_797_, v_f_798_, v___y_799_, v___y_800_, v___y_801_, v___y_802_, v___y_803_, v___y_804_, v___y_805_);
lean_dec(v___y_805_);
lean_dec_ref(v___y_804_);
lean_dec(v___y_803_);
lean_dec_ref(v___y_802_);
lean_dec(v___y_801_);
lean_dec(v___y_800_);
lean_dec_ref(v___y_799_);
return v_res_807_;
}
}
static lean_object* _init_lp_aesop_Aesop_TreeM_instMonadStateOfTree(void){
_start:
{
lean_object* v___x_812_; lean_object* v_toApplicative_813_; lean_object* v_toFunctor_814_; lean_object* v_toSeq_815_; lean_object* v_toSeqLeft_816_; lean_object* v_toSeqRight_817_; lean_object* v___f_818_; lean_object* v___f_819_; lean_object* v___f_820_; lean_object* v___f_821_; lean_object* v___x_822_; lean_object* v___f_823_; lean_object* v___f_824_; lean_object* v___f_825_; lean_object* v___x_826_; lean_object* v___x_827_; lean_object* v___x_828_; lean_object* v_toApplicative_829_; lean_object* v___x_831_; uint8_t v_isShared_832_; uint8_t v_isSharedCheck_864_; 
v___x_812_ = lean_obj_once(&lp_aesop_Aesop_TreeM_instMonad___closed__1, &lp_aesop_Aesop_TreeM_instMonad___closed__1_once, _init_lp_aesop_Aesop_TreeM_instMonad___closed__1);
v_toApplicative_813_ = lean_ctor_get(v___x_812_, 0);
v_toFunctor_814_ = lean_ctor_get(v_toApplicative_813_, 0);
v_toSeq_815_ = lean_ctor_get(v_toApplicative_813_, 2);
v_toSeqLeft_816_ = lean_ctor_get(v_toApplicative_813_, 3);
v_toSeqRight_817_ = lean_ctor_get(v_toApplicative_813_, 4);
v___f_818_ = ((lean_object*)(lp_aesop_Aesop_TreeM_instMonad___closed__2));
v___f_819_ = ((lean_object*)(lp_aesop_Aesop_TreeM_instMonad___closed__3));
lean_inc_ref_n(v_toFunctor_814_, 2);
v___f_820_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_820_, 0, v_toFunctor_814_);
v___f_821_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_821_, 0, v_toFunctor_814_);
v___x_822_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_822_, 0, v___f_820_);
lean_ctor_set(v___x_822_, 1, v___f_821_);
lean_inc(v_toSeqRight_817_);
v___f_823_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_823_, 0, v_toSeqRight_817_);
lean_inc(v_toSeqLeft_816_);
v___f_824_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_824_, 0, v_toSeqLeft_816_);
lean_inc(v_toSeq_815_);
v___f_825_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_825_, 0, v_toSeq_815_);
v___x_826_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_826_, 0, v___x_822_);
lean_ctor_set(v___x_826_, 1, v___f_818_);
lean_ctor_set(v___x_826_, 2, v___f_825_);
lean_ctor_set(v___x_826_, 3, v___f_824_);
lean_ctor_set(v___x_826_, 4, v___f_823_);
v___x_827_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_827_, 0, v___x_826_);
lean_ctor_set(v___x_827_, 1, v___f_819_);
v___x_828_ = l_StateRefT_x27_instMonad___redArg(v___x_827_);
v_toApplicative_829_ = lean_ctor_get(v___x_828_, 0);
v_isSharedCheck_864_ = !lean_is_exclusive(v___x_828_);
if (v_isSharedCheck_864_ == 0)
{
lean_object* v_unused_865_; 
v_unused_865_ = lean_ctor_get(v___x_828_, 1);
lean_dec(v_unused_865_);
v___x_831_ = v___x_828_;
v_isShared_832_ = v_isSharedCheck_864_;
goto v_resetjp_830_;
}
else
{
lean_inc(v_toApplicative_829_);
lean_dec(v___x_828_);
v___x_831_ = lean_box(0);
v_isShared_832_ = v_isSharedCheck_864_;
goto v_resetjp_830_;
}
v_resetjp_830_:
{
lean_object* v_toFunctor_833_; lean_object* v_toSeq_834_; lean_object* v_toSeqLeft_835_; lean_object* v_toSeqRight_836_; lean_object* v___x_838_; uint8_t v_isShared_839_; uint8_t v_isSharedCheck_862_; 
v_toFunctor_833_ = lean_ctor_get(v_toApplicative_829_, 0);
v_toSeq_834_ = lean_ctor_get(v_toApplicative_829_, 2);
v_toSeqLeft_835_ = lean_ctor_get(v_toApplicative_829_, 3);
v_toSeqRight_836_ = lean_ctor_get(v_toApplicative_829_, 4);
v_isSharedCheck_862_ = !lean_is_exclusive(v_toApplicative_829_);
if (v_isSharedCheck_862_ == 0)
{
lean_object* v_unused_863_; 
v_unused_863_ = lean_ctor_get(v_toApplicative_829_, 1);
lean_dec(v_unused_863_);
v___x_838_ = v_toApplicative_829_;
v_isShared_839_ = v_isSharedCheck_862_;
goto v_resetjp_837_;
}
else
{
lean_inc(v_toSeqRight_836_);
lean_inc(v_toSeqLeft_835_);
lean_inc(v_toSeq_834_);
lean_inc(v_toFunctor_833_);
lean_dec(v_toApplicative_829_);
v___x_838_ = lean_box(0);
v_isShared_839_ = v_isSharedCheck_862_;
goto v_resetjp_837_;
}
v_resetjp_837_:
{
lean_object* v___f_840_; lean_object* v___f_841_; lean_object* v___f_842_; lean_object* v___f_843_; lean_object* v___f_844_; lean_object* v___f_845_; lean_object* v___f_846_; lean_object* v___f_847_; lean_object* v___x_848_; lean_object* v___f_849_; lean_object* v___f_850_; lean_object* v___f_851_; lean_object* v___x_853_; 
v___f_840_ = ((lean_object*)(lp_aesop_Aesop_TreeM_instMonadStateOfTree___closed__0));
v___f_841_ = ((lean_object*)(lp_aesop_Aesop_TreeM_instMonadStateOfTree___closed__1));
v___f_842_ = ((lean_object*)(lp_aesop_Aesop_TreeM_instMonadStateOfTree___closed__2));
v___f_843_ = ((lean_object*)(lp_aesop_Aesop_TreeM_instMonadStateOfTree___closed__3));
v___f_844_ = ((lean_object*)(lp_aesop_Aesop_TreeM_instMonad___closed__4));
v___f_845_ = ((lean_object*)(lp_aesop_Aesop_TreeM_instMonad___closed__5));
lean_inc_ref(v_toFunctor_833_);
v___f_846_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_846_, 0, v_toFunctor_833_);
v___f_847_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_847_, 0, v_toFunctor_833_);
v___x_848_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_848_, 0, v___f_846_);
lean_ctor_set(v___x_848_, 1, v___f_847_);
v___f_849_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_849_, 0, v_toSeqRight_836_);
v___f_850_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_850_, 0, v_toSeqLeft_835_);
v___f_851_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_851_, 0, v_toSeq_834_);
if (v_isShared_839_ == 0)
{
lean_ctor_set(v___x_838_, 4, v___f_849_);
lean_ctor_set(v___x_838_, 3, v___f_850_);
lean_ctor_set(v___x_838_, 2, v___f_851_);
lean_ctor_set(v___x_838_, 1, v___f_844_);
lean_ctor_set(v___x_838_, 0, v___x_848_);
v___x_853_ = v___x_838_;
goto v_reusejp_852_;
}
else
{
lean_object* v_reuseFailAlloc_861_; 
v_reuseFailAlloc_861_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_861_, 0, v___x_848_);
lean_ctor_set(v_reuseFailAlloc_861_, 1, v___f_844_);
lean_ctor_set(v_reuseFailAlloc_861_, 2, v___f_851_);
lean_ctor_set(v_reuseFailAlloc_861_, 3, v___f_850_);
lean_ctor_set(v_reuseFailAlloc_861_, 4, v___f_849_);
v___x_853_ = v_reuseFailAlloc_861_;
goto v_reusejp_852_;
}
v_reusejp_852_:
{
lean_object* v___x_855_; 
if (v_isShared_832_ == 0)
{
lean_ctor_set(v___x_831_, 1, v___f_845_);
lean_ctor_set(v___x_831_, 0, v___x_853_);
v___x_855_ = v___x_831_;
goto v_reusejp_854_;
}
else
{
lean_object* v_reuseFailAlloc_860_; 
v_reuseFailAlloc_860_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_860_, 0, v___x_853_);
lean_ctor_set(v_reuseFailAlloc_860_, 1, v___f_845_);
v___x_855_ = v_reuseFailAlloc_860_;
goto v_reusejp_854_;
}
v_reusejp_854_:
{
lean_object* v___x_856_; lean_object* v___x_857_; lean_object* v___x_858_; lean_object* v___x_859_; 
v___x_856_ = l_StateRefT_x27_instMonad___redArg(v___x_855_);
v___x_857_ = l_StateRefT_x27_instMonad___redArg(v___x_856_);
v___x_858_ = lean_alloc_closure((void*)(l_ReaderT_bind___boxed), 8, 7);
lean_closure_set(v___x_858_, 0, lean_box(0));
lean_closure_set(v___x_858_, 1, lean_box(0));
lean_closure_set(v___x_858_, 2, v___x_857_);
lean_closure_set(v___x_858_, 3, lean_box(0));
lean_closure_set(v___x_858_, 4, lean_box(0));
lean_closure_set(v___x_858_, 5, v___f_840_);
lean_closure_set(v___x_858_, 6, v___f_841_);
v___x_859_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_859_, 0, v___x_858_);
lean_ctor_set(v___x_859_, 1, v___f_842_);
lean_ctor_set(v___x_859_, 2, v___f_843_);
return v___x_859_;
}
}
}
}
}
}
static lean_object* _init_lp_aesop_Aesop_TreeM_instInhabited___lam__0___closed__0(void){
_start:
{
lean_object* v___x_866_; lean_object* v___f_867_; 
v___x_866_ = l_Lean_instMonadExceptOfExceptionCoreM;
v___f_867_ = lean_alloc_closure((void*)(l_StateRefT_x27_instMonadExceptOf___redArg___lam__0___boxed), 4, 1);
lean_closure_set(v___f_867_, 0, v___x_866_);
return v___f_867_;
}
}
static lean_object* _init_lp_aesop_Aesop_TreeM_instInhabited___lam__0___closed__1(void){
_start:
{
lean_object* v___x_868_; lean_object* v___f_869_; 
v___x_868_ = l_Lean_instMonadExceptOfExceptionCoreM;
v___f_869_ = lean_alloc_closure((void*)(l_StateRefT_x27_instMonadExceptOf___redArg___lam__2), 5, 1);
lean_closure_set(v___f_869_, 0, v___x_868_);
return v___f_869_;
}
}
static lean_object* _init_lp_aesop_Aesop_TreeM_instInhabited___lam__0___closed__2(void){
_start:
{
lean_object* v___f_870_; lean_object* v___f_871_; lean_object* v___x_872_; 
v___f_870_ = lean_obj_once(&lp_aesop_Aesop_TreeM_instInhabited___lam__0___closed__1, &lp_aesop_Aesop_TreeM_instInhabited___lam__0___closed__1_once, _init_lp_aesop_Aesop_TreeM_instInhabited___lam__0___closed__1);
v___f_871_ = lean_obj_once(&lp_aesop_Aesop_TreeM_instInhabited___lam__0___closed__0, &lp_aesop_Aesop_TreeM_instInhabited___lam__0___closed__0_once, _init_lp_aesop_Aesop_TreeM_instInhabited___lam__0___closed__0);
v___x_872_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_872_, 0, v___f_871_);
lean_ctor_set(v___x_872_, 1, v___f_870_);
return v___x_872_;
}
}
static lean_object* _init_lp_aesop_Aesop_TreeM_instInhabited___lam__0___closed__3(void){
_start:
{
lean_object* v___x_873_; lean_object* v___f_874_; 
v___x_873_ = lean_obj_once(&lp_aesop_Aesop_TreeM_instInhabited___lam__0___closed__2, &lp_aesop_Aesop_TreeM_instInhabited___lam__0___closed__2_once, _init_lp_aesop_Aesop_TreeM_instInhabited___lam__0___closed__2);
v___f_874_ = lean_alloc_closure((void*)(l_ReaderT_instMonadExceptOf___redArg___lam__0___boxed), 4, 1);
lean_closure_set(v___f_874_, 0, v___x_873_);
return v___f_874_;
}
}
static lean_object* _init_lp_aesop_Aesop_TreeM_instInhabited___lam__0___closed__4(void){
_start:
{
lean_object* v___x_875_; lean_object* v___f_876_; 
v___x_875_ = lean_obj_once(&lp_aesop_Aesop_TreeM_instInhabited___lam__0___closed__2, &lp_aesop_Aesop_TreeM_instInhabited___lam__0___closed__2_once, _init_lp_aesop_Aesop_TreeM_instInhabited___lam__0___closed__2);
v___f_876_ = lean_alloc_closure((void*)(l_ReaderT_instMonadExceptOf___redArg___lam__2), 5, 1);
lean_closure_set(v___f_876_, 0, v___x_875_);
return v___f_876_;
}
}
static lean_object* _init_lp_aesop_Aesop_TreeM_instInhabited___lam__0___closed__5(void){
_start:
{
lean_object* v___f_877_; lean_object* v___f_878_; lean_object* v___x_879_; 
v___f_877_ = lean_obj_once(&lp_aesop_Aesop_TreeM_instInhabited___lam__0___closed__4, &lp_aesop_Aesop_TreeM_instInhabited___lam__0___closed__4_once, _init_lp_aesop_Aesop_TreeM_instInhabited___lam__0___closed__4);
v___f_878_ = lean_obj_once(&lp_aesop_Aesop_TreeM_instInhabited___lam__0___closed__3, &lp_aesop_Aesop_TreeM_instInhabited___lam__0___closed__3_once, _init_lp_aesop_Aesop_TreeM_instInhabited___lam__0___closed__3);
v___x_879_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_879_, 0, v___f_878_);
lean_ctor_set(v___x_879_, 1, v___f_877_);
return v___x_879_;
}
}
static lean_object* _init_lp_aesop_Aesop_TreeM_instInhabited___lam__0___closed__10(void){
_start:
{
lean_object* v___x_884_; lean_object* v___x_885_; lean_object* v___x_886_; lean_object* v___x_887_; 
v___x_884_ = l_Lean_Core_instMonadQuotationCoreM;
v___x_885_ = ((lean_object*)(lp_aesop_Aesop_TreeM_instInhabited___lam__0___closed__9));
v___x_886_ = ((lean_object*)(lp_aesop_Aesop_TreeM_instInhabited___lam__0___closed__8));
v___x_887_ = l_Lean_instMonadQuotationOfMonadFunctorOfMonadLift___redArg(v___x_886_, v___x_885_, v___x_884_);
return v___x_887_;
}
}
static lean_object* _init_lp_aesop_Aesop_TreeM_instInhabited___lam__0___closed__11(void){
_start:
{
lean_object* v___x_888_; lean_object* v___f_889_; lean_object* v___f_890_; lean_object* v___x_891_; 
v___x_888_ = lean_obj_once(&lp_aesop_Aesop_TreeM_instInhabited___lam__0___closed__10, &lp_aesop_Aesop_TreeM_instInhabited___lam__0___closed__10_once, _init_lp_aesop_Aesop_TreeM_instInhabited___lam__0___closed__10);
v___f_889_ = ((lean_object*)(lp_aesop_Aesop_TreeM_instInhabited___lam__0___closed__7));
v___f_890_ = ((lean_object*)(lp_aesop_Aesop_TreeM_instInhabited___lam__0___closed__6));
v___x_891_ = l_Lean_instMonadQuotationOfMonadFunctorOfMonadLift___redArg(v___f_890_, v___f_889_, v___x_888_);
return v___x_891_;
}
}
static lean_object* _init_lp_aesop_Aesop_TreeM_instInhabited___lam__0___closed__13(void){
_start:
{
lean_object* v___x_893_; lean_object* v___x_894_; 
v___x_893_ = ((lean_object*)(lp_aesop_Aesop_TreeM_instInhabited___lam__0___closed__12));
v___x_894_ = l_Lean_stringToMessageData(v___x_893_);
return v___x_894_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TreeM_instInhabited___lam__0(lean_object* v___y_895_, lean_object* v___y_896_, lean_object* v___y_897_, lean_object* v___y_898_, lean_object* v___y_899_, lean_object* v___y_900_, lean_object* v___y_901_){
_start:
{
lean_object* v___x_903_; lean_object* v_toApplicative_904_; lean_object* v_toFunctor_905_; lean_object* v_toSeq_906_; lean_object* v_toSeqLeft_907_; lean_object* v_toSeqRight_908_; lean_object* v___f_909_; lean_object* v___f_910_; lean_object* v___f_911_; lean_object* v___f_912_; lean_object* v___x_913_; lean_object* v___f_914_; lean_object* v___f_915_; lean_object* v___f_916_; lean_object* v___x_917_; lean_object* v___x_918_; lean_object* v___x_919_; lean_object* v_toApplicative_920_; lean_object* v___x_922_; uint8_t v_isShared_923_; uint8_t v_isSharedCheck_956_; 
v___x_903_ = lean_obj_once(&lp_aesop_Aesop_TreeM_instMonad___closed__1, &lp_aesop_Aesop_TreeM_instMonad___closed__1_once, _init_lp_aesop_Aesop_TreeM_instMonad___closed__1);
v_toApplicative_904_ = lean_ctor_get(v___x_903_, 0);
v_toFunctor_905_ = lean_ctor_get(v_toApplicative_904_, 0);
v_toSeq_906_ = lean_ctor_get(v_toApplicative_904_, 2);
v_toSeqLeft_907_ = lean_ctor_get(v_toApplicative_904_, 3);
v_toSeqRight_908_ = lean_ctor_get(v_toApplicative_904_, 4);
v___f_909_ = ((lean_object*)(lp_aesop_Aesop_TreeM_instMonad___closed__2));
v___f_910_ = ((lean_object*)(lp_aesop_Aesop_TreeM_instMonad___closed__3));
lean_inc_ref_n(v_toFunctor_905_, 2);
v___f_911_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_911_, 0, v_toFunctor_905_);
v___f_912_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_912_, 0, v_toFunctor_905_);
v___x_913_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_913_, 0, v___f_911_);
lean_ctor_set(v___x_913_, 1, v___f_912_);
lean_inc(v_toSeqRight_908_);
v___f_914_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_914_, 0, v_toSeqRight_908_);
lean_inc(v_toSeqLeft_907_);
v___f_915_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_915_, 0, v_toSeqLeft_907_);
lean_inc(v_toSeq_906_);
v___f_916_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_916_, 0, v_toSeq_906_);
v___x_917_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_917_, 0, v___x_913_);
lean_ctor_set(v___x_917_, 1, v___f_909_);
lean_ctor_set(v___x_917_, 2, v___f_916_);
lean_ctor_set(v___x_917_, 3, v___f_915_);
lean_ctor_set(v___x_917_, 4, v___f_914_);
v___x_918_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_918_, 0, v___x_917_);
lean_ctor_set(v___x_918_, 1, v___f_910_);
v___x_919_ = l_StateRefT_x27_instMonad___redArg(v___x_918_);
v_toApplicative_920_ = lean_ctor_get(v___x_919_, 0);
v_isSharedCheck_956_ = !lean_is_exclusive(v___x_919_);
if (v_isSharedCheck_956_ == 0)
{
lean_object* v_unused_957_; 
v_unused_957_ = lean_ctor_get(v___x_919_, 1);
lean_dec(v_unused_957_);
v___x_922_ = v___x_919_;
v_isShared_923_ = v_isSharedCheck_956_;
goto v_resetjp_921_;
}
else
{
lean_inc(v_toApplicative_920_);
lean_dec(v___x_919_);
v___x_922_ = lean_box(0);
v_isShared_923_ = v_isSharedCheck_956_;
goto v_resetjp_921_;
}
v_resetjp_921_:
{
lean_object* v_toFunctor_924_; lean_object* v_toSeq_925_; lean_object* v_toSeqLeft_926_; lean_object* v_toSeqRight_927_; lean_object* v___x_929_; uint8_t v_isShared_930_; uint8_t v_isSharedCheck_954_; 
v_toFunctor_924_ = lean_ctor_get(v_toApplicative_920_, 0);
v_toSeq_925_ = lean_ctor_get(v_toApplicative_920_, 2);
v_toSeqLeft_926_ = lean_ctor_get(v_toApplicative_920_, 3);
v_toSeqRight_927_ = lean_ctor_get(v_toApplicative_920_, 4);
v_isSharedCheck_954_ = !lean_is_exclusive(v_toApplicative_920_);
if (v_isSharedCheck_954_ == 0)
{
lean_object* v_unused_955_; 
v_unused_955_ = lean_ctor_get(v_toApplicative_920_, 1);
lean_dec(v_unused_955_);
v___x_929_ = v_toApplicative_920_;
v_isShared_930_ = v_isSharedCheck_954_;
goto v_resetjp_928_;
}
else
{
lean_inc(v_toSeqRight_927_);
lean_inc(v_toSeqLeft_926_);
lean_inc(v_toSeq_925_);
lean_inc(v_toFunctor_924_);
lean_dec(v_toApplicative_920_);
v___x_929_ = lean_box(0);
v_isShared_930_ = v_isSharedCheck_954_;
goto v_resetjp_928_;
}
v_resetjp_928_:
{
lean_object* v___f_931_; lean_object* v___f_932_; lean_object* v___f_933_; lean_object* v___f_934_; lean_object* v___x_935_; lean_object* v___f_936_; lean_object* v___f_937_; lean_object* v___f_938_; lean_object* v___x_940_; 
v___f_931_ = ((lean_object*)(lp_aesop_Aesop_TreeM_instMonad___closed__4));
v___f_932_ = ((lean_object*)(lp_aesop_Aesop_TreeM_instMonad___closed__5));
lean_inc_ref(v_toFunctor_924_);
v___f_933_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_933_, 0, v_toFunctor_924_);
v___f_934_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_934_, 0, v_toFunctor_924_);
v___x_935_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_935_, 0, v___f_933_);
lean_ctor_set(v___x_935_, 1, v___f_934_);
v___f_936_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_936_, 0, v_toSeqRight_927_);
v___f_937_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_937_, 0, v_toSeqLeft_926_);
v___f_938_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_938_, 0, v_toSeq_925_);
if (v_isShared_930_ == 0)
{
lean_ctor_set(v___x_929_, 4, v___f_936_);
lean_ctor_set(v___x_929_, 3, v___f_937_);
lean_ctor_set(v___x_929_, 2, v___f_938_);
lean_ctor_set(v___x_929_, 1, v___f_931_);
lean_ctor_set(v___x_929_, 0, v___x_935_);
v___x_940_ = v___x_929_;
goto v_reusejp_939_;
}
else
{
lean_object* v_reuseFailAlloc_953_; 
v_reuseFailAlloc_953_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_953_, 0, v___x_935_);
lean_ctor_set(v_reuseFailAlloc_953_, 1, v___f_931_);
lean_ctor_set(v_reuseFailAlloc_953_, 2, v___f_938_);
lean_ctor_set(v_reuseFailAlloc_953_, 3, v___f_937_);
lean_ctor_set(v_reuseFailAlloc_953_, 4, v___f_936_);
v___x_940_ = v_reuseFailAlloc_953_;
goto v_reusejp_939_;
}
v_reusejp_939_:
{
lean_object* v___x_942_; 
if (v_isShared_923_ == 0)
{
lean_ctor_set(v___x_922_, 1, v___f_932_);
lean_ctor_set(v___x_922_, 0, v___x_940_);
v___x_942_ = v___x_922_;
goto v_reusejp_941_;
}
else
{
lean_object* v_reuseFailAlloc_952_; 
v_reuseFailAlloc_952_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_952_, 0, v___x_940_);
lean_ctor_set(v_reuseFailAlloc_952_, 1, v___f_932_);
v___x_942_ = v_reuseFailAlloc_952_;
goto v_reusejp_941_;
}
v_reusejp_941_:
{
lean_object* v___x_943_; lean_object* v___x_944_; lean_object* v_toMonadRef_945_; lean_object* v___x_946_; lean_object* v___x_947_; lean_object* v___x_948_; lean_object* v___x_949_; lean_object* v___x_287__overap_950_; lean_object* v___x_951_; 
v___x_943_ = lean_obj_once(&lp_aesop_Aesop_TreeM_instInhabited___lam__0___closed__5, &lp_aesop_Aesop_TreeM_instInhabited___lam__0___closed__5_once, _init_lp_aesop_Aesop_TreeM_instInhabited___lam__0___closed__5);
v___x_944_ = lean_obj_once(&lp_aesop_Aesop_TreeM_instInhabited___lam__0___closed__11, &lp_aesop_Aesop_TreeM_instInhabited___lam__0___closed__11_once, _init_lp_aesop_Aesop_TreeM_instInhabited___lam__0___closed__11);
v_toMonadRef_945_ = lean_ctor_get(v___x_944_, 0);
v___x_946_ = l_Lean_Meta_instAddMessageContextMetaM;
lean_inc_ref(v___x_942_);
v___x_947_ = l_Lean_instAddErrorMessageContextOfAddMessageContextOfMonad___redArg(v___x_946_, v___x_942_);
lean_inc_ref(v_toMonadRef_945_);
v___x_948_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_948_, 0, v___x_943_);
lean_ctor_set(v___x_948_, 1, v_toMonadRef_945_);
lean_ctor_set(v___x_948_, 2, v___x_947_);
v___x_949_ = lean_obj_once(&lp_aesop_Aesop_TreeM_instInhabited___lam__0___closed__13, &lp_aesop_Aesop_TreeM_instInhabited___lam__0___closed__13_once, _init_lp_aesop_Aesop_TreeM_instInhabited___lam__0___closed__13);
v___x_287__overap_950_ = l_Lean_throwError___redArg(v___x_942_, v___x_948_, v___x_949_);
lean_inc(v___y_901_);
lean_inc_ref(v___y_900_);
lean_inc(v___y_899_);
lean_inc_ref(v___y_898_);
v___x_951_ = lean_apply_5(v___x_287__overap_950_, v___y_898_, v___y_899_, v___y_900_, v___y_901_, lean_box(0));
return v___x_951_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TreeM_instInhabited___lam__0___boxed(lean_object* v___y_958_, lean_object* v___y_959_, lean_object* v___y_960_, lean_object* v___y_961_, lean_object* v___y_962_, lean_object* v___y_963_, lean_object* v___y_964_, lean_object* v___y_965_){
_start:
{
lean_object* v_res_966_; 
v_res_966_ = lp_aesop_Aesop_TreeM_instInhabited___lam__0(v___y_958_, v___y_959_, v___y_960_, v___y_961_, v___y_962_, v___y_963_, v___y_964_);
lean_dec(v___y_964_);
lean_dec_ref(v___y_963_);
lean_dec(v___y_962_);
lean_dec_ref(v___y_961_);
lean_dec(v___y_960_);
lean_dec(v___y_959_);
lean_dec_ref(v___y_958_);
return v_res_966_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TreeM_instInhabited(lean_object* v_00_u03b1_968_){
_start:
{
lean_object* v___f_969_; 
v___f_969_ = ((lean_object*)(lp_aesop_Aesop_TreeM_instInhabited___closed__0));
return v___f_969_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TreeM_run_x27___redArg(lean_object* v_ctx_970_, lean_object* v_tree_971_, lean_object* v_x_972_, lean_object* v_a_973_, lean_object* v_a_974_, lean_object* v_a_975_, lean_object* v_a_976_, lean_object* v_a_977_){
_start:
{
lean_object* v___x_979_; lean_object* v___x_980_; 
v___x_979_ = lean_st_mk_ref(v_tree_971_);
lean_inc(v_a_977_);
lean_inc_ref(v_a_976_);
lean_inc(v_a_975_);
lean_inc_ref(v_a_974_);
lean_inc(v_a_973_);
lean_inc(v___x_979_);
v___x_980_ = lean_apply_8(v_x_972_, v_ctx_970_, v___x_979_, v_a_973_, v_a_974_, v_a_975_, v_a_976_, v_a_977_, lean_box(0));
if (lean_obj_tag(v___x_980_) == 0)
{
lean_object* v_a_981_; lean_object* v___x_983_; uint8_t v_isShared_984_; uint8_t v_isSharedCheck_990_; 
v_a_981_ = lean_ctor_get(v___x_980_, 0);
v_isSharedCheck_990_ = !lean_is_exclusive(v___x_980_);
if (v_isSharedCheck_990_ == 0)
{
v___x_983_ = v___x_980_;
v_isShared_984_ = v_isSharedCheck_990_;
goto v_resetjp_982_;
}
else
{
lean_inc(v_a_981_);
lean_dec(v___x_980_);
v___x_983_ = lean_box(0);
v_isShared_984_ = v_isSharedCheck_990_;
goto v_resetjp_982_;
}
v_resetjp_982_:
{
lean_object* v___x_985_; lean_object* v___x_986_; lean_object* v___x_988_; 
v___x_985_ = lean_st_ref_get(v___x_979_);
lean_dec(v___x_979_);
v___x_986_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_986_, 0, v_a_981_);
lean_ctor_set(v___x_986_, 1, v___x_985_);
if (v_isShared_984_ == 0)
{
lean_ctor_set(v___x_983_, 0, v___x_986_);
v___x_988_ = v___x_983_;
goto v_reusejp_987_;
}
else
{
lean_object* v_reuseFailAlloc_989_; 
v_reuseFailAlloc_989_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_989_, 0, v___x_986_);
v___x_988_ = v_reuseFailAlloc_989_;
goto v_reusejp_987_;
}
v_reusejp_987_:
{
return v___x_988_;
}
}
}
else
{
lean_object* v_a_991_; lean_object* v___x_993_; uint8_t v_isShared_994_; uint8_t v_isSharedCheck_998_; 
lean_dec(v___x_979_);
v_a_991_ = lean_ctor_get(v___x_980_, 0);
v_isSharedCheck_998_ = !lean_is_exclusive(v___x_980_);
if (v_isSharedCheck_998_ == 0)
{
v___x_993_ = v___x_980_;
v_isShared_994_ = v_isSharedCheck_998_;
goto v_resetjp_992_;
}
else
{
lean_inc(v_a_991_);
lean_dec(v___x_980_);
v___x_993_ = lean_box(0);
v_isShared_994_ = v_isSharedCheck_998_;
goto v_resetjp_992_;
}
v_resetjp_992_:
{
lean_object* v___x_996_; 
if (v_isShared_994_ == 0)
{
v___x_996_ = v___x_993_;
goto v_reusejp_995_;
}
else
{
lean_object* v_reuseFailAlloc_997_; 
v_reuseFailAlloc_997_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_997_, 0, v_a_991_);
v___x_996_ = v_reuseFailAlloc_997_;
goto v_reusejp_995_;
}
v_reusejp_995_:
{
return v___x_996_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TreeM_run_x27___redArg___boxed(lean_object* v_ctx_999_, lean_object* v_tree_1000_, lean_object* v_x_1001_, lean_object* v_a_1002_, lean_object* v_a_1003_, lean_object* v_a_1004_, lean_object* v_a_1005_, lean_object* v_a_1006_, lean_object* v_a_1007_){
_start:
{
lean_object* v_res_1008_; 
v_res_1008_ = lp_aesop_Aesop_TreeM_run_x27___redArg(v_ctx_999_, v_tree_1000_, v_x_1001_, v_a_1002_, v_a_1003_, v_a_1004_, v_a_1005_, v_a_1006_);
lean_dec(v_a_1006_);
lean_dec_ref(v_a_1005_);
lean_dec(v_a_1004_);
lean_dec_ref(v_a_1003_);
lean_dec(v_a_1002_);
return v_res_1008_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TreeM_run_x27(lean_object* v_00_u03b1_1009_, lean_object* v_ctx_1010_, lean_object* v_tree_1011_, lean_object* v_x_1012_, lean_object* v_a_1013_, lean_object* v_a_1014_, lean_object* v_a_1015_, lean_object* v_a_1016_, lean_object* v_a_1017_){
_start:
{
lean_object* v___x_1019_; 
v___x_1019_ = lp_aesop_Aesop_TreeM_run_x27___redArg(v_ctx_1010_, v_tree_1011_, v_x_1012_, v_a_1013_, v_a_1014_, v_a_1015_, v_a_1016_, v_a_1017_);
return v___x_1019_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TreeM_run_x27___boxed(lean_object* v_00_u03b1_1020_, lean_object* v_ctx_1021_, lean_object* v_tree_1022_, lean_object* v_x_1023_, lean_object* v_a_1024_, lean_object* v_a_1025_, lean_object* v_a_1026_, lean_object* v_a_1027_, lean_object* v_a_1028_, lean_object* v_a_1029_){
_start:
{
lean_object* v_res_1030_; 
v_res_1030_ = lp_aesop_Aesop_TreeM_run_x27(v_00_u03b1_1020_, v_ctx_1021_, v_tree_1022_, v_x_1023_, v_a_1024_, v_a_1025_, v_a_1026_, v_a_1027_, v_a_1028_);
lean_dec(v_a_1028_);
lean_dec_ref(v_a_1027_);
lean_dec(v_a_1026_);
lean_dec_ref(v_a_1025_);
lean_dec(v_a_1024_);
return v_res_1030_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getRootMVarCluster___redArg(lean_object* v_a_1031_){
_start:
{
lean_object* v___x_1033_; lean_object* v_root_1034_; lean_object* v___x_1035_; 
v___x_1033_ = lean_st_ref_get(v_a_1031_);
v_root_1034_ = lean_ctor_get(v___x_1033_, 0);
lean_inc(v_root_1034_);
lean_dec(v___x_1033_);
v___x_1035_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1035_, 0, v_root_1034_);
return v___x_1035_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getRootMVarCluster___redArg___boxed(lean_object* v_a_1036_, lean_object* v_a_1037_){
_start:
{
lean_object* v_res_1038_; 
v_res_1038_ = lp_aesop_Aesop_getRootMVarCluster___redArg(v_a_1036_);
lean_dec(v_a_1036_);
return v_res_1038_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getRootMVarCluster(lean_object* v_a_1039_, lean_object* v_a_1040_, lean_object* v_a_1041_, lean_object* v_a_1042_, lean_object* v_a_1043_, lean_object* v_a_1044_, lean_object* v_a_1045_){
_start:
{
lean_object* v___x_1047_; 
v___x_1047_ = lp_aesop_Aesop_getRootMVarCluster___redArg(v_a_1040_);
return v___x_1047_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getRootMVarCluster___boxed(lean_object* v_a_1048_, lean_object* v_a_1049_, lean_object* v_a_1050_, lean_object* v_a_1051_, lean_object* v_a_1052_, lean_object* v_a_1053_, lean_object* v_a_1054_, lean_object* v_a_1055_){
_start:
{
lean_object* v_res_1056_; 
v_res_1056_ = lp_aesop_Aesop_getRootMVarCluster(v_a_1048_, v_a_1049_, v_a_1050_, v_a_1051_, v_a_1052_, v_a_1053_, v_a_1054_);
lean_dec(v_a_1054_);
lean_dec_ref(v_a_1053_);
lean_dec(v_a_1052_);
lean_dec_ref(v_a_1051_);
lean_dec(v_a_1050_);
lean_dec(v_a_1049_);
lean_dec_ref(v_a_1048_);
return v_res_1056_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getRootMetaState___redArg(lean_object* v_a_1057_){
_start:
{
lean_object* v___x_1059_; lean_object* v_rootMetaState_1060_; lean_object* v___x_1061_; 
v___x_1059_ = lean_st_ref_get(v_a_1057_);
v_rootMetaState_1060_ = lean_ctor_get(v___x_1059_, 1);
lean_inc_ref(v_rootMetaState_1060_);
lean_dec(v___x_1059_);
v___x_1061_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1061_, 0, v_rootMetaState_1060_);
return v___x_1061_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getRootMetaState___redArg___boxed(lean_object* v_a_1062_, lean_object* v_a_1063_){
_start:
{
lean_object* v_res_1064_; 
v_res_1064_ = lp_aesop_Aesop_getRootMetaState___redArg(v_a_1062_);
lean_dec(v_a_1062_);
return v_res_1064_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getRootMetaState(lean_object* v_a_1065_, lean_object* v_a_1066_, lean_object* v_a_1067_, lean_object* v_a_1068_, lean_object* v_a_1069_, lean_object* v_a_1070_, lean_object* v_a_1071_){
_start:
{
lean_object* v___x_1073_; 
v___x_1073_ = lp_aesop_Aesop_getRootMetaState___redArg(v_a_1066_);
return v___x_1073_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getRootMetaState___boxed(lean_object* v_a_1074_, lean_object* v_a_1075_, lean_object* v_a_1076_, lean_object* v_a_1077_, lean_object* v_a_1078_, lean_object* v_a_1079_, lean_object* v_a_1080_, lean_object* v_a_1081_){
_start:
{
lean_object* v_res_1082_; 
v_res_1082_ = lp_aesop_Aesop_getRootMetaState(v_a_1074_, v_a_1075_, v_a_1076_, v_a_1077_, v_a_1078_, v_a_1079_, v_a_1080_);
lean_dec(v_a_1080_);
lean_dec_ref(v_a_1079_);
lean_dec(v_a_1078_);
lean_dec_ref(v_a_1077_);
lean_dec(v_a_1076_);
lean_dec(v_a_1075_);
lean_dec_ref(v_a_1074_);
return v_res_1082_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_getRootGoal_spec__0___redArg(lean_object* v_msg_1083_, lean_object* v___y_1084_, lean_object* v___y_1085_, lean_object* v___y_1086_, lean_object* v___y_1087_){
_start:
{
lean_object* v_ref_1089_; lean_object* v___x_1090_; lean_object* v_a_1091_; lean_object* v___x_1093_; uint8_t v_isShared_1094_; uint8_t v_isSharedCheck_1099_; 
v_ref_1089_ = lean_ctor_get(v___y_1086_, 5);
v___x_1090_ = lp_aesop_Lean_addMessageContextFull___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_mkInitialTree_spec__3_spec__5_spec__7(v_msg_1083_, v___y_1084_, v___y_1085_, v___y_1086_, v___y_1087_);
v_a_1091_ = lean_ctor_get(v___x_1090_, 0);
v_isSharedCheck_1099_ = !lean_is_exclusive(v___x_1090_);
if (v_isSharedCheck_1099_ == 0)
{
v___x_1093_ = v___x_1090_;
v_isShared_1094_ = v_isSharedCheck_1099_;
goto v_resetjp_1092_;
}
else
{
lean_inc(v_a_1091_);
lean_dec(v___x_1090_);
v___x_1093_ = lean_box(0);
v_isShared_1094_ = v_isSharedCheck_1099_;
goto v_resetjp_1092_;
}
v_resetjp_1092_:
{
lean_object* v___x_1095_; lean_object* v___x_1097_; 
lean_inc(v_ref_1089_);
v___x_1095_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1095_, 0, v_ref_1089_);
lean_ctor_set(v___x_1095_, 1, v_a_1091_);
if (v_isShared_1094_ == 0)
{
lean_ctor_set_tag(v___x_1093_, 1);
lean_ctor_set(v___x_1093_, 0, v___x_1095_);
v___x_1097_ = v___x_1093_;
goto v_reusejp_1096_;
}
else
{
lean_object* v_reuseFailAlloc_1098_; 
v_reuseFailAlloc_1098_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1098_, 0, v___x_1095_);
v___x_1097_ = v_reuseFailAlloc_1098_;
goto v_reusejp_1096_;
}
v_reusejp_1096_:
{
return v___x_1097_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_getRootGoal_spec__0___redArg___boxed(lean_object* v_msg_1100_, lean_object* v___y_1101_, lean_object* v___y_1102_, lean_object* v___y_1103_, lean_object* v___y_1104_, lean_object* v___y_1105_){
_start:
{
lean_object* v_res_1106_; 
v_res_1106_ = lp_aesop_Lean_throwError___at___00Aesop_getRootGoal_spec__0___redArg(v_msg_1100_, v___y_1101_, v___y_1102_, v___y_1103_, v___y_1104_);
lean_dec(v___y_1104_);
lean_dec_ref(v___y_1103_);
lean_dec(v___y_1102_);
lean_dec_ref(v___y_1101_);
return v_res_1106_;
}
}
static lean_object* _init_lp_aesop_Aesop_getRootGoal___closed__1(void){
_start:
{
lean_object* v___x_1108_; lean_object* v___x_1109_; 
v___x_1108_ = ((lean_object*)(lp_aesop_Aesop_getRootGoal___closed__0));
v___x_1109_ = l_Lean_stringToMessageData(v___x_1108_);
return v___x_1109_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getRootGoal(lean_object* v_a_1110_, lean_object* v_a_1111_, lean_object* v_a_1112_, lean_object* v_a_1113_, lean_object* v_a_1114_, lean_object* v_a_1115_, lean_object* v_a_1116_){
_start:
{
lean_object* v___x_1118_; lean_object* v_a_1119_; lean_object* v___x_1121_; uint8_t v_isShared_1122_; uint8_t v_isSharedCheck_1142_; 
v___x_1118_ = lp_aesop_Aesop_getRootMVarCluster___redArg(v_a_1111_);
v_a_1119_ = lean_ctor_get(v___x_1118_, 0);
v_isSharedCheck_1142_ = !lean_is_exclusive(v___x_1118_);
if (v_isSharedCheck_1142_ == 0)
{
v___x_1121_ = v___x_1118_;
v_isShared_1122_ = v_isSharedCheck_1142_;
goto v_resetjp_1120_;
}
else
{
lean_inc(v_a_1119_);
lean_dec(v___x_1118_);
v___x_1121_ = lean_box(0);
v_isShared_1122_ = v_isSharedCheck_1142_;
goto v_resetjp_1120_;
}
v_resetjp_1120_:
{
lean_object* v___x_1123_; lean_object* v___x_1124_; lean_object* v_elimMVarCluster_1125_; lean_object* v___x_1126_; lean_object* v_goals_1127_; lean_object* v___x_1128_; lean_object* v___x_1129_; uint8_t v___x_1130_; 
v___x_1123_ = lean_st_ref_get(v_a_1119_);
lean_dec(v_a_1119_);
v___x_1124_ = lp_aesop_Aesop_treeImpl;
v_elimMVarCluster_1125_ = lean_ctor_get(v___x_1124_, 5);
lean_inc_ref(v_elimMVarCluster_1125_);
v___x_1126_ = lean_apply_1(v_elimMVarCluster_1125_, v___x_1123_);
v_goals_1127_ = lean_ctor_get(v___x_1126_, 1);
lean_inc_ref(v_goals_1127_);
lean_dec_ref(v___x_1126_);
v___x_1128_ = lean_array_get_size(v_goals_1127_);
v___x_1129_ = lean_unsigned_to_nat(1u);
v___x_1130_ = lean_nat_dec_eq(v___x_1128_, v___x_1129_);
if (v___x_1130_ == 0)
{
lean_object* v___x_1131_; lean_object* v___x_1132_; lean_object* v___x_1133_; lean_object* v___x_1134_; lean_object* v___x_1135_; lean_object* v___x_1136_; 
lean_dec_ref(v_goals_1127_);
lean_del_object(v___x_1121_);
v___x_1131_ = lean_obj_once(&lp_aesop_Aesop_getRootGoal___closed__1, &lp_aesop_Aesop_getRootGoal___closed__1_once, _init_lp_aesop_Aesop_getRootGoal___closed__1);
v___x_1132_ = l_Nat_reprFast(v___x_1128_);
v___x_1133_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1133_, 0, v___x_1132_);
v___x_1134_ = l_Lean_MessageData_ofFormat(v___x_1133_);
v___x_1135_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1135_, 0, v___x_1131_);
lean_ctor_set(v___x_1135_, 1, v___x_1134_);
v___x_1136_ = lp_aesop_Lean_throwError___at___00Aesop_getRootGoal_spec__0___redArg(v___x_1135_, v_a_1113_, v_a_1114_, v_a_1115_, v_a_1116_);
return v___x_1136_;
}
else
{
lean_object* v___x_1137_; lean_object* v___x_1138_; lean_object* v___x_1140_; 
v___x_1137_ = lean_unsigned_to_nat(0u);
v___x_1138_ = lean_array_fget(v_goals_1127_, v___x_1137_);
lean_dec_ref(v_goals_1127_);
if (v_isShared_1122_ == 0)
{
lean_ctor_set(v___x_1121_, 0, v___x_1138_);
v___x_1140_ = v___x_1121_;
goto v_reusejp_1139_;
}
else
{
lean_object* v_reuseFailAlloc_1141_; 
v_reuseFailAlloc_1141_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1141_, 0, v___x_1138_);
v___x_1140_ = v_reuseFailAlloc_1141_;
goto v_reusejp_1139_;
}
v_reusejp_1139_:
{
return v___x_1140_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getRootGoal___boxed(lean_object* v_a_1143_, lean_object* v_a_1144_, lean_object* v_a_1145_, lean_object* v_a_1146_, lean_object* v_a_1147_, lean_object* v_a_1148_, lean_object* v_a_1149_, lean_object* v_a_1150_){
_start:
{
lean_object* v_res_1151_; 
v_res_1151_ = lp_aesop_Aesop_getRootGoal(v_a_1143_, v_a_1144_, v_a_1145_, v_a_1146_, v_a_1147_, v_a_1148_, v_a_1149_);
lean_dec(v_a_1149_);
lean_dec_ref(v_a_1148_);
lean_dec(v_a_1147_);
lean_dec_ref(v_a_1146_);
lean_dec(v_a_1145_);
lean_dec(v_a_1144_);
lean_dec_ref(v_a_1143_);
return v_res_1151_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_getRootGoal_spec__0(lean_object* v_00_u03b1_1152_, lean_object* v_msg_1153_, lean_object* v___y_1154_, lean_object* v___y_1155_, lean_object* v___y_1156_, lean_object* v___y_1157_, lean_object* v___y_1158_, lean_object* v___y_1159_, lean_object* v___y_1160_){
_start:
{
lean_object* v___x_1162_; 
v___x_1162_ = lp_aesop_Lean_throwError___at___00Aesop_getRootGoal_spec__0___redArg(v_msg_1153_, v___y_1157_, v___y_1158_, v___y_1159_, v___y_1160_);
return v___x_1162_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_getRootGoal_spec__0___boxed(lean_object* v_00_u03b1_1163_, lean_object* v_msg_1164_, lean_object* v___y_1165_, lean_object* v___y_1166_, lean_object* v___y_1167_, lean_object* v___y_1168_, lean_object* v___y_1169_, lean_object* v___y_1170_, lean_object* v___y_1171_, lean_object* v___y_1172_){
_start:
{
lean_object* v_res_1173_; 
v_res_1173_ = lp_aesop_Lean_throwError___at___00Aesop_getRootGoal_spec__0(v_00_u03b1_1163_, v_msg_1164_, v___y_1165_, v___y_1166_, v___y_1167_, v___y_1168_, v___y_1169_, v___y_1170_, v___y_1171_);
lean_dec(v___y_1171_);
lean_dec_ref(v___y_1170_);
lean_dec(v___y_1169_);
lean_dec_ref(v___y_1168_);
lean_dec(v___y_1167_);
lean_dec(v___y_1166_);
lean_dec_ref(v___y_1165_);
return v_res_1173_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getRootMVarId(lean_object* v_a_1174_, lean_object* v_a_1175_, lean_object* v_a_1176_, lean_object* v_a_1177_, lean_object* v_a_1178_, lean_object* v_a_1179_, lean_object* v_a_1180_){
_start:
{
lean_object* v___x_1182_; 
v___x_1182_ = lp_aesop_Aesop_getRootGoal(v_a_1174_, v_a_1175_, v_a_1176_, v_a_1177_, v_a_1178_, v_a_1179_, v_a_1180_);
if (lean_obj_tag(v___x_1182_) == 0)
{
lean_object* v_a_1183_; lean_object* v___x_1185_; uint8_t v_isShared_1186_; uint8_t v_isSharedCheck_1195_; 
v_a_1183_ = lean_ctor_get(v___x_1182_, 0);
v_isSharedCheck_1195_ = !lean_is_exclusive(v___x_1182_);
if (v_isSharedCheck_1195_ == 0)
{
v___x_1185_ = v___x_1182_;
v_isShared_1186_ = v_isSharedCheck_1195_;
goto v_resetjp_1184_;
}
else
{
lean_inc(v_a_1183_);
lean_dec(v___x_1182_);
v___x_1185_ = lean_box(0);
v_isShared_1186_ = v_isSharedCheck_1195_;
goto v_resetjp_1184_;
}
v_resetjp_1184_:
{
lean_object* v___x_1187_; lean_object* v___x_1188_; lean_object* v_elimGoal_1189_; lean_object* v___x_1190_; lean_object* v_preNormGoal_1191_; lean_object* v___x_1193_; 
v___x_1187_ = lean_st_ref_get(v_a_1183_);
lean_dec(v_a_1183_);
v___x_1188_ = lp_aesop_Aesop_treeImpl;
v_elimGoal_1189_ = lean_ctor_get(v___x_1188_, 1);
lean_inc_ref(v_elimGoal_1189_);
v___x_1190_ = lean_apply_1(v_elimGoal_1189_, v___x_1187_);
v_preNormGoal_1191_ = lean_ctor_get(v___x_1190_, 5);
lean_inc(v_preNormGoal_1191_);
lean_dec_ref(v___x_1190_);
if (v_isShared_1186_ == 0)
{
lean_ctor_set(v___x_1185_, 0, v_preNormGoal_1191_);
v___x_1193_ = v___x_1185_;
goto v_reusejp_1192_;
}
else
{
lean_object* v_reuseFailAlloc_1194_; 
v_reuseFailAlloc_1194_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1194_, 0, v_preNormGoal_1191_);
v___x_1193_ = v_reuseFailAlloc_1194_;
goto v_reusejp_1192_;
}
v_reusejp_1192_:
{
return v___x_1193_;
}
}
}
else
{
lean_object* v_a_1196_; lean_object* v___x_1198_; uint8_t v_isShared_1199_; uint8_t v_isSharedCheck_1203_; 
v_a_1196_ = lean_ctor_get(v___x_1182_, 0);
v_isSharedCheck_1203_ = !lean_is_exclusive(v___x_1182_);
if (v_isSharedCheck_1203_ == 0)
{
v___x_1198_ = v___x_1182_;
v_isShared_1199_ = v_isSharedCheck_1203_;
goto v_resetjp_1197_;
}
else
{
lean_inc(v_a_1196_);
lean_dec(v___x_1182_);
v___x_1198_ = lean_box(0);
v_isShared_1199_ = v_isSharedCheck_1203_;
goto v_resetjp_1197_;
}
v_resetjp_1197_:
{
lean_object* v___x_1201_; 
if (v_isShared_1199_ == 0)
{
v___x_1201_ = v___x_1198_;
goto v_reusejp_1200_;
}
else
{
lean_object* v_reuseFailAlloc_1202_; 
v_reuseFailAlloc_1202_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1202_, 0, v_a_1196_);
v___x_1201_ = v_reuseFailAlloc_1202_;
goto v_reusejp_1200_;
}
v_reusejp_1200_:
{
return v___x_1201_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getRootMVarId___boxed(lean_object* v_a_1204_, lean_object* v_a_1205_, lean_object* v_a_1206_, lean_object* v_a_1207_, lean_object* v_a_1208_, lean_object* v_a_1209_, lean_object* v_a_1210_, lean_object* v_a_1211_){
_start:
{
lean_object* v_res_1212_; 
v_res_1212_ = lp_aesop_Aesop_getRootMVarId(v_a_1204_, v_a_1205_, v_a_1206_, v_a_1207_, v_a_1208_, v_a_1209_, v_a_1210_);
lean_dec(v_a_1210_);
lean_dec_ref(v_a_1209_);
lean_dec(v_a_1208_);
lean_dec_ref(v_a_1207_);
lean_dec(v_a_1206_);
lean_dec(v_a_1205_);
lean_dec_ref(v_a_1204_);
return v_res_1212_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_incrementNumGoals___redArg(lean_object* v_increment_1213_, lean_object* v_a_1214_){
_start:
{
lean_object* v___x_1216_; lean_object* v_root_1217_; lean_object* v_rootMetaState_1218_; lean_object* v_numGoals_1219_; lean_object* v_numRapps_1220_; lean_object* v_nextGoalId_1221_; lean_object* v_nextRappId_1222_; lean_object* v_allIntroducedMVars_1223_; lean_object* v___x_1225_; uint8_t v_isShared_1226_; uint8_t v_isSharedCheck_1234_; 
v___x_1216_ = lean_st_ref_take(v_a_1214_);
v_root_1217_ = lean_ctor_get(v___x_1216_, 0);
v_rootMetaState_1218_ = lean_ctor_get(v___x_1216_, 1);
v_numGoals_1219_ = lean_ctor_get(v___x_1216_, 2);
v_numRapps_1220_ = lean_ctor_get(v___x_1216_, 3);
v_nextGoalId_1221_ = lean_ctor_get(v___x_1216_, 4);
v_nextRappId_1222_ = lean_ctor_get(v___x_1216_, 5);
v_allIntroducedMVars_1223_ = lean_ctor_get(v___x_1216_, 6);
v_isSharedCheck_1234_ = !lean_is_exclusive(v___x_1216_);
if (v_isSharedCheck_1234_ == 0)
{
v___x_1225_ = v___x_1216_;
v_isShared_1226_ = v_isSharedCheck_1234_;
goto v_resetjp_1224_;
}
else
{
lean_inc(v_allIntroducedMVars_1223_);
lean_inc(v_nextRappId_1222_);
lean_inc(v_nextGoalId_1221_);
lean_inc(v_numRapps_1220_);
lean_inc(v_numGoals_1219_);
lean_inc(v_rootMetaState_1218_);
lean_inc(v_root_1217_);
lean_dec(v___x_1216_);
v___x_1225_ = lean_box(0);
v_isShared_1226_ = v_isSharedCheck_1234_;
goto v_resetjp_1224_;
}
v_resetjp_1224_:
{
lean_object* v___x_1227_; lean_object* v___x_1229_; 
v___x_1227_ = lean_nat_add(v_numGoals_1219_, v_increment_1213_);
lean_dec(v_numGoals_1219_);
if (v_isShared_1226_ == 0)
{
lean_ctor_set(v___x_1225_, 2, v___x_1227_);
v___x_1229_ = v___x_1225_;
goto v_reusejp_1228_;
}
else
{
lean_object* v_reuseFailAlloc_1233_; 
v_reuseFailAlloc_1233_ = lean_alloc_ctor(0, 7, 0);
lean_ctor_set(v_reuseFailAlloc_1233_, 0, v_root_1217_);
lean_ctor_set(v_reuseFailAlloc_1233_, 1, v_rootMetaState_1218_);
lean_ctor_set(v_reuseFailAlloc_1233_, 2, v___x_1227_);
lean_ctor_set(v_reuseFailAlloc_1233_, 3, v_numRapps_1220_);
lean_ctor_set(v_reuseFailAlloc_1233_, 4, v_nextGoalId_1221_);
lean_ctor_set(v_reuseFailAlloc_1233_, 5, v_nextRappId_1222_);
lean_ctor_set(v_reuseFailAlloc_1233_, 6, v_allIntroducedMVars_1223_);
v___x_1229_ = v_reuseFailAlloc_1233_;
goto v_reusejp_1228_;
}
v_reusejp_1228_:
{
lean_object* v___x_1230_; lean_object* v___x_1231_; lean_object* v___x_1232_; 
v___x_1230_ = lean_st_ref_set(v_a_1214_, v___x_1229_);
v___x_1231_ = lean_box(0);
v___x_1232_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1232_, 0, v___x_1231_);
return v___x_1232_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_incrementNumGoals___redArg___boxed(lean_object* v_increment_1235_, lean_object* v_a_1236_, lean_object* v_a_1237_){
_start:
{
lean_object* v_res_1238_; 
v_res_1238_ = lp_aesop_Aesop_incrementNumGoals___redArg(v_increment_1235_, v_a_1236_);
lean_dec(v_a_1236_);
lean_dec(v_increment_1235_);
return v_res_1238_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_incrementNumGoals(lean_object* v_increment_1239_, lean_object* v_a_1240_, lean_object* v_a_1241_, lean_object* v_a_1242_, lean_object* v_a_1243_, lean_object* v_a_1244_, lean_object* v_a_1245_, lean_object* v_a_1246_){
_start:
{
lean_object* v___x_1248_; 
v___x_1248_ = lp_aesop_Aesop_incrementNumGoals___redArg(v_increment_1239_, v_a_1241_);
return v___x_1248_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_incrementNumGoals___boxed(lean_object* v_increment_1249_, lean_object* v_a_1250_, lean_object* v_a_1251_, lean_object* v_a_1252_, lean_object* v_a_1253_, lean_object* v_a_1254_, lean_object* v_a_1255_, lean_object* v_a_1256_, lean_object* v_a_1257_){
_start:
{
lean_object* v_res_1258_; 
v_res_1258_ = lp_aesop_Aesop_incrementNumGoals(v_increment_1249_, v_a_1250_, v_a_1251_, v_a_1252_, v_a_1253_, v_a_1254_, v_a_1255_, v_a_1256_);
lean_dec(v_a_1256_);
lean_dec_ref(v_a_1255_);
lean_dec(v_a_1254_);
lean_dec_ref(v_a_1253_);
lean_dec(v_a_1252_);
lean_dec(v_a_1251_);
lean_dec_ref(v_a_1250_);
lean_dec(v_increment_1249_);
return v_res_1258_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_incrementNumRapps___redArg(lean_object* v_increment_1259_, lean_object* v_a_1260_){
_start:
{
lean_object* v___x_1262_; lean_object* v_root_1263_; lean_object* v_rootMetaState_1264_; lean_object* v_numGoals_1265_; lean_object* v_numRapps_1266_; lean_object* v_nextGoalId_1267_; lean_object* v_nextRappId_1268_; lean_object* v_allIntroducedMVars_1269_; lean_object* v___x_1271_; uint8_t v_isShared_1272_; uint8_t v_isSharedCheck_1280_; 
v___x_1262_ = lean_st_ref_take(v_a_1260_);
v_root_1263_ = lean_ctor_get(v___x_1262_, 0);
v_rootMetaState_1264_ = lean_ctor_get(v___x_1262_, 1);
v_numGoals_1265_ = lean_ctor_get(v___x_1262_, 2);
v_numRapps_1266_ = lean_ctor_get(v___x_1262_, 3);
v_nextGoalId_1267_ = lean_ctor_get(v___x_1262_, 4);
v_nextRappId_1268_ = lean_ctor_get(v___x_1262_, 5);
v_allIntroducedMVars_1269_ = lean_ctor_get(v___x_1262_, 6);
v_isSharedCheck_1280_ = !lean_is_exclusive(v___x_1262_);
if (v_isSharedCheck_1280_ == 0)
{
v___x_1271_ = v___x_1262_;
v_isShared_1272_ = v_isSharedCheck_1280_;
goto v_resetjp_1270_;
}
else
{
lean_inc(v_allIntroducedMVars_1269_);
lean_inc(v_nextRappId_1268_);
lean_inc(v_nextGoalId_1267_);
lean_inc(v_numRapps_1266_);
lean_inc(v_numGoals_1265_);
lean_inc(v_rootMetaState_1264_);
lean_inc(v_root_1263_);
lean_dec(v___x_1262_);
v___x_1271_ = lean_box(0);
v_isShared_1272_ = v_isSharedCheck_1280_;
goto v_resetjp_1270_;
}
v_resetjp_1270_:
{
lean_object* v___x_1273_; lean_object* v___x_1275_; 
v___x_1273_ = lean_nat_add(v_numRapps_1266_, v_increment_1259_);
lean_dec(v_numRapps_1266_);
if (v_isShared_1272_ == 0)
{
lean_ctor_set(v___x_1271_, 3, v___x_1273_);
v___x_1275_ = v___x_1271_;
goto v_reusejp_1274_;
}
else
{
lean_object* v_reuseFailAlloc_1279_; 
v_reuseFailAlloc_1279_ = lean_alloc_ctor(0, 7, 0);
lean_ctor_set(v_reuseFailAlloc_1279_, 0, v_root_1263_);
lean_ctor_set(v_reuseFailAlloc_1279_, 1, v_rootMetaState_1264_);
lean_ctor_set(v_reuseFailAlloc_1279_, 2, v_numGoals_1265_);
lean_ctor_set(v_reuseFailAlloc_1279_, 3, v___x_1273_);
lean_ctor_set(v_reuseFailAlloc_1279_, 4, v_nextGoalId_1267_);
lean_ctor_set(v_reuseFailAlloc_1279_, 5, v_nextRappId_1268_);
lean_ctor_set(v_reuseFailAlloc_1279_, 6, v_allIntroducedMVars_1269_);
v___x_1275_ = v_reuseFailAlloc_1279_;
goto v_reusejp_1274_;
}
v_reusejp_1274_:
{
lean_object* v___x_1276_; lean_object* v___x_1277_; lean_object* v___x_1278_; 
v___x_1276_ = lean_st_ref_set(v_a_1260_, v___x_1275_);
v___x_1277_ = lean_box(0);
v___x_1278_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1278_, 0, v___x_1277_);
return v___x_1278_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_incrementNumRapps___redArg___boxed(lean_object* v_increment_1281_, lean_object* v_a_1282_, lean_object* v_a_1283_){
_start:
{
lean_object* v_res_1284_; 
v_res_1284_ = lp_aesop_Aesop_incrementNumRapps___redArg(v_increment_1281_, v_a_1282_);
lean_dec(v_a_1282_);
lean_dec(v_increment_1281_);
return v_res_1284_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_incrementNumRapps(lean_object* v_increment_1285_, lean_object* v_a_1286_, lean_object* v_a_1287_, lean_object* v_a_1288_, lean_object* v_a_1289_, lean_object* v_a_1290_, lean_object* v_a_1291_, lean_object* v_a_1292_){
_start:
{
lean_object* v___x_1294_; 
v___x_1294_ = lp_aesop_Aesop_incrementNumRapps___redArg(v_increment_1285_, v_a_1287_);
return v___x_1294_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_incrementNumRapps___boxed(lean_object* v_increment_1295_, lean_object* v_a_1296_, lean_object* v_a_1297_, lean_object* v_a_1298_, lean_object* v_a_1299_, lean_object* v_a_1300_, lean_object* v_a_1301_, lean_object* v_a_1302_, lean_object* v_a_1303_){
_start:
{
lean_object* v_res_1304_; 
v_res_1304_ = lp_aesop_Aesop_incrementNumRapps(v_increment_1295_, v_a_1296_, v_a_1297_, v_a_1298_, v_a_1299_, v_a_1300_, v_a_1301_, v_a_1302_);
lean_dec(v_a_1302_);
lean_dec_ref(v_a_1301_);
lean_dec(v_a_1300_);
lean_dec_ref(v_a_1299_);
lean_dec(v_a_1298_);
lean_dec(v_a_1297_);
lean_dec_ref(v_a_1296_);
lean_dec(v_increment_1295_);
return v_res_1304_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getAllIntroducedMVars___redArg(lean_object* v_a_1305_){
_start:
{
lean_object* v___x_1307_; lean_object* v_allIntroducedMVars_1308_; lean_object* v___x_1309_; 
v___x_1307_ = lean_st_ref_get(v_a_1305_);
v_allIntroducedMVars_1308_ = lean_ctor_get(v___x_1307_, 6);
lean_inc_ref(v_allIntroducedMVars_1308_);
lean_dec(v___x_1307_);
v___x_1309_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1309_, 0, v_allIntroducedMVars_1308_);
return v___x_1309_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getAllIntroducedMVars___redArg___boxed(lean_object* v_a_1310_, lean_object* v_a_1311_){
_start:
{
lean_object* v_res_1312_; 
v_res_1312_ = lp_aesop_Aesop_getAllIntroducedMVars___redArg(v_a_1310_);
lean_dec(v_a_1310_);
return v_res_1312_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getAllIntroducedMVars(lean_object* v_a_1313_, lean_object* v_a_1314_, lean_object* v_a_1315_, lean_object* v_a_1316_, lean_object* v_a_1317_, lean_object* v_a_1318_, lean_object* v_a_1319_){
_start:
{
lean_object* v___x_1321_; 
v___x_1321_ = lp_aesop_Aesop_getAllIntroducedMVars___redArg(v_a_1314_);
return v___x_1321_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getAllIntroducedMVars___boxed(lean_object* v_a_1322_, lean_object* v_a_1323_, lean_object* v_a_1324_, lean_object* v_a_1325_, lean_object* v_a_1326_, lean_object* v_a_1327_, lean_object* v_a_1328_, lean_object* v_a_1329_){
_start:
{
lean_object* v_res_1330_; 
v_res_1330_ = lp_aesop_Aesop_getAllIntroducedMVars(v_a_1322_, v_a_1323_, v_a_1324_, v_a_1325_, v_a_1326_, v_a_1327_, v_a_1328_);
lean_dec(v_a_1328_);
lean_dec_ref(v_a_1327_);
lean_dec(v_a_1326_);
lean_dec_ref(v_a_1325_);
lean_dec(v_a_1324_);
lean_dec(v_a_1323_);
lean_dec_ref(v_a_1322_);
return v_res_1330_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getAndIncrementNextGoalId___redArg(lean_object* v_a_1331_){
_start:
{
lean_object* v___x_1333_; lean_object* v_root_1334_; lean_object* v_rootMetaState_1335_; lean_object* v_numGoals_1336_; lean_object* v_numRapps_1337_; lean_object* v_nextGoalId_1338_; lean_object* v_nextRappId_1339_; lean_object* v_allIntroducedMVars_1340_; lean_object* v___x_1342_; uint8_t v_isShared_1343_; uint8_t v_isSharedCheck_1350_; 
v___x_1333_ = lean_st_ref_take(v_a_1331_);
v_root_1334_ = lean_ctor_get(v___x_1333_, 0);
v_rootMetaState_1335_ = lean_ctor_get(v___x_1333_, 1);
v_numGoals_1336_ = lean_ctor_get(v___x_1333_, 2);
v_numRapps_1337_ = lean_ctor_get(v___x_1333_, 3);
v_nextGoalId_1338_ = lean_ctor_get(v___x_1333_, 4);
v_nextRappId_1339_ = lean_ctor_get(v___x_1333_, 5);
v_allIntroducedMVars_1340_ = lean_ctor_get(v___x_1333_, 6);
v_isSharedCheck_1350_ = !lean_is_exclusive(v___x_1333_);
if (v_isSharedCheck_1350_ == 0)
{
v___x_1342_ = v___x_1333_;
v_isShared_1343_ = v_isSharedCheck_1350_;
goto v_resetjp_1341_;
}
else
{
lean_inc(v_allIntroducedMVars_1340_);
lean_inc(v_nextRappId_1339_);
lean_inc(v_nextGoalId_1338_);
lean_inc(v_numRapps_1337_);
lean_inc(v_numGoals_1336_);
lean_inc(v_rootMetaState_1335_);
lean_inc(v_root_1334_);
lean_dec(v___x_1333_);
v___x_1342_ = lean_box(0);
v_isShared_1343_ = v_isSharedCheck_1350_;
goto v_resetjp_1341_;
}
v_resetjp_1341_:
{
lean_object* v___x_1344_; lean_object* v___x_1346_; 
v___x_1344_ = lp_aesop_Aesop_GoalId_succ(v_nextGoalId_1338_);
if (v_isShared_1343_ == 0)
{
lean_ctor_set(v___x_1342_, 4, v___x_1344_);
v___x_1346_ = v___x_1342_;
goto v_reusejp_1345_;
}
else
{
lean_object* v_reuseFailAlloc_1349_; 
v_reuseFailAlloc_1349_ = lean_alloc_ctor(0, 7, 0);
lean_ctor_set(v_reuseFailAlloc_1349_, 0, v_root_1334_);
lean_ctor_set(v_reuseFailAlloc_1349_, 1, v_rootMetaState_1335_);
lean_ctor_set(v_reuseFailAlloc_1349_, 2, v_numGoals_1336_);
lean_ctor_set(v_reuseFailAlloc_1349_, 3, v_numRapps_1337_);
lean_ctor_set(v_reuseFailAlloc_1349_, 4, v___x_1344_);
lean_ctor_set(v_reuseFailAlloc_1349_, 5, v_nextRappId_1339_);
lean_ctor_set(v_reuseFailAlloc_1349_, 6, v_allIntroducedMVars_1340_);
v___x_1346_ = v_reuseFailAlloc_1349_;
goto v_reusejp_1345_;
}
v_reusejp_1345_:
{
lean_object* v___x_1347_; lean_object* v___x_1348_; 
v___x_1347_ = lean_st_ref_set(v_a_1331_, v___x_1346_);
v___x_1348_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1348_, 0, v_nextGoalId_1338_);
return v___x_1348_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getAndIncrementNextGoalId___redArg___boxed(lean_object* v_a_1351_, lean_object* v_a_1352_){
_start:
{
lean_object* v_res_1353_; 
v_res_1353_ = lp_aesop_Aesop_getAndIncrementNextGoalId___redArg(v_a_1351_);
lean_dec(v_a_1351_);
return v_res_1353_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getAndIncrementNextGoalId(lean_object* v_a_1354_, lean_object* v_a_1355_, lean_object* v_a_1356_, lean_object* v_a_1357_, lean_object* v_a_1358_, lean_object* v_a_1359_, lean_object* v_a_1360_){
_start:
{
lean_object* v___x_1362_; 
v___x_1362_ = lp_aesop_Aesop_getAndIncrementNextGoalId___redArg(v_a_1355_);
return v___x_1362_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getAndIncrementNextGoalId___boxed(lean_object* v_a_1363_, lean_object* v_a_1364_, lean_object* v_a_1365_, lean_object* v_a_1366_, lean_object* v_a_1367_, lean_object* v_a_1368_, lean_object* v_a_1369_, lean_object* v_a_1370_){
_start:
{
lean_object* v_res_1371_; 
v_res_1371_ = lp_aesop_Aesop_getAndIncrementNextGoalId(v_a_1363_, v_a_1364_, v_a_1365_, v_a_1366_, v_a_1367_, v_a_1368_, v_a_1369_);
lean_dec(v_a_1369_);
lean_dec_ref(v_a_1368_);
lean_dec(v_a_1367_);
lean_dec_ref(v_a_1366_);
lean_dec(v_a_1365_);
lean_dec(v_a_1364_);
lean_dec_ref(v_a_1363_);
return v_res_1371_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getAndIncrementNextRappId___redArg(lean_object* v_a_1372_){
_start:
{
lean_object* v___x_1374_; lean_object* v_root_1375_; lean_object* v_rootMetaState_1376_; lean_object* v_numGoals_1377_; lean_object* v_numRapps_1378_; lean_object* v_nextGoalId_1379_; lean_object* v_nextRappId_1380_; lean_object* v_allIntroducedMVars_1381_; lean_object* v___x_1383_; uint8_t v_isShared_1384_; uint8_t v_isSharedCheck_1391_; 
v___x_1374_ = lean_st_ref_take(v_a_1372_);
v_root_1375_ = lean_ctor_get(v___x_1374_, 0);
v_rootMetaState_1376_ = lean_ctor_get(v___x_1374_, 1);
v_numGoals_1377_ = lean_ctor_get(v___x_1374_, 2);
v_numRapps_1378_ = lean_ctor_get(v___x_1374_, 3);
v_nextGoalId_1379_ = lean_ctor_get(v___x_1374_, 4);
v_nextRappId_1380_ = lean_ctor_get(v___x_1374_, 5);
v_allIntroducedMVars_1381_ = lean_ctor_get(v___x_1374_, 6);
v_isSharedCheck_1391_ = !lean_is_exclusive(v___x_1374_);
if (v_isSharedCheck_1391_ == 0)
{
v___x_1383_ = v___x_1374_;
v_isShared_1384_ = v_isSharedCheck_1391_;
goto v_resetjp_1382_;
}
else
{
lean_inc(v_allIntroducedMVars_1381_);
lean_inc(v_nextRappId_1380_);
lean_inc(v_nextGoalId_1379_);
lean_inc(v_numRapps_1378_);
lean_inc(v_numGoals_1377_);
lean_inc(v_rootMetaState_1376_);
lean_inc(v_root_1375_);
lean_dec(v___x_1374_);
v___x_1383_ = lean_box(0);
v_isShared_1384_ = v_isSharedCheck_1391_;
goto v_resetjp_1382_;
}
v_resetjp_1382_:
{
lean_object* v___x_1385_; lean_object* v___x_1387_; 
v___x_1385_ = lp_aesop_Aesop_RappId_succ(v_nextRappId_1380_);
if (v_isShared_1384_ == 0)
{
lean_ctor_set(v___x_1383_, 5, v___x_1385_);
v___x_1387_ = v___x_1383_;
goto v_reusejp_1386_;
}
else
{
lean_object* v_reuseFailAlloc_1390_; 
v_reuseFailAlloc_1390_ = lean_alloc_ctor(0, 7, 0);
lean_ctor_set(v_reuseFailAlloc_1390_, 0, v_root_1375_);
lean_ctor_set(v_reuseFailAlloc_1390_, 1, v_rootMetaState_1376_);
lean_ctor_set(v_reuseFailAlloc_1390_, 2, v_numGoals_1377_);
lean_ctor_set(v_reuseFailAlloc_1390_, 3, v_numRapps_1378_);
lean_ctor_set(v_reuseFailAlloc_1390_, 4, v_nextGoalId_1379_);
lean_ctor_set(v_reuseFailAlloc_1390_, 5, v___x_1385_);
lean_ctor_set(v_reuseFailAlloc_1390_, 6, v_allIntroducedMVars_1381_);
v___x_1387_ = v_reuseFailAlloc_1390_;
goto v_reusejp_1386_;
}
v_reusejp_1386_:
{
lean_object* v___x_1388_; lean_object* v___x_1389_; 
v___x_1388_ = lean_st_ref_set(v_a_1372_, v___x_1387_);
v___x_1389_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1389_, 0, v_nextRappId_1380_);
return v___x_1389_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getAndIncrementNextRappId___redArg___boxed(lean_object* v_a_1392_, lean_object* v_a_1393_){
_start:
{
lean_object* v_res_1394_; 
v_res_1394_ = lp_aesop_Aesop_getAndIncrementNextRappId___redArg(v_a_1392_);
lean_dec(v_a_1392_);
return v_res_1394_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getAndIncrementNextRappId(lean_object* v_a_1395_, lean_object* v_a_1396_, lean_object* v_a_1397_, lean_object* v_a_1398_, lean_object* v_a_1399_, lean_object* v_a_1400_, lean_object* v_a_1401_){
_start:
{
lean_object* v___x_1403_; 
v___x_1403_ = lp_aesop_Aesop_getAndIncrementNextRappId___redArg(v_a_1396_);
return v___x_1403_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getAndIncrementNextRappId___boxed(lean_object* v_a_1404_, lean_object* v_a_1405_, lean_object* v_a_1406_, lean_object* v_a_1407_, lean_object* v_a_1408_, lean_object* v_a_1409_, lean_object* v_a_1410_, lean_object* v_a_1411_){
_start:
{
lean_object* v_res_1412_; 
v_res_1412_ = lp_aesop_Aesop_getAndIncrementNextRappId(v_a_1404_, v_a_1405_, v_a_1406_, v_a_1407_, v_a_1408_, v_a_1409_, v_a_1410_);
lean_dec(v_a_1410_);
lean_dec_ref(v_a_1409_);
lean_dec(v_a_1408_);
lean_dec_ref(v_a_1407_);
lean_dec(v_a_1406_);
lean_dec(v_a_1405_);
lean_dec_ref(v_a_1404_);
return v_res_1412_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_RuleSet(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Tree_Data(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Forward_State_Initial(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_Tree_TreeM(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_RuleSet(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Tree_Data(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Forward_State_Initial(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_aesop_Aesop_TreeM_instMonad = _init_lp_aesop_Aesop_TreeM_instMonad();
lean_mark_persistent(lp_aesop_Aesop_TreeM_instMonad);
lp_aesop_Aesop_TreeM_instMonadStateOfTree = _init_lp_aesop_Aesop_TreeM_instMonadStateOfTree();
lean_mark_persistent(lp_aesop_Aesop_TreeM_instMonadStateOfTree);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_Tree_TreeM(uint8_t builtin) {
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
lean_object* initialize_aesop_Aesop_RuleSet(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Tree_Data(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Forward_State_Initial(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_Tree_TreeM(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_RuleSet(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Tree_Data(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Forward_State_Initial(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Tree_TreeM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_Tree_TreeM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_Tree_TreeM(builtin);
}
#ifdef __cplusplus
}
#endif
